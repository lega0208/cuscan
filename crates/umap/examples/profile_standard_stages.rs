//! Per-stage Standard UMAP fit timing companion to comparison/compare_stages.py.
//!
//! usage: profile_standard_stages INPUT_RAW ROWS FEATURES EPOCHS
//!        [--backend cpu|cuda] [--search exact|approximate] [--runs R]
//!        [--neighbors K] [--init random|spectral|recursive|auto] [--seed S] [--min-dist D]
//!        [--trace PATH] [--components N] [--embedding-out PATH]
//!        [--optimizer compatibility|adam|momentum]
//!
//! `--components` sets `n_components` (default 2). `--embedding-out PATH` writes
//! each run's fitted embedding as raw little-endian f32 row-major bytes to
//! `PATH.<run>`.
//!
//! Runs R fits in this process without warmup, so run 0 is process-cold
//! (includes kernel JIT/compilation) and later runs are warm. Stage spans come
//! from the library `tracing` spans, which insert device barriers. Prints one
//! JSON line with per-run `fit_seconds` and `stage_seconds`.
//! Stderr run markers use Unix nanoseconds. Calibrate realtime against perf's
//! monotonic clock before and after recording to select warm sample windows;
//! summary durations are not timestamps.
//! Modern order spans measure host wall time: shuffle includes copying the previous
//! order, while upload completion includes CPU mapped copies or CUDA submission,
//! synchronization and pinned-buffer return, not pure DMA. Source and reference
//! spans run in the producer thread and can overlap device work. Fit momentum
//! shares the two orders, so it has no separate reference spans. These spans are
//! not kernel durations. CubeCL's `CUBECL_DEBUG_LOG=stderr` and
//! `CUBECL_DEBUG_OPTION=profile` enable its own profiling logger; interpret its
//! timing method and backend support independently, and account for its overhead.

use std::{collections::BTreeMap, error::Error, path::Path, time::Instant};
use tracing_subscriber::layer::SubscriberExt;
use umap::{BackendChoice, Init, NeighborSearch, Optimizer, UMAP, UMAPConfig};

const STAGES: [&str; 6] = [
    "neighbor_search",
    "fuzzy_graph",
    "schedule",
    "spectral_graph_export",
    "initialization",
    "optimization",
];

fn read_matrix(path: &Path, rows: usize, features: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let expected = rows.checked_mul(features).and_then(|n| n.checked_mul(4));
    if rows == 0 || features == 0 || expected != Some(bytes.len()) {
        return Err(format!("invalid float32 matrix shape: {}", path.display()).into());
    }
    let flat: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    Ok(flat.chunks_exact(features).map(<[f32]>::to_vec).collect())
}

/// Pairs Chrome-trace B/E events per (thread, stage); the i-th entry is fit i.
fn stage_seconds(events: &[serde_json::Value]) -> Result<BTreeMap<String, Vec<f64>>, &'static str> {
    let mut starts = std::collections::HashMap::<(u64, String), Vec<f64>>::new();
    let mut seconds = BTreeMap::<String, Vec<f64>>::new();
    for event in events {
        let Some(name) = event["name"].as_str().filter(|name| STAGES.contains(name)) else {
            continue;
        };
        let Some(timestamp) = event["ts"].as_f64() else {
            continue;
        };
        let key = (event["tid"].as_u64().unwrap_or(0), name.to_string());
        match event["ph"].as_str() {
            Some("B") => starts.entry(key).or_default().push(timestamp),
            Some("E") => {
                let begin = starts
                    .get_mut(&key)
                    .and_then(Vec::pop)
                    .ok_or("unbalanced stage trace")?;
                seconds
                    .entry(name.to_string())
                    .or_default()
                    .push((timestamp - begin) / 1e6);
            }
            _ => {}
        }
    }
    Ok(seconds)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        return Err("usage: profile_standard_stages INPUT_RAW ROWS FEATURES EPOCHS [--backend cpu|cuda] [--search exact|approximate] [--optimizer compatibility|adam|momentum] [--runs R] [--neighbors K] [--init random|spectral|recursive|auto] [--seed S] [--min-dist D] [--trace PATH] [--components N] [--embedding-out PATH]".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let epochs: usize = args[4].parse()?;
    let mut backend = BackendChoice::Cpu;
    let mut search = NeighborSearch::Approximate;
    let mut optimizer = Optimizer::Adam;
    let mut runs = 2usize;
    let mut neighbors = 15usize;
    let mut init = Init::Recursive;
    let mut seed = 42u64;
    let mut min_dist = 0.1f32;
    let mut trace = None;
    let mut n_components = 2usize;
    let mut embedding_out: Option<String> = None;
    let mut rest = args[5..].iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--backend" => {
                backend = match value.as_str() {
                    "cpu" => BackendChoice::Cpu,
                    "cuda" => BackendChoice::Cuda { device_index: 0 },
                    other => return Err(format!("unknown backend {other}").into()),
                }
            }
            "--search" => {
                search = match value.as_str() {
                    "exact" => NeighborSearch::Exact,
                    "approximate" => NeighborSearch::Approximate,
                    other => return Err(format!("unknown search mode {other}").into()),
                }
            }
            "--optimizer" => {
                optimizer = match value.as_str() {
                    "compatibility" => Optimizer::Compatibility,
                    "adam" => Optimizer::Adam,
                    "momentum" => Optimizer::Momentum,
                    other => return Err(format!("unknown optimizer {other}").into()),
                }
            }
            "--init" => {
                init = match value.as_str() {
                    "random" => Init::Random,
                    "spectral" => Init::Spectral,
                    "recursive" => Init::Recursive,
                    "auto" => Init::Auto,
                    other => return Err(format!("unknown initialization {other}").into()),
                }
            }
            "--runs" => runs = value.parse()?,
            "--neighbors" => neighbors = value.parse()?,
            "--seed" => seed = value.parse()?,
            "--min-dist" => min_dist = value.parse()?,
            "--trace" => trace = Some(value.clone()),
            "--n-components" => n_components = value.parse()?,
            "--embedding-out" => embedding_out = Some(value.clone()),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if runs == 0 || epochs == 0 || neighbors < 2 || neighbors >= rows {
        return Err("invalid runs, epochs, or neighbor count".into());
    }
    let data = read_matrix(Path::new(&args[1]), rows, features)?;
    let trace_path = trace.unwrap_or_else(|| {
        std::env::temp_dir()
            .join(format!(
                "profile_standard_stages-{}.json",
                std::process::id()
            ))
            .display()
            .to_string()
    });
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&trace_path)
        .include_args(true)
        .build();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("warn,umap=info"))
            .with(layer),
    )?;
    let mut fit_seconds = Vec::with_capacity(runs);
    for run in 0..runs {
        let model = UMAP::new(
            UMAPConfig {
                n_neighbors: neighbors,
                n_components,
                n_epochs: Some(epochs),
                min_dist,
                optimizer,
                init,
                random_state: Some(seed + run as u64),
                transform_seed: seed + run as u64,
                ..UMAPConfig::default()
            },
            backend,
            search,
        );
        eprintln!(
            "profile_run_start run={run} unix_ns={}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let start = Instant::now();
        let fitted = model.fit(&data)?;
        fit_seconds.push(start.elapsed().as_secs_f64());
        eprintln!(
            "profile_run_end run={run} unix_ns={}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        if fitted.embedding().iter().flatten().any(|v| !v.is_finite()) {
            return Err("fit produced a non-finite embedding".into());
        }
        if let Some(path) = &embedding_out {
            let bytes: Vec<u8> = fitted
                .embedding()
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            std::fs::write(format!("{path}.{run}"), bytes)?;
        }
    }
    drop(guard);
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&trace_path)?)?;
    if optimizer != Optimizer::Compatibility {
        let mut main_windows = Vec::new();
        let mut start = None;
        for event in &events {
            if event["name"].as_str() != Some("optimization") {
                continue;
            }
            match event["ph"].as_str() {
                Some("B") => start = event["ts"].as_f64(),
                Some("E") => {
                    if let (Some(start), Some(end)) = (start.take(), event["ts"].as_f64()) {
                        main_windows.push(start..=end);
                    }
                }
                _ => {}
            }
        }
        let preparations = events
            .iter()
            .filter(|event| {
                event["name"].as_str() == Some("modern_source_shuffle_copy")
                    && event["ph"].as_str() == Some("B")
                    && event["ts"]
                        .as_f64()
                        .is_some_and(|ts| main_windows.iter().any(|window| window.contains(&ts)))
            })
            .count();
        if main_windows.len() != runs || Some(preparations) != runs.checked_mul(epochs) {
            return Err("modern optimizer producer span count does not match runs × epochs; refusing fallback or incomplete trace".into());
        }
    }
    let stages = stage_seconds(&events)?;
    if stages.values().any(|values| values.len() != runs) {
        return Err("stage span count does not match run count".into());
    }
    println!(
        "{}",
        serde_json::json!({
            "fit_seconds": fit_seconds,
            "stage_seconds": stages,
            "backend": format!("{backend:?}"),
            "optimizer": format!("{optimizer:?}"),
            "search": format!("{search:?}"),
            "init": format!("{init:?}"),
            "neighbors": neighbors,
            "epochs": epochs,
            "components": n_components,
            "rows": rows,
            "features": features,
            "runs": runs,
            "seed": seed,
            "min_dist": min_dist,
            "trace": trace_path,
            "environment": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "cwd": std::env::current_dir()?.display().to_string(),
                "cubecl_debug_log": std::env::var("CUBECL_DEBUG_LOG").ok(),
                "cubecl_debug_option": std::env::var("CUBECL_DEBUG_OPTION").ok(),
                "cuda_visible_devices": std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            },
        })
    );
    Ok(())
}
