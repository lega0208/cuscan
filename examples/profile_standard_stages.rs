//! Per-stage Standard UMAP fit timing companion to comparison/compare_stages.py.
//!
//! usage: profile_standard_stages INPUT_RAW ROWS FEATURES EPOCHS
//!        [--backend cpu|cuda] [--search exact|approximate] [--runs R]
//!        [--neighbors K] [--init random|spectral] [--seed S] [--min-dist D]
//!        [--trace PATH]
//!
//! Runs R fits in this process without warmup, so run 0 is process-cold
//! (includes kernel JIT/compilation) and later runs are warm. Stage spans come
//! from the library `tracing` spans, which insert device barriers. Prints one
//! JSON line with per-run `fit_seconds` and `stage_seconds`.

use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};
use std::{collections::BTreeMap, error::Error, path::Path, time::Instant};
use tracing_subscriber::layer::SubscriberExt;

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
        return Err("usage: profile_standard_stages INPUT_RAW ROWS FEATURES EPOCHS [--backend cpu|cuda] [--search exact|approximate] [--runs R] [--neighbors K] [--init random|spectral] [--seed S] [--min-dist D] [--trace PATH]".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let epochs: usize = args[4].parse()?;
    let mut backend = BackendChoice::Cpu;
    let mut search = NeighborSearch::Approximate;
    let mut runs = 3usize;
    let mut neighbors = 15usize;
    let mut init = Init::Spectral;
    let mut seed = 42u64;
    let mut min_dist = 0.1f32;
    let mut trace = None;
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
            "--init" => {
                init = match value.as_str() {
                    "random" => Init::Random,
                    "spectral" => Init::Spectral,
                    other => return Err(format!("unknown initialization {other}").into()),
                }
            }
            "--runs" => runs = value.parse()?,
            "--neighbors" => neighbors = value.parse()?,
            "--seed" => seed = value.parse()?,
            "--min-dist" => min_dist = value.parse()?,
            "--trace" => trace = Some(value.clone()),
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
            .with(tracing_subscriber::EnvFilter::new("warn,fast_umap=info"))
            .with(layer),
    )?;
    let mut fit_seconds = Vec::with_capacity(runs);
    for run in 0..runs {
        let model = UMAP::new(
            UMAPConfig {
                n_neighbors: neighbors,
                n_components: 2,
                n_epochs: Some(epochs),
                min_dist,
                init,
                random_state: Some(seed + run as u64),
                transform_seed: seed + run as u64,
                ..UMAPConfig::default()
            },
            backend,
            search,
        );
        let start = Instant::now();
        let fitted = model.fit(&data)?;
        fit_seconds.push(start.elapsed().as_secs_f64());
        if fitted.embedding().iter().flatten().any(|v| !v.is_finite()) {
            return Err("fit produced a non-finite embedding".into());
        }
    }
    drop(guard);
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&trace_path)?)?;
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
            "search": format!("{search:?}"),
            "init": format!("{init:?}"),
            "neighbors": neighbors,
            "epochs": epochs,
        })
    );
    Ok(())
}
