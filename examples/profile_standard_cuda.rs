//! CUDA-only companion to scripts/compare_umap_spans.py. Build before timing.
//!
//! usage: profile_standard_cuda INPUT_RAW ROWS FEATURES EPOCHS OUT_DIR
//!        --search exact|approximate [--warmups W] [--runs R] [--trace PATH]
//!
//! Runs W untimed warmup fits, then R timed fits. Each timed fit includes
//! neighbor search and host embedding readback. After each fit's timer stops,
//! the embedding (`embedding_<i>.f32le`) and actual fit neighbor ids
//! (`graph_<i>.i32le`, non-self) are written to OUT_DIR. `--trace` records a
//! Chrome trace of the timed fits only (host spans, not GPU durations).
use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};
use std::{error::Error, path::PathBuf, time::Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err(
            "usage: profile_standard_cuda INPUT_RAW ROWS FEATURES EPOCHS OUT_DIR \
                    --search exact|approximate [--warmups W] [--runs R] [--trace PATH]"
                .into(),
        );
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let epochs: usize = args[4].parse()?;
    let out_dir = PathBuf::from(&args[5]);
    let mut search = None;
    let mut warmups = 1usize;
    let mut runs = 1usize;
    let mut trace = None;
    let mut rest = args[6..].iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--search" => {
                search = Some(match value.as_str() {
                    "exact" => NeighborSearch::Exact,
                    "approximate" => NeighborSearch::Approximate,
                    other => return Err(format!("unknown search mode {other}").into()),
                })
            }
            "--warmups" => warmups = value.parse()?,
            "--runs" => runs = value.parse()?,
            "--trace" => trace = Some(value.clone()),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    let search = search.ok_or("--search exact|approximate is required (no Auto)")?;
    let bytes = std::fs::read(&args[1])?;
    let expected = rows.checked_mul(features).and_then(|n| n.checked_mul(4));
    if rows <= 15 || features == 0 || epochs == 0 || runs == 0 || expected != Some(bytes.len()) {
        return Err("invalid shape, epochs, runs, or float32 input length".into());
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let data: Vec<Vec<f32>> = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    if fast_umap::cuda_backend::device(0).is_none() {
        return Err("CUDA device 0 unavailable; no CPU fallback permitted".into());
    }
    std::fs::create_dir_all(&out_dir)?;
    let estimator = |seed: u64| {
        UMAP::new(
            UMAPConfig {
                n_neighbors: 15,
                n_components: 2,
                n_epochs: Some(epochs),
                init: Init::Random,
                random_state: Some(seed),
                ..UMAPConfig::default()
            },
            BackendChoice::Cuda { device_index: 0 },
            search,
        )
    };
    for warmup in 0..warmups {
        estimator(1000 + warmup as u64).fit(&data)?;
    }
    let guard = match &trace {
        Some(path) => {
            let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new().file(path).build();
            tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new("info,fast_umap=trace"))
                .with(layer)
                .try_init()?;
            Some(guard)
        }
        None => None,
    };
    let mut fit_seconds = Vec::with_capacity(runs);
    let mut graph_width = 0usize;
    for run in 0..runs {
        let model = estimator(42 + run as u64);
        let start = Instant::now();
        // fit materializes the embedding on the host, synchronizing CUDA work.
        let fitted = model.fit(&data)?;
        fit_seconds.push(start.elapsed().as_secs_f64());
        let embedding: Vec<f32> = fitted.embedding().iter().flatten().copied().collect();
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err("CUDA fit produced non-finite embedding".into());
        }
        let (graph, width) = fitted.diagnostic_fit_graph()?;
        graph_width = width;
        std::fs::write(
            out_dir.join(format!("embedding_{run}.f32le")),
            embedding
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
        std::fs::write(
            out_dir.join(format!("graph_{run}.i32le")),
            graph
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )?;
    }
    drop(guard);
    let approximate_iterations = ((rows as f64).log2().round() as usize).max(5);
    println!(
        "{}",
        serde_json::json!({
            "fit_seconds": fit_seconds,
            "search": format!("{search:?}"),
            "fit_graph_width": graph_width,
            "approximate_iterations_rule": "max(round(log2(n)), 5)",
            "approximate_iterations": approximate_iterations,
            "warmups": warmups,
        })
    );
    Ok(())
}
