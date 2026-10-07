//! CUDA-only companion to scripts/compare_umap_spans.py. Build before timing.
use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};
use std::{error::Error, time::Instant};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: profile_standard_cuda INPUT_RAW ROWS FEATURES EPOCHS TRACE_PATH".into(),
        );
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let epochs: usize = args[4].parse()?;
    let bytes = std::fs::read(&args[1])?;
    let expected = rows.checked_mul(features).and_then(|n| n.checked_mul(4));
    if rows <= 15 || features == 0 || epochs == 0 || expected != Some(bytes.len()) {
        return Err("invalid shape, epochs, or float32 input length".into());
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let data: Vec<Vec<f32>> = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    if fast_umap::cuda_backend::device(0).is_none() {
        return Err("CUDA device 0 unavailable; no CPU fallback permitted".into());
    }
    let estimator = UMAP::new(
        UMAPConfig {
            n_neighbors: 15,
            n_components: 2,
            n_epochs: Some(epochs),
            init: Init::Random,
            random_state: Some(42),
            ..UMAPConfig::default()
        },
        BackendChoice::Cuda { device_index: 0 },
        NeighborSearch::Exact,
    );
    // fit materializes the embedding on the host, synchronizing CUDA work.
    estimator.fit(&data)?;
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&args[5])
        .build();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info,fast_umap=trace"))
        .with(layer)
        .try_init()?;
    let start = Instant::now();
    let fitted = estimator.fit(&data)?;
    let fit_seconds = start.elapsed().as_secs_f64();
    if fitted
        .embedding()
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err("CUDA fit produced non-finite embedding".into());
    }
    drop(guard);
    println!("{}", serde_json::json!({"fit_seconds": fit_seconds}));
    Ok(())
}
