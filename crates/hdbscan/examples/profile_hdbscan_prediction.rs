//! Cold/warm prediction timing on a caller-supplied float32 matrix.
//! Requires the `tracing` feature: `required-features` in Cargo.toml makes this example
//! unbuildable without it, so the reported trace file is always a real trace.
//! Compilation is outside the per-case budget.

use hdbscan::{BackendChoice, HDBSCAN, HdbscanConfig, PredictionCoreStatus, PredictionCoresState};
use std::{error::Error, time::Instant};
use tracing_subscriber::layer::SubscriberExt;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: profile_hdbscan_prediction INPUT_F32 ROWS FEATURES QUERIES MIN_SAMPLES cpu|cuda TRACE_JSON".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let queries: usize = args[4].parse()?;
    let min_samples: usize = args[5].parse()?;
    let backend = match args[6].as_str() {
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => return Err("backend must be cpu or cuda".into()),
    };
    if rows == 0 || features == 0 || queries == 0 || queries > rows {
        return Err("require positive dimensions and queries <= rows".into());
    }
    let bytes = std::fs::read(&args[1])?;
    let elements = rows.checked_mul(features).ok_or("input shape overflow")?;
    if bytes.len() != elements.checked_mul(4).ok_or("input byte overflow")? {
        return Err("raw input length does not match shape".into());
    }
    let values: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .copied()
        .map(f32::from_le_bytes)
        .collect();
    let data: Vec<Vec<f32>> = values.chunks_exact(features).map(<[f32]>::to_vec).collect();
    // Fit drops non-finite rows, so geometry and core evidence use this finite count.
    let finite_rows = data
        .iter()
        .filter(|row| row.iter().all(|v| v.is_finite()))
        .count();
    let query: Vec<Vec<f32>> = data[..queries]
        .iter()
        .map(|row| row.iter().map(|v| *v + 0.003125).collect())
        .collect();
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&args[7])
        .include_args(true)
        .build();
    let _subscriber = tracing::subscriber::set_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("warn,hdbscan=info"))
            .with(layer),
    );
    let start = Instant::now();
    let fitted = HDBSCAN::new(
        HdbscanConfig {
            min_cluster_size: 5,
            min_samples: Some(min_samples),
            ..Default::default()
        },
        backend,
    )
    .fit(&data)?;
    let fit_seconds = start.elapsed().as_secs_f64();
    if fitted.finite_index.len() != finite_rows {
        return Err("fitted finite rows disagree with the all-finite input row count".into());
    }
    let before = fitted.prediction_cache_status()?;
    let plan = fitted.prediction_plan(&query)?;
    let start = Instant::now();
    let cold = fitted.approximate_predict(&query)?;
    let cold_seconds = start.elapsed().as_secs_f64();
    let mut warm_seconds = Vec::new();
    for _ in 0..3 {
        let start = Instant::now();
        let warm = fitted.approximate_predict(&query)?;
        warm_seconds.push(start.elapsed().as_secs_f64());
        if cold.labels != warm.labels
            || !cold
                .probabilities
                .iter()
                .map(|v| v.to_bits())
                .eq(warm.probabilities.iter().map(|v| v.to_bits()))
            || cold.connecting_points != warm.connecting_points
        {
            return Err("cold/warm predictions differ".into());
        }
    }
    let after = fitted.prediction_cache_status()?;
    let prediction_core_bytes = match after.cores {
        PredictionCoreStatus::Ready { len, .. } => len * 8,
        PredictionCoreStatus::NotRequiredNoClusterTree | PredictionCoreStatus::Uninitialized => 0,
    };
    drop(guard);
    let trace_bytes = std::fs::metadata(&args[7])?.len();
    // Export only after all timed calls and trace collection. This public API copies
    // the portable predictor state, including prediction cores in finite-row order.
    let predictor_state = fitted.to_predictor()?.export_state()?;
    let (prediction_core_status, prediction_core_rank, prediction_core_bits) =
        match predictor_state.prediction_cores {
            PredictionCoresState::NotRequiredNoClusterTree => {
                ("not_required_no_cluster_tree", None, None)
            }
            PredictionCoresState::Ready { rank, values } => (
                "ready",
                Some(rank),
                Some(
                    values
                        .iter()
                        .map(|v| format!("{:016x}", v.to_bits()))
                        .collect::<Vec<_>>(),
                ),
            ),
        };
    let probability_bits: Vec<_> = cold
        .probabilities
        .iter()
        .map(|v| format!("{:016x}", v.to_bits()))
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "rows": rows, "features": features, "queries": queries, "finite_rows": finite_rows,
            "min_samples": min_samples, "backend": args[6],
            "fit_seconds": fit_seconds, "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds, "no_cluster_tree": cold.no_cluster_tree,
            "noise_count": cold.labels.iter().filter(|&&v| v < 0).count(),
            "labels": cold.labels,
            "connecting_points": cold.connecting_points,
            "probability_bits_f64_hex": probability_bits,
            "prediction_core_status": prediction_core_status,
            "prediction_core_rank": prediction_core_rank,
            "prediction_core_bits_f64_hex_finite_row_order": prediction_core_bits,
            "geometry_bytes": finite_rows * features * 4,
            "prediction_core_bytes": prediction_core_bytes,
            "canonical_host_bytes": after.canonical_bytes,
            "planned_query_batch": plan.as_ref().map(|p| p.query_batch),
            "planned_core_batch": plan.as_ref().map(|p| p.core_batch),
            "planned_device_peak_bytes": plan.as_ref().map(|p| p.device_peak_bytes),
            "planned_host_peak_bytes": plan.as_ref().map(|p| p.host_peak_bytes),
            "peak_memory_kind": "conservative planned live-buffer peaks, not measured process RSS or device usage; planned_host_peak_bytes covers one call's input, outputs, planned retained KD tree and upload/core/query transients, and excludes the canonical fitted state already resident before prediction, so it is not total RSS",
            "tree_bytes_at_fit": before.tree_bytes, "tree_bytes_after_prediction": after.tree_bytes,
            "runtime_warmth": "fit initializes runtime/JIT; first prediction includes prediction-specific JIT; three warm calls reuse host caches but upload/pack each call",
            "trace": args[7], "trace_bytes": trace_bytes,
        })
    );
    Ok(())
}
