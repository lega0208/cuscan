use cubecl_runtime::logging::{LaunchObservation, LaunchObserver};
use fast_hdbscan::{BackendChoice, HDBSCAN, HdbscanConfig, HdbscanSearch};
use std::{
    collections::BTreeMap,
    error::Error,
    fs,
    sync::{Arc, Mutex},
};

#[derive(Default)]
struct Counts(Mutex<BTreeMap<&'static str, usize>>);
impl LaunchObserver for Counts {
    fn launched(&self, kernel: &'static str) {
        *self.0.lock().unwrap().entry(kernel).or_default() += 1;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 7 {
        return Err("usage: capture_launches RAW ROWS FEATURES MIN_SAMPLES BACKEND SEARCH".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let bytes = fs::read(&args[1])?;
    if rows < 2
        || features == 0
        || rows.checked_mul(features).and_then(|n| n.checked_mul(4)) != Some(bytes.len())
    {
        return Err("invalid matrix dimensions".into());
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let matrix: Vec<Vec<f32>> = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    let backend = match args[5].as_str() {
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => return Err("invalid backend".into()),
    };
    let search = match args[6].as_str() {
        "brute-force" => HdbscanSearch::BruteForce,
        "kd-tree" => HdbscanSearch::KdTree,
        _ => return Err("invalid search".into()),
    };
    let estimator = HDBSCAN::new(
        HdbscanConfig {
            min_cluster_size: 5,
            min_samples: Some(args[4].parse()?),
            ..Default::default()
        },
        backend,
    )
    .with_search(search);
    estimator.fit(&matrix)?;
    if cubecl_runtime::dry_run::dry_run() {
        return Err("refusing to count skipped launches".into());
    }
    let counts = Arc::new(Counts::default());
    let observation = LaunchObservation::new(counts.clone());
    let fitted = estimator.fit(&matrix)?;
    drop(observation);
    let kernels = counts.0.lock().unwrap();
    println!(
        "{}",
        serde_json::json!({
            "kind": "algorithm-launch-observer-report", "timing": "disabled", "warmup_fits": 1,
            "observed_fits": 1, "dry_run": false, "graph_capture_or_replay": false,
            "total_issued_launches": kernels.values().sum::<usize>(), "kernels": *kernels,
            "labels": fitted.labels(), "core_f32_bits": fitted.core_distances().iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        })
    );
    Ok(())
}
