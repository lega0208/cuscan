//! Per-stage HDBSCAN fit timing companion to comparison/compare_hdbscan_stages.py.
//!
//! Runs without warmup; the first fit includes process-cold kernel compilation.
//! Prints one JSON line with fit and native stage durations in seconds.

use fast_hdbscan::{BackendChoice, ClusterSelectionMethod, HDBSCAN, HdbscanConfig, HdbscanSearch};
use std::{collections::BTreeMap, error::Error, path::Path, time::Instant};
use tracing_subscriber::layer::SubscriberExt;

const STAGES: [&str; 9] = [
    "input_validation",
    "backend_setup",
    "input_preparation",
    "tree_build",
    "input_upload",
    "core_distances",
    "mst",
    "cluster_extraction",
    "result_finalization",
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
    if flat.iter().any(|value| !value.is_finite()) {
        return Err("input matrix contains a non-finite value".into());
    }
    Ok(flat.chunks_exact(features).map(<[f32]>::to_vec).collect())
}

fn stage_seconds(
    events: &[serde_json::Value],
) -> Result<BTreeMap<String, Vec<f64>>, Box<dyn Error>> {
    let mut starts = std::collections::HashMap::<(u64, String), Vec<f64>>::new();
    let mut seconds = BTreeMap::<String, Vec<f64>>::new();
    for event in events {
        let Some(name) = event["name"].as_str().filter(|name| STAGES.contains(name)) else {
            continue;
        };
        let key = (event["tid"].as_u64().unwrap_or(0), name.to_string());
        match event["ph"].as_str() {
            Some("B") => {
                let timestamp = event["ts"]
                    .as_f64()
                    .ok_or("stage start has invalid timestamp")?;
                starts.entry(key).or_default().push(timestamp);
            }
            Some("E") => {
                let timestamp = event["ts"]
                    .as_f64()
                    .ok_or("stage end has invalid timestamp")?;
                let begin = starts
                    .get_mut(&key)
                    .and_then(Vec::pop)
                    .ok_or("unbalanced stage trace")?;
                let duration = (timestamp - begin) / 1e6;
                if !duration.is_finite() || duration < 0.0 {
                    return Err("stage trace has invalid duration".into());
                }
                seconds.entry(name.to_string()).or_default().push(duration);
            }
            _ => {}
        }
    }
    if starts.values().any(|open| !open.is_empty()) {
        return Err("unclosed stage trace span".into());
    }
    Ok(seconds)
}

#[cfg(test)]
mod tests {
    use super::{STAGES, stage_seconds};
    use serde_json::json;

    #[test]
    fn parses_each_granular_stage_and_ignores_unrelated_spans() {
        let mut events = Vec::new();
        for (index, name) in STAGES.iter().enumerate() {
            let begin = index as f64 * 10.0;
            events.push(json!({"name": name, "tid": 1, "ph": "B", "ts": begin}));
            events.push(json!({"name": name, "tid": 1, "ph": "E", "ts": begin + 2.0}));
        }
        events.push(json!({"name": "fit", "tid": 1, "ph": "B", "ts": 0.0}));
        events.push(json!({"name": "fit", "tid": 1, "ph": "E", "ts": 100.0}));

        let stages = stage_seconds(&events).expect("valid stage trace");
        assert_eq!(stages.len(), STAGES.len());
        for name in STAGES {
            assert_eq!(stages[name], [2e-6]);
        }
        assert!(!stages.contains_key("minimum_spanning_tree"));
    }

    #[test]
    fn rejects_unmatched_and_invalid_duration_events() {
        let unmatched = [json!({"name": "mst", "tid": 7, "ph": "E", "ts": 2.0})];
        assert!(stage_seconds(&unmatched).is_err());

        let invalid_duration = [
            json!({"name": "mst", "tid": 7, "ph": "B", "ts": 3.0}),
            json!({"name": "mst", "tid": 7, "ph": "E", "ts": 2.0}),
        ];
        assert!(stage_seconds(&invalid_duration).is_err());
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: profile_hdbscan_stages INPUT_RAW ROWS FEATURES [--backend cpu|cuda] [--runs R] [--min-cluster-size N] [--min-samples N] [--cluster-selection-method eom|leaf] [--allow-single-cluster] [--search auto|brute-force|kd-tree]".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let mut backend = BackendChoice::Cpu;
    let mut runs = 1usize;
    let mut min_cluster_size = 5usize;
    let mut min_samples = None;
    let mut selection = ClusterSelectionMethod::Eom;
    let mut allow_single_cluster = false;
    let mut search = HdbscanSearch::Auto;
    let mut i = 4;
    while i < args.len() {
        let flag = &args[i];
        if flag == "--allow-single-cluster" {
            allow_single_cluster = true;
            i += 1;
            continue;
        }
        let value = args.get(i + 1).ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--backend" => {
                backend = match value.as_str() {
                    "cpu" => BackendChoice::Cpu,
                    "cuda" => BackendChoice::Cuda { device_index: 0 },
                    other => return Err(format!("unknown backend {other}").into()),
                }
            }
            "--runs" => runs = value.parse()?,
            "--min-cluster-size" => min_cluster_size = value.parse()?,
            "--min-samples" => min_samples = Some(value.parse()?),
            "--cluster-selection-method" => {
                selection = match value.as_str() {
                    "eom" => ClusterSelectionMethod::Eom,
                    "leaf" => ClusterSelectionMethod::Leaf,
                    other => return Err(format!("unknown cluster-selection-method {other}").into()),
                }
            }
            "--search" => {
                search = match value.as_str() {
                    "auto" => HdbscanSearch::Auto,
                    "brute-force" => HdbscanSearch::BruteForce,
                    "kd-tree" => HdbscanSearch::KdTree,
                    other => return Err(format!("unknown search mode {other}").into()),
                }
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
        i += 2;
    }
    if rows < 2 || features == 0 || runs == 0 || min_cluster_size < 2 || min_samples == Some(0) {
        return Err("rows and min-cluster-size must be at least 2; features, runs, and min-samples must be positive".into());
    }
    let data = read_matrix(Path::new(&args[1]), rows, features)?;
    HdbscanConfig {
        min_cluster_size,
        min_samples,
        cluster_selection_method: selection,
        allow_single_cluster,
        ..HdbscanConfig::default()
    }
    .validate()?;

    let trace_path = std::env::temp_dir().join(format!(
        "profile_hdbscan_stages-{}.json",
        std::process::id()
    ));
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&trace_path)
        .build();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("warn,fast_hdbscan=info"))
            .with(layer),
    )?;
    let config = HdbscanConfig {
        min_cluster_size,
        min_samples,
        cluster_selection_method: selection,
        allow_single_cluster,
        ..HdbscanConfig::default()
    };
    let estimator = HDBSCAN::new(config, backend).with_search(search);
    let mut fit_seconds = Vec::with_capacity(runs);
    let mut cluster_counts = Vec::with_capacity(runs);
    let mut noise_counts = Vec::with_capacity(runs);
    for run in 0..runs {
        eprintln!(
            "profile_run_start run={run} unix_ns={}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let start = Instant::now();
        let fitted = estimator.fit(&data)?;
        fit_seconds.push(start.elapsed().as_secs_f64());
        eprintln!(
            "profile_run_end run={run} unix_ns={}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        );
        let mut clusters = std::collections::HashSet::new();
        let mut noise = 0;
        for &label in fitted.labels() {
            if label < 0 {
                noise += 1;
            } else {
                clusters.insert(label);
            }
        }
        cluster_counts.push(clusters.len());
        noise_counts.push(noise);
    }
    drop(guard);
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&trace_path)?)?;
    let stages = stage_seconds(&events)?;
    if stages.len() != STAGES.len()
        || STAGES
            .iter()
            .any(|stage| stages.get(*stage).is_none_or(|values| values.len() != runs))
    {
        return Err("stage span count does not match run count".into());
    }
    std::fs::remove_file(&trace_path)?;
    println!(
        "{}",
        serde_json::json!({
            "fit_seconds": fit_seconds,
            "stage_seconds": stages,
            "cluster_counts": cluster_counts,
            "noise_counts": noise_counts,
            "backend": format!("{backend:?}"),
            "search": format!("{search:?}"),
        })
    );
    Ok(())
}
