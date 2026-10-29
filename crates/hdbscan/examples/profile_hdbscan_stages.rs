//! Per-stage HDBSCAN fit timing companion to comparison/compare_hdbscan_stages.py.
//!
//! Performs one excluded warmup before the requested measured fits.
//! Prints one JSON line with fit and native stage durations in seconds.

use hdbscan::{BackendChoice, ClusterSelectionMethod, HDBSCAN, HdbscanConfig, HdbscanSearch};
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
    named_seconds(events, &STAGES)
}

fn exclude_warmup<T>(samples: &mut Vec<T>, runs: usize) -> Result<(), Box<dyn Error>> {
    if runs == 0 || Some(samples.len()) != runs.checked_add(1) {
        return Err("sample count must equal one warmup plus measured runs".into());
    }
    samples.remove(0);
    Ok(())
}

fn named_seconds(
    events: &[serde_json::Value],
    names: &[&str],
) -> Result<BTreeMap<String, Vec<f64>>, Box<dyn Error>> {
    let mut starts = std::collections::HashMap::<(u64, String), Vec<f64>>::new();
    let mut seconds = BTreeMap::<String, Vec<f64>>::new();
    for event in events {
        let Some(name) = event["name"].as_str().filter(|name| names.contains(name)) else {
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

const BORUVKA_ROUND_STAGES: [&str; 5] = [
    "boruvka_metadata_prepare_upload",
    "boruvka_candidate_kernel",
    "boruvka_component_reduction",
    "boruvka_winner_readback",
    "boruvka_host_merge",
];
const CORE_FINE_STAGES: [&str; 4] = [
    "core_neighbor_kernel",
    "core_extraction",
    "core_neighbor_readback",
    "core_final_assembly_readback",
];

fn core_fine_samples(
    events: &[serde_json::Value],
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut samples = Vec::new();
    let mut start = None;
    for (index, event) in events.iter().enumerate() {
        if event["name"] != "core_distances" {
            continue;
        }
        if event["ph"] == "B" {
            if start.replace(index).is_some() {
                return Err("nested core distance spans".into());
            }
        } else if event["ph"] == "E" {
            let begin = start.take().ok_or("unbalanced core distance span")?;
            let durations = named_seconds(&events[begin..=index], &CORE_FINE_STAGES)?;
            let mut stages = BTreeMap::new();
            for name in CORE_FINE_STAGES {
                stages.insert(name, durations.get(name).cloned().unwrap_or_default());
            }
            samples.push(serde_json::json!({
                "stage_seconds": stages,
                "scope": "Neighbor kernel includes device synchronization; extraction is disjoint from kernel and readback; readback includes transfer and host conversion. Fine spans can repeat per brute-force batch. Assembly is host validation and core-distance/seed assembly. These disjoint scopes are nested within inclusive core_distances."
            }));
        }
    }
    if start.is_some() {
        return Err("unclosed core distance span".into());
    }
    Ok(samples)
}

fn resolved_searches(events: &[serde_json::Value]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| {
            (event["name"] == "mst" && event["ph"] == "B")
                .then(|| event["args"]["search"].as_str().map(str::to_string))
                .flatten()
        })
        .collect()
}

fn boruvka_samples(events: &[serde_json::Value]) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut samples = Vec::new();
    let mut start = None;
    let mut round_components = Vec::new();
    for (index, event) in events.iter().enumerate() {
        if event["name"] != "boruvka_total" {
            continue;
        }
        if event["ph"] == "B" {
            if start.replace(index).is_some() {
                return Err("nested Boruvka totals".into());
            }
            round_components.clear();
        } else if event["ph"] == "E" {
            let begin = start.take().ok_or("unbalanced Boruvka total")?;
            let names = [
                "boruvka_total",
                "boruvka_round",
                "boruvka_seed_init",
                "boruvka_finalize",
            ]
            .into_iter()
            .chain(BORUVKA_ROUND_STAGES)
            .collect::<Vec<_>>();
            let durations = named_seconds(&events[begin..=index], &names)?;
            for round in &events[begin..=index] {
                if round["name"] == "boruvka_round" && round["ph"] == "B" {
                    let count = round["args"]["component_count"]
                        .as_str()
                        .ok_or("Boruvka round lacks component_count")?
                        .parse::<u64>()?;
                    round_components.push(count);
                }
            }
            let total = durations["boruvka_total"][0];
            let rounds = durations.get("boruvka_round").cloned().unwrap_or_default();
            let mut stages = BTreeMap::new();
            for name in BORUVKA_ROUND_STAGES {
                let values = durations.get(name).cloned().unwrap_or_default();
                if values.len() != rounds.len() {
                    return Err(format!("Boruvka {name} and round count mismatch").into());
                }
                stages.insert(name, values);
            }
            let round_total: f64 = rounds.iter().sum();
            samples.push(serde_json::json!({
                "total_seconds": total,
                "round_count": rounds.len(),
                "round_seconds": rounds,
                "round_stage_seconds": stages,
                "component_counts_per_round": round_components,
                "seed_init_seconds": durations.get("boruvka_seed_init").cloned().unwrap_or_default(),
                "finalize_seconds": durations.get("boruvka_finalize").cloned().unwrap_or_default(),
                "seed_and_round_overhead_seconds": total - round_total,
                "scope": "Round stages are disjoint host wall-clock scopes. Metadata includes labels, stable CSR and node-label preparation/uploads; candidate and reduction include profiling device synchronization; winner readback transfers compact outputs; host merge includes component recount. Total and round timings are inclusive; residual is seeding and work outside rounds, not isolated kernel time."
            }));
        }
    }
    if start.is_some() {
        return Err("unclosed Boruvka total".into());
    }
    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::{
        BORUVKA_ROUND_STAGES, CORE_FINE_STAGES, STAGES, boruvka_samples, core_fine_samples,
        exclude_warmup, stage_seconds,
    };
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

    #[test]
    fn excludes_exactly_one_warmup_and_rejects_wrong_counts() {
        for runs in [1, 3] {
            let mut samples: Vec<usize> = (0..=runs).collect();
            exclude_warmup(&mut samples, runs).unwrap();
            assert_eq!(samples, (1..=runs).collect::<Vec<_>>());
            assert!(exclude_warmup(&mut samples, runs).is_err());
        }
        assert!(exclude_warmup(&mut vec![0], 0).is_err());
        assert!(exclude_warmup(&mut vec![0], usize::MAX).is_err());
    }

    #[test]
    fn separates_round_samples_and_accounts_for_driver_work() {
        let mut events = vec![
            json!({"name":"boruvka_total", "ph":"B", "ts":0.0}),
            json!({"name":"boruvka_round", "ph":"B", "ts":1.0, "args":{"component_count":"8"}}),
        ];
        for (index, stage) in BORUVKA_ROUND_STAGES.into_iter().enumerate() {
            events.push(json!({"name":stage, "ph":"B", "ts":1.0 + index as f64}));
            events.push(json!({"name":stage, "ph":"E", "ts":2.0 + index as f64}));
        }
        events.extend([
            json!({"name":"boruvka_round", "ph":"E", "ts":7.0}),
            json!({"name":"boruvka_total", "ph":"E", "ts":10.0}),
        ]);
        let samples = boruvka_samples(&events).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0]["round_count"], 1);
        assert_eq!(samples[0]["component_counts_per_round"], json!([8]));
        assert_eq!(samples[0]["round_seconds"], json!([6e-6]));
        assert!(
            (samples[0]["seed_and_round_overhead_seconds"]
                .as_f64()
                .unwrap()
                - 4e-6)
                .abs()
                < 1e-15
        );
        for stage in BORUVKA_ROUND_STAGES {
            assert_eq!(samples[0]["round_stage_seconds"][stage], json!([1e-6]));
        }
        assert!(boruvka_samples(&events[..events.len() - 1]).is_err());
        assert!(boruvka_samples(&events[events.len() - 1..]).is_err());
        events.remove(2);
        assert!(boruvka_samples(&events).is_err());
    }

    #[test]
    fn aggregates_fine_core_spans_per_fit_and_round_component_metadata() {
        let events = vec![
            json!({"name":"core_distances", "tid":1, "ph":"B", "ts":0.0}),
            json!({"name":"core_neighbor_kernel", "tid":1, "ph":"B", "ts":1.0}),
            json!({"name":"core_neighbor_kernel", "tid":1, "ph":"E", "ts":3.0}),
            json!({"name":"core_extraction", "tid":1, "ph":"B", "ts":3.0}),
            json!({"name":"core_extraction", "tid":1, "ph":"E", "ts":4.0}),
            json!({"name":"core_neighbor_readback", "tid":1, "ph":"B", "ts":4.0}),
            json!({"name":"core_neighbor_readback", "tid":1, "ph":"E", "ts":7.0}),
            json!({"name":"core_final_assembly_readback", "tid":1, "ph":"B", "ts":7.0}),
            json!({"name":"core_final_assembly_readback", "tid":1, "ph":"E", "ts":8.0}),
            json!({"name":"core_distances", "tid":1, "ph":"E", "ts":9.0}),
        ];
        let samples = core_fine_samples(&events).unwrap();
        assert_eq!(samples.len(), 1);
        for name in CORE_FINE_STAGES {
            assert_eq!(
                samples[0]["stage_seconds"][name].as_array().unwrap().len(),
                1
            );
        }
        assert_eq!(
            samples[0]["stage_seconds"]["core_neighbor_kernel"],
            json!([2e-6])
        );
    }

    fn traced_fit(backend: hdbscan::BackendChoice) {
        use tracing_subscriber::layer::SubscriberExt;
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.json");
        let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(&path)
            .include_args(true)
            .build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("warn,hdbscan=info"))
            .with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let rows = (0..48)
                .map(|row| {
                    let x = row as f32;
                    vec![(x * 0.173).sin(), (x * 0.119).cos(), (x * 0.071).sin()]
                })
                .collect::<Vec<_>>();
            for search in [
                hdbscan::HdbscanSearch::BruteForce,
                hdbscan::HdbscanSearch::KdTree,
            ] {
                hdbscan::HDBSCAN::new(
                    hdbscan::HdbscanConfig {
                        min_cluster_size: 4,
                        min_samples: Some(4),
                        mst_graph: hdbscan::MstGraph::Complete,
                        ..Default::default()
                    },
                    backend,
                )
                .with_search(search)
                .fit(&rows)
                .unwrap();
            }
        });
        drop(guard);
        let events =
            serde_json::from_slice::<Vec<serde_json::Value>>(&std::fs::read(path).unwrap())
                .unwrap();
        let samples = boruvka_samples(&events).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(super::resolved_searches(&events), ["BruteForce", "KdTree"]);
        let core_samples = core_fine_samples(&events).unwrap();
        assert_eq!(core_samples.len(), 2);
        for (index, sample) in core_samples.into_iter().enumerate() {
            for stage in CORE_FINE_STAGES {
                let expected = if index == 0 && stage != "core_final_assembly_readback" {
                    2
                } else {
                    1
                };
                assert_eq!(
                    sample["stage_seconds"][stage].as_array().unwrap().len(),
                    expected
                );
            }
        }
        for sample in samples {
            let rounds = sample["round_count"].as_u64().unwrap() as usize;
            assert!(rounds > 0, "fixture must exercise component reduction");
            assert_eq!(
                sample["component_counts_per_round"]
                    .as_array()
                    .unwrap()
                    .len(),
                rounds
            );
            assert_eq!(sample["seed_init_seconds"].as_array().unwrap().len(), 1);
            assert_eq!(sample["finalize_seconds"].as_array().unwrap().len(), 1);
            for stage in BORUVKA_ROUND_STAGES {
                assert_eq!(
                    sample["round_stage_seconds"][stage]
                        .as_array()
                        .unwrap()
                        .len(),
                    rounds
                );
            }
        }
        assert!(
            events
                .iter()
                .all(|event| event["name"] != "boruvka_candidates_readback")
        );
    }

    #[test]
    fn real_traced_fit_cpu_stage_counts() {
        traced_fit(hdbscan::BackendChoice::Cpu);
    }

    #[test]
    fn real_traced_fit_cuda_stage_counts() {
        let count = cuscan_common::cuda_backend::try_device_count().unwrap();
        if count == 0 {
            assert_ne!(
                std::env::var("CUSCAN_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED tracing CUDA: no usable device");
            return;
        }
        traced_fit(hdbscan::BackendChoice::Cuda { device_index: 0 });
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: profile_hdbscan_stages INPUT_RAW ROWS FEATURES [--backend cpu|cuda] [--runs R] [--min-cluster-size N] [--min-samples N] [--cluster-selection-method eom|leaf] [--allow-single-cluster] [--search auto|brute-force|kd-tree] [--labels-output PATH] [--trace-output PATH]".into());
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
    let mut labels_output = None;
    let mut trace_output = None;
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
            "--labels-output" => labels_output = Some(value.clone()),
            "--trace-output" => trace_output = Some(value.clone()),
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
        .include_args(true)
        .build();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new("warn,hdbscan=info"))
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
    let warmup_start = Instant::now();
    estimator.fit(&data)?;
    let warmup_seconds = warmup_start.elapsed().as_secs_f64();
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
        if run + 1 == runs
            && let Some(path) = &labels_output
        {
            std::fs::write(path, serde_json::to_vec(fitted.labels())?)?;
        }
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
    let mut stages = stage_seconds(&events)?;
    let mut boruvka = boruvka_samples(&events)?;
    let mut core_fine = core_fine_samples(&events)?;
    let mut resolved = resolved_searches(&events);
    exclude_warmup(&mut boruvka, runs)?;
    exclude_warmup(&mut core_fine, runs)?;
    exclude_warmup(&mut resolved, runs)?;
    if core_fine.len() != runs || resolved.len() != runs {
        return Err("fine timing or resolved-search sample count does not match run count".into());
    }
    if stages.len() != STAGES.len()
        || STAGES.iter().any(|stage| {
            stages
                .get(*stage)
                .is_none_or(|values| values.len() != runs + 1)
        })
    {
        return Err("stage span count does not match run count".into());
    }
    for values in stages.values_mut() {
        exclude_warmup(values, runs)?;
    }
    if let Some(path) = trace_output {
        std::fs::copy(&trace_path, path)?;
    }
    std::fs::remove_file(&trace_path)?;
    println!(
        "{}",
        serde_json::json!({
            "fit_seconds": fit_seconds,
            "warmup_seconds": warmup_seconds,
            "boruvka_samples": boruvka,
            "core_distance_samples": core_fine,
            "stage_seconds": stages,
            "cluster_counts": cluster_counts,
            "noise_counts": noise_counts,
            "backend": format!("{backend:?}"),
            "search": format!("{search:?}"),
            "resolved_search_per_fit": resolved,
        })
    );
    Ok(())
}
