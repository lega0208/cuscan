//! M4-b ANN width campaign.
//!
//! Ignored hardware campaigns write `cpu.json` / `cuda.json` under `FAST_UMAP_CAMPAIGN_DIR`.
//! `compare_campaign_pairs` aligns both files and enforces the cross-backend gates.
//! The ARI unit test is ordinary and needs no hardware.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use fast_cubecl_common::selected_device;
use fast_hdbscan::{
    BackendChoice, FittedHDBSCAN, HdbscanConfig, KnnDiagnostics, KnnNeighborSource, KnnSource,
    fit_knn_graph_with_source,
};
use serde_json::{Value, json};

const CAMPAIGN_DIR_ENV: &str = "FAST_UMAP_CAMPAIGN_DIR";
const SCHEMA: &str = "fast-hdbscan.knn-width-campaign.v1";
const ROWS: usize = 6000;
const CENTRES: usize = 8;
const MIN_SAMPLES: usize = 25;
const REQUESTED_K: [usize; 3] = [15, 45, 60];
const SEEDS: [u64; 3] = [42, 43, 44];
const DATASETS: [(&str, usize); 2] = [("blobs50", 50), ("blobs40", 40)];
const RECALL_LIMIT: f64 = 0.90;
const ARI_LIMIT: f64 = 0.95;
const SUM_RELATIVE_LIMIT: f64 = 0.01;
const WARMUP_FITS: usize = 1;
const TIMED_FITS: usize = 3;
const PAIR_RECORDS: usize = DATASETS.len() * REQUESTED_K.len() * SEEDS.len();

// ---------- deterministic data ----------

/// SplitMix64: small, stable, and dependency-free (rand is not a fast-hdbscan dependency).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }

    /// Box-Muller standard normal.
    fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(f64::MIN_POSITIVE);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Eight uniform centres in [-10, 10]^d with unit Gaussian noise; row i belongs to centre i % 8.
fn make_dataset(dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed ^ ((dims as u64) << 32));
    let centres: Vec<Vec<f64>> = (0..CENTRES)
        .map(|_| (0..dims).map(|_| rng.uniform(-10.0, 10.0)).collect())
        .collect();
    (0..ROWS)
        .map(|row| {
            let centre = &centres[row % CENTRES];
            (0..dims)
                .map(|d| (centre[d] + rng.normal()) as f32)
                .collect()
        })
        .collect()
}

// ---------- width and metric helpers ----------

/// Upstream graph width: effective_k = min(max(requested, s), n - 1).
fn effective_width(requested: usize, n: usize) -> usize {
    requested.max(MIN_SAMPLES).min(n.saturating_sub(1))
}

fn squared_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let delta = f64::from(x) - f64::from(y);
            delta * delta
        })
        .sum()
}

/// Exact kth non-self squared-distance cutoff per row, for each requested width.
/// Matches the UMAP tie-aware metric: the cutoff is the kth smallest non-self distance, inclusive.
fn exact_cutoffs(data: &[Vec<f32>], widths: &[usize]) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut cutoffs = vec![vec![f64::INFINITY; n]; widths.len()];
    for row in 0..n {
        let mut distances: Vec<f64> = (0..n)
            .filter(|&other| other != row)
            .map(|other| squared_distance(&data[row], &data[other]))
            .collect();
        distances.sort_unstable_by(f64::total_cmp);
        for (slot, &k) in widths.iter().enumerate() {
            cutoffs[slot][row] = distances[k - 1];
        }
    }
    cutoffs
}

/// Tie-aware non-self recall: unique ANN ids whose exact squared distance is within the kth cutoff.
fn tie_aware_recall(ids: &[i32], data: &[Vec<f32>], k: usize, cutoffs: &[f64]) -> f64 {
    let n = data.len();
    let mut hits = 0usize;
    for row in 0..n {
        let mut seen = HashSet::new();
        for &id in &ids[row * k..(row + 1) * k] {
            if id < 0 || id as usize >= n || id as usize == row {
                continue;
            }
            if seen.insert(id) && squared_distance(&data[row], &data[id as usize]) <= cutoffs[row] {
                hits += 1;
            }
        }
    }
    hits as f64 / (n * k) as f64
}

/// Hubert and Arabie adjusted Rand index. Label -1 is an ordinary class.
fn adjusted_rand_index(a: &[i32], b: &[i32]) -> f64 {
    assert_eq!(a.len(), b.len(), "ARI requires equal-length labelings");
    let n = a.len();
    let comb2 = |x: u64| x as f64 * (x as f64 - 1.0) / 2.0;
    let mut index_a: HashMap<i32, usize> = HashMap::new();
    let mut index_b: HashMap<i32, usize> = HashMap::new();
    let mut cells: HashMap<(usize, usize), u64> = HashMap::new();
    let mut row_sums: Vec<u64> = Vec::new();
    let mut col_sums: Vec<u64> = Vec::new();
    for (&x, &y) in a.iter().zip(b) {
        let len_a = index_a.len();
        let i = *index_a.entry(x).or_insert(len_a);
        let len_b = index_b.len();
        let j = *index_b.entry(y).or_insert(len_b);
        if i == row_sums.len() {
            row_sums.push(0);
        }
        if j == col_sums.len() {
            col_sums.push(0);
        }
        row_sums[i] += 1;
        col_sums[j] += 1;
        *cells.entry((i, j)).or_insert(0) += 1;
    }
    let total = comb2(n as u64);
    if total == 0.0 {
        return 1.0;
    }
    let sum_cells: f64 = cells.values().map(|&c| comb2(c)).sum();
    let sum_a: f64 = row_sums.iter().map(|&c| comb2(c)).sum();
    let sum_b: f64 = col_sums.iter().map(|&c| comb2(c)).sum();
    let expected = sum_a * sum_b / total;
    let maximum = 0.5 * (sum_a + sum_b);
    if maximum == expected {
        return 1.0;
    }
    (sum_cells - expected) / (maximum - expected)
}

#[test]
fn ari_matches_known_pair() {
    // Known pair: contingency cells (2, 1, 1); sum_cells=1, sum_a=1, sum_b=2, total=6.
    // expected=1/3, max=3/2, ARI=(1-1/3)/(3/2-1/3)=4/7.
    let ari = adjusted_rand_index(&[0, 0, 1, 2], &[0, 0, 1, 1]);
    assert!((ari - 4.0 / 7.0).abs() < 1e-12, "ARI {ari} != 4/7");
    assert!((adjusted_rand_index(&[0, 0, 1, 1], &[5, 5, -1, -1]) - 1.0).abs() < 1e-12);
    assert!((adjusted_rand_index(&[3, 3, 3], &[7, 7, 7]) - 1.0).abs() < 1e-12);
}

// ---------- fitting and snapshots ----------

struct Snapshot {
    neighbors: Vec<i32>,
    distance_bits: Vec<u32>,
    k: usize,
    source: KnnNeighborSource,
    bridges: usize,
    finite_sum: f64,
    allocation_bytes: usize,
    labels: Vec<i32>,
    mst: Vec<(u32, u32, u64)>,
}

impl Snapshot {
    fn from_fit(fitted: &FittedHDBSCAN) -> Result<Self, String> {
        let diagnostics: &KnnDiagnostics = fitted
            .knn_diagnostics()
            .ok_or_else(|| "missing kNN diagnostics".to_string())?;
        Ok(Self {
            neighbors: diagnostics.neighbors.clone(),
            distance_bits: diagnostics
                .distances_sq
                .iter()
                .map(|d| d.to_bits())
                .collect(),
            k: diagnostics.k,
            source: diagnostics.source,
            bridges: diagnostics.bridges,
            finite_sum: diagnostics.finite_weight_sum,
            allocation_bytes: diagnostics.allocation_bytes,
            labels: fitted.labels().to_vec(),
            mst: fitted
                .minimum_spanning_tree()
                .iter()
                .map(|e| (e.src, e.dst, e.weight.to_bits()))
                .collect(),
        })
    }

    fn same_outputs(&self, other: &Self) -> bool {
        self.neighbors == other.neighbors
            && self.distance_bits == other.distance_bits
            && self.labels == other.labels
            && self.mst == other.mst
            && self.bridges == other.bridges
    }

    fn edge_hash(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &(src, dst, weight_bits) in &self.mst {
            for byte in src
                .to_le_bytes()
                .into_iter()
                .chain(dst.to_le_bytes())
                .chain(weight_bits.to_le_bytes())
            {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        hash
    }
}

fn campaign_config(requested: usize) -> HdbscanConfig {
    HdbscanConfig {
        min_samples: Some(MIN_SAMPLES),
        knn_k: Some(requested),
        ..Default::default()
    }
}

fn timed_fit(
    data: &[Vec<f32>],
    requested: usize,
    backend: BackendChoice,
    source: KnnSource,
) -> Result<(Snapshot, Duration), String> {
    let start = Instant::now();
    let fitted = fit_knn_graph_with_source(data, campaign_config(requested), backend, source)
        .map_err(|error| error.to_string())?;
    let elapsed = start.elapsed();
    Ok((Snapshot::from_fit(&fitted)?, elapsed))
}

fn check_graph_shape(
    label: &str,
    diagnostics: &Snapshot,
    rows: usize,
    effective_k: usize,
    expected_source: KnnNeighborSource,
    failures: &mut Vec<String>,
) {
    if diagnostics.k != effective_k {
        failures.push(format!(
            "{label}: width {} != effective {effective_k}",
            diagnostics.k
        ));
    }
    if diagnostics.neighbors.len() != rows * effective_k {
        failures.push(format!(
            "{label}: neighbor length {}",
            diagnostics.neighbors.len()
        ));
    }
    if diagnostics.source != expected_source {
        failures.push(format!(
            "{label}: source {:?} != {expected_source:?}",
            diagnostics.source
        ));
    }
    for (slot, &id) in diagnostics.neighbors.iter().enumerate() {
        let row = slot / effective_k.max(1);
        if id < 0 {
            failures.push(format!("{label}: padding at row {row}"));
            break;
        }
        if id as usize >= rows || id as usize == row {
            failures.push(format!("{label}: invalid id {id} at row {row}"));
            break;
        }
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
}

// ---------- campaign runner ----------

fn campaign_dir() -> PathBuf {
    let dir = std::env::var_os(CAMPAIGN_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{CAMPAIGN_DIR_ENV} must name the campaign output directory"));
    fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("create {}: {error}", dir.display()));
    dir
}

fn relative_error(exact: f64, ann: f64) -> f64 {
    if exact == 0.0 {
        if ann == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (ann - exact).abs() / exact.abs()
    }
}

/// Runs every (dataset, seed, requested k) case, writes `<backend>.json`, and returns all failures.
/// The JSON is written before any gate is asserted by the caller.
fn run_backend(backend: BackendChoice, backend_name: &str, dir: &Path) -> Vec<String> {
    let mut failures = Vec::new();
    let mut records = Vec::new();
    // (dataset index, requested index) -> per-seed recall, None when the case failed.
    let mut recalls: BTreeMap<(usize, usize), Vec<Option<f64>>> = BTreeMap::new();

    for (dataset_index, &(dataset, dims)) in DATASETS.iter().enumerate() {
        for &seed in &SEEDS {
            let data = make_dataset(dims, seed);
            let n = data.len();
            let widths: Vec<usize> = REQUESTED_K.iter().map(|&k| effective_width(k, n)).collect();
            let cutoffs = exact_cutoffs(&data, &widths);

            for (requested_index, &requested) in REQUESTED_K.iter().enumerate() {
                let effective = widths[requested_index];
                let case = format!("{backend_name} {dataset} seed={seed} requested_k={requested}");
                let mut case_failures = Vec::new();

                let exact = match timed_fit(&data, requested, backend, KnnSource::Exact) {
                    Ok((snapshot, _)) => Some(snapshot),
                    Err(error) => {
                        case_failures.push(format!("{case}: exact fit failed: {error}"));
                        None
                    }
                };
                if let Some(exact) = &exact {
                    check_graph_shape(
                        &format!("{case} exact"),
                        exact,
                        n,
                        effective,
                        KnnNeighborSource::Exact,
                        &mut case_failures,
                    );
                }

                let mut ann_runs: Vec<(Snapshot, Duration)> = Vec::new();
                let mut ann_error = None;
                for run in 0..WARMUP_FITS + TIMED_FITS {
                    match timed_fit(&data, requested, backend, KnnSource::Ann) {
                        Ok(fit) => ann_runs.push(fit),
                        Err(error) => {
                            ann_error = Some(format!("{case}: ANN fit {run} failed: {error}"));
                            break;
                        }
                    }
                }
                if let Some(error) = ann_error {
                    case_failures.push(error);
                }

                let mut record = json!({
                    "dataset": dataset,
                    "dims": dims,
                    "seed": seed,
                    "requested_k": requested,
                    "effective_k": effective,
                    "min_samples": MIN_SAMPLES,
                    "rows": n,
                });

                let complete = exact.is_some() && ann_runs.len() == WARMUP_FITS + TIMED_FITS;
                if complete {
                    let exact = exact.as_ref().expect("checked complete");
                    let (warmup_and_timed, timed) = ann_runs.split_at(WARMUP_FITS);
                    let reference = &timed[0].0;
                    check_graph_shape(
                        &format!("{case} ann"),
                        reference,
                        n,
                        effective,
                        KnnNeighborSource::Approximate,
                        &mut case_failures,
                    );

                    let repeat_identical = warmup_and_timed
                        .iter()
                        .chain(timed[1..].iter())
                        .all(|(snapshot, _)| snapshot.same_outputs(reference));
                    if !repeat_identical {
                        case_failures.push(format!("{case}: repeated ANN fits differ"));
                    }

                    let recall = tie_aware_recall(
                        &reference.neighbors,
                        &data,
                        effective,
                        &cutoffs[requested_index],
                    );
                    let ari = adjusted_rand_index(&reference.labels, &exact.labels);
                    let sum_error = relative_error(exact.finite_sum, reference.finite_sum);
                    if !ari.is_finite() || ari < ARI_LIMIT {
                        case_failures.push(format!("{case}: ARI {ari:.6} < {ARI_LIMIT}"));
                    }
                    if reference.bridges != exact.bridges {
                        case_failures.push(format!(
                            "{case}: bridges {} != exact {}",
                            reference.bridges, exact.bridges
                        ));
                    }
                    if !sum_error.is_finite() || sum_error > SUM_RELATIVE_LIMIT {
                        case_failures.push(format!(
                            "{case}: finite-sum relative error {sum_error:.6} > {SUM_RELATIVE_LIMIT}"
                        ));
                    }

                    let walltimes: Vec<f64> = timed
                        .iter()
                        .map(|(_, elapsed)| elapsed.as_secs_f64() * 1000.0)
                        .collect();
                    let warmup_ms = warmup_and_timed[0].1.as_secs_f64() * 1000.0;
                    let median_ms = median(&mut walltimes.clone());

                    recalls
                        .entry((dataset_index, requested_index))
                        .or_default()
                        .push(Some(recall));
                    record["recall"] = json!(recall);
                    record["ari"] = json!(ari);
                    record["exact_bridges"] = json!(exact.bridges);
                    record["ann_bridges"] = json!(reference.bridges);
                    record["exact_finite_weight_sum"] = json!(exact.finite_sum);
                    record["ann_finite_weight_sum"] = json!(reference.finite_sum);
                    record["finite_weight_relative_error"] = json!(sum_error);
                    record["exact_allocation_bytes"] = json!(exact.allocation_bytes);
                    record["ann_allocation_bytes"] = json!(reference.allocation_bytes);
                    record["ann_source"] = json!(format!("{:?}", reference.source));
                    record["repeat_identical"] = json!(repeat_identical);
                    record["warmup_walltime_ms"] = json!(warmup_ms);
                    record["walltimes_ms"] = json!(walltimes);
                    record["median_walltime_ms"] = json!(median_ms);
                    record["edge_hash"] = json!(format!("{:016x}", reference.edge_hash()));
                    record["exact_edge_hash"] = json!(format!("{:016x}", exact.edge_hash()));
                    record["labels"] = json!(reference.labels);
                    record["failures"] = json!(case_failures);
                } else {
                    recalls
                        .entry((dataset_index, requested_index))
                        .or_default()
                        .push(None);
                    record["failures"] = json!(case_failures);
                }
                failures.extend(case_failures);
                records.push(record);
            }
        }
    }

    let mut width_gates = Vec::new();
    for (&(dataset_index, requested_index), values) in &recalls {
        let (dataset, _) = DATASETS[dataset_index];
        let requested = REQUESTED_K[requested_index];
        let label = format!("{backend_name} {dataset} requested_k={requested}");
        let mean = if values.iter().all(Option::is_some) {
            Some(values.iter().flatten().sum::<f64>() / values.len() as f64)
        } else {
            None
        };
        let passed = mean.is_some_and(|m| m >= RECALL_LIMIT);
        if !passed {
            failures.push(format!("{label}: mean recall {mean:?} < {RECALL_LIMIT}"));
        }
        width_gates.push(json!({
            "dataset": dataset,
            "requested_k": requested,
            "effective_k": effective_width(requested, ROWS),
            "mean_recall": mean,
            "passed": passed,
        }));
    }

    let root = json!({
        "schema": SCHEMA,
        "backend": backend_name,
        "backend_choice": format!("{backend:?}"),
        "rows": ROWS,
        "min_samples": MIN_SAMPLES,
        "requested_k": REQUESTED_K,
        "seeds": SEEDS,
        "datasets": DATASETS.iter().map(|(name, dims)| json!({"name": name, "dims": dims})).collect::<Vec<_>>(),
        "limits": {
            "recall_mean": RECALL_LIMIT,
            "ari": ARI_LIMIT,
            "finite_sum_relative_error": SUM_RELATIVE_LIMIT,
        },
        "warmup_fits": WARMUP_FITS,
        "timed_fits": TIMED_FITS,
        "width_gates": width_gates,
        "records": records,
        "failures": failures,
        "passed": failures.is_empty(),
    });
    let path = dir.join(format!("{backend_name}.json"));
    fs::write(
        &path,
        serde_json::to_string_pretty(&root).expect("serialize campaign"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", path.display()));
    failures
}

#[test]
#[ignore = "hardware campaign; run with FAST_UMAP_CAMPAIGN_DIR set"]
fn campaign_cpu() {
    let dir = campaign_dir();
    let failures = run_backend(BackendChoice::Cpu, "cpu", &dir);
    assert!(
        failures.is_empty(),
        "CPU ANN width campaign failed {} gate(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
#[ignore = "hardware campaign; CUDA must be present, never skipped"]
fn campaign_cuda() {
    let backend = BackendChoice::Cuda { device_index: 0 };
    if let Err(error) = selected_device(backend) {
        panic!("CUDA campaign requires a CUDA device: {error}");
    }
    let dir = campaign_dir();
    let failures = run_backend(backend, "cuda", &dir);
    assert!(
        failures.is_empty(),
        "CUDA ANN width campaign failed {} gate(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ---------- paired comparison ----------

fn load_campaign(dir: &Path, name: &str) -> Value {
    let path = dir.join(name);
    let text = fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "{} missing or unreadable ({error}); run the matching campaign first",
            path.display()
        )
    });
    serde_json::from_str(&text).unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn keyed_records(
    root: &Value,
    name: &str,
    failures: &mut Vec<String>,
) -> BTreeMap<(String, u64, u64), Value> {
    let mut map = BTreeMap::new();
    for record in root["records"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        let key = (
            record["dataset"].as_str().unwrap_or_default().to_string(),
            record["requested_k"].as_u64().unwrap_or(0),
            record["seed"].as_u64().unwrap_or(0),
        );
        if map.insert(key.clone(), record.clone()).is_some() {
            failures.push(format!("{name}: duplicate record {key:?}"));
        }
    }
    if map.len() != PAIR_RECORDS {
        failures.push(format!(
            "{name}: {} records, expected {PAIR_RECORDS}",
            map.len()
        ));
    }
    map
}

fn labels_of(record: &Value) -> Option<Vec<i32>> {
    record["labels"]
        .as_array()?
        .iter()
        .map(|v| v.as_i64().and_then(|x| i32::try_from(x).ok()))
        .collect()
}

#[test]
#[ignore = "requires cpu.json and cuda.json from the campaigns"]
fn compare_campaign_pairs() {
    let dir = campaign_dir();
    let cpu = load_campaign(&dir, "cpu.json");
    let cuda = load_campaign(&dir, "cuda.json");
    let mut failures = Vec::new();
    let cpu_records = keyed_records(&cpu, "cpu.json", &mut failures);
    let cuda_records = keyed_records(&cuda, "cuda.json", &mut failures);

    for key in cpu_records.keys().chain(cuda_records.keys()) {
        if !(cpu_records.contains_key(key) && cuda_records.contains_key(key)) {
            failures.push(format!("record {key:?} is not present on both backends"));
        }
    }

    for (key, cpu_record) in &cpu_records {
        let Some(cuda_record) = cuda_records.get(key) else {
            continue;
        };
        let label = format!("{key:?}");
        let mut pair_ok = true;
        for (backend, record) in [("cpu", cpu_record), ("cuda", cuda_record)] {
            let case_failures = record["failures"].as_array().map_or(0, Vec::len);
            if case_failures != 0 {
                failures.push(format!(
                    "{label} {backend}: record carries {case_failures} failure(s)"
                ));
                pair_ok = false;
                continue;
            }
            let exact_bridges = record["exact_bridges"].as_u64();
            let ann_bridges = record["ann_bridges"].as_u64();
            if exact_bridges.is_none() || exact_bridges != ann_bridges {
                failures.push(format!(
                    "{label} {backend}: bridges {ann_bridges:?} != exact {exact_bridges:?}"
                ));
            }
            let exact_sum = record["exact_finite_weight_sum"]
                .as_f64()
                .unwrap_or(f64::NAN);
            let ann_sum = record["ann_finite_weight_sum"].as_f64().unwrap_or(f64::NAN);
            let error = relative_error(exact_sum, ann_sum);
            if !error.is_finite() || error > SUM_RELATIVE_LIMIT {
                failures.push(format!(
                    "{label} {backend}: finite-sum error {error} > {SUM_RELATIVE_LIMIT}"
                ));
            }
        }
        if !pair_ok {
            continue;
        }
        match (labels_of(cpu_record), labels_of(cuda_record)) {
            (Some(cpu_labels), Some(cuda_labels)) => {
                let ari = adjusted_rand_index(&cpu_labels, &cuda_labels);
                if !ari.is_finite() || ari < ARI_LIMIT {
                    failures.push(format!("{label}: cpu/cuda ARI {ari:.6} < {ARI_LIMIT}"));
                }
            }
            _ => failures.push(format!("{label}: labels missing on one backend")),
        }
    }

    assert!(
        failures.is_empty(),
        "campaign pair comparison failed {} gate(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
