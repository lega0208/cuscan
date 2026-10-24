//! ANN width campaign, schema v2.
//!
//! Hardware campaigns (`#[ignore]`) run the families selected by `FAST_UMAP_CAMPAIGN_FAMILIES`
//! (comma-separated; default all) in matrix order: baseline gates first, the bounded high-density
//! probe last. Each invocation merges into `<backend>.json` under `FAST_UMAP_CAMPAIGN_DIR` and
//! atomically rewrites that file after every case, so earlier evidence survives a later
//! resource or runtime failure. Matrix cases no invocation has produced are stored as
//! `status: "omitted"` with a reason. An omitted asserted case fails `compare_campaign_pairs`.
//! An omitted characterization case is reported and never counted as passed.
//!
//! Gate classes:
//! - `asserted` (families `blobs50`, `blobs40`): original and expanded width gates. Mean recall per
//!   dataset/width, per-case ARI, bridges, finite-sum error, structure and repeatability.
//! - `characterization` (other families): structure and repeatability are asserted. ARI, bridges and
//!   finite-sum misses are recorded as `characterization_miss` and require documented disposition.
//!
//! Environment:
//! - `FAST_UMAP_CAMPAIGN_DIR` (required): output directory. Use a fresh directory for schema v2.
//! - `FAST_UMAP_CAMPAIGN_FAMILIES`: comma-separated family names; unknown names panic.
//! - `FAST_UMAP_CAMPAIGN_OMIT_REASON`: reason recorded for matrix cases not run by this invocation.
//! - `FAST_UMAP_CAMPAIGN_REVISION`: revision string recorded in the metadata.
//! - `FAST_UMAP_CAMPAIGN_FAST=1`: bounded budget, applied only to characterization families.
//!   Asserted families and the default runner are unchanged. Completed records are never rerun in
//!   fast mode. Unselected characterization seeds become `omitted` placeholders with the approved
//!   fast-budget reason (never `passed`).
//!   Policy: seed 42 only; one ANN fit per case; one extra repeat fit at the largest requested
//!   width <= 1024 (high-density 4096 is not repeated, 1024 is used instead); single-fit walltimes
//!   with no warmup and no median claim. Large 12000-row widths 60/120/256 are all kept.
//!   Command (leader): `FAST_UMAP_CAMPAIGN_FAST=1 FAST_UMAP_CAMPAIGN_FAMILIES=<family> cargo test
//!   --release -p fast-hdbscan --test test_knn_width_campaign campaign_cpu -- --ignored --nocapture`
//!   (and `campaign_cuda`), then `compare_campaign_pairs`.
//!
//! Sequential runner commands (release, real hardware; run in this order):
//! ```text
//! export FAST_UMAP_CAMPAIGN_DIR=<fresh-dir>
//! FAST_UMAP_CAMPAIGN_FAMILIES=blobs50,blobs40 cargo test --release -p fast-hdbscan --test test_knn_width_campaign campaign_cpu -- --ignored --nocapture
//! FAST_UMAP_CAMPAIGN_FAMILIES=blobs50,blobs40 cargo test --release -p fast-hdbscan --test test_knn_width_campaign campaign_cuda -- --ignored --nocapture
//! FAST_UMAP_CAMPAIGN_FAMILIES=less_separated40,anisotropic40 ... (same two commands)
//! FAST_UMAP_CAMPAIGN_FAMILIES=large_rows50 ... (same two commands)
//! FAST_UMAP_CAMPAIGN_FAMILIES=high_density31 ... (same two commands, last)
//! cargo test --release -p fast-hdbscan --test test_knn_width_campaign compare_campaign_pairs -- --ignored --nocapture
//! ```
//! The ARI and comparator unit tests are ordinary and need no hardware.

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use fast_cubecl_common::{neighbors::ann_host_estimate, selected_device};
use fast_hdbscan::{
    BackendChoice, FittedHDBSCAN, HDBSCAN, HdbscanConfig, KnnDiagnostics, KnnNeighborSource,
    KnnSource, fit_knn_graph_with_source,
};
use serde_json::{Value, json};

const CAMPAIGN_DIR_ENV: &str = "FAST_UMAP_CAMPAIGN_DIR";
const FAMILIES_ENV: &str = "FAST_UMAP_CAMPAIGN_FAMILIES";
const OMIT_REASON_ENV: &str = "FAST_UMAP_CAMPAIGN_OMIT_REASON";
const REVISION_ENV: &str = "FAST_UMAP_CAMPAIGN_REVISION";
const SCHEMA: &str = "fast-hdbscan.knn-width-campaign.v2";
const MIN_SAMPLES: usize = 25;
/// Mirrors the estimator's ANN seed and iteration policy (not exported by the public API).
const ANN_SEED: u64 = 42;
const CENTRES: usize = 8;
const ORIGINAL_ROWS: usize = 6000;
const SEEDS: [u64; 3] = [42, 43, 44];
const RECALL_LIMIT: f64 = 0.90;
const ARI_LIMIT: f64 = 0.95;
const SUM_RELATIVE_LIMIT: f64 = 0.01;
const WARMUP_FITS: usize = 1;
const TIMED_FITS: usize = 3;
const FAST_ENV: &str = "FAST_UMAP_CAMPAIGN_FAST";
/// Fast mode repeats the largest requested width at or below this cap. High-density 4096 is excluded.
const FAST_REPEAT_MAX_WIDTH: usize = 1024;
const FAST_POLICY: &str = "FAST_UMAP_CAMPAIGN_FAST=1 applies to characterization families only: seed 42 only; one ANN fit per case; one extra repeat at the largest requested width <= 1024; single-fit walltimes, no warmup, no median; completed records are not rerun";
const FAST_OMIT_REASON: &str =
    "approved FAST budget (FAST_UMAP_CAMPAIGN_FAST=1): characterization seed not run; not a pass";

// ---------- campaign matrix ----------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Gate {
    Asserted,
    Characterization,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Shape {
    Blobs,
    LessSeparated,
    Anisotropic,
}

struct Family {
    name: &'static str,
    gate: Gate,
    shape: Shape,
    rows: usize,
    dims: usize,
    requested: &'static [usize],
    seeds: &'static [u64],
    /// Warmup plus timed repeats. Single fit when false (bounded high-density cells).
    timed: bool,
    /// Also run Auto and compare it with the forced source it resolves to.
    auto_probe: bool,
}

/// Matrix order is execution order: baseline gates first, bounded high-density probe last.
static FAMILIES: [Family; 6] = [
    Family {
        name: "blobs50",
        gate: Gate::Asserted,
        shape: Shape::Blobs,
        rows: ORIGINAL_ROWS,
        dims: 50,
        requested: &[15, 45, 60, 61, 75, 120, 256, 512],
        seeds: &SEEDS,
        timed: true,
        auto_probe: false,
    },
    Family {
        name: "blobs40",
        gate: Gate::Asserted,
        shape: Shape::Blobs,
        rows: ORIGINAL_ROWS,
        dims: 40,
        requested: &[15, 45, 60, 61, 75, 120, 256, 512],
        seeds: &SEEDS,
        timed: true,
        auto_probe: false,
    },
    Family {
        name: "less_separated40",
        gate: Gate::Characterization,
        shape: Shape::LessSeparated,
        rows: ORIGINAL_ROWS,
        dims: 40,
        requested: &[60, 75, 120, 256],
        seeds: &SEEDS,
        timed: true,
        auto_probe: false,
    },
    Family {
        name: "anisotropic40",
        gate: Gate::Characterization,
        shape: Shape::Anisotropic,
        rows: ORIGINAL_ROWS,
        dims: 40,
        requested: &[60, 75, 120, 256],
        seeds: &SEEDS,
        timed: true,
        auto_probe: false,
    },
    Family {
        name: "large_rows50",
        gate: Gate::Characterization,
        shape: Shape::Blobs,
        rows: 12000,
        dims: 50,
        requested: &[60, 120, 256],
        seeds: &SEEDS,
        timed: true,
        auto_probe: false,
    },
    Family {
        name: "high_density31",
        gate: Gate::Characterization,
        shape: Shape::Blobs,
        rows: 4097,
        dims: 31,
        requested: &[1024, 4096],
        seeds: &[42],
        timed: false,
        auto_probe: true,
    },
];

fn case_key(family: &Family, seed: u64, requested: usize) -> String {
    format!("{}|seed={seed}|requested_k={requested}", family.name)
}

/// Every matrix case key in execution order.
fn matrix_keys() -> Vec<String> {
    FAMILIES
        .iter()
        .flat_map(|family| {
            family.seeds.iter().flat_map(move |&seed| {
                family
                    .requested
                    .iter()
                    .map(move |&requested| case_key(family, seed, requested))
            })
        })
        .collect()
}

/// Per-case execution plan. Default mode keeps the matrix policy. Fast mode changes only
/// characterization families (see `FAST_POLICY`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CasePlan {
    /// Run this seed under the current mode. Unselected cases get a fast-budget placeholder.
    selected: bool,
    /// Warmup plus timed median fits. False means single unrepeated fits, with no median claim.
    timed: bool,
    /// ANN fits for this case.
    ann_fits: usize,
    /// Fast budget applied to this case.
    fast: bool,
}

fn fast_mode_enabled(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Fast mode never touches asserted families.
fn fast_applies(family: &Family, fast: bool) -> bool {
    fast && family.gate == Gate::Characterization
}

/// Largest requested width at or below `FAST_REPEAT_MAX_WIDTH`; the repeat fit runs there.
fn fast_repeat_width(family: &Family) -> Option<usize> {
    family
        .requested
        .iter()
        .copied()
        .filter(|&k| k <= FAST_REPEAT_MAX_WIDTH)
        .max()
}

fn case_plan(family: &Family, seed: u64, requested: usize, fast: bool) -> CasePlan {
    if !fast_applies(family, fast) {
        let timed = family.timed;
        return CasePlan {
            selected: true,
            timed,
            ann_fits: if timed { WARMUP_FITS + TIMED_FITS } else { 1 },
            fast: false,
        };
    }
    CasePlan {
        selected: seed == family.seeds[0],
        timed: false,
        ann_fits: if fast_repeat_width(family) == Some(requested) {
            2
        } else {
            1
        },
        fast: true,
    }
}

/// A record counts as completed unless it is an omitted placeholder.
fn completed(map: &BTreeMap<String, Value>, key: &str) -> bool {
    map.get(key)
        .is_some_and(|record| record["status"] != "omitted")
}

fn family_named(name: &str) -> &'static Family {
    FAMILIES
        .iter()
        .find(|family| family.name == name)
        .expect("known family")
}

fn gate_name(gate: Gate) -> &'static str {
    match gate {
        Gate::Asserted => "asserted",
        Gate::Characterization => "characterization",
    }
}

fn shape_name(shape: Shape) -> &'static str {
    match shape {
        Shape::Blobs => "blobs",
        Shape::LessSeparated => "less_separated",
        Shape::Anisotropic => "anisotropic",
    }
}

fn build_mode() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn revision() -> String {
    std::env::var(REVISION_ENV).unwrap_or_else(|_| "unrecorded".to_string())
}

/// Mirrors the estimator's iteration count `max(5, round(log2 n))` (not exported).
fn ann_iterations(n: usize) -> u32 {
    (n as f64).log2().round().max(5.0) as u32
}

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

/// Eight uniform centres in [-10, 10]^d; row i belongs to centre i % 8.
/// Blobs use unit noise and reproduce the v1 recipe. Less-separated uses noise 2.5.
/// Anisotropic rescales axes and adds a nonlinear coupling between neighbouring axes.
fn make_dataset(family: &Family, seed: u64) -> Vec<Vec<f32>> {
    let dims = family.dims;
    let spread = if family.shape == Shape::LessSeparated {
        2.5
    } else {
        1.0
    };
    let mut rng = SplitMix64(seed ^ ((dims as u64) << 32));
    let centres: Vec<Vec<f64>> = (0..CENTRES)
        .map(|_| (0..dims).map(|_| rng.uniform(-10.0, 10.0)).collect())
        .collect();
    (0..family.rows)
        .map(|row| {
            let centre = &centres[row % CENTRES];
            let mut point: Vec<f64> = (0..dims)
                .map(|d| centre[d] + spread * rng.normal())
                .collect();
            if family.shape == Shape::Anisotropic {
                for (d, value) in point.iter_mut().enumerate() {
                    *value *= 0.25 * (1 + d % 7) as f64;
                }
                for d in 1..dims {
                    point[d] += 0.5 * point[d - 1].sin();
                }
            }
            point.into_iter().map(|x| x as f32).collect()
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

/// Exact kth non-self squared-distance cutoff per row, for each width.
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

fn relative_error(exact: f64, ann: f64) -> f64 {
    if exact == 0.0 {
        if ann == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (ann - exact).abs() / exact.abs()
    }
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    values[values.len() / 2]
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

/// Auto selection with no forced source; the resolved source is read back from diagnostics.
fn timed_auto_fit(
    data: &[Vec<f32>],
    requested: usize,
    backend: BackendChoice,
) -> Result<(Snapshot, Duration), String> {
    let start = Instant::now();
    let fitted = HDBSCAN::new(campaign_config(requested), backend)
        .fit(data)
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

// ---------- one case ----------

/// Runs one (family, seed, requested k) case. Structural and repeat failures always count.
/// Backend identity shared by every case of one invocation.
struct BackendRun<'a> {
    choice: BackendChoice,
    name: &'a str,
    resolved: String,
}

/// Quality metrics count only for asserted families; characterization misses are recorded.
fn run_case(
    family: &Family,
    seed: u64,
    requested: usize,
    data: &[Vec<f32>],
    cutoffs: &[f64],
    run: &BackendRun,
    plan: CasePlan,
) -> Value {
    let backend = run.choice;
    let backend_name = run.name;
    let resolved_backend = run.resolved.as_str();
    let n = data.len();
    let effective = effective_width(requested, n);
    let case = format!(
        "{backend_name} {} seed={seed} requested_k={requested}",
        family.name
    );
    // Shape-only, advisory estimate from the public shared ANN contract at the effective width.
    let estimate = ann_host_estimate(n, family.dims, effective).ok();
    let mut record = json!({
        "key": case_key(family, seed, requested),
        "family": family.name,
        "gate": gate_name(family.gate),
        "shape": shape_name(family.shape),
        "dataset": family.name,
        "dims": family.dims,
        "rows": n,
        "seed": seed,
        "requested_k": requested,
        "effective_k": effective,
        "retained_width": estimate.map(|e| e.retained_width),
        "min_samples": MIN_SAMPLES,
        "sample_capacity": estimate.map(|e| e.sample_capacity),
        "ann_trees": estimate.map(|e| e.trees),
        "ann_phase_bytes": estimate.map(|e| json!(e.phases.iter().map(|p| json!({
            "phase": format!("{:?}", p.phase),
            "host_bytes": p.host_bytes,
            "runtime_bytes": p.runtime_bytes,
        })).collect::<Vec<_>>())),
        "ann_host_peak_bytes": estimate.map(|e| e.host_peak_bytes),
        "ann_runtime_peak_bytes": estimate.map(|e| e.runtime_peak_bytes),
        "ann_cpu_combined_peak_bytes": estimate.map(|e| e.cpu_combined_peak_bytes),
        "estimate_note": "ann_host_estimate: shape-only and advisory; not measured allocator peaks",
        "resolved_backend": resolved_backend,
        "ann_iterations": ann_iterations(n),
        "ann_seed": ANN_SEED,
        "backend": backend_name,
        "backend_choice": format!("{backend:?}"),
        "build_mode": build_mode(),
        "revision": revision(),
        "timed": plan.timed,
        "ann_fits": plan.ann_fits,
        "fast_mode": plan.fast,
        "stop_iteration": null,
        "final_new_count": null,
        "convergence_note": "not exported by fit diagnostics",
    });
    let mut structural = Vec::new();
    let mut gate_fails = Vec::new();
    let mut misses = Vec::new();
    if estimate.is_none() {
        structural.push(format!("{case}: ann_host_estimate rejected the shape"));
    }

    let exact = match timed_fit(data, requested, backend, KnnSource::Exact) {
        Ok((snapshot, _)) => Some(snapshot),
        Err(error) => {
            structural.push(format!("{case}: exact fit failed: {error}"));
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
            &mut structural,
        );
    }

    let runs = plan.ann_fits;
    let mut ann_runs: Vec<(Snapshot, Duration)> = Vec::new();
    for run in 0..runs {
        match timed_fit(data, requested, backend, KnnSource::Ann) {
            Ok(fit) => ann_runs.push(fit),
            Err(error) => {
                structural.push(format!("{case}: ANN fit {run} failed: {error}"));
                break;
            }
        }
    }

    if let (Some(exact), true) = (&exact, ann_runs.len() == runs) {
        let timed_start = if plan.timed { WARMUP_FITS } else { 0 };
        let reference = &ann_runs[timed_start].0;
        check_graph_shape(
            &format!("{case} ann"),
            reference,
            n,
            effective,
            KnnNeighborSource::Approximate,
            &mut structural,
        );
        let repeat_identical = ann_runs
            .iter()
            .all(|(snapshot, _)| snapshot.same_outputs(reference));
        if ann_runs.len() > 1 && !repeat_identical {
            structural.push(format!("{case}: repeated ANN fits differ"));
        }

        let recall = tie_aware_recall(&reference.neighbors, data, effective, cutoffs);
        let ari = adjusted_rand_index(&reference.labels, &exact.labels);
        let sum_error = relative_error(exact.finite_sum, reference.finite_sum);
        let quality: Vec<String> = [
            (!(ari.is_finite() && ari >= ARI_LIMIT))
                .then(|| format!("{case}: ARI {ari:.6} < {ARI_LIMIT}")),
            (reference.bridges != exact.bridges).then(|| {
                format!(
                    "{case}: bridges {} != exact {}",
                    reference.bridges, exact.bridges
                )
            }),
            (!(sum_error.is_finite() && sum_error <= SUM_RELATIVE_LIMIT)).then(|| {
                format!("{case}: finite-sum relative error {sum_error:.6} > {SUM_RELATIVE_LIMIT}")
            }),
        ]
        .into_iter()
        .flatten()
        .collect();
        let recall_miss = (recall < RECALL_LIMIT)
            .then(|| format!("{case}: per-case recall {recall:.6} < {RECALL_LIMIT}"));
        if family.gate == Gate::Asserted {
            gate_fails.extend(quality);
        } else {
            misses.extend(quality);
            misses.extend(recall_miss);
        }

        // Post-cutoff: Auto above 4096 rows must resolve to Approximate and match forced ANN.
        let auto = if family.auto_probe {
            match timed_auto_fit(data, requested, backend) {
                Ok((snapshot, elapsed)) => {
                    check_graph_shape(
                        &format!("{case} auto"),
                        &snapshot,
                        n,
                        effective,
                        snapshot.source,
                        &mut structural,
                    );
                    let matches_resolved = snapshot.source == KnnNeighborSource::Approximate
                        && snapshot.same_outputs(reference);
                    if !matches_resolved {
                        structural.push(format!(
                            "{case}: Auto must resolve to Approximate and match forced ANN"
                        ));
                    }
                    Some(json!({
                        "source": format!("{:?}", snapshot.source),
                        "walltime_ms": elapsed.as_secs_f64() * 1000.0,
                        "bridges": snapshot.bridges,
                        "matches_resolved_forced": matches_resolved,
                        "edge_hash": format!("{:016x}", snapshot.edge_hash()),
                    }))
                }
                Err(error) => {
                    structural.push(format!("{case}: Auto fit failed: {error}"));
                    None
                }
            }
        } else {
            None
        };

        let walltimes: Vec<f64> = ann_runs[timed_start..]
            .iter()
            .map(|(_, elapsed)| elapsed.as_secs_f64() * 1000.0)
            .collect();
        let median_ms = plan.timed.then(|| median(&mut walltimes.clone()));
        record["recall"] = json!(recall);
        record["recall_threshold_met"] = json!(recall >= RECALL_LIMIT);
        record["ari"] = json!(ari);
        record["exact_bridges"] = json!(exact.bridges);
        record["ann_bridges"] = json!(reference.bridges);
        record["exact_finite_weight_sum"] = json!(exact.finite_sum);
        record["ann_finite_weight_sum"] = json!(reference.finite_sum);
        record["finite_weight_relative_error"] = json!(sum_error);
        record["exact_allocation_bytes"] = json!(exact.allocation_bytes);
        record["ann_allocation_bytes"] = json!(reference.allocation_bytes);
        record["ann_source"] = json!(format!("{:?}", reference.source));
        record["repeat_identical"] = if ann_runs.len() > 1 {
            json!(repeat_identical)
        } else {
            Value::Null
        };
        record["fast_policy"] = if plan.fast {
            json!(FAST_POLICY)
        } else {
            Value::Null
        };
        record["walltime_policy"] = json!(if plan.timed {
            "warmup_excluded_median_of_timed_fits"
        } else {
            "single_fit_no_median"
        });
        record["warmup_walltime_ms"] = if plan.timed {
            json!(ann_runs[0].1.as_secs_f64() * 1000.0)
        } else {
            Value::Null
        };
        record["walltimes_ms"] = json!(walltimes);
        record["median_walltime_ms"] = json!(median_ms);
        record["edge_hash"] = json!(format!("{:016x}", reference.edge_hash()));
        record["exact_edge_hash"] = json!(format!("{:016x}", exact.edge_hash()));
        record["labels"] = json!(reference.labels);
        record["auto_probe"] = auto.unwrap_or(Value::Null);
    }

    let status = if !structural.is_empty() || !gate_fails.is_empty() {
        "failed"
    } else if !misses.is_empty() {
        "characterization_miss"
    } else {
        "passed"
    };
    record["status"] = json!(status);
    record["structural_failures"] = json!(structural);
    record["gate_failures"] = json!(gate_fails);
    record["characterization_misses"] = json!(misses);
    record
}

fn omitted_record(
    family: &Family,
    seed: u64,
    requested: usize,
    backend_name: &str,
    reason: &str,
) -> Value {
    json!({
        "key": case_key(family, seed, requested),
        "family": family.name,
        "gate": gate_name(family.gate),
        "dataset": family.name,
        "rows": family.rows,
        "seed": seed,
        "requested_k": requested,
        "effective_k": effective_width(requested, family.rows),
        "backend": backend_name,
        "status": "omitted",
        "reason": reason,
        "structural_failures": [],
        "gate_failures": [],
        "characterization_misses": [],
    })
}

// ---------- checkpointed runner ----------

fn campaign_dir() -> PathBuf {
    let dir = std::env::var_os(CAMPAIGN_DIR_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("{CAMPAIGN_DIR_ENV} must name the campaign output directory"));
    fs::create_dir_all(&dir).unwrap_or_else(|error| panic!("create {}: {error}", dir.display()));
    dir
}

/// Backend resolution as reported by `selected_device`. The crate exposes no device name.
fn resolved_backend_name(backend: BackendChoice) -> String {
    let (_device, resolved) = selected_device(backend)
        .unwrap_or_else(|error| panic!("backend resolution failed: {error}"));
    format!("{resolved:?}")
}

fn selected_families() -> Vec<&'static str> {
    let Ok(list) = std::env::var(FAMILIES_ENV) else {
        return FAMILIES.iter().map(|family| family.name).collect();
    };
    list.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(|name| {
            FAMILIES
                .iter()
                .find(|family| family.name == name)
                .unwrap_or_else(|| panic!("unknown family {name} in {FAMILIES_ENV}"))
                .name
        })
        .collect()
}

/// Loads records from an existing campaign file so later invocations merge rather than erase.
fn load_existing(path: &Path) -> BTreeMap<String, Value> {
    let Ok(text) = fs::read_to_string(path) else {
        return BTreeMap::new();
    };
    let root: Value = serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()));
    assert_eq!(
        root["schema"].as_str(),
        Some(SCHEMA),
        "{} uses another schema; use a fresh {CAMPAIGN_DIR_ENV}",
        path.display()
    );
    root["records"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .map(|record| {
            (
                record["key"].as_str().unwrap_or_default().to_string(),
                record.clone(),
            )
        })
        .collect()
}

/// Width-gate summary for asserted families. Mean recall is null while any seed record is absent
/// or omitted. A gate is a failure only when its mean is complete and below the limit.
fn width_gates(map: &BTreeMap<String, Value>) -> (Vec<Value>, Vec<String>) {
    let mut gates = Vec::new();
    let mut failures = Vec::new();
    for family in FAMILIES.iter() {
        for &requested in family.requested {
            let values: Option<Vec<f64>> = family
                .seeds
                .iter()
                .map(|&seed| {
                    let record = map.get(&case_key(family, seed, requested))?;
                    if record["status"] == "omitted" {
                        return None;
                    }
                    record["recall"].as_f64()
                })
                .collect();
            let mean = values
                .as_ref()
                .map(|values| values.iter().sum::<f64>() / values.len() as f64);
            let passed = mean.is_some_and(|m| m >= RECALL_LIMIT);
            if let Some(mean) = mean
                .filter(|m| *m < RECALL_LIMIT)
                .filter(|_| family.gate == Gate::Asserted)
            {
                failures.push(format!(
                    "{} requested_k={requested}: mean recall {mean:.6} < {RECALL_LIMIT}",
                    family.name
                ));
            }
            gates.push(json!({
                "dataset": family.name,
                "gate": gate_name(family.gate),
                "below_limit": mean.is_some_and(|m| m < RECALL_LIMIT),
                "requested_k": requested,
                "effective_k": effective_width(requested, family.rows),
                "mean_recall": mean,
                "passed": passed,
                "complete": mean.is_some(),
            }));
        }
    }
    (gates, failures)
}

fn family_json(family: &Family) -> Value {
    json!({
        "name": family.name,
        "gate": gate_name(family.gate),
        "shape": shape_name(family.shape),
        "rows": family.rows,
        "dims": family.dims,
        "requested_k": family.requested,
        "seeds": family.seeds,
        "timed": family.timed,
        "auto_probe": family.auto_probe,
    })
}

/// Atomically rewrites the campaign file. Records are emitted in matrix order.
fn checkpoint(
    path: &Path,
    backend_name: &str,
    backend_choice: &str,
    map: &BTreeMap<String, Value>,
    complete: bool,
    fast: bool,
) -> Vec<String> {
    let records: Vec<Value> = matrix_keys()
        .iter()
        .filter_map(|key| map.get(key).cloned())
        .collect();
    let (gates, gate_failures) = width_gates(map);
    let fast_policy = if fast {
        json!(FAST_POLICY)
    } else {
        Value::Null
    };
    let root = json!({
        "schema": SCHEMA,
        "backend": backend_name,
        "backend_choice": backend_choice,
        "build_mode": build_mode(),
        "revision": revision(),
        "min_samples": MIN_SAMPLES,
        "limits": {
            "recall_mean": RECALL_LIMIT,
            "ari": ARI_LIMIT,
            "finite_sum_relative_error": SUM_RELATIVE_LIMIT,
        },
        "warmup_fits": WARMUP_FITS,
        "timed_fits": TIMED_FITS,
        "families": FAMILIES.iter().map(family_json).collect::<Vec<_>>(),
        "complete": complete,
        "fast_mode": fast,
        "fast_policy": fast_policy,
        "width_gates": gates,
        "records": records,
    });
    let tmp = path.with_extension("json.tmp");
    fs::write(
        &tmp,
        serde_json::to_string_pretty(&root).expect("serialize campaign"),
    )
    .unwrap_or_else(|error| panic!("write {}: {error}", tmp.display()));
    fs::rename(&tmp, path).unwrap_or_else(|error| panic!("rename {}: {error}", path.display()));
    gate_failures
}

/// Runs the selected families for one backend. The file is checkpointed after every case.
/// Returns asserted failures from this invocation's cases and from complete width gates.
fn run_backend(backend: BackendChoice, backend_name: &str, dir: &Path) -> Vec<String> {
    let path = dir.join(format!("{backend_name}.json"));
    let mut map = load_existing(&path);
    let selected = selected_families();
    let fast = fast_mode_enabled(std::env::var(FAST_ENV).ok().as_deref());
    let backend_choice = format!("{backend:?}");
    let run = BackendRun {
        choice: backend,
        name: backend_name,
        resolved: resolved_backend_name(backend),
    };
    let mut failures = Vec::new();

    for family in FAMILIES
        .iter()
        .filter(|family| selected.contains(&family.name))
    {
        for &seed in family.seeds {
            // Pending cases follow the mode plan. Completed records are never rerun in fast mode.
            let pending: Vec<(usize, usize, CasePlan)> = family
                .requested
                .iter()
                .enumerate()
                .map(|(index, &requested)| {
                    (index, requested, case_plan(family, seed, requested, fast))
                })
                .filter(|(_, requested, plan)| {
                    plan.selected && !(fast && completed(&map, &case_key(family, seed, *requested)))
                })
                .collect();
            if pending.is_empty() {
                continue;
            }
            let data = make_dataset(family, seed);
            let n = data.len();
            let widths: Vec<usize> = family
                .requested
                .iter()
                .map(|&k| effective_width(k, n))
                .collect();
            let cutoffs = exact_cutoffs(&data, &widths);
            for (index, requested, plan) in pending {
                let record = run_case(family, seed, requested, &data, &cutoffs[index], &run, plan);
                for field in ["structural_failures", "gate_failures"] {
                    for message in record[field].as_array().map(Vec::as_slice).unwrap_or(&[]) {
                        failures.push(message.as_str().unwrap_or_default().to_string());
                    }
                }
                map.insert(case_key(family, seed, requested), record);
                checkpoint(&path, backend_name, &backend_choice, &map, false, fast);
            }
        }
    }

    let reason =
        std::env::var(OMIT_REASON_ENV).unwrap_or_else(|_| "not run in this invocation".to_string());
    for family in FAMILIES.iter() {
        for &seed in family.seeds {
            for &requested in family.requested {
                let key = case_key(family, seed, requested);
                if case_plan(family, seed, requested, fast).selected {
                    map.entry(key).or_insert_with(|| {
                        omitted_record(family, seed, requested, backend_name, &reason)
                    });
                } else if !completed(&map, &key) {
                    // Fast-budget placeholder; never replaces a completed record.
                    map.insert(
                        key,
                        omitted_record(family, seed, requested, backend_name, FAST_OMIT_REASON),
                    );
                }
            }
        }
    }
    let complete = map.values().all(|record| record["status"] != "omitted");
    failures.extend(checkpoint(
        &path,
        backend_name,
        &backend_choice,
        &map,
        complete,
        fast,
    ));
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

fn index_records(root: &Value) -> BTreeMap<String, Value> {
    root["records"]
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(&[])
        .iter()
        .map(|record| {
            (
                record["key"].as_str().unwrap_or_default().to_string(),
                record.clone(),
            )
        })
        .collect()
}

fn labels_of(record: &Value) -> Option<Vec<i32>> {
    record["labels"]
        .as_array()?
        .iter()
        .map(|v| v.as_i64().and_then(|x| i32::try_from(x).ok()))
        .collect()
}

/// Pure pairing of one CPU and one CUDA campaign root. Every matrix key is visited.
/// - Asserted keys must have a non-omitted record on both backends, or the pair fails.
/// - Characterization keys missing or omitted are reported with disposition `omitted` and never
///   counted as passed. Their misses are reported as `characterization_miss`.
///
/// Returns failures and one summary entry per matrix key.
fn compare_roots(cpu: &Value, cuda: &Value) -> (Vec<String>, Vec<Value>) {
    let mut failures = Vec::new();
    let mut pairs = Vec::new();
    for (name, root) in [("cpu", cpu), ("cuda", cuda)] {
        if root["schema"].as_str() != Some(SCHEMA) {
            failures.push(format!("{name}: schema is not {SCHEMA}"));
        }
    }
    let cpu_map = index_records(cpu);
    let cuda_map = index_records(cuda);

    for family in FAMILIES.iter() {
        let asserted = family.gate == Gate::Asserted;
        for &seed in family.seeds {
            for &requested in family.requested {
                let key = case_key(family, seed, requested);
                let c = cpu_map.get(&key).filter(|r| r["status"] != "omitted");
                let d = cuda_map.get(&key).filter(|r| r["status"] != "omitted");
                let (Some(c), Some(d)) = (c, d) else {
                    if asserted {
                        failures.push(format!(
                            "{key}: mandatory gate record missing or omitted (cpu present={}, cuda present={})",
                            cpu_map.get(&key).is_some_and(|r| r["status"] != "omitted"),
                            cuda_map.get(&key).is_some_and(|r| r["status"] != "omitted"),
                        ));
                    }
                    // Omissions stay visible with their reason. They are never counted as passed.
                    let reason = c
                        .or(cpu_map.get(&key))
                        .or(cuda_map.get(&key))
                        .and_then(|r| r["reason"].as_str())
                        .unwrap_or("no record");
                    pairs.push(json!({
                        "key": key,
                        "gate": gate_name(family.gate),
                        "disposition": "omitted",
                        "omission_reason": reason,
                    }));
                    continue;
                };

                for (name, record) in [("cpu", c), ("cuda", d)] {
                    for message in record["structural_failures"]
                        .as_array()
                        .map(Vec::as_slice)
                        .unwrap_or(&[])
                    {
                        failures.push(format!(
                            "{key} {name}: {}",
                            message.as_str().unwrap_or_default()
                        ));
                    }
                    if asserted {
                        for message in record["gate_failures"]
                            .as_array()
                            .map(Vec::as_slice)
                            .unwrap_or(&[])
                        {
                            failures.push(format!(
                                "{key} {name}: {}",
                                message.as_str().unwrap_or_default()
                            ));
                        }
                    }
                }

                let cross_ari = match (labels_of(c), labels_of(d)) {
                    (Some(a), Some(b)) => Some(adjusted_rand_index(&a, &b)),
                    _ => None,
                };
                let cross_ok = cross_ari.is_some_and(|ari| ari.is_finite() && ari >= ARI_LIMIT);
                if !cross_ok {
                    let message = format!("{key}: cpu/cuda ARI {cross_ari:?} < {ARI_LIMIT}");
                    if asserted {
                        failures.push(message);
                    }
                }
                let disposition = if asserted {
                    "asserted"
                } else if cross_ok
                    && c["status"] != "characterization_miss"
                    && d["status"] != "characterization_miss"
                {
                    "characterization_pass"
                } else {
                    "characterization_miss"
                };
                pairs.push(json!({
                    "key": key,
                    "gate": gate_name(family.gate),
                    "cpu_status": c["status"],
                    "cuda_status": d["status"],
                    "cross_ari": cross_ari,
                    "cross_backend_passed": cross_ok,
                    "disposition": disposition,
                }));
            }
        }
    }

    for (name, root) in [("cpu", cpu), ("cuda", cuda)] {
        for gate in root["width_gates"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or(&[])
        {
            if gate["gate"] == "asserted" && gate["passed"] != true {
                failures.push(format!(
                    "{name} {} requested_k={}: width gate not passed (mean recall {})",
                    gate["dataset"], gate["requested_k"], gate["mean_recall"]
                ));
            }
        }
    }
    (failures, pairs)
}

#[test]
#[ignore = "requires cpu.json and cuda.json from the campaigns"]
fn compare_campaign_pairs() {
    let dir = campaign_dir();
    let cpu = load_campaign(&dir, "cpu.json");
    let cuda = load_campaign(&dir, "cuda.json");
    let (failures, pairs) = compare_roots(&cpu, &cuda);
    let summary = json!({"schema": SCHEMA, "pairs": pairs, "failures": failures});
    fs::write(
        dir.join("pairs.json"),
        serde_json::to_string_pretty(&summary).expect("serialize pairs"),
    )
    .expect("write pairs.json");
    assert!(
        failures.is_empty(),
        "campaign pair comparison failed {} gate(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn fast_mode_is_opt_in_exact_value_and_characterization_only() {
    assert!(fast_mode_enabled(Some("1")));
    assert!(!fast_mode_enabled(None));
    assert!(!fast_mode_enabled(Some("0")));
    assert!(!fast_mode_enabled(Some("true")));
    assert!(!fast_applies(family_named("blobs50"), true));
    assert!(!fast_applies(family_named("blobs40"), true));
    assert!(fast_applies(family_named("less_separated40"), true));
    assert!(!fast_applies(family_named("less_separated40"), false));
}

#[test]
fn default_and_asserted_plans_are_unchanged_by_fast_mode() {
    for family in FAMILIES
        .iter()
        .filter(|family| family.gate == Gate::Asserted)
    {
        for &seed in family.seeds {
            for &requested in family.requested {
                for fast in [false, true] {
                    assert_eq!(
                        case_plan(family, seed, requested, fast),
                        CasePlan {
                            selected: true,
                            timed: true,
                            ann_fits: WARMUP_FITS + TIMED_FITS,
                            fast: false,
                        },
                        "{} seed {seed} k {requested} fast {fast}",
                        family.name
                    );
                }
            }
        }
    }
    assert_eq!(
        case_plan(family_named("less_separated40"), 43, 60, false),
        CasePlan {
            selected: true,
            timed: true,
            ann_fits: WARMUP_FITS + TIMED_FITS,
            fast: false,
        }
    );
    assert_eq!(
        case_plan(family_named("high_density31"), 42, 4096, false),
        CasePlan {
            selected: true,
            timed: false,
            ann_fits: 1,
            fast: false,
        }
    );
}

#[test]
fn fast_plan_selects_seed42_single_fit_with_one_repeat_width() {
    for name in ["less_separated40", "anisotropic40", "large_rows50"] {
        let family = family_named(name);
        assert_eq!(fast_repeat_width(family), Some(256), "{name}");
        for &requested in family.requested {
            let seed42 = case_plan(family, 42, requested, true);
            assert!(
                seed42.selected && seed42.fast && !seed42.timed,
                "{name} k {requested}"
            );
            assert_eq!(
                seed42.ann_fits,
                if requested == 256 { 2 } else { 1 },
                "{name} k {requested}"
            );
            assert!(!case_plan(family, 43, requested, true).selected);
            assert!(!case_plan(family, 44, requested, true).selected);
        }
    }
    let large = family_named("large_rows50");
    assert_eq!(large.rows, 12000);
    assert_eq!(large.requested, &[60, 120, 256][..]);
}

#[test]
fn fast_high_density_repeats_1024_not_4096() {
    let high = family_named("high_density31");
    assert_eq!(fast_repeat_width(high), Some(1024));
    let at_1024 = case_plan(high, 42, 1024, true);
    assert!(at_1024.selected && !at_1024.timed && at_1024.ann_fits == 2);
    let at_4096 = case_plan(high, 42, 4096, true);
    assert!(at_4096.selected && at_4096.ann_fits == 1);
}

#[test]
fn completed_records_are_not_reported_as_omitted() {
    let mut map = BTreeMap::new();
    map.insert("a".to_string(), json!({"status": "passed"}));
    map.insert("b".to_string(), json!({"status": "omitted"}));
    assert!(completed(&map, "a"));
    assert!(!completed(&map, "b"));
    assert!(!completed(&map, "missing"));
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

#[test]
fn matrix_matches_plan_order_and_gate_classes() {
    let keys = matrix_keys();
    assert_eq!(
        keys.len(),
        keys.iter().collect::<HashSet<_>>().len(),
        "matrix keys must be unique"
    );
    assert_eq!(FAMILIES[0].name, "blobs50", "baseline gates run first");
    assert_eq!(
        FAMILIES[FAMILIES.len() - 1].name,
        "high_density31",
        "bounded probe runs last"
    );
    let asserted: Vec<&str> = FAMILIES
        .iter()
        .filter(|family| family.gate == Gate::Asserted)
        .map(|family| family.name)
        .collect();
    assert_eq!(asserted, ["blobs50", "blobs40"]);
    for family in FAMILIES
        .iter()
        .filter(|family| family.gate == Gate::Asserted)
    {
        assert_eq!(family.requested, [15, 45, 60, 61, 75, 120, 256, 512]);
        assert_eq!(family.seeds, SEEDS);
        assert!(family.timed);
    }
    let high = FAMILIES.last().expect("high density family");
    assert_eq!(
        (high.rows, high.requested, high.seeds),
        (4097, &[1024, 4096][..], &[42][..])
    );
    assert!(!high.timed && high.auto_probe);
}

#[test]
fn comparator_requires_asserted_records_and_tolerates_characterization_omissions() {
    fn synthetic_root(omit: impl Fn(&Family, u64, usize) -> bool) -> Value {
        let records: Vec<Value> = FAMILIES
            .iter()
            .flat_map(|family| {
                family.seeds.iter().flat_map(move |&seed| {
                    family
                        .requested
                        .iter()
                        .map(move |&requested| (family, seed, requested))
                })
            })
            .map(|(family, seed, requested)| {
                let key = case_key(family, seed, requested);
                if omit(family, seed, requested) {
                    json!({"key": key, "family": family.name, "gate": gate_name(family.gate), "status": "omitted"})
                } else {
                    json!({
                        "key": key,
                        "family": family.name,
                        "gate": gate_name(family.gate),
                        "status": "passed",
                        "structural_failures": [],
                        "gate_failures": [],
                        "characterization_misses": [],
                        "labels": [0, 0, 1, 1],
                    })
                }
            })
            .collect();
        json!({"schema": SCHEMA, "backend": "test", "width_gates": [], "records": records})
    }

    let complete = synthetic_root(|_, _, _| false);
    let char_omitted = synthetic_root(|family, _, _| family.gate == Gate::Characterization);
    let (failures, pairs) = compare_roots(&complete, &char_omitted);
    assert!(failures.is_empty(), "{failures:?}");
    assert!(pairs.iter().any(|pair| pair["disposition"] == "omitted"));

    let gate_omitted = synthetic_root(|family, seed, requested| {
        family.name == "blobs50" && seed == 43 && requested == 61
    });
    let (failures, _) = compare_roots(&complete, &gate_omitted);
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("mandatory gate record missing")),
        "{failures:?}"
    );
}
