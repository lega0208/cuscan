//! Phase 7 step 7a (D6) transform stage isolation on one fitted snapshot.
//!
//! `cargo run -p umap --release --features tracing --example transform_stages --
//! --load-snapshot DIR --backend cpu|cuda --output-dir OUT`
//!
//! Loads a `profile_transform_snapshot_v1` directory (written by `profile_transform
//! --save-snapshot`), reconstructs the model with `FittedUMAP::from_state`, and runs the
//! measurement-only stage-replay hook on the same fitted embedding for every variant:
//! - `approximate`: the snapshot as fitted (queue size from the snapshot config);
//! - `exact`: the same state rebuilt with `requested = resolved = Exact`, no query graph,
//!   `query_graph_width = 0` and `transform_queue_size = 4.0`;
//! - `queue_<q>`: the approximate state with `transform_queue_size` overridden
//!   (`--queue-sizes`, default `4,16`);
//! - `c_sweep_<C>_<V>`: fixed visits V (`--sweep-visits`, default the queue-derived V) and
//!   varying candidates (`--sweep-candidates`); `v_sweep_<C>_<V>`: fixed candidates
//!   (`--sweep-fixed-candidates`, default the queue-derived C) and varying visits
//!   (`--sweep-visits-list`). Defaults are derived from the snapshot queue budget; pairs
//!   violating `k <= C <= V <= n` are skipped and recorded;
//! - `epochs_<e>`: approximate search with transform epochs `e` (`--transform-epochs`,
//!   e.g. `0,5,10,11,100`; config `n_epochs = 3e`, since transform uses `n_epochs / 3`);
//! - `inject_knn`: `--inject-knn DIR` with `indices.i32le` and `distances.f32le`, `[q, k]`,
//!   through the graph entry;
//! - `inject_layout`: `--inject-layout DIR` with `heads.i32le`, `tails.i32le`,
//!   `weights.f32le` and `initial.f32le` (`[q, n_components]`) through the layout entry.
//!
//! Queries come from the snapshot's `query.f32le`, or from `--query FILE` (raw f32le,
//! `features` columns). Output: `OUT/summary.json` plus one directory per variant with
//! raw little-endian arrays named `<field>.<dtype>le`; `summary.json` lists each array's
//! dtype, shape, bytes and FNV-1a64, the stage scalars, and recall@k of each variant's
//! query neighbor IDs against the `exact` variant (set overlap; tie handling and all
//! quality metrics are left to the Python scorer). OUT must not exist.

use std::{
    collections::BTreeSet,
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};
use umap::{
    BackendChoice, FittedUMAP, Init, ModelState, NeighborSearch, Optimizer, UMAPConfig,
    standard_umap::{QueryBudget, approximate_query_budget, query_neighbor_width},
};

const SNAPSHOT_FORMAT: &str = "profile_transform_snapshot_v1";
/// Must match `profile_transform`'s `TRANSFORM_SEED`; checked against the manifest.
const TRANSFORM_SEED: u64 = 42;

type Res<T> = Result<T, Box<dyn Error>>;

struct Options {
    snapshot: PathBuf,
    backend: String,
    output_dir: PathBuf,
    query: Option<PathBuf>,
    queue_sizes: Vec<f64>,
    sweep_candidates: Option<Vec<usize>>,
    sweep_visits: Option<usize>,
    sweep_fixed_candidates: Option<usize>,
    sweep_visits_list: Option<Vec<usize>>,
    transform_epochs: Vec<usize>,
    inject_knn: Option<PathBuf>,
    inject_layout: Option<PathBuf>,
    skip_sweeps: bool,
}

fn list<T: std::str::FromStr>(value: &str) -> Res<Vec<T>>
where
    T::Err: Error + 'static,
{
    value
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<T>().map_err(|e| e.into()))
        .collect()
}

fn options() -> Res<Options> {
    let (mut snapshot, mut output_dir, mut query) = (None, None, None);
    let mut backend = "cpu".to_string();
    let mut queue_sizes = vec![4.0_f64, 16.0];
    let (mut sweep_candidates, mut sweep_visits) = (None, None);
    let (mut sweep_fixed_candidates, mut sweep_visits_list) = (None, None);
    let mut transform_epochs = Vec::new();
    let (mut inject_knn, mut inject_layout) = (None, None);
    let mut skip_sweeps = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "transform_stages --load-snapshot DIR --output-dir OUT [--backend cpu|cuda] [--query FILE] [--queue-sizes F,F] [--sweep-candidates C,C --sweep-visits V] [--sweep-fixed-candidates C --sweep-visits-list V,V] [--skip-sweeps true|false] [--transform-epochs E,E] [--inject-knn DIR] [--inject-layout DIR]"
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--load-snapshot" => snapshot = Some(PathBuf::from(value)),
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--backend" => backend = value,
            "--query" => query = Some(PathBuf::from(value)),
            "--queue-sizes" => queue_sizes = list(&value)?,
            "--sweep-candidates" => sweep_candidates = Some(list(&value)?),
            "--sweep-visits" => sweep_visits = Some(value.parse()?),
            "--sweep-fixed-candidates" => sweep_fixed_candidates = Some(value.parse()?),
            "--sweep-visits-list" => sweep_visits_list = Some(list(&value)?),
            "--skip-sweeps" => skip_sweeps = value.parse()?,
            "--transform-epochs" => transform_epochs = list(&value)?,
            "--inject-knn" => inject_knn = Some(PathBuf::from(value)),
            "--inject-layout" => inject_layout = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {flag}").into()),
        }
    }
    if backend != "cpu" && backend != "cuda" {
        return Err("--backend must be cpu or cuda".into());
    }
    if queue_sizes.iter().any(|q| !q.is_finite() || *q <= 0.0) {
        return Err("--queue-sizes must be finite and positive".into());
    }
    let output_dir = output_dir.ok_or("--output-dir is required")?;
    if output_dir.exists() {
        return Err("output directory must not exist".into());
    }
    Ok(Options {
        snapshot: snapshot.ok_or("--load-snapshot is required")?,
        backend,
        output_dir,
        query,
        queue_sizes,
        sweep_candidates,
        sweep_visits,
        sweep_fixed_candidates,
        sweep_visits_list,
        transform_epochs,
        inject_knn,
        inject_layout,
        skip_sweeps,
    })
}

fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn read_checked(dir: &Path, manifest: &serde_json::Value, name: &str) -> Res<Option<Vec<u8>>> {
    let entry = &manifest["files"][name];
    if entry.is_null() {
        return Ok(None);
    }
    let bytes = std::fs::read(dir.join(name))?;
    if entry["bytes"].as_u64() != Some(bytes.len() as u64)
        || entry["fnv1a64"].as_str() != Some(fnv1a64(&bytes).as_str())
    {
        return Err(format!("snapshot file {name} fails its checksum").into());
    }
    Ok(Some(bytes))
}

fn le4<T>(bytes: &[u8], what: &str, f: fn([u8; 4]) -> T) -> Res<Vec<T>> {
    if bytes.len() % 4 != 0 {
        return Err(format!("{what} is not a whole number of 4-byte values").into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f(c.try_into().unwrap()))
        .collect())
}

fn parse_search(name: Option<&str>) -> Res<NeighborSearch> {
    Ok(match name {
        Some("exact") => NeighborSearch::Exact,
        Some("approximate") => NeighborSearch::Approximate,
        Some("auto") => NeighborSearch::Auto,
        _ => return Err("snapshot search invalid".into()),
    })
}

struct Snapshot {
    state: ModelState,
    query: Vec<Vec<f32>>,
    manifest_fnv1a64: String,
    query_fnv1a64: String,
    parameters: serde_json::Value,
}

/// Load a `profile_transform_snapshot_v1` directory; the config is rebuilt exactly as
/// `profile_transform::fit_config` does (the snapshot saver verified that equality).
fn load_snapshot(opts: &Options) -> Res<Snapshot> {
    let dir = &opts.snapshot;
    let manifest_bytes = std::fs::read(dir.join("manifest.json"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest["format"] != SNAPSHOT_FORMAT {
        return Err("unsupported snapshot format".into());
    }
    let p = &manifest["parameters"];
    if p["backend"] != opts.backend.as_str() {
        return Err("snapshot was fitted on a different backend".into());
    }
    if p["transform_seed"].as_u64() != Some(TRANSFORM_SEED) {
        return Err("snapshot transform seed differs from the reconstructible config".into());
    }
    let s = &manifest["state"];
    let num = |v: &serde_json::Value, key: &str| -> Res<u64> {
        v[key]
            .as_u64()
            .ok_or_else(|| format!("snapshot field {key} missing").into())
    };
    let size =
        |v: &serde_json::Value, key: &str| -> Res<usize> { Ok(usize::try_from(num(v, key)?)?) };
    let optimizer = match p["optimizer"].as_str() {
        Some("compatibility") => Optimizer::Compatibility,
        Some("adam") => Optimizer::Adam,
        Some("momentum") => Optimizer::Momentum,
        _ => return Err("snapshot optimizer invalid".into()),
    };
    let queue_size = p["fit_queue_size"]
        .as_f64()
        .ok_or("snapshot field fit_queue_size missing")?;
    let config = UMAPConfig {
        n_neighbors: size(p, "neighbors")?,
        n_components: size(p, "components")?,
        n_epochs: Some(size(p, "epochs")?),
        init: Init::Random,
        optimizer,
        random_state: Some(num(p, "fit_seed")?),
        transform_seed: TRANSFORM_SEED,
        transform_queue_size: queue_size,
        ..UMAPConfig::default()
    };
    let f32s = |name: &str| -> Res<Vec<f32>> {
        let bytes =
            read_checked(dir, &manifest, name)?.ok_or(format!("snapshot missing {name}"))?;
        le4(&bytes, name, f32::from_le_bytes)
    };
    let query_graph = read_checked(dir, &manifest, "query_graph.i32le")?
        .map(|bytes| le4(&bytes, "query graph", i32::from_le_bytes))
        .transpose()?;
    let original_to_training = read_checked(dir, &manifest, "original_to_training.u64le")?
        .map(|bytes| {
            bytes
                .chunks_exact(8)
                .map(|c| usize::try_from(u64::from_le_bytes(c.try_into().unwrap())))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let bits = |key: &str| -> Res<f32> { Ok(f32::from_bits(u32::try_from(num(s, key)?)?)) };
    let state = ModelState {
        config,
        requested_search: parse_search(s["requested_search"].as_str())?,
        resolved_search: parse_search(s["resolved_search"].as_str())?,
        a: bits("a_bits")?,
        b: bits("b_bits")?,
        training_features: f32s("training_features.f32le")?,
        training_rows: size(s, "training_rows")?,
        n_features: size(s, "n_features")?,
        embedding: f32s("embedding.f32le")?,
        embedding_rows: size(s, "embedding_rows")?,
        n_components: size(s, "n_components")?,
        query_graph,
        query_graph_width: size(s, "query_graph_width")?,
        original_to_training,
    };
    let features = state.n_features;
    let flat = match &opts.query {
        Some(path) => le4(&std::fs::read(path)?, "query file", f32::from_le_bytes)?,
        None => f32s("query.f32le")?,
    };
    if features == 0 || flat.is_empty() || flat.len() % features != 0 {
        return Err("query size is not a nonzero multiple of the feature count".into());
    }
    let query_fnv1a64 = fnv1a64(&f32_bytes(&flat));
    let query = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    Ok(Snapshot {
        state,
        query,
        manifest_fnv1a64: fnv1a64(&manifest_bytes),
        query_fnv1a64,
        parameters: p.clone(),
    })
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn backend_choice(name: &str) -> BackendChoice {
    match name {
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => BackendChoice::Cpu,
    }
}

fn exact_state(state: &ModelState) -> ModelState {
    let mut exact = state.clone();
    exact.requested_search = NeighborSearch::Exact;
    exact.resolved_search = NeighborSearch::Exact;
    exact.query_graph = None;
    exact.query_graph_width = 0;
    exact.config.transform_queue_size = 4.0;
    exact
}

/// Writes one array and returns its summary entry.
fn array<T: Copy>(
    dir: &Path,
    field: &str,
    dtype: &str,
    values: &[T],
    shape: &[usize],
    to_le: fn(T) -> Vec<u8>,
) -> Res<serde_json::Value> {
    let bytes: Vec<u8> = values.iter().flat_map(|&v| to_le(v)).collect();
    let name = format!("{field}.{dtype}le");
    std::fs::write(dir.join(&name), &bytes)?;
    Ok(
        serde_json::json!({"file": name, "dtype": dtype, "shape": shape,
        "bytes": bytes.len(), "fnv1a64": fnv1a64(&bytes)}),
    )
}

/// Shape for a field: `[rows, width]` when it divides evenly, else flat.
fn shape(len: usize, rows: usize) -> Vec<usize> {
    if rows > 0 && len % rows == 0 && len > 0 {
        let width = len / rows;
        if width == 1 {
            vec![rows]
        } else {
            vec![rows, width]
        }
    } else {
        vec![len]
    }
}

struct Run<'a> {
    name: String,
    model: &'a FittedUMAP,
    kind: Kind<'a>,
    info: serde_json::Value,
}

enum Kind<'a> {
    Search(Option<QueryBudget>),
    Graph(&'a [i32], &'a [f32]),
    Layout(&'a [i32], &'a [i32], &'a [f32], &'a [f32]),
}

fn execute(
    run: Run,
    query: &[Vec<f32>],
    out: &Path,
    components: usize,
) -> Res<(serde_json::Value, Vec<i32>)> {
    let q = query.len();
    let start = Instant::now();
    let replay = match run.kind {
        Kind::Search(budget) => run.model.replay_transform_search(query, budget)?,
        Kind::Graph(indices, distances) => {
            run.model.replay_transform_graph(q, indices, distances)?
        }
        Kind::Layout(heads, tails, weights, initial) => run
            .model
            .replay_transform_layout(q, heads, tails, weights, initial)?,
    };
    let wall = start.elapsed().as_secs_f64();
    if replay.embedding.len() != q * components || replay.embedding.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "{}: embedding has invalid shape or non-finite values",
            run.name
        )
        .into());
    }
    let dir = out.join(&run.name);
    std::fs::create_dir(&dir)?;
    let f32le = |v: f32| v.to_le_bytes().to_vec();
    let i32le = |v: i32| v.to_le_bytes().to_vec();
    let u32le = |v: u32| v.to_le_bytes().to_vec();
    let f64le = |v: f64| v.to_le_bytes().to_vec();
    let mut arrays = serde_json::Map::new();
    let mut put = |field: &str, entry: Res<serde_json::Value>| -> Res<()> {
        arrays.insert(field.into(), entry?);
        Ok(())
    };
    let r = &replay;
    let edges = r.edge_weights.len();
    put(
        "neighbor_indices",
        array(
            &dir,
            "neighbor_indices",
            "i32",
            &r.neighbor_indices,
            &shape(r.neighbor_indices.len(), q),
            i32le,
        ),
    )?;
    put(
        "neighbor_distances",
        array(
            &dir,
            "neighbor_distances",
            "f32",
            &r.neighbor_distances,
            &shape(r.neighbor_distances.len(), q),
            f32le,
        ),
    )?;
    put(
        "query_stats",
        array(
            &dir,
            "query_stats",
            "i32",
            &r.query_stats,
            &shape(r.query_stats.len(), q),
            i32le,
        ),
    )?;
    put(
        "rho",
        array(&dir, "rho", "f32", &r.rho, &[r.rho.len()], f32le),
    )?;
    put(
        "sigma",
        array(&dir, "sigma", "f32", &r.sigma, &[r.sigma.len()], f32le),
    )?;
    put(
        "memberships",
        array(
            &dir,
            "memberships",
            "f32",
            &r.memberships,
            &shape(r.memberships.len(), q),
            f32le,
        ),
    )?;
    put(
        "graph_heads",
        array(
            &dir,
            "graph_heads",
            "i32",
            &r.graph_heads,
            &[r.graph_heads.len()],
            i32le,
        ),
    )?;
    put(
        "graph_tails",
        array(
            &dir,
            "graph_tails",
            "i32",
            &r.graph_tails,
            &[r.graph_tails.len()],
            i32le,
        ),
    )?;
    put(
        "graph_weights",
        array(
            &dir,
            "graph_weights",
            "f32",
            &r.graph_weights,
            &[r.graph_weights.len()],
            f32le,
        ),
    )?;
    put(
        "edge_heads",
        array(
            &dir,
            "edge_heads",
            "i32",
            &r.edge_heads,
            &[r.edge_heads.len()],
            i32le,
        ),
    )?;
    put(
        "edge_tails",
        array(
            &dir,
            "edge_tails",
            "i32",
            &r.edge_tails,
            &[r.edge_tails.len()],
            i32le,
        ),
    )?;
    put(
        "edge_weights",
        array(
            &dir,
            "edge_weights",
            "f32",
            &r.edge_weights,
            &[edges],
            f32le,
        ),
    )?;
    put(
        "csr_offsets",
        array(
            &dir,
            "csr_offsets",
            "u32",
            &r.csr_offsets,
            &[r.csr_offsets.len()],
            u32le,
        ),
    )?;
    put(
        "csr_indices",
        array(
            &dir,
            "csr_indices",
            "u32",
            &r.csr_indices,
            &[r.csr_indices.len()],
            u32le,
        ),
    )?;
    put(
        "csr_weights",
        array(
            &dir,
            "csr_weights",
            "f32",
            &r.csr_weights,
            &[r.csr_weights.len()],
            f32le,
        ),
    )?;
    put(
        "periods",
        array(
            &dir,
            "periods",
            "f64",
            &r.periods,
            &[r.periods.len()],
            f64le,
        ),
    )?;
    put(
        "negative_periods",
        array(
            &dir,
            "negative_periods",
            "f64",
            &r.negative_periods,
            &[r.negative_periods.len()],
            f64le,
        ),
    )?;
    put(
        "next_positive",
        array(
            &dir,
            "next_positive",
            "f64",
            &r.next_positive,
            &[r.next_positive.len()],
            f64le,
        ),
    )?;
    put(
        "next_negative",
        array(
            &dir,
            "next_negative",
            "f64",
            &r.next_negative,
            &[r.next_negative.len()],
            f64le,
        ),
    )?;
    put(
        "initial",
        array(
            &dir,
            "initial",
            "f32",
            &r.initial,
            &shape(r.initial.len(), q),
            f32le,
        ),
    )?;
    put(
        "embedding",
        array(
            &dir,
            "embedding",
            "f32",
            &r.embedding,
            &[q, components],
            f32le,
        ),
    )?;
    println!(
        "{} wall={wall:.6}s epochs={} edges={edges}",
        run.name, r.epochs
    );
    let entry = serde_json::json!({
        "name": run.name, "directory": run.name, "info": run.info, "rows": q,
        "wall_seconds": wall, "epochs": r.epochs, "initial_alpha": r.initial_alpha,
        "arrays": arrays,
    });
    Ok((entry, replay.neighbor_indices))
}

/// Mean per-row |A ∩ B| / k over neighbor-ID sets; `None` when shapes differ.
fn recall(ids: &[i32], exact: &[i32], rows: usize) -> Option<f64> {
    if rows == 0 || ids.len() != exact.len() || ids.is_empty() || ids.len() % rows != 0 {
        return None;
    }
    let k = ids.len() / rows;
    let total: usize = ids
        .chunks_exact(k)
        .zip(exact.chunks_exact(k))
        .map(|(a, b)| {
            let a: BTreeSet<i32> = a.iter().copied().collect();
            let b: BTreeSet<i32> = b.iter().copied().collect();
            a.intersection(&b).count()
        })
        .sum();
    Some(total as f64 / (rows * k) as f64)
}

fn load_raw(dir: &Path, name: &str) -> Res<Vec<u8>> {
    std::fs::read(dir.join(name)).map_err(|e| format!("{}: {e}", dir.join(name).display()).into())
}

fn main() -> Res<()> {
    let opts = options()?;
    let snap = load_snapshot(&opts)?;
    let backend = backend_choice(&opts.backend);
    let state = &snap.state;
    let components = state.n_components;
    let n = state.training_rows;
    let k = query_neighbor_width(n, state.config.n_neighbors)?;
    let base_queue = state.config.transform_queue_size;
    let approximate = state.resolved_search == NeighborSearch::Approximate;

    // Inputs are read and validated before any model is built.
    let injected_knn = opts
        .inject_knn
        .as_ref()
        .map(|dir| -> Res<(Vec<i32>, Vec<f32>)> {
            Ok((
                le4(
                    &load_raw(dir, "indices.i32le")?,
                    "indices",
                    i32::from_le_bytes,
                )?,
                le4(
                    &load_raw(dir, "distances.f32le")?,
                    "distances",
                    f32::from_le_bytes,
                )?,
            ))
        })
        .transpose()?;
    let injected_layout = opts
        .inject_layout
        .as_ref()
        .map(|dir| -> Res<(Vec<i32>, Vec<i32>, Vec<f32>, Vec<f32>)> {
            Ok((
                le4(&load_raw(dir, "heads.i32le")?, "heads", i32::from_le_bytes)?,
                le4(&load_raw(dir, "tails.i32le")?, "tails", i32::from_le_bytes)?,
                le4(
                    &load_raw(dir, "weights.f32le")?,
                    "weights",
                    f32::from_le_bytes,
                )?,
                le4(
                    &load_raw(dir, "initial.f32le")?,
                    "initial",
                    f32::from_le_bytes,
                )?,
            ))
        })
        .transpose()?;

    std::fs::create_dir(&opts.output_dir)?;
    let start = Instant::now();
    let model = FittedUMAP::from_state(state.clone(), backend)?;
    let load_wall = start.elapsed().as_secs_f64();
    let exact_model = FittedUMAP::from_state(exact_state(state), backend)?;

    let mut runs: Vec<serde_json::Value> = Vec::new();
    let mut ids: Vec<(String, Vec<i32>)> = Vec::new();
    let mut skipped: Vec<serde_json::Value> = Vec::new();
    let go = |run: Run,
              runs: &mut Vec<serde_json::Value>,
              ids: &mut Vec<(String, Vec<i32>)>|
     -> Res<()> {
        let name = run.name.clone();
        let (entry, neighbor_ids) = execute(run, &snap.query, &opts.output_dir, components)?;
        runs.push(entry);
        ids.push((name, neighbor_ids));
        Ok(())
    };

    go(
        Run {
            name: "exact".into(),
            model: &exact_model,
            kind: Kind::Search(None),
            info: serde_json::json!({"search": "exact"}),
        },
        &mut runs,
        &mut ids,
    )?;
    go(
        Run {
            name: "approximate".into(),
            model: &model,
            kind: Kind::Search(None),
            info: serde_json::json!({"search": if approximate { "approximate" } else { "exact" },
            "queue_size": base_queue}),
        },
        &mut runs,
        &mut ids,
    )?;

    let mut base_budget = None;
    if approximate {
        for &queue in &opts.queue_sizes {
            let mut s = state.clone();
            s.config.transform_queue_size = queue;
            let m = FittedUMAP::from_state(s, backend)?;
            let budget = approximate_query_budget(n, k, queue)?;
            go(
                Run {
                    name: format!("queue_{queue}"),
                    model: &m,
                    kind: Kind::Search(None),
                    info: serde_json::json!({"search": "approximate", "queue_size": queue,
                    "candidates": budget.candidates, "visits": budget.visits}),
                },
                &mut runs,
                &mut ids,
            )?;
        }
        let base = approximate_query_budget(n, k, base_queue)?;
        base_budget = Some(base);
        if !opts.skip_sweeps {
            let fixed_v = opts.sweep_visits.unwrap_or(base.visits);
            let c_list = opts.sweep_candidates.clone().unwrap_or_else(|| {
                vec![
                    k,
                    (k + base.candidates) / 2,
                    base.candidates,
                    base.candidates.saturating_mul(2),
                    fixed_v,
                ]
            });
            let fixed_c = opts.sweep_fixed_candidates.unwrap_or(base.candidates);
            let v_list = opts.sweep_visits_list.clone().unwrap_or_else(|| {
                vec![
                    fixed_c,
                    base.visits / 4,
                    base.visits / 2,
                    base.visits,
                    base.visits.saturating_mul(2),
                ]
            });
            let mut pairs: Vec<(&str, usize, usize)> = Vec::new();
            for c in c_list.iter().copied().collect::<BTreeSet<_>>() {
                pairs.push(("c_sweep", c, fixed_v));
            }
            for v in v_list.iter().copied().collect::<BTreeSet<_>>() {
                pairs.push(("v_sweep", fixed_c, v));
            }
            for (sweep, c, v) in pairs {
                let info = serde_json::json!({"search": "approximate", "sweep": sweep,
                    "candidates": c, "visits": v, "fixed": if sweep == "c_sweep" { "visits" } else { "candidates" }});
                if !(k <= c && c <= v && v <= n) {
                    skipped.push(
                        serde_json::json!({"sweep": sweep, "candidates": c, "visits": v,
                        "reason": format!("violates k={k} <= C <= V <= n={n}")}),
                    );
                    continue;
                }
                go(
                    Run {
                        name: format!("{sweep}_c{c}_v{v}"),
                        model: &model,
                        kind: Kind::Search(Some(QueryBudget {
                            candidates: c,
                            visits: v,
                        })),
                        info,
                    },
                    &mut runs,
                    &mut ids,
                )?;
            }
        }
    } else {
        skipped.push(serde_json::json!({"sweep": "queue/c/v", "reason": "snapshot search is not approximate"}));
    }

    for &epochs in &opts.transform_epochs {
        let mut s = state.clone();
        s.config.n_epochs = Some(epochs.checked_mul(3).ok_or("transform epochs overflow")?);
        let m = FittedUMAP::from_state(s, backend)?;
        go(
            Run {
                name: format!("epochs_{epochs}"),
                model: &m,
                kind: Kind::Search(None),
                info: serde_json::json!({"search": "snapshot", "transform_epochs": epochs,
                "config_n_epochs": epochs * 3}),
            },
            &mut runs,
            &mut ids,
        )?;
    }

    if let Some((indices, distances)) = &injected_knn {
        go(
            Run {
                name: "inject_knn".into(),
                model: &model,
                kind: Kind::Graph(indices, distances),
                info: serde_json::json!({"entry": "graph", "source": opts.inject_knn}),
            },
            &mut runs,
            &mut ids,
        )?;
    }
    if let Some((heads, tails, weights, initial)) = &injected_layout {
        go(
            Run {
                name: "inject_layout".into(),
                model: &model,
                kind: Kind::Layout(heads, tails, weights, initial),
                info: serde_json::json!({"entry": "layout", "source": opts.inject_layout}),
            },
            &mut runs,
            &mut ids,
        )?;
    }

    let q = snap.query.len();
    let exact_ids = ids[0].1.clone();
    let recalls: serde_json::Map<String, serde_json::Value> = ids
        .iter()
        .map(|(name, v)| (name.clone(), serde_json::json!(recall(v, &exact_ids, q))))
        .collect();
    let summary = serde_json::json!({
        "format": "transform_stages_v1",
        "snapshot": {"path": opts.snapshot, "manifest_fnv1a64": snap.manifest_fnv1a64,
            "parameters": snap.parameters, "requested_search": format!("{:?}", state.requested_search),
            "resolved_search": format!("{:?}", state.resolved_search)},
        "backend": opts.backend,
        "query": {"source": opts.query.as_ref().map(|p| p.display().to_string()).unwrap_or("snapshot".into()),
            "rows": q, "features": state.n_features, "fnv1a64": snap.query_fnv1a64},
        "training_rows": n, "query_width": k, "n_components": components,
        "base_queue_size": base_queue,
        "base_budget": base_budget.map(|b| serde_json::json!({"candidates": b.candidates, "visits": b.visits})),
        "load_wall_seconds": load_wall,
        "recall_vs_exact": recalls,
        "recall_note": "set overlap of neighbor IDs against the exact variant; ties not resolved",
        "runs": runs,
        "skipped": skipped,
    });
    std::fs::write(
        opts.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    println!("summary={}", opts.output_dir.join("summary.json").display());
    Ok(())
}
