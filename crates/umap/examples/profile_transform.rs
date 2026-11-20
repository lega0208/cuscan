//! Measure a single fit (or a loaded snapshot) and repeated held-out transforms.
//!
//! Traced build: `cargo run -p umap --release --features tracing --example profile_transform --
//! --backend cpu --output-dir /tmp/transform-profile` (use `cuda` for strict CUDA).
//! Wall-only build: omit `--features tracing`; no trace, no stage spans, no barriers
//! (`--stage-barriers`/`--paired-modes` must stay false); only public transform wall time.
//! Snapshots: `--save-snapshot DIR` writes the fitted `export_state` (raw little-endian
//! arrays plus `manifest.json` with FNV-1a64 checksums and fit parameters) and the
//! deterministic query input; `--snapshot-only true` stops after saving. `--load-snapshot DIR`
//! skips fitting and reconstructs the model with `from_state`; fit-defining parameters come
//! from the manifest (explicit conflicting flags are errors), while `--query-rows` and
//! `--queue-size` (a transform-only override) may differ. `--fit-seed N` (42) seeds the fit.
//! Options: --train-rows N (1000), --query-rows N (128), --features N (16),
//! --components N (2), --neighbors N (15), --epochs N (200), --queue-size F
//! (4.0), --init random|recursive (random), --warm-repeats N (2, minimum 2), --stage-barriers true|false
//! (default true), --paired-modes true|false (default false), --output-dir PATH
//! (required). Paired mode alternates both modes on the same fitted model.
//! The output directory must not exist; its parent must already exist. Trace and
//! summary contain no input or embedding dumps. Stage tracing adds overhead and
//! estimator instrumentation may synchronize devices; these are not GPU event times.

use std::{
    collections::HashSet,
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};
#[cfg(feature = "tracing")]
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
#[cfg(feature = "tracing")]
use tracing::{Id, Subscriber};
#[cfg(feature = "tracing")]
use tracing_subscriber::{
    layer::{Context, Layer, SubscriberExt},
    registry::LookupSpan,
};
use umap::{
    BackendChoice, FittedUMAP, Init, ModelState, NeighborSearch, Optimizer, UMAP, UMAPConfig,
};

const TRACING: bool = cfg!(feature = "tracing");
const SNAPSHOT_FORMAT: &str = "profile_transform_snapshot_v1";
const TRANSFORM_SEED: u64 = 42;
#[cfg(feature = "tracing")]
const STAGES: [&str; 3] = ["neighbor_search", "fuzzy_graph", "optimization"];
const DATA_SEED: u64 = 0x1234_5678_9abc_def0;
const MODEL_SEED: u64 = 42;

struct Options {
    backend: String,
    optimizer: Optimizer,
    init: Init,
    train_rows: usize,
    query_rows: usize,
    features: usize,
    components: usize,
    neighbors: usize,
    epochs: usize,
    queue_size: f64,
    warm_repeats: usize,
    stage_barriers: bool,
    paired_modes: bool,
    output_dir: PathBuf,
    fit_seed: u64,
    save_snapshot: Option<PathBuf>,
    load_snapshot: Option<PathBuf>,
    snapshot_only: bool,
    explicit: HashSet<String>,
}

fn options() -> Result<Options, Box<dyn Error>> {
    let mut backend = "cpu".to_string();
    let mut optimizer = Optimizer::Compatibility;
    let mut init = Init::Random;
    let (mut train_rows, mut query_rows, mut features, mut components): (
        usize,
        usize,
        usize,
        usize,
    ) = (1000, 128, 16, 2);
    let (mut neighbors, mut epochs, mut queue_size, mut warm_repeats): (usize, usize, f64, usize) =
        (15, 200, 4.0, 2);
    let mut output_dir = None;
    let mut stage_barriers = TRACING;
    let mut paired_modes = false;
    let mut fit_seed = MODEL_SEED;
    let (mut save_snapshot, mut load_snapshot, mut snapshot_only) = (None, None, false);
    let mut explicit = HashSet::new();
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "profile_transform --output-dir PATH [--backend cpu|cuda] [--optimizer compatibility|adam|momentum] [--init random|recursive] [--train-rows N] [--query-rows N] [--features N] [--components N] [--neighbors N] [--epochs N] [--fit-seed N] [--queue-size F] [--warm-repeats N] [--stage-barriers true|false] [--paired-modes true|false] [--save-snapshot DIR [--snapshot-only true|false] | --load-snapshot DIR]\nStage tracing requires --features tracing; without it the build is wall-only. Output and snapshot directories must not exist."
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--backend" => backend = value,
            "--optimizer" => {
                optimizer = match value.as_str() {
                    "compatibility" => Optimizer::Compatibility,
                    "adam" => Optimizer::Adam,
                    "momentum" => Optimizer::Momentum,
                    _ => return Err("optimizer must be compatibility, adam, or momentum".into()),
                };
            }
            "--init" => init = parse_init(&value)?,
            "--train-rows" => train_rows = value.parse()?,
            "--query-rows" => query_rows = value.parse()?,
            "--features" => features = value.parse()?,
            "--components" => components = value.parse()?,
            "--neighbors" => neighbors = value.parse()?,
            "--epochs" => epochs = value.parse()?,
            "--queue-size" => queue_size = value.parse::<f64>()?,
            "--warm-repeats" => warm_repeats = value.parse()?,
            "--stage-barriers" => stage_barriers = value.parse()?,
            "--paired-modes" => paired_modes = value.parse()?,
            "--output-dir" => output_dir = Some(PathBuf::from(value)),
            "--fit-seed" => fit_seed = value.parse()?,
            "--save-snapshot" => save_snapshot = Some(PathBuf::from(value)),
            "--load-snapshot" => load_snapshot = Some(PathBuf::from(value)),
            "--snapshot-only" => snapshot_only = value.parse()?,
            _ => return Err(format!("unknown option {flag}").into()),
        }
        explicit.insert(flag);
    }
    if !TRACING && (stage_barriers || paired_modes) {
        return Err("--stage-barriers and --paired-modes require --features tracing".into());
    }
    if save_snapshot.is_some() && load_snapshot.is_some() {
        return Err("--save-snapshot and --load-snapshot are exclusive".into());
    }
    if snapshot_only && save_snapshot.is_none() {
        return Err("--snapshot-only requires --save-snapshot".into());
    }
    if backend != "cpu" && backend != "cuda" {
        return Err("backend must be cpu or cuda".into());
    }
    if train_rows < 3
        || query_rows == 0
        || features == 0
        || components == 0
        || neighbors < 2
        || neighbors >= train_rows
        || epochs < 3
        || warm_repeats < 2
        || !queue_size.is_finite()
        || queue_size <= 0.0
        || train_rows
            .checked_add(query_rows)
            .and_then(|n| n.checked_mul(features))
            .is_none()
        || query_rows.checked_mul(components).is_none()
        || warm_repeats
            .checked_add(1)
            .and_then(|n| n.checked_mul(if paired_modes { 2 } else { 1 }))
            .is_none()
    {
        return Err("invalid dimensions, neighbors, epochs (minimum 3), queue size, or warm repeats (minimum 2)".into());
    }
    Ok(Options {
        backend,
        optimizer,
        init,
        train_rows,
        query_rows,
        features,
        components,
        neighbors,
        epochs,
        queue_size,
        warm_repeats,
        stage_barriers,
        paired_modes,
        output_dir: output_dir.ok_or("--output-dir is required")?,
        fit_seed,
        save_snapshot,
        load_snapshot,
        snapshot_only,
        explicit,
    })
}

#[cfg(feature = "tracing")]
#[derive(Default)]
struct BarrierRun {
    stages: BTreeMap<String, f64>,
    sync_seconds: f64,
    enabled: bool,
}

#[cfg(feature = "tracing")]
#[derive(Default)]
struct BarrierState {
    runs: Vec<BarrierRun>,
    markers: HashMap<Id, usize>,
    active: HashMap<Id, (usize, String, Instant)>,
    errors: Vec<String>,
}

#[cfg(feature = "tracing")]
struct BarrierLayer {
    device: burn::tensor::Device,
    state: Arc<Mutex<BarrierState>>,
    enabled: Arc<AtomicBool>,
}

#[cfg(feature = "tracing")]
impl<S> Layer<S> for BarrierLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let metadata = span.metadata();
        if metadata.name() == "profile_transform_run" && metadata.target() == "profile_transform" {
            let mut state = self.state.lock().unwrap();
            let run = state.runs.len();
            state.runs.push(BarrierRun {
                enabled: self.enabled.load(Ordering::SeqCst),
                ..BarrierRun::default()
            });
            state.markers.insert(id.clone(), run);
            return;
        }
        if !STAGES.contains(&metadata.name())
            || !metadata
                .target()
                .starts_with("umap::standard_umap::estimator")
        {
            return;
        }
        let run = {
            let state = self.state.lock().unwrap();
            span.scope()
                .find_map(|parent| state.markers.get(&parent.id()).copied())
        };
        let Some(run) = run else { return };
        if !self.state.lock().unwrap().runs[run].enabled {
            return;
        }
        let before_sync = Instant::now();
        let sync = self.device.sync();
        let after_sync = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.runs[run].sync_seconds += after_sync.duration_since(before_sync).as_secs_f64();
        if let Err(error) = sync {
            state
                .errors
                .push(format!("stage entry synchronization failed: {error}"));
        }
        if state
            .active
            .insert(id.clone(), (run, metadata.name().to_string(), after_sync))
            .is_some()
        {
            state.errors.push("stage entered twice without exit".into());
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if span.metadata().name() == "profile_transform_run"
            && span.metadata().target() == "profile_transform"
        {
            self.state.lock().unwrap().markers.remove(id);
            return;
        }
        let active = self.state.lock().unwrap().active.remove(id);
        let Some((run, name, start)) = active else {
            return;
        };
        let before_sync = Instant::now();
        let sync = self.device.sync();
        let end = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.runs[run].sync_seconds += end.duration_since(before_sync).as_secs_f64();
        *state.runs[run].stages.entry(name).or_default() += end.duration_since(start).as_secs_f64();
        if let Err(error) = sync {
            state
                .errors
                .push(format!("stage exit synchronization failed: {error}"));
        }
    }
}

// Sum of twelve uniforms approximates a standard normal, without a distribution dependency.
fn data(rows: usize, features: usize, state: &mut u64) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            let cluster = (row % 8) as f32;
            (0..features)
                .map(|column| {
                    let mut noise = 0.0;
                    for _ in 0..12 {
                        *state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        noise += ((*state >> 40) as f32) / 16_777_216.0;
                    }
                    let center = (((cluster as usize * 7 + column * 3) % 11) as f32 - 5.0) * 1.5;
                    center + noise - 6.0
                })
                .collect()
        })
        .collect()
}

fn validate_embedding(
    embedding: &[Vec<f32>],
    rows: usize,
    components: usize,
) -> Result<String, Box<dyn Error>> {
    if embedding.len() != rows
        || embedding
            .iter()
            .any(|row| row.len() != components || row.iter().any(|v| !v.is_finite()))
    {
        return Err("embedding has invalid shape or non-finite values".into());
    }
    // FNV-1a over little-endian IEEE-754 bits; comparison is informational only.
    Ok(rows_checksum(embedding))
}

fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn f32_bytes<'a>(values: impl IntoIterator<Item = &'a f32>) -> Vec<u8> {
    values.into_iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn rows_checksum(rows: &[Vec<f32>]) -> String {
    fnv1a64(&f32_bytes(rows.iter().flatten()))
}

fn optimizer_name(optimizer: Optimizer) -> &'static str {
    match optimizer {
        Optimizer::Compatibility => "compatibility",
        Optimizer::Adam => "adam",
        Optimizer::Momentum => "momentum",
    }
}

fn search_name(search: NeighborSearch) -> &'static str {
    match search {
        NeighborSearch::Exact => "exact",
        NeighborSearch::Approximate => "approximate",
        NeighborSearch::Auto => "auto",
    }
}

fn parse_search(name: &str) -> Result<NeighborSearch, Box<dyn Error>> {
    Ok(match name {
        "exact" => NeighborSearch::Exact,
        "approximate" => NeighborSearch::Approximate,
        "auto" => NeighborSearch::Auto,
        _ => return Err(format!("unknown search {name}").into()),
    })
}

fn parse_init(name: &str) -> Result<Init, Box<dyn Error>> {
    match name {
        "random" => Ok(Init::Random),
        "recursive" => Ok(Init::Recursive),
        _ => Err("init must be random or recursive".into()),
    }
}

fn init_name(init: Init) -> &'static str {
    match init {
        Init::Random => "random",
        Init::Recursive => "recursive",
        _ => unreachable!("profile_transform only supports random and recursive init"),
    }
}

/// The complete fit configuration; snapshots store these parameters instead of the config.
fn fit_config(opts: &Options) -> UMAPConfig {
    UMAPConfig {
        n_neighbors: opts.neighbors,
        n_components: opts.components,
        n_epochs: Some(opts.epochs),
        init: opts.init,
        optimizer: opts.optimizer,
        random_state: Some(opts.fit_seed),
        transform_seed: TRANSFORM_SEED,
        transform_queue_size: opts.queue_size,
        ..UMAPConfig::default()
    }
}

/// Training then query rows from one generator stream, as in every prior run.
fn inputs(opts: &Options) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut state = DATA_SEED;
    let training = data(opts.train_rows, opts.features, &mut state);
    let query = data(opts.query_rows, opts.features, &mut state);
    (training, query)
}

fn file_entry(dir: &Path, name: &str, bytes: &[u8]) -> Result<serde_json::Value, Box<dyn Error>> {
    std::fs::write(dir.join(name), bytes)?;
    Ok(serde_json::json!({"bytes": bytes.len(), "fnv1a64": fnv1a64(bytes)}))
}

fn save_snapshot(
    dir: &Path,
    opts: &Options,
    state: &ModelState,
    training: &[Vec<f32>],
    query: &[Vec<f32>],
) -> Result<String, Box<dyn Error>> {
    if state.config != fit_config(opts) {
        return Err("exported config differs from the reconstructible fit config".into());
    }
    std::fs::create_dir(dir)?;
    let mut files = serde_json::Map::new();
    files.insert(
        "training_features.f32le".into(),
        file_entry(
            dir,
            "training_features.f32le",
            &f32_bytes(&state.training_features),
        )?,
    );
    files.insert(
        "embedding.f32le".into(),
        file_entry(dir, "embedding.f32le", &f32_bytes(&state.embedding))?,
    );
    files.insert(
        "query.f32le".into(),
        file_entry(dir, "query.f32le", &f32_bytes(query.iter().flatten()))?,
    );
    if let Some(graph) = &state.query_graph {
        let bytes: Vec<u8> = graph.iter().flat_map(|v| v.to_le_bytes()).collect();
        files.insert(
            "query_graph.i32le".into(),
            file_entry(dir, "query_graph.i32le", &bytes)?,
        );
    }
    if let Some(map) = &state.original_to_training {
        let bytes: Vec<u8> = map.iter().flat_map(|&v| (v as u64).to_le_bytes()).collect();
        files.insert(
            "original_to_training.u64le".into(),
            file_entry(dir, "original_to_training.u64le", &bytes)?,
        );
    }
    let manifest = serde_json::json!({
        "format": SNAPSHOT_FORMAT,
        "parameters": {"backend": opts.backend, "train_rows": opts.train_rows, "query_rows": opts.query_rows,
            "features": opts.features, "components": opts.components, "neighbors": opts.neighbors,
            "epochs": opts.epochs, "optimizer": optimizer_name(opts.optimizer), "fit_seed": opts.fit_seed,
            "fit_queue_size": opts.queue_size, "transform_seed": TRANSFORM_SEED, "dataset_seed": DATA_SEED,
            "search": "approximate", "initialization": init_name(opts.init)},
        "state": {"requested_search": search_name(state.requested_search),
            "resolved_search": search_name(state.resolved_search), "a_bits": state.a.to_bits(),
            "b_bits": state.b.to_bits(), "training_rows": state.training_rows, "n_features": state.n_features,
            "embedding_rows": state.embedding_rows, "n_components": state.n_components,
            "query_graph_width": state.query_graph_width},
        "inputs": {"training_fnv1a64": rows_checksum(training), "query_fnv1a64": rows_checksum(query)},
        "files": files,
    });
    let bytes = serde_json::to_vec_pretty(&manifest)?;
    std::fs::write(dir.join("manifest.json"), &bytes)?;
    Ok(fnv1a64(&bytes))
}

fn read_checked(
    dir: &Path,
    manifest: &serde_json::Value,
    name: &str,
) -> Result<Option<Vec<u8>>, Box<dyn Error>> {
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

fn chunks4(bytes: &[u8]) -> impl Iterator<Item = [u8; 4]> + '_ {
    bytes.chunks_exact(4).map(|c| c.try_into().unwrap())
}

/// Load a snapshot; fit-defining options are replaced by the manifest parameters.
/// Returns the model state (with the CLI queue size) and the manifest checksum.
fn load_snapshot(dir: &Path, opts: &mut Options) -> Result<(ModelState, String), Box<dyn Error>> {
    let manifest_bytes = std::fs::read(dir.join("manifest.json"))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest["format"] != SNAPSHOT_FORMAT {
        return Err("unsupported snapshot format".into());
    }
    let p = &manifest["parameters"];
    let number = |v: &serde_json::Value, key: &str| -> Result<u64, Box<dyn Error>> {
        v[key]
            .as_u64()
            .ok_or_else(|| format!("snapshot field {key} missing").into())
    };
    if p["backend"] != opts.backend.as_str() {
        return Err("snapshot was fitted on a different backend".into());
    }
    let mut conflict = Vec::new();
    let mut take = |flag: &str, slot: &mut usize, key: &str| -> Result<(), Box<dyn Error>> {
        let value = usize::try_from(number(p, key)?)?;
        if opts.explicit.contains(flag) && *slot != value {
            conflict.push(flag.to_string());
        }
        *slot = value;
        Ok(())
    };
    take("--train-rows", &mut opts.train_rows, "train_rows")?;
    take("--features", &mut opts.features, "features")?;
    take("--components", &mut opts.components, "components")?;
    take("--neighbors", &mut opts.neighbors, "neighbors")?;
    take("--epochs", &mut opts.epochs, "epochs")?;
    let saved_query_rows = usize::try_from(number(p, "query_rows")?)?;
    if !opts.explicit.contains("--query-rows") {
        opts.query_rows = saved_query_rows;
    }
    let fit_seed = number(p, "fit_seed")?;
    if opts.explicit.contains("--fit-seed") && opts.fit_seed != fit_seed {
        conflict.push("--fit-seed".into());
    }
    opts.fit_seed = fit_seed;
    let optimizer = match p["optimizer"].as_str() {
        Some("compatibility") => Optimizer::Compatibility,
        Some("adam") => Optimizer::Adam,
        Some("momentum") => Optimizer::Momentum,
        _ => return Err("snapshot optimizer invalid".into()),
    };
    if opts.explicit.contains("--optimizer") && opts.optimizer != optimizer {
        conflict.push("--optimizer".into());
    }
    opts.optimizer = optimizer;
    let init = parse_init(
        p["initialization"]
            .as_str()
            .ok_or("snapshot initialization missing")?,
    )?;
    if opts.explicit.contains("--init") && opts.init != init {
        conflict.push("--init".into());
    }
    opts.init = init;
    if !conflict.is_empty() {
        return Err(format!("flags conflict with snapshot parameters: {conflict:?}").into());
    }
    if opts.neighbors >= opts.train_rows
        || opts
            .train_rows
            .checked_add(opts.query_rows)
            .and_then(|n| n.checked_mul(opts.features))
            .is_none()
        || opts.query_rows.checked_mul(opts.components).is_none()
    {
        return Err("snapshot dimensions are invalid".into());
    }
    let s = &manifest["state"];
    let f32s = |name: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        let bytes =
            read_checked(dir, &manifest, name)?.ok_or(format!("snapshot missing {name}"))?;
        Ok(chunks4(&bytes).map(f32::from_le_bytes).collect())
    };
    let training_features = f32s("training_features.f32le")?;
    let embedding = f32s("embedding.f32le")?;
    let query_graph = read_checked(dir, &manifest, "query_graph.i32le")?
        .map(|bytes| chunks4(&bytes).map(i32::from_le_bytes).collect());
    let original_to_training = read_checked(dir, &manifest, "original_to_training.u64le")?
        .map(|bytes| {
            bytes
                .chunks_exact(8)
                .map(|c| usize::try_from(u64::from_le_bytes(c.try_into().unwrap())))
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let as_f32 = |key: &str| -> Result<f32, Box<dyn Error>> {
        Ok(f32::from_bits(u32::try_from(number(s, key)?)?))
    };
    let as_usize =
        |key: &str| -> Result<usize, Box<dyn Error>> { Ok(usize::try_from(number(s, key)?)?) };
    let mut config = fit_config(opts);
    config.transform_queue_size = opts.queue_size;
    let state = ModelState {
        config,
        requested_search: parse_search(s["requested_search"].as_str().unwrap_or(""))?,
        resolved_search: parse_search(s["resolved_search"].as_str().unwrap_or(""))?,
        a: as_f32("a_bits")?,
        b: as_f32("b_bits")?,
        training_features,
        training_rows: as_usize("training_rows")?,
        n_features: as_usize("n_features")?,
        embedding,
        embedding_rows: as_usize("embedding_rows")?,
        n_components: as_usize("n_components")?,
        query_graph,
        query_graph_width: as_usize("query_graph_width")?,
        original_to_training,
    };
    // Guard against generator drift between the saving and loading binaries.
    let (training, query) = inputs(opts);
    if manifest["inputs"]["training_fnv1a64"].as_str() != Some(rows_checksum(&training).as_str())
        || (opts.query_rows == saved_query_rows
            && manifest["inputs"]["query_fnv1a64"].as_str() != Some(rows_checksum(&query).as_str()))
    {
        return Err("regenerated inputs do not match snapshot input checksums".into());
    }
    Ok((state, fnv1a64(&manifest_bytes)))
}

#[cfg(feature = "tracing")]
struct OpenSpan {
    name: String,
    start: f64,
    run: Option<usize>,
    stage: bool,
}

// Chrome B/E events form per-thread stacks. Only estimator spans beneath a
// transform marker count; fit stages and unrelated identically named spans do not.
#[cfg(feature = "tracing")]
fn stages(
    events: &[serde_json::Value],
    expected: usize,
) -> Result<Vec<BTreeMap<String, f64>>, Box<dyn Error>> {
    let mut stacks: HashMap<(u64, u64), Vec<OpenSpan>> = HashMap::new();
    let mut result: Vec<BTreeMap<String, f64>> = Vec::new();
    for event in events {
        let phase = event["ph"].as_str().unwrap_or("");
        if phase != "B" && phase != "E" {
            continue;
        }
        let name = event["name"].as_str().ok_or("trace span missing name")?;
        let ts = event["ts"].as_f64().ok_or("trace span missing timestamp")?;
        let key = (
            event["pid"].as_u64().ok_or("trace span missing pid")?,
            event["tid"].as_u64().ok_or("trace span missing tid")?,
        );
        let stack = stacks.entry(key).or_default();
        if phase == "B" {
            let parent_run = stack.iter().rev().find_map(|span| span.run);
            let run = if name == "profile_transform_run" {
                if parent_run.is_some() {
                    return Err("nested transform markers".into());
                }
                result.push(BTreeMap::new());
                Some(result.len() - 1)
            } else {
                parent_run
            };
            let stage = run.is_some()
                && STAGES.contains(&name)
                && event["cat"]
                    .as_str()
                    .is_some_and(|cat| cat.starts_with("umap::standard_umap::estimator"));
            stack.push(OpenSpan {
                name: name.to_string(),
                start: ts,
                run,
                stage,
            });
        } else {
            let open = stack.pop().ok_or("unbalanced trace: unexpected end")?;
            if open.name != name || ts < open.start {
                return Err("unbalanced trace: mismatched span or timestamp".into());
            }
            if open.stage {
                let entry = result[open.run.ok_or("stage missing run")?]
                    .entry(open.name)
                    .or_insert(0.0);
                *entry += (ts - open.start) / 1_000_000.0;
            }
        }
    }
    if stacks.values().any(|stack| !stack.is_empty()) || result.len() != expected {
        return Err("unbalanced trace or unexpected transform marker count".into());
    }
    for run in &result {
        if STAGES.iter().any(|stage| !run.contains_key(*stage)) {
            return Err("missing estimator transform stage span; tracing must be enabled".into());
        }
    }
    Ok(result)
}

fn mode_sequence(warm_repeats: usize, paired: bool, single_mode: bool) -> Vec<(bool, usize)> {
    let mut sequence = Vec::new();
    for repeat in 0..=warm_repeats {
        if paired {
            let first = repeat % 2 == 1;
            sequence.push((first, repeat));
            sequence.push((!first, repeat));
        } else {
            sequence.push((single_mode, repeat));
        }
    }
    sequence
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut opts = options()?;
    let backend = match opts.backend.as_str() {
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => unreachable!(),
    };
    #[cfg_attr(not(feature = "tracing"), allow(unused_variables))]
    let (device, _) = cuscan_common::selected_device(backend)?;
    // Snapshot reading, validation and input generation stay outside every timer.
    let loaded = match opts.load_snapshot.clone() {
        Some(dir) => Some(load_snapshot(&dir, &mut opts)?),
        None => None,
    };
    std::fs::create_dir(&opts.output_dir)?;
    let (training, query) = inputs(&opts);
    #[cfg(feature = "tracing")]
    let trace_path = opts.output_dir.join("trace.json");
    #[cfg(feature = "tracing")]
    let (barrier_state, barriers_enabled, guard) = {
        let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(&trace_path)
            .include_args(true)
            .build();
        let barrier_state = Arc::new(Mutex::new(BarrierState::default()));
        let barriers_enabled = Arc::new(AtomicBool::new(false));
        let barrier_layer = BarrierLayer {
            device,
            state: Arc::clone(&barrier_state),
            enabled: Arc::clone(&barriers_enabled),
        };
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new(
                    "warn,umap=info,profile_transform=info",
                ))
                .with(layer)
                .with(barrier_layer),
        )?;
        (barrier_state, barriers_enabled, guard)
    };
    let (fitted, fit_seconds, load_seconds, snapshot) = if let Some((state, manifest_hash)) = loaded
    {
        let start = Instant::now();
        let fitted = FittedUMAP::from_state(state, backend)?;
        let load_seconds = start.elapsed().as_secs_f64();
        let snapshot = serde_json::json!({"mode": "loaded", "path": opts.load_snapshot, "manifest_fnv1a64": manifest_hash});
        (fitted, None, Some(load_seconds), snapshot)
    } else {
        let model = UMAP::new(fit_config(&opts), backend, NeighborSearch::Approximate);
        let (fitted, fit_seconds) = {
            #[cfg(feature = "tracing")]
            let _marker = tracing::info_span!("profile_fit").entered();
            let start = Instant::now();
            let fitted = model.fit(&training)?;
            (fitted, start.elapsed().as_secs_f64())
        };
        let snapshot = match &opts.save_snapshot {
            Some(dir) => {
                let hash = save_snapshot(dir, &opts, &fitted.export_state()?, &training, &query)?;
                serde_json::json!({"mode": "saved", "path": dir, "manifest_fnv1a64": hash})
            }
            None => serde_json::json!({"mode": "fit"}),
        };
        (fitted, Some(fit_seconds), None, snapshot)
    };
    validate_embedding(fitted.embedding(), opts.train_rows, opts.components)?;
    let parameters = serde_json::json!({"backend": opts.backend, "train_rows": opts.train_rows, "query_rows": opts.query_rows,
        "features": opts.features, "components": opts.components, "neighbors": opts.neighbors,
        "fit_epochs": opts.epochs, "transform_epochs": opts.epochs / 3, "queue_size": opts.queue_size,
        "warm_repeats": opts.warm_repeats, "stage_barriers": opts.stage_barriers, "paired_modes": opts.paired_modes, "dataset_seed": DATA_SEED, "fit_seed": opts.fit_seed,
        "optimizer": optimizer_name(opts.optimizer), "transform_seed": TRANSFORM_SEED, "search": "approximate", "initialization": init_name(opts.init)});
    let common = serde_json::json!({"build": if TRACING { "tracing" } else { "wall_only" },
        "snapshot": snapshot, "load_wall_seconds": load_seconds,
        "training_input_fnv1a64": rows_checksum(&training), "query_input_fnv1a64": rows_checksum(&query)});
    if opts.snapshot_only {
        #[cfg(feature = "tracing")]
        drop(guard);
        let summary = serde_json::json!({"parameters": parameters, "fit_wall_seconds": fit_seconds, "run": common, "transforms": []});
        std::fs::write(
            opts.output_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
        return Ok(());
    }
    let mut timings = Vec::new();
    let mut hashes = Vec::new();
    let modes = mode_sequence(opts.warm_repeats, opts.paired_modes, opts.stage_barriers);
    for (_run, &(_stage_barriers, _repeat)) in modes.iter().enumerate() {
        #[cfg(feature = "tracing")]
        barriers_enabled.store(_stage_barriers, Ordering::SeqCst);
        let (embedding, elapsed) = {
            #[cfg(feature = "tracing")]
            let _marker = tracing::info_span!(
                "profile_transform_run",
                run = _run,
                repeat = _repeat,
                stage_barriers = _stage_barriers
            )
            .entered();
            let start = Instant::now();
            let embedding = fitted.transform(&query)?;
            (embedding, start.elapsed().as_secs_f64())
        };
        timings.push(elapsed);
        hashes.push(validate_embedding(
            &embedding,
            opts.query_rows,
            opts.components,
        )?);
    }
    #[cfg(feature = "tracing")]
    let (runs, stage_measurement, timing_notes) = {
        drop(guard);
        let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&trace_path)?)?;
        let trace_runs = stages(&events, timings.len())?;
        let barrier_data = barrier_state.lock().unwrap();
        if !barrier_data.errors.is_empty()
            || !barrier_data.active.is_empty()
            || !barrier_data.markers.is_empty()
            || barrier_data.runs.len() != timings.len()
        {
            return Err(format!("invalid barrier measurements: {:?}", barrier_data.errors).into());
        }
        let mut runs = Vec::new();
        for (index, ((total, hash), trace_stage)) in
            timings.iter().zip(&hashes).zip(&trace_runs).enumerate()
        {
            let (stage_barriers, repeat) = modes[index];
            if barrier_data.runs[index].enabled != stage_barriers {
                return Err("barrier mode does not match transform marker".into());
            }
            let stage = if stage_barriers {
                &barrier_data.runs[index].stages
            } else {
                trace_stage
            };
            if STAGES.iter().any(|name| !stage.contains_key(*name)) {
                return Err("missing stage measurement".into());
            }
            let sync_seconds = stage_barriers.then(|| barrier_data.runs[index].sync_seconds);
            let remainder = total - stage.values().sum::<f64>();
            println!(
                "{} mode={} total={:.6}s neighbor_search={:.6}s fuzzy_graph={:.6}s optimization={:.6}s remainder={:.6}s sync={:?} hash={}",
                if repeat == 0 { "first" } else { "warm" },
                if stage_barriers {
                    "barrier_instrumented"
                } else {
                    "enqueue_spans"
                },
                total,
                stage["neighbor_search"],
                stage["fuzzy_graph"],
                stage["optimization"],
                remainder,
                sync_seconds,
                hash
            );
            runs.push(serde_json::json!({"run": index, "kind": if repeat == 0 { "first" } else { "warm" }, "mode_repeat": repeat, "process_first": index == 0, "stage_barriers": stage_barriers, "stage_measurement": if stage_barriers { "device-synchronized stage durations collected by harness layer" } else { "Chrome enqueue spans, not completed device execution time" }, "wall_seconds": total, "stage_seconds": stage, "trace_span_seconds": trace_stage, "remainder_seconds": remainder, "sync_call_seconds": sync_seconds, "embedding_fnv1a64": hash}));
        }
        let stage_measurement = if opts.paired_modes {
            "per-transform; see each record"
        } else if opts.stage_barriers {
            "device-synchronized stage durations collected by harness layer"
        } else {
            "Chrome enqueue spans, not completed device execution time"
        };
        (
            runs,
            stage_measurement,
            "Tracing adds overhead; fit tracing also synchronizes. In barrier mode each stage syncs before its timer and before stopping; sync_call_seconds sums both calls and is not additive to stage durations: exit sync is included in stages, entry sync in remainder. Entry sync may wait for prior unspanned work. Wall time includes public API input flattening and output readback, excluding validation. Remainder is wall time minus stage durations, not an initialization measurement; small negative values may reflect trace/timer overhead. Without barriers spans include any incidental synchronization but are not reliable completed execution costs. Hashes are informational, not determinism assertions.",
        )
    };
    #[cfg(not(feature = "tracing"))]
    let (runs, stage_measurement, timing_notes) = {
        let mut runs = Vec::new();
        for (index, (total, hash)) in timings.iter().zip(&hashes).enumerate() {
            let repeat = modes[index].1;
            println!(
                "{} mode=wall_only total={:.6}s hash={}",
                if repeat == 0 { "first" } else { "warm" },
                total,
                hash
            );
            runs.push(serde_json::json!({"run": index, "kind": if repeat == 0 { "first" } else { "warm" }, "mode_repeat": repeat, "process_first": index == 0, "stage_barriers": false, "stage_measurement": "none (wall-only build without tracing)", "wall_seconds": total, "embedding_fnv1a64": hash}));
        }
        (
            runs,
            "none (wall-only build without tracing)",
            "Wall-only build: the product tracing feature is disabled, so there are no spans or stage synchronization. Wall time covers the public transform call including input flattening and output readback, excluding validation. Hashes are informational, not determinism assertions.",
        )
    };
    let summary = serde_json::json!({
        "parameters": parameters,
        "fit_wall_seconds": fit_seconds, "run": common, "transforms": runs,
        "stage_measurement": stage_measurement,
        "mode_order": modes.iter().map(|(barriers, repeat)| serde_json::json!({"stage_barriers": barriers, "mode_repeat": repeat})).collect::<Vec<_>>(),
        "timing_notes": timing_notes
    });
    std::fs::write(
        opts.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tracing")]
    use super::stages;
    use super::{DATA_SEED, data, mode_sequence, validate_embedding};
    #[cfg(feature = "tracing")]
    use serde_json::{Value, json};

    #[cfg(feature = "tracing")]
    fn event(name: &str, category: &str, phase: &str, timestamp: u64) -> Value {
        json!({"name": name, "cat": category, "ph": phase, "ts": timestamp, "pid": 1, "tid": 1})
    }

    #[cfg(feature = "tracing")]
    fn span(events: &mut Vec<Value>, name: &str, category: &str, start: u64, end: u64) {
        events.push(event(name, category, "B", start));
        events.push(event(name, category, "E", end));
    }

    #[cfg(feature = "tracing")]
    const ESTIMATOR: &str = "umap::standard_umap::estimator";

    #[test]
    fn paired_modes_alternate_order_and_keep_per_mode_repeat_indices() {
        assert_eq!(
            mode_sequence(3, true, true),
            vec![
                (false, 0),
                (true, 0),
                (true, 1),
                (false, 1),
                (false, 2),
                (true, 2),
                (true, 3),
                (false, 3),
            ]
        );
        assert_eq!(
            mode_sequence(3, false, true),
            vec![(true, 0), (true, 1), (true, 2), (true, 3)]
        );
        assert_eq!(
            mode_sequence(3, false, false),
            vec![(false, 0), (false, 1), (false, 2), (false, 3)]
        );
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn parser_excludes_fit_and_attributes_nested_stages_to_each_run() {
        let mut events = vec![event("profile_fit", "profile_transform", "B", 0)];
        span(&mut events, "neighbor_search", ESTIMATOR, 1, 900_001);
        events.push(event("profile_fit", "profile_transform", "E", 900_002));
        for (start, factor) in [(1_000_000, 1), (10_000_000, 2)] {
            events.push(event(
                "profile_transform_run",
                "profile_transform",
                "B",
                start,
            ));
            events.push(event("nested", "umap", "B", start + 1));
            span(
                &mut events,
                "neighbor_search",
                ESTIMATOR,
                start + 2,
                start + 2 + factor * 100_000,
            );
            span(
                &mut events,
                "neighbor_search",
                "other",
                start + 300_000,
                start + 400_000,
            );
            span(
                &mut events,
                "fuzzy_graph",
                ESTIMATOR,
                start + 500_000,
                start + 500_000 + factor * 100_000,
            );
            span(
                &mut events,
                "optimization",
                ESTIMATOR,
                start + 800_000,
                start + 800_000 + factor * 100_000,
            );
            events.push(event("nested", "umap", "E", start + 1_100_000));
            events.push(event(
                "profile_transform_run",
                "profile_transform",
                "E",
                start + 1_200_000,
            ));
        }
        let runs = stages(&events, 2).unwrap();
        for name in ["neighbor_search", "fuzzy_graph", "optimization"] {
            assert!((runs[0][name] - 0.1).abs() < 1e-12);
            assert!((runs[1][name] - 0.2).abs() < 1e-12);
        }
        assert!(stages(&events, 1).is_err());
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn parser_rejects_unbalanced_and_malformed_spans() {
        let begin = event("profile_transform_run", "profile_transform", "B", 10);
        assert!(stages(&[begin.clone()], 1).is_err());
        assert!(
            stages(
                &[event("profile_transform_run", "profile_transform", "E", 10)],
                0
            )
            .is_err()
        );
        assert!(
            stages(
                &[begin.clone(), event("other", "profile_transform", "E", 11)],
                1
            )
            .is_err()
        );
        assert!(
            stages(
                &[
                    begin.clone(),
                    event("profile_transform_run", "profile_transform", "E", 9)
                ],
                1
            )
            .is_err()
        );
        assert!(
            stages(
                &[begin, json!({"ph": "B", "pid": 1, "tid": 1, "ts": 11})],
                1
            )
            .is_err()
        );
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn parser_rejects_missing_transform_stage_even_when_fit_contains_it() {
        let mut events = vec![event("profile_fit", "profile_transform", "B", 0)];
        span(&mut events, "optimization", ESTIMATOR, 1, 2);
        events.push(event("profile_fit", "profile_transform", "E", 3));
        events.push(event("profile_transform_run", "profile_transform", "B", 4));
        span(&mut events, "neighbor_search", ESTIMATOR, 5, 6);
        span(&mut events, "fuzzy_graph", ESTIMATOR, 7, 8);
        events.push(event("profile_transform_run", "profile_transform", "E", 9));
        assert!(
            stages(&events, 1)
                .unwrap_err()
                .to_string()
                .contains("missing estimator transform stage")
        );
    }

    #[test]
    fn generated_held_out_data_and_embedding_hashes_are_stable() {
        let mut seed = DATA_SEED;
        let train = data(4, 3, &mut seed);
        let query = data(4, 3, &mut seed);
        let mut repeated_seed = DATA_SEED;
        assert_eq!(train, data(4, 3, &mut repeated_seed));
        assert_eq!(query, data(4, 3, &mut repeated_seed));
        assert_ne!(train, query);
        let hash = validate_embedding(&query, 4, 3).unwrap();
        assert_eq!(hash, validate_embedding(&query, 4, 3).unwrap());
        assert_ne!(hash, validate_embedding(&train, 4, 3).unwrap());
        assert!(validate_embedding(&query, 3, 3).is_err());
        assert!(validate_embedding(&query, 4, 2).is_err());
        assert!(validate_embedding(&[vec![f32::NAN]], 1, 1).is_err());
    }
}
