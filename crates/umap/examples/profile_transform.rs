//! Measure a single fit and repeated held-out transforms with synchronized tracing.
//!
//! Run with `cargo run -p umap --release --features tracing --example profile_transform --
//! --backend cpu --output-dir /tmp/transform-profile` (use `cuda` for strict CUDA).
//! Options: --train-rows N (1000), --query-rows N (128), --features N (16),
//! --components N (2), --neighbors N (15), --epochs N (200), --queue-size F
//! (4.0), --warm-repeats N (3, minimum 3), --stage-barriers true|false
//! (default true), --paired-modes true|false (default false), --output-dir PATH
//! (required). Paired mode alternates both modes on the same fitted model.
//! The output directory must not exist; its parent must already exist. Trace and
//! summary contain no input or embedding dumps. Stage tracing adds overhead and
//! estimator instrumentation may synchronize devices; these are not GPU event times.

#[cfg(not(feature = "tracing"))]
compile_error!("profile_transform requires --features tracing");

use std::{
    collections::{BTreeMap, HashMap},
    error::Error,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
use tracing::{Id, Subscriber};
use tracing_subscriber::{
    layer::{Context, Layer, SubscriberExt},
    registry::LookupSpan,
};
use umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

const STAGES: [&str; 3] = ["neighbor_search", "fuzzy_graph", "optimization"];
const DATA_SEED: u64 = 0x1234_5678_9abc_def0;
const MODEL_SEED: u64 = 42;

struct Options {
    backend: String,
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
}

fn options() -> Result<Options, Box<dyn Error>> {
    let mut backend = "cpu".to_string();
    let (mut train_rows, mut query_rows, mut features, mut components): (
        usize,
        usize,
        usize,
        usize,
    ) = (1000, 128, 16, 2);
    let (mut neighbors, mut epochs, mut queue_size, mut warm_repeats): (usize, usize, f64, usize) =
        (15, 200, 4.0, 3);
    let mut output_dir = None;
    let mut stage_barriers = true;
    let mut paired_modes = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            println!(
                "profile_transform --output-dir PATH [--backend cpu|cuda] [--train-rows N] [--query-rows N] [--features N] [--components N] [--neighbors N] [--epochs N] [--queue-size F] [--warm-repeats N] [--stage-barriers true|false] [--paired-modes true|false]\nRequires --features tracing; output directory must not exist."
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--backend" => backend = value,
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
            _ => return Err(format!("unknown option {flag}").into()),
        }
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
        || warm_repeats < 3
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
        return Err("invalid dimensions, neighbors, epochs (minimum 3), queue size, or warm repeats (minimum 3)".into());
    }
    Ok(Options {
        backend,
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
    })
}

#[derive(Default)]
struct BarrierRun {
    stages: BTreeMap<String, f64>,
    sync_seconds: f64,
    enabled: bool,
}

#[derive(Default)]
struct BarrierState {
    runs: Vec<BarrierRun>,
    markers: HashMap<Id, usize>,
    active: HashMap<Id, (usize, String, Instant)>,
    errors: Vec<String>,
}

struct BarrierLayer {
    device: burn::tensor::Device,
    state: Arc<Mutex<BarrierState>>,
    enabled: Arc<AtomicBool>,
}

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
    let mut hash = 0xcbf29ce484222325u64;
    for value in embedding.iter().flatten() {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("{hash:016x}"))
}

struct OpenSpan {
    name: String,
    start: f64,
    run: Option<usize>,
    stage: bool,
}

// Chrome B/E events form per-thread stacks. Only estimator spans beneath a
// transform marker count; fit stages and unrelated identically named spans do not.
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
    let opts = options()?;
    let backend = match opts.backend.as_str() {
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => unreachable!(),
    };
    let (device, _) = cuscan_common::selected_device(backend)?;
    std::fs::create_dir(&opts.output_dir)?;
    let mut state = DATA_SEED;
    let training = data(opts.train_rows, opts.features, &mut state);
    let query = data(opts.query_rows, opts.features, &mut state);
    let trace_path = opts.output_dir.join("trace.json");
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
    let model = UMAP::new(
        UMAPConfig {
            n_neighbors: opts.neighbors,
            n_components: opts.components,
            n_epochs: Some(opts.epochs),
            init: Init::Random,
            random_state: Some(MODEL_SEED),
            transform_seed: 42,
            transform_queue_size: opts.queue_size,
            ..UMAPConfig::default()
        },
        backend,
        NeighborSearch::Approximate,
    );
    let (fitted, fit_seconds) = {
        let _marker = tracing::info_span!("profile_fit").entered();
        let start = Instant::now();
        let fitted = model.fit(&training)?;
        (fitted, start.elapsed().as_secs_f64())
    };
    validate_embedding(fitted.embedding(), opts.train_rows, opts.components)?;
    let mut timings = Vec::new();
    let mut hashes = Vec::new();
    let modes = mode_sequence(opts.warm_repeats, opts.paired_modes, opts.stage_barriers);
    for (run, &(stage_barriers, repeat)) in modes.iter().enumerate() {
        barriers_enabled.store(stage_barriers, Ordering::SeqCst);
        let (embedding, elapsed) = {
            let _marker =
                tracing::info_span!("profile_transform_run", run, repeat, stage_barriers).entered();
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
    let summary = serde_json::json!({
        "parameters": {"backend": opts.backend, "train_rows": opts.train_rows, "query_rows": opts.query_rows,
            "features": opts.features, "components": opts.components, "neighbors": opts.neighbors,
            "fit_epochs": opts.epochs, "transform_epochs": opts.epochs / 3, "queue_size": opts.queue_size,
            "warm_repeats": opts.warm_repeats, "stage_barriers": opts.stage_barriers, "paired_modes": opts.paired_modes, "dataset_seed": DATA_SEED, "fit_seed": MODEL_SEED,
            "transform_seed": 42, "search": "approximate", "initialization": "random"},
        "fit_wall_seconds": fit_seconds, "transforms": runs,
        "stage_measurement": if opts.paired_modes { "per-transform; see each record" } else if opts.stage_barriers { "device-synchronized stage durations collected by harness layer" } else { "Chrome enqueue spans, not completed device execution time" },
        "mode_order": modes.iter().map(|(barriers, repeat)| serde_json::json!({"stage_barriers": barriers, "mode_repeat": repeat})).collect::<Vec<_>>(),
        "timing_notes": "Tracing adds overhead; fit tracing also synchronizes. In barrier mode each stage syncs before its timer and before stopping; sync_call_seconds sums both calls and is not additive to stage durations: exit sync is included in stages, entry sync in remainder. Entry sync may wait for prior unspanned work. Wall time includes public API input flattening and output readback, excluding validation. Remainder is wall time minus stage durations, not an initialization measurement; small negative values may reflect trace/timer overhead. Without barriers spans include any incidental synchronization but are not reliable completed execution costs. Hashes are informational, not determinism assertions."
    });
    std::fs::write(
        opts.output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{DATA_SEED, data, mode_sequence, stages, validate_embedding};
    use serde_json::{Value, json};

    fn event(name: &str, category: &str, phase: &str, timestamp: u64) -> Value {
        json!({"name": name, "cat": category, "ph": phase, "ts": timestamp, "pid": 1, "tid": 1})
    }

    fn span(events: &mut Vec<Value>, name: &str, category: &str, start: u64, end: u64) {
        events.push(event(name, category, "B", start));
        events.push(event(name, category, "E", end));
    }

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
