//! CUDA-only companion to scripts/compare_umap_spans.py. Build before timing.
//!
//! usage: profile_standard_cuda INPUT_RAW ROWS FEATURES EPOCHS OUT_DIR
//!        --search exact|approximate [--min-dist D] [--warmups W] [--runs R]
//!        [--neighbors K] [--init random|spectral] [--seed S]
//!        [--query INPUT_RAW --query-rows Q]... [--trace PATH]
//!
//! Each timed run fits once and transforms every supplied query batch with
//! that fitted model. Fit/transform timers include host result materialization;
//! diagnostic graph exports are outside those timers. Stage spans include
//! explicit device barriers; they are synchronized wall times, not GPU events.
use burn::{
    backend::Dispatch,
    tensor::{Tensor, TensorData},
};
use cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};
use fast_umap::{
    BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig, backend::StandardUmapOps,
    standard_umap::approximate_query_budget,
};
use std::{
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing_subscriber::layer::SubscriberExt;

#[path = "profile_standard_cuda/compilation.rs"]
mod compilation;

struct QueryInput {
    path: PathBuf,
    rows: usize,
}

fn read_matrix(path: &Path, rows: usize, features: usize) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let expected = rows.checked_mul(features).and_then(|n| n.checked_mul(4));
    if rows == 0 || features == 0 || expected != Some(bytes.len()) {
        return Err(format!(
            "invalid float32 matrix shape or input length: {}",
            path.display()
        )
        .into());
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let matrix = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    Ok(matrix)
}

fn write_f32(path: PathBuf, values: &[f32]) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn write_i32(path: PathBuf, values: &[i32]) -> Result<(), Box<dyn Error>> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(path, bytes)?;
    Ok(())
}

fn query_graph(
    fitted: &fast_umap::standard_umap::FittedUMAP,
    training: &[Vec<f32>],
    query: &[Vec<f32>],
    neighbors: usize,
    search: NeighborSearch,
    seed: u64,
) -> Result<Vec<i32>, Box<dyn Error>> {
    let device = fast_umap::cuda_backend::device(0)
        .ok_or("CUDA device 0 unavailable; no CPU fallback permitted")?;
    let reference_rows = training.len();
    let features = fitted.n_features();
    let reference_values = training.iter().flatten().copied().collect::<Vec<_>>();
    let reference = Tensor::<2>::from_data(
        TensorData::new(reference_values, [reference_rows, features]),
        &device,
    );
    let (graph, graph_width) = fitted.diagnostic_fit_graph()?;
    let graph_tensor = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(graph, [reference_rows, graph_width]),
        &device,
    );
    let k = neighbors.min(reference_rows);
    let mut all_ids = Vec::with_capacity(query.len().saturating_mul(k));
    for start in (0..query.len()).step_by(64) {
        let end = (start + 64).min(query.len());
        let values = query[start..end]
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let block =
            Tensor::<2>::from_data(TensorData::new(values, [end - start, features]), &device);
        let ids = match search {
            NeighborSearch::Exact => {
                let (ids, _) = <Dispatch as StandardUmapOps>::exact_neighbors(
                    block.into_dispatch(),
                    reference.clone().into_dispatch(),
                    k as u32,
                    false,
                    u32::try_from(start)?,
                );
                ids
            }
            NeighborSearch::Approximate => {
                let budget = approximate_query_budget(reference_rows, k, 4.0)?;
                let (ids, _, _) = <Dispatch as StandardUmapOps>::approximate_query(
                    block.into_dispatch(),
                    reference.clone().into_dispatch(),
                    graph_tensor.clone().into_dispatch(),
                    k as u32,
                    budget.candidates as u32,
                    budget.visits as u32,
                    seed,
                    start as u64,
                );
                ids
            }
            NeighborSearch::Auto => return Err("Auto search is not supported".into()),
        };
        all_ids.extend(
            Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
                .to_data()
                .try_to_vec::<i32>()?,
        );
    }
    Ok(all_ids)
}

fn fit_stage_seconds(
    events: &[serde_json::Value],
) -> Result<std::collections::BTreeMap<String, Vec<f64>>, &'static str> {
    let mut starts = std::collections::HashMap::<(u64, String), Vec<f64>>::new();
    let mut stage_seconds = std::collections::BTreeMap::<String, Vec<f64>>::new();
    let mut measured = false;
    let mut in_fit = false;
    for event in events {
        match event["args"]["message"].as_str() {
            Some("measured_runs_begin") => {
                measured = true;
                continue;
            }
            Some("measured_fit_begin") if measured => {
                in_fit = true;
                continue;
            }
            Some("measured_fit_end") if measured => {
                in_fit = false;
                continue;
            }
            _ => {}
        }
        if !measured || !in_fit {
            continue;
        }
        let Some(name) = event["name"].as_str() else {
            continue;
        };
        if ![
            "neighbor_search",
            "fuzzy_graph",
            "schedule",
            "spectral_graph_export",
            "initialization",
            "optimization",
        ]
        .contains(&name)
        {
            continue;
        }
        let key = (event["tid"].as_u64().unwrap_or(0), name.to_string());
        let Some(timestamp) = event["ts"].as_f64() else {
            continue;
        };
        match event["ph"].as_str() {
            Some("B") => starts.entry(key).or_default().push(timestamp),
            Some("E") => {
                let begin = starts
                    .get_mut(&key)
                    .and_then(Vec::pop)
                    .ok_or("unbalanced stage trace")?;
                stage_seconds
                    .entry(name.to_string())
                    .or_default()
                    .push((timestamp - begin) / 1e6);
            }
            _ => {}
        }
    }
    Ok(stage_seconds)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        return Err("usage: profile_standard_cuda INPUT_RAW ROWS FEATURES EPOCHS OUT_DIR --search exact|approximate [--min-dist D] [--warmups W] [--runs R] [--neighbors K] [--init random|spectral] [--seed S] [--query INPUT_RAW --query-rows Q]... [--trace PATH]".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let epochs: usize = args[4].parse()?;
    let out_dir = PathBuf::from(&args[5]);
    let mut search = None;
    let mut warmups = 1usize;
    let mut runs = 1usize;
    let mut neighbors = 15usize;
    let mut init = Init::Spectral;
    let mut seed = 42u64;
    let mut min_dist = 0.0f32;
    let mut trace = None;
    let mut queries = Vec::<QueryInput>::new();
    let mut pending_query = None;
    let mut rest = args[6..].iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or(format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--search" => {
                search = Some(match value.as_str() {
                    "exact" => NeighborSearch::Exact,
                    "approximate" => NeighborSearch::Approximate,
                    other => return Err(format!("unknown search mode {other}").into()),
                })
            }
            "--warmups" => warmups = value.parse()?,
            "--runs" => runs = value.parse()?,
            "--neighbors" => neighbors = value.parse()?,
            "--seed" => seed = value.parse()?,
            "--min-dist" => min_dist = value.parse()?,
            "--init" => {
                init = match value.as_str() {
                    "random" => Init::Random,
                    "spectral" => Init::Spectral,
                    other => return Err(format!("unknown initialization {other}").into()),
                }
            }
            "--query" => {
                if pending_query.replace(PathBuf::from(value)).is_some() {
                    return Err(
                        "each --query must be followed by --query-rows before another query".into(),
                    );
                }
            }
            "--query-rows" => {
                let path = pending_query
                    .take()
                    .ok_or("--query-rows must follow --query PATH")?;
                queries.push(QueryInput {
                    path,
                    rows: value.parse()?,
                });
            }
            "--trace" => trace = Some(value.clone()),
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if pending_query.is_some() {
        return Err("--query requires --query-rows".into());
    }
    let search = search.ok_or("--search exact|approximate is required (no Auto)")?;
    if rows <= 15 || features == 0 || epochs == 0 || runs == 0 || neighbors < 2 || neighbors >= rows
    {
        return Err("invalid shape, epochs, runs, or neighbor count".into());
    }
    let data = read_matrix(Path::new(&args[1]), rows, features)?;
    let query_data = queries
        .iter()
        .map(|input| read_matrix(&input.path, input.rows, features))
        .collect::<Result<Vec<_>, _>>()?;
    std::fs::create_dir_all(&out_dir)?;
    let run_id = format!(
        "{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    );
    let environment_root = out_dir.join(format!("cubecl-environment-{run_id}"));
    std::fs::create_dir(&environment_root)?;
    let environment_name = format!("profile-standard-cuda-{run_id}");
    compilation::install_warning_capture()?;
    let _runtime_config = CubeClRuntimeConfig::get();
    cubecl::environment::set_root(&environment_root);
    cubecl::environment::activate(&environment_name);
    if fast_umap::cuda_backend::device(0).is_none() {
        return Err("CUDA device 0 unavailable; no CPU fallback permitted".into());
    }
    cubecl::records::configure(cubecl::records::RecordsConfig {
        level: cubecl::records::RecordLevel::Basic,
        keep_sessions: None,
    });
    let session_label = format!("profile_standard_cuda:{run_id}");
    cubecl::records::label(session_label.clone());
    let estimator = |run_seed: u64| {
        UMAP::new(
            UMAPConfig {
                n_neighbors: neighbors,
                n_components: 2,
                n_epochs: Some(epochs),
                min_dist,
                init,
                random_state: Some(run_seed),
                transform_seed: run_seed,
                ..UMAPConfig::default()
            },
            BackendChoice::Cuda { device_index: 0 },
            search,
        )
    };
    let trace = trace.unwrap_or_else(|| out_dir.join("stages.trace.json").display().to_string());
    let trace_path = trace.clone();
    let trace = Some(trace);
    let guard = match &trace {
        Some(path) => {
            let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
                .file(path)
                .include_args(true)
                .build();
            let subscriber = tracing_subscriber::registry()
                .with(tracing_subscriber::EnvFilter::new("info,fast_umap=trace"))
                .with(layer);
            tracing::subscriber::set_global_default(subscriber)?;
            Some(guard)
        }
        None => None,
    };
    let mut warmup_seconds = Vec::with_capacity(warmups);
    let mut compilation_phases = Vec::new();
    for warmup in 0..warmups {
        let phase = format!("warmup_{warmup}");
        let _mark = cubecl::records::Mark::new(phase.clone());
        let start = Instant::now();
        let fitted = estimator(1000 + warmup as u64).fit(&data)?;
        for query in &query_data {
            fitted.transform(query)?;
            query_graph(
                &fitted,
                &data,
                query,
                neighbors,
                search,
                1000 + warmup as u64,
            )?;
        }
        warmup_seconds.push(start.elapsed().as_secs_f64());
        compilation_phases.push(phase);
    }
    tracing::info!("measured_runs_begin");
    let mut fit_seconds = Vec::with_capacity(runs);
    let mut transform_seconds = Vec::with_capacity(runs);
    let mut fit_graph_width = 0usize;
    let mut query_graph_width = neighbors.min(rows);
    for run in 0..runs {
        let run_seed = seed + run as u64;
        let model = estimator(run_seed);
        let phase = format!("measured_fit_{run}");
        tracing::info!("measured_fit_begin");
        let fitted = {
            let _mark = cubecl::records::Mark::new(phase.clone());
            let start = Instant::now();
            let fitted = model.fit(&data)?;
            fit_seconds.push(start.elapsed().as_secs_f64());
            fitted
        };
        tracing::info!("measured_fit_end");
        compilation_phases.push(phase);
        let embedding: Vec<f32> = fitted.embedding().iter().flatten().copied().collect();
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err("CUDA fit produced non-finite embedding".into());
        }
        let (graph, width) = fitted.diagnostic_fit_graph()?;
        fit_graph_width = width;
        write_f32(out_dir.join(format!("embedding_{run}.f32le")), &embedding)?;
        write_i32(out_dir.join(format!("graph_{run}.i32le")), &graph)?;
        let mut run_transform_seconds = Vec::with_capacity(query_data.len());
        for (batch, query) in query_data.iter().enumerate() {
            let phase = format!("measured_transform_{run}_{batch}");
            let _mark = cubecl::records::Mark::new(phase.clone());
            let start = Instant::now();
            let embedding = fitted.transform(query)?;
            run_transform_seconds.push(start.elapsed().as_secs_f64());
            let embedding: Vec<f32> = embedding.iter().flatten().copied().collect();
            if embedding.iter().any(|value| !value.is_finite()) {
                return Err(
                    format!("CUDA transform {run}/{batch} produced non-finite embedding").into(),
                );
            }
            write_f32(
                out_dir.join(format!("transform_{run}_{batch}.f32le")),
                &embedding,
            )?;
            let graph = query_graph(&fitted, &data, query, neighbors, search, run_seed)?;
            query_graph_width = if query.is_empty() {
                0
            } else {
                neighbors.min(rows)
            };
            write_i32(
                out_dir.join(format!("query_graph_{run}_{batch}.i32le")),
                &graph,
            )?;
            compilation_phases.push(phase);
        }
        transform_seconds.push(run_transform_seconds);
    }
    drop(guard);
    let events: Vec<serde_json::Value> = serde_json::from_slice(&std::fs::read(&trace_path)?)?;
    let stage_seconds = fit_stage_seconds(&events)?;
    let spectral_fallback = events.iter().any(|event| {
        event["args"]["spectral_fallback"] == true || event["args"]["spectral_fallback"] == "true"
    });
    let approximate_iterations = ((rows as f64).log2().round() as usize).max(5);
    let compilation = compilation::capture(
        &environment_name,
        &session_label,
        &environment_root,
        &compilation_phases,
        warmups,
    );
    let result = serde_json::json!({
        "fit_seconds": fit_seconds,
        "stage_seconds": stage_seconds,
        "stage_seconds_scope": "fit_only",
        "synchronized_stages": true,
        "spectral_fallback": spectral_fallback,
        "warmup_seconds": warmup_seconds,
        "compilation": compilation,
        "min_dist": min_dist,
        "transform_seconds": transform_seconds,
        "search": format!("{search:?}"),
        "neighbors": neighbors,
        "init": format!("{init:?}"),
        "seed": seed,
        "fit_graph_width": fit_graph_width,
        "query_graph_width": query_graph_width,
        "approximate_iterations_rule": "max(round(log2(n)), 5)",
        "approximate_iterations": approximate_iterations,
        "warmups": warmups,
        "query_batches": query_data.len(),
    });
    std::fs::write(
        out_dir.join("result.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    if spectral_fallback {
        return Err("spectral fallback invalidates profiler benchmark (including warmup)".into());
    }
    println!("{result}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::fit_stage_seconds;

    #[test]
    fn transform_spans_do_not_contaminate_fit_stage_aggregates() {
        let events = [
            serde_json::json!({"args": {"message": "measured_runs_begin"}}),
            serde_json::json!({"args": {"message": "measured_fit_begin"}}),
            serde_json::json!({"name": "neighbor_search", "tid": 1, "ts": 1000.0, "ph": "B"}),
            serde_json::json!({"name": "neighbor_search", "tid": 1, "ts": 3000.0, "ph": "E"}),
            serde_json::json!({"args": {"message": "measured_fit_end"}}),
            serde_json::json!({"name": "neighbor_search", "tid": 1, "ts": 5000.0, "ph": "B"}),
            serde_json::json!({"name": "neighbor_search", "tid": 1, "ts": 15000.0, "ph": "E"}),
        ];

        let stages = fit_stage_seconds(&events).unwrap();

        assert_eq!(stages["neighbor_search"], vec![0.002]);
    }
}
