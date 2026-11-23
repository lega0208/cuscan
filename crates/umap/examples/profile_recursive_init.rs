//! Synthetic 10M-node recursive-init profile. This is not an end-to-end fit.
//!
//! Run only after the real 10M fit has failed with the documented ANN admission
//! error. The input contains at least 10M rows of 32 little-endian f32 features.
//! Usage: profile_recursive_init --synthetic-10m FEATURES [--backend cpu|cuda]
//!        [--runs N] [--trace PATH]
//! Seed 11 and the default recursive configuration are fixed by the protocol.

use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    error::Error,
    hash::Hasher,
    io::{BufReader, Read},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use tracing_subscriber::layer::SubscriberExt;
use umap::{BackendChoice, standard_umap::profile_recursive_driver};
use umap_core::recursive_init::graph::HostCsr;

const N: usize = 10_000_000;
const FEATURES: usize = 32;
const BLOCK: usize = 10_000;
const MULTIPLIER: u64 = 2_654_435_761;

fn read_features(path: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let mut reader = BufReader::new(std::fs::File::open(path)?);
    let mut values = Vec::new();
    values.try_reserve_exact(N * FEATURES)?;
    let mut bytes = vec![0u8; 1024 * 1024];
    while values.len() < N * FEATURES {
        let count = (N * FEATURES - values.len()).min(bytes.len() / 4);
        reader.read_exact(&mut bytes[..count * 4])?;
        for chunk in bytes[..count * 4].as_chunks::<4>().0 {
            let value = f32::from_le_bytes(*chunk);
            if !value.is_finite() {
                return Err("synthetic features contain non-finite values".into());
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn target(row: usize) -> usize {
    ((row as u64 * MULTIPLIER + 1) % N as u64) as usize
}

/// Construct the fuzzy union directly, including coincident ring/hash edges.
fn synthetic_graph() -> Result<HostCsr, Box<dyn Error>> {
    let mut incoming = vec![usize::MAX; N];
    for row in 0..N {
        let col = target(row);
        if incoming[col] != usize::MAX {
            return Err("synthetic hash mapping is not a permutation".into());
        }
        incoming[col] = row;
    }
    let mut indptr = Vec::with_capacity(N + 1);
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    indices.try_reserve_exact(N * 16)?;
    weights.try_reserve_exact(N * 16)?;
    indptr.push(0);
    let mut edges = Vec::with_capacity(16);
    for (row, &incoming_row) in incoming.iter().enumerate().take(N) {
        let base = row / BLOCK * BLOCK;
        let local = row % BLOCK;
        // Each tuple contains destination and the forward/reverse membership.
        edges.clear();
        for offset in 1..=7 {
            let w = (-(offset as f32) / 4.0).exp();
            edges.push((base + (local + offset) % BLOCK, w, w));
            edges.push((base + (local + BLOCK - offset) % BLOCK, w, w));
        }
        edges.push((target(row), 0.05, 0.0));
        edges.push((incoming_row, 0.0, 0.05));
        edges.sort_unstable_by_key(|edge| edge.0);
        let mut i = 0;
        while i < edges.len() {
            let col = edges[i].0;
            let mut forward = 0.0f32;
            let mut reverse = 0.0f32;
            while i < edges.len() && edges[i].0 == col {
                // Duplicate directed memberships combine by fuzzy union.
                forward = forward + edges[i].1 - forward * edges[i].1;
                reverse = reverse + edges[i].2 - reverse * edges[i].2;
                i += 1;
            }
            let weight = forward + reverse - forward * reverse;
            // The graph maximum is at most 1, so every generated edge exceeds
            // the default 200-epoch threshold (maximum / 200).
            if weight > 1.0 / 200.0 {
                indices.push(col);
                weights.push(weight);
            }
        }
        indptr.push(indices.len());
    }
    Ok(HostCsr {
        n: N,
        indptr,
        indices,
        weights,
    })
}

fn graph_metadata(graph: &HostCsr) -> (String, BTreeMap<usize, usize>, usize) {
    // Hash canonical row offsets, indices and f32 bit patterns as little-endian
    // u64/u32 values. DefaultHasher is identified explicitly, not a SHA digest.
    let mut hash = DefaultHasher::new();
    hash.write(&(graph.n as u64).to_le_bytes());
    for &offset in &graph.indptr {
        hash.write(&(offset as u64).to_le_bytes());
    }
    for (&index, &weight) in graph.indices.iter().zip(&graph.weights) {
        hash.write(&(index as u64).to_le_bytes());
        hash.write(&weight.to_bits().to_le_bytes());
    }
    let mut histogram = BTreeMap::new();
    for row in 0..graph.n {
        *histogram
            .entry(graph.indptr[row + 1] - graph.indptr[row])
            .or_insert(0) += 1;
    }
    let mut seen = vec![false; graph.n];
    let mut stack = Vec::new();
    let mut components = 0;
    for root in 0..graph.n {
        if seen[root] {
            continue;
        }
        components += 1;
        seen[root] = true;
        stack.push(root);
        while let Some(row) = stack.pop() {
            for &col in &graph.indices[graph.indptr[row]..graph.indptr[row + 1]] {
                if !seen[col] {
                    seen[col] = true;
                    stack.push(col);
                }
            }
        }
    }
    (
        format!("default_hasher64:{:016x}", hash.finish()),
        histogram,
        components,
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut input = None;
    let mut backend = BackendChoice::Cpu;
    let mut runs = 1usize;
    let mut trace = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("{flag} requires a value"))?;
        match flag.as_str() {
            "--synthetic-10m" => input = Some(value),
            "--backend" => {
                backend = match value.as_str() {
                    "cpu" => BackendChoice::Cpu,
                    "cuda" => BackendChoice::Cuda { device_index: 0 },
                    _ => return Err(format!("unknown backend {value}").into()),
                }
            }
            "--runs" => runs = value.parse()?,
            "--trace" => trace = Some(value),
            _ => return Err(format!("unknown flag {flag}").into()),
        }
    }
    let input = input.ok_or("--synthetic-10m FEATURES is required")?;
    if runs == 0 {
        return Err("runs must be positive".into());
    }
    let data = read_features(&input)?;
    let graph = synthetic_graph()?;
    let (graph_hash, histogram, components) = graph_metadata(&graph);
    if components != 1 {
        return Err(format!("synthetic graph has {components} components, expected 1").into());
    }
    let trace = trace.unwrap_or_else(|| {
        std::env::temp_dir()
            .join(format!(
                "profile_recursive_init-{}.json",
                std::process::id()
            ))
            .display()
            .to_string()
    });
    let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(&trace)
        .include_args(true)
        .build();
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::new(
                "warn,umap=info,umap_core=info",
            ))
            .with(layer),
    )?;
    let mut seconds = Vec::with_capacity(runs);
    for run in 0..runs {
        eprintln!(
            "profile_run_start run={run} unix_ns={}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        let start = Instant::now();
        let embedding = profile_recursive_driver(&graph, &data, FEATURES, backend, 11)?;
        seconds.push(start.elapsed().as_secs_f64());
        eprintln!(
            "profile_run_end run={run} unix_ns={}",
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos()
        );
        if embedding.len() != N * 2 || embedding.iter().any(|value| !value.is_finite()) {
            return Err("invalid synthetic recursive embedding".into());
        }
    }
    drop(guard);
    println!(
        "{}",
        serde_json::json!({
            "workload": "synthetic-10m", "end_to_end_fit": false,
            "features_path": input, "rows": N, "features": FEATURES,
            "edges": graph.indices.len(), "graph_hash": graph_hash,
            "degree_histogram": histogram, "connected_components": components,
            "expected_active_parts": [2500000, 625000, 156250, 39062, 9765, 2441, 610],
            "backend": format!("{backend:?}"), "seed": 11, "runs": runs,
            "recursive_seconds": seconds, "trace": trace,
        })
    );
    Ok(())
}
