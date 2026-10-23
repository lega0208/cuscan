//! Persistent CUDA ANN construction worker for construction-only campaigns.
//!
//! Build with `cargo build --release --example ann_construction_worker`. Invoke as
//! `ann_construction_worker INPUT.f32le ROWS FEATURES [DEVICE_INDEX]`; it reads
//! one JSON command per stdin line and emits one JSON response per command.

use burn::{
    backend::Dispatch,
    tensor::{Int, Tensor, TensorData},
};
use fast_umap::backend::StandardUmapOps;
use serde_json::{Value, json};
use std::{
    error::Error,
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

fn read_features(path: &Path, rows: usize, features: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let expected = rows
        .checked_mul(features)
        .and_then(|size| size.checked_mul(4));
    if rows == 0 || features == 0 || expected != Some(bytes.len()) {
        return Err(format!(
            "invalid f32le matrix dimensions or file length: {}",
            path.display()
        )
        .into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|part| f32::from_le_bytes(part.try_into().expect("4-byte chunk")))
        .collect())
}

fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

fn write_graph(prefix: &Path, ids: &[i32], distances: &[f32]) -> Result<(), Box<dyn Error>> {
    let ids_path = PathBuf::from(format!("{}.ids.i32le", prefix.display()));
    let distances_path = PathBuf::from(format!("{}.distances.f32le", prefix.display()));
    std::fs::write(
        ids_path,
        ids.iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    std::fs::write(
        distances_path,
        distances
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

fn perform(
    command: &Value,
    host_features: &[f32],
    rows: usize,
    features: usize,
    device: &burn::tensor::Device,
    is_warmup: bool,
) -> Result<Value, Box<dyn Error>> {
    let k: u32 = command["k"]
        .as_u64()
        .ok_or("command k must be an unsigned integer")?
        .try_into()?;
    let seed = command["seed"]
        .as_u64()
        .ok_or("command seed must be an unsigned integer")?;
    let cap: u32 = command["cap"]
        .as_u64()
        .ok_or("command cap must be an unsigned integer")?
        .try_into()?;
    if k == 0 || k as usize >= rows {
        return Err(format!("k must be in 1..{rows}").into());
    }

    let started = Instant::now();
    let input = Tensor::<2>::from_data(
        TensorData::new(host_features.to_vec(), [rows, features]),
        device,
    );
    let (indices, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(input.into_dispatch(), k, seed, cap);
    let ids = Tensor::<2, Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()?;
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()?;
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    if ids.len() != rows * k as usize || distances.len() != ids.len() {
        return Err(format!(
            "materialized graph has {} IDs and {} distances; expected {}",
            ids.len(),
            distances.len(),
            rows * k as usize
        )
        .into());
    }
    for row in 0..rows {
        let start = row * k as usize;
        for offset in 0..k as usize {
            let id = ids[start + offset];
            if id < 0
                || id as usize >= rows
                || id as usize == row
                || !distances[start + offset].is_finite()
                || distances[start + offset] < 0.0
            {
                return Err(format!("invalid host graph entry at ({row}, {offset})").into());
            }
            if ids[start..start + offset].contains(&id) {
                return Err(format!("duplicate host neighbor ID at ({row}, {offset})").into());
            }
            if offset > 0 && distances[start + offset] < distances[start + offset - 1] {
                return Err(
                    format!("host graph row {row} has descending materialized distances").into(),
                );
            }
        }
    }

    let id_hash = fnv1a(ids.iter().flat_map(|value| value.to_le_bytes()));
    let distance_hash = fnv1a(distances.iter().flat_map(|value| value.to_le_bytes()));
    let graph_files = if let Some(path) = command["graph_prefix"].as_str() {
        let prefix = Path::new(path);
        write_graph(prefix, &ids, &distances)?;
        Some(json!({
            "ids": format!("{}.ids.i32le", prefix.display()),
            "distances": format!("{}.distances.f32le", prefix.display()),
        }))
    } else {
        None
    };

    Ok(json!({
        "ok": true,
        "kind": if is_warmup { "warmup" } else { "build" },
        "elapsed_ms": elapsed_ms,
        "rows": rows,
        "features": features,
        "k_nonself": k,
        "seed": seed,
        "cap": cap,
        "ids_hash_fnv1a64": id_hash,
        "distances_hash_fnv1a64": distance_hash,
        "graph_files": graph_files,
        "timer": "host before feature tensor creation/upload through host IDs and distances materialization; excludes graph file I/O",
        "compile_signal": {
            "status": "cubecl_config_compilation_log",
            "config_path": std::env::current_dir().ok().map(|cwd| cwd.join("cubecl.toml")),
            "record_marker": "[START_KERNEL_COMPILATION]",
            "detail": "CubeCL 0.11 cubecl.toml config enables compilation.logger.level=full and writes one marker record per newly compiled kernel. profiling.logger.level is disabled independently.",
            "logging_cost": "Full compilation formatting, including source, occurs only on new compilation events; no compilation record is emitted for an already compiled kernel. Profiling output is disabled."
        }
    }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "usage: ann_construction_worker INPUT.f32le ROWS FEATURES [DEVICE_INDEX]".into(),
        );
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let device_index: usize = args.get(4).map_or(Ok(0), |value| value.parse())?;
    let host_features = read_features(Path::new(&args[1]), rows, features)?;
    let device = fast_umap::cuda_backend::device(device_index).ok_or_else(|| {
        format!("CUDA device {device_index} unavailable; CPU fallback is forbidden")
    })?;

    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let response = match serde_json::from_str::<Value>(&line) {
            Ok(command) => {
                let op = command["op"].as_str().unwrap_or("");
                if op != "build" && op != "warmup" {
                    json!({ "ok": false, "error": "op must be build or warmup" })
                } else {
                    match perform(
                        &command,
                        &host_features,
                        rows,
                        features,
                        &device,
                        op == "warmup",
                    ) {
                        Ok(result) => result,
                        Err(error) => json!({ "ok": false, "error": error.to_string() }),
                    }
                }
            }
            Err(error) => json!({ "ok": false, "error": format!("invalid JSON command: {error}") }),
        };
        serde_json::to_writer(&mut stdout, &response)?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
