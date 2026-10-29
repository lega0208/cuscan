//! Scalar Phase 3 KD traversal timing; input is row-major little-endian float32.
use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use cubecl::Device as CubeDevice;
use hdbscan::{DeviceKdTree, HdbscanOps};
use hdbscan_core::kdtree::{KdTree, MAX_KD_LEVELS};
use rayon::prelude::*;
use std::{error::Error, hint::black_box, time::Instant};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn bound(data: &[f32], tree: &KdTree, row: usize, node: usize) -> f32 {
    let d = tree.n_features;
    let mut sum = 0.0f32;
    for feature in 0..d {
        let point = data[row * d + feature];
        let delta = (tree.bounds_min[node * d + feature] - point)
            .max(point - tree.bounds_max[node * d + feature])
            .max(0.0);
        sum += delta * delta;
    }
    sum
}

fn greater(distance_a: f32, index_a: i32, distance_b: f32, index_b: i32) -> bool {
    distance_a > distance_b || (distance_a == distance_b && index_a > index_b)
}

fn sift(indices: &mut [i32], distances: &mut [f32], len: usize) {
    let mut parent = 0;
    while parent < len / 2 {
        let left = parent * 2 + 1;
        let right = left + 1;
        let mut largest = left;
        if right < len
            && greater(
                distances[right],
                indices[right],
                distances[left],
                indices[left],
            )
        {
            largest = right;
        }
        if distances[parent] > distances[largest]
            || (distances[parent] == distances[largest] && indices[parent] >= indices[largest])
        {
            break;
        }
        indices.swap(parent, largest);
        distances.swap(parent, largest);
        parent = largest;
    }
}

fn traverse(data: &[f32], tree: &KdTree, row: usize, ids: &mut [i32], distances: &mut [f32]) {
    let width = ids.len();
    ids.fill(tree.indices.len() as i32);
    distances.fill(f32::INFINITY);
    let mut nodes = [0usize; MAX_KD_LEVELS + 1];
    let mut bounds = [0.0f32; MAX_KD_LEVELS + 1];
    bounds[0] = bound(data, tree, row, 0);
    let mut len = 1;
    let d = tree.n_features;
    while len > 0 {
        len -= 1;
        let node = nodes[len];
        if bounds[len] <= distances[0] {
            if tree.is_leaf[node] != 0 {
                for position in tree.starts[node] as usize..tree.ends[node] as usize {
                    let candidate = tree.indices[position] as usize;
                    if candidate == row {
                        continue;
                    }
                    let mut distance = 0.0f32;
                    for feature in 0..d {
                        let delta = data[row * d + feature] - data[candidate * d + feature];
                        distance += delta * delta;
                    }
                    if distance < distances[0]
                        || (distance == distances[0] && candidate < ids[0] as usize)
                    {
                        ids[0] = candidate as i32;
                        distances[0] = distance;
                        sift(ids, distances, width);
                    }
                }
            } else {
                let left = 2 * node + 1;
                let right = left + 1;
                let lb = bound(data, tree, row, left);
                let rb = bound(data, tree, row, right);
                // Push farther first; equality visits the left child first.
                let ordered = if lb <= rb {
                    [(right, rb), (left, lb)]
                } else {
                    [(left, lb), (right, rb)]
                };
                for (child, lower) in ordered {
                    if lower <= distances[0] {
                        nodes[len] = child;
                        bounds[len] = lower;
                        len += 1;
                    }
                }
            }
        }
    }
    for end in (1..ids.len()).rev() {
        ids.swap(0, end);
        distances.swap(0, end);
        sift(ids, distances, end);
    }
}

fn device_tree(tree: &KdTree, device: &Device) -> Result<DeviceKdTree<Dispatch>> {
    let nodes = tree.starts.len();
    let int_tensor = |values: Vec<i32>| {
        let len = values.len();
        Tensor::<1, Int>::from_data(TensorData::new(values, [len]), device).into_dispatch()
    };
    Ok(DeviceKdTree {
        indices: int_tensor(tree.indices.clone()),
        starts: int_tensor(tree.starts.clone()),
        ends: int_tensor(tree.ends.clone()),
        is_leaf: int_tensor(tree.is_leaf.clone()),
        bounds_min: Tensor::<2>::from_data(
            TensorData::new(tree.bounds_min.clone(), [nodes, tree.n_features]),
            device,
        )
        .into_dispatch(),
        bounds_max: Tensor::<2>::from_data(
            TensorData::new(tree.bounds_max.clone(), [nodes, tree.n_features]),
            device,
        )
        .into_dispatch(),
        node_components: int_tensor(tree.node_components(&vec![0; tree.indices.len()])?),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    })
}

fn positive(value: &str, name: &str) -> Result<usize> {
    let parsed: usize = value
        .parse()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if parsed == 0 {
        return Err(format!("{name} must be positive").into());
    }
    Ok(parsed)
}

fn filled<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result.try_reserve_exact(len)?;
    result.resize(len, value);
    Ok(result)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err("usage: kd_host_baseline INPUT ROWS FEATURES [--k 5] [--runs 5] [--threads T] [--verify]".into());
    }
    let rows = positive(&args[2], "ROWS")?;
    let features = positive(&args[3], "FEATURES")?;
    let mut k = 5;
    let mut runs = 5;
    let mut threads = None;
    let mut verify = false;
    let mut seen = std::collections::HashSet::new();
    let mut position = 4;
    while position < args.len() {
        let flag = args[position].as_str();
        if !seen.insert(flag) {
            return Err(format!("duplicate option {flag}").into());
        }
        match flag {
            "--verify" => verify = true,
            "--k" | "--runs" | "--threads" => {
                position += 1;
                let value = args
                    .get(position)
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                let value = positive(value, flag)?;
                match flag {
                    "--k" => k = value,
                    "--runs" => runs = value,
                    _ => threads = Some(value),
                }
            }
            _ => return Err(format!("unknown option {flag}").into()),
        }
        position += 1;
    }
    if rows > i32::MAX as usize || k > u32::MAX as usize || features > u32::MAX as usize {
        return Err("shape or k exceeds the public KD integer range".into());
    }
    let elements = rows
        .checked_mul(features)
        .ok_or("input element count overflow")?;
    let expected_bytes = elements.checked_mul(4).ok_or("input byte count overflow")?;
    let output_len = rows.checked_mul(k).ok_or("output element count overflow")?;
    if elements > u32::MAX as usize || output_len > u32::MAX as usize {
        return Err("input or output element count exceeds u32".into());
    }
    if std::fs::metadata(&args[1])?.len() != u64::try_from(expected_bytes)? {
        return Err("input byte length does not match ROWS * FEATURES * 4".into());
    }
    let bytes = std::fs::read(&args[1])?;
    if bytes.len() != expected_bytes {
        return Err("input byte length changed while reading".into());
    }
    let data: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    let tree = KdTree::build(&data, rows, features)?;
    let threads = match threads {
        Some(value) => value,
        None => CubeDevice::Cpu(Default::default())
            .client()
            .properties()
            .hardware
            .num_cpu_cores
            .ok_or("CubeCL CPU runtime did not report num_cpu_cores")? as usize,
    };
    if threads == 0 {
        return Err("CubeCL CPU runtime reported zero cores".into());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    let mut ids = filled(output_len, rows as i32)?;
    let mut distances = filled(output_len, f32::INFINITY)?;
    let mut times = Vec::new();
    times.try_reserve_exact(runs)?;
    let mut search = || {
        pool.install(|| {
            ids.par_chunks_mut(k)
                .zip(distances.par_chunks_mut(k))
                .enumerate()
                .for_each(|(row, (ids, distances))| traverse(&data, &tree, row, ids, distances));
        });
        black_box((&ids, &distances));
    };
    search(); // Excluded warmup; building, pool creation and output allocation are not timed.
    for _ in 0..runs {
        let start = Instant::now();
        search();
        times.push(start.elapsed().as_secs_f64());
    }
    let verification = if verify {
        let device = Device::cpu();
        let (public_ids, public_distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
            Tensor::<2>::from_data(TensorData::new(data.clone(), [rows, features]), &device)
                .into_dispatch(),
            device_tree(&tree, &device)?,
            k as u32,
        );
        let public_ids = Tensor::<2, Int>::from_dispatch(public_ids)
            .to_data()
            .try_to_vec::<i32>()?;
        let public_distances = Tensor::<2>::from_dispatch(public_distances)
            .to_data()
            .try_to_vec::<f32>()?;
        let id_mismatches = ids.iter().zip(&public_ids).filter(|(a, b)| a != b).count();
        let distance_bit_mismatches = distances
            .iter()
            .zip(&public_distances)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        serde_json::json!({
            "ids_equal": ids == public_ids,
            "distance_bits_equal": distances.len() == public_distances.len() && distance_bit_mismatches == 0,
            "id_mismatches": id_mismatches,
            "distance_bit_mismatches": distance_bit_mismatches,
            "public_id_count": public_ids.len(), "public_distance_count": public_distances.len(),
        })
    } else {
        serde_json::Value::Null
    };
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    let median = if runs % 2 == 0 {
        (sorted[runs / 2 - 1] + sorted[runs / 2]) / 2.0
    } else {
        sorted[runs / 2]
    };
    println!(
        "{}",
        serde_json::json!({
            "input": args[1], "rows": rows, "features": features, "k": k,
            "runs": runs, "threads": threads, "warmup_runs": 1,
            "run_seconds": times, "median_seconds": median, "verification": verification,
            "timing_scope": "scalar KD traversal, output initialization and heapsort; Rayon scheduling included",
        })
    );
    Ok(())
}
