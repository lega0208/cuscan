use burn::{
    backend::Dispatch,
    tensor::{Device, Tensor, TensorData},
};
use fast_umap::backend::StandardUmapOps;
#[cfg(any(feature = "cpu", feature = "cuda"))]
use serde_json::json;
#[cfg(feature = "cuda")]
use std::io::Write;
#[cfg(feature = "cuda")]
use std::process::{Command, Stdio};

#[cfg(feature = "cuda")]
fn source_tree_digest(commit: &str) -> String {
    let output = Command::new("git")
        .args(["rev-parse", &format!("{commit}:src")])
        .output()
        .expect("git must be available to identify product source");
    assert!(
        output.status.success(),
        "could not resolve {commit}:src tree digest"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[cfg(feature = "cuda")]
fn frozen_source_digest() -> String {
    let commit = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    for args in [
        vec!["diff", "--quiet", commit, "--", "src"],
        vec!["diff", "--cached", "--quiet", commit, "--", "src"],
    ] {
        let status = Command::new("git")
            .args(&args)
            .status()
            .expect("git must be available to verify frozen product source");
        assert!(
            status.success(),
            "working product src differs from frozen source commit"
        );
    }
    source_tree_digest(commit)
}

#[cfg(feature = "cuda")]
fn git_blob_digest(bytes: &[u8]) -> String {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "could not digest deterministic ANN inputs"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn run(device: Device, seed: u64, iterations: u32) -> (Vec<i32>, Vec<f32>) {
    let n = 48usize;
    let points = (0..n)
        .flat_map(|i| {
            let x = i as f32 * 0.15;
            [x, (x * 0.7).sin()]
        })
        .collect::<Vec<_>>();
    let data = Tensor::<2>::from_data(TensorData::new(points, [n, 2]), &device);
    let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
        data.into_dispatch(),
        5,
        seed,
        iterations,
    );
    (
        Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
    )
}

fn check_rows(ids: &[i32], distances: &[f32], n: usize, k: usize) {
    for row in 0..n {
        let mut unique = std::collections::HashSet::new();
        for slot in 0..k {
            let index = row * k + slot;
            let id = ids[index] as usize;
            assert!(id < n && id != row, "invalid/self neighbor at row {row}");
            assert!(unique.insert(id), "duplicate neighbor at row {row}");
            assert!(distances[index].is_finite() && distances[index] >= 0.0);
            let x = row as f32 * 0.15;
            let y = (x * 0.7).sin();
            let other_x = id as f32 * 0.15;
            let other_y = (other_x * 0.7).sin();
            let expected = ((x - other_x).powi(2) + (y - other_y).powi(2)).sqrt();
            assert!((distances[index] - expected).abs() <= 1e-5);
            if slot > 0 {
                let previous = row * k + slot - 1;
                assert!(
                    distances[previous] < distances[index]
                        || (distances[previous] == distances[index] && ids[previous] < ids[index]),
                    "row is not ordered by (distance,index)"
                );
            }
        }
    }
}

fn recall_score(ids: &[i32]) -> usize {
    let n = 48usize;
    let k = 5usize;
    let mut hits = 0;
    for row in 0..n {
        let mut exact = (0..n)
            .filter(|&other| other != row)
            .map(|other| {
                let x = row as f32 * 0.15;
                let y = (x * 0.7).sin();
                let ox = other as f32 * 0.15;
                let oy = (ox * 0.7).sin();
                let distance = (x - ox).powi(2) + (y - oy).powi(2);
                (distance, other)
            })
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let truth = exact
            .into_iter()
            .take(k)
            .map(|(_, id)| id as i32)
            .collect::<std::collections::HashSet<_>>();
        hits += ids[row * k..row * k + k]
            .iter()
            .filter(|id| truth.contains(id))
            .count();
    }
    hits
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn old_rescan_oracle(
    points: &[f32],
    n: usize,
    features: usize,
    k: usize,
    seed: u64,
    iterations: u32,
) -> (Vec<i32>, Vec<u32>) {
    let width = (k * 4).min(64).min(n - 1);
    let mut candidates = vec![0i32; n * width];
    for row in 0..n {
        let modulus = 2_147_483_647i64;
        let seed_base = (seed % 2_147_483_646) as i64 + 1;
        let mut state = (row as i64 + seed_base) % modulus;
        if state == 0 {
            state = 1;
        }
        for column in 0..width {
            let hi = state / 127_773;
            let lo = state - hi * 127_773;
            let test = lo * 16_807 - hi * 2_836;
            state = if test <= 0 { test + modulus } else { test };
            let t = state - 1;
            let j = n - 1 - width + column;
            let draw = t % (j + 1) as i64;
            let collision = (0..column).any(|previous| {
                let old = candidates[row * width + previous] as usize;
                let previous_raw = if old > row { old - 1 } else { old };
                previous_raw as i64 == draw
            });
            let chosen = if collision { j as i64 } else { draw };
            candidates[row * width + column] = if chosen >= row as i64 {
                (chosen + 1) as i32
            } else {
                chosen as i32
            };
        }
    }

    let mut final_ids = vec![n as i32; n * width];
    let mut final_squared = vec![f32::MAX; n * width];
    for iteration in 0..=iterations {
        let mut next_ids = vec![n as i32; n * width];
        let mut next_squared = vec![f32::MAX; n * width];
        for row in 0..n {
            let mut consider_candidate = |candidate: i32| {
                let candidate = candidate as usize;
                if candidate >= n || candidate == row {
                    return;
                }
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        points[row * features + feature] - points[candidate * features + feature];
                    squared += delta * delta;
                }
                let base = row * width;
                let mut position = width;
                let mut duplicate = false;
                for slot in 0..width {
                    let old = next_ids[base + slot];
                    duplicate |= old == candidate as i32;
                    if position == width
                        && (squared < next_squared[base + slot]
                            || (squared == next_squared[base + slot] && candidate < old as usize))
                    {
                        position = slot;
                    }
                }
                if !duplicate && position < width {
                    for slot in (position + 1..width).rev() {
                        next_ids[base + slot] = next_ids[base + slot - 1];
                        next_squared[base + slot] = next_squared[base + slot - 1];
                    }
                    next_ids[base + position] = candidate as i32;
                    next_squared[base + position] = squared;
                }
            };
            for slot in 0..width {
                consider_candidate(candidates[row * width + slot]);
            }
            if iteration > 0 {
                for slot in 0..width {
                    let neighbor = candidates[row * width + slot] as usize;
                    if neighbor < n {
                        for other in 0..width {
                            consider_candidate(candidates[neighbor * width + other]);
                        }
                    }
                }
                for source in 0..n {
                    if (0..width).any(|slot| candidates[source * width + slot] as usize == row) {
                        for slot in 0..width {
                            consider_candidate(candidates[source * width + slot]);
                        }
                    }
                }
            }
        }
        candidates = next_ids.clone();
        final_ids = next_ids;
        final_squared = next_squared;
    }
    let mut ids = Vec::with_capacity(n * k);
    let mut distance_bits = Vec::with_capacity(n * k);
    for row in 0..n {
        for slot in 0..k {
            ids.push(final_ids[row * width + slot]);
            distance_bits.push(final_squared[row * width + slot].sqrt().to_bits());
        }
    }
    (ids, distance_bits)
}

struct FullJoinTrace {
    ids: Vec<i32>,
    distance_bits: Vec<u32>,
    new_ids_per_expansion: Vec<usize>,
    stop_reason: &'static str,
}

fn generalized_full_join_oracle(
    points: &[f32],
    n: usize,
    features: usize,
    k: usize,
    seed: u64,
    max_iterations: u32,
) -> FullJoinTrace {
    let width = k.max(k.saturating_mul(4).min(32)).min(n - 1);
    let mut candidates = vec![0i32; n * width];
    const MODULUS: i64 = 2_147_483_647;
    let seed_base = (seed % (MODULUS as u64 - 1)) as i64 + 1;
    for row in 0..n {
        let mut state = (row as i64 + seed_base) % MODULUS;
        if state == 0 {
            state = 1;
        }
        for column in 0..width {
            let hi = state / 127_773;
            let lo = state - hi * 127_773;
            let test = lo * 16_807 - hi * 2_836;
            state = if test <= 0 { test + MODULUS } else { test };
            let draw = (state - 1) % (n - 1 - width + column + 1) as i64;
            let collision = (0..column).any(|previous| {
                let old = candidates[row * width + previous] as usize;
                let old_raw = if old > row { old - 1 } else { old };
                old_raw as i64 == draw
            });
            let chosen = if collision {
                (n - 1 - width + column) as i64
            } else {
                draw
            };
            candidates[row * width + column] = if chosen >= row as i64 {
                (chosen + 1) as i32
            } else {
                chosen as i32
            };
        }
    }

    let mut new_ids_per_expansion = Vec::new();
    let mut stop_reason = "cap";
    for expansion in 0..=max_iterations {
        let previous = candidates.clone();
        let mut next = vec![n as i32; n * width];
        let mut next_squared = vec![f32::MAX; n * width];
        for row in 0..n {
            let mut proposals = Vec::with_capacity(width * (width + 2));
            proposals.extend_from_slice(&previous[row * width..(row + 1) * width]);
            if expansion > 0 {
                for slot in 0..width {
                    let neighbor = previous[row * width + slot] as usize;
                    if neighbor < n {
                        proposals
                            .extend_from_slice(&previous[neighbor * width..(neighbor + 1) * width]);
                    }
                }
                for source in 0..n {
                    if previous[source * width..(source + 1) * width]
                        .iter()
                        .any(|&candidate| candidate as usize == row)
                    {
                        proposals
                            .extend_from_slice(&previous[source * width..(source + 1) * width]);
                    }
                }
            }
            let base = row * width;
            for candidate in proposals {
                let candidate = candidate as usize;
                if candidate >= n || candidate == row {
                    continue;
                }
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        points[row * features + feature] - points[candidate * features + feature];
                    squared += delta * delta;
                }
                if !squared.is_finite() {
                    continue;
                }
                let mut duplicate = false;
                let mut position = width;
                for slot in 0..width {
                    let old = next[base + slot];
                    duplicate |= old == candidate as i32;
                    if position == width
                        && (squared < next_squared[base + slot]
                            || (squared == next_squared[base + slot] && candidate < old as usize))
                    {
                        position = slot;
                    }
                }
                if !duplicate && position < width {
                    for slot in (position + 1..width).rev() {
                        next[base + slot] = next[base + slot - 1];
                        next_squared[base + slot] = next_squared[base + slot - 1];
                    }
                    next[base + position] = candidate as i32;
                    next_squared[base + position] = squared;
                }
            }
        }
        candidates = next;
        if expansion > 0 {
            let mut newly_admitted = 0;
            for row in 0..n {
                let old = &previous[row * width..(row + 1) * width];
                newly_admitted += candidates[row * width..(row + 1) * width]
                    .iter()
                    .filter(|&&id| id < n as i32 && !old.contains(&id))
                    .count();
            }
            new_ids_per_expansion.push(newly_admitted);
            if newly_admitted == 0 {
                stop_reason = "zero";
                break;
            }
        }
        if expansion == max_iterations {
            stop_reason = "cap";
            break;
        }
    }

    let mut ids = Vec::with_capacity(n * k);
    let mut distance_bits = Vec::with_capacity(n * k);
    for row in 0..n {
        for slot in 0..k {
            ids.push(candidates[row * width + slot]);
            let id = candidates[row * width + slot];
            let squared = if id >= 0 && (id as usize) < n {
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        points[row * features + feature] - points[id as usize * features + feature];
                    squared += delta * delta;
                }
                squared
            } else {
                f32::MAX
            };
            distance_bits.push(squared.sqrt().to_bits());
        }
    }
    FullJoinTrace {
        ids,
        distance_bits,
        new_ids_per_expansion,
        stop_reason,
    }
}

fn e2_points(shape_index: usize, n: usize, features: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|row| {
            (0..features).map(move |feature| match shape_index % 3 {
                0 => (row % 4) as f32 + ((row / 4 + feature) % 3) as f32 * 0.25,
                1 => {
                    if row % 3 == 0 {
                        0.0
                    } else {
                        ((row * (feature + 3) + feature * 7) % 11) as f32 * 0.125
                    }
                }
                _ => {
                    if row == 0 {
                        0.0
                    } else {
                        (((row * (feature + 2)) % 11 + 1) as f32 * 0.125) + feature as f32 * 0.0625
                    }
                }
            })
        })
        .collect()
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn assert_generalized_full_join_cases(device: &Device) {
    for (shape_index, n, features, k, iterations) in [
        (0usize, 7usize, 3usize, 2usize, 3u32),
        (1, 19, 1, 3, 4),
        (2, 31, 5, 4, 5),
        (3, 67, 3, 8, 5),
    ] {
        let points = e2_points(shape_index, n, features);
        let expected = generalized_full_join_oracle(&points, n, features, k, 0x5179, iterations);
        let data = Tensor::<2>::from_data(TensorData::new(points, [n, features]), device);
        let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            data.into_dispatch(),
            k as u32,
            0x5179,
            iterations,
        );
        let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let distance_bits = Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        assert_eq!(
            ids, expected.ids,
            "full-join oracle n={n}, f={features}, k={k}"
        );
        assert_eq!(distance_bits, expected.distance_bits);
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_matches_frozen_full_join_oracle_grid() {
    assert_generalized_full_join_cases(&Device::cpu());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_matches_frozen_full_join_oracle_grid() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    assert_generalized_full_join_cases(&device);
}

fn ordered_host_proposals(
    points: &[f32],
    n: usize,
    features: usize,
    row: usize,
    width: usize,
    proposals: &[i32],
) -> (Vec<i32>, Vec<u32>) {
    let mut ranked = Vec::new();
    for &id in proposals {
        if id < 0 || id as usize >= n || id as usize == row {
            continue;
        }
        let id = id as usize;
        let mut squared = 0.0f32;
        for feature in 0..features {
            let delta = points[row * features + feature] - points[id * features + feature];
            squared += delta * delta;
        }
        if squared.is_finite() && !ranked.iter().any(|&(old, _): &(usize, f32)| old == id) {
            ranked.push((id, squared));
        }
    }
    ranked.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
    let mut ids = vec![n as i32; width];
    let mut bits = vec![f32::MAX.to_bits(); width];
    for (slot, (id, squared)) in ranked.into_iter().take(width).enumerate() {
        ids[slot] = id as i32;
        bits[slot] = squared.to_bits();
    }
    (ids, bits)
}

#[test]
fn generalized_full_join_host_oracle_is_reverse_order_independent_and_rejects_overflow() {
    let points = [0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 2.0];
    let proposals = [3, 1, 2, 3, -1, 4];
    let forward = ordered_host_proposals(&points, 4, 2, 0, 3, &proposals);
    let reverse = ordered_host_proposals(
        &points,
        4,
        2,
        0,
        3,
        &proposals.iter().rev().copied().collect::<Vec<_>>(),
    );
    assert_eq!(
        forward, reverse,
        "reverse proposal permutation changed top-W"
    );

    let huge = [0.0f32, f32::MAX, -f32::MAX, 1.0];
    let overflow = ordered_host_proposals(&huge, 4, 1, 0, 2, &[1, 2, 3, 4]);
    assert_eq!(overflow.0, [3, 4]);
    assert_eq!(overflow.1, [1.0f32.to_bits(), f32::MAX.to_bits()]);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_matches_generalized_full_join_e2_grid() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    let shapes = [
        (2usize, 1usize, 1usize),
        (5, 2, 2),
        (19, 2, 3),
        (23, 3, 3),
        (31, 4, 3),
        (48, 2, 8),
        (67, 3, 33),
        (120, 5, 24),
    ];
    let seeds = [0x5179u64, 29, 0x5eed, 1, 47];
    let caps = [0u32, 1, 2, 5, 8, 14];
    let configurations = shapes
        .iter()
        .map(|&(n, features, k)| json!([n, features, k]))
        .collect::<Vec<_>>();
    let configuration_bytes = serde_json::to_vec(&json!({
        "shapes": configurations,
        "seeds": seeds,
        "caps": caps,
        "generator": "lattice/duplicate-heavy/hub-heavy-v1",
    }))
    .unwrap();
    let configuration_hash = git_blob_digest(&configuration_bytes);
    let configuration = serde_json::from_slice::<serde_json::Value>(&configuration_bytes).unwrap();
    let mut traces = Vec::with_capacity(shapes.len() * seeds.len() * caps.len());

    for (shape_index, &(n, features, k)) in shapes.iter().enumerate() {
        let points = e2_points(shape_index, n, features);
        for &seed in &seeds {
            for &cap in &caps {
                let data =
                    Tensor::<2>::from_data(TensorData::new(points.clone(), [n, features]), &device);
                let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
                    data.into_dispatch(),
                    k as u32,
                    seed,
                    cap,
                );
                let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
                    .to_data()
                    .try_to_vec::<i32>()
                    .unwrap();
                let distance_bits = Tensor::<2>::from_dispatch(distances)
                    .to_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>();
                let expected = generalized_full_join_oracle(&points, n, features, k, seed, cap);
                assert_eq!(
                    ids, expected.ids,
                    "ids mismatch n={n}, features={features}, k={k}, seed={seed}, cap={cap}"
                );
                assert_eq!(
                    distance_bits, expected.distance_bits,
                    "distance bits mismatch n={n}, features={features}, k={k}, seed={seed}, cap={cap}"
                );
                if expected.stop_reason == "zero" {
                    assert_eq!(expected.new_ids_per_expansion.last(), Some(&0));
                    let capped = generalized_full_join_oracle(
                        &points,
                        n,
                        features,
                        k,
                        seed,
                        caps[caps.len() - 1],
                    );
                    assert_eq!(
                        expected.ids, capped.ids,
                        "zero-stop ids differ from capped run"
                    );
                    assert_eq!(
                        expected.distance_bits, capped.distance_bits,
                        "zero-stop distances differ from capped run"
                    );
                } else if let Some(last_new_ids) = expected.new_ids_per_expansion.last() {
                    assert_ne!(
                        *last_new_ids, 0,
                        "zero NEW count must use the zero stop reason"
                    );
                }
                traces.push(json!({
                    "n": n,
                    "features": features,
                    "k": k,
                    "seed": seed,
                    "cap": cap,
                    "new_ids_per_expansion": expected.new_ids_per_expansion,
                    "stop_reason": expected.stop_reason,
                }));
            }
        }
    }
    assert_eq!(traces.len(), 240, "frozen E2 grid comparison count");

    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ANN_E2_TRACE_PATH") {
        let artifact = json!({
            "schema": "fast-umap.cuda-ann-e2-grid-trace.v1",
            "configuration_hash": configuration_hash,
            "configuration_hash_algorithm": "git-sha1-blob",
            "configuration": configuration,
            "comparisons": traces,
        });
        std::fs::write(path, serde_json::to_vec_pretty(&artifact).unwrap()).unwrap();
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_matches_independent_full_join_host_reference() {
    let n = 31usize;
    let features = 4usize;
    let k = 3usize;
    let seed = 0x5179;
    let iterations = 5;
    let points = e2_points(2, n, features);
    let expected = generalized_full_join_oracle(&points, n, features, k, seed, iterations);
    assert!(matches!(expected.stop_reason, "zero" | "cap"));
    assert!(expected.new_ids_per_expansion.len() <= iterations as usize + 1);
    if expected.stop_reason == "zero" {
        assert_eq!(expected.new_ids_per_expansion.last(), Some(&0));
    } else {
        assert_ne!(expected.new_ids_per_expansion.last(), Some(&0));
    }
    let data = Tensor::<2>::from_data(TensorData::new(points, [n, features]), &Device::cpu());
    let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
        data.into_dispatch(),
        k as u32,
        seed,
        iterations,
    );
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect::<Vec<_>>();
    assert_eq!(
        ids, expected.ids,
        "CPU ANN differs from independent host reference"
    );
    assert_eq!(distance_bits, expected.distance_bits);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_covers_parallel_width_boundaries() {
    let cores = std::thread::available_parallelism().unwrap().get();
    for n in [
        cores.saturating_sub(1).max(2),
        cores.max(2),
        cores + 1,
        2 * cores + 3,
    ] {
        let k = 3.min(n - 1);
        let points = e2_points(2, n, 4);
        let expected = generalized_full_join_oracle(&points, n, 4, k, 42, 3);
        let data = Tensor::<2>::from_data(TensorData::new(points, [n, 4]), &Device::cpu());
        let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            data.into_dispatch(),
            k as u32,
            42,
            3,
        );
        assert_eq!(
            Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap(),
            expected.ids,
            "row coverage n={n}"
        );
        assert_eq!(
            Tensor::<2>::from_dispatch(distances)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
            expected.distance_bits,
            "distance bits n={n}"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_matches_old_rescan_host_oracle_across_reverse_fixtures() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    assert_old_rescan_oracle_cases(&device);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_matches_old_rescan_host_oracle_across_reverse_fixtures() {
    assert_old_rescan_oracle_cases(&Device::cpu());
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn assert_old_rescan_oracle_cases(device: &Device) {
    let fixtures = [
        ("hub-ties", 19usize, 2usize),
        ("duplicate-points", 23usize, 3usize),
        ("repeated-iterations", 31usize, 4usize),
    ];
    for (fixture, n, features) in fixtures {
        let points = (0..n)
            .flat_map(|row| {
                if fixture == "duplicate-points" && row % 5 == 0 {
                    vec![0.0f32; features]
                } else if fixture == "hub-ties" {
                    (0..features)
                        .map(|column| if column == 0 { (row % 3) as f32 } else { 0.0 })
                        .collect()
                } else {
                    (0..features)
                        .map(|column| ((row * (column + 3)) % 17) as f32 * 0.125)
                        .collect()
                }
            })
            .collect::<Vec<_>>();
        let k = 3usize;
        let iterations = 5;
        let data = Tensor::<2>::from_data(TensorData::new(points.clone(), [n, features]), device);
        let (actual_ids, actual_distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            data.into_dispatch(),
            k as u32,
            0x5179,
            iterations,
        );
        let actual_ids = Tensor::<2, burn::tensor::Int>::from_dispatch(actual_ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let actual_distance_bits = Tensor::<2>::from_dispatch(actual_distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let expected = old_rescan_oracle(&points, n, features, k, 0x5179, iterations);
        assert_eq!(
            actual_ids, expected.0,
            "old-rescan ids differ for {fixture}"
        );
        assert_eq!(
            actual_distance_bits, expected.1,
            "distance bits differ for {fixture}"
        );
    }
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn assert_ann_candidate_width_matrix(device: &Device) {
    for (n, features, k) in [(7usize, 3usize, 1usize), (43, 1, 7), (43, 3, 8), (47, 5, 9)] {
        let points = (0..n)
            .flat_map(|row| {
                (0..features)
                    .map(move |column| ((row * (column + 3) + column * 5) % 23) as f32 * 0.125)
            })
            .collect::<Vec<_>>();
        let width = (k * 4).min(64).min(n - 1);
        if k >= 7 {
            assert!(matches!(width, 28 | 32 | 36));
        }
        let iterations = 4;
        let data = Tensor::<2>::from_data(TensorData::new(points.clone(), [n, features]), device);
        let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            data.into_dispatch(),
            k as u32,
            0x5179,
            iterations,
        );
        let actual_ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let actual_distance_bits = Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>();
        let expected = old_rescan_oracle(&points, n, features, k, 0x5179, iterations);
        assert_eq!(
            actual_ids, expected.0,
            "ANN width {width}, n={n}, f={features}"
        );
        assert_eq!(
            actual_distance_bits, expected.1,
            "ANN distance width {width}, n={n}, f={features}"
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_matches_independent_oracle_across_candidate_widths_and_odd_shapes() {
    assert_ann_candidate_width_matrix(&Device::cpu());
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_matches_independent_oracle_across_candidate_widths_and_odd_shapes() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    assert_ann_candidate_width_matrix(&device);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_is_deterministic_unique_and_sorted() {
    let (ids, distances) = run(Device::cpu(), 7, 6);
    assert_eq!((ids.len(), distances.len()), (48 * 5, 48 * 5));
    check_rows(&ids, &distances, 48, 5);
    assert_eq!((ids, distances), run(Device::cpu(), 7, 6));
    // Zero iterations expose the seeded random starts directly.
    assert_ne!(run(Device::cpu(), 7, 0).0, run(Device::cpu(), 8, 0).0);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_resolves_equal_distances_by_neighbor_index() {
    let device = Device::cpu();
    let data = Tensor::<2>::from_data(
        TensorData::new(
            vec![0.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [5, 2],
        ),
        &device,
    );
    let (indices, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 4, 13, 3);
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(&ids[..4], &[1, 2, 3, 4]);
    assert_eq!(&distances[..4], &[1.0; 4]);
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_refines_seed_candidates_with_more_neighbor_joins() {
    let (seeded, _) = run(Device::cpu(), 29, 0);
    let (refined, _) = run(Device::cpu(), 29, 8);
    assert!(
        recall_score(&refined) > recall_score(&seeded),
        "NN-Descent iterations should improve the seed graph"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_runs_on_cuda_and_obeys_required_device_mode() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    let (ids, distances) = run(device, 7, 6);
    check_rows(&ids, &distances, 48, 5);
    assert_eq!(
        (ids, distances),
        run(fast_umap::cuda_backend::device(0).unwrap(), 7, 6)
    );
    assert_ne!(
        run(fast_umap::cuda_backend::device(0).unwrap(), 7, 0).0,
        run(fast_umap::cuda_backend::device(0).unwrap(), 8, 0).0,
    );
    let (seeded, _) = run(fast_umap::cuda_backend::device(0).unwrap(), 29, 0);
    let (refined, _) = run(fast_umap::cuda_backend::device(0).unwrap(), 29, 8);
    assert!(
        recall_score(&refined) > recall_score(&seeded),
        "CUDA NN-Descent iterations should improve the seed graph"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn required_cuda_mode_fails_when_cuda_support_is_not_compiled() {
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
    );
}

#[cfg(feature = "cpu")]
#[test]
fn ann_rejects_invalid_tiny_width_instead_of_exact_fallback() {
    let device = Device::cpu();
    let data = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0], [2, 1]), &device);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ =
                <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 0, 1, 3);
        }))
        .is_err()
    );
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_handles_minimal_two_row_input_and_rejects_single_row_input() {
    let device = Device::cpu();
    let data = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 3.0], [2, 1]), &device);
    let (indices, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 1, 42, 3);
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(ids, [1, 0]);
    assert_eq!(distances, [3.0, 3.0]);

    let singleton = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = <Dispatch as StandardUmapOps>::approximate_neighbors(
                singleton.into_dispatch(),
                1,
                42,
                3,
            );
        }))
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_source_tagged_capture_oracle() {
    const SOURCE: &str = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    let rows = 67usize;
    let features = 3usize;
    let points = odd_dimension_points(rows);
    let input_bits = points
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let input_bytes = points
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let data = Tensor::<2>::from_data(TensorData::new(points.clone(), [rows, features]), &device);
    let (indices, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
        data.clone().into_dispatch(),
        33,
        0x5eed,
        7,
    );
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let queries = Tensor::<2>::from_data(
        TensorData::new(points[..65 * features].to_vec(), [65, features]),
        &device,
    );
    let fit_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(ids.clone(), [rows, 33]),
        &device,
    );
    let mut query_ids = Vec::new();
    let mut query_distance_bits = Vec::new();
    let mut query_stats = Vec::new();
    for start in [0usize, 64] {
        let end = (start + 64).min(65);
        let block = queries.clone().slice([start..end, 0..features]);
        let (found, found_distances, stats) = <Dispatch as StandardUmapOps>::approximate_query(
            block.into_dispatch(),
            data.clone().into_dispatch(),
            fit_graph.clone().into_dispatch(),
            33,
            48,
            67,
            0x5eed,
            start as u64,
        );
        query_ids.extend(
            Tensor::<2, burn::tensor::Int>::from_dispatch(found)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap(),
        );
        query_distance_bits.extend(
            Tensor::<2>::from_dispatch(found_distances)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.to_bits()),
        );
        query_stats.extend(
            Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap(),
        );
    }
    let semantic_capture = json!({
        "input_git_blob_digest": git_blob_digest(&input_bytes),
        "query_input_git_blob_digest": git_blob_digest(
            &input_bytes[..65 * features * std::mem::size_of::<f32>()],
        ),
        "schema": "fast-umap.cuda-ann-capture.v1",
        "operation": "approximate_neighbors+approximate_query",
        "dtype": "f32/i32",
        "fit_shape": [rows, features, 33],
        "query_shape": [65, features, 33],
        "seed": 0x5eedu64,
        "iterations": 7,
        "query_budget": [48, 67],
        "query_offsets": [0, 64],
        "input_f32_bits": input_bits,
        "fit_ids": ids,
        "fit_distance_bits": distance_bits,
        "query_ids": query_ids,
        "query_distance_bits": query_distance_bits,
        "query_peak_candidates_unique_evaluations": query_stats,
        "invalid_sentinel_overflow_and_budget_cases": ann_edge_cases(&device),
    });
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ANN_CAPTURE_PATH") {
        // Capture export is the frozen-baseline operation; never label candidate
        // product source as the baseline oracle's source.
        let mut capture = semantic_capture.as_object().unwrap().clone();
        capture.insert("source_commit".to_owned(), json!(SOURCE));
        capture.insert(
            "product_src_tree_digest".to_owned(),
            json!(frozen_source_digest()),
        );
        std::fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
    }
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ANN_ORACLE_PATH") {
        let oracle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(oracle["source_commit"], SOURCE, "oracle source revision");
        assert_eq!(
            oracle["product_src_tree_digest"],
            source_tree_digest(SOURCE),
            "oracle product source provenance"
        );
        let oracle_keys = oracle
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let expected_keys = semantic_capture
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut expected_oracle_keys = expected_keys.clone();
        expected_oracle_keys.insert("source_commit".to_owned());
        expected_oracle_keys.insert("product_src_tree_digest".to_owned());
        assert_eq!(oracle_keys, expected_oracle_keys, "oracle semantic key set");
        for (field, value) in semantic_capture.as_object().unwrap() {
            assert_eq!(
                oracle[field], *value,
                "CUDA ANN semantic oracle mismatch in {field}"
            );
        }
    }
    assert_eq!(ids, run_odd_dimension_case(&device));
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn ann_edge_cases(device: &Device) -> serde_json::Value {
    use fast_umap::standard_umap::approximate_query_budget;
    let reference_values = vec![0.0f32, 1.0, 2.0, 3.0];
    let query_values = vec![0.5f32];
    let reference =
        Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [4, 1]), device);
    let query = Tensor::<2>::from_data(TensorData::new(query_values.clone(), [1, 1]), device);
    let sentinel_graph_values = vec![-1i32, 1, 2, 3];
    let invalid_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(sentinel_graph_values.clone(), [4, 1]),
        device,
    );
    let (_, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
        query.clone().into_dispatch(),
        reference.clone().into_dispatch(),
        invalid_graph.into_dispatch(),
        1,
        2,
        4,
        17,
        0,
    );
    let sentinel_stats = Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    assert!(
        sentinel_stats.iter().all(|&value| value >= 0),
        "-1 is a valid disconnected sentinel"
    );
    let invalid_graph_values = vec![-2i32, 1, 2, 3];
    let invalid_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(invalid_graph_values.clone(), [4, 1]),
        device,
    );
    let (_, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
        query.clone().into_dispatch(),
        reference.clone().into_dispatch(),
        invalid_graph.into_dispatch(),
        1,
        2,
        4,
        17,
        0,
    );
    let invalid_stats = Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    assert_eq!(
        invalid_stats,
        vec![-1, -1],
        "invalid graph IDs must use the existing invalid marker"
    );
    let extreme_values = vec![f32::MAX, -f32::MAX, 0.0, 1.0];
    let extreme = Tensor::<2>::from_data(TensorData::new(extreme_values.clone(), [4, 1]), device);
    let (ids, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(extreme.into_dispatch(), 2, 991, 0);
    let overflow_ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let overflow_distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert!(
        (extreme_values[0] - extreme_values[1]).is_infinite(),
        "fixture must reach finite-input squared-distance overflow"
    );
    let budget_error = approximate_query_budget(usize::MAX, usize::MAX, f64::MAX).unwrap_err();
    let budget_error = match budget_error {
        fast_umap::standard_umap::UmapError::Overflow(subject) => json!({
            "variant": "Overflow", "payload": subject,
        }),
        other => panic!("expected checked budget overflow, got {other:?}"),
    };
    #[cfg(feature = "cuda")]
    let overflow_input_digest = git_blob_digest(
        &extreme_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    #[cfg(not(feature = "cuda"))]
    let overflow_input_digest = String::new();
    json!({
        "operation": "approximate_query-invalid-markers+approximate_neighbors-overflow+approximate_query_budget-overflow",
        "dtype": "f32/i32/u64",
        "shapes": {"reference": [4, 1], "query": [1, 1], "graph": [4, 1], "overflow_fit": [4, 1]},
        "query_budget": [2, 4],
        "query_offsets": [0],
        "sentinel_graph": sentinel_graph_values,
        "sentinel_stats": sentinel_stats,
        "invalid_graph": [-2, 1, 2, 3],
        "invalid_graph_stats": invalid_stats,
        "overflow_input_bits": extreme_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "overflow_input_digest": overflow_input_digest,
        "overflow_fit_ids": overflow_ids,
        "overflow_fit_distance_bits": overflow_distance_bits,
        "representability_error": budget_error,
    })
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_edge_cases_preserve_markers_and_checked_overflow() {
    let cases = ann_edge_cases(&Device::cpu());
    assert!(
        cases["sentinel_stats"]
            .as_array()
            .unwrap()
            .iter()
            .all(|value| value.as_i64().unwrap() >= 0)
    );
    assert_eq!(cases["invalid_graph_stats"], serde_json::json!([-1, -1]));
    assert_eq!(cases["overflow_fit_ids"].as_array().unwrap().len(), 8);
    assert!(cases["representability_error"].is_object());
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn odd_dimension_points(rows: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            if row == 1 {
                return [0.0, 0.0, 0.0];
            }
            let x = (row % 17) as f32 * 0.25;
            [
                x,
                ((row * 7) % 13) as f32 * 0.125,
                ((row * 11) % 19) as f32 * 0.0625,
            ]
        })
        .collect()
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn run_odd_dimension_case(device: &Device) -> Vec<i32> {
    let data = Tensor::<2>::from_data(TensorData::new(odd_dimension_points(67), [67, 3]), device);
    let (indices, _) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 33, 0x5eed, 7);
    Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap()
}

#[cfg(feature = "cpu")]
#[test]
fn cpu_ann_handles_odd_feature_and_row_counts() {
    let ids = run_odd_dimension_case(&Device::cpu());
    let points = odd_dimension_points(67);
    let expected = old_rescan_oracle(&points, 67, 3, 33, 0x5eed, 7);
    assert_eq!(ids.len(), 67 * 33);
    assert_eq!(
        ids, expected.0,
        "odd-shape IDs differ from frozen host oracle"
    );
    for (row, neighbors) in ids.chunks_exact(33).enumerate() {
        assert!(
            neighbors
                .iter()
                .all(|&id| (0..67).contains(&id) && id != row as i32)
        );
    }
}
