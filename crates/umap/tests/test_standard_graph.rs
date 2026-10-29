use burn::backend::Dispatch;
use burn::tensor::{Device, Tensor, TensorData};
use serde_json::Value;
use umap::backend::StandardUmapOps;

const FIXTURE: &str = include_str!("fixtures/standard_umap_reference.json");

struct GraphValues {
    heads: Vec<i32>,
    tails: Vec<i32>,
    weights: Vec<f32>,
    rho: Vec<f32>,
    sigma: Vec<f32>,
    directed: Vec<f32>,
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

fn ints(value: &Value) -> Vec<i32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i32)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_graph(
    device: &Device,
    indices: Vec<i32>,
    distances: Vec<f32>,
    n: usize,
    k: usize,
    local_connectivity: f32,
    bandwidth: f32,
    mix: f32,
    bipartite: bool,
    n_reference: usize,
) -> GraphValues {
    let indices =
        Tensor::<2, burn::tensor::Int>::from_data(TensorData::new(indices, [n, k]), device);
    let distances = Tensor::<2>::from_data(TensorData::new(distances, [n, k]), device);
    let (heads, tails, weights, rho, sigma, directed) = <Dispatch as StandardUmapOps>::fuzzy_graph(
        indices.into_dispatch(),
        distances.into_dispatch(),
        local_connectivity,
        bandwidth,
        mix,
        bipartite,
        u32::try_from(n_reference).unwrap(),
    );
    GraphValues {
        heads: Tensor::<1, burn::tensor::Int>::from_dispatch(heads)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        tails: Tensor::<1, burn::tensor::Int>::from_dispatch(tails)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        weights: Tensor::<1>::from_dispatch(weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        rho: Tensor::<1>::from_dispatch(rho)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        sigma: Tensor::<1>::from_dispatch(sigma)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
        directed: Tensor::<2>::from_dispatch(directed)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
    }
}

fn close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 1e-5 + 1e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}]: expected {expected}, got {actual} (tol {tolerance})"
        );
    }
}

fn f64_mean_oracle(
    indices: &[i32],
    distances: &[f32],
    n: usize,
    k: usize,
    local_connectivity: f32,
    bipartite: bool,
    n_reference: usize,
) -> GraphValues {
    let mean =
        (distances.iter().map(|&value| value as f64).sum::<f64>() / distances.len() as f64) as f32;
    let floor_connectivity = local_connectivity as usize;
    let fraction = local_connectivity - floor_connectivity as f32;
    let minimum_positive =
        floor_connectivity + usize::from(local_connectivity > floor_connectivity as f32);
    let target = (k as f32).log2();
    let mut rho = vec![0.0; n];
    let mut sigma = vec![0.0; n];
    let mut directed = vec![0.0; n * k];
    for row in 0..n {
        let mut positive: Vec<f32> = distances[row * k..(row + 1) * k]
            .iter()
            .copied()
            .filter(|value| *value > 0.0)
            .collect();
        positive.sort_by(f32::total_cmp);
        if !positive.is_empty() {
            let positive_max = *positive.last().unwrap();
            rho[row] = if floor_connectivity == 0 {
                positive[0] * fraction
            } else {
                let before = positive
                    .get(floor_connectivity - 1)
                    .copied()
                    .unwrap_or(positive_max);
                let after = positive
                    .get(floor_connectivity)
                    .copied()
                    .unwrap_or(positive_max);
                if fraction > 1e-5 {
                    before + (after - before) * fraction
                } else {
                    before
                }
            };
            if positive.len() < minimum_positive {
                rho[row] = positive_max;
            }
        }
        let floor_mean = if rho[row] > 0.0 {
            distances[row * k..(row + 1) * k].iter().sum::<f32>() / k as f32
        } else {
            mean
        };
        let (mut lo, mut hi, mut mid) = (0.0f32, f32::MAX, 1.0f32);
        for _ in 0..64 {
            let mut sum = 0.0f32;
            for &distance in &distances[row * k + 1..(row + 1) * k] {
                let delta = distance - rho[row];
                let mut value = (-delta.max(0.0) / if mid == 0.0 { 1.0 } else { mid }).exp();
                if delta <= 0.0 {
                    value = 1.0;
                }
                sum += value;
            }
            if (sum - target).abs() < 1e-5 {
                break;
            }
            if sum > target {
                hi = mid;
                mid = (lo + hi) * 0.5;
            } else if hi == f32::MAX {
                lo = mid;
                mid *= 2.0;
            } else {
                lo = mid;
                mid = (lo + hi) * 0.5;
            }
        }
        sigma[row] = mid.max(floor_mean * 1e-3);
        for slot in 0..k {
            let distance = distances[row * k + slot];
            let delta = distance - rho[row];
            let mut value =
                (-delta.max(0.0) / if sigma[row] == 0.0 { 1.0 } else { sigma[row] }).exp();
            if distance <= rho[row] || sigma[row] == 0.0 {
                value = 1.0;
            }
            let index = indices[row * k + slot];
            if index < 0 || index as usize >= n_reference || (!bipartite && index as usize == row) {
                value = 0.0;
            }
            directed[row * k + slot] = value;
        }
    }
    let mut forward = std::collections::BTreeMap::<(i32, i32), f32>::new();
    for row in 0..n {
        for slot in 0..k {
            let tail = indices[row * k + slot];
            if tail < 0 || tail as usize >= n_reference || directed[row * k + slot] <= 0.0 {
                continue;
            }
            *forward.entry((row as i32, tail)).or_default() += directed[row * k + slot];
        }
    }
    let mut heads = vec![0; 2 * n * k];
    let mut tails = vec![0; 2 * n * k];
    let mut weights = vec![0.0; 2 * n * k];
    let mut edge_count = 0;
    for (&(head, tail), &membership) in &forward {
        let weight = if bipartite {
            membership
        } else {
            let reverse = forward.get(&(tail, head)).copied().unwrap_or(0.0);
            membership + reverse - membership * reverse
        };
        if weight > 0.0 {
            heads[edge_count] = head;
            tails[edge_count] = tail;
            weights[edge_count] = weight;
            edge_count += 1;
            if !bipartite && head != tail {
                heads[edge_count] = tail;
                tails[edge_count] = head;
                weights[edge_count] = weight;
                edge_count += 1;
            }
        }
    }
    GraphValues {
        heads,
        tails,
        weights,
        rho,
        sigma,
        directed,
    }
}

fn verify_f64_mean_case(
    device: &Device,
    indices: Vec<i32>,
    distances: Vec<f32>,
    shape: [usize; 2],
    local_connectivity: f32,
    bipartite: bool,
    n_reference: usize,
) {
    let [n, k] = shape;
    let expected = f64_mean_oracle(
        &indices,
        &distances,
        n,
        k,
        local_connectivity,
        bipartite,
        n_reference,
    );
    let graph = run_graph(
        device,
        indices,
        distances,
        n,
        k,
        local_connectivity,
        1.0,
        1.0,
        bipartite,
        n_reference,
    );
    close(&graph.rho, &expected.rho, "f64 mean oracle rho");
    close(&graph.sigma, &expected.sigma, "f64 mean oracle sigma");
    close(
        &graph.directed,
        &expected.directed,
        "f64 mean oracle directed",
    );
    let actual_edges: std::collections::BTreeMap<_, _> = graph
        .heads
        .iter()
        .zip(&graph.tails)
        .zip(&graph.weights)
        .filter_map(|((&head, &tail), &weight)| (weight > 0.0).then_some(((head, tail), weight)))
        .collect();
    let expected_edges: std::collections::BTreeMap<_, _> = expected
        .heads
        .iter()
        .zip(&expected.tails)
        .zip(&expected.weights)
        .filter_map(|((&head, &tail), &weight)| (weight > 0.0).then_some(((head, tail), weight)))
        .collect();
    assert_eq!(
        actual_edges
            .keys()
            .collect::<std::collections::BTreeSet<_>>(),
        expected_edges
            .keys()
            .collect::<std::collections::BTreeSet<_>>(),
        "positive final fuzzy edge support"
    );
    close(
        &actual_edges.values().copied().collect::<Vec<_>>(),
        &expected_edges.values().copied().collect::<Vec<_>>(),
        "f64 mean oracle final edge weights",
    );
    let epochs = 200.0f32;
    let actual_maximum = actual_edges.values().copied().fold(0.0f32, f32::max);
    let expected_maximum = expected_edges.values().copied().fold(0.0f32, f32::max);
    let actual_eligible: std::collections::BTreeSet<_> = actual_edges
        .iter()
        .filter_map(|(&edge, &weight)| (weight >= actual_maximum / epochs).then_some(edge))
        .collect();
    let expected_eligible: std::collections::BTreeSet<_> = expected_edges
        .iter()
        .filter_map(|(&edge, &weight)| (weight >= expected_maximum / epochs).then_some(edge))
        .collect();
    assert_eq!(
        actual_eligible, expected_eligible,
        "positive-epoch edge eligibility at the standard max_weight/epochs cutoff"
    );
}

fn fixture_cases() -> Value {
    serde_json::from_str(FIXTURE).unwrap()
}

fn verify_reference_cases(device: &Device) {
    for case in fixture_cases()["graph_cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let k = case["k"].as_u64().unwrap() as usize;
        let neighbor_indices = ints(&case["indices"]);
        let graph = run_graph(
            device,
            neighbor_indices.clone(),
            floats(&case["distances"]),
            n,
            k,
            case["local_connectivity"].as_f64().unwrap() as f32,
            case["bandwidth"].as_f64().unwrap() as f32,
            case["set_op_mix_ratio"].as_f64().unwrap() as f32,
            false,
            n,
        );
        let name = case["name"].as_str().unwrap();
        close(&graph.rho, &floats(&case["rho"]), &format!("{name} rho"));
        close(
            &graph.sigma,
            &floats(&case["sigma"]),
            &format!("{name} sigma"),
        );
        let directed_dense = floats(&case["directed"]);
        assert_eq!(
            directed_dense.len(),
            n * n,
            "{name} fixture directed matrix shape"
        );
        let expected_directed_slots: Vec<f32> = neighbor_indices
            .iter()
            .enumerate()
            .map(|(slot_index, &neighbor)| {
                let row = slot_index / k;
                if neighbor >= 0 && (neighbor as usize) < n {
                    directed_dense[row * n + neighbor as usize]
                } else {
                    0.0
                }
            })
            .collect();
        close(
            &graph.directed,
            &expected_directed_slots,
            &format!("{name} directed slots"),
        );
        let mut dense = vec![0.0f32; n * n];
        let mut previous = None;
        for (&head, (&tail, &weight)) in graph
            .heads
            .iter()
            .zip(graph.tails.iter().zip(&graph.weights))
        {
            if weight == 0.0 {
                assert_eq!(
                    (head, tail),
                    (0, 0),
                    "unused edge capacity must be zero-filled"
                );
                continue;
            }
            assert!(head >= 0 && tail >= 0 && (head as usize) < n && (tail as usize) < n);
            assert_ne!(head, tail);
            let pair = (head, tail);
            assert!(
                previous.is_none_or(|last| last < pair),
                "edges must be sorted and unique"
            );
            previous = Some(pair);
            dense[head as usize * n + tail as usize] = weight;
        }
        close(
            &dense,
            &floats(&case["symmetric"]),
            &format!("{name} symmetric"),
        );
    }
}

#[test]
fn standard_fuzzy_graph_matches_reference_fixtures_on_cpu() {
    verify_reference_cases(&Device::cpu());
}

fn verify_global_mean_oracle_cases(device: &Device) {
    verify_f64_mean_case(
        device,
        vec![1, 1, 0, 0],
        vec![0.0, 0.0, 0.5, 0.5],
        [2, 2],
        1.0,
        false,
        2,
    );
    verify_f64_mean_case(
        device,
        vec![1, 0, 0, 1, 1, 0],
        vec![0.0, 0.5, 0.25, 0.0, 0.0, 1.5],
        [3, 2],
        0.0,
        false,
        3,
    );
    verify_f64_mean_case(
        device,
        vec![1, 0, 0, 1, 1, 0],
        vec![0.0, 0.5, 0.25, 0.0, 0.0, 1.5],
        [3, 2],
        1.00001,
        false,
        3,
    );
    verify_f64_mean_case(
        device,
        vec![0, 1, 2, 1, 2, 3],
        vec![0.0, 0.5, 0.0, 1.0, 1.5, 0.25],
        [3, 2],
        0.5,
        true,
        4,
    );
    for local_connectivity in [0.5, 1.5, 2.0] {
        verify_f64_mean_case(
            device,
            vec![1, 2, 0, 2, 0, 1],
            vec![0.0, 0.5, 0.25, 0.0, 0.0, 1.5],
            [3, 2],
            local_connectivity,
            false,
            3,
        );
    }

    let (n, k) = (69, 15);
    let indices = (0..n)
        .flat_map(|row| (1..=k).map(move |slot| ((row + slot) % n) as i32))
        .collect();
    let distances = (0..n * k)
        .map(|index| {
            if index < k {
                0.0
            } else {
                ((index * 37 % 997) as f32 + 1.0) / 113.0
            }
        })
        .collect();
    // Irregular flattened length: the mean must be correct when `n * k` is not a
    // multiple of 1024, independent of any reduction chunking.
    assert_ne!((n * k) % 1024, 0);
    verify_f64_mean_case(device, indices, distances, [n, k], 0.5, false, n);
}

#[test]
fn standard_fuzzy_graph_global_mean_cases_match_f64_oracle_on_cpu() {
    verify_global_mean_oracle_cases(&Device::cpu());
}

#[test]
fn standard_fuzzy_graph_global_mean_cases_match_f64_oracle_on_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    verify_global_mean_oracle_cases(&device);
}

#[test]
fn standard_fuzzy_graph_keeps_equal_index_bipartite_neighbors() {
    let device = Device::cpu();
    let graph = run_graph(
        &device,
        vec![0, 1, 1, 2],
        vec![0.0, 0.5, 0.0, 1.0],
        2,
        2,
        1.0,
        1.0,
        1.0,
        true,
        3,
    );
    assert_eq!(
        graph.directed[0], 1.0,
        "equal numeric indices are valid in bipartite mode"
    );
    assert!(graph.weights.iter().any(|weight| *weight > 0.0));
    assert!(
        graph
            .heads
            .iter()
            .zip(&graph.tails)
            .zip(&graph.weights)
            .any(|((&head, &tail), &weight)| head == 0 && tail == 0 && weight > 0.0)
    );
    // Bipartite edges stay (query row, reference id) and are emitted once.
    for ((&head, &tail), &weight) in graph.heads.iter().zip(&graph.tails).zip(&graph.weights) {
        if weight > 0.0 {
            assert!(
                head < 2 && tail < 3,
                "edge ({head},{tail}) leaves bipartite index spaces"
            );
        }
    }
    assert_eq!(
        graph.weights.iter().filter(|weight| **weight > 0.0).count(),
        4
    );
}

const BIPARTITE_MIX_N: usize = 3;
const BIPARTITE_MIX_K: usize = 3;
const BIPARTITE_MIX_N_REFERENCE: usize = 4;
const BIPARTITE_MIX_INDICES: [i32; BIPARTITE_MIX_N * BIPARTITE_MIX_K] = [0, 1, 1, 2, 0, 3, 1, 2, 2];
const BIPARTITE_MIX_DISTANCES: [f32; BIPARTITE_MIX_N * BIPARTITE_MIX_K] =
    [0.5, 1.0, 1.5, 0.25, 2.0, 1.0, 0.75, 0.5, 0.9];
// Rows reference {0,1}, {2,0,3}, {1,2}: seven distinct query/reference pairs,
// two of them coalesced from duplicate slots.
const BIPARTITE_MIX_EDGES: usize = 7;

fn bipartite_mix_graph(device: &Device, mix: f32) -> GraphValues {
    run_graph(
        device,
        BIPARTITE_MIX_INDICES.to_vec(),
        BIPARTITE_MIX_DISTANCES.to_vec(),
        BIPARTITE_MIX_N,
        BIPARTITE_MIX_K,
        1.0,
        1.0,
        mix,
        true,
        BIPARTITE_MIX_N_REFERENCE,
    )
}

/// Every bipartite edge must keep its coalesced forward membership unchanged,
/// stay oriented (query row, reference id), and be emitted exactly once per
/// distinct pair, whatever the fit set_op_mix_ratio is.
fn verify_bipartite_forward_memberships(graph: &GraphValues, mix: f32) {
    let mut emitted = 0usize;
    let mut previous: Option<(i32, i32)> = None;
    for ((&head, &tail), &weight) in graph.heads.iter().zip(&graph.tails).zip(&graph.weights) {
        if weight == 0.0 {
            assert_eq!(
                (head, tail),
                (0, 0),
                "mix {mix}: unused edge capacity must be zero-filled"
            );
            continue;
        }
        assert!(
            (head as usize) < BIPARTITE_MIX_N && (tail as usize) < BIPARTITE_MIX_N_REFERENCE,
            "mix {mix}: edge ({head},{tail}) leaves bipartite index spaces"
        );
        let pair = (head, tail);
        assert!(
            previous.is_none_or(|last| last < pair),
            "mix {mix}: bipartite edges must be sorted and unique"
        );
        previous = Some(pair);
        let forward: f32 = (0..BIPARTITE_MIX_K)
            .filter(|&slot| BIPARTITE_MIX_INDICES[head as usize * BIPARTITE_MIX_K + slot] == tail)
            .map(|slot| graph.directed[head as usize * BIPARTITE_MIX_K + slot])
            .sum();
        close(
            &[weight],
            &[forward],
            &format!("mix {mix}: forward membership of edge ({head},{tail})"),
        );
        emitted += 1;
    }
    assert_eq!(
        emitted, BIPARTITE_MIX_EDGES,
        "mix {mix}: every distinct query/reference pair must survive"
    );
}

#[test]
fn bipartite_graph_preserves_forward_memberships_across_set_op_mix_ratio_on_cpu() {
    let device = Device::cpu();
    let zero = bipartite_mix_graph(&device, 0.0);
    let intermediate = bipartite_mix_graph(&device, 0.4);
    verify_bipartite_forward_memberships(&zero, 0.0);
    verify_bipartite_forward_memberships(&intermediate, 0.4);
    assert_eq!(
        zero.heads, intermediate.heads,
        "set_op_mix_ratio must not change bipartite heads"
    );
    assert_eq!(
        zero.tails, intermediate.tails,
        "set_op_mix_ratio must not change bipartite tails"
    );
    close(
        &zero.weights,
        &intermediate.weights,
        "set_op_mix_ratio must not change bipartite weights",
    );
}

#[test]
fn bipartite_graph_preserves_forward_memberships_across_set_op_mix_ratio_on_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let zero = bipartite_mix_graph(&device, 0.0);
    let intermediate = bipartite_mix_graph(&device, 0.4);
    verify_bipartite_forward_memberships(&zero, 0.0);
    verify_bipartite_forward_memberships(&intermediate, 0.4);
    assert_eq!(
        zero.heads, intermediate.heads,
        "set_op_mix_ratio must not change cuda bipartite heads"
    );
    assert_eq!(
        zero.tails, intermediate.tails,
        "set_op_mix_ratio must not change cuda bipartite tails"
    );
    close(
        &zero.weights,
        &intermediate.weights,
        "set_op_mix_ratio must not change cuda bipartite weights",
    );
}

#[test]
fn bipartite_smooth_knn_matches_reference_when_nearest_distances_are_nonzero() {
    let graph = run_graph(
        &Device::cpu(),
        vec![0, 1, 2, 0, 1, 2],
        vec![0.5, 1.0, 2.0, 0.5, 1.0, 2.0],
        2,
        3,
        0.0,
        1.0,
        1.0,
        true,
        3,
    );

    // smooth_knn_dist uses rho=0 at local_connectivity=0 and always sums
    // distance slots 1..K, including for bipartite queries. Thus sigma solves
    // exp(-1/sigma) + exp(-2/sigma) = log2(3); slot 0 still receives a
    // membership, and equal numeric query/reference indices remain valid.
    let expected_sigma = 6.364_746;
    close(&graph.rho, &[0.0, 0.0], "bipartite rho");
    close(
        &graph.sigma,
        &[expected_sigma, expected_sigma],
        "bipartite sigma",
    );
    close(
        &graph.directed,
        &[
            0.924_448_67,
            0.854_605_4,
            0.730_350_4,
            0.924_448_67,
            0.854_605_4,
            0.730_350_4,
        ],
        "bipartite directed memberships",
    );
    assert!(
        graph
            .heads
            .iter()
            .zip(&graph.tails)
            .zip(&graph.weights)
            .any(|((&head, &tail), &weight)| head == 0 && tail == 0 && weight > 0.0)
    );
}

#[test]
fn standard_fuzzy_graph_marks_invalid_reference_indices() {
    let graph = run_graph(
        &Device::cpu(),
        vec![0, 3, 1, -2],
        vec![0.0, 1.0, 0.0, 1.0],
        2,
        2,
        1.0,
        1.0,
        1.0,
        false,
        2,
    );
    assert_eq!(
        (graph.heads[0], graph.tails[0], graph.weights[0]),
        (-2, -2, -1.0)
    );
}

#[test]
fn standard_fuzzy_graph_matches_reference_fixtures_on_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    verify_reference_cases(&device);
}
