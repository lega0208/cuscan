use burn::backend::Dispatch;
use burn::tensor::{Device, Tensor, TensorData};
use fast_umap::backend::StandardUmapOps;
use serde_json::Value;

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
fn standard_fuzzy_graph_matches_reference_fixtures_on_flex() {
    verify_reference_cases(&Device::flex());
}

#[test]
fn standard_fuzzy_graph_keeps_equal_index_bipartite_neighbors() {
    let device = Device::flex();
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
    assert!(graph
        .heads
        .iter()
        .zip(&graph.tails)
        .zip(&graph.weights)
        .any(|((&head, &tail), &weight)| head == 0 && tail == 0 && weight > 0.0));
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
fn bipartite_graph_preserves_forward_memberships_across_set_op_mix_ratio_on_flex() {
    let device = Device::flex();
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

#[cfg(feature = "cuda")]
#[test]
fn bipartite_graph_preserves_forward_memberships_across_set_op_mix_ratio_on_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
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
    // Both backends match the shared reference fixtures within 1e-5, so a
    // 1e-4 budget covers their compounded sigma binary-search difference while
    // staying far below the mix scaling the old bug introduced.
    let flex = bipartite_mix_graph(&Device::flex(), 0.0);
    assert_eq!(zero.heads, flex.heads, "cuda/flex bipartite heads");
    assert_eq!(zero.tails, flex.tails, "cuda/flex bipartite tails");
    for (label, cuda_weights) in [
        ("mix 0.0", &zero.weights),
        ("mix 0.4", &intermediate.weights),
    ] {
        for (index, (&cuda, &flex_weight)) in cuda_weights.iter().zip(&flex.weights).enumerate() {
            let tolerance = 1e-4 + 1e-4 * flex_weight.abs();
            assert!(
                (cuda - flex_weight).abs() <= tolerance,
                "cuda/flex bipartite weights ({label})[{index}]: flex {flex_weight}, cuda {cuda} (tol {tolerance})"
            );
        }
    }
}

#[test]
fn bipartite_smooth_knn_matches_reference_when_nearest_distances_are_nonzero() {
    let graph = run_graph(
        &Device::flex(),
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
    assert!(graph
        .heads
        .iter()
        .zip(&graph.tails)
        .zip(&graph.weights)
        .any(|((&head, &tail), &weight)| head == 0 && tail == 0 && weight > 0.0));
}

#[test]
fn standard_fuzzy_graph_marks_invalid_reference_indices() {
    let graph = run_graph(
        &Device::flex(),
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

#[cfg(feature = "cuda")]
#[test]
fn standard_fuzzy_graph_matches_reference_fixtures_on_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    verify_reference_cases(&device);
}
