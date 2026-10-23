use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
#[cfg(feature = "cuda")]
use fast_hdbscan::BackendChoice;
use fast_hdbscan::{DeviceKdTree, HdbscanOps};
use fast_hdbscan_core::kdtree::KdTree;

const N: usize = 81;
const D: usize = 2;

fn data_values() -> Vec<f32> {
    (0..N)
        .flat_map(|i| {
            [
                ((i * 17 + i / 7) % 13) as f32,
                ((i * 11 + i / 3) % 9) as f32,
            ]
        })
        .collect()
}

fn component_values() -> Vec<i32> {
    (0..N).map(|i| ((i * 7 + i / 5) % 3) as i32).collect()
}

fn make_device_tree(tree: &KdTree, components: &[i32], device: &Device) -> DeviceKdTree<Dispatch> {
    let nodes = tree.node_components(components).unwrap();
    DeviceKdTree {
        indices: Tensor::<1, Int>::from_data(
            TensorData::new(tree.indices.clone(), [tree.indices.len()]),
            device,
        )
        .into_dispatch(),
        starts: Tensor::<1, Int>::from_data(
            TensorData::new(tree.starts.clone(), [tree.starts.len()]),
            device,
        )
        .into_dispatch(),
        ends: Tensor::<1, Int>::from_data(
            TensorData::new(tree.ends.clone(), [tree.ends.len()]),
            device,
        )
        .into_dispatch(),
        is_leaf: Tensor::<1, Int>::from_data(
            TensorData::new(tree.is_leaf.clone(), [tree.is_leaf.len()]),
            device,
        )
        .into_dispatch(),
        bounds_min: Tensor::<2>::from_data(
            TensorData::new(
                tree.bounds_min.clone(),
                [tree.starts.len(), tree.n_features],
            ),
            device,
        )
        .into_dispatch(),
        bounds_max: Tensor::<2>::from_data(
            TensorData::new(
                tree.bounds_max.clone(),
                [tree.starts.len(), tree.n_features],
            ),
            device,
        )
        .into_dispatch(),
        node_components: Tensor::<1, Int>::from_data(
            TensorData::new(nodes, [tree.starts.len()]),
            device,
        )
        .into_dispatch(),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    }
}

fn check_extension_roundtrip(device: &Device) {
    let values = data_values();
    let components_values = component_values();
    let tree = KdTree::build(&values, N, D).unwrap();
    let data = Tensor::<2>::from_data(TensorData::new(values, [N, D]), device);
    let components =
        Tensor::<1, Int>::from_data(TensorData::new(components_values.clone(), [N]), device);
    let core = Tensor::<1>::from_data(TensorData::new(vec![0.0; N], [N]), device);
    let (weights, indices) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
        data.into_dispatch(),
        core.into_dispatch(),
        components.clone().into_dispatch(),
        make_device_tree(&tree, &components_values, device),
    );
    assert_eq!(
        Tensor::<1, Int>::from_dispatch(indices)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap()
            .len(),
        N
    );
    assert_eq!(
        Tensor::<1>::from_dispatch(weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .len(),
        N
    );
}

fn check_kd_oracle(device: &Device) {
    let values = data_values();
    let components_values = component_values();
    let tree = KdTree::build(&values, N, D).unwrap();
    let device_tree = make_device_tree(&tree, &components_values, device);
    let data_tensor = Tensor::<2>::from_data(TensorData::new(values.clone(), [N, D]), device);
    let (neighbors, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
        data_tensor.clone().into_dispatch(),
        device_tree,
        3,
    );
    let neighbors = Tensor::<2, Int>::from_dispatch(neighbors)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let mut expected_neighbors = vec![N as i32; N * 3];
    let mut expected_distances = vec![f32::INFINITY; N * 3];
    for i in 0..N {
        let mut candidates = Vec::new();
        for j in 0..N {
            if i == j {
                continue;
            }
            let mut distance = 0.0f32;
            for feature in 0..D {
                let delta = values[i * D + feature] - values[j * D + feature];
                distance += delta * delta;
            }
            candidates.push((distance, j as i32));
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for slot in 0..3 {
            expected_distances[i * 3 + slot] = candidates[slot].0;
            expected_neighbors[i * 3 + slot] = candidates[slot].1;
        }
    }
    assert_eq!(neighbors, expected_neighbors);
    assert_eq!(distances, expected_distances);

    let (neighbors, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
        Tensor::<2>::from_data(TensorData::new(values.clone(), [N, D]), device).into_dispatch(),
        make_device_tree(&tree, &components_values, device),
        N as u32,
    );
    let neighbors = Tensor::<2, Int>::from_dispatch(neighbors)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    for row in 0..N {
        assert_eq!(neighbors[row * N + N - 1], N as i32);
        assert!(distances[row * N + N - 1].is_infinite());
    }

    let components_tensor =
        Tensor::<1, Int>::from_data(TensorData::new(components_values.clone(), [N]), device);
    let core_values = (0..N).map(|i| (i % 7) as f32 * 0.125).collect::<Vec<_>>();
    let core_tensor = Tensor::<1>::from_data(TensorData::new(core_values.clone(), [N]), device);
    let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
        data_tensor.into_dispatch(),
        core_tensor.into_dispatch(),
        components_tensor.into_dispatch(),
        make_device_tree(&tree, &components_values, device),
    );
    let weights = Tensor::<1>::from_dispatch(weights)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let destinations = Tensor::<1, Int>::from_dispatch(destinations)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let mut expected_weights = vec![f32::INFINITY; N];
    let mut expected_destinations = vec![N as i32; N];
    for i in 0..N {
        for j in 0..N {
            if components_values[i] == components_values[j] {
                continue;
            }
            let mut distance = 0.0f32;
            for feature in 0..D {
                let delta = values[i * D + feature] - values[j * D + feature];
                distance += delta * delta;
            }
            let weight = distance.max(core_values[i]).max(core_values[j]);
            if (weight, j as i32) < (expected_weights[i], expected_destinations[i]) {
                expected_weights[i] = weight;
                expected_destinations[i] = j as i32;
            }
        }
    }
    assert_eq!(weights, expected_weights);
    assert_eq!(destinations, expected_destinations);

    let infinite_cores =
        Tensor::<1>::from_data(TensorData::new(vec![f32::INFINITY; N], [N]), device);
    let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
        Tensor::<2>::from_data(TensorData::new(values.clone(), [N, D]), device).into_dispatch(),
        infinite_cores.into_dispatch(),
        Tensor::<1, Int>::from_data(TensorData::new(components_values.clone(), [N]), device)
            .into_dispatch(),
        make_device_tree(&tree, &components_values, device),
    );
    let weights = Tensor::<1>::from_dispatch(weights)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let destinations = Tensor::<1, Int>::from_dispatch(destinations)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    assert!(weights.iter().all(|weight| weight.is_infinite()));
    for (row, &destination) in destinations.iter().enumerate() {
        let expected = components_values
            .iter()
            .position(|&component| component != components_values[row])
            .unwrap();
        assert_eq!(destination, expected as i32);
    }
}

fn check_small_neighbor_boundaries(device: &Device) {
    let cases = [
        vec![vec![2.0f32, -1.0]],
        vec![vec![0.0, 0.0], vec![3.0, 4.0]],
        vec![vec![0.0, 0.0]; 4],
    ];
    for input in cases {
        let n = input.len();
        let values = input.iter().flatten().copied().collect::<Vec<_>>();
        let tree = KdTree::build(&values, n, D).unwrap();
        let mut widths = vec![1, n, n + 5];
        if n > 1 {
            widths.push(n - 1);
        }
        widths.sort_unstable();
        widths.dedup();
        for k in widths {
            let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
                Tensor::<2>::from_data(TensorData::new(values.clone(), [n, D]), device)
                    .into_dispatch(),
                make_device_tree(&tree, &vec![0; n], device),
                k as u32,
            );
            let ids = Tensor::<2, Int>::from_dispatch(ids)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap();
            let distances = Tensor::<2>::from_dispatch(distances)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap();
            let mut expected_ids = vec![n as i32; n * k];
            let mut expected_distances = vec![f32::INFINITY; n * k];
            for row in 0..n {
                let mut candidates = (0..n)
                    .filter(|&candidate| candidate != row)
                    .map(|candidate| {
                        let distance = (0..D).fold(0.0f32, |sum, feature| {
                            let delta = values[row * D + feature] - values[candidate * D + feature];
                            sum + delta * delta
                        });
                        (distance, candidate as i32)
                    })
                    .collect::<Vec<_>>();
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                for (slot, &(distance, candidate)) in candidates.iter().take(k).enumerate() {
                    expected_ids[row * k + slot] = candidate;
                    expected_distances[row * k + slot] = distance;
                }
            }
            assert_eq!(ids, expected_ids, "n={n}, k={k}");
            assert_eq!(distances, expected_distances, "n={n}, k={k}");
        }
    }
}

fn check_permuted_tree_row_oracles(device: &Device) {
    let n = 163usize;
    let d = 3usize;
    let mut values = (0..n)
        .flat_map(|row| {
            [
                (row % 9) as f32,
                ((row * 5 + row / 4) % 13) as f32,
                ((row * 3 + row / 7) % 11) as f32,
            ]
        })
        .collect::<Vec<_>>();
    let duplicate = values[0..d].to_vec();
    values[3 * d..4 * d].copy_from_slice(&duplicate);
    let mut components = vec![0; n];
    components[n - 1] = 1;
    let tree = KdTree::build(&values, n, d).unwrap();
    assert_ne!(tree.indices, (0..n as i32).collect::<Vec<_>>());
    let data = Tensor::<2>::from_data(TensorData::new(values.clone(), [n, d]), device);

    for k in [1usize, 2, 3, 5, 8, 9, 31, 32, 33, n + 2] {
        let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
            data.clone().into_dispatch(),
            make_device_tree(&tree, &components, device),
            k as u32,
        );
        let ids = Tensor::<2, Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let distances = Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let mut expected_ids = vec![n as i32; n * k];
        let mut expected_distances = vec![f32::INFINITY; n * k];
        for row in 0..n {
            let mut candidates = (0..n)
                .filter(|&candidate| candidate != row)
                .map(|candidate| {
                    let distance = (0..d).fold(0.0f32, |sum, feature| {
                        let delta = values[row * d + feature] - values[candidate * d + feature];
                        sum + delta * delta
                    });
                    (distance, candidate as i32)
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (slot, &(distance, candidate)) in candidates.iter().take(k).enumerate() {
                expected_ids[row * k + slot] = candidate;
                expected_distances[row * k + slot] = distance;
            }
        }
        assert_eq!(ids, expected_ids);
        assert_eq!(distances, expected_distances);
    }

    let core = (0..n)
        .map(|row| (row % 5) as f32 * 0.25)
        .collect::<Vec<_>>();
    let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
        data.into_dispatch(),
        Tensor::<1>::from_data(TensorData::new(core.clone(), [n]), device).into_dispatch(),
        Tensor::<1, Int>::from_data(TensorData::new(components.clone(), [n]), device)
            .into_dispatch(),
        make_device_tree(&tree, &components, device),
    );
    let weights = Tensor::<1>::from_dispatch(weights)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    let destinations = Tensor::<1, Int>::from_dispatch(destinations)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    for row in 0..n {
        let mut expected = (0..n)
            .filter(|&candidate| components[candidate] != components[row])
            .map(|candidate| {
                let distance = (0..d).fold(0.0f32, |sum, feature| {
                    let delta = values[row * d + feature] - values[candidate * d + feature];
                    sum + delta * delta
                });
                (
                    distance.max(core[row]).max(core[candidate]),
                    candidate as i32,
                )
            })
            .collect::<Vec<_>>();
        expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(weights[row], expected[0].0, "candidate weight at row {row}");
        assert_eq!(
            destinations[row], expected[0].1,
            "candidate index at row {row}"
        );
    }
}

#[cfg(feature = "cpu")]
#[test]
fn test_device_kdtree_extension_roundtrip_cpu() {
    check_extension_roundtrip(&Device::cpu());
}

#[cfg(feature = "cuda")]
#[test]
fn test_device_kdtree_extension_roundtrip_cuda() {
    let device = cuda_device();
    if let Some(device) = device {
        check_extension_roundtrip(&device);
    }
}

#[cfg(feature = "cpu")]
#[test]
fn kd_traversals_match_exact_host_oracles_cpu() {
    check_kd_oracle(&Device::cpu());
    check_small_neighbor_boundaries(&Device::cpu());
    check_permuted_tree_row_oracles(&Device::cpu());
    check_overflow_distances(&Device::cpu(), fast_hdbscan::BackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn kd_traversals_match_exact_host_oracles_cuda() {
    let device = cuda_device();
    if let Some(device) = device {
        check_kd_oracle(&device);
        check_small_neighbor_boundaries(&device);
        check_permuted_tree_row_oracles(&device);
        check_overflow_distances(&device, BackendChoice::Cuda { device_index: 0 });
    }
}

fn check_overflow_distances(device: &Device, backend: fast_hdbscan::BackendChoice) {
    use fast_hdbscan::{HDBSCAN, HdbscanConfig, HdbscanSearch};

    // Finite inputs whose every pairwise f32 squared distance overflows. Real
    // infinite neighbors must beat the equally infinite missing-point sentinel.
    let input = vec![
        vec![f32::MAX, 0.0],
        vec![-f32::MAX, 0.0],
        vec![0.0, f32::MAX],
    ];
    let flat = input.iter().flatten().copied().collect::<Vec<_>>();
    let tree = KdTree::build(&flat, 3, D).unwrap();
    let data = Tensor::<2>::from_data(TensorData::new(flat, [3, D]), device);
    let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
        data.clone().into_dispatch(),
        make_device_tree(&tree, &[0, 1, 2], device),
        3,
    );
    assert_eq!(
        Tensor::<2, Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        [1, 2, 3, 0, 2, 3, 0, 1, 3]
    );
    assert!(
        Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == f32::INFINITY)
    );
    let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
        data.into_dispatch(),
        Tensor::<1>::from_data(TensorData::new(vec![0.0; 3], [3]), device).into_dispatch(),
        Tensor::<1, Int>::from_data(TensorData::new(vec![0, 1, 2], [3]), device).into_dispatch(),
        make_device_tree(&tree, &[0, 1, 2], device),
    );
    assert_eq!(
        Tensor::<1, Int>::from_dispatch(destinations)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        [1, 0, 0]
    );
    assert!(
        Tensor::<1>::from_dispatch(weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == f32::INFINITY)
    );

    for min_samples in [1, 2, 3, 8] {
        for search in [HdbscanSearch::BruteForce, HdbscanSearch::KdTree] {
            let fitted = HDBSCAN::new(
                HdbscanConfig {
                    min_cluster_size: 2,
                    min_samples: Some(min_samples),
                    ..Default::default()
                },
                backend,
            )
            .with_search(search)
            .fit(&input)
            .unwrap();
            assert_eq!(fitted.search(), search);
            assert_eq!(fitted.minimum_spanning_tree().len(), 2);
            assert!(
                fitted
                    .minimum_spanning_tree()
                    .iter()
                    .all(|edge| edge.weight == f64::INFINITY && edge.src < 3 && edge.dst < 3)
            );
            let width = min_samples.min(2);
            let mut expected_neighbors = vec![-1; 3 * min_samples];
            for row in 0..3 {
                let mut candidates = (0..3)
                    .filter(|&candidate| candidate != row)
                    .collect::<Vec<_>>();
                candidates.sort_unstable();
                for (slot, candidate) in candidates.into_iter().take(width).enumerate() {
                    expected_neighbors[row * min_samples + slot] = candidate as i32;
                }
            }
            assert_eq!(fitted.neighbors(), expected_neighbors);
            let expected_core = if min_samples >= 3 {
                f32::INFINITY
            } else if min_samples == 1 {
                0.0
            } else {
                f32::INFINITY
            };
            assert!(
                fitted
                    .core_distances()
                    .iter()
                    .all(|&core| core == expected_core)
            );
            assert!(fitted.probabilities().iter().all(|value| value.is_finite()));
            let expected_cut = if min_samples == 1 {
                [0, 1, 2]
            } else {
                [-1, -1, -1]
            };
            assert_eq!(fitted.dbscan_clustering(0.0).unwrap(), expected_cut);
            assert!(fitted.dbscan_clustering(f64::NAN).is_err());
        }
    }
}

#[cfg(feature = "cuda")]
fn cuda_device() -> Option<Device> {
    match fast_cubecl_common::selected_device(
        BackendChoice::Cuda { device_index: 0 },
        cfg!(feature = "cpu"),
        true,
    ) {
        Ok((device, _)) => Some(device),
        Err(error) => {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "required CUDA backend unavailable: {error}"
            );
            eprintln!("SKIPPED: no CUDA device");
            None
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn required_cuda_kdtree_tests_fail_when_cuda_is_not_compiled() {
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
    );
}
