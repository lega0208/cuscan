use burn::{
    backend::{Dispatch, DispatchDevice},
    tensor::{Device, Int, Tensor, TensorData},
};
use hdbscan::BackendChoice;
use hdbscan::{DeviceKdTree, HdbscanOps};
use hdbscan_core::kdtree::KdTree;

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
        true,
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
        true,
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
        true,
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

fn check_kd_oracle_high_dim(device: &Device) {
    let rows = 1_000usize;
    for features in [16usize, 30] {
        let values = (0..rows)
            .flat_map(|row| {
                (0..features).map(move |feature| {
                    // Three base-11 digits make every row unique; all integer
                    // coordinates keep squared distances exactly representable.
                    let coordinate = if feature < 3 {
                        let mut code = row;
                        for _ in 0..feature {
                            code /= 11;
                        }
                        (code % 11) as i32 - 5
                    } else {
                        ((row * 1_103 + feature * 7_919 + row * feature * 31) % 17) as i32 - 8
                    };
                    coordinate as f32
                })
            })
            .collect::<Vec<_>>();
        let tree = KdTree::build(&values, rows, features).unwrap();
        let (neighbors, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
            Tensor::<2>::from_data(TensorData::new(values.clone(), [rows, features]), device)
                .into_dispatch(),
            make_device_tree(&tree, &vec![-1; rows], device),
            5,
        );
        let neighbors = Tensor::<2, Int>::from_dispatch(neighbors)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        let distances = Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        for row in 0..rows {
            let mut candidates = Vec::with_capacity(rows - 1);
            for candidate in 0..rows {
                if row == candidate {
                    continue;
                }
                let mut distance = 0.0f32;
                for feature in 0..features {
                    let delta =
                        values[row * features + feature] - values[candidate * features + feature];
                    distance += delta * delta;
                }
                candidates.push((distance, candidate as i32));
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for slot in 0..5 {
                let offset = row * 5 + slot;
                assert_eq!(
                    neighbors[offset], candidates[slot].1,
                    "row {row}, d={features}"
                );
                assert_eq!(
                    distances[offset].to_bits(),
                    candidates[slot].0.to_bits(),
                    "distance bits at row {row}, d={features}, slot {slot}"
                );
                assert!(
                    candidates[slot].0 > 0.0,
                    "selected distance must exercise accumulation at row {row}, d={features}, slot {slot}"
                );
            }
        }
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
        true,
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

#[test]
fn test_device_kdtree_extension_roundtrip_cpu() {
    check_extension_roundtrip(&Device::cpu());
}

#[test]
fn test_device_kdtree_extension_roundtrip_cuda() {
    let device = cuda_device();
    if let Some(device) = device {
        check_extension_roundtrip(&device);
    }
}

#[test]
fn kd_traversals_match_exact_host_oracles_cpu() {
    check_kd_oracle(&Device::cpu());
    check_small_neighbor_boundaries(&Device::cpu());
    check_permuted_tree_row_oracles(&Device::cpu());
    check_overflow_distances(&Device::cpu(), hdbscan::BackendChoice::Cpu);
}

#[test]
fn kd_high_dimensional_neighbors_match_exact_host_oracle_cpu() {
    check_kd_oracle_high_dim(&Device::cpu());
}

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

#[test]
fn kd_high_dimensional_neighbors_match_exact_host_oracle_cuda() {
    let device = cuda_device();
    if let Some(device) = device {
        check_kd_oracle_high_dim(&device);
    }
}

fn check_overflow_distances(device: &Device, backend: hdbscan::BackendChoice) {
    use hdbscan::{HDBSCAN, HdbscanConfig, HdbscanSearch};

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
        true,
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

fn cuda_device() -> Option<Device> {
    match cuscan_common::selected_device(BackendChoice::Cuda { device_index: 0 }) {
        Ok((device, _)) => Some(device),
        Err(error) => {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "required CUDA backend unavailable: {error}"
            );
            eprintln!("SKIPPED: no CUDA device");
            None
        }
    }
}

// ---- Query/reference KD search (prediction traversal) ----

fn brute_query(
    query: &[f32],
    reference: &[f32],
    d: usize,
    k: usize,
    exclude_offset: Option<usize>,
) -> (Vec<i32>, Vec<u32>) {
    let rows = query.len() / d;
    let n = reference.len() / d;
    let mut ids = vec![n as i32; rows * k];
    let mut distances = vec![f32::INFINITY.to_bits(); rows * k];
    for row in 0..rows {
        let mut candidates = (0..n)
            .filter(|&c| exclude_offset.is_none_or(|offset| c != offset + row))
            .map(|c| {
                let distance = (0..d).fold(0.0f32, |sum, f| {
                    let delta = query[row * d + f] - reference[c * d + f];
                    sum + delta * delta
                });
                (distance, c as i32)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (slot, (distance, id)) in candidates.into_iter().take(k).enumerate() {
            ids[row * k + slot] = id;
            distances[row * k + slot] = distance.to_bits();
        }
    }
    (ids, distances)
}

fn host_tree_to_device(tree: &KdTree, device: &Device) -> DeviceKdTree<Dispatch> {
    let n = tree.indices.len();
    make_device_tree(tree, &vec![0; n], device)
}

/// Prepares `values` once and returns a batch query closure that reuses the
/// same prepared reference (cloned handles, no repacking) for every call.
fn prepared_query(
    values: &[f32],
    tree: &KdTree,
    d: usize,
    device: &Device,
) -> impl Fn(&[f32], usize, bool, usize) -> (Vec<i32>, Vec<u32>) {
    let n = values.len() / d;
    let data = Tensor::<2>::from_data(TensorData::new(values.to_vec(), [n, d]), device);
    let DispatchDevice::Cube(cube) = device.as_dispatch();
    let client = cube.client();
    let mut widths = vec![0];
    if client.properties().hardware.num_cpu_cores.is_some() {
        widths.push(
            client
                .io_optimized_vector_sizes(4)
                .filter(|&lanes| lanes <= 16)
                .max()
                .unwrap(),
        );
    }
    let references: Vec<_> = widths
        .into_iter()
        .map(|lanes| {
            <Dispatch as HdbscanOps>::prepare_kd_reference(
                data.clone().into_dispatch(),
                host_tree_to_device(tree, device),
                lanes as u32,
            )
        })
        .collect();
    let device = device.clone();
    move |query: &[f32], k: usize, exclude_self: bool, offset: usize| {
        let mut results = Vec::new();
        for reference in &references {
            let rows = query.len() / d;
            let tensor =
                Tensor::<2>::from_data(TensorData::new(query.to_vec(), [rows, d]), &device);
            let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd_query(
                tensor.into_dispatch(),
                reference.clone(),
                k as u32,
                exclude_self,
                offset as u32,
            );
            let ids = Tensor::<2, Int>::from_dispatch(ids)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap();
            assert_eq!(ids.len(), rows * k, "output is query rows by k");
            let distances = Tensor::<2>::from_dispatch(distances)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect();
            results.push((ids, distances));
        }
        for result in &results[1..] {
            assert_eq!(result, &results[0], "public fused layouts agree");
        }
        results.remove(0)
    }
}

fn lattice_values(n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|row| {
            (0..d).map(move |f| (((row * (f + 3) + row / (f + 2)) % (5 + f)) as f32) * 0.5)
        })
        .collect()
}

fn fitting_rows(
    values: &[f32],
    tree: &KdTree,
    d: usize,
    k: usize,
    device: &Device,
) -> (Vec<i32>, Vec<u32>) {
    let n = values.len() / d;
    let data = Tensor::<2>::from_data(TensorData::new(values.to_vec(), [n, d]), device);
    let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
        data.into_dispatch(),
        host_tree_to_device(tree, device),
        k as u32,
    );
    (
        Tensor::<2, Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect(),
    )
}

fn check_query_external_and_offsets(device: &Device) {
    for d in [3usize, 16, 30] {
        let n = 163usize;
        let mut values = lattice_values(n, d);
        // Identical coordinates at distinct IDs.
        let duplicate = values[0..d].to_vec();
        values[3 * d..4 * d].copy_from_slice(&duplicate);
        values[90 * d..91 * d].copy_from_slice(&duplicate);
        let tree = KdTree::build(&values, n, d).unwrap();
        let query = prepared_query(&values, &tree, d, device);

        // External queries: unequal row count, arbitrary order, exact copies of
        // reference rows (kept: no value-based exclusion), and off-lattice rows.
        let mut external = Vec::new();
        for row in [90usize, 0, 162, 3, 77, 41] {
            external.extend_from_slice(&values[row * d..(row + 1) * d]);
        }
        for row in 0..31usize {
            external.extend((0..d).map(|f| ((row * 7 + f * 3) % 11) as f32 * 0.375 - 0.25));
        }
        for k in [1usize, 2, 5, 33, n, n + 2] {
            assert_eq!(
                query(&external, k, false, 0),
                brute_query(&external, &values, d, k, None),
                "external d={d} k={k}"
            );
        }

        // Batched self queries with explicit offsets and a tail batch; both modes.
        let k = 6usize;
        let (fit_ids, fit_distances) = fitting_rows(&values, &tree, d, k, device);
        let mut self_ids = Vec::new();
        let mut self_distances = Vec::new();
        for start in (0..n).step_by(64) {
            let end = (start + 64).min(n);
            let batch = &values[start * d..end * d];
            let excluded = query(batch, k, true, start);
            assert_eq!(
                excluded,
                brute_query(batch, &values, d, k, Some(start)),
                "d={d} start={start}"
            );
            self_ids.extend(excluded.0);
            self_distances.extend(excluded.1);
            let included = query(batch, k, false, start);
            let expected = brute_query(batch, &values, d, k, None);
            assert_eq!(included, expected, "inclusive d={d} start={start}");
            for row in 0..end - start {
                assert_eq!(included.1[row * k], 0f32.to_bits(), "self distance is zero");
            }
        }
        // The fitting wrapper and the shared exclusion path agree bit for bit.
        assert_eq!(self_ids, fit_ids, "fitting ids d={d}");
        assert_eq!(self_distances, fit_distances, "fitting distances d={d}");
    }
}

fn check_query_ties_and_irregular_leaves(device: &Device) {
    // Equal-distance ring: lower IDs must win and equal node bounds must not
    // hide them. Points 0..8 sit at distance 1 from the origin.
    let d = 2usize;
    // Many repeated unit-ring points split across leaves whose node bounds
    // tie the current worst distance, plus far points.
    let unit = [[1.0f32, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
    let mut values = Vec::new();
    for row in 0..1000usize {
        if row % 5 == 4 {
            values.extend_from_slice(&[3.0 + row as f32 * 0.01, -3.0]);
        } else {
            values.extend_from_slice(&unit[row % 4]);
        }
    }
    let n = values.len() / d;
    let tree = KdTree::build(&values, n, d).unwrap();
    assert!(tree.is_leaf.iter().filter(|&&leaf| leaf != 0).count() > 2);
    let query = prepared_query(&values, &tree, d, device);
    let origin = [0.0f32, 0.0, 2.0, 0.0, 0.0, 1.0];
    for k in [1usize, 3, 7, 40, 799, 800, 801, n] {
        assert_eq!(
            query(&origin, k, false, 0),
            brute_query(&origin, &values, d, k, None),
            "ties k={k}"
        );
    }
    assert_eq!(query(&origin, 3, false, 0).0[..3], [0, 1, 2]);

    // Hand-built valid tree with irregular leaves: [0,2) and [2,7), permuted.
    let values = vec![
        5.0f32, 5.0, 0.0, 0.0, 6.0, 5.0, 0.5, 0.0, 5.0, 6.0, 7.0, 7.0, 0.0, 0.5,
    ];
    let indices = vec![1i32, 3, 0, 2, 4, 5, 6];
    let bounds = |rows: &[i32]| {
        let mut minimum = vec![f32::INFINITY; d];
        let mut maximum = vec![f32::NEG_INFINITY; d];
        for &row in rows {
            for f in 0..d {
                minimum[f] = minimum[f].min(values[row as usize * d + f]);
                maximum[f] = maximum[f].max(values[row as usize * d + f]);
            }
        }
        (minimum, maximum)
    };
    let (root_min, root_max) = bounds(&indices);
    let (left_min, left_max) = bounds(&indices[0..2]);
    let (right_min, right_max) = bounds(&indices[2..7]);
    let tree = KdTree {
        indices: indices.clone(),
        starts: vec![0, 0, 2],
        ends: vec![7, 2, 7],
        is_leaf: vec![0, 1, 1],
        bounds_min: [root_min, left_min, right_min].concat(),
        bounds_max: [root_max, left_max, right_max].concat(),
        n_levels: 1,
        n_features: d,
    };
    let query = prepared_query(&values, &tree, d, device);
    let external = vec![0.125f32, 0.125, 5.5, 5.5, 6.875, 7.125, 2.5, 2.5, 0.0, 0.0];
    for k in [1usize, 2, 3, 6, 7, 9] {
        assert_eq!(
            query(&external, k, false, 0),
            brute_query(&external, &values, d, k, None),
            "irregular k={k}"
        );
        assert_eq!(
            query(&values[2 * d..], k, true, 2),
            brute_query(&values[2 * d..], &values, d, k, Some(2)),
            "irregular offset k={k}"
        );
    }
}

#[test]
fn kd_query_external_rows_offsets_and_fitting_parity_cpu() {
    check_query_external_and_offsets(&Device::cpu());
}

#[test]
fn kd_query_ties_and_irregular_leaves_cpu() {
    check_query_ties_and_irregular_leaves(&Device::cpu());
}

#[test]
fn kd_query_external_rows_offsets_and_fitting_parity_cuda() {
    if let Some(device) = cuda_device() {
        check_query_external_and_offsets(&device);
    }
}

#[test]
fn kd_query_ties_and_irregular_leaves_cuda() {
    if let Some(device) = cuda_device() {
        check_query_ties_and_irregular_leaves(&device);
    }
}
