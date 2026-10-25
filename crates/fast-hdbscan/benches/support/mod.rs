use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use criterion::Criterion;
use fast_cubecl_common::neighbors;
use fast_hdbscan::{
    BackendChoice, ClusterSelectionMethod, DeviceKdTree, HDBSCAN, HdbscanConfig, HdbscanOps,
    HdbscanSearch, MstGraph,
};
use serde_json::Value;
use std::env;

#[path = "../../../fast-hdbscan-core/tests/support/mod.rs"]
#[allow(dead_code)]
mod fixture_support;

#[derive(Clone)]
struct Case {
    name: String,
    data: Vec<Vec<f32>>,
    config: HdbscanConfig,
}

fn synthetic(rows: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..dimensions)
                .map(|feature| {
                    let seed = (row as u64 + 1) * 0x9e37_79b9 + (feature as u64 + 3) * 0x85eb_ca6b;
                    let mixed = seed ^ (seed >> 17) ^ (seed << 13);
                    let noise = (mixed % 10_007) as f32 / 10_007.0 - 0.5;
                    let cluster = (row % 4) as f32 * 4.0;
                    cluster + feature as f32 * 0.03125 + noise * 0.05
                })
                .collect()
        })
        .collect()
}

fn fixture_case(id: &str) -> Case {
    let root = fixture_support::fixture_root();
    let metadata = fixture_support::read_json(&root.join(format!("{id}.json")));
    let arrays = fixture_support::load_npz(&root.join(format!("{id}.npz")))
        .unwrap_or_else(|error| panic!("fixture {id}: {error}"));
    let header = &arrays["input"];
    let flat = fixture_support::read_f32s(header).unwrap();
    let data = flat
        .chunks_exact(header.shape[1])
        .map(<[f32]>::to_vec)
        .collect();
    let parameters = &metadata["parameters"];
    Case {
        name: format!("fixture/{id}"),
        data,
        config: fixture_config(parameters),
    }
}

fn fixture_config(parameters: &Value) -> HdbscanConfig {
    HdbscanConfig {
        min_cluster_size: parameters["min_cluster_size"].as_u64().unwrap() as usize,
        min_samples: Some(parameters["min_samples"].as_u64().unwrap() as usize),
        cluster_selection_method: match parameters["selection"].as_str().unwrap() {
            "eom" => ClusterSelectionMethod::Eom,
            "leaf" => ClusterSelectionMethod::Leaf,
            other => panic!("unknown fixture selection {other}"),
        },
        allow_single_cluster: parameters["allow_single_cluster"].as_bool().unwrap(),
        max_cluster_size: parameters["max_cluster_size"].as_u64().map(|n| n as usize),
        cluster_selection_epsilon: parameters["epsilon"].as_f64().unwrap(),
        cluster_selection_persistence: parameters["persistence"].as_f64().unwrap(),
        ..Default::default()
    }
}

fn cases() -> Vec<Case> {
    let mut cases = vec![fixture_case("blobs_d2"), fixture_case("moons_leaf")];
    for (rows, dimensions) in [
        (64, 2),
        (64, 10),
        (64, 30),
        (64, 50),
        (256, 2),
        (256, 10),
        (256, 30),
        (256, 50),
        (1024, 2),
        (1024, 10),
        (1024, 30),
        (1024, 50),
        (2048, 2),
        (2048, 10),
        (2048, 30),
        (2048, 50),
        (4096, 2),
        (4096, 10),
    ] {
        for min_samples in [5, 32] {
            cases.push(Case {
                name: format!("synthetic/n{rows}_d{dimensions}_ms{min_samples}"),
                data: synthetic(rows, dimensions),
                config: HdbscanConfig {
                    min_cluster_size: 5,
                    min_samples: Some(min_samples),
                    ..HdbscanConfig::default()
                },
            });
        }
    }
    cases
}

fn selected_case_filter() -> Option<Vec<String>> {
    env::var("FAST_HDBSCAN_BENCH_CASES").ok().map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .map(str::to_owned)
            .collect()
    })
}

fn include_case(name: &str, filter: &Option<Vec<String>>) -> bool {
    filter
        .as_ref()
        .is_none_or(|names| names.iter().any(|entry| name.contains(entry)))
}

fn config() -> criterion::Criterion {
    criterion::Criterion::default()
        .sample_size(100)
        .confidence_level(0.95)
}

pub fn run(c: &mut Criterion, backend: BackendChoice, backend_name: &str) {
    let filter = selected_case_filter();
    let all_cases = cases();
    let selected = all_cases
        .iter()
        .filter(|case| include_case(&case.name, &filter))
        .collect::<Vec<_>>();
    assert!(
        !selected.is_empty(),
        "FAST_HDBSCAN_BENCH_CASES selected no cases"
    );
    eprintln!(
        "HDBSCAN {backend_name} benchmark cases: {}",
        selected
            .iter()
            .map(|case| case.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let mut group = c.benchmark_group(format!("hdbscan/{backend_name}"));
    for case in selected {
        let flat = case.data.iter().flatten().copied().collect::<Vec<_>>();
        let n = case.data.len();
        let d = case.data[0].len();
        group.bench_function(format!("{}/fit/brute_force", case.name), |b| {
            let estimator =
                HDBSCAN::new(case.config.clone(), backend).with_search(HdbscanSearch::BruteForce);
            b.iter(|| {
                std::hint::black_box(
                    estimator
                        .fit(std::hint::black_box(&case.data))
                        .unwrap_or_else(|error| {
                            panic!("{backend_name} {} brute-force fit: {error}", case.name)
                        }),
                )
            })
        });
        group.bench_function(format!("{}/fit/kd_tree", case.name), |b| {
            let estimator =
                HDBSCAN::new(case.config.clone(), backend).with_search(HdbscanSearch::KdTree);
            b.iter(|| {
                std::hint::black_box(
                    estimator
                        .fit(std::hint::black_box(&case.data))
                        .unwrap_or_else(|error| {
                            panic!("{backend_name} {} KD-tree fit: {error}", case.name)
                        }),
                )
            })
        });
        // Public core construction allows isolated tree-build measurements. Device operation,
        // upload, and synchronization are internal to fit and are not relabeled as separate timings.
        group.bench_function(format!("{}/kd_tree_build", case.name), |b| {
            b.iter(|| {
                std::hint::black_box(
                    fast_hdbscan_core::kdtree::KdTree::build(std::hint::black_box(&flat), n, d)
                        .unwrap(),
                )
            })
        });
        if env::var("FAST_HDBSCAN_BENCH_STAGES").as_deref() == Ok("1") {
            benchmark_stages(&mut group, case, backend, backend_name, &flat, n, d);
        }
    }
    group.finish();
    run_kd_bound(c, backend, backend_name);
}

// Full-fit reproducibility comparison on the complete-graph KD route. Every case lies inside the
// Auto KD region (>=256 rows, <=30 features), so both toggles measure the
// same route that Auto would select for these shapes.
fn run_kd_bound(c: &mut Criterion, backend: BackendChoice, backend_name: &str) {
    let mut group = c.benchmark_group(format!("kd_bound/{backend_name}"));
    for (rows, dimensions) in [(1024, 10), (1024, 30), (2048, 10), (2048, 30)] {
        let data = synthetic(rows, dimensions);
        for min_samples in [16, 32] {
            for reproducible in [false, true] {
                let config = HdbscanConfig {
                    min_cluster_size: 5,
                    min_samples: Some(min_samples),
                    mst_graph: MstGraph::Complete,
                    reproducible,
                    ..HdbscanConfig::default()
                };
                let estimator = HDBSCAN::new(config, backend).with_search(HdbscanSearch::KdTree);
                let id = format!("{rows}x{dimensions}_s{min_samples}/reproducible={reproducible}");
                group.bench_function(id, |b| {
                    b.iter(|| {
                        std::hint::black_box(
                            estimator
                                .fit(std::hint::black_box(&data))
                                .unwrap_or_else(|error| {
                                    panic!(
                                        "{backend_name} kd_bound {rows}x{dimensions} fit: {error}"
                                    )
                                }),
                        )
                    })
                });
            }
        }
    }
    group.finish();
}

fn tensor_f32_2(values: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn tensor_f32_1(values: Vec<f32>, device: &Device) -> Tensor<1> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), device)
}

fn tensor_i32_1(values: Vec<i32>, device: &Device) -> Tensor<1, Int> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), device)
}

fn upload_tree(
    tree: &fast_hdbscan_core::kdtree::KdTree,
    device: &Device,
    node_components: Vec<i32>,
) -> DeviceKdTree<Dispatch> {
    let nodes = tree.starts.len();
    let d = tree.n_features;
    DeviceKdTree {
        indices: tensor_i32_1(tree.indices.clone(), device).into_dispatch(),
        starts: tensor_i32_1(tree.starts.clone(), device).into_dispatch(),
        ends: tensor_i32_1(tree.ends.clone(), device).into_dispatch(),
        is_leaf: tensor_i32_1(tree.is_leaf.clone(), device).into_dispatch(),
        bounds_min: tensor_f32_2(tree.bounds_min.clone(), [nodes, d], device).into_dispatch(),
        bounds_max: tensor_f32_2(tree.bounds_max.clone(), [nodes, d], device).into_dispatch(),
        node_components: tensor_i32_1(node_components, device).into_dispatch(),
        n_levels: tree.n_levels,
        n_features: d,
    }
}

fn clone_device_tree(tree: &DeviceKdTree<Dispatch>) -> DeviceKdTree<Dispatch> {
    DeviceKdTree {
        indices: tree.indices.clone(),
        starts: tree.starts.clone(),
        ends: tree.ends.clone(),
        is_leaf: tree.is_leaf.clone(),
        bounds_min: tree.bounds_min.clone(),
        bounds_max: tree.bounds_max.clone(),
        node_components: tree.node_components.clone(),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    }
}

fn benchmark_stages(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    case: &Case,
    backend: BackendChoice,
    backend_name: &str,
    flat: &[f32],
    n: usize,
    d: usize,
) {
    let (device, _) = fast_cubecl_common::selected_device(backend)
        .unwrap_or_else(|error| panic!("{backend_name} stage benchmark device: {error}"));
    let data = tensor_f32_2(flat.to_vec(), [n, d], &device);
    let width = case.config.resolved_min_samples().min(n.saturating_sub(1));
    let width_u32 = u32::try_from(width).expect("fixture neighbor width fits u32");
    let tree = fast_hdbscan_core::kdtree::KdTree::build(flat, n, d).unwrap();

    group.bench_function(
        format!("{}/stage/neighbor_brute_readback", case.name),
        |b| {
            b.iter(|| {
                let (ids, distances) = neighbors::exact_neighbors(
                    data.clone(),
                    data.clone(),
                    width_u32,
                    true,
                    0,
                    true,
                );
                std::hint::black_box((
                    ids.to_data().try_to_vec::<i32>().unwrap(),
                    distances.to_data().try_to_vec::<f32>().unwrap(),
                ))
            })
        },
    );

    // The tree is static for neighbor traversal, so it is uploaded outside the timed loop.
    let kd_tree = upload_tree(&tree, &device, vec![-1; tree.starts.len()]);
    group.bench_function(format!("{}/stage/neighbor_kd_readback", case.name), |b| {
        b.iter(|| {
            let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
                data.clone().into_dispatch(),
                clone_device_tree(&kd_tree),
                width_u32,
            );
            let ids: Tensor<2, Int> = Tensor::from_dispatch(ids);
            let distances: Tensor<2> = Tensor::from_dispatch(distances);
            std::hint::black_box((
                ids.to_data().try_to_vec::<i32>().unwrap(),
                distances.to_data().try_to_vec::<f32>().unwrap(),
            ))
        })
    });

    // Candidate-round workloads use singleton point components. They measure exactly one
    // independent-candidate kernel round, not convergence or complete MST construction.
    // Core squared radii are computed from exact brute-force neighbor distances before
    // timing; candidates receive these squared values directly, without sqrt/square conversion.
    let (_, knn_distances) =
        neighbors::exact_neighbors(data.clone(), data.clone(), width_u32, true, 0, true);
    let knn_sq = knn_distances.to_data().try_to_vec::<f32>().unwrap();
    let requested_min_samples = case.config.resolved_min_samples();
    let core_sq = (0..n)
        .map(|row| {
            if requested_min_samples >= n {
                f32::INFINITY
            } else if requested_min_samples == 1 {
                0.0
            } else {
                knn_sq[row * width + width - 1]
            }
        })
        .collect::<Vec<_>>();
    let components = (0..n).map(|i| i as i32).collect::<Vec<_>>();
    let core_tensor = tensor_f32_1(core_sq, &device);
    let component_tensor = tensor_i32_1(components.clone(), &device);
    let node_components = tree.node_components(&components).unwrap();
    let candidate_tree = upload_tree(&tree, &device, vec![-1; tree.starts.len()]);
    group.bench_function(
        format!("{}/stage/candidate_brute_single_round_readback", case.name),
        |b| {
            b.iter(|| {
                let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates(
                    data.clone().into_dispatch(),
                    core_tensor.clone().into_dispatch(),
                    component_tensor.clone().into_dispatch(),
                );
                let weights: Tensor<1> = Tensor::from_dispatch(weights);
                let destinations: Tensor<1, Int> = Tensor::from_dispatch(destinations);
                std::hint::black_box((
                    weights.to_data().try_to_vec::<f32>().unwrap(),
                    destinations.to_data().try_to_vec::<i32>().unwrap(),
                ))
            })
        },
    );
    group.bench_function(
        format!("{}/stage/candidate_kd_single_round_readback", case.name),
        |b| {
            b.iter(|| {
                let mut fresh_tree = clone_device_tree(&candidate_tree);
                fresh_tree.node_components =
                    tensor_i32_1(node_components.clone(), &device).into_dispatch();
                let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
                    data.clone().into_dispatch(),
                    core_tensor.clone().into_dispatch(),
                    component_tensor.clone().into_dispatch(),
                    fresh_tree,
                    true,
                );
                let weights: Tensor<1> = Tensor::from_dispatch(weights);
                let destinations: Tensor<1, Int> = Tensor::from_dispatch(destinations);
                std::hint::black_box((
                    weights.to_data().try_to_vec::<f32>().unwrap(),
                    destinations.to_data().try_to_vec::<i32>().unwrap(),
                ))
            })
        },
    );
    // Upload timing includes readback of every uploaded field, forcing completion rather
    // than measuring only host-side tensor construction or asynchronous enqueue.
    group.bench_function(
        format!("{}/stage/kd_tree_upload_readback", case.name),
        |b| {
            b.iter(|| {
                let uploaded = upload_tree(&tree, &device, node_components.clone());
                let indices: Tensor<1, Int> = Tensor::from_dispatch(uploaded.indices.clone());
                let starts: Tensor<1, Int> = Tensor::from_dispatch(uploaded.starts.clone());
                let ends: Tensor<1, Int> = Tensor::from_dispatch(uploaded.ends.clone());
                let is_leaf: Tensor<1, Int> = Tensor::from_dispatch(uploaded.is_leaf.clone());
                let bounds_min: Tensor<2> = Tensor::from_dispatch(uploaded.bounds_min.clone());
                let bounds_max: Tensor<2> = Tensor::from_dispatch(uploaded.bounds_max.clone());
                let node_components: Tensor<1, Int> =
                    Tensor::from_dispatch(uploaded.node_components.clone());
                std::hint::black_box((
                    indices.to_data().try_to_vec::<i32>().unwrap(),
                    starts.to_data().try_to_vec::<i32>().unwrap(),
                    ends.to_data().try_to_vec::<i32>().unwrap(),
                    is_leaf.to_data().try_to_vec::<i32>().unwrap(),
                    bounds_min.to_data().try_to_vec::<f32>().unwrap(),
                    bounds_max.to_data().try_to_vec::<f32>().unwrap(),
                    node_components.to_data().try_to_vec::<i32>().unwrap(),
                ))
            })
        },
    );
}

pub fn criterion_config() -> Criterion {
    config()
}
