#[path = "../../fast-hdbscan-core/tests/support/mod.rs"]
mod support;

use fast_hdbscan::estimator::{KnnNeighborSource, KnnSource, fit_knn_graph_with_source};
use fast_hdbscan::{
    BackendChoice, ClusterSelectionMethod, FittedHDBSCAN, HDBSCAN, HdbscanConfig, HdbscanSearch,
};
use fast_hdbscan_core::MstGraph;

fn fixture_config(metadata: &serde_json::Value) -> HdbscanConfig {
    let parameters = &metadata["parameters"];
    HdbscanConfig {
        min_cluster_size: parameters["min_cluster_size"].as_u64().unwrap() as usize,
        min_samples: Some(parameters["min_samples"].as_u64().unwrap() as usize),
        cluster_selection_method: match parameters["selection"].as_str().unwrap() {
            "eom" => ClusterSelectionMethod::Eom,
            "leaf" => ClusterSelectionMethod::Leaf,
            other => panic!("unknown selection method {other}"),
        },
        allow_single_cluster: parameters["allow_single_cluster"].as_bool().unwrap(),
        max_cluster_size: parameters["max_cluster_size"].as_u64().map(|n| n as usize),
        cluster_selection_epsilon: parameters["epsilon"].as_f64().unwrap(),
        cluster_selection_persistence: parameters["persistence"].as_f64().unwrap(),
        // Pinned fixtures are complete-graph references; the Auto route would pick KnnGraph at d>30.
        mst_graph: MstGraph::Complete,
        ..Default::default()
    }
}

fn read_f64(header: &support::NpyHeader) -> Vec<f64> {
    assert_eq!(header.descr, "<f8");
    assert_eq!(header.payload.len() % 8, 0);
    header
        .payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| f64::from_le_bytes(*v))
        .collect()
}

fn condensed_tree(header: &support::NpyHeader) -> fast_hdbscan_core::CondensedTree {
    assert_eq!(header.shape.len(), 1);
    let row_width = header.payload.len() / header.shape[0].max(1);
    assert!(
        row_width == 28 || row_width == 32,
        "unsupported structured condensed row width {row_width}"
    );
    assert_eq!(header.payload.len(), row_width * header.shape[0]);
    let mut tree = fast_hdbscan_core::CondensedTree::default();
    for row in header.payload.chunks_exact(row_width) {
        tree.parent
            .push(i64::from_le_bytes(row[0..8].try_into().unwrap()) as usize);
        tree.child
            .push(i64::from_le_bytes(row[8..16].try_into().unwrap()) as usize);
        tree.lambda
            .push(f64::from_le_bytes(row[16..24].try_into().unwrap()) as f32);
        tree.child_size.push(if row_width == 28 {
            f32::from_le_bytes(row[24..28].try_into().unwrap())
        } else {
            f64::from_le_bytes(row[24..32].try_into().unwrap()) as f32
        });
    }
    tree
}

fn canonical_condensed(
    tree: &fast_hdbscan_core::CondensedTree,
) -> Vec<(Vec<usize>, Vec<usize>, f32, f32)> {
    fn leaves(
        node: usize,
        tree: &fast_hdbscan_core::CondensedTree,
        visiting: &mut Vec<usize>,
    ) -> Vec<usize> {
        if visiting.contains(&node) {
            panic!("cycle in condensed fixture at node {node}");
        }
        let rows = tree
            .parent
            .iter()
            .enumerate()
            .filter_map(|(row, &parent)| (parent == node).then_some(row))
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return vec![node];
        }
        visiting.push(node);
        let mut points = Vec::new();
        for row in rows {
            let child = tree.child[row];
            if tree.child_size[row] > 1.0 {
                points.extend(leaves(child, tree, visiting));
            } else {
                points.push(child);
            }
        }
        visiting.pop();
        points.sort_unstable();
        points
    }

    let mut rows = (0..tree.len())
        .map(|row| {
            (
                leaves(tree.parent[row], tree, &mut Vec::new()),
                if tree.child_size[row] > 1.0 {
                    leaves(tree.child[row], tree, &mut Vec::new())
                } else {
                    vec![tree.child[row]]
                },
                tree.lambda[row],
                tree.child_size[row],
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
            .then(a.3.total_cmp(&b.3))
    });
    rows
}

fn assert_fixture(id: &str, backend: BackendChoice) {
    let root = support::fixture_root();
    let metadata = support::read_json(&root.join(format!("{id}.json")));
    let arrays = support::load_npz(&root.join(format!("{id}.npz"))).unwrap();
    let shape = &arrays["input"].shape;
    let flat = support::read_f32s(&arrays["input"]).unwrap();
    let input = flat
        .chunks_exact(shape[1])
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    let fitted = HDBSCAN::new(fixture_config(&metadata), backend)
        .with_search(HdbscanSearch::BruteForce)
        .fit(&input)
        .unwrap();

    let expected_labels = support::read_i64s(&arrays["labels"]).unwrap();
    assert_eq!(
        fitted.labels,
        expected_labels
            .iter()
            .map(|&v| v as i32)
            .collect::<Vec<_>>(),
        "{id}: labels"
    );
    let expected_probabilities = support::read_f32s(&arrays["probabilities"]).unwrap();
    assert_eq!(fitted.probabilities.len(), expected_probabilities.len());
    for (actual, expected) in fitted.probabilities.iter().zip(expected_probabilities) {
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "{id}: probability {actual} != {expected}"
        );
    }
    let expected_mst = read_f64(&arrays["mst"]);
    assert_eq!(expected_mst.len(), fitted.mst.len() * 3, "{id}: MST size");
    let tied = id == "duplicates_ties";
    if tied {
        use fast_hdbscan_core::oracle::brute_force_mst;
        let expected = brute_force_mst(
            &flat,
            shape[0],
            shape[1],
            fixture_config(&metadata).resolved_min_samples(),
        )
        .unwrap();
        assert_eq!(
            fitted.mst.len(),
            expected.len(),
            "{id}: oracle MST edge count"
        );
        for (actual, expected) in fitted.mst.iter().zip(expected) {
            assert_eq!(
                (actual.src, actual.dst),
                (expected.src, expected.dst),
                "{id}: deterministic host-oracle MST edge"
            );
            assert!(
                actual.weight == expected.weight
                    || (actual.weight - expected.weight).abs()
                        <= 1.0e-5 * expected.weight.abs().max(1.0),
                "{id}: oracle MST weight"
            );
        }
    }
    for (edge, expected) in fitted
        .mst
        .iter()
        .zip(expected_mst.as_chunks::<3>().0.iter())
    {
        if !tied {
            assert_eq!(
                [edge.src as f64, edge.dst as f64],
                [expected[0], expected[1]],
                "{id}: MST endpoints"
            );
        }
        assert!(
            edge.weight == expected[2]
                || (edge.weight - expected[2]).abs() <= 1.0e-5 * expected[2].abs().max(1.0),
            "{id}: MST weight"
        );
    }
    let expected_linkage = read_f64(arrays.get("remapped_linkage").unwrap_or(&arrays["linkage"]));
    assert_eq!(
        expected_linkage.len(),
        fitted.linkage.len() * 4,
        "{id}: linkage size"
    );
    for (row, expected) in fitted
        .linkage
        .iter()
        .zip(expected_linkage.as_chunks::<4>().0.iter())
    {
        if !tied {
            assert_eq!(
                [row.left as f64, row.right as f64, row.size as f64],
                [expected[0], expected[1], expected[3]],
                "{id}: linkage structure"
            );
        } else {
            assert_eq!(row.size as f64, expected[3], "{id}: linkage merge size");
        }
        assert!(
            row.distance == expected[2]
                || (row.distance - expected[2]).abs() <= 1.0e-5 * expected[2].abs().max(1.0),
            "{id}: linkage distance"
        );
    }
    let expected_condensed = condensed_tree(
        arrays
            .get("remapped_condensed_tree")
            .unwrap_or(&arrays["condensed_tree"]),
    );
    let actual_condensed = canonical_condensed(&fitted.condensed);
    let expected_condensed = canonical_condensed(&expected_condensed);
    assert_eq!(
        actual_condensed.len(),
        expected_condensed.len(),
        "{id}: condensed tree row count"
    );
    for (actual, expected) in actual_condensed.iter().zip(expected_condensed) {
        if !tied {
            assert_eq!(
                (&actual.0, &actual.1),
                (&expected.0, &expected.1),
                "{id}: normalized condensed parent/child leaves"
            );
        }
        assert!(
            actual.2 == expected.2
                || (actual.2 - expected.2).abs() <= 1.0e-5 * expected.2.abs().max(1.0),
            "{id}: condensed lambda {} != {}",
            actual.2,
            expected.2
        );
        assert!(
            (actual.3 - expected.3).abs() <= 1.0e-5,
            "{id}: condensed child size {} != {}",
            actual.3,
            expected.3
        );
    }
    assert_eq!(
        fitted.finite_index,
        support::read_i64s(&arrays["finite_index"])
            .unwrap()
            .iter()
            .map(|&v| v as u32)
            .collect::<Vec<_>>(),
        "{id}: finite-row mapping"
    );
}

fn fixture_suite(backend: BackendChoice) {
    for id in [
        "blobs_d2",
        "blobs_d10",
        "blobs_d50",
        "moons_leaf",
        "varying_density",
        "single_cluster",
        "duplicates_ties",
        "mixed_nonfinite",
        "overflow_finite",
    ] {
        assert_fixture(id, backend);
    }
}

#[test]
fn estimator_matches_pinned_fixtures_on_cpu() {
    fixture_suite(BackendChoice::Cpu);
}

#[test]
fn estimator_matches_pinned_fixtures_on_cuda() {
    let resolved = fast_cubecl_common::selected_device(BackendChoice::Cuda { device_index: 0 });
    let Ok(_) = resolved else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    fixture_suite(BackendChoice::Cuda { device_index: 0 });
}

#[test]
fn estimator_validates_configuration_and_input_and_dbscan_epsilon() {
    let invalid = HdbscanConfig {
        min_cluster_size: 0,
        ..Default::default()
    };
    assert!(
        HDBSCAN::new(invalid, BackendChoice::Cpu)
            .fit(&[vec![0.0]])
            .is_err()
    );
    let valid = HDBSCAN::new(HdbscanConfig::default(), BackendChoice::Cpu);
    assert!(valid.fit(&[]).is_err());
    assert!(valid.fit(&[vec![0.0], vec![0.0, 1.0]]).is_err());
    let fitted = valid.fit(&[vec![0.0], vec![1.0], vec![2.0]]).unwrap();
    assert!(fitted.dbscan_clustering(f64::NAN).is_err());
    for search in [
        HdbscanSearch::BruteForce,
        HdbscanSearch::KdTree,
        HdbscanSearch::Auto,
    ] {
        let unavailable = HDBSCAN::new(
            HdbscanConfig::default(),
            BackendChoice::Cuda {
                device_index: usize::MAX,
            },
        )
        .with_search(search)
        .fit(&[vec![0.0], vec![1.0], vec![2.0]]);
        assert!(
            matches!(
                unavailable,
                Err(fast_hdbscan::HdbscanError::BackendUnavailable(_))
            ),
            "an explicit unavailable device must not switch runtime or search route"
        );
    }
}

#[test]
fn estimator_min_samples_boundaries_match_independent_neighbor_oracle_on_cpu() {
    min_samples_boundary_cases(BackendChoice::Cpu);
}

#[test]
fn estimator_min_samples_boundaries_match_independent_neighbor_oracle_on_cuda() {
    let Ok(_) = fast_cubecl_common::selected_device(BackendChoice::Cuda { device_index: 0 }) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    min_samples_boundary_cases(BackendChoice::Cuda { device_index: 0 });
}

fn min_samples_boundary_cases(backend: BackendChoice) {
    use fast_hdbscan_core::oracle::{brute_force_mst, exact_neighbors};

    let cases = [
        (vec![vec![4.0, -2.0]], false),
        (vec![vec![0.0, 0.0], vec![3.0, 4.0]], false),
        (
            vec![
                vec![0.0, 0.0],
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![1.0, 0.0],
            ],
            false,
        ),
        (
            vec![
                vec![0.0, 0.0],
                vec![1.0, 0.25],
                vec![3.0, -0.5],
                vec![7.0, 2.0],
            ],
            true,
        ),
    ];
    for (input, check_host_mst_order) in &cases {
        let flat = input.iter().flatten().copied().collect::<Vec<_>>();
        let n = input.len();
        let mut min_samples_values = vec![1, n, n + 5];
        if n > 1 {
            min_samples_values.push(n - 1);
        }
        min_samples_values.sort_unstable();
        min_samples_values.dedup();
        for min_samples in min_samples_values {
            let expected = exact_neighbors(&flat, n, 2, min_samples).unwrap();
            let width = min_samples.min(n.saturating_sub(1));
            let mut expected_neighbors = if n == 1 {
                // The estimator deliberately retains its existing singleton contract:
                // no public neighbor rows are emitted at all.
                Vec::new()
            } else {
                vec![-1; n * min_samples]
            };
            if n > 1 {
                for row in 0..n {
                    expected_neighbors[row * min_samples..row * min_samples + width]
                        .copy_from_slice(&expected.neighbors[row * width..(row + 1) * width]);
                }
            }
            for search in [HdbscanSearch::BruteForce, HdbscanSearch::KdTree] {
                let config = HdbscanConfig {
                    min_cluster_size: 2,
                    min_samples: Some(min_samples),
                    ..Default::default()
                };
                let fitted = HDBSCAN::new(config, backend)
                    .with_search(search)
                    .fit(input)
                    .unwrap();
                assert_eq!(fitted.search(), search);
                assert_eq!(
                    fitted.neighbors(),
                    expected_neighbors,
                    "n={n}, min_samples={min_samples}, search={search:?}"
                );
                for (row, &actual) in fitted.core_distances().iter().enumerate() {
                    let expected_core = if min_samples >= n {
                        f32::INFINITY
                    } else if min_samples == 1 {
                        0.0
                    } else {
                        expected.core_sq[row].sqrt()
                    };
                    assert_eq!(
                        actual, expected_core,
                        "n={n}, min_samples={min_samples}, row={row}, search={search:?}"
                    );
                }

                // Tied MST topology has a separate device Boruvka tie contract. For the
                // tie-free case, assert the full host-sorted edge sequence and orientation.
                if *check_host_mst_order && min_samples == 1 {
                    let oracle = brute_force_mst(&flat, n, 2, min_samples).unwrap();
                    assert_eq!(
                        fitted.minimum_spanning_tree(),
                        oracle,
                        "host MST source order for search={search:?}"
                    );
                }
            }
        }
    }
    estimator_batch_boundary_cases(backend);
}

fn estimator_batch_boundary_cases(backend: BackendChoice) {
    use fast_hdbscan_core::oracle::exact_neighbors;

    for n in [63usize, 64, 65, 257, 258, 513, 514] {
        let input = (0..n)
            .map(|row| {
                vec![
                    ((row * 37 + row / 5) % 101) as f32,
                    ((row * 19 + row / 7) % 83) as f32,
                ]
            })
            .collect::<Vec<_>>();
        let flat = input.iter().flatten().copied().collect::<Vec<_>>();
        let min_samples = 3;
        let expected = exact_neighbors(&flat, n, 2, min_samples).unwrap();
        let config = HdbscanConfig {
            min_cluster_size: 2,
            min_samples: Some(min_samples),
            ..Default::default()
        };
        let search = HdbscanSearch::BruteForce;
        let fitted = HDBSCAN::new(config, backend)
            .with_search(search)
            .fit(&input)
            .unwrap();
        assert_eq!(fitted.search(), search);
        assert_eq!(fitted.neighbors(), expected.neighbors, "n={n}, {search:?}");
        for (row, &core) in fitted.core_distances().iter().enumerate() {
            assert_eq!(
                core,
                expected.core_sq[row].sqrt(),
                "n={n}, row={row}, {search:?}"
            );
        }
    }
}

#[test]
fn estimator_handles_finite_row_remapping_and_small_finite_sets_on_cpu() {
    finite_remapping_cases(BackendChoice::Cpu);
}

#[test]
fn estimator_handles_finite_row_remapping_and_small_finite_sets_on_cuda() {
    let Ok(_) = fast_cubecl_common::selected_device(BackendChoice::Cuda { device_index: 0 }) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    finite_remapping_cases(BackendChoice::Cuda { device_index: 0 });
}

fn finite_remapping_cases(backend: BackendChoice) {
    let config = HdbscanConfig {
        min_cluster_size: 2,
        min_samples: Some(2),
        ..Default::default()
    };
    let mixed = [
        vec![0.0, 0.0],
        vec![f32::NAN, 1.0],
        vec![2.0, 0.0],
        vec![f32::INFINITY, 0.0],
    ];
    let fitted = HDBSCAN::new(config.clone(), backend).fit(&mixed).unwrap();
    assert_eq!(fitted.finite_index, [0, 2]);
    assert_eq!(fitted.labels.len(), mixed.len());
    assert_eq!(fitted.labels[1], -1);
    assert_eq!(fitted.labels[3], -1);

    let singleton = HDBSCAN::new(config.clone(), backend)
        .fit(&[vec![4.0, 2.0]])
        .unwrap();
    assert_eq!(singleton.labels, [-1]);
    assert!(singleton.mst.is_empty());
    assert!(singleton.neighbors.is_empty());
    assert_eq!(singleton.core_distances, [f32::INFINITY]);

    let empty = HDBSCAN::new(config, backend).fit(&[vec![f32::NAN, 0.0]]);
    assert!(empty.is_err());
}

fn compare_search_routes(
    input: &[Vec<f32>],
    config: HdbscanConfig,
    backend: BackendChoice,
    tied: bool,
) {
    use fast_hdbscan_core::oracle::brute_force_mst;

    let brute = HDBSCAN::new(config.clone(), backend)
        .with_search(HdbscanSearch::BruteForce)
        .fit(input)
        .unwrap();
    let kd = HDBSCAN::new(config.clone(), backend)
        .with_search(HdbscanSearch::KdTree)
        .fit(input)
        .unwrap();
    assert_eq!(brute.search(), HdbscanSearch::BruteForce);
    assert_eq!(kd.search(), HdbscanSearch::KdTree);

    assert_eq!(brute.neighbors(), kd.neighbors(), "route neighbor IDs");
    assert_eq!(brute.core_distances().len(), kd.core_distances().len());
    for (&left, &right) in brute.core_distances().iter().zip(kd.core_distances()) {
        assert!(
            left == right
                || (left.is_finite()
                    && right.is_finite()
                    && (left - right).abs() <= 1.0e-6 * left.abs().max(1.0)),
            "route core distances: {left:?} != {right:?}"
        );
    }
    assert_eq!(
        brute.minimum_spanning_tree().len(),
        kd.minimum_spanning_tree().len()
    );
    if tied {
        let flat = input.iter().flatten().copied().collect::<Vec<_>>();
        let dimensions = input[0].len();
        let oracle = brute_force_mst(
            &flat,
            input.len(),
            dimensions,
            config.resolved_min_samples(),
        )
        .unwrap();
        // The independent oracle starts from singleton components, whereas both device
        // routes seed components from kNN. Tied topology can differ between these
        // algorithms, but the two device routes must preserve identical tie choices.
        for (left, right) in brute
            .minimum_spanning_tree()
            .iter()
            .zip(kd.minimum_spanning_tree())
        {
            assert_eq!(
                (left.src, left.dst),
                (right.src, right.dst),
                "tied route MST endpoints"
            );
        }
        let mut brute_weights = brute
            .minimum_spanning_tree()
            .iter()
            .map(|edge| edge.weight)
            .collect::<Vec<_>>();
        let mut kd_weights = kd
            .minimum_spanning_tree()
            .iter()
            .map(|edge| edge.weight)
            .collect::<Vec<_>>();
        brute_weights.sort_by(f64::total_cmp);
        kd_weights.sort_by(f64::total_cmp);
        assert_eq!(brute_weights, kd_weights, "tied route MST weights");
        let mut oracle_weights = oracle.iter().map(|edge| edge.weight).collect::<Vec<_>>();
        oracle_weights.sort_by(f64::total_cmp);
        assert_eq!(
            brute_weights, oracle_weights,
            "independent tied MST weight multiset"
        );
        assert_eq!(brute.labels().len(), kd.labels().len());
        assert_eq!(brute.probabilities().len(), kd.probabilities().len());
        assert_eq!(brute.labels(), kd.labels(), "tied route labels");
        assert_eq!(brute.linkage(), kd.linkage(), "tied route linkage");
        assert_eq!(
            canonical_condensed(brute.condensed_tree()),
            canonical_condensed(kd.condensed_tree()),
            "tied route condensed trees"
        );
        for (actual, expected) in brute.probabilities().iter().zip(kd.probabilities()) {
            assert!(
                actual == expected || (actual - expected).abs() <= 1.0e-5,
                "tied route probability {actual} != {expected}"
            );
        }
        assert_tree_invariants(brute.linkage(), input.len());
        assert_tree_invariants(kd.linkage(), input.len());
        return;
    }

    assert_eq!(brute.labels(), kd.labels(), "tie-free route labels");
    assert_eq!(brute.linkage(), kd.linkage(), "tie-free route linkage");
    assert_eq!(
        canonical_condensed(brute.condensed_tree()),
        canonical_condensed(kd.condensed_tree()),
        "tie-free route condensed trees"
    );
    assert_eq!(brute.finite_index(), kd.finite_index());
    for (&left, &right) in brute.probabilities().iter().zip(kd.probabilities()) {
        assert!(
            left == right
                || (left.is_finite() && right.is_finite() && (left - right).abs() <= 1.0e-5),
            "route probabilities: {left:?} != {right:?}"
        );
    }
    for (left, right) in brute
        .minimum_spanning_tree()
        .iter()
        .zip(kd.minimum_spanning_tree())
    {
        assert_eq!(
            (left.src, left.dst),
            (right.src, right.dst),
            "tie-free route MST edge"
        );
        assert!(
            left.weight == right.weight
                || (left.weight - right.weight).abs() <= 1.0e-5 * left.weight.abs().max(1.0),
            "route MST weight"
        );
    }
    assert_tree_invariants(brute.linkage(), input.len());
    assert_tree_invariants(kd.linkage(), input.len());
}

fn assert_tree_invariants(linkage: &[fast_hdbscan_core::LinkageRow], n: usize) {
    assert_eq!(linkage.len(), n.saturating_sub(1));
    let mut sizes = vec![1usize; n.saturating_mul(2).saturating_sub(1)];
    for (row, merge) in linkage.iter().enumerate() {
        let parent = n + row;
        assert!(merge.left < parent && merge.right < parent);
        assert_ne!(merge.left, merge.right);
        assert_eq!(merge.size, sizes[merge.left] + sizes[merge.right]);
        sizes[parent] = merge.size;
        if row > 0 {
            assert!(linkage[row - 1].distance <= merge.distance);
        }
    }
    if n > 0 {
        assert_eq!(sizes[n + linkage.len() - 1], n);
    }
}

fn route_parity_cases(backend: BackendChoice) {
    let root = support::fixture_root();
    for id in ["blobs_d10", "blobs_d50"] {
        let metadata = support::read_json(&root.join(format!("{id}.json")));
        let arrays = support::load_npz(&root.join(format!("{id}.npz"))).unwrap();
        let shape = &arrays["input"].shape;
        let flat = support::read_f32s(&arrays["input"]).unwrap();
        let input = flat
            .chunks_exact(shape[1])
            .map(<[f32]>::to_vec)
            .collect::<Vec<_>>();
        compare_search_routes(&input, fixture_config(&metadata), backend, false);
    }

    let random = (0..47)
        .map(|row| {
            (0..7)
                .map(|feature| {
                    let value = ((row * 101 + feature * 37 + row * feature * 13) % 997) as f32;
                    value / 31.0 + (feature as f32) * 0.017
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    compare_search_routes(
        &random,
        HdbscanConfig {
            min_cluster_size: 4,
            min_samples: Some(5),
            ..Default::default()
        },
        backend,
        false,
    );

    let duplicates = (0..24)
        .map(|row| {
            let point = row % 8;
            vec![(point % 4) as f32, (point / 4) as f32]
        })
        .collect::<Vec<_>>();
    compare_search_routes(
        &duplicates,
        HdbscanConfig {
            min_cluster_size: 3,
            min_samples: Some(4),
            ..Default::default()
        },
        backend,
        true,
    );
}

#[test]
fn forced_search_routes_match_on_cpu() {
    route_parity_cases(BackendChoice::Cpu);
}

#[test]
fn forced_search_routes_match_on_cuda() {
    if fast_cubecl_common::selected_device(BackendChoice::Cuda { device_index: 0 }).is_err() {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    route_parity_cases(BackendChoice::Cuda { device_index: 0 });
}

fn cuda_backend_or_skip() -> Option<BackendChoice> {
    let backend = BackendChoice::Cuda { device_index: 0 };
    if fast_cubecl_common::selected_device(backend).is_ok() {
        return Some(backend);
    }
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
    );
    eprintln!("SKIPPED: no CUDA device");
    None
}

fn deterministic_rows(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|row| {
            (0..d)
                .map(|feature| {
                    ((row * 37 + feature * 17 + row * feature * 5) % 1009) as f32 * 0.03125
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn fit_cpu(input: &[Vec<f32>], config: HdbscanConfig) -> FittedHDBSCAN {
    HDBSCAN::new(config, BackendChoice::Cpu).fit(input).unwrap()
}

#[test]
fn knn_graph_routing_and_configuration() {
    let thirty = deterministic_rows(6, 30);
    let thirty_one = deterministic_rows(6, 31);
    let default = HdbscanConfig {
        min_cluster_size: 2,
        min_samples: Some(2),
        ..Default::default()
    };

    // Auto resolves to Complete at d=30 and KnnGraph at d=31.
    let at_30 = fit_cpu(&thirty, default.clone());
    assert_eq!(at_30.mst_graph(), MstGraph::Auto);
    assert_eq!(at_30.resolved_mst_graph(), MstGraph::Complete);
    assert!(at_30.knn_diagnostics().is_none());

    let at_31 = fit_cpu(&thirty_one, default.clone());
    assert_eq!(at_31.mst_graph(), MstGraph::Auto);
    assert_eq!(at_31.resolved_mst_graph(), MstGraph::KnnGraph);
    assert!(at_31.knn_diagnostics().is_some());

    // Explicit values override the dimension rule in both directions.
    let forced_complete = fit_cpu(
        &thirty_one,
        HdbscanConfig {
            mst_graph: MstGraph::Complete,
            ..default.clone()
        },
    );
    assert_eq!(forced_complete.mst_graph(), MstGraph::Complete);
    assert_eq!(forced_complete.resolved_mst_graph(), MstGraph::Complete);
    assert!(forced_complete.knn_diagnostics().is_none());

    let forced_knn = fit_cpu(
        &thirty,
        HdbscanConfig {
            mst_graph: MstGraph::KnnGraph,
            ..default.clone()
        },
    );
    assert_eq!(forced_knn.mst_graph(), MstGraph::KnnGraph);
    assert_eq!(forced_knn.resolved_mst_graph(), MstGraph::KnnGraph);
    assert!(forced_knn.knn_diagnostics().is_some());

    // knn_k is ignored under Complete: the output must be bitwise identical to the default knn_k.
    let complete_plain = fit_cpu(
        &thirty_one,
        HdbscanConfig {
            mst_graph: MstGraph::Complete,
            ..default.clone()
        },
    );
    let complete_with_knn_k = fit_cpu(
        &thirty_one,
        HdbscanConfig {
            mst_graph: MstGraph::Complete,
            knn_k: Some(3),
            ..default.clone()
        },
    );
    assert_eq!(complete_with_knn_k.labels(), complete_plain.labels());
    assert_eq!(
        complete_with_knn_k.minimum_spanning_tree(),
        complete_plain.minimum_spanning_tree()
    );
    assert_eq!(complete_with_knn_k.neighbors(), complete_plain.neighbors());
    assert_eq!(
        complete_with_knn_k.core_distances(),
        complete_plain.core_distances()
    );
    assert_eq!(complete_with_knn_k.linkage(), complete_plain.linkage());

    // Search resolution is consulted only for Complete.
    let complete_kd = HDBSCAN::new(
        HdbscanConfig {
            mst_graph: MstGraph::Complete,
            ..default.clone()
        },
        BackendChoice::Cpu,
    )
    .with_search(HdbscanSearch::KdTree)
    .fit(&thirty_one)
    .unwrap();
    assert_eq!(complete_kd.search(), HdbscanSearch::KdTree);

    // knn_k = Some(0) is invalid under every graph route.
    for mst_graph in [MstGraph::Auto, MstGraph::Complete, MstGraph::KnnGraph] {
        let result = HDBSCAN::new(
            HdbscanConfig {
                mst_graph,
                knn_k: Some(0),
                ..default.clone()
            },
            BackendChoice::Cpu,
        )
        .fit(&thirty_one);
        assert!(
            matches!(result, Err(fast_hdbscan::HdbscanError::InvalidConfig(_))),
            "knn_k = Some(0) must be InvalidConfig for {mst_graph:?}"
        );
    }
}

fn knn_config(min_samples: usize, knn_k: Option<usize>) -> HdbscanConfig {
    HdbscanConfig {
        min_cluster_size: 2,
        min_samples: Some(min_samples),
        mst_graph: MstGraph::KnnGraph,
        knn_k,
        ..Default::default()
    }
}

/// Effective width k_graph = min(max(knn_k or max(3s, 15), s), m-1).
fn knn_width(min_samples: usize, knn_k: Option<usize>, rows: usize) -> usize {
    knn_k
        .unwrap_or(min_samples.saturating_mul(3).max(15))
        .max(min_samples)
        .min(rows.saturating_sub(1))
}

/// Fits the forced exact KnnGraph source and checks it bitwise against the test-support oracle.
/// Non-finite rows are excluded from the oracle and remapped through `finite_index`.
fn assert_knn_graph_matches_oracle(
    input: &[Vec<f32>],
    min_samples: usize,
    knn_k: Option<usize>,
    backend: BackendChoice,
    case: &str,
) -> FittedHDBSCAN {
    use fast_hdbscan_core::{
        clusters_from_spanning_tree,
        finite::{scatter_labels, scatter_probabilities},
        oracle::knn_kruskal_oracle,
    };

    let config = knn_config(min_samples, knn_k);
    let forced =
        fit_knn_graph_with_source(input, config.clone(), backend, KnnSource::Exact).unwrap();
    let public = HDBSCAN::new(config.clone(), backend).fit(input).unwrap();
    assert_eq!(public.labels(), forced.labels(), "{case}: public labels");
    assert_eq!(
        public.minimum_spanning_tree(),
        forced.minimum_spanning_tree(),
        "{case}: public MST"
    );
    assert_eq!(
        public.neighbors(),
        forced.neighbors(),
        "{case}: public neighbors"
    );
    assert_eq!(
        public.core_distances(),
        forced.core_distances(),
        "{case}: public cores"
    );

    let finite_rows = input
        .iter()
        .filter(|row| row.iter().all(|value| value.is_finite()))
        .collect::<Vec<_>>();
    let rows = finite_rows.len();
    if rows < 2 {
        // Existing one-row degenerate contract: noise, no MST, no graph diagnostics.
        assert!(forced.labels().iter().all(|&label| label == -1), "{case}");
        assert!(forced.minimum_spanning_tree().is_empty(), "{case}");
        assert!(forced.knn_diagnostics().is_none(), "{case}");
        return forced;
    }

    let dimensions = input[0].len();
    let flat = finite_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect::<Vec<f32>>();
    let width = knn_width(min_samples, knn_k, rows);
    let oracle = knn_kruskal_oracle(&flat, rows, dimensions, min_samples, width).unwrap();

    let diagnostics = forced.knn_diagnostics().expect("KnnGraph diagnostics");
    assert_eq!(
        diagnostics.source,
        KnnNeighborSource::Exact,
        "{case}: source"
    );
    assert_eq!(diagnostics.k, width, "{case}: width");
    assert_eq!(diagnostics.neighbors, oracle.ids, "{case}: neighbor ids");
    assert_eq!(
        diagnostics.distances_sq, oracle.dist_sq,
        "{case}: squared distances"
    );
    assert_eq!(diagnostics.bridges, oracle.bridges, "{case}: bridges");
    let finite_sum = oracle
        .edges
        .iter()
        .map(|edge| edge.weight)
        .filter(|weight| weight.is_finite())
        .sum::<f64>();
    assert!(
        (diagnostics.finite_weight_sum - finite_sum).abs() <= 1.0e-9 * finite_sum.abs().max(1.0),
        "{case}: finite weight sum {} != {finite_sum}",
        diagnostics.finite_weight_sum
    );

    assert_eq!(
        forced.minimum_spanning_tree(),
        oracle.edges.as_slice(),
        "{case}: MST edges"
    );
    let expected = clusters_from_spanning_tree(&oracle.edges, &config).unwrap();
    let labels = scatter_labels(&expected.labels, forced.finite_index(), input.len()).unwrap();
    assert_eq!(forced.labels(), labels.as_slice(), "{case}: labels");
    let probabilities =
        scatter_probabilities(&expected.probabilities, forced.finite_index(), input.len()).unwrap();
    assert_eq!(
        forced.probabilities(),
        probabilities.as_slice(),
        "{case}: probabilities"
    );

    if rows == input.len() {
        // Without remapping, the finite-row indexing equals raw row indexing.
        assert_eq!(forced.core_distances().len(), oracle.core.len(), "{case}");
        for (row, (&actual, &expected_core)) in
            forced.core_distances().iter().zip(&oracle.core).enumerate()
        {
            assert_eq!(actual, expected_core as f32, "{case}: core row {row}");
        }
        assert_eq!(
            forced.linkage(),
            expected.linkage.as_slice(),
            "{case}: linkage"
        );
        assert_eq!(
            canonical_condensed(forced.condensed_tree()),
            canonical_condensed(&expected.condensed),
            "{case}: condensed tree"
        );
        assert_tree_invariants(forced.linkage(), rows);
    }
    forced
}

fn knn_graph_oracle_cases(backend: BackendChoice) {
    use fast_hdbscan_core::oracle::knn_kruskal_oracle;

    // Three far clusters of four points; k=2 keeps each cluster's kNN graph separate.
    let disconnected = (0..12)
        .map(|row| {
            vec![
                (row / 4) as f32 * 1000.0 + (row % 4) as f32 * 0.5,
                (row % 3) as f32 * 0.25,
            ]
        })
        .collect::<Vec<_>>();
    let flat = disconnected.iter().flatten().copied().collect::<Vec<_>>();
    assert!(
        knn_kruskal_oracle(&flat, 12, 2, 2, 2).unwrap().bridges > 0,
        "disconnected case must produce bridges"
    );
    assert_knn_graph_matches_oracle(&disconnected, 2, Some(2), backend, "disconnected");

    // knn_k below, equal to and above s; default width.
    let random = deterministic_rows(20, 2);
    for (knn_k, label) in [
        (Some(3), "knn_k below s"),
        (Some(5), "knn_k equal s"),
        (Some(9), "knn_k above s"),
        (None, "knn_k default"),
    ] {
        assert_knn_graph_matches_oracle(&random, 5, knn_k, backend, label);
    }

    // Width clipped to m-1.
    assert_knn_graph_matches_oracle(
        &deterministic_rows(9, 2),
        3,
        Some(50),
        backend,
        "width clipped to m-1",
    );

    // s in {1, m-1, m, >m}; s >= m gives infinite cores and only bridges.
    let twelve = deterministic_rows(12, 3);
    for (min_samples, label) in [(1, "s=1"), (11, "s=m-1"), (12, "s=m"), (14, "s>m")] {
        assert_knn_graph_matches_oracle(&twelve, min_samples, None, backend, label);
    }

    let duplicates = (0..8)
        .map(|row| vec![(row / 2) as f32, ((row / 2) % 2) as f32])
        .collect::<Vec<_>>();
    assert_knn_graph_matches_oracle(&duplicates, 2, Some(3), backend, "duplicate rows");

    // Every pairwise f32 squared distance among the first four rows overflows to +inf.
    let overflow = vec![
        vec![1.0e20f32, 0.0],
        vec![-1.0e20, 0.0],
        vec![0.0, 1.0e20],
        vec![0.0, -1.0e20],
        vec![5.0, 5.0],
    ];
    assert_knn_graph_matches_oracle(&overflow, 2, None, backend, "squared-distance overflow");

    let two = vec![vec![0.0f32, 0.0], vec![3.0, 4.0]];
    for min_samples in [1, 2, 3] {
        assert_knn_graph_matches_oracle(&two, min_samples, None, backend, "two finite rows");
    }
    assert_knn_graph_matches_oracle(&[vec![1.0f32, 2.0]], 1, None, backend, "one finite row");

    // Non-finite rows are excluded; the oracle runs on finite rows, remapped via finite_index.
    let mixed = vec![
        vec![0.0f32, 0.0],
        vec![f32::NAN, 1.0],
        vec![2.0, 0.0],
        vec![f32::INFINITY, 0.0],
        vec![1.0, 1.0],
        vec![3.0, 3.0],
        vec![-2.0, 5.0],
        vec![f32::NAN, f32::NAN],
    ];
    for (min_samples, knn_k) in [(2, None), (1, Some(3))] {
        let fitted = assert_knn_graph_matches_oracle(
            &mixed,
            min_samples,
            knn_k,
            backend,
            "finite-row remapping",
        );
        assert_eq!(fitted.finite_index(), &[0u32, 2, 4, 5, 6][..]);
        assert_eq!(fitted.labels().len(), mixed.len());
    }
}

#[test]
fn knn_graph_exact_matches_oracle_on_cpu() {
    knn_graph_oracle_cases(BackendChoice::Cpu);
}

#[test]
fn knn_graph_exact_matches_oracle_on_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        knn_graph_oracle_cases(backend);
    }
}

/// Shape and finiteness checks shared by every route; `source` is the expected neighbor source.
fn assert_large_width_diagnostics(
    fitted: &FittedHDBSCAN,
    n: usize,
    width: usize,
    source: KnnNeighborSource,
    case: &str,
) {
    assert_eq!(fitted.resolved_mst_graph(), MstGraph::KnnGraph, "{case}");
    let diagnostics = fitted.knn_diagnostics().expect("KnnGraph diagnostics");
    assert_eq!(diagnostics.source, source, "{case}: neighbor source");
    assert_eq!(diagnostics.k, width, "{case}: effective width");
    assert_eq!(diagnostics.neighbors.len(), n * width, "{case}");
    assert_eq!(diagnostics.distances_sq.len(), n * width, "{case}");
    assert!(
        diagnostics
            .neighbors
            .iter()
            .all(|&id| id >= 0 && (id as usize) < n),
        "{case}: finite rows must have no padding"
    );
    assert_eq!(fitted.minimum_spanning_tree().len(), n - 1, "{case}");
}

fn knn_graph_large_width_source_selection_cases(backend: BackendChoice) {
    // m > 4096 with effective k_graph <= 60 is eligible for ANN under Auto; k_graph > 60 stays exact.
    let n = 4097;
    let d = 31;
    let input = deterministic_rows(n, d);

    // s=5 gives k_graph = max(max(3s, 15), s) = 15, so Auto selects ANN.
    let width15 = knn_width(5, None, n);
    let config15 = HdbscanConfig {
        min_cluster_size: 2,
        min_samples: Some(5),
        ..Default::default()
    };
    let auto15 = HDBSCAN::new(config15, backend).fit(&input).unwrap();
    assert_large_width_diagnostics(
        &auto15,
        n,
        width15,
        KnnNeighborSource::Approximate,
        "auto width 15",
    );

    // s=25 gives k_graph = 75 > 60, so Auto must select exact and match a forced exact fit.
    let width75 = knn_width(25, None, n);
    assert_eq!(width75, 75);
    let config75 = HdbscanConfig {
        min_cluster_size: 2,
        min_samples: Some(25),
        ..Default::default()
    };
    let auto75 = HDBSCAN::new(config75.clone(), backend).fit(&input).unwrap();
    assert_large_width_diagnostics(
        &auto75,
        n,
        width75,
        KnnNeighborSource::Exact,
        "auto width 75",
    );
    let forced75 =
        fit_knn_graph_with_source(&input, knn_config(25, None), backend, KnnSource::Exact).unwrap();
    assert_large_width_diagnostics(
        &forced75,
        n,
        width75,
        KnnNeighborSource::Exact,
        "forced exact width 75",
    );
    assert_fits_bitwise_equal(&auto75, &forced75, "auto vs forced exact width 75");
    let auto_diag = auto75.knn_diagnostics().unwrap();
    let forced_diag = forced75.knn_diagnostics().unwrap();
    assert_eq!(auto_diag.neighbors, forced_diag.neighbors, "width 75 ids");
    assert_eq!(
        f32_bits(&auto_diag.distances_sq),
        f32_bits(&forced_diag.distances_sq),
        "width 75 squared-distance bits"
    );
    assert_eq!(auto_diag.bridges, forced_diag.bridges, "width 75 bridges");
    assert_eq!(
        auto_diag.finite_weight_sum.to_bits(),
        forced_diag.finite_weight_sum.to_bits(),
        "width 75 finite weight sum bits"
    );
}

/// Separated Gaussian-like blobs from deterministic offsets; ANN recall is not asserted here,
/// only that repeated forced-ANN fits are bitwise identical.
fn separated_blob_rows(n: usize, d: usize) -> Vec<Vec<f32>> {
    (0..n)
        .map(|row| {
            let center = (row % 4) as f32 * 100.0;
            (0..d)
                .map(|feature| center + ((row * 37 + feature * 11) % 17) as f32 * 0.25)
                .collect()
        })
        .collect()
}

fn knn_graph_forced_ann_repeatable_cases(backend: BackendChoice) {
    let n = 600;
    let d = 8;
    let input = separated_blob_rows(n, d);
    let width = knn_width(5, None, n);
    let first =
        fit_knn_graph_with_source(&input, knn_config(5, None), backend, KnnSource::Ann).unwrap();
    let second =
        fit_knn_graph_with_source(&input, knn_config(5, None), backend, KnnSource::Ann).unwrap();
    assert_large_width_diagnostics(
        &first,
        n,
        width,
        KnnNeighborSource::Approximate,
        "forced ANN",
    );
    assert_large_width_diagnostics(
        &second,
        n,
        width,
        KnnNeighborSource::Approximate,
        "forced ANN repeat",
    );
    let first_diag = first.knn_diagnostics().unwrap();
    let second_diag = second.knn_diagnostics().unwrap();
    assert_eq!(first_diag.neighbors, second_diag.neighbors, "ANN ids");
    assert_eq!(
        f32_bits(&first_diag.distances_sq),
        f32_bits(&second_diag.distances_sq),
        "ANN squared-distance bits"
    );
    assert_eq!(first_diag.bridges, second_diag.bridges, "ANN bridges");
    assert_eq!(
        first_diag.finite_weight_sum.to_bits(),
        second_diag.finite_weight_sum.to_bits(),
        "ANN finite weight sum bits"
    );
    assert_fits_bitwise_equal(&first, &second, "forced ANN repeat");
}

#[test]
fn knn_graph_large_width_source_selection_on_cpu() {
    knn_graph_large_width_source_selection_cases(BackendChoice::Cpu);
}

#[test]
fn knn_graph_large_width_source_selection_on_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        knn_graph_large_width_source_selection_cases(backend);
    }
}

#[test]
fn knn_graph_forced_ann_repeatable_on_cpu() {
    knn_graph_forced_ann_repeatable_cases(BackendChoice::Cpu);
}

#[test]
fn knn_graph_forced_ann_repeatable_on_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        knn_graph_forced_ann_repeatable_cases(backend);
    }
}

/// Loads a pinned fixture as its complete-graph estimator config and raw input rows.
fn fixture_case(id: &str) -> (HdbscanConfig, Vec<Vec<f32>>) {
    let root = support::fixture_root();
    let metadata = support::read_json(&root.join(format!("{id}.json")));
    let arrays = support::load_npz(&root.join(format!("{id}.npz"))).unwrap();
    let shape = &arrays["input"].shape;
    let flat = support::read_f32s(&arrays["input"]).unwrap();
    let input = flat
        .chunks_exact(shape[1])
        .map(<[f32]>::to_vec)
        .collect::<Vec<_>>();
    (fixture_config(&metadata), input)
}

/// Tie-heavy integer lattice rows from a fixed-seed LCG; the same seed always gives the same rows.
fn seeded_tie_rows(seed: u64, n: usize, d: usize) -> Vec<Vec<f32>> {
    let mut state = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    ((state >> 33) % 4) as f32
                })
                .collect()
        })
        .collect()
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Bitwise comparison of every fitted output that the Complete/KD route produces.
fn assert_fits_bitwise_equal(actual: &FittedHDBSCAN, expected: &FittedHDBSCAN, case: &str) {
    assert_eq!(actual.labels(), expected.labels(), "{case}: labels");
    assert_eq!(
        actual.finite_index(),
        expected.finite_index(),
        "{case}: finite index"
    );
    assert_eq!(
        actual.neighbors(),
        expected.neighbors(),
        "{case}: neighbors"
    );
    assert_eq!(
        f32_bits(actual.probabilities()),
        f32_bits(expected.probabilities()),
        "{case}: probability bits"
    );
    assert_eq!(
        f32_bits(actual.core_distances()),
        f32_bits(expected.core_distances()),
        "{case}: core distance bits"
    );
    let mst_bits = |fit: &FittedHDBSCAN| {
        fit.minimum_spanning_tree()
            .iter()
            .map(|edge| (edge.src, edge.dst, edge.weight.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(mst_bits(actual), mst_bits(expected), "{case}: MST bits");
    let linkage_bits = |fit: &FittedHDBSCAN| {
        fit.linkage()
            .iter()
            .map(|row| (row.left, row.right, row.distance.to_bits(), row.size))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        linkage_bits(actual),
        linkage_bits(expected),
        "{case}: linkage bits"
    );
}

fn complete_kd_fit(
    config: &HdbscanConfig,
    input: &[Vec<f32>],
    backend: BackendChoice,
    reproducible: bool,
) -> FittedHDBSCAN {
    HDBSCAN::new(
        HdbscanConfig {
            mst_graph: MstGraph::Complete,
            reproducible,
            ..config.clone()
        },
        backend,
    )
    .with_search(HdbscanSearch::KdTree)
    .fit(input)
    .unwrap()
}

fn assert_shared_and_reproducible_match(
    config: &HdbscanConfig,
    input: &[Vec<f32>],
    backend: BackendChoice,
    case: &str,
) {
    let shared = complete_kd_fit(config, input, backend, false);
    let reproducible = complete_kd_fit(config, input, backend, true);
    assert_eq!(
        shared.search(),
        HdbscanSearch::KdTree,
        "{case}: shared route"
    );
    assert_eq!(
        reproducible.search(),
        HdbscanSearch::KdTree,
        "{case}: reproducible route"
    );
    assert_fits_bitwise_equal(&shared, &reproducible, case);
}

fn shared_and_reproducible_complete_kd_cases(backend: BackendChoice) {
    for id in [
        "blobs_d2",
        "blobs_d10",
        "duplicates_ties",
        "persistence_equal",
        "varying_density",
    ] {
        let (config, input) = fixture_case(id);
        assert_shared_and_reproducible_match(&config, &input, backend, id);
    }
    for seed in [3u64, 17, 101] {
        for (n, d) in [(96usize, 2usize), (160, 3)] {
            let input = seeded_tie_rows(seed, n, d);
            let config = HdbscanConfig {
                min_cluster_size: 3,
                min_samples: Some(4),
                ..Default::default()
            };
            assert_shared_and_reproducible_match(
                &config,
                &input,
                backend,
                &format!("seed {seed}, n {n}, d {d}"),
            );
        }
    }
}

#[test]
fn shared_and_reproducible_complete_kd_mst_bitwise_on_cpu() {
    shared_and_reproducible_complete_kd_cases(BackendChoice::Cpu);
}

#[test]
fn shared_and_reproducible_complete_kd_mst_bitwise_on_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        shared_and_reproducible_complete_kd_cases(backend);
    }
}

fn reproducible_kd_region_cases(backend: BackendChoice) {
    // n=300, d=3, s=5 lies inside the Auto KD region (256..=2048 rows, d<=30, s<=32), so the
    // default search resolves to KdTree and the fit runs real Boruvka rounds.
    let input = seeded_tie_rows(7, 300, 3);
    let config = HdbscanConfig {
        min_cluster_size: 4,
        min_samples: Some(5),
        mst_graph: MstGraph::Complete,
        ..Default::default()
    };
    let shared = HDBSCAN::new(config.clone(), backend).fit(&input).unwrap();
    assert_eq!(shared.search(), HdbscanSearch::KdTree);
    assert!(!shared.config().reproducible);
    assert_eq!(shared.minimum_spanning_tree().len(), input.len() - 1);
    assert!(!shared.linkage().is_empty());

    let reproducible = HDBSCAN::new(
        HdbscanConfig {
            reproducible: true,
            ..config
        },
        backend,
    )
    .fit(&input)
    .unwrap();
    assert_eq!(reproducible.search(), HdbscanSearch::KdTree);
    assert!(reproducible.config().reproducible);
    assert_eq!(reproducible.minimum_spanning_tree().len(), input.len() - 1);
    assert_fits_bitwise_equal(&shared, &reproducible, "KD region reproducible flag");
}

#[test]
fn reproducible_flag_in_kd_region_on_cpu() {
    reproducible_kd_region_cases(BackendChoice::Cpu);
}

#[test]
fn reproducible_flag_in_kd_region_on_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        reproducible_kd_region_cases(backend);
    }
}

/// Observational KD bound counters. Ignored by default; run with `--features tracing`,
/// `FAST_HDBSCAN_KD_DIAGNOSTICS=1`, and `--ignored`. Each fit is bracketed by stderr JSON markers
/// so a parser can attribute kd-tree counters per fit and exclude exact-neighbor counters.
fn kd_bound_diagnostics_cases(backend: BackendChoice, backend_name: &str) {
    let tracing_enabled = cfg!(feature = "tracing");
    assert!(
        tracing_enabled,
        "kd bound diagnostics require --features tracing; refusing a silent no-counter run"
    );
    assert_eq!(
        std::env::var("FAST_HDBSCAN_KD_DIAGNOSTICS").as_deref(),
        Ok("1"),
        "kd bound diagnostics require FAST_HDBSCAN_KD_DIAGNOSTICS=1"
    );
    let n = 1024usize;
    for d in [10usize, 30] {
        let input = deterministic_rows(n, d);
        for s in [16usize, 32] {
            for reproducible in [false, true] {
                let config = HdbscanConfig {
                    min_cluster_size: 5,
                    min_samples: Some(s),
                    mst_graph: MstGraph::Complete,
                    ..Default::default()
                };
                eprintln!(
                    "{{\"measurement\":\"kd_bound\",\"backend\":\"{backend_name}\",\"n\":{n},\"d\":{d},\"s\":{s},\"reproducible\":{reproducible}}}"
                );
                let fitted = complete_kd_fit(&config, &input, backend, reproducible);
                eprintln!("{{\"measurement\":\"kd_bound_end\"}}");
                assert_eq!(
                    fitted.search(),
                    HdbscanSearch::KdTree,
                    "n={n}, d={d}, s={s}, reproducible={reproducible}"
                );
                assert_eq!(
                    fitted.minimum_spanning_tree().len(),
                    n - 1,
                    "n={n}, d={d}, s={s}, reproducible={reproducible}: MST edge count"
                );
            }
        }
    }
}

#[test]
#[ignore = "observational KD bound counters; needs --features tracing and FAST_HDBSCAN_KD_DIAGNOSTICS=1"]
fn kd_bound_diagnostics_cpu() {
    kd_bound_diagnostics_cases(BackendChoice::Cpu, "cpu");
}

#[test]
#[ignore = "observational KD bound counters; needs --features tracing and FAST_HDBSCAN_KD_DIAGNOSTICS=1"]
fn kd_bound_diagnostics_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        kd_bound_diagnostics_cases(backend, "cuda");
    }
}
