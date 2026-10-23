#[path = "../../fast-hdbscan-core/tests/support/mod.rs"]
mod support;

use fast_hdbscan::{BackendChoice, ClusterSelectionMethod, HDBSCAN, HdbscanConfig, HdbscanSearch};

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

#[cfg(feature = "cpu")]
#[test]
fn estimator_matches_pinned_fixtures_on_cpu() {
    fixture_suite(BackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn estimator_matches_pinned_fixtures_on_cuda() {
    let resolved = fast_cubecl_common::selected_device(
        BackendChoice::Cuda { device_index: 0 },
        cfg!(feature = "cpu"),
        true,
    );
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

#[cfg(not(feature = "cuda"))]
#[test]
fn required_cuda_estimator_fails_when_cuda_is_not_compiled() {
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
    );
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
    let backend = if cfg!(feature = "cpu") {
        BackendChoice::Cpu
    } else {
        BackendChoice::Cuda { device_index: 0 }
    };
    let valid = HDBSCAN::new(HdbscanConfig::default(), backend);
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

#[cfg(feature = "cpu")]
#[test]
fn estimator_min_samples_boundaries_match_independent_neighbor_oracle_on_cpu() {
    min_samples_boundary_cases(BackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn estimator_min_samples_boundaries_match_independent_neighbor_oracle_on_cuda() {
    let Ok(_) = fast_cubecl_common::selected_device(
        BackendChoice::Cuda { device_index: 0 },
        cfg!(feature = "cpu"),
        true,
    ) else {
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

#[cfg(feature = "cpu")]
#[test]
fn estimator_handles_finite_row_remapping_and_small_finite_sets_on_cpu() {
    finite_remapping_cases(BackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn estimator_handles_finite_row_remapping_and_small_finite_sets_on_cuda() {
    let Ok(_) = fast_cubecl_common::selected_device(
        BackendChoice::Cuda { device_index: 0 },
        cfg!(feature = "cpu"),
        true,
    ) else {
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

#[cfg(feature = "cpu")]
#[test]
fn forced_search_routes_match_on_cpu() {
    route_parity_cases(BackendChoice::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn forced_search_routes_match_on_cuda() {
    if fast_cubecl_common::selected_device(
        BackendChoice::Cuda { device_index: 0 },
        cfg!(feature = "cpu"),
        true,
    )
    .is_err()
    {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    route_parity_cases(BackendChoice::Cuda { device_index: 0 });
}

#[cfg(not(feature = "cpu"))]
#[test]
fn explicit_cpu_backend_fails_when_cpu_is_not_compiled() {
    assert!(
        HDBSCAN::new(HdbscanConfig::default(), BackendChoice::Cpu)
            .fit(&[vec![0.0]])
            .is_err()
    );
}
