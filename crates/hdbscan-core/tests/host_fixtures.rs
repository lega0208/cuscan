mod support;

use hdbscan_core::{
    ClusterSelectionMethod, HdbscanConfig, SortedMstEdge, clusters_from_spanning_tree,
};
use serde_json::Value;
use std::path::Path;
use support::{NpyHeader, fixture_root, load_npz, read_f32s, read_i64s, read_json};

fn f64_values(header: &NpyHeader) -> Vec<f64> {
    assert_eq!(header.descr, "<f8");
    assert!(header.payload.len().is_multiple_of(8));
    header
        .payload
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&b| f64::from_le_bytes(b))
        .collect()
}

fn f32_values(header: &NpyHeader) -> Vec<f32> {
    read_f32s(header).unwrap()
}

fn i64_values(header: &NpyHeader) -> Vec<i64> {
    read_i64s(header).unwrap()
}

fn condensed_rows(header: &NpyHeader) -> hdbscan_core::CondensedTree {
    let row_width = header.payload.len() / header.shape[0].max(1);
    assert!(row_width == 28 || row_width == 32);
    let mut tree = hdbscan_core::CondensedTree::default();
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

fn parameter_usize(params: &Value, key: &str) -> Option<usize> {
    params.get(key).and_then(Value::as_u64).map(|v| v as usize)
}

fn config(params: &Value) -> HdbscanConfig {
    HdbscanConfig {
        min_cluster_size: parameter_usize(params, "min_cluster_size").unwrap_or(5),
        min_samples: parameter_usize(params, "min_samples"),
        cluster_selection_method: match params["selection"].as_str().unwrap_or("eom") {
            "leaf" => ClusterSelectionMethod::Leaf,
            "eom" => ClusterSelectionMethod::Eom,
            other => panic!("unknown fixture selection method {other}"),
        },
        allow_single_cluster: params["allow_single_cluster"].as_bool().unwrap_or(false),
        max_cluster_size: parameter_usize(params, "max_cluster_size"),
        cluster_selection_epsilon: params["epsilon"].as_f64().unwrap_or(0.0),
        cluster_selection_persistence: params["persistence"].as_f64().unwrap_or(0.0),
        ..Default::default()
    }
}

fn matrix(
    path: &Path,
    arrays: &std::collections::BTreeMap<String, NpyHeader>,
    name: &str,
) -> Vec<f64> {
    let header = arrays
        .get(name)
        .unwrap_or_else(|| panic!("{} missing fixture array {name}", path.display()));
    f64_values(header)
}

fn assert_f32_bits(actual: f32, expected: f32, context: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{context}: actual={actual:?}, expected={expected:?}"
    );
}

#[cfg(feature = "test-support")]
fn adjusted_rand_index(left: &[i32], right: &[i64]) -> f64 {
    assert_eq!(left.len(), right.len());
    let mut left_counts = std::collections::HashMap::<i32, usize>::new();
    let mut right_counts = std::collections::HashMap::<i64, usize>::new();
    let mut pair_counts = std::collections::HashMap::<(i32, i64), usize>::new();
    for (&a, &b) in left.iter().zip(right) {
        *left_counts.entry(a).or_default() += 1;
        *right_counts.entry(b).or_default() += 1;
        *pair_counts.entry((a, b)).or_default() += 1;
    }
    let choose2 = |n: usize| (n * n.saturating_sub(1)) as f64 / 2.0;
    let pairs = pair_counts.values().map(|&n| choose2(n)).sum::<f64>();
    let left_pairs = left_counts.values().map(|&n| choose2(n)).sum::<f64>();
    let right_pairs = right_counts.values().map(|&n| choose2(n)).sum::<f64>();
    let total = choose2(left.len());
    if total == 0.0 {
        return 1.0;
    }
    let expected = left_pairs * right_pairs / total;
    let maximum = (left_pairs + right_pairs) / 2.0;
    if maximum == expected {
        1.0
    } else {
        (pairs - expected) / (maximum - expected)
    }
}

#[test]
fn reference_fixtures_match_host_phase_one_outputs() {
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let cases = manifest["cases"].as_array().expect("manifest cases");
    let mut compared = 0usize;
    let mut skipped = 0usize;
    let mut outlier_remaps_compared = 0usize;
    for case in cases {
        let id = case["id"].as_str().expect("case id");
        let meta = read_json(&root.join(format!("{id}.json")));
        match meta["status"].as_str().unwrap_or("") {
            "validation_failure" | "upstream_failure" => {
                skipped += 1;
                continue;
            }
            "reference" | "port_contract" => {}
            status => panic!("{id}: unrecognized fixture status {status:?}"),
        }
        let arrays =
            load_npz(&root.join(format!("{id}.npz"))).unwrap_or_else(|e| panic!("{id}: {e}"));
        let edges = matrix(&root, &arrays, "mst")
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| SortedMstEdge {
                src: r[0] as u32,
                dst: r[1] as u32,
                weight: r[2],
            })
            .collect::<Vec<_>>();
        if edges.is_empty() {
            skipped += 1;
            continue;
        }
        let actual = clusters_from_spanning_tree(&edges, &config(&meta["parameters"]))
            .unwrap_or_else(|e| panic!("{id}: host extraction failed: {e}"));
        let linkage = matrix(&root, &arrays, "linkage");
        let actual_linkage = actual
            .linkage
            .iter()
            .flat_map(|r| [r.left as f64, r.right as f64, r.distance, r.size as f64])
            .collect::<Vec<_>>();
        assert_eq!(actual_linkage, linkage, "{id}: linkage is exact (f64)");
        let finite_index = i64_values(&arrays["finite_index"])
            .into_iter()
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        let raw_n = meta["input_shape"][0].as_u64().unwrap() as usize;
        let has_outliers = finite_index.len() < raw_n;
        if has_outliers && meta["status"] == "reference" {
            for name in [
                "outlier_order",
                "remapped_linkage",
                "remapped_condensed_tree",
            ] {
                assert!(
                    arrays.contains_key(name),
                    "{id}: missing required outlier remap oracle {name}"
                );
            }
            outlier_remaps_compared += 1;
        }
        let raw_labels = hdbscan_core::finite::scatter_labels(&actual.labels, &finite_index, raw_n)
            .unwrap_or_else(|e| panic!("{id}: label remap failed: {e}"));
        assert_eq!(
            raw_labels.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            i64_values(&arrays["labels"]),
            "{id}: labels are exact"
        );
        let expected_probabilities = f32_values(&arrays["probabilities"]);
        let raw_probabilities = hdbscan_core::finite::scatter_probabilities(
            &actual.probabilities,
            &finite_index,
            raw_n,
        )
        .unwrap_or_else(|e| panic!("{id}: probability remap failed: {e}"));
        assert_eq!(
            raw_probabilities.len(),
            expected_probabilities.len(),
            "{id}: probability shape"
        );
        for (i, (&got, &want)) in raw_probabilities
            .iter()
            .zip(&expected_probabilities)
            .enumerate()
        {
            assert_f32_bits(got, want, &format!("{id}: probability {i}"));
        }
        if let Some(expected) = arrays.get("condensed_tree") {
            assert_eq!(
                actual.condensed.len(),
                expected.shape[0],
                "{id}: condensed row count"
            );
            // Structured records are packed parent:i64, child:i64, lambda:f64, size:f32.
            assert_eq!(
                expected.payload.len(),
                expected.shape[0] * 28,
                "{id}: condensed payload"
            );
            for (i, row) in expected.payload.as_chunks::<28>().0.iter().enumerate() {
                assert_eq!(
                    actual.condensed.parent[i] as i64,
                    i64::from_le_bytes(row[0..8].try_into().unwrap()),
                    "{id}: condensed parent {i}"
                );
                assert_eq!(
                    actual.condensed.child[i] as i64,
                    i64::from_le_bytes(row[8..16].try_into().unwrap()),
                    "{id}: condensed child {i}"
                );
                let lambda = f64::from_le_bytes(row[16..24].try_into().unwrap());
                assert_eq!(
                    actual.condensed.lambda[i].to_bits(),
                    (lambda as f32).to_bits(),
                    "{id}: condensed lambda {i}"
                );
                let size = f32::from_le_bytes(row[24..28].try_into().unwrap());
                assert_eq!(
                    actual.condensed.child_size[i].to_bits(),
                    size.to_bits(),
                    "{id}: condensed size {i}"
                );
            }
        }
        if let Some(order) = arrays.get("outlier_order") {
            let outlier_order = i64_values(order)
                .into_iter()
                .map(|x| x as u32)
                .collect::<Vec<_>>();
            let remapped_linkage =
                hdbscan_core::finite::remap_linkage(&actual.linkage, &finite_index, &outlier_order)
                    .unwrap_or_else(|e| panic!("{id}: linkage remap failed: {e}"));
            if let Some(expected) = arrays.get("remapped_linkage") {
                let got = remapped_linkage
                    .iter()
                    .flat_map(|r| [r.left as f64, r.right as f64, r.distance, r.size as f64])
                    .collect::<Vec<_>>();
                assert_eq!(
                    got,
                    f64_values(expected),
                    "{id}: remapped linkage is exact and follows fixture outlier order"
                );
            }
            let remapped_condensed = hdbscan_core::finite::remap_condensed(
                &actual.condensed,
                &finite_index,
                &outlier_order,
            )
            .unwrap_or_else(|e| panic!("{id}: condensed remap failed: {e}"));
            if let Some(expected) = arrays.get("remapped_condensed_tree") {
                assert_eq!(
                    remapped_condensed.len(),
                    expected.shape[0],
                    "{id}: remapped condensed row count"
                );
                let row_width = expected.payload.len() / expected.shape[0].max(1);
                assert!(
                    row_width == 28 || row_width == 32,
                    "{id}: unexpected remapped condensed record width {row_width}"
                );
                assert_eq!(
                    expected.payload.len(),
                    expected.shape[0] * row_width,
                    "{id}: remapped condensed payload"
                );
                for (i, row) in expected.payload.chunks_exact(row_width).enumerate() {
                    assert_eq!(
                        remapped_condensed.parent[i] as i64,
                        i64::from_le_bytes(row[0..8].try_into().unwrap()),
                        "{id}: remapped condensed parent {i}"
                    );
                    assert_eq!(
                        remapped_condensed.child[i] as i64,
                        i64::from_le_bytes(row[8..16].try_into().unwrap()),
                        "{id}: remapped condensed child {i}"
                    );
                    assert_eq!(
                        remapped_condensed.lambda[i].to_bits(),
                        (f64::from_le_bytes(row[16..24].try_into().unwrap()) as f32).to_bits(),
                        "{id}: remapped condensed lambda {i}"
                    );
                    let size = if row_width == 28 {
                        f32::from_le_bytes(row[24..28].try_into().unwrap())
                    } else {
                        f64::from_le_bytes(row[24..32].try_into().unwrap()) as f32
                    };
                    assert_eq!(
                        remapped_condensed.child_size[i].to_bits(),
                        size.to_bits(),
                        "{id}: remapped condensed size {i}"
                    );
                }
            }
        }
        compared += 1;
    }
    assert!(compared > 0, "no reference fixtures were exercised");
    assert!(
        outlier_remaps_compared > 0,
        "no reference outlier remaps were compared"
    );
    assert!(
        skipped > 0,
        "fixture statuses require explicit skipped-case accounting"
    );
}

#[test]
fn fixture_dbscan_labels_match_cut_contract() {
    use hdbscan_core::labels::dbscan_cut;
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let mut compared = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let meta = read_json(&root.join(format!("{id}.json")));
        if !matches!(meta["status"].as_str(), Some("reference" | "port_contract")) {
            continue;
        }
        let arrays = load_npz(&root.join(format!("{id}.npz"))).unwrap();
        let Some(expected) = arrays.get("dbscan_labels") else {
            continue;
        };
        let mst = matrix(&root, &arrays, "mst");
        if mst.is_empty() {
            continue;
        }
        let edges = mst
            .as_chunks::<3>()
            .0
            .iter()
            .map(|r| SortedMstEdge {
                src: r[0] as u32,
                dst: r[1] as u32,
                weight: r[2],
            })
            .collect::<Vec<_>>();
        let extraction = clusters_from_spanning_tree(&edges, &config(&meta["parameters"])).unwrap();
        let params = &meta["parameters"];
        let outliers = arrays
            .get("outlier_order")
            .map(i64_values)
            .unwrap_or_default()
            .into_iter()
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        let finite_index = i64_values(&arrays["finite_index"])
            .into_iter()
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        let linkage =
            hdbscan_core::finite::remap_linkage(&extraction.linkage, &finite_index, &outliers)
                .unwrap();
        let labels = dbscan_cut(
            &linkage,
            params["epsilon"].as_f64().unwrap_or(0.0),
            parameter_usize(params, "min_samples").unwrap_or(5),
        )
        .unwrap();
        assert_eq!(
            labels.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            i64_values(expected),
            "{id}"
        );
        if let Some(expected_remapped) = arrays.get("remapped_dbscan_labels") {
            assert_eq!(
                labels.iter().map(|&x| x as i64).collect::<Vec<_>>(),
                i64_values(expected_remapped),
                "{id}: remapped DBSCAN labels follow fixture outlier order"
            );
        }
        compared += 1;
    }
    assert!(compared > 0, "no fixture DBSCAN arrays were compared");
}

#[test]
fn one_finite_host_degenerate_contract_has_no_tree_or_labels() {
    let result = hdbscan_core::HostClustering::one_finite();
    assert_eq!(result.labels, [-1]);
    assert_eq!(result.probabilities, [0.0]);
    assert!(result.linkage.is_empty());
    assert!(result.condensed.is_empty());
    assert!(result.sorted_mst.is_empty());
}

#[test]
fn persistence_fixture_below_equal_and_above_boundaries() {
    let root = fixture_root();
    let meta = read_json(&root.join("persistence_equal.json"));
    let arrays = load_npz(&root.join("persistence_equal.npz")).unwrap();
    let input = condensed_rows(&arrays["boundary_input"]);
    let threshold = meta["persistence_boundary_evidence"]["threshold"]
        .as_f64()
        .unwrap() as f32;
    let below = f32::from_bits(threshold.to_bits() - 1);
    let above = f32::from_bits(threshold.to_bits() + 1);
    for (name, level) in [
        ("boundary_below", below),
        ("boundary_equal", threshold),
        ("boundary_above", above),
    ] {
        let actual = hdbscan_core::condense::simplify_hierarchy(input.clone(), level as f64);
        let expected = condensed_rows(&arrays[name]);
        assert_eq!(actual, expected, "persistence {name} boundary");
    }
}

#[cfg(feature = "test-support")]
#[test]
fn oracle_fixture_core_distances_and_mst_weights_match() {
    use hdbscan_core::oracle::{brute_force_mst, exact_neighbors, knn_kruskal_oracle};
    let root = fixture_root();
    let manifest = read_json(&root.join("manifest.json"));
    let mut compared = 0;
    for case in manifest["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let meta = read_json(&root.join(format!("{id}.json")));
        if !matches!(meta["status"].as_str(), Some("reference" | "port_contract")) {
            continue;
        }
        let arrays = load_npz(&root.join(format!("{id}.npz"))).unwrap();
        let Some(core_header) = arrays.get("core_distances") else {
            continue;
        };
        let input_header = &arrays["input"];
        let raw = f32_values(input_header);
        let n_raw = input_header.shape[0];
        let d = input_header.shape[1];
        let finite = hdbscan_core::finite::split_finite(&raw, d).unwrap();
        let compact = hdbscan_core::finite::compact_rows(&raw, d, &finite.finite_index).unwrap();
        let n = finite.finite_index.len();
        if n == 0 {
            panic!("{id}: applicable reference fixture has no finite rows");
        }
        let ms = parameter_usize(&meta["parameters"], "min_samples").unwrap_or(5);
        let graph_route = match meta["parameters"]["mst_graph"].as_str() {
            None => false,
            Some("knn_graph") => true,
            Some(other) => panic!("{id}: unknown fixture mst_graph {other}"),
        };
        // Graph fixtures use the independent kNN-Kruskal reference, whose graph width is
        // upstream's min(max(knn_k, min_samples), n - 1). Its bridges are infinite edges.
        let (actual_cores, actual, graph_bridges) = if graph_route {
            let knn_k = parameter_usize(&meta["parameters"], "knn_k")
                .unwrap_or_else(|| panic!("{id}: knn_graph fixture missing knn_k"));
            let k = knn_k.max(ms).min(n.saturating_sub(1));
            let oracle = knn_kruskal_oracle(&compact, n, d, ms, k).unwrap();
            (oracle.core, oracle.edges, Some(oracle.bridges))
        } else {
            let knn = exact_neighbors(&compact, n, d, ms).unwrap();
            let cores = knn
                .core_sq
                .iter()
                .map(|&sq| sq.sqrt() as f64)
                .collect::<Vec<f64>>();
            (cores, brute_force_mst(&compact, n, d, ms).unwrap(), None)
        };
        let expected_cores = f32_values(core_header)
            .into_iter()
            .map(f64::from)
            .collect::<Vec<f64>>();
        assert_eq!(
            actual_cores.len(),
            expected_cores.len(),
            "{id}: core-distance length"
        );
        for (i, (&got, &want)) in actual_cores.iter().zip(&expected_cores).enumerate() {
            assert!(
                got == want
                    || (got.is_finite()
                        && want.is_finite()
                        && want != 0.0
                        && (got - want).abs() <= 1.0e-5 * want.abs()),
                "{id}: core distance {i}: {got:?} != {want:?}"
            );
        }
        let expected = matrix(&root, &arrays, "mst_weights");
        assert_eq!(actual.len(), expected.len(), "{id}: oracle MST edge count");
        let mut weights = actual.iter().map(|edge| edge.weight).collect::<Vec<_>>();
        weights.sort_by(f64::total_cmp);
        let mut expected = expected;
        expected.sort_by(f64::total_cmp);
        for (i, (&got, &want)) in weights.iter().zip(&expected).enumerate() {
            assert!(
                got == want
                    || (got.is_finite()
                        && want.is_finite()
                        && want != 0.0
                        && (got - want).abs() <= 1.0e-5 * want.abs()),
                "{id}: MST weight {i}: {got:?} != {want:?}"
            );
        }
        if let Some(bridges) = graph_bridges {
            let infinite = weights.iter().filter(|w| w.is_infinite()).count();
            assert_eq!(infinite, bridges, "{id}: graph infinite bridge count");
        }
        compared += 1;
        let oracle_host = clusters_from_spanning_tree(&actual, &config(&meta["parameters"]))
            .unwrap_or_else(|e| panic!("{id}: oracle MST phase-1 pipeline: {e}"));
        assert_eq!(
            oracle_host.linkage.len(),
            n - 1,
            "{id}: oracle linkage must be a spanning tree"
        );
        let mut node_sizes = vec![1usize; 2 * n - 1];
        for (row, merge) in oracle_host.linkage.iter().enumerate() {
            let parent = n + row;
            assert!(
                merge.left < parent && merge.right < parent,
                "{id}: linkage row {row} references a non-prior node"
            );
            assert_ne!(
                merge.left, merge.right,
                "{id}: linkage row {row} joins a node to itself"
            );
            let expected_size = node_sizes[merge.left] + node_sizes[merge.right];
            assert_eq!(
                merge.size, expected_size,
                "{id}: oracle linkage component size at merge {row}"
            );
            node_sizes[parent] = expected_size;
            if row > 0 {
                assert!(
                    oracle_host.linkage[row - 1].distance <= merge.distance,
                    "{id}: oracle linkage distances are nondecreasing"
                );
            }
        }
        let finite_index = i64_values(&arrays["finite_index"])
            .into_iter()
            .map(|x| x as u32)
            .collect::<Vec<_>>();
        let labels =
            hdbscan_core::finite::scatter_labels(&oracle_host.labels, &finite_index, n_raw)
                .unwrap();
        let expected_labels = i64_values(&arrays["labels"]);
        let ari = adjusted_rand_index(&labels, &expected_labels);
        let has_tie = meta["tie_classification"]
            .as_array()
            .is_some_and(|entries| !entries.is_empty());
        if has_tie {
            eprintln!("{id}: annotated tie case oracle label ARI={ari:.6}");
        } else {
            assert_eq!(
                ari, 1.0,
                "{id}: labels must be permutation-equivalent away from ties"
            );
        }
        let oracle_probabilities = hdbscan_core::finite::scatter_probabilities(
            &oracle_host.probabilities,
            &finite_index,
            n_raw,
        )
        .unwrap();
        let expected_probabilities = f32_values(&arrays["probabilities"]);
        assert_eq!(
            oracle_probabilities.len(),
            expected_probabilities.len(),
            "{id}: oracle probability shape"
        );
        for (point, (&got, &want)) in oracle_probabilities
            .iter()
            .zip(&expected_probabilities)
            .enumerate()
        {
            assert!(
                got == want
                    || (got.is_finite() && want.is_finite() && (got - want).abs() <= 1.0e-5),
                "{id}: oracle probability {point} differs beyond 1e-5: {got:?} != {want:?}"
            );
        }
        assert_eq!(n_raw, input_header.shape[0]);
    }
    assert!(
        compared > 0,
        "no fixture core/MST oracle cases were exercised"
    );
}

#[test]
fn dbscan_epsilon_sweep_covers_boundaries_and_nan() {
    use hdbscan_core::{LinkageRow, labels::dbscan_cut};
    let tree = [
        LinkageRow {
            left: 1,
            right: 0,
            distance: 1.0,
            size: 2,
        },
        LinkageRow {
            left: 3,
            right: 2,
            distance: 2.0,
            size: 2,
        },
        LinkageRow {
            left: 5,
            right: 4,
            distance: 3.0,
            size: 4,
        },
    ];
    for cut in [-1.0, 0.0, 1.0] {
        assert_eq!(
            dbscan_cut(&tree, cut, 2).unwrap(),
            [-1, -1, -1, -1],
            "cut={cut}"
        );
    }
    assert_eq!(dbscan_cut(&tree, 1.5, 2).unwrap(), [0, 0, -1, -1]);
    assert_eq!(dbscan_cut(&tree, 2.0, 2).unwrap(), [0, 0, -1, -1]);
    assert_eq!(dbscan_cut(&tree, 2.5, 2).unwrap(), [0, 0, 1, 1]);
    assert_eq!(dbscan_cut(&tree, f64::INFINITY, 2).unwrap(), [0, 0, 0, 0]);
    assert!(dbscan_cut(&tree, f64::NAN, 2).is_err());
}

#[test]
fn ascending_outlier_order_is_deterministic_for_remaps() {
    use hdbscan_core::finite::{remap_condensed, remap_linkage, split_finite};
    use hdbscan_core::{CondensedTree, LinkageRow};
    let raw = [0.0f32, f32::NAN, 1.0, f32::INFINITY, 2.0];
    let split = split_finite(&raw, 1).unwrap();
    assert_eq!(split.finite_index, [0, 2, 4]);
    assert_eq!(split.outliers, [1, 3]);
    let linkage = [
        LinkageRow {
            left: 2,
            right: 1,
            distance: 1.0,
            size: 2,
        },
        LinkageRow {
            left: 3,
            right: 0,
            distance: 2.0,
            size: 3,
        },
    ];
    let remapped = remap_linkage(&linkage, &split.finite_index, &split.outliers).unwrap();
    assert_eq!(
        (remapped[2].left, remapped[2].right, remapped[2].distance),
        (1, 6, f64::INFINITY)
    );
    assert_eq!(
        (remapped[3].left, remapped[3].right, remapped[3].distance),
        (3, 7, f64::INFINITY)
    );
    let tree = CondensedTree {
        parent: vec![3],
        child: vec![0],
        lambda: vec![1.0],
        child_size: vec![3.0],
    };
    let remapped = remap_condensed(&tree, &split.finite_index, &split.outliers).unwrap();
    assert_eq!(&remapped.child[..2], &[1, 3]);
}
