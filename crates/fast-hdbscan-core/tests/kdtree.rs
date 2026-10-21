use fast_hdbscan_core::{
    HdbscanError,
    kdtree::{KdTree, LEAF_SIZE, MAX_KD_LEVELS},
};

fn make_data(n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|point| (0..d).map(move |feature| (point * (feature + 3) % 97) as f32))
        .collect()
}

fn assert_invariants(tree: &KdTree, data: &[f32], n: usize, d: usize) {
    assert!(tree.n_levels <= MAX_KD_LEVELS);
    assert_eq!(tree.n_features, d);
    assert_eq!(tree.indices.len(), n);
    assert_eq!(tree.starts.len(), tree.ends.len());
    assert_eq!(tree.starts.len(), tree.is_leaf.len());
    assert_eq!(tree.bounds_min.len(), tree.starts.len() * d);
    assert_eq!(tree.bounds_max.len(), tree.starts.len() * d);

    let mut seen = vec![0usize; n];
    let mut nonempty_leaves = 0;
    for node in 0..tree.starts.len() {
        let start = tree.starts[node] as usize;
        let end = tree.ends[node] as usize;
        if start == end {
            assert_eq!(tree.is_leaf[node], 0);
            assert_eq!(start, 0, "unused slots have empty 0..0 ranges");
            continue;
        }
        assert!(start < end && end <= n);
        let child = 2 * node + 1;
        if child >= tree.starts.len() {
            assert_eq!(tree.is_leaf[node], 1);
        }
        if tree.is_leaf[node] == 1 {
            nonempty_leaves += 1;
            for &point in &tree.indices[start..end] {
                seen[point as usize] += 1;
            }
            for feature in 0..d {
                let values = tree.indices[start..end]
                    .iter()
                    .map(|&point| data[point as usize * d + feature]);
                let expected_min = values.clone().fold(f32::INFINITY, f32::min);
                let expected_max = values.fold(f32::NEG_INFINITY, f32::max);
                let offset = node * d + feature;
                assert_eq!(tree.bounds_min[offset], expected_min);
                assert_eq!(tree.bounds_max[offset], expected_max);
            }
        } else {
            assert_eq!(tree.is_leaf[node], 0);
            assert!(child + 1 < tree.starts.len());
            assert_eq!(tree.starts[child], tree.starts[node]);
            assert_eq!(tree.ends[child] as usize, start + (end - start) / 2);
            assert_eq!(tree.starts[child + 1] as usize, start + (end - start) / 2);
            assert_eq!(tree.ends[child + 1], tree.ends[node]);
        }
        for feature in 0..d {
            let offset = node * d + feature;
            for &point in &tree.indices[start..end] {
                let value = data[point as usize * d + feature];
                assert!(tree.bounds_min[offset] <= value);
                assert!(value <= tree.bounds_max[offset]);
            }
        }
    }
    assert!(nonempty_leaves > 0);
    assert!(
        seen.iter().all(|&count| count == 1),
        "each point appears in one leaf"
    );
}

#[test]
fn build_handles_leaf_size_boundaries_and_heap_layout() {
    assert_eq!(LEAF_SIZE, 40);
    for n in [1, 39, 40, 41, 79, 80, 81] {
        let d = 3;
        let data = make_data(n, d);
        let tree = KdTree::build(&data, n, d).unwrap();
        assert_invariants(&tree, &data, n, d);
        assert_eq!(tree.n_levels, if n <= 80 { 1 } else { 2 });
        assert_eq!(tree.starts.len(), if n <= 80 { 1 } else { 3 });
    }
}

#[test]
fn build_is_deterministic_with_duplicate_and_equal_feature_values() {
    let data = (0..81)
        .flat_map(|point| [1.0, (point / 2) as f32])
        .collect::<Vec<_>>();
    let first = KdTree::build(&data, 81, 2).unwrap();
    let second = KdTree::build(&data, 81, 2).unwrap();
    assert_eq!(first, second);
    assert_invariants(&first, &data, 81, 2);
}

#[test]
fn equal_spreads_choose_the_lower_feature_axis() {
    let data = (0..81)
        .flat_map(|point| [point as f32, (80 - point) as f32])
        .collect::<Vec<_>>();
    let tree = KdTree::build(&data, 81, 2).unwrap();
    assert!(tree.indices[..40].iter().all(|&point| point < 40));
    assert_invariants(&tree, &data, 81, 2);
}

#[test]
fn node_components_are_bottom_up_and_noise_is_not_homogeneous() {
    let data = make_data(81, 2);
    let tree = KdTree::build(&data, 81, 2).unwrap();
    let mut labels = vec![7; 81];
    assert_eq!(
        tree.node_components(&labels).unwrap(),
        vec![7; tree.starts.len()]
    );
    labels[0] = 8;
    let components = tree.node_components(&labels).unwrap();
    assert_eq!(components[0], -1);
    labels.fill(-1);
    assert!(
        tree.node_components(&labels)
            .unwrap()
            .iter()
            .all(|&label| label == -1)
    );
    assert!(matches!(
        tree.node_components(&labels[..80]),
        Err(HdbscanError::InvalidInput(_))
    ));
}

#[test]
fn build_rejects_invalid_dimensions_and_non_finite_values() {
    for (data, n, d) in [(&[][..], 0, 1), (&[1.0][..], 1, 0), (&[1.0][..], 2, 1)] {
        assert!(matches!(
            KdTree::build(data, n, d),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            KdTree::build(&[value], 1, 1),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
}
