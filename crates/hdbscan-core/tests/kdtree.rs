use hdbscan_core::{
    HdbscanError,
    kdtree::{KdTree, LEAF_SIZE, MAX_KD_LEVELS},
};

fn make_data(n: usize, d: usize) -> Vec<f32> {
    (0..n)
        .flat_map(|point| (0..d).map(move |feature| (point * (feature + 3) % 97) as f32))
        .collect()
}

fn assert_invariants(tree: &KdTree, data: &[f32], n: usize, d: usize) {
    assert!((1..=MAX_KD_LEVELS).contains(&tree.n_levels));
    assert_eq!(tree.n_features, d);
    assert_eq!(tree.indices.len(), n);
    assert_eq!(tree.starts.len(), tree.ends.len());
    assert_eq!(tree.starts.len(), tree.is_leaf.len());
    assert_eq!(tree.bounds_min.len(), tree.starts.len() * d);
    assert_eq!(tree.bounds_max.len(), tree.starts.len() * d);
    assert_eq!(tree.starts.len(), (1usize << tree.n_levels) - 1);
    assert_eq!(tree.starts[0], 0);
    assert_eq!(tree.ends[0] as usize, n);
    let mut permutation = tree.indices.clone();
    permutation.sort_unstable();
    assert_eq!(permutation, (0..n as i32).collect::<Vec<_>>());

    let mut seen = vec![0usize; n];
    let mut covered_positions = vec![0usize; n];
    let mut reachable = vec![false; tree.starts.len()];
    let mut stack = vec![(0usize, 1usize)];
    let mut deepest = 0;
    while let Some((node, level)) = stack.pop() {
        assert!(!reachable[node], "each node has exactly one parent");
        reachable[node] = true;
        deepest = deepest.max(level);
        assert!(level <= tree.n_levels);
        assert!(tree.starts[node] < tree.ends[node]);
        if tree.is_leaf[node] == 0 {
            let left = 2 * node + 1;
            assert!(left + 1 < tree.starts.len());
            stack.push((left + 1, level + 1));
            stack.push((left, level + 1));
            // Worst-case depth-first traversal with neither child pruned.
            assert!(stack.len() <= tree.n_levels);
            assert!(stack.len() <= MAX_KD_LEVELS);
        } else {
            assert_eq!(tree.is_leaf[node], 1);
        }
    }
    assert_eq!(deepest, tree.n_levels);
    let mut nonempty_leaves = 0;
    for (node, &node_reachable) in reachable.iter().enumerate() {
        let start = tree.starts[node] as usize;
        let end = tree.ends[node] as usize;
        if start == end {
            assert!(!node_reachable);
            assert_eq!(tree.is_leaf[node], 0);
            assert_eq!(start, 0, "unused slots have empty 0..0 ranges");
            continue;
        }
        assert!(node_reachable, "nonempty nodes must be reachable from root");
        assert!(start < end && end <= n);
        let child = 2 * node + 1;
        if child >= tree.starts.len() {
            assert_eq!(tree.is_leaf[node], 1);
        }
        if tree.is_leaf[node] == 1 {
            nonempty_leaves += 1;
            for count in &mut covered_positions[start..end] {
                *count += 1;
            }
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
            for feature in 0..d {
                let offset = node * d + feature;
                for descendant in [child, child + 1] {
                    let child_offset = descendant * d + feature;
                    assert!(tree.bounds_min[offset] <= tree.bounds_min[child_offset]);
                    assert!(tree.bounds_max[child_offset] <= tree.bounds_max[offset]);
                }
            }
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
    assert!(covered_positions.iter().all(|&count| count == 1));
    assert!(
        seen.iter().all(|&count| count == 1),
        "each point appears in one leaf"
    );
}

#[test]
fn build_handles_leaf_size_boundaries_and_heap_layout() {
    assert_eq!(LEAF_SIZE, 40);
    for n in [
        1, 2, 3, 39, 40, 41, 79, 80, 81, 159, 160, 161, 319, 320, 321,
    ] {
        let d = 3;
        let data = make_data(n, d);
        let tree = KdTree::build(&data, n, d).unwrap();
        assert_invariants(&tree, &data, n, d);
        let expected_levels = match n {
            1..=80 => 1,
            81..=160 => 2,
            161..=320 => 3,
            _ => 4,
        };
        assert_eq!(tree.n_levels, expected_levels);
        assert_eq!(tree.starts.len(), (1 << expected_levels) - 1);
    }
}

#[test]
fn build_invariants_cover_supported_feature_sizes_and_duplicates() {
    for d in [1, 2, 3, 4, 5, 8, 12, 16, 20, 24, 28, 29, 30, 32, 33, 40] {
        for n in [1, 3, 41, 81, 257, 512] {
            for duplicate_heavy in [false, true] {
                let data = if duplicate_heavy {
                    (0..n)
                        .flat_map(|point| {
                            (0..d).map(move |feature| {
                                if feature % 2 == 0 {
                                    1.0
                                } else {
                                    (point % 3) as f32
                                }
                            })
                        })
                        .collect()
                } else {
                    make_data(n, d)
                };
                let tree = KdTree::build(&data, n, d).unwrap();
                assert_invariants(&tree, &data, n, d);
                if duplicate_heavy {
                    assert_eq!(tree, KdTree::build(&data, n, d).unwrap());
                }
            }
        }
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
    assert!(matches!(
        KdTree::build(&[], usize::MAX, 2),
        Err(HdbscanError::Overflow("KD-tree input element count"))
    ));
    // This oversized shape must fail without attempting any tree allocation.
    // It exercises shape rejection, not the later row-count/level guards.
    assert!(matches!(
        KdTree::build(&[], i32::MAX as usize + 1, 1),
        Err(HdbscanError::InvalidInput(_))
    ));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            KdTree::build(&[value], 1, 1),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
}
