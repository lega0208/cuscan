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
    for (n, d, duplicate_heavy) in [
        (1, 1, false),
        (3, 2, true),
        (41, 3, false),
        (81, 16, true),
        (81, 32, false),
        (257, 33, true),
        (512, 40, false),
        (81, 40, true),
    ] {
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
        if (n, d, duplicate_heavy) == (81, 16, true) {
            let mut actual = tree.indices[..40].to_vec();
            actual.sort_unstable();
            let mut expected: Vec<i32> = (0..=78).step_by(3).chain((1..=37).step_by(3)).collect();
            expected.sort_unstable();
            assert_eq!(actual, expected);
        }
    }
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

/// The pre-parallel serial builder, kept as an exact reference for node layout, leaf point
/// order and bound bits.
fn reference_build(data: &[f32], n: usize, d: usize) -> KdTree {
    let quotient = (n.saturating_sub(1) / LEAF_SIZE).max(1);
    let n_levels = (usize::BITS - quotient.leading_zeros()) as usize;
    let n_nodes = (1usize << n_levels) - 1;
    let mut tree = KdTree {
        indices: (0..n as i32).collect(),
        starts: vec![0; n_nodes],
        ends: vec![0; n_nodes],
        is_leaf: vec![0; n_nodes],
        bounds_min: vec![0.0; n_nodes * d],
        bounds_max: vec![0.0; n_nodes * d],
        n_levels,
        n_features: d,
    };
    let feature_bounds = |indices: &[i32], feature: usize| {
        let first = data[indices[0] as usize * d + feature];
        indices[1..]
            .iter()
            .fold((first, first), |(min, max), &point| {
                let value = data[point as usize * d + feature];
                (
                    if value.total_cmp(&min).is_lt() {
                        value
                    } else {
                        min
                    },
                    if value.total_cmp(&max).is_gt() {
                        value
                    } else {
                        max
                    },
                )
            })
    };
    let mut stack = vec![(0usize, 0usize, n)];
    while let Some((node, start, end)) = stack.pop() {
        let child = 2 * node + 1;
        tree.starts[node] = start as i32;
        tree.ends[node] = end as i32;
        for feature in 0..d {
            let (min, max) = feature_bounds(&tree.indices[start..end], feature);
            tree.bounds_min[node * d + feature] = min;
            tree.bounds_max[node * d + feature] = max;
        }
        if child >= n_nodes || end - start < 2 {
            tree.is_leaf[node] = 1;
            continue;
        }
        let mut axis = 0;
        let mut best_spread = -1.0f64;
        for feature in 0..d {
            let (min, max) = feature_bounds(&tree.indices[start..end], feature);
            let spread = f64::from(max) - f64::from(min);
            if spread > best_spread {
                best_spread = spread;
                axis = feature;
            }
        }
        let mid = start + (end - start) / 2;
        tree.indices[start..end].select_nth_unstable_by(mid - start, |&left, &right| {
            data[left as usize * d + axis]
                .total_cmp(&data[right as usize * d + axis])
                .then(left.cmp(&right))
        });
        stack.push((child + 1, mid, end));
        stack.push((child, start, mid));
    }
    tree
}

/// Deterministic values with heavy duplication, signed zeros and negatives.
fn make_tricky_data(n: usize, d: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n * d)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let bits = (state >> 33) as u32;
            match bits % 8 {
                0 => 0.0,
                1 => -0.0,
                2 => (bits % 5) as f32 - 2.0,
                _ => (bits % 20_011) as f32 * 0.001 - 10.0,
            }
        })
        .collect()
}

#[test]
fn parallel_build_matches_serial_reference_bit_for_bit() {
    // 100_003 rows cross the parallel split threshold at several levels.
    for (n, d) in [
        (1, 1),
        (81, 1),
        (5_000, 2),
        (100_003, 2),
        (20_000, 16),
        (3_001, 33),
    ] {
        let data = make_tricky_data(n, d, (n * 31 + d) as u64);
        let tree = KdTree::build(&data, n, d).unwrap();
        let expected = reference_build(&data, n, d);
        assert_eq!(tree.starts, expected.starts, "n={n} d={d}");
        assert_eq!(tree.ends, expected.ends, "n={n} d={d}");
        assert_eq!(tree.is_leaf, expected.is_leaf, "n={n} d={d}");
        assert_eq!(tree.indices, expected.indices, "n={n} d={d}");
        let bits = |values: &[f32]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&tree.bounds_min),
            bits(&expected.bounds_min),
            "n={n} d={d}"
        );
        assert_eq!(
            bits(&tree.bounds_max),
            bits(&expected.bounds_max),
            "n={n} d={d}"
        );
        assert_eq!((tree.n_levels, tree.n_features), (expected.n_levels, d));
        assert_invariants(&tree, &data, n, d);
    }
}

#[test]
fn sequential_and_pool_sizes_build_identical_trees() {
    let (n, d) = (100_003, 3);
    let data = make_tricky_data(n, d, 7);
    let expected = reference_build(&data, n, d);
    let bits = |tree: &KdTree| {
        let mut bits: Vec<u32> = tree
            .bounds_min
            .iter()
            .map(|value| value.to_bits())
            .collect();
        bits.extend(tree.bounds_max.iter().map(|value| value.to_bits()));
        (
            (tree.n_levels, tree.n_features),
            tree.indices.clone(),
            tree.starts.clone(),
            tree.ends.clone(),
            tree.is_leaf.clone(),
            bits,
        )
    };
    assert_eq!(
        bits(&KdTree::build_sequential(&data, n, d).unwrap()),
        bits(&expected)
    );
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let tree = pool.install(|| KdTree::build(&data, n, d).unwrap());
        assert_eq!(bits(&tree), bits(&expected), "threads={threads}");
    }
}

#[test]
fn signed_zero_extrema_keep_their_bits_in_leaves_and_ancestors() {
    // 81 rows: a root over two leaves. Feature 0 splits; feature 1 holds only signed zeros, with
    // -0.0 only in the low half and +0.0 only in the high half.
    let n = 81;
    let data: Vec<f32> = (0..n)
        .flat_map(|row| [row as f32, if row < 40 { -0.0 } else { 0.0 }])
        .collect();
    let tree = KdTree::build(&data, n, 2).unwrap();
    assert_eq!(tree.n_levels, 2);
    let feature = |values: &[f32], node: usize| values[node * 2 + 1].to_bits();
    assert_eq!(feature(&tree.bounds_min, 1), (-0.0f32).to_bits());
    assert_eq!(feature(&tree.bounds_max, 1), (-0.0f32).to_bits());
    assert_eq!(feature(&tree.bounds_min, 2), 0.0f32.to_bits());
    assert_eq!(feature(&tree.bounds_max, 2), 0.0f32.to_bits());
    assert_eq!(feature(&tree.bounds_min, 0), (-0.0f32).to_bits());
    assert_eq!(feature(&tree.bounds_max, 0), 0.0f32.to_bits());
}
