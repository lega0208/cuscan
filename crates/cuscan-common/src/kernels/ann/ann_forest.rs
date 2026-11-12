//! Deterministic RP-forest leaf membership and bounded ANN workspace planning.
//!
//! The forest stores only contiguous row membership and leaf/tree offsets. It
//! deliberately does not materialize any within-leaf pairs.

use rayon::prelude::*;

pub(super) const MAX_TREES: usize = 64;
pub(super) const MAX_CANDIDATES: usize = 60;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ForestLeaves {
    /// Row ids grouped contiguously by leaf across all trees.
    pub(super) row_ids: Vec<usize>,
    /// `leaf_offsets[leaf..leaf + 1]` bounds that leaf in `row_ids`.
    pub(super) leaf_offsets: Vec<usize>,
    /// `tree_leaf_offsets[tree..tree + 1]` bounds a tree's leaves.
    pub(super) tree_leaf_offsets: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WorkspacePlan {
    pub(super) max_candidates_per_class: usize,
    pub(super) pair_slots_per_center: usize,
    /// Peak host-only phase: feature copy plus forest and recursion scratch.
    pub(super) host_peak_bytes: usize,
    /// Leaf size limit used by the forest builder (`min(max(10, 2H), n)`).
    pub(super) forest_leaf_limit: usize,
    /// Total membership entries across all trees (`trees * n`).
    pub(super) forest_membership_len: usize,
    /// Total leaves across all trees.
    pub(super) forest_leaf_count: usize,
    /// Total within-leaf pairs across all trees (the forest pair prefix).
    pub(super) forest_pair_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkspaceError {
    InvalidDimensions,
    ArithmeticOverflow,
    /// A valid shape exceeds a documented implementation limit.
    Capacity {
        quantity: &'static str,
        value: usize,
        limit: usize,
    },
}

/// Plan `n_trees = min(64, 5 + round(sqrt(n)/20))`.
pub(super) fn tree_count(n: usize) -> Result<usize, WorkspaceError> {
    if n == 0 {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let extra = ((n as f64).sqrt() / 20.0).round();
    let extra = usize::try_from(extra as u64).map_err(|_| WorkspaceError::ArithmeticOverflow)?;
    Ok(5usize.saturating_add(extra).min(MAX_TREES))
}

/// Build a deterministic forest. Each node uses a seed derived from its tree
/// and path; equal projections are split by a seeded row priority, ensuring
/// balanced progress even for duplicate points or a zero normal. Trees are
/// built in parallel and collected in indexed tree order before concatenation.
pub(super) fn build_forest(
    data: &[f32],
    n: usize,
    features: usize,
    seed: u64,
    tree_count: usize,
    leaf_size: usize,
) -> Result<ForestLeaves, WorkspaceError> {
    if n == 0 || features == 0 || leaf_size == 0 || tree_count == 0 || tree_count > MAX_TREES {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let expected = n
        .checked_mul(features)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    if data.len() != expected {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let membership_len = n
        .checked_mul(tree_count)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let per_tree = (0..tree_count)
        .into_par_iter()
        .map(|tree| {
            let mut result = ForestLeaves {
                row_ids: Vec::with_capacity(n),
                leaf_offsets: vec![0],
                tree_leaf_offsets: Vec::new(),
            };
            let mut rows = (0..n).collect::<Vec<_>>();
            split_node(
                data,
                features,
                &mut rows,
                leaf_size.min(n),
                seed_for(seed, tree as u64, 1),
                &mut result,
            );
            debug_assert_eq!(result.row_ids.len(), n);
            result
        })
        .collect::<Vec<_>>();
    let mut result = ForestLeaves {
        row_ids: Vec::with_capacity(membership_len),
        leaf_offsets: Vec::new(),
        tree_leaf_offsets: Vec::with_capacity(tree_count + 1),
    };
    result.leaf_offsets.push(0);
    for tree in per_tree {
        result.tree_leaf_offsets.push(result.leaf_offsets.len() - 1);
        let row_offset = result.row_ids.len();
        result.row_ids.extend(tree.row_ids);
        result.leaf_offsets.extend(
            tree.leaf_offsets
                .into_iter()
                .skip(1)
                .map(|offset| row_offset + offset),
        );
    }
    result.tree_leaf_offsets.push(result.leaf_offsets.len() - 1);
    Ok(result)
}

fn split_node(
    data: &[f32],
    features: usize,
    rows: &mut [usize],
    leaf_size: usize,
    node_seed: u64,
    out: &mut ForestLeaves,
) {
    if rows.len() <= leaf_size {
        out.row_ids.extend_from_slice(rows);
        out.leaf_offsets.push(out.row_ids.len());
        return;
    }

    let mut state = node_seed;
    let normal = (0..features)
        .map(|_| next_unit(&mut state) * 2.0 - 1.0)
        .collect::<Vec<_>>();
    let mut projected = Vec::with_capacity(rows.len());
    for &row in rows.iter() {
        let base = row * features;
        let projection = (0..features)
            .map(|feature| f64::from(data[base + feature]) * f64::from(normal[feature]))
            .sum::<f64>();
        projected.push((row, projection, mix64(node_seed ^ row as u64)));
    }
    let compare = |a: &(usize, f64, u64), b: &(usize, f64, u64)| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    };
    let midpoint = projected.len() / 2;
    projected.select_nth_unstable_by(midpoint, compare);
    // Each side is ordered only enough to partition by projection; equal
    // projections split by deterministic node/row priority at the midpoint.
    for (row, item) in rows.iter_mut().zip(projected.iter()) {
        *row = item.0;
    }
    drop(projected);
    drop(normal);
    let path_left = seed_for(node_seed, 0, 0);
    let path_right = seed_for(node_seed, 1, 0);
    let (left, right) = rows.split_at_mut(midpoint);
    split_node(data, features, left, leaf_size, path_left, out);
    split_node(data, features, right, leaf_size, path_right, out);
}

// Phase 0 prototype only. Keep the production builder above unchanged so it is
// also the oracle for both variants.
#[cfg(any(test, feature = "test-support"))]
pub(super) fn probe_forest(
    data: &[f32],
    n: usize,
    features: usize,
    seed: u64,
    trees: usize,
    leaf_size: usize,
    join_threshold: Option<usize>,
) -> Result<ForestLeaves, WorkspaceError> {
    if n == 0 || features == 0 || leaf_size == 0 || trees == 0 || trees > MAX_TREES {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let expected = n
        .checked_mul(features)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    if data.len() != expected {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let membership_len = n
        .checked_mul(trees)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let per_tree: Vec<_> = (0..trees)
        .into_par_iter()
        .map(|tree| {
            let mut rows = (0..n).collect::<Vec<_>>();
            let mut out = ForestLeaves {
                row_ids: Vec::with_capacity(n),
                leaf_offsets: vec![0],
                tree_leaf_offsets: Vec::new(),
            };
            probe_split(
                data,
                features,
                &mut rows,
                leaf_size.min(n),
                seed_for(seed, tree as u64, 1),
                join_threshold,
                &mut out,
            );
            out
        })
        .collect();
    let mut result = ForestLeaves {
        row_ids: Vec::with_capacity(membership_len),
        leaf_offsets: vec![0],
        tree_leaf_offsets: Vec::with_capacity(trees + 1),
    };
    for tree in per_tree {
        result.tree_leaf_offsets.push(result.leaf_offsets.len() - 1);
        let row_offset = result.row_ids.len();
        result.row_ids.extend(tree.row_ids);
        result.leaf_offsets.extend(
            tree.leaf_offsets
                .into_iter()
                .skip(1)
                .map(|offset| row_offset + offset),
        );
    }
    result.tree_leaf_offsets.push(result.leaf_offsets.len() - 1);
    Ok(result)
}

#[cfg(any(test, feature = "test-support"))]
fn probe_split(
    data: &[f32],
    features: usize,
    rows: &mut [usize],
    leaf_size: usize,
    node_seed: u64,
    join_threshold: Option<usize>,
    out: &mut ForestLeaves,
) {
    if rows.len() <= leaf_size {
        out.row_ids.extend_from_slice(rows);
        out.leaf_offsets.push(out.row_ids.len());
        return;
    }
    let mut state = node_seed;
    let normal: Vec<_> = (0..features)
        .map(|_| next_unit(&mut state) * 2.0 - 1.0)
        .collect();
    let mut projected = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(8) {
        // `Iterator::sum::<f64>()` starts from negative zero.
        let mut sums = [-0.0f64; 8];
        for feature in 0..features {
            for (lane, &row) in chunk.iter().enumerate() {
                sums[lane] +=
                    f64::from(data[row * features + feature]) * f64::from(normal[feature]);
            }
        }
        for (lane, &row) in chunk.iter().enumerate() {
            projected.push((row, sums[lane], mix64(node_seed ^ row as u64)));
        }
    }
    let compare = |a: &(usize, f64, u64), b: &(usize, f64, u64)| {
        a.1.total_cmp(&b.1)
            .then_with(|| a.2.cmp(&b.2))
            .then_with(|| a.0.cmp(&b.0))
    };
    let midpoint = projected.len() / 2;
    projected.select_nth_unstable_by(midpoint, compare);
    for (row, item) in rows.iter_mut().zip(projected.iter()) {
        *row = item.0;
    }
    drop(projected);
    drop(normal);
    let join = join_threshold.is_some_and(|threshold| rows.len() >= threshold);
    let (left, right) = rows.split_at_mut(midpoint);
    let left_seed = seed_for(node_seed, 0, 0);
    let right_seed = seed_for(node_seed, 1, 0);
    if join {
        let mut right_out = ForestLeaves {
            row_ids: Vec::with_capacity(right.len()),
            leaf_offsets: vec![0],
            tree_leaf_offsets: Vec::new(),
        };
        rayon::join(
            || {
                probe_split(
                    data,
                    features,
                    left,
                    leaf_size,
                    left_seed,
                    join_threshold,
                    out,
                )
            },
            || {
                probe_split(
                    data,
                    features,
                    right,
                    leaf_size,
                    right_seed,
                    join_threshold,
                    &mut right_out,
                )
            },
        );
        let offset = out.row_ids.len();
        out.row_ids.extend(right_out.row_ids);
        out.leaf_offsets.extend(
            right_out
                .leaf_offsets
                .into_iter()
                .skip(1)
                .map(|value| offset + value),
        );
    } else {
        probe_split(
            data,
            features,
            left,
            leaf_size,
            left_seed,
            join_threshold,
            out,
        );
        probe_split(
            data,
            features,
            right,
            leaf_size,
            right_seed,
            join_threshold,
            out,
        );
    }
}

/// Plan forest and join shape, independent of the memory budget and tile size.
///
/// For `C=min(max_candidates,n-1)`, one center has
/// `P=C*(C-1)/2+C*C` fixed pair slots.
pub(super) fn plan_workspace(
    n: usize,
    heap_width: usize,
    max_candidates: usize,
    features: usize,
    trees: usize,
) -> Result<WorkspacePlan, WorkspaceError> {
    if n < 2
        || heap_width == 0
        || max_candidates == 0
        || features == 0
        || trees == 0
        || trees > MAX_TREES
    {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let forest = forest_totals(n, forest_leaf_limit(n, heap_width), trees)?;
    let c = max_candidates.min(n - 1);
    let pairs_per_center = c
        .checked_mul(c.checked_sub(1).ok_or(WorkspaceError::InvalidDimensions)?)
        .and_then(|v| v.checked_div(2))
        .and_then(|v| v.checked_add(c.checked_mul(c)?))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // Peak host memory includes the downloaded byte buffer and decoded f32
    // feature copy simultaneously (8*n*d bytes), temporary per-tree results
    // and final concatenated membership, concurrent Rayon recursion workspaces,
    // and the heap seed/upload buffers. The feature copies and tree temporaries
    // are released before pair staging is allocated.
    let membership_bytes = n
        .checked_mul(trees)
        .and_then(|v| v.checked_mul(48))
        .and_then(|v| v.checked_add(trees.checked_add(1)?.checked_mul(8)?))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let feature_copy_bytes = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(8))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // Worst case: every tree builds concurrently. Using the thread count would
    // make the public estimate depend on the host machine, so the bound is the
    // tree count, which is shape-only and never underestimates the parallel build.
    let concurrent_trees = trees;
    let recursion_scratch_bytes = n
        .checked_mul(32)
        .and_then(|v| v.checked_add(features.checked_mul(4)?))
        .and_then(|v| v.checked_mul(concurrent_trees))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let host_graph_bytes = n
        .checked_mul(heap_width)
        .and_then(|v| v.checked_mul(28)) // Neighbor vectors plus upload id/distance/flag arrays.
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let host_peak_bytes = membership_bytes
        .checked_add(feature_copy_bytes)
        .and_then(|v| v.checked_add(recursion_scratch_bytes))
        .and_then(|v| v.checked_add(host_graph_bytes))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    n.checked_mul(pairs_per_center)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    Ok(WorkspacePlan {
        max_candidates_per_class: c,
        pair_slots_per_center: pairs_per_center,
        host_peak_bytes,
        forest_leaf_limit: forest.leaf_limit,
        forest_membership_len: forest.membership_len,
        forest_leaf_count: forest.leaf_count,
        forest_pair_count: forest.pair_count,
    })
}

/// Leaf size limit shared by the forest planner and `forest_seed`.
pub(super) fn forest_leaf_limit(n: usize, heap_width: usize) -> usize {
    10usize.max(heap_width.saturating_mul(2)).min(n)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ForestTotals {
    leaf_limit: usize,
    membership_len: usize,
    leaf_count: usize,
    pair_count: usize,
}

/// Shape-only forest totals from the midpoint split recurrence used by
/// `split_node`: a node of size `m <= L` is one leaf contributing
/// `m*(m-1)/2` pairs; otherwise it splits into `floor(m/2)` and `ceil(m/2)`.
/// Every tree has identical totals, so trees contribute `trees * per_tree`.
/// All tree totals, the i32-typed membership/offset values and the per-leaf pair
/// count are checked here, before any forest allocation.
fn forest_totals(
    n: usize,
    leaf_limit: usize,
    trees: usize,
) -> Result<ForestTotals, WorkspaceError> {
    if n < 2 || leaf_limit == 0 || trees == 0 {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let mut memo = std::collections::BTreeMap::new();
    let (leaves, pairs) = node_totals(n, leaf_limit, &mut memo)?;
    let membership_len = n
        .checked_mul(trees)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let leaf_count = leaves
        .checked_mul(trees)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let pair_count = pairs
        .checked_mul(trees)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // I32 storage: membership ids and leaf row offsets are bounded by membership_len.
    if membership_len > i32::MAX as usize {
        return Err(WorkspaceError::Capacity {
            quantity: "forest membership length (trees x rows)",
            value: membership_len,
            limit: i32::MAX as usize,
        });
    }
    // The forest pair prefix is I64, so the total is bounded only by the checked
    // usize arithmetic above. Within a leaf the kernel narrows the local pair to u32
    // and walks it in O(L), so one leaf's pairs must stay within i32. For every shape
    // `host_plan` accepts this is implied by its reverse gate (n * width <= i32::MAX):
    // L = 2w <= n gives L(L-1)/2 < 2w^2 <= n*w; L = n gives w >= n/2 and
    // L(L-1)/2 < n^2/2 <= n*w; L = 10 is trivially small. It is kept here because this
    // function takes an arbitrary leaf limit.
    let max_leaf = leaf_limit.min(n);
    let leaf_pairs = max_leaf
        .checked_mul(max_leaf - 1)
        .ok_or(WorkspaceError::ArithmeticOverflow)?
        / 2;
    if leaf_pairs > i32::MAX as usize {
        return Err(WorkspaceError::Capacity {
            quantity: "per-leaf pair count (L(L-1)/2)",
            value: leaf_pairs,
            limit: i32::MAX as usize,
        });
    }
    Ok(ForestTotals {
        leaf_limit,
        membership_len,
        leaf_count,
        pair_count,
    })
}

/// Returns `(leaves, pairs)` for a node of size `m`. Each recursion level has at
/// most two distinct sizes, so memoization keeps the work logarithmic in `n`.
fn node_totals(
    m: usize,
    leaf_limit: usize,
    memo: &mut std::collections::BTreeMap<usize, (usize, usize)>,
) -> Result<(usize, usize), WorkspaceError> {
    if m <= leaf_limit {
        let pairs = m
            .checked_mul(m - 1)
            .ok_or(WorkspaceError::ArithmeticOverflow)?
            / 2;
        return Ok((1, pairs));
    }
    if let Some(&total) = memo.get(&m) {
        return Ok(total);
    }
    let (left_leaves, left_pairs) = node_totals(m / 2, leaf_limit, memo)?;
    let (right_leaves, right_pairs) = node_totals(m.div_ceil(2), leaf_limit, memo)?;
    let total = (
        left_leaves
            .checked_add(right_leaves)
            .ok_or(WorkspaceError::ArithmeticOverflow)?,
        left_pairs
            .checked_add(right_pairs)
            .ok_or(WorkspaceError::ArithmeticOverflow)?,
    );
    memo.insert(m, total);
    Ok(total)
}

fn seed_for(seed: u64, stream: u64, path: u64) -> u64 {
    mix64(seed ^ mix64(stream.wrapping_add(0x9e37_79b9_7f4a_7c15)) ^ mix64(path))
}

fn mix64(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn next_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    (mix64(*state) >> 40) as f32 / (1u32 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_variants_match_production_forest_across_thread_counts() {
        let n = 37;
        let features = 9;
        let data: Vec<f32> = (0..n)
            .flat_map(|row| {
                (0..features).map(move |feature| {
                    if feature == 0 {
                        3.0
                    } else {
                        ((row / 3 * (feature + 1)) % 17) as f32 / 4.0
                    }
                })
            })
            .collect();
        let identical = vec![2.0; n * features];
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for points in [&data, &identical] {
                    for leaf_size in [4, n, n + 1] {
                        let expected = build_forest(points, n, features, 42, 3, leaf_size).unwrap();
                        for threshold in [None, Some(8)] {
                            let actual =
                                probe_forest(points, n, features, 42, 3, leaf_size, threshold)
                                    .unwrap();
                            assert_eq!(
                                actual, expected,
                                "threads={threads}, leaf_size={leaf_size}, threshold={threshold:?}"
                            );
                        }
                    }
                }
            });
        }
    }

    fn membership(value: usize) -> WorkspaceError {
        WorkspaceError::Capacity {
            quantity: "forest membership length (trees x rows)",
            value,
            limit: i32::MAX as usize,
        }
    }

    /// A memo is only valid for one leaf limit, so each call gets a fresh one.
    fn totals(m: usize, leaf_limit: usize) -> (usize, usize) {
        let mut memo = std::collections::BTreeMap::new();
        node_totals(m, leaf_limit, &mut memo).unwrap()
    }

    #[test]
    fn node_totals_follow_midpoint_split_recurrence() {
        assert_eq!(totals(10, 10), (1, 45));
        assert_eq!(totals(11, 10), (2, 25));
        assert_eq!(totals(7, 1), (7, 0));
        assert_eq!(totals(1000, 1000), (1, 499_500));
        // Multi-level split: 1000 -> 500 -> 250 -> 125 -> (62, 63) leaves.
        assert_eq!(totals(1000, 100), (16, 30_752));
        // Shared memo for one fixed limit: cached interior nodes match.
        let mut memo = std::collections::BTreeMap::new();
        assert_eq!(node_totals(1000, 100, &mut memo).unwrap(), (16, 30_752));
        assert_eq!(node_totals(500, 100, &mut memo).unwrap(), (8, 15_376));
    }

    #[test]
    fn forest_totals_reach_plan_for_small_shape() {
        // 3 trees over 11 rows with limit 10 -> 6 leaves, 75 pairs.
        let plan = plan_workspace(11, 4, 10, 2, 3).unwrap();
        assert_eq!(plan.forest_leaf_limit, 10);
        assert_eq!(plan.forest_membership_len, 33);
        assert_eq!(plan.forest_leaf_count, 6);
        assert_eq!(plan.forest_pair_count, 75);
    }

    #[test]
    fn plan_workspace_accepts_forest_prefix_above_i32() {
        assert_eq!(tree_count(390_000).unwrap(), 36);
        // Leaf limit min(max(10, 2*260), n) = 520 gives 1024 leaves per tree:
        // 390000 = 1024*380 + 880, so 880 leaves of 381 rows and 144 of 380:
        // 880*72390 + 144*72010 = 74_072_640 pairs per tree; x36 = 2_666_615_040,
        // above i32::MAX and carried by the I64 forest pair prefix.
        let plan = plan_workspace(390_000, 260, MAX_CANDIDATES, 2, 36).unwrap();
        assert_eq!(plan.forest_pair_count, 2_666_615_040);
    }

    #[test]
    fn forest_totals_accept_and_reject_adjacent_i32_limits() {
        // Membership = trees * n. One leaf per row (limit 1) keeps pairs at zero.
        assert_eq!(
            forest_totals(i32::MAX as usize, 1, 1)
                .unwrap()
                .membership_len,
            i32::MAX as usize
        );
        assert_eq!(
            forest_totals(i32::MAX as usize + 1, 1, 1),
            Err(membership(i32::MAX as usize + 1))
        );
        assert_eq!(
            forest_totals(i32::MAX as usize, 1, 2),
            Err(membership(2 * i32::MAX as usize))
        );
        // Per-leaf pairs m(m-1)/2 for a single leaf (limit = n):
        // 65536*65535/2 = 2_147_450_880 accepted; 65537*65536/2 = 2_147_516_416 rejected.
        assert_eq!(
            forest_totals(65_536, 65_536, 1).unwrap().pair_count,
            2_147_450_880
        );
        assert_eq!(
            forest_totals(65_537, 65_537, 1),
            Err(WorkspaceError::Capacity {
                quantity: "per-leaf pair count (L(L-1)/2)",
                value: 2_147_516_416,
                limit: i32::MAX as usize,
            })
        );
        // Two trees put the pair total above i32; the I64 prefix accepts it.
        assert_eq!(
            forest_totals(65_536, 65_536, 2).unwrap().pair_count,
            4_294_901_760
        );
    }

    #[test]
    fn forest_is_deterministic_and_partitions_duplicate_rows_balanced() {
        let points = vec![3.0f32; 19];
        let first = build_forest(&points, 19, 1, 0x5179, 3, 4).unwrap();
        let second = build_forest(&points, 19, 1, 0x5179, 3, 4).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.row_ids.len(), 57);
        assert_eq!(first.tree_leaf_offsets, vec![0, 7, 14, 21]);
        for tree in 0..3 {
            let leaf_start = first.tree_leaf_offsets[tree];
            let leaf_end = first.tree_leaf_offsets[tree + 1];
            let mut members = first.row_ids
                [first.leaf_offsets[leaf_start]..first.leaf_offsets[leaf_end]]
                .to_vec();
            members.sort_unstable();
            assert_eq!(members, (0..19).collect::<Vec<_>>());
            assert!(
                first.leaf_offsets[leaf_start..=leaf_end]
                    .windows(2)
                    .all(|pair| pair[1] - pair[0] <= 4)
            );
        }
    }

    #[test]
    fn forest_keeps_leaf_membership_contiguous_and_tree_offsets_well_formed() {
        let points = (0..30).map(|value| value as f32).collect::<Vec<_>>();
        let forest = build_forest(&points, 10, 3, 42, 2, 3).unwrap();
        assert_eq!(forest.row_ids.len(), 20);
        assert_eq!(forest.leaf_offsets.first(), Some(&0));
        assert_eq!(forest.leaf_offsets.last(), Some(&forest.row_ids.len()));
        assert_eq!(forest.tree_leaf_offsets, vec![0, 4, 8]);
        assert!(forest.leaf_offsets.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn workspace_plan_reports_shape_and_host_peak() {
        let plan = plan_workspace(50_000, 14, MAX_CANDIDATES, 32, 16).unwrap();
        assert_eq!(plan.max_candidates_per_class, 60);
        assert_eq!(plan.pair_slots_per_center, 5_370);
        assert_eq!(plan.host_peak_bytes, 96_402_184);
        assert_eq!(
            plan_workspace(50_000, 14, MAX_CANDIDATES, 32, 16).unwrap(),
            plan
        );
    }

    #[test]
    fn workspace_plan_preserves_forest_shape_across_candidate_widths() {
        let join = plan_workspace(11, 4, 10, 2, 3).unwrap();
        assert_eq!(join.pair_slots_per_center, 145);
        let forest = plan_workspace(11, 4, 1, 2, 3).unwrap();
        assert_eq!(forest.max_candidates_per_class, 1);
        assert_eq!(forest.pair_slots_per_center, 1);
        assert_eq!(forest.forest_pair_count, 75);
        assert_eq!(forest.forest_pair_count, join.forest_pair_count);
        assert_eq!(forest.forest_leaf_count, join.forest_leaf_count);
        assert_eq!(forest.host_peak_bytes, join.host_peak_bytes);
    }

    #[test]
    fn workspace_plan_rejects_shape_overflow() {
        assert_eq!(
            plan_workspace(usize::MAX, 1, 1, 1, 1),
            Err(WorkspaceError::ArithmeticOverflow)
        );
    }

    #[test]
    fn workspace_plan_scales_past_former_fixed_limit() {
        for n in [250_000usize, 1_000_000] {
            let trees = tree_count(n).unwrap();
            let plan = plan_workspace(n, 19, MAX_CANDIDATES, 784, trees).unwrap();
            assert_eq!(plan.pair_slots_per_center, 5_370);
            assert!(plan.host_peak_bytes > 512 * 1024 * 1024);
        }
    }

    #[test]
    fn tree_count_uses_contract_formula_and_cap() {
        assert_eq!(tree_count(1).unwrap(), 5);
        assert_eq!(tree_count(40_000).unwrap(), 15);
        assert_eq!(tree_count(50_000).unwrap(), 16);
        assert_eq!(tree_count(100_000_000).unwrap(), MAX_TREES);
        assert_eq!(tree_count(0), Err(WorkspaceError::InvalidDimensions));
    }
}
