//! Deterministic RP-forest leaf membership and bounded ANN workspace planning.
//!
//! The forest stores only contiguous row membership and leaf/tree offsets. It
//! deliberately does not materialize any within-leaf pairs.

use rayon::prelude::*;

pub(super) const MAX_TREES: usize = 64;
pub(super) const MAX_CANDIDATES: usize = 60;
pub(super) const STAGING_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const WORKSPACE_FIXED_BYTES: usize = 1024 * 1024;

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
    pub(super) max_pair_slots_per_tile: usize,
    pub(super) centers_per_tile: usize,
    pub(super) max_pairs_per_tile: usize,
    pub(super) max_endpoint_records: usize,
    /// Actual peak auxiliary allocation estimate, including the host forest.
    pub(super) workspace_bytes: usize,
    /// Approved W expression for this tile, including its one-MiB reserve.
    pub(super) formula_workspace_bytes: usize,
    /// Peak host-only phase: feature copy plus forest and recursion scratch.
    pub(super) host_peak_bytes: usize,
    /// Pair and endpoint staging allocation charged to the 256-MiB cap.
    pub(super) staging_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WorkspaceError {
    InvalidDimensions,
    ArithmeticOverflow,
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

/// Return the largest center tile satisfying the staging cap.
///
/// For `C=min(max_candidates,n-1)`, one center has
/// `P=C*(C-1)/2+C*C` fixed pair slots. A tile may end mid-center; for `Q`
/// flattened slots and `R=2*Q` endpoint records, the conservative workspace is
/// exactly
/// `W=48*n*H + 32*n*C + 32*(n+1) + 32*R + 1MiB`; staging is charged as
/// `32*R` and capped at 256 MiB. The persistent and host terms are linear in
/// `n` and cannot be tiled, so they are not capped here; device allocation
/// failures surface from the runtime. All terms and conversions are checked.
/// No whole-leaf pair array is planned or allocated.
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
    let c = max_candidates.min(n - 1);
    let pairs_per_center = c
        .checked_mul(c.checked_sub(1).ok_or(WorkspaceError::InvalidDimensions)?)
        .and_then(|v| v.checked_div(2))
        .and_then(|v| v.checked_add(c.checked_mul(c)?))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let formula_persistent = n
        .checked_mul(heap_width)
        .and_then(|v| v.checked_mul(48))
        .and_then(|v| v.checked_add(n.checked_mul(c)?.checked_mul(32)?))
        .and_then(|v| v.checked_add(n.checked_add(1)?.checked_mul(32)?))
        .and_then(|v| v.checked_add(WORKSPACE_FIXED_BYTES))
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
    let concurrent_trees = rayon::current_num_threads().min(trees).max(1);
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
    // Conservative estimate of allocated persistent buffers, kept separate
    // from W's reserve so the forest is charged against the reserve rather
    // than added on top of W.
    let actual_persistent = n
        .checked_mul(heap_width)
        .and_then(|v| v.checked_mul(32))
        .and_then(|v| v.checked_add(n.checked_mul(c)?.checked_mul(16)?))
        .and_then(|v| v.checked_add(n.checked_add(1)?.checked_mul(32)?))
        .and_then(|v| v.checked_add(WORKSPACE_FIXED_BYTES))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // Host forest, downloaded data, and seed heaps are released before pair
    // staging allocation; the device persistent buffers remain for the join.
    debug_assert!(actual_persistent <= formula_persistent);
    let total_pair_slots = n
        .checked_mul(pairs_per_center)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let staging_bytes_per_pair_slot = 64usize; // R=2Q records at 32 bytes each.
    let max_slots_by_staging = STAGING_LIMIT_BYTES / staging_bytes_per_pair_slot;
    let max_pair_slots_per_tile = total_pair_slots.min(max_slots_by_staging);
    let centers_per_tile = max_pair_slots_per_tile.div_ceil(pairs_per_center);
    let max_pairs_per_tile = max_pair_slots_per_tile;
    let max_endpoint_records = max_pairs_per_tile
        .checked_mul(2)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let staging_bytes = max_endpoint_records
        .checked_mul(32)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let formula_workspace_bytes = formula_persistent
        .checked_add(staging_bytes)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let stage_peak_bytes = actual_persistent
        .checked_add(staging_bytes)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let workspace_bytes = host_peak_bytes.max(stage_peak_bytes);
    Ok(WorkspacePlan {
        max_candidates_per_class: c,
        pair_slots_per_center: pairs_per_center,
        max_pair_slots_per_tile,
        centers_per_tile,
        max_pairs_per_tile,
        max_endpoint_records,
        workspace_bytes,
        formula_workspace_bytes,
        host_peak_bytes,
        staging_bytes,
    })
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
    fn workspace_plan_respects_both_caps_and_uses_pair_slot_formula() {
        let plan = plan_workspace(50_000, 14, MAX_CANDIDATES, 32, 16).unwrap();
        let c = 60usize;
        let per_center = c * (c - 1) / 2 + c * c;
        assert_eq!(plan.max_candidates_per_class, c);
        assert_eq!(plan.max_pairs_per_tile, plan.max_pair_slots_per_tile);
        assert_eq!(plan.pair_slots_per_center, per_center);
        assert!(plan.max_pair_slots_per_tile > 0);
        assert_eq!(
            plan.centers_per_tile,
            plan.max_pair_slots_per_tile.div_ceil(per_center)
        );
        assert_eq!(plan.max_endpoint_records, 2 * plan.max_pairs_per_tile);
        assert_eq!(plan.staging_bytes, 32 * plan.max_endpoint_records);
        assert!(plan.workspace_bytes <= plan.formula_workspace_bytes);
        assert!(plan.host_peak_bytes <= plan.workspace_bytes);
        assert!(plan.staging_bytes <= STAGING_LIMIT_BYTES);
        assert!(plan.host_peak_bytes >= 4 * 50_000 * 32);
        let larger = plan_workspace(50_000, 14, MAX_CANDIDATES, 32, 16).unwrap();
        assert_eq!(plan, larger);
    }

    #[test]
    fn workspace_plan_rejects_persistent_overflow() {
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
            assert!(plan.max_pair_slots_per_tile > 0);
            assert!(plan.staging_bytes <= STAGING_LIMIT_BYTES);
            assert!(plan.workspace_bytes > 512 * 1024 * 1024);
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
