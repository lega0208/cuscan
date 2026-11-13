//! Deterministic RP-forest leaf membership and bounded ANN workspace planning.
//!
//! The forest stores only contiguous row membership and leaf/tree offsets. It
//! deliberately does not materialize any within-leaf pairs.
//!
//! Construction has two phases with identical results. While every node of a
//! level is larger than [`TOP_NODE_BYTES`] of features, the level is built
//! breadth-first: one row-major pass over the data computes every tree's
//! projection of each row, so the feature matrix is streamed once per level
//! instead of once per tree. Below that size each node's rows are cache-sized
//! and the subtrees are built depth-first, eight rows per projection batch,
//! with Rayon joins for large subtrees. Every projection is the same sequence
//! of f64 operations in feature order, so membership and offsets do not
//! depend on the phase split, the batch width or the thread count.

use rayon::prelude::*;

pub(super) const MAX_TREES: usize = 64;
pub(super) const MAX_CANDIDATES: usize = 60;

/// Node size, in feature bytes, above which a level is built row-major.
const TOP_NODE_BYTES: usize = 2 << 20;
/// Breadth-first levels are capped so a node index fits in `u16`.
const MAX_TOP_LEVELS: usize = 16;
/// Depth-first subtrees at least this many rows split their halves with `rayon::join`.
const JOIN_ROWS: usize = 8192;
/// Rows of the same node projected together with independent accumulators.
const LANES: usize = 8;
/// Rows per task of the row-major projection pass.
const ROW_BLOCK: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ForestLeaves {
    /// Row ids grouped contiguously by leaf across all trees; uploaded as I32.
    pub(super) row_ids: Vec<i32>,
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
/// balanced progress even for duplicate points or a zero normal.
pub(super) fn build_forest(
    data: &[f32],
    n: usize,
    features: usize,
    seed: u64,
    tree_count: usize,
    leaf_size: usize,
) -> Result<ForestLeaves, WorkspaceError> {
    build_forest_with(
        data,
        n,
        features,
        seed,
        tree_count,
        leaf_size,
        TOP_NODE_BYTES,
        JOIN_ROWS,
    )
}

/// Selection entry: projection, seeded priority, row. The comparison order is
/// the split contract: projection, then priority, then row id.
type Entry = (f64, u64, u32);

fn compare(a: &Entry, b: &Entry) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0)
        .then_with(|| a.1.cmp(&b.1))
        .then_with(|| a.2.cmp(&b.2))
}

/// Number of leading levels built row-major. At level `l` the smallest node has
/// `floor(n / 2^l)` rows; a level is row-major only if that node still splits
/// and is at least `top_node_bytes` of features. Shared with the planner.
fn top_levels(n: usize, features: usize, leaf_size: usize, top_node_bytes: usize) -> usize {
    let min_rows = top_node_bytes
        .div_ceil(features.saturating_mul(4).max(1))
        .max(1);
    let mut levels = 0;
    let mut rows = n;
    while levels < MAX_TOP_LEVELS && rows > leaf_size && rows >= min_rows {
        levels += 1;
        rows /= 2;
    }
    levels
}

#[allow(clippy::too_many_arguments)]
fn build_forest_with(
    data: &[f32],
    n: usize,
    features: usize,
    seed: u64,
    trees: usize,
    leaf_size: usize,
    top_node_bytes: usize,
    join_rows: usize,
) -> Result<ForestLeaves, WorkspaceError> {
    if n == 0 || features == 0 || leaf_size == 0 || trees == 0 || trees > MAX_TREES {
        return Err(WorkspaceError::InvalidDimensions);
    }
    if n > i32::MAX as usize {
        return Err(WorkspaceError::Capacity {
            quantity: "forest rows",
            value: n,
            limit: i32::MAX as usize,
        });
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
    let leaf_size = leaf_size.min(n);
    // Each tree owns one `n`-row slice and permutes it in place.
    let mut membership = vec![0i32; membership_len];
    membership.par_chunks_mut(n).for_each(|rows| {
        for (slot, row) in rows.iter_mut().zip(0i32..) {
            *slot = row;
        }
    });
    // Node bounds of the current level are shared by all trees: every split
    // is at the midpoint, so node sizes depend only on `n`.
    let mut bounds = vec![0, n];
    let mut seeds = (0..trees)
        .map(|tree| vec![seed_for(seed, tree as u64, 1)])
        .collect::<Vec<_>>();
    let levels = top_levels(n, features, leaf_size, top_node_bytes);
    if levels > 0 {
        let mut projections = vec![0.0f64; membership_len];
        let mut node_of = vec![0u16; membership_len];
        for _ in 0..levels {
            split_level(
                data,
                features,
                n,
                &mut membership,
                &bounds,
                &seeds,
                &mut projections,
                &mut node_of,
            );
            let mut next = Vec::with_capacity(2 * bounds.len() - 1);
            for node in bounds.windows(2) {
                next.push(node[0]);
                next.push(node[0] + (node[1] - node[0]) / 2);
            }
            next.push(n);
            bounds = next;
            for tree_seeds in &mut seeds {
                *tree_seeds = tree_seeds
                    .iter()
                    .flat_map(|&node| [seed_for(node, 0, 0), seed_for(node, 1, 0)])
                    .collect();
            }
        }
    }
    // Depth-first below the row-major levels; per-tree leaf end offsets,
    // relative to the tree's slice, in left-to-right order.
    let tree_leaves = membership
        .par_chunks_mut(n)
        .zip(seeds.par_iter())
        .map(|(tree_rows, tree_seeds)| {
            let mut nodes = Vec::with_capacity(tree_seeds.len());
            let mut rest = tree_rows;
            for (node, &node_seed) in bounds.windows(2).zip(tree_seeds) {
                let (rows, tail) = rest.split_at_mut(node[1] - node[0]);
                nodes.push((rows, node[0], node_seed));
                rest = tail;
            }
            nodes
                .into_par_iter()
                .map(|(rows, base, node_seed)| {
                    let mut leaves = Vec::new();
                    let mut scratch = Scratch::default();
                    split_subtree(
                        data,
                        features,
                        rows,
                        base,
                        leaf_size,
                        node_seed,
                        join_rows,
                        &mut scratch,
                        &mut leaves,
                    );
                    leaves
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let leaf_count = tree_leaves.iter().flatten().map(Vec::len).sum::<usize>();
    let mut leaf_offsets = Vec::with_capacity(leaf_count + 1);
    let mut tree_leaf_offsets = Vec::with_capacity(trees + 1);
    leaf_offsets.push(0);
    for (tree, nodes) in tree_leaves.into_iter().enumerate() {
        tree_leaf_offsets.push(leaf_offsets.len() - 1);
        let row_offset = tree * n;
        leaf_offsets.extend(nodes.into_iter().flatten().map(|end| row_offset + end));
    }
    tree_leaf_offsets.push(leaf_offsets.len() - 1);
    Ok(ForestLeaves {
        row_ids: membership,
        leaf_offsets,
        tree_leaf_offsets,
    })
}

/// Split every node of one level in every tree. `bounds` are the level's node
/// offsets and `seeds[tree][node]` the node seeds. `projections` is indexed
/// `[row][tree]` and `node_of` `[tree][row]`.
#[allow(clippy::too_many_arguments)]
fn split_level(
    data: &[f32],
    features: usize,
    n: usize,
    membership: &mut [i32],
    bounds: &[usize],
    seeds: &[Vec<u64>],
    projections: &mut [f64],
    node_of: &mut [u16],
) {
    let trees = seeds.len();
    let nodes = bounds.len() - 1;
    let mut normals = vec![0.0f32; trees * nodes * features];
    normals
        .par_chunks_mut(nodes * features)
        .zip(seeds.par_iter())
        .for_each(|(tree_normals, tree_seeds)| {
            for (normal, &node_seed) in tree_normals.chunks_mut(features).zip(tree_seeds) {
                fill_normal(normal, node_seed);
            }
        });
    node_of
        .par_chunks_mut(n)
        .zip(membership.par_chunks(n))
        .for_each(|(tree_nodes, tree_rows)| {
            for (node, range) in bounds.windows(2).enumerate() {
                for &row in &tree_rows[range[0]..range[1]] {
                    tree_nodes[row as usize] = node as u16;
                }
            }
        });
    // One pass over the rows; each row is read once for all trees.
    projections
        .par_chunks_mut(ROW_BLOCK * trees)
        .enumerate()
        .for_each(|(block, out)| {
            let start = block * ROW_BLOCK;
            let rows = out.len() / trees;
            for tree in 0..trees {
                let tree_nodes = &node_of[tree * n + start..tree * n + start + rows];
                let tree_normals = &normals[tree * nodes * features..(tree + 1) * nodes * features];
                for lane_start in (0..rows).step_by(LANES) {
                    let lanes = LANES.min(rows - lane_start);
                    // Short tails repeat the first row; extra sums are discarded.
                    let lane = |lane: usize| lane_start + if lane < lanes { lane } else { 0 };
                    let sums = project_lanes(
                        std::array::from_fn(|l| {
                            let row = start + lane(l);
                            &data[row * features..(row + 1) * features]
                        }),
                        std::array::from_fn(|l| {
                            let node = tree_nodes[lane(l)] as usize;
                            &tree_normals[node * features..(node + 1) * features]
                        }),
                    );
                    for (lane, &sum) in sums.iter().enumerate().take(lanes) {
                        out[(lane_start + lane) * trees + tree] = sum;
                    }
                }
            }
        });
    let projections = &*projections;
    membership
        .par_chunks_mut(n)
        .zip(seeds.par_iter())
        .enumerate()
        .for_each(|(tree, (tree_rows, tree_seeds))| {
            let mut nodes = Vec::with_capacity(tree_seeds.len());
            let mut rest = tree_rows;
            for (node, &node_seed) in bounds.windows(2).zip(tree_seeds) {
                let (rows, tail) = rest.split_at_mut(node[1] - node[0]);
                nodes.push((rows, node_seed));
                rest = tail;
            }
            nodes.into_par_iter().for_each(|(rows, node_seed)| {
                let mut entries = rows
                    .iter()
                    .map(|&row| {
                        (
                            projections[row as usize * trees + tree],
                            mix64(node_seed ^ row as u64),
                            row as u32,
                        )
                    })
                    .collect::<Vec<_>>();
                select_midpoint(rows, &mut entries);
            });
        });
}

/// Partition `rows` at the midpoint by `entries` (built in `rows` order) and
/// write the selected order back.
fn select_midpoint(rows: &mut [i32], entries: &mut [Entry]) {
    entries.select_nth_unstable_by(rows.len() / 2, compare);
    // Each side is ordered only enough to partition by projection; equal
    // projections split by deterministic node/row priority at the midpoint.
    for (row, entry) in rows.iter_mut().zip(entries.iter()) {
        *row = entry.2 as i32;
    }
}

/// Projections of `LANES` rows, each onto its own normal. Every lane is the
/// serial f64 sum `Iterator::sum` computes (starting from negative zero, in
/// feature order); lanes are independent, so batching changes no bits.
#[inline(always)]
fn project_lanes(rows: [&[f32]; LANES], normals: [&[f32]; LANES]) -> [f64; LANES] {
    let features = rows[0].len();
    for lane in 0..LANES {
        assert!(rows[lane].len() == features && normals[lane].len() == features);
    }
    let mut sums = [-0.0f64; LANES];
    for feature in 0..features {
        for lane in 0..LANES {
            sums[lane] += f64::from(rows[lane][feature]) * f64::from(normals[lane][feature]);
        }
    }
    sums
}

fn fill_normal(normal: &mut [f32], node_seed: u64) {
    let mut state = node_seed;
    for value in normal {
        *value = next_unit(&mut state) * 2.0 - 1.0;
    }
}

/// Per-task buffers reused across the nodes of one depth-first subtree.
#[derive(Default)]
struct Scratch {
    entries: Vec<Entry>,
    normal: Vec<f32>,
}

/// Depth-first split of one node; appends leaf end offsets (`base` plus the
/// end within `rows`) left to right.
#[allow(clippy::too_many_arguments)]
fn split_subtree(
    data: &[f32],
    features: usize,
    rows: &mut [i32],
    base: usize,
    leaf_size: usize,
    node_seed: u64,
    join_rows: usize,
    scratch: &mut Scratch,
    leaves: &mut Vec<usize>,
) {
    if rows.len() <= leaf_size {
        leaves.push(base + rows.len());
        return;
    }
    scratch.normal.resize(features, 0.0);
    fill_normal(&mut scratch.normal, node_seed);
    scratch.entries.clear();
    let normal = scratch.normal.as_slice();
    for chunk in rows.chunks(LANES) {
        // Short tails repeat the first row; extra sums are discarded.
        let sums = project_lanes(
            std::array::from_fn(|lane| {
                let row = chunk.get(lane).unwrap_or(&chunk[0]);
                let row = *row as usize;
                &data[row * features..(row + 1) * features]
            }),
            [normal; LANES],
        );
        for (&sum, &row) in sums.iter().zip(chunk) {
            scratch
                .entries
                .push((sum, mix64(node_seed ^ row as u64), row as u32));
        }
    }
    select_midpoint(rows, &mut scratch.entries);
    let midpoint = rows.len() / 2;
    let left_seed = seed_for(node_seed, 0, 0);
    let right_seed = seed_for(node_seed, 1, 0);
    let (left, right) = rows.split_at_mut(midpoint);
    let args = (data, features, leaf_size, join_rows);
    if left.len() + right.len() >= join_rows {
        // Release this node's buffers before forking so in-flight join frames
        // hold no selection entries; live entries stay within `n` per tree.
        *scratch = Scratch::default();
        let mut right_leaves = Vec::new();
        rayon::join(
            || {
                let mut left_scratch = Scratch::default();
                subtree(args, left, base, left_seed, &mut left_scratch, leaves)
            },
            || {
                let mut right_scratch = Scratch::default();
                subtree(
                    args,
                    right,
                    base + midpoint,
                    right_seed,
                    &mut right_scratch,
                    &mut right_leaves,
                )
            },
        );
        leaves.append(&mut right_leaves);
    } else {
        subtree(args, left, base, left_seed, scratch, leaves);
        subtree(args, right, base + midpoint, right_seed, scratch, leaves);
    }
}

fn subtree(
    (data, features, leaf_size, join_rows): (&[f32], usize, usize, usize),
    rows: &mut [i32],
    base: usize,
    node_seed: u64,
    scratch: &mut Scratch,
    leaves: &mut Vec<usize>,
) {
    split_subtree(
        data, features, rows, base, leaf_size, node_seed, join_rows, scratch, leaves,
    );
}

/// Frozen copy of the original serial-sum builder, the bit-exactness oracle
/// for [`build_forest`] in tests and the forest probe example.
#[cfg(any(test, feature = "test-support"))]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "used by the ann_forest_probe example")
)]
pub(super) fn reference_build_forest(
    data: &[f32],
    n: usize,
    features: usize,
    seed: u64,
    tree_count: usize,
    leaf_size: usize,
) -> Result<ForestLeaves, WorkspaceError> {
    struct Reference {
        row_ids: Vec<usize>,
        leaf_offsets: Vec<usize>,
    }
    fn split(
        data: &[f32],
        features: usize,
        rows: &mut [usize],
        leaf_size: usize,
        node_seed: u64,
        out: &mut Reference,
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
        for (row, item) in rows.iter_mut().zip(projected.iter()) {
            *row = item.0;
        }
        let (left, right) = rows.split_at_mut(midpoint);
        split(
            data,
            features,
            left,
            leaf_size,
            seed_for(node_seed, 0, 0),
            out,
        );
        split(
            data,
            features,
            right,
            leaf_size,
            seed_for(node_seed, 1, 0),
            out,
        );
    }
    if n == 0 || features == 0 || leaf_size == 0 || tree_count == 0 || tree_count > MAX_TREES {
        return Err(WorkspaceError::InvalidDimensions);
    }
    if data.len()
        != n.checked_mul(features)
            .ok_or(WorkspaceError::ArithmeticOverflow)?
    {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let per_tree = (0..tree_count)
        .into_par_iter()
        .map(|tree| {
            let mut result = Reference {
                row_ids: Vec::with_capacity(n),
                leaf_offsets: vec![0],
            };
            let mut rows = (0..n).collect::<Vec<_>>();
            split(
                data,
                features,
                &mut rows,
                leaf_size.min(n),
                seed_for(seed, tree as u64, 1),
                &mut result,
            );
            result
        })
        .collect::<Vec<_>>();
    let mut result = ForestLeaves {
        row_ids: Vec::with_capacity(n * tree_count),
        leaf_offsets: vec![0],
        tree_leaf_offsets: Vec::with_capacity(tree_count + 1),
    };
    for tree in per_tree {
        result.tree_leaf_offsets.push(result.leaf_offsets.len() - 1);
        let row_offset = result.row_ids.len();
        result
            .row_ids
            .extend(tree.row_ids.into_iter().map(|row| row as i32));
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
    // Peak host memory: the downloaded bytes and a decoded f32 copy (8*n*d,
    // the unaligned path; the aligned path reuses the download), the I32
    // membership, leaf offsets with their I32 row and I64 pair-prefix upload
    // copies, selection scratch, the row-major level buffers and the heap
    // seed/upload buffers. All phases are summed, which bounds the peak.
    let membership_bytes = n
        .checked_mul(trees)
        .and_then(|v| v.checked_mul(4))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let leaf_bytes = forest
        .leaf_count
        .checked_add(1)
        .and_then(|v| v.checked_mul(8 + 4 + 8))
        .and_then(|v| v.checked_add(trees.checked_add(1)?.checked_mul(8)?))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let feature_copy_bytes = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(8))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // Worst case: every tree builds concurrently. Using the thread count would
    // make the public estimate depend on the host machine, so the bound is the
    // tree count, which is shape-only and never underestimates the parallel build.
    // Each tree holds at most `n` 24-byte selection entries and one normal.
    let concurrent_trees = trees;
    let recursion_scratch_bytes = n
        .checked_mul(24)
        .and_then(|v| v.checked_add(features.checked_mul(4)?))
        .and_then(|v| v.checked_mul(concurrent_trees))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    // Row-major levels: f64 projections and u16 node ids per (tree, row), and
    // the normals of the widest row-major level.
    let levels = top_levels(n, features, forest.leaf_limit, TOP_NODE_BYTES);
    let top_level_bytes = if levels == 0 {
        0
    } else {
        let widest_nodes = 1usize << (levels - 1);
        n.checked_mul(trees)
            .and_then(|v| v.checked_mul(8 + 2))
            .and_then(|v| {
                v.checked_add(
                    trees
                        .checked_mul(widest_nodes)?
                        .checked_mul(features)?
                        .checked_mul(4)?,
                )
            })
            .ok_or(WorkspaceError::ArithmeticOverflow)?
    };
    let host_graph_bytes = n
        .checked_mul(heap_width)
        .and_then(|v| v.checked_mul(28)) // Neighbor vectors plus upload id/distance/flag arrays.
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let host_peak_bytes = membership_bytes
        .checked_add(leaf_bytes)
        .and_then(|v| v.checked_add(top_level_bytes))
        .and_then(|v| v.checked_add(feature_copy_bytes))
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

    /// Duplicated rows (groups of three), a zero-variance first feature,
    /// signed zeros and a few wide magnitudes.
    fn mixed_points(n: usize, features: usize) -> Vec<f32> {
        (0..n)
            .flat_map(|row| {
                (0..features).map(move |feature| match feature {
                    0 => 3.0,
                    1 if row % 5 == 0 => -0.0,
                    2 if row % 7 == 0 => 1.0e30 * if row % 2 == 0 { 1.0 } else { -1.0 },
                    _ => ((row / 3 * (feature + 1)) % 17) as f32 / 4.0 - 2.0,
                })
            })
            .collect()
    }

    /// Production equals the frozen serial builder for every row-major level
    /// count (forced by tiny node-byte thresholds), join threshold and thread
    /// count, including duplicates, zero-variance features, non-power-of-two n
    /// and `leaf_size >= n`.
    #[test]
    fn forest_matches_frozen_reference_across_phases_and_threads() {
        let cases: [(usize, usize, usize); 4] =
            [(37, 9, 3), (1000, 5, 7), (777, 33, 2), (64, 1, 5)];
        // Alternating +0/-0 rows project to signed zeros, checking the -0.0 start.
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for (n, features, trees) in cases {
                    let mixed = mixed_points(n, features);
                    let identical = vec![2.0; n * features];
                    let zeros = (0..n * features)
                        .map(|i| if (i / features) % 2 == 0 { 0.0 } else { -0.0 })
                        .collect::<Vec<f32>>();
                    for points in [&mixed, &identical, &zeros] {
                        for leaf_size in [1, 4, 10, n, n + 1] {
                            let expected =
                                reference_build_forest(points, n, features, 42, trees, leaf_size)
                                    .unwrap();
                            assert_eq!(
                                build_forest(points, n, features, 42, trees, leaf_size).unwrap(),
                                expected
                            );
                            // Node bytes of 1 make every splitting level row-major
                            // (up to the u16 cap); larger values stop part-way.
                            for top_node_bytes in [1, 4 * features * n / 8, usize::MAX] {
                                for join_rows in [2, 8, usize::MAX] {
                                    let actual = build_forest_with(
                                        points,
                                        n,
                                        features,
                                        42,
                                        trees,
                                        leaf_size,
                                        top_node_bytes,
                                        join_rows,
                                    )
                                    .unwrap();
                                    assert_eq!(
                                        actual, expected,
                                        "n={n} features={features} trees={trees} \
                                         leaf_size={leaf_size} threads={threads} \
                                         top_node_bytes={top_node_bytes} join_rows={join_rows}"
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }
    }

    /// Row-major levels stop at the `MAX_TOP_LEVELS` cap (2^16 rows, leaf size
    /// 1) and the depth-first phase finishes the remaining levels.
    #[test]
    fn forest_matches_reference_across_top_level_cap() {
        let n = 70_001;
        assert_eq!(top_levels(n, 1, 1, 1), MAX_TOP_LEVELS);
        let points = mixed_points(n, 1);
        let expected = reference_build_forest(&points, n, 1, 7, 1, 1).unwrap();
        assert_eq!(
            build_forest_with(&points, n, 1, 7, 1, 1, 1, 8192).unwrap(),
            expected
        );
    }

    #[test]
    fn top_levels_stop_at_leaf_size_node_bytes_and_cap() {
        // 1.5M x 256: nodes of at least 2 MiB / 1 KiB = 2048 rows; floor(1.5M/2^l)
        // is 2929 at l = 9 and 1464 at l = 10, so levels 0..=9 are row-major.
        assert_eq!(top_levels(1_500_000, 256, 36, TOP_NODE_BYTES), 10);
        // 50k x 32: 2 MiB / 128 B = 16384 rows; 50000 and 25000 split row-major.
        assert_eq!(top_levels(50_000, 32, 28, TOP_NODE_BYTES), 2);
        // 1000 x 32 (128 KB) is below one node.
        assert_eq!(top_levels(1000, 32, 28, TOP_NODE_BYTES), 0);
        // Leaf size stops first: 100 -> 50 -> 25 with leaf size 30 splits twice.
        assert_eq!(top_levels(100, 1, 30, 1), 2);
        assert_eq!(top_levels(1 << 20, 1, 1, 1), MAX_TOP_LEVELS);
        assert_eq!(top_levels(10, 1, 10, 1), 0);
    }

    #[test]
    fn forest_rejects_rows_beyond_i32_ids() {
        let n = i32::MAX as usize + 1;
        assert_eq!(
            build_forest(&[], n, 1, 0, 1, 1),
            Err(WorkspaceError::Capacity {
                quantity: "forest rows",
                value: n,
                limit: i32::MAX as usize,
            })
        );
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
            assert_eq!(members, (0..19).collect::<Vec<i32>>());
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
        // Leaf limit 28 gives 2048 leaves per tree (50000 / 2^11 = 24.4) and two
        // row-major levels: membership 3_200_000 + leaves 32_769*20 + 17*8 =
        // 655_516 + row-major 50_000*16*10 + 16*2*32*4 = 8_004_096 + features
        // 12_800_000 + scratch (1_200_000 + 128)*16 = 19_202_048 + heaps
        // 50_000*14*28 = 19_600_000.
        assert_eq!(plan.host_peak_bytes, 63_461_660);
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
    fn workspace_plan_charges_row_major_levels_at_acceptance_shape() {
        let plan = plan_workspace(1_500_000, 18, MAX_CANDIDATES, 256, 64).unwrap();
        assert_eq!(plan.forest_leaf_count, 65_536 * 64);
        // membership 384_000_000 + leaves 4_194_305*20 + 65*8 = 83_886_620
        // + row-major 1.5M*64*10 + 64*512*256*4 = 993_554_432
        // + features 3_072_000_000 + scratch (36_000_000 + 1024)*64 = 2_304_065_536
        // + heaps 1.5M*18*28 = 756_000_000.
        assert_eq!(plan.host_peak_bytes, 7_593_506_588);
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
