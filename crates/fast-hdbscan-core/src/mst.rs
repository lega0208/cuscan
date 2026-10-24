//! Host-side squared mutual-reachability MST helpers (S5–S7).

#[cfg(any(test, feature = "test-support"))]
use std::collections::HashMap;

use crate::dsu::RankDsu;
use crate::error::{HdbscanError, Result};
use crate::linkage::{SortedMstEdge, sort_mst};

/// An edge whose mutual-reachability weight is still squared.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MstEdge {
    pub src: u32,
    pub dst: u32,
    pub weight_sq: f32,
}

/// Stable CSR membership and component-contained tile ranges for one Boruvka round.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentReductionPlan {
    pub members: Vec<i32>,
    pub offsets: Vec<i32>,
    pub tile_starts: Vec<i32>,
    pub tile_ends: Vec<i32>,
    pub partial_offsets: Vec<i32>,
}

impl ComponentReductionPlan {
    /// Build stable component segments and tiles no wider than `tile_width`.
    pub fn build(labels: &[i32], tile_width: usize) -> Result<Self> {
        let n = labels.len();
        if n > i32::MAX as usize {
            return Err(HdbscanError::Overflow("component member index"));
        }
        if tile_width == 0 {
            return Err(HdbscanError::InvalidInput(
                "component tile width must be positive".into(),
            ));
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(n)
            .map_err(|_| HdbscanError::Overflow("component slots"))?;
        slots.resize(n, -1_i32);
        let mut counts = Vec::new();
        counts
            .try_reserve_exact(n)
            .map_err(|_| HdbscanError::Overflow("component counts"))?;
        counts.resize(n, 0_usize);
        let mut component_count = 0_usize;
        for &label in labels {
            if label < 0 || label as usize >= n {
                return Err(HdbscanError::InvalidInput(format!(
                    "component root {label} is out of range for {n} points"
                )));
            }
            let slot = &mut slots[label as usize];
            if *slot == -1 {
                *slot = i32::try_from(component_count)
                    .map_err(|_| HdbscanError::Overflow("component count"))?;
                component_count = component_count
                    .checked_add(1)
                    .ok_or(HdbscanError::Overflow("component count"))?;
            }
            counts[*slot as usize] = counts[*slot as usize]
                .checked_add(1)
                .ok_or(HdbscanError::Overflow("component size"))?;
        }

        let offset_len = component_count
            .checked_add(1)
            .ok_or(HdbscanError::Overflow("component offsets"))?;
        let mut offsets = Vec::new();
        offsets
            .try_reserve_exact(offset_len)
            .map_err(|_| HdbscanError::Overflow("component offsets"))?;
        offsets.push(0_i32);
        let mut partial_offsets = Vec::new();
        partial_offsets
            .try_reserve_exact(offset_len)
            .map_err(|_| HdbscanError::Overflow("partial offsets"))?;
        partial_offsets.push(0_i32);
        let mut partial_count = 0_usize;
        for &count in &counts[..component_count] {
            let next = (offsets.last().copied().unwrap() as usize)
                .checked_add(count)
                .ok_or(HdbscanError::Overflow("component offsets"))?;
            offsets.push(
                i32::try_from(next).map_err(|_| HdbscanError::Overflow("component offsets"))?,
            );
            partial_count = partial_count
                .checked_add(count.div_ceil(tile_width))
                .ok_or(HdbscanError::Overflow("component partial count"))?;
            partial_offsets.push(
                i32::try_from(partial_count)
                    .map_err(|_| HdbscanError::Overflow("partial offsets"))?,
            );
        }
        let mut members = Vec::new();
        members
            .try_reserve_exact(n)
            .map_err(|_| HdbscanError::Overflow("component members"))?;
        members.resize(n, 0_i32);
        let mut cursor = Vec::new();
        cursor
            .try_reserve_exact(component_count)
            .map_err(|_| HdbscanError::Overflow("component scatter cursor"))?;
        cursor.extend(offsets[..component_count].iter().map(|&x| x as usize));
        for (source, &label) in labels.iter().enumerate() {
            let slot = slots[label as usize] as usize;
            members[cursor[slot]] = i32::try_from(source)
                .map_err(|_| HdbscanError::Overflow("component member index"))?;
            cursor[slot] += 1;
        }
        let mut tile_starts = Vec::new();
        let mut tile_ends = Vec::new();
        tile_starts
            .try_reserve_exact(partial_count)
            .map_err(|_| HdbscanError::Overflow("component tile starts"))?;
        tile_ends
            .try_reserve_exact(partial_count)
            .map_err(|_| HdbscanError::Overflow("component tile ends"))?;
        for slot in 0..component_count {
            let mut start = offsets[slot] as usize;
            let end = offsets[slot + 1] as usize;
            while start < end {
                let tile_end = start.saturating_add(tile_width).min(end);
                tile_starts.push(
                    i32::try_from(start)
                        .map_err(|_| HdbscanError::Overflow("component tile start"))?,
                );
                tile_ends.push(
                    i32::try_from(tile_end)
                        .map_err(|_| HdbscanError::Overflow("component tile end"))?,
                );
                start = tile_end;
            }
        }
        Ok(Self {
            members,
            offsets,
            tile_starts,
            tile_ends,
            partial_offsets,
        })
    }
}

/// Merge compact component winners produced by the device reduction.
/// `metadata` is row-major `[component_count, 5]` with columns
/// `(source, destination, bad_row, error_kind, bad_destination)`.
pub fn merge_component_winners(
    dsu: &mut RankDsu,
    plan: &ComponentReductionPlan,
    weights: &[f32],
    metadata: &[i32],
    edges: &mut Vec<MstEdge>,
) -> Result<usize> {
    let n = plan.members.len();
    let c = plan.offsets.len().checked_sub(1).ok_or_else(|| {
        HdbscanError::InvalidInput("component offsets must contain a sentinel".into())
    })?;
    let expected_metadata = c
        .checked_mul(5)
        .ok_or(HdbscanError::Overflow("component winner metadata"))?;
    if dsu.len() != n || weights.len() != c || metadata.len() != expected_metadata {
        return Err(HdbscanError::InvalidInput(
            "component winner arrays, plan, and disjoint set have inconsistent lengths".into(),
        ));
    }
    validate_reduction_plan(plan)?;

    let mut earliest_error: Option<(i32, i32, i32)> = None;
    for slot in 0..c {
        let row = &metadata[slot * 5..slot * 5 + 5];
        let (source, destination, bad_row, kind, bad_destination) =
            (row[0], row[1], row[2], row[3], row[4]);
        if (source == -1) != (destination == -1) {
            return Err(HdbscanError::InvalidInput(
                "component winner must have both endpoints present or absent".into(),
            ));
        }
        if source == -1 {
            if weights[slot].to_bits() != f32::INFINITY.to_bits() {
                return Err(HdbscanError::InvalidInput(
                    "absent component winner must have +inf weight".into(),
                ));
            }
        } else {
            if source < 0 || source as usize >= n || destination < 0 || destination as usize >= n {
                return Err(HdbscanError::InvalidInput(
                    "component winner endpoint is out of range".into(),
                ));
            }
            if !plan.members[plan.offsets[slot] as usize..plan.offsets[slot + 1] as usize]
                .contains(&source)
            {
                return Err(HdbscanError::InvalidInput(
                    "component winner source is outside its component".into(),
                ));
            }
        }
        if kind != 0 {
            if !(kind == 1 || kind == 2) || bad_row < 0 || bad_row as usize >= n {
                return Err(HdbscanError::InvalidInput(
                    "malformed component winner error metadata".into(),
                ));
            }
            if !plan.members[plan.offsets[slot] as usize..plan.offsets[slot + 1] as usize]
                .contains(&bad_row)
            {
                return Err(HdbscanError::InvalidInput(
                    "component error row is outside its component".into(),
                ));
            }
            if kind == 2 && (bad_destination < 0 || ((bad_destination as usize) < n)) {
                return Err(HdbscanError::InvalidInput(
                    "malformed component winner bad destination".into(),
                ));
            }
            if kind == 1 && bad_destination != -1 {
                return Err(HdbscanError::InvalidInput(
                    "numerical component error must not carry a bad destination".into(),
                ));
            }
            if earliest_error
                .is_none_or(|(best, _, _)| bad_row < best || (bad_row == best && kind == 1))
            {
                earliest_error = Some((bad_row, kind, bad_destination));
            }
            continue;
        }
        if bad_row != n as i32 || bad_destination != -1 {
            return Err(HdbscanError::InvalidInput(
                "valid component winner has malformed error fields".into(),
            ));
        }
    }
    if let Some((_, kind, destination)) = earliest_error {
        return if kind == 1 {
            Err(HdbscanError::Numerical(
                "Boruvka candidate weight must be nonnegative and not NaN".into(),
            ))
        } else {
            Err(HdbscanError::InvalidInput(format!(
                "candidate {destination} is out of range for {n} points"
            )))
        };
    }

    for slot in 0..c {
        if metadata[slot * 5] >= 0 && (weights[slot].is_nan() || weights[slot] < 0.0) {
            return Err(HdbscanError::Numerical(
                "Boruvka candidate weight must be nonnegative and not NaN".into(),
            ));
        }
    }

    let before = edges.len();
    for slot in 0..c {
        let row = &metadata[slot * 5..slot * 5 + 5];
        if row[0] < 0 {
            continue;
        }
        let src = row[0] as usize;
        let dst = row[1] as usize;
        let a = dsu.find(src);
        let b = dsu.find(dst);
        if a != b {
            edges.push(MstEdge {
                src: src as u32,
                dst: dst as u32,
                weight_sq: weights[slot],
            });
            dsu.union_roots(a, b);
        }
    }
    let added = edges.len() - before;
    if added == 0 {
        return Err(HdbscanError::Numerical(
            "Boruvka round made no progress".into(),
        ));
    }
    Ok(added)
}

fn validate_reduction_plan(plan: &ComponentReductionPlan) -> Result<()> {
    let n = plan.members.len();
    if n > i32::MAX as usize {
        return Err(HdbscanError::Overflow("component member index"));
    }
    let c = plan.offsets.len() - 1;
    let expected_offset_len = c
        .checked_add(1)
        .ok_or(HdbscanError::Overflow("component offsets"))?;
    if plan.offsets.first() != Some(&0)
        || plan.offsets.last().copied() != Some(n as i32)
        || plan.offsets.len() != expected_offset_len
        || plan.partial_offsets.len() != expected_offset_len
        || plan.partial_offsets.first() != Some(&0)
        || plan.tile_starts.len() != plan.tile_ends.len()
        || plan.partial_offsets.last().copied() != Some(plan.tile_starts.len() as i32)
    {
        return Err(HdbscanError::InvalidInput(
            "malformed component reduction plan".into(),
        ));
    }
    for slot in 0..c {
        let start = plan.offsets[slot];
        let end = plan.offsets[slot + 1];
        let partial_start = plan.partial_offsets[slot];
        let partial_end = plan.partial_offsets[slot + 1];
        if start < 0
            || end <= start
            || end as usize > n
            || partial_start < 0
            || partial_end <= partial_start
            || partial_end as usize > plan.tile_starts.len()
        {
            return Err(HdbscanError::InvalidInput(
                "malformed component reduction segment".into(),
            ));
        }
        let members = &plan.members[start as usize..end as usize];
        if members.windows(2).any(|pair| pair[0] >= pair[1])
            || (slot > 0 && plan.members[plan.offsets[slot - 1] as usize] >= members[0])
        {
            return Err(HdbscanError::InvalidInput(
                "component members are not in stable first-appearance order".into(),
            ));
        }
        let mut cursor = start;
        for tile in partial_start as usize..partial_end as usize {
            let a = plan.tile_starts[tile];
            let b = plan.tile_ends[tile];
            if a != cursor || b <= a || b > end {
                return Err(HdbscanError::InvalidInput(
                    "malformed component tile range".into(),
                ));
            }
            cursor = b;
        }
        if cursor != end {
            return Err(HdbscanError::InvalidInput(
                "component tiles do not cover segment".into(),
            ));
        }
    }
    let mut seen = Vec::new();
    seen.try_reserve_exact(n)
        .map_err(|_| HdbscanError::Overflow("component plan validation"))?;
    seen.resize(n, false);
    for &member in &plan.members {
        if member < 0 || member as usize >= n || std::mem::replace(&mut seen[member as usize], true)
        {
            return Err(HdbscanError::InvalidInput(
                "component members are not a permutation".into(),
            ));
        }
    }
    Ok(())
}

/// Initialize a forest from a flat row-major exact kNN result.
///
/// For each point, the first non-padding neighbor whose core distance is no greater
/// than the source core distance is selected. Edges are oriented from the query point.
pub fn init_from_knn(
    core_sq: &[f32],
    neighbors: &[i32],
    knn_sq: &[f32],
    k: usize,
    dsu: &mut RankDsu,
) -> Result<Vec<MstEdge>> {
    let n = core_sq.len();
    let expected = n
        .checked_mul(k)
        .ok_or(HdbscanError::Overflow("kNN result size"))?;
    if neighbors.len() != expected || knn_sq.len() != expected {
        return Err(HdbscanError::InvalidInput(
            "kNN arrays must have n * k entries".into(),
        ));
    }
    if dsu.len() != n {
        return Err(HdbscanError::InvalidInput(
            "disjoint set size must equal the number of points".into(),
        ));
    }
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let core_i = core_sq[i];
        let row = i * k;
        for j in 0..k {
            let dst = neighbors[row + j];
            if dst < 0 {
                continue;
            }
            let dst = dst as usize;
            if dst >= n {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN neighbor {dst} is out of range for {n} points"
                )));
            }
            if core_i >= core_sq[dst] {
                let a = dsu.find(i);
                let b = dsu.find(dst);
                if a != b {
                    edges.push(MstEdge {
                        src: i as u32,
                        dst: dst as u32,
                        weight_sq: core_i.max(knn_sq[row + j]),
                    });
                    dsu.union_roots(a, b);
                }
                break;
            }
        }
    }
    Ok(edges)
}

/// Select and merge one candidate edge per component, preserving first appearance order.
///
/// Candidates are reduced with strict `<`, so the earliest point wins distance ties.
#[cfg(any(test, feature = "test-support"))]
pub fn merge_components(
    dsu: &mut RankDsu,
    point_components: &[i32],
    cand_dist: &[f32],
    cand_index: &[i32],
    edges: &mut Vec<MstEdge>,
) -> Result<usize> {
    let n = point_components.len();
    if cand_dist.len() != n || cand_index.len() != n {
        return Err(HdbscanError::InvalidInput(
            "component and candidate arrays must have equal lengths".into(),
        ));
    }
    if dsu.len() != n {
        return Err(HdbscanError::InvalidInput(
            "disjoint set size must equal the number of points".into(),
        ));
    }
    let mut slots = HashMap::<i32, usize>::new();
    let mut winners = Vec::<(u32, u32, f32, bool)>::new();
    for i in 0..n {
        if cand_dist[i].is_nan() || cand_dist[i] < 0.0 {
            return Err(HdbscanError::Numerical(
                "Boruvka candidate weight must be nonnegative and not NaN".into(),
            ));
        }
        let j = cand_index[i];
        if j >= 0 && j as usize >= n {
            return Err(HdbscanError::InvalidInput(format!(
                "candidate {j} is out of range for {n} points"
            )));
        }
        let slot = match slots.entry(point_components[i]) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                let slot = winners.len();
                entry.insert(slot);
                winners.push((i as u32, u32::MAX, f32::INFINITY, false));
                slot
            }
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
        };
        if j < 0 {
            continue;
        }
        let winner = &mut winners[slot];
        if !winner.3 || cand_dist[i] < winner.2 {
            *winner = (i as u32, j as u32, cand_dist[i], true);
        }
    }
    let before = edges.len();
    for (src, dst, weight_sq, has_candidate) in winners {
        if !has_candidate {
            continue;
        }
        let a = dsu.find(src as usize);
        let b = dsu.find(dst as usize);
        if a != b {
            edges.push(MstEdge {
                src,
                dst,
                weight_sq,
            });
            dsu.union_roots(a, b);
        }
    }
    let added = edges.len() - before;
    if added == 0 {
        return Err(HdbscanError::Numerical(
            "Boruvka round made no progress".into(),
        ));
    }
    Ok(added)
}

/// Convert squared f32 weights to Euclidean f64 weights and sort the result.
pub fn finalize_mst(edges: Vec<MstEdge>) -> Vec<SortedMstEdge> {
    let mut output: Vec<_> = edges
        .into_iter()
        .map(|edge| SortedMstEdge {
            src: edge.src,
            dst: edge.dst,
            weight: (edge.weight_sq as f64).sqrt(),
        })
        .collect();
    sort_mst(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruned_lowest_row_preserves_component_slot_order() {
        let components = [10, 20, 10, 20];
        let distances = [f32::INFINITY, 1.0, 1.0, 1.0];
        let indexes = [-1, 2, 3, 2];
        let mut full_dsu = RankDsu::new(4);
        let mut pruned_dsu = RankDsu::new(4);
        let mut full_edges = Vec::new();
        let mut pruned_edges = Vec::new();

        assert_eq!(
            merge_components(
                &mut full_dsu,
                &components,
                &distances,
                &[3, 2, 3, 2],
                &mut full_edges
            ),
            Ok(2)
        );
        assert_eq!(
            merge_components(
                &mut pruned_dsu,
                &components,
                &distances,
                &indexes,
                &mut pruned_edges
            ),
            Ok(2)
        );
        assert_eq!(pruned_edges, full_edges);
        assert_eq!(
            pruned_edges[0],
            MstEdge {
                src: 2,
                dst: 3,
                weight_sq: 1.0
            }
        );
        assert_eq!(
            pruned_edges[1],
            MstEdge {
                src: 1,
                dst: 2,
                weight_sq: 1.0
            }
        );
        assert_eq!(pruned_dsu.find(0), full_dsu.find(0));
        assert_eq!(pruned_dsu.find(1), full_dsu.find(1));
        assert_eq!(pruned_dsu.find(2), full_dsu.find(2));
        assert_eq!(pruned_dsu.find(3), full_dsu.find(3));
    }

    #[test]
    fn all_placeholder_component_is_skipped_while_another_progresses() {
        let mut dsu = RankDsu::new(3);
        let mut edges = Vec::new();
        assert_eq!(
            merge_components(
                &mut dsu,
                &[7, 7, 9],
                &[1.0, 1.0, 2.0],
                &[-1, -1, 1],
                &mut edges
            ),
            Ok(1)
        );
        assert_eq!(
            edges,
            vec![MstEdge {
                src: 2,
                dst: 1,
                weight_sq: 2.0
            }]
        );
        assert_eq!(dsu.find(2), dsu.find(1));
        assert_ne!(dsu.find(0), dsu.find(1));
    }

    #[test]
    fn all_placeholders_keep_no_progress_error() {
        let mut dsu = RankDsu::new(2);
        let mut edges = Vec::new();
        assert_eq!(
            merge_components(
                &mut dsu,
                &[0, 1],
                &[f32::INFINITY; 2],
                &[-1, -1],
                &mut edges
            ),
            Err(HdbscanError::Numerical(
                "Boruvka round made no progress".into()
            ))
        );
        assert!(edges.is_empty());
    }

    #[test]
    fn real_infinity_replaces_placeholder_and_unions() {
        let mut dsu = RankDsu::new(2);
        let mut edges = Vec::new();
        assert_eq!(
            merge_components(&mut dsu, &[4, 4], &[f32::INFINITY; 2], &[-1, 0], &mut edges),
            Ok(1)
        );
        assert_eq!(
            edges,
            vec![MstEdge {
                src: 1,
                dst: 0,
                weight_sq: f32::INFINITY
            }]
        );
        assert_eq!(dsu.find(0), dsu.find(1));
    }

    #[test]
    fn invalid_candidates_remain_errors_even_for_padding_rows() {
        for weight in [f32::NAN, -1.0] {
            let mut dsu = RankDsu::new(1);
            assert!(matches!(
                merge_components(&mut dsu, &[0], &[weight], &[-1], &mut Vec::new()),
                Err(HdbscanError::Numerical(_))
            ));
        }
        let mut dsu = RankDsu::new(1);
        assert!(matches!(
            merge_components(&mut dsu, &[0], &[0.0], &[1], &mut Vec::new()),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
}
