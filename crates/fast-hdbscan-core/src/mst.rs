//! Host-side squared mutual-reachability MST helpers (S5–S7).

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
