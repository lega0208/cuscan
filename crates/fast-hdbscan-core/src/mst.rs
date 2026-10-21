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
    let mut winners = Vec::<(u32, u32, f32)>::new();
    for i in 0..n {
        if cand_dist[i].is_nan() || cand_dist[i] < 0.0 {
            return Err(HdbscanError::Numerical(
                "Boruvka candidate weight must be nonnegative and not NaN".into(),
            ));
        }
        let j = cand_index[i];
        if j < 0 {
            continue;
        }
        if j as usize >= n {
            return Err(HdbscanError::InvalidInput(format!(
                "candidate {j} is out of range for {n} points"
            )));
        }
        match slots.entry(point_components[i]) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(winners.len());
                winners.push((i as u32, j as u32, cand_dist[i]));
            }
            std::collections::hash_map::Entry::Occupied(entry) => {
                let winner = &mut winners[*entry.get()];
                if cand_dist[i] < winner.2 {
                    *winner = (i as u32, j as u32, cand_dist[i]);
                }
            }
        }
    }
    let before = edges.len();
    for (src, dst, weight_sq) in winners {
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
