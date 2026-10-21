//! Independent brute-force host reference for exact HDBSCAN MST construction.
//!
//! This module is exposed only by the crate's `test-support` feature.

use crate::dsu::RankDsu;
use crate::error::{HdbscanError, Result};
use crate::finite::validate_dims;
use crate::linkage::{SortedMstEdge, sort_mst};

/// Exact host kNN information used by S5 initialization and oracle comparisons.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactNeighbors {
    pub core_sq: Vec<f32>,
    pub neighbors: Vec<i32>,
    pub knn_sq: Vec<f32>,
    pub k: usize,
}

/// Compute exact squared Euclidean kNN rows, excluding self, and core radii.
/// Neighbor rows are ordered by `(squared distance, point index)`.
pub fn exact_neighbors(
    data: &[f32],
    n: usize,
    d: usize,
    min_samples: usize,
) -> Result<ExactNeighbors> {
    validate_data(data, n, d)?;
    if min_samples == 0 {
        return Err(HdbscanError::InvalidInput(
            "min_samples must be positive".into(),
        ));
    }
    let k = min_samples.min(n.saturating_sub(1));
    let mut core_sq = vec![f32::INFINITY; n];
    let mut neighbors = vec![
        -1;
        n.checked_mul(k)
            .ok_or(HdbscanError::Overflow("kNN result size"))?
    ];
    let mut knn_sq = vec![f32::INFINITY; neighbors.len()];
    for i in 0..n {
        let mut row = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if i != j {
                row.push((j, squared_distance(data, d, i, j)));
            }
        }
        row.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        for (column, &(j, distance)) in row.iter().take(k).enumerate() {
            neighbors[i * k + column] = j as i32;
            knn_sq[i * k + column] = distance;
        }
        core_sq[i] = if min_samples == 1 {
            0.0
        } else if n <= min_samples {
            f32::INFINITY
        } else {
            row[min_samples - 1].1
        };
    }
    Ok(ExactNeighbors {
        core_sq,
        neighbors,
        knn_sq,
        k,
    })
}

/// Construct an exact MST by all-pairs mutual-reachability Borůvka rounds.
pub fn brute_force_mst(
    data: &[f32],
    n: usize,
    d: usize,
    min_samples: usize,
) -> Result<Vec<SortedMstEdge>> {
    validate_data(data, n, d)?;
    if min_samples == 0 {
        return Err(HdbscanError::InvalidInput(
            "min_samples must be positive".into(),
        ));
    }
    if n < 2 {
        return Ok(Vec::new());
    }
    let exact = exact_neighbors(data, n, d, min_samples)?;
    let mut dsu = RankDsu::new(n);
    let mut edges = Vec::with_capacity(n - 1);
    let mut components = n;
    while components > 1 {
        let mut best: Vec<Option<(usize, usize, f32)>> = vec![None; n];
        for i in 0..n {
            let root_i = dsu.find(i);
            let mut point_best: Option<(usize, f32)> = None;
            for j in 0..n {
                if dsu.find(j) == root_i {
                    continue;
                }
                let distance = squared_distance(data, d, i, j);
                let weight = exact.core_sq[i].max(exact.core_sq[j]).max(distance);
                let replace = point_best.is_none_or(|(current_j, current_weight)| {
                    weight.total_cmp(&current_weight).is_lt()
                        || (weight.total_cmp(&current_weight).is_eq() && j < current_j)
                });
                if replace {
                    point_best = Some((j, weight));
                }
            }
            if let Some((j, weight)) = point_best
                && best[root_i].is_none_or(|(_, _, current_weight)| weight < current_weight)
            {
                best[root_i] = Some((i, j, weight));
            }
        }
        let mut added = 0;
        // Visiting roots in ascending point order is deterministic; candidate orientation is
        // always (source point, selected destination), independent of root identity.
        for (root, &candidate) in best.iter().enumerate() {
            if dsu.find(root) != root {
                continue;
            }
            if let Some((src, dst, weight_sq)) = candidate {
                let a = dsu.find(src);
                let b = dsu.find(dst);
                if a != b {
                    edges.push((src, dst, weight_sq));
                    dsu.union_roots(a, b);
                    added += 1;
                }
            }
        }
        if added == 0 {
            return Err(HdbscanError::Numerical(
                "brute-force Boruvka round made no progress".into(),
            ));
        }
        components -= added;
    }
    if edges.len() != n - 1 {
        return Err(HdbscanError::Numerical(format!(
            "brute-force MST has {} edges for {n} points",
            edges.len()
        )));
    }
    let mut check = RankDsu::new(n);
    let mut output = Vec::with_capacity(edges.len());
    for (src, dst, weight_sq) in edges {
        let a = check.find(src);
        let b = check.find(dst);
        if a == b {
            return Err(HdbscanError::Numerical(
                "brute-force MST contains a cycle".into(),
            ));
        }
        check.union_roots(a, b);
        output.push(SortedMstEdge {
            src: src as u32,
            dst: dst as u32,
            weight: (weight_sq as f64).sqrt(),
        });
    }
    if output.len() != n - 1 {
        return Err(HdbscanError::Numerical(
            "brute-force MST failed the acyclicity edge-count check".into(),
        ));
    }
    sort_mst(&mut output);
    Ok(output)
}

fn validate_data(data: &[f32], n: usize, d: usize) -> Result<()> {
    validate_dims(n, d)?;
    let expected = n
        .checked_mul(d)
        .ok_or(HdbscanError::Overflow("input element count"))?;
    if data.len() != expected {
        return Err(HdbscanError::InvalidInput(
            "flat data length must equal n * d".into(),
        ));
    }
    if data.iter().any(|value| !value.is_finite()) {
        return Err(HdbscanError::InvalidInput(
            "brute-force MST oracle requires finite input values".into(),
        ));
    }
    Ok(())
}

fn squared_distance(data: &[f32], d: usize, i: usize, j: usize) -> f32 {
    let mut sum = 0.0f32;
    for feature in 0..d {
        let delta = data[i * d + feature] - data[j * d + feature];
        sum += delta * delta;
    }
    sum
}
