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

/// Independent neighbor graph and reference MST for graph-route tests.
#[derive(Debug, Default)]
pub struct KnnKruskalOracle {
    pub ids: Vec<i32>,
    pub dist_sq: Vec<f32>,
    pub core: Vec<f64>,
    pub edges: Vec<SortedMstEdge>,
    pub bridges: usize,
}

/// Independently construct exact non-self neighbors and a Kruskal reference tree.
pub fn knn_kruskal_oracle(
    data: &[f32],
    rows: usize,
    d: usize,
    min_samples: usize,
    k: usize,
) -> Result<KnnKruskalOracle> {
    if rows == 0 && d > 0 && data.is_empty() {
        return Ok(KnnKruskalOracle::default());
    }
    validate_data(data, rows, d)?;
    if rows > i32::MAX as usize {
        return Err(HdbscanError::Overflow("kNN point index"));
    }
    let slots = rows
        .checked_mul(k)
        .ok_or(HdbscanError::Overflow("kNN graph allocation"))?;
    for (elements, size) in [
        (slots, std::mem::size_of::<i32>()),
        (slots, std::mem::size_of::<f32>()),
        (rows, std::mem::size_of::<f64>()),
        (slots, std::mem::size_of::<(f64, usize, usize, usize)>()),
        (rows.saturating_sub(1), std::mem::size_of::<SortedMstEdge>()),
        (
            rows,
            std::mem::size_of::<usize>() + std::mem::size_of::<u32>(),
        ),
    ] {
        let bytes = elements
            .checked_mul(size)
            .ok_or(HdbscanError::Overflow("kNN graph allocation"))?;
        if bytes > isize::MAX as usize {
            return Err(HdbscanError::Overflow("kNN graph allocation"));
        }
    }
    let neighbor_bytes = rows
        .saturating_sub(1)
        .checked_mul(std::mem::size_of::<(f32, usize)>())
        .ok_or(HdbscanError::Overflow("kNN oracle neighbor storage"))?;
    if neighbor_bytes > isize::MAX as usize {
        return Err(HdbscanError::Overflow("kNN oracle neighbor storage"));
    }
    let mut ids = vec![-1; slots];
    let mut distances = vec![f32::INFINITY; slots];
    for row in 0..rows {
        let mut neighbors = Vec::with_capacity(rows.saturating_sub(1));
        for other in 0..rows {
            if other != row {
                neighbors.push((squared_distance(data, d, row, other), other));
            }
        }
        neighbors.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (column, &(distance, other)) in neighbors.iter().take(k).enumerate() {
            ids[row * k + column] = other as i32;
            distances[row * k + column] = distance;
        }
    }
    let mut core = vec![0.0; rows];
    if min_samples > 1 {
        for row in 0..rows {
            let column = min_samples - 1;
            if column < k && ids[row * k + column] >= 0 {
                core[row] = (distances[row * k + column] as f64).sqrt();
            } else {
                core[row] = f64::INFINITY;
            }
        }
    }
    let mut candidates = Vec::new();
    for row in 0..rows {
        for column in 0..k {
            let ordinal = row * k + column;
            let other = ids[ordinal];
            if other < 0 {
                continue;
            }
            let weight = core[row]
                .max(core[other as usize])
                .max((distances[ordinal] as f64).sqrt())
                + 0.0;
            if weight.is_finite() {
                candidates.push((weight, ordinal, row, other as usize));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut parent: Vec<usize> = (0..rows).collect();
    let mut ranks = vec![0u32; rows];
    fn root(parent: &mut [usize], mut value: usize) -> usize {
        while parent[value] != value {
            parent[value] = parent[parent[value]];
            value = parent[value];
        }
        value
    }
    let mut edges = Vec::new();
    for (weight, _, src, dst) in candidates {
        let (a, b) = (root(&mut parent, src), root(&mut parent, dst));
        if a != b {
            let (mut winner, mut loser) = (a, b);
            if ranks[winner] < ranks[loser] {
                std::mem::swap(&mut winner, &mut loser);
            }
            parent[loser] = winner;
            if ranks[winner] == ranks[loser] {
                ranks[winner] += 1;
            }
            edges.push(SortedMstEdge {
                src: src as u32,
                dst: dst as u32,
                weight,
            });
        }
    }
    let mut minima = vec![usize::MAX; rows];
    for point in 0..rows {
        let r = root(&mut parent, point);
        minima[r] = minima[r].min(point);
    }
    let mut reps: Vec<usize> = minima.into_iter().filter(|&x| x != usize::MAX).collect();
    reps.sort_unstable();
    let bridges = reps.len().saturating_sub(1);
    for pair in reps.windows(2) {
        edges.push(SortedMstEdge {
            src: pair[0] as u32,
            dst: pair[1] as u32,
            weight: f64::INFINITY,
        });
    }
    edges.sort_by(|a, b| {
        a.weight
            .total_cmp(&b.weight)
            .then(a.src.cmp(&b.src))
            .then(a.dst.cmp(&b.dst))
    });
    Ok(KnnKruskalOracle {
        ids,
        dist_sq: distances,
        core,
        edges,
        bridges,
    })
}
