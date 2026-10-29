//! Deterministic Kruskal over a validated row-major kNN graph.

use crate::dsu::RankDsu;
use crate::error::{HdbscanError, Result};
use crate::linkage::{SortedMstEdge, sort_mst};

/// Row-major neighbor ids and squared distances. `-1` denotes trailing padding.
#[derive(Debug, Clone, Copy)]
pub struct KnnGraph<'a> {
    pub ids: &'a [i32],
    pub dist_sq: &'a [f32],
    pub rows: usize,
    pub k: usize,
}

/// MST and per-row core radii produced from a kNN graph.
#[derive(Debug, Clone, PartialEq)]
pub struct KnnMst {
    pub edges: Vec<SortedMstEdge>,
    pub core: Vec<f64>,
    pub bridges: usize,
}

fn checked_buffer_bytes(elements: usize, element_size: usize) -> Result<()> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or(HdbscanError::Overflow("kNN graph allocation"))?;
    if bytes > isize::MAX as usize {
        return Err(HdbscanError::Overflow("kNN graph allocation"));
    }
    Ok(())
}

fn validate(graph: KnnGraph<'_>) -> Result<usize> {
    if graph.rows > i32::MAX as usize {
        return Err(HdbscanError::InvalidInput(format!(
            "kNN row count {} exceeds supported maximum {}",
            graph.rows,
            i32::MAX
        )));
    }
    let slots = graph
        .rows
        .checked_mul(graph.k)
        .ok_or(HdbscanError::Overflow("kNN graph allocation"))?;
    for size in [
        (graph.rows, std::mem::size_of::<usize>()),
        (graph.rows, std::mem::size_of::<f64>()),
        (slots, std::mem::size_of::<SortedMstEdge>()),
        (slots, std::mem::size_of::<(f64, usize, usize, usize)>()),
        (
            graph.rows.saturating_sub(1),
            std::mem::size_of::<SortedMstEdge>(),
        ),
        (
            graph.rows,
            std::mem::size_of::<usize>() + std::mem::size_of::<u32>(),
        ),
    ] {
        checked_buffer_bytes(size.0, size.1)?;
    }
    if graph.ids.len() != slots || graph.dist_sq.len() != slots {
        return Err(HdbscanError::InvalidInput(
            "kNN graph buffers must have rows * k elements".into(),
        ));
    }
    let mut seen = vec![usize::MAX; graph.rows];
    for row in 0..graph.rows {
        let mut padding = false;
        let mut previous = None;
        for column in 0..graph.k {
            let index = row * graph.k + column;
            let id = graph.ids[index];
            let distance = graph.dist_sq[index];
            if distance.is_nan() || distance < 0.0 {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN distance at slot {index} must be nonnegative and not NaN"
                )));
            }
            if id == -1 {
                padding = true;
                continue;
            }
            if id < -1 {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN id at slot {index} is below -1"
                )));
            }
            if padding {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN row {row} has a valid id after padding"
                )));
            }
            if id as usize >= graph.rows {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN id at slot {index} is out of range"
                )));
            }
            if id as usize == row {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN row {row} contains itself"
                )));
            }
            if seen[id as usize] == row {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN row {row} contains duplicate id {id}"
                )));
            }
            seen[id as usize] = row;
            if previous.is_some_and(|value| distance < value) {
                return Err(HdbscanError::InvalidInput(format!(
                    "kNN distances decrease in row {row}"
                )));
            }
            previous = Some(distance);
        }
    }
    Ok(slots)
}

/// Join a forest over every point in `0..rows` with deterministic `+inf` edges.
///
/// `dsu.len()` must equal `rows`; `edges` must contain an acyclic forest over those points.
/// Appends one edge per remaining component boundary and returns that count.
pub(crate) fn bridge_forest_with_inf(
    dsu: &mut RankDsu,
    edges: &mut Vec<SortedMstEdge>,
    rows: usize,
) -> usize {
    assert_eq!(dsu.len(), rows, "DSU length must match bridge point count");
    assert!(
        rows <= i32::MAX as usize,
        "bridge point ids must fit supported index range"
    );
    let checked_bytes = |count: usize, size: usize| {
        count
            .checked_mul(size)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .expect("bridge allocation size exceeds isize::MAX");
    };
    checked_bytes(rows, std::mem::size_of::<usize>());
    checked_bytes(rows.saturating_sub(1), std::mem::size_of::<SortedMstEdge>());
    let mut representatives = vec![usize::MAX; rows];
    for point in 0..rows {
        let root = dsu.find(point);
        representatives[root] = representatives[root].min(point);
    }
    let mut representatives: Vec<usize> = representatives
        .into_iter()
        .filter(|&point| point != usize::MAX)
        .collect();
    representatives.sort_unstable();
    let bridges = representatives.len().saturating_sub(1);
    for pair in representatives.windows(2) {
        edges.push(SortedMstEdge {
            src: pair[0] as u32,
            dst: pair[1] as u32,
            weight: f64::INFINITY,
        });
    }
    bridges
}

/// Validate and construct a spanning tree of the supplied kNN graph without repairing it.
pub fn knn_kruskal(graph: KnnGraph<'_>, min_samples: usize) -> Result<KnnMst> {
    let slots = validate(graph)?;
    checked_buffer_bytes(graph.rows, std::mem::size_of::<f64>())?;
    let mut core = vec![0.0; graph.rows];
    if min_samples > 1 {
        for (row, radius) in core.iter_mut().enumerate() {
            let index = (min_samples <= graph.k).then(|| row * graph.k + min_samples - 1);
            *radius = index
                .filter(|&i| graph.ids[i] != -1)
                .map_or(f64::INFINITY, |i| (graph.dist_sq[i] as f64).sqrt());
        }
    }
    let mut candidates: Vec<(f64, usize, usize, usize)> = Vec::with_capacity(slots);
    for row in 0..graph.rows {
        for column in 0..graph.k {
            let ordinal = row * graph.k + column;
            let id = graph.ids[ordinal];
            if id < 0 {
                continue;
            }
            let weight = core[row]
                .max(core[id as usize])
                .max((graph.dist_sq[ordinal] as f64).sqrt())
                + 0.0;
            if weight.is_finite() {
                candidates.push((weight, ordinal, row, id as usize));
            }
        }
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut dsu = RankDsu::new(graph.rows);
    let mut edges = Vec::with_capacity(graph.rows.saturating_sub(1));
    for (weight, _, src, dst) in candidates {
        let a = dsu.find(src);
        let b = dsu.find(dst);
        if a != b {
            dsu.union_roots(a, b);
            edges.push(SortedMstEdge {
                src: src as u32,
                dst: dst as u32,
                weight,
            });
            if edges.len() == graph.rows.saturating_sub(1) {
                break;
            }
        }
    }
    let bridges = bridge_forest_with_inf(&mut dsu, &mut edges, graph.rows);
    sort_mst(&mut edges);
    Ok(KnnMst {
        edges,
        core,
        bridges,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_malformed_rows_and_distances() {
        type InvalidGraph<'a> = (&'a [i32], &'a [f32], usize, usize, &'a str);
        let cases: &[InvalidGraph<'_>] = &[
            (
                &[1, -1, 2, 0, 2, -1, 0, 1, -1],
                &[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                3,
                3,
                "valid id after padding",
            ),
            (&[1, 0], &[f32::NAN, 0.0], 2, 1, "distance"),
            (&[1, 0], &[-1.0, 0.0], 2, 1, "distance"),
            (&[0, 0], &[0.0, 0.0], 2, 1, "self"),
            (&[2, 0], &[0.0, 0.0], 2, 1, "out of range"),
            (&[-2, 0], &[0.0, 0.0], 2, 1, "below -1"),
            (&[1, 1, 0, 0], &[0.0, 1.0, 0.0, 1.0], 2, 2, "duplicate"),
            (
                &[1, 2, 0, 2, 0, 1],
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                3,
                2,
                "decrease",
            ),
        ];
        for &(ids, dist_sq, rows, k, expected) in cases {
            match knn_kruskal(
                KnnGraph {
                    ids,
                    dist_sq,
                    rows,
                    k,
                },
                1,
            ) {
                Err(HdbscanError::InvalidInput(message)) => {
                    assert!(message.contains(expected), "{message}")
                }
                other => panic!("expected InvalidInput containing {expected:?}, got {other:?}"),
            }
        }
    }

    #[test]
    fn accepts_signed_zero_and_trailing_padding() {
        let result = knn_kruskal(
            KnnGraph {
                ids: &[1, -1],
                dist_sq: &[-0.0, 0.0],
                rows: 2,
                k: 1,
            },
            1,
        )
        .unwrap();
        assert_eq!(result.edges[0].weight.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn core_rank_above_width_cannot_leak_into_next_row() {
        let result = knn_kruskal(
            KnnGraph {
                ids: &[1, -1, 0, -1],
                dist_sq: &[1.0, 99.0, 1.0, 99.0],
                rows: 2,
                k: 2,
            },
            3,
        )
        .unwrap();
        assert!(result.core.iter().all(|core| *core == f64::INFINITY));
        assert_eq!(result.edges.len(), 1);
        assert_eq!(result.bridges, 1);
    }
}
