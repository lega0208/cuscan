//! S8: single-linkage tree from a sorted minimum spanning tree (`mst_to_linkage_tree`).

use crate::error::{HdbscanError, Result};

/// Finalized MST edge: f64 Euclidean mutual-reachability weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SortedMstEdge {
    pub src: u32,
    pub dst: u32,
    pub weight: f64,
}

/// One single-linkage merge; node ids `>= n` name earlier merge rows (`n + row`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkageRow {
    /// Larger component id.
    pub left: usize,
    /// Smaller component id.
    pub right: usize,
    pub distance: f64,
    pub size: usize,
}

/// Sort edges by `(weight, src, dst)` like upstream's stable `np.lexsort((dst, src, weight))`.
pub fn sort_mst(edges: &mut [SortedMstEdge]) {
    for edge in edges.iter_mut() {
        if edge.weight == 0.0 {
            edge.weight = 0.0;
        }
    }
    edges.sort_by(|a, b| {
        a.weight
            .total_cmp(&b.weight)
            .then(a.src.cmp(&b.src))
            .then(a.dst.cmp(&b.dst))
    });
}

/// Build the single-linkage tree of `edges.len() + 1` points from an already sorted MST.
///
/// Rejects NaN weights, out-of-range endpoints and edges that close a cycle.
pub fn mst_to_linkage(edges: &[SortedMstEdge]) -> Result<Vec<LinkageRow>> {
    let n = edges
        .len()
        .checked_add(1)
        .ok_or(HdbscanError::Overflow("MST point count"))?;
    let slots = n
        .checked_mul(2)
        .ok_or(HdbscanError::Overflow("linkage node count"))?
        - 1;
    let mut parent: Vec<usize> = (0..slots).collect();
    let mut size = vec![0usize; slots];
    size[..n].fill(1);

    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            let p = parent[x];
            parent[x] = parent[p];
            x = p;
        }
        x
    }

    let mut rows = Vec::with_capacity(edges.len());
    for (index, edge) in edges.iter().enumerate() {
        let (src, dst) = (edge.src as usize, edge.dst as usize);
        if src >= n || dst >= n {
            return Err(HdbscanError::InvalidInput(format!(
                "MST edge {index} endpoint is out of range for {n} points"
            )));
        }
        if edge.weight.is_nan() || edge.weight < 0.0 {
            return Err(HdbscanError::InvalidInput(format!(
                "MST edge {index} weight must be nonnegative and not NaN"
            )));
        }
        let left = find(&mut parent, src);
        let right = find(&mut parent, dst);
        if left == right {
            return Err(HdbscanError::InvalidInput(format!(
                "MST edge {index} closes a cycle"
            )));
        }
        let (big, small) = if left > right {
            (left, right)
        } else {
            (right, left)
        };
        let next = n + index;
        size[next] = size[left] + size[right];
        parent[left] = next;
        parent[right] = next;
        rows.push(LinkageRow {
            left: big,
            right: small,
            distance: edge.weight,
            size: size[next],
        });
    }
    Ok(rows)
}
