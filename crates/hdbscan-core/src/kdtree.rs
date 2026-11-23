//! Deterministic, backend-free host KD-tree construction for HDBSCAN.

use crate::error::{HdbscanError, Result};
use rayon::prelude::*;

pub const LEAF_SIZE: usize = 40;
/// Maximum number of levels accepted by fixed-stack device traversals.
pub const MAX_KD_LEVELS: usize = 32;

/// An implicit-heap KD tree. Node ranges are half-open offsets into `indices`.
/// Unused heap slots have an empty `0..0` range, a zero leaf flag, and zero bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct KdTree {
    pub indices: Vec<i32>,
    pub starts: Vec<i32>,
    pub ends: Vec<i32>,
    pub is_leaf: Vec<i32>,
    /// Flat row-major `[node][feature]` minimum bounds.
    pub bounds_min: Vec<f32>,
    /// Flat row-major `[node][feature]` maximum bounds.
    pub bounds_max: Vec<f32>,
    pub n_levels: usize,
    pub n_features: usize,
}

impl KdTree {
    /// Build a deterministic implicit heap, splitting nodes only when a child level exists.
    ///
    /// Large builds run on the current Rayon pool. Callers holding a lock that other Rayon
    /// tasks may wait on must use [`KdTree::build_sequential`] instead: a worker blocked in
    /// `rayon::join` can steal such a task and deadlock on the held lock.
    pub fn build(data: &[f32], n: usize, d: usize) -> Result<Self> {
        Self::build_with(data, n, d, true)
    }

    /// Build the same tree as [`KdTree::build`] on the calling thread, without Rayon.
    pub fn build_sequential(data: &[f32], n: usize, d: usize) -> Result<Self> {
        Self::build_with(data, n, d, false)
    }

    fn build_with(data: &[f32], n: usize, d: usize, parallel: bool) -> Result<Self> {
        if n == 0 || d == 0 {
            return Err(HdbscanError::InvalidInput(
                "KD-tree input must have nonzero rows and features".into(),
            ));
        }
        let element_count = n
            .checked_mul(d)
            .ok_or(HdbscanError::Overflow("KD-tree input element count"))?;
        if data.len() != element_count {
            return Err(HdbscanError::InvalidInput(
                "KD-tree input length does not match dimensions".into(),
            ));
        }
        if n > i32::MAX as usize {
            return Err(HdbscanError::InvalidInput(
                "KD-tree row count exceeds i32".into(),
            ));
        }
        if data.iter().any(|value| !value.is_finite()) {
            return Err(HdbscanError::InvalidInput(
                "KD-tree input values must all be finite".into(),
            ));
        }

        let quotient = n
            .saturating_sub(1)
            .checked_div(LEAF_SIZE)
            .unwrap_or(0)
            .max(1);
        let n_levels = (usize::BITS - quotient.leading_zeros()) as usize;
        if n_levels > MAX_KD_LEVELS {
            return Err(HdbscanError::Overflow("KD-tree level count"));
        }
        let heap_width = 1usize
            .checked_shl(
                u32::try_from(n_levels)
                    .map_err(|_| HdbscanError::Overflow("KD-tree level count"))?,
            )
            .ok_or(HdbscanError::Overflow("KD-tree node count"))?;
        let n_nodes = heap_width
            .checked_sub(1)
            .ok_or(HdbscanError::Overflow("KD-tree node count"))?;
        if n_nodes > i32::MAX as usize {
            return Err(HdbscanError::Overflow("KD-tree node index"));
        }
        let bounds_len = n_nodes
            .checked_mul(d)
            .ok_or(HdbscanError::Overflow("KD-tree bounds element count"))?;

        let mut tree = Self {
            indices: try_filled(n, 0i32, "KD-tree indices")?,
            starts: try_filled(n_nodes, 0i32, "KD-tree starts")?,
            ends: try_filled(n_nodes, 0i32, "KD-tree ends")?,
            is_leaf: try_filled(n_nodes, 0i32, "KD-tree leaf flags")?,
            bounds_min: try_filled(bounds_len, 0.0f32, "KD-tree minimum bounds")?,
            bounds_max: try_filled(bounds_len, 0.0f32, "KD-tree maximum bounds")?,
            n_levels,
            n_features: d,
        };
        for (index, slot) in tree.indices.iter_mut().enumerate() {
            *slot =
                i32::try_from(index).map_err(|_| HdbscanError::Overflow("KD-tree point index"))?;
        }

        let scratch_len = d
            .checked_mul(2)
            .ok_or(HdbscanError::Overflow("KD-tree split scratch"))?;
        if n_nodes > 1 {
            let ctx = BuildContext {
                data,
                d,
                n_nodes,
                scratch_len,
                parallel,
            };
            let mut bounds = try_filled(scratch_len, 0.0f32, "KD-tree split scratch")?;
            split_subtree(&ctx, 0, &mut tree.indices, &mut bounds)?;
        }

        let mut stack = Vec::new();
        stack
            .try_reserve_exact(n_levels + 1)
            .map_err(|_| HdbscanError::Overflow("KD-tree build stack allocation"))?;
        stack.push((0usize, 0usize, n));
        while let Some((node, start, end)) = stack.pop() {
            let child = node * 2 + 1;
            tree.starts[node] =
                i32::try_from(start).map_err(|_| HdbscanError::Overflow("KD-tree range start"))?;
            tree.ends[node] =
                i32::try_from(end).map_err(|_| HdbscanError::Overflow("KD-tree range end"))?;
            if child >= n_nodes || end - start < 2 {
                tree.is_leaf[node] = 1;
                continue;
            }
            let mid = start + (end - start) / 2;
            stack.push((child + 1, mid, end));
            stack.push((child, start, mid));
        }
        tree.fill_bounds(data, parallel && n >= PARALLEL_SPLIT_ROWS);
        Ok(tree)
    }

    /// Return each node's homogeneous component, or `-1` for heterogeneous/unused nodes.
    pub fn node_components(&self, point_components: &[i32]) -> Result<Vec<i32>> {
        if point_components.len() != self.indices.len() {
            return Err(HdbscanError::InvalidInput(
                "point component count does not match KD-tree rows".into(),
            ));
        }
        let mut components = try_filled(self.starts.len(), -1i32, "KD-tree node components")?;
        for node in (0..self.starts.len()).rev() {
            let start = self.starts[node];
            let end = self.ends[node];
            if start >= end {
                continue;
            }
            if self.is_leaf[node] != 0 {
                let start = usize::try_from(start)
                    .map_err(|_| HdbscanError::InvalidInput("invalid KD-tree range".into()))?;
                let end = usize::try_from(end)
                    .map_err(|_| HdbscanError::InvalidInput("invalid KD-tree range".into()))?;
                let mut homogeneous = None;
                for &point in &self.indices[start..end] {
                    let point = usize::try_from(point).map_err(|_| {
                        HdbscanError::InvalidInput("invalid KD-tree point index".into())
                    })?;
                    let label = *point_components.get(point).ok_or_else(|| {
                        HdbscanError::InvalidInput("invalid KD-tree point index".into())
                    })?;
                    if label < 0 || homogeneous.is_some_and(|first| first != label) {
                        homogeneous = Some(-1);
                        break;
                    }
                    homogeneous = Some(label);
                }
                components[node] = homogeneous.unwrap_or(-1);
            } else {
                let left = node.checked_mul(2).and_then(|value| value.checked_add(1));
                let right = left.and_then(|value| value.checked_add(1));
                if let (Some(left), Some(right)) = (left, right)
                    && right < components.len()
                    && self.starts[left] < self.ends[left]
                    && self.starts[right] < self.ends[right]
                    && components[left] >= 0
                    && components[left] == components[right]
                {
                    components[node] = components[left];
                }
            }
        }
        Ok(components)
    }

    /// Bottom-up node bounds: leaves scan their rows, internal nodes union their children.
    /// Levels run on Rayon only when `parallel` is set and the level has several nodes.
    fn fill_bounds(&mut self, data: &[f32], parallel: bool) {
        let d = self.n_features;
        for level in (0..self.n_levels).rev() {
            let first = (1usize << level) - 1;
            let count = 1usize << level;
            let (lower_min, child_min) = self.bounds_min.split_at_mut((first + count) * d);
            let (lower_max, child_max) = self.bounds_max.split_at_mut((first + count) * d);
            let (child_min, child_max) = (&*child_min, &*child_max);
            let (starts, ends, is_leaf, indices) =
                (&self.starts, &self.ends, &self.is_leaf, &self.indices);
            let node_bounds = |(offset, (min, max)): (usize, (&mut [f32], &mut [f32]))| {
                let node = first + offset;
                let (start, end) = (starts[node] as usize, ends[node] as usize);
                if start == end {
                    return;
                }
                if is_leaf[node] != 0 {
                    point_bounds(data, d, &indices[start..end], min, max);
                } else {
                    let left = (node * 2 + 1 - (first + count)) * d;
                    min.copy_from_slice(&child_min[left..left + d]);
                    max.copy_from_slice(&child_max[left..left + d]);
                    fold_bounds(&child_min[left + d..left + 2 * d], min, max);
                    fold_bounds(&child_max[left + d..left + 2 * d], min, max);
                }
            };
            let (level_min, level_max) = (&mut lower_min[first * d..], &mut lower_max[first * d..]);
            if parallel && count > 1 {
                level_min
                    .par_chunks_mut(d)
                    .zip(level_max.par_chunks_mut(d))
                    .enumerate()
                    .for_each(node_bounds);
            } else {
                level_min
                    .chunks_mut(d)
                    .zip(level_max.chunks_mut(d))
                    .enumerate()
                    .for_each(node_bounds);
            }
        }
    }
}

/// Rows at or above this count split their children in parallel.
const PARALLEL_SPLIT_ROWS: usize = 32_768;

struct BuildContext<'a> {
    data: &'a [f32],
    d: usize,
    n_nodes: usize,
    scratch_len: usize,
    parallel: bool,
}

/// Recursively median-split `indices` for `node`; disjoint subtrees run in parallel.
///
/// `bounds` is `2 * d` scratch for this node's min/max, reused by the serial recursion.
/// Every split depends only on its own point set, so the result is independent of scheduling.
fn split_subtree(
    ctx: &BuildContext<'_>,
    node: usize,
    indices: &mut [i32],
    bounds: &mut [f32],
) -> Result<()> {
    let len = indices.len();
    let child = node * 2 + 1;
    if child >= ctx.n_nodes || len < 2 {
        return Ok(());
    }
    let d = ctx.d;
    let (min, max) = bounds.split_at_mut(d);
    point_bounds(ctx.data, d, indices, min, max);
    let mut axis = 0;
    let mut best_spread = -1.0f64;
    for feature in 0..d {
        let spread = f64::from(max[feature]) - f64::from(min[feature]);
        if spread > best_spread {
            best_spread = spread;
            axis = feature;
        }
    }
    let data = ctx.data;
    let mid = len / 2;
    indices.select_nth_unstable_by(mid, |&left, &right| {
        data[left as usize * d + axis]
            .total_cmp(&data[right as usize * d + axis])
            .then(left.cmp(&right))
    });
    let (left, right) = indices.split_at_mut(mid);
    if ctx.parallel && len >= PARALLEL_SPLIT_ROWS {
        let mut right_bounds = try_filled(ctx.scratch_len, 0.0f32, "KD-tree split scratch")?;
        let (left, right) = rayon::join(
            || split_subtree(ctx, child, left, bounds),
            || split_subtree(ctx, child + 1, right, &mut right_bounds),
        );
        left.and(right)
    } else {
        split_subtree(ctx, child, left, bounds)?;
        split_subtree(ctx, child + 1, right, bounds)
    }
}

/// Monotone `i32` image of `f32::total_cmp`.
#[inline]
fn total_order_key(value: f32) -> i32 {
    let bits = value.to_bits() as i32;
    bits ^ ((((bits >> 31) as u32) >> 1) as i32)
}

#[inline]
fn fold_bounds(row: &[f32], min: &mut [f32], max: &mut [f32]) {
    for ((lo, hi), &value) in min.iter_mut().zip(max.iter_mut()).zip(row) {
        let key = total_order_key(value);
        *lo = if key < total_order_key(*lo) {
            value
        } else {
            *lo
        };
        *hi = if key > total_order_key(*hi) {
            value
        } else {
            *hi
        };
    }
}

/// Exact `total_cmp` per-feature bounds of the given points, one row-major pass.
fn point_bounds(data: &[f32], d: usize, points: &[i32], min: &mut [f32], max: &mut [f32]) {
    let first = &data[points[0] as usize * d..][..d];
    min.copy_from_slice(first);
    max.copy_from_slice(first);
    for &point in &points[1..] {
        fold_bounds(&data[point as usize * d..][..d], min, max);
    }
}

fn try_filled<T: Clone>(len: usize, value: T, subject: &'static str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| HdbscanError::Overflow(subject))?;
    values.resize(len, value);
    Ok(values)
}
