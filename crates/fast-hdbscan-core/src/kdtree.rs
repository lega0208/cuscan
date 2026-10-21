//! Deterministic, backend-free host KD-tree construction for HDBSCAN.

use crate::error::{HdbscanError, Result};

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
    pub fn build(data: &[f32], n: usize, d: usize) -> Result<Self> {
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

        let mut stack = Vec::new();
        stack
            .try_reserve_exact(n_levels)
            .map_err(|_| HdbscanError::Overflow("KD-tree build stack allocation"))?;
        stack.push((0usize, 0usize, n));
        while let Some((node, start, end)) = stack.pop() {
            let child = node
                .checked_mul(2)
                .and_then(|value| value.checked_add(1))
                .ok_or(HdbscanError::Overflow("KD-tree child index"))?;
            tree.starts[node] =
                i32::try_from(start).map_err(|_| HdbscanError::Overflow("KD-tree range start"))?;
            tree.ends[node] =
                i32::try_from(end).map_err(|_| HdbscanError::Overflow("KD-tree range end"))?;
            tree.set_bounds(data, d, node, start, end);
            if child >= n_nodes || end - start < 2 {
                tree.is_leaf[node] = 1;
                continue;
            }

            let axis = tree.widest_axis(data, d, start, end);
            let mid = start + (end - start) / 2;
            tree.indices[start..end].select_nth_unstable_by(mid - start, |&left, &right| {
                let left_offset = left as usize * d + axis;
                let right_offset = right as usize * d + axis;
                data[left_offset]
                    .total_cmp(&data[right_offset])
                    .then(left.cmp(&right))
            });
            let right = child
                .checked_add(1)
                .ok_or(HdbscanError::Overflow("KD-tree child index"))?;
            stack.push((right, mid, end));
            stack.push((child, start, mid));
        }
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

    fn set_bounds(&mut self, data: &[f32], d: usize, node: usize, start: usize, end: usize) {
        for feature in 0..d {
            let first = data[self.indices[start] as usize * d + feature];
            let mut min = first;
            let mut max = first;
            for &point in &self.indices[start + 1..end] {
                let value = data[point as usize * d + feature];
                if value.total_cmp(&min).is_lt() {
                    min = value;
                }
                if value.total_cmp(&max).is_gt() {
                    max = value;
                }
            }
            let offset = node * d + feature;
            self.bounds_min[offset] = min;
            self.bounds_max[offset] = max;
        }
    }

    fn widest_axis(&self, data: &[f32], d: usize, start: usize, end: usize) -> usize {
        let mut best_axis = 0;
        let mut best_spread = -1.0f64;
        for feature in 0..d {
            let first = data[self.indices[start] as usize * d + feature];
            let mut min = first;
            let mut max = first;
            for &point in &self.indices[start + 1..end] {
                let value = data[point as usize * d + feature];
                if value.total_cmp(&min).is_lt() {
                    min = value;
                }
                if value.total_cmp(&max).is_gt() {
                    max = value;
                }
            }
            let spread = f64::from(max) - f64::from(min);
            if spread > best_spread {
                best_spread = spread;
                best_axis = feature;
            }
        }
        best_axis
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
