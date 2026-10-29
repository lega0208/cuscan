//! S1, S2 and S13 input handling for non-finite rows.

use crate::condense::CondensedTree;
use crate::error::{HdbscanError, Result};
use crate::linkage::LinkageRow;

/// Mapping from compact finite rows to raw input rows, plus raw non-finite rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FiniteSplit {
    /// Internal row → raw row, ascending.
    pub finite_index: Vec<u32>,
    /// Raw rows containing at least one non-finite feature, ascending.
    pub outliers: Vec<u32>,
}

/// Validate dimensions without allocating, including dimensions used by the device interface.
pub fn validate_dims(n: usize, d: usize) -> Result<()> {
    if n == 0 {
        return Err(HdbscanError::InvalidInput(
            "input must contain at least one row".into(),
        ));
    }
    if d == 0 {
        return Err(HdbscanError::InvalidInput(
            "input must contain at least one feature".into(),
        ));
    }
    n.checked_mul(d)
        .ok_or(HdbscanError::Overflow("input element count"))?;
    if n > i32::MAX as usize {
        return Err(HdbscanError::InvalidInput(format!(
            "row count {n} exceeds the supported maximum {}",
            i32::MAX
        )));
    }
    Ok(())
}

/// Validate and flatten row-major input, returning `(values, n_rows, n_features)`.
pub fn flatten_input(rows: &[Vec<f32>]) -> Result<(Vec<f32>, usize, usize)> {
    if rows.is_empty() {
        return Err(HdbscanError::InvalidInput(
            "input must contain at least one row".into(),
        ));
    }
    let n = rows.len();
    let d = rows[0].len();
    if rows.iter().any(|row| row.len() != d) {
        return Err(HdbscanError::InvalidInput(
            "input rows must all have the same number of features".into(),
        ));
    }
    validate_dims(n, d)?;
    let capacity = n
        .checked_mul(d)
        .ok_or(HdbscanError::Overflow("input element count"))?;
    let mut values = Vec::with_capacity(capacity);
    for row in rows {
        values.extend_from_slice(row);
    }
    Ok((values, n, d))
}

/// Split flattened row-major data into ascending finite and outlier row indices.
pub fn split_finite(data: &[f32], n_features: usize) -> Result<FiniteSplit> {
    if n_features == 0 {
        return Err(HdbscanError::InvalidInput(
            "input must contain at least one feature".into(),
        ));
    }
    if !data.len().is_multiple_of(n_features) {
        return Err(HdbscanError::InvalidInput(
            "flattened input length is not divisible by feature count".into(),
        ));
    }
    let n = data.len() / n_features;
    validate_dims(n, n_features)?;

    let mut split = FiniteSplit::default();
    for (row, values) in data.chunks_exact(n_features).enumerate() {
        let index = u32::try_from(row)
            .map_err(|_| HdbscanError::InvalidInput("row index exceeds u32".into()))?;
        if values.iter().all(|value| value.is_finite()) {
            split.finite_index.push(index);
        } else {
            split.outliers.push(index);
        }
    }
    if split.finite_index.is_empty() {
        return Err(HdbscanError::InvalidInput(
            "input has no rows with all finite features".into(),
        ));
    }
    Ok(split)
}

/// Copy finite rows from flattened row-major data into compact row-major storage.
pub fn compact_rows(data: &[f32], n_features: usize, finite_index: &[u32]) -> Result<Vec<f32>> {
    if n_features == 0 {
        return Err(HdbscanError::InvalidInput(
            "input must contain at least one feature".into(),
        ));
    }
    if !data.len().is_multiple_of(n_features) {
        return Err(HdbscanError::InvalidInput(
            "flattened input length is not divisible by feature count".into(),
        ));
    }
    let n_rows = data.len() / n_features;
    let capacity = finite_index
        .len()
        .checked_mul(n_features)
        .ok_or(HdbscanError::Overflow("compacted input element count"))?;
    let mut compact = Vec::with_capacity(capacity);
    for &index in finite_index {
        let start = (index as usize)
            .checked_mul(n_features)
            .ok_or(HdbscanError::Overflow("input row offset"))?;
        let end = start
            .checked_add(n_features)
            .ok_or(HdbscanError::Overflow("input row end"))?;
        let row = data
            .get(start..end)
            .filter(|_| (index as usize) < n_rows)
            .ok_or_else(|| {
                HdbscanError::InvalidInput(format!("finite row index {index} is out of range"))
            })?;
        compact.extend_from_slice(row);
    }
    Ok(compact)
}

/// Scatter finite-row labels into raw row order, using `-1` for outliers.
pub fn scatter_labels(labels: &[i32], finite_index: &[u32], n_rows: usize) -> Result<Vec<i32>> {
    if labels.len() != finite_index.len() {
        return Err(HdbscanError::InvalidInput(
            "label count does not match finite row count".into(),
        ));
    }
    let mut scattered = vec![-1; n_rows];
    for (&label, &raw_index) in labels.iter().zip(finite_index) {
        let slot = scattered.get_mut(raw_index as usize).ok_or_else(|| {
            HdbscanError::InvalidInput(format!("finite row index {raw_index} is out of range"))
        })?;
        *slot = label;
    }
    Ok(scattered)
}

/// Scatter finite-row membership strengths into raw row order, using `0` for outliers.
pub fn scatter_probabilities(
    probabilities: &[f32],
    finite_index: &[u32],
    n_rows: usize,
) -> Result<Vec<f32>> {
    if probabilities.len() != finite_index.len() {
        return Err(HdbscanError::InvalidInput(
            "probability count does not match finite row count".into(),
        ));
    }
    let mut scattered = vec![0.0; n_rows];
    for (&probability, &raw_index) in probabilities.iter().zip(finite_index) {
        let slot = scattered.get_mut(raw_index as usize).ok_or_else(|| {
            HdbscanError::InvalidInput(format!("finite row index {raw_index} is out of range"))
        })?;
        *slot = probability;
    }
    Ok(scattered)
}

fn shifted_node(node: usize, finite_count: usize, outlier_count: usize) -> Result<usize> {
    if node < finite_count {
        return Err(HdbscanError::InvalidInput(
            "tree parent unexpectedly refers to a point node".into(),
        ));
    }
    node.checked_add(outlier_count)
        .ok_or(HdbscanError::Overflow("remapped tree node id"))
}

fn remap_child(node: usize, finite_index: &[u32], outlier_count: usize) -> Result<usize> {
    if node < finite_index.len() {
        Ok(finite_index[node] as usize)
    } else {
        node.checked_add(outlier_count)
            .ok_or(HdbscanError::Overflow("remapped tree node id"))
    }
}

/// Remap a finite-row condensed tree and prepend one zero-lambda row per outlier.
///
/// `outlier_order` is injectable because upstream's set-derived order is not stable.
pub fn remap_condensed(
    tree: &CondensedTree,
    finite_index: &[u32],
    outlier_order: &[u32],
) -> Result<CondensedTree> {
    if tree.is_empty() {
        if tree.child.is_empty() && tree.lambda.is_empty() && tree.child_size.is_empty() {
            return Ok(tree.clone());
        }
        return Err(HdbscanError::InvalidInput(
            "condensed tree columns have inconsistent lengths".into(),
        ));
    }
    if tree.child.len() != tree.len()
        || tree.lambda.len() != tree.len()
        || tree.child_size.len() != tree.len()
    {
        return Err(HdbscanError::InvalidInput(
            "condensed tree columns have inconsistent lengths".into(),
        ));
    }
    let outlier_count = outlier_order.len();
    let root = shifted_node(tree.parent[0], finite_index.len(), outlier_count)?;
    let capacity = tree
        .len()
        .checked_add(outlier_count)
        .ok_or(HdbscanError::Overflow("remapped condensed tree length"))?;
    let mut remapped = CondensedTree {
        parent: Vec::with_capacity(capacity),
        child: Vec::with_capacity(capacity),
        lambda: Vec::with_capacity(capacity),
        child_size: Vec::with_capacity(capacity),
    };
    for &outlier in outlier_order {
        remapped.parent.push(root);
        remapped.child.push(outlier as usize);
        remapped.lambda.push(0.0);
        remapped.child_size.push(1.0);
    }
    for index in 0..tree.len() {
        remapped.parent.push(shifted_node(
            tree.parent[index],
            finite_index.len(),
            outlier_count,
        )?);
        remapped
            .child
            .push(remap_child(tree.child[index], finite_index, outlier_count)?);
        remapped.lambda.push(tree.lambda[index]);
        remapped.child_size.push(tree.child_size[index]);
    }
    Ok(remapped)
}

fn remap_linkage_node(node: usize, finite_index: &[u32], outlier_count: usize) -> Result<usize> {
    remap_child(node, finite_index, outlier_count)
}

/// Remap linkage point ids and append an infinite-distance chain for outliers.
///
/// `outlier_order` is injectable because upstream's set-derived order is not stable.
pub fn remap_linkage(
    tree: &[LinkageRow],
    finite_index: &[u32],
    outlier_order: &[u32],
) -> Result<Vec<LinkageRow>> {
    if tree.is_empty() {
        return Ok(Vec::new());
    }
    let capacity = tree
        .len()
        .checked_add(outlier_order.len())
        .ok_or(HdbscanError::Overflow("remapped linkage length"))?;
    let mut remapped = Vec::with_capacity(capacity);
    for row in tree {
        remapped.push(LinkageRow {
            left: remap_linkage_node(row.left, finite_index, outlier_order.len())?,
            right: remap_linkage_node(row.right, finite_index, outlier_order.len())?,
            distance: row.distance,
            size: row.size,
        });
    }
    let last = remapped.last().expect("non-empty linkage tree");
    let mut last_id = last.left.max(last.right);
    let mut last_size = last.size;
    for &outlier in outlier_order {
        last_id = last_id
            .checked_add(1)
            .ok_or(HdbscanError::Overflow("remapped linkage node id"))?;
        last_size = last_size
            .checked_add(1)
            .ok_or(HdbscanError::Overflow("remapped linkage cluster size"))?;
        remapped.push(LinkageRow {
            left: outlier as usize,
            right: last_id,
            distance: f64::INFINITY,
            size: last_size,
        });
    }
    Ok(remapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_split_compact_and_scatter() {
        let (data, n, d) =
            flatten_input(&[vec![1.0, 2.0], vec![f32::NAN, 0.0], vec![3.0, 4.0]]).unwrap();
        assert_eq!((n, d), (3, 2));
        let split = split_finite(&data, d).unwrap();
        assert_eq!(split.finite_index, [0, 2]);
        assert_eq!(split.outliers, [1]);
        assert_eq!(
            compact_rows(&data, d, &split.finite_index).unwrap(),
            [1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(
            scatter_labels(&[5, 6], &split.finite_index, n).unwrap(),
            [5, -1, 6]
        );
        assert_eq!(
            scatter_probabilities(&[0.25, 1.0], &split.finite_index, n).unwrap(),
            [0.25, 0.0, 1.0]
        );
    }

    #[test]
    fn validation_dimensions_without_allocations() {
        assert!(matches!(
            validate_dims(i32::MAX as usize + 1, 1),
            Err(HdbscanError::InvalidInput(_))
        ));
        assert_eq!(
            validate_dims(usize::MAX, 2),
            Err(HdbscanError::Overflow("input element count"))
        );
    }

    #[test]
    fn remap_condensed_injected_outlier_order_and_empty_tree() {
        let tree = CondensedTree {
            parent: vec![2, 2],
            child: vec![0, 3],
            lambda: vec![1.0, 2.0],
            child_size: vec![1.0, 2.0],
        };
        let mapped = remap_condensed(&tree, &[1, 3], &[2, 0]).unwrap();
        assert_eq!(mapped.parent, [4, 4, 4, 4]);
        assert_eq!(mapped.child, [2, 0, 1, 5]);
        assert_eq!(mapped.lambda, [0.0, 0.0, 1.0, 2.0]);
        assert!(
            remap_condensed(&CondensedTree::default(), &[], &[0])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn remap_linkage_injected_outlier_order_and_empty_tree() {
        let tree = [LinkageRow {
            left: 1,
            right: 0,
            distance: 2.0,
            size: 2,
        }];
        let mapped = remap_linkage(&tree, &[2, 4], &[3, 0]).unwrap();
        assert_eq!(mapped[0].left, 4);
        assert_eq!(mapped[0].right, 2);
        assert_eq!(
            mapped[1],
            LinkageRow {
                left: 3,
                right: 5,
                distance: f64::INFINITY,
                size: 3,
            }
        );
        assert_eq!(mapped[2].left, 0);
        assert_eq!(mapped[2].right, 6);
        assert!(remap_linkage(&[], &[], &[0]).unwrap().is_empty());
    }
}
