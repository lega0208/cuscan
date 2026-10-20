//! Bounded selected-backend neighbor search orchestration.

use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor};

use super::{Result, UmapError};
use crate::backend::StandardUmapOps;

/// Neighbor IDs and Euclidean distances, ordered by (distance, row index).
#[derive(Debug, Clone)]
pub struct NeighborList {
    pub indices: Tensor<2, Int>,
    pub distances: Tensor<2>,
}

/// Search rectangular query blocks without materializing a full fit distance matrix.
/// Self exclusion applies only to aligned fit rows, never bipartite queries.
pub fn exact_neighbors(
    query: Tensor<2>,
    reference: Tensor<2>,
    k: usize,
    exclude_self: bool,
) -> Result<NeighborList> {
    let [query_rows, features] = query.dims();
    let [reference_rows, reference_features] = reference.dims();
    if query_rows == 0 || reference_rows == 0 || features == 0 {
        return Err(UmapError::InvalidInput(
            "neighbor inputs must be nonempty".into(),
        ));
    }
    if features != reference_features || query.device() != reference.device() {
        return Err(UmapError::InvalidInput(
            "neighbor inputs must have matching features and devices".into(),
        ));
    }
    if exclude_self && query_rows != reference_rows {
        return Err(UmapError::InvalidInput(
            "self exclusion requires aligned fit rows".into(),
        ));
    }
    let available = reference_rows - usize::from(exclude_self);
    if k == 0 || k > available {
        return Err(UmapError::InvalidInput(
            "neighbor width exceeds available rows".into(),
        ));
    }
    if reference_rows > i32::MAX as usize {
        return Err(UmapError::Overflow("neighbor row indices"));
    }
    let width = u32::try_from(k).map_err(|_| UmapError::Overflow("neighbor width"))?;
    query_rows
        .checked_mul(k)
        .and_then(|count| count.checked_mul(4))
        .ok_or(UmapError::Overflow("neighbor output allocation"))?;
    // Even a tiny fit never dispatches an N-by-N query tile.
    let block_rows = reference_rows.saturating_sub(1).clamp(1, 64);
    let mut indices = Vec::new();
    let mut distances = Vec::new();
    for start in (0..query_rows).step_by(block_rows) {
        let end = start.saturating_add(block_rows).min(query_rows);
        let offset = u32::try_from(start).map_err(|_| UmapError::Overflow("query row offset"))?;
        let block = query.clone().slice([start..end, 0..features]);
        let (block_indices, block_distances) = <Dispatch as StandardUmapOps>::exact_neighbors(
            block.into_dispatch(),
            reference.clone().into_dispatch(),
            width,
            exclude_self,
            offset,
        );
        indices.push(Tensor::<2, Int>::from_dispatch(block_indices));
        distances.push(Tensor::<2>::from_dispatch(block_distances));
    }
    Ok(NeighborList {
        indices: Tensor::cat(indices, 0),
        distances: Tensor::cat(distances, 0),
    })
}
