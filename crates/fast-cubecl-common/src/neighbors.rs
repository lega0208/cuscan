use crate::backend::NeighborOps;
use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor};

pub(crate) const REFERENCE_TILE: usize = 128;
pub(crate) const MAX_REFERENCE_GROUPS: usize = 4;
pub(crate) const MAX_NEIGHBOR_SCRATCH_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedNeighborScratch {
    pub query_rows: usize,
    pub live_bytes: usize,
}

/// Checked GPU scratch plan for bounded exact-neighbor partials and merge lists.
pub fn bounded_neighbor_scratch(
    query_rows: usize,
    reference_rows: usize,
    k: usize,
    distance_element_size: usize,
    max_page: usize,
    alignment: usize,
) -> Option<BoundedNeighborScratch> {
    if query_rows == 0
        || reference_rows == 0
        || k == 0
        || distance_element_size == 0
        || max_page == 0
        || alignment == 0
    {
        return None;
    }
    let groups = reference_rows
        .div_ceil(REFERENCE_TILE)
        .min(MAX_REFERENCE_GROUPS);
    let align = |bytes: usize| {
        bytes
            .checked_add(alignment.checked_sub(1)?)
            .map(|value| value / alignment * alignment)
    };
    let index_row = k.checked_mul(4)?;
    let distance_row = k.checked_mul(distance_element_size)?;
    let partial_index_row = index_row.checked_mul(groups)?;
    let partial_distance_row = distance_row.checked_mul(groups)?;
    let live_row = partial_index_row
        .checked_add(partial_distance_row)?
        .checked_add(index_row.checked_mul(2)?)?
        .checked_add(distance_row.checked_mul(2)?)?;
    let mut rows = (MAX_NEIGHBOR_SCRATCH_BYTES / live_row)
        .min(max_page / partial_index_row)
        .min(max_page / partial_distance_row)
        .min(max_page / index_row)
        .min(max_page / distance_row)
        .max(1)
        .min(query_rows);
    loop {
        let pi = align(rows.checked_mul(partial_index_row)?)?;
        let pd = align(rows.checked_mul(partial_distance_row)?)?;
        let ri = align(rows.checked_mul(index_row)?)?;
        let rd = align(rows.checked_mul(distance_row)?)?;
        let live_bytes = pi
            .checked_add(pd)?
            .checked_add(ri.checked_add(rd)?.checked_mul(2)?)?;
        if pi <= max_page
            && pd <= max_page
            && ri <= max_page
            && rd <= max_page
            && live_bytes <= MAX_NEIGHBOR_SCRATCH_BYTES
        {
            return Some(BoundedNeighborScratch {
                query_rows: rows,
                live_bytes,
            });
        }
        if rows == 1 {
            return None;
        }
        rows -= 1;
    }
}

/// Device exact-neighbor operation dispatched on the input backend.
///
/// Squared mode permits missing neighbors, represented by raw reference-row-count
/// indices and positive infinity. Euclidean mode preserves UMAP's strict width.
/// Brute-force scratch is bounded by 32 MiB total, further capped by the selected
/// device's per-allocation limit; query groups shrink as runtime `k` grows.
/// CPU and unsquared requests retain the direct serial-reference kernel; squared
/// CUDA requests use grouped partial lists and merge dispatches.
pub fn exact_neighbors(
    query: Tensor<2>,
    reference: Tensor<2>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
    squared: bool,
) -> (Tensor<2, Int>, Tensor<2>) {
    let (ids, distances) = <Dispatch as NeighborOps>::exact_neighbors(
        query.into_dispatch(),
        reference.into_dispatch(),
        k,
        exclude_self,
        query_offset,
        squared,
    );
    (Tensor::from_dispatch(ids), Tensor::from_dispatch(distances))
}

/// Compute one bounded query-by-reference squared-distance block.
pub fn squared_euclidean_distance_block(query: Tensor<2>, reference: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as NeighborOps>::squared_euclidean_distance_block(
        query.into_dispatch(),
        reference.into_dispatch(),
    ))
}

#[cfg(all(test, any(feature = "cpu", feature = "cuda")))]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    fn devices() -> Vec<Device> {
        #[cfg(all(feature = "cpu", feature = "cuda"))]
        {
            let mut devices = vec![Device::cpu()];
            let count = crate::cuda_backend::try_device_count().expect("CUDA discovery failed");
            if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
                assert!(count > 0, "required CUDA execution has no usable device");
            }
            if count > 0 {
                devices.push(Device::cuda(0));
            }
            devices
        }
        #[cfg(all(feature = "cpu", not(feature = "cuda")))]
        {
            vec![Device::cpu()]
        }
        #[cfg(all(feature = "cuda", not(feature = "cpu")))]
        {
            let count = crate::cuda_backend::try_device_count().expect("CUDA discovery failed");
            if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
                assert!(count > 0, "required CUDA execution has no usable device");
            }
            if count > 0 {
                vec![Device::cuda(0)]
            } else {
                Vec::new()
            }
        }
    }

    fn tensor(values: Vec<f32>, rows: usize, device: &Device) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values, [rows, 1]), device)
    }

    fn exact_oracle(
        query: &[f32],
        reference: &[f32],
        k: usize,
        offset: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut ids = Vec::with_capacity(query.len() * k);
        let mut distances = Vec::with_capacity(query.len() * k);
        for (row, &value) in query.iter().enumerate() {
            let mut candidates = reference
                .iter()
                .enumerate()
                .filter(|(id, _)| *id != row + offset)
                .map(|(id, &other)| {
                    let delta = value - other;
                    ((delta * delta, id as i32), delta * delta)
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|a, b| a.0.0.total_cmp(&b.0.0).then(a.0.1.cmp(&b.0.1)));
            for slot in 0..k {
                if let Some(candidate) = candidates.get(slot) {
                    ids.push(candidate.0.1);
                    distances.push(candidate.1);
                } else {
                    ids.push(reference.len() as i32);
                    distances.push(f32::INFINITY);
                }
            }
        }
        (ids, distances)
    }

    #[test]
    fn tiled_merge_matches_independent_oracle_across_query_and_reference_tails() {
        for device in devices() {
            let query: Vec<f32> = (0..701).map(|i| (i % 17) as f32).collect();
            let mut reference: Vec<f32> = (0..763).map(|i| ((i * 7) % 23) as f32).collect();
            reference[12..713].copy_from_slice(&query);
            let (actual_ids, actual_distances) = exact_neighbors(
                tensor(query.clone(), query.len(), &device),
                tensor(reference.clone(), reference.len(), &device),
                1300,
                true,
                12,
                true,
            );
            let (expected_ids, expected_distances) = exact_oracle(&query, &reference, 1300, 12);
            assert_eq!(
                actual_ids.to_data().try_to_vec::<i32>().unwrap(),
                expected_ids
            );
            assert_eq!(
                actual_distances.to_data().try_to_vec::<f32>().unwrap(),
                expected_distances
            );
        }
    }

    #[test]
    fn squared_mode_orders_ties_and_preserves_squared_distances() {
        for device in devices() {
            let q = tensor(vec![0.0], 1, &device);
            let r = tensor(vec![1.0, -1.0, 2.0], 3, &device);
            let (ids, d) = exact_neighbors(q, r, 3, false, 0, true);
            assert_eq!(ids.dims(), [1, 3]);
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [0, 1, 2]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [1.0, 1.0, 4.0]);
        }
    }

    #[test]
    fn squared_mode_real_infinity_precedes_raw_n_padding() {
        for device in devices() {
            let q = tensor(vec![f32::MAX], 1, &device);
            let r = tensor(vec![-f32::MAX, -f32::MAX], 2, &device);
            let (ids, d) = exact_neighbors(q, r, 3, false, 0, true);
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [0, 1, 2]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [f32::INFINITY; 3]);
        }
    }

    #[test]
    fn squared_mode_excludes_offset_self_and_pads_missing_slots() {
        for device in devices() {
            let q = tensor(vec![10.0, 20.0], 2, &device);
            let r = tensor(vec![0.0, 10.0, 20.0], 3, &device);
            let (ids, d) = exact_neighbors(q, r, 3, true, 1, true);
            assert_eq!(
                ids.to_data().try_to_vec::<i32>().unwrap(),
                [0, 2, 3, 1, 0, 3]
            );
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [100.0, 100.0, f32::INFINITY, 100.0, 400.0, f32::INFINITY]
            );
            let (ids, d) = exact_neighbors(
                tensor(vec![0.0], 1, &device),
                tensor(vec![0.0], 1, &device),
                2,
                true,
                0,
                true,
            );
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [1, 1]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [f32::INFINITY; 2]);
        }
    }

    #[test]
    fn euclidean_mode_roots_distances_and_retains_overflow_sentinels() {
        for device in devices() {
            let (_, d) = exact_neighbors(
                tensor(vec![0.0], 1, &device),
                tensor(vec![3.0, 4.0], 2, &device),
                2,
                false,
                0,
                false,
            );
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [3.0, 4.0]);
            let (ids, d) = exact_neighbors(
                tensor(vec![f32::MAX], 1, &device),
                tensor(vec![-f32::MAX, -f32::MAX], 2, &device),
                2,
                false,
                0,
                false,
            );
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [2, 2]);
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [f32::MAX.sqrt(); 2]
            );
        }
    }

    #[test]
    #[should_panic(expected = "neighbor count exceeds available references")]
    fn euclidean_mode_rejects_excess_width_at_execution() {
        #[cfg(feature = "cpu")]
        let device = Device::cpu();
        #[cfg(not(feature = "cpu"))]
        let device = Device::cuda(0);
        let (_, d) = exact_neighbors(
            tensor(vec![0.0], 1, &device),
            tensor(vec![3.0, 4.0], 2, &device),
            3,
            false,
            0,
            false,
        );
        let _ = d.to_data();
    }

    #[test]
    fn distance_blocks_keep_shape_and_squared_values() {
        for device in devices() {
            let d = squared_euclidean_distance_block(
                tensor(vec![0.0, 2.0], 2, &device),
                tensor(vec![1.0, 4.0, 6.0], 3, &device),
            );
            assert_eq!(d.dims(), [2, 3]);
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [1.0, 16.0, 36.0, 1.0, 4.0, 16.0]
            );
        }
    }
}
