use crate::backend::NeighborOps;
use burn::backend::fusion::custom::{OperationIr, OperationOutput, StreamId};
use burn::backend::fusion::{Client, FusionTensor, NoOp};
use burn::backend::ir::{BackendIr, InitOperationIr};
use burn::backend::{Dispatch, TensorMetadata};
use burn::tensor::{Int, Tensor};
use burn_cubecl::{Cube, CubeBackend, fusion::FusionCubeRuntime, tensor::CubeTensor};

fn register_ann_output(
    client: &Client<FusionCubeRuntime>,
    tensor: CubeTensor,
    integer: bool,
) -> FusionTensor<FusionCubeRuntime> {
    let (shape, dtype) = (tensor.shape(), tensor.dtype);
    let handle = if integer {
        CubeBackend::int_tensor_handle(tensor)
    } else {
        CubeBackend::float_tensor_handle(tensor)
    };
    let init = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));
    client
        .register(
            StreamId::current(),
            OperationIr::Init(init),
            NoOp::<CubeBackend>::new(),
        )
        .output()
}

/// Execute ANN with the previously admitted plan, without querying a new budget.
///
/// Materialization drains pending input fusion operations. The raw driver validates
/// executable shape, tile and launch decisions before reading features or building
/// the forest; reporting-only memory fields do not choose a new execution route.
pub fn approximate_neighbors(
    data: Tensor<2>,
    plan: &crate::AnnPlan,
    seed: u64,
    max_iterations: u32,
    squared: bool,
) -> Result<(Tensor<2, Int>, Tensor<2>), crate::AnnError> {
    let fused = data.try_into_primitive::<Cube>().map_err(|error| {
        crate::AnnError::InvalidInput(format!("ANN requires a CubeCL tensor: {error:?}"))
    })?;
    let client = fused.client.clone();
    let raw = client.resolve_tensor_float::<CubeBackend>(fused);
    let (ids, distances) = crate::kernels::ann::approximate_neighbors_planned(
        raw,
        plan,
        seed,
        max_iterations,
        squared,
    )?;
    Ok((
        Tensor::from_primitive::<Cube>(register_ann_output(&client, ids, true)),
        Tensor::from_primitive::<Cube>(register_ann_output(&client, distances, false)),
    ))
}

/// Client-independent checked ANN allocation estimate; see `ann_host_estimate`.
pub use crate::kernels::ann::{AnnEstimate, AnnPhase, AnnPhaseBytes, ann_host_estimate};

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    #[cfg(feature = "test-support")]
    mod fusion_roundtrip {
        use super::*;

        fn roundtrip(device: Device) {
            let (n, features, k, seed, iterations) = (64usize, 8usize, 8u32, 7u64, 4u32);
            let values: Vec<f32> = (0..n * features)
                .map(|i| ((i * 2654435761usize) % 1009) as f32 / 1009.0)
                .collect();
            for squared in [false, true] {
                let data =
                    Tensor::<2>::from_data(TensorData::new(values.clone(), [n, features]), &device);
                let budget = crate::memory::resolve_memory_budget(&device, None).unwrap();
                let plan = crate::ann_preflight(&device, n, features, k as usize, &budget).unwrap();
                let (expected_ids, expected_distances) =
                    approximate_neighbors(data.clone(), &plan, seed, iterations, squared).unwrap();

                // Materialize a pending fusion operation before entering the raw driver.
                let fused = (data * 1.0).try_into_primitive::<Cube>().unwrap();
                let client = fused.client.clone();
                let raw = client.resolve_tensor_float::<CubeBackend>(fused);
                // This equals the default shape-only tile for n=64, capacity=60.
                let (ids, distances) = crate::approximate_neighbors_with_tile_slots(
                    raw,
                    k as usize,
                    seed,
                    iterations,
                    squared,
                    64 * (60 * 59 / 2 + 60 * 60),
                )
                .unwrap();

                let wrap_int = |tensor: CubeTensor| {
                    Tensor::<2, Int>::from_primitive::<Cube>(register_ann_output(
                        &client, tensor, true,
                    ))
                };
                let wrap_float = |tensor: CubeTensor| {
                    Tensor::<2>::from_primitive::<Cube>(register_ann_output(&client, tensor, false))
                };
                let ids = wrap_int(ids);
                let distances = wrap_float(distances);
                let expected_ids = expected_ids.into_data().try_to_vec::<i32>().unwrap();
                let expected_bits: Vec<u32> = expected_distances
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(
                    (ids.clone() + 0).into_data().try_to_vec::<i32>().unwrap(),
                    expected_ids
                );
                assert_eq!(ids.into_data().try_to_vec::<i32>().unwrap(), expected_ids);
                let actual_bits: Vec<u32> = distances
                    .clone()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(actual_bits, expected_bits);
                let downstream_bits: Vec<u32> = (distances + 0.0)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect();
                assert_eq!(downstream_bits, expected_bits);
            }
        }

        #[test]
        fn p0_3b_fusion_roundtrip_cpu() {
            roundtrip(Device::cpu());
        }

        fn strided_input_roundtrip(device: Device) {
            for features in [3, 100] {
                let n = 64;
                let values: Vec<f32> = (0..n * features)
                    .map(|i| ((i * 73 + i / features * 19) % 1009) as f32 / 1009.0)
                    .collect();
                let transposed: Vec<_> = (0..features)
                    .flat_map(|column| (0..n).map(move |row| (row, column)))
                    .map(|(row, column)| values[row * features + column])
                    .collect();
                let source =
                    Tensor::<2>::from_data(TensorData::new(transposed, [features, n]), &device)
                        .swap_dims(0, 1);
                let fused = source.try_into_primitive::<Cube>().unwrap();
                let client = fused.client.clone();
                let raw = client.resolve_tensor_float::<CubeBackend>(fused);
                assert!(
                    !raw.is_contiguous(),
                    "fixture must retain its strided layout"
                );
                let (bytes, _, _) = TensorData::new(values.clone(), [n, features]).into_parts();
                let contiguous = CubeTensor::new_contiguous(
                    raw.client.clone(),
                    raw.device.clone(),
                    burn::tensor::Shape::new([n, features]),
                    raw.client.create(bytes),
                    burn::tensor::DType::F32,
                );
                assert!(contiguous.is_contiguous());
                let wrap = |raw| {
                    Tensor::<2>::from_primitive::<Cube>(register_ann_output(&client, raw, false))
                };
                let budget = crate::resolve_memory_budget(&device, None).unwrap();
                let plan = crate::ann_preflight(&device, n, features, 8, &budget).unwrap();
                let (expected_ids, expected_distances) =
                    approximate_neighbors(wrap(contiguous), &plan, 7, 4, false).unwrap();
                let expected_ids = expected_ids.into_data().try_to_vec::<i32>().unwrap();
                let bits = |tensor: Tensor<2>| {
                    tensor
                        .into_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>()
                };
                let expected_bits = bits(expected_distances);
                let pitched =
                    Tensor::<2>::from_data(TensorData::new(values, [n, features]), &device);
                let fused = pitched.try_into_primitive::<Cube>().unwrap();
                let pitched_raw = client.resolve_tensor_float::<CubeBackend>(fused);
                let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch();
                if matches!(cube, cubecl::Device::Cuda(_)) {
                    assert!(
                        !pitched_raw.is_contiguous(),
                        "CUDA fixture must exercise row pitch"
                    );
                }
                for input in [wrap(raw), wrap(pitched_raw)] {
                    let (ids, distances) =
                        approximate_neighbors(input, &plan, 7, 4, false).unwrap();
                    assert_eq!(ids.into_data().try_to_vec::<i32>().unwrap(), expected_ids);
                    assert_eq!(bits(distances), expected_bits);
                }
            }
        }

        #[test]
        fn planned_ann_strided_input_cpu() {
            strided_input_roundtrip(Device::cpu());
        }

        #[test]
        fn planned_ann_strided_input_cuda() {
            if crate::cuda_backend::try_device_count().expect("CUDA discovery failed") == 0 {
                assert!(
                    std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                    "CUDA is required"
                );
                return;
            }
            strided_input_roundtrip(Device::cuda(0));
        }

        #[test]
        fn p0_3b_fusion_roundtrip_cuda() {
            let count = crate::cuda_backend::try_device_count().expect("CUDA discovery failed");
            if count == 0 {
                assert!(
                    std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                    "required CUDA execution has no usable device"
                );
                eprintln!("P0-3b CUDA roundtrip not run: no usable CUDA device");
                return;
            }
            roundtrip(Device::cuda(0));
        }
    }

    fn devices() -> Vec<Device> {
        let mut devices = vec![Device::cpu()];
        let count = crate::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            assert!(count > 0, "required CUDA execution has no usable device");
        }
        if count > 0 {
            devices.push(Device::cuda(0));
        }
        devices
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
        let device = Device::cpu();
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
