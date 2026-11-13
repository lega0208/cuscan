use super::*;

use burn_cubecl::CubeDevice;
use std::collections::BTreeSet;

fn raw_tensor<T: burn::tensor::Element>(
    device: &CubeDevice,
    shape: &[usize],
    values: Vec<T>,
    dtype: DType,
) -> CubeTensor {
    let shape = Shape::from(shape.to_vec());
    let data = burn::tensor::TensorData::new(values, shape.clone());
    assert_eq!(data.dtype(), dtype);
    let (bytes, _, _) = data.into_parts();
    let client = device.client();
    CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        shape,
        client.create(bytes),
        dtype,
    )
}

fn i32_tensor(device: &CubeDevice, shape: &[usize], values: Vec<i32>) -> CubeTensor {
    raw_tensor(device, shape, values, DType::I32)
}

fn f32_tensor(device: &CubeDevice, shape: &[usize], values: Vec<f32>) -> CubeTensor {
    raw_tensor(device, shape, values, DType::F32)
}

#[cfg(debug_assertions)]
fn lifetime_allocator() -> AnnAllocator {
    let device = CubeDevice::Cpu(Default::default());
    let dispatch = burn::tensor::Device::new(burn::backend::DispatchDevice::Cube(device.clone()));
    let budget = crate::memory::MemoryBudget {
        total_bytes: u64::MAX,
        free_bytes: u64::MAX,
        limit_bytes: None,
    };
    let plan = super::super::orchestration::ann_preflight(&dispatch, 11, 2, 3, &budget).unwrap();
    let table = ann_buffer_table(
        11,
        2,
        3,
        plan.host.width,
        plan.host.capacity,
        &plan.host.workspace,
        plan.tile_slots,
    )
    .unwrap();
    AnnAllocator {
        table,
        allocated: Vec::new(),
        client: device.client(),
        device,
        plan,
    }
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "retained outside phase")]
fn allocator_rejects_wrong_phase_retained_handle() {
    let mut allocator = lifetime_allocator();
    let rows = allocator
        .table
        .iter()
        .find(|entry| entry.buffer == Buffer::ForestRows)
        .unwrap()
        .shape[0];
    let _rows = allocator.empty(Buffer::ForestRows, Shape::new([rows]), DType::I32);
    allocator.assert_live_phase(4);
}

#[test]
#[cfg(debug_assertions)]
fn allocator_release_tracks_swapped_heap_identity() {
    let mut allocator = lifetime_allocator();
    let width = allocator.plan.host.width;
    let mut candidates = allocator.empty(Buffer::HeapIds, Shape::new([11, width]), DType::I32);
    let mut working = allocator.empty(Buffer::WorkingIds, Shape::new([11, width]), DType::I32);
    let rows = allocator
        .table
        .iter()
        .find(|entry| entry.buffer == Buffer::ForestRows)
        .unwrap()
        .shape[0];
    let forest = allocator.empty(Buffer::ForestRows, Shape::new([rows]), DType::I32);
    drop(forest);
    allocator.release_except(4);
    assert!(
        !allocator
            .allocated
            .iter()
            .any(|(key, _)| *key == Buffer::ForestRows)
    );
    allocator.assert_live_phase(4);
    std::mem::swap(&mut candidates, &mut working);
    allocator.swap_keys(Buffer::HeapIds, Buffer::WorkingIds);
    assert_ne!(
        candidates.handle.memory.clone().binding().id(),
        working.handle.memory.clone().binding().id()
    );
    drop(working);
    allocator.release_except(8);
    allocator.assert_live_phase(8);
    allocator.assert_heap_identity(&candidates);
    assert_eq!(allocator.allocated.len(), 1);
}

#[test]
fn memory_failure_preserves_fresh_availability_and_server_error() {
    for stage in ["allocation", "input buffer synchronization"] {
        let error = memory_failure(stage, "synthetic server error", 500, 400, Ok(123));
        match error {
            AnnError::Memory {
                required_bytes,
                available_bytes,
                message,
            } => {
                assert_eq!(required_bytes, 500);
                assert_eq!(available_bytes, 123);
                assert!(message.contains(stage));
                assert!(message.contains("synthetic server error"));
                assert!(message.contains("fresh free 123 bytes"));
                assert!(message.contains("planned usable 400 bytes"));
            }
            other => panic!("expected memory error, got {other:?}"),
        }
    }
}

#[test]
fn memory_failure_preserves_query_failure_as_unknown_availability() {
    for stage in ["allocation", "input buffer synchronization"] {
        let error = memory_failure(
            stage,
            "synthetic server error",
            500,
            400,
            Err("synthetic query error".into()),
        );
        match error {
            AnnError::Memory {
                required_bytes,
                available_bytes,
                message,
            } => {
                assert_eq!(required_bytes, 500);
                assert_eq!(available_bytes, 0);
                assert!(message.contains(stage));
                assert!(message.contains("synthetic server error"));
                assert!(message.contains("synthetic query error"));
                assert!(message.contains("unknown"));
                assert!(message.contains("planned usable 400 bytes"));
            }
            other => panic!("expected memory error, got {other:?}"),
        }
    }
}

#[test]
fn cpu_combined_budget_rejects_before_forest_readback() {
    let (n, features, k) = (64, 1024, 8);
    let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
    let burn::backend::DispatchDevice::Cube(cube) = device.as_dispatch();
    let client = cube.client();
    let alignment = usize::try_from(client.properties().memory.alignment).unwrap();
    let ample = crate::memory::MemoryBudget {
        total_bytes: u64::MAX,
        free_bytes: u64::MAX,
        limit_bytes: None,
    };
    let plan = super::super::orchestration::ann_preflight(&device, n, features, k, &ample).unwrap();
    let slots = (n * plan.host.workspace.pair_slots_per_center)
        .max(plan.host.workspace.forest_pair_count)
        .min(1 << 20);
    let table = ann_buffer_table(
        n,
        features,
        k,
        plan.host.width,
        plan.host.capacity,
        &plan.host.workspace,
        slots,
    )
    .unwrap();
    let mut fixed = [0u64; 4];
    let mut staging = 0u64;
    for entry in table
        .iter()
        .filter(|entry| entry.route == BufferRoute::Both)
    {
        let bytes = (entry.bytes().unwrap().div_ceil(alignment) * alignment) as u64;
        if entry.staging {
            staging += bytes;
        } else {
            for (phase, total) in fixed.iter_mut().enumerate() {
                if entry.phases & (1 << phase) != 0 {
                    *total += bytes;
                }
            }
        }
    }
    let runtime_min = *fixed.iter().max().unwrap() + staging;
    fixed[1] += plan.host.workspace.host_peak_bytes as u64;
    let cpu_min = *fixed.iter().max().unwrap() + staging;
    assert!(cpu_min > runtime_min);
    let usable = runtime_min + (cpu_min - runtime_min) / 2;
    let budget = crate::memory::MemoryBudget {
        limit_bytes: Some(usable + crate::memory::MEMORY_RESERVE_BYTES),
        ..ample
    };
    assert_eq!(budget.usable_bytes(0), usable);

    // The logical shape is valid, but forest_seed rejects this short readback.
    let data = CubeTensor::new_contiguous(
        client.clone(),
        cube.clone(),
        Shape::new([n, features]),
        client.empty(4),
        DType::F32,
    );
    ALLOCATION_LOG.with(|log| log.borrow_mut().clear());
    let result = super::super::orchestration::ann_preflight(&device, n, features, k, &budget)
        .and_then(|plan| {
            super::super::orchestration::approximate_neighbors_planned(data, &plan, 42, 1, false)
        });
    match result {
        Err(AnnError::Memory {
            required_bytes,
            available_bytes,
            ..
        }) => {
            assert_eq!(required_bytes, cpu_min);
            assert_eq!(available_bytes, usable);
            assert!(runtime_min <= available_bytes && available_bytes < required_bytes);
        }
        other => panic!("expected combined CPU memory rejection, got {other:?}"),
    }
    ALLOCATION_LOG.with(|log| assert!(log.borrow().is_empty()));
}

fn driver_allocation_fixture(device: CubeDevice, pitched: bool) {
    use std::collections::HashMap;

    let n = 16;
    let features = 3;
    let k = 3;
    let values: Vec<f32> = (0..n * features)
        .map(|i| ((i * 17 + i * i * 3) % 101) as f32 / 13.0)
        .collect();
    let contiguous = f32_tensor(&device, &[n, features], values.clone());
    let strided = if pitched {
        let data = burn::tensor::TensorData::new(values.clone(), Shape::new([n, features]));
        let (bytes, shape, dtype) = data.into_parts();
        let client = device.client();
        let allocation = client.create_tensor(bytes, shape, dtype.size());
        let mut tensor = contiguous.clone();
        tensor.handle = allocation.memory;
        tensor.meta.strides = allocation.strides;
        tensor
    } else {
        let transposed: Vec<f32> = (0..features)
            .flat_map(|column| {
                (0..n).map({
                    let values = &values;
                    move |row| values[row * features + column]
                })
            })
            .collect();
        let mut tensor = f32_tensor(&device, &[features, n], transposed);
        tensor.meta.shape.swap(0, 1);
        tensor.meta.strides.swap(0, 1);
        tensor
    };
    assert!(contiguous.is_contiguous());
    assert!(
        !strided.is_contiguous(),
        "fixture must exercise input copying"
    );
    let device = burn::tensor::Device::new(burn::backend::DispatchDevice::Cube(device));
    let budget = crate::memory::MemoryBudget {
        total_bytes: u64::MAX,
        free_bytes: u64::MAX,
        limit_bytes: None,
    };
    let plan =
        super::super::orchestration::ann_preflight(&device, n, features, k, &budget).unwrap();
    let table = ann_buffer_table(
        n,
        features,
        k,
        plan.host.width,
        plan.host.capacity,
        &plan.host.workspace,
        plan.tile_slots,
    )
    .unwrap();
    let mut outputs = Vec::new();
    for (input, copied) in [(contiguous, false), (strided, true)] {
        ALLOCATION_LOG.with(|log| log.borrow_mut().clear());
        let (ids, distances) = approximate_neighbors(input, k, 42, 2, false, plan).unwrap();
        let log = ALLOCATION_LOG.with(|log| log.borrow().clone());
        let mut counts = HashMap::new();
        for key in &log {
            *counts.entry(*key).or_insert(0usize) += 1;
        }
        let survival = cfg!(feature = "test-support")
            && std::env::var_os("CUSCAN_ANN_PROFILE").is_some()
            && std::env::var_os("CUSCAN_ANN_SURVIVAL_PROFILE").is_some();
        for entry in &table {
            let expected = match entry.buffer {
                Buffer::Input => 0,
                Buffer::InputCopy => usize::from(copied),
                Buffer::ProfileCounts => {
                    usize::from(std::env::var_os("CUSCAN_ANN_PROFILE").is_some())
                }
                Buffer::SurvivalCounts => usize::from(survival),
                _ if entry.route == BufferRoute::Cuda && plan.merge_geometry().0 => 0,
                _ => 1,
            };
            assert_eq!(
                counts.remove(&entry.buffer).unwrap_or(0),
                expected,
                "allocation {:?}",
                entry.buffer
            );
        }
        assert!(
            counts.is_empty(),
            "allocations absent from table: {counts:?}"
        );
        let forest = [
            Buffer::ForestRows,
            Buffer::ForestLeafRows,
            Buffer::ForestPairPrefix,
            Buffer::HeapIds,
            Buffer::HeapDistances,
            Buffer::Flags,
        ];
        let start = usize::from(copied);
        assert_eq!(&log[start..start + forest.len()], &forest);
        outputs.push((
            ids.client.read_one_unchecked(ids.handle.clone()).to_vec(),
            distances
                .client
                .read_one_unchecked(distances.handle.clone())
                .to_vec(),
        ));
    }
    assert_eq!(
        outputs[0], outputs[1],
        "input layout changed ANN output bits"
    );
}

#[test]
fn driver_allocation_log_and_strided_input_cpu() {
    driver_allocation_fixture(CubeDevice::Cpu(Default::default()), false);
}

#[test]
fn driver_allocation_log_and_pitched_input_cuda() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    if count == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    driver_allocation_fixture(
        CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }),
        true,
    );
}

fn read_i32(tensor: &CubeTensor) -> Vec<i32> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| i32::from_ne_bytes(*bytes))
        .collect()
}

fn read_f32(tensor: &CubeTensor) -> Vec<f32> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_ne_bytes(*bytes))
        .collect()
}

fn launch_units(client: &cubecl::client::Client, units: usize) -> (CubeCount, CubeDim) {
    ann_independent_launch(client, u32::try_from(units).unwrap()).unwrap()
}

#[cfg(feature = "test-support")]
fn survival_fixture(device: CubeDevice) {
    let client = device.client();
    let counts = i32_tensor(&device, &[2], vec![123, 456]);
    let heap_ids = i32_tensor(&device, &[4, 2], vec![1, 2, 0, 2, 0, 4, 0, 1]);
    let heap_distances = f32_tensor(
        &device,
        &[4, 2],
        vec![1.0, 5.0, 1.0, 5.0, 1.0, f32::INFINITY, 1.0, 5.0],
    );
    // The diagnostic tests only the tile-start worst-entry predicate, not the
    // merge's separate nonnegative/finite validity rules or batch tightening.
    let cases = [
        (0, 1, 5.0, 1, 2, 2),      // lower IDs win both exact ties
        (0, 3, 5.0, 1, 2, 1),      // higher ID loses at row 0
        (0, 2, f32::NAN, 1, 2, 1), // NaN fails full row, passes non-full row
        (0, 1, -1.0, 1, 2, 2),     // negative passes the threshold predicate
        (0, 1, f32::NEG_INFINITY, 1, 2, 2),
        (0, 2, f32::INFINITY, 1, 2, 1),
        (99, 99, 0.0, 0, 0, 0), // invalid slot must not access either row
    ];
    for (left_id, right_id, distance, is_valid, produced, predicted) in cases {
        let left = i32_tensor(&device, &[1], vec![left_id]);
        let right = i32_tensor(&device, &[1], vec![right_id]);
        let distances = f32_tensor(&device, &[1], vec![distance]);
        let valid = i32_tensor(&device, &[1], vec![is_valid * PRODUCED]);
        let inputs = [
            &left,
            &right,
            &distances,
            &valid,
            &heap_ids,
            &heap_distances,
        ];
        let before: Vec<Vec<u8>> = inputs
            .iter()
            .map(|tensor| client.read_one_unchecked(tensor.handle.clone()).to_vec())
            .collect();
        let actual = profile_tile_survival(
            &client,
            &counts,
            &left,
            &right,
            &distances,
            &valid,
            &heap_ids,
            &heap_distances,
            4,
            2,
            1,
        )
        .unwrap();
        assert_eq!(
            actual,
            (produced, predicted),
            "left={left_id} right={right_id} distance={distance:?} valid={is_valid}"
        );
        let after: Vec<Vec<u8>> = inputs
            .iter()
            .map(|tensor| client.read_one_unchecked(tensor.handle.clone()).to_vec())
            .collect();
        assert_eq!(before, after, "diagnostic mutated an input");
    }
}

#[cfg(feature = "test-support")]
#[test]
fn survival_profile_fixture_cpu() {
    survival_fixture(CubeDevice::Cpu(Default::default()));
}

#[cfg(feature = "test-support")]
#[test]
#[ignore = "requires a CUDA device"]
fn survival_profile_fixture_cuda() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    assert!(
        count > 0,
        "survival profile CUDA fixture requires a usable CUDA device"
    );
    survival_fixture(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }));
}

/// Row-worst thresholds of `n` non-full rows: the initial `(n, +inf)`.
fn open_worst_ids(device: &CubeDevice, n: usize) -> CubeTensor {
    i32_tensor(device, &[n], vec![n as i32; n])
}

fn open_worst_distances(device: &CubeDevice, n: usize) -> CubeTensor {
    f32_tensor(device, &[n], vec![f32::INFINITY; n])
}

/// Sentinel row-worst values that no merge writes; rows without records keep them.
const UNTOUCHED_WORST_ID: i32 = -7;
const UNTOUCHED_WORST_DISTANCE: f32 = -7.5;

/// Run the backend's merge and return the row-worst arrays it leaves, starting
/// from the untouched sentinels.
fn fixture_merge(
    device: &CubeDevice,
    offsets: &CubeTensor,
    endpoint_ids: &CubeTensor,
    endpoint_distances: &CubeTensor,
    ids: &CubeTensor,
    distances: &CubeTensor,
    workspace_ids: &CubeTensor,
    workspace_distances: &CubeTensor,
    n: usize,
    width: usize,
) -> (Vec<i32>, Vec<f32>) {
    let worst_ids = i32_tensor(device, &[n], vec![UNTOUCHED_WORST_ID; n]);
    let worst_distances = f32_tensor(device, &[n], vec![UNTOUCHED_WORST_DISTANCE; n]);
    fixture_merge_into(
        device,
        offsets,
        endpoint_ids,
        endpoint_distances,
        ids,
        distances,
        workspace_ids,
        workspace_distances,
        &worst_ids,
        &worst_distances,
        width,
    );
    (read_i32(&worst_ids), read_f32(&worst_distances))
}

/// Run the backend's merge with caller-owned row-worst arrays.
fn fixture_merge_into(
    device: &CubeDevice,
    offsets: &CubeTensor,
    endpoint_ids: &CubeTensor,
    endpoint_distances: &CubeTensor,
    ids: &CubeTensor,
    distances: &CubeTensor,
    workspace_ids: &CubeTensor,
    workspace_distances: &CubeTensor,
    worst_ids: &CubeTensor,
    worst_distances: &CubeTensor,
    width: usize,
) {
    let client = device.client();
    let n = ids.shape()[0];
    let (cpu, count, dim) =
        super::super::orchestration::ann_launch_shape(&client, DType::F32.size(), n as u32)
            .unwrap()
            .geometry();
    let dtype = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ];
    if cpu {
        endpoint_merge_cpu::launch(
            &client,
            count,
            dim,
            offsets.clone().into_tensor_arg(),
            endpoint_ids.clone().into_tensor_arg(),
            endpoint_distances.clone().into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            worst_ids.clone().into_tensor_arg(),
            worst_distances.clone().into_tensor_arg(),
            width as u32,
            dtype,
        );
    } else {
        endpoint_merge_cuda::launch(
            &client,
            count,
            dim,
            offsets.clone().into_tensor_arg(),
            endpoint_ids.clone().into_tensor_arg(),
            endpoint_distances.clone().into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            workspace_ids.clone().into_tensor_arg(),
            workspace_distances.clone().into_tensor_arg(),
            worst_ids.clone().into_tensor_arg(),
            worst_distances.clone().into_tensor_arg(),
            width as u32,
            dtype,
        );
    }
}

/// Scalar oracle of the endpoint filter: IEEE comparisons, so `-0.0 == 0.0`.
fn endpoint_mask_oracle(left: usize, right: usize, distance: f32, worst: &[(i32, f32)]) -> i32 {
    let n = worst.len();
    if left >= n || right >= n || left == right || distance.is_nan() || distance < 0.0 {
        return 0;
    }
    let passes = |row: usize, candidate: usize| {
        let (id, threshold) = worst[row];
        let id = id as usize;
        id >= n || distance < threshold || (distance == threshold && candidate < id)
    };
    PRODUCED
        | if passes(left, right) { KEEP_LEFT } else { 0 }
        | if passes(right, left) { KEEP_RIGHT } else { 0 }
}

/// The endpoint predicate on exact edge values, then count and scatter honoring
/// each mask bit independently, and the profiling record counters.
fn endpoint_mask_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 4usize;
    // Row 0 full at (5, id 2); row 1 not full; row 2 full at (+inf, id 1);
    // row 3 full at (0, id 2).
    let worst = [(2, 5.0), (4, f32::INFINITY), (1, f32::INFINITY), (2, 0.0)];
    let cases: [(i32, i32, f32, i32); 16] = [
        (0, 1, 5.0, 7),           // tie, lower id kept at row 0; row 1 not full
        (0, 3, 5.0, 4),           // tie, higher id rejected at 0; worse at 3
        (0, 2, 5.0, 6),           // tie with the worst id itself is rejected
        (0, 1, f32::NAN, 0),      // NaN produces nothing, even for a non-full row
        (0, 1, -1.0, 0),          // negative produces nothing
        (0, 1, -0.0, 7),          // signed zero is a valid distance
        (3, 1, -0.0, 7),          // -0 ties +0 at row 3, lower id kept
        (3, 2, 0.0, 6),           // +0 tie with the worst id rejected at row 3
        (2, 0, f32::INFINITY, 5), // +inf tie at row 2 with lower id; worse at 0
        (2, 3, f32::INFINITY, 4), // +inf tie with higher id rejected
        (1, 2, f32::INFINITY, 5), // +inf kept by a non-full row; tie with worst id at 2
        (0, 3, 4.0, 5),           // left only
        (3, 0, 4.0, 6),           // right only
        (1, 1, 1.0, 0),           // self pair
        (4, 0, 1.0, 0),           // invalid id
        (0, 4, 1.0, 0),           // invalid id
    ];
    for &(left, right, distance, expected) in &cases {
        assert_eq!(
            endpoint_mask_oracle(left as usize, right as usize, distance, &worst),
            expected,
            "oracle left={left} right={right} distance={distance:?}"
        );
    }
    let slots = cases.len();
    let left = i32_tensor(&device, &[slots], cases.iter().map(|c| c.0).collect());
    let right = i32_tensor(&device, &[slots], cases.iter().map(|c| c.1).collect());
    let distances = f32_tensor(&device, &[slots], cases.iter().map(|c| c.2).collect());
    let worst_ids = i32_tensor(&device, &[n], worst.iter().map(|w| w.0).collect());
    let worst_distances = f32_tensor(&device, &[n], worst.iter().map(|w| w.1).collect());
    let masks = i32_tensor(&device, &[slots], vec![-1; slots]);
    let (count, dim) = launch_units(&client, slots);
    let dtypes = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ];
    endpoint_mask_probe::launch(
        &client,
        count.clone(),
        dim,
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        worst_ids.into_tensor_arg(),
        worst_distances.into_tensor_arg(),
        masks.clone().into_tensor_arg(),
        dtypes,
    );
    let expected: Vec<i32> = cases.iter().map(|c| c.3).collect();
    assert_eq!(read_i32(&masks), expected);

    let counts = i32_tensor(&device, &[n], vec![-1; n]);
    let (row_count, row_dim) = launch_units(&client, n);
    endpoint_count_zero::launch(
        &client,
        row_count,
        row_dim,
        counts.clone().into_tensor_arg(),
        dtype_to_storage_type(DType::I32),
    );
    endpoint_count_kernel::launch(
        &client,
        count.clone(),
        dim,
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        masks.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        slots as u32,
        dtype_to_storage_type(DType::I32),
    );
    let mut records: Vec<Vec<(i32, u32)>> = vec![Vec::new(); n];
    for &(l, r, d, mask) in &cases {
        if mask & KEEP_LEFT != 0 {
            records[l as usize].push((r, d.to_bits()));
        }
        if mask & KEEP_RIGHT != 0 {
            records[r as usize].push((l, d.to_bits()));
        }
    }
    let expected_counts: Vec<i32> = records.iter().map(|row| row.len() as i32).collect();
    assert_eq!(read_i32(&counts), expected_counts);
    let offsets = i32_tensor(&device, &[n + 1], vec![-1; n + 1]);
    let cursors = i32_tensor(&device, &[n], vec![-1; n]);
    let sums = i32_tensor(&device, &[n.div_ceil(PREFIX_CHUNK)], vec![0]);
    exclusive_prefix(
        &client,
        &counts,
        &offsets,
        &cursors,
        &sums,
        DType::I32,
        true,
    )
    .unwrap();
    let total: usize = records.iter().map(Vec::len).sum();
    let endpoint_ids = i32_tensor(&device, &[total], vec![-1; total]);
    let endpoint_distances = f32_tensor(&device, &[total], vec![-1.0; total]);
    endpoint_scatter_kernel::launch(
        &client,
        count,
        dim,
        left.into_tensor_arg(),
        right.into_tensor_arg(),
        distances.into_tensor_arg(),
        masks.clone().into_tensor_arg(),
        cursors.into_tensor_arg(),
        endpoint_ids.clone().into_tensor_arg(),
        endpoint_distances.clone().into_tensor_arg(),
        slots as u32,
        dtypes,
    );
    let offsets_host = read_i32(&offsets);
    let ids_host = read_i32(&endpoint_ids);
    let distances_host = read_f32(&endpoint_distances);
    for (row, expected_row) in records.iter_mut().enumerate() {
        let range = offsets_host[row] as usize..offsets_host[row + 1] as usize;
        let mut actual: Vec<(i32, u32)> = range
            .map(|position| (ids_host[position], distances_host[position].to_bits()))
            .collect();
        actual.sort_unstable();
        expected_row.sort_unstable();
        assert_eq!(&actual, expected_row, "row {row}");
    }
    let profile = i32_tensor(&device, &[1], vec![-1]);
    let produced = cases.iter().filter(|c| c.3 & PRODUCED != 0).count() as u64;
    assert_eq!(
        profile_tile_records(&client, &profile, &masks, &offsets, n, slots as u32).unwrap(),
        (2 * produced, total as u64)
    );
}

/// Forest pairs of one leaf holding every row, staged in tiles through the
/// filtered pipeline with persistent thresholds, against the unfiltered scalar
/// merge. Rows become full mid-phase, so later tiles see tightened (and, within
/// a tile, stale) thresholds. A second pass of the same pairs is all duplicates
/// and must leave the heap and thresholds unchanged.
fn filtered_pipeline_fixture(device: CubeDevice) {
    let client = device.client();
    // Ties (3, 3 and equal spacing), zero distances, +inf from the extremes and
    // a NaN row whose records are all rejected.
    let values = [0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 6.0, 3.0e38, -3.0e38, f32::NAN];
    let n = values.len();
    let width = 3usize;
    let pairs = n * (n - 1) / 2;
    let mut proposals = Vec::new();
    for left in 0..n {
        for right in left + 1..n {
            let difference = values[left] - values[right];
            proposals.push((left, right, difference * difference));
        }
    }
    let oracle = merge_proposals(
        &vec![Vec::new(); n],
        &proposals,
        &vec![BTreeSet::new(); n],
        width,
    );
    let data = f32_tensor(&device, &[n, 1], values.to_vec());
    let forest_rows = i32_tensor(&device, &[n], (0..n as i32).collect());
    let leaf_rows = i32_tensor(&device, &[2], vec![0, n as i32]);
    let pair_prefix = i64_tensor(&device, &[2], vec![0, pairs as i64]);
    let dtypes = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ];
    for tile in [1usize, 4, 7, pairs] {
        let ids = i32_tensor(&device, &[n, width], vec![n as i32; n * width]);
        let distances = f32_tensor(&device, &[n, width], vec![f32::INFINITY; n * width]);
        let workspace_ids = i32_tensor(&device, &[n, width], vec![n as i32; n * width]);
        let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
        let worst_ids = i32_tensor(&device, &[n], vec![-1; n]);
        let worst_distances = f32_tensor(&device, &[n], vec![-1.0; n]);
        let (row_count, row_dim) = launch_units(&client, n);
        row_worst_refresh_kernel::launch(
            &client,
            row_count.clone(),
            row_dim,
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            worst_ids.clone().into_tensor_arg(),
            worst_distances.clone().into_tensor_arg(),
            width as u32,
            dtypes,
        );
        assert_eq!(read_i32(&worst_ids), vec![n as i32; n]);
        let mut kept_per_pass = Vec::new();
        let mut snapshots = Vec::new();
        for _pass in 0..2 {
            let mut kept = 0usize;
            let mut offset = 0usize;
            while offset < pairs {
                let slots = tile.min(pairs - offset);
                let left = i32_tensor(&device, &[slots], vec![-1; slots]);
                let right = i32_tensor(&device, &[slots], vec![-1; slots]);
                let pair_distances = f32_tensor(&device, &[slots], vec![-1.0; slots]);
                let valid = i32_tensor(&device, &[slots], vec![-1; slots]);
                let (count, dim) = launch_units(&client, slots);
                forest_pair_slots_kernel::launch(
                    &client,
                    count.clone(),
                    dim,
                    data.clone().into_tensor_arg(),
                    forest_rows.clone().into_tensor_arg(),
                    leaf_rows.clone().into_tensor_arg(),
                    pair_prefix.clone().into_tensor_arg(),
                    worst_ids.clone().into_tensor_arg(),
                    worst_distances.clone().into_tensor_arg(),
                    left.clone().into_tensor_arg(),
                    right.clone().into_tensor_arg(),
                    pair_distances.clone().into_tensor_arg(),
                    valid.clone().into_tensor_arg(),
                    offset as u64,
                    slots as u32,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                        dtype_to_storage_type(DType::I64),
                    ],
                );
                let counts = i32_tensor(&device, &[n], vec![-1; n]);
                endpoint_count_zero::launch(
                    &client,
                    row_count.clone(),
                    row_dim,
                    counts.clone().into_tensor_arg(),
                    dtype_to_storage_type(DType::I32),
                );
                endpoint_count_kernel::launch(
                    &client,
                    count.clone(),
                    dim,
                    left.clone().into_tensor_arg(),
                    right.clone().into_tensor_arg(),
                    valid.clone().into_tensor_arg(),
                    counts.clone().into_tensor_arg(),
                    slots as u32,
                    dtype_to_storage_type(DType::I32),
                );
                let offsets = i32_tensor(&device, &[n + 1], vec![-1; n + 1]);
                let cursors = i32_tensor(&device, &[n], vec![-1; n]);
                let sums = i32_tensor(&device, &[n.div_ceil(PREFIX_CHUNK)], vec![0]);
                exclusive_prefix(
                    &client,
                    &counts,
                    &offsets,
                    &cursors,
                    &sums,
                    DType::I32,
                    true,
                )
                .unwrap();
                let total = read_i32(&offsets)[n] as usize;
                kept += total;
                let staged = (2 * slots).max(1);
                let endpoint_ids = i32_tensor(&device, &[staged], vec![-1; staged]);
                let endpoint_distances = f32_tensor(&device, &[staged], vec![-1.0; staged]);
                endpoint_scatter_kernel::launch(
                    &client,
                    count,
                    dim,
                    left.into_tensor_arg(),
                    right.into_tensor_arg(),
                    pair_distances.into_tensor_arg(),
                    valid.into_tensor_arg(),
                    cursors.into_tensor_arg(),
                    endpoint_ids.clone().into_tensor_arg(),
                    endpoint_distances.clone().into_tensor_arg(),
                    slots as u32,
                    dtypes,
                );
                fixture_merge_into(
                    &device,
                    &offsets,
                    &endpoint_ids,
                    &endpoint_distances,
                    &ids,
                    &distances,
                    &workspace_ids,
                    &workspace_distances,
                    &worst_ids,
                    &worst_distances,
                    width,
                );
                offset += slots;
            }
            kept_per_pass.push(kept);
            snapshots.push((
                read_i32(&ids),
                read_f32(&distances)
                    .iter()
                    .map(|d| d.to_bits())
                    .collect::<Vec<_>>(),
                read_i32(&worst_ids),
                read_f32(&worst_distances)
                    .iter()
                    .map(|d| d.to_bits())
                    .collect::<Vec<_>>(),
            ));
        }
        let (heap_ids, heap_bits, worst_id_values, worst_bits) = &snapshots[0];
        for row in 0..n {
            let expected = &oracle.rows[row];
            for slot in 0..width {
                let index = row * width + slot;
                let (id, bits) = expected
                    .get(slot)
                    .map_or((n as i32, f32::INFINITY.to_bits()), |neighbor| {
                        (neighbor.id as i32, neighbor.distance.to_bits())
                    });
                assert_eq!(
                    (heap_ids[index], heap_bits[index]),
                    (id, bits),
                    "tile={tile} row={row} slot={slot}"
                );
            }
            // Every row's threshold is its heap's last entry (the NaN row
            // keeps the refreshed not-full value).
            let last = row * width + width - 1;
            assert_eq!(
                worst_id_values[row], heap_ids[last],
                "tile={tile} row={row}"
            );
            assert_eq!(worst_bits[row], heap_bits[last], "tile={tile} row={row}");
        }
        assert_eq!(
            snapshots[0], snapshots[1],
            "tile={tile}: duplicates changed state"
        );
        // A single tile sees only the initial not-full thresholds and keeps all.
        let produced = 2 * (pairs - (n - 1));
        if tile == pairs {
            assert_eq!(kept_per_pass[0], produced);
        } else {
            assert!(
                kept_per_pass[0] < produced,
                "tile={tile}: filter kept every produced record"
            );
        }
        assert!(
            kept_per_pass[1] < kept_per_pass[0],
            "tile={tile}: duplicate pass was not filtered"
        );
    }
}

fn merge_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 3usize;
    let width = 1usize;
    let source_ids = i32_tensor(&device, &[n, width], vec![1, 0, 1]);
    // Inject proposals for pair 0-2 at 5, then pair 0-1 at 1.
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 2, 3, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 1, 0, 0]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![5.0, 1.0, 1.0, 5.0]);
    let output_ids = i32_tensor(&device, &[n, width], vec![1, 0, 1]);
    let output_distances = f32_tensor(&device, &[n, width], vec![10.0, 10.0, 9.0]);
    let workspace_ids = i32_tensor(&device, &[n, width], vec![n as i32; n * width]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let worst = fixture_merge(
        &device,
        &offsets,
        &endpoint_ids,
        &endpoint_distances,
        &output_ids,
        &output_distances,
        &workspace_ids,
        &workspace_distances,
        n,
        width,
    );
    assert_eq!(worst, (vec![1, 0, 0], vec![1.0, 1.0, 5.0]));
    assert_eq!(read_i32(&output_ids), [1, 0, 0]);
    assert_eq!(read_f32(&output_distances), [1.0, 1.0, 5.0]);
    assert_eq!(read_i32(&source_ids), [1, 0, 1]);

    let source_flags = i32_tensor(&device, &[n, width], vec![1; n * width]);
    let sampled = i32_tensor(&device, &[n, 2], vec![3; n * 2]);
    let output_flags = i32_tensor(&device, &[n, width], vec![0; n * width]);
    let new_count = i32_tensor(&device, &[1], vec![0]);
    let (count, dim) = launch_units(&client, n);
    finalize_flags_kernel::launch(
        &client,
        count,
        dim,
        source_ids.into_tensor_arg(),
        source_flags.into_tensor_arg(),
        sampled.into_tensor_arg(),
        output_ids.clone().into_tensor_arg(),
        output_flags.clone().into_tensor_arg(),
        new_count.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        1,
        dtype_to_storage_type(DType::I32),
    );
    assert_eq!(read_i32(&output_flags), [1, 1, 0]);
    assert_eq!(read_i32(&new_count), [1]);
}

fn duplicate_merge_fixture(device: CubeDevice) {
    let n = 3usize;
    let width = 2usize;
    // The same pair arrives twice in one merge batch; each endpoint retains one id at min distance.
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 2, 2, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 2, 0, 0]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![5.0, 4.0, 5.0, 4.0]);
    let output_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let output_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let workspace_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let worst = fixture_merge(
        &device,
        &offsets,
        &endpoint_ids,
        &endpoint_distances,
        &output_ids,
        &output_distances,
        &workspace_ids,
        &workspace_distances,
        n,
        width,
    );
    // Row 1 has no records and keeps its threshold.
    assert_eq!(
        worst,
        (
            vec![3, UNTOUCHED_WORST_ID, 3],
            vec![f32::MAX, UNTOUCHED_WORST_DISTANCE, f32::MAX]
        )
    );
    assert_eq!(read_i32(&output_ids), [2, 3, 3, 3, 0, 3]);
    assert_eq!(
        read_f32(&output_distances),
        [4.0, f32::MAX, f32::MAX, f32::MAX, 4.0, f32::MAX]
    );
}

/// A worse duplicate after an improving one keeps the improved distance, on a
/// non-full row (0) and a full row (1), and the row thresholds follow the heap.
fn worse_duplicate_merge_fixture(device: CubeDevice) {
    let n = 3usize;
    let width = 2usize;
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 2, 4, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 2, 2, 2]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![3.0, 8.0, 1.0, 6.0]);
    let output_ids = i32_tensor(&device, &[n, width], vec![3, 3, 0, 2, 3, 3]);
    let output_distances = f32_tensor(
        &device,
        &[n, width],
        vec![
            f32::INFINITY,
            f32::INFINITY,
            2.0,
            5.0,
            f32::INFINITY,
            f32::INFINITY,
        ],
    );
    let workspace_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let worst = fixture_merge(
        &device,
        &offsets,
        &endpoint_ids,
        &endpoint_distances,
        &output_ids,
        &output_distances,
        &workspace_ids,
        &workspace_distances,
        n,
        width,
    );
    assert_eq!(read_i32(&output_ids), [2, 3, 2, 0, 3, 3]);
    let bits = |values: Vec<f32>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(read_f32(&output_distances)),
        bits(vec![
            3.0,
            f32::INFINITY,
            1.0,
            2.0,
            f32::INFINITY,
            f32::INFINITY
        ])
    );
    assert_eq!(worst.0, [3, 0, UNTOUCHED_WORST_ID]);
    assert_eq!(
        bits(worst.1),
        bits(vec![f32::INFINITY, 2.0, UNTOUCHED_WORST_DISTANCE])
    );
}

fn untouched_merge_rows_fixture(device: CubeDevice) {
    let n = 4usize;
    let width = 1usize;
    // Row 0 has no records; row 1 receives only a worse proposal. Row 2 has
    // one accepted proposal and row 3 has two, exercising both CUDA toggles.
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 0, 1, 2, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 1, 1, 2]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![2.0, 0.5, 0.75, 0.25]);
    let output_ids = i32_tensor(&device, &[n, width], vec![1, 0, 0, 0]);
    let output_distances = f32_tensor(&device, &[n, width], vec![1.0, 1.0, 1.0, 1.0]);
    let workspace_ids = i32_tensor(&device, &[n, width], vec![n as i32; n]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n]);
    let worst = fixture_merge(
        &device,
        &offsets,
        &endpoint_ids,
        &endpoint_distances,
        &output_ids,
        &output_distances,
        &workspace_ids,
        &workspace_distances,
        n,
        width,
    );
    // Row 0 has no records; row 1's rejected record still rewrites its unchanged worst.
    assert_eq!(
        worst,
        (
            vec![UNTOUCHED_WORST_ID, 0, 1, 2],
            vec![UNTOUCHED_WORST_DISTANCE, 1.0, 0.5, 0.25]
        )
    );
    assert_eq!(read_i32(&output_ids), [1, 0, 1, 2]);
    assert_eq!(read_f32(&output_distances), [1.0, 1.0, 0.5, 0.25]);
}

fn tiled_and_zero_slot_iteration_fixture(device: CubeDevice) {
    let dispatch = burn::tensor::Device::new(burn::backend::DispatchDevice::Cube(device.clone()));
    let budget = crate::memory::MemoryBudget {
        total_bytes: u64::MAX,
        free_bytes: u64::MAX,
        limit_bytes: None,
    };
    for (n, k, iterations) in [(2usize, 1usize, 3u32), (8, 2, 3)] {
        let values: Vec<f32> = (0..n).map(|i| (i * i + 3 * i) as f32).collect();
        let plan = super::super::orchestration::ann_preflight(&dispatch, n, 1, k, &budget).unwrap();
        let mut tiled = plan;
        tiled.tile_slots = 1;
        let (expected_ids, expected_distances) = approximate_neighbors(
            f32_tensor(&device, &[n, 1], values.clone()),
            k,
            42,
            iterations,
            true,
            plan,
        )
        .unwrap();
        let (actual_ids, actual_distances) = approximate_neighbors(
            f32_tensor(&device, &[n, 1], values.clone()),
            k,
            42,
            iterations,
            true,
            tiled,
        )
        .unwrap();
        assert_eq!(read_i32(&actual_ids), read_i32(&expected_ids));
        assert_eq!(
            read_f32(&actual_distances)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            read_f32(&expected_distances)
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        // Each row has only one possible neighbor, so its disjoint new/old
        // pools cannot contain a pair; every refinement iteration has zero slots.
        if n == 2 {
            let (initial_ids, initial_distances) =
                approximate_neighbors(f32_tensor(&device, &[n, 1], values), k, 42, 0, true, plan)
                    .unwrap();
            assert_eq!(read_i32(&actual_ids), read_i32(&initial_ids));
            assert_eq!(
                read_f32(&actual_distances)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                read_f32(&initial_distances)
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
}

fn sampled_pair_slots_fixture(device: CubeDevice) {
    let client = device.client();
    let data = f32_tensor(&device, &[5, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    // [new(1,2), old(3,4)]: one i<j pair then four new×old pairs.
    let sampled_ids = i32_tensor(&device, &[2, 4], vec![1, 2, 3, 4, 1, 2, 3, 4]);
    let slots = 10usize;
    let left = i32_tensor(&device, &[slots], vec![0; slots]);
    let right = i32_tensor(&device, &[slots], vec![0; slots]);
    let distances = f32_tensor(&device, &[slots], vec![0.0; slots]);
    let valid = i32_tensor(&device, &[slots], vec![0; slots]);
    let pool_sizes = i32_tensor(&device, &[4], vec![2, 2, 2, 2]);
    let center_offsets = i64_tensor(&device, &[3], vec![0, 5, 10]);
    let (count, dim) = launch_units(&client, slots);
    pair_slots_kernel::launch(
        &client,
        count,
        dim,
        data.clone().into_tensor_arg(),
        sampled_ids.into_tensor_arg(),
        pool_sizes.into_tensor_arg(),
        center_offsets.into_tensor_arg(),
        open_worst_ids(&device, 5).into_tensor_arg(),
        open_worst_distances(&device, 5).into_tensor_arg(),
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        0,
        slots as u32,
        2,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
            dtype_to_storage_type(DType::I64),
        ],
    );
    assert_eq!(read_i32(&left), [1, 1, 1, 2, 2, 1, 1, 1, 2, 2]);
    assert_eq!(read_i32(&right), [2, 3, 4, 3, 4, 2, 3, 4, 3, 4]);
    assert_eq!(read_i32(&valid), [PRODUCED | KEEP_LEFT | KEEP_RIGHT; 10]);
    assert_eq!(
        read_f32(&distances),
        [1.0, 4.0, 9.0, 1.0, 4.0, 1.0, 4.0, 9.0, 1.0, 4.0]
    );

    // Uneven pools are compacted: center 0 has new=[1], old=[3,4] (two cross
    // pairs); center 1 has new=[1,2], old=[3] (one i<j pair, two cross pairs).
    // The tile starts mid-stream at flattened slot 1.
    let sampled_ids = i32_tensor(&device, &[2, 4], vec![1, 5, 3, 4, 1, 2, 3, 5]);
    let pool_sizes = i32_tensor(&device, &[4], vec![1, 2, 2, 1]);
    let center_offsets = i64_tensor(&device, &[3], vec![0, 2, 5]);
    let slots = 4usize;
    let left = i32_tensor(&device, &[slots], vec![0; slots]);
    let right = i32_tensor(&device, &[slots], vec![0; slots]);
    let distances = f32_tensor(&device, &[slots], vec![0.0; slots]);
    let valid = i32_tensor(&device, &[slots], vec![0; slots]);
    let (count, dim) = launch_units(&client, slots);
    pair_slots_kernel::launch(
        &client,
        count,
        dim,
        data.into_tensor_arg(),
        sampled_ids.into_tensor_arg(),
        pool_sizes.into_tensor_arg(),
        center_offsets.into_tensor_arg(),
        open_worst_ids(&device, 5).into_tensor_arg(),
        open_worst_distances(&device, 5).into_tensor_arg(),
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        1,
        slots as u32,
        2,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
            dtype_to_storage_type(DType::I64),
        ],
    );
    assert_eq!(read_i32(&left), [1, 1, 1, 2]);
    assert_eq!(read_i32(&right), [4, 2, 3, 3]);
    assert_eq!(read_i32(&valid), [PRODUCED | KEEP_LEFT | KEEP_RIGHT; 4]);
    assert_eq!(read_f32(&distances), [9.0, 1.0, 4.0, 1.0]);
}

fn forest_pair_slots_tiled_fixture(device: CubeDevice) {
    let client = device.client();
    let data = f32_tensor(&device, &[4, 1], vec![0.0, 1.0, 2.0, 3.0]);
    let forest_rows = i32_tensor(&device, &[4], vec![0, 1, 2, 3]);
    // Membership starts and cumulative pair prefix for two leaves plus terminal:
    // [0,1,2] has three pairs; [3] is a zero-pair singleton.
    let leaf_rows = i32_tensor(&device, &[3], vec![0, 3, 4]);
    let pair_prefix = i64_tensor(&device, &[3], vec![0, 3, 3]);
    let slots = 2usize;
    let left = i32_tensor(&device, &[slots], vec![0; slots]);
    let right = i32_tensor(&device, &[slots], vec![0; slots]);
    let distances = f32_tensor(&device, &[slots], vec![0.0; slots]);
    let valid = i32_tensor(&device, &[slots], vec![0; slots]);
    let (count, dim) = launch_units(&client, slots);
    forest_pair_slots_kernel::launch(
        &client,
        count.clone(),
        dim,
        data.into_tensor_arg(),
        forest_rows.into_tensor_arg(),
        leaf_rows.into_tensor_arg(),
        pair_prefix.into_tensor_arg(),
        open_worst_ids(&device, 4).into_tensor_arg(),
        open_worst_distances(&device, 4).into_tensor_arg(),
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        1,
        slots as u32,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
            dtype_to_storage_type(DType::I64),
        ],
    );
    assert_eq!(read_i32(&left), [0, 1]);
    assert_eq!(read_i32(&right), [2, 2]);
    assert_eq!(read_f32(&distances), [4.0, 1.0]);
    assert_eq!(read_i32(&valid), [PRODUCED | KEEP_LEFT | KEEP_RIGHT; 2]);

    // The two staged pairs fan out symmetrically to their four endpoints.
    let cursors = i32_tensor(&device, &[4], vec![0, 1, 2, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![0; 4]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![0.0; 4]);
    endpoint_scatter_kernel::launch(
        &client,
        count.clone(),
        dim,
        left.into_tensor_arg(),
        right.into_tensor_arg(),
        distances.into_tensor_arg(),
        valid.into_tensor_arg(),
        cursors.into_tensor_arg(),
        endpoint_ids.clone().into_tensor_arg(),
        endpoint_distances.clone().into_tensor_arg(),
        slots as u32,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    let endpoint_ids = read_i32(&endpoint_ids);
    let endpoint_distances = read_f32(&endpoint_distances);
    assert_eq!(&endpoint_ids[..2], [2, 2]);
    assert_eq!(&endpoint_distances[..2], [4.0, 1.0]);
    let mut center_two = endpoint_ids[2..]
        .iter()
        .copied()
        .zip(endpoint_distances[2..].iter().copied())
        .collect::<Vec<_>>();
    center_two.sort_unstable_by_key(|&(id, _)| id);
    assert_eq!(center_two, [(0, 4.0), (1, 1.0)]);
}

/// Padding contract for the large-offset fixtures: offsets are monotone, padding
/// entries (zero pools / zero membership) absorb the gaps, and no tile lands in a
/// padding entry or past the terminal total.
fn sampled_pair_slots_large_offset_fixture(device: CubeDevice) {
    let client = device.client();
    let two31 = 1i64 << 31;
    let two32 = 1i64 << 32;
    let data = f32_tensor(&device, &[5, 1], vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    // Centers 1, 2, 4, 5: new=[1,2], old=[3,4] (5 slots); centers 0 and 3 are padding.
    let mut ids = Vec::new();
    let mut sizes = Vec::new();
    for center in 0..6 {
        if center == 0 || center == 3 {
            ids.extend([5, 5, 5, 5]);
            sizes.extend([0, 0]);
        } else {
            ids.extend([1, 2, 3, 4]);
            sizes.extend([2, 2]);
        }
    }
    let sampled_ids = i32_tensor(&device, &[6, 4], ids);
    let pool_sizes = i32_tensor(&device, &[12], sizes);
    let offsets = vec![
        0,
        two31 - 3,
        two31 + 2,
        two31 + 7,
        two32 - 2,
        two32 + 3,
        two32 + 8,
    ];
    let center_offsets = i64_tensor(&device, &[7], offsets);
    let pools = SampledPools {
        new: vec![1, 2],
        old: vec![3, 4],
    };
    let center_pairs = local_pairs(&pools, 0, 5);
    let expected: Vec<(usize, usize)> = center_pairs.iter().chain(&center_pairs).copied().collect();
    for start in [two31 - 3, two32 - 2] {
        let slots = 10usize;
        let left = i32_tensor(&device, &[slots], vec![0; slots]);
        let right = i32_tensor(&device, &[slots], vec![0; slots]);
        let distances = f32_tensor(&device, &[slots], vec![0.0; slots]);
        let valid = i32_tensor(&device, &[slots], vec![0; slots]);
        let (count, dim) = launch_units(&client, slots);
        pair_slots_kernel::launch(
            &client,
            count,
            dim,
            data.clone().into_tensor_arg(),
            sampled_ids.clone().into_tensor_arg(),
            pool_sizes.clone().into_tensor_arg(),
            center_offsets.clone().into_tensor_arg(),
            open_worst_ids(&device, 5).into_tensor_arg(),
            open_worst_distances(&device, 5).into_tensor_arg(),
            left.clone().into_tensor_arg(),
            right.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            valid.clone().into_tensor_arg(),
            start as u64,
            slots as u32,
            2,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
                dtype_to_storage_type(DType::I64),
            ],
        );
        let left = read_i32(&left);
        let right = read_i32(&right);
        let got: Vec<(usize, usize)> = left
            .iter()
            .zip(&right)
            .map(|(&l, &r)| (l as usize, r as usize))
            .collect();
        assert_eq!(got, expected, "start={start}");
        assert_eq!(
            read_i32(&valid),
            [PRODUCED | KEEP_LEFT | KEEP_RIGHT; 10],
            "start={start}"
        );
        let want: Vec<f32> = expected
            .iter()
            .map(|&(l, r)| ((r - l) * (r - l)) as f32)
            .collect();
        assert_eq!(read_f32(&distances), want, "start={start}");
    }
}

fn forest_pair_slots_large_offset_fixture(device: CubeDevice) {
    let client = device.client();
    let two31 = 1i64 << 31;
    let two32 = 1i64 << 32;
    let data = f32_tensor(&device, &[12, 1], (0..12).map(|v| v as f32).collect());
    let forest_rows = i32_tensor(&device, &[12], (0..12).collect());
    // Leaves 1, 2, 4, 5 hold three rows (three pairs); leaves 0 and 3 are padding
    // with zero membership whose prefix gaps no tile touches.
    let leaf_rows = i32_tensor(&device, &[7], vec![0, 0, 3, 6, 6, 9, 12]);
    let prefix = vec![
        0,
        two31 - 3,
        two31,
        two31 + 3,
        two32 - 2,
        two32 + 1,
        two32 + 4,
    ];
    let pair_prefix = i64_tensor(&device, &[7], prefix);
    // Tile A crosses 2^31 at the 1->2 boundary; tile B crosses 2^32 inside leaf 4.
    for (start, first_row) in [(two31 - 3, 0i32), (two32 - 2, 6)] {
        let slots = 6usize;
        let left = i32_tensor(&device, &[slots], vec![-1; slots]);
        let right = i32_tensor(&device, &[slots], vec![-1; slots]);
        let distances = f32_tensor(&device, &[slots], vec![-1.0; slots]);
        let valid = i32_tensor(&device, &[slots], vec![-1; slots]);
        let (count, dim) = launch_units(&client, slots);
        forest_pair_slots_kernel::launch(
            &client,
            count,
            dim,
            data.clone().into_tensor_arg(),
            forest_rows.clone().into_tensor_arg(),
            leaf_rows.clone().into_tensor_arg(),
            pair_prefix.clone().into_tensor_arg(),
            open_worst_ids(&device, 12).into_tensor_arg(),
            open_worst_distances(&device, 12).into_tensor_arg(),
            left.clone().into_tensor_arg(),
            right.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            valid.clone().into_tensor_arg(),
            start as u64,
            slots as u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
                dtype_to_storage_type(DType::I64),
            ],
        );
        let a = first_row;
        let b = first_row + 3;
        assert_eq!(read_i32(&left), [a, a, a + 1, b, b, b + 1], "start={start}");
        assert_eq!(read_i32(&right), [a + 1, a + 2, a + 2, b + 1, b + 2, b + 2]);
        assert_eq!(read_f32(&distances), [1.0, 4.0, 1.0, 1.0, 4.0, 1.0]);
        assert_eq!(read_i32(&valid), [PRODUCED | KEEP_LEFT | KEEP_RIGHT; 6]);
    }
}

/// Run the production reverse CSR (count, prefix, tagged scatter) and the pool
/// kernel on a heap whose rows hold each valid id at most once, then compare
/// every row with the scalar `sample_pools` oracle. The pool kernel is rerun on
/// the reverse CSR with each target's entries reversed: scatter order must not
/// change the pools.
fn check_sampled_pools(
    device: &CubeDevice,
    n: usize,
    width: usize,
    capacity: usize,
    candidate_values: &[i32],
    flag_values: &[i32],
    label: &str,
) {
    let client = device.client();
    for row in 0..n {
        let ids = &candidate_values[row * width..(row + 1) * width];
        let valid: BTreeSet<_> = ids.iter().filter(|&&id| (id as usize) < n).collect();
        let valid_count = ids.iter().filter(|&&id| (id as usize) < n).count();
        assert_eq!(valid.len(), valid_count, "{label}: row {row} repeats an id");
    }
    let candidates = i32_tensor(device, &[n, width], candidate_values.to_vec());
    let flags = i32_tensor(device, &[n, width], flag_values.to_vec());
    let counts = i32_tensor(device, &[n], vec![-1; n]);
    let offsets = i32_tensor(device, &[n + 1], vec![-1; n + 1]);
    let cursors = i32_tensor(device, &[n], vec![-1; n]);
    let sums = i32_tensor(
        device,
        &[n.div_ceil(PREFIX_CHUNK)],
        vec![0; n.div_ceil(PREFIX_CHUNK)],
    );
    let sources = i32_tensor(device, &[n * width], vec![-1; n * width]);
    let (count, dim) = launch_units(&client, n);
    reverse_zero::launch(
        &client,
        count.clone(),
        dim,
        counts.clone().into_tensor_arg(),
        dtype_to_storage_type(DType::I32),
    );
    reverse_count::launch(
        &client,
        count.clone(),
        dim,
        candidates.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        dtype_to_storage_type(DType::I32),
    );
    exclusive_prefix(
        &client,
        &counts,
        &offsets,
        &cursors,
        &sums,
        DType::I32,
        true,
    )
    .unwrap();
    reverse_scatter::launch(
        &client,
        count.clone(),
        dim,
        candidates.clone().into_tensor_arg(),
        flags.clone().into_tensor_arg(),
        cursors.into_tensor_arg(),
        sources.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        dtype_to_storage_type(DType::I32),
    );
    let offset_values = read_i32(&offsets);
    let source_values = read_i32(&sources);
    // Every reverse entry is a distinct source whose slot holds the target, tagged
    // exactly when that slot is new.
    let mut reverse = vec![Vec::new(); n];
    for target in 0..n {
        let range = offset_values[target] as usize..offset_values[target + 1] as usize;
        let mut entries: Vec<u32> = source_values[range].iter().map(|&v| v as u32).collect();
        entries.sort_unstable();
        let mut expected = Vec::new();
        for source in 0..n {
            for slot in 0..width {
                let index = source * width + slot;
                if candidate_values[index] as usize == target && source != target {
                    let tag = if flag_values[index] == 0 {
                        REVERSE_NEW_BIT
                    } else {
                        0
                    };
                    expected.push(source as u32 | tag);
                    reverse[target].push(Candidate {
                        id: source,
                        is_new: flag_values[index] == 0,
                    });
                }
            }
        }
        expected.sort_unstable();
        assert_eq!(entries, expected, "{label}: reverse entries of {target}");
    }
    let mut reversed = source_values.clone();
    for target in 0..n {
        reversed[offset_values[target] as usize..offset_values[target + 1] as usize].reverse();
    }
    let mut runs = Vec::new();
    for order in [source_values, reversed] {
        let sources = i32_tensor(device, &[order.len()], order);
        let sampled_ids = i32_tensor(device, &[n, capacity * 2], vec![-1; n * capacity * 2]);
        sampled_pool_kernel::launch(
            &client,
            count.clone(),
            dim,
            candidates.clone().into_tensor_arg(),
            flags.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            sources.into_tensor_arg(),
            sampled_ids.clone().into_tensor_arg(),
            n as u32,
            width as u32,
            capacity as u32,
            43,
            7,
            dtype_to_storage_type(DType::I32),
        );
        runs.push(read_i32(&sampled_ids));
    }
    assert_eq!(runs[0], runs[1], "{label}: reverse order changed the pools");
    for row in 0..n {
        let forward = (0..width)
            .map(|slot| Candidate {
                id: candidate_values[row * width + slot] as usize,
                is_new: flag_values[row * width + slot] == 0,
            })
            .collect::<Vec<_>>();
        let expected = sample_pools(row, n, &forward, &reverse[row], capacity, 43, 7);
        let mut expected_ids = vec![n as i32; capacity * 2];
        for (class, pool) in [(0usize, &expected.new), (1, &expected.old)] {
            for (slot, &id) in pool.iter().enumerate() {
                expected_ids[class * capacity + slot] = id as i32;
            }
        }
        let base = row * capacity * 2;
        assert_eq!(
            &runs[0][base..base + capacity * 2],
            expected_ids,
            "{label}: sampled IDs row {row}"
        );
    }
}

fn sampled_pool_fixture(device: CubeDevice) {
    // Ids repeat across rows and disagree on class between directions.
    check_sampled_pools(
        &device,
        4,
        2,
        2,
        &[1, 2, 0, 3, 0, 3, 0, 2],
        &[1, 0, 0, 1, 0, 1, 1, 0],
        "small",
    );
}

/// Sampled-pool scalar agreement at candidate widths straddling the sampling budget.
/// Capacity is the production `MAX_CANDIDATES` budget; widths 59..65 and 120/256 exercise
/// both pool-saturated and pool-sparse rows, with some padding ids.
fn sampled_pool_width_edge_fixture(device: CubeDevice) {
    let n = 300usize;
    let capacity = MAX_CANDIDATES.min(n - 1);
    for width in [59usize, 60, 61, 64, 65, 120, 256] {
        let mut candidate_values = vec![0i32; n * width];
        let mut flag_values = vec![0i32; n * width];
        for row in 0..n {
            for slot in 0..width {
                // Offsets 1 + 7*slot mod 299 are distinct and never zero.
                let id = if row % 5 == 0 && slot + 1 == width {
                    n
                } else {
                    (row + 1 + (slot * 7) % (n - 1)) % n
                };
                candidate_values[row * width + slot] = id as i32;
                flag_values[row * width + slot] = i32::from((row + slot) % 3 == 0);
            }
        }
        check_sampled_pools(
            &device,
            n,
            width,
            capacity,
            &candidate_values,
            &flag_values,
            &format!("width {width}"),
        );
    }
}

/// Hub row 0 with a new set larger than the pool. Its forward new ids are
/// ordered by priority so that every overflowing offer is accepted with an
/// eviction (`improving`) or rejected (`worsening`). Each forward id's own row
/// holds 0 as old, so the old reverse loop offers every new id again and must
/// exclude it even after eviction or rejection; the last forward slot is an old
/// id whose row holds 0 as new. Row `n - 1` is the source of a tagged entry.
fn hub_pool_fixture(device: CubeDevice) {
    let n = 16usize;
    let width = 7usize;
    let capacity = 2usize;
    let hub_new = 6usize;
    let mut by_priority: Vec<usize> = (1..n).collect();
    by_priority.sort_by_key(|&id| (priority(43, 7, 0, id, true), id));
    for improving in [true, false] {
        // The old id has the best priority, so its reverse new offer is accepted
        // too in the evicting case.
        let old_id = by_priority[0];
        let mut chosen: Vec<usize> = by_priority[1..hub_new].to_vec();
        if improving {
            chosen.reverse();
        }
        let mut candidate_values = vec![n as i32; n * width];
        let mut flag_values = vec![1i32; n * width];
        for (slot, &id) in chosen.iter().enumerate() {
            candidate_values[slot] = id as i32;
            flag_values[slot] = 0;
        }
        candidate_values[width - 1] = old_id as i32;
        flag_values[width - 1] = 1;
        for row in 1..n {
            let base = row * width;
            let mut slot = 0usize;
            if chosen.contains(&row) {
                candidate_values[base] = 0;
                flag_values[base] = 1;
                slot = 1;
            } else if row == old_id {
                candidate_values[base] = 0;
                flag_values[base] = 0;
                slot = 1;
            }
            let mut id = 1usize;
            while slot < width {
                if id != row && !chosen.contains(&id) && id != old_id {
                    candidate_values[base + slot] = id as i32;
                    flag_values[base + slot] = i32::from((row + slot).is_multiple_of(2));
                    slot += 1;
                }
                id += 1;
            }
        }
        check_sampled_pools(
            &device,
            n,
            width,
            capacity,
            &candidate_values,
            &flag_values,
            if improving {
                "hub evicting"
            } else {
                "hub rejecting"
            },
        );
    }
    // A full new pool offered only duplicates: rows 1 and 2 hold 0 as new and
    // row 0 holds them as new, so the reverse offers repeat pool members. Old id
    // 3 is then answered from the complete pool; row 4 is all padding.
    #[rustfmt::skip]
    let candidate_values = [
        1, 2, 3,
        0, 3, 4,
        0, 1, 4,
        0, 4, 5,
        5, 5, 5,
    ];
    #[rustfmt::skip]
    let flag_values = [
        0, 0, 1,
        0, 1, 1,
        0, 0, 1,
        1, 1, 0,
        1, 1, 1,
    ];
    check_sampled_pools(
        &device,
        5,
        3,
        2,
        &candidate_values,
        &flag_values,
        "duplicates only",
    );
}

fn flag_transition_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 3usize;
    let width = 2usize;
    let source_ids = i32_tensor(&device, &[n, width], vec![1, 2, 3, 3, 3, 3]);
    let source_flags = i32_tensor(&device, &[n, width], vec![0, 0, 0, 0, 0, 0]);
    let output_ids = source_ids.clone();
    // A sampled source-new edge can be sampled from its reverse endpoint too.
    let sampled = i32_tensor(&device, &[n, 2], vec![3, 3, 0, 3, 3, 3]);
    let output_flags = i32_tensor(&device, &[n, width], vec![0; n * width]);
    let new_count = i32_tensor(&device, &[1], vec![0]);
    let (count, dim) = launch_units(&client, n);
    finalize_flags_kernel::launch(
        &client,
        count,
        dim,
        source_ids.into_tensor_arg(),
        source_flags.into_tensor_arg(),
        sampled.into_tensor_arg(),
        output_ids.into_tensor_arg(),
        output_flags.clone().into_tensor_arg(),
        new_count.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        1,
        dtype_to_storage_type(DType::I32),
    );
    assert_eq!(read_i32(&output_flags), [1, 0, 1, 1, 1, 1]);
    assert_eq!(read_i32(&new_count), [0]);
}

fn squared_output_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 2usize;
    let width = 2usize;
    let k = 2usize;
    let indices = i32_tensor(&device, &[n, width], vec![1, 0, 0, 1]);
    // Zero, ordinary, infinite and ordinary squared distances.
    let squared = f32_tensor(&device, &[n, width], vec![0.0, 4.0, f32::INFINITY, 9.0]);
    // Squared output keeps the raw id and infinite distance for the unrepresentable slot.
    // Unsquared output maps that slot to the documented sentinel (n, sqrt(f32::MAX)), the
    // existing UMAP contract matching exact unsquared output. Each mode is asserted separately.
    let sentinel_distance = f32::MAX.sqrt();
    for (squared_output, expected_ids, expected) in [
        (true, [1, 0, 0, 1], vec![0.0, 4.0, f32::INFINITY, 9.0]),
        (false, [1, 0, 2, 1], vec![0.0, 2.0, sentinel_distance, 3.0]),
    ] {
        let output_indices = i32_tensor(&device, &[n, k], vec![0; n * k]);
        let output_distances = f32_tensor(&device, &[n, k], vec![0.0; n * k]);
        let (count, dim) = launch_units(&client, n);
        ann_output::launch(
            &client,
            count,
            dim,
            indices.clone().into_tensor_arg(),
            squared.clone().into_tensor_arg(),
            output_indices.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            width as u32,
            k as u32,
            squared_output,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        assert_eq!(
            read_i32(&output_indices),
            expected_ids,
            "squared_output={squared_output} ids"
        );
        assert_eq!(
            read_f32(&output_distances),
            expected,
            "squared_output={squared_output} distances"
        );
    }
}

fn i64_tensor(device: &CubeDevice, shape: &[usize], values: Vec<i64>) -> CubeTensor {
    raw_tensor(device, shape, values, DType::I64)
}

fn read_i64(tensor: &CubeTensor) -> Vec<i64> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| i64::from_ne_bytes(*bytes))
        .collect()
}

/// Spike kernel: one lane per slot writes `offset + slot` in the 64-bit prefix type.
#[cube(launch)]
fn wide_offset_probe_kernel<P: Int>(
    output: &mut Tensor<P>,
    slot_offset: u64,
    #[define(P)] _dtype: ElemType,
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < output.shape(0) {
        output[slot] = P::cast_from(slot_offset) + P::cast_from(slot as u32);
    }
}

/// Spike kernel mirroring the planned 64-bit `pair_slots_kernel` search: largest
/// `center` with `offsets[center] <= flattened` (compared in P), then the local
/// slot `flattened - offsets[center]` narrowed to u32.
#[cube(launch)]
fn wide_search_probe_kernel<P: Int, I: Int>(
    offsets: &Tensor<P>,
    centers: &mut Tensor<I>,
    locals: &mut Tensor<I>,
    slot_offset: u64,
    slots: u32,
    #[define(P, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize {
        let flattened = P::cast_from(slot_offset) + P::cast_from(slot as u32);
        let mut low = 0usize;
        let mut high = offsets.shape(0) - 1;
        while high - low > 1 {
            let middle = (low + high) / 2;
            if offsets[middle] <= flattened {
                low = middle;
            } else {
                high = middle;
            }
        }
        let local = u32::cast_from(flattened - offsets[low]);
        centers[slot] = I::cast_from(low as u32);
        locals[slot] = I::cast_from(local);
    }
}

fn host_prefix(values: &[u64]) -> Vec<u64> {
    let mut running = 0u64;
    let mut out = Vec::with_capacity(values.len() + 1);
    for &value in values {
        out.push(running);
        running += value;
    }
    out.push(running);
    out
}

/// I64 exclusive prefix over 3 full chunks plus a partial chunk, with partial sums
/// crossing 2^31 and 2^32 inside chunks and across chunk boundaries.
fn exclusive_prefix_i64_fixture(device: CubeDevice) {
    let client = device.client();
    let len = 300usize;
    let values: Vec<u64> = (0..len)
        .map(|index| match index {
            5 => 1_500_000_000,
            127 => 3_000_000_000,
            130 => 1_500_000_000,
            260 => 3_000_000_000,
            _ => (index as u64 * 7) % 13,
        })
        .collect();
    let expected = host_prefix(&values);
    assert!(expected[len] > 1 << 33);
    let chunks = len.div_ceil(PREFIX_CHUNK);
    for seed_cursors in [true, false] {
        let input = i64_tensor(&device, &[len], values.iter().map(|&v| v as i64).collect());
        let offsets = i64_tensor(&device, &[len + 1], vec![-1; len + 1]);
        let cursor_len = if seed_cursors { len } else { 1 };
        let cursors = i64_tensor(&device, &[cursor_len], vec![-7; cursor_len]);
        let sums = i64_tensor(&device, &[chunks], vec![0; chunks]);
        exclusive_prefix(
            &client,
            &input,
            &offsets,
            &cursors,
            &sums,
            DType::I64,
            seed_cursors,
        )
        .unwrap();
        let offsets = read_i64(&offsets);
        assert_eq!(
            offsets,
            expected.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            "seed_cursors={seed_cursors}"
        );
        let cursors = read_i64(&cursors);
        if seed_cursors {
            assert_eq!(cursors, offsets[..len]);
        } else {
            assert_eq!(cursors, vec![-7], "placeholder must be untouched");
        }
    }
}

/// The I32 path is unchanged by the dtype parameter.
fn i32_prefix_regression_fixture(device: CubeDevice) {
    let client = device.client();
    let len = 300usize;
    let values: Vec<u64> = (0..len).map(|index| (index as u64 * 5) % 11).collect();
    let expected = host_prefix(&values);
    let chunks = len.div_ceil(PREFIX_CHUNK);
    let input = i32_tensor(&device, &[len], values.iter().map(|&v| v as i32).collect());
    let offsets = i32_tensor(&device, &[len + 1], vec![-1; len + 1]);
    let cursors = i32_tensor(&device, &[len], vec![-1; len]);
    let sums = i32_tensor(&device, &[chunks], vec![0; chunks]);
    exclusive_prefix(&client, &input, &offsets, &cursors, &sums, DType::I32, true).unwrap();
    let expected: Vec<i32> = expected.iter().map(|&v| v as i32).collect();
    assert_eq!(read_i32(&offsets), expected);
    assert_eq!(read_i32(&cursors), expected[..len]);
}

/// A `u64` scalar launch argument reaches the kernel intact above 2^32.
fn wide_offset_scalar_fixture(device: CubeDevice) {
    let client = device.client();
    for offset in [(1u64 << 32) - 2, (1u64 << 33) + 7] {
        let slots = 6usize;
        let output = i64_tensor(&device, &[slots], vec![0; slots]);
        let (count, dim) = launch_units(&client, slots);
        wide_offset_probe_kernel::launch(
            &client,
            count,
            dim,
            output.clone().into_tensor_arg(),
            offset,
            dtype_to_storage_type(DType::I64),
        );
        let expected: Vec<i64> = (0..slots as u64)
            .map(|slot| (offset + slot) as i64)
            .collect();
        assert_eq!(read_i64(&output), expected, "offset={offset}");
    }
}

/// 64-bit binary search with padding intervals 0 and 3; real intervals 1, 2, 4, 5
/// hold 5 slots each. Tiles cross 2^31 (1->2) and 2^32 (4->5).
fn i64_search_fixture(device: CubeDevice) {
    let client = device.client();
    let two31 = 1u64 << 31;
    let two32 = 1u64 << 32;
    let offsets = [
        0,
        two31 - 3,
        two31 + 2,
        two31 + 7,
        two32 - 2,
        two32 + 3,
        two32 + 8,
    ];
    let table = i64_tensor(
        &device,
        &[offsets.len()],
        offsets.iter().map(|&v| v as i64).collect(),
    );
    for start in [two31 - 3, two32 - 2] {
        let slots = 10usize;
        let centers = i32_tensor(&device, &[slots], vec![-1; slots]);
        let locals = i32_tensor(&device, &[slots], vec![-1; slots]);
        let (count, dim) = launch_units(&client, slots);
        wide_search_probe_kernel::launch(
            &client,
            count,
            dim,
            table.clone().into_tensor_arg(),
            centers.clone().into_tensor_arg(),
            locals.clone().into_tensor_arg(),
            start,
            slots as u32,
            [
                dtype_to_storage_type(DType::I64),
                dtype_to_storage_type(DType::I32),
            ],
        );
        let (mut want_centers, mut want_locals) = (Vec::new(), Vec::new());
        for slot in 0..slots as u64 {
            let flattened = start + slot;
            let center = offsets.iter().rposition(|&o| o <= flattened).unwrap();
            assert!(center != 0 && center != 3 && center < offsets.len() - 1);
            want_centers.push(center as i32);
            want_locals.push((flattened - offsets[center]) as i32);
        }
        assert!(want_locals.iter().all(|&local| local < 5));
        assert_eq!(read_i32(&centers), want_centers, "start={start}");
        assert_eq!(read_i32(&locals), want_locals, "start={start}");
    }
}

fn run_wide_prefix_fixtures(device: CubeDevice) {
    exclusive_prefix_i64_fixture(device.clone());
    i32_prefix_regression_fixture(device.clone());
    wide_offset_scalar_fixture(device.clone());
    i64_search_fixture(device);
}

#[test]
fn wide_prefix_spike_cpu() {
    run_wide_prefix_fixtures(CubeDevice::Cpu(Default::default()));
}

#[test]
fn wide_prefix_spike_cuda() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    if count == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    run_wide_prefix_fixtures(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }));
}

fn run_kernel_fixtures(device: CubeDevice) {
    squared_output_fixture(device.clone());
    merge_fixture(device.clone());
    duplicate_merge_fixture(device.clone());
    untouched_merge_rows_fixture(device.clone());
    worse_duplicate_merge_fixture(device.clone());
    endpoint_mask_fixture(device.clone());
    filtered_pipeline_fixture(device.clone());
    tiled_and_zero_slot_iteration_fixture(device.clone());
    sampled_pair_slots_fixture(device.clone());
    forest_pair_slots_tiled_fixture(device.clone());
    sampled_pair_slots_large_offset_fixture(device.clone());
    forest_pair_slots_large_offset_fixture(device.clone());
    sampled_pool_fixture(device.clone());
    sampled_pool_width_edge_fixture(device.clone());
    hub_pool_fixture(device.clone());
    flag_transition_fixture(device);
}

#[test]
fn sampled_kernels_cpu_match_injected_contract_fixtures() {
    run_kernel_fixtures(CubeDevice::Cpu(Default::default()));
}

#[test]
fn sampled_kernels_cuda_match_injected_contract_fixtures() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    if count == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    run_kernel_fixtures(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }));
}

#[test]
fn scalar_fixture_expected_values_are_explicit() {
    let pools = SampledPools {
        new: vec![1, 2, 1, 0, 99],
        old: vec![2, 3, 3, 1],
    };
    assert_eq!(local_pairs(&pools, 0, 5), vec![(1, 2), (1, 3), (2, 3)]);

    let result = merge_proposals(
        &[
            vec![Neighbor {
                id: 1,
                distance: 10.0,
                is_new: false,
            }],
            vec![Neighbor {
                id: 0,
                distance: 10.0,
                is_new: false,
            }],
            vec![Neighbor {
                id: 1,
                distance: 9.0,
                is_new: false,
            }],
        ],
        &[(0, 2, 5.0), (0, 1, 1.0)],
        &vec![BTreeSet::new(); 3],
        1,
    );
    assert_eq!(
        result.rows.iter().map(|row| row[0].id).collect::<Vec<_>>(),
        [1, 0, 0]
    );
    assert_eq!(
        result
            .rows
            .iter()
            .map(|row| row[0].distance)
            .collect::<Vec<_>>(),
        [1.0, 1.0, 5.0]
    );
    assert_eq!(
        result
            .rows
            .iter()
            .map(|row| row[0].is_new)
            .collect::<Vec<_>>(),
        [false, false, true]
    );
    assert_eq!(result.surviving_additions, 1);
}
