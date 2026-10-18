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

fn read_i32(tensor: &CubeTensor) -> Vec<i32> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .chunks_exact(4)
        .map(|bytes| i32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_f32(tensor: &CubeTensor) -> Vec<f32> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn launch_units(client: &cubecl::client::Client, units: usize) -> (CubeCount, CubeDim) {
    ann_independent_launch(client, u32::try_from(units).unwrap())
}

fn merge_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 3usize;
    let width = 1usize;
    let source_ids = i32_tensor(&device, &[n, width], vec![1, 0, 1]);
    let source_distances = f32_tensor(&device, &[n, width], vec![10.0, 10.0, 9.0]);
    // Inject proposals for pair 0-2 at 5, then pair 0-1 at 1.
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 2, 3, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 1, 0, 0]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![5.0, 1.0, 1.0, 5.0]);
    let output_ids = i32_tensor(&device, &[n, width], vec![n as i32; n * width]);
    let output_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let workspace_ids = i32_tensor(&device, &[n, width], vec![n as i32; n * width]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let (cpu_merge, count, dim) =
        super::super::ann_launch_geometry(&client, DType::F32.size(), n as u32);
    endpoint_merge_kernel::launch(
        &client,
        count,
        dim,
        source_ids.clone().into_tensor_arg(),
        source_distances.clone().into_tensor_arg(),
        offsets.into_tensor_arg(),
        endpoint_ids.into_tensor_arg(),
        endpoint_distances.into_tensor_arg(),
        output_ids.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        workspace_ids.into_tensor_arg(),
        workspace_distances.into_tensor_arg(),
        width as u32,
        cpu_merge,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    assert_eq!(read_i32(&output_ids), [1, 0, 0]);
    assert_eq!(read_f32(&output_distances), [1.0, 1.0, 5.0]);

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
    let client = device.client();
    let n = 3usize;
    let width = 2usize;
    let source_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let source_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    // The same pair arrives twice in one merge batch; each endpoint retains one id at min distance.
    let offsets = i32_tensor(&device, &[n + 1], vec![0, 2, 2, 4]);
    let endpoint_ids = i32_tensor(&device, &[4], vec![2, 2, 0, 0]);
    let endpoint_distances = f32_tensor(&device, &[4], vec![5.0, 4.0, 5.0, 4.0]);
    let output_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let output_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let workspace_ids = i32_tensor(&device, &[n, width], vec![3; n * width]);
    let workspace_distances = f32_tensor(&device, &[n, width], vec![f32::MAX; n * width]);
    let (cpu_merge, count, dim) =
        super::super::ann_launch_geometry(&client, DType::F32.size(), n as u32);
    endpoint_merge_kernel::launch(
        &client,
        count,
        dim,
        source_ids.into_tensor_arg(),
        source_distances.into_tensor_arg(),
        offsets.into_tensor_arg(),
        endpoint_ids.into_tensor_arg(),
        endpoint_distances.into_tensor_arg(),
        output_ids.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        workspace_ids.into_tensor_arg(),
        workspace_distances.into_tensor_arg(),
        width as u32,
        cpu_merge,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    assert_eq!(read_i32(&output_ids), [2, 3, 3, 3, 0, 3]);
    assert_eq!(
        read_f32(&output_distances),
        [4.0, f32::MAX, f32::MAX, f32::MAX, 4.0, f32::MAX]
    );
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
    let center_offsets = i32_tensor(&device, &[3], vec![0, 5, 10]);
    let (count, dim) = launch_units(&client, slots);
    pair_slots_kernel::launch(
        &client,
        count,
        dim,
        data.clone().into_tensor_arg(),
        sampled_ids.into_tensor_arg(),
        pool_sizes.into_tensor_arg(),
        center_offsets.into_tensor_arg(),
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
        ],
    );
    assert_eq!(read_i32(&left), [1, 1, 1, 2, 2, 1, 1, 1, 2, 2]);
    assert_eq!(read_i32(&right), [2, 3, 4, 3, 4, 2, 3, 4, 3, 4]);
    assert_eq!(read_i32(&valid), [1; 10]);
    assert_eq!(
        read_f32(&distances),
        [1.0, 4.0, 9.0, 1.0, 4.0, 1.0, 4.0, 9.0, 1.0, 4.0]
    );

    // Uneven pools are compacted: center 0 has new=[1], old=[3,4] (two cross
    // pairs); center 1 has new=[1,2], old=[3] (one i<j pair, two cross pairs).
    // The tile starts mid-stream at flattened slot 1.
    let sampled_ids = i32_tensor(&device, &[2, 4], vec![1, 5, 3, 4, 1, 2, 3, 5]);
    let pool_sizes = i32_tensor(&device, &[4], vec![1, 2, 2, 1]);
    let center_offsets = i32_tensor(&device, &[3], vec![0, 2, 5]);
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
        ],
    );
    assert_eq!(read_i32(&left), [1, 1, 1, 2]);
    assert_eq!(read_i32(&right), [4, 2, 3, 3]);
    assert_eq!(read_i32(&valid), [1; 4]);
    assert_eq!(read_f32(&distances), [9.0, 1.0, 4.0, 1.0]);
}

fn forest_pair_slots_tiled_fixture(device: CubeDevice) {
    let client = device.client();
    let data = f32_tensor(&device, &[4, 1], vec![0.0, 1.0, 2.0, 3.0]);
    let forest_rows = i32_tensor(&device, &[4], vec![0, 1, 2, 3]);
    // Interleaved (membership_start, cumulative_pair_prefix) for three leaves:
    // [0,1,2] has three pairs; [3] is a zero-pair singleton.
    let leaf_offsets = i32_tensor(&device, &[3, 2], vec![0, 0, 3, 3, 4, 3]);
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
        leaf_offsets.into_tensor_arg(),
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        1,
        slots as u32,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    assert_eq!(read_i32(&left), [0, 1]);
    assert_eq!(read_i32(&right), [2, 2]);
    assert_eq!(read_f32(&distances), [4.0, 1.0]);
    assert_eq!(read_i32(&valid), [1, 1]);

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

fn sampled_pool_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 4usize;
    let width = 2usize;
    let capacity = 2usize;
    // Forward and reverse sources intentionally repeat ids and disagree on class.
    let candidate_values = vec![1, 2, 0, 3, 0, 3, 0, 2];
    let flag_values = vec![1, 0, 0, 1, 0, 1, 1, 0];
    let candidates = i32_tensor(&device, &[n, width], candidate_values.clone());
    let flags = i32_tensor(&device, &[n, width], flag_values.clone());
    let reverse_offsets = i32_tensor(&device, &[n + 1], vec![0, 3, 4, 6, 8]);
    let reverse_sources = i32_tensor(&device, &[8], vec![1, 2, 3, 0, 0, 3, 1, 2]);
    let sampled_ids = i32_tensor(
        &device,
        &[n, capacity * 2],
        vec![n as i32; n * capacity * 2],
    );
    let sampled_priorities = i32_tensor(
        &device,
        &[n, capacity * 2],
        vec![i32::MAX; n * capacity * 2],
    );
    let (count, dim) = launch_units(&client, n);
    sampled_pool_kernel::launch(
        &client,
        count,
        dim,
        candidates.clone().into_tensor_arg(),
        flags.clone().into_tensor_arg(),
        reverse_offsets.clone().into_tensor_arg(),
        reverse_sources.clone().into_tensor_arg(),
        sampled_ids.clone().into_tensor_arg(),
        sampled_priorities.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        capacity as u32,
        43,
        7,
        dtype_to_storage_type(DType::I32),
    );
    let actual_ids = read_i32(&sampled_ids);
    let actual_priorities = read_i32(&sampled_priorities);

    // Build the scalar expectation from explicitly enumerated forward/reverse entries.
    for row in 0..n {
        let forward = (0..width)
            .map(|slot| Candidate {
                id: candidate_values[row * width + slot] as usize,
                is_new: flag_values[row * width + slot] == 0,
            })
            .collect::<Vec<_>>();
        let mut reverse = Vec::new();
        for source in 0..n {
            for slot in 0..width {
                let index = source * width + slot;
                if candidate_values[index] as usize == row {
                    reverse.push(Candidate {
                        id: source,
                        is_new: flag_values[index] == 0,
                    });
                }
            }
        }
        let expected = sample_pools(row, n, &forward, &reverse, capacity, 43, 7);
        let base = row * capacity * 2;
        let mut expected_ids = vec![n as i32; capacity * 2];
        let mut expected_priorities = vec![i32::MAX; capacity * 2];
        for (class, pool) in [(0usize, &expected.new), (1, &expected.old)] {
            for (slot, &id) in pool.iter().enumerate() {
                expected_ids[class * capacity + slot] = id as i32;
                expected_priorities[class * capacity + slot] =
                    priority(43, 7, row, id, class == 0) as i32;
            }
        }
        assert_eq!(
            &actual_ids[base..base + capacity * 2],
            expected_ids,
            "sampled IDs row {row}"
        );
        assert_eq!(
            &actual_priorities[base..base + capacity * 2],
            expected_priorities,
            "sampled priority row {row}"
        );
    }

    let reordered_sources = i32_tensor(&device, &[8], vec![3, 2, 1, 0, 3, 0, 2, 1]);
    let reordered_ids = i32_tensor(
        &device,
        &[n, capacity * 2],
        vec![n as i32; n * capacity * 2],
    );
    let reordered_priorities = i32_tensor(
        &device,
        &[n, capacity * 2],
        vec![i32::MAX; n * capacity * 2],
    );
    let (count, dim) = launch_units(&client, n);
    sampled_pool_kernel::launch(
        &client,
        count,
        dim,
        candidates.into_tensor_arg(),
        flags.into_tensor_arg(),
        reverse_offsets.into_tensor_arg(),
        reordered_sources.into_tensor_arg(),
        reordered_ids.clone().into_tensor_arg(),
        reordered_priorities.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        capacity as u32,
        43,
        7,
        dtype_to_storage_type(DType::I32),
    );
    assert_eq!(
        read_i32(&reordered_ids),
        actual_ids,
        "reverse CSR scatter order must not bias pool selection"
    );
    assert_eq!(read_i32(&reordered_priorities), actual_priorities);
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

fn run_kernel_fixtures(device: CubeDevice) {
    merge_fixture(device.clone());
    duplicate_merge_fixture(device.clone());
    sampled_pair_slots_fixture(device.clone());
    forest_pair_slots_tiled_fixture(device.clone());
    sampled_pool_fixture(device.clone());
    flag_transition_fixture(device);
}

#[cfg(feature = "cpu")]
#[test]
fn sampled_kernels_cpu_match_injected_contract_fixtures() {
    run_kernel_fixtures(CubeDevice::Cpu(Default::default()));
}

#[cfg(feature = "cuda")]
#[test]
fn sampled_kernels_cuda_match_injected_contract_fixtures() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    if count == 0 {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
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
