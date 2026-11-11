use super::*;

use super::super::ann_forest::MAX_CANDIDATES;
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
        super::super::orchestration::ann_launch_shape(&client, DType::F32.size(), n as u32)
            .unwrap()
            .geometry();
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
        super::super::orchestration::ann_launch_shape(&client, DType::F32.size(), n as u32)
            .unwrap()
            .geometry();
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
    assert_eq!(read_i32(&valid), [1; 4]);
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
        assert_eq!(read_i32(&valid), [1; 10], "start={start}");
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
        assert_eq!(read_i32(&valid), [1; 6]);
    }
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

/// Sampled-pool scalar agreement at candidate widths straddling the sampling budget.
/// Capacity is the production `MAX_CANDIDATES` budget; widths 59..65 and 120/256 exercise
/// both pool-saturated and pool-sparse rows. Pools are compared against `sample_pools`.
fn sampled_pool_width_edge_fixture(device: CubeDevice) {
    let client = device.client();
    let n = 200usize;
    let capacity = MAX_CANDIDATES.min(n - 1);
    for width in [59usize, 60, 61, 64, 65, 120, 256] {
        let mut candidate_values = vec![0i32; n * width];
        let mut flag_values = vec![0i32; n * width];
        for row in 0..n {
            for slot in 0..width {
                let mut id = (row * 7 + slot * 13 + 1) % n;
                if id == row {
                    id = (id + 1) % n;
                }
                candidate_values[row * width + slot] = id as i32;
                flag_values[row * width + slot] = i32::from((row + slot) % 3 == 0);
            }
        }
        // Reverse CSR grouped by target, sources ascending, matching the scalar reference order.
        let mut reverse_offsets = vec![0i32; n + 1];
        let mut reverse_sources = Vec::with_capacity(n * width);
        for target in 0..n {
            for (index, &candidate) in candidate_values.iter().enumerate() {
                if candidate as usize == target {
                    reverse_sources.push((index / width) as i32);
                }
            }
            reverse_offsets[target + 1] = reverse_sources.len() as i32;
        }
        let candidates = i32_tensor(&device, &[n, width], candidate_values.clone());
        let flags = i32_tensor(&device, &[n, width], flag_values.clone());
        let offsets = i32_tensor(&device, &[n + 1], reverse_offsets.clone());
        let sources = i32_tensor(&device, &[reverse_sources.len()], reverse_sources.clone());
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
            candidates.into_tensor_arg(),
            flags.into_tensor_arg(),
            offsets.into_tensor_arg(),
            sources.into_tensor_arg(),
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
                "width {width} sampled IDs row {row}"
            );
            assert_eq!(
                &actual_priorities[base..base + capacity * 2],
                expected_priorities,
                "width {width} sampled priority row {row}"
            );
        }
    }
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
    sampled_pair_slots_fixture(device.clone());
    forest_pair_slots_tiled_fixture(device.clone());
    sampled_pair_slots_large_offset_fixture(device.clone());
    forest_pair_slots_large_offset_fixture(device.clone());
    sampled_pool_fixture(device.clone());
    sampled_pool_width_edge_fixture(device.clone());
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
