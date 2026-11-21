use super::super::*;
use super::*;
use burn_cubecl::CubeDevice;

fn tensor<T: burn::tensor::Element>(
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
fn ints(device: &CubeDevice, values: Vec<i32>) -> CubeTensor {
    tensor(device, &[values.len()], values, DType::I32)
}
fn floats(device: &CubeDevice, values: Vec<f32>) -> CubeTensor {
    tensor(device, &[values.len()], values, DType::F32)
}
fn read_i32(t: &CubeTensor) -> Vec<i32> {
    t.client
        .read_one_unchecked(t.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| i32::from_ne_bytes(*v))
        .collect()
}
fn read_f32(t: &CubeTensor) -> Vec<f32> {
    t.client
        .read_one_unchecked(t.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_ne_bytes(*v))
        .collect()
}
fn geometry(device: &CubeDevice, count: usize) -> (CubeCount, CubeDim) {
    ann_independent_launch(&device.client(), count as u32).unwrap()
}
fn dtypes() -> [ElemType; 2] {
    [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ]
}
fn wide_dtypes() -> [ElemType; 3] {
    [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
        dtype_to_storage_type(DType::I64),
    ]
}

fn owner_generate_case(ch: usize, partitions: usize) {
    let device = CubeDevice::Cpu(Default::default());
    let client = device.client();
    let n = 8usize;
    let capacity = 4usize;
    let pools: Vec<(Vec<i32>, Vec<i32>)> = vec![
        (vec![], vec![]),
        (vec![], vec![1, 2, 3, 4]),
        (vec![0], vec![1]),
        (vec![0, 1], vec![2, 3]),
        (vec![0, 1, 2, 3], vec![4, 5, 8, 7]),
        (vec![8, 2], vec![2, 4]),
        (vec![6], vec![]),
        (vec![3, 4], vec![5]),
    ];
    let mut ids = Vec::new();
    let mut sizes = Vec::new();
    let mut offsets = vec![0i64];
    for (new, old) in &pools {
        sizes.extend([new.len() as i32, old.len() as i32]);
        ids.extend(new.iter().copied());
        ids.resize(ids.len() + capacity - new.len(), n as i32);
        ids.extend(old.iter().copied());
        ids.resize(ids.len() + capacity - old.len(), n as i32);
        offsets.push(
            offsets.last().unwrap()
                + (new.len() * new.len().saturating_sub(1) / 2 + new.len() * old.len()) as i64,
        );
    }
    let slots = *offsets.last().unwrap() as usize;
    let data = tensor(
        &device,
        &[n, 1],
        (0..n).map(|x| x as f32).collect(),
        DType::F32,
    );
    let sampled = ints(&device, ids);
    let sizes = ints(&device, sizes);
    let offsets = tensor(&device, &[n + 1], offsets, DType::I64);
    let worst_ids = ints(&device, vec![8, 8, 1, 2, 8, 4, 8, 8]);
    let worst_distances = floats(
        &device,
        vec![
            f32::MAX,
            f32::MAX,
            1.0,
            1.0,
            f32::MAX,
            1.0,
            f32::MAX,
            f32::MAX,
        ],
    );
    let rows = ints(&device, vec![-1; slots * 2]);
    let out_ids = ints(&device, vec![-1; slots * 2]);
    let out_dist = floats(&device, vec![-1.0; slots * 2]);
    let order = ints(&device, vec![-1; slots * 2]);
    let chunks = n.div_ceil(ch);
    let chunk_offsets = ints(&device, vec![-1; chunks * (D_MAX + 1)]);
    let profile = tensor(&device, &[2], vec![0u64; 2], DType::U64);
    let (count, dim) = geometry(&device, chunks);
    owner_join_generate::launch(
        &client,
        count,
        dim,
        1 as VectorSize,
        data.clone().into_tensor_arg(),
        sampled.clone().into_tensor_arg(),
        sizes.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        worst_ids.clone().into_tensor_arg(),
        worst_distances.clone().into_tensor_arg(),
        rows.clone().into_tensor_arg(),
        out_ids.clone().into_tensor_arg(),
        out_dist.clone().into_tensor_arg(),
        order.clone().into_tensor_arg(),
        chunk_offsets.clone().into_tensor_arg(),
        profile.into_tensor_arg(),
        n as u32,
        capacity as u32,
        1,
        0,
        n as u32,
        partitions as u32,
        ch,
        1,
        false,
        false,
        wide_dtypes(),
    );
    let left = ints(&device, vec![-1; slots]);
    let right = ints(&device, vec![-1; slots]);
    let dist = floats(&device, vec![-1.0; slots]);
    let masks = ints(&device, vec![-1; slots]);
    let (count, dim) = geometry(&device, slots);
    pair_slots_kernel::launch(
        &client,
        count,
        dim,
        1 as VectorSize,
        data.into_tensor_arg(),
        sampled.into_tensor_arg(),
        sizes.into_tensor_arg(),
        offsets.into_tensor_arg(),
        worst_ids.into_tensor_arg(),
        worst_distances.into_tensor_arg(),
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        dist.clone().into_tensor_arg(),
        masks.clone().into_tensor_arg(),
        n as u32,
        1,
        0,
        slots as u32,
        capacity as u32,
        1,
        wide_dtypes(),
    );
    let (left, right, dist, masks) = (
        read_i32(&left),
        read_i32(&right),
        read_f32(&dist),
        read_i32(&masks),
    );
    let mut expected = Vec::new();
    for i in 0..slots {
        if masks[i] & KEEP_LEFT != 0 {
            expected.push((left[i], right[i], dist[i].to_bits()));
        }
        if masks[i] & KEEP_RIGHT != 0 {
            expected.push((right[i], left[i], dist[i].to_bits()));
        }
    }
    let (rows, ids, distances, order, offsets) = (
        read_i32(&rows),
        read_i32(&out_ids),
        read_f32(&out_dist),
        read_i32(&order),
        read_i32(&chunk_offsets),
    );
    let mut actual = Vec::new();
    for chunk in 0..chunks {
        let boundaries = &offsets[chunk * (D_MAX + 1)..chunk * (D_MAX + 1) + partitions + 1];
        assert!(boundaries.windows(2).all(|w| w[0] <= w[1]));
        for p in 0..partitions {
            for pos in boundaries[p]..boundaries[p + 1] {
                let r = order[pos as usize] as usize;
                assert_eq!(rows[r] as usize / n.div_ceil(partitions), p);
                actual.push((rows[r], ids[r], distances[r].to_bits()));
            }
        }
    }
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

#[test]
fn owner_generate_matches_slot_multiset_across_chunk_and_partition_sizes() {
    for ch in [1, 7, 16, 64] {
        for partitions in [1, 3, 8] {
            owner_generate_case(ch, partitions);
        }
    }
}

#[test]
fn owner_forest_generates_leaf_pairs_and_partition_offsets() {
    let device = CubeDevice::Cpu(Default::default());
    let client = device.client();
    let n = 8usize;
    // Skip the first leaf to exercise a nonzero prefix base; include a
    // singleton, a full x4 batch, a tail, and a repeated pair across leaves.
    let leaves = [vec![7, 6], vec![0], vec![0, 1, 2, 3, 4, 5], vec![0, 1, 7]];
    let mut membership = Vec::new();
    let mut leaf_offsets = vec![0i32];
    let mut prefixes = vec![0i64];
    for leaf in &leaves {
        membership.extend(leaf.iter().copied());
        leaf_offsets.push(membership.len() as i32);
        prefixes.push(
            prefixes.last().unwrap() + (leaf.len() * leaf.len().saturating_sub(1) / 2) as i64,
        );
    }
    let slots = (prefixes[4] - prefixes[1]) as usize;
    let data = floats(&device, (0..n).map(|x| x as f32).collect());
    let rows = ints(&device, vec![-1; slots * 2]);
    let ids = ints(&device, vec![-1; slots * 2]);
    let distances = floats(&device, vec![-1.0; slots * 2]);
    let order = ints(&device, vec![-1; slots * 2]);
    let offsets = ints(&device, vec![-1; 2 * (D_MAX + 1)]);
    let profile = tensor(&device, &[2], vec![0u64; 2], DType::U64);
    let (count, dim) = geometry(&device, 2);
    owner_join_generate::launch(
        &client,
        count,
        dim,
        1 as VectorSize,
        data.into_tensor_arg(),
        ints(&device, membership).into_tensor_arg(),
        ints(&device, leaf_offsets).into_tensor_arg(),
        tensor(&device, &[prefixes.len()], prefixes, DType::I64).into_tensor_arg(),
        ints(&device, vec![n as i32; n]).into_tensor_arg(),
        floats(&device, vec![f32::INFINITY; n]).into_tensor_arg(),
        rows.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        order.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        profile.into_tensor_arg(),
        n as u32,
        0,
        1,
        1,
        4,
        3,
        2,
        1,
        true,
        false,
        wide_dtypes(),
    );
    let (rows, ids, distances, order, offsets) = (
        read_i32(&rows),
        read_i32(&ids),
        read_f32(&distances),
        read_i32(&order),
        read_i32(&offsets),
    );
    let mut actual = Vec::new();
    for chunk in 0..2 {
        let boundaries = &offsets[chunk * (D_MAX + 1)..chunk * (D_MAX + 1) + 4];
        assert!(boundaries.windows(2).all(|w| w[0] <= w[1]));
        for p in 0..3 {
            for pos in boundaries[p]..boundaries[p + 1] {
                let index = order[pos as usize] as usize;
                assert_eq!(rows[index] as usize / n.div_ceil(3), p);
                actual.push((rows[index], ids[index], distances[index].to_bits()));
            }
        }
    }
    let mut expected = Vec::new();
    for leaf in &leaves[1..] {
        for i in 0..leaf.len() {
            for j in i + 1..leaf.len() {
                let (a, b) = (leaf[i], leaf[j]);
                let d = ((a - b) as f32).powi(2).to_bits();
                expected.push((a, b, d));
                expected.push((b, a, d));
            }
        }
    }
    actual.sort_unstable();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

// Each row is sorted by (distance,id), with one entry per ID and sentinel padding.
fn top_h(n: usize, width: usize, row: usize, entries: &[(i32, f32)]) -> Vec<(i32, f32)> {
    let mut unique: Vec<(i32, f32)> = Vec::new();
    for &(id, distance) in entries {
        if id < 0 || id as usize >= n || id as usize == row || distance.is_nan() || distance < 0.0 {
            continue;
        }
        if let Some(old) = unique.iter_mut().find(|x| x.0 == id) {
            if distance < old.1 {
                old.1 = distance;
            }
        } else {
            unique.push((id, distance));
        }
    }
    unique.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    unique.truncate(width);
    unique.resize(width, (n as i32, f32::INFINITY));
    unique
}
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn apply_case(
    initial: &[Vec<(i32, f32)>],
    records: &[(usize, i32, f32)],
    chunks: usize,
    partitions: usize,
    compare_slots: bool,
) {
    let device = CubeDevice::Cpu(Default::default());
    let client = device.client();
    let n = initial.len();
    let width = initial[0].len();
    let flat_ids: Vec<i32> = initial.iter().flatten().map(|v| v.0).collect();
    let flat_dist: Vec<f32> = initial.iter().flatten().map(|v| v.1).collect();
    let output_ids = tensor(&device, &[n, width], flat_ids.clone(), DType::I32);
    let output_dist = tensor(&device, &[n, width], flat_dist.clone(), DType::F32);
    let worst_ids = ints(&device, initial.iter().map(|r| r[width - 1].0).collect());
    let worst_dist = floats(&device, initial.iter().map(|r| r[width - 1].1).collect());
    let rr = ints(&device, records.iter().map(|r| r.0 as i32).collect());
    let ri = ints(&device, records.iter().map(|r| r.1).collect());
    let rd = floats(&device, records.iter().map(|r| r.2).collect());
    let mut order = Vec::new();
    let mut offsets = vec![0i32; chunks * (D_MAX + 1)];
    for chunk in 0..chunks {
        let start = records.len() * chunk / chunks;
        let end = records.len() * (chunk + 1) / chunks;
        for p in 0..partitions {
            offsets[chunk * (D_MAX + 1) + p] = order.len() as i32;
            for (i, record) in records.iter().enumerate().take(end).skip(start) {
                if record.0 / n.div_ceil(partitions) == p {
                    order.push(i as i32);
                }
            }
        }
        offsets[chunk * (D_MAX + 1) + partitions] = order.len() as i32;
    }
    let order = ints(&device, order);
    let offsets = ints(&device, offsets);
    let (count, dim) = geometry(&device, partitions);
    owner_join_apply::launch(
        &client,
        count,
        dim,
        rr.into_tensor_arg(),
        ri.into_tensor_arg(),
        rd.into_tensor_arg(),
        order.into_tensor_arg(),
        offsets.into_tensor_arg(),
        output_ids.clone().into_tensor_arg(),
        output_dist.clone().into_tensor_arg(),
        worst_ids.clone().into_tensor_arg(),
        worst_dist.clone().into_tensor_arg(),
        n as u32,
        width as u32,
        chunks as u32,
        partitions as u32,
        dtypes(),
    );
    let mut expected = Vec::new();
    for (row, initial_row) in initial.iter().enumerate() {
        let mut all = initial_row.clone();
        all.extend(records.iter().filter(|r| r.0 == row).map(|r| (r.1, r.2)));
        expected.push(top_h(n, width, row, &all));
    }
    let expected_ids: Vec<i32> = expected.iter().flatten().map(|v| v.0).collect();
    let expected_dist: Vec<f32> = expected.iter().flatten().map(|v| v.1).collect();
    assert_eq!(read_i32(&output_ids), expected_ids);
    assert_eq!(bits(&read_f32(&output_dist)), bits(&expected_dist));
    assert_eq!(
        read_i32(&worst_ids),
        expected.iter().map(|r| r[width - 1].0).collect::<Vec<_>>()
    );
    assert_eq!(
        bits(&read_f32(&worst_dist)),
        bits(&expected.iter().map(|r| r[width - 1].1).collect::<Vec<_>>())
    );
    if compare_slots {
        let mut offsets = vec![0i32];
        let mut ids = Vec::new();
        let mut distances = Vec::new();
        for row in 0..n {
            for &(_, id, d) in records.iter().filter(|r| r.0 == row) {
                ids.push(id);
                distances.push(d);
            }
            offsets.push(ids.len() as i32);
        }
        let si = tensor(&device, &[n, width], flat_ids, DType::I32);
        let sd = tensor(&device, &[n, width], flat_dist, DType::F32);
        let wi = ints(&device, vec![-1; n]);
        let wd = floats(&device, vec![-1.0; n]);
        let (count, dim) = geometry(&device, n);
        endpoint_merge_cpu::launch(
            &client,
            count,
            dim,
            ints(&device, offsets).into_tensor_arg(),
            ints(&device, ids).into_tensor_arg(),
            floats(&device, distances).into_tensor_arg(),
            si.clone().into_tensor_arg(),
            sd.clone().into_tensor_arg(),
            wi.clone().into_tensor_arg(),
            wd.clone().into_tensor_arg(),
            width as u32,
            dtypes(),
        );
        assert_eq!(read_i32(&si), expected_ids);
        assert_eq!(bits(&read_f32(&sd)), bits(&expected_dist));
        assert_eq!(read_i32(&wi), read_i32(&worst_ids));
        assert_eq!(bits(&read_f32(&wd)), bits(&read_f32(&worst_dist)));
    }
}

fn random(state: &mut u64) -> u32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 32) as u32
}

#[test]
fn owner_apply_matches_host_and_slot_merge() {
    let n = 9;
    let width = 4;
    let mut seed = 71;
    let initial: Vec<_> = (0..n)
        .map(|row| {
            let entries: Vec<_> = (0..if row % 2 == 0 { width } else { 2 })
                .map(|i| (((row + i + 1) % n) as i32, (i + 2) as f32))
                .collect();
            top_h(n, width, row, &entries)
        })
        .collect();
    let records: Vec<_> = (0..150)
        .map(|_| {
            (
                (random(&mut seed) as usize) % n,
                (random(&mut seed) as usize % n) as i32,
                (random(&mut seed) % 7) as f32,
            )
        })
        .collect();
    for chunks in [1, 3, 9] {
        for partitions in [1, 3, 9] {
            apply_case(&initial, &records, chunks, partitions, true);
        }
    }
}

#[test]
fn owner_apply_permutations_obey_top_h_lemma_with_snapshot_filtering() {
    let n = 9;
    let width = 3;
    let mut seed = 31337;
    for _ in 0..12 {
        let initial: Vec<_> = (0..n)
            .map(|row| {
                let entries: Vec<_> = (0..5)
                    .map(|_| {
                        (
                            (random(&mut seed) % n as u32) as i32,
                            (random(&mut seed) % 3) as f32,
                        )
                    })
                    .collect();
                top_h(n, width, row, &entries)
            })
            .collect();
        let all: Vec<_> = (0..100)
            .map(|_| {
                (
                    (random(&mut seed) as usize) % n,
                    (random(&mut seed) % n as u32) as i32,
                    (random(&mut seed) % 3) as f32,
                )
            })
            .collect();
        let mut filtered: Vec<_> = all
            .iter()
            .copied()
            .filter(|&(row, id, d)| {
                let (worst_id, worst_d) = initial[row][width - 1];
                worst_id as usize >= n || d < worst_d || (d == worst_d && id < worst_id)
            })
            .collect();
        // Check the snapshot filter independently of the device and permutation.
        for (row, initial_row) in initial.iter().enumerate() {
            let mut full = initial_row.clone();
            full.extend(all.iter().filter(|r| r.0 == row).map(|r| (r.1, r.2)));
            let mut pruned = initial_row.clone();
            pruned.extend(filtered.iter().filter(|r| r.0 == row).map(|r| (r.1, r.2)));
            assert_eq!(top_h(n, width, row, &full), top_h(n, width, row, &pruned));
        }
        for _ in 0..4 {
            for i in (1..filtered.len()).rev() {
                let j = random(&mut seed) as usize % (i + 1);
                filtered.swap(i, j);
            }
            apply_case(&initial, &filtered, 5, 3, false);
        }
    }
}
