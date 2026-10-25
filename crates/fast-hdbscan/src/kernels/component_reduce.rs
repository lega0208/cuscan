//! CSR-ordered staging and two vectorized winner passes, with no candidate readback.
use burn::{
    backend::{
        TensorMetadata,
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
    },
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::features::Plane;

#[cfg(test)]
std::thread_local! {
    static TEST_LAUNCHES: std::cell::Cell<[usize; 3]> = const { std::cell::Cell::new([0; 3]) };
    static TEST_BLOCK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static TEST_MAX_CUBES: std::cell::Cell<u32> = const { std::cell::Cell::new(u32::MAX) };
}

#[cube]
fn numerical(bits: u32) -> bool {
    let magnitude = bits & 0x7fff_ffffu32;
    (bits & 0x8000_0000u32 != 0 && magnitude != 0) || magnitude > 0x7f80_0000u32
}

#[cube(launch)]
fn stage_pass<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    members: &Tensor<I>,
    n: u32,
    p: u32,
    n_vectors: u32,
    p_padded: u32,
    offset: u32,
    keys: &mut Tensor<Vector<u32, Const<4>>>,
    bits: &mut Tensor<Vector<u32, Const<4>>>,
    sources: &mut Tensor<Vector<u32, Const<4>>>,
    bads: &mut Tensor<Vector<u32, Const<4>>>,
    partial_keys: &mut Tensor<u32>,
    partial_bits: &mut Tensor<u32>,
    partial_sources: &mut Tensor<u32>,
    partial_bad: &mut Tensor<u32>,
    #[define(F, I)] _types: [ElemType; 2],
) {
    let vector = ABSOLUTE_POS + offset as usize;
    if vector < n_vectors as usize {
        let mut key_vector = Vector::<u32, Const<4>>::new(u32::MAX);
        let mut bit_vector = Vector::<u32, Const<4>>::new(0x7f80_0000u32);
        let mut source_vector = Vector::<u32, Const<4>>::new(n);
        let mut bad_vector = Vector::<u32, Const<4>>::new(n);
        #[unroll]
        for lane in 0usize..4 {
            let position = vector * 4 + lane;
            let mut key = u32::MAX.runtime();
            let mut value = 0x7f80_0000u32.runtime();
            let mut source = n;
            let mut bad = n;
            if position < n as usize {
                // Membership is indirect; only the resulting CSR-ordered fields
                // are contiguous vector data. Keep original-source payload bits.
                let member = u32::cast_from(members[position]);
                let original = u32::reinterpret(weights[member as usize]);
                let destination = indices[member as usize];
                if numerical(original) || destination >= I::cast_from(n) {
                    bad = member;
                } else if destination >= I::cast_from(0) {
                    key = if original & 0x7fff_ffffu32 == 0 {
                        0u32
                    } else {
                        original
                    };
                    value = original;
                    source = member;
                }
            }
            key_vector.insert(lane, key);
            bit_vector.insert(lane, value);
            source_vector.insert(lane, source);
            bad_vector.insert(lane, bad);
        }
        keys[vector] = key_vector;
        bits[vector] = bit_vector;
        sources[vector] = source_vector;
        bads[vector] = bad_vector;
    }
    // Disjoint scalar tail initialization: tile leaders own exactly [0,p).
    let partial = p as usize + vector;
    if partial < p_padded as usize {
        partial_keys[partial] = u32::MAX;
        partial_bits[partial] = 0x7f80_0000u32;
        partial_sources[partial] = n;
        partial_bad[partial] = n;
    }
}

#[cube]
fn fold_segment(
    keys: &Tensor<Vector<u32, Const<4>>>,
    bits: &Tensor<Vector<u32, Const<4>>>,
    sources: &Tensor<Vector<u32, Const<4>>>,
    bads: &Tensor<Vector<u32, Const<4>>>,
    start: usize,
    end: usize,
    n: u32,
    first: usize,
    step: usize,
) -> (u32, u32, u32, u32) {
    let mut key_acc = Vector::<u32, Const<4>>::new(u32::MAX);
    let mut bit_acc = Vector::<u32, Const<4>>::new(0x7f80_0000u32);
    let mut source_acc = Vector::<u32, Const<4>>::new(n);
    let mut bad_acc = Vector::<u32, Const<4>>::new(n);
    let mut g = start / 4 + first;
    while g < end.div_ceil(4) {
        let base = (g * 4) as u32;
        let mut positions = Vector::<u32, Const<4>>::empty();
        positions.insert(0usize, base);
        positions.insert(1usize, base + 1);
        positions.insert(2usize, base + 2);
        positions.insert(3usize, base + 3);
        let mask = positions
            .greater_equal(&Vector::new(start as u32))
            .vec_and(positions.less_than(&Vector::new(end as u32)));
        let k = select_many(mask, keys[g], Vector::new(u32::MAX));
        let b = select_many(mask, bits[g], Vector::new(0x7f80_0000u32));
        let s = select_many(mask, sources[g], Vector::new(n));
        let bad = select_many(mask, bads[g], Vector::new(n));
        let take = k
            .less_than(&key_acc)
            .or(k.equal(&key_acc).vec_and(s.less_than(&source_acc)));
        key_acc = select_many(take, k, key_acc);
        bit_acc = select_many(take, b, bit_acc);
        source_acc = select_many(take, s, source_acc);
        bad_acc = bad_acc.min(bad);
        g += step;
    }
    // Horizontal extraction happens once, after the lane-wise vector fold.
    let mut key = u32::MAX.runtime();
    let mut bits = 0x7f80_0000u32.runtime();
    let mut source = n;
    let mut bad = n;
    #[unroll]
    for lane in 0usize..4 {
        let k = key_acc.extract(lane);
        let s = source_acc.extract(lane);
        if k < key || (k == key && s < source) {
            key = k;
            bits = bit_acc.extract(lane);
            source = s;
        }
        bad = bad.min(bad_acc.extract(lane));
    }
    (key, bits, source, bad)
}

#[cube]
fn combine_planes(
    key: u32,
    bits: u32,
    source: u32,
    bad: u32,
    n: u32,
    #[comptime] cpu: bool,
    #[comptime] plane_width: u32,
    #[comptime] plane_count: usize,
) -> (u32, u32, u32, u32) {
    comptime![assert!(cpu || plane_width.is_power_of_two())];
    let mut final_key = key;
    let mut final_bits = bits;
    let mut final_source = source;
    let mut final_bad = bad;
    if comptime![!cpu] {
        let plane_key = plane_min(key);
        let plane_source = plane_min(select(key == plane_key, source, n));
        let plane_bits = plane_max(select(source == plane_source, bits, 0u32));
        let plane_bad = plane_min(bad);
        let mut shared_keys = Shared::<[u32]>::new_slice(plane_count);
        let mut shared_bits = Shared::<[u32]>::new_slice(plane_count);
        let mut shared_sources = Shared::<[u32]>::new_slice(plane_count);
        let mut shared_bad = Shared::<[u32]>::new_slice(plane_count);
        if UNIT_POS_PLANE == 0 {
            let plane = PLANE_POS as usize;
            shared_keys[plane] = plane_key;
            shared_bits[plane] = plane_bits;
            shared_sources[plane] = plane_source;
            shared_bad[plane] = plane_bad;
        }
        sync_cube();
        if UNIT_POS == 0 {
            final_key = u32::MAX;
            final_bits = 0x7f80_0000u32;
            final_source = n;
            final_bad = n;
            for plane in 0..plane_count {
                let k = shared_keys[plane];
                let s = shared_sources[plane];
                if k < final_key || (k == final_key && s < final_source) {
                    final_key = k;
                    final_bits = shared_bits[plane];
                    final_source = s;
                }
                final_bad = final_bad.min(shared_bad[plane]);
            }
        }
    }
    (final_key, final_bits, final_source, final_bad)
}

#[cube(launch)]
fn tile_pass<I: Int>(
    keys: &Tensor<Vector<u32, Const<4>>>,
    bits: &Tensor<Vector<u32, Const<4>>>,
    sources: &Tensor<Vector<u32, Const<4>>>,
    bads: &Tensor<Vector<u32, Const<4>>>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    n: u32,
    tiles: u32,
    offset: u32,
    partial_keys: &mut Tensor<u32>,
    partial_bits: &mut Tensor<u32>,
    partial_sources: &mut Tensor<u32>,
    partial_bad: &mut Tensor<u32>,
    #[comptime] cpu: bool,
    #[comptime] block: usize,
    #[comptime] plane_width: u32,
    #[comptime] plane_count: usize,
    #[define(I)] _type: ElemType,
) {
    let lane = UNIT_POS_X as usize;
    let mut task = (CUBE_POS_X + offset) as usize;
    let mut task_end = task + 1;
    let mut first = lane;
    let mut step = block.runtime();
    if comptime![cpu] {
        let chunk = (tiles as usize).div_ceil(CUBE_DIM_X as usize);
        task = lane * chunk;
        task_end = (task + chunk).min(tiles as usize);
        first = 0;
        step = 1;
    }
    while task < task_end && task < tiles as usize {
        let start = u32::cast_from(starts[task]) as usize;
        let end = u32::cast_from(ends[task]) as usize;
        let (key, value, source, bad) =
            fold_segment(keys, bits, sources, bads, start, end, n, first, step);
        let (key, value, source, bad) =
            combine_planes(key, value, source, bad, n, cpu, plane_width, plane_count);
        if comptime![cpu] || UNIT_POS == 0 {
            partial_keys[task] = key;
            partial_bits[task] = value;
            partial_sources[task] = source;
            partial_bad[task] = bad;
        }
        task += 1;
    }
}

#[cube(launch)]
fn finish_pass<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    keys: &Tensor<Vector<u32, Const<4>>>,
    bits: &Tensor<Vector<u32, Const<4>>>,
    sources: &Tensor<Vector<u32, Const<4>>>,
    bads: &Tensor<Vector<u32, Const<4>>>,
    offsets: &Tensor<I>,
    n: u32,
    components: u32,
    offset: u32,
    out_weights: &mut Tensor<F>,
    out_meta: &mut Tensor<I>,
    #[comptime] cpu: bool,
    #[comptime] block: usize,
    #[comptime] plane_width: u32,
    #[comptime] plane_count: usize,
    #[define(F, I)] _types: [ElemType; 2],
) {
    let lane = UNIT_POS_X as usize;
    let mut task = (CUBE_POS_X + offset) as usize;
    let mut task_end = task + 1;
    let mut first = lane;
    let mut step = block.runtime();
    if comptime![cpu] {
        let chunk = (components as usize).div_ceil(CUBE_DIM_X as usize);
        task = lane * chunk;
        task_end = (task + chunk).min(components as usize);
        first = 0;
        step = 1;
    }
    while task < task_end && task < components as usize {
        let start = u32::cast_from(offsets[task]) as usize;
        let end = u32::cast_from(offsets[task + 1]) as usize;
        let (key, value, source, bad) =
            fold_segment(keys, bits, sources, bads, start, end, n, first, step);
        let (_, value, source, bad) =
            combine_planes(key, value, source, bad, n, cpu, plane_width, plane_count);
        if comptime![cpu] || UNIT_POS == 0 {
            store_winner(
                weights,
                indices,
                out_weights,
                out_meta,
                task,
                n,
                value,
                source,
                bad,
            );
        }
        task += 1;
    }
}

#[cube]
fn store_winner<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    out_weights: &mut Tensor<F>,
    out_meta: &mut Tensor<I>,
    component: usize,
    n: u32,
    bits: u32,
    source: u32,
    bad: u32,
) {
    let mut destination = I::cast_from(-1i32);
    if source < n {
        destination = indices[source as usize];
    }
    let mut kind = 0u32.runtime();
    let mut bad_destination = I::cast_from(-1i32);
    if bad < n {
        if numerical(u32::reinterpret(weights[bad as usize])) {
            kind = 1;
        } else {
            kind = 2;
            bad_destination = indices[bad as usize];
        }
    }
    out_weights[component] = F::reinterpret(bits);
    let base = component * 5;
    out_meta[base] = if source < n {
        I::cast_from(source)
    } else {
        I::cast_from(-1i32)
    };
    out_meta[base + 1] = destination;
    out_meta[base + 2] = I::cast_from(bad);
    out_meta[base + 3] = I::cast_from(kind);
    out_meta[base + 4] = bad_destination;
}

fn bytes(count: usize, size: usize) -> usize {
    count
        .checked_mul(size)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .expect("component reducer allocation overflow")
}

#[cfg(test)]
#[cube(launch)]
fn vector_u32x4_probe(
    input: &Tensor<Vector<u32, Const<4>>>,
    output: &mut Tensor<Vector<u32, Const<4>>>,
    logical_len: u32,
) {
    let index = ABSOLUTE_POS as u32;
    let mut offsets = Vector::<u32, Const<4>>::empty();
    offsets.insert(0usize, index * 4);
    offsets.insert(1usize, index * 4 + 1);
    offsets.insert(2usize, index * 4 + 2);
    offsets.insert(3usize, index * 4 + 3);
    let mask = offsets.less_than(&Vector::new(logical_len));
    let loaded = if index * 4 < logical_len {
        input[index as usize]
    } else {
        Vector::new(0xffff_ffffu32)
    };
    let other = Vector::new(7u32);
    let choose_loaded = loaded.less_than(&other);
    let selected = select_many(choose_loaded, loaded, other);
    output[index as usize] = select_many(mask, selected, Vector::new(0xffff_ffffu32));
}

#[cfg(test)]
#[cube(launch)]
fn plane_tuple_probe(output: &mut Tensor<u32>, #[comptime] expected_plane_dim: u32) {
    let key = UNIT_POS;
    let source = UNIT_POS;
    let bits = source + 100;
    let bad = if UNIT_POS.is_multiple_of(7) {
        UNIT_POS
    } else {
        u32::MAX
    };
    let best_key = plane_min(key);
    let best_source = plane_min(select(key == best_key, source, u32::MAX));
    let best_bits = plane_max(select(source == best_source, bits, 0u32));
    let best_bad = plane_min(bad);
    if UNIT_POS_PLANE == 0 {
        let base = PLANE_POS as usize * 5;
        output[base] = best_key;
        output[base + 1] = best_source;
        output[base + 2] = best_bits;
        output[base + 3] = best_bad;
        output[base + 4] = PLANE_DIM;
    }
    sync_cube();
    if UNIT_POS == 0 {
        output[expected_plane_dim as usize * 5] = PLANE_DIM;
    }
}

pub(super) fn component_reduce(
    weights: FloatTensor<CubeBackend>,
    indices: IntTensor<CubeBackend>,
    members: IntTensor<CubeBackend>,
    tile_starts: IntTensor<CubeBackend>,
    tile_ends: IntTensor<CubeBackend>,
    partial_offsets: IntTensor<CubeBackend>,
    c: u32,
) -> (FloatTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let [n] = weights.shape().dims::<1>();
    let [p] = tile_starts.shape().dims::<1>();
    assert_eq!(weights.dtype, DType::F32);
    for (tensor, length) in [
        (&indices, n),
        (&members, n),
        (&tile_starts, p),
        (&tile_ends, p),
        (&partial_offsets, c as usize + 1),
    ] {
        assert_eq!(tensor.dtype, DType::I32);
        assert_eq!(tensor.shape().dims::<1>(), [length]);
        assert_eq!(
            tensor.device, weights.device,
            "reduction inputs must share a device"
        );
    }
    assert!(
        (n == 0 && c == 0 && p == 0) || (n > 0 && c > 0 && p > 0),
        "invalid empty/component metadata"
    );
    assert!(n <= i32::MAX as usize && c as usize <= p && p <= n);
    let client = weights.client.clone();
    let memory = &client.properties().memory;
    let max_page =
        usize::try_from(memory.max_page_size).expect("device max page size exceeds usize");
    let alignment = usize::try_from(memory.alignment).expect("device alignment exceeds usize");
    assert!(
        max_page > 0 && alignment > 0,
        "invalid device allocation limits"
    );
    let n_padded = n.checked_add(3).expect("staged row rounding overflow") / 4 * 4;
    let p_padded = p.checked_add(3).expect("partial rounding overflow") / 4 * 4;
    let n_vectors = n_padded / 4;
    let p_vectors = p_padded / 4;
    let mut workspace = 0usize;
    // Account for each input/output/scratch allocation, not a rectangular matrix.
    for length in [
        n,
        n,
        n,
        p,
        p,
        c as usize + 1,
        // Conservative upper bound for into_contiguous copies of all inputs.
        n,
        n,
        n,
        p,
        p,
        c as usize + 1,
        n_padded,
        n_padded,
        n_padded,
        n_padded,
        p_padded,
        p_padded,
        p_padded,
        p_padded,
        c as usize,
        bytes(c as usize, 5),
    ] {
        let allocation = bytes(length.max(1), 4);
        let aligned = allocation
            .checked_add(alignment - 1)
            .expect("allocation alignment overflow")
            .checked_div(alignment)
            .and_then(|pages| pages.checked_mul(alignment))
            .expect("allocation alignment overflow");
        assert!(
            aligned <= max_page,
            "component reducer allocation exceeds device page limit"
        );
        workspace = workspace
            .checked_add(aligned)
            .expect("component reducer workspace overflow");
    }
    assert!(
        workspace <= isize::MAX as usize,
        "component reducer workspace exceeds addressable bytes"
    );
    let allocate = |shape: Shape, dtype: DType, length: usize| {
        // Logical empty outputs have backing storage, but never a zero-byte raw allocation.
        CubeTensor::new_contiguous(
            client.clone(),
            weights.device.clone(),
            shape,
            client.empty(bytes(length.max(1), 4)),
            dtype,
        )
    };
    let output = allocate(Shape::new([c as usize]), DType::F32, c as usize);
    let meta = allocate(
        Shape::new([c as usize, 5]),
        DType::I32,
        bytes(c as usize, 5),
    );
    if n == 0 {
        return (output, meta);
    }
    let staged_key = allocate(Shape::new([n_padded]), DType::U32, n_padded);
    let staged_bits = allocate(Shape::new([n_padded]), DType::U32, n_padded);
    let staged_sources = allocate(Shape::new([n_padded]), DType::U32, n_padded);
    let staged_bad = allocate(Shape::new([n_padded]), DType::U32, n_padded);
    let partial_key = allocate(Shape::new([p_padded]), DType::U32, p_padded);
    let partial_bits = allocate(Shape::new([p_padded]), DType::U32, p_padded);
    let partial_sources = allocate(Shape::new([p_padded]), DType::U32, p_padded);
    let partial_bad = allocate(Shape::new([p_padded]), DType::U32, p_padded);
    let weights = into_contiguous(weights);
    let indices = into_contiguous(indices);
    let members = into_contiguous(members);
    let tile_starts = into_contiguous(tile_starts);
    let tile_ends = into_contiguous(tile_ends);
    let partial_offsets = into_contiguous(partial_offsets);
    // SAFETY: each contiguous u32 buffer has exactly vectors*4 scalar slots;
    // vector views use unit vector stride and never participate by capacity.
    let vector_view = |tensor: &CubeTensor, vectors: usize| unsafe {
        TensorArg::from_raw_parts(tensor.handle.clone(), [1].into(), [vectors].into())
    };
    let hardware = &client.properties().hardware;
    let cpu = hardware.num_cpu_cores.is_some();
    let desired = if let Some(cores) = hardware.num_cpu_cores {
        cores.min(p as u32).max(1)
    } else {
        128
    };
    let legal = fast_cubecl_common::legal_cube_dim(&client, CubeDim::new_1d(desired)).x;
    let (block, plane_width, plane_count) = if cpu {
        (legal, 0u32, 0usize)
    } else {
        assert!(
            client.features().plane.contains(Plane::Ops),
            "CUDA plane operations unavailable"
        );
        let width = hardware.plane_size_min;
        assert_eq!(
            width, hardware.plane_size_max,
            "nonuniform CUDA plane width"
        );
        assert!(
            width > 0 && width.is_power_of_two(),
            "unsupported CUDA plane width"
        );
        let shared_units = (hardware.max_shared_memory_size / 16)
            .checked_mul(width as usize)
            .expect("shared geometry overflow");
        let limit = (legal as usize).min(shared_units);
        assert!(
            limit >= width as usize,
            "no legal full-plane reduction cube"
        );
        let block = 1u32 << limit.ilog2();
        assert_eq!(block % width, 0);
        (block, width, (block / width) as usize)
    };
    let dim = CubeDim::new_1d(block);
    let max_cubes = hardware.max_cube_count.0;
    #[cfg(test)]
    let max_cubes = max_cubes.min(TEST_MAX_CUBES.get());
    assert!(max_cubes > 0, "device has no cube launch capacity");
    let stage_block = fast_cubecl_common::legal_cube_dim(&client, CubeDim::new_1d(128)).x;
    assert!(stage_block > 0);
    let stage_batch = (max_cubes as usize)
        .checked_mul(stage_block as usize)
        .expect("stage batch overflow");
    let types = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ];
    let stage_tasks = n_vectors.max(p_padded - p);
    for offset in (0..stage_tasks).step_by(stage_batch) {
        let count = (stage_tasks - offset)
            .min(stage_batch)
            .div_ceil(stage_block as usize) as u32;
        stage_pass::launch(
            &client,
            CubeCount::Static(count, 1, 1),
            CubeDim::new_1d(stage_block),
            weights.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            members.clone().into_tensor_arg(),
            n as u32,
            p as u32,
            n_vectors as u32,
            p_padded as u32,
            offset as u32,
            vector_view(&staged_key, n_vectors),
            vector_view(&staged_bits, n_vectors),
            vector_view(&staged_sources, n_vectors),
            vector_view(&staged_bad, n_vectors),
            partial_key.clone().into_tensor_arg(),
            partial_bits.clone().into_tensor_arg(),
            partial_sources.clone().into_tensor_arg(),
            partial_bad.clone().into_tensor_arg(),
            types,
        );
        #[cfg(test)]
        TEST_LAUNCHES.with(|counts| {
            let mut v = counts.get();
            v[0] += 1;
            counts.set(v);
        });
    }
    for offset in (0..p).step_by(if cpu { p } else { max_cubes as usize }) {
        let count = if cpu {
            1
        } else {
            (p - offset).min(max_cubes as usize) as u32
        };
        tile_pass::launch(
            &client,
            CubeCount::Static(count, 1, 1),
            dim,
            vector_view(&staged_key, n_vectors),
            vector_view(&staged_bits, n_vectors),
            vector_view(&staged_sources, n_vectors),
            vector_view(&staged_bad, n_vectors),
            tile_starts.clone().into_tensor_arg(),
            tile_ends.clone().into_tensor_arg(),
            n as u32,
            p as u32,
            offset as u32,
            partial_key.clone().into_tensor_arg(),
            partial_bits.clone().into_tensor_arg(),
            partial_sources.clone().into_tensor_arg(),
            partial_bad.clone().into_tensor_arg(),
            cpu,
            block as usize,
            plane_width,
            plane_count,
            dtype_to_storage_type(DType::I32),
        );
        #[cfg(test)]
        TEST_LAUNCHES.with(|counts| {
            let mut value = counts.get();
            value[1] += 1;
            counts.set(value);
            TEST_BLOCK.set(block);
        });
    }
    for offset in (0..c as usize).step_by(if cpu { c as usize } else { max_cubes as usize }) {
        let count = if cpu {
            1
        } else {
            (c as usize - offset).min(max_cubes as usize) as u32
        };
        finish_pass::launch(
            &client,
            CubeCount::Static(count, 1, 1),
            dim,
            weights.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            vector_view(&partial_key, p_vectors),
            vector_view(&partial_bits, p_vectors),
            vector_view(&partial_sources, p_vectors),
            vector_view(&partial_bad, p_vectors),
            partial_offsets.clone().into_tensor_arg(),
            n as u32,
            c,
            offset as u32,
            output.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
            cpu,
            block as usize,
            plane_width,
            plane_count,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        #[cfg(test)]
        TEST_LAUNCHES.with(|counts| {
            let mut value = counts.get();
            value[2] += 1;
            counts.set(value);
        });
    }
    (output, meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ops::{FloatTensorOps, IntTensorOps};
    use burn::tensor::TensorData;
    use burn_cubecl::{CubeDevice, ops::into_data_sync};
    use fast_hdbscan_core::mst::ComponentReductionPlan;
    use std::time::Instant;

    fn dispatch_counts(device: CubeDevice) {
        let int = |values: Vec<i32>| {
            let count = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [count]), &device)
        };
        for n in [0, 1, 4] {
            let labels = match n {
                0 => vec![],
                1 => vec![0],
                _ => vec![0, 0, 2, 2],
            };
            let plan = ComponentReductionPlan::build(&labels, 512).unwrap();
            let c = plan.offsets.len() - 1;
            let weights = CubeBackend::float_from_data(
                TensorData::new(
                    if n == 0 {
                        vec![]
                    } else if n == 1 {
                        vec![1.0f32]
                    } else {
                        vec![4.0f32, 3.0, 2.0, 1.0]
                    },
                    [n],
                ),
                &device,
            );
            TEST_LAUNCHES.set([0, 0, 0]);
            TEST_BLOCK.set(0);
            // n=p=1 exercises stage padding tasks (3) exceeding row vectors (1).
            TEST_MAX_CUBES.set(if n == 1 { 1 } else { u32::MAX });
            let (weights, metadata) = component_reduce(
                weights,
                int(vec![0; n]),
                int(plan.members),
                int(plan.tile_starts),
                int(plan.tile_ends),
                int(plan.partial_offsets),
                c as u32,
            );
            TEST_MAX_CUBES.set(u32::MAX);
            let actual = into_data_sync(weights).try_to_vec::<f32>().unwrap();
            assert_eq!(metadata.shape().dims::<2>(), [c, 5]);
            let expected = if n == 0 { [0, 0, 0] } else { [1, 1, 1] };
            assert_eq!(TEST_LAUNCHES.get(), expected);
            if n == 1 {
                assert_eq!(actual, [1.0]);
            } else if n > 0 {
                assert_eq!(actual, [3.0, 1.0]);
            } else {
                assert!(actual.is_empty());
            }
            eprintln!(
                "component reducer device={device:?} n={n} launches={expected:?} actual_block={}",
                TEST_BLOCK.get()
            );
        }
    }

    #[test]
    fn component_empty_and_three_pass_dispatch_counts_cpu() {
        dispatch_counts(CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn component_empty_and_three_pass_dispatch_counts_cuda() {
        if fast_cubecl_common::cuda_backend::try_device_count().unwrap() == 0 {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED dispatch count CUDA: no usable device");
            return;
        }
        dispatch_counts(CubeDevice::Cuda(Default::default()));
    }

    fn dispatch_batches(device: CubeDevice) {
        struct Restore;
        impl Drop for Restore {
            fn drop(&mut self) {
                TEST_MAX_CUBES.set(u32::MAX);
            }
        }
        let _restore = Restore;
        TEST_MAX_CUBES.set(1);
        TEST_LAUNCHES.set([0; 3]);
        let n = 513;
        let labels = (0..n).map(|row| (row % 2) as i32).collect::<Vec<_>>();
        let plan = ComponentReductionPlan::build(&labels, 4).unwrap();
        let p = plan.tile_starts.len();
        let ints = |values: Vec<i32>| {
            let len = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [len]), &device)
        };
        let weights = CubeBackend::float_from_data(TensorData::new(vec![1.0f32; n], [n]), &device);
        let cpu = weights.client.properties().hardware.num_cpu_cores.is_some();
        let stage_block =
            fast_cubecl_common::legal_cube_dim(&weights.client, CubeDim::new_1d(128)).x;
        let (weights, meta) = component_reduce(
            weights,
            ints(vec![0; n]),
            ints(plan.members),
            ints(plan.tile_starts),
            ints(plan.tile_ends),
            ints(plan.partial_offsets),
            2,
        );
        assert_eq!(
            into_data_sync(weights).try_to_vec::<f32>().unwrap(),
            [1.0, 1.0]
        );
        let meta = into_data_sync(meta).try_to_vec::<i32>().unwrap();
        assert_eq!(meta, [0, 0, n as i32, 0, -1, 1, 0, n as i32, 0, -1]);
        assert_eq!(
            TEST_LAUNCHES.get(),
            [
                n.div_ceil(4).div_ceil(stage_block as usize),
                if cpu { 1 } else { p },
                if cpu { 1 } else { 2 }
            ]
        );
    }

    #[test]
    fn component_three_pass_exact_batches_cpu() {
        dispatch_batches(CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn component_three_pass_exact_batches_cuda() {
        if fast_cubecl_common::cuda_backend::try_device_count().unwrap() == 0 {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED exact batches CUDA: no usable device");
            return;
        }
        dispatch_batches(CubeDevice::Cuda(Default::default()));
    }

    fn poison_reuse(device: CubeDevice) {
        const N: usize = 17;
        const CAPACITY: usize = 32;
        let floats = |values: Vec<f32>| {
            let len = values.len();
            CubeBackend::float_from_data(TensorData::new(values, [len]), &device)
        };
        let ints = |values: Vec<i32>| {
            let len = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [len]), &device)
        };
        let uints = |values: Vec<u32>| {
            let len = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [len]), &device)
        };
        let mut values = (0..N).map(|row| (N - row) as f32).collect::<Vec<_>>();
        values[0] = -0.0;
        values[1] = 0.0;
        let mut padded_values = values.clone();
        padded_values.resize(CAPACITY, f32::NAN);
        let weights = floats(padded_values);
        let indices = ints(vec![2; CAPACITY]);
        let staged_key = uints(vec![0xdead_beef; CAPACITY]);
        let staged_bits = uints(vec![0xdead_beef; CAPACITY]);
        let staged_sources = uints(vec![0xdead_beef; CAPACITY]);
        let staged_bad = uints(vec![0; CAPACITY]);
        let partial_key = uints(vec![0xdead_beef; CAPACITY]);
        let partial_bits = uints(vec![0xdead_beef; CAPACITY]);
        let partial_sources = uints(vec![u32::MAX; CAPACITY]);
        let partial_bad = uints(vec![0; CAPACITY]);
        let output = floats(vec![f32::NAN; CAPACITY]);
        let metadata = ints(vec![i32::MIN; CAPACITY * 5]);
        let client = weights.client.clone();
        let cpu = client.properties().hardware.num_cpu_cores.is_some();
        let block = if cpu { 2 } else { 128 };
        let plane_width = if cpu {
            0
        } else {
            client.properties().hardware.plane_size_min
        };
        let plane_count = if cpu {
            0
        } else {
            block as usize / client.properties().hardware.plane_size_min as usize
        };
        let dim = CubeDim::new_1d(block);
        let vector_view = |tensor: &CubeTensor| unsafe {
            TensorArg::from_raw_parts(tensor.handle.clone(), [1].into(), [CAPACITY / 4].into())
        };
        let types = [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ];
        for labels in [
            (0..N).map(|row| (row % 3) as i32).collect::<Vec<_>>(),
            vec![0; N],
            (0..N).map(|row| ((row + 1) % 2) as i32).collect(),
        ] {
            let plan = ComponentReductionPlan::build(&labels, 4).unwrap();
            let c = plan.offsets.len() - 1;
            let p = plan.tile_starts.len();
            let padded = |mut values: Vec<i32>| {
                values.resize(CAPACITY, i32::MAX);
                ints(values)
            };
            let members = padded(plan.members.clone());
            let starts = padded(plan.tile_starts.clone());
            let ends = padded(plan.tile_ends.clone());
            let offsets = padded(plan.partial_offsets.clone());
            let stage_tasks = N.div_ceil(4).max(p.next_multiple_of(4) - p);
            for offset in (0..stage_tasks).step_by(block as usize) {
                stage_pass::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    weights.clone().into_tensor_arg(),
                    indices.clone().into_tensor_arg(),
                    members.clone().into_tensor_arg(),
                    N as u32,
                    p as u32,
                    N.div_ceil(4) as u32,
                    p.next_multiple_of(4) as u32,
                    offset as u32,
                    vector_view(&staged_key),
                    vector_view(&staged_bits),
                    vector_view(&staged_sources),
                    vector_view(&staged_bad),
                    partial_key.clone().into_tensor_arg(),
                    partial_bits.clone().into_tensor_arg(),
                    partial_sources.clone().into_tensor_arg(),
                    partial_bad.clone().into_tensor_arg(),
                    types,
                );
            }
            // Force one-cube batches independently of the much larger hardware
            // cap. CPU specialization covers all tasks in its one cube.
            for offset in 0..if cpu { 1 } else { p } {
                tile_pass::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    vector_view(&staged_key),
                    vector_view(&staged_bits),
                    vector_view(&staged_sources),
                    vector_view(&staged_bad),
                    starts.clone().into_tensor_arg(),
                    ends.clone().into_tensor_arg(),
                    N as u32,
                    p as u32,
                    offset as u32,
                    partial_key.clone().into_tensor_arg(),
                    partial_bits.clone().into_tensor_arg(),
                    partial_sources.clone().into_tensor_arg(),
                    partial_bad.clone().into_tensor_arg(),
                    cpu,
                    block as usize,
                    plane_width,
                    plane_count,
                    dtype_to_storage_type(DType::I32),
                );
            }
            for offset in 0..if cpu { 1 } else { c } {
                finish_pass::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    weights.clone().into_tensor_arg(),
                    indices.clone().into_tensor_arg(),
                    vector_view(&partial_key),
                    vector_view(&partial_bits),
                    vector_view(&partial_sources),
                    vector_view(&partial_bad),
                    offsets.clone().into_tensor_arg(),
                    N as u32,
                    c as u32,
                    offset as u32,
                    output.clone().into_tensor_arg(),
                    metadata.clone().into_tensor_arg(),
                    cpu,
                    block as usize,
                    plane_width,
                    plane_count,
                    types,
                );
            }
            let actual = into_data_sync(output.clone()).try_to_vec::<f32>().unwrap();
            let actual_meta = into_data_sync(metadata.clone())
                .try_to_vec::<i32>()
                .unwrap();
            for slot in 0..c {
                let segment =
                    &plan.members[plan.offsets[slot] as usize..plan.offsets[slot + 1] as usize];
                let mut source = segment[0] as usize;
                for &row in segment {
                    if values[row as usize] < values[source] {
                        source = row as usize;
                    }
                }
                assert_eq!(actual[slot].to_bits(), values[source].to_bits());
                assert_eq!(
                    &actual_meta[slot * 5..slot * 5 + 5],
                    &[source as i32, 2, N as i32, 0, -1]
                );
            }
            assert!(actual[CAPACITY - 1].is_nan());
            assert_eq!(actual_meta[CAPACITY * 5 - 1], i32::MIN);
            for (tensor, start, identity) in [
                (&staged_key, N, u32::MAX),
                (&staged_bits, N, 0x7f80_0000),
                (&staged_sources, N, N as u32),
                (&partial_key, p, u32::MAX),
                (&partial_bits, p, 0x7f80_0000),
                (&partial_sources, p, N as u32),
            ] {
                let data = into_data_sync(tensor.clone()).try_to_vec::<u32>().unwrap();
                assert!(
                    data[start..start.next_multiple_of(4)]
                        .iter()
                        .all(|&value| value == identity)
                );
            }
            let staged_tail = into_data_sync(staged_bad.clone())
                .try_to_vec::<u32>()
                .unwrap();
            assert!(
                staged_tail[N..N.next_multiple_of(4)]
                    .iter()
                    .all(|&bad| bad == N as u32)
            );
            assert_eq!(staged_tail[CAPACITY - 1], 0);
            let partial_tail = into_data_sync(partial_bad.clone())
                .try_to_vec::<u32>()
                .unwrap();
            assert!(
                partial_tail[p..p.next_multiple_of(4)]
                    .iter()
                    .all(|&bad| bad == N as u32)
            );
        }
    }

    #[test]
    fn component_scratch_poison_reuse_capacity_and_batches_cpu() {
        poison_reuse(CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn component_scratch_poison_reuse_capacity_and_batches_cuda() {
        if fast_cubecl_common::cuda_backend::try_device_count().unwrap() == 0 {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED component scratch CUDA: no usable device");
            return;
        }
        poison_reuse(CubeDevice::Cuda(Default::default()));
    }

    fn run_vector_probe(device: CubeDevice) {
        let values = (0..16).collect::<Vec<i32>>();
        let input = CubeBackend::int_from_data(TensorData::new(values, [16]), &device);
        let output = CubeBackend::int_from_data(TensorData::new(vec![0; 16], [16]), &device);
        let input_arg =
            unsafe { TensorArg::from_raw_parts(input.handle.clone(), [1].into(), [4].into()) };
        let output_arg =
            unsafe { TensorArg::from_raw_parts(output.handle.clone(), [1].into(), [4].into()) };
        vector_u32x4_probe::launch(
            &input.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(4),
            input_arg,
            output_arg,
            13,
        );
        let actual = into_data_sync(output).try_to_vec::<i32>().unwrap();
        let expected = (0..16)
            .map(|value| {
                if value < 13 {
                    (value as u32).min(7) as i32
                } else {
                    -1
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn component_vector_u32x4_compare_select_masked_load_cpu() {
        run_vector_probe(CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn component_vector_u32x4_compare_select_masked_load_cuda() {
        if fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed") == 0
        {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable for vector probe"
            );
            eprintln!("SKIPPED vector probe CUDA: no usable device");
            return;
        }
        run_vector_probe(CubeDevice::Cuda(Default::default()));
    }

    #[test]
    fn component_cuda_plane_tuple_smoke() {
        use cubecl::features::Plane;
        if fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed") == 0
        {
            assert_ne!(
                std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable for plane probe"
            );
            eprintln!("SKIPPED plane probe CUDA: no usable device");
            return;
        }
        let device = CubeDevice::Cuda(Default::default());
        let output = CubeBackend::int_from_data(
            TensorData::new(vec![i32::MIN; 5 * 129], [5 * 129]),
            &device,
        );
        let client = output.client.clone();
        assert!(
            client.features().plane.contains(Plane::Ops),
            "CUDA plane operations unavailable"
        );
        let hardware = &client.properties().hardware;
        let min_width = hardware.plane_size_min;
        let width = hardware.plane_size_max;
        assert_eq!(min_width, width, "CUDA plane width is non-uniform");
        assert!(
            width > 0 && width.is_power_of_two(),
            "unsupported CUDA plane width {width}"
        );
        assert!(
            width <= 128,
            "probe output capacity does not support plane width {width}"
        );
        plane_tuple_probe::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(width),
            output.clone().into_tensor_arg(),
            width,
        );
        let actual = into_data_sync(output).try_to_vec::<i32>().unwrap();
        assert_eq!(actual[0..5], [0, 0, 100, 0, width as i32]);
        assert_eq!(
            actual[width as usize * 5],
            width as i32,
            "PLANE_DIM builtin disagrees with hardware metadata"
        );
    }

    #[test]
    #[ignore = "manual immutable scalar baseline; see component_reduce_report output"]
    fn component_reduce_report() {
        fn digest(weights: &[f32], metadata: &[i32]) -> u64 {
            weights
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .chain(metadata.iter().flat_map(|value| value.to_le_bytes()))
                .fold(0xcbf29ce484222325_u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                })
        }
        fn input_digest(weights: &[f32], arrays: &[&[i32]]) -> u64 {
            weights
                .iter()
                .flat_map(|value| value.to_bits().to_le_bytes())
                .chain(
                    arrays
                        .iter()
                        .flat_map(|array| array.iter().flat_map(|value| value.to_le_bytes())),
                )
                .fold(0xcbf29ce484222325_u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                })
        }
        fn labels_for(name: &str, n: usize) -> Vec<i32> {
            match name {
                "singleton" => (0..n).map(|i| i as i32).collect(),
                "uniform16" => (0..n).map(|i| (i % 16) as i32).collect(),
                "ragged1..7" => {
                    let mut labels = Vec::with_capacity(n);
                    let mut component = 0_i32;
                    while labels.len() < n {
                        let len = (component as usize % 7) + 1;
                        labels.extend(std::iter::repeat_n(component, len.min(n - labels.len())));
                        component += 1;
                    }
                    labels
                }
                "interleaved" => (0..n).map(|i| (i % 7) as i32).collect(),
                "random" => (0..n)
                    .map(|i| {
                        ((i.wrapping_mul(1_103_515_245).wrapping_add(12_345) >> 16) % 31) as i32
                    })
                    .collect(),
                "giant+tiny" => (0..n)
                    .map(|i| if i + 7 < n { 0 } else { i as i32 })
                    .collect(),
                "one-giant" => vec![0; n],
                "all-absent" => vec![0; n],
                _ => unreachable!(),
            }
        }
        let samples = 5;
        let warmups = 1;
        let tile_width = 512;
        assert!(
            fast_cubecl_common::cuda_backend::try_device_count().unwrap_or(0) > 0,
            "component_reduce_report requires CUDA; set up a usable CUDA device"
        );
        let devices = [
            ("cpu", CubeDevice::Cpu(Default::default())),
            ("cuda", CubeDevice::Cuda(Default::default())),
        ];
        let defaults = [
            ("ragged1..7", 28),
            ("uniform16", 16_384),
            ("giant+tiny", 65_536),
        ];
        let selected = std::env::var("FAST_HDBSCAN_COMPONENT_REPORT_CASES")
            .unwrap_or_else(|_| "ragged1..7,uniform16,giant+tiny".into());
        let cases = selected
            .split(',')
            .map(|name| {
                defaults
                    .iter()
                    .copied()
                    .find(|(candidate, _)| *candidate == name)
                    .or_else(|| {
                        Some((
                            name,
                            match name {
                                "singleton" | "interleaved" | "random" | "all-absent"
                                | "one-giant" => 16_384,
                                _ => panic!("unknown report case {name}"),
                            },
                        ))
                    })
                    .unwrap()
            })
            .collect::<Vec<_>>();
        eprintln!(
            "component_reduce_report schema=1 crate_version={} os={} arch={} revision={} cases={} backends=2 warmups={warmups} measured_invocations={samples} tile={tile_width} total_invocations={}",
            env!("CARGO_PKG_VERSION"),
            std::env::consts::OS,
            std::env::consts::ARCH,
            std::env::var("FAST_HDBSCAN_COMPONENT_REPORT_REVISION")
                .unwrap_or_else(|_| "unspecified".into()),
            cases.len(),
            cases.len() * devices.len() * (warmups + samples)
        );
        for (case, n) in cases {
            let labels = labels_for(case, n);
            let plan = ComponentReductionPlan::build(&labels, tile_width).unwrap();
            let weights = (0..n)
                .map(|i| {
                    f32::from_bits(
                        0x3f00_0000 | ((i.wrapping_mul(2654435761) as u32) & 0x007f_ffff),
                    )
                })
                .collect::<Vec<_>>();
            let indices = if case == "all-absent" {
                vec![-1; n]
            } else {
                vec![0; n]
            };
            let input_digest = input_digest(
                &weights,
                &[
                    &indices,
                    &labels,
                    &plan.members,
                    &plan.offsets,
                    &plan.tile_starts,
                    &plan.tile_ends,
                    &plan.partial_offsets,
                ],
            );
            let mut expected_output_digest = None;
            for (backend, device) in &devices {
                let weight_tensor =
                    CubeBackend::float_from_data(TensorData::new(weights.clone(), [n]), device);
                let int = |values: Vec<i32>| {
                    let len = values.len();
                    CubeBackend::int_from_data(TensorData::new(values, [len]), device)
                };
                let index_tensor = int(indices.clone());
                let members = int(plan.members.clone());
                let starts = int(plan.tile_starts.clone());
                let ends = int(plan.tile_ends.clone());
                let partial_offsets = int(plan.partial_offsets.clone());
                let run = || {
                    let (out_weights, out_meta) = component_reduce(
                        weight_tensor.clone(),
                        index_tensor.clone(),
                        members.clone(),
                        starts.clone(),
                        ends.clone(),
                        partial_offsets.clone(),
                        (plan.offsets.len() - 1) as u32,
                    );
                    let out_weights = into_data_sync(out_weights).try_to_vec::<f32>().unwrap();
                    let out_meta = into_data_sync(out_meta).try_to_vec::<i32>().unwrap();
                    digest(&out_weights, &out_meta)
                };
                let host_input_bytes = (plan.members.len()
                    + plan.tile_starts.len()
                    + plan.tile_ends.len()
                    + plan.partial_offsets.len()
                    + labels.len())
                    * std::mem::size_of::<i32>()
                    + weights.len() * std::mem::size_of::<f32>()
                    + indices.len() * std::mem::size_of::<i32>();
                for _ in 0..warmups {
                    std::hint::black_box(run());
                }
                let mut durations = Vec::with_capacity(samples);
                let mut output_digest = 0;
                for _ in 0..samples {
                    let start = Instant::now();
                    output_digest = std::hint::black_box(run());
                    durations.push(start.elapsed().as_nanos());
                }
                if let Some(expected) = expected_output_digest {
                    assert_eq!(
                        output_digest, expected,
                        "backend output digest mismatch for {case} n={n}"
                    );
                } else {
                    expected_output_digest = Some(output_digest);
                }
                eprintln!(
                    "case={case} n={n} backend={backend} device={device:?} components={} tiles={} input_digest={input_digest:016x} output_digest={output_digest:016x} known_host_input_bytes={host_input_bytes} warmups={warmups} samples_ns={durations:?}",
                    plan.offsets.len() - 1,
                    plan.tile_starts.len()
                );
            }
        }
    }
}
