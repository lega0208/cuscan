//! Two ordered passes over stable host-built membership, with no candidate readback.
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

#[cfg(test)]
std::thread_local! {
    static TEST_LAUNCHES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
    static TEST_BLOCK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cube]
fn numerical(bits: u32) -> bool {
    let magnitude = bits & 0x7fff_ffffu32;
    (bits & 0x8000_0000u32 != 0 && magnitude != 0) || magnitude > 0x7f80_0000u32
}

#[cube]
fn wins(bits: u32, source: u32, best_bits: u32, best_source: u32, n: u32) -> bool {
    // All participating values are validated nonnegative f32. Integer keys avoid
    // fast-math/FTZ changing zero ties, subnormal ordering or original payloads.
    let key = if bits & 0x7fff_ffffu32 == 0 {
        0u32
    } else {
        bits
    };
    let best_key = if best_bits & 0x7fff_ffffu32 == 0 {
        0u32
    } else {
        best_bits
    };
    source < n && (best_source == n || key < best_key || (key == best_key && source < best_source))
}

#[cube(launch)]
fn tile_pass<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    members: &Tensor<I>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    n: u32,
    tiles: u32,
    offset: u32,
    partial_bits: &mut Tensor<u32>,
    partial_sources: &mut Tensor<u32>,
    partial_bad: &mut Tensor<u32>,
    #[comptime] cpu: bool,
    #[comptime] block: usize,
    #[define(F, I)] _types: [ElemType; 2],
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
        let mut best_source = n;
        let mut best_bits = 0x7f80_0000u32.runtime();
        let mut bad = n;
        let mut position = u32::cast_from(starts[task]) as usize + first;
        let end = u32::cast_from(ends[task]) as usize;
        while position < end {
            let source = u32::cast_from(members[position]);
            let bits = u32::reinterpret(weights[source as usize]);
            let destination = indices[source as usize];
            if numerical(bits) || destination >= I::cast_from(n) {
                bad = bad.min(source);
            } else if destination >= I::cast_from(0)
                && wins(bits, source, best_bits, best_source, n)
            {
                best_bits = bits;
                best_source = source;
            }
            position += step;
        }
        if comptime![cpu] {
            partial_bits[task] = best_bits;
            partial_sources[task] = best_source;
            partial_bad[task] = bad;
        } else {
            let mut shared_bits = Shared::<[u32]>::new_slice(block);
            let mut shared_sources = Shared::<[u32]>::new_slice(block);
            let mut shared_bad = Shared::<[u32]>::new_slice(block);
            shared_bits[lane] = best_bits;
            shared_sources[lane] = best_source;
            shared_bad[lane] = bad;
            sync_cube();
            let mut stride = comptime![block / 2].runtime();
            while stride > 0 {
                if lane < stride {
                    if wins(
                        shared_bits[lane + stride],
                        shared_sources[lane + stride],
                        shared_bits[lane],
                        shared_sources[lane],
                        n,
                    ) {
                        shared_bits[lane] = shared_bits[lane + stride];
                        shared_sources[lane] = shared_sources[lane + stride];
                    }
                    shared_bad[lane] = shared_bad[lane].min(shared_bad[lane + stride]);
                }
                sync_cube();
                stride /= 2;
            }
            if lane == 0 {
                partial_bits[task] = shared_bits[0];
                partial_sources[task] = shared_sources[0];
                partial_bad[task] = shared_bad[0];
            }
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

#[cube(launch)]
fn finish_pass<F: Float, I: Int>(
    weights: &Tensor<F>,
    indices: &Tensor<I>,
    partial_bits: &Tensor<u32>,
    partial_sources: &Tensor<u32>,
    partial_bad: &Tensor<u32>,
    offsets: &Tensor<I>,
    n: u32,
    components: u32,
    offset: u32,
    out_weights: &mut Tensor<F>,
    out_meta: &mut Tensor<I>,
    #[comptime] cpu: bool,
    #[comptime] block: usize,
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
        let mut best_source = n;
        let mut best_bits = 0x7f80_0000u32.runtime();
        let mut bad = n;
        let mut position = u32::cast_from(offsets[task]) as usize + first;
        let end = u32::cast_from(offsets[task + 1]) as usize;
        while position < end {
            let source = partial_sources[position];
            let bits = partial_bits[position];
            bad = bad.min(partial_bad[position]);
            if wins(bits, source, best_bits, best_source, n) {
                best_bits = bits;
                best_source = source;
            }
            position += step;
        }
        if comptime![cpu] {
            store_winner(
                weights,
                indices,
                out_weights,
                out_meta,
                task,
                n,
                best_bits,
                best_source,
                bad,
            );
        } else {
            let mut shared_bits = Shared::<[u32]>::new_slice(block);
            let mut shared_sources = Shared::<[u32]>::new_slice(block);
            let mut shared_bad = Shared::<[u32]>::new_slice(block);
            shared_bits[lane] = best_bits;
            shared_sources[lane] = best_source;
            shared_bad[lane] = bad;
            sync_cube();
            let mut stride = comptime![block / 2].runtime();
            while stride > 0 {
                if lane < stride {
                    if wins(
                        shared_bits[lane + stride],
                        shared_sources[lane + stride],
                        shared_bits[lane],
                        shared_sources[lane],
                        n,
                    ) {
                        shared_bits[lane] = shared_bits[lane + stride];
                        shared_sources[lane] = shared_sources[lane + stride];
                    }
                    shared_bad[lane] = shared_bad[lane].min(shared_bad[lane + stride]);
                }
                sync_cube();
                stride /= 2;
            }
            if lane == 0 {
                store_winner(
                    weights,
                    indices,
                    out_weights,
                    out_meta,
                    task,
                    n,
                    shared_bits[0],
                    shared_sources[0],
                    shared_bad[0],
                );
            }
        }
        task += 1;
    }
}

fn bytes(count: usize, size: usize) -> usize {
    count
        .checked_mul(size)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .expect("component reducer allocation overflow")
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
    let mut workspace = 0usize;
    // Account for each input/output/scratch allocation, not a rectangular matrix.
    for length in [
        n,
        n,
        n,
        p,
        p,
        c as usize + 1,
        p,
        p,
        p,
        c as usize,
        bytes(c as usize, 5),
    ] {
        let allocation = bytes(length, 4);
        let aligned = allocation
            .checked_add(alignment - 1)
            .expect("allocation alignment overflow")
            / alignment
            * alignment;
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
    let partial_bits = allocate(Shape::new([p]), DType::U32, p);
    let partial_sources = allocate(Shape::new([p]), DType::U32, p);
    let partial_bad = allocate(Shape::new([p]), DType::U32, p);
    let weights = into_contiguous(weights);
    let indices = into_contiguous(indices);
    let members = into_contiguous(members);
    let tile_starts = into_contiguous(tile_starts);
    let tile_ends = into_contiguous(tile_ends);
    let partial_offsets = into_contiguous(partial_offsets);
    let hardware = &client.properties().hardware;
    let cpu = hardware.num_cpu_cores.is_some();
    let desired = if let Some(cores) = hardware.num_cpu_cores {
        cores.min(p as u32).max(1)
    } else {
        128
    };
    let legal = fast_cubecl_common::legal_cube_dim(&client, CubeDim::new_1d(desired)).x;
    let block = if cpu {
        legal
    } else {
        let shared_limit = (hardware.max_shared_memory_size / 12).min(128) as u32;
        let limit = legal.min(shared_limit).min(128);
        assert!(limit > 0, "no legal reduction cube/shared-memory capacity");
        1u32 << limit.ilog2()
    };
    let dim = CubeDim::new_1d(block);
    let max_cubes = hardware.max_cube_count.0;
    assert!(max_cubes > 0, "device has no cube launch capacity");
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
            weights.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            members.clone().into_tensor_arg(),
            tile_starts.clone().into_tensor_arg(),
            tile_ends.clone().into_tensor_arg(),
            n as u32,
            p as u32,
            offset as u32,
            partial_bits.clone().into_tensor_arg(),
            partial_sources.clone().into_tensor_arg(),
            partial_bad.clone().into_tensor_arg(),
            cpu,
            block as usize,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        #[cfg(test)]
        TEST_LAUNCHES.with(|counts| {
            let mut value = counts.get();
            value[0] += 1;
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
            partial_bits.clone().into_tensor_arg(),
            partial_sources.clone().into_tensor_arg(),
            partial_bad.clone().into_tensor_arg(),
            partial_offsets.clone().into_tensor_arg(),
            n as u32,
            c,
            offset as u32,
            output.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
            cpu,
            block as usize,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        #[cfg(test)]
        TEST_LAUNCHES.with(|counts| {
            let mut value = counts.get();
            value[1] += 1;
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

    fn dispatch_counts(device: CubeDevice) {
        let int = |values: Vec<i32>| {
            let count = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [count]), &device)
        };
        for n in [0, 4] {
            let labels = if n == 0 { vec![] } else { vec![0, 0, 2, 2] };
            let plan = ComponentReductionPlan::build(&labels, 512).unwrap();
            let c = plan.offsets.len() - 1;
            let weights = CubeBackend::float_from_data(
                TensorData::new(
                    if n == 0 {
                        vec![]
                    } else {
                        vec![4.0f32, 3.0, 2.0, 1.0]
                    },
                    [n],
                ),
                &device,
            );
            TEST_LAUNCHES.set([0, 0]);
            TEST_BLOCK.set(0);
            let (weights, metadata) = component_reduce(
                weights,
                int(vec![0; n]),
                int(plan.members),
                int(plan.tile_starts),
                int(plan.tile_ends),
                int(plan.partial_offsets),
                c as u32,
            );
            let actual = into_data_sync(weights).try_to_vec::<f32>().unwrap();
            assert_eq!(metadata.shape().dims::<2>(), [c, 5]);
            let expected = if n == 0 { [0, 0] } else { [1, 1] };
            assert_eq!(TEST_LAUNCHES.get(), expected);
            if n > 0 {
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
    fn component_empty_and_two_pass_dispatch_counts_cpu() {
        dispatch_counts(CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn component_empty_and_two_pass_dispatch_counts_cuda() {
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
        let partial_bits = uints(vec![0xdead_beef; CAPACITY]);
        let partial_sources = uints(vec![u32::MAX; CAPACITY]);
        let partial_bad = uints(vec![0; CAPACITY]);
        let output = floats(vec![f32::NAN; CAPACITY]);
        let metadata = ints(vec![i32::MIN; CAPACITY * 5]);
        let client = weights.client.clone();
        let cpu = client.properties().hardware.num_cpu_cores.is_some();
        let block = if cpu { 2 } else { 128 };
        let dim = CubeDim::new_1d(block);
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
            // Force one-cube batches independently of the much larger hardware
            // cap. CPU specialization covers all tasks in its one cube.
            for offset in 0..if cpu { 1 } else { p } {
                tile_pass::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    weights.clone().into_tensor_arg(),
                    indices.clone().into_tensor_arg(),
                    members.clone().into_tensor_arg(),
                    starts.clone().into_tensor_arg(),
                    ends.clone().into_tensor_arg(),
                    N as u32,
                    p as u32,
                    offset as u32,
                    partial_bits.clone().into_tensor_arg(),
                    partial_sources.clone().into_tensor_arg(),
                    partial_bad.clone().into_tensor_arg(),
                    cpu,
                    block as usize,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                    ],
                );
            }
            for offset in 0..if cpu { 1 } else { c } {
                finish_pass::launch(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    dim,
                    weights.clone().into_tensor_arg(),
                    indices.clone().into_tensor_arg(),
                    partial_bits.clone().into_tensor_arg(),
                    partial_sources.clone().into_tensor_arg(),
                    partial_bad.clone().into_tensor_arg(),
                    offsets.clone().into_tensor_arg(),
                    N as u32,
                    c as u32,
                    offset as u32,
                    output.clone().into_tensor_arg(),
                    metadata.clone().into_tensor_arg(),
                    cpu,
                    block as usize,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                    ],
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
}
