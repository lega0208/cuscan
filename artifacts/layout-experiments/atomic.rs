//! Race-free parallel Euclidean SGD epoch for CUDA standard UMAP.
//!
//! Three ordered launches per epoch on one stream:
//! 1. `layout_clock_kernel` — one lane per edge is the unique writer of that
//!    edge's clocks and records an immutable fire mask (`-1` = no positive
//!    event, otherwise the non-negative negative-sample count).
//! 2. `layout_copy_kernel` initializes a fresh embedding buffer.
//! 3. `layout_vertex_atomic_kernel` — one lane per head vertex processes its
//!    outgoing CSR edges in stable order, consuming its own per-head RNG and
//!    atomically applying attraction to both endpoints and repulsion to the
//!    head only. Every concurrent coordinate access is atomic.
//!
//! Fit (`shared`) returns the next buffer for both head and tail; transform
//! returns the fitted tail unchanged. Outputs may differ from the serial
//! reference order; quality, not coordinate equality, is the contract.

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

const LAYOUT_CUBE_DIM: CubeDim = CubeDim::new_1d(64);

#[cube(launch)]
fn layout_clock_kernel<F: Float, C: Float, M: Int>(
    weights: &Tensor<F>,
    epochs_per_sample: &Tensor<C>,
    epoch_of_next_sample: &mut Tensor<C>,
    epochs_per_negative_sample: &Tensor<C>,
    epoch_of_next_negative_sample: &mut Tensor<C>,
    mask: &mut Tensor<M>,
    epoch: u32,
    negative_sample_rate: f32,
    #[define(F, C, M)] _dtypes: [ElemType; 3],
) {
    let edge = ABSOLUTE_POS_X as usize;
    if edge < weights.shape(0) {
        let mut fired = M::cast_from(-1i32);
        if weights[edge] > F::new(0.0f32) && epoch_of_next_sample[edge] <= C::cast_from(epoch) {
            fired = M::cast_from(0i32);
            epoch_of_next_sample[edge] += epochs_per_sample[edge];
            if negative_sample_rate > 0.0 {
                let negative_clock = f64::cast_from(epoch_of_next_negative_sample[edge]);
                let negative_interval = f64::cast_from(epochs_per_negative_sample[edge]);
                let count =
                    i32::cast_from((f64::cast_from(epoch) - negative_clock) / negative_interval);
                if count > 0 {
                    fired = M::cast_from(count);
                }
                let next_negative = negative_clock + f64::cast_from(count) * negative_interval;
                epoch_of_next_negative_sample[edge] = C::cast_from(next_negative);
            }
        }
        mask[edge] = fired;
    }
}

#[cube]
fn layout_atomic_read<F: Float>(positions: &Tensor<Atomic<F>>, index: usize) -> F {
    // CUDA's CubeCL Atomic::load lowers to a volatile ordinary load. An atomic
    // add of zero returns the old value and keeps reads atomic with writers.
    positions[index].fetch_add(F::new(0.0f32))
}

#[cube]
fn layout_coordinate<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    shared: bool,
    row: usize,
    column: usize,
    dimensions: usize,
) -> F {
    if shared {
        layout_atomic_read(positions, row * dimensions + column)
    } else {
        fixed_tail[row * dimensions + column]
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn layout_copy_kernel<F: Float>(old_head: &Tensor<F>, next: &mut Tensor<F>) {
    let index = ABSOLUTE_POS_X as usize;
    if index < old_head.shape(0) * old_head.shape(1) {
        next[index] = old_head[index];
    }
}

#[cube]
fn layout_attract_atomic<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    head: usize,
    tail: usize,
    dimensions: usize,
    shared: bool,
    a: f32,
    b: f32,
    alpha: f32,
) {
    let mut distance = F::new(0.0f32);
    for c in 0..dimensions {
        let difference = layout_atomic_read(positions, head * dimensions + c)
            - layout_coordinate(positions, fixed_tail, shared, tail, c, dimensions);
        distance += difference * difference;
    }
    if distance > F::new(0.0f32) {
        let coefficient = F::cast_from(-2.0f32 * a * b) * distance.powf(F::cast_from(b - 1.0f32))
            / (F::cast_from(a) * distance.powf(F::cast_from(b)) + F::new(1.0f32));
        for c in 0..dimensions {
            let h = head * dimensions + c;
            let delta = layout_atomic_read(positions, h)
                - layout_coordinate(positions, fixed_tail, shared, tail, c, dimensions);
            let gradient = (coefficient * delta).clamp(F::new(-4.0f32), F::new(4.0f32));
            let update = gradient * F::cast_from(alpha);
            positions[h].fetch_add(update);
            if shared {
                positions[tail * dimensions + c].fetch_add(-update);
            }
        }
    }
}

#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn layout_vertex_atomic_kernel<F: Float, I: Int, J: Int, M: Int>(
    positions: &Tensor<Atomic<F>>,
    tail_rows: &Tensor<F>,
    tails: &Tensor<J>,
    head_offsets: &Tensor<J>,
    head_edges: &Tensor<J>,
    mask: &Tensor<M>,
    rng_state: &mut Tensor<I>,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    shared_embeddings: u32,
    #[define(F, I, J, M)] _dtypes: [ElemType; 4],
) {
    let vertex = ABSOLUTE_POS_X as usize;
    let head_count = positions.shape(0);
    if vertex < head_count {
        let dimensions = positions.shape(1);
        let tail_count = tail_rows.shape(0);
        let shared = shared_embeddings == 1;
        let out_start = u32::cast_from(head_offsets[vertex]) as usize;
        let out_end = u32::cast_from(head_offsets[vertex + 1]) as usize;
        for slot in out_start..out_end {
            let edge = u32::cast_from(head_edges[slot]) as usize;
            let fired = i32::cast_from(mask[edge]);
            if fired >= 0 {
                let tail = u32::cast_from(tails[edge]) as usize;
                if tail < tail_count && (!shared || tail != vertex) {
                    layout_attract_atomic(
                        positions, tail_rows, vertex, tail, dimensions, shared, a, b, alpha,
                    );
                }
                let base = vertex * 3;
                for _ in 0..u32::cast_from(fired) {
                    let mut r0 = rng_state[base];
                    let mut r1 = rng_state[base + 1];
                    let mut r2 = rng_state[base + 2];
                    r0 = (((r0 & I::cast_from(4294967294i64)) << I::cast_from(12i64))
                        & I::cast_from(0xFFFFFFFFi64))
                        ^ ((((r0 << I::cast_from(13i64)) & I::cast_from(0xFFFFFFFFi64)) ^ r0)
                            >> I::cast_from(19i64));
                    r1 = (((r1 & I::cast_from(4294967288i64)) << I::cast_from(4i64))
                        & I::cast_from(0xFFFFFFFFi64))
                        ^ ((((r1 << I::cast_from(2i64)) & I::cast_from(0xFFFFFFFFi64)) ^ r1)
                            >> I::cast_from(25i64));
                    r2 = (((r2 & I::cast_from(4294967280i64)) << I::cast_from(17i64))
                        & I::cast_from(0xFFFFFFFFi64))
                        ^ ((((r2 << I::cast_from(3i64)) & I::cast_from(0xFFFFFFFFi64)) ^ r2)
                            >> I::cast_from(11i64));
                    rng_state[base] = r0;
                    rng_state[base + 1] = r1;
                    rng_state[base + 2] = r2;
                    let signed_random = i32::cast_from(r0 ^ r1 ^ r2);
                    let remainder = signed_random % i32::cast_from(tail_count);
                    let sample_index = if remainder < 0 {
                        remainder + i32::cast_from(tail_count)
                    } else {
                        remainder
                    };
                    let sampled = u32::cast_from(sample_index) as usize;
                    if !shared || sampled != vertex {
                        let mut distance = F::new(0.0f32);
                        for c in 0..dimensions {
                            let difference = layout_atomic_read(positions, vertex * dimensions + c)
                                - layout_coordinate(
                                    positions, tail_rows, shared, sampled, c, dimensions,
                                );
                            distance += difference * difference;
                        }
                        if distance > F::new(0.0f32) {
                            let coefficient = F::cast_from(2.0f32 * gamma * b)
                                / ((F::new(0.001f32) + distance)
                                    * (F::cast_from(a) * distance.powf(F::cast_from(b))
                                        + F::new(1.0f32)));
                            for c in 0..dimensions {
                                let index = vertex * dimensions + c;
                                let difference = layout_atomic_read(positions, index)
                                    - layout_coordinate(
                                        positions, tail_rows, shared, sampled, c, dimensions,
                                    );
                                let gradient = (coefficient * difference)
                                    .clamp(F::new(-4.0f32), F::new(4.0f32));
                                positions[index].fetch_add(gradient * F::cast_from(alpha));
                            }
                        }
                    }
                }
            }
        }
    }
}

fn empty(template: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    let bytes = shape
        .num_elements()
        .checked_mul(dtype.size())
        .expect("layout scratch allocation overflows usize");
    CubeTensor::new_contiguous(
        template.client.clone(),
        template.device.clone(),
        shape,
        template.client.empty(bytes),
        dtype,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(crate) fn optimize_epoch(
    head_embedding: FloatTensor<CubeBackend>,
    tail_embedding: FloatTensor<CubeBackend>,
    heads: IntTensor<CubeBackend>,
    tails: IntTensor<CubeBackend>,
    weights: FloatTensor<CubeBackend>,
    epochs_per_sample: FloatTensor<CubeBackend>,
    epoch_of_next_sample: FloatTensor<CubeBackend>,
    epochs_per_negative_sample: FloatTensor<CubeBackend>,
    epoch_of_next_negative_sample: FloatTensor<CubeBackend>,
    rng_state: IntTensor<CubeBackend>,
    head_offsets: IntTensor<CubeBackend>,
    head_edges: IntTensor<CubeBackend>,
    tail_offsets: IntTensor<CubeBackend>,
    tail_edges: IntTensor<CubeBackend>,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    negative_sample_rate: f32,
    move_other: bool,
    shared_embeddings: bool,
) -> (
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    assert_eq!(
        move_other, shared_embeddings,
        "CUDA layout supports shared fit (move_other) or fixed-tail transform only"
    );
    let head_embedding = into_contiguous(head_embedding);
    let tail_embedding = into_contiguous(tail_embedding);
    let heads = into_contiguous(heads);
    let tails = into_contiguous(tails);
    let weights = into_contiguous(weights);
    let epochs_per_sample = into_contiguous(epochs_per_sample);
    let epoch_of_next_sample = into_contiguous(epoch_of_next_sample);
    let epochs_per_negative_sample = into_contiguous(epochs_per_negative_sample);
    let epoch_of_next_negative_sample = into_contiguous(epoch_of_next_negative_sample);
    let rng_state = into_contiguous(rng_state);
    let head_offsets = into_contiguous(head_offsets);
    let head_edges = into_contiguous(head_edges);
    let tail_offsets = into_contiguous(tail_offsets);
    let tail_edges = into_contiguous(tail_edges);
    let [head_count, dimensions] = head_embedding.shape().dims::<2>();
    let [tail_vertices, tail_dimensions] = tail_embedding.shape().dims::<2>();
    let [edge_count] = heads.shape().dims::<1>();
    assert_eq!(tail_dimensions, dimensions);
    if shared_embeddings {
        assert_eq!(
            tail_vertices, head_count,
            "shared embeddings must have equal row count"
        );
    }
    for length in [
        tails.shape().dims::<1>()[0],
        weights.shape().dims::<1>()[0],
        epochs_per_sample.shape().dims::<1>()[0],
        epoch_of_next_sample.shape().dims::<1>()[0],
        epochs_per_negative_sample.shape().dims::<1>()[0],
        epoch_of_next_negative_sample.shape().dims::<1>()[0],
    ] {
        assert_eq!(length, edge_count, "edge array length differs");
    }
    assert_eq!(head_offsets.shape().dims::<1>()[0], head_count + 1);
    assert_eq!(tail_offsets.shape().dims::<1>()[0], tail_vertices + 1);
    assert_eq!(rng_state.shape().dims::<2>(), [head_count, 3]);
    assert_eq!(head_embedding.dtype, DType::F32);
    assert_eq!(tail_embedding.dtype, DType::F32);
    for index_dtype in [
        heads.dtype,
        tails.dtype,
        head_offsets.dtype,
        head_edges.dtype,
        tail_offsets.dtype,
        tail_edges.dtype,
    ] {
        assert_eq!(index_dtype, DType::I32);
    }
    assert_eq!(rng_state.dtype, DType::I64);
    assert_eq!(
        weights.dtype,
        DType::F32,
        "positive edge weights must be f32"
    );
    assert!(head_count > 0 && tail_vertices > 0 && dimensions > 0);
    let clock_dtype = epochs_per_sample.dtype;
    assert!(matches!(clock_dtype, DType::F32 | DType::F64));
    assert_eq!(epoch_of_next_sample.dtype, clock_dtype);
    assert_eq!(epochs_per_negative_sample.dtype, clock_dtype);
    assert_eq!(epoch_of_next_negative_sample.dtype, clock_dtype);
    assert!(
        negative_sample_rate.is_finite() && negative_sample_rate >= 0.0,
        "negative sample rate must be finite and non-negative"
    );
    if negative_sample_rate > 0.0 {
        let bound = (2.0f64 * f64::from(negative_sample_rate) + 1.0).ceil();
        assert!(
            bound <= f64::from(i32::MAX),
            "bounded negative sample count exceeds i32 range"
        );
    }
    let coordinate_count = head_count
        .checked_mul(dimensions)
        .expect("head coordinate count overflows usize");
    let _ = tail_vertices
        .checked_mul(dimensions)
        .expect("tail coordinate count overflows usize");
    let head_u32 = u32::try_from(head_count).expect("layout head count exceeds CubeCL range");
    let coordinate_u32 =
        u32::try_from(coordinate_count).expect("layout coordinate count exceeds CubeCL range");
    let _ = u32::try_from(tail_vertices).expect("layout tail count exceeds CubeCL range");
    let _ = u32::try_from(dimensions).expect("layout dimension count exceeds CubeCL range");
    let edge_u32 = u32::try_from(edge_count).expect("layout edge count exceeds CubeCL range");

    let client = head_embedding.client.clone();
    let mask = empty(&head_embedding, Shape::new([edge_count]), DType::I32);
    layout_clock_kernel::launch(
        &client,
        CubeCount::Static(edge_u32.div_ceil(LAYOUT_CUBE_DIM.x), 1, 1),
        LAYOUT_CUBE_DIM,
        weights.into_tensor_arg(),
        epochs_per_sample.into_tensor_arg(),
        epoch_of_next_sample.clone().into_tensor_arg(),
        epochs_per_negative_sample.into_tensor_arg(),
        epoch_of_next_negative_sample.clone().into_tensor_arg(),
        mask.clone().into_tensor_arg(),
        epoch,
        negative_sample_rate,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(clock_dtype),
            dtype_to_storage_type(DType::I32),
        ],
    );
    let next = empty(
        &head_embedding,
        Shape::new([head_count, dimensions]),
        DType::F32,
    );
    layout_copy_kernel::launch::<f32>(
        &client,
        CubeCount::Static(coordinate_u32.div_ceil(LAYOUT_CUBE_DIM.x), 1, 1),
        LAYOUT_CUBE_DIM,
        head_embedding.clone().into_tensor_arg(),
        next.clone().into_tensor_arg(),
    );
    // Shared fit reads updated coordinates atomically from `next`; transform
    // reads the immutable fitted tail and still reads its mutable head atomically.
    let tail_rows = tail_embedding.clone();
    layout_vertex_atomic_kernel::launch(
        &client,
        CubeCount::Static(head_u32.div_ceil(LAYOUT_CUBE_DIM.x), 1, 1),
        LAYOUT_CUBE_DIM,
        next.clone().into_tensor_arg(),
        tail_rows.into_tensor_arg(),
        tails.into_tensor_arg(),
        head_offsets.into_tensor_arg(),
        head_edges.into_tensor_arg(),
        mask.into_tensor_arg(),
        rng_state.clone().into_tensor_arg(),
        a,
        b,
        gamma,
        alpha,
        shared_embeddings as u32,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I64),
            dtype_to_storage_type(DType::I32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    let tail = if shared_embeddings {
        next.clone()
    } else {
        tail_embedding
    };
    (
        next,
        tail,
        epoch_of_next_sample,
        epoch_of_next_negative_sample,
        rng_state,
    )
}
