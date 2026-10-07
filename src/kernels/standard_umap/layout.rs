use burn::{
    backend::{
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
        TensorMetadata,
    },
    tensor::{DType, Shape},
};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor, CubeBackend};
use cubecl::prelude::*;

const LAYOUT_CUBE_DIM: CubeDim = CubeDim::new_1d(1);

// The orchestration layer supplies finite positive intervals and valid next-event clocks.
// Embeddings, clocks and RNG state are updated in place: callers must pass
// buffers that no other live tensor aliases (e.g. clocks cloned from intervals).
#[cube(launch)]
fn optimize_epoch_kernel<F: Float, C: Float, I: Int, J: Int, S: Int>(
    head_embedding: &mut Tensor<F>,
    tail_embedding: &mut Tensor<F>,
    heads: &Tensor<J>,
    tails: &Tensor<J>,
    weights: &Tensor<F>,
    epochs_per_sample: &Tensor<C>,
    epoch_of_next_sample: &mut Tensor<C>,
    epochs_per_negative_sample: &Tensor<C>,
    epoch_of_next_negative_sample: &mut Tensor<C>,
    rng_state: &mut Tensor<I>,
    stats: &mut Tensor<S>,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    negative_sample_rate: f32,
    move_other: u32,
    shared_embeddings: u32,
    #[define(F, C, I, J, S)] _dtypes: [ElemType; 5],
) {
    // One serial unit owns the epoch; any extra launched units stay idle.
    if ABSOLUTE_POS_X == 0 {
        let edge_count = heads.shape(0);
        let head_count = head_embedding.shape(0);
        let tail_count = tail_embedding.shape(0);
        let dimensions = head_embedding.shape(1);
        let mut positive_events = 0u32;
        let mut negative_events = 0u32;
        for edge in 0..edge_count {
            let weight = weights[edge];
            if weight > F::new(0.0f32) && epoch_of_next_sample[edge] <= C::cast_from(epoch) {
                positive_events += 1;
                let head = u32::cast_from(heads[edge]) as usize;
                let tail = u32::cast_from(tails[edge]) as usize;
                if head < head_count && tail < tail_count {
                    let mut distance = F::new(0.0f32);
                    for coordinate in 0..dimensions {
                        let hi = head * dimensions + coordinate;
                        let ti = tail * dimensions + coordinate;
                        let difference = head_embedding[hi]
                            - if shared_embeddings == 1 {
                                head_embedding[ti]
                            } else {
                                tail_embedding[ti]
                            };
                        distance += difference * difference;
                    }
                    let mut coefficient = F::new(0.0f32);
                    if distance > F::new(0.0f32) {
                        let power = distance.powf(F::cast_from(b));
                        let prior_power = distance.powf(F::cast_from(b - 1.0f32));
                        coefficient = F::cast_from(-2.0f32 * a * b) * prior_power
                            / (F::cast_from(a) * power + F::new(1.0f32));
                    }
                    for axis in 0..dimensions {
                        let h = head * dimensions + axis;
                        let t = tail * dimensions + axis;
                        let current = head_embedding[h];
                        let other = if shared_embeddings == 1 {
                            head_embedding[t]
                        } else {
                            tail_embedding[t]
                        };
                        let delta = current - other;
                        let gradient = (coefficient * delta).clamp(F::new(-4.0f32), F::new(4.0f32));
                        head_embedding[h] = current + gradient * F::cast_from(alpha);
                        if shared_embeddings == 1 {
                            head_embedding[t] = other - gradient * F::cast_from(alpha);
                        } else if move_other == 1 {
                            tail_embedding[t] = other - gradient * F::cast_from(alpha);
                        }
                    }
                }

                epoch_of_next_sample[edge] += epochs_per_sample[edge];
                if negative_sample_rate > 0.0 {
                    let negative_clock = f64::cast_from(epoch_of_next_negative_sample[edge]);
                    let negative_interval = f64::cast_from(epochs_per_negative_sample[edge]);
                    let count = i32::cast_from(
                        (f64::cast_from(epoch) - negative_clock) / negative_interval,
                    );
                    if count > 0 {
                        let head = u32::cast_from(heads[edge]) as usize;
                        if head < head_count {
                            let sample_count = u32::cast_from(count);
                            for sample in 0..sample_count {
                                negative_events += 1;
                                let base = head * 3;
                                let mut r0 = rng_state[base];
                                let mut r1 = rng_state[base + 1];
                                let mut r2 = rng_state[base + 2];
                                r0 = (((r0 & I::cast_from(4294967294i64)) << I::cast_from(12i64))
                                    & I::cast_from(0xFFFFFFFFi64))
                                    ^ ((((r0 << I::cast_from(13i64))
                                        & I::cast_from(0xFFFFFFFFi64))
                                        ^ r0)
                                        >> I::cast_from(19i64));
                                r1 = (((r1 & I::cast_from(4294967288i64)) << I::cast_from(4i64))
                                    & I::cast_from(0xFFFFFFFFi64))
                                    ^ ((((r1 << I::cast_from(2i64))
                                        & I::cast_from(0xFFFFFFFFi64))
                                        ^ r1)
                                        >> I::cast_from(25i64));
                                r2 = (((r2 & I::cast_from(4294967280i64)) << I::cast_from(17i64))
                                    & I::cast_from(0xFFFFFFFFi64))
                                    ^ ((((r2 << I::cast_from(3i64))
                                        & I::cast_from(0xFFFFFFFFi64))
                                        ^ r2)
                                        >> I::cast_from(11i64));
                                rng_state[base] = r0;
                                rng_state[base + 1] = r1;
                                rng_state[base + 2] = r2;
                                let random = r0 ^ r1 ^ r2;
                                let signed_random = i32::cast_from(random);
                                let remainder = signed_random % i32::cast_from(tail_count);
                                let sample_index = if remainder < 0 {
                                    remainder + i32::cast_from(tail_count)
                                } else {
                                    remainder
                                };
                                let sampled = u32::cast_from(sample_index) as usize;
                                let same = sampled == head;
                                let mut distance = F::new(0.0f32);
                                for coordinate in 0..dimensions {
                                    let current = head_embedding[head * dimensions + coordinate];
                                    let other = if shared_embeddings == 1 {
                                        head_embedding[sampled * dimensions + coordinate]
                                    } else {
                                        tail_embedding[sampled * dimensions + coordinate]
                                    };
                                    let difference = current - other;
                                    distance += difference * difference;
                                }
                                if !(distance == F::new(0.0f32) && same) {
                                    let mut coefficient = F::new(0.0f32);
                                    if distance > F::new(0.0f32) {
                                        coefficient = F::cast_from(2.0f32 * gamma * b)
                                            / ((F::new(0.001f32) + distance)
                                                * (F::cast_from(a)
                                                    * distance.powf(F::cast_from(b))
                                                    + F::new(1.0f32)));
                                    }
                                    for coordinate in 0..dimensions {
                                        let h = head * dimensions + coordinate;
                                        let t = sampled * dimensions + coordinate;
                                        let current = head_embedding[h];
                                        let other = if shared_embeddings == 1 {
                                            head_embedding[t]
                                        } else {
                                            tail_embedding[t]
                                        };
                                        let delta = current - other;
                                        let gradient = if coefficient > F::new(0.0f32) {
                                            (coefficient * delta)
                                                .clamp(F::new(-4.0f32), F::new(4.0f32))
                                        } else {
                                            F::new(0.0f32)
                                        };
                                        head_embedding[h] =
                                            current + gradient * F::cast_from(alpha);
                                    }
                                }
                            }
                        }
                    }
                    let count = i32::cast_from(
                        (f64::cast_from(epoch) - negative_clock) / negative_interval,
                    );
                    let next_negative = negative_clock + f64::cast_from(count) * negative_interval;
                    epoch_of_next_negative_sample[edge] = C::cast_from(next_negative);
                }
            }
        }
        if shared_embeddings == 1 {
            for index in 0..head_count * dimensions {
                tail_embedding[index] = head_embedding[index];
            }
        }
        stats[0] = S::cast_from(positive_events);
        stats[1] = S::cast_from(negative_events);
    }
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
    IntTensor<CubeBackend>,
) {
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
    assert_eq!(tails.shape().dims::<1>()[0], edge_count);
    assert_eq!(weights.shape().dims::<1>()[0], edge_count);
    assert_eq!(epochs_per_sample.shape().dims::<1>()[0], edge_count);
    assert_eq!(epoch_of_next_sample.shape().dims::<1>()[0], edge_count);
    assert_eq!(
        epochs_per_negative_sample.shape().dims::<1>()[0],
        edge_count
    );
    assert_eq!(
        epoch_of_next_negative_sample.shape().dims::<1>()[0],
        edge_count
    );
    assert_eq!(rng_state.shape().dims::<2>(), [head_count, 3]);
    assert_eq!(head_embedding.dtype, DType::F32);
    assert_eq!(tail_embedding.dtype, DType::F32);
    assert_eq!(heads.dtype, DType::I32);
    assert_eq!(tails.dtype, DType::I32);
    assert_eq!(rng_state.dtype, DType::I64);
    assert_eq!(
        weights.dtype,
        DType::F32,
        "positive edge weights must be f32"
    );
    assert!(head_count > 0 && tail_vertices > 0 && dimensions > 0);
    let [clock_dtype] = [epochs_per_sample.dtype];
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
    let _ = head_count
        .checked_mul(dimensions)
        .expect("head coordinate count overflows usize");
    let _ = tail_vertices
        .checked_mul(dimensions)
        .expect("tail coordinate count overflows usize");
    let _ = u32::try_from(head_count).expect("layout head count exceeds CubeCL range");
    let _ = u32::try_from(tail_vertices).expect("layout tail count exceeds CubeCL range");
    let _ = u32::try_from(dimensions).expect("layout dimension count exceeds CubeCL range");
    let _ = u32::try_from(edge_count).expect("layout edge count exceeds CubeCL range");
    let stat_bytes = 2usize
        .checked_mul(DType::I32.size())
        .expect("layout stats allocation overflow");
    let stats = CubeTensor::new_contiguous(
        head_embedding.client.clone(),
        head_embedding.device.clone(),
        Shape::new([2]),
        head_embedding.client.empty(stat_bytes),
        DType::I32,
    );
    let client = head_embedding.client.clone();
    let dtypes = [
        dtype_to_storage_type(head_embedding.dtype),
        dtype_to_storage_type(epochs_per_sample.dtype),
        dtype_to_storage_type(rng_state.dtype),
        dtype_to_storage_type(heads.dtype),
        dtype_to_storage_type(stats.dtype),
    ];
    optimize_epoch_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        LAYOUT_CUBE_DIM,
        head_embedding.clone().into_tensor_arg(),
        tail_embedding.clone().into_tensor_arg(),
        heads.into_tensor_arg(),
        tails.into_tensor_arg(),
        weights.into_tensor_arg(),
        epochs_per_sample.into_tensor_arg(),
        epoch_of_next_sample.clone().into_tensor_arg(),
        epochs_per_negative_sample.into_tensor_arg(),
        epoch_of_next_negative_sample.clone().into_tensor_arg(),
        rng_state.clone().into_tensor_arg(),
        stats.clone().into_tensor_arg(),
        epoch,
        a,
        b,
        gamma,
        alpha,
        negative_sample_rate,
        move_other as u32,
        shared_embeddings as u32,
        dtypes,
    );
    (
        head_embedding,
        tail_embedding,
        epoch_of_next_sample,
        epoch_of_next_negative_sample,
        rng_state,
        stats,
    )
}
