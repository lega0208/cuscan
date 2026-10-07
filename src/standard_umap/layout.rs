//! Native Flex Euclidean SGD epoch for standard UMAP (CPU kernel).

use burn::{
    backend::{
        tensor::{FloatTensor, IntTensor},
        Flex,
    },
    tensor::DType,
};

/// Run one reference-ordered Euclidean SGD epoch directly on Flex storage,
/// mirroring `umap.layouts._optimize_layout_euclidean_single_epoch` (0.5.12)
/// and the CUDA serial kernel. Mutable operands use Flex copy-on-write, so
/// aliased inputs are never observed. Schedule tensors are trusted to be the
/// finite, positive intervals and valid next-event clocks created by
/// standard-UMAP orchestration.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(crate) fn optimize_epoch(
    head_embedding: FloatTensor<Flex>,
    tail_embedding: FloatTensor<Flex>,
    heads: IntTensor<Flex>,
    tails: IntTensor<Flex>,
    weights: FloatTensor<Flex>,
    epochs_per_sample: FloatTensor<Flex>,
    epoch_of_next_sample: FloatTensor<Flex>,
    epochs_per_negative_sample: FloatTensor<Flex>,
    epoch_of_next_negative_sample: FloatTensor<Flex>,
    rng_state: IntTensor<Flex>,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    alpha: f32,
    negative_sample_rate: f32,
    move_other: bool,
    shared_embeddings: bool,
) -> (
    FloatTensor<Flex>,
    FloatTensor<Flex>,
    FloatTensor<Flex>,
    FloatTensor<Flex>,
    IntTensor<Flex>,
    IntTensor<Flex>,
) {
    let mut head_embedding = head_embedding.to_contiguous();
    let mut tail_embedding = tail_embedding.to_contiguous();
    let heads = heads.to_contiguous();
    let tails = tails.to_contiguous();
    let weights = weights.to_contiguous();
    let epochs_per_sample = epochs_per_sample.to_contiguous();
    let mut epoch_of_next_sample = epoch_of_next_sample.to_contiguous();
    let epochs_per_negative_sample = epochs_per_negative_sample.to_contiguous();
    let mut epoch_of_next_negative_sample = epoch_of_next_negative_sample.to_contiguous();
    let mut rng_state = rng_state.to_contiguous();
    let head_shape = head_embedding.layout().shape().to_vec();
    let tail_shape = tail_embedding.layout().shape().to_vec();
    assert!(
        head_shape.len() == 2 && tail_shape.len() == 2,
        "layout embeddings must be rank 2"
    );
    let (head_count, dimensions) = (head_shape[0], head_shape[1]);
    let tail_count = tail_shape[0];
    let edge_count = heads.layout().shape()[0];
    assert_eq!(
        tail_shape[1], dimensions,
        "head/tail embedding dimensions differ"
    );
    for (tensor_len, label) in [
        (tails.layout().shape()[0], "tails"),
        (weights.layout().shape()[0], "weights"),
        (epochs_per_sample.layout().shape()[0], "positive intervals"),
        (epoch_of_next_sample.layout().shape()[0], "positive clocks"),
        (
            epochs_per_negative_sample.layout().shape()[0],
            "negative intervals",
        ),
        (
            epoch_of_next_negative_sample.layout().shape()[0],
            "negative clocks",
        ),
    ] {
        assert_eq!(
            tensor_len, edge_count,
            "{label} length differs from edge count"
        );
    }
    assert_eq!(
        rng_state.layout().shape().to_vec(),
        vec![head_count, 3],
        "RNG state must have shape [H, 3]"
    );
    assert!(
        head_count > 0 && tail_count > 0 && dimensions > 0,
        "layout coordinates must be non-empty"
    );
    assert_eq!(
        head_embedding.dtype(),
        DType::F32,
        "embedding coordinates must be f32"
    );
    assert_eq!(
        tail_embedding.dtype(),
        DType::F32,
        "embedding coordinates must be f32"
    );
    assert_eq!(heads.dtype(), DType::I32);
    assert_eq!(tails.dtype(), DType::I32);
    assert_eq!(
        weights.dtype(),
        DType::F32,
        "positive edge weights must be f32"
    );
    assert_eq!(
        rng_state.dtype(),
        DType::I64,
        "Tausworthe state must be i64"
    );
    let clock_dtype = epoch_of_next_sample.dtype();
    assert!(
        matches!(clock_dtype, DType::F32 | DType::F64),
        "optimizer clocks must be f32 or f64"
    );
    for dtype in [
        epochs_per_sample.dtype(),
        epochs_per_negative_sample.dtype(),
        epoch_of_next_negative_sample.dtype(),
    ] {
        assert_eq!(dtype, clock_dtype, "optimizer schedule dtypes differ");
    }
    assert!(
        negative_sample_rate.is_finite() && negative_sample_rate >= 0.0,
        "negative sample rate must be finite and non-negative"
    );
    if shared_embeddings {
        assert_eq!(
            tail_count, head_count,
            "shared layout embeddings must have equal shape"
        );
    }
    assert!(
        tail_count <= i32::MAX as usize,
        "tail vertex count exceeds i32"
    );

    let read_clock = |tensor: &FloatTensor<Flex>, index: usize| -> f64 {
        match clock_dtype {
            DType::F64 => tensor.storage::<f64>()[index],
            _ => f64::from(tensor.storage::<f32>()[index]),
        }
    };
    let positive_intervals: Vec<f64> = (0..edge_count)
        .map(|edge| read_clock(&epochs_per_sample, edge))
        .collect();
    let negative_intervals: Vec<f64> = (0..edge_count)
        .map(|edge| read_clock(&epochs_per_negative_sample, edge))
        .collect();
    let mut next_positive: Vec<f64> = (0..edge_count)
        .map(|edge| read_clock(&epoch_of_next_sample, edge))
        .collect();
    let mut next_negative: Vec<f64> = (0..edge_count)
        .map(|edge| read_clock(&epoch_of_next_negative_sample, edge))
        .collect();
    let edge_heads = heads.storage::<i32>();
    let edge_tails = tails.storage::<i32>();
    let edge_weights = weights.storage::<f32>();
    let mut head: Vec<f32> = head_embedding.storage::<f32>().to_vec();
    let mut tail: Vec<f32> = if shared_embeddings {
        Vec::new()
    } else {
        tail_embedding.storage::<f32>().to_vec()
    };
    let rng = rng_state.storage_mut::<i64>();

    let epoch_f = f64::from(epoch);
    let mut positive_events = 0i32;
    let mut negative_events = 0i32;
    // Clock arithmetic follows numba's float64 promotion; f32 clocks store
    // their float64 updates back at f32 precision after every event.
    let store = |value: f64| {
        if clock_dtype == DType::F32 {
            f64::from(value as f32)
        } else {
            value
        }
    };
    for edge in 0..edge_count {
        if !(edge_weights[edge] > 0.0 && next_positive[edge] <= epoch_f) {
            continue;
        }
        positive_events += 1;
        let j = edge_heads[edge] as usize;
        let k = edge_tails[edge] as usize;
        assert!(
            j < head_count && k < tail_count,
            "edge endpoint out of range"
        );
        let mut distance = 0.0f32;
        for d in 0..dimensions {
            let other = if shared_embeddings {
                head[k * dimensions + d]
            } else {
                tail[k * dimensions + d]
            };
            let diff = head[j * dimensions + d] - other;
            distance += diff * diff;
        }
        let coefficient = if distance > 0.0 {
            -2.0 * a * b * distance.powf(b - 1.0) / (a * distance.powf(b) + 1.0)
        } else {
            0.0
        };
        for d in 0..dimensions {
            let h = j * dimensions + d;
            let t = k * dimensions + d;
            let current = head[h];
            let other = if shared_embeddings { head[t] } else { tail[t] };
            let gradient = (coefficient * (current - other)).clamp(-4.0, 4.0);
            head[h] = current + gradient * alpha;
            if shared_embeddings {
                head[t] = other - gradient * alpha;
            } else if move_other {
                tail[t] = other - gradient * alpha;
            }
        }
        next_positive[edge] = store(next_positive[edge] + positive_intervals[edge]);
        if negative_sample_rate <= 0.0 {
            continue;
        }
        let count = ((epoch_f - next_negative[edge]) / negative_intervals[edge]) as i64;
        for _ in 0..count.max(0) {
            negative_events += 1;
            let state = &mut rng[j * 3..j * 3 + 3];
            state[0] = (((state[0] & 4_294_967_294) << 12) & 0xFFFF_FFFF)
                ^ ((((state[0] << 13) & 0xFFFF_FFFF) ^ state[0]) >> 19);
            state[1] = (((state[1] & 4_294_967_288) << 4) & 0xFFFF_FFFF)
                ^ ((((state[1] << 2) & 0xFFFF_FFFF) ^ state[1]) >> 25);
            state[2] = (((state[2] & 4_294_967_280) << 17) & 0xFFFF_FFFF)
                ^ ((((state[2] << 3) & 0xFFFF_FFFF) ^ state[2]) >> 11);
            let random = (state[0] ^ state[1] ^ state[2]) as i32;
            let sampled = random.rem_euclid(tail_count as i32) as usize;
            let mut distance = 0.0f32;
            for d in 0..dimensions {
                let other = if shared_embeddings {
                    head[sampled * dimensions + d]
                } else {
                    tail[sampled * dimensions + d]
                };
                let diff = head[j * dimensions + d] - other;
                distance += diff * diff;
            }
            let coefficient = if distance > 0.0 {
                2.0 * gamma * b / ((0.001 + distance) * (a * distance.powf(b) + 1.0))
            } else if j == sampled {
                continue;
            } else {
                0.0
            };
            for d in 0..dimensions {
                let h = j * dimensions + d;
                let other = if shared_embeddings {
                    head[sampled * dimensions + d]
                } else {
                    tail[sampled * dimensions + d]
                };
                let gradient = if coefficient > 0.0 {
                    (coefficient * (head[h] - other)).clamp(-4.0, 4.0)
                } else {
                    0.0
                };
                head[h] += gradient * alpha;
            }
        }
        next_negative[edge] = store(next_negative[edge] + count as f64 * negative_intervals[edge]);
    }

    head_embedding.storage_mut::<f32>().copy_from_slice(&head);
    if shared_embeddings {
        tail_embedding.storage_mut::<f32>().copy_from_slice(&head);
    } else {
        tail_embedding.storage_mut::<f32>().copy_from_slice(&tail);
    }
    for (tensor, values) in [
        (&mut epoch_of_next_sample, &next_positive),
        (&mut epoch_of_next_negative_sample, &next_negative),
    ] {
        match clock_dtype {
            DType::F64 => tensor.storage_mut::<f64>().copy_from_slice(values),
            _ => tensor
                .storage_mut::<f32>()
                .iter_mut()
                .zip(values.iter())
                .for_each(|(out, value)| *out = *value as f32),
        }
    }
    let stats = IntTensor::<Flex>::from_data(burn::tensor::TensorData::new(
        vec![positive_events, negative_events],
        [2],
    ));
    (
        head_embedding,
        tail_embedding,
        epoch_of_next_sample,
        epoch_of_next_negative_sample,
        rng_state,
        stats,
    )
}
