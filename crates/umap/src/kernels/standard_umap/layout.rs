//! Parallel Euclidean SGD epoch for standard UMAP (CubeCL CPU and CUDA).
//!
//! Three ordered stages per epoch on one stream:
//! 1. `layout_clock_kernel` — one lane per edge is the unique writer of that
//!    edge's clocks and records an immutable fire mask (`-1` = no positive
//!    event, otherwise the non-negative negative-sample count).
//! 2. `layout_copy_kernel` initializes a fresh embedding buffer.
//! 3. A vertex kernel processes each head's outgoing CSR edges, consuming that
//!    head's serial RNG draw sequence. Shared fit processes four ascending
//!    contiguous head batches; transform uses one unpartitioned launch.
//!
//! Vertex kernel selection is fixed by capability before launch (no knob, no
//! execution fallback):
//!
//! | `n_components` | 32-lane plane device (Ops+Sync, 64-thread cube, 256 B shared) | other devices |
//! | --- | --- | --- |
//! | 1..=16 | `layout_vertex_fused_kernel`, group G (16→4, 8/10/12/14→2, else 1), no padding, 2 planes per cube | `layout_vertex_headcache_kernel` |
//! | 17..=32 | `layout_vertex_fused_kernel`, G=4 while `ceil(d/4) <= 5` (17–20), else G=8 (21–32), padded virtually to a multiple of G | `layout_vertex_headcache_kernel` |
//! | > 32 | `layout_vertex_atomic_kernel` | `layout_vertex_atomic_kernel` |
//!
//! The 17..=32 G rule is a heuristic tuned on one device (RTX 3080); register
//! pressure at 5 values per lane is untested on other GPUs.
//!
//! Semantics. Coordinates are read with `Atomic::load` and every tail update is
//! an atomic add, so no update is lost and every vertex consumes exactly the
//! serial RNG draw sequence (post-epoch RNG state is identical across kernels).
//! - Head-cache and fused hold the head row in registers and flush it once per
//!   vertex with atomic adds: other threads see the head stale until the flush,
//!   and the vertex does not see concurrent moves of itself. They reuse
//!   distance-pass differences for the gradient, as umap-learn does.
//! - Fused additionally applies the events of one plane step (up to `32/G`,
//!   positives and negatives mixed) against the same head snapshot and sums
//!   their moves at once; the cross-group reduction changes summation order.
//! - Padding above 16 components is virtual: storage and output keep the real
//!   width. Each fused lane holds `padded / G` register slots; slots past the
//!   real width are never loaded, read, updated or flushed, so they contribute
//!   exactly 0 to the distance. Only the butterfly summation order changes.
//! - The atomic kernel uses a runtime coordinate loop that rereads differences
//!   and does not cache the head.
//!
//! Quality, not coordinate equality, is the contract. Measured tradeoff: on
//! MNIST (16 and 2 components, spectral/random init, 3 seeds) fused matched
//! head-cache kNN preservation and accuracy; on 200k×64 blobs, fused G=4
//! preservation was 0.0598–0.0631 versus head-cache 0.0767–0.0787 (about 20%
//! relative loss) for a faster layout. On MNIST at 17–32 components (random
//! init, kNN preservation k=15) fused was 1.0–1.1% below atomic (at 32:
//! .2895–.2911 versus .2926–.2929) with equal 15-NN accuracy (.969–.972), for
//! a 9–22× faster layout on 200k×64. Padding alone had no measurable quality
//! effect. This is not universal quality equivalence.
//!
//! Fit (`shared`) returns the next buffer for both head and tail; transform
//! (`shared = false`) uses the same kernels with tail moves compiled out and
//! returns the fitted tail unchanged. Each `(shared, reg_dims, dims, group)`
//! combination is a separate JIT specialization. Launch failures surface at the
//! estimator's operation boundary (`sync_buffers` over the written handles).

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
    positions[index].load()
}

#[cube]
fn layout_coordinate<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    #[comptime] shared: bool,
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

/// Largest `n_components` handled by the fused plane kernel (padded virtually above 16).
const FUSED_MAX_DIMS: usize = 32;

/// Largest `n_components` handled by the head-cache kernel on non-plane devices.
/// Measured on CubeCL LLVM CPU (100k×32, random init, 200 epochs, warm optimization s):
/// head-cache vs atomic 8.2/13.3 at 17, 9.7/18.1 at 24, 11.6/23.3 at 32 components.
const HEADCACHE_MAX_DIMS: usize = 32;

/// Squared distance between head row and tail row.
#[cube]
fn layout_distance<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    head: usize,
    tail: usize,
    dimensions: usize,
    #[comptime] shared: bool,
) -> F {
    let mut distance = F::new(0.0f32);
    for c in 0..dimensions {
        let difference = layout_atomic_read(positions, head * dimensions + c)
            - layout_coordinate(positions, fixed_tail, shared, tail, c, dimensions);
        distance += difference * difference;
    }
    distance
}

#[cube]
fn layout_difference<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    head: usize,
    tail: usize,
    c: usize,
    dimensions: usize,
    #[comptime] shared: bool,
) -> F {
    layout_atomic_read(positions, head * dimensions + c)
        - layout_coordinate(positions, fixed_tail, shared, tail, c, dimensions)
}

#[cube]
fn layout_apply_one<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    head: usize,
    tail: usize,
    c: usize,
    dimensions: usize,
    coefficient: F,
    alpha: f32,
    #[comptime] shared: bool,
    #[comptime] move_tail: bool,
) {
    let delta = layout_difference::<F>(positions, fixed_tail, head, tail, c, dimensions, shared);
    let gradient = (coefficient * delta).clamp(F::new(-4.0f32), F::new(4.0f32));
    let update = gradient * F::cast_from(alpha);
    positions[head * dimensions + c].fetch_add(update);
    if comptime![shared && move_tail] {
        positions[tail * dimensions + c].fetch_add(-update);
    }
}

#[cube]
fn layout_apply<F: Float>(
    positions: &Tensor<Atomic<F>>,
    fixed_tail: &Tensor<F>,
    head: usize,
    tail: usize,
    dimensions: usize,
    coefficient: F,
    alpha: f32,
    #[comptime] shared: bool,
    #[comptime] move_tail: bool,
) {
    {
        for c in 0..dimensions {
            layout_apply_one::<F>(
                positions,
                fixed_tail,
                head,
                tail,
                c,
                dimensions,
                coefficient,
                alpha,
                shared,
                move_tail,
            );
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
    head_start: u32,
    head_end: u32,
    #[comptime] shared: bool,
    #[define(F, I, J, M)] _dtypes: [ElemType; 4],
) {
    let vertex = head_start as usize + ABSOLUTE_POS_X as usize;
    if vertex < head_end as usize {
        let dimensions = positions.shape(1);
        let tail_count = tail_rows.shape(0);
        let out_start = u32::cast_from(head_offsets[vertex]) as usize;
        let out_end = u32::cast_from(head_offsets[vertex + 1]) as usize;
        let base = vertex * 3;
        let mut r0 = rng_state[base];
        let mut r1 = rng_state[base + 1];
        let mut r2 = rng_state[base + 2];
        let mut rng_used = false;
        for slot in out_start..out_end {
            let edge = u32::cast_from(head_edges[slot]) as usize;
            let fired = i32::cast_from(mask[edge]);
            if fired >= 0 {
                let tail = u32::cast_from(tails[edge]) as usize;
                if tail < tail_count && (!shared || tail != vertex) {
                    let distance = layout_distance::<F>(
                        positions, tail_rows, vertex, tail, dimensions, shared,
                    );
                    if distance > F::new(0.0f32) {
                        let coefficient = F::cast_from(-2.0f32 * a * b)
                            * distance.powf(F::cast_from(b - 1.0f32))
                            / (F::cast_from(a) * distance.powf(F::cast_from(b)) + F::new(1.0f32));
                        layout_apply::<F>(
                            positions,
                            tail_rows,
                            vertex,
                            tail,
                            dimensions,
                            coefficient,
                            alpha,
                            shared,
                            true,
                        );
                    }
                }
                for _ in 0..u32::cast_from(fired) {
                    rng_used = true;
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
                    let signed_random = i32::cast_from(r0 ^ r1 ^ r2);
                    let remainder = signed_random % i32::cast_from(tail_count);
                    let sample_index = if remainder < 0 {
                        remainder + i32::cast_from(tail_count)
                    } else {
                        remainder
                    };
                    let sampled = u32::cast_from(sample_index) as usize;
                    if !shared || sampled != vertex {
                        let distance = layout_distance::<F>(
                            positions, tail_rows, vertex, sampled, dimensions, shared,
                        );
                        if distance > F::new(0.0f32) {
                            let coefficient = F::cast_from(2.0f32 * gamma * b)
                                / ((F::new(0.001f32) + distance)
                                    * (F::cast_from(a) * distance.powf(F::cast_from(b))
                                        + F::new(1.0f32)));
                            layout_apply::<F>(
                                positions,
                                tail_rows,
                                vertex,
                                sampled,
                                dimensions,
                                coefficient,
                                alpha,
                                shared,
                                false,
                            );
                        }
                    }
                }
            }
        }
        if rng_used {
            rng_state[base] = r0;
            rng_state[base + 1] = r1;
            rng_state[base + 2] = r2;
        }
    }
}

/// head row cached in registers per vertex, own updates applied locally and
/// flushed once as an atomic add; tail reads/moves stay atomic.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn layout_vertex_headcache_kernel<F: Float, I: Int, J: Int, M: Int>(
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
    head_start: u32,
    head_end: u32,
    #[comptime] shared: bool,
    #[comptime] reg_dims: usize,
    #[define(F, I, J, M)] _dtypes: [ElemType; 4],
) {
    let vertex = head_start as usize + ABSOLUTE_POS_X as usize;
    if vertex < head_end as usize {
        let tail_count = tail_rows.shape(0);
        let mut head = Array::<F>::new(reg_dims);
        let mut delta = Array::<F>::new(reg_dims);
        let mut diffs = Array::<F>::new(reg_dims);
        #[unroll]
        for c in 0..reg_dims {
            head[c] = positions[vertex * reg_dims + c].load();
            delta[c] = F::new(0.0f32);
        }
        let out_start = u32::cast_from(head_offsets[vertex]) as usize;
        let out_end = u32::cast_from(head_offsets[vertex + 1]) as usize;
        let base = vertex * 3;
        let mut r0 = rng_state[base];
        let mut r1 = rng_state[base + 1];
        let mut r2 = rng_state[base + 2];
        let mut rng_used = false;
        for slot in out_start..out_end {
            let edge = u32::cast_from(head_edges[slot]) as usize;
            let fired = i32::cast_from(mask[edge]);
            if fired >= 0 {
                let tail = u32::cast_from(tails[edge]) as usize;
                if tail < tail_count && (!shared || tail != vertex) {
                    let mut distance = F::new(0.0f32);
                    #[unroll]
                    for c in 0..reg_dims {
                        let other =
                            layout_coordinate(positions, tail_rows, shared, tail, c, reg_dims);
                        let difference = head[c] - other;
                        diffs[c] = difference;
                        distance += difference * difference;
                    }
                    if distance > F::new(0.0f32) {
                        let coefficient = F::cast_from(-2.0f32 * a * b)
                            * distance.powf(F::cast_from(b - 1.0f32))
                            / (F::cast_from(a) * distance.powf(F::cast_from(b)) + F::new(1.0f32));
                        #[unroll]
                        for c in 0..reg_dims {
                            let gradient =
                                (coefficient * diffs[c]).clamp(F::new(-4.0f32), F::new(4.0f32));
                            let update = gradient * F::cast_from(alpha);
                            head[c] += update;
                            delta[c] += update;
                            if shared {
                                positions[tail * reg_dims + c].fetch_add(-update);
                            }
                        }
                    }
                }
                for _ in 0..u32::cast_from(fired) {
                    rng_used = true;
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
                        #[unroll]
                        for c in 0..reg_dims {
                            let other = layout_coordinate(
                                positions, tail_rows, shared, sampled, c, reg_dims,
                            );
                            let difference = head[c] - other;
                            diffs[c] = difference;
                            distance += difference * difference;
                        }
                        if distance > F::new(0.0f32) {
                            let coefficient = F::cast_from(2.0f32 * gamma * b)
                                / ((F::new(0.001f32) + distance)
                                    * (F::cast_from(a) * distance.powf(F::cast_from(b))
                                        + F::new(1.0f32)));
                            #[unroll]
                            for c in 0..reg_dims {
                                let gradient =
                                    (coefficient * diffs[c]).clamp(F::new(-4.0f32), F::new(4.0f32));
                                let update = gradient * F::cast_from(alpha);
                                head[c] += update;
                                delta[c] += update;
                            }
                        }
                    }
                }
            }
        }
        #[unroll]
        for c in 0..reg_dims {
            positions[vertex * reg_dims + c].fetch_add(delta[c]);
        }
        if rng_used {
            rng_state[base] = r0;
            rng_state[base + 1] = r1;
            rng_state[base + 2] = r2;
        }
    }
}

/// Sum over the `group` adjacent lanes of one event (xor butterfly; every lane of the
/// group receives the bit-identical total).
#[cube]
fn layout_group_sum<F: Float>(value: F, #[comptime] group: usize) -> F {
    let mut total = value;
    #[unroll]
    for step in 0..comptime![group.trailing_zeros() as usize] {
        total += plane_shuffle_xor(total, comptime![1u32 << step as u32]);
    }
    total
}

/// Sum over the lanes with the same in-group index (lanes `sub`, `sub + group`, ...).
#[cube]
fn layout_cross_sum<F: Float>(value: F, #[comptime] group: usize) -> F {
    let mut total = value;
    let mut mask = comptime![group as u32].runtime();
    while mask < 32u32 {
        total += plane_shuffle_xor(total, mask);
        mask *= 2;
    }
    total
}

/// Fused plane kernel: a chunk of up to 32
/// out-edge slots becomes one compacted event list (the chunk's fired positive events,
/// then its negative samples in serial RNG order), processed `32 / group` events
/// per step. Positive and negative events of one step share the head snapshot, and idle
/// slots (unfired edges) take no event lanes. Requires 32-lane planes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn layout_vertex_fused_kernel<F: Float, I: Int, J: Int, M: Int>(
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
    head_start: u32,
    head_end: u32,
    #[comptime] shared: bool,
    #[comptime] reg_dims: usize,
    #[comptime] dims: usize,
    #[comptime] group: usize,
    #[comptime] planes: usize,
    #[define(F, I, J, M)] _dtypes: [ElemType; 4],
) {
    let per = comptime![reg_dims / group];
    let lane = UNIT_POS_PLANE;
    let plane = (UNIT_POS_X / 32u32) as usize;
    let vertex = head_start as usize + CUBE_POS_X as usize * planes + plane;
    let mut positive_tails = Shared::<[u32]>::new_slice(comptime![planes * 32]);
    let tail_base = plane * 32;
    // Plane-uniform: every lane of a plane shares `vertex`.
    if vertex < head_end as usize {
        let tail_count = tail_rows.shape(0);
        let sub = lane as usize % group;
        let event = lane / comptime![group as u32];
        let events = 32u32 / comptime![group as u32];
        let col0 = sub * per;
        let mut head = Array::<F>::new(per);
        let mut delta = Array::<F>::new(per);
        let mut diffs = Array::<F>::new(per);
        let mut step_delta = Array::<F>::new(per);
        #[unroll]
        for j in 0..per {
            // Padded slots (`col0 + j >= dims`) stay 0 and are never loaded or written.
            head[j] = F::new(0.0f32);
            if comptime![dims == reg_dims] || col0 + j < dims {
                head[j] = positions[vertex * dims + col0 + j].load();
            }
            delta[j] = F::new(0.0f32);
        }
        let out_start = u32::cast_from(head_offsets[vertex]) as usize;
        let out_end = u32::cast_from(head_offsets[vertex + 1]) as usize;
        let base = vertex * 3;
        let mut r0 = rng_state[base];
        let mut r1 = rng_state[base + 1];
        let mut r2 = rng_state[base + 2];
        let mut rng_used = false;
        let attract = F::cast_from(-2.0f32 * a * b);
        let repel = F::cast_from(2.0f32 * gamma * b);
        let mut chunk = out_start;
        while chunk < out_end {
            let slot = chunk + lane as usize;
            let mut fired = (-1i32).runtime();
            let mut tail = vertex;
            if slot < out_end {
                let edge = u32::cast_from(head_edges[slot]) as usize;
                fired = i32::cast_from(mask[edge]);
                tail = u32::cast_from(tails[edge]) as usize;
            }
            let positive = fired >= 0 && tail < tail_count && (!shared || tail != vertex);
            let mut positive_flag = 0u32.runtime();
            if positive {
                positive_flag = 1u32;
            }
            let mut negative_count = 0u32.runtime();
            if fired > 0 {
                negative_count = u32::cast_from(fired);
            }
            let positive_rank = plane_exclusive_sum(positive_flag);
            let positive_total = plane_sum(positive_flag);
            let negative_total = plane_sum(negative_count);
            if negative_total > 0 {
                rng_used = true;
            }
            // The previous chunk's reads of `positive_tails` are complete.
            sync_plane();
            if positive {
                positive_tails[tail_base + positive_rank as usize] = tail as u32;
            }
            sync_plane();
            let event_total = positive_total + negative_total;
            let mut generated = 0u32.runtime();
            let mut start = 0u32.runtime();
            while start < event_total {
                let k = start + event;
                let mut covered = start + events;
                if covered > positive_total {
                    covered -= positive_total;
                } else {
                    covered = 0u32;
                }
                if covered > negative_total {
                    covered = negative_total;
                }
                let is_positive = k < positive_total;
                let mut other = vertex;
                if is_positive {
                    other = positive_tails[tail_base + k as usize] as usize;
                }
                let mut is_negative = false;
                let mut captured = r0;
                let mut has_sample = false;
                while generated < covered {
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
                    // Capture only; the modulo runs once per lane after the serial loop.
                    if !is_positive && k < event_total && generated == k - positive_total {
                        captured = r0 ^ r1 ^ r2;
                        has_sample = true;
                    }
                    generated += 1u32;
                }
                if has_sample {
                    let signed_random = i32::cast_from(captured);
                    let remainder = signed_random % i32::cast_from(tail_count);
                    let sample_index = if remainder < 0 {
                        remainder + i32::cast_from(tail_count)
                    } else {
                        remainder
                    };
                    other = u32::cast_from(sample_index) as usize;
                    is_negative = !shared || other != vertex;
                }
                let active = is_positive || is_negative;
                let mut partial = F::new(0.0f32);
                #[unroll]
                for j in 0..per {
                    diffs[j] = F::new(0.0f32);
                    step_delta[j] = F::new(0.0f32);
                }
                if active {
                    #[unroll]
                    for j in 0..per {
                        if comptime![dims == reg_dims] || col0 + j < dims {
                            let value = layout_coordinate(
                                positions,
                                tail_rows,
                                shared,
                                other,
                                col0 + j,
                                dims,
                            );
                            let difference = head[j] - value;
                            diffs[j] = difference;
                            partial += difference * difference;
                        }
                    }
                }
                let distance = layout_group_sum::<F>(partial, group);
                if active && distance > F::new(0.0f32) {
                    let power = distance.powf(F::cast_from(b));
                    let denominator = F::cast_from(a) * power + F::new(1.0f32);
                    let coefficient = if is_positive {
                        attract * power / (distance * denominator)
                    } else {
                        repel / ((F::new(0.001f32) + distance) * denominator)
                    };
                    #[unroll]
                    for j in 0..per {
                        let gradient =
                            (coefficient * diffs[j]).clamp(F::new(-4.0f32), F::new(4.0f32));
                        let update = gradient * F::cast_from(alpha);
                        step_delta[j] = update;
                        if shared && is_positive && (comptime![dims == reg_dims] || col0 + j < dims)
                        {
                            positions[other * dims + col0 + j].fetch_add(-update);
                        }
                    }
                }
                #[unroll]
                for j in 0..per {
                    let total = layout_cross_sum::<F>(step_delta[j], group);
                    head[j] += total;
                    delta[j] += total;
                }
                start += events;
            }
            chunk += 32;
        }
        if event == 0 {
            #[unroll]
            for j in 0..per {
                if comptime![dims == reg_dims] || col0 + j < dims {
                    positions[vertex * dims + col0 + j].fetch_add(delta[j]);
                }
            }
        }
        if rng_used && lane == 0 {
            rng_state[base] = r0;
            rng_state[base + 1] = r1;
            rng_state[base + 2] = r2;
        }
    }
}

const FUSED_PLANES_PER_CUBE: u32 = 2;

#[cfg(test)]
pub(crate) static INJECT_BAD_LAYOUT_LAUNCH: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);

#[derive(Clone, Copy)]
enum LayoutKernel {
    Atomic,
    HeadCache,
    Fused { group: usize, padded: usize },
}

fn plane_capable_values(
    ops: bool,
    sync: bool,
    widths: (u32, u32),
    max_dim: (u32, u32, u32),
    units: u32,
    shared: usize,
) -> bool {
    ops && sync
        && widths == (32, 32)
        && max_dim.0 >= 64
        && max_dim.1 >= 1
        && max_dim.2 >= 1
        && units >= 64
        && shared >= 256
}

fn plane_capable(client: &burn::cubecl::client::Client) -> bool {
    use burn::cubecl::features::Plane;
    let hw = &client.properties().hardware;
    plane_capable_values(
        client.features().plane.contains(Plane::Ops),
        client.features().plane.contains(Plane::Sync),
        (hw.plane_size_min, hw.plane_size_max),
        hw.max_cube_dim,
        hw.max_units_per_cube,
        hw.max_shared_memory_size,
    )
}

/// Fused lane grouping: `(group, padded_width)`. ≤ 16 keeps the measured divisor rule
/// (largest power-of-two divisor with ≥ 4 values per lane, no padding); 17..=32 uses
/// G=4 up to 5 values per lane, else G=8, padded virtually to a multiple of G.
fn fused_layout(dims: usize) -> (usize, usize) {
    debug_assert!((1..=FUSED_MAX_DIMS).contains(&dims));
    let (group, padded) = if dims <= 16 {
        let mut group = 1;
        while dims.is_multiple_of(group * 2) && dims / (group * 2) >= 4 {
            group *= 2;
        }
        (group, dims)
    } else {
        let group = if dims.div_ceil(4) <= 5 { 4 } else { 8 };
        (group, dims.div_ceil(group) * group)
    };
    debug_assert!(padded.is_multiple_of(group) && padded - dims < group);
    debug_assert!(dims <= 16 || padded / group >= 3);
    (group, padded)
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
    assert_eq!(rng_state.shape().dims::<2>(), [head_count, 3]);
    assert_eq!(head_embedding.dtype, DType::F32);
    assert_eq!(tail_embedding.dtype, DType::F32);
    for index_dtype in [
        heads.dtype,
        tails.dtype,
        head_offsets.dtype,
        head_edges.dtype,
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
    let cube_dim = cuscan_common::legal_cube_dim(&client, CubeDim::new_1d(64));
    let mask = empty(&head_embedding, Shape::new([edge_count]), DType::I32);
    layout_clock_kernel::launch(
        &client,
        CubeCount::Static(edge_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
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
        CubeCount::Static(coordinate_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        head_embedding.clone().into_tensor_arg(),
        next.clone().into_tensor_arg(),
    );
    // Shared fit reads updated coordinates atomically from `next`; transform
    // reads the immutable fitted tail and still reads its mutable head atomically.
    let tail_rows = tail_embedding.clone();
    let kernel = if dimensions <= FUSED_MAX_DIMS && plane_capable(&client) {
        let (group, padded) = fused_layout(dimensions);
        LayoutKernel::Fused { group, padded }
    } else if dimensions <= HEADCACHE_MAX_DIMS {
        LayoutKernel::HeadCache
    } else {
        LayoutKernel::Atomic
    };
    let dtypes = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I64),
        dtype_to_storage_type(DType::I32),
        dtype_to_storage_type(DType::I32),
    ];
    let launch_vertex_batch = |head_start: u32, head_end: u32| {
        let count = CubeCount::Static((head_end - head_start).div_ceil(cube_dim.x), 1, 1);
        match kernel {
            LayoutKernel::Fused { group, padded } => {
                #[cfg(test)]
                let fused_dim = {
                    let sentinel =
                        INJECT_BAD_LAYOUT_LAUNCH.load(std::sync::atomic::Ordering::SeqCst);
                    if sentinel != 0 && epoch == 0 && alpha.to_bits() == sentinel {
                        CubeDim::new_1d(2048)
                    } else {
                        CubeDim::new_1d(32 * FUSED_PLANES_PER_CUBE)
                    }
                };
                #[cfg(not(test))]
                let fused_dim = CubeDim::new_1d(32 * FUSED_PLANES_PER_CUBE);
                layout_vertex_fused_kernel::launch(
                    &client,
                    CubeCount::Static(
                        (head_end - head_start).div_ceil(FUSED_PLANES_PER_CUBE),
                        1,
                        1,
                    ),
                    fused_dim,
                    next.clone().into_tensor_arg(),
                    tail_rows.clone().into_tensor_arg(),
                    tails.clone().into_tensor_arg(),
                    head_offsets.clone().into_tensor_arg(),
                    head_edges.clone().into_tensor_arg(),
                    mask.clone().into_tensor_arg(),
                    rng_state.clone().into_tensor_arg(),
                    a,
                    b,
                    gamma,
                    alpha,
                    head_start,
                    head_end,
                    shared_embeddings,
                    padded,
                    dimensions,
                    group,
                    FUSED_PLANES_PER_CUBE as usize,
                    dtypes,
                );
            }
            LayoutKernel::HeadCache => layout_vertex_headcache_kernel::launch(
                &client,
                count,
                cube_dim,
                next.clone().into_tensor_arg(),
                tail_rows.clone().into_tensor_arg(),
                tails.clone().into_tensor_arg(),
                head_offsets.clone().into_tensor_arg(),
                head_edges.clone().into_tensor_arg(),
                mask.clone().into_tensor_arg(),
                rng_state.clone().into_tensor_arg(),
                a,
                b,
                gamma,
                alpha,
                head_start,
                head_end,
                shared_embeddings,
                dimensions,
                dtypes,
            ),
            LayoutKernel::Atomic => layout_vertex_atomic_kernel::launch(
                &client,
                count,
                cube_dim,
                next.clone().into_tensor_arg(),
                tail_rows.clone().into_tensor_arg(),
                tails.clone().into_tensor_arg(),
                head_offsets.clone().into_tensor_arg(),
                head_edges.clone().into_tensor_arg(),
                mask.clone().into_tensor_arg(),
                rng_state.clone().into_tensor_arg(),
                a,
                b,
                gamma,
                alpha,
                head_start,
                head_end,
                shared_embeddings,
                dtypes,
            ),
        }
    };
    if shared_embeddings {
        let batch_size = head_u32.div_ceil(4);
        for batch in 0..4 {
            let start = batch * batch_size;
            let end = start.saturating_add(batch_size).min(head_u32);
            if start < end {
                launch_vertex_batch(start, end);
            }
        }
    } else {
        launch_vertex_batch(0, head_u32);
    }
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

#[cfg(test)]
mod tests {
    use super::{fused_layout, plane_capable_values};

    #[test]
    fn layout_fused_layout_mapping() {
        let expected = [1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 4];
        for (dims, group) in (1..=16).zip(expected) {
            assert_eq!(fused_layout(dims), (group, dims), "dimensions={dims}");
        }
        for dims in 17..=20 {
            assert_eq!(fused_layout(dims), (4, 20), "dimensions={dims}");
        }
        for dims in 21..=24 {
            assert_eq!(fused_layout(dims), (8, 24), "dimensions={dims}");
        }
        for dims in 25..=32 {
            assert_eq!(fused_layout(dims), (8, 32), "dimensions={dims}");
        }
        // G=8 is chosen at 28 (7 values per lane at G=4), so 28 pads to 32 although it
        // is a multiple of 4.
        assert_eq!(fused_layout(28), (8, 32));
        for dims in 1..=32 {
            let (group, padded) = fused_layout(dims);
            assert!(padded.is_multiple_of(group) && padded >= dims && padded - dims < group);
            if dims > 16 {
                assert!(padded / group >= 3, "dimensions={dims}");
            }
        }
    }

    #[test]
    fn layout_plane_capabilities() {
        let check = |ops, sync, widths, dim, units, shared| {
            plane_capable_values(ops, sync, widths, dim, units, shared)
        };
        assert!(check(true, true, (32, 32), (64, 1, 1), 64, 256));
        assert!(!check(false, true, (32, 32), (64, 1, 1), 64, 256));
        assert!(!check(true, false, (32, 32), (64, 1, 1), 64, 256));
        assert!(!check(true, true, (16, 32), (64, 1, 1), 64, 256));
        assert!(!check(true, true, (32, 64), (64, 1, 1), 64, 256));
        assert!(!check(true, true, (32, 32), (63, 1, 1), 64, 256));
        assert!(!check(true, true, (32, 32), (64, 0, 1), 64, 256));
        assert!(!check(true, true, (32, 32), (64, 1, 1), 63, 256));
        assert!(!check(true, true, (32, 32), (64, 1, 1), 64, 255));
    }
}
