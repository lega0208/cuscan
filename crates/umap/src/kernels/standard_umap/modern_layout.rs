//! Low-level ordered modern layout primitives. One-dimensional tensors must be
//! contiguous; coordinate tensors use their row and column strides.
use burn::{
    backend::{
        TensorMetadata,
        tensor::{FloatTensor, IntTensor},
    },
    cubecl::prelude::*,
    tensor::DType,
};
use burn_cubecl::CubeBackend;

type FT = FloatTensor<CubeBackend>;
type IT = IntTensor<CubeBackend>;

#[cube]
fn distance(
    head: &Tensor<f32>,
    tail: &Tensor<f32>,
    i: usize,
    j: usize,
    d: usize,
    #[comptime] shared: bool,
) -> f32 {
    let mut q = 0.0f32;
    for c in 0..d {
        let y = if shared {
            head[j * head.stride(0) + c * head.stride(1)]
        } else {
            tail[j * tail.stride(0) + c * tail.stride(1)]
        };
        let z = head[i * head.stride(0) + c * head.stride(1)] - y;
        q += z * z;
    }
    q
}

#[cube]
fn accumulate(
    head: &Tensor<f32>,
    tail: &Tensor<f32>,
    updates: &mut Tensor<f32>,
    i: usize,
    j: usize,
    d: usize,
    coefficient: f32,
    #[comptime] shared: bool,
) {
    for c in 0..d {
        let y = if shared {
            head[j * head.stride(0) + c * head.stride(1)]
        } else {
            tail[j * tail.stride(0) + c * tail.stride(1)]
        };
        let ui = i * updates.stride(0) + c * updates.stride(1);
        updates[ui] += coefficient * (head[i * head.stride(0) + c * head.stride(1)] - y);
    }
}

/// Whether `j` is a neighbor in the sorted CSR row `begin..stop`, or the
/// source itself when `shared` explicitly selects fit.
#[cube]
fn excluded(
    indices: &Tensor<i32>,
    begin: usize,
    stop: usize,
    i: usize,
    j: usize,
    #[comptime] shared: bool,
) -> bool {
    let mut lo = begin;
    let mut hi = stop;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if usize::cast_from(indices[mid]) < j {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    let mut hit = false;
    if lo < stop {
        hit = usize::cast_from(indices[lo]) == j;
    }
    if shared {
        hit = hit || i == j;
    }
    hit
}

#[cube]
fn repel(
    head: &Tensor<f32>,
    tail: &Tensor<f32>,
    updates: &mut Tensor<f32>,
    i: usize,
    j: usize,
    d: usize,
    a: f32,
    b: f32,
    gamma: f32,
    s: f32,
    multiplier: f32,
    #[comptime] shared: bool,
) {
    let q = distance(head, tail, i, j, d, shared);
    if q > 0.0f32 && gamma > 0.0f32 {
        let coeff = s * 2.0f32 * gamma * b / ((0.001f32 + q) * (1.0f32 + a * q.powf(b)));
        if coeff > 0.0f32 {
            let norm = (coeff * coeff * q).sqrt();
            if norm > 0.0f32 {
                let clip = gamma.min(16.0f32) * (norm / gamma).tanh() / norm;
                accumulate(
                    head,
                    tail,
                    updates,
                    i,
                    j,
                    d,
                    multiplier * coeff * clip,
                    shared,
                );
            }
        }
    }
}

#[cube(launch)]
fn gradient_kernel(
    head: &Tensor<f32>,
    tail: &Tensor<f32>,
    offsets: &Tensor<i32>,
    indices: &Tensor<i32>,
    source_order: &Tensor<i32>,
    negative_order: &Tensor<i32>,
    source_ranks: &Tensor<i32>,
    periods: &Tensor<f64>,
    positive: &mut Tensor<f64>,
    negative_periods: &Tensor<f64>,
    negative: &mut Tensor<f64>,
    updates: &mut Tensor<f32>,
    start: u32,
    end: u32,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    scale: f32,
    negative_range: u32,
    multiplier: f32,
    #[comptime] shared: bool,
    #[comptime] small: bool,
    #[comptime] negatives: bool,
    #[comptime] exclude: bool,
) {
    let slot = ABSOLUTE_POS_X + start;
    if slot < end {
        let i = usize::cast_from(source_order[slot as usize]);
        let d = head.shape(1);
        let n = negative_order.shape(0);
        let n64 = u64::cast_from(n);
        let range = u64::cast_from(negative_range);
        let begin = usize::cast_from(offsets[i]);
        let stop = usize::cast_from(offsets[i + 1]);
        for e in begin..stop {
            if positive[e] <= f64::cast_from(epoch) {
                let j = usize::cast_from(indices[e]);
                let q = distance(head, tail, i, j, d, shared);
                if q > 0.0f32 {
                    let factor = if small { 0.5f32 } else { 1.0f32 };
                    let coeff = factor * multiplier * (-2.0f32 * a * b * q.powf(b - 1.0f32))
                        / (1.0f32 + a * q.powf(b));
                    accumulate(head, tail, updates, i, j, d, coeff, shared);
                }
                positive[e] += periods[e];
                if negatives {
                    let count =
                        i32::cast_from((f64::cast_from(epoch) - negative[e]) / negative_periods[e]);
                    if count > 0 {
                        if small {
                            if exclude {
                                // Retry ranks (e*(epoch+1)+attempt) mod N until
                                // `count` candidates are accepted or count*N
                                // attempts are spent. Base and attempt are
                                // reduced mod N separately to stay within u64.
                                let total = u64::cast_from(count);
                                let attempts = total * n64;
                                let base =
                                    (u64::cast_from(e) * (u64::cast_from(epoch) + 1u64)) % n64;
                                let mut accepted = 0u64;
                                let mut attempt = 0u64;
                                while accepted < total && attempt < attempts {
                                    let rank = (base + attempt % n64) % n64;
                                    attempt += 1u64;
                                    let j =
                                        usize::cast_from(negative_order[usize::cast_from(rank)]);
                                    if !excluded(indices, begin, stop, i, j, shared) {
                                        accepted += 1u64;
                                        repel(
                                            head, tail, updates, i, j, d, a, b, gamma, 1.0f32,
                                            multiplier, shared,
                                        );
                                    }
                                }
                            } else {
                                for p in 0..u32::cast_from(count) {
                                    let hash = (u64::cast_from(e)
                                        * (u64::cast_from(epoch) + u64::cast_from(p) + 1u64))
                                        % n64;
                                    let j =
                                        usize::cast_from(negative_order[usize::cast_from(hash)]);
                                    repel(
                                        head, tail, updates, i, j, d, a, b, gamma, 1.0f32,
                                        multiplier, shared,
                                    );
                                }
                            }
                        } else {
                            // Centered window [first, first+R) in reference
                            // projection order; R=N is global and skips ranks.
                            let mut first = 0u64;
                            if range < n64 {
                                let rank = u64::cast_from(source_ranks[i]);
                                let half = range / 2u64;
                                if rank > half {
                                    first = rank - half;
                                }
                                if first > n64 - range {
                                    first = n64 - range;
                                }
                            }
                            for p in 0..u32::cast_from(count) {
                                let selection = (u64::cast_from(e)
                                    * (u64::cast_from(epoch) + u64::cast_from(p) + 1u64))
                                    % range;
                                let j = usize::cast_from(
                                    negative_order[usize::cast_from(first + selection)],
                                );
                                let mut keep = true;
                                if exclude {
                                    keep = !excluded(indices, begin, stop, i, j, shared);
                                }
                                if keep {
                                    repel(
                                        head, tail, updates, i, j, d, a, b, gamma, scale,
                                        multiplier, shared,
                                    );
                                }
                            }
                        }
                    }
                    negative[e] += f64::cast_from(count) * negative_periods[e];
                }
            }
        }
    }
}

#[cube(launch)]
fn project_kernel(points: &Tensor<f32>, direction: &Tensor<f32>, projection: &mut Tensor<f32>) {
    let i = ABSOLUTE_POS_X as usize;
    if i < points.shape(0) {
        let mut sum = 0.0f32;
        for c in 0..points.shape(1) {
            sum += points[i * points.stride(0) + c * points.stride(1)] * direction[c];
        }
        projection[i] = sum;
    }
}

#[cube(launch)]
fn inverse_rank_kernel(order: &Tensor<i32>, ranks: &mut Tensor<i32>) {
    let j = ABSOLUTE_POS_X as usize;
    if j < order.shape(0) {
        ranks[usize::cast_from(order[j])] = i32::cast_from(j);
    }
}

#[cube(launch)]
fn lower_bound_kernel(sorted: &Tensor<f32>, query: &Tensor<f32>, ranks: &mut Tensor<i32>) {
    let i = ABSOLUTE_POS_X as usize;
    if i < query.shape(0) {
        let n = sorted.shape(0);
        let value = query[i];
        let mut lo = 0usize;
        let mut hi = n;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if sorted[mid] < value {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        ranks[i] = i32::cast_from(lo.min(n - 1));
    }
}

/// Unscaled clipped repulsive factor for one adaptation candidate, matching
/// upstream `resultants`: the delta vector is `factor * (x_src - x_j)`.
#[cube]
fn resultant_factor(
    embedding: &Tensor<f32>,
    offsets: &Tensor<i32>,
    indices: &Tensor<i32>,
    src: usize,
    j: usize,
    a: f64,
    b: f64,
    gamma: f64,
    #[comptime] exclude: bool,
) -> f64 {
    let d = embedding.shape(1);
    let mut keep = true;
    if exclude {
        let begin = usize::cast_from(offsets[src]);
        let stop = usize::cast_from(offsets[src + 1]);
        keep = !excluded(indices, begin, stop, src, j, true);
    }
    let mut factor = 0.0f64;
    if keep {
        let mut q = 0.0f64;
        for c in 0..d {
            let z = f64::cast_from(embedding[src * embedding.stride(0) + c * embedding.stride(1)])
                - f64::cast_from(embedding[j * embedding.stride(0) + c * embedding.stride(1)]);
            q += z * z;
        }
        let coefficient = 2.0f64 * gamma * b / ((0.001f64 + q) * (a * q.powf(b) + 1.0f64));
        let norm = coefficient * q.sqrt();
        // Upstream clips only positive norms; a zero norm (zero distance or
        // zero gamma) yields a zero vector, so skip it without dividing.
        if norm > 0.0f64 {
            factor = coefficient * gamma * (norm / gamma).tanh() / norm;
        }
    }
    factor
}

#[cube]
fn resultant_candidate(
    order: &Tensor<i32>,
    sample_ranks: &Tensor<i32>,
    local_offsets: &Tensor<i32>,
    global_ranks: &Tensor<i32>,
    s: usize,
    k: usize,
    local: bool,
    raw_range: u32,
) -> usize {
    let n = order.shape(0);
    let mut rank = usize::cast_from(global_ranks[s * 5 + k]);
    if local {
        let range = raw_range as usize;
        let source_rank = usize::cast_from(sample_ranks[s]);
        let half = range / 2;
        let mut first = 0usize;
        if source_rank > half {
            first = source_rank - half;
        }
        if first > n - range {
            first = n - range;
        }
        rank = first + usize::cast_from(local_offsets[s * 5 + k]);
    }
    usize::cast_from(order[rank])
}

#[cube(launch)]
fn resultant_kernel(
    embedding: &Tensor<f32>,
    order: &Tensor<i32>,
    offsets: &Tensor<i32>,
    indices: &Tensor<i32>,
    sample_ranks: &Tensor<i32>,
    local_offsets: &Tensor<i32>,
    global_ranks: &Tensor<i32>,
    resultants: &mut Tensor<f64>,
    raw_range: u32,
    a: f32,
    b: f32,
    gamma: f32,
    #[comptime] exclude: bool,
) {
    let t = ABSOLUTE_POS_X as usize;
    let samples = sample_ranks.shape(0);
    if t < 2 * samples {
        let local = t >= samples;
        let mut s = t;
        if local {
            s = t - samples;
        }
        let src = usize::cast_from(order[usize::cast_from(sample_ranks[s])]);
        let a64 = f64::cast_from(a);
        let b64 = f64::cast_from(b);
        let gamma64 = f64::cast_from(gamma);
        let j0 = resultant_candidate(
            order,
            sample_ranks,
            local_offsets,
            global_ranks,
            s,
            0,
            local,
            raw_range,
        );
        let j1 = resultant_candidate(
            order,
            sample_ranks,
            local_offsets,
            global_ranks,
            s,
            1,
            local,
            raw_range,
        );
        let j2 = resultant_candidate(
            order,
            sample_ranks,
            local_offsets,
            global_ranks,
            s,
            2,
            local,
            raw_range,
        );
        let j3 = resultant_candidate(
            order,
            sample_ranks,
            local_offsets,
            global_ranks,
            s,
            3,
            local,
            raw_range,
        );
        let j4 = resultant_candidate(
            order,
            sample_ranks,
            local_offsets,
            global_ranks,
            s,
            4,
            local,
            raw_range,
        );
        let f0 = resultant_factor(
            embedding, offsets, indices, src, j0, a64, b64, gamma64, exclude,
        );
        let f1 = resultant_factor(
            embedding, offsets, indices, src, j1, a64, b64, gamma64, exclude,
        );
        let f2 = resultant_factor(
            embedding, offsets, indices, src, j2, a64, b64, gamma64, exclude,
        );
        let f3 = resultant_factor(
            embedding, offsets, indices, src, j3, a64, b64, gamma64, exclude,
        );
        let f4 = resultant_factor(
            embedding, offsets, indices, src, j4, a64, b64, gamma64, exclude,
        );
        let mut squared = 0.0f64;
        for c in 0..embedding.shape(1) {
            let col = c * embedding.stride(1);
            let x = f64::cast_from(embedding[src * embedding.stride(0) + col]);
            let sum = f0 * (x - f64::cast_from(embedding[j0 * embedding.stride(0) + col]))
                + f1 * (x - f64::cast_from(embedding[j1 * embedding.stride(0) + col]))
                + f2 * (x - f64::cast_from(embedding[j2 * embedding.stride(0) + col]))
                + f3 * (x - f64::cast_from(embedding[j3 * embedding.stride(0) + col]))
                + f4 * (x - f64::cast_from(embedding[j4 * embedding.stride(0) + col]));
            squared += sum * sum;
        }
        resultants[t] = squared.sqrt();
    }
}

#[cube(launch)]
fn projection_status_reset(status: &mut Tensor<Atomic<u32>>) {
    if ABSOLUTE_POS_X == 0 {
        status[0].store(0xFFFF_FFFFu32);
    }
}

/// Record the lowest row whose projection is NaN or infinite.
#[cube(launch)]
fn projection_status_kernel(projection: &Tensor<f32>, status: &mut Tensor<Atomic<u32>>) {
    let i = ABSOLUTE_POS_X as usize;
    if i < projection.shape(0) {
        let bits = u32::reinterpret(projection[i]);
        if (bits & 0x7F80_0000u32) == 0x7F80_0000u32 {
            status[0].fetch_min(u32::cast_from(i));
        }
    }
}

#[cube(launch)]
fn adam_kernel(
    head: &mut Tensor<f32>,
    updates: &mut Tensor<f32>,
    m: &mut Tensor<f32>,
    v: &mut Tensor<f32>,
    alpha: f32,
    correction1: f32,
    correction2: f32,
) {
    let i = ABSOLUTE_POS_X as usize;
    if i < head.shape(0) * head.shape(1) {
        let row = i / head.shape(1);
        let col = i % head.shape(1);
        let hi = row * head.stride(0) + col * head.stride(1);
        let ui = row * updates.stride(0) + col * updates.stride(1);
        let mi = row * m.stride(0) + col * m.stride(1);
        let vi = row * v.stride(0) + col * v.stride(1);
        let g = updates[ui];
        if g != 0.0f32 {
            let first = 0.9f32 * m[mi] + (1.0f32 - 0.9f32) * g;
            let second = 0.99f32 * v[vi] + (1.0f32 - 0.99f32) * g * g;
            m[mi] = first;
            v[vi] = second;
            head[hi] += alpha * (first / correction1) / ((second / correction2).sqrt() + 0.0001f32);
        }
        updates[ui] = 0.0f32;
    }
}

#[cube(launch)]
fn apply_kernel(
    head: &mut Tensor<f32>,
    velocity: &Tensor<f32>,
    order: &Tensor<i32>,
    start: u32,
    end: u32,
) {
    let slot = ABSOLUTE_POS_X + start;
    if slot < end {
        let row = usize::cast_from(order[slot as usize]);
        let d = head.shape(1);
        for c in 0..d {
            let hi = row * head.stride(0) + c * head.stride(1);
            let vi = row * velocity.stride(0) + c * velocity.stride(1);
            head[hi] += velocity[vi];
        }
    }
}

#[cube(launch)]
fn decay_kernel(velocity: &mut Tensor<f32>, mu: f32) {
    let i = ABSOLUTE_POS_X as usize;
    if i < velocity.shape(0) * velocity.shape(1) {
        let row = i / velocity.shape(1);
        let col = i % velocity.shape(1);
        let vi = row * velocity.stride(0) + col * velocity.stride(1);
        velocity[vi] *= mu;
    }
}

/// In-place gradient; multiplier is 1 for Adam and epoch alpha for momentum.
/// `start..end` indexes source_order, not source rows. Clock periods must be
/// positive and representable; when negative sampling is disabled set negatives=false.
///
/// `negative_range` is the caller-resolved effective window R in `1..=N`,
/// N = `negative_order` length. Large layouts (H > 1024) read
/// `negative_order[clamp(source_ranks[i] - R/2, 0, N-R) + e*(epoch+p+1) mod R]`
/// using u64 arithmetic; `source_ranks` (fit inverse ranks or transform
/// insertion ranks, indexed by source row, values in `0..N`) is read only when
/// R < N. Small layouts keep global unit-scale selection and ignore R/ranks.
/// With `exclude`, candidates found by binary search in the current sorted CSR
/// row, or the source itself when `shared` (explicit fit), are dropped in large
/// layouts and retried in small layouts as `(e*(epoch+1)+attempt) mod N` for at
/// most count*N attempts. Rejected candidates still consume negative clocks.
#[allow(clippy::too_many_arguments)]
pub(crate) fn modern_gradient_block(
    head: &FT,
    tail: &FT,
    offsets: &IT,
    indices: &IT,
    source_order: &IT,
    negative_order: &IT,
    source_ranks: &IT,
    periods: &FT,
    positive: &FT,
    negative_periods: &FT,
    negative: &FT,
    updates: &FT,
    start: u32,
    end: u32,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    scale: f32,
    negative_range: u32,
    multiplier: f32,
    shared: bool,
    negatives: bool,
    exclude: bool,
) {
    assert!(start <= end && end as usize <= source_order.shape().dims::<1>()[0]);
    let references = negative_order.shape().dims::<1>()[0];
    assert!(
        negative_range >= 1 && negative_range as usize <= references,
        "negative range must be in 1..=N"
    );
    let small = head.shape().dims::<2>()[0] <= 1024;
    if !small && (negative_range as usize) < references {
        assert_eq!(
            source_ranks.shape().dims::<1>()[0],
            head.shape().dims::<2>()[0],
            "source ranks must cover every source row"
        );
    }
    assert_eq!(periods.dtype, DType::F64);
    assert_eq!(positive.dtype, DType::F64);
    assert_eq!(negative_periods.dtype, DType::F64);
    assert_eq!(negative.dtype, DType::F64);
    if start == end {
        return;
    }
    let client = &head.client;
    let cube = cuscan_common::legal_cube_dim(client, CubeDim::new_1d(64));
    gradient_kernel::launch(
        client,
        CubeCount::Static((end - start).div_ceil(cube.x), 1, 1),
        cube,
        head.clone().into_tensor_arg(),
        tail.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        indices.clone().into_tensor_arg(),
        source_order.clone().into_tensor_arg(),
        negative_order.clone().into_tensor_arg(),
        source_ranks.clone().into_tensor_arg(),
        periods.clone().into_tensor_arg(),
        positive.clone().into_tensor_arg(),
        negative_periods.clone().into_tensor_arg(),
        negative.clone().into_tensor_arg(),
        updates.clone().into_tensor_arg(),
        start,
        end,
        epoch,
        a,
        b,
        gamma,
        scale,
        negative_range,
        multiplier,
        shared,
        small,
        negatives,
        exclude,
    );
}

fn launch_rows(client: &burn::cubecl::client::Client, rows: usize) -> Option<(CubeCount, CubeDim)> {
    let rows = u32::try_from(rows).expect("projection rows exceed CubeCL range");
    if rows == 0 {
        return None;
    }
    let cube = cuscan_common::legal_cube_dim(client, CubeDim::new_1d(64));
    Some((CubeCount::Static(rows.div_ceil(cube.x), 1, 1), cube))
}

/// Project coordinates onto a caller-supplied unit `direction` (contiguous
/// f32, length D) into preallocated contiguous f32 outputs, using each
/// coordinate tensor's physical row/column strides. `reference_projection`
/// has N entries. When `shared` (explicit fit) `source` and
/// `source_projection` are ignored; otherwise `source_projection` receives the
/// H query projections and `reference` is never written. Each projection is a
/// sequential f32 sum over columns. No allocation, host sort or readback;
/// finite checking is owned by the sort and runtime validation.
pub(crate) fn modern_project(
    reference: &FT,
    source: &FT,
    direction: &FT,
    reference_projection: &FT,
    source_projection: &FT,
    shared: bool,
) {
    let [rows, dims] = reference.shape().dims::<2>();
    assert_eq!(direction.shape().dims::<1>()[0], dims);
    assert_eq!(reference_projection.shape().dims::<1>()[0], rows);
    let client = &reference.client;
    if let Some((count, cube)) = launch_rows(client, rows) {
        project_kernel::launch(
            client,
            count,
            cube,
            reference.clone().into_tensor_arg(),
            direction.clone().into_tensor_arg(),
            reference_projection.clone().into_tensor_arg(),
        );
    }
    if !shared {
        let [rows, source_dims] = source.shape().dims::<2>();
        assert_eq!(source_dims, dims);
        assert_eq!(source_projection.shape().dims::<1>()[0], rows);
        if let Some((count, cube)) = launch_rows(client, rows) {
            project_kernel::launch(
                client,
                count,
                cube,
                source.clone().into_tensor_arg(),
                direction.clone().into_tensor_arg(),
                source_projection.clone().into_tensor_arg(),
            );
        }
    }
}

/// Write per-source window ranks into preallocated contiguous i32 `ranks`.
/// `order` is the stable ascending reference permutation (N entries) and
/// `sorted_projection` the matching sorted projections. When `shared`
/// (explicit fit) ranks has N entries and receives the inverse permutation
/// `ranks[order[j]] = j`; `query_projection` and `sorted_projection` are
/// ignored. Otherwise (transform, including H = N) ranks has H entries and
/// receives the lower bound of each query projection in `sorted_projection`
/// clamped to N-1; N must be positive. Ties insert before equal references.
pub(crate) fn modern_projection_ranks(
    order: &IT,
    sorted_projection: &FT,
    query_projection: &FT,
    ranks: &IT,
    shared: bool,
) {
    let references = order.shape().dims::<1>()[0];
    let client = &order.client;
    if shared {
        assert_eq!(ranks.shape().dims::<1>()[0], references);
        if let Some((count, cube)) = launch_rows(client, references) {
            inverse_rank_kernel::launch(
                client,
                count,
                cube,
                order.clone().into_tensor_arg(),
                ranks.clone().into_tensor_arg(),
            );
        }
    } else {
        assert!(references > 0, "transform ranks require reference rows");
        assert_eq!(sorted_projection.shape().dims::<1>()[0], references);
        let queries = query_projection.shape().dims::<1>()[0];
        assert_eq!(ranks.shape().dims::<1>()[0], queries);
        if let Some((count, cube)) = launch_rows(client, queries) {
            lower_bound_kernel::launch(
                client,
                count,
                cube,
                sorted_projection.clone().into_tensor_arg(),
                query_projection.clone().into_tensor_arg(),
                ranks.clone().into_tensor_arg(),
            );
        }
    }
}

/// Fit-only adaptation resultants (upstream `_sample_negative_force_ratio`
/// `resultants`), written to preallocated contiguous F64 `resultants` of
/// length 2*S: global resultants at `[0, S)`, local at `[S, 2S)`. Inputs are
/// contiguous i32: `sample_ranks` (S ranks in `0..N`, with replacement),
/// `local_offsets` and `global_ranks` (S*5 each, offsets in `0..raw_range`,
/// ranks in `0..N`). Source is `order[sample_rank]`; local candidates are
/// `order[clamp(rank - Q/2, 0, N-Q) + offset]` with raw Q = `raw_range` in
/// `1..=N`. Each of the five candidates contributes the unscaled clipped
/// vector `c*gamma*tanh(norm/gamma)/norm * (x_src - x_j)` with
/// `c = 2*gamma*b/((0.001+q)*(a*q^b+1))`, `norm = c*sqrt(q)` (no 16 cap),
/// zero when `norm` is not positive; with `exclude`, candidates equal to the
/// source or in its sorted CSR row contribute zero. Accumulation is scalar
/// f64 from strided f32 coordinates; the stored value is the Euclidean norm
/// of the summed vector. No allocation or readback.
#[allow(clippy::too_many_arguments)]
pub(crate) fn modern_sample_resultants(
    embedding: &FT,
    order: &IT,
    offsets: &IT,
    indices: &IT,
    sample_ranks: &IT,
    local_offsets: &IT,
    global_ranks: &IT,
    resultants: &FT,
    raw_range: u32,
    a: f32,
    b: f32,
    gamma: f32,
    exclude: bool,
) {
    let references = order.shape().dims::<1>()[0];
    assert_eq!(embedding.shape().dims::<2>()[0], references);
    assert!(
        raw_range >= 1 && raw_range as usize <= references,
        "raw range must be in 1..=N"
    );
    let samples = sample_ranks.shape().dims::<1>()[0];
    let candidates = samples
        .checked_mul(5)
        .expect("adaptation candidates overflow");
    assert_eq!(local_offsets.shape().dims::<1>()[0], candidates);
    assert_eq!(global_ranks.shape().dims::<1>()[0], candidates);
    assert_eq!(resultants.dtype, DType::F64);
    let outputs = samples.checked_mul(2).expect("adaptation outputs overflow");
    assert_eq!(resultants.shape().dims::<1>()[0], outputs);
    let client = &embedding.client;
    if let Some((count, cube)) = launch_rows(client, outputs) {
        resultant_kernel::launch(
            client,
            count,
            cube,
            embedding.clone().into_tensor_arg(),
            order.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            sample_ranks.clone().into_tensor_arg(),
            local_offsets.clone().into_tensor_arg(),
            global_ranks.clone().into_tensor_arg(),
            resultants.clone().into_tensor_arg(),
            raw_range,
            a,
            b,
            gamma,
            exclude,
        );
    }
}

/// Validate transform query projections before `modern_projection_ranks`.
/// `status` is a caller-preallocated one-word 32-bit integer tensor; a reset
/// launch writes `u32::MAX` (i32 bits -1) and the scan atomically stores the
/// lowest non-finite row with `fetch_min`. Both launches are ordered on the
/// projection's client; the caller reads back the single word.
pub(crate) fn modern_query_projection_status(query_projection: &FT, status: &IT) {
    assert_eq!(status.shape().num_elements(), 1);
    let client = &query_projection.client;
    projection_status_reset::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        status.clone().into_tensor_arg(),
    );
    let rows = query_projection.shape().dims::<1>()[0];
    if let Some((count, cube)) = launch_rows(client, rows) {
        projection_status_kernel::launch(
            client,
            count,
            cube,
            query_projection.clone().into_tensor_arg(),
            status.clone().into_tensor_arg(),
        );
    }
}

/// Bias corrections are 1-beta^(epoch+1), computed for the global epoch.
pub(crate) fn modern_adam_step(
    head: &FT,
    updates: &FT,
    m: &FT,
    v: &FT,
    alpha: f32,
    correction1: f32,
    correction2: f32,
) {
    let count =
        u32::try_from(head.shape().num_elements()).expect("coordinate count exceeds CubeCL range");
    if count == 0 {
        return;
    }
    let client = &head.client;
    let cube = cuscan_common::legal_cube_dim(client, CubeDim::new_1d(64));
    adam_kernel::launch(
        client,
        CubeCount::Static(count.div_ceil(cube.x), 1, 1),
        cube,
        head.clone().into_tensor_arg(),
        updates.clone().into_tensor_arg(),
        m.clone().into_tensor_arg(),
        v.clone().into_tensor_arg(),
        alpha,
        correction1,
        correction2,
    );
}

pub(crate) fn modern_momentum_apply_block(
    head: &FT,
    velocity: &FT,
    order: &IT,
    start: u32,
    end: u32,
) {
    assert!(start <= end && end as usize <= order.shape().dims::<1>()[0]);
    if start == end {
        return;
    }
    let client = &head.client;
    let cube = cuscan_common::legal_cube_dim(client, CubeDim::new_1d(64));
    apply_kernel::launch(
        client,
        CubeCount::Static((end - start).div_ceil(cube.x), 1, 1),
        cube,
        head.clone().into_tensor_arg(),
        velocity.clone().into_tensor_arg(),
        order.clone().into_tensor_arg(),
        start,
        end,
    );
}

pub(crate) fn modern_momentum_finalize(velocity: &FT, mu: f32) {
    let count = u32::try_from(velocity.shape().num_elements())
        .expect("coordinate count exceeds CubeCL range");
    if count == 0 {
        return;
    }
    let client = &velocity.client;
    let cube = cuscan_common::legal_cube_dim(client, CubeDim::new_1d(64));
    decay_kernel::launch(
        client,
        CubeCount::Static(count.div_ceil(cube.x), 1, 1),
        cube,
        velocity.clone().into_tensor_arg(),
        mu,
    );
}

/// Consume preallocated buffers and enqueue one epoch on their ordered client.
/// Cloned handles refer to the same storage; callers must uphold the backend
/// operation's shape, one-dimensional contiguity, aliasing, and buffer-lifetime preconditions.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn modern_epoch(
    head: FT,
    fixed_tail: FT,
    offsets: IT,
    indices: IT,
    source_order: IT,
    negative_order: IT,
    source_ranks: IT,
    periods: FT,
    next_positive: FT,
    negative_periods: FT,
    next_negative: FT,
    updates: FT,
    m: FT,
    v: FT,
    epoch: u32,
    a: f32,
    b: f32,
    gamma: f32,
    scale: f32,
    negative_range: u32,
    alpha: f32,
    mu: f32,
    correction1: f32,
    correction2: f32,
    momentum: bool,
    shared: bool,
    negatives: bool,
    exclude_graph_neighbors: bool,
) -> (FT, FT, FT, FT, FT, FT) {
    let rows = u32::try_from(head.shape().dims::<2>()[0]).expect("layout rows exceed CubeCL range");
    assert!(rows > 0, "modern layout requires source rows");
    if momentum {
        let block = if rows <= 1024 {
            rows
        } else {
            (rows / 2).max(1)
        };
        let mut start = 0;
        while start < rows {
            let end = start.saturating_add(block).min(rows);
            modern_gradient_block(
                &head,
                &fixed_tail,
                &offsets,
                &indices,
                &source_order,
                &negative_order,
                &source_ranks,
                &periods,
                &next_positive,
                &negative_periods,
                &next_negative,
                &updates,
                start,
                end,
                epoch,
                a,
                b,
                gamma,
                scale,
                negative_range,
                alpha,
                shared,
                negatives,
                exclude_graph_neighbors,
            );
            modern_momentum_apply_block(&head, &updates, &source_order, start, end);
            start = end;
        }
        modern_momentum_finalize(&updates, mu);
    } else {
        modern_gradient_block(
            &head,
            &fixed_tail,
            &offsets,
            &indices,
            &source_order,
            &negative_order,
            &source_ranks,
            &periods,
            &next_positive,
            &negative_periods,
            &next_negative,
            &updates,
            0,
            rows,
            epoch,
            a,
            b,
            gamma,
            scale,
            negative_range,
            1.0,
            shared,
            negatives,
            exclude_graph_neighbors,
        );
        modern_adam_step(&head, &updates, &m, &v, alpha, correction1, correction2);
    }
    (head, next_positive, next_negative, updates, m, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::ops::{FloatTensorOps, IntTensorOps},
        tensor::TensorData,
    };
    use burn_cubecl::{CubeDevice, ops::into_data_sync};
    use umap_core::modern_layout::{Schedule32, accumulate_force, adam_step, momentum_finalize};

    fn f32_tensor(device: &CubeDevice, values: Vec<f32>, shape: [usize; 2]) -> FT {
        CubeBackend::float_from_data(TensorData::new(values, shape), device)
    }
    fn f64_tensor(device: &CubeDevice, values: Vec<f64>) -> FT {
        let n = values.len();
        CubeBackend::float_from_data(TensorData::new(values, [n]), device)
    }
    fn i32_tensor(device: &CubeDevice, values: Vec<i32>) -> IT {
        let n = values.len();
        CubeBackend::int_from_data(TensorData::new(values, [n]), device)
    }
    fn read32(t: &FT) -> Vec<f32> {
        into_data_sync(t.clone()).try_to_vec::<f32>().unwrap()
    }
    fn read64(t: &FT) -> Vec<f64> {
        into_data_sync(t.clone()).try_to_vec::<f64>().unwrap()
    }
    fn close(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len());
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 2e-4 * (1.0 + b.abs()),
                "{context}: coordinate {i}: {a} != {b}"
            );
        }
    }

    // Sparse but nontrivial compact CSR: empty rows, two different clocks and
    // distinct source/negative permutations. All rows nevertheless participate
    // in the step, exposing lazy Adam and persistent velocity behavior.
    fn fixture(
        device: &CubeDevice,
        rows: usize,
        dims: usize,
        shared: bool,
        adam: bool,
        gamma: f32,
    ) {
        let initial: Vec<f32> = (0..rows * dims)
            .map(|i| ((i / dims % 11) as f32) * 0.13 + ((i % dims) as f32) * 0.007)
            .collect();
        let reference: Vec<f32> = initial.iter().map(|x| x * 0.7 + 0.19).collect();
        let mut expected = initial.clone();
        let mut velocity = vec![0.0; rows * dims];
        let mut first = vec![0.0; rows * dims];
        let mut second = vec![0.0; rows * dims];
        // Give an untouched row nonzero optimizer state to distinguish lazy
        // Adam from unconditional decay and momentum from zeroing updates.
        velocity[2 * dims] = 0.12;
        first[2 * dims] = 0.3;
        second[2 * dims] = 0.2;
        let mut offsets = vec![3i32; rows + 1];
        offsets[0] = 0;
        offsets[1] = 2;
        offsets[2] = 3;
        let indices = vec![1i32, 2, 0];
        let source: Vec<i32> = (0..rows as i32).rev().collect();
        let negative: Vec<i32> = (0..rows as i32).map(|i| (i + 1) % rows as i32).collect();
        let periods = vec![1.0f64, 2.0, 1.0];
        let negative_periods = vec![0.5f64, 1.0, 0.5];
        let mut pos = periods.clone();
        let mut neg = negative_periods.clone();
        let head = f32_tensor(device, initial, [rows, dims]);
        let tail = f32_tensor(device, reference.clone(), [rows, dims]);
        let off = i32_tensor(device, offsets.clone());
        let idx = i32_tensor(device, indices.clone());
        let order = i32_tensor(device, source.clone());
        let neg_order = i32_tensor(device, negative.clone());
        let per = f64_tensor(device, periods.clone());
        let next = f64_tensor(device, pos.clone());
        let nper = f64_tensor(device, negative_periods.clone());
        let nnext = f64_tensor(device, neg.clone());
        let update = f32_tensor(
            device,
            if adam {
                vec![0.0; rows * dims]
            } else {
                velocity.clone()
            },
            [rows, dims],
        );
        let m = f32_tensor(device, first.clone(), [rows, dims]);
        let v = f32_tensor(device, second.clone(), [rows, dims]);
        let small = rows <= 1024;
        for epoch in 1..=3u32 {
            let context = format!(
                "rows={rows} dims={dims} shared={shared} adam={adam} gamma={gamma} epoch={epoch}"
            );
            let block = if adam || small { rows } else { rows / 2 };
            let mut gradient = vec![0.0f32; rows * dims];
            for start in (0..rows).step_by(block) {
                let end = (start + block).min(rows);
                let snapshot = expected.clone();
                for &row in &source[start..end] {
                    let row = row as usize;
                    for edge in offsets[row] as usize..offsets[row + 1] as usize {
                        if pos[edge] > epoch as f64 {
                            continue;
                        }
                        let target = if shared { &snapshot } else { &reference };
                        let i = row * dims;
                        let j = indices[edge] as usize * dims;
                        accumulate_force(
                            &mut gradient[i..i + dims],
                            &snapshot[i..i + dims],
                            &target[j..j + dims],
                            1.3,
                            0.8,
                            gamma,
                            0.4,
                            small,
                            false,
                        )
                        .unwrap();
                        pos[edge] += periods[edge];
                        let count =
                            ((epoch as f64 - neg[edge]) / negative_periods[edge]).trunc() as i32;
                        for p in 0..count.max(0) {
                            let selection = (edge * (epoch as usize + p as usize + 1)) % rows;
                            let j = negative[selection] as usize * dims;
                            accumulate_force(
                                &mut gradient[i..i + dims],
                                &snapshot[i..i + dims],
                                &target[j..j + dims],
                                1.3,
                                0.8,
                                gamma,
                                0.4,
                                small,
                                true,
                            )
                            .unwrap();
                        }
                        neg[edge] += count as f64 * negative_periods[edge];
                    }
                }
                modern_gradient_block(
                    &head,
                    &tail,
                    &off,
                    &idx,
                    &order,
                    &neg_order,
                    &order,
                    &per,
                    &next,
                    &nper,
                    &nnext,
                    &update,
                    start as u32,
                    end as u32,
                    epoch,
                    1.3,
                    0.8,
                    gamma,
                    0.4,
                    rows as u32,
                    if adam { 1.0 } else { 0.2 },
                    shared,
                    true,
                    false,
                );
                if !adam {
                    for &row in &source[start..end] {
                        for c in row as usize * dims..(row as usize + 1) * dims {
                            velocity[c] += 0.2 * gradient[c];
                        }
                    }
                    for &row in &source[start..end] {
                        for c in row as usize * dims..(row as usize + 1) * dims {
                            expected[c] += velocity[c];
                        }
                    }
                    modern_momentum_apply_block(&head, &update, &order, start as u32, end as u32);
                }
            }
            if adam {
                let schedule = Schedule32 {
                    alpha: 0.2,
                    momentum: 0.0,
                    beta1: 0.9,
                    beta2: 0.99,
                };
                for i in 0..expected.len() {
                    adam_step(
                        &mut expected[i],
                        &mut gradient[i],
                        &mut first[i],
                        &mut second[i],
                        epoch as usize,
                        schedule,
                    );
                }
                let c1 = (1.0 - (0.9f32 as f64).powi(epoch as i32 + 1)) as f32;
                let c2 = (1.0 - (0.99f32 as f64).powi(epoch as i32 + 1)) as f32;
                modern_adam_step(&head, &update, &m, &v, 0.2, c1, c2);
                close(&read32(&m), &first, &format!("{context} first moment"));
                close(&read32(&v), &second, &format!("{context} second moment"));
                close(
                    &read32(&update),
                    &gradient,
                    &format!("{context} cleared gradients"),
                );
            } else {
                momentum_finalize(&mut velocity, 0.37);
                modern_momentum_finalize(&update, 0.37);
                close(&read32(&update), &velocity, &format!("{context} velocity"));
            }
            close(&read32(&head), &expected, &format!("{context} head"));
            assert_eq!(read64(&next), pos);
            assert_eq!(read64(&nnext), neg);
            close(&read32(&tail), &reference, &format!("{context} tail"));
        }
    }

    fn singleton_empty_row(device: &CubeDevice, shared: bool) {
        let initial: Vec<f32> = (0..48).map(|i| i as f32 * 0.01).collect();
        let reference: Vec<f32> = initial.iter().map(|x| x + 1.0).collect();
        let velocity: Vec<f32> = (0..48).map(|i| 0.1 + i as f32 * 0.001).collect();
        let head = f32_tensor(device, initial.clone(), [1, 48]);
        let tail = f32_tensor(device, reference.clone(), [1, 48]);
        let offsets = i32_tensor(device, vec![0, 0]);
        let indices = i32_tensor(device, vec![0]);
        let order = i32_tensor(device, vec![0]);
        let periods = f64_tensor(device, vec![1.0]);
        let positive = f64_tensor(device, vec![1.0]);
        let negative_periods = f64_tensor(device, vec![0.5]);
        let negative = f64_tensor(device, vec![0.5]);
        let updates = f32_tensor(device, velocity.clone(), [1, 48]);
        modern_gradient_block(
            &head,
            &tail,
            &offsets,
            &indices,
            &order,
            &order,
            &order,
            &periods,
            &positive,
            &negative_periods,
            &negative,
            &updates,
            0,
            1,
            2,
            1.3,
            0.8,
            1.2,
            0.4,
            1,
            0.2,
            shared,
            true,
            false,
        );
        modern_momentum_apply_block(&head, &updates, &order, 0, 1);
        modern_momentum_finalize(&updates, 0.37);
        let moved: Vec<f32> = initial.iter().zip(&velocity).map(|(x, v)| x + v).collect();
        let decayed: Vec<f32> = velocity.iter().map(|v| v * 0.37).collect();
        close(&read32(&head), &moved, "singleton momentum head");
        close(&read32(&updates), &decayed, "singleton momentum velocity");
        assert_eq!(read64(&positive), vec![1.0]);
        assert_eq!(read64(&negative), vec![0.5]);
        close(&read32(&tail), &reference, "singleton momentum fixed tail");

        let head = f32_tensor(device, initial.clone(), [1, 48]);
        let gradient = f32_tensor(device, vec![0.0; 48], [1, 48]);
        let first = f32_tensor(device, vec![0.3; 48], [1, 48]);
        let second = f32_tensor(device, vec![0.2; 48], [1, 48]);
        modern_gradient_block(
            &head,
            &tail,
            &offsets,
            &indices,
            &order,
            &order,
            &order,
            &periods,
            &positive,
            &negative_periods,
            &negative,
            &gradient,
            0,
            1,
            2,
            1.3,
            0.8,
            1.2,
            0.4,
            1,
            1.0,
            shared,
            true,
            false,
        );
        modern_adam_step(&head, &gradient, &first, &second, 0.2, 0.271, 0.029701);
        close(&read32(&head), &initial, "singleton lazy Adam head");
        close(&read32(&gradient), &[0.0; 48], "singleton cleared gradient");
        close(&read32(&first), &[0.3; 48], "singleton lazy first moment");
        close(&read32(&second), &[0.2; 48], "singleton lazy second moment");
        assert_eq!(read64(&positive), vec![1.0]);
        assert_eq!(read64(&negative), vec![0.5]);
        close(&read32(&tail), &reference, "singleton Adam fixed tail");
    }

    fn f32_1d(device: &CubeDevice, values: Vec<f32>) -> FT {
        let n = values.len();
        CubeBackend::float_from_data(TensorData::new(values, [n]), device)
    }
    fn i32_read(t: &IT) -> Vec<i32> {
        into_data_sync(t.clone()).try_to_vec::<i32>().unwrap()
    }

    /// Row-major [rows, dims] values; `transposed` stores them column-major
    /// and exposes a [rows, dims] view with strides (1, rows).
    fn coords(
        device: &CubeDevice,
        values: &[f32],
        rows: usize,
        dims: usize,
        transposed: bool,
    ) -> FT {
        if !transposed {
            return f32_tensor(device, values.to_vec(), [rows, dims]);
        }
        let mut columns = vec![0.0f32; rows * dims];
        for r in 0..rows {
            for c in 0..dims {
                columns[c * rows + r] = values[r * dims + c];
            }
        }
        CubeBackend::float_swap_dims(f32_tensor(device, columns, [dims, rows]), 0, 1)
    }

    // Exact duplicates every 13 rows; values include negatives and zero.
    fn duplicated(rows: usize, dims: usize, salt: usize) -> Vec<f32> {
        (0..rows * dims)
            .map(|i| {
                let (r, c) = (i / dims % 13 + salt, i % dims);
                ((r * 31 + c * 7) % 17) as f32 * 0.125 - 1.0
            })
            .collect()
    }

    fn lower_bound(sorted: &[f32], value: f32) -> i32 {
        (sorted.partition_point(|&x| x < value).min(sorted.len() - 1)) as i32
    }

    fn projection_case(device: &CubeDevice, n: usize, dims: usize, transposed: bool) {
        let context = format!("n={n} dims={dims} transposed={transposed}");
        let reference = duplicated(n, dims, 0);
        let raw: Vec<f32> = (0..dims).map(|c| (c as f32 * 0.37 + 0.2).sin()).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let direction: Vec<f32> = raw.iter().map(|x| x / norm).collect();
        let project = |values: &[f32], rows: usize| -> Vec<f32> {
            (0..rows)
                .map(|r| (0..dims).fold(0.0f32, |s, c| s + values[r * dims + c] * direction[c]))
                .collect()
        };
        // Queries: H = N (explicit transform) built from reference copies
        // (exact ties), extreme endpoints and zeros.
        let mut query = reference.clone();
        for c in 0..dims {
            query[c] = 100.0 * direction[c];
            query[(n - 1) * dims + c] = -100.0 * direction[c];
            if n > 2 {
                query[2 * dims + c] = 0.0;
            }
        }
        let reference_tensor = coords(device, &reference, n, dims, transposed);
        let query_tensor = coords(device, &query, n, dims, transposed);
        let direction_tensor = f32_1d(device, direction.clone());
        let rp = f32_1d(device, vec![9.0; n]);
        let qp = f32_1d(device, vec![9.0; n]);
        let untouched = f32_1d(device, vec![9.0; n]);
        modern_project(
            &reference_tensor,
            &query_tensor,
            &direction_tensor,
            &rp,
            &untouched,
            true,
        );
        close(
            &read32(&untouched),
            &vec![9.0; n],
            &format!("{context} shared ignores source"),
        );
        modern_project(
            &reference_tensor,
            &query_tensor,
            &direction_tensor,
            &rp,
            &qp,
            false,
        );
        let reference_projection = read32(&rp);
        let query_projection = read32(&qp);
        close(
            &reference_projection,
            &project(&reference, n),
            &format!("{context} reference"),
        );
        close(
            &query_projection,
            &project(&query, n),
            &format!("{context} query"),
        );
        close(
            &read32(&reference_tensor),
            &reference,
            &format!("{context} immutable reference"),
        );
        // Stable host oracle of device projections (sorting is not under test).
        let mut order: Vec<i32> = (0..n as i32).collect();
        order.sort_by(|&x, &y| {
            reference_projection[x as usize].total_cmp(&reference_projection[y as usize])
        });
        let sorted: Vec<f32> = order
            .iter()
            .map(|&j| reference_projection[j as usize])
            .collect();
        let order_tensor = i32_tensor(device, order.clone());
        let sorted_tensor = f32_1d(device, sorted.clone());
        let ranks = i32_tensor(device, vec![-7; n]);
        modern_projection_ranks(&order_tensor, &sorted_tensor, &qp, &ranks, true);
        let mut inverse = vec![0i32; n];
        for (j, &row) in order.iter().enumerate() {
            inverse[row as usize] = j as i32;
        }
        assert_eq!(i32_read(&ranks), inverse, "{context} fit inverse ranks");
        modern_projection_ranks(&order_tensor, &sorted_tensor, &qp, &ranks, false);
        let expected: Vec<i32> = query_projection
            .iter()
            .map(|&q| lower_bound(&sorted, q))
            .collect();
        assert_eq!(i32_read(&ranks), expected, "{context} transform H=N ranks");
        assert_eq!(
            expected[0],
            (n - 1) as i32,
            "{context} upper endpoint clamps to N-1"
        );
        assert_eq!(expected[n - 1], 0, "{context} lower endpoint");
        // Tail: fewer queries than references, including exact ties.
        let tail_queries = vec![sorted[0], sorted[n - 1], sorted[n / 2], f32::MAX];
        let tail = f32_1d(device, tail_queries.clone());
        let tail_ranks = i32_tensor(device, vec![-7; 4]);
        modern_projection_ranks(&order_tensor, &sorted_tensor, &tail, &tail_ranks, false);
        let expected: Vec<i32> = tail_queries
            .iter()
            .map(|&q| lower_bound(&sorted, q))
            .collect();
        assert_eq!(
            i32_read(&tail_ranks),
            expected,
            "{context} tie lower bounds"
        );

        let status = i32_tensor(device, vec![5]);
        modern_query_projection_status(&qp, &status);
        assert_eq!(i32_read(&status), vec![-1], "{context} finite status");
        let bad = vec![
            0.0,
            1.0,
            -0.0,
            f32::NAN,
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        let bad = f32_1d(device, bad);
        modern_query_projection_status(&bad, &status);
        assert_eq!(
            i32_read(&status),
            vec![3],
            "{context} lowest non-finite row"
        );
    }

    #[derive(Clone, Copy)]
    struct WindowCase {
        h: usize,
        n: usize,
        dims: usize,
        shared: bool,
        range: u32,
        exclude: bool,
        epoch: u32,
        transposed: bool,
    }

    fn is_excluded(
        indices: &[i32],
        row: std::ops::Range<usize>,
        i: usize,
        j: usize,
        shared: bool,
    ) -> bool {
        (shared && i == j) || indices[row].binary_search(&(j as i32)).is_ok()
    }

    fn window_case(device: &CubeDevice, case: &WindowCase) {
        let WindowCase {
            h,
            n,
            dims,
            shared,
            range,
            exclude,
            epoch,
            transposed,
        } = *case;
        let context = format!(
            "h={h} n={n} dims={dims} shared={shared} range={range} exclude={exclude} epoch={epoch}"
        );
        let small = h <= 1024;
        let head_values: Vec<f32> = (0..h * dims)
            .map(|i| ((i / dims % 37) as f32) * 0.11 + ((i % dims) as f32) * 0.013 - 1.5)
            .collect();
        let tail_values: Vec<f32> = if shared {
            head_values.clone()
        } else {
            (0..n * dims)
                .map(|i| ((i / dims % 41) as f32) * 0.09 - ((i % dims) as f32) * 0.011 + 0.3)
                .collect()
        };
        // Sorted unique CSR rows; small layouts get one full row to exhaust
        // retries, large layouts one empty row.
        let mut offsets = vec![0i32];
        let mut indices = Vec::new();
        for i in 0..h {
            let mut row: Vec<i32> = if small && i == 0 {
                (0..n as i32).collect()
            } else if !small && i == 1 {
                Vec::new()
            } else {
                (0..4).map(|t| ((i * 3 + t * 5 + 1) % n) as i32).collect()
            };
            row.sort();
            row.dedup();
            indices.extend(row);
            offsets.push(indices.len() as i32);
        }
        let edges = indices.len();
        // Pseudo-random reference order and supplied ranks.
        let mut order: Vec<i32> = (0..n as i32).collect();
        let mut state = 0x9E37_79B9u64;
        for k in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            order.swap(k, (state >> 33) as usize % (k + 1));
        }
        let ranks: Vec<i32> = if shared {
            let mut inverse = vec![0i32; n];
            for (j, &row) in order.iter().enumerate() {
                inverse[row as usize] = j as i32;
            }
            inverse
        } else {
            (0..h)
                .map(|i| match i {
                    0 => 0,
                    1 => n as i32 - 1,
                    _ => ((i * 7919) % n) as i32,
                })
                .collect()
        };
        let periods = vec![1.0f64; edges];
        let positive = vec![0.0f64; edges];
        let negative_periods = vec![0.5f64; edges];
        let negative = vec![epoch as f64 - 1.5; edges];
        let (a, b, gamma, scale) = (1.3f32, 0.8f32, 1.2f32, 0.4f32);

        let mut gradient = vec![0.0f32; h * dims];
        let n64 = n as u64;
        for i in 0..h {
            let row = offsets[i] as usize..offsets[i + 1] as usize;
            let current = &head_values[i * dims..(i + 1) * dims];
            for e in row.clone() {
                let g = &mut gradient[i * dims..(i + 1) * dims];
                let j = indices[e] as usize;
                accumulate_force(
                    g,
                    current,
                    &tail_values[j * dims..(j + 1) * dims],
                    a,
                    b,
                    gamma,
                    scale,
                    small,
                    false,
                )
                .unwrap();
                let count = ((epoch as f64 - negative[e]) / negative_periods[e]) as i64;
                assert_eq!(count, 3);
                let mut candidates = Vec::new();
                if small && exclude {
                    let base = (e as u64 * (epoch as u64 + 1)) % n64;
                    let (mut accepted, mut attempt) = (0u64, 0u64);
                    while accepted < count as u64 && attempt < count as u64 * n64 {
                        let j = order[((base + attempt % n64) % n64) as usize] as usize;
                        attempt += 1;
                        if !is_excluded(&indices, row.clone(), i, j, shared) {
                            accepted += 1;
                            candidates.push(j);
                        }
                    }
                } else if small {
                    for p in 0..count as u64 {
                        let hash = (e as u128 * (epoch as u128 + p as u128 + 1)) % n as u128;
                        candidates.push(order[hash as usize] as usize);
                    }
                } else {
                    let r = range as i64;
                    let first = (ranks[i] as i64 - r / 2).clamp(0, n as i64 - r) as usize;
                    for p in 0..count as u64 {
                        let sel = (e as u128 * (epoch as u128 + p as u128 + 1)) % range as u128;
                        let j = order[first + sel as usize] as usize;
                        if !(exclude && is_excluded(&indices, row.clone(), i, j, shared)) {
                            candidates.push(j);
                        }
                    }
                }
                for j in candidates {
                    accumulate_force(
                        g,
                        current,
                        &tail_values[j * dims..(j + 1) * dims],
                        a,
                        b,
                        gamma,
                        scale,
                        small,
                        true,
                    )
                    .unwrap();
                }
            }
        }

        let head = coords(device, &head_values, h, dims, transposed);
        let tail = if shared {
            head.clone()
        } else {
            coords(device, &tail_values, n, dims, transposed)
        };
        let source = i32_tensor(device, (0..h as i32).collect());
        let order_tensor = i32_tensor(device, order);
        let rank_tensor = i32_tensor(device, ranks);
        let next = f64_tensor(device, positive);
        let nnext = f64_tensor(device, negative.clone());
        let updates = f32_tensor(device, vec![0.0; h * dims], [h, dims]);
        modern_gradient_block(
            &head,
            &tail,
            &i32_tensor(device, offsets),
            &i32_tensor(device, indices),
            &source,
            &order_tensor,
            &rank_tensor,
            &f64_tensor(device, periods),
            &next,
            &f64_tensor(device, negative_periods),
            &nnext,
            &updates,
            0,
            h as u32,
            epoch,
            a,
            b,
            gamma,
            scale,
            range,
            1.0,
            shared,
            true,
            exclude,
        );
        close(&read32(&updates), &gradient, &context);
        assert_eq!(read64(&next), vec![1.0; edges], "{context} positive clocks");
        let advanced: Vec<f64> = negative.iter().map(|x| x + 1.5).collect();
        assert_eq!(
            read64(&nnext),
            advanced,
            "{context} rejected samples consume clocks"
        );
        close(
            &read32(&head),
            &head_values,
            &format!("{context} immutable head"),
        );
        if !shared {
            close(
                &read32(&tail),
                &tail_values,
                &format!("{context} immutable reference"),
            );
        }
    }

    fn windows(device: &CubeDevice) {
        let mut cases = Vec::new();
        for epoch in [3u32, 4_000_000_000] {
            for exclude in [false, true] {
                let mut add = |h, n, dims, shared, range, transposed| {
                    cases.push(WindowCase {
                        h,
                        n,
                        dims,
                        shared,
                        range,
                        exclude,
                        epoch,
                        transposed,
                    })
                };
                for range in [1, 200, 1024, 1025] {
                    add(1025, 1025, 16, true, range, false);
                }
                add(1025, 1025, 48, true, 200, true);
                for range in [200, 1024, 1025] {
                    add(1025, 1025, 16, false, range, false); // H=N transform
                }
                for range in [200, 299, 300] {
                    add(1025, 300, 16, false, range, false);
                }
                add(1025, 2000, 48, false, 200, true);
                add(8, 8, 16, true, 8, false);
                add(1024, 1024, 16, true, 200, false);
                add(8, 50, 48, false, 50, true);
                add(8, 8, 16, false, 8, false);
            }
        }
        for case in &cases {
            window_case(device, case);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resultant_oracle(
        x: &[f64],
        dims: usize,
        src: usize,
        candidates: &[usize],
        indices: &[i32],
        row: std::ops::Range<usize>,
        (a, b, gamma): (f64, f64, f64),
        exclude: bool,
    ) -> f64 {
        let mut sum = vec![0.0f64; dims];
        for &j in candidates {
            if exclude && is_excluded(indices, row.clone(), src, j, true) {
                continue;
            }
            let delta: Vec<f64> = (0..dims)
                .map(|c| x[src * dims + c] - x[j * dims + c])
                .collect();
            let q: f64 = delta.iter().map(|z| z * z).sum();
            let coefficient = 2.0 * gamma * b / ((0.001 + q) * (a * q.powf(b) + 1.0));
            let norm = coefficient * q.sqrt();
            let clip = if norm > 0.0 {
                gamma * (norm / gamma).tanh() / norm
            } else {
                1.0
            };
            for c in 0..dims {
                sum[c] += coefficient * delta[c] * clip;
            }
        }
        sum.iter().map(|z| z * z).sum::<f64>().sqrt()
    }

    fn resultants(device: &CubeDevice) {
        let (n, dims) = (300usize, 16usize);
        let values = duplicated(n, dims, 3);
        let x: Vec<f64> = values.iter().map(|&v| v as f64).collect();
        let mut offsets = vec![0i32];
        let mut indices = Vec::new();
        for i in 0..n {
            let mut row: Vec<i32> = (0..4).map(|t| ((i * 3 + t * 5 + 1) % n) as i32).collect();
            row.sort();
            row.dedup();
            indices.extend(row);
            offsets.push(indices.len() as i32);
        }
        let order: Vec<i32> = (0..n as i32).map(|j| (j * 7 + 3) % n as i32).collect();
        let samples = vec![0i32, 299, 150, 150, 1, 77];
        let s = samples.len();
        let globals: Vec<i32> = (0..s * 5).map(|k| ((k * 61 + 5) % n) as i32).collect();
        for transposed in [false, true] {
            for raw_range in [1u32, 2, 200, 300] {
                for gamma in [0.0f32, 1.2] {
                    for exclude in [false, true] {
                        let context = format!(
                            "raw={raw_range} gamma={gamma} exclude={exclude} transposed={transposed}"
                        );
                        let locals: Vec<i32> = (0..s * 5)
                            .map(|k| ((k * 13 + 1) % raw_range as usize) as i32)
                            .collect();
                        let mut expected = vec![0.0f64; 2 * s];
                        for t in 0..s {
                            let rank = samples[t] as i64;
                            let src = order[rank as usize] as usize;
                            let row = offsets[src] as usize..offsets[src + 1] as usize;
                            let first =
                                (rank - raw_range as i64 / 2).clamp(0, n as i64 - raw_range as i64);
                            let global: Vec<usize> = (0..5)
                                .map(|k| order[globals[t * 5 + k] as usize] as usize)
                                .collect();
                            let local: Vec<usize> = (0..5)
                                .map(|k| {
                                    order[(first + locals[t * 5 + k] as i64) as usize] as usize
                                })
                                .collect();
                            let p = (1.3_f32 as f64, 0.8_f32 as f64, gamma as f64);
                            expected[t] = resultant_oracle(
                                &x,
                                dims,
                                src,
                                &global,
                                &indices,
                                row.clone(),
                                p,
                                exclude,
                            );
                            expected[s + t] =
                                resultant_oracle(&x, dims, src, &local, &indices, row, p, exclude);
                        }
                        let output = f64_tensor(device, vec![-1.0; 2 * s]);
                        modern_sample_resultants(
                            &coords(device, &values, n, dims, transposed),
                            &i32_tensor(device, order.clone()),
                            &i32_tensor(device, offsets.clone()),
                            &i32_tensor(device, indices.clone()),
                            &i32_tensor(device, samples.clone()),
                            &i32_tensor(device, locals),
                            &i32_tensor(device, globals.clone()),
                            &output,
                            raw_range,
                            1.3,
                            0.8,
                            gamma,
                            exclude,
                        );
                        let actual = read64(&output);
                        for (k, (a, e)) in actual.iter().zip(&expected).enumerate() {
                            assert!(
                                (a - e).abs() <= 1e-9 * (1.0 + e.abs()),
                                "{context} resultant {k}: {a} != {e}"
                            );
                        }
                        if gamma == 0.0 {
                            assert!(actual.iter().all(|&v| v == 0.0), "{context} zero gamma");
                        }
                    }
                }
            }
        }
    }

    fn run_projection(device: CubeDevice) {
        for dims in [16, 48] {
            for transposed in [false, true] {
                for n in [1, 2, 300] {
                    projection_case(&device, n, dims, transposed);
                }
            }
        }
        windows(&device);
        resultants(&device);
    }

    // Characterization only. Example (run CPU and CUDA separately with the
    // corresponding test name):
    // CUSCAN_MODERN_PROFILE_ROWS=20000 CUSCAN_MODERN_PROFILE_SEED=11 cargo test
    //   --release --lib modern_stage_profile_cpu -- --ignored --nocapture --test-threads=1
    // The global comparator uses identity order, not projection-window selection;
    // timings therefore do not establish quality-equivalent speedups.
    fn modern_stage_profile(device: CubeDevice, backend: &str) {
        use burn::cubecl::{MemoryScope, future::block_on};
        use cuscan_common::radix::{RadixDeviceLimits, RadixSortConfig, RadixSortWorkspace};
        use std::time::Instant;

        let n: usize = std::env::var("CUSCAN_MODERN_PROFILE_ROWS")
            .expect("CUSCAN_MODERN_PROFILE_ROWS is required")
            .parse()
            .expect("invalid profile rows");
        assert!(
            [20_000, 200_000].contains(&n),
            "profile rows must be 20000 or 200000"
        );
        let seed: u64 = std::env::var("CUSCAN_MODERN_PROFILE_SEED")
            .unwrap_or_else(|_| "11".into())
            .parse()
            .expect("invalid profile seed");
        assert_eq!(seed, 11, "approved profile seed is 11");
        for dims in [2, 16, 48] {
            for shared in [true, false] {
                let h = if shared { n } else { 300 };
                for momentum in [false, true] {
                    let optimizer = if momentum { "momentum" } else { "adam" };
                    let values = |rows: usize, salt: u64| -> Vec<f32> {
                        (0..rows * dims)
                            .map(|i| {
                                let mut x = (i as u64).wrapping_add(seed).wrapping_add(salt);
                                x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                                x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
                                ((x ^ (x >> 31)) % 20001) as f32 / 10000.0 - 1.0
                            })
                            .collect()
                    };
                    let reference_values = values(n, 123);
                    let head = coords(&device, &values(h, 456), h, dims, true);
                    let tail = coords(&device, &reference_values, n, dims, true);
                    let reference = if shared { &head } else { &tail };
                    let client = &head.client;
                    let sync = || block_on(client.sync()).expect("profile sync failed");
                    let measure = |stage: &str, semantics: &str, f: &mut dyn FnMut(usize)| {
                        let mut times = Vec::with_capacity(2);
                        for repeat in 0..2 {
                            sync();
                            let start = Instant::now();
                            f(repeat);
                            sync();
                            times.push(start.elapsed().as_secs_f64() * 1000.0);
                        }
                        let memory = format!("{:?}", client.memory_report(MemoryScope::Device));
                        println!(
                            "{}",
                            serde_json::json!({
                                "n": n, "h": h, "d": dims, "seed": seed, "backend": backend,
                                "optimizer": optimizer, "stage": stage, "selection": semantics,
                                "cold_ms": times[0], "warm_ms": &times[1..],
                                "duration_scope": "synchronized host wall; first invocation may include JIT; prior cases may have warmed kernels",
                                "memory_sample_after_stage": memory,
                                "memory_scope": "client device memory_report sample, not peak",
                                "limitations": "synthetic fixed CSR15; no ANN or spectral initialization; selection semantics differ; H=300 uses small-layout global sampling and ignores Q; raw_epoch excludes projection, sort and adaptation; characterization only"
                            })
                        );
                    };
                    let offsets = i32_tensor(&device, (0..=h).map(|i| (i * 15) as i32).collect());
                    let indices = i32_tensor(
                        &device,
                        (0..h)
                            .flat_map(|i| {
                                let mut row: Vec<i32> =
                                    (1..=15).map(|j| ((i * 17 + j * 101) % n) as i32).collect();
                                row.sort_unstable();
                                row
                            })
                            .collect(),
                    );
                    let source_order = i32_tensor(&device, (0..h as i32).collect());
                    let global_order = i32_tensor(&device, (0..n as i32).collect());
                    let direction = f32_1d(&device, vec![1.0 / (dims as f32).sqrt(); dims]);
                    let projection = f32_1d(&device, vec![0.0; n]);
                    let query = f32_1d(&device, vec![0.0; h]);
                    let ranks = i32_tensor(&device, vec![0; h]);
                    let config =
                        RadixSortConfig::conservative(RadixDeviceLimits::from_client(client).path);
                    let mut workspace =
                        RadixSortWorkspace::new(client, n, config).expect("radix workspace");
                    measure("projection", "projection_window_q512", &mut |_| {
                        modern_project(reference, &head, &direction, &projection, &query, shared);
                    });
                    let mut sorted = None;
                    measure("stable_sort", "projection_window_q512", &mut |_| {
                        sorted = Some(workspace.sort(&projection.handle, n).expect("stable sort"));
                    });
                    let sorted = sorted.expect("sort output");
                    let order = IT::new_contiguous(
                        client.clone(),
                        head.device.clone(),
                        burn::tensor::Shape::new([n]),
                        sorted.ids,
                        DType::I32,
                    );
                    let sorted_values = FT::new_contiguous(
                        client.clone(),
                        head.device.clone(),
                        burn::tensor::Shape::new([n]),
                        sorted.sorted_values,
                        DType::F32,
                    );
                    measure(
                        if shared { "inverse" } else { "lower_bound" },
                        "projection_window_q512",
                        &mut |_| {
                            modern_projection_ranks(&order, &sorted_values, &query, &ranks, shared);
                        },
                    );
                    if shared && !momentum {
                        let s = n.min(128);
                        let samples = i32_tensor(
                            &device,
                            (0..s)
                                .map(|i| ((i * 131 + seed as usize) % n) as i32)
                                .collect(),
                        );
                        let locals = i32_tensor(
                            &device,
                            (0..s * 5)
                                .map(|i| ((i * 37 + seed as usize) % 512) as i32)
                                .collect(),
                        );
                        let globals = i32_tensor(
                            &device,
                            (0..s * 5)
                                .map(|i| ((i * 7919 + seed as usize) % n) as i32)
                                .collect(),
                        );
                        let resultants = f64_tensor(&device, vec![0.0; s * 2]);
                        measure(
                            "sampled_force_adaptation",
                            "projection_window_raw_q512_five_candidates_s128",
                            &mut |_| {
                                modern_sample_resultants(
                                    &head,
                                    &order,
                                    &offsets,
                                    &indices,
                                    &samples,
                                    &locals,
                                    &globals,
                                    &resultants,
                                    512,
                                    1.3,
                                    0.8,
                                    1.0,
                                    true,
                                );
                            },
                        );
                        assert!(read64(&resultants).iter().all(|x| x.is_finite()));
                    }
                    for (selection, negative_order, range) in [
                        (
                            if shared {
                                "projection_window_q512"
                            } else {
                                "small_layout_global_projection_order_q_ignored"
                            },
                            &order,
                            512u32,
                        ),
                        (
                            if shared {
                                "global_identity_order_r_n_not_quality_equivalent"
                            } else {
                                "small_layout_global_identity_order_q_ignored"
                            },
                            &global_order,
                            n as u32,
                        ),
                    ] {
                        for stage in ["gradient", "raw_epoch"] {
                            let profile_head = coords(&device, &values(h, 456), h, dims, true);
                            let periods = f64_tensor(&device, vec![1.0; h * 15]);
                            let positive = f64_tensor(&device, vec![1.0; h * 15]);
                            let negative_periods = f64_tensor(&device, vec![0.2; h * 15]);
                            let negative = f64_tensor(&device, vec![0.2; h * 15]);
                            let updates = coords(&device, &vec![0.0; h * dims], h, dims, true);
                            let m = coords(&device, &vec![0.0; h * dims], h, dims, true);
                            let v = coords(&device, &vec![0.0; h * dims], h, dims, true);
                            measure(stage, selection, &mut |repeat| {
                                let epoch = repeat as u32 + 1;
                                if stage == "gradient" {
                                    modern_gradient_block(
                                        &profile_head,
                                        &tail,
                                        &offsets,
                                        &indices,
                                        &source_order,
                                        negative_order,
                                        &ranks,
                                        &periods,
                                        &positive,
                                        &negative_periods,
                                        &negative,
                                        &updates,
                                        0,
                                        h as u32,
                                        epoch,
                                        1.3,
                                        0.8,
                                        1.0,
                                        1.0,
                                        range,
                                        1.0,
                                        shared,
                                        true,
                                        true,
                                    );
                                } else {
                                    modern_epoch(
                                        profile_head.clone(),
                                        tail.clone(),
                                        offsets.clone(),
                                        indices.clone(),
                                        source_order.clone(),
                                        negative_order.clone(),
                                        ranks.clone(),
                                        periods.clone(),
                                        positive.clone(),
                                        negative_periods.clone(),
                                        negative.clone(),
                                        updates.clone(),
                                        m.clone(),
                                        v.clone(),
                                        epoch,
                                        1.3,
                                        0.8,
                                        1.0,
                                        1.0,
                                        range,
                                        0.01,
                                        0.9,
                                        0.1,
                                        0.001,
                                        momentum,
                                        shared,
                                        true,
                                        true,
                                    );
                                }
                            });
                            assert!(read32(&profile_head).iter().all(|x| x.is_finite()));
                            assert!(read32(&updates).iter().all(|x| x.is_finite()));
                        }
                    }
                    if !shared {
                        assert_eq!(read32(&tail), reference_values);
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "explicit synthetic stage characterization only"]
    fn modern_stage_profile_cpu() {
        modern_stage_profile(CubeDevice::Cpu(Default::default()), "cpu");
    }

    #[test]
    #[ignore = "explicit synthetic stage characterization only"]
    fn modern_stage_profile_cuda() {
        assert!(
            crate::cuda_backend::device(0).is_some(),
            "CUDA required for CUDA profile"
        );
        modern_stage_profile(CubeDevice::Cuda(Default::default()), "cuda");
    }
    #[test]
    fn modern_projection_windows_cpu() {
        run_projection(CubeDevice::Cpu(Default::default()));
    }
    #[test]
    fn modern_projection_windows_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        run_projection(CubeDevice::Cuda(Default::default()));
    }

    fn run(device: CubeDevice) {
        for shared in [false, true] {
            singleton_empty_row(&device, shared);
        }
        for (rows, dims) in [(1024, 16), (1025, 48)] {
            for shared in [false, true] {
                for adam in [false, true] {
                    for gamma in [0.0, 1.2] {
                        fixture(&device, rows, dims, shared, adam, gamma);
                    }
                }
            }
        }
    }
    #[test]
    fn modern_primitives_cpu() {
        run(CubeDevice::Cpu(Default::default()));
    }
    #[test]
    fn modern_primitives_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        run(CubeDevice::Cuda(Default::default()));
    }
}
