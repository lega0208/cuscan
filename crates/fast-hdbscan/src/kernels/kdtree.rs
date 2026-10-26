//! Set `FAST_HDBSCAN_KD_DIAGNOSTICS=1` to emit counters when built with `tracing`.
use crate::backend::DeviceKdTree;
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

use burn_cubecl::ops::into_data_sync;

const KD_CUBE_DIM: CubeDim = CubeDim::new_1d(32);
const KD_NEIGHBORS_CUBE_DIM: CubeDim = CubeDim::new_1d(64);
const KD_STACK_LEN: usize = 33;
/// Lanes per padded KD vector chunk.
const KD_VECTOR_LANES: usize = 4;
/// Widest feature count whose query is held in registers by `kd_neighbors_kernel`.
const KD_REGISTER_QUERY_MAX_FEATURES: usize = 32;

type KdVector<F> = Vector<F, Const<4>>;

/// Returns `(d_chunks, reg_features)`: the padded chunk count per row and the exact
/// comptime register-query feature count (`0` selects the runtime chunk loop).
fn kd_vector_chunks(features: usize) -> (usize, usize) {
    let chunks = features.div_ceil(KD_VECTOR_LANES);
    let register = if features <= KD_REGISTER_QUERY_MAX_FEATURES {
        features
    } else {
        0
    };
    (chunks, register)
}

/// Scalar width of one zero-padded KD row: `4⌈features/4⌉`.
pub(crate) fn kd_padded_width(features: usize) -> Option<usize> {
    features
        .div_ceil(KD_VECTOR_LANES)
        .checked_mul(KD_VECTOR_LANES)
}

/// Device bytes of the zero-padded KD data and bound copies made per neighbor search.
pub(crate) fn kd_padded_bytes(rows: usize, nodes: usize, features: usize) -> Option<usize> {
    let width = kd_padded_width(features)?;
    nodes
        .checked_mul(2)?
        .checked_add(rows)?
        .checked_mul(width)?
        .checked_mul(DType::F32.size())
}

#[cube(launch)]
fn pad_rows_kernel<F: Float>(
    source: &Tensor<F>,
    padded: &mut Tensor<F>,
    rows: u32,
    d: u32,
    padded_width: u32,
    #[define(F)] _dtype: ElemType,
) {
    let total = rows as usize * padded_width as usize;
    let stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    let mut position = ABSOLUTE_POS_X as usize;
    while position < total {
        let row = position / padded_width as usize;
        let feature = position % padded_width as usize;
        if feature < d as usize {
            padded[position] = source[row * d as usize + feature];
        } else {
            padded[position] = F::new(0.0f32);
        }
        position += stride;
    }
}

#[cube]
fn load_query<F: Float>(
    data_v: &Tensor<KdVector<F>>,
    row: usize,
    d_chunks: usize,
    query: &mut Array<KdVector<F>>,
    #[comptime] reg_features: usize,
) {
    #[unroll]
    for chunk in 0..reg_features.div_ceil(4) {
        query[chunk] = data_v[row * d_chunks + chunk];
    }
}

/// Adds the first `lanes` squared lane differences, in lane order.
#[cube]
fn chunk_distance<F: Float>(
    query: KdVector<F>,
    candidate: KdVector<F>,
    distance: F,
    #[comptime] lanes: usize,
) -> F {
    let mut distance = distance;
    #[unroll]
    for lane in 0..lanes {
        let delta = query.extract(lane) - candidate.extract(lane);
        distance += delta * delta;
    }
    distance
}

/// Adds the first `lanes` squared point-to-box lane gaps, in lane order.
#[cube]
fn chunk_box_bound<F: Float>(
    query: KdVector<F>,
    minimum: KdVector<F>,
    maximum: KdVector<F>,
    sum: F,
    #[comptime] lanes: usize,
) -> F {
    let mut sum = sum;
    #[unroll]
    for lane in 0..lanes {
        let point = query.extract(lane);
        let delta = F::max(
            F::max(minimum.extract(lane) - point, point - maximum.extract(lane)),
            F::new(0.0f32),
        );
        sum += delta * delta;
    }
    sum
}

// The register path sums exactly `reg_features` real lanes; the runtime path also adds the
// tail pad lanes, whose `+0` terms leave every sum bit-unchanged.
#[cube]
fn point_distance<F: Float>(
    data_v: &Tensor<KdVector<F>>,
    query: &Array<KdVector<F>>,
    row: usize,
    candidate: usize,
    d_chunks: usize,
    #[comptime] reg_features: usize,
) -> F {
    let mut distance = F::new(0.0f32);
    if comptime![reg_features > 0] {
        let chunks = comptime![reg_features.div_ceil(4)];
        let mut values = Array::<KdVector<F>>::new(chunks);
        #[unroll]
        for chunk in 0..chunks {
            values[chunk] = data_v[candidate * d_chunks + chunk];
        }
        #[unroll]
        for chunk in 0..chunks {
            distance = chunk_distance::<F>(
                query[chunk],
                values[chunk],
                distance,
                comptime![(reg_features - chunk * 4).min(4)],
            );
        }
    } else {
        for chunk in 0..d_chunks {
            distance = chunk_distance::<F>(
                data_v[row * d_chunks + chunk],
                data_v[candidate * d_chunks + chunk],
                distance,
                4usize,
            );
        }
    }
    distance
}

#[cube]
fn padded_box_lower_bound<F: Float>(
    bounds_min_v: &Tensor<KdVector<F>>,
    bounds_max_v: &Tensor<KdVector<F>>,
    data_v: &Tensor<KdVector<F>>,
    query: &Array<KdVector<F>>,
    row: usize,
    node: usize,
    d_chunks: usize,
    #[comptime] reg_features: usize,
) -> F {
    let mut sum = F::new(0.0f32);
    if comptime![reg_features > 0] {
        let chunks = comptime![reg_features.div_ceil(4)];
        let mut minima = Array::<KdVector<F>>::new(chunks);
        let mut maxima = Array::<KdVector<F>>::new(chunks);
        #[unroll]
        for chunk in 0..chunks {
            minima[chunk] = bounds_min_v[node * d_chunks + chunk];
            maxima[chunk] = bounds_max_v[node * d_chunks + chunk];
        }
        #[unroll]
        for chunk in 0..chunks {
            sum = chunk_box_bound::<F>(
                query[chunk],
                minima[chunk],
                maxima[chunk],
                sum,
                comptime![(reg_features - chunk * 4).min(4)],
            );
        }
    } else {
        for chunk in 0..d_chunks {
            sum = chunk_box_bound::<F>(
                data_v[row * d_chunks + chunk],
                bounds_min_v[node * d_chunks + chunk],
                bounds_max_v[node * d_chunks + chunk],
                sum,
                4usize,
            );
        }
    }
    sum
}

#[cube]
fn box_lower_bound<F: Float>(
    data: &Tensor<F>,
    bounds_min: &Tensor<F>,
    bounds_max: &Tensor<F>,
    row: usize,
    node: usize,
) -> F {
    let d = data.shape(1);
    let nodes = bounds_min.shape(0);
    let mut sum = F::new(0.0f32);
    for feature in 0..d {
        let point = data[row * d + feature];
        let minimum = bounds_min[node * d + feature];
        let maximum = bounds_max[node * d + feature];
        let delta = F::max(F::max(minimum - point, point - maximum), F::new(0.0f32));
        sum += delta * delta;
    }
    let _ = nodes;
    sum
}

#[cfg(test)]
#[cube(launch)]
fn box_lower_bound_test_kernel<F: Float>(
    data: &Tensor<F>,
    bounds_min: &Tensor<F>,
    bounds_max: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let position = ABSOLUTE_POS_X as usize;
    let row = position / bounds_min.shape(0);
    let node = position % bounds_min.shape(0);
    if row < data.shape(0) && node < bounds_min.shape(0) {
        output[row * bounds_min.shape(0) + node] =
            box_lower_bound(data, bounds_min, bounds_max, row, node);
    }
}

#[cube(launch)]
fn reset_component_bounds<I: Int>(
    bounds: &mut Tensor<I>,
    length: u32,
    #[define(I)] _dtype: ElemType,
) {
    let mut position = ABSOLUTE_POS_X as usize;
    let stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while position < length as usize {
        bounds[position] = I::cast_from(0x7f80_0000i32);
        position += stride;
    }
}

#[cube(launch)]
fn kd_candidates_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    core_sq: &Tensor<F>,
    components: &Tensor<I>,
    indices: &Tensor<I>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    is_leaf: &Tensor<I>,
    bounds_min: &Tensor<F>,
    bounds_max: &Tensor<F>,
    node_components: &Tensor<I>,
    component_bounds: &mut Tensor<Atomic<I>>,
    row_count: u32,
    #[comptime] shared_bound: bool,
    #[comptime] diagnostics_enabled: bool,
    out_weights: &mut Tensor<F>,
    out_indices: &mut Tensor<I>,
    diagnostics: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let mut local_row = ABSOLUTE_POS_X as usize;
    let row_stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while local_row < row_count as usize {
        if local_row < data.shape(0) {
            let row = local_row;
            let n = data.shape(0);
            let component = components[row];
            let core = core_sq[row];
            let mut component_bound = F::new(f32::INFINITY);
            if comptime![shared_bound] {
                component_bound =
                    F::reinterpret(component_bounds[u32::cast_from(component) as usize].load());
            }
            let mut best = F::new(f32::INFINITY);
            let mut best_index = n;
            let mut stack = Array::<u32>::new(KD_STACK_LEN);
            let mut bound_stack = Array::<F>::new(KD_STACK_LEN);
            stack[0] = 0;
            bound_stack[0] = F::max(box_lower_bound(data, bounds_min, bounds_max, row, 0), core);
            let mut stack_len = 1usize;
            // Equality must remain active: the first source wins component ties.
            if comptime![shared_bound] && core > component_bound {
                stack_len = 0;
            }
            let mut nodes_popped = 0i32;
            let mut leaf_positions_scanned = 0i32;
            while stack_len > 0 {
                stack_len -= 1;
                if comptime![diagnostics_enabled] {
                    nodes_popped += 1i32;
                }
                let node = stack[stack_len] as usize;
                let lower = bound_stack[stack_len];
                if comptime![shared_bound] {
                    component_bound = F::min(
                        component_bound,
                        F::reinterpret(component_bounds[u32::cast_from(component) as usize].load()),
                    );
                }
                if lower <= best && lower <= component_bound && node_components[node] != component {
                    if is_leaf[node] != I::cast_from(0) {
                        let start = u32::cast_from(starts[node]) as usize;
                        let end = u32::cast_from(ends[node]) as usize;
                        for position in start..end {
                            if comptime![diagnostics_enabled] {
                                leaf_positions_scanned += 1i32;
                            }
                            let candidate = u32::cast_from(indices[position]) as usize;
                            if components[candidate] != component {
                                let d = data.shape(1);
                                let mut distance = F::new(0.0f32);
                                for feature in 0..d {
                                    let delta =
                                        data[row * d + feature] - data[candidate * d + feature];
                                    distance += delta * delta;
                                }
                                let weight = F::max(F::max(distance, core), core_sq[candidate]);
                                if weight < best || (weight == best && candidate < best_index) {
                                    best = weight;
                                    best_index = candidate;
                                    if comptime![shared_bound] && best < component_bound {
                                        let mut bits = I::reinterpret(best);
                                        // Float addition can be erased by fast-math; M0 proves this integer encoding.
                                        if bits == I::cast_from(i32::MIN) {
                                            bits = I::cast_from(0);
                                        }
                                        let previous = F::reinterpret(
                                            component_bounds[u32::cast_from(component) as usize]
                                                .fetch_min(bits),
                                        );
                                        component_bound = F::min(previous, best);
                                    }
                                }
                            }
                        }
                    } else {
                        let left = 2 * node + 1;
                        let right = left + 1;
                        let left_bound = F::max(
                            box_lower_bound(data, bounds_min, bounds_max, row, left),
                            core,
                        );
                        let right_bound = F::max(
                            box_lower_bound(data, bounds_min, bounds_max, row, right),
                            core,
                        );
                        // Push farther first so the nearer child is popped first; left wins equal bounds.
                        if left_bound <= right_bound {
                            if right_bound <= best && right_bound <= component_bound {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                            if left_bound <= best && left_bound <= component_bound {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                        } else {
                            if left_bound <= best && left_bound <= component_bound {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                            if right_bound <= best && right_bound <= component_bound {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                        }
                    }
                }
            }
            out_weights[row] = best;
            out_indices[row] = I::cast_from(best_index);
            if comptime![shared_bound] && best_index == n {
                out_indices[row] = I::cast_from(-1);
            }
            if comptime![diagnostics_enabled] {
                diagnostics[row * 2] = I::cast_from(nodes_popped);
                diagnostics[row * 2 + 1] = I::cast_from(leaf_positions_scanned);
            }
        }
        local_row += row_stride;
    }
}

#[cfg(test)]
#[cube(launch)]
fn kd_neighbors_reference_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    indices: &Tensor<I>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    is_leaf: &Tensor<I>,
    bounds_min: &Tensor<F>,
    bounds_max: &Tensor<F>,
    row_count: u32,
    k: u32,
    #[comptime] diagnostics_enabled: bool,
    out_indices: &mut Tensor<I>,
    out_distances: &mut Tensor<F>,
    diagnostics: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let mut local_row = ABSOLUTE_POS_X as usize;
    let row_stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while local_row < row_count as usize {
        if local_row < data.shape(0) {
            let row = local_row;
            let n = data.shape(0);
            let width = k as usize;
            for slot in 0..width {
                out_indices[row * width + slot] = I::cast_from(n);
                out_distances[row * width + slot] = F::new(f32::INFINITY);
            }
            let mut stack = Array::<u32>::new(KD_STACK_LEN);
            let mut bound_stack = Array::<F>::new(KD_STACK_LEN);
            stack[0] = 0;
            bound_stack[0] = box_lower_bound(data, bounds_min, bounds_max, row, 0);
            let mut stack_len = 1usize;
            let mut nodes_popped = 0i32;
            let mut leaf_positions_scanned = 0i32;
            while stack_len > 0 {
                stack_len -= 1;
                if comptime![diagnostics_enabled] {
                    nodes_popped += 1i32;
                }
                let node = stack[stack_len] as usize;
                let worst = out_distances[row * width];
                let lower = bound_stack[stack_len];
                if lower <= worst {
                    if is_leaf[node] != I::cast_from(0) {
                        let start = u32::cast_from(starts[node]) as usize;
                        let end = u32::cast_from(ends[node]) as usize;
                        for position in start..end {
                            if comptime![diagnostics_enabled] {
                                leaf_positions_scanned += 1i32;
                            }
                            let candidate = u32::cast_from(indices[position]) as usize;
                            if candidate != row {
                                let d = data.shape(1);
                                let mut distance = F::new(0.0f32);
                                for feature in 0..d {
                                    let delta =
                                        data[row * d + feature] - data[candidate * d + feature];
                                    distance += delta * delta;
                                }
                                let root = row * width;
                                let worst_index = u32::cast_from(out_indices[root]) as usize;
                                if distance < out_distances[root]
                                    || (distance == out_distances[root] && candidate < worst_index)
                                {
                                    out_distances[root] = distance;
                                    out_indices[root] = I::cast_from(candidate);
                                    let mut parent = 0usize;
                                    let mut heap = true;
                                    while heap && parent < width / 2 {
                                        let left = parent * 2 + 1;
                                        let right = left + 1;
                                        let mut largest = left;
                                        if right < width {
                                            let left_at = row * width + left;
                                            let right_at = row * width + right;
                                            let left_index =
                                                u32::cast_from(out_indices[left_at]) as usize;
                                            let right_index =
                                                u32::cast_from(out_indices[right_at]) as usize;
                                            if out_distances[right_at] > out_distances[left_at]
                                                || (out_distances[right_at]
                                                    == out_distances[left_at]
                                                    && right_index > left_index)
                                            {
                                                largest = right;
                                            }
                                        }
                                        let parent_at = row * width + parent;
                                        let child_at = row * width + largest;
                                        let parent_index =
                                            u32::cast_from(out_indices[parent_at]) as usize;
                                        let child_index =
                                            u32::cast_from(out_indices[child_at]) as usize;
                                        if out_distances[parent_at] > out_distances[child_at]
                                            || (out_distances[parent_at] == out_distances[child_at]
                                                && parent_index >= child_index)
                                        {
                                            heap = false;
                                        } else {
                                            let parent_distance = out_distances[parent_at];
                                            let parent_index = out_indices[parent_at];
                                            out_distances[parent_at] = out_distances[child_at];
                                            out_indices[parent_at] = out_indices[child_at];
                                            out_distances[child_at] = parent_distance;
                                            out_indices[child_at] = parent_index;
                                            parent = largest;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        let left = 2 * node + 1;
                        let right = left + 1;
                        let left_bound = box_lower_bound(data, bounds_min, bounds_max, row, left);
                        let right_bound = box_lower_bound(data, bounds_min, bounds_max, row, right);
                        if left_bound <= right_bound {
                            if right_bound <= out_distances[row * width] {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                            if left_bound <= out_distances[row * width] {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                        } else {
                            if left_bound <= out_distances[row * width] {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                            if right_bound <= out_distances[row * width] {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                        }
                    }
                }
            }
            if comptime![diagnostics_enabled] {
                diagnostics[row * 2] = I::cast_from(nodes_popped);
                diagnostics[row * 2 + 1] = I::cast_from(leaf_positions_scanned);
            }
            // In-place heapsort turns the output-backed max-heap into ascending order.
            let mut heap_len = width;
            while heap_len > 1 {
                let root = row * width;
                let last = row * width + heap_len - 1;
                let root_distance = out_distances[root];
                let root_index = out_indices[root];
                out_distances[root] = out_distances[last];
                out_indices[root] = out_indices[last];
                out_distances[last] = root_distance;
                out_indices[last] = root_index;
                heap_len -= 1;

                let mut parent = 0usize;
                let mut heap = true;
                while heap && parent < heap_len / 2 {
                    let left = parent * 2 + 1;
                    let right = left + 1;
                    let mut largest = left;
                    if right < heap_len {
                        let left_at = row * width + left;
                        let right_at = row * width + right;
                        let left_index = u32::cast_from(out_indices[left_at]) as usize;
                        let right_index = u32::cast_from(out_indices[right_at]) as usize;
                        if out_distances[right_at] > out_distances[left_at]
                            || (out_distances[right_at] == out_distances[left_at]
                                && right_index > left_index)
                        {
                            largest = right;
                        }
                    }
                    let parent_at = row * width + parent;
                    let child_at = row * width + largest;
                    let parent_index = u32::cast_from(out_indices[parent_at]) as usize;
                    let child_index = u32::cast_from(out_indices[child_at]) as usize;
                    if out_distances[parent_at] > out_distances[child_at]
                        || (out_distances[parent_at] == out_distances[child_at]
                            && parent_index >= child_index)
                    {
                        heap = false;
                    } else {
                        let parent_distance = out_distances[parent_at];
                        let parent_index = out_indices[parent_at];
                        out_distances[parent_at] = out_distances[child_at];
                        out_indices[parent_at] = out_indices[child_at];
                        out_distances[child_at] = parent_distance;
                        out_indices[child_at] = parent_index;
                        parent = largest;
                    }
                }
            }
        }
        local_row += row_stride;
    }
}

#[cube(launch)]
fn kd_neighbors_kernel<F: Float, I: Int>(
    data_v: &Tensor<KdVector<F>>,
    indices: &Tensor<I>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    is_leaf: &Tensor<I>,
    bounds_min_v: &Tensor<KdVector<F>>,
    bounds_max_v: &Tensor<KdVector<F>>,
    row_count: u32,
    n: u32,
    k: u32,
    d_chunks: u32,
    #[comptime] reg_features: usize,
    #[comptime] diagnostics_enabled: bool,
    out_indices: &mut Tensor<I>,
    out_distances: &mut Tensor<F>,
    diagnostics: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let mut local_row = ABSOLUTE_POS_X as usize;
    let row_stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while local_row < row_count as usize {
        if local_row < n as usize {
            let row = local_row;
            let n = n as usize;
            let d_chunks = d_chunks as usize;
            let width = k as usize;
            for slot in 0..width {
                out_indices[row * width + slot] = I::cast_from(n);
                out_distances[row * width + slot] = F::new(f32::INFINITY);
            }
            let root = row * width;
            let mut worst_distance = F::new(f32::INFINITY);
            let mut worst_index = n;
            let mut stack = Array::<u32>::new(KD_STACK_LEN);
            let mut bound_stack = Array::<F>::new(KD_STACK_LEN);
            stack[0] = 0;
            let mut query = Array::<KdVector<F>>::new(comptime![reg_features.div_ceil(4).max(1)]);
            load_query::<F>(data_v, row, d_chunks, &mut query, reg_features);
            bound_stack[0] = padded_box_lower_bound::<F>(
                bounds_min_v,
                bounds_max_v,
                data_v,
                &query,
                row,
                0,
                d_chunks,
                reg_features,
            );
            let mut stack_len = 1usize;
            let mut nodes_popped = 0i32;
            let mut leaf_positions_scanned = 0i32;
            while stack_len > 0 {
                stack_len -= 1;
                if comptime![diagnostics_enabled] {
                    nodes_popped += 1i32;
                }
                let node = stack[stack_len] as usize;
                let lower = bound_stack[stack_len];
                if lower <= worst_distance {
                    if is_leaf[node] != I::cast_from(0) {
                        let start = u32::cast_from(starts[node]) as usize;
                        let end = u32::cast_from(ends[node]) as usize;
                        for position in start..end {
                            if comptime![diagnostics_enabled] {
                                leaf_positions_scanned += 1i32;
                            }
                            let candidate = u32::cast_from(indices[position]) as usize;
                            if candidate != row {
                                let distance = point_distance::<F>(
                                    data_v,
                                    &query,
                                    row,
                                    candidate,
                                    d_chunks,
                                    reg_features,
                                );
                                if distance < worst_distance
                                    || (distance == worst_distance && candidate < worst_index)
                                {
                                    out_distances[root] = distance;
                                    out_indices[root] = I::cast_from(candidate);
                                    let mut parent = 0usize;
                                    let mut heap = true;
                                    while heap && parent < width / 2 {
                                        let left = parent * 2 + 1;
                                        let right = left + 1;
                                        let mut largest = left;
                                        if right < width {
                                            let left_at = row * width + left;
                                            let right_at = row * width + right;
                                            let left_index =
                                                u32::cast_from(out_indices[left_at]) as usize;
                                            let right_index =
                                                u32::cast_from(out_indices[right_at]) as usize;
                                            if out_distances[right_at] > out_distances[left_at]
                                                || (out_distances[right_at]
                                                    == out_distances[left_at]
                                                    && right_index > left_index)
                                            {
                                                largest = right;
                                            }
                                        }
                                        let parent_at = row * width + parent;
                                        let child_at = row * width + largest;
                                        let parent_index =
                                            u32::cast_from(out_indices[parent_at]) as usize;
                                        let child_index =
                                            u32::cast_from(out_indices[child_at]) as usize;
                                        if out_distances[parent_at] > out_distances[child_at]
                                            || (out_distances[parent_at] == out_distances[child_at]
                                                && parent_index >= child_index)
                                        {
                                            heap = false;
                                        } else {
                                            let parent_distance = out_distances[parent_at];
                                            let parent_index = out_indices[parent_at];
                                            out_distances[parent_at] = out_distances[child_at];
                                            out_indices[parent_at] = out_indices[child_at];
                                            out_distances[child_at] = parent_distance;
                                            out_indices[child_at] = parent_index;
                                            parent = largest;
                                        }
                                    }
                                    worst_distance = out_distances[root];
                                    worst_index = u32::cast_from(out_indices[root]) as usize;
                                }
                            }
                        }
                    } else {
                        let left = 2 * node + 1;
                        let right = left + 1;
                        let left_bound = padded_box_lower_bound::<F>(
                            bounds_min_v,
                            bounds_max_v,
                            data_v,
                            &query,
                            row,
                            left,
                            d_chunks,
                            reg_features,
                        );
                        let right_bound = padded_box_lower_bound::<F>(
                            bounds_min_v,
                            bounds_max_v,
                            data_v,
                            &query,
                            row,
                            right,
                            d_chunks,
                            reg_features,
                        );
                        if left_bound <= right_bound {
                            if right_bound <= worst_distance {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                            if left_bound <= worst_distance {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                        } else {
                            if left_bound <= worst_distance {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                            if right_bound <= worst_distance {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                        }
                    }
                }
            }
            if comptime![diagnostics_enabled] {
                diagnostics[row * 2] = I::cast_from(nodes_popped);
                diagnostics[row * 2 + 1] = I::cast_from(leaf_positions_scanned);
            }
            // In-place heapsort turns the output-backed max-heap into ascending order.
            let mut heap_len = width;
            while heap_len > 1 {
                let root = row * width;
                let last = row * width + heap_len - 1;
                let root_distance = out_distances[root];
                let root_index = out_indices[root];
                out_distances[root] = out_distances[last];
                out_indices[root] = out_indices[last];
                out_distances[last] = root_distance;
                out_indices[last] = root_index;
                heap_len -= 1;

                let mut parent = 0usize;
                let mut heap = true;
                while heap && parent < heap_len / 2 {
                    let left = parent * 2 + 1;
                    let right = left + 1;
                    let mut largest = left;
                    if right < heap_len {
                        let left_at = row * width + left;
                        let right_at = row * width + right;
                        let left_index = u32::cast_from(out_indices[left_at]) as usize;
                        let right_index = u32::cast_from(out_indices[right_at]) as usize;
                        if out_distances[right_at] > out_distances[left_at]
                            || (out_distances[right_at] == out_distances[left_at]
                                && right_index > left_index)
                        {
                            largest = right;
                        }
                    }
                    let parent_at = row * width + parent;
                    let child_at = row * width + largest;
                    let parent_index = u32::cast_from(out_indices[parent_at]) as usize;
                    let child_index = u32::cast_from(out_indices[child_at]) as usize;
                    if out_distances[parent_at] > out_distances[child_at]
                        || (out_distances[parent_at] == out_distances[child_at]
                            && parent_index >= child_index)
                    {
                        heap = false;
                    } else {
                        let parent_distance = out_distances[parent_at];
                        let parent_index = out_indices[parent_at];
                        out_distances[parent_at] = out_distances[child_at];
                        out_indices[parent_at] = out_indices[child_at];
                        out_distances[child_at] = parent_distance;
                        out_indices[child_at] = parent_index;
                        parent = largest;
                    }
                }
            }
        }
        local_row += row_stride;
    }
}

fn launch_count(rows: usize, dim: u32, max_cubes: u32) -> u32 {
    assert!(dim > 0, "KD cube width must be positive");
    assert!(max_cubes > 0, "device reports no legal KD launch capacity");
    u32::try_from(rows.div_ceil(dim as usize).min(max_cubes as usize))
        .expect("bounded KD launch count")
}

#[cfg(test)]
fn distinct_legal_widths(
    client: &cubecl::client::Client,
    requested: impl IntoIterator<Item = u32>,
) -> Vec<u32> {
    let mut seen = Vec::new();
    requested
        .into_iter()
        .filter(|width| {
            let dim = fast_cubecl_common::legal_cube_dim(client, CubeDim::new_1d(*width));
            if seen.contains(&(dim.x, dim.y, dim.z)) {
                false
            } else {
                seen.push((dim.x, dim.y, dim.z));
                true
            }
        })
        .collect()
}

fn kd_launch_geometry(client: &cubecl::client::Client, rows: usize) -> (CubeCount, CubeDim) {
    kd_launch_geometry_with_width(client, rows, KD_CUBE_DIM, None)
}

fn kd_launch_geometry_with_width(
    client: &cubecl::client::Client,
    rows: usize,
    preferred_dim: CubeDim,
    max_cubes_override: Option<u32>,
) -> (CubeCount, CubeDim) {
    let hardware = &client.properties().hardware;
    let (dim, max_cubes) = if let Some(cores) = hardware.num_cpu_cores {
        (
            fast_cubecl_common::legal_cube_dim(client, CubeDim::new_1d(cores.max(1))),
            1,
        )
    } else {
        (
            fast_cubecl_common::legal_cube_dim(client, preferred_dim),
            max_cubes_override.map_or(hardware.max_cube_count.0, |cap| {
                cap.min(hardware.max_cube_count.0)
            }),
        )
    };
    assert!(max_cubes > 0, "device reports no legal KD launch capacity");
    assert!(dim.x > 0, "device reports no legal KD cube width");
    (
        CubeCount::Static(launch_count(rows, dim.x, max_cubes), 1, 1),
        dim,
    )
}

#[cfg(test)]
fn kd_neighbors_launch_geometry(
    client: &cubecl::client::Client,
    rows: usize,
    width: u32,
    max_cubes_override: Option<u32>,
) -> (CubeCount, CubeDim) {
    assert!(width > 0, "preferred KD cube width must be positive");
    kd_launch_geometry_with_width(client, rows, CubeDim::new_1d(width), max_cubes_override)
}

fn validate_tree(data: &FloatTensor<CubeBackend>, tree: &DeviceKdTree<CubeBackend>) {
    let [rows, features] = data.shape().dims::<2>();
    let [index_count] = tree.indices.shape().dims::<1>();
    let [nodes] = tree.starts.shape().dims::<1>();
    assert!(rows > 0 && features > 0, "KD input data must be non-empty");
    i32::try_from(rows).expect("KD row count must fit destination indices");
    i32::try_from(nodes).expect("KD node count must fit diagnostic counters");
    assert_eq!(data.dtype, DType::F32, "KD data must be f32");
    assert_eq!(
        tree.indices.dtype,
        DType::I32,
        "KD point indices must be i32"
    );
    assert_eq!(tree.starts.dtype, DType::I32, "KD starts must be i32");
    assert_eq!(tree.ends.dtype, DType::I32, "KD ends must be i32");
    assert_eq!(tree.is_leaf.dtype, DType::I32, "KD leaf flags must be i32");
    assert_eq!(
        tree.node_components.dtype,
        DType::I32,
        "KD node components must be i32"
    );
    assert_eq!(
        tree.bounds_min.dtype,
        DType::F32,
        "KD minimum bounds must be f32"
    );
    assert_eq!(
        tree.bounds_max.dtype,
        DType::F32,
        "KD maximum bounds must be f32"
    );
    assert_eq!(features, tree.n_features, "KD feature count mismatch");
    assert_eq!(tree.ends.shape().dims::<1>(), [nodes]);
    assert_eq!(tree.is_leaf.shape().dims::<1>(), [nodes]);
    assert_eq!(tree.node_components.shape().dims::<1>(), [nodes]);
    assert_eq!(tree.bounds_min.shape().dims::<2>(), [nodes, features]);
    assert_eq!(tree.bounds_max.shape().dims::<2>(), [nodes, features]);
    assert_eq!(index_count, rows, "KD point permutation size mismatch");
    assert!(
        tree.n_levels < KD_STACK_LEN,
        "KD tree exceeds fixed traversal stack"
    );
}

fn into_contiguous_tree(tree: DeviceKdTree<CubeBackend>) -> DeviceKdTree<CubeBackend> {
    DeviceKdTree {
        indices: into_contiguous(tree.indices),
        starts: into_contiguous(tree.starts),
        ends: into_contiguous(tree.ends),
        is_leaf: into_contiguous(tree.is_leaf),
        bounds_min: into_contiguous(tree.bounds_min),
        bounds_max: into_contiguous(tree.bounds_max),
        node_components: into_contiguous(tree.node_components),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    }
}

pub(super) fn boruvka_candidates_kd(
    data: FloatTensor<CubeBackend>,
    core_sq: FloatTensor<CubeBackend>,
    point_components: IntTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    reproducible: bool,
    scratch: &mut super::KdBoundScratch,
) -> (FloatTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let data = into_contiguous(data);
    let core_sq = into_contiguous(core_sq);
    let point_components = into_contiguous(point_components);
    let tree = into_contiguous_tree(tree);
    validate_tree(&data, &tree);
    let [n, _] = data.shape().dims::<2>();
    assert_eq!(core_sq.shape().dims::<1>(), [n]);
    assert_eq!(point_components.shape().dims::<1>(), [n]);
    assert_eq!(core_sq.dtype, DType::F32, "KD core distances must be f32");
    assert_eq!(
        point_components.dtype,
        DType::I32,
        "KD component labels must be i32"
    );
    let diagnostics_enabled = kd_diagnostics_enabled();
    let weights = allocate_float(&data, n);
    let destinations = allocate_int(&data, n);
    let diagnostics = allocate_diagnostics(&data, n, diagnostics_enabled);
    let client = data.client.clone();
    let (cube_count, dim) = kd_launch_geometry(&client, n);
    let bound_len = if reproducible { 1 } else { n };
    assert_eq!(scratch.bounds.shape().dims::<1>(), [bound_len]);
    assert_eq!(scratch.bounds.dtype, DType::I32);
    // Scratch is private and allocated per call. CubeCL's asynchronous allocation
    // initialization holds transient handle clones, so can_mut cannot prove that ownership.
    // Sole reset, on this launch's client/stream. Never seed from kNN distances.
    reset_component_bounds::launch(
        &client,
        cube_count.clone(),
        dim,
        scratch.bounds.clone().into_tensor_arg(),
        u32::try_from(bound_len).expect("validated KD bound length"),
        dtype_to_storage_type(DType::I32),
    );
    kd_candidates_kernel::launch(
        &client,
        cube_count,
        dim,
        data.clone().into_tensor_arg(),
        core_sq.clone().into_tensor_arg(),
        point_components.clone().into_tensor_arg(),
        tree.indices.clone().into_tensor_arg(),
        tree.starts.clone().into_tensor_arg(),
        tree.ends.clone().into_tensor_arg(),
        tree.is_leaf.clone().into_tensor_arg(),
        tree.bounds_min.clone().into_tensor_arg(),
        tree.bounds_max.clone().into_tensor_arg(),
        tree.node_components.clone().into_tensor_arg(),
        scratch.bounds.clone().into_tensor_arg(),
        u32::try_from(n).expect("validated KD row count"),
        !reproducible,
        diagnostics_enabled,
        weights.clone().into_tensor_arg(),
        destinations.clone().into_tensor_arg(),
        diagnostics.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    if diagnostics_enabled {
        emit_kd_diagnostics(
            "kd_candidates",
            diagnostics,
            n,
            tree.starts.shape().dims::<1>()[0],
        );
    }
    (weights, destinations)
}

pub(super) fn exact_neighbors_kd(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let diagnostics_enabled = kd_diagnostics_enabled();
    let node_count = tree.starts.shape().dims::<1>()[0];
    let (output_indices, output_distances, diagnostics) =
        launch_kd_neighbors(data, tree, k, diagnostics_enabled);
    if diagnostics_enabled {
        let n = output_indices.shape().dims::<2>()[0];
        emit_kd_diagnostics("kd_neighbors", diagnostics, n, node_count);
    }
    (output_indices, output_distances)
}

fn launch_kd_neighbors(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    #[cfg(test)]
    {
        launch_kd_neighbors_impl(
            data,
            tree,
            k,
            diagnostics_enabled,
            KD_NEIGHBORS_CUBE_DIM.x,
            None,
        )
    }
    #[cfg(not(test))]
    {
        launch_kd_neighbors_impl(data, tree, k, diagnostics_enabled)
    }
}

#[cfg(test)]
fn launch_kd_neighbors_with_width(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
    preferred_width: u32,
    max_cubes_override: Option<u32>,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    launch_kd_neighbors_impl(
        data,
        tree,
        k,
        diagnostics_enabled,
        preferred_width,
        max_cubes_override,
    )
}

fn launch_kd_neighbors_impl(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
    #[cfg(test)] preferred_width: u32,
    #[cfg(test)] max_cubes_override: Option<u32>,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    let data = into_contiguous(data);
    let tree = into_contiguous_tree(tree);
    validate_tree(&data, &tree);
    let [n, features] = data.shape().dims::<2>();
    assert!(k > 0, "KD neighbor count must be positive");
    let output_len = n
        .checked_mul(k as usize)
        .expect("KD neighbor output size overflow");
    let output_indices = allocate_int_2d(&data, n, k as usize, output_len);
    let output_distances = allocate_float_2d(&data, n, k as usize, output_len);
    let diagnostics = allocate_diagnostics(&data, n, diagnostics_enabled);
    let client = data.client.clone();
    let [nodes] = tree.starts.shape().dims::<1>();
    let (d_chunks, reg_features) = kd_vector_chunks(features);
    let data_padded = pad_rows(&data, n, features);
    let bounds_min_padded = pad_rows(&tree.bounds_min, nodes, features);
    let bounds_max_padded = pad_rows(&tree.bounds_max, nodes, features);
    #[cfg(test)]
    let (cube_count, dim) =
        kd_neighbors_launch_geometry(&client, n, preferred_width, max_cubes_override);
    #[cfg(not(test))]
    let (cube_count, dim) = kd_launch_geometry_with_width(&client, n, KD_NEIGHBORS_CUBE_DIM, None);
    kd_neighbors_kernel::launch(
        &client,
        cube_count,
        dim,
        padded_vector_view(&data_padded, n, d_chunks),
        tree.indices.clone().into_tensor_arg(),
        tree.starts.clone().into_tensor_arg(),
        tree.ends.clone().into_tensor_arg(),
        tree.is_leaf.clone().into_tensor_arg(),
        padded_vector_view(&bounds_min_padded, nodes, d_chunks),
        padded_vector_view(&bounds_max_padded, nodes, d_chunks),
        u32::try_from(n).expect("validated KD row count"),
        u32::try_from(n).expect("validated KD row count"),
        k,
        u32::try_from(d_chunks).expect("KD feature chunk count must fit u32"),
        reg_features,
        diagnostics_enabled,
        output_indices.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        diagnostics.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    (output_indices, output_distances, diagnostics)
}

#[cfg(test)]
fn exact_neighbors_kd_reference(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    let data = into_contiguous(data);
    let tree = into_contiguous_tree(tree);
    validate_tree(&data, &tree);
    let [n, _] = data.shape().dims::<2>();
    assert!(k > 0, "KD neighbor count must be positive");
    let output_len = n
        .checked_mul(k as usize)
        .expect("KD neighbor output size overflow");
    let output_indices = allocate_int_2d(&data, n, k as usize, output_len);
    let output_distances = allocate_float_2d(&data, n, k as usize, output_len);
    let diagnostics = allocate_diagnostics(&data, n, diagnostics_enabled);
    let client = data.client.clone();
    let (cube_count, dim) = kd_launch_geometry(&client, n);
    kd_neighbors_reference_kernel::launch(
        &client,
        cube_count,
        dim,
        data.clone().into_tensor_arg(),
        tree.indices.clone().into_tensor_arg(),
        tree.starts.clone().into_tensor_arg(),
        tree.ends.clone().into_tensor_arg(),
        tree.is_leaf.clone().into_tensor_arg(),
        tree.bounds_min.clone().into_tensor_arg(),
        tree.bounds_max.clone().into_tensor_arg(),
        u32::try_from(n).expect("validated KD row count"),
        k,
        diagnostics_enabled,
        output_indices.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        diagnostics.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    (output_indices, output_distances, diagnostics)
}

/// Copies a contiguous `[rows, features]` f32 tensor into a 1D buffer of
/// `rows * 4⌈features/4⌉` values whose tail lanes hold `+0.0`.
fn pad_rows(
    source: &FloatTensor<CubeBackend>,
    rows: usize,
    features: usize,
) -> FloatTensor<CubeBackend> {
    assert_eq!(source.shape().dims::<2>(), [rows, features]);
    let width = kd_padded_width(features).expect("KD padded width overflow");
    let len = rows
        .checked_mul(width)
        .expect("KD padded element count overflow");
    u32::try_from(len).expect("KD padded element count must fit u32");
    let padded = allocate_float(source, len);
    let client = source.client.clone();
    let (cube_count, dim) = kd_launch_geometry(&client, len);
    pad_rows_kernel::launch(
        &client,
        cube_count,
        dim,
        source.clone().into_tensor_arg(),
        padded.clone().into_tensor_arg(),
        u32::try_from(rows).expect("KD padded row count must fit u32"),
        u32::try_from(features).expect("KD feature count must fit u32"),
        u32::try_from(width).expect("KD padded width must fit u32"),
        dtype_to_storage_type(DType::F32),
    );
    padded
}

/// Views a `pad_rows` buffer as `rows * d_chunks` four-lane vectors.
fn padded_vector_view(tensor: &CubeTensor, rows: usize, d_chunks: usize) -> TensorArg {
    let vectors = rows
        .checked_mul(d_chunks)
        .expect("KD padded vector count overflow");
    assert_eq!(
        tensor.shape().dims::<1>(),
        [vectors * KD_VECTOR_LANES],
        "KD padded buffer length mismatch"
    );
    // SAFETY: `pad_rows` returns a fresh contiguous f32 allocation with exactly
    // vectors*4 scalar slots; the view uses unit vector stride, and the kernel
    // bounds every access with scalar `n`/`d_chunks` arguments, never view shapes.
    unsafe { TensorArg::from_raw_parts(tensor.handle.clone(), [1].into(), [vectors].into()) }
}

fn kd_diagnostics_enabled() -> bool {
    cfg!(feature = "tracing")
        && std::env::var_os("FAST_HDBSCAN_KD_DIAGNOSTICS")
            .and_then(|value| value.into_string().ok())
            .is_some_and(|value| value == "1")
}

fn allocate_diagnostics(
    data: &FloatTensor<CubeBackend>,
    rows: usize,
    enabled: bool,
) -> IntTensor<CubeBackend> {
    let elements = if enabled {
        rows.checked_mul(2)
            .expect("KD diagnostic element count overflow")
    } else {
        1
    };
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([elements]),
        data.client.empty(
            elements
                .checked_mul(DType::I32.size())
                .expect("KD diagnostic allocation overflow"),
        ),
        DType::I32,
    )
}

fn emit_kd_diagnostics(
    operation: &str,
    tensor: IntTensor<CubeBackend>,
    rows: usize,
    node_count: usize,
) {
    let counts = into_data_sync(tensor)
        .try_to_vec::<i32>()
        .expect("KD diagnostic output must be a contiguous i32 buffer");
    assert_eq!(
        counts.len(),
        rows.checked_mul(2).expect("KD diagnostic size overflow")
    );
    let mut nodes_popped = Vec::with_capacity(rows);
    let mut leaf_positions_scanned = Vec::with_capacity(rows);
    for row in 0..rows {
        let popped = u32::try_from(counts[row * 2]).expect("KD node pop count must fit u32");
        let scanned = u32::try_from(counts[row * 2 + 1]).expect("KD leaf scan count must fit u32");
        assert!(
            popped as usize <= node_count,
            "KD node pop count exceeds tree size"
        );
        assert!(
            scanned as usize <= rows,
            "KD leaf scan count exceeds point count"
        );
        nodes_popped.push(popped);
        leaf_positions_scanned.push(scanned);
    }
    let popped = summarize_counts(&nodes_popped);
    let scanned = summarize_counts(&leaf_positions_scanned);
    eprintln!(
        "{{\"operation\":\"{operation}\",\"rows\":{rows},\"nodes_popped\":{{\"total\":{},\"min\":{},\"max\":{},\"per_row\":{:?}}},\"leaf_positions_scanned\":{{\"total\":{},\"min\":{},\"max\":{},\"per_row\":{:?}}}}}",
        popped.total,
        popped.min,
        popped.max,
        nodes_popped,
        scanned.total,
        scanned.min,
        scanned.max,
        leaf_positions_scanned,
    );
}

#[derive(Debug, PartialEq, Eq)]
struct CountSummary {
    total: u64,
    min: u32,
    max: u32,
}

fn summarize_counts(counts: &[u32]) -> CountSummary {
    assert!(
        !counts.is_empty(),
        "KD diagnostics require at least one row"
    );
    let total = counts
        .iter()
        .try_fold(0u64, |total, &count| total.checked_add(u64::from(count)))
        .expect("KD diagnostic total overflow");
    CountSummary {
        total,
        min: *counts.iter().min().expect("non-empty counts"),
        max: *counts.iter().max().expect("non-empty counts"),
    }
}

fn allocate_float(data: &FloatTensor<CubeBackend>, rows: usize) -> FloatTensor<CubeBackend> {
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows]),
        data.client.empty(
            rows.checked_mul(DType::F32.size())
                .expect("KD output allocation overflow"),
        ),
        DType::F32,
    )
}

pub(super) fn allocate_int(data: &FloatTensor<CubeBackend>, rows: usize) -> IntTensor<CubeBackend> {
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows]),
        data.client.empty(
            rows.checked_mul(DType::I32.size())
                .expect("KD output allocation overflow"),
        ),
        DType::I32,
    )
}

fn allocate_float_2d(
    data: &FloatTensor<CubeBackend>,
    rows: usize,
    cols: usize,
    len: usize,
) -> FloatTensor<CubeBackend> {
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows, cols]),
        data.client.empty(
            len.checked_mul(DType::F32.size())
                .expect("KD output allocation overflow"),
        ),
        DType::F32,
    )
}

fn allocate_int_2d(
    data: &FloatTensor<CubeBackend>,
    rows: usize,
    cols: usize,
    len: usize,
) -> IntTensor<CubeBackend> {
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows, cols]),
        data.client.empty(
            len.checked_mul(DType::I32.size())
                .expect("KD output allocation overflow"),
        ),
        DType::I32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ops::{FloatTensorOps, IntTensorOps};
    use burn::tensor::TensorData;
    use burn_cubecl::{CubeDevice, ops::into_data_sync};

    fn splitmix_values(rows: usize, features: usize) -> Vec<f32> {
        let mut state = 0x5eed_cafe_d15c_a11u64;
        (0..rows * features)
            .map(|index| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut value = state;
                value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                value ^= value >> 31;
                let unit = (value >> 40) as f32 / (1u32 << 24) as f32;
                let cluster = ((index / features) % 7) as f32 - 3.0;
                (unit * 8.0 - 4.0) + cluster * 0.125
            })
            .collect()
    }

    fn make_neighbor_inputs(
        values: &[f32],
        rows: usize,
        features: usize,
        device: &CubeDevice,
    ) -> (FloatTensor<CubeBackend>, DeviceKdTree<CubeBackend>) {
        let host = fast_hdbscan_core::kdtree::KdTree::build(values, rows, features).unwrap();
        let data = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(values.to_vec(), [rows, features]),
            device,
        ));
        let tree = DeviceKdTree {
            indices: into_contiguous(CubeBackend::int_from_data(
                TensorData::new(host.indices.clone(), [rows]),
                device,
            )),
            starts: into_contiguous(CubeBackend::int_from_data(
                TensorData::new(host.starts.clone(), [host.starts.len()]),
                device,
            )),
            ends: into_contiguous(CubeBackend::int_from_data(
                TensorData::new(host.ends.clone(), [host.ends.len()]),
                device,
            )),
            is_leaf: into_contiguous(CubeBackend::int_from_data(
                TensorData::new(host.is_leaf.clone(), [host.is_leaf.len()]),
                device,
            )),
            bounds_min: into_contiguous(CubeBackend::float_from_data(
                TensorData::new(host.bounds_min.clone(), [host.starts.len(), features]),
                device,
            )),
            bounds_max: into_contiguous(CubeBackend::float_from_data(
                TensorData::new(host.bounds_max.clone(), [host.starts.len(), features]),
                device,
            )),
            node_components: into_contiguous(CubeBackend::int_from_data(
                TensorData::new(vec![-1; host.starts.len()], [host.starts.len()]),
                device,
            )),
            n_levels: host.n_levels,
            n_features: features,
        };
        (data, tree)
    }

    fn check_kd_neighbors_match_reference(device: &CubeDevice) {
        let mut cases = Vec::new();
        for features in [2, 16, 30] {
            for k in 1..=17 {
                cases.push((2000, features, k, None));
            }
            for k in [24, 30] {
                cases.push((2000, features, k, None));
            }
        }
        for features in [1, 3, 4, 5, 8, 12, 20, 24, 28, 29, 32] {
            for k in [1, 5, 16, 17] {
                cases.push((2000, features, k, None));
            }
        }
        // Runtime chunk-loop path (`reg_features == 0`), including a padded tail.
        for features in [33, 40] {
            cases.push((2000, features, 5, None));
        }
        for k in [1, 5, 16] {
            cases.push((2000, 16, k, Some("duplicates")));
        }
        for features in [3, 16] {
            for k in [5, 17] {
                cases.push((2000, features, k, Some("duplicates")));
            }
        }
        for k in [5, 16] {
            cases.push((512, 3, k, Some("grid")));
        }
        cases.push((512, 2, 5, Some("grid")));
        for k in [1, 5, 16] {
            cases.push((3, 2, k, Some("overflow")));
        }
        for features in [1, 2, 3, 5] {
            for k in [1, 5] {
                cases.push((9, features, k, Some("overflow")));
            }
        }
        cases.extend([
            (1, 2, 1, None),
            (4, 2, 3, None),
            (4, 2, 5, None),
            (4, 2, 16, None),
            (9, 2, 16, None),
            (257, 16, 5, None),
        ]);

        for (rows, features, k, special) in cases {
            let mut values = splitmix_values(rows, features);
            match special {
                Some("duplicates") => {
                    let points = splitmix_values(50, features);
                    for row in 0..rows {
                        values[row * features..(row + 1) * features].copy_from_slice(
                            &points[(row % 50) * features..(row % 50 + 1) * features],
                        );
                    }
                }
                Some("grid") => {
                    for row in 0..rows {
                        for feature in 0..features {
                            values[row * features + feature] = ((row >> (3 * feature)) % 8) as f32;
                        }
                    }
                }
                Some("overflow") if rows == 3 && features == 2 => {
                    values = vec![f32::MAX, 0.0, -f32::MAX, 0.0, 0.0, f32::MAX];
                }
                Some("overflow") => {
                    // Values near f32::MAX give +inf partial sums before the pad terms.
                    for (index, value) in values.iter_mut().enumerate() {
                        *value =
                            [f32::MAX, 0.0, -f32::MAX, 0.5 * f32::MAX][(index * 7 + index / 3) % 4];
                    }
                }
                _ => {}
            }
            let (reference_data, reference_tree) =
                make_neighbor_inputs(&values, rows, features, device);
            let (reference_ids, reference_distances, reference_diagnostics) =
                exact_neighbors_kd_reference(reference_data, reference_tree, k, true);
            let ref_ids = into_data_sync(reference_ids).try_to_vec::<i32>().unwrap();
            let ref_distances = into_data_sync(reference_distances)
                .try_to_vec::<f32>()
                .unwrap();
            let ref_diagnostics = into_data_sync(reference_diagnostics)
                .try_to_vec::<i32>()
                .unwrap();
            let (probe, _) = make_neighbor_inputs(&values, rows, features, device);
            let client = probe.client.clone();
            let widths = if matches!(device, CubeDevice::Cuda(_)) {
                let hardware = &client.properties().hardware;
                eprintln!(
                    "KD geometry hardware: max_cube_dim={:?}, max_units_per_cube={}, plane_size_min={}, max_cube_count={:?}",
                    hardware.max_cube_dim,
                    hardware.max_units_per_cube,
                    hardware.plane_size_min,
                    hardware.max_cube_count
                );
                for width in [32, 64, 128] {
                    let dim = fast_cubecl_common::legal_cube_dim(&client, CubeDim::new_1d(width));
                    eprintln!(
                        "KD width requested={width}, actual=({},{},{}), max_cube_count={:?}",
                        dim.x, dim.y, dim.z, hardware.max_cube_count
                    );
                }
                distinct_legal_widths(&client, [32, 64, 128])
            } else {
                vec![32]
            };
            for width in widths {
                let (data, tree) = make_neighbor_inputs(&values, rows, features, device);
                let (production_ids, production_distances, production_diagnostics) =
                    launch_kd_neighbors_with_width(data, tree, k, true, width, None);
                assert_eq!(
                    into_data_sync(production_ids.clone())
                        .try_to_vec::<i32>()
                        .unwrap(),
                    ref_ids,
                    "neighbor IDs, n={rows}, d={features}, k={k}, {special:?}"
                );
                assert_eq!(
                    into_data_sync(production_distances.clone())
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    ref_distances
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    "neighbor distance bits, n={rows}, d={features}, k={k}, {special:?}"
                );
                assert_eq!(
                    into_data_sync(production_diagnostics)
                        .try_to_vec::<i32>()
                        .unwrap(),
                    ref_diagnostics,
                    "diagnostics, n={rows}, d={features}, k={k}, {special:?}"
                );
                let (disabled_data, disabled_tree) =
                    make_neighbor_inputs(&values, rows, features, device);
                let (disabled_ids, disabled_distances, _) = launch_kd_neighbors_with_width(
                    disabled_data,
                    disabled_tree,
                    k,
                    false,
                    width,
                    None,
                );
                assert_eq!(
                    into_data_sync(disabled_ids).try_to_vec::<i32>().unwrap(),
                    into_data_sync(production_ids).try_to_vec::<i32>().unwrap(),
                    "disabled diagnostics IDs, n={rows}, d={features}, k={k}"
                );
                assert_eq!(
                    into_data_sync(disabled_distances)
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    into_data_sync(production_distances)
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn kd_neighbors_match_frozen_reference_cpu() {
        check_kd_neighbors_match_reference(&CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn kd_neighbor_geometry_count_boundaries_and_cpu_independence() {
        for width in [32u32, 64, 128] {
            for rows in [
                1usize,
                width as usize - 1,
                width as usize,
                width as usize + 1,
            ] {
                let count = launch_count(rows, width, u32::MAX);
                assert_eq!(count, rows.div_ceil(width as usize) as u32);
            }
            assert_eq!(launch_count(width as usize * 9 + 1, width, 3), 3);
        }
        assert_eq!(launch_count(usize::MAX, 1, u32::MAX), u32::MAX);
        assert!(std::panic::catch_unwind(|| launch_count(1, 1, 0)).is_err());
        assert!(std::panic::catch_unwind(|| launch_count(1, 0, 1)).is_err());
        let (data, _) = make_neighbor_inputs(&[0.0], 1, 1, &CubeDevice::Cpu(Default::default()));
        let hw = &data.client.properties().hardware;
        for width in [32, 64, 128] {
            let legal = fast_cubecl_common::legal_cube_dim(&data.client, CubeDim::new_1d(width));
            assert!(legal.x > 0 && legal.y > 0 && legal.z > 0);
            assert!(legal.x <= hw.max_cube_dim.0 && legal.y <= hw.max_cube_dim.1);
            assert!(legal.z <= hw.max_cube_dim.2);
            assert!(
                u64::from(legal.x) * u64::from(legal.y) * u64::from(legal.z)
                    <= u64::from(hw.max_units_per_cube)
            );
            let (count, dim) = kd_neighbors_launch_geometry(&data.client, 129, width, Some(1));
            let (base_count, base_dim) = kd_launch_geometry(&data.client, 129);
            assert_eq!((dim.x, dim.y, dim.z), (base_dim.x, base_dim.y, base_dim.z));
            let (count_x, count_y, count_z) = match count {
                CubeCount::Static(x, y, z) => (x, y, z),
                _ => panic!("KD launch geometry must be static"),
            };
            let (base_x, base_y, base_z) = match base_count {
                CubeCount::Static(x, y, z) => (x, y, z),
                _ => panic!("KD launch geometry must be static"),
            };
            assert_eq!((count_x, count_y, count_z), (base_x, base_y, base_z));
        }
        let candidate_dims = [32, 64, 128]
            .into_iter()
            .map(|width| {
                let dim = fast_cubecl_common::legal_cube_dim(&data.client, CubeDim::new_1d(width));
                (dim.x, dim.y, dim.z)
            })
            .collect::<std::collections::HashSet<_>>();
        let distinct_widths = distinct_legal_widths(&data.client, [32, 64, 128]);
        assert_eq!(distinct_widths.len(), candidate_dims.len());
        assert!(distinct_widths.windows(2).all(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn kd_neighbors_capped_grid_stride_matches_reference_cuda() {
        let count =
            fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if count == 0 {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let device = CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 });
        let (probe, _) = make_neighbor_inputs(&[0.0], 1, 1, &device);
        assert_eq!(KD_NEIGHBORS_CUBE_DIM.x, 64);
        assert_eq!(KD_CUBE_DIM.x, 32);
        for preferred in [KD_NEIGHBORS_CUBE_DIM, KD_CUBE_DIM] {
            let (count, actual) =
                kd_launch_geometry_with_width(&probe.client, 129, preferred, None);
            let expected = fast_cubecl_common::legal_cube_dim(&probe.client, preferred);
            assert_eq!(
                (actual.x, actual.y, actual.z),
                (expected.x, expected.y, expected.z)
            );
            assert!(
                matches!(count, CubeCount::Static(x, 1, 1) if x == 129u32.div_ceil(expected.x))
            );
        }
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                kd_neighbors_launch_geometry(&probe.client, 1, 32, Some(0))
            }))
            .is_err()
        );
        for requested_width in [32, 64, 128] {
            let actual_width =
                fast_cubecl_common::legal_cube_dim(&probe.client, CubeDim::new_1d(requested_width))
                    .x as usize;
            let cap_capacity = 2 * actual_width;
            let mut row_counts = vec![
                1,
                actual_width - 1,
                actual_width,
                actual_width + 1,
                cap_capacity - 1,
                cap_capacity,
                cap_capacity + 1,
                5 * actual_width + 3,
            ];
            row_counts.sort_unstable();
            row_counts.dedup();
            for rows in row_counts {
                let values = splitmix_values(rows, 3);
                let (reference_data, reference_tree) =
                    make_neighbor_inputs(&values, rows, 3, &device);
                let (reference_ids, reference_distances, reference_diagnostics) =
                    exact_neighbors_kd_reference(reference_data, reference_tree, 5, true);
                let ref_ids = into_data_sync(reference_ids).try_to_vec::<i32>().unwrap();
                let ref_distance_bits = into_data_sync(reference_distances)
                    .try_to_vec::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>();
                let ref_diagnostics = into_data_sync(reference_diagnostics)
                    .try_to_vec::<i32>()
                    .unwrap();
                let (data, tree) = make_neighbor_inputs(&values, rows, 3, &device);
                let (ids, distances, diagnostics) =
                    launch_kd_neighbors_with_width(data, tree, 5, true, requested_width, Some(2));
                assert_eq!(
                    into_data_sync(ids).try_to_vec::<i32>().unwrap(),
                    ref_ids,
                    "capped grid-stride IDs at requested width {requested_width}, actual {actual_width}, rows {rows}"
                );
                assert_eq!(
                    into_data_sync(distances)
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                    ref_distance_bits,
                    "capped grid-stride distances at requested width {requested_width}, actual {actual_width}, rows {rows}"
                );
                assert_eq!(
                    into_data_sync(diagnostics).try_to_vec::<i32>().unwrap(),
                    ref_diagnostics,
                    "capped grid-stride diagnostics at requested width {requested_width}, actual {actual_width}, rows {rows}"
                );
            }
        }
    }

    #[test]
    fn kd_neighbors_match_frozen_reference_cuda() {
        let count =
            fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if count == 0 {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        check_kd_neighbors_match_reference(&CubeDevice::Cuda(cubecl::device::CudaDevice {
            index: 0,
        }));
    }

    #[test]
    fn kd_vector_chunks_dispatches_register_and_runtime_paths() {
        for features in 1..=KD_REGISTER_QUERY_MAX_FEATURES {
            let chunks = features.div_ceil(4);
            assert_eq!(
                kd_vector_chunks(features),
                (chunks, features),
                "d={features}"
            );
        }
        assert_eq!(kd_vector_chunks(33), (9, 0));
        assert_eq!(kd_vector_chunks(40), (10, 0));
        assert_eq!(kd_vector_chunks(41), (11, 0));
    }

    #[test]
    fn kd_padded_bytes_counts_data_and_both_bounds() {
        assert_eq!(kd_padded_bytes(100_000, 0, 30), Some(12_800_000));
        assert_eq!(kd_padded_bytes(10, 3, 5), Some((10 + 6) * 8 * 4));
        assert_eq!(kd_padded_bytes(usize::MAX, 1, 1), None);
    }

    fn check_pad_rows(device: &CubeDevice) {
        for features in [1usize, 4, 5, 30, 33] {
            let rows = 37usize;
            // Includes -0.0 so a pad written as a copy of real data would be caught.
            let mut values = splitmix_values(rows, features);
            values[0] = -0.0;
            let source = into_contiguous(CubeBackend::float_from_data(
                TensorData::new(values.clone(), [rows, features]),
                device,
            ));
            let padded = into_data_sync(pad_rows(&source, rows, features))
                .try_to_vec::<f32>()
                .unwrap();
            let width = features.div_ceil(4) * 4;
            assert_eq!(padded.len(), rows * width, "d={features}");
            for row in 0..rows {
                for lane in 0..width {
                    let bits = padded[row * width + lane].to_bits();
                    if lane < features {
                        assert_eq!(
                            bits,
                            values[row * features + lane].to_bits(),
                            "d={features}"
                        );
                    } else {
                        assert_eq!(bits, 0x0000_0000, "pad lane d={features} row={row}");
                    }
                }
            }
            // Bounds are packed by the same helper from the device tree buffers.
            let host = fast_hdbscan_core::kdtree::KdTree::build(&values, rows, features).unwrap();
            let nodes = host.starts.len();
            for bounds in [&host.bounds_min, &host.bounds_max] {
                let tensor = into_contiguous(CubeBackend::float_from_data(
                    TensorData::new(bounds.clone(), [nodes, features]),
                    device,
                ));
                let packed = into_data_sync(pad_rows(&tensor, nodes, features))
                    .try_to_vec::<f32>()
                    .unwrap();
                for node in 0..nodes {
                    for lane in 0..width {
                        let expected = if lane < features {
                            bounds[node * features + lane].to_bits()
                        } else {
                            0
                        };
                        assert_eq!(packed[node * width + lane].to_bits(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn pad_rows_copies_bits_and_zeroes_tail_cpu() {
        check_pad_rows(&CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn pad_rows_copies_bits_and_zeroes_tail_cuda() {
        let count =
            fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if count == 0 {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        check_pad_rows(&CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }));
    }

    #[test]
    fn kd_diagnostic_distribution_reports_totals_and_bounds() {
        assert_eq!(
            summarize_counts(&[0, 3, 7, 3]),
            CountSummary {
                total: 13,
                min: 0,
                max: 7,
            }
        );
        assert_eq!(
            summarize_counts(&[i32::MAX as u32, 0]),
            CountSummary {
                total: i32::MAX as u64,
                min: 0,
                max: i32::MAX as u32,
            }
        );
    }

    #[test]
    fn kd_neighbors_enabled_diagnostics_match_disabled_launch() {
        let device = CubeDevice::Cpu(Default::default());
        let rows = 9usize;
        let features = 2usize;
        let values = (0..rows)
            .flat_map(|row| [row as f32, (row % 3) as f32])
            .collect::<Vec<_>>();
        let host_tree = fast_hdbscan_core::kdtree::KdTree::build(&values, rows, features).unwrap();
        let input = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(values, [rows, features]),
            &device,
        ));
        let tree = |values: Vec<i32>, shape: [usize; 1]| {
            into_contiguous(CubeBackend::int_from_data(
                TensorData::new(values, shape),
                &device,
            ))
        };
        let indices = tree(host_tree.indices.clone(), [rows]);
        let starts = tree(host_tree.starts.clone(), [host_tree.starts.len()]);
        let ends = tree(host_tree.ends.clone(), [host_tree.ends.len()]);
        let leaves = tree(host_tree.is_leaf.clone(), [host_tree.is_leaf.len()]);
        let bounds_min = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(
                host_tree.bounds_min.clone(),
                [host_tree.starts.len(), features],
            ),
            &device,
        ));
        let bounds_max = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(
                host_tree.bounds_max.clone(),
                [host_tree.starts.len(), features],
            ),
            &device,
        ));
        let k = 3usize;
        let make_output = || {
            CubeTensor::new_contiguous(
                input.client.clone(),
                input.device.clone(),
                Shape::new([rows, k]),
                input.client.empty(rows * k * DType::I32.size()),
                DType::I32,
            )
        };
        let make_distances = || {
            CubeTensor::new_contiguous(
                input.client.clone(),
                input.device.clone(),
                Shape::new([rows, k]),
                input.client.empty(rows * k * DType::F32.size()),
                DType::F32,
            )
        };
        let enabled_indices = make_output();
        let enabled_distances = make_distances();
        let enabled_counts = allocate_diagnostics(&input, rows, true);
        let client = input.client.clone();
        let (cube_count, dim) = kd_launch_geometry(&client, rows);
        let nodes = host_tree.starts.len();
        let (d_chunks, reg_features) = kd_vector_chunks(features);
        let input_padded = pad_rows(&input, rows, features);
        let bounds_min_padded = pad_rows(&bounds_min, nodes, features);
        let bounds_max_padded = pad_rows(&bounds_max, nodes, features);
        let launch = |diagnostics_enabled,
                      output_indices: &CubeTensor,
                      output_distances: &CubeTensor,
                      diagnostics: &CubeTensor| {
            kd_neighbors_kernel::launch(
                &client,
                cube_count.clone(),
                dim,
                padded_vector_view(&input_padded, rows, d_chunks),
                indices.clone().into_tensor_arg(),
                starts.clone().into_tensor_arg(),
                ends.clone().into_tensor_arg(),
                leaves.clone().into_tensor_arg(),
                padded_vector_view(&bounds_min_padded, nodes, d_chunks),
                padded_vector_view(&bounds_max_padded, nodes, d_chunks),
                rows as u32,
                rows as u32,
                k as u32,
                d_chunks as u32,
                reg_features,
                diagnostics_enabled,
                output_indices.clone().into_tensor_arg(),
                output_distances.clone().into_tensor_arg(),
                diagnostics.clone().into_tensor_arg(),
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
        };
        launch(true, &enabled_indices, &enabled_distances, &enabled_counts);
        let disabled_indices = make_output();
        let disabled_distances = make_distances();
        let disabled_counts = allocate_diagnostics(&input, rows, false);
        launch(
            false,
            &disabled_indices,
            &disabled_distances,
            &disabled_counts,
        );

        assert_eq!(
            into_data_sync(enabled_indices).try_to_vec::<i32>().unwrap(),
            into_data_sync(disabled_indices)
                .try_to_vec::<i32>()
                .unwrap()
        );
        assert_eq!(
            into_data_sync(enabled_distances)
                .try_to_vec::<f32>()
                .unwrap(),
            into_data_sync(disabled_distances)
                .try_to_vec::<f32>()
                .unwrap()
        );
        let counts = into_data_sync(enabled_counts).try_to_vec::<i32>().unwrap();
        assert_eq!(counts.len(), rows * 2);
        for pair in counts.as_chunks::<2>().0 {
            assert!(pair[0] > 0 && pair[0] as usize <= host_tree.starts.len());
            assert!(pair[1] >= 0 && pair[1] as usize <= rows);
        }
    }

    #[test]
    fn validate_tree_rejects_levels_and_malformed_shapes() {
        let device = CubeDevice::Cpu(Default::default());
        let rows = 4usize;
        let features = 2usize;
        let host = fast_hdbscan_core::kdtree::KdTree::build(
            &[0., 0., 1., 0., 0., 1., 1., 1.],
            rows,
            features,
        )
        .unwrap();
        let data = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(vec![0.0f32, 0., 1., 0., 0., 1., 1., 1.], [rows, features]),
            &device,
        ));
        let make_tree = || DeviceKdTree {
            indices: CubeBackend::int_from_data(
                TensorData::new(host.indices.clone(), [rows]),
                &device,
            ),
            starts: CubeBackend::int_from_data(
                TensorData::new(host.starts.clone(), [host.starts.len()]),
                &device,
            ),
            ends: CubeBackend::int_from_data(
                TensorData::new(host.ends.clone(), [host.ends.len()]),
                &device,
            ),
            is_leaf: CubeBackend::int_from_data(
                TensorData::new(host.is_leaf.clone(), [host.is_leaf.len()]),
                &device,
            ),
            bounds_min: CubeBackend::float_from_data(
                TensorData::new(host.bounds_min.clone(), [host.starts.len(), features]),
                &device,
            ),
            bounds_max: CubeBackend::float_from_data(
                TensorData::new(host.bounds_max.clone(), [host.starts.len(), features]),
                &device,
            ),
            node_components: CubeBackend::int_from_data(
                TensorData::new(vec![-1; host.starts.len()], [host.starts.len()]),
                &device,
            ),
            n_levels: 32,
            n_features: features,
        };
        let base = make_tree();
        validate_tree(&data, &base);
        for invalid_levels in [33, usize::MAX] {
            let mut malformed = make_tree();
            malformed.n_levels = invalid_levels;
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| validate_tree(
                    &data, &malformed
                )))
                .is_err()
            );
        }
        let mut bad_bounds = make_tree();
        bad_bounds.bounds_min = CubeBackend::float_from_data(
            TensorData::new(
                vec![0.0f32; host.starts.len() * (features - 1)],
                [host.starts.len(), features - 1],
            ),
            &device,
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| validate_tree(
                &data,
                &bad_bounds
            )))
            .is_err()
        );
        let mut bad_indices = make_tree();
        bad_indices.indices =
            CubeBackend::int_from_data(TensorData::new(vec![0; rows - 1], [rows - 1]), &device);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| validate_tree(
                &data,
                &bad_indices
            )))
            .is_err()
        );
    }

    fn verify_box_bounds(device: &CubeDevice) {
        let rows = 81usize;
        let features = 3usize;
        let tree = fast_hdbscan_core::kdtree::KdTree::build(
            &(0..rows)
                .flat_map(|row| {
                    [
                        row as f32 * 0.25,
                        (row % 7) as f32 - 2.0,
                        (row % 11) as f32 * 0.5,
                    ]
                })
                .collect::<Vec<_>>(),
            rows,
            features,
        )
        .unwrap();
        let values = (0..rows)
            .flat_map(|row| {
                [
                    row as f32 * 0.25,
                    (row % 7) as f32 - 2.0,
                    (row % 11) as f32 * 0.5,
                ]
            })
            .collect::<Vec<_>>();
        // Match production traversal: raw CubeBackend uploads can have padded
        // last-axis strides, while the kernel indexes compact row-major buffers.
        let input = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(values.clone(), [rows, features]),
            device,
        ));
        let bounds_min = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(tree.bounds_min.clone(), [tree.starts.len(), features]),
            device,
        ));
        let bounds_max = into_contiguous(CubeBackend::float_from_data(
            TensorData::new(tree.bounds_max.clone(), [tree.starts.len(), features]),
            device,
        ));
        let nodes = tree.starts.len();
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            Shape::new([rows, nodes]),
            input.client.empty(
                rows.checked_mul(nodes)
                    .unwrap()
                    .checked_mul(DType::F32.size())
                    .unwrap(),
            ),
            DType::F32,
        );
        let client = input.client.clone();
        let dim = fast_cubecl_common::legal_cube_dim(&client, KD_CUBE_DIM);
        box_lower_bound_test_kernel::launch(
            &client,
            CubeCount::Static(u32::try_from(rows * nodes).unwrap().div_ceil(dim.x), 1, 1),
            dim,
            input.into_tensor_arg(),
            bounds_min.into_tensor_arg(),
            bounds_max.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            dtype_to_storage_type(DType::F32),
        );
        let actual = into_data_sync(output).try_to_vec::<f32>().unwrap();
        for row in 0..rows {
            for node in 0..nodes {
                let mut expected = 0.0f32;
                for feature in 0..features {
                    let value = values[row * features + feature];
                    let min = tree.bounds_min[node * features + feature];
                    let max = tree.bounds_max[node * features + feature];
                    let delta = if value < min {
                        min - value
                    } else if value > max {
                        value - max
                    } else {
                        0.0
                    };
                    expected += delta * delta;
                }
                let actual = actual[row * nodes + node];
                assert_eq!(actual, expected, "query row {row}, KD node {node}");
            }
        }
    }

    #[test]
    fn box_lower_bound_matches_point_box_oracle_cpu() {
        verify_box_bounds(&CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn box_lower_bound_matches_point_box_oracle_cuda() {
        let count =
            fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if count == 0 {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        }
        let cuda = CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 });
        verify_box_bounds(&cuda);
    }
}
