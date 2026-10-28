//! Set `FAST_HDBSCAN_KD_DIAGNOSTICS=1` to emit counters when built with `tracing`.
use crate::backend::{DeviceKdReference, DeviceKdTree};
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
use cubecl::features::Plane;

const KD_CUBE_DIM: CubeDim = CubeDim::new_1d(32);
const KD_NEIGHBORS_CUBE_DIM: CubeDim = CubeDim::new_1d(64);
const KD_STACK_LEN: usize = 33;
/// Lanes per padded KD vector chunk.
const KD_VECTOR_LANES: usize = 4;
/// Widest feature count whose query is held in registers by `kd_neighbors_kernel`.
const KD_REGISTER_QUERY_MAX_FEATURES: usize = 32;

type KdVector<F> = Vector<F, Const<4>>;

const KD_QUERY_BLOCK: usize = 64;
const KD_SOA_MAX_LANES: usize = 16;
const KD_TILE_PLANES: usize = 2;
const KD_TILE_MIN_FEATURES: usize = 16;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KdCandidateLayout {
    PaddedRows,
    LeafRows { tile: usize },
    LeafSoa { lanes: usize },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KdQueryOrder {
    Cyclic,
    PlaneTiles,
    LeafBlocks,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KdNeighborPlan {
    pub(crate) layout: KdCandidateLayout,
    pub(crate) order: KdQueryOrder,
}
pub(crate) fn kd_neighbor_plan(client: &cubecl::client::Client, features: usize) -> KdNeighborPlan {
    if let Some(plan) = kd_tile_plan(client, features) {
        return plan;
    }
    if client.properties().hardware.num_cpu_cores.is_none() {
        return KdNeighborPlan {
            layout: KdCandidateLayout::PaddedRows,
            order: KdQueryOrder::Cyclic,
        };
    }
    let lanes = client
        .io_optimized_vector_sizes(4)
        .find(|&w| w <= KD_SOA_MAX_LANES)
        .expect("CPU KD vector width");
    KdNeighborPlan {
        layout: KdCandidateLayout::LeafSoa { lanes },
        order: KdQueryOrder::LeafBlocks,
    }
}
fn kd_soa_slots(rows: usize, lanes: usize) -> Option<usize> {
    if lanes == 0 {
        return None;
    }
    (rows / lanes)
        .checked_add(usize::from(!rows.is_multiple_of(lanes)))?
        .checked_mul(lanes)
}
fn kd_tile_plan(client: &cubecl::client::Client, features: usize) -> Option<KdNeighborPlan> {
    kd_tile_plan_with_gate(client, features, KD_TILE_MIN_FEATURES)
}
fn kd_tile_plan_with_gate(
    client: &cubecl::client::Client,
    features: usize,
    gate: usize,
) -> Option<KdNeighborPlan> {
    let hw = &client.properties().hardware;
    let tile = hw.plane_size_min as usize;
    if hw.num_cpu_cores.is_some()
        || !client.features().plane.contains(Plane::Ops)
        || !client.features().plane.contains(Plane::Sync)
        || tile == 0
        || hw.plane_size_min != hw.plane_size_max
        || features < gate
        || features > KD_REGISTER_QUERY_MAX_FEATURES
    {
        return None;
    }
    if !kd_tile_resources_fit(
        features,
        tile,
        hw.max_shared_memory_size,
        hw.max_units_per_cube as usize,
        hw.max_cube_dim.0 as usize,
    ) {
        return None;
    }
    Some(KdNeighborPlan {
        layout: KdCandidateLayout::LeafRows { tile },
        order: KdQueryOrder::PlaneTiles,
    })
}
#[cfg(test)]
fn kd_tile_plan_for_test(
    client: &cubecl::client::Client,
    features: usize,
) -> Option<KdNeighborPlan> {
    kd_tile_plan_with_gate(client, features, 1)
}
fn kd_tile_shared_bytes(features: usize, tile: usize) -> Option<usize> {
    if tile == 0 {
        return None;
    }
    tile.checked_mul(KD_TILE_PLANES)?
        .checked_mul(kd_padded_width(features)?.checked_mul(4)?.checked_add(4)?)
}
fn kd_tile_resources_fit(
    features: usize,
    tile: usize,
    shared_limit: usize,
    unit_limit: usize,
    dim_limit: usize,
) -> bool {
    let Some(units) = tile.checked_mul(KD_TILE_PLANES) else {
        return false;
    };
    let Some(shared) = kd_tile_shared_bytes(features, tile) else {
        return false;
    };
    shared <= shared_limit && units <= unit_limit && units <= dim_limit
}

/// Scalar elements in the data pack and diagnostics-only staging counter allocation.
/// Bounds and CPU SoA allocations are checked separately by the estimator.
pub(crate) fn kd_neighbor_pack_allocations(
    plan: KdNeighborPlan,
    rows: usize,
    features: usize,
) -> Option<[usize; 2]> {
    let (packed_rows, staged) = match plan.layout {
        KdCandidateLayout::LeafRows { tile } => (
            kd_soa_slots(rows, tile)?,
            kd_tile_diagnostic_bytes(rows, tile, kd_diagnostics_enabled())? / 4,
        ),
        _ => (rows, 0),
    };
    Some([packed_rows.checked_mul(kd_padded_width(features)?)?, staged])
}
/// Per-call packs for a valid builder-produced tree; no content validation is performed.
pub(crate) fn kd_neighbor_pack_bytes(
    plan: KdNeighborPlan,
    rows: usize,
    nodes: usize,
    features: usize,
) -> Option<usize> {
    let padded = kd_padded_bytes(rows, nodes, features)?;
    match plan.layout {
        KdCandidateLayout::LeafRows { tile } => kd_soa_slots(rows, tile)?
            .checked_add(nodes.checked_mul(2)?)?
            .checked_mul(kd_padded_width(features)?)?
            .checked_mul(4)?
            .checked_add(kd_tile_diagnostic_bytes(
                rows,
                tile,
                kd_diagnostics_enabled(),
            )?),
        KdCandidateLayout::PaddedRows => Some(padded),
        KdCandidateLayout::LeafSoa { lanes } => padded.checked_add(
            kd_soa_slots(rows, lanes)?
                .checked_mul(features.checked_add(1)?)?
                .checked_mul(4)?,
        ),
    }
}
fn kd_tile_diagnostic_bytes(rows: usize, tile: usize, enabled: bool) -> Option<usize> {
    let slots = kd_soa_slots(rows, tile)?;
    if enabled {
        (slots / tile).checked_mul(4)
    } else {
        Some(0)
    }
}
#[cube(launch)]
fn pack_leaf_soa_kernel<F: Float, I: Int>(
    data_v: &Tensor<KdVector<F>>,
    indices: &Tensor<I>,
    soa: &mut Tensor<F>,
    slot_ids: &mut Tensor<I>,
    n: u32,
    d: u32,
    d_chunks: u32,
    lanes: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let mut e = ABSOLUTE_POS_X as usize;
    let stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while e < soa.len() {
        let tile = e / (d as usize * lanes as usize);
        let feature = (e / lanes as usize) % d as usize;
        let position = tile * lanes as usize + e % lanes as usize;
        if position < n as usize {
            let candidate = u32::cast_from(indices[position]) as usize;
            let chunk = data_v[candidate * d_chunks as usize + feature / 4];
            let lane = feature % 4;
            if lane == 0 {
                soa[e] = chunk.extract(0usize);
            } else if lane == 1 {
                soa[e] = chunk.extract(1usize);
            } else if lane == 2 {
                soa[e] = chunk.extract(2usize);
            } else {
                soa[e] = chunk.extract(3usize);
            }
            if feature == 0 {
                slot_ids[position] = indices[position];
            }
        } else {
            soa[e] = F::new(0.0_f32);
            if feature == 0 {
                slot_ids[position] = I::cast_from(n);
            }
        }
        e += stride;
    }
}

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
    query_v: &Tensor<KdVector<F>>,
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
                query_v[row * d_chunks + chunk],
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
    query_v: &Tensor<KdVector<F>>,
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
                query_v[row * d_chunks + chunk],
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
    query_v: &Tensor<KdVector<F>>,
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
    query_offset: u32,
    #[comptime] reg_features: usize,
    #[comptime] diagnostics_enabled: bool,
    #[comptime] exclude_self: bool,
    out_indices: &mut Tensor<I>,
    out_distances: &mut Tensor<F>,
    diagnostics: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    // `row_count` query rows of `query_v` search `n` reference rows of `data_v`.
    // The excluded self ID is `query_offset + row`; value equality is never identity.
    let mut local_row = ABSOLUTE_POS_X as usize;
    let row_stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while local_row < row_count as usize {
        {
            let row = local_row;
            let self_id = row + query_offset as usize;
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
            load_query::<F>(query_v, row, d_chunks, &mut query, reg_features);
            bound_stack[0] = padded_box_lower_bound::<F>(
                bounds_min_v,
                bounds_max_v,
                query_v,
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
                            if !exclude_self || candidate != self_id {
                                let distance = point_distance::<F>(
                                    data_v,
                                    query_v,
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
                            query_v,
                            &query,
                            row,
                            left,
                            d_chunks,
                            reg_features,
                        );
                        let right_bound = padded_box_lower_bound::<F>(
                            bounds_min_v,
                            bounds_max_v,
                            query_v,
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

#[cube(launch)]
fn pack_leaf_rows_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    indices: &Tensor<I>,
    output: &mut Tensor<F>,
    n: u32,
    d: u32,
    padded_width: u32,
    len: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let mut at = ABSOLUTE_POS_X;
    let stride = CUBE_DIM_X * CUBE_COUNT_X;
    while at < len {
        let position = at / padded_width;
        let feature = at % padded_width;
        let mut value = F::new(0.0f32);
        if position < n && feature < d {
            let row = u32::cast_from(indices[position as usize]);
            value = data[(row * d + feature) as usize];
        }
        output[at as usize] = value;
        at += stride;
    }
}

#[cube(launch)]
fn kd_neighbors_tile_kernel<F: Float, I: Int>(
    data_v: &Tensor<KdVector<F>>,
    indices: &Tensor<I>,
    n: u32,
    k: u32,
    #[comptime] reg_features: usize,
    #[comptime] tile_width: usize,
    #[comptime] diagnostics_enabled: bool,
    out_indices: &mut Tensor<I>,
    out_distances: &mut Tensor<F>,
    diagnostics: &mut Tensor<I>,
    staged_positions: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let d_chunks = comptime![reg_features.div_ceil(4)];
    let lane = UNIT_POS_PLANE as usize;
    let plane = UNIT_POS as usize / tile_width;
    let candidate_base = plane * tile_width * d_chunks;
    let id_base = plane * tile_width;
    let mut candidates = Shared::<[KdVector<F>]>::new_slice(comptime![
        KD_TILE_PLANES * tile_width * reg_features.div_ceil(4)
    ]);
    let mut ids = Shared::<[I]>::new_slice(comptime![KD_TILE_PLANES * tile_width]);
    let mut tile = CUBE_POS_X as usize * KD_TILE_PLANES + plane;
    let tile_stride = CUBE_COUNT_X as usize * KD_TILE_PLANES;
    let tile_count = (n as usize).div_ceil(tile_width);
    while tile < tile_count {
        let position = tile * tile_width + lane;
        let active = position < n as usize;
        let mut row = 0usize;
        if active {
            row = u32::cast_from(indices[position]) as usize;
        }
        let width = k as usize;
        let root = row * width;
        if active {
            for slot in 0..width {
                out_indices[root + slot] = I::cast_from(n);
                out_distances[root + slot] = F::new(f32::INFINITY);
            }
        }
        let mut query = Array::<KdVector<F>>::new(comptime![reg_features.div_ceil(4)]);
        load_query::<F>(data_v, position, d_chunks, &mut query, reg_features);
        let mut worst_distance = F::new(f32::INFINITY);
        let mut worst_index = n as usize;
        let mut leaf_positions_scanned = 0i32;
        let mut staged = 0i32;
        let mut s = 0usize;
        let end = n as usize;
        while s < end {
            let mut len = end - s;
            if len > tile_width {
                len = tile_width;
            }
            let mut element = lane;
            while element < len * d_chunks {
                candidates[candidate_base + element] = data_v[s * d_chunks + element];
                element += tile_width;
            }
            if lane < len {
                ids[id_base + lane] = indices[s + lane];
            }
            sync_plane();
            if comptime![diagnostics_enabled] {
                staged += len as i32;
            }
            if active {
                if comptime![diagnostics_enabled] {
                    leaf_positions_scanned += len as i32;
                }
                for j in 0..len {
                    let candidate = u32::cast_from(ids[id_base + j]) as usize;
                    if candidate != row {
                        let mut distance = F::new(0.0f32);
                        #[unroll]
                        for c in 0..comptime![reg_features.div_ceil(4)] {
                            let lanes = comptime![(reg_features - c * 4).min(4)];
                            distance = chunk_distance::<F>(
                                query[c],
                                candidates[candidate_base + j * d_chunks + c],
                                distance,
                                lanes,
                            );
                        }
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
                                    let left_index = u32::cast_from(out_indices[left_at]) as usize;
                                    let right_index =
                                        u32::cast_from(out_indices[right_at]) as usize;
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
                            worst_distance = out_distances[root];
                            worst_index = u32::cast_from(out_indices[root]) as usize;
                        }
                    }
                }
            }
            sync_plane();
            s += tile_width;
        }
        if active {
            if comptime![diagnostics_enabled] {
                diagnostics[row * 2] = I::cast_from(0);
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
        if comptime![diagnostics_enabled] && lane == 0 {
            staged_positions[tile] = I::cast_from(staged);
        }
        tile += tile_stride;
    }
}

#[cube(launch)]
fn kd_neighbors_cpu_kernel<F: Float, I: Int, Lanes: Size>(
    data_v: &Tensor<KdVector<F>>,
    data: &Tensor<F>,
    soa: &Tensor<Vector<F, Lanes>>,
    slot_ids: &Tensor<I>,
    indices: &Tensor<I>,
    starts: &Tensor<I>,
    ends: &Tensor<I>,
    is_leaf: &Tensor<I>,
    bounds_min_v: &Tensor<KdVector<F>>,
    bounds_max_v: &Tensor<KdVector<F>>,
    d: u32,
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
    let unit = ABSOLUTE_POS_X as usize;
    let mut work = 0usize;
    let mut local_row = unit * KD_QUERY_BLOCK;
    let row_stride = (CUBE_DIM_X * CUBE_COUNT_X) as usize;
    while local_row < row_count as usize {
        if local_row < n as usize {
            let row = u32::cast_from(indices[local_row]) as usize;
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

                        if start < end {
                            if comptime![diagnostics_enabled] {
                                leaf_positions_scanned += (end - start) as i32;
                            }
                            let lanes = Lanes::value();
                            for tile in start / lanes..end.div_ceil(lanes) {
                                let mut acc = Vector::<F, Lanes>::new(F::new(0.0_f32));
                                if comptime![reg_features > 0] {
                                    #[unroll]
                                    for feature in 0..reg_features {
                                        let delta = Vector::<F, Lanes>::new(
                                            query[feature / 4].extract(feature % 4),
                                        ) - soa[tile * d as usize + feature];
                                        acc += delta * delta;
                                    }
                                } else {
                                    for feature in 0..d as usize {
                                        let delta = Vector::<F, Lanes>::new(
                                            data[row * d as usize + feature],
                                        ) - soa[tile * d as usize + feature];
                                        acc += delta * delta;
                                    }
                                }
                                let mut hit = false;
                                #[unroll]
                                for lane in 0..Lanes::value() {
                                    let position = tile * lanes + lane;
                                    if position >= start
                                        && position < end
                                        && acc.extract(lane) <= worst_distance
                                    {
                                        hit = true;
                                    }
                                }
                                if hit {
                                    #[unroll]
                                    for lane in 0..Lanes::value() {
                                        let position = tile * lanes + lane;
                                        if position >= start && position < end {
                                            let candidate =
                                                u32::cast_from(slot_ids[position]) as usize;
                                            let distance = acc.extract(lane);

                                            if candidate != row
                                                && (distance < worst_distance
                                                    || (distance == worst_distance
                                                        && candidate < worst_index))
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
                                                            u32::cast_from(out_indices[left_at])
                                                                as usize;
                                                        let right_index =
                                                            u32::cast_from(out_indices[right_at])
                                                                as usize;
                                                        if out_distances[right_at]
                                                            > out_distances[left_at]
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
                                                        u32::cast_from(out_indices[parent_at])
                                                            as usize;
                                                    let child_index =
                                                        u32::cast_from(out_indices[child_at])
                                                            as usize;
                                                    if out_distances[parent_at]
                                                        > out_distances[child_at]
                                                        || (out_distances[parent_at]
                                                            == out_distances[child_at]
                                                            && parent_index >= child_index)
                                                    {
                                                        heap = false;
                                                    } else {
                                                        let parent_distance =
                                                            out_distances[parent_at];
                                                        let parent_index = out_indices[parent_at];
                                                        out_distances[parent_at] =
                                                            out_distances[child_at];
                                                        out_indices[parent_at] =
                                                            out_indices[child_at];
                                                        out_distances[child_at] = parent_distance;
                                                        out_indices[child_at] = parent_index;
                                                        parent = largest;
                                                    }
                                                }
                                                worst_distance = out_distances[root];
                                                worst_index =
                                                    u32::cast_from(out_indices[root]) as usize;
                                            }
                                        }
                                    }
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
        work += 1;

        local_row =
            (unit + (work / KD_QUERY_BLOCK) * row_stride) * KD_QUERY_BLOCK + work % KD_QUERY_BLOCK;
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

/// Launch plan of every query/reference search: bounds-pruned per-query traversal
/// over padded rows. Exhaustive F2 tiles and CPU leaf-SoA blocks stay exclusive to
/// the fitting self-query entry point; this is a fixed choice, not a fallback.
pub(crate) const fn kd_query_plan() -> KdNeighborPlan {
    KdNeighborPlan {
        layout: KdCandidateLayout::PaddedRows,
        order: KdQueryOrder::Cyclic,
    }
}

/// Device bytes of one prepared reference pack (padded rows and bounds).
pub(crate) fn kd_reference_pack_bytes(rows: usize, nodes: usize, features: usize) -> Option<usize> {
    kd_padded_bytes(rows, nodes, features)
}

/// Live device bytes of one query batch: padded query copy, `[rows, k]` outputs
/// and the diagnostics allocation.
pub(crate) fn kd_query_pack_bytes(query_rows: usize, features: usize, k: usize) -> Option<usize> {
    let diagnostics = if kd_diagnostics_enabled() {
        query_rows.checked_mul(8)?
    } else {
        4
    };
    query_rows
        .checked_mul(kd_padded_width(features)?)?
        .checked_mul(4)?
        .checked_add(query_rows.checked_mul(k)?.checked_mul(8)?)?
        .checked_add(diagnostics)
}

pub(super) fn prepare_kd_reference(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
) -> DeviceKdReference<CubeBackend> {
    let data = into_contiguous(data);
    let tree = into_contiguous_tree(tree);
    validate_tree(&data, &tree);
    let [rows, features] = data.shape().dims::<2>();
    let [nodes] = tree.starts.shape().dims::<1>();
    u32::try_from(rows).expect("KD reference rows must fit u32");
    DeviceKdReference {
        padded_data: pad_rows(&data, rows, features),
        padded_min: pad_rows(&tree.bounds_min, nodes, features),
        padded_max: pad_rows(&tree.bounds_max, nodes, features),
        indices: tree.indices,
        starts: tree.starts,
        ends: tree.ends,
        is_leaf: tree.is_leaf,
        rows,
        nodes,
        n_levels: tree.n_levels,
        n_features: features,
    }
}

pub(super) fn exact_neighbors_kd_query(
    query: FloatTensor<CubeBackend>,
    reference: DeviceKdReference<CubeBackend>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let diagnostics_enabled = kd_diagnostics_enabled();
    let nodes = reference.nodes;
    let (output_indices, output_distances, diagnostics) = launch_kd_query(
        query,
        reference,
        k,
        exclude_self,
        query_offset,
        diagnostics_enabled,
        KD_NEIGHBORS_CUBE_DIM.x,
        None,
    );
    if diagnostics_enabled {
        let rows = output_indices.shape().dims::<2>()[0];
        emit_kd_diagnostics("kd_neighbors_query", diagnostics, rows, nodes);
    }
    (output_indices, output_distances)
}

fn validate_reference(reference: &DeviceKdReference<CubeBackend>) {
    let rows = reference.rows;
    let nodes = reference.nodes;
    let width = kd_padded_width(reference.n_features).expect("KD padded width overflow");
    assert!(
        rows > 0 && reference.n_features > 0,
        "KD reference must be non-empty"
    );
    i32::try_from(rows).expect("KD reference rows must fit destination indices");
    i32::try_from(nodes).expect("KD node count must fit diagnostic counters");
    for (tensor, len, name) in [
        (&reference.padded_data, rows * width, "padded data"),
        (
            &reference.padded_min,
            nodes * width,
            "padded minimum bounds",
        ),
        (
            &reference.padded_max,
            nodes * width,
            "padded maximum bounds",
        ),
    ] {
        assert_eq!(tensor.dtype, DType::F32, "KD reference {name} must be f32");
        assert_eq!(
            tensor.shape().dims::<1>(),
            [len],
            "KD reference {name} size"
        );
    }
    for (tensor, len, name) in [
        (&reference.indices, rows, "indices"),
        (&reference.starts, nodes, "starts"),
        (&reference.ends, nodes, "ends"),
        (&reference.is_leaf, nodes, "leaf flags"),
    ] {
        assert_eq!(tensor.dtype, DType::I32, "KD reference {name} must be i32");
        assert_eq!(
            tensor.shape().dims::<1>(),
            [len],
            "KD reference {name} size"
        );
    }
    assert!(
        reference.n_levels < KD_STACK_LEN,
        "KD tree exceeds fixed traversal stack"
    );
}

#[allow(clippy::too_many_arguments)]
fn launch_kd_query(
    query: FloatTensor<CubeBackend>,
    reference: DeviceKdReference<CubeBackend>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
    diagnostics_enabled: bool,
    preferred_width: u32,
    max_cubes_override: Option<u32>,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    let query = into_contiguous(query);
    let reference = DeviceKdReference {
        padded_data: into_contiguous(reference.padded_data),
        padded_min: into_contiguous(reference.padded_min),
        padded_max: into_contiguous(reference.padded_max),
        indices: into_contiguous(reference.indices),
        starts: into_contiguous(reference.starts),
        ends: into_contiguous(reference.ends),
        is_leaf: into_contiguous(reference.is_leaf),
        ..reference
    };
    validate_reference(&reference);
    let [rows, features] = query.shape().dims::<2>();
    assert_eq!(query.dtype, DType::F32, "KD queries must be f32");
    assert_eq!(
        features, reference.n_features,
        "KD query feature count mismatch"
    );
    assert!(rows > 0, "KD query batch must be non-empty");
    assert!(k > 0, "KD neighbor count must be positive");
    let rows_u32 = u32::try_from(rows).expect("KD query rows must fit u32");
    query_offset
        .checked_add(rows_u32)
        .expect("KD query offset range must fit u32");
    let output_len = rows
        .checked_mul(k as usize)
        .expect("KD neighbor output size overflow");
    let output_indices = allocate_int_2d(&query, rows, k as usize, output_len);
    let output_distances = allocate_float_2d(&query, rows, k as usize, output_len);
    let diagnostics = allocate_diagnostics(&query, rows, diagnostics_enabled);
    let client = query.client.clone();
    let (d_chunks, reg_features) = kd_vector_chunks(features);
    // Fixed pruned-traversal plan: never the fitting F2 or leaf-SoA specializations.
    assert_eq!(kd_query_plan().layout, KdCandidateLayout::PaddedRows);
    let query_padded = pad_rows(&query, rows, features);
    let (cube_count, dim) = kd_launch_geometry_with_width(
        &client,
        rows,
        CubeDim::new_1d(preferred_width),
        max_cubes_override,
    );
    kd_neighbors_kernel::launch(
        &client,
        cube_count,
        dim,
        padded_vector_view(&reference.padded_data, reference.rows, d_chunks),
        padded_vector_view(&query_padded, rows, d_chunks),
        reference.indices.clone().into_tensor_arg(),
        reference.starts.clone().into_tensor_arg(),
        reference.ends.clone().into_tensor_arg(),
        reference.is_leaf.clone().into_tensor_arg(),
        padded_vector_view(&reference.padded_min, reference.nodes, d_chunks),
        padded_vector_view(&reference.padded_max, reference.nodes, d_chunks),
        rows_u32,
        u32::try_from(reference.rows).expect("validated KD reference rows"),
        k,
        u32::try_from(d_chunks).expect("KD feature chunk count must fit u32"),
        query_offset,
        reg_features,
        diagnostics_enabled,
        exclude_self,
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
    let plan = kd_neighbor_plan(&data.client, data.shape().dims::<2>()[1]);
    #[cfg(test)]
    {
        launch_kd_neighbors_impl(
            data,
            tree,
            k,
            diagnostics_enabled,
            plan,
            KD_NEIGHBORS_CUBE_DIM.x,
            None,
        )
    }
    #[cfg(not(test))]
    {
        launch_kd_neighbors_impl(data, tree, k, diagnostics_enabled, plan)
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
        KdNeighborPlan {
            layout: KdCandidateLayout::PaddedRows,
            order: KdQueryOrder::Cyclic,
        },
        preferred_width,
        max_cubes_override,
    )
}

fn launch_kd_neighbors_impl(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
    plan: KdNeighborPlan,
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
    let bounds_min_padded = pad_rows(&tree.bounds_min, nodes, features);
    let bounds_max_padded = pad_rows(&tree.bounds_max, nodes, features);
    if plan.order == KdQueryOrder::PlaneTiles {
        let KdCandidateLayout::LeafRows { tile } = plan.layout else {
            panic!("tile layout required")
        };
        #[cfg(test)]
        let expected_plan = kd_tile_plan_for_test(&client, features);
        #[cfg(not(test))]
        let expected_plan = kd_tile_plan(&client, features);
        assert_eq!(expected_plan, Some(plan));
        kd_neighbor_pack_bytes(plan, n, nodes, features).expect("KD leaf rows pack size overflow");
        let slots = kd_soa_slots(n, tile).expect("KD tile slots overflow");
        let len = slots
            .checked_mul(kd_padded_width(features).expect("KD padded width overflow"))
            .expect("KD leaf rows length overflow");
        let len_u32 = u32::try_from(len).expect("KD leaf rows elements");
        let leaf_rows = allocate_float(&data, len);
        let (pack_count, pack_dim) = kd_launch_geometry(&client, len);
        pack_leaf_rows_kernel::launch(
            &client,
            pack_count,
            pack_dim,
            data.clone().into_tensor_arg(),
            tree.indices.clone().into_tensor_arg(),
            leaf_rows.clone().into_tensor_arg(),
            u32::try_from(n).expect("KD rows"),
            u32::try_from(features).expect("KD features"),
            u32::try_from(kd_padded_width(features).unwrap()).expect("KD width"),
            len_u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        let tile_count = n.div_ceil(tile);
        kd_tile_diagnostic_bytes(n, tile, diagnostics_enabled)
            .expect("KD staged diagnostic size overflow");
        let staged = if diagnostics_enabled {
            allocate_int(&data, tile_count)
        } else {
            diagnostics.clone()
        };
        let dim = CubeDim::new_1d(u32::try_from(tile * KD_TILE_PLANES).expect("KD tile units"));
        #[cfg(test)]
        let max_cubes = max_cubes_override.unwrap_or(client.properties().hardware.max_cube_count.0);
        #[cfg(not(test))]
        let max_cubes = client.properties().hardware.max_cube_count.0;
        let count = launch_count(slots, dim.x, max_cubes);
        kd_neighbors_tile_kernel::launch(
            &client,
            CubeCount::Static(count, 1, 1),
            dim,
            padded_vector_view(&leaf_rows, slots, d_chunks),
            tree.indices.clone().into_tensor_arg(),
            u32::try_from(n).expect("KD rows"),
            k,
            reg_features,
            tile,
            diagnostics_enabled,
            output_indices.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            diagnostics.clone().into_tensor_arg(),
            staged.clone().into_tensor_arg(),
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        if diagnostics_enabled {
            emit_tile_staging(staged, n, tile);
        }
        return (output_indices, output_distances, diagnostics);
    }
    let data_padded = pad_rows(&data, n, features);
    if plan
        != (KdNeighborPlan {
            layout: KdCandidateLayout::PaddedRows,
            order: KdQueryOrder::Cyclic,
        })
    {
        assert!(
            client.properties().hardware.num_cpu_cores.is_some(),
            "CPU KD plan on non-CPU client"
        );
        let lanes = match plan.layout {
            KdCandidateLayout::LeafSoa { lanes } => lanes,
            _ => panic!("unsupported CPU KD layout"),
        };
        assert!(client.io_optimized_vector_sizes(4).any(|w| w == lanes));
        assert_eq!(plan.order, KdQueryOrder::LeafBlocks);
        let (soa, slot_ids) = pack_leaf_soa(&data, &data_padded, &tree.indices, n, features, lanes);
        let (cube_count, dim) = kd_launch_geometry(&client, n);
        kd_neighbors_cpu_kernel::launch(
            &client,
            cube_count,
            dim,
            lanes as VectorSize,
            padded_vector_view(&data_padded, n, d_chunks),
            data.clone().into_tensor_arg(),
            soa_vector_view(
                &soa,
                kd_soa_slots(n, lanes).unwrap() / lanes * features,
                lanes,
            ),
            slot_ids.into_tensor_arg(),
            tree.indices.clone().into_tensor_arg(),
            tree.starts.clone().into_tensor_arg(),
            tree.ends.clone().into_tensor_arg(),
            tree.is_leaf.clone().into_tensor_arg(),
            padded_vector_view(&bounds_min_padded, nodes, d_chunks),
            padded_vector_view(&bounds_max_padded, nodes, d_chunks),
            features as u32,
            n as u32,
            n as u32,
            k,
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
        return (output_indices, output_distances, diagnostics);
    }
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
        0,
        reg_features,
        diagnostics_enabled,
        true,
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

fn pack_leaf_soa(
    data: &FloatTensor<CubeBackend>,
    padded: &FloatTensor<CubeBackend>,
    indices: &IntTensor<CubeBackend>,
    rows: usize,
    features: usize,
    lanes: usize,
) -> (FloatTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let slots = kd_soa_slots(rows, lanes).expect("KD SoA slots overflow");
    let len = slots
        .checked_mul(features)
        .expect("KD SoA elements overflow");
    u32::try_from(slots).expect("KD slots must fit u32");
    u32::try_from(len).expect("KD SoA elements must fit u32");
    let soa = allocate_float(data, len);
    let ids = allocate_int(data, slots);
    let (count, dim) = kd_launch_geometry(&data.client, len);
    pack_leaf_soa_kernel::launch(
        &data.client,
        count,
        dim,
        padded_vector_view(padded, rows, features.div_ceil(4)),
        indices.clone().into_tensor_arg(),
        soa.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),
        rows as u32,
        features as u32,
        features.div_ceil(4) as u32,
        lanes as u32,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    (soa, ids)
}
fn soa_vector_view(tensor: &CubeTensor, vectors: usize, lanes: usize) -> TensorArg {
    assert_eq!(
        tensor.shape().dims::<1>(),
        [vectors.checked_mul(lanes).expect("KD SoA view overflow")]
    );
    // SAFETY: fresh contiguous f32 allocation with exactly vectors * lanes scalar slots.
    unsafe { TensorArg::from_raw_parts(tensor.handle.clone(), [1].into(), [vectors].into()) }
}
#[cfg(test)]
fn launch_kd_neighbors_with_plan(
    data: FloatTensor<CubeBackend>,
    tree: DeviceKdTree<CubeBackend>,
    k: u32,
    diagnostics_enabled: bool,
    plan: KdNeighborPlan,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    launch_kd_neighbors_impl(data, tree, k, diagnostics_enabled, plan, 64, None)
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

#[cfg(test)]
std::thread_local! {
    static TILE_STAGED: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn emit_tile_staging(tensor: IntTensor<CubeBackend>, rows: usize, tile: usize) {
    let counts = into_data_sync(tensor)
        .try_to_vec::<i32>()
        .expect("KD staged counters");
    assert_eq!(counts.len(), rows.div_ceil(tile));
    let sum = counts
        .iter()
        .try_fold(0u64, |sum, &value| {
            sum.checked_add(u64::try_from(value).expect("nonnegative staged positions"))
        })
        .expect("KD staged total overflow");
    assert_eq!(sum, rows as u64 * rows.div_ceil(tile) as u64);
    #[cfg(test)]
    TILE_STAGED.with(|last| *last.borrow_mut() = counts.clone());
    eprintln!(
        "{{\"operation\":\"kd_neighbors_tile_staging\",\"rows\":{rows},\"tile_width\":{tile},\"planes_per_cube\":2,\"traverse\":false,\"staged_positions\":{sum},\"per_tile\":{counts:?},\"nodes_popped_semantics\":\"zero_flat_scan\",\"leaf_positions_scanned_semantics\":\"lane_evaluated_positions_including_self\"}}"
    );
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

    fn phase7_cuda() -> Option<CubeDevice> {
        let count =
            fast_cubecl_common::cuda_backend::try_device_count().expect("CUDA discovery failed");
        if count == 0 {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED: phase7 requires CUDA");
            None
        } else {
            Some(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }))
        }
    }

    #[test]
    fn phase7_pack_leaf_rows_bits_and_tail_cuda() {
        let Some(device) = phase7_cuda() else { return };
        for rows in [1, 31, 33, 257, 1000, 4109] {
            for features in [1, 2, 3, 4, 5, 8, 16, 29, 30, 32] {
                let mut values = splitmix_values(rows, features);
                for i in (0..values.len()).step_by(13) {
                    values[i] = -0.0;
                }
                let (data, tree) = make_neighbor_inputs(&values, rows, features, &device);
                let tile = data.client.properties().hardware.plane_size_min as usize;
                let slots = kd_soa_slots(rows, tile).unwrap();
                let width = kd_padded_width(features).unwrap();
                let len = slots * width;
                let output = allocate_float(&data, len);
                let (count, dim) = kd_launch_geometry(&data.client, len);
                pack_leaf_rows_kernel::launch(
                    &data.client,
                    count,
                    dim,
                    data.clone().into_tensor_arg(),
                    tree.indices.clone().into_tensor_arg(),
                    output.clone().into_tensor_arg(),
                    rows as u32,
                    features as u32,
                    width as u32,
                    len as u32,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                    ],
                );
                let padded = into_data_sync(pad_rows(&data, rows, features))
                    .try_to_vec::<f32>()
                    .unwrap();
                let order = into_data_sync(tree.indices).try_to_vec::<i32>().unwrap();
                let actual = into_data_sync(output).try_to_vec::<f32>().unwrap();
                for p in 0..slots {
                    for f in 0..width {
                        let expected = if p < rows {
                            padded[order[p] as usize * width + f]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            actual[p * width + f].to_bits(),
                            expected.to_bits(),
                            "n={rows}, d={features}, p={p}, f={f}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn phase7_checked_pack_and_diagnostic_sizing() {
        let plan = KdNeighborPlan {
            layout: KdCandidateLayout::LeafRows { tile: 32 },
            order: KdQueryOrder::PlaneTiles,
        };
        let expected = (1024 + 126) * 8 * 4
            + kd_tile_diagnostic_bytes(1000, 32, kd_diagnostics_enabled()).unwrap();
        assert_eq!(kd_neighbor_pack_bytes(plan, 1000, 63, 5), Some(expected));
        assert_eq!(
            kd_neighbor_pack_allocations(plan, 1000, 5),
            Some([
                1024 * 8,
                kd_tile_diagnostic_bytes(1000, 32, kd_diagnostics_enabled()).unwrap() / 4
            ])
        );
        for (n, nodes, d) in [
            (usize::MAX, 0, 1),
            (1, usize::MAX, 1),
            (1, 0, usize::MAX),
            (usize::MAX / 8, 0, 32),
        ] {
            assert_eq!(kd_neighbor_pack_bytes(plan, n, nodes, d), None);
        }
        assert_eq!(kd_neighbor_pack_allocations(plan, usize::MAX, 5), None);
        assert_eq!(kd_neighbor_pack_allocations(plan, 1, usize::MAX), None);
        assert_eq!(kd_tile_diagnostic_bytes(4109, 32, true), Some(129 * 4));
        assert_eq!(kd_tile_diagnostic_bytes(4109, 32, false), Some(0));
        assert_eq!(kd_tile_diagnostic_bytes(usize::MAX, 32, true), None);
        assert_eq!(kd_tile_diagnostic_bytes(1, 0, true), None);
    }

    #[test]
    fn phase7_tile_resource_limits_reject_insufficient_capacity() {
        for features in [1usize, 16, 32] {
            let units = 32 * KD_TILE_PLANES;
            let shared = units * (features.div_ceil(4) * 16 + 4);
            assert_eq!(kd_tile_shared_bytes(features, 32), Some(shared));
            assert!(kd_tile_resources_fit(features, 32, shared, units, units));
            assert!(!kd_tile_resources_fit(
                features,
                32,
                shared - 1,
                units,
                units
            ));
            assert!(!kd_tile_resources_fit(
                features,
                32,
                shared,
                units - 1,
                units
            ));
            assert!(!kd_tile_resources_fit(
                features,
                32,
                shared,
                units,
                units - 1
            ));
        }
        assert!(!kd_tile_resources_fit(
            32,
            usize::MAX,
            usize::MAX,
            usize::MAX,
            usize::MAX
        ));
        assert!(!kd_tile_resources_fit(
            usize::MAX,
            32,
            usize::MAX,
            usize::MAX,
            usize::MAX
        ));
        assert!(!kd_tile_resources_fit(
            16,
            0,
            usize::MAX,
            usize::MAX,
            usize::MAX
        ));
    }

    #[test]
    fn phase7_flat_chunks_are_bounded_and_partial() {
        for rows in [257usize, 287, 1000, 4109] {
            let mut consumed = Vec::new();
            for start in (0..rows).step_by(32) {
                let len = (rows - start).min(32);
                for chunks in 1..=8 {
                    for lane in 0..32 {
                        for element in (lane..len * chunks).step_by(32) {
                            assert!(start * chunks + element < rows * chunks);
                        }
                    }
                }
                consumed.extend(start..start + len);
            }
            assert_eq!(consumed, (0..rows).collect::<Vec<_>>());
            if rows == 4109 {
                assert_eq!((rows / 32 * 32, rows % 32), (4096, 13));
            }
        }
    }

    #[test]
    fn phase7_routing_cpu_and_cuda() {
        let (cpu, _) = make_neighbor_inputs(&[0.0], 1, 1, &CubeDevice::Cpu(Default::default()));
        for d in [1, 2, 8, 15, 16, 32, 33] {
            assert_eq!(kd_tile_plan(&cpu.client, d), None);
            assert!(matches!(
                kd_neighbor_plan(&cpu.client, d).layout,
                KdCandidateLayout::LeafSoa { .. }
            ));
        }
        let Some(device) = phase7_cuda() else { return };
        let (data, _) = make_neighbor_inputs(&[0.0], 1, 1, &device);
        for d in [16, 32] {
            assert_eq!(
                kd_neighbor_plan(&data.client, d).order,
                KdQueryOrder::PlaneTiles
            );
        }
        for d in [1, 2, 8, 15, 33, 40] {
            assert_eq!(kd_tile_plan(&data.client, d), None);
            assert_eq!(
                kd_neighbor_plan(&data.client, d).order,
                KdQueryOrder::Cyclic
            );
        }
    }

    fn phase7_reference_matrix(features: usize) {
        let Some(device) = phase7_cuda() else { return };
        for rows in [257, 287, 1000, 4109] {
            for pattern in 0..3 {
                let mut values = splitmix_values(rows, features);
                if pattern == 1 {
                    for r in 0..rows {
                        for f in 0..features {
                            values[r * features + f] = ((r % 19 + f) % 7) as f32;
                        }
                    }
                } else if pattern == 2 {
                    for r in 0..rows {
                        for f in 0..features {
                            values[r * features + f] = if r % 7 == 0 {
                                -0.0
                            } else {
                                ((r >> (f % 8)) % 3) as f32 - 1.0
                            };
                        }
                    }
                }
                for k in [1, 5, 16, 17] {
                    let (data, tree) = make_neighbor_inputs(&values, rows, features, &device);
                    let tile = data.client.properties().hardware.plane_size_min as usize;
                    let (ref_ids, ref_distances, _) =
                        exact_neighbors_kd_reference(data, tree, k, false);
                    let ref_ids = into_data_sync(ref_ids).try_to_vec::<i32>().unwrap();
                    let ref_bits = into_data_sync(ref_distances)
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();

                    let mut default_staging = None;
                    for cap in [None, Some(1)] {
                        for enabled in [false, true] {
                            let (data, tree) =
                                make_neighbor_inputs(&values, rows, features, &device);
                            let plan = kd_tile_plan_for_test(&data.client, features)
                                .expect("required tile capability");
                            let (ids, distances, diagnostics) =
                                launch_kd_neighbors_impl(data, tree, k, enabled, plan, 64, cap);
                            let context = format!(
                                "n={rows}, d={features}, k={k}, pattern={pattern}, cap={cap:?}, diagnostics={enabled}"
                            );
                            assert_eq!(
                                into_data_sync(ids).try_to_vec::<i32>().unwrap(),
                                ref_ids,
                                "{context}"
                            );
                            assert_eq!(
                                into_data_sync(distances)
                                    .try_to_vec::<f32>()
                                    .unwrap()
                                    .into_iter()
                                    .map(f32::to_bits)
                                    .collect::<Vec<_>>(),
                                ref_bits,
                                "{context}"
                            );
                            if enabled {
                                let counts =
                                    into_data_sync(diagnostics).try_to_vec::<i32>().unwrap();
                                assert_eq!(counts.len(), rows * 2);
                                for row in 0..rows {
                                    assert_eq!(
                                        (counts[row * 2], counts[row * 2 + 1]),
                                        (0, rows as i32),
                                        "{context}"
                                    );
                                }
                                TILE_STAGED.with(|last| {
                                    let staged = last.borrow();
                                    assert_eq!(staged.len(), rows.div_ceil(tile));
                                    assert!(staged.iter().all(|&v| v >= 0 && v <= rows as i32));

                                    assert!(staged.iter().all(|&v| v == rows as i32));

                                    if cap.is_none() {
                                        default_staging = Some(staged.clone());
                                    } else {
                                        assert_eq!(
                                            Some(&*staged),
                                            default_staging.as_ref(),
                                            "capped-grid staged positions differ: {context}"
                                        );
                                    }
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn phase7_reference_d1_cuda() {
        phase7_reference_matrix(1);
    }
    #[test]
    fn phase7_reference_d2_cuda() {
        phase7_reference_matrix(2);
    }
    #[test]
    fn phase7_reference_d3_cuda() {
        phase7_reference_matrix(3);
    }
    #[test]
    fn phase7_reference_d4_cuda() {
        phase7_reference_matrix(4);
    }
    #[test]
    fn phase7_reference_d5_cuda() {
        phase7_reference_matrix(5);
    }
    #[test]
    fn phase7_reference_d8_cuda() {
        phase7_reference_matrix(8);
    }
    #[test]
    fn phase7_reference_d16_cuda() {
        phase7_reference_matrix(16);
    }
    #[test]
    fn phase7_reference_d29_cuda() {
        phase7_reference_matrix(29);
    }
    #[test]
    fn phase7_reference_d30_cuda() {
        phase7_reference_matrix(30);
    }
    #[test]
    fn phase7_reference_d32_cuda() {
        phase7_reference_matrix(32);
    }

    #[test]
    fn check_pack_leaf_soa() {
        let device = CubeDevice::Cpu(Default::default());
        for features in [1, 3, 30, 33] {
            let values: Vec<_> = (0..37 * features)
                .map(|i| if i % 7 == 0 { -0.0 } else { (i % 19) as f32 })
                .collect();
            let (data, tree) = make_neighbor_inputs(&values, 37, features, &device);
            let indices = into_data_sync(tree.indices.clone())
                .try_to_vec::<i32>()
                .unwrap();
            let padded = pad_rows(&data, 37, features);
            for lanes in data.client.io_optimized_vector_sizes(4) {
                let (soa, ids) = pack_leaf_soa(&data, &padded, &tree.indices, 37, features, lanes);
                let actual = into_data_sync(soa).try_to_vec::<f32>().unwrap();
                let ids = into_data_sync(ids).try_to_vec::<i32>().unwrap();
                for position in 0..ids.len() {
                    assert_eq!(
                        ids[position],
                        if position < 37 { indices[position] } else { 37 }
                    );
                    for feature in 0..features {
                        let expected = if position < 37 {
                            values[indices[position] as usize * features + feature]
                        } else {
                            0.0
                        };
                        assert_eq!(
                            actual[(position / lanes * features + feature) * lanes
                                + position % lanes]
                                .to_bits(),
                            expected.to_bits()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn kd_neighbor_pack_sizing_checked() {
        let plan = KdNeighborPlan {
            layout: KdCandidateLayout::LeafSoa { lanes: 16 },
            order: KdQueryOrder::LeafBlocks,
        };
        assert_eq!(kd_neighbor_pack_bytes(plan, 1000, 63, 5), Some(60224));
        assert_eq!(kd_neighbor_pack_bytes(plan, usize::MAX, 0, 1), None);
        assert_eq!(kd_neighbor_pack_bytes(plan, 1, usize::MAX, 1), None);
        assert_eq!(kd_neighbor_pack_bytes(plan, 1, 0, usize::MAX), None);
        assert_eq!(
            kd_neighbor_pack_bytes(
                KdNeighborPlan {
                    layout: KdCandidateLayout::LeafSoa { lanes: 0 },
                    ..plan
                },
                1,
                0,
                1
            ),
            None
        );
        assert_eq!(kd_soa_slots(usize::MAX, 16), None);
        assert_eq!(kd_neighbor_pack_bytes(plan, usize::MAX / 8, 0, 1), None);
    }

    #[test]
    fn kd_cpu_enabled_diagnostics_match_disabled_launch() {
        let device = CubeDevice::Cpu(Default::default());
        let values = splitmix_values(257, 33);
        let (data, _) = make_neighbor_inputs(&values, 257, 33, &device);
        for lanes in data.client.io_optimized_vector_sizes(4) {
            let plan = KdNeighborPlan {
                layout: KdCandidateLayout::LeafSoa { lanes },
                order: KdQueryOrder::LeafBlocks,
            };
            let (enabled_data, enabled_tree) = make_neighbor_inputs(&values, 257, 33, &device);
            let (disabled_data, disabled_tree) = make_neighbor_inputs(&values, 257, 33, &device);
            let (a, b, _) =
                launch_kd_neighbors_with_plan(enabled_data, enabled_tree, 17, true, plan);
            let (c, d, _) =
                launch_kd_neighbors_with_plan(disabled_data, disabled_tree, 17, false, plan);
            assert_eq!(
                into_data_sync(a).try_to_vec::<i32>().unwrap(),
                into_data_sync(c).try_to_vec::<i32>().unwrap()
            );
            assert_eq!(
                into_data_sync(b)
                    .try_to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                into_data_sync(d)
                    .try_to_vec::<f32>()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }

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

    fn brute_query(
        query: &[f32],
        reference: &[f32],
        features: usize,
        k: usize,
        exclude_offset: Option<usize>,
    ) -> (Vec<i32>, Vec<u32>) {
        let rows = query.len() / features;
        let n = reference.len() / features;
        let mut ids = vec![n as i32; rows * k];
        let mut distances = vec![f32::INFINITY.to_bits(); rows * k];
        for row in 0..rows {
            let mut candidates = (0..n)
                .filter(|&c| exclude_offset.is_none_or(|offset| c != offset + row))
                .map(|c| {
                    let distance = (0..features).fold(0.0f32, |sum, f| {
                        let delta = query[row * features + f] - reference[c * features + f];
                        sum + delta * delta
                    });
                    (distance, c as i32)
                })
                .collect::<Vec<_>>();
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            for (slot, (distance, id)) in candidates.into_iter().take(k).enumerate() {
                ids[row * k + slot] = id;
                distances[row * k + slot] = distance.to_bits();
            }
        }
        (ids, distances)
    }

    fn check_query_routes(device: &CubeDevice) {
        // The query plan never adopts the fitting F2 or leaf-SoA routes, even at
        // widths where the fitting plan does.
        assert_eq!(
            kd_query_plan(),
            KdNeighborPlan {
                layout: KdCandidateLayout::PaddedRows,
                order: KdQueryOrder::Cyclic,
            }
        );
        for features in [2usize, 16, 30, 33] {
            let n = 203usize;
            // Eighth-quantized values keep every squared sum exact, so the host
            // oracle is bit-comparable regardless of device contraction.
            let quantize = |v: Vec<f32>| {
                v.into_iter()
                    .map(|x| (x * 8.0).round() / 8.0)
                    .collect::<Vec<_>>()
            };
            let values = quantize(splitmix_values(n, features));
            let (data, tree) = make_neighbor_inputs(&values, n, features, device);
            let fitting_plan = kd_neighbor_plan(&data.client, features);
            if data.client.properties().hardware.num_cpu_cores.is_some() {
                assert_eq!(fitting_plan.order, KdQueryOrder::LeafBlocks);
            }
            let reference = prepare_kd_reference(data.clone(), tree);
            let rows = 71usize;
            let query_values = quantize(
                splitmix_values(rows + 5, features)[5 * features..]
                    .iter()
                    .map(|x| x * 0.75 + 0.3)
                    .collect(),
            );
            let query = CubeBackend::float_from_data(
                TensorData::new(query_values.clone(), [rows, features]),
                device,
            );
            let k = 9usize;
            let expected = brute_query(&query_values, &values, features, k, None);
            for (width, cap) in [(64u32, None), (32, Some(1u32)), (1, Some(3))] {
                for diagnostics in [false, true] {
                    let (ids, distances, counters) = launch_kd_query(
                        query.clone(),
                        reference.clone(),
                        k as u32,
                        false,
                        0,
                        diagnostics,
                        width,
                        cap,
                    );
                    let ids = into_data_sync(ids).try_to_vec::<i32>().unwrap();
                    let distances = into_data_sync(distances)
                        .try_to_vec::<f32>()
                        .unwrap()
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>();
                    assert_eq!((ids, distances), expected, "d={features} width={width}");
                    if diagnostics {
                        let counters = into_data_sync(counters).try_to_vec::<i32>().unwrap();
                        assert_eq!(counters.len(), rows * 2);
                        // Pruning must skip leaf positions; exhaustive F2 would scan all n.
                        let scanned = counters.chunks(2).map(|c| c[1] as usize).sum::<usize>();
                        assert!(counters.chunks(2).all(|c| c[0] > 0));
                        // Per-row node counters exist only on the traversal kernel (F2 writes
                        // per-tile staging instead); low-dimensional data must also prune.
                        assert!(scanned <= rows * n);
                        if features == 2 {
                            assert!(scanned < rows * n, "d={features} must prune leaves");
                        }
                    }
                }
            }
            // Self exclusion uses explicit offsets, matching the fitting self query.
            let offset = 40usize;
            let batch = CubeBackend::float_from_data(
                TensorData::new(
                    values[offset * features..(offset + rows) * features].to_vec(),
                    [rows, features],
                ),
                device,
            );
            let (fit_ids, fit_distances) = exact_neighbors_kd(
                data.clone(),
                make_neighbor_inputs(&values, n, features, device).1,
                k as u32,
            );
            let fit_ids = into_data_sync(fit_ids).try_to_vec::<i32>().unwrap();
            let fit_distances = into_data_sync(fit_distances)
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            let (ids, distances) =
                exact_neighbors_kd_query(batch, reference, k as u32, true, offset as u32);
            let ids = into_data_sync(ids).try_to_vec::<i32>().unwrap();
            let distances = into_data_sync(distances)
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>();
            assert_eq!(ids, fit_ids[offset * k..(offset + rows) * k]);
            assert_eq!(distances, fit_distances[offset * k..(offset + rows) * k]);
        }
    }

    #[test]
    fn kd_query_pruned_routes_cpu() {
        check_query_routes(&CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn kd_query_pruned_routes_cuda() {
        let Some(device) = phase7_cuda() else { return };
        check_query_routes(&device);
    }

    #[test]
    fn kd_query_sizing_checked() {
        assert_eq!(kd_reference_pack_bytes(10, 3, 5), kd_padded_bytes(10, 3, 5));
        let diagnostics = if kd_diagnostics_enabled() { 7 * 8 } else { 4 };
        assert_eq!(
            kd_query_pack_bytes(7, 5, 3),
            Some(7 * 8 * 4 + 7 * 3 * 8 + diagnostics)
        );
        assert_eq!(kd_query_pack_bytes(usize::MAX, 5, 3), None);
        assert_eq!(kd_query_pack_bytes(7, usize::MAX, 3), None);
        assert_eq!(kd_query_pack_bytes(7, 5, usize::MAX), None);
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

        for rows in [1, 4, 9, 15, 17, 33, 257] {
            for features in [2, 16] {
                cases.push((rows, features, 5, Some("valid_special")));
            }
        }
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
                Some("valid_special") => {
                    for (i, value) in values.iter_mut().enumerate() {
                        *value = [0.0, -0.0, 1.0, 1.0, f32::MAX, -f32::MAX][i % 6];
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
            let mut launches = Vec::new();
            if client.properties().hardware.num_cpu_cores.is_some() {
                launches.push((32, kd_neighbor_plan(&client, features)));
                for lanes in client.io_optimized_vector_sizes(4) {
                    launches.push((
                        32,
                        KdNeighborPlan {
                            layout: KdCandidateLayout::LeafSoa { lanes },
                            order: KdQueryOrder::LeafBlocks,
                        },
                    ));
                }
            } else {
                let plan = KdNeighborPlan {
                    layout: KdCandidateLayout::PaddedRows,
                    order: KdQueryOrder::Cyclic,
                };
                assert_eq!(
                    kd_neighbor_plan(&client, features).order,
                    if (16..=32).contains(&features) {
                        KdQueryOrder::PlaneTiles
                    } else {
                        KdQueryOrder::Cyclic
                    }
                );
                launches.extend(widths.into_iter().map(|width| (width, plan)));
            }
            for (width, plan) in launches {
                let (data, tree) = make_neighbor_inputs(&values, rows, features, device);
                let (production_ids, production_distances, production_diagnostics) =
                    launch_kd_neighbors_impl(data, tree, k, true, plan, width, None);
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
                let (disabled_ids, disabled_distances, _) = launch_kd_neighbors_impl(
                    disabled_data,
                    disabled_tree,
                    k,
                    false,
                    plan,
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
                0,
                reg_features,
                diagnostics_enabled,
                true,
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
