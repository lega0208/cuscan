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
const KD_STACK_LEN: usize = 33;

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
    row_count: u32,
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
            let mut best = F::new(f32::INFINITY);
            let mut best_index = n;
            let mut stack = Array::<u32>::new(KD_STACK_LEN);
            let mut bound_stack = Array::<F>::new(KD_STACK_LEN);
            stack[0] = 0;
            bound_stack[0] = F::max(box_lower_bound(data, bounds_min, bounds_max, row, 0), core);
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
                if lower <= best && node_components[node] != component {
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
                            if right_bound <= best {
                                stack[stack_len] = right as u32;
                                bound_stack[stack_len] = right_bound;
                                stack_len += 1;
                            }
                            if left_bound <= best {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                        } else {
                            if left_bound <= best {
                                stack[stack_len] = left as u32;
                                bound_stack[stack_len] = left_bound;
                                stack_len += 1;
                            }
                            if right_bound <= best {
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
            if comptime![diagnostics_enabled] {
                diagnostics[row * 2] = I::cast_from(nodes_popped);
                diagnostics[row * 2 + 1] = I::cast_from(leaf_positions_scanned);
            }
        }
        local_row += row_stride;
    }
}

#[cube(launch)]
fn kd_neighbors_kernel<F: Float, I: Int>(
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

fn launch_count(rows: usize, dim: u32, max_cubes: u32) -> u32 {
    u32::try_from(rows.div_ceil(dim as usize).min(max_cubes as usize))
        .expect("bounded KD launch count")
}

fn kd_launch_geometry(client: &cubecl::client::Client, rows: usize) -> (CubeCount, CubeDim) {
    let hardware = &client.properties().hardware;
    let (dim, max_cubes) = if let Some(cores) = hardware.num_cpu_cores {
        (
            fast_cubecl_common::legal_cube_dim(client, CubeDim::new_1d(cores.max(1))),
            1,
        )
    } else {
        (
            fast_cubecl_common::legal_cube_dim(client, KD_CUBE_DIM),
            hardware.max_cube_count.0,
        )
    };
    assert!(max_cubes > 0, "device reports no legal KD launch capacity");
    (
        CubeCount::Static(launch_count(rows, dim.x, max_cubes), 1, 1),
        dim,
    )
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
        u32::try_from(n).expect("validated KD row count"),
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
    let diagnostics_enabled = kd_diagnostics_enabled();
    let diagnostics = allocate_diagnostics(&data, n, diagnostics_enabled);
    let client = data.client.clone();
    let (cube_count, dim) = kd_launch_geometry(&client, n);
    kd_neighbors_kernel::launch(
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
    if diagnostics_enabled {
        emit_kd_diagnostics(
            "kd_neighbors",
            diagnostics,
            n,
            tree.starts.shape().dims::<1>()[0],
        );
    }
    (output_indices, output_distances)
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

fn allocate_int(data: &FloatTensor<CubeBackend>, rows: usize) -> IntTensor<CubeBackend> {
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

    #[cfg(feature = "cpu")]
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
        let launch = |diagnostics_enabled,
                      output_indices: &CubeTensor,
                      output_distances: &CubeTensor,
                      diagnostics: &CubeTensor| {
            kd_neighbors_kernel::launch(
                &client,
                cube_count.clone(),
                dim,
                input.clone().into_tensor_arg(),
                indices.clone().into_tensor_arg(),
                starts.clone().into_tensor_arg(),
                ends.clone().into_tensor_arg(),
                leaves.clone().into_tensor_arg(),
                bounds_min.clone().into_tensor_arg(),
                bounds_max.clone().into_tensor_arg(),
                rows as u32,
                k as u32,
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
        for pair in counts.chunks_exact(2) {
            assert!(pair[0] > 0 && pair[0] as usize <= host_tree.starts.len());
            assert!(pair[1] >= 0 && pair[1] as usize <= rows);
        }
    }

    #[cfg(feature = "cpu")]
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

    #[cfg(feature = "cpu")]
    #[test]
    fn box_lower_bound_matches_point_box_oracle_cpu() {
        verify_box_bounds(&CubeDevice::Cpu(Default::default()));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn box_lower_bound_matches_point_box_oracle_cuda() {
        let cuda = CubeDevice::enumerate_all()
            .into_iter()
            .find(|device| matches!(device, CubeDevice::Cuda(_)));
        let Some(cuda) = cuda else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        };
        verify_box_bounds(&cuda);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn required_cuda_box_bounds_fail_when_cuda_is_not_compiled() {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
        );
    }
}
