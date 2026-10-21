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
    levels: u32,
    row_offset: u32,
    row_count: u32,
    out_weights: &mut Tensor<F>,
    out_indices: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let local_row = ABSOLUTE_POS_X as usize;
    let row = local_row + row_offset as usize;
    if local_row < row_count as usize && row < data.shape(0) {
        let n = data.shape(0);
        let component = components[row];
        let core = core_sq[row];
        let mut best = F::new(f32::INFINITY);
        let mut best_index = n;
        let mut stack = Array::<u32>::new(KD_STACK_LEN);
        stack[0] = 0;
        let mut stack_len = 1usize;
        while stack_len > 0 {
            stack_len -= 1;
            let node = stack[stack_len] as usize;
            let lower = F::max(
                box_lower_bound(data, bounds_min, bounds_max, row, node),
                core,
            );
            if lower <= best && node_components[node] != component {
                if is_leaf[node] != I::cast_from(0) {
                    let start = u32::cast_from(starts[node]) as usize;
                    let end = u32::cast_from(ends[node]) as usize;
                    for position in start..end {
                        let candidate = u32::cast_from(indices[position]) as usize;
                        if components[candidate] != component {
                            let d = data.shape(1);
                            let mut distance = F::new(0.0f32);
                            for feature in 0..d {
                                let delta = data[row * d + feature] - data[candidate * d + feature];
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
                        stack[stack_len] = right as u32;
                        stack[stack_len + 1] = left as u32;
                    } else {
                        stack[stack_len] = left as u32;
                        stack[stack_len + 1] = right as u32;
                    }
                    stack_len += 2;
                }
            }
        }
        out_weights[row] = best;
        out_indices[row] = I::cast_from(best_index);
        let _ = levels;
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
    levels: u32,
    row_offset: u32,
    row_count: u32,
    k: u32,
    out_indices: &mut Tensor<I>,
    out_distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let local_row = ABSOLUTE_POS_X as usize;
    let row = local_row + row_offset as usize;
    if local_row < row_count as usize && row < data.shape(0) {
        let n = data.shape(0);
        let width = k as usize;
        for slot in 0..width {
            out_indices[row * width + slot] = I::cast_from(n);
            out_distances[row * width + slot] = F::new(f32::INFINITY);
        }
        let mut stack = Array::<u32>::new(KD_STACK_LEN);
        stack[0] = 0;
        let mut stack_len = 1usize;
        while stack_len > 0 {
            stack_len -= 1;
            let node = stack[stack_len] as usize;
            let worst = out_distances[row * width];
            if box_lower_bound(data, bounds_min, bounds_max, row, node) <= worst {
                if is_leaf[node] != I::cast_from(0) {
                    let start = u32::cast_from(starts[node]) as usize;
                    let end = u32::cast_from(ends[node]) as usize;
                    for position in start..end {
                        let candidate = u32::cast_from(indices[position]) as usize;
                        if candidate != row {
                            let d = data.shape(1);
                            let mut distance = F::new(0.0f32);
                            for feature in 0..d {
                                let delta = data[row * d + feature] - data[candidate * d + feature];
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
                                            || (out_distances[right_at] == out_distances[left_at]
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
                        stack[stack_len] = right as u32;
                        stack[stack_len + 1] = left as u32;
                    } else {
                        stack[stack_len] = left as u32;
                        stack[stack_len + 1] = right as u32;
                    }
                    stack_len += 2;
                }
            }
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
        let _ = levels;
    }
}

fn launch_count(rows: usize, dim: u32, max_cubes: u32) -> u32 {
    u32::try_from(rows.div_ceil(dim as usize).min(max_cubes as usize))
        .expect("bounded KD launch count")
}

fn validate_tree(data: &FloatTensor<CubeBackend>, tree: &DeviceKdTree<CubeBackend>) {
    let [rows, features] = data.shape().dims::<2>();
    let [index_count] = tree.indices.shape().dims::<1>();
    let [nodes] = tree.starts.shape().dims::<1>();
    assert!(rows > 0 && features > 0, "KD input data must be non-empty");
    i32::try_from(rows).expect("KD row count must fit destination indices");
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
    let weights = allocate_float(&data, n);
    let destinations = allocate_int(&data, n);
    let client = data.client.clone();
    let dim = fast_cubecl_common::legal_cube_dim(&client, KD_CUBE_DIM);
    let max_cubes = client.properties().hardware.max_cube_count.0;
    assert!(max_cubes > 0, "device reports no legal KD launch capacity");
    let rows_per_launch = (dim.x as usize)
        .checked_mul(max_cubes as usize)
        .expect("KD launch capacity fits usize");
    for row_offset in (0..n).step_by(rows_per_launch) {
        let rows = rows_per_launch.min(n - row_offset);
        kd_candidates_kernel::launch(
            &client,
            CubeCount::Static(launch_count(rows, dim.x, max_cubes), 1, 1),
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
            u32::try_from(tree.n_levels).expect("validated KD tree level count"),
            u32::try_from(row_offset).expect("validated KD row offset"),
            u32::try_from(rows).expect("validated KD launch row count"),
            weights.clone().into_tensor_arg(),
            destinations.clone().into_tensor_arg(),
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
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
    let client = data.client.clone();
    let dim = fast_cubecl_common::legal_cube_dim(&client, KD_CUBE_DIM);
    let max_cubes = client.properties().hardware.max_cube_count.0;
    assert!(max_cubes > 0, "device reports no legal KD launch capacity");
    let rows_per_launch = (dim.x as usize)
        .checked_mul(max_cubes as usize)
        .expect("KD launch capacity fits usize");
    for row_offset in (0..n).step_by(rows_per_launch) {
        let rows = rows_per_launch.min(n - row_offset);
        kd_neighbors_kernel::launch(
            &client,
            CubeCount::Static(launch_count(rows, dim.x, max_cubes), 1, 1),
            dim,
            data.clone().into_tensor_arg(),
            tree.indices.clone().into_tensor_arg(),
            tree.starts.clone().into_tensor_arg(),
            tree.ends.clone().into_tensor_arg(),
            tree.is_leaf.clone().into_tensor_arg(),
            tree.bounds_min.clone().into_tensor_arg(),
            tree.bounds_max.clone().into_tensor_arg(),
            u32::try_from(tree.n_levels).expect("validated KD tree level count"),
            u32::try_from(row_offset).expect("validated KD row offset"),
            u32::try_from(rows).expect("validated KD launch row count"),
            k,
            output_indices.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
    }
    (output_indices, output_distances)
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
    use burn::backend::ops::FloatTensorOps;
    use burn::tensor::TensorData;
    use burn_cubecl::{CubeDevice, ops::into_data_sync};

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
