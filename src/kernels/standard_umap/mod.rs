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

pub(super) mod ann;
pub(super) mod graph;
pub(super) mod layout;
pub(super) mod query;

pub(super) use ann::approximate_neighbors;
pub(super) use graph::fuzzy_graph;
pub(super) use layout::optimize_epoch;
pub(super) use query::approximate_query;

const DISTANCE_TILE: CubeDim = CubeDim::new_2d(16, 16);
const NEIGHBOR_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

#[cube(launch)]
fn exact_neighbors_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    k: u32,
    exclude_self: u32,
    query_offset: u32,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let query_rows = query.shape(0);
    let reference_rows = reference.shape(0);
    let features = query.shape(1);
    let k_usize = k as usize;
    if row < query_rows {
        for slot in 0..k_usize {
            indices[row * k_usize + slot] = I::cast_from(reference_rows);
            distances[row * k_usize + slot] = F::new(3.402_823_5e38_f32);
        }

        for reference_row in 0..reference_rows {
            if exclude_self == 0 || reference_row != (query_offset as usize + row) {
                let mut squared_distance = F::new(0.0f32);
                for feature in 0..features {
                    let difference = query[row * features + feature]
                        - reference[reference_row * features + feature];
                    squared_distance += difference * difference;
                }
                let worst = row * k_usize + k_usize - 1;
                let current_worst_index = u32::cast_from(indices[worst]) as usize;
                if squared_distance < distances[worst]
                    || (squared_distance == distances[worst] && reference_row < current_worst_index)
                {
                    let mut slot = k_usize - 1;
                    while slot > 0 {
                        let previous = row * k_usize + slot - 1;
                        let previous_index = u32::cast_from(indices[previous]) as usize;
                        if squared_distance < distances[previous]
                            || (squared_distance == distances[previous]
                                && reference_row < previous_index)
                        {
                            indices[row * k_usize + slot] = indices[previous];
                            distances[row * k_usize + slot] = distances[previous];
                            slot -= 1;
                        } else {
                            break;
                        }
                    }
                    indices[row * k_usize + slot] = I::cast_from(reference_row);
                    distances[row * k_usize + slot] = squared_distance;
                }
            }
        }

        for slot in 0..k_usize {
            let output = row * k_usize + slot;
            distances[output] = F::sqrt(distances[output]);
        }
    }
}

#[cube(launch)]
fn squared_euclidean_distance_block_kernel<F: Float>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let query_row = ABSOLUTE_POS_X as usize;
    let reference_row = ABSOLUTE_POS_Y as usize;
    let query_rows = query.shape(0);
    let reference_rows = reference.shape(0);
    let features = query.shape(1);

    if query_row < query_rows && reference_row < reference_rows {
        let mut squared_distance = F::new(0.0f32);
        for feature in 0..features {
            let difference = query[query_row * features + feature]
                - reference[reference_row * features + feature];
            squared_distance += difference * difference;
        }
        output[query_row * reference_rows + reference_row] = squared_distance;
    }
}

pub(super) fn exact_neighbors(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [query_rows, query_features] = query.shape().dims::<2>();
    let [reference_rows, reference_features] = reference.shape().dims::<2>();
    assert!(
        query_rows > 0 && reference_rows > 0,
        "neighbor inputs must be non-empty"
    );
    assert!(
        query_features > 0,
        "neighbor feature dimension must be non-empty"
    );
    assert_eq!(
        query_features, reference_features,
        "query and reference feature dimensions must match"
    );
    assert_eq!(
        query.dtype, reference.dtype,
        "query and reference dtype must match"
    );
    assert!(k > 0, "neighbor count must be positive");
    let query_rows_u32 = u32::try_from(query_rows).expect("query row count exceeds u32");
    let reference_rows_u32 =
        u32::try_from(reference_rows).expect("reference row count exceeds u32");
    i32::try_from(reference_rows).expect("reference indices exceed i32");
    let query_end = query_offset
        .checked_add(query_rows_u32)
        .expect("query offset plus block rows exceeds u32");
    if exclude_self {
        assert!(
            query_end <= reference_rows_u32,
            "self-excluded query indices must be valid reference indices"
        );
    }
    assert!(
        k <= reference_rows_u32 - exclude_self as u32,
        "neighbor count exceeds available references"
    );
    let output_elements = query_rows
        .checked_mul(k as usize)
        .expect("neighbor output dimensions overflow usize");
    let index_bytes = output_elements
        .checked_mul(DType::I32.size())
        .expect("neighbor index allocation overflows usize");
    let distance_bytes = output_elements
        .checked_mul(query.dtype.size())
        .expect("neighbor distance allocation overflows usize");
    let query_rows_u32 = u32::try_from(query_rows).expect("query row count exceeds u32");
    let indices = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, k as usize]),
        query.client.empty(index_bytes),
        DType::I32,
    );
    let distances = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, k as usize]),
        query.client.empty(distance_bytes),
        query.dtype,
    );
    let client = query.client.clone();
    exact_neighbors_kernel::launch(
        &client,
        CubeCount::Static(query_rows_u32.div_ceil(NEIGHBOR_CUBE_DIM.x), 1, 1),
        NEIGHBOR_CUBE_DIM,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        k,
        exclude_self as u32,
        query_offset,
        indices.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(distances.dtype),
            dtype_to_storage_type(indices.dtype),
        ],
    );
    (indices, distances)
}

/// Compute one explicitly bounded query-by-reference squared-distance tile.
pub(super) fn squared_euclidean_distance_block(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
) -> FloatTensor<CubeBackend> {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [query_rows, query_features] = query.shape().dims::<2>();
    let [reference_rows, reference_features] = reference.shape().dims::<2>();
    assert!(
        query_rows > 0 && reference_rows > 0,
        "distance blocks must be non-empty"
    );
    assert!(
        query_features > 0,
        "distance feature dimension must be non-empty"
    );
    assert_eq!(
        query_features, reference_features,
        "query and reference feature dimensions must match"
    );
    let output_elements = query_rows
        .checked_mul(reference_rows)
        .expect("squared-distance output dimensions overflow usize");
    let output_bytes = output_elements
        .checked_mul(query.dtype.size())
        .expect("squared-distance output allocation overflows usize");
    let query_rows_u32 = u32::try_from(query_rows).expect("query block row count exceeds u32");
    let reference_rows_u32 =
        u32::try_from(reference_rows).expect("reference row count exceeds u32");
    let output = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, reference_rows]),
        query.client.empty(output_bytes),
        query.dtype,
    );
    let client = query.client.clone();
    let cube_count = CubeCount::Static(
        query_rows_u32.div_ceil(DISTANCE_TILE.x),
        reference_rows_u32.div_ceil(DISTANCE_TILE.y),
        1,
    );
    squared_euclidean_distance_block_kernel::launch(
        &client,
        cube_count,
        DISTANCE_TILE,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        dtype_to_storage_type(output.dtype),
    );
    output
}
