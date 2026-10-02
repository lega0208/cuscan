use super::kernel::*;
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

const KNN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

/// Select the `k` nearest neighbours for every row using the CubeCL kernel.
pub fn forward(
    pairwise_distances: FloatTensor<CubeBackend>,
    k: u32,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let pairwise_distances = into_contiguous(pairwise_distances);
    let n = pairwise_distances.shape().dims::<2>()[0];
    let device = pairwise_distances.device.clone();
    let client = pairwise_distances.client.clone();
    let int_dtype = DType::I32;
    let output_shape = Shape::new([n, k as usize]);
    let indices = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        output_shape.clone(),
        client.empty(output_shape.num_elements() * int_dtype.size()),
        int_dtype,
    );
    let distances = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        output_shape.clone(),
        client.empty(output_shape.num_elements() * pairwise_distances.dtype.size()),
        pairwise_distances.dtype,
    );
    let local_distances = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        output_shape.clone(),
        client.empty(output_shape.num_elements() * pairwise_distances.dtype.size()),
        pairwise_distances.dtype,
    );
    let local_indices = CubeTensor::new_contiguous(
        client.clone(),
        device,
        output_shape.clone(),
        client.empty(output_shape.num_elements() * int_dtype.size()),
        int_dtype,
    );

    let cube_dim = KNN_CUBE_DIM;
    let cube_count = CubeCount::Static((n as u32).div_ceil(cube_dim.x), 1, 1);
    knn_kernel::launch(
        &client,
        cube_count,
        cube_dim,
        pairwise_distances.clone().into_tensor_arg(),
        k,
        local_distances.into_tensor_arg(),
        local_indices.into_tensor_arg(),
        indices.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(pairwise_distances.dtype),
            dtype_to_storage_type(int_dtype),
        ],
    );
    (indices, distances)
}

/// Propagate gradients through the selected k-NN distances on CubeCL.
pub fn backward(
    pairwise_distances: FloatTensor<CubeBackend>,
    k: u32,
    grad_output: FloatTensor<CubeBackend>,
) -> FloatTensor<CubeBackend> {
    let pairwise_distances = into_contiguous(pairwise_distances);
    let grad_output = into_contiguous(grad_output);
    let n = pairwise_distances.shape().dims::<2>()[0];
    let int_dtype = DType::I32;
    let shape = pairwise_distances.shape().clone();
    let grad_pairwise_distances = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        shape.clone(),
        pairwise_distances.client.create_from_slice(&vec![
            0u8;
            shape.num_elements()
                * pairwise_distances.dtype.size()
        ]),
        pairwise_distances.dtype,
    );
    let scratch_shape = Shape::new([n, k as usize]);
    let local_distances = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        scratch_shape.clone(),
        pairwise_distances
            .client
            .empty(scratch_shape.num_elements() * pairwise_distances.dtype.size()),
        pairwise_distances.dtype,
    );
    let local_indices = CubeTensor::new_contiguous(
        pairwise_distances.client.clone(),
        pairwise_distances.device.clone(),
        scratch_shape.clone(),
        pairwise_distances
            .client
            .empty(scratch_shape.num_elements() * int_dtype.size()),
        int_dtype,
    );

    let cube_dim = KNN_CUBE_DIM;
    let cube_count = CubeCount::Static((n as u32).div_ceil(cube_dim.x), 1, 1);
    knn_backward_kernel::launch(
        &pairwise_distances.client,
        cube_count,
        cube_dim,
        pairwise_distances.clone().into_tensor_arg(),
        k,
        local_distances.into_tensor_arg(),
        local_indices.into_tensor_arg(),
        grad_output.into_tensor_arg(),
        grad_pairwise_distances.clone().into_tensor_arg(),
        [
            dtype_to_storage_type(pairwise_distances.dtype),
            dtype_to_storage_type(int_dtype),
        ],
    );
    grad_pairwise_distances
}
