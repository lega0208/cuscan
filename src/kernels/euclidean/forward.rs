use super::kernel::*;
use crate::kernels::{CubeBackend, DEFAULT_CUBE_DIM};
use burn::{
    backend::{TensorMetadata, cubecl::dtype_to_storage_type, tensor::FloatTensor},
    cubecl::prelude::*,
    tensor::Shape,
};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor};

/// Compute the `[n, n]` Euclidean pairwise distance matrix on CubeCL.
pub fn forward(x: FloatTensor<CubeBackend>) -> FloatTensor<CubeBackend> {
    let x = into_contiguous(x);
    let n = x.shape().dims::<2>()[0];
    let output_shape = Shape::new([n, n]);
    let output = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        output_shape,
        x.client.empty(n * n * x.dtype.size()),
        x.dtype,
    );

    let cube_dim = DEFAULT_CUBE_DIM;
    let cube_count = CubeCount::Static(
        (n as u32).div_ceil(cube_dim.x),
        (n as u32).div_ceil(cube_dim.y),
        1,
    );
    let client = x.client.clone();
    euclidean_pairwise_distance_kernel::launch(
        &client,
        cube_count,
        cube_dim,
        x.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        dtype_to_storage_type(output.dtype),
    );
    output
}

/// Propagate pairwise-distance gradients to the input vectors on CubeCL.
pub fn backward(
    grad_pairwise: FloatTensor<CubeBackend>,
    x: FloatTensor<CubeBackend>,
    pairwise: FloatTensor<CubeBackend>,
) -> FloatTensor<CubeBackend> {
    let x = into_contiguous(x);
    let pairwise = into_contiguous(pairwise);
    let grad_pairwise = into_contiguous(grad_pairwise);
    let dims = x.shape().dims::<2>();
    assert_eq!(dims.len(), 2, "Euclidean backward expects a rank-2 input");
    let n = dims[0];
    let d = dims[1];
    let grad_x = CubeTensor::new_contiguous(
        x.client.clone(),
        x.device.clone(),
        Shape::new([n, d]),
        x.client.empty(n * d * x.dtype.size()),
        x.dtype,
    );

    let cube_dim = DEFAULT_CUBE_DIM;
    let cube_count = CubeCount::Static(
        (n as u32).div_ceil(cube_dim.x),
        (d as u32).div_ceil(cube_dim.y),
        1,
    );
    euclidean_pairwise_distance_backward_kernel::launch(
        &x.client,
        cube_count,
        cube_dim,
        x.clone().into_tensor_arg(),
        pairwise.into_tensor_arg(),
        grad_pairwise.into_tensor_arg(),
        grad_x.clone().into_tensor_arg(),
        dtype_to_storage_type(grad_x.dtype),
    );
    grad_x
}
