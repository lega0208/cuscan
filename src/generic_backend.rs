//! Flex implementations of fast-umap operations using standard Burn tensor ops.
//!
//! These operations use the dynamic tensor facade to retain the Flex device
//! and runtime dispatch while avoiding backend-generic conversion bounds.

use burn::backend::{
    tensor::{FloatTensor, IntTensor},
    Flex,
};
use burn::tensor::{DType, Int, Tensor, TensorData};

/// Compute the full `[n, n]` Euclidean pairwise distance matrix.
pub fn euclidean_pairwise_distance(x: FloatTensor<Flex>) -> FloatTensor<Flex> {
    let tensor: Tensor<2> = Tensor::from_primitive::<Flex>(x);
    let [n, _d] = tensor.dims();

    let sq_norms = tensor.clone().powf_scalar(2.0).sum_dim(1).reshape([n, 1]);
    let cross = tensor.clone().matmul(tensor.transpose());
    let neg2_cross = cross.mul_scalar(-2.0f32);
    let sq_norms_t = sq_norms.clone().reshape([1, n]);
    let sq_dists = neg2_cross.add(sq_norms).add(sq_norms_t);
    let dists = sq_dists.clamp_min(0.0f32).sqrt();

    dists
        .try_into_primitive::<Flex>()
        .expect("pairwise distance output must remain on Flex")
}

/// Backward pass for Euclidean pairwise distances.
pub fn euclidean_pairwise_distance_backward(
    grad_pairwise: FloatTensor<Flex>,
    x: FloatTensor<Flex>,
    pairwise: FloatTensor<Flex>,
) -> FloatTensor<Flex> {
    let x_t: Tensor<2> = Tensor::from_primitive::<Flex>(x);
    let grad_t: Tensor<2> = Tensor::from_primitive::<Flex>(grad_pairwise);
    let pw_t: Tensor<2> = Tensor::from_primitive::<Flex>(pairwise);
    let [n, _d] = x_t.dims();

    let sym_grad = grad_t.clone().add(grad_t.transpose());
    let inv_dist = pw_t.clamp_min(1e-8f32).recip();
    let scale = sym_grad.mul(inv_dist);
    let grad_x = scale.clone().sum_dim(1).reshape([n, 1]).mul(x_t.clone()) - scale.matmul(x_t);

    grad_x
        .try_into_primitive::<Flex>()
        .expect("pairwise distance gradient must remain on Flex")
}

/// Select the `k` nearest neighbours per row using the shared CPU selector.
pub fn knn(pairwise_distances: FloatTensor<Flex>, k: u32) -> (IntTensor<Flex>, FloatTensor<Flex>) {
    let pw: Tensor<2> = Tensor::from_primitive::<Flex>(pairwise_distances);
    let [n, _n2] = pw.dims();
    let device = pw.device();
    let dtype = pw.dtype();
    let k = k as usize;
    let flat = pw.to_data_dtype(DType::F64).try_to_vec::<f64>().unwrap();
    let (idx_flat, dist_flat) =
        crate::graph::get_distance_by_metric::knn_from_pairwise_cpu(&flat, n, k);

    let indices: Tensor<2, Int> = Tensor::from_data(TensorData::new(idx_flat, [n, k]), &device);
    let distances: Tensor<2> =
        Tensor::from_data(TensorData::new(dist_flat, [n, k]), (&device, dtype));

    (
        indices
            .try_into_primitive::<Flex>()
            .expect("k-NN indices must remain on Flex"),
        distances
            .try_into_primitive::<Flex>()
            .expect("k-NN distances must remain on Flex"),
    )
}

/// Scatter selected neighbor-distance gradients into the pairwise gradient.
pub fn knn_backward(
    pairwise_distances: FloatTensor<Flex>,
    k: u32,
    grad_output: FloatTensor<Flex>,
) -> FloatTensor<Flex> {
    let pairwise: Tensor<2> = Tensor::from_primitive::<Flex>(pairwise_distances);
    let output_grad: Tensor<2> = Tensor::from_primitive::<Flex>(grad_output);
    let [n, _] = pairwise.dims();
    let device = pairwise.device();
    let pairwise_dtype = pairwise.dtype();
    let grad_dtype = output_grad.dtype();
    assert_eq!(
        pairwise_dtype, grad_dtype,
        "k-NN output gradient dtype must match pairwise distances"
    );
    let flat = pairwise
        .to_data_dtype(DType::F64)
        .try_to_vec::<f64>()
        .unwrap();
    let grad_values = output_grad
        .to_data_dtype(DType::F64)
        .try_to_vec::<f64>()
        .unwrap();
    let k = k as usize;
    let (indices, _) = crate::graph::get_distance_by_metric::knn_from_pairwise_cpu(&flat, n, k);

    let mut grad_pairwise = vec![0.0f64; n * n];
    for row in 0..n {
        for neighbor in 0..k {
            let col = indices[row * k + neighbor] as usize;
            grad_pairwise[row * n + col] += grad_values[row * k + neighbor];
        }
    }

    Tensor::<2>::from_data(
        TensorData::new(grad_pairwise, [n, n]),
        (&device, pairwise_dtype),
    )
    .try_into_primitive::<Flex>()
    .expect("k-NN gradient must remain on Flex")
}
