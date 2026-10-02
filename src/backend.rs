use burn::backend::{
    backend_extension,
    tensor::{FloatTensor, IntTensor},
    Backend,
};

/// Backend operations used by fast-umap and routed through Burn's dynamic
/// backend dispatch.
///
/// Two kernels are exposed:
///
/// * **Euclidean pairwise distance** — computes the full `[n, n]` symmetric
///   distance matrix for an `[n, d]` input; differentiable via a custom
///   backward kernel.
/// * **k-NN** — selects the `k` nearest neighbours for every row of a
///   pairwise distance matrix; differentiable (backward propagates through
///   the selected entries).
///
/// Implementations are provided by CubeCL, Flex, and the handwritten
/// autodiff adapter in `autodiff_ops`.
#[backend_extension(
    Cube: cfg(any(feature = "gpu", feature = "cuda")),
    Flex,
    Autodiff,
    Fusion: cfg(any(feature = "gpu", feature = "cuda")),
)]
pub trait UmapOps: Backend {
    #[fusion(dtype = x, shape = {
        let mut shape = x.clone();
        shape[1] = shape[0];
        shape
    })]
    /// Compute the full `[n, n]` symmetric Euclidean pairwise distance matrix
    /// for an input tensor of shape `[n, d]`.
    ///
    /// The result satisfies `output[i, i] = 0` and
    /// `output[i, j] = ||x[i] − x[j]||₂` for `i ≠ j`.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Backward pass for [`euclidean_pairwise_distance`](Self::euclidean_pairwise_distance).
    ///
    /// Computes `∂loss/∂x` given:
    ///
    /// * `grad_pairwise` — upstream gradient `∂loss/∂pairwise_distances`, shape `[n, n]`
    /// * `x`             — original input saved from the forward pass, shape `[n, d]`
    /// * `pairwise`      — precomputed distance matrix saved from the forward pass, shape `[n, n]`
    ///
    /// Returns `∂loss/∂x`, shape `[n, d]`.
    #[fusion(dtype = x, shape = x)]
    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |pairwise_distances, k| {
        let shape = [pairwise_distances.shape[0], *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                pairwise_distances.dtype,
            ),
        )
    })]
    /// Select the `k` nearest neighbours for every row of `pairwise_distances`.
    ///
    /// Returns `(indices, distances)`:
    /// * `indices`   — `[n, k]` integer tensor of neighbour column indices
    /// * `distances` — `[n, k]` float tensor of corresponding distances
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>);

    /// Backward pass for [`knn`](Self::knn).
    ///
    /// Re-runs the forward k-NN pass to recover which neighbours were selected,
    /// then propagates `grad_output` (shape `[n, k]`) back to a gradient with
    /// respect to `pairwise_distances` (shape `[n, n]`).
    ///
    /// The autodiff adapter registers this operation in Burn's gradient graph.
    #[fusion(dtype = pairwise_distances, shape = pairwise_distances)]
    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}
