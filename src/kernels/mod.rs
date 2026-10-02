use burn::backend::tensor::{FloatTensor, IntTensor};
use cubecl::CubeDim;

use crate::backend::UmapOps;
use burn_cubecl::CubeBackend;

mod euclidean;
mod knn;

/// Default work-group (cube) dimensions used by all custom CubeCL kernels.
///
/// A 32 × 32 × 1 cube means up to 1 024 threads share a work-group.  This
/// matches the typical warp/wavefront size on WGPU targets (Metal, Vulkan,
/// DX12) and maximises occupancy for the matrix-shaped dispatch patterns used
/// by the Euclidean and k-NN kernels.
pub const DEFAULT_CUBE_DIM: CubeDim = CubeDim::new_2d(32, 32);

/// GPU implementation of the fast-umap [`UmapOps`] extension for [`CubeBackend`].
///
/// Each method dispatches to the corresponding CubeCL kernel in the
/// [`euclidean`] or [`knn`] sub-modules.
impl UmapOps for CubeBackend {
    /// Launch the Euclidean pairwise distance forward kernel.
    ///
    /// Computes the full `[n, n]` symmetric distance matrix for an `[n, d]`
    /// input using one GPU thread per `(row, col)` upper-triangle pair.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        euclidean::forward::forward(x)
    }

    /// Launch the Euclidean pairwise distance backward kernel.
    ///
    /// Dispatches one GPU thread per `(row, feature)` element of `grad_x`.
    /// Each thread iterates over all `n` columns to accumulate gradient
    /// contributions from both symmetric entries `pairwise[row, col]` and
    /// `pairwise[col, row]`.
    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        euclidean::forward::backward(grad_pairwise, x, pairwise)
    }

    /// Launch the k-NN forward kernel.
    ///
    /// Selects the `k` nearest neighbours for every row using an
    /// insertion-sort approach — one GPU thread per row.
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        knn::forward::forward(pairwise_distances, k)
    }

    /// Launch the k-NN backward kernel.
    ///
    /// Re-runs the forward sort to recover the selected neighbours, then
    /// propagates `grad_output` (shape `[n, k]`) back to
    /// `grad_pairwise_distances` (shape `[n, n]`).
    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        knn::forward::backward(pairwise_distances, k, grad_output)
    }
}

// The autodiff implementation is handwritten in src/autodiff_ops.rs.
