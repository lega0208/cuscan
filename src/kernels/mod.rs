use burn::{
    backend::tensor::{FloatTensor, IntTensor},
    cubecl::CubeDim,
};
use burn_cubecl::CubeBackend;

use crate::backend::UmapOps;

mod euclidean;
mod knn;
#[cfg(feature = "cuda")]
mod standard_umap;

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

#[cfg(feature = "cuda")]
impl crate::backend::StandardUmapOps for CubeBackend {
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        standard_umap::approximate_neighbors(data, k, seed, max_iterations)
    }

    fn approximate_query(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        graph: IntTensor<Self>,
        k: u32,
        candidate_capacity: u32,
        visit_capacity: u32,
        seed: u64,
        query_offset: u64,
    ) -> (IntTensor<Self>, FloatTensor<Self>, IntTensor<Self>) {
        standard_umap::approximate_query(
            query,
            reference,
            graph,
            k,
            candidate_capacity,
            visit_capacity,
            seed,
            query_offset,
        )
    }

    fn fuzzy_graph(
        indices: IntTensor<Self>,
        distances: FloatTensor<Self>,
        local_connectivity: f32,
        bandwidth: f32,
        set_op_mix_ratio: f32,
        bipartite: bool,
        n_reference: u32,
    ) -> (
        IntTensor<Self>,
        IntTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
    ) {
        standard_umap::fuzzy_graph(
            indices,
            distances,
            local_connectivity,
            bandwidth,
            set_op_mix_ratio,
            bipartite,
            n_reference,
        )
    }

    fn exact_neighbors(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        standard_umap::exact_neighbors(query, reference, k, exclude_self, query_offset)
    }

    fn squared_euclidean_distance_block(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        standard_umap::squared_euclidean_distance_block(query, reference)
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_epoch(
        head_embedding: FloatTensor<Self>,
        tail_embedding: FloatTensor<Self>,
        heads: IntTensor<Self>,
        tails: IntTensor<Self>,
        weights: FloatTensor<Self>,
        epochs_per_sample: FloatTensor<Self>,
        epoch_of_next_sample: FloatTensor<Self>,
        epochs_per_negative_sample: FloatTensor<Self>,
        epoch_of_next_negative_sample: FloatTensor<Self>,
        rng_state: IntTensor<Self>,
        head_offsets: IntTensor<Self>,
        head_edges: IntTensor<Self>,
        epoch: u32,
        a: f32,
        b: f32,
        gamma: f32,
        alpha: f32,
        negative_sample_rate: f32,
        move_other: bool,
        shared_embeddings: bool,
    ) -> (
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        FloatTensor<Self>,
        IntTensor<Self>,
    ) {
        standard_umap::optimize_epoch(
            head_embedding,
            tail_embedding,
            heads,
            tails,
            weights,
            epochs_per_sample,
            epoch_of_next_sample,
            epochs_per_negative_sample,
            epoch_of_next_negative_sample,
            rng_state,
            head_offsets,
            head_edges,
            epoch,
            a,
            b,
            gamma,
            alpha,
            negative_sample_rate,
            move_other,
            shared_embeddings,
        )
    }
}

// The autodiff implementation is handwritten in src/autodiff_ops.rs.
