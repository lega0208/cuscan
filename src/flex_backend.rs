//! Flex backend implementation for fast-umap.
//!
//! Implements the custom [`UmapOps`](crate::backend::UmapOps) extension for
//! Burn's CPU [`Flex`](burn::backend::Flex) backend by delegating to the
//! generic tensor-based implementations in [`generic_backend`](crate::generic_backend).

use burn::backend::tensor::{FloatTensor, IntTensor};

use crate::backend::UmapOps;

impl UmapOps for burn::backend::Flex {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance_backward(grad_pairwise, x, pairwise)
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        crate::generic_backend::knn(pairwise_distances, k)
    }

    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::knn_backward(pairwise_distances, k, grad_output)
    }
}
