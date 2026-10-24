use super::neighbors;
use crate::backend::NeighborOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;

impl NeighborOps for CubeBackend {
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
        squared: bool,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        super::ann::approximate_neighbors(data, k, seed, max_iterations, squared)
    }

    fn exact_neighbors(
        q: FloatTensor<Self>,
        r: FloatTensor<Self>,
        k: u32,
        exclude: bool,
        offset: u32,
        squared: bool,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        neighbors::exact_neighbors(q, r, k, exclude, offset, squared)
    }
    fn squared_euclidean_distance_block(
        q: FloatTensor<Self>,
        r: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        neighbors::squared_euclidean_distance_block(q, r)
    }
}
