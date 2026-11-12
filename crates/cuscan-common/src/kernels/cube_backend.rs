use super::neighbors;
use crate::backend::NeighborOps;
use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;

impl NeighborOps for CubeBackend {
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
