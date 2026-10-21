#[cfg(any(feature = "cpu", feature = "cuda"))]
mod boruvka;
#[cfg(any(feature = "cpu", feature = "cuda"))]
mod kdtree;

#[cfg(any(feature = "cpu", feature = "cuda"))]
pub use boruvka::{BoruvkaCandidateSizing, validate_boruvka_candidate_sizing};

use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;

#[cfg(any(feature = "cpu", feature = "cuda"))]
impl crate::backend::HdbscanOps for CubeBackend {
    fn boruvka_candidates(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        boruvka::boruvka_candidates(data, core_sq, point_components)
    }

    fn exact_neighbors_kd(
        data: FloatTensor<Self>,
        tree: crate::backend::DeviceKdTree<Self>,
        k: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        kdtree::exact_neighbors_kd(data, tree, k)
    }

    fn boruvka_candidates_kd(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
        tree: crate::backend::DeviceKdTree<Self>,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        kdtree::boruvka_candidates_kd(data, core_sq, point_components, tree)
    }
}
