#![allow(unused_variables)]
use burn::backend::{
    Backend, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

#[backend_extension(Cube, Fusion)]
pub trait NeighborOps: Backend {
    #[fusion(meta = |data, k, seed, max_iterations, squared| {
        assert!(data.shape.len() == 2 && data.shape[0] >= 2 && data.shape[1] > 0,
            "ANN input requires at least two rows and one feature");
        assert!(*k > 0 && (*k as usize) < data.shape[0],
            "ANN neighbor width must be in 1..N");
        let shape = [data.shape[0], *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), burn::tensor::DType::I32),
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), data.dtype),
        )
    })]
    /// Bounded NN-descent nonself neighbors, sorted by `(distance, index)`.
    /// Call `ann_preflight` before dispatch. Squared mode skips the output square root.
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
        squared: bool,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |query, reference, k, exclude_self, query_offset, squared| {
        let shape = [query.shape[0], *k as usize];
        (burn::backend::fusion::custom::TensorSpec::new(shape.into(), burn::tensor::DType::I32), burn::backend::fusion::custom::TensorSpec::new(shape.into(), query.dtype))
    })]
    fn exact_neighbors(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
        squared: bool,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(dtype = query, shape = { let mut shape = query.clone(); shape[1] = reference[0]; shape })]
    fn squared_euclidean_distance_block(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}
