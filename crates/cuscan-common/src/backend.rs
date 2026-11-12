#![allow(unused_variables)]
use burn::backend::{
    Backend, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

#[backend_extension(Cube, Fusion)]
pub trait NeighborOps: Backend {
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
