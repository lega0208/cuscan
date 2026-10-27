#![allow(unused_variables)]

use burn::backend::{
    Backend, ExtensionType, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

/// Uploaded KD-tree buffers consumed by exact traversal operations.
///
/// Callers must supply a valid tree built by `fast_hdbscan_core::kdtree::KdTree::build`
/// for the same finite data, uploaded unchanged, or an equivalent valid tree/data
/// pair. Indices must be exactly a permutation of `0..n`; reachable leaf ranges
/// must partition `0..n` without gaps or overlap. Reachable heap children must
/// exist and partition their parent's range, terminate at leaves within the
/// builder's level/stack limits, and have bounds enclosing their assigned points
/// and child bounds. Data and tree contents must remain unchanged together during
/// traversal. These content preconditions are not checked by runtime scans or
/// readbacks; shape/dtype validation and checked sizing do not establish them.
#[derive(ExtensionType)]
#[extension_type(fusion)]
pub struct DeviceKdTree<B: Backend> {
    pub indices: IntTensor<B>,
    pub starts: IntTensor<B>,
    pub ends: IntTensor<B>,
    pub is_leaf: IntTensor<B>,
    pub bounds_min: FloatTensor<B>,
    pub bounds_max: FloatTensor<B>,
    pub node_components: IntTensor<B>,
    pub n_levels: usize,
    pub n_features: usize,
}

/// Backend operation used to find each point's exact outgoing Borůvka edge.
#[backend_extension(Cube, Fusion)]
pub trait HdbscanOps: Backend {
    #[fusion(meta = |data, core_sq, point_components| {
        let shape = [data.shape[0]];
        (
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), data.dtype),
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
        )
    })]
    /// Returns each point's lexicographically smallest `(MRD², destination)`
    /// among points in other components. Missing destinations use sentinel `n`.
    fn boruvka_candidates(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
    ) -> (FloatTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |_weights, _indices, _members, _tile_starts, _tile_ends, _partial_offsets, component_count| {
        let c = *component_count as usize;
        (
            burn::backend::fusion::custom::TensorSpec::new([c].into(), burn::tensor::DType::F32),
            burn::backend::fusion::custom::TensorSpec::new([c, 5].into(), burn::tensor::DType::I32),
        )
    })]
    fn component_reduce(
        candidate_weights: FloatTensor<Self>,
        candidate_indices: IntTensor<Self>,
        members: IntTensor<Self>,
        tile_starts: IntTensor<Self>,
        tile_ends: IntTensor<Self>,
        partial_offsets: IntTensor<Self>,
        component_count: u32,
    ) -> (FloatTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |data, tree, k| {
        let shape = [data.shape[0], *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), data.dtype),
        )
    })]
    /// Exact squared-distance neighbors, ordered by `(distance², point index)`.
    /// Self is excluded and absent neighbors are padded with `(inf, n)`.
    ///
    /// Requires finite `data` and an unchanged builder-produced (or equivalent
    /// valid) tree for that same data, satisfying the permutation, partition,
    /// topology, bounds and level/stack preconditions on [`DeviceKdTree`]. No
    /// runtime tree-content validation is performed. Estimator calls meet this
    /// contract after finite-row compaction; nonfinite-row handling and original
    /// row remapping belong to the estimator, not this operation.
    fn exact_neighbors_kd(
        data: FloatTensor<Self>,
        #[extension_type] tree: DeviceKdTree<Self>,
        k: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |data, core_sq, point_components, tree, reproducible| {
        let shape = [data.shape[0]];
        (
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), data.dtype),
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
        )
    })]
    /// MRD² candidates preserving component winners. Reproducible mode returns
    /// every row's exact candidate. Shared mode preserves only first-source
    /// component winners; other rows may return heavier, scheduling-dependent
    /// candidates or be pruned to `(inf, -1)`.
    fn boruvka_candidates_kd(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
        #[extension_type] tree: DeviceKdTree<Self>,
        reproducible: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>);
}
