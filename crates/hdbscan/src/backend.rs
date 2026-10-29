#![allow(unused_variables)]

use burn::backend::{
    Backend, ExtensionType, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

/// Uploaded KD-tree buffers consumed by exact traversal operations.
///
/// Callers must supply a valid tree built by `hdbscan_core::kdtree::KdTree::build`
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

/// Call-local prepared KD reference reused by query/reference searches.
///
/// Built once by [`HdbscanOps::prepare_kd_reference`] from compact reference
/// rows and their tree; every bounded query batch clones this value instead of
/// repacking. `candidates` holds padded rows when `lanes == 0`, otherwise
/// lane-tiled leaf SoA data; `slot_ids` holds the corresponding IDs.
/// `padded_min`/`padded_max` hold padded bounds.
/// The former `padded_data` field is replaced by `candidates`, `slot_ids` and
/// `lanes`; this is a deliberate Rust API break with no compatibility shim.
/// The tree preconditions of [`DeviceKdTree`] apply to the source data/tree.
/// Drop the value at the end of the call to release its device allocations.
#[derive(Clone, ExtensionType)]
#[extension_type(fusion)]
pub struct DeviceKdReference<B: Backend> {
    pub candidates: FloatTensor<B>,
    pub slot_ids: IntTensor<B>,
    pub lanes: usize,
    pub padded_min: FloatTensor<B>,
    pub padded_max: FloatTensor<B>,
    pub indices: IntTensor<B>,
    pub starts: IntTensor<B>,
    pub ends: IntTensor<B>,
    pub is_leaf: IntTensor<B>,
    pub rows: usize,
    pub nodes: usize,
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

    #[fusion(meta = |data, tree, lanes| {
        let rows = data.shape[0];
        let features = data.shape[1];
        let nodes = tree.starts.shape[0];
        let width = features.div_ceil(4) * 4;
        let lanes = *lanes as usize;
        let slots = if lanes == 0 { rows } else { rows.div_ceil(lanes) * lanes };
        let float = |len: usize| {
            burn::backend::fusion::custom::TensorSpec::new([len].into(), data.dtype)
        };
        DeviceKdReferenceMetadata {
            candidates: float(if lanes == 0 { rows * width } else { slots * features }),
            slot_ids: if lanes == 0 { tree.indices.clone() } else {
                burn::backend::fusion::custom::TensorSpec::new([slots].into(), burn::tensor::DType::I32)
            },
            lanes,
            padded_min: float(nodes * width),
            padded_max: float(nodes * width),
            indices: tree.indices.clone(),
            starts: tree.starts.clone(),
            ends: tree.ends.clone(),
            is_leaf: tree.is_leaf.clone(),
            rows,
            nodes,
            n_levels: tree.n_levels,
            n_features: features,
        }
    })]
    /// Validates shapes/dtypes and packs compact reference rows and tree bounds
    /// once for repeated [`HdbscanOps::exact_neighbors_kd_query`] batches. The
    /// tree-content preconditions on [`DeviceKdTree`] are not checked.
    /// `lanes = 0` selects padded rows on any device; a nonzero width requires a CPU client
    /// and a supported `io_optimized_vector_sizes(4)` width no greater than 16.
    /// Violating these layout preconditions panics. The explicit `lanes` argument
    /// is a deliberate Rust API break; Python's prediction API is unchanged.
    fn prepare_kd_reference(
        data: FloatTensor<Self>,
        #[extension_type] tree: DeviceKdTree<Self>,
        lanes: u32,
    ) -> DeviceKdReference<Self>;

    #[fusion(meta = |query, reference, k, exclude_self, query_offset| {
        let shape = [query.shape[0], *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), query.dtype),
        )
    })]
    /// Exact squared-distance neighbors of each `query` row among the prepared
    /// reference rows, ordered by `(distance², reference ID)` and returned in
    /// query row order with shape `[query rows, k]`. Absent slots are padded
    /// with `(inf, reference rows)`. With `exclude_self`, query row `r` skips
    /// reference ID `query_offset + r`; identical coordinates are never
    /// excluded by value. CPU prediction selects bounds-pruned traversal over leaf SoA;
    /// references prepared with `lanes = 0` use padded rows on any device.
    /// Equal bounds are still visited. Results
    /// are exact under the kernel's per-feature arithmetic; floating-point
    /// contraction can change rounding and near-tie IDs across layouts, affecting
    /// downstream labels, probabilities and connectors. Exactly representable
    /// quantized squared sums agree bitwise; arbitrary inputs need not agree with
    /// the previous CPU release or with another backend.
    fn exact_neighbors_kd_query(
        query: FloatTensor<Self>,
        #[extension_type] reference: DeviceKdReference<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
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
