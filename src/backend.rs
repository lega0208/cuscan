use burn::backend::{
    Backend, backend_extension,
    tensor::{FloatTensor, IntTensor},
};

/// Backend operations used by fast-umap and routed through Burn's dynamic
/// backend dispatch.
///
/// Two kernels are exposed:
///
/// * **Euclidean pairwise distance** — computes the full `[n, n]` symmetric
///   distance matrix for an `[n, d]` input; differentiable via a custom
///   backward kernel.
/// * **k-NN** — selects the `k` nearest neighbours for every row of a
///   pairwise distance matrix; differentiable (backward propagates through
///   the selected entries).
///
/// Implementations are provided by CubeCL and the handwritten
/// autodiff adapter in `autodiff_ops`.
#[backend_extension(
    Cube: cfg(any(feature = "cpu", feature = "gpu", feature = "cuda")),
    Autodiff,
    Fusion: cfg(any(feature = "cpu", feature = "gpu", feature = "cuda")),
)]
pub trait UmapOps: Backend {
    #[fusion(dtype = x, shape = {
        let mut shape = x.clone();
        shape[1] = shape[0];
        shape
    })]
    /// Compute the full `[n, n]` symmetric Euclidean pairwise distance matrix
    /// for an input tensor of shape `[n, d]`.
    ///
    /// The result satisfies `output[i, i] = 0` and
    /// `output[i, j] = ||x[i] − x[j]||₂` for `i ≠ j`.
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self>;

    /// Backward pass for [`euclidean_pairwise_distance`](Self::euclidean_pairwise_distance).
    ///
    /// Computes `∂loss/∂x` given:
    ///
    /// * `grad_pairwise` — upstream gradient `∂loss/∂pairwise_distances`, shape `[n, n]`
    /// * `x`             — original input saved from the forward pass, shape `[n, d]`
    /// * `pairwise`      — precomputed distance matrix saved from the forward pass, shape `[n, n]`
    ///
    /// Returns `∂loss/∂x`, shape `[n, d]`.
    #[fusion(dtype = x, shape = x)]
    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |pairwise_distances, k| {
        let shape = [pairwise_distances.shape[0], *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                burn::tensor::DType::I32,
            ),
            burn::backend::fusion::custom::TensorSpec::new(
                shape.into(),
                pairwise_distances.dtype,
            ),
        )
    })]
    /// Select the `k` nearest neighbours for every row of `pairwise_distances`.
    ///
    /// Returns `(indices, distances)`:
    /// * `indices`   — `[n, k]` integer tensor of neighbour column indices
    /// * `distances` — `[n, k]` float tensor of corresponding distances
    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>);

    /// Backward pass for [`knn`](Self::knn).
    ///
    /// Re-runs the forward k-NN pass to recover which neighbours were selected,
    /// then propagates `grad_output` (shape `[n, k]`) back to a gradient with
    /// respect to `pairwise_distances` (shape `[n, n]`).
    ///
    /// The autodiff adapter registers this operation in Burn's gradient graph.
    #[fusion(dtype = pairwise_distances, shape = pairwise_distances)]
    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

/// Forward-only operations used by the nonparametric standard UMAP path.
///
/// Kept separate from [`UmapOps`] so bounded graph/search work cannot
/// accidentally route through the differentiable full-pairwise operations.
#[backend_extension(Cube: cfg(any(feature = "cpu", feature = "cuda")), Fusion: cfg(any(feature = "cpu", feature = "cuda")))]
pub trait StandardUmapOps: Backend {
    #[fusion(meta = |data, k, seed, max_iterations| {
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
    /// Approximate fit neighbors by bounded NN-Descent. Returns `[N, k]` nonself
    /// indices and Euclidean distances sorted by `(distance, index)`.
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    #[fusion(meta = |query, reference, graph, k, candidate_capacity, visit_capacity, seed, query_offset| {
        let rows = query.shape[0];
        let shape = [rows, *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), graph.dtype),
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), query.dtype),
            burn::backend::fusion::custom::TensorSpec::new([rows, 2].into(), graph.dtype),
        )
    })]
    /// Search a proximity graph with bounded retained and unique-evaluation capacities.
    /// The last result contains `[peak_candidates, unique_evaluations]` per query;
    /// invalid graph indices are marked by `[-1, -1]`.
    #[allow(clippy::too_many_arguments)]
    fn approximate_query(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        graph: IntTensor<Self>,
        k: u32,
        candidate_capacity: u32,
        visit_capacity: u32,
        seed: u64,
        query_offset: u64,
    ) -> (IntTensor<Self>, FloatTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |query, reference, graph, k, candidate_capacity, visit_capacity, seed, query_offset| {
        let rows = query.shape[0];
        let shape = [rows, *k as usize];
        (
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), graph.dtype),
            burn::backend::fusion::custom::TensorSpec::new(shape.into(), query.dtype),
            burn::backend::fusion::custom::TensorSpec::new([rows, 2].into(), graph.dtype),
        )
    })]
    /// Search a proximity graph previously validated by the caller.
    ///
    /// Every graph entry must be `-1` or in `[0, reference.shape(0))`; the
    /// caller must have validated this graph once (for example, via the fit
    /// marker and its private validation token). This skips the per-row whole-
    /// graph scan. If the precondition is violated, per-edge bounds checks
    /// still protect memory safety, but results are not promised to mark the
    /// graph with `[-1, -1]` statistics.
    #[allow(clippy::too_many_arguments)]
    fn approximate_query_prevalidated(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        graph: IntTensor<Self>,
        k: u32,
        candidate_capacity: u32,
        visit_capacity: u32,
        seed: u64,
        query_offset: u64,
    ) -> (IntTensor<Self>, FloatTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |indices, distances, local_connectivity, bandwidth, set_op_mix_ratio, bipartite, n_reference| {
        let n = indices.shape[0];
        let k = indices.shape[1];
        let capacity = n.checked_mul(k).and_then(|value| value.checked_mul(2)).expect("fuzzy graph capacity overflows usize");
        (
            burn::backend::fusion::custom::TensorSpec::new([capacity].into(), burn::tensor::DType::I32),
            burn::backend::fusion::custom::TensorSpec::new([capacity].into(), burn::tensor::DType::I32),
            burn::backend::fusion::custom::TensorSpec::new([capacity].into(), distances.dtype),
            burn::backend::fusion::custom::TensorSpec::new([n].into(), distances.dtype),
            burn::backend::fusion::custom::TensorSpec::new([n].into(), distances.dtype),
            burn::backend::fusion::custom::TensorSpec::new(indices.shape.clone(), distances.dtype),
        )
    })]
    /// Compute smooth-kNN memberships and their sparse fuzzy symmetrization.
    /// COO arrays are sorted by `(head, tail)`, unique, and have `2*N*K` capacity;
    /// unused slots are zero. Bipartite query and reference ids retain their indices.
    /// If an index other than the valid range or the `-1` disconnected marker is
    /// present, the first COO entry is marked by head/tail `-2` and weight `-1`.
    #[allow(clippy::type_complexity)]
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
    );

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
    /// Select exact nearest reference rows for a bounded query block.
    /// Results are sorted by `(Euclidean distance, reference index)`.
    fn exact_neighbors(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>);

    /// Compute squared Euclidean distances from a query block `[B, D]` to
    /// reference rows `[N, D]`, returning only the bounded `[B, N]` block.
    #[fusion(dtype = query, shape = {
        let mut shape = query.clone();
        shape[1] = reference[0];
        shape
    })]
    fn squared_euclidean_distance_block(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
    ) -> FloatTensor<Self>;

    #[fusion(meta = |heads, weights, head_rows| {
        let offsets_len = usize::try_from(*head_rows)
            .expect("layout head rows exceed usize")
            .checked_add(1)
            .expect("layout head offsets overflow usize");
        let ids_len = heads.shape[0].max(1);
        (
            burn::backend::fusion::custom::TensorSpec::new(
                [offsets_len].into(),
                burn::tensor::DType::I32,
            ),
            burn::backend::fusion::custom::TensorSpec::new(
                [ids_len].into(),
                burn::tensor::DType::I32,
            ),
        )
    })]
    /// Build a stable per-head CSR of positive-weight layout edges.
    ///
    /// `offsets` has shape `[head_rows + 1]`; its final value is the valid
    /// prefix length in `ids`, whose fixed shape is `[heads.len().max(1)]`.
    /// Each head's prefix segment contains original COO edge positions in
    /// ascending order. Unused `ids` capacity is initialized to `1`; when the
    /// valid prefix is empty, this is the same non-empty dummy used by the host
    /// oracle. Invalid endpoints set `offsets[0]` to `-2`, and non-finite or
    /// negative weights set it to `-3`. When several edges are invalid, the
    /// earliest edge wins and an invalid endpoint wins over an invalid weight
    /// on that same edge.
    fn head_csr(
        heads: IntTensor<Self>,
        weights: FloatTensor<Self>,
        head_rows: u32,
    ) -> (IntTensor<Self>, IntTensor<Self>);

    #[fusion(meta = |head_embedding, tail_embedding, heads, tails, weights, epochs_per_sample, epoch_of_next_sample, epochs_per_negative_sample, epoch_of_next_negative_sample, rng_state, head_offsets, head_edges, epoch, a, b, gamma, alpha, negative_sample_rate, move_other, shared_embeddings| {
        (
            burn::backend::fusion::custom::TensorSpec::new(head_embedding.shape.clone(), head_embedding.dtype),
            burn::backend::fusion::custom::TensorSpec::new(tail_embedding.shape.clone(), tail_embedding.dtype),
            burn::backend::fusion::custom::TensorSpec::new(epoch_of_next_sample.shape.clone(), epoch_of_next_sample.dtype),
            burn::backend::fusion::custom::TensorSpec::new(epoch_of_next_negative_sample.shape.clone(), epoch_of_next_negative_sample.dtype),
            burn::backend::fusion::custom::TensorSpec::new(rng_state.shape.clone(), rng_state.dtype),
        )
    })]
    /// Optimize one standard UMAP epoch using only selected-backend tensors.
    ///
    /// `head_offsets`/`head_edges` list, per head row, the original ids of
    /// positive-weight edges with that head in ascending id order;
    /// Parallel backends use atomic coordinate updates and exclusive
    /// per-head RNG ownership. Returns the updated head, tail,
    /// positive clocks, negative clocks and RNG state.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
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
    );
}
