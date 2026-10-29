use burn::backend::tensor::{FloatTensor, IntTensor};
use burn_cubecl::CubeBackend;

mod standard_umap;

impl crate::backend::StandardUmapOps for CubeBackend {
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

    fn approximate_query_prevalidated(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        graph: IntTensor<Self>,
        k: u32,
        candidate_capacity: u32,
        visit_capacity: u32,
        seed: u64,
        query_offset: u64,
    ) -> (IntTensor<Self>, FloatTensor<Self>, IntTensor<Self>) {
        standard_umap::approximate_query_prevalidated(
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

    fn head_csr(
        heads: IntTensor<Self>,
        weights: FloatTensor<Self>,
        head_rows: u32,
    ) -> (IntTensor<Self>, IntTensor<Self>) {
        standard_umap::head_csr(heads, weights, head_rows)
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
