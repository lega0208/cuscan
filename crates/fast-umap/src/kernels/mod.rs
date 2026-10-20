use burn::{
    backend::tensor::{FloatTensor, IntTensor},
    cubecl::CubeDim,
};
use burn_cubecl::CubeBackend;

mod standard_umap;

/// Bound independent-unit kernels to the selected runtime's actual limits.
/// Callers derive their cube counts from the returned dimensions, so reducing
/// a dimension cannot truncate a row, tile, or tail.
pub(crate) fn legal_cube_dim(client: &cubecl::client::Client, preferred: CubeDim) -> CubeDim {
    let hardware = &client.properties().hardware;
    legal_cube_dim_for_limits(
        preferred,
        hardware.max_cube_dim,
        hardware.max_units_per_cube,
    )
}

fn legal_cube_dim_for_limits(
    preferred: CubeDim,
    (max_x, max_y, max_z): (u32, u32, u32),
    max_units_per_cube: u32,
) -> CubeDim {
    let mut remaining = max_units_per_cube.max(1);
    let x = preferred.x.min(max_x).min(remaining).max(1);
    remaining /= x;
    let y = preferred.y.min(max_y).min(remaining).max(1);
    remaining /= y;
    let z = preferred.z.min(max_z).min(remaining).max(1);
    CubeDim::new_3d(x, y, z)
}

impl crate::backend::StandardUmapOps for CubeBackend {
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        standard_umap::approximate_neighbors(data, k, seed, max_iterations)
    }

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

    fn exact_neighbors(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        standard_umap::exact_neighbors(query, reference, k, exclude_self, query_offset)
    }

    fn squared_euclidean_distance_block(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        standard_umap::squared_euclidean_distance_block(query, reference)
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

#[cfg(test)]
mod launch_limit_tests {
    use super::*;

    #[test]
    fn legal_cube_dim_respects_axis_and_unit_limits_and_preserves_tails() {
        for (preferred, axis_limits, units, expected) in [
            (CubeDim::new_1d(32), (32, 1, 1), 1, (1, 1, 1)),
            (CubeDim::new_1d(32), (32, 1, 1), 8, (8, 1, 1)),
            (CubeDim::new_1d(32), (32, 1, 1), 64, (32, 1, 1)),
            (CubeDim::new_1d(64), (64, 1, 1), 64, (64, 1, 1)),
            (CubeDim::new_1d(64), (16, 1, 1), 64, (16, 1, 1)),
            (CubeDim::new_2d(16, 16), (16, 16, 1), 256, (16, 16, 1)),
            (CubeDim::new_2d(32, 32), (32, 32, 1), 1024, (32, 32, 1)),
            (CubeDim::new_2d(32, 32), (32, 32, 1), 64, (32, 2, 1)),
        ] {
            let dim = legal_cube_dim_for_limits(preferred, axis_limits, units);
            assert_eq!((dim.x, dim.y, dim.z), expected);
            let work = 67u32;
            let cubes = work.div_ceil(dim.x);
            assert!(cubes * dim.x >= work, "tail is covered by the final cube");
        }
    }
}
