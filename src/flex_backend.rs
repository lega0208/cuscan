//! Flex backend implementation for fast-umap.
//!
//! Implements the custom [`UmapOps`](crate::backend::UmapOps) extension for
//! Burn's CPU [`Flex`](burn::backend::Flex) backend by delegating to the
//! generic tensor-based implementations in [`generic_backend`](crate::generic_backend).

use burn::backend::tensor::{FloatTensor, IntTensor};

use crate::backend::{StandardUmapOps, UmapOps};

impl UmapOps for burn::backend::Flex {
    fn euclidean_pairwise_distance(x: FloatTensor<Self>) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance(x)
    }

    fn euclidean_pairwise_distance_backward(
        grad_pairwise: FloatTensor<Self>,
        x: FloatTensor<Self>,
        pairwise: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::euclidean_pairwise_distance_backward(grad_pairwise, x, pairwise)
    }

    fn knn(pairwise_distances: FloatTensor<Self>, k: u32) -> (IntTensor<Self>, FloatTensor<Self>) {
        crate::generic_backend::knn(pairwise_distances, k)
    }

    fn knn_backward(
        pairwise_distances: FloatTensor<Self>,
        k: u32,
        grad_output: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        crate::generic_backend::knn_backward(pairwise_distances, k, grad_output)
    }
}

impl StandardUmapOps for burn::backend::Flex {
    fn approximate_neighbors(
        data: FloatTensor<Self>,
        k: u32,
        seed: u64,
        max_iterations: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        crate::standard_umap::ann::approximate_neighbors::<Self>(data, k, seed, max_iterations)
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
        crate::standard_umap::query::approximate_query::<Self>(
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
        crate::standard_umap::graph::fuzzy_graph::<Self>(
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
        let query_primitive = query;
        let reference_primitive = reference;
        let query: burn::tensor::Tensor<2> =
            burn::tensor::Tensor::from_primitive::<Self>(query_primitive.clone());
        let reference: burn::tensor::Tensor<2> =
            burn::tensor::Tensor::from_primitive::<Self>(reference_primitive.clone());
        let [query_rows, query_features] = query.dims();
        let [reference_rows, reference_features] = reference.dims();
        assert!(
            query_rows > 0 && reference_rows > 0,
            "neighbor inputs must be non-empty"
        );
        assert!(
            query_features > 0,
            "neighbor feature dimension must be non-empty"
        );
        assert_eq!(
            query_features, reference_features,
            "query and reference feature dimensions must match"
        );
        assert!(k > 0, "neighbor count must be positive");
        let query_rows_u32 = u32::try_from(query_rows).expect("query row count exceeds u32");
        let reference_rows_u32 =
            u32::try_from(reference_rows).expect("reference row count exceeds u32");
        i32::try_from(reference_rows).expect("reference indices exceed i32");
        let query_end = query_offset
            .checked_add(query_rows_u32)
            .expect("query offset plus block rows exceeds u32");
        if exclude_self {
            assert!(
                query_end <= reference_rows_u32,
                "self-excluded query indices must be valid reference indices"
            );
        }
        let available = reference_rows_u32 - exclude_self as u32;
        assert!(
            k <= available,
            "neighbor count exceeds available references"
        );
        let k = k as usize;
        query_rows
            .checked_mul(k)
            .expect("neighbor output dimensions overflow usize");

        let squared = burn::tensor::Tensor::<2>::from_primitive::<Self>(
            Self::squared_euclidean_distance_block(query_primitive, reference_primitive),
        );
        let reference_indices = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(
            0..i64::try_from(reference_rows).expect("reference rows exceed i64"),
            &query.device(),
        )
        .reshape([1, reference_rows]);
        let mut remaining = squared;
        if exclude_self {
            let query_indices = burn::tensor::Tensor::<1, burn::tensor::Int>::arange(
                0..i64::try_from(query_rows).expect("query rows exceed i64"),
                &query.device(),
            )
            .add_scalar(i64::from(query_offset))
            .reshape([query_rows, 1]);
            remaining = remaining.mask_fill(
                reference_indices.clone().equal(query_indices),
                f32::INFINITY,
            );
        }

        let mut selected_indices = Vec::with_capacity(k);
        let mut selected_distances = Vec::with_capacity(k);
        for _ in 0..k {
            let (squared_distance, index) = remaining.clone().min_dim_with_indices(1);
            selected_indices.push(index);
            selected_distances.push(squared_distance.clamp_min(0.0f32).sqrt());
            remaining = remaining.mask_fill(
                reference_indices
                    .clone()
                    .equal(selected_indices.last().unwrap().clone()),
                f32::INFINITY,
            );
        }

        (
            burn::tensor::Tensor::cat(selected_indices, 1)
                .try_into_primitive::<Self>()
                .expect("neighbor indices must remain on Flex"),
            burn::tensor::Tensor::cat(selected_distances, 1)
                .try_into_primitive::<Self>()
                .expect("neighbor distances must remain on Flex"),
        )
    }

    fn squared_euclidean_distance_block(
        query: FloatTensor<Self>,
        reference: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        let query: burn::tensor::Tensor<2> = burn::tensor::Tensor::from_primitive::<Self>(query);
        let reference: burn::tensor::Tensor<2> =
            burn::tensor::Tensor::from_primitive::<Self>(reference);
        let [query_rows, query_features] = query.dims();
        let [reference_rows, reference_features] = reference.dims();
        assert!(
            query_rows > 0 && reference_rows > 0,
            "distance blocks must be non-empty"
        );
        assert!(
            query_features > 0,
            "distance feature dimension must be non-empty"
        );
        assert_eq!(
            query_features, reference_features,
            "query and reference feature dimensions must match"
        );
        query_rows
            .checked_mul(reference_rows)
            .expect("squared-distance output dimensions overflow usize");

        let mut squared_distances =
            burn::tensor::Tensor::<2>::zeros([query_rows, reference_rows], &query.device());
        for feature in 0..query_features {
            let query_feature = query
                .clone()
                .slice_dim(1, feature..feature + 1)
                .reshape([query_rows, 1]);
            let reference_feature = reference
                .clone()
                .slice_dim(1, feature..feature + 1)
                .reshape([1, reference_rows]);
            let difference = query_feature.sub(reference_feature);
            squared_distances = squared_distances.add(difference.clone().mul(difference));
        }
        squared_distances
            .try_into_primitive::<Self>()
            .expect("squared-distance output must remain on Flex")
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
        _head_offsets: IntTensor<Self>,
        _head_edges: IntTensor<Self>,
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
        // The serial reference walks edges in id order; the ownership CSR is
        // only needed by parallel backends.
        crate::standard_umap::layout::optimize_epoch(
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
