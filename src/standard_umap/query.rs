//! Bounded best-first search over a retained training proximity graph.

use burn::{
    backend::{
        tensor::{FloatTensor, IntTensor},
        Backend, DispatchKindConversion, DispatchTensor,
    },
    tensor::{Int, Tensor},
};

fn row_distances(query: Tensor<2>, reference: Tensor<2>, ids: Tensor<1, Int>) -> Tensor<1> {
    let [batch, features] = query.dims();
    let [n, _] = reference.dims();
    let safe_ids = ids.clone().clamp(0, n.saturating_sub(1) as i32);
    let gather_ids = safe_ids.reshape([batch, 1]).repeat_dim(1, features);
    let selected = reference.gather(0, gather_ids);
    query
        .sub(selected)
        .powf_scalar(2.0)
        .sum_dim(1)
        .reshape([batch])
}

/// Approximate bipartite query against a bounded retained proximity graph.
///
/// All candidate IDs, distances, visit state, and queue updates remain tensors
/// on `B`. The host loop only schedules at most `visit_capacity` graph steps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn approximate_query<B: Backend>(
    query: FloatTensor<B>,
    reference: FloatTensor<B>,
    graph: IntTensor<B>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
) -> (IntTensor<B>, FloatTensor<B>, IntTensor<B>)
where
    DispatchTensor: DispatchKindConversion<B>,
{
    let query: Tensor<2> = Tensor::from_primitive::<B>(query);
    let reference: Tensor<2> = Tensor::from_primitive::<B>(reference);
    let graph: Tensor<2, Int> = Tensor::from_primitive::<B>(graph);
    let [batch, features] = query.dims();
    let [n, reference_features] = reference.dims();
    let [graph_rows, degree] = graph.dims();
    assert!(batch > 0 && n > 0 && features > 0 && degree > 0);
    assert_eq!(features, reference_features);
    assert_eq!(graph_rows, n);
    assert_eq!(query.dtype(), burn::tensor::DType::F32);
    assert_eq!(reference.dtype(), burn::tensor::DType::F32);
    assert!(n <= i32::MAX as usize);
    assert!(k > 0 && k as usize <= n);
    assert!(candidate_capacity >= k && candidate_capacity as usize <= n);
    assert!(visit_capacity >= k && visit_capacity as usize <= n);
    assert!(
        visit_capacity >= candidate_capacity,
        "visit budget must include the unique initial starts"
    );
    batch
        .checked_mul(candidate_capacity as usize)
        .expect("query candidate allocation overflows usize");
    batch
        .checked_mul(visit_capacity as usize)
        .expect("query visit allocation overflows usize");
    let c = candidate_capacity as usize;
    let v = visit_capacity as usize;
    let device = query.device();
    let infinity = f32::MAX;

    // Floyd sampling produces C unique starts without replacement for each
    // query. Park-Miller supplies deterministic pseudorandom draws keyed by
    // seed and absolute query offset; all state remains on the selected backend.
    const RNG_MODULUS: i32 = 2_147_483_647;
    let row_ids =
        Tensor::<1, Int>::arange(0..batch as i64, &device).cast(burn::tensor::IntDType::I32);
    let offset = (query_offset % RNG_MODULUS as u64) as i32;
    let offset_wrap = row_ids.clone().greater_equal_elem(RNG_MODULUS - offset);
    let absolute_rows = row_ids.clone().add_scalar(offset).mask_where(
        offset_wrap,
        row_ids.clone().sub_scalar(RNG_MODULUS - offset),
    );
    let seed_base = (seed % (RNG_MODULUS as u64 - 1)) as i32 + 1;
    let wraps = absolute_rows
        .clone()
        .greater_equal_elem(RNG_MODULUS - seed_base);
    let mut rng = absolute_rows
        .clone()
        .add_scalar(seed_base)
        .mask_where(wraps, absolute_rows.sub_scalar(RNG_MODULUS - seed_base));
    rng = rng.clone().mask_where(
        rng.clone().equal_elem(0),
        Tensor::<1, Int>::ones([batch], &device),
    );
    let mut starts = Tensor::<2, Int>::full([batch, c], -1, &device);
    for slot in 0..c {
        let j = n - c + slot;
        // Schrage's form avoids overflowing signed 32-bit tensor arithmetic.
        let hi = rng.clone().div_scalar(127_773);
        let lo = rng.clone().sub(hi.clone().mul_scalar(127_773));
        let test = lo.mul_scalar(16_807).sub(hi.mul_scalar(2_836));
        rng = test.clone().mask_where(
            test.clone().lower_equal_elem(0),
            test.add_scalar(RNG_MODULUS),
        );
        let t = rng.clone().sub_scalar(1);
        let draw = t.clone().sub(
            t.clone()
                .div_scalar((j + 1) as i32)
                .mul_scalar((j + 1) as i32),
        );
        let collision = starts
            .clone()
            .equal(draw.clone().reshape([batch, 1]))
            .int()
            .sum_dim(1)
            .reshape([batch])
            .greater_elem(0);
        let selected = draw.mask_where(
            collision,
            Tensor::<1, Int>::full([batch], j as i32, &device),
        );
        starts = starts.scatter(
            1,
            Tensor::<2, Int>::full([batch, 1], slot as i32, &device),
            selected.reshape([batch, 1]),
            burn::tensor::IndexingUpdateOp::Assign,
        );
    }
    let mut candidate_ids = starts.clone();
    let mut candidate_distances = Tensor::<2>::full([batch, c], infinity, &device);
    for slot in 0..c {
        let start = starts.clone().slice_dim(1, slot..slot + 1).reshape([batch]);
        let distance = row_distances(query.clone(), reference.clone(), start);
        candidate_distances = candidate_distances.scatter(
            1,
            Tensor::<2, Int>::full([batch, 1], slot as i32, &device),
            distance.reshape([batch, 1]),
            burn::tensor::IndexingUpdateOp::Assign,
        );
    }
    let mut expanded = Tensor::<2, Int>::zeros([batch, c], &device);
    let mut seen_ids = Tensor::<2, Int>::full([batch, v], -1, &device);
    let mut seen_distances = Tensor::<2>::full([batch, v], infinity, &device);
    for slot in 0..c {
        let start = starts.clone().slice_dim(1, slot..slot + 1);
        let distance = candidate_distances.clone().slice_dim(1, slot..slot + 1);
        seen_ids = seen_ids.scatter(
            1,
            Tensor::<2, Int>::full([batch, 1], slot as i32, &device),
            start,
            burn::tensor::IndexingUpdateOp::Assign,
        );
        seen_distances = seen_distances.scatter(
            1,
            Tensor::<2, Int>::full([batch, 1], slot as i32, &device),
            distance,
            burn::tensor::IndexingUpdateOp::Assign,
        );
    }
    let mut evaluated = Tensor::<1, Int>::full([batch], c as i32, &device);
    let mut peak = Tensor::<1, Int>::full([batch], c as i32, &device);
    let queue_slots = Tensor::<1, Int>::arange(0..c as i64, &device)
        .cast(burn::tensor::IntDType::I32)
        .reshape([1, c])
        .repeat_dim(0, batch);
    let seen_slots = Tensor::<1, Int>::arange(0..v as i64, &device)
        .cast(burn::tensor::IntDType::I32)
        .reshape([1, v])
        .repeat_dim(0, batch);
    for _ in 0..v {
        let available = candidate_ids
            .clone()
            .greater_equal_elem(0)
            .bool_and(expanded.clone().equal_elem(0));
        let pending = candidate_distances
            .clone()
            .mask_fill(available.clone().bool_not(), infinity);
        let best_distance = pending.clone().min_dim(1).reshape([batch]);
        let tied_ids = candidate_ids.clone().mask_fill(
            pending
                .clone()
                .equal(best_distance.clone().reshape([batch, 1]))
                .bool_not(),
            i32::MAX,
        );
        let best_id_value = tied_ids.min_dim(1).reshape([batch]);
        let best_slot = queue_slots
            .clone()
            .mask_fill(
                candidate_ids
                    .clone()
                    .equal(best_id_value.clone().reshape([batch, 1]))
                    .bool_not(),
                i32::MAX,
            )
            .min_dim(1)
            .reshape([batch]);
        let best_id = candidate_ids
            .clone()
            .gather(1, best_slot.clone().reshape([batch, 1]))
            .reshape([batch]);
        let has_best = best_distance.lower_elem(infinity);
        let mark_expanded = queue_slots
            .clone()
            .equal(best_slot.reshape([batch, 1]))
            .bool_and(has_best.clone().reshape([batch, 1]));
        expanded = expanded.mask_where(
            mark_expanded.clone(),
            Tensor::<2, Int>::ones([batch, c], &device),
        );
        candidate_ids = candidate_ids.mask_where(
            mark_expanded,
            Tensor::<2, Int>::full([batch, c], -1, &device),
        );

        let safe_best = best_id.clone().clamp(0, n.saturating_sub(1) as i32);
        let neighbor_ids = graph
            .clone()
            .gather(0, safe_best.reshape([batch, 1]).repeat_dim(1, degree));
        for edge_slot in 0..degree {
            let neighbor = neighbor_ids
                .clone()
                .slice_dim(1, edge_slot..edge_slot + 1)
                .reshape([batch]);
            let valid = has_best
                .clone()
                .bool_and(neighbor.clone().greater_equal_elem(0))
                .bool_and(neighbor.clone().lower_elem(n as i32))
                .bool_and(evaluated.clone().lower_elem(v as i32));
            let comparisons = seen_ids
                .clone()
                .equal(neighbor.clone().reshape([batch, 1]).repeat_dim(1, v));
            let was_seen = comparisons
                .int()
                .sum_dim(1)
                .reshape([batch])
                .greater_elem(0);
            let novel = valid.clone().bool_and(was_seen.bool_not());
            let safe_neighbor = neighbor.clone().clamp(0, n.saturating_sub(1) as i32);
            let distance = row_distances(query.clone(), reference.clone(), safe_neighbor.clone());

            let seen_pos = evaluated
                .clone()
                .clamp(0, (v - 1) as i32)
                .reshape([batch, 1]);
            let seen_update = novel.clone().reshape([batch, 1]);
            seen_ids = seen_ids
                .clone()
                .scatter(
                    1,
                    seen_pos.clone(),
                    neighbor.clone().reshape([batch, 1]),
                    burn::tensor::IndexingUpdateOp::Assign,
                )
                .mask_where(seen_update.clone().bool_not(), seen_ids.clone());
            seen_distances = seen_distances
                .clone()
                .scatter(
                    1,
                    seen_pos,
                    distance.clone().reshape([batch, 1]),
                    burn::tensor::IndexingUpdateOp::Assign,
                )
                .mask_where(seen_update.clone().bool_not(), seen_distances.clone());
            evaluated = evaluated.add(novel.clone().int());

            let empty = candidate_ids.clone().equal_elem(-1);
            let (empty_value, empty_slot) = empty.int().max_dim_with_indices(1);
            let empty_slot = empty_slot
                .cast(burn::tensor::IntDType::I32)
                .reshape([batch]);
            let free = empty_value.reshape([batch]).greater_elem(0);
            let worst_distance = candidate_distances
                .clone()
                .mask_fill(candidate_ids.clone().equal_elem(-1), -1.0f32)
                .max_dim(1)
                .reshape([batch]);
            let worst_id = candidate_ids
                .clone()
                .mask_fill(
                    candidate_distances
                        .clone()
                        .equal(worst_distance.clone().reshape([batch, 1]))
                        .bool_not(),
                    -1,
                )
                .max_dim(1)
                .reshape([batch]);
            let worst_slot = queue_slots
                .clone()
                .mask_fill(
                    candidate_ids
                        .clone()
                        .equal(worst_id.clone().reshape([batch, 1]))
                        .bool_not(),
                    i32::MAX,
                )
                .min_dim(1)
                .reshape([batch]);
            let replace_worst = distance
                .clone()
                .lower(worst_distance.clone().reshape([batch]))
                .bool_or(
                    distance
                        .clone()
                        .equal(worst_distance.reshape([batch]))
                        .bool_and(neighbor.clone().lower(worst_id)),
                );
            let insertion_slot = empty_slot.mask_where(free.clone().bool_not(), worst_slot);
            let insertion_allowed = novel.clone().bool_and(free.clone().bool_or(replace_worst));
            let insertion_mask = queue_slots
                .clone()
                .equal(insertion_slot.clone().reshape([batch, 1]))
                .bool_and(insertion_allowed.reshape([batch, 1]));
            candidate_ids = candidate_ids
                .clone()
                .scatter(
                    1,
                    insertion_slot.clone().reshape([batch, 1]),
                    neighbor.clone().reshape([batch, 1]),
                    burn::tensor::IndexingUpdateOp::Assign,
                )
                .mask_where(insertion_mask.clone().bool_not(), candidate_ids.clone());
            candidate_distances = candidate_distances
                .clone()
                .scatter(
                    1,
                    insertion_slot.clone().reshape([batch, 1]),
                    distance.reshape([batch, 1]),
                    burn::tensor::IndexingUpdateOp::Assign,
                )
                .mask_where(
                    insertion_mask.clone().bool_not(),
                    candidate_distances.clone(),
                );
            expanded = expanded
                .clone()
                .scatter(
                    1,
                    insertion_slot.reshape([batch, 1]),
                    Tensor::<2, Int>::zeros([batch, 1], &device),
                    burn::tensor::IndexingUpdateOp::Assign,
                )
                .mask_where(insertion_mask.bool_not(), expanded.clone());
        }
        let queue_count = candidate_ids
            .clone()
            .greater_equal_elem(0)
            .int()
            .sum_dim(1)
            .reshape([batch]);
        peak = peak
            .clone()
            .mask_where(queue_count.clone().greater(peak.clone()), queue_count);
        let continue_mask = evaluated.clone().lower_elem(v as i32).bool_and(
            candidate_ids
                .clone()
                .greater_equal_elem(0)
                .int()
                .sum_dim(1)
                .reshape([batch])
                .greater_elem(0),
        );
        // Exhausted query rows are marked by clearing all queue entries; the
        // fixed host loop then performs only inert device tensor operations.
        let keep = continue_mask.reshape([batch, 1]).repeat_dim(1, c);
        candidate_ids = candidate_ids.mask_fill(keep.clone().bool_not(), -1);
        expanded = expanded.mask_fill(keep.bool_not(), 1);
    }

    let mut output_ids = Vec::with_capacity(k as usize);
    let mut output_distances = Vec::with_capacity(k as usize);
    let mut remaining = seen_distances
        .clone()
        .mask_fill(seen_ids.clone().equal_elem(-1), infinity);
    for _ in 0..k {
        let distance = remaining.clone().min_dim(1).reshape([batch]);
        let tied_ids = seen_ids.clone().mask_fill(
            remaining
                .clone()
                .equal(distance.clone().reshape([batch, 1]))
                .bool_not(),
            i32::MAX,
        );
        let minimum_id = tied_ids.min_dim(1).reshape([batch]);
        let slot = seen_slots
            .clone()
            .mask_fill(
                seen_ids
                    .clone()
                    .equal(minimum_id.reshape([batch, 1]))
                    .bool_not(),
                i32::MAX,
            )
            .min_dim(1)
            .reshape([batch]);
        let id = seen_ids
            .clone()
            .gather(1, slot.clone().reshape([batch, 1]))
            .reshape([batch]);
        output_ids.push(id.reshape([batch, 1]));
        output_distances.push(distance.clamp_min(0.0f32).sqrt().reshape([batch, 1]));
        remaining =
            remaining.mask_fill(seen_slots.clone().equal(slot.reshape([batch, 1])), infinity);
    }
    let result_ids = Tensor::<2, Int>::cat(output_ids, 1);
    let result_distances = Tensor::<2>::cat(output_distances, 1);
    let invalid_graph = graph
        .clone()
        .lower_elem(-1)
        .bool_or(graph.clone().greater_equal_elem(n as i32))
        .int()
        .sum()
        .greater_elem(0);
    let invalid_rows = Tensor::<1, Int>::full([batch], -1, &device);
    let stats = Tensor::<2, Int>::cat(
        vec![peak.reshape([batch, 1]), evaluated.reshape([batch, 1])],
        1,
    )
    .mask_where(
        invalid_graph
            .reshape([1, 1])
            .repeat_dim(0, batch)
            .repeat_dim(1, 2),
        invalid_rows.reshape([batch, 1]).repeat_dim(1, 2),
    );
    (
        result_ids
            .try_into_primitive::<B>()
            .expect("query indices must remain on selected backend"),
        result_distances
            .try_into_primitive::<B>()
            .expect("query distances must remain on selected backend"),
        stats
            .try_into_primitive::<B>()
            .expect("query stats must remain on selected backend"),
    )
}
