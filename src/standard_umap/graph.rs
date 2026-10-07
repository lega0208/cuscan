use burn::{
    backend::{
        tensor::{FloatTensor, IntTensor},
        Backend,
    },
    tensor::{Bool, Int, Tensor},
};

/// Construct a sparse fuzzy graph using only operations on the input backend.
/// The fixed-size edge arrays hold at most two entries for each input neighbor.
#[allow(clippy::type_complexity)]
pub(crate) fn fuzzy_graph<B: Backend>(
    indices: IntTensor<B>,
    distances: FloatTensor<B>,
    local_connectivity: f32,
    bandwidth: f32,
    mix: f32,
    bipartite: bool,
    n_reference: u32,
) -> (
    IntTensor<B>,
    IntTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
    FloatTensor<B>,
)
where
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let indices: Tensor<2, Int> = Tensor::from_primitive::<B>(indices);
    let distances: Tensor<2> = Tensor::from_primitive::<B>(distances);
    let [n, k] = indices.dims();
    assert!(n > 0 && k > 0, "neighbor tensors must be non-empty");
    assert_eq!(distances.dims(), [n, k], "neighbor distance shape mismatch");
    assert_eq!(
        distances.dtype(),
        burn::tensor::DType::F32,
        "fuzzy graph requires f32 distances"
    );
    assert!(local_connectivity.is_finite() && local_connectivity >= 0.0);
    assert!(bandwidth.is_finite() && bandwidth > 0.0);
    assert!(mix.is_finite() && (0.0..=1.0).contains(&mix));
    let n_reference = usize::try_from(n_reference).expect("reference count exceeds usize");
    assert!(n_reference > 0 && n_reference <= i32::MAX as usize);
    if !bipartite {
        assert_eq!(
            n, n_reference,
            "fit graph reference count must equal query count"
        );
    }
    let nodes = if bipartite {
        n.checked_add(n_reference)
            .expect("bipartite node count overflows usize")
    } else {
        n
    };
    assert!(
        nodes <= i32::MAX as usize,
        "graph node ids exceed i32 range"
    );
    let key_sentinel = i64::try_from(
        nodes
            .checked_mul(nodes)
            .expect("lexicographic edge key overflows usize"),
    )
    .expect("graph edge key exceeds i64 range");
    let capacity = n
        .checked_mul(k)
        .and_then(|value| value.checked_mul(2))
        .expect("fuzzy graph capacity overflows usize");
    assert!(
        capacity <= i32::MAX as usize,
        "fuzzy graph COO indices exceed i32 range"
    );
    let device = indices.device();

    // Smooth-kNN rho uses positive distances only; input indices are handled
    // later when directed memberships are formed.
    let valid_index = indices
        .clone()
        .greater_equal_elem(0)
        .bool_and(indices.clone().lower_elem(n_reference as i32));
    let row_ids = Tensor::<1, Int>::arange(0..n as i64, &device)
        .reshape([n, 1])
        .repeat_dim(1, k);
    let positive = distances.clone().greater_elem(0.0f32);
    let positive_count = positive.clone().int().sum_dim(1);
    let mut sort_working = distances
        .clone()
        .mask_fill(positive.clone().bool_not(), f32::MAX);
    let column_ids = Tensor::<1, Int>::arange(0..k as i64, &device)
        .reshape([1, k])
        .repeat_dim(0, n);
    let mut ordered_values = Vec::with_capacity(k);
    for _ in 0..k {
        let (smallest, selected) = sort_working.clone().min_dim_with_indices(1);
        ordered_values.push(smallest);
        sort_working = sort_working.mask_fill(column_ids.clone().equal(selected), f32::MAX);
    }
    let ordered = Tensor::cat(ordered_values, 1);
    let positive_max = distances
        .clone()
        .mask_fill(positive.clone().bool_not(), 0.0f32)
        .max_dim(1);
    let floor_connectivity = local_connectivity.floor() as usize;
    let fraction = local_connectivity - floor_connectivity as f32;
    let interpolated = if floor_connectivity == 0 {
        ordered.slice_dim(1, 0..1).mul_scalar(fraction)
    } else {
        let lo = (floor_connectivity - 1).min(k - 1);
        let hi = floor_connectivity.min(k - 1);
        let before = ordered.clone().slice_dim(1, lo..lo + 1);
        if fraction > 1e-5 {
            let after = ordered.slice_dim(1, hi..hi + 1);
            before.clone().add(after.sub(before).mul_scalar(fraction))
        } else {
            before
        }
    };
    let enough = positive_count
        .clone()
        .greater_equal_elem(local_connectivity.ceil() as i32);
    let any_positive = positive_count.greater_elem(0);
    let rho = positive_max
        .mask_where(enough, interpolated)
        .mask_fill(any_positive.bool_not(), 0.0f32);

    // Reference smooth-kNN bandwidth: fixed-iteration row-wise binary search.
    let input_count = n
        .checked_mul(k)
        .expect("neighbor dimensions overflow usize");
    let global_mean = distances
        .clone()
        .sum()
        .div_scalar(input_count as f32)
        .reshape([1, 1]);
    let row_mean = distances.clone().sum_dim(1).div_scalar(k as f32);
    let floor_mean = global_mean.mask_where(rho.clone().greater_elem(0.0f32), row_mean);
    let target = (k as f32).log2() * bandwidth;
    let mut lower = Tensor::<2>::zeros([n, 1], &device);
    let mut upper = Tensor::<2>::full([n, 1], f32::MAX, &device);
    let mut mid = Tensor::<2>::ones([n, 1], &device);
    for _ in 0..64 {
        let mut sum = Tensor::<2>::zeros([n, 1], &device);
        for slot in 1..k {
            let delta = distances
                .clone()
                .slice_dim(1, slot..slot + 1)
                .sub(rho.clone());
            let value = delta
                .clone()
                .mul_scalar(-1.0)
                .div(mid.clone())
                .exp()
                .mask_fill(delta.lower_equal_elem(0.0f32), 1.0f32);
            sum = sum.add(value);
        }
        let converged = sum.clone().sub_scalar(target).abs().lower_elem(1e-5f32);
        let above = sum
            .clone()
            .greater_elem(target)
            .bool_and(converged.clone().bool_not());
        let below = sum
            .lower_elem(target)
            .bool_and(converged.clone().bool_not());
        let high_unset = upper.clone().equal_elem(f32::MAX);
        let current_mid = mid.clone();
        let next_lower = lower.clone().mask_where(below.clone(), current_mid.clone());
        let next_upper = upper.clone().mask_where(above.clone(), current_mid.clone());
        let should_double = high_unset.bool_and(below);
        let bisected = next_lower.clone().add(next_upper.clone()).mul_scalar(0.5);
        let doubled = mid.clone().mul_scalar(2.0);
        let next_mid = bisected.mask_where(should_double.clone(), doubled);
        mid = next_mid.mask_where(converged, current_mid);
        lower = next_lower;
        upper = next_upper;
    }
    let sigma_floor = floor_mean.mul_scalar(1e-3);
    let sigma = mid
        .clone()
        .mask_where(mid.clone().lower(sigma_floor.clone()), sigma_floor);

    // Directed memberships stay in input slot order and are masked on-device.
    let rho_rows = rho.clone().repeat_dim(1, k);
    let sigma_rows = sigma.clone().repeat_dim(1, k);
    let delta = distances.clone().sub(rho_rows);
    let directed_valid = if bipartite {
        valid_index.clone()
    } else {
        valid_index
            .clone()
            .bool_and(indices.clone().not_equal(row_ids.clone()))
    };
    let safe_sigma = sigma_rows
        .clone()
        .mask_fill(sigma_rows.clone().equal_elem(0.0f32), 1.0f32);
    let directed = delta
        .clone()
        .clamp_min(0.0f32)
        .mul_scalar(-1.0)
        .div(safe_sigma)
        .exp()
        .mask_fill(
            delta
                .lower_equal_elem(0.0f32)
                .bool_or(sigma_rows.equal_elem(0.0f32)),
            1.0f32,
        )
        .mask_fill(directed_valid.bool_not(), 0.0f32);

    // Sum any repeated directed slots, search reciprocal rows with bounded
    // [N,K] gathers, then emit each undirected pair once.
    let safe_tail = indices
        .clone()
        .clamp(0, n_reference.saturating_sub(1) as i32);
    let mut reverse_weight = Tensor::<2>::zeros([n, k], &device);
    let mut has_reverse = Tensor::<2, Bool>::full([n, k], false, &device);
    if !bipartite {
        for slot in 0..k {
            let reverse_index = indices
                .clone()
                .slice_dim(1, slot..slot + 1)
                .repeat_dim(1, k)
                .gather(0, safe_tail.clone());
            let reverse_value = directed
                .clone()
                .slice_dim(1, slot..slot + 1)
                .repeat_dim(1, k)
                .gather(0, safe_tail.clone());
            let matches = reverse_index.equal(row_ids.clone());
            has_reverse = has_reverse.bool_or(matches.clone());
            reverse_weight =
                reverse_weight.add(reverse_value.mask_fill(matches.bool_not(), 0.0f32));
        }
    }
    let edge_heads = row_ids.clone();
    let edge_tails = indices.clone();
    let nonself = if bipartite {
        Tensor::<2, Bool>::full([n, k], true, &device)
    } else {
        edge_heads.clone().not_equal(indices.clone())
    };
    let valid_edge = valid_index.bool_and(nonself);
    let slot_ids = Tensor::<1, Int>::arange(0..k as i64, &device)
        .reshape([1, k])
        .repeat_dim(0, n);
    let mut first_occurrence = Tensor::<2, Bool>::full([n, k], true, &device);
    let mut coalesced_forward = Tensor::<2>::zeros([n, k], &device);
    for slot in 0..k {
        let slot_index = indices
            .clone()
            .slice_dim(1, slot..slot + 1)
            .repeat_dim(1, k);
        let same_edge = indices.clone().equal(slot_index.clone());
        let later = slot_ids.clone().greater_elem(slot as i32);
        first_occurrence = first_occurrence.bool_and(same_edge.clone().bool_and(later).bool_not());
        let slot_weight = directed
            .clone()
            .slice_dim(1, slot..slot + 1)
            .repeat_dim(1, k);
        coalesced_forward =
            coalesced_forward.add(slot_weight.mask_fill(same_edge.bool_not(), 0.0f32));
    }
    let low_first = edge_heads.clone().lower(edge_tails.clone());
    let reverse_only = edge_heads
        .clone()
        .greater(edge_tails.clone())
        .bool_and(has_reverse.bool_not());
    let emit = if bipartite {
        valid_edge.clone()
    } else {
        valid_edge.bool_and(low_first.bool_or(reverse_only))
    };
    let emit = emit.bool_and(first_occurrence);
    let fuzzy = if bipartite {
        // The bipartite transform graph keeps directed forward memberships
        // unchanged: set_op_mix_ratio only blends the reciprocal memberships
        // of the fit graph. The bipartite reverse membership is fixed at zero,
        // so the blend would scale every edge by mix, emptying the graph at
        // mix = 0.
        coalesced_forward
    } else {
        let intersection = coalesced_forward.clone().mul(reverse_weight.clone());
        coalesced_forward
            .add(reverse_weight)
            .sub(intersection.clone())
            .mul_scalar(mix)
            .add(intersection.mul_scalar(1.0 - mix))
    };
    // Bipartite rows and reference ids live in different index spaces: keep
    // (query, reference) orientation and emit each directed edge once.
    let swap = edge_heads
        .clone()
        .greater(edge_tails.clone())
        .bool_and(Tensor::<2, Bool>::full([n, k], !bipartite, &device));
    let low = edge_heads
        .clone()
        .mask_where(swap.clone(), edge_tails.clone());
    let high = edge_tails.clone().mask_where(swap, edge_heads.clone());
    let emit = emit.bool_and(fuzzy.clone().greater_elem(0.0f32));
    let key_base = nodes as i64;
    let forward_key = low
        .clone()
        .cast(burn::tensor::IntDType::I64)
        .mul_scalar(key_base)
        .add(high.clone().cast(burn::tensor::IntDType::I64))
        .mask_fill(emit.clone().bool_not(), key_sentinel);
    let reverse_key = high
        .clone()
        .cast(burn::tensor::IntDType::I64)
        .mul_scalar(key_base)
        .add(low.clone().cast(burn::tensor::IntDType::I64))
        .mask_fill(
            emit.clone()
                .bool_not()
                .bool_or(Tensor::<2, Bool>::full([n, k], bipartite, &device)),
            key_sentinel,
        );
    let mut keys = Tensor::cat(
        vec![forward_key.reshape([n * k]), reverse_key.reshape([n * k])],
        0,
    );
    let mut heads = Tensor::cat(
        vec![low.clone().reshape([n * k]), high.clone().reshape([n * k])],
        0,
    );
    let mut tails = Tensor::cat(
        vec![high.clone().reshape([n * k]), low.clone().reshape([n * k])],
        0,
    );
    let mut weights = Tensor::cat(
        vec![fuzzy.clone().reshape([n * k]), fuzzy.reshape([n * k])],
        0,
    );
    // Burn's generic integer argsort synchronously reads data on backends without
    // a native sort. Stable radix passes keep this sparse output device-resident.
    let radix_bits = (i64::BITS - key_sentinel.leading_zeros()) as usize;
    for bit in 0..radix_bits {
        let zero = keys
            .clone()
            .bitwise_right_shift_scalar(bit as i64)
            .bitwise_and_scalar(1)
            .equal_elem(0);
        let zero_prefix = zero
            .clone()
            .int()
            .cast(burn::tensor::IntDType::I64)
            .cumsum(0);
        let one_prefix = zero
            .clone()
            .bool_not()
            .int()
            .cast(burn::tensor::IntDType::I64)
            .cumsum(0);
        let zero_total = zero_prefix
            .clone()
            .slice((capacity - 1)..capacity)
            .repeat_dim(0, capacity);
        let one_rank = zero_total.add(one_prefix).sub_scalar(1);
        let zero_rank = zero_prefix.sub_scalar(1);
        let rank = one_rank
            .mask_where(zero, zero_rank)
            .cast(burn::tensor::IntDType::I32);
        keys = Tensor::<1, Int>::full([capacity], 0, &device)
            .cast(burn::tensor::IntDType::I64)
            .scatter(
                0,
                rank.clone(),
                keys,
                burn::tensor::IndexingUpdateOp::Assign,
            );
        heads = Tensor::<1, Int>::zeros([capacity], &device).scatter(
            0,
            rank.clone(),
            heads,
            burn::tensor::IndexingUpdateOp::Assign,
        );
        tails = Tensor::<1, Int>::zeros([capacity], &device).scatter(
            0,
            rank.clone(),
            tails,
            burn::tensor::IndexingUpdateOp::Assign,
        );
        weights = Tensor::<1>::zeros([capacity], &device).scatter(
            0,
            rank,
            weights,
            burn::tensor::IndexingUpdateOp::Assign,
        );
    }
    let unused = keys.equal_elem(key_sentinel);
    let heads = heads.mask_fill(unused.clone(), 0);
    let tails = tails.mask_fill(unused.clone(), 0);
    let weights = weights.mask_fill(unused, 0.0f32);
    let invalid_index = indices
        .clone()
        .lower_elem(-1)
        .bool_or(indices.clone().greater_equal_elem(n_reference as i32));
    let invalid_any = invalid_index.int().sum().greater_elem(0).reshape([1]);
    let heads = heads.clone().slice_assign(
        0..1,
        heads.clone().slice(0..1).mask_where(
            invalid_any.clone(),
            Tensor::<1, Int>::full([1], -2, &device),
        ),
    );
    let tails = tails.clone().slice_assign(
        0..1,
        tails.clone().slice(0..1).mask_where(
            invalid_any.clone(),
            Tensor::<1, Int>::full([1], -2, &device),
        ),
    );
    let weights = weights.clone().slice_assign(
        0..1,
        weights
            .clone()
            .slice(0..1)
            .mask_where(invalid_any, Tensor::<1>::full([1], -1.0f32, &device)),
    );
    debug_assert_eq!(heads.dims(), [capacity]);
    (
        heads
            .try_into_primitive::<B>()
            .expect("graph heads remain on selected backend"),
        tails
            .try_into_primitive::<B>()
            .expect("graph tails remain on selected backend"),
        weights
            .try_into_primitive::<B>()
            .expect("graph weights remain on selected backend"),
        rho.reshape([n])
            .try_into_primitive::<B>()
            .expect("rho remains on selected backend"),
        sigma
            .reshape([n])
            .try_into_primitive::<B>()
            .expect("sigma remains on selected backend"),
        directed
            .try_into_primitive::<B>()
            .expect("directed memberships remain on selected backend"),
    )
}
