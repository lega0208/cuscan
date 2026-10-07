//! Backend-native, bounded NN-Descent neighbor search for standard UMAP.

use burn::{
    backend::{
        tensor::{FloatTensor, IntTensor},
        Backend,
    },
    tensor::{Int, Tensor},
};

const MAX_CANDIDATES: usize = 32;

pub(crate) fn candidate_width(n: usize, k: usize) -> usize {
    k.max(k.saturating_mul(4).min(MAX_CANDIDATES)).min(n - 1)
}

/// Run bounded NN-Descent using native Flex tensor operations. Candidate ids,
/// distances, deduplication and selection stay on the selected device.
pub(crate) fn approximate_neighbors<B: Backend>(
    data: FloatTensor<B>,
    k: u32,
    seed: u64,
    max_iterations: u32,
) -> (IntTensor<B>, FloatTensor<B>)
where
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let data: Tensor<2> = Tensor::from_primitive::<B>(data);
    let [n, features] = data.dims();
    assert!(
        n >= 2 && features > 0,
        "ANN input requires at least two rows and one feature"
    );
    assert!(n <= i32::MAX as usize, "ANN row indices exceed i32");
    let k = usize::try_from(k).expect("neighbor width exceeds usize");
    assert!(k > 0 && k < n, "ANN neighbor width must be in 1..N");
    let width = candidate_width(n, k);
    n.checked_mul(width)
        .expect("ANN candidate allocation overflows usize");
    width
        .checked_mul(width + 1)
        .expect("ANN candidate block overflows usize");
    n.checked_mul(features)
        .expect("ANN feature allocation overflows usize");
    n.checked_mul(
        width
            .checked_mul(width + 2)
            .expect("ANN candidate block overflows usize"),
    )
    .expect("ANN proposal allocation overflows usize");
    let device = data.device();
    // Floyd sampling without replacement over [0, N-1), using a per-row
    // Park-Miller stream. All RNG state and integer selections are device ops.
    const RNG_MODULUS: i64 = 2_147_483_647;
    let rng_rows = Tensor::<1, Int>::arange(0..n as i64, &device).cast(burn::tensor::IntDType::I64);
    let candidate_rows = rng_rows.clone().cast(burn::tensor::IntDType::I32);
    let seed_base = (seed % (RNG_MODULUS as u64 - 1)) as i64 + 1;
    let seed_wrap = rng_rows.clone().greater_equal_elem(RNG_MODULUS - seed_base);
    let mut rng = rng_rows.clone().add_scalar(seed_base).mask_where(
        seed_wrap,
        rng_rows.clone().sub_scalar(RNG_MODULUS - seed_base),
    );
    rng = rng.clone().mask_where(
        rng.clone().equal_elem(0),
        Tensor::<1, Int>::ones([n], &device).cast(burn::tensor::IntDType::I64),
    );
    let mut starts =
        Tensor::<2, Int>::full([n, width], -1, &device).cast(burn::tensor::IntDType::I32);
    for slot in 0..width {
        let j = n - 1 - width + slot;
        let hi = rng.clone().div_scalar(127_773i64);
        let lo = rng.clone().sub(hi.clone().mul_scalar(127_773i64));
        let test = lo.mul_scalar(16_807i64).sub(hi.mul_scalar(2_836i64));
        rng = test.clone().mask_where(
            test.clone().lower_equal_elem(0),
            test.add_scalar(RNG_MODULUS),
        );
        let t = rng.clone().sub_scalar(1);
        let draw = t
            .clone()
            .sub(
                t.clone()
                    .div_scalar((j + 1) as i64)
                    .mul_scalar((j + 1) as i64),
            )
            .cast(burn::tensor::IntDType::I32);
        let prior_raw = starts.clone().sub_scalar(1).mask_where(
            starts
                .clone()
                .lower_equal(candidate_rows.clone().reshape([n, 1])),
            starts.clone(),
        );
        let collision = prior_raw
            .equal(draw.clone().reshape([n, 1]))
            .int()
            .sum_dim(1)
            .reshape([n])
            .greater_elem(0);
        let chosen = draw.mask_where(
            collision,
            Tensor::<1, Int>::full([n], j as i32, &device).cast(burn::tensor::IntDType::I32),
        );
        let remapped = chosen.clone().mask_where(
            chosen.clone().greater_equal(candidate_rows.clone()),
            chosen.clone().add_scalar(1),
        );
        starts = starts.scatter(
            1,
            Tensor::<2, Int>::full([n, 1], slot as i32, &device).cast(burn::tensor::IntDType::I32),
            remapped.reshape([n, 1]),
            burn::tensor::IndexingUpdateOp::Assign,
        );
    }
    let row_ids = Tensor::<1, Int>::arange(0..n as i64, &device)
        .cast(burn::tensor::IntDType::I32)
        .reshape([n, 1]);
    let (mut candidates, mut distances) =
        select_candidates(&data, starts, n, features, width, width, &row_ids);

    for _ in 0..max_iterations {
        let iteration_snapshot = candidates.clone();
        let mut proposals = vec![iteration_snapshot.clone()];
        for slot in 0..width {
            let center = iteration_snapshot
                .clone()
                .slice_dim(1, slot..slot + 1)
                .repeat_dim(1, width);
            proposals.push(iteration_snapshot.clone().gather(0, center));
        }
        // Bounded reverse joins: for slot s, every edge row->c records row as a
        // reverse candidate of c in column s (one O(N) scatter per slot).
        for slot in 0..width {
            let tails = iteration_snapshot.clone().slice_dim(1, slot..slot + 1);
            let reverse = Tensor::<2, Int>::full([n, 1], -1, &device)
                .cast(burn::tensor::IntDType::I32)
                .scatter(
                    0,
                    tails,
                    row_ids.clone(),
                    burn::tensor::IndexingUpdateOp::Assign,
                );
            proposals.push(reverse);
        }
        let proposals = Tensor::cat(proposals, 1);
        let proposal_width = width * (width + 2);
        (candidates, distances) = select_candidates(
            &data,
            proposals,
            n,
            features,
            proposal_width,
            width,
            &row_ids,
        );
    }

    let indices = candidates.slice_dim(1, 0..k);
    let distances = distances.slice_dim(1, 0..k).sqrt();
    (
        indices
            .try_into_primitive::<B>()
            .expect("ANN ids remain on selected backend"),
        distances
            .try_into_primitive::<B>()
            .expect("ANN distances remain on selected backend"),
    )
}

fn select_candidates<B: Backend>(
    data: &Tensor<2>,
    proposals: Tensor<2, Int>,
    n: usize,
    features: usize,
    proposal_width: usize,
    width: usize,
    row_ids: &Tensor<2, Int>,
) -> (Tensor<2, Int>, Tensor<2>)
where
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let safe_proposals = proposals
        .clone()
        .mask_fill(proposals.clone().lower_elem(0), 0);
    let mut remaining = candidate_distances(data, &safe_proposals, n, features, proposal_width);
    remaining = remaining.mask_fill(
        proposals
            .clone()
            .equal(row_ids.clone().repeat_dim(1, proposal_width)),
        f32::MAX,
    );
    remaining = remaining.mask_fill(proposals.clone().lower_elem(0), f32::MAX);
    // Sort ids so duplicates are adjacent, drop repeats, then stably sort by
    // distance: equal distances keep ascending-id order.
    let (sorted_ids, permutation) = proposals.sort_with_indices(1);
    let sorted_distances = remaining.gather(1, permutation);
    let repeated = sorted_ids
        .clone()
        .slice_dim(1, 1..proposal_width)
        .equal(sorted_ids.clone().slice_dim(1, 0..proposal_width - 1));
    let repeated = Tensor::cat(
        vec![
            Tensor::<2, burn::tensor::Bool>::full([n, 1], false, &sorted_ids.device()),
            repeated,
        ],
        1,
    );
    let sorted_distances = sorted_distances.mask_fill(repeated, f32::MAX);
    let (best, order) = sorted_distances.sort_with_indices(1);
    let order = order.slice_dim(1, 0..width);
    (sorted_ids.gather(1, order), best.slice_dim(1, 0..width))
}

fn candidate_distances<B: Backend>(
    data: &Tensor<2>,
    candidates: &Tensor<2, Int>,
    n: usize,
    features: usize,
    width: usize,
) -> Tensor<2>
where
    burn::backend::DispatchTensor: burn::backend::DispatchKindConversion<B>,
{
    let mut distances = Tensor::<2>::zeros([n, width], &data.device());
    for feature_start in (0..features).step_by(32) {
        let feature_end = (feature_start + 32).min(features);
        let feature_block = feature_end - feature_start;
        let block = data.clone().slice_dim(1, feature_start..feature_end);
        let selected = block
            .clone()
            .select(0, candidates.clone().reshape([n * width]))
            .reshape([n, width, feature_block]);
        let delta = selected.sub(block.reshape([n, 1, feature_block]));
        distances = distances.add(delta.powf_scalar(2.0).sum_dim(2).reshape([n, width]));
    }
    distances
}
