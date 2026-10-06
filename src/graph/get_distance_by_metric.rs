use crate::backend::UmapOps;
use burn::{
    backend::Dispatch,
    tensor::{Int, Tensor, TensorData},
};

/// Compute the full [n, n] pairwise Euclidean distance matrix for `data` [n, d].
///
/// Differentiable: the custom `euclidean_pairwise_distance` kernel registers its
/// own backward pass, so gradients flow back through this call during training.
pub fn pairwise_distances(data: Tensor<2>) -> Tensor<2> {
    let pairwise = Dispatch::euclidean_pairwise_distance(data.into_dispatch());
    Tensor::from_dispatch(pairwise)
}

/// CPU-side k-NN: given a flat row-major `[n × n]` floating-point distance matrix, find
/// the k nearest neighbours for every row (excluding the self-distance).
///
/// Returns `(indices, distances)` each flat in row-major order, length `n * k`.
/// * `indices`   – neighbour column indices as i32
/// * `distances` – corresponding distances, retaining the input element type
pub fn knn_from_pairwise_cpu<T>(flat: &[T], n: usize, k: usize) -> (Vec<i32>, Vec<T>)
where
    T: num::Float + Copy,
{
    if k == 0 {
        return (Vec::new(), Vec::new());
    }
    assert!(k < n, "k ({k}) must be strictly less than n ({n})");

    let mut out_idx = vec![0i32; n * k];
    let mut out_dist = vec![T::zero(); n * k];

    for i in 0..n {
        let mut row: Vec<(T, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (flat[i * n + j], j))
            .collect();

        // Partial-sort: first k entries become the k smallest (not yet ordered).
        if k < row.len() {
            row.select_nth_unstable_by(k - 1, |a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        // Sort the k winners in ascending order.
        row[..k].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        for (slot, (dist, idx)) in row[..k].iter().enumerate() {
            out_idx[i * k + slot] = *idx as i32;
            out_dist[i * k + slot] = *dist;
        }
    }

    (out_idx, out_dist)
}

/// Pack raw CPU k-NN output into GPU tensors.
///
/// Returns `(knn_indices [n, k] Int, knn_dist [n, k] Float)`.
/// The distances are raw Euclidean values (not normalised) — the UMAP loss
/// uses the indices to gather distances and applies the Student-t kernel
/// directly, so no pre-normalisation is needed here.
#[allow(dead_code)]
pub fn knn_tensors_from_cpu(
    idx_flat: Vec<i32>,
    dist_flat: Vec<f32>,
    n: usize,
    k: usize,
    device: &burn::tensor::Device,
) -> (Tensor<2, Int>, Tensor<2>) {
    let knn_indices: Tensor<2, Int> = Tensor::from_data(TensorData::new(idx_flat, [n, k]), device);
    let knn_dist: Tensor<2> = Tensor::from_data(TensorData::new(dist_flat, [n, k]), device);
    (knn_indices, knn_dist)
}

#[cfg(test)]
mod tests {
    use super::knn_from_pairwise_cpu;

    #[test]
    fn knn_sorts_exact_distance_ties_by_candidate_order() {
        let distances = [0.0f32, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let (indices, selected) = knn_from_pairwise_cpu(&distances, 3, 2);
        assert_eq!(indices, [1, 2, 0, 2, 0, 1]);
        assert_eq!(selected, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn knn_partial_selection_preserves_pre_move_cutoff_ties() {
        // Independently captured from the old helper at b9d5bbd6: three tied
        // candidates compete for two slots, exercising select_nth_unstable_by.
        let mut distances = vec![1.0f32; 16];
        for i in 0..4 {
            distances[i * 4 + i] = 0.0;
        }
        let (indices, selected) = knn_from_pairwise_cpu(&distances, 4, 2);
        assert_eq!(indices, [1, 2, 0, 2, 0, 1, 0, 1]);
        assert_eq!(selected, [1.0; 8]);
    }

    #[test]
    fn knn_handles_zero_neighbors_and_rejects_out_of_range_k() {
        assert_eq!(knn_from_pairwise_cpu::<f32>(&[], 0, 0), (vec![], vec![]));
        assert_eq!(knn_from_pairwise_cpu::<f32>(&[], 3, 0), (vec![], vec![]));

        let distances = [0.0f32, 2.0, 1.0, 2.0, 0.0, 3.0, 1.0, 3.0, 0.0];
        assert_eq!(
            knn_from_pairwise_cpu(&distances, 3, 2).0,
            [2, 1, 0, 2, 0, 1]
        );
        assert!(std::panic::catch_unwind(|| knn_from_pairwise_cpu(&distances, 3, 3)).is_err());
        assert!(std::panic::catch_unwind(|| knn_from_pairwise_cpu::<f32>(&[], 0, 1)).is_err());
    }

    #[test]
    fn knn_short_pairwise_storage_panics_when_row_is_indexed() {
        let short = [0.0f32, 1.0, 1.0, 0.0, 1.0];
        assert!(std::panic::catch_unwind(|| knn_from_pairwise_cpu(&short, 3, 1)).is_err());
    }
}
