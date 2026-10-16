//! Shared bounded NN-Descent candidate sizing.

const MAX_CANDIDATES: usize = 32;

pub(crate) fn candidate_width(n: usize, k: usize) -> usize {
    k.max(k.saturating_mul(4).min(MAX_CANDIDATES)).min(n - 1)
}
