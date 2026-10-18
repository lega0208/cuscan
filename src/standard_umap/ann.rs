//! Working graph width, independent of the sampled new/old pool capacity.

// The k-only graph lost exact neighbors on the small nonlinear regression
// (213/240 hits). A four-edge margin preserves alternative paths for refinement.
const HEAP_MARGIN: usize = 4;

pub(crate) fn candidate_width(n: usize, k: usize) -> usize {
    k.saturating_add(HEAP_MARGIN).min(n - 1)
}
