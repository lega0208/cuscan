//! Bounded sampled NN-descent ANN: launch plan, forest, and shared CPU/CUDA kernels.

mod ann_forest;
mod ann_sampled;
#[path = "ann.rs"]
mod orchestration;

pub(crate) use orchestration::approximate_neighbors_planned;
pub use orchestration::{
    AnnEstimate, AnnPhase, AnnPhaseBytes, AnnPlan, PlannedMemory, ann_host_estimate, ann_preflight,
};
#[cfg(feature = "test-support")]
pub use orchestration::{ann_preflight_with_tile_slots, approximate_neighbors_with_tile_slots};
