//! Bounded sampled NN-descent ANN: launch plan, forest, and shared CPU/CUDA kernels.

mod ann_forest;
mod ann_sampled;
#[path = "ann.rs"]
mod orchestration;

pub(crate) use orchestration::approximate_neighbors;
pub use orchestration::{AnnPlan, ann_preflight};
