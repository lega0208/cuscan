pub mod config;
pub use config::{GraphParams, ManifoldParams, Metric, fit_ab};
pub mod get_distance_by_metric;
pub(crate) mod sparse;
#[cfg(feature = "parametric")]
pub(crate) use sparse::{
    NN_DESCENT_THRESHOLD, WeightedEdge, directed_memberships, smooth_knn_dist, sparse_inputs,
    symmetrize, with_self_slots,
};
