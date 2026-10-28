//! Exact HDBSCAN estimator with CPU and CUDA execution.

mod backend;
pub mod estimator;
pub mod kernels;
pub mod prediction;
pub mod resources;

pub use backend::{DeviceKdTree, HdbscanOps};
pub use estimator::{FittedHDBSCAN, HDBSCAN, HdbscanSearch};
#[doc(hidden)]
pub use estimator::{KnnDiagnostics, KnnNeighborSource, KnnSource, fit_knn_graph_with_source};
pub use fast_cubecl_common::BackendChoice;
pub use fast_hdbscan_core::{ClusterSelectionMethod, HdbscanConfig, HdbscanError, MstGraph};
pub use prediction::PredictionResult;
#[doc(hidden)]
pub use prediction::{PredictionCacheStatus, PredictionCoreStatus};
pub use resources::{
    ADVISORY_THRESHOLD_BYTES, FitResourceEstimate, KnnResourceEstimate, KnnResourceSource,
    ResourceAdvisory, ResourceBackend, ResourcePhase, ResourcePhaseBytes, ResourceScenario,
    estimate_fit_resources,
};
