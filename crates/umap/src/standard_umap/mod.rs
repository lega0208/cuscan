//! Standard (nonparametric) UMAP estimator and selected-backend primitives.
//!
//! Backend-free host items are defined in `umap-core` and re-exported here.

mod backend_choice;
mod estimator;
mod modern_negatives;
mod modern_orders;
mod modern_runtime;

pub use umap_core::init;
pub mod neighbors;
pub(crate) use umap_core::query;

pub use backend_choice::{BackendChoice, validate_backend_choice};
#[doc(hidden)]
pub use estimator::profile_recursive_driver;
pub use estimator::{FittedUMAP, UMAP};
pub use umap_core::{
    Embedding, FitNeighborWidths, HubSelection, Init, InitRescale, ModelState, NeighborSearch,
    Optimizer, QueryBudget, RecursiveInitConfig, Result, UMAPConfig, UmapError,
    approximate_query_budget, fit_curve, fit_neighbor_widths, initialize, query_neighbor_width,
    validate_input, validate_search_settings,
};
