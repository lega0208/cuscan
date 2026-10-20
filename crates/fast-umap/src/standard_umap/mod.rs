//! Standard (nonparametric) UMAP estimator and selected-backend primitives.
//!
//! Backend-free host items are defined in `fast-umap-core` and re-exported here.

mod backend_choice;
mod estimator;

pub use fast_umap_core::init;
pub mod neighbors;
pub(crate) use fast_umap_core::query;

pub use backend_choice::{BackendChoice, validate_backend_choice};
pub use estimator::{FittedUMAP, UMAP};
pub use fast_umap_core::{
    Embedding, FitNeighborWidths, Init, ModelState, NeighborSearch, QueryBudget, Result,
    UMAPConfig, UmapError, approximate_query_budget, fit_curve, fit_neighbor_widths, initialize,
    query_neighbor_width, validate_input, validate_search_settings,
};
