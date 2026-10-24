//! Backend-free host code for Standard UMAP: configuration, validation, errors, model state,
//! host-side query sizing math, and spectral initialization.

mod config;
pub mod init;
pub mod query;
mod state;

pub use config::{
    FitNeighborWidths, Init, NeighborSearch, QueryBudget, UMAPConfig, approximate_query_budget,
    fit_curve, fit_neighbor_widths, query_neighbor_width, validate_input, validate_search_settings,
};
pub use init::initialize;
pub use state::ModelState;

/// Embedding shape used by the standard UMAP pipeline.
pub type Embedding = Vec<Vec<f32>>;

/// Result type for standard UMAP configuration and input validation.
pub type Result<T> = std::result::Result<T, UmapError>;

/// Errors produced before or during standard UMAP setup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UmapError {
    InvalidConfig(String),
    InvalidInput(String),
    BackendUnavailable(String),
    Numerical(String),
    SpectralConvergence(String),
    Overflow(&'static str),
}

impl std::fmt::Display for UmapError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message)
            | Self::InvalidInput(message)
            | Self::BackendUnavailable(message)
            | Self::Numerical(message)
            | Self::SpectralConvergence(message) => formatter.write_str(message),
            Self::Overflow(subject) => write!(formatter, "{subject} overflows usize"),
        }
    }
}

impl std::error::Error for UmapError {}
