//! Standard (nonparametric) UMAP estimator and selected-backend primitives.

pub(crate) mod ann;
mod config;
mod estimator;
pub(crate) mod graph;
pub mod init;
pub(crate) mod layout;
pub mod neighbors;
pub(crate) mod query;

pub use estimator::{FittedUMAP, UMAP};
pub use init::initialize;

pub use config::{
    approximate_query_budget, fit_curve, fit_neighbor_widths, query_neighbor_width,
    validate_backend_choice, validate_input, validate_search_settings, BackendChoice,
    FitNeighborWidths, Init, NeighborSearch, QueryBudget, UMAPConfig,
};

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
