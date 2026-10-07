#![allow(unused_variables)]
#![allow(unused_imports)]
//! Parametric UMAP with shared fuzzy-graph and distance infrastructure.
//!
//! The estimator and neural configuration use explicit parametric names:
//!
//! ```
//! use fast_umap::{
//!     FittedParametricUMAP, ParametricUMAP, ParametricUMAPConfig,
//!     ParametricUMAPLossReduction, ParametricUMAPOptimizationParams,
//! };
//! use fast_umap::{GraphParams, ManifoldParams, Metric};
//! ```
//!
//! The former estimator names are intentionally not exported.
//!
//! ```compile_fail
//! use fast_umap::Umap;
//! ```
//!
//! ```compile_fail
//! use fast_umap::FittedUmap;
//! ```
//!
//! ```compile_fail
//! use fast_umap::UmapConfig;
//! ```
//!
//! ```compile_fail
//! use fast_umap::TrainingConfig;
//! ```
//!
//! ```compile_fail
//! use fast_umap::TrainingConfigBuilder;
//! ```
//!
//! Neural implementation modules are not part of the crate-level module API.
//!
//! ```compile_fail
//! use fast_umap::model;
//! ```
//!
//! ```compile_fail
//! use fast_umap::train;
//! ```
//!
//! ```compile_fail
//! use fast_umap::serialize;
//! ```
//!
//! The sparse trainer is private even inside the parametric namespace.
//!
//! ```compile_fail
//! use fast_umap::parametric_umap::trainer;
//! ```

pub mod autodiff_ops;
pub mod backend;
#[cfg(feature = "plotters")]
pub mod chart;
#[cfg(feature = "cpu")]
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod distances;
pub mod flex_backend;
pub mod generic_backend;
pub mod graph;
#[cfg(any(feature = "gpu", feature = "cuda"))]
pub mod kernels;
pub mod macros;
#[cfg(feature = "nn-descent")]
pub mod nn_descent;
pub mod normalizer;
pub mod parametric_umap;
#[cfg(feature = "pca")]
pub mod pca;
pub mod prelude;
pub mod standard_umap;
pub mod utils;

pub use graph::{GraphParams, ManifoldParams, Metric};
pub use parametric_umap::{
    validate_parametric_umap_data, validate_parametric_umap_input, FittedParametricUMAP,
    ParametricUMAP, ParametricUMAPConfig, ParametricUMAPEpochProgress, ParametricUMAPLossReduction,
    ParametricUMAPModel, ParametricUMAPModelConfig, ParametricUMAPModelConfigBuilder,
    ParametricUMAPOptimizationParams,
};
pub use standard_umap::{
    BackendChoice, Embedding, FittedUMAP, Init, NeighborSearch, Result, UMAPConfig, UmapError, UMAP,
};
