#![allow(unused_variables)]
#![allow(unused_imports)]
//! Parametric UMAP with shared fuzzy-graph and distance infrastructure.
//!
//! The estimator and neural configuration use explicit parametric names:
//!
//! ```
//! #[cfg(feature = "parametric")]
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

#[cfg(not(any(feature = "cpu", feature = "cuda", feature = "gpu")))]
compile_error!("enable at least one backend feature: cpu, cuda, or gpu");

pub mod autodiff_ops;
pub mod backend;
#[cfg(feature = "plotters")]
pub mod chart;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod distances;
pub mod graph;
#[cfg(any(feature = "cpu", feature = "gpu", feature = "cuda"))]
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
#[cfg(feature = "parametric")]
pub use parametric_umap::{
    FittedParametricUMAP, ParametricUMAP, ParametricUMAPConfig, ParametricUMAPEpochProgress,
    ParametricUMAPLossReduction, ParametricUMAPModel, ParametricUMAPModelConfig,
    ParametricUMAPModelConfigBuilder, ParametricUMAPOptimizationParams,
    validate_parametric_umap_data, validate_parametric_umap_input,
};
pub use standard_umap::{
    BackendChoice, Embedding, FittedUMAP, Init, NeighborSearch, Result, UMAP, UMAPConfig, UmapError,
};

// export for python bindings
pub use burn::tensor::{Device, DeviceIndex, DeviceKind};
