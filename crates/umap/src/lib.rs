//! Standard (non-parametric) UMAP with CubeCL CPU and CUDA backends.
//!
//! ```no_run
//! use umap::{BackendChoice, NeighborSearch, UMAP, UMAPConfig};
//! ```

pub mod backend;
pub mod cuda_backend;
pub mod kernels;
pub mod standard_umap;

pub use standard_umap::{
    BackendChoice, Embedding, FittedUMAP, HubSelection, Init, InitRescale, ModelState,
    NeighborSearch, Optimizer, RecursiveInitConfig, Result, UMAP, UMAPConfig, UmapError,
};
