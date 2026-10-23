//! Standard (non-parametric) UMAP with CubeCL CPU and CUDA backends.
//!
//! ```no_run
//! use fast_umap::{BackendChoice, NeighborSearch, UMAP, UMAPConfig};
//! ```

pub mod backend;
pub mod cuda_backend;
pub mod kernels;
pub mod standard_umap;

pub use standard_umap::{
    BackendChoice, Embedding, FittedUMAP, Init, ModelState, NeighborSearch, Result, UMAP,
    UMAPConfig, UmapError,
};
