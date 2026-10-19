//! Standard (non-parametric) UMAP with CubeCL CPU and CUDA backends.
//!
//! ```no_run
//! use fast_umap::{BackendChoice, NeighborSearch, UMAP, UMAPConfig};
//! ```

#[cfg(not(any(feature = "cpu", feature = "cuda")))]
compile_error!("enable at least one backend feature: cpu or cuda");

pub mod backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod kernels;
pub mod standard_umap;

pub use standard_umap::{
    BackendChoice, Embedding, FittedUMAP, Init, NeighborSearch, Result, UMAP, UMAPConfig, UmapError,
};
