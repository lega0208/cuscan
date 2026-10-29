//! Backend-neutral CubeCL discovery and neighbor operations.
pub mod backend;
mod backend_choice;
pub mod cuda_backend;
mod device;
mod error;
mod kernels;
pub mod neighbors;

pub use backend_choice::{BackendChoice, validate_backend_choice};
pub use device::{construct_cpu_device, selected_device};
pub use error::{AnnError, BackendError};
pub use kernels::ann::{AnnPlan, ann_preflight};
pub use kernels::legal_cube_dim;
