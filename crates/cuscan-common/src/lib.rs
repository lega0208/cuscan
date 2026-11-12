//! Backend-neutral CubeCL discovery and neighbor operations.
pub mod backend;
mod backend_choice;
pub mod cuda_backend;
mod device;
mod error;
mod kernels;
pub mod memory;
pub mod neighbors;

pub use backend_choice::{BackendChoice, validate_backend_choice};
pub use device::{construct_cpu_device, selected_device};
pub use error::{AnnError, BackendError};
pub use kernels::ann::{AnnPlan, PlannedMemory, ann_preflight};
#[cfg(feature = "test-support")]
pub use kernels::ann::{ann_preflight_with_tile_slots, approximate_neighbors_with_tile_slots};
pub use kernels::legal_cube_dim;
pub use memory::{MemoryBudget, resolve_memory_budget};
