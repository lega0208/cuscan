//! Backend selection and feature/device availability checks.

use super::{Result, UmapError};

/// Select the compute backend. WGPU is deliberately not a supported choice.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    #[default]
    Auto,
    Cpu,
    Cuda {
        device_index: usize,
    },
}

impl BackendChoice {
    /// Check feature availability and, for CUDA, the requested visible device.
    pub fn validate(self) -> Result<()> {
        validate_backend_choice(self)
    }
}

/// Validate explicit backend availability without silently substituting another backend.
pub fn validate_backend_choice(choice: BackendChoice) -> Result<()> {
    match choice {
        BackendChoice::Auto => {
            #[cfg(feature = "cuda")]
            {
                let count = crate::cuda_backend::try_device_count().map_err(|error| {
                    UmapError::BackendUnavailable(format!("CUDA discovery failed: {error}"))
                })?;
                if count == 0 && !cfg!(feature = "cpu") {
                    return Err(UmapError::BackendUnavailable(
                        "CUDA is unavailable and CPU backend requires the `cpu` feature".into(),
                    ));
                }
            }
            #[cfg(not(feature = "cuda"))]
            if !cfg!(feature = "cpu") {
                return Err(UmapError::BackendUnavailable(
                    "no compute backend is available; enable the `cpu` or `cuda` feature".into(),
                ));
            }
            Ok(())
        }
        BackendChoice::Cpu => {
            #[cfg(feature = "cpu")]
            {
                Ok(())
            }
            #[cfg(not(feature = "cpu"))]
            {
                Err(UmapError::BackendUnavailable(
                    "CPU backend requires the `cpu` feature".into(),
                ))
            }
        }
        BackendChoice::Cuda { device_index } => {
            #[cfg(feature = "cuda")]
            {
                let count = crate::cuda_backend::try_device_count().map_err(|error| {
                    UmapError::BackendUnavailable(format!("CUDA discovery failed: {error}"))
                })?;
                if device_index >= count {
                    return Err(UmapError::BackendUnavailable(format!(
                        "CUDA device index {device_index} is unavailable; {count} device(s) detected"
                    )));
                }
                Ok(())
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_index;
                Err(UmapError::BackendUnavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ))
            }
        }
    }
}
