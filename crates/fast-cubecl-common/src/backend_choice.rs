use crate::BackendError;
#[cfg(feature = "cuda")]
use crate::cuda_backend;

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
    pub fn validate(self) -> Result<(), BackendError> {
        validate_backend_choice(self, cfg!(feature = "cpu"), cfg!(feature = "cuda"))
    }
}

pub fn validate_backend_choice(
    choice: BackendChoice,
    cpu_allowed: bool,
    cuda_allowed: bool,
) -> Result<(), BackendError> {
    match choice {
        BackendChoice::Auto => {
            if cuda_allowed {
                #[cfg(feature = "cuda")]
                {
                    let count = cuda_backend::try_device_count().map_err(|e| {
                        BackendError::Unavailable(format!("CUDA discovery failed: {e}"))
                    })?;
                    if count > 0 {
                        return Ok(());
                    }
                    if count == 0 && !(cpu_allowed && cfg!(feature = "cpu")) {
                        return Err(BackendError::Unavailable(
                            "CUDA is unavailable and CPU backend requires the `cpu` feature".into(),
                        ));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Unavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ));
            }
            if !cpu_allowed && !cuda_allowed {
                return Err(BackendError::Unavailable(
                    "no compute backend is available; enable the `cpu` or `cuda` feature".into(),
                ));
            }
            if cpu_allowed && !cfg!(feature = "cpu") {
                return Err(BackendError::Unavailable(
                    "CPU backend requires the `cpu` feature".into(),
                ));
            }
            Ok(())
        }
        BackendChoice::Cpu => {
            if cpu_allowed && cfg!(feature = "cpu") {
                Ok(())
            } else {
                Err(BackendError::Unavailable(
                    "CPU backend requires the `cpu` feature".into(),
                ))
            }
        }
        BackendChoice::Cuda { device_index } => {
            if !cuda_allowed {
                return Err(BackendError::Unavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ));
            }
            #[cfg(feature = "cuda")]
            {
                let count = cuda_backend::try_device_count().map_err(|e| {
                    BackendError::Unavailable(format!("CUDA discovery failed: {e}"))
                })?;
                if device_index >= count {
                    return Err(BackendError::Unavailable(format!(
                        "CUDA device index {device_index} is unavailable; {count} device(s) detected"
                    )));
                }
                Ok(())
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_index;
                Err(BackendError::Unavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_default_and_unavailable_feature_messages_are_stable() {
        assert_eq!(BackendChoice::default(), BackendChoice::Auto);
        assert_eq!(
            validate_backend_choice(BackendChoice::Cpu, false, true)
                .unwrap_err()
                .to_string(),
            "CPU backend requires the `cpu` feature"
        );
        assert_eq!(
            validate_backend_choice(BackendChoice::Cuda { device_index: 0 }, true, false)
                .unwrap_err()
                .to_string(),
            "CUDA backend requires the `cuda` feature"
        );
        assert_eq!(
            validate_backend_choice(BackendChoice::Auto, false, false)
                .unwrap_err()
                .to_string(),
            "no compute backend is available; enable the `cpu` or `cuda` feature"
        );
    }
}
