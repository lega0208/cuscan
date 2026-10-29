use crate::{BackendError, cuda_backend};

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
        validate_backend_choice(self)
    }
}

pub fn validate_backend_choice(choice: BackendChoice) -> Result<(), BackendError> {
    match choice {
        BackendChoice::Auto => {
            // CPU construction is intentionally deferred to selected_device.
            cuda_backend::try_device_count()
                .map_err(|e| BackendError::Unavailable(format!("CUDA discovery failed: {e}")))?;
            Ok(())
        }
        BackendChoice::Cpu => Ok(()),
        BackendChoice::Cuda { device_index } => {
            let count = cuda_backend::try_device_count()
                .map_err(|e| BackendError::Unavailable(format!("CUDA discovery failed: {e}")))?;
            if device_index >= count {
                return Err(BackendError::Unavailable(format!(
                    "CUDA device index {device_index} is unavailable; {count} device(s) detected"
                )));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn choice_default_and_validation() {
        assert_eq!(BackendChoice::default(), BackendChoice::Auto);
        assert!(validate_backend_choice(BackendChoice::Cpu).is_ok());
    }
}
