use crate::{BackendChoice, BackendError};
use burn::tensor::Device;

/// Resolve a requested backend, discovering CUDA only when the choice needs it.
pub fn selected_device(choice: BackendChoice) -> Result<(Device, BackendChoice), BackendError> {
    let cuda_count = if choice != BackendChoice::Cpu {
        Some(crate::cuda_backend::try_device_count())
    } else {
        None
    };
    resolve_backend_choice(choice, cuda_count)
}

fn resolve_backend_choice(
    choice: BackendChoice,
    cuda_count: Option<Result<usize, String>>,
) -> Result<(Device, BackendChoice), BackendError> {
    match choice {
        BackendChoice::Auto => match cuda_count.expect("Auto choice requires CUDA discovery") {
            Ok(count) if count > 0 => {
                Ok((Device::cuda(0), BackendChoice::Cuda { device_index: 0 }))
            }
            Ok(_) => Ok((construct_cpu_device(Device::cpu)?, BackendChoice::Cpu)),
            Err(error) => Err(BackendError::Unavailable(format!(
                "CUDA discovery failed: {error}"
            ))),
        },
        BackendChoice::Cpu => Ok((construct_cpu_device(Device::cpu)?, BackendChoice::Cpu)),
        BackendChoice::Cuda { device_index } => {
            let count = cuda_count
                .expect("CUDA choice requires discovery")
                .map_err(|error| {
                    BackendError::Unavailable(format!("CUDA discovery failed: {error}"))
                })?;
            if device_index >= count {
                return Err(BackendError::Unavailable(format!(
                    "CUDA device index {device_index} is unavailable; {count} device(s) detected"
                )));
            }
            Ok((
                Device::cuda(device_index),
                BackendChoice::Cuda { device_index },
            ))
        }
    }
}

pub fn construct_cpu_device(make_device: impl FnOnce() -> Device) -> Result<Device, BackendError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(make_device)).map_err(|_| {
        BackendError::Unavailable(
            "CPU backend device construction failed; verify the `cpu` runtime is available".into(),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_distinguishes_cuda_presence_absence_and_discovery_failure() {
        let resolved = resolve_backend_choice(BackendChoice::Auto, Some(Ok(2))).unwrap();
        assert_eq!(resolved.1, BackendChoice::Cuda { device_index: 0 });
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, Some(Ok(0)))
                .unwrap()
                .1,
            BackendChoice::Cpu
        );
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, Some(Err("driver fault".into())))
                .unwrap_err()
                .to_string(),
            "CUDA discovery failed: driver fault"
        );
    }

    #[test]
    fn explicit_cpu_selection_does_not_probe_cuda() {
        assert_eq!(
            resolve_backend_choice(BackendChoice::Cpu, None).unwrap().1,
            BackendChoice::Cpu
        );
    }

    #[test]
    fn explicit_cuda_selection_rejects_out_of_range_index() {
        assert_eq!(
            resolve_backend_choice(BackendChoice::Cuda { device_index: 1 }, Some(Ok(1)))
                .unwrap_err()
                .to_string(),
            "CUDA device index 1 is unavailable; 1 device(s) detected"
        );
    }

    #[test]
    fn explicit_cuda_selection_rejects_absence_and_discovery_failure() {
        let choice = BackendChoice::Cuda { device_index: 0 };
        assert_eq!(
            resolve_backend_choice(choice, Some(Ok(0)))
                .unwrap_err()
                .to_string(),
            "CUDA device index 0 is unavailable; 0 device(s) detected"
        );
        assert_eq!(
            resolve_backend_choice(choice, Some(Err("driver fault".into())))
                .unwrap_err()
                .to_string(),
            "CUDA discovery failed: driver fault"
        );
    }

    #[test]
    fn cpu_device_construction_failure_is_reported_without_fallback() {
        assert_eq!(
            construct_cpu_device(|| panic!("injected CPU failure"))
                .unwrap_err()
                .to_string(),
            "CPU backend device construction failed; verify the `cpu` runtime is available"
        );
    }
}
