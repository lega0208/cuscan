use crate::{BackendChoice, BackendError};
use burn::tensor::Device;

/// Resolve a requested backend using the calling facade's capabilities, not
/// the union of features enabled elsewhere in the Cargo dependency graph.
pub fn selected_device(
    choice: BackendChoice,
    cpu_allowed: bool,
    cuda_allowed: bool,
) -> Result<(Device, BackendChoice), BackendError> {
    let cuda_count = if cuda_allowed && choice != BackendChoice::Cpu {
        #[cfg(feature = "cuda")]
        {
            Some(crate::cuda_backend::try_device_count())
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    } else {
        None
    };
    resolve_backend_choice(choice, cpu_allowed, cuda_allowed, cuda_count)
}

fn resolve_backend_choice(
    choice: BackendChoice,
    cpu_allowed: bool,
    cuda_allowed: bool,
    cuda_count: Option<Result<usize, String>>,
) -> Result<(Device, BackendChoice), BackendError> {
    #[cfg(not(feature = "cuda"))]
    let _ = &cuda_count;
    match choice {
        BackendChoice::Auto => {
            if cuda_allowed {
                #[cfg(feature = "cuda")]
                match cuda_count.expect("CUDA permission requires discovery") {
                    Ok(count) if count > 0 => {
                        return Ok((Device::cuda(0), BackendChoice::Cuda { device_index: 0 }));
                    }
                    Ok(_) => {}
                    Err(error) => {
                        return Err(BackendError::Unavailable(format!(
                            "CUDA discovery failed: {error}"
                        )));
                    }
                }
                #[cfg(not(feature = "cuda"))]
                return Err(BackendError::Unavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ));
            }
            if cpu_allowed {
                #[cfg(feature = "cpu")]
                return Ok((construct_cpu_device(Device::cpu)?, BackendChoice::Cpu));
                #[cfg(not(feature = "cpu"))]
                return Err(BackendError::Unavailable(
                    "CPU backend requires the `cpu` feature".into(),
                ));
            }
            let message = if cuda_allowed {
                "CUDA is unavailable and CPU backend requires the `cpu` feature"
            } else {
                "no compute backend is available; enable the `cpu` or `cuda` feature"
            };
            Err(BackendError::Unavailable(message.into()))
        }
        BackendChoice::Cpu => {
            if !cpu_allowed {
                return Err(BackendError::Unavailable(
                    "CPU backend requires the `cpu` feature".into(),
                ));
            }
            #[cfg(feature = "cpu")]
            {
                Ok((construct_cpu_device(Device::cpu)?, BackendChoice::Cpu))
            }
            #[cfg(not(feature = "cpu"))]
            {
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
                let count = cuda_count
                    .expect("CUDA permission requires discovery")
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
    #[cfg(feature = "cuda")]
    fn resolution_distinguishes_cuda_presence_absence_and_discovery_failure() {
        let resolved =
            resolve_backend_choice(BackendChoice::Auto, true, true, Some(Ok(2))).unwrap();
        assert_eq!(resolved.1, BackendChoice::Cuda { device_index: 0 });
        #[cfg(feature = "cpu")]
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, true, true, Some(Ok(0)))
                .unwrap()
                .1,
            BackendChoice::Cpu
        );
        #[cfg(not(feature = "cpu"))]
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, true, true, Some(Ok(0)))
                .unwrap_err()
                .to_string(),
            "CPU backend requires the `cpu` feature"
        );
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, false, true, Some(Ok(0)))
                .unwrap_err()
                .to_string(),
            "CUDA is unavailable and CPU backend requires the `cpu` feature"
        );
        assert_eq!(
            resolve_backend_choice(
                BackendChoice::Auto,
                true,
                true,
                Some(Err("driver fault".into()))
            )
            .unwrap_err()
            .to_string(),
            "CUDA discovery failed: driver fault"
        );
    }

    #[test]
    fn caller_permissions_override_unified_common_features() {
        assert_eq!(
            resolve_backend_choice(BackendChoice::Cpu, false, true, Some(Ok(1)))
                .unwrap_err()
                .to_string(),
            "CPU backend requires the `cpu` feature"
        );
        assert_eq!(
            resolve_backend_choice(BackendChoice::Cuda { device_index: 0 }, true, false, None)
                .unwrap_err()
                .to_string(),
            "CUDA backend requires the `cuda` feature"
        );
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, false, false, None)
                .unwrap_err()
                .to_string(),
            "no compute backend is available; enable the `cpu` or `cuda` feature"
        );
        #[cfg(feature = "cpu")]
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, true, false, None)
                .unwrap()
                .1,
            BackendChoice::Cpu
        );
        #[cfg(feature = "cuda")]
        assert_eq!(
            resolve_backend_choice(BackendChoice::Auto, false, true, Some(Ok(1)))
                .unwrap()
                .1,
            BackendChoice::Cuda { device_index: 0 }
        );
    }

    #[test]
    fn explicit_cpu_selection_does_not_probe_cuda() {
        #[cfg(feature = "cpu")]
        assert_eq!(
            resolve_backend_choice(
                BackendChoice::Cpu,
                true,
                true,
                Some(Err("must not probe".into()))
            )
            .unwrap()
            .1,
            BackendChoice::Cpu
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn explicit_cuda_selection_rejects_out_of_range_index() {
        assert_eq!(
            resolve_backend_choice(
                BackendChoice::Cuda { device_index: 1 },
                true,
                true,
                Some(Ok(1))
            )
            .unwrap_err()
            .to_string(),
            "CUDA device index 1 is unavailable; 1 device(s) detected"
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
