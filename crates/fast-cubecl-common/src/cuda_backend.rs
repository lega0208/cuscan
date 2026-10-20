//! CUDA discovery helpers shared by CubeCL consumers.

/// A missing runtime and the driver's explicit no-device result are absence;
/// all other driver failures are surfaced to backend selection.
#[cfg(feature = "cuda")]
pub fn try_device_count() -> Result<usize, String> {
    // SAFETY: This only checks whether the CUDA driver library can be loaded.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return Ok(0);
    }
    match cudarc::driver::CudaContext::device_count() {
        Ok(count) => {
            usize::try_from(count).map_err(|_| format!("invalid CUDA device count: {count}"))
        }
        Err(error) if error.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_NO_DEVICE => Ok(0),
        Err(error) => Err(error.to_string()),
    }
}

#[cfg(feature = "cuda")]
pub fn device_count() -> usize {
    try_device_count().unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"))
}

#[cfg(feature = "cuda")]
pub fn is_available() -> bool {
    device_count() > 0
}

#[cfg(feature = "cuda")]
pub fn device(index: usize) -> Option<burn::tensor::Device> {
    (index < device_count()).then(|| burn::tensor::Device::cuda(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_range_device_is_none() {
        assert!(device(usize::MAX).is_none());
    }

    #[test]
    fn required_cuda_discovery_has_a_usable_device() {
        if std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref() == Ok("1") {
            assert!(try_device_count().expect("required CUDA discovery failed") > 0);
            assert!(device(0).is_some());
        }
    }
}
