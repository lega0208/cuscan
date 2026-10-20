//! CUDA backend implementation for fast-umap.
//!
//! Exposes CUDA device discovery helpers used by Standard UMAP backend selection.

/// Discover CUDA devices without treating unexpected driver errors as absence.
/// A missing driver library or the driver's explicit no-device result is
/// classified as unavailable; initialization and other discovery errors propagate.
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

/// Return the visible device count, propagating unexpected discovery failures.
pub fn device_count() -> usize {
    try_device_count().unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"))
}

/// Return whether at least one CUDA device is available.
pub fn is_available() -> bool {
    device_count() > 0
}

/// Return a CUDA device by index, or `None` when the driver is unavailable or the index is out of range.
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
}
