//! UMAP-local error mapping for shared backend selection.
use super::{Result, UmapError};
pub use fast_cubecl_common::BackendChoice;

pub(super) fn map_backend_error(error: fast_cubecl_common::BackendError) -> UmapError {
    UmapError::BackendUnavailable(error.to_string())
}

pub fn validate_backend_choice(choice: BackendChoice) -> Result<()> {
    fast_cubecl_common::validate_backend_choice(
        choice,
        cfg!(feature = "cpu"),
        cfg!(feature = "cuda"),
    )
    .map_err(map_backend_error)
}

#[cfg(test)]
mod mapping_tests {
    use super::*;
    use fast_cubecl_common::BackendError;

    #[test]
    fn backend_errors_preserve_exact_facade_display_messages() {
        let cases = [
            (
                BackendError::Unavailable("CPU backend requires the `cpu` feature".into()),
                "CPU backend requires the `cpu` feature",
            ),
            (
                BackendError::Unavailable("CUDA backend requires the `cuda` feature".into()),
                "CUDA backend requires the `cuda` feature",
            ),
            (
                BackendError::Unavailable("CUDA discovery failed: driver fault".into()),
                "CUDA discovery failed: driver fault",
            ),
            (
                BackendError::Unavailable(
                    "CUDA device index 3 is unavailable; 2 device(s) detected".into(),
                ),
                "CUDA device index 3 is unavailable; 2 device(s) detected",
            ),
            (
                BackendError::Unavailable(
                    "no compute backend is available; enable the `cpu` or `cuda` feature".into(),
                ),
                "no compute backend is available; enable the `cpu` or `cuda` feature",
            ),
        ];

        for (backend_error, expected_message) in cases {
            let facade_error = map_backend_error(backend_error);
            assert!(matches!(facade_error, UmapError::BackendUnavailable(_)));
            assert_eq!(facade_error.to_string(), expected_message);
        }
    }
}

#[cfg(all(test, not(all(feature = "cpu", feature = "cuda"))))]
mod tests {
    use super::*;

    #[test]
    fn facade_permissions_are_not_expanded_by_peer_features() {
        #[cfg(not(feature = "cpu"))]
        assert_eq!(
            validate_backend_choice(BackendChoice::Cpu)
                .unwrap_err()
                .to_string(),
            "CPU backend requires the `cpu` feature"
        );
        #[cfg(not(feature = "cuda"))]
        assert_eq!(
            validate_backend_choice(BackendChoice::Cuda { device_index: 0 })
                .unwrap_err()
                .to_string(),
            "CUDA backend requires the `cuda` feature"
        );
    }
}
