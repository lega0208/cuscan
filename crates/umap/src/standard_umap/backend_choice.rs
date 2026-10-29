//! UMAP-local error mapping for shared backend selection.
use super::{Result, UmapError};
pub use cuscan_common::BackendChoice;

pub(super) fn map_backend_error(error: cuscan_common::BackendError) -> UmapError {
    UmapError::BackendUnavailable(error.to_string())
}

pub fn validate_backend_choice(choice: BackendChoice) -> Result<()> {
    cuscan_common::validate_backend_choice(choice).map_err(map_backend_error)
}

#[cfg(test)]
mod mapping_tests {
    use super::*;
    use cuscan_common::BackendError;

    #[test]
    fn backend_errors_preserve_exact_facade_display_messages() {
        let cases = [
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
        ];

        for (backend_error, expected_message) in cases {
            let facade_error = map_backend_error(backend_error);
            assert!(matches!(facade_error, UmapError::BackendUnavailable(_)));
            assert_eq!(facade_error.to_string(), expected_message);
        }
    }
}
