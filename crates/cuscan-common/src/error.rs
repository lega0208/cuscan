use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    Unavailable(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for BackendError {}

/// Failure to validate an approximate-neighbor launch before dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnError {
    InvalidInput(String),
    Capability(String),
    Workspace(String),
    Overflow(String),
    Memory {
        required_bytes: u64,
        available_bytes: u64,
        message: String,
    },
}

impl fmt::Display for AnnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message)
            | Self::Capability(message)
            | Self::Workspace(message)
            | Self::Overflow(message) => f.write_str(message),
            Self::Memory { message, .. } => f.write_str(message),
        }
    }
}

impl std::error::Error for AnnError {}

#[cfg(test)]
mod tests {
    use super::AnnError;

    #[test]
    fn ann_memory_error_preserves_bytes_and_message() {
        let error = AnnError::Memory {
            required_bytes: 123,
            available_bytes: 45,
            message: "insufficient ANN memory".into(),
        };
        assert_eq!(error.to_string(), "insufficient ANN memory");
        assert!(matches!(
            error,
            AnnError::Memory {
                required_bytes: 123,
                available_bytes: 45,
                ..
            }
        ));
    }
}
