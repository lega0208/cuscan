//! Error type shared by every HDBSCAN host stage.

/// Result type for HDBSCAN host stages.
pub type Result<T> = std::result::Result<T, HdbscanError>;

/// Errors produced by HDBSCAN validation, host stages, and backends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdbscanError {
    InvalidConfig(String),
    InvalidInput(String),
    BackendUnavailable(String),
    Numerical(String),
    Overflow(&'static str),
    /// Valid input exceeds a documented implementation limit; the message names the
    /// quantity, its value, the limit and the shape.
    Capacity(String),
    /// Available memory is insufficient for the requested operation.
    Memory(String),
}

impl std::fmt::Display for HdbscanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(message)
            | Self::InvalidInput(message)
            | Self::BackendUnavailable(message)
            | Self::Numerical(message)
            | Self::Memory(message)
            | Self::Capacity(message) => formatter.write_str(message),
            Self::Overflow(subject) => {
                write!(formatter, "{subject} exceeds the supported integer range")
            }
        }
    }
}

impl std::error::Error for HdbscanError {}
