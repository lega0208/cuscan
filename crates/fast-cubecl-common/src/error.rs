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
