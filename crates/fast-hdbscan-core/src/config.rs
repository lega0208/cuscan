//! Backend-free HDBSCAN extraction configuration (S1).

use crate::error::{HdbscanError, Result};

/// Flat cluster extraction method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ClusterSelectionMethod {
    /// Excess of mass.
    #[default]
    Eom,
    /// Leaves of the condensed cluster tree.
    Leaf,
}

/// HDBSCAN configuration shared by every backend. Backend choice is not part of this type.
#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanConfig {
    pub min_cluster_size: usize,
    /// `None` resolves to `min_cluster_size`.
    pub min_samples: Option<usize>,
    pub cluster_selection_method: ClusterSelectionMethod,
    pub allow_single_cluster: bool,
    /// `None` stands for upstream's `np.inf`.
    pub max_cluster_size: Option<usize>,
    pub cluster_selection_epsilon: f64,
    pub cluster_selection_persistence: f64,
}

impl Default for HdbscanConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 5,
            min_samples: None,
            cluster_selection_method: ClusterSelectionMethod::Eom,
            allow_single_cluster: false,
            max_cluster_size: None,
            cluster_selection_epsilon: 0.0,
            cluster_selection_persistence: 0.0,
        }
    }
}

impl HdbscanConfig {
    /// Reject non-positive sizes and non-finite or negative selection thresholds.
    pub fn validate(&self) -> Result<()> {
        if self.min_cluster_size == 0 {
            return Err(HdbscanError::InvalidConfig(
                "min_cluster_size must be positive".into(),
            ));
        }
        if self.min_samples == Some(0) {
            return Err(HdbscanError::InvalidConfig(
                "min_samples must be positive".into(),
            ));
        }
        for (name, value) in [
            ("cluster_selection_epsilon", self.cluster_selection_epsilon),
            (
                "cluster_selection_persistence",
                self.cluster_selection_persistence,
            ),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(HdbscanError::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    /// `min_samples`, defaulting to `min_cluster_size`.
    pub fn resolved_min_samples(&self) -> usize {
        self.min_samples.unwrap_or(self.min_cluster_size)
    }
}
