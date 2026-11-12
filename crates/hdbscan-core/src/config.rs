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

/// Graph used to construct the mutual-reachability spanning tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MstGraph {
    /// Use the kNN graph above 30 features, otherwise the complete graph.
    #[default]
    Auto,
    Complete,
    KnnGraph,
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
    pub mst_graph: MstGraph,
    /// Nonself graph width; ignored for the complete graph.
    pub knn_k: Option<usize>,
    /// Use independent bounds on the complete-graph KD route.
    pub reproducible: bool,
    /// Optional upper bound on ANN memory usage in bytes; must be positive.
    pub memory_limit_bytes: Option<u64>,
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
            mst_graph: MstGraph::Auto,
            knn_k: None,
            reproducible: false,
            memory_limit_bytes: None,
        }
    }
}

impl HdbscanConfig {
    /// Reject non-positive sizes and non-finite or negative selection thresholds.
    pub fn validate(&self) -> Result<()> {
        if self.memory_limit_bytes == Some(0) {
            return Err(HdbscanError::InvalidInput(
                "memory_limit_bytes must be greater than zero".into(),
            ));
        }
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
        if self.knn_k == Some(0) {
            return Err(HdbscanError::InvalidConfig("knn_k must be positive".into()));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_memory_limit() {
        let mut config = HdbscanConfig::default();
        assert_eq!(config.memory_limit_bytes, None);
        assert!(config.validate().is_ok());
        config.memory_limit_bytes = Some(1);
        assert!(config.validate().is_ok());
        config.memory_limit_bytes = Some(0);
        assert_eq!(
            config.validate(),
            Err(HdbscanError::InvalidInput(
                "memory_limit_bytes must be greater than zero".into()
            ))
        );
    }
}
