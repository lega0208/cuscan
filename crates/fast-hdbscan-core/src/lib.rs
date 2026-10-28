//! Backend-free HDBSCAN host extraction and exact MST reference infrastructure.
//!
//! Host extraction consumes a supplied spanning tree; it never selects or executes a backend.

mod argsort;
pub mod condense;
pub mod config;
pub mod dsu;
pub mod error;
pub mod finite;
pub mod kdtree;
pub mod knn_kruskal;
pub mod labels;
pub mod linkage;
pub mod mst;
#[cfg(feature = "test-support")]
pub mod oracle;
pub mod prediction;
pub mod select;

pub use condense::CondensedTree;
pub use config::{ClusterSelectionMethod, HdbscanConfig, MstGraph};
pub use error::HdbscanError;
pub use kdtree::KdTree;
pub use linkage::{LinkageRow, SortedMstEdge};
pub use prediction::{PointPrediction, PredictionColumns, PredictionHierarchy};

/// Finite-row-indexed host extraction results, before raw-row remapping.
#[derive(Debug, Clone, PartialEq)]
pub struct HostClustering {
    pub labels: Vec<i32>,
    pub probabilities: Vec<f32>,
    pub linkage: Vec<LinkageRow>,
    pub condensed: CondensedTree,
    pub sorted_mst: Vec<SortedMstEdge>,
}

impl HostClustering {
    /// Contract for exactly one finite input row: all noise with empty trees.
    pub fn one_finite() -> Self {
        Self {
            labels: vec![-1],
            probabilities: vec![0.0],
            linkage: Vec::new(),
            condensed: CondensedTree::default(),
            sorted_mst: Vec::new(),
        }
    }
}

/// Run S8–S12 on a supplied MST of `edges.len() + 1` finite points.
///
/// Empty edges describe the one-finite-row contract, not an empty raw input.
/// Callers validate and split raw input before using this host-stage entry point.
pub fn clusters_from_spanning_tree(
    edges: &[SortedMstEdge],
    config: &HdbscanConfig,
) -> Result<HostClustering, HdbscanError> {
    config.validate()?;
    if edges.is_empty() {
        return Ok(HostClustering::one_finite());
    }
    let n_points = edges
        .len()
        .checked_add(1)
        .ok_or(HdbscanError::Overflow("MST point count"))?;
    if n_points > i32::MAX as usize {
        return Err(HdbscanError::Overflow("point index"));
    }
    let mut sorted_mst = edges.to_vec();
    linkage::sort_mst(&mut sorted_mst);
    let linkage = linkage::mst_to_linkage(&sorted_mst)?;
    let mut condensed = condense::condense_tree(&linkage, config.min_cluster_size);
    if config.cluster_selection_persistence > 0.0 && !condensed.is_empty() {
        condensed = condense::simplify_hierarchy(condensed, config.cluster_selection_persistence);
    }
    let selected = select::select_clusters(&condensed, n_points, config);
    let labels = labels::get_cluster_label_vector(
        &condensed,
        &selected,
        config.cluster_selection_epsilon,
        n_points,
    );
    let mut sorted_selected = selected;
    sorted_selected.sort_unstable();
    let probabilities =
        labels::membership_strengths(&condensed, &sorted_selected, &labels, n_points);
    Ok(HostClustering {
        labels,
        probabilities,
        linkage,
        condensed,
        sorted_mst,
    })
}
