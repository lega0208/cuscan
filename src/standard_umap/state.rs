use super::{NeighborSearch, UMAPConfig};

/// Storage-independent snapshot of a fitted model. All fields are untrusted on import.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelState {
    pub config: UMAPConfig,
    pub requested_search: NeighborSearch,
    pub resolved_search: NeighborSearch,
    pub a: f32,
    pub b: f32,
    pub training_features: Vec<f32>,
    pub training_rows: usize,
    pub n_features: usize,
    pub embedding: Vec<f32>,
    pub embedding_rows: usize,
    pub n_components: usize,
    pub query_graph: Option<Vec<i32>>,
    pub query_graph_width: usize,
    pub original_to_training: Option<Vec<usize>>,
}
