use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::graph::{GraphParams, ManifoldParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParametricUMAPLossReduction {
    Mean,
    Sum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricUMAPOptimizationParams {
    #[serde(default = "default_seed")]
    pub seed: u64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub penalty: f32,
    pub repulsion_strength: f32,
    pub patience: Option<i32>,
    pub loss_reduction: ParametricUMAPLossReduction,
    pub min_desired_loss: Option<f64>,
    pub timeout: Option<u64>,
    pub verbose: bool,
    pub neg_sample_rate: usize,
    pub cooldown_ms: u64,
    #[serde(skip)]
    pub figures_dir: Option<PathBuf>,
}

impl Default for ParametricUMAPOptimizationParams {
    fn default() -> Self {
        Self {
            seed: default_seed(),
            n_epochs: 100,
            batch_size: 1000,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            penalty: 1e-5,
            repulsion_strength: 1.0,
            patience: None,
            loss_reduction: ParametricUMAPLossReduction::Sum,
            min_desired_loss: None,
            timeout: None,
            verbose: false,
            neg_sample_rate: 5,
            cooldown_ms: 0,
            figures_dir: None,
        }
    }
}

fn default_seed() -> u64 {
    9999
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricUMAPConfig {
    pub n_components: usize,
    pub hidden_sizes: Vec<usize>,
    pub manifold: ManifoldParams,
    pub graph: GraphParams,
    pub optimization: ParametricUMAPOptimizationParams,
}

impl Default for ParametricUMAPConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            hidden_sizes: vec![100],
            manifold: ManifoldParams::default(),
            graph: GraphParams::default(),
            optimization: ParametricUMAPOptimizationParams::default(),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TrainerConfig {
    pub seed: u64,
    pub epochs: usize,
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub penalty: f32,
    pub verbose: bool,
    pub patience: Option<i32>,
    pub k_neighbors: usize,
    pub local_connectivity: f64,
    pub bandwidth: f64,
    pub set_op_mix_ratio: f64,
    pub min_desired_loss: Option<f64>,
    pub timeout: Option<u64>,
    pub repulsion_strength: f32,
    pub kernel_a: f32,
    pub kernel_b: f32,
    pub neg_sample_rate: usize,
    pub cooldown_ms: u64,
    #[cfg(feature = "plotters")]
    pub figures_dir: Option<PathBuf>,
}
