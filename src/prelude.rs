#[cfg(feature = "plotters")]
use crate::chart;
use crate::{serialize, train, utils};

// Re-export common utilities for easier use
#[cfg(feature = "plotters")]
pub use chart::{chart_tensor, chart_vector};

use num::Float;

// Re-export the new umap-rs style API
pub use crate::{
    EpochProgress, FittedUmap, GraphParams, LossReduction, ManifoldParams, Metric,
    OptimizationParams, Umap, UmapConfig,
};

// Re-export legacy types for backward compatibility
pub use train::{TrainingConfig, TrainingConfigBuilder};
pub use utils::generate_test_data;

/// Convenience function for fitting UMAP with the default Burn device.
///
/// This creates a `Umap` with default configuration (2-D output) and fits it
/// to the data.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample.
///
/// # Returns
/// A fitted UMAP model.
///
/// # Example
/// ```rust,ignore
/// let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
/// let model = umap(data);
/// ```
pub fn umap<F: Float>(data: Vec<Vec<F>>) -> FittedUmap
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    Umap::new(UmapConfig::default()).fit(data, None)
}

/// Convenience function for running UMAP with a custom output size.
///
/// # Arguments
/// * `data` - A vector of vectors, where each inner vector represents a data sample.
/// * `output_size` - Number of dimensions for the reduced output.
///
/// # Returns
/// A fitted UMAP model with the requested output dimensionality.
pub fn umap_size<F: Float>(data: Vec<Vec<F>>, output_size: usize) -> FittedUmap
where
    F: num::FromPrimitive + burn::tensor::Element,
{
    let config = UmapConfig {
        n_components: output_size,
        ..UmapConfig::default()
    };
    Umap::new(config).fit(data, None)
}
