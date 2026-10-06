//! Convenience exports for the parametric estimator and shared graph settings.

#[cfg(feature = "plotters")]
pub use crate::chart::{chart_tensor, chart_vector};
pub use crate::utils::generate_test_data;
pub use crate::{
    FittedParametricUMAP, GraphParams, ManifoldParams, Metric, ParametricUMAP,
    ParametricUMAPConfig, ParametricUMAPEpochProgress, ParametricUMAPLossReduction,
    ParametricUMAPOptimizationParams,
};
use num::Float;

/// Fit a parametric embedding using default settings and the default Burn device.
pub fn parametric_umap<F>(data: Vec<Vec<F>>) -> FittedParametricUMAP
where
    F: Float + num::FromPrimitive + burn::tensor::Element,
{
    ParametricUMAP::new(ParametricUMAPConfig::default()).fit(data, None)
}

/// Fit a parametric embedding with the requested output dimensionality.
pub fn parametric_umap_size<F>(data: Vec<Vec<F>>, output_size: usize) -> FittedParametricUMAP
where
    F: Float + num::FromPrimitive + burn::tensor::Element,
{
    let config = ParametricUMAPConfig {
        n_components: output_size,
        ..Default::default()
    };
    ParametricUMAP::new(config).fit(data, None)
}
