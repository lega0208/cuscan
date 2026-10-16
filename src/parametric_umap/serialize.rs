use burn::{module::Module, tensor::Device};
use std::path::Path;

use super::model::ParametricUMAPModel;

/// Save a UMAP model's weights to a file.
///
/// # Arguments
///
/// * `model` - The UMAP model to save
/// * `path` - Path to the output file
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
/// use burn::tensor::Device;
/// let device = Device::cpu();
/// let config = ParametricUMAPConfig::default();
/// let umap = ParametricUMAP::with_device(config, device.clone());
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
/// ```
pub fn save_model(
    model: &ParametricUMAPModel,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    model
        .clone()
        .save_file(path)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Load a UMAP model's weights from a file.
///
/// # Arguments
///
/// * `model` - The UMAP model to load weights into
/// * `path` - Path to the input file
/// * `device` - The device where the model should be loaded
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
/// use burn::tensor::Device;
/// let device = Device::cpu();
/// let config = ParametricUMAPConfig::default();
/// let umap = ParametricUMAP::with_device(config, device.clone());
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
///
/// // Later...
/// let loaded_fitted = FittedParametricUMAP::load("model.umap", config, device)?;
/// ```
pub fn load_model(
    model: &mut ParametricUMAPModel,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<(), Box<dyn std::error::Error>> {
    *model = model
        .clone()
        .to_device(device)
        .try_load_file(path)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
    Ok(())
}
