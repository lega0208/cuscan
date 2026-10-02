use burn::{module::Module, tensor::Device};
use std::path::Path;

use crate::model::UMAPModel;

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
/// let device = Device::flex();
/// let config = UmapConfig::default();
/// let umap = Umap::with_device(config, device.clone());
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
/// ```
pub fn save_model(
    model: &UMAPModel,
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
/// let device = Device::flex();
/// let config = UmapConfig::default();
/// let umap = Umap::with_device(config, device.clone());
/// let fitted = umap.fit(data, None);
/// fitted.save("model.umap").expect("Failed to save model");
///
/// // Later...
/// let loaded_fitted = FittedUmap::load("model.umap", config, device)?;
/// ```
pub fn load_model(
    model: &mut UMAPModel,
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
