#![cfg(feature = "parametric")]
#![allow(unused_variables)]
#![allow(unused_imports)]

//! # fast-umap
//!
//! GPU-accelerated parametric UMAP (Uniform Manifold Approximation and Projection)
//! in Rust, built on [burn](https://github.com/tracel-ai/burn) +
//! [CubeCL](https://github.com/tracel-ai/cubecl).
//!
//! **Up to 4.7× faster** than [umap-rs](https://crates.io/crates/umap-rs) on
//! datasets ≥ 10 000 samples, with the ability to [`transform()`](FittedParametricUMAP::transform)
//! new unseen data (something classical UMAP cannot do).
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use fast_umap::prelude::*;
//!
//! // 100 samples × 10 features
//! let data: Vec<Vec<f64>> = generate_test_data(100, 10)
//!     .chunks(10)
//!     .map(|c| c.iter().map(|&x: &f32| x as f64).collect())
//!     .collect();
//!
//! // Configure and fit UMAP
//! let config = ParametricUMAPConfig::default();
//! // let umap = ParametricUMAP::new(config);
//! // let fitted = umap.fit(data.clone(), None);
//! // let embedding = fitted.embedding();
//! // let new_embedding = fitted.transform(new_data);
//! // fitted.save("model.umap")?; // Save trained model
//! // let loaded = FittedParametricUMAP::load("model.umap", config, input_size, device)?;
//! ```
//!
//! ## Interface
//!
//! The public API mirrors the [`umap-rs`](https://crates.io/crates/umap-rs) crate:
//!
//! * [`ParametricUMAP`] — Main algorithm struct, created via `ParametricUMAP::new(config)`
//! * [`FittedParametricUMAP`] — Fitted model returned from `umap.fit(data)`, with
//!   [`embedding()`](FittedParametricUMAP::embedding),
//!   [`into_embedding()`](FittedParametricUMAP::into_embedding),
//!   [`transform()`](FittedParametricUMAP::transform), and
//!   [`config()`](FittedParametricUMAP::config)
//! * [`ParametricUMAPConfig`] — Configuration with nested [`GraphParams`] and [`ParametricUMAPOptimizationParams`]
//! * [`Metric`] — Distance metric enum (`Euclidean`, `Manhattan`, `Cosine`, …)
//!
//! ## Performance
//!
//! | Dataset | fast-umap | umap-rs | Speedup |
//! |---------|-----------|---------|---------|
//! | 5 000 × 100 | 6.75s | 2.31s | 0.34× *(umap-rs faster)* |
//! | 10 000 × 100 | 5.93s | 8.68s | **1.5× faster** |
//! | 20 000 × 100 | 7.32s | 34.10s | **4.7× faster** |
//!
//! Benchmarked on Apple M3 Max. Reproduce with
//! `cargo run --release --example crate_comparison`.
//!
//! ## Architecture
//!
//! The dimensionality reduction is performed by a small feed-forward neural
//! network (`ParametricUMAPModel`) trained with the UMAP cross-entropy loss using
//! sparse edge subsampling and negative sampling:
//!
//! ```text
//! attraction  =  mean_{sampled k-NN edges}   [ −log q_ij ]
//! repulsion   =  mean_{negative samples}     [ −log (1 − q_ij) ]
//! loss        =  attraction  +  repulsion_strength × repulsion
//! ```
//!
//! where `q_ij = 1 / (1 + a · d_ij^(2b))` is the UMAP kernel applied to
//! embedding distances (`a` and `b` are fitted from `min_dist` / `spread`).
//! Per-epoch cost is O(min(n·k, 50K)) regardless of dataset size.
//!
//! Training and weight persistence are private implementation details of this
//! module; reusable graph and distance operations live in [`crate::graph`].

use burn::module::Module;
use burn::tensor::{Device, Tensor};
use crossbeam_channel::Receiver;
use num::Float;

use self::config::TrainerConfig;
use self::trainer::train_sparse;
use crate::graph::{GraphParams, fit_ab};
use crate::utils::{convert_tensor_to_vector, convert_vector_to_tensor};

mod config;
pub mod model;
pub mod serialize;
mod trainer;

pub use config::{
    ParametricUMAPConfig, ParametricUMAPLossReduction, ParametricUMAPOptimizationParams,
};
pub use model::{ParametricUMAPModel, ParametricUMAPModelConfig, ParametricUMAPModelConfigBuilder};
pub use trainer::ParametricUMAPEpochProgress;

/// Validate data dimensions and model settings before fitting.
pub fn validate_parametric_umap_input(
    config: &ParametricUMAPConfig,
    n_samples: usize,
    n_features: usize,
) -> Result<(), String> {
    if n_samples == 0 {
        return Err("n_samples must be greater than zero".to_string());
    }
    if n_features == 0 {
        return Err("n_features must be greater than zero".to_string());
    }
    if config.n_components == 0 {
        return Err("n_components must be greater than zero".to_string());
    }
    if config.graph.n_neighbors < 2 {
        return Err("n_neighbors must be at least two (including self)".to_string());
    }
    if n_samples <= config.graph.n_neighbors {
        return Err("n_samples must be greater than n_neighbors".to_string());
    }
    if !config.graph.local_connectivity.is_finite() || config.graph.local_connectivity < 0.0 {
        return Err("local_connectivity must be finite and nonnegative".to_string());
    }
    if !config.graph.bandwidth.is_finite() || config.graph.bandwidth <= 0.0 {
        return Err("bandwidth must be finite and greater than zero".to_string());
    }
    if !config.graph.set_op_mix_ratio.is_finite()
        || !(0.0..=1.0).contains(&config.graph.set_op_mix_ratio)
    {
        return Err("set_op_mix_ratio must be finite and between zero and one".to_string());
    }
    if config.optimization.n_epochs == 0 {
        return Err("n_epochs must be greater than zero".to_string());
    }
    if config.hidden_sizes.is_empty() {
        return Err("hidden_sizes must not be empty".to_string());
    }
    Ok(())
}

/// Validate that fitting data is non-empty, rectangular, and compatible with
/// the requested model configuration.
pub fn validate_parametric_umap_data<F>(
    config: &ParametricUMAPConfig,
    data: &[Vec<F>],
) -> Result<(), String> {
    let n_features = data
        .first()
        .ok_or_else(|| "n_samples must be greater than zero".to_string())?
        .len();
    if let Some((row, values)) = data
        .iter()
        .enumerate()
        .find(|(_, values)| values.len() != n_features)
    {
        return Err(format!(
            "row {row} has {} features; expected {n_features}",
            values.len()
        ));
    }
    validate_parametric_umap_input(config, data.len(), n_features)
}

/// UMAP dimensionality reduction algorithm (GPU-accelerated, parametric).
///
/// This struct holds the configuration for UMAP. Use [`ParametricUMAP::new`] to create
/// an instance, then call [`fit`](ParametricUMAP::fit) to train and obtain a
/// [`FittedParametricUMAP`].
///
/// The interface mirrors [`umap-rs`](https://crates.io/crates/umap-rs), but
/// this implementation uses a parametric neural network trained on a GPU,
/// which means:
/// - It can [`transform`](FittedParametricUMAP::transform) new (unseen) data
/// - It leverages GPU acceleration via burn/CubeCL
/// - It does **not** require precomputed KNN (computed internally per batch)
///
/// # Example
///
/// ```ignore
/// use fast_umap::prelude::*;
///
/// let config = ParametricUMAPConfig {
///     n_components: 2,
///     graph: GraphParams {
///         n_neighbors: 15,
///         ..Default::default()
///     },
///     optimization: ParametricUMAPOptimizationParams {
///         n_epochs: 200,
///         ..Default::default()
///     },
///     ..Default::default()
/// };
///
/// let umap = ParametricUMAP::new(config);
/// let fitted = umap.fit(data, None);
/// let embedding = fitted.embedding();
/// ```
pub struct ParametricUMAP {
    config: ParametricUMAPConfig,
    device: Device,
}

impl ParametricUMAP {
    /// Create a new UMAP instance with the given configuration.
    ///
    /// Uses the default device for the backend.
    ///
    /// # Arguments
    ///
    /// * `config` - UMAP configuration parameters
    pub fn new(config: ParametricUMAPConfig) -> Self {
        Self {
            config,
            device: Device::default().autodiff(),
        }
    }

    /// Create a new UMAP instance with a specific device.
    ///
    /// # Arguments
    ///
    /// * `config` - UMAP configuration parameters
    /// * `device` - The device (CPU or GPU) on which to perform computation
    pub fn with_device(config: ParametricUMAPConfig, device: Device) -> Self {
        Self {
            config,
            device: device.autodiff(),
        }
    }

    /// Get a reference to the configuration.
    pub fn config(&self) -> &ParametricUMAPConfig {
        &self.config
    }

    /// Fit the UMAP model to the given data.
    ///
    /// Trains a parametric neural network to learn a low-dimensional embedding
    /// of the input data. Features are used as provided, without automatic
    /// scaling; apply any desired preprocessing consistently before fitting and
    /// transforming. The model can then transform new data.
    ///
    /// # Arguments
    ///
    /// * `data` - Input data as a vector of vectors (n_samples × n_features)
    /// * `labels` - Optional per-sample labels for coloured epoch snapshots
    ///
    /// # Returns
    ///
    /// A [`FittedParametricUMAP`] containing the trained model and embedding.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = ParametricUMAPConfig::default();
    /// let umap = ParametricUMAP::new(config);
    /// let fitted = umap.fit(data, None);
    /// let embedding = fitted.embedding();
    /// ```
    pub fn fit<F: Float>(
        self,
        data: Vec<Vec<F>>,
        labels: Option<Vec<String>>,
    ) -> FittedParametricUMAP
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        let (exit_tx, exit_rx) = crossbeam_channel::unbounded();

        ctrlc::set_handler(move || {
            let _ = exit_tx.send(());
        })
        .ok(); // Ignore error if handler already set

        self.fit_with_signal(data, labels, exit_rx)
    }

    /// Fit the UMAP model with an external cancellation signal.
    ///
    /// Same as [`fit`](ParametricUMAP::fit) but accepts an external `Receiver<()>` for
    /// graceful cancellation (e.g., from Ctrl-C).
    ///
    /// # Arguments
    ///
    /// * `data` - Input data as a vector of vectors (n_samples × n_features)
    /// * `labels` - Optional per-sample labels for coloured epoch snapshots
    /// * `exit_rx` - Channel receiver for cancellation signal
    ///
    /// # Returns
    ///
    /// A [`FittedParametricUMAP`] containing the trained model and embedding.
    pub fn fit_with_signal<F: Float>(
        self,
        data: Vec<Vec<F>>,
        labels: Option<Vec<String>>,
        exit_rx: Receiver<()>,
    ) -> FittedParametricUMAP
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        validate_parametric_umap_data(&self.config, &data)
            .unwrap_or_else(|error| panic!("invalid fit input: {error}"));
        let default_name = "model";
        let num_samples = data.len();
        let num_features = data[0].len();
        let seed = self.config.optimization.seed;
        self.device.seed(seed);

        // Flatten input data
        let train_data: Vec<F> = data.into_iter().flatten().collect();

        // Build model config
        let model_config = ParametricUMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_sizes(self.config.hidden_sizes.clone())
            .output_size(self.config.n_components)
            .build()
            .unwrap();

        let model = ParametricUMAPModel::new(&model_config, &self.device);

        // Build training config from ParametricUMAPConfig
        let (kernel_a, kernel_b) =
            fit_ab(self.config.manifold.min_dist, self.config.manifold.spread);
        let training_config = TrainerConfig {
            seed,
            epochs: self.config.optimization.n_epochs,
            learning_rate: self.config.optimization.learning_rate,
            beta1: self.config.optimization.beta1,
            beta2: self.config.optimization.beta2,
            penalty: self.config.optimization.penalty,
            verbose: self.config.optimization.verbose,
            patience: self.config.optimization.patience,
            k_neighbors: self.config.graph.n_neighbors,
            local_connectivity: self.config.graph.local_connectivity,
            bandwidth: self.config.graph.bandwidth,
            set_op_mix_ratio: self.config.graph.set_op_mix_ratio,
            min_desired_loss: self.config.optimization.min_desired_loss,
            timeout: self.config.optimization.timeout,
            repulsion_strength: self.config.optimization.repulsion_strength,
            kernel_a,
            kernel_b,
            neg_sample_rate: self.config.optimization.neg_sample_rate,
            cooldown_ms: self.config.optimization.cooldown_ms,
            #[cfg(feature = "plotters")]
            figures_dir: self.config.optimization.figures_dir.clone(),
        };

        // Use the sparse training path (O(n·k) per epoch) by default.
        let (model, _losses, _best_loss): (ParametricUMAPModel, Vec<F>, F) = train_sparse(
            default_name,
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &training_config,
            self.device.clone(),
            exit_rx,
            labels,
            None,
        );

        Self::finalize(
            model,
            train_data,
            num_samples,
            num_features,
            self.device,
            self.config,
        )
    }

    /// Like [`fit_with_signal`], but also accepts a progress callback that is
    /// invoked every few epochs with timing and loss information.
    pub fn fit_with_progress<F: Float>(
        self,
        data: Vec<Vec<F>>,
        labels: Option<Vec<String>>,
        exit_rx: Receiver<()>,
        on_progress: Box<dyn Fn(ParametricUMAPEpochProgress) + Send>,
    ) -> FittedParametricUMAP
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        validate_parametric_umap_data(&self.config, &data)
            .unwrap_or_else(|error| panic!("invalid fit input: {error}"));
        let default_name = "model";
        let num_samples = data.len();
        let num_features = data[0].len();
        let seed = self.config.optimization.seed;
        self.device.seed(seed);

        let train_data: Vec<F> = data.into_iter().flatten().collect();

        let model_config = ParametricUMAPModelConfigBuilder::default()
            .input_size(num_features)
            .hidden_sizes(self.config.hidden_sizes.clone())
            .output_size(self.config.n_components)
            .build()
            .unwrap();

        let model = ParametricUMAPModel::new(&model_config, &self.device);

        let (kernel_a, kernel_b) =
            fit_ab(self.config.manifold.min_dist, self.config.manifold.spread);
        let training_config = TrainerConfig {
            seed,
            epochs: self.config.optimization.n_epochs,
            learning_rate: self.config.optimization.learning_rate,
            beta1: self.config.optimization.beta1,
            beta2: self.config.optimization.beta2,
            penalty: self.config.optimization.penalty,
            verbose: self.config.optimization.verbose,
            patience: self.config.optimization.patience,
            k_neighbors: self.config.graph.n_neighbors,
            local_connectivity: self.config.graph.local_connectivity,
            bandwidth: self.config.graph.bandwidth,
            set_op_mix_ratio: self.config.graph.set_op_mix_ratio,
            min_desired_loss: self.config.optimization.min_desired_loss,
            timeout: self.config.optimization.timeout,
            repulsion_strength: self.config.optimization.repulsion_strength,
            kernel_a,
            kernel_b,
            neg_sample_rate: self.config.optimization.neg_sample_rate,
            cooldown_ms: self.config.optimization.cooldown_ms,
            #[cfg(feature = "plotters")]
            figures_dir: self.config.optimization.figures_dir.clone(),
        };

        let (model, _losses, _best_loss): (ParametricUMAPModel, Vec<F>, F) = train_sparse(
            default_name,
            model,
            num_samples,
            num_features,
            train_data.clone(),
            &training_config,
            self.device.clone(),
            exit_rx,
            labels,
            Some(on_progress),
        );

        Self::finalize(
            model,
            train_data,
            num_samples,
            num_features,
            self.device,
            self.config,
        )
    }

    /// Shared finalization: extract embedding from trained model.
    fn finalize<F: Float>(
        model: ParametricUMAPModel,
        train_data: Vec<F>,
        num_samples: usize,
        num_features: usize,
        device: Device,
        config: ParametricUMAPConfig,
    ) -> FittedParametricUMAP
    where
        F: num::FromPrimitive + burn::tensor::Element,
    {
        let model = model.valid();
        let device = device.without_autodiff();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &device);
        let embedding_tensor = model.forward(global);
        let embedding: Vec<Vec<f64>> = convert_tensor_to_vector(embedding_tensor);

        FittedParametricUMAP {
            model,
            device,
            config,
            embedding,
            num_features,
        }
    }
}

/// A fitted UMAP model containing the trained neural network and embedding.
///
/// This struct is returned by [`ParametricUMAP::fit`] and provides methods to:
/// - Access the computed embedding via [`embedding`](FittedParametricUMAP::embedding)
///   and [`into_embedding`](FittedParametricUMAP::into_embedding)
/// - Transform new data via [`transform`](FittedParametricUMAP::transform) and
///   [`transform_to_tensor`](FittedParametricUMAP::transform_to_tensor)
/// - Access the configuration via [`config`](FittedParametricUMAP::config)
///
/// Because this is a parametric UMAP (neural network), it can project unseen
/// data through [`transform`](FittedParametricUMAP::transform) — unlike classical UMAP
/// implementations.
pub struct FittedParametricUMAP {
    model: ParametricUMAPModel,
    device: Device,
    config: ParametricUMAPConfig,
    embedding: Vec<Vec<f64>>,
    num_features: usize,
}

impl FittedParametricUMAP {
    /// Get the computed embedding for the training data.
    ///
    /// Returns a reference to the embedding coordinates. Each inner vector
    /// represents one input sample in the low-dimensional space.
    ///
    /// # Returns
    ///
    /// A reference to a `Vec<Vec<f64>>` of shape (n_samples, n_components).
    pub fn embedding(&self) -> &Vec<Vec<f64>> {
        &self.embedding
    }

    /// Consume the fitted model and return the embedding, avoiding a copy.
    ///
    /// # Returns
    ///
    /// The embedding as `Vec<Vec<f64>>` of shape (n_samples, n_components).
    pub fn into_embedding(self) -> Vec<Vec<f64>> {
        self.embedding
    }

    /// Get a reference to the configuration used for this fit.
    pub fn config(&self) -> &ParametricUMAPConfig {
        &self.config
    }

    /// Get the number of input features expected by this fitted model.
    pub fn n_features(&self) -> usize {
        self.num_features
    }

    /// Transform new data points into the embedding space.
    ///
    /// Projects new, unseen data through the trained neural network to obtain
    /// low-dimensional coordinates. Input features are used as provided; apply
    /// any desired preprocessing consistently before fitting and transforming.
    ///
    /// # Arguments
    ///
    /// * `data` - New data points as `Vec<Vec<f64>>` (n_new_samples × n_features)
    ///
    /// # Returns
    ///
    /// Embeddings for the new data points as `Vec<Vec<f64>>` (n_new_samples × n_components).
    pub fn transform(&self, data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let local = self.transform_to_tensor(data);
        convert_tensor_to_vector(local)
    }

    /// Transform new data into the embedding space, returning a tensor.
    /// Input features are used as provided, without automatic scaling.
    ///
    /// # Arguments
    ///
    /// * `data` - New data points as `Vec<Vec<f64>>` (n_new_samples × n_features)
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[n_new_samples, n_components]`.
    pub fn transform_to_tensor(&self, data: Vec<Vec<f64>>) -> Tensor<2> {
        if data.is_empty() {
            return Tensor::from_data(
                burn::tensor::TensorData::new(Vec::<f64>::new(), [0, self.config.n_components]),
                &self.device,
            );
        }
        let num_samples = data.len();
        let num_features = data[0].len();
        assert_eq!(
            num_features, self.num_features,
            "expected {} features, got {}",
            self.num_features, num_features
        );
        assert!(
            data.iter().all(|row| row.len() == num_features),
            "transform data rows must all have {num_features} features"
        );

        let train_data: Vec<f64> = data.into_iter().flatten().collect();

        let global = convert_vector_to_tensor(train_data, num_samples, num_features, &self.device);

        self.model.forward(global)
    }

    /// Save the model's weights to a file.
    ///
    /// Serializes the trained model weights to disk for later use.
    /// The configuration and embedding are not saved, only the neural network weights.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the output file
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or an error if saving fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use fast_umap::prelude::*;
    ///
    /// let config = ParametricUMAPConfig::default();
    /// let umap = ParametricUMAP::new(config);
    /// let fitted = umap.fit(data, None);
    /// fitted.save("model.umap").expect("Failed to save model");
    /// ```
    pub fn save(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        serialize::save_model(&self.model, path)
    }

    /// Load a model's weights from a file into a new FittedParametricUMAP instance.
    ///
    /// Creates a new FittedParametricUMAP with the loaded weights. The configuration must
    /// match the saved model (same input size, hidden layer sizes, and output
    /// size). Input features are used as provided, just as they were during fit.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the input file
    /// * `config` - The UMAP configuration that matches the saved model
    /// * `input_size` - The number of features in the input data (must match the saved model)
    /// * `device` - The device where the model should be loaded
    ///
    /// # Returns
    ///
    /// A new `FittedParametricUMAP` instance with the loaded weights, or an error if loading fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use fast_umap::prelude::*;
    ///
    /// // Load a previously saved model
    /// let config = ParametricUMAPConfig::default();
    /// let device = Default::default();
    /// let input_size = 100; // Must match the data used during training
    /// let fitted = FittedParametricUMAP::load("model.umap", config, input_size, device)?;
    /// let embedding = fitted.transform(new_data);
    /// ```
    pub fn load(
        path: impl AsRef<std::path::Path>,
        config: ParametricUMAPConfig,
        input_size: usize,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let device = device.without_autodiff();

        // Create a new model with the same architecture
        let model_config = ParametricUMAPModelConfigBuilder::default()
            .input_size(input_size)
            .hidden_sizes(config.hidden_sizes.clone())
            .output_size(config.n_components)
            .build()?;

        let mut model = ParametricUMAPModel::new(&model_config, &device);

        // Load the weights
        serialize::load_model(&mut model, path, &device)?;

        // Create a FittedParametricUMAP instance (embedding will be empty since we don't have training data)
        Ok(Self {
            model,
            device,
            config,
            embedding: Vec::new(),
            num_features: input_size,
        })
    }

    /// Load a model's weights from a file, inferring input size from sample data.
    ///
    /// This convenience method creates a new FittedParametricUMAP with the loaded weights,
    /// using a sample data point to determine the input size.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the input file
    /// * `config` - The UMAP configuration that matches the saved model
    /// * `sample_data` - A single sample data point to infer the input size
    /// * `device` - The device where the model should be loaded
    ///
    /// # Returns
    ///
    /// A new `FittedParametricUMAP` instance with the loaded weights, or an error if loading fails.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use fast_umap::prelude::*;
    ///
    /// // Load a previously saved model
    /// let config = ParametricUMAPConfig::default();
    /// let device = Default::default();
    /// let sample = vec![1.0, 2.0, 3.0]; // Same dimensionality as training data
    /// let fitted = FittedParametricUMAP::load_with_sample("model.umap", config, sample, device)?;
    /// ```
    pub fn load_with_sample(
        path: impl AsRef<std::path::Path>,
        config: ParametricUMAPConfig,
        sample_data: Vec<f64>,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let input_size = sample_data.len();
        Self::load(path, config, input_size, device)
    }
}

#[cfg(test)]
mod raw_input_tests {
    use super::*;

    #[test]
    fn fitted_embedding_and_transform_forward_raw_feature_values() {
        let data = vec![
            vec![0.0f64, 0.0, 7.0],
            vec![0.0, 100.0, 7.0],
            vec![10.0, 1.0, 7.0],
            vec![0.0, 10_000.0, 7.0],
            vec![2.0, 250.0, 7.0],
        ];
        let config = ParametricUMAPConfig {
            graph: GraphParams {
                n_neighbors: 2,
                ..Default::default()
            },
            optimization: ParametricUMAPOptimizationParams {
                n_epochs: 1,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let (_exit_tx, exit_rx) = crossbeam_channel::unbounded();
        let fitted = ParametricUMAP::with_device(config, Device::flex().autodiff())
            .fit_with_signal(data.clone(), None, exit_rx);

        let raw_training_tensor = convert_vector_to_tensor(
            data.clone().into_iter().flatten().collect(),
            data.len(),
            3,
            &fitted.device,
        );
        let expected_training_embedding: Vec<Vec<f64>> =
            convert_tensor_to_vector(fitted.model.forward(raw_training_tensor));
        assert_eq!(*fitted.embedding(), expected_training_embedding);

        let probe = vec![vec![2.0, 250.0, 7.0]];
        let raw_probe_tensor = convert_vector_to_tensor(
            probe.clone().into_iter().flatten().collect(),
            1,
            3,
            &fitted.device,
        );
        let expected_probe_embedding: Vec<Vec<f64>> =
            convert_tensor_to_vector(fitted.model.forward(raw_probe_tensor));
        assert_eq!(fitted.transform(probe), expected_probe_embedding);
    }
}
