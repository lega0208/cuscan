use burn::prelude::*;
use nn::{Linear, LinearConfig, Relu};
use serde::{Deserialize, Serialize};

/// A neural network model with a configurable number of layers and dynamic sizes.
/// The model can have multiple hidden layers, with each layer having its own configurable size.
///
#[derive(Module, Debug)]
// Keep field names and order stable: Burn records persist these module keys.
pub struct ParametricUMAPModel {
    layers: Vec<Linear>, // Vector to store dynamic layers
    activation: Relu,    // ReLU activation function
}

impl ParametricUMAPModel {
    /// Creates a new instance of `ParametricUMAPModel` with the specified configuration and device.
    ///
    /// # Arguments
    /// * `config` - Configuration struct containing the input size, hidden layer sizes, and output size.
    /// * `device` - The device on which the model should be initialized (e.g., CPU or GPU).
    ///
    /// # Returns
    /// A new `ParametricUMAPModel` instance initialized with the provided configuration.
    pub fn new(config: &ParametricUMAPModelConfig, device: &Device) -> Self {
        // Build the layers dynamically based on the hidden layer sizes.
        let mut layers = Vec::new();
        let mut input_size = config.input_size;

        // Create hidden layers
        for &hidden_size in &config.hidden_sizes {
            layers.push(
                LinearConfig::new(input_size, hidden_size)
                    .with_bias(true)
                    .init(device),
            );
            input_size = hidden_size; // Update input size for the next layer
        }

        // Add the output layer
        layers.push(
            LinearConfig::new(input_size, config.output_size)
                .with_bias(true)
                .init(device),
        );

        // Initialize ReLU activation function
        let activation = Relu::new();

        // Return the ParametricUMAPModel with the initialized layers
        ParametricUMAPModel { layers, activation }
    }

    /// Perform a forward pass through the model.
    ///
    /// # Arguments
    /// * `input` - A 2D tensor representing the input data, with shape (n_samples, n_features).
    ///
    /// # Returns
    /// A 2D tensor representing the output after passing through all the layers and activations.
    pub fn forward(&self, input: Tensor<2>) -> Tensor<2> {
        let mut x = input;

        // Forward pass through each layer with activation
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(x); // Apply linear transformation

            // Apply activation only if it's not the last layer
            if i < self.layers.len() - 1 {
                x = self.activation.forward(x); // Apply ReLU activation
            }
        }

        x
    }
}

/// Configuration structure for creating a `ParametricUMAPModel`.
///
/// # Fields
/// * `input_size` - Number of input features.
/// * `hidden_sizes` - Vector of sizes for the hidden layers.
/// * `output_size` - Number of output features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricUMAPModelConfig {
    pub input_size: usize,        // Number of input features
    pub hidden_sizes: Vec<usize>, // Sizes of hidden layers
    pub output_size: usize,       // Number of output features
}

impl ParametricUMAPModelConfig {
    /// Creates a new builder for the `ParametricUMAPModelConfig`.
    ///
    /// # Returns
    /// A new `ParametricUMAPModelConfigBuilder` to configure the model.
    pub fn builder() -> ParametricUMAPModelConfigBuilder {
        ParametricUMAPModelConfigBuilder::default()
    }
}

/// Builder pattern for the `ParametricUMAPModelConfig` struct.
///
/// # Fields
/// * `input_size` - Option for the number of input features.
/// * `hidden_sizes` - Option for the sizes of the hidden layers.
/// * `output_size` - Option for the number of output features.
#[derive(Debug, Clone)]
pub struct ParametricUMAPModelConfigBuilder {
    input_size: Option<usize>,
    hidden_sizes: Option<Vec<usize>>,
    output_size: Option<usize>,
}

impl Default for ParametricUMAPModelConfigBuilder {
    fn default() -> Self {
        ParametricUMAPModelConfigBuilder {
            input_size: Some(100),
            hidden_sizes: Some(vec![100, 100, 100]), // Default to 3 hidden layers of size 100
            output_size: Some(2),
        }
    }
}

impl ParametricUMAPModelConfigBuilder {
    /// Set the input size for the model.
    ///
    /// # Arguments
    /// * `input_size` - The number of input features.
    ///
    /// # Returns
    /// The updated `ParametricUMAPModelConfigBuilder` with the specified input size.
    pub fn input_size(mut self, input_size: usize) -> Self {
        self.input_size = Some(input_size);
        self
    }

    /// Set the hidden layer sizes for the model.
    ///
    /// # Arguments
    /// * `hidden_sizes` - The sizes of the hidden layers.
    ///
    /// # Returns
    /// The updated `ParametricUMAPModelConfigBuilder` with the specified hidden sizes.
    pub fn hidden_sizes(mut self, hidden_sizes: Vec<usize>) -> Self {
        self.hidden_sizes = Some(hidden_sizes);
        self
    }

    /// Set the output size for the model.
    ///
    /// # Arguments
    /// * `output_size` - The number of output features.
    ///
    /// # Returns
    /// The updated `ParametricUMAPModelConfigBuilder` with the specified output size.
    pub fn output_size(mut self, output_size: usize) -> Self {
        self.output_size = Some(output_size);
        self
    }

    /// Build and return the final `ParametricUMAPModelConfig`.
    ///
    /// # Returns
    /// A `Result` containing the built `ParametricUMAPModelConfig` or an error message if required fields are missing.
    pub fn build(self) -> Result<ParametricUMAPModelConfig, String> {
        // Ensure that all required fields are set
        Ok(ParametricUMAPModelConfig {
            input_size: self
                .input_size
                .ok_or_else(|| "Input size must be set".to_string())?,
            hidden_sizes: self
                .hidden_sizes
                .ok_or_else(|| "Hidden sizes must be set".to_string())?,
            output_size: self
                .output_size
                .ok_or_else(|| "Output size must be set".to_string())?,
        })
    }
}
