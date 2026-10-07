#![cfg(feature = "parametric")]

//! CPU training demonstration using Burn's Flex backend.
//!
//! Flex provides pure-Rust CPU execution for parametric UMAP without a GPU.

use burn::tensor::Device;
use fast_umap::prelude::*;
use rand::RngExt;

fn main() {
    println!("=== Flex CPU UMAP Demo ===");

    let num_samples = 100;
    let num_features = 5;

    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!(
        "Generated {} samples with {} features each",
        num_samples, num_features
    );

    let config = ParametricUMAPConfig {
        n_components: 2,
        graph: GraphParams {
            n_neighbors: 15,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            n_epochs: 100,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let fitted = ParametricUMAP::with_device(config, Device::flex()).fit(data, None);
    let embedding = fitted.embedding();
    println!("Flex CPU UMAP completed successfully!");
    println!(
        "Embedding shape: {} × {}",
        embedding.len(),
        embedding[0].len()
    );

    println!("\nFirst 5 embedding points:");
    for (i, point) in embedding.iter().take(5).enumerate() {
        println!("  Sample {}: {:?}", i, point);
    }
}
