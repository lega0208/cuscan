//! Feature-based compilation demonstration
//!
//! This example shows how to compile fast-umap with different feature combinations
//! to optimize for your specific needs and reduce compile times.

use burn::tensor::Device;
use fast_umap::prelude::*;
use rand::RngExt;

fn main() {
    println!("=== Fast-UMAP Feature Demo ===");
    println!();
    println!("Available feature combinations:");
    println!();

    println!("1. Minimal GPU (default):");
    println!("   cargo build --release");
    println!("   Features: gpu, verbose");
    println!("   Best for: Production use with GPU");
    println!();

    println!("2. CPU-only with Flex:");
    println!("   cargo build --release");
    println!("   Select Device::flex() at runtime");
    println!("   Best for: CPU-only environments");
    println!();

    println!("3. Minimal (no backend):");
    println!("   cargo build --release --no-default-features");
    println!("   Features: (none)");
    println!("   Best for: Library usage, custom backends");
    println!();

    println!("4. All features:");
    println!("   cargo build --release --features all");
    println!("   Features: gpu, cpu, pca, nn-descent, verbose, plotters");
    println!("   Best for: Development and testing");
    println!();

    println!("5. Custom combination:");
    println!("   cargo build --release --features \"gpu,plotters\"");
    println!("   Features: gpu, plotters, verbose");
    println!("   Best for: GPU with visualization");
    println!();

    // Generate test data and run on the Flex CPU device.
    let num_samples = 30;
    let num_features = 3;
    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    println!(
        "Generated test data: {} samples × {} features",
        num_samples, num_features
    );
    println!();

    let config = ParametricUMAPConfig {
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            n_epochs: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let fitted = ParametricUMAP::with_device(config, Device::flex()).fit(data, None);
    println!(
        "Flex CPU UMAP embedding: {} samples × {} dimensions",
        fitted.embedding().len(),
        fitted.embedding()[0].len()
    );

    // Report optional integrations (Flex CPU execution is always available).
    #[cfg(feature = "gpu")]
    {
        println!("✓ GPU backend available");
        println!("  - Parametric UMAP with neural networks");
        println!("  - GPU acceleration via WGPU");
        println!("  - Transform new data support");
    }

    println!("✓ Flex CPU backend available");
    println!("  - Parametric UMAP with neural networks");
    println!("  - No GPU required");

    #[cfg(feature = "verbose")]
    println!("✓ Verbose output enabled");

    #[cfg(feature = "plotters")]
    println!("✓ Plotting support enabled");

    println!();
    println!("=== Feature Demo Complete ===");
    println!();
    println!("Tip: Choose features based on your target environment:");
    println!("  - Cloud GPU instances: --features gpu");
    println!("  - CPU-only servers: Device::flex()");
    println!("  - Edge devices: Device::flex() with minimal features");
    println!("  - Development: --features all");
}
