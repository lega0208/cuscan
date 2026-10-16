/// Compare UMAP training with PCA warm-start enabled.
///
/// Usage:
///     cargo run --release --features gpu,cpu,parametric,pca --example pca_comparison
use std::time::Instant;

use fast_umap::{
    GraphParams, ParametricUMAP, ParametricUMAPConfig, ParametricUMAPOptimizationParams,
    utils::generate_test_data,
};

fn device() -> burn::tensor::Device {
    #[cfg(feature = "gpu")]
    {
        burn::tensor::Device::wgpu(burn::tensor::DeviceKind::DefaultDevice)
    }
    #[cfg(not(feature = "gpu"))]
    {
        burn::tensor::Device::cpu()
    }
}

fn run(n_samples: usize, n_features: usize, epochs: usize) {
    let flat: Vec<f32> = generate_test_data(n_samples, n_features);
    let data: Vec<Vec<f64>> = flat
        .chunks(n_features)
        .map(|c| c.iter().map(|&x| x as f64).collect())
        .collect();

    let config = ParametricUMAPConfig {
        n_components: 2,
        hidden_sizes: vec![128],
        graph: GraphParams {
            n_neighbors: 15,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            n_epochs: epochs,
            batch_size: n_samples,
            learning_rate: 1e-3,
            verbose: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let start = Instant::now();
    let (_, exit_rx) = crossbeam_channel::unbounded();
    let umap = ParametricUMAP::with_device(config, device());
    let fitted = umap.fit_with_signal(data, None, exit_rx);
    let elapsed = start.elapsed().as_secs_f64();

    let emb = fitted.embedding();
    println!(
        "\n  Result: {:.3}s, {} samples x {} dims\n",
        elapsed,
        emb.len(),
        emb[0].len()
    );
}

fn main() {
    println!("\n=== UMAP with PCA Warm-Start ===\n");

    println!("--- 5000 x 100, 200 epochs ---");
    run(5_000, 100, 200);

    println!("--- 10000 x 100, 200 epochs ---");
    run(10_000, 100, 200);
}
