use fast_umap::cuda_backend::device_count;
use fast_umap::prelude::{GraphParams, OptimizationParams, UmapConfig};
use rand::RngExt;

fn main() {
    println!("CUDA device count: {}", device_count());
    let device = burn::tensor::Device::cuda(0);

    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..50)
        .map(|_| (0..5).map(|_| rng.random::<f64>()).collect())
        .collect();
    let config = UmapConfig {
        graph: GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: OptimizationParams {
            n_epochs: 2,
            batch_size: 50,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let fitted = fast_umap::Umap::with_device(config, device).fit(data, None);
    let embedding = fitted.embedding();
    println!(
        "Embedding shape: {} × {}",
        embedding.len(),
        embedding[0].len()
    );
}
