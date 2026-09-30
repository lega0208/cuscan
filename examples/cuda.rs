use fast_umap::cuda_backend::{device, device_count, CudaAutodiff};
use fast_umap::prelude::{GraphParams, OptimizationParams, UmapConfig};
use rand::Rng;

fn main() {
    let device = device(0).expect("CUDA device 0 is required to run this example");
    println!("CUDA device count: {}", device_count());

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

    let fitted = fast_umap::Umap::<CudaAutodiff>::with_device(config, device).fit(data, None);
    let embedding = fitted.embedding();
    println!(
        "Embedding shape: {} × {}",
        embedding.len(),
        embedding[0].len()
    );
}
