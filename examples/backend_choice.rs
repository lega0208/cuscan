/// Runtime device selection: wgpu or Flex CPU.
///
/// Usage:
///     cargo run --release --features cuda --example backend_choice -- cuda
///     cargo run --release --features gpu --example backend_choice -- wgpu
///     cargo run --release --example backend_choice -- flex
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;
use rand::RngExt;

fn main() {
    let num_samples = 100;
    let num_features = 3;

    let mut rng = rand::rng();
    let data: Vec<Vec<f64>> = (0..num_samples * num_features)
        .map(|_| rng.random::<f64>())
        .collect::<Vec<f64>>()
        .chunks_exact(num_features)
        .map(|chunk| chunk.to_vec())
        .collect();

    let device_choice = std::env::args().nth(1).unwrap_or_else(|| {
        if cfg!(feature = "cuda") {
            "cuda"
        } else if cfg!(feature = "gpu") {
            "wgpu"
        } else {
            "flex"
        }
        .to_string()
    });

    println!("Running UMAP with {device_choice} device");

    let device = match device_choice.as_str() {
        "wgpu" | "gpu" => {
            #[cfg(feature = "gpu")]
            {
                Device::wgpu(DeviceKind::DiscreteGpu(0))
            }
            #[cfg(not(feature = "gpu"))]
            {
                eprintln!("WGPU device not available. Compile with --features gpu");
                std::process::exit(1);
            }
        }
        "flex" | "cpu" => Device::flex(),
        _ => {
            eprintln!("Unknown device '{device_choice}'. Use: wgpu or flex");
            std::process::exit(1);
        }
    };

    let config = UmapConfig::default();
    let fitted = Umap::with_device(config, device).fit(data.clone(), None);

    let embedding = fitted.embedding();
    println!(
        "Embedding shape: {} x {}",
        embedding.len(),
        embedding[0].len()
    );

    let _new = fitted.transform(data);
    println!("{device_choice} device successfully transformed new data");
}
