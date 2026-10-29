use std::time::Instant;
use umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

const SIZES: [usize; 6] = [1_000, 2_000, 4_000, 8_000, 16_000, 32_000];
const FEATURES: usize = 64;

fn data(rows: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..FEATURES)
                .map(|feature| {
                    let phase = row as f32 * 0.013 + feature as f32 * 0.17;
                    (phase * (feature as f32 + 1.0)).sin() + (phase * 0.37).cos() * 0.2
                })
                .collect()
        })
        .collect()
}

fn measure(backend: BackendChoice, data: &[Vec<f32>], search: NeighborSearch) -> f64 {
    let config = UMAPConfig {
        n_neighbors: 15,
        n_components: 2,
        init: Init::Random,
        n_epochs: Some(5),
        random_state: Some(47),
        ..UMAPConfig::default()
    };
    let start = Instant::now();
    UMAP::new(config, backend, search).fit(data).unwrap();
    start.elapsed().as_secs_f64()
}

fn benchmark(label: &str, backend: BackendChoice) -> Option<usize> {
    let mut crossover = None;
    println!("{label} exact-vs-approximate fit times (features={FEATURES})");
    for rows in SIZES {
        let points = data(rows);
        let exact = measure(backend, &points, NeighborSearch::Exact);
        let approximate = measure(backend, &points, NeighborSearch::Approximate);
        println!("N={rows:>5}: exact={exact:>8.3}s approximate={approximate:>8.3}s");
        if crossover.is_none() && approximate <= exact {
            crossover = Some(rows);
        }
    }
    println!("{label} approximate-search crossover: {crossover:?}");
    crossover
}

fn main() {
    let cpu_crossover = benchmark("CPU", BackendChoice::Cpu);
    let cuda_crossover = if let Some(device) = umap::cuda_backend::device(0) {
        let _ = device;
        benchmark("CUDA", BackendChoice::Cuda { device_index: 0 })
    } else {
        println!("CUDA unavailable; no CUDA crossover measurement");
        None
    };
    // AUTO_EXACT_MAX_ROWS follows the measured CUDA crossover when present, otherwise CPU.
    println!(
        "CUDA-preferred recommended AUTO_EXACT_MAX_ROWS: {:?}",
        cuda_crossover.or(cpu_crossover)
    );
}
