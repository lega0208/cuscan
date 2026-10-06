use burn::{prelude::*, tensor::DeviceKind};
use crossbeam_channel::{unbounded, Receiver};
use fast_umap::{
    chart::{self, ChartConfigBuilder},
    utils::convert_vector_to_tensor,
    GraphParams, ParametricUMAP, ParametricUMAPConfig, ParametricUMAPLossReduction,
};
use mnist::*;
use std::sync::{Arc, Mutex};

fn generate_model_name(
    learning_rate: f64,
    penalty: f32,
    hidden_sizes: &[usize],
    epochs: usize,
) -> String {
    let hidden_sizes = hidden_sizes
        .iter()
        .map(|size| size.to_string())
        .collect::<Vec<_>>()
        .join("_");
    format!(
        "mnist_sparse_parametric_lr_{learning_rate:.0e}_pen_{penalty:.0e}_hs_{hidden_sizes}_ep_{epochs}"
    )
}

fn execute(
    name: String,
    num_samples: usize,
    train_data: Vec<Vec<f32>>,
    labels: Vec<String>,
    config: ParametricUMAPConfig,
    device: Device,
    exit_rx: Receiver<()>,
) -> f64 {
    let best_loss = Arc::new(Mutex::new(f64::MAX));
    let observed_best_loss = Arc::clone(&best_loss);
    let fitted = ParametricUMAP::with_device(config, device.clone()).fit_with_progress(
        train_data,
        Some(labels.clone()),
        exit_rx,
        Box::new(move |progress| {
            *observed_best_loss.lock().unwrap() = progress.best_loss;
        }),
    );

    let embedding: Vec<f64> = fitted.embedding().iter().flatten().copied().collect();
    let embedding_tensor = convert_vector_to_tensor(embedding, num_samples, 2, &device);
    let chart_config = ChartConfigBuilder::default()
        .caption(&name)
        .path(format!("figures/{name}.png").as_str())
        .build();
    chart::chart_tensor(embedding_tensor, Some(labels), Some(chart_config));

    let result = *best_loss.lock().unwrap();
    result
}

fn find_best_hyperparameters(
    train_data: Vec<Vec<f32>>,
    labels: Vec<String>,
    device: Device,
    config: ParametricUMAPConfig,
    learning_rates: &[f64],
    penalties: &[f32],
    hidden_size_options: &[Vec<usize>],
    epochs_options: &[usize],
    exit_rx: Receiver<()>,
) -> (String, f64) {
    let mut best_loss = f64::MAX;
    let mut best_config = String::new();
    let num_samples = train_data.len();

    for &learning_rate in learning_rates {
        for &penalty in penalties {
            for hidden_sizes in hidden_size_options {
                for &epochs in epochs_options {
                    let model_name =
                        generate_model_name(learning_rate, penalty, hidden_sizes, epochs);
                    println!("{model_name}");

                    let mut current_config = config.clone();
                    current_config.hidden_sizes = hidden_sizes.clone();
                    current_config.optimization.learning_rate = learning_rate;
                    current_config.optimization.penalty = penalty;
                    current_config.optimization.n_epochs = epochs;

                    let loss = execute(
                        model_name.clone(),
                        num_samples,
                        train_data.clone(),
                        labels.clone(),
                        current_config,
                        device.clone(),
                        exit_rx.clone(),
                    );

                    if loss < best_loss {
                        best_loss = loss;
                        best_config = model_name;
                    }
                }
            }
        }
    }

    (best_config, best_loss)
}

fn main() {
    let (exit_tx, exit_rx) = unbounded();
    ctrlc::set_handler(move || {
        let _ = exit_tx.send(());
    })
    .expect("failed to install Ctrl-C handler");

    let device = Device::wgpu(DeviceKind::DefaultDevice).autodiff();
    let num_samples = 10_000;
    let num_features = 28 * 28;

    device.seed(9999);

    let Mnist {
        trn_img, trn_lbl, ..
    } = MnistBuilder::new()
        .download_and_extract()
        .label_format_digit()
        .training_set_length(num_samples as u32)
        .finalize();

    let train_data: Vec<Vec<f32>> = trn_img
        .chunks_exact(num_features)
        .map(|sample| sample.iter().map(|&byte| byte as f32).collect())
        .collect();
    let labels: Vec<String> = trn_lbl.iter().map(|digit| digit.to_string()).collect();

    let mut config = ParametricUMAPConfig::default();
    config.graph = GraphParams {
        n_neighbors: 15,
        ..Default::default()
    };
    config.optimization.seed = 9999;
    config.optimization.beta1 = 0.9;
    config.optimization.beta2 = 0.999;
    config.optimization.verbose = true;
    config.optimization.min_desired_loss = Some(1e-4);
    config.optimization.loss_reduction = ParametricUMAPLossReduction::Mean;

    let learning_rates = [1e-4, 1e-3, 1e-5];
    let penalties = [1e-6_f32, 1e-7, 1e-8];
    let hidden_size_options = vec![
        vec![100],
        vec![200],
        vec![300],
        vec![500],
        vec![1000],
        vec![1500],
        vec![100, 100],
        vec![200, 200],
        vec![300, 300],
        vec![500, 500],
        vec![1000, 1000],
        vec![1500, 1500],
        vec![100, 100, 100],
        vec![200, 200, 200],
        vec![300, 300, 300],
        vec![500, 500, 500],
    ];
    let epochs_options = [500];

    println!(
        "Sparse parametric MNIST results are not comparable with historical dense benchmark results."
    );
    let (best_config, best_loss) = find_best_hyperparameters(
        train_data,
        labels,
        device,
        config,
        &learning_rates,
        &penalties,
        &hidden_size_options,
        &epochs_options,
        exit_rx,
    );

    println!("Best sparse parametric model configuration: {best_config}");
    println!("Best sparse parametric loss: {best_loss}");
}
