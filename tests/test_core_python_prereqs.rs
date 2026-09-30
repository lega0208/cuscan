use burn::backend::{Autodiff, NdArray};
use fast_umap::{validate_fit_data, validate_fit_input, OptimizationParams, TrainingConfig, Umap, UmapConfig};

type CpuBackend = Autodiff<NdArray<f32>>;

fn sample_data() -> Vec<Vec<f32>> {
    (0..60)
        .map(|sample| {
            (0..5)
                .map(|feature| {
                    let x = sample as f32 * 0.17 + feature as f32 * 0.31;
                    (x.sin() + (x * 0.37).cos()) * (feature as f32 + 1.0)
                })
                .collect()
        })
        .collect()
}

fn fit(seed: u64, data: Vec<Vec<f32>>) -> fast_umap::FittedUmap<CpuBackend> {
    let config = UmapConfig {
        graph: fast_umap::GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: OptimizationParams {
            seed,
            n_epochs: 5,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let (_exit_tx, exit_rx) = crossbeam_channel::unbounded();
    Umap::<CpuBackend>::new(config).fit_with_signal(data, None, exit_rx)
}

#[test]
fn ndarray_fit_transform_is_finite_reproducible_and_normalized_consistently() {
    let data = sample_data();
    let first = fit(9999, data.clone());
    let same_seed = fit(9999, data.clone());
    let other_seed = fit(10000, data.clone());

    assert_eq!(first.n_features(), 5);
    let embedding = first.embedding();
    assert_eq!(embedding.len(), 60);
    assert!(embedding.iter().all(|row| row.len() == 2));
    assert!(embedding.iter().flatten().all(|value| value.is_finite()));
    assert_eq!(embedding, same_seed.embedding());
    assert_ne!(embedding, other_seed.embedding());

    let as_f64: Vec<Vec<f64>> = data
        .into_iter()
        .map(|row| row.into_iter().map(f64::from).collect())
        .collect();
    let transformed = first.transform(as_f64);
    assert_eq!(transformed.len(), embedding.len());
    assert!(transformed.iter().all(|row| row.len() == 2));
    for (train_row, transformed_row) in embedding.iter().zip(&transformed) {
        assert_eq!(train_row.len(), transformed_row.len());
        for (train_value, transformed_value) in train_row.iter().zip(transformed_row) {
            assert!((train_value - transformed_value).abs() <= 1e-5);
        }
    }
    assert_eq!(
        first.transform_to_tensor(Vec::new()).dims(),
        [0, 2]
    );
    let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        first.transform_to_tensor(vec![vec![0.0; 4]])
    }));
    assert!(mismatch.is_err());
}

#[test]
fn validate_fit_input_rejects_invalid_dimensions_and_model_settings() {
    let config = UmapConfig::default();
    assert!(validate_fit_input(&config, 0, 5).is_err());
    assert!(validate_fit_input(&config, 20, 0).is_err());
    assert!(validate_fit_input(&config, 20, 5).is_ok());

    let mut invalid = config.clone();
    invalid.n_components = 0;
    assert!(validate_fit_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.n_neighbors = 0;
    assert!(validate_fit_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.n_neighbors = 20;
    assert!(validate_fit_input(&invalid, 20, 5).is_err());

    let mut invalid = config;
    invalid.hidden_sizes.clear();
    assert!(validate_fit_input(&invalid, 20, 5).is_err());
}

#[test]
fn fit_data_validation_rejects_ragged_rows() {
    let data = vec![vec![1.0f32, 2.0], vec![3.0]];
    let error = validate_fit_data(&UmapConfig::default(), &data).unwrap_err();
    assert!(error.contains("row 1"));
    assert!(validate_fit_data(&UmapConfig::default(), &[] as &[Vec<f32>]).is_err());
}

#[test]
fn seed_defaults_survive_deserialization_and_builder_construction() {
    assert_eq!(OptimizationParams::default().seed, 9999);
    let mut config = serde_json::to_value(UmapConfig::default()).unwrap();
    config["optimization"]
        .as_object_mut()
        .unwrap()
        .remove("seed");
    let config: UmapConfig = serde_json::from_value(config).unwrap();
    assert_eq!(config.optimization.seed, 9999);
    assert_eq!(TrainingConfig::builder().build().unwrap().seed, 9999);
}
