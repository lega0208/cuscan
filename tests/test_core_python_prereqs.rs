#![cfg(feature = "parametric")]
use burn::tensor::Device;
use fast_umap::{
    ParametricUMAP, ParametricUMAPConfig, ParametricUMAPOptimizationParams,
    validate_parametric_umap_data, validate_parametric_umap_input,
};

// CPU seeding is process-global; seeded training cases must not race each other.
static TRAINING_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn fit(seed: u64, data: Vec<Vec<f32>>) -> fast_umap::FittedParametricUMAP {
    let config = ParametricUMAPConfig {
        graph: fast_umap::GraphParams {
            n_neighbors: 5,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            seed,
            n_epochs: 5,
            verbose: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let (_exit_tx, exit_rx) = crossbeam_channel::unbounded();
    ParametricUMAP::with_device(config, Device::cpu().autodiff())
        .fit_with_signal(data, None, exit_rx)
}

#[test]
fn cpu_fit_transform_is_finite_reproducible_and_consistent() {
    let _training_guard = TRAINING_LOCK.lock().unwrap();
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
    let transformed_tensor = first.transform_to_tensor(vec![vec![0.0; 5]]);
    assert!(!transformed_tensor.device().is_autodiff());
    let empty_tensor = first.transform_to_tensor(Vec::new());
    assert_eq!(empty_tensor.dims(), [0, 2]);
    assert!(!empty_tensor.device().is_autodiff());
    let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        first.transform_to_tensor(vec![vec![0.0; 4]])
    }));
    assert!(mismatch.is_err());
}

#[test]
fn fit_and_transform_keep_unequal_scale_features_raw() {
    let _training_guard = TRAINING_LOCK.lock().unwrap();
    let data = vec![
        vec![0.0f32, 0.0, 7.0],
        vec![0.0, 100.0, 7.0],
        vec![10.0, 1.0, 7.0],
        vec![0.0, 10_000.0, 7.0],
        vec![1.0, 5_000.0, 7.0],
        vec![9.0, 2_000.0, 7.0],
    ];
    let mut standardized: Vec<f32> = data.iter().flatten().copied().collect();
    fast_umap::utils::normalize_data(&mut standardized, data.len(), 3);
    let standardized = standardized
        .as_chunks::<3>()
        .0
        .iter()
        .map(|row| row.to_vec())
        .collect();
    let raw_fitted = fit(321, data);
    let standardized_fitted = fit(321, standardized);

    // An automatic scaler would erase this deliberate geometry/input difference.
    assert!(
        raw_fitted
            .embedding()
            .iter()
            .flatten()
            .zip(standardized_fitted.embedding().iter().flatten())
            .any(|(raw, scaled)| (raw - scaled).abs() > 1e-3)
    );
}

#[test]
fn validate_parametric_umap_input_rejects_invalid_dimensions_and_model_settings() {
    let config = ParametricUMAPConfig::default();
    assert!(validate_parametric_umap_input(&config, 0, 5).is_err());
    assert!(validate_parametric_umap_input(&config, 20, 0).is_err());
    assert!(validate_parametric_umap_input(&config, 20, 5).is_ok());

    let mut invalid = config.clone();
    invalid.n_components = 0;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.n_neighbors = 0;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.n_neighbors = 1;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.n_neighbors = 20;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.hidden_sizes.clear();
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.local_connectivity = f64::NAN;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());
    invalid.graph.local_connectivity = -0.1;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.bandwidth = f64::INFINITY;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());
    invalid.graph.bandwidth = 0.0;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config.clone();
    invalid.graph.set_op_mix_ratio = -0.1;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());
    invalid.graph.set_op_mix_ratio = f64::NAN;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());

    let mut invalid = config;
    invalid.optimization.n_epochs = 0;
    assert!(validate_parametric_umap_input(&invalid, 20, 5).is_err());
}

#[test]
fn fit_data_validation_rejects_ragged_rows() {
    let data = vec![vec![1.0f32, 2.0], vec![3.0]];
    let error = validate_parametric_umap_data(&ParametricUMAPConfig::default(), &data).unwrap_err();
    assert!(error.contains("row 1"));
    assert!(
        validate_parametric_umap_data(&ParametricUMAPConfig::default(), &[] as &[Vec<f32>])
            .is_err()
    );
}

#[test]
fn seed_defaults_survive_deserialization() {
    assert_eq!(ParametricUMAPOptimizationParams::default().seed, 9999);
    let mut config = serde_json::to_value(ParametricUMAPConfig::default()).unwrap();
    config["optimization"]
        .as_object_mut()
        .unwrap()
        .remove("seed");
    let config: ParametricUMAPConfig = serde_json::from_value(config).unwrap();
    assert_eq!(config.optimization.seed, 9999);
    assert_eq!(ParametricUMAPConfig::default().optimization.seed, 9999);
}

#[test]
fn graph_controls_default_and_round_trip_through_estimator_config() {
    let defaults = fast_umap::GraphParams::default();
    assert_eq!(defaults.local_connectivity, 1.0);
    assert_eq!(defaults.bandwidth, 1.0);
    assert_eq!(defaults.set_op_mix_ratio, 1.0);
    let training_defaults = ParametricUMAPConfig::default();
    assert_eq!(training_defaults.graph.local_connectivity, 1.0);
    assert_eq!(training_defaults.graph.bandwidth, 1.0);
    assert_eq!(training_defaults.graph.set_op_mix_ratio, 1.0);

    let mut config = ParametricUMAPConfig::default();
    config.graph.local_connectivity = 2.5;
    config.graph.bandwidth = 0.75;
    config.graph.set_op_mix_ratio = 0.4;
    assert_eq!(config.graph.local_connectivity, 2.5);
    assert_eq!(config.graph.bandwidth, 0.75);
    assert_eq!(config.graph.set_op_mix_ratio, 0.4);
    let serialized = serde_json::to_value(&config).unwrap();
    let round_trip: ParametricUMAPConfig = serde_json::from_value(serialized).unwrap();
    assert_eq!(round_trip.graph.local_connectivity, 2.5);
    assert_eq!(round_trip.graph.bandwidth, 0.75);
    assert_eq!(round_trip.graph.set_op_mix_ratio, 0.4);
}

#[test]
fn custom_graph_controls_reach_signal_progress_and_fitted_config() {
    let _training_guard = TRAINING_LOCK.lock().unwrap();
    let data = sample_data();
    let mut config = ParametricUMAPConfig::default();
    config.graph.n_neighbors = 5;
    config.graph.local_connectivity = 0.5;
    config.graph.bandwidth = 0.75;
    config.graph.set_op_mix_ratio = 0.5;
    config.optimization.n_epochs = 5;
    config.optimization.verbose = true;
    let (_tx, rx) = crossbeam_channel::unbounded();
    let signal = ParametricUMAP::with_device(config.clone(), Device::cpu().autodiff())
        .fit_with_signal(data.clone(), None, rx);
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = events.clone();
    let (_tx, rx) = crossbeam_channel::unbounded();
    let progress = ParametricUMAP::with_device(config.clone(), Device::cpu().autodiff())
        .fit_with_progress(
            data.clone(),
            None,
            rx,
            Box::new(move |event| sink.lock().unwrap().push(event)),
        );
    assert!(!events.lock().unwrap().is_empty());
    for fitted in [&signal, &progress] {
        assert_eq!(fitted.config().graph.local_connectivity, 0.5);
        assert_eq!(fitted.config().graph.bandwidth, 0.75);
        assert_eq!(fitted.config().graph.set_op_mix_ratio, 0.5);
        assert!(
            fitted
                .embedding()
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        let input: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().map(|value| f64::from(*value)).collect())
            .collect();
        assert_eq!(fitted.embedding(), &fitted.transform(input));
    }
    assert_eq!(signal.embedding(), progress.embedding());
    config.graph.local_connectivity = 1.0;
    config.graph.bandwidth = 1.0;
    config.graph.set_op_mix_ratio = 1.0;
    let defaults = ParametricUMAP::with_device(config, Device::cpu().autodiff()).fit(data, None);
    assert_ne!(signal.embedding(), defaults.embedding());
}

#[test]
fn cancellation_before_first_epoch_keeps_configured_graph_counts_valid() {
    let _training_guard = TRAINING_LOCK.lock().unwrap();
    let mut config = ParametricUMAPConfig::default();
    config.graph.n_neighbors = 5;
    config.graph.local_connectivity = 0.0;
    config.optimization.n_epochs = 10;
    let (tx, rx) = crossbeam_channel::unbounded();
    tx.send(()).unwrap();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = events.clone();
    let fitted = ParametricUMAP::with_device(config, Device::cpu().autodiff()).fit_with_progress(
        sample_data(),
        None,
        rx,
        Box::new(move |event| sink.lock().unwrap().push(event)),
    );
    assert!(events.lock().unwrap().is_empty());
    assert_eq!(fitted.config().optimization.n_epochs, 10);
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
}
