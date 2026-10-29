//! Portable-state validation for the prediction-only HDBSCAN API.

use hdbscan::{
    BackendChoice, HDBSCAN, HDBSCANPredictor, HdbscanConfig, HdbscanError, PredictionCoresState,
    PredictionHierarchyState,
};

fn cuda_backend_or_skip() -> Option<BackendChoice> {
    let backend = BackendChoice::Cuda { device_index: 0 };
    if cuscan_common::selected_device(backend).is_ok() {
        return Some(backend);
    }
    assert!(
        std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
        "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
    );
    eprintln!("SKIPPED: no CUDA device");
    None
}

fn config(min_cluster_size: usize, min_samples: usize) -> HdbscanConfig {
    HdbscanConfig {
        min_cluster_size,
        min_samples: Some(min_samples),
        ..HdbscanConfig::default()
    }
}

fn data() -> Vec<Vec<f32>> {
    (0..24)
        .map(|i| {
            vec![
                (i % 12) as f32 * 0.25 + if i < 12 { 0.0 } else { 100.0 },
                (i % 5) as f32 * 0.125,
            ]
        })
        .collect()
}

fn clustered_state() -> hdbscan::HdbscanPredictorState {
    HDBSCAN::new(config(3, 2), BackendChoice::Cpu)
        .fit(&data())
        .unwrap()
        .to_predictor()
        .unwrap()
        .export_state()
        .unwrap()
}

type StateMutation = fn(&mut hdbscan::HdbscanPredictorState);

#[test]
fn clustered_predictor_state_roundtrip_is_ready_and_exact() {
    let state = clustered_state();
    assert_eq!(state.min_samples, 2);
    let PredictionCoresState::Ready { rank, values } = &state.prediction_cores else {
        panic!("clustered state must have ready cores")
    };
    assert_eq!(*rank, 2);
    assert_eq!(values.len(), 24);
    assert!(
        values
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    );
    let predictor = HDBSCANPredictor::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(predictor.export_state().unwrap(), state);
    assert_eq!(predictor.n_features(), 2);
    let query = vec![vec![0.25, 0.125], vec![100.5, 0.25], vec![1000.0, 0.0]];
    let result = predictor.approximate_predict(&query).unwrap();
    assert!(!result.no_cluster_tree);
    assert_eq!(result.labels.len(), query.len());
    assert_eq!(result.connecting_points.len(), query.len());
    assert_eq!(result.probabilities.len(), query.len());

    // Import is immediately ready: prediction only performs query work, never core preparation.
    let status = predictor.prediction_cache_status().unwrap();
    assert!(status.tree_ready);
    assert_eq!(
        status.cores,
        hdbscan::PredictionCoreStatus::Ready { rank: 2, len: 24 }
    );
}

#[test]
fn exported_cpu_state_imports_on_cuda_and_predicts_identically() {
    let Some(cuda) = cuda_backend_or_skip() else {
        return;
    };
    let state = clustered_state();
    let cpu = HDBSCANPredictor::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    let cuda_predictor = HDBSCANPredictor::from_state(state, cuda).unwrap();
    let query = vec![vec![0.25, 0.125], vec![100.5, 0.25], vec![1000.0, 0.0]];
    let expected = cpu.approximate_predict(&query).unwrap();
    let actual = cuda_predictor.approximate_predict(&query).unwrap();
    assert_eq!(actual.no_cluster_tree, expected.no_cluster_tree);
    assert_eq!(actual.labels, expected.labels);
    assert_eq!(actual.connecting_points, expected.connecting_points);
    for (actual, expected) in actual.probabilities.iter().zip(&expected.probabilities) {
        assert!((actual - expected).abs() <= 1e-6 + 1e-5 * expected.abs());
    }
}

#[test]
fn export_is_identical_before_and_after_fitted_prediction() {
    let model = HDBSCAN::new(config(3, 2), BackendChoice::Cpu)
        .fit(&data())
        .unwrap();
    let predictor = model.to_predictor().unwrap();
    let before = predictor.export_state().unwrap();
    model
        .approximate_predict(&[vec![0.25, 0.125], vec![100.5, 0.25]])
        .unwrap();
    let after = model.to_predictor().unwrap().export_state().unwrap();
    assert_eq!(after, before);
}

fn conversion_and_import_preserve_ready_cores(backend: BackendChoice) {
    let mut training = data();
    training[0] = vec![f32::NAN, 0.0];
    training[13] = vec![f32::INFINITY, 0.0];
    training[2] = training[3].clone();
    let cold = HDBSCAN::new(config(3, 2), backend).fit(&training).unwrap();
    let warm = HDBSCAN::new(config(3, 2), backend).fit(&training).unwrap();
    let query: Vec<Vec<f32>> = (0..1029)
        .map(|i| match i % 4 {
            0 => training[2].clone(),
            1 => vec![0.25, 0.125],
            2 => vec![100.5, 0.25],
            _ => vec![1000.0, 0.0],
        })
        .collect();
    let warm_result = warm.approximate_predict(&query).unwrap();
    assert!(!warm_result.no_cluster_tree);
    let cold_predictor = cold.to_predictor().unwrap();
    let warm_predictor = warm.to_predictor().unwrap();
    let cold_state = cold_predictor.export_state().unwrap();
    let warm_state = warm_predictor.export_state().unwrap();
    let (
        PredictionCoresState::Ready { rank, values },
        PredictionCoresState::Ready {
            rank: warm_rank,
            values: warm_values,
        },
    ) = (&cold_state.prediction_cores, &warm_state.prediction_cores)
    else {
        panic!("conversion must export ready cores")
    };
    assert_eq!(*rank, 2);
    assert_eq!(rank, warm_rank);
    assert_eq!(values.len(), 22);
    assert_eq!(values.len(), warm_values.len());
    for (cold, warm) in values.iter().zip(warm_values) {
        assert_eq!(cold.to_bits(), warm.to_bits());
    }
    assert_eq!(cold_state, warm_state);
    let restored = HDBSCANPredictor::from_state(cold_state.clone(), backend).unwrap();
    let imported_state = restored.export_state().unwrap();
    assert_eq!(imported_state, cold_state);
    let PredictionCoresState::Ready {
        values: imported_values,
        ..
    } = &imported_state.prediction_cores
    else {
        panic!("import must retain ready cores")
    };
    assert_eq!(imported_values.len(), values.len());
    for (imported, original) in imported_values.iter().zip(values) {
        assert_eq!(imported.to_bits(), original.to_bits());
    }
    assert_eq!(
        restored.prediction_cache_status().unwrap().cores,
        hdbscan::PredictionCoreStatus::Ready { rank: 2, len: 22 }
    );
    for predictor in [&cold_predictor, &warm_predictor, &restored] {
        let actual = predictor.approximate_predict(&query).unwrap();
        assert_eq!(actual.no_cluster_tree, warm_result.no_cluster_tree);
        assert_eq!(actual.labels, warm_result.labels);
        assert_eq!(actual.connecting_points, warm_result.connecting_points);
        assert_eq!(actual.probabilities.len(), warm_result.probabilities.len());
        for (actual, expected) in actual.probabilities.iter().zip(&warm_result.probabilities) {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(predictor.export_state().unwrap(), cold_state);
    }
}

#[test]
fn conversion_and_import_preserve_ready_cores_cpu() {
    conversion_and_import_preserve_ready_cores(BackendChoice::Cpu);
}

#[test]
fn conversion_and_import_preserve_ready_cores_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        conversion_and_import_preserve_ready_cores(backend);
    }
}

#[test]
fn mixed_nonfinite_and_tied_geometry_roundtrips_raw_connectors() {
    let mut training = data();
    training[0] = vec![f32::NAN, 0.0];
    training[1] = vec![f32::INFINITY, 0.0];
    // Duplicate finite rows exercise tied candidate ordering and preserve raw IDs.
    training[2] = training[3].clone();
    let model = HDBSCAN::new(config(3, 2), BackendChoice::Cpu)
        .fit(&training)
        .unwrap();
    let predictor = model.to_predictor().unwrap();
    let state = predictor.export_state().unwrap();
    let restored = HDBSCANPredictor::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.export_state().unwrap(), state);
    assert!(state.hierarchy.finite_index.windows(2).all(|w| w[0] < w[1]));
    assert!(!state.hierarchy.finite_index.contains(&0));
    assert!(!state.hierarchy.finite_index.contains(&1));
    let query = [training[2].clone(), training[3].clone(), vec![1000.0, 0.0]];
    let expected = model.approximate_predict(&query).unwrap();
    let actual = restored.approximate_predict(&query).unwrap();
    assert_eq!(actual.labels, expected.labels);
    assert_eq!(actual.probabilities, expected.probabilities);
    assert_eq!(actual.connecting_points, expected.connecting_points);
    assert!(
        actual
            .connecting_points
            .iter()
            .all(|&raw| raw == -1 || raw >= 2)
    );
    assert!(
        actual
            .connecting_points
            .iter()
            .all(|&raw| training[raw as usize].iter().all(|v| v.is_finite()))
    );
}

#[test]
fn no_cluster_shortcut_state_roundtrips_without_payload() {
    let fitted = HDBSCAN::new(config(5, 40), BackendChoice::Cpu)
        .fit(&[vec![1.0, 2.0]])
        .unwrap();
    let state = fitted.to_predictor().unwrap().export_state().unwrap();
    assert_eq!(state.min_samples, 40);
    assert!(state.geometry.is_empty());
    assert_eq!(
        state.prediction_cores,
        PredictionCoresState::NotRequiredNoClusterTree
    );
    let restored = HDBSCANPredictor::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.export_state().unwrap(), state);
    assert_eq!(
        restored.prediction_cache_status().unwrap().cores,
        hdbscan::PredictionCoreStatus::NotRequiredNoClusterTree
    );
    assert_eq!(
        restored.approximate_predict(&[vec![1.0, 2.0]]).unwrap(),
        fitted.approximate_predict(&[vec![1.0, 2.0]]).unwrap()
    );
}

#[test]
fn rejects_malformed_predictor_dimensions_cores_and_hierarchy() {
    let cases: [(&str, StateMutation, bool); 15] = [
        ("zero feature count", |s| s.n_features = 0, false),
        ("zero min_samples", |s| s.min_samples = 0, false),
        (
            "geometry count mismatch",
            |s| {
                s.geometry.pop();
            },
            false,
        ),
        (
            "core rank mismatch",
            |s| {
                if let PredictionCoresState::Ready { rank, .. } = &mut s.prediction_cores {
                    *rank += 1
                }
            },
            false,
        ),
        (
            "NaN prediction core",
            |s| {
                if let PredictionCoresState::Ready { values, .. } = &mut s.prediction_cores {
                    values[0] = f64::NAN
                }
            },
            false,
        ),
        (
            "negative prediction core",
            |s| {
                if let PredictionCoresState::Ready { values, .. } = &mut s.prediction_cores {
                    values[0] = -1.0
                }
            },
            false,
        ),
        (
            "infinite prediction core",
            |s| {
                if let PredictionCoresState::Ready { values, .. } = &mut s.prediction_cores {
                    values[0] = f64::INFINITY
                }
            },
            false,
        ),
        (
            "prediction core count mismatch",
            |s| {
                if let PredictionCoresState::Ready { values, .. } = &mut s.prediction_cores {
                    values.pop();
                }
            },
            false,
        ),
        ("nonfinite geometry", |s| s.geometry[0] = f32::NAN, false),
        (
            "invalid root parent",
            |s| s.hierarchy.node_parent_indices[0] = 0,
            false,
        ),
        (
            "nonascending finite map",
            |s| s.hierarchy.finite_index[1] = s.hierarchy.finite_index[0],
            false,
        ),
        (
            "out-of-range point node",
            |s| s.hierarchy.point_node_indices[0] = u64::MAX,
            false,
        ),
        (
            "NaN node birth",
            |s| s.hierarchy.node_birth_lambdas[0] = f64::NAN,
            false,
        ),
        (
            "geometry element count overflow",
            |s| s.n_features = usize::MAX,
            true,
        ),
        (
            "candidate width overflow",
            |s| s.min_samples = usize::MAX,
            true,
        ),
    ];
    for (case, corrupt, expects_overflow) in cases {
        let mut state = clustered_state();
        corrupt(&mut state);
        let result = HDBSCANPredictor::from_state(state, BackendChoice::Cpu);
        if expects_overflow {
            assert!(
                matches!(result, Err(HdbscanError::Overflow(_))),
                "{case}: expected typed overflow, got {result:?}"
            );
        } else {
            assert!(
                matches!(result, Err(HdbscanError::InvalidInput(_))),
                "{case}: expected invalid input, got {result:?}"
            );
        }
    }

    let mut mismatched_clustered = clustered_state();
    mismatched_clustered.prediction_cores = PredictionCoresState::NotRequiredNoClusterTree;
    assert!(matches!(
        HDBSCANPredictor::from_state(mismatched_clustered, BackendChoice::Cpu),
        Err(HdbscanError::InvalidInput(_))
    ));

    let no_tree = HDBSCAN::new(config(5, 40), BackendChoice::Cpu)
        .fit(&[vec![1.0, 2.0]])
        .unwrap();
    let mut mismatched_shortcut = no_tree.to_predictor().unwrap().export_state().unwrap();
    mismatched_shortcut.prediction_cores = PredictionCoresState::Ready {
        rank: 40,
        values: Vec::new(),
    };
    assert!(matches!(
        HDBSCANPredictor::from_state(mismatched_shortcut, BackendChoice::Cpu),
        Err(HdbscanError::InvalidInput(_))
    ));
}

#[test]
fn hierarchy_portable_columns_are_part_of_public_state_contract() {
    let state = clustered_state();
    let hierarchy: &PredictionHierarchyState = &state.hierarchy;
    assert!(!hierarchy.finite_index.is_empty());
    assert_eq!(
        hierarchy.node_ids.len(),
        hierarchy.node_parent_indices.len()
    );
    assert_eq!(hierarchy.node_ids.len(), hierarchy.node_birth_lambdas.len());
    assert_eq!(hierarchy.node_ids.len(), hierarchy.node_labels.len());
    assert_eq!(hierarchy.node_ids.len(), hierarchy.node_max_lambdas.len());
    assert_eq!(
        hierarchy.finite_index.len(),
        hierarchy.point_node_indices.len()
    );
    assert_eq!(
        hierarchy.finite_index.len(),
        hierarchy.point_departure_lambdas.len()
    );
}

#[test]
fn rank_one_state_requires_canonical_zero_cores() {
    let fitted = HDBSCAN::new(config(3, 1), BackendChoice::Cpu)
        .fit(&data())
        .unwrap();
    let mut state = fitted.to_predictor().unwrap().export_state().unwrap();
    let PredictionCoresState::Ready { values, .. } = &mut state.prediction_cores else {
        panic!("clustered model must have ready cores")
    };
    assert!(
        values
            .iter()
            .all(|value| value.to_bits() == 0.0f64.to_bits())
    );
    values[0] = -0.0;
    assert!(matches!(
        HDBSCANPredictor::from_state(state, BackendChoice::Cpu),
        Err(HdbscanError::InvalidInput(_))
    ));
}

#[test]
fn explicit_unavailable_cuda_backend_does_not_fall_back() {
    let state = clustered_state();
    let backend = BackendChoice::Cuda {
        device_index: usize::MAX,
    };
    assert!(HDBSCANPredictor::from_state(state, backend).is_err());
}
