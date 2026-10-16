use fast_umap::standard_umap::{
    BackendChoice, Init, NeighborSearch, UMAPConfig, approximate_query_budget, fit_curve,
    fit_neighbor_widths, query_neighbor_width, validate_backend_choice, validate_input,
    validate_search_settings,
};

#[test]
fn standard_config_defaults_match_the_reference_controls() {
    let config = UMAPConfig::default();
    assert_eq!(config.n_components, 2);
    assert_eq!(config.n_neighbors, 15);
    assert_eq!(config.min_dist, 0.1);
    assert_eq!(config.spread, 1.0);
    assert_eq!(config.local_connectivity, 1.0);
    assert_eq!(config.set_op_mix_ratio, 1.0);
    assert_eq!(config.n_epochs, None);
    assert_eq!(config.learning_rate, 1.0);
    assert_eq!(config.negative_sample_rate, 5);
    assert_eq!(config.repulsion_strength, 1.0);
    assert_eq!(config.init, Init::Spectral);
    assert_eq!(config.random_state, None);
    assert_eq!(config.transform_seed, 42);
    assert_eq!(config.transform_queue_size, 4.0);
    config.validate().unwrap();
}

#[test]
fn fit_and_query_widths_keep_their_distinct_self_semantics() {
    assert_eq!(fit_neighbor_widths(100, 15).unwrap().self_inclusive, 15);
    assert_eq!(fit_neighbor_widths(100, 15).unwrap().nonself, 14);
    assert_eq!(fit_neighbor_widths(5, 15).unwrap().self_inclusive, 4);
    assert_eq!(fit_neighbor_widths(5, 15).unwrap().nonself, 3);
    assert_eq!(fit_neighbor_widths(1, 15).unwrap().self_inclusive, 1);
    assert_eq!(fit_neighbor_widths(1, 15).unwrap().nonself, 0);
    // N=2 has no non-self fit neighbor at the approved min(request, N-1) width;
    // validation must not invent a generic fallback edge.
    assert_eq!(fit_neighbor_widths(2, 15).unwrap().nonself, 0);
    assert_eq!(query_neighbor_width(5, 15).unwrap(), 5);
    assert_eq!(query_neighbor_width(100, 15).unwrap(), 15);
}

#[test]
fn input_validation_checks_shape_finiteness_and_feature_matching() {
    assert_eq!(
        validate_input(&[vec![0.0, 1.0], vec![2.0, 3.0]], None).unwrap(),
        (2, 2)
    );
    assert!(validate_input(&[], None).is_err());
    assert!(validate_input(&[vec![]], None).is_err());
    assert!(validate_input(&[vec![1.0], vec![2.0, 3.0]], None).is_err());
    assert!(validate_input(&[vec![1.0, f32::INFINITY]], None).is_err());
    assert!(validate_input(&[vec![1.0, f32::NAN]], None).is_err());
    assert!(validate_input(&[vec![1.0, 2.0]], Some(3)).is_err());
}

#[test]
fn config_rejects_invalid_control_values() {
    let config = UMAPConfig {
        min_dist: f32::NAN,
        ..Default::default()
    };
    assert!(config.validate().is_err());
    let config = UMAPConfig {
        min_dist: 1.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
    let config = UMAPConfig {
        transform_queue_size: f64::INFINITY,
        ..Default::default()
    };
    assert!(config.validate().is_err());
}

#[test]
fn reference_permits_zero_epochs_and_optimizer_controls() {
    let config = UMAPConfig {
        n_epochs: Some(0),
        learning_rate: 0.0,
        negative_sample_rate: 0,
        repulsion_strength: 0.0,
        ..Default::default()
    };
    assert!(config.validate().is_ok());
}

#[test]
fn optimizer_controls_reject_negative_or_nonfinite_floats() {
    let mut config = UMAPConfig {
        learning_rate: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
    config.learning_rate = f32::INFINITY;
    assert!(config.validate().is_err());

    let mut config = UMAPConfig {
        repulsion_strength: -0.1,
        ..Default::default()
    };
    assert!(config.validate().is_err());
    config.repulsion_strength = f32::NAN;
    assert!(config.validate().is_err());
}

#[test]
fn fit_curve_matches_pinned_umap_0512_coefficients() {
    let fixtures = [
        ((0.1, 1.0), (1.5769434602697652, 0.8950608778515733)),
        ((0.0, 1.0), (1.93280839734315, 0.7904949732233831)),
        ((0.5, 1.0), (0.5830300203414425, 1.3341669924314914)),
        ((0.1, 2.0), (0.5446605399418663, 0.8420554268341789)),
    ];
    for ((min_dist, spread), (expected_a, expected_b)) in fixtures {
        let (a, b) = fit_curve(min_dist, spread).unwrap();
        assert!(
            (f64::from(a) - expected_a).abs() < 1.0e-4,
            "a={a}, expected {expected_a}"
        );
        assert!(
            (f64::from(b) - expected_b).abs() < 1.0e-4,
            "b={b}, expected {expected_b}"
        );

        // The returned coefficients evaluate the fitted low-dimensional curve.
        for x in [0.0_f32, min_dist, spread, 2.0 * spread, 3.0 * spread] {
            let actual = 1.0 / (1.0 + a * x.powf(2.0 * b));
            let reference = 1.0 / (1.0 + expected_a as f32 * x.powf(2.0 * expected_b as f32));
            assert!((actual - reference).abs() < 1.0e-5);
        }
    }
}

#[test]
fn fit_curve_rejects_invalid_controls() {
    assert!(fit_curve(f32::NAN, 1.0).is_err());
    assert!(fit_curve(-0.1, 1.0).is_err());
    assert!(fit_curve(0.1, 0.0).is_err());
    assert!(fit_curve(0.1, f32::INFINITY).is_err());
}

#[test]
fn approximate_queue_budgets_are_checked_and_capped() {
    assert_eq!(
        approximate_query_budget(100, 10, 4.0).unwrap(),
        fast_umap::standard_umap::QueryBudget {
            candidates: 40,
            visits: 100
        }
    );
    assert_eq!(
        approximate_query_budget(5, 4, 4.0).unwrap(),
        fast_umap::standard_umap::QueryBudget {
            candidates: 5,
            visits: 5
        }
    );
    assert!(approximate_query_budget(usize::MAX, usize::MAX, f64::MAX).is_err());
    assert!(approximate_query_budget(10, 0, 4.0).is_err());
}

#[test]
fn queue_control_is_exact_compatible_only_at_its_default() {
    assert!(validate_search_settings(NeighborSearch::Exact, 4.0, None).is_ok());
    assert!(validate_search_settings(NeighborSearch::Exact, 4.1, None).is_err());
    assert!(validate_search_settings(NeighborSearch::Approximate, 4.1, None).is_ok());
    assert!(validate_search_settings(NeighborSearch::Auto, 4.0, None).is_err());
    assert!(
        validate_search_settings(NeighborSearch::Auto, 4.1, Some(NeighborSearch::Approximate))
            .is_ok()
    );
    assert!(
        validate_search_settings(NeighborSearch::Auto, 4.1, Some(NeighborSearch::Exact)).is_err()
    );
}

#[test]
fn backend_selection_never_falls_back_from_cuda() {
    assert!(validate_backend_choice(BackendChoice::Cpu).is_ok());
    let error = validate_backend_choice(BackendChoice::Cuda {
        device_index: usize::MAX,
    });
    assert!(error.is_err());
    #[cfg(not(feature = "cuda"))]
    assert!(
        error
            .unwrap_err()
            .to_string()
            .contains("requires the `cuda` feature")
    );
}
