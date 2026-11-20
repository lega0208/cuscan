// Backend coverage is explicit; CUDA is required when CUSCAN_REQUIRE_CUDA is set.

use umap::{
    BackendChoice, FittedUMAP, InitRescale, NeighborSearch, Optimizer, UMAP, UMAPConfig, UmapError,
    standard_umap::ModelState,
};

fn estimator(search: NeighborSearch, unique: bool, n_epochs: Option<usize>) -> UMAP {
    UMAP::new(
        UMAPConfig {
            n_neighbors: 3,
            init: umap::Init::Random,
            optimizer: umap::Optimizer::Compatibility,
            random_state: Some(7),
            unique,
            n_epochs,
            ..Default::default()
        },
        BackendChoice::Cpu,
        search,
    )
}

fn rows() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![2.0, 1.0],
        vec![3.0, 0.0],
    ]
}

#[test]
fn optimizer_settings_roundtrip_and_persisted_range_validation() {
    let mut backends = vec![BackendChoice::Cpu];
    if umap::cuda_backend::device(0).is_some() {
        backends.push(BackendChoice::Cuda { device_index: 0 });
    } else if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
        panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
    }
    let mut data = rows();
    data.push(data[1].clone());
    let equal_rows_query = rows();
    let small_query = [vec![0.5, 0.5], vec![2.5, 0.5]];
    for backend in backends {
        for optimizer in [
            Optimizer::Compatibility,
            Optimizer::Adam,
            Optimizer::Momentum,
        ] {
            for scale in [None, Some(0.0), Some(1.25)] {
                let config = UMAPConfig {
                    n_neighbors: 3,
                    init: umap::Init::Random,
                    random_state: Some(7),
                    transform_seed: 19,
                    unique: true,
                    n_epochs: Some(8),
                    optimizer,
                    negative_selection_range: 5,
                    negative_sample_scale: scale,
                    ..Default::default()
                };
                let fitted = UMAP::new(config, backend, NeighborSearch::Exact)
                    .fit(&data)
                    .unwrap();
                let state = fitted.export_state().unwrap();
                assert_eq!(state.training_rows, equal_rows_query.len());
                assert!(state.original_to_training.is_some());
                assert_eq!(fitted.embedding()[1], fitted.embedding()[4]);
                assert_eq!(state.config.optimizer, optimizer);
                assert_eq!(state.config.negative_selection_range, 5);
                assert_eq!(state.config.negative_sample_scale, scale);
                let restored = FittedUMAP::from_state(state.clone(), backend).unwrap();
                assert_eq!(restored.export_state().unwrap(), state);
                for query in [&equal_rows_query[..], &small_query[..]] {
                    let expected = fitted.transform(query).unwrap();
                    assert_eq!(restored.transform(query).unwrap(), expected);
                    assert_eq!(fitted.export_state().unwrap(), state);
                    assert_eq!(restored.export_state().unwrap(), state);
                }

                // Exercise malformed imports and the raw reference-range guard once
                // per optimizer and backend without repeating the fits.
                if scale == Some(1.25) {
                    for range in [0, i32::MAX as usize + 1, usize::MAX] {
                        let mut invalid = state.clone();
                        invalid.config.negative_selection_range = range;
                        assert!(matches!(
                            FittedUMAP::from_state(
                                invalid,
                                BackendChoice::Cuda {
                                    device_index: usize::MAX
                                }
                            ),
                            Err(UmapError::InvalidConfig(_))
                        ));
                    }
                    for invalid_scale in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                        let mut invalid = state.clone();
                        invalid.config.negative_sample_scale = Some(invalid_scale);
                        assert!(matches!(
                            FittedUMAP::from_state(
                                invalid,
                                BackendChoice::Cuda {
                                    device_index: usize::MAX
                                }
                            ),
                            Err(UmapError::InvalidConfig(_))
                        ));
                    }
                    let mut narrow = state.clone();
                    narrow.config.negative_selection_range = state.training_rows - 1;
                    let loaded = FittedUMAP::from_state(narrow.clone(), backend).unwrap();
                    assert_eq!(loaded.export_state().unwrap(), narrow);
                    if optimizer == Optimizer::Compatibility {
                        assert_eq!(
                            loaded.transform(&small_query).unwrap(),
                            fitted.transform(&small_query).unwrap()
                        );
                    } else {
                        let before = loaded.export_state().unwrap();
                        let transformed = loaded.transform(&small_query).unwrap();
                        assert_eq!(transformed.len(), small_query.len());
                        assert!(transformed.iter().flatten().all(|value| value.is_finite()));
                        assert_eq!(loaded.export_state().unwrap(), before);
                    }
                }
            }
        }
    }
}

#[test]
fn phase4_settings_roundtrip_and_persisted_validation() {
    let mut backends = vec![BackendChoice::Cpu];
    if umap::cuda_backend::device(0).is_some() {
        backends.push(BackendChoice::Cuda { device_index: 0 });
    } else if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
        panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
    }
    assert!(!UMAPConfig::default().exclude_graph_neighbors);
    assert_eq!(
        UMAPConfig::default().negative_sample_scale_adaptation_samples,
        128
    );
    let data = rows();
    let equal_rows_query = rows();
    let small_query = [vec![0.5, 0.5], vec![2.5, 0.5]];
    for backend in backends {
        for optimizer in [
            Optimizer::Compatibility,
            Optimizer::Adam,
            Optimizer::Momentum,
        ] {
            for (exclude, samples) in [(false, 128), (true, 0), (true, i32::MAX as usize)] {
                let config = UMAPConfig {
                    n_neighbors: 3,
                    init: umap::Init::Random,
                    random_state: Some(7),
                    transform_seed: 19,
                    n_epochs: Some(8),
                    optimizer,
                    negative_selection_range: data.len(),
                    exclude_graph_neighbors: exclude,
                    negative_sample_scale_adaptation_samples: samples,
                    ..Default::default()
                };
                let fitted = UMAP::new(config, backend, NeighborSearch::Exact)
                    .fit(&data)
                    .unwrap();
                let state = fitted.export_state().unwrap();
                assert_eq!(state.config.exclude_graph_neighbors, exclude);
                assert_eq!(
                    state.config.negative_sample_scale_adaptation_samples,
                    samples
                );
                let restored = FittedUMAP::from_state(state.clone(), backend).unwrap();
                assert_eq!(restored.export_state().unwrap(), state);
                for query in [&equal_rows_query[..], &small_query[..]] {
                    assert_eq!(
                        restored.transform(query).unwrap(),
                        fitted.transform(query).unwrap()
                    );
                    assert_eq!(restored.export_state().unwrap(), state);
                }
                for invalid_samples in [i32::MAX as usize + 1, usize::MAX] {
                    let mut invalid = state.clone();
                    invalid.config.negative_sample_scale_adaptation_samples = invalid_samples;
                    assert!(matches!(
                        FittedUMAP::from_state(
                            invalid,
                            BackendChoice::Cuda {
                                device_index: usize::MAX
                            }
                        ),
                        Err(UmapError::InvalidConfig(_))
                    ));
                }
            }
        }
    }
}

#[test]
fn rescale_settings_roundtrip_and_do_not_change_fixed_reference_transform() {
    let mut backends = vec![BackendChoice::Cpu];
    if umap::cuda_backend::device(0).is_some() {
        backends.push(BackendChoice::Cuda { device_index: 0 });
    } else if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
        panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
    }
    let mut data = rows();
    data.push(data[1].clone());
    let query = [vec![0.5, 0.5], vec![2.5, 0.5]];
    for backend in backends {
        for search in [NeighborSearch::Exact, NeighborSearch::Approximate] {
            for mode in [
                InitRescale::Isotropic,
                InitRescale::PerAxis,
                InitRescale::GraphEdge,
            ] {
                let config = UMAPConfig {
                    n_neighbors: 3,
                    init: umap::Init::Random,
                    init_rescale: mode,
                    optimizer: umap::Optimizer::Compatibility,
                    init_scale_factor: 2.25,
                    random_state: Some(7),
                    unique: true,
                    n_epochs: Some(8),
                    ..Default::default()
                };
                let fitted = UMAP::new(config, backend, search).fit(&data).unwrap();
                let state = fitted.export_state().unwrap();
                assert_eq!(state.config.init_rescale, mode);
                assert_eq!(state.config.init_scale_factor, 2.25);
                assert_eq!(fitted.embedding()[1], fitted.embedding()[4]);
                let restored = FittedUMAP::from_state(state.clone(), backend).unwrap();
                assert_eq!(restored.export_state().unwrap(), state);
                assert_eq!(restored.embedding(), fitted.embedding());
                let expected = fitted.transform(&query).unwrap();
                assert_eq!(restored.transform(&query).unwrap(), expected);
                // Hold the imported reference embedding and all transform controls fixed.
                for other in [
                    InitRescale::Isotropic,
                    InitRescale::PerAxis,
                    InitRescale::GraphEdge,
                ] {
                    let mut modified = state.clone();
                    modified.config.init_rescale = other;
                    modified.config.init_scale_factor = 0.5;
                    let imported = FittedUMAP::from_state(modified, backend).unwrap();
                    assert_eq!(imported.embedding(), fitted.embedding());
                    assert_eq!(imported.transform(&query).unwrap(), expected);
                }
            }
        }
    }
}

#[test]
fn state_roundtrip_retains_exact_fit_and_transform() {
    let fitted = estimator(NeighborSearch::Exact, false, Some(8))
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    let restored = FittedUMAP::from_state(state.clone(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
    assert_eq!(
        restored.transform(&[vec![0.5, 0.5]]).unwrap(),
        fitted.transform(&[vec![0.5, 0.5]]).unwrap()
    );
    assert_eq!(restored.export_state().unwrap(), state);
}

#[test]
fn unique_roundtrip_restores_original_order() {
    let data = vec![
        vec![1.0, 2.0],
        vec![3.0, 4.0],
        vec![1.0, 2.0],
        vec![5.0, 6.0],
    ];
    let fitted = estimator(NeighborSearch::Exact, true, Some(5))
        .fit(&data)
        .unwrap();
    let restored =
        FittedUMAP::from_state(fitted.export_state().unwrap(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
}

#[test]
fn auto_and_singleton_roundtrip() {
    let fitted = estimator(NeighborSearch::Auto, false, None)
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!(state.requested_search, NeighborSearch::Auto);
    assert_eq!(state.resolved_search, NeighborSearch::Exact);
    FittedUMAP::from_state(state, BackendChoice::Cpu).unwrap();
    let singleton = UMAP::new(
        UMAPConfig {
            n_neighbors: 2,
            init: umap::Init::Random,
            optimizer: umap::Optimizer::Compatibility,
            ..Default::default()
        },
        BackendChoice::Cpu,
        NeighborSearch::Auto,
    )
    .fit(&[vec![1.0, 2.0]])
    .unwrap();
    let restored =
        FittedUMAP::from_state(singleton.export_state().unwrap(), BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), &[vec![0.0, 0.0]]);
}

#[test]
fn approximate_search_and_automatic_epochs_roundtrip() {
    let fitted = estimator(NeighborSearch::Approximate, false, None)
        .fit(&rows())
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!(state.requested_search, NeighborSearch::Approximate);
    assert_eq!(state.resolved_search, NeighborSearch::Approximate);
    let restored = FittedUMAP::from_state(state, BackendChoice::Cpu).unwrap();
    assert_eq!(restored.embedding(), fitted.embedding());
    let query = [vec![0.5, 0.5], vec![2.5, 0.5]];
    assert_eq!(
        restored.transform(&query).unwrap(),
        fitted.transform(&query).unwrap()
    );
}

fn serialization_backends() -> Vec<BackendChoice> {
    let mut backends = vec![BackendChoice::Cpu];
    if umap::cuda_backend::device(0).is_some() {
        backends.push(BackendChoice::Cuda { device_index: 0 });
    } else if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
        panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
    }
    backends
}

/// A 40-row helix whose neighbor graph is connected, so spectral init applies.
fn roundtrip_rows() -> Vec<Vec<f32>> {
    (0..40)
        .map(|row| {
            let angle = row as f32 * 0.37;
            vec![angle.sin(), angle.cos(), row as f32 * 0.05]
        })
        .collect()
}

fn assert_roundtrip(config: UMAPConfig, backend: BackendChoice, expected: (umap::Init, Optimizer)) {
    let data = roundtrip_rows();
    let query = [vec![0.5, 0.5, 0.1], vec![-0.5, 0.8, 1.2], data[3].clone()];
    let fitted = UMAP::new(config, backend, NeighborSearch::Exact)
        .fit(&data)
        .unwrap();
    let state = fitted.export_state().unwrap();
    assert_eq!((state.config.init, state.config.optimizer), expected);
    let restored = FittedUMAP::from_state(state.clone(), backend).unwrap();
    assert_eq!(restored.export_state().unwrap(), state);
    assert_eq!(restored.embedding(), fitted.embedding());
    let transformed = fitted.transform(&query).unwrap();
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));
    assert_eq!(restored.transform(&query).unwrap(), transformed);
}

#[test]
fn default_fit_roundtrips_auto_adam() {
    for backend in serialization_backends() {
        assert_roundtrip(
            UMAPConfig {
                n_neighbors: 5,
                n_epochs: Some(20),
                random_state: Some(7),
                transform_seed: 7,
                ..Default::default()
            },
            backend,
            (umap::Init::Auto, Optimizer::Adam),
        );
    }
}

#[test]
fn explicit_compatibility_spectral_roundtrips() {
    // The state an old-default (classic) fit persists loads and transforms as such.
    for backend in serialization_backends() {
        assert_roundtrip(
            UMAPConfig {
                n_neighbors: 5,
                n_epochs: Some(20),
                random_state: Some(7),
                transform_seed: 7,
                init: umap::Init::Spectral,
                optimizer: Optimizer::Compatibility,
                ..Default::default()
            },
            backend,
            (umap::Init::Spectral, Optimizer::Compatibility),
        );
    }
}

fn invalid(mut state: ModelState) {
    assert!(FittedUMAP::from_state(state.clone(), BackendChoice::Cpu).is_err());
    state.config.n_neighbors = 0;
    assert!(FittedUMAP::from_state(state, BackendChoice::Cpu).is_err());
}

#[test]
fn rejects_malformed_arrays_graph_mapping_and_search() {
    let valid = estimator(NeighborSearch::Exact, false, Some(4))
        .fit(&rows())
        .unwrap()
        .export_state()
        .unwrap();
    let mut state = valid.clone();
    state.n_features += 1;
    invalid(state);
    let mut state = valid.clone();
    state.embedding[0] = f32::NAN;
    invalid(state);
    let mut state = valid.clone();
    state.original_to_training = Some(vec![0]);
    invalid(state);
    let approximate = estimator(NeighborSearch::Approximate, false, Some(4))
        .fit(&rows())
        .unwrap()
        .export_state()
        .unwrap();
    let mut state = approximate.clone();
    state.query_graph.as_mut().unwrap()[0] = -1;
    invalid(state);
    let mut state = valid.clone();
    state.resolved_search = NeighborSearch::Approximate;
    invalid(state);

    let mut state = approximate.clone();
    state.config.n_neighbors = 2;
    state.training_rows = 2;
    state.embedding_rows = 2;
    state.training_features.truncate(4);
    state.embedding.truncate(2 * state.n_components);
    state.query_graph = Some(Vec::new());
    state.query_graph_width = 0;
    assert!(FittedUMAP::from_state(state, BackendChoice::Cpu).is_err());

    for state in [valid, approximate] {
        for (a, b) in [
            (0.0, state.b),
            (state.a, 0.0),
            (-1.0, state.b),
            (state.a, -1.0),
        ] {
            let mut malformed = state.clone();
            malformed.a = a;
            malformed.b = b;
            assert!(FittedUMAP::from_state(malformed, BackendChoice::Cpu).is_err());
        }
    }
}
