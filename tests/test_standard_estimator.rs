use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

#[test]
fn unique_fit_expands_original_order_and_preserves_transform_state() {
    let distinct = clustered_data();
    let order: Vec<usize> = (0..120).map(|i| (i * 17) % distinct.len()).collect();
    let rows: Vec<Vec<f32>> = order.iter().map(|&i| distinct[i].clone()).collect();
    // Build manual distinct rows in the same first-occurrence order.
    let mut first = Vec::new();
    let mut inverse = Vec::new();
    for row in &rows {
        let index = first
            .iter()
            .position(|value| value == row)
            .unwrap_or_else(|| {
                first.push(row.clone());
                first.len() - 1
            });
        inverse.push(index);
    }
    let settings = UMAPConfig {
        init: Init::Random,
        n_epochs: Some(10),
        ..config()
    };
    let manual = UMAP::new(settings.clone(), BackendChoice::Flex, NeighborSearch::Exact)
        .fit(&first)
        .unwrap();
    let (unique, returned) = UMAP::new(
        UMAPConfig {
            unique: true,
            ..settings
        },
        BackendChoice::Flex,
        NeighborSearch::Exact,
    )
    .fit_transform(&rows)
    .unwrap();
    assert_eq!(&returned, unique.embedding());
    assert_eq!(unique.embedding().len(), rows.len());
    for (row, &index) in unique.embedding().iter().zip(&inverse) {
        assert_eq!(row, &manual.embedding()[index]);
    }
    assert_eq!(
        unique.transform(&distinct[..5]).unwrap(),
        manual.transform(&distinct[..5]).unwrap()
    );
}

#[test]
fn unique_signed_zero_and_validation_are_exact() {
    let estimator = UMAP::new(
        UMAPConfig {
            unique: true,
            init: Init::Random,
            ..config()
        },
        BackendChoice::Flex,
        NeighborSearch::Exact,
    );
    let rows = vec![
        vec![0.0, 1.0],
        vec![-0.0, 1.0],
        vec![2.0, 3.0],
        vec![4.0, 5.0],
    ];
    let fitted = estimator.fit(&rows).unwrap();
    assert_eq!(fitted.embedding()[0], fitted.embedding()[1]);
    assert!(
        estimator
            .fit(&[vec![0.0], vec![-0.0]])
            .unwrap_err()
            .to_string()
            .contains("two distinct")
    );
    assert!(
        estimator
            .fit(&[vec![0.0], vec![f32::NAN]])
            .unwrap_err()
            .to_string()
            .contains("not finite")
    );
    assert!(
        estimator
            .fit(&[vec![0.0], vec![0.0, 1.0]])
            .unwrap_err()
            .to_string()
            .contains("expected")
    );
    assert!(
        estimator
            .fit(&[vec![0.0], vec![1.0]])
            .unwrap_err()
            .to_string()
            .contains("graph is empty")
    );
    let raw = UMAP::new(
        UMAPConfig {
            init: Init::Random,
            ..config()
        },
        BackendChoice::Flex,
        NeighborSearch::Exact,
    )
    .fit(&rows)
    .unwrap();
    assert_eq!(raw.embedding().len(), rows.len());
    assert_ne!(raw.embedding()[0], raw.embedding()[1]);
}

fn clustered_data() -> Vec<Vec<f32>> {
    (0..60)
        .map(|index| {
            let cluster = index / 20;
            let angle = (index % 20) as f32 * 0.41;
            let center = match cluster {
                0 => [-8.0, 0.0],
                1 => [8.0, 0.0],
                _ => [0.0, 12.0],
            };
            vec![
                center[0] + angle.sin() * 0.35,
                center[1] + angle.cos() * 0.35,
            ]
        })
        .collect()
}

fn config() -> UMAPConfig {
    UMAPConfig {
        n_neighbors: 8,
        n_epochs: Some(0),
        init: Init::Spectral,
        random_state: Some(17),
        ..Default::default()
    }
}

fn fit(search: NeighborSearch, config: UMAPConfig) -> fast_umap::Result<fast_umap::FittedUMAP> {
    UMAP::new(config, BackendChoice::Flex, search).fit(&clustered_data())
}

#[test]
fn singleton_and_two_row_cases_are_explicit() {
    let singleton_config = UMAPConfig {
        n_components: 3,
        ..config()
    };
    let fitted = UMAP::new(singleton_config, BackendChoice::Flex, NeighborSearch::Exact)
        .fit(&[vec![1.0, 2.0]])
        .unwrap();
    assert_eq!(fitted.embedding(), &[vec![0.0, 0.0, 0.0]]);
    assert!(
        fitted
            .transform(&[vec![1.0, 2.0]])
            .unwrap_err()
            .to_string()
            .contains("only a single data sample")
    );
    let error = UMAP::new(config(), BackendChoice::Flex, NeighborSearch::Exact)
        .fit(&[vec![0.0], vec![1.0]])
        .unwrap_err();
    assert!(error.to_string().contains("graph is empty"));
}

#[test]
fn invalid_config_and_input_are_reported_as_errors() {
    let bad_config = UMAPConfig {
        n_components: 0,
        ..Default::default()
    };
    assert!(
        UMAP::new(bad_config, BackendChoice::Flex, NeighborSearch::Exact)
            .fit(&[vec![0.0]])
            .is_err()
    );
    let estimator = UMAP::new(config(), BackendChoice::Flex, NeighborSearch::Exact);
    assert!(estimator.fit(&[]).is_err());
    assert!(estimator.fit(&[vec![0.0], vec![f32::NAN]]).is_err());
    assert!(estimator.fit(&[vec![0.0], vec![1.0, 2.0]]).is_err());
}

#[test]
fn exact_and_approximate_fit_are_finite_deterministic_and_clustered() {
    for search in [NeighborSearch::Exact, NeighborSearch::Approximate] {
        let first = fit(search, config()).unwrap();
        let second = fit(search, config()).unwrap();
        assert_eq!(first.embedding(), second.embedding());
        let embedding = first.embedding();
        assert_eq!(embedding.len(), 60);
        assert!(
            embedding
                .iter()
                .all(|row| row.len() == 2 && row.iter().all(|value| value.is_finite()))
        );
        let centroids: Vec<[f32; 2]> = (0..3)
            .map(|cluster| {
                let rows = &embedding[cluster * 20..cluster * 20 + 20];
                [
                    rows.iter().map(|row| row[0]).sum::<f32>() / 20.0,
                    rows.iter().map(|row| row[1]).sum::<f32>() / 20.0,
                ]
            })
            .collect();
        let within = (0..3)
            .map(|cluster| {
                let center = centroids[cluster];
                embedding[cluster * 20..cluster * 20 + 20]
                    .iter()
                    .map(|row| ((row[0] - center[0]).powi(2) + (row[1] - center[1]).powi(2)).sqrt())
                    .sum::<f32>()
                    / 20.0
            })
            .fold(0.0f32, f32::max);
        let mut separation = f32::INFINITY;
        for left in 0..3 {
            for right in left + 1..3 {
                separation = separation.min(
                    ((centroids[left][0] - centroids[right][0]).powi(2)
                        + (centroids[left][1] - centroids[right][1]).powi(2))
                    .sqrt(),
                );
            }
        }
        assert!(
            separation > 2.0 * within,
            "clusters not separated: {separation} <= {within}"
        );
    }
}

#[test]
fn transform_uses_fitted_embedding_and_search_settings() {
    let fitted = fit(NeighborSearch::Exact, config()).unwrap();
    assert_eq!(fitted.n_features(), 2);
    let result = fitted
        .transform(&[vec![-8.0, 0.0], vec![8.0, 0.0], vec![0.0, 12.0]])
        .unwrap();
    assert_eq!(result.len(), 3);
    assert!(result.iter().flatten().all(|value| value.is_finite()));
    for (query, expected_cluster) in result.iter().zip(0..3) {
        let nearest = (0..3)
            .min_by(|&left, &right| {
                let distance = |cluster: usize| {
                    let rows = &fitted.embedding()[cluster * 20..cluster * 20 + 20];
                    let centroid = [
                        rows.iter().map(|row| row[0]).sum::<f32>() / 20.0,
                        rows.iter().map(|row| row[1]).sum::<f32>() / 20.0,
                    ];
                    (query[0] - centroid[0]).powi(2) + (query[1] - centroid[1]).powi(2)
                };
                distance(left).total_cmp(&distance(right))
            })
            .unwrap();
        assert_eq!(
            nearest, expected_cluster,
            "transformed point landed in the wrong cluster"
        );
    }
    let bad_query = fitted.transform(&[vec![0.0]]).unwrap_err();
    assert!(bad_query.to_string().contains("expected 2"));

    let exact_config = UMAPConfig {
        transform_queue_size: 3.0,
        ..config()
    };
    assert!(
        UMAP::new(exact_config, BackendChoice::Flex, NeighborSearch::Exact)
            .fit(&clustered_data())
            .is_err()
    );
    let approx_config = UMAPConfig {
        transform_queue_size: 2.0,
        ..config()
    };
    let approx = fit(NeighborSearch::Approximate, approx_config).unwrap();
    assert_eq!(approx.transform(&[vec![8.0, 0.0]]).unwrap().len(), 1);
}

#[test]
fn zero_epochs_returns_scaled_initialization() {
    let fitted = fit(NeighborSearch::Exact, config()).unwrap();
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| (0.0..=10.0).contains(value))
    );
}

#[test]
fn fit_and_transform_run_sgd_epochs() {
    let data: Vec<Vec<f32>> = (0..12)
        .map(|row| {
            let offset = if row < 6 { 0.0 } else { 5.0 };
            vec![offset + row as f32 * 0.03, (row as f32 * 0.7).sin()]
        })
        .collect();
    let active = UMAPConfig {
        n_neighbors: 4,
        n_epochs: Some(3),
        init: Init::Random,
        random_state: Some(31),
        ..Default::default()
    };
    let fitted = UMAP::new(active, BackendChoice::Flex, NeighborSearch::Exact)
        .fit(&data)
        .unwrap();
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    let transformed = fitted.transform(&[vec![0.1, 0.0], vec![5.1, 0.0]]).unwrap();
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn fit_transform_returns_reusable_model_and_matching_embedding() {
    let (fitted, embedding) = UMAP::new(config(), BackendChoice::Flex, NeighborSearch::Exact)
        .fit_transform(&clustered_data())
        .unwrap();
    assert_eq!(embedding.len(), 60);
    assert!(
        embedding
            .iter()
            .all(|row| row.len() == 2 && row.iter().all(|value| value.is_finite()))
    );
    assert_eq!(fitted.embedding(), &embedding);
    let transformed = fitted
        .transform(&[vec![-8.0, 0.0], vec![8.0, 0.0], vec![0.0, 12.0]])
        .unwrap();
    assert_eq!(transformed.len(), 3);
    assert!(transformed.iter().flatten().all(|value| value.is_finite()));
}

#[test]
fn auto_search_fits_small_inputs_with_exact_search() {
    let auto = fit(NeighborSearch::Auto, config()).unwrap();
    assert!(
        auto.embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
    // 60 rows is below AUTO_EXACT_MAX_ROWS, so Auto must reproduce the exact path.
    let exact = fit(NeighborSearch::Exact, config()).unwrap();
    assert_eq!(auto.embedding(), exact.embedding());
    assert!(
        auto.transform(&[vec![8.0, 0.0]])
            .unwrap()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
}

#[cfg(feature = "cuda")]
#[test]
fn standard_estimator_smoke_runs_on_requested_cuda_device() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    drop(device);
    let fitted = UMAP::new(
        config(),
        BackendChoice::Cuda { device_index: 0 },
        NeighborSearch::Exact,
    )
    .fit(&clustered_data())
    .unwrap();
    assert_eq!(fitted.embedding().len(), 60);
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
}
