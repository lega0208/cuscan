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
    assert_eq!(first.len(), distinct.len(), "all unique rows are retained");
    for (row, &index) in rows.iter().zip(&inverse) {
        assert_eq!(row, &first[index], "inverse map preserves input row order");
    }
    let settings = UMAPConfig {
        unique: true,
        init: Init::Random,
        n_epochs: Some(10),
        learning_rate: 1.0,
        ..config()
    };
    let (unique, returned) = UMAP::new(settings, BackendChoice::Cpu, NeighborSearch::Exact)
        .fit_transform(&rows)
        .unwrap();
    assert_eq!(&returned, unique.embedding());
    assert_eq!(unique.embedding().len(), rows.len());
    let embedding = unique.embedding();
    assert_eq!(embedding.len(), rows.len());
    assert!(embedding.iter().all(|row| {
        row.len() == config().n_components && row.iter().all(|value| value.is_finite())
    }));
    for (row_index, (&index, row)) in inverse.iter().zip(embedding).enumerate() {
        let first_position = inverse[..row_index]
            .iter()
            .position(|&earlier| earlier == index);
        if let Some(first_position) = first_position {
            assert_eq!(row, &embedding[first_position]);
        }
    }
    let retained_embedding = embedding.clone();
    let transformed = unique.transform(&distinct[..5]).unwrap();
    assert_eq!(unique.embedding(), &retained_embedding);
    assert_eq!(transformed.len(), 5);
    assert!(transformed.iter().all(|row| {
        row.len() == config().n_components && row.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn unique_signed_zero_and_validation_are_exact() {
    let estimator = UMAP::new(
        UMAPConfig {
            unique: true,
            init: Init::Random,
            ..config()
        },
        BackendChoice::Cpu,
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
        BackendChoice::Cpu,
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
    UMAP::new(config, BackendChoice::Cpu, search).fit(&clustered_data())
}

#[test]
fn singleton_and_two_row_cases_are_explicit() {
    let singleton_config = UMAPConfig {
        n_components: 3,
        ..config()
    };
    for search in [
        NeighborSearch::Exact,
        NeighborSearch::Approximate,
        NeighborSearch::Auto,
    ] {
        let fitted = UMAP::new(singleton_config.clone(), BackendChoice::Cpu, search)
            .fit(&[vec![1.0, 2.0]])
            .unwrap();
        assert_eq!(fitted.embedding(), &[vec![0.0, 0.0, 0.0]]);
        assert_eq!(
            fitted.transform(&[vec![1.0, 2.0]]).unwrap_err().to_string(),
            "Transform unavailable when model was fit with only a single data sample.",
            "single-sample transform error for {search:?}"
        );
    }
    let error = UMAP::new(config(), BackendChoice::Cpu, NeighborSearch::Exact)
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
        UMAP::new(bad_config, BackendChoice::Cpu, NeighborSearch::Exact)
            .fit(&[vec![0.0]])
            .is_err()
    );
    let estimator = UMAP::new(config(), BackendChoice::Cpu, NeighborSearch::Exact);
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
        UMAP::new(exact_config, BackendChoice::Cpu, NeighborSearch::Exact)
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
fn transform_q1_and_65_are_finite_and_repeatable_for_explicit_search_modes() {
    let data: Vec<Vec<f32>> = (0..16)
        .map(|row| {
            let value = row as f32;
            vec![value, (value * 0.31).sin()]
        })
        .collect();
    let config = UMAPConfig {
        n_neighbors: 15,
        n_epochs: Some(0),
        init: Init::Random,
        random_state: Some(41),
        ..Default::default()
    };

    for search in [NeighborSearch::Exact, NeighborSearch::Approximate] {
        let fitted = UMAP::new(config.clone(), BackendChoice::Cpu, search)
            .fit(&data)
            .unwrap();
        for rows in [1, 65] {
            let query: Vec<Vec<f32>> = (0..rows)
                .map(|row| {
                    let value = row as f32;
                    vec![value * 0.17 + 0.25, (value * 0.23).cos()]
                })
                .collect();
            let first = fitted.transform(&query).unwrap();
            assert_eq!(first.len(), rows, "{search:?}, {rows} query rows");
            assert!(
                first
                    .iter()
                    .all(|row| row.len() == 2 && row.iter().all(|value| value.is_finite())),
                "{search:?}, {rows} query rows"
            );
            let repeated = fitted.transform(&query).unwrap();
            let bits = |embedding: &[Vec<f32>]| {
                embedding
                    .iter()
                    .map(|row| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                bits(&first),
                bits(&repeated),
                "{search:?}, {rows} query rows"
            );
        }
    }
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
    let fitted = UMAP::new(active, BackendChoice::Cpu, NeighborSearch::Exact)
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
    let (fitted, embedding) = UMAP::new(config(), BackendChoice::Cpu, NeighborSearch::Exact)
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

#[cfg(feature = "tracing")]
#[test]
fn query_diagnostics_preserve_output_and_expose_actual_neighbors() {
    let data = clustered_data();
    let query = vec![vec![-8.0, 0.0], vec![8.0, 0.0], vec![0.0, 12.0]];
    for search in [
        NeighborSearch::Exact,
        NeighborSearch::Approximate,
        NeighborSearch::Auto,
    ] {
        let fitted = UMAP::new(config(), BackendChoice::Cpu, search)
            .fit(&data)
            .unwrap();
        let original = fitted.embedding().clone();
        let expected = fitted.transform(&query).unwrap();
        let (actual, indices) = fitted.diagnostic_transform(&query).unwrap();
        let bits = |embedding: &Vec<Vec<f32>>| {
            embedding
                .iter()
                .flatten()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&expected), bits(&actual));
        assert_eq!(fitted.embedding(), &original);
        assert_eq!(indices.len(), query.len() * config().n_neighbors);
        for neighbors in indices.chunks_exact(config().n_neighbors) {
            let unique: std::collections::HashSet<_> = neighbors.iter().copied().collect();
            assert_eq!(unique.len(), config().n_neighbors);
            assert!(
                neighbors
                    .iter()
                    .all(|&id| id >= 0 && (id as usize) < data.len())
            );
        }
        let invalid = vec![vec![f32::NAN, 0.0]];
        assert_eq!(
            fitted.transform(&invalid).unwrap_err().to_string(),
            fitted
                .diagnostic_transform(&invalid)
                .unwrap_err()
                .to_string()
        );
    }
}
