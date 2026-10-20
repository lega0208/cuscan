use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

#[cfg(feature = "cpu")]
fn estimator_test_backend() -> BackendChoice {
    BackendChoice::Cpu
}

#[cfg(all(not(feature = "cpu"), feature = "cuda"))]
fn estimator_test_backend() -> BackendChoice {
    BackendChoice::Cuda { device_index: 0 }
}

#[cfg(not(any(feature = "cpu", feature = "cuda")))]
fn estimator_test_backend() -> BackendChoice {
    BackendChoice::Auto
}

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
    let (unique, returned) = UMAP::new(settings, estimator_test_backend(), NeighborSearch::Exact)
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
        estimator_test_backend(),
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
        estimator_test_backend(),
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
    UMAP::new(config, estimator_test_backend(), search).fit(&clustered_data())
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
        let fitted = UMAP::new(singleton_config.clone(), estimator_test_backend(), search)
            .fit(&[vec![1.0, 2.0]])
            .unwrap();
        assert_eq!(fitted.embedding(), &[vec![0.0, 0.0, 0.0]]);
        assert_eq!(
            fitted.transform(&[vec![1.0, 2.0]]).unwrap_err().to_string(),
            "Transform unavailable when model was fit with only a single data sample.",
            "single-sample transform error for {search:?}"
        );
    }
    let error = UMAP::new(config(), estimator_test_backend(), NeighborSearch::Exact)
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
        UMAP::new(bad_config, estimator_test_backend(), NeighborSearch::Exact)
            .fit(&[vec![0.0]])
            .is_err()
    );
    let estimator = UMAP::new(config(), estimator_test_backend(), NeighborSearch::Exact);
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
        UMAP::new(
            exact_config,
            estimator_test_backend(),
            NeighborSearch::Exact
        )
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
        let fitted = UMAP::new(config.clone(), estimator_test_backend(), search)
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
    let fitted = UMAP::new(active, estimator_test_backend(), NeighborSearch::Exact)
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
    let (fitted, embedding) = UMAP::new(config(), estimator_test_backend(), NeighborSearch::Exact)
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
        let fitted = UMAP::new(config(), estimator_test_backend(), search)
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

// Ignored seeded numerical diagnostic. It records fuzzy graph weights and exact/approximate
// fit and transform embeddings for fixed input and seed 42, repeated three times per case.
// It asserts only shape, finiteness, and workload identity; drift and repeat spread are
// reported as data for comparison across builds. Run it explicitly, for example:
//   FAST_UMAP_DRIFT_OUT=/abs/path/outside/repo/drift-before-cpu.json \
//     cargo test --locked --no-default-features --features cpu --test test_standard_estimator \
//     seeded_drift_diagnostic -- --ignored --nocapture
// Optional FAST_UMAP_DRIFT_BASELINE=/abs/path/to/previous.json compares against a prior capture.
// FAST_UMAP_DRIFT_LABEL is an optional free-text tag stored in the output.
#[cfg(any(feature = "cpu", feature = "cuda"))]
mod seeded_drift {
    use burn::backend::Dispatch;
    use burn::tensor::{Device, Int, Tensor, TensorData};
    use fast_umap::backend::StandardUmapOps;
    use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};
    use serde_json::{Value, json};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    const SCHEMA: &str = "fast-umap.seeded-drift.v1";
    const SEED: u64 = 42;
    const REPEATS: usize = 3;
    const FIT_ROWS: usize = 512;
    const QUERY_ROWS: usize = 64;
    const FEATURES: usize = 16;
    const NEIGHBORS: usize = 8;
    const EPOCHS: usize = 50;
    const COMPONENTS: usize = 2;

    #[cfg(feature = "cpu")]
    fn drift_backend() -> (&'static str, BackendChoice, Device) {
        ("cpu", BackendChoice::Cpu, Device::cpu())
    }

    #[cfg(all(not(feature = "cpu"), feature = "cuda"))]
    fn drift_backend() -> (&'static str, BackendChoice, Device) {
        let device = fast_umap::cuda_backend::device(0)
            .expect("seeded drift diagnostic requires a CUDA device");
        ("cuda", BackendChoice::Cuda { device_index: 0 }, device)
    }

    fn drift_data(rows: usize, features: usize) -> Vec<Vec<f32>> {
        (0..rows)
            .map(|row| {
                let cluster = row % 4;
                (0..features)
                    .map(|feature| {
                        let center = cluster as f32 * 4.0;
                        let wave = ((row * (feature + 3) + feature * 11) as f32 * 0.173).sin();
                        center + feature as f32 * 0.23 + wave * 0.08
                    })
                    .collect()
            })
            .collect()
    }

    fn drift_config() -> UMAPConfig {
        UMAPConfig {
            n_components: COMPONENTS,
            n_neighbors: NEIGHBORS,
            n_epochs: Some(EPOCHS),
            init: Init::Random,
            random_state: Some(SEED),
            transform_seed: SEED,
            ..UMAPConfig::default()
        }
    }

    /// Host brute-force kNN (self excluded, ties by index), the same input contract as
    /// the estimator's exact neighbor search: Euclidean distances, row-major `[n, k]`.
    fn host_knn(data: &[Vec<f32>], k: usize) -> (Vec<i32>, Vec<f32>) {
        let mut indices = Vec::with_capacity(data.len() * k);
        let mut distances = Vec::with_capacity(data.len() * k);
        for (row, query) in data.iter().enumerate() {
            let mut candidates: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .filter(|(other, _)| *other != row)
                .map(|(other, value)| {
                    let squared = query.iter().zip(value).fold(0.0f32, |sum, (a, b)| {
                        let difference = a - b;
                        sum + difference * difference
                    });
                    (squared, other)
                })
                .collect();
            candidates.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
            for &(squared, other) in candidates.iter().take(k) {
                indices.push(i32::try_from(other).unwrap());
                distances.push(squared.sqrt());
            }
        }
        (indices, distances)
    }

    /// Fuzzy graph weights through the same `StandardUmapOps::fuzzy_graph` route as the graph tests.
    fn fuzzy_weights(
        device: &Device,
        indices: &[i32],
        distances: &[f32],
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let indices =
            Tensor::<2, Int>::from_data(TensorData::new(indices.to_vec(), [n, k]), device);
        let distances = Tensor::<2>::from_data(TensorData::new(distances.to_vec(), [n, k]), device);
        let (_, _, weights, _, _, _) = <Dispatch as StandardUmapOps>::fuzzy_graph(
            indices.into_dispatch(),
            distances.into_dispatch(),
            1.0,
            1.0,
            1.0,
            false,
            u32::try_from(n).unwrap(),
        );
        Tensor::<1>::from_dispatch(weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    fn flatten(embedding: &[Vec<f32>]) -> Vec<f32> {
        embedding.iter().flatten().copied().collect()
    }

    /// Run one case `REPEATS` times, returning every repeat's output and wall time in ms.
    fn repeat_case(mut run: impl FnMut() -> Vec<f32>) -> (Vec<Vec<f32>>, Vec<f64>) {
        let mut values = Vec::with_capacity(REPEATS);
        let mut wall = Vec::with_capacity(REPEATS);
        for _ in 0..REPEATS {
            let start = Instant::now();
            values.push(run());
            wall.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        (values, wall)
    }

    /// Returns (max absolute difference, mean absolute difference, bit mismatch count).
    fn compare(actual: &[f32], expected: &[f32]) -> (f64, f64, usize) {
        assert_eq!(actual.len(), expected.len(), "comparison length");
        let mut max_abs = 0f64;
        let mut sum_abs = 0f64;
        let mut bit_mismatches = 0usize;
        for (&a, &e) in actual.iter().zip(expected) {
            let difference = (f64::from(a) - f64::from(e)).abs();
            max_abs = max_abs.max(difference);
            sum_abs += difference;
            bit_mismatches += usize::from(a.to_bits() != e.to_bits());
        }
        let mean_abs = if actual.is_empty() {
            0.0
        } else {
            sum_abs / actual.len() as f64
        };
        (max_abs, mean_abs, bit_mismatches)
    }

    fn summarize_case(
        name: &str,
        shape: [usize; 2],
        repeats: &[Vec<f32>],
        wall_ms: &[f64],
        baseline: Option<&Value>,
    ) -> Value {
        let expected_len = shape[0] * shape[1];
        for (index, values) in repeats.iter().enumerate() {
            assert_eq!(values.len(), expected_len, "{name} repeat {index} shape");
            assert!(
                values.iter().all(|value| value.is_finite()),
                "{name} repeat {index} contains a non-finite value"
            );
        }
        let mut spread_max = 0f64;
        let mut spread_mean = 0f64;
        let mut spread_bits = 0usize;
        for values in &repeats[1..] {
            let (max_abs, mean_abs, bits) = compare(values, &repeats[0]);
            spread_max = spread_max.max(max_abs);
            spread_mean += mean_abs;
            spread_bits += bits;
        }
        if repeats.len() > 1 {
            spread_mean /= (repeats.len() - 1) as f64;
        }
        let mut case = json!({
            "shape": [shape[0], shape[1]],
            "wall_ms": wall_ms,
            "repeat_spread": {
                "max_abs": spread_max,
                "mean_abs": spread_mean,
                "bit_mismatches": spread_bits,
            },
            "repeats": repeats,
            "baseline_drift": Value::Null,
        });
        if let Some(baseline) = baseline {
            let base_case = &baseline["cases"][name];
            let base_shape: Vec<u64> = base_case["shape"]
                .as_array()
                .unwrap_or_else(|| panic!("baseline {name} has no shape"))
                .iter()
                .map(|value| value.as_u64().unwrap())
                .collect();
            assert_eq!(
                base_shape,
                vec![shape[0] as u64, shape[1] as u64],
                "{name} shape differs from baseline"
            );
            let base_first: Vec<f32> = base_case["repeats"][0]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let per_repeat: Vec<Value> = repeats
                .iter()
                .map(|values| {
                    let (max_abs, mean_abs, bits) = compare(values, &base_first);
                    json!({"max_abs": max_abs, "mean_abs": mean_abs, "bit_mismatches": bits})
                })
                .collect();
            case["baseline_drift"] = json!({
                "reference": "repeat 0 of baseline",
                "per_repeat": per_repeat,
            });
        }
        case
    }

    fn required_outside_repo_output() -> PathBuf {
        let raw = std::env::var_os("FAST_UMAP_DRIFT_OUT").expect(
            "FAST_UMAP_DRIFT_OUT must name an absolute output JSON path outside the repository",
        );
        let path = PathBuf::from(raw);
        assert!(path.is_absolute(), "FAST_UMAP_DRIFT_OUT must be absolute");
        let parent = path
            .parent()
            .and_then(|parent| parent.canonicalize().ok())
            .expect("FAST_UMAP_DRIFT_OUT parent directory must exist");
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        assert!(
            !parent.starts_with(&repo),
            "FAST_UMAP_DRIFT_OUT must be outside the repository ({})",
            repo.display()
        );
        assert!(
            !path.exists(),
            "refusing to overwrite existing drift evidence {}",
            path.display()
        );
        path
    }

    fn optional_baseline(backend_name: &str, workload: &Value) -> Option<Value> {
        let raw = std::env::var_os("FAST_UMAP_DRIFT_BASELINE")?;
        let path = PathBuf::from(raw);
        assert!(
            path.is_absolute(),
            "FAST_UMAP_DRIFT_BASELINE must be absolute"
        );
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("could not read {}: {error}", path.display()));
        let baseline: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(baseline["schema"], SCHEMA, "baseline schema");
        assert_eq!(baseline["backend"], backend_name, "baseline backend");
        assert_eq!(baseline["workload"], *workload, "baseline workload");
        Some(baseline)
    }

    #[test]
    #[ignore = "numerical drift diagnostic; requires FAST_UMAP_DRIFT_OUT and --ignored"]
    fn seeded_drift_diagnostic() {
        let (backend_name, backend, device) = drift_backend();
        let output = required_outside_repo_output();
        let workload = json!({
            "seed": SEED,
            "repeats": REPEATS,
            "fit_rows": FIT_ROWS,
            "query_rows": QUERY_ROWS,
            "features": FEATURES,
            "neighbors": NEIGHBORS,
            "epochs": EPOCHS,
            "components": COMPONENTS,
            "approximate_transform_queue_size": 4.0,
        });
        let baseline = optional_baseline(backend_name, &workload);
        let all = drift_data(FIT_ROWS + QUERY_ROWS, FEATURES);
        let (fit_rows, query_rows) = all.split_at(FIT_ROWS);
        let started = Instant::now();
        let mut cases = serde_json::Map::new();
        let mut record = |name: &str, shape: [usize; 2], values: &[Vec<f32>], wall: &[f64]| {
            let summary = summarize_case(name, shape, values, wall, baseline.as_ref());
            eprintln!("{name}: {summary}");
            cases.insert(name.to_owned(), summary);
        };

        let (indices, distances) = host_knn(fit_rows, NEIGHBORS);
        let (weights, wall) =
            repeat_case(|| fuzzy_weights(&device, &indices, &distances, FIT_ROWS, NEIGHBORS));
        record(
            "fuzzy_graph_weights",
            [1, FIT_ROWS * NEIGHBORS * 2],
            &weights,
            &wall,
        );

        for (search, label) in [
            (NeighborSearch::Exact, "exact"),
            (NeighborSearch::Approximate, "approximate"),
        ] {
            let estimator = UMAP::new(drift_config(), backend, search);
            let (fit_values, fit_wall) =
                repeat_case(|| flatten(estimator.fit(fit_rows).unwrap().embedding()));
            record(
                &format!("{label}_fit_embedding"),
                [FIT_ROWS, COMPONENTS],
                &fit_values,
                &fit_wall,
            );

            // Transform repeats reuse one fitted model so that only query work is repeated.
            let fitted = estimator.fit(fit_rows).unwrap();
            let (transform_values, transform_wall) =
                repeat_case(|| flatten(&fitted.transform(query_rows).unwrap()));
            record(
                &format!("{label}_transform_embedding"),
                [QUERY_ROWS, COMPONENTS],
                &transform_values,
                &transform_wall,
            );
        }

        let document = json!({
            "schema": SCHEMA,
            "label": std::env::var("FAST_UMAP_DRIFT_LABEL").unwrap_or_else(|_| "unlabeled".into()),
            "backend": backend_name,
            "workload": workload,
            "baseline": std::env::var("FAST_UMAP_DRIFT_BASELINE").ok(),
            "total_wall_ms": started.elapsed().as_secs_f64() * 1000.0,
            "cases": Value::Object(cases),
        });
        std::fs::write(&output, serde_json::to_string_pretty(&document).unwrap())
            .unwrap_or_else(|error| panic!("could not write {}: {error}", output.display()));
        eprintln!("wrote {}", output.display());
    }
}
