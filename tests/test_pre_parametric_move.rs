
#![cfg(feature = "parametric")]

use burn::tensor::Device;
use fast_umap::prelude::*;
use fast_umap::{FittedParametricUMAP, ParametricUMAP};

// Captured before the namespace refactor with the old Umap/FittedUmap API:
// cargo run --no-default-features --example pre_parametric_move_capture.
// The old saver produced the retained Burnpack independently of this test.
// Do not regenerate these fixtures with the renamed implementation.
const TOLERANCE: f32 = 1e-5;

fn assert_vectors_close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_row, expected_row) in actual.iter().zip(expected) {
        assert_eq!(actual_row.len(), expected_row.len());
        for (actual, expected) in actual_row.iter().zip(expected_row) {
            assert!(
                (actual - expected).abs() <= TOLERANCE,
                "{actual} differs from {expected} by more than {TOLERANCE}"
            );
        }
    }
}

fn as_f32(rows: Vec<Vec<f64>>) -> Vec<Vec<f32>> {
    rows.into_iter()
        .map(|row| row.into_iter().map(|value| value as f32).collect())
        .collect()
}

#[test]
fn pre_parametric_move_weights_and_fit_outputs_remain_compatible() {
    let data = vec![
        vec![0.0, 0.1, 0.2],
        vec![0.2, 0.0, 0.3],
        vec![0.4, 0.2, 0.1],
        vec![0.6, 0.4, 0.2],
        vec![0.8, 0.5, 0.4],
        vec![1.0, 0.7, 0.5],
        vec![1.2, 0.8, 0.6],
        vec![1.4, 1.0, 0.8],
    ];
    let unseen = vec![vec![0.15, 0.12, 0.22], vec![0.9, 0.6, 0.45]];
    let config = ParametricUMAPConfig {
        n_components: 2,
        hidden_sizes: vec![8],
        graph: GraphParams {
            n_neighbors: 3,
            ..Default::default()
        },
        optimization: ParametricUMAPOptimizationParams {
            n_epochs: 12,
            seed: 24680,
            ..Default::default()
        },
        ..Default::default()
    };
    let expected: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/pre_parametric_move_vectors.json")).unwrap();
    let expected_fit: Vec<Vec<f32>> = serde_json::from_value(expected["fit"].clone()).unwrap();
    let expected_unseen: Vec<Vec<f32>> =
        serde_json::from_value(expected["transform"].clone()).unwrap();

    let fitted = ParametricUMAP::with_device(config.clone(), Device::flex().autodiff())
        .fit(data.clone(), None);
    assert_vectors_close(&as_f32(fitted.embedding().clone()), &expected_fit);
    assert_vectors_close(&as_f32(fitted.transform(data.clone())), &expected_fit);
    assert_vectors_close(&as_f32(fitted.transform(unseen.clone())), &expected_unseen);

    let loaded = FittedParametricUMAP::load(
        "tests/fixtures/pre_parametric_move.burnpack",
        config,
        3,
        Device::flex(),
    )
    .expect("pre-move Burnpack weights must load with the renamed API");
    assert_vectors_close(&as_f32(loaded.transform(data)), &expected_fit);
    assert_vectors_close(&as_f32(loaded.transform(unseen)), &expected_unseen);
}
