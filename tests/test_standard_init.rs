use fast_umap::standard_umap::{initialize, Init, UMAPConfig, UmapError};

fn spectral_config(dimensions: usize) -> UMAPConfig {
    UMAPConfig {
        init: Init::Spectral,
        n_components: dimensions,
        ..UMAPConfig::default()
    }
}

#[test]
fn connected_three_path_matches_analytic_nontrivial_spectral_shape() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0]];
    let embedding = initialize(
        &data,
        &[0, 1],
        &[1, 2],
        &[1.0, 1.0],
        &spectral_config(1),
        17,
    )
    .expect("bounded 3-path spectral initialization");
    assert_eq!(embedding.len(), 3);
    let [left, middle, right] = embedding.as_slice() else {
        panic!("expected three scalar coordinates");
    };
    assert!((*left + *right - 10.0).abs() < 1.0e-3);
    assert!(((*left - *middle).abs() - 5.0).abs() < 1.0e-3);
    assert!(((*right - *middle).abs() - 5.0).abs() < 1.0e-3);
}

#[test]
fn repeated_connected_cycle_modes_are_resolved_with_orthogonal_deflation() {
    let n = 12;
    let data = (0..n).map(|i| vec![i as f32]).collect::<Vec<_>>();
    let heads = (0..n).collect::<Vec<_>>();
    let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
    let weights = vec![1.0; n];
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(2), 73)
        .expect("two-dimensional repeated cycle eigenspace");
    assert_eq!(embedding.len(), n * 2);
    assert!(embedding.iter().all(|value| value.is_finite()));
    for dim in 0..2 {
        let values = (0..n).map(|row| embedding[row * 2 + dim]);
        assert!(values.clone().fold(f32::INFINITY, f32::min).abs() < 1.0e-6);
        assert!((values.fold(f32::NEG_INFINITY, f32::max) - 10.0).abs() < 1.0e-6);
    }
    let first_mode_variance = (0..n)
        .map(|row| embedding[row * 2])
        .map(|value| (value - 5.0).powi(2))
        .sum::<f32>();
    let second_mode_variance = (0..n)
        .map(|row| embedding[row * 2 + 1])
        .map(|value| (value - 5.0).powi(2))
        .sum::<f32>();
    assert!(first_mode_variance > 1.0 && second_mode_variance > 1.0);
}

#[test]
fn production_sized_cycle_converges_to_its_repeated_low_laplacian_eigenspace() {
    let n = 300;
    let data = (0..n).map(|i| vec![i as f32]).collect::<Vec<_>>();
    let heads = (0..n).collect::<Vec<_>>();
    let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
    let weights = vec![1.0; n];
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(2), 2026)
        .expect("300-vertex cycle should converge without a dense graph solver");
    assert_eq!(embedding.len(), n * 2);

    let means = (0..2)
        .map(|dim| (0..n).map(|row| embedding[row * 2 + dim]).sum::<f32>() / n as f32)
        .collect::<Vec<_>>();
    let mut centered = [vec![0.0_f64; n], vec![0.0_f64; n]];
    for row in 0..n {
        for dim in 0..2 {
            centered[dim][row] = f64::from(embedding[row * 2 + dim] - means[dim]);
        }
    }

    // For the unweighted cycle, normalized adjacency has the repeated
    // eigenvalue cos(2π/n), whose eigenspace is span(cos(2πi/n), sin(2πi/n)).
    let expected_eigenvalue = (2.0 * std::f64::consts::PI / n as f64).cos();
    for (dim, mode) in centered.iter().enumerate() {
        let mut residual_squared = 0.0;
        let mut norm_squared = 0.0;
        for row in 0..n {
            let applied = (mode[(row + n - 1) % n] + mode[(row + 1) % n]) / 2.0;
            residual_squared += (applied - expected_eigenvalue * mode[row]).powi(2);
            norm_squared += mode[row].powi(2);
        }
        let relative_residual = (residual_squared / norm_squared).sqrt();
        assert!(
            relative_residual < 1.0e-3,
            "mode {dim} residual {relative_residual}"
        );
    }

    let cross_product = centered[0]
        .iter()
        .zip(&centered[1])
        .map(|(&left, &right)| left * right)
        .sum::<f64>();
    let first_norm = centered[0]
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    let second_norm = centered[1]
        .iter()
        .map(|value| value.powi(2))
        .sum::<f64>()
        .sqrt();
    let normalized_overlap = cross_product.abs() / (first_norm * second_norm);
    assert!(
        normalized_overlap < 1.0e-3,
        "mode overlap {normalized_overlap}"
    );
}

#[test]
fn random_initialization_is_seeded_and_reference_scaled() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let config = UMAPConfig {
        init: Init::Random,
        n_components: 3,
        ..UMAPConfig::default()
    };
    let first = initialize(&data, &[], &[], &[], &config, 123).unwrap();
    let repeated = initialize(&data, &[], &[], &[], &config, 123).unwrap();
    let different = initialize(&data, &[], &[], &[], &config, 124).unwrap();
    assert_eq!(first, repeated);
    assert_ne!(first, different);
    for dim in 0..3 {
        let values = (0..4).map(|row| first[row * 3 + dim]);
        let min = values.clone().fold(f32::INFINITY, f32::min);
        let max = values.fold(f32::NEG_INFINITY, f32::max);
        assert!(min.abs() < 1.0e-6);
        assert!((max - 10.0).abs() < 1.0e-6);
    }
}

#[test]
fn disconnected_components_use_signed_axis_centers_and_seeded_local_layouts() {
    let data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 2.0],
        vec![5.0, 0.0],
        vec![5.0, 1.0],
        vec![5.0, 2.0],
    ];
    let config = spectral_config(2);
    let first = initialize(&data, &[0, 1, 3, 4], &[1, 2, 4, 5], &[1.0; 4], &config, 22).unwrap();
    let repeated = initialize(&data, &[0, 1, 3, 4], &[1, 2, 4, 5], &[1.0; 4], &config, 22).unwrap();
    assert_eq!(first, repeated);
    assert_eq!(first.len(), data.len() * 2);
    assert!(first
        .iter()
        .all(|value| value.is_finite() && (0.0..=10.0).contains(value)));
    let left_center = (0..3).map(|row| first[row * 2]).sum::<f32>() / 3.0;
    let right_center = (3..6).map(|row| first[row * 2]).sum::<f32>() / 3.0;
    assert!((left_center - right_center).abs() > 1.0);
}

#[test]
fn many_components_use_matrix_free_centroid_affinity_layout() {
    let mut data = Vec::new();
    let mut heads = Vec::new();
    let mut tails = Vec::new();
    let mut weights = Vec::new();
    for component in 0..5 {
        let base = component * 4;
        for point in [[0.0_f32, 0.0_f32], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]] {
            data.push(vec![
                point[0] + component as f32 * 3.1,
                point[1] + component as f32 * 0.37,
            ]);
        }
        for offset in 0..4 {
            heads.push(base + offset);
            tails.push(base + (offset + 1) % 4);
            weights.push(1.0);
        }
    }
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(2), 91)
        .expect("matrix-free complete centroid affinity and per-component spectral layouts");
    assert_eq!(embedding.len(), 40);
    assert!(embedding
        .iter()
        .all(|value| value.is_finite() && (0.0..=10.0).contains(value)));
    for dim in 0..2 {
        let values = (0..20).map(|row| embedding[row * 2 + dim]);
        assert!(values.clone().fold(f32::INFINITY, f32::min).abs() < 1.0e-6);
        assert!((values.fold(f32::NEG_INFINITY, f32::max) - 10.0).abs() < 1.0e-6);
    }
}

#[test]
fn one_sample_reference_zero_span_is_explicitly_zero() {
    let embedding = initialize(&[vec![3.0, -2.0]], &[], &[], &[], &spectral_config(2), 4).unwrap();
    assert_eq!(embedding, vec![0.0, 0.0]);
}

#[test]
fn malformed_graphs_and_unavailable_spectral_modes_are_errors() {
    let data = vec![vec![0.0], vec![1.0], vec![2.0]];
    let config = spectral_config(1);
    assert!(matches!(
        initialize(&data, &[0], &[], &[1.0], &config, 1),
        Err(UmapError::InvalidInput(_))
    ));
    assert!(matches!(
        initialize(&data, &[0], &[9], &[1.0], &config, 1),
        Err(UmapError::InvalidInput(_))
    ));
    assert!(matches!(
        initialize(&data, &[0], &[1], &[f32::NAN], &config, 1),
        Err(UmapError::InvalidInput(_))
    ));
    assert!(matches!(
        initialize(&data, &[0, 1], &[1, 2], &[1.0, 1.0], &spectral_config(3), 1),
        Err(UmapError::Numerical(_))
    ));
    assert!(matches!(
        initialize(&data, &[0, 1], &[1, 2], &[1.0, 1.0], &spectral_config(2), 1),
        Err(UmapError::Numerical(_))
    ));
}
