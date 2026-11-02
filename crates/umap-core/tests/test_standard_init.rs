use umap_core::{Init, UMAPConfig, UmapError, initialize};

fn spectral_config(dimensions: usize) -> UMAPConfig {
    UMAPConfig {
        init: Init::Spectral,
        n_components: dimensions,
        ..UMAPConfig::default()
    }
}

// Test-only process isolation: an iteration budget is not a wall-clock limit.
// A timed-out child is killed and the numerical regression fails, never passes.
fn isolated_spectral_case(test: &str, case: &str) -> bool {
    use std::io::{Read, Seek};

    const CHILD_CASE: &str = "CUSCAN_SPECTRAL_CHILD_CASE";
    let selected_case = format!("{test}:{case}");
    if let Ok(selected) = std::env::var(CHILD_CASE) {
        return selected != selected_case;
    }
    let mut stderr = tempfile::tempfile().expect("create spectral child stderr file");
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD_CASE, &selected_case)
        .stderr(
            stderr
                .try_clone()
                .expect("clone spectral child stderr file"),
        )
        .spawn()
        .expect("spawn isolated spectral case");
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = child.try_wait().expect("poll spectral child") {
            stderr.rewind().expect("rewind spectral child stderr");
            let mut output = String::new();
            stderr
                .read_to_string(&mut output)
                .expect("read spectral child stderr");
            assert!(
                status.success(),
                "spectral case {selected_case} failed: {status}\n{output}"
            );
            if !output.is_empty() {
                eprint!("{output}");
            }
            assert!(
                !output.to_ascii_lowercase().contains("using seeded uniform"),
                "spectral case {selected_case} fell back to random initialization:\n{output}"
            );
            return true;
        }
        if started.elapsed() >= std::time::Duration::from_secs(300) {
            child.kill().expect("kill timed-out spectral child");
            child.wait().expect("reap timed-out spectral child");
            stderr
                .rewind()
                .expect("rewind timed-out spectral child stderr");
            let mut output = String::new();
            stderr
                .read_to_string(&mut output)
                .expect("read timed-out spectral child stderr");
            panic!("spectral case {selected_case} timed out after 300 seconds\n{output}");
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

#[test]
fn connected_three_path_matches_analytic_nontrivial_spectral_shape() {
    if isolated_spectral_case(
        "connected_three_path_matches_analytic_nontrivial_spectral_shape",
        "main",
    ) {
        return;
    }
    let data = vec![vec![0.0], vec![1.0], vec![2.0]];
    let embedding = initialize(
        &data,
        &[0, 1],
        &[1, 2],
        &[1.0, 1.0],
        &spectral_config(1),
        17,
    )
    .expect("3-path spectral initialization");
    assert_eq!(embedding.len(), 3);
    let [left, middle, right] = embedding.as_slice() else {
        panic!("expected three scalar coordinates");
    };
    assert!((*left + *right - 10.0).abs() < 1.0e-3);
    assert!(((*left - *middle).abs() - 5.0).abs() < 1.0e-3);
    assert!(((*right - *middle).abs() - 5.0).abs() < 1.0e-3);
}

#[test]
fn repeated_connected_cycle_modes_resolve_the_correct_subspace() {
    if isolated_spectral_case(
        "repeated_connected_cycle_modes_resolve_the_correct_subspace",
        "main",
    ) {
        return;
    }
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
    if isolated_spectral_case(
        "production_sized_cycle_converges_to_its_repeated_low_laplacian_eigenspace",
        "main",
    ) {
        return;
    }
    let n = 300;
    let data = (0..n).map(|i| vec![i as f32]).collect::<Vec<_>>();
    let heads = (0..n).collect::<Vec<_>>();
    let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
    let weights = vec![1.0; n];
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(2), 2026)
        .expect("300-vertex cycle should converge through the sparse spectral solver");
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
fn dense_cutoff_and_large_dimension_sparse_boundary_initialize() {
    for (n, dimensions) in [(64, 2), (65, 2), (70, 40), (83, 40)] {
        if isolated_spectral_case(
            "dense_cutoff_and_large_dimension_sparse_boundary_initialize",
            &format!("n={n},dims={dimensions}"),
        ) {
            continue;
        }
        let data = (0..n).map(|i| vec![i as f32]).collect::<Vec<_>>();
        let heads = (0..n).collect::<Vec<_>>();
        let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
        let embedding = initialize(
            &data,
            &heads,
            &tails,
            &vec![1.0; n],
            &spectral_config(dimensions),
            20261014,
        )
        .unwrap_or_else(|error| panic!("n={n}, dimensions={dimensions}: {error}"));
        assert_eq!(embedding.len(), n * dimensions);
        assert!(embedding.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn random_initialization_is_seeded_and_reference_scaled() {
    if isolated_spectral_case(
        "random_initialization_is_seeded_and_reference_scaled",
        "main",
    ) {
        return;
    }
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
    if isolated_spectral_case(
        "disconnected_components_use_signed_axis_centers_and_seeded_local_layouts",
        "main",
    ) {
        return;
    }
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
    assert!(
        first
            .iter()
            .all(|value| value.is_finite() && (0.0..=10.0).contains(value))
    );
    let left_center = (0..3).map(|row| first[row * 2]).sum::<f32>() / 3.0;
    let right_center = (3..6).map(|row| first[row * 2]).sum::<f32>() / 3.0;
    assert!((left_center - right_center).abs() > 1.0);
}

#[test]
fn many_components_use_matrix_free_centroid_affinity_layout() {
    if isolated_spectral_case(
        "many_components_use_matrix_free_centroid_affinity_layout",
        "main",
    ) {
        return;
    }
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
    assert!(
        embedding
            .iter()
            .all(|value| value.is_finite() && (0.0..=10.0).contains(value))
    );
    for dim in 0..2 {
        let values = (0..20).map(|row| embedding[row * 2 + dim]);
        assert!(values.clone().fold(f32::INFINITY, f32::min).abs() < 1.0e-6);
        assert!((values.fold(f32::NEG_INFINITY, f32::max) - 10.0).abs() < 1.0e-6);
    }
}

type SquareGraph = (Vec<Vec<f32>>, Vec<usize>, Vec<usize>, Vec<f32>);

/// `count` unit-square 4-cycles with component offsets `(component * dx, component * dy)`.
fn square_components(count: usize, dx: f32, dy: f32) -> SquareGraph {
    let mut data = Vec::new();
    let mut heads = Vec::new();
    let mut tails = Vec::new();
    let mut weights = Vec::new();
    for component in 0..count {
        let base = component * 4;
        for point in [[0.0_f32, 0.0_f32], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]] {
            data.push(vec![
                point[0] + component as f32 * dx,
                point[1] + component as f32 * dy,
            ]);
        }
        for offset in 0..4 {
            heads.push(base + offset);
            tails.push(base + (offset + 1) % 4);
            weights.push(1.0);
        }
    }
    (data, heads, tails, weights)
}

#[test]
fn separated_centroids_report_actionable_underflow_error() {
    if isolated_spectral_case(
        "separated_centroids_report_actionable_underflow_error",
        "main",
    ) {
        return;
    }
    // Offsets of 40 give pairwise squared centroid distances >= 1600; exp(-1600) == 0.
    let (data, heads, tails, weights) = square_components(5, 40.0, 0.0);
    for dimensions in [1, 2] {
        let result = initialize(
            &data,
            &heads,
            &tails,
            &weights,
            &spectral_config(dimensions),
            91,
        );
        let Err(UmapError::Numerical(message)) = result else {
            panic!("expected numerical error for {dimensions} dimensions, got {result:?}");
        };
        for fragment in [
            "5 connected components".to_owned(),
            format!("2 * n_components = {}", 2 * dimensions),
            "5 of 5".to_owned(),
            "underflows exp".to_owned(),
            "init=\"random\"".to_owned(),
        ] {
            assert!(
                message.contains(&fragment),
                "{fragment:?} missing: {message}"
            );
        }
    }
}

#[test]
fn single_isolated_component_reports_its_own_distance() {
    if isolated_spectral_case("single_isolated_component_reports_its_own_distance", "main") {
        return;
    }
    // Components 0..4 are 1 apart (linked); component 4 is moved 100 away and isolated.
    let (mut data, heads, tails, weights) = square_components(5, 1.0, 0.0);
    for row in &mut data[16..20] {
        row[0] += 100.0;
    }
    let result = initialize(&data, &heads, &tails, &weights, &spectral_config(1), 91);
    let Err(UmapError::Numerical(message)) = result else {
        panic!("expected numerical error, got {result:?}");
    };
    for fragment in ["1 of 5", "first: component 4, 4 rows", "is 1.020e4"] {
        assert!(
            message.contains(fragment),
            "{fragment:?} missing: {message}"
        );
    }
}

#[test]
fn close_centroids_one_dimension_spectral_succeeds() {
    if isolated_spectral_case("close_centroids_one_dimension_spectral_succeeds", "main") {
        return;
    }
    let (data, heads, tails, weights) = square_components(5, 3.1, 0.37);
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(1), 91)
        .expect("one-dimensional centroid layout");
    assert_eq!(embedding.len(), 20);
    assert!(
        embedding
            .iter()
            .all(|value| value.is_finite() && (0.0..=10.0).contains(value))
    );
    assert!(
        embedding
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min)
            .abs()
            < 1.0e-6
    );
    assert!((embedding.iter().copied().fold(f32::NEG_INFINITY, f32::max) - 10.0).abs() < 1.0e-6);
}

#[test]
fn partially_underflowed_centroid_affinity_is_accepted() {
    if isolated_spectral_case(
        "partially_underflowed_centroid_affinity_is_accepted",
        "main",
    ) {
        return;
    }
    // Chain with offsets of 15: consecutive squared distance 225 gives exp(-225) ~ 1.9e-98
    // (nonzero); ends are 60 apart, squared 3600, so exp(-3600) == 0. Every component keeps
    // at least one nonzero link, so the partially underflowed operator must be accepted.
    assert!((-225.0_f64).exp() > 0.0 && (-3600.0_f64).exp() == 0.0);
    let (data, heads, tails, weights) = square_components(5, 15.0, 0.0);
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(1), 91)
        .expect("partial underflow keeps the centroid operator connected");
    assert_eq!(embedding.len(), 20);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn few_separated_components_use_axes_without_underflow() {
    if isolated_spectral_case(
        "few_separated_components_use_axes_without_underflow",
        "main",
    ) {
        return;
    }
    // 4 <= 2 * 2 components use signed axes, so separation alone is not an error.
    let (data, heads, tails, weights) = square_components(4, 40.0, 0.0);
    let embedding = initialize(&data, &heads, &tails, &weights, &spectral_config(2), 91)
        .expect("signed-axis component placement");
    assert_eq!(embedding.len(), 32);
    assert!(embedding.iter().all(|value| value.is_finite()));
}

#[test]
fn one_sample_reference_zero_span_is_explicitly_zero() {
    if isolated_spectral_case("one_sample_reference_zero_span_is_explicitly_zero", "main") {
        return;
    }
    let embedding = initialize(&[vec![3.0, -2.0]], &[], &[], &[], &spectral_config(2), 4).unwrap();
    assert_eq!(embedding, vec![0.0, 0.0]);
}

#[test]
fn malformed_graphs_and_unavailable_spectral_modes_are_errors() {
    if isolated_spectral_case(
        "malformed_graphs_and_unavailable_spectral_modes_are_errors",
        "main",
    ) {
        return;
    }
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
