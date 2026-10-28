//! Native fitted `approximate_predict` tests: ownership, routes, batching, caches and
//! strict CPU/CUDA execution. The expected predictions use an independent host
//! brute-force candidate oracle fed through the backend-free core hierarchy math, so these
//! tests isolate KD search/orchestration/remapping from the separately tested formulas.

use std::sync::Arc;

use fast_hdbscan::{
    BackendChoice, ClusterSelectionMethod, FittedHDBSCAN, HDBSCAN, HdbscanConfig, HdbscanError,
    HdbscanSearch, KnnSource, MstGraph, PredictionCoreStatus, PredictionResult,
    fit_knn_graph_with_source,
};
use fast_hdbscan_core::{PredictionColumns, PredictionHierarchy};

fn cuda_backend_or_skip() -> Option<BackendChoice> {
    let backend = BackendChoice::Cuda { device_index: 0 };
    if fast_cubecl_common::selected_device(backend).is_ok() {
        return Some(backend);
    }
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
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

/// Dyadic coordinates keep f32 squared distances exact, so oracle ordering is exact.
fn blobs(per_group: usize, features: usize) -> Vec<Vec<f32>> {
    let centers = [0.0f32, 64.0, 128.0];
    let mut rows = Vec::new();
    for i in 0..per_group {
        for (group, center) in centers.iter().enumerate() {
            let scale = 1.0 + group as f32 * 0.5;
            rows.push(
                (0..features)
                    .map(|f| center + scale * (((i * 7 + f * 3 + group) % 17) as f32) * 0.125)
                    .collect(),
            );
        }
    }
    rows
}

fn queries(features: usize, count: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|i| {
            let center = [0.0f32, 64.0, 128.0, 512.0][i % 4];
            (0..features)
                .map(|f| center + (((i * 5 + f * 11) % 13) as f32) * 0.0625)
                .collect()
        })
        .collect()
}

fn squared(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

/// Brute-force `(distance², compact id)` ordered candidates against compact training rows.
fn brute_candidates(train: &[Vec<f32>], query: &[Vec<f32>], k: usize) -> (Vec<i32>, Vec<f32>) {
    let mut ids = Vec::new();
    let mut distances = Vec::new();
    for q in query {
        let mut all: Vec<(f32, usize)> = train
            .iter()
            .enumerate()
            .map(|(id, row)| (squared(q, row), id))
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for &(d, id) in &all[..k] {
            ids.push(id as i32);
            distances.push(d);
        }
    }
    (ids, distances)
}

/// Independent expected predictions computed from the model's public fitted views.
fn oracle(model: &FittedHDBSCAN, train_raw: &[Vec<f32>], query: &[Vec<f32>]) -> PredictionResult {
    let hierarchy =
        PredictionHierarchy::new(&model.condensed, &model.labels, &model.finite_index).unwrap();
    let m = query.len();
    if hierarchy.no_cluster_tree() {
        return PredictionResult {
            labels: vec![-1; m],
            probabilities: vec![0.0; m],
            connecting_points: vec![-1; m],
            no_cluster_tree: true,
        };
    }
    let train: Vec<Vec<f32>> = model
        .finite_index
        .iter()
        .map(|&raw| train_raw[raw as usize].clone())
        .collect();
    let s = model.config.resolved_min_samples();
    let (_, core_sq) = brute_candidates(&train, &train, s);
    let cores: Vec<f64> = core_sq
        .chunks_exact(s)
        .map(|row| f64::from(row[s - 1]).sqrt())
        .collect();
    let (ids, distances) = brute_candidates(&train, query, 2 * s);
    let mut out = PredictionColumns::default();
    hierarchy
        .predict_squared_batch(&ids, &distances, &cores, s, &mut out)
        .unwrap();
    PredictionResult {
        labels: out.labels,
        probabilities: out.probabilities,
        connecting_points: out.connecting_points,
        no_cluster_tree: false,
    }
}

fn assert_matches(actual: &PredictionResult, expected: &PredictionResult, context: &str) {
    assert_eq!(
        actual.no_cluster_tree, expected.no_cluster_tree,
        "{context}"
    );
    assert_eq!(actual.labels, expected.labels, "{context}: labels");
    assert_eq!(
        actual.connecting_points, expected.connecting_points,
        "{context}: connectors"
    );
    for (a, e) in actual.probabilities.iter().zip(&expected.probabilities) {
        assert!(
            (a - e).abs() <= 1e-6 + 1e-5 * e.abs(),
            "{context}: strength {a} vs {e}"
        );
        assert!((0.0..=1.0).contains(a), "{context}: strength range {a}");
    }
}

fn fit(
    backend: BackendChoice,
    config: HdbscanConfig,
    search: HdbscanSearch,
    data: &[Vec<f32>],
) -> FittedHDBSCAN {
    HDBSCAN::new(config, backend)
        .with_search(search)
        .fit(data)
        .unwrap()
}

fn separated_clusters_and_noise(backend: BackendChoice) {
    let data: Vec<Vec<f32>> = (0..12)
        .map(|i| vec![(i % 6) as f32 * 0.25 + if i < 6 { 0.0 } else { 100.0 }])
        .collect();
    let model = fit(backend, config(3, 2), HdbscanSearch::BruteForce, &data);
    let left = model.labels[0];
    let right = model.labels[6];
    assert!(left >= 0 && right >= 0 && left != right);
    let result = model
        .approximate_predict(&[vec![0.5], vec![100.5], vec![1000.0]])
        .unwrap();
    assert!(!result.no_cluster_tree);
    assert_eq!(result.labels[..2], [left, right]);
    assert_eq!(result.labels[2], -1, "distant query is noise");
    assert_eq!(result.probabilities[2], 0.0);
    assert!(
        (0..12).contains(&result.connecting_points[2]),
        "noise still reports its valid connector"
    );
    assert_matches(
        &result,
        &oracle(&model, &data, &[vec![0.5], vec![100.5], vec![1000.0]]),
        "separated",
    );
}

#[test]
fn separated_clusters_and_noise_cpu() {
    separated_clusters_and_noise(BackendChoice::Cpu);
}

#[test]
fn separated_clusters_and_noise_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        separated_clusters_and_noise(backend);
    }
}

fn every_fit_route(backend: BackendChoice) {
    for features in [2usize, 16, 30] {
        let data = blobs(30, features);
        let query = queries(features, 37);
        let mut models = vec![
            (
                "brute",
                fit(backend, config(5, 3), HdbscanSearch::BruteForce, &data),
            ),
            (
                "kd",
                fit(backend, config(5, 3), HdbscanSearch::KdTree, &data),
            ),
            (
                "knn-exact",
                fit_knn_graph_with_source(&data, config(5, 3), backend, KnnSource::Exact).unwrap(),
            ),
        ];
        if backend == BackendChoice::Cpu || features <= 16 {
            models.push((
                "knn-ann",
                fit_knn_graph_with_source(&data, config(5, 3), backend, KnnSource::Ann).unwrap(),
            ));
        }
        for (route, model) in models {
            let context = format!("{route} features={features} backend={backend:?}");
            let kd_fit = model.search() == HdbscanSearch::KdTree;
            let status = model.prediction_cache_status().unwrap();
            assert_eq!(status.tree_ready, kd_fit, "{context}: retained fit tree");
            assert_eq!(
                status.cores,
                PredictionCoreStatus::Uninitialized,
                "{context}"
            );
            let fit_cores = model.core_distances.clone();
            let result = model.approximate_predict(&query).unwrap();
            assert_matches(&result, &oracle(&model, &data, &query), &context);
            assert_eq!(
                model.core_distances, fit_cores,
                "{context}: fit cores untouched"
            );
            let status = model.prediction_cache_status().unwrap();
            assert!(status.tree_ready && status.tree_bytes > 0, "{context}");
            assert_eq!(
                status.cores,
                PredictionCoreStatus::Ready { rank: 3, len: 90 },
                "{context}"
            );
            // Training-coordinate queries are external queries; self is never excluded.
            let train_result = model.approximate_predict(&data[..10]).unwrap();
            assert_matches(
                &train_result,
                &oracle(&model, &data, &data[..10]),
                &format!("{context} training queries"),
            );
        }
    }
}

#[test]
fn every_fit_route_cpu() {
    every_fit_route(BackendChoice::Cpu);
}

#[test]
fn every_fit_route_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        every_fit_route(backend);
    }
}

fn selection_settings(backend: BackendChoice) {
    let data = blobs(25, 2);
    let query = queries(2, 23);
    let variants = [
        HdbscanConfig {
            cluster_selection_method: ClusterSelectionMethod::Leaf,
            ..config(5, 3)
        },
        HdbscanConfig {
            cluster_selection_epsilon: 8.0,
            ..config(5, 3)
        },
        HdbscanConfig {
            cluster_selection_persistence: 0.05,
            ..config(5, 3)
        },
        HdbscanConfig {
            max_cluster_size: Some(20),
            ..config(5, 3)
        },
        HdbscanConfig {
            allow_single_cluster: true,
            ..config(5, 3)
        },
        HdbscanConfig {
            mst_graph: MstGraph::KnnGraph,
            ..config(5, 3)
        },
    ];
    for variant in variants {
        let model = fit(backend, variant.clone(), HdbscanSearch::Auto, &data);
        let result = model.approximate_predict(&query).unwrap();
        assert_matches(
            &result,
            &oracle(&model, &data, &query),
            &format!("{variant:?}"),
        );
        for label in &result.labels {
            assert!(
                *label == -1 || model.labels.contains(label),
                "fitted label IDs"
            );
        }
    }
}

#[test]
fn selection_settings_use_final_fitted_labels_cpu() {
    selection_settings(BackendChoice::Cpu);
}

#[test]
fn selection_settings_use_final_fitted_labels_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        selection_settings(backend);
    }
}

fn mixed_nonfinite_training(backend: BackendChoice) {
    let mut data = blobs(15, 2);
    data.insert(0, vec![f32::NAN, 1.0]);
    data.insert(10, vec![f32::INFINITY, 0.0]);
    data.insert(25, vec![0.0, f32::NEG_INFINITY]);
    let query = queries(2, 19);
    for search in [HdbscanSearch::BruteForce, HdbscanSearch::KdTree] {
        let model = fit(backend, config(5, 3), search, &data);
        let result = model.approximate_predict(&query).unwrap();
        assert_matches(&result, &oracle(&model, &data, &query), "mixed nonfinite");
        for &connector in &result.connecting_points {
            let raw = usize::try_from(connector).unwrap();
            assert!(
                data[raw].iter().all(|v| v.is_finite()),
                "raw finite connector"
            );
            assert!(model.finite_index.contains(&(raw as u32)));
        }
    }
}

#[test]
fn mixed_nonfinite_training_cpu() {
    mixed_nonfinite_training(BackendChoice::Cpu);
}

#[test]
fn mixed_nonfinite_training_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        mixed_nonfinite_training(backend);
    }
}

#[test]
fn one_finite_row_and_no_cluster_models_take_shortcut() {
    let model = HDBSCAN::new(config(2, 1), BackendChoice::Cpu)
        .fit(&[vec![f32::NAN, 0.0], vec![1.0, 2.0]])
        .unwrap();
    let status = model.prediction_cache_status().unwrap();
    assert_eq!(status.cores, PredictionCoreStatus::NotRequiredNoClusterTree);
    let result = model
        .approximate_predict(&[vec![1.0, 2.0], vec![9.0, 9.0]])
        .unwrap();
    assert_eq!(
        result,
        PredictionResult {
            labels: vec![-1, -1],
            probabilities: vec![0.0, 0.0],
            connecting_points: vec![-1, -1],
            no_cluster_tree: true,
        }
    );
    assert!(!model.prediction_cache_status().unwrap().tree_ready);
    // Query validation precedes the shortcut.
    for bad in [
        vec![],
        vec![vec![1.0]],
        vec![vec![f32::NAN, 0.0]],
        vec![vec![f32::INFINITY, 0.0]],
    ] {
        assert!(matches!(
            model.approximate_predict(&bad),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
    // Large min_samples on tiny data: no cluster edges, still the shortcut (no width error).
    let tiny = HDBSCAN::new(config(5, 50), BackendChoice::Cpu)
        .fit(&[vec![0.0], vec![1.0], vec![2.0]])
        .unwrap();
    let shortcut = tiny.approximate_predict(&[vec![0.5]]).unwrap();
    assert!(shortcut.no_cluster_tree);
}

#[test]
fn cluster_tree_predicts_at_minimal_width() {
    let data: Vec<Vec<f32>> = (0..12)
        .map(|i| vec![(i % 6) as f32 * 0.25 + if i < 6 { 0.0 } else { 100.0 }])
        .collect();
    // Core distance is the s-th non-self neighbour, so each group of six stays intact only when
    // s <= 5. Two such groups form a nonempty cluster tree at n = 2s + 2.
    let model = fit(
        BackendChoice::Cpu,
        config(3, 5),
        HdbscanSearch::BruteForce,
        &data,
    );
    assert_eq!(
        model.prediction_cache_status().unwrap().cores,
        PredictionCoreStatus::Uninitialized
    );
    let result = model.approximate_predict(&[vec![0.5]]).unwrap();
    assert!(!result.no_cluster_tree);
    assert!(model.prediction_cache_status().unwrap().tree_ready);
    // Malformed query rows are rejected on the nonempty tree before any further cold work.
    assert!(matches!(
        model.approximate_predict(&[vec![0.5, 0.5]]),
        Err(HdbscanError::InvalidInput(_))
    ));
    assert!(matches!(
        model.approximate_predict(&[vec![0.5], vec![]]),
        Err(HdbscanError::InvalidInput(_))
    ));
}

#[test]
fn width_boundary_without_cluster_tree_takes_shortcut() {
    // s = 6 needs six non-self neighbours but each group of six has only five: every core crosses
    // the gap, so no cluster tree forms and the width is never checked. This is n == 2s.
    let data: Vec<Vec<f32>> = (0..12)
        .map(|i| vec![(i % 6) as f32 * 0.25 + if i < 6 { 0.0 } else { 100.0 }])
        .collect();
    let at_boundary = fit(
        BackendChoice::Cpu,
        config(3, 6),
        HdbscanSearch::BruteForce,
        &data,
    );
    assert_eq!(
        at_boundary.prediction_cache_status().unwrap().cores,
        PredictionCoreStatus::NotRequiredNoClusterTree
    );
    assert_eq!(
        at_boundary.approximate_predict(&[vec![0.5]]).unwrap(),
        PredictionResult {
            labels: vec![-1],
            probabilities: vec![0.0],
            connecting_points: vec![-1],
            no_cluster_tree: true,
        }
    );
    assert!(!at_boundary.prediction_cache_status().unwrap().tree_ready);
    // n < 2*s: a nonempty cluster tree needs n >= 2s + 2 here, so s = 7 is the shortcut and the
    // width is never checked. Width rejection for a nonempty tree is covered in src unit tests.
    let over = fit(
        BackendChoice::Cpu,
        HdbscanConfig {
            allow_single_cluster: true,
            ..config(3, 7)
        },
        HdbscanSearch::BruteForce,
        &data,
    );
    assert_eq!(
        over.prediction_cache_status().unwrap().cores,
        PredictionCoreStatus::NotRequiredNoClusterTree
    );
    let result = over.approximate_predict(&[vec![0.5]]).unwrap();
    assert!(result.no_cluster_tree);
    assert!(!over.prediction_cache_status().unwrap().tree_ready);
}

fn batch_invariance(backend: BackendChoice) {
    let data = blobs(40, 3);
    let model = fit(backend, config(5, 4), HdbscanSearch::KdTree, &data);
    // Above the internal batch cap with a tail.
    let query = queries(3, 1024 + 1024 + 5);
    let all = model.approximate_predict(&query).unwrap();
    assert_matches(&all, &oracle(&model, &data, &query), "batched");
    for (start, chunk) in [(0usize, 1usize), (1023, 3), (2047, 1), (5, 1029)] {
        let part = model
            .approximate_predict(&query[start..start + chunk])
            .unwrap();
        assert_eq!(part.labels, all.labels[start..start + chunk]);
        assert_eq!(
            part.connecting_points,
            all.connecting_points[start..start + chunk]
        );
        assert_eq!(part.probabilities, all.probabilities[start..start + chunk]);
    }
}

#[test]
fn batch_partitioning_is_invariant_cpu() {
    batch_invariance(BackendChoice::Cpu);
}

#[test]
fn batch_partitioning_is_invariant_cuda() {
    if let Some(backend) = cuda_backend_or_skip() {
        batch_invariance(backend);
    }
}

#[test]
fn cpu_and_cuda_agree() {
    let Some(cuda) = cuda_backend_or_skip() else {
        return;
    };
    let data = blobs(30, 5);
    let query = queries(5, 41);
    let cpu = fit(
        BackendChoice::Cpu,
        config(5, 3),
        HdbscanSearch::KdTree,
        &data,
    );
    let gpu = fit(cuda, config(5, 3), HdbscanSearch::KdTree, &data);
    assert_eq!(cpu.labels, gpu.labels);
    let a = cpu.approximate_predict(&query).unwrap();
    let b = gpu.approximate_predict(&query).unwrap();
    assert_matches(&b, &a, "cpu vs cuda");
}

#[test]
fn invalid_queries_are_rejected() {
    let data = blobs(20, 2);
    let model = fit(BackendChoice::Cpu, config(5, 3), HdbscanSearch::Auto, &data);
    for bad in [
        vec![],
        vec![vec![]],
        vec![vec![1.0]],
        vec![vec![1.0, 2.0, 3.0]],
        vec![vec![1.0, 2.0], vec![1.0]],
        vec![vec![f32::NAN, 2.0]],
        vec![vec![1.0, f32::NEG_INFINITY]],
    ] {
        assert!(
            matches!(
                model.approximate_predict(&bad),
                Err(HdbscanError::InvalidInput(_))
            ),
            "{bad:?}"
        );
    }
    let status = model.prediction_cache_status().unwrap();
    assert_eq!(
        status.cores,
        PredictionCoreStatus::Uninitialized,
        "no cold work"
    );
}

#[test]
fn snapshot_ignores_public_mutation_and_source_drop() {
    let mut data = blobs(25, 2);
    let query = queries(2, 17);
    let mut model = fit(
        BackendChoice::Cpu,
        config(5, 3),
        HdbscanSearch::BruteForce,
        &data,
    );
    let expected = oracle(&model, &data, &query);
    // Mutate and drop the source input.
    for row in &mut data {
        row[0] = 1.0e6;
    }
    drop(data);
    // Mutate every public view.
    model.labels.iter_mut().for_each(|l| *l = 99);
    model.probabilities.clear();
    model.condensed.parent.clear();
    model.condensed.child.clear();
    model.linkage.clear();
    model.mst.clear();
    model.neighbors.clear();
    model.core_distances.clear();
    model.finite_index.reverse();
    model.config.min_samples = Some(1);
    model.config.min_cluster_size = 2;
    model.backend = BackendChoice::Cuda { device_index: 7 };
    let result = model.approximate_predict(&query).unwrap();
    assert_matches(&result, &expected, "mutated views");
    // Result mutation does not affect later predictions.
    let mut owned = result.clone();
    owned.labels.fill(5);
    assert_matches(
        &model.approximate_predict(&query).unwrap(),
        &expected,
        "after output mutation",
    );
    assert!(model.dbscan_clustering(1.0).is_ok());
}

#[test]
fn clones_share_cache_and_concurrent_cold_calls_agree() {
    let data = blobs(40, 4);
    let query = queries(4, 33);
    let model = fit(
        BackendChoice::Cpu,
        config(5, 4),
        HdbscanSearch::BruteForce,
        &data,
    );
    let expected = oracle(&model, &data, &query);
    let shared = Arc::new(query);
    let handles: Vec<_> = (0..6)
        .map(|_| {
            let clone = model.clone();
            let query = Arc::clone(&shared);
            std::thread::spawn(move || clone.approximate_predict(&query).unwrap())
        })
        .collect();
    for handle in handles {
        assert_matches(&handle.join().unwrap(), &expected, "concurrent cold");
    }
    let status = model.prediction_cache_status().unwrap();
    assert!(status.tree_ready, "clone initialization is shared");
    assert_eq!(
        status.cores,
        PredictionCoreStatus::Ready { rank: 4, len: 120 }
    );
    // Repeated warm calls reuse cores and stay identical.
    let warm = model.approximate_predict(&shared).unwrap();
    assert_matches(&warm, &expected, "warm");
}

#[test]
fn retained_state_bytes_are_reported_from_actual_allocations() {
    let data = blobs(30, 3);
    let model = fit(
        BackendChoice::Cpu,
        config(5, 3),
        HdbscanSearch::KdTree,
        &data,
    );
    let status = model.prediction_cache_status().unwrap();
    // Canonical state includes at least the owned geometry and the duplicated outputs.
    assert!(status.canonical_bytes >= 90 * 3 * 4 + 90 * 4 * 2);
    // Fit-created tree is retained: permutation plus node arrays and bounds.
    assert!(status.tree_bytes >= 90 * 4);
    // Hierarchy metadata is retained from the fitted condensed tree.
    assert!(status.hierarchy_bytes > 0);
}
