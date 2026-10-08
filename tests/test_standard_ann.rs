use burn::{
    backend::Dispatch,
    tensor::{Device, Tensor, TensorData},
};
use fast_umap::backend::StandardUmapOps;
use serde_json::json;
use std::io::Write;
use std::process::{Command, Stdio};

fn frozen_source_digest() -> String {
    let commit = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    for args in [
        vec!["diff", "--quiet", commit, "--", "src"],
        vec!["diff", "--cached", "--quiet", commit, "--", "src"],
    ] {
        let status = Command::new("git")
            .args(&args)
            .status()
            .expect("git must be available to verify frozen product source");
        assert!(
            status.success(),
            "working product src differs from frozen source commit"
        );
    }
    let output = Command::new("git")
        .args(["rev-parse", "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e:src"])
        .output()
        .expect("git must be available to identify frozen product source");
    assert!(
        output.status.success(),
        "could not resolve frozen src tree digest"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn git_blob_digest(bytes: &[u8]) -> String {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "could not digest deterministic ANN inputs"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn run(device: Device, seed: u64, iterations: u32) -> (Vec<i32>, Vec<f32>) {
    let n = 48usize;
    let points = (0..n)
        .flat_map(|i| {
            let x = i as f32 * 0.15;
            [x, (x * 0.7).sin()]
        })
        .collect::<Vec<_>>();
    let data = Tensor::<2>::from_data(TensorData::new(points, [n, 2]), &device);
    let (ids, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
        data.into_dispatch(),
        5,
        seed,
        iterations,
    );
    (
        Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap(),
        Tensor::<2>::from_dispatch(distances)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap(),
    )
}

fn check_rows(ids: &[i32], distances: &[f32], n: usize, k: usize) {
    for row in 0..n {
        let mut unique = std::collections::HashSet::new();
        for slot in 0..k {
            let index = row * k + slot;
            let id = ids[index] as usize;
            assert!(id < n && id != row, "invalid/self neighbor at row {row}");
            assert!(unique.insert(id), "duplicate neighbor at row {row}");
            assert!(distances[index].is_finite() && distances[index] >= 0.0);
            let x = row as f32 * 0.15;
            let y = (x * 0.7).sin();
            let other_x = id as f32 * 0.15;
            let other_y = (other_x * 0.7).sin();
            let expected = ((x - other_x).powi(2) + (y - other_y).powi(2)).sqrt();
            assert!((distances[index] - expected).abs() <= 1e-5);
            if slot > 0 {
                let previous = row * k + slot - 1;
                assert!(
                    distances[previous] < distances[index]
                        || (distances[previous] == distances[index] && ids[previous] < ids[index]),
                    "row is not ordered by (distance,index)"
                );
            }
        }
    }
}

fn recall_score(ids: &[i32]) -> usize {
    let n = 48usize;
    let k = 5usize;
    let mut hits = 0;
    for row in 0..n {
        let mut exact = (0..n)
            .filter(|&other| other != row)
            .map(|other| {
                let x = row as f32 * 0.15;
                let y = (x * 0.7).sin();
                let ox = other as f32 * 0.15;
                let oy = (ox * 0.7).sin();
                let distance = (x - ox).powi(2) + (y - oy).powi(2);
                (distance, other)
            })
            .collect::<Vec<_>>();
        exact.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let truth = exact
            .into_iter()
            .take(k)
            .map(|(_, id)| id as i32)
            .collect::<std::collections::HashSet<_>>();
        hits += ids[row * k..row * k + k]
            .iter()
            .filter(|id| truth.contains(id))
            .count();
    }
    hits
}

#[test]
fn flex_ann_is_deterministic_unique_and_sorted() {
    let (ids, distances) = run(Device::flex(), 7, 6);
    assert_eq!((ids.len(), distances.len()), (48 * 5, 48 * 5));
    check_rows(&ids, &distances, 48, 5);
    assert_eq!((ids, distances), run(Device::flex(), 7, 6));
    // Zero iterations expose the seeded random starts directly.
    assert_ne!(run(Device::flex(), 7, 0).0, run(Device::flex(), 8, 0).0);
}

#[test]
fn flex_ann_resolves_equal_distances_by_neighbor_index() {
    let device = Device::flex();
    let data = Tensor::<2>::from_data(
        TensorData::new(
            vec![0.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0],
            [5, 2],
        ),
        &device,
    );
    let (indices, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 4, 13, 3);
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distances = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(&ids[..4], &[1, 2, 3, 4]);
    assert_eq!(&distances[..4], &[1.0; 4]);
}

#[test]
fn flex_ann_refines_seed_candidates_with_more_neighbor_joins() {
    let (seeded, _) = run(Device::flex(), 29, 0);
    let (refined, _) = run(Device::flex(), 29, 8);
    assert!(
        recall_score(&refined) > recall_score(&seeded),
        "NN-Descent iterations should improve the seed graph"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_runs_on_cuda_and_obeys_required_device_mode() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    let (ids, distances) = run(device, 7, 6);
    check_rows(&ids, &distances, 48, 5);
    assert_eq!(
        (ids, distances),
        run(fast_umap::cuda_backend::device(0).unwrap(), 7, 6)
    );
    assert_ne!(
        run(fast_umap::cuda_backend::device(0).unwrap(), 7, 0).0,
        run(fast_umap::cuda_backend::device(0).unwrap(), 8, 0).0,
    );
    let (seeded, _) = run(fast_umap::cuda_backend::device(0).unwrap(), 29, 0);
    let (refined, _) = run(fast_umap::cuda_backend::device(0).unwrap(), 29, 8);
    assert!(
        recall_score(&refined) > recall_score(&seeded),
        "CUDA NN-Descent iterations should improve the seed graph"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn required_cuda_mode_fails_when_cuda_support_is_not_compiled() {
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
    );
}

#[test]
fn ann_rejects_invalid_tiny_width_instead_of_exact_fallback() {
    let device = Device::flex();
    let data = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0], [2, 1]), &device);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ =
                <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 0, 1, 3);
        }))
        .is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_ann_source_tagged_capture_oracle() {
    const SOURCE: &str = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    let rows = 67usize;
    let features = 3usize;
    let points = odd_dimension_points(rows);
    let input_bits = points
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let input_bytes = points
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let data = Tensor::<2>::from_data(TensorData::new(points.clone(), [rows, features]), &device);
    let (indices, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
        data.clone().into_dispatch(),
        33,
        0x5eed,
        7,
    );
    let ids = Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    let queries = Tensor::<2>::from_data(
        TensorData::new(points[..65 * features].to_vec(), [65, features]),
        &device,
    );
    let fit_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(ids.clone(), [rows, 33]),
        &device,
    );
    let mut query_ids = Vec::new();
    let mut query_distance_bits = Vec::new();
    let mut query_stats = Vec::new();
    for start in [0usize, 64] {
        let end = (start + 64).min(65);
        let block = queries.clone().slice([start..end, 0..features]);
        let (found, found_distances, stats) = <Dispatch as StandardUmapOps>::approximate_query(
            block.into_dispatch(),
            data.clone().into_dispatch(),
            fit_graph.clone().into_dispatch(),
            33,
            48,
            67,
            0x5eed,
            start as u64,
        );
        query_ids.extend(
            Tensor::<2, burn::tensor::Int>::from_dispatch(found)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap(),
        );
        query_distance_bits.extend(
            Tensor::<2>::from_dispatch(found_distances)
                .to_data()
                .try_to_vec::<f32>()
                .unwrap()
                .iter()
                .map(|value| value.to_bits()),
        );
        query_stats.extend(
            Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap(),
        );
    }
    let capture = json!({
        "source_commit": SOURCE,
        "product_src_tree_digest": frozen_source_digest(),
        "input_git_blob_digest": git_blob_digest(&input_bytes),
        "query_input_git_blob_digest": git_blob_digest(
            &input_bytes[..65 * features * std::mem::size_of::<f32>()],
        ),
        "schema": "fast-umap.cuda-ann-capture.v1",
        "operation": "approximate_neighbors+approximate_query",
        "dtype": "f32/i32",
        "fit_shape": [rows, features, 33],
        "query_shape": [65, features, 33],
        "seed": 0x5eedu64,
        "iterations": 7,
        "query_budget": [48, 67],
        "query_offsets": [0, 64],
        "input_f32_bits": input_bits,
        "fit_ids": ids,
        "fit_distance_bits": distance_bits,
        "query_ids": query_ids,
        "query_distance_bits": query_distance_bits,
        "query_peak_candidates_unique_evaluations": query_stats,
        "invalid_sentinel_overflow_and_budget_cases": ann_edge_cases(&device),
    });
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ANN_CAPTURE_PATH") {
        std::fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
    }
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ANN_ORACLE_PATH") {
        let oracle: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(oracle["source_commit"], SOURCE, "oracle source revision");
        assert_eq!(
            capture, oracle,
            "CUDA ANN output differs from frozen-source oracle"
        );
    }
    assert_eq!(ids, run_odd_dimension_case(&device));
}

#[cfg(feature = "cuda")]
fn ann_edge_cases(device: &Device) -> serde_json::Value {
    use fast_umap::standard_umap::approximate_query_budget;
    let reference_values = vec![0.0f32, 1.0, 2.0, 3.0];
    let query_values = vec![0.5f32];
    let reference =
        Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [4, 1]), device);
    let query = Tensor::<2>::from_data(TensorData::new(query_values.clone(), [1, 1]), device);
    let sentinel_graph_values = vec![-1i32, 1, 2, 3];
    let invalid_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(sentinel_graph_values.clone(), [4, 1]),
        device,
    );
    let (_, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
        query.clone().into_dispatch(),
        reference.clone().into_dispatch(),
        invalid_graph.into_dispatch(),
        1,
        2,
        4,
        17,
        0,
    );
    let sentinel_stats = Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    assert!(
        sentinel_stats.iter().all(|&value| value >= 0),
        "-1 is a valid disconnected sentinel"
    );
    let invalid_graph_values = vec![-2i32, 1, 2, 3];
    let invalid_graph = Tensor::<2, burn::tensor::Int>::from_data(
        TensorData::new(invalid_graph_values.clone(), [4, 1]),
        device,
    );
    let (_, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
        query.clone().into_dispatch(),
        reference.clone().into_dispatch(),
        invalid_graph.into_dispatch(),
        1,
        2,
        4,
        17,
        0,
    );
    let invalid_stats = Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    assert_eq!(
        invalid_stats,
        vec![-1, -1],
        "invalid graph IDs must use the existing invalid marker"
    );
    let extreme_values = vec![f32::MAX, -f32::MAX, 0.0, 1.0];
    let extreme = Tensor::<2>::from_data(TensorData::new(extreme_values.clone(), [4, 1]), device);
    let (ids, distances) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(extreme.into_dispatch(), 2, 991, 0);
    let overflow_ids = Tensor::<2, burn::tensor::Int>::from_dispatch(ids)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let overflow_distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect::<Vec<_>>();
    assert!(
        (extreme_values[0] - extreme_values[1]).is_infinite(),
        "fixture must reach finite-input squared-distance overflow"
    );
    let budget_error = approximate_query_budget(usize::MAX, usize::MAX, f64::MAX).unwrap_err();
    let budget_error = match budget_error {
        fast_umap::standard_umap::UmapError::Overflow(subject) => json!({
            "variant": "Overflow", "payload": subject,
        }),
        other => panic!("expected checked budget overflow, got {other:?}"),
    };
    json!({
        "operation": "approximate_query-invalid-markers+approximate_neighbors-overflow+approximate_query_budget-overflow",
        "dtype": "f32/i32/u64",
        "shapes": {"reference": [4, 1], "query": [1, 1], "graph": [4, 1], "overflow_fit": [4, 1]},
        "query_budget": [2, 4],
        "query_offsets": [0],
        "sentinel_graph": sentinel_graph_values,
        "sentinel_stats": sentinel_stats,
        "invalid_graph": [-2, 1, 2, 3],
        "invalid_graph_stats": invalid_stats,
        "overflow_input_bits": extreme_values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "overflow_input_digest": git_blob_digest(&extreme_values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()),
        "overflow_fit_ids": overflow_ids,
        "overflow_fit_distance_bits": overflow_distance_bits,
        "representability_error": budget_error,
    })
}

#[cfg(feature = "cuda")]
fn odd_dimension_points(rows: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            if row == 1 {
                return [0.0, 0.0, 0.0];
            }
            let x = (row % 17) as f32 * 0.25;
            [
                x,
                ((row * 7) % 13) as f32 * 0.125,
                ((row * 11) % 19) as f32 * 0.0625,
            ]
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn run_odd_dimension_case(device: &Device) -> Vec<i32> {
    let data = Tensor::<2>::from_data(TensorData::new(odd_dimension_points(67), [67, 3]), device);
    let (indices, _) =
        <Dispatch as StandardUmapOps>::approximate_neighbors(data.into_dispatch(), 33, 0x5eed, 7);
    Tensor::<2, burn::tensor::Int>::from_dispatch(indices)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap()
}
