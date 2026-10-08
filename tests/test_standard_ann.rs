use burn::{
    backend::Dispatch,
    tensor::{Device, Tensor, TensorData},
};
use fast_umap::backend::StandardUmapOps;

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
