use burn::backend::Dispatch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use fast_umap::backend::StandardUmapOps;

#[allow(clippy::too_many_arguments)]
fn run_query(
    query: Tensor<2>,
    reference: Tensor<2>,
    graph: Tensor<2, Int>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let (indices, distances, stats) = <Dispatch as StandardUmapOps>::approximate_query(
        query.into_dispatch(),
        reference.into_dispatch(),
        graph.into_dispatch(),
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        query_offset,
    );
    (
        Tensor::<2, Int>::from_dispatch(indices)
            .into_data()
            .try_to_vec::<i32>()
            .unwrap(),
        Tensor::<2>::from_dispatch(distances)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap(),
        Tensor::<2, Int>::from_dispatch(stats)
            .into_data()
            .try_to_vec::<i32>()
            .unwrap(),
    )
}

fn inputs(device: &Device) -> (Tensor<2>, Tensor<2>, Tensor<2, Int>) {
    // A path graph makes traversal observable: a query near row 5 can reach
    // its nearest point only by following several retained training edges.
    let reference = Tensor::<2>::from_data(
        TensorData::new((0..12).map(|x| x as f32).collect(), [12, 1]),
        device,
    );
    let query = Tensor::<2>::from_data(
        TensorData::new(vec![5.1f32, 8.1, 2.1, 10.1], [4, 1]),
        device,
    );
    let mut graph = Vec::new();
    for row in 0..12i32 {
        graph.push(if row == 0 { -1 } else { row - 1 });
        graph.push(if row == 11 { -1 } else { row + 1 });
    }
    let graph = Tensor::<2, Int>::from_data(TensorData::new(graph, [12, 2]), device);
    (query, reference, graph)
}

#[test]
fn approximate_query_is_seeded_bounded_and_bipartite() {
    let device = Device::flex();
    let (query, reference, graph) = inputs(&device);
    let first = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        2,
        4,
        10,
        19,
        40,
    );
    let repeated = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        2,
        4,
        10,
        19,
        40,
    );
    assert_eq!(first, repeated);
    assert_eq!(first.0.len(), 8);
    assert_eq!(first.1.len(), 8);
    for (ids, distances) in first
        .0
        .as_chunks::<2>()
        .0
        .iter()
        .zip(first.1.as_chunks::<2>().0)
    {
        assert!(distances[0] < distances[1] || (distances[0] == distances[1] && ids[0] < ids[1]));
        assert!(ids.iter().all(|&id| (0..12).contains(&id)));
    }
    for row in first.2.as_chunks::<2>().0 {
        assert!(
            row[0] >= 1 && row[0] <= 4,
            "peak queue must respect C: {row:?}"
        );
        assert!(
            row[1] >= 1 && row[1] <= 10,
            "evaluations must respect V: {row:?}"
        );
    }

    // Equal query/reference numeric indices are valid in bipartite search.
    let coincident = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    let (ids, distances, _) = run_query(coincident, reference, graph, 1, 2, 2, 0, 0);
    assert_eq!(ids, [0]);
    assert_eq!(distances, [0.0]);
}

#[test]
fn approximate_query_seed_offsets_preserve_block_identity() {
    let device = Device::flex();
    let (query, reference, graph) = inputs(&device);
    let whole = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        1,
        3,
        8,
        77,
        20,
    );
    let mut split_ids = Vec::new();
    let mut split_distances = Vec::new();
    let mut split_stats = Vec::new();
    for row in 0..4 {
        let q = query.clone().slice([row..row + 1, 0..1]);
        let part = run_query(
            q,
            reference.clone(),
            graph.clone(),
            1,
            3,
            8,
            77,
            20 + row as u64,
        );
        split_ids.extend(part.0);
        split_distances.extend(part.1);
        split_stats.extend(part.2);
    }
    assert_eq!(whole.0, split_ids);
    assert_eq!(whole.1, split_distances);
    assert_eq!(whole.2, split_stats);
}

#[test]
fn approximate_query_queue_capacity_changes_work_on_adversarial_graph() {
    let device = Device::flex();
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0], [8, 1]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.1f32], [1, 1]), &device);
    // Seed 4 starts at the branching node: capacity one retains the closest
    // dead-end branch, while a wider queue also follows the second branch.
    let graph = Tensor::<2, Int>::from_data(
        TensorData::new(
            vec![5, -1, 6, -1, 7, -1, -1, -1, 0, 1, 4, -1, 4, 2, 4, 3],
            [8, 2],
        ),
        &device,
    );
    let narrow = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        1,
        1,
        8,
        4,
        0,
    );
    let wide = run_query(query, reference, graph, 1, 4, 8, 4, 0);
    assert_ne!(
        narrow.2[1], wide.2[1],
        "queue capacity must affect actual search work"
    );
}

#[test]
fn approximate_query_orders_equal_distances_by_reference_id() {
    let device = Device::flex();
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 1.0, -1.0], [4, 1]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    let graph = Tensor::<2, Int>::from_data(
        TensorData::new(vec![1, 2, 0, 3, 0, 3, 1, 2], [4, 2]),
        &device,
    );
    let (ids, distances, _) = run_query(query, reference, graph, 3, 4, 4, 0, 0);
    assert_eq!(ids, [0, 1, 2]);
    // Rows 0 and 1 coincide; rows 2 and 3 tie at distance 1, ordered by id.
    assert_eq!(distances, [0.0, 0.0, 1.0]);
}

#[test]
fn approximate_query_marks_invalid_graph_indices_in_stats() {
    let device = Device::flex();
    let reference = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0], [2, 1]), &device);
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    let graph = Tensor::<2, Int>::from_data(TensorData::new(vec![2, -1], [2, 1]), &device);
    let (_, _, stats) = run_query(query, reference, graph, 1, 1, 1, 0, 0);
    assert_eq!(stats, [-1, -1]);
}

#[test]
fn approximate_query_uses_seeded_unique_floyd_starts() {
    let device = Device::flex();
    let n = 23usize;
    let reference = Tensor::<2>::from_data(
        TensorData::new((0..n).map(|row| row as f32).collect(), [n, 1]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    let graph = Tensor::<2, Int>::from_data(TensorData::new(vec![-1; n], [n, 1]), &device);
    let sample = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        7,
        7,
        7,
        9021,
        41,
    );
    let repeated = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        7,
        7,
        7,
        9021,
        41,
    );
    let other_seed = run_query(query, reference, graph, 7, 7, 7, 9022, 41);
    assert_eq!(sample, repeated);
    assert_eq!(sample.0.len(), 7);
    assert_eq!(
        sample
            .0
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        7
    );
    assert_ne!(sample.0, other_seed.0);
    assert_eq!(sample.2, [7, 7]);
}

#[cfg(feature = "cuda")]
#[test]
fn approximate_query_runs_on_cuda_when_requested() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 1.0, 0.0, -1.0, 0.0, 2.0, 0.0], [4, 2]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0], [1, 2]), &device);
    let graph = Tensor::<2, Int>::from_data(
        TensorData::new(vec![1, 2, 0, 3, 0, 3, 1, 2], [4, 2]),
        &device,
    );
    let (ids, distances, stats) = run_query(query, reference, graph, 2, 3, 4, 11, 0);
    assert_eq!(ids, [0, 1]);
    assert_eq!(distances, [0.0, 1.0]);
    assert!(stats[0] <= 3 && stats[1] <= 4);

    let reference = Tensor::<2>::from_data(
        TensorData::new((0..9).map(|row| row as f32).collect(), [9, 1]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), &device);
    let graph = Tensor::<2, Int>::from_data(TensorData::new(vec![-1; 9], [9, 1]), &device);
    let (starts, _, stats) = run_query(query, reference, graph, 4, 4, 4, 123, 0);
    assert_eq!(
        starts
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4
    );
    assert_eq!(stats, [4, 4]);
}
