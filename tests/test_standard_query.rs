use burn::backend::Dispatch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use fast_umap::backend::StandardUmapOps;

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cpu", feature = "cuda"))]
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
    run_query_with_entry(
        query,
        reference,
        graph,
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        query_offset,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cpu", feature = "cuda"))]
fn run_prevalidated_query(
    query: Tensor<2>,
    reference: Tensor<2>,
    graph: Tensor<2, Int>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    run_query_with_entry(
        query,
        reference,
        graph,
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        query_offset,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cpu", feature = "cuda"))]
fn run_query_with_entry(
    query: Tensor<2>,
    reference: Tensor<2>,
    graph: Tensor<2, Int>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
    prevalidated: bool,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let query = query.into_dispatch();
    let reference = reference.into_dispatch();
    let graph = graph.into_dispatch();
    let (indices, distances, stats) = if prevalidated {
        <Dispatch as StandardUmapOps>::approximate_query_prevalidated(
            query,
            reference,
            graph,
            k,
            candidate_capacity,
            visit_capacity,
            seed,
            query_offset,
        )
    } else {
        <Dispatch as StandardUmapOps>::approximate_query(
            query,
            reference,
            graph,
            k,
            candidate_capacity,
            visit_capacity,
            seed,
            query_offset,
        )
    };
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

#[cfg(any(feature = "cpu", feature = "cuda"))]
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

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn reference_floyd_starts(n: usize, capacity: usize, seed: u64, query_offset: u64) -> Vec<i32> {
    const MODULUS: i64 = 2_147_483_647;
    let n_u64 = u64::try_from(n).unwrap();
    let query_key = (query_offset % n_u64) as usize;
    let rng_seed = ((seed % n_u64) as usize + query_key) % n;
    let mut state = (rng_seed as i64 + 1) % MODULUS;
    if state == 0 {
        state = 1;
    }
    let mut starts = Vec::with_capacity(capacity);
    for j in n - capacity..n {
        let high = state / 127_773;
        let low = state - high * 127_773;
        let test = 16_807 * low - 2_836 * high;
        state = if test <= 0 { test + MODULUS } else { test };
        let draw = usize::try_from(state - 1).unwrap() % (j + 1);
        starts.push(if starts.contains(&(draw as i32)) {
            j as i32
        } else {
            draw as i32
        });
    }
    starts
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn self_edge_query_reference(
    n: usize,
    capacity: usize,
    seed: u64,
    query_offset: u64,
    query_rows: usize,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let mut indices = Vec::with_capacity(query_rows * capacity);
    let mut distances = Vec::with_capacity(query_rows * capacity);
    let mut stats = Vec::with_capacity(query_rows * 2);
    for row in 0..query_rows {
        let absolute_offset = query_offset + u64::try_from(row).unwrap();
        let mut ranked = reference_floyd_starts(n, capacity, seed, absolute_offset)
            .into_iter()
            .map(|id| (id, (id as f32 - 0.5).abs()))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| left.1.total_cmp(&right.1).then(left.0.cmp(&right.0)));
        indices.extend(ranked.iter().map(|&(id, _)| id));
        distances.extend(ranked.iter().map(|&(_, distance)| distance));
        stats.extend([capacity as i32, capacity as i32]);
    }
    (indices, distances, stats)
}

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_is_seeded_bounded_and_bipartite() {
    let device = Device::cpu();
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

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_seed_offsets_preserve_block_identity() {
    let device = Device::cpu();
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

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_queue_capacity_changes_work_on_adversarial_graph() {
    let device = Device::cpu();
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

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_orders_equal_distances_by_reference_id() {
    let device = Device::cpu();
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

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_marks_invalid_graph_indices_in_stats() {
    assert_invalid_graph_indices_in_stats(&Device::cpu());
}

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_prevalidated_matches_validating_entry_on_cpu() {
    let device = Device::cpu();
    let (query, reference, graph) = inputs(&device);
    let validating = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        2,
        4,
        10,
        19,
        40,
    );
    let prevalidated = run_prevalidated_query(query, reference, graph, 2, 4, 10, 19, 40);
    assert_eq!(prevalidated, validating);
}

#[cfg(any(feature = "cpu", feature = "cuda"))]
fn assert_invalid_graph_indices_in_stats(device: &Device) {
    let reference =
        Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0], [4, 1]), device);
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device);
    for invalid_index in [i32::MIN, -2, 4, i32::MAX] {
        // Put the invalid edge in an otherwise disconnected, non-start row:
        // validation covers arbitrary graph contents, not just visited edges.
        let mut graph = vec![-1; 4 * 2];
        graph[7] = invalid_index;
        let graph = Tensor::<2, Int>::from_data(TensorData::new(graph, [4, 2]), device);
        let (_, _, stats) = run_query(query.clone(), reference.clone(), graph, 1, 1, 1, 0, 0);
        assert_eq!(stats, [-1, -1], "invalid graph index {invalid_index}");
    }
}

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_uses_seeded_unique_floyd_starts() {
    let device = Device::cpu();
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

#[cfg(feature = "cuda")]
#[test]
fn approximate_query_prevalidated_matches_validating_entry_on_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let (query, reference, graph) = inputs(&device);
    let validating = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        2,
        4,
        10,
        19,
        40,
    );
    let prevalidated = run_prevalidated_query(query, reference, graph, 2, 4, 10, 19, 40);
    assert_eq!(prevalidated.0, validating.0, "indices differ");
    assert_eq!(prevalidated.1, validating.1, "distances differ");
    assert_eq!(prevalidated.2, validating.2, "stats differ");
    assert!(!prevalidated.2.contains(&-1));
}

#[cfg(feature = "cuda")]
#[test]
fn approximate_query_prevalidated_chunks_large_cuda_batch_with_host_oracle() {
    let Some(cuda_device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_prevalidated_chunks_large_batch_with_host_oracle(&cuda_device);
}

#[cfg(feature = "cpu")]
#[test]
fn approximate_query_cpu_chunks_large_batch_with_host_oracle() {
    assert_prevalidated_chunks_large_batch_with_host_oracle(&Device::cpu());
}

fn assert_prevalidated_chunks_large_batch_with_host_oracle(device: &Device) {
    let n = 5_000usize;
    let query_rows = 7_000usize;
    let k = 15;
    let candidates = 15;
    let visits = 5_000;
    let seed = 42u64;
    let query_offset = 17u64;
    let scratch_budget = 256 * 1024 * 1024;
    let scratch_per_row = usize::try_from(candidates).unwrap() * (4 + 4 + 4)
        + usize::try_from(visits).unwrap() * (4 + 4)
        + (4 + 4)
        + 32 * 4;
    let max_chunk_rows = scratch_budget / scratch_per_row;
    assert!(query_rows * scratch_per_row > scratch_budget);
    assert!(max_chunk_rows < query_rows);
    assert_ne!(query_rows % max_chunk_rows, 0);
    let reference_values = (0..n).map(|row| row as f32).collect::<Vec<_>>();
    let graph_values = (0..n).map(|row| row as i32).collect::<Vec<_>>();
    let query_values = vec![0.5f32; query_rows];

    let reference =
        Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [n, 1]), device);
    let query = Tensor::<2>::from_data(
        TensorData::new(query_values.clone(), [query_rows, 1]),
        device,
    );
    let graph = Tensor::<2, Int>::from_data(TensorData::new(graph_values.clone(), [n, 1]), device);
    let actual = run_prevalidated_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        k,
        candidates,
        visits,
        seed,
        query_offset,
    );
    let checked = run_query(
        query,
        reference,
        graph,
        k,
        candidates,
        visits,
        seed,
        query_offset,
    );

    // With self-edges, search cannot visit beyond the initial Floyd starts.
    // Compute those starts, sorted distances, and exact search statistics on
    // the host rather than using another production backend as an oracle.
    let expected =
        self_edge_query_reference(n, candidates as usize, seed, query_offset, query_rows);

    assert_eq!(checked, actual, "checked/prevalidated chunk outputs differ");
    assert_eq!(actual.0, expected.0, "chunked query indices differ");
    assert_eq!(actual.1, expected.1, "chunked query distances differ");
    assert_eq!(actual.2, expected.2, "chunked query stats differ");
    assert!(
        actual
            .2
            .chunks_exact(2)
            .all(|stats| stats[0] == 15 && stats[1] == 15)
    );

    let mut invalid_graph_values = graph_values;
    invalid_graph_values[n - 1] = -2;
    let invalid_graph =
        Tensor::<2, Int>::from_data(TensorData::new(invalid_graph_values, [n, 1]), device);
    let (_, _, invalid_stats) = run_query(
        Tensor::<2>::from_data(TensorData::new(query_values, [query_rows, 1]), device),
        Tensor::<2>::from_data(TensorData::new(reference_values, [n, 1]), device),
        invalid_graph,
        k,
        candidates,
        visits,
        seed,
        query_offset,
    );
    assert_eq!(invalid_stats, vec![-1; query_rows * 2]);
}

#[cfg(feature = "cuda")]
#[test]
fn approximate_query_marks_invalid_graph_indices_in_stats_on_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_invalid_graph_indices_in_stats(&device);
}
