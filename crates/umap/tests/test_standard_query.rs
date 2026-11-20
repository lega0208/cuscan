use burn::backend::Dispatch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use umap::backend::StandardUmapOps;

#[test]
fn query_phase1_byte_capture() {
    let Ok(directory) = std::env::var("CUSCAN_QUERY_PHASE1_CAPTURE_DIR") else {
        return;
    };
    let mode = std::env::var("CUSCAN_QUERY_PHASE1_CAPTURE_MODE")
        .expect("explicit capture or verify mode is required");
    assert!(mode == "capture" || mode == "verify");
    let device = match std::env::var("CUSCAN_QUERY_PHASE1_CAPTURE_DEVICE")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => Device::cpu(),
        "cuda" => umap::cuda_backend::device(0).expect("CUDA capture requires a CUDA device"),
        other => panic!("unsupported capture device: {other}"),
    };
    let n = 32768usize;
    // Scratch per row: 4*12 + n*8 + 8 + 128 = 262328 bytes.
    // The 256 MiB budget therefore fits 1023 rows; 1027 spans two chunks.
    let rows = 1027usize;
    let reference = Tensor::<2>::from_data(
        TensorData::new((0..n).map(|i| i as f32 * 0.125).collect(), [n, 1]),
        &device,
    );
    let query = Tensor::<2>::from_data(
        TensorData::new(
            (0..rows).map(|i| i as f32 * 0.25 + 0.1).collect(),
            [rows, 1],
        ),
        &device,
    );
    let graph =
        Tensor::<2, Int>::from_data(TensorData::new((0..n as i32).collect(), [n, 1]), &device);
    let start = std::time::Instant::now();
    let (indices, distances, _) = run_query(query, reference, graph, 4, 4, n as u32, 42, 17);
    eprintln!("phase1 query rows={rows} elapsed={:?}", start.elapsed());
    let directory = std::path::Path::new(&directory);
    for (name, bytes) in [
        (
            "indices.i32le",
            indices
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        (
            "distances.f32le",
            distances
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
    ] {
        let path = directory.join(name);
        if mode == "capture" {
            std::fs::write(path, bytes).unwrap();
        } else {
            assert_eq!(
                std::fs::read(path).unwrap(),
                bytes,
                "query bytes differ: {name}"
            );
        }
    }
}

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

fn check_approximate_query_overflow_uses_unsquared_sentinel(device: &Device) {
    let seed = 42;
    let offset = 17;
    let mut starts = reference_floyd_starts(4, 4, seed, offset);
    starts.sort_unstable();
    assert_eq!(starts, vec![0, 1, 2, 3]);

    for (query_value, reference_values, expected_indices, expected_distances) in [
        (
            1e20f32,
            vec![0.0f32, 1.0, 2.0, 3.0],
            vec![4, 4, 4, 4],
            vec![f32::MAX.sqrt(); 4],
        ),
        (
            1e19f32,
            vec![-1e19f32, 0.0, 1.0, 2.0],
            vec![1, 2, 3, 4],
            vec![(1e19f32 * 1e19f32).sqrt(); 3]
                .into_iter()
                .chain([f32::MAX.sqrt()])
                .collect(),
        ),
    ] {
        for prevalidated in [false, true] {
            let query = Tensor::<2>::from_data(TensorData::new(vec![query_value], [1, 1]), device);
            let reference =
                Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [4, 1]), device);
            let graph =
                Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 2, 3], [4, 1]), device);
            let (indices, distances, stats) =
                run_query_with_entry(query, reference, graph, 4, 4, 4, seed, offset, prevalidated);
            assert_eq!(indices, expected_indices, "prevalidated={prevalidated}");
            assert!(distances.iter().all(|distance| distance.is_finite()));
            assert_eq!(
                distances
                    .iter()
                    .map(|distance| distance.to_bits())
                    .collect::<Vec<_>>(),
                expected_distances
                    .iter()
                    .map(|distance| distance.to_bits())
                    .collect::<Vec<_>>(),
                "prevalidated={prevalidated}"
            );
            assert_eq!(stats, vec![starts.len() as i32, starts.len() as i32]);
        }
    }
}

#[test]
fn approximate_query_overflow_uses_unsquared_sentinel_cpu() {
    check_approximate_query_overflow_uses_unsquared_sentinel(&Device::cpu());
}

#[test]
fn approximate_query_overflow_uses_unsquared_sentinel_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    check_approximate_query_overflow_uses_unsquared_sentinel(&device);
}

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

// This oracle models the frontier as optional live entries, independently of
// the kernel's expanded flags and scratch buffers. Fixtures use finite 1D data.
#[allow(clippy::too_many_arguments)]
fn query_reference(
    reference: &[f32],
    graph: &[i32],
    width: usize,
    query: f32,
    k: usize,
    capacity: usize,
    visits: usize,
    seed: u64,
    offset: u64,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let n = reference.len();
    let distance = |id: i32| {
        let d = query - reference[id as usize];
        d * d
    };
    let mut seen = reference_floyd_starts(n, capacity, seed, offset)
        .into_iter()
        .map(|id| (id, distance(id)))
        .collect::<Vec<_>>();
    let mut frontier = seen.iter().copied().map(Some).collect::<Vec<_>>();
    let rank = |a: &(i32, f32), b: &(i32, f32)| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0));
    let mut peak = capacity;
    while seen.len() < visits {
        let Some(slot) = frontier
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.map(|v| (i, v)))
            .min_by(|a, b| rank(&a.1, &b.1))
            .map(|(i, _)| i)
        else {
            break;
        };
        let (id, _) = frontier[slot].take().unwrap();
        for &neighbor in &graph[id as usize * width..(id as usize + 1) * width] {
            if seen.len() == visits {
                break;
            }
            if neighbor < 0 || neighbor as usize >= n || seen.iter().any(|x| x.0 == neighbor) {
                continue;
            }
            let candidate = (neighbor, distance(neighbor));
            seen.push(candidate);
            let free = frontier.iter().position(Option::is_none);
            let replacement = free.or_else(|| {
                let (i, worst) = frontier
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i, x.unwrap()))
                    .max_by(|a, b| rank(&a.1, &b.1))
                    .unwrap();
                (rank(&candidate, &worst).is_lt()).then_some(i)
            });
            if let Some(i) = replacement {
                frontier[i] = Some(candidate);
            }
            peak = peak.max(frontier.iter().filter(|x| x.is_some()).count());
        }
    }
    let evaluated = seen.len();
    seen.sort_by(rank);
    (
        seen.iter().take(k).map(|x| x.0).collect(),
        seen.iter().take(k).map(|x| x.1.sqrt()).collect(),
        vec![peak as i32, evaluated as i32],
    )
}

fn self_edge_query_reference(
    n: usize,
    capacity: usize,
    seed: u64,
    query_offset: u64,
    query_rows: usize,
) -> (Vec<i32>, Vec<f32>, Vec<i32>) {
    let reference = (0..n).map(|i| i as f32).collect::<Vec<_>>();
    let graph = (0..n as i32).collect::<Vec<_>>();
    let mut result = (Vec::new(), Vec::new(), Vec::new());
    for row in 0..query_rows {
        let expected = query_reference(
            &reference,
            &graph,
            1,
            0.5,
            capacity,
            capacity,
            n,
            seed,
            query_offset + row as u64,
        );
        result.0.extend(expected.0);
        result.1.extend(expected.1);
        result.2.extend(expected.2);
    }
    result
}

fn assert_retention_cases(device: &Device) {
    // Build the graph around the actual Floyd starts, so the first start is a
    // branching node in physical slot zero and the second is a live candidate.
    for seed in [0u64, 4, 19] {
        let n = 8usize;
        let starts = reference_floyd_starts(n, 2, seed, 0);
        let others = (0..n as i32)
            .filter(|id| !starts.contains(id))
            .collect::<Vec<_>>();
        let (a, b, x, rejected, replacement, target) = (
            starts[0], starts[1], others[0], others[1], others[2], others[3],
        );
        let mut reference = vec![20.0f32; n];
        for (id, value) in [
            (a, 1.0),
            (b, 4.0),
            (x, 2.0),
            (rejected, 8.0),
            (replacement, 3.0),
            (target, 0.5),
        ] {
            reference[id as usize] = value;
        }
        let width = 4;
        let mut graph = vec![-1; n * width];
        graph[a as usize * width..a as usize * width + width].copy_from_slice(&[
            x,
            rejected,
            replacement,
            x,
        ]);
        graph[x as usize * width..x as usize * width + width]
            .copy_from_slice(&[target, a, x, target]);
        graph[replacement as usize * width] = replacement;
        for visits in [4usize, 7, 8] {
            let expected = query_reference(&reference, &graph, width, 0.0, 2, 2, visits, seed, 0);
            let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device);
            let refs = Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 1]), device);
            let edges =
                Tensor::<2, Int>::from_data(TensorData::new(graph.clone(), [n, width]), device);
            let actual = run_query(
                query.clone(),
                refs.clone(),
                edges.clone(),
                2,
                2,
                visits as u32,
                seed,
                0,
            );
            let unchecked =
                run_prevalidated_query(query, refs, edges, 2, 2, visits as u32, seed, 0);
            assert_eq!(actual, expected, "seed {seed}, visits {visits}");
            assert_eq!(unchecked, expected);
            if visits >= 7 {
                assert_eq!(actual.0[0], target);
            }
        }

        // Equal-distance candidates exercise both directions of the ID tie rule.
        for id in [b, x, rejected, replacement] {
            reference[id as usize] = 2.0;
        }
        let expected = query_reference(&reference, &graph, width, 0.0, 2, 2, 8, seed, 0);
        let actual = run_query(
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device),
            Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 1]), device),
            Tensor::<2, Int>::from_data(TensorData::new(graph.clone(), [n, width]), device),
            2,
            2,
            8,
            seed,
            0,
        );
        assert_eq!(actual, expected, "equal-distance seed {seed}");

        // Expand the second physical start first and reuse its nonzero slot.
        reference[a as usize] = 4.0;
        reference[b as usize] = 1.0;
        graph[b as usize * width..b as usize * width + width].copy_from_slice(&[
            x,
            rejected,
            replacement,
            b,
        ]);
        let expected = query_reference(&reference, &graph, width, 0.0, 2, 2, 8, seed, 0);
        let actual = run_query(
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device),
            Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 1]), device),
            Tensor::<2, Int>::from_data(TensorData::new(graph.clone(), [n, width]), device),
            2,
            2,
            8,
            seed,
            0,
        );
        assert_eq!(actual, expected, "nonzero reusable slot seed {seed}");

        // Disconnected starts and absolute offsets must agree between a batch
        // and independently launched rows, including an offset across N.
        let disconnected = vec![-1; n];
        let refs = Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 1]), device);
        let edges =
            Tensor::<2, Int>::from_data(TensorData::new(disconnected.clone(), [n, 1]), device);
        let batch = run_query(
            Tensor::<2>::from_data(TensorData::new(vec![0.0f32; 3], [3, 1]), device),
            refs.clone(),
            edges.clone(),
            2,
            2,
            8,
            seed,
            7,
        );
        for row in 0..3 {
            let expected = query_reference(
                &reference,
                &disconnected,
                1,
                0.0,
                2,
                2,
                8,
                seed,
                7 + row as u64,
            );
            let single = run_query(
                Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device),
                refs.clone(),
                edges.clone(),
                2,
                2,
                8,
                seed,
                7 + row as u64,
            );
            assert_eq!(single, expected);
            assert_eq!(&batch.0[row * 2..row * 2 + 2], single.0);
            assert_eq!(&batch.1[row * 2..row * 2 + 2], single.1);
            assert_eq!(&batch.2[row * 2..row * 2 + 2], single.2);
        }

        // A complete adjacency row exhausts the visit budget mid-row and, at
        // V=N, must return the same ordering as exhaustive one-dimensional kNN.
        let complete = (0..n).flat_map(|_| 0..n as i32).collect::<Vec<_>>();
        for visits in [5usize, n] {
            let expected = query_reference(&reference, &complete, n, 0.0, 2, 2, visits, seed, 3);
            let actual = run_query(
                Tensor::<2>::from_data(TensorData::new(vec![0.0f32], [1, 1]), device),
                Tensor::<2>::from_data(TensorData::new(reference.clone(), [n, 1]), device),
                Tensor::<2, Int>::from_data(TensorData::new(complete.clone(), [n, n]), device),
                2,
                2,
                visits as u32,
                seed,
                3,
            );
            assert_eq!(actual, expected);
            if visits == n {
                let mut exhaustive = (0..n as i32).collect::<Vec<_>>();
                exhaustive.sort_by(|&a, &b| {
                    reference[a as usize]
                        .abs()
                        .total_cmp(&reference[b as usize].abs())
                        .then(a.cmp(&b))
                });
                assert_eq!(actual.0, exhaustive[..2]);
            }
        }
    }
}

#[test]
fn approximate_query_retention_cases_cpu() {
    assert_retention_cases(&Device::cpu());
}

#[test]
fn approximate_query_retention_cases_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_retention_cases(&device);
}

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
    let prevalidated =
        run_prevalidated_query(query, reference.clone(), graph.clone(), 2, 4, 10, 19, 40);
    assert_eq!(prevalidated, first);
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

#[test]
fn approximate_query_marks_invalid_graph_indices_in_stats() {
    assert_invalid_graph_indices_in_stats(&Device::cpu());
}

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

#[test]
fn approximate_query_runs_on_cuda_when_requested() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
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
    let checked = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        2,
        3,
        4,
        11,
        0,
    );
    let prevalidated = run_prevalidated_query(query, reference, graph, 2, 3, 4, 11, 0);
    assert_eq!(prevalidated.0, checked.0, "indices differ");
    assert_eq!(prevalidated.1, checked.1, "distances differ");
    assert_eq!(prevalidated.2, checked.2, "stats differ");
    assert!(!prevalidated.2.contains(&-1));
    let (ids, distances, stats) = checked;
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

#[test]
fn approximate_query_cuda_packed_tail_byte_and_stats_parity() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let n = 37usize;
    let reference_values = (0..n).map(|row| row as f32).collect::<Vec<_>>();
    let graph_values = (0..n).map(|row| ((row + 1) % n) as i32).collect::<Vec<_>>();
    let reference =
        Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [n, 1]), &device);
    let graph = Tensor::<2, Int>::from_data(TensorData::new(graph_values.clone(), [n, 1]), &device);
    for rows in [31, 32, 33, 63, 64, 65, 127, 128, 129] {
        let query_values = (0..rows)
            .map(|row| (row % n) as f32 + 0.5)
            .collect::<Vec<_>>();
        let query =
            Tensor::<2>::from_data(TensorData::new(query_values.clone(), [rows, 1]), &device);
        let mut expected = (Vec::new(), Vec::new(), Vec::new());
        for (row, &value) in query_values.iter().enumerate() {
            let result = query_reference(
                &reference_values,
                &graph_values,
                1,
                value,
                4,
                4,
                20,
                42,
                17 + row as u64,
            );
            expected.0.extend(result.0);
            expected.1.extend(result.1);
            expected.2.extend(result.2);
        }
        assert!(
            expected
                .2
                .as_chunks::<2>()
                .0
                .iter()
                .all(|stats| stats[1] == 20)
        );
        for prevalidated in [false, true, false] {
            let actual = run_query_with_entry(
                query.clone(),
                reference.clone(),
                graph.clone(),
                4,
                4,
                20,
                42,
                17,
                prevalidated,
            );
            assert_eq!(actual.0, expected.0, "indices at {rows} rows");
            assert_eq!(
                actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "distance bits at {rows} rows"
            );
            assert_eq!(actual.2, expected.2, "stats at {rows} rows");
        }
    }
}

#[test]
fn approximate_query_checked_chunks_large_cuda_batch_with_host_oracle() {
    let Some(cuda_device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_checked_chunks_large_batch_with_host_oracle(&cuda_device);
}

#[test]
fn approximate_query_cpu_chunks_large_batch_with_host_oracle() {
    assert_checked_chunks_large_batch_with_host_oracle(&Device::cpu());
}

#[test]
fn approximate_query_cpu_handles_single_row_chunk() {
    let device = Device::cpu();
    let n = 8usize;
    let query_rows = 1usize;
    let seed = 42u64;
    let query_offset = 17u64;
    let actual = run_prevalidated_query(
        Tensor::<2>::from_data(TensorData::new(vec![0.5f32], [query_rows, 1]), &device),
        Tensor::<2>::from_data(
            TensorData::new((0..n).map(|row| row as f32).collect(), [n, 1]),
            &device,
        ),
        Tensor::<2, Int>::from_data(TensorData::new((0..n as i32).collect(), [n, 1]), &device),
        4,
        4,
        n as u32,
        seed,
        query_offset,
    );
    let expected = self_edge_query_reference(n, 4, seed, query_offset, query_rows);
    assert_eq!(actual, expected);
}

fn assert_checked_chunks_large_batch_with_host_oracle(device: &Device) {
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
    // Each even vertex discovers its odd partner; odd vertices terminate.
    // This exercises expansion without making the large-budget oracle quadratic in n.
    let graph_values = (0..n).map(|row| (row | 1) as i32).collect::<Vec<_>>();
    let query_values = (0..query_rows)
        .map(|row| (row % n) as f32 + 0.5)
        .collect::<Vec<_>>();

    let reference =
        Tensor::<2>::from_data(TensorData::new(reference_values.clone(), [n, 1]), device);
    let query = Tensor::<2>::from_data(
        TensorData::new(query_values.clone(), [query_rows, 1]),
        device,
    );
    let graph = Tensor::<2, Int>::from_data(TensorData::new(graph_values.clone(), [n, 1]), device);
    let actual = run_query(
        query.clone(),
        reference.clone(),
        graph.clone(),
        k,
        candidates,
        visits,
        seed,
        query_offset,
    );
    let repeated = run_query(
        query,
        reference,
        graph,
        k,
        candidates,
        visits,
        seed,
        query_offset,
    );

    // Independently model each row, including discoveries in the second chunk.
    let mut expected = (Vec::new(), Vec::new(), Vec::new());
    for (row, &value) in query_values.iter().enumerate() {
        let result = query_reference(
            &reference_values,
            &graph_values,
            1,
            value,
            k as usize,
            candidates as usize,
            visits as usize,
            seed,
            query_offset + row as u64,
        );
        expected.0.extend(result.0);
        expected.1.extend(result.1);
        expected.2.extend(result.2);
    }

    assert_eq!(actual.0, expected.0, "chunked query indices differ");
    assert_eq!(
        actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "chunked query distance bits differ"
    );
    assert_eq!(actual.2, expected.2, "chunked query stats differ");
    assert_eq!(repeated.0, actual.0, "repeated chunked indices differ");
    assert_eq!(
        repeated.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        "repeated chunked distance bits differ"
    );
    assert_eq!(repeated.2, actual.2, "repeated chunked stats differ");
    for chunk in expected.2.as_chunks::<2>().0.chunks(max_chunk_rows) {
        assert!(
            chunk
                .iter()
                .any(|stats| stats[1] as usize > candidates as usize)
        );
    }

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

#[test]
fn approximate_query_marks_invalid_graph_indices_in_stats_on_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_invalid_graph_indices_in_stats(&device);
}

/// Transform at 21 components (padded fused G=8 on CUDA, head-cache on CPU): the
/// non-shared path reads the fitted tail at stride `n_components`, separately from fit.
fn assert_twenty_one_component_transform(backend: umap::standard_umap::BackendChoice) {
    use umap::standard_umap::{Init, NeighborSearch, UMAP, UMAPConfig};
    const COMPONENTS: usize = 21;
    let data: Vec<Vec<f32>> = (0..64)
        .map(|row| {
            let value = row as f32;
            let offset = if row < 32 { 0.0 } else { 6.0 };
            vec![
                offset + (value * 0.37).sin(),
                (value * 0.19).cos(),
                value * 0.01,
            ]
        })
        .collect();
    let config = UMAPConfig {
        n_components: COMPONENTS,
        n_neighbors: 10,
        optimizer: umap::Optimizer::Compatibility,
        n_epochs: Some(50),
        init: Init::Random,
        random_state: Some(7),
        transform_seed: 7,
        ..UMAPConfig::default()
    };
    let fitted = UMAP::new(config, backend, NeighborSearch::Exact)
        .fit(&data)
        .unwrap();
    assert_eq!(fitted.embedding().len(), 64);
    assert!(
        fitted
            .embedding()
            .iter()
            .all(|row| { row.len() == COMPONENTS && row.iter().all(|value| value.is_finite()) })
    );
    let query: Vec<Vec<f32>> = (0..9)
        .map(|row| {
            let value = row as f32;
            vec![(value * 0.5).sin() + 0.1, (value * 0.3).cos(), value * 0.02]
        })
        .collect();
    let first = fitted.transform(&query).unwrap();
    assert_eq!(first.len(), query.len(), "{backend:?}");
    assert!(
        first
            .iter()
            .all(|row| row.len() == COMPONENTS && row.iter().all(|value| value.is_finite())),
        "{backend:?}: transform output shape or finiteness"
    );
    let bits = |embedding: &[Vec<f32>]| {
        embedding
            .iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    let repeated = fitted.transform(&query).unwrap();
    assert_eq!(
        bits(&first),
        bits(&repeated),
        "{backend:?}: transform determinism"
    );
}

#[test]
fn transform_twenty_one_components_cpu() {
    assert_twenty_one_component_transform(umap::standard_umap::BackendChoice::Cpu);
}

#[test]
fn transform_twenty_one_components_cuda() {
    if umap::cuda_backend::device(0).is_none() {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    assert_twenty_one_component_transform(umap::standard_umap::BackendChoice::Cuda {
        device_index: 0,
    });
}
