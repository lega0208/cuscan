use burn::{
    backend::Dispatch,
    tensor::{Device, Tensor, TensorData},
};
use cuscan_common::{
    ann_preflight, memory::resolve_memory_budget, neighbors::approximate_neighbors,
};
use umap::backend::StandardUmapOps;

fn points(rows: usize, features: usize, salt: usize) -> Vec<f32> {
    (0..rows)
        .flat_map(|row| {
            (0..features).map(move |column| {
                ((row * (column + 3 + salt) + column * 5 + salt) % 23) as f32 * 0.125
            })
        })
        .collect()
}

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

fn approximate(
    device: &Device,
    values: Vec<f32>,
    n: usize,
    features: usize,
    k: usize,
    seed: u64,
    iterations: u32,
) -> (Vec<i32>, Vec<f32>) {
    let data = Tensor::<2>::from_data(TensorData::new(values, [n, features]), device);
    let budget = resolve_memory_budget(device, None).unwrap();
    let plan = ann_preflight(device, n, features, k, &budget).unwrap();
    let (ids, distances) = approximate_neighbors(data, &plan, seed, iterations, false).unwrap();
    (
        ids.to_data().try_to_vec::<i32>().unwrap(),
        distances.to_data().try_to_vec::<f32>().unwrap(),
    )
}

fn check_rows(ids: &[i32], distances: &[f32], values: &[f32], n: usize, features: usize, k: usize) {
    assert_eq!(ids.len(), n * k);
    assert_eq!(distances.len(), n * k);
    for row in 0..n {
        let mut unique = std::collections::HashSet::new();
        for slot in 0..k {
            let index = row * k + slot;
            let id = ids[index];
            assert!(
                (0..n as i32).contains(&id) && id != row as i32,
                "invalid/self neighbor at row {row}"
            );
            assert!(unique.insert(id), "duplicate neighbor at row {row}");
            let mut squared = 0.0f32;
            for feature in 0..features {
                let delta =
                    values[row * features + feature] - values[id as usize * features + feature];
                squared += delta * delta;
            }
            let expected = squared.sqrt();
            assert!(distances[index].is_finite() && distances[index] >= 0.0);
            assert!(
                (distances[index] - expected).abs() <= 1e-5,
                "incorrect Euclidean distance at row {row}, id {id}"
            );
            if slot > 0 {
                let previous = row * k + slot - 1;
                let previous_id = ids[previous] as usize;
                let mut previous_squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        values[row * features + feature] - values[previous_id * features + feature];
                    previous_squared += delta * delta;
                }
                assert!(
                    previous_squared < squared
                        || (previous_squared == squared && ids[previous] < id),
                    "row is not ordered by (squared distance,id)"
                );
            }
        }
    }
}

fn run_shape_cases(device: &Device) {
    for (n, features, k, iterations) in [
        (7usize, 3usize, 2usize, 3u32),
        (19, 1, 3, 4),
        (43, 3, 8, 4),
        (67, 3, 33, 7),
    ] {
        let values = if n == 67 {
            odd_dimension_points(n)
        } else {
            points(n, features, n + features)
        };
        let (ids, distances) =
            approximate(device, values.clone(), n, features, k, 0x5179, iterations);
        check_rows(&ids, &distances, &values, n, features, k);
    }
}

#[test]
fn cpu_ann_preserves_shape_and_geometric_contracts() {
    run_shape_cases(&Device::cpu());
}

#[test]
fn cuda_ann_preserves_shape_and_geometric_contracts() {
    let Some(device) = umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    run_shape_cases(&device);
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Neighbor {
    id: usize,
    distance: f32,
    is_new: bool,
}

// Independent scalar contract model: sample pools are fixed by the fixture; this
// intentionally does not model randomized pool selection or production scheduling.
fn local_join_pairs(center: usize, n: usize, new: &[usize], old: &[usize]) -> Vec<(usize, usize)> {
    let mut new_pool = Vec::new();
    for &id in new {
        if id < n && id != center && !new_pool.contains(&id) {
            new_pool.push(id);
        }
    }
    let mut old_pool = Vec::new();
    for &id in old {
        if id < n && id != center && !new_pool.contains(&id) && !old_pool.contains(&id) {
            old_pool.push(id);
        }
    }
    let mut pairs = Vec::new();
    for left in 0..new_pool.len() {
        for right in left + 1..new_pool.len() {
            pairs.push(ordered_pair(new_pool[left], new_pool[right]));
        }
    }
    for &left in &new_pool {
        for &right in &old_pool {
            pairs.push(ordered_pair(left, right));
        }
    }
    pairs
}

fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

fn merge_injected_pairs(
    source: &[Vec<Neighbor>],
    sampled_new: &[Vec<usize>],
    injected: &[((usize, usize), f32)],
    width: usize,
) -> (Vec<Vec<Neighbor>>, usize) {
    let mut rows = source.to_vec();
    for (pair, distance) in injected {
        for (row, id) in [(pair.0, pair.1), (pair.1, pair.0)] {
            if let Some(existing) = rows[row].iter_mut().find(|neighbor| neighbor.id == id) {
                existing.distance = existing.distance.min(*distance);
            } else {
                rows[row].push(Neighbor {
                    id,
                    distance: *distance,
                    is_new: true,
                });
            }
            rows[row].sort_by(|a, b| a.distance.total_cmp(&b.distance).then(a.id.cmp(&b.id)));
            rows[row].truncate(width);
        }
    }
    let mut additions = 0;
    for row in 0..rows.len() {
        for neighbor in &mut rows[row] {
            let source_edge = source[row].iter().find(|old| old.id == neighbor.id);
            neighbor.is_new = match source_edge {
                None => {
                    additions += 1;
                    true
                }
                Some(old) => old.is_new && !sampled_new[row].contains(&old.id),
            };
        }
    }
    (rows, additions)
}

#[test]
fn scalar_sampled_join_and_symmetric_merge_match_hand_derived_fixture() {
    // New/new and both cross-pool orientations are represented by distinct centers;
    // duplicates across classes prefer new, old/old is excluded, invalid and self IDs drop.
    assert_eq!(
        local_join_pairs(0, 6, &[1, 2, 2], &[2, 3, 0, 99]),
        vec![(1, 2), (1, 3), (2, 3)]
    );
    assert_eq!(local_join_pairs(4, 6, &[2], &[1]), vec![(1, 2)]);
    assert_eq!(local_join_pairs(4, 6, &[1], &[2]), vec![(1, 2)]);
    assert_eq!(
        local_join_pairs(5, 6, &[], &[1, 2]),
        Vec::<(usize, usize)>::new()
    );

    // Hand calculation per Phase2: pair 0-2 is evaluated at two centers but is
    // merged by unique ID; then pair 0-1 improves both endpoint rows.
    let source = vec![
        vec![Neighbor {
            id: 1,
            distance: 10.0,
            is_new: false,
        }],
        vec![Neighbor {
            id: 0,
            distance: 10.0,
            is_new: false,
        }],
        vec![Neighbor {
            id: 1,
            distance: 9.0,
            is_new: false,
        }],
    ];
    let sampled = vec![vec![], vec![], vec![]];
    let local_a = local_join_pairs(1, 3, &[0, 2], &[]);
    let local_b = local_join_pairs(2, 3, &[0], &[1]);
    let repeated_local = local_join_pairs(3, 4, &[0, 2], &[]);
    assert_eq!(local_a, vec![(0, 2)]);
    assert_eq!(local_b, vec![(0, 1)]);
    assert_eq!(repeated_local.len(), 1);
    assert_eq!(repeated_local[0], (0, 2));
    let evaluations = [
        (local_a[0], 5.0),
        (repeated_local[0], 5.0),
        (local_b[0], 1.0),
    ];
    assert_eq!(
        evaluations
            .iter()
            .filter(|(pair, _)| *pair == (0, 2))
            .count(),
        2
    );
    let (rows, additions) = merge_injected_pairs(&source, &sampled, &evaluations, 1);
    assert_eq!(
        rows,
        vec![
            vec![Neighbor {
                id: 1,
                distance: 1.0,
                is_new: false
            }],
            vec![Neighbor {
                id: 0,
                distance: 1.0,
                is_new: false
            }],
            vec![Neighbor {
                id: 0,
                distance: 5.0,
                is_new: true
            }],
        ]
    );
    assert_eq!(additions, 1);

    // An unsampled source-new edge survives as new, but is not a final addition.
    let source = vec![
        vec![
            Neighbor {
                id: 1,
                distance: 1.0,
                is_new: true,
            },
            Neighbor {
                id: 2,
                distance: 2.0,
                is_new: false,
            },
        ],
        vec![Neighbor {
            id: 0,
            distance: 1.0,
            is_new: false,
        }],
        vec![Neighbor {
            id: 0,
            distance: 2.0,
            is_new: false,
        }],
    ];
    let (rows, additions) = merge_injected_pairs(&source, &[vec![], vec![], vec![]], &[], 2);
    assert!(rows[0][0].is_new);
    assert_eq!(additions, 0);

    let (rows, additions) = merge_injected_pairs(&source, &[vec![1], vec![], vec![]], &[], 2);
    assert!(!rows[0][0].is_new, "a sampled source-new edge becomes old");
    assert_eq!(additions, 0);
}

fn recall_score(ids: &[i32], values: &[f32], n: usize, features: usize, k: usize) -> usize {
    let mut hits = 0;
    for row in 0..n {
        let mut exact = (0..n)
            .filter(|&other| other != row)
            .map(|other| {
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        values[row * features + feature] - values[other * features + feature];
                    squared += delta * delta;
                }
                (squared, other)
            })
            .collect::<Vec<_>>();
        exact.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let truth = exact
            .into_iter()
            .take(k)
            .map(|(_, id)| id as i32)
            .collect::<std::collections::HashSet<_>>();
        hits += ids[row * k..(row + 1) * k]
            .iter()
            .filter(|id| truth.contains(id))
            .count();
    }
    hits
}

fn check_refinement_quality(
    device: &Device,
    values: Vec<f32>,
    n: usize,
    features: usize,
    k: usize,
    seed: u64,
) {
    let (seeded, seeded_distances) = approximate(device, values.clone(), n, features, k, seed, 0);
    check_rows(&seeded, &seeded_distances, &values, n, features, k);
    let seeded_hits = recall_score(&seeded, &values, n, features, k);
    let exact_hits = n * k;

    let (refined, refined_distances) = approximate(device, values.clone(), n, features, k, seed, 8);
    check_rows(&refined, &refined_distances, &values, n, features, k);
    let refined_hits = recall_score(&refined, &values, n, features, k);
    assert!(
        refined_hits >= seeded_hits,
        "refinement must not reduce exact-neighbor recall: {refined_hits} < {seeded_hits}"
    );
    assert!(
        refined_hits * 100 >= exact_hits * 95,
        "refined graph recall below 95%: {refined_hits}/{exact_hits}"
    );
}

fn run_quality_case(device: &Device) {
    let n = 48;
    let values = (0..n)
        .flat_map(|i| {
            let x = i as f32 * 0.15;
            [x, (x * 0.7).sin()]
        })
        .collect::<Vec<_>>();
    check_refinement_quality(device, values, n, 2, 5, 29);

    let n = 257;
    let features = 7;
    let values = (0..n)
        .flat_map(|row| {
            (0..features).map(move |feature| {
                let t = row as f32 * 0.037 + feature as f32 * 0.41;
                (t.sin() * (feature as f32 + 1.0) * 0.7)
                    + (t * 0.37).cos() * (row % 11) as f32 * 0.013
            })
        })
        .collect::<Vec<_>>();
    check_refinement_quality(device, values, n, features, 15, 0x5eed);
}

#[test]
fn cpu_ann_refines_seed_candidates_with_more_neighbor_joins() {
    run_quality_case(&Device::cpu());
}

#[test]
fn cuda_ann_refines_seed_candidates_with_more_neighbor_joins() {
    let Some(device) = umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    run_quality_case(&device);
}

#[test]
fn cpu_ann_resolves_equal_distances_by_neighbor_id() {
    let values = vec![0.0f32, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0];
    let (ids, distances) = approximate(&Device::cpu(), values.clone(), 5, 2, 4, 13, 3);
    check_rows(&ids, &distances, &values, 5, 2, 4);
    assert_eq!(&ids[..4], &[1, 2, 3, 4]);
    assert_eq!(&distances[..4], &[1.0; 4]);
}

#[test]
fn ann_rejects_invalid_tiny_width_and_singleton_input() {
    let device = Device::cpu();
    let budget = resolve_memory_budget(&device, None).unwrap();
    for (n, features, k) in [(2, 1, 0), (1, 1, 1), (2, 1, 2), (2, 0, 1)] {
        assert!(matches!(
            ann_preflight(&device, n, features, k, &budget),
            Err(cuscan_common::AnnError::InvalidInput(_))
        ));
    }
}

#[test]
fn cpu_ann_handles_minimal_two_row_input() {
    let (ids, distances) = approximate(&Device::cpu(), vec![0.0, 3.0], 2, 1, 1, 42, 3);
    check_rows(&ids, &distances, &[0.0, 3.0], 2, 1, 1);
}

fn check_query_sentinels_and_budget(device: &Device) {
    use umap::standard_umap::approximate_query_budget;
    let reference =
        Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 1.0, 2.0, 3.0], [4, 1]), device);
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.5f32], [1, 1]), device);
    for (marker, expected) in [(-1i32, vec![0, 0]), (-2i32, vec![-1, -1])] {
        let graph = Tensor::<2, burn::tensor::Int>::from_data(
            TensorData::new(vec![marker, 1, 2, 3], [4, 1]),
            device,
        );
        let (_, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
            query.clone().into_dispatch(),
            reference.clone().into_dispatch(),
            graph.into_dispatch(),
            1,
            2,
            4,
            17,
            0,
        );
        let stats = Tensor::<2, burn::tensor::Int>::from_dispatch(stats)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        if marker == -1 {
            assert!(stats.iter().all(|&value| value >= 0));
        } else {
            assert_eq!(stats, expected);
        }
    }
    assert!(matches!(
        approximate_query_budget(usize::MAX, usize::MAX, f64::MAX),
        Err(umap::standard_umap::UmapError::Overflow(_))
    ));
}

fn check_finite_input_distance_overflow(device: &Device) {
    let values = vec![f32::MAX, -f32::MAX, 0.0, 1.0];
    let (ids, distances) = approximate(device, values.clone(), 4, 1, 2, 991, 0);
    assert_eq!(ids.len(), 8);
    assert_eq!(distances.len(), 8);
    for (row, neighbors) in ids.as_chunks::<2>().0.iter().enumerate() {
        let mut unique = std::collections::HashSet::new();
        for slot in 0..2 {
            let id = neighbors[slot];
            let distance = distances[row * 2 + slot];
            if id == 4 {
                assert!(
                    distance == f32::MAX.sqrt() || distance.is_infinite(),
                    "disconnected overflow marker must carry sqrt(max)/inf Euclidean distance"
                );
                continue;
            }
            assert!((0..4).contains(&id) && id != row as i32);
            assert!(unique.insert(id), "valid neighbors must remain unique");
            let delta = values[row] - values[id as usize];
            let squared = delta * delta;
            assert!(
                squared.is_finite(),
                "unrepresentable neighbor must use sentinel"
            );
            assert!(distance.is_finite() && (distance - squared.sqrt()).abs() <= 1e-5);
        }
    }
}

#[test]
fn cpu_query_sentinels_and_checked_budget_overflow() {
    check_query_sentinels_and_budget(&Device::cpu());
}

#[test]
fn cpu_ann_handles_finite_coordinates_whose_squared_distance_overflows() {
    check_finite_input_distance_overflow(&Device::cpu());
}

#[test]
fn cuda_ann_runs_on_device_and_query_sentinels() {
    let Some(device) = umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    check_query_sentinels_and_budget(&device);
    check_finite_input_distance_overflow(&device);
}

/// SplitMix64 for the above-ceiling fixtures: fixed and dependency-free.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 bits of precision.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Seeded blobs: 16 centers in [-10, 10)^features, rows assigned round-robin and
/// jittered uniformly in [-0.5, 0.5).
fn blob_values(n: usize, features: usize, seed: u64) -> Vec<f32> {
    let mut random = SplitMix64(seed);
    let centers: Vec<f32> = (0..16 * features)
        .map(|_| random.unit() * 20.0 - 10.0)
        .collect();
    (0..n)
        .flat_map(|row| {
            let center = (row % 16) * features;
            (0..features)
                .map(|feature| centers[center + feature] + random.unit() - 0.5)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Recall over the given rows only: each row's exact k nearest (ties by id) are
/// computed against all n rows, as in `recall_score`, which is infeasible at large n.
fn sampled_recall_score(
    ids: &[i32],
    values: &[f32],
    n: usize,
    features: usize,
    k: usize,
    rows: &[usize],
) -> f64 {
    let mut hits = 0usize;
    for &row in rows {
        let mut exact = (0..n)
            .filter(|&other| other != row)
            .map(|other| {
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        values[row * features + feature] - values[other * features + feature];
                    squared += delta * delta;
                }
                (squared, other)
            })
            .collect::<Vec<_>>();
        exact.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let truth = exact[..k]
            .iter()
            .map(|&(_, id)| id as i32)
            .collect::<std::collections::HashSet<_>>();
        hits += ids[row * k..(row + 1) * k]
            .iter()
            .filter(|id| truth.contains(id))
            .count();
    }
    hits as f64 / (rows.len() * k) as f64
}

/// Large-shape row contract: valid unique non-self ids, finite distances within a
/// relative tolerance of a host recomputation, and non-decreasing returned distances.
/// Rows are ordered by `(squared distance, id)` on the device; the square root can map
/// distinct squared distances to one output value, so the id tie-break is only
/// observable on equal squared distances, and host recomputation (different rounding,
/// e.g. FMA contraction) cannot recover them. Order is therefore judged on the
/// kernel's own output distances.
fn check_large_rows(
    ids: &[i32],
    distances: &[f32],
    values: &[f32],
    n: usize,
    features: usize,
    k: usize,
) {
    assert_eq!(ids.len(), n * k);
    assert_eq!(distances.len(), n * k);
    for row in 0..n {
        let mut unique = std::collections::HashSet::new();
        for slot in 0..k {
            let index = row * k + slot;
            let id = ids[index];
            assert!(
                (0..n as i32).contains(&id) && id != row as i32,
                "invalid/self neighbor at row {row}"
            );
            assert!(unique.insert(id), "duplicate neighbor at row {row}");
            let distance = distances[index];
            assert!(
                distance.is_finite() && distance >= 0.0,
                "row {row} slot {slot}"
            );
            let mut squared = 0.0f32;
            for feature in 0..features {
                let delta =
                    values[row * features + feature] - values[id as usize * features + feature];
                squared += delta * delta;
            }
            let expected = squared.sqrt();
            assert!(
                (distance - expected).abs() <= 1e-5 * expected.max(1.0),
                "row {row} id {id}: distance {distance} vs host {expected}"
            );
            if slot > 0 {
                let previous = distances[index - 1];
                assert!(
                    previous <= distance,
                    "row {row} distances decrease at slot {slot}: {previous} then {distance}"
                );
            }
        }
    }
}

/// 420,000 rows exceed the former i32 join prefix ceiling (399,903 rows at
/// max_candidates=60); the 64-bit pair prefixes must carry the run end to end.
fn run_above_former_join_ceiling(device: &Device) {
    let (n, features, k) = (420_000usize, 8usize, 14usize);
    assert!(n * 5370 > i32::MAX as usize);
    let iterations = (n as f64).log2().round().max(5.0) as u32;
    let values = blob_values(n, features, 42);
    let started = std::time::Instant::now();
    let (ids, distances) = approximate(device, values.clone(), n, features, k, 42, iterations);
    eprintln!(
        "ANN n={n} wall time {:.1}s",
        started.elapsed().as_secs_f64()
    );
    check_large_rows(&ids, &distances, &values, n, features, k);
    let mut random = SplitMix64(7);
    let rows: Vec<usize> = (0..256)
        .map(|_| (random.next_u64() % n as u64) as usize)
        .collect();
    let recall = sampled_recall_score(&ids, &values, n, features, k, &rows);
    eprintln!("ANN n={n} sampled recall {recall:.4}");
    assert!(recall >= 0.95, "sampled recall {recall} below 0.95");
}

#[test]
#[ignore = "420k-row ANN above the former i32 join ceiling; run in --release with an external timeout"]
fn ann_above_former_join_ceiling_cpu() {
    run_above_former_join_ceiling(&Device::cpu());
}

#[test]
#[ignore = "420k-row ANN above the former i32 join ceiling; run in --release with an external timeout"]
fn ann_above_former_join_ceiling_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    run_above_former_join_ceiling(&device);
}
