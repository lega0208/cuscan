use fast_hdbscan_core::error::HdbscanError;
use fast_hdbscan_core::knn_kruskal::{KnnGraph, knn_kruskal};
use fast_hdbscan_core::oracle::{brute_force_mst, knn_kruskal_oracle};

fn same_edges(a: &[fast_hdbscan_core::SortedMstEdge], b: &[fast_hdbscan_core::SortedMstEdge]) {
    assert_eq!(a.len(), b.len());
    for (left, right) in a.iter().zip(b) {
        assert_eq!((left.src, left.dst), (right.src, right.dst));
        assert_eq!(left.weight.to_bits(), right.weight.to_bits());
    }
}

fn route(
    data: &[f32],
    rows: usize,
    d: usize,
    samples: usize,
    k: usize,
) -> fast_hdbscan_core::knn_kruskal::KnnMst {
    let reference = knn_kruskal_oracle(data, rows, d, samples, k).unwrap();
    let result = knn_kruskal(
        KnnGraph {
            ids: &reference.ids,
            dist_sq: &reference.dist_sq,
            rows,
            k,
        },
        samples,
    )
    .unwrap();
    assert_eq!(
        result.core.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        reference
            .core
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    same_edges(&result.edges, &reference.edges);
    assert_eq!(result.bridges, reference.bridges);
    result
}

#[test]
fn graph_route_matches_independent_oracle_and_complete_graph_weight_multiset() {
    let data = [0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 7.0, 1.0, 8.0, 4.0];
    for &(samples, k) in &[(0, 2), (1, 2), (2, 2), (4, 2), (5, 2), (8, 2), (2, 4)] {
        let result = route(&data, 5, 2, samples, k);
        if k == 4 {
            let mut expected = brute_force_mst(&data, 5, 2, samples.max(1)).unwrap();
            let mut actual_weights: Vec<_> = result
                .edges
                .iter()
                .map(|edge| edge.weight.to_bits())
                .collect();
            let mut expected_weights: Vec<_> = expected
                .drain(..)
                .map(|edge| edge.weight.to_bits())
                .collect();
            actual_weights.sort_unstable();
            expected_weights.sort_unstable();
            assert_eq!(actual_weights, expected_weights);
        }
    }
}

#[test]
fn equal_weight_order_uses_row_major_edge_ordinal() {
    // 1→2 precedes 1→0 by ordinal; sorting endpoints instead would choose 1→0.
    let ids = [2, -1, 2, 0, 1, 0];
    let dist_sq = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let result = knn_kruskal(
        KnnGraph {
            ids: &ids,
            dist_sq: &dist_sq,
            rows: 3,
            k: 2,
        },
        1,
    )
    .unwrap();
    let endpoints: Vec<_> = result
        .edges
        .iter()
        .map(|edge| (edge.src, edge.dst))
        .collect();
    assert_eq!(endpoints, [(0, 2), (1, 2)]);
}

#[test]
fn zero_width_empty_singleton_padding_infinity_and_overflow_cases() {
    for (rows, data) in [(0, &[][..]), (1, &[3.0][..])] {
        let result = route(data, rows, 1, 2, 0);
        assert!(result.edges.is_empty());
    }
    let empty = route(&[0.0, 1.0, 2.0], 3, 1, 0, 0);
    assert_eq!(empty.edges.len(), 2);
    assert!(empty.edges.iter().all(|edge| edge.weight == f64::INFINITY));
    let infinite_cores = route(&[0.0, 1.0, 2.0], 3, 1, 4, 1);
    assert!(
        infinite_cores
            .core
            .iter()
            .all(|core| *core == f64::INFINITY)
    );
    assert_eq!(infinite_cores.bridges, 2);
    assert_eq!(infinite_cores.edges.len(), 2);
    assert!(
        infinite_cores
            .edges
            .iter()
            .all(|edge| edge.weight == f64::INFINITY)
    );
    let overflow = route(&[f32::MAX, -f32::MAX], 2, 1, 1, 1);
    assert_eq!(overflow.edges.len(), 1);
    assert_eq!(overflow.bridges, 1);
    assert_eq!(overflow.edges[0].weight, f64::INFINITY);

    let ids = [1, -1, 0, -1];
    let dist = [0.0, f32::INFINITY, 0.0, f32::INFINITY];
    let padded = knn_kruskal(
        KnnGraph {
            ids: &ids,
            dist_sq: &dist,
            rows: 2,
            k: 2,
        },
        1,
    )
    .unwrap();
    assert_eq!(padded.edges.len(), 1);
    let missing_rank = knn_kruskal(
        KnnGraph {
            ids: &[1, -1, 0, -1],
            dist_sq: &[1.0, 77.0, 1.0, 88.0],
            rows: 2,
            k: 2,
        },
        2,
    )
    .unwrap();
    assert!(missing_rank.core.iter().all(|core| *core == f64::INFINITY));
}

#[test]
fn duplicate_points_and_disconnected_components_are_deterministic() {
    let result = route(&[0.0, 0.0, 0.0, 10.0, 20.0], 5, 1, 1, 1);
    assert_eq!(result.edges.len(), 4);
    let disjoint_ids = [1, 0, 3, 2, 5, 4];
    let distances = [1.0; 6];
    let result = knn_kruskal(
        KnnGraph {
            ids: &disjoint_ids,
            dist_sq: &distances,
            rows: 6,
            k: 1,
        },
        1,
    )
    .unwrap();
    assert_eq!(result.bridges, 2);
    let bridges: Vec<_> = result
        .edges
        .iter()
        .filter(|edge| edge.weight.is_infinite())
        .map(|edge| (edge.src, edge.dst))
        .collect();
    assert_eq!(bridges, [(0, 2), (2, 4)]);
}

#[test]
fn rejects_all_malformed_graph_boundaries() {
    let malformed: &[(&[i32], &[f32], usize, usize)] = &[
        (&[1], &[], 2, 1),                            // mismatched buffers
        (&[0, 0], &[0.0, 0.0], 2, 1),                 // self
        (&[2, 0], &[0.0, 0.0], 2, 1),                 // out of range
        (&[-2, 0], &[0.0, 0.0], 2, 1),                // invalid negative id
        (&[1, -1, 2], &[0.0, 0.0, 0.0], 3, 3),        // valid after padding
        (&[1, 1, 0, 0], &[0.0, 1.0, 0.0, 1.0], 2, 2), // duplicate
        (&[1, 0], &[f32::NAN, 0.0], 2, 1),
        (&[1, 0], &[-1.0, 0.0], 2, 1),
        (&[1, -1, 0, -1], &[0.0, f32::NAN, 0.0, f32::INFINITY], 2, 2), // NaN in padding
        (&[1, -1, 0, -1], &[0.0, -0.1, 0.0, f32::INFINITY], 2, 2), // negative distance in padding
        (&[1, 2, 0, 2, 0, 1], &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 3, 2), // decreasing
    ];
    for &(ids, dist_sq, rows, k) in malformed {
        match knn_kruskal(
            KnnGraph {
                ids,
                dist_sq,
                rows,
                k,
            },
            1,
        ) {
            Err(HdbscanError::InvalidInput(message)) => {
                let expected = if ids.len() != rows * k || dist_sq.len() != rows * k {
                    "buffers"
                } else if dist_sq.iter().any(|d| d.is_nan() || *d < 0.0) {
                    "distance"
                } else if ids.chunks_exact(k).any(|row| {
                    row.iter()
                        .position(|&id| id == -1)
                        .is_some_and(|p| row[p + 1..].iter().any(|&id| id >= 0))
                }) {
                    "padding"
                } else if ids.iter().any(|&id| id < -1) {
                    "below -1"
                } else if ids.iter().any(|&id| id as usize >= rows) {
                    "out of range"
                } else if ids
                    .chunks_exact(k)
                    .enumerate()
                    .any(|(row, row_ids)| row_ids.contains(&(row as i32)))
                {
                    "itself"
                } else if ids.chunks_exact(k).any(|row| {
                    row.iter()
                        .enumerate()
                        .any(|(index, id)| row[..index].contains(id))
                }) {
                    "duplicate"
                } else {
                    "decrease"
                };
                assert!(
                    message.contains(expected),
                    "expected {expected:?}: {message}"
                );
            }
            other => panic!("expected malformed graph rejection, got {other:?}"),
        }
    }
    let too_many = knn_kruskal(
        KnnGraph {
            ids: &[],
            dist_sq: &[],
            rows: i32::MAX as usize + 1,
            k: 0,
        },
        1,
    );
    assert!(matches!(too_many, Err(HdbscanError::InvalidInput(_))));
    let product = knn_kruskal(
        KnnGraph {
            ids: &[],
            dist_sq: &[],
            rows: 2,
            k: usize::MAX,
        },
        1,
    );
    assert!(matches!(product, Err(HdbscanError::Overflow(_))));
    let bytes = knn_kruskal(
        KnnGraph {
            ids: &[],
            dist_sq: &[],
            rows: 2,
            k: isize::MAX as usize / 2,
        },
        1,
    );
    assert!(matches!(bytes, Err(HdbscanError::Overflow(_))));
}
