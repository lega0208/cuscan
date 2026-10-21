use fast_hdbscan_core::dsu::RankDsu;
use fast_hdbscan_core::error::HdbscanError;
use fast_hdbscan_core::mst::{MstEdge, finalize_mst, init_from_knn, merge_components};
use fast_hdbscan_core::oracle::{brute_force_mst, exact_neighbors};

#[test]
fn merge_rejects_nan_and_negative_candidates() {
    for weight in [f32::NAN, -1.0] {
        let mut dsu = RankDsu::new(2);
        let mut edges = Vec::new();
        assert!(matches!(
            merge_components(&mut dsu, &[0, 1], &[weight, 1.0], &[1, 0], &mut edges),
            Err(HdbscanError::Numerical(_))
        ));
        assert!(edges.is_empty());
    }
}

#[test]
fn knn_initialization_skips_padding_and_picks_first_eligible_neighbor() {
    let core = [2.0, 2.0, 3.0];
    let neighbors = [-1, 1, 0, -1, 0, -1];
    let distances = [f32::INFINITY, 1.0, 1.0, f32::INFINITY, 4.0, f32::INFINITY];
    let mut dsu = RankDsu::new(3);
    let edges = init_from_knn(&core, &neighbors, &distances, 2, &mut dsu).unwrap();
    assert_eq!(
        edges,
        vec![
            MstEdge {
                src: 0,
                dst: 1,
                weight_sq: 2.0
            },
            MstEdge {
                src: 2,
                dst: 0,
                weight_sq: 4.0
            },
        ]
    );
}

#[test]
fn merge_preserves_first_component_appearance_and_first_tie() {
    let mut dsu = RankDsu::new(4);
    let mut edges = Vec::new();
    let added = merge_components(
        &mut dsu,
        &[7, 3, 7, 3],
        &[1.0, 2.0, 1.0, 2.0],
        &[1, 2, 3, 0],
        &mut edges,
    )
    .unwrap();
    assert_eq!(added, 2);
    assert_eq!(
        edges,
        vec![
            MstEdge {
                src: 0,
                dst: 1,
                weight_sq: 1.0
            },
            MstEdge {
                src: 1,
                dst: 2,
                weight_sq: 2.0
            },
        ]
    );
}

#[test]
fn merge_reports_zero_progress() {
    let mut dsu = RankDsu::new(2);
    dsu.union(0, 1);
    let error =
        merge_components(&mut dsu, &[0, 0], &[1.0, 1.0], &[1, 0], &mut Vec::new()).unwrap_err();
    assert!(matches!(error, HdbscanError::Numerical(_)));
}

#[test]
fn finalize_uses_f64_sqrt_then_sorted_output() {
    let output = finalize_mst(vec![
        MstEdge {
            src: 2,
            dst: 1,
            weight_sq: 9.0,
        },
        MstEdge {
            src: 0,
            dst: 2,
            weight_sq: 1.0,
        },
    ]);
    assert_eq!(output[0].weight, 1.0);
    assert_eq!(
        (output[1].src, output[1].dst, output[1].weight),
        (2, 1, 3.0)
    );
}

#[test]
fn exact_core_distance_boundaries_and_infinite_mst_are_deterministic() {
    let data = [0.0, 1.0, 3.0];
    assert_eq!(exact_neighbors(&data, 3, 1, 1).unwrap().core_sq, [0.0; 3]);
    assert_eq!(
        exact_neighbors(&data, 3, 1, 2).unwrap().core_sq,
        [9.0, 4.0, 9.0]
    );
    assert!(
        exact_neighbors(&data, 3, 1, 3)
            .unwrap()
            .core_sq
            .iter()
            .all(|x| x.is_infinite())
    );
    assert!(
        exact_neighbors(&data, 3, 1, 5)
            .unwrap()
            .core_sq
            .iter()
            .all(|x| x.is_infinite())
    );

    let first = brute_force_mst(&data, 3, 1, 3).unwrap();
    let second = brute_force_mst(&data, 3, 1, 3).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.len(), 2);
    assert!(first.iter().all(|edge| edge.weight.is_infinite()));
}

#[test]
fn duplicate_points_use_index_ties_and_return_a_tree() {
    let data = [0.0, 0.0, 1.0, 2.0];
    let exact = exact_neighbors(&data, 4, 1, 2).unwrap();
    assert_eq!(&exact.neighbors[0..2], &[1, 2]);
    let edges = brute_force_mst(&data, 4, 1, 2).unwrap();
    assert_eq!(edges.len(), 3);
}
