use hdbscan_core::dsu::RankDsu;
use hdbscan_core::error::HdbscanError;
#[cfg(feature = "test-support")]
use hdbscan_core::mst::merge_components;
use hdbscan_core::mst::{
    ComponentReductionPlan, MstEdge, finalize_mst, init_from_knn, merge_component_winners,
};
use hdbscan_core::oracle::{brute_force_mst, exact_neighbors};

#[test]
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
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
#[cfg(feature = "test-support")]
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

#[test]
fn component_plan_is_stable_first_appearance_csr_with_component_tiles() {
    let plan = ComponentReductionPlan::build(&[2, 1, 2, 0, 1, 1], 2).unwrap();
    assert_eq!(plan.members, [0, 2, 1, 4, 5, 3]);
    assert_eq!(plan.offsets, [0, 2, 5, 6]);
    assert_eq!(plan.tile_starts, [0, 2, 4, 5]);
    assert_eq!(plan.tile_ends, [2, 4, 5, 6]);
    assert_eq!(plan.partial_offsets, [0, 1, 3, 4]);
}

#[test]
fn component_plan_checks_labels_and_tile_width_and_accepts_empty_input() {
    for labels in [&[-1][..], &[1][..], &[0, 2][..]] {
        assert!(matches!(
            ComponentReductionPlan::build(labels, 1),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
    assert!(matches!(
        ComponentReductionPlan::build(&[0], 0),
        Err(HdbscanError::InvalidInput(_))
    ));
    let empty = ComponentReductionPlan::build(&[], 4).unwrap();
    assert!(empty.members.is_empty());
    assert_eq!(empty.offsets, [0]);
    assert_eq!(empty.partial_offsets, [0]);
}

#[test]
fn compact_winners_preserve_order_ties_signed_zero_and_weight_bits() {
    let plan = ComponentReductionPlan::build(&[3, 1, 3, 1], 2).unwrap();
    // Component 3's equal signed-zero candidates choose source 0 and retain -0 bits.
    // Component 1's +inf candidate is real and beats its absent row.
    let weights = [-0.0, f32::INFINITY];
    let metadata = [0, 1, 4, 0, -1, 3, 2, 4, 0, -1];
    let mut dsu = RankDsu::new(4);
    let mut edges = Vec::new();
    assert_eq!(
        merge_component_winners(&mut dsu, &plan, &weights, &metadata, &mut edges),
        Ok(2)
    );
    assert_eq!(
        edges[0],
        MstEdge {
            src: 0,
            dst: 1,
            weight_sq: -0.0
        }
    );
    assert_eq!(edges[0].weight_sq.to_bits(), (-0.0_f32).to_bits());
    assert_eq!(
        edges[1],
        MstEdge {
            src: 3,
            dst: 2,
            weight_sq: f32::INFINITY
        }
    );
}

#[test]
fn compact_winners_report_earliest_bad_row_before_any_union() {
    let plan = ComponentReductionPlan::build(&[0, 1, 0, 1], 2).unwrap();
    let mut dsu = RankDsu::new(4);
    let before = (0..4).map(|point| dsu.find(point)).collect::<Vec<_>>();
    let mut edges = Vec::new();
    // The later numerical error beats the earlier-indexed component's later row;
    // row 1 is the globally earliest failure and is an index error.
    let metadata = [0, 1, 4, 0, -1, -1, -1, 1, 2, 4];
    let error = merge_component_winners(
        &mut dsu,
        &plan,
        &[1.0, f32::INFINITY],
        &metadata,
        &mut edges,
    )
    .unwrap_err();
    assert!(matches!(error, HdbscanError::InvalidInput(_)));
    assert!(edges.is_empty());
    assert_eq!(
        (0..4).map(|point| dsu.find(point)).collect::<Vec<_>>(),
        before
    );

    // Numerical wins over index error when both describe the same earliest row.
    let metadata = [0, 1, 4, 0, -1, -1, -1, 1, 1, -1];
    let error = merge_component_winners(
        &mut dsu,
        &plan,
        &[1.0, f32::INFINITY],
        &metadata,
        &mut edges,
    )
    .unwrap_err();
    assert!(matches!(error, HdbscanError::Numerical(_)));
    assert!(edges.is_empty());

    // Error records may retain a valid winner; the error still prevents any union.
    let retained_winner = [0, 1, 4, 0, -1, 3, 2, 1, 2, 4];
    let error = merge_component_winners(&mut dsu, &plan, &[1.0, 2.0], &retained_winner, &mut edges)
        .unwrap_err();
    assert!(matches!(error, HdbscanError::InvalidInput(_)));
    assert!(edges.is_empty());
    assert_eq!(
        (0..4).map(|point| dsu.find(point)).collect::<Vec<_>>(),
        before
    );
}

#[test]
fn compact_winners_reject_malformed_shapes_endpoints_weights_and_plans_atomically() {
    let plan = ComponentReductionPlan::build(&[0, 1], 1).unwrap();
    let mut dsu = RankDsu::new(2);
    let mut edges = Vec::new();
    assert!(matches!(
        merge_component_winners(&mut dsu, &plan, &[1.0], &[0; 5], &mut edges),
        Err(HdbscanError::InvalidInput(_))
    ));
    for (weight, metadata) in [
        (f32::NAN, [0, 1, 2, 0, -1, 1, 0, 2, 0, -1]),
        (-1.0, [0, 1, 2, 0, -1, 1, 0, 2, 0, -1]),
        (1.0, [0, 2, 2, 0, -1, 1, 0, 2, 0, -1]),
        (1.0, [1, 0, 2, 0, -1, 1, 0, 2, 0, -1]),
    ] {
        assert!(
            merge_component_winners(&mut dsu, &plan, &[weight, 1.0], &metadata, &mut edges)
                .is_err()
        );
        assert!(edges.is_empty());
        assert_ne!(dsu.find(0), dsu.find(1));
    }
    let absent = [-1, -1, 2, 0, -1, -1, -1, 2, 0, -1];
    assert!(matches!(
        merge_component_winners(&mut dsu, &plan, &[f32::INFINITY; 2], &absent, &mut edges),
        Err(HdbscanError::Numerical(_))
    ));

    let mut malformed_plan = plan.clone();
    malformed_plan.tile_ends[0] = 2;
    let valid = [0, 1, 2, 0, -1, -1, -1, 2, 0, -1];
    assert!(matches!(
        merge_component_winners(
            &mut dsu,
            &malformed_plan,
            &[1.0, f32::INFINITY],
            &valid,
            &mut edges
        ),
        Err(HdbscanError::InvalidInput(_))
    ));
    assert!(edges.is_empty());

    let mut out_of_bounds_partials = plan.clone();
    out_of_bounds_partials.partial_offsets[1] = i32::MAX;
    assert!(matches!(
        merge_component_winners(
            &mut dsu,
            &out_of_bounds_partials,
            &[1.0, f32::INFINITY],
            &valid,
            &mut edges
        ),
        Err(HdbscanError::InvalidInput(_))
    ));
}

#[test]
fn compact_winners_preserve_no_progress_error() {
    let plan = ComponentReductionPlan::build(&[0, 0], 2).unwrap();
    let mut dsu = RankDsu::new(2);
    dsu.union(0, 1);
    let metadata = [0, 1, 2, 0, -1];
    let error =
        merge_component_winners(&mut dsu, &plan, &[1.0], &metadata, &mut Vec::new()).unwrap_err();
    assert!(matches!(error, HdbscanError::Numerical(_)));
}
