use hdbscan_core::{
    config::HdbscanConfig,
    dsu::RankDsu,
    error::HdbscanError,
    linkage::{SortedMstEdge, mst_to_linkage, sort_mst},
};

#[test]
fn rank_ties_keep_first_argument_root() {
    let mut dsu = RankDsu::new(8);
    dsu.union(3, 2);
    dsu.union(1, 0);
    dsu.union(1, 3);
    for point in 0..4 {
        assert_eq!(dsu.find(point), 1);
    }
    dsu.union(5, 4);
    dsu.union(7, 6);
    dsu.union(7, 5);
    dsu.union(7, 1);
    for point in 0..8 {
        assert_eq!(dsu.find(point), 7);
    }
}

#[test]
fn invalid_configuration_is_typed() {
    let configs = [
        HdbscanConfig {
            min_cluster_size: 0,
            ..Default::default()
        },
        HdbscanConfig {
            min_samples: Some(0),
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_epsilon: -1.0,
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_epsilon: f64::INFINITY,
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_epsilon: f64::NAN,
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_persistence: -1.0,
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_persistence: f64::INFINITY,
            ..Default::default()
        },
        HdbscanConfig {
            cluster_selection_persistence: f64::NAN,
            ..Default::default()
        },
    ];
    for config in configs {
        assert!(matches!(
            config.validate(),
            Err(HdbscanError::InvalidConfig(_))
        ));
    }
}

#[test]
fn sorted_mst_zero_ties_use_endpoints_and_linkage_orders_roots() {
    let mut edges = [
        SortedMstEdge {
            src: 2,
            dst: 1,
            weight: -0.0,
        },
        SortedMstEdge {
            src: 0,
            dst: 1,
            weight: 0.0,
        },
    ];
    sort_mst(&mut edges);
    assert_eq!(edges[0].src, 0);
    assert_eq!(edges[1].weight.to_bits(), 0.0_f64.to_bits());
    let tree = mst_to_linkage(&edges).unwrap();
    assert_eq!((tree[0].left, tree[0].right, tree[0].size), (1, 0, 2));
    assert_eq!((tree[1].left, tree[1].right, tree[1].size), (3, 2, 3));
}

#[test]
fn malformed_mst_is_rejected() {
    for edges in [
        vec![SortedMstEdge {
            src: 0,
            dst: 2,
            weight: 1.0,
        }],
        vec![SortedMstEdge {
            src: 0,
            dst: 1,
            weight: f64::NAN,
        }],
        vec![
            SortedMstEdge {
                src: 0,
                dst: 1,
                weight: 1.0,
            },
            SortedMstEdge {
                src: 1,
                dst: 0,
                weight: 2.0,
            },
        ],
    ] {
        assert!(matches!(
            mst_to_linkage(&edges),
            Err(HdbscanError::InvalidInput(_))
        ));
    }
}
