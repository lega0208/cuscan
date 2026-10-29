//! Replay the real producer -> reducer chain without an estimator test seam.
use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use hdbscan::{BackendChoice, DeviceKdTree, HdbscanOps};
use hdbscan_core::{
    dsu::RankDsu,
    kdtree::KdTree,
    mst::{ComponentReductionPlan, MstEdge, merge_component_winners, merge_components},
};

fn upload_int(values: &[i32], device: &Device) -> burn::backend::tensor::IntTensor<Dispatch> {
    Tensor::<1, Int>::from_data(TensorData::new(values.to_vec(), [values.len()]), device)
        .into_dispatch()
}

fn upload_tree(tree: &KdTree, labels: &[i32], device: &Device) -> DeviceKdTree<Dispatch> {
    let nodes = tree.starts.len();
    DeviceKdTree {
        indices: upload_int(&tree.indices, device),
        starts: upload_int(&tree.starts, device),
        ends: upload_int(&tree.ends, device),
        is_leaf: upload_int(&tree.is_leaf, device),
        node_components: upload_int(&tree.node_components(labels).unwrap(), device),
        bounds_min: Tensor::<2>::from_data(
            TensorData::new(tree.bounds_min.clone(), [nodes, tree.n_features]),
            device,
        )
        .into_dispatch(),
        bounds_max: Tensor::<2>::from_data(
            TensorData::new(tree.bounds_max.clone(), [nodes, tree.n_features]),
            device,
        )
        .into_dispatch(),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    }
}

fn edge_bits(edges: &[MstEdge]) -> Vec<(u32, u32, u32)> {
    edges
        .iter()
        .map(|edge| (edge.src, edge.dst, edge.weight_sq.to_bits()))
        .collect()
}

fn replay(device: &Device, kd: bool, reproducible: bool) -> Vec<(u32, u32, u32)> {
    const N: usize = 129;
    const D: usize = 3;
    let values = (0..N)
        .flat_map(|row| {
            // Separated local groups force multiple Boruvka rounds.
            let group = (row / 8) as f32;
            [
                group * 10.0 + (row % 8) as f32 * 0.01,
                ((row * 17) % 31) as f32 * 0.001,
                ((row * 13) % 29) as f32 * 0.001,
            ]
        })
        .collect::<Vec<_>>();
    let tree = KdTree::build(&values, N, D).unwrap();
    let data = Tensor::<2>::from_data(TensorData::new(values, [N, D]), device);
    let core = Tensor::<1>::from_data(TensorData::new(vec![0.0f32; N], [N]), device);
    let mut dsu = RankDsu::new(N);
    let mut edges = Vec::new();
    let mut rounds = 0;
    loop {
        let labels = (0..N).map(|row| dsu.find(row) as i32).collect::<Vec<_>>();
        let plan = ComponentReductionPlan::build(&labels, 512).unwrap();
        let c = plan.offsets.len() - 1;
        if c == 1 {
            break;
        }
        assert!(rounds < N, "round replay failed to converge");
        rounds += 1;
        let (candidate_weights, candidate_indices) = if kd {
            <Dispatch as HdbscanOps>::boruvka_candidates_kd(
                data.clone().into_dispatch(),
                core.clone().into_dispatch(),
                upload_int(&labels, device),
                upload_tree(&tree, &labels, device),
                reproducible,
            )
        } else {
            <Dispatch as HdbscanOps>::boruvka_candidates(
                data.clone().into_dispatch(),
                core.clone().into_dispatch(),
                upload_int(&labels, device),
            )
        };
        // Clone tensors, not recomputed candidates: the oracle sees exactly the
        // same scheduling-dependent shared-bound results as the reducer.
        let oracle_weights = candidate_weights.clone();
        let oracle_indices = candidate_indices.clone();
        let (weights, metadata) = <Dispatch as HdbscanOps>::component_reduce(
            candidate_weights,
            candidate_indices,
            upload_int(&plan.members, device),
            upload_int(&plan.tile_starts, device),
            upload_int(&plan.tile_ends, device),
            upload_int(&plan.partial_offsets, device),
            c as u32,
        );
        let weights = Tensor::<1>::from_dispatch(weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let metadata = Tensor::<2, Int>::from_dispatch(metadata)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        // Test-only full candidate readback occurs after the reduction chain.
        let oracle_weights = Tensor::<1>::from_dispatch(oracle_weights)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        let oracle_indices = Tensor::<1, Int>::from_dispatch(oracle_indices)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        for slot in 0..c {
            let mut winner = None;
            for position in plan.offsets[slot] as usize..plan.offsets[slot + 1] as usize {
                let source = plan.members[position] as usize;
                assert!(!oracle_weights[source].is_nan() && oracle_weights[source] >= 0.0);
                assert!(oracle_indices[source] < N as i32);
                if oracle_indices[source] >= 0
                    && winner.is_none_or(|old: usize| oracle_weights[source] < oracle_weights[old])
                {
                    winner = Some(source);
                }
            }
            let row = &metadata[slot * 5..slot * 5 + 5];
            assert_eq!(&row[2..], &[N as i32, 0, -1]);
            match winner {
                Some(source) => {
                    assert_eq!(&row[..2], &[source as i32, oracle_indices[source]]);
                    assert_eq!(weights[slot].to_bits(), oracle_weights[source].to_bits());
                }
                None => {
                    assert_eq!(&row[..2], &[-1, -1]);
                    assert_eq!(weights[slot].to_bits(), f32::INFINITY.to_bits());
                }
            }
        }
        let mut oracle_dsu = dsu.clone();
        let mut oracle_edges = edges.clone();
        let expected_added = merge_components(
            &mut oracle_dsu,
            &labels,
            &oracle_weights,
            &oracle_indices,
            &mut oracle_edges,
        )
        .unwrap();
        let added =
            merge_component_winners(&mut dsu, &plan, &weights, &metadata, &mut edges).unwrap();
        assert_eq!(added, expected_added);
        assert_eq!(edge_bits(&edges), edge_bits(&oracle_edges));
        for row in 0..N {
            assert_eq!(dsu.find(row), oracle_dsu.find(row));
        }
    }
    assert!(
        rounds >= 2,
        "fixture must cover changed partitions and root reuse"
    );
    assert_eq!(edges.len(), N - 1);
    edge_bits(&edges)
}

fn check(device: &Device) {
    let brute = replay(device, false, true);
    assert_eq!(replay(device, true, true), brute);
    assert_eq!(replay(device, true, false), brute);
    // Brute flag invariance is estimator-level in the external immutable harness;
    // the flagless low-level brute producer is replayed only once here.
}

#[test]
fn real_candidate_component_rounds_cpu() {
    check(&Device::cpu());
}

#[test]
fn real_candidate_component_rounds_cuda() {
    match cuscan_common::selected_device(BackendChoice::Cuda { device_index: 0 }) {
        Ok((device, _)) => check(&device),
        Err(error) => {
            assert_ne!(
                std::env::var("CUSCAN_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable: {error}"
            );
            eprintln!("SKIPPED component round replay CUDA: {error}");
        }
    }
}
