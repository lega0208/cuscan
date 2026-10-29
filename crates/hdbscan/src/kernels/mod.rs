mod boruvka;
mod component_reduce;
mod kdtree;

pub const COMPONENT_REDUCTION_TILE_WIDTH: usize = 512;

pub use boruvka::{BoruvkaCandidateSizing, validate_boruvka_candidate_sizing};
pub(crate) use kdtree::{
    KdCandidateLayout, KdNeighborPlan, kd_neighbor_pack_allocations, kd_neighbor_pack_bytes,
    kd_neighbor_plan, kd_padded_width, kd_query_pack_bytes, kd_query_plan, kd_reference_pack_bytes,
    kd_reference_pack_peak_bytes,
};
#[cfg(test)]
pub(crate) use kdtree::{KdQueryOrder, kd_padded_bytes};

use burn::backend::{
    TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use burn_cubecl::CubeBackend;

/// Private, unaliased bound allocation. Only the backend wrapper resets it.
/// Burn's Dispatch/Fusion extension cannot pass mutable scratch references;
/// the operation owns this allocation instead of exposing mutable host state.
struct KdBoundScratch {
    bounds: IntTensor<CubeBackend>,
}

impl KdBoundScratch {
    fn new(data: &FloatTensor<CubeBackend>, rows: usize, reproducible: bool) -> Self {
        Self {
            bounds: kdtree::allocate_int(data, if reproducible { 1 } else { rows }),
        }
    }
}

impl crate::backend::HdbscanOps for CubeBackend {
    fn boruvka_candidates(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        boruvka::boruvka_candidates(data, core_sq, point_components)
    }

    fn component_reduce(
        candidate_weights: FloatTensor<Self>,
        candidate_indices: IntTensor<Self>,
        members: IntTensor<Self>,
        tile_starts: IntTensor<Self>,
        tile_ends: IntTensor<Self>,
        partial_offsets: IntTensor<Self>,
        component_count: u32,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        component_reduce::component_reduce(
            candidate_weights,
            candidate_indices,
            members,
            tile_starts,
            tile_ends,
            partial_offsets,
            component_count,
        )
    }

    fn exact_neighbors_kd(
        data: FloatTensor<Self>,
        tree: crate::backend::DeviceKdTree<Self>,
        k: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        kdtree::exact_neighbors_kd(data, tree, k)
    }

    fn prepare_kd_reference(
        data: FloatTensor<Self>,
        tree: crate::backend::DeviceKdTree<Self>,
        lanes: u32,
    ) -> crate::backend::DeviceKdReference<Self> {
        kdtree::prepare_kd_reference(data, tree, lanes)
    }

    fn exact_neighbors_kd_query(
        query: FloatTensor<Self>,
        reference: crate::backend::DeviceKdReference<Self>,
        k: u32,
        exclude_self: bool,
        query_offset: u32,
    ) -> (IntTensor<Self>, FloatTensor<Self>) {
        kdtree::exact_neighbors_kd_query(query, reference, k, exclude_self, query_offset)
    }

    fn boruvka_candidates_kd(
        data: FloatTensor<Self>,
        core_sq: FloatTensor<Self>,
        point_components: IntTensor<Self>,
        tree: crate::backend::DeviceKdTree<Self>,
        reproducible: bool,
    ) -> (FloatTensor<Self>, IntTensor<Self>) {
        let rows = data.shape().dims::<2>()[0];
        let mut scratch = KdBoundScratch::new(&data, rows, reproducible);
        kdtree::boruvka_candidates_kd(
            data,
            core_sq,
            point_components,
            tree,
            reproducible,
            &mut scratch,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{DeviceKdTree, HdbscanOps};
    use burn::backend::ops::{FloatTensorOps, IntTensorOps};
    use burn::tensor::TensorData;
    use burn_cubecl::{CubeDevice, ops::into_data_sync};
    use std::collections::BTreeMap;

    fn upload_tree(
        host: &hdbscan_core::KdTree,
        components: &[i32],
        device: &CubeDevice,
    ) -> DeviceKdTree<CubeBackend> {
        let nodes = host.starts.len();
        let int = |values: Vec<i32>| {
            let count = values.len();
            CubeBackend::int_from_data(TensorData::new(values, [count]), device)
        };
        DeviceKdTree {
            indices: int(host.indices.clone()),
            starts: int(host.starts.clone()),
            ends: int(host.ends.clone()),
            is_leaf: int(host.is_leaf.clone()),
            bounds_min: CubeBackend::float_from_data(
                TensorData::new(host.bounds_min.clone(), [nodes, host.n_features]),
                device,
            ),
            bounds_max: CubeBackend::float_from_data(
                TensorData::new(host.bounds_max.clone(), [nodes, host.n_features]),
                device,
            ),
            node_components: int(host.node_components(components).unwrap()),
            n_levels: host.n_levels,
            n_features: host.n_features,
        }
    }

    fn candidates(
        data: &FloatTensor<CubeBackend>,
        host: &hdbscan_core::KdTree,
        components: &[i32],
        reproducible: bool,
        scratch: &mut KdBoundScratch,
    ) -> (Vec<f32>, Vec<i32>) {
        let device = &data.device;
        let n = components.len();
        let (weights, destinations) = kdtree::boruvka_candidates_kd(
            data.clone(),
            CubeBackend::float_from_data(TensorData::new(vec![0.0f32; n], [n]), device),
            CubeBackend::int_from_data(TensorData::new(components.to_vec(), [n]), device),
            upload_tree(host, components, device),
            reproducible,
            scratch,
        );
        (
            into_data_sync(weights).try_to_vec().unwrap(),
            into_data_sync(destinations).try_to_vec().unwrap(),
        )
    }

    // Same first-source strict-less rule as host merging, independently reduced.
    fn winners(
        components: &[i32],
        candidates: &(Vec<f32>, Vec<i32>),
    ) -> BTreeMap<i32, (u32, usize, i32)> {
        let mut winners = BTreeMap::new();
        for (row, &component) in components.iter().enumerate() {
            let weight = candidates.0[row];
            let destination = candidates.1[row];
            if destination < 0 {
                assert_eq!(weight, f32::INFINITY);
                continue;
            }
            assert!((destination as usize) < components.len());
            assert_ne!(components[destination as usize], component);
            let entry = winners
                .entry(component)
                .or_insert((weight.to_bits(), row, destination));
            if weight < f32::from_bits(entry.0) {
                *entry = (weight.to_bits(), row, destination);
            }
        }
        winners
    }

    fn verify_production_bounds(device: &CubeDevice) {
        const N: usize = 128;
        let values = (0..N)
            .flat_map(|row| [(row / 16) as f32 * 10.0 + (row % 16) as f32 * 0.01, 0.0])
            .collect::<Vec<_>>();
        let host = hdbscan_core::KdTree::build(&values, N, 2).unwrap();
        let data = CubeBackend::float_from_data(TensorData::new(values, [N, 2]), device);
        let first = (0..N).map(|row| (row / 2 * 2) as i32).collect::<Vec<_>>();
        let second = (0..N).map(|row| (row / 16 * 16) as i32).collect::<Vec<_>>();

        // A zero poison would prune every positive edge without the production reset.
        let mut poisoned = KdBoundScratch {
            bounds: CubeBackend::int_from_data(TensorData::new(vec![0i32; N], [N]), device),
        };
        let poisoned_result = candidates(&data, &host, &first, false, &mut poisoned);
        let mut fresh = KdBoundScratch::new(&data, N, false);
        let fresh_result = candidates(&data, &host, &first, false, &mut fresh);
        // Per-row pruning may vary with scheduling; component winners must not.
        assert_eq!(
            winners(&first, &poisoned_result),
            winners(&first, &fresh_result)
        );
        let mut independent = KdBoundScratch::new(&data, N, true);
        assert_eq!(independent.bounds.shape().dims::<1>(), [1]);
        let independent_result = candidates(&data, &host, &first, true, &mut independent);
        assert_eq!(
            winners(&first, &poisoned_result),
            winners(&first, &independent_result)
        );
        assert_eq!(
            into_data_sync(independent.bounds.clone())
                .try_to_vec::<i32>()
                .unwrap(),
            vec![0x7f80_0000]
        );

        let bounds = into_data_sync(poisoned.bounds.clone())
            .try_to_vec::<i32>()
            .unwrap();
        let first_winners = winners(&first, &poisoned_result);
        assert_eq!(first_winners.len(), N / 2);
        for (component, bits) in bounds.iter().enumerate() {
            let expected = first_winners
                .get(&(component as i32))
                .map_or(f32::INFINITY.to_bits(), |winner| winner.0);
            assert_eq!(*bits as u32, expected, "bound at root {component}");
        }

        // Reuse the very same allocation and root IDs with much larger outgoing weights.
        // Its round-one bounds are genuine poison for this coarsened partition.
        let second_shared = candidates(&data, &host, &second, false, &mut poisoned);
        let second_independent = candidates(&data, &host, &second, true, &mut independent);
        assert_eq!(
            winners(&second, &second_shared),
            winners(&second, &second_independent)
        );
        assert_eq!(winners(&second, &second_shared).len(), N / 16);
        assert!(second_independent.0.iter().all(|&value| value > 90.0));
        assert!(f32::from_bits(bounds[0] as u32) < 1.0);

        // Check the complete production op, including its own scratch allocation.
        let (weights, destinations) = CubeBackend::boruvka_candidates_kd(
            data,
            CubeBackend::float_from_data(TensorData::new(vec![0.0f32; N], [N]), device),
            CubeBackend::int_from_data(TensorData::new(second.clone(), [N]), device),
            upload_tree(&host, &second, device),
            false,
        );
        let production = (
            into_data_sync(weights).try_to_vec().unwrap(),
            into_data_sync(destinations).try_to_vec().unwrap(),
        );
        assert_eq!(
            winners(&second, &production),
            winners(&second, &second_independent)
        );
    }

    #[test]
    fn kd_bound_production_reset_and_winners_cpu() {
        verify_production_bounds(&CubeDevice::Cpu(Default::default()));
    }

    #[test]
    fn kd_bound_production_reset_and_winners_cuda() {
        let count = cuscan_common::cuda_backend::try_device_count().unwrap();
        if count == 0 {
            assert_ne!(
                std::env::var("CUSCAN_REQUIRE_CUDA").as_deref(),
                Ok("1"),
                "required CUDA unavailable"
            );
            eprintln!("SKIPPED KD bounds CUDA: no usable device");
            return;
        }
        verify_production_bounds(&CubeDevice::Cuda(Default::default()));
    }
}
