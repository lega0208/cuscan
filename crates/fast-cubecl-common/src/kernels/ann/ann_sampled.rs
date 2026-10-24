//! Bounded sampled ANN phase kernels and scalar contract oracles.
//!
//! Kernels share the candidate, pair, endpoint-CSR, and merge algorithm across
//! CPU/CUDA; only row geometry and endpoint merge execution differ. Launch
//! orchestration remains in `ann.rs`.

#[cfg(test)]
#[path = "ann_sampled_tests.rs"]
mod injected_kernel_tests;

#[cfg(test)]
use std::collections::{BTreeMap, BTreeSet};

use super::ann_forest::{WorkspaceError, build_forest, plan_workspace, tree_count};
use super::orchestration::{
    AnnPlan, ann_independent_launch, ann_output, ann_should_stop, plan_for_client, reverse_count,
    reverse_scatter, reverse_zero,
};
use crate::error::AnnError;
use burn::{
    backend::{TensorMetadata, cubecl::dtype_to_storage_type},
    tensor::{DType, Shape},
};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor};
use cubecl::prelude::*;

#[cfg(test)]
const PRIORITY_MODULUS: u64 = 2_147_483_647;
// C=60 produced 268.5M mostly masked slots per iteration on the fixed input
// (CPU pair/merge ~5s). A separate per-class cap keeps staging work bounded.
const PREFIX_CHUNK: usize = 128;

/// Build bounded forest leaf membership on the host. Distances and graph
/// updates are executed by the shared pair-slot/endpoint pipeline.
pub(super) fn forest_seed(
    data: &CubeTensor,
    n: usize,
    features: usize,
    heap_width: usize,
    max_candidates: usize,
    seed: u64,
) -> Result<
    (
        CubeTensor,
        CubeTensor,
        CubeTensor,
        CubeTensor,
        CubeTensor,
        usize,
    ),
    WorkspaceError,
> {
    let trees = tree_count(n)?;
    let leaf_size = 10usize.max(heap_width.saturating_mul(2)).min(n);
    plan_workspace(n, heap_width, max_candidates, features, trees)?;
    let input_bytes = data.client.read_one_unchecked(data.handle.clone());
    let expected_bytes = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(DType::F32.size()))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    if input_bytes.len() != expected_bytes {
        return Err(WorkspaceError::InvalidDimensions);
    }
    let features_host = input_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk))
        .collect::<Vec<_>>();
    drop(input_bytes);
    let forest = build_forest(&features_host, n, features, seed, trees, leaf_size)?;
    let forest_pair_slots = forest
        .leaf_offsets
        .windows(2)
        .try_fold(0usize, |sum, bounds| {
            let leaf_size = bounds[1] - bounds[0];
            leaf_size
                .checked_mul(leaf_size.saturating_sub(1))
                .and_then(|pairs| sum.checked_add(pairs / 2))
                .ok_or(WorkspaceError::ArithmeticOverflow)
        })?;
    let heap_len = n
        .checked_mul(heap_width)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let sentinel_id = i32::try_from(n).map_err(|_| WorkspaceError::InvalidDimensions)?;
    let ids = vec![sentinel_id; heap_len];
    let distances = vec![f32::MAX; heap_len];
    let flags = vec![0i32; heap_len];
    let mut membership = Vec::with_capacity(forest.row_ids.len());
    for &id in &forest.row_ids {
        membership.push(i32::try_from(id).map_err(|_| WorkspaceError::InvalidDimensions)?);
    }
    let mut leaf_offsets = Vec::with_capacity(
        forest
            .leaf_offsets
            .len()
            .checked_mul(2)
            .ok_or(WorkspaceError::ArithmeticOverflow)?,
    );
    let mut pair_prefix = 0usize;
    for (leaf, &offset) in forest.leaf_offsets.iter().enumerate() {
        leaf_offsets.push(i32::try_from(offset).map_err(|_| WorkspaceError::InvalidDimensions)?);
        leaf_offsets
            .push(i32::try_from(pair_prefix).map_err(|_| WorkspaceError::ArithmeticOverflow)?);
        if let Some(&end) = forest.leaf_offsets.get(leaf + 1) {
            let length = end - offset;
            pair_prefix = pair_prefix
                .checked_add(
                    length
                        .checked_mul(length.saturating_sub(1))
                        .ok_or(WorkspaceError::ArithmeticOverflow)?
                        / 2,
                )
                .ok_or(WorkspaceError::ArithmeticOverflow)?;
        }
    }
    let rows_tensor = cube_tensor_from_vec(
        data.client.clone(),
        data.device.clone(),
        Shape::new([membership.len()]),
        DType::I32,
        membership,
    );
    let offsets_tensor = cube_tensor_from_vec(
        data.client.clone(),
        data.device.clone(),
        Shape::new([leaf_offsets.len() / 2, 2]),
        DType::I32,
        leaf_offsets,
    );
    drop(forest);
    drop(features_host);
    let client = data.client.clone();
    let device = data.device.clone();
    let shape = Shape::new([n, heap_width]);
    let ids = cube_tensor_from_vec(
        client.clone(),
        device.clone(),
        shape.clone(),
        DType::I32,
        ids,
    );
    let distances = cube_tensor_from_vec(
        client.clone(),
        device.clone(),
        shape.clone(),
        DType::F32,
        distances,
    );
    let flags = cube_tensor_from_vec(client, device, shape, DType::I32, flags);
    Ok((
        ids,
        distances,
        flags,
        rows_tensor,
        offsets_tensor,
        forest_pair_slots,
    ))
}

fn cube_tensor_from_vec<T: burn::tensor::Element>(
    client: cubecl::client::Client,
    device: burn_cubecl::CubeDevice,
    shape: Shape,
    dtype: DType,
    values: Vec<T>,
) -> CubeTensor {
    let tensor_data = burn::tensor::TensorData::new(values, shape.clone());
    assert_eq!(tensor_data.dtype(), dtype);
    let (bytes, _, _) = tensor_data.into_parts();
    CubeTensor::new_contiguous(client.clone(), device, shape, client.create(bytes), dtype)
}

fn empty_tensor(data: &CubeTensor, shape: Shape, dtype: DType) -> CubeTensor {
    let bytes = shape
        .num_elements()
        .checked_mul(dtype.size())
        .expect("ANN scratch tensor size overflows usize");
    CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        shape,
        data.client.empty(bytes),
        dtype,
    )
}

/// Production shared sampled NN-descent orchestration. Source heaps and flags
/// are frozen for the entire iteration; endpoint merge tiles accumulate into
/// alternating destination buffers before flags/convergence are finalized.
pub(super) fn approximate_neighbors(
    data: CubeTensor,
    k: usize,
    seed: u64,
    max_iterations: u32,
    squared: bool,
) -> Result<(CubeTensor, CubeTensor), AnnError> {
    let profile = std::env::var_os("FAST_UMAP_ANN_PROFILE").is_some();
    let init_start = std::time::Instant::now();
    let data = into_contiguous(data);
    let [n, features] = data.shape().dims::<2>();
    assert_eq!(data.dtype, DType::F32);
    let client = data.client.clone();
    let plan: AnnPlan = plan_for_client(&client, n, features, k)?;
    let width = plan.host.width;
    let capacity = plan.host.capacity;
    let workspace = plan.host.workspace;
    let n_u32 = u32::try_from(n).expect("ANN row count validated by plan");
    let width_u32 = u32::try_from(width).expect("ANN width validated by plan");
    let cap_u32 = u32::try_from(capacity).expect("ANN sample capacity validated by plan");
    let pair_slots = workspace.max_pair_slots_per_tile;
    let pair_elements = pair_slots;
    let endpoint_elements = workspace.max_endpoint_records;
    let max_total_slots = n
        .checked_mul(workspace.pair_slots_per_center)
        .expect("ANN pair slot count overflows usize");
    let (mut candidates, mut distances, mut flags, forest_rows, forest_offsets, forest_pairs) =
        forest_seed(&data, n, features, width, capacity, seed)
            .expect("ANN RP forest initialization exceeds workspace contract");
    let pair_left = empty_tensor(&data, Shape::new([pair_elements]), DType::I32);
    let pair_right = empty_tensor(&data, Shape::new([pair_elements]), DType::I32);
    let pair_distances = empty_tensor(&data, Shape::new([pair_elements]), DType::F32);
    let pair_valid = empty_tensor(&data, Shape::new([pair_elements]), DType::I32);
    let endpoint_counts = empty_tensor(&data, Shape::new([n]), DType::I32);
    let endpoint_offsets = empty_tensor(&data, Shape::new([n + 1]), DType::I32);
    let endpoint_cursors = empty_tensor(&data, Shape::new([n]), DType::I32);
    let reverse_counts = empty_tensor(&data, Shape::new([n]), DType::I32);
    let reverse_offsets = empty_tensor(&data, Shape::new([n + 1]), DType::I32);
    let reverse_cursors = empty_tensor(&data, Shape::new([n]), DType::I32);
    let reverse_elements = n.checked_mul(width).expect("ANN reverse size overflow");
    let reverse_sources = empty_tensor(&data, Shape::new([reverse_elements]), DType::I32);
    let endpoint_ids = empty_tensor(&data, Shape::new([endpoint_elements]), DType::I32);
    let endpoint_distances = empty_tensor(&data, Shape::new([endpoint_elements]), DType::F32);
    let sampled_ids = empty_tensor(&data, Shape::new([n, capacity * 2]), DType::I32);
    let sampled_priorities = empty_tensor(&data, Shape::new([n, capacity * 2]), DType::I32);
    let mut destination_a_ids = empty_tensor(&data, Shape::new([n, width]), DType::I32);
    let mut destination_a_distances = empty_tensor(&data, Shape::new([n, width]), DType::F32);
    let mut destination_b_ids = empty_tensor(&data, Shape::new([n, width]), DType::I32);
    let mut destination_b_distances = empty_tensor(&data, Shape::new([n, width]), DType::F32);
    let merge_workspace_ids = empty_tensor(&data, Shape::new([n, width]), DType::I32);
    let merge_workspace_distances = empty_tensor(&data, Shape::new([n, width]), DType::F32);
    let mut next_flags = empty_tensor(&data, Shape::new([n, width]), DType::I32);
    let new_count = empty_tensor(&data, Shape::new([1]), DType::I32);
    let pool_sizes = empty_tensor(&data, Shape::new([n * 2]), DType::I32);
    let pair_counts = empty_tensor(&data, Shape::new([n]), DType::I32);
    let center_offsets = empty_tensor(&data, Shape::new([n + 1]), DType::I32);
    let center_cursors = empty_tensor(&data, Shape::new([n]), DType::I32);
    let chunk_sums = empty_tensor(&data, Shape::new([n.div_ceil(PREFIX_CHUNK)]), DType::I32);
    let prefix = |values: &CubeTensor, offsets: &CubeTensor, cursors: &CubeTensor| {
        exclusive_prefix(&client, values, offsets, cursors, &chunk_sums, n)
    };
    let (helper_count, helper_dim) = ann_independent_launch(&client, n_u32)?;
    let (row_count, row_dim) = ann_independent_launch(&client, n_u32)?;
    let mut forest_offset = 0usize;
    let mut forest_destination_a = false;
    let mut forest_current_ids = candidates.clone();
    let mut forest_current_distances = distances.clone();
    while forest_offset < forest_pairs {
        let slots = (forest_pairs - forest_offset).min(pair_slots);
        let slots_u32 = u32::try_from(slots).expect("ANN forest tile exceeds CubeCL range");
        let (pair_count, pair_dim) = ann_independent_launch(&client, slots_u32)?;
        forest_pair_slots_kernel::launch(
            &client,
            pair_count.clone(),
            pair_dim,
            data.clone().into_tensor_arg(),
            forest_rows.clone().into_tensor_arg(),
            forest_offsets.clone().into_tensor_arg(),
            pair_left.clone().into_tensor_arg(),
            pair_right.clone().into_tensor_arg(),
            pair_distances.clone().into_tensor_arg(),
            pair_valid.clone().into_tensor_arg(),
            u32::try_from(forest_offset).expect("ANN forest offset exceeds CubeCL range"),
            slots_u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        endpoint_count_zero::launch(
            &client,
            row_count.clone(),
            row_dim,
            endpoint_counts.clone().into_tensor_arg(),
            dtype_to_storage_type(DType::I32),
        );
        endpoint_count_kernel::launch(
            &client,
            pair_count.clone(),
            pair_dim,
            pair_left.clone().into_tensor_arg(),
            pair_right.clone().into_tensor_arg(),
            pair_valid.clone().into_tensor_arg(),
            endpoint_counts.clone().into_tensor_arg(),
            slots_u32,
            dtype_to_storage_type(DType::I32),
        );
        prefix(&endpoint_counts, &endpoint_offsets, &endpoint_cursors)?;
        endpoint_scatter_kernel::launch(
            &client,
            pair_count.clone(),
            pair_dim,
            pair_left.clone().into_tensor_arg(),
            pair_right.clone().into_tensor_arg(),
            pair_distances.clone().into_tensor_arg(),
            pair_valid.clone().into_tensor_arg(),
            endpoint_cursors.clone().into_tensor_arg(),
            endpoint_ids.clone().into_tensor_arg(),
            endpoint_distances.clone().into_tensor_arg(),
            slots_u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        let (cpu_merge, merge_count, merge_dim) = plan.merge_geometry();
        let target_is_a = !forest_destination_a;
        if target_is_a {
            endpoint_merge_kernel::launch(
                &client,
                merge_count,
                merge_dim,
                forest_current_ids.clone().into_tensor_arg(),
                forest_current_distances.clone().into_tensor_arg(),
                endpoint_offsets.clone().into_tensor_arg(),
                endpoint_ids.clone().into_tensor_arg(),
                endpoint_distances.clone().into_tensor_arg(),
                destination_a_ids.clone().into_tensor_arg(),
                destination_a_distances.clone().into_tensor_arg(),
                merge_workspace_ids.clone().into_tensor_arg(),
                merge_workspace_distances.clone().into_tensor_arg(),
                width_u32,
                cpu_merge,
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            forest_current_ids = destination_a_ids.clone();
            forest_current_distances = destination_a_distances.clone();
        } else {
            endpoint_merge_kernel::launch(
                &client,
                merge_count,
                merge_dim,
                forest_current_ids.clone().into_tensor_arg(),
                forest_current_distances.clone().into_tensor_arg(),
                endpoint_offsets.clone().into_tensor_arg(),
                endpoint_ids.clone().into_tensor_arg(),
                endpoint_distances.clone().into_tensor_arg(),
                destination_b_ids.clone().into_tensor_arg(),
                destination_b_distances.clone().into_tensor_arg(),
                merge_workspace_ids.clone().into_tensor_arg(),
                merge_workspace_distances.clone().into_tensor_arg(),
                width_u32,
                cpu_merge,
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            forest_current_ids = destination_b_ids.clone();
            forest_current_distances = destination_b_distances.clone();
        }
        forest_destination_a = target_is_a;
        forest_offset += slots;
    }
    if forest_pairs > 0 {
        if forest_destination_a {
            std::mem::swap(&mut candidates, &mut destination_a_ids);
            std::mem::swap(&mut distances, &mut destination_a_distances);
        } else {
            std::mem::swap(&mut candidates, &mut destination_b_ids);
            std::mem::swap(&mut distances, &mut destination_b_distances);
        }
    }
    random_fill_kernel::launch(
        &client,
        row_count.clone(),
        row_dim,
        data.clone().into_tensor_arg(),
        candidates.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        flags.clone().into_tensor_arg(),
        n_u32,
        width_u32,
        (seed % 2_147_483_646) as u32 + 1,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    if profile {
        let _ = client.read_one_unchecked(candidates.handle.clone());
        eprintln!(
            "ANN_PROFILE init_ms={:.3} forest_pairs={forest_pairs} H={width} C={capacity} workspace_bytes={} W={} staging_bytes={}",
            init_start.elapsed().as_secs_f64() * 1000.0,
            workspace.workspace_bytes,
            workspace.formula_workspace_bytes,
            workspace.staging_bytes
        );
    }
    for iteration in 0..max_iterations {
        let pool_start = std::time::Instant::now();
        let source_ids = candidates.clone();
        let source_distances = distances.clone();
        let source_flags = flags.clone();
        let (reverse_count_units, reverse_dim) = ann_independent_launch(&client, n_u32)?;
        reverse_zero::launch(
            &client,
            reverse_count_units.clone(),
            reverse_dim,
            reverse_counts.clone().into_tensor_arg(),
            dtype_to_storage_type(DType::I32),
        );
        reverse_count::launch(
            &client,
            reverse_count_units.clone(),
            reverse_dim,
            source_ids.clone().into_tensor_arg(),
            reverse_counts.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype_to_storage_type(DType::I32),
        );
        prefix(&reverse_counts, &reverse_offsets, &reverse_cursors)?;
        reverse_scatter::launch(
            &client,
            reverse_count_units,
            reverse_dim,
            source_ids.clone().into_tensor_arg(),
            reverse_cursors.clone().into_tensor_arg(),
            reverse_sources.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype_to_storage_type(DType::I32),
        );
        sampled_pool_kernel::launch(
            &client,
            row_count.clone(),
            row_dim,
            source_ids.clone().into_tensor_arg(),
            source_flags.clone().into_tensor_arg(),
            reverse_offsets.clone().into_tensor_arg(),
            reverse_sources.clone().into_tensor_arg(),
            sampled_ids.clone().into_tensor_arg(),
            sampled_priorities.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            cap_u32,
            (seed % 2_147_483_646) as u32 + 1,
            iteration,
            dtype_to_storage_type(DType::I32),
        );
        pool_pair_count_kernel::launch(
            &client,
            row_count.clone(),
            row_dim,
            sampled_ids.clone().into_tensor_arg(),
            pool_sizes.clone().into_tensor_arg(),
            pair_counts.clone().into_tensor_arg(),
            n_u32,
            cap_u32,
            dtype_to_storage_type(DType::I32),
        );
        prefix(&pair_counts, &center_offsets, &center_cursors)?;
        let total_bytes = client.read_one_unchecked(
            center_offsets
                .handle
                .clone()
                .offset_start((n * DType::I32.size()) as u64),
        );
        let total_slots =
            i32::from_ne_bytes(total_bytes[..4].try_into().expect("ANN pair total is i32"));
        let total_slots = usize::try_from(total_slots).expect("ANN pair total is negative");
        assert!(
            total_slots <= max_total_slots,
            "ANN compact pair total exceeds planned bound"
        );
        // Forest pairs are already represented in the heap; only sampled join
        // slots pass through pair staging here.
        if profile {
            let _ = client.read_one_unchecked(sampled_ids.handle.clone());
            eprintln!(
                "ANN_PROFILE iteration={iteration} pools_ms={:.3}",
                pool_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        let join_start = std::time::Instant::now();
        let mut slot_offset = 0usize;
        let mut has_destination = false;
        let mut destination_is_a = false;
        let mut current_ids = source_ids.clone();
        let mut current_distances = source_distances.clone();
        while slot_offset < total_slots {
            let slots = (total_slots - slot_offset).min(pair_slots);
            let slots_u32 = u32::try_from(slots).expect("ANN pair tile exceeds CubeCL range");
            let (pair_count, pair_dim) = ann_independent_launch(&client, slots_u32)?;
            pair_slots_kernel::launch(
                &client,
                pair_count.clone(),
                pair_dim,
                data.clone().into_tensor_arg(),
                sampled_ids.clone().into_tensor_arg(),
                pool_sizes.clone().into_tensor_arg(),
                center_offsets.clone().into_tensor_arg(),
                pair_left.clone().into_tensor_arg(),
                pair_right.clone().into_tensor_arg(),
                pair_distances.clone().into_tensor_arg(),
                pair_valid.clone().into_tensor_arg(),
                u32::try_from(slot_offset).expect("ANN pair offset exceeds CubeCL range"),
                slots_u32,
                cap_u32,
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            endpoint_count_zero::launch(
                &client,
                row_count.clone(),
                row_dim,
                endpoint_counts.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            endpoint_count_kernel::launch(
                &client,
                pair_count.clone(),
                pair_dim,
                pair_left.clone().into_tensor_arg(),
                pair_right.clone().into_tensor_arg(),
                pair_valid.clone().into_tensor_arg(),
                endpoint_counts.clone().into_tensor_arg(),
                slots_u32,
                dtype_to_storage_type(DType::I32),
            );
            prefix(&endpoint_counts, &endpoint_offsets, &endpoint_cursors)?;
            endpoint_scatter_kernel::launch(
                &client,
                pair_count.clone(),
                pair_dim,
                pair_left.clone().into_tensor_arg(),
                pair_right.clone().into_tensor_arg(),
                pair_distances.clone().into_tensor_arg(),
                pair_valid.clone().into_tensor_arg(),
                endpoint_cursors.clone().into_tensor_arg(),
                endpoint_ids.clone().into_tensor_arg(),
                endpoint_distances.clone().into_tensor_arg(),
                slots_u32,
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            let (cpu_merge, merge_count, merge_dim) = plan.merge_geometry();
            let target_is_a = !has_destination || !destination_is_a;
            if target_is_a {
                endpoint_merge_kernel::launch(
                    &client,
                    merge_count,
                    merge_dim,
                    current_ids.clone().into_tensor_arg(),
                    current_distances.clone().into_tensor_arg(),
                    endpoint_offsets.clone().into_tensor_arg(),
                    endpoint_ids.clone().into_tensor_arg(),
                    endpoint_distances.clone().into_tensor_arg(),
                    destination_a_ids.clone().into_tensor_arg(),
                    destination_a_distances.clone().into_tensor_arg(),
                    merge_workspace_ids.clone().into_tensor_arg(),
                    merge_workspace_distances.clone().into_tensor_arg(),
                    width_u32,
                    cpu_merge,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                    ],
                );
                current_ids = destination_a_ids.clone();
                current_distances = destination_a_distances.clone();
            } else {
                endpoint_merge_kernel::launch(
                    &client,
                    merge_count,
                    merge_dim,
                    current_ids.clone().into_tensor_arg(),
                    current_distances.clone().into_tensor_arg(),
                    endpoint_offsets.clone().into_tensor_arg(),
                    endpoint_ids.clone().into_tensor_arg(),
                    endpoint_distances.clone().into_tensor_arg(),
                    destination_b_ids.clone().into_tensor_arg(),
                    destination_b_distances.clone().into_tensor_arg(),
                    merge_workspace_ids.clone().into_tensor_arg(),
                    merge_workspace_distances.clone().into_tensor_arg(),
                    width_u32,
                    cpu_merge,
                    [
                        dtype_to_storage_type(DType::F32),
                        dtype_to_storage_type(DType::I32),
                    ],
                );
                current_ids = destination_b_ids.clone();
                current_distances = destination_b_distances.clone();
            }
            destination_is_a = target_is_a;
            has_destination = true;
            slot_offset += slots;
        }
        if profile {
            let _ = client.read_one_unchecked(current_ids.handle.clone());
            eprintln!(
                "ANN_PROFILE iteration={iteration} pair_merge_ms={:.3} pair_slots={total_slots}",
                join_start.elapsed().as_secs_f64() * 1000.0
            );
        }
        // Return the old source allocation to the destination pool. Assigning
        // a cloned destination here would alias next iteration's frozen source.
        // With zero compact pairs no merge ran and the source heap stands.
        if has_destination && destination_is_a {
            std::mem::swap(&mut candidates, &mut destination_a_ids);
            std::mem::swap(&mut distances, &mut destination_a_distances);
        } else if has_destination {
            std::mem::swap(&mut candidates, &mut destination_b_ids);
            std::mem::swap(&mut distances, &mut destination_b_distances);
        }
        convergence_zero::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            new_count.clone().into_tensor_arg(),
            dtype_to_storage_type(DType::I32),
        );
        finalize_flags_kernel::launch(
            &client,
            row_count.clone(),
            row_dim,
            source_ids.into_tensor_arg(),
            source_flags.into_tensor_arg(),
            sampled_ids.clone().into_tensor_arg(),
            candidates.clone().into_tensor_arg(),
            next_flags.clone().into_tensor_arg(),
            new_count.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            cap_u32,
            dtype_to_storage_type(DType::I32),
        );
        std::mem::swap(&mut flags, &mut next_flags);
        if iteration > 0 {
            let bytes = client.read_one_unchecked(new_count.handle.clone());
            let count = i32::from_ne_bytes(bytes[..4].try_into().expect("ANN counter is i32"));
            if ann_should_stop(count, n, width) {
                break;
            }
        }
    }
    let output_indices = empty_tensor(&data, Shape::new([n, k]), DType::I32);
    let output_distances = empty_tensor(&data, Shape::new([n, k]), DType::F32);
    ann_output::launch(
        &client,
        helper_count,
        helper_dim,
        candidates.into_tensor_arg(),
        distances.into_tensor_arg(),
        output_indices.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        width_u32,
        u32::try_from(k).expect("ANN output width exceeds CubeCL range"),
        squared,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    Ok((output_indices, output_distances))
}

/// Parallel exclusive prefix of `values[0..len]` into `offsets[0..=len]`,
/// also seeding scatter cursors. Callers own the `ceil(len/128)` chunk buffer.
fn exclusive_prefix(
    client: &cubecl::client::Client,
    values: &CubeTensor,
    offsets: &CubeTensor,
    cursors: &CubeTensor,
    chunk_sums: &CubeTensor,
    len: usize,
) -> Result<(), AnnError> {
    let chunks = u32::try_from(len.div_ceil(PREFIX_CHUNK)).expect("ANN prefix chunks exceed u32");
    let chunk = PREFIX_CHUNK as u32;
    let (count, dim) = ann_independent_launch(client, chunks)?;
    chunk_sum_kernel::launch(
        client,
        count.clone(),
        dim,
        values.clone().into_tensor_arg(),
        chunk_sums.clone().into_tensor_arg(),
        chunk,
        dtype_to_storage_type(DType::I32),
    );
    chunk_prefix_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        chunk_sums.clone().into_tensor_arg(),
        dtype_to_storage_type(DType::I32),
    );
    chunk_scatter_kernel::launch(
        client,
        count,
        dim,
        values.clone().into_tensor_arg(),
        chunk_sums.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        cursors.clone().into_tensor_arg(),
        chunk,
        dtype_to_storage_type(DType::I32),
    );
    Ok(())
}

#[cube]
fn has_new_candidate<I: Int>(
    candidates: &Tensor<I>,
    retained: &Tensor<I>,
    row: usize,
    candidate: I,
    n: usize,
    width: usize,
) -> bool {
    let row_base = row * width;
    let mut found = false;
    for slot in 0..width {
        if candidates[row_base + slot] == candidate
            && retained[row_base + slot] == I::cast_from(0i32)
        {
            found = true;
        }
    }
    // A reverse new edge candidate -> row exists exactly when the candidate's
    // own row holds `row` with a new flag; scan it directly.
    let source = u32::cast_from(candidate) as usize;
    if source < n {
        let source_base = source * width;
        for slot in 0..width {
            if candidates[source_base + slot] == I::cast_from(row as i32)
                && retained[source_base + slot] == I::cast_from(0i32)
            {
                found = true;
            }
        }
    }
    found
}

#[cube]
fn add_mod<I: Int>(left: I, right: I, modulus: I) -> I {
    if left >= modulus - right {
        left - (modulus - right)
    } else {
        left + right
    }
}

#[cube]
fn park_miller<I: Int>(state: I) -> I {
    let modulus = I::cast_from(2_147_483_647i32);
    let state = if state == I::cast_from(0i32) {
        I::cast_from(1i32)
    } else {
        state
    };
    let hi = state / I::cast_from(127_773i32);
    let lo = state - hi * I::cast_from(127_773i32);
    let value = lo * I::cast_from(16_807i32) - hi * I::cast_from(2_836i32);
    if value <= I::cast_from(0i32) {
        value + modulus
    } else {
        value
    }
}

#[cube]
fn candidate_priority<I: Int>(
    row: usize,
    candidate: usize,
    seed: u32,
    iteration: u32,
    class: u32,
) -> I {
    let modulus = I::cast_from(2_147_483_647i32);
    let row_mod = I::cast_from((row % 2_147_483_647usize) as i32);
    let candidate_mod = I::cast_from((candidate % 2_147_483_647usize) as i32);
    let mut state = park_miller::<I>(add_mod::<I>(candidate_mod, I::cast_from(1i32), modulus));
    state = park_miller::<I>(add_mod::<I>(state, row_mod, modulus));
    let seed_mod = I::cast_from((seed % 2_147_483_647) as i32);
    state = park_miller::<I>(add_mod::<I>(state, seed_mod, modulus));
    let iteration_mod = I::cast_from((iteration % 2_147_483_647) as i32);
    state = park_miller::<I>(add_mod::<I>(state, iteration_mod, modulus));
    let class_mod = I::cast_from((class % 2_147_483_647) as i32);
    park_miller::<I>(add_mod::<I>(state, class_mod, modulus))
}

/// Insert into a priority-sorted pool holding `len` valid members and return
/// the new length. The pool keeps the `capacity` lowest `(priority, id)`
/// members; duplicates (which always carry the same priority) are ignored.
#[cube]
fn sampled_insert<I: Int>(
    ids: &mut Tensor<I>,
    priorities: &mut Tensor<I>,
    row_base: usize,
    capacity: usize,
    list: usize,
    id: I,
    priority: I,
    len: usize,
) -> usize {
    let base = row_base + list * capacity;
    let mut duplicate = false;
    for slot in 0..len {
        if ids[base + slot] == id {
            duplicate = true;
        }
    }
    let mut accepted = !duplicate;
    if accepted && len == capacity {
        let last = base + capacity - 1;
        accepted = priority < priorities[last] || (priority == priorities[last] && id < ids[last]);
    }
    let mut result = len;
    if accepted {
        let start = if len < capacity { len } else { capacity - 1 };
        let mut insert_at = 0usize;
        let mut placed = false;
        for step in 0..start {
            if !placed {
                let previous = base + start - step - 1;
                if priority < priorities[previous]
                    || (priority == priorities[previous] && id < ids[previous])
                {
                    ids[previous + 1] = ids[previous];
                    priorities[previous + 1] = priorities[previous];
                } else {
                    insert_at = start - step;
                    placed = true;
                }
            }
        }
        ids[base + insert_at] = id;
        priorities[base + insert_at] = priority;
        if len < capacity {
            result = len + 1;
        }
    }
    result
}

#[cube(launch)]
pub(super) fn random_fill_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    ids: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    flags: &mut Tensor<I>,
    n: u32,
    width: u32,
    seed: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if row < n {
        let base = row * width;
        let mut slot = 0usize;
        while slot < width && (u32::cast_from(ids[base + slot]) as usize) < n {
            slot += 1;
        }
        let mut attempt = 0u32;
        let mut state = candidate_priority::<I>(row, row + seed as usize, seed, 0, 7);
        while slot < width && attempt < (n as u32) * 4 {
            state = park_miller::<I>(add_mod::<I>(
                state,
                I::cast_from(attempt as i32 + 1),
                I::cast_from(2_147_483_647i32),
            ));
            let candidate = u32::cast_from(state) as usize % n;
            let mut duplicate = candidate == row;
            for existing in 0..slot {
                if u32::cast_from(ids[base + existing]) as usize == candidate {
                    duplicate = true;
                }
            }
            if !duplicate {
                let features = data.shape(1);
                let mut squared = F::new(0.0f32);
                for feature in 0..features {
                    let difference =
                        data[row * features + feature] - data[candidate * features + feature];
                    squared += difference * difference;
                }
                let distance_valid =
                    squared >= F::new(0.0f32) && squared <= F::new(3.402_823_5e38_f32);
                if distance_valid {
                    let mut rank = 0usize;
                    while rank < slot {
                        let old_id = u32::cast_from(ids[base + rank]) as usize;
                        if squared < distances[base + rank]
                            || (squared == distances[base + rank] && candidate < old_id)
                        {
                            break;
                        }
                        rank += 1;
                    }
                    let mut shift = slot;
                    while shift > rank {
                        ids[base + shift] = ids[base + shift - 1];
                        distances[base + shift] = distances[base + shift - 1];
                        flags[base + shift] = flags[base + shift - 1];
                        shift -= 1;
                    }
                    ids[base + rank] = I::cast_from(candidate as i32);
                    distances[base + rank] = squared;
                    flags[base + rank] = I::cast_from(0i32);
                    slot += 1;
                }
            }
            attempt += 1;
        }
        while slot < width {
            ids[base + slot] = I::cast_from(n as i32);
            distances[base + slot] = F::new(3.402_823_5e38_f32);
            flags[base + slot] = I::cast_from(0i32);
            slot += 1;
        }
    }
}

/// Build bounded priority-sampled new and old pools from frozen forward and
/// reverse adjacency. New-class membership takes precedence across both
/// directions; reverse CSR order does not influence the retained sample.
#[cube(launch)]
pub(super) fn sampled_pool_kernel<I: Int>(
    candidates: &Tensor<I>,
    retained: &Tensor<I>,
    reverse_offsets: &Tensor<I>,
    reverse_sources: &Tensor<I>,
    sampled_ids: &mut Tensor<I>,
    sampled_priorities: &mut Tensor<I>,
    n: u32,
    width: u32,
    capacity: u32,
    seed: u32,
    iteration: u32,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    let capacity = capacity as usize;
    if row < n && capacity > 0 {
        let output_base = row * capacity * 2;
        let sentinel_id = I::cast_from(n as i32);
        let sentinel_priority = I::cast_from(2_147_483_647i32);
        for slot in 0..(capacity * 2) {
            sampled_ids[output_base + slot] = sentinel_id;
            sampled_priorities[output_base + slot] = sentinel_priority;
        }
        let input_base = row * width;
        let mut new_len = 0usize;
        let mut old_len = 0usize;
        // New forward edges.
        for slot in 0..width {
            if retained[input_base + slot] == I::cast_from(0i32) {
                let candidate = u32::cast_from(candidates[input_base + slot]) as usize;
                if candidate < n && candidate != row {
                    new_len = sampled_insert(
                        sampled_ids,
                        sampled_priorities,
                        output_base,
                        capacity,
                        0,
                        I::cast_from(candidate as i32),
                        candidate_priority::<I>(row, candidate, seed, iteration, 1),
                        new_len,
                    );
                }
            }
        }
        // New reverse edges.
        let reverse_start = u32::cast_from(reverse_offsets[row]) as usize;
        let reverse_end = u32::cast_from(reverse_offsets[row + 1]) as usize;
        for position in reverse_start..reverse_end {
            let source = u32::cast_from(reverse_sources[position]) as usize;
            if source < n {
                let source_base = source * width;
                for slot in 0..width {
                    if candidates[source_base + slot] == I::cast_from(row as i32)
                        && retained[source_base + slot] == I::cast_from(0i32)
                    {
                        new_len = sampled_insert(
                            sampled_ids,
                            sampled_priorities,
                            output_base,
                            capacity,
                            0,
                            I::cast_from(source as i32),
                            candidate_priority::<I>(row, source, seed, iteration, 1),
                            new_len,
                        );
                    }
                }
            }
        }
        // Old forward edges, excluding any id that occurs in the new class.
        for slot in 0..width {
            if retained[input_base + slot] != I::cast_from(0i32) {
                let candidate = u32::cast_from(candidates[input_base + slot]) as usize;
                if candidate < n
                    && candidate != row
                    && !has_new_candidate(
                        candidates,
                        retained,
                        row,
                        I::cast_from(candidate as i32),
                        n,
                        width,
                    )
                {
                    old_len = sampled_insert(
                        sampled_ids,
                        sampled_priorities,
                        output_base,
                        capacity,
                        1,
                        I::cast_from(candidate as i32),
                        candidate_priority::<I>(row, candidate, seed, iteration, 2),
                        old_len,
                    );
                }
            }
        }
        // Old reverse edges.
        for position in reverse_start..reverse_end {
            let source = u32::cast_from(reverse_sources[position]) as usize;
            if source < n {
                let source_base = source * width;
                for slot in 0..width {
                    if candidates[source_base + slot] == I::cast_from(row as i32)
                        && retained[source_base + slot] != I::cast_from(0i32)
                        && !has_new_candidate(
                            candidates,
                            retained,
                            row,
                            I::cast_from(source as i32),
                            n,
                            width,
                        )
                    {
                        old_len = sampled_insert(
                            sampled_ids,
                            sampled_priorities,
                            output_base,
                            capacity,
                            1,
                            I::cast_from(source as i32),
                            candidate_priority::<I>(row, source, seed, iteration, 2),
                            old_len,
                        );
                    }
                }
            }
        }
    }
}

#[cube(launch)]
pub(super) fn forest_pair_slots_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    forest_rows: &Tensor<I>,
    leaf_offsets: &Tensor<I>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    slot_offset: u32,
    slots: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize {
        let global_pair = slot_offset as usize + slot;
        let leaf_count = leaf_offsets.shape(0) - 1;
        let mut leaf = 0usize;
        let mut upper = leaf_count;
        while upper - leaf > 1 {
            let middle = leaf + (upper - leaf) / 2;
            let prefix = u32::cast_from(leaf_offsets[middle * 2 + 1]) as usize;
            if prefix <= global_pair {
                leaf = middle;
            } else {
                upper = middle;
            }
        }
        let found = global_pair < u32::cast_from(leaf_offsets[leaf_count * 2 + 1]) as usize;
        if found {
            let mut local_pair = global_pair - u32::cast_from(leaf_offsets[leaf * 2 + 1]) as usize;
            let start = u32::cast_from(leaf_offsets[leaf * 2]) as usize;
            let end = u32::cast_from(leaf_offsets[(leaf + 1) * 2]) as usize;
            let length = end - start;
            let mut left_slot = 0usize;
            while local_pair >= length - left_slot - 1 {
                local_pair -= length - left_slot - 1;
                left_slot += 1;
            }
            let right_slot = left_slot + 1 + local_pair;
            let left = u32::cast_from(forest_rows[start + left_slot]) as usize;
            let right = u32::cast_from(forest_rows[start + right_slot]) as usize;
            let features = data.shape(1);
            let mut squared = F::new(0.0f32);
            for feature in 0..features {
                let difference = data[left * features + feature] - data[right * features + feature];
                squared += difference * difference;
            }
            let valid = squared >= F::new(0.0f32) && squared <= F::new(3.402_823_5e38_f32);
            pair_left[slot] = I::cast_from(left as i32);
            pair_right[slot] = I::cast_from(right as i32);
            pair_distances[slot] = squared;
            pair_valid[slot] = I::cast_from(if valid { 1i32 } else { 0i32 });
        }
        if !found {
            pair_left[slot] = I::cast_from(0i32);
            pair_right[slot] = I::cast_from(0i32);
            pair_distances[slot] = F::new(0.0f32);
            pair_valid[slot] = I::cast_from(0i32);
        }
    }
}

/// Count compact local-join pairs per center: `nn*(nn-1)/2 + nn*no` for the
/// valid prefix of each sampled class. Pools are priority-sorted with sentinel
/// ids last, so the valid members are exactly the leading run.
#[cube(launch)]
pub(super) fn pool_pair_count_kernel<I: Int>(
    sampled_ids: &Tensor<I>,
    pool_sizes: &mut Tensor<I>,
    pair_counts: &mut Tensor<I>,
    n: u32,
    capacity: u32,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let capacity = capacity as usize;
    if row < n {
        let base = row * capacity * 2;
        let mut new_count = 0usize;
        let mut old_count = 0usize;
        for slot in 0..capacity {
            if (u32::cast_from(sampled_ids[base + slot]) as usize) < n {
                new_count += 1;
            }
            if (u32::cast_from(sampled_ids[base + capacity + slot]) as usize) < n {
                old_count += 1;
            }
        }
        pool_sizes[row * 2] = I::cast_from(new_count as i32);
        pool_sizes[row * 2 + 1] = I::cast_from(old_count as i32);
        let pairs = new_count * (new_count.saturating_sub(1)) / 2 + new_count * old_count;
        pair_counts[row] = I::cast_from(pairs as i32);
    }
}

/// Per-chunk sums for the parallel exclusive prefix.
#[cube(launch)]
pub(super) fn chunk_sum_kernel<I: Int>(
    values: &Tensor<I>,
    chunk_sums: &mut Tensor<I>,
    chunk: u32,
    #[define(I)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS_X as usize;
    let chunk = chunk as usize;
    let len = values.shape(0);
    if index < chunk_sums.shape(0) {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        let mut sum = I::cast_from(0i32);
        for position in start..end {
            sum += values[position];
        }
        chunk_sums[index] = sum;
    }
}

/// Serial exclusive prefix over chunk sums (one short pass).
#[cube(launch)]
pub(super) fn chunk_prefix_kernel<I: Int>(
    chunk_sums: &mut Tensor<I>,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let mut running = I::cast_from(0i32);
        for index in 0..chunk_sums.shape(0) {
            let value = chunk_sums[index];
            chunk_sums[index] = running;
            running += value;
        }
    }
}

/// Write `offsets[0..=len]` (exclusive prefix plus total) and seed cursors.
#[cube(launch)]
pub(super) fn chunk_scatter_kernel<I: Int>(
    values: &Tensor<I>,
    chunk_offsets: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<I>,
    chunk: u32,
    #[define(I)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS_X as usize;
    let chunk = chunk as usize;
    let len = values.shape(0);
    if index < chunk_offsets.shape(0) {
        let start = index * chunk;
        let end = (start + chunk).min(len);
        let mut running = chunk_offsets[index];
        for position in start..end {
            offsets[position] = running;
            cursors[position] = running;
            running += values[position];
        }
        if end == len {
            offsets[len] = running;
        }
    }
}

/// Emit compact local-join slots `[slot_offset, slot_offset+slots)` using the
/// per-center exclusive pair offsets: new×new `i<j` first, then new×old.
#[cube(launch)]
pub(super) fn pair_slots_kernel<F: Float, I: Int>(
    data: &Tensor<F>,
    sampled_ids: &Tensor<I>,
    pool_sizes: &Tensor<I>,
    center_offsets: &Tensor<I>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    slot_offset: u32,
    slots: u32,
    capacity: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    let capacity = capacity as usize;
    if slot < slots as usize {
        let n = data.shape(0);
        let features = data.shape(1);
        let flattened = slot_offset as usize + slot;
        // Largest center with center_offsets[center] <= flattened.
        let mut low = 0usize;
        let mut high = center_offsets.shape(0) - 1;
        while high - low > 1 {
            let middle = (low + high) / 2;
            if (u32::cast_from(center_offsets[middle]) as usize) <= flattened {
                low = middle;
            } else {
                high = middle;
            }
        }
        let center = low;
        let mut local = flattened - u32::cast_from(center_offsets[center]) as usize;
        let new_count = u32::cast_from(pool_sizes[center * 2]) as usize;
        let old_count = u32::cast_from(pool_sizes[center * 2 + 1]) as usize;
        let base = center * capacity * 2;
        let sentinel = I::cast_from(n as i32);
        let mut left = sentinel;
        let mut right = sentinel;
        let triangle = new_count * (new_count.saturating_sub(1)) / 2;
        if local < triangle {
            let mut i = 0usize;
            let mut done = false;
            while !done {
                let row_len = new_count - 1 - i;
                if local < row_len {
                    left = sampled_ids[base + i];
                    right = sampled_ids[base + i + 1 + local];
                    done = true;
                } else {
                    local -= row_len;
                    i += 1;
                }
            }
        } else if old_count > 0 {
            let cross = local - triangle;
            let safe_old_count = old_count.max(1);
            left = sampled_ids[base + cross / safe_old_count];
            right = sampled_ids[base + capacity + cross % safe_old_count];
        }
        let left_id = u32::cast_from(left) as usize;
        let right_id = u32::cast_from(right) as usize;
        let valid = left_id < n && right_id < n && left_id != right_id;
        pair_left[slot] = left;
        pair_right[slot] = right;
        let mut squared = F::new(0.0f32);
        if valid {
            for feature in 0..features {
                let difference =
                    data[left_id * features + feature] - data[right_id * features + feature];
                squared += difference * difference;
            }
        }
        let max_distance = F::new(3.402_823_5e38_f32);
        let distance_valid = valid && squared >= F::new(0.0f32) && squared <= max_distance;
        pair_valid[slot] = I::cast_from(if distance_valid { 1i32 } else { 0i32 });
        pair_distances[slot] = squared;
    }
}

#[cube(launch)]
pub(super) fn endpoint_count_zero<I: Int>(
    endpoint_counts: &mut Tensor<I>,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    if row < endpoint_counts.shape(0) {
        endpoint_counts[row] = I::cast_from(0i32);
    }
}

#[cube(launch)]
pub(super) fn endpoint_count_kernel<I: Int>(
    pair_left: &Tensor<I>,
    pair_right: &Tensor<I>,
    pair_valid: &Tensor<I>,
    endpoint_counts: &mut Tensor<Atomic<I>>,
    slots: u32,
    #[define(I)] _dtype: ElemType,
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize && pair_valid[slot] != I::cast_from(0i32) {
        let left = u32::cast_from(pair_left[slot]) as usize;
        let right = u32::cast_from(pair_right[slot]) as usize;
        endpoint_counts[left].fetch_add(I::cast_from(1i32));
        endpoint_counts[right].fetch_add(I::cast_from(1i32));
    }
}

#[cube(launch)]
pub(super) fn endpoint_scatter_kernel<F: Float, I: Int>(
    pair_left: &Tensor<I>,
    pair_right: &Tensor<I>,
    pair_distances: &Tensor<F>,
    pair_valid: &Tensor<I>,
    endpoint_cursors: &mut Tensor<Atomic<I>>,
    endpoint_neighbor_ids: &mut Tensor<I>,
    endpoint_distances: &mut Tensor<F>,
    slots: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize && pair_valid[slot] != I::cast_from(0i32) {
        let left = u32::cast_from(pair_left[slot]) as usize;
        let right = u32::cast_from(pair_right[slot]) as usize;
        let left_position =
            u32::cast_from(endpoint_cursors[left].fetch_add(I::cast_from(1i32))) as usize;
        let right_position =
            u32::cast_from(endpoint_cursors[right].fetch_add(I::cast_from(1i32))) as usize;
        endpoint_neighbor_ids[left_position] = I::cast_from(right as i32);
        endpoint_distances[left_position] = pair_distances[slot];
        endpoint_neighbor_ids[right_position] = I::cast_from(left as i32);
        endpoint_distances[right_position] = pair_distances[slot];
    }
}

#[cube]
fn serial_endpoint_insert<F: Float, I: Int>(
    n: usize,
    width: usize,
    row: usize,
    id: I,
    proposal_distance: F,
    ids: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    base: usize,
) {
    let id_value = u32::cast_from(id) as usize;
    let max_distance = F::new(3.402_823_5e38_f32);
    let worst_index = u32::cast_from(ids[base + width - 1]) as usize;
    let worst_distance = distances[base + width - 1];
    // A full heap cannot change for a record not better than its worst entry.
    let improves = worst_index >= n
        || proposal_distance < worst_distance
        || (proposal_distance == worst_distance && id_value < worst_index);
    if improves
        && id_value < n
        && id_value != row
        && proposal_distance >= F::new(0.0f32)
        && proposal_distance <= max_distance
    {
        let mut duplicate = width;
        for slot in 0..width {
            if ids[base + slot] == id {
                duplicate = slot;
            }
        }
        let distance = if duplicate < width {
            let old_distance = distances[base + duplicate];
            if proposal_distance < old_distance {
                proposal_distance
            } else {
                old_distance
            }
        } else {
            proposal_distance
        };
        if duplicate < width {
            let mut slot = duplicate;
            while slot + 1 < width {
                ids[base + slot] = ids[base + slot + 1];
                distances[base + slot] = distances[base + slot + 1];
                slot += 1;
            }
            ids[base + width - 1] = I::cast_from(n as i32);
            distances[base + width - 1] = F::new(3.402_823_5e38_f32);
        }
        let mut rank = 0usize;
        while rank < width {
            let old_id = ids[base + rank];
            let old_index = u32::cast_from(old_id) as usize;
            if old_index >= n
                || distance < distances[base + rank]
                || (distance == distances[base + rank] && id_value < old_index)
            {
                break;
            }
            rank += 1;
        }
        if rank < width {
            let mut slot = width - 1;
            while slot > rank {
                ids[base + slot] = ids[base + slot - 1];
                distances[base + slot] = distances[base + slot - 1];
                slot -= 1;
            }
            ids[base + rank] = id;
            distances[base + rank] = distance;
        }
    }
}

#[cube]
fn cooperative_endpoint_insert<F: Float, I: Int>(
    n: usize,
    width: usize,
    row: usize,
    id: I,
    proposal_distance: F,
    toggle: u32,
    ids_a: &mut Tensor<I>,
    distances_a: &mut Tensor<F>,
    ids_b: &mut Tensor<I>,
    distances_b: &mut Tensor<F>,
    duplicate_slots: &mut Shared<[I]>,
    rank_counts: &mut Shared<[I]>,
    metadata: &mut Shared<[I]>,
    shared_distance: &mut Shared<[F]>,
    base: usize,
) -> u32 {
    let lane = UNIT_POS_X as usize;
    let id_value = u32::cast_from(id) as usize;
    let max_distance = F::new(3.402_823_5e38_f32);
    let valid = id_value < n
        && id_value != row
        && proposal_distance >= F::new(0.0f32)
        && proposal_distance <= max_distance;
    let mut slot = lane;
    let mut duplicate_slot = width;
    while slot < width {
        let old_id = if toggle == 0 {
            ids_a[base + slot]
        } else {
            ids_b[base + slot]
        };
        if old_id == id {
            duplicate_slot = slot;
        }
        slot += 32;
    }
    duplicate_slots[lane] = I::cast_from(duplicate_slot as i32);
    sync_cube();
    if lane == 0 {
        let mut found = width;
        for thread in 0..32 {
            let candidate = u32::cast_from(duplicate_slots[thread]) as usize;
            if candidate < found {
                found = candidate;
            }
        }
        metadata[0] = I::cast_from(found as i32);
        let old_distance = if found < width {
            if toggle == 0 {
                distances_a[base + found]
            } else {
                distances_b[base + found]
            }
        } else {
            proposal_distance
        };
        shared_distance[0] = if proposal_distance < old_distance {
            proposal_distance
        } else {
            old_distance
        };
    }
    sync_cube();
    let duplicate = u32::cast_from(metadata[0]) as usize;
    let distance = shared_distance[0];
    let mut rank_count = 0i32;
    slot = lane;
    while slot < width {
        let old_id = if toggle == 0 {
            ids_a[base + slot]
        } else {
            ids_b[base + slot]
        };
        let old_distance = if toggle == 0 {
            distances_a[base + slot]
        } else {
            distances_b[base + slot]
        };
        let old_index = u32::cast_from(old_id) as usize;
        if slot != duplicate
            && old_index < n
            && (old_distance < distance || (old_distance == distance && old_index < id_value))
        {
            rank_count += 1;
        }
        slot += 32;
    }
    rank_counts[lane] = I::cast_from(rank_count);
    sync_cube();
    if lane == 0 {
        let mut rank = 0i32;
        for thread in 0..32 {
            rank += i32::cast_from(rank_counts[thread]);
        }
        metadata[1] = I::cast_from(rank);
        metadata[2] = I::cast_from(if valid && (duplicate < width || (rank as usize) < width) {
            1i32
        } else {
            0i32
        });
    }
    sync_cube();
    let rank = u32::cast_from(metadata[1]) as usize;
    let accepted = metadata[2] != I::cast_from(0i32);
    if valid && accepted {
        let output_toggle = toggle ^ 1u32;
        slot = lane;
        while slot < width {
            let mut source_slot = slot;
            let mut proposal_here = false;
            if duplicate < width {
                if rank < duplicate {
                    if slot == rank {
                        proposal_here = true;
                    } else if slot > rank && slot <= duplicate {
                        source_slot = slot - 1;
                    } else if slot > duplicate {
                        source_slot = slot;
                    }
                } else if rank > duplicate {
                    if slot == rank {
                        proposal_here = true;
                    } else if slot >= duplicate && slot < rank {
                        source_slot = slot + 1;
                    }
                } else if slot == rank {
                    proposal_here = true;
                }
            } else if slot == rank {
                proposal_here = true;
            } else if slot > rank {
                source_slot = slot - 1;
            }
            if output_toggle == 0 {
                if proposal_here {
                    ids_a[base + slot] = id;
                    distances_a[base + slot] = distance;
                } else if toggle == 0 {
                    ids_a[base + slot] = ids_a[base + source_slot];
                    distances_a[base + slot] = distances_a[base + source_slot];
                } else {
                    ids_a[base + slot] = ids_b[base + source_slot];
                    distances_a[base + slot] = distances_b[base + source_slot];
                }
            } else if proposal_here {
                ids_b[base + slot] = id;
                distances_b[base + slot] = distance;
            } else if toggle == 0 {
                ids_b[base + slot] = ids_a[base + source_slot];
                distances_b[base + slot] = distances_a[base + source_slot];
            } else {
                ids_b[base + slot] = ids_b[base + source_slot];
                distances_b[base + slot] = distances_b[base + source_slot];
            }
            slot += 32;
        }
        sync_cube();
        output_toggle
    } else {
        toggle
    }
}

/// Merge endpoint CSR records with frozen source heaps. Candidate ordering is
/// total `(distance,id)`; duplicate ids retain their minimum proposal distance.
/// CPU owns a row sequentially, while CUDA lanes distribute candidate slots
/// and write their unique rank in the sorted output.
#[cube(launch)]
pub(super) fn endpoint_merge_kernel<F: Float, I: Int>(
    source_ids: &Tensor<I>,
    source_distances: &Tensor<F>,
    endpoint_offsets: &Tensor<I>,
    endpoint_ids: &Tensor<I>,
    endpoint_distances: &Tensor<F>,
    output_ids: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    workspace_ids: &mut Tensor<I>,
    workspace_distances: &mut Tensor<F>,
    width: u32,
    #[comptime] cpu_merge: bool,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let lane = UNIT_POS_X as usize;
    let row = if comptime![cpu_merge] {
        (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize
    } else {
        CUBE_POS_X as usize
    };
    let n = source_ids.shape(0);
    let width = width as usize;
    if row < n {
        let start = u32::cast_from(endpoint_offsets[row]) as usize;
        let end = u32::cast_from(endpoint_offsets[row + 1]) as usize;
        let row_base = row * width;
        if comptime![cpu_merge] {
            for slot in 0..width {
                output_ids[row_base + slot] = source_ids[row_base + slot];
                output_distances[row_base + slot] = source_distances[row_base + slot];
            }
            for position in start..end {
                serial_endpoint_insert::<F, I>(
                    n,
                    width,
                    row,
                    endpoint_ids[position],
                    endpoint_distances[position],
                    output_ids,
                    output_distances,
                    row_base,
                );
            }
        } else {
            let mut slot = lane;
            while slot < width {
                output_ids[row_base + slot] = source_ids[row_base + slot];
                output_distances[row_base + slot] = source_distances[row_base + slot];
                slot += 32;
            }
            sync_cube();
            let mut duplicate_slots = Shared::<[I]>::new_slice(32usize);
            let mut rank_counts = Shared::<[I]>::new_slice(32usize);
            let mut metadata = Shared::<[I]>::new_slice(3usize);
            let mut shared_distance = Shared::<[F]>::new_slice(1usize);
            let mut survivors = Shared::<[u32]>::new_slice(32usize);
            let mut toggle = 0u32;
            let mut batch = start;
            // The merge result is the top-`width` of the union with per-id
            // minimum distances, independent of record order. A record that is
            // not better than the current worst entry of a full heap can never
            // change it, so lanes filter a 32-record batch in parallel against
            // the (conservatively stale) batch-start worst entry and only the
            // survivors take the cooperative insert path.
            while batch < end {
                let position = batch + lane;
                let mut keep = 0u32;
                if position < end {
                    let worst = row_base + width - 1;
                    let worst_id = if toggle == 0 {
                        output_ids[worst]
                    } else {
                        workspace_ids[worst]
                    };
                    let worst_distance = if toggle == 0 {
                        output_distances[worst]
                    } else {
                        workspace_distances[worst]
                    };
                    let worst_index = u32::cast_from(worst_id) as usize;
                    let proposal = endpoint_distances[position];
                    let proposal_index = u32::cast_from(endpoint_ids[position]) as usize;
                    if worst_index >= n
                        || proposal < worst_distance
                        || (proposal == worst_distance && proposal_index < worst_index)
                    {
                        keep = 1u32;
                    }
                }
                survivors[lane] = keep;
                sync_cube();
                for offset in 0..32usize {
                    if survivors[offset] != 0u32 {
                        toggle = cooperative_endpoint_insert::<F, I>(
                            n,
                            width,
                            row,
                            endpoint_ids[batch + offset],
                            endpoint_distances[batch + offset],
                            toggle,
                            output_ids,
                            output_distances,
                            workspace_ids,
                            workspace_distances,
                            &mut duplicate_slots,
                            &mut rank_counts,
                            &mut metadata,
                            &mut shared_distance,
                            row_base,
                        );
                    }
                }
                sync_cube();
                batch += 32;
            }
            if toggle != 0 {
                slot = lane;
                while slot < width {
                    output_ids[row_base + slot] = workspace_ids[row_base + slot];
                    output_distances[row_base + slot] = workspace_distances[row_base + slot];
                    slot += 32;
                }
            }
            sync_cube();
        }
    }
}

#[cube(launch)]
pub(super) fn convergence_zero<I: Int>(new_count: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    if ABSOLUTE_POS_X == 0 {
        new_count[0] = I::cast_from(0i32);
    }
}

/// Finalize edge new/old flags from the immutable source and frozen sampled
/// pools. The convergence counter counts only final ids absent from source.
#[cube(launch)]
pub(super) fn finalize_flags_kernel<I: Int>(
    source_ids: &Tensor<I>,
    source_flags: &Tensor<I>,
    sampled_ids: &Tensor<I>,
    output_ids: &Tensor<I>,
    output_flags: &mut Tensor<I>,
    new_count: &mut Tensor<Atomic<I>>,
    n: u32,
    width: u32,
    capacity: u32,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    let capacity = capacity as usize;
    if row < n {
        let mut additions = I::cast_from(0i32);
        let row_base = row * width;
        let sampled_base = row * capacity * 2;
        for slot in 0..width {
            let id = output_ids[row_base + slot];
            let id_value = u32::cast_from(id) as usize;
            let mut in_source = false;
            let mut source_new = false;
            if id_value < n {
                for source_slot in 0..width {
                    if source_ids[row_base + source_slot] == id {
                        in_source = true;
                        if source_flags[row_base + source_slot] == I::cast_from(0i32) {
                            source_new = true;
                        }
                    }
                }
            }
            let mut sampled = false;
            if in_source && source_new {
                for sampled_slot in 0..capacity {
                    if sampled_ids[sampled_base + sampled_slot] == id {
                        sampled = true;
                    }
                    // The source-new edge `row -> id` was also consumed when
                    // the reverse source `row` was selected in id's new pool.
                    if sampled_ids[(id_value * capacity * 2) + sampled_slot]
                        == I::cast_from(row as i32)
                    {
                        sampled = true;
                    }
                }
            }
            let is_new = id_value < n && (!in_source || (source_new && !sampled));
            output_flags[row_base + slot] = I::cast_from(if is_new { 0i32 } else { 1i32 });
            if id_value < n && !in_source {
                additions += I::cast_from(1i32);
            }
        }
        if additions > I::cast_from(0i32) {
            new_count[0].fetch_add(additions);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct Candidate {
    pub(super) id: usize,
    pub(super) is_new: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg(test)]
pub(super) struct SampledPools {
    pub(super) new: Vec<usize>,
    pub(super) old: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg(test)]
pub(super) struct Neighbor {
    pub(super) id: usize,
    pub(super) distance: f32,
    pub(super) is_new: bool,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg(test)]
pub(super) struct MergeResult {
    pub(super) rows: Vec<Vec<Neighbor>>,
    pub(super) surviving_additions: usize,
}

/// Deduplicate forward and reverse candidates, giving the new class
/// precedence, then deterministically retain the lowest-priority `capacity`
/// members in each independent class.
#[cfg(test)]
pub(super) fn sample_pools(
    row: usize,
    n: usize,
    forward: &[Candidate],
    reverse: &[Candidate],
    capacity: usize,
    seed: u64,
    iteration: u32,
) -> SampledPools {
    let mut classes = BTreeMap::<usize, bool>::new();
    for candidate in forward.iter().chain(reverse) {
        if candidate.id < n && candidate.id != row {
            classes
                .entry(candidate.id)
                .and_modify(|is_new| *is_new |= candidate.is_new)
                .or_insert(candidate.is_new);
        }
    }
    let mut new = Vec::new();
    let mut old = Vec::new();
    for (id, is_new) in classes {
        let priority = priority(seed, iteration, row, id, is_new);
        if is_new {
            new.push((priority, id));
        } else {
            old.push((priority, id));
        }
    }
    let trim = |values: &mut Vec<(u64, usize)>| {
        values.sort_unstable();
        values.truncate(capacity);
        values.iter().map(|&(_, id)| id).collect()
    };
    SampledPools {
        new: trim(&mut new),
        old: trim(&mut old),
    }
}

/// Enumerate local-join pair slots: new×new with `i<j`, then new×old.
/// Invalid/self ids are rejected and duplicate ids within either pool are
/// ignored, matching the pair-once-per-center contract.
#[cfg(test)]
pub(super) fn local_pairs(pools: &SampledPools, center: usize, n: usize) -> Vec<(usize, usize)> {
    let new = unique_valid(&pools.new, center, n);
    let old = unique_valid(&pools.old, center, n)
        .into_iter()
        .filter(|id| !new.contains(id))
        .collect::<Vec<_>>();
    let mut pairs =
        Vec::with_capacity(new.len() * (new.len().saturating_sub(1)) / 2 + new.len() * old.len());
    for left in 0..new.len() {
        for right in left + 1..new.len() {
            pairs.push(ordered_pair(new[left], new[right]));
        }
    }
    for &new_id in &new {
        for &old_id in &old {
            pairs.push(ordered_pair(new_id, old_id));
        }
    }
    pairs
}

/// Merge endpoint proposals against a frozen source graph. Distances for a
/// pair are shared by both endpoint updates; duplicate proposals keep the
/// minimum distance, and convergence counts only final additions absent from
/// the source row.
#[cfg(test)]
pub(super) fn merge_proposals(
    source: &[Vec<Neighbor>],
    proposals: &[(usize, usize, f32)],
    sampled_new: &[BTreeSet<usize>],
    heap_width: usize,
) -> MergeResult {
    let n = source.len();
    assert_eq!(sampled_new.len(), n);
    assert!(heap_width > 0);
    let mut rows = source.to_vec();
    for row in &mut rows {
        row.sort_unstable_by(neighbor_order);
        row.dedup_by_key(|neighbor| neighbor.id);
        row.truncate(heap_width);
    }
    for &(left, right, distance) in proposals {
        if left >= n || right >= n || left == right || !distance.is_finite() || distance < 0.0 {
            continue;
        }
        merge_endpoint(&mut rows[left], right, distance, heap_width);
        merge_endpoint(&mut rows[right], left, distance, heap_width);
    }
    let mut surviving_additions = 0usize;
    for row in 0..n {
        rows[row].sort_unstable_by(neighbor_order);
        let source_ids = source[row]
            .iter()
            .map(|neighbor| neighbor.id)
            .collect::<BTreeSet<_>>();
        for neighbor in &mut rows[row] {
            let was_source_new = source[row]
                .iter()
                .find(|old| old.id == neighbor.id)
                .is_some_and(|old| old.is_new);
            neighbor.is_new = !source_ids.contains(&neighbor.id)
                || (was_source_new
                    && !sampled_new[row].contains(&neighbor.id)
                    && !sampled_new
                        .get(neighbor.id)
                        .is_some_and(|reverse| reverse.contains(&row)));
            if !source_ids.contains(&neighbor.id) {
                surviving_additions = surviving_additions
                    .checked_add(1)
                    .expect("surviving addition count overflow");
            }
        }
    }
    MergeResult {
        rows,
        surviving_additions,
    }
}

#[cfg(test)]
fn merge_endpoint(row: &mut Vec<Neighbor>, id: usize, distance: f32, capacity: usize) {
    if let Some(existing) = row.iter_mut().find(|neighbor| neighbor.id == id) {
        existing.distance = existing.distance.min(distance);
        return;
    }
    row.push(Neighbor {
        id,
        distance,
        is_new: true,
    });
    row.sort_unstable_by(neighbor_order);
    row.truncate(capacity);
}

#[cfg(test)]
fn neighbor_order(left: &Neighbor, right: &Neighbor) -> std::cmp::Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.id.cmp(&right.id))
}

#[cfg(test)]
fn unique_valid(ids: &[usize], center: usize, n: usize) -> Vec<usize> {
    ids.iter()
        .copied()
        .filter(|&id| id < n && id != center)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
fn ordered_pair(left: usize, right: usize) -> (usize, usize) {
    if left < right {
        (left, right)
    } else {
        (right, left)
    }
}

#[cfg(test)]
fn priority(seed: u64, iteration: u32, row: usize, id: usize, is_new: bool) -> u64 {
    fn pm(state: u64) -> u64 {
        let state = state % PRIORITY_MODULUS;
        let state = if state == 0 { 1 } else { state };
        let hi = state / 127_773;
        let lo = state - hi * 127_773;
        let value = (lo * 16_807) as i64 - (hi * 2_836) as i64;
        if value <= 0 {
            (value + PRIORITY_MODULUS as i64) as u64
        } else {
            value as u64
        }
    }
    fn add(state: u64, value: u64) -> u64 {
        (state + value % PRIORITY_MODULUS) % PRIORITY_MODULUS
    }
    let mut state = pm(add(id as u64, 1));
    state = pm(add(state, row as u64));
    state = pm(add(state, seed));
    state = pm(add(state, iteration as u64));
    pm(add(state, if is_new { 1 } else { 2 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_deduplicated_forward_reverse_candidates_by_class_and_priority() {
        let forward = [
            Candidate {
                id: 1,
                is_new: false,
            },
            Candidate {
                id: 2,
                is_new: true,
            },
            Candidate {
                id: 3,
                is_new: true,
            },
            Candidate {
                id: 99,
                is_new: true,
            },
        ];
        let reverse = [
            Candidate {
                id: 1,
                is_new: true,
            },
            Candidate {
                id: 2,
                is_new: false,
            },
            Candidate {
                id: 4,
                is_new: false,
            },
            Candidate {
                id: 0,
                is_new: true,
            },
        ];
        let pools = sample_pools(0, 8, &forward, &reverse, 1, 42, 3);
        let repeat = sample_pools(0, 8, &forward, &reverse, 1, 42, 3);
        let reordered_forward = forward.iter().rev().copied().collect::<Vec<_>>();
        let reordered_reverse = reverse.iter().rev().copied().collect::<Vec<_>>();
        let reordered = sample_pools(0, 8, &reordered_forward, &reordered_reverse, 1, 42, 3);
        assert_eq!(pools, repeat);
        assert_eq!(pools, reordered);
        assert_eq!(pools.new.len(), 1);
        assert_eq!(pools.old.len(), 1);
        assert!(!pools.new.contains(&0));
        assert!(!pools.new.contains(&99));
        assert!(pools.new.iter().all(|id| [1, 2, 3].contains(id)));
        assert!(pools.old.iter().all(|id| [4].contains(id)));
    }

    #[test]
    fn local_join_pairs_include_new_new_and_cross_class_once() {
        let pools = SampledPools {
            new: vec![1, 2, 1, 0],
            old: vec![2, 3, 3],
        };
        assert_eq!(local_pairs(&pools, 0, 5), vec![(1, 2), (1, 3), (2, 3)]);
    }

    #[test]
    fn duplicate_centers_retain_local_pair_multiplicity_but_merge_unique_edges() {
        let pools = SampledPools {
            new: vec![0, 2],
            old: vec![],
        };
        assert_eq!(local_pairs(&pools, 1, 4), vec![(0, 2)]);
        assert_eq!(local_pairs(&pools, 3, 4), vec![(0, 2)]);
        let source = vec![vec![], vec![], vec![], vec![]];
        let result = merge_proposals(
            &source,
            &[(0, 2, 5.0), (0, 2, 4.0)],
            &vec![BTreeSet::new(); 4],
            2,
        );
        assert_eq!(result.rows[0].len(), 1);
        assert_eq!(result.rows[0][0].distance, 4.0);
        assert_eq!(result.rows[2][0].distance, 4.0);
        assert_eq!(result.surviving_additions, 2);
    }

    #[test]
    fn final_addition_count_and_flags_ignore_transient_additions() {
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
        let result = merge_proposals(
            &source,
            &[(0, 2, 5.0), (0, 1, 1.0)],
            &vec![BTreeSet::new(); 3],
            1,
        );
        assert_eq!(
            result.rows[0],
            vec![Neighbor {
                id: 1,
                distance: 1.0,
                is_new: false
            }]
        );
        assert_eq!(
            result.rows[1],
            vec![Neighbor {
                id: 0,
                distance: 1.0,
                is_new: false
            }]
        );
        assert_eq!(
            result.rows[2],
            vec![Neighbor {
                id: 0,
                distance: 5.0,
                is_new: true
            }]
        );
        assert_eq!(result.surviving_additions, 1);
    }

    #[test]
    fn sampled_source_new_edges_become_old_but_unsampled_survivors_remain_new() {
        let source = vec![vec![
            Neighbor {
                id: 1,
                distance: 1.0,
                is_new: true,
            },
            Neighbor {
                id: 2,
                distance: 2.0,
                is_new: true,
            },
        ]];
        let result = merge_proposals(&source, &[], &[BTreeSet::from([1])], 2);
        assert!(!result.rows[0][0].is_new);
        assert!(result.rows[0][1].is_new);
        assert_eq!(result.surviving_additions, 0);
    }

    #[test]
    fn reverse_selected_source_new_edge_becomes_old_without_counting_as_addition() {
        let source = vec![
            vec![Neighbor {
                id: 1,
                distance: 1.0,
                is_new: true,
            }],
            vec![],
        ];
        let result = merge_proposals(&source, &[], &[BTreeSet::new(), BTreeSet::from([0])], 1);
        assert!(!result.rows[0][0].is_new);
        assert_eq!(result.surviving_additions, 0);
    }
}
