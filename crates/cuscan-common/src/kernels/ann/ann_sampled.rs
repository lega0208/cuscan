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

use super::ann_forest::{
    ForestLeaves, MAX_CANDIDATES, WorkspaceError, build_forest, forest_leaf_limit, plan_workspace,
    tree_count,
};
use super::orchestration::{
    AnnPlan, Buffer, BufferEntry, BufferRoute, REVERSE_NEW_BIT, ann_buffer_table,
    ann_independent_launch, ann_output, ann_should_stop, reverse_count, reverse_scatter,
    reverse_zero, workspace_error,
};
use crate::error::AnnError;
use burn::{
    backend::{TensorMetadata, cubecl::dtype_to_storage_type},
    tensor::{DType, Shape},
};
use burn_cubecl::tensor::CubeTensor;
use cubecl::prelude::*;

#[cfg(test)]
const PRIORITY_MODULUS: u64 = 2_147_483_647;
// C=60 produced 268.5M mostly masked slots per iteration on the fixed input
// (CPU pair/merge ~5s). A separate per-class cap keeps staging work bounded.
pub(super) const PREFIX_CHUNK: usize = 128;

/// Build bounded forest leaf membership on the host. Distances and graph
/// updates are executed by the shared pair-slot/endpoint pipeline.
/// Device state produced by `forest_seed`: seeded heaps, the forest membership,
/// its I32 leaf row offsets and I64 pair prefix, and the forest pair total.
pub(super) struct ForestSeed {
    pub(super) ids: CubeTensor,
    pub(super) distances: CubeTensor,
    pub(super) flags: CubeTensor,
    pub(super) rows: CubeTensor,
    pub(super) leaf_rows: CubeTensor,
    pub(super) pair_prefix: CubeTensor,
    pub(super) pair_count: usize,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forest_seed(
    data: &CubeTensor,
    allocator: &mut AnnAllocator,
    n: usize,
    features: usize,
    heap_width: usize,
    max_candidates: usize,
    seed: u64,
    profile: bool,
) -> Result<ForestSeed, WorkspaceError> {
    let trees = tree_count(n)?;
    let leaf_size = forest_leaf_limit(n, heap_width);
    let workspace = plan_workspace(n, heap_width, max_candidates, features, trees)?;
    let readback_start = profile.then(std::time::Instant::now);
    let input_bytes = data.client.read_one_unchecked(data.handle.clone());
    if let Some(start) = readback_start {
        eprintln!(
            "ANN_PROFILE forest_readback_ms={:.3} bytes={}",
            start.elapsed().as_secs_f64() * 1000.0,
            input_bytes.len()
        );
    }
    let expected_bytes = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(DType::F32.size()))
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    if input_bytes.len() != expected_bytes {
        return Err(WorkspaceError::InvalidDimensions);
    }
    // View the downloaded bytes as f32 when they are 4-byte aligned; otherwise
    // decode a copy (the planner charges the copy).
    enum FeatureSource<B> {
        Downloaded(B),
        Decoded(Vec<f32>),
    }
    let source = if bytemuck::try_cast_slice::<u8, f32>(&input_bytes).is_ok() {
        FeatureSource::Downloaded(input_bytes)
    } else {
        FeatureSource::Decoded(
            input_bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_ne_bytes(*chunk))
                .collect(),
        )
    };
    let features_host: &[f32] = match &source {
        FeatureSource::Downloaded(bytes) => bytemuck::cast_slice(&bytes[..]),
        FeatureSource::Decoded(values) => values,
    };
    if profile {
        eprintln!(
            "ANN_PROFILE forest_feature_copy_decoded={}",
            matches!(source, FeatureSource::Decoded(_))
        );
    }
    let build_start = profile.then(std::time::Instant::now);
    let forest = build_forest(features_host, n, features, seed, trees, leaf_size)?;
    drop(source);
    if let Some(start) = build_start {
        eprintln!(
            "ANN_PROFILE forest_build_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
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
    // The plan and the built forest must agree exactly; both come from the
    // same midpoint recurrence, so any mismatch is a planner defect.
    assert_eq!(forest.row_ids.len(), workspace.forest_membership_len);
    assert_eq!(forest.leaf_offsets.len() - 1, workspace.forest_leaf_count);
    assert_eq!(forest_pair_slots, workspace.forest_pair_count);
    let heap_len = n
        .checked_mul(heap_width)
        .ok_or(WorkspaceError::ArithmeticOverflow)?;
    let sentinel_id = i32::try_from(n).map_err(|_| WorkspaceError::InvalidDimensions)?;
    let ids = vec![sentinel_id; heap_len];
    // Empty slots carry the same (n, +inf) missing-neighbor contract as exact search.
    let distances = vec![f32::INFINITY; heap_len];
    let flags = vec![0i32; heap_len];
    // Membership offsets stay I32 (bounded by trees * n <= i32::MAX); the pair
    // prefix is I64 so the forest pair total is not limited to i32.
    let mut leaf_rows = Vec::with_capacity(forest.leaf_offsets.len());
    let mut leaf_pair_prefix = Vec::with_capacity(forest.leaf_offsets.len());
    let mut pair_prefix = 0usize;
    for (leaf, &offset) in forest.leaf_offsets.iter().enumerate() {
        leaf_rows.push(i32::try_from(offset).map_err(|_| WorkspaceError::InvalidDimensions)?);
        leaf_pair_prefix
            .push(i64::try_from(pair_prefix).map_err(|_| WorkspaceError::ArithmeticOverflow)?);
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
    let ForestLeaves {
        row_ids: membership,
        leaf_offsets,
        tree_leaf_offsets: _,
    } = forest;
    drop(leaf_offsets);
    let upload_start = profile.then(std::time::Instant::now);
    let rows_tensor = allocator.upload(
        Buffer::ForestRows,
        Shape::new([membership.len()]),
        DType::I32,
        membership,
    );
    debug_assert_eq!(leaf_rows.len(), leaf_pair_prefix.len());
    let leaf_rows_tensor = allocator.upload(
        Buffer::ForestLeafRows,
        Shape::new([leaf_rows.len()]),
        DType::I32,
        leaf_rows,
    );
    let pair_prefix_tensor = allocator.upload(
        Buffer::ForestPairPrefix,
        Shape::new([leaf_pair_prefix.len()]),
        DType::I64,
        leaf_pair_prefix,
    );
    let shape = Shape::new([n, heap_width]);
    let ids = allocator.upload(Buffer::HeapIds, shape.clone(), DType::I32, ids);
    let distances = allocator.upload(Buffer::HeapDistances, shape.clone(), DType::F32, distances);
    let flags = allocator.upload(Buffer::Flags, shape, DType::I32, flags);
    if let Some(start) = upload_start {
        eprintln!(
            "ANN_PROFILE forest_upload_enqueue_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok(ForestSeed {
        ids,
        distances,
        flags,
        rows: rows_tensor,
        leaf_rows: leaf_rows_tensor,
        pair_prefix: pair_prefix_tensor,
        pair_count: forest_pair_slots,
    })
}

pub(super) struct AnnAllocator {
    table: Vec<BufferEntry>,
    allocated: Vec<(Buffer, CubeTensor)>,
    client: cubecl::client::Client,
    device: burn_cubecl::CubeDevice,
    plan: AnnPlan,
}

#[cfg(test)]
thread_local! {
    static ALLOCATION_LOG: std::cell::RefCell<Vec<Buffer>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn memory_failure(
    stage: &str,
    error: &str,
    required_bytes: u64,
    usable_bytes: u64,
    fresh_free: Result<u64, String>,
) -> AnnError {
    let (available_bytes, availability) = match fresh_free {
        Ok(bytes) => (bytes, format!("fresh free {bytes} bytes")),
        Err(error) => (
            0,
            format!(
                "fresh free memory unknown (availability query failed: {error}); available_bytes=0 denotes unknown"
            ),
        ),
    };
    AnnError::Memory {
        required_bytes,
        available_bytes,
        message: format!(
            "ANN {stage} failed: {error}; planned {required_bytes} bytes, {availability}, planned usable {usable_bytes} bytes"
        ),
    }
}

impl AnnAllocator {
    fn memory_error(&self, stage: &str, error: impl std::fmt::Debug) -> AnnError {
        let device =
            burn::tensor::Device::new(burn::backend::DispatchDevice::Cube(self.device.clone()));
        let fresh_free = crate::resolve_memory_budget(&device, None)
            .map(|budget| budget.free_bytes)
            .map_err(|error| error.to_string());
        let required_bytes = self
            .plan
            .memory
            .fixed_peak_bytes
            .saturating_add(self.plan.memory.staging_bytes);
        memory_failure(
            stage,
            &format!("{error:?}"),
            required_bytes,
            self.plan.memory.usable_bytes,
            fresh_free,
        )
    }

    fn entry(&self, key: Buffer, shape: &Shape, dtype: DType) -> &BufferEntry {
        assert_ne!(key, Buffer::Input, "ANN input is externally owned");
        assert!(
            !self.allocated.iter().any(|(buffer, _)| *buffer == key),
            "duplicate ANN allocation {key:?}"
        );
        let mut matches = self.table.iter().filter(|entry| entry.buffer == key);
        let entry = matches.next().expect("missing ANN allocation table entry");
        assert!(
            entry.route != BufferRoute::Cuda || !self.plan.merge_geometry().0,
            "CUDA-only ANN allocation on CPU route: {key:?}"
        );
        assert!(
            matches.next().is_none(),
            "duplicate ANN table entry {key:?}"
        );
        assert_eq!(
            Shape::from(entry.shape.clone()),
            *shape,
            "ANN allocation shape {key:?}"
        );
        assert_eq!(entry.dtype, dtype, "ANN allocation dtype {key:?}");
        assert_eq!(entry.elements(), Some(shape.num_elements()));
        entry
    }

    fn empty(&mut self, key: Buffer, shape: Shape, dtype: DType) -> CubeTensor {
        let bytes = self
            .entry(key, &shape, dtype)
            .bytes()
            .expect("ANN allocation size overflow");
        let handle = self
            .client
            .memory_dedicated_allocation((), |_| self.client.empty(bytes));
        #[cfg(test)]
        ALLOCATION_LOG.with(|log| log.borrow_mut().push(key));
        let tensor = CubeTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            shape,
            handle,
            dtype,
        );
        self.allocated.push((key, tensor.clone()));
        tensor
    }

    fn upload<T: burn::tensor::Element>(
        &mut self,
        key: Buffer,
        shape: Shape,
        dtype: DType,
        values: Vec<T>,
    ) -> CubeTensor {
        self.entry(key, &shape, dtype);
        assert_eq!(
            values.len(),
            shape.num_elements(),
            "ANN upload element count {key:?}"
        );
        let tensor_data = burn::tensor::TensorData::new(values, shape.clone());
        assert_eq!(tensor_data.dtype(), dtype);
        let (bytes, _, _) = tensor_data.into_parts();
        let handle = self
            .client
            .memory_dedicated_allocation((), |_| self.client.create(bytes));
        #[cfg(test)]
        ALLOCATION_LOG.with(|log| log.borrow_mut().push(key));
        let tensor = CubeTensor::new_contiguous(
            self.client.clone(),
            self.device.clone(),
            shape,
            handle,
            dtype,
        );
        self.allocated.push((key, tensor.clone()));
        tensor
    }

    fn check(&self, phase: u8) -> Result<(), AnnError> {
        #[cfg(debug_assertions)]
        self.assert_live_phase(phase);
        for entry in &self.table {
            if entry.buffer != Buffer::Input
                && entry.buffer != Buffer::InputCopy
                && (entry.route == BufferRoute::Both
                    || (entry.route == BufferRoute::Cuda && !self.plan.merge_geometry().0))
                && entry.phases & phase != 0
            {
                assert!(
                    self.allocated.iter().any(|(key, _)| *key == entry.buffer),
                    "missing ANN allocation {:?}",
                    entry.buffer
                );
            }
        }
        let handles = self.allocated.iter().map(|(_, tensor)| &tensor.handle);
        if let Err(error) = cubecl::future::block_on(self.client.sync_buffers(handles)) {
            return Err(self.memory_error("allocation", error));
        }
        Ok(())
    }

    #[cfg(debug_assertions)]
    fn assert_live_phase(&self, phase: u8) {
        for (key, _) in &self.allocated {
            debug_assert!(
                self.table
                    .iter()
                    .any(|entry| entry.buffer == *key && entry.phases & phase != 0),
                "ANN allocation {key:?} retained outside phase {phase}"
            );
        }
    }

    #[cfg(debug_assertions)]
    fn assert_heap_identity(&self, candidates: &CubeTensor) {
        let retained = self
            .allocated
            .iter()
            .find(|(key, _)| *key == Buffer::HeapIds)
            .expect("retained ANN heap IDs");
        debug_assert_eq!(
            retained.1.handle.memory.clone().binding().id(),
            candidates.handle.memory.clone().binding().id()
        );
        debug_assert_eq!(retained.1.handle.service, candidates.handle.service);
        debug_assert_eq!(
            retained.1.handle.offset_start,
            candidates.handle.offset_start
        );
        debug_assert_eq!(retained.1.handle.offset_end, candidates.handle.offset_end);
    }

    fn release_except(&mut self, phase: u8) {
        self.allocated.retain(|(key, _)| {
            self.table
                .iter()
                .any(|entry| entry.buffer == *key && entry.phases & phase != 0)
        });
    }

    fn swap_keys(&mut self, left: Buffer, right: Buffer) {
        for (key, _) in &mut self.allocated {
            if *key == left {
                *key = right;
            } else if *key == right {
                *key = left;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_endpoint_merge(
    client: &cubecl::client::Client,
    plan: &AnnPlan,
    offsets: &CubeTensor,
    endpoint_ids: &CubeTensor,
    endpoint_distances: &CubeTensor,
    ids: &CubeTensor,
    distances: &CubeTensor,
    row_worst_ids: &CubeTensor,
    row_worst_distances: &CubeTensor,
    workspace: &Option<(CubeTensor, CubeTensor)>,
    width: u32,
) {
    let (cpu, count, dim) = plan.merge_geometry();
    let dtypes = [
        dtype_to_storage_type(DType::F32),
        dtype_to_storage_type(DType::I32),
    ];
    if cpu {
        endpoint_merge_cpu::launch(
            client,
            count,
            dim,
            offsets.clone().into_tensor_arg(),
            endpoint_ids.clone().into_tensor_arg(),
            endpoint_distances.clone().into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            row_worst_ids.clone().into_tensor_arg(),
            row_worst_distances.clone().into_tensor_arg(),
            width,
            dtypes,
        );
    } else {
        let (workspace_ids, workspace_distances) =
            workspace.as_ref().expect("CUDA merge workspace");
        endpoint_merge_cuda::launch(
            client,
            count,
            dim,
            offsets.clone().into_tensor_arg(),
            endpoint_ids.clone().into_tensor_arg(),
            endpoint_distances.clone().into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            workspace_ids.clone().into_tensor_arg(),
            workspace_distances.clone().into_tensor_arg(),
            row_worst_ids.clone().into_tensor_arg(),
            row_worst_distances.clone().into_tensor_arg(),
            width,
            dtypes,
        );
    }
}

#[cube(launch)]
fn heap_copy_kernel<I: Int>(
    source: &Tensor<I>,
    destination: &mut Tensor<I>,
    #[define(I)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS_X as usize;
    if index < source.shape(0) * source.shape(1) {
        destination[index] = source[index];
    }
}

/// Production sampled NN-descent orchestration. Source IDs and flags stay frozen
/// for each iteration; merge tiles update a working ID heap and one distance heap.
/// Heap-distance launches are forest/refinement merge, random fill, the row-worst
/// refresh and output. Pair kernels read only the row-worst thresholds, which the
/// merge rewrites for every row it touches and the refresh rebuilds after the
/// initial upload and after random fill (the only heap writers outside the merge).
/// The optional survival diagnostic additionally reads the current heap before a
/// tile; neither sampled-pool construction nor flag finalization reads distances.
pub(super) fn approximate_neighbors(
    data: CubeTensor,
    k: usize,
    seed: u64,
    max_iterations: u32,
    squared: bool,
    plan: AnnPlan,
) -> Result<(CubeTensor, CubeTensor), AnnError> {
    let profile = std::env::var_os("CUSCAN_ANN_PROFILE").is_some();
    let init_start = std::time::Instant::now();
    if data.meta.is_tiled() {
        return Err(AnnError::InvalidInput(
            "ANN requires untiled input; untile it first".into(),
        ));
    }
    let [n, features] = data.shape().dims::<2>();
    assert_eq!(data.dtype, DType::F32);
    let client = data.client.clone();
    let width = plan.host.width;
    let capacity = plan.host.capacity;
    // `sampled_pool_kernel` sizes its local priority array for this bound.
    assert!(
        capacity <= MAX_CANDIDATES,
        "ANN sample capacity exceeds MAX_CANDIDATES"
    );
    let workspace = plan.host.workspace;
    let table = ann_buffer_table(n, features, k, width, capacity, &workspace, plan.tile_slots)
        .ok_or_else(|| AnnError::Overflow("ANN allocation table overflow".into()))?;
    let mut allocator = AnnAllocator {
        table,
        allocated: Vec::new(),
        client: client.clone(),
        device: data.device.clone(),
        plan,
    };
    cubecl::future::block_on(client.sync_buffers([&data.handle]))
        .map_err(|error| allocator.memory_error("input buffer synchronization", error))?;
    let data = if data.is_contiguous() {
        data
    } else {
        let copy = allocator.empty(Buffer::InputCopy, Shape::new([n, features]), DType::F32);
        allocator.check(1)?;
        cubecl::std::tensor::copy_into(
            &client,
            data.binding(),
            copy.clone().binding(),
            dtype_to_storage_type(DType::F32),
        );
        allocator.check(1)?;
        copy
    };
    let n_u32 = u32::try_from(n).expect("ANN row count validated by plan");
    let width_u32 = u32::try_from(width).expect("ANN width validated by plan");
    let cap_u32 = u32::try_from(capacity).expect("ANN sample capacity validated by plan");
    let pair_slots = plan.tile_slots;
    let pair_elements = pair_slots;
    let endpoint_elements = pair_slots
        .checked_mul(2)
        .expect("ANN endpoint tile validated by plan");
    let max_total_slots = n
        .checked_mul(workspace.pair_slots_per_center)
        .expect("ANN pair slot count overflows usize");
    let ForestSeed {
        ids: mut candidates,
        distances,
        mut flags,
        rows: forest_rows,
        leaf_rows: forest_leaf_rows,
        pair_prefix: forest_pair_prefix,
        pair_count: forest_pairs,
    } = forest_seed(
        &data,
        &mut allocator,
        n,
        features,
        width,
        capacity,
        seed,
        profile,
    )
    .map_err(|error| workspace_error("RP forest initialization", error))?;
    let pair_left = allocator.empty(Buffer::PairLeft, Shape::new([pair_elements]), DType::I32);
    let pair_right = allocator.empty(Buffer::PairRight, Shape::new([pair_elements]), DType::I32);
    let pair_distances = allocator.empty(
        Buffer::PairDistances,
        Shape::new([pair_elements]),
        DType::F32,
    );
    let pair_valid = allocator.empty(Buffer::PairValid, Shape::new([pair_elements]), DType::I32);
    let row_worst_ids = allocator.empty(Buffer::RowWorstIds, Shape::new([n]), DType::I32);
    let row_worst_distances =
        allocator.empty(Buffer::RowWorstDistances, Shape::new([n]), DType::F32);
    let endpoint_counts = allocator.empty(Buffer::EndpointCounts, Shape::new([n]), DType::I32);
    let endpoint_offsets =
        allocator.empty(Buffer::EndpointOffsets, Shape::new([n + 1]), DType::I32);
    let endpoint_cursors = allocator.empty(Buffer::EndpointCursors, Shape::new([n]), DType::I32);
    let endpoint_ids = allocator.empty(
        Buffer::EndpointIds,
        Shape::new([endpoint_elements]),
        DType::I32,
    );
    let endpoint_distances = allocator.empty(
        Buffer::EndpointDistances,
        Shape::new([endpoint_elements]),
        DType::F32,
    );
    let merge_workspace = if plan.merge_geometry().0 {
        None
    } else {
        Some((
            allocator.empty(
                Buffer::MergeWorkspaceIds,
                Shape::new([n, width]),
                DType::I32,
            ),
            allocator.empty(
                Buffer::MergeWorkspaceDistances,
                Shape::new([n, width]),
                DType::F32,
            ),
        ))
    };
    let chunks = n.div_ceil(PREFIX_CHUNK);
    let chunk_sums = allocator.empty(Buffer::ChunkSums, Shape::new([chunks]), DType::I32);
    let prefix_i32 = |values: &CubeTensor, offsets: &CubeTensor, cursors: &CubeTensor| {
        exclusive_prefix(
            &client,
            values,
            offsets,
            cursors,
            &chunk_sums,
            DType::I32,
            true,
        )
    };
    let (helper_count, helper_dim) = ann_independent_launch(&client, n_u32)?;
    let (row_count, row_dim) = ann_independent_launch(&client, n_u32)?;
    let refresh_row_worst = |ids: &CubeTensor| {
        row_worst_refresh_kernel::launch(
            &client,
            row_count.clone(),
            row_dim,
            ids.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            row_worst_ids.clone().into_tensor_arg(),
            row_worst_distances.clone().into_tensor_arg(),
            width_u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
    };
    let profile_counts =
        profile.then(|| allocator.empty(Buffer::ProfileCounts, Shape::new([1]), DType::I32));
    let mut forest_records = (0u64, 0u64);
    let mut join_records = (0u64, 0u64);
    // Per-tile readbacks and atomics perturb timings: these are diagnostic runs,
    // not representative ANN performance measurements.
    #[cfg(feature = "test-support")]
    let survival_counts = (profile && std::env::var_os("CUSCAN_ANN_SURVIVAL_PROFILE").is_some())
        .then(|| allocator.empty(Buffer::SurvivalCounts, Shape::new([2]), DType::I32));
    #[cfg(feature = "test-support")]
    let mut forest_survival = (0u64, 0u64);
    #[cfg(feature = "test-support")]
    let mut join_survival = (0u64, 0u64);
    let mut forest_offset = 0usize;
    allocator.check(2)?;
    // The heap holds the initial `(n, +inf)` rows: every row starts not full.
    refresh_row_worst(&candidates);
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
            forest_leaf_rows.clone().into_tensor_arg(),
            forest_pair_prefix.clone().into_tensor_arg(),
            row_worst_ids.clone().into_tensor_arg(),
            row_worst_distances.clone().into_tensor_arg(),
            pair_left.clone().into_tensor_arg(),
            pair_right.clone().into_tensor_arg(),
            pair_distances.clone().into_tensor_arg(),
            pair_valid.clone().into_tensor_arg(),
            u64::try_from(forest_offset).expect("ANN forest offset exceeds u64"),
            slots_u32,
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
                dtype_to_storage_type(DType::I64),
            ],
        );
        #[cfg(feature = "test-support")]
        if let Some(counts) = &survival_counts {
            let (produced, predicted) = profile_tile_survival(
                &client,
                counts,
                &pair_left,
                &pair_right,
                &pair_distances,
                &pair_valid,
                &candidates,
                &distances,
                n_u32,
                width_u32,
                slots_u32,
            )?;
            forest_survival.0 += produced;
            forest_survival.1 += predicted;
        }
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
        prefix_i32(&endpoint_counts, &endpoint_offsets, &endpoint_cursors)?;
        if let Some(counts) = &profile_counts {
            let (produced, kept) = profile_tile_records(
                &client,
                counts,
                &pair_valid,
                &endpoint_offsets,
                n,
                slots_u32,
            )?;
            forest_records.0 += produced;
            forest_records.1 += kept;
        }
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
        launch_endpoint_merge(
            &client,
            &plan,
            &endpoint_offsets,
            &endpoint_ids,
            &endpoint_distances,
            &candidates,
            &distances,
            &row_worst_ids,
            &row_worst_distances,
            &merge_workspace,
            width_u32,
        );
        forest_offset += slots;
    }
    #[cfg(feature = "test-support")]
    if survival_counts.is_some() {
        eprintln!(
            "ANN_PROFILE survival phase=forest produced={} predicted_surviving={} (tile-start predicate; timing-sensitive)",
            forest_survival.0, forest_survival.1
        );
    }
    if profile {
        eprintln!(
            "ANN_PROFILE records phase=forest produced={} kept={}",
            forest_records.0, forest_records.1
        );
    }
    drop(forest_rows);
    drop(forest_leaf_rows);
    drop(forest_pair_prefix);
    allocator.release_except(4);
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
    // Random fill is the only heap writer outside the merge.
    refresh_row_worst(&candidates);
    if profile {
        let _ = client.read_one_unchecked(candidates.handle.clone());
        eprintln!(
            "ANN_PROFILE init_ms={:.3} forest_pairs={forest_pairs} forest_tiles={} H={width} C={capacity} tile_slots={} fixed_peak_bytes={} staging_bytes={} usable_bytes={}",
            init_start.elapsed().as_secs_f64() * 1000.0,
            forest_pairs.div_ceil(pair_slots),
            plan.tile_slots,
            plan.memory.fixed_peak_bytes,
            plan.memory.staging_bytes,
            plan.memory.usable_bytes
        );
    }
    let reverse_counts = allocator.empty(Buffer::ReverseCounts, Shape::new([n]), DType::I32);
    let reverse_offsets = allocator.empty(Buffer::ReverseOffsets, Shape::new([n + 1]), DType::I32);
    let reverse_cursors = allocator.empty(Buffer::ReverseCursors, Shape::new([n]), DType::I32);
    let reverse_sources =
        allocator.empty(Buffer::ReverseSources, Shape::new([n * width]), DType::I32);
    let sampled_ids = allocator.empty(
        Buffer::SampledIds,
        Shape::new([n, capacity * 2]),
        DType::I32,
    );
    let mut working_ids = allocator.empty(Buffer::WorkingIds, Shape::new([n, width]), DType::I32);
    let mut next_flags = allocator.empty(Buffer::NextFlags, Shape::new([n, width]), DType::I32);
    let new_count = allocator.empty(Buffer::NewCount, Shape::new([1]), DType::I32);
    let pool_sizes = allocator.empty(Buffer::PoolSizes, Shape::new([n * 2]), DType::I32);
    let pair_counts = allocator.empty(Buffer::PairCounts, Shape::new([n]), DType::I64);
    let center_offsets = allocator.empty(Buffer::CenterOffsets, Shape::new([n + 1]), DType::I64);
    let join_chunk_sums = allocator.empty(Buffer::JoinChunkSums, Shape::new([chunks]), DType::I64);
    let join_cursor_placeholder =
        allocator.empty(Buffer::JoinCursorPlaceholder, Shape::new([1]), DType::I64);
    let join_prefix = || {
        exclusive_prefix(
            &client,
            &pair_counts,
            &center_offsets,
            &join_cursor_placeholder,
            &join_chunk_sums,
            DType::I64,
            false,
        )
    };
    allocator.check(4)?;
    let mut stop_reason = "max_iterations";
    let mut iterations_run = 0u32;
    let mut last_new_count: Option<i32> = None;
    for iteration in 0..max_iterations {
        let pool_start = std::time::Instant::now();
        let source_ids = candidates.clone();
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
        prefix_i32(&reverse_counts, &reverse_offsets, &reverse_cursors)?;
        reverse_scatter::launch(
            &client,
            reverse_count_units,
            reverse_dim,
            source_ids.clone().into_tensor_arg(),
            source_flags.clone().into_tensor_arg(),
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
            [
                dtype_to_storage_type(DType::I32),
                dtype_to_storage_type(DType::I64),
            ],
        );
        join_prefix()?;
        let total_bytes = client.read_one_unchecked(
            center_offsets
                .handle
                .clone()
                .offset_start((n * DType::I64.size()) as u64),
        );
        let total_slots =
            i64::from_ne_bytes(total_bytes[..8].try_into().expect("ANN pair total is i64"));
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
        if total_slots > 0 {
            let elements = u32::try_from(n * width).expect("ANN heap size validated by plan");
            let (copy_count, copy_dim) = ann_independent_launch(&client, elements)?;
            heap_copy_kernel::launch(
                &client,
                copy_count,
                copy_dim,
                source_ids.clone().into_tensor_arg(),
                working_ids.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
        }
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
                row_worst_ids.clone().into_tensor_arg(),
                row_worst_distances.clone().into_tensor_arg(),
                pair_left.clone().into_tensor_arg(),
                pair_right.clone().into_tensor_arg(),
                pair_distances.clone().into_tensor_arg(),
                pair_valid.clone().into_tensor_arg(),
                u64::try_from(slot_offset).expect("ANN pair offset exceeds u64"),
                slots_u32,
                cap_u32,
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                    dtype_to_storage_type(DType::I64),
                ],
            );
            #[cfg(feature = "test-support")]
            if let Some(counts) = &survival_counts {
                let (produced, predicted) = profile_tile_survival(
                    &client,
                    counts,
                    &pair_left,
                    &pair_right,
                    &pair_distances,
                    &pair_valid,
                    &working_ids,
                    &distances,
                    n_u32,
                    width_u32,
                    slots_u32,
                )?;
                join_survival.0 += produced;
                join_survival.1 += predicted;
            }
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
            prefix_i32(&endpoint_counts, &endpoint_offsets, &endpoint_cursors)?;
            if let Some(counts) = &profile_counts {
                let (produced, kept) = profile_tile_records(
                    &client,
                    counts,
                    &pair_valid,
                    &endpoint_offsets,
                    n,
                    slots_u32,
                )?;
                join_records.0 += produced;
                join_records.1 += kept;
            }
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
            launch_endpoint_merge(
                &client,
                &plan,
                &endpoint_offsets,
                &endpoint_ids,
                &endpoint_distances,
                &working_ids,
                &distances,
                &row_worst_ids,
                &row_worst_distances,
                &merge_workspace,
                width_u32,
            );
            slot_offset += slots;
        }
        if profile {
            let _ = client.read_one_unchecked(if total_slots > 0 {
                working_ids.handle.clone()
            } else {
                candidates.handle.clone()
            });
            eprintln!(
                "ANN_PROFILE iteration={iteration} pair_merge_ms={:.3} pair_slots={total_slots} pair_tiles={}",
                join_start.elapsed().as_secs_f64() * 1000.0,
                total_slots.div_ceil(pair_slots)
            );
        }
        if total_slots > 0 {
            std::mem::swap(&mut candidates, &mut working_ids);
            allocator.swap_keys(Buffer::HeapIds, Buffer::WorkingIds);
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
        iterations_run = iteration + 1;
        if iteration > 0 || profile {
            let bytes = client.read_one_unchecked(new_count.handle.clone());
            let count = i32::from_ne_bytes(bytes[..4].try_into().expect("ANN counter is i32"));
            last_new_count = Some(count);
            if iteration > 0 && ann_should_stop(count, n, width) {
                stop_reason = "converged";
                break;
            }
        }
    }
    #[cfg(feature = "test-support")]
    if survival_counts.is_some() {
        eprintln!(
            "ANN_PROFILE survival phase=join produced={} predicted_surviving={} (tile-start predicate; timing-sensitive)",
            join_survival.0, join_survival.1
        );
    }
    if profile {
        eprintln!(
            "ANN_PROFILE records phase=join produced={} kept={}",
            join_records.0, join_records.1
        );
        eprintln!(
            "ANN_PROFILE stop reason={stop_reason} iterations={iterations_run} max_iterations={max_iterations} final_new_count={last_new_count:?} capacity={capacity} width={width}"
        );
    }
    drop(pair_left);
    drop(pair_right);
    drop(pair_distances);
    drop(pair_valid);
    drop(row_worst_ids);
    drop(row_worst_distances);
    drop(profile_counts);
    drop(endpoint_counts);
    drop(endpoint_offsets);
    drop(endpoint_cursors);
    drop(endpoint_ids);
    drop(endpoint_distances);
    drop(merge_workspace);
    drop(reverse_counts);
    drop(reverse_offsets);
    drop(reverse_cursors);
    drop(reverse_sources);
    drop(sampled_ids);
    drop(working_ids);
    drop(flags);
    drop(next_flags);
    drop(new_count);
    drop(pool_sizes);
    drop(pair_counts);
    drop(center_offsets);
    drop(chunk_sums);
    drop(join_chunk_sums);
    drop(join_cursor_placeholder);
    #[cfg(feature = "test-support")]
    drop(survival_counts);
    allocator.release_except(8);
    #[cfg(debug_assertions)]
    allocator.assert_heap_identity(&candidates);
    let output_indices = allocator.empty(Buffer::OutputIds, Shape::new([n, k]), DType::I32);
    let output_distances = allocator.empty(Buffer::OutputDistances, Shape::new([n, k]), DType::F32);
    allocator.check(8)?;
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

/// Parallel exclusive prefix of `values[0..len]` into `offsets[0..=len]` in `dtype`,
/// seeding scatter cursors only when `seed_cursors` is set. Callers own the
/// `ceil(len/128)` chunk buffer of the same dtype.
fn exclusive_prefix(
    client: &cubecl::client::Client,
    values: &CubeTensor,
    offsets: &CubeTensor,
    cursors: &CubeTensor,
    chunk_sums: &CubeTensor,
    dtype: DType,
    seed_cursors: bool,
) -> Result<(), AnnError> {
    let len = values.shape()[0];
    debug_assert!(
        [values, offsets, cursors, chunk_sums]
            .iter()
            .all(|tensor| tensor.dtype == dtype),
        "ANN prefix operands must share the prefix dtype"
    );
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
        dtype_to_storage_type(dtype),
    );
    chunk_prefix_kernel::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        chunk_sums.clone().into_tensor_arg(),
        dtype_to_storage_type(dtype),
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
        seed_cursors,
        dtype_to_storage_type(dtype),
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

/// Whether `id` is among the `len` valid members of pool `list`. Read-only; the
/// pools are mutable only because the kernel holds them that way.
#[cube]
fn pool_contains<I: Int>(
    ids: &mut Tensor<I>,
    row_base: usize,
    capacity: usize,
    list: usize,
    id: I,
    len: usize,
) -> bool {
    let base = row_base + list * capacity;
    let mut found = false;
    for slot in 0..len {
        if ids[base + slot] == id {
            found = true;
        }
    }
    found
}

/// Insert an id absent from a priority-sorted pool holding `len` valid members
/// and return the new length. The pool keeps the `capacity` lowest
/// `(priority, id)` members; `priorities` is the kernel-local mirror of both
/// pools (`list * capacity + slot`), not device memory.
#[cube]
fn sampled_insert<I: Int>(
    ids: &mut Tensor<I>,
    priorities: &mut Array<I>,
    row_base: usize,
    capacity: usize,
    list: usize,
    id: I,
    priority: I,
    len: usize,
) -> usize {
    let base = row_base + list * capacity;
    let local = list * capacity;
    let mut accepted = true;
    if len == capacity {
        let last = capacity - 1;
        let last_priority = priorities[local + last];
        accepted = priority < last_priority || (priority == last_priority && id < ids[base + last]);
    }
    let mut result = len;
    if accepted {
        let start = if len < capacity { len } else { capacity - 1 };
        let mut insert_at = 0usize;
        let mut placed = false;
        for step in 0..start {
            if !placed {
                let previous = start - step - 1;
                let previous_id = ids[base + previous];
                let previous_priority = priorities[local + previous];
                if priority < previous_priority
                    || (priority == previous_priority && id < previous_id)
                {
                    ids[base + previous + 1] = previous_id;
                    priorities[local + previous + 1] = previous_priority;
                } else {
                    insert_at = start - step;
                    placed = true;
                }
            }
        }
        ids[base + insert_at] = id;
        priorities[local + insert_at] = priority;
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
                // +inf is a real overflowed squared distance; NaN and negatives are rejected.
                let distance_valid = squared >= F::new(0.0f32);
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
            distances[base + slot] = F::new(f32::INFINITY);
            flags[base + slot] = I::cast_from(0i32);
            slot += 1;
        }
    }
}

/// Build bounded priority-sampled new and old pools from frozen forward and
/// reverse adjacency. New-class membership takes precedence across both
/// directions; reverse CSR order does not influence the retained sample.
///
/// `reverse_sources` entries carry the source id with [`REVERSE_NEW_BIT`] set when
/// the source's edge to this row is new (a source row holds each valid target at
/// most once, so one entry describes the whole edge). While no distinct new id has
/// been offered to a full new pool, that pool holds the complete new set and
/// answers old-class exclusion; after an overflow the exact scan is used.
/// Pool priorities live in a kernel-local array sized for `capacity <=
/// MAX_CANDIDATES`.
#[cube(launch)]
pub(super) fn sampled_pool_kernel<I: Int>(
    candidates: &Tensor<I>,
    retained: &Tensor<I>,
    reverse_offsets: &Tensor<I>,
    reverse_sources: &Tensor<I>,
    sampled_ids: &mut Tensor<I>,
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
        for slot in 0..(capacity * 2) {
            sampled_ids[output_base + slot] = sentinel_id;
        }
        let mut priorities = Array::<I>::new(2 * MAX_CANDIDATES);
        let input_base = row * width;
        let mut new_len = 0usize;
        let mut old_len = 0usize;
        let mut overflow = false;
        // New forward edges.
        for slot in 0..width {
            if retained[input_base + slot] == I::cast_from(0i32) {
                let candidate = u32::cast_from(candidates[input_base + slot]) as usize;
                if candidate < n && candidate != row {
                    let id = I::cast_from(candidate as i32);
                    if !pool_contains(sampled_ids, output_base, capacity, 0, id, new_len) {
                        if new_len == capacity {
                            overflow = true;
                        }
                        new_len = sampled_insert(
                            sampled_ids,
                            &mut priorities,
                            output_base,
                            capacity,
                            0,
                            id,
                            candidate_priority::<I>(
                                row,
                                u32::cast_from(id) as usize,
                                seed,
                                iteration,
                                1,
                            ),
                            new_len,
                        );
                    }
                }
            }
        }
        // New reverse edges.
        let reverse_start = u32::cast_from(reverse_offsets[row]) as usize;
        let reverse_end = u32::cast_from(reverse_offsets[row + 1]) as usize;
        for position in reverse_start..reverse_end {
            let entry = u32::cast_from(reverse_sources[position]);
            if entry & REVERSE_NEW_BIT != 0 {
                let id = I::cast_from((entry & (REVERSE_NEW_BIT - 1)) as i32);
                if !pool_contains(sampled_ids, output_base, capacity, 0, id, new_len) {
                    if new_len == capacity {
                        overflow = true;
                    }
                    new_len = sampled_insert(
                        sampled_ids,
                        &mut priorities,
                        output_base,
                        capacity,
                        0,
                        id,
                        candidate_priority::<I>(
                            row,
                            u32::cast_from(id) as usize,
                            seed,
                            iteration,
                            1,
                        ),
                        new_len,
                    );
                }
            }
        }
        // Old forward edges, excluding any id that occurs in the new class.
        for slot in 0..width {
            if retained[input_base + slot] != I::cast_from(0i32) {
                let candidate = u32::cast_from(candidates[input_base + slot]) as usize;
                if candidate < n && candidate != row {
                    let id = I::cast_from(candidate as i32);
                    let is_new = if overflow {
                        has_new_candidate(candidates, retained, row, id, n, width)
                    } else {
                        pool_contains(sampled_ids, output_base, capacity, 0, id, new_len)
                    };
                    if !is_new && !pool_contains(sampled_ids, output_base, capacity, 1, id, old_len)
                    {
                        old_len = sampled_insert(
                            sampled_ids,
                            &mut priorities,
                            output_base,
                            capacity,
                            1,
                            id,
                            candidate_priority::<I>(
                                row,
                                u32::cast_from(id) as usize,
                                seed,
                                iteration,
                                2,
                            ),
                            old_len,
                        );
                    }
                }
            }
        }
        // Old reverse edges.
        for position in reverse_start..reverse_end {
            let entry = u32::cast_from(reverse_sources[position]);
            if entry & REVERSE_NEW_BIT == 0 {
                let id = I::cast_from(entry as i32);
                let is_new = if overflow {
                    has_new_candidate(candidates, retained, row, id, n, width)
                } else {
                    pool_contains(sampled_ids, output_base, capacity, 0, id, new_len)
                };
                if !is_new && !pool_contains(sampled_ids, output_base, capacity, 1, id, old_len) {
                    old_len = sampled_insert(
                        sampled_ids,
                        &mut priorities,
                        output_base,
                        capacity,
                        1,
                        id,
                        candidate_priority::<I>(
                            row,
                            u32::cast_from(id) as usize,
                            seed,
                            iteration,
                            2,
                        ),
                        old_len,
                    );
                }
            }
        }
    }
}

#[cube(launch)]
pub(super) fn row_worst_refresh_kernel<F: Float, I: Int>(
    ids: &Tensor<I>,
    distances: &Tensor<F>,
    row_worst_ids: &mut Tensor<I>,
    row_worst_distances: &mut Tensor<F>,
    width: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    if row < ids.shape(0) {
        let worst = row * width as usize + width as usize - 1;
        row_worst_ids[row] = ids[worst];
        row_worst_distances[row] = distances[worst];
    }
}

/// `pair_valid` bit: the record for `left` (`candidate = right`) is kept.
pub(super) const KEEP_LEFT: i32 = 1;
/// `pair_valid` bit: the record for `right` (`candidate = left`) is kept.
pub(super) const KEEP_RIGHT: i32 = 2;
/// `pair_valid` bit: the pair produced two records (a valid pair with a
/// non-negative, non-NaN distance), whether or not either endpoint is kept.
/// Count and scatter ignore it; profiling counters and the survival probe read it.
pub(super) const PRODUCED: i32 = 4;

/// Whether `row` keeps the record `(distance, candidate)` against its tile-start
/// threshold. A non-full row (worst id `>= n`) keeps every record; otherwise the
/// record must be lexicographically below `(worst distance, worst id)`, the same
/// predicate as the merge's own filter. Thresholds only tighten, so a rejected
/// record would also be rejected by the merge.
#[cube]
fn endpoint_passes<F: Float, I: Int>(
    row: usize,
    candidate: usize,
    distance: F,
    n: usize,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
) -> bool {
    let worst_id = u32::cast_from(row_worst_ids[row]) as usize;
    let worst_distance = row_worst_distances[row];
    worst_id >= n
        || distance < worst_distance
        || (distance == worst_distance && candidate < worst_id)
}

/// The 2-bit endpoint mask plus [`PRODUCED`] for a pair. Invalid, self and
/// negative/NaN-distance pairs produce no records (mask 0), as the merge rejects
/// them; +inf and signed zero go through the ordinary comparisons.
#[cube]
fn endpoint_mask<F: Float, I: Int>(
    left: usize,
    right: usize,
    distance: F,
    n: usize,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
) -> i32 {
    let mut mask = 0i32;
    if left < n && right < n && left != right && distance >= F::new(0.0f32) {
        mask = PRODUCED;
        if endpoint_passes::<F, I>(left, right, distance, n, row_worst_ids, row_worst_distances) {
            mask += KEEP_LEFT;
        }
        if endpoint_passes::<F, I>(right, left, distance, n, row_worst_ids, row_worst_distances) {
            mask += KEEP_RIGHT;
        }
    }
    mask
}

/// Test access to [`endpoint_mask`] with arbitrary pair distances.
#[cfg(test)]
#[cube(launch)]
pub(super) fn endpoint_mask_probe<F: Float, I: Int>(
    left: &Tensor<I>,
    right: &Tensor<I>,
    distances: &Tensor<F>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    masks: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < masks.shape(0) {
        masks[slot] = I::cast_from(endpoint_mask::<F, I>(
            u32::cast_from(left[slot]) as usize,
            u32::cast_from(right[slot]) as usize,
            distances[slot],
            row_worst_ids.shape(0),
            row_worst_ids,
            row_worst_distances,
        ));
    }
}

#[cube(launch)]
///
/// `leaf_rows[leaf..=leaf+1]` bound a leaf's membership (I32); `pair_prefix` is the
/// 64-bit exclusive within-leaf pair prefix with its total as the last entry. Both
/// tables have `leaves + 1` entries. `slot_offset` is the 64-bit tile start.
/// `local_pair` narrows to u32 after subtraction: it is below the leaf's pair count,
/// which `forest_totals` bounds by `i32::MAX`.
pub(super) fn forest_pair_slots_kernel<F: Float, I: Int, P: Int>(
    data: &Tensor<F>,
    forest_rows: &Tensor<I>,
    leaf_rows: &Tensor<I>,
    pair_prefix: &Tensor<P>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    slot_offset: u64,
    slots: u32,
    #[define(F, I, P)] _dtypes: [ElemType; 3],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize {
        let global_pair = P::cast_from(slot_offset) + P::cast_from(slot as u32);
        let leaf_count = pair_prefix.shape(0) - 1;
        let mut leaf = 0usize;
        let mut upper = leaf_count;
        while upper - leaf > 1 {
            let middle = leaf + (upper - leaf) / 2;
            if pair_prefix[middle] <= global_pair {
                leaf = middle;
            } else {
                upper = middle;
            }
        }
        let found = global_pair < pair_prefix[leaf_count];
        if found {
            let mut local_pair = u32::cast_from(global_pair - pair_prefix[leaf]) as usize;
            let start = u32::cast_from(leaf_rows[leaf]) as usize;
            let end = u32::cast_from(leaf_rows[leaf + 1]) as usize;
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
            let mask = endpoint_mask::<F, I>(
                left,
                right,
                squared,
                data.shape(0),
                row_worst_ids,
                row_worst_distances,
            );
            pair_left[slot] = I::cast_from(left as i32);
            pair_right[slot] = I::cast_from(right as i32);
            pair_distances[slot] = squared;
            pair_valid[slot] = I::cast_from(mask);
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
pub(super) fn pool_pair_count_kernel<I: Int, P: Int>(
    sampled_ids: &Tensor<I>,
    pool_sizes: &mut Tensor<I>,
    pair_counts: &mut Tensor<P>,
    n: u32,
    capacity: u32,
    #[define(I, P)] _dtypes: [ElemType; 2],
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
        pair_counts[row] = P::cast_from(pairs as u32);
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

/// Write `offsets[0..=len]` (exclusive prefix plus total). With `seed_cursors`,
/// also copy the offsets into `cursors`; otherwise `cursors` is never touched.
#[cube(launch)]
pub(super) fn chunk_scatter_kernel<I: Int>(
    values: &Tensor<I>,
    chunk_offsets: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<I>,
    chunk: u32,
    #[comptime] seed_cursors: bool,
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
            if comptime![seed_cursors] {
                cursors[position] = running;
            }
            running += values[position];
        }
        if end == len {
            offsets[len] = running;
        }
    }
}

/// Emit compact local-join slots `[slot_offset, slot_offset+slots)` using the
/// per-center exclusive pair offsets: new×new `i<j` first, then new×old.
///
/// `center_offsets` is the 64-bit exclusive prefix (`[n+1]`, total last) and
/// `slot_offset` the 64-bit tile start. `local` narrows to u32 after subtraction:
/// it is below `pair_counts[center] <= C(C-1)/2 + C^2` (5370 at C=60).
#[cube(launch)]
pub(super) fn pair_slots_kernel<F: Float, I: Int, P: Int>(
    data: &Tensor<F>,
    sampled_ids: &Tensor<I>,
    pool_sizes: &Tensor<I>,
    center_offsets: &Tensor<P>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    slot_offset: u64,
    slots: u32,
    capacity: u32,
    #[define(F, I, P)] _dtypes: [ElemType; 3],
) {
    let slot = ABSOLUTE_POS_X as usize;
    let capacity = capacity as usize;
    if slot < slots as usize {
        let n = data.shape(0);
        let features = data.shape(1);
        let flattened = P::cast_from(slot_offset) + P::cast_from(slot as u32);
        // Largest center with center_offsets[center] <= flattened.
        let mut low = 0usize;
        let mut high = center_offsets.shape(0) - 1;
        while high - low > 1 {
            let middle = (low + high) / 2;
            if center_offsets[middle] <= flattened {
                low = middle;
            } else {
                high = middle;
            }
        }
        let center = low;
        let mut local = u32::cast_from(flattened - center_offsets[center]) as usize;
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
        let mask = endpoint_mask::<F, I>(
            left_id,
            right_id,
            squared,
            n,
            row_worst_ids,
            row_worst_distances,
        );
        pair_valid[slot] = I::cast_from(mask);
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
    if slot < slots as usize {
        let mask = i32::cast_from(pair_valid[slot]);
        if mask & KEEP_LEFT != 0 {
            let left = u32::cast_from(pair_left[slot]) as usize;
            endpoint_counts[left].fetch_add(I::cast_from(1i32));
        }
        if mask & KEEP_RIGHT != 0 {
            let right = u32::cast_from(pair_right[slot]) as usize;
            endpoint_counts[right].fetch_add(I::cast_from(1i32));
        }
    }
}

// This probe observes the heap before the tile merges, not the progressively
// tightened heap within the merge. Passing the predicate is not insertion.
#[cfg(feature = "test-support")]
#[cube(launch)]
fn survival_zero<I: Int>(counts: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    counts[0] = I::cast_from(0i32);
    counts[1] = I::cast_from(0i32);
}

#[cfg(feature = "test-support")]
#[cube(launch)]
fn survival_probe<F: Float, I: Int>(
    left: &Tensor<I>,
    right: &Tensor<I>,
    distances: &Tensor<F>,
    valid: &Tensor<I>,
    heap_ids: &Tensor<I>,
    heap_distances: &Tensor<F>,
    counts: &mut Tensor<Atomic<I>>,
    n: u32,
    width: u32,
    slots: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let slot = ABSOLUTE_POS_X as usize;
    if slot < slots as usize && i32::cast_from(valid[slot]) & PRODUCED != 0 {
        let l = u32::cast_from(left[slot]) as usize;
        let r = u32::cast_from(right[slot]) as usize;
        let d = distances[slot];
        let l_worst = l * width as usize + width as usize - 1;
        let r_worst = r * width as usize + width as usize - 1;
        let l_id = u32::cast_from(heap_ids[l_worst]) as usize;
        let r_id = u32::cast_from(heap_ids[r_worst]) as usize;
        let l_distance = heap_distances[l_worst];
        let r_distance = heap_distances[r_worst];
        let l_pass = l_id >= n as usize || d < l_distance || (d == l_distance && r < l_id);
        let r_pass = r_id >= n as usize || d < r_distance || (d == r_distance && l < r_id);
        counts[0].fetch_add(I::cast_from(2i32));
        if l_pass {
            counts[1].fetch_add(I::cast_from(1i32));
        }
        if r_pass {
            counts[1].fetch_add(I::cast_from(1i32));
        }
    }
}

#[cfg(feature = "test-support")]
fn profile_tile_survival(
    client: &cubecl::client::Client,
    counts: &CubeTensor,
    left: &CubeTensor,
    right: &CubeTensor,
    distances: &CubeTensor,
    valid: &CubeTensor,
    heap_ids: &CubeTensor,
    heap_distances: &CubeTensor,
    n: u32,
    width: u32,
    slots: u32,
) -> Result<(u64, u64), AnnError> {
    survival_zero::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        counts.clone().into_tensor_arg(),
        dtype_to_storage_type(DType::I32),
    );
    let (count, dim) = ann_independent_launch(client, slots)?;
    survival_probe::launch(
        client,
        count,
        dim,
        left.clone().into_tensor_arg(),
        right.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        valid.clone().into_tensor_arg(),
        heap_ids.clone().into_tensor_arg(),
        heap_distances.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        n,
        width,
        slots,
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    let bytes = client.read_one_unchecked(counts.handle.clone());
    let produced = i32::from_ne_bytes(bytes[..4].try_into().expect("survival count is i32"));
    let predicted = i32::from_ne_bytes(bytes[4..8].try_into().expect("survival count is i32"));
    Ok((
        u64::try_from(produced).expect("negative survival count"),
        u64::try_from(predicted).expect("negative survival count"),
    ))
}

/// Profiling workers for [`produced_pair_count`]; each strides over the tile and
/// adds once, so the counter takes at most this many atomics per tile.
const PROFILE_WORKERS: u32 = 4096;

/// Count the tile's pairs with [`PRODUCED`] set into `count[0]`.
#[cube(launch)]
fn produced_pair_count<I: Int>(
    pair_valid: &Tensor<I>,
    count: &mut Tensor<Atomic<I>>,
    slots: u32,
    workers: u32,
    #[define(I)] _dtype: ElemType,
) {
    let worker = ABSOLUTE_POS_X;
    if worker < workers {
        let mut produced = 0i32;
        let mut slot = worker;
        while slot < slots {
            if i32::cast_from(pair_valid[slot as usize]) & PRODUCED != 0 {
                produced += 1;
            }
            slot += workers;
        }
        if produced != 0 {
            count[0].fetch_add(I::cast_from(produced));
        }
    }
}

/// Actual record counts of one staged tile, for `CUSCAN_ANN_PROFILE`: records
/// produced (two per produced pair) and records kept by the endpoint filter (the
/// endpoint prefix total, already computed for scatter). Reads back and so
/// synchronizes; only profiled runs call it.
fn profile_tile_records(
    client: &cubecl::client::Client,
    count: &CubeTensor,
    pair_valid: &CubeTensor,
    endpoint_offsets: &CubeTensor,
    n: usize,
    slots: u32,
) -> Result<(u64, u64), AnnError> {
    count_zero::launch(
        client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        count.clone().into_tensor_arg(),
        dtype_to_storage_type(DType::I32),
    );
    let workers = slots.min(PROFILE_WORKERS);
    if workers > 0 {
        let (units, dim) = ann_independent_launch(client, workers)?;
        produced_pair_count::launch(
            client,
            units,
            dim,
            pair_valid.clone().into_tensor_arg(),
            count.clone().into_tensor_arg(),
            slots,
            workers,
            dtype_to_storage_type(DType::I32),
        );
    }
    let pairs = client.read_one_unchecked(count.handle.clone());
    let pairs = i32::from_ne_bytes(pairs[..4].try_into().expect("profile count is i32"));
    let kept = client.read_one_unchecked(
        endpoint_offsets
            .handle
            .clone()
            .offset_start((n * DType::I32.size()) as u64),
    );
    let kept = i32::from_ne_bytes(kept[..4].try_into().expect("endpoint total is i32"));
    Ok((
        2 * u64::try_from(pairs).expect("negative produced pair count"),
        u64::try_from(kept).expect("negative endpoint total"),
    ))
}

#[cube(launch)]
fn count_zero<I: Int>(count: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    count[0] = I::cast_from(0i32);
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
    if slot < slots as usize {
        let mask = i32::cast_from(pair_valid[slot]);
        let left = u32::cast_from(pair_left[slot]) as usize;
        let right = u32::cast_from(pair_right[slot]) as usize;
        if mask & KEEP_LEFT != 0 {
            let position =
                u32::cast_from(endpoint_cursors[left].fetch_add(I::cast_from(1i32))) as usize;
            endpoint_neighbor_ids[position] = I::cast_from(right as i32);
            endpoint_distances[position] = pair_distances[slot];
        }
        if mask & KEEP_RIGHT != 0 {
            let position =
                u32::cast_from(endpoint_cursors[right].fetch_add(I::cast_from(1i32))) as usize;
            endpoint_neighbor_ids[position] = I::cast_from(left as i32);
            endpoint_distances[position] = pair_distances[slot];
        }
    }
}

/// Insert a record that passed the caller's worst-entry filter and validity
/// checks into a sorted heap row, keeping each id once at its minimum distance.
#[cube]
fn serial_endpoint_insert<F: Float, I: Int>(
    n: usize,
    width: usize,
    id: I,
    proposal_distance: F,
    ids: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    base: usize,
) {
    let id_value = u32::cast_from(id) as usize;
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
        distances[base + width - 1] = F::new(f32::INFINITY);
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
    let valid = id_value < n && id_value != row && proposal_distance >= F::new(0.0f32);
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

/// CPU owns each row and updates its heap directly.
#[cube(launch)]
pub(super) fn endpoint_merge_cpu<F: Float, I: Int>(
    endpoint_offsets: &Tensor<I>,
    endpoint_ids: &Tensor<I>,
    endpoint_distances: &Tensor<F>,
    output_ids: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    row_worst_ids: &mut Tensor<I>,
    row_worst_distances: &mut Tensor<F>,
    width: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let n = output_ids.shape(0);
    let width = width as usize;
    if row < n {
        let start = u32::cast_from(endpoint_offsets[row]) as usize;
        let end = u32::cast_from(endpoint_offsets[row + 1]) as usize;
        let base = row * width;
        let last = base + width - 1;
        // The worst entry stays in registers and is reloaded only after an
        // insert, the only operation that can change it.
        let mut worst_id = output_ids[last];
        let mut worst_distance = output_distances[last];
        for position in start..end {
            let id = endpoint_ids[position];
            let distance = endpoint_distances[position];
            let id_value = u32::cast_from(id) as usize;
            let worst_index = u32::cast_from(worst_id) as usize;
            // A full heap cannot change for a record not better than its worst entry.
            let improves = worst_index >= n
                || distance < worst_distance
                || (distance == worst_distance && id_value < worst_index);
            if improves && id_value < n && id_value != row && distance >= F::new(0.0f32) {
                serial_endpoint_insert::<F, I>(
                    n,
                    width,
                    id,
                    distance,
                    output_ids,
                    output_distances,
                    base,
                );
                worst_id = output_ids[last];
                worst_distance = output_distances[last];
            }
        }
        if start < end {
            row_worst_ids[row] = worst_id;
            row_worst_distances[row] = worst_distance;
        }
    }
}

/// CUDA uses the heap as toggle state zero and a separate workspace as state one.
#[cube(launch)]
pub(super) fn endpoint_merge_cuda<F: Float, I: Int>(
    endpoint_offsets: &Tensor<I>,
    endpoint_ids: &Tensor<I>,
    endpoint_distances: &Tensor<F>,
    output_ids: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    workspace_ids: &mut Tensor<I>,
    workspace_distances: &mut Tensor<F>,
    row_worst_ids: &mut Tensor<I>,
    row_worst_distances: &mut Tensor<F>,
    width: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let lane = UNIT_POS_X as usize;
    let row = CUBE_POS_X as usize;
    let n = output_ids.shape(0);
    let width = width as usize;
    if row < n {
        let start = u32::cast_from(endpoint_offsets[row]) as usize;
        let end = u32::cast_from(endpoint_offsets[row + 1]) as usize;
        let row_base = row * width;
        if start < end {
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
                let mut slot = lane;
                while slot < width {
                    output_ids[row_base + slot] = workspace_ids[row_base + slot];
                    output_distances[row_base + slot] = workspace_distances[row_base + slot];
                    slot += 32;
                }
            }
            sync_cube();
            if lane == 0 {
                let worst = row_base + width - 1;
                row_worst_ids[row] = output_ids[worst];
                row_worst_distances[row] = output_distances[worst];
            }
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
        if left >= n || right >= n || left == right || distance.is_nan() || distance < 0.0 {
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
