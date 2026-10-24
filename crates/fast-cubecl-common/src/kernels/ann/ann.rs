//! Shared launch geometry, workspace planning and output conversion for sampled NN-descent.

use super::ann_forest::{
    MAX_CANDIDATES, WorkspaceError, WorkspacePlan, plan_workspace, tree_count,
};
use super::ann_sampled;
use crate::error::AnnError;
use burn::{backend::DispatchDevice, cubecl::prelude::*, tensor::DType};
use burn_cubecl::tensor::CubeTensor;

const ANN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

// The k-only graph lost exact neighbors on the small nonlinear regression
// (213/240 hits). A four-edge margin preserves alternative paths for refinement.
const HEAP_MARGIN: usize = 4;

#[inline]
pub(super) fn candidate_width(n: usize, k: usize) -> usize {
    k.saturating_add(HEAP_MARGIN).min(n - 1)
}

/// Client-independent shape decisions for one ANN call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostPlan {
    pub(super) width: usize,
    pub(super) capacity: usize,
    pub(super) workspace: WorkspacePlan,
}

/// Launch decision for the merge phase. Stored as plain numbers so that
/// preflight and the in-op plan can be compared for equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct LaunchShape {
    cpu_merge: bool,
    cubes: u32,
    dim_x: u32,
}

impl LaunchShape {
    pub(super) fn geometry(self) -> (bool, CubeCount, CubeDim) {
        (
            self.cpu_merge,
            CubeCount::Static(self.cubes, 1, 1),
            CubeDim::new_1d(self.dim_x),
        )
    }
}

/// Checked plan for one approximate-neighbor launch. `ann_preflight` and the
/// op both derive it through `plan_for_client`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnPlan {
    pub(super) host: HostPlan,
    merge: LaunchShape,
}

impl AnnPlan {
    pub(super) fn merge_geometry(&self) -> (bool, CubeCount, CubeDim) {
        self.merge.geometry()
    }
}

pub(super) fn workspace_error(stage: &str, error: WorkspaceError) -> AnnError {
    match error {
        WorkspaceError::InvalidDimensions => {
            AnnError::Workspace(format!("ANN {stage} rejected invalid dimensions"))
        }
        WorkspaceError::ArithmeticOverflow => AnnError::Overflow(format!("ANN {stage} overflows")),
    }
}

/// Rejects one device allocation whose size, rounded up to `alignment`, exceeds
/// the device's maximum page. Pure, so synthetic limits can be tested directly.
pub(super) fn check_page_limit(
    stage: &str,
    bytes: usize,
    max_page: usize,
    alignment: usize,
) -> Result<(), AnnError> {
    if alignment == 0 || max_page == 0 {
        return Err(AnnError::Capability(
            "device reports invalid allocation limits".into(),
        ));
    }
    let rounded = bytes
        .div_ceil(alignment)
        .checked_mul(alignment)
        .ok_or_else(|| AnnError::Overflow(format!("ANN {stage} allocation size overflows")))?;
    if rounded > max_page {
        return Err(AnnError::Capability(format!(
            "ANN {stage} allocation of {rounded} bytes exceeds device max page of {max_page} bytes"
        )));
    }
    Ok(())
}

/// Byte size of each distinct ANN device allocation for this shape, mirroring
/// the buffers created in `ann_sampled::approximate_neighbors` and
/// `forest_seed`. Every entry is checked against the device page limit.
fn ann_allocation_bytes(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
) -> Result<[(&'static str, usize); 11], AnnError> {
    let overflow = || AnnError::Overflow("ANN allocation size overflows usize".into());
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let ws = host.workspace;
    let w = host.width;
    let c = host.capacity;
    let chunks = n.div_ceil(ann_sampled::PREFIX_CHUNK);
    Ok([
        ("input copy", mul(mul(n, features)?, 4)?),
        ("forest membership", mul(ws.forest_membership_len, 4)?),
        (
            "forest leaf offsets",
            mul(add(ws.forest_leaf_count, 1)?, 8)?,
        ),
        ("heap and merge", mul(mul(n, w)?, 4)?),
        ("pair staging", mul(ws.max_pair_slots_per_tile, 4)?),
        ("endpoint staging", mul(ws.max_endpoint_records, 4)?),
        ("sampled pools", mul(mul(n, c * 2)?, 4)?),
        ("pool sizes", mul(mul(n, 2)?, 4)?),
        ("row offsets", mul(add(n, 1)?, 4)?),
        ("chunk sums", mul(chunks, 4)?),
        ("output", mul(mul(n, k)?, 4)?),
    ])
}

/// Pure shape and workspace checks. No client access.
fn host_plan(n: usize, features: usize, k: usize) -> Result<HostPlan, AnnError> {
    if n < 2 {
        return Err(AnnError::InvalidInput(format!(
            "ANN requires at least two rows, got {n}"
        )));
    }
    if features == 0 {
        return Err(AnnError::InvalidInput(
            "ANN requires at least one feature".into(),
        ));
    }
    if k == 0 || k >= n {
        return Err(AnnError::InvalidInput(format!(
            "ANN neighbor count k={k} must be in 1..{n}"
        )));
    }
    if n > i32::MAX as usize {
        return Err(AnnError::Overflow(format!(
            "ANN row count {n} exceeds i32 range"
        )));
    }
    let width = candidate_width(n, k);
    let capacity = MAX_CANDIDATES.min(n - 1);
    let trees = tree_count(n).map_err(|error| workspace_error("tree count", error))?;
    let workspace = plan_workspace(n, width, capacity, features, trees)
        .map_err(|error| workspace_error("workspace plan", error))?;
    u32::try_from(n)
        .map_err(|_| AnnError::Overflow("ANN row count exceeds CubeCL range".into()))?;
    u32::try_from(width)
        .map_err(|_| AnnError::Overflow("ANN width exceeds CubeCL range".into()))?;
    u32::try_from(capacity)
        .map_err(|_| AnnError::Overflow("ANN sample capacity exceeds CubeCL range".into()))?;
    let max_total_slots = n
        .checked_mul(workspace.pair_slots_per_center)
        .ok_or_else(|| AnnError::Overflow("ANN pair slot count overflows usize".into()))?;
    if max_total_slots > i32::MAX as usize {
        return Err(AnnError::Overflow(
            "ANN pair slot prefix exceeds i32".into(),
        ));
    }
    let reverse_elements = n
        .checked_mul(width)
        .ok_or_else(|| AnnError::Overflow("ANN reverse size overflows usize".into()))?;
    if reverse_elements > i32::MAX as usize {
        return Err(AnnError::Overflow("ANN reverse prefix exceeds i32".into()));
    }
    Ok(HostPlan {
        width,
        capacity,
        workspace,
    })
}

#[derive(Clone, Copy, Debug)]
struct CooperativeLimits {
    plane_ops: bool,
    plane_size_min: u32,
    max_units_per_cube: u32,
    max_cube_dim_x: u32,
    max_shared_memory_size: usize,
    max_cube_count_x: u32,
}

fn cooperative_limits(client: &cubecl::client::Client) -> CooperativeLimits {
    use cubecl::features::Plane;
    let hardware = &client.properties().hardware;
    CooperativeLimits {
        plane_ops: client.features().plane.contains(Plane::Ops),
        plane_size_min: hardware.plane_size_min,
        max_units_per_cube: hardware.max_units_per_cube,
        max_cube_dim_x: hardware.max_cube_dim.0,
        max_shared_memory_size: hardware.max_shared_memory_size,
        max_cube_count_x: hardware.max_cube_count.0,
    }
}

fn validate_cooperative_ann_launch(
    limits: CooperativeLimits,
    feature_size: usize,
    n_rows: u32,
) -> Result<(), AnnError> {
    let capability = |message: &str| Err(AnnError::Capability(message.into()));
    if !limits.plane_ops {
        return capability("cooperative ANN requires Plane::Ops");
    }
    if limits.plane_size_min < ANN_CUBE_DIM.x {
        return capability("cooperative ANN requires guaranteed plane width >=32");
    }
    if limits.max_units_per_cube < ANN_CUBE_DIM.x || limits.max_cube_dim_x < ANN_CUBE_DIM.x {
        return capability("cooperative ANN requires 32 units and x dimension");
    }
    let shared_bytes = 32 * (2 * DType::I32.size() + feature_size) + DType::I32.size();
    if shared_bytes > limits.max_shared_memory_size {
        return capability("cooperative ANN shared-memory capacity exceeded");
    }
    if limits.max_cube_count_x < n_rows {
        return capability("cooperative ANN x cube capacity exceeded");
    }
    Ok(())
}

pub(super) fn ann_launch_shape(
    client: &cubecl::client::Client,
    feature_size: usize,
    n_rows: u32,
) -> Result<LaunchShape, AnnError> {
    let cpu_merge = client.name() == "cpu";
    if !cpu_merge {
        validate_cooperative_ann_launch(cooperative_limits(client), feature_size, n_rows)?;
    }
    let hardware = &client.properties().hardware;
    ann_launch_shape_with_limits(
        cpu_merge,
        hardware.max_cube_dim,
        hardware.max_units_per_cube,
        hardware.max_cube_count.0,
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .try_into()
            .unwrap_or(u32::MAX),
        n_rows,
    )
}

fn ann_launch_shape_with_limits(
    cpu_merge: bool,
    max_cube_dim: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cubes: u32,
    cores: u32,
    n_rows: u32,
) -> Result<LaunchShape, AnnError> {
    if n_rows == 0 {
        return Err(AnnError::InvalidInput(
            "ANN row count must be nonzero".into(),
        ));
    }
    if cpu_merge {
        let dim_x = n_rows
            .min(cores)
            .min(max_cube_dim.0)
            .min(max_units_per_cube)
            .max(1);
        let cubes = ann_independent_cube_count(n_rows, dim_x, max_cubes)?;
        Ok(LaunchShape {
            cpu_merge: true,
            cubes,
            dim_x,
        })
    } else {
        if n_rows > max_cubes {
            return Err(AnnError::Capability(
                "ANN cooperative cube count exceeds hardware limit".into(),
            ));
        }
        Ok(LaunchShape {
            cpu_merge: false,
            cubes: n_rows,
            dim_x: ANN_CUBE_DIM.x,
        })
    }
}

pub(super) fn ann_independent_launch(
    client: &cubecl::client::Client,
    units: u32,
) -> Result<(CubeCount, CubeDim), AnnError> {
    let dim = crate::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x));
    let count =
        ann_independent_cube_count(units, dim.x, client.properties().hardware.max_cube_count.0)?;
    Ok((CubeCount::Static(count, 1, 1), dim))
}

fn ann_independent_cube_count(
    units: u32,
    units_per_cube: u32,
    max_cubes: u32,
) -> Result<u32, AnnError> {
    if units_per_cube == 0 {
        return Err(AnnError::Capability(
            "ANN independent launch dimension must be nonzero".into(),
        ));
    }
    let count = units.div_ceil(units_per_cube);
    if count > max_cubes {
        return Err(AnnError::Capability(
            "ANN independent cube count exceeds hardware limit".into(),
        ));
    }
    Ok(count)
}

/// Checked plan from the client's hardware and the problem shape. Both
/// `ann_preflight` and the op call this function.
pub(super) fn plan_for_client(
    client: &cubecl::client::Client,
    n: usize,
    features: usize,
    k: usize,
) -> Result<AnnPlan, AnnError> {
    let host = host_plan(n, features, k)?;
    let memory = &client.properties().memory;
    let max_page = usize::try_from(memory.max_page_size)
        .map_err(|_| AnnError::Capability("device max page size exceeds usize".into()))?;
    let alignment = usize::try_from(memory.alignment)
        .map_err(|_| AnnError::Capability("device alignment exceeds usize".into()))?;
    for (stage, bytes) in ann_allocation_bytes(n, features, k, &host)? {
        check_page_limit(stage, bytes, max_page, alignment)?;
    }
    let n_u32 = u32::try_from(n).expect("ANN row count validated by host_plan");
    let merge = ann_launch_shape(client, DType::F32.size(), n_u32)?;
    // Largest independent launch is bounded by rows and by one pair tile.
    let independent_units = n.max(host.workspace.max_pair_slots_per_tile);
    let independent_units = u32::try_from(independent_units)
        .map_err(|_| AnnError::Overflow("ANN independent launch exceeds CubeCL range".into()))?;
    let units_per_cube = crate::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x)).x;
    ann_independent_cube_count(
        independent_units,
        units_per_cube,
        client.properties().hardware.max_cube_count.0,
    )?;
    Ok(AnnPlan { host, merge })
}

/// Public launch preflight. Returns typed errors for invalid shapes and
/// unavailable hardware capability; never launches a kernel.
pub fn ann_preflight(
    device: &burn::tensor::Device,
    n: usize,
    features: usize,
    k: usize,
) -> Result<AnnPlan, AnnError> {
    let DispatchDevice::Cube(cube_device) = device.as_dispatch();
    plan_for_client(&cube_device.client(), n, features, k)
}

/// Allocation phase of one ANN call, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnPhase {
    Input,
    ForestInit,
    Refinement,
    Output,
}

/// Bytes live during one phase. Host and runtime are kept separate; they are
/// summed within a phase and never mixed with another phase's peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnPhaseBytes {
    pub phase: AnnPhase,
    /// Host RAM allocated by ANN itself during this phase.
    pub host_bytes: usize,
    /// ANN-owned runtime buffers (device on CUDA, host RAM on CPU), including
    /// the input tensor and forest tensors, which persist through refinement.
    pub runtime_bytes: usize,
}

/// Client-independent, shape-only allocation estimate for one ANN call.
///
/// Totals are advisory. They are not allocator reservations, per-buffer device
/// limits, or total process memory. Per-buffer limits remain runtime checks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnEstimate {
    /// Retained/output candidate width, `min(k + 4, n - 1)`.
    pub retained_width: usize,
    /// Sampling budget per new/old class, `min(60, n - 1)`. Not the output width.
    pub sample_capacity: usize,
    pub trees: usize,
    /// Output edges, `n * k`.
    pub directed_edges: usize,
    /// Phases in execution order: Input, ForestInit, Refinement, Output.
    pub phases: [AnnPhaseBytes; 4],
    /// Maximum host bytes across phases.
    pub host_peak_bytes: usize,
    /// Maximum runtime bytes across phases.
    pub runtime_peak_bytes: usize,
    /// Maximum over phases of `host + runtime`, for CPU where both are host RAM.
    pub cpu_combined_peak_bytes: usize,
}

/// Pure, checked allocation estimate for `ann_preflight` shapes. Uses the same
/// `host_plan` as preflight and execution, and makes no device access.
pub fn ann_host_estimate(n: usize, features: usize, k: usize) -> Result<AnnEstimate, AnnError> {
    let host = host_plan(n, features, k)?;
    estimate_from_plan(n, features, k, &host)
}

fn estimate_from_plan(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
) -> Result<AnnEstimate, AnnError> {
    let overflow = || AnnError::Overflow("ANN allocation estimate overflows usize".into());
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let ws = host.workspace;
    let w = host.width;
    let c = host.capacity;
    // Every ANN element type is a 4-byte I32 or F32.
    let input = mul(mul(n, features)?, 4)?;
    // Eleven n x H buffers: heap ids/dists/flags, merge destinations A/B,
    // merge workspace, next flags, and reverse sources (n x H elements).
    let heap = mul(mul(n, w)?, 44)?;
    // Heap ids/dists/flags created by forest initialization.
    let init_heap = mul(mul(n, w)?, 12)?;
    // Forest membership is i32 (4 bytes). forest_seed stores one i32 pair per
    // leaf offset entry, including the terminal offset: (leaves + 1) pairs.
    let forest = add(
        mul(ws.forest_membership_len, 4)?,
        mul(add(ws.forest_leaf_count, 1)?, 8)?,
    )?;
    // Per-row and per-center scratch: 8n counts/pools/cursors, three (n+1)
    // offsets, 4nC sampled ids/priorities, chunk sums, and new_count.
    let scratch_elems = add(
        add(mul(n, 8)?, mul(add(n, 1)?, 3)?)?,
        add(mul(mul(n, c)?, 4)?, add(n.div_ceil(128), 1)?)?,
    )?;
    let scratch = mul(scratch_elems, 4)?;
    // Pair staging: four 4-byte pair arrays. Endpoint staging: id and distance.
    let staging = add(
        mul(ws.max_pair_slots_per_tile, 16)?,
        mul(ws.max_endpoint_records, 8)?,
    )?;
    let output = mul(mul(n, k)?, 8)?;
    let trees = tree_count(n).map_err(|error| workspace_error("tree count", error))?;

    let refinement_runtime = add(add(add(input, forest)?, add(heap, scratch)?)?, staging)?;
    let phases = [
        AnnPhaseBytes {
            phase: AnnPhase::Input,
            host_bytes: mul(input, 2)?,
            runtime_bytes: input,
        },
        AnnPhaseBytes {
            phase: AnnPhase::ForestInit,
            host_bytes: ws.host_peak_bytes,
            runtime_bytes: add(add(input, forest)?, init_heap)?,
        },
        AnnPhaseBytes {
            phase: AnnPhase::Refinement,
            host_bytes: 0,
            runtime_bytes: refinement_runtime,
        },
        AnnPhaseBytes {
            phase: AnnPhase::Output,
            host_bytes: 0,
            runtime_bytes: add(refinement_runtime, output)?,
        },
    ];
    let host_peak_bytes = phases.iter().map(|p| p.host_bytes).max().unwrap_or(0);
    let runtime_peak_bytes = phases.iter().map(|p| p.runtime_bytes).max().unwrap_or(0);
    let cpu_combined_peak_bytes = phases.iter().try_fold(0usize, |peak, p| {
        add(p.host_bytes, p.runtime_bytes).map(|v| peak.max(v))
    })?;
    Ok(AnnEstimate {
        retained_width: w,
        sample_capacity: c,
        trees,
        directed_edges: mul(n, k)?,
        phases,
        host_peak_bytes,
        runtime_peak_bytes,
        cpu_combined_peak_bytes,
    })
}

pub(super) fn ann_should_stop(new_count: i32, n: usize, width: usize) -> bool {
    let capacity = n
        .checked_mul(width)
        .expect("ANN candidate elements overflow usize");
    assert!(capacity <= i32::MAX as usize, "ANN count exceeds i32");
    assert!(
        new_count >= 0 && new_count as usize <= capacity,
        "ANN new-membership count is outside candidate capacity"
    );
    (new_count as u64) * 1000 <= capacity as u64
}

#[cube]
fn reverse_first_hit<I: Int>(
    candidates: &Tensor<I>,
    source: usize,
    slot: usize,
    n: usize,
    width: usize,
) -> bool {
    let target = u32::cast_from(candidates[source * width + slot]) as usize;
    let mut first = target < n && target != source;
    if first {
        for previous in 0..slot {
            if candidates[source * width + previous] == candidates[source * width + slot] {
                first = false;
            }
        }
    }
    first
}

#[cube(launch)]
pub(super) fn reverse_zero<I: Int>(counts: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    let target = ABSOLUTE_POS_X as usize;
    if target < counts.shape(0) {
        counts[target] = I::cast_from(0i32);
    }
}

#[cube(launch)]
pub(super) fn reverse_count<I: Int>(
    candidates: &Tensor<I>,
    counts: &mut Tensor<Atomic<I>>,
    n: u32,
    width: u32,
    #[define(I)] _dtype: ElemType,
) {
    let source = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if source < n {
        for slot in 0..width {
            if reverse_first_hit(candidates, source, slot, n, width) {
                let target = u32::cast_from(candidates[source * width + slot]) as usize;
                counts[target].fetch_add(I::cast_from(1i32));
            }
        }
    }
}

#[cube(launch)]
pub(super) fn reverse_scatter<I: Int>(
    candidates: &Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    sources: &mut Tensor<I>,
    n: u32,
    width: u32,
    #[define(I)] _dtype: ElemType,
) {
    let source = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if source < n {
        for slot in 0..width {
            if reverse_first_hit(candidates, source, slot, n, width) {
                let target = u32::cast_from(candidates[source * width + slot]) as usize;
                let position =
                    u32::cast_from(cursors[target].fetch_add(I::cast_from(1i32))) as usize;
                sources[position] = I::cast_from(source as i32);
            }
        }
    }
}

#[cube(launch)]
pub(super) fn ann_output<F: Float, I: Int>(
    indices: &Tensor<I>,
    squared_distances: &Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    width: u32,
    k: u32,
    #[comptime] squared: bool,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    // The candidate tensor is [n, width], so its first dimension is the row count.
    let n = indices.shape(0);
    let width = width as usize;
    let k = k as usize;
    if row < n {
        for slot in 0..k {
            let source = row * width + slot;
            let distance = squared_distances[source];
            if comptime![squared] {
                // Squared output keeps real +inf overflow neighbors as-is.
                output_indices[row * k + slot] = indices[source];
                output_distances[row * k + slot] = distance;
            } else if distance == F::new(f32::INFINITY) {
                // Unsquared output has no infinite-distance neighbors: an
                // unrepresentable or missing slot becomes the documented
                // sentinel (n, sqrt(max)), matching exact unsquared output.
                output_indices[row * k + slot] = I::cast_from(n as i32);
                output_distances[row * k + slot] = F::sqrt(F::new(f32::MAX));
            } else {
                output_indices[row * k + slot] = indices[source];
                output_distances[row * k + slot] = F::sqrt(distance);
            }
        }
    }
}

/// Bounded sampled NN-descent. `squared` skips the output square root.
///
/// The trait returns a tensor pair, so an invalid launch is reported by
/// panicking. The estimator must call `ann_preflight` before dispatch to
/// surface `AnnError` as a typed error.
pub(crate) fn approximate_neighbors(
    data: CubeTensor,
    k: u32,
    seed: u64,
    max_iterations: u32,
    squared: bool,
) -> (CubeTensor, CubeTensor) {
    let k = usize::try_from(k).expect("ANN width exceeds usize");
    ann_sampled::approximate_neighbors(data, k, seed, max_iterations, squared)
        .unwrap_or_else(|error| panic!("ANN launch rejected after preflight: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_limits() -> CooperativeLimits {
        CooperativeLimits {
            plane_ops: true,
            plane_size_min: 32,
            max_units_per_cube: 32,
            max_cube_dim_x: 32,
            max_shared_memory_size: 32 * 1024,
            max_cube_count_x: 67,
        }
    }

    #[test]
    fn host_plan_forest_prefix_overflow_follows_passing_pair_slot_check() {
        // Pair-slot prefix n * 5370 is below i32::MAX, so only forest totals fail.
        assert!(390_000usize * 5370 <= i32::MAX as usize);
        let error = ann_host_estimate(390_000, 2, 256).unwrap_err();
        assert!(matches!(
            &error,
            AnnError::Overflow(message) if message.as_str() == "ANN workspace plan overflows"
        ));
    }

    #[test]
    fn host_estimate_counts_forest_terminal_metadata_and_phase_totals() {
        // n=11, d=2, k=3: width 7, capacity 10, 5 trees, one leaf per tree.
        let estimate = ann_host_estimate(11, 2, 3).unwrap();
        assert_eq!(estimate.retained_width, 7);
        assert_eq!(estimate.sample_capacity, 10);
        assert_eq!(estimate.trees, 5);
        assert_eq!(estimate.directed_edges, 33);
        let [input, forest_init, refinement, output] = estimate.phases;
        assert_eq!(input.runtime_bytes, 88);
        // Forest: 55 i32 membership (220 B) + (5 leaves + terminal) i32 pairs (48 B),
        // plus 12*n*H heap seeds (924 B) and the input tensor.
        assert_eq!(forest_init.runtime_bytes, 88 + 268 + 924);
        // Refinement adds 11*n*H heap buffers, 4-byte scratch, and pair/endpoint staging.
        assert_eq!(refinement.runtime_bytes, 57_048);
        // Output adds 8*n*k bytes of I32/F32 output tensors.
        assert_eq!(output.runtime_bytes, 57_312);
        assert_eq!(estimate.runtime_peak_bytes, 57_312);
        let cpu = estimate
            .phases
            .iter()
            .map(|phase| phase.host_bytes + phase.runtime_bytes)
            .max();
        assert_eq!(Some(estimate.cpu_combined_peak_bytes), cpu);
    }

    #[test]
    fn page_limit_is_inclusive_after_alignment_rounding() {
        // Exact page size is accepted; one byte over fails.
        assert!(check_page_limit("synthetic", 4096, 4096, 256).is_ok());
        assert!(check_page_limit("synthetic", 4097, 4096, 256).is_err());
        // 3841 bytes round up to 4096 under 256-byte alignment: accepted.
        assert!(check_page_limit("synthetic", 3841, 4096, 256).is_ok());
        // 4081 bytes round up to 4096: accepted at page 4096, rejected at 4095.
        assert!(check_page_limit("synthetic", 4081, 4096, 256).is_ok());
        assert!(check_page_limit("synthetic", 4081, 4095, 256).is_err());
        // Zero-sized allocation is always within the page.
        assert!(check_page_limit("synthetic", 0, 4096, 256).is_ok());
    }

    #[test]
    fn page_limit_rejects_invalid_limits_and_rounding_overflow() {
        assert!(matches!(
            check_page_limit("synthetic", 1, 4096, 0),
            Err(AnnError::Capability(_))
        ));
        assert!(matches!(
            check_page_limit("synthetic", 1, 0, 256),
            Err(AnnError::Capability(_))
        ));
        assert!(matches!(
            check_page_limit("synthetic", usize::MAX, usize::MAX, 256),
            Err(AnnError::Overflow(_))
        ));
    }

    #[test]
    fn allocation_list_covers_largest_buffer_for_small_shape() {
        // n=11, d=2, k=3: width 7, capacity 10, one leaf per tree, 5 trees.
        // Endpoint staging is 2 * (11 * 145) = 3190 i32/f32 elements = 12760 bytes,
        // the largest entry (pair staging 6380, membership 220, output 132).
        let host = host_plan(11, 2, 3).unwrap();
        let entries = ann_allocation_bytes(11, 2, 3, &host).unwrap();
        let largest = entries.iter().map(|(_, bytes)| *bytes).max().unwrap();
        assert_eq!(largest, 12_760);
        assert!(
            entries
                .iter()
                .any(|(stage, bytes)| { *stage == "forest membership" && *bytes == 220 })
        );
        assert!(
            entries
                .iter()
                .any(|(stage, bytes)| *stage == "output" && *bytes == 132)
        );
        // A page limit one byte below the largest entry rejects the whole plan.
        assert!(check_page_limit("endpoint staging", largest, largest - 1, 4).is_err());
    }

    #[test]
    fn ann_stop_threshold_is_inclusive_and_checked() {
        assert!(ann_should_stop(0, 1, 1));
        assert!(ann_should_stop(1, 1, 1000));
        assert!(!ann_should_stop(1, 1, 999));
        assert!(!ann_should_stop(2, 1, 1000));
        assert!(std::panic::catch_unwind(|| ann_should_stop(-1, 1, 1)).is_err());
        assert!(std::panic::catch_unwind(|| ann_should_stop(2, 1, 1)).is_err());
        assert!(std::panic::catch_unwind(|| ann_should_stop(0, usize::MAX, 2)).is_err());
    }

    #[test]
    fn geometry_preserves_row_ownership_and_hardware_limits() {
        assert_eq!(
            ann_launch_shape_with_limits(true, (8, 1, 1), 8, 16, 6, 67).unwrap(),
            LaunchShape {
                cpu_merge: true,
                cubes: 12,
                dim_x: 6,
            }
        );
        assert_eq!(
            ann_launch_shape_with_limits(false, (32, 1, 1), 32, 67, 6, 67).unwrap(),
            LaunchShape {
                cpu_merge: false,
                cubes: 67,
                dim_x: 32,
            }
        );
        assert!(matches!(
            ann_independent_cube_count(67, 32, 2),
            Err(AnnError::Capability(_))
        ));
        assert!(matches!(
            ann_independent_cube_count(1, 0, 2),
            Err(AnnError::Capability(_))
        ));
        assert!(matches!(
            ann_launch_shape_with_limits(false, (32, 1, 1), 32, 66, 6, 67),
            Err(AnnError::Capability(_))
        ));
        assert!(matches!(
            ann_launch_shape_with_limits(false, (32, 1, 1), 32, 66, 6, 0),
            Err(AnnError::InvalidInput(_))
        ));
    }

    #[test]
    fn simulated_cooperative_capability_absence_is_typed() {
        assert!(validate_cooperative_ann_launch(full_limits(), 4, 67).is_ok());
        let cases = [
            CooperativeLimits {
                plane_ops: false,
                ..full_limits()
            },
            CooperativeLimits {
                plane_size_min: 16,
                ..full_limits()
            },
            CooperativeLimits {
                max_units_per_cube: 16,
                ..full_limits()
            },
            CooperativeLimits {
                max_cube_dim_x: 16,
                ..full_limits()
            },
            CooperativeLimits {
                max_shared_memory_size: 0,
                ..full_limits()
            },
            CooperativeLimits {
                max_cube_count_x: 66,
                ..full_limits()
            },
        ];
        for limits in cases {
            assert!(matches!(
                validate_cooperative_ann_launch(limits, 4, 67),
                Err(AnnError::Capability(_))
            ));
        }
    }

    #[test]
    fn host_plan_rejects_tiny_n_and_invalid_k_without_panicking() {
        assert!(matches!(host_plan(0, 3, 1), Err(AnnError::InvalidInput(_))));
        assert!(matches!(host_plan(1, 3, 1), Err(AnnError::InvalidInput(_))));
        assert!(matches!(host_plan(5, 3, 0), Err(AnnError::InvalidInput(_))));
        assert!(matches!(host_plan(5, 3, 5), Err(AnnError::InvalidInput(_))));
        assert!(matches!(host_plan(5, 0, 2), Err(AnnError::InvalidInput(_))));
        assert!(host_plan(67, 3, 5).is_ok());
    }

    #[test]
    fn host_plan_rejects_reverse_prefix_overflow_before_allocation() {
        assert!(matches!(
            host_plan(65_536, 1, 65_535),
            Err(AnnError::Overflow(_))
        ));
        assert!(matches!(
            host_plan(3, usize::MAX, 1),
            Err(AnnError::Overflow(_))
        ));
    }

    #[test]
    fn preflight_plan_equals_in_op_plan() {
        let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
        let DispatchDevice::Cube(cube_device) = device.as_dispatch();
        let client = cube_device.client();
        for (n, features, k) in [(67, 3, 5), (200, 8, 10)] {
            let preflight = ann_preflight(&device, n, features, k).unwrap();
            let in_op = plan_for_client(&client, n, features, k).unwrap();
            assert_eq!(preflight, in_op);
        }
    }

    #[test]
    fn preflight_rejects_tiny_n_and_invalid_k_through_device() {
        let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
        assert!(matches!(
            ann_preflight(&device, 1, 3, 1),
            Err(AnnError::InvalidInput(_))
        ));
        assert!(matches!(
            ann_preflight(&device, 5, 3, 5),
            Err(AnnError::InvalidInput(_))
        ));
    }
}
