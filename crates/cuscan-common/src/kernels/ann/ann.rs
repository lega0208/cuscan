//! Shared launch geometry, workspace planning and output conversion for sampled NN-descent.

pub(super) use super::ann_forest::candidate_width;
use super::ann_forest::{
    NnDescentParams, WorkspaceError, WorkspacePlan, nn_descent_params, plan_workspace,
};
use super::ann_sampled;
use crate::error::AnnError;
use burn::backend::TensorMetadata;
use burn::{backend::DispatchDevice, cubecl::prelude::*, tensor::DType};
use burn_cubecl::tensor::CubeTensor;

const ANN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);
const BUDGET_TILE_CEILING_SLOTS: usize = 1 << 24;
pub(super) const OWNER_CH: usize = 64;
pub(super) const OWNER_CH_LEAF: usize = 1;
pub(super) const OWNER_D_MAX: usize = 64;

/// Join implementation selected at preflight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnnJoinRoute {
    Owner,
    Slots,
}

impl AnnJoinRoute {
    fn for_cpu(cpu: bool) -> Self {
        if cpu { Self::Owner } else { Self::Slots }
    }
}

/// Bounds used by the budgeted tile planner, independent of runtime handles.
#[derive(Clone, Copy, Debug)]
pub(super) struct TileCaps {
    pub max_slots: usize,
    pub budget_ceiling_slots: usize,
    pub max_page: usize,
    pub alignment: usize,
    pub units_per_cube: u32,
    pub max_cubes: u32,
    pub cpu_merge: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum TileRequest {
    Default,
    Exact(usize),
    Budget { usable_bytes: u64, fixed_bytes: u64 },
}

/// Sum the route's staging allocations after individual alignment.
fn tile_staging_bytes(
    slots: usize,
    alignment: usize,
    route: AnnJoinRoute,
) -> Result<u64, AnnError> {
    staging_entries(slots, route)
        .ok_or_else(|| AnnError::Overflow("ANN staging shape overflows usize".into()))?
        .iter()
        .try_fold(0u64, |sum, entry| {
            let bytes = entry
                .bytes()
                .ok_or_else(|| AnnError::Overflow("ANN staging bytes overflow usize".into()))?;
            sum.checked_add(aligned_bytes(bytes, alignment)?)
                .ok_or_else(|| AnnError::Overflow("ANN staging total overflows u64".into()))
        })
}

fn plan_tiles(caps: TileCaps, request: TileRequest) -> Result<usize, AnnError> {
    if caps.max_slots == 0 || caps.max_page == 0 || caps.alignment == 0 || caps.units_per_cube == 0
    {
        return Err(AnnError::Capability(
            "ANN tile limits must be nonzero".into(),
        ));
    }
    let minimum = caps.max_slots.min(1 << 20);
    let aligned_page = caps.max_page / caps.alignment * caps.alignment;
    let launch_cap = u64::from(caps.max_cubes) * u64::from(caps.units_per_cube);
    let upper = caps
        .max_slots
        .min(i32::MAX as usize / 2)
        .min(aligned_page / 8)
        .min(usize::try_from(launch_cap).unwrap_or(usize::MAX))
        .min(if caps.cpu_merge { 1 << 26 } else { usize::MAX });
    if upper < minimum {
        return Err(AnnError::Capability(format!(
            "ANN minimum tile {minimum} exceeds device tile cap {upper}"
        )));
    }
    match request {
        TileRequest::Default => Ok(upper.min(1 << 22)),
        TileRequest::Exact(slots) => {
            if slots < minimum || slots > upper {
                Err(AnnError::InvalidInput(format!(
                    "ANN tile {slots} outside {minimum}..={upper}"
                )))
            } else {
                Ok(slots)
            }
        }
        TileRequest::Budget {
            usable_bytes,
            fixed_bytes,
        } => {
            let minimum_bytes = tile_staging_bytes(
                minimum,
                caps.alignment,
                AnnJoinRoute::for_cpu(caps.cpu_merge),
            )?;
            let required = fixed_bytes
                .checked_add(minimum_bytes)
                .ok_or_else(|| AnnError::Overflow("ANN minimum memory exceeds u64".into()))?;
            if usable_bytes < required {
                return Err(AnnError::Memory {
                    required_bytes: required,
                    available_bytes: usable_bytes,
                    message: format!("ANN needs {required} bytes but {usable_bytes} are available"),
                });
            }
            let elastic = usable_bytes - fixed_bytes;
            let (mut low, mut high) = (minimum, upper.min(caps.budget_ceiling_slots));
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if tile_staging_bytes(mid, caps.alignment, AnnJoinRoute::for_cpu(caps.cpu_merge))?
                    <= elastic
                {
                    low = mid;
                } else {
                    high = mid - 1;
                }
            }
            Ok(low)
        }
    }
}

/// Client-independent shape decisions for one ANN call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostPlan {
    pub(super) width: usize,
    pub(super) params: NnDescentParams,
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

/// Pair-slot kernel and its cooperative execution parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotRecipe {
    PerSlot,
    Plane {
        group: usize,
        decode: bool,
        unroll: usize,
    },
}

impl std::fmt::Display for SlotRecipe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PerSlot => f.write_str("per_slot"),
            Self::Plane {
                group,
                decode,
                unroll,
            } => write!(
                f,
                "plane:g{group}:decode{}:u{unroll}",
                if *decode { "plane" } else { "loop" }
            ),
        }
    }
}

/// Recipes selected for wide and scalar input bindings respectively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotPolicy {
    pub wide: SlotRecipe,
    pub scalar: SlotRecipe,
}

impl SlotPolicy {
    fn for_cpu(cpu: bool) -> Self {
        if cpu {
            Self::default()
        } else {
            Self {
                wide: SlotRecipe::Plane {
                    group: 2,
                    decode: false,
                    unroll: 1,
                },
                scalar: SlotRecipe::PerSlot,
            }
        }
    }

    fn validate(self, cpu: bool) -> Result<(), AnnError> {
        for (recipe, wide) in [(self.wide, true), (self.scalar, false)] {
            if let SlotRecipe::Plane { group, unroll, .. } = recipe {
                if cpu
                    || ![1, 2, 4, 8].contains(&group)
                    || (group == 2 && !wide)
                    || ![1, 2].contains(&unroll)
                {
                    return Err(AnnError::InvalidInput("invalid ANN slot policy".into()));
                }
            }
        }
        Ok(())
    }

    pub(super) fn select(self, width: usize) -> (SlotRecipe, usize) {
        let recipe = if width == 1 { self.scalar } else { self.wide };
        let binding = match recipe {
            SlotRecipe::Plane { group: 2, .. } => 4,
            SlotRecipe::Plane { group: 4 | 8, .. } => 1,
            _ => width,
        };
        (recipe, binding)
    }
}

impl Default for SlotPolicy {
    fn default() -> Self {
        Self {
            wide: SlotRecipe::PerSlot,
            scalar: SlotRecipe::PerSlot,
        }
    }
}

/// Checked shape, launch geometry and immutable memory admission for one ANN call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnnPlan {
    pub(super) host: HostPlan,
    merge: LaunchShape,
    pub(super) vector_widths: &'static [usize],
    pub(crate) join_route: AnnJoinRoute,
    pub(super) slot_policy: SlotPolicy,
    pub n: usize,
    pub features: usize,
    pub k: usize,
    pub tile_slots: usize,
    pub memory: PlannedMemory,
    #[cfg(test)]
    pub(super) owner_test_geometry: Option<(usize, u32, usize)>,
}

/// Immutable admission numbers; executable validation does not query availability again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlannedMemory {
    pub fixed_peak_bytes: u64,
    pub staging_bytes: u64,
    pub usable_bytes: u64,
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
        WorkspaceError::ArithmeticOverflow => {
            AnnError::Overflow(format!("ANN {stage} overflows usize"))
        }
        WorkspaceError::Capacity {
            quantity,
            value,
            limit,
        } => AnnError::Overflow(format!("ANN {stage}: {quantity} {value} exceeds {limit}")),
    }
}

/// Appends the shape context to a capacity message produced by `host_plan`.
fn with_shape(error: AnnError, shape: &str) -> AnnError {
    match error {
        AnnError::Overflow(message) => AnnError::Overflow(format!("{message} ({shape})")),
        other => other,
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
        .ok_or_else(|| {
            AnnError::Overflow(format!(
                "ANN {stage} allocation size {bytes} rounded to alignment {alignment} overflows usize"
            ))
        })?;
    if rounded > max_page {
        return Err(AnnError::Capability(format!(
            "ANN {stage} allocation of {rounded} bytes exceeds device max page of {max_page} bytes"
        )));
    }
    Ok(())
}

/// Each variant names one allocation. Input is caller-owned and credited as live;
/// InputCopy conservatively reserves row-major materialization on every route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Buffer {
    Input,
    InputCopy,
    ForestRows,
    ForestLeafRows,
    ForestPairPrefix,
    HeapIds,
    HeapDistances,
    RowWorstIds,
    RowWorstDistances,
    Flags,
    PairLeft,
    PairRight,
    PairDistances,
    PairValid,
    EndpointCounts,
    EndpointOffsets,
    EndpointCursors,
    ReverseCounts,
    ReverseOffsets,
    ReverseCursors,
    ReverseSources,
    EndpointIds,
    EndpointDistances,
    SampledIds,
    WorkingIds,
    MergeWorkspaceIds,
    MergeWorkspaceDistances,
    NextFlags,
    NewCount,
    PoolSizes,
    PairCounts,
    CenterOffsets,
    ChunkSums,
    JoinChunkSums,
    JoinCursorPlaceholder,
    OutputIds,
    OutputDistances,
    ProfileCounts,
    SurvivalCounts,
    OwnerRecordRows,
    OwnerRecordIds,
    OwnerRecordDistances,
    OwnerRecordOrder,
    OwnerChunkOffsets,
    OwnerProfile,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferRoute {
    Both,
    Cuda,
    Slots,
    Owner,
    /// Allocated only when profiling; not part of the memory plan.
    Profile,
}

#[derive(Clone, Debug)]
pub(super) struct BufferEntry {
    pub buffer: Buffer,
    pub shape: Vec<usize>,
    pub dtype: DType,
    /// Bit mask for Input, ForestInit, Refinement and Output respectively.
    pub phases: u8,
    pub route: BufferRoute,
    pub staging: bool,
}

impl BufferEntry {
    pub(super) fn elements(&self) -> Option<usize> {
        self.shape.iter().try_fold(1usize, |a, &b| a.checked_mul(b))
    }

    pub(super) fn bytes(&self) -> Option<usize> {
        self.elements()?.checked_mul(self.dtype.size())
    }
}

fn staging_entries(slots: usize, route: AnnJoinRoute) -> Option<Vec<BufferEntry>> {
    use Buffer::*;
    let endpoints = slots.checked_mul(2)?;
    let entries = match route {
        AnnJoinRoute::Slots => vec![
            (PairLeft, DType::I32, slots),
            (PairRight, DType::I32, slots),
            (PairDistances, DType::F32, slots),
            (PairValid, DType::I32, slots),
            (EndpointIds, DType::I32, endpoints),
            (EndpointDistances, DType::F32, endpoints),
        ],
        AnnJoinRoute::Owner => vec![
            (OwnerRecordRows, DType::I32, endpoints),
            (OwnerRecordIds, DType::I32, endpoints),
            (OwnerRecordDistances, DType::F32, endpoints),
            (OwnerRecordOrder, DType::I32, endpoints),
        ],
    };
    Some(
        entries
            .into_iter()
            .map(|(buffer, dtype, count)| BufferEntry {
                buffer,
                shape: vec![count],
                dtype,
                phases: 6,
                route: match route {
                    AnnJoinRoute::Owner => BufferRoute::Owner,
                    AnnJoinRoute::Slots => BufferRoute::Slots,
                },
                staging: true,
            })
            .collect(),
    )
}

/// The driver's allocations and their live phases, including caller-owned input.
#[allow(clippy::too_many_arguments)]
pub(super) fn ann_buffer_table(
    n: usize,
    features: usize,
    k: usize,
    width: usize,
    capacity: usize,
    ws: &WorkspacePlan,
    tile_slots: usize,
    route: AnnJoinRoute,
) -> Option<Vec<BufferEntry>> {
    use Buffer::*;
    let chunks = n.div_ceil(ann_sampled::PREFIX_CHUNK);
    let leaf_entries = ws.forest_leaf_count.checked_add(1)?;
    let row_entries = n.checked_add(1)?;
    let pool_width = capacity.checked_mul(2)?;
    let pool_sizes = n.checked_mul(2)?;
    let mut entries = Vec::new();
    macro_rules! entry {
        ($name:ident, $dtype:ident, $phases:expr, $staging:expr, $($dim:expr),+) => {
            entries.push(BufferEntry { buffer: $name, shape: vec![$($dim),+], dtype: DType::$dtype,
                phases: $phases, route: BufferRoute::Both, staging: $staging });
        };
    }
    entry!(Input, F32, 15, false, n, features);
    entry!(InputCopy, F32, 15, false, n, features);
    entry!(ForestRows, I32, 2, false, ws.forest_membership_len);
    entry!(ForestLeafRows, I32, 2, false, leaf_entries);
    entry!(ForestPairPrefix, I64, 2, false, leaf_entries);
    entry!(HeapIds, I32, 14, false, n, width);
    entry!(HeapDistances, F32, 14, false, n, width);
    entry!(RowWorstIds, I32, 6, false, n);
    entry!(RowWorstDistances, F32, 6, false, n);
    entry!(Flags, I32, 6, false, n, width);
    entries.extend(staging_entries(tile_slots, route)?);
    if route == AnnJoinRoute::Slots {
        entry!(EndpointCounts, I32, 6, false, n);
        entries.last_mut()?.route = BufferRoute::Slots;
        entry!(EndpointOffsets, I32, 6, false, row_entries);
        entries.last_mut()?.route = BufferRoute::Slots;
        entry!(EndpointCursors, I32, 6, false, n);
        entries.last_mut()?.route = BufferRoute::Slots;
    } else {
        let owner_chunks = n
            .div_ceil(OWNER_CH)
            .max(ws.forest_leaf_count.div_ceil(OWNER_CH_LEAF));
        entry!(
            OwnerChunkOffsets,
            I32,
            6,
            false,
            owner_chunks,
            OWNER_D_MAX + 1
        );
        entries.last_mut()?.route = BufferRoute::Owner;
        entry!(OwnerProfile, U64, 6, false, 2);
        entries.last_mut()?.route = BufferRoute::Owner;
    }
    entry!(ReverseCounts, I32, 4, false, n);
    entry!(ReverseOffsets, I32, 4, false, row_entries);
    entry!(ReverseCursors, I32, 4, false, n);
    entry!(ReverseSources, I32, 4, false, n.checked_mul(width)?);
    entry!(SampledIds, I32, 4, false, n, pool_width);
    entry!(WorkingIds, I32, 4, false, n, width);
    entry!(MergeWorkspaceIds, I32, 6, false, n, width);
    entries.last_mut()?.route = BufferRoute::Cuda;
    entry!(MergeWorkspaceDistances, F32, 6, false, n, width);
    entries.last_mut()?.route = BufferRoute::Cuda;
    entry!(NextFlags, I32, 4, false, n, width);
    entry!(NewCount, I32, 4, false, 1);
    entry!(PoolSizes, I32, 4, false, pool_sizes);
    entry!(PairCounts, I64, 4, false, n);
    entry!(CenterOffsets, I64, 4, false, row_entries);
    entry!(ChunkSums, I32, 6, false, chunks);
    entry!(JoinChunkSums, I64, 4, false, chunks);
    entry!(JoinCursorPlaceholder, I64, 4, false, 1);
    entry!(OutputIds, I32, 8, false, n, k);
    entry!(OutputDistances, F32, 8, false, n, k);
    // Optional profiling counters: allocated only on profiled runs and excluded
    // from the memory plan.
    for (buffer, elements) in [(ProfileCounts, 1), (SurvivalCounts, 2)] {
        entries.push(BufferEntry {
            buffer,
            shape: vec![elements],
            dtype: DType::I32,
            phases: 6,
            route: BufferRoute::Profile,
            staging: false,
        });
    }
    Some(entries)
}

/// Byte size of each distinct ANN device allocation for this shape. Every entry
/// is checked against the device page limit.
fn ann_allocation_bytes(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
    tile_slots: usize,
    cpu: bool,
    route: AnnJoinRoute,
) -> Result<Vec<(String, usize)>, AnnError> {
    let overflow = || {
        AnnError::Overflow(format!(
            "ANN allocation size overflows usize (n={n}, features={features}, k={k})"
        ))
    };
    let table = ann_buffer_table(
        n,
        features,
        k,
        host.width,
        host.workspace.max_candidates_per_class,
        &host.workspace,
        tile_slots,
        route,
    )
    .ok_or_else(overflow)?;
    table
        .into_iter()
        .filter(|entry| entry.route != BufferRoute::Cuda || !cpu)
        .map(|entry| {
            Ok((
                format!("{:?}", entry.buffer),
                entry.bytes().ok_or_else(overflow)?,
            ))
        })
        .collect()
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
    let base_shape = format!("n={n}, features={features}, k={k}");
    if n > i32::MAX as usize {
        return Err(AnnError::Overflow(format!(
            "ANN row count {n} exceeds i32 maximum {} ({base_shape})",
            i32::MAX
        )));
    }
    let width = candidate_width(n, k);
    let params = nn_descent_params(n, k).map_err(|error| {
        with_shape(workspace_error("NN-descent parameters", error), &base_shape)
    })?;
    let capacity = params.candidates;
    let shape = format!(
        "{base_shape}, width={width}, trees={}, leaf limit={}",
        params.trees, params.leaf_limit
    );
    let workspace = plan_workspace(n, width, params, features)
        .map_err(|error| with_shape(workspace_error("workspace plan", error), &shape))?;
    // n <= i32::MAX was checked above; width < n and capacity < n follow.
    debug_assert!(u32::try_from(width).is_ok() && u32::try_from(capacity).is_ok());
    let reverse_elements = n
        .checked_mul(width)
        .ok_or_else(|| AnnError::Overflow(format!("ANN reverse size overflows usize ({shape})")))?;
    if reverse_elements > i32::MAX as usize {
        return Err(AnnError::Overflow(format!(
            "ANN reverse prefix {reverse_elements} (n x width) exceeds i32 maximum {} ({shape})",
            i32::MAX
        )));
    }
    let host = HostPlan {
        width,
        params,
        workspace,
    };
    let tile_slots = default_tile_slots(n, &workspace)?;
    check_buffer_elements(n, features, k, &host, tile_slots, AnnJoinRoute::Slots)?;
    Ok(host)
}

// CubeCL addresses buffers with U32 elements. Check only the selected route;
// the device-free shape gate uses the original slot route.
fn check_buffer_elements(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
    tile_slots: usize,
    route: AnnJoinRoute,
) -> Result<(), AnnError> {
    let shape = format!(
        "n={n}, features={features}, k={k}, width={}, trees={}, leaf limit={}",
        host.width, host.params.trees, host.params.leaf_limit
    );
    let table = ann_buffer_table(
        n,
        features,
        k,
        host.width,
        host.workspace.max_candidates_per_class,
        &host.workspace,
        tile_slots,
        route,
    )
    .ok_or_else(|| {
        AnnError::Overflow(format!(
            "ANN buffer element count overflows usize ({shape})"
        ))
    })?;
    for entry in table {
        let elements = entry
            .elements()
            .ok_or_else(|| AnnError::Overflow(format!("ANN buffer size overflows ({shape})")))?;
        if elements > u32::MAX as usize {
            return Err(AnnError::Overflow(format!(
                "ANN {:?} buffer elements {elements} exceeds u32 addressing maximum {} ({shape})",
                entry.buffer,
                u32::MAX
            )));
        }
    }
    Ok(())
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

fn plan_with_host_route(
    client: &cubecl::client::Client,
    n: usize,
    features: usize,
    k: usize,
    host: HostPlan,
    tile_slots: usize,
    override_route: Option<AnnJoinRoute>,
    override_policy: Option<SlotPolicy>,
) -> Result<AnnPlan, AnnError> {
    let memory = &client.properties().memory;
    let max_page = usize::try_from(memory.max_page_size)
        .map_err(|_| AnnError::Capability("device max page size exceeds usize".into()))?;
    let alignment = usize::try_from(memory.alignment)
        .map_err(|_| AnnError::Capability("device alignment exceeds usize".into()))?;
    let n_u32 = u32::try_from(n).expect("ANN row count validated by host_plan");
    let merge = ann_launch_shape(client, DType::F32.size(), n_u32)?;
    let join_route = override_route.unwrap_or_else(|| AnnJoinRoute::for_cpu(merge.cpu_merge));
    let slot_policy = override_policy.unwrap_or_else(|| SlotPolicy::for_cpu(merge.cpu_merge));
    slot_policy.validate(merge.cpu_merge)?;
    if !merge.cpu_merge && join_route == AnnJoinRoute::Owner {
        return Err(AnnError::InvalidInput("ANN owner join requires CPU".into()));
    }
    #[cfg(not(feature = "test-support"))]
    if join_route != AnnJoinRoute::for_cpu(merge.cpu_merge) {
        return Err(AnnError::InvalidInput(
            "ANN join route does not match device".into(),
        ));
    }
    if join_route == AnnJoinRoute::Owner {
        check_buffer_elements(n, features, k, &host, tile_slots, join_route)?;
    }
    let vector_widths = ann_vector_width_policy(client);
    for (stage, bytes) in ann_allocation_bytes(
        n,
        features,
        k,
        &host,
        tile_slots,
        merge.cpu_merge,
        join_route,
    )? {
        check_page_limit(&stage, bytes, max_page, alignment)?;
    }
    // The ID copy covers n * width elements; pair launches cover one tile.
    let independent_units = (n * host.width).max(tile_slots);
    let independent_units = u32::try_from(independent_units).map_err(|_| {
        AnnError::Overflow(format!(
            "ANN independent launch units {independent_units} exceeds u32 maximum {}",
            u32::MAX
        ))
    })?;
    let units_per_cube = crate::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x)).x;
    ann_independent_cube_count(
        independent_units,
        units_per_cube,
        client.properties().hardware.max_cube_count.0,
    )?;
    let (fixed_peak_bytes, _) = fixed_memory_route(
        n,
        features,
        k,
        &host,
        alignment,
        merge.cpu_merge,
        join_route,
    )?;
    let staging_bytes = tile_staging_bytes(tile_slots, alignment, join_route)?;
    Ok(AnnPlan {
        host,
        merge,
        vector_widths,
        join_route,
        slot_policy,
        n,
        features,
        k,
        tile_slots,
        #[cfg(test)]
        owner_test_geometry: None,
        memory: PlannedMemory {
            fixed_peak_bytes,
            staging_bytes,
            usable_bytes: fixed_peak_bytes
                .checked_add(staging_bytes)
                .ok_or_else(|| AnnError::Overflow("ANN planned memory overflows u64".into()))?,
        },
    })
}

fn plan_with_host(
    client: &cubecl::client::Client,
    n: usize,
    features: usize,
    k: usize,
    host: HostPlan,
    tile_slots: usize,
) -> Result<AnnPlan, AnnError> {
    plan_with_host_route(client, n, features, k, host, tile_slots, None, None)
}

#[cfg(test)]
pub(super) fn owner_test_plan(
    client: &cubecl::client::Client,
    n: usize,
    features: usize,
    k: usize,
    tile: Option<usize>,
    ch: usize,
    partitions: u32,
    width: usize,
) -> Result<AnnPlan, AnnError> {
    let host = host_plan(n, features, k)?;
    let caps = tile_caps(client, n, &host)?;
    let tile_slots = plan_tiles(caps, tile.map_or(TileRequest::Default, TileRequest::Exact))?;
    let hw = &client.properties().hardware;
    let units = (std::thread::available_parallelism().map_or(1, |v| v.get()) as u32)
        .min(hw.max_cube_dim.0)
        .min(hw.max_units_per_cube);
    if client.name() != "cpu"
        || ch == 0
        || partitions == 0
        || partitions > units.min(OWNER_D_MAX as u32).min(n as u32)
        || ![1, 4, 8].contains(&width)
        || features % width != 0
    {
        return Err(AnnError::InvalidInput("invalid owner test geometry".into()));
    }
    let mut plan = plan_with_host_route(
        client,
        n,
        features,
        k,
        host,
        tile_slots,
        Some(AnnJoinRoute::Owner),
        None,
    )?;
    let chunks = n
        .div_ceil(ch)
        .max(host.workspace.forest_leaf_count.div_ceil(OWNER_CH_LEAF));
    let launch_chunks = u32::try_from(chunks)
        .map_err(|_| AnnError::Overflow("owner test chunk count exceeds u32".into()))?;
    ann_independent_cube_count(launch_chunks, units, hw.max_cube_count.0)?;
    let elements = chunks
        .checked_mul(OWNER_D_MAX + 1)
        .ok_or_else(|| AnnError::Overflow("owner test offsets overflow".into()))?;
    if elements > u32::MAX as usize {
        return Err(AnnError::Overflow("owner test offsets exceed u32".into()));
    }
    let bytes = elements
        .checked_mul(DType::I32.size())
        .ok_or_else(|| AnnError::Overflow("owner test offsets overflow".into()))?;
    check_page_limit("owner test offsets", bytes, caps.max_page, caps.alignment)?;
    let old = n
        .div_ceil(OWNER_CH)
        .max(host.workspace.forest_leaf_count.div_ceil(OWNER_CH_LEAF))
        * (OWNER_D_MAX + 1)
        * DType::I32.size();
    let extra =
        aligned_bytes(bytes, caps.alignment)?.saturating_sub(aligned_bytes(old, caps.alignment)?);
    plan.memory.fixed_peak_bytes = plan
        .memory
        .fixed_peak_bytes
        .checked_add(extra)
        .ok_or_else(|| AnnError::Overflow("owner test memory overflow".into()))?;
    plan.memory.usable_bytes = plan
        .memory
        .fixed_peak_bytes
        .checked_add(plan.memory.staging_bytes)
        .ok_or_else(|| AnnError::Overflow("owner test memory overflow".into()))?;
    plan.owner_test_geometry = Some((ch, partitions, width));
    Ok(plan)
}

fn ann_vector_width_policy(client: &cubecl::client::Client) -> &'static [usize] {
    if client.properties().hardware.num_cpu_cores.is_some() {
        return &[];
    }
    let has_eight = client.io_optimized_vector_sizes(4).any(|width| width == 8);
    let has_four = client.io_optimized_vector_sizes(4).any(|width| width == 4);
    if has_eight && has_four {
        &[8, 4]
    } else if has_eight {
        &[8]
    } else if has_four {
        &[4]
    } else {
        &[]
    }
}

fn aligned_bytes(bytes: usize, alignment: usize) -> Result<u64, AnnError> {
    if alignment == 0 {
        return Err(AnnError::Capability(
            "device reports zero allocation alignment".into(),
        ));
    }
    bytes
        .checked_add(alignment - 1)
        .and_then(|value| (value / alignment).checked_mul(alignment))
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| AnnError::Overflow("ANN aligned allocation exceeds address space".into()))
}

fn fixed_memory(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
    alignment: usize,
    cpu: bool,
) -> Result<(u64, Vec<(String, u64)>), AnnError> {
    fixed_memory_route(
        n,
        features,
        k,
        host,
        alignment,
        cpu,
        AnnJoinRoute::for_cpu(cpu),
    )
}

fn fixed_memory_route(
    n: usize,
    features: usize,
    k: usize,
    host: &HostPlan,
    alignment: usize,
    cpu: bool,
    route: AnnJoinRoute,
) -> Result<(u64, Vec<(String, u64)>), AnnError> {
    let overflow = || AnnError::Overflow("ANN fixed memory overflows address space".into());
    let table = ann_buffer_table(
        n,
        features,
        k,
        host.width,
        host.workspace.max_candidates_per_class,
        &host.workspace,
        0,
        route,
    )
    .ok_or_else(overflow)?;
    let mut phases = [0u64; 4];
    let mut contributors = Vec::new();
    for entry in table {
        if entry.staging
            || entry.route == BufferRoute::Profile
            || (cpu && entry.route == BufferRoute::Cuda)
        {
            continue;
        }
        let bytes = aligned_bytes(entry.bytes().ok_or_else(overflow)?, alignment)?;
        contributors.push((format!("{:?}", entry.buffer), bytes));
        for (phase, total) in phases.iter_mut().enumerate() {
            if entry.phases & (1 << phase) != 0 {
                *total = total.checked_add(bytes).ok_or_else(overflow)?;
            }
        }
    }
    if cpu {
        phases[1] = phases[1]
            .checked_add(u64::try_from(host.workspace.host_peak_bytes).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        contributors.push((
            "host forest/readback/scratch".into(),
            host.workspace.host_peak_bytes as u64,
        ));
        if route == AnnJoinRoute::Owner {
            let forest_prefix = host
                .workspace
                .forest_leaf_count
                .checked_add(1)
                .and_then(|v| v.checked_mul(8))
                .and_then(|v| u64::try_from(v).ok())
                .ok_or_else(overflow)?;
            let refinement_prefix = n
                .checked_add(1)
                .and_then(|v| v.checked_mul(8))
                .and_then(|v| u64::try_from(v).ok())
                .ok_or_else(overflow)?;
            phases[1] = phases[1].checked_add(forest_prefix).ok_or_else(overflow)?;
            phases[2] = phases[2]
                .checked_add(refinement_prefix)
                .ok_or_else(overflow)?;
            contributors.push(("host forest prefix".into(), forest_prefix));
            contributors.push(("host refinement prefix".into(), refinement_prefix));
        }
    }
    contributors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    Ok((
        *phases.iter().max().expect("four allocation phases"),
        contributors,
    ))
}

fn tile_caps(
    client: &cubecl::client::Client,
    n: usize,
    host: &HostPlan,
) -> Result<TileCaps, AnnError> {
    let memory = &client.properties().memory;
    let max_slots = n
        .checked_mul(host.workspace.pair_slots_per_center)
        .ok_or_else(|| AnnError::Overflow("ANN total pair slots overflow usize".into()))?
        .max(host.workspace.forest_pair_count);
    let cpu_merge = client.name() == "cpu";
    Ok(TileCaps {
        max_slots,
        budget_ceiling_slots: BUDGET_TILE_CEILING_SLOTS,
        max_page: usize::try_from(memory.max_page_size)
            .map_err(|_| AnnError::Capability("device max page exceeds usize".into()))?,
        alignment: usize::try_from(memory.alignment)
            .map_err(|_| AnnError::Capability("device alignment exceeds usize".into()))?,
        units_per_cube: crate::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x)).x,
        max_cubes: client.properties().hardware.max_cube_count.0,
        cpu_merge,
    })
}

fn default_tile_slots(n: usize, workspace: &WorkspacePlan) -> Result<usize, AnnError> {
    let max_slots = n
        .checked_mul(workspace.pair_slots_per_center)
        .ok_or_else(|| AnnError::Overflow("ANN total pair slots overflow usize".into()))?
        .max(workspace.forest_pair_count);
    plan_tiles(
        TileCaps {
            max_slots,
            budget_ceiling_slots: usize::MAX,
            max_page: usize::MAX,
            alignment: 1,
            units_per_cube: 32,
            max_cubes: u32::MAX,
            cpu_merge: false,
        },
        TileRequest::Default,
    )
}

/// Run the existing ANN driver with an explicitly sized pair tile for experiments.
///
/// This support-only entry point accepts an F32 rank-two CubeTensor and returns
/// raw I32 neighbor IDs and F32 distances. The tile is used in both forest and
/// refinement phases; it is validated against device allocation and launch limits.
#[cfg(feature = "test-support")]
pub fn approximate_neighbors_with_tile_slots(
    data: CubeTensor,
    k: usize,
    seed: u64,
    max_iterations: u32,
    squared: bool,
    tile_slots: usize,
) -> Result<(CubeTensor, CubeTensor), AnnError> {
    if data.dtype != DType::F32 || data.rank() != 2 {
        return Err(AnnError::InvalidInput(
            "ANN input must be a rank-two F32 tensor".into(),
        ));
    }
    let [n, features] = data.shape().dims::<2>();
    let host = host_plan(n, features, k)?;
    plan_tiles(
        tile_caps(&data.client, n, &host)?,
        TileRequest::Exact(tile_slots),
    )?;
    let plan = plan_with_host(&data.client, n, features, k, host, tile_slots)?;
    ann_sampled::approximate_neighbors(data, k, seed, max_iterations, squared, plan)
}

/// Shape/capability-only exact tile plan for bitwise invariance tests.
#[cfg(feature = "test-support")]
pub fn ann_preflight_with_tile_slots(
    device: &burn::tensor::Device,
    n: usize,
    features: usize,
    k: usize,
    slots: usize,
) -> Result<AnnPlan, AnnError> {
    let DispatchDevice::Cube(device) = device.as_dispatch();
    let client = device.client();
    let host = host_plan(n, features, k)?;
    plan_tiles(tile_caps(&client, n, &host)?, TileRequest::Exact(slots))?;
    plan_with_host(&client, n, features, k, host, slots)
}

/// Exact tile preflight with an explicit experimental slot policy.
#[cfg(feature = "test-support")]
pub fn ann_preflight_with_slot_policy(
    device: &burn::tensor::Device,
    n: usize,
    features: usize,
    k: usize,
    slots: usize,
    policy: SlotPolicy,
) -> Result<AnnPlan, AnnError> {
    let DispatchDevice::Cube(device) = device.as_dispatch();
    let client = device.client();
    let host = host_plan(n, features, k)?;
    plan_tiles(tile_caps(&client, n, &host)?, TileRequest::Exact(slots))?;
    plan_with_host_route(&client, n, features, k, host, slots, None, Some(policy))
}

/// Public launch preflight. Returns typed errors for invalid shapes and
/// unavailable hardware capability; never launches a kernel.
pub fn ann_preflight(
    device: &burn::tensor::Device,
    n: usize,
    features: usize,
    k: usize,
    budget: &crate::memory::MemoryBudget,
) -> Result<AnnPlan, AnnError> {
    let DispatchDevice::Cube(cube_device) = device.as_dispatch();
    let client = cube_device.client();
    let host = host_plan(n, features, k)?;
    let caps = tile_caps(&client, n, &host)?;
    let (fixed, contributors) =
        fixed_memory(n, features, k, &host, caps.alignment, caps.cpu_merge)?;
    let live = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| AnnError::Overflow("ANN live input size overflows usize".into()))?;
    let usable = budget.usable_bytes(aligned_bytes(live, caps.alignment)?);
    let slots = plan_tiles(caps, TileRequest::Budget { usable_bytes: usable, fixed_bytes: fixed })
        .map_err(|error| match error {
            AnnError::Memory { required_bytes, available_bytes, .. } => {
                let largest = contributors.iter().take(3)
                    .map(|(name, bytes)| format!("{name} {bytes}"))
                    .collect::<Vec<_>>().join(", ");
                let limit = budget.limit_bytes.map_or_else(|| "none".into(), |v| v.to_string());
                AnnError::Memory { required_bytes, available_bytes,
                    message: format!("ANN needs {required_bytes} bytes of device memory but {available_bytes} are available (free {}, total {}, limit {limit}); largest buffers: {largest}; (n={n}, features={features}, k={k})", budget.free_bytes, budget.total_bytes) }
            }
            other => other,
        })?;
    let mut plan = plan_with_host(&client, n, features, k, host, slots)?;
    plan.memory.usable_bytes = usable;
    Ok(plan)
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
    /// Component-wise maximum of owner CPU and slot CUDA runtime, including input.
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
    /// Sampling budget per new/old class, `min(k + 1, 60, n - 1)`. Not the output width.
    pub sample_capacity: usize,
    /// Projection forest trees, `min(8, 5 + round_ties_even(sqrt(n) / 20))`.
    pub trees: usize,
    /// Output edges, `n * k`.
    pub directed_edges: usize,
    /// Phases in execution order: Input, ForestInit, Refinement, Output.
    pub phases: [AnnPhaseBytes; 4],
    /// Maximum host bytes across phases.
    pub host_peak_bytes: usize,
    /// Maximum runtime bytes across phases.
    pub runtime_peak_bytes: usize,
    /// Maximum over phases of host plus CPU-route runtime, excluding CUDA workspaces.
    pub cpu_combined_peak_bytes: usize,
    /// CUDA-conservative peak runtime bytes excluding pair and endpoint staging.
    pub fixed_runtime_peak_bytes: usize,
    /// Unaligned runtime bytes per pair slot across six staging buffers.
    pub staging_bytes_per_slot: usize,
    /// CUDA-conservative fixed runtime peak plus the minimum tile, without alignment rounding.
    pub min_runtime_bytes: usize,
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
    let overflow = || {
        AnnError::Overflow(format!(
            "ANN allocation estimate overflows usize (n={n}, features={features}, k={k})"
        ))
    };
    let mul = |a: usize, b: usize| a.checked_mul(b).ok_or_else(overflow);
    let add = |a: usize, b: usize| a.checked_add(b).ok_or_else(overflow);
    let ws = host.workspace;
    let w = host.width;
    let c = host.workspace.max_candidates_per_class;
    let slots = default_tile_slots(n, &ws)?;
    let table = ann_buffer_table(n, features, k, w, c, &ws, slots, AnnJoinRoute::Slots)
        .ok_or_else(overflow)?;
    let owner_table = ann_buffer_table(n, features, k, w, c, &ws, slots, AnnJoinRoute::Owner)
        .ok_or_else(overflow)?;
    let mut runtime = [0usize; 4];
    let mut cpu_runtime = [0usize; 4];
    let mut fixed = [0usize; 4];
    for entry in &table {
        if entry.route == BufferRoute::Profile {
            continue;
        }
        let bytes = entry.bytes().ok_or_else(overflow)?;
        for phase in 0..4 {
            if entry.phases & (1 << phase) != 0 {
                runtime[phase] = add(runtime[phase], bytes)?;
                if !entry.staging {
                    fixed[phase] = add(fixed[phase], bytes)?;
                }
            }
        }
    }
    for entry in &owner_table {
        if matches!(entry.route, BufferRoute::Profile | BufferRoute::Cuda) {
            continue;
        }
        let bytes = entry.bytes().ok_or_else(overflow)?;
        for (phase, runtime) in cpu_runtime.iter_mut().enumerate() {
            if entry.phases & (1 << phase) != 0 {
                *runtime = add(*runtime, bytes)?;
            }
        }
    }
    let owner_forest_host = add(ws.host_peak_bytes, mul(add(ws.forest_leaf_count, 1)?, 8)?)?;
    let owner_refinement_host = mul(add(n, 1)?, 8)?;
    let input = table
        .iter()
        .find(|entry| entry.buffer == Buffer::Input)
        .and_then(BufferEntry::bytes)
        .ok_or_else(overflow)?;
    let trees = host.params.trees;
    let phases = [
        AnnPhaseBytes {
            phase: AnnPhase::Input,
            host_bytes: mul(input, 2)?,
            runtime_bytes: runtime[0].max(cpu_runtime[0]),
        },
        AnnPhaseBytes {
            phase: AnnPhase::ForestInit,
            host_bytes: owner_forest_host,
            runtime_bytes: runtime[1].max(cpu_runtime[1]),
        },
        AnnPhaseBytes {
            phase: AnnPhase::Refinement,
            host_bytes: owner_refinement_host,
            runtime_bytes: runtime[2].max(cpu_runtime[2]),
        },
        AnnPhaseBytes {
            phase: AnnPhase::Output,
            host_bytes: 0,
            runtime_bytes: runtime[3].max(cpu_runtime[3]),
        },
    ];
    let host_peak_bytes = phases.iter().map(|p| p.host_bytes).max().unwrap_or(0);
    let runtime_peak_bytes = phases.iter().map(|p| p.runtime_bytes).max().unwrap_or(0);
    let cpu_combined_peak_bytes = phases.iter().enumerate().try_fold(0usize, |peak, (i, p)| {
        add(p.host_bytes, cpu_runtime[i]).map(|v| peak.max(v))
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
        fixed_runtime_peak_bytes: *fixed.iter().max().unwrap(),
        staging_bytes_per_slot: usize::try_from(tile_staging_bytes(1, 1, AnnJoinRoute::Slots)?)
            .map_err(|_| overflow())?,
        min_runtime_bytes: add(
            *fixed.iter().max().unwrap(),
            usize::try_from(tile_staging_bytes(
                n.checked_mul(ws.pair_slots_per_center)
                    .ok_or_else(overflow)?
                    .max(ws.forest_pair_count)
                    .min(1 << 20),
                1,
                AnnJoinRoute::Slots,
            )?)
            .map_err(|_| overflow())?,
        )?,
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

/// Set in a `reverse_sources` entry when the source's edge to the target is new
/// (`retained == 0`). Valid ids are below `n <= i32::MAX`, so bit 31 is free.
pub(super) const REVERSE_NEW_BIT: u32 = 1 << 31;

/// Scatter each distinct reverse edge as its source id, tagged with
/// [`REVERSE_NEW_BIT`] from the source slot's flag.
#[cube(launch)]
pub(super) fn reverse_scatter<I: Int>(
    candidates: &Tensor<I>,
    retained: &Tensor<I>,
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
                let mut entry = source as u32;
                if retained[source * width + slot] == I::cast_from(0i32) {
                    entry |= REVERSE_NEW_BIT;
                }
                sources[position] = I::cast_from(entry as i32);
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

/// Execute a checked plan on its original shape and device capabilities.
pub fn approximate_neighbors_planned(
    data: CubeTensor,
    plan: &AnnPlan,
    seed: u64,
    max_iterations: u32,
    squared: bool,
) -> Result<(CubeTensor, CubeTensor), AnnError> {
    if data.dtype != DType::F32
        || data.rank() != 2
        || data.shape().dims::<2>() != [plan.n, plan.features]
    {
        return Err(AnnError::InvalidInput(
            "ANN plan does not match input shape or dtype".into(),
        ));
    }
    let host = host_plan(plan.n, plan.features, plan.k)?;
    plan_tiles(
        tile_caps(&data.client, plan.n, &host)?,
        TileRequest::Exact(plan.tile_slots),
    )?;
    let mut checked = plan_with_host_route(
        &data.client,
        plan.n,
        plan.features,
        plan.k,
        host,
        plan.tile_slots,
        Some(plan.join_route),
        Some(plan.slot_policy),
    )?;
    checked.memory = plan.memory;
    if checked != *plan {
        return Err(AnnError::InvalidInput(
            "ANN plan does not match executable device fields".into(),
        ));
    }
    ann_sampled::approximate_neighbors(data, plan.k, seed, max_iterations, squared, *plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_policy_validation_and_binding_mapping() {
        let per_slot = SlotRecipe::PerSlot;
        let plane = |group, unroll| SlotRecipe::Plane {
            group,
            decode: true,
            unroll,
        };
        let default = SlotPolicy::default();
        assert_eq!(default.select(1), (per_slot, 1));
        assert_eq!(default.select(4), (per_slot, 4));
        assert_eq!(default.select(8), (per_slot, 8));
        assert!(default.validate(true).is_ok());
        assert_eq!(SlotPolicy::for_cpu(true), default);
        let cuda = SlotPolicy::for_cpu(false);
        assert_eq!(cuda.select(1), (per_slot, 1));
        assert_eq!(
            cuda.select(4),
            (
                SlotRecipe::Plane {
                    group: 2,
                    decode: false,
                    unroll: 1
                },
                4
            )
        );
        assert_eq!(
            cuda.select(8),
            (
                SlotRecipe::Plane {
                    group: 2,
                    decode: false,
                    unroll: 1
                },
                4
            )
        );
        assert!(cuda.validate(false).is_ok());
        for group in [1, 4, 8] {
            let policy = SlotPolicy {
                wide: plane(group, 1),
                scalar: plane(group, 2),
            };
            assert!(policy.validate(false).is_ok());
            assert!(matches!(
                policy.validate(true),
                Err(AnnError::InvalidInput(_))
            ));
            assert_eq!(policy.select(1).1, 1);
            assert_eq!(policy.select(4).1, if group == 1 { 4 } else { 1 });
            assert_eq!(policy.select(8).1, if group == 1 { 8 } else { 1 });
        }
        let wide_two = SlotPolicy {
            wide: plane(2, 1),
            scalar: per_slot,
        };
        assert!(wide_two.validate(false).is_ok());
        assert_eq!(wide_two.select(4).1, 4);
        assert_eq!(wide_two.select(8).1, 4);
        for invalid in [
            SlotPolicy {
                wide: per_slot,
                scalar: plane(2, 1),
            },
            SlotPolicy {
                wide: plane(3, 1),
                scalar: per_slot,
            },
            SlotPolicy {
                wide: plane(1, 0),
                scalar: per_slot,
            },
            SlotPolicy {
                wide: plane(1, 3),
                scalar: per_slot,
            },
        ] {
            assert!(matches!(
                invalid.validate(false),
                Err(AnnError::InvalidInput(_))
            ));
        }
    }

    #[test]
    fn owner_route_table_and_estimate_cover_both_routes() {
        assert_eq!(AnnJoinRoute::for_cpu(true), AnnJoinRoute::Owner);
        assert_eq!(AnnJoinRoute::for_cpu(false), AnnJoinRoute::Slots);
        for (n, features, k) in [(11, 2, 3), (4096, 8, 15)] {
            let host = host_plan(n, features, k).unwrap();
            let slots = default_tile_slots(n, &host.workspace).unwrap();
            let table = |route| {
                ann_buffer_table(
                    n,
                    features,
                    k,
                    host.width,
                    host.workspace.max_candidates_per_class,
                    &host.workspace,
                    slots,
                    route,
                )
                .unwrap()
            };
            let owner = table(AnnJoinRoute::Owner);
            let slot = table(AnnJoinRoute::Slots);
            assert!(!owner.iter().any(|e| e.route == BufferRoute::Slots));
            assert!(
                !slot
                    .iter()
                    .any(|e| e.route == BufferRoute::Owner || e.buffer == Buffer::OwnerProfile)
            );
            assert_eq!(owner.iter().filter(|e| e.staging).count(), 4);
            assert_eq!(slot.iter().filter(|e| e.staging).count(), 6);
            let chunks = n
                .div_ceil(OWNER_CH)
                .max(host.workspace.forest_leaf_count.div_ceil(OWNER_CH_LEAF));
            assert_eq!(
                owner
                    .iter()
                    .find(|e| e.buffer == Buffer::OwnerChunkOffsets)
                    .unwrap()
                    .elements(),
                Some(chunks * (OWNER_D_MAX + 1))
            );
            assert_eq!(
                tile_staging_bytes(slots, 1, AnnJoinRoute::Owner).unwrap(),
                (slots * 32) as u64
            );
            assert_eq!(
                tile_staging_bytes(slots, 1, AnnJoinRoute::Slots).unwrap(),
                (slots * 32) as u64
            );
            let estimate = ann_host_estimate(n, features, k).unwrap();
            for (index, phase) in estimate.phases.iter().enumerate() {
                let bytes = [(&owner, true), (&slot, false)].map(|(entries, cpu)| {
                    let bytes: usize = entries
                        .iter()
                        .filter(|e| {
                            e.route != BufferRoute::Profile
                                && (!cpu || e.route != BufferRoute::Cuda)
                                && e.phases & (1 << index) != 0
                        })
                        .map(|e| e.bytes().unwrap())
                        .sum();
                    bytes
                });
                assert_eq!(phase.runtime_bytes, bytes[0].max(bytes[1]));
            }
            assert_eq!(
                estimate.phases[1].host_bytes,
                host.workspace.host_peak_bytes + (host.workspace.forest_leaf_count + 1) * 8
            );
            assert_eq!(estimate.phases[2].host_bytes, (n + 1) * 8);
            assert_eq!(estimate.phases[0].host_bytes, 2 * n * features * 4);
            assert_eq!(estimate.phases[3].host_bytes, 0);
        }
        assert!(staging_entries(usize::MAX, AnnJoinRoute::Owner).is_none());
    }

    fn ample_budget() -> crate::memory::MemoryBudget {
        crate::memory::MemoryBudget {
            total_bytes: u64::MAX,
            free_bytes: u64::MAX,
            limit_bytes: None,
        }
    }

    #[test]
    fn cpu_combined_forest_peak_rejects_runtime_only_budget() {
        let (n, features, k) = (64, 1024, 8);
        let host = host_plan(n, features, k).unwrap();
        let caps = TileCaps {
            max_slots: (n * host.workspace.pair_slots_per_center)
                .max(host.workspace.forest_pair_count),
            budget_ceiling_slots: BUDGET_TILE_CEILING_SLOTS,
            max_page: usize::MAX,
            alignment: 1,
            units_per_cube: 32,
            max_cubes: u32::MAX,
            cpu_merge: true,
        };
        let runtime_fixed = fixed_memory(n, features, k, &host, caps.alignment, false)
            .unwrap()
            .0;
        let cpu_fixed = fixed_memory(n, features, k, &host, caps.alignment, true)
            .unwrap()
            .0;
        let staging = tile_staging_bytes(
            caps.max_slots.min(1 << 20),
            caps.alignment,
            AnnJoinRoute::Owner,
        )
        .unwrap();
        let runtime_min = runtime_fixed + staging;
        let cpu_min = cpu_fixed + staging;
        assert!(cpu_min > runtime_min);
        let usable = runtime_min + (cpu_min - runtime_min) / 2;
        assert!(
            plan_tiles(
                caps,
                TileRequest::Budget {
                    usable_bytes: usable,
                    fixed_bytes: runtime_fixed,
                }
            )
            .is_ok()
        );
        match plan_tiles(
            caps,
            TileRequest::Budget {
                usable_bytes: usable,
                fixed_bytes: cpu_fixed,
            },
        ) {
            Err(AnnError::Memory {
                required_bytes,
                available_bytes,
                ..
            }) => {
                assert_eq!(required_bytes, cpu_min);
                assert_eq!(available_bytes, usable);
                assert!(runtime_min <= available_bytes && available_bytes < required_bytes);
            }
            other => panic!("expected combined CPU memory rejection, got {other:?}"),
        }
    }

    #[test]
    fn budget_tile_alignment_minimum_and_caps() {
        let caps = TileCaps {
            max_slots: 8_000_000,
            budget_ceiling_slots: BUDGET_TILE_CEILING_SLOTS,
            max_page: usize::MAX,
            alignment: 512,
            units_per_cube: 32,
            max_cubes: u32::MAX,
            cpu_merge: false,
        };
        let minimum = 1 << 20;
        let fixed_bytes = 123;
        let minimum_bytes =
            tile_staging_bytes(minimum, caps.alignment, AnnJoinRoute::Slots).unwrap();
        assert!(matches!(
            plan_tiles(
                caps,
                TileRequest::Budget {
                    usable_bytes: fixed_bytes + minimum_bytes - 1,
                    fixed_bytes,
                }
            ),
            Err(AnnError::Memory { .. })
        ));
        assert_eq!(
            plan_tiles(
                caps,
                TileRequest::Budget {
                    usable_bytes: fixed_bytes + minimum_bytes,
                    fixed_bytes,
                }
            )
            .unwrap(),
            minimum
        );
        let q = minimum + 129;
        let bytes = tile_staging_bytes(q, caps.alignment, AnnJoinRoute::Slots).unwrap();
        assert!(
            plan_tiles(
                caps,
                TileRequest::Budget {
                    usable_bytes: fixed_bytes + bytes - 1,
                    fixed_bytes
                }
            )
            .unwrap()
                < q
        );
        assert!(
            plan_tiles(
                caps,
                TileRequest::Budget {
                    usable_bytes: fixed_bytes + bytes,
                    fixed_bytes
                }
            )
            .unwrap()
                >= q
        );
        assert_eq!(plan_tiles(caps, TileRequest::Default).unwrap(), 1 << 22);
        assert!(plan_tiles(caps, TileRequest::Exact(minimum - 1)).is_err());
        let limited = TileCaps {
            max_cubes: minimum as u32 / 32,
            ..caps
        };
        assert_eq!(
            plan_tiles(limited, TileRequest::Exact(minimum)).unwrap(),
            minimum
        );
        assert!(plan_tiles(limited, TileRequest::Exact(minimum + 1)).is_err());
    }

    #[test]
    fn budget_ceiling_is_separate_from_validity_limits() {
        let cuda = TileCaps {
            max_slots: 1 << 27,
            budget_ceiling_slots: BUDGET_TILE_CEILING_SLOTS,
            max_page: usize::MAX,
            alignment: 1,
            units_per_cube: 32,
            max_cubes: u32::MAX,
            cpu_merge: false,
        };
        let ample = TileRequest::Budget {
            usable_bytes: u64::MAX,
            fixed_bytes: 123,
        };
        assert_eq!(plan_tiles(cuda, ample).unwrap(), BUDGET_TILE_CEILING_SLOTS);

        let q = (1 << 22) + 129;
        let bytes = tile_staging_bytes(q, cuda.alignment, AnnJoinRoute::Slots).unwrap();
        assert!(tile_staging_bytes(q + 1, cuda.alignment, AnnJoinRoute::Slots).unwrap() > bytes);
        assert_eq!(
            plan_tiles(
                cuda,
                TileRequest::Budget {
                    usable_bytes: 123 + bytes,
                    fixed_bytes: 123,
                }
            )
            .unwrap(),
            q
        );

        let cpu = TileCaps {
            cpu_merge: true,
            ..cuda
        };
        assert_eq!(plan_tiles(cpu, ample).unwrap(), BUDGET_TILE_CEILING_SLOTS);
        assert_eq!(
            plan_tiles(cpu, TileRequest::Exact(1 << 26)).unwrap(),
            1 << 26
        );
        assert!(plan_tiles(cpu, TileRequest::Exact((1 << 26) + 1)).is_err());
        assert_eq!(plan_tiles(cpu, TileRequest::Default).unwrap(), 1 << 22);
        assert_eq!(
            plan_tiles(cuda, TileRequest::Exact(1 << 26)).unwrap(),
            1 << 26
        );
        assert!(plan_tiles(cuda, TileRequest::Exact((1 << 27) + 1)).is_err());
        assert_eq!(plan_tiles(cuda, TileRequest::Default).unwrap(), 1 << 22);
    }

    #[cfg(feature = "test-support")]
    #[test]
    fn forced_tile_checks_endpoint_limits_and_allocation_sizes() {
        let host = host_plan(40_000, 32, 14).unwrap();
        let caps = TileCaps {
            max_slots: 2_000_000_000,
            budget_ceiling_slots: BUDGET_TILE_CEILING_SLOTS,
            max_page: usize::MAX,
            alignment: 1,
            units_per_cube: 32,
            max_cubes: u32::MAX,
            cpu_merge: false,
        };
        assert!(plan_tiles(caps, TileRequest::Exact(0)).is_err());
        assert!(plan_tiles(caps, TileRequest::Exact(usize::MAX)).is_err());
        assert!(plan_tiles(caps, TileRequest::Exact(i32::MAX as usize / 2 + 1)).is_err());
        assert_eq!(
            plan_tiles(caps, TileRequest::Exact(1 << 24)).unwrap(),
            1 << 24
        );
        let allocations =
            ann_allocation_bytes(40_000, 32, 14, &host, 1 << 24, false, AnnJoinRoute::Slots)
                .unwrap();
        let (_, endpoint_bytes) = allocations
            .iter()
            .find(|(name, _)| *name == "EndpointIds")
            .unwrap();
        assert_eq!(*endpoint_bytes, 1 << 27);
        assert!(check_page_limit("endpoint staging", *endpoint_bytes, (1 << 27) - 1, 1).is_err());
        assert!(ann_independent_cube_count(1 << 24, 32, (1 << 19) - 1).is_err());
    }

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
    fn host_plan_accepts_forest_prefix_above_i32() {
        // P1 has 8 trees and 256-row leaves here. The independent midpoint
        // recurrence in ann_forest.rs establishes that this is the first n
        // whose forest pair prefix exceeds i32::MAX.
        let host = host_plan(2_974_013, 2, 51).unwrap();
        assert_eq!(host.workspace.forest_pair_count, 2_147_483_912);
        assert!(ann_host_estimate(2_974_013, 2, 51).is_ok());
    }

    #[test]
    fn host_estimate_recipe_matches_nn_descent_params() {
        for (n, features, k) in [(67, 3, 5), (1_000, 32, 75), (2_501, 8, 29)] {
            let params = nn_descent_params(n, k).unwrap();
            let estimate = ann_host_estimate(n, features, k).unwrap();
            assert_eq!(estimate.trees, params.trees);
            assert_eq!(estimate.sample_capacity, params.candidates);
        }
    }

    fn overflow_message(n: usize, features: usize, k: usize) -> String {
        match ann_host_estimate(n, features, k) {
            Err(AnnError::Overflow(message)) => message,
            other => panic!("expected AnnError::Overflow, got {other:?}"),
        }
    }

    #[test]
    fn host_estimate_accepts_former_join_ceiling_shapes() {
        // P1's shape-derived join totals are carried by the I64 join prefix.
        for k in [14, 29] {
            let estimate = ann_host_estimate(1_500_000, 256, k).unwrap();
            assert_eq!(estimate.trees, 8);
            let host = host_plan(1_500_000, 256, k).unwrap();
            let n = 1_500_000usize;
            let width = k + 4;
            let c = k + 1;
            let chunks = n.div_ceil(128);
            let input = 2 * 4 * n * 256;
            let staging = 32 * (1 << 22);
            // IDs, distances, two flag buffers, working IDs, reverse sources,
            // and two CUDA workspaces; forest membership is already released.
            // The n-sized I32/F32 scratch includes the two row-worst arrays;
            // the sampled pools hold ids only (`2c` per row, no priorities).
            let heap = 32 * n * width;
            let scratch =
                4 * (8 * n + 2 * (n + 1) + 2 * n * c + chunks + 1) + 8 * (n + (n + 1) + chunks + 1);
            assert_eq!(estimate.retained_width, width);
            assert_eq!(host.width, width);
            assert_eq!(
                estimate.phases[2].runtime_bytes,
                input + heap + scratch + staging
            );
        }
    }

    #[test]
    fn membership_gate_is_exact_at_its_boundary() {
        // 8 trees x 268_435_455 = 2_147_483_640 <= i32::MAX; one more row exceeds it.
        let n = 268_435_455;
        let params = nn_descent_params(n, 1).unwrap();
        assert!(plan_workspace(n, candidate_width(n, 1), params, 1).is_ok());
        assert!(ann_host_estimate(n, 1, 1).is_ok());
        let message = overflow_message(268_435_456, 1, 1);
        for part in [
            "forest membership",
            "2147483648",
            "2147483647",
            "n=268435456",
        ] {
            assert!(message.contains(part), "{part} missing: {message}");
        }
    }

    #[test]
    fn owner_element_limit_does_not_restrict_device_free_estimate() {
        let fits = |n| {
            let host = host_plan(n, 1, 1).unwrap();
            let slots = default_tile_slots(n, &host.workspace).unwrap();
            check_buffer_elements(n, 1, 1, &host, slots, AnnJoinRoute::Owner).is_ok()
        };
        let (mut low, mut high) = (2usize, 268_435_455usize);
        assert!(fits(low));
        assert!(!fits(high));
        while low < high {
            let mid = low + (high - low).div_ceil(2);
            if fits(mid) {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        let host = host_plan(low + 1, 1, 1).unwrap();
        let slots = default_tile_slots(low + 1, &host.workspace).unwrap();
        let error =
            check_buffer_elements(low + 1, 1, 1, &host, slots, AnnJoinRoute::Owner).unwrap_err();
        assert!(format!("{error:?}").contains("OwnerChunkOffsets"));
        assert!(ann_host_estimate(low, 1, 1).is_ok());
        assert!(ann_host_estimate(low + 1, 1, 1).is_ok());
        assert!(check_buffer_elements(low + 1, 1, 1, &host, slots, AnnJoinRoute::Slots).is_ok());
    }

    #[test]
    fn host_capacity_limits_match_independent_buffer_boundaries() {
        let i32_limit = i32::MAX as usize;
        let u32_limit = u32::MAX as usize;
        for k in [1usize, 14, 29, 59, 255] {
            for features in [1usize, 32, 256] {
                let width = k + 4;
                let candidates = (k + 1).min(60);
                // Independently bound every relevant I32/U32-addressed shape:
                // rows, forest membership, reverse/heap rows, input, sampled IDs,
                // and output.  All resulting n exceed k+4, so width and capacity
                // are constant at both adjacent boundary points.
                let limit = [
                    i32_limit,
                    i32_limit / 8,
                    i32_limit / width,
                    u32_limit / features,
                    u32_limit / (2 * candidates),
                    u32_limit / k,
                ]
                .into_iter()
                .min()
                .unwrap();
                assert!(
                    host_plan(limit, features, k).is_ok(),
                    "expected n={limit} to fit for features={features}, k={k}"
                );
                assert!(
                    host_plan(limit + 1, features, k).is_err(),
                    "expected n={} to exceed a checked limit for features={features}, k={k}",
                    limit + 1
                );
            }
        }
    }

    #[test]
    fn addressing_gate_is_exact_at_its_boundary() {
        // 1_500_000 x 2_863 = 4_294_500_000 <= u32::MAX; x 2_864 = 4_296_000_000.
        assert!(ann_host_estimate(1_500_000, 2_863, 14).is_ok());
        let message = overflow_message(1_500_000, 2_864, 14);
        for part in ["Input", "4296000000", "4294967295", "features=2864"] {
            assert!(message.contains(part), "{part} missing: {message}");
        }
    }

    #[test]
    fn host_plan_row_count_and_reverse_messages_name_values_and_limits() {
        let message = overflow_message(i32::MAX as usize + 1, 1, 1);
        assert!(message.contains("row count 2147483648"), "{message}");
        assert!(message.contains("2147483647"), "{message}");
        assert!(message.contains("features=1, k=1"), "{message}");
        // 65536 x width 65535 = 4_294_901_760 > i32::MAX; the single leaf's
        // 65536*65535/2 pairs stay within the per-leaf gate.
        let message = overflow_message(65_536, 1, 65_535);
        for part in ["reverse prefix", "4294901760", "2147483647", "width=65535"] {
            assert!(message.contains(part), "{part} missing: {message}");
        }
    }

    fn assert_preflight_accepts_former_ceiling(device: &burn::tensor::Device) {
        for k in [14, 29] {
            let plan = ann_preflight(device, 1_500_000, 256, k, &ample_budget()).unwrap();
            let shape = host_plan(1_500_000, 256, k).unwrap();
            assert_eq!(
                (
                    plan.host.width,
                    plan.host.workspace.max_candidates_per_class
                ),
                (shape.width, shape.workspace.max_candidates_per_class)
            );
            assert!(plan.tile_slots >= 1 << 20);
            assert_eq!(
                plan.host.workspace.forest_pair_count,
                shape.workspace.forest_pair_count
            );
        }
    }

    #[test]
    fn preflight_accepts_former_join_ceiling_shape_on_cpu() {
        let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
        assert_preflight_accepts_former_ceiling(&device);
    }

    #[test]
    fn preflight_accepts_former_join_ceiling_shape_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            return;
        };
        assert_preflight_accepts_former_ceiling(&device);
    }

    #[test]
    fn workspace_capacity_error_names_stage_quantity_value_and_limit() {
        let error = workspace_error(
            "workspace plan",
            WorkspaceError::Capacity {
                quantity: "forest membership length (trees x rows)",
                value: 4_294_967_294,
                limit: i32::MAX as usize,
            },
        );
        let AnnError::Overflow(message) = error else {
            panic!("expected overflow");
        };
        assert_eq!(
            message,
            "ANN workspace plan: forest membership length (trees x rows) 4294967294 exceeds 2147483647"
        );
    }

    #[test]
    fn host_estimate_counts_forest_terminal_metadata_and_phase_totals() {
        // n=11, d=2, k=3: width 7, capacity 4, 5 trees, one leaf per tree.
        let estimate = ann_host_estimate(11, 2, 3).unwrap();
        assert_eq!(estimate.retained_width, 7);
        assert_eq!(estimate.sample_capacity, 4);
        assert_eq!(estimate.trees, 5);
        assert_eq!(estimate.directed_edges, 33);
        let [input, forest_init, refinement, output] = estimate.phases;
        // Original input plus the conservatively reserved contiguous copy: 2*11*2*4.
        assert_eq!(input.runtime_bytes, 176);
        assert_eq!(forest_init.runtime_bytes, 11_600);
        assert_eq!(refinement.runtime_bytes, 13_012);
        assert_eq!(output.runtime_bytes, 1_056);
        let host = host_plan(11, 2, 3).unwrap();
        let table = ann_buffer_table(
            11,
            2,
            3,
            host.width,
            host.workspace.max_candidates_per_class,
            &host.workspace,
            275,
            AnnJoinRoute::Slots,
        )
        .unwrap();
        assert_eq!(table.iter().filter(|entry| entry.staging).count(), 6);
        assert!(
            estimate.cpu_combined_peak_bytes
                <= estimate
                    .phases
                    .iter()
                    .map(|phase| phase.host_bytes + phase.runtime_bytes)
                    .max()
                    .unwrap()
        );
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
        // n=11, d=2, k=3: width 7, capacity 4, one leaf per tree, 5 trees.
        // The forest pair prefix selects the 275-slot default tile. Each endpoint
        // buffer has 2*275 elements = 2200 bytes, the largest single allocation.
        let host = host_plan(11, 2, 3).unwrap();
        let entries =
            ann_allocation_bytes(11, 2, 3, &host, 275, false, AnnJoinRoute::Slots).unwrap();
        let largest = entries.iter().map(|(_, bytes)| *bytes).max().unwrap();
        assert_eq!(largest, 2_200);
        assert!(
            entries
                .iter()
                .any(|(stage, bytes)| { *stage == "ForestRows" && *bytes == 220 })
        );
        assert!(
            entries
                .iter()
                .any(|(stage, bytes)| *stage == "OutputIds" && *bytes == 132)
        );
        // A page limit one byte below the largest entry rejects the whole plan.
        assert!(check_page_limit("endpoint staging", largest, largest - 1, 4).is_err());
    }

    #[test]
    fn allocation_table_is_unique_and_matches_estimate_phases() {
        let host = host_plan(11, 2, 3).unwrap();
        let table = ann_buffer_table(
            11,
            2,
            3,
            host.width,
            host.workspace.max_candidates_per_class,
            &host.workspace,
            275,
            AnnJoinRoute::Slots,
        )
        .unwrap();
        assert_eq!(table.len(), 39);
        let mut seen = std::collections::HashSet::new();
        for entry in &table {
            assert!(seen.insert(entry.buffer));
            assert!(entry.bytes().is_some());
            if matches!(
                entry.buffer,
                Buffer::ForestRows | Buffer::ForestLeafRows | Buffer::ForestPairPrefix
            ) {
                assert_eq!(entry.phases, 2);
            }
            if matches!(
                entry.buffer,
                Buffer::MergeWorkspaceIds | Buffer::MergeWorkspaceDistances
            ) {
                assert_eq!(entry.route, BufferRoute::Cuda);
                assert_eq!(entry.phases, 6);
            }
        }
        assert_eq!(table.iter().filter(|entry| entry.staging).count(), 6);
        let estimate = estimate_from_plan(11, 2, 3, &host).unwrap();
        let owner_table = ann_buffer_table(
            11,
            2,
            3,
            host.width,
            host.workspace.max_candidates_per_class,
            &host.workspace,
            275,
            AnnJoinRoute::Owner,
        )
        .unwrap();
        for (index, phase) in estimate.phases.iter().enumerate() {
            let expected: usize = table
                .iter()
                .filter(|entry| {
                    entry.route != BufferRoute::Profile && entry.phases & (1 << index) != 0
                })
                .map(|entry| entry.bytes().unwrap())
                .sum();
            let cpu: usize = owner_table
                .iter()
                .filter(|entry| {
                    !matches!(entry.route, BufferRoute::Cuda | BufferRoute::Profile)
                        && entry.phases & (1 << index) != 0
                })
                .map(|entry| entry.bytes().unwrap())
                .sum();
            assert_eq!(phase.runtime_bytes, expected.max(cpu));
        }
        assert_eq!(estimate.phases[0].runtime_bytes, 176);
        let cpu_peak = estimate
            .phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                phase.host_bytes
                    + owner_table
                        .iter()
                        .filter(|entry| {
                            !matches!(entry.route, BufferRoute::Cuda | BufferRoute::Profile)
                                && entry.phases & (1 << index) != 0
                        })
                        .map(|entry| entry.bytes().unwrap())
                        .sum::<usize>()
            })
            .max()
            .unwrap();
        assert_eq!(estimate.cpu_combined_peak_bytes, cpu_peak);
        assert_eq!(estimate.staging_bytes_per_slot, 32);
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
            let preflight = ann_preflight(&device, n, features, k, &ample_budget()).unwrap();
            let host = host_plan(n, features, k).unwrap();
            plan_tiles(
                tile_caps(&client, n, &host).unwrap(),
                TileRequest::Exact(preflight.tile_slots),
            )
            .unwrap();
            let mut in_op =
                plan_with_host(&client, n, features, k, host, preflight.tile_slots).unwrap();
            in_op.memory = preflight.memory;
            assert_eq!(preflight, in_op);
            let capped = crate::memory::MemoryBudget {
                limit_bytes: Some(512 << 20),
                ..ample_budget()
            };
            let mut other = ann_preflight(&device, n, features, k, &capped).unwrap();
            assert_ne!(preflight.memory.usable_bytes, other.memory.usable_bytes);
            other.memory = preflight.memory;
            assert_eq!(
                preflight, other,
                "reporting-only budgets must not change capped execution"
            );
        }
    }

    #[test]
    fn preflight_rejects_tiny_n_and_invalid_k_through_device() {
        let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
        assert!(matches!(
            ann_preflight(&device, 1, 3, 1, &ample_budget()),
            Err(AnnError::InvalidInput(_))
        ));
        assert!(matches!(
            ann_preflight(&device, 5, 3, 5, &ample_budget()),
            Err(AnnError::InvalidInput(_))
        ));
    }

    #[test]
    fn planned_execution_rejects_shape_tile_and_launch_mismatches() {
        let device = crate::construct_cpu_device(burn::tensor::Device::cpu).unwrap();
        let DispatchDevice::Cube(cube) = device.as_dispatch();
        let client = cube.client();
        // Deliberately undersized backing: reaching forest readback would produce
        // a workspace/read error, not the required pre-execution InvalidInput.
        let data = CubeTensor::new_contiguous(
            client.clone(),
            cube.clone(),
            burn::tensor::Shape::new([64, 8]),
            client.empty(4),
            DType::F32,
        );
        let plan = ann_preflight(&device, 64, 8, 8, &ample_budget()).unwrap();
        let mut bad_shape = plan;
        bad_shape.n += 1;
        let mut bad_tile = plan;
        bad_tile.tile_slots = 0;
        let mut bad_launch = plan;
        bad_launch.merge.cubes += 1;
        let mut bad_width = plan;
        bad_width.host.width += 1;
        let mut bad_vector_widths = plan;
        bad_vector_widths.vector_widths = if plan.vector_widths.is_empty() {
            &[4]
        } else {
            &[]
        };
        for bad in [
            bad_shape,
            bad_tile,
            bad_launch,
            bad_width,
            bad_vector_widths,
        ] {
            assert!(matches!(
                approximate_neighbors_planned(data.clone(), &bad, 42, 1, false),
                Err(AnnError::InvalidInput(_))
            ));
        }
    }
}
