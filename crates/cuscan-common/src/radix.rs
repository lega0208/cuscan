//! Stable full-key device argsort for finite `f32` values.
//!
//! Keys are the monotone `u32` image of each finite float (signed zeros share one
//! key), sorted by least-significant-digit radix passes. Every executed pass is
//! reduce-then-scan: per-cube digit histograms stored digit-major as
//! `table[digit * cubes + cube]`, a hierarchical exclusive scan of that table
//! (yielding global digit bases and preceding-cube totals at once), then a stable
//! scatter. Each cube owns one consecutive input chunk and walks its tiles in
//! input order, carrying per-digit counts across tiles, so equal keys keep input
//! order within and across cubes. No cross-cube forward progress is assumed.
//!
//! The serial path (CubeCL CPU runtime) runs one unit per cube. The cooperative
//! path (plane-capable GPU runtimes such as CUDA) ranks a tile of `cube_dim` keys
//! with `plane_ballot` peer masks, lower-lane popcounts and shared-memory
//! prefixes over preceding planes and prior tiles.
//!
//! Finiteness and pass skipping are decided by a single status readback after
//! key encoding and a global digit histogram for every pass (per-cube counts
//! reduced over cubes; counts are order invariant, so they are computed once
//! from the encoded keys). The lowest non-finite index is a numerical error
//! before any pass runs, and a pass is skipped exactly when its global
//! histogram places every row in one bucket. Skipped passes leave the current
//! ping-pong slot unchanged.
#![allow(clippy::too_many_arguments)]
use std::fmt;

use burn::tensor::{DType, Shape};
use burn_cubecl::tensor::CubeTensor;
use cubecl::client::Client;
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::memory::MemoryBudget;

/// Widest digit accepted; larger tables do not fit the conservative plan.
pub const RADIX_MAX_DIGIT_BITS: u32 = 11;
/// Rows handled serially by one CPU cube before another cube is added.
pub const RADIX_CPU_MIN_CHUNK_ROWS: u32 = 16_384;
/// Elements scanned serially by one CPU scan cube.
pub const RADIX_CPU_SCAN_BLOCK: usize = 4_096;
/// Deepest supported hierarchical-scan level count.
pub const RADIX_MAX_SCAN_LEVELS: usize = 8;
const NO_INVALID_INDEX: u32 = u32::MAX;

/// Execution strategy, derived from the client and never switched silently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RadixPath {
    /// One unit per cube; serial per-cube histogram and scatter (CPU runtime).
    SerialCpu,
    /// Plane-ballot ranking with shared memory (CUDA and other plane runtimes).
    CooperativePlane,
}

/// Untuned launch parameters. Different valid parameters produce identical output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadixSortConfig {
    /// Bits per LSD digit, `1..=RADIX_MAX_DIGIT_BITS`.
    pub digit_bits: u32,
    /// Upper bound on cubes owning input chunks.
    pub max_cubes: u32,
    /// Units per cube; must be 1 on the serial path and a plane multiple otherwise.
    pub cube_dim: u32,
    /// Serial path only: minimum rows per cube chunk.
    pub cpu_min_chunk_rows: u32,
}

impl RadixSortConfig {
    /// Conservative, unmeasured defaults; numeric performance tuning is pending.
    pub fn conservative(path: RadixPath) -> Self {
        match path {
            RadixPath::SerialCpu => Self {
                digit_bits: 8,
                max_cubes: 64,
                cube_dim: 1,
                cpu_min_chunk_rows: RADIX_CPU_MIN_CHUNK_ROWS,
            },
            RadixPath::CooperativePlane => Self {
                digit_bits: 8,
                max_cubes: 1_024,
                cube_dim: 256,
                cpu_min_chunk_rows: RADIX_CPU_MIN_CHUNK_ROWS,
            },
        }
    }
}

/// Device limits consulted by planning; constructible directly for limit tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadixDeviceLimits {
    pub path: RadixPath,
    pub alignment: u64,
    pub max_allocation_bytes: u64,
    pub max_cube_count_x: u32,
    pub max_units_per_cube: u32,
    pub max_cube_dim_x: u32,
    pub plane_ops: bool,
    pub plane_size_min: u32,
    pub plane_size_max: u32,
    pub max_shared_memory_bytes: usize,
}

impl RadixDeviceLimits {
    pub fn from_client(client: &Client) -> Self {
        use cubecl::features::Plane;
        let properties = client.properties();
        let hardware = &properties.hardware;
        Self {
            path: if client.name() == "cpu" {
                RadixPath::SerialCpu
            } else {
                RadixPath::CooperativePlane
            },
            alignment: properties.memory.alignment,
            max_allocation_bytes: properties.memory.max_page_size,
            max_cube_count_x: hardware.max_cube_count.0,
            max_units_per_cube: hardware.max_units_per_cube,
            max_cube_dim_x: hardware.max_cube_dim.0,
            plane_ops: client.features().plane.contains(Plane::Ops),
            plane_size_min: hardware.plane_size_min,
            plane_size_max: hardware.plane_size_max,
            max_shared_memory_bytes: hardware.max_shared_memory_size,
        }
    }
}

/// Defined failures; none trigger a backend or host-sort fallback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RadixSortError {
    InvalidInput(String),
    /// First (lowest) row holding NaN or an infinity.
    NonFinite {
        index: usize,
    },
    Capability(String),
    Overflow(String),
    Allocation {
        buffer: &'static str,
        required_bytes: u64,
        limit_bytes: u64,
    },
    Workspace(String),
    Device(String),
}

impl fmt::Display for RadixSortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(m)
            | Self::Capability(m)
            | Self::Overflow(m)
            | Self::Workspace(m)
            | Self::Device(m) => f.write_str(m),
            Self::NonFinite { index } => {
                write!(f, "radix argsort input row {index} is not finite")
            }
            Self::Allocation {
                buffer,
                required_bytes,
                limit_bytes,
            } => write!(
                f,
                "radix argsort buffer {buffer} needs {required_bytes} aligned bytes; \
                 device allocation limit is {limit_bytes}"
            ),
        }
    }
}

impl std::error::Error for RadixSortError {}

/// One individually allocated device buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixAllocation {
    pub buffer: &'static str,
    pub elements: usize,
    pub bytes: u64,
    pub aligned_bytes: u64,
}

/// Checked launch geometry and scratch accounting.
///
/// `cubes`, `chunk_len`, `table_len` and `scan_levels` describe a sort of exactly
/// `rows`; `allocations` are sized for any row count up to `rows`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RadixSortPlan {
    pub path: RadixPath,
    pub rows: usize,
    pub digit_bits: u32,
    pub radix: usize,
    pub passes: u32,
    pub cube_dim: u32,
    pub planes_per_cube: u32,
    pub cubes: u32,
    pub cube_capacity: u32,
    pub chunk_len: usize,
    pub table_len: usize,
    pub table_capacity: usize,
    pub scan_block: usize,
    /// Block-sum lengths per scan level for `table_len`; the last is 1.
    pub scan_levels: Vec<usize>,
    /// Block-sum lengths per scan level for `table_capacity`.
    pub scan_capacity: Vec<usize>,
    pub shared_bytes: usize,
    /// Rows per ownership granule (serial chunk minimum or cube_dim).
    pub granule: usize,
    /// `min(config.max_cubes, max_cube_count_x)`.
    pub max_cubes: u32,
    /// Status words: `[0]` lowest non-finite row, then `passes * radix` global
    /// digit counts at `1 + pass * radix + digit`.
    pub status_words: usize,
    /// Host bytes received by the single per-sort status readback.
    pub host_status_bytes: u64,
    pub allocations: Vec<RadixAllocation>,
    pub total_aligned_bytes: u64,
    pub max_aligned_allocation_bytes: u64,
}

fn overflow(what: &str) -> RadixSortError {
    RadixSortError::Overflow(format!("radix argsort {what} overflows"))
}

fn capability(message: impl Into<String>) -> RadixSortError {
    RadixSortError::Capability(message.into())
}

#[cfg(test)]
fn scan_lengths(len: usize, block: usize) -> Vec<usize> {
    let (levels, count) = scan_levels_fixed(len, block).unwrap();
    levels[..count].to_vec()
}

/// Block-sum lengths per scan level, ending at 1; no heap allocation.
fn scan_levels_fixed(
    mut len: usize,
    block: usize,
) -> Result<([usize; RADIX_MAX_SCAN_LEVELS], usize), RadixSortError> {
    let mut levels = [0usize; RADIX_MAX_SCAN_LEVELS];
    let mut count = 0;
    loop {
        if count == RADIX_MAX_SCAN_LEVELS {
            return Err(capability("radix scan hierarchy deeper than supported"));
        }
        len = len.div_ceil(block).max(1);
        levels[count] = len;
        count += 1;
        if len == 1 {
            return Ok((levels, count));
        }
    }
}

/// Launch geometry for one sort size; `Copy` and allocation-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadixGeometry {
    pub cubes: u32,
    pub chunk_len: usize,
    pub table_len: usize,
    pub scan_levels: [usize; RADIX_MAX_SCAN_LEVELS],
    pub scan_level_count: usize,
}

impl RadixGeometry {
    pub fn levels(&self) -> &[usize] {
        &self.scan_levels[..self.scan_level_count]
    }
}

/// Geometry for `rows` given validated granule, cube bound, radix and scan block.
fn radix_geometry(
    rows: usize,
    granule: usize,
    max_cubes: usize,
    radix: usize,
    scan_block: usize,
) -> Result<RadixGeometry, RadixSortError> {
    let rows_or_one = rows.max(1);
    let granules = rows_or_one.div_ceil(granule);
    let owners = granules.min(max_cubes).max(1);
    let chunk_len = granules
        .div_ceil(owners)
        .checked_mul(granule)
        .ok_or_else(|| overflow("chunk length"))?;
    if chunk_len > u32::MAX as usize {
        return Err(overflow("chunk length as u32"));
    }
    let cubes = rows_or_one.div_ceil(chunk_len);
    let table_len = radix
        .checked_mul(cubes)
        .ok_or_else(|| overflow("histogram table length"))?;
    let (scan_levels, scan_level_count) = scan_levels_fixed(table_len, scan_block)?;
    Ok(RadixGeometry {
        cubes: cubes as u32,
        chunk_len,
        table_len,
        scan_levels,
        scan_level_count,
    })
}

/// Plan a sort against explicit limits without touching a device.
pub fn radix_sort_plan_for_limits(
    rows: usize,
    config: RadixSortConfig,
    limits: RadixDeviceLimits,
) -> Result<RadixSortPlan, RadixSortError> {
    if rows > u32::MAX as usize {
        return Err(RadixSortError::InvalidInput(format!(
            "radix argsort rows {rows} exceed u32 ID capacity {}",
            u32::MAX
        )));
    }
    if !(1..=RADIX_MAX_DIGIT_BITS).contains(&config.digit_bits) {
        return Err(RadixSortError::InvalidInput(format!(
            "radix digit bits {} outside 1..={RADIX_MAX_DIGIT_BITS}",
            config.digit_bits
        )));
    }
    if config.max_cubes == 0 {
        return Err(RadixSortError::InvalidInput(
            "radix max_cubes must be positive".into(),
        ));
    }
    if limits.alignment == 0 {
        return Err(capability("device reports zero allocation alignment"));
    }
    if limits.max_cube_count_x == 0 {
        return Err(capability("device reports zero cube count capacity"));
    }
    let digit_bits = config.digit_bits;
    let radix = 1usize << digit_bits;
    let passes = 32u32.div_ceil(digit_bits);
    let max_cubes = config.max_cubes.min(limits.max_cube_count_x) as usize;
    let rows_or_one = rows.max(1);

    let (cube_dim, planes, granule, scan_block, shared_bytes) = match limits.path {
        RadixPath::SerialCpu => {
            if config.cube_dim != 1 {
                return Err(RadixSortError::InvalidInput(
                    "serial radix path requires cube_dim 1".into(),
                ));
            }
            if config.cpu_min_chunk_rows == 0 {
                return Err(RadixSortError::InvalidInput(
                    "radix cpu_min_chunk_rows must be positive".into(),
                ));
            }
            (
                1u32,
                1u32,
                config.cpu_min_chunk_rows as usize,
                RADIX_CPU_SCAN_BLOCK,
                0usize,
            )
        }
        RadixPath::CooperativePlane => {
            if !limits.plane_ops {
                return Err(capability("cooperative radix argsort requires Plane::Ops"));
            }
            let plane = limits.plane_size_min;
            if plane != limits.plane_size_max || !plane.is_power_of_two() || plane > 128 {
                return Err(capability(format!(
                    "cooperative radix argsort requires one power-of-two plane width <=128; \
                     device reports {}..={}",
                    limits.plane_size_min, limits.plane_size_max
                )));
            }
            let usable = config
                .cube_dim
                .min(limits.max_units_per_cube)
                .min(limits.max_cube_dim_x);
            let dim = usable / plane * plane;
            if dim == 0 {
                return Err(capability(format!(
                    "radix cube_dim {} cannot hold one plane of {plane} units",
                    config.cube_dim
                )));
            }
            let planes = dim / plane;
            let rank_words = (planes as usize)
                .checked_mul(radix)
                .and_then(|v| v.checked_add(radix))
                .ok_or_else(|| overflow("shared ranking size"))?;
            let rank_bytes = rank_words
                .checked_mul(4)
                .ok_or_else(|| overflow("shared ranking bytes"))?;
            let scan_bytes = (dim as usize) * 2 * 4;
            let shared = rank_bytes.max(scan_bytes);
            if shared > limits.max_shared_memory_bytes {
                return Err(capability(format!(
                    "radix ranking needs {shared} shared bytes; device allows {}",
                    limits.max_shared_memory_bytes
                )));
            }
            (dim, planes, dim as usize, dim as usize, shared)
        }
    };

    let geometry = radix_geometry(rows, granule, max_cubes, radix, scan_block)?;
    let cube_capacity = rows_or_one.div_ceil(granule).min(max_cubes).max(1);
    let table_capacity = radix
        .checked_mul(cube_capacity)
        .ok_or_else(|| overflow("histogram table capacity"))?;
    let (capacity_levels, capacity_count) = scan_levels_fixed(table_capacity, scan_block)?;
    let scan_capacity = capacity_levels[..capacity_count].to_vec();
    for &len in std::iter::once(&table_capacity).chain(scan_capacity.iter()) {
        let blocks = len.div_ceil(scan_block);
        if blocks > limits.max_cube_count_x as usize {
            return Err(capability(format!(
                "radix scan needs {blocks} cubes; device allows {}",
                limits.max_cube_count_x
            )));
        }
    }
    let reduce_cubes = radix.div_ceil(cube_dim as usize);
    if reduce_cubes > limits.max_cube_count_x as usize {
        return Err(capability(format!(
            "radix histogram reduction needs {reduce_cubes} cubes; device allows {}",
            limits.max_cube_count_x
        )));
    }
    let status_words = (passes as usize)
        .checked_mul(radix)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| overflow("status length"))?;
    let host_status_bytes = (status_words as u64)
        .checked_mul(4)
        .ok_or_else(|| overflow("status bytes"))?;
    let stride = (cube_capacity as u64) * cube_dim as u64;
    if stride > u32::MAX as u64 {
        return Err(overflow("launch stride"));
    }

    let alignment = limits.alignment;
    let mut buffers: Vec<(&'static str, usize)> = vec![
        ("keys_a", rows_or_one),
        ("keys_b", rows_or_one),
        ("ids_a", rows_or_one),
        ("ids_b", rows_or_one),
        ("sorted_values", rows_or_one),
        ("histogram_table", table_capacity),
    ];
    const SCAN_NAMES: [&str; 8] = [
        "scan_sums_0",
        "scan_sums_1",
        "scan_sums_2",
        "scan_sums_3",
        "scan_sums_4",
        "scan_sums_5",
        "scan_sums_6",
        "scan_sums_7",
    ];
    for (level, &len) in scan_capacity.iter().enumerate() {
        buffers.push((SCAN_NAMES[level], len));
    }
    buffers.push(("status", status_words));

    let mut allocations = Vec::with_capacity(buffers.len());
    let mut total: u64 = 0;
    let mut largest: u64 = 0;
    for (buffer, elements) in buffers {
        let bytes = (elements as u64)
            .checked_mul(4)
            .ok_or_else(|| overflow("allocation bytes"))?;
        let aligned = bytes
            .checked_add(alignment - 1)
            .map(|v| v / alignment * alignment)
            .ok_or_else(|| overflow("aligned allocation bytes"))?;
        if aligned > limits.max_allocation_bytes {
            return Err(RadixSortError::Allocation {
                buffer,
                required_bytes: aligned,
                limit_bytes: limits.max_allocation_bytes,
            });
        }
        usize::try_from(aligned).map_err(|_| overflow("allocation size as usize"))?;
        total = total
            .checked_add(aligned)
            .ok_or_else(|| overflow("total scratch bytes"))?;
        largest = largest.max(aligned);
        allocations.push(RadixAllocation {
            buffer,
            elements,
            bytes,
            aligned_bytes: aligned,
        });
    }

    Ok(RadixSortPlan {
        path: limits.path,
        rows,
        digit_bits,
        radix,
        passes,
        cube_dim,
        planes_per_cube: planes,
        cubes: geometry.cubes,
        cube_capacity: cube_capacity as u32,
        chunk_len: geometry.chunk_len,
        table_len: geometry.table_len,
        table_capacity,
        scan_block,
        scan_levels: geometry.levels().to_vec(),
        scan_capacity,
        shared_bytes,
        granule,
        max_cubes: max_cubes as u32,
        status_words,
        host_status_bytes,
        allocations,
        total_aligned_bytes: total,
        max_aligned_allocation_bytes: largest,
    })
}

/// Plan a sort for this client's limits.
pub fn radix_sort_plan(
    client: &Client,
    rows: usize,
    config: RadixSortConfig,
) -> Result<RadixSortPlan, RadixSortError> {
    radix_sort_plan_for_limits(rows, config, RadixDeviceLimits::from_client(client))
}

/// Per-call execution facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RadixSortStats {
    pub rows: usize,
    pub passes_executed: u32,
    pub passes_skipped: u32,
    /// Ping-pong slot (0 = `a`, 1 = `b`) holding the result; parity of executed passes.
    pub result_slot: usize,
}

/// Handles into workspace storage; valid until the next `sort` on the same workspace.
#[derive(Clone)]
pub struct RadixSortOutput {
    /// Original row IDs in stable ascending order; first `rows` u32 elements.
    pub ids: Handle,
    /// `values[ids[j]]` with original bits (signed zeros preserved); first `rows` f32.
    pub sorted_values: Handle,
    pub stats: RadixSortStats,
}

impl RadixSortPlan {
    /// Aggregate feasibility of all workspace buffers against a resolved budget.
    ///
    /// `live_bytes` are bytes already owned by the caller that the budget may
    /// reclaim (see [`MemoryBudget::usable_bytes`]).
    pub fn check_budget(
        &self,
        budget: &MemoryBudget,
        live_bytes: u64,
    ) -> Result<(), RadixSortError> {
        let usable = budget.usable_bytes(live_bytes);
        if self.total_aligned_bytes > usable {
            return Err(RadixSortError::Allocation {
                buffer: "radix_workspace_total",
                required_bytes: self.total_aligned_bytes,
                limit_bytes: usable,
            });
        }
        Ok(())
    }
}

/// Reusable owned device scratch for repeated sorts of up to `capacity` rows.
///
/// After construction, `sort` performs no host heap allocation apart from the
/// runtime's status readback buffer and error messages.
pub struct RadixSortWorkspace {
    client: Client,
    plan: RadixSortPlan,
    keys: [Handle; 2],
    ids: [Handle; 2],
    sorted_values: Handle,
    table: Handle,
    scan_sums: Vec<Handle>,
    status: Handle,
}

impl RadixSortWorkspace {
    /// Plan, check aggregate feasibility against the device's reported total
    /// memory, then allocate every buffer before any launch.
    ///
    /// Runtime integration should prefer [`Self::with_budget`], which checks the
    /// aggregate against resolved free memory.
    pub fn new(
        client: &Client,
        capacity: usize,
        config: RadixSortConfig,
    ) -> Result<Self, RadixSortError> {
        let plan = radix_sort_plan(client, capacity, config)?;
        if let Some(total) = client.properties().memory.max_memory()
            && plan.total_aligned_bytes > total
        {
            return Err(RadixSortError::Allocation {
                buffer: "radix_workspace_total",
                required_bytes: plan.total_aligned_bytes,
                limit_bytes: total,
            });
        }
        Ok(Self::allocate(client, plan))
    }

    /// Like [`Self::new`], but rejects plans exceeding `budget.usable_bytes(live_bytes)`.
    pub fn with_budget(
        client: &Client,
        capacity: usize,
        config: RadixSortConfig,
        budget: &MemoryBudget,
        live_bytes: u64,
    ) -> Result<Self, RadixSortError> {
        let plan = radix_sort_plan(client, capacity, config)?;
        plan.check_budget(budget, live_bytes)?;
        Ok(Self::allocate(client, plan))
    }

    fn allocate(client: &Client, plan: RadixSortPlan) -> Self {
        let mut handles = plan
            .allocations
            .iter()
            .map(|allocation| client.empty(allocation.bytes as usize))
            .collect::<Vec<_>>()
            .into_iter();
        let mut next = || handles.next().expect("plan lists every workspace buffer");
        let keys = [next(), next()];
        let ids = [next(), next()];
        let sorted_values = next();
        let table = next();
        let scan_sums = (0..plan.scan_capacity.len()).map(|_| next()).collect();
        let status = next();
        Self {
            client: client.clone(),
            plan,
            keys,
            ids,
            sorted_values,
            table,
            scan_sums,
            status,
        }
    }

    /// Plan used for allocation (capacity rows).
    pub fn plan(&self) -> &RadixSortPlan {
        &self.plan
    }

    pub fn capacity(&self) -> usize {
        self.plan.rows
    }

    /// Stable argsort of the first `rows` f32 elements of `values`.
    ///
    /// Encodes keys, builds the global per-pass digit histograms from them and
    /// synchronizes once to read status before any radix pass. Returns
    /// `NonFinite` with the lowest offending row. A pass is skipped exactly when
    /// its global histogram places all rows in one bucket.
    pub fn sort(
        &mut self,
        values: &Handle,
        rows: usize,
    ) -> Result<RadixSortOutput, RadixSortError> {
        if rows > self.plan.rows {
            return Err(RadixSortError::Workspace(format!(
                "radix workspace capacity {} is smaller than {rows} rows",
                self.plan.rows
            )));
        }
        let value_bytes = rows.checked_mul(4).ok_or_else(|| overflow("input bytes"))?;
        if (values.size() as u128) < value_bytes as u128 {
            return Err(RadixSortError::InvalidInput(format!(
                "radix input handle holds {} bytes; {rows} f32 rows need {value_bytes}",
                values.size()
            )));
        }
        let plan = &self.plan;
        let mut stats = RadixSortStats {
            rows,
            passes_executed: 0,
            passes_skipped: 0,
            result_slot: 0,
        };
        if rows == 0 {
            stats.passes_skipped = plan.passes;
            return Ok(self.output(stats));
        }
        let geometry = radix_geometry(
            rows,
            plan.granule,
            plan.max_cubes as usize,
            plan.radix,
            plan.scan_block,
        )?;
        if geometry.table_len > plan.table_capacity
            || geometry.scan_level_count > plan.scan_capacity.len()
            || geometry
                .levels()
                .iter()
                .zip(&plan.scan_capacity)
                .any(|(need, have)| need > have)
        {
            return Err(RadixSortError::Workspace(
                "radix histogram/scan scratch exceeds workspace capacity".into(),
            ));
        }
        let client = &self.client;
        let rows_u32 = rows as u32;
        let dim = CubeDim::new_1d(plan.cube_dim);
        let count = CubeCount::Static(geometry.cubes, 1, 1);
        let stride = geometry.cubes * plan.cube_dim;
        let n = rows;
        let status_words = plan.status_words;
        let arg =
            |handle: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(handle.clone(), len) };

        radix_reset_status::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            arg(&self.status, status_words),
        );
        radix_encode::launch(
            client,
            count.clone(),
            dim,
            arg(values, n),
            arg(&self.keys[0], n),
            arg(&self.ids[0], n),
            arg(&self.status, status_words),
            rows_u32,
            stride,
        );
        // Global digit histograms: per-cube counts of the encoded keys, reduced
        // over cubes into the status buffer. Counts are order invariant.
        for pass in 0..plan.passes {
            let (shift, mask) = digit(plan, pass);
            self.histogram(&geometry, 0, rows_u32, shift, mask);
            radix_reduce_histogram::launch(
                client,
                CubeCount::Static(plan.radix.div_ceil(plan.cube_dim as usize) as u32, 1, 1),
                dim,
                arg(&self.table, geometry.table_len),
                arg(&self.status, status_words),
                geometry.cubes,
                1 + pass * plan.radix as u32,
                plan.radix as u32,
            );
        }
        let status = client
            .read_one(self.status.clone())
            .map_err(|error| RadixSortError::Device(format!("radix status readback: {error:?}")))?;
        let status = u32::from_bytes(&status);
        if status.len() < status_words {
            return Err(RadixSortError::Device(format!(
                "radix status readback returned {} words; expected {status_words}",
                status.len()
            )));
        }
        if status[0] != NO_INVALID_INDEX {
            return Err(RadixSortError::NonFinite {
                index: status[0] as usize,
            });
        }

        let mut slot = 0usize;
        for pass in 0..plan.passes {
            let (shift, mask) = digit(plan, pass);
            let base = 1 + pass as usize * plan.radix;
            if status[base..base + plan.radix]
                .iter()
                .any(|&bucket| bucket as usize == rows)
            {
                stats.passes_skipped += 1;
                continue;
            }
            self.histogram(&geometry, slot, rows_u32, shift, mask);
            self.scan(&geometry);
            self.scatter(&geometry, slot, rows_u32, shift, mask);
            slot ^= 1;
            stats.passes_executed += 1;
        }
        stats.result_slot = slot;

        radix_gather::launch(
            client,
            count,
            dim,
            arg(values, n),
            arg(&self.ids[slot], n),
            arg(&self.sorted_values, n),
            rows_u32,
            stride,
        );
        Ok(self.output(stats))
    }

    fn output(&self, stats: RadixSortStats) -> RadixSortOutput {
        RadixSortOutput {
            ids: self.ids[stats.result_slot].clone(),
            sorted_values: self.sorted_values.clone(),
            stats,
        }
    }

    fn histogram(&self, geometry: &RadixGeometry, slot: usize, rows: u32, shift: u32, mask: u32) {
        self.pass_kernel(geometry, slot, rows, shift, mask, false);
    }

    fn scatter(&self, geometry: &RadixGeometry, slot: usize, rows: u32, shift: u32, mask: u32) {
        self.pass_kernel(geometry, slot, rows, shift, mask, true);
    }

    fn pass_kernel(
        &self,
        geometry: &RadixGeometry,
        slot: usize,
        rows: u32,
        shift: u32,
        mask: u32,
        scatter: bool,
    ) {
        let plan = &self.plan;
        let n = rows as usize;
        let arg =
            |handle: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(handle.clone(), len) };
        let count = CubeCount::Static(geometry.cubes, 1, 1);
        let chunk = geometry.chunk_len as u32;
        match (plan.path, scatter) {
            (RadixPath::SerialCpu, false) => {
                radix_cpu_histogram::launch(
                    &self.client,
                    count,
                    CubeDim::new_1d(1),
                    arg(&self.keys[slot], n),
                    arg(&self.table, geometry.table_len),
                    rows,
                    chunk,
                    geometry.cubes,
                    shift,
                    mask,
                    plan.radix,
                );
            }
            (RadixPath::SerialCpu, true) => {
                radix_cpu_scatter::launch(
                    &self.client,
                    count,
                    CubeDim::new_1d(1),
                    arg(&self.keys[slot], n),
                    arg(&self.ids[slot], n),
                    arg(&self.table, geometry.table_len),
                    arg(&self.keys[slot ^ 1], n),
                    arg(&self.ids[slot ^ 1], n),
                    rows,
                    chunk,
                    geometry.cubes,
                    shift,
                    mask,
                );
            }
            (RadixPath::CooperativePlane, scatter) => {
                radix_plane_pass::launch(
                    &self.client,
                    count,
                    CubeDim::new_1d(plan.cube_dim),
                    arg(&self.keys[slot], n),
                    arg(&self.ids[slot], n),
                    arg(&self.table, geometry.table_len),
                    arg(&self.keys[slot ^ 1], n),
                    arg(&self.ids[slot ^ 1], n),
                    rows,
                    chunk,
                    geometry.cubes,
                    shift,
                    mask,
                    plan.digit_bits,
                    plan.radix,
                    plan.planes_per_cube as usize,
                    plan.planes_per_cube as usize * plan.radix,
                    scatter,
                );
            }
        }
    }

    /// Exclusive scan of `table[..table_len]`: block scans upward, then uniform
    /// adds of scanned block sums downward.
    fn scan(&self, geometry: &RadixGeometry) {
        let plan = &self.plan;
        let arg =
            |handle: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(handle.clone(), len) };
        let block = plan.scan_block;
        let levels = geometry.levels();
        let data_of = |level: usize| -> (&Handle, usize) {
            if level == 0 {
                (&self.table, geometry.table_len)
            } else {
                (&self.scan_sums[level - 1], levels[level - 1])
            }
        };
        for (level, &blocks) in levels.iter().enumerate() {
            let (data, len) = data_of(level);
            let count = CubeCount::Static(blocks as u32, 1, 1);
            match plan.path {
                RadixPath::SerialCpu => {
                    radix_scan_block_serial::launch(
                        &self.client,
                        count,
                        CubeDim::new_1d(1),
                        arg(data, len),
                        arg(&self.scan_sums[level], blocks),
                        len as u32,
                        block as u32,
                    );
                }
                RadixPath::CooperativePlane => {
                    radix_scan_block_shared::launch(
                        &self.client,
                        count,
                        CubeDim::new_1d(block as u32),
                        arg(data, len),
                        arg(&self.scan_sums[level], blocks),
                        len as u32,
                        block,
                        block * 2,
                    );
                }
            }
        }
        for level in (0..levels.len() - 1).rev() {
            let (data, len) = data_of(level);
            let blocks = levels[level];
            radix_uniform_add::launch(
                &self.client,
                CubeCount::Static(blocks as u32, 1, 1),
                CubeDim::new_1d(plan.cube_dim),
                arg(data, len),
                arg(&self.scan_sums[level], blocks),
                len as u32,
                block as u32,
            );
        }
    }
}

/// Shift and mask of LSD digit `pass`.
fn digit(plan: &RadixSortPlan, pass: u32) -> (u32, u32) {
    let shift = pass * plan.digit_bits;
    let width = plan.digit_bits.min(32 - shift);
    (shift, (1u32 << width) - 1)
}

/// Owned frontend: stable argsort of a contiguous rank-1 f32 `CubeTensor`.
///
/// Returns `(ids: U32, sorted_values: F32)` tensors of the input length.
pub fn stable_argsort_f32(
    values: &CubeTensor,
    config: RadixSortConfig,
) -> Result<(CubeTensor, CubeTensor, RadixSortStats), RadixSortError> {
    if values.dtype != DType::F32 {
        return Err(RadixSortError::InvalidInput(format!(
            "radix argsort requires f32 values, got {:?}",
            values.dtype
        )));
    }
    let shape = values.meta.shape().clone();
    if shape.num_dims() != 1 || !values.is_contiguous() {
        return Err(RadixSortError::InvalidInput(
            "radix argsort requires a contiguous rank-1 tensor".into(),
        ));
    }
    let rows = shape.num_elements();
    let mut workspace = RadixSortWorkspace::new(&values.client, rows, config)?;
    let output = workspace.sort(&values.handle, rows)?;
    let tensor = |handle: Handle, dtype| {
        CubeTensor::new_contiguous(
            values.client.clone(),
            values.device.clone(),
            Shape::new([rows]),
            handle,
            dtype,
        )
    };
    Ok((
        tensor(output.ids, DType::U32),
        tensor(output.sorted_values, DType::F32),
        output.stats,
    ))
}

/// Host reference key mapping; identical to the device encoding.
pub fn radix_key_f32(value: f32) -> u32 {
    let mut bits = value.to_bits();
    if bits == 0x8000_0000 {
        bits = 0;
    }
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

#[cube]
fn finite_key(raw: u32) -> u32 {
    let mut bits = raw;
    if bits == 0x8000_0000u32 {
        bits = 0u32;
    }
    let mut key = bits ^ 0x8000_0000u32;
    if (bits & 0x8000_0000u32) != 0u32 {
        key = bits ^ 0xFFFF_FFFFu32;
    }
    key
}

#[cube(launch)]
fn radix_reset_status(status: &[Atomic<u32>]) {
    if ABSOLUTE_POS == 0 {
        status[0].store(0xFFFF_FFFFu32);
    }
}

/// Encode keys, seed IDs and record the lowest non-finite row.
#[cube(launch)]
fn radix_encode(
    values: &[f32],
    keys: &mut [u32],
    ids: &mut [u32],
    status: &[Atomic<u32>],
    rows: u32,
    stride: u32,
) {
    let n = rows as usize;
    let step = stride as usize;
    let mut i = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    while i < n {
        let bits = u32::reinterpret(values[i]);
        let index = u32::cast_from(i);
        if (bits & 0x7F80_0000u32) == 0x7F80_0000u32 {
            status[0].fetch_min(index);
        }
        keys[i] = finite_key(bits);
        ids[i] = index;
        i += step;
    }
}

/// Sum the digit-major per-cube table into `status[offset + digit]`.
#[cube(launch)]
fn radix_reduce_histogram(table: &[u32], status: &mut [u32], cubes: u32, offset: u32, radix: u32) {
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if d < radix as usize {
        let columns = cubes as usize;
        let mut sum = 0u32;
        let mut c = 0usize;
        while c < columns {
            sum += table[d * columns + c];
            c += 1;
        }
        status[offset as usize + d] = sum;
    }
}

#[cube(launch)]
fn radix_gather(values: &[f32], ids: &[u32], out: &mut [f32], rows: u32, stride: u32) {
    let n = rows as usize;
    let step = stride as usize;
    let mut i = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    while i < n {
        out[i] = values[ids[i] as usize];
        i += step;
    }
}

/// Serial path: one unit counts its cube's chunk into its digit-major column.
#[cube(launch)]
fn radix_cpu_histogram(
    keys: &[u32],
    table: &mut [u32],
    rows: u32,
    chunk_len: u32,
    cubes: u32,
    shift: u32,
    mask: u32,
    #[comptime] radix: usize,
) {
    if UNIT_POS_X == 0 {
        let cube = CUBE_POS_X as usize;
        let columns = cubes as usize;
        let start = cube * chunk_len as usize;
        let mut end = start + chunk_len as usize;
        if end > rows as usize {
            end = rows as usize;
        }
        let mut digit = 0usize;
        while digit < radix {
            table[digit * columns + cube] = 0u32;
            digit += 1;
        }
        let mut i = start;
        while i < end {
            let slot = ((keys[i] >> shift) & mask) as usize * columns + cube;
            table[slot] += 1u32;
            i += 1;
        }
    }
}

/// Serial path: stable scatter using the scanned column as a running cursor.
#[cube(launch)]
fn radix_cpu_scatter(
    keys: &[u32],
    ids: &[u32],
    table: &mut [u32],
    out_keys: &mut [u32],
    out_ids: &mut [u32],
    rows: u32,
    chunk_len: u32,
    cubes: u32,
    shift: u32,
    mask: u32,
) {
    if UNIT_POS_X == 0 {
        let cube = CUBE_POS_X as usize;
        let columns = cubes as usize;
        let start = cube * chunk_len as usize;
        let mut end = start + chunk_len as usize;
        if end > rows as usize {
            end = rows as usize;
        }
        let mut i = start;
        while i < end {
            let key = keys[i];
            let slot = ((key >> shift) & mask) as usize * columns + cube;
            let position = table[slot];
            table[slot] = position + 1u32;
            out_keys[position as usize] = key;
            out_ids[position as usize] = ids[i];
            i += 1;
        }
    }
}

/// Lanes strictly below `lane` within ballot word starting at `base`.
#[cube]
fn lower_lane_mask(lane: u32, base: u32) -> u32 {
    let mut mask = 0u32;
    if lane >= base + 32u32 {
        mask = 0xFFFF_FFFFu32;
    } else if lane > base {
        mask = (1u32 << (lane - base)) - 1u32;
    }
    mask
}

/// Cooperative path: rank each tile of `CUBE_DIM_X` keys with plane ballots.
///
/// Histogram mode (`scatter == false`) writes the cube's per-digit totals to
/// `table[digit * cubes + cube]`. Scatter mode starts each digit at the scanned
/// table offset and writes keys/IDs to their stable destinations. Every unit runs
/// the same tile loop, so ballots and barriers are cube-uniform; lanes past the
/// chunk end are inactive and masked out of every peer set.
#[cube(launch)]
fn radix_plane_pass(
    keys: &[u32],
    ids: &[u32],
    table: &mut [u32],
    out_keys: &mut [u32],
    out_ids: &mut [u32],
    rows: u32,
    chunk_len: u32,
    cubes: u32,
    shift: u32,
    mask: u32,
    #[comptime] bits: u32,
    #[comptime] radix: usize,
    #[comptime] planes: usize,
    #[comptime] plane_slots: usize,
    #[comptime] scatter: bool,
) {
    let cube = CUBE_POS_X as usize;
    let columns = cubes as usize;
    let unit = UNIT_POS_X as usize;
    let dim = CUBE_DIM_X as usize;
    let lane = UNIT_POS_PLANE;
    let plane = (UNIT_POS_X / PLANE_DIM) as usize;
    let mut plane_counts = Shared::<[u32]>::new_slice(plane_slots);
    let mut running = Shared::<[u32]>::new_slice(radix);

    let mut d = unit;
    while d < radix {
        if scatter {
            running[d] = table[d * columns + cube];
        } else {
            running[d] = 0u32;
        }
        d += dim;
    }
    let start = cube * chunk_len as usize;
    let mut end = start + chunk_len as usize;
    if end > rows as usize {
        end = rows as usize;
    }
    let mut tile = start;
    while tile < end {
        let mut z = unit;
        while z < plane_slots {
            plane_counts[z] = 0u32;
            z += dim;
        }
        sync_cube();

        let index = tile + unit;
        let active = index < end;
        let mut key = 0u32;
        if active {
            key = keys[index];
        }
        let digit = (key >> shift) & mask;
        let act = plane_ballot(active);
        let mut p0 = act.extract(0usize);
        let mut p1 = act.extract(1usize);
        let mut p2 = act.extract(2usize);
        let mut p3 = act.extract(3usize);
        for b in 0..bits {
            let set = ((digit >> b) & 1u32) == 1u32;
            let vote = plane_ballot(set);
            let mut flip = 0xFFFF_FFFFu32;
            if set {
                flip = 0u32;
            }
            p0 &= vote.extract(0usize) ^ flip;
            p1 &= vote.extract(1usize) ^ flip;
            p2 &= vote.extract(2usize) ^ flip;
            p3 &= vote.extract(3usize) ^ flip;
        }
        let lower = (p0 & lower_lane_mask(lane, 0u32)).count_ones()
            + (p1 & lower_lane_mask(lane, 32u32)).count_ones()
            + (p2 & lower_lane_mask(lane, 64u32)).count_ones()
            + (p3 & lower_lane_mask(lane, 96u32)).count_ones();
        let peers = p0.count_ones() + p1.count_ones() + p2.count_ones() + p3.count_ones();
        let digit_slot = digit as usize;
        if active && lower == 0u32 {
            plane_counts[plane * radix + digit_slot] = peers;
        }
        sync_cube();

        if scatter && active {
            let mut rank = running[digit_slot] + lower;
            let mut p = 0usize;
            while p < plane {
                rank += plane_counts[p * radix + digit_slot];
                p += 1;
            }
            out_keys[rank as usize] = key;
            out_ids[rank as usize] = ids[index];
        }
        sync_cube();

        let mut d = unit;
        while d < radix {
            let mut sum = 0u32;
            let mut p = 0usize;
            while p < planes {
                sum += plane_counts[p * radix + d];
                p += 1;
            }
            running[d] += sum;
            d += dim;
        }
        sync_cube();
        tile += dim;
    }

    if !scatter {
        let mut d = unit;
        while d < radix {
            table[d * columns + cube] = running[d];
            d += dim;
        }
    }
}

/// Serial exclusive scan of one block; writes the block total to `sums`.
#[cube(launch)]
fn radix_scan_block_serial(data: &mut [u32], sums: &mut [u32], len: u32, block: u32) {
    if UNIT_POS_X == 0 {
        let cube = CUBE_POS_X as usize;
        let start = cube * block as usize;
        let mut end = start + block as usize;
        if end > len as usize {
            end = len as usize;
        }
        let mut total = 0u32;
        let mut i = start;
        while i < end {
            let value = data[i];
            data[i] = total;
            total += value;
            i += 1;
        }
        sums[cube] = total;
    }
}

/// Shared-memory Hillis-Steele exclusive scan of one `block`-element block.
#[cube(launch)]
fn radix_scan_block_shared(
    data: &mut [u32],
    sums: &mut [u32],
    len: u32,
    #[comptime] block: usize,
    #[comptime] buffer_len: usize,
) {
    let cube = CUBE_POS_X as usize;
    let unit = UNIT_POS_X as usize;
    let mut buffer = Shared::<[u32]>::new_slice(buffer_len);
    let index = cube * block + unit;
    let active = index < len as usize;
    let mut value = 0u32;
    if active {
        value = data[index];
    }
    buffer[unit] = value;
    sync_cube();
    let mut source = 0usize;
    let mut offset = 1usize;
    while offset < block {
        let mut sum = buffer[source * block + unit];
        if unit >= offset {
            sum += buffer[source * block + unit - offset];
        }
        buffer[(1 - source) * block + unit] = sum;
        source = 1 - source;
        sync_cube();
        offset *= 2;
    }
    let inclusive = buffer[source * block + unit];
    if active {
        data[index] = inclusive - value;
    }
    if unit == block - 1 {
        sums[cube] = inclusive;
    }
}

/// Add each block's scanned base to its elements.
#[cube(launch)]
fn radix_uniform_add(data: &mut [u32], bases: &[u32], len: u32, block: u32) {
    let cube = CUBE_POS_X as usize;
    let start = cube * block as usize;
    let mut end = start + block as usize;
    if end > len as usize {
        end = len as usize;
    }
    let base = bases[cube];
    let mut i = start + UNIT_POS_X as usize;
    while i < end {
        data[i] += base;
        i += CUBE_DIM_X as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_limits() -> RadixDeviceLimits {
        RadixDeviceLimits {
            path: RadixPath::CooperativePlane,
            alignment: 256,
            max_allocation_bytes: u64::MAX,
            max_cube_count_x: 65_535,
            max_units_per_cube: 1_024,
            max_cube_dim_x: 1_024,
            plane_ops: true,
            plane_size_min: 32,
            plane_size_max: 32,
            max_shared_memory_bytes: 48 * 1024,
        }
    }

    #[test]
    fn key_mapping_is_monotone_and_merges_signed_zero() {
        let ordered = [
            f32::MIN,
            -1.0,
            -f32::MIN_POSITIVE,
            -f32::from_bits(1),
            0.0,
            f32::from_bits(1),
            f32::MIN_POSITIVE,
            1.0,
            f32::MAX,
        ];
        for pair in ordered.windows(2) {
            assert!(radix_key_f32(pair[0]) < radix_key_f32(pair[1]));
        }
        assert_eq!(radix_key_f32(-0.0), radix_key_f32(0.0));
    }

    #[test]
    fn scan_hierarchy_ends_at_one_block() {
        assert_eq!(scan_lengths(1, 256), vec![1]);
        assert_eq!(scan_lengths(256, 256), vec![1]);
        assert_eq!(scan_lengths(257, 256), vec![2, 1]);
        assert_eq!(scan_lengths(256 * 1024, 256), vec![1024, 4, 1]);
    }

    #[test]
    fn plan_covers_rows_and_tiles() {
        let plan = radix_sort_plan_for_limits(
            1_000_003,
            RadixSortConfig::conservative(RadixPath::CooperativePlane),
            plane_limits(),
        )
        .unwrap();
        assert_eq!(plan.chunk_len % plan.cube_dim as usize, 0);
        assert!(plan.cubes as usize * plan.chunk_len >= 1_000_003);
        assert!((plan.cubes as usize - 1) * plan.chunk_len < 1_000_003);
        assert!(plan.cubes <= plan.cube_capacity);
        assert_eq!(plan.table_len, plan.radix * plan.cubes as usize);
    }
}
