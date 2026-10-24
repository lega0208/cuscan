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

fn workspace_error(stage: &str, error: WorkspaceError) -> AnnError {
    match error {
        WorkspaceError::InvalidDimensions => {
            AnnError::Workspace(format!("ANN {stage} rejected invalid dimensions"))
        }
        WorkspaceError::ArithmeticOverflow => AnnError::Overflow(format!("ANN {stage} overflows")),
    }
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
    let width = width as usize;
    let k = k as usize;
    if row < indices.shape(0) {
        for slot in 0..k {
            output_indices[row * k + slot] = indices[row * width + slot];
            if comptime![squared] {
                output_distances[row * k + slot] = squared_distances[row * width + slot];
            } else {
                output_distances[row * k + slot] = F::sqrt(squared_distances[row * width + slot]);
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
