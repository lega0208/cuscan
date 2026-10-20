//! Shared launch geometry, reverse CSR and output conversion for sampled NN-descent.

use burn::{cubecl::prelude::*, tensor::DType};
use burn_cubecl::tensor::CubeTensor;

#[path = "ann_forest.rs"]
mod ann_forest;
#[path = "ann_sampled.rs"]
mod ann_sampled;

const ANN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

fn ann_launch_geometry(
    client: &cubecl::client::Client,
    feature_size: usize,
    n_rows: u32,
) -> (bool, CubeCount, CubeDim) {
    let cpu_merge = client.name() == "cpu";
    let hardware = &client.properties().hardware;
    if !cpu_merge {
        validate_cooperative_ann_launch(hardware, client.features(), feature_size, n_rows);
    }
    ann_launch_geometry_with_limits(
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

fn ann_launch_geometry_with_limits(
    cpu_merge: bool,
    max_cube_dim: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cubes: u32,
    cores: u32,
    n_rows: u32,
) -> (bool, CubeCount, CubeDim) {
    assert!(n_rows > 0, "ANN row count must be nonzero");
    if cpu_merge {
        let dim_x = n_rows
            .min(cores)
            .min(max_cube_dim.0)
            .min(max_units_per_cube)
            .max(1);
        let dim = CubeDim::new_1d(dim_x);
        let count = ann_independent_cube_count(n_rows, dim.x, max_cubes);
        (true, CubeCount::Static(count, 1, 1), dim)
    } else {
        assert!(
            n_rows <= max_cubes,
            "ANN cooperative cube count exceeds hardware limit"
        );
        (false, CubeCount::Static(n_rows, 1, 1), ANN_CUBE_DIM)
    }
}

fn validate_cooperative_ann_launch(
    hardware: &cubecl::ir::HardwareProperties,
    features: &cubecl::features::Features,
    feature_size: usize,
    n_rows: u32,
) {
    use cubecl::features::Plane;
    assert!(
        features.plane.contains(Plane::Ops),
        "cooperative ANN requires Plane::Ops"
    );
    assert!(
        hardware.plane_size_min >= ANN_CUBE_DIM.x,
        "cooperative ANN requires guaranteed plane width >=32"
    );
    assert!(
        hardware.max_units_per_cube >= ANN_CUBE_DIM.x && hardware.max_cube_dim.0 >= ANN_CUBE_DIM.x,
        "cooperative ANN requires 32 units and x dimension"
    );
    let shared_bytes = 32 * (2 * DType::I32.size() + feature_size) + DType::I32.size();
    assert!(
        shared_bytes <= hardware.max_shared_memory_size,
        "cooperative ANN shared-memory capacity exceeded"
    );
    assert!(
        hardware.max_cube_count.0 >= n_rows,
        "cooperative ANN x cube capacity exceeded"
    );
}

fn ann_independent_launch(client: &cubecl::client::Client, units: u32) -> (CubeCount, CubeDim) {
    let dim = crate::kernels::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x));
    let count =
        ann_independent_cube_count(units, dim.x, client.properties().hardware.max_cube_count.0);
    (CubeCount::Static(count, 1, 1), dim)
}

fn ann_independent_cube_count(units: u32, units_per_cube: u32, max_cubes: u32) -> u32 {
    assert!(
        units_per_cube > 0,
        "ANN independent launch dimension must be nonzero"
    );
    let count = units.div_ceil(units_per_cube);
    assert!(
        count <= max_cubes,
        "ANN independent cube count exceeds hardware limit"
    );
    count
}

fn ann_should_stop(new_count: i32, n: usize, width: usize) -> bool {
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
fn reverse_zero<I: Int>(counts: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    let target = ABSOLUTE_POS_X as usize;
    if target < counts.shape(0) {
        counts[target] = I::cast_from(0i32);
    }
}

#[cube(launch)]
fn reverse_count<I: Int>(
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
fn reverse_scatter<I: Int>(
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
fn ann_output<F: Float, I: Int>(
    indices: &Tensor<I>,
    squared_distances: &Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    width: u32,
    k: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let width = width as usize;
    let k = k as usize;
    if row < indices.shape(0) {
        for slot in 0..k {
            output_indices[row * k + slot] = indices[row * width + slot];
            output_distances[row * k + slot] = F::sqrt(squared_distances[row * width + slot]);
        }
    }
}

pub(crate) fn approximate_neighbors(
    data: CubeTensor,
    k: u32,
    seed: u64,
    max_iterations: u32,
) -> (CubeTensor, CubeTensor) {
    ann_sampled::approximate_neighbors(
        data,
        usize::try_from(k).expect("ANN width exceeds usize"),
        seed,
        max_iterations,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let (cpu, count, dim) = ann_launch_geometry_with_limits(true, (8, 1, 1), 8, 16, 6, 67);
        assert!(cpu);
        assert_eq!(dim.x, 6);
        assert!(matches!(count, CubeCount::Static(12, 1, 1)));
        let (cpu, count, dim) = ann_launch_geometry_with_limits(false, (32, 1, 1), 32, 67, 6, 67);
        assert!(!cpu);
        assert_eq!(dim.x, 32);
        assert!(matches!(count, CubeCount::Static(67, 1, 1)));
        assert!(std::panic::catch_unwind(|| ann_independent_cube_count(67, 32, 2)).is_err());
        assert!(std::panic::catch_unwind(|| ann_independent_cube_count(1, 0, 2)).is_err());
        assert!(
            std::panic::catch_unwind(|| ann_launch_geometry_with_limits(
                false,
                (32, 1, 1),
                32,
                66,
                6,
                67
            ))
            .is_err()
        );
    }
}
