use burn::{
    backend::{
        TensorMetadata,
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
    },
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use hdbscan_core::error::HdbscanError;

const REFERENCE_TILE: u32 = 256;
const SCRATCH_BUDGET_BYTES: usize = 64 * 1024 * 1024;
const CANDIDATE_CUBE_DIM: CubeDim = CubeDim::new_1d(256);

/// Checked dimensions used by the row-blocked candidate kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoruvkaCandidateSizing {
    pub rows: usize,
    pub tiles: usize,
    pub tile_width: usize,
    pub block_rows: usize,
    pub scratch_bytes: usize,
    pub output_bytes: usize,
}

/// Validate all per-call tensor and scratch sizes before the estimator uploads inputs.
/// The fixed scratch budget bounds each temporary even for very large point sets.
pub fn validate_boruvka_candidate_sizing(
    rows: usize,
) -> Result<BoruvkaCandidateSizing, HdbscanError> {
    if rows == 0 {
        return Err(HdbscanError::InvalidInput(
            "Boruvka candidate input must contain at least one row".into(),
        ));
    }
    i32::try_from(rows).map_err(|_| HdbscanError::Overflow("Boruvka row indices"))?;
    let max_tiles =
        SCRATCH_BUDGET_BYTES / (std::mem::size_of::<f32>() + std::mem::size_of::<i32>());
    let tile_width = REFERENCE_TILE as usize;
    let tile_width = tile_width.max(rows.div_ceil(max_tiles));
    let tiles = rows.div_ceil(tile_width);
    let bytes_per_row = tiles
        .checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<i32>())
        .ok_or(HdbscanError::Overflow("Boruvka candidate scratch row"))?;
    let block_rows = (SCRATCH_BUDGET_BYTES / bytes_per_row).max(1).min(rows);
    let scratch_bytes = block_rows
        .checked_mul(bytes_per_row)
        .ok_or(HdbscanError::Overflow(
            "Boruvka candidate scratch allocation",
        ))?;
    let output_bytes = rows
        .checked_mul(std::mem::size_of::<f32>() + std::mem::size_of::<i32>())
        .ok_or(HdbscanError::Overflow(
            "Boruvka candidate output allocation",
        ))?;
    Ok(BoruvkaCandidateSizing {
        rows,
        tiles,
        tile_width,
        block_rows,
        scratch_bytes,
        output_bytes,
    })
}

#[cube(launch)]
fn tile_candidates<F: Float, I: Int>(
    data: &Tensor<F>,
    core_sq: &Tensor<F>,
    components: &Tensor<I>,
    row_offset: u32,
    work_offset: u64,
    tiles: u32,
    tile_width: u32,
    out_dist: &mut Tensor<F>,
    out_index: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let work = ABSOLUTE_POS_X as usize + work_offset as usize;
    let local_row = work / tiles as usize;
    let tile = work % tiles as usize;
    if local_row < out_dist.shape(0) {
        let n = data.shape(0);
        let d = data.shape(1);
        let row = row_offset as usize + local_row;
        let component = components[row];
        let core = core_sq[row];
        let mut best = F::new(f32::INFINITY);
        let mut best_index = n;
        let start = tile * tile_width as usize;

        for offset in 0..tile_width as usize {
            let candidate = start + offset;
            if candidate < n && components[candidate] != component {
                let mut distance = F::new(0.0f32);
                for feature in 0..d {
                    let delta = data[row * d + feature] - data[candidate * d + feature];
                    distance += delta * delta;
                }
                let weight = F::max(F::max(distance, core), core_sq[candidate]);
                if weight < best || (weight == best && candidate < best_index) {
                    best = weight;
                    best_index = candidate;
                }
            }
        }
        out_dist[local_row * tiles as usize + tile] = best;
        out_index[local_row * tiles as usize + tile] = I::cast_from(best_index);
    }
}

#[cube(launch)]
fn reduce_tiles<F: Float, I: Int>(
    tile_dist: &Tensor<F>,
    tile_index: &Tensor<I>,
    out_dist: &mut Tensor<F>,
    out_index: &mut Tensor<I>,
    row_offset: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    if row < tile_dist.shape(0) {
        let tiles = tile_dist.shape(1);
        let mut best = tile_dist[row * tiles];
        let mut best_index = tile_index[row * tiles];
        for tile in 1..tiles {
            let weight = tile_dist[row * tiles + tile];
            let index = tile_index[row * tiles + tile];
            if weight < best || (weight == best && index < best_index) {
                best = weight;
                best_index = index;
            }
        }
        let destination = row_offset as usize + row;
        out_dist[destination] = best;
        out_index[destination] = best_index;
    }
}

fn checked_launch_count(units: usize, dim_x: u32, max_cubes: u32) -> u32 {
    let cubes = units.div_ceil(dim_x as usize);
    u32::try_from(cubes.min(max_cubes as usize)).expect("bounded CubeCL launch count")
}

pub(super) fn boruvka_candidates(
    data: FloatTensor<CubeBackend>,
    core_sq: FloatTensor<CubeBackend>,
    point_components: IntTensor<CubeBackend>,
) -> (FloatTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let data = into_contiguous(data);
    let core_sq = into_contiguous(core_sq);
    let point_components = into_contiguous(point_components);
    let [rows, features] = data.shape().dims::<2>();
    let [core_rows] = core_sq.shape().dims::<1>();
    let [component_rows] = point_components.shape().dims::<1>();
    assert!(rows > 0 && features > 0, "Boruvka data must be non-empty");
    assert_eq!(core_rows, rows, "Boruvka core count must equal row count");
    assert_eq!(
        component_rows, rows,
        "Boruvka component count must equal row count"
    );
    assert_eq!(data.dtype, DType::F32, "Boruvka data must be f32");
    assert_eq!(
        core_sq.dtype,
        DType::F32,
        "Boruvka core distances must be f32"
    );
    assert_eq!(
        point_components.dtype,
        DType::I32,
        "Boruvka labels must be i32"
    );
    let sizing = validate_boruvka_candidate_sizing(rows)
        .expect("estimator must validate Boruvka sizing before backend dispatch");
    let tiles_u32 = u32::try_from(sizing.tiles).expect("validated tile count fits u32");
    let tile_width_u32 = u32::try_from(sizing.tile_width).expect("validated tile width fits u32");

    let output_float = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows]),
        data.client.empty(
            rows.checked_mul(DType::F32.size())
                .expect("validated output size"),
        ),
        DType::F32,
    );
    let output_index = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([rows]),
        data.client.empty(
            rows.checked_mul(DType::I32.size())
                .expect("validated output size"),
        ),
        DType::I32,
    );
    let client = data.client.clone();
    let hardware = &client.properties().hardware;
    let dim = cuscan_common::legal_cube_dim(&client, CANDIDATE_CUBE_DIM);
    let max_cubes = hardware.max_cube_count.0;
    assert!(max_cubes > 0, "device reports no legal x launch capacity");
    let max_launch_units = (dim.x as usize)
        .checked_mul(max_cubes as usize)
        .expect("CubeCL launch capacity fits usize");
    let launch_block_rows = (max_launch_units / sizing.tiles).max(1);

    let mut row_offset = 0usize;
    while row_offset < rows {
        let block_rows = sizing
            .block_rows
            .min(launch_block_rows)
            .min(rows - row_offset);
        let scratch_len = block_rows
            .checked_mul(sizing.tiles)
            .expect("validated Boruvka scratch dimensions");
        let scratch_float = CubeTensor::new_contiguous(
            data.client.clone(),
            data.device.clone(),
            Shape::new([block_rows, sizing.tiles]),
            data.client.empty(
                scratch_len
                    .checked_mul(DType::F32.size())
                    .expect("validated scratch size"),
            ),
            DType::F32,
        );
        let scratch_index = CubeTensor::new_contiguous(
            data.client.clone(),
            data.device.clone(),
            Shape::new([block_rows, sizing.tiles]),
            data.client.empty(
                scratch_len
                    .checked_mul(DType::I32.size())
                    .expect("validated scratch size"),
            ),
            DType::I32,
        );
        let work_units = block_rows
            .checked_mul(sizing.tiles)
            .expect("validated Boruvka launch dimensions");
        for work_offset in (0..work_units).step_by(max_launch_units) {
            let count = checked_launch_count(
                (work_units - work_offset).min(max_launch_units),
                dim.x,
                max_cubes,
            );
            tile_candidates::launch(
                &client,
                CubeCount::Static(count, 1, 1),
                dim,
                data.clone().into_tensor_arg(),
                core_sq.clone().into_tensor_arg(),
                point_components.clone().into_tensor_arg(),
                u32::try_from(row_offset).expect("validated row offset"),
                u64::try_from(work_offset).expect("validated work offset"),
                tiles_u32,
                tile_width_u32,
                scratch_float.clone().into_tensor_arg(),
                scratch_index.clone().into_tensor_arg(),
                [
                    dtype_to_storage_type(DType::F32),
                    dtype_to_storage_type(DType::I32),
                ],
            );
        }
        let reduce_dim = cuscan_common::legal_cube_dim(&client, CubeDim::new_1d(256));
        let reduce_cubes = checked_launch_count(block_rows, reduce_dim.x, max_cubes);
        reduce_tiles::launch(
            &client,
            CubeCount::Static(reduce_cubes, 1, 1),
            reduce_dim,
            scratch_float.into_tensor_arg(),
            scratch_index.into_tensor_arg(),
            output_float.clone().into_tensor_arg(),
            output_index.clone().into_tensor_arg(),
            u32::try_from(row_offset).expect("validated row offset"),
            [
                dtype_to_storage_type(DType::F32),
                dtype_to_storage_type(DType::I32),
            ],
        );
        row_offset += block_rows;
    }
    (output_float, output_index)
}
