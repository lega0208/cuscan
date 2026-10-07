//! Direct CubeCL NN-Descent kernel for the CUDA standard-UMAP path.

use burn::{
    backend::{TensorMetadata, cubecl::dtype_to_storage_type, tensor::FloatTensor},
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

use crate::standard_umap::ann::candidate_width;

const ANN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

#[cube]
fn consider<F: Float, I: Int>(
    data: &Tensor<F>,
    row: usize,
    candidate: I,
    features: usize,
    n: usize,
    width: usize,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
) {
    let candidate_index = u32::cast_from(candidate) as usize;
    if candidate_index < n && candidate_index != row {
        let mut squared = F::new(0.0f32);
        for feature in 0..features {
            let delta = data[row * features + feature] - data[candidate_index * features + feature];
            squared += delta * delta;
        }
        let base = row * width;
        let mut position = width;
        let mut duplicate = false;
        for slot in 0..width {
            let old = indices[base + slot];
            if old == candidate {
                duplicate = true;
            }
            if position == width {
                let old_distance = distances[base + slot];
                let old_index = u32::cast_from(old) as usize;
                if squared < old_distance
                    || (squared == old_distance && candidate_index < old_index)
                {
                    position = slot;
                }
            }
        }
        if !duplicate && position < width {
            let mut slot = width - 1;
            while slot > position {
                indices[base + slot] = indices[base + slot - 1];
                distances[base + slot] = distances[base + slot - 1];
                slot -= 1;
            }
            indices[base + position] = candidate;
            distances[base + position] = squared;
        }
    }
}

#[cube(launch)]
fn ann_init<I: Int>(
    indices: &mut Tensor<I>,
    n: u32,
    width: u32,
    seed_base: u32,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if row < n {
        let rng_modulus = I::cast_from(2_147_483_647i32);
        let seed_base = I::cast_from(seed_base);
        let row_value = I::cast_from(row as i32);
        let mut state = if row_value >= rng_modulus - seed_base {
            row_value - (rng_modulus - seed_base)
        } else {
            row_value + seed_base
        };
        if state == I::cast_from(0i32) {
            state = I::cast_from(1i32);
        }
        for column in 0..width {
            let hi = state / I::cast_from(127_773i32);
            let lo = state - hi * I::cast_from(127_773i32);
            let test = lo * I::cast_from(16_807i32) - hi * I::cast_from(2_836i32);
            state = if test <= I::cast_from(0i32) {
                test + I::cast_from(2_147_483_647i32)
            } else {
                test
            };
            let t = state - I::cast_from(1i32);
            let j = n - 1 - width + column;
            let bound = I::cast_from((j + 1) as i32);
            let draw = t - (t / bound) * bound;
            let mut collision = false;
            for previous in 0..column {
                let previous_id = I::cast_from(indices[row * width + previous]);
                let previous_raw = if previous_id > row_value {
                    previous_id - I::cast_from(1i32)
                } else {
                    previous_id
                };
                if previous_raw == draw {
                    collision = true;
                }
            }
            let chosen = if collision {
                I::cast_from(j as i32)
            } else {
                draw
            };
            let candidate = if chosen >= row_value {
                chosen + I::cast_from(1i32)
            } else {
                chosen
            };
            indices[row * width + column] = candidate;
        }
    }
}

#[cube(launch)]
fn ann_step<F: Float, I: Int>(
    data: &Tensor<F>,
    candidates: &Tensor<I>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    width: u32,
    do_expand: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = data.shape(0);
    let features = data.shape(1);
    let width = width as usize;
    if row < n {
        let base = row * width;
        for slot in 0..width {
            output_indices[base + slot] = I::cast_from(n as u32);
            output_distances[base + slot] = F::new(3.402_823_5e38_f32);
        }
        for slot in 0..width {
            consider(
                data,
                row,
                candidates[base + slot],
                features,
                n,
                width,
                output_indices,
                output_distances,
            );
        }
        if do_expand != 0 {
            // Join each current neighbor's list (neighbor-of-neighbor proposals).
            for slot in 0..width {
                let neighbor = u32::cast_from(candidates[base + slot]) as usize;
                if neighbor < n {
                    for other in 0..width {
                        consider(
                            data,
                            row,
                            candidates[neighbor * width + other],
                            features,
                            n,
                            width,
                            output_indices,
                            output_distances,
                        );
                    }
                }
            }
            // Reverse joins: find rows whose candidate lists contain this row and
            // evaluate those rows' candidate lists, without an NxN scratch array.
            for source in 0..n {
                let mut reverse = false;
                for slot in 0..width {
                    if u32::cast_from(candidates[source * width + slot]) as usize == row {
                        reverse = true;
                    }
                }
                if reverse {
                    for slot in 0..width {
                        consider(
                            data,
                            row,
                            candidates[source * width + slot],
                            features,
                            n,
                            width,
                            output_indices,
                            output_distances,
                        );
                    }
                }
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
    let n = indices.shape(0);
    let width = width as usize;
    let k = k as usize;
    if row < n {
        for slot in 0..k {
            let source = row * width + slot;
            let output = row * k + slot;
            output_indices[output] = indices[source];
            output_distances[output] = F::sqrt(squared_distances[source]);
        }
    }
}

/// Approximate neighbors on CUDA. Candidate lists and all feature-distance
/// evaluations and deterministic candidate seeding reside on CubeCL.
pub(crate) fn approximate_neighbors(
    data: CubeTensor,
    k: u32,
    seed: u64,
    max_iterations: u32,
) -> (CubeTensor, CubeTensor) {
    let data = into_contiguous(data);
    let [n, features] = data.shape().dims::<2>();
    assert!(
        n >= 2 && features > 0,
        "ANN input requires at least two rows and one feature"
    );
    assert_eq!(data.dtype, DType::F32, "ANN requires float32 features");
    assert!(n <= i32::MAX as usize, "ANN row indices exceed i32");
    let k = usize::try_from(k).expect("ANN width exceeds usize");
    assert!(k > 0 && k < n, "ANN neighbor width must be in 1..N");
    let width = candidate_width(n, k);
    let elements = n
        .checked_mul(width)
        .expect("ANN candidate elements overflow usize");
    let index_bytes = elements
        .checked_mul(DType::I32.size())
        .expect("ANN index storage overflows usize");
    let distance_bytes = elements
        .checked_mul(DType::F32.size())
        .expect("ANN distance storage overflows usize");
    let mut candidates = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    ann_init::launch(
        &data.client,
        CubeCount::Static(
            u32::try_from(n)
                .expect("ANN row count exceeds CubeCL range")
                .div_ceil(ANN_CUBE_DIM.x),
            1,
            1,
        ),
        ANN_CUBE_DIM,
        candidates.clone().into_tensor_arg(),
        u32::try_from(n).expect("ANN row count exceeds CubeCL range"),
        u32::try_from(width).expect("ANN candidate width exceeds CubeCL range"),
        (seed % 2_147_483_646u64) as u32 + 1,
        dtype_to_storage_type(DType::I32),
    );
    let n_u32 = u32::try_from(n).expect("ANN row count exceeds CubeCL range");
    let width_u32 = u32::try_from(width).expect("ANN candidate width exceeds CubeCL range");
    let mut candidate_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(distance_bytes),
        DType::F32,
    );
    for iteration in 0..=max_iterations {
        let next_ids = CubeTensor::new_contiguous(
            data.client.clone(),
            data.device.clone(),
            Shape::new([n, width]),
            data.client.empty(index_bytes),
            DType::I32,
        );
        let next_distances = CubeTensor::new_contiguous(
            data.client.clone(),
            data.device.clone(),
            Shape::new([n, width]),
            data.client.empty(distance_bytes),
            DType::F32,
        );
        ann_step::launch(
            &data.client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            data.clone().into_tensor_arg(),
            candidates.into_tensor_arg(),
            next_ids.clone().into_tensor_arg(),
            next_distances.clone().into_tensor_arg(),
            width_u32,
            u32::from(iteration > 0),
            [
                dtype_to_storage_type(data.dtype),
                dtype_to_storage_type(DType::I32),
            ],
        );
        candidates = next_ids;
        candidate_distances = next_distances;
    }
    let output_elements = n
        .checked_mul(k)
        .expect("ANN output dimensions overflow usize");
    let output_indices = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, k]),
        data.client.empty(
            output_elements
                .checked_mul(DType::I32.size())
                .expect("ANN output allocation overflows usize"),
        ),
        DType::I32,
    );
    let output_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, k]),
        data.client.empty(
            output_elements
                .checked_mul(DType::F32.size())
                .expect("ANN output allocation overflows usize"),
        ),
        DType::F32,
    );
    ann_output::launch(
        &data.client,
        CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
        ANN_CUBE_DIM,
        candidates.into_tensor_arg(),
        candidate_distances.into_tensor_arg(),
        output_indices.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        width_u32,
        u32::try_from(k).expect("ANN width exceeds CubeCL range"),
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    (output_indices, output_distances)
}
