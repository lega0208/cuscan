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
fn reverse_first_hit<I: Int>(
    candidates: &Tensor<I>,
    source: usize,
    slot: usize,
    n: usize,
    width: usize,
) -> bool {
    let target = u32::cast_from(candidates[source * width + slot]) as usize;
    let mut first = target < n;
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
fn reverse_prefix<I: Int>(
    counts: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let n = counts.shape(0);
        let mut running = I::cast_from(0i32);
        offsets[0] = running;
        for target in 0..n {
            cursors[target].store(running);
            running += counts[target];
            offsets[target + 1] = running;
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
    reverse_offsets: &Tensor<I>,
    reverse_sources: &Tensor<I>,
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
            // Reverse joins traverse the complete CSR segment for this row.
            let start = u32::cast_from(reverse_offsets[row]) as usize;
            let end = u32::cast_from(reverse_offsets[row + 1]) as usize;
            for position in start..end {
                let source = u32::cast_from(reverse_sources[position]) as usize;
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
    let offset_len = n
        .checked_add(1)
        .expect("ANN reverse offsets length overflows usize");
    assert!(
        elements <= i32::MAX as usize,
        "ANN candidate source capacity and reverse CSR prefix exceed i32"
    );
    let index_bytes = elements
        .checked_mul(DType::I32.size())
        .expect("ANN index storage overflows usize");
    let distance_bytes = elements
        .checked_mul(DType::F32.size())
        .expect("ANN distance storage overflows usize");
    let reverse_count_bytes = n
        .checked_mul(DType::I32.size())
        .expect("ANN reverse count allocation overflows usize");
    let reverse_offset_bytes = offset_len
        .checked_mul(DType::I32.size())
        .expect("ANN reverse offset allocation overflows usize");
    let n_u32 = u32::try_from(n).expect("ANN row count exceeds CubeCL range");
    let width_u32 = u32::try_from(width).expect("ANN candidate width exceeds CubeCL range");
    let _feature_count_u32 =
        u32::try_from(features).expect("ANN feature count exceeds CubeCL range");
    let client = data.client.clone();
    let device = data.device.clone();
    let reverse_counts = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(reverse_count_bytes),
        DType::I32,
    );
    let reverse_offsets = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([offset_len]),
        client.empty(reverse_offset_bytes),
        DType::I32,
    );
    let reverse_cursors = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(reverse_count_bytes),
        DType::I32,
    );
    let reverse_sources = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([elements]),
        client.empty(index_bytes),
        DType::I32,
    );
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
        if iteration > 0 {
            reverse_zero::launch(
                &data.client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                reverse_counts.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            reverse_count::launch(
                &data.client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                candidates.clone().into_tensor_arg(),
                reverse_counts.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                dtype_to_storage_type(DType::I32),
            );
            reverse_prefix::launch(
                &data.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                reverse_counts.clone().into_tensor_arg(),
                reverse_offsets.clone().into_tensor_arg(),
                reverse_cursors.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            reverse_scatter::launch(
                &data.client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                candidates.clone().into_tensor_arg(),
                reverse_cursors.clone().into_tensor_arg(),
                reverse_sources.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                dtype_to_storage_type(DType::I32),
            );
        }
        ann_step::launch(
            &data.client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            data.clone().into_tensor_arg(),
            candidates.into_tensor_arg(),
            reverse_offsets.clone().into_tensor_arg(),
            reverse_sources.clone().into_tensor_arg(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeDevice;

    fn raw_tensor<T: burn::tensor::Element>(
        data: &[T],
        shape: Shape,
        dtype: DType,
        device: &CubeDevice,
        client: &cubecl::client::Client,
    ) -> CubeTensor {
        let tensor_data = burn::tensor::TensorData::new(data.to_vec(), shape.clone());
        assert_eq!(tensor_data.dtype(), dtype);
        let (bytes, _, _) = tensor_data.into_parts();
        let handle = client.create(bytes);
        CubeTensor::new_contiguous(client.clone(), device.clone(), shape, handle, dtype)
    }

    fn read_i32(tensor: &CubeTensor) -> Vec<i32> {
        let bytes = tensor.client.read_one_unchecked(tensor.handle.clone());
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_f32(tensor: &CubeTensor) -> Vec<f32> {
        let bytes = tensor.client.read_one_unchecked(tensor.handle.clone());
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn old_rescan_step(
        points: &[f32],
        candidates: &[i32],
        n: usize,
        features: usize,
        width: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut ids = vec![n as i32; n * width];
        let mut distances = vec![f32::MAX; n * width];
        for row in 0..n {
            let mut consider_candidate = |candidate: i32| {
                if candidate < 0 || candidate as usize >= n || candidate as usize == row {
                    return;
                }
                let candidate = candidate as usize;
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        points[row * features + feature] - points[candidate * features + feature];
                    squared += delta * delta;
                }
                let base = row * width;
                let mut position = width;
                let mut duplicate = false;
                for slot in 0..width {
                    let old = ids[base + slot];
                    duplicate |= old == candidate as i32;
                    if position == width
                        && (squared < distances[base + slot]
                            || (squared == distances[base + slot] && candidate < old as usize))
                    {
                        position = slot;
                    }
                }
                if !duplicate && position < width {
                    for slot in (position + 1..width).rev() {
                        ids[base + slot] = ids[base + slot - 1];
                        distances[base + slot] = distances[base + slot - 1];
                    }
                    ids[base + position] = candidate as i32;
                    distances[base + position] = squared;
                }
            };
            for slot in 0..width {
                consider_candidate(candidates[row * width + slot]);
            }
            for slot in 0..width {
                let neighbor = candidates[row * width + slot];
                if neighbor >= 0 && (neighbor as usize) < n {
                    for other in 0..width {
                        consider_candidate(candidates[neighbor as usize * width + other]);
                    }
                }
            }
            for source in 0..n {
                if (0..width).any(|slot| candidates[source * width + slot] == row as i32) {
                    for slot in 0..width {
                        consider_candidate(candidates[source * width + slot]);
                    }
                }
            }
        }
        (ids, distances)
    }

    #[test]
    fn reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers() {
        let Some(_device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        let raw_device = CubeDevice::cuda(0).expect("CUDA device was just detected");
        let client = raw_device.client();
        let n = 67usize;
        let width = 13usize;
        let elements = n * width;
        let count_bytes = n * DType::I32.size();
        let offset_bytes = (n + 1) * DType::I32.size();
        let counts = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let offsets = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n + 1]),
            client.empty(offset_bytes),
            DType::I32,
        );
        let cursors = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let sources = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([elements]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let mut state = 0x5eed_1234u32;
        let mut matrices = Vec::new();
        let mut hub = vec![-1i32; elements];
        for source in 0..n {
            hub[source * width] = 0;
            hub[source * width + 1] = 0;
            hub[source * width + 2] = (source % n) as i32;
            hub[source * width + 3] = n as i32;
            hub[source * width + 4] = (n + 9) as i32;
            hub[source * width + 5] = -2;
        }
        matrices.push(hub);
        let mut random = vec![0i32; elements];
        for value in &mut random {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = match state % 8 {
                0 => -1,
                1 => -2,
                2 => n as i32,
                3 => (n + 17) as i32,
                _ => ((state >> 8) as usize % n) as i32,
            };
        }
        matrices.push(random);
        let mut changed_support = vec![-1i32; elements];
        for source in 0..n {
            for slot in 0..width {
                if (source + slot) % 7 == 0 {
                    changed_support[source * width + slot] =
                        ((source * 11 + slot * 3 + 1) % n) as i32;
                }
            }
        }
        matrices.push(changed_support);

        let launch_dtype = dtype_to_storage_type(DType::I32);
        let n_u32 = u32::try_from(n).unwrap();
        let width_u32 = u32::try_from(width).unwrap();
        for (iteration, matrix) in matrices.into_iter().enumerate() {
            let candidates = raw_tensor(
                &matrix,
                Shape::new([elements]),
                DType::I32,
                &raw_device,
                &client,
            );
            reverse_zero::launch(
                &client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                counts.clone().into_tensor_arg(),
                launch_dtype,
            );
            reverse_count::launch(
                &client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                candidates.clone().into_tensor_arg(),
                counts.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                launch_dtype,
            );
            reverse_prefix::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                counts.clone().into_tensor_arg(),
                offsets.clone().into_tensor_arg(),
                cursors.clone().into_tensor_arg(),
                launch_dtype,
            );
            reverse_scatter::launch(
                &client,
                CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
                ANN_CUBE_DIM,
                candidates.into_tensor_arg(),
                cursors.clone().into_tensor_arg(),
                sources.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                launch_dtype,
            );

            let count_values = read_i32(&counts);
            let offset_values = read_i32(&offsets);
            let cursor_values = read_i32(&cursors);
            let source_values = read_i32(&sources);
            let mut expected = vec![Vec::<i32>::new(); n];
            for source in 0..n {
                for slot in 0..width {
                    let target = matrix[source * width + slot];
                    if target >= 0
                        && (target as usize) < n
                        && !matrix[source * width..source * width + slot].contains(&target)
                    {
                        expected[target as usize].push(source as i32);
                    }
                }
            }
            let count_total: usize = count_values.iter().map(|&count| count as usize).sum();
            assert_eq!(
                count_total, offset_values[n] as usize,
                "prefix total, iteration {iteration}"
            );
            for target in 0..n {
                let start = offset_values[target] as usize;
                let end = offset_values[target + 1] as usize;
                assert_eq!(
                    count_values[target] as usize,
                    end - start,
                    "count, target {target}, iteration {iteration}"
                );
                assert_eq!(
                    cursor_values[target],
                    offset_values[target + 1],
                    "cursor completion, target {target}, iteration {iteration}"
                );
                let mut actual = source_values[start..end].to_vec();
                actual.sort_unstable();
                expected[target].sort_unstable();
                assert_eq!(
                    actual, expected[target],
                    "source set, target {target}, iteration {iteration}"
                );
            }
            if iteration == 0 {
                assert_eq!(
                    count_values[0], n as i32,
                    "fixture must exercise uncapped hub indegree"
                );
            }
        }
    }

    #[test]
    fn ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates() {
        let Some(_device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        let n = 9usize;
        let width = 6usize;
        let features = 2usize;
        let points = (0..n)
            .flat_map(|row| [row as f32 * 0.25, ((row * 5) % 7) as f32 * 0.5])
            .collect::<Vec<_>>();
        let raw_device = CubeDevice::cuda(0).expect("CUDA device was just detected");
        let client = raw_device.client();
        let data = raw_tensor(
            &points,
            Shape::new([n, features]),
            DType::F32,
            &raw_device,
            &client,
        );
        let mut candidates = vec![n as i32; n * width];
        for source in 0..n {
            let base = source * width;
            candidates[base..base + width].copy_from_slice(&[
                0,
                0,
                ((source + 2) % n) as i32,
                -1,
                (n + 4) as i32,
                if source % 2 == 0 { 3 } else { -2 },
            ]);
        }
        // Make row zero an uncapped reverse hub while retaining duplicate and invalid slots.
        for source in 0..n {
            candidates[source * width] = 0;
        }
        let candidate_tensor = raw_tensor(
            &candidates,
            Shape::new([n, width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let elements = n * width;
        let count_bytes = n * DType::I32.size();
        let counts = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let offsets = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n + 1]),
            client.empty((n + 1) * DType::I32.size()),
            DType::I32,
        );
        let cursors = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let sources = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([elements]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let n_u32 = u32::try_from(n).unwrap();
        let width_u32 = u32::try_from(width).unwrap();
        let dtype = dtype_to_storage_type(DType::I32);
        reverse_zero::launch(
            &client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            counts.clone().into_tensor_arg(),
            dtype,
        );
        reverse_count::launch(
            &client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            candidate_tensor.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype,
        );
        reverse_prefix::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            cursors.clone().into_tensor_arg(),
            dtype,
        );
        reverse_scatter::launch(
            &client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            candidate_tensor.clone().into_tensor_arg(),
            cursors.clone().into_tensor_arg(),
            sources.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype,
        );
        let output_ids = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let output_distances = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::F32.size()),
            DType::F32,
        );
        ann_step::launch(
            &client,
            CubeCount::Static(n_u32.div_ceil(ANN_CUBE_DIM.x), 1, 1),
            ANN_CUBE_DIM,
            data.clone().into_tensor_arg(),
            candidate_tensor.into_tensor_arg(),
            offsets.into_tensor_arg(),
            sources.into_tensor_arg(),
            output_ids.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            width_u32,
            1,
            [dtype_to_storage_type(DType::F32), dtype],
        );
        let actual_ids = read_i32(&output_ids);
        let actual_distances = read_f32(&output_distances);
        let expected = old_rescan_step(&points, &candidates, n, features, width);
        assert_eq!(actual_ids, expected.0);
        assert_eq!(
            actual_distances
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .1
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
}
