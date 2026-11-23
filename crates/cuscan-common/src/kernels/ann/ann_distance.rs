//! Canonical R8 squared Euclidean distance used by every ANN distance path.

#[cfg(test)]
#[path = "ann_distance_tests.rs"]
mod tests;

#[cfg(test)]
mod cuda_group_tests {
    use super::*;
    use burn::{
        backend::cubecl::dtype_to_storage_type,
        tensor::{DType, Shape},
    };
    use burn_cubecl::{CubeDevice, tensor::CubeTensor};

    #[cube(launch)]
    fn group_probe<F: Float, N: Size>(
        data: &Tensor<Vector<F, N>>,
        output: &mut Tensor<F>,
        oracle: &mut Tensor<F>,
        features: u32,
        #[comptime] group: usize,
        #[comptime] width: usize,
        #[comptime] unroll: usize,
        #[define(F)] _dtype: ElemType,
    ) {
        let lane = UNIT_POS_PLANE as usize;
        let pair = lane / group;
        let left = pair % 4 * 2;
        let right = left + 1;
        let valid = pair % 3 != 1;
        output[lane] = squared_l2_r8_group::<F, N>(
            data,
            left,
            right,
            features as usize,
            valid,
            group,
            width,
            unroll,
        );
        if valid {
            oracle[lane] = squared_l2_r8::<F, N>(data, left, right, features as usize, width);
        } else {
            oracle[lane] = F::new(0.0f32);
        }
    }

    #[cube(launch)]
    fn triangle_probe(output: &mut Tensor<u32>, members: u32, count: u32) {
        let local = ABSOLUTE_POS_X;
        if local < count {
            let (row, column) = triangle_decode(local, members);
            output[local as usize * 2] = row;
            output[local as usize * 2 + 1] = column;
        }
    }

    fn tensor<T: burn::tensor::Element>(
        device: &CubeDevice,
        shape: &[usize],
        values: Vec<T>,
        dtype: DType,
    ) -> CubeTensor {
        let shape = Shape::from(shape.to_vec());
        let data = burn::tensor::TensorData::new(values, shape.clone());
        let (bytes, _, _) = data.into_parts();
        let client = device.client();
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape,
            client.create(bytes),
            dtype,
        )
    }

    fn read_words(tensor: &CubeTensor) -> Vec<u32> {
        tensor
            .client
            .read_one_unchecked(tensor.handle.clone())
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect()
    }

    fn cuda() -> Option<CubeDevice> {
        let count = crate::cuda_backend::try_device_count()
            .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
        if count == 0 {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
            );
            eprintln!("SKIPPED: no CUDA device");
            None
        } else {
            Some(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }))
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn group_r8_matches_scalar_and_host_cuda() {
        let Some(device) = cuda() else { return };
        for features in [
            1usize, 2, 3, 4, 7, 8, 9, 12, 15, 16, 31, 32, 33, 50, 255, 256, 260,
        ] {
            let mut rows = vec![vec![0.0f32; features]; 8];
            for f in 0..features {
                rows[0][f] = (f as f32 * 0.125) - 2.0;
                rows[1][f] = (f as f32 * -0.0625) + 1.0;
                rows[2][f] = if f % 2 == 0 { -0.0 } else { f32::from_bits(1) };
                rows[3][f] = rows[2][f];
                rows[4][f] = f32::MAX;
                rows[5][f] = -f32::MAX;
                rows[6][f] = if f == features - 1 { f32::NAN } else { 0.0 };
            }
            let input = tensor(
                &device,
                &[8, features],
                rows.iter().flatten().copied().collect(),
                DType::F32,
            );
            for (group, width) in [(2usize, 4usize), (4, 1), (8, 1)] {
                if features % width != 0 {
                    continue;
                }
                for unroll in [1usize, 2] {
                    let output = tensor(&device, &[32], vec![0.0f32; 32], DType::F32);
                    let oracle = tensor(&device, &[32], vec![0.0f32; 32], DType::F32);
                    let data_arg = if width == 1 {
                        input.clone().into_tensor_arg()
                    } else {
                        // SAFETY: contiguous F32 input is aligned and its length is divisible by four.
                        unsafe {
                            TensorArg::from_raw_parts(
                                input.handle.clone(),
                                [1].into(),
                                [8 * features / width].into(),
                            )
                        }
                    };
                    group_probe::launch(
                        &device.client(),
                        CubeCount::Static(1, 1, 1),
                        CubeDim::new_1d(32),
                        width as VectorSize,
                        data_arg,
                        output.clone().into_tensor_arg(),
                        oracle.clone().into_tensor_arg(),
                        features as u32,
                        group,
                        width,
                        unroll,
                        dtype_to_storage_type(DType::F32),
                    );
                    let actual = read_words(&output);
                    let reference = read_words(&oracle);
                    for lane in 0..32 {
                        let pair = lane / group;
                        let row = pair % 4 * 2;
                        let host = if pair % 3 != 1 {
                            host_squared_l2_r8(&rows[row], &rows[row + 1])
                        } else {
                            0.0
                        };
                        let result = f32::from_bits(actual[lane]);
                        let scalar = f32::from_bits(reference[lane]);
                        if host.is_nan() {
                            assert!(result.is_nan() && scalar.is_nan());
                        } else {
                            assert_eq!(
                                actual[lane],
                                host.to_bits(),
                                "features={features} group={group} unroll={unroll} lane={lane}"
                            );
                            assert_eq!(
                                reference[lane],
                                host.to_bits(),
                                "scalar features={features} group={group} lane={lane}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn triangle_decode_all_locals_cuda() {
        let Some(device) = cuda() else { return };
        for members in [2u32, 3, 30, 31, 60, 61, 150, 255, 256] {
            let count = members * (members - 1) / 2;
            let output = tensor(
                &device,
                &[count as usize * 2],
                vec![0u32; count as usize * 2],
                DType::U32,
            );
            triangle_probe::launch(
                &device.client(),
                CubeCount::Static(count.div_ceil(32), 1, 1),
                CubeDim::new_1d(32),
                output.clone().into_tensor_arg(),
                members,
                count,
            );
            let actual = read_words(&output);
            for local in 0..count {
                let expected = host_triangle_decode(local, members);
                assert_eq!(
                    (actual[local as usize * 2], actual[local as usize * 2 + 1]),
                    expected,
                    "members={members} local={local}"
                );
            }
        }
    }
}

use cubecl::prelude::*;

/// Every plane lane must call this under uniform control flow with the same
/// feature count and with safe, in-row ids, including for invalid slots.
#[cube]
pub(super) fn squared_l2_r8_group<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right: usize,
    features: usize,
    valid: bool,
    #[comptime] group: usize,
    #[comptime] binding_width: usize,
    #[comptime] unroll: usize,
) -> F {
    comptime![assert!(
        (group == 2 && binding_width == 4) || ((group == 4 || group == 8) && binding_width == 1)
    )];
    comptime![assert!(unroll == 1 || unroll == 2)];
    let lane = UNIT_POS_PLANE as usize % group;
    let steps = features.div_ceil(8);
    let mut a = Vector::<F, N>::new(F::new(0.0f32));
    let mut b = Vector::<F, N>::new(F::new(0.0f32));
    let mut step = 0usize;
    while step < steps {
        let (d0, e0) = group_deltas::<F, N>(
            data,
            left,
            right,
            features,
            valid,
            lane,
            step,
            group,
            binding_width,
        );
        let mut d1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut e1 = Vector::<F, N>::new(F::new(0.0f32));
        if comptime![unroll == 2] && step + 1 < steps {
            let (d, e) = group_deltas::<F, N>(
                data,
                left,
                right,
                features,
                valid,
                lane,
                step + 1,
                group,
                binding_width,
            );
            d1 = d;
            e1 = e;
        }
        a = fma(d0, d0, a);
        b = fma(e0, e0, b);
        if comptime![unroll == 2] {
            a = fma(d1, d1, a);
            b = fma(e1, e1, b);
        }
        step += unroll;
    }
    let mut sum = if comptime![group == 2] {
        (a.extract(0usize) + a.extract(1usize)) + (a.extract(2usize) + a.extract(3usize))
    } else if comptime![group == 4] {
        a.extract(0usize) + b.extract(0usize)
    } else {
        a.extract(0usize)
    };
    sum += plane_shuffle_xor(sum, 1u32);
    if comptime![group >= 4] {
        sum += plane_shuffle_xor(sum, 2u32);
    }
    if comptime![group == 8] {
        sum += plane_shuffle_xor(sum, 4u32);
    }
    sum
}

#[cube]
fn group_deltas<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right: usize,
    features: usize,
    valid: bool,
    lane: usize,
    step: usize,
    #[comptime] group: usize,
    #[comptime] binding_width: usize,
) -> (Vector<F, N>, Vector<F, N>) {
    let mut a = Vector::<F, N>::new(F::new(0.0f32));
    let mut b = Vector::<F, N>::new(F::new(0.0f32));
    let feature = step * 8 + lane * comptime![8 / group];
    if valid && feature < features {
        let index = feature / binding_width;
        let stride = features / binding_width;
        a = data[left * stride + index] - data[right * stride + index];
    }
    if comptime![group == 4] && valid && feature + 1 < features {
        b = data[left * features + feature + 1] - data[right * features + feature + 1];
    }
    (a, b)
}

#[cube]
pub(super) fn triangle_decode(local: u32, members: u32) -> (u32, u32) {
    let width = 2 * members - 1;
    let estimate =
        (f32::cast_from(width) - f32::sqrt(f32::cast_from(width * width - 8 * local))) * 0.5;
    let mut row = u32::cast_from(estimate).min(members - 2);
    while (row + 1) * (2 * members - row - 2) / 2 <= local {
        row += 1;
    }
    while row * (2 * members - row - 1) / 2 > local {
        row -= 1;
    }
    let start = row * (2 * members - row - 1) / 2;
    (row, row + 1 + local - start)
}

#[cfg(test)]
pub(super) fn host_triangle_decode(local: u32, members: u32) -> (u32, u32) {
    let width = 2 * members as u64 - 1;
    let discriminant = width * width - 8 * local as u64;
    let estimate = (width as f32 - (discriminant as f32).sqrt()) * 0.5;
    let mut row = (estimate as u32).min(members - 2);
    let start = |i: u32| i as u64 * (2 * members as u64 - i as u64 - 1) / 2;
    while start(row + 1) <= local as u64 {
        row += 1;
    }
    while start(row) > local as u64 {
        row -= 1;
    }
    (row, row + 1 + local - start(row) as u32)
}

#[cfg(test)]
#[test]
fn triangle_decode_host_exhaustive_and_boundaries() {
    let mut max_corrections = 0u32;
    let mut check = |members: u32, row: u32, local: u32| {
        let start = row as u64 * (2 * members as u64 - row as u64 - 1) / 2;
        assert_eq!(
            host_triangle_decode(local, members),
            (row, row + 1 + local - start as u32)
        );
        let width = 2 * members as u64 - 1;
        let estimate =
            ((width as f32 - ((width * width - 8 * local as u64) as f32).sqrt()) * 0.5) as u32;
        max_corrections = max_corrections.max(estimate.min(members - 2).abs_diff(row));
    };
    for members in 2..=256u32 {
        let mut local = 0;
        for row in 0..members - 1 {
            for _ in row + 1..members {
                check(members, row, local);
                local += 1;
            }
        }
    }
    for members in [257u32, 1024, 32768, 65536] {
        for row in 0..members - 1 {
            let start = row as u64 * (2 * members as u64 - row as u64 - 1) / 2;
            let end = start + (members - row - 2) as u64;
            check(members, row, start as u32);
            check(members, row, end as u32);
        }
    }
    eprintln!("maximum triangle decode corrections: {max_corrections}");
}

/// Accumulate features in eight independent chains and reduce them in the
/// fixed R8 tree. Width one uses scalar chains; widths four/eight load vector
/// chunks and preserve the same per-lane R8 order.
// Keep compile-time width specialization separate from runtime chunk parity.
#[allow(clippy::if_same_then_else)]
#[cube]
pub(super) fn squared_l2_r8<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right: usize,
    features: usize,
    #[comptime] vector_width: usize,
) -> F {
    let mut s0 = F::new(0.0f32);
    let mut s1 = F::new(0.0f32);
    let mut s2 = F::new(0.0f32);
    let mut s3 = F::new(0.0f32);
    let mut s4 = F::new(0.0f32);
    let mut s5 = F::new(0.0f32);
    let mut s6 = F::new(0.0f32);
    let mut s7 = F::new(0.0f32);
    if comptime![vector_width == 1] {
        let chunks = features / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let d0 = data[left * features + base].extract(0usize)
                - data[right * features + base].extract(0usize);
            let d1 = data[left * features + base + 1].extract(0usize)
                - data[right * features + base + 1].extract(0usize);
            let d2 = data[left * features + base + 2].extract(0usize)
                - data[right * features + base + 2].extract(0usize);
            let d3 = data[left * features + base + 3].extract(0usize)
                - data[right * features + base + 3].extract(0usize);
            let d4 = data[left * features + base + 4].extract(0usize)
                - data[right * features + base + 4].extract(0usize);
            let d5 = data[left * features + base + 5].extract(0usize)
                - data[right * features + base + 5].extract(0usize);
            let d6 = data[left * features + base + 6].extract(0usize)
                - data[right * features + base + 6].extract(0usize);
            let d7 = data[left * features + base + 7].extract(0usize)
                - data[right * features + base + 7].extract(0usize);
            s0 = fma(d0, d0, s0);
            s1 = fma(d1, d1, s1);
            s2 = fma(d2, d2, s2);
            s3 = fma(d3, d3, s3);
            s4 = fma(d4, d4, s4);
            s5 = fma(d5, d5, s5);
            s6 = fma(d6, d6, s6);
            s7 = fma(d7, d7, s7);
        }
        let tail = chunks * 8;
        if tail < features {
            let d = data[left * features + tail].extract(0usize)
                - data[right * features + tail].extract(0usize);
            s0 = fma(d, d, s0);
        }
        if tail + 1 < features {
            let d = data[left * features + tail + 1].extract(0usize)
                - data[right * features + tail + 1].extract(0usize);
            s1 = fma(d, d, s1);
        }
        if tail + 2 < features {
            let d = data[left * features + tail + 2].extract(0usize)
                - data[right * features + tail + 2].extract(0usize);
            s2 = fma(d, d, s2);
        }
        if tail + 3 < features {
            let d = data[left * features + tail + 3].extract(0usize)
                - data[right * features + tail + 3].extract(0usize);
            s3 = fma(d, d, s3);
        }
        if tail + 4 < features {
            let d = data[left * features + tail + 4].extract(0usize)
                - data[right * features + tail + 4].extract(0usize);
            s4 = fma(d, d, s4);
        }
        if tail + 5 < features {
            let d = data[left * features + tail + 5].extract(0usize)
                - data[right * features + tail + 5].extract(0usize);
            s5 = fma(d, d, s5);
        }
        if tail + 6 < features {
            let d = data[left * features + tail + 6].extract(0usize)
                - data[right * features + tail + 6].extract(0usize);
            s6 = fma(d, d, s6);
        }
    } else {
        let lanes = vector_width;
        let vector_count = features / lanes;
        let mut acc0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut acc1 = Vector::<F, N>::new(F::new(0.0f32));
        let row_vectors = features / lanes;
        for chunk in 0..vector_count {
            let delta = data[left * row_vectors + chunk] - data[right * row_vectors + chunk];
            if comptime![vector_width == 8] {
                acc0 = fma(delta, delta, acc0);
            } else if chunk % 2 == 0 {
                acc0 = fma(delta, delta, acc0);
            } else {
                acc1 = fma(delta, delta, acc1);
            }
        }
        if comptime![vector_width == 8] {
            s0 = acc0.extract(0usize);
            s1 = acc0.extract(1usize);
            s2 = acc0.extract(2usize);
            s3 = acc0.extract(3usize);
            s4 = acc0.extract(4usize);
            s5 = acc0.extract(5usize);
            s6 = acc0.extract(6usize);
            s7 = acc0.extract(7usize);
        } else {
            s0 = acc0.extract(0usize);
            s1 = acc0.extract(1usize);
            s2 = acc0.extract(2usize);
            s3 = acc0.extract(3usize);
            s4 = acc1.extract(0usize);
            s5 = acc1.extract(1usize);
            s6 = acc1.extract(2usize);
            s7 = acc1.extract(3usize);
        }
    }
    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

/// Four independent fixed-R8 distances. Width one deliberately repeats the
/// scalar pair kernel; vector widths share the left-hand vector loads.
// Keep compile-time width specialization separate from runtime chunk parity.
#[allow(clippy::if_same_then_else)]
#[cube]
pub(super) fn squared_l2_r8_x4<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right0: usize,
    right1: usize,
    right2: usize,
    right3: usize,
    features: usize,
    #[comptime] vector_width: usize,
) -> (F, F, F, F) {
    if comptime![vector_width == 1] {
        (
            squared_l2_r8::<F, N>(data, left, right0, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right1, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right2, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right3, features, vector_width),
        )
    } else {
        let mut a0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a2 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a3 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b2 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b3 = Vector::<F, N>::new(F::new(0.0f32));
        let row_vectors = features / vector_width;
        for chunk in 0..row_vectors {
            let value = data[left * row_vectors + chunk];
            let d0 = value - data[right0 * row_vectors + chunk];
            let d1 = value - data[right1 * row_vectors + chunk];
            let d2 = value - data[right2 * row_vectors + chunk];
            let d3 = value - data[right3 * row_vectors + chunk];
            if comptime![vector_width == 8] {
                a0 = fma(d0, d0, a0);
                a1 = fma(d1, d1, a1);
                a2 = fma(d2, d2, a2);
                a3 = fma(d3, d3, a3);
            } else if chunk % 2 == 0 {
                a0 = fma(d0, d0, a0);
                a1 = fma(d1, d1, a1);
                a2 = fma(d2, d2, a2);
                a3 = fma(d3, d3, a3);
            } else {
                b0 = fma(d0, d0, b0);
                b1 = fma(d1, d1, b1);
                b2 = fma(d2, d2, b2);
                b3 = fma(d3, d3, b3);
            }
        }
        (
            reduce_r8_vectors::<F, N>(a0, b0, vector_width),
            reduce_r8_vectors::<F, N>(a1, b1, vector_width),
            reduce_r8_vectors::<F, N>(a2, b2, vector_width),
            reduce_r8_vectors::<F, N>(a3, b3, vector_width),
        )
    }
}

#[cube]
fn reduce_r8_vectors<F: Float, N: Size>(
    a: Vector<F, N>,
    b: Vector<F, N>,
    #[comptime] width: usize,
) -> F {
    let low = (a.extract(0usize) + a.extract(1usize)) + (a.extract(2usize) + a.extract(3usize));
    if comptime![width == 8] {
        low + ((a.extract(4usize) + a.extract(5usize)) + (a.extract(6usize) + a.extract(7usize)))
    } else {
        low + ((b.extract(0usize) + b.extract(1usize)) + (b.extract(2usize) + b.extract(3usize)))
    }
}

#[cfg(test)]
pub(super) fn host_squared_l2_r8(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sums = [0.0f32; 8];
    for (feature, (&left, &right)) in a.iter().zip(b).enumerate() {
        let difference = left - right;
        sums[feature % 8] = difference.mul_add(difference, sums[feature % 8]);
    }
    ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
}
