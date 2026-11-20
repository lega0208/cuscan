use super::*;
use burn::{
    backend::cubecl::dtype_to_storage_type,
    tensor::{DType, Shape},
};
use burn_cubecl::CubeDevice;
use burn_cubecl::tensor::CubeTensor;

#[cube(launch)]
fn distance_probe<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    pairs: &Tensor<i32>,
    output: &mut Tensor<F>,
    features: u32,
    #[comptime] vector_width: usize,
    #[define(F)] _dtype: ElemType,
) {
    let index = ABSOLUTE_POS_X as usize;
    if index < output.shape(0) {
        let left = u32::cast_from(pairs[index * 2]) as usize;
        let right = u32::cast_from(pairs[index * 2 + 1]) as usize;
        output[index] = squared_l2_r8::<F, N>(data, left, right, features as usize, vector_width);
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
    assert_eq!(data.dtype(), dtype);
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

fn read_f32(tensor: &CubeTensor) -> Vec<f32> {
    tensor
        .client
        .read_one_unchecked(tensor.handle.clone())
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_le_bytes(*bytes))
        .collect()
}

fn fusion_witness() -> (f32, f32) {
    let mut state = 0x9e37_79b9u32;
    for _ in 0..2_000_000 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let left = f32::from_bits((state & 0x807f_ffff) | 0x3f00_0000);
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let right = f32::from_bits((state & 0x807f_ffff) | 0x3f00_0000);
        let difference = left - right;
        let fused = difference.mul_add(difference, 1.0);
        let unfused = difference * difference + 1.0;
        if fused.to_bits() != unfused.to_bits() {
            return (left, right);
        }
    }
    panic!("failed to find a fused multiply-add witness");
}

fn distance_fixture(device: CubeDevice) {
    const FEATURES: &[usize] = &[
        1, 3, 4, 7, 8, 9, 12, 15, 16, 17, 31, 32, 33, 64, 128, 255, 256, 257,
    ];
    let (witness_left, witness_right) = fusion_witness();
    for &features in FEATURES {
        let mut witness_a = vec![0.0; features];
        let mut witness_b = vec![0.0; features];
        witness_a[0] = 1.0;
        if features > 8 {
            witness_a[8] = witness_left;
            witness_b[8] = witness_right;
        }
        let mut cancellation_a = vec![0.0; features];
        let mut cancellation_b = vec![0.0; features];
        for feature in 0..features {
            match feature % 4 {
                0 => {
                    cancellation_a[feature] = 16_777_216.0;
                    cancellation_b[feature] = 16_777_215.0;
                }
                1 => {
                    cancellation_a[feature] = f32::from_bits(1);
                }
                2 => {
                    cancellation_a[feature] = -0.0;
                    cancellation_b[feature] = 0.0;
                }
                _ => {
                    cancellation_a[feature] = (feature as f32 + 0.25) * 0.125;
                    cancellation_b[feature] = -0.375;
                }
            }
        }
        let overflow_a = vec![f32::MAX; features];
        let overflow_b = vec![-f32::MAX; features];
        let mut nan_a = vec![0.0; features];
        let nan_b = vec![0.0; features];
        nan_a[features - 1] = f32::NAN;
        let rows = [
            &witness_a,
            &witness_b,
            &cancellation_a,
            &cancellation_b,
            &overflow_a,
            &overflow_b,
            &nan_a,
            &nan_b,
        ];
        let values = rows
            .into_iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let pairs = vec![0i32, 1, 1, 0, 2, 3, 4, 5, 6, 7];
        let expected = [
            host_squared_l2_r8(&witness_a, &witness_b),
            host_squared_l2_r8(&witness_b, &witness_a),
            host_squared_l2_r8(&cancellation_a, &cancellation_b),
            host_squared_l2_r8(&overflow_a, &overflow_b),
            host_squared_l2_r8(&nan_a, &nan_b),
        ];
        let input = tensor(&device, &[8, features], values, DType::F32);
        let pair_tensor = tensor(&device, &[5, 2], pairs, DType::I32);
        for width in [1usize, 4, 8]
            .into_iter()
            .filter(|width| features.is_multiple_of(*width))
        {
            let output = tensor(&device, &[5], vec![0.0f32; 5], DType::F32);
            let client = device.client();
            let data_arg = if width == 1 {
                input.clone().into_tensor_arg()
            } else {
                // SAFETY: input is fresh contiguous F32 with 8*features values;
                // this view is aligned by construction and has 8*features/width vectors.
                unsafe {
                    cubecl::prelude::TensorArg::from_raw_parts(
                        input.handle.clone(),
                        [1].into(),
                        [8 * features / width].into(),
                    )
                }
            };
            distance_probe::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(32),
                width as VectorSize,
                data_arg,
                pair_tensor.clone().into_tensor_arg(),
                output.clone().into_tensor_arg(),
                features as u32,
                width,
                dtype_to_storage_type(DType::F32),
            );
            let actual = read_f32(&output);
            for index in 0..4 {
                assert_eq!(
                    actual[index].to_bits(),
                    expected[index].to_bits(),
                    "features={features}, width={width}, pair={index}"
                );
            }
            assert_eq!(
                actual[0].to_bits(),
                actual[1].to_bits(),
                "symmetry, features={features}, width={width}"
            );
            assert!(
                actual[4].is_nan(),
                "NaN input, features={features}, width={width}"
            );
            if features > 8 {
                let difference = witness_left - witness_right;
                assert_ne!(
                    difference.mul_add(difference, 1.0).to_bits(),
                    (difference * difference + 1.0).to_bits()
                );
                assert_eq!(
                    actual[0].to_bits(),
                    difference.mul_add(difference, 1.0).to_bits(),
                    "fusion witness, features={features}, width={width}"
                );
            }
        }
    }
}

#[test]
fn ann_distance_r8_oracle_and_symmetry_cpu() {
    distance_fixture(CubeDevice::Cpu(Default::default()));
}

#[test]
fn ann_distance_r8_oracle_and_symmetry_cuda() {
    let count = crate::cuda_backend::try_device_count()
        .unwrap_or_else(|error| panic!("CUDA discovery failed: {error}"));
    if count == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    }
    distance_fixture(CubeDevice::Cuda(cubecl::device::CudaDevice { index: 0 }));
}
