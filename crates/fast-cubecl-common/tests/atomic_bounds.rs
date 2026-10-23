//! M0: prove nonnegative f32 bounds can use signed integer atomics on each runtime.

use cubecl::Device;
use cubecl::prelude::*;
use fast_cubecl_common::cuda_backend;

#[cube]
fn encode_bound(value: f32) -> i32 {
    let bits = i32::reinterpret(value);
    // Integer canonicalization survives fast-math folding that can erase `value + 0.0`.
    if bits == i32::MIN { 0 } else { bits }
}

#[cube(launch)]
fn round_trip(values: &[f32], bits: &mut [i32], decoded: &mut [f32]) {
    let index = ABSOLUTE_POS;
    if index < values.len() {
        let encoded = encode_bound(values[index]);
        bits[index] = encoded;
        decoded[index] = f32::reinterpret(encoded);
    }
}

#[cube(launch)]
fn contended_min(values: &[f32], bound: &[Atomic<i32>]) {
    let index = ABSOLUTE_POS;
    if index < values.len() {
        let encoded = encode_bound(values[index]);
        bound[0].fetch_min(encoded);
    }
}

#[cube(launch)]
fn read_bound(bound: &[Atomic<i32>], decoded: &mut [f32]) {
    if ABSOLUTE_POS == 0 {
        decoded[0] = f32::reinterpret(bound[0].load());
    }
}

fn check_encoding(client: &Client) {
    let values = [0.0, -0.0, f32::from_bits(1), 1.0, f32::MAX, f32::INFINITY];
    let expected = [0_i32, 0, 1, 0x3f80_0000, 0x7f7f_ffff, 0x7f80_0000];
    let input = client.create_from_slice(f32::as_bytes(&values));
    let bits = client.create_from_slice(i32::as_bytes(&[0; 6]));
    let decoded = client.create_from_slice(f32::as_bytes(&[0.0; 6]));
    round_trip::launch(
        client,
        CubeCount::new_1d(2),
        CubeDim::new_1d(4),
        unsafe { BufferArg::from_raw_parts(input, values.len()) },
        unsafe { BufferArg::from_raw_parts(bits.clone(), values.len()) },
        unsafe { BufferArg::from_raw_parts(decoded.clone(), values.len()) },
    );
    assert_eq!(i32::from_bytes(&client.read_one_unchecked(bits)), &expected);
    assert_eq!(
        i32::from_bytes(&client.read_one_unchecked(decoded)),
        &expected
    );
}

fn check_contention(client: &Client) {
    let cases = [
        (
            vec![f32::INFINITY, f32::MAX, 1.0, f32::from_bits(1), -0.0, 0.0],
            0_i32,
        ),
        (
            vec![f32::INFINITY, f32::MAX, 1.0, f32::MAX, f32::INFINITY],
            0x3f80_0000,
        ),
        (
            vec![f32::MAX, f32::from_bits(1), f32::INFINITY, 1.0, f32::MAX],
            1,
        ),
        (
            vec![f32::MAX, f32::INFINITY, f32::MAX, f32::INFINITY, f32::MAX],
            0x7f7f_ffff,
        ),
        (vec![f32::INFINITY; 6], 0x7f80_0000),
    ];
    for (values, expected) in cases {
        let input = client.create_from_slice(f32::as_bytes(&values));
        // Each case resets the shared bound. Separate cubes contend on the same address.
        let bound = client.create_from_slice(i32::as_bytes(&[0x7f80_0000]));
        let decoded = client.create_from_slice(f32::as_bytes(&[0.0]));
        for _ in 0..4 {
            contended_min::launch(
                client,
                CubeCount::new_1d(2),
                CubeDim::new_1d(4),
                unsafe { BufferArg::from_raw_parts(input.clone(), values.len()) },
                unsafe { BufferArg::from_raw_parts(bound.clone(), 1) },
            );
        }
        read_bound::launch(
            client,
            CubeCount::new_1d(1),
            CubeDim::new_1d(1),
            unsafe { BufferArg::from_raw_parts(bound.clone(), 1) },
            unsafe { BufferArg::from_raw_parts(decoded.clone(), 1) },
        );
        assert_eq!(
            i32::from_bytes(&client.read_one_unchecked(bound)),
            &[expected]
        );
        assert_eq!(
            i32::from_bytes(&client.read_one_unchecked(decoded)),
            &[expected]
        );
    }
}

#[test]
fn m0_atomic_cpu_round_trip_and_contention() {
    let client = Device::cpu().expect("CPU device construction").client();
    check_encoding(&client);
    check_contention(&client);
}

#[test]
fn m0_atomic_cuda_round_trip_and_contention() {
    let count =
        cuda_backend::try_device_count().expect("CUDA discovery failed, not confirmed absence");
    if count == 0 {
        assert_ne!(
            std::env::var("FAST_UMAP_REQUIRE_CUDA").as_deref(),
            Ok("1"),
            "required CUDA unavailable"
        );
        eprintln!("SKIPPED M0 CUDA: no usable CUDA device");
        return;
    }
    let client = Device::cuda(0)
        .expect("construct discovered CUDA device")
        .client();
    check_encoding(&client);
    check_contention(&client);
}
