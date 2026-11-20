//! Isolated stable LSD radix argsort research; deliberately not a production operation.
//! Each lane handles one sequential chunk. The serial histogram scan and 2-bit
//! digits favor a simple correctness baseline over GPU throughput.
use burn::{
    cubecl::prelude::*,
    tensor::{DType, Shape, TensorData},
};
use burn_cubecl::tensor::CubeTensor;
use cubecl::future::block_on;
use std::{error::Error, time::Instant};

const CHUNK: usize = 256;
const RADIX: usize = 4;

// Histograms are digit-major, so an exclusive scan also supplies digit bases.
#[cube(launch)]
fn reduce(keys: &Tensor<u32>, counts: &mut Tensor<u32>, shift: u32) {
    let chunk = ABSOLUTE_POS_X as usize;
    let chunks = counts.shape(0) / 4;
    if chunk < chunks {
        let start = chunk * 256;
        let end = (start + 256).min(keys.shape(0));
        let mut a = 0u32;
        let mut b = 0u32;
        let mut c = 0u32;
        let mut d = 0u32;
        for i in start..end {
            let digit = (keys[i] >> shift) & 3u32;
            if digit == 0 {
                a += 1;
            } else if digit == 1 {
                b += 1;
            } else if digit == 2 {
                c += 1;
            } else {
                d += 1;
            }
        }
        counts[chunk] = a;
        counts[chunks + chunk] = b;
        counts[2 * chunks + chunk] = c;
        counts[3 * chunks + chunk] = d;
    }
}

#[cube(launch)]
fn scan(counts: &mut Tensor<u32>) {
    if ABSOLUTE_POS_X == 0 {
        let mut sum = 0u32;
        for i in 0..counts.shape(0) {
            let count = counts[i];
            counts[i] = sum;
            sum += count;
        }
    }
}

#[cube(launch)]
fn scatter(
    keys: &Tensor<u32>,
    ids: &Tensor<u32>,
    offsets: &Tensor<u32>,
    out_keys: &mut Tensor<u32>,
    out_ids: &mut Tensor<u32>,
    shift: u32,
) {
    let chunk = ABSOLUTE_POS_X as usize;
    let chunks = offsets.shape(0) / 4;
    if chunk < chunks {
        let mut a = offsets[chunk];
        let mut b = offsets[chunks + chunk];
        let mut c = offsets[2 * chunks + chunk];
        let mut d = offsets[3 * chunks + chunk];
        let start = chunk * 256;
        let end = (start + 256).min(keys.shape(0));
        for i in start..end {
            let key = keys[i];
            let digit = (key >> shift) & 3u32;
            let pos = if digit == 0 {
                let pos = a;
                a += 1;
                pos
            } else if digit == 1 {
                let pos = b;
                b += 1;
                pos
            } else if digit == 2 {
                let pos = c;
                c += 1;
                pos
            } else {
                let pos = d;
                d += 1;
                pos
            };
            out_keys[pos as usize] = key;
            out_ids[pos as usize] = ids[i];
        }
    }
}

// Numeric ordering treats signed zeros as equal, preserving their input order.
fn key(value: f32) -> u32 {
    let bits = if value == 0.0 { 0 } else { value.to_bits() };
    if bits & 0x8000_0000 != 0 {
        !bits
    } else {
        bits ^ 0x8000_0000
    }
}

fn value(i: usize, rows: usize, case: &str) -> f32 {
    match case {
        "ordered" => i as f32,
        "reversed" => (rows - i) as f32,
        "equal" => -0.0,
        _ => {
            // Finite bit patterns, repeated values and both zero signs.
            match i % 13 {
                0 => -0.0,
                1 => 0.0,
                2 => -1.0,
                3 => 1.0,
                _ => {
                    let mut x = (i as u32)
                        .wrapping_mul(747_796_405)
                        .wrapping_add(2_891_336_453);
                    x ^= x >> 16;
                    if x & 0x7f80_0000 == 0x7f80_0000 {
                        x ^= 0x0080_0000;
                    }
                    f32::from_bits(x)
                }
            }
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut backend = String::from("cpu");
    let mut rows = 1003usize;
    let mut case = String::from("mixed");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => backend = args.next().ok_or("missing backend")?,
            "--rows" => rows = args.next().ok_or("missing rows")?.parse()?,
            "--case" => case = args.next().ok_or("missing case")?,
            "--help" => {
                println!(
                    "usage: umap06_radix_spike [--backend cpu|cuda] [--rows N] [--case mixed|ordered|reversed|equal]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    if rows == 0 || rows > u32::MAX as usize {
        return Err("rows must be in 1..=u32::MAX".into());
    }
    if !["mixed", "ordered", "reversed", "equal"].contains(&case.as_str()) {
        return Err("unknown case".into());
    }
    let device = match backend.as_str() {
        "cpu" => cubecl::Device::cpu()?,
        "cuda" => cubecl::Device::cuda(0)?,
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let client = device.client();
    let chunks = rows.div_ceil(CHUNK);
    let count_len = chunks.checked_mul(RADIX).ok_or("count length overflow")?;
    let data_bytes = rows.checked_mul(4).ok_or("data allocation overflow")?;
    let count_bytes = count_len
        .checked_mul(4)
        .ok_or("histogram allocation overflow")?;
    let memory = &client.properties().memory;
    let alignment = usize::try_from(memory.alignment)?;
    let max_page = usize::try_from(memory.max_page_size)?;
    if alignment == 0 {
        return Err("invalid device alignment".into());
    }
    let aligned = |n: usize| -> Result<usize, Box<dyn Error>> {
        Ok(n.checked_add(alignment - 1).ok_or("alignment overflow")? / alignment * alignment)
    };
    let data_aligned = aligned(data_bytes)?;
    let counts_aligned = aligned(count_bytes)?;
    if data_aligned > max_page || counts_aligned > max_page {
        return Err("allocation exceeds device max_page_size".into());
    }
    let total_aligned = data_aligned
        .checked_mul(4)
        .and_then(|v| v.checked_add(counts_aligned))
        .ok_or("total allocation overflow")?;
    let hw = &client.properties().hardware;
    let dim = 32u32.min(hw.max_units_per_cube).min(hw.max_cube_dim.0);
    if dim == 0 {
        return Err("device has no legal cube dimension".into());
    }
    let cubes = u32::try_from(chunks.div_ceil(dim as usize))?;
    if cubes > hw.max_cube_count.0 {
        return Err("histogram launch exceeds device cube count".into());
    }
    let values: Vec<f32> = (0..rows).map(|i| value(i, rows, &case)).collect();
    let keys: Vec<u32> = values.iter().copied().map(key).collect();
    let varying = keys.iter().fold(0u32, |mask, &k| mask | (k ^ keys[0]));
    let ids: Vec<u32> = (0..rows as u32).collect();
    let host_start = Instant::now();
    let mut expected = ids.clone();
    expected.sort_by(|&a, &b| values[a as usize].partial_cmp(&values[b as usize]).unwrap());
    let host_ms = host_start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let tensor = |len: usize, handle| {
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            Shape::new([len]),
            handle,
            DType::U32,
        )
    };
    let (key_bytes, _, _) = TensorData::new(keys, [rows]).into_parts();
    let (id_bytes, _, _) = TensorData::new(ids, [rows]).into_parts();
    let mut src_keys = tensor(rows, client.create(key_bytes));
    let mut src_ids = tensor(rows, client.create(id_bytes));
    let mut dst_keys = tensor(rows, client.empty(data_bytes));
    let mut dst_ids = tensor(rows, client.empty(data_bytes));
    let counts = tensor(count_len, client.empty(count_bytes));
    block_on(client.sync())?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let mut passes = 0;
    for shift in (0..32).step_by(2) {
        if (varying >> shift) & 3 == 0 {
            continue;
        }
        reduce::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            src_keys.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            shift,
        );
        scan::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
        );
        scatter::launch(
            &client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(dim),
            src_keys.clone().into_tensor_arg(),
            src_ids.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            dst_keys.clone().into_tensor_arg(),
            dst_ids.clone().into_tensor_arg(),
            shift,
        );
        std::mem::swap(&mut src_keys, &mut dst_keys);
        std::mem::swap(&mut src_ids, &mut dst_ids);
        passes += 1;
    }
    block_on(client.sync())?;
    let device_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let bytes = client.read_one(src_ids.handle.clone())?;
    let actual: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| u32::from_ne_bytes(*x))
        .collect();
    let readback_ms = start.elapsed().as_secs_f64() * 1000.0;
    let correct = actual == expected;
    println!(
        "{{\"backend\":\"{backend}\",\"case\":\"{case}\",\"rows\":{rows},\"passes\":{passes},\"correct\":{correct},\"data_allocation_bytes\":{data_bytes},\"histogram_allocation_bytes\":{count_bytes},\"planned_aligned_device_bytes\":{total_aligned},\"host_sort_ms\":{host_ms:.6},\"setup_ms\":{setup_ms:.6},\"device_ms_including_jit\":{device_ms:.6},\"readback_ms\":{readback_ms:.6}}}"
    );
    if !correct {
        return Err("radix argsort differs from stable host sort".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run()
}
