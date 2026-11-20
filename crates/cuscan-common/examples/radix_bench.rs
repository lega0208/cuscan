//! Explicit-size benchmark for the stable radix argsort. Never run automatically.
//!
//! usage: radix_bench --backend cpu|cuda --rows N [--digit-bits B] [--max-cubes C]
//!        [--cube-dim D] [--cpu-min-chunk R] [--iterations K] [--verify]
use cubecl::prelude::*;
use cuscan_common::radix::*;
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let mut backend = None;
    let mut rows = None;
    let mut overrides: Vec<(String, u32)> = Vec::new();
    let mut iterations = 5usize;
    let mut verify = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => backend = Some(args.next().ok_or("missing backend")?),
            "--rows" => rows = Some(args.next().ok_or("missing rows")?.parse::<usize>()?),
            "--iterations" => iterations = args.next().ok_or("missing iterations")?.parse()?,
            "--verify" => verify = true,
            "--digit-bits" | "--max-cubes" | "--cube-dim" | "--cpu-min-chunk" => {
                overrides.push((arg.clone(), args.next().ok_or("missing value")?.parse()?))
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        }
    }
    let backend = backend.ok_or("--backend is required")?;
    let rows = rows.ok_or("--rows is required (no default size)")?;
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let device = match backend.as_str() {
        "cpu" => cubecl::Device::cpu()?,
        "cuda" => cubecl::Device::cuda(0)?,
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let client = device.client();
    let mut config = RadixSortConfig::conservative(RadixDeviceLimits::from_client(&client).path);
    for (name, value) in overrides {
        match name.as_str() {
            "--digit-bits" => config.digit_bits = value,
            "--max-cubes" => config.max_cubes = value,
            "--cube-dim" => config.cube_dim = value,
            _ => config.cpu_min_chunk_rows = value,
        }
    }
    let values: Vec<f32> = (0..rows)
        .map(|i| {
            let mut x = (i as u32)
                .wrapping_mul(747_796_405)
                .wrapping_add(2_891_336_453);
            x ^= x >> 16;
            f32::from_bits(x & 0xBFFF_FFFF) // exponent MSB cleared: always finite
        })
        .collect();
    let input = client.create_from_slice(f32::as_bytes(if rows == 0 { &[0.0] } else { &values }));
    let mut workspace = RadixSortWorkspace::new(&client, rows, config)?;
    let plan = workspace.plan().clone();
    let mut times = Vec::with_capacity(iterations);
    let mut stats = None;
    for _ in 0..iterations + 1 {
        cubecl::future::block_on(client.sync())?;
        let start = Instant::now();
        let output = workspace.sort(&input, rows)?;
        cubecl::future::block_on(client.sync())?;
        times.push(start.elapsed().as_secs_f64() * 1e3);
        stats = Some(output.stats);
    }
    let warmup_ms = times.remove(0);
    times.sort_by(f64::total_cmp);
    let mut correct = None;
    if verify {
        let output = workspace.sort(&input, rows)?;
        let ids = u32::from_bytes(&client.read_one(output.ids)?)[..rows].to_vec();
        let mut expected: Vec<u32> = (0..rows as u32).collect();
        expected.sort_by(|&a, &b| values[a as usize].partial_cmp(&values[b as usize]).unwrap());
        correct = Some(ids == expected);
    }
    let stats = stats.expect("at least one iteration");
    println!(
        "{{\"backend\":\"{backend}\",\"rows\":{rows},\"digit_bits\":{},\"cube_dim\":{},\"cubes\":{},\"chunk_len\":{},\"passes_executed\":{},\"total_aligned_bytes\":{},\"max_aligned_allocation_bytes\":{},\"warmup_ms_including_jit\":{warmup_ms:.6},\"median_ms\":{:.6},\"min_ms\":{:.6},\"iterations\":{iterations},\"correct\":{}}}",
        plan.digit_bits,
        plan.cube_dim,
        plan.cubes,
        plan.chunk_len,
        stats.passes_executed,
        plan.total_aligned_bytes,
        plan.max_aligned_allocation_bytes,
        times[times.len() / 2],
        times[0],
        correct.map_or("null".to_string(), |c| c.to_string()),
    );
    if correct == Some(false) {
        return Err("radix argsort differs from stable host sort".into());
    }
    Ok(())
}
