//! Phase 0 bitwise tile-invariance probe; no baseline is recaptured.
use burn::tensor::{DType, Shape, Tensor, TensorData};
use burn_cubecl::tensor::CubeTensor;
use cuscan_common::approximate_neighbors_with_tile_slots;
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: ann_tile_probe cpu|cuda ROWS [FEATURES] [REPEATS] [SLOTS|budget]".into(),
        );
    }
    let device = match args[1].as_str() {
        "cpu" => cubecl::Device::cpu()?,
        "cuda" => cubecl::Device::cuda(0)?,
        _ => return Err("backend must be cpu or cuda".into()),
    };
    let rows: usize = args[2].parse()?;
    let features: usize = args.get(3).map_or(Ok(32), |s| s.parse())?;
    let repeats: usize = args.get(4).map_or(Ok(5), |s| s.parse())?;
    let budgeted = args.get(5).is_some_and(|s| s == "budget");
    let fixed_slots = args
        .get(5)
        .filter(|_| !budgeted)
        .map(|s| s.parse::<usize>())
        .transpose()?;
    if rows <= 14 || features == 0 || repeats == 0 {
        return Err("requires rows > 14, features > 0, repeats > 0".into());
    }
    let len = rows.checked_mul(features).ok_or("input size overflow")?;
    println!(
        "estimate={:?}",
        cuscan_common::neighbors::ann_host_estimate(rows, features, 14)?
    );
    let mut state = 42u64;
    let values: Vec<f32> = (0..len)
        .map(|i| {
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ((z >> 40) as f32 / 16777216.0) + ((i / features) % 4) as f32 * 4.0
        })
        .collect();
    let iterations = (rows as f64).log2().round().max(5.0) as u32;
    if budgeted {
        let choice = match args[1].as_str() {
            "cpu" => cuscan_common::BackendChoice::Cpu,
            "cuda" => cuscan_common::BackendChoice::Cuda { device_index: 0 },
            _ => unreachable!(),
        };
        let (device, _) = cuscan_common::selected_device(choice)?;
        let input = Tensor::<2>::from_data(TensorData::new(values, [rows, features]), &device);

        let mut baseline = None;
        for repeat in 0..=repeats {
            let started = Instant::now();
            let budget = cuscan_common::resolve_memory_budget(&device, None)?;
            let plan = cuscan_common::ann_preflight(&device, rows, features, 14, &budget)?;
            let admitted = Instant::now();
            let (ids, distances) = cuscan_common::neighbors::approximate_neighbors(
                input.clone(),
                &plan,
                42,
                iterations,
                false,
            )?;
            let executed = Instant::now();
            let (ids, _, _) = ids.into_data().into_parts();
            let (distances, _, _) = distances.into_data().into_parts();
            let read_back = Instant::now();
            if std::env::var_os("CUSCAN_ANN_PROFILE").is_some() {
                println!(
                    "ANN_PROBE_PROFILE admission_ms={:.3} execution_ms={:.3} readback_ms={:.3}",
                    admitted.duration_since(started).as_secs_f64() * 1000.0,
                    executed.duration_since(admitted).as_secs_f64() * 1000.0,
                    read_back.duration_since(executed).as_secs_f64() * 1000.0,
                );
            }
            let result = (ids.to_vec(), distances.to_vec());
            if let Some(expected) = &baseline {
                if expected != &result {
                    return Err(format!("tile invariance FAILED: backend={} rows={rows} features={features} repeat={repeat} slots={}", args[1], plan.tile_slots).into());
                }
            } else {
                baseline = Some(result);
            }
            println!(
                "backend={} rows={rows} features={features} repeat={repeat} first_pass={} tile_slots={} budget={budget:?} memory={:?} elapsed_ms={:.3} bits_equal=true",
                args[1],
                repeat == 0,
                plan.tile_slots,
                plan.memory,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        return Ok(());
    }
    let (bytes, _, _) = TensorData::new(values, [rows, features]).into_parts();
    let client = device.client();
    let input = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([rows, features]),
        client.create(bytes),
        DType::F32,
    );
    let mut baseline = None;
    for repeat in 0..=repeats {
        // Alternate order to avoid confounding tile size with warm-up/order.
        let tiles = match fixed_slots {
            Some(slots) => vec![slots],
            None if repeat % 2 == 0 => vec![1 << 20, 1 << 22, 1 << 24, 1 << 26],
            None => vec![1 << 26, 1 << 24, 1 << 22, 1 << 20],
        };
        for slots in tiles {
            let started = Instant::now();
            let (ids, distances) = approximate_neighbors_with_tile_slots(
                input.clone(),
                14,
                42,
                iterations,
                false,
                slots,
            )?;
            let executed = Instant::now();
            let ids = client.read_one(ids.handle)?;
            let distances = client.read_one(distances.handle)?;
            let read_back = Instant::now();
            if std::env::var_os("CUSCAN_ANN_PROFILE").is_some() {
                println!(
                    "ANN_PROBE_PROFILE execution_ms={:.3} readback_ms={:.3}",
                    executed.duration_since(started).as_secs_f64() * 1000.0,
                    read_back.duration_since(executed).as_secs_f64() * 1000.0,
                );
            }
            let result = (ids.to_vec(), distances.to_vec());
            if let Some(expected) = &baseline {
                if expected != &result {
                    return Err(format!("tile invariance FAILED: backend={} rows={rows} features={features} repeat={repeat} slots={slots}", args[1]).into());
                }
            } else {
                baseline = Some(result);
            }
            println!(
                "backend={} rows={rows} features={features} repeat={repeat} first_pass={} requested_slots={slots} elapsed_ms={:.3} bits_equal=true",
                args[1],
                repeat == 0,
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
    }
    Ok(())
}
