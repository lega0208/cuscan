//! Phase 0 host forest probe. Includes the same source without exporting probe APIs.
#[cfg(feature = "test-support")]
#[allow(dead_code)]
#[path = "../src/kernels/ann/ann_forest.rs"]
mod ann_forest;

#[cfg(feature = "test-support")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use std::time::Instant;
    let args: Vec<_> = std::env::args().collect();
    if !(4..=6).contains(&args.len()) {
        return Err(
            "usage: ann_forest_probe ROWS FEATURES THREADS [REPEATS] [JOIN_THRESHOLD]".into(),
        );
    }
    let rows: usize = args[1].parse()?;
    let features: usize = args[2].parse()?;
    let threads: usize = args[3].parse()?;
    let repeats: usize = args.get(4).map_or(Ok(3), |s| s.parse())?;
    let threshold: usize = args.get(5).map_or(Ok(8192), |s| s.parse())?;
    if rows == 0 || features == 0 || threads == 0 || repeats == 0 || threshold == 0 {
        return Err("all arguments must be positive".into());
    }
    let len = rows.checked_mul(features).ok_or("input size overflow")?;
    let mut state = 42u64;
    let data: Vec<f32> = (0..len)
        .map(|_| {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^= z >> 31;
            (z >> 40) as f32 / (1u32 << 24) as f32
        })
        .collect();
    let trees = ann_forest::tree_count(rows).map_err(|e| format!("tree count: {e:?}"))?;
    // k=14 uses a four-edge heap margin, as in the ANN acceptance campaign.
    let leaf_size = ann_forest::forest_leaf_limit(rows, 18.min(rows - 1));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    pool.install(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for repeat in 0..repeats {
            let mut expected = None;
            // Rotate order to reduce systematic warm-cache/order bias.
            for offset in 0..3 {
                let variant = (repeat + offset) % 3;
                let started = Instant::now();
                let result = match variant {
                    0 => ann_forest::build_forest(&data, rows, features, 42, trees, leaf_size),
                    1 => ann_forest::probe_forest(&data, rows, features, 42, trees, leaf_size, None),
                    _ => ann_forest::probe_forest(&data, rows, features, 42, trees, leaf_size, Some(threshold)),
                }.map_err(|e| format!("forest: {e:?}"))?;
                let elapsed = started.elapsed();
                if let Some(ref baseline) = expected {
                    if baseline != &result {
                        return Err(format!("forest mismatch repeat={repeat} variant={variant}").into());
                    }
                } else {
                    expected = Some(result);
                }
                let name = ["current", "lanes", "lanes-joins"][variant];
                println!("variant={name} rows={rows} features={features} threads={threads} trees={trees} leaf_size={leaf_size} join_threshold={threshold} repeat={repeat} elapsed_ms={:.3}", elapsed.as_secs_f64() * 1000.0);
            }
        }
        Ok(())
    })
}

#[cfg(not(feature = "test-support"))]
fn main() {
    eprintln!("ann_forest_probe requires --features test-support");
    std::process::exit(1);
}
