use hdbscan::{BackendChoice, HDBSCAN, HdbscanConfig, HdbscanSearch};
use std::{error::Error, fs};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 8 {
        return Err("usage: capture_semantics RAW ROWS FEATURES MIN_SAMPLES BACKEND SEARCH MIN_CLUSTER_SIZE".into());
    }
    let rows: usize = args[2].parse()?;
    let features: usize = args[3].parse()?;
    let bytes = fs::read(&args[1])?;
    if rows == 0
        || features == 0
        || rows.checked_mul(features).and_then(|n| n.checked_mul(4)) != Some(bytes.len())
    {
        return Err("invalid matrix dimensions".into());
    }
    let flat: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let matrix: Vec<Vec<f32>> = flat.chunks_exact(features).map(<[f32]>::to_vec).collect();
    let backend = match args[5].as_str() {
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index: 0 },
        _ => return Err("invalid backend".into()),
    };
    let search = match args[6].as_str() {
        "brute-force" => HdbscanSearch::BruteForce,
        "kd-tree" => HdbscanSearch::KdTree,
        _ => return Err("invalid search".into()),
    };
    let config = HdbscanConfig {
        min_cluster_size: args[7].parse()?,
        min_samples: Some(args[4].parse()?),
        ..Default::default()
    };
    let fitted = HDBSCAN::new(config, backend)
        .with_search(search)
        .fit(&matrix)?;
    println!("{fitted:#?}");
    println!(
        "core_f32_bits={:?}",
        fitted
            .core_distances()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    println!(
        "probability_f32_bits={:?}",
        fitted
            .probabilities()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    println!(
        "linkage_f64_bits={:?}",
        fitted
            .linkage()
            .iter()
            .map(|x| x.distance.to_bits())
            .collect::<Vec<_>>()
    );
    println!(
        "mst_f64_bits={:?}",
        fitted
            .minimum_spanning_tree()
            .iter()
            .map(|x| x.weight.to_bits())
            .collect::<Vec<_>>()
    );
    println!(
        "condensed_lambda_f32_bits={:?}",
        fitted
            .condensed_tree()
            .lambda
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    println!(
        "condensed_child_size_f32_bits={:?}",
        fitted
            .condensed_tree()
            .child_size
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );
    Ok(())
}
