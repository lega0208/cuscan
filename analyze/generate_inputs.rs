use std::{error::Error, fs::File, io::Write, path::Path};

// Exact f32 arithmetic copied from crates/fast-hdbscan/benches/support/mod.rs.
fn synthetic(rows: usize, dimensions: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..dimensions)
                .map(|feature| {
                    let seed = (row as u64 + 1) * 0x9e37_79b9 + (feature as u64 + 3) * 0x85eb_ca6b;
                    let mixed = seed ^ (seed >> 17) ^ (seed << 13);
                    let noise = (mixed % 10_007) as f32 / 10_007.0 - 0.5;
                    let cluster = (row % 4) as f32 * 4.0;
                    cluster + feature as f32 * 0.03125 + noise * 0.05
                })
                .collect()
        })
        .collect()
}

fn save(path: &Path, rows: &[Vec<f32>]) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    for row in rows {
        for value in row {
            file.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let directory = std::env::args().nth(1).ok_or("output directory required")?;
    let root = Path::new(&directory);
    std::fs::create_dir_all(root)?;
    for (n, d) in [(64, 50), (2048, 30), (4096, 10), (257, 2)] {
        save(
            &root.join(format!("synthetic_n{n}_d{d}.raw")),
            &synthetic(n, d),
        )?;
    }
    save(
        &root.join("duplicates_n9_d2.raw"),
        &vec![vec![2.0, -3.0]; 9],
    )?;
    save(
        &root.join("overflow_n3_d2.raw"),
        &[vec![1e20, 0.0], vec![-1e20, 0.0], vec![0.0, 1e20]],
    )?;
    println!("Generated six exact f32 little-endian inputs");
    Ok(())
}
