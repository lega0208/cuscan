//! Bit-exact ANN baselines, captured on the pre-move HEAD tree and compared after the move.
//!
//! Files live in `FAST_UMAP_ANN_BASELINE_DIR` (relative paths resolve against the workspace
//! root). Every file is raw little-endian words, one file per artifact.
//!
//! - B0 (`capture_{cpu,cuda}`, `compare_{cpu,cuda}`): k=15, 8 iterations, seed 42.
//!   `{label}-ids.bin` (i32) and `{label}-distance-bits.bin` (u32) are compared.
//!   `{label}-embedding-bits.bin` (u32) is written by `capture_*` as a historical record only.
//!   `compare_*` never reads it: the parallel atomic layout is intentionally nondeterministic
//!   (D15), so embedding regressions are judged by the CPU/CUDA quality suites.
//! - Estimator (`capture_estimator_{cpu,cuda}`, `compare_estimator_{cpu,cuda}`): the ANN call with
//!   the parameters `UMAP::fit` derives for n=512, features=8, config.n_neighbors=15: nonself
//!   k=14, iterations=9, seed 42. Files `{label}-estimator-ids.bin` (i32) and
//!   `{label}-estimator-distance-bits.bin` (u32).
//!
//! Capture refuses to overwrite existing files; delete a file deliberately to recapture it.
//! Compare fails on a missing or changed file and prints the capture command.
//! The estimator baseline must be captured from the pristine HEAD worktree
//! (`target/m2-ann-baseline`). Its copy of this file calls `StandardUmapOps`, so the baseline
//! is not produced by relocated code. This live file calls `NeighborOps` (the post-move path).

use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use fast_cubecl_common::backend::NeighborOps;
use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};
use std::{
    fs,
    path::{Path, PathBuf},
};

const BASELINE_DIR_ENV: &str = "FAST_UMAP_ANN_BASELINE_DIR";
const REQUIRE_CUDA_ENV: &str = "FAST_UMAP_REQUIRE_CUDA";
const SEED: u64 = 42;
const BLOBS: usize = 4;
const ROWS: usize = 512;
const FEATURES: usize = 8;
const N_NEIGHBORS: usize = 15;
const EMBEDDING_DIMS: usize = 2;
const FIT_EPOCHS: usize = 50;
/// B0 neighbor width and iteration count, fixed when the original baseline was captured.
const B0_K: usize = 15;
const B0_ITERATIONS: u32 = 8;
/// Estimator parameters for ROWS=512. `fit_neighbor_widths` gives nonself
/// `min(n_neighbors, n - 1) - 1`, and the estimator uses `round(log2(n)).max(5)` iterations.
const ESTIMATOR_K: usize = 14;
const ESTIMATOR_ITERATIONS: u32 = 9;

#[derive(Clone, Copy, Debug)]
enum Target {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Capture,
    Compare,
}

#[derive(Clone, Copy, Debug)]
enum Family {
    B0,
    Estimator,
}

impl Family {
    fn k(self) -> usize {
        match self {
            Family::B0 => B0_K,
            Family::Estimator => ESTIMATOR_K,
        }
    }

    fn iterations(self) -> u32 {
        match self {
            Family::B0 => B0_ITERATIONS,
            Family::Estimator => ESTIMATOR_ITERATIONS,
        }
    }

    /// Infix between the backend label and the artifact name: `cpu-ids.bin` or
    /// `cpu-estimator-ids.bin`.
    fn file_infix(self) -> &'static str {
        match self {
            Family::B0 => "",
            Family::Estimator => "estimator-",
        }
    }

    /// Infix between `capture_`/`compare_` and the backend label.
    fn test_infix(self) -> &'static str {
        match self {
            Family::B0 => "",
            Family::Estimator => "estimator_",
        }
    }
}

/// SplitMix64: fixed, dependency-free generator so the baseline dataset cannot drift
/// with upstream `rand` changes.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 bits of precision.
    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Seed-42 blob dataset: `BLOBS` centers in [-10, 10)^FEATURES, rows assigned round-robin,
/// each coordinate jittered by a uniform offset in [-0.25, 0.25).
fn blob_rows() -> Vec<Vec<f32>> {
    let mut random = SplitMix64(SEED);
    let centers: Vec<Vec<f32>> = (0..BLOBS)
        .map(|_| (0..FEATURES).map(|_| random.unit() * 20.0 - 10.0).collect())
        .collect();
    (0..ROWS)
        .map(|row| {
            let center = &centers[row % BLOBS];
            center
                .iter()
                .map(|&value| value + (random.unit() - 0.5) * 0.5)
                .collect()
        })
        .collect()
}

fn target_device(target: Target) -> Option<(Device, BackendChoice)> {
    match target {
        Target::Cpu => Some((Device::cpu(), BackendChoice::Cpu)),
        Target::Cuda => {
            let Some(device) = fast_umap::cuda_backend::device(0) else {
                assert!(
                    std::env::var_os(REQUIRE_CUDA_ENV).is_none(),
                    "{REQUIRE_CUDA_ENV} is set, but no CUDA device is available; \
                     the CUDA baseline cannot be captured or compared"
                );
                eprintln!("SKIPPED: no CUDA device");
                return None;
            };
            Some((device, BackendChoice::Cuda { device_index: 0 }))
        }
    }
}

/// The only ANN call site in this file. Uses `NeighborOps` with `squared = false`, the
/// estimator's distance mode. Returns `(ids, distance bits)` for the `[ROWS, k]` table.
fn ann_neighbors(
    device: &Device,
    values: &[f32],
    k: usize,
    iterations: u32,
) -> (Vec<i32>, Vec<u32>) {
    let data = Tensor::<2>::from_data(TensorData::new(values.to_vec(), [ROWS, FEATURES]), device);
    let (ids, distances) = <Dispatch as NeighborOps>::approximate_neighbors(
        data.into_dispatch(),
        k as u32,
        SEED,
        iterations,
        false,
    );
    let ids = Tensor::<2, Int>::from_dispatch(ids)
        .to_data()
        .try_to_vec::<i32>()
        .unwrap();
    let distance_bits = Tensor::<2>::from_dispatch(distances)
        .to_data()
        .try_to_vec::<f32>()
        .unwrap()
        .into_iter()
        .map(f32::to_bits)
        .collect();
    (ids, distance_bits)
}

/// Historical B0 embedding bits, written by capture only. Never compared (D15).
fn fitted_embedding_bits(backend: BackendChoice, rows: &[Vec<f32>]) -> Vec<u32> {
    let config = UMAPConfig {
        n_components: EMBEDDING_DIMS,
        n_neighbors: N_NEIGHBORS,
        n_epochs: Some(FIT_EPOCHS),
        init: Init::Random,
        random_state: Some(SEED),
        ..Default::default()
    };
    let fitted = UMAP::new(config, backend, NeighborSearch::Approximate)
        .fit(rows)
        .unwrap();
    fitted
        .embedding()
        .iter()
        .flat_map(|row| row.iter().map(|value| value.to_bits()))
        .collect()
}

fn to_le_bytes_i32(words: &[i32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn to_le_bytes_u32(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

fn baseline_dir() -> PathBuf {
    let configured = std::env::var_os(BASELINE_DIR_ENV)
        .unwrap_or_else(|| panic!("{BASELINE_DIR_ENV} must name the ANN baseline directory"));
    let path = PathBuf::from(configured);
    if path.is_absolute() {
        path
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    }
}

fn target_label(target: Target) -> &'static str {
    match target {
        Target::Cpu => "cpu",
        Target::Cuda => "cuda",
    }
}

fn artifact_path(dir: &Path, label: &str, infix: &str, name: &str) -> PathBuf {
    dir.join(format!("{label}-{infix}{name}"))
}

/// Byte-exact comparison against the captured baseline. The first differing word is reported.
fn compare_artifact(
    dir: &Path,
    label: &str,
    infix: &str,
    name: &str,
    actual: &[u8],
    capture_cmd: &str,
) {
    let path = artifact_path(dir, label, infix, name);
    let expected = fs::read(&path).unwrap_or_else(|error| {
        panic!(
            "missing baseline file {}: {error}\n\
             capture it first with:\n  {capture_cmd}",
            path.display()
        )
    });
    if expected == actual {
        return;
    }
    if expected.len() != actual.len() {
        panic!(
            "baseline {} has {} bytes, current run produced {} bytes; \
             row count, k, or neighbor table width changed",
            path.display(),
            expected.len(),
            actual.len()
        );
    }
    let index = expected
        .as_chunks::<4>()
        .0
        .iter()
        .zip(actual.as_chunks::<4>().0.iter())
        .position(|(left, right)| left != right)
        .expect("byte mismatch must contain a differing 4-byte word");
    let expected_word = u32::from_le_bytes(expected[index * 4..index * 4 + 4].try_into().unwrap());
    let actual_word = u32::from_le_bytes(actual[index * 4..index * 4 + 4].try_into().unwrap());
    panic!(
        "{label} {name} changed against {}: word {index} (byte offset {}) expected 0x{expected_word:08x} \
         got 0x{actual_word:08x}; ANN output is no longer bit-identical to the baseline",
        path.display(),
        index * 4
    );
}

fn capture_command(target: Target, family: Family) -> String {
    let label = target_label(target);
    let infix = family.test_infix();
    let env = match target {
        Target::Cpu => "",
        Target::Cuda => "FAST_UMAP_REQUIRE_CUDA=1 ",
    };
    format!(
        "{env}{BASELINE_DIR_ENV}=<dir> cargo test -p fast-umap --test test_ann_bit_baseline capture_{infix}{label} -- --ignored --exact --nocapture"
    )
}

fn run(target: Target, family: Family, mode: Mode) {
    assert_eq!(
        ESTIMATOR_K,
        N_NEIGHBORS.min(ROWS - 1) - 1,
        "estimator nonself width drifted from fit_neighbor_widths"
    );
    assert_eq!(
        ESTIMATOR_ITERATIONS,
        (ROWS as f64).log2().round().max(5.0) as u32,
        "estimator iteration count drifted from the estimator formula"
    );
    let Some((device, backend)) = target_device(target) else {
        return;
    };
    let label = target_label(target);
    let rows = blob_rows();
    let values: Vec<f32> = rows.iter().flatten().copied().collect();

    let (ids, distance_bits) = ann_neighbors(&device, &values, family.k(), family.iterations());
    let mut artifacts = vec![
        ("ids.bin", to_le_bytes_i32(&ids)),
        ("distance-bits.bin", to_le_bytes_u32(&distance_bits)),
    ];
    // Embedding bits are a historical B0 capture only. Compare never reads them.
    if matches!((family, mode), (Family::B0, Mode::Capture)) {
        let embedding_bits = fitted_embedding_bits(backend, &rows);
        artifacts.push(("embedding-bits.bin", to_le_bytes_u32(&embedding_bits)));
    }

    let infix = family.file_infix();
    let dir = baseline_dir();
    let capture_cmd = capture_command(target, family);
    match mode {
        Mode::Capture => {
            fs::create_dir_all(&dir).unwrap_or_else(|error| {
                panic!("cannot create baseline dir {}: {error}", dir.display())
            });
            for (name, _) in &artifacts {
                let path = artifact_path(&dir, label, infix, name);
                assert!(
                    !path.exists(),
                    "baseline {} already exists; delete it to recapture",
                    path.display()
                );
            }
            for (name, bytes) in &artifacts {
                let path = artifact_path(&dir, label, infix, name);
                fs::write(&path, bytes).unwrap_or_else(|error| {
                    panic!("cannot write baseline {}: {error}", path.display())
                });
                eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
            }
        }
        Mode::Compare => {
            for (name, bytes) in &artifacts {
                compare_artifact(&dir, label, infix, name, bytes, &capture_cmd);
            }
        }
    }
}

#[test]
#[ignore = "captures the B0 CPU ANN baseline (k=15) and historical embedding bits; run with --ignored --exact"]
fn capture_cpu() {
    run(Target::Cpu, Family::B0, Mode::Capture);
}

#[test]
#[ignore = "captures the B0 CUDA ANN baseline (k=15) and historical embedding bits; run with --ignored --exact"]
fn capture_cuda() {
    run(Target::Cuda, Family::B0, Mode::Capture);
}

#[test]
#[ignore = "compares CPU ANN ids and distance bits against the B0 baseline; run with --ignored --exact"]
fn compare_cpu() {
    run(Target::Cpu, Family::B0, Mode::Compare);
}

#[test]
#[ignore = "compares CUDA ANN ids and distance bits against the B0 baseline; run with --ignored --exact"]
fn compare_cuda() {
    run(Target::Cuda, Family::B0, Mode::Compare);
}

#[test]
#[ignore = "captures the estimator-parameter CPU ANN baseline (k=14, 9 iterations); run from the pristine HEAD worktree"]
fn capture_estimator_cpu() {
    run(Target::Cpu, Family::Estimator, Mode::Capture);
}

#[test]
#[ignore = "captures the estimator-parameter CUDA ANN baseline (k=14, 9 iterations); run from the pristine HEAD worktree"]
fn capture_estimator_cuda() {
    run(Target::Cuda, Family::Estimator, Mode::Capture);
}

#[test]
#[ignore = "compares CPU estimator-parameter ANN ids and distance bits against the pristine baseline"]
fn compare_estimator_cpu() {
    run(Target::Cpu, Family::Estimator, Mode::Compare);
}

#[test]
#[ignore = "compares CUDA estimator-parameter ANN ids and distance bits against the pristine baseline"]
fn compare_estimator_cuda() {
    run(Target::Cuda, Family::Estimator, Mode::Compare);
}
