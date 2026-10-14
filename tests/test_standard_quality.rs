use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use fast_umap::{
    backend::StandardUmapOps,
    standard_umap::{
        BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig, UmapError, approximate_query_budget,
    },
};
use serde_json::Value;
use std::{
    collections::HashSet,
    fs::File,
    io::Read,
    io::Write,
    process::{Command, Stdio},
};
use zip::ZipArchive;

const REFERENCE: &str = include_str!("fixtures/standard_umap_reference.json");
const QUALITY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/standard_umap_quality.npz"
);
const TRUST_GAP_LIMIT: f64 = 0.02;
const RECALL_LIMIT: f64 = 0.90;

#[cfg(feature = "cuda")]
fn frozen_source_digest() -> String {
    let commit = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    for args in [
        vec!["diff", "--quiet", commit, "--", "src"],
        vec!["diff", "--cached", "--quiet", commit, "--", "src"],
    ] {
        let status = Command::new("git")
            .args(&args)
            .status()
            .expect("git must be available to verify frozen product source");
        assert!(
            status.success(),
            "working product src differs from frozen source commit"
        );
    }
    source_tree_digest(commit)
}

#[cfg(feature = "cuda")]
fn source_tree_digest(commit: &str) -> String {
    let output = Command::new("git")
        .args(["rev-parse", &format!("{commit}:src")])
        .output()
        .expect("git must be available to identify product source");
    assert!(
        output.status.success(),
        "could not resolve {commit}:src tree digest"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[cfg(feature = "cuda")]
fn git_blob_digest(bytes: &[u8]) -> String {
    let mut child = Command::new("git")
        .args(["hash-object", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(bytes).unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "could not digest deterministic error inputs"
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    values: Vec<f32>,
}

impl Matrix {
    fn rows_as_vec(&self) -> Vec<Vec<f32>> {
        self.values
            .chunks_exact(self.cols)
            .map(<[f32]>::to_vec)
            .collect()
    }
}

fn npy_matrix(bytes: &[u8]) -> Matrix {
    assert!(
        bytes.starts_with(b"\x93NUMPY"),
        "fixture member is not an NPY array"
    );
    let version = (bytes[6], bytes[7]);
    let (header_len, start) = match version {
        (1, 0) => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        (2 | 3, 0) => (
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize,
            12,
        ),
        _ => panic!("unsupported NPY version {version:?}"),
    };
    let header = std::str::from_utf8(&bytes[start..start + header_len]).unwrap();
    assert!(
        header.contains("'fortran_order': False"),
        "expected C-order fixture array"
    );
    let descriptor = header
        .split("'descr': '")
        .nth(1)
        .unwrap()
        .split('\'')
        .next()
        .unwrap();
    let shape = header
        .split("'shape': (")
        .nth(1)
        .unwrap()
        .split(')')
        .next()
        .unwrap();
    let dimensions: Vec<usize> = shape
        .split(',')
        .filter_map(|part| part.trim().parse().ok())
        .collect();
    assert_eq!(dimensions.len(), 2, "expected a matrix in fixture");
    let rows = dimensions[0];
    let cols = dimensions[1];
    let payload = &bytes[start + header_len..];
    let values: Vec<f32> = match descriptor {
        "<f8" | "=f8" | "|f8" => payload
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk) as f32)
            .collect(),
        "<f4" | "=f4" | "|f4" => payload
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect(),
        other => panic!("unexpected NPY dtype {other}"),
    };
    assert_eq!(values.len(), rows * cols, "fixture array payload length");
    Matrix { rows, cols, values }
}

fn matrix_from_npz(archive: &mut ZipArchive<File>, name: &str) -> Matrix {
    let mut member = archive
        .by_name(&format!("{name}.npy"))
        .unwrap_or_else(|error| panic!("missing {name}: {error}"));
    let mut bytes = Vec::new();
    member.read_to_end(&mut bytes).unwrap();
    npy_matrix(&bytes)
}

fn matrix_from_npz_path(path: &std::path::Path, name: &str) -> Matrix {
    let mut archive = ZipArchive::new(File::open(path).unwrap())
        .unwrap_or_else(|error| panic!("could not open {}: {error}", path.display()));
    matrix_from_npz(&mut archive, name)
}

fn fixture() -> (Value, Vec<(String, Matrix, Matrix)>) {
    let reference: Value = serde_json::from_str(REFERENCE).unwrap();
    let mut archive = ZipArchive::new(File::open(QUALITY).unwrap()).unwrap();
    let cases = reference["quality"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| {
            let name = case.as_str().unwrap().to_owned();
            let train = matrix_from_npz(&mut archive, &format!("{name}__train"));
            let heldout = matrix_from_npz(&mut archive, &format!("{name}__heldout"));
            (name, train, heldout)
        })
        .collect();
    (reference, cases)
}

#[cfg(feature = "cuda")]
fn device_for_cuda() -> Option<Device> {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return None;
    };
    Some(device)
}

#[cfg(feature = "cuda")]
fn backend(device: &Device) -> BackendChoice {
    if cfg!(feature = "cuda") && fast_umap::cuda_backend::device(0).as_ref() == Some(device) {
        BackendChoice::Cuda { device_index: 0 }
    } else {
        BackendChoice::Flex
    }
}

#[cfg(not(feature = "cuda"))]
fn backend(_: &Device) -> BackendChoice {
    BackendChoice::Flex
}

fn distances(left: &[f32], right: &[f32], width: usize) -> f64 {
    left.iter()
        .zip(right)
        .take(width)
        .map(|(&a, &b)| {
            let delta = f64::from(a) - f64::from(b);
            delta * delta
        })
        .sum::<f64>()
        .sqrt()
}

fn tie_aware_recall(
    actual: &[i32],
    query: &Matrix,
    reference: &Matrix,
    k: usize,
    exclude_self: bool,
) -> f64 {
    let mut hit = 0usize;
    for row in 0..query.rows {
        let start = row * k;
        let actual_row = &actual[start..start + k];
        let mut exact: Vec<f64> = (0..reference.rows)
            .filter(|&other| !(exclude_self && row == other))
            .map(|other| {
                distances(
                    &query.values[row * query.cols..(row + 1) * query.cols],
                    &reference.values[other * reference.cols..(other + 1) * reference.cols],
                    query.cols,
                )
            })
            .collect();
        exact.sort_by(f64::total_cmp);
        let exact_cutoff = exact[k - 1];
        let mut seen = HashSet::new();
        hit += actual_row
            .iter()
            .filter(|&&id| {
                seen.insert(id) && {
                    distances(
                        &query.values[row * query.cols..(row + 1) * query.cols],
                        &reference.values
                            [id as usize * reference.cols..(id as usize + 1) * reference.cols],
                        query.cols,
                    ) <= exact_cutoff
                }
            })
            .count();
    }
    hit as f64 / (query.rows * k) as f64
}

fn trustworthiness(original: &Matrix, embedded: &Matrix, k: usize) -> f64 {
    let n = original.rows;
    assert_eq!(embedded.rows, n);
    let mut original_ranks = vec![0usize; n * n];
    let mut embedding_neighbors = Vec::with_capacity(n * k);
    for i in 0..n {
        let mut input_order: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    distances(
                        &original.values[i * original.cols..(i + 1) * original.cols],
                        &original.values[j * original.cols..(j + 1) * original.cols],
                        original.cols,
                    ),
                    j,
                )
            })
            .collect();
        input_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (rank, &(_, j)) in input_order.iter().enumerate() {
            original_ranks[i * n + j] = rank + 1;
        }
        let mut embedded_order: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                (
                    distances(
                        &embedded.values[i * embedded.cols..(i + 1) * embedded.cols],
                        &embedded.values[j * embedded.cols..(j + 1) * embedded.cols],
                        embedded.cols,
                    ),
                    j,
                )
            })
            .collect();
        embedded_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        embedding_neighbors.extend(embedded_order.into_iter().take(k).map(|(_, j)| j));
    }
    let mut penalty = 0usize;
    for i in 0..n {
        for &j in &embedding_neighbors[i * k..(i + 1) * k] {
            let rank = original_ranks[i * n + j];
            if rank > k {
                penalty += rank - k;
            }
        }
    }
    let normalizer = n as f64 * k as f64 * (2 * n - 3 * k - 1) as f64;
    1.0 - 2.0 * penalty as f64 / normalizer
}

fn matrix_from_embedding(embedding: &[Vec<f32>]) -> Matrix {
    Matrix {
        rows: embedding.len(),
        cols: embedding[0].len(),
        values: embedding.iter().flatten().copied().collect(),
    }
}

fn bounded_trustworthiness(original: &Matrix, embedded: &Matrix, k: usize) -> f64 {
    let n = original.rows;
    assert_eq!(embedded.rows, n, "trustworthiness row shape");
    assert!(k > 0 && k < n / 2, "invalid trustworthiness k={k}, n={n}");
    let mut penalty = 0usize;
    let mut original_order = Vec::with_capacity(n - 1);
    let mut embedded_order = Vec::with_capacity(n - 1);
    let mut ranks = vec![0usize; n];
    for row in 0..n {
        original_order.clear();
        embedded_order.clear();
        for other in 0..n {
            if other == row {
                continue;
            }
            original_order.push((
                distances(
                    &original.values[row * original.cols..(row + 1) * original.cols],
                    &original.values[other * original.cols..(other + 1) * original.cols],
                    original.cols,
                ),
                other,
            ));
            embedded_order.push((
                distances(
                    &embedded.values[row * embedded.cols..(row + 1) * embedded.cols],
                    &embedded.values[other * embedded.cols..(other + 1) * embedded.cols],
                    embedded.cols,
                ),
                other,
            ));
        }
        original_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        embedded_order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for (rank, &(_, id)) in original_order.iter().enumerate() {
            ranks[id] = rank + 1;
        }
        penalty += embedded_order[..k]
            .iter()
            .map(|&(_, id)| ranks[id].saturating_sub(k))
            .sum::<usize>();
    }
    let normalizer = n as f64 * k as f64 * (2 * n - 3 * k - 1) as f64;
    1.0 - 2.0 * penalty as f64 / normalizer
}

fn assert_approximate_fit_quality(
    device: &Device,
    name: &str,
    train: &Matrix,
    seeds: &[u64],
    n_neighbors: usize,
    trust_k: usize,
    reference_trust: &[f64],
) {
    assert_eq!(
        reference_trust.len(),
        seeds.len(),
        "{name}: reference seed count"
    );
    assert_eq!(
        train.values.len(),
        train.rows * train.cols,
        "{name}: input shape"
    );
    assert!(
        train.rows > n_neighbors && train.cols > 0,
        "{name}: invalid training shape"
    );
    assert!(
        train.values.iter().all(|value| value.is_finite()),
        "{name}: non-finite input"
    );
    assert!(
        reference_trust.iter().all(|value| value.is_finite()),
        "{name}: non-finite reference score"
    );
    let train_rows = train.rows_as_vec();
    let mut recall_scores = Vec::with_capacity(seeds.len());
    let mut trust_scores = Vec::with_capacity(seeds.len());
    for (seed_index, &seed) in seeds.iter().enumerate() {
        let k = n_neighbors - 1;
        let train_tensor = Tensor::<2>::from_data(
            TensorData::new(train.values.clone(), [train.rows, train.cols]),
            device,
        );
        let (ids, _) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            train_tensor.clone().into_dispatch(),
            k as u32,
            seed,
            (train.rows as f64).log2().round().max(5.0) as u32,
        );
        let ids = Tensor::<2, Int>::from_dispatch(ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        assert_eq!(ids.len(), train.rows * k, "{name} seed {seed}: ANN shape");
        assert!(
            ids.iter().all(|&id| id >= 0 && (id as usize) < train.rows),
            "{name} seed {seed}: invalid ANN id"
        );
        for row in 0..train.rows {
            let neighbors = &ids[row * k..(row + 1) * k];
            let unique = neighbors.iter().copied().collect::<HashSet<_>>();
            assert_eq!(
                unique.len(),
                k,
                "{name} seed {seed} row {row}: duplicate ANN id"
            );
            assert!(
                !unique.contains(&(row as i32)),
                "{name} seed {seed} row {row}: self neighbor was not removed by identity"
            );
        }
        let recall = tie_aware_recall(&ids, train, train, k, true);
        assert!(recall.is_finite(), "{name} seed {seed}: non-finite recall");

        let fitted = UMAP::new(
            UMAPConfig {
                n_neighbors,
                n_components: 2,
                min_dist: 0.1,
                spread: 1.0,
                local_connectivity: 1.0,
                set_op_mix_ratio: 1.0,
                n_epochs: Some(200),
                learning_rate: 1.0,
                negative_sample_rate: 5,
                repulsion_strength: 1.0,
                init: Init::Spectral,
                random_state: Some(seed),
                transform_seed: seed,
                ..UMAPConfig::default()
            },
            backend(device),
            NeighborSearch::Approximate,
        )
        .fit(&train_rows)
        .unwrap_or_else(|error| panic!("{name} seed {seed}: approximate fit failed: {error}"));
        let embedding = matrix_from_embedding(fitted.embedding());
        assert_eq!(
            (embedding.rows, embedding.cols),
            (train.rows, 2),
            "{name} seed {seed}: embedding shape"
        );
        assert!(
            embedding.values.iter().all(|value| value.is_finite()),
            "{name} seed {seed}: non-finite embedding"
        );
        let trust = bounded_trustworthiness(train, &embedding, trust_k);
        assert!(
            trust.is_finite(),
            "{name} seed {seed}: non-finite trustworthiness"
        );
        let gap = reference_trust[seed_index] - trust;
        recall_scores.push(recall);
        trust_scores.push(trust);
        eprintln!(
            "QUALITY_OBSERVATION {{\"case\":\"{name}\",\"seed\":{seed},\"recall_at_{}\":{recall:.9},\"reference_fit_trustworthiness_at_{trust_k}\":{:.9},\"candidate_fit_trustworthiness_at_{trust_k}\":{trust:.9},\"reference_minus_candidate_trustworthiness\":{gap:.9}}}",
            n_neighbors - 1,
            reference_trust[seed_index],
        );
    }
    let recall_mean = recall_scores.iter().sum::<f64>() / seeds.len() as f64;
    let trust_loss_mean = reference_trust
        .iter()
        .zip(&trust_scores)
        .map(|(reference, candidate)| reference - candidate)
        .sum::<f64>()
        / seeds.len() as f64;
    assert!(
        recall_mean >= RECALL_LIMIT,
        "{name}: five-seed mean recall@{} {recall_mean:.6} below {RECALL_LIMIT}",
        n_neighbors - 1
    );
    assert!(
        trust_loss_mean <= TRUST_GAP_LIMIT,
        "{name}: one-sided five-seed reference-minus-candidate trustworthiness@{trust_k} {trust_loss_mean:.6} exceeds {TRUST_GAP_LIMIT}"
    );
    eprintln!(
        "QUALITY_CASE_MEAN {{\"case\":\"{name}\",\"mean_recall_at_{}\":{recall_mean:.9},\"mean_reference_minus_candidate_trustworthiness_at_{trust_k}\":{trust_loss_mean:.9}}}",
        n_neighbors - 1
    );
}

fn run_quality(device: Device) {
    let backend = backend(&device);
    let (reference, cases) = fixture();
    let seeds: Vec<u64> = reference["controls"]["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|seed| seed.as_u64().unwrap())
        .collect();
    let trust_k = reference["controls"]["trustworthiness_k"].as_u64().unwrap() as usize;
    // The estimator derives transform epochs as fit n_epochs / 3 (umap-learn's
    // `UMAP.transform` rule); the frozen transform scores assume that rule.
    assert_eq!(
        reference["controls"]["transform"]["n_epochs"]
            .as_u64()
            .unwrap(),
        reference["controls"]["fit"]["n_epochs"].as_u64().unwrap() / 3,
        "estimator transform epochs no longer match frozen controls.transform.n_epochs"
    );
    let mut fit_actual = Vec::new();
    let mut fit_expected = Vec::new();
    let mut transform_actual = Vec::new();
    let mut transform_expected = Vec::new();
    for (name, train, heldout) in &cases {
        let train_rows = train.rows_as_vec();
        let heldout_rows = heldout.rows_as_vec();
        for &seed in &seeds {
            let controls = &reference["controls"]["fit"];
            let config = UMAPConfig {
                n_components: reference["controls"]["n_components"].as_u64().unwrap() as usize,
                n_neighbors: reference["controls"]["n_neighbors"].as_u64().unwrap() as usize,
                min_dist: controls["min_dist"].as_f64().unwrap() as f32,
                spread: controls["spread"].as_f64().unwrap() as f32,
                local_connectivity: controls["local_connectivity"].as_f64().unwrap() as f32,
                set_op_mix_ratio: controls["set_op_mix_ratio"].as_f64().unwrap() as f32,
                n_epochs: Some(controls["n_epochs"].as_u64().unwrap() as usize),
                learning_rate: controls["learning_rate"].as_f64().unwrap() as f32,
                negative_sample_rate: controls["negative_sample_rate"].as_u64().unwrap() as usize,
                repulsion_strength: controls["repulsion_strength"].as_f64().unwrap() as f32,
                init: Init::Spectral,
                random_state: Some(seed),
                transform_seed: seed,
                ..UMAPConfig::default()
            };
            let fitted = UMAP::new(config, backend, NeighborSearch::Exact)
                .fit(&train_rows)
                .unwrap();
            let fit_trust =
                trustworthiness(train, &matrix_from_embedding(fitted.embedding()), trust_k);
            let transformed = fitted.transform(&heldout_rows).unwrap();
            let transform_trust =
                trustworthiness(heldout, &matrix_from_embedding(&transformed), trust_k);
            let frozen = &reference["quality"]["scores"][name]
                [seeds.iter().position(|&s| s == seed).unwrap()];
            fit_actual.push(fit_trust);
            fit_expected.push(frozen["fit_trustworthiness_at_15"].as_f64().unwrap());
            transform_actual.push(transform_trust);
            transform_expected.push(frozen["transform_trustworthiness_at_15"].as_f64().unwrap());
        }
    }
    let fit_gap = (fit_actual.iter().sum::<f64>() / fit_actual.len() as f64
        - fit_expected.iter().sum::<f64>() / fit_expected.len() as f64)
        .abs();
    eprintln!(
        "{} fit trustworthiness@15 mean gap: {fit_gap:.6}",
        if matches!(backend, BackendChoice::Flex) {
            "Flex"
        } else {
            "CUDA"
        }
    );
    assert!(
        fit_gap <= TRUST_GAP_LIMIT,
        "fit trustworthiness mean gap {fit_gap:.6} exceeds {TRUST_GAP_LIMIT}"
    );
    let transform_gap = (transform_actual.iter().sum::<f64>() / transform_actual.len() as f64
        - transform_expected.iter().sum::<f64>() / transform_expected.len() as f64)
        .abs();
    eprintln!(
        "{} transform trustworthiness@15 mean gap: {transform_gap:.6}",
        if matches!(backend, BackendChoice::Flex) {
            "Flex"
        } else {
            "CUDA"
        }
    );
    assert!(
        transform_gap <= TRUST_GAP_LIMIT,
        "transform trustworthiness mean gap {transform_gap:.6} exceeds {TRUST_GAP_LIMIT}"
    );
}

fn run_recall(device: Device) {
    let (_, cases) = fixture();
    let mut fit_weighted_hits = 0usize;
    let mut fit_weighted_rows = 0usize;
    let mut query_weighted_hits = 0usize;
    let mut query_weighted_rows = 0usize;
    for (_, train, heldout) in &cases {
        let train_tensor = Tensor::<2>::from_data(
            TensorData::new(train.values.clone(), [train.rows, train.cols]),
            &device,
        );
        let fit_k = 14usize;
        let seed = 47u64;
        let (ids, _) = <Dispatch as StandardUmapOps>::approximate_neighbors(
            train_tensor.clone().into_dispatch(),
            fit_k as u32,
            seed,
            (train.rows as f64).log2().round().max(5.0) as u32,
        );
        let graph = Tensor::<2, Int>::from_dispatch(ids);
        let approximate_fit = graph.to_data().try_to_vec::<i32>().unwrap();
        let fit_recall = tie_aware_recall(&approximate_fit, train, train, fit_k, true);
        fit_weighted_hits += (fit_recall * (train.rows * fit_k) as f64).round() as usize;
        fit_weighted_rows += train.rows * fit_k;

        let query_tensor = Tensor::<2>::from_data(
            TensorData::new(heldout.values.clone(), [heldout.rows, heldout.cols]),
            &device,
        );
        let budget = approximate_query_budget(train.rows, 15, 4.0).unwrap();
        let (query_ids, _, stats) = <Dispatch as StandardUmapOps>::approximate_query(
            query_tensor.into_dispatch(),
            train_tensor.clone().into_dispatch(),
            graph.into_dispatch(),
            15,
            budget.candidates as u32,
            budget.visits as u32,
            seed,
            0,
        );
        let query_ids = Tensor::<2, Int>::from_dispatch(query_ids)
            .to_data()
            .try_to_vec::<i32>()
            .unwrap();
        assert!(
            Tensor::<2, Int>::from_dispatch(stats)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap()
                .iter()
                .all(|&value| value >= 0)
        );
        let recall = tie_aware_recall(&query_ids, heldout, train, 15, false);
        query_weighted_hits += (recall * (heldout.rows * 15) as f64).round() as usize;
        query_weighted_rows += heldout.rows * 15;
    }
    let fit_recall = fit_weighted_hits as f64 / fit_weighted_rows as f64;
    let query_recall = query_weighted_hits as f64 / query_weighted_rows as f64;
    eprintln!(
        "{} ANN tie-aware recall: fit@14={fit_recall:.6}, transform-query@15={query_recall:.6}",
        if device == Device::flex() {
            "Flex"
        } else {
            "CUDA"
        }
    );
    assert!(
        fit_recall >= RECALL_LIMIT,
        "fit ANN recall {fit_recall:.6} below {RECALL_LIMIT}"
    );
    assert!(
        query_recall >= RECALL_LIMIT,
        "transform-query ANN recall {query_recall:.6} below {RECALL_LIMIT}"
    );
}

#[test]
fn standard_ann_recall_on_flex() {
    run_recall(Device::flex());
}

#[test]
fn standard_quality_trustworthiness_on_flex() {
    run_quality(Device::flex());
}

#[cfg(feature = "cuda")]
#[test]
fn standard_ann_recall_on_cuda() {
    let Some(device) = device_for_cuda() else {
        return;
    };
    run_recall(device);
}

#[cfg(feature = "cuda")]
#[test]
fn standard_quality_trustworthiness_on_cuda() {
    let Some(device) = device_for_cuda() else {
        return;
    };
    run_quality(device);
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_approximate_fit_quality_per_frozen_fixture_case() {
    let Some(device) = device_for_cuda() else {
        return;
    };
    let (reference, cases) = fixture();
    let seeds: Vec<u64> = reference["controls"]["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|seed| seed.as_u64().unwrap())
        .collect();
    let n_neighbors = reference["controls"]["n_neighbors"].as_u64().unwrap() as usize;
    let trust_k = reference["controls"]["trustworthiness_k"].as_u64().unwrap() as usize;
    for (name, train, _) in cases {
        let seed_scores = reference["quality"]["scores"][name.as_str()]
            .as_array()
            .unwrap();
        assert_eq!(seed_scores.len(), seeds.len(), "{name}: frozen score count");
        let reference_trust = seed_scores
            .iter()
            .map(|score| score["fit_trustworthiness_at_15"].as_f64().unwrap())
            .collect::<Vec<_>>();
        assert_approximate_fit_quality(
            &device,
            &name,
            &train,
            &seeds,
            n_neighbors,
            trust_k,
            &reference_trust,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_approximate_fit_quality_campaign() {
    if std::env::var_os("FAST_UMAP_RUN_ANN_CAMPAIGN").is_none() {
        eprintln!("SKIPPED: set FAST_UMAP_RUN_ANN_CAMPAIGN=1 to run the frozen large ANN campaign");
        return;
    }
    let Some(device) = device_for_cuda() else {
        return;
    };
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let reference_path = root.join("ann_campaign_reference.json");
    let quality_path = root.join("ann_campaign_quality.npz");
    let reference: Value =
        serde_json::from_reader(File::open(&reference_path).unwrap_or_else(|error| {
            panic!(
                "campaign reference {} unavailable: {error}",
                reference_path.display()
            )
        }))
        .unwrap();
    assert_eq!(reference["schema"], "fast-umap-ann-campaign-quality-v1");
    assert_eq!(reference["data"]["shape"], serde_json::json!([16384, 16]));
    assert_eq!(reference["controls"]["n_neighbors"], 25);
    assert_eq!(
        reference["controls"]["seeds"],
        serde_json::json!([11, 29, 47, 71, 101])
    );
    let seeds = reference["controls"]["seeds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|seed| seed.as_u64().unwrap())
        .collect::<Vec<_>>();
    let train = matrix_from_npz_path(&quality_path, "campaign__train");
    assert_eq!(
        (train.rows, train.cols),
        (16384, 16),
        "campaign training shape"
    );
    let scores = reference["quality"]["scores"]["campaign"]
        .as_array()
        .unwrap();
    assert_eq!(scores.len(), seeds.len(), "campaign frozen score count");
    let reference_trust = scores
        .iter()
        .map(|score| {
            assert!(
                score["seed"].as_u64().is_some(),
                "campaign score missing seed"
            );
            score["fit_trustworthiness_at_15"].as_f64().unwrap()
        })
        .collect::<Vec<_>>();
    for (score, &seed) in scores.iter().zip(&seeds) {
        assert_eq!(
            score["seed"].as_u64(),
            Some(seed),
            "campaign score seed order"
        );
    }
    assert_approximate_fit_quality(
        &device,
        "campaign",
        &train,
        &seeds,
        25,
        15,
        &reference_trust,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_transform_error_source_tagged_capture_oracle() {
    const SOURCE: &str = "84b84ca7bbaece2aa92a9b01dd7bbc64d3fe987e";
    let Some(device) = device_for_cuda() else {
        return;
    };
    let fitted = UMAP::new(
        UMAPConfig {
            n_neighbors: 2,
            n_components: 2,
            init: Init::Random,
            random_state: Some(19),
            ..UMAPConfig::default()
        },
        backend(&device),
        NeighborSearch::Exact,
    )
    .fit(&[vec![1.0f32, 2.0]])
    .unwrap();
    let error = fitted.transform(&[vec![3.0f32, 4.0]]).unwrap_err();
    let invalid_input = structured_error(error);
    let invalid_config = structured_error(
        UMAPConfig {
            n_neighbors: 1,
            ..UMAPConfig::default()
        }
        .validate()
        .unwrap_err(),
    );
    let overflow =
        structured_error(approximate_query_budget(usize::MAX, usize::MAX, f64::MAX).unwrap_err());
    let unavailable = structured_error(
        UMAP::new(
            UMAPConfig {
                n_epochs: Some(1),
                init: Init::Random,
                random_state: Some(19),
                ..UMAPConfig::default()
            },
            BackendChoice::Cuda {
                device_index: usize::MAX,
            },
            NeighborSearch::Exact,
        )
        .fit(&[vec![0.0f32], vec![1.0], vec![2.0]])
        .unwrap_err(),
    );
    assert_eq!(invalid_input["variant"], "InvalidInput");
    assert_eq!(invalid_config["variant"], "InvalidConfig");
    assert_eq!(overflow["variant"], "Overflow");
    assert_eq!(unavailable["variant"], "BackendUnavailable");
    let mut input_bytes = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    input_bytes.extend_from_slice(b"n_neighbors=1;budget=usize::MAX");
    let semantic_capture = serde_json::json!({
        "schema": "fast-umap.cuda-error-capture.v1",
        "operation": ["single-row-transform", "config-validation", "query-budget-validation", "backend-validation"],
        "dtype": "f32/u64",
        "shapes": [[1, 2], [0, 0]],
        "input_digest": git_blob_digest(&input_bytes),
        "input_f32_bits": [1.0f32.to_bits(), 2.0f32.to_bits(), 3.0f32.to_bits(), 4.0f32.to_bits()],
        "errors": { "invalid_input": invalid_input, "invalid_config": invalid_config, "overflow": overflow, "backend_unavailable": unavailable },
        "unavailable_error_coverage": ["Numerical", "SpectralConvergence"],
    });
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ERROR_CAPTURE_PATH") {
        let mut capture = semantic_capture.as_object().unwrap().clone();
        capture.insert("source_commit".to_owned(), serde_json::json!(SOURCE));
        capture.insert(
            "product_src_tree_digest".to_owned(),
            serde_json::json!(frozen_source_digest()),
        );
        std::fs::write(path, serde_json::to_vec_pretty(&capture).unwrap()).unwrap();
    }
    if let Some(path) = std::env::var_os("FAST_UMAP_CUDA_ERROR_ORACLE_PATH") {
        let oracle: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(oracle["source_commit"], SOURCE, "oracle source revision");
        assert_eq!(
            oracle["product_src_tree_digest"],
            source_tree_digest(SOURCE),
            "oracle product source provenance"
        );
        let oracle_keys = oracle
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let expected_keys = semantic_capture
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let mut expected_oracle_keys = expected_keys;
        expected_oracle_keys.insert("source_commit".to_owned());
        expected_oracle_keys.insert("product_src_tree_digest".to_owned());
        assert_eq!(oracle_keys, expected_oracle_keys, "oracle semantic key set");
        for (field, value) in semantic_capture.as_object().unwrap() {
            assert_eq!(
                oracle[field], *value,
                "CUDA error semantic oracle mismatch in {field}"
            );
        }
    }
}

#[cfg(feature = "cuda")]
fn structured_error(error: UmapError) -> serde_json::Value {
    match error {
        UmapError::InvalidConfig(payload) => {
            serde_json::json!({"variant":"InvalidConfig", "payload":payload})
        }
        UmapError::InvalidInput(payload) => {
            serde_json::json!({"variant":"InvalidInput", "payload":payload})
        }
        UmapError::BackendUnavailable(payload) => {
            serde_json::json!({"variant":"BackendUnavailable", "payload":payload})
        }
        UmapError::Numerical(payload) => {
            serde_json::json!({"variant":"Numerical", "payload":payload})
        }
        UmapError::SpectralConvergence(payload) => {
            serde_json::json!({"variant":"SpectralConvergence", "payload":payload})
        }
        UmapError::Overflow(payload) => {
            serde_json::json!({"variant":"Overflow", "payload":payload})
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[test]
fn cuda_quality_is_required_when_requested_without_cuda_feature() {
    assert!(
        std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
        "FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled"
    );
}

fn resource_rows(rows: usize, features: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            (0..features)
                .map(|feature| {
                    let x = row as f32 * 0.001 + feature as f32 * 0.01;
                    (x * (feature as f32 + 1.0)).sin()
                })
                .collect()
        })
        .collect()
}

fn fit_without_dense_matrix(backend: BackendChoice, rows: usize) {
    let data = resource_rows(rows, 32);
    let config = UMAPConfig {
        n_epochs: Some(10),
        init: Init::Random,
        random_state: Some(101),
        ..UMAPConfig::default()
    };
    let start = std::time::Instant::now();
    let fitted = UMAP::new(config, backend, NeighborSearch::Approximate)
        .fit(&data)
        .unwrap();
    eprintln!(
        "{backend:?} approximate fit N={rows}: {:?}",
        start.elapsed()
    );
    assert_eq!(fitted.embedding().len(), rows);
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    );
}

/// A dense N x N f32 matrix at N=20,000 needs 1.6 GB; the sparse pipeline
/// stays O(N*K). Measured ~40 s in release on a Ryzen 7 9800X3D.
#[test]
fn flex_fit_larger_than_dense_memory_budget_completes() {
    let rows = 20_000usize;
    assert!(rows * rows * std::mem::size_of::<f32>() > 1_500_000_000);
    fit_without_dense_matrix(BackendChoice::Flex, rows);
}

/// N=60,000 dense f32 distances need 14.4 GB, beyond a 10 GiB RTX 3080.
#[cfg(feature = "cuda")]
#[test]
fn cuda_fit_larger_than_device_dense_memory_completes() {
    let Some(_device) = device_for_cuda() else {
        return;
    };
    let rows = 60_000usize;
    assert!(rows * rows * std::mem::size_of::<f32>() > 12 * 1024 * 1024 * 1024);
    fit_without_dense_matrix(BackendChoice::Cuda { device_index: 0 }, rows);
}
