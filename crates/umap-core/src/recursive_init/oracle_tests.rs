use super::geometry::{Anchor, base_case, procrustes_align, prolongate};
use super::graph::HostCsr;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    base_cases: Vec<BaseCase>,
    prolongation_cases: Vec<ProlongationCase>,
    procrustes_cases: Vec<ProcrustesCase>,
}

#[derive(Deserialize)]
struct Graph {
    offsets: Vec<usize>,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

impl Graph {
    fn into_csr(self) -> HostCsr {
        HostCsr {
            n: self.offsets.len() - 1,
            indptr: self.offsets,
            indices: self.indices,
            weights: self.weights,
        }
    }
}

#[derive(Deserialize)]
struct BaseCase {
    name: String,
    graph: Graph,
    anchor: Vec<Vec<f32>>,
    sample_mask: Vec<bool>,
    depth: usize,
    result: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct ProlongationCase {
    name: String,
    graph: Graph,
    labels: Vec<i32>,
    coarse: Vec<Vec<f32>>,
    centered_f32: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct ProcrustesCase {
    name: String,
    anchor: Vec<Vec<f32>>,
    embedding: Vec<Vec<f32>>,
    aligned: Vec<Vec<f32>>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("../../tests/fixtures/recursive_oracle.json")).unwrap()
}

fn flatten(rows: &[Vec<f32>], dimensions: usize) -> Vec<f32> {
    assert!(rows.iter().all(|row| row.len() == dimensions));
    rows.iter().flatten().copied().collect()
}

fn assert_f32_oracle(name: &str, actual: &[f32], expected: &[Vec<f32>], dimensions: usize) {
    let expected = flatten(expected, dimensions);
    assert_eq!(actual.len(), expected.len(), "{name}: shape");
    for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{name}[{index}]: non-finite"
        );
        let tolerance = 1e-6 + 1e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{name}[{index}]: actual {actual}, expected {expected}, tolerance {tolerance}"
        );
    }
}

#[test]
fn base_cases_match_upstream() {
    for case in fixture().base_cases {
        let graph = case.graph.into_csr();
        let dimensions = case.anchor[0].len();
        assert_eq!(case.sample_mask.len(), graph.n, "{}: mask shape", case.name);
        assert_eq!(case.result.len(), graph.n, "{}: output rows", case.name);
        assert_eq!(
            case.anchor.len(),
            case.sample_mask.iter().filter(|&&x| x).count()
        );
        let mut values = vec![0.0; graph.n * dimensions];
        let mut sampled = case.anchor.iter();
        for (row, &present) in case.sample_mask.iter().enumerate() {
            if present {
                let coords = sampled.next().unwrap();
                assert_eq!(coords.len(), dimensions);
                values[row * dimensions..(row + 1) * dimensions].copy_from_slice(coords);
            }
        }
        let anchor = Anchor {
            values,
            mask: case.sample_mask,
            dimensions,
        };
        let actual = base_case(&graph, &anchor, case.depth).unwrap();
        assert_f32_oracle(&case.name, &actual, &case.result, dimensions);
    }
}

#[test]
fn prolongation_cases_match_upstream() {
    for case in fixture().prolongation_cases {
        let graph = case.graph.into_csr();
        let dimensions = case.coarse[0].len();
        assert_eq!(case.labels.len(), graph.n, "{}: label shape", case.name);
        assert_eq!(
            case.centered_f32.len(),
            graph.n,
            "{}: output rows",
            case.name
        );
        let coarse = flatten(&case.coarse, dimensions);
        let actual = prolongate(&graph, &case.labels, &coarse, dimensions).unwrap();
        assert_f32_oracle(&case.name, &actual, &case.centered_f32, dimensions);
    }
}

#[test]
fn procrustes_cases_match_upstream() {
    for case in fixture().procrustes_cases {
        let dimensions = case.anchor[0].len();
        assert_eq!(
            case.embedding.len(),
            case.anchor.len(),
            "{}: input rows",
            case.name
        );
        assert_eq!(
            case.aligned.len(),
            case.anchor.len(),
            "{}: output rows",
            case.name
        );
        let anchor = Anchor {
            values: flatten(&case.anchor, dimensions),
            mask: vec![true; case.anchor.len()],
            dimensions,
        };
        let mut actual = flatten(&case.embedding, dimensions);
        procrustes_align(&mut actual, &anchor).unwrap();
        assert_f32_oracle(&case.name, &actual, &case.aligned, dimensions);
    }
}
