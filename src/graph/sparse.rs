// Currently only used for the parametric UMAP implementation, remove the feature guard if needed elsewhere.
#![cfg(feature = "parametric")]

use burn::tensor::{Device, Tensor};
use num::{Float, ToPrimitive};
use std::collections::{BTreeMap, BTreeSet};

use crate::graph::get_distance_by_metric::{knn_from_pairwise_cpu, pairwise_distances};
use crate::utils::convert_vector_to_tensor;

pub(crate) const NN_DESCENT_THRESHOLD: usize = 10_000;

/// Directed graph edge membership after fuzzy graph symmetrization.
#[derive(Debug, Clone, Copy)]
pub(crate) struct WeightedEdge {
    pub(crate) head: usize,
    pub(crate) tail: usize,
    pub(crate) weight: f32,
}

/// Compute smooth-kNN bandwidths for rows whose first slot is self.
pub(crate) fn smooth_knn_dist(
    distances: &[f32],
    n: usize,
    k: usize,
    local_connectivity: f64,
    bandwidth: f64,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(distances.len(), n * k);
    assert!(n > 0 && k >= 2);
    assert!(local_connectivity.is_finite() && local_connectivity >= 0.0);
    assert!(bandwidth.is_finite() && bandwidth > 0.0);
    assert!(distances.iter().all(|d| d.is_finite() && *d >= 0.0));

    let target = (k as f64).log2() * bandwidth;
    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    let mut sigmas = Vec::with_capacity(n);
    let mut rhos = Vec::with_capacity(n);

    for row in distances.chunks_exact(k) {
        let positive: Vec<f32> = row.iter().copied().filter(|d| *d > 0.0).collect();
        let mut rho = 0.0f32;
        if !positive.is_empty() {
            if positive.len() as f64 >= local_connectivity {
                let index = local_connectivity.floor() as usize;
                let interpolation = local_connectivity - index as f64;
                if index > 0 {
                    rho = positive[index - 1];
                    if interpolation > 1e-5 {
                        rho +=
                            (interpolation * (positive[index] - positive[index - 1]) as f64) as f32;
                    }
                } else {
                    rho = (interpolation * positive[0] as f64) as f32;
                }
            } else {
                rho = positive.iter().copied().fold(0.0f32, f32::max);
            }
        }

        let mut lo = 0.0f64;
        let mut hi = f32::MAX as f64;
        let mut mid = 1.0f64;
        for _ in 0..64 {
            let sum: f64 = row[1..]
                .iter()
                .map(|distance| {
                    let d = *distance - rho;
                    if d > 0.0 {
                        (-(d as f64 / mid)).exp()
                    } else {
                        1.0
                    }
                })
                .sum();
            if (sum - target).abs() < 1e-5 {
                break;
            }
            if sum > target {
                hi = mid;
                mid = (lo + hi) / 2.0;
            } else {
                lo = mid;
                mid = if hi >= f32::MAX as f64 {
                    mid * 2.0
                } else {
                    (lo + hi) / 2.0
                };
            }
        }
        let floor_mean = if rho > 0.0 {
            row.iter().sum::<f32>() / k as f32
        } else {
            mean
        };
        sigmas.push((mid as f32).max(1e-3 * floor_mean));
        rhos.push(rho);
    }
    (sigmas, rhos)
}

pub(crate) fn directed_memberships(
    indices: &[i32],
    distances: &[f32],
    sigmas: &[f32],
    rhos: &[f32],
    k: usize,
) -> BTreeMap<(usize, usize), f32> {
    let n = sigmas.len();
    assert_eq!(rhos.len(), n);
    assert_eq!(indices.len(), n * k);
    assert_eq!(distances.len(), n * k);
    let mut edges = BTreeMap::new();
    for head in 0..n {
        for slot in 0..k {
            let offset = head * k + slot;
            let tail = indices[offset];
            if tail == -1 || tail == head as i32 {
                continue;
            }
            assert!(tail >= 0 && (tail as usize) < n, "invalid neighbor index");
            let d = distances[offset] - rhos[head];
            let weight = if d <= 0.0 || sigmas[head] == 0.0 {
                1.0
            } else {
                (-d / sigmas[head]).exp()
            };
            if weight > 0.0 {
                *edges.entry((head, tail as usize)).or_insert(0.0) += weight;
            }
        }
    }
    edges
}

pub(crate) fn symmetrize(directed: &BTreeMap<(usize, usize), f32>, mix: f64) -> Vec<WeightedEdge> {
    assert!(mix.is_finite() && (0.0..=1.0).contains(&mix));
    let mix = mix as f32;
    let mut pairs = BTreeMap::new();
    for &(head, tail) in directed.keys() {
        pairs.insert((head.min(tail), head.max(tail)), ());
    }
    let mut edges = Vec::with_capacity(pairs.len() * 2);
    for ((head, tail), ()) in pairs {
        let a = directed.get(&(head, tail)).copied().unwrap_or(0.0);
        let b = directed.get(&(tail, head)).copied().unwrap_or(0.0);
        let intersection = a * b;
        let weight = mix * (a + b - intersection) + (1.0 - mix) * intersection;
        if weight > 0.0 {
            edges.push(WeightedEdge { head, tail, weight });
            edges.push(WeightedEdge {
                head: tail,
                tail: head,
                weight,
            });
        }
    }
    edges.sort_by_key(|edge| (edge.head, edge.tail));
    edges
}

pub(crate) fn with_self_slots(
    indices: Vec<i32>,
    distances: Vec<f32>,
    n: usize,
    k: usize,
) -> (Vec<i32>, Vec<f32>) {
    assert!(k >= 2);
    assert_eq!(indices.len(), n * (k - 1));
    assert_eq!(distances.len(), indices.len());
    let mut out_indices = Vec::with_capacity(n * k);
    let mut out_distances = Vec::with_capacity(n * k);
    for head in 0..n {
        out_indices.push(head as i32);
        out_distances.push(0.0);
        let mut seen = BTreeSet::new();
        let start = head * (k - 1);
        for slot in start..start + k - 1 {
            let tail = indices[slot];
            if tail == head as i32 {
                continue;
            }
            assert!(
                tail >= 0 && (tail as usize) < n && seen.insert(tail),
                "invalid or duplicate neighbor row"
            );
            assert!(
                distances[slot].is_finite() && distances[slot] >= 0.0,
                "invalid neighbor distance"
            );
            out_indices.push(tail);
            out_distances.push(distances[slot]);
        }
        assert_eq!(
            out_indices.len(),
            (head + 1) * k,
            "short neighbor row after self removal"
        );
    }
    (out_indices, out_distances)
}

pub(crate) fn sparse_inputs<F: Float + ToPrimitive + burn::tensor::Element>(
    data: Vec<F>,
    num_samples: usize,
    num_features: usize,
    k: usize,
    device: &Device,
) -> (Tensor<2>, Vec<i32>, Vec<f32>) {
    let all_data_tensor = convert_vector_to_tensor(data.clone(), num_samples, num_features, device);

    #[cfg(feature = "nn-descent")]
    if num_samples >= NN_DESCENT_THRESHOLD {
        let data_f32: Vec<f32> = data
            .iter()
            .map(|x| num::ToPrimitive::to_f32(x).unwrap())
            .collect();
        let (indices, distances) =
            crate::nn_descent::nn_descent(&data_f32, num_samples, num_features, k - 1);
        let (indices, distances) = with_self_slots(indices, distances, num_samples, k);
        return (all_data_tensor, indices, distances);
    }

    let pairwise = pairwise_distances(all_data_tensor.clone());
    let flat: Vec<f32> = pairwise.to_data().try_to_vec::<f32>().unwrap();
    let (indices, distances) = knn_from_pairwise_cpu(&flat, num_samples, k - 1);
    let (indices, distances) = with_self_slots(indices, distances, num_samples, k);
    (all_data_tensor, indices, distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use std::collections::BTreeMap;

    fn floats(value: &serde_json::Value) -> Vec<f32> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    }

    fn close(actual: &[f32], expected: &[f32], context: &str) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (*a - *b).abs() <= 1e-5 + 1e-4 * b.abs(),
                "{context}[{index}]: {a} != {b}"
            );
        }
    }

    #[test]
    fn reference_graph_fixtures_match_memberships_and_symmetrization() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/umap_learn_0.5.12_fuzzy_graph.json"
        ))
        .unwrap();
        assert_eq!(fixture["reference_version"], "0.5.12");
        for case in fixture["cases"].as_array().unwrap() {
            let n = case["n"].as_u64().unwrap() as usize;
            let k = case["k"].as_u64().unwrap() as usize;
            let name = case["name"].as_str().unwrap();
            let indices: Vec<i32> = case["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap() as i32)
                .collect();
            let distances = floats(&case["distances"]);
            let (sigmas, rhos) = smooth_knn_dist(
                &distances,
                n,
                k,
                case["local_connectivity"].as_f64().unwrap(),
                case["bandwidth"].as_f64().unwrap(),
            );
            close(&sigmas, &floats(&case["sigma"]), name);
            close(&rhos, &floats(&case["rho"]), name);
            let directed = directed_memberships(&indices, &distances, &sigmas, &rhos, k);
            let mut dense = vec![0.0; n * n];
            for (&(head, tail), &weight) in &directed {
                dense[head * n + tail] = weight;
            }
            close(&dense, &floats(&case["directed"]), name);
            let edges = symmetrize(&directed, case["set_op_mix_ratio"].as_f64().unwrap());
            let mut symmetric = vec![0.0; n * n];
            for edge in &edges {
                symmetric[edge.head * n + edge.tail] = edge.weight;
                assert_eq!(
                    edge.weight,
                    edges
                        .iter()
                        .find(|other| other.head == edge.tail && other.tail == edge.head)
                        .unwrap()
                        .weight
                );
            }
            close(&symmetric, &floats(&case["symmetric"]), name);
            for i in 0..n {
                assert_eq!(symmetric[i * n + i], 0.0);
            }
        }
    }

    #[test]
    fn duplicate_memberships_coalesce_without_confusing_self() {
        let directed =
            directed_memberships(&[0, 1, 1, 1, 0, -1], &[0.0; 6], &[0.0; 2], &[0.0; 2], 3);
        assert_eq!(directed.len(), 2);
        assert_eq!(directed[&(0, 1)], 2.0);
        assert_eq!(directed[&(1, 0)], 1.0);
        let asymmetric = BTreeMap::from([((0, 1), 0.5), ((1, 0), 0.2), ((1, 2), 0.8)]);
        let union = symmetrize(&asymmetric, 1.0);
        close(
            &union.iter().map(|edge| edge.weight).collect::<Vec<_>>(),
            &[0.6, 0.6, 0.8, 0.8],
            "union",
        );
        let intersection = symmetrize(&asymmetric, 0.0);
        assert_eq!(intersection.len(), 2);
        close(
            &intersection
                .iter()
                .map(|edge| edge.weight)
                .collect::<Vec<_>>(),
            &[0.1, 0.1],
            "intersection",
        );
    }

    #[test]
    fn self_slot_preserves_duplicate_zero_neighbors_and_rejects_short_rows() {
        let (indices, distances) = with_self_slots(vec![1, 0], vec![0.0, 0.0], 2, 2);
        assert_eq!(indices, vec![0, 1, 1, 0]);
        assert_eq!(distances, vec![0.0; 4]);
        assert!(
            std::panic::catch_unwind(|| with_self_slots(vec![0, 1], vec![0.0; 2], 2, 2)).is_err()
        );
    }

    #[test]
    fn self_slot_rejects_duplicate_self_and_invalid_neighbor_indices() {
        assert!(
            std::panic::catch_unwind(|| { with_self_slots(vec![1, 1, 0, 2], vec![0.0; 4], 2, 3) })
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { with_self_slots(vec![0, 0, 1, 2], vec![0.0; 4], 2, 3) })
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { with_self_slots(vec![-2, 0], vec![0.0; 2], 2, 2) })
                .is_err()
        );
        assert!(
            std::panic::catch_unwind(|| { with_self_slots(vec![2, 0], vec![0.0; 2], 2, 2) })
                .is_err()
        );
    }

    #[test]
    fn self_slot_rejects_invalid_distances_for_valid_neighbors() {
        for distance in [f32::NAN, f32::INFINITY, -1.0] {
            assert!(
                std::panic::catch_unwind(|| {
                    with_self_slots(vec![1, 0], vec![distance, 0.0], 2, 2)
                })
                .is_err()
            );
        }
    }

    #[test]
    fn directed_memberships_rejects_non_sentinel_invalid_indices() {
        for index in [-2, 2] {
            assert!(
                std::panic::catch_unwind(|| {
                    directed_memberships(&[0, index], &[0.0, 1.0], &[1.0], &[0.0], 2)
                })
                .is_err()
            );
        }
    }

    #[test]
    fn directed_memberships_preserves_existing_invalid_distance_semantics() {
        // The helper does not validate distances: negative deltas, including
        // negative infinity, clamp to membership one; NaN and positive
        // infinity yield no edge through the final positive-weight check.
        let directed = directed_memberships(
            &[0, 1, 1, 0],
            &[f32::NAN, -1.0, f32::NAN, f32::NEG_INFINITY],
            &[1.0; 2],
            &[0.0; 2],
            2,
        );
        assert_eq!(directed, BTreeMap::from([((0, 1), 1.0), ((1, 0), 1.0)]));

        // Self slots and the -1 padding sentinel are skipped before distance
        // arithmetic, even when their corresponding distances are non-finite.
        let skipped = directed_memberships(
            &[0, -1, 1, -1],
            &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            &[1.0; 2],
            &[0.0; 2],
            2,
        );
        assert!(skipped.is_empty());
    }

    #[test]
    fn sparse_preparation_uses_raw_values_for_knn_and_encoder_input() {
        let data = vec![
            0.0f32, 0.0, 7.0, // A
            0.0, 100.0, 7.0, // B
            10.0, 1.0, 7.0, // C
            0.0, 10_000.0, 7.0, // scale outlier
        ];
        let (encoder_input, knn_indices, distances) = sparse_inputs(data, 4, 3, 2, &Device::cpu());

        // In raw space A is closest to C; z-scoring would make B closer.
        assert_eq!(knn_indices[..2], [0, 2]);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 101.0f32.sqrt()).abs() < 1e-5);
        let encoder_values = encoder_input.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(
            encoder_values,
            vec![
                0.0, 0.0, 7.0, 0.0, 100.0, 7.0, 10.0, 1.0, 7.0, 0.0, 10_000.0, 7.0,
            ]
        );
    }

    #[cfg(feature = "nn-descent")]
    #[test]
    fn approximate_sparse_preparation_preserves_raw_values_at_threshold() {
        let num_samples = NN_DESCENT_THRESHOLD;
        let data: Vec<f32> = (0..num_samples)
            .flat_map(|index| match index {
                0 => [0.0, 0.0, 7.0],
                _ => {
                    let jitter = (index / 3) as f32 * 0.000_001;
                    match index % 3 {
                        0 => [0.0, 100.0 + jitter, 7.0],
                        1 => [10.0, 1.0 + jitter, 7.0],
                        _ => [0.0, 10_000.0 + jitter, 7.0],
                    }
                }
            })
            .collect();
        let k = 15;
        let (encoder_input, indices, distances) =
            sparse_inputs(data.clone(), num_samples, 3, k, &Device::cpu());

        // Raw query neighbors come from [10, ~1, 7]; z-scoring the
        // NN-Descent-only input instead would select [0, ~100, 7].
        for &neighbor in &indices[1..k] {
            assert_eq!(data[neighbor as usize * 3], 10.0);
        }
        assert_eq!(encoder_input.to_data().try_to_vec::<f32>().unwrap(), data);
        assert_eq!(indices.len(), num_samples * k);
        for (sample, neighbors) in indices.chunks_exact(k).enumerate() {
            assert_eq!(neighbors[0] as usize, sample);
            assert_eq!(distances[sample * k], 0.0);
            for &neighbor in &neighbors[1..] {
                assert!(neighbor >= 0 && (neighbor as usize) < num_samples);
                assert_ne!(neighbor as usize, sample);
            }
        }
    }
}
