use serde_json::Value;
use umap_core::modern_layout::{
    CompactGraph, Schedule32, accumulate_force, adam_schedule, adam_step, advance_clock,
    is_small_layout, momentum_accumulate, momentum_apply, momentum_finalize, momentum_schedule,
    negative_index,
};

fn usize_value(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}
fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter().map(|x| number(x) as f32))
        .collect()
}
fn ints(v: &Value) -> Vec<usize> {
    v.as_array().unwrap().iter().map(usize_value).collect()
}
fn doubles(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(number).collect()
}
fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!((a - b).abs() <= 3e-5, "{name}[{i}]: {a} != {b}");
    }
}

#[test]
fn pinned_upstream_modern_oracle() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/modern_layout_upstream.json"
    );
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(path)
            .expect("generate modern fixture with comparison/umap06/generate_modern_oracle.py"),
    )
    .unwrap();
    assert_eq!(fixture["provenance"]["version"], "0.6.0");
    assert_eq!(
        fixture["provenance"]["layouts_sha256"]
            .as_str()
            .unwrap()
            .len(),
        64
    );
    for s in fixture["schedules"].as_array().unwrap() {
        let n = usize_value(&s["epochs"]);
        let good = s["good"].as_bool().unwrap();
        let values: Vec<_> = if s["optimizer"] == "adam" {
            adam_schedule(n, 1.0, good).collect()
        } else {
            momentum_schedule(n, 1.0, good).collect()
        };
        for (i, value) in values.iter().enumerate() {
            assert!((value.alpha - number(&s["alpha"][i])).abs() < 1e-12);
            assert!((value.momentum - number(&s["momentum"][i])).abs() < 1e-12);
        }
    }
    for c in fixture["cases"].as_array().unwrap() {
        let rows = usize_value(&c["rows"]);
        let refs = usize_value(&c["refs"]);
        let dim = usize_value(&c["dim"]);
        let epoch = usize_value(&c["epoch"]);
        let small = is_small_layout(rows);
        let adam = c["optimizer"] == "adam";
        let fit = c["operation"] == "fit";
        let alpha = number(&c["alpha"]) as f32;
        let gamma = number(&c["gamma"]) as f32;
        let scale = number(&c["scale"]) as f32;
        let offsets = ints(&c["offsets"]);
        let indices = ints(&c["indices"]);
        let from_order = ints(&c["from_order"]);
        let to_order = ints(&c["to_order"]);
        let mut graph = CompactGraph {
            offsets: offsets.iter().map(|&x| x as u32).collect(),
            indices: indices.iter().map(|&x| x as u32).collect(),
            weights: vec![1.0; indices.len()],
            periods: doubles(&c["periods"]),
            negative_periods: doubles(&c["negative_periods"]),
            next_positive: doubles(&c["next_positive"]),
            next_negative: doubles(&c["next_negative"]),
        };
        let mut head = floats(&c["head"]);
        let tail = floats(&c["tail"]);
        let mut updates = floats(&c["initial_updates"]);
        let mut first = vec![0.0; head.len()];
        let mut second = vec![0.0; head.len()];
        let mut raw_updates = vec![0.0; head.len()];
        let schedule = Schedule32 {
            alpha,
            momentum: 0.0,
            beta1: 0.9,
            beta2: 0.99,
        };
        let block = usize_value(&c["block"]);
        for start in (0..rows).step_by(block) {
            let end = (start + block).min(rows);
            for &ordered in &from_order[start..end] {
                let row = if adam && fit {
                    to_order[ordered]
                } else {
                    ordered
                };
                let source = head[row * dim..(row + 1) * dim].to_vec();
                let mut gradient = vec![0.0; dim];
                for (edge, &target) in indices
                    .iter()
                    .enumerate()
                    .take(offsets[row + 1])
                    .skip(offsets[row])
                {
                    let Some(count) = advance_clock(&mut graph, edge, epoch).unwrap() else {
                        continue;
                    };
                    let reference = if fit { &head } else { &tail };
                    accumulate_force(
                        &mut gradient,
                        &source,
                        &reference[target * dim..(target + 1) * dim],
                        1.0,
                        1.0,
                        gamma,
                        scale,
                        small,
                        false,
                    )
                    .unwrap();
                    for sample in 0..count {
                        let rank = negative_index(edge, epoch, sample, refs).unwrap();
                        let target = to_order[rank];
                        accumulate_force(
                            &mut gradient,
                            &source,
                            &reference[target * dim..(target + 1) * dim],
                            1.0,
                            1.0,
                            gamma,
                            scale,
                            small,
                            true,
                        )
                        .unwrap();
                    }
                }
                for (d, &value) in gradient.iter().enumerate() {
                    let i = row * dim + d;
                    if adam {
                        updates[i] += value;
                    } else {
                        momentum_accumulate(&mut updates[i], value, alpha);
                    }
                }
            }
            for &ordered in &from_order[start..end] {
                let row = if adam && fit {
                    to_order[ordered]
                } else {
                    ordered
                };
                for d in 0..dim {
                    let i = row * dim + d;
                    raw_updates[i] = updates[i];
                    if adam {
                        adam_step(
                            &mut head[i],
                            &mut updates[i],
                            &mut first[i],
                            &mut second[i],
                            epoch,
                            schedule,
                        );
                    } else {
                        momentum_apply(&mut head[i], updates[i]);
                    }
                }
            }
        }
        assert_eq!(graph.next_positive, doubles(&c["expected_positive"]));
        assert_eq!(graph.next_negative, doubles(&c["expected_negative"]));
        close(&head, &floats(&c["expected_head"]), "head");
        close(&raw_updates, &floats(&c["expected_updates"]), "updates");
        close(&first, &floats(&c["expected_first"]), "first moment");
        close(&second, &floats(&c["expected_second"]), "second moment");
        if !adam {
            momentum_finalize(&mut updates, number(&c["momentum"]) as f32);
        }
        close(
            &updates,
            &floats(&c["expected_final_updates"]),
            "final updates",
        );
    }
}
