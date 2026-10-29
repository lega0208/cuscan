use hdbscan_core::{
    CondensedTree, HdbscanError, PredictionColumns, PredictionHierarchy,
    prediction::{candidate_width, prediction_core_from_squared_rows},
};

fn tree(rows: &[(usize, usize, f32, f32)]) -> CondensedTree {
    CondensedTree {
        parent: rows.iter().map(|row| row.0).collect(),
        child: rows.iter().map(|row| row.1).collect(),
        lambda: rows.iter().map(|row| row.2).collect(),
        child_size: rows.iter().map(|row| row.3).collect(),
    }
}

fn identity(n: u32) -> Vec<u32> {
    (0..n).collect()
}

/// Root 6; cluster 7 = points 0..3 (max lambda 4); cluster 8 = points 3..6 (max 1).
fn two_clusters() -> CondensedTree {
    tree(&[
        (6, 7, 0.5, 3.0),
        (6, 8, 0.5, 3.0),
        (7, 0, 2.0, 1.0),
        (7, 1, 2.0, 1.0),
        (7, 2, 4.0, 1.0),
        (8, 3, 1.0, 1.0),
        (8, 4, 1.0, 1.0),
        (8, 5, 1.0, 1.0),
    ])
}

fn two_cluster_model() -> PredictionHierarchy {
    PredictionHierarchy::new(&two_clusters(), &[0, 0, 0, 1, 1, 1], &identity(6)).unwrap()
}

fn assert_close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() <= 1e-12, "{actual} != {expected}");
}

fn is_invalid<T: std::fmt::Debug>(result: Result<T, HdbscanError>) -> bool {
    matches!(result, Err(HdbscanError::InvalidInput(_)))
}

fn is_numerical<T: std::fmt::Debug>(result: Result<T, HdbscanError>) -> bool {
    matches!(result, Err(HdbscanError::Numerical(_)))
}

#[test]
fn core_rank_includes_self_at_s_minus_one() {
    let cores = prediction_core_from_squared_rows(&[0.0, 4.0, 0.0, 9.0], 2).unwrap();
    assert_eq!(cores, vec![2.0, 3.0]);
    let cores = prediction_core_from_squared_rows(&[0.0, 1.0, 4.0], 1).unwrap();
    assert_eq!(cores, vec![0.0; 3]);
    assert!(is_invalid(prediction_core_from_squared_rows(&[0.0; 3], 2)));
    assert!(is_invalid(prediction_core_from_squared_rows(&[], 0)));
    assert!(is_numerical(prediction_core_from_squared_rows(
        &[0.0, f32::NAN],
        2
    )));
    assert!(is_numerical(prediction_core_from_squared_rows(
        &[0.0, -1.0],
        2
    )));
    assert!(is_numerical(prediction_core_from_squared_rows(
        &[0.0, f32::INFINITY],
        2
    )));
}

#[test]
fn candidate_width_is_checked() {
    assert_eq!(candidate_width(3, 6).unwrap(), 6);
    assert!(is_invalid(candidate_width(4, 6)));
    assert!(is_invalid(candidate_width(0, 6)));
    assert!(matches!(
        candidate_width(usize::MAX, usize::MAX),
        Err(HdbscanError::Overflow(_))
    ));
}

#[test]
fn query_core_uses_rank_s_and_neighbor_core_matters() {
    let model = two_cluster_model();
    // s = 2: query core is distances[2] = 1.0 -> lambda 1 -> 1/4.
    let cores = vec![0.0; 6];
    let p = model
        .predict_candidates(&[0, 1, 2, 3], &[0.2, 0.5, 1.0, 1.5], &cores, 2)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (0, 0));
    assert_close(p.probability, 0.25);
    // Neighbor core of id 3 dominates position 0, so position 1 (id 4) wins.
    let mut cores = vec![0.0; 6];
    cores[3] = 3.0;
    let p = model
        .predict_candidates(&[3, 4, 0, 1], &[0.2, 0.5, 1.0, 1.5], &cores, 2)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (1, 4));
    assert_close(p.probability, 1.0);
    // Raw distance dominates both cores.
    let p = model
        .predict_candidates(&[0, 1], &[0.1, 0.2], &[0.0; 6], 1)
        .unwrap();
    assert_close(p.probability, 1.0); // lambda 5 > max 4
}

#[test]
fn equal_mrd_keeps_first_candidate_not_lowest_id() {
    let model = two_cluster_model();
    let mut cores = vec![0.0; 6];
    cores[3] = 1.0;
    cores[1] = 1.0;
    let p = model
        .predict_candidates(&[3, 1, 0, 2], &[0.2, 0.9, 1.0, 1.5], &cores, 2)
        .unwrap();
    assert_eq!(p.connecting_point, 3);
    assert_eq!(p.label, 1);
}

#[test]
fn zero_mrd_uses_f64_max_and_infinite_maxima_follow_formula() {
    let model = two_cluster_model();
    let p = model
        .predict_candidates(&[2, 1], &[0.0, 0.0], &[0.0; 6], 1)
        .unwrap();
    assert_eq!(p.label, 0);
    assert_close(p.probability, 1.0);

    let mut infinite = two_clusters();
    infinite.lambda[4] = f32::INFINITY;
    let model = PredictionHierarchy::new(&infinite, &[0, 0, 0, 1, 1, 1], &identity(6)).unwrap();
    // Duplicate query: finite f64::MAX lambda against infinite max gives strength 0.
    let p = model
        .predict_candidates(&[2, 1], &[0.0, 0.0], &[0.0; 6], 1)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (0, 2));
    assert_eq!(p.probability, 0.0);
}

#[test]
fn departure_and_birth_boundaries() {
    let model = two_cluster_model();
    let cores = [0.0; 6];
    let predict = |query_core: f64| {
        model
            .predict_candidates(&[0, 1], &[0.0, query_core], &cores, 1)
            .unwrap()
    };
    // departure 2 < lambda 4: no ascent.
    let p = predict(0.25);
    assert_eq!(p.label, 0);
    assert_close(p.probability, 1.0);
    // departure 2 == lambda 2: strict `>` means no ascent.
    let p = predict(0.5);
    assert_eq!(p.label, 0);
    assert_close(p.probability, 0.5);
    // departure 2 > lambda 1, birth 0.5 < lambda: stop at cluster 7.
    let p = predict(1.0);
    assert_eq!(p.label, 0);
    assert_close(p.probability, 0.25);
    // birth 0.5 == lambda 0.5: `>=` ascends to the unmapped root; connector kept.
    let p = predict(2.0);
    assert_eq!((p.label, p.probability, p.connecting_point), (-1, 0.0, 0));
    // birth 0.5 > lambda 0.25: ascends to root.
    let p = predict(4.0);
    assert_eq!((p.label, p.probability, p.connecting_point), (-1, 0.0, 0));
}

#[test]
fn descendants_inherit_anchor_label_and_max_including_cluster_rows() {
    // Selected 7 owns points 0,1 (lambda 1) and child clusters 8, 9 (lambda 2).
    let condensed = tree(&[
        (6, 7, 0.5, 6.0),
        (7, 0, 1.0, 1.0),
        (7, 1, 1.0, 1.0),
        (7, 8, 2.0, 2.0),
        (7, 9, 2.0, 2.0),
        (8, 2, 5.0, 1.0),
        (8, 3, 5.0, 1.0),
        (9, 4, 6.0, 1.0),
        (9, 5, 6.0, 1.0),
    ]);
    let model = PredictionHierarchy::new(&condensed, &[0; 6], &identity(6)).unwrap();
    let cores = [0.0; 6];
    // Stays in descendant 8, using ancestor max 2.
    let p = model
        .predict_candidates(&[2, 3], &[0.0, 0.1], &cores, 1)
        .unwrap();
    assert_eq!(p.label, 0);
    assert_close(p.probability, 1.0);
    // Ascends 8 -> 7 (birth 2 >= 1.25), max 2 from cluster-child rows.
    let p = model
        .predict_candidates(&[2, 3], &[0.0, 0.8], &cores, 1)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (0, 2));
    assert_close(p.probability, 0.625);
}

#[test]
fn reference_anchor_quirk_leaves_sibling_unmapped() {
    let condensed = tree(&[
        (4, 5, 1.0, 2.0),
        (4, 6, 1.0, 2.0),
        (5, 0, 2.0, 1.0),
        (5, 1, 2.0, 1.0),
        (6, 2, 3.0, 1.0),
        (6, 3, 3.0, 1.0),
    ]);
    let model = PredictionHierarchy::new(&condensed, &[0; 4], &identity(4)).unwrap();
    let cores = [0.0; 4];
    let p = model
        .predict_candidates(&[0, 1], &[0.0, 0.1], &cores, 1)
        .unwrap();
    assert_eq!(p.label, 0);
    let p = model
        .predict_candidates(&[2, 3], &[0.0, 0.1], &cores, 1)
        .unwrap();
    assert_eq!((p.label, p.probability, p.connecting_point), (-1, 0.0, 2));
}

#[test]
fn label_gaps_are_preserved_and_noise_is_not_an_anchor() {
    let model =
        PredictionHierarchy::new(&two_clusters(), &[0, 0, -1, 3, 3, 3], &identity(6)).unwrap();
    let p = model
        .predict_candidates(&[4, 3], &[0.0, 0.5], &[0.0; 6], 1)
        .unwrap();
    assert_eq!(p.label, 3);
    let p = model
        .predict_candidates(&[2, 1], &[0.0, 0.25], &[0.0; 6], 1)
        .unwrap();
    assert_eq!(p.label, 0);
}

#[test]
fn cluster_edges_without_labels_still_predict_connectors() {
    let model = PredictionHierarchy::new(&two_clusters(), &[-1; 6], &identity(6)).unwrap();
    assert!(!model.no_cluster_tree());
    let p = model
        .predict_candidates(&[5, 4], &[0.0, 0.5], &[0.0; 6], 1)
        .unwrap();
    assert_eq!((p.label, p.probability, p.connecting_point), (-1, 0.0, 5));
}

#[test]
fn no_cluster_tree_states() {
    let root_only = tree(&[
        (4, 0, 1.0, 1.0),
        (4, 1, 1.0, 1.0),
        (4, 2, 1.0, 1.0),
        (4, 3, 1.0, 1.0),
    ]);
    let model = PredictionHierarchy::new(&root_only, &[0; 4], &identity(4)).unwrap();
    assert!(model.no_cluster_tree());
    assert!(is_invalid(model.predict_candidates(
        &[0, 1],
        &[0.0, 0.1],
        &[0.0; 4],
        1
    )));
    let one = PredictionHierarchy::new(&CondensedTree::default(), &[-1], &[0]).unwrap();
    assert!(one.no_cluster_tree());
    assert_eq!(
        PredictionColumns::no_cluster_tree(2),
        PredictionColumns {
            labels: vec![-1, -1],
            probabilities: vec![0.0, 0.0],
            connecting_points: vec![-1, -1],
        }
    );
}

#[test]
fn nonfinite_training_rows_map_connectors_to_raw_ids() {
    let condensed = tree(&[
        (7, 3, 0.0, 1.0),
        (7, 8, 0.5, 3.0),
        (7, 9, 0.5, 3.0),
        (8, 0, 2.0, 1.0),
        (8, 1, 2.0, 1.0),
        (8, 2, 2.0, 1.0),
        (9, 4, 2.0, 1.0),
        (9, 5, 2.0, 1.0),
        (9, 6, 2.0, 1.0),
    ]);
    let model =
        PredictionHierarchy::new(&condensed, &[0, 0, 0, -1, 1, 1, 1], &[0, 1, 2, 4, 5, 6]).unwrap();
    let p = model
        .predict_candidates(&[3, 4], &[0.0, 0.25], &[0.0; 6], 1)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (1, 4));
}

#[test]
fn squared_batch_appends_in_order_and_is_atomic() {
    let model = two_cluster_model();
    let mut out = PredictionColumns::default();
    model
        .predict_squared_batch(
            &[0, 1, 4, 3],
            &[0.0, 0.0625, 0.0, 1.0],
            &[0.0; 6],
            1,
            &mut out,
        )
        .unwrap();
    assert_eq!(out.labels, vec![0, 1]);
    assert_eq!(out.connecting_points, vec![0, 4]);
    assert_close(out.probabilities[0], 1.0);
    assert_close(out.probabilities[1], 1.0);
    let before = out.clone();
    assert!(is_numerical(model.predict_squared_batch(
        &[0, 1, 4, 6],
        &[0.0; 4],
        &[0.0; 6],
        1,
        &mut out
    )));
    assert!(is_numerical(model.predict_squared_batch(
        &[0, 1, 4, 3],
        &[0.0, 0.0, 0.0, f32::NAN],
        &[0.0; 6],
        1,
        &mut out
    )));
    assert!(is_invalid(model.predict_squared_batch(
        &[0, 1, 4],
        &[0.0; 3],
        &[0.0; 6],
        1,
        &mut out
    )));
    assert!(is_invalid(model.predict_squared_batch(
        &[0, 1],
        &[0.0; 2],
        &[0.0; 5],
        1,
        &mut out
    )));
    assert!(is_numerical(model.predict_squared_batch(
        &[0, 1],
        &[0.0; 2],
        &[f64::NAN, 0.0, 0.0, 0.0, 0.0, 0.0],
        1,
        &mut out
    )));
    assert_eq!(out, before);
}

#[test]
fn malformed_candidates_are_typed_errors() {
    let model = two_cluster_model();
    let cores = [0.0; 6];
    assert!(is_invalid(model.predict_candidates(
        &[0],
        &[0.0],
        &cores,
        1
    )));
    assert!(is_numerical(model.predict_candidates(
        &[0, -1],
        &[0.0, 0.1],
        &cores,
        1
    )));
    assert!(is_numerical(model.predict_candidates(
        &[0, 6],
        &[0.0, 0.1],
        &cores,
        1
    )));
    assert!(is_numerical(model.predict_candidates(
        &[0, 1],
        &[0.0, f64::NAN],
        &cores,
        1
    )));
    assert!(is_numerical(model.predict_candidates(
        &[0, 1],
        &[-1.0, 0.1],
        &cores,
        1
    )));
    assert!(is_invalid(model.predict_candidates(
        &[0, 1],
        &[0.0, 0.1],
        &cores[..5],
        1
    )));
    let mut bad_cores = cores;
    bad_cores[1] = -1.0;
    assert!(is_numerical(model.predict_candidates(
        &[0, 1],
        &[0.0, 0.1],
        &bad_cores,
        1
    )));
    assert!(is_invalid(
        model.predict_candidates(&[0; 8], &[0.0; 8], &cores, 4)
    ));
    assert!(is_invalid(model.predict_candidates(&[], &[], &cores, 0)));
}

#[test]
fn malformed_hierarchy_is_rejected() {
    let labels = [0, 0, 0, 1, 1, 1];
    let ids = identity(6);
    let build = |condensed: &CondensedTree| PredictionHierarchy::new(condensed, &labels, &ids);

    let mut columns = two_clusters();
    columns.lambda.pop();
    assert!(is_invalid(build(&columns)));

    let mut duplicate = two_clusters();
    duplicate.child[7] = 4;
    assert!(is_invalid(build(&duplicate)));

    let mut missing = two_clusters();
    for column in [&mut missing.parent, &mut missing.child] {
        column.pop();
    }
    missing.lambda.pop();
    missing.child_size.pop();
    assert!(is_invalid(build(&missing)));

    let mut point_parent = two_clusters();
    point_parent.parent[2] = 1;
    assert!(is_invalid(build(&point_parent)));

    let mut cycle = two_clusters();
    cycle.parent[1] = 9;
    cycle.child[1] = 8;
    assert!(is_invalid(build(&cycle)));

    let mut decreasing = two_clusters();
    decreasing.parent[0] = 8;
    decreasing.child[0] = 7;
    assert!(is_invalid(build(&decreasing)));

    let mut unknown_parent = two_clusters();
    unknown_parent.parent[5] = 20;
    assert!(is_invalid(build(&unknown_parent)));

    let mut nan = two_clusters();
    nan.lambda[3] = f32::NAN;
    assert!(is_invalid(build(&nan)));

    let mut negative = two_clusters();
    negative.lambda[3] = -1.0;
    assert!(is_invalid(build(&negative)));

    let mut sized_point = two_clusters();
    sized_point.child_size[3] = 2.0;
    assert!(is_invalid(build(&sized_point)));

    let mut small_cluster = two_clusters();
    small_cluster.child_size[0] = 1.0;
    assert!(is_invalid(build(&small_cluster)));

    let tree = two_clusters();
    assert!(is_invalid(PredictionHierarchy::new(
        &tree,
        &[0, 0, 0, 1, 1, -2],
        &ids
    )));
    // Two labels anchored at the same cluster.
    assert!(is_invalid(PredictionHierarchy::new(
        &tree,
        &[0, 2, 0, 1, 1, 1],
        &ids
    )));
    assert!(is_invalid(PredictionHierarchy::new(
        &tree,
        &labels,
        &[0, 1, 3, 2, 4, 5]
    )));
    assert!(is_invalid(PredictionHierarchy::new(
        &tree,
        &labels,
        &[0, 1, 2, 3, 4, 6]
    )));
    assert!(is_invalid(PredictionHierarchy::new(&tree, &labels, &[])));
    // Label on a nonfinite raw row.
    assert!(is_invalid(PredictionHierarchy::new(
        &tree,
        &labels,
        &[0, 1, 2, 3, 4]
    )));
}

#[test]
fn validate_prediction_core_checks_full_vector() {
    let model = two_cluster_model();
    assert!(model.validate_prediction_core(&[0.0; 6]).is_ok());
    assert!(is_invalid(model.validate_prediction_core(&[0.0; 5])));
    assert!(is_numerical(model.validate_prediction_core(&[
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        f64::INFINITY
    ])));
}

#[test]
fn deep_hierarchy_ascends_iteratively() {
    let depth = 50_000usize;
    let raw_n = 2;
    let root = raw_n;
    let mut rows = Vec::with_capacity(depth + 2);
    for level in 0..depth {
        rows.push((root + level, root + level + 1, 1.0, 2.0));
    }
    let leaf = root + depth;
    rows.push((leaf, 0, 1.0, 1.0));
    rows.push((leaf, 1, 1.0, 1.0));
    let model = PredictionHierarchy::new(&tree(&rows), &[-1, -1], &identity(2)).unwrap();
    // Lambda 0.5: every birth 1 >= 0.5, ascend through all levels to the root.
    let p = model
        .predict_candidates(&[0, 1], &[0.0, 2.0], &[0.0; 2], 1)
        .unwrap();
    assert_eq!((p.label, p.connecting_point), (-1, 0));
}

mod oracle_fixture {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../hdbscan/tests/fixtures/hdbscan_prediction.json"
        );
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    fn float(value: &Value) -> f64 {
        match value {
            Value::String(text) => match text.as_str() {
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                "nan" => f64::NAN,
                other => panic!("bad float {other}"),
            },
            other => other.as_f64().unwrap(),
        }
    }

    fn floats(value: &Value) -> Vec<f64> {
        value.as_array().unwrap().iter().map(float).collect()
    }

    fn ints(value: &Value) -> Vec<i64> {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item.as_i64().unwrap())
            .collect()
    }

    fn raw_tree(value: &Value) -> CondensedTree {
        let usizes = |key: &str| -> Vec<usize> {
            ints(&value[key])
                .into_iter()
                .map(|item| item as usize)
                .collect()
        };
        let f32s = |key: &str| -> Vec<f32> {
            floats(&value[key])
                .into_iter()
                .map(|item| item as f32)
                .collect()
        };
        CondensedTree {
            parent: usizes("parent"),
            child: usizes("child"),
            lambda: f32s("lambda_val"),
            child_size: f32s("child_size"),
        }
    }

    fn labels(value: &Value) -> Vec<i32> {
        ints(value).into_iter().map(|item| item as i32).collect()
    }

    fn assert_rel(actual: f64, expected: f64, context: &str) {
        let tolerance = 1e-6 + 1e-5 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: {actual} != {expected}"
        );
    }

    fn inclusive_squared_rows(points: &[Vec<f64>], s: usize) -> Vec<f32> {
        let mut rows = Vec::with_capacity(points.len() * s);
        for query in points {
            let mut distances: Vec<f32> = points
                .iter()
                .map(|other| {
                    query
                        .iter()
                        .zip(other)
                        .map(|(a, b)| {
                            let delta = *a as f32 - *b as f32;
                            delta * delta
                        })
                        .sum()
                })
                .collect();
            distances.sort_by(f32::total_cmp);
            rows.extend_from_slice(&distances[..s]);
        }
        rows
    }

    fn matrix(value: &Value) -> Vec<Vec<f64>> {
        value.as_array().unwrap().iter().map(floats).collect()
    }

    #[test]
    fn controlled_reference_queries() {
        let data = fixture();
        let trees = &data["controlled"]["trees"];
        let queries = data["controlled"]["queries"].as_array().unwrap();
        assert!(!queries.is_empty());
        for query in queries {
            let name = query["name"].as_str().unwrap();
            let spec = &trees[query["tree"].as_str().unwrap()];
            let labels = labels(&spec["labels"]);
            let finite: Vec<u32> = (0..labels.len() as u32).collect();
            let model = PredictionHierarchy::new(&raw_tree(&spec["raw_tree"]), &labels, &finite)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let ids: Vec<i32> = ints(&query["indices"])
                .into_iter()
                .map(|id| id as i32)
                .collect();
            let prediction = model
                .predict_candidates(
                    &ids,
                    &floats(&query["distances"]),
                    &floats(&query["core_distances"]),
                    query["min_samples"].as_u64().unwrap() as usize,
                )
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let expected = &query["expected"];
            assert_eq!(
                prediction.label as i64,
                expected["label"].as_i64().unwrap(),
                "{name}"
            );
            assert_eq!(
                prediction.connecting_point as i64,
                expected["neighbor"].as_i64().unwrap(),
                "{name}"
            );
            assert_rel(prediction.probability, float(&expected["strength"]), name);
        }
    }

    #[test]
    fn core_rank_and_width_reference() {
        let data = fixture();
        let core = &data["core_rank"];
        let s = core["min_samples"].as_u64().unwrap() as usize;
        let rows = inclusive_squared_rows(&matrix(&core["train"]), s);
        let cores = prediction_core_from_squared_rows(&rows, s).unwrap();
        for (actual, expected) in cores
            .iter()
            .zip(floats(&core["prediction_core_inclusive_self"]))
        {
            assert_rel(*actual, expected, "core_rank");
        }
        let query_distances = floats(&core["query_candidate_distances"]);
        assert_eq!(query_distances[s], float(&core["query_core"]));

        for case in data["width"].as_array().unwrap() {
            let result = candidate_width(
                case["min_samples"].as_u64().unwrap() as usize,
                case["n_finite"].as_u64().unwrap() as usize,
            );
            match case["project_outcome"].as_str().unwrap() {
                "ok" => assert!(result.is_ok(), "{case}"),
                "ValueError" => assert!(is_invalid(result), "{case}"),
                other => panic!("unknown outcome {other}"),
            }
        }
    }

    #[test]
    fn fitted_reference_models() {
        let data = fixture();
        let fitted = data["fitted"].as_array().unwrap();
        assert!(!fitted.is_empty());
        for model_spec in fitted {
            let id = model_spec["id"].as_str().unwrap();
            let labels = labels(&model_spec["labels_compact"]);
            let compact: Vec<u32> = (0..labels.len() as u32).collect();
            let finite_index: Vec<u32> = ints(&model_spec["finite_index"])
                .into_iter()
                .map(|item| item as u32)
                .collect();
            let model =
                PredictionHierarchy::new(&raw_tree(&model_spec["raw_tree"]), &labels, &compact)
                    .unwrap_or_else(|error| panic!("{id}: {error}"));
            let expected = &model_spec["expected"];
            let no_cluster = model_spec["no_cluster_tree"].as_bool().unwrap();
            assert_eq!(model.no_cluster_tree(), no_cluster, "{id}");
            let queries = model_spec["queries_float32"].as_array().unwrap().len();
            if no_cluster {
                let shortcut = PredictionColumns::no_cluster_tree(queries);
                assert_eq!(shortcut.labels, labels_of(&expected["labels"]), "{id}");
                assert_eq!(
                    shortcut.connecting_points,
                    labels_of(&expected["connecting_points_raw"]),
                    "{id}"
                );
                continue;
            }
            let s = model_spec["resolved_min_samples"].as_u64().unwrap() as usize;
            let raw_train = matrix(&model_spec["train_float32"]);
            let train: Vec<Vec<f64>> = if raw_train.len() == finite_index.len() {
                raw_train
            } else {
                finite_index
                    .iter()
                    .map(|&raw| raw_train[raw as usize].clone())
                    .collect()
            };
            let cores =
                prediction_core_from_squared_rows(&inclusive_squared_rows(&train, s), s).unwrap();
            let reference_cores = floats(&model_spec["prediction_core_distances"]);
            for (actual, expected) in cores.iter().zip(&reference_cores) {
                assert_rel(*actual, *expected, id);
            }
            model.validate_prediction_core(&reference_cores).unwrap();
            let candidate_ids = model_spec["candidate_indices_compact"].as_array().unwrap();
            let candidate_distances = model_spec["candidate_distances"].as_array().unwrap();
            let expected_labels = ints(&expected["labels"]);
            let expected_compact = ints(&expected["connecting_points_compact"]);
            let expected_raw = ints(&expected["connecting_points_raw"]);
            let expected_strengths = floats(&expected["strengths"]);
            assert_eq!(candidate_ids.len(), queries, "{id}");
            for row in 0..queries {
                let ids: Vec<i32> = ints(&candidate_ids[row])
                    .into_iter()
                    .map(|item| item as i32)
                    .collect();
                let prediction = model
                    .predict_candidates(
                        &ids,
                        &floats(&candidate_distances[row]),
                        &reference_cores,
                        s,
                    )
                    .unwrap_or_else(|error| panic!("{id} row {row}: {error}"));
                assert_eq!(
                    prediction.label as i64, expected_labels[row],
                    "{id} row {row}"
                );
                assert_eq!(
                    prediction.connecting_point as i64, expected_compact[row],
                    "{id} row {row}"
                );
                assert_eq!(
                    finite_index[prediction.connecting_point as usize] as i64, expected_raw[row],
                    "{id} row {row}"
                );
                assert_rel(prediction.probability, expected_strengths[row], id);
            }
            // Raw-row model with the actual finite mapping and remapped hierarchy.
            if finite_index.len() != compact.len() || finite_index != compact {
                let raw_n = *finite_index.last().unwrap() as usize + 1;
                let mut raw_labels = vec![-1; raw_n];
                for (compact_row, &raw) in finite_index.iter().enumerate() {
                    raw_labels[raw as usize] = labels[compact_row];
                }
                let compact_tree = raw_tree(&model_spec["raw_tree"]);
                let outliers: Vec<u32> = (0..raw_n as u32)
                    .filter(|row| finite_index.binary_search(row).is_err())
                    .collect();
                let remapped =
                    hdbscan_core::finite::remap_condensed(&compact_tree, &finite_index, &outliers)
                        .unwrap();
                let raw_model =
                    PredictionHierarchy::new(&remapped, &raw_labels, &finite_index).unwrap();
                for row in 0..queries {
                    let ids: Vec<i32> = ints(&candidate_ids[row])
                        .into_iter()
                        .map(|item| item as i32)
                        .collect();
                    let prediction = raw_model
                        .predict_candidates(
                            &ids,
                            &floats(&candidate_distances[row]),
                            &reference_cores,
                            s,
                        )
                        .unwrap();
                    assert_eq!(
                        prediction.label as i64, expected_labels[row],
                        "{id} raw {row}"
                    );
                    assert_eq!(
                        prediction.connecting_point as i64, expected_raw[row],
                        "{id} raw {row}"
                    );
                }
            }
        }
    }

    fn labels_of(value: &Value) -> Vec<i32> {
        labels(value)
    }
}

#[test]
fn allocation_bound_covers_retained_bytes_and_overflows() {
    let model = two_cluster_model();
    assert_eq!(model.cluster_nodes(), 3);
    let bound = PredictionHierarchy::allocation_bytes_bound(6, 3).unwrap();
    assert!(bound >= model.retained_bytes());
    assert!(bound > PredictionHierarchy::allocation_bytes_bound(6, 0).unwrap());
    assert_eq!(PredictionHierarchy::allocation_bytes_bound(0, 0), Some(0));
    assert_eq!(
        PredictionHierarchy::allocation_bytes_bound(usize::MAX, 0),
        None
    );
    assert_eq!(
        PredictionHierarchy::allocation_bytes_bound(0, usize::MAX),
        None
    );
    assert_eq!(
        PredictionHierarchy::allocation_bytes_bound(usize::MAX / 8, usize::MAX / 64),
        None
    );
    let one = PredictionHierarchy::new(&CondensedTree::default(), &[-1], &[0]).unwrap();
    assert_eq!(one.cluster_nodes(), 0);
    assert!(PredictionHierarchy::allocation_bytes_bound(1, 0).unwrap() >= one.retained_bytes());
}

#[test]
fn constructed_hierarchy_capacity_is_exact_and_within_bound() {
    let model = two_cluster_model();
    let (n, clusters) = (model.n_finite(), model.cluster_nodes());
    assert_eq!((n, clusters), (6, 3));
    // Retained vectors are allocated at exact length, so real capacity meets the bound.
    let bound = PredictionHierarchy::allocation_bytes_bound(n, clusters).unwrap();
    assert_eq!(model.retained_bytes(), bound);
    assert_eq!(bound, 6 * 4 + 6 * 16 + 3 * 48);
}

#[test]
fn outlier_injected_hierarchy_keeps_finite_attachments_within_bound() {
    // Compact rows 0..6 map to raw [0,1,2,4,5,6]; raw row 3 is an outlier injected by
    // `remap_condensed` as a point row under the shifted root (raw 7).
    let finite_index = [0u32, 1, 2, 4, 5, 6];
    let remapped =
        hdbscan_core::finite::remap_condensed(&two_clusters(), &finite_index, &[3]).unwrap();
    let raw_labels = [0, 0, 0, -1, 1, 1, 1];
    let raw = PredictionHierarchy::new(&remapped, &raw_labels, &finite_index).unwrap();
    let plain = two_cluster_model();
    assert_eq!(raw.n_finite(), plain.n_finite());
    assert_eq!(raw.cluster_nodes(), plain.cluster_nodes());
    // Only finite attachments are retained, so capacity equals the finite-row bound.
    let bound = PredictionHierarchy::allocation_bytes_bound(6, 3).unwrap();
    assert_eq!(raw.retained_bytes(), bound);
    assert_eq!(raw.retained_bytes(), plain.retained_bytes());
    let cores = [0.0; 6];
    let queries: [(&[i32], &[f64]); 3] = [
        (&[0, 1], &[0.0, 0.5]),
        (&[3, 4], &[0.0, 0.25]),
        (&[5, 2], &[0.0, 1.0]),
    ];
    for (ids, distances) in queries {
        let expected = plain.predict_candidates(ids, distances, &cores, 1).unwrap();
        let actual = raw.predict_candidates(ids, distances, &cores, 1).unwrap();
        assert_eq!(actual.label, expected.label);
        assert_eq!(actual.probability, expected.probability);
        assert_eq!(
            actual.connecting_point,
            finite_index[expected.connecting_point as usize] as i32
        );
    }
}

#[test]
fn root_only_tree_with_outlier_rows_validates_finite_rows_only() {
    let finite_index = [0u32, 1];
    // Raw row 2 is an injected outlier attached to root 3.
    let ok = tree(&[(3, 0, 1.0, 1.0), (3, 2, 0.0, 1.0), (3, 1, 1.0, 1.0)]);
    let model = PredictionHierarchy::new(&ok, &[0, 0, -1], &finite_index).unwrap();
    assert!(model.no_cluster_tree());
    assert!(model.retained_bytes() <= PredictionHierarchy::allocation_bytes_bound(2, 0).unwrap());
    let missing = tree(&[(3, 0, 1.0, 1.0), (3, 2, 0.0, 1.0)]);
    assert!(is_invalid(PredictionHierarchy::new(
        &missing,
        &[0, 0, -1],
        &finite_index
    )));
}

#[test]
fn empty_condensed_tree_rejects_several_finite_rows() {
    let empty = tree(&[]);
    assert!(is_invalid(PredictionHierarchy::new(
        &empty,
        &[-1, -1],
        &identity(2)
    )));
    let single = PredictionHierarchy::new(&empty, &[-1], &identity(1)).unwrap();
    assert!(single.no_cluster_tree());
}

#[test]
fn root_only_tree_attachments_are_validated() {
    let ids = identity(3);
    let dup = tree(&[(3, 0, 1.0, 1.0), (3, 0, 1.0, 1.0), (3, 2, 1.0, 1.0)]);
    assert!(is_invalid(PredictionHierarchy::new(&dup, &[0; 3], &ids)));
    let missing = tree(&[(3, 0, 1.0, 1.0), (3, 1, 1.0, 1.0)]);
    assert!(is_invalid(PredictionHierarchy::new(
        &missing, &[0; 3], &ids
    )));
    let parents = tree(&[(3, 0, 1.0, 1.0), (4, 1, 1.0, 1.0), (3, 2, 1.0, 1.0)]);
    assert!(is_invalid(PredictionHierarchy::new(
        &parents, &[0; 3], &ids
    )));
    let ok = tree(&[(3, 0, 1.0, 1.0), (3, 1, 1.0, 1.0), (3, 2, 1.0, 1.0)]);
    assert!(
        PredictionHierarchy::new(&ok, &[0; 3], &ids)
            .unwrap()
            .no_cluster_tree()
    );
}
