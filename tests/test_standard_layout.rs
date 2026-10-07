use burn::backend::Dispatch;
use burn::tensor::{Device, Int, Tensor, TensorData};
use fast_umap::backend::StandardUmapOps;
use serde_json::Value;

const FIXTURE: &str = include_str!("fixtures/standard_umap_reference.json");

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

fn matrix(value: &Value) -> (Vec<f32>, usize, usize) {
    let rows = value.as_array().unwrap();
    let columns = rows[0].as_array().unwrap().len();
    (rows.iter().flat_map(floats).collect(), rows.len(), columns)
}

fn rng_rows(value: &Value) -> Vec<i64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|item| item.as_i64().unwrap())
        })
        .collect()
}

fn check_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 1e-5,
            "{label}[{index}]: {actual} != {expected}"
        );
    }
}

fn seed_for(case: &str) -> i64 {
    match case {
        "fit_unequal_weights_live_events" => 41,
        "bipartite_equal_raw_indices_positive_distance" => 73,
        "coincident_distinct_zero_distance" => 97,
        "same_index_zero_distance_skip" => 101,
        "clipped_gradient_and_frozen_transform_tail" => 19,
        _ => panic!("unknown frozen optimizer probe {case}"),
    }
}

fn initial_rng(case: &Value) -> (Vec<i64>, usize) {
    let (head, rows, _) = matrix(&case["head_initial"]);
    let seed = seed_for(case["name"].as_str().unwrap());
    let mut state = Vec::with_capacity(rows * 3);
    for row in 0..rows {
        let offset = f64::from(head[row * 2]).to_bits() as i64;
        for component in 0..3 {
            state.push(seed.wrapping_add(component as i64).wrapping_add(offset));
        }
    }
    (state, rows)
}

fn verify_frames(device: &Device, case: &Value, edge_count: usize, frames: &[Value]) {
    let (head_data, head_rows, dimensions) = matrix(&case["head_initial"]);
    let (tail_data, tail_rows, tail_dimensions) = matrix(&case["tail_initial"]);
    assert_eq!(dimensions, tail_dimensions);
    let edge_rows = case["edges_head_tail"].as_array().unwrap();
    let all_heads: Vec<i32> = edge_rows
        .iter()
        .map(|edge| edge[0].as_i64().unwrap() as i32)
        .collect();
    let all_tails: Vec<i32> = edge_rows
        .iter()
        .map(|edge| edge[1].as_i64().unwrap() as i32)
        .collect();
    let heads = &all_heads[..edge_count];
    let tails = &all_tails[..edge_count];
    let all_weights = floats(&case["positive_weights"]);
    let weights = &all_weights[..edge_count];
    let all_positive_intervals = floats(&case["epochs_per_sample"]);
    let positive_intervals = &all_positive_intervals[..edge_count];
    let rate = case["negative_sample_rate"].as_f64().unwrap() as f32;
    let negative_intervals: Vec<f32> = positive_intervals
        .iter()
        .map(|interval| interval / rate)
        .collect();
    let (mut rng_data, vertices) = initial_rng(case);
    let shared = case["head_tail_share_storage"].as_bool().unwrap();
    let mut expected_positive = positive_intervals.to_vec();
    let mut expected_negative = negative_intervals.clone();
    let mut expected_head = head_data;
    let mut expected_tail = tail_data;

    for frame in frames {
        let epoch = frame["epoch"].as_u64().unwrap() as u32;
        let before_positive = expected_positive.clone();
        let before_negative = expected_negative.clone();
        let mut positive_events = 0i32;
        let mut negative_events = 0i32;
        for edge in 0..edge_count {
            if weights[edge] > 0.0 && before_positive[edge] <= epoch as f32 {
                positive_events += 1;
                // Source promotes the schedule arithmetic to float64.
                let count = ((f64::from(epoch) - f64::from(before_negative[edge]))
                    / f64::from(negative_intervals[edge]))
                .trunc()
                .max(0.0) as i32;
                negative_events += count;
            }
        }

        let head = Tensor::<2>::from_data(
            TensorData::new(expected_head.clone(), [head_rows, dimensions]),
            device,
        );
        let tail = Tensor::<2>::from_data(
            TensorData::new(expected_tail.clone(), [tail_rows, dimensions]),
            device,
        );
        let edge_heads =
            Tensor::<1, Int>::from_data(TensorData::new(heads.to_vec(), [edge_count]), device);
        let edge_tails =
            Tensor::<1, Int>::from_data(TensorData::new(tails.to_vec(), [edge_count]), device);
        let weights_tensor =
            Tensor::<1>::from_data(TensorData::new(weights.to_vec(), [edge_count]), device);
        let positive_interval_tensor = Tensor::<1>::from_data(
            TensorData::new(positive_intervals.to_vec(), [edge_count]),
            device,
        );
        let next_positive_tensor = Tensor::<1>::from_data(
            TensorData::new(expected_positive.clone(), [edge_count]),
            device,
        );
        let negative_interval_tensor = Tensor::<1>::from_data(
            TensorData::new(negative_intervals.clone(), [edge_count]),
            device,
        );
        let next_negative_tensor = Tensor::<1>::from_data(
            TensorData::new(expected_negative.clone(), [edge_count]),
            device,
        );
        let rng_tensor = Tensor::<2, Int>::from_data(
            TensorData::new(rng_data.clone(), [vertices, 3]),
            (device, burn::tensor::DType::I64),
        );
        let (new_head, new_tail, next_positive, next_negative, new_rng, stats) =
            <Dispatch as StandardUmapOps>::optimize_epoch(
                head.into_dispatch(),
                tail.into_dispatch(),
                edge_heads.into_dispatch(),
                edge_tails.into_dispatch(),
                weights_tensor.into_dispatch(),
                positive_interval_tensor.into_dispatch(),
                next_positive_tensor.into_dispatch(),
                negative_interval_tensor.into_dispatch(),
                next_negative_tensor.into_dispatch(),
                rng_tensor.into_dispatch(),
                epoch,
                1.576_943_5,
                0.895_060_9,
                1.7,
                frame["alpha_used"].as_f64().unwrap() as f32,
                rate,
                case["move_other"].as_bool().unwrap(),
                shared,
            );
        let new_head: Tensor<2> = Tensor::from_dispatch(new_head);
        let new_tail: Tensor<2> = Tensor::from_dispatch(new_tail);
        let next_positive: Tensor<1> = Tensor::from_dispatch(next_positive);
        let next_negative: Tensor<1> = Tensor::from_dispatch(next_negative);
        let new_rng: Tensor<2, Int> = Tensor::from_dispatch(new_rng);
        let stats: Tensor<1, Int> = Tensor::from_dispatch(stats);
        expected_head = new_head.to_data().try_to_vec::<f32>().unwrap();
        expected_tail = new_tail.to_data().try_to_vec::<f32>().unwrap();
        expected_positive = next_positive.to_data().try_to_vec::<f32>().unwrap();
        expected_negative = next_negative.to_data().try_to_vec::<f32>().unwrap();
        rng_data = new_rng.to_data().try_to_vec::<i64>().unwrap();

        let (wanted_head, _, _) = matrix(&frame["head_embedding"]);
        let (wanted_tail, _, _) = matrix(&frame["tail_embedding"]);
        check_close(&expected_head, &wanted_head, "epoch head coordinates");
        check_close(&expected_tail, &wanted_tail, "epoch tail coordinates");
        check_close(
            &expected_positive,
            &floats(&frame["epoch_of_next_sample"]),
            "positive schedule",
        );
        check_close(
            &expected_negative,
            &floats(&frame["epoch_of_next_negative_sample"]),
            "negative schedule",
        );
        assert_eq!(
            rng_data,
            rng_rows(&frame["rng_state_per_sample"]),
            "per-head Tausworthe state"
        );
        assert_eq!(
            stats.to_data().try_to_vec::<i32>().unwrap(),
            [positive_events, negative_events],
            "epoch event counts"
        );
    }
}

fn verify_fixture(device: &Device) {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in fixture["optimizer_probes"].as_array().unwrap() {
        verify_frames(
            device,
            case,
            case["edges_head_tail"].as_array().unwrap().len(),
            case["epochs"].as_array().unwrap(),
        );
        for prefix in case["ordered_event_prefixes"].as_array().unwrap() {
            let edge_count = prefix["through_edge_index"].as_u64().unwrap() as usize + 1;
            verify_frames(
                device,
                case,
                edge_count,
                prefix["epochs"].as_array().unwrap(),
            );
        }
    }
}

#[test]
fn standard_layout_matches_frozen_optimizer_epochs_and_ordered_prefixes_on_flex() {
    verify_fixture(&Device::flex());
}

fn verify_zero_controls_and_repeats(device: &Device) {
    let case: Value = serde_json::from_str(FIXTURE).unwrap();
    let probe = &case["optimizer_probes"][0];
    let frames = probe["epochs"].as_array().unwrap();
    verify_frames(device, probe, 2, frames);
    verify_frames(device, probe, 2, frames);

    let head_data = vec![0.0f32, 0.0, 1.0, 0.0];
    let tail_data = vec![0.0f32, 0.0, 1.0, 0.0];
    let rng_data = vec![41i64, 42, 43, 44, 45, 46];
    let run = |edge_count: usize, alpha: f32, gamma: f32, rate: f32| {
        let head = Tensor::<2>::from_data(TensorData::new(head_data.clone(), [2, 2]), device);
        let tail = Tensor::<2>::from_data(TensorData::new(tail_data.clone(), [2, 2]), device);
        let heads = Tensor::<1, Int>::from_data(
            TensorData::new(vec![0i32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let tails = Tensor::<1, Int>::from_data(
            TensorData::new(vec![1i32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let weights = Tensor::<1>::from_data(
            TensorData::new(vec![1.0f32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let positive = Tensor::<1>::from_data(
            TensorData::new(vec![1.0f32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let next_positive = positive.clone();
        let negative = Tensor::<1>::from_data(
            TensorData::new(vec![1.0f32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let next_negative = Tensor::<1>::from_data(
            TensorData::new(vec![0.0f32][..edge_count].to_vec(), [edge_count]),
            device,
        );
        let rng = Tensor::<2, Int>::from_data(
            TensorData::new(rng_data.clone(), [2, 3]),
            (device, burn::tensor::DType::I64),
        );
        <Dispatch as StandardUmapOps>::optimize_epoch(
            head.into_dispatch(),
            tail.into_dispatch(),
            heads.into_dispatch(),
            tails.into_dispatch(),
            weights.into_dispatch(),
            positive.into_dispatch(),
            next_positive.into_dispatch(),
            negative.into_dispatch(),
            next_negative.into_dispatch(),
            rng.into_dispatch(),
            1,
            1.0,
            1.0,
            gamma,
            alpha,
            rate,
            false,
            false,
        )
    };

    let (head, tail, next_pos, next_neg, rng, stats) = run(1, 0.0, 1.0, 0.0);
    let head: Tensor<2> = Tensor::from_dispatch(head);
    let tail: Tensor<2> = Tensor::from_dispatch(tail);
    let next_pos: Tensor<1> = Tensor::from_dispatch(next_pos);
    let next_neg: Tensor<1> = Tensor::from_dispatch(next_neg);
    let rng: Tensor<2, Int> = Tensor::from_dispatch(rng);
    let stats: Tensor<1, Int> = Tensor::from_dispatch(stats);
    assert_eq!(head.to_data().try_to_vec::<f32>().unwrap(), head_data);
    assert_eq!(tail.to_data().try_to_vec::<f32>().unwrap(), tail_data);
    assert_eq!(next_pos.to_data().try_to_vec::<f32>().unwrap(), [2.0]);
    assert_eq!(next_neg.to_data().try_to_vec::<f32>().unwrap(), [0.0]);
    assert_eq!(rng.to_data().try_to_vec::<i64>().unwrap(), rng_data);
    assert_eq!(stats.to_data().try_to_vec::<i32>().unwrap(), [1, 0]);

    let (expected_head, expected_tail, _, _, _, _) = run(1, 0.8, 0.0, 0.0);
    let (head, tail, _, _, rng, stats) = run(1, 0.8, 0.0, 1.0);
    let head: Tensor<2> = Tensor::from_dispatch(head);
    let tail: Tensor<2> = Tensor::from_dispatch(tail);
    let rng: Tensor<2, Int> = Tensor::from_dispatch(rng);
    let stats: Tensor<1, Int> = Tensor::from_dispatch(stats);
    assert_eq!(
        head.to_data().try_to_vec::<f32>().unwrap(),
        Tensor::<2>::from_dispatch(expected_head)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
    );
    assert_eq!(
        tail.to_data().try_to_vec::<f32>().unwrap(),
        Tensor::<2>::from_dispatch(expected_tail)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
    );
    assert_ne!(rng.to_data().try_to_vec::<i64>().unwrap(), rng_data);
    assert_eq!(stats.to_data().try_to_vec::<i32>().unwrap(), [1, 1]);
}

#[test]
fn standard_layout_zero_controls_and_repeated_runs_are_deterministic() {
    verify_zero_controls_and_repeats(&Device::flex());
}

#[cfg(feature = "cuda")]
#[test]
fn standard_layout_matches_frozen_optimizer_epochs_on_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    verify_fixture(&device);
    verify_zero_controls_and_repeats(&device);
}

#[cfg(all(not(feature = "cuda"), test))]
#[test]
fn cuda_layout_is_required_when_requested_without_cuda_feature() {
    if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
        panic!("FAST_UMAP_REQUIRE_CUDA is set, but CUDA support is not compiled");
    }
}
