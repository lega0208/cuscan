use burn::backend::Dispatch;
use burn::backend::tensor::IntTensor;
use burn::tensor::{Device, Int, Tensor, TensorData};
use serde_json::Value;
use umap::backend::StandardUmapOps;
use umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

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

fn tausworthe_step(state: &mut [i64]) -> i32 {
    let mut next = [0i64; 3];
    next[0] = (((state[0] & 4_294_967_294i64).wrapping_shl(12)) & 0xffff_ffff)
        ^ ((((state[0].wrapping_shl(13)) & 0xffff_ffff) ^ state[0]) >> 19);
    next[1] = (((state[1] & 4_294_967_288i64).wrapping_shl(4)) & 0xffff_ffff)
        ^ ((((state[1].wrapping_shl(2)) & 0xffff_ffff) ^ state[1]) >> 25);
    next[2] = (((state[2] & 4_294_967_280i64).wrapping_shl(17)) & 0xffff_ffff)
        ^ ((((state[2].wrapping_shl(3)) & 0xffff_ffff) ^ state[2]) >> 11);
    state.copy_from_slice(&next);
    (next[0] ^ next[1] ^ next[2]) as i32
}

fn single_head_reference(
    head: &mut [f32],
    fixed_tail: &[f32],
    dimensions: usize,
    next_positive: &mut f64,
    next_negative: &mut f64,
    rng: &mut [i64],
    epoch: u32,
) {
    if *next_positive <= f64::from(epoch) {
        *next_positive += 1.0;
        let a = 1.576_943_5f32;
        let b = 0.895_060_9f32;
        let alpha = 0.25f32;
        let mut distance = 0.0f32;
        for column in 0..dimensions {
            let difference = head[column] - fixed_tail[dimensions + column];
            distance += difference * difference;
        }
        if distance > 0.0 {
            let coefficient =
                -2.0f32 * a * b * distance.powf(b - 1.0f32) / (a * distance.powf(b) + 1.0f32);
            for column in 0..dimensions {
                let difference = head[column] - fixed_tail[dimensions + column];
                let gradient = (coefficient * difference).clamp(-4.0, 4.0);
                head[column] += gradient * alpha;
            }
        }
        let count = (f64::from(epoch) - *next_negative) as i32;
        *next_negative += f64::from(count);
        for _ in 0..count {
            let remainder = tausworthe_step(rng) % 2;
            let sampled = (if remainder < 0 {
                remainder + 2
            } else {
                remainder
            }) as usize;
            let mut distance = 0.0f32;
            for column in 0..dimensions {
                let difference = head[column] - fixed_tail[sampled * dimensions + column];
                distance += difference * difference;
            }
            if distance > 0.0 {
                let coefficient =
                    2.0f32 * 1.7 * b / ((0.001f32 + distance) * (a * distance.powf(b) + 1.0f32));
                for column in 0..dimensions {
                    let difference = head[column] - fixed_tail[sampled * dimensions + column];
                    let gradient = (coefficient * difference).clamp(-4.0, 4.0);
                    head[column] += gradient * alpha;
                }
            }
        }
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

type Csr = (IntTensor<Dispatch>, IntTensor<Dispatch>);

/// Positive-weight edge ids grouped by head.
fn csr(device: &Device, heads: &[i32], weights: &[f32], head_rows: usize) -> Csr {
    let mut offsets = vec![0i32; head_rows + 1];
    let mut ids = Vec::new();
    for row in 0..head_rows {
        for (edge, (&endpoint, &weight)) in heads.iter().zip(weights).enumerate() {
            if weight > 0.0 && endpoint as usize == row {
                ids.push(edge as i32);
            }
        }
        offsets[row + 1] = ids.len() as i32;
    }
    if ids.is_empty() {
        ids.push(0);
    }
    let len = ids.len();
    (
        Tensor::<1, Int>::from_data(TensorData::new(offsets, [head_rows + 1]), device)
            .into_dispatch(),
        Tensor::<1, Int>::from_data(TensorData::new(ids, [len]), device).into_dispatch(),
    )
}

fn verify_frames(
    device: &Device,
    case: &Value,
    edge_count: usize,
    frames: &[Value],
    reference_order: bool,
) {
    let (head_data, head_rows, dimensions) = matrix(&case["head_initial"]);
    let (tail_data, tail_rows, tail_dimensions) = matrix(&case["tail_initial"]);
    let shared = case["move_other"].as_bool().unwrap();
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
    let mut expected_positive = positive_intervals.to_vec();
    let mut expected_negative = negative_intervals.clone();
    let mut expected_head = head_data;
    let mut expected_tail = tail_data;

    for frame in frames {
        let epoch = frame["epoch"].as_u64().unwrap() as u32;
        let (head_offsets, head_edges) = csr(device, heads, weights, head_rows);
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
        let retained_head = head.clone();
        let retained_tail = tail.clone();
        let (new_head, new_tail, next_positive, next_negative, new_rng) =
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
                head_offsets,
                head_edges,
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
        let actual_head = new_head.to_data().try_to_vec::<f32>().unwrap();
        let actual_tail = new_tail.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(
            retained_head.to_data().try_to_vec::<f32>().unwrap(),
            expected_head,
            "retained head observer must not be mutated"
        );
        assert_eq!(
            retained_tail.to_data().try_to_vec::<f32>().unwrap(),
            expected_tail,
            "retained tail observer must not be mutated"
        );
        expected_positive = next_positive.to_data().try_to_vec::<f32>().unwrap();
        expected_negative = next_negative.to_data().try_to_vec::<f32>().unwrap();
        rng_data = new_rng.to_data().try_to_vec::<i64>().unwrap();

        let (wanted_head, _, _) = matrix(&frame["head_embedding"]);
        let (wanted_tail, _, _) = matrix(&frame["tail_embedding"]);
        if reference_order {
            check_close(&actual_head, &wanted_head, "epoch head coordinates");
            check_close(&actual_tail, &wanted_tail, "epoch tail coordinates");
        } else {
            // Parallel vertex-owner order may differ from serial coordinates.
            assert!(
                actual_head
                    .iter()
                    .chain(&actual_tail)
                    .all(|v| v.is_finite()),
                "epoch coordinates must be finite"
            );
            assert_eq!(actual_head.len(), wanted_head.len());
            assert!(actual_tail.iter().all(|v| v.is_finite()));
            if shared {
                assert_eq!(actual_head, actual_tail, "fit outputs must share positions");
            } else {
                assert_eq!(actual_tail, expected_tail, "transform tail must stay fixed");
            }
        }
        // Continue each frame from the reference state so clocks/RNG are
        // checked per epoch independent of coordinate order.
        expected_head = wanted_head;
        expected_tail = wanted_tail;
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
    }
}

fn verify_fixture(device: &Device, reference_order: bool) {
    let fixture: Value = serde_json::from_str(FIXTURE).unwrap();
    for case in fixture["optimizer_probes"].as_array().unwrap() {
        verify_frames(
            device,
            case,
            case["edges_head_tail"].as_array().unwrap().len(),
            case["epochs"].as_array().unwrap(),
            reference_order,
        );
        for prefix in case["ordered_event_prefixes"].as_array().unwrap() {
            let edge_count = prefix["through_edge_index"].as_u64().unwrap() as usize + 1;
            verify_frames(
                device,
                case,
                edge_count,
                prefix["epochs"].as_array().unwrap(),
                reference_order,
            );
        }
    }
}

#[test]
fn standard_layout_matches_frozen_optimizer_epochs_and_ordered_prefixes_on_cpu() {
    verify_fixture(&Device::cpu(), false);
}

/// Exercises the fused Dispatch backend with production dtypes: f32
/// coordinates/atomics, f64 clocks, i64 RNG `[head_rows, 3]`, and i32 indices.
#[test]
fn standard_layout_single_head_f64_reference_continues_actual_state() {
    let device = Device::cpu();
    for dimensions in [1usize, 2, 3, 5] {
        let mut expected_head = (0..dimensions)
            .map(|column| 0.35f32 + column as f32 * 0.07)
            .collect::<Vec<_>>();
        let expected_tail = (0..2 * dimensions)
            .map(|index| -0.2f32 + index as f32 * 0.11)
            .collect::<Vec<_>>();
        let mut expected_positive = 0.0f64;
        let mut expected_negative = 0.0f64;
        // The first Tausworthe draw has a negative signed i32 result, so the
        // reference also exercises the kernel's nonnegative-index fixup.
        let mut expected_rng = vec![1i64 << 19, 0, 0];
        let mut actual_positive = 0.0f64;
        let mut actual_negative = 0.0f64;
        let mut actual_rng = expected_rng.clone();
        let mut head = Tensor::<2>::from_data(
            TensorData::new(expected_head.clone(), [1, dimensions]),
            &device,
        );
        let mut tail = Tensor::<2>::from_data(
            TensorData::new(expected_tail.clone(), [2, dimensions]),
            &device,
        );
        let (observed_offsets, observed_edges) = <Dispatch as StandardUmapOps>::head_csr(
            Tensor::<1, Int>::from_data(TensorData::new(vec![0i32], [1]), &device).into_dispatch(),
            Tensor::<1>::from_data(TensorData::new(vec![1.0f32], [1]), &device).into_dispatch(),
            1,
        );
        let observed_offsets: Tensor<1, Int> = Tensor::from_dispatch(observed_offsets);
        let observed_edges: Tensor<1, Int> = Tensor::from_dispatch(observed_edges);
        assert_eq!(
            observed_offsets.to_data().try_to_vec::<i32>().unwrap(),
            [0, 1]
        );
        assert_eq!(observed_edges.to_data().try_to_vec::<i32>().unwrap(), [0]);
        let (head_offsets, head_edges) = csr(&device, &[0], &[1.0], 1);
        for epoch in 0..2 {
            let head_before = expected_head.clone();
            let positive = Tensor::<1>::from_data(
                TensorData::new(vec![1.0f64], [1]),
                (&device, burn::tensor::DType::F64),
            );
            let next_positive = Tensor::<1>::from_data(
                TensorData::new(vec![actual_positive], [1]),
                (&device, burn::tensor::DType::F64),
            );
            let negative = Tensor::<1>::from_data(
                TensorData::new(vec![1.0f64], [1]),
                (&device, burn::tensor::DType::F64),
            );
            let next_negative = Tensor::<1>::from_data(
                TensorData::new(vec![actual_negative], [1]),
                (&device, burn::tensor::DType::F64),
            );
            let rng = Tensor::<2, Int>::from_data(
                TensorData::new(actual_rng.clone(), [1, 3]),
                (&device, burn::tensor::DType::I64),
            );
            let heads = Tensor::<1, Int>::from_data(TensorData::new(vec![0i32], [1]), &device);
            let tails = Tensor::<1, Int>::from_data(TensorData::new(vec![1i32], [1]), &device);
            let weights = Tensor::<1>::from_data(TensorData::new(vec![1.0f32], [1]), &device);
            let retained_head = head.clone();
            let retained_tail = tail.clone();
            let (next_head, next_tail, next_positive, next_negative, next_rng) =
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
                    head_offsets.clone(),
                    head_edges.clone(),
                    epoch,
                    1.576_943_5,
                    0.895_060_9,
                    1.7,
                    0.25,
                    1.0,
                    false,
                    false,
                );
            head = Tensor::from_dispatch(next_head);
            tail = Tensor::from_dispatch(next_tail);
            let next_positive: Tensor<1> = Tensor::from_dispatch(next_positive);
            let next_negative: Tensor<1> = Tensor::from_dispatch(next_negative);
            let next_rng: Tensor<2, Int> = Tensor::from_dispatch(next_rng);
            actual_positive = next_positive.to_data().try_to_vec::<f64>().unwrap()[0];
            actual_negative = next_negative.to_data().try_to_vec::<f64>().unwrap()[0];
            actual_rng = next_rng.to_data().try_to_vec::<i64>().unwrap();
            single_head_reference(
                &mut expected_head,
                &expected_tail,
                dimensions,
                &mut expected_positive,
                &mut expected_negative,
                &mut expected_rng,
                epoch,
            );
            let actual_head = head.to_data().try_to_vec::<f32>().unwrap();
            // A single active head serializes all coordinate writes; the host
            // reference preserves f32 operation and edge order. Keep the
            // tolerance at 1e-5 for backend `powf` rounding rather than
            // claiming a bitwise contract.
            check_close(
                &actual_head,
                &expected_head,
                "single-head reference coordinates",
            );
            assert_eq!(
                retained_head.to_data().try_to_vec::<f32>().unwrap(),
                head_before
            );
            assert_eq!(tail.to_data().try_to_vec::<f32>().unwrap(), expected_tail);
            assert_eq!(
                retained_tail.to_data().try_to_vec::<f32>().unwrap(),
                expected_tail
            );
            assert_eq!(actual_positive, expected_positive);
            assert_eq!(actual_negative, expected_negative);
            assert_eq!(actual_rng, expected_rng);
        }
    }
}

#[test]
fn standard_layout_runs_estimator_produced_f64_clock_state() {
    let data = (0..12)
        .map(|row| vec![row as f32 * 0.1, (row as f32 * 0.4).sin()])
        .collect::<Vec<_>>();
    let config = UMAPConfig {
        n_neighbors: 4,
        n_epochs: Some(2),
        init: Init::Random,
        random_state: Some(31),
        ..Default::default()
    };
    let baseline = UMAP::new(
        UMAPConfig {
            n_epochs: Some(0),
            ..config.clone()
        },
        BackendChoice::Cpu,
        NeighborSearch::Exact,
    )
    .fit(&data)
    .unwrap();
    let fitted = UMAP::new(config, BackendChoice::Cpu, NeighborSearch::Exact)
        .fit(&data)
        .unwrap();
    assert_eq!(fitted.embedding().len(), data.len());
    assert_ne!(fitted.embedding(), baseline.embedding());
    assert!(
        fitted
            .embedding()
            .iter()
            .flatten()
            .all(|coordinate| coordinate.is_finite())
    );
}

#[test]
fn standard_layout_plain_cube_backend_active_update_preserves_observers() {
    use burn::backend::ops::{FloatTensorOps, IntTensorOps};
    use burn_cubecl::{CubeBackend, CubeDevice, ops::into_data_sync, tensor::CubeTensor};

    let device = CubeDevice::Cpu(Default::default());
    let make_float = |values: Vec<f32>, shape: [usize; 2]| {
        CubeBackend::float_from_data(TensorData::new(values, shape), &device)
    };
    let make_float_1d = |values: Vec<f32>| {
        CubeBackend::float_from_data(TensorData::new(values.clone(), [values.len()]), &device)
    };
    let make_i32 = |values: Vec<i32>, shape: [usize; 1]| {
        CubeBackend::int_from_data(TensorData::new(values, shape), &device)
    };
    let head_data = vec![0.0f32, 1.0];
    let tail_data = vec![1.0f32, 0.0];
    let head = make_float(head_data.clone(), [1, 2]);
    let tail = make_float(tail_data.clone(), [1, 2]);
    let retained_head = head.clone();
    let retained_tail = tail.clone();
    let (new_head, new_tail, positive_clock, negative_clock, rng) =
        <CubeBackend as StandardUmapOps>::optimize_epoch(
            head,
            tail,
            make_i32(vec![0], [1]),
            make_i32(vec![0], [1]),
            make_float_1d(vec![1.0]),
            CubeBackend::float_from_data(TensorData::new(vec![1.0f64], [1]), &device),
            CubeBackend::float_from_data(TensorData::new(vec![0.0f64], [1]), &device),
            CubeBackend::float_from_data(TensorData::new(vec![1.0f64], [1]), &device),
            CubeBackend::float_from_data(TensorData::new(vec![0.0f64], [1]), &device),
            CubeBackend::int_from_data(TensorData::new(vec![11i64, 12, 13], [1, 3]), &device),
            make_i32(vec![0, 1], [2]),
            make_i32(vec![0], [1]),
            0,
            1.576_943_5,
            0.895_060_9,
            1.7,
            0.25,
            0.0,
            false,
            false,
        );
    let output = |tensor: CubeTensor| into_data_sync(tensor);
    let new_head = output(new_head);
    let new_tail = output(new_tail);
    let positive_clock = output(positive_clock);
    let negative_clock = output(negative_clock);
    let rng = output(rng);
    assert_ne!(new_head, TensorData::new(head_data.clone(), [1, 2]));
    assert_eq!(new_tail, TensorData::new(tail_data.clone(), [1, 2]));
    assert_eq!(output(retained_head), TensorData::new(head_data, [1, 2]));
    assert_eq!(output(retained_tail), TensorData::new(tail_data, [1, 2]));
    assert_eq!(positive_clock, TensorData::new(vec![1.0f64], [1]));
    assert_eq!(negative_clock, TensorData::new(vec![0.0f64], [1]));
    assert_eq!(rng, TensorData::new(vec![11i64, 12, 13], [1, 3]));
}

fn verify_zero_controls_and_repeats(device: &Device, _reference_order: bool) {
    let case: Value = serde_json::from_str(FIXTURE).unwrap();
    let probe = &case["optimizer_probes"][0];
    let frames = probe["epochs"].as_array().unwrap();
    verify_frames(device, probe, 2, frames, _reference_order);
    verify_frames(device, probe, 2, frames, _reference_order);

    let head_data = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let tail_data = vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0];
    let rng_data = vec![41i64, 42, 43, 44, 45, 46];
    let run = |edge_count: usize, alpha: f32, gamma: f32, rate: f32| {
        let empty_heads = vec![0i32; edge_count];
        let empty_weights = vec![1.0f32; edge_count];
        let (head_offsets, head_edges) = csr(device, &empty_heads, &empty_weights, 2);
        let head = Tensor::<2>::from_data(TensorData::new(head_data.clone(), [2, 3]), device);
        let tail = Tensor::<2>::from_data(TensorData::new(tail_data.clone(), [2, 3]), device);
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
            TensorData::new(vec![1.0f64][..edge_count].to_vec(), [edge_count]),
            (device, burn::tensor::DType::F64),
        );
        let next_positive = positive.clone();
        let negative = Tensor::<1>::from_data(
            TensorData::new(vec![1.0f64][..edge_count].to_vec(), [edge_count]),
            (device, burn::tensor::DType::F64),
        );
        let next_negative = Tensor::<1>::from_data(
            TensorData::new(vec![0.0f64][..edge_count].to_vec(), [edge_count]),
            (device, burn::tensor::DType::F64),
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
            head_offsets,
            head_edges,
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

    let (head, tail, next_pos, next_neg, rng) = run(1, 0.0, 1.0, 0.0);
    let head: Tensor<2> = Tensor::from_dispatch(head);
    let tail: Tensor<2> = Tensor::from_dispatch(tail);
    let next_pos: Tensor<1> = Tensor::from_dispatch(next_pos);
    let next_neg: Tensor<1> = Tensor::from_dispatch(next_neg);
    let rng: Tensor<2, Int> = Tensor::from_dispatch(rng);
    let head_data_after = head.to_data().try_to_vec::<f32>().unwrap();
    let tail_data_after = tail.to_data().try_to_vec::<f32>().unwrap();
    assert_eq!(head_data_after, head_data);
    assert_eq!(tail_data_after, tail_data);
    assert_eq!(next_pos.to_data().try_to_vec::<f64>().unwrap(), [2.0]);
    assert_eq!(next_neg.to_data().try_to_vec::<f64>().unwrap(), [0.0]);
    assert_eq!(rng.to_data().try_to_vec::<i64>().unwrap(), rng_data);

    let (head, tail, next_pos, next_neg, rng) = run(0, 0.8, 1.0, 1.0);
    let head: Tensor<2> = Tensor::from_dispatch(head);
    let tail: Tensor<2> = Tensor::from_dispatch(tail);
    let next_pos: Tensor<1> = Tensor::from_dispatch(next_pos);
    let next_neg: Tensor<1> = Tensor::from_dispatch(next_neg);
    let rng: Tensor<2, Int> = Tensor::from_dispatch(rng);
    assert_eq!(head.to_data().try_to_vec::<f32>().unwrap(), head_data);
    assert_eq!(tail.to_data().try_to_vec::<f32>().unwrap(), tail_data);
    assert!(next_pos.to_data().try_to_vec::<f64>().unwrap().is_empty());
    assert!(next_neg.to_data().try_to_vec::<f64>().unwrap().is_empty());
    assert_eq!(rng.to_data().try_to_vec::<i64>().unwrap(), rng_data);

    let (expected_head, expected_tail, _, _, _) = run(1, 0.8, 0.0, 0.0);
    let (head, tail, _, _, rng) = run(1, 0.8, 0.0, 1.0);
    let head: Tensor<2> = Tensor::from_dispatch(head);
    let tail: Tensor<2> = Tensor::from_dispatch(tail);
    let rng: Tensor<2, Int> = Tensor::from_dispatch(rng);
    let head_data_after = head.to_data().try_to_vec::<f32>().unwrap();
    let tail_data_after = tail.to_data().try_to_vec::<f32>().unwrap();
    assert_eq!(
        head_data_after,
        Tensor::<2>::from_dispatch(expected_head)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
    );
    assert_eq!(
        tail_data_after,
        Tensor::<2>::from_dispatch(expected_tail)
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
    );
    assert_ne!(rng.to_data().try_to_vec::<i64>().unwrap(), rng_data);
}

#[test]
fn standard_layout_zero_controls_and_repeated_fixture_runs_remain_valid() {
    verify_zero_controls_and_repeats(&Device::cpu(), false);
}

#[test]
fn standard_layout_matches_frozen_optimizer_epochs_on_cuda() {
    let Some(device) = umap::cuda_backend::device(0) else {
        if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() {
            panic!("CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    verify_fixture(&device, false);
    verify_zero_controls_and_repeats(&device, false);
}
