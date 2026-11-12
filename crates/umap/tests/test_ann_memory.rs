//! ANN admission and budget-invariant output, independent of layout nondeterminism.
use burn::tensor::{Device, Tensor, TensorData};
use cuscan_common::{ann_preflight, memory::MemoryBudget, neighbors::approximate_neighbors};
use umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig, UmapError};

fn fixture(rows: usize, features: usize) -> Vec<f32> {
    (0..rows * features)
        .map(|i| ((i * 73 + i / features * 19) % 1009) as f32 / 1009.0)
        .collect()
}

fn limit_invariance(device: Device) {
    let (rows, features, k) = (512, 8, 14);
    let input = Tensor::<2>::from_data(
        TensorData::new(fixture(rows, features), [rows, features]),
        &device,
    );
    let unlimited = MemoryBudget {
        total_bytes: 2 << 30,
        free_bytes: 2 << 30,
        limit_bytes: None,
    };
    let limited = MemoryBudget {
        limit_bytes: Some(128 << 20),
        ..unlimited
    };
    let full_plan = ann_preflight(&device, rows, features, k, &unlimited).unwrap();
    let small_plan = ann_preflight(&device, rows, features, k, &limited).unwrap();
    assert!(small_plan.tile_slots < full_plan.tile_slots);
    let (expected_ids, expected_distances) =
        approximate_neighbors(input.clone(), &full_plan, 42, 9, false).unwrap();
    let (actual_ids, actual_distances) =
        approximate_neighbors(input, &small_plan, 42, 9, false).unwrap();
    assert_eq!(
        actual_ids.into_data().try_to_vec::<i32>().unwrap(),
        expected_ids.into_data().try_to_vec::<i32>().unwrap()
    );
    let bits = |tensor: Tensor<2>| {
        tensor
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    };
    assert_eq!(bits(actual_distances), bits(expected_distances));
}

fn fit_rejection(backend: BackendChoice) {
    let values = fixture(128, 8);
    let rows: Vec<_> = values
        .as_chunks::<8>()
        .0
        .iter()
        .map(|row| row.to_vec())
        .collect();
    let result = UMAP::new(
        UMAPConfig {
            memory_limit_bytes: Some(1),
            random_state: Some(42),
            init: Init::Random,
            n_epochs: Some(2),
            ..Default::default()
        },
        backend,
        NeighborSearch::Approximate,
    )
    .fit(&rows);
    match result {
        Err(UmapError::Memory(message)) => {
            assert!(message.contains("ANN needs"), "{message}");
            assert!(message.contains("available"), "{message}");
            assert!(message.contains("largest buffers"), "{message}");
        }
        other => panic!("expected typed ANN memory rejection, got {other:?}"),
    }
}

#[test]
fn memory_admission_and_limit_invariance_cpu() {
    fit_rejection(BackendChoice::Cpu);
    limit_invariance(Device::cpu());
}

#[test]
fn memory_admission_and_limit_invariance_cuda() {
    if cuscan_common::cuda_backend::try_device_count().expect("CUDA discovery failed") == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA execution required but unavailable"
        );
        return;
    }
    fit_rejection(BackendChoice::Cuda { device_index: 0 });
    limit_invariance(Device::cuda(0));
}

fn large_tile_invariance(device: Device) {
    let (rows, features, k) = (40_000, 32, 14);
    let input = Tensor::<2>::from_data(
        TensorData::new(fixture(rows, features), [rows, features]),
        &device,
    );
    let budget = MemoryBudget {
        total_bytes: 2 << 30,
        free_bytes: 2 << 30,
        limit_bytes: None,
    };
    let fixed = ann_preflight(&device, rows, features, k, &budget)
        .unwrap()
        .memory
        .fixed_peak_bytes;
    let mut expected = None;
    for slots in [1 << 20, 1 << 22, 1 << 24] {
        // All six staging lengths are alignment multiples at these tile sizes.
        // Cap usable memory at exactly fixed + 32 bytes per pair slot.
        let limited = MemoryBudget {
            limit_bytes: Some(fixed + (64 << 20) + 32 * slots as u64),
            ..budget
        };
        let plan = ann_preflight(&device, rows, features, k, &limited).unwrap();
        assert_eq!(plan.tile_slots, slots);
        let (ids, distances) = approximate_neighbors(input.clone(), &plan, 42, 15, false).unwrap();
        let result = (
            ids.into_data().try_to_vec::<i32>().unwrap(),
            distances
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>(),
        );
        if let Some(expected) = &expected {
            assert_eq!(
                &result, expected,
                "tile size {slots} changed ANN output bits"
            );
        } else {
            expected = Some(result);
        }
    }
}

#[test]
fn permanent_large_tile_invariance_cpu() {
    large_tile_invariance(Device::cpu());
}

#[test]
fn permanent_large_tile_invariance_cuda() {
    if cuscan_common::cuda_backend::try_device_count().expect("CUDA discovery failed") == 0 {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUDA execution required but unavailable"
        );
        return;
    }
    large_tile_invariance(Device::cuda(0));
}
