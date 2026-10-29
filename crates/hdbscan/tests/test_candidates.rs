use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData},
};
use hdbscan::BackendChoice;
use hdbscan::HdbscanOps;

fn check_candidates(device: &Device, n: usize, components_count: usize) {
    let dimensions = 3;
    let data = (0..n)
        .flat_map(|i| {
            let x = ((i * 17 + i / 7) % 97) as f32 * 0.125;
            [x, ((i * 11) % 53) as f32 * 0.25, (i % 13) as f32]
        })
        .collect::<Vec<_>>();
    let core = (0..n).map(|i| (i % 9) as f32 * 0.125).collect::<Vec<_>>();
    let component = (0..n)
        .map(|i| ((i * 7 + i / 19) % components_count) as i32)
        .collect::<Vec<_>>();
    let input = Tensor::<2>::from_data(TensorData::new(data.clone(), [n, dimensions]), device);
    let core_sq = Tensor::<1>::from_data(TensorData::new(core.clone(), [n]), device);
    let components = Tensor::<1, Int>::from_data(TensorData::new(component.clone(), [n]), device);
    let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates(
        input.into_dispatch(),
        core_sq.into_dispatch(),
        components.into_dispatch(),
    );
    let weights: Tensor<1> = Tensor::from_dispatch(weights);
    let destinations: Tensor<1, Int> = Tensor::from_dispatch(destinations);
    let weights = weights.to_data().try_to_vec::<f32>().unwrap();
    let destinations = destinations.to_data().try_to_vec::<i32>().unwrap();

    let mut expected_weights = vec![f32::INFINITY; n];
    let mut expected_destinations = vec![n as i32; n];
    for i in 0..n {
        for j in 0..n {
            if component[i] == component[j] {
                continue;
            }
            let mut distance = 0.0f32;
            for feature in 0..dimensions {
                let delta = data[i * dimensions + feature] - data[j * dimensions + feature];
                distance += delta * delta;
            }
            let weight = distance.max(core[i]).max(core[j]);
            if (weight, j as i32) < (expected_weights[i], expected_destinations[i]) {
                expected_weights[i] = weight;
                expected_destinations[i] = j as i32;
            }
        }
    }
    assert_eq!(weights, expected_weights);
    assert_eq!(destinations, expected_destinations);
}

#[test]
fn boruvka_candidates_match_host_oracle_across_reference_tiles_cpu() {
    for (n, components) in [(1, 1), (2, 2), (257, 2), (513, 7)] {
        check_candidates(&Device::cpu(), n, components);
    }
}

#[test]
fn boruvka_candidates_match_host_oracle_across_reference_tiles_cuda() {
    let resolved = cuscan_common::selected_device(BackendChoice::Cuda { device_index: 0 });
    let Ok((device, _)) = resolved else {
        assert!(
            std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
            "CUSCAN_REQUIRE_CUDA is set, but no CUDA device is available"
        );
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    for (n, components) in [(1, 1), (2, 2), (257, 2), (513, 7)] {
        check_candidates(&device, n, components);
    }
}
