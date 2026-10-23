use burn::tensor::{Device, Tensor, TensorData};
use fast_umap::standard_umap::neighbors::exact_neighbors;

fn exercise_blocks(device: Device) {
    let points = Tensor::<2>::from_data(
        TensorData::new((0..70).map(|i| i as f32).collect::<Vec<_>>(), [70, 1]),
        &device,
    );
    let fit = exact_neighbors(points.clone(), points.clone(), 2, true).unwrap();
    assert_eq!(fit.indices.dims(), [70, 2]);
    assert_eq!(fit.indices.device(), device);
    let ids = fit.indices.to_data().try_to_vec::<i32>().unwrap();
    let distances = fit.distances.to_data().try_to_vec::<f32>().unwrap();
    for row in 0..70 {
        let expected = if row == 0 {
            [1, 2]
        } else if row == 69 {
            [68, 67]
        } else {
            [row - 1, row + 1]
        };
        assert_eq!(
            &ids[row * 2..row * 2 + 2],
            &[expected[0] as i32, expected[1] as i32]
        );
        for column in 0..2 {
            assert_eq!(
                distances[row * 2 + column],
                row.abs_diff(expected[column]) as f32
            );
        }
    }
    let queries = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 64.0], [2, 1]), &device);
    let bipartite = exact_neighbors(queries, points, 2, false).unwrap();
    assert_eq!(
        bipartite.indices.to_data().try_to_vec::<i32>().unwrap(),
        [0, 1, 64, 63]
    );
    assert_eq!(
        bipartite.distances.to_data().try_to_vec::<f32>().unwrap(),
        [0.0, 1.0, 0.0, 1.0]
    );
}

#[test]
fn exact_consumer_crosses_blocks_on_cpu() {
    exercise_blocks(Device::cpu());
}

#[test]
fn exact_consumer_crosses_blocks_on_actual_cuda() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        assert!(
            std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
            "CUDA is required"
        );
        return;
    };
    exercise_blocks(device);
}

#[test]
fn exact_consumer_rejects_invalid_widths_and_alignment() {
    let device = Device::cpu();
    let fit = Tensor::<2>::zeros([3, 2], &device);
    let query = Tensor::<2>::zeros([2, 2], &device);
    assert!(exact_neighbors(fit.clone(), fit.clone(), 0, true).is_err());
    assert!(exact_neighbors(fit.clone(), fit.clone(), 3, true).is_err());
    assert!(exact_neighbors(query.clone(), fit.clone(), 1, true).is_err());
    assert!(exact_neighbors(query, Tensor::<2>::zeros([3, 1], &device), 1, false).is_err());
}
