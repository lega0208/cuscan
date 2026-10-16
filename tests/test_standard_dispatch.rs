use burn::backend::Dispatch;
use burn::tensor::{Device, Tensor, TensorData};
use fast_umap::backend::StandardUmapOps;

fn run_exact_neighbors(
    query: Tensor<2>,
    reference: Tensor<2>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
) -> (Vec<i32>, Vec<f32>) {
    let (indices, distances) = <Dispatch as StandardUmapOps>::exact_neighbors(
        query.into_dispatch(),
        reference.into_dispatch(),
        k,
        exclude_self,
        query_offset,
    );
    let indices: Tensor<2, burn::tensor::Int> = Tensor::from_dispatch(indices);
    let distances: Tensor<2> = Tensor::from_dispatch(distances);
    (
        indices.to_data().try_to_vec::<i32>().unwrap(),
        distances.to_data().try_to_vec::<f32>().unwrap(),
    )
}

fn run_distance_block(device: &Device) -> Vec<f32> {
    let query =
        Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0, 2.0, 0.0], [2, 2]), device);
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 3.0, 0.0], [3, 2]),
        device,
    );
    let squared_distances: Tensor<2> = Tensor::from_dispatch(
        <Dispatch as StandardUmapOps>::squared_euclidean_distance_block(
            query.into_dispatch(),
            reference.into_dispatch(),
        ),
    );
    assert_eq!(squared_distances.dims(), [2, 3]);
    squared_distances.to_data().try_to_vec::<f32>().unwrap()
}

fn run_large_offset_distance_case(device: &Device) {
    let query = Tensor::<2>::from_data(TensorData::new(vec![1_000_002.0f32], [1, 1]), device);
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![1_000_000.0f32, 1_000_001.0, 1_000_002.0], [3, 1]),
        device,
    );
    let distances = Tensor::<2>::from_dispatch(
        <Dispatch as StandardUmapOps>::squared_euclidean_distance_block(
            query.clone().into_dispatch(),
            reference.clone().into_dispatch(),
        ),
    );
    assert_eq!(
        distances.to_data().try_to_vec::<f32>().unwrap(),
        [4.0, 1.0, 0.0]
    );

    let (indices, neighbor_distances) = run_exact_neighbors(query, reference, 1, true, 2);
    assert_eq!(indices, [1]);
    assert_eq!(neighbor_distances, [1.0]);
}

#[cfg(feature = "cpu")]
#[test]
fn standard_squared_distance_dispatch_uses_cpu_for_rectangular_blocks() {
    assert_eq!(
        run_distance_block(&Device::cpu()),
        [0.0, 0.0, 9.0, 4.0, 4.0, 1.0]
    );
}

#[cfg(feature = "cpu")]
#[test]
fn standard_squared_distance_and_self_exclusion_preserve_large_offset_precision_on_cpu() {
    run_large_offset_distance_case(&Device::cpu());
}

#[cfg(feature = "cpu")]
#[test]
fn exact_neighbors_excludes_global_self_and_orders_ties_by_index_on_cpu() {
    let device = Device::cpu();
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [4, 2]),
        &device,
    );
    let query = Tensor::<2>::from_data(
        TensorData::new(vec![-1.0f32, 0.0, 0.0, 0.0], [2, 2]),
        &device,
    );
    let (indices, distances) = run_exact_neighbors(query, reference, 3, true, 2);
    assert_eq!(indices, [0, 3, 1, 0, 1, 2]);
    assert_eq!(distances, [1.0, 1.0, 2.0, 0.0, 1.0, 1.0]);
}

#[cfg(feature = "cpu")]
#[test]
fn exact_neighbors_is_bipartite_without_implicit_equal_index_exclusion() {
    let device = Device::cpu();
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 0.0], [3, 2]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32, 0.0], [1, 2]), &device);
    let (indices, distances) = run_exact_neighbors(query, reference, 3, false, 0);
    assert_eq!(indices, [0, 1, 2]);
    assert_eq!(distances, [0.0, 0.0, 2.0]);
}

#[cfg(feature = "cpu")]
#[test]
fn exact_neighbors_handles_query_block_boundary_on_cpu() {
    let device = Device::cpu();
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; 33 * 2], [33, 2]), &device);
    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 0.0], [3, 2]),
        &device,
    );
    let (indices, distances) = run_exact_neighbors(query, reference, 2, false, 0);
    assert_eq!(indices.len(), 33 * 2);
    assert_eq!(distances.len(), 33 * 2);
    for row in 0..33 {
        assert_eq!(&indices[row * 2..row * 2 + 2], [0, 1]);
        assert_eq!(&distances[row * 2..row * 2 + 2], [0.0, 0.0]);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn standard_squared_distance_dispatch_uses_cuda_when_available() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    assert_eq!(run_distance_block(&device), [0.0, 0.0, 9.0, 4.0, 4.0, 1.0]);
    run_large_offset_distance_case(&device);
}

#[cfg(feature = "cuda")]
#[test]
fn exact_neighbors_dispatches_cuda_for_ties_zeros_and_block_boundary() {
    let Some(device) = fast_umap::cuda_backend::device(0) else {
        if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
            panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
        }
        eprintln!("SKIPPED: no CUDA device");
        return;
    };
    let fit_reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0], [4, 2]),
        &device,
    );
    let fit_query = Tensor::<2>::from_data(
        TensorData::new(vec![-1.0f32, 0.0, 0.0, 0.0], [2, 2]),
        &device,
    );
    let (fit_indices, fit_distances) = run_exact_neighbors(fit_query, fit_reference, 3, true, 2);
    assert_eq!(fit_indices, [0, 3, 1, 0, 1, 2]);
    assert_eq!(fit_distances, [1.0, 1.0, 2.0, 0.0, 1.0, 1.0]);

    let reference = Tensor::<2>::from_data(
        TensorData::new(vec![0.0f32, 0.0, 0.0, 0.0, 2.0, 0.0], [3, 2]),
        &device,
    );
    let query = Tensor::<2>::from_data(TensorData::new(vec![0.0f32; 33 * 2], [33, 2]), &device);
    let (indices, distances) = run_exact_neighbors(query, reference, 2, false, 0);
    for row in 0..33 {
        assert_eq!(&indices[row * 2..row * 2 + 2], [0, 1]);
        assert_eq!(&distances[row * 2..row * 2 + 2], [0.0, 0.0]);
    }
}
