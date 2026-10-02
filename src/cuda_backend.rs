//! CUDA backend implementation for fast-umap.
//!
//! Reuses the generic CubeCL kernels in [`crate::kernels`] with Burn's CUDA
//! runtime, and exposes basic CUDA device discovery helpers.

#[cfg(test)]
use burn::tensor::{Int, Tensor, TensorData};

/// Return the number of CUDA devices visible to the CUDA driver.
///
/// Returns zero if the CUDA driver is unavailable or device discovery fails.
pub fn device_count() -> usize {
    // SAFETY: This only checks whether the CUDA driver library can be loaded.
    if !unsafe { cudarc::driver::sys::is_culib_present() } {
        return 0;
    }

    usize::try_from(cudarc::driver::CudaContext::device_count().unwrap_or(0)).unwrap_or(0)
}

/// Return whether at least one CUDA device is available.
pub fn is_available() -> bool {
    device_count() > 0
}

/// Return a CUDA device by index, or `None` when the driver is unavailable or the index is out of range.
pub fn device(index: usize) -> Option<burn::tensor::Device> {
    (index < device_count()).then(|| burn::tensor::Device::cuda(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::UmapOps;
    use burn::backend::Dispatch;
    use rand::RngExt;

    fn pairwise_distance(x: Tensor<2>) -> Tensor<2> {
        Tensor::from_dispatch(<Dispatch as UmapOps>::euclidean_pairwise_distance(
            x.into_dispatch(),
        ))
    }

    fn knn(x: Tensor<2>, k: u32) -> (Tensor<2, Int>, Tensor<2>) {
        let (indices, distances) = <Dispatch as UmapOps>::knn(x.into_dispatch(), k);
        (
            Tensor::from_dispatch(indices),
            Tensor::from_dispatch(distances),
        )
    }

    fn knn_backward(pairwise: Tensor<2>, k: u32, grad: Tensor<2>) -> Tensor<2> {
        Tensor::from_dispatch(<Dispatch as UmapOps>::knn_backward(
            pairwise.into_dispatch(),
            k,
            grad.into_dispatch(),
        ))
    }

    fn euclidean_pairwise_distance_backward(
        grad_pairwise: Tensor<2>,
        points: Tensor<2>,
        pairwise: Tensor<2>,
    ) -> Tensor<2> {
        Tensor::from_dispatch(<Dispatch as UmapOps>::euclidean_pairwise_distance_backward(
            grad_pairwise.into_dispatch(),
            points.into_dispatch(),
            pairwise.into_dispatch(),
        ))
    }

    fn available_device_or_skip() -> Option<burn::tensor::Device> {
        if !is_available() {
            if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
                panic!("FAST_UMAP_REQUIRE_CUDA is set, but no CUDA device is available");
            }
            eprintln!("SKIPPED: no CUDA device");
            return None;
        }

        assert!(
            device(0).is_some(),
            "device zero must exist when CUDA is available"
        );
        Some(device(0).expect("device zero is available"))
    }

    #[test]
    fn out_of_range_device_is_none() {
        assert!(device(usize::MAX).is_none());
    }

    #[test]
    fn cuda_custom_kernels_match_expected_values_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };

        let device = device.autodiff();
        let points = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 3.0, 0.0, 0.0, 4.0], [3, 2]),
            &device,
        );
        let distance_tensor = pairwise_distance(points);
        let actual_distances = distance_tensor.to_data().try_to_vec::<f32>().unwrap();
        let expected_distances = [0.0, 3.0, 4.0, 3.0, 0.0, 5.0, 4.0, 5.0, 0.0];
        for (actual, expected) in actual_distances.iter().zip(expected_distances) {
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        }

        let (indices, nearest_distances) = knn(distance_tensor, 1);
        let actual_indices = indices.to_data().try_to_vec::<i32>().unwrap();
        assert_eq!(actual_indices, [1, 0, 0]);

        let actual_nearest_distances = nearest_distances.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(actual_nearest_distances, [3.0, 3.0, 4.0]);
    }

    #[test]
    fn cuda_knn_k3_returns_ordered_neighbors_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };

        // Distances for points at 0, 1, 4, 10, 18, and 29 on a line.
        let pairwise = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    0.0f32, 1.0, 4.0, 10.0, 18.0, 29.0, 1.0, 0.0, 3.0, 9.0, 17.0, 28.0, 4.0, 3.0,
                    0.0, 6.0, 14.0, 25.0, 10.0, 9.0, 6.0, 0.0, 8.0, 19.0, 18.0, 17.0, 14.0, 8.0,
                    0.0, 11.0, 29.0, 28.0, 25.0, 19.0, 11.0, 0.0,
                ],
                [6, 6],
            ),
            &device,
        );
        let (indices, distances) = knn(pairwise, 3);
        assert_eq!(
            indices.to_data().try_to_vec::<i32>().unwrap(),
            [1, 2, 3, 0, 2, 3, 1, 0, 3, 2, 4, 1, 3, 5, 2, 4, 3, 2]
        );

        let actual_distances = distances.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(
            actual_distances,
            [
                1.0, 4.0, 10.0, 1.0, 3.0, 9.0, 3.0, 4.0, 6.0, 6.0, 8.0, 9.0, 8.0, 11.0, 14.0, 11.0,
                19.0, 25.0
            ]
        );
        for row in actual_distances.chunks_exact(3) {
            assert!(
                row[0] < row[1] && row[1] < row[2],
                "neighbors not ascending: {row:?}"
            );
        }
    }

    #[test]
    fn cuda_knn_backward_k1_matches_expected_gradient_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };

        let pairwise = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 3.0, 4.0, 3.0, 0.0, 5.0, 4.0, 5.0, 0.0], [3, 3]),
            &device,
        );
        let grad_output =
            Tensor::<2>::from_data(TensorData::new(vec![1.0f32, 1.0, 1.0], [3, 1]), &device);
        let grad = knn_backward(pairwise, 1, grad_output);
        assert_eq!(
            grad.to_data().try_to_vec::<f32>().unwrap(),
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn cuda_knn_backward_k2_matches_expected_gradient_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };

        // Four points on a line at 0, 2, 5, and 11 have unambiguous top-2 neighbors.
        let pairwise = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    0.0f32, 2.0, 5.0, 11.0, 2.0, 0.0, 3.0, 9.0, 5.0, 3.0, 0.0, 6.0, 11.0, 9.0, 6.0,
                    0.0,
                ],
                [4, 4],
            ),
            &device,
        );
        let grad_output = Tensor::<2>::from_data(
            TensorData::new(vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], [4, 2]),
            &device,
        );
        let grad = knn_backward(pairwise, 2, grad_output);
        assert_eq!(
            grad.to_data().try_to_vec::<f32>().unwrap(),
            [0.0, 1.0, 10.0, 0.0, 2.0, 0.0, 20.0, 0.0, 30.0, 3.0, 0.0, 0.0, 0.0, 40.0, 4.0, 0.0]
        );
    }

    #[test]
    fn cuda_euclidean_pairwise_backward_matches_analytic_gradient_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };

        let points = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 0.0, 3.0, 0.0], [2, 2]),
            &device,
        );
        let pairwise = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 3.0, 3.0, 0.0], [2, 2]),
            &device,
        );
        let grad_pairwise = Tensor::<2>::from_data(
            TensorData::new(vec![0.0f32, 1.0, 0.0, 0.0], [2, 2]),
            &device,
        );
        let grad = euclidean_pairwise_distance_backward(grad_pairwise, points, pairwise);
        let actual = grad.to_data().try_to_vec::<f32>().unwrap();
        for (actual, expected) in actual.iter().zip([-1.0, 0.0, 1.0, 0.0]) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn cuda_umap_fit_smoke_test_when_available() {
        let Some(device) = available_device_or_skip() else {
            return;
        };
        use crate::prelude::{GraphParams, OptimizationParams, UmapConfig};

        let mut rng = rand::rng();
        let data: Vec<Vec<f64>> = (0..50)
            .map(|_| (0..5).map(|_| rng.random::<f64>()).collect())
            .collect();
        let config = UmapConfig {
            graph: GraphParams {
                n_neighbors: 5,
                ..Default::default()
            },
            optimization: OptimizationParams {
                n_epochs: 2,
                batch_size: 50,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let fitted = crate::Umap::with_device(config, device.autodiff()).fit(data, None);
        let embedding = fitted.embedding();
        assert_eq!(embedding.len(), 50);
        assert!(embedding.iter().all(|row| row.len() == 2));
        assert!(embedding.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn cuda_umap_save_load_round_trip_when_available() -> Result<(), Box<dyn std::error::Error>> {
        let Some(device) = available_device_or_skip() else {
            return Ok(());
        };
        use crate::prelude::{GraphParams, OptimizationParams, UmapConfig};

        // Each feature has mean zero and standard deviation one, matching the
        // identity normalization used when loading serialized model weights.
        let data = vec![
            vec![-1.0, -1.0],
            vec![-1.0, 1.0],
            vec![1.0, -1.0],
            vec![1.0, 1.0],
        ];
        let config = UmapConfig {
            graph: GraphParams {
                n_neighbors: 3,
                ..Default::default()
            },
            optimization: OptimizationParams {
                n_epochs: 2,
                batch_size: 4,
                verbose: false,
                ..Default::default()
            },
            ..Default::default()
        };

        let fitted = crate::Umap::with_device(config.clone(), device.clone().autodiff())
            .fit(data.clone(), None);
        let temp_dir = tempfile::Builder::new()
            .prefix("fast-umap-cuda-")
            .tempdir_in(std::env::temp_dir())?;
        let model_path = temp_dir.path().join("model.bin");
        fitted.save(&model_path)?;
        let loaded = crate::FittedUmap::load(&model_path, config, 2, device.autodiff())?;

        let original_embedding = fitted.transform(data.clone());
        let loaded_embedding = loaded.transform(data);
        assert_eq!(original_embedding.len(), loaded_embedding.len());
        for (original_row, loaded_row) in original_embedding.iter().zip(&loaded_embedding) {
            assert_eq!(original_row.len(), loaded_row.len());
            for (original, loaded) in original_row.iter().zip(loaded_row) {
                assert!(
                    (original - loaded).abs() < 1e-5,
                    "serialized CUDA model changed embedding: {original} != {loaded}"
                );
            }
        }

        Ok(())
    }
}
