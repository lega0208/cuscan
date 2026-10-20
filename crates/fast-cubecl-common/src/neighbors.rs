use crate::backend::NeighborOps;
use burn::backend::Dispatch;
use burn::tensor::{Int, Tensor};

/// Device exact-neighbor operation dispatched on the input backend.
///
/// Squared mode permits missing neighbors, represented by raw reference-row-count
/// indices and positive infinity. Euclidean mode preserves UMAP's strict width.
pub fn exact_neighbors(
    query: Tensor<2>,
    reference: Tensor<2>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
    squared: bool,
) -> (Tensor<2, Int>, Tensor<2>) {
    let (ids, distances) = <Dispatch as NeighborOps>::exact_neighbors(
        query.into_dispatch(),
        reference.into_dispatch(),
        k,
        exclude_self,
        query_offset,
        squared,
    );
    (Tensor::from_dispatch(ids), Tensor::from_dispatch(distances))
}

/// Compute one bounded query-by-reference squared-distance block.
pub fn squared_euclidean_distance_block(query: Tensor<2>, reference: Tensor<2>) -> Tensor<2> {
    Tensor::from_dispatch(<Dispatch as NeighborOps>::squared_euclidean_distance_block(
        query.into_dispatch(),
        reference.into_dispatch(),
    ))
}

#[cfg(all(test, any(feature = "cpu", feature = "cuda")))]
mod tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    fn devices() -> Vec<Device> {
        let mut devices = Vec::new();
        #[cfg(feature = "cpu")]
        devices.push(Device::cpu());
        #[cfg(feature = "cuda")]
        {
            let count = crate::cuda_backend::try_device_count().expect("CUDA discovery failed");
            if std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_some() {
                assert!(count > 0, "required CUDA execution has no usable device");
            }
            if count > 0 {
                devices.push(Device::cuda(0));
            }
        }
        devices
    }

    fn tensor(values: Vec<f32>, rows: usize, device: &Device) -> Tensor<2> {
        Tensor::from_data(TensorData::new(values, [rows, 1]), device)
    }

    #[test]
    fn squared_mode_orders_ties_and_preserves_squared_distances() {
        for device in devices() {
            let q = tensor(vec![0.0], 1, &device);
            let r = tensor(vec![1.0, -1.0, 2.0], 3, &device);
            let (ids, d) = exact_neighbors(q, r, 3, false, 0, true);
            assert_eq!(ids.dims(), [1, 3]);
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [0, 1, 2]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [1.0, 1.0, 4.0]);
        }
    }

    #[test]
    fn squared_mode_real_infinity_precedes_raw_n_padding() {
        for device in devices() {
            let q = tensor(vec![f32::MAX], 1, &device);
            let r = tensor(vec![-f32::MAX, -f32::MAX], 2, &device);
            let (ids, d) = exact_neighbors(q, r, 3, false, 0, true);
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [0, 1, 2]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [f32::INFINITY; 3]);
        }
    }

    #[test]
    fn squared_mode_excludes_offset_self_and_pads_missing_slots() {
        for device in devices() {
            let q = tensor(vec![10.0, 20.0], 2, &device);
            let r = tensor(vec![0.0, 10.0, 20.0], 3, &device);
            let (ids, d) = exact_neighbors(q, r, 3, true, 1, true);
            assert_eq!(
                ids.to_data().try_to_vec::<i32>().unwrap(),
                [0, 2, 3, 1, 0, 3]
            );
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [100.0, 100.0, f32::INFINITY, 100.0, 400.0, f32::INFINITY]
            );
            let (ids, d) = exact_neighbors(
                tensor(vec![0.0], 1, &device),
                tensor(vec![0.0], 1, &device),
                2,
                true,
                0,
                true,
            );
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [1, 1]);
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [f32::INFINITY; 2]);
        }
    }

    #[test]
    fn euclidean_mode_roots_distances_and_retains_overflow_sentinels() {
        for device in devices() {
            let (_, d) = exact_neighbors(
                tensor(vec![0.0], 1, &device),
                tensor(vec![3.0, 4.0], 2, &device),
                2,
                false,
                0,
                false,
            );
            assert_eq!(d.to_data().try_to_vec::<f32>().unwrap(), [3.0, 4.0]);
            let (ids, d) = exact_neighbors(
                tensor(vec![f32::MAX], 1, &device),
                tensor(vec![-f32::MAX, -f32::MAX], 2, &device),
                2,
                false,
                0,
                false,
            );
            assert_eq!(ids.to_data().try_to_vec::<i32>().unwrap(), [2, 2]);
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [f32::MAX.sqrt(); 2]
            );
        }
    }

    #[test]
    #[should_panic(expected = "neighbor count exceeds available references")]
    fn euclidean_mode_rejects_excess_width_at_execution() {
        #[cfg(feature = "cpu")]
        let device = Device::cpu();
        #[cfg(not(feature = "cpu"))]
        let device = Device::cuda(0);
        let (_, d) = exact_neighbors(
            tensor(vec![0.0], 1, &device),
            tensor(vec![3.0, 4.0], 2, &device),
            3,
            false,
            0,
            false,
        );
        let _ = d.to_data();
    }

    #[test]
    fn distance_blocks_keep_shape_and_squared_values() {
        for device in devices() {
            let d = squared_euclidean_distance_block(
                tensor(vec![0.0, 2.0], 2, &device),
                tensor(vec![1.0, 4.0, 6.0], 3, &device),
            );
            assert_eq!(d.dims(), [2, 3]);
            assert_eq!(
                d.to_data().try_to_vec::<f32>().unwrap(),
                [1.0, 16.0, 36.0, 1.0, 4.0, 16.0]
            );
        }
    }
}
