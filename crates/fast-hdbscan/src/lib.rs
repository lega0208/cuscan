//! HDBSCAN facade infrastructure.
//!
//! This crate currently provides dependency and backend feature wiring only.

#[cfg(not(any(feature = "cpu", feature = "cuda")))]
compile_error!("enable at least one backend feature: cpu or cuda");

#[cfg(all(test, not(all(feature = "cpu", feature = "cuda"))))]
mod tests {
    use fast_cubecl_common::{BackendChoice, selected_device};

    #[test]
    fn facade_permissions_are_not_expanded_by_peer_features() {
        let cpu = cfg!(feature = "cpu");
        let cuda = cfg!(feature = "cuda");
        #[cfg(not(feature = "cpu"))]
        assert_eq!(
            selected_device(BackendChoice::Cpu, cpu, cuda)
                .unwrap_err()
                .to_string(),
            "CPU backend requires the `cpu` feature"
        );
        #[cfg(not(feature = "cuda"))]
        {
            assert_eq!(
                selected_device(BackendChoice::Cuda { device_index: 0 }, cpu, cuda)
                    .unwrap_err()
                    .to_string(),
                "CUDA backend requires the `cuda` feature"
            );
            assert_eq!(
                selected_device(BackendChoice::Auto, cpu, cuda).unwrap().1,
                BackendChoice::Cpu
            );
        }
    }
}
