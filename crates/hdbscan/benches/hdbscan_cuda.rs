//! Full-fit CUDA comparison of forced brute-force and KD-tree HDBSCAN routes.
//!
//! The default suite covers pinned fixtures and the complete documented synthetic grid.
//! Use `CUSCAN_REQUIRE_CUDA=1` to make absent CUDA support/device a hard failure and
//! `CUSCAN_BENCH_CASES=fixture/blobs_d2,n64_d2_ms5` for a smoke subset.
//! Fit returns host-side fitted results, synchronizing device work before timing completes.

mod support;

fn hdbscan_cuda(c: &mut criterion::Criterion) {
    let backend = hdbscan::BackendChoice::Cuda { device_index: 0 };
    match cuscan_common::selected_device(backend) {
        Ok(_) => support::run(c, backend, "cuda"),
        Err(error) if std::env::var_os("CUSCAN_REQUIRE_CUDA").is_some() => {
            panic!("CUSCAN_REQUIRE_CUDA is set but CUDA is unavailable: {error}")
        }
        Err(error) => eprintln!("SKIPPED CUDA HDBSCAN benchmarks: {error}"),
    }
}

criterion::criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = hdbscan_cuda
}
criterion::criterion_main!(benches);
