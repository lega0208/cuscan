//! Full-fit CUDA comparison of forced brute-force and KD-tree HDBSCAN routes.
//!
//! The default suite covers pinned fixtures and the complete documented synthetic grid.
//! Use `FAST_HDBSCAN_REQUIRE_CUDA=1` to make absent CUDA support/device a hard failure and
//! `FAST_HDBSCAN_BENCH_CASES=fixture/blobs_d2,n64_d2_ms5` for a smoke subset.
//! Fit returns host-side fitted results, synchronizing device work before timing completes.

mod support;

fn hdbscan_cuda(c: &mut criterion::Criterion) {
    support::run_cuda(c);
}

criterion::criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = hdbscan_cuda
}
criterion::criterion_main!(benches);
