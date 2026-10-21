//! Full-fit CPU comparison of forced brute-force and KD-tree HDBSCAN routes.
//!
//! The default suite covers pinned fixtures and the complete documented synthetic grid.
//! Use `FAST_HDBSCAN_BENCH_CASES=fixture/blobs_d2,n64_d2_ms5` to select a smoke subset.
//! Results use Criterion's 100 samples and 95% confidence interval.

mod support;

fn hdbscan_cpu(c: &mut criterion::Criterion) {
    support::run_cpu(c);
}

criterion::criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = hdbscan_cpu
}
criterion::criterion_main!(benches);
