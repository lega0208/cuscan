//! CubeCL CPU standard UMAP profiler. Quick smoke run: `cargo bench
//! --bench standard_umap_cpu -- --test`. Add `--features tracing` and set
//! `CUSCAN_TRACE=console` for span-close output, or set it to a Chrome trace
//! filename (for example `CUSCAN_TRACE=umap.json`). `RUST_LOG` controls the
//! tracing filter. Tracing setup is compiled only with the `tracing` feature and
//! no subscriber is installed unless `CUSCAN_TRACE` is set.

mod support;

fn standard_umap_cpu(c: &mut criterion::Criterion) {
    support::run(c, umap::BackendChoice::Cpu, "cpu");
}

criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default();
    targets = standard_umap_cpu
}
criterion::criterion_main!(benches);
