//! Flex standard UMAP profiler. Quick smoke run: `cargo bench --no-default-features
//! --bench standard_umap_flex -- --test`. Add `--features tracing` and set
//! `FAST_UMAP_TRACE=console` for span-close output, or set it to a Chrome trace
//! filename (for example `FAST_UMAP_TRACE=umap.json`). `RUST_LOG` controls the
//! tracing filter. Tracing setup is compiled only with the `tracing` feature and
//! no subscriber is installed unless `FAST_UMAP_TRACE` is set.

mod support;

fn standard_umap_flex(c: &mut criterion::Criterion) {
    support::run(c, fast_umap::BackendChoice::Flex, "flex");
}

criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default();
    targets = standard_umap_flex
}
criterion::criterion_main!(benches);
