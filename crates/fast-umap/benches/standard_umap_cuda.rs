//! CUDA standard UMAP profiler; this target requires the `cuda` feature and
//! never falls back to CPU. Quick smoke run: `cargo bench --no-default-features
//! --features cuda --bench standard_umap_cuda -- --test`. Add `tracing` and set
//! `FAST_UMAP_TRACE=console` for span-close output, or set it to a Chrome trace
//! filename (for example `FAST_UMAP_TRACE=umap.json`). `RUST_LOG` controls the
//! tracing filter. Tracing setup is compiled only with that feature and no
//! subscriber is installed unless `FAST_UMAP_TRACE` is set.

mod support;

fn standard_umap_cuda(c: &mut criterion::Criterion) {
    support::run(
        c,
        fast_umap::BackendChoice::Cuda { device_index: 0 },
        "cuda",
    );
}

criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default();
    targets = standard_umap_cuda
}
criterion::criterion_main!(benches);
