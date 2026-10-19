use criterion::Criterion;
use fast_umap::{BackendChoice, Init, NeighborSearch, UMAP, UMAPConfig};

const SEED: u64 = 42;

struct Workload {
    name: &'static str,
    fit_rows: usize,
    query_rows: usize,
    features: usize,
    neighbors: usize,
    epochs: usize,
    search: NeighborSearch,
    transform_queue_size: f64,
}

// `representative` is the exact-search baseline workload and must stay unchanged so that
// before/after timings remain comparable. `approximate` uses 8,192 fit rows, which is above
// the documented Auto cutoff (4,096), and requests `NeighborSearch::Approximate` explicitly so
// that both the approximate fit and approximate transform query paths always execute.
const WORKLOADS: [Workload; 3] = [
    Workload {
        name: "smoke",
        fit_rows: 96,
        query_rows: 16,
        features: 8,
        neighbors: 8,
        epochs: 9,
        search: NeighborSearch::Exact,
        transform_queue_size: 4.0,
    },
    Workload {
        name: "representative",
        fit_rows: 1024,
        query_rows: 128,
        features: 32,
        neighbors: 8,
        epochs: 100,
        search: NeighborSearch::Exact,
        transform_queue_size: 4.0,
    },
    Workload {
        name: "approximate",
        fit_rows: 8192,
        query_rows: 128,
        features: 32,
        neighbors: 15,
        epochs: 100,
        search: NeighborSearch::Approximate,
        transform_queue_size: 4.0,
    },
];

fn data(rows: usize, features: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|row| {
            let cluster = row % 4;
            (0..features)
                .map(|feature| {
                    let center = cluster as f32 * 4.0;
                    let wave = ((row * (feature + 3) + feature * 11) as f32 * 0.173).sin();
                    center + feature as f32 * 0.23 + wave * 0.08
                })
                .collect()
        })
        .collect()
}

fn estimator(backend: BackendChoice, workload: &Workload) -> UMAP {
    let config = UMAPConfig {
        n_components: 2,
        n_neighbors: workload.neighbors,
        n_epochs: Some(workload.epochs),
        init: Init::Random,
        random_state: Some(SEED),
        transform_seed: SEED,
        transform_queue_size: workload.transform_queue_size,
        ..UMAPConfig::default()
    };
    UMAP::new(config, backend, workload.search)
}

fn search_label(search: NeighborSearch) -> &'static str {
    match search {
        NeighborSearch::Exact => "exact",
        NeighborSearch::Approximate => "approximate",
        NeighborSearch::Auto => "auto",
    }
}

pub fn benchmarks(c: &mut Criterion, backend_name: &str, backend: BackendChoice) {
    // Fitting returns a host embedding via tensor_embedding(...).to_data(), which
    // synchronizes device work; transform returns its host embedding the same way.
    for workload in &WORKLOADS {
        let fit_data = data(workload.fit_rows, workload.features);
        let query_data = data(workload.query_rows, workload.features);
        let estimator = estimator(backend, workload);
        let transform_epochs = workload.epochs / 3;
        let search = search_label(workload.search);
        let mut group =
            c.benchmark_group(format!("standard_umap/{backend_name}/{}", workload.name));
        group.bench_function(
            format!(
                "fit/{}x{}_{search}_random_{}epochs",
                workload.fit_rows, workload.features, workload.epochs
            ),
            |b| {
                b.iter(|| {
                    estimator
                        .fit(std::hint::black_box(&fit_data))
                        .unwrap_or_else(|error| panic!("{backend_name} fit failed: {error}"))
                })
            },
        );
        group.bench_function(
            format!(
                "transform/{}x{}_against_{}_{search}_random_{}epochs",
                workload.query_rows, workload.features, workload.fit_rows, transform_epochs
            ),
            |b| {
                // Criterion invokes this only when the benchmark matches its filter;
                // prepare the fitted model before the timed transform iterations.
                let fitted = estimator.fit(&fit_data).unwrap_or_else(|error| {
                    panic!(
                        "{backend_name} {} benchmark setup fit failed: {error}",
                        workload.name
                    )
                });
                b.iter(|| {
                    fitted
                        .transform(std::hint::black_box(&query_data))
                        .unwrap_or_else(|error| panic!("{backend_name} transform failed: {error}"))
                })
            },
        );
        group.finish();
    }
}

#[cfg(feature = "tracing")]
pub fn init_tracing() -> Option<tracing_chrome::FlushGuard> {
    use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

    let target = std::env::var_os("FAST_UMAP_TRACE")?;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    if target == "console" {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .try_init()
            .expect("could not initialize tracing subscriber");
        None
    } else {
        let (layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
            .file(target)
            .build();
        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .try_init()
            .expect("could not initialize Chrome tracing subscriber");
        Some(guard)
    }
}

#[cfg(not(feature = "tracing"))]
pub fn init_tracing() -> Option<()> {
    None
}

pub fn run(c: &mut Criterion, backend: BackendChoice, backend_name: &str) {
    // The guard must remain alive until Criterion has completed every benchmark.
    let _trace_guard = init_tracing();
    benchmarks(c, backend_name, backend);
}
