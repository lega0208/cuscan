# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Breaking

- Removed parametric UMAP from the Rust crate and Python package: `ParametricUMAP`, `FittedParametricUMAP`, `ParametricUMAPConfig`, the neural model, sparse trainer, weight persistence, `fit_parametric_umap`, and `fast_umap.ParametricUMAP`/`NotFittedError`. Also removed the parametric-only infrastructure: the differentiable `UmapOps` extension and its Euclidean/k-NN kernels, `autodiff_ops`, `distances`, `graph` (`Metric`, `GraphParams`, `ManifoldParams`), `nn_descent`, `pca`, `chart`, `normalizer`, `utils`, `macros`, `prelude`, the `bench_report` binary, the `umap_bench` benchmark, the parametric examples, and the `gpu`, `parametric`, `pca`, `nn-descent`, `plotters`, `umap-rs`, and `verbose` features. The crate now provides only Standard UMAP on the `cpu` and `cuda` features.

### Added

- **HDBSCAN kNN-graph route:** `MstGraph::{Auto, Complete, KnnGraph}` and `knn_k` in `HdbscanConfig`. The graph width is `min(max(knn_k or max(3s, 15), s), n_finite - 1)`. Exact neighbors are used for at most 4096 finite rows; larger inputs use common ANN at the resolved width (seed 42, squared distances), with no width cap. Kruskal runs on the kNN edges, and disconnected components are joined by `+inf` bridges. Python exposes `HDBSCAN(knn_k=...)`, and `reproducible` is forwarded to Rust.
- **HDBSCAN resource advisories:** Rust `fast_hdbscan::estimate_fit_resources` returns a shape-only `FitResourceEstimate` (no data reads, no device discovery). kNN-route fits emit one advisory when a CPU or CUDA scenario reaches 1 GiB (2^30 bytes): Python emits a filterable `UserWarning`, and Rust emits a `tracing` warning under the `tracing` feature. The advisory is informational only and does not change the fit, configuration or backend.
- **Python fitted-model serialization:** `fast_umap.UMAP.save(path, overwrite=False)` and `UMAP.load(path, backend="auto", device_index=0)` persist fitted numeric state in versioned HDF5 schema v1 using the optional `serialization` extra (`h5py` and `hdf5plugin`). Compression uses lossless Blosc2 Zstandard level 1 with byte shuffle in 32768-row chunks; backend/device remain load-time choices, and Rust does not link HDF5.
- **Standard (non-parametric) UMAP:** Rust `UMAP`/`FittedUMAP` with `UMAPConfig`, `BackendChoice` (Flex or CUDA) and `NeighborSearch` (`Exact`, `Approximate`, `Auto`), plus Python `fast_umap.UMAP`. Follows umap-learn 0.5.12 (smooth-kNN fuzzy graph, bounded Lanczos spectral initialization, reference SGD schedule and Tausworthe sampling); fit and transform run neighbor search, graph construction and layout on the selected backend through the forward-only `StandardUmapOps` extension.
- Added an optional CUDA backend that reuses the CubeCL kernels; enable it with the `cuda` feature. Runtime requires an NVIDIA GPU and CUDA runtime libraries (`libcuda`, `libnvrtc`); `nvcc`/CUDA toolkit are optional at build time.
- **Python bindings** — Installable `fast_umap` package with a scikit-learn-style `ParametricUMAP` estimator, selectable GPU (WGPU) or CPU (Flex) backend via `device=`.
- Added `ParametricUMAPOptimizationParams.seed`; the sampling RNG is seeded as well as the model RNG.
- Added `validate_parametric_umap_input` and `validate_parametric_umap_data` to reject invalid fit inputs before training.
- Added primary Rust/Python `local_connectivity`, `bandwidth`, and `set_op_mix_ratio` controls (all default to 1), with reference-labelled umap-learn 0.5.12 graph fixtures.

### Changed

- Reorganized the repository into a virtual Cargo workspace: `crates/fast-umap-core` (backend-free config, errors, validation, model state, query/ANN sizing and spectral initialization; tests without a backend feature or LLVM bundle), `crates/fast-umap` (facade: backend trait, kernels, CUDA discovery, `BackendChoice`, estimator) and `crates/fast-umap-python` (formerly `bindings/python`). Every existing `fast_umap` public path is preserved through re-exports. Shared dependency requirements live in `[workspace.dependencies]`; all members are `publish = false`. Cargo commands now select a package with `-p`.
- **Breaking: primary `n_neighbors` now includes self and must be at least 2.** A fixed value selects one fewer non-self neighbor. Exact and optional NN-descent preparation retain distances and insert explicit self at slot zero; duplicate observations remain distinct neighbors. Shared search helpers and deprecated dense graph behavior are unchanged.
- Primary sparse training now computes smooth-kNN rho/sigma, density-adaptive directed memberships, and fuzzy union/intersection weights. Configured positive epochs E fix float32 pruning (`w < max(w)/E`) and integer counts (`floor(E*w)`) before training. Checked u64 prefix sampling draws `min(50_000, sum(counts))` positives independently with replacement each epoch, without expanding repeated edges. Empty mass or count overflow fails explicitly. Fresh uniform non-self negatives replace the cached 16-batch cycle. This does not reproduce the reference training schedule, negative distribution, loss, or optimizer.
- **Breaking: `ParametricUMAP` fit and transform do not standardize features internally.** They receive caller-provided values as-is, so feature scales affect geometry. Saved weights preserve the caller-input contract; weights trained with former internal standardization require the original external preprocessing at inference or retraining; no automatic migration is provided.
- Dropped the Burn `train` feature to provide deterministic CPU execution.
- Python `transform` handles empty inputs and reports feature-count mismatches as `ValueError`.
- **HDBSCAN default route above 30 features:** `MstGraph::Auto` (the default) now builds a kNN graph when there are more than 30 features, matching upstream `fast_hdbscan` 0.3.2. Previously every dimension used the exact complete graph, so MST, labels and probabilities for those inputs can change. Up to 30 features the exact complete graph is unchanged.
- **HDBSCAN `reproducible` is no longer a no-op:** `false` (default) shares per-component KD bounds within each launch; `true` uses independent per-row candidates. The brute-force and kNN-graph routes ignore it.
- HDBSCAN ANN workspace and forest planning use checked arithmetic; overflow surfaces as typed `HdbscanError::Overflow`.
- HDBSCAN approximate-neighbor kernels moved from `fast-umap` into `fast-cubecl-common` (`NeighborOps::approximate_neighbors` with a squared-output flag). Standalone ANN IDs and distance bits matched the captured baselines on CPU and required CUDA; UMAP embeddings are covered by the existing quality gates, not bitwise equality.

---

## [1.5.0] — 2026-04-11

### Changed

- **GPU dependencies are now optional** — `burn-cubecl` and `cubecl` are gated behind the `gpu` feature flag, reducing compile times and dependency footprint when building without GPU support.
- Added `required-features` for GPU/plotters-dependent examples and binaries (`bench_mnist`, `mnist_benchmark`, `crate_comparison`, `bench_report`).

---

## [1.4.0] — 2026-04-11

### Added

- **Model serialization/deserialization** — Save trained UMAP models to disk and load them back for later use.
  - `FittedUmap::save(&path)` — Serialize model weights to a binary file
  - `FittedUmap::load(path, config, input_size, device)` — Load model with explicit input size
  - `FittedUmap::load_with_sample(path, config, sample_data, device)` — Load model with automatic input size detection
  - Uses burn's binary serialization for compact, efficient storage (~4-6 bytes per parameter)
  - Full precision preservation with fast load times (<1ms for all model sizes)

- **New `serialize` module** — Core serialization logic with comprehensive error handling
  - `save_model()` — Low-level model saving function
  - `load_model()` — Low-level model loading function
  - Supports all backend types (WGPU, CPU, etc.)

- **Examples and benchmarks** — Practical demonstrations and performance analysis
  - `examples/serialization_demo.rs` — Complete usage example
  - `examples/benchmark_serialization.rs` — Performance benchmarking
  - `examples/benchmark_scalability.rs` — Scalability testing across model sizes

- **Comprehensive documentation** — Complete API reference and usage guides
  - `SERIALIZATION_PERFORMANCE.md` — Detailed performance analysis (5-45 MB/s speeds)
  - `TEST_COVERAGE_REPORT.md` — Complete test coverage report (100% coverage)

### Changed

- **Version bump to 1.4.0** — Minor version increment for new serialization feature

---

## [1.2.3] — 2026-03-10

### Changed

- **`plotters` feature is now optional** — `plotters` and `hsl` are no longer
  unconditional dependencies. They are gated behind a new `plotters` cargo
  feature, which is included in `default` so existing users see no change.
  Downstreams that don't need chart output can opt out with
  `default-features = false` (or `features = ["verbose"]`) to shed `plotters`,
  `plotters-bitmap`, `plotters-backend`, `hsl`, and `image v0.24` entirely.
- All chart-related code (`pub mod chart`, imports, `figures_dir`/`can_plot`
  setup, every `plot_loss` / `chart_tensor` call, `PLOT_INTERVAL` constant)
  is gated with `#[cfg(feature = "plotters")]`. Snapshot blocks that require
  both features use `#[cfg(all(feature = "plotters", feature = "verbose"))]`.

---

## [1.2.2] — 2026-03-03

### Added

- **`cooldown_ms` parameter** (`OptimizationParams`, `TrainingConfig`,
  `TrainingConfigBuilder`) — configurable per-epoch sleep in milliseconds that
  gives the GPU scheduler breathing room and prevents 100 % utilisation during
  long training runs. Default is `0` (disabled, no change to existing behaviour).
  Propagated through both the sparse (`train_sparse`) and dense (`train`) loops.

---

## [1.2.0] — 2026-02-24

### Added

- **Proper UMAP kernel** — `q = 1 / (1 + a · d^(2b))` with `a` and `b` fitted
  from `min_dist` and `spread` via grid search + coordinate descent refinement.
  Replaces the fixed Student-t kernel `1/(1+d²)` which was causing clusters to
  condense into blobs.
- **`neg_sample_rate` parameter** (`OptimizationParams`, `TrainingConfig`,
  `TrainingConfigBuilder`) — configurable number of negative (repulsion) samples
  per positive (attraction) edge per epoch. Default: 5.
- **`kernel_a` / `kernel_b` fields** on `TrainingConfig` — automatically
  computed from `ManifoldParams` and passed to both dense and sparse training.
- **`fit_ab(min_dist, spread)` public function** — computes UMAP kernel
  parameters; available for advanced use.

### Changed

- **`ManifoldParams` is now active** — `min_dist` and `spread` were previously
  defined in the config but completely ignored during training. They now control
  the kernel shape that determines how tightly clusters pack and how far apart
  groups separate.
- **Negative sample count fixed** — was `n_pos × neg_rate / k` (giving only
  16 666 negatives for 50 000 positives with k=15); now correctly
  `n_pos × neg_rate` (250 000 with default `neg_sample_rate=5`).
- **All training log output gated behind `verbose`** — `println!` calls in
  `train_sparse` and `train` are now silent when `verbose = false` (the default).
  Previously all runs printed k-NN progress and kernel info unconditionally.
- **Improved log messages** — structured `[fast-umap]` prefix; configuration
  summary at start; k-NN timing with edge count breakdown (positive + negative);
  stop-reason messages (timeout, early stopping, desired loss); final training
  summary with elapsed time and best loss.
- **Progress bar shows epoch/total** — e.g. `Epoch: 42/200` instead of just
  `Epoch: 42`.
- **Kernel doc comments updated** — lib.rs and README now reference the UMAP
  kernel `1/(1 + a·d^(2b))` instead of the Student-t kernel.

### Performance

Updated benchmarks on Apple M3 Max (fast-umap: 50 epochs GPU, umap-rs: 200 epochs CPU):

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|---------|
| 500 × 50 | 0.84s | 0.08s | 0.10× *(umap-rs faster)* |
| 1 000 × 50 | 2.19s | 0.12s | 0.05× *(umap-rs faster)* |
| 2 000 × 100 | 3.65s | 0.44s | 0.12× *(umap-rs faster)* |
| 5 000 × 100 | 6.75s | 2.31s | 0.34× *(umap-rs faster)* |
| 10 000 × 100 | 5.93s | 8.68s | **1.5× faster** |
| 20 000 × 100 | 7.32s | 34.10s | **4.7× faster** |

---

## [1.1.0] — 2026-02-24

### Added

- **New public API** mirroring [umap-rs](https://crates.io/crates/umap-rs):
  - `Umap<B>` — main algorithm struct (`Umap::new(config)`)
  - `FittedUmap<B>` — fitted model with `.embedding()`, `.into_embedding()`, `.transform()`, `.config()`
  - `UmapConfig` — configuration with nested `GraphParams` and `OptimizationParams`
  - `ManifoldParams`, `GraphParams`, `OptimizationParams` — structured config types (all `Serialize`/`Deserialize`)
  - `Metric` enum — `Euclidean`, `EuclideanKNN`, `Manhattan`, `Cosine`
- **Sparse training path** (`train_sparse`) — O(n·k) per epoch instead of O(n²):
  - Global KNN precomputed once on GPU
  - Edge subsampling — caps at 50K positive edges per epoch regardless of dataset size
  - Negative sampling — random non-neighbor pairs for repulsion term
  - Pre-batched index tensors — 16 shuffled batches uploaded to GPU once, cycled per epoch
  - Fused gather — 2 GPU `select()` calls instead of 4
  - Async loss readback — GPU→CPU sync every 5 epochs to avoid pipeline stalls
  - In-memory checkpointing — best model kept in RAM, no disk I/O during training
- **Crate comparison benchmark** (`examples/crate_comparison.rs`):
  - Benchmarks fast-umap vs umap-rs on 6 dataset sizes (500–20K samples)
  - Outputs `figures/crate_comparison.{json,svg,md}`
  - GPU warmup pass to exclude shader compilation from timings
- **MNIST benchmark** (`examples/bench_mnist.rs`):
  - Trains UMAP on 10K MNIST digits and generates `figures/mnist.png` + loss curve
- **Unified benchmark script** (`bench.sh`):
  - Runs all benchmarks: hardware micro-benchmarks, crate comparison, MNIST
  - Supports `--only <name>`, `--skip-mnist`, `--criterion` flags
  - Replaces old `generate_images.sh`
- **Prelude updated** — re-exports `UmapConfig`, `ManifoldParams`, `GraphParams`, `OptimizationParams`, `Metric`

### Changed

- **Default training path**: `Umap::fit()` uses sparse training (`train_sparse`); legacy `UMAP::fit()` still uses dense `train()` for backward compatibility
- **Examples no longer generate figures** — examples are lightweight smoke tests; use `./bench.sh` for figure generation
- `run_all.sh` simplified to only run examples (no figures); points to `bench.sh` for benchmarks
- Benchmark results and micro-benchmark tables in README updated to latest run

### Performance

Benchmarked on Apple M3 Max (fast-umap: 50 epochs GPU, umap-rs: 200 epochs CPU):

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|---------|
| 500 × 50 | 0.22s | 0.06s | 0.29× *(umap-rs faster)* |
| 1 000 × 50 | 0.81s | 0.12s | 0.15× *(umap-rs faster)* |
| 2 000 × 100 | 0.92s | 0.44s | 0.48× *(umap-rs faster)* |
| 5 000 × 100 | 1.62s | 2.27s | **1.4× faster** |
| 10 000 × 100 | 2.06s | 8.67s | **4.2× faster** |
| 20 000 × 100 | 3.72s | 34.22s | **9.2× faster** |

### Removed

- `generate_images.sh` — replaced by `bench.sh`

---

## [1.0.0] — 2026-02-23

### Added

- **burn 0.20.1** upgrade (from 0.18); cubecl 0.9 (from 0.6)
- **36 unit tests** — covering normalization, tensor conversion, model shape, distance math (all CPU-only via NdArray)
- **Hardware-tagged benchmarks** (`cargo run --release --bin bench_report`):
  - Auto-detects CPU & GPU
  - Writes `.md` + `.svg` to `benches/results/`
  - CPU vs GPU comparison chart
- **MNIST example** (`examples/mnist.rs`) — 10K digits projected to 2-D with coloured class labels
- **Hyperparameter benchmark** (`examples/mnist_benchmark.rs`) — sweep over learning rates, batch sizes, penalties, hidden layer configs
- **Charting** — `chart` module with scatter plots and loss curves via plotters
- **Labels in plots** — coloured by class when labels provided
- **Early stopping** — `patience`, `min_desired_loss`, `timeout` parameters
- **Distance metrics** — `Euclidean`, `EuclideanKNN`, `Manhattan`, `Cosine`
- **Custom CubeCL GPU kernels** — Euclidean pairwise distance and KNN

### Fixed

- `Backend::seed` signature updated for burn 0.20 (`(&device, seed)`)
- `as_tensor_arg` no longer takes a generic type parameter
- `NodeID` renamed to `NodeId`
- `ABSOLUTE_POS_X/Y` cast to `usize` for shape indexing in CubeCL kernels
- `normalize_tensor` replaced `.to_vec::<bool>()` with f32 arithmetic (WGPU stores bools as u32)
- All 4 `unused Result` warnings from kernel launches resolved with `.expect()`

---

## [0.1.0] — 2024-12-01

### Added

- Initial release
- GPU-accelerated UMAP via burn + CubeCL
- `UMAP` struct with `.fit()` and `.transform()`
- `UMAPModel` neural network with configurable hidden layers
- `TrainingConfig` builder
- Basic examples (simple, advanced)
