# fast-umap

GPU-accelerated parametric UMAP (Uniform Manifold Approximation and Projection)
in Rust, built on [burn](https://github.com/tracel-ai/burn) +
[CubeCL](https://github.com/tracel-ai/cubecl).

See [docs.rs](https://docs.rs/crate/fast-umap/latest) for the full API reference.

---

## Highlights

- Historical Apple Silicon MLX benchmark results are retained below for reference;
  MLX is no longer a supported runtime backend.
- **Up to 4.7× faster** than [umap-rs](https://crates.io/crates/umap-rs) on
  datasets ≥ 10 000 samples (see [benchmarks](#performance--fast-umap-vs-umap-rs) below)
- **Scales to 50K+ samples** — NN-Descent approximate KNN avoids the O(n^2) distance matrix
- **Parametric** — trains a neural network, so you can
  [`transform()`](#transform-new-data) new unseen data instantly
- **Runtime-selected devices** — Standard UMAP supports auto, CPU, or CUDA; parametric execution uses Burn devices
- **PCA warm-start** — optional PCA pre-training for faster convergence
- **Parametric Rust API** — use `ParametricUMAP::new(config).fit(data, labels)`

---

## Performance — fast-umap vs umap-rs

Benchmarked on Apple M3 Max. Both crates receive identical random data.
fast-umap runs 50 epochs (parametric, GPU); umap-rs runs 200 epochs
(classical SGD, CPU). Total time includes data prep + KNN + fit + extract.

![Crate comparison chart](figures/crate_comparison.svg)

| Dataset | fast-umap | umap-rs | Speedup |
|---------|-----------|---------|---------|
| 500 × 50 | 0.84s | 0.08s | 0.10× *(umap-rs faster)* |
| 1 000 × 50 | 2.19s | 0.12s | 0.05× *(umap-rs faster)* |
| 2 000 × 100 | 3.65s | 0.44s | 0.12× *(umap-rs faster)* |
| 5 000 × 100 | 6.75s | 2.31s | 0.34× *(umap-rs faster)* |
| **10 000 × 100** | **5.93s** | **8.68s** | **1.5× faster** 🚀 |
| **20 000 × 100** | **7.32s** | **34.10s** | **4.7× faster** 🚀 |

> **Crossover ≈ 10 000 samples.** Below that, umap-rs wins on raw CPU
> efficiency for small data. Above it, fast-umap pulls ahead and the gap
> widens with dataset size — umap-rs's brute-force KNN scales O(n²) while
> fast-umap's per-epoch cost is capped.

**Why fast-umap wins at scale:**

| Technique | Effect |
|-----------|--------|
| Sparse edge-based loss | O(n·k) per epoch instead of O(n²) |
| Integer-weighted positive draws | Caps positive draws/epoch at 50K without expanding repeated edges |
| Fresh positive and negative batches | Seeded CPU sampling and bounded index uploads each epoch |
| Fused index gather | 2 GPU `select()` calls instead of 4 |
| Async loss readback | GPU→CPU sync every 5 epochs, not every epoch |
| In-memory checkpointing | No disk I/O during training |
| GPU-accelerated KNN | Full n×n pairwise distance on GPU (one-time cost) |

**Reproduce:**

```shell
./bench.sh --only comparison
```

Or run all benchmarks at once (hardware + comparison + MNIST):

```shell
./bench.sh
```

---

## Historical benchmark — MLX vs WGPU (Apple Silicon)

These measurements were collected on an earlier version of fast-umap and are
preserved as historical results only. MLX has since been removed; these numbers
do not describe the current runtime or imply current performance.

![Backend benchmark chart](figures/backend_benchmark.svg)

| Scenario | Samples | Features | Hidden | Out | WGPU | MLX | Speedup |
|----------|---------|----------|--------|-----|------|-----|---------|
| Small | 500 | 20 | [64] | 2D | 2.49s | 0.10s | **MLX 25x** |
| Medium | 1K | 50 | [128] | 2D | 1.65s | 0.08s | **MLX 21x** |
| Standard | 5K | 100 | [128] | 2D | 4.94s | 0.34s | **MLX 15x** |
| Large | 10K | 100 | [128] | 2D | 6.13s | 0.78s | **MLX 7.8x** |
| XL | 20K | 100 | [128] | 2D | 7.37s | 2.22s | **MLX 3.3x** |
| High-dim | 2K | 500 | [256] | 2D | 7.43s | 0.27s | **MLX 28x** |
| MNIST-like | 5K | 784 | [256] | 2D | 8.69s | 0.71s | **MLX 12x** |
| Deep net | 5K | 100 | [256,128,64] | 2D | 9.93s | 0.44s | **MLX 23x** |
| 3-D output | 5K | 100 | [128] | 3D | 12.31s | 0.33s | **MLX 38x** |
| Huge | 50K | 100 | [128] | 2D | -- | 14.04s | MLX only |

> Historical measurements: 50 epochs per scenario on Apple Silicon. The 50K scenario used NN-Descent
> approximate KNN (`nn-descent` feature) — WGPU can't run it because the
> exact pairwise distance matrix would need 10 GB.

---

## What's New (v1.6.0)

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

| Area | Change |
|------|--------|
| **PCA warm-start** | Pre-trains the network to approximate PCA output before UMAP loss, improving convergence (feature: `pca`) |
| **NN-Descent** | Approximate KNN graph construction in O(n*k) memory — enables 50K+ sample datasets without the O(n^2) distance matrix (feature: `nn-descent`) |
| **Shared CubeCL ops** | Pairwise distance and KNN use shared kernels on the selected CPU, CUDA, or WGPU runtime; Standard UMAP supports CPU and CUDA |
| **Autodiff registration** | Custom-op gradients are registered separately in `autodiff_ops` for supported execution devices |

---

## Features

- **Dimensionality Reduction** — projects high-dimensional data to 2-D or 3-D for visualization
- **Parametric model** — learned neural network can project new, unseen data via `transform()`
- **Execution devices** — CubeCL LLVM CPU and CUDA for Standard UMAP; parametric device support is separate
- **PCA warm-start** — optional pre-training phase initializes embeddings from PCA for faster convergence (`pca` feature)
- **NN-Descent** — approximate KNN in O(n*k) memory for datasets > 10K samples, avoiding the O(n^2) distance matrix (`nn-descent` feature)
- **Automatic differentiation** — full autograd through custom kernels via burn's autodiff backend
- **Sparse training** — edge subsampling + negative sampling keeps per-epoch cost constant regardless of dataset size
- **Flexible architecture** — configurable hidden layers, output dims, distance metric, learning rate, early stopping, timeout
- **36 unit tests** — covering normalization, tensor conversion, model shape, distance math

---

## Installation

```shell
cargo add fast-umap
```

`Cargo.toml`:

```toml
[dependencies]
fast-umap  = "1.6.0"
burn       = { version = "=0.22.0-pre.4", features = ["wgpu", "autodiff", "autotune"] }
cubecl     = { version = "=0.11.0-pre.4", features = ["wgpu"] }
```

---

## Quick Start

```rust
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;

// 100 samples × 10 features
let data: Vec<Vec<f64>> = generate_test_data(100, 10)
    .chunks(10)
    .map(|c| c.iter().map(|&x: &f32| x as f64).collect())
    .collect();

// Fit parametric UMAP (default: 2-D output, Euclidean distance)
let config = ParametricUMAPConfig::default();
let umap = fast_umap::ParametricUMAP::with_device(
    config,
    Device::wgpu(DeviceKind::DefaultDevice),
);
let fitted = umap.fit(data.clone(), None);

// Get embedding
let embedding = fitted.embedding();
println!("Shape: {} × {}", embedding.len(), embedding[0].len());
```

---

## Python

The Python package exposes the parametric model and Standard UMAP. Standard UMAP
accepts `backend="auto"`, `backend="cpu"`, or `backend="cuda"`. `auto` prefers
a discovered usable CUDA device and selects CPU only when CUDA is unavailable;
explicit selections are strict and execution failures do not trigger fallback.

The Standard CPU backend uses CubeCL's LLVM runtime. The documented local target
is Linux x86-64, including WSL2. LLVM bundle acquisition may require network
access on the first build and uses a local cache; JIT compilation requires LLVM
support. This is not a claim of wheel availability or support for other platforms.

From the repository root, install in editable mode with `uv`:

```shell
uv sync
```

For a CPU-only local extension build, explicitly select the CPU feature. Avoid
running `uv sync` before tests, because syncing can replace the local extension:

```shell
uv run maturin develop --release --no-default-features --features cpu
uv run --no-sync pytest python/tests/test_standard_*.py
```

```python
import numpy as np
from fast_umap import ParametricUMAP

X = np.random.default_rng(42).normal(size=(100, 10)).astype(np.float32)
model = ParametricUMAP(device="cpu")  # choose "gpu" for GPU execution
embedding = model.fit_transform(X)
new_embedding = model.transform(X[:5])
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_neighbors` | `15` | Neighborhood size including self; at least 2, with 14 non-self neighbors by default. |
| `local_connectivity` | `1.0` | Finite non-negative local connectivity; fractional values interpolate rho. |
| `bandwidth` | `1.0` | Finite positive multiplier of the smooth-kNN target. |
| `set_op_mix_ratio` | `1.0` | Fuzzy intersection (0) to union (1); finite and in [0, 1]. |
| `n_components` | `2` | Number of embedding dimensions. |
| `metric` | `"euclidean"` | Input-space distance metric; only Euclidean is supported. |
| `n_epochs` | `None` (100 epochs) | Number of training epochs. |
| `learning_rate` | `0.001` | Adam optimizer learning rate. |
| `min_dist` | `0.1` | Minimum distance used by the low-dimensional curve. |
| `spread` | `1.0` | Effective scale of embedded points. |
| `repulsion_strength` | `1.0` | Weight of negative-sample repulsion. |
| `negative_sample_rate` | `5` | Number of negative samples per positive edge. |
| `random_state` | `None` | Integer seed, NumPy `RandomState` or `Generator`; CPU runs are bit-identical for a fixed seed. |
| `verbose` | `False` | Enable training progress output. |
| `device` | `"gpu"` | Execution device: `"gpu"` or `"cpu"`. |
| `hidden_sizes` | `(100,)` | Widths of the neural-network hidden layers. |
| `patience` | `None` | Early-stopping patience. |
| `timeout` | `None` | Training timeout in seconds. |

### Differences from umap-learn

- This is parametric UMAP: `transform` is a neural-network forward pass, not a
  nearest-neighbor interpolation. The returned `embedding_` uses `float32`.
- Only the Euclidean metric is supported. Supervised UMAP, densMAP, and
  `inverse_transform` are not supported. umap-learn parameters without a
  fast-umap equivalent (`init`, `target_*`, `low_memory`, ...) are not accepted
  and raise `TypeError`.
- `learning_rate` defaults to `0.001` (Adam), and `n_epochs` defaults to `100`.
- `n_neighbors >= n_samples` raises `ValueError` rather than warning and
  adjusting the neighbor count.
- The primary Rust and Python routes compute local fuzzy memberships and symmetric
  union/intersection weights. Configured positive epochs E determine pruning at
  `w < max(w)/E` and integer counts `floor(E*w)`, using float32 arithmetic.
  Each epoch independently draws `min(50_000, sum(counts))` positives with
  replacement, with probability `count/sum(counts)`, and fresh uniform non-self
  negatives. Empty positive mass is an error; no unweighted fallback is used.
  This matches graph-helper mathematics on identical neighbor lists, not the
  reference's finite-pass training schedule, negative law, loss, or optimizer.
- A fitted model is bound to the thread that created it. Ctrl-C is checked every
  few epochs and only on the main thread.
- With a fixed `random_state`, CPU execution is bit-identical.

### Python development

```shell
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
```

---

## Parametric devices and Standard UMAP backends

Parametric and Standard UMAP have separate backend contracts. The Burn-device
examples below describe the parametric API only; they do not establish
parametric CPU runtime acceptance for the CubeCL CPU migration.

### GPU Backend (WGPU) - Primary Recommended Backend

For GPU-accelerated execution (requires WGPU-compatible GPU):

```rust
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;

let config = ParametricUMAPConfig::default();
let estimator = ParametricUMAP::with_device(
    config,
    Device::wgpu(DeviceKind::DefaultDevice),
);
let fitted = estimator.fit(data, None);
```

**Features:**
- ✅ Full parametric UMAP with neural network training
- ✅ GPU acceleration via WGPU (Metal/Vulkan/DX12)
- ✅ Transform new data through trained model
- ✅ Custom GPU kernels for efficient computation
- ✅ Automatic differentiation for training

### Runtime device selection

`ParametricUMAP` and `FittedParametricUMAP` accept a Burn device. This API is
distinct from Standard UMAP backend selection:

```rust
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;

let config = ParametricUMAPConfig::default();
let cpu = ParametricUMAP::with_device(config.clone(), Device::cpu());
let gpu = ParametricUMAP::with_device(config, Device::wgpu(DeviceKind::DefaultDevice));
let fitted = cpu.fit(data, None);
```

The WGPU device requires the `gpu` feature. CUDA can also be selected through
Burn's runtime device API when compiled with the `cuda` feature.

## Feature-Based Compilation

fast-umap uses Cargo features to enable only the backends you need:

```shell
# Default features: cuda, cpu, verbose
cargo build --release

# Standard UMAP CubeCL LLVM CPU backend
cargo build --release --no-default-features --features cpu

# Optional PCA warm-start and approximate NN-Descent
cargo build --release --features pca,nn-descent

# All features (development)
cargo build --release --features all
```

See [`feature_demo` example](examples/feature_demo.rs) for complete feature matrix.

### Runtime Backend Selection

The [`backend_choice` example](examples/backend_choice.rs) demonstrates runtime backend selection:

```shell
# Run with GPU backend
cargo run --release --features gpu --example backend_choice -- wgpu

# Run with CPU backend
cargo run --release --no-default-features --features cpu --example backend_choice -- cpu
```

### CPU Testing Examples

For working CPU examples, see the test suite:

```shell
# Run CPU-based tests
cargo test --lib
```

Standard UMAP CPU behavior should be checked with its feature-specific tests;
parametric tests do not establish CubeCL CPU runtime acceptance.

---

## Standard UMAP

`UMAP` / `FittedUMAP` (Rust) and `fast_umap.UMAP` (Python) are a separate,
non-parametric estimator that follows umap-learn 0.5.12: smooth-kNN fuzzy graph,
spectral (or random) initialization, and direct coordinate SGD with the
reference Tausworthe negative sampling and epoch schedule. `ParametricUMAP` is
unchanged.

```rust
use fast_umap::{BackendChoice, NeighborSearch, UMAP, UMAPConfig};

let config = UMAPConfig { random_state: Some(42), ..UMAPConfig::default() };
let fitted = UMAP::new(config, BackendChoice::Auto, NeighborSearch::Auto).fit(&data)?;
let embedding = fitted.embedding();          // Vec<Vec<f32>>, one row per sample
let projected = fitted.transform(&new_data)?; // place unseen rows in the fitted map
```

```python
import fast_umap

model = fast_umap.UMAP(n_neighbors=15, random_state=42, backend="auto")
embedding = model.fit_transform(X)
projected = model.transform(X_new)
```

- **Backends:** `BackendChoice::{Auto, Cpu, Cuda { device_index }}`. `Auto`
  prefers discovered usable CUDA and selects CPU only when CUDA is unavailable.
  Unexpected discovery errors propagate; explicit selections are strict, and
  construction or execution failures never cause backend substitution. WGPU
  is not newly supported for Standard UMAP. Spectral initialization is host-side.
- **Neighbor search:** `Exact` (blocked exact search, no N x N matrix),
  `Approximate` (NN-Descent for fit, bounded graph search for transform), or
  `Auto` (`Exact` up to `AUTO_EXACT_MAX_ROWS` rows, `Approximate` above).
  `transform_queue_size` (default 4.0) controls approximate transform effort and
  is rejected for exact search.
- **Errors:** invalid input or configuration returns `UmapError`; Python raises
  `ValueError` (input/config) or `RuntimeError` (backend/numerical). As in
  umap-learn, a one-row fit yields a zero embedding and cannot transform, and a
  graph with no edges (for example two rows) is an error.
- **Exact unique fitting:** set `UMAPConfig { unique: true, ..Default::default() }`
  or Python `UMAP(unique=True)` to fit first-occurrence distinct input rows.
  The default remains false. Signed zeros compare equal; nonfinite inputs are
  rejected before deduplication. Returned training embeddings retain original
  row order, with identical coordinates for identical rows, while transform
  uses the distinct training graph. Unique fits need at least two distinct rows;
  the existing empty-graph error still applies to a two-row training graph.
- **Quality:** previously recorded backend quality measurements do not establish
  CubeCL CPU acceptance; no new CPU quality claim is made here.
- **Spectral solver:** small components use faer 0.24.4 dense self-adjoint EVD;
  larger components use locked-trivial block thick-restart Krylov iteration
  with cached operator images and faer Rayleigh–Ritz EVD. The sparse iteration has a fixed numerical budget,
  not a wall-clock guarantee. The failed single-start faer matrix-free path is
  removed. Numerical regression harnesses use child-process timeouts and fail
  on timeout or fallback; there is no production watchdog. Reference-relative
  runtime and separately approved quality margins remain acceptance gates;
  passing numerical or older quality tests alone does not establish acceptance.

### Profiling standard UMAP

The optional `tracing` feature instruments standard UMAP phases and enables
Burn/CubeCL tracing. The library does not install a subscriber. Profiling benches
install one only when `FAST_UMAP_TRACE` is set; leave it unset for timing runs.

```sh
cargo bench --no-default-features --features cpu --bench standard_umap_cpu
cargo bench --no-default-features --features cuda --bench standard_umap_cuda

# Chrome/Perfetto trace (import the JSON file into ui.perfetto.dev)
FAST_UMAP_TRACE=standard-cpu.json RUST_LOG=fast_umap=trace cargo bench \
  --no-default-features --features cpu,tracing --bench standard_umap_cpu -- --test
FAST_UMAP_TRACE=standard-cuda.json RUST_LOG=fast_umap=trace,cubecl=trace,cubecl_cuda=trace cargo bench \
  --no-default-features --features cuda,tracing --bench standard_umap_cuda -- --test

# Console span timings instead of a trace file
FAST_UMAP_TRACE=console RUST_LOG=fast_umap=debug cargo bench \
  --no-default-features --features cpu,tracing --bench standard_umap_cpu -- --test
```

CUDA requires a working NVIDIA runtime. Traces are
host-side spans, not GPU kernel timings. Fit stage boundaries explicitly synchronize
the selected device when `tracing` is enabled; enabling detailed tracing adds
overhead. The benches measure complete fit and transform calls, including host
embedding readback, with deterministic inputs prepared outside the timed loop.

Compare standard UMAP with **umap-learn 0.5.12**, using CUDA only:

```sh
cargo build --release --no-default-features --features cuda,tracing --example profile_standard_cuda
uv run --no-sync python scripts/compare_umap_spans.py --output comparison.json --trace cuda-trace.json
```

The build is separate from the 115-second run deadline. Defaults are 512 samples,
16 features, 30 epochs, 15 exact neighbors, `min_dist=0.0`, random initialization
and seed 42.
Both implementations consume identical float32 inputs and warm up before timing.
Python timings wrap actual reference callsites without copying its implementation.
Initialization cannot be isolated from embedding setup; nested neighbor selection
is subtracted from Python fuzzy-graph time. Fit CUDA spans include device barriers,
so queued GPU work is not charged to the host spectral solver or graph export.
Transform spans remain host-side dispatch spans.

The maintained evaluator emits provenance, fixed-representative quality metrics,
uncertainty, explicit gate results, and full-range/robust-percentile HTML plots:

```sh
uv run --no-sync python comparison/evaluate.py --input samples.f32le \
  --rows 15000 --features 128 --device cuda --seeds 42,43,44,45,46 \
  --output comparison-results.json
```

Quality gates without explicitly supplied thresholds remain unevaluated. Native
spectral fallback warnings and subprocess timeouts fail the comparison. The CUDA
profiler persists synchronized stage times and separate warmup wall times in
`result.json`. Its compilation duration sums CubeCL's observed artifact-acquisition
records, which include compilation/cache/persistence/device load and exclude kernel
execution; it overlaps rather than subtracts from independent full warmup wall times.
Per-phase outcome counts and coverage evidence are reported. Missing, corrupt, or
uncertain retained records make compilation timing unavailable, never a measured zero.
The JSON reports inclusive spans, counts, residual time and explicitly labeled
host-span ratios, not per-kernel GPU speedups. Rows are limited to 16–4095 to keep
the reference on its exact-search branch. Missing CUDA fails without CPU fallback.

---

## API Overview

The public API is explicitly parametric; graph and manifold types are shared
with graph utilities:

| Type | Description |
|------|-------------|
| [`ParametricUMAP`](https://docs.rs/fast-umap/latest/fast_umap/struct.ParametricUMAP.html) | Backend-independent neural estimator — `ParametricUMAP::new(config)` or `ParametricUMAP::with_device(config, device)` |
| [`FittedParametricUMAP`](https://docs.rs/fast-umap/latest/fast_umap/struct.FittedParametricUMAP.html) | Fitted neural model — `.embedding()`, `.transform()`, `.into_embedding()`, `.config()` |
| [`ParametricUMAPConfig`](https://docs.rs/fast-umap/latest/fast_umap/struct.ParametricUMAPConfig.html) | Estimator configuration with shared graph/manifold settings and parametric optimization settings |
| [`ManifoldParams`](https://docs.rs/fast-umap/latest/fast_umap/struct.ManifoldParams.html) | `min_dist`, `spread` — control cluster tightness and separation |
| [`GraphParams`](https://docs.rs/fast-umap/latest/fast_umap/struct.GraphParams.html) | `n_neighbors`, `local_connectivity`, `bandwidth`, `set_op_mix_ratio`, `metric`, `normalized`, `minkowski_p` |
| [`ParametricUMAPOptimizationParams`](https://docs.rs/fast-umap/latest/fast_umap/struct.ParametricUMAPOptimizationParams.html) | `n_epochs`, `learning_rate`, `patience`, `timeout`, `verbose`, `neg_sample_rate`, … |
| [`Metric`](https://docs.rs/fast-umap/latest/fast_umap/enum.Metric.html) | Shared graph distance metric settings |

### Configuration

Use `ParametricUMAPConfig` for estimator settings, with shared `GraphParams`
and `ManifoldParams` plus `ParametricUMAPOptimizationParams`. Model architecture
is described by the `ParametricUMAPModelConfig` builder; loss reduction and
epoch callbacks are named `ParametricUMAPLossReduction` and
`ParametricUMAPEpochProgress`.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 2 | Output dimensionality (2-D or 3-D) |
| `hidden_sizes` | `[100, 100, 100]` | Neural network hidden layer sizes |
| `min_dist` | 0.1 | Min distance in embedding — smaller = tighter clusters |
| `spread` | 1.0 | Effective scale of embedded points |
| `n_neighbors` | 15 | KNN neighborhood size including self (primary route) |
| `local_connectivity` | 1.0 | Local rho interpolation, finite and non-negative |
| `bandwidth` | 1.0 | Smooth-kNN target multiplier, finite and positive |
| `set_op_mix_ratio` | 1.0 | Fuzzy intersection (0) to union (1) |
| `n_epochs` | 200 | Training epochs |
| `learning_rate` | 1e-3 | Adam step size |
| `batch_size` | 1 000 | Samples per batch |
| `penalty` | 0.0 | L2 weight decay |
| `metric` | `Euclidean` | Distance metric |
| `repulsion_strength` | 1.0 | Repulsion term weight |
| `neg_sample_rate` | 5 | Negative (repulsion) samples per positive edge per epoch |
| `cooldown_ms` | `0` | Sleep N ms between epochs to reduce GPU utilisation (0 = disabled) |
| `patience` | `None` | Early-stop epochs without improvement |
| `min_desired_loss` | `None` | Stop when loss ≤ threshold |
| `timeout` | `None` | Hard time limit (seconds) |
| `verbose` | `true` | Progress bar + loss plots |

### Transform new data

Because fast-umap is *parametric* (neural network), it can project new
unseen data — something classical UMAP cannot do:

The Rust `ParametricUMAP` API and Python `ParametricUMAP` pass feature values through as
provided; they do not fit or apply feature scaling. Preprocessing is optional
and caller-owned. If you scale features, fit the scaler on training data and
apply that same transform to both the training data and every inference batch.
Feature scales affect distances and therefore the neighborhood geometry and
embedding.

```rust
let fitted = umap.fit(training_data, None);

// Project new data through the trained model
let new_embedding = fitted.transform(new_data);
```

---

## Examples

### CPU Capabilities Demo

Demonstrates CPU backend capabilities for utility functions and tensor operations:

```shell
cargo run --release --example cpu_training_demo
```

### Backend Choice — CPU vs GPU selection

Demonstrates selecting configured devices for the parametric model; this example
does not document or verify Standard UMAP CPU support:

```shell
# Run with WGPU backend
cargo run --release --features gpu --example backend_choice -- wgpu

# Run with CPU backend
cargo run --release --no-default-features --features cpu --example backend_choice -- cpu
```

This example selects Burn devices for the parametric model. Its CPU example is
not acceptance evidence for Standard UMAP or a claim of parametric CPU support.

### Simple — random data, 2-D embedding

```shell
cargo run --release --example simple
```

```rust
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;

let config = ParametricUMAPConfig::default();
let estimator = ParametricUMAP::with_device(config, Device::wgpu(DeviceKind::DefaultDevice));
let fitted = estimator.fit(data.clone(), None);
let embedding = fitted.embedding();
let new_embedding = fitted.transform(data);
```

### Advanced — 1 000 samples, custom config

```shell
cargo run --release --example advanced
```

```rust
use burn::tensor::{Device, DeviceKind};
use fast_umap::prelude::*;

let config = ParametricUMAPConfig::default();
let estimator = ParametricUMAP::with_device(
    config,
    Device::wgpu(DeviceKind::DefaultDevice),
);
let fitted: FittedParametricUMAP = estimator.fit(data, None);
let embedding = fitted.embedding();
```

### MNIST — 10 000 hand-written digits projected to 2-D

```shell
cargo run --release --example mnist         # quick run (no figures)
./bench.sh --only mnist                     # generate figures
```

Downloads the MNIST dataset on first run (~12 MB), trains UMAP on 10K
digits (784 features → 2-D).

| 2-D digit embedding (coloured by class) | Loss curve |
|:----------------------------------------:|:----------:|
| ![MNIST embedding](figures/mnist.png) | ![MNIST loss](figures/losses_model.png) |

```rust
let config = ParametricUMAPConfig::default();
let estimator = ParametricUMAP::with_device(config, Device::wgpu(DeviceKind::DefaultDevice));
let fitted = estimator.fit(data, Some(labels));
let embedding = fitted.embedding();
```

---

## Generating Figures

All figures are generated by the benchmark suite, not by the examples.
Examples are lightweight smoke tests that verify correctness without
writing any files.

```shell
# Run all benchmarks and generate all figures
./bench.sh

# Run only specific benchmarks
./bench.sh --only comparison          # fast-umap vs umap-rs chart
./bench.sh --only mnist               # MNIST embedding + loss curve
./bench.sh --only hardware            # CPU + GPU micro-benchmarks
./bench.sh --only mnist comparison    # combine multiple

# Skip MNIST (saves ~70s)
./bench.sh --skip-mnist

# Include Criterion statistical suite
./bench.sh --criterion
```

### What generates what

| Benchmark | Output |
|-----------|--------|
| `comparison` | `figures/crate_comparison.{svg,json,md}` |
| `mnist` | `figures/mnist.png`, `figures/losses_model.png` |
| `hardware` | `benches/results/{cpu,gpu,comparison}_*.{svg,md}` |
| `criterion` | `target/criterion/` (HTML reports) |

### Run examples (no figures)

```shell
./run_all.sh                  # all examples
./run_all.sh --skip-mnist     # skip MNIST download
```

### CPU Capabilities Demo

Demonstrates CPU backend capabilities:

```shell
# Run CPU training demo (parametric behavior is outside Standard CPU acceptance)
cargo run --release --example cpu_training_demo
```

### Developer Experience Improvements

The feature-based compilation system provides:

**✅ Faster Compilation**
- Minimal builds: `cargo build --release --no-default-features`
- Feature-specific examples and documentation

**✅ Smaller Binaries**
- GPU-only: exclude CPU dependencies (save ~5MB)
- CPU-only Standard builds select the `cpu` feature without the `gpu` feature
- Library mode: minimal footprint for integration

**✅ Clear Feature Matrix**

| Feature | Description | When to Use |
|---------|-------------|--------------|
| `gpu` | GPU backend (WGPU) | Cross-platform GPU, large datasets |
| `cuda` | GPU backend (CUDA) | NVIDIA GPU and CUDA runtime libraries (`libcuda`, `libnvrtc`) at runtime; `nvcc`/CUDA toolkit optional at build time; excluded from `all` and default features |
| `pca` | PCA warm-start for embeddings | Better convergence quality |
| `nn-descent` | Approximate KNN for large data | Datasets > 10K samples (avoids O(n^2) memory) |
| `verbose` | Progress output | Development, debugging |
| `plotters` | Visualization | Exploration, analysis |
| `all` | All optional features except `cuda` | Development, testing |

**✅ Better Error Messages**
- Clear feature requirements at compile time
- Graceful runtime fallback handling
- Comprehensive validation

See [`feature_demo` example](examples/feature_demo.rs) for complete feature guide.

---

## Architecture

`ParametricUMAP` learns a feed-forward neural encoder; `FittedParametricUMAP`
uses that model to transform unseen samples. Its sparse graph/training schedule,
loss aggregation, negative sampling, and optimizer are not algorithmically
identical to umap-learn; see the [parity report](python_parity_report.md) for
the audited caveats.

### Training pipeline

```
Input data [n, features]
    │
    ├─ n < 10K ──→ GPU pairwise distance → exact KNN (O(n²))
    │
    ├─ n ≥ 10K ──→ NN-Descent approximate KNN (O(n·k))  [nn-descent]
    │
    ▼
Self-inclusive KNN → local fuzzy memberships → symmetric fuzzy graph
    │
    ├─── [pca] PCA warm-start: pre-train network to match PCA output
    │
    ▼
┌─── Per epoch (cost: O(min(n·k, 50K))) ───┐
│  Forward pass: data → neural net → [n, 2] │
│  Fresh integer-weighted positive draws    │
│  Fresh uniform non-self negative draws    │
│  UMAP cross-entropy loss                   │
│  Backward pass + Adam optimizer step       │
└────────────────────────────────────────────┘
    │
    ▼
FittedParametricUMAP with .embedding() and .transform()
```

### Modules

| Module | Feature | Description |
|--------|---------|-------------|
| [`parametric_umap`] | -- | Parametric estimator/model, private trainer, and weight persistence |
| [`graph`] | -- | Shared graph, metric, and manifold math/configuration |
| [`backend`] | -- | Backend trait extension for custom kernel dispatch |
| [`autodiff_ops`] | -- | Autodiff backward registration for custom ops |
| [`kernels`] | `cpu`, `cuda`, or `gpu` | Shared CubeCL kernels; Standard UMAP kernels require `cpu` or `cuda` |
| [`pca`] | `pca` | PCA warm-start for initial embedding |
| [`nn_descent`] | `nn-descent` | NN-Descent approximate KNN graph construction |
| [`chart`] | `plotters` | 2-D scatter plots and loss curves |
| [`distances`] | -- | CPU-side distance functions (Euclidean, cosine, Minkowski) |
| [`utils`] | -- | Data generation, tensor conversion, normalisation |
| [`prelude`] | -- | Re-exports of the most commonly used items |

---

## Convenience API

The convenience `ParametricUMAP` API is nongeneric and uses the default Burn device:

```rust
use fast_umap::prelude::*;

let fitted = fast_umap::parametric_umap(data.clone());
let embedding = fitted.embedding();
let transformed = fitted.transform(data);
```

For explicit device selection or configuration, use `ParametricUMAP::with_device(config, device)`.

---

## Testing

The Standard CPU test lane uses CubeCL LLVM and does not require a GPU. Parametric
test behavior is separate and is not acceptance evidence for Standard CPU.

```shell
cargo test
```

| Category | What is covered |
|----------|----------------|
| `normalize_data` | correctness, zero-mean/unit-std, constant columns |
| `format_duration` | zero, seconds, minutes, hours |
| `generate_test_data` | shape, bounds `[0, 1)` |
| `tensor_convert` | round-trip `Vec → Tensor → Vec`, NaN → 0 |
| `normalize_tensor` | output in `[0, 1]`, constant-input safety |
| `layer_normalize` | no NaN, shape preserved |
| `ParametricUMAPModelConfig` builder | defaults and custom values |
| `ParametricUMAPOptimizationParams` | parametric optimizer settings |
| `ParametricUMAPModel` | neural model output shape and inference |
| Distance math | Euclidean self=0, 3-4-5 triangle, symmetry; Manhattan |

---

## Micro-benchmarks

**Reproduce:** `./bench.sh --only hardware`

Full detail files:
[cpu\_apple\_m3\_max.md](benches/results/cpu_apple_m3_max.md) ·
[gpu\_apple\_m3\_max.md](benches/results/gpu_apple_m3_max.md) ·
[comparison\_apple\_m3\_max.md](benches/results/comparison_apple_m3_max.md) ·
[cpu\_apple\_silicon\_aarch64.md](benches/results/cpu_apple_silicon_aarch64.md)

### Historical CPU benchmark — Apple M3 Max (NdArray backend)

![CPU benchmark chart](benches/results/cpu_apple_m3_max.svg)

| Benchmark | Input | Min | **Mean** | Max |
|-----------|-------|-----|----------|-----|
| `normalize_data` | 100×10 | 345 µs | **517 µs** | 986 µs |
| `normalize_data` | 500×30 | 1.92 ms | **2.31 ms** | 2.77 ms |
| `normalize_data` | 1 000×50 | 4.35 ms | **4.78 ms** | 5.79 ms |
| `normalize_data` | 5 000×100 | 16.4 ms | **18.2 ms** | 19.4 ms |
| `generate_test_data` | 100×10 | 3.62 µs | **4.03 µs** | 5.92 µs |
| `generate_test_data` | 500×30 | 56.3 µs | **57.8 µs** | 61.9 µs |
| `generate_test_data` | 1 000×50 | 246 µs | **258 µs** | 288 µs |
| `generate_test_data` | 5 000×100 | 2.46 ms | **2.48 ms** | 2.54 ms |
| `tensor_convert` | 100×10 | 5.08 µs | **5.16 µs** | 5.33 µs |
| `tensor_convert` | 500×30 | 32.4 µs | **33.0 µs** | 35.8 µs |
| `tensor_convert` | 1 000×50 | 74.6 µs | **78.1 µs** | 89.5 µs |
| `model_forward` | 16s×10f \[32\]→2 | 21.4 µs | **34.5 µs** | 55.8 µs |
| `model_forward` | 64s×50f \[64\]→2 | 26.7 µs | **34.2 µs** | 51.2 µs |
| `model_forward` | 128s×50f \[128\]→2 | 55.9 µs | **58.6 µs** | 64.2 µs |
| `model_forward` | 64s×100f \[128,64\]→3 | 70.0 µs | **80.3 µs** | 106 µs |
| `model_forward` | 256s×100f \[256,128\]→2 | 279 µs | **293 µs** | 310 µs |
| `normalize_tensor` | n=64 | 1.88 µs | **1.95 µs** | 2.29 µs |
| `normalize_tensor` | n=512 | 2.79 µs | **2.86 µs** | 3.00 µs |
| `normalize_tensor` | n=4 096 | 9.50 µs | **9.59 µs** | 10.3 µs |
| `normalize_tensor` | n=32 768 | 70.0 µs | **70.3 µs** | 70.7 µs |
| `layer_normalize` | 32×16 | 3.50 µs | **3.62 µs** | 3.96 µs |
| `layer_normalize` | 128×64 | 19.7 µs | **20.0 µs** | 20.9 µs |
| `layer_normalize` | 512×128 | 115 µs | **117 µs** | 130 µs |
| `layer_normalize` | 1 000×256 | 412 µs | **420 µs** | 454 µs |

### GPU — Apple M3 Max (WGPU / Metal)

![GPU benchmark chart](benches/results/gpu_apple_m3_max.svg)

| Benchmark | Input | Min | **Mean** | Max |
|-----------|-------|-----|----------|-----|
| `model_forward` | 16s×10f \[32\]→2 | 408 µs | **617 µs** | 894 µs |
| `model_forward` | 64s×50f \[64\]→2 | 430 µs | **481 µs** | 776 µs |
| `model_forward` | 128s×50f \[128\]→2 | 432 µs | **475 µs** | 576 µs |
| `model_forward` | 64s×100f \[128,64\]→3 | 549 µs | **688 µs** | 1.82 ms |
| `model_forward` | 256s×100f \[256,128\]→2 | 631 µs | **691 µs** | 828 µs |
| `model_forward` | 512s×100f \[256,128\]→2 | 926 µs | **1.08 ms** | 1.42 ms |
| `normalize_tensor` | n=512 | 572 µs | **695 µs** | 1.28 ms |
| `normalize_tensor` | n=4 096 | 590 µs | **662 µs** | 821 µs |
| `normalize_tensor` | n=32 768 | 629 µs | **712 µs** | 883 µs |
| `normalize_tensor` | n=262 144 | 1.08 ms | **1.12 ms** | 1.22 ms |
| `layer_normalize` | 128×64 | 437 µs | **471 µs** | 609 µs |
| `layer_normalize` | 512×128 | 467 µs | **500 µs** | 648 µs |
| `layer_normalize` | 1 000×256 | 617 µs | **662 µs** | 777 µs |
| `layer_normalize` | 4 000×512 | 1.81 ms | **1.93 ms** | 2.15 ms |

### CPU vs GPU — Apple M3 Max

![CPU vs GPU comparison](benches/results/comparison_apple_m3_max.svg)

| Benchmark | Input | CPU | GPU | Speedup |
|-----------|-------|-----|-----|---------|
| `model_forward` | 16s×10f \[32\]→2 | **34.5 µs** | 617 µs | 0.06× *(CPU faster)* |
| `model_forward` | 64s×50f \[64\]→2 | **34.2 µs** | 481 µs | 0.07× *(CPU faster)* |
| `model_forward` | 128s×50f \[128\]→2 | **58.6 µs** | 475 µs | 0.12× *(CPU faster)* |
| `model_forward` | 64s×100f \[128,64\]→3 | **80.3 µs** | 688 µs | 0.12× *(CPU faster)* |
| `model_forward` | 256s×100f \[256,128\]→2 | **293 µs** | 691 µs | 0.42× *(CPU faster)* |
| `normalize_tensor` | n=512 | **2.86 µs** | 695 µs | 0.00× *(CPU faster)* |
| `normalize_tensor` | n=4 096 | **9.59 µs** | 662 µs | 0.01× *(CPU faster)* |
| `normalize_tensor` | n=32 768 | **70.3 µs** | 712 µs | 0.10× *(CPU faster)* |
| `layer_normalize` | 128×64 | **20.0 µs** | 471 µs | 0.04× *(CPU faster)* |
| `layer_normalize` | 512×128 | **117 µs** | 500 µs | 0.23× *(CPU faster)* |
| `layer_normalize` | 1 000×256 | **420 µs** | 662 µs | 0.63× *(CPU faster)* |

> **Note:** WGPU/Metal has a fixed dispatch overhead of ~400–700 µs per kernel
> call. For the small model sizes above, that overhead dominates. The GPU wins
> in full UMAP training loops over thousands of samples where operations are
> chained without intermediate CPU readbacks.

### Running benchmarks

```shell
# All benchmarks (hardware + comparison + MNIST)
./bench.sh

# + Criterion statistics (~5 min)
./bench.sh --criterion

# Just hardware micro-benchmarks
./bench.sh --only hardware
```

---

## Roadmap

- [x] MNIST dataset example with intermediate plots
- [x] Charting behind a feature flag
- [x] Labels in plots
- [x] Batching + accumulated gradient
- [x] CubeCL kernels for distance computation
- [x] Hyperparameter testbench (`patience` vs `n_features` vs `epochs` …)
- [x] Unit tests (36) and hardware benchmarks
- [x] Historical v1.6.0 API mirrored umap-rs; current API is explicitly parametric (`ParametricUMAP`, `FittedParametricUMAP`, `ParametricUMAPConfig`)
- [x] Sparse training with edge subsampling + negative sampling
- [x] Crate comparison benchmark (fast-umap vs umap-rs)
- [x] Historical MLX backend benchmark (runtime backend removed)
- [x] PCA warm-start for initial embedding
- [x] Approximate KNN (NN-descent) for datasets > 10K

---

## License

MIT — see [LICENSE](LICENSE).

## Copyright

2024-2026, Eugene Hauptmann

## Citations

If you use fast-umap in research or a project, please cite the original UMAP
paper, this repository, and acknowledge the Burn and CubeCL frameworks:

### fast-umap

```bibtex
@software{hauptmann2024fastumap,
  title   = {fast-umap: GPU-Accelerated UMAP in Rust},
  author  = {Hauptmann, Eugene},
  year    = {2024},
  url     = {https://github.com/eugenehp/fast-umap},
  version = {1.6.0}
}
```

> Hauptmann, E. (2024). *fast-umap: GPU-Accelerated UMAP in Rust* (v1.6.0).
> <https://github.com/eugenehp/fast-umap>

### UMAP algorithm

```bibtex
@article{mcinnes2018umap,
  title   = {{UMAP}: Uniform Manifold Approximation and Projection for Dimension Reduction},
  author  = {McInnes, Leland and Healy, John and Melville, James},
  journal = {arXiv preprint arXiv:1802.03426},
  year    = {2018},
  url     = {https://arxiv.org/abs/1802.03426}
}
```

> McInnes, L., Healy, J., & Melville, J. (2018).
> *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.*
> arXiv:1802.03426. <https://arxiv.org/abs/1802.03426>

### Burn deep-learning framework

```bibtex
@software{burn2024,
  title   = {Burn: A Flexible and Comprehensive Deep Learning Framework in Rust},
  author  = {{Tracel AI}},
  year    = {2024},
  url     = {https://github.com/tracel-ai/burn}
}
```

### CubeCL GPU compute framework

```bibtex
@software{cubecl2024,
  title  = {CubeCL: Multi-platform GPU Compute Language for Rust},
  author = {{Tracel AI}},
  year   = {2024},
  url    = {https://github.com/tracel-ai/cubecl}
}
```

---

## Thank you

Inspired by the original UMAP [paper](https://arxiv.org/abs/1802.03426).
