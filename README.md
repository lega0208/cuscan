# fast-umap

Standard (non-parametric) UMAP (Uniform Manifold Approximation and Projection)
in Rust with a Python extension, built on [burn](https://github.com/tracel-ai/burn) +
[CubeCL](https://github.com/tracel-ai/cubecl). It follows umap-learn 0.5.12
semantics and runs on the CubeCL LLVM CPU runtime or CUDA.

See [docs.rs](https://docs.rs/crate/fast-umap/latest) for the full API reference.

---

## Installation

```toml
[dependencies]
fast-umap = { version = "1.6.0", default-features = false, features = ["cpu"] }
```

| Feature | Description |
|---------|-------------|
| `cpu` | CubeCL LLVM CPU runtime (default) |
| `cuda` | NVIDIA CUDA runtime (default) |
| `tracing` | Instrument fit/transform phases and enable Burn/CubeCL tracing |
| `cubecl-ir-dump` | Dump CubeCL CPU IR for kernel inspection |

At least one of `cpu` or `cuda` must be enabled.

---

## Python

`fast_umap.UMAP` accepts `backend="auto"`, `backend="cpu"`, or `backend="cuda"`.
`auto` prefers a discovered usable CUDA device and selects CPU only when CUDA is
unavailable; explicit selections are strict and execution failures do not
trigger fallback.

The Standard CPU backend uses CubeCL's LLVM runtime. The documented local target
is Linux x86-64, including WSL2. LLVM bundle acquisition may require network
access on the first build and uses a local cache; JIT compilation requires LLVM
support. This is not a claim of wheel availability or support for other platforms.

CubeCL caches autotune results, CPU throughput calibration and compiled CUDA
kernels on disk: in `target/environment/` when the working directory is inside a
Cargo workspace, otherwise in the user cache directory (`~/.cache/cubecl` on
Linux). The first fit on a machine pays a one-time calibration (several seconds on
CPU); later processes reuse it. CPU kernels are still JIT-compiled once per process.

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
import fast_umap

X = np.random.default_rng(42).normal(size=(100, 10)).astype(np.float32)
model = fast_umap.UMAP(n_neighbors=15, random_state=42, backend="cpu")
embedding = model.fit_transform(X)
new_embedding = model.transform(X[:5])
```

### Fitted model serialization

HDF5 persistence is an optional Python feature; install it with
`uv sync --extra serialization` or `pip install 'fast-umap[serialization]'`.
The extra provides `h5py` and `hdf5plugin`; normal fit/transform use does not
import either dependency.

```python
model = fast_umap.UMAP(random_state=42).fit(X)
model.save("fitted-model.h5")  # refuses to replace an existing path
restored = fast_umap.UMAP.load("fitted-model.h5", backend="auto")
restored.transform(X_new)
```

`save(path, overwrite=False)` writes schema v1 to one HDF5 file. Passing
`overwrite=True` explicitly replaces an existing destination. Publication uses
a same-directory temporary file and atomic no-clobber hard-link commit by
default (`os.replace` when overwriting); on the documented Linux target the
directory is synced after commit. A failure syncing after commit reports
uncertain durability and does not roll back the published file.

The versioned schema stores typed numeric fitted state (training data,
embedding, approximate-search query graph when applicable, and optional
deduplication row mapping), configuration and curve coefficients. It uses
lossless Blosc2 Zstandard compression (level 1) with byte shuffle in
32768-row chunks, not pickle or lossy
quantization. Exact serialized array/curve numeric state is retained; this does
not promise identical execution across backend, hardware, or software versions.
Bitwise transform equivalence is intended only for the same backend; cross-
backend behavior follows the existing numerical tolerances. The backend and
device are deliberately not persisted. `load(path, backend="auto",
device_index=0)` selects the runtime: `auto` uses the normal CUDA-then-CPU
selection policy, while explicit `cpu`/`cuda` is strict and explicit CUDA uses
the caller's `device_index` (default 0).

Files contain training data and derived model state and may expose sensitive
information. Protect them with the same access controls as the original data.
External readers need the Blosc2 HDF5 filter plugin (filter ID 32026), such as
`hdf5plugin`, registered with their HDF5 library. The Rust crate and extension
do not link HDF5 and provide no Rust HDF5 save/load API.

The storage benchmark compares compression levels and chunk row sizes for
continuous and integer-valued f32 features, embeddings, graphs, and a
deduplicated-model state. It reports file size, isolated save/load latency and
peak host-process RSS; it does not measure device memory. One quick run on
Linux x86-64/WSL2 with Python 3.14, using synthetic 4,096-row, 32-feature data,
gave the following results at 1,024-row chunks (latency in milliseconds, RSS
in MiB):

| Synthetic case | Blosc2 level | File bytes | Save ms | Load ms | Save RSS MiB | Load RSS MiB |
|---|---:|---:|---:|---:|---:|---:|
| Continuous f32 features | 1 | 474,450 | 29.25 | 5.55 | 61.1 | 55.5 |
| Continuous f32 features | 9 | 457,808 | 94.22 | 5.91 | 62.8 | 55.8 |
| Deduplicated model state | 1 | 305,631 | 33.88 | 8.90 | 60.3 | 55.0 |
| Deduplicated model state | 9 | 298,053 | 113.92 | 2.16 | 62.9 | 55.2 |

In these single-trial quick measurements, level 9 reduced file size by about
2.9–3.0% versus level 1, but took about 3.3–3.4× as long to save and used up to
1.7 MiB more peak RSS. A 1,024-row chunk was the smallest of the tested chunk
sizes in both rows. This supports level 1 with 1,024-row chunks as the
size/save-time tradeoff; production settings should still be evaluated on the
target workload. These measurements are synthetic and single-trial, and
latency (especially load latency) is noisy; RSS includes interpreter and
dependency baseline. Device memory was unavailable because the benchmark uses
host-side NumPy and h5py only. Run the full sweep and retain its JSON output
before drawing workload-specific conclusions.

```shell
uv run --no-sync python scripts/benchmark_serialization.py --quick
uv run --no-sync python scripts/benchmark_serialization.py --output serialization-benchmark.json
```

### Python development

```shell
uv sync
uv run maturin develop --release --no-default-features --features cpu
uv run --no-sync pytest python/tests
uv run ruff check
uv run ruff format --check
uv run ty check
```

---

## Standard UMAP

`UMAP` / `FittedUMAP` (Rust) and `fast_umap.UMAP` (Python) are a
non-parametric estimator that follows umap-learn 0.5.12: smooth-kNN fuzzy graph,
spectral (or random) initialization, and direct coordinate SGD with the
reference Tausworthe negative sampling and epoch schedule.

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
  construction or execution failures never cause backend substitution. Spectral initialization is host-side.
  `BackendChoice::validate()` returns the shared `BackendError` and validates
  common-crate capabilities. UMAP operations enforce their own enabled backend
  features and map backend failures to `UmapError::BackendUnavailable`.
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
cargo bench -p fast-umap --no-default-features --features cpu --bench standard_umap_cpu
cargo bench -p fast-umap --no-default-features --features cuda --bench standard_umap_cuda

# Chrome/Perfetto trace (import the JSON file into ui.perfetto.dev)
FAST_UMAP_TRACE=standard-cpu.json RUST_LOG=fast_umap=trace cargo bench -p fast-umap \
  --no-default-features --features cpu,tracing --bench standard_umap_cpu -- --test
FAST_UMAP_TRACE=standard-cuda.json RUST_LOG=fast_umap=trace,cubecl=trace,cubecl_cuda=trace cargo bench -p fast-umap \
  --no-default-features --features cuda,tracing --bench standard_umap_cuda -- --test

# Console span timings instead of a trace file
FAST_UMAP_TRACE=console RUST_LOG=fast_umap=debug cargo bench -p fast-umap \
  --no-default-features --features cpu,tracing --bench standard_umap_cpu -- --test
```

CUDA requires a working NVIDIA runtime. Traces are
host-side spans, not GPU kernel timings. Fit stage boundaries explicitly synchronize
the selected device when `tracing` is enabled; enabling detailed tracing adds
overhead. The benches measure complete fit and transform calls, including host
embedding readback, with deterministic inputs prepared outside the timed loop.

Compare standard UMAP with **umap-learn 0.5.12**, using CUDA only:

```sh
cargo build -p fast-umap --release --no-default-features --features cuda,tracing --example profile_standard_cuda
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

## Testing

The Standard CPU test lane uses CubeCL LLVM and does not require a GPU.

```shell
cargo test -p fast-umap-core
cargo test -p fast-umap --no-default-features --features cpu
bash scripts/test_features.sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common --no-default-features --features cuda -- --nocapture
```

Default quality fixtures retain every dataset with three fixed seeds, 10 graph
neighbors, 180 fit epochs, and 60 transform epochs. Trustworthiness is still
evaluated at 15 neighbors. For fit and transform separately, acceptance pools
all case/seed scores and requires `reference_mean - candidate_mean <= 0.025`;
improvements beyond 0.025 pass and are not penalized. Absolute gaps and per-case
signed gaps are diagnostics only. Recall thresholds are unchanged.
An opt-in 200-epoch fit/transform case retains longer-optimization coverage:

```shell
FAST_UMAP_RUN_LONG_QUALITY=1 cargo test -p fast-umap --no-default-features --features cpu --test test_standard_quality long_quality
FAST_UMAP_REQUIRE_CUDA=1 FAST_UMAP_RUN_LONG_QUALITY=1 cargo test -p fast-umap --no-default-features --features cuda --test test_standard_quality long_quality
```

Regenerate only quality fixtures (without replacing graph/layout references) with:

```shell
uv run --no-sync python crates/fast-umap/tests/fixtures/generate_standard_umap.py --quality-only
```

HDBSCAN infrastructure fixtures use the published `fast_hdbscan==0.3.2` release,
not a later source checkout. Verify their provenance and semantic regeneration with:

```shell
uv run --no-sync python crates/fast-hdbscan/tests/fixtures/generate_fast_hdbscan.py --verify-provenance
uv run --no-sync python crates/fast-hdbscan/tests/fixtures/generate_fast_hdbscan.py --check
cargo test -p fast-hdbscan-core --test test_fixtures
cargo test -p fast-hdbscan-core --features test-support
```

The reference remains unpatched: upstream [PR #72](https://github.com/TutteInstitute/fast_hdbscan/pull/72)
fixes leaf traversal in `segments_in_branch` after 0.3.2, preventing unrelated
clusters from being dropped during epsilon selection. The fixture catalogue
records that release behavior separately from the post-release correction.
The backend-free core implements host extraction from supplied MSTs and finite-row
remapping. Its independent brute-force MST oracle is available only with the
non-default `test-support` feature, not as a production execution fallback.
There is no HDBSCAN estimator or Python API yet.

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
  title   = {fast-umap: UMAP in Rust},
  author  = {Hauptmann, Eugene},
  year    = {2024},
  url     = {https://github.com/eugenehp/fast-umap},
  version = {1.6.0}
}
```

> Hauptmann, E. (2024). *fast-umap: UMAP in Rust* (v1.6.0).
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
