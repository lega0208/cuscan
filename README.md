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
fast-umap = { version = "1.6.0" }
```

| Feature | Description |
|---------|-------------|
| `tracing` | Instrument fit/transform phases and enable Burn/CubeCL tracing |
| `cubecl-ir-dump` | Dump CubeCL CPU IR for kernel inspection |

CubeCL LLVM CPU and CUDA support are always compiled. CUDA still requires a
working NVIDIA runtime/device; compiling it does not initialize CUDA or require
CUDA hardware. Runtime backend selection is controlled by the `backend` argument.

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

Build the local extension with both backends available. Avoid running `uv sync`
before tests, because syncing can replace the local extension:

```shell
uv run maturin develop --release
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
uv run maturin develop --release
uv run --no-sync pytest python/tests
uv run ruff check
uv run ruff format --check
uv run ty check
```

### UMAP and HDBSCAN comparison scripts

Run these scripts from the repository root after `uv sync` and a release
`maturin develop` build as above. Use `uv run --no-sync` to retain the locally
built extension. The reference implementations are **umap-learn 0.5.12** and
**fast-hdbscan 0.3.2**, respectively.

#### Side-by-side point plots

- `comparison/plot_comparison.py` fits reference and local UMAP on the same
  sampled `embedding` rows from `comparison/feedback_clustered.parquet`.
  Points are colored by the input `topic_global` column. The default output is
  `comparison/umap_comparison.html`.
- `comparison/save_2d_embeddings.py` reads that same input file, computes local
  UMAP coordinates, and writes `comparison/embeddings_2d.parquet`, preserving
  the original columns and adding `embedding_2d`. It processes the entire file
  and has no CLI options.
- `comparison/plot_hdbscan_comparison.py` reads the saved parquet and compares
  reference and local HDBSCAN labels. `--cluster-on 2d` (default) clusters
  `embedding_2d`; `--cluster-on full` clusters `embedding`. Both panels always
  use the saved `embedding_2d` coordinates for visualization. Colors and legends
  are estimator-specific cluster labels, not matched clusters across panels;
  gray indicates noise (`-1`). The default output is
  `comparison/hdbscan_comparison.html`.

```sh
uv run --no-sync python comparison/plot_comparison.py --device cpu --sample-size 10000
uv run --no-sync python comparison/save_2d_embeddings.py
uv run --no-sync python comparison/plot_hdbscan_comparison.py \
  --device cpu --cluster-on 2d --sample-size 10000
uv run --no-sync python comparison/plot_hdbscan_comparison.py \
  --device cpu --cluster-on full --sample-size 10000 \
  --output comparison/hdbscan_full_comparison.html
```

Both plot scripts support `--input`, `--output`, `--sample-size` (default
1,000,000), and `--device auto|cpu|cuda`. Sampling uses seed 42 and preserves
row alignment; this does not make unseeded estimator fits deterministic.
Outputs are standalone Plotly HTML files, and fit wall times are printed.
UMAP also accepts `--n-epochs`. HDBSCAN accepts `--min-cluster-size` (default 5),
`--min-samples`, `--cluster-selection-method eom|leaf`, and
`--allow-single-cluster`.

#### Per-stage timing reports

`comparison/compare_stages.py` compares native UMAP with umap-learn on identical
generated float32 data (default 50,000 rows, 32 features, 200 epochs).
`comparison/compare_hdbscan_stages.py` compares native HDBSCAN with fast_hdbscan
on generated data (default 5,000 rows, 16 features), or on parquet supplied with
`--input` and `--cluster-on 2d|full`. The latter needs only the selected column;
it does not generate point plots. Both scripts print timing tables and accept
`--html` for stage charts and `--json` for raw measurements and metadata.

Build their native companions separately so build time is not measured:

```sh
cargo build -p fast-umap --release --features tracing --example profile_standard_stages
cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages

uv run --no-sync python comparison/compare_stages.py --backend cpu \
  --html comparison/umap_stages.html --json comparison/umap_stages.json
uv run --no-sync python comparison/compare_hdbscan_stages.py --backend cpu \
  --input comparison/embeddings_2d.parquet --cluster-on 2d --sample-size 10000 \
  --html comparison/hdbscan_stages.html --json comparison/hdbscan_stages.json
```

Stage scripts default to CPU; for CUDA, build the corresponding companion with
`cuda,tracing` and pass `--backend cuda`. Each implementation runs in a separate
subprocess. One fit is process-cold (persistent caches may still apply).
`--rust-runs N` with N >= 2 compares the median of warmed native runs, excluding
run 0; the UMAP reference still runs once. HDBSCAN also supports `--python-runs N`
with the same cold/warmed rule for the reference.

UMAP reports neighbor search, fuzzy graph, schedule, spectral graph export,
initialization, and optimization; its `other` row is fit time outside measured
stages. HDBSCAN instead reports exclusive, non-overlapping stages:

| HDBSCAN stage | Measured work |
| --- | --- |
| `input_validation` | Configuration/input checks, flattening and finite-row detection in native; estimator/input/parameter validation in the reference |
| `backend_setup` | Native backend/device resolution, runtime initialization, allocation properties and search-route selection |
| `input_preparation` | Native result-buffer allocation, finite-row compaction and device allocation-size checks |
| `tree_build` | Host KD-tree construction and native tree allocation checks (only a branch check for native brute force) |
| `input_upload` | Native input/tree tensor uploads, including synchronization |
| `core_distances` | Native neighbor search, readback, core distances and retained neighbor/core output copies |
| `mst` | Native Borůvka spanning tree construction and finalization |
| `mst_setup` | Reference MST dispatch/validation outside the measured tree/Borůvka calls |
| `core_distances_and_mst` | Reference fused Numba Borůvka call, which cannot expose separate core-distance and MST timings through Python wrappers |
| `approximate_neighbors_and_mst` | Reference approximate-neighbor/MST path above 30 features, instead of its KD-tree/Borůvka stages |
| `cluster_extraction` | Linkage, condensation, cluster selection, labels and membership strengths |
| `result_finalization` | Native result remapping/scattering and fitted-state assembly; reference fitted-attribute/record-array conversion |

HDBSCAN estimator construction is outside fit timing. `unattributed` retains
only the per-fit difference between total time and measured exclusive stages;
it is timing/bookkeeping or uncaptured overhead, not a measured algorithm phase.
Missing or inapplicable stages show `n/a`, not zero, and do not receive a speedup
ratio. Warmed summaries take medians independently, so displayed stage medians
need not sum to the median total. Raw JSON retains the per-run measurements.

Native device-work stages include synchronization; these are host wall timings,
not individual GPU kernel timings. HDBSCAN's
`--search auto|brute-force|kd-tree` selects only the native complete-graph search route;
the port's kNN-graph route above 30 input features is not covered by `--search`. Above
30 input features, fast_hdbscan uses approximate pynndescent neighbors instead
of its KD-tree/Borůvka path; the script warns about this algorithm mismatch.
Timing results alone do not establish clustering or embedding quality. Each
script's `--help` lists its full options.

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
  common-crate capabilities. UMAP operations use the always-compiled backends
  and map backend failures to `UmapError::BackendUnavailable`.
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
cargo bench -p fast-umap --bench standard_umap_cpu
cargo bench -p fast-umap --bench standard_umap_cuda

# Chrome/Perfetto trace (import the JSON file into ui.perfetto.dev)
FAST_UMAP_TRACE=standard-cpu.json RUST_LOG=fast_umap=trace cargo bench -p fast-umap \
  --features tracing --bench standard_umap_cpu -- --test
FAST_UMAP_TRACE=standard-cuda.json RUST_LOG=fast_umap=trace,cubecl=trace,cubecl_cuda=trace cargo bench -p fast-umap \
  --features tracing --bench standard_umap_cuda -- --test

# Console span timings instead of a trace file
FAST_UMAP_TRACE=console RUST_LOG=fast_umap=debug cargo bench -p fast-umap \
  --features tracing --bench standard_umap_cpu -- --test
```

CUDA requires a working NVIDIA runtime. Traces are
host-side spans, not GPU kernel timings. Fit stage boundaries explicitly synchronize
the selected device when `tracing` is enabled; enabling detailed tracing adds
overhead. The benches measure complete fit and transform calls, including host
embedding readback, with deterministic inputs prepared outside the timed loop.

Compare standard UMAP with **umap-learn 0.5.12**, using CUDA only:

```sh
cargo build -p fast-umap --release --features tracing --example profile_standard_cuda
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
cargo test -p fast-umap
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common -- --nocapture
```

Default quality fixtures retain every dataset with three fixed seeds, 10 graph
neighbors, 180 fit epochs, and 60 transform epochs. Trustworthiness is still
evaluated at 15 neighbors. For fit and transform separately, acceptance pools
all case/seed scores and requires `reference_mean - candidate_mean <= 0.025`;
improvements beyond 0.025 pass and are not penalized. Absolute gaps and per-case
signed gaps are diagnostics only. Recall thresholds are unchanged.
An opt-in 200-epoch fit/transform case retains longer-optimization coverage:

```shell
FAST_UMAP_RUN_LONG_QUALITY=1 cargo test -p fast-umap --test test_standard_quality long_quality
FAST_UMAP_REQUIRE_CUDA=1 FAST_UMAP_RUN_LONG_QUALITY=1 cargo test -p fast-umap --test test_standard_quality long_quality
```

Regenerate only quality fixtures (without replacing graph/layout references) with:

```shell
uv run --no-sync python crates/fast-umap/tests/fixtures/generate_standard_umap.py --quality-only
```

HDBSCAN reference fixtures use the published `fast_hdbscan==0.3.2` release,
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
The Python package exposes `HDBSCAN` alongside `UMAP`:

```python
from fast_umap import HDBSCAN

model = HDBSCAN(min_cluster_size=5, backend="cpu").fit(X)
labels = model.labels_
cut_labels = model.dbscan_clustering(0.5)
```

`backend="auto"` chooses usable CUDA when available, otherwise CPU; explicit
`"cpu"` and `"cuda"` selections never fall back after an execution error.
`reproducible` (default `False`) selects the bound mode of the complete-graph KD-tree
route. `False` shares per-component pruning bounds within each Borůvka launch and returns
exact component winners. `True` gives each row independent candidates without the shared
bound. Tests show bitwise-identical MST endpoints and weights between the two modes on five
fixtures, seeded tie sets and one Auto-KD-region dataset. Brute-force and kNN-graph routes
ignore the flag. Only exact Euclidean, unweighted clustering is supported. Non-default
metric, `algorithm="kruskal"`, sample-weight, cannot-link and semi-supervised options raise
`NotImplementedError`. `knn_k` accepts `None` or a positive integer; bool and float values
raise `ValueError`.

Route selection follows upstream's dimension rule. Up to 30 features, the estimator builds
the exact complete mutual-reachability graph. Above 30 features it builds a kNN graph.
The graph width is `min(max(knn_k or max(3 * min_samples, 15), min_samples), n_finite - 1)`.
It uses exact neighbors when `n_finite <= 4096`. Above 4096 finite rows it uses approximate
neighbors at the resolved width (CubeCL ANN, fixed seed 42, squared distances,
`max(5, round(log2(n)))` iterations); the width is not separately capped. Kruskal runs over the kNN edges. Components left
disconnected are joined by `+inf` bridges between each component's smallest finite-row index.
The exact branch costs O(n²·d) work, like the complete route. The ANN branch's width is
independent of the exact-route row limit. Quality evidence is empirical: on synthetic 6000×40 and
6000×50 data at widths up to 60, three seeds per backend gave mean recall of at least 0.9998, per-case
ARI of 1.0, and a maximum finite-MST-sum relative error of 7.1e-6. The current width campaign completed
76 cases per backend, all passing; seven planned cases per backend were not completed (six large-seed
cases skipped under the time budget and the high-density 4097×31 case at width 4096 timed out
without output), so it does not establish
comprehensive validation across all shapes. Resource estimates (see the advisory above) are practical
guidance for memory use, not guarantees of memory or quality for every shape or dataset. The campaign's
fit times are observations, not speedup claims.

Divergences from upstream: the port removes the query row from its own neighbor list, while
upstream drops column 0, which can be a duplicate row; `knn_k=0` is rejected; the ANN seed is
fixed. ANN-branch results therefore match upstream in distribution, not bit for bit. Python
selects the route by feature count only. Rust `MstGraph::KnnGraph` can be forced at any
dimension. `algorithm="kruskal"` on the complete graph remains unimplemented.
Equal-distance edges use deterministic port-side ordering, so tied-data cluster
memberships can differ from upstream even when the threshold connectivity
hierarchy is identical; upstream traversal-order parity is not promised.

Labels, probabilities, linkage and condensed trees use original-row indexing;
nonfinite rows become noise. `neighbors_`, `core_distances_` and
`minimum_spanning_tree_` use finite-row indices, mapped by `finite_index_`.
Arrays are independent copies of the Rust-owned fitted snapshot. Probabilities
are float32 for finite input and float64 when nonfinite rows are remapped,
matching upstream. Trees are raw NumPy arrays, not plotting wrapper objects.
Failed refits preserve the previous fitted snapshot and its DBSCAN cuts.

Before a kNN-route fit, `fast_umap.HDBSCAN` calls the native shape-only estimator with the
finite-row count, feature count and the same configuration, then emits one `UserWarning` when any
applicable scenario reaches 1 GiB (2^30 bytes). The estimator reads no data, does not initialize
CUDA and performs no device discovery. The complete graph and fewer than two finite rows produce
no advisory. The graph width is `min(max(knn_k or max(3 * min_samples, 15), min_samples), n_finite - 1)`:
nonfinite rows are not counted, and when `knn_k < min_samples` the `min_samples` floor applies. Directed
edges and the dominant graph storage are `O(n * k)` with `n` finite rows and `k` the effective width.
Approximate-neighbor fits also report a retained candidate width, `min(width + 4, n_finite - 1)`, which
is distinct from the effective width and is not used for the directed-edge count.

`backend="auto"` reports CPU and CUDA as conditional alternatives. They are not summed, and only the
backend that fit resolves runs. The CPU figure is one overlapping host+runtime peak. The CUDA figures
are separate host and device peaks. Each is the maximum over fit phases, not a sum of phases. These are
estimates of the allocations the fit plan requests. They are not allocator reservations, cached memory,
interpreter or other process memory, or an available-memory check. The estimate excludes the caller's
raw input buffer and the outlier-compaction copy made when nonfinite rows are dropped, so it is a lower
bound for inputs with dropped rows. The advisory does not block the fit, change the configuration, retry
or fall back to another backend. Use the standard `warnings` filters to ignore or escalate it; `fast_umap`
adds no separate option and keeps no duplicate estimator policy.

The `fast-hdbscan` crate also exposes the Rust `HDBSCAN` estimator. For example,
with its `cpu` feature enabled:

```rust
use fast_hdbscan::{BackendChoice, HDBSCAN, HdbscanConfig, HdbscanSearch};

let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![5.0, 5.0]];
let fitted = HDBSCAN::new(
    HdbscanConfig { min_cluster_size: 2, min_samples: Some(1), ..Default::default() },
    BackendChoice::Cpu,
)
.with_search(HdbscanSearch::BruteForce)
.fit(&data)?;
let labels = fitted.labels();
# Ok::<(), fast_hdbscan::HdbscanError>(())
```

| `BruteForce` and `KdTree` select exact Euclidean search at every dimension;
neither is an approximate mode or a fallback after execution errors. On CPU and
CUDA, `Auto` selects KD search for 256–2048 finite rows with at most 30 features,
or 2049–4096 rows with at most 10 features, when resolved `min_samples <= 32`;
otherwise it selects brute force. The low-dimensional/intermediate-size region
is a conservative policy: the 2049–4096-row, at-most-10-feature extension is
low-dimensional/`min_samples` interpolation, not a separately measured guarantee
for every size or data distribution.
Backend selection is independent: `BackendChoice::Auto` prefers usable CUDA,
while explicit CPU/CUDA choices are strict. Input features are not rescaled.
Nonfinite rows become noise in labels/probabilities and are remapped into the
trees; neighbors, core distances and MST endpoints remain finite-row indexed,
with the original-row mapping available through `finite_index()`. Neighbors
exclude self, use `-1` padding, and core distances include infinite padding.
`dbscan_clustering(epsilon)` returns a `Result` and rejects NaN.

`HdbscanConfig::mst_graph` (`MstGraph::{Auto, Complete, KnnGraph}`, default `Auto`) selects
the graph and `knn_k` sets the kNN width; both are Rust-only fields except that Python
exposes `knn_k`. `Auto` resolves to `KnnGraph` above 30 features and to `Complete` otherwise.
`with_search` and `FittedHDBSCAN::search()` apply only to the complete-graph route.
`HdbscanConfig::reproducible` (default `false`) has the bound-mode meaning described above.

`fast_hdbscan::estimate_fit_resources(n_finite, features, &config, backend_choice)` returns the same
shape-only `FitResourceEstimate` without reading data or touching a device. `knn` is `None` on the
complete route. With the crate's `tracing` feature, `fit` emits one `tracing::warn!` event carrying the
same advisory fields; without that feature `fit` emits nothing. The same exclusions and CPU/CUDA
semantics described for Python apply.

```shell
cargo test -p fast-hdbscan
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan
cargo bench -p fast-hdbscan --bench hdbscan_cpu
FAST_UMAP_REQUIRE_CUDA=1 cargo bench -p fast-hdbscan --bench hdbscan_cuda
```

Criterion benchmarks compare forced routes on the same fixtures and seeded
synthetic data, with separate host tree-build timings. Use
`FAST_HDBSCAN_BENCH_CASES` with comma-separated case-name substrings to select
cases; full-fit timings include upload, device execution and host readback.
Set `FAST_HDBSCAN_BENCH_STAGES=1` for isolated neighbor, single candidate-round
and synchronized tree-upload timings. Candidate-stage cases use singleton
components, not total MST construction; upload timings include readback.
The default `reproducible=false` mode uses shared component bounds on the complete-graph KD
route. Device-side component winner reduction is not implemented.

Local Phase 4 measurements on Ryzen 7 9800X3D CPU and RTX 3080 CUDA used 100
samples, 95% confidence intervals, three separate processes per runtime, two
pinned fixtures and the full seeded synthetic grid: 64/256/1024/2048 rows,
2/10/30/50 features, `min_samples` 5/32, plus 4096 rows at 2/10 features.
For every measured case with at least 256 rows and at most 30 features, the KD
mean's upper confidence limit was below the brute-force mean's lower limit in
all three runs. Small CUDA inputs were inconsistent, so they remain brute force.
The synthetic data is clustered; this does not establish distribution-independent
performance or extrapolate a speed claim to unmeasured hardware/dimensions.
Runs used `--warm-up-time 0.1 --measurement-time 0.2 --noplot`; Criterion extended
collection when necessary to retain all 100 samples. Reference tile widths
128/256/512 were investigated separately on both runtimes; the production
default remains 256, without a runtime-specific tile tuning claim.

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
