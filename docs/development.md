# Development and analysis

Run commands from the repository root. Measurements below are historical results from the predecessor checkout, not rebenchmarked cuscan results.

## Continuous integration

The workflow suite is intended to run on pull requests, main pushes, merge queues and dispatch. `ci-ok` aggregates results and treats intentionally skipped jobs as success; configure it as the sole required check in the branch ruleset. Pull requests and pushes classify changed paths; unknown paths and workflow changes run the full suite. Draft PRs defer heavy tests until ready. The weekly scheduled workflow adds a floating-nightly canary, Windows Rust tests, long-quality and feature contracts, minimum NumPy, source-distribution build, cold build and cargo-deny; failures are reported to the open `Scheduled CI failure` issue. The pinned compiler is `nightly-2026-08-08` (rustc `1a98b1e13`).

| Change                                 | Intended work                                                                 |
| -------------------------------------- | ----------------------------------------------------------------------------- |
| Docs only                              | Change classification and `ci-ok`                                             |
| Python tests                           | Python lint, wheel reuse/build and wheel tests                                |
| Python package or Rust source          | Relevant lint, wheel builds and wheel tests; Rust changes also run Rust tests |
| Dependency manifests or lockfiles      | Relevant checks plus cargo-deny                                               |
| Workflows, CI scripts or unknown paths | Full suite                                                                    |

An extracted plain sdist built and imported in fresh manylinux without `.cargo`, linker or `RUSTFLAGS` overrides and without an extension `build.rs` (`artifact://76`). The pinned Rust target defaults to self-contained `gnu-lld-cc`; the wheel-linux class explicitly chooses gcc, which changes the flavor to GNU ld and exposes LLVM archive-order failure, so it explicitly selects bundled LLD. The sdist class installs only the toolchain and sets no such flags.

CI caches are an optimization, not correctness evidence: Cargo caches are separated by build class and releases build without caches. Bump `CI_CACHE_EPOCH` in the shared `ci-env` action to orphan suspect or bloated caches. CubeCL's runtime cache is not shared across runners. GitHub cache, runner-disk and release behavior have not been validated; estimates in `ci_workflows_plan.md` are not measured GitHub results. The user approved the scoped `paste` exception: `deny.toml` ignores only RUSTSEC-2024-0436 with rationale, and all four `cargo deny` checks passed with ordinary duplicate and license warnings.

### Local act

Install Docker and `act`, then from the repository root:

```sh
scripts/ci/act.sh list
scripts/ci/act.sh dry pr
scripts/ci/act.sh dry tag
scripts/ci/act.sh pr-docs
scripts/ci/act.sh pr -j lint-python
scripts/ci/act.sh pr -j lint-rust
scripts/ci/act.sh pr -j test-rust
scripts/ci/act.sh pr -j wheels-linux
scripts/ci/act.sh release-dry
```

`release-dry` performs a real Linux rehearsal build and checks that the sdist source build executed; it requires a `-dev.N` workspace version. The other `dry` commands only evaluate the graph. The docs-only scenario uses synthetic `ci_changed_paths` honored only when the event has `act: true`; real GitHub events use the normal change classifier. ACT manylinux bootstraps pinned Node 24.11.1, and setup-uv installs the requested Python because setup-python does not support AlmaLinux. act does not emulate Windows jobs, environment approvals, OIDC publishing, attestations or issue reporting. Its local cache server is separate from GitHub's and cannot validate GitHub quota/eviction or runner disk headroom. Docker without an NVIDIA runtime cannot validate CUDA; use the native lane below.

### CUDA lane

CUDA validation is independent of GitHub and CI. On a trusted NVIDIA host with Rust, uv, Python and the driver installed, review the selected commit and run:

```sh
scripts/ci/cuda_lane.sh --ref <commit>
```

The script tests the selected commit in a detached worktree, requires CUDA for Rust and Python, and records logs under `${XDG_CACHE_HOME:-$HOME/.cache}/cuscan-cuda/logs/`. Optional `--rows 20000` or `--rows 100000` runs diagnostic HDBSCAN analysis and requires the relevant Nsight/counter access. Running a selected commit executes its code on the host; review it first.

The lane builds the extension with `maturin develop --release --features tracing`

## Historical serialization benchmark

The historical storage benchmark compared compression levels and chunk row sizes for
continuous and integer-valued f32 features, embeddings, graphs, and a
deduplicated-model state. It reports file size, isolated save/load latency and
peak host-process RSS; it does not measure device memory. One quick run on
Linux x86-64/WSL2 with Python 3.14, using synthetic 4,096-row, 32-feature data,
gave the following results at 1,024-row chunks (latency in milliseconds, RSS
in MiB):

| Synthetic case           | Blosc2 level | File bytes | Save ms | Load ms | Save RSS MiB | Load RSS MiB |
| ------------------------ | -----------: | ---------: | ------: | ------: | -----------: | -----------: |
| Continuous f32 features  |            1 |    474,450 |   29.25 |    5.55 |         61.1 |         55.5 |
| Continuous f32 features  |            9 |    457,808 |   94.22 |    5.91 |         62.8 |         55.8 |
| Deduplicated model state |            1 |    305,631 |   33.88 |    8.90 |         60.3 |         55.0 |
| Deduplicated model state |            9 |    298,053 |  113.92 |    2.16 |         62.9 |         55.2 |

In these single-trial quick measurements, level 9 reduced file size by about
2.9–3.0% versus level 1, but took about 3.3–3.4× as long to save and used up to
1.7 MiB more peak RSS. A 1,024-row chunk was the smallest of the tested chunk
sizes in both rows. This supports level 1 with 1,024-row chunks as the
size/save-time tradeoff; production settings should still be evaluated on the
target workload. These measurements are synthetic and single-trial, and
latency (especially load latency) is noisy; RSS includes interpreter and
dependency baseline. Device memory was unavailable because the benchmark uses
host-side NumPy and h5py only. The original driver `scripts/benchmark_serialization.py`
is unavailable in this checkout, so the measurements cannot be reproduced using
the repository as shipped; workload-specific conclusions require new measurements.

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

The local parquet datasets under `comparison/` are gitignored and are not shipped.
Commands using them require user-provided schema-compatible parquet files. For the
default point-plot workflow, supply `data.parquet` with `embedding`
and `topic_global` columns; `save_embeddings.py` reads that fixed path and produces
the `embedding_2d` and `embedding_16d` columns used by the HDBSCAN plots.

#### Side-by-side point plots

- `comparison/plot_comparison.py` fits reference and local UMAP on the same
  sampled `embedding` rows from `comparison/data.parquet`.
  Points are colored by the input `topic_global` column. The default output is
  `comparison/umap_comparison.html`.
- `comparison/save_embeddings.py` reads that same input file and writes `embedding_2d`
  and `embedding_16d` to `comparison/embeddings.parquet`. It processes the entire file
  and has no CLI options.
- `comparison/plot_hdbscan_comparison.py` reads the saved parquet and compares
  reference and local HDBSCAN labels. `--cluster-on 2d` clusters
  `embedding_2d`; `--cluster-on 16d` (default) clusters `embedding_16d`. Both panels always
  use the saved `embedding_2d` coordinates for visualization. Colors and legends
  are estimator-specific cluster labels, not matched clusters across panels;
  gray indicates noise (`-1`). The default output is
  `comparison/hdbscan_comparison.html`.

```sh
uv run --no-sync python comparison/plot_comparison.py --device cpu --sample-size 10000
uv run --no-sync python comparison/save_embeddings.py
uv run --no-sync python comparison/plot_hdbscan_comparison.py \
  --device cpu --cluster-on 2d --sample-size 10000
uv run --no-sync python comparison/plot_hdbscan_comparison.py \
  --device cpu --cluster-on 16d --sample-size 10000 \
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
`--input` and `--cluster-on 2d|16d` (default `16d`). The latter needs only the selected column;
it does not generate point plots. Both scripts print timing tables and accept
`--html` for stage charts and `--json` for raw measurements and metadata.

Build the native companions separately so build time
is not measured:

```sh
cargo build -p umap --release --features tracing --example profile_standard_stages
cargo build -p hdbscan --release --features tracing --example profile_hdbscan_stages

uv run --no-sync python comparison/compare_stages.py --backend cpu \
  --html comparison/umap_stages.html --json comparison/umap_stages.json
uv run --no-sync python comparison/compare_hdbscan_stages.py --backend cpu \
  --input comparison/embeddings.parquet --cluster-on 2d --sample-size 10000 \
  --html comparison/hdbscan_stages.html --json comparison/hdbscan_stages.json
```

Stage scripts default to CPU; for CUDA, keep the `tracing` build above and pass
`--backend cuda` (both backends are always compiled). Each implementation runs in
a separate subprocess. UMAP's first fit is process-cold (persistent caches may
still apply); `--rust-runs N` with N >= 2 compares warmed runs excluding run 0,
while its reference runs once. HDBSCAN instead performs one excluded warmup then
exactly `--rust-runs N` and `--python-runs N` measured fits respectively.
For native-only hotspot attribution, see [HDBSCAN stage analysis](#hdbscan-stage-analysis).

#### HDBSCAN stage analysis

`scripts/analyze_hdbscan_stages.sh` analyzes native CPU/CUDA hotspots, not performance
against reference `fast_hdbscan`. It requires Linux x86-64 (including WSL2), Cargo,
a configured uv environment (`uv sync`), a working NVIDIA CUDA runtime, Nsight
Systems (`nsys`), Nsight Compute (`ncu`), and GPU counter permissions.

```sh
bash scripts/analyze_hdbscan_stages.sh --help
bash scripts/analyze_hdbscan_stages.sh                     # 20,000 rows
bash scripts/analyze_hdbscan_stages.sh --rows 50000
bash scripts/analyze_hdbscan_stages.sh --rows 100000
bash scripts/analyze_hdbscan_stages.sh --rows 20000 --output-dir artifacts/hdbscan_stages/my-run
uv run --no-sync python analyze/summarize_hdbscan_stages.py \
  --benchmark artifacts/hdbscan_stages/my-run/benchmark.json \
  --kernels artifacts/hdbscan_stages/my-run/cuda_kernels.json \
  --counters artifacts/hdbscan_stages/my-run/cuda_counters.json
```

Explicit output directories must not exist; omitted directories are unique.
Invocation from another working directory is supported. The fixed workload has
16 features, eight centers, center scale/noise 1.0, seed 42, min-samples and
min-cluster-size 5, and Auto search. All captures share identical little-endian
float32 input bytes and record their SHA-256. Native CPU and CUDA each perform
one excluded warmup and three measured fits, serially, retaining raw samples and
medians. Exact final-label agreement is required, not proof of external-data
clustering quality.

Compilation is outside execution timeouts; runtime initialization and JIT are
inside. At 20k, each native/profiler child has a 10-second deadline, not a
10-second total-pipeline promise. At 50k and 100k, native children have 240 seconds and
each profiler capture 90 seconds. SQLite export has a separate 60-second limit.
There are no automatic retries, timeout inflation, backend fallbacks, or skips.
Failures retain diagnostics and return nonzero without a successful summary.

The three canonical JSON files above suffice for offline summarization without
CUDA or Nsight. `raw/` retains input, manifest, native profiles, traces, labels,
stdout/stderr, Nsight reports, SQLite, and counter CSVs; allow substantial disk
space. Existing ignore rules exclude raw `.f32le`, `.sqlite`, and `.nsys-rep`
artifacts; inspectable JSON/CSV/host traces remain available.

Nsight Systems attributes only the measured fit using epoch markers and its own
native stage timings. Summed device activity is not elapsed wall time, especially
with overlapping kernels. Nsight Compute samples the first matching warmup
launch of the resolved CUDA route's neighbor and candidate kernels separately
(`partial_kernel`/`tile_candidates` for brute force,
`kd_neighbors_kernel`/`kd_candidates_kernel` for KD-tree), not whole-fit
averages. Synchronization, instrumentation, replay, and cache behavior perturb
measurements. Counter diagnoses are conditional hypotheses; they cannot establish
CPU instruction bottlenecks or universal speedups. Independently computed medians
are not additive; stage shares use aligned per-fit samples.

UMAP reports neighbor search, fuzzy graph, schedule, spectral graph export,
initialization, and optimization; its `other` row is fit time outside measured
stages. HDBSCAN instead reports exclusive, non-overlapping stages:

| HDBSCAN stage                   | Measured work                                                                                                                    |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `input_validation`              | Configuration/input checks, flattening and finite-row detection in native; estimator/input/parameter validation in the reference |
| `backend_setup`                 | Native backend/device resolution, runtime initialization, allocation properties and search-route selection                       |
| `input_preparation`             | Native result-buffer allocation, finite-row compaction and device allocation-size checks                                         |
| `tree_build`                    | Host KD-tree construction and native tree allocation checks (only a branch check for native brute force)                         |
| `input_upload`                  | Native input/tree tensor uploads, including synchronization                                                                      |
| `core_distances`                | Native neighbor search, readback, core distances and retained neighbor/core output copies                                        |
| `mst`                           | Native Borůvka spanning tree construction and finalization                                                                       |
| `mst_setup`                     | Reference MST dispatch/validation outside the measured tree/Borůvka calls                                                        |
| `core_distances_and_mst`        | Reference fused Numba Borůvka call, which cannot expose separate core-distance and MST timings through Python wrappers           |
| `approximate_neighbors_and_mst` | Reference approximate-neighbor/MST path above 30 features, instead of its KD-tree/Borůvka stages                                 |
| `cluster_extraction`            | Linkage, condensation, cluster selection, labels and membership strengths                                                        |
| `result_finalization`           | Native result remapping/scattering and fitted-state assembly; reference fitted-attribute/record-array conversion                 |

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

`UMAP` / `FittedUMAP` (Rust) and `cuscan.UMAP` (Python) are a
non-parametric estimator that follows umap-learn 0.5.12: smooth-kNN fuzzy graph,
spectral (or random) initialization, and direct coordinate SGD with the
reference Tausworthe negative sampling and epoch schedule.

```rust
use umap::{BackendChoice, NeighborSearch, UMAP, UMAPConfig};

let config = UMAPConfig { random_state: Some(42), ..UMAPConfig::default() };
let fitted = UMAP::new(config, BackendChoice::Auto, NeighborSearch::Auto).fit(&data)?;
let embedding = fitted.embedding();          // Vec<Vec<f32>>, one row per sample
let projected = fitted.transform(&new_data)?; // place unseen rows in the fitted map
```

```python
import cuscan

model = cuscan.UMAP(n_neighbors=15, random_state=42, backend="auto")
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
install one only when `CUSCAN_TRACE` is set; leave it unset for timing runs.

```sh
cargo bench -p umap --bench standard_umap_cpu
cargo bench -p umap --bench standard_umap_cuda

# Chrome/Perfetto trace (import the JSON file into ui.perfetto.dev)
CUSCAN_TRACE=standard-cpu.json RUST_LOG=umap=trace cargo bench -p umap \
  --features tracing --bench standard_umap_cpu -- --test
CUSCAN_TRACE=standard-cuda.json RUST_LOG=umap=trace,cubecl=trace,cubecl_cuda=trace cargo bench -p umap \
  --features tracing --bench standard_umap_cuda -- --test

# Console span timings instead of a trace file
CUSCAN_TRACE=console RUST_LOG=umap=debug cargo bench -p umap \
  --features tracing --bench standard_umap_cpu -- --test
```

CUDA requires a working NVIDIA runtime. Traces are
host-side spans, not GPU kernel timings. Fit stage boundaries explicitly synchronize
the selected device when `tracing` is enabled; enabling detailed tracing adds
overhead. The benches measure complete fit and transform calls, including host
embedding readback, with deterministic inputs prepared outside the timed loop.

Compare standard UMAP with **umap-learn 0.5.12**, using CUDA only:

```sh
cargo build -p umap --release --features tracing --example profile_standard_cuda
uv run --no-sync python scripts/compare_umap_spans.py --output comparison.json
```

The build is separate from the default 600-second global run deadline. The primary
workload is 5,000 rows, 32 features and 20 epochs; small regression workloads are
512 rows, 16 features and 30 epochs, and 1,024 rows, 32 features and 100 epochs.
The suite runs exact and approximate search with 15 neighbors, `min_dist=0.0` and
random initialization. Both implementations consume identical float32 inputs and
warm up before timing; Python fits are unseeded. The suite does not accept `--trace`;
the native `profile_standard_cuda` example accepts it when invoked directly.

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
The native profiler requires more than 15 rows, with neighbor count smaller than
the row count; it imposes no 4,095-row cap. Missing CUDA fails without CPU fallback.

---

## Testing

The Standard CPU test lane uses CubeCL LLVM and does not require a GPU.

```shell
cargo test -p umap-core
cargo test -p umap
CUSCAN_REQUIRE_CUDA=1 cargo test -p cuscan-common -- --nocapture
```

Default quality fixtures retain every dataset with three fixed seeds, 10 graph
neighbors, 180 fit epochs, and 60 transform epochs. Trustworthiness is still
evaluated at 15 neighbors. For fit and transform separately, acceptance pools
all case/seed scores and requires `reference_mean - candidate_mean <= 0.025`;
improvements beyond 0.025 pass and are not penalized. Absolute gaps and per-case
signed gaps are diagnostics only. Recall thresholds are unchanged.
An opt-in 200-epoch fit/transform case retains longer-optimization coverage:

```shell
CUSCAN_RUN_LONG_QUALITY=1 cargo test -p umap --test test_standard_quality long_quality
CUSCAN_REQUIRE_CUDA=1 CUSCAN_RUN_LONG_QUALITY=1 cargo test -p umap --test test_standard_quality long_quality
```

Regenerate only quality fixtures (without replacing graph/layout references) with:

```shell
uv run --no-sync python crates/umap/tests/fixtures/generate_standard_umap.py --quality-only
```

HDBSCAN reference fixtures use the published `fast_hdbscan==0.3.2` release,
not a later source checkout. Verify their provenance and semantic regeneration with:

```shell
uv run --no-sync python crates/hdbscan/tests/fixtures/generate_fast_hdbscan.py --verify-provenance
uv run --no-sync python crates/hdbscan/tests/fixtures/generate_fast_hdbscan.py --check
cargo test -p hdbscan-core --test test_fixtures
cargo test -p hdbscan-core --features test-support
```

Regeneration preserves input arrays, labels, neighbors, topology, counts, dtypes,
shapes, nonfinite values and provenance exactly. Named derived floating outputs
(distances, weights, probabilities and hierarchy lambda values) use `rtol=1e-6`
and `atol=1e-7`, allowing small CPU-target-dependent upstream arithmetic differences.
The same bounds apply only to the derived persistence-boundary evidence fields;
parameters and categorical metadata remain exact. The below/equal/above persistence
boundary assertions also remain exact. Checked-in fixtures are not rewritten by
`--check`.

The reference remains unpatched: upstream [PR #72](https://github.com/TutteInstitute/fast_hdbscan/pull/72)
fixes leaf traversal in `segments_in_branch` after 0.3.2, preventing unrelated
clusters from being dropped during epsilon selection. The fixture catalogue
records that release behavior separately from the post-release correction.
The backend-free core implements host extraction from supplied MSTs and finite-row
remapping. Its independent brute-force MST oracle is available only with the
non-default `test-support` feature, not as a production execution fallback.
The Python package exposes `HDBSCAN` alongside `UMAP`:

```python
from cuscan import HDBSCAN

model = HDBSCAN(min_cluster_size=5, backend="cpu").fit(X)
labels = model.labels_
cut_labels = model.dbscan_clustering(0.5)
new_labels, strengths = model.approximate_predict(X_new)
new_labels, strengths, connecting_rows = model.approximate_predict(
    X_new, return_connecting_points=True
)
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

#### Predicting into fitted clusters

`approximate_predict` assigns each query independently to the existing fitted
hierarchy; it never refits, discovers clusters, or uses other queries as neighbors.
Labels and optional connectors are newly owned `np.intp` arrays; strengths are
float64. Connectors identify original training rows selected by minimum
mutual reachability, not necessarily Euclidean nearest neighbors. Noise after a
real search still has a valid connector.

For a prediction-only object that can outlive the estimator, convert explicitly:

```python
predictor = model.to_predictor()
del model
new_labels, strengths, connecting_rows = predictor.approximate_predict(
    X_new, return_connecting_points=True
)
```

The predictor retains only finite float32 geometry and validated prediction
hierarchy/runtime state, including its prediction KD-tree cache when built; it
does not retain labels, probabilities, fit-output trees, fit neighbors, fit core
distances or a reference to the estimator. Conversion may synchronously
perform cold prediction-core preparation on the concrete backend selected at fit
time; conversion has no backend override and uses that backend strictly. It can
fail before returning. Core preparation can be expensive; it is performed once
during conversion so later standalone use does
not depend on the estimator or an uninitialized core cache. Rank-one and
no-cluster-tree shortcuts avoid that search. Predictions still use call-local
uploads and bounded batches; converting is not a backend-portable artifact.

Prediction supports only finite, nonempty Euclidean feature matrices with the
fitted feature width. It uses the concrete CPU/CUDA backend chosen by fit, strictly
without execution fallback, regardless of later constructor-attribute changes.
All fitting routes, including ANN fitting, use exact bounds-pruned KD search for
prediction. Prediction cores include self at rank `min_samples - 1`; query cores
use rank `min_samples`. Consequently, predicting training coordinates need not
reproduce `probabilities_`. Nonempty cluster trees require at least
`2 * min_samples` finite training rows; unsupported widths raise an error rather
than being clamped. Nonfinite training rows remain excluded and connectors are
mapped back to original row IDs.

CPU prediction uses C16's per-query pruned traversal over a call-local leaf-tiled
SoA reference pack; CUDA prediction keeps the padded-row per-query kernel.
Neighbors are exact in `(squared distance, reference ID)` order under each kernel's
own per-feature float32 arithmetic, with `(inf, n)` padding. Exactly representable
quantized squared sums agree bitwise. On arbitrary inputs, vector/scalar FMA
contraction can round differently: CPU prediction distances may differ by rounding
from the previous release, and near ties can change neighbors, labels, strengths
or connectors. CPU prediction cores now use the same arithmetic as CPU fitting.

Models retain compact float32 geometry (`4*n*d` bytes), hierarchy metadata and
host KD trees built during fitting. Other routes build their host tree lazily.
The first prediction additionally computes and caches `8*n` bytes of float64
prediction cores; repeated calls reuse immutable host caches. Each call uploads
the reference coordinates/tree and prepares reference packs once, reuses them
across bounded core/query batches, and releases device buffers on return or error.
There is no persistent device cache or full pairwise-distance matrix. Warm calls
still pay upload, packing and readback costs. Cold tree construction and exact
core searches can be expensive: worst-case all-training KD search is quadratic,
and external queries can cost `O(m*n*d)`, especially in high dimensions or with
duplicates. Fit-only resource estimates do not cover lazy prediction peaks.

The algorithm follows original Python `hdbscan.approximate_predict`, not the
separate `fast_hdbscan` package. Geometry/distance arithmetic is float32 here,
versus the reference's float64; ties use deterministic distance/point-ID ordering,
not a promise of reference connector identity at numerical boundaries. Lambda
and strength arithmetic is float64. Upstream quirks are preserved: duplicate
queries can have zero strength when a cluster maximum lambda is infinite;
root-only/empty cluster-edge trees warn once and return noise, zero strengths and
`-1` connectors; label-derived anchors can leave a sibling unmapped when a selected
ancestor has no direct point rows.

Prediction reads an authoritative immutable complete fit snapshot. Public mutable
Rust fields and Python arrays are detached views: editing them changes neither
prediction nor the canonical state intended for future export. Clones share
immutable host data and synchronized caches.

`HDBSCANPredictor` alone supports HDF5 save/load; this is not persistence for the
fitted `HDBSCAN` estimator and does not preserve fit outputs. Install the optional
dependencies locally with `uv sync --extra serialization`, then rebuild the local
extension with `uv run maturin develop --release`. Saving exports the already-ready
prediction state before filesystem publication and performs no prediction or
core preparation. Files use the shared strict schema-v1 HDF5 publisher with
atomic no-clobber publication by default; `overwrite=True` explicitly replaces
an existing file. Publication flushes a same-directory temporary file and syncs the
directory on POSIX; a post-commit sync failure reports uncertain durability without
rolling back the published file. Windows flushes file data and commits atomically,
but does not guarantee directory-entry durability across a crash. Loading validates
the predictor schema and rebuilds the host KD tree, then chooses a runtime
backend independently of fit provenance:
`HDBSCANPredictor.load(path, backend="auto", device_index=0)` uses the normal
auto-selection policy, while explicit `cpu` or `cuda` is strict. The load-time
backend is validated and selected even for no-cluster-tree shortcut predictors;
those predictors still perform no device work when queried. No fallback or
backend migration occurs. Predictor files contain sensitive training geometry
and derived prediction state; protect them like the source data. HDF5 compression
and schema validation do not authenticate files, and Rust does not provide this
HDF5 persistence API.

Before a kNN-route fit, `cuscan.HDBSCAN` calls the native shape-only estimator with the
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
or fall back to another backend. Use the standard `warnings` filters to ignore or escalate it; `cuscan`
adds no separate option and keeps no duplicate estimator policy.

The `hdbscan` crate also exposes the Rust `HDBSCAN` estimator. For example,
with either compiled backend:

```rust
use hdbscan::{BackendChoice, HDBSCAN, HdbscanConfig, HdbscanSearch};

let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![5.0, 5.0]];
let fitted = HDBSCAN::new(
    HdbscanConfig { min_cluster_size: 2, min_samples: Some(1), ..Default::default() },
    BackendChoice::Cpu,
)
.with_search(HdbscanSearch::BruteForce)
.fit(&data)?;
let labels = fitted.labels();
# Ok::<(), hdbscan::HdbscanError>(())
```

| `BruteForce` and `KdTree` select exact Euclidean search at every dimension;
neither is an approximate mode or a fallback after execution errors. On CPU and
CUDA, `Auto` selects KD search for at least 256 finite rows with at most 30 features,
with no upper row or `min_samples` limit, matching upstream fast_hdbscan's KD-tree
Borůvka route. Intentional deviations: fewer than 256 rows use brute force (small
inputs showed no consistent KD win), and a forced `MstGraph::Complete` above 30
features uses brute force where upstream would switch to approximate neighbors.
Backend selection is independent: `BackendChoice::Auto` prefers usable CUDA,
while explicit CPU/CUDA choices are strict. Input features are not rescaled.
Nonfinite rows become noise in labels/probabilities and are remapped into the
trees; neighbors, core distances and MST endpoints remain finite-row indexed,
with the original-row mapping available through `finite_index()`. Neighbors
exclude self, use `-1` padding, and core distances include infinite padding.
`dbscan_clustering(epsilon)` returns a `Result` and rejects NaN.

KD neighbor implementation: CPU retains C16 leaf-packed SoA and leaf-block queries.
On eligible CUDA devices, 16–32 features use a flat plane-cooperative scan with
two planes per cube and no row-count gate. Below 16, above 32, without the required
plane operations/synchronization, or with insufficient launch/shared-memory
resources, CUDA uses the per-query kernel (pre-launch selection, not error recovery).
Leaf order preserves original IDs and output rows; feature sums remain ordered.
The per-call CUDA pack replaces padded data with tile-rounded padded leaf rows,
retains padded bounds, and adds per-tile staging counters only with diagnostics.
Estimator live-buffer accounting and allocation-limit checks follow that same
plan; the public shape-only resource advisory is not a device-specific allocation
guarantee. Flat-kernel diagnostics count zero node pops and n scanned positions
per query including self; staged-position totals count copied candidates once per
query tile, not evaluated pairs. See [Phase 7 results](../archive/engineering/hdbscan_perf_report.md#phase-7-results-cuda-flat-plane-kernel-f2-accepted)
for historical measured scope and reported final-source verification. The raw
performance artifacts may be local and unavailable in a distributed checkout.

`HdbscanConfig::mst_graph` (`MstGraph::{Auto, Complete, KnnGraph}`, default `Auto`) selects
the graph and `knn_k` sets the kNN width; both are Rust-only fields except that Python
exposes `knn_k`. `Auto` resolves to `KnnGraph` above 30 features and to `Complete` otherwise.
`with_search` and `FittedHDBSCAN::search()` apply only to the complete-graph route.
`HdbscanConfig::reproducible` (default `false`) has the bound-mode meaning described above.

`hdbscan::estimate_fit_resources(n_finite, features, &config, backend_choice)` returns the same
shape-only `FitResourceEstimate` without reading data or touching a device. `knn` is `None` on the
complete route. With the crate's `tracing` feature, `fit` emits one `tracing::warn!` event carrying the
same advisory fields; without that feature `fit` emits nothing. The same exclusions and CPU/CUDA
semantics described for Python apply.

```shell
cargo test -p hdbscan
CUSCAN_REQUIRE_CUDA=1 cargo test -p hdbscan
cargo bench -p hdbscan --bench hdbscan_cpu
CUSCAN_REQUIRE_CUDA=1 cargo bench -p hdbscan --bench hdbscan_cuda
```

Criterion benchmarks compare forced routes on the same fixtures and seeded
synthetic data, with separate host tree-build timings. Use
`CUSCAN_BENCH_CASES` with comma-separated case-name substrings to select
cases; full-fit timings include upload, device execution and host readback.
Set `CUSCAN_BENCH_STAGES=1` for isolated neighbor, single candidate-round
and synchronized tree-upload timings. Candidate-stage cases use singleton
components, not total MST construction; upload timings include readback.
The default `reproducible=false` mode uses shared component bounds on the complete-graph KD
route. Both complete-graph candidate routes feed a two-pass device component-winner
reduction using stable host CSR membership. Only compact ordered winners return to the
host DSU; candidate tensors remain device-resident. CUDA uses shared-tree reductions;
CPU uses barrier-free tiled folds. Metadata upload and two dispatches can outweigh the
smaller readback, particularly in early rounds; this is not a speedup guarantee.
The tracing profiler reports per-round metadata prepare/upload, candidate kernel,
component reduction, winner readback and host merge scopes. Kernel scopes include
profiling synchronization, not isolated device execution time. The retrospective
`analyze/analyze_recovery.py` byte model is explicitly historical (pre-M7), not a
component-reduction transfer model.

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
