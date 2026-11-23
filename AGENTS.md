# Repository Guidelines

## Project Overview

`cuscan` provides Standard (non-parametric) UMAP with recursive initialization and Adam layouts by default, plus umap-learn 0.6-style random/spectral initialization, compatibility SGD and momentum layouts, and HDBSCAN, with Rust facades and a Python extension. This is not full 0.6 feature or bit parity. It has CUDA and CubeCL LLVM CPU execution.

## Architecture & Data Flow

- The workspace contains backend-free `umap-core` and `hdbscan-core`, shared runtime crate `cuscan-common`, facades `umap` and `hdbscan`, and the `cuscan-python` extension. Python depends on both facades; each facade depends on its own core and common, never on the other facade. HDBSCAN core owns host extraction, finite-row remapping, MST helpers and deterministic KD-tree construction; its independent exact MST oracle requires the non-default `test-support` feature. The HDBSCAN facade owns the Rust `HDBSCAN`/`FittedHDBSCAN` API, exact brute-force/KD device search and graph routing; the Python HDBSCAN wrapper forwards validated options through the extension. UMAP core owns configuration, errors, model state, sizing and host spectral initialization; HDBSCAN must not acquire its spectral dependencies.
- `crates/umap/src/standard_umap/` owns the `UMAP`/`FittedUMAP` estimator. Query, fuzzy graph and layout remain in UMAP's forward-only `StandardUmapOps`; common owns approximate neighbors (including candidate-width sizing), and exact neighbors and squared distance blocks through `NeighborOps`. ANN fit resolves its budget once, passes a checked `AnnPlan` from preflight to the ANN operation, and uses one buffer table for planning and allocations. Each extension trait and its `CubeBackend` implementation stay in the same crate (orphan rule). UMAP neighbor search explicitly uses unsquared mode; common exact squared mode has different infinity/padding semantics. Kernels import host helpers from cores, never estimator modules. Spectral initialization remains host-side. Layout kernels are selected by capability before launch: fused plane kernel up to 32 components on 32-lane plane devices (CUDA; padded virtually above 16), head-cache kernel up to 32 components elsewhere (CPU), atomic kernel beyond; all update embedding/clock/RNG operands in place, so callers must pass unaliased buffers. Feature values are used as-is, without scaling.
- `cuscan-common` owns backend choice/errors, CUDA discovery and device resolution. CPU and CUDA implementations are always compiled; UMAP maps common errors locally to preserve its error contract.
- ANN distances use the shared fixed-order R8 helpers, including the group helper, in `crates/cuscan-common/src/kernels/ann/ann_distance.rs`; do not reimplement the reduction in individual kernels.
- CPU ANN forest and refinement joins use the owner route in `crates/cuscan-common/src/kernels/ann/ann_owner.rs`; CUDA retains the slot route. Preserve bit-identical graphs across routes and reuse the shared `cpu_row_apply` and R8 helpers rather than reimplementing them.
- UMAP and HDBSCAN's ANN branch share `nn_descent_params`, which uses the umap-learn 0.6 non-compatibility forest recipe and rinnd's effective candidate width.

Trace actual consumers before documenting or changing configuration behavior.

UMAP defaults to `Init::Auto` (resolves to recursive for every optimizer) and
`Optimizer::Adam`; a default model transforms with Adam. Explicit
`Optimizer::Compatibility` with `Init::Spectral` (Python
`optimizer="compatibility", init="spectral"`) gives the previous defaults.
Recursive initialization does host graph coarsening in
`umap-core::recursive_init`; `UMAP::fit` in the facade drives it and runs the
coarse Adam layouts on the device. `umap_core::init::initialize` is host-only:
it accepts explicit `Random`/`Spectral` and rejects `Recursive`/`Auto` with
`InvalidConfig`, so it errors on `UMAPConfig::default()`.
Modern layouts use compact positive sorted CSR edge positions for negative
hashing, source-only gradients and separate ordered steps, including arbitrary
embedding dimensions. Transform references are immutable even when H=N. Raw Q =
`negative_selection_range` < N enables per-epoch projection refresh; large layouts
(H>1024) use windows of effective R=min(N,max(Q,200)) (R=N is global), while
adaptation uses raw Q. Small source layouts (H<=1024) sample globally with unit
scale. Adaptation (default 128 samples, 0 disables) is Adam-fit-only, anchored
and applied before the gradient; explicit scale 0 is absorbing; momentum and
transform use the initial scale; automatic scale is R/N when adaptation is
disabled. `exclude_graph_neighbors` (default false) excludes CSR neighbors (and
self for fit only). UMAP persistence is strict schema v5; HDBSCAN predictor
persistence remains v1.
Modern coordinate kernels use each tensor's physical row and column strides;
CUDA uploads can be pitched (for example, 48 logical components need not have
a row stride of 48). Logical element counts must not index physical padding.

Approximate transform uses corrected bounded-frontier retention on the stored
forward graph; reverse edges and a larger default queue were not adopted. CUDA
packs 32 query lanes with slot-major scratch; CPU retains per-query geometry.
Transform pruning uses its actual epoch count, including counts at most 10,
unlike fit's short-epoch rule. D8 copies the embedding of the smallest training
ID with raw coalesced membership exactly 1.0; a required copy without a valid ID
returns a typed numerical error. Exact-search transform outputs can change too.
These changes do not alter the public API or v5
schema.

HDBSCAN `approximate_predict` CPU KD queries use the C16 leaf-SoA candidate scan in
input query order, including cold prediction-core batches. CUDA queries retain the
padded-row per-query kernel. The call-local SoA replaces padded reference rows;
checked prediction planning includes its lane-rounded IDs/data and the transient
padded-row preparation peak. Vector/scalar contraction can change rounding and
near-tie prediction outputs; exact ordering is under each kernel's own arithmetic.

HDBSCAN KD neighbors retain CPU C16 (leaf SoA, leaf-block queries). CUDA uses the per-query traversal kernel at every feature count; routing is selected before launch, not as recovery from execution errors. Leaf-ordered packs preserve original IDs/output rows and scalar feature-sum order. Valid builder trees (or equivalent low-level permutation/tree preconditions) are required; no runtime content scan is added. The estimator consumes `kd_neighbor_plan`, `kd_neighbor_pack_bytes` and `kd_neighbor_pack_allocations` for matching checked live-buffer accounting and allocation limits, including padded data and padded bounds. The historical Phase 7 report and subsequent removal evidence are in the archive; raw performance artifacts may be local and unavailable in a distributed checkout. Experimental gates are not substituted for final-source proof.

## Key Directories

- `crates/umap-core/`: backend-free host crate and its tests (`test_standard_init`).
- `crates/umap/src/`: Rust API, estimator, and backend implementations.
- `crates/hdbscan-core/` and `crates/hdbscan/`: backend-free HDBSCAN core and its facade.
- `crates/cuscan-common/`: shared backend and neighbor runtime.
- `crates/cuscan-python/`: PyO3 extension crate; `python/cuscan/`: Python package; `python/tests/`: Python tests.
- `crates/umap/tests/`: Rust integration tests for Standard UMAP neighbors, graph, serialization, layout, query, dispatch, estimator and quality.
- `crates/umap/examples/`: Standard UMAP profiling and ANN investigation programs.
- `crates/umap/benches/`: Criterion benchmarks; `scripts/`: comparison helpers.

## Development Commands

Run from the repository root. Prefer `soldr cargo` whenever `soldr` is available on PATH; use plain `cargo` only when `soldr` is unavailable or keeps failing. CPU and CUDA backends are always compiled; only optional capabilities such as tracing need feature selection:

```sh
soldr cargo build -p umap --release
soldr cargo test -p umap-core
soldr cargo test -p cuscan-common
soldr cargo test -p hdbscan-core
soldr cargo test -p hdbscan
soldr cargo test -p umap
soldr cargo fmt --all -- --check
soldr cargo clippy --workspace --all-targets
soldr cargo test -p umap
soldr cargo bench -p umap --bench standard_umap_cpu
```

Python development and checks:

```sh
uv sync
uv run maturin develop --release
uv run --no-sync pytest python/tests/test_standard_*.py
uv run ruff check
uv run ruff format --check
uv run ty check
```

Run Python scripts with `uv`. For scripts that need the release-built extension, build it first and disable syncing for the script run so uv does not replace it:

```sh
uv run maturin develop --release && uv run --no-sync myscript.py
```

## References

- [Burn Tensor ops](references/burn_tensor_ops.md): A list of available tensor ops and utilities for Burn Tensors, along with their pytorch equivalents.

## Code Conventions & Common Patterns

- Use standard rustfmt formatting, Rust naming conventions, module documentation, and existing configuration struct-update patterns (`..Default::default()`). Python uses Ruff with a 100-character line limit.
- Burn uses dynamic dispatch: `Tensor<2>` and `Device`, not backend-generic public tensors.
- Backend selection is explicit through `BackendChoice`; CPU and CUDA code are always compiled, with runtime choice controlling discovery and execution.
- Standard UMAP returns `Result<_, UmapError>` for invalid input, configuration, backend, and numerical failures. Preserve intentional error contracts and test invalid inputs; do not introduce silent recovery.

## Important Files

- `Cargo.toml` / `Cargo.lock`: virtual workspace manifest and single lockfile; member manifests in `crates/*/Cargo.toml` own features and example prerequisites; shared requirements live in `[workspace.dependencies]` and members use `{ workspace = true }`.
- `crates/umap-core/src/`: configuration, errors, model state, query/ANN sizing, spectral initialization.
- `crates/umap/src/standard_umap/`: estimator, backend choice, neighbors.
- `pyproject.toml` / `uv.lock`: Maturin packaging and Python QA tooling.

## Runtime/Tooling Preferences

Use Cargo for Rust (edition 2024) and uv for Python (3.12+). Maturin builds the native Python module. `rust-toolchain.toml` pins `nightly-2026-08-08` (rustc `1a98b1e13`); `.cargo/config.toml` configures `/usr/sbin/clang`, mold and nightly-only flags for x86-64 Linux. Install those tools or deliberately adjust the local configuration. `.vscode/settings.json` is optional personal editor configuration; no soldr editor extension is required.

CPU (CubeCL LLVM) and CUDA backends are always compiled; default-feature selection does not disable either backend. The local LLVM bundle/build/JIT path can require network access, cache space, and LLVM support. The documented deployment target is Linux x86-64 including WSL2; the locally validated wheel baseline is manylinux 2.34, not a claim of other platform support or publication. CUDA still needs NVIDIA runtime support when selected and is not initialized merely by compiling it. Standard backend selection is Auto/Cpu/Cuda: Auto prefers discovered usable CUDA and uses CPU only when discovery confirms CUDA unavailable; explicit selections are strict, with no execution fallback.

Burn/CubeCL requirements are declared once in `[workspace.dependencies]`; keep them aligned there. Both cores are backend-free. Internal dependency edges may disable unrelated default features; backend availability is unconditional. Python forwards optional features such as tracing to both facades.

## Testing & QA

Native HDBSCAN hotspot reproduction (strict CPU/CUDA, Nsight and counter access):
`bash scripts/analyze_hdbscan_stages.sh --rows 20000`. After relevant changes, require
`bash scripts/analyze_hdbscan_stages.sh --rows 100000` final validation. The fixed synthetic
`umap_like_v1` workload and per-child 10s feedback cap include runtime JIT, not Cargo compilation;
compilation-only or skipped CUDA execution is not validation. See
[HDBSCAN stage analysis](docs/development.md#hdbscan-stage-analysis) for final-scale limits.
Offline: `uv run --no-sync python analyze/summarize_hdbscan_stages.py --benchmark RUN/benchmark.json --kernels RUN/cuda_kernels.json --counters RUN/cuda_counters.json`.
Focused QA: `uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py`.

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Standard CPU tests use the CubeCL LLVM CPU runtime and do not require a GPU.

CI uses the `ci-ok` aggregator as the intended sole required check. Changes are classified so unaffected jobs can skip; the weekly nightly canary intentionally uses a floating nightly. For workflow changes use `scripts/ci/act.sh list` and the relevant scenarios, but act cannot validate Windows, trusted publishing or GPU execution. See [CI and release runbooks](docs/development.md#continuous-integration) for setup, CUDA and release requirements.

Release Please owns the reviewed release PR, tag and GitHub release notes. Cargo
workspace version is build authority; version.txt, root manifest proposal and all
six workspace lock entries must agree in release PRs. Never execute candidate
release-branch scripts during lock synchronization. Keep App credentials out of
Cargo and use a leased Cargo.lock-only push. Production remains tag-only; dispatch
is TestPyPI rehearsal. Never create a missing release, overwrite assets, blanket
skip existing PyPI files or move tags to recover publication. Initial 0.1.0 needs
the bootstrap cleanup and owner prerequisites in the release activation checklist.

For required CUDA coverage rather than silent hardware skips:

```sh
CUSCAN_REQUIRE_CUDA=1 cargo test -p cuscan-common -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, and validation failures. New backend operations should share assertions across CPU and CUDA. Use `tempfile` for new filesystem tests.

Run focused tests for changed behavior and CPU and CUDA runtime behavior after backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
