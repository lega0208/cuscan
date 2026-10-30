# Repository Guidelines

## Project Overview

`cuscan` provides Standard (non-parametric) UMAP following umap-learn 0.5.12 semantics and HDBSCAN, with Rust facades and a Python extension. It has CUDA and CubeCL LLVM CPU execution.

## Architecture & Data Flow

- The workspace contains backend-free `umap-core` and `hdbscan-core`, shared runtime crate `cuscan-common`, facades `umap` and `hdbscan`, and the `cuscan-python` extension. Python depends on both facades; each facade depends on its own core and common, never on the other facade. HDBSCAN core owns host extraction, finite-row remapping, MST helpers and deterministic KD-tree construction; its independent exact MST oracle requires the non-default `test-support` feature. The HDBSCAN facade owns the Rust `HDBSCAN`/`FittedHDBSCAN` API, exact brute-force/KD device search and graph routing; the Python HDBSCAN wrapper forwards validated options through the extension. UMAP core owns configuration, errors, model state, sizing and host spectral initialization; HDBSCAN must not acquire its spectral dependencies.
- `crates/umap/src/standard_umap/` owns the `UMAP`/`FittedUMAP` estimator. Query, fuzzy graph and layout remain in UMAP's forward-only `StandardUmapOps`; common owns approximate neighbors (including candidate-width sizing), exact neighbors and squared distance blocks through `NeighborOps`. ANN callers validate launches with common's typed preflight before dispatch. Each extension trait and its `CubeBackend` implementation stay in the same crate (orphan rule). UMAP neighbor search explicitly uses unsquared mode; common exact squared mode has different infinity/padding semantics. Kernels import host helpers from cores, never estimator modules. Spectral initialization remains host-side. The CUDA layout kernel updates embedding/clock/RNG operands in place, so callers must pass unaliased buffers. Feature values are used as-is, without scaling.
- `cuscan-common` owns backend choice/errors, CUDA discovery and device resolution. CPU and CUDA implementations are always compiled; UMAP maps common errors locally to preserve its error contract.

Trace actual consumers before documenting or changing configuration behavior.

HDBSCAN `approximate_predict` CPU KD queries use the C16 leaf-SoA candidate scan in
input query order, including cold prediction-core batches. CUDA queries retain the
padded-row per-query kernel. The call-local SoA replaces padded reference rows;
checked prediction planning includes its lane-rounded IDs/data and the transient
padded-row preparation peak. Vector/scalar contraction can change rounding and
near-tie prediction outputs; exact ordering is under each kernel's own arithmetic.

HDBSCAN KD neighbors retain CPU C16 (leaf SoA, leaf-block queries). CUDA uses the Phase 7 F2 flat plane kernel at 16–32 features when plane operations/synchronization and device resources permit: two planes per cube, no row-count gate. Other CUDA cases use the per-query kernel; this is pre-launch routing, not recovery from execution errors. Leaf-ordered packs preserve original IDs/output rows and scalar feature-sum order. Valid builder trees (or equivalent low-level permutation/tree preconditions) are required; no runtime content scan is added. The estimator consumes `kd_neighbor_plan`, `kd_neighbor_pack_bytes` and `kd_neighbor_pack_allocations` for matching checked live-buffer accounting and allocation limits: tile-rounded padded leaf rows replace padded data, padded bounds remain, and staging counters are diagnostics-only. F2 diagnostics report zero node pops, n scanned positions per query (including self), and separately summed staged positions per tile. The historical Phase 7 report describes final-source verification; its raw performance artifacts may be local and unavailable in a distributed checkout. Experimental H1/release/debug gates are not substituted for final-source proof.

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

Run from the repository root. Use Cargo. CPU and CUDA backends are always compiled; only optional capabilities such as tracing need feature selection:

```sh
cargo build -p umap --release
cargo test -p umap-core
cargo test -p cuscan-common
cargo test -p hdbscan-core
cargo test -p hdbscan
cargo test -p umap
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p umap
cargo bench -p umap --bench standard_umap_cpu
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

Use Cargo for Rust (edition 2024) and uv for Python (3.10+). Maturin builds the native Python module. `rust-toolchain.toml` pins `nightly-2026-08-08` (rustc `1a98b1e13`); `.cargo/config.toml` configures `/usr/sbin/clang`, mold and nightly-only flags for x86-64 Linux. Install those tools or deliberately adjust the local configuration. `.vscode/settings.json` is optional personal editor configuration; no soldr editor extension is required.

CPU (CubeCL LLVM) and CUDA backends are always compiled; default-feature selection does not disable either backend. The local LLVM bundle/build/JIT path can require network access, cache space, and LLVM support. The documented deployment target is Linux x86-64 including WSL2; the locally validated wheel baseline is manylinux 2.34, not a claim of other platform support or publication. CUDA still needs NVIDIA runtime support when selected and is not initialized merely by compiling it. Standard backend selection is Auto/Cpu/Cuda: Auto prefers discovered usable CUDA and uses CPU only when discovery confirms CUDA unavailable; explicit selections are strict, with no execution fallback.

Burn/CubeCL requirements are declared once in `[workspace.dependencies]`; keep them aligned there. Both cores are backend-free. Internal dependency edges may disable unrelated default features; backend availability is unconditional. Python forwards optional features such as tracing to both facades.

## Testing & QA

Native HDBSCAN hotspot reproduction (strict CPU/CUDA, Nsight and counter access):
`bash scripts/analyze_hdbscan_stages.sh --rows 20000`. After relevant changes, require
`bash scripts/analyze_hdbscan_stages.sh --rows 100000` final validation. The fixed synthetic
workload and per-child 10s feedback cap include runtime JIT, not Cargo compilation;
compilation-only or skipped CUDA execution is not validation. See
[HDBSCAN stage analysis](docs/development.md#hdbscan-stage-analysis) for final-scale limits.
Offline: `uv run --no-sync python analyze/summarize_hdbscan_stages.py --benchmark RUN/benchmark.json --kernels RUN/cuda_kernels.json --counters RUN/cuda_counters.json`.
Focused QA: `uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py`.

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Standard CPU tests use the CubeCL LLVM CPU runtime and do not require a GPU.

CI uses the `ci-ok` aggregator as the intended sole required check. Changes are classified so unaffected jobs can skip; the weekly nightly canary intentionally uses a floating nightly. For workflow changes use `scripts/ci/act.sh list` and the relevant scenarios, but act cannot validate Windows, trusted publishing or GPU execution. See [CI and release runbooks](docs/development.md#continuous-integration) for setup, CUDA and release requirements.

For required CUDA coverage rather than silent hardware skips:

```sh
CUSCAN_REQUIRE_CUDA=1 cargo test -p cuscan-common -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, and validation failures. New backend operations should share assertions across CPU and CUDA. Use `tempfile` for new filesystem tests.

Run focused tests for changed behavior and CPU and CUDA runtime behavior after backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
