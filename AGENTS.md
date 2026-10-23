# Repository Guidelines

## Project Overview

`fast-umap` is a Rust library for Standard (non-parametric) UMAP following umap-learn 0.5.12 semantics, with a Python extension. It has CUDA and CubeCL LLVM CPU execution.

## Architecture & Data Flow

- The workspace contains backend-free `fast-umap-core` and `fast-hdbscan-core`, shared runtime crate `fast-cubecl-common`, facades `fast-umap` and `fast-hdbscan`, and the `fast-umap-python` extension. Python depends on both facades; each facade depends on its own core and common, never on the other facade. HDBSCAN core owns host extraction, finite-row remapping, MST helpers and deterministic KD-tree construction; its independent exact MST oracle requires the non-default `test-support` feature. The HDBSCAN facade owns the Rust `HDBSCAN`/`FittedHDBSCAN` API and exact brute-force/KD device search; there is no HDBSCAN Python API. UMAP core owns configuration, errors, model state, sizing and host spectral initialization; HDBSCAN must not acquire its spectral dependencies.
- `crates/fast-umap/src/standard_umap/` owns the `UMAP`/`FittedUMAP` estimator. Approximate neighbors, query, fuzzy graph and layout remain in UMAP's forward-only `StandardUmapOps`; common owns exact neighbors and squared distance blocks through `NeighborOps`. Each extension trait and its `CubeBackend` implementation stay in the same crate (orphan rule). UMAP exact search explicitly uses unsquared mode; common squared mode has different infinity/padding semantics. Kernels import host helpers from cores, never estimator modules. Spectral initialization remains host-side. The CUDA layout kernel updates embedding/clock/RNG operands in place, so callers must pass unaliased buffers. Feature values are used as-is, without scaling.
- `fast-cubecl-common` owns backend choice/errors, CUDA discovery and device resolution. CPU and CUDA implementations are always compiled; UMAP maps common errors locally to preserve its error contract.


Trace actual consumers before documenting or changing configuration behavior.

## Key Directories

- `crates/fast-umap-core/`: backend-free host crate and its tests (`test_standard_init`).
- `crates/fast-umap/src/`: Rust API, estimator, and backend implementations.
- `crates/fast-umap-python/`: PyO3 extension crate; `python/fast_umap/`: Python package; `python/tests/`: Python tests.
- `crates/fast-umap/tests/`: Rust integration tests for Standard UMAP neighbors, graph, serialization, layout, query, dispatch, estimator and quality.
- `crates/fast-umap/examples/`: Standard UMAP profiling and ANN investigation programs.
- `crates/fast-umap/benches/`: Criterion benchmarks; `scripts/`: comparison helpers.

## Development Commands

Run from the repository root. CPU and CUDA backends are always compiled; only optional capabilities such as tracing need feature selection:

```sh
cargo build -p fast-umap --release
cargo test -p fast-umap-core
cargo test -p fast-cubecl-common
cargo test -p fast-hdbscan-core
cargo test -p fast-hdbscan
cargo test -p fast-umap
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-umap
cargo bench -p fast-umap --bench standard_umap_cpu
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
- `crates/fast-umap-core/src/`: configuration, errors, model state, query/ANN sizing, spectral initialization.
- `crates/fast-umap/src/standard_umap/`: estimator, backend choice, neighbors.
- `pyproject.toml` / `uv.lock`: Maturin packaging and Python QA tooling.

## Runtime/Tooling Preferences

Use Cargo for Rust (edition 2024) and uv for Python (3.10+). Maturin builds the native Python module. No Rust toolchain pin or CI configuration is present.

CPU (CubeCL LLVM) and CUDA backends are always compiled; default-feature selection does not disable either backend. The local LLVM bundle/build/JIT path can require network access, cache space, and LLVM support. The documented deployment target is Linux x86-64 including WSL2; do not claim wheels or other platform support. CUDA still needs NVIDIA runtime support when selected and is not initialized merely by compiling it. Standard backend selection is Auto/Cpu/Cuda: Auto prefers discovered usable CUDA and uses CPU only when discovery confirms CUDA unavailable; explicit selections are strict, with no execution fallback.

Burn/CubeCL requirements are declared once in `[workspace.dependencies]`; keep them aligned there. Both cores are backend-free. Internal dependency edges may disable unrelated default features; backend availability is unconditional. Python forwards optional features such as tracing to both facades.

## Testing & QA

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Standard CPU tests use the CubeCL LLVM CPU runtime and do not require a GPU.

For required CUDA coverage rather than silent hardware skips:

```sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, and validation failures. New backend operations should share assertions across CPU and CUDA. Use `tempfile` for new filesystem tests.

Run focused tests for changed behavior and CPU and CUDA runtime behavior after backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
