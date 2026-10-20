# Repository Guidelines

## Project Overview

`fast-umap` is a Rust library for Standard (non-parametric) UMAP following umap-learn 0.5.12 semantics, with a Python extension. It has CUDA and CubeCL LLVM CPU execution.

## Architecture & Data Flow

- Cargo workspace with three crates: `crates/fast-umap-core` (backend-free host code: `UMAPConfig`, `UmapError`, validation, curve/width/budget math, `ModelState`, ANN/query sizing helpers, host spectral initialization), `crates/fast-umap` (facade: `StandardUmapOps`, kernels, CUDA discovery, `BackendChoice`, estimator, neighbors; re-exports every core item at its existing path) and `crates/fast-umap-python` (PyO3 extension). Dependency graph: python → fast-umap → fast-umap-core. `StandardUmapOps` and its only `impl` for `CubeBackend` must stay in the same crate (orphan rule).
- `crates/fast-umap/src/standard_umap/` owns the `UMAP`/`FittedUMAP` estimator. Neighbor search, fuzzy graph and layout use the forward-only `StandardUmapOps` extension (`crates/fast-umap/src/backend.rs`) and the CubeCL kernels in `crates/fast-umap/src/kernels/`. Kernels import host helpers from `fast_umap_core`, never from `crate::standard_umap`. Spectral initialization is host-side (`crates/fast-umap-core/src/init.rs`). The CUDA layout kernel updates its embedding/clock/RNG operands in place, so callers must pass unaliased buffers. Fit/transform use caller-provided feature values as-is; there is no feature scaling.
- `crates/fast-umap/src/cuda_backend.rs` provides CUDA device discovery used by backend selection (`standard_umap/backend_choice.rs`).


Trace actual consumers before documenting or changing configuration behavior.

## Key Directories

- `crates/fast-umap-core/`: backend-free host crate and its tests (`test_standard_init`).
- `crates/fast-umap/src/`: Rust API, estimator, and backend implementations.
- `crates/fast-umap-python/`: PyO3 extension crate; `python/fast_umap/`: Python package; `python/tests/`: Python tests.
- `crates/fast-umap/tests/`: Rust integration tests for Standard UMAP neighbors, graph, serialization, layout, query, dispatch, estimator and quality.
- `crates/fast-umap/examples/`: Standard UMAP profiling and ANN investigation programs.
- `crates/fast-umap/benches/`: Criterion benchmarks; `scripts/`: feature checks and comparison helpers.

## Development Commands

Run from the repository root. Specify features explicitly:

```sh
cargo build -p fast-umap --release --no-default-features --features cpu
cargo test -p fast-umap-core
cargo test -p fast-umap --no-default-features --features cpu
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features fast-umap/cpu
cargo test -p fast-umap --no-default-features --features cuda
cargo bench -p fast-umap --no-default-features --features cpu --bench standard_umap_cpu
bash scripts/test_features.sh
```

Python development and checks:

```sh
uv sync
uv run maturin develop --release --no-default-features --features cpu
uv run --no-sync pytest python/tests/test_standard_*.py
uv run ruff check
uv run ruff format --check
uv run ty check
```

Run Python scripts with `uv`. For scripts that need the release-built extension, build it first and disable syncing for the script run so uv does not replace it:

```sh
uv run maturin develop --release --no-default-features --features cpu && uv run --no-sync myscript.py
```

## References
- [Burn Tensor ops](references/burn_tensor_ops.md): A list of available tensor ops and utilities for Burn Tensors, along with their pytorch equivalents.

## Code Conventions & Common Patterns

- Use standard rustfmt formatting, Rust naming conventions, module documentation, and existing configuration struct-update patterns (`..Default::default()`). Python uses Ruff with a 100-character line limit.
- Burn uses dynamic dispatch: `Tensor<2>` and `Device`, not backend-generic public tensors.
- Backend selection is explicit through `BackendChoice`; keep backend-specific logic behind feature gates and backend extensions.
- Standard UMAP returns `Result<_, UmapError>` for invalid input, configuration, backend, and numerical failures. Preserve intentional error contracts and test invalid inputs; do not introduce silent recovery.

## Important Files

- `Cargo.toml` / `Cargo.lock`: virtual workspace manifest and single lockfile; member manifests in `crates/*/Cargo.toml` own features and example prerequisites; shared requirements live in `[workspace.dependencies]` and members use `{ workspace = true }`.
- `crates/fast-umap-core/src/`: configuration, errors, model state, query/ANN sizing, spectral initialization.
- `crates/fast-umap/src/standard_umap/`: estimator, backend choice, neighbors.
- `pyproject.toml` / `uv.lock`: Maturin packaging and Python QA tooling.

## Runtime/Tooling Preferences

Use Cargo for Rust (edition 2024) and uv for Python (3.10+). Maturin builds the native Python module. No Rust toolchain pin or CI configuration is present.

The `fast-umap` and `fast-umap-python` defaults are `cuda,cpu`; `fast-umap-core` has no backend features. The `cpu` feature selects CubeCL's LLVM CPU runtime; a CPU-only Standard build uses `-p fast-umap --no-default-features --features cpu`. The local LLVM bundle/build/JIT path can require network access, cache space, and LLVM support. The documented deployment target is Linux x86-64 including WSL2; do not claim wheels or other platform support. `cuda` needs NVIDIA runtime support. Standard backend selection is Auto/Cpu/Cuda: Auto prefers discovered usable CUDA and uses CPU only when discovery confirms CUDA unavailable; explicit selections are strict, with no execution fallback.

Burn/CubeCL requirements are declared once in `[workspace.dependencies]`; keep them aligned there.

## Testing & QA

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Standard CPU tests use the CubeCL LLVM CPU feature and do not require a GPU.

For required CUDA coverage rather than silent hardware skips:

```sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-umap --lib cuda_backend -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, and validation failures. New backend operations should share assertions across CPU and CUDA. Use `tempfile` for new filesystem tests.

Run focused tests for changed behavior and the feature matrix after feature/backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
