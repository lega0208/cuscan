# Repository Guidelines

## Project Overview

`fast-umap` is a Rust library for parametric and non-parametric Standard UMAP, with a Python extension. A Burn neural network learns an embedding and can transform unseen samples. Standard UMAP has CUDA and CubeCL LLVM CPU execution; do not infer parametric CPU runtime acceptance from Standard CPU support.

## Architecture & Data Flow

- `src/parametric_umap/` owns `ParametricUMAP`, `FittedParametricUMAP`, `ParametricUMAPConfig`, the neural model, private sparse trainer, and weight persistence.
- `src/standard_umap/` owns the separate non-parametric `UMAP`/`FittedUMAP` estimator (umap-learn 0.5.12 semantics). Neighbor search, fuzzy graph and layout use the forward-only `StandardUmapOps` extension (`src/backend.rs`) and shared CubeCL kernels. Spectral initialization is host-side. The CUDA layout kernel updates its embedding/clock/RNG operands in place, so callers must pass unaliased buffers.
- `src/graph/` owns shared graph/math APIs such as `Metric`, `GraphParams`, and `ManifoldParams`; fitting delegates shared graph preparation and parametric training to their respective modules.
- Sparse training uses caller-provided feature values as-is and computes a global self-inclusive k-NN graph once (`n_neighbors >= 2`). It retains distances, computes smooth-kNN rho/sigma and directed memberships, then mixes fuzzy union/intersection using `local_connectivity`, `bandwidth`, and `set_op_mix_ratio` (defaults 1). Configured positive epochs E fix float32 pruning and `floor(E*w)` counts; checked u64 prefix tickets drive fresh capped replacement positives and uniform non-self negatives each epoch. Empty support/overflow is an explicit failure, not a fallback. Optional `nn-descent` handles approximate neighbor search for large inputs; `pca` enables a warm start. Graph-helper parity on identical neighbor lists does not imply full reference training parity.
- The Rust and Python estimator fit/transform paths do not fit or apply feature scaling, and fitted models have no normalization statistics. The parametric model uses Linear/ReLU hidden layers and a linear output layer; saved and loaded weights operate consistently on caller-provided inputs.
- `src/backend.rs` defines the `UmapOps` backend extension. `src/kernels/` implements shared CubeCL kernels; `src/autodiff_ops.rs` supplies gradients.

**Behavioral caveats:** some exposed graph/optimization settings are not honored by training (including metric selection and public API batch size). Trace actual consumers before documenting or changing configuration behavior.

## Key Directories

- `src/`: Rust API, model, training, tensor utilities, and backend implementations.
- `bindings/python/`: PyO3 extension crate; `python/fast_umap/`: Python package; `python/tests/`: Python tests.
- `tests/`: Rust integration tests for tensor operations, gradients, fitting, validation, and serialization.
- `examples/`: backend selection, training, serialization, and benchmark programs.
- `benches/`: Criterion benchmarks and recorded results; `scripts/`: feature checks and benchmark/example runners.

## Development Commands

Run from the repository root. Specify features explicitly:

```sh
cargo build --release --no-default-features --features cpu
cargo test --no-default-features --features cpu
cargo fmt --all -- --check
cargo clippy --no-default-features --features cpu --all-targets
cargo run --release --no-default-features --features cpu --example cpu_training_demo
cargo run --release --no-default-features --features gpu --example simple
cargo test --no-default-features --features gpu
cargo bench --no-default-features --bench umap_bench
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

`scripts/run_all.sh` and `scripts/bench.sh` write outputs and may download MNIST. Their feature assumptions are stale; prefer explicit Cargo commands and check each example's `required-features` in `Cargo.toml`.

## References
- [Burn Tensor ops](references/burn_tensor_ops.md): A list of available tensor ops and utilities for Burn Tensors, along with their pytorch equivalents.

## Code Conventions & Common Patterns

- Use standard rustfmt formatting, Rust naming conventions, module documentation, and existing configuration struct-update patterns (`..Default::default()`). Python uses Ruff with a 100-character line limit.
- Burn uses dynamic dispatch: `Tensor<2>` and `Device`, not backend-generic public tensors. Training uses `.autodiff()`; inference uses `model.valid()` and `.without_autodiff()`.
- Device selection is explicit through `ParametricUMAP::with_device`; keep backend-specific logic behind feature gates and backend extensions.
- Data conversion uses row-major flat vectors plus sample/feature counts; reuse utilities in `src/utils.rs`.
- Existing fitting/shape validation panics, while save/load returns `Result`. Preserve intentional error contracts and test invalid inputs; do not introduce silent recovery.
- Training is synchronous. Cancellation uses a crossbeam receiver; progress uses a `ParametricUMAPEpochProgress` callback. Learned model state belongs in `FittedParametricUMAP`.

## Important Files

- `Cargo.toml` / `Cargo.lock`: workspace, dependencies, features, and example prerequisites.
- `src/parametric_umap/`: estimator/configuration, neural model/training, and Burn weight persistence.
- `src/graph/`: shared graph/math/configuration; `src/prelude.rs`: public convenience exports.
- `pyproject.toml` / `uv.lock`: Maturin packaging and Python QA tooling.

## Runtime/Tooling Preferences

Use Cargo for Rust (edition 2024) and uv for Python (3.10+). Maturin builds the native Python module. No Rust toolchain pin or CI configuration is present.

The manifest defaults are `cuda,cpu,verbose`. The `cpu` feature selects CubeCL's LLVM CPU runtime; a CPU-only Standard build uses `--no-default-features --features cpu`. The local LLVM bundle/build/JIT path can require network access, cache space, and LLVM support. The documented deployment target is Linux x86-64 including WSL2; do not claim wheels or other platform support. `gpu` enables WGPU; `cuda` needs NVIDIA runtime support. Standard backend selection is Auto/Cpu/Cuda: Auto prefers discovered usable CUDA and uses CPU only when discovery confirms CUDA unavailable; explicit selections are strict, with no execution fallback. WGPU is not newly supported for Standard UMAP.

Burn/CubeCL dependencies use exact prerelease pins; keep related versions aligned across root and binding manifests. Tests have a Git-sourced MNIST dependency, so initial dependency resolution can require network access.

## Testing & QA

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Standard CPU tests use the CubeCL LLVM CPU feature and do not require a GPU. WGPU dispatch/gradient tests require `gpu` and an available adapter.

For required CUDA coverage rather than silent hardware skips:

```sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test --lib cuda_backend -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, gradients, validation failures, cancellation, and persistence errors. New backend operations should share forward/gradient assertions across devices. Use `tempfile` for new filesystem tests; existing serialization tests contain potentially colliding `/tmp` paths.

Run focused tests for changed behavior and the feature matrix after feature/backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
