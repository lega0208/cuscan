# Repository Guidelines

## Project Overview

`fast-umap` is a Rust library for GPU-accelerated **parametric UMAP**, with a Python extension. A Burn neural network learns an embedding and can transform unseen samples. Flex provides real CPU training; WGPU and CUDA provide accelerated backends.

## Architecture & Data Flow

- `src/lib.rs` exposes `Umap`, `FittedUmap`, input validation, fitting, transformation, and persistence. Prefer this API over the deprecated `UMAP` API.
- Fitting validates rectangular input, seeds the device, flattens rows, creates the MLP, maps `UmapConfig` into training configuration, and calls `src/train/train_sparse.rs`.
- Sparse training uses caller-provided feature values as-is and computes a global self-inclusive k-NN graph once (`n_neighbors >= 2`). It retains distances, computes smooth-kNN rho/sigma and directed memberships, then mixes fuzzy union/intersection using `local_connectivity`, `bandwidth`, and `set_op_mix_ratio` (defaults 1). Configured positive epochs E fix float32 pruning and `floor(E*w)` counts; checked u64 prefix tickets drive fresh capped replacement positives and uniform non-self negatives each epoch. Empty support/overflow is an explicit failure, not a fallback. Optional `nn-descent` handles approximate neighbor search for large inputs; `pca` enables a warm start. Graph-helper parity on identical neighbor lists does not imply full reference training parity.
- The primary `Umap` and Python fit/transform paths do not fit or apply feature scaling, and fitted models have no normalization statistics. `src/model.rs` implements Linear/ReLU hidden layers and a linear output layer; saved and loaded weights therefore operate consistently on the caller-provided inputs. The deprecated dense `UMAP` route remains distinct, and existing scaling helpers remain available; neither implies scaling in the primary path.
- `src/backend.rs` defines the `UmapOps` backend extension. `src/generic_backend.rs` and `src/flex_backend.rs` implement Flex operations; `src/kernels/` implements CubeCL GPU kernels; `src/autodiff_ops.rs` supplies gradients.
- `src/train/mod.rs` also contains the legacy dense trainer. Avoid confusing its batch-local graph and disk checkpoints with the sparse public API path.

**Behavioral caveats:** several exposed graph/optimization settings are not honored by training (including metric selection and public API batch size). Trace actual consumers before documenting or changing configuration behavior. The feature-gated `src/cpu_backend.rs` is a placeholder, not the Flex training implementation.

## Key Directories

- `src/`: Rust API, model, training, tensor utilities, and backend implementations.
- `bindings/python/`: PyO3 extension crate; `python/fast_umap/`: Python package; `python/tests/`: Python tests.
- `tests/`: Rust integration tests for tensor operations, gradients, fitting, validation, and serialization.
- `examples/`: backend selection, training, serialization, and benchmark programs.
- `benches/`: Criterion benchmarks and recorded results; `scripts/`: feature checks and benchmark/example runners.

## Development Commands

Run from the repository root. Specify features explicitly:

```sh
cargo build --release --no-default-features
cargo test --no-default-features
cargo fmt --all -- --check
cargo clippy --no-default-features --all-targets
cargo run --release --no-default-features --example cpu_training_demo
cargo run --release --no-default-features --features gpu --example simple
cargo test --no-default-features --features gpu
cargo bench --no-default-features --bench umap_bench
bash scripts/test_features.sh
```

Python development and checks:

```sh
uv sync
uv run pytest
uv run ruff check
uv run ruff format --check
uv run ty check
```

Run Python scripts with `uv`. For scripts that need the release-built extension, build it first and disable syncing for the script run so uv does not replace it:

```sh
uv run maturin develop --release && uv run --no-sync myscript.py
```

`scripts/run_all.sh` and `scripts/bench.sh` write outputs and may download MNIST. Their feature assumptions are stale; prefer explicit Cargo commands and check each example's `required-features` in `Cargo.toml`.

## Code Conventions & Common Patterns

- Use standard rustfmt formatting, Rust naming conventions, module documentation, and existing configuration struct-update patterns (`..Default::default()`). Python uses Ruff with a 100-character line limit.
- Burn uses dynamic dispatch: `Tensor<2>` and `Device`, not backend-generic public tensors. Training uses `.autodiff()`; inference uses `model.valid()` and `.without_autodiff()`.
- Device selection is explicit through `Umap::with_device`; keep backend-specific logic behind feature gates and backend extensions.
- Data conversion uses row-major flat vectors plus sample/feature counts; reuse utilities in `src/utils.rs`.
- Existing fitting/shape validation panics, while save/load returns `Result`. Preserve intentional error contracts and test invalid inputs; do not introduce silent recovery.
- Training is synchronous. Cancellation uses a crossbeam receiver; progress uses an `EpochProgress` callback. Learned model and normalization state belong in `FittedUmap`.

## Important Files

- `Cargo.toml` / `Cargo.lock`: workspace, dependencies, features, and example prerequisites.
- `src/train/config.rs`: nested configuration, defaults, metrics, and UMAP curve fitting.
- `src/train/get_distance_by_metric.rs`: distance dispatch and CPU neighbor selection.
- `src/serialize.rs`: Burn weight persistence; `src/prelude.rs`: public convenience exports.
- `pyproject.toml` / `uv.lock`: Maturin packaging and Python QA tooling.

## Runtime/Tooling Preferences

Use Cargo for Rust (edition 2021) and uv for Python (3.10+). Maturin builds the native Python module. No Rust toolchain pin or CI configuration is present.

The manifest defaults are currently `cuda,verbose`, despite older documentation claiming WGPU defaults. `--no-default-features` selects a portable Flex build. `gpu` enables WGPU; `cuda` needs NVIDIA runtime support. The `all` feature does not include CUDA, but adding it without disabling defaults still retains CUDA. Python bindings currently inherit CUDA defaults and also enable WGPU.

Burn/CubeCL dependencies use exact prerelease pins; keep related versions aligned across root and binding manifests. Tests have a Git-sourced MNIST dependency, so initial dependency resolution can require network access.

## Testing & QA

Rust uses `#[test]` integration/unit tests and Criterion benchmarks; Python uses pytest. Most Rust tests explicitly use Flex and need no GPU. WGPU dispatch/gradient tests require `gpu` and an available adapter.

For required CUDA coverage rather than silent hardware skips:

```sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test --no-default-features --features cuda --lib cuda_backend -- --nocapture
```

Test observable shape, finite outputs, numerical tolerances, deterministic seeds, gradients, validation failures, cancellation, and persistence errors. New backend operations should share forward/gradient assertions across devices. Use `tempfile` for new filesystem tests; existing serialization tests contain potentially colliding `/tmp` paths.

Run focused tests for changed behavior and the feature matrix after feature/backend changes. There is no configured coverage threshold; do not substitute benchmark results or skipped hardware tests for correctness evidence.
