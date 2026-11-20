# cuscan

cuscan 0.1.0 provides non-parametric [UMAP](https://github.com/lmcinnes/umap) dimensionality reduction and [HDBSCAN](https://github.com/TutteInstitute/fast_hdbscan) clustering in Rust and Python. UMAP defaults to recursive initialization (host graph coarsening with Adam layouts at coarse levels) followed by an Adam layout, with umap-learn 0.6-style random or spectral initialization and rescaling, compatibility SGD, and momentum layouts also available; it does not implement all 0.6 features or promise bit parity. The implementation uses [Burn](https://github.com/tracel-ai/burn) and [CubeCL](https://github.com/tracel-ai/cubecl) for CPU and CUDA execution.

## Local build

The documented local target is Linux x86-64, including WSL2. Python 3.12+, uv and Maturin are required. `rust-toolchain.toml` pins `nightly-2026-08-08` (rustc `1a98b1e13`), and the x86-64 Linux Cargo configuration requires `/usr/sbin/clang` and mold and enables nightly flags. From the repository root:

```sh
uv sync
uv run maturin develop --release
```

Use `uv run --no-sync` for subsequent scripts or tests so syncing does not replace the locally built extension. The CubeCL LLVM bundle/build/JIT path can require network access, cache space and LLVM support. CPU and CUDA are both compiled; CUDA execution requires a working NVIDIA runtime and device. Compilation alone does not initialize CUDA.

### Wheel validation

The lowest validated Linux x86-64 wheel baseline is **manylinux 2.34**. The manylinux 2.28 image fails to build because `tracel-llvm-23.1.0-4`'s
`llvm-config` requires GLIBC 2.32, 2.33 and 2.34.

## Python

```python
import numpy as np
from cuscan import HDBSCAN, UMAP

X = np.random.default_rng(42).normal(size=(100, 10)).astype(np.float32)
mapper = UMAP(n_neighbors=15, random_state=42, backend="cpu")
embedding = mapper.fit_transform(X)
projected = mapper.transform(X[:5])

clusterer = HDBSCAN(min_cluster_size=5, backend="cpu").fit(X)
labels = clusterer.labels_
new_labels, strengths = clusterer.approximate_predict(X[:5])
```

Both estimators accept `backend="auto"`, `"cpu"`, or `"cuda"`. Auto prefers a discovered usable CUDA device and selects CPU only when CUDA is unavailable. Explicit choices are strict; execution errors do not trigger fallback.

UMAP fit defaults to `init="auto"`, which resolves to recursive initialization for every optimizer, including compatibility: host graph coarsening with CPU or CUDA Adam layouts at coarse levels. Rescaling defaults to `init_rescale="isotropic"` with `init_scale_factor=1.0`; `"per_axis"` and `"graph_edge"` are also available. `init="spectral"` and `init="random"` remain available. Python accepts `recursive_coarsening_ratio` (2, 3, or 4) and `recursive_init_kwds` with `base_init_threshold`, `n_embedding_epochs`, `hub_selection`, or `anchor_sample_size`. These controls do not affect transform initialization. cuscan does not promise bit parity with umap-learn; see [Standard UMAP](docs/development.md#standard-umap).

UMAP defaults to `optimizer="adam"`, so a default model also transforms with Adam. Pass `optimizer="compatibility", init="spectral"` to get the previous defaults (classic SGD after spectral initialization). `"adam"` and `"momentum"` use source-only modern layouts for fit and transform. When the fitted reference row count (after `unique=True`) exceeds `negative_selection_range` (default 200000), layouts with more than 1024 source rows draw negatives from per-epoch random-projection windows; smaller layouts, including small transform batches, sample globally with scale 1. `exclude_graph_neighbors=False` controls whether graph neighbors are excluded from negatives. `negative_sample_scale_adaptation_samples=128` adapts the scale during Adam fits only (0 disables); momentum and transform keep the initial scale. `negative_sample_scale=None` resolves the automatic scale; an explicit finite nonnegative scale is used as given (0 stays 0). Saved models retain these fit-time settings. Measured quality and performance results, and the items still open, are recorded in the [Standard UMAP notes](docs/development.md#standard-umap) and the [Phase 5](umap_06_phase5_measurement_gaps.md) and [Phase 6](umap_06_phase6_measurement_gaps.md) gap reports; full-seed quality is not claimed. Known limitation: transform quality trails upstream umap-learn; closing that gap is Phase 7 work (see the Phase 6 gap report, item N4).

### Fitted-model serialization

Install the optional HDF5 dependencies locally with `uv sync --extra serialization`, then rebuild with `uv run maturin develop --release`. The extra supplies `h5py` and `hdf5plugin`; ordinary fitting does not require them.

```python
mapper.save("umap.h5")
restored = UMAP.load("umap.h5", backend="cpu")
projected = restored.transform(X[:5])

predictor = clusterer.to_predictor()
predictor.save("predictor.h5")
```

`UMAP` saves fitted state; HDBSCAN serialization is for `HDBSCANPredictor`, not the fitted estimator or its fit outputs. Load a predictor with `HDBSCANPredictor.load(path, backend="auto", device_index=0)`. Backend and device are chosen at load time. `save` refuses to overwrite by default; `overwrite=True` explicitly replaces a file. Publication flushes a same-directory temporary file and commits atomically. On POSIX, a post-commit directory-sync failure reports uncertain durability. Windows flushes file data but does not guarantee directory-entry durability across a crash.

Files use a versioned HDF5 schema and lossless Blosc2 Zstandard compression, not pickle. External HDF5 readers need the Blosc2 filter plugin (ID 32026). Files contain training data or geometry and derived model state; protect them like the original data. Schema validation and compression do not authenticate files. Rust does not provide an HDF5 persistence API. Serialized numeric state does not guarantee identical execution across hardware, software versions or backends.

UMAP uses strict schema version 5 and rejects older UMAP files; retrain and save them in the new format. HDBSCAN predictor files remain version 1. There are no compatibility aliases for prior imports, crates or environment variables.

## Rust

The workspace exposes the `umap` and `hdbscan` facades. Build and test them locally:

```sh
cargo build --release -p umap -p hdbscan
cargo test -p umap
cargo test -p hdbscan
```

See [development and analysis](docs/development.md) for Rust API examples, detailed backend behavior, profiling, comparisons, serialization benchmarks and testing guidance. Historical benchmark measurements there are not cuscan rebenchmarks.

The [CI and release guidance](docs/development.md#continuous-integration) describes the intended `ci-ok` gate, local act, independent CUDA validation, trusted publishing and the maintainer's settings checklist.

Releases use one reviewed Release Please PR and a shared Cargo/Python version.
Squash PRs with Conventional Commit titles (`feat:`, `fix:`, or `feat!:` for a
breaking change). Before 1.0, breaking changes bump minor. Merging the release PR
creates the public GitHub release first; PyPI artifacts arrive only after the
separate cold-build publishing workflow succeeds. A GitHub release alone is not
proof of publication. See the [activation checklist](docs/development.md#release-activation-and-recovery).

Unmodified engineering reports are retained under `archive/engineering/`; its
`index.json` maps original paths to archived files and records their content hashes.
Their commands, identities and source paths describe historical revisions, not the
current checkout.

## References and acknowledgements

- McInnes, Healy and Melville, [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) (2018).
- Campello, Moulavi and Sander, [Density-Based Clustering Based on Hierarchical Density Estimates](https://doi.org/10.1007/978-3-642-37456-2_14) (2013).
- The external comparison reference is [fast_hdbscan](https://github.com/TutteInstitute/fast_hdbscan), pinned to `fast-hdbscan==0.3.2`; it is distinct from cuscan.
- Burn and CubeCL by Tracel AI provide the tensor and compute frameworks used by this project.
- Historical predecessor: Eugene Hauptmann, [fast-umap: UMAP in Rust](https://github.com/eugenehp/fast-umap) (v1.6.0).

## License

The project is declared MIT; see [LICENSE](LICENSE) for the retained legal notice.
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES) contains scoped upstream notices and
provenance limitations. Upstream provenance and bundled-native notices require
review separately from the project's maintainer identity; see
[migration_provenance.json](migration_provenance.json) for unresolved questions.
