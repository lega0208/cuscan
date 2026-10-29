# cuscan

cuscan 0.1.0 provides non-parametric UMAP dimensionality reduction and HDBSCAN clustering in Rust and Python. UMAP follows umap-learn 0.5.12 semantics. The implementation uses [Burn](https://github.com/tracel-ai/burn) and [CubeCL](https://github.com/tracel-ai/cubecl) for CPU and CUDA execution.

Repository: <https://github.com/lega0208/cuscan>

## Local build

The documented local target is Linux x86-64, including WSL2. Python 3.10+, Rust nightly, uv and Maturin are required. `rust-toolchain.toml` selects the floating nightly channel, and the x86-64 Linux Cargo configuration requires `/usr/sbin/clang` and mold and enables nightly flags. From the repository root:

```sh
uv sync
uv run maturin develop --release
```

Use `uv run --no-sync` for subsequent scripts or tests so syncing does not replace the locally built extension. The CubeCL LLVM bundle/build/JIT path can require network access, cache space and LLVM support. CPU and CUDA are both compiled; CUDA execution requires a working NVIDIA runtime and device. Compilation alone does not initialize CUDA.

### Wheel validation

The lowest validated Linux x86-64 wheel baseline is **manylinux 2.34**, not a
proven universal minimum. A Docker manylinux 2.34 build of the final sdist used
official Rust nightly 1.99.0 (1a98b1e13, 2026-08-07) mounted read-only at
`/opt/rust`; `auditwheel show` confirmed the 2.34 tag. The resulting local ABI3
wheel passed CPU container smoke checks on Python 3.10 and 3.14 for UMAP and
HDBSCAN fit/predict, new persistence and rejection of old files, and a host
Python 3.14 CUDA smoke check for the same wheel. This does not establish that
every Python version from 3.10 onward was tested.
The image digest was `sha256:834a2a31de5eabacd4411417b3bbc9fcb6ff9f959972711ab3b73355e7b0928d`;
the local wheel was `/tmp/cuscan-manylinux-dist/cuscan-0.1.0-cp310-abi3-manylinux_2_34_x86_64.whl`.

The manylinux 2.28 image failed to build because `tracel-llvm-23.1.0-4`'s
`llvm-config` requires GLIBC 2.32, 2.33 and 2.34. A native glibc 2.44 wheel was
rejected by PyPI; use the validated manylinux wheel for release instead. No PyPI
upload or GitHub rename has been completed, and remote and provenance blockers
remain; registry installation is not established.
The failed 2.28 image digest was
`sha256:e0adf18521ad35192df733a6c62d35df7801957d24e0470238aacecfab2188a3`.

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

### Fitted-model serialization

Install the optional HDF5 dependencies locally with `uv sync --extra serialization`, then rebuild with `uv run maturin develop --release`. The extra supplies `h5py` and `hdf5plugin`; ordinary fitting does not require them.

```python
mapper.save("umap.h5")
restored = UMAP.load("umap.h5", backend="cpu")
projected = restored.transform(X[:5])

predictor = clusterer.to_predictor()
predictor.save("predictor.h5")
```

`UMAP` saves fitted state; HDBSCAN serialization is for `HDBSCANPredictor`, not the fitted estimator or its fit outputs. Load a predictor with `HDBSCANPredictor.load(path, backend="auto", device_index=0)`. Backend and device are chosen at load time. `save` refuses to overwrite by default; `overwrite=True` explicitly replaces a file. Publication uses a same-directory temporary file and atomic commit; a post-commit directory-sync failure reports uncertain durability.

Files use a versioned HDF5 schema and lossless Blosc2 Zstandard compression, not pickle. External HDF5 readers need the Blosc2 filter plugin (ID 32026). Files contain training data or geometry and derived model state; protect them like the original data. Schema validation and compression do not authenticate files. Rust does not provide an HDF5 persistence API. Serialized numeric state does not guarantee identical execution across hardware, software versions or backends.

The `cuscan_fitted_model` format rejects older model files; retrain and save them in the new format. There are no compatibility aliases for prior imports, crates or environment variables.

## Rust

The workspace exposes the `umap` and `hdbscan` facades. Build and test them locally:

```sh
cargo build --release -p umap -p hdbscan
cargo test -p umap
cargo test -p hdbscan
```

See [development and analysis](docs/development.md) for Rust API examples, detailed backend behavior, profiling, comparisons, serialization benchmarks and testing guidance. Historical benchmark measurements there are not cuscan rebenchmarks.

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
