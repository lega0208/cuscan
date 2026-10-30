# Changelog

All notable changes to this project will be documented in this file.

Release Please generates version sections from Conventional Commits. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), with breaking changes
bumping the minor version before 1.0.

---

### Pre-automation project baseline

- Unified Python `cuscan` package with non-parametric UMAP and HDBSCAN, including transform, clustering and approximate prediction, and optional HDF5 fitted-model serialization.
- Rust `umap` and `hdbscan` facades with CPU (CubeCL LLVM) and CUDA backends, selected at runtime.
