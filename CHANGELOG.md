# Changelog

## 0.1.0 (2026-10-30)


### Features

* add CPU backend support with umap-rs integration ([52ca79b](https://github.com/lega0208/cuscan/commit/52ca79b8e1d2fcf21c74a4d894108428b9dad03d))
* **ci:** automate reviewed releases ([#5](https://github.com/lega0208/cuscan/issues/5)) ([fdf35d8](https://github.com/lega0208/cuscan/commit/fdf35d811b2d1f822e4498d59e7f2bef7b0fb5fa))
* upgraded burn and cubecl warn: versions. didn't do full tests yet! ([dad731e](https://github.com/lega0208/cuscan/commit/dad731e27f8860b9f444a1f300d8746a937d7867))


### Bug Fixes

* **ci:** more issues ([#8](https://github.com/lega0208/cuscan/issues/8)) ([139bee6](https://github.com/lega0208/cuscan/commit/139bee69c223536fc412f5f079376c0faece98c6))
* **ci:** release please workflow error ([#6](https://github.com/lega0208/cuscan/issues/6)) ([803fa70](https://github.com/lega0208/cuscan/commit/803fa70900f9ee8478ec439215f815f9f0ed7d04))

## Changelog

All notable changes to this project will be documented in this file.

Release Please generates version sections from Conventional Commits. Versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html), with breaking changes
bumping the minor version before 1.0.

---

### Pre-automation project baseline

- Unified Python `cuscan` package with non-parametric UMAP and HDBSCAN, including transform, clustering and approximate prediction, and optional HDF5 fitted-model serialization.
- Rust `umap` and `hdbscan` facades with CPU (CubeCL LLVM) and CUDA backends, selected at runtime.
