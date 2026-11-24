# Changelog

## [0.1.3](https://github.com/lega0208/cuscan/compare/v0.1.2...v0.1.3) (2026-11-24)


### Features

* add validation script ([6bde2d0](https://github.com/lega0208/cuscan/commit/6bde2d01979ba48b769d7da12be6ca8c47c9a349))
* CUDA ANN pairwise kernel rewrite ([2e7b8a6](https://github.com/lega0208/cuscan/commit/2e7b8a6db305a644d40952dda097c61a54d79260))
* Improve ANN performance ([2d7feae](https://github.com/lega0208/cuscan/commit/2d7feae0abc3d378b4ae3d2391596d5d046ef737))
* port RINND to cubecl kernels for CPU ANN (close enough for now) ([dead1de](https://github.com/lega0208/cuscan/commit/dead1de713c10dbf6a507a7b51917d9721491ff6))
* port UMAP 0.6.0 changes ([b349ff7](https://github.com/lega0208/cuscan/commit/b349ff72113beec7cc9fd1e5dfb995477d51d2e9))


### Bug Fixes

* **ci:** Change python version from 3.14 to 3.13 because of `rinnd` dependency; update cargo deny ([ee67c76](https://github.com/lega0208/cuscan/commit/ee67c76de267fafd04c1c32fc068881f2d1ce44a))
* **ci:** windows tests failing because of `fcntl` package ([ea1fb6e](https://github.com/lega0208/cuscan/commit/ea1fb6efe3af3e60707067be8fecbdaedd49db53))
* **ci:** windows tests failing because of `resource` package ([be19074](https://github.com/lega0208/cuscan/commit/be1907490f3c07a85e5aaf70e910999dd5071b4f))
* clippy lint ([15d1bff](https://github.com/lega0208/cuscan/commit/15d1bff32346ee7bbff708b85d7355ae94559c66))
* missing cuda device throwing error and other CI issues ([200bcb8](https://github.com/lega0208/cuscan/commit/200bcb8632bac22c2ba7308ebf9983df26e2d6ab))
* **perf:** remove HDBSCAN kernel that turned out to only be good for gaussian noise ([438ef99](https://github.com/lega0208/cuscan/commit/438ef99f2e91209628be1e7e6510daa1b9c2ca03))
* update pynndescent fixture provenance versions ([4386c92](https://github.com/lega0208/cuscan/commit/4386c926a38c8004ca7ef958eed6be5db4a5fc85))


### Performance

* improve HDBSCAN kdtree build performance ([521ec09](https://github.com/lega0208/cuscan/commit/521ec096e4cd85c278b6ecc2f361c18af2d0b154))

## [0.1.2](https://github.com/lega0208/cuscan/compare/v0.1.1...v0.1.2) (2026-11-13)


### Features

* **ann:** Add memory accounting/budgeting ([e77262d](https://github.com/lega0208/cuscan/commit/e77262de50b77aaf93d1ca03c74d0f98423a4f47))
* **ann:** Available memory-based tile-counts to avoid over-allocation ([e77262d](https://github.com/lega0208/cuscan/commit/e77262de50b77aaf93d1ca03c74d0f98423a4f47))
* **ann:** In-place merge and lifetimes; buffer frees ([f2d7ff2](https://github.com/lega0208/cuscan/commit/f2d7ff2193d8bd62ed72b4d9690c14dc2229ae61))
* **ann:** Optimize forest construction by using breadth-first projection when input size is above a certain threshold ([e27c9f0](https://github.com/lega0208/cuscan/commit/e27c9f024665d144f5d0606dd7e68bb082175d24))
* **ann:** Proper user-facing errors when over-allocating or not enough memory available ([e77262d](https://github.com/lega0208/cuscan/commit/e77262de50b77aaf93d1ca03c74d0f98423a4f47))
* **perf:** ANN endpoint filtering + sampled pool ([0e0956b](https://github.com/lega0208/cuscan/commit/0e0956b8fd1680959d27456f031a49740cc4f3e4))


### Bug Fixes

* **ann:** Make host memory utils cross-platform ([cca61f5](https://github.com/lega0208/cuscan/commit/cca61f57a0451a8afffbaa7dbf4e06d8e4dd78d2))
* clippy lint errors ([899fee6](https://github.com/lega0208/cuscan/commit/899fee6cd3c1e9768ae59cd56a17a3c08a5806f9))
* python test race condition ([038d918](https://github.com/lega0208/cuscan/commit/038d918305f9c77ca5dec3ab43169b296f18ef33))
* usize overflow in ANN ([0c9138f](https://github.com/lega0208/cuscan/commit/0c9138f834d58bda4780bac12e070c33585acebf))

## [0.1.1](https://github.com/lega0208/cuscan/compare/v0.1.0...v0.1.1) (2026-11-03)


### Features

* umap layout optimizations ([#10](https://github.com/lega0208/cuscan/issues/10)) ([69c4272](https://github.com/lega0208/cuscan/commit/69c4272625877d1d49ca3f903b86ede2b30a766e))


### Bug Fixes

* broken release ([#13](https://github.com/lega0208/cuscan/issues/13)) ([519132e](https://github.com/lega0208/cuscan/commit/519132e93dbe6033cf679e710461a374c8d19015))
* flaky python test ([#12](https://github.com/lega0208/cuscan/issues/12)) ([9ecc447](https://github.com/lega0208/cuscan/commit/9ecc447658173600371322472c7ff5f36b9ff924))

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
