# Parallel Borůvka: M0/M1 implementation notes

## Decisions and findings
- M0 passes on LLVM CPU and required CUDA (RTX 3080), using CubeCL 0.11.1 `Atomic<i32>::load`/`fetch_min` and device-side f32↔i32 reinterpretation. No CAS fallback is needed. Tests cover both zeros, the smallest subnormal, 1, f32::MAX, infinity, shared-address contention across cubes, and reset bounds.
- Important deviation: `value + 0.0` did **not** canonicalize -0 on the CPU JIT (observed bits `0x80000000`). Canonicalize that exact integer bit pattern to zero instead; both runtimes pass. Future M5 must retain this rule rather than trusting fast-math arithmetic.
- M1 adds backend-free `knn_kruskal` with f64 cores/MRD after widening squared f32 distances, ordinal-stable tie decisions, final `sort_mst`, and minimum-component-ID infinity bridges. `KnnGraph.rows` is explicit so empty/singleton/zero-width graphs are representable.
- Validation rejects malformed IDs/rows, decreasing valid distances, and NaN/negative distances (including padding). Padding needs no particular sentinel distance; missing ranked neighbors give infinite cores, except min_samples≤1 gives zero cores. Size arithmetic and allocation byte bounds are checked; no new error variant or recoverable-OOM promise.
- Fixed `merge_components` first-component-appearance slot order. Explicit candidate presence distinguishes pruned placeholders from real infinity candidates; later strict-less tie behavior and no-progress errors are preserved.
- The test-support oracle constructs exact neighbors, Kruskal, DSU and bridges independently of production graph helpers. Tests compare exact graph-route edges/core bits, complete-graph weight multisets, ordinal ties, invalid inputs, disconnected forests, padding, overflow, and sample-count boundaries.

## Verification
All passed:
- `cargo test -p fast-hdbscan-core --features test-support` — 57 tests.
- `cargo test -p fast-hdbscan-core` — backend-free/default-feature coverage.
- `cargo test -p fast-cubecl-common --test atomic_bounds m0_atomic_cpu_` and `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common --test atomic_bounds m0_atomic_cuda_` — one real runtime test each, not skipped.
- `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common`.
- `cargo test -p fast-hdbscan --test test_kdtree_search kd_traversals_match_exact_host_oracles_cpu` — 1 test; `cargo test -p fast-hdbscan --test test_estimator _cpu` — 4 tests.
- `cargo fmt --all -- --check`; Clippy with `-D warnings` for core/all-targets/test-support and common's atomic_bounds target.
- `cargo tree -p fast-hdbscan-core --no-default-features -e normal` — no normal dependencies.

Current manifests enable CPU/CUDA unconditionally; old `--features cpu/cuda` guidance is stale. Cargo reports existing future-incompatibility warnings in burn-cubecl/burn-cubecl-fusion 0.22.1; these were not suppressed.

## Deferred / untouched
M2–M8, estimator routing/configuration, ANN movement, device winner reduction, and performance tuning remain out of scope. M0 proves primitives only, not future KD-bound pruning correctness or performance. Unrelated `comparison/hdbscan_stages.html` changes are preserved. The consensus ralplan is the refined plan; no further planning rewrite was performed.
