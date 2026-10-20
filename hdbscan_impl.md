# HDBSCAN Implementation Plan (high level)

Port HDBSCAN from the Python package `fast_hdbscan` 0.3.2 (TutteInstitute/fast_hdbscan, commit
`43c573ba61043a6ad6f10b396656bfaa7aaeead8`) to Rust on Burn 0.22 / CubeCL 0.11, the same way
Standard UMAP was ported from umap-learn 0.5.12: follow upstream semantics, run compute-heavy
stages as CubeCL kernels on the CUDA and LLVM CPU runtimes, keep sequential tree work on the host,
and check parity against fixtures generated from the pinned reference package.

Algorithm stages, upstream code and porting detail: `hdbscan_algo_impl.md`.

## Goals and non-goals

### Goals (first release)

- Euclidean HDBSCAN with exact mutual-reachability MST (upstream `algorithm="boruvka"`,
  `metric="euclidean"`).
- Upstream cluster-extraction options: `min_cluster_size`, `min_samples`,
  `cluster_selection_method` (`eom`/`leaf`), `allow_single_cluster`, `max_cluster_size`,
  `cluster_selection_epsilon`, `cluster_selection_persistence`.
- Outputs: `labels`, `probabilities`, single-linkage tree, condensed tree, MST, neighbors, core
  distances, and `dbscan_clustering(epsilon)`.
- Non-finite rows behave as upstream: they are excluded from the fit and returned as noise
  (label `-1`, probability `0`) and remapped into the trees.
- Backends: `BackendChoice::{Auto, Cpu, Cuda}` with the strict semantics UMAP already uses
  (explicit choice never falls back; Auto uses CPU only when discovery confirms CUDA is
  unavailable).
- Python estimator usable as `HDBSCAN(...).fit_predict(UMAP(...).fit_transform(X))`.

### Deferred (separate plans, in this order)

1. `algorithm="kruskal"` with `knn_k` (approximate MST on a kNN graph, `+inf` bridges); upstream
   obtains neighbors through exact KD-tree `parallel_tree_query`, not our ANN kernel.
2. `sample_weights`. Upstream's weighted Boruvka path mixes squared and unsquared distances:
   `parallel_tree_query` is called without `output_rdist=True` (`boruvka.py:567`), but the
   squared Boruvka search still uses it (`boruvka.py:203-207`) and the final `sqrt`
   (`boruvka.py:646-647`) runs on top. Settle whether to replicate that behavior before porting
   it.
3. `metric="precomputed"` (sparse graph input), cannot-link constraints, semi-supervised BC.
4. `PLSCAN`, `BranchDetector` / `find_branch_sub_clusters`, layer clusters.
5. Non-Euclidean metrics. Upstream delegates these to pynndescent. Our neighbor kernels are
   Euclidean only.

Known deliberate divergence: for `D > 30` with no weights, upstream switches to an *approximate*
pynndescent kNN graph plus Kruskal (`hdbscan.py:447-463`) before considering `algorithm`.
Therefore even `algorithm="kruskal", knn_k=None` does not select the exact public path. Our first
release always computes the exact MST. For `D > 30`, the fixture generator must call pinned
upstream `parallel_boruvka(..., reproducible=True)` directly, then pass that MST through upstream's
tree/label/probability functions.

## Crate structure

The existing workspace is `fast-umap-python → fast-umap → fast-umap-core`
(`repo_reorg.md`). Algorithm-neutral infrastructure currently lives inside the UMAP facade:

- Backend selection: `crates/fast-umap/src/standard_umap/backend_choice.rs`.
- CUDA discovery: `crates/fast-umap/src/cuda_backend.rs`.
- Device resolution: `selected_device` and `resolve_backend_choice` in
  `standard_umap/estimator.rs:1164-1275`.
- Launch limits: `legal_cube_dim` in `kernels/mod.rs:9-33`.
- Exact and approximate Euclidean neighbor kernels: the `StandardUmapOps::exact_neighbors` and
  `approximate_neighbors` methods, implemented in `kernels/standard_umap/{mod.rs,ann*.rs}` and
  orchestrated by `standard_umap/neighbors.rs`.

HDBSCAN needs backend plumbing and exact neighbors only. It does not use ANN, UMAP's fuzzy graph,
layout, spectral init (`faer`) or model state.

### Options considered

| Option | Verdict |
|---|---|
| A. New `fast-hdbscan-core` + `fast-hdbscan` crates that depend on `fast-umap` for infrastructure | Rejected: the clustering crate would depend on a UMAP crate and pull in spectral/`faer` |
| B. `hdbscan` modules inside `fast-umap-core` / `fast-umap` | Rejected: mixes two independent algorithms, couples their features and build times, and makes `StandardUmapOps` grow unrelated ops |
| C. Rename `fast-umap` into a generic crate holding both algorithms | Rejected: churns every UMAP path for no consumer benefit |
| **A + shared common crate** | **Chosen**: extract only algorithm-neutral infrastructure into `fast-cubecl-common`; each algorithm gets a core crate and a facade crate |

### Target layout

```text
crates/
  fast-cubecl-common/          # NEW: backend choice, CUDA discovery, device resolution,
    src/                       #      launch limits, NeighborOps (+ its CubeBackend impl),
      lib.rs                   #      exact Euclidean neighbor kernels
      backend_choice.rs        # moved from fast-umap/src/standard_umap/backend_choice.rs
      cuda_backend.rs          # moved from fast-umap/src/cuda_backend.rs
      device.rs                # selected_device/resolve_backend_choice from estimator.rs
      error.rs                 # BackendError (mapped by each facade)
      backend.rs               # trait NeighborOps (#[backend_extension])
      neighbors.rs             # exact_neighbors orchestration, NeighborList
      kernels/{mod.rs, neighbors/…} # exact neighbors only; no ANN
  fast-hdbscan-core/           # NEW: backend-free host code
    src/
      lib.rs  config.rs  error.rs
      kdtree.rs                # host KD-tree build (implicit heap layout)
      dsu.rs                   # union-find (rank + path compression)
      mst.rs                   # Boruvka orchestration helpers, edge ordering, bridging
      linkage.rs               # MST → single-linkage tree
      condense.rs              # condensed tree, simplify_hierarchy
      select.rs                # EOM / leaf / epsilon / max size
      labels.rs                # labels, membership strengths, dbscan cut
      finite.rs                # non-finite row filtering and tree remapping
    tests/
  fast-hdbscan/                # NEW: facade
    src/
      lib.rs                   # re-exports core + common types at stable paths
      backend.rs               # trait HdbscanOps (#[backend_extension])
      kernels/{mod.rs, boruvka.rs}   # impl HdbscanOps for CubeBackend (orphan rule)
      estimator.rs             # HDBSCAN / FittedHDBSCAN
    tests/  benches/  examples/
  fast-umap-core/              # keeps ANN helpers
  fast-umap/                   # uses fast-cubecl-common; keeps ANN and UMAP-only ops
  fast-umap-python/
    src/{lib.rs, umap.rs, hdbscan.rs}   # one extension module, split source files
python/fast_umap/{__init__.py, standard_umap.py, hdbscan.py}
python/tests/test_hdbscan*.py
```

### Dependency graph

```text
fast-umap-python ─┬─> fast-umap ────┬─> fast-umap-core
                  │                 └─> fast-cubecl-common ─> burn / cubecl / cudarc
                  └─> fast-hdbscan ─┬─> fast-hdbscan-core
                                    └─> fast-cubecl-common
```

Both core crates stay backend-free (no Burn/CubeCL). `fast-hdbscan-core` should not need `faer`
or `rand`.

### Trait and orphan-rule boundaries

- A `#[backend_extension]` trait and its only `impl … for CubeBackend` must be in the same
  crate. Hence:
  - `NeighborOps` contains exact neighbors, squared-distance block, and backend plumbing only;
    its trait and CubeBackend impl live in `fast-cubecl-common`. ANN kernels/helpers and
    `approximate_neighbors` remain in `fast-umap` (HDBSCAN v1 does not use ANN).
    `StandardUmapOps` loses only the exact-neighbor and squared-distance methods; ANN and other
    UMAP-specific methods remain in place.
  - `HdbscanOps` (Boruvka candidate search, optional on-device component bookkeeping) is a new
    trait in `fast-hdbscan` with its impl in `fast-hdbscan/src/kernels`. Do not add these ops to
    `StandardUmapOps`.
- `legal_cube_dim` becomes `pub` in common so both facades use the same launch limits.

### Features

`cfg(feature = …)` is evaluated per package, so every crate that checks a backend `cfg` declares
the feature (`repo_reorg.md` Features section).

| Crate | default | Features |
|---|---|---|
| `fast-cubecl-common` | none | `cpu`, `cuda` (same Burn/CubeCL/cudarc enables as today's `fast-umap`), `tracing` |
| `fast-hdbscan-core` | none | `tracing` |
| `fast-hdbscan` | `cuda,cpu` | `cpu`/`cuda` forward to `fast-cubecl-common`, `tracing`; same `compile_error!` guard as UMAP |
| `fast-umap` | `cuda,cpu` | as today, now also forwarding `cpu`/`cuda` to `fast-cubecl-common` |
| `fast-umap-python` | `cuda,cpu` | `cpu`/`cuda`/`tracing` forwarded to both facades |

All internal edges use `default-features = false`. Burn/CubeCL versions stay in
`[workspace.dependencies]`.

### Errors

- `fast-cubecl-common::BackendError { Unavailable(String), Overflow(&'static str), … }`.
- `fast-hdbscan-core::HdbscanError { InvalidConfig, InvalidInput, BackendUnavailable, Numerical,
  Overflow }` remains backend-free; no backend-error dependency or conversion.
- Facades explicitly convert backend failures using local `map_err`: `fast-umap` to its existing
  `UmapError` variants/messages and `fast-hdbscan` to `HdbscanError` (unavailable →
  `BackendUnavailable`, allocation/size overflow → `Overflow`). No `From<BackendError>` impl.

### Public Rust API sketch

```rust
pub enum ClusterSelectionMethod { Eom, Leaf }

pub struct HdbscanConfig {
    pub min_cluster_size: usize,          // default 5 (upstream class default)
    pub min_samples: Option<usize>,       // None → min_cluster_size
    pub cluster_selection_method: ClusterSelectionMethod,
    pub allow_single_cluster: bool,
    pub max_cluster_size: Option<usize>,  // None ≙ np.inf
    pub cluster_selection_epsilon: f64,
    pub cluster_selection_persistence: f64,
}

pub struct HDBSCAN { config: HdbscanConfig, backend: BackendChoice }
impl HDBSCAN {
    pub fn new(config: HdbscanConfig, backend: BackendChoice) -> Result<Self>;
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<FittedHDBSCAN>;
}

pub struct FittedHDBSCAN {
    labels: Vec<i32>, probabilities: Vec<f32>,
    single_linkage_tree: Vec<LinkageRow>,          // (left, right, distance: f64, size)
    condensed_tree: CondensedTree,                 // SoA: parent, child, lambda_val, child_size
    minimum_spanning_tree: Vec<MstEdge>,           // (src, dst, weight: f64), sorted
    neighbors: Vec<Vec<i32>>, core_distances: Vec<f32>,
    backend: BackendChoice,
}
impl FittedHDBSCAN {
    pub fn labels(&self) -> &[i32];
    pub fn probabilities(&self) -> &[f32];
    pub fn dbscan_clustering(&self, epsilon: f64) -> Vec<i32>;
    // tree/MST/neighbor/core accessors
}
```

`BackendChoice` stays outside the backend-free config, matching UMAP.

### Python surface

- One native module `fast_umap._fast_umap` (maturin builds exactly one module:
  `pyproject.toml [tool.maturin] module-name`) gains `fit_hdbscan(...)` and a frozen
  `FittedHDBSCAN` class in `crates/fast-umap-python/src/hdbscan.rs`.
- `python/fast_umap/hdbscan.py`: `HDBSCAN` estimator mirroring upstream parameters in scope,
  plus `backend="auto"` and `device_index=0`. Attributes: `labels_`, `probabilities_`,
  `condensed_tree_` (structured array with upstream dtype), `single_linkage_tree_`,
  `minimum_spanning_tree_` (raw arrays; upstream wraps them in `hdbscan.plots` classes only if
  `hdbscan` is installed, and we return the arrays), and `dbscan_clustering(eps)`. Unsupported
  upstream parameters raise `NotImplementedError` and are never silently ignored.
- `from fast_umap import UMAP, HDBSCAN`. The distribution keeps the name `fast-umap`. Renaming the
  package is a separate decision.
- Python exceptions: `InvalidConfig`/`InvalidInput` → `ValueError`; `BackendUnavailable` →
  `RuntimeError`; `Overflow` → `OverflowError`; `Numerical` → `RuntimeError`. Backend choice
  validation preserves UMAP's current invalid-choice/unavailable mappings.

## Prerequisite refactors (before any HDBSCAN code)

Each step preserves UMAP's observable outputs and fixture results.

1. **Create `fast-cubecl-common`** for backend choice/device resolution, CUDA discovery,
   `legal_cube_dim`, and exact-neighbor backend plumbing only. Common's
   `BackendChoice::validate` returns `Result<(), BackendError>`; each facade provides its local
   `validate_backend_choice(choice) -> Result<(), UmapError>` or
   `validate_backend_choice(choice) -> Result<(), HdbscanError>` wrapper and maps the common error
   explicitly, preserving facade error variants and Python mappings (including UMAP's existing
   invalid-choice/unavailable behavior). Do not claim signature stability for the moved method or
   add a compatibility layer. Move `resolve_backend_choice` tests with the moved function, and
   retain caller tests for both facade wrappers.
2. **Extract exact neighbors only.** Split `exact_neighbors` and
   `squared_euclidean_distance_block` from `StandardUmapOps` into common's `NeighborOps`; keep
   `approximate_neighbors`, `ann_sampled.rs`, `candidate_width`, and ANN orchestration in
   `fast-umap`. HDBSCAN uses the squared mode; every UMAP caller passes `squared=false` and
   preserves its `f32::MAX` slot initialization/overflow behavior (separate entry point or
   explicit mode). Update every `<Dispatch as StandardUmapOps>` caller, fusion metadata, examples,
   and tests for the trait/signature change. Keep `standard_umap/neighbors.rs` in UMAP.
3. Update feature forwarding, scripts and guidance for the new crate. Do not edit pyproject cache
   keys: existing keys already glob `crates/*/Cargo.toml` and `crates/**/*.rs`.

Gate: UMAP observable outputs and fixtures remain unchanged on supported inputs (CPU/Python lanes
and `test_features.sh`); run the CPU benchmark before/after and investigate any regression.

## Delivery phases

| Phase | Content | Gate |
|---|---|---|
| 0 | Prerequisite refactors above | UMAP outputs and fixtures unchanged |
| 0.5 | Add pinned `fast_hdbscan==0.3.2` dev dependency, fixture schema, and generator at `crates/fast-hdbscan/tests/fixtures`; store inputs, f64 MST weights and upstream structured condensed-tree records | Fixtures generated; D>30 uses pinned `parallel_boruvka(reproducible=True)` exact internal entrypoint |
| 1 | `fast-hdbscan-core` host pipeline: DSU, linkage, condense, select, labels, probabilities, dbscan cut, finite remap. Driven by MSTs taken from upstream fixtures | Core tests reproduce upstream labels/probabilities/trees exactly from upstream's MST |
| 2 | Reference MST on host (exact Boruvka over brute-force candidates) as a test oracle | MST weight multiset equals upstream's |
| 3 | `fast-hdbscan` facade: core distances via shared `exact_neighbors`; brute-force tiled Boruvka candidate kernel (`HdbscanOps`); estimator; CPU + CUDA | End-to-end parity vs fixtures on CPU; CUDA uses parity tolerances and ARI rules below (near ties may change labels) |
| 4 | KD-tree Boruvka kernel (host-built tree, device traversal) as the low-dimensional fast path, plus a kNN-initialized first round | Same results as phase 3; benchmarks vs upstream |
| 5 | Python binding + estimator + pytest parity (including UMAP→HDBSCAN pipeline) | pytest parity suite green |
| 6+ | Deferred items, each with its own plan | — |

## Testing strategy

- **Fixtures** (added in phase 0.5): `crates/fast-hdbscan/tests/fixtures/generate_fast_hdbscan.py`,
  pinned to `fast_hdbscan==0.3.2` with `reproducible=True`. Store inputs as NPZ, metadata as JSON,
  MST weights as f64, and condensed trees in upstream structured dtype. Record package version,
  source commit, entrypoint, reproducible flag, and parameters in fixture metadata. For every
  fixture, invoke pinned `parallel_boruvka(..., reproducible=True)` directly, then upstream
  tree/label/probability routines; never rely on its public fit path as the exact oracle. Core
  integration tests read the shared files via `../fast-hdbscan/tests/fixtures` relative to the
  `fast-hdbscan-core` package root; declare the existing workspace `serde_json` and `zip` crates
  as core dev-dependencies for JSON metadata and NPZ ZIP-archive access. Datasets: blobs (2D, 10D, 50D), moons,
  varying-density, duplicates/ties, a single cluster, all noise, n < min_samples, n = 1 and 2,
  and rows with NaN/inf. For each one, store inputs, sorted MST, linkage tree, condensed tree,
  labels and probabilities for a parameter sweep (`min_samples`, `min_cluster_size`,
  `leaf`/`eom`, `allow_single_cluster`, epsilon, persistence, `max_cluster_size`).
- **Comparisons**:
  - Phase 1 (host stages on upstream's MST): exact equality.
  - End to end: MST weights to a relative tolerance of 1e-5. Labels equal up to permutation.
    ARI = 1 on datasets whose relevant candidate, merge, and cluster-selection comparisons are
    separated from ties by more than the numerical tolerance; MST-weight gaps alone do not prove
    that every downstream decision is unambiguous. Probabilities to 1e-5. Datasets with exact or
    near ties in those decisions are checked with ARI and documented, because tie-breaking may
    legitimately differ.
- **CPU/CUDA parity** uses the same MST/probability tolerances and ARI criteria as upstream parity;
  allow label differences only where near-tie floating-point ordering warrants them. CUDA lanes use
  `FAST_UMAP_REQUIRE_CUDA=1`.
- **Invalid input**: empty, zero features, `min_samples`/`min_cluster_size` = 0,
  negative/non-finite epsilon or persistence, ragged rows, too many rows for i32, unsupported
  parameters (Python).
- **Benchmarks**: Criterion `hdbscan_cpu` / `hdbscan_cuda` plus a Python timing comparison
  against `fast_hdbscan`. No performance claims without these.

## Risks

- **Exactness vs speed.** The brute-force candidate kernel costs O(N²·D) per Boruvka round. That
  is fine for the oracle and moderate N, but phase 4 is needed before large inputs are usable.
  KD-tree traversal in CubeCL needs an explicit stack, and divergence on CUDA may limit speedup.
  Benchmark before claiming anything.
- **Tie-breaking.** Upstream's parallel candidate selection is racy outside reproducible mode.
  Ours is deterministic (`(distance, index)` order), so trees can differ on exact ties. Parity
  tests must handle this explicitly.
- **Precision.** Upstream searches in squared f32 and applies `sqrt` in f64 at the end. Our
  `exact_neighbors` returns Euclidean f32. Keep Boruvka in squared space to match.
- **Refactor blast radius.** Moving exact-neighbor ops and backend plumbing requires caller,
  fusion-metadata, example and test updates. Keep UMAP's old output path (including overflow slot
  semantics) unchanged and gate on observable outputs and fixtures.
- **Packaging naming.** HDBSCAN shipped inside `fast_umap` is pragmatic but awkward. Renaming
  the distribution later is a breaking change.
