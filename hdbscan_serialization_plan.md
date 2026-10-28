# HDBSCAN fitted-model serialization — implementation plan

Status: draft for review. Scope: add `save`/`load` for fitted HDBSCAN models, following the
existing Standard UMAP persistence design (Rust backend-free state export/import, Python
versioned HDF5 file via `h5py` + `hdf5plugin` Blosc2/zstd, strict schema validation, atomic
publication). Section 11 lists every decision taken where the request or existing documents
were ambiguous.

## 0. Prerequisites and baseline

- The uncommitted `approximate_predict` implementation in the working tree
  (`crates/fast-hdbscan/src/prediction.rs`, `crates/fast-hdbscan-core/src/prediction.rs`,
  estimator/bridge/Python changes) is the base of this work. It defines `FittedState`,
  `PredictionRuntime`, `CoreStatus` and the documented "future persistence boundary"
  (`prediction.rs` module docs; `approx_predict_plan.md` §6 "Serialization-ready state
  boundary"). That work must be committed (by the user) before serialization starts; this
  plan does not modify or re-review it beyond the seams named below.
- The UMAP reference implementation is:
  - Rust: `fast_umap_core::ModelState` (`crates/fast-umap-core/src/state.rs`),
    `FittedUMAP::export_state` / `FittedUMAP::from_state`
    (`crates/fast-umap/src/standard_umap/estimator.rs`), tests in
    `crates/fast-umap/tests/test_standard_serialization.rs`.
  - Bridge: `FittedUMAP.export_state()` and `_fast_umap.load(arrays, metadata, backend,
    device_index)` (`crates/fast-umap-python/src/lib.rs`).
  - Python: `python/fast_umap/_serialization_v1.py` (`save_model`/`load_model`,
    `DurabilityUncertainError`), `UMAP.save`/`UMAP.load` (`standard_umap.py`), tests in
    `python/tests/test_serialization.py`, optional extra `serialization = ["h5py>=3.11",
    "hdf5plugin>=5.0"]` in `pyproject.toml`, docs in README "Fitted model serialization".

## 1. Goal and non-goals

Goal: `HDBSCAN.save(path, overwrite=False)` and `HDBSCAN.load(path, *, backend="auto",
device_index=0)` (Python), backed by `FittedHDBSCAN::export_state()` and
`FittedHDBSCAN::from_state(state, backend)` (Rust). A loaded model is **complete**: every
fitted attribute (`labels_`, `probabilities_`, `condensed_tree_`, `single_linkage_tree_`,
`minimum_spanning_tree_`, `neighbors_`, `core_distances_`, `finite_index_`,
`n_features_in_`), `dbscan_clustering`, and `approximate_predict` behave as on the original
model. Predictions after load are bit-identical when the runtime backend matches the one
used for the comparison, and tolerance-equal across backends.

Non-goals:
- No Rust-native file format or Rust file I/O API (UMAP has none either; Rust exposes the
  storage-independent state only).
- No pickle / `__getstate__` support, no cross-version migration (schema version 1 only;
  unknown versions are rejected).
- No change to fitting algorithms, KD routing, or prediction math.
- No persistence of device tensors, mutexes, or runtime caches as such.

## 2. What is persisted (state inventory)

Source of truth is the canonical `FittedState` (never the detached public mutable fields of
`FittedHDBSCAN`, which users may have mutated). Every field and its fate:

| `FittedState` field | Persisted as | Notes |
|---|---|---|
| `config: HdbscanConfig` | `config` attrs | All 10 fields; optionals as `*_is_none` + value (UMAP pattern). |
| `training_backend` | `provenance` attrs `training_backend` (`"cpu"`/`"cuda"`), `training_device_index` | Provenance only; never drives load execution. |
| `labels` | dataset `<i4`, `(raw_n,)` | |
| `probabilities` | dataset `<f4`, `(raw_n,)` | |
| `linkage` | 4 column datasets `linkage_left <u8`, `linkage_right <u8`, `linkage_distance <f8`, `linkage_size <u8` | Length `L` (may be 0). |
| `condensed` | 4 column datasets `condensed_parent <u8`, `condensed_child <u8`, `condensed_lambda <f4`, `condensed_child_size <f4` | Length `C` (may be 0). |
| `mst` | `mst_src <u4`, `mst_dst <u4`, `mst_weight <f8` | Length `n_finite-1` (0 for one finite row). |
| `neighbors` | dataset `<i4`, 1-D flat, length `n_finite*s` when `n_finite > 1`, **length 0 when `n_finite == 1`** (fit leaves it empty; bridge exposes `(1,0)`) | `s = resolved_min_samples`; `-1` padding preserved verbatim; reshaped by the bridge, never synthesized. |
| `core_distances` | dataset `<f4`, `(n_finite,)` | `+inf` is a legitimate fitted value (all rows when `s >= n_finite` on the complete route, padded kNN rows, singleton, f32 overflow); see 4.3 step 5. |
| `finite_index` | dataset `<u4`, `(n_finite,)` | |
| `n_features`, `raw_n` | `state` attrs `<u8` | `n_finite` is derived from `finite_index` and also stored for cross-check. |
| `search` (resolved) | `provenance` attr `resolved_search` (`"brute_force"`/`"kd_tree"`) | |
| `resolved_graph` | `provenance` attr `resolved_graph` (`"complete"`/`"knn_graph"`) | |
| `knn_diagnostics: Option<KnnDiagnostics>` | optional group `knn_diagnostics` (presence attr on `state`) | Present iff `resolved_graph == KnnGraph` and `n_finite > 1`. Datasets `neighbors <i4 (n_finite*k,)`, `distances_sq <f4 (n_finite*k,)`; attrs `source` (`"exact"`/`"approximate"`), `k <u8`, `bridges <u8`, `finite_weight_sum <f8`, `allocation_bytes <u8`. |
| `geometry` | dataset `<f4`, flat `n_finite*n_features` (row-major) | Compact finite training coordinates. |
| `min_samples` | not stored | Derived from config (`resolved_min_samples`). |
| `hierarchy: PredictionHierarchy` | not stored | Reconstructed on load with `PredictionHierarchy::new` (the existing documented contract). |
| runtime `CoreStatus` | `prediction` group: attr `cores_status` (`"ready"` / `"not_required_no_cluster_tree"`), attr `cores_rank <u8` (0 when not required), dataset `cores <f8 (n_finite,)` present iff ready | Save always prepares cores (section 4). |
| runtime KD tree | not stored | Rebuilt deterministically and eagerly on load (decision D2). |

## 3. File schema v1

Same container conventions as UMAP v1 (`_serialization_v1.py`):

Root attributes (exact set, nothing else allowed):
`format="fast_umap_fitted_model"`, `model_kind="hdbscan"`, `schema_version=<u4 1`,
`producer_version=<text, nonempty>`, `filter_id=<u4 32026`, `codec="zstd"`,
`shuffle="shuffle"`, `clevel=<u1 1`.

Root groups (exact set): `config`, `provenance`, `state`, `prediction`, and
`knn_diagnostics` iff `state.attrs["knn_diagnostics_present"]` is true.

- `config` attrs (exact set): `min_cluster_size <u8`, `min_samples_is_none |b1`,
  `min_samples_value <u8`, `cluster_selection_method` (`"eom"`/`"leaf"`),
  `allow_single_cluster |b1`, `max_cluster_size_is_none |b1`, `max_cluster_size_value <u8`,
  `cluster_selection_epsilon <f8`, `cluster_selection_persistence <f8`, `mst_graph`
  (`"auto"`/`"complete"`/`"knn_graph"`), `knn_k_is_none |b1`, `knn_k_value <u8`,
  `reproducible |b1`. Noncanonical optional payloads (`*_is_none` true with nonzero value)
  are rejected, as in UMAP.
- `provenance` attrs (exact set): `training_backend`, `training_device_index <u8` (0 for
  CPU), `resolved_search`, `resolved_graph`.
- `state` attrs (exact set): `raw_n <u8`, `n_finite <u8`, `n_features <u8`,
  `knn_diagnostics_present |b1`; datasets (exact set): every state dataset in section 2.
  All are always present; datasets whose valid length can be zero (`linkage_*`,
  `condensed_*`, `mst_*`) are written with length 0 (decision D5).
- `prediction` attrs (exact set): `cores_status`, `cores_rank`; dataset `cores` iff ready.
- `knn_diagnostics` attrs/datasets as in section 2.

All persisted datasets are 1-D (2-D shapes such as `neighbors`, `geometry`, diagnostics are
flattened row-major and reshaped by the bridge/Python from `n_finite`, `s`, `k`,
`n_features`); this avoids zero-extent trailing axes in HDF5 chunking. Every dataset:
exact dtype and rank 1, length cross-checked against `state` attrs (and `s`, `k`), chunk
rows `max(1, min(len, 32_768))`, Blosc2 filter with the recorded `clevel`,
shuffle and zstd verified via `_filters`, no virtual/external storage, no soft/external
links, no hard-link aliases (`_validate_links`). Arrays are written little-endian C order.

`UMAP.load` on an HDBSCAN file and `HDBSCAN.load` on a UMAP file both fail with
`ValueError` naming the found `model_kind` and the correct class (decision D3).

## 4. Rust design (`crates/fast-hdbscan`)

### 4.1 Public state type

New module `crates/fast-hdbscan/src/state.rs`, re-exported from `lib.rs` as
`fast_hdbscan::HdbscanModelState` (plus a `PredictionCoresState` enum):

```rust
/// Storage-independent snapshot of a fitted model. All fields are untrusted on import.
#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanModelState {
    pub config: HdbscanConfig,
    pub training_backend: BackendChoice,      // Cpu or Cuda { device_index } only
    pub resolved_search: HdbscanSearch,        // BruteForce or KdTree only
    pub resolved_graph: MstGraph,              // Complete or KnnGraph only
    pub raw_n: usize,
    pub n_features: usize,
    pub labels: Vec<i32>,
    pub probabilities: Vec<f32>,
    pub linkage: Vec<LinkageRow>,
    pub condensed: CondensedTree,
    pub mst: Vec<SortedMstEdge>,
    pub neighbors: Vec<i32>,
    pub core_distances: Vec<f32>,
    pub finite_index: Vec<u32>,
    pub geometry: Vec<f32>,
    pub knn_diagnostics: Option<KnnDiagnostics>,
    pub prediction_cores: PredictionCoresState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionCoresState {
    NotRequiredNoClusterTree,
    Ready { rank: usize, values: Vec<f64> },
}
```

It lives in the facade, not `fast-hdbscan-core`, because it references facade/common types
(`HdbscanSearch`, `KnnDiagnostics`, `BackendChoice`) and the core is backend-free with no
common dependency (decision D1). `KnnDiagnostics` gains `PartialEq` (it is `#[doc(hidden)]`
and already public). `Uninitialized` is intentionally not representable: an exported state
is always prediction-ready.

### 4.2 `FittedHDBSCAN::export_state(&self) -> Result<HdbscanModelState>`

1. Read only `self.state` (canonical), never the public mutable fields.
2. Prepare prediction cores through the runtime:
   - No-cluster-tree models (`hierarchy.no_cluster_tree()`): `NotRequiredNoClusterTree`,
     no device work, no tree build.
   - Otherwise use two helpers extracted from `prediction::approximate_predict`
     (lines ~984–1002 today), preserving its single-upload contract:
     - `PredictionRuntime::cores_with(&self, state, call: &CallReference, core_batch) ->
       Result<Arc<[f64]>>`: exactly the current double-checked block (`ready_cores`, take
       `init`, re-check, `compute_cores` on the **caller's** reference, publish `Ready`).
       `approximate_predict` calls it with its already-uploaded reference, so a cold predict
       still uploads geometry/tree once and reuses it for core and query batches.
     - `prepare_cores(state, runtime) -> Result<Arc<[f64]>>` (export only): returns cached
       cores if ready; else `candidate_width(s, n)` (rejects `2*s > n_finite` before any
       cold work), `preflight` for a 1-row plan, `runtime.tree(state)` (before taking
       `init`; never `tree` while holding `init`), `CallReference::upload`, `cores_with`,
       then drops the reference.
     Lock order (`init` then `cache`) and failure semantics (core failure keeps tree, leaves
     `Uninitialized`) are unchanged.
   - Consequence (inherited contract, see D6): a model with a nonempty cluster tree but
     unsupported prediction width (`2*s > n_finite`) cannot be saved; `export_state`
     returns `InvalidInput` with the same message prediction would give. The error is
     surfaced before the file is created (Python calls export first).
3. Clone the canonical vectors into the state. Use `checked_allocation_bytes` for the
   summed export size before cloning so oversized exports fail cleanly.
4. Export is idempotent and semantically independent of whether `approximate_predict` ran
   before (cached cores are reused; otherwise computed identically).

### 4.3 `FittedHDBSCAN::from_state(state, backend: BackendChoice) -> Result<Self>`

Validates every field as untrusted, in this order, before any device work or allocation
proportional to unvalidated sizes:

1. Dimensions: `config.validate()`; `raw_n > 0`, `n_features > 0`, `1 <= n_finite <= raw_n`,
   `n_finite <= i32::MAX`, `raw_n <= i32::MAX` (fit's limit; executor confirms the exact
   fit-time bound and reuses it); every `len()` equals its checked-product expected size
   (`checked_mul`, `HdbscanError::Overflow` on overflow).
2. Enumerations: `training_backend` is `Cpu` or `Cuda{..}` (not `Auto`); `resolved_search`
   is not `Auto`; `resolved_graph == resolve_graph(config.mst_graph, n_features)`;
   `resolved_graph == Complete` implies the search is consistent with fit's routing rule
   (`HdbscanSearch::resolve` cannot be re-run without the requested search, see D8; only
   reject combinations fit can never produce, e.g. `KdTree` with `n_features > 30` on the
   kNN-graph route — executor derives the exact producible set from `estimator.rs`
   lines ~980–1000 and tests each rejected combination).
3. Mapping and geometry: `finite_index` strictly ascending, `< raw_n`; geometry finite.
4. Clustering consistency (decision D4): recompute the host clustering from the persisted
   MST and config using the **same code path fit uses**. Refactor the tail of `HDBSCAN::fit`
   (estimator.rs ~1324–1357: `clusters_from_spanning_tree` / `HostClustering::one_finite`,
   `remap_linkage`, `scatter_labels`, `scatter_probabilities`, `remap_condensed`) into one
   `pub(crate) fn finalize_clustering(sorted_mst, config, finite_index, raw_n) ->
   Result<FinalClustering>` called by both fit and `from_state`. Before calling it validate
   the MST itself (length `n_finite-1`, endpoints `< n_finite`, weights non-NaN,
   nonnegative, nondecreasing — whatever `clusters_from_spanning_tree` assumes; the
   executor reads that function and validates exactly its preconditions so malformed input
   yields an error, never a panic). Then require bit-exact equality
   (`f32::to_bits`/`f64::to_bits`) of `labels`, `probabilities`, `linkage`, `condensed`
   with the recomputed values. Any mismatch → `InvalidInput("model state clustering is
   inconsistent with its spanning tree")`. This replaces bespoke structural validation of
   linkage/condensed trees.
5. Per-row outputs: `neighbors.len()` is `0` when `n_finite == 1`, else `n_finite * s`;
   each value `-1` or in `0..n_finite`. The same id rule applies to diagnostic neighbors.
   (No positional `-1` rule: whether padding is strictly trailing on every route, including
   ANN, is not established, so it is not enforced.) `core_distances.len() == n_finite`,
   each non-NaN and `>= 0`; `+inf` is allowed anywhere, because fit produces it for
   `s >= n_finite` (complete route, estimator.rs ~474/535/584, asserted by
   `tests/test_kdtree_search.rs` ~565–609), for padded kNN rows (core `knn_kruskal.rs`
   ~168–175), for singletons and on f32 overflow. Core distances are not covered by the D4
   recomputation; they are fitted outputs validated by range only (decision D13).
   `knn_diagnostics` present iff `resolved_graph == KnnGraph && n_finite > 1`; `k` must
   equal `graph_width(n_finite, s, config.knn_k)`; arrays `n_finite * k`; distances non-NaN,
   `>= 0`; `bridges <= n_finite - 1`; `finite_weight_sum` bit-equal to the sum of finite
   MST weights computed with fit's arithmetic (estimator.rs ~901–906, shared helper);
   `source` may be either value (the hidden `fit_knn_graph_with_source` selects Exact/Ann
   independently of the Auto row threshold, and the requested source is not persisted);
   `allocation_bytes` is unchecked provenance.
6. Hierarchy: `PredictionHierarchy::new(&condensed, &labels, &finite_index)` (already
   validating) via the normal `FittedState::new(FittedOutputs { .. })`.
7. Prediction cores: status must agree with `hierarchy.no_cluster_tree()`; `Ready` requires
   `rank == s`, `candidate_width(s, n_finite)` ok, `values.len() == n_finite`, every value
   finite and `>= 0`.
8. Runtime: resolve `backend` to a concrete runtime backend with the same helper fit uses to
   compute `actual_backend` (Auto → discovered CUDA else CPU, explicit choices strict, no
   fallback). Build the KD tree eagerly with `KdTree::build(&geometry, n_finite,
   n_features)` (decision D2) and construct `PredictionRuntime` with that tree and the
   validated `CoreStatus` (new constructor `PredictionRuntime::restored(backend, tree,
   cores)`). The eager build is skipped for no-cluster-tree models, whose prediction is the
   shortcut and never queries the tree (tree stays `None`, as after fit on brute-force
   routes).
9. Construct via the existing private `FittedHDBSCAN::from_state`-style constructor (rename
   the current private `from_state(state, runtime_backend, tree)` to `from_parts` to free the
   public name). The public `backend` field keeps the **training** backend (provenance, as
   today); prediction uses the runtime backend.

Atomicity: `from_state` publishes nothing on error (it only returns a value on success).

### 4.4 Module docs

Replace the "Future persistence boundary (not implemented)" section in
`crates/fast-hdbscan/src/prediction.rs` with the implemented contract (export prepares
cores; tree rebuilt on load; hierarchy reconstructed; runtime backend chosen at load;
training backend is provenance).

## 5. Native bridge (`crates/fast-umap-python/src/hdbscan.rs`)

- `FittedHDBSCAN.export_state(self) -> (dict[str, ndarray | None], dict[str, object])`,
  GIL released (`py.detach`) around `export_state`. Arrays keyed by the dataset names of
  section 3, all 1-D flat as stored (Python reshapes `neighbors`/`geometry`/diagnostics for
its public attributes; `neighbors_` keeps today's `(1,0)` singleton shape). Metadata keys: the config attrs (as Python
  values, optionals as `None`/int), provenance strings, `raw_n`, `n_finite`, `n_features`,
  `cores_status`, `cores_rank`, knn diagnostic scalars. String spellings are defined once in
  Rust (`fn search_name`, `fn graph_name`, …) with matching parsers.
- `#[pyfunction] load_hdbscan(arrays, metadata, backend="auto", device_index=0) ->
  FittedHDBSCAN`: strict extraction (exact dtypes via `PyReadonlyArray`), missing keys →
  `ValueError`, builds `HdbscanModelState`, calls `from_state` with GIL released, maps
  errors with the existing `map_hdbscan_error`. Backend string parsing reuses the existing
  HDBSCAN kwargs backend parser (`"auto"`, `"cpu"`, `"cuda"` + `device_index`).
- Register in the module init next to `fit_hdbscan`; add stubs to
  `python/fast_umap/_fast_umap.pyi`.

## 6. Python design

### 6.1 Shared serialization helpers

Create `python/fast_umap/_serialization_common.py` and move, unchanged in behavior, the
model-agnostic pieces of `_serialization_v1.py`: `DurabilityUncertainError`, `_deps(kind)`
(error message names the model kind), `_filters`, `_validate_links`, `_check_scalar`,
`_check_text`, `_attr`, root-attribute writing/validation, dataset writer (chunks, Blosc2),
and the atomic publication routine (`mkstemp` sibling, write, `fsync`, `os.link` for
no-clobber or `os.replace` for overwrite, directory `fsync`, `DurabilityUncertainError`,
temp cleanup). `_serialization_v1.py` keeps the UMAP schema and re-exports
`DurabilityUncertainError` so its import path is unchanged (decision D7). Existing
`python/tests/test_serialization.py` must pass unmodified; it is the regression proof of the
refactor.

### 6.2 `python/fast_umap/_hdbscan_serialization_v1.py`

- `save_model(path, arrays, metadata, *, overwrite)` writes schema v1 through the common
  publisher.
- `load_model(cls, path, *, backend, device_index)`: validates backend/device_index first
  (same rules as UMAP: `device_index` non-bool non-negative int), opens read-only, validates
  links, exact group/attr/dataset sets, dtypes/ranks/shapes vs. metadata, filter options,
  canonical optionals, enum spellings, then calls `_fast_umap.load_hdbscan`. Semantic
  validation is Rust's job; Python validates the container so malformed files fail before
  native import and before reading dataset payloads larger than the declared shapes.
- Wrong `model_kind` → `ValueError` naming the expected class.

### 6.3 `HDBSCAN` (`python/fast_umap/hdbscan.py`)

- Extract the post-fit publication of `_fitted_state` from `_fit` into
  `_publish_fitted(self, native, n_features)`. It derives `finite_rows` from
  `native.finite_index` and `len(native.labels)` instead of the input mask, so fit and load
  share it (preserving the float32-when-all-finite / float64-scattered `probabilities_`
  quirk identically).
- `save(self, path, overwrite=False) -> None`: requires fitted; calls
  `self._native.export_state()` (may run prediction-core preparation on the fit-time
  runtime backend and may raise, e.g. unsupported width) **before** touching the
  filesystem, then `save_model`.
- `@classmethod load(cls, path, *, backend="auto", device_index=0) -> HDBSCAN`: constructs
  `cls(**constructor)` from the persisted native config (`max_cluster_size=None` for the
  native `None`; every unsupported upstream parameter at its default) plus the load-time
  `backend`/`device_index`, then `_publish_fitted`. Constructor parameters on the loaded
  instance therefore describe the restored model and the load-time backend (decision D9).
- Docstrings document: optional `[serialization]` extra, prediction-readiness preparation
  at save (cost and the unsupported-width failure), runtime backend chosen at load,
  training backend kept only as provenance, numeric tolerance across backends.

## 7. Tests

### 7.1 Rust — `crates/fast-hdbscan/tests/test_serialization.rs` (CPU; CUDA cases gated)

Shared helper asserting every public accessor, `knn_diagnostics()`, `search()`,
`resolved_mst_graph()`, `dbscan_clustering` at several eps, and `approximate_predict` on
fixed queries are equal (bitwise on the same backend) between original and restored.

Round trips (each both never-predicted and already-predicted, asserting equal exported
states, i.e. cache-independence):
1. brute-force complete graph (< 256 rows);
2. KD complete graph (≥ 256 rows, ≤ 30 features), `reproducible` both values;
3. kNN graph route (> 30 features, and forced `MstGraph::KnnGraph`), with diagnostics;
4. mixed nonfinite raw rows (remapped linkage/condensed, scattered labels/probabilities);
5. no-cluster-tree, root-only, and single-finite-row models (`NotRequiredNoClusterTree`,
   empty linkage/MST/condensed where applicable, `+inf` core distance);
6. epsilon / persistence / `max_cluster_size` / `leaf` / `allow_single_cluster` configs;
7. `export_state` → `from_state` → `export_state` identity.

Failure cases (each a distinct mutated state, asserting error variant):
unsupported width at export (`2*s > n_finite`, nonempty tree) with no state published;
dimension/length mismatches and `checked_mul` overflow sizes; nonfinite geometry;
unsorted/duplicate/out-of-range `finite_index`; `Auto` backend/search/graph in state;
graph inconsistent with `mst_graph`/features; malformed MST (length, endpoints, NaN,
negative, unsorted) without panics; single flipped label, probability bit, linkage row,
condensed entry (clustering-consistency check); neighbors out of range (in-domain `-1` padding is preserved, tested by round trip);
negative/NaN core distances; cores status/rank/length/value violations; knn diagnostics
presence/shape mismatches.

Backend: CPU always. CUDA round trip and CPU-saved → CUDA-loaded prediction
(tolerance-based) under the repository's `FAST_UMAP_REQUIRE_CUDA=1` convention (skip only
when unset and CUDA unavailable). Loading with an explicit unavailable backend fails
strictly.

Additional round trips: multi-row infinite-core models (`s >= n_finite`, both search
routes, no-cluster-tree so they are saveable); kNN route with ANN-source diagnostics
(`n_finite > 4096`, > 30 features); CUDA-saved → CPU-loaded as well as CPU → CUDA.
Failure cases also cover diagnostics `k` mismatch (including `k=0`), `bridges` out of
range, wrong `finite_weight_sum`. Round trip also covers `fit_knn_graph_with_source` with
explicit `KnnSource::Ann` below 4096 rows and `KnnSource::Exact` above it. The comparison
helper excludes `prediction_cache_status()` (tree readiness legitimately differs after
load).

Unit tests (in-crate): `cores_with` shared by predict/export — upload counter is exactly 1
for a cold predict and 1 for a warm predict, and 1 for export of a never-predicted model; export of a model whose core computation fails (fault injection
`Injection::Cores`) returns the error and leaves cores `Uninitialized` for retry.

### 7.2 Python — `python/tests/test_hdbscan_serialization.py` (uses `tmp_path`)

Mirrors `test_serialization.py`: round trip of all fitted attributes (dtype included, incl.
float64 probabilities for nonfinite input), `dbscan_clustering`, `approximate_predict` with
and without connectors, no-cluster-tree warning behavior after load, constructor params of
loaded model; never-predicted vs predicted saves produce equal files' datasets; independent
`h5py` reader without plugin path can list the schema; device_index forwarding and bool
rejection (monkeypatched `load_hdbscan`); CUDA strict load (skip unless CUDA); no-clobber
and overwrite; bad version/format/kind (incl. UMAP file to `HDBSCAN.load` and vice versa);
external/soft links; corrupt dataset rejected before native import (monkeypatch asserts
native not called); parametrized malformed schema entries (missing/extra attr, wrong dtype,
noncanonical optional, bad enum, wrong shape, missing/extra dataset, cores present when not
required); invalid filter settings; lazy optional-dependency error; temp cleanup, no-clobber
race, pre-commit failures, post-commit directory-fsync failure (reuse the UMAP test
patterns, now targeting the shared publisher); unfitted `save` raises; unsupported-width
`save` raises and creates no file.

## 8. Documentation

- README: extend "Fitted model serialization" with an HDBSCAN subsection (API, extra, what
  is stored, save-time prediction preparation and its failure mode, load-time backend,
  provenance, tolerance across backends, schema v1 reference).
- AGENTS.md: architecture paragraph notes HDBSCAN `HdbscanModelState` export/import in the
  facade and the Python `_hdbscan_serialization_v1.py`/`_serialization_common.py` split;
  test list mentions the new tests.
- `prediction.rs` module docs per 4.4.
- `approx_predict_plan.md` is a historical artifact and is not edited.

## 9. Implementation sequence

Each stage leaves the tree building and its existing tests passing.

1. **Stage A — spike (no product code):** confirm with a throwaway script that h5py +
   hdf5plugin create, read and report the Blosc2 filter on zero-length 1-D chunked
   datasets (`shape=(0,)`, `chunks=(1,)`) for every dtype used. If unsupported, switch D5 to "absent with presence attr" before Stage E.
2. **Stage B — Rust refactors, no behavior change:** `finalize_clustering` extraction from
   fit; `PredictionRuntime::cores_with` / `prepare_cores` extraction (4.2) used by
  `approximate_predict`; rename
   private constructor to `from_parts`. Run `cargo test -p fast-hdbscan` and the existing
   prediction tests.
3. **Stage C — Rust state:** `state.rs`, `export_state`, `from_state`,
   `PredictionRuntime::restored`, `KnnDiagnostics: PartialEq`, re-exports; Rust tests 7.1.
4. **Stage D — bridge:** `export_state`, `load_hdbscan`, `.pyi` stubs.
5. **Stage E — Python:** `_serialization_common.py` refactor (UMAP tests unchanged and
   green), `_hdbscan_serialization_v1.py`, `HDBSCAN.save/load`, `_publish_fitted`; tests 7.2.
6. **Stage F — docs** (section 8) and full gates (section 10).

## 10. Verification gates

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-hdbscan-core
cargo test -p fast-hdbscan
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_serialization -- --nocapture
cargo test -p fast-umap          # UMAP untouched in Rust; sanity only
uv sync --extra serialization
uv run maturin develop --release
uv run --no-sync pytest python/tests/test_serialization.py python/tests/test_hdbscan_serialization.py \
  python/tests/test_hdbscan.py python/tests/test_hdbscan_prediction.py python/tests/test_hdbscan_resources.py
uv run ruff check && uv run ruff format --check && uv run ty check
bash analyze_hdbscan_stages.sh --rows 100000   # Stage B touches the fit tail
```

The stage-analysis run is required by AGENTS.md because Stage B refactors fit
finalization; it must show no fit-stage regression and real CUDA execution. Compilation or
skipped CUDA tests are not acceptable substitutes.

## 11. Completion criteria

- All gates in section 10 pass, with CUDA tests actually executed.
- Round trip preserves every fitted attribute and method for every route in 7.1.
- Every malformed-file and malformed-state case in 7.1/7.2 fails with a specific error
  before native import (Python container errors) or before model construction (Rust).
- Saving never depends on whether prediction ran first.
- Docs updated as in section 8.

## 12. Decisions and rationale

| # | Decision | Rationale | Alternatives considered |
|---|---|---|---|
| D1 | `HdbscanModelState` lives in the `fast-hdbscan` facade, not in `fast-hdbscan-core`. | UMAP keeps `ModelState` in its core, but the HDBSCAN state references `HdbscanSearch`, `KnnDiagnostics` (facade) and `BackendChoice` (common); `fast-hdbscan-core` has no common dependency and AGENTS.md requires it stay backend-free. Moving those types into core is churn with no consumer benefit. | Move `HdbscanSearch`/`KnnDiagnostics` to core and add a common dependency to core (rejected: violates crate layering). |
| D2 | Do **not** persist the KD tree; rebuild it eagerly with `KdTree::build` on load (skipped for no-cluster-tree models, which never query it). | The builder is deterministic host code over the persisted geometry, so the rebuilt tree is identical to the fit-time one. Persisting it would require a full untrusted-topology validator (permutation, ranges, leaf flags, bound enclosure, level limits) whose only safe implementation is "rebuild and compare", making storage pointless. Eager build makes load fail early and makes the loaded model prediction-ready like a saved-after-predict model. Cost: O(n log n · d) host time and the tree memory at load. This deviates from the wording in `approx_predict_plan.md` §6 ("always prepares and saves the host KD tree"); that text said "as explicitly chosen", and this is the explicit choice. | Persist and validate the tree (rejected: large validator, no benefit); lazy rebuild at first predict (rejected: defers failures, contradicts "complete" loaded model). |
| D3 | Reuse the UMAP container: same `format` string, discriminated by `model_kind="hdbscan"`, its own `schema_version` sequence starting at 1. | One file family for the package; kind mismatch gives a precise error instead of "not a fast_umap model". | New `format` string per model (rejected: worse error messages, no benefit). |
| D4 | Validate linkage/condensed/labels/probabilities by recomputing the host clustering from the persisted MST + config through the shared fit code and requiring bit-exact equality. | One deterministic host pass replaces several bespoke tree validators, guarantees the loaded model is internally consistent (so `dbscan_clustering`, prediction and attributes agree), and cannot drift from fit because the code is shared. Persisted outputs remain the public values; the recomputation is a check, not a second authority. Cost is the host clustering stage only (no neighbor search). | Persist only the MST and recompute everything (rejected: makes outputs implicitly derived and hides corruption); structural validation only (rejected: misses semantically inconsistent but well-formed trees). |
| D5 | Datasets that can legitimately be empty are always written with length 0. | Keeps the dataset set fixed (simpler exact-set validation) and the shapes self-describing. UMAP's "nonempty" rule does not fit HDBSCAN, whose linkage/MST/condensed trees are empty for single-finite-row models. Verified by Stage A; fall back to presence attributes if the filter rejects empty chunked datasets. | Presence flags per array (kept as fallback). |
| D6 | Keep the inherited contract: a nonempty-cluster-tree model with unsupported prediction width (`2*s > n_finite`) cannot be saved; export fails before any file is created. | Explicitly stated in the approved `approx_predict_plan.md` §6 ("always-prediction-ready contract"). Such a model can never predict anyway. Flagged for reviewer attention because it makes a successfully fitted model unsaveable. | Add a `cores_status="unsupported_width"` so all other outputs persist and prediction errors after load exactly as before (simple, arguably friendlier; recommended if the user prefers saveability). |
| D7 | Factor model-agnostic I/O into `_serialization_common.py`; keep `_serialization_v1.py` as the UMAP schema and its `DurabilityUncertainError` import path. | Avoids duplicating ~150 lines of security-relevant validation/atomic publication logic; existing UMAP tests prove the refactor. | Copy-paste per model (rejected: two copies of link/filter/durability logic to keep in sync). |
| D8 | Persist only the resolved search, not the requested `HdbscanSearch`. | `FittedState` does not retain the requested search and `FittedHDBSCAN::search()` exposes only the resolved route; prediction never re-resolves. Adding it would change fit state solely for persistence. Load validates only combinations fit can produce. | Add `requested_search` to `FittedState` (UMAP stores requested search; deferred until a consumer needs it). |
| D9 | Loaded Python instance's constructor attributes come from the persisted native config plus the load-time `backend`/`device_index`; upstream-only/unsupported parameters take defaults; no Python-precision group. | HDBSCAN floats are native `f64`, so no precision is lost (UMAP needed `python_precision` because its native floats are `f32`). The loaded estimator then refits with the same settings on the chosen backend. Original Python spellings (e.g. `max_cluster_size=np.inf` vs `None`, numpy ints) are normalized, which is observable but harmless. | Persist raw Python constructor values (rejected: they are not authoritative after fit and may be arbitrary objects). |
| D10 | Training backend stored as provenance (`training_backend`, `training_device_index`); the runtime backend is resolved at load from the `backend` argument with fit's strict rules. The public Rust `backend` field keeps the training backend. | Matches the existing documented contract ("load selects the runtime backend independently of the training backend recorded as provenance") and UMAP's load-time backend argument. | Default load to the training backend (rejected: files must be portable between CPU and CUDA hosts). |
| D11 | Persist `KnnDiagnostics` (including `allocation_bytes`) when present. | The prior plan requires a complete model including diagnostics; `knn_diagnostics()` is a public (hidden) accessor that must round-trip. `allocation_bytes` is fit-time provenance, persisted verbatim. Cost: `2·n·k·4` bytes on kNN-route models. | Omit diagnostics (rejected: incomplete model). |
| D13 | All datasets are stored 1-D and reshaped on read; singleton `neighbors` is stored with length 0; `core_distances` and `neighbors` are validated by domain/range only (no positional `+inf`/padding rules). | Fit produces `+inf` cores and `-1` padding across several branches; encoding branch-specific rules invites rejecting valid models, and these arrays do not affect clustering consistency (D4). 1-D storage sidesteps zero-extent chunk axes. | Recompute cores from geometry on load (rejected: requires neighbor search at load). |
| D14 | `export_state` shares the core-initialization block with prediction through `cores_with(&CallReference)`; export owns a temporary upload. | Keeps cold prediction at one upload per call (approx_predict_plan.md §6). | Standalone helper with its own upload (rejected: double cold upload). |
| D12 | No Rust file I/O API; Rust exposes `export_state`/`from_state` only. | Same split as UMAP; keeps HDF5 out of the Rust dependency graph. | Rust HDF5/`serde` format (rejected: new dependencies, duplicates the Python path). |
