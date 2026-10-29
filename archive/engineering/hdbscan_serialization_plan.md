# Standalone HDBSCANPredictor and persistence design

Status: **Both design review gates approved.** The standalone predictor design in §§1–7 passed Gate 1 review by architect `3-PredictorArch` and critic `4-PredictorCritic`. After persistence was added, both reviewers independently approved the combined design at Gate 2 with no blocking findings. Review receipts: `agent://3-PredictorArch` and `agent://4-PredictorCritic`. These are design approvals, not implementation authorization or runtime verification; this task changes only this plan.

Review sequence:

1. **Gate 1 — approved:** standalone `HDBSCANPredictor` API, ownership, readiness and prediction design. Both architect and critic approved; no blocking findings. Critic advisories are resolved in §§3–7 and D22.
2. **Gate 2 — approved:** architect and critic approved the predictor-persistence design after Gate 1, with no required changes. Advisory implementation notes: §8.5's keyword-only load signature governs; the KD row bound is `n_finite <= i32::MAX` alongside checked index conversions; explicit CUDA load performs discovery even for shortcut models and must be documented.
3. Implementation remains a separately authorized task. Complete fitted-estimator persistence is out of scope.

No user commit is a prerequisite; decisions must be based on the current source tree. No tests, builds, formatters or review gates are run by this design-authoring task.

## 1. Goal and non-goals

Add an explicit conversion from a fitted model to a small prediction-only object:

```rust
impl FittedHDBSCAN {
    pub fn to_predictor(&self) -> Result<HDBSCANPredictor>;
}

impl HDBSCANPredictor {
    pub fn approximate_predict(&self, input: &[Vec<f32>]) -> Result<PredictionResult>;
}
```

The Python estimator gains `HDBSCAN.to_predictor() -> HDBSCANPredictor`; the returned Python predictor supports the same `approximate_predict(X, return_connecting_points=False)` contract as the fitted estimator.

The predictor must continue to work after the fitted estimator is dropped. It owns/references only prediction-specific host state and prediction runtime state, not labels, probabilities, linkage, condensed tree, MST, fit neighbors, fit core distances, config, or kNN diagnostics. It uses the same exact KD search, candidate ordering, hierarchy math, batching, output semantics, and strict backend selected for the fitted model. Do not add an alternate predictor algorithm.

Non-goals:

- No fitted-estimator serialization, reconstruction, or round-trip of fit-only outputs.
- No product implementation until Gate 2 is approved; this document adds a persistence design, not working `save`/`load` behavior.
- No pickle or `__getstate__` support, schema migration, or compatibility fallback for unknown versions.
- No backend-switch argument on conversion or prediction; no CPU/CUDA fallback.
- No changes to fitting, neighbor search, prediction formulas, or the public behavior of existing fitted-model prediction.
- No new dependencies.

## 2. Existing implementation traced

### 2.1 Canonical fit state and hierarchy

`crates/fast-hdbscan/src/prediction.rs` currently defines `FittedState`, `FittedOutputs`, `PredictionRuntime`, the lazy `CoreStatus`, checked prediction planning, and the facade orchestration. `FittedState` contains both fit outputs and prediction inputs: compact row-major finite training `geometry: Vec<f32>`, `n_features`, resolved `min_samples`, and a validated `PredictionHierarchy`, in addition to labels, probabilities, linkage, condensed tree, MST, neighbors, fit core distances, finite mapping, route information, and diagnostics.

The hierarchy is `fast_hdbscan_core::PredictionHierarchy` in `crates/fast-hdbscan-core/src/prediction.rs`. Its constructor consumes the final raw-row condensed tree, raw fitted labels, and ascending finite-to-raw mapping. The resulting compact validated hierarchy owns the mapping needed to convert a compact neighbor id back to a raw training-row connector; it retains only cluster nodes/point attachments and that mapping, not the original labels or condensed-tree columns. Its public methods validate prediction cores and execute the established hierarchy math.

`FittedHDBSCAN` in `crates/fast-hdbscan/src/estimator.rs` exposes detached mutable fit-output fields, while its canonical `Arc<FittedState>` is immutable. `PredictionRuntime` is separately shared by `Arc` and holds the concrete runtime backend plus a lazy host KD tree and prediction-core status. The fit-time `core_distances: Vec<f32>` are public fit results; they are not the prediction core vector.

### 2.2 Current prediction path and invariants

`FittedHDBSCAN::approximate_predict` calls the single implementation in `crates/fast-hdbscan/src/prediction.rs`:

1. Validate a nonempty rectangular finite query matrix with exactly the fitted feature count; flatten it to row-major `f32`.
2. If `PredictionHierarchy::no_cluster_tree()`, return all `-1` labels, `0.0` strengths, `-1` connectors, and `no_cluster_tree = true`. This precedes candidate-width checks and does no backend resolution, tree construction, device upload, or search.
3. Otherwise require `candidate_width(s, n_finite)`, i.e. checked `2*s <= n_finite`, then preflight checked device and host sizing before cold work.
4. Resolve `Target` from the runtime's already-concrete backend. `Target::resolve` rejects a resolved backend different from the requested runtime backend. The operation is strict; it does not retry on another backend.
5. Obtain the retained host `KdTree` or build it from compact finite geometry. `CallReference::upload` creates a call-local device reference and packed KD data. The device reference is dropped at the end of the call.
6. If prediction cores are not cached, find each training point's inclusive-self exact KD neighbors at width `s`, read back squared `f32` distances, convert the rank `s-1` value to an `f64` Euclidean core, validate the full vector, and publish it as `CoreStatus::Ready { rank: s, values: Arc<[f64]> }`. `s == 1` yields all-zero prediction cores without a core search.
7. Query exact KD candidates at width `2*s` in bounded batches. `PredictionHierarchy::predict_squared_batch` converts the returned squared distances to `f64`, applies the existing mutual-reachability and hierarchy calculations, and returns raw-row connectors.

The current `PredictionRuntime` serializes fallible cold initialization with `init`, publishes caches under `cache`, and performs warm searches without holding either lock. A failed tree build publishes no tree. A failed core computation leaves a valid published tree in place, publishes no partial cores, and permits retry. Preserve this behavior.

The fitting route may already have retained a host KD tree (KD complete-graph fit); other routes build it lazily for prediction. Runtime backend is the concrete backend selected by fit, not the config's requested `Auto` value. CPU and CUDA both use the same exact KD prediction orchestration and core math.

### 2.3 Current resource preflight

`PredictionPlan` and `plan_prediction` use checked shapes matching the tree builder and upload path. They bound individual device allocations by the selected device's allocation limit, use checked live-set sums, and account host/device peaks by phase rather than treating the plan as a memory-availability guarantee. Current prediction planning includes both the core-preparation and query phases, including query outputs; it is not an exact conversion-only plan.

A predictor conversion that prepares cores must not fake a query row merely to reuse the current combined planner. Factor/reuse the sizing calculations so conversion preflights the actual tree/reference/core phases only, and query prediction continues to preflight its own real query/output batches before tree build, upload, or search. Keep size-overflow and device-limit errors ahead of cold allocations.

## 3. Rust design

### 3.1 Prediction-only shared payload and ownership

Refactor the current prediction inputs into one crate-private immutable prediction-only payload (proposed name: `PredictionState`) containing:

- compact finite geometry in row-major `f32`;
- `n_features` and resolved `min_samples` (`s`);
- the already validated `PredictionHierarchy`, which owns the compact-to-raw mapping and cluster attachment metadata.

The fit canonical state should hold an `Arc` to this payload and keep the fit-only canonical outputs separately. Move, do not clone, the compact geometry into the prediction payload. The fitted facade's public `finite_index` remains its existing output view; do not add a second standalone mapping when the hierarchy already owns the exact mapping used for connector translation.

`HDBSCANPredictor` should contain only an `Arc` to the prediction payload and an `Arc` to `PredictionRuntime`. Refactor runtime/tree/core helpers and the shared orchestration to accept the prediction payload rather than requiring `&FittedState`. `PredictionRuntime` must not retain or indirectly own `FittedState`. Thus cloning a predictor and dropping the fitted model retains only the compact geometry, hierarchy/mapping, runtime backend, host tree/core caches, and synchronization state. Existing fitted-model prediction delegates to that same refactored orchestration and remains behaviorally unchanged.

The runtime cache remains shared between the fitted model and its predictor: conversion reuses a fit-created tree or cached prediction cores, and successful cold conversion publishes the same canonical core cache that fitted prediction uses. The predictor's own type/API exposes no fit outputs or clustering methods.

### 3.2 Hierarchy authority

Use the existing validated `PredictionHierarchy` as the sole in-memory hierarchy authority. Build it once through `PredictionHierarchy::new` from the final fitted condensed tree, raw labels, and finite mapping during fit-state construction (as today); `to_predictor` shares that immutable validated value. Do not retain or copy the full labels or condensed tree into the predictor, and do not independently recalculate hierarchy math during conversion.

This makes the predictor boundary a compact derived hierarchy rather than a fitted-model snapshot. The persistence design below defines a minimal portable hierarchy representation with a validating constructor; it does not serialize the source labels or condensed tree and does not treat current private hierarchy fields as trusted bytes.

### 3.3 Conversion readiness and failure atomicity

`FittedHDBSCAN::to_predictor(&self) -> Result<HDBSCANPredictor>` is fallible and returns only a ready predictor:

- No-cluster-tree: return a predictor sharing the prediction payload/runtime immediately. Preserve the existing `NotRequired(NoClusterTree)` status; skip `2*s` width checks, target resolution, tree build, upload, and core computation. Its `approximate_predict` still validates query shape/values before returning the established no-cluster shortcut.
- Cluster tree: validate `candidate_width(s, n_finite)` before any cold work, including the rank-one case. Return the same typed width error that prediction would return. For `s == 1`, checked-allocate/publish `n_finite` zero `f64` prediction cores at rank 1 without core KD search or device work. Do not build a host tree solely for conversion; query prediction later uses the existing lazy tree/upload path and strict runtime backend.
- Cluster tree and `s > 1`: preflight the conversion-only core phase, resolve the stored concrete runtime backend strictly, obtain/build the host tree, create one call-local reference upload, compute exact inclusive-self rank-`s` cores with the existing `compute_cores` arithmetic and batching, and validate with `PredictionHierarchy::validate_prediction_core`.
- Publish a complete `Ready { rank: s, values }` only after the entire core vector passes validation. Then construct/return the predictor. A failure returns no predictor and never modifies canonical fit outputs. A valid host tree may remain cached if core work fails, as in existing prediction semantics; partial cores must never be published. Retry must work.

If cores were already ready, reuse them without another core search or upload. `to_predictor` itself performs no geometry/hierarchy clone: only `Arc` clones after readiness succeeds. The predictor invariant is therefore: no cluster tree with `NotRequired`, or a cluster tree with validated rank-`s`, length-`n_finite` finite nonnegative `f64` cores.

The public predictor method is:

```rust
impl HDBSCANPredictor {
    pub fn approximate_predict(&self, input: &[Vec<f32>]) -> Result<PredictionResult>;
}
```

It delegates to the same checked query path and exact KD/hierarchy code. Keep `PredictionResult` and its fields/semantics unchanged. No public predictor constructor from arbitrary arrays is added in this stage; validation/source identity are provided by conversion from a fitted model.

Expose `pub fn n_features(&self) -> usize` on `HDBSCANPredictor`, matching the existing `FittedHDBSCAN::n_features()` accessor. The native bridge uses this accessor for the existing fitted-width error rather than retaining the fitted estimator or inferring width from data. Also expose `#[doc(hidden)] pub fn prediction_cache_status(&self) -> Result<PredictionCacheStatus>` on the predictor so cache/resource behavior remains inspectable after the fitted model is dropped; this is an explicit diagnostic/test seam, not a fit-output API.

### 3.4 Backend and error policy

Conversion does not accept a backend override. A predictor inherits the fitted model's concrete runtime backend (`PredictionRuntime::backend()`), not the training `Auto` choice or public config. `Target::resolve` remains strict on every actual device operation; unavailable CUDA, device-index errors, allocation errors, and numerical failures propagate without fallback. This gives one meaning to existing fitted prediction and to detached prediction. Persistence has a distinct explicit load-time backend choice (§8.4); it does not alter conversion or fitted-model prediction.

## 4. Python API and bridge

### 4.1 Native extension

In `crates/fast-umap-python/src/hdbscan.rs`, add a frozen native `HDBSCANPredictor` pyclass owning `fast_hdbscan::HDBSCANPredictor`, and add `FittedHDBSCAN.to_predictor()`. Detach the GIL while Rust prepares cores; map the existing `HdbscanError` variants through `map_hdbscan_error`. The returned native object owns no `RustFittedHDBSCAN` and no Python reference to its wrapper.

Give the native predictor the same `approximate_predict` signature and output conversions as native fitted prediction: input `float32` 2-D array, optional strict bool for connecting points, labels/connectors as `np.intp`, strengths as `float64`, raw-row connector IDs, and exactly one existing no-cluster-tree `UserWarning` at the requested stack level. Factor common bridge output/warning handling where that avoids behavioral drift; do not fork prediction calculations. Register the class in `crates/fast-umap-python/src/lib.rs` and add both methods/types to `python/fast_umap/_fast_umap.pyi`.

The native predictor pyclass validates feature width using its owned Rust `inner.n_features()`; add a Rust `HDBSCANPredictor::n_features()` accessor and a native read-only `n_features` getter, and reflect it in the stub. This resolves Gate 1's feature-count advisory without retaining the fitted model.

### 4.2 High-level Python wrapper

Add a public lightweight `HDBSCANPredictor` in `python/fast_umap/hdbscan.py`, export it from `python/fast_umap/__init__.py`, and expose:

```python
predictor = fitted_hdbscan.to_predictor()
labels, strengths = predictor.approximate_predict(X)
labels, strengths, connectors = predictor.approximate_predict(X, return_connecting_points=True)
```

`HDBSCAN.to_predictor()` requires the estimator to be fitted, converts through its native fitted object, and stores only the returned native predictor plus `n_features_in_`; it must not retain the parent estimator or its fitted arrays. Reuse the same input normalization, finite-value/feature validation, strict flag handling, tuple shaping, dtype conversion, and warning stack-level behavior as `HDBSCAN.approximate_predict`. The existing fitted estimator method remains available and unchanged to callers.

Document that conversion may synchronously perform one cold prediction-core preparation on the fit-time runtime backend, and may fail before returning a predictor; subsequent predictions still use call-local uploads and bounded batches. Do not imply that conversion creates a serialized or backend-portable artifact.

## 5. Resource and lifetime accounting

Distinguish retained prediction state from per-operation peaks; estimates are checked plans, not guarantees of available memory.

- Retained host state: compact geometry `n_finite * n_features * sizeof(f32)`; validated hierarchy including its finite-to-raw mapping, nodes and point attachments; optional host `KdTree` arrays (`indices`, node range/flags, and min/max bounds); and prediction-specific `f64` cores (`n_finite * sizeof(f64)`) when a cluster tree exists. Fitted-only arrays are not retained by `HDBSCANPredictor`.
- Cold conversion for `s > 1`: check `2*s <= n_finite`, dimensions/products and every planned allocation/live sum before cold work; model tree build and temporary stack; one host geometry/tree clone for `CallReference::upload`; device reference data, tree arrays, bounds and packed reference; bounded `core_batch * s` IDs and squared distances/readback; `f64` core vector plus publication storage. The call-local device reference is dropped before return. Reuse the existing KD sizing/packing formulas and accounting; add a core-only plan instead of charging nonexistent query/output allocations.
- Conversion at `s == 1` with a cluster tree: validate `2*s <= n_finite`, then account for the zero `f64` core vector/publication only; no tree or backend/device work is needed to make cores ready.
- Warm conversion: ready core Arc and tree Arc are shared, with no data clone or core computation. It must not charge the full fitted-model state.
- Each non-shortcut query call: preserve existing preflight of query/output bytes, bounded query batches and exact `2*s` candidates, live host/device phase sums, allocation page limits and call-local reference lifetime. Predictor and fitted paths must share these calculations.
- No-cluster-tree prediction: retains normal output allocations but performs no backend discovery, tree construction, device upload, or neighbor search, regardless of `s`.

**Required diagnostic rescope:** keep the `PredictionCacheStatus.canonical_bytes` name but change its meaning to only the prediction payload's non-hierarchy canonical allocation: compact geometry `len * size_of::<f32>()` (the existing byte-count convention does not count inline scalar metadata). It must exclude every fit-only array and exclude hierarchy, tree, and prediction cores, which remain separately reported by `hierarchy_bytes`, `tree_bytes`, and `cores.len() * size_of::<f64>()`. Thus it no longer reports the previous complete `FittedState` bytes; summing those four components gives retained prediction host buffers without double-counting. Add/use the doc-hidden `HDBSCANPredictor::prediction_cache_status()` accessor from Rust tests to verify cache state after dropping the fitted model. Do not add a fit-resource estimate that claims conversion/query peaks are included unless it models those phases explicitly.

Update the module-level persistence note in `crates/fast-hdbscan/src/prediction.rs` as part of implementing the predictor boundary. Remove its stale proposal to serialize full `FittedState`, require a KD tree even for a singleton, and reconstruct hierarchy from fitted outputs. Replace it with the approved prediction-only payload, ready-core and no-cluster shortcut contracts, portable hierarchy export/import boundary, load-time runtime backend rule, and the fact that no persistence exists until the separately gated implementation lands.

## 6. Tests to add or extend (design only)

### Rust

Extend `crates/fast-hdbscan/tests/test_prediction.rs` and in-module prediction tests:

- For clustered models, assert `to_predictor().approximate_predict(queries)` matches fitted prediction for labels, raw connectors, strengths, and `no_cluster_tree`, across a cold and already-warm fitted runtime and across brute-force/KD/kNN fit routes. Use existing oracle and CUDA gating conventions; fitting route does not change predictor search route.
- Convert, drop the fitted model, and predict successfully. Assert the public predictor has no fit-output API and that its predictions are independent of mutations to detached public fitted output views made before conversion.
- Assert conversion shares ready cores/tree and does not duplicate geometry/hierarchy; cold conversion prepares cores once, later predictor queries reuse cores, and conversion from an already-ready model does no core search. Count uploads for the `s > 1` cold conversion and per-call query uploads without changing the existing one-upload-per-prediction-call contract.
- Cover `s == 1` cluster-tree conversion with zero cores and no conversion-time backend/tree/upload work; subsequent query remains correct.
- Cover no-cluster-tree conversion with `s` larger than candidate width: conversion and shortcut prediction succeed without tree/core/device work and produce exact all-noise/zero/`-1` results.
- Inject tree/core failures: no predictor returned; failed tree build publishes no tree; failed core build may retain a valid tree but publishes no partial cores; retry returns a usable predictor. Test width failure precedes all cold work for a nonempty-tree state.
- Exercise resource-plan overflow/tiny allocation limits and verify the conversion-only plan does not include fictitious query/output rows. Keep CPU always-run and use existing CUDA requirement convention.

### Python

Extend `python/tests/test_hdbscan_prediction.py`:

- Compare fitted and standalone predictor output tuples/dtypes with and without connectors, for clustered and no-cluster-tree models; verify the warning count/message and stacklevel behavior.
- Verify `to_predictor` on an unfitted estimator raises the established fitted-state error; predictor works after estimator deletion or refit and does not retain parent arrays/native fitted object.
- Cover `ArrayLike` conversion, float32 finite input, wrong feature count, nonfinite values, empty dimensions, bool/`np.bool_` flag acceptance and invalid flag rejection exactly as fitted prediction.
- Check conversion failure propagates as the existing Python exception mapping and no Python predictor object is returned.

The tests above are Gate 1 predictor tests. The distinct persistence test plan is in §8. Do not implement either test suite before its corresponding implementation approval.

## 7. Gate 1 approval record and predictor implementation surface

Gate 1 was approved by architect `3-PredictorArch` and critic `4-PredictorCritic` (review receipts `agent://3-PredictorArch` and `agent://4-PredictorCritic`). Both approved the prediction-only payload/runtime, validated hierarchy authority, ready-or-error conversion, strict inherited backend, no-cluster/rank-one shortcuts, shared exact prediction path, Python parity, and test program. Approval does not cover §8 persistence or implementation correctness. The critic recorded four nonblocking implementation advisories; all are resolved in this document: (1) Rust/native `n_features` accessor, (2) required rescope of `canonical_bytes`, (3) explicit doc-hidden predictor cache diagnostic access, and (4) replacement of stale prediction module persistence docs.

This table describes the Gate 1 implementation impact. It is not permission to implement persistence before Gate 2.

| File | Expected change |
|---|---|
| `crates/fast-hdbscan/src/prediction.rs` | Separate prediction-only shared state from fitted outputs; make runtime/core preparation and exact orchestration operate on it; add ready conversion support and prediction-only resource preflight. |
| `crates/fast-hdbscan/src/estimator.rs` | Add `FittedHDBSCAN::to_predictor`; add `HDBSCANPredictor`, `n_features`, and doc-hidden cache-status APIs; make fitted prediction delegate to shared prediction state. |
| `crates/fast-hdbscan/src/lib.rs` | Re-export the public predictor. |
| `crates/fast-hdbscan/tests/test_prediction.rs` | Predictor behavior, detached lifetime, backend/route, error and parity coverage. |
| `crates/fast-hdbscan/src/prediction.rs` unit tests | Cache readiness, conversion atomicity, upload counts, core-only sizing, and re-scoped `canonical_bytes`. |
| `crates/fast-umap-python/src/hdbscan.rs` | Native predictor class and fitted conversion method, preserving bridge warning/result semantics. |
| `crates/fast-umap-python/src/lib.rs` | Register the native predictor class. |
| `python/fast_umap/_fast_umap.pyi` | Type native conversion/prediction APIs and predictor `n_features`. |
| `python/fast_umap/hdbscan.py` | High-level conversion and detached prediction wrapper sharing fitted input/output semantics. |
| `python/fast_umap/__init__.py` | Export `HDBSCANPredictor`. |
| `python/tests/test_hdbscan_prediction.py` | High-level API parity and detached ownership tests. |
| `README.md` | Document standalone predictor lifetime, conversion cost/readiness, strict backend behavior, and that the predictor API is separate from fitted-estimator persistence. |

The existing core hierarchy is sufficient for Gate 1 prediction. Gate 2 persistence needs the validated portable hierarchy constructor/exporter defined in §8.2; those core additions belong to the later persistence implementation, not to an unrelated fit-state expansion.

## 8. Gate 2 persistence design — HDBSCANPredictor only

### 8.1 Persistence contract and public APIs

Persist only the approved prediction object. The Python API is:

```python
predictor = fitted_hdbscan.to_predictor()
predictor.save(path, overwrite=False)
restored = HDBSCANPredictor.load(path, backend="auto", device_index=0)
```

There is no `HDBSCAN.save`, `HDBSCAN.load`, `FittedHDBSCAN.export_state`, or file representation for fitted outputs in this design. `HDBSCANPredictor.save` exports the already-ready predictor state before touching the filesystem; it performs no prediction, device query, tree construction, or core preparation. `HDBSCANPredictor.load` validates the HDF5 container in Python, then delegates semantic validation and native object construction to the Rust bridge. The restored object is an `HDBSCANPredictor`, not a refitted or partially reconstructed estimator.

The storage-independent Rust type is proposed in `fast-hdbscan` and re-exported from its root; re-export `PredictionHierarchyState` there as well so users can name all public state field types:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct HdbscanPredictorState {
    pub n_features: usize,
    pub min_samples: usize,
    /// Flat row-major finite training coordinates; empty only for no-cluster-tree state.
    pub geometry: Vec<f32>,
    pub hierarchy: PredictionHierarchyState,
    pub prediction_cores: PredictionCoresState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionCoresState {
    NotRequiredNoClusterTree,
    Ready { rank: usize, values: Vec<f64> },
}

impl HDBSCANPredictor {
    pub fn export_state(&self) -> Result<HdbscanPredictorState>;
    pub fn from_state(
        state: HdbscanPredictorState,
        backend: BackendChoice,
    ) -> Result<Self>;
}
```

`HDBSCANPredictor::from_state` is the only public state-import path. All fields are untrusted, including Rust callers' in-memory values; no unchecked public constructor is added. `export_state` uses only predictor-owned prediction state/runtime caches, never `FittedState` or detached fitted views. It first computes checked aggregate export bytes (geometry, hierarchy columns and cores), then clones columns into the returned owned state. Since the Gate 1 predictor is ready, export never does GPU work and never publishes a partially ready state.

No training-backend provenance, fit `HdbscanConfig`, labels, probabilities, condensed tree, linkage, MST, fit neighbors, fit `core_distances`, resolved fit search/graph route, `KnnDiagnostics`, Python constructor options, or device tensors are exported. Runtime backend is selected anew when loading. `min_samples` is stored as the predictor's positive resolved rank, not reconstructed from fit config.

### 8.2 Minimal portable hierarchy and exact validation

The current `PredictionHierarchy` private nodes and point attachments are the prediction-time authority, but are not a stable portable layout. Add a core-layer, backend-free `PredictionHierarchyState` in `fast-hdbscan-core::prediction` with exact integer-indexed columns:

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionHierarchyState {
    pub finite_index: Vec<u32>,
    pub node_ids: Vec<u64>,
    /// Parent node index; `u64::MAX` marks the single root at node index 0.
    pub node_parent_indices: Vec<u64>,
    /// Birth lambda on the parent edge; root stores canonical +0.0.
    pub node_birth_lambdas: Vec<f64>,
    /// Propagated fitted label (`-1` means unmapped/no selected cluster).
    pub node_labels: Vec<i32>,
    /// Propagated selected-anchor maximum lambda.
    pub node_max_lambdas: Vec<f64>,
    /// Point attachment for each compact finite row, in compact row order.
    pub point_node_indices: Vec<u64>,
    pub point_departure_lambdas: Vec<f64>,
}
```

Add `PredictionHierarchy::export_state()` and validated `PredictionHierarchy::from_state(state)` conversions in the core crate. `export_state` preserves float values exactly as `f64` and indices exactly as portable unsigned columns. `from_state` does checked `u64 -> usize` conversions, validates all invariants below before constructing private nodes/attachments, fixes the internal root at index 0, and returns a fully usable `PredictionHierarchy`. It does not rebuild or retain source labels or condensed rows.

The portable hierarchy is deliberately the minimum complete input used by `predict_row`: node IDs, parent links and birth lambdas drive ancestor ascent; propagated labels and max lambdas determine predicted labels/strength; per-point node/departure attachments select that cluster; finite-to-raw IDs map returned connectors. Omitting any of those would change prediction semantics or require keeping the prohibited fit-only source arrays. Do not persist derived tree-build scratch maps, children vectors, `root` (always 0), hash maps, or KD-tree state.

`PredictionHierarchy::from_state` rejects malformed clustered states before exposing an object:

1. **Canonical empty shortcut:** all eight columns empty means no cluster tree. Any mixture of empty/nonempty columns is invalid. The finite mapping and point attachments are omitted because no-cluster prediction always returns connector `-1`; this also handles root-only fitted trees, which current `no_cluster_tree()` intentionally treats as the all-noise shortcut even if fitted labels contain a selected single cluster.
2. **Cluster dimensions:** a nonempty `finite_index` and `node_ids` are required; every node column has exactly `node_ids.len()` entries; point node/lambda columns each have exactly `finite_index.len()` entries. Each count/product and index conversion is checked; counts/connector IDs must fit the existing `i32` search/output indexing limits.
3. **Finite-to-raw mapping:** `finite_index` is strictly ascending, unique by strict order, and each value is `<= i32::MAX` because the exact prediction output casts the chosen raw connector to `i32`. No raw row count is stored: the predictor needs the original IDs themselves, not the fit's raw matrix length.
4. **Node identity and root:** node IDs are unique and fit the host `usize`; `node_ids[0]` is the minimum ID. Exactly node 0 has parent sentinel `u64::MAX`; no other node has the sentinel. All non-root IDs are greater than the root ID, matching `predict_row`'s `node.id > root_id` ascent termination.
5. **Parent graph:** every non-root parent index is in range, not self, and names a node whose ID is strictly less than the child node ID. Strict ID descent plus one parent for every non-root proves a connected acyclic chain to the unique root; no recursive graph walk is needed. Every parent birth lambda is nonnegative and not NaN; positive infinity is retained because the source accepts `+inf` condensed lambdas.
6. **Labels and propagated maxima:** labels are `-1` or nonnegative `i32`; max lambdas are nonnegative and not NaN (`+inf` is valid). The root's label/max pair must be canonical (`label == -1` implies max lambda is `+0.0`). For every child, if its propagated label equals its parent's label, its propagated maximum must be bit-equal to the parent's (the constructor inherited both values); otherwise the child label must be nonnegative, and the distinct label must be anchored at only one node. This validates inheritance/anchor structure without pretending the omitted selected-label source is present. For an anchored node, the stored maximum is authoritative; it cannot be recomputed exactly from finite attachments because the source hierarchy's maximum also includes condensed rows for non-finite outliers, which are intentionally absent from a predictor.
7. **Point attachments:** every point node index is in range. Departure lambdas are nonnegative and not NaN; `+inf` is valid. One entry per finite row is guaranteed by exact column lengths and compact order.

The validator does not claim authenticity or prove that an otherwise valid hierarchy/core was generated by a particular fit. It enforces memory/index safety and the invariants required by prediction while preserving the stored prediction authority. HDF5 compression/checksums are not a cryptographic integrity mechanism.

`HdbscanPredictorState` validation in the facade then requires:

- `n_features > 0` and `min_samples > 0`; checked `n_finite * n_features` and all byte sizes before reconstruction.
- Empty hierarchy iff `prediction_cores == NotRequiredNoClusterTree`; in this case `geometry` is empty, no finite map/nodes/points/cores are stored, and `min_samples` remains a positive inert resolved fit rank. Skip candidate-width validation, exactly as current no-cluster prediction does.
- Nonempty hierarchy iff `prediction_cores == Ready { rank, values }`; `n_finite = hierarchy.finite_index.len()` is in `1..=i32::MAX`; geometry has exactly `n_finite * n_features` finite `f32` values; `candidate_width(min_samples, n_finite)` succeeds before tree construction; `rank == min_samples`; core vector length is `n_finite`; each core is finite and nonnegative via `PredictionHierarchy::validate_prediction_core`.
- If `min_samples == 1`, every core value must be canonical `+0.0`, matching `compute_cores`'s no-search rank-one result. For larger ranks, do not run a second neighbor search during import to recompute cores; validate their domain/rank and trust the validated portable state values, as with the hierarchy authority.

The portable state has no `Uninitialized` core state: serialization is of an already-ready predictor. No-cluster state has no core dataset and rank 0. Clustered state has `Ready`, rank `s`, and all finite nonnegative `f64` values. Export is cache-independent after `to_predictor`; loading never does lazy core initialization.

### 8.3 HDF5 schema v1

Create a predictor-only version-1 schema using the established UMAP container conventions, discriminated by `model_kind="hdbscan_predictor"`. Root attributes have the exact same set and types as `python/fast_umap/_serialization_v1.py`: `format="fast_umap_fitted_model"`, `model_kind="hdbscan_predictor"`, `schema_version=<u4 1`, nonempty `producer_version`, `filter_id=<u4 32026`, `codec="zstd"`, `shuffle="shuffle"`, and `clevel=<u1 1`. The only root group is `state`; reject extra/missing root attributes or groups and unknown schema versions. This is a predictor schema, not the older complete-fitted-model HDBSCAN proposal.

`state` has exactly these scalar attrs:

| Attribute | Type | Clustered predictor | No-cluster shortcut |
|---|---|---|---|
| `n_features` | `<u8` | positive fitted width | positive fitted width |
| `n_finite` | `<u8` | finite mapping length | canonical `0` (mapping omitted and unused) |
| `min_samples` | `<u8` | resolved positive rank `s` | resolved positive but prediction-inert rank |
| `cluster_nodes` | `<u8` | node count, at least 1 | canonical `0` |
| `cores_status` | text | `"ready"` | `"not_required_no_cluster_tree"` |
| `cores_rank` | `<u8` | exactly `min_samples` | canonical `0` |

For each dataset name below, `state` also has an exact bool presence attribute `<name>_present`; the dataset exists iff its flag is true. This follows UMAP's optional-array contract and permits a fixed strict attribute set while making the no-cluster omission explicit.

| Dataset | HDF5 dtype | Shape / role |
|---|---|---|
| `geometry` | `<f4` | Flat row-major, length `n_finite * n_features`; prediction training coordinates. |
| `finite_index` | `<u4` | `(n_finite,)`; compact-to-raw mapping for connector output. |
| `node_ids` | `<u8` | `(cluster_nodes,)`; raw hierarchy node IDs. |
| `node_parent_indices` | `<u8` | `(cluster_nodes,)`; parent node indices, with `u64::MAX` only at root index 0. |
| `node_birth_lambdas` | `<f8` | `(cluster_nodes,)`; parent-edge births; root is canonical `+0.0`. |
| `node_labels` | `<i4` | `(cluster_nodes,)`; propagated existing fit labels. |
| `node_max_lambdas` | `<f8` | `(cluster_nodes,)`; propagated maximum lambda values. |
| `point_node_indices` | `<u8` | `(n_finite,)`; node attachment per compact finite row. |
| `point_departure_lambdas` | `<f8` | `(n_finite,)`; attachment departure lambda per compact row. |
| `prediction_cores` | `<f8` | `(n_finite,)`; ready prediction-only cores, not fit `core_distances`. |

All datasets are one-dimensional, little-endian, contiguous value arrays; geometry is flattened row-major and never stored as a 2-D empty-axis shape. Clustered state has all ten datasets present and every length positive. No-cluster state has all ten presence attrs false and no datasets at all—no geometry, finite map, hierarchy rows, or cores. It carries only scalar prediction dimensions/status, the minimum needed to validate query feature count and restore shortcut behavior. There are no empty compressed datasets.

Each present dataset uses the UMAP's Blosc2 zstd/shuffle filter (`filter_id=32026`, codec id 5, shuffle id 1, compression level 1) and row chunks `min(length, 32_768)` (at least one). HDF5 validation rejects virtual/external-storage datasets, external/soft links, hard-link aliases/cycles, wrong filter options, wrong dtype/rank, noncanonical scalar/enum/presence metadata, and any exact-set mismatch. Python checks shape relationships from scalar metadata before reading payload; Rust repeats semantic validation before constructing a predictor.

No `config`, `provenance`, training backend, requested/resolved fit search, `raw_n`, labels, condensed tree, or arbitrary JSON metadata group is written. Save/load of a predictor with no cluster tree uses the same schema and status rules, not a fake one-row hierarchy. An HDBSCAN predictor loader given a UMAP file (or vice versa) fails with a `ValueError` naming the found model kind and the expected class.

### 8.4 Backend selection, import readiness and tree rebuilding

`HDBSCANPredictor::from_state(state, backend)` validates all checked state dimensions, columns, scalar domains and allocation arithmetic before resolving a runtime backend or building caches. It accepts the existing `BackendChoice`: `Auto` uses `fast_cubecl_common::selected_device(Auto)`'s concrete CPU/CUDA choice; explicit CPU/CUDA is strict and preserves the requested CUDA device index. Store only the resolved concrete backend in `PredictionRuntime`; do not persist either fit-time backend or load-time requested `Auto`. No runtime fallback or backend migration occurs after load.

For a clustered state, after full semantic validation, rebuild `KdTree::build(&geometry, n_finite, n_features)` deterministically on the host and initialize runtime `CoreStatus::Ready { rank: s, values }` from the validated imported f64 core vector. The file never stores KD tree arrays or device tensors. This yields a ready returned predictor: query-time core search and tree build do not repeat, although each prediction still performs the current per-call device upload, bounded query batches and exact KD queries. For no-cluster state, build no tree and initialize `NotRequired(NoClusterTree)`; prediction validates input then takes the current no-device shortcut.

The load-time backend is resolved for both states so an explicit unavailable backend fails consistently and a restored object's concrete runtime setting is well-defined. This resolution creates/selects a runtime device but performs no model upload. No-cluster queries still do not resolve/discover devices at prediction time or execute device work. If the selected backend is available but later query sizing/allocation fails, `approximate_predict` returns the same strict error as it does for a converted predictor.

`KdTree::build` uses the same canonical total ordering on feature value then row ID and produces the tree consumed by exact bounds-pruned traversal. Rebuild rather than persist internal tree arrays: the tree is derived solely from exact finite f32 geometry; its private layout/build scratch is not portable state. Check all row/feature/node/bounds products and host byte sizes before allocating; retain `KdTree::build`'s checked allocation failures as a second guard. Query device allocation limits are preflighted later using the real query dimensions; load does not claim device memory availability or upload geometry.

Resource accounting remains scoped to the prediction object: retained state is geometry + validated hierarchy + f64 cores, with the rebuilt host KD tree reported separately. Save holds the ready native predictor while making an owned Rust export snapshot, native NumPy arrays and HDF5 chunk/compression buffers; it accounts checked bytes before cloning but does no device work. Load may transiently hold decoded NumPy arrays, Rust-owned portable state and reconstructed hierarchy/tree; validate every declared count/product and byte total against checked `usize`/`isize`/`np.intp` limits before reading HDF5 payloads or allocating additional Rust buffers. Use fallible reservations where the existing state/tree builders provide them, and return typed errors on overflow/allocation failure. The importer may have unavoidable overlapping Python/Rust copies, so document these peaks and avoid extra dtype/layout copies; it must never include or account for full fitted-model arrays. These checks guard representability and requested allocation sizes, not available RAM or GPU memory.

### 8.5 Native bridge and Python persistence wrapper

In `crates/fast-umap-python/src/hdbscan.rs`:

- Add `HDBSCANPredictor.export_state()` returning `(arrays, metadata)` from Rust `export_state`, with GIL detached only around Rust preparation/cloning. Arrays are independent NumPy snapshots and use the exact schema dataset keys/dtypes; no-cluster keys map to `None`. Metadata contains only `n_features`, `n_finite`, `min_samples`, `cluster_nodes`, `cores_status`, `cores_rank`.
- Add registered native `load_hdbscan_predictor(arrays, metadata, backend="auto", device_index=0)`. Extract exact NumPy dtypes; reject missing/extra keys, rank/dimension disagreements, invalid scalar spellings/types and unsupported backend spellings before passing owned values to Rust. Call `HDBSCANPredictor::from_state` under `py.detach`, map errors using existing `map_hdbscan_error`, and return native `HDBSCANPredictor`.
- Native API feature-width check uses the Gate 1 `inner.n_features()` accessor. `HDBSCANPredictor` stays frozen and owns only the Rust predictor. Expose the same prediction tuple, dtypes, bool validation, warning text and stacklevel as fitted prediction; do not return fit attributes.
- Register native class/function in `crates/fast-umap-python/src/lib.rs`; update `python/fast_umap/_fast_umap.pyi` with `export_state`, `n_features`, predictor prediction overloads and loader.

In `python/fast_umap/hdbscan.py`, implement high-level `HDBSCANPredictor` with only native predictor and `n_features_in_`, populated from the native predictor's owned read-only `n_features` accessor on both conversion and load. It implements `approximate_predict` by sharing the existing fitted wrapper's input validation/output conversion path. Its `save(path, overwrite=False)` lazily imports the HDBSCAN predictor serializer, obtains native `export_state()` first, then asks the Python file layer to publish. Its `@classmethod load(path, *, backend="auto", device_index=0)` returns only `HDBSCANPredictor`, never `HDBSCAN`.

Create `python/fast_umap/_hdbscan_predictor_serialization_v1.py` for HDBSCAN predictor schema-specific reads/writes. Keep h5py/hdf5plugin optional and lazy; the existing `serialization` extra is reused with no dependency changes. `save_model` accepts native arrays/metadata and `overwrite`; `load_model` validates the file and calls `_fast_umap.load_hdbscan_predictor`. Backend spelling and `device_index` (reject bool, non-int and negative values) are validated before file payload import, matching UMAP `load_model` conventions.

### 8.6 Shared atomic HDF5 implementation; no UMAP drift

Do not copy UMAP's security/durability-sensitive file I/O into a second module. Extract the model-agnostic HDF5 helpers and atomic publisher from `python/fast_umap/_serialization_v1.py` into `python/fast_umap/_serialization_common.py`, then have both UMAP schema v1 and HDBSCAN predictor schema v1 call those same implementations. Common helpers cover lazy dependency loading with model-specific messages, exact scalar/text attribute readers, Blosc2 filter verification, link graph validation, dataset creation with the shared chunk/filter settings, standard root attrs/model-kind verification, and atomic publication. Schema-specific group/attribute/dtype/semantic validation remains in each versioned serializer.

The common atomic publisher must preserve UMAP's actual sequence: require an existing parent directory; create a temporary sibling via `mkstemp`; close the raw descriptor; write and close HDF5; fsync the completed temporary file; publish with `os.link(temp, destination)` for no-overwrite or `os.replace` for overwrite; fsync the parent directory; on post-publication directory-fsync failure raise the existing `DurabilityUncertainError` explaining that the destination is committed; clean up only the temporary link/file. Pre-commit failure leaves any existing destination intact and removes the temp; no-clobber races preserve the concurrent destination. HDBSCAN state export and all semantic checks that can run before the file layer occur before creating a temporary file.

Keep `_serialization_v1.py` as UMAP's versioned schema module and re-export `DurabilityUncertainError` there so existing imports remain valid. Keep its imported `os` module binding so existing UMAP tests that monkeypatch `_serialization_v1.os.link/fsync` continue to exercise the shared `os` module used by the common publisher; alternatively update those test seams in the same refactor, but do not weaken the existing atomicity tests. Move implementation, not behavior: `python/tests/test_serialization.py` remains the regression oracle for existing UMAP bytes/schema/link validation, no-clobber/overwrite, temp cleanup, and durability uncertainty. Add shared publisher tests once, avoid reimplementing atomic publication per model.

### 8.7 Persistence tests

#### Rust state/export/import tests

Extend `crates/fast-hdbscan/tests/test_prediction.rs` or add a focused `test_predictor_state.rs`:

- Round-trip `HdbscanPredictorState -> HDBSCANPredictor::from_state(Cpu) -> export_state`; compare every portable scalar/column exactly and compare predictions with the original predictor and the independent original-hierarchy oracle. Cover same-backend exact results, CPU/CUDA tolerance under existing CUDA gating, mixed nonfinite training rows/raw connector mapping, duplicate/tied coordinates, and several query batches.
- Confirm predictor export after conversion is the same whether fitted prediction ran first; after conversion and fitted-object drop, `prediction_cache_status()` works, `canonical_bytes` measures only prediction payload, imported state excludes fit outputs, and imported predictor predictions match.
- Round-trip a no-cluster/root-only predictor whose original finite-row count violates search width (`2*s > original_n_finite`); assert schema state is minimal (all predictor arrays absent, no KD tree/core) and all shortcut outputs remain exactly `-1`, `0.0`, `-1` with `no_cluster_tree=true`.
- Round-trip `s == 1`, including exact +0.0 core requirement; verify reconstructed KD tree/caches and exact predictions.
- Malformed portable hierarchy cases each fail without panic: mismatched/overflowing lengths, nonascending or out-of-range finite IDs, duplicate/overflowing node IDs, multiple/missing roots, bad parent index/self/cross-root edge, parent ID not less than child, invalid labels/duplicate anchors/inherited-label max mismatch, negative/NaN lambdas, invalid point nodes, no-tree partially populated state, and index values not representable on target. Positive-infinity hierarchy lambdas/maxima remain accepted where permitted.
- Malformed facade state rejects zero dimensions/rank, wrong geometry product/nonfinite geometry, clustered/no-tree status mismatch, bad core rank/count/NaN/infinity/negative values, rank-one nonzero cores, and candidate width before KD tree allocation. Test arithmetic overflow and deterministic-tree host allocation failures where injectable; never inject memory exhaustion process-wide.
- Backend import resolves Auto to concrete runtime, honors explicit device index, fails strictly for unavailable explicit CUDA; no runtime fallback. Import does not upload geometry.

#### Python schema and API tests

Create `python/tests/test_hdbscan_predictor_serialization.py` using the existing `tmp_path` and UMAP patterns:

- Clustered predictor save/load on CPU; compare prediction labels/connectors exactly and f64 strengths bitwise for same backend; test loaded predictor after deleting the source fitted model. Confirm no estimator attributes/config/fit arrays are exposed or serialized.
- Cover no-cluster and root-only minimal files, `s == 1`, mixed nonfinite training mapping, multiple features, and boundary `min_samples`/candidate width. `HDBSCANPredictor.load` returns the predictor type and follows warning/dtype/stacklevel behavior.
- Independent h5py reader verifies root attrs, kind/version, exact groups/attrs, dtype and shape tables, Blosc2 filter, and that all no-cluster predictor array presence flags are false. No zero-extent datasets. UMAP loader rejects predictor model kind and predictor loader rejects a UMAP file with a precise model-kind message.
- Reject corrupted schema before native import: unknown/missing/extra root/group attr/dataset, wrong scalar dtype/rank/text enum, noncanonical core status/rank/presence, wrong dataset dtype/rank/length, shape/metadata mismatch, wrong filter, virtual/external storage, external/soft links and hard-link alias/cycles. Corrupted payload values that pass container checks are rejected by native state validation; monkeypatch asserts Python does not call Rust when schema validation itself fails.
- Validate `backend` and `device_index` first, including bool/non-int/negative index; forward device index to native load; explicit unavailable backend fails strictly. CUDA cross-backend tests use the existing `FAST_UMAP_REQUIRE_CUDA=1` convention.
- Assert optional dependency errors are lazy and name `fast-umap[serialization]`; test no-clobber, overwrite, race behavior, temp cleanup, precommit failures and postcommit `DurabilityUncertainError` using the shared publisher. Keep all existing `python/tests/test_serialization.py` tests unchanged and passing through the refactor.

#### Gate requirements

- Gate 1 approval is recorded above and applies only to the predictor API/design.
- Gate 2 must approve this persistence design (portable hierarchy representation, no-tree schema, Rust/Python state contract, strict load backend, shared atomic I/O, file sequence and tests) before any code implementation in §§8.1–8.8.
- After implementation, the parent agent runs focused Rust/Python tests, existing UMAP serialization regression tests and builds needed for native/Python packaging. Gate 2 is the final architect/critic design review; post-implementation verification is evidence, not another design-approval gate. This plan-authoring task runs none of those checks and does not claim implementation correctness.

### 8.8 Implementation file sequence after Gate 2

Keep the approved predictor/API implementation separate from persistence work. Since Gate 1 is approved, its implementation can proceed under the parent's execution direction; complete it and verify its tests before persistence code is layered on it. Persistence itself proceeds only after Gate 2 approval, in this order:

1. **Core portable hierarchy + facade state:** `crates/fast-hdbscan-core/src/prediction.rs` (portable columns, checked export/constructor); `crates/fast-hdbscan-core/src/lib.rs` if needed for re-export; `crates/fast-hdbscan/src/prediction.rs` (predictor state, readiness/export/import, deterministic tree rebuild, diagnostics canonical-byte rescope and stale module docs); `crates/fast-hdbscan/src/estimator.rs` / `lib.rs` (public `HdbscanPredictorState`, `PredictionCoresState`, export/import APIs and `n_features`/hidden diagnostic accessors); focused Rust state/prediction tests. Verify before moving on.
2. **Native bridge state conversion:** `crates/fast-umap-python/src/hdbscan.rs`, `crates/fast-umap-python/src/lib.rs`, and `python/fast_umap/_fast_umap.pyi`; transfer arrays/metadata and test native import/export, shape/error mapping, backend selection, and predictor lifetime. Do not add h5py code to Rust.
3. **Shared Python file layer and schemas:** add `python/fast_umap/_serialization_common.py`; refactor `python/fast_umap/_serialization_v1.py` to call it without changing UMAP behavior; add `python/fast_umap/_hdbscan_predictor_serialization_v1.py`; add Python schema/shared-publisher tests and verify all existing UMAP serialization tests before wrapper integration.
4. **Public Python API/docs:** update `python/fast_umap/hdbscan.py` and `python/fast_umap/__init__.py` with predictor save/load and type export; add `python/tests/test_hdbscan_predictor_serialization.py`; update README predictor and persistence sections plus any directly affected architecture guidance. Run focused Python tests after integration.
5. Parent verifies the union of changes once: Rust hierarchy/predictor/state tests, existing Rust prediction tests, Python predictor serialization and existing UMAP serialization suites, and native build/import checks. Do not run redundant project-wide gates between slices; Gate 2 is the pre-implementation architect/critic approval.

## 9. Decision log

| ID | Decision | Rationale / review focus |
|---|---|---|
| D1 | Add a distinct `HDBSCANPredictor`; do not make a “prediction-only fitted model” or load a `FittedHDBSCAN`. | Current fitted state intentionally includes full fit outputs. A separate type makes the no-fit-only-state promise visible and avoids exposing fit APIs on a detached predictor. Review whether a separate public type is the right user-facing boundary. |
| D2 | Share one immutable prediction-only payload and one prediction runtime through `Arc`; predictor owns these, never `FittedState`. | Geometry/hierarchy/runtime caches are already immutable/shared or synchronized. Sharing avoids duplicating large geometry, tree and cores, lets a predictor outlive the fitted estimator, and preserves fitted/predictor math identity. Runtime helper signatures must be refactored so no hidden `FittedState` reference remains. |
| D3 | The predictor's in-memory hierarchy is the existing validated `PredictionHierarchy`, built from final condensed tree + raw labels + finite mapping; do not retain those source fit outputs. | The validated hierarchy is the exact connector/prediction authority. Persistence exports only its minimal validated integer-indexed columns and never serializes source fit outputs or trusts private fields. |
| D4 | Keep prediction cores as their own validated `f64` rank-`s` vector; never substitute fitted `core_distances: f32`. | The current prediction implementation computes inclusive-self exact KD cores and converts squared `f32` values to Euclidean `f64`; these are a separate cache from fit output core distances. |
| D5 | Conversion is ready-or-error; prepare/cache cores before returning for cluster trees. | A standalone predictor must not depend on a fitted estimator to finish lazy initialization. Caches publish only complete valid values; a failed operation returns no predictor. A completed tree may survive core failure per the established retry contract. |
| D6 | Preserve no-cluster-tree shortcut before candidate-width/device work; `s == 1` gets zero cores without conversion-time KD work. | The current no-tree path intentionally supports widths that would be invalid for an actual search. `compute_cores` already has the rank-one all-zero special case. Avoid expensive or irrelevant conversion work. |
| D7 | Do not add a backend override; inherit the concrete fit-time runtime backend and keep strict selection/no fallback. | Current fitted prediction is bound to `PredictionRuntime::backend()`, whose `Target` resolution rejects backend changes. A second backend policy would change semantics and multiply tests. |
| D8 | Keep one exact KD implementation for fitted and standalone prediction. | Exact neighbor order, `2*s` external candidates, batched resource planning and f64 hierarchy math are correctness contracts already tested against independent oracles. The predictor is an ownership/API change, not an algorithm change. |
| D9 | Preflight conversion's real core-only lifetime rather than reusing the query-plus-core planner with a fabricated query. | Existing `plan_prediction` includes query tensors/output and both phases. Charging a fake query can spuriously reject conversion and misstate peak memory; factor shared checked sizing instead. |
| D10 | Python predictor stores only native predictor + feature count and reuses fitted wrapper semantics. | Avoid keeping a reference to the fitted Python estimator or duplicating input/warning/dtype behavior. |
| D11 | Gate 1 is approved; this document drafts prediction-only persistence, and Gate 2 approval is required before its implementation. | The new object boundary is prediction-only, not a fitted snapshot. Gate 1 did not approve storage, so persistence receives its own architect/critic review without weakening or expanding the predictor boundary. |
| D12 | Do not require a user commit before design or implementation. | Git commit state is not a technical precondition; use and review the actual checked-out source. |
| D13 | Keep this plan-revision task limited to this plan file; no product code, test, build, format, or gate execution. | The assigned deliverable is the revised design for parent-managed review, not implementation. |
| D14 | Persist an owned `HdbscanPredictorState` with geometry, `min_samples`, validated hierarchy columns, and ready `f64` prediction cores only. | This is exactly the approved prediction payload; it excludes all fit-only outputs, device tensors, and training provenance. State import is untrusted and validates before object construction. |
| D15 | Encode hierarchy with finite raw IDs, indexed cluster nodes/parents/birth lambdas/propagated labels/maxima, and per-point attachment/departure columns. | These are precisely the values consumed by ancestor ascent, label/strength computation, and connector translation. Root index 0 and `u64::MAX` parent sentinel are canonical; source labels/condensed rows are neither needed nor retained. |
| D16 | Represent no-cluster-tree as the all-empty hierarchy, no geometry/map/core arrays, `NotRequiredNoClusterTree`, and zero core rank. | Prediction validates feature width then returns the existing exact shortcut without candidate width, geometry, hierarchy mapping, tree, or device work. `n_features` and positive resolved `min_samples` are sufficient scalar state; all-noise/root-only models share this contract. |
| D17 | Persist the ready rank-`s` f64 prediction cores and rebuild the deterministic host KD tree on load. | Cores are prediction-specific canonical values, not fit `f32` cores; recomputing them would redo cold work and can vary by backend. The KD tree is an exact deterministic derivative of retained geometry and its private layout is not portable. |
| D18 | Select a strict concrete runtime backend at load, never serialize a backend or fall back. | The loader's explicit backend/device index is a runtime choice independent of training provenance. Resolve even for no-cluster state so explicit unavailable requests fail consistently; the shortcut itself performs no device work. |
| D19 | Use a strict HDF5 schema v1 with exact field/dtype/presence sets and no migrations. | Explicit empty-state and ready-state encodings make rank/status consistency auditable; rejection of unknown versions/fields avoids silently misreading a changed predictor authority. |
| D20 | Both UMAP and HDBSCAN schema modules use one shared atomic HDF5 publisher/security helper. | Reuse the existing same-directory temp, fsync, no-clobber hard link/overwrite replace, directory fsync, durability-uncertain error, link validation and Blosc filter logic rather than allowing two serializers to drift. Preserve existing UMAP tests and behavior. |
| D21 | Gate 2 approves the complete design before persistence implementation; implement Rust state, native bridge, shared I/O/schema, then public Python API. | Separating the approved predictor from the new persistence boundary limits coupling and gives each layer a focused verification point. Parent owns the final combined checks and review. |
| D22 | Resolve Gate 1 advisories with `n_features`, re-scoped `canonical_bytes`, a doc-hidden predictor diagnostic accessor, and updated module docs. | Native feature validation must use owned predictor metadata; diagnostics must stop counting fit-only state; tests need supported post-drop cache inspection; the old full-fitted-state persistence note contradicts the new boundary. |
