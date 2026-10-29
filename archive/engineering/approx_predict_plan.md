# Fitted HDBSCAN `approximate_predict` implementation plan

## 1. Objective and scope

Implement the algorithm behind **original Python `hdbscan.approximate_predict`** as a method on fitted models:

- Rust: `fast_hdbscan::FittedHDBSCAN::approximate_predict`.
- Python: `fast_umap.HDBSCAN.approximate_predict` after `.fit(...)`.
- Native Python bridge: `_fast_umap.FittedHDBSCAN.approximate_predict`.

Prediction assigns each query independently to the **existing fitted clustering**. It must not refit, rebuild the MST, change training labels, discover new clusters, or allow queries to become neighbors of other queries. “Approximate” describes fixed-hierarchy assignment rather than approximate nearest-neighbor search.

This is a planning artifact, not execution approval. No product code has been changed for this plan.

### Included

- Euclidean feature-space prediction, matching this repository's supported fitting metric.
- Labels, membership strengths, and optional connecting training-point indices.
- CPU and CUDA execution through a generalized shared exact KD traversal, always used for prediction regardless of fitting route.
- Owned fit-time geometry, prediction-specific core distances, hierarchy metadata, input validation, finite-training-row remapping, and lifecycle tests.
- Fit-resource accounting and documentation of retained memory and cold/warm prediction cost.
- A private serialization-ready backend-free fitted-state contract and future save-readiness seam, designed now without implementing persistence.

### Excluded

- `approximate_predict_scores`, soft membership vectors, exemplars, branch prediction, online clustering, arbitrary metrics, sparse/precomputed inputs, and a standalone public Python prediction function.
- Actual file save/load, persistence bindings, and round-trip implementation. Their state contract is specified in section 6; no public persistence stubs ship in this feature.
- ANN prediction or production brute-force fallback. Generalizing the existing training-only KD operation to arbitrary queries is explicitly in scope.
- New runtime dependencies on `hdbscan`, scikit-learn, or the separate `fast_hdbscan` Python package. Original `hdbscan` is a development oracle only.

## 2. Reference implementation and research evidence

### Authoritative installed reference

Inspected `.venv/lib/python3.14/site-packages/hdbscan/`, distribution **hdbscan 0.8.44**. Do not substitute the installed `fast_hdbscan` package or this repository's existing fast-hdbscan fitting fixtures for prediction semantics.

`prediction.py` SHA-256 at inspection:

```text
e43d6e95dfd42e825c66272467094fb4dcf403a904da9861e51892387c0f867c
```

Record the installed package version and source hash in generated prediction fixture provenance. The development dependency currently permits `hdbscan>=0.8.44`; a future environment can differ from this inspected source.

| Reference                                                | Relevant behavior                                                                                                                                                   |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prediction.py:166–218`, `PredictionData.__init__`       | Converts training coordinates to float64; builds a spatial tree; computes inclusive-self prediction core distances; builds selected-cluster maps and parent maxima. |
| `prediction.py:221–268`, `_find_neighbor_and_lambda`     | Query core rank, mutual-reachability minimum, first-minimum tie policy, finite maximum-double lambda for zero distance.                                             |
| `prediction.py:330–413`, `_find_cluster_and_probability` | Connect point to hierarchy, walk ancestors, map to a fitted label, and compute strength.                                                                            |
| `prediction.py:416–520`, `approximate_predict`           | `2 * min_samples` query width, optional connectors, empty-cluster-tree noise shortcut.                                                                              |
| `plots.py:235–245`, `CondensedTree._select_clusters`     | Reconstructs selected segments from actual fitted labels and point-parent rows; it does not simply rerun EOM selection.                                             |
| `hdbscan_.py:1303–1327`, `generate_prediction_data`      | Resolves `min_samples or min_cluster_size` and creates `PredictionData`.                                                                                            |

### Read-only probes already executed

These establish reference behavior; they are **not** tests of a new implementation:

1. Inclusive-self core rank differs from fitting's nonself rank. For training coordinates `[0, 1, 3, 10]` and `s=2`, prediction cores are `[1, 1, 2, 7]`; second-nonself cores are `[3, 2, 3, 9]`.
2. On two separated one-dimensional groups, ordinary prediction returns cluster labels for nearby queries and noise for a distant query. A noise prediction can still have a valid connecting point.
3. The empty-cluster-tree branch returns noise, zero strengths, connector `-1`, and a warning. A root-only selected cluster also takes this branch, even when training labels contain cluster `0`.
4. Duplicate-only groups can have maximum lambda `+inf`. Exact duplicate queries returned strength **0**, not 1: zero MRD becomes `f64::MAX`, then `f64::MAX / inf == 0`.
5. Ordinary nonempty-tree prediction rejects empty queries and NaN queries. Original prediction-data generation rejects NaN training coordinates and fails when its requested core-neighbor width exceeds the training count.
6. A test-only same-hierarchy adapter using `CondensedTree(raw_tree, labels)`, `PredictionData`, and `types.SimpleNamespace` reproduced the reference estimator's prediction outputs exactly on the separated-groups probe. This provides a way to isolate prediction from fitting differences.

## 3. Current repository integration points

### Missing fitted state

`crates/fast-hdbscan/src/estimator.rs` currently:

- Validates and compacts finite rows in `HDBSCAN::fit_with_source` around lines 918–920 and 967–972.
- Consumes compact geometry into backend tensors around lines 991 and 1210; geometry and host/device KD trees are fit-local.
- Remaps public trees and scatters labels around lines 1274–1285.
- Constructs `FittedHDBSCAN` around lines 1300–1316.
- Defines fitted state around lines 1323–1343 and existing fitted methods around lines 1346–1410.

The fitted model has labels, probabilities, hierarchy, neighbors, fitting core distances, backend, feature count, and `finite_index`; it **does not retain training coordinates or prediction metadata**.

### Existing conventions to reuse

- `fast-hdbscan-core` owns backend-free hierarchy calculations; add prediction hierarchy logic there.
- `fast-hdbscan` owns device selection, resource validation, neighbor dispatch, and estimator orchestration.
- `fast-cubecl-common::neighbors::exact_neighbors(query, reference, k, exclude_self, query_offset, squared)` provides an independent brute-force **test oracle only** for prediction; do not change its semantics or use it as a production prediction fallback.
- `HdbscanOps::exact_neighbors_kd` currently accepts only training data/tree/k and excludes self. Generalize its underlying traversal into query/reference search with explicit self exclusion and query offset; retain the fitting entry point as a wrapper over that same traversal, preserving fitting arithmetic, ordering, launch policy, and performance behavior. Never concatenate queries into the training tree.
- Public trees and labels use original/raw row IDs; fitted neighbors and core distances use compact finite-row IDs.
- The Python wrapper publishes `_fitted_state` only after all fitting/conversion succeeds. Preserve this atomic publication.
- `crates/fast-umap-python/src/hdbscan.rs` already demonstrates detached native execution and HDBSCAN error conversion in `dbscan_clustering`.
- `python/fast_umap/hdbscan.py::_input_array` provides the current real-numeric/float32 representability policy.
- `python/fast_umap/_fast_umap.pyi` must describe the new native method.

## 4. Explicit public contract

### Python method

```python
model = HDBSCAN(backend="cpu", min_samples=5).fit(X_train)
labels, strengths = model.approximate_predict(X_new)
labels, strengths, connecting_points = model.approximate_predict(
    X_new, return_connecting_points=True
)
```

Proposed signature, matching the reference option's positional availability:

```python
def approximate_predict(
    self,
    X: ArrayLike,
    return_connecting_points: bool = False,
) -> (
    tuple[NDArray[np.intp], NDArray[np.float64]]
    | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
): ...
```

Add `Literal[False]`/`Literal[True]` overloads to the Python/native typing surfaces where appropriate, plus a general-bool overload so dynamic flags remain supported.

Decisions:

- Every newly fitted model is prediction-capable. **Do not add `prediction_data=True` or a second public generation lifecycle.** Retain the required geometry automatically and lazily compute the expensive prediction core-distance cache.
- Labels and connector arrays use `np.intp`, following this package's existing index convention rather than original hdbscan's int32 output.
- Strengths always use float64, including the no-cluster branch. Do not reproduce the reference's branch-dependent float32/float64 dtype.
- Outputs are newly owned arrays, in query order. Modifying them cannot mutate fitted state or later predictions.
- The method uses immutable fit/load-time options and the concrete runtime backend chosen by fit or future load, not subsequently modified Python constructor attributes or serialized original backend provenance.
- Predictions always refer to existing fitted label IDs. Never renumber clusters independently during prediction.
- A connector is the selected **mutual-reachability** neighbor, not necessarily the nearest Euclidean point despite the reference docstring's shorthand.
- A connector is in original training-row space. Noise after a real neighbor search still returns its valid connector. Only the no-cluster-tree shortcut returns `-1` for every connector.

### Rust method and result

Add and export a concrete result type, preferably from a facade `prediction` module:

```rust
pub struct PredictionResult {
    pub labels: Vec<i32>,
    pub probabilities: Vec<f64>,
    pub connecting_points: Vec<i32>,
    /// Reference empty-cluster-tree shortcut; lets the Python bridge emit its warning.
    pub no_cluster_tree: bool,
}

impl FittedHDBSCAN {
    pub fn approximate_predict(&self, input: &[Vec<f32>]) -> Result<PredictionResult>;
}
```

Rust always returns connectors; the Python return flag controls tuple shape rather than numerical behavior. The internal status flag avoids inferring the warning condition from all-noise labels, which is not equivalent.

### Input/error policy

Validate at both Python and native boundaries:

1. The Python object must be fitted; otherwise use the existing fitted-state `ValueError` convention.
2. Require a nonempty real numeric 2-D matrix with at least one feature and exactly the fitted feature width. Reject ragged Rust rows.
3. Reject complex, nonnumeric, NaN, and infinite query features. Unlike training, prediction must not silently compact/drop query rows.
4. Reject finite Python values outside float32 range before conversion. Accept ordinary float32/float64 and noncontiguous numeric input through the current conversion pattern.
5. Require a boolean return flag; explicitly reject unrelated truthy objects.
6. Check element-count multiplication, `2*s`, row/index conversions, tensor pitch, allocation sizes, and scratch planning before dispatch. Preserve fit's `raw_n <= i32::MAX` invariant and check raw connector conversions before writing `Vec<i32>`; NumPy's wider `np.intp` does not remove native index limits.
7. In a model with nonempty cluster-tree edges, require `2*s <= n_finite` **before initializing the core cache or uploading/searching geometry**. Do not clamp width or consume padded reference-count sentinel IDs. Return a clear invalid-input error for unsupported width, as the reference tree query does.
8. Validate query shape and finiteness **before** the no-cluster shortcut. Then the no-cluster shortcut needs no neighbor-width check or core-cache construction.
9. Map invalid input to `ValueError`, arithmetic sizing overflow to `OverflowError`, and backend/numerical failures to `RuntimeError`, following existing mapping. Never convert failures into noise or retry CUDA work on CPU.

Deliberate wrapper normalization: consistently reject empty/nonfinite queries even on the no-cluster branch; the original function can bypass its spatial-tree validation there. Correctly shaped empty-output support is not part of this implementation.

## 5. Algorithm to port exactly

Let `s = fitted_config.resolved_min_samples()`, `n = n_finite`, and `k = 2*s`. Keep this value separate from any fitting graph-width cap.

### 5.1 Build immutable hierarchy metadata

Derive metadata from the **final fitted condensed tree and labels**, after any persistence simplification and raw-row remapping:

- Point attachment: for each compact training point `i`, locate the raw condensed-tree point row whose child is `finite_index[i]`; cache its parent and departure lambda.
- Cluster edges: rows with `child_size > 1`; cache child-cluster → `(parent, birth_lambda)` and parent-to-child adjacency.
- Root: minimum parent over cluster edges, when such edges exist. Preserve an explicit `no_cluster_tree` state rather than reducing an empty array.
- Selected cluster anchors: reproduce `CondensedTree._select_clusters`' essential rule by grouping point-row parents by actual nonnegative fitted label and taking the minimum parent for each label. Ignore noise. Do not rerun EOM/leaf extraction, and do not identify anchors from training probabilities.
- Store anchor → actual fitted label. For normal dense labelings, this matches original hdbscan's sorted-anchor enumeration. If the repository's fitting path leaves label gaps, retain its actual IDs rather than silently shifting them; include a synthetic fixture for this contract.
- For each parent, compute maximum lambda over **all** its outgoing condensed rows, including cluster children, not only point children.
- Propagate each selected anchor's label and its maximum lambda to every descendant cluster. Unmapped clusters remain noise.

Preserve the reference's label-derived anchor quirk deliberately: when a selected cluster has no direct point rows and immediately splits into child clusters, the minimum observed point-parent can be a child segment rather than the originally selected ancestor. Do not silently replace it with a selection-based ancestor. A synthetic four-point probe with all labels `0`, root `4`, and child clusters `5`/`6` reconstructed anchor `5` and mapped only cluster `5` in original `PredictionData`. Add a fixture and document the resulting possible unmapped sibling. This is reference behavior, not a claim that the reconstruction recovers every original selection correctly.

Use linear passes and iterative tree traversals with indexed arrays/compact tables. No repeated whole-tree mask scans, recursion proportional to hierarchy depth, dense `raw_n × cluster_count` tables, or per-query topology rebuilding.

Do not reuse `labels::membership_strengths`: it computes fitted training strengths with point-only maxima and special handling for nonfinite lambdas, which is not the prediction formula.

All hierarchy IDs in these metadata can remain raw IDs. Only point attachments need compact-row indexing. This avoids a second inverse-remapping implementation; nonfinite training rows never enter the query reference geometry.

### 5.2 Compute prediction-specific training core distances

Original `PredictionData` computes:

```text
prediction_core[j] = distance of neighbor at index s-1
                    in training-to-training k=s query INCLUDING self
```

This differs from the estimator's fitting `core_distances`, whose rank excludes self. It can also differ from cores obtained through an ANN fitting graph. **Do not reuse the existing public fitting cores.**

- Query compact training coordinates against themselves with `exclude_self=false`, `k=s`, squared mode.
- Read each row's last squared distance and convert it to Euclidean distance in f64.
- `s=1` can produce an all-zero core vector without a search.
- Keep this private, separate from public fitting cores. Building it must not mutate `core_distances()` or exported Python arrays.
- Execute training-core queries through the shared KD traversal in bounded batches with explicit compact query offsets and self inclusion. Batch query counts need not equal reference count.

### 5.3 Search external queries

For each bounded query batch, query the original compact training coordinates:

```text
exact_neighbors_kd_query(query, training, training_tree, k=2*s,
                         exclude_self=false, query_offset=0)
```

Do not exclude a training coordinate merely because a query has the same values or position in a batch. Never add queries to the reference set.

The shared operation orders candidates by `(squared_distance, compact_point_id)`. Keep that order after converting distances. Compact IDs preserve raw-row ordering because `finite_index` is ascending.

Prune a KD node only when its lower bound safely excludes every competitive `(distance², reference ID)` candidate. In particular, an equal-distance bound must not discard a lower-ID tie; retain equal-bound nodes unless a proven ID bound establishes safety. Preserve existing scalar f32 feature-sum order, fitting distance bits, padded layout, and launch routing. The fitting self-query wrapper passes explicit self exclusion and compact query offset into the shared traversal; value equality never implies self identity.

Preserve CPU C16 leaf-SoA/leaf-block fitting queries and CUDA Phase 7 F2 flat-plane routing at 16–32 features when plane operations/synchronization/resources permit (two planes per cube, no row-count gate); preserve the other CUDA per-query route. These are pre-launch choices, never error fallback. External query packing/order is independent of reference leaf ordering: do not index arbitrary queries by training leaf offsets. Keep reference packs/tree strictly in compact training space, track any query permutation separately, and restore original query row order. Fusion output dimensions and dispatch bounds derive from query rows, never reference rows. Preserve existing fitting packing/resource helpers and account explicitly for separate external-query packs.

Review clarification: the existing CUDA F2 kernel scans every reference tile; it does **not** traverse/prune the tree. Preserve that optimized route for existing fitting calls only. To honor the agreed always-KD prediction design, both prediction-core construction and external prediction queries use actual bounds-pruned traversal: generalize the existing CPU pruned traversal and CUDA per-query pruned traversal rather than routing prediction through F2. Share traversal/candidate arithmetic and preparation helpers, while keeping the fitting entry point's existing F2 specialization intact. Select prediction's pruned launch plan explicitly; do not infer search mode from tensor aliasing or query/reference row-count equality, and do not silently reuse a fitting planner that would select exhaustive F2. A new plane-based pruned kernel is not required.

### 5.4 Find the mutual-reachability connector and lambda

For a query's ordered candidate list:

```text
query_core = euclidean_neighbor_distances[s]  # zero-based s, NOT s-1
mrd[j] = max(prediction_core[neighbor_ids[j]],
             query_core,
             euclidean_neighbor_distances[j])
winning_position = first position with minimum mrd
connector = neighbor_ids[winning_position]
lambda = 1 / mrd[winning_position]  if mrd > 0
         f64::MAX                 otherwise
```

Two independent off-by-one rules must be tested: inclusive-self training core index `s-1`, external-query core index `s`.

Preserve the **first minimum in candidate order**. Do not instead reduce by `(MRD, point_id)`: a closer Euclidean candidate can precede a lower-ID candidate with equal MRD.

Use f64 for square roots after distance readback, core/lambda storage, comparisons against promoted hierarchy lambdas, and strength arithmetic. Do not substitute `+inf` for the reference's `f64::MAX` zero-distance lambda.

### 5.5 Attach to the fitted hierarchy

Start with the connector point row's parent, `potential_cluster`.

Only if the connector's departure lambda is **strictly greater** than the query lambda:

```text
while potential_cluster > tree_root:
    edge = cluster_parent_lookup[potential_cluster], if present
    if edge is absent or edge.birth_lambda < query_lambda:
        break
    potential_cluster = edge.parent
```

The upward comparison is **`>=`**; the initial departure comparison is **`>`**. Equality-boundary unit tests are mandatory.

If `potential_cluster` has no propagated selected-label entry, return noise and strength zero. Otherwise return its mapped fitted label and:

```text
max_lambda > 0: min(max_lambda, query_lambda) / max_lambda
otherwise:      1.0
```

Do not clamp query lambda to the connector's departure lambda, and do not reuse `_extend_condensed_tree`'s separate rules.

For infinite maxima, preserve the formula. In particular a zero-MRD duplicate query has finite `f64::MAX` lambda and can have strength zero. A generic “identical point means probability 1” fast path would be wrong.

Map the winning compact connector through `finite_index` for the returned original training-row ID.

### 5.6 No-cluster-tree branch

If there are no cluster edges (`child_size > 1`), return labels `-1`, strengths `0`, connectors `-1`, and the explicit status flag. The Python bridge emits one `UserWarning` per call with the reference's meaning.

This condition is **not** “there are no selected labels.” A hierarchy with cluster edges but no selected labels still goes through neighbor search and can return valid connectors. Conversely, a root-only selected cluster with no cluster edges still takes the reference's all-noise shortcut.

## 6. Ownership, caching, and backend execution

### Fit-time state

Add private immutable prediction state to `FittedHDBSCAN`, shared across Rust clones through `Arc`:

- Owned compact row-major float32 training coordinates, including the one-finite-row path.
- Feature count, finite/raw indexing information required by prediction, and resolved `s`; keep the concrete runtime backend selected by fit/future load in the separate runtime owner.
- Immutable point/cluster/label/max-lambda metadata described above.
- A lazily populated prediction-core cache.
- A retained immutable host `KdTree` whenever fitting creates one; other fit routes lazily build/cache it from retained finite coordinates. Retain a valid builder-compatible singleton (one leaf, index 0, range `0..1`, enclosing bounds) when a tree is required; never call a builder with zero rows.

Capture one authoritative immutable fit/load snapshot of complete fitted results, not an independently evolving prediction copy. Prediction and future export read this canonical snapshot. Existing public mutable Rust fields and Python arrays are detached output/config views; mutations affect neither prediction nor future canonical export. Do not export such mutated views as though they were authoritative. Document this focused snapshot/view contract without broad unrelated public API cleanup. Runtime backend/device selection is separate from stored training provenance.

Retain at most one additional permanent host geometry copy. Current tensor upload consumes a `Vec`; explicitly model any temporary clone needed for uploading while retaining an owned copy. Do not retain Python input buffers, duplicate the entire dataset again in Python, or retain device tensors after fit merely to make future queries possible.

### Lazy host caches

Use private synchronized tree/core cache state inside the shared runtime owner, with immutable `Arc` payloads and serialized fallible initialization:

1. Validate input, shortcut status, widths, and checked resource limits before cold work. Under one documented initialization lock/order, reuse an existing host tree or build and validate a complete local tree, then publish that tree atomically.
2. Reuse call-local uploaded coordinates/tree to compute and validate a complete local core vector once while initialization is serialized. Publish cores only on success; release initialization locks before external-query batches.
3. On core failure preserve an already retained/published valid tree and all fitted results; leave cores uninitialized so retry is possible. Tree construction failure publishes no partial tree. No helper may recursively acquire the initialization lock; define ownership of upload/init helpers to avoid lock inversion/deadlock.
4. Report poisoned-lock/internal errors explicitly rather than panicking. Concurrent cold calls must not publish mismatched tree/core generations or perform unbounded duplicate initialization; warm calls briefly clone immutable payloads, not hold locks across searches.

This avoids unstable fallible `OnceLock` initialization APIs, permanently cached transient backend errors, or unbounded duplicate first-call work by concurrent callers. Do not hold the cache mutex through every warm prediction.

No-cluster-tree models use canonical `NotRequired(NoClusterTree)` prediction-core status, including root-only and one-row models; do not force impossible `k=s` queries. Tree availability is tracked separately: current shortcut prediction need not build a tree, while future save prepares a valid singleton/ordinary tree. Ordinary models distinguish `Uninitialized` from validated `Ready { rank: s, values }`. Cache state belongs to a fitted model; successful refit creates a new one, while failed refit preserves the previous state/cache.

### Bounded search lifecycle

- Use the concrete runtime backend chosen by fit or future load strictly; never rerun `Auto` selection on prediction or bind execution to serialized training provenance.
- Upload host reference coordinates and tree once per prediction call, reuse their tensors/packs through cold-core queries and all external batches, then release every device buffer on return/error. No persistent device cache. Warm calls still pay uploads/packing; no-cluster shortcuts need none.
- Generalize the existing KD backend/kernel operation; no facade-to-UMAP dependency or production brute-force fallback.
- Use checked KD batch planning based on independent query/reference counts, `k`, feature width, stack/heap limits, packs, device limits, outputs, row pitch, and allocation lifetimes. A fixed 256-row cap alone is insufficient.
- Read back only a batch of candidate IDs/distances, perform backend-free hierarchy traversal on the host, append final row results, and release transient batch buffers. Do not materialize an `m × n` distance matrix or all `n × s` training-neighbor outputs.
- Reject sentinel IDs, missing candidates, NaN distances, and unexpected nonfinite Euclidean distance results before indexing. Finite float32 coordinates can still overflow squared-distance accumulation; report numerical failure rather than treating overflow as a missing/noise prediction.
- Keep CPU and CUDA compiled unconditionally; do not introduce backend feature flags or execution fallback.

### Expected cost and limitations

For training count `n`, query count `m`, dimensions `d`, width `s`, and batch rows `b`:

- Retained geometry: `4*n*d` bytes, host KD permutation/ranges/flags/bounds, O(n + cluster_count) metadata, and eventual `8*n` cores. Count actual host tree allocations and canonical fitted outputs, not assumed packed struct sizes; fit-created trees are retained rather than rebuilt/copied needlessly.
- Cold initialization: host tree construction only where absent, plus n exact KD queries for cores (unless `s=1` or not required). Tree build/packing and search are separate measured phases. Typical KD pruning reduces work; worst-case KD search remains quadratic for all-training queries, especially high dimensions/duplicates. No unconditional quadratic startup claim or universal subquadratic guarantee.
- Warm search: m exact KD queries plus candidate selection and ancestry walks; worst case O(m*n*d). Candidate outputs are O(b\*s), not a retained full pairwise matrix or n-by-s graph.
- Each call uploads coordinates and KD buffers once and constructs/reuses the route's call-local reference packs. Include query packs, padded/tile-rounded rows, stack/candidate scratch, readbacks, initialization overlap and error cleanup in checked peak accounting. No device cache is retained.

Report cold and warm KD cost separately, including host build, uploads and packing. A 100k-row cold initialization may still be expensive; disclose it rather than hiding it in fit or substituting ANN fitting cores. Upstream KD speed parity is not promised.

### Serialization-ready state boundary (current design; future persistence implementation)

Define a private backend-free host state/export contract now, following the separation in `fast-umap-core/src/state.rs`, not its UMAP-specific schema. Keep immutable canonical fitted/prediction data separate from runtime mutexes, `Arc` internals, selected runtime backend/device, tensors and handles. Never serialize those runtime objects or Python object/pickle graphs. No Rust-native file-format API or stub public save/load methods belong in prediction implementation.

The current deliverable is the authoritative state type/ownership contract used by fit/predict and a documented future preparation/export boundary, not an unused save/export function. Add callable persistence preparation/export/import only with the later serialization implementation and its consumers.

The schema-design inventory must cover compact finite f32 coordinates; ascending finite-to-raw mapping, raw/finite/feature sizes; complete fit-time configuration and requested/resolved fitting choices; labels/probabilities, condensed and single-linkage trees, MST, neighbor IDs/distances, fitting cores and all diagnostics/public fitted outputs; host KD indices, node ranges/flags/bounds/levels; separate prediction-core rank/required status/values; and the canonical source data for validated reconstruction of immutable prediction label maps/point attachments/maxima/tree metadata, preserving every quirk in section 5. Optional fitted outputs require explicit presence flags. Exporting a complete fitted model must preserve all fitted methods, not merely prediction. Preserve numeric dtypes, hierarchy IDs and raw-vs-compact domains explicitly; do not copy UMAP array names/schema verbatim.

Document a private future save-readiness/export boundary around that canonical state. Current fit/predict stay lazy; **future save always prepares and saves the host KD tree and required prediction cores even when predict was never called**. Future save may build a tree/run KD queries and fail. For no-cluster/root-only/singleton shortcuts, store explicit `NotRequired(NoClusterTree)` cores rather than attempting impossible ranks; still prepare the valid host tree. For a nonempty cluster tree, reject invalid `s`/`2*s` widths before any cold work. Saving before/after first prediction must yield semantically equivalent complete state, not cache-dependent file support. Preparation may initialize legitimate complete caches, but errors cannot change core fitted results or publish partial caches/files.

For future storage, persist canonical fitted hierarchy/labels and reconstruct prediction point attachments, cluster-label maps and maxima deterministically on load using section 5; do not persist a second independently authoritative copy of those derived lookup tables. Persist the KD tree and required prediction cores as explicitly chosen. The serialization documentation must state that a nonempty-cluster-tree model with unsupported prediction width cannot be saved under this always-prediction-ready contract, even if fitting itself succeeded.

Future persistence follows UMAP's Python versioned HDF5 convention with Rust backend-free export/import, strict format/model-kind/version handling, temporary sibling file, checked overwrite policy, atomic publication and explicit durability-uncertain handling. Actual I/O, bindings and round-trip tests are separate future work. Backend-independent files restore a **complete** fitted model; future load accepts Auto/Cpu/Cuda plus device selection, resolves runtime execution there, and records original training backend only as provenance. Prediction thereafter uses that runtime selection with strict errors/no fallback. Cross-backend predictions use numeric tolerances, not a universal bitwise promise.

Treat every future imported field as untrusted before constructing/publishing a model or invoking GPU traversal. Validate checked dimensions/byte counts and resource/overflow limits before allocation; finite coordinates; raw remapping injectivity/order/ranges; config/ranks; all fitted shapes/index domains; labels and hierarchy/attachment consistency; and prediction-cache status/length/rank/nonnegative values. Explicitly validate KD permutation, reachable topology, child partition/range coverage, leaf flags, bounds enclosing assigned points/children, and level/stack limits, including canonical unused slots and singleton representation. Shape checks alone do not establish tree validity. Reconstruct and validate label maps/maxima from canonical results, retaining allowed infinite hierarchy lambdas and rejecting invalid NaNs. Complete host validation precedes untrusted GPU traversal; load publication is atomic.

Future save/load tests must include never-predicted and predicted models for every fitting route, mixed-nonfinite raw remapping, no-cluster/root-only/singleton cases, invalid-width readiness failures, CPU↔CUDA and explicit load overrides, corrupt topology/permutations/bounds/config/cache/mapping, allocation-overflow attacks, and atomic failure behavior. Check all fitted arrays/diagnostics/methods after load; saved predictions are exact where backend/arithmetic match and tolerance-based otherwise. These are future persistence acceptance tests, not claims of implemented file support.

## 7. Numerical and compatibility boundaries

The original package queries float64 geometry; this project fits and executes shared distance kernels with float32 geometry/arithmetic. The implementation targets the reference **algorithm and fitted-model semantics**, not bitwise equivalence to original double-precision fitting or neighbor selection at numerical boundaries.

- Compare original hdbscan using the same float32-rounded input promoted to float64, so preprocessing differences are not confused with prediction differences.
- Keep lambda/probability calculations f64, while acknowledging that their distances and tree lambdas originated in f32.
- Use deterministic shared-kernel distance/index ordering for geometric ties. scikit-learn's KD tree can return a different equal-distance order; do not claim identical connecting IDs for such fixtures.
- MRD ties preserve candidate order, even where Euclidean distances are not tied.
- Same-tree/candidate-list tests must match exact labels and formula behavior. Keep departure/birth equality-boundary assertions on controlled candidate/lambda values, not independently fitted f32/f64 trees. Full independent-fit comparisons require untied, well-separated fixtures and explicit label alignment.
- Strength tolerances for untied end-to-end comparisons should start at `rtol=1e-5, atol=1e-6`, with exact noise masks and finite/range checks. Any necessary relaxation needs a fixture-specific numerical explanation; do not silently widen tolerances to hide an algorithm error.
- Preserve original prediction behavior for epsilon/persistence/max-size-selected models by deriving mappings from their final fitted labels/tree. Do not reapply fitting thresholds during prediction. Original upstream warns about some selection options; that is a reason for dedicated tests, not a reason to silently ignore the final fitted selection or refuse all such models.
- Mixed nonfinite **training** data remains supported here through finite compaction and raw connector remapping. This is a deliberate extension beyond the inspected original prediction-data generator, which rejects NaN training coordinates.

## 8. File-by-file implementation sequence

Each stage must leave a working, testable layer. No placeholder methods or fake fallbacks.

Each implementation lane is bounded to at most five explicit source/test files listed below; sequence dependent contracts (shared KD before estimator integration). Fixture data is not a license for package-wide source edits. If another source file proves necessary, split a new bounded lane rather than silently expanding a lane. This plan revision runs no gates; commands below are future implementation obligations.

### Stage A — Pin prediction semantics and independent oracles

Files:

1. New `python/tests/test_hdbscan_prediction.py`.
2. New `crates/fast-hdbscan/tests/fixtures/generate_hdbscan_prediction.py`.
3. New prediction-specific fixture JSON/data under the existing fixture directory.

Work:

- Use original `hdbscan`, never the existing `generate_fast_hdbscan.py` oracle, for prediction fixtures.
- Record package/source identity, parameters, float32 input representation, query arrays, raw hierarchy, labels, candidate arrays, prediction cores, maxima, expected outputs, and tie annotations.
- Build a test-only same-hierarchy adapter with `CondensedTree(raw_tree, labels)` and `PredictionData`; use a minimal object with `prediction_data_`, `condensed_tree_`, `min_samples`, and `min_cluster_size` for calling original `approximate_predict`.
- Keep ordinary upstream estimator tests as a second, separate comparison layer.
- For nonfinite-training extension cases, use compact finite coordinates with explicitly mapped point attachments; original `PredictionData` cannot consume NaN rows as-is.

Acceptance: fixtures expose both core-rank conventions, MRD first-minimum behavior, ancestry boundaries, infinite maxima, root-only shortcut, and connectors for noise. Generated reference data is reproducible without native implementation dependencies.

### Stage B — Backend-free prediction hierarchy and math

Files:

1. New `crates/fast-hdbscan-core/src/prediction.rs`.
2. Update `crates/fast-hdbscan-core/src/lib.rs` exports.
3. New `crates/fast-hdbscan-core/tests/prediction.rs`.

Work:

- Add validated immutable hierarchy metadata construction from final tree/labels/finite index.
- Implement candidate-list-to-label/strength/connector math with f64 intermediates.
- Validate column lengths, point attachments, IDs, parent relationships, and candidate/core lengths at construction/API boundaries. Do not let malformed data cause unchecked indexing or unbounded ancestor loops.
- Use iterative ancestry traversal and precomputed lookup tables; do not add Burn/CubeCL/scikit-learn dependencies to the core.

Acceptance: independent handcrafted and original-derived fixtures prove the formulas and branches without device execution. Malformed metadata/candidate input returns typed errors rather than panics. Existing host extraction behavior is unchanged.

### Stage C1 — Generalize shared KD traversal before model integration

Files (one bounded implementation lane):

1. Update `crates/fast-hdbscan/src/backend.rs` query/reference operation and fusion metadata.
2. Update `crates/fast-hdbscan/src/kernels/kdtree.rs` shared traversal, query packing and wrapper.
3. Update `crates/fast-hdbscan/src/kernels/mod.rs` registration/dispatch as required.
4. Extend `crates/fast-hdbscan/tests/test_kdtree_search.rs`.

Work: introduce explicit query/reference tensors, self exclusion and query offset; retain the fitting wrapper over shared traversal. Output shape is query-count by k. Preserve scalar feature arithmetic, existing C16/F2/per-query fitting routes, IDs, diagnostic behavior, tree preconditions, packing and launches. Separate arbitrary-query packing/permutation from training leaf ordering and restore output query order. Implement safe equal-bound pruning. Test external unequal row counts, offset self queries, identical coordinates, irregular valid leaf structures, batch tails and equal-distance lower-ID winners against the common brute-force test oracle. Test both self modes on CPU/required CUDA. No production fallback.

Keep exhaustive F2 confined to its existing fitting entry point. Prediction, including training-core queries, explicitly selects the generalized bounds-pruned traversal; test the prediction launch plan at F2-eligible widths such as 16 and 30 to prevent accidental exhaustive dispatch. Refactor reference/tree upload and pack preparation into a call-local reusable owner: batching today's launch function unchanged would repack/reallocate each invocation and would not meet the lifecycle contract.

Acceptance: shared fitting exact IDs/distance bits/diagnostics remain unchanged against the current baseline on CPU and CUDA; packing-inclusive fit timings have no unexplained regression. Run focused fitting regression before ownership integration; later final 20k/100k validation remains mandatory. Do not infer prediction coverage from a fitting self-query test.

### Stage C2 — Fitted ownership and bounded exact-search orchestration

Files:

1. Update `crates/fast-hdbscan/src/estimator.rs`.
2. New `crates/fast-hdbscan/src/prediction.rs`.
3. Update `crates/fast-hdbscan/src/lib.rs` exports.
4. New `crates/fast-hdbscan/tests/test_prediction.rs`.

Work:

- Retain compact coordinates and canonical immutable complete fitted/prediction state in every successful fitting branch, including complete/KD, complete/brute-force, kNN exact, kNN ANN, and one-finite-row models. Retain fit-created host KD trees and lazily build/cache missing trees. Define the private backend-free state/readiness/export boundary from section 6 without implementing file I/O or public persistence methods.
- Build metadata from final labels/tree. Keep existing fitting core distances and diagnostics unchanged.
- Add the Rust method, synchronized tree/core caches with canonical not-required state, strict runtime-backend execution, independent training-core/external-query ranks, bounded batch sizing, and raw connector conversion. Preserve trees on core initialization failure; avoid nested locking/partial publication.
- Use shared KD exclusively for prediction, uploading reference coordinates/tree once per call through cold cores and external batches, then releasing device state. Move/reuse existing KD allocation/packing helpers with crate-private visibility instead of copying sizing logic.
- Keep output state publication atomic. Ensure source-input mutation or dropping the source does not affect predictions.

Acceptance: real CPU and CUDA predictions satisfy the same candidate/math assertions. Repeated calls reuse prediction cores, batch partitioning does not change results, and cloned fitted models share only immutable data/cache. Search failures do not corrupt fitted state or publish partial cache/results.

### Stage D — Resource accounting

Files:

1. Update `crates/fast-hdbscan/src/resources.rs`.
2. Extend `python/tests/test_hdbscan_resources.py`.
3. Extend `crates/fast-hdbscan/tests/test_prediction.rs` sizing/lifetime assertions if needed.

Work:

- Include retained training geometry, host KD tree and canonical prediction/fitted metadata in relevant fit lifetimes/output accounting. Model temporary upload copies, existing leaf-SoA/F2 packs, independent query packs and CPU/CUDA overlap, not just persistent bytes; reuse checked `kd_neighbor_plan`, `kd_neighbor_pack_bytes` and `kd_neighbor_pack_allocations` contracts where applicable.
- Prediction resource planning must reflect its explicitly pruned kernel/layout, not fitting's F2 selection; parameterize/reuse allocation calculations without copying an inapplicable fitting launch plan. Reference packs are prepared once per call and reused across all batches, not rebuilt by every dispatch.
- The existing estimate accepts finite shape and has explicit exclusions; update those explanations carefully. Newly retained model-owned finite geometry is not merely an excluded caller buffer.
- Preserve the current advisory's graph-route contract; do not unexpectedly add complete-route warnings.
- Document route-dependent host tree retention versus lazy construction and lazy O(n) cores, plus call-local upload/pack costs. Lazy first-call peaks are not covered by a fit-only estimator; a fit-created tree already counts during fit.
- Test checked sizing without allocating enormous arrays. Include row-pitch and very large `s` cases.

Acceptance: estimates match the implemented lifetimes and existing advisory tests still pass after expected-number updates. No full pairwise matrix is estimated or allocated by prediction.

### Stage E — Native bridge, Python method, and types

Files:

1. Update `crates/fast-umap-python/src/hdbscan.rs`.
2. Update `python/fast_umap/hdbscan.py`.
3. Update `python/fast_umap/_fast_umap.pyi`.
4. Extend `python/tests/test_hdbscan_prediction.py`.

Work:

- Add native fitted prediction using owned input conversion before `py.detach`, native validation, existing error mapping, and NumPy result construction after reattaching.
- Emit the no-cluster-tree warning from the explicit native result status, once per public invocation; do not emit duplicate wrapper/native warnings. Ensure the warning points to the caller.
- Add the public fitted guard, query conversion/finite/feature validation, return-flag validation, and thin native delegation. Do not read current mutable constructor hyperparameters to predict.
- Return two or three arrays as requested; document connector semantics and dtypes in both method and stub.
- Preserve successful-refit replacement and failed-refit rollback. A failed conversion of prediction results must not mutate the fitted model.

Acceptance: public and direct native entry points agree; shapes/dtypes, ownership, validation, optional connectors, warnings, and error classes are covered. Type checking handles both literal and dynamic return flags.

### Stage F — Documentation and final validation

Files:

1. Update existing `README.md` HDBSCAN API/examples and limitations.
2. Update native/public method documentation in the already changed source files.

Work:

- Show `.fit(...).approximate_predict(...)` and optional original-row connectors.
- Explain fixed-cluster semantics, geometry/host-tree retention, lazy tree/core cold start, exact KD query costs and worst cases, one-upload-per-call lifecycle, no device cache, Euclidean-only scope, float32 boundary and CPU/CUDA strictness. Document canonical immutable snapshot/export versus detached mutable output views and current-state-readiness versus future persistence scope.
- Explain prediction strengths need not equal `probabilities_` even when querying training coordinates. Explain the upstream-compatible duplicate/infinite-lambda, root-only, and label-derived anchor cases.
- Do not add an export to `python/fast_umap/__init__.py`: this is an instance method on an already exported class.
- Do not modify UMAP, common brute-force neighbor semantics, dependency manifests/locks, or serialization modules. Existing dependencies suffice; future persistence work is separate.

Acceptance: documentation describes the implementation that actually ships, including its cold cost. All required focused gates below pass; missing hardware validation is explicitly a blocker rather than a pass.

## 9. Verification matrix

### Core algorithm tests

- Inclusive-self training core rank `s-1`; query core rank `s`; `s=1`.
- Candidate MRD depends separately on neighbor core, query core, and raw distance.
- Equal MRD with different Euclidean distances chooses the first candidate, not lowest point ID.
- Zero MRD uses `f64::MAX`; finite and infinite parent maxima follow the original formula.
- Maximum lambda includes cluster-child rows, not only singleton point rows.
- Immediate attachment versus ascent: departure `<`, `==`, and `>` query lambda; birth `<`, `==`, and `>` query lambda; root termination.
- Descendant selected-label inheritance and ancestor maximum inheritance.
- Unmapped potential cluster gives `(-1, 0)` while preserving the valid connector.
- Empty cluster edges versus no selected labels despite existing edges; root-only cluster behavior.
- Selected ancestor with no direct point rows reproduces the reference's child-anchor/unmapped-sibling behavior.
- Fitted-label gaps are not renumbered; noise is not included as a selected anchor. Gate the gap fixture against this project's explicit contract, not original `_select_clusters`' positional grouping/renumbering behavior.
- Deep hierarchy, malformed columns, missing/duplicate point attachment, invalid ancestry, mismatched candidate widths, invalid IDs, NaN/negative invalid distances, and overflow return errors without panic/hang.

### Native estimator/backend tests

- Shared KD traversal regression: fitting wrapper retains exact IDs/distance bits/diagnostics and existing CPU C16/CUDA F2/per-query routing; packing-inclusive measurements cover the same baseline. External query row counts/order/packing must be independent from training leaves, including batch offsets, permutations and tails. Equal-distance node bounds cannot hide lower-ID candidates.
- At F2-eligible feature widths, verify fitting still selects its existing exhaustive F2 route where eligible, while prediction-core and external-query operations select actual pruned traversal. Verify reusable prepared reference packs survive all call-local batches without re-upload/repacking and are released on return/error.
- Fit then predict on separated clusters, varying density, far-away noise, leaves, and training-coordinate queries.
- Complete/KD, complete/brute-force, and kNN fitting routes; kNN ANN cores must not leak into prediction-core construction.
- EOM/leaf, epsilon-selected, persistence-simplified, max-size-limited, and single-cluster settings use final fitted label/tree mappings.
- Mixed nonfinite training rows, with removed rows before/between valid rows: connector IDs remain original IDs, and connector point attachments address the correct raw tree rows.
- One finite row; no-cluster models; `n < 2*s`, `n == 2*s`, `s > n`, and checked `2*s` overflow. Verify shortcut ordering separately from ordinary width errors.
- CPU/CUDA identical labels/connectors on untied controlled inputs and close strengths; deterministic tie policy within the project's arithmetic contract.
- Query batch sizes 1, internal-limit boundaries, and tails give identical results.
- Invalid/missing/sentinel candidates and finite-input squared-distance overflow are numerical errors, not noise.
- Runtime backend selection remains fit/load-selected and strict, not bound to serialized provenance. Injected tree/core initialization failures publish no partial payloads; core failure preserves a retained valid tree and later calls can succeed. No silent fallback, deadlock or permanently cached error.
- Input mutation/drop after fit, Rust clone sharing, repeated calls, and concurrent cold initialization preserve state and bounded initialization.
- Changing public Rust result/config/backend snapshots does not change stored prediction state.
- Verify all fit routes use exact KD prediction; common brute force appears only as test oracle. Verify one reference/tree upload per call across cold-core and query batches, no retained device buffers after success/error, retained tree reuse, canonical shortcut status and singleton bounds, and strict width rejection before cold work.
- Verify the private canonical host contract contains complete fitted outputs/diagnostics and validated prediction metadata/cache statuses without runtime locks/device handles. Future persistence gates in section 6 are deferred, not stubbed or claimed passing.

### Python lifecycle/API tests

- Unfitted access; correct two/three tuple shape; index/strength dtypes; boolean flag validation.
- Wrong dimensions/feature count, ragged/object/complex/nonnumeric input, zero rows/features, NaN/Inf, and float64 finite values beyond float32 range.
- Float64 and noncontiguous inputs follow the same preprocessing policy as fit.
- Output arrays are independent; mutation does not affect future calls.
- Mutating constructor attributes after fit does not change predictions.
- Failed refit preserves old predictions/cache; successful refit replaces both geometry and metadata.
- Public native bridge agrees with the high-level method; NumPy conversion errors preserve fitted state.
- No-cluster warning occurs once, with the expected tuple and sentinel outputs. All-noise labels alone do not trigger the shortcut warning.

### Reference parity: separate the sources of disagreement

1. **Fixed candidate arrays and fixed hierarchy:** compare backend-free math against original helpers with the same arrays. This is the strict algorithm gate, including exact branch tests.
2. **Same fitted hierarchy and labels:** construct the test-only original `PredictionData` adapter from the new model's tree and float32-rounded geometry. Compare ordinary untied predictions, prediction cores, strengths, and raw connector IDs. For mixed-nonfinite training extension tests, explicitly translate compact/original IDs in the oracle.
3. **Independent original fit:** fit original `hdbscan.HDBSCAN(prediction_data=True, ...)` on the same rounded training data. Use stable, untied examples; align non-noise label IDs from training memberships before comparing predictions. Do not mistake an MST/selection difference for a prediction bug.
4. **Ties/duplicates:** gate exact project behavior using controlled candidate lists; annotate the reference's alternative equal-distance ordering. Do not blanket-exempt label/strength discrepancies under a generic “ties” annotation.

Original hdbscan must be present for the focused parity gate. Do not use `pytest.importorskip` to turn a missing reference dependency into a passing parity claim.

## 10. Execution gates and measurement

Run these during implementation, not as evidence that this plan already implements anything.

### Focused correctness

```sh
cargo test -p fast-hdbscan-core --test prediction
cargo test -p fast-hdbscan --test test_kdtree_search
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_kdtree_search
cargo test -p fast-hdbscan --test test_prediction
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_prediction
cargo test -p fast-hdbscan --test test_estimator
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_estimator
cargo test -p fast-hdbscan-core
cargo test -p fast-hdbscan

uv run maturin develop --release
FAST_UMAP_TEST_BACKEND=cpu uv run --no-sync pytest \
  python/tests/test_hdbscan_prediction.py \
  python/tests/test_hdbscan.py \
  python/tests/test_hdbscan_resources.py

FAST_UMAP_TEST_BACKEND=cuda FAST_UMAP_REQUIRE_CUDA=1 uv run --no-sync pytest \
  python/tests/test_hdbscan_prediction.py \
  python/tests/test_hdbscan.py \
  python/tests/test_hdbscan_resources.py
```

New Rust backend tests must enforce `FAST_UMAP_REQUIRE_CUDA=1` rather than silently skipping when CUDA is required. Run them explicitly with that variable set. Existing Python CPU/CUDA fixture behavior is in `python/tests/conftest.py`.

### Formatting and static checks

Run formatters once across the union of changed code after implementation lanes finish, then check:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
uv run ruff check
uv run ruff format --check
uv run ty check
```

Do not run independent racing formatters or project-wide gates in delegated subtasks.

### Fit-regression validation required by repository guidance

Because shared KD fitting traversal, packing and fitted ownership/resource lifetimes change, run the native HDBSCAN stage analysis with real strict CPU/CUDA execution. Compare current baseline fitting IDs, distance bits, outputs/diagnostics and packing-inclusive timings; do not compare against an obsolete route or remove concurrent performance work:

```sh
bash scripts/analyze_hdbscan_stages.sh --rows 20000
bash scripts/analyze_hdbscan_stages.sh --rows 100000
```

The final 100k run remains required. Compilation-only, skipped CUDA, or benchmark-only evidence does not replace correctness. Preserve unrelated performance work already in the tree.

### Prediction-specific measurements

Measure original hdbscan and native prediction separately from fit. Record input seed/shape, `s`, backend, batch size, source identity, and whether runtime/JIT was warm.

- Training sizes: small semantic fixtures, 1k, 20k, and 100k where feasible.
- Feature widths: 2, 16, 30, and a high-dimensional kNN-fit case such as 50.
- Query sizes: 1, a batch tail, and 1k; do not require all Cartesian combinations.
- Report fit overhead and retained host tree/geometry bytes, first prediction separating lazy tree build, cores, JIT, uploads and packing, warm repeated prediction including upload/pack/readback, and peak transient memory. Distinguish fitting-retained tree routes from lazily built routes.
- Report tree build plus exact KD-query startup, with worst-case quadratic search caveat and measured pruning behavior rather than unconditional quadratic claims. Do not describe a 100k fit benchmark as validation of 100k prediction.
- Select and record a bounded prediction profiling budget before running large cases; an over-budget case is reported as such, never silently omitted or counted as passing. There is no numerical speed target supplied by this feature request.
- Any material fit regression must be disclosed with measured cause and reviewed engineering judgment; do not hide retention/cold work outside measured phases or weaken correctness for speed.

## 11. Completion criteria

Implementation is complete only when:

- Both fitted Rust and Python methods work end to end with optional raw-row connectors.
- Prediction state owns its geometry and remains consistent with fit-time labels/tree/options/backend.
- Prediction always uses generalized exact KD traversal; fitting arithmetic/order/routing/packing behavior is preserved, fit-created host trees survive, and lazy tree/core publication is failure-safe. Each prediction call reuses one reference/tree upload without a device cache.
- A complete backend-free canonical fitted-state/readiness contract is defined, including validated host KD and required/not-required prediction cores, immutable snapshot/view semantics and future load-time runtime backend selection. Actual Python HDF5 save/load, Rust export/import bindings and round-trip implementation remain explicitly future work, with no public persistence stubs.
- Prediction-specific core ranks, MRD tie selection, ancestry inequalities, maxima propagation, and infinite-lambda behavior are proven against original hdbscan.
- Every supported fit route yields usable prediction state; existing public fitting outputs remain unchanged.
- Invalid input, unsupported width, allocation overflow, and backend/numerical failures remain explicit errors.
- Focused CPU and actual required CUDA correctness gates pass; no hardware skip is presented as coverage.
- Resource estimates and existing documentation account for the additional fitted ownership and lazy initialization.
- Cold/warm performance and float32-vs-reference boundaries are reported honestly.
- No unrelated work is reverted, no placeholder API ships, and no new runtime dependency on either Python hdbscan implementation is introduced.
