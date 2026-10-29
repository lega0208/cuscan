# HDBSCAN upstream parity audit

Audit date: 2026-10-29. Local source revision: `66865aea9550dd684e6af399a54e55292a06527f` (initial working tree clean).

## 1. Scope and verdict

This compares **`fast_umap.HDBSCAN` / Rust `fast-hdbscan`** with **TutteInstitute/fast_hdbscan**, not with the larger scikit-learn-contrib `hdbscan` package or sklearn's HDBSCAN implementation.

**Verdict: a substantial, deliberately restricted implementation, not a drop-in upstream replacement.** Ordinary unweighted Euclidean HDBSCAN, EOM/leaf selection and hierarchy cuts are implemented. Upstream's weighted, semi-supervised, constrained, non-Euclidean, sparse-precomputed, PLSCAN and branch-detection capabilities are absent. The port adds CPU/CUDA execution, prediction and portable prediction-only persistence. Even within the common subset, neighbor algorithms, precision, tie handling, public tree objects and sklearn integration differ.

### Baselines

- Primary upstream source: [`43c573ba61043a6ad6f10b396656bfaa7aaeead8`](https://github.com/TutteInstitute/fast_hdbscan/commit/43c573ba61043a6ad6f10b396656bfaa7aaeead8), returned by the upstream `main` API during this audit.
- Published-release baseline: `fast_hdbscan==0.3.2`; release tag [`21b24c4cf4e47cd26716695ce38fb3460732c99b`](https://github.com/TutteInstitute/fast_hdbscan/tree/21b24c4cf4e47cd26716695ce38fb3460732c99b). Local reference fixtures target this release, **not current main**.
- Main also declares version 0.3.2. A version string alone does not establish source identity. The fixture manifest records different release/main blob IDs for `hdbscan.py`, `cluster_trees.py` and `numba_kdtree.py`.
- Installed reference package reported 0.3.2 and the broader constructor signature during this audit. Arbitrary metrics, high-dimensional NNDescent routing and constraints are not merely hypothetical future upstream features: they also exist in the inspected release source.

“Supported” below means implemented at the inspected source, not exhaustive empirical equivalence. Runtime checks performed here are listed in §10. Low-level implementation helpers are discussed where they affect public behavior; this is not an enumeration of every kernel symbol.

## 2. Public feature inventory

| Feature | Upstream fast_hdbscan | Local port | Parity / consequence |
|---|---|---|---|
| `HDBSCAN` estimator | `fast_hdbscan.HDBSCAN` | `fast_umap.HDBSCAN`; Rust `HDBSCAN` / `FittedHDBSCAN` | Different package and type identity |
| `fit`, `fit_predict` | Yes | Yes in Python; Rust fits then exposes labels | Common basic workflow |
| Functional `fast_hdbscan(...)` | Yes; optional seven-value tree return | No equivalent top-level Python function | Native `fit_hdbscan` is a different, private-extension interface |
| sklearn `BaseEstimator` / `ClusterMixin` | Yes | No inheritance, no `get_params` / `set_params` | Not sklearn clone/grid-search compatible as-is; do not infer full estimator interoperability from method names |
| Dense real Euclidean features | Yes | Yes | Shared supported domain; numerical differences remain |
| Other metrics / metric keyword arguments | NNDescent-supported metrics | No | Non-default `metric` or any non-`None` `metric_kwds` raises `NotImplementedError`, including `{}` |
| Sparse precomputed distance graph | Yes, `metric='precomputed'` | No | Upstream requires a sparse square graph, not a dense distance matrix |
| Sparse feature matrix | Not established end-to-end by this audit | No supported wrapper path | Do not confuse sparse precomputed support with sparse features |
| Sample weights | Scalar or per-row vector, subject to route restrictions | No | Any non-`None` `sample_weight` rejected |
| Semi-supervised labels / BCubed selection | `semi_supervised`, `bc` / `bc_simple` | No | Any non-`None` `y` rejected, even with default unsupervised configuration |
| Cannot-link pairs/groups | Available on applicable Kruskal routes | No | Constructor names exist locally but non-default values reject |
| Borůvka complete-graph MST | Yes for applicable Euclidean input | Yes | Implementations differ: Numba KD versus CubeCL brute force/KD |
| Explicit `algorithm='kruskal'` | Yes, subject to dispatch caveats | No | Local high-D route internally uses Kruskal, but this does not implement the public algorithm switch |
| kNN graph route / `knn_k` | Yes | Yes | Same dimension threshold, different neighbor construction and seed behavior |
| EOM and leaf extraction | Yes | Yes | Release-specific and tied-data caveats below |
| Epsilon, persistence, maximum size, single-cluster options | Yes | Yes | Interaction and validation details below |
| DBSCAN-style hierarchy cut | `dbscan_clustering(epsilon)` | `dbscan_clustering(eps)` | Same strict cut principle; keyword name differs |
| Tree plotting/export wrappers | contrib `hdbscan.plots` objects | Raw NumPy arrays | Upstream `.plot()`, `.to_numpy()` etc. cannot be called on local arrays |
| PLSCAN | Yes | No | No multilayer persistence-landscape estimator or layer outputs |
| `BranchDetector`, `find_branch_sub_clusters` | Yes | No | No FLASC-style branch postprocessing, propagation or approximation-graph API |
| Generic subcluster utilities | Upstream internal modules | No corresponding supported Python API | Not implied by HDBSCAN core extraction |
| CPU execution | Numba/NumPy, parallel KD route | CubeCL LLVM CPU | Different runtime/dependencies |
| CUDA/device selection | No corresponding public backend interface | `backend`, `device_index` | Local addition; not upstream parity |
| New-point `approximate_predict` | Not present in inspected upstream | Yes | Additional contract based on contrib `hdbscan`, not fast_hdbscan |
| Detached `HDBSCANPredictor` | Absent | Yes | Local addition |
| Custom save/load model format | No custom HDBSCAN format found | Predictor-only HDF5 | Not full estimator serialization or upstream file compatibility |
| Resource estimates / large-kNN advisory | No equivalent inspected interface | Yes | Local addition, not an allocation guarantee |
| Outlier scores, cluster persistence array, relative validity, weighted centroid/medoid | Not found on inspected upstream HDBSCAN | No corresponding supported wrapper API | Not a parity gap against this upstream; these are often confused with contrib APIs |
| All-cluster soft membership vectors | Not found | Not implemented | `probabilities_` and prediction strengths are assigned-cluster strengths, not probability distributions over all clusters |

Upstream's functional API defaults `min_samples=10` and `min_cluster_size=10`; its estimator and the local estimator default `min_cluster_size=5`, `min_samples=None`. Replacing the functional API with the estimator without specifying sizes changes clustering.

Additional unsupported upstream surfaces are substantial, not aliases for ordinary HDBSCAN:

- **PLSCAN** builds multiple cluster resolutions, with `max_layers`, `base_min_cluster_size`, `base_n_clusters` and `layer_similarity_threshold` controls. It exposes `cluster_layers_`, `membership_strength_layers_`, `layer_persistence_scores_`, `min_cluster_sizes_`, `total_persistence_` and `cluster_tree_`; primary `labels_` / `membership_strengths_` come from the highest-persistence layer. None of these layer APIs exists locally.
- **BranchDetector** performs branch/subcluster selection using centrality relative to a probability-weighted centroid, with label propagation and approximation-graph outputs. This is neither local leaf selection nor new-point prediction.
- **Sparse precomputed graphs** preserve explicit zero-distance edges, remove diagonals, reduce asymmetric/duplicate edges by undirected minimum, and allow disconnected components with infinity bridges. A missing sparse entry means no edge, not zero distance. The local feature-array API implements none of this graph-input contract.

Sources: [U1–U6], [L1–L4].

## 3. Constructor and fit parameters

| Parameter | Local behavior compared with upstream |
|---|---|
| `min_cluster_size=5` | Supported positive integer; `min_samples=None` resolves to this. Both sources allow 1 at initial validation, but local fitting can fail later at this boundary (observed below). Do not claim reliable parity for size 1. |
| `min_samples=None` | Supported positive integer or `None`; non-self neighbor rank matters, especially with duplicates and insufficient rows. Local Python rejects booleans. |
| `cluster_selection_method='eom'` | `'eom'` and `'leaf'` supported. Ordinary EOM prefers parent on equal stability; `max_cluster_size` is not a general leaf size cap. |
| `allow_single_cluster=False` | Supported. Root-selection labeling still uses lambda thresholds: `True` does not guarantee every point belongs to one cluster. |
| `max_cluster_size` | Upstream default `np.inf`; local default `None`, with `+inf` normalized to unlimited. Otherwise local Python requires a positive integer. Unlimited semantics match; accepted types do not. |
| `cluster_selection_epsilon=0.0` | Supported; floating type required, not integer `0`. Local requires finite and nonnegative. Upstream checks nonnegativity/type but does not explicitly reject all nonfinite thresholds. |
| `cluster_selection_persistence=0.0` | Supported with the same local finite/nonnegative floating-type restriction. Exact equality and tie behavior need boundary-aware comparison. |
| `reproducible=False` | Accepted but implementation-specific, not a cross-library reproducibility guarantee; see §4. |
| `knn_k=None` | Supported positive integer or `None`, used only by graph route. Local rejects `0`, bool and float; upstream zero/default handling differs. |
| `semi_supervised=False` | Only the default accepted locally. |
| `ss_algorithm='bc'` | Only default accepted; even `'bc_simple'` rejects locally. |
| `metric='euclidean'` | Only this spelling accepted locally; no metric aliases/custom callables. |
| `algorithm='boruvka'` | Only this value accepted locally. It does not mean every local fit actually executes Borůvka. |
| `cannot_link=None`, `cannot_link_groups=None` | Only `None` accepted locally. |
| `validate_cannot_link=True` | Only literal default accepted locally; disabling validation is unsupported even without constraints. |
| `metric_kwds=None` | Only `None` accepted locally. |
| `backend='auto'`, `device_index=0` | Local-only controls. `auto` selects device 0; `device_index` selects explicit CUDA, not the Auto device. |

Validation generally occurs at `fit`, not construction. Local `fit(X, y=None, sample_weight=None, **fit_params)` rejects non-`None` `y`, non-`None` weights and every extra fit keyword with `NotImplementedError`. Parameter names in the signature therefore **do not mean those features are implemented**.

Upstream supports scalar/vector weights, rejects all-zero weights, and disallows weights for non-Euclidean/precomputed modes. Its inspected fit path does not enable the weight helper's nonnegative check; this audit does not promise that upstream rejects negative weights. Semi-supervision requires usable labels and supports BC/BC-simple in EOM, not a general supervised version of every extraction mode.

Sources: [U1], [L1], [L3].

## 4. Graph construction, exactness and reproducibility

### Dispatch

| Input / choice | Upstream | Local |
|---|---|---|
| Unweighted Euclidean, ≤30 features | Default parallel Numba KD-tree Borůvka | Complete mutual-reachability route; Auto uses brute force below 256 finite rows, KD at ≥256 rows |
| Unweighted Euclidean, >30 features | NNDescent kNN graph and Kruskal | kNN graph and host Kruskal; exact neighbors through 4096 finite rows, CubeCL ANN above 4096 |
| Other non-precomputed metric | NNDescent graph/Kruskal | Unsupported |
| Sparse precomputed | Sparse graph MST paths | Unsupported |
| Explicit complete/KD/brute-force route control | Different upstream algorithm controls | Rust `MstGraph::{Auto,Complete,KnnGraph}` and `HdbscanSearch::{Auto,BruteForce,KdTree}`; not exposed by high-level Python |

Local forced complete Auto-search above 30D uses brute force. `with_search` controls complete-graph search, not the kNN route. Exact nearest neighbors on the kNN route **do not make the resulting sparse-graph MST an exact complete-graph MST**.

Both graph routes use default width `max(3 * min_samples, 15)` and floor it at `min_samples`; local caps at `n_finite - 1`. Local ANN uses seed 42 and `max(5, round(log2(n)))` iterations. Upstream public routing does not pass a random state to its NNDescent helper. These are different algorithms, not interchangeable implementations of an identical seeded graph.

### Important behavioral differences

- **Self exclusion:** local removes the actual query row by ID. Upstream NNDescent strips the first returned neighbor column, which can remove a duplicate instead of self. Duplicate geometry can therefore change core distances and downstream clustering.
- **Ordering:** local distance/ID and edge ordering is deterministic on its own arithmetic. Upstream traversal, duplicate ordering and parallel behavior need not agree. Compare partitions after label alignment, noise masks, hierarchy connectivity and appropriate numeric tolerances—not raw cluster IDs alone.
- **Disconnected graph:** local Kruskal connects components with `+inf` bridges using component-minimum finite-row IDs. Upstream also bridges disconnected approximate/precomputed graphs with infinity; bridge topology/order need not match. Infinity is meaningful, not necessarily an execution failure.
- **Insufficient neighbors:** local neighbor outputs pad with `-1`, unavailable cores become infinity where required. kNN core calculation special-cases `min_samples<=1` to zero; do not assume every route has identical rank/degenerate behavior.
- **`reproducible`:** local changes pruning-bound mode only on complete-graph KD (independent per-row bounds versus shared component bounds). Brute-force and kNN routes ignore it. It does not select an upstream-compatible RNG or promise bitwise equality across devices.
- **Upstream dispatch caveat:** the non-Euclidean / unweighted >30D NNDescent branch does not forward `algorithm`, `reproducible`, `cannot_link` or `cannot_link_groups`. Constructor acceptance must not be reported as constraint enforcement/reproducibility on that route. Upstream validates constraint/algorithm combinations, but this is not proof those constraints reach NNDescent.
- **Precision:** local training geometry is float32. Widening output arrays to float64 does not recover lost geometric or hierarchy precision. CPU vector/scalar contraction and CUDA arithmetic can change near-ties; backend-wide bitwise identity is not promised.

The README sentence “Only exact Euclidean, unweighted clustering is supported” is too broad if read as a guarantee for every route: the same section documents approximate neighbors above 4096 rows in >30D. Actual dispatch is the controlling evidence.

Sources: [U1], [U3], [U4], [L3–L6], `README.md:504-541`, `AGENTS.md` backend arithmetic notes.

## 5. Extraction and known release/main differences

Shared pipeline: sort MST edges, build single linkage, condense by minimum size, optionally simplify by persistence, select EOM/leaf clusters, apply epsilon merging, then produce noise labels and membership strengths. This is broad algorithmic parity, not equality for every hierarchy.

- Ordinary EOM's maximum-size constraint does not impose a maximum on leaf-selected clusters. Upstream's semi-supervised BCubed route has separate selection behavior; it has no local counterpart.
- Epsilon selection merges selected clusters upwards; root/single-cluster and zero/infinite-lambda cases have special rules.
- **Known preserved release behavior:** upstream [PR #72](https://github.com/TutteInstitute/fast_hdbscan/pull/72) corrects `segments_in_branch` leaf traversal after 0.3.2. The local `select.rs` test explicitly expects the release reproducer `{6,8,9}` rather than corrected `{6}`. Therefore local release-oriented epsilon selection must not be described as parity with the corrected main implementation. The fixture manifest records both results.
- **Measured `moons_leaf` divergence:** this audit reran the existing comparison. Core distances and sorted MST weights had zero maximum absolute difference; connectivity matched at all 28 union thresholds. After label alignment, row 22 changed cluster membership, noise masks agreed, ARI was about 0.909106 and maximum probability difference about 0.0667292. This is a real membership difference despite an equivalent measured threshold hierarchy. It is fixture-scoped evidence, not proof of a universal cause.
- `duplicates_ties`, persistence equality, overflow and padding have explicit fixture annotations. A “reference” fixture or permissive tie-safe test is not proof of exact upstream row-for-row equality.
- Current upstream tree code additionally guards lambda-zero bridge labeling. This audit did not exhaustively establish local equivalence for every post-release bridge/tree fix; release fixtures alone cannot establish it.

Sources: [U2], [L7], `crates/fast-hdbscan/tests/fixtures/manifest.json:158-173,320-371,440-466`, `python/tests/test_hdbscan.py:217-342`.

## 6. Input, errors, state and outputs

### Input and edge behavior

Local Python accepts real numeric dense 2-D arraylikes, validates through float64, then creates contiguous float32 geometry. Complex, nonnumeric and sparse objects are not supported; finite coordinates outside float32 range reject. Representable coordinates can still overflow squared-distance arithmetic: coordinate validation is not a guarantee of finite intermediate distances. No feature scaling is applied.

Training removes rows containing NaN/infinity from geometry and restores their labels/probabilities as `-1`/`0`. At least one fully finite row is required. A single finite row has an explicit noise/zero/empty-hierarchy contract; it still resolves the selected backend. The fixture catalogue distinguishes one-finite-row handling from release remapping crashes and rejects all-nonfinite input before MST work. Upstream estimator and standalone functional API differ in nonfinite acceptance; they should not be treated as one input contract.

Local errors are explicit: unsupported features raise `NotImplementedError`, invalid wrapper inputs generally `ValueError`, and native Rust failures use typed `HdbscanError`/mapped Python errors. There is no promise of matching upstream exception types/messages, especially Numba typing failures. Failed local refits preserve the previous fitted snapshot, predictions and cuts; successful refits replace it atomically after output construction. Returned properties are independent copies, not writable access to canonical model state.

### Output contract

| Output | Local Python | Difference / qualification |
|---|---|---|
| `labels_` | `np.intp`, original row order; noise `-1` | Label numbers are not portable cluster identities |
| `probabilities_` | float32 for fully finite training input, float64 after nonfinite remapping | Mirrors observed upstream release dtype convention; not full soft membership |
| `condensed_tree_` | Structured NumPy array: `parent:intp`, `child:intp`, `lambda_val:float64`, `child_size:float32` | Not upstream `CondensedTree` object |
| `single_linkage_tree_` | `(rows,4)` float64 array: left, right, distance, size | Not upstream `SingleLinkageTree` object |
| `minimum_spanning_tree_` | `(edges,3)` float64 array: source, target, weight | Not upstream plotting wrapper; finite-row indices |
| `neighbors_` | `np.intp`, finite-row indexing, resolved-min-samples width with `-1` padding | Upstream uses private `_neighbors`; graph construction internals need not have same width |
| `core_distances_` | float32, Euclidean distances, finite-row indexing | Upstream counterpart private `_core_distances` |
| `finite_index_` | `np.intp`, finite row → original row | Local public remapping aid |
| `n_features_in_` | Public fitted property | Local explicit feature-count contract |

Labels/probabilities and remapped linkage/condensed trees use original-row space; MST/neighbors/core distances use compact finite-row space. Map through `finite_index_` before mixing them. Public upstream plot wrappers optionally depend on contrib `hdbscan`; precomputed MST plotting warns/returns `None` without raw feature data. Local raw arrays have no such visualization layer.

`dbscan_clustering` unions distances **strictly less than** the cut, then removes components smaller than resolved `min_samples` (or `min_cluster_size` when omitted). It cuts an existing hierarchy, does not independently refit DBSCAN, and is not the contrib method with an extra minimum-cluster-size argument. Local rejects NaN, allows meaningful negative/infinite cuts, and uses keyword `eps` rather than upstream `epsilon`.

Sources: [U1–U2], [L1], [L3], [L7–L8].

## 7. Local-only prediction and persistence

Neither `approximate_predict` nor prediction-data generation was found in the inspected **fast_hdbscan** estimator. Local prediction is instead modeled on **scikit-learn-contrib `hdbscan.approximate_predict`**. Existing prediction oracle tests use that separate project; they are not evidence of upstream-fast_hdbscan prediction parity.

Implemented locally:

- `HDBSCAN.approximate_predict(X, return_connecting_points=False)` and the same method on detached `HDBSCANPredictor`.
- Existing fitted cluster IDs only; no new clusters, refitting or query-to-query neighbors.
- Fresh `np.intp` labels, float64 strengths and optional `np.intp` original-training-row connecting points. Connector is the mutual-reachability choice, not necessarily Euclidean nearest. Noise can retain a valid connector.
- Queries must be nonempty, wholly finite, correct-dimensional and float32-representable. Prediction does not silently discard invalid rows.
- A tree without cluster edges—including relevant root-only cases—warns once per call and returns noise/zero strengths/`-1` connectors. Duplicate/infinite-lambda cases have explicit semantics; querying training coordinates need not reproduce `probabilities_`.
- Fit-time configuration/backend remains authoritative after constructor attributes mutate. `to_predictor()` prepares needed cores synchronously and may fail; the detached predictor survives refit or estimator destruction.
- Lazy host caches and transient device buffers are implementation details; cold prediction-core work can be expensive. Fit resource estimates exclude lazy prediction peaks. Float32 geometry, arithmetic and distance/ID ties prevent a general reference boundary/connector identity guarantee.

Persistence is **only** `HDBSCANPredictor.save(...)` / `.load(..., backend=..., device_index=...)`: strict schema-v1 HDF5, optional `h5py` and `hdf5plugin`, prediction geometry/hierarchy/ready cores, atomic publication and no-clobber default. Loading selects a runtime independently of fit provenance; no automatic execution fallback. Rust exposes storage-independent `export_state` / `from_state`, not an HDF5 API.

There is no local `HDBSCAN.save/load`, no recovery of all fit outputs from a predictor file, and no fast_hdbscan/contrib file interchange contract. Upstream has no inspected custom save/load schema; generic sklearn/Python pickle behavior was **not** tested, so it is not classified as unsupported. Local pickle behavior was not established either. Predictor files contain sensitive training geometry; schema validation/compression is not authentication.

Sources: [L1], [L9], `README.md:551-648`, `python/tests/test_hdbscan_prediction.py`.

## 8. Runtime and Rust-specific additions

- Local CPU and CUDA implementations are always compiled. Runtime selection is `auto`/`cpu`/`cuda`; there is no exposed supported WGPU mode. Deployment guidance is Linux x86-64 including WSL2, not verified general wheel/platform coverage.
- Explicit CPU bypasses CUDA discovery. Auto uses CUDA device 0 when discovery succeeds with devices and CPU when discovery confirms none; discovery failure and execution failure are errors, **not** CPU recovery. Explicit CUDA validates the requested index and never silently switches backend.
- Fit and prediction mix device geometry kernels with host hierarchy/extraction; “CUDA backend” does not mean every operation is GPU-resident.
- Local Rust offers backend-free supplied-MST extraction through `fast-hdbscan-core`, finite remapping, deterministic KD construction, and explicit graph/search controls. Its independent exact MST oracle requires `test-support`; it is not a production fallback.
- Rust config is not identical to Python validation: for example, the inspected Rust config does not explicitly reject `max_cluster_size=Some(0)`, while Python does. Rust inputs are rectangular `Vec<f32>` rows rather than general Python arraylikes.
- Local fit resource estimates produce conditional CPU/CUDA scenarios without initializing devices. Python warns for estimated kNN allocations at 1 GiB; this is advisory, not a cap or available-memory check. Estimates exclude caller input and some preparation/prediction allocations. Warning filters can suppress or escalate the advisory.

Sources: [L3–L6], [L9–L10], `AGENTS.md`, `pyproject.toml`.

## 9. Practical compatibility decisions

1. **Use as a supported-subset port, not an import substitution.** Keep required fit parameters explicit and reject reliance on unsupported constructor options.
2. **Align partitions before comparison.** Exact labels, MST endpoints under ties, hierarchy traversal and ANN graphs are not portable identities. Still treat changed memberships/noise/strengths as real differences, not mere renumbering.
3. **Separate release parity from main parity.** In particular, epsilon branch traversal deliberately retains a release behavior corrected upstream. Do not silently replace fixture provenance with latest-main claims.
4. **Separate exact and graph-relative guarantees.** High-dimensional kNN/ANN fits are not globally exact complete-graph HDBSCAN, even when the neighbor list itself was computed exactly.
5. **Do not overclaim prediction or persistence parity.** They are local extensions with their own reference and format, not implementations of upstream fast_hdbscan APIs.
6. **Remaining upstream feature gaps:** weights; semi-supervision/BC-simple; constraints; non-Euclidean metrics/kwargs; sparse precomputed graphs; explicit algorithm selection; functional Python API; sklearn estimator protocol; plotting wrappers; PLSCAN; branch/subcluster tooling. These are an inventory, not an implementation commitment.

## 10. Verification performed and limitations

Performed during this audit:

- Read current upstream commit API and pinned source through independent upstream/local source investigations; inspected local wrapper, dispatch, fixture provenance and extraction tests.
- Probed installed signatures and method availability. Upstream has `get_params`/`set_params`, local does not; local has `approximate_predict`/`to_predictor`, upstream does not.
- On a six-row 1D float32 input: upstream accepted `metric='manhattan'`, local raised `NotImplementedError`; upstream accepted `min_cluster_size=1`, local raised `ValueError: condensed tree row 3 mixes point/cluster child 10 with size 1`. This is a concrete local boundary gap, not merely stricter initial validation. Upstream `max_cluster_size=None` reached a Numba `TypingError`; local accepted it. Both rejected integer epsilon `0` with `ValueError`. The Manhattan probe emitted an upstream insufficient-neighbors warning.
- Ran:

  ```sh
  FAST_UMAP_TEST_BACKEND=cpu uv run --no-sync pytest -q python/tests/test_hdbscan.py \
    -k 'phase_one_fixtures_match_upstream or known_upstream_divergences or sample_weights_are_explicitly_unsupported or single_finite_and_too_few_finite_contract'
  ```

  **11 passed, 56 deselected, 1 warning.** The warning was the intentional `moons_leaf` divergence diagnostic summarized in §5. Passing means these bounded assertions passed, not that all source baselines/features are equivalent.

Not performed: full test suite; CUDA execution; new benchmarks; full release-to-main diff; every parameter cross-product; sparse/weighted/constrained upstream runtime scenarios; predictor serialization round trips; generic pickle checks. Existing tests/documented historical quality campaigns were inspected but not rerun beyond the command above. No product code changed in this audit.

## Source index

Upstream links are pinned to the primary commit unless stated otherwise:

- [U1] [`fast_hdbscan/hdbscan.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/hdbscan.py): functional API/validation (199–337), MST dispatch (339–494), extraction (496–573), estimator (575–885), PLSCAN (887–1184).
- [U2] [`cluster_trees.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/cluster_trees.py): selection, epsilon, labeling, cut and membership strengths (673–1001).
- [U3] [`nndescent.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/nndescent.py): neighbor construction and graph MST.
- [U4] [`precomputed.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/precomputed.py): sparse graph validation and preparation.
- [U5] [`__init__.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/__init__.py), [`branches.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/43c573ba61043a6ad6f10b396656bfaa7aaeead8/fast_hdbscan/branches.py): exports and branch tooling.
- [U6] [Release `hdbscan.py`](https://github.com/TutteInstitute/fast_hdbscan/blob/21b24c4cf4e47cd26716695ce38fb3460732c99b/fast_hdbscan/hdbscan.py), [PR #72](https://github.com/TutteInstitute/fast_hdbscan/pull/72): release/main distinction.

Local paths are relative to the audited repository revision:

- [L1] `python/fast_umap/hdbscan.py:30-133,148-559`; `python/fast_umap/__init__.py`.
- [L2] `crates/fast-umap-python/src/hdbscan.rs`: fitted output conversion, Python native signatures, backend mapping.
- [L3] `crates/fast-hdbscan-core/src/config.rs`; `crates/fast-hdbscan/src/estimator.rs:36-142,924-1022,1384-1557`.
- [L4] `crates/fast-hdbscan/src/lib.rs`; `crates/fast-hdbscan-core/src/lib.rs:1-104`.
- [L5] `crates/fast-hdbscan-core/src/knn_kruskal.rs`; `crates/fast-hdbscan-core/src/finite.rs`.
- [L6] `crates/fast-cubecl-common/src/device.rs:4-45`.
- [L7] `crates/fast-hdbscan-core/src/select.rs:207-250`; `condense.rs`; `labels.rs:153-209` in the same directory.
- [L8] `crates/fast-hdbscan/tests/fixtures/manifest.json`; `generate_fast_hdbscan.py`; `python/tests/test_hdbscan.py`.
- [L9] `crates/fast-hdbscan/src/prediction.rs`; `python/fast_umap/_hdbscan_predictor_serialization_v1.py`; `python/fast_umap/_serialization_common.py`.
- [L10] `crates/fast-hdbscan/src/resources.rs`; `python/tests/test_hdbscan_resources.py`.
