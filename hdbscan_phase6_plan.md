# Phase 6 — Euclidean Kruskal HDBSCAN

## Purpose and boundary

High-level follow-up to `hdbscan_plan_phases.md`, Phase 6+, using `hdbscan_impl.md` for intent and the published `fast_hdbscan==0.3.2` source for semantics. This proposal is not authorization to implement.

Add unweighted `algorithm="kruskal"` with `knn_k`, on the existing CPU/CUDA backends, Rust estimator and Python API. Preserve the default exact Borůvka path and reuse the existing extraction pipeline. Implementation starts from an accepted Phase 5 baseline; this proposal does not certify its completion.

Excluded: sample weights, sparse/precomputed inputs, constraints, semi-supervised clustering, non-Euclidean metrics, ANN/pynndescent, branch/layer clustering, package renaming and speculative GPU sorting or DSU implementations.

## Important clarification: Kruskal has two modes

The earlier overview describes the kNN-graph follow-up, but upstream's `knn_k=None` is **not an automatically chosen graph width**:

| Selection | Intended behavior |
|---|---|
| `boruvka`, `knn_k=None` | Existing exact algorithm, unchanged |
| `kruskal`, positive integer `knn_k` | Exact Euclidean neighbors; approximate MST restricted to their graph; disconnected components joined with synthetic `+inf` edges |
| `kruskal`, `knn_k=None` | Exact complete-graph Kruskal; quadratic edge storage and global sorting |

Deliver both Kruskal modes as separately gated milestones. Do not silently substitute Borůvka for dense Kruskal or invent a default k. A kNN graph built with exact neighbors can still miss complete-graph MST edges.

For D>30, continue the deliberate divergence: honor the selected algorithm and use exact distances/neighbors, never upstream public fit's automatic pynndescent dispatch. Direct pinned Kruskal internals, not public fit, are the strict oracle there.

## Proposed contracts

### API and validation

- Add a backend-free algorithm enum (`Boruvka`, `Kruskal`) and `knn_k: Option<usize>` to Rust configuration; default to Borůvka/None. Keep this separate from `HdbscanSearch::{Auto, BruteForce, KdTree}`, which chooses an exact neighbor/search implementation, not an MST algorithm.
- Python accepts only `algorithm="boruvka"` or `"kruskal"`. Accept `knn_k=None` or a positive integer, including NumPy integers but excluding booleans. Unknown algorithms, invalid k and non-None k with Borůvka are configuration errors (`ValueError` / `InvalidConfig`), not ignored options. This intentionally tightens upstream's incomplete k validation.
- Keep unsupported capabilities rejected with `NotImplementedError`; retain existing backend/error contracts and deterministic behavior. No execution fallback between backends or MST algorithms.
- Preserve existing result names, finite-row indexing, tree remapping, probability widening on non-finite remap, and `dbscan_clustering` semantics. Zero finite rows remains an error; one finite row remains the existing all-noise degenerate result.

### Graph and numeric semantics

- Let m be the finite-row count and s the resolved `min_samples`. For integer k, the non-self graph width is `max(k, s)` when s>1, otherwise k. Upstream uses **all** queried graph columns, not only the first k. Return only s neighbor columns, with -1 padding as needed; s=1 returns one column.
- Exclude the actual query-row id, rather than blindly dropping the first neighbor. Retain deterministic distance/index ties. Record this deliberate duplicate-point divergence from upstream's positional self removal in tie fixtures.
- Query only available non-self rows and represent unavailable output slots by -1/+inf. For s>=m, cores are infinite; s=1 gives zero cores for ordinary m>1 fits. Oversized k must not generate invalid endpoints or unchecked allocations.
- Integer-k distances start as the existing squared f32 exact-neighbor output. Widen to f64 **before** sqrt, then compute f64 mutual reachability `max(core[u], core[v], distance[u,v])`. Do not route Kruskal through Borůvka's f32-core/squared-weight finalizer.
- Dense mode follows upstream's distinct precision: compute Euclidean pair distances with f64 accumulation over f32 input values, derive f64 cores, and emit one edge per upper-triangle pair. Reusing f32 squared blocks would change overflow and ordering semantics. Provide backend operations for this precision on both CPU and CUDA; inability to verify either lane blocks that milestone, not permission for silent host fallback.
- Keep internal Kruskal cores and MST weights f64 through graph construction. Preserve the current public f32 core-distance output by narrowing only at the accessor/result boundary; document this port-level dtype divergence. Linkage remains f64 and downstream extraction retains its existing mixed precision. Do not change Borůvka numerics.
- Dense mode returns `min(max(1,s), m-1)` neighbor columns, as upstream does, except the shared one-row special case. Document this branch-specific shape rather than pretending it is always s columns.
- Use stable weight ordering with deterministic input ordinal: row-major directed query edges for kNN mode, upper-triangle enumeration for dense mode. Do not deduplicate or reorient graph edges before selection unless equivalence, including tie choices, is established.
- In kNN mode filter padding, actual self-loops and non-finite MRD edges before Kruskal. Dense mode retains its legitimate infinite MRD edges, matching its upstream branch. Never index a core array with -1.
- Host union-find selects a spanning forest. If c components remain, take each component's smallest finite-row id, sort those representatives, and append c-1 consecutive `+inf` bridges. For m>1 the completed result has exactly m-1 edges, is acyclic and connected. Bridges are synthetic topology, not discovered finite distances.
- Feed the completed tree through the existing MST sorting, linkage, condensation, selection, membership and finite-row remapping. Infinite bridges must produce valid lambda-zero behavior without NaN. Do not alter the established linkage-cut behavior for `epsilon=+inf` to accommodate this feature.

## Delivery sequence and exit gates

### 6.0 — Freeze semantics and add reference evidence

Extend the existing fixture generator/schema/manifest rather than replacing Borůvka fixtures. Pin and hash the release `kruskal.py` and relevant `precomputed.py` bridge implementation, in addition to existing provenance. The comparison commit `43c573ba61043a6ad6f10b396656bfaa7aaeead8` is context, not a substitute for the published artifact: existing fixture provenance already records differences between release and commit files.

Generate direct `kruskal_mst_from_feature_matrix` references for both modes and run their MSTs through upstream extraction. Capture graph neighbors/distances, cores, pre-bridge components, bridge edges and final outputs. Separate ordinary fixtures from traversal/duplicate/near-tie fixtures. Include D<=30 and D>30.

**Gate:** reviewable semantics table and reproducible fixtures establish effective width, self/padding behavior, precision, dense neighbor shape and bridging. Confirm the proposed public dtype and stricter validation divergences before implementation.

### 6.1 — Backend-free graph Kruskal and bridging

In `fast-hdbscan-core`, add a focused Kruskal module using the existing rank/path-compression DSU and sorted-MST/extraction types. Accept validated f64-weighted edges; preserve construction ordinal during stable selection; implement deterministic component bridging. Do not reuse Borůvka's initialization or per-component candidate-merging algorithm.

**Gate:** hand-constructed connected, disconnected, empty-edge and all-infinite graphs; asymmetric/reciprocal graph entries; equal weights; malformed endpoints/NaN rejection; cycle/count/connectivity invariants. Small independent exhaustive or alternative MST checks must establish minimum forest weight without using production Kruskal as its own oracle. Host extraction tests consume supplied bridged trees before device integration.

### 6.2 — Integer-k estimator path on CPU/CUDA

In `fast-hdbscan`, separate exact-neighbor acquisition from MST construction. Reuse exact KD-tree and blocked brute-force searches with the larger graph width; maintain a small exposed-neighbor view separately from graph storage. Add algorithm dispatch after shared validation/finite splitting, before Borůvka-only allocations. Construct MRD edges and perform sorting/DSU on the host, then reuse extraction/remapping.

Keep HDBSCAN-specific operations in the facade and its `HdbscanOps` implementation. The core remains backend-free. Do not change common's UMAP sentinel/overflow semantics or import UMAP ANN operations.

Check products, index conversions, padded outputs, graph-edge count, sorting scratch, host bytes and per-buffer device limits before allocation. O(m * max(k,s)) graph storage is expected; no hidden dense matrix.

**Gate:** KD/brute exact-neighbor equivalence and CPU/CUDA parity against integer-k fixtures; no Borůvka regression. A disconnected small-k fixture must demonstrate the exact bridge rule. A complete-neighbor fixture should recover complete-graph connectivity/MST weights within branch-appropriate numerical tolerance, not promise identical tied topology.

### 6.3 — Dense exact Kruskal

Add the f64 pair-distance path and complete upper-triangle MRD edge construction. Tile device computation to bound scratch; retain the unavoidable O(m²) host graph and O(m² log m) sort cost. Avoid materializing a second full square matrix merely to enumerate edges. Define and document a deterministic resource-admission policy covering edges plus sort scratch; reject unsupported sizes with typed allocation/overflow errors before work begins. No arbitrary silent reduction of k or algorithm replacement.

Keep dense distances/cores separate from f32 kNN assumptions. Do not force dense mode through KD neighbor ordering where that would change its f64 core-distance calculation.

**Gate:** direct dense-reference parity on both backends, including large finite f32 coordinates whose squared f32 distance overflows but f64 distance does not. Verify dense neighbor shapes, s>=m, resource rejection and tie classification. Compare ordinary dense results with exact Borůvka as an additional diagnostic, not a bitwise oracle across distinct precision branches.

### 6.4 — Python exposure and release evidence

Thread algorithm/k through the existing PyO3 binding and native argument validation, Python estimator and stubs. Replace only the two obsolete unsupported-option checks; keep all other exclusions. Test direct native calls as well as the Python wrapper so validation cannot be bypassed. Preserve fitted-state ownership and failure behavior.

Update README/API guidance and affected phase/status documentation to explain exact versus graph MSTs, k=None cost, effective graph width, dtype/shape differences, deterministic ties, infinite bridges and D>30 divergence. Retain measured stage/memory evidence on a bounded grid of dimensions, sizes and k values; make no speed/scalability claims without measurements.

**Gate:** end-to-end Rust/Python CPU and required CUDA parity for both modes, UMAP→HDBSCAN composition with graph Kruskal, unchanged default Borůvka and UMAP regressions, backend feature matrix, formatting/lint/type checks. No hardware skip can satisfy CUDA acceptance.

## Acceptance matrix

- Ordinary non-tied data: labels equal up to permutation (ARI=1), unchanged noise mask, probability tolerance 1e-5, MST/core tolerances explicitly appropriate to each precision branch; compare infinity by masks rather than subtraction.
- Tied/duplicate data: deterministic repeated results and CPU/CUDA agreement; compare equal-weight threshold connectivity and annotate upstream decision differences. Do not inherit Phase 5's fixture-specific `moons_leaf` relaxation or broadly relax numerical tolerances.
- Edge cases: one/two finite rows; all non-finite input; mixed finite remapping; s=1, s=m-1, s=m and s>m; k<s, k=s, k>s and k>=m; duplicates, zero distances, overflowed squared distances, disconnected graphs and all-infinite MRD.
- Bridge/extraction behavior: exactly c-1 correctly anchored bridges, m-1 completed edges, valid trees, finite probabilities in [0,1], deterministic noise behavior and existing DBSCAN cuts including zero, finite, +inf and NaN epsilon.
- Errors: zero/negative/noninteger/bool k, unknown algorithm, k with Borůvka, unsupported options, checked resource/index failures and unavailable explicit backend.

## Verification commands for implementation

Existing command patterns below are future acceptance work, not commands run for this planning task. Add focused test targets within these suites as each milestone lands.

```sh
cargo test -p fast-hdbscan-core --features test-support
cargo test -p fast-hdbscan --no-default-features --features cpu
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --no-default-features --features cuda
uv run maturin develop --release --no-default-features --features cpu,cuda
FAST_UMAP_TEST_BACKEND=cpu uv run --no-sync pytest python/tests/test_hdbscan.py python/tests/test_standard_*.py
FAST_UMAP_TEST_BACKEND=cuda FAST_UMAP_REQUIRE_CUDA=1 uv run --no-sync pytest python/tests/test_hdbscan.py
bash scripts/test_features.sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features fast-umap/cpu,fast-hdbscan/cpu,fast-umap-python/cpu
uv run ruff check
uv run ruff format --check
uv run ty check
```

Retain fixture provenance, actual command results, benchmark/resource evidence and unverified hardware lanes with completion evidence. Later follow-ups remain separate: sample weights first requires resolving upstream weighted-distance semantics; precomputed/constraints and other deferred features do not enter this phase.

## Sources and architectural review

- `hdbscan_plan_phases.md`: follow-up ordering, cross-phase invariants and acceptance requirements.
- `hdbscan_impl.md`: deferred Kruskal intent and deliberate high-dimensional divergence.
- Installed release source: `.venv/lib/python3.14/site-packages/fast_hdbscan/kruskal.py`, especially `_build_mrd_edges_knn`, `_finish_mst`, `_kruskal_mst_brute` and `_kruskal_mst_knn`; `precomputed.py:308-357`; `hdbscan.py` algorithm dispatch.
- Existing core configuration/MST helpers, facade estimator/backend operations, Python wrapper/binding and fixture generator were inspected for reuse boundaries.
- A read-only architect assessment independently checked upstream semantics and identified dense-mode scope, precision, duplicate self removal and k validation as the primary design risks. Those findings are incorporated above; no product source was changed and no implementation gates were run.
