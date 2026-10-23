# Plan: upstream `parallel_boruvka` gaps — D>30 kNN-graph route and `reproducible=False`

Status: **plan only, revision 5** (updates the build/runtime baseline after backend feature
removal; revision 4 adds the device winner-reduction design in §4.8). Architect review scope
is recorded in §9. No product source has been changed **for this plan**. Every kernel or Rust
fragment below is **pseudocode: not compiled and not run on
any backend**, unless a line explicitly says otherwise.

Build baseline: CPU (CubeCL LLVM) and CUDA are now unconditional capabilities of common,
both facades, and the Python extension, including `--no-default-features` builds. Repository
`cpu`/`cuda` Cargo features and caller-permission arguments no longer exist. Upstream backend
dependency features remain enabled. This completed baseline change does not implement any
Borůvka/ANN work proposed here. Optional `tracing`, UMAP `cubecl-ir-dump`, and core
`test-support` remain available.

Backend coverage is selected at runtime: deterministic CPU fixtures explicitly request CPU;
CUDA cases discover availability before device construction and honor `FAST_UMAP_REQUIRE_CUDA=1`.
Explicit CPU must not probe CUDA; Auto prefers CUDA and uses CPU only on confirmed absence;
discovery errors propagate and execution failures never trigger fallback. Shared runtime calls
use `selected_device(choice)` and `validate_backend_choice(choice)`, without permission flags.

## 0. Goal and scope

This plan closes the two remaining differences between this port and the public euclidean
`fast_hdbscan.HDBSCAN`. The installed reference is `fast_hdbscan` 0.3.2 with `pynndescent` 0.6.0,
under `.venv/lib/python3.14/site-packages/`.

1. **D>30 route.** Upstream `compute_minimum_spanning_tree` (`hdbscan.py:412-428`) sends every
   unweighted euclidean fit with `data.shape[1] > 30` to `nndescent.compute_mst_from_knn_graph`.
   That check runs *before* the `algorithm` branch, so neither Borůvka nor Kruskal runs on the
   complete graph above 30 dimensions. The port always runs exact Borůvka. This divergence is
   deliberate and recorded in `hdbscan_impl.md`, `hdbscan_phase6_plan.md:23` and
   `python/tests/test_hdbscan.py:26-36`.
2. **`reproducible=False`.** Upstream `parallel_boruvka(..., reproducible=False)` lets threads
   race on a shared per-component bound (`boruvka.py:154-287`: pruning rules 1a/1b/1c and the
   leaf guard `core[idx] < component_best`). The port accepts `reproducible` in Python, then drops
   it (`crates/fast-umap-python/src/hdbscan.rs:215` binds `_reproducible`).

Out of scope:
- Complete-graph Kruskal, both dense and `knn_k` at d≤30 (Phase 6, `hdbscan_phase6_plan.md`).
- Sample weights.
- Non-euclidean and precomputed metrics.
- Constraints.
- Semi-supervised fitting.

This plan **owns** the shared kNN-graph Kruskal primitive and the configuration fields that
Phase 6 then extends (§2, §3.2). It supersedes three Phase 6 decisions (§8).

## 1. Upstream semantics to reproduce

### 1.1 D>30 route (`nndescent.py:179-285`, `kruskal.py`, `precomputed.py:308-357`)

`s` is the resolved `min_samples`. `m` is the number of finite rows.

| Step | Upstream | Port decision |
| --- | --- | --- |
| Trigger | `metric=="euclidean" and d>30 and sample_weights is None`, regardless of `algorithm` | Same rule in `MstGraph::Auto` resolution (§3.2). The port has no weights, so the trigger is `d>30` |
| Graph width | `knn_k = max(3s, 15)` if `None`; then `knn_k = max(knn_k, s)` | Same formula. Effective `k_graph = min(knn_k, m-1)` |
| `knn_k` validation | None: 0 or negative values silently become `s` | Positive integer or `None`. This is stricter; documented divergence |
| Neighbors | `NNDescent(n_neighbors=knn_k+1, random_state=None)`, then **column 0 stripped positionally** (`nndescent.py:89-90`) | Exact non-self neighbors for `m <= 4096`; relocated CubeCL ANN above that, with fixed seed 42. The actual query id is excluded (see divergences) |
| Core distance | `0` if `s<=1`; `knn_d[:, s-1]` if `s<=knn_k`; else `inf` | Same. `s >= m` gives `inf` (existing exact-path rule) |
| Edges | `_build_mrd_edges_knn`: row-major `(i, j, max(core_i, core_j, d_ij))` in f64 | Same, row-major `(i, slot)`. f64 MRD after widening squared f32 → f64 → `sqrt`, as Phase 6 contracts (`hdbscan_phase6_plan.md:39`) |
| Filtering | `valid = (row != col) & isfinite(weight)` (`nndescent.py:281-285`) | Same: drop self, padding and **non-finite** edges before sorting |
| MST | `_kruskal_core`: stable argsort by weight, union-by-rank DSU with path compression, stop at `n-1` | Stable sort by `(weight, edge ordinal)`, then `RankDsu` |
| Disconnected | `bridge_forest_with_inf`: smallest node id per component, sorted, consecutive pairs joined with `+inf` | Same: exactly `c-1` bridges, where `c` counts components after filtering |
| `neighbors_` | `knn_indices[:, :s]` | Same width. `-1` padding when `k_graph < s` (existing layout, `estimator.rs:846-858`) |
| Determinism | Nondeterministic (public `HDBSCAN` has no `random_state`) | Deterministic per backend: fixed seed and deterministic sort/tie order |

Documented divergences for this route. Each is recorded in docs and fixture metadata, never
covered by relaxed tolerances:
- **Self removal.** Upstream removes column 0, which can be a duplicate row rather than the
  query itself. The port excludes the actual query id. Duplicate-row cases can therefore differ.
- **Small `m`.** Upstream asks NNDescent for `knn_k+1` neighbors even when `knn_k+1 > m`. That
  relies on NNDescent's internal handling or padding. The port clips to `k_graph = min(knn_k, m-1)`
  and treats `s >= m` as infinite cores.
- **Seed.** Upstream does not seed. The port fixes seed 42 (UMAP's `DEFAULT_RANDOM_SEED`), so it
  matches upstream only in distribution on the ANN source. Exact neighbors (`m <= 4096`) match
  pynndescent wherever pynndescent's graph is exact.
- **`knn_k` validation** is stricter, as listed in the table.

### 1.2 `reproducible=False` (`boruvka.py`)

Upstream keeps one shared `component_nearest_neighbor_dist[c]` per component. It prunes a query
point or a tree node when the node's lower bound, or the point's own core distance, exceeds that
bound. Because the update is racy, **per-point** candidates may be `inf`/`-1` or non-minimal.
Only the **per-component minimum weight** is exact. Reproducible mode (`boruvka.py:405-471`)
processes points in blocks and does not use the cross-thread bound.

Port target: a shared per-component device bound that gets the same pruning benefit while the
**merged MST stays bitwise identical** to reproducible mode. §4.2 proves this. It needs one
ordering fix in `merge_components` (§4.3).

## 2. Shared host primitive: kNN-graph Kruskal (`fast-hdbscan-core`)

New module `crates/fast-hdbscan-core/src/knn_kruskal.rs`. It is backend-free and owned by core,
per `AGENTS.md`. Phase 6 integer-`knn_k` Kruskal reuses it unchanged, and Phase 6 dense mode
reuses only `bridge_forest_with_inf`.

```rust
// PSEUDOCODE — not compiled.
pub struct KnnGraph<'a> {
    pub ids: &'a [i32],      // row-major m × k, non-self, -1 = trailing padding
    pub dist_sq: &'a [f32],  // same layout, squared f32 distances (may be +inf on overflow)
    pub k: usize,
}

/// 1. Validate (InvalidInput on: id out of range, id == row, duplicate id in a row,
///    valid id after -1, NaN or negative distance, length mismatch).
/// 2. core[i] (f64) = 0 if s<=1; sqrt(dist_sq[i*k + s-1] as f64) if s<=k; else +inf.
/// 3. For each (i, slot) in row-major order with a valid id j:
///    w = max(core[i], core[j], sqrt(dist_sq as f64)); keep only if w.is_finite().
/// 4. Stable sort by (w via f64::total_cmp, ordinal i*k+slot); Kruskal with RankDsu
///    (path compression + union by rank), stopping at m-1 edges.
/// 5. bridge_forest_with_inf. Returns exactly m-1 SortedMstEdge (f64 weight), sorted with sort_mst.
pub fn knn_kruskal(graph: KnnGraph<'_>, min_samples: usize) -> Result<KnnMst>;

pub struct KnnMst {
    pub edges: Vec<SortedMstEdge>,
    pub core: Vec<f64>,     // public core_distances_ for this route
    pub bridges: usize,     // c-1, exposed for tests
}

/// Upstream order: smallest node id per DSU component, ascending, consecutive pairs, +inf.
pub fn bridge_forest_with_inf(dsu: &mut RankDsu, edges: &mut Vec<SortedMstEdge>, m: usize);
```

Rules:
- The function never repairs input silently. Every validation failure is a typed
  `HdbscanError`.
- `sort_mst` orders by `(weight, src, dst)` (`linkage.rs:24-25`). Kruskal's acceptance order
  follows the stable `(weight, ordinal)` key, so tie decisions follow upstream's stable argsort
  even though the final sort differs.
- `FittedHDBSCAN::core_distances()` currently returns `&[f32]`. This route stores `core` as f32
  for the public accessor, as the existing routes do, and keeps f64 internally for MRD. The
  Python wrapper already widens to f64.
- Memory: allocations of `m*k` edges are guarded by `checked_allocation_bytes`.
- Test oracle under the `test-support` feature, next to the existing exact MST oracle: an
  O(m²·d) host routine that builds the exact non-self kNN graph and runs a textbook Kruskal with
  the same filtering and bridges. Tests only.

Core unit tests:
- With `k = m-1`, the result equals the exact complete-graph MST weight multiset.
- Disconnected graphs produce `c-1` bridges anchored on min ids.
- All-infinite cores (`s > k`) give 0 Kruskal edges and `m-1` bridges.
- Overflowing squared distances give filtered edges plus bridges.
- Duplicate rows and zero distances are deterministic.
- Padding is handled.
- Every validation error is covered.
- `s ∈ {1, m-1, m, >m}`.

## 3. D>30 route

### 3.1 ANN relocation (`fast-cubecl-common`)

`AGENTS.md` forbids `fast-hdbscan → fast-umap`, and ANN lives only in UMAP:
- `crates/fast-umap/src/kernels/standard_umap/ann.rs` (272 lines)
- `ann_forest.rs` (391 lines)
- `ann_sampled.rs` (2219 lines)
- `ann_sampled_tests.rs` (551 lines)
- `fast_umap_core::ann::candidate_width`

These move to `crates/fast-cubecl-common/src/kernels/ann/` as a pure relocation:
- **Backend op.** `fn approximate_neighbors(data, k: u32, seed: u64, max_iterations: u32,
  squared: bool) -> (IntTensor, FloatTensor)` joins common's `NeighborOps`
  (`crates/fast-cubecl-common/src/backend.rs`, a `backend_extension` trait). Its `CubeBackend`
  impl stays in the same crate (orphan rule). The signature carries only tensors and scalars,
  because the `backend_extension`/Fusion forwarding cannot carry a host plan object.
- **`squared` flag.** This is a comptime flag on the existing `ann_output` kernel
  (`ann.rs:198-217`). With it set, the kernel writes `squared_distances` unchanged instead of
  `F::sqrt`. UMAP passes `false`.
- **Preflight, fallible and public.** `ann_preflight(device, n, features, k) -> Result<AnnPlan,
  AnnError>` covers:
  - the shape asserts (`n>=2`, `features>0`, `0<k<n`, `n <= i32::MAX`);
  - `tree_count`;
  - `plan_workspace` and the u32/i32 slot-range asserts that follow it (today `.expect`/`assert!`
    at `ann_sampled.rs:199-216`);
  - the cooperative-launch capability checks (`ann.rs:55-58, 70-90, 106-109`, `Plane::Ops`).

  **Boundary (option a).** The op does not receive the `AnnPlan`. Inside the backend impl it
  re-derives the identical plan from `(n, features, k, client)` using the same pure functions
  preflight uses: `tree_count`, `plan_workspace` and the launch-geometry helpers. These are
  deterministic, so after a successful preflight the op's `expect`s are unreachable. A unit test
  asserts that preflight and in-op derivation return equal plans for the same inputs. The asserts
  at `ann_sampled.rs:481-484` check post-readback invariants rather than inputs. They stay as
  documented `unreachable` invariants and are not preflight checks.

  Error mapping in HDBSCAN:
  - arithmetic and workspace overflow → `HdbscanError::Overflow`;
  - missing device capability (`Plane::Ops`, launch limits) → `HdbscanError::BackendUnavailable`;
  - shape problems → `InvalidInput`.

  These are the existing variants in `crates/fast-hdbscan-core/src/error.rs:8-14`. UMAP calls the
  same preflight and maps into its own `UmapError` contract.
- **Supporting moves.**
  - Move `candidate_width`/`HEAP_MARGIN` into common and delete `fast-umap-core/src/ann.rs`.
  - Add `rayon = { workspace = true }` to `fast-cubecl-common` (`ann_forest` needs it).
  - Point the moved tests' `crate::cuda_backend` use (`ann_sampled_tests.rs:487`) at common's
    CUDA discovery (`fast_cubecl_common::cuda_backend`/`device`).
- **UMAP callsites.**
  - Remove `StandardUmapOps::approximate_neighbors` (`crates/fast-umap/src/backend.rs:26`,
    `kernels/mod.rs:7-13`, `kernels/standard_umap/mod.rs:6`).
  - Point `standard_umap/estimator.rs:182-194` at common.
  - Update `examples/ann_construction_worker.rs`, `tests/test_standard_ann.rs` and
    `tests/test_standard_quality.rs`.
- **Docs.** Update `AGENTS.md`: common owns exact **and** approximate neighbors, and UMAP keeps
  query, fuzzy graph and layout in `StandardUmapOps`.

**Gate M2.**
- UMAP outputs are bitwise unchanged on CPU and on required CUDA: `test_standard_ann`,
  `test_standard_quality`, the estimator tests, and Python `test_standard_*.py`.
- The moved `ann_sampled_tests` pass in common.
- A new test checks that `squared=true` output equals `squared=false` output squared within f32
  rounding, with zero and overflowed rows exact.
- Preflight error tests cover tiny `n`, `k >= n`, and simulated capability absence through the
  `_with_limits` variants (`ann.rs:36`).

### 3.2 Configuration and route selection

To keep a single home for these choices, **all MST-shaping options live in `HdbscanConfig`**
(`crates/fast-hdbscan-core/src/config.rs`). `HdbscanSearch` keeps its current meaning: the
exact-neighbor/candidate implementation of the complete-graph route.

```rust
// PSEUDOCODE — not compiled.
pub enum MstGraph {
    /// Upstream dispatch: KnnGraph when n_features > 30, otherwise Complete.
    #[default] Auto,
    /// Complete mutual-reachability graph (today's exact Borůvka). Rust-only opt-in at d>30.
    Complete,
    /// kNN-subgraph Kruskal (§2) + bridges, at any dimension.
    KnnGraph,
}
pub struct HdbscanConfig {
    /* existing fields … */
    pub mst_graph: MstGraph,      // default Auto
    pub knn_k: Option<usize>,     // default None; validate(): Some(0) → InvalidConfig
    pub reproducible: bool,       // default false (upstream default); §4
}
```

Resolution happens in the estimator, next to `HdbscanSearch::resolve` (`estimator.rs:40-69`):
- `Auto` with `n_features > 30` resolves to `KnnGraph`.
- `Auto` otherwise resolves to `Complete`.
- Explicit values are used as given.
- `HdbscanSearch` resolution runs only for `Complete`. Its thresholds are unchanged.
- `knn_k` is consumed only by `KnnGraph`. Under `Complete` it is accepted and ignored, as
  upstream ignores it in `parallel_boruvka` (§8 supersedes Phase 6's error rule).
- `reproducible` is consumed only by the `Complete` + `KdTree` route (§4.4).

Phase 6 extends this design rather than adding a parallel enum:
- It adds `algorithm: MstAlgorithm { Boruvka, Kruskal }` for the `Complete` graph.
- Upstream's d>30 rule still resolves `Auto` to `KnnGraph` whatever the algorithm.
- An explicit `KnnGraph` at d≤30 is exactly upstream's `algorithm="kruskal", knn_k=<int>` path,
  apart from KD vs brute-force exact neighbors, which give identical neighbor sets under
  deterministic ties.

`FittedHDBSCAN` gains `mst_graph() -> MstGraph`, which reports the resolved value. `search()`
stays meaningful only for `Complete`.

Neighbor source inside `KnnGraph`. `KNN_GRAPH_EXACT_MAX_ROWS = 4096` is the same value as UMAP's
measured `AUTO_EXACT_MAX_ROWS`; re-measure at HDBSCAN widths before tuning it.
The neighbor source is chosen deterministically. It never changes the `KnnGraph` semantics, only
whether neighbors come from exact search or ANN. `ANN_VALIDATED_MAX_WIDTH = 60` comes from
`ann_forest::MAX_CANDIDATES = 60`. ANN recall at widths near or above 60 is unmeasured; UMAP gates
only k=14 and k=24 (`test_standard_quality.rs:665, 954-975`).
- **Exact source** when `m <= 4096` **or** `k_graph > ANN_VALIDATED_MAX_WIDTH`. Generalise
  `core_distances()` (`estimator.rs:331-461`) to take the neighbor width separately from
  `min_samples`; today it uses `k = min_samples.min(n-1)`. Use the brute-force branch with width
  `k_graph`. It is batched (`NEIGHBOR_BATCH_ROWS`), memory-checked, and costs O(m²·d) compute,
  which is the same order as today's exact Borůvka at these sizes. Fits that succeed today
  therefore keep succeeding, for example default `s >= 21` with m > 4096 and d > 30.
  Document this cost.
- **ANN source** otherwise, meaning `m > 4096` and `k_graph <= 60`. Use `ann_preflight` and then
  common ANN with seed 42, `iterations = round(log2(m)).max(5)` (UMAP's formula) and
  `squared=true`. ANN output goes through the same `knn_kruskal` validation, so `-1`/`-2` markers
  and duplicates become typed errors, never silent repairs.
- The M4 campaign (§3.3) may raise `ANN_VALIDATED_MAX_WIDTH`, but only to a measured value.

Fit flow for `KnnGraph`:

```text
// PSEUDOCODE — host orchestration, not compiled.
m == 1 → existing one-row degenerate result (unchanged)
k_graph = min(max(knn_k.unwrap_or(max(3s, 15)), s), m-1)
(ids, dist_sq) = exact_neighbors(width=k_graph)            if m <= 4096 || k_graph > 60
               = ann(preflight?, k_graph, seed=42, sq=true) otherwise
KnnMst{edges, core, ..} = fast_hdbscan_core::knn_kruskal(KnnGraph{ids, dist_sq, k_graph}, s)?
→ existing linkage / condense / select on `edges`
public neighbors = first min(s, k_graph) columns, -1 padded to s
```

This route has no Borůvka loop and uploads no KD tree.

### 3.3 Tests for the route

Rust tests (`crates/fast-hdbscan/tests/test_estimator.rs`) share their assertions between CPU and
required CUDA.

Resolution and configuration:
- `d=30` gives `Complete` and `d=31` gives `KnnGraph`.
- Explicit values are honoured.
- `knn_k = Some(0)` returns `InvalidConfig`.

Exact-source `KnnGraph`, compared bitwise (edges, cores, labels, bridges) with the core
`test-support` oracle, on these cases:
- disconnected graphs;
- `knn_k` below, equal to and above `s`;
- `k_graph` clipped to `m-1`;
- `s ∈ {1, m-1, m, >m}`;
- duplicate rows;
- finite coordinates whose squared distance overflows;
- a one-row and a two-row finite input.

ANN source (m > 4096), on structured data only (50-d and 40-d Gaussian blobs, 6000 rows):
- Repeated fits on the same backend are bitwise identical.
- Recall against exact neighbors is at least the threshold of UMAP's ANN quality test.
- ARI against the exact-source `KnnGraph` labels is at least 0.95.
- Bridge count equals the exact-source count.
- The sum of **finite** MST weights is within 1% of the exact-source value.

Run these gates at `k_graph ∈ {15, 45, 60}` and record the memory and timing of each.
Source selection test: with `s = 25` (default `knn_k` gives `k_graph = 75 > 60`), m > 4096 and
d > 30, the fit succeeds on the exact source. Its result equals the exact-source oracle on the
same data, and the ANN entry point is not called (checked through a test-visible resolved-source
accessor).
CPU-vs-CUDA comparison uses the same quality gates. No bitwise equality across backends is
claimed for ANN.

`blobs_d50` (n=48) stays as the Rust exact Borůvka gate through
`HdbscanConfig { mst_graph: MstGraph::Complete, .. }`.

Fixtures (`crates/fast-hdbscan/tests/fixtures/generate_fast_hdbscan.py`):
- Add case `blobs_d50_knn`, built with `nndescent.compute_mst_from_knn_graph(x, min_samples,
  knn_k=None, random_state=42)` followed by the generator's existing cluster extraction.
- That helper returns only `neighbors[:, :s]` (`nndescent.py:237-255`), so the generator
  separately calls `nndescent.build_knn_graph(x, knn_k, random_state=42)` with the same resolved
  `knn_k`. It must **assert** that all `knn_k` columns, after column-0 stripping, equal a numpy
  exact non-self kNN graph, and that the input has no duplicate rows. At n=48 this is expected.
  If the assertion fails, the generator fails instead of writing an approximate oracle.
- Record the `nndescent.py`, `kruskal.py` and `precomputed.py` blob ids and the `pynndescent`
  version in `manifest.json` provenance.
- Extend `schema.json` and `validate_manifest` (`generate_fast_hdbscan.py:562`) for the new
  entrypoint and `knn_k` parameter.

Python tests (`python/tests/test_hdbscan.py`):
- Replace `test_known_upstream_divergences_keep_exact_fixture_as_gate[blobs_d50]` with a strict
  gate against `blobs_d50_knn`: ARI=1, probabilities within 1e-5, MST weights within 1e-5, cores
  within rtol 1e-6. Keep the informational upstream comparison.
- Remove `blobs_d50` from `KNOWN_UPSTREAM_DIVERGENCES` and from the `PARITY_CASES` exclusion
  (lines 18-36). Keep `moons_leaf`.
- `test_cuda_and_cpu_match_on_phase_one_fixtures[blobs_d50]` keeps comparing CPU with CUDA, now on
  the `KnnGraph` path.
- `test_hdbscan.py:388`: `({"knn_k": 3}, NotImplementedError)` becomes accepted. Add `ValueError`
  cases for `0`, negative values, `True` and `2.0` in both the wrapper and a direct native
  `_fast_umap.fit_hdbscan` call.

## 4. `reproducible=False`: shared per-component bound

### 4.1 Current code facts

- `kd_candidates_kernel` (`crates/fast-hdbscan/src/kernels/kdtree.rs:59-173`):
  - does a per-row stack traversal;
  - prunes nodes by `lower <= best && node_components[node] != component`;
  - picks the lexicographic `(weight, candidate index)`;
  - writes `best_index = n` when nothing is found.
- `merge_components` (`fast-hdbscan-core/src/mst.rs:78-145`):
  - **skips `j < 0` before creating the component slot** (lines 104-107, 113);
  - errors on `j >= n`;
  - keeps the first source with strictly smaller weight;
  - unions winners in slot-creation order;
  - already returns `Numerical("Boruvka round made no progress")` when nothing merges (140-144).
- The host loop (`estimator.rs:463-570`) uploads fresh `point_components` and `node_components`
  every round and reads candidates back.

### 4.2 Invariant that keeps results identical

Let `w*(c)` be the true minimum outgoing MRD weight of component `c`, as computed **by the
candidate kernel's own arithmetic**. Each round, the bound `B(c)` is reset to `+inf`. After that,
`B(c)` only ever takes values that some row of `c` actually computed with that same kernel
arithmetic (§4.4). So `B(c) >= w*(c)` holds at every instant, under any interleaving and any
stale read.

Take a row `i` in `c` whose own true minimum `(w*(c), j)` is under the reproducible rule. Every
node on the path to `j` has `lower <= w*(c) <= B(c)`, and `core_i <= w*(c) <= B(c)`. So
**non-strict** pruning, which means
- skip the row only when `core_i > B`,
- keep a node when `lower <= B`,
- accept a candidate when `weight <= B` under the existing `(weight, index)` rule,

never discards a node that row needs. The row therefore produces the same `(weight, index)` as in
reproducible mode. A row whose own minimum is above `w*(c)` can only report a value `> w*(c)` or
`(inf, -1)`.

A strict `<` (upstream's `core[idx] < component_best`) would drop rows tied at `w*(c)` and change
the lowest-source winner, so this plan rejects it.

Publication hazard: only the atomic word itself is shared. No dependent non-atomic data is
published through it, so the skill's "atomics and publication" hazard does not apply. A stale
read only weakens pruning.

### 4.3 Required `merge_components` ordering fix (core)

In shared mode, the lowest row of a component may be pruned to `(inf, -1)`. Today that row
creates no slot, which changes slot order. That reorders unions, flips mutual-winner orientation,
and changes which equal-weight cross-component edge survives. The fix is to call
`slots.entry(point_components[i])` **before** the `j < 0` skip, inserting a placeholder winner
`(i, u32::MAX, +inf)` that has an explicit `has_candidate = false` flag. The first real candidate
(`j >= 0`) of that component **always** replaces a placeholder, even at weight `+inf`. After that,
the existing strict-`<` rule applies. Placeholders that survive are skipped at union time. The
placeholder must not compete on weight: an `+inf` real candidate would never beat an `+inf`
placeholder under `<`.

Reproducible mode is unchanged: every row there has a candidate, so slot order already follows the
first row index.

Core unit tests:
- A component whose lowest row is `(inf, -1)` produces the same edges, orientation and union
  order as the all-candidates input.
- An all-placeholder component is skipped.
- The existing no-progress error is preserved.

### 4.4 Bound encoding and verified CubeCL 0.11.1 API

`Cargo.lock` pins `cubecl`, `cubecl-core`, `cubecl-cpu` and `cubecl-cuda` at 0.11.1, Burn at
0.22.1 and CubeK at 0.3.1. The skill's `examples/` crate uses 0.11.2. The APIs below were read
from the 0.11.1 registry sources; they were read, not compiled, for this plan.
- **Integer atomics.** `cubecl-core-0.11.1/src/frontend/element/atomic.rs:53-55, 71-130`:
  `Atomic<Inner>` provides `load`, `store`, `exchange` and `fetch_{add,sub,min,max}`. `i32`
  `fetch_min` maps to `AtomicSMinOp` and `u32` to `AtomicUMinOp`. `compare_exchange_weak` for
  `Int` is at 132-147.
- **CUDA lowering.** `cubecl-cpp-0.11.1/src/shared/atomic.rs:126-139` lowers S/U min to
  `atomicMin`. `AtomicLoadOp` lowers to a volatile plain load, which is sufficient for a monotone
  bound.
- **Repo precedent, compiled and tested on CPU and CUDA today.** `Atomic<I: Int>::fetch_min` at
  `crates/fast-umap/src/kernels/standard_umap/graph.rs:841-844`, and `fetch_max` at `query.rs:61`.
- **`Reinterpret::reinterpret::<From>`** (`cast.rs:60-90`): a bit cast that asserts equal sizes.
  The KD route already enforces f32 data and i32 indices (`kdtree.rs:410-423`). **Nothing in this
  repo uses it yet**, so the M0 spike must compile and run it on both runtimes.
- **`Atomic<f32>::fetch_min`** (`AtomicFMinOp`) lowers to a float `atomicMin`, which CUDA does
  not provide natively. It is **not** a fallback. If `reinterpret` fails in M0, the fallback is
  `compare_exchange_weak` on `Atomic<u32>`, using bits produced by a host-agnostic encoding.
  That fallback also needs a bit cast, so if M0 fails entirely, M5 stops and reports. It does not
  ship a no-op.

Encoding:
- Candidate weights are `max(distance_sq, core_i, core_j)`, built from sums of squares that
  start at `+0.0` and from non-negative cores. Their IEEE bit patterns therefore order like
  non-negative `i32`, and `+inf = 0x7F800000` stays positive.
- Canonicalise with `weight + 0.0` to rule out `-0.0`.
- NaN is impossible: rows are validated finite, and `merge_components` rejects NaN.
- `component_bound` is a `Tensor<Atomic<I>>` of length `n`, indexed by the DSU root ids already in
  `point_components`.
- **The whole tensor is rewritten to `+inf` bits every round**, using a device fill before the
  kernel launch. DSU root ids are reused across rounds, so stale entries would break §4.2.

Seeding `B(c)` from retained kNN distances would prune more in round 1. The exact-neighbor
kernels compute distances with different code than `kd_candidates_kernel` (`kdtree.rs:112-119`),
so a seed 1 ulp below the kernel's `w*` would prune the true winner. This plan therefore does
**not** seed. Seeding may be proposed later only together with a bitwise distance contract and a
CPU/CUDA test showing `seed >= kernel w*`.

**Missing primitive, named explicitly:** none for the bound itself. Replacing the full per-round
candidate readback needs a device-side per-component winner reduction: a segmented argmin with
first-source tie order. CubeK has ordinary tensor-axis argmin, but not this ragged-component
operation with candidate validation and ordered host merging. §4.8 now plans that primitive,
superseding the earlier exclusion (`hdbscan_plan_phases_completed.md:134`). The DSU, union order
and a smaller per-round winner readback remain on the host; this is not device-side Borůvka.

### 4.5 Kernel change (KD route)

```rust
// PSEUDOCODE — NOT COMPILED, NOT RUN on any backend.
#[cube(launch)]
fn kd_candidates_kernel<F: Float, I: Int>(
    /* existing args … */
    component_bound: &mut Tensor<Atomic<I>>,   // NEW, len n, i32 bits of non-negative f32
    #[comptime] shared_bound: bool,             // NEW
    /* outputs, diagnostics, dtypes … */
) {
    // per row, after `component` and `core` are read:
    let mut bound = F::new(f32::INFINITY);
    let mut active = true;
    if comptime![shared_bound] {
        bound = F::reinterpret(component_bound[component as usize].load());
        active = core <= bound;                     // rule 1b, non-strict
    }
    // traversal runs only when `active`; stack/bound arrays as today.
    // node pop:   keep = lower <= best && node_components[node] != component
    //             if shared_bound { bound = min(bound, reinterpret(load())); keep &= lower <= bound }
    // child push: unchanged `child_bound <= best`, plus `child_bound <= bound` if shared_bound.
    // leaf:       candidate (weight, index) rule unchanged. If shared_bound and the row's
    //             `best` improved and best < bound:
    //               let old = F::reinterpret(component_bound[c].fetch_min(I::reinterpret(best + 0.0)));
    //               bound = F::min(old, best);
    // output:     shared_bound && best_index == n → (inf, -1); otherwise as today.
}
```

The `best <= bound` acceptance is implied: `best` only decreases, and nodes with `lower > bound`
are never opened. The candidate rule is the unmodified `weight < best || (weight == best &&
candidate < best_index)`, so tie order is untouched.

Reproducible mode (`shared_bound = false`) binds a one-element dummy bound tensor. The gate is
**bitwise output equality** with the current kernel on all fixtures and the random tie sets in
§4.7. No claim about identical IR is made.

Contention: there is one atomic word per component, and it is written only when a row strictly
improves below its cached bound. Early rounds have about `n` components, so contention is low.
Late rounds have few components and many writers, but only the first improvements land. Measure
before adding per-cube pre-reduction.

Brute-force route (`kernels/boruvka.rs` `tile_candidates`/`reduce_tiles`): unchanged. Upstream's
flag exists only for its KD traversal. An exhaustive tile scan has nothing to prune, and a row
skip would need a non-`+inf` seed, which §4.4 excludes. This decision is documented, not hidden.

### 4.6 API plumbing

- **Rust.** `HdbscanConfig::reproducible` (default `false`, the upstream default; §3.2) is passed
  to `HdbscanOps::boruvka_candidates_kd` as the comptime flag. `KnnGraph` and the brute-force
  `Complete` route ignore it, as upstream's `compute_mst_from_knn_graph` does. The default stays
  `false` because results are bitwise identical by construction (§4.2, enforced by §4.7). The M5
  benchmark is evidence about performance, not about the default.
- **Native binding.**
  - `crates/fast-umap-python/src/hdbscan.rs` adds `knn_k` to the kwargs allowlist (lines
    159-160).
  - It forwards `reproducible` (line 215) and `knn_k` into `HdbscanConfig`.
  - It validates `knn_k` natively: a positive int or `None`, `bool` rejected, mirroring
    `_positive_integer`.
- **Python wrapper.** `python/fast_umap/hdbscan.py` removes `knn_k` from the unsupported tuple
  (line 206), validates it with `_positive_integer(..., allow_none=True)`, and passes it to
  `fit_hdbscan`. Stubs and docstrings are updated.
- **Python test.** `test_hdbscan.py:594` `test_reproducible_option_is_accepted_without_changing_results`
  gets stricter. It uses a dataset in the KD Auto region (256–2048 rows, d ≤ 30, s ≤ 32) and
  asserts that MST, labels, probabilities and cores are bitwise equal for `True` and `False` on
  CPU and CUDA.

### 4.7 Tests for the bound

These assertions are shared between CPU and CUDA (`crates/fast-hdbscan/tests/test_kdtree_search.rs`
and `test_candidates.rs`).

MST bitwise equality:
- MST edges `(src, dst, weight)` are bitwise equal with `reproducible` true and false.
- Fixtures: `blobs_d2`, `blobs_d10`, `duplicates_ties`, `persistence_equal`, `varying_density`.
- Seeded random tie-heavy sets: integer grid points and duplicated rows.

Per round, through the backend op directly:
- For every component, the winner `(i, j, w)` chosen by `merge_components` is identical in both
  modes.
- Every non-winner row reports `(inf, -1)` or a weight `>= w*(c)`.
- The bound tensor after the launch equals the per-component minimum of the reported weights.

Primitives:
- A minimal kernel test of `reinterpret` round trips and atomic-min ordering on
  `{+0.0, -0.0, smallest subnormal, 1.0, f32::MAX, +inf}`, on both runtimes (M0).
- A per-round reset test: stale bounds from a previous round never prune, using a forced
  two-round input whose DSU root ids get reused.

Performance evidence, which is not a correctness gate:
- Criterion KD candidate rounds at the existing Auto KD region sizes, on CPU and CUDA, with
  `reproducible` true and false.
- The existing `diagnostics_enabled` counters (`nodes_popped`, `leaf_positions_scanned`) show the
  pruning.
- The `Auto` thresholds do not change without a new measured campaign.

### 4.8 Device-side component winner reduction (new implementation plan)

#### 4.8.1 Scope and API evidence

Implement the reduction for both complete-graph candidate producers (KD and brute force), in
both reproducibility modes. Do not change candidate arithmetic, destination tie-breaking,
initial kNN merging, final MST sorting, Python options or the kNN-graph Kruskal route. Keep
`RankDsu` and serial union-by-rank on the CPU. The improvement sought is replacing downloads of
`n` weights and `n` destinations with `c` ordered winner/error records, not eliminating round
synchronization. No kernels in this extension have been compiled or run on either backend.

Read for this extension:
- Book `getting-started/parallel_reduction.md` and its `src/bin/v4-gpu.rs`: each unit owns a row
  and loops over its columns. This parallelizes **across rows**, not within a long row. Mapping
  one unit to one component is suitable for small segments, but insufficient for late rounds
  with only a few large components.
- Book `vectorized_reduction.md` and `src/bin/v5-gpu.rs`: `Tensor<Vector<F, N>>`, a vector-width
  launch argument after `CubeDim`, scalar-element strides divided by `N`, and a final lane
  reduction. These examples assume rectangular, divisible rows; neither solves ragged segments
  or preserves an argmin's source payload merely by calling `vector_sum()`.
- Skill `examples/src/portable.rs::cube_partial_sums` supplies the shared-memory tree pattern;
  `examples/src/segmented.rs` supplies multi-dispatch segmented composition. They target
  0.11.2, not this project's 0.11.1: use their decomposition, not unchecked copied signatures.
- Pinned registry `cubecl-core-0.11.1/src/runtime_tests/launch.rs:54-78,98-107` demonstrates
  slice launch arguments, `Shared::new_slice`, `sync_cube`, `BufferArg` and generated launches;
  `runtime_tests/barrier.rs:9-45` demonstrates the vector-width launch position.
- Pinned `cubek-reduce-0.3.1/src/lib.rs:96-180` exposes `reduce` and `reduce_with_indices` over
  a tensor **axis**; `components/instructions/mixed.rs` includes `ArgMin`. No ragged offsets
  argument exists there. Padding to `[c, max_component_size]` can cost O(c*n), and launching
  CubeK separately for each component costs O(c) launches. Neither is the chosen pipeline.

**Decision:** host-built CSR membership plus tiled device argmin. The host already computes
every DSU label each round (`estimator.rs:489-495`), so building membership there avoids adding
a device sort/scan subsystem solely to group these labels. Grouping is metadata preparation;
candidate weights and destinations stay on the device until reduced. This pays an additional
O(n) metadata upload, which the end-to-end benchmark must count. Component ids are arbitrary
root labels, not contiguous runs in row order; a head flag on adjacent labels would be wrong.

#### 4.8.2 Exact semantics and associative state

Apply §4.3's slot-order correction **before** using the device reducer as an oracle replacement.
Define segment `s` by the component's first source row, ascending, including components whose
first row has no candidate. Within a component, select the real candidate minimizing
`(weight_sq, original_source_row)`. Do not minimize destination here: destination ties have
already been resolved per source by the candidate kernel.

The reduction state is `(weight_sq: f32, source: i32, first_bad_row: i32)`. Use `source = n`
as the invalid identity, with weight `+inf`; `first_bad_row = n` means no error. A valid real
candidate beats an identity even at `+inf`. Between valid candidates choose smaller weight,
then smaller original source for numeric equality. Preserve the selected weight's original
bits: `-0.0 == +0.0`, so signed zeros tie by source, not `total_cmp` or integer bit order.
Never add, square-root, normalize, or otherwise recompute the selected weight.

Validate **every** row before treating a negative destination as absent, matching
`mst.rs:98-111`: NaN or weight `< 0` is `Numerical`, including on absent rows; otherwise
`destination >= n` is `InvalidInput`; any negative destination means absent. `+inf` is valid.
For erroneous or absent rows contribute no candidate. Independently carry the minimum bad
source row through every reduction. At the final output, inspect that row's original weight
and destination to encode its error category and offending destination. A numerical error has
precedence over an index error on the same row. This preserves the first-error-in-source-order
behavior even when that row would not have won its component.

The existing producers use `n` when no destination exists; this remains an out-of-range error,
not an absent candidate silently accepted by the reducer. Only §4.5's explicitly pruned shared-
bound rows introduce `(inf,-1)`. The reducer's internal identity source `n` is unrelated to
candidate destination validity. Self/same-component destinations retain existing host behavior:
they are not new validation errors, and the ordered DSU merge skips already-connected endpoints.

On validated rows the winner operation is associative: it selects a unique smallest source
among equal minimum weights, and the identity is explicitly invalid. The error operation is
integer minimum, also associative. Thus arbitrary tile/tree order gives the same payload bits
as the corrected serial strict-`<` loop. NaNs never participate in floating-point comparisons.
No atomic float argmin, packed 64-bit atomic key, racing payload publication or device-wide
barrier is required.

#### 4.8.3 Host membership and buffer contracts

Build a backend-free `ComponentReductionPlan` from the current round's host labels. Use a
root-to-slot lookup of length `n`, initialized to missing, and scan sources ascending; first
appearance assigns the slot. Count sizes, prefix-sum on the **host**, and stable-scatter source
ids. Validate internal root labels are in `[0,n)` at this boundary. Do not assume the root is
the component's smallest source, and never iterate a hash map to determine output order.

Buffers (all indices/offsets fit i32 under the existing `n <= i32::MAX` input bound):

| Buffer | Shape / ownership | Invariant |
| --- | --- | --- |
| `members` | i32[n], host-built, uploaded | permutation of original source ids, ascending within each segment |
| `offsets` | i32[c+1], host-built | starts at 0, ends at n; nonempty component segments |
| `tile_starts`, `tile_ends` | i32[p] each, host-built, uploaded | disjoint subranges of members, each within one component and at most T elements |
| `partial_offsets` | i32[c+1], host-built, uploaded | component s owns consecutive partials `[a_s,a_(s+1))` |
| candidate weights / destinations | f32[n] / i32[n], existing producer | indexed by original source id, not member position |
| partial weight / source / bad-row | f32[p], i32[p], i32[p], device scratch | one writer per tile; every logical element overwritten each round |
| winner weights | f32[c], final device output | original weight bits; +inf for absent winner |
| winner metadata | i32[c,5], final device output | `(source, destination, bad_row, error_kind, bad_destination)` |

Use `source = destination = -1` for an absent final winner, and `bad_row = n`, error kind 0
when valid; error kinds 1/2 mean numerical/index failure. Bad destination is meaningful only
for kind 2. Keep `offsets` on the host unless a kernel actually consumes it; device kernels
need tile ranges and partial offsets. With `p = sum_s ceil(size_s/T)`, `c <= p <= n` for
nonempty input. Metadata construction, scratch and processing are O(n+c+p), not O(n*c).
Use checked arithmetic, `checked_allocation_bytes`, and device allocation limits for all
buffers and the combined workspace. Do not allocate a rectangular component-by-point matrix.

#### 4.8.4 Kernels, launch geometry and orchestration

**GPU baseline, vector width 1:**
1. `component_tile_argmin`: one cube per tile. Start with a legal power-of-two cube size B
   (initial tuning point 128, reduce to a supported power of two) and T=4*B. Unit u folds
   member positions `tile_start + u + k*B` within the explicit tile end, gathering candidates
   through `members`. Every unit starts with identity and validates its rows. Reduce the
   three state fields together using three `Shared` arrays and a halving tree. Every unit
   reaches every `sync_cube`; tails supply identity, never skip a barrier. Unit 0 writes the
   partial record. Singleton tiles are valid, if not necessarily efficient.
2. `component_finish_argmin`: one cube per component. Unit u folds partials
   `a_s + u + k*B`, then the same shared tree selects the winner and first bad row. Unit 0
   gathers destination from the winning **original source**, or writes the absent sentinel;
   it also gathers/classifies the first bad row. The dispatch boundary orders partial stores
   before these reads. This pass reads O(p) partials total, not n candidates per component.

If one component has many tiles, the second cube loops over its partials rather than placing
the entire component in one serial unit. Measure this residual skew before adding another
hierarchical level. Stop the existing estimator loop at `c=1`; the standalone reducer still
supports one segment for testing. No launch is needed for `n=c=0`; return correctly shaped
empty outputs without indexing `offsets[1]` or constructing zero-byte raw buffers.

Use `CUBE_POS` as usize and cast stored i32 offsets only after validation. `UNIT_POS` is u32;
cast to usize for indexing. Mutable runtime initializers use `.runtime()`. Respect shared
memory and cube-count limits; batch cube ranges with an explicit task offset, as the existing
Borůvka host launcher does, or use a verified multi-axis cube-count helper with whole-cube
guards for surplus cubes. Never truncate a launch to the hardware cap. Pass explicit logical
counts for reused buffers instead of trusting raw-buffer capacity or `BufferArg` length.

**CPU runtime:** retain the same membership, state and two-pass outputs but use barrier-free
kernels. Launch one cube with at most `num_cpu_cores` units; divide tile ids into contiguous
chunks per unit and fold each tile serially in registers. In the second dispatch divide
component ids similarly and fold their partials serially. Do not launch the GPU's 128-thread
barrier tree on the LLVM CPU runtime. This follows the book's row-per-unit approach without
making a giant late-round segment a single first-pass task. A single large final segment may
still dominate the second CPU pass; benchmark it separately.

**Vectorization:** not a prerequisite for the first correct implementation. CSR membership
loads are contiguous, but candidate loads are indirect gathers. Do not reinterpret unrelated
source ids as contiguous vector candidate loads. A later measured optimization may gather
candidate tuples into contiguous segment storage, then use `Vector<F,N>` and matching source
vectors with an explicit lane-wise pair argmin. It must handle unaligned segment starts,
segments shorter than N and scalar tails; never discard the winning source with a value-only
`vector_min`. Benchmark the extra gather buffer and dispatch against scalar tiled gathering.
No plane operations are required for the baseline, and no fixed plane width is assumed.

Proposed orchestration (algorithm pseudocode, **not compiled or run**):

```text
labels = current host DSU labels                         # already needed by candidate search
plan = ComponentReductionPlan::build(labels, tile_width) # first-appearance component order
upload plan.members, tile ranges, partial_offsets
weights, destinations = existing candidate op(...)      # keep both device-resident
partials = launch component_tile_argmin(..., logical n,p)
winner_weights, winner_metadata = launch component_finish_argmin(..., logical n,c,p)
download c weights and c*5 metadata ints in one Transaction; validate shapes first
select earliest bad_row across ALL outputs; return its typed error before any unions
validate winner source belongs to its slot and both endpoints are in range; do not reject cycles
merge ordered real winners into host RankDsu; preserve orientation and no-progress error
```

The successful path reads `24*c` bytes instead of `8*n`, plus uploads about `4*n + 8*p +
4*(c+1)` bytes of new grouping metadata. This can lose in early rounds or small fits; there
is no guaranteed transfer reduction until `c < n/3`, even before metadata upload is counted.
Buffer allocation reuse must retain actual dispatch dependencies and overwrite logical output
ranges. No candidate `to_data`, `into_data` or host read occurs between the two reduction
dispatches. Host union and next-round label uploads remain synchronization points.

#### 4.8.5 Integration and implementation slices

1. **Core contract and oracle:** in `fast-hdbscan-core/src/mst.rs`, land §4.3; add membership
   planning and a compact `merge_component_winners` contract. Keep a serial strict-`<` reducer
   under `test-support` as the independent oracle, not as an estimator execution fallback.
   Share the ordered union loop so both oracle tests and compact merging preserve the existing
   `dsu.find`, `union_roots`, edge orientation and no-progress error. Validate the complete
   batch before mutating the DSU, as `merge_components` currently does.
2. **Device primitive:** add `fast-hdbscan/src/kernels/component_reduce.rs` and register it in
   `kernels/mod.rs`; implement GPU tiled and CPU barrier-free launchers. In `backend.rs`, add
   a single tensor/scalar-only `HdbscanOps` operation accepting the two candidate tensors,
   membership, tile ranges, partial offsets and logical counts, returning f32[c] and i32[c,5].
   Its Fusion metadata must declare both exact output shapes and dtypes. Keep the extension
   trait and `CubeBackend` implementation in the HDBSCAN facade; do not expose host plans
   through Fusion or move HDBSCAN-specific validation into common.
3. **Estimator wiring:** in `estimator.rs::spanning_tree`, build/upload fresh membership each
   round, feed both candidate routes into the new op, and replace the n-row `Transaction`
   and host candidate reduction with compact readback/merging. Remove the obsolete production
   candidate-reduction path rather than keeping a second selectable algorithm. Split tracing
   into metadata preparation/upload, candidate kernel, component reduction, winner readback
   and host merge; remove the misleading `boruvka_candidates_readback` span name.
4. **Tests and measurements:** add `tests/test_component_reduce.rs`, core tests, and a focused
   benchmark following the existing Criterion conventions. Update only directly affected
   implementation/performance documentation after measurements. Public Rust/Python estimator
   signatures and model formats remain unchanged, so no Python binding or serialization
   changes are needed for this extension.

The membership structure belongs in core because it is backend-free host planning. GPU scratch
and execution policy belong in the facade kernel module. Name and exact signatures are to be
finalized with these contracts in the first slice; no dependency or configuration flag is added.

#### 4.8.6 Acceptance gates and performance evidence

Correctness gates, against the serial core oracle, shared across CPU and required CUDA:
- Exact source, destination, weight **bits**, slot order, emitted oriented edges and final DSU
  roots; equal-weight cycles, mutual winners and union-by-rank ties must not reorder unions.
- Interleaved/nonmonotonic root labels; root different from first member; singleton components;
  all singleton, two huge, one huge plus many tiny components; multi-tile random partitions.
- Equal weights across units/tiles; source tie order different from destination and member
  position order in a deliberately permuted test membership; signed zeros, smallest positive
  subnormal, f32::MAX and valid +inf; all-absent and first-row-absent components.
- Absent destinations of -1 and another negative value; invalid destination n or larger;
  negative/NaN weights including on absent and losing rows. Multiple errors across tiles and
  components return the earliest source's correct category before any DSU or edge mutation.
- Empty and one-element standalone inputs; segment sizes B±1 and T±1, partial last cubes,
  legal small B, forced launch batching, excess scratch capacity, poisoned reused buffers,
  and consecutive rounds with changed membership and reused root ids.
- Both candidate routes through the real estimator; KD shared-bound true/false round winners
  match the corrected host oracle (§4.7), including pruned first rows. Existing CPU/CUDA MST,
  clustering, probability and non-finite-row-remapping fixture tests stay strict.
- Fusion output metadata and dependency execution are exercised by the Dispatch integration
  tests; all logical outputs must be initialized, not merely the rows with real candidates.

Future focused commands (not run for this plan):

```sh
cargo test -p fast-hdbscan-core --features test-support
cargo test -p fast-hdbscan --test test_component_reduce
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_component_reduce
```

Then run the affected existing HDBSCAN suites and build-contract matrix from §7. The new CUDA harness
must honor `FAST_UMAP_REQUIRE_CUDA=1`; a hardware skip is not a passing gate. No speedup is
claimed by this plan, nor by a passing CPU-runtime test.

Measure warm device-only reduction and **complete round / complete fit** wall time, separately
on CPU and CUDA, recording cold JIT separately. Compare against the current host reduction
with the same §4.3 semantics, candidate tensors and DSU state. Include membership construction,
uploads, allocations, readback, synchronization, validation and host union. Sweep n, c/n,
segment skew, both search routes and tied/+inf candidates; include real late-round partitions
from the existing fixtures and profiling datasets, not only uniformly sized synthetic groups.
Report launch count, metadata/candidate/winner bytes, scratch high-water mark and medians with
dispersion. A kernel-only gain is insufficient evidence to claim a faster Borůvka fit. Small
segments wasting a GPU cube, indirect gathers, metadata uploads and the final few-segment pass
are the named bottlenecks to measure before introducing adaptive paths or more primitives.

## 5. Docs to update

- `README.md`, HDBSCAN section.
- `hdbscan_impl.md`: remove the D>30 divergence and the "reproducible ignored" note, and add the
  §1.1 divergences.
- `hdbscan_algo_impl.md:133` (S1).
- `hdbscan_plan_phases.md` (exclusion list).
- `hdbscan_phase6_plan.md`: the §8 supersessions and the `MstGraph`/`MstAlgorithm` layering.
- `AGENTS.md`: ANN ownership, and that `HdbscanConfig` owns MST-shaping options.
- `CHANGELOG.md`.
- Python docstrings for `knn_k` and `reproducible`.

## 6. Milestones

Each milestone ends at a passing gate, and the product works at every step.

| # | Work | Gate |
| --- | --- | --- |
| M0 | Spike: test kernels for `Atomic<i32>` `load`/`fetch_min` with `reinterpret` f32↔i32; if that fails, `compare_exchange_weak` on `Atomic<u32>`; otherwise **stop and report** | Kernel test passes on CPU and on `FAST_UMAP_REQUIRE_CUDA=1` CUDA |
| M1 | §2 `knn_kruskal`, bridges and `test-support` oracle; §4.3 `merge_components` slot-order fix | `cargo test -p fast-hdbscan-core --features test-support` |
| M2 | §3.1 ANN relocation, `squared` output, `ann_preflight`, `AGENTS.md` | UMAP Rust/Python bitwise unchanged on CPU and required CUDA; common tests |
| M3 | §3.2 config fields, `MstGraph` resolution, exact-source `KnnGraph` | §3.3 exact-oracle tests; `blobs_d50` `Complete` gate |
| M4 | ANN source for m > 4096, width campaign at k ∈ {15, 45, 60} | §3.3 ANN gates; the measured width limit is recorded |
| M5 | §4.4–4.5 shared bound on the KD route | §4.7 bitwise equality on CPU and CUDA |
| M6 | Python plumbing, `blobs_d50_knn` fixture and schema, test updates, docs | Python suites on CPU and CUDA, ruff/ty, build-contract matrix |
| M7 | §4.8 core membership/ordered-merge contract and standalone device reduction | §4.8.6 oracle parity and error gates on CPU and required CUDA |
| M8 | §4.8 compact estimator readback, tracing and end-to-end measurements | Both candidate routes and bound modes retain bitwise MST outputs; full transfer/fit cost reported |

Dependencies:
- M0 → M5.
- M1 → M3 and M5.
- M2 → M4.
- M3 → M4.
- M6 needs M0–M5; the reduction extension does not block that earlier deliverable.
- M1 → M7 → M8. M5 → M8 for the shared-bound integration gates.

## 7. Verification commands

These are future acceptance commands for this plan's implementation. Baseline checks run during
backend feature removal do not prove any proposed kernel, relocation, or algorithm here.
Both Rust backend test variants compile into the same artifacts; ordinary runs may skip absent
CUDA hardware, while required-CUDA runs must execute it. Python's CUDA GIL diagnostic requires
the optional `tracing` build below.

```sh
cargo test -p fast-hdbscan-core --features test-support
cargo test -p fast-cubecl-common
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common
cargo test -p fast-hdbscan
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan
cargo test -p fast-umap
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-umap
uv run maturin develop --release --features tracing
FAST_UMAP_TEST_BACKEND=cpu uv run --no-sync pytest python/tests/test_hdbscan.py python/tests/test_standard_*.py
FAST_UMAP_TEST_BACKEND=cuda FAST_UMAP_REQUIRE_CUDA=1 uv run --no-sync pytest python/tests/test_hdbscan.py python/tests/test_standard_*.py
bash scripts/test_features.sh
cargo fmt --all -- --check
cargo check --workspace --all-targets --no-default-features
cargo check --workspace --all-targets --features fast-umap/tracing,fast-hdbscan/tracing,fast-umap-python/tracing
cargo check -p fast-umap --all-targets --features cubecl-ir-dump
cargo clippy --workspace --all-targets
uv run ruff check && uv run ruff format --check && uv run ty check
```

A skipped CUDA lane never satisfies a CUDA gate. A CPU-runtime pass proves the logic, not GPU
performance.

## 8. Supersessions and risks

Supersessions:
- This plan supersedes the D>30 "never auto-dispatch pynndescent" divergence in
  `hdbscan_impl.md`, `hdbscan_plan_phases.md` and `hdbscan_phase6_plan.md:23`.
- It supersedes Phase 6's rule that "non-None k with Borůvka is an error"
  (`hdbscan_phase6_plan.md:30`). Upstream accepts `knn_k` there: it is used above 30 dimensions
  and ignored otherwise.
- It supersedes Phase 6's plan to put a standalone `knn_k` in config (`hdbscan_phase6_plan.md:29`).
  `knn_k` and `MstGraph` are introduced here. Phase 6 adds only `MstAlgorithm` for the `Complete`
  graph, and it imports §2 instead of defining its own kNN Kruskal.

Risks:
- **ANN output differs across backends.** Gates use same-backend determinism plus quality
  thresholds, not bitwise equality across backends.
- **ANN width above 60 is unmeasured.** Those widths use the exact source deterministically
  (O(m²·d)) until M4 measures a higher validated width.
- **`reinterpret` is unsupported on a runtime.** M0 runs first and blocks M5. There is no
  silent no-op.
- **Atomic contention in late rounds erases the gain.** Results are identical either way; only
  performance is affected, and the benchmark documents it.
- **Pynndescent behaviour on duplicate rows or small `m`.** These are documented divergences
  (§1.1), and the fixture generator asserts that the graph is exact before writing an oracle.

## 9. Architect review log

r1 verdict was **REVISE**. Every finding is resolved in this revision:

| # | Sev | Finding | Resolution |
| --- | --- | --- | --- |
| 1 | blocking | `merge_components` slot order depends on pruned rows | §4.3 placeholder slot before the `j<0` skip, plus core tests |
| 2 | major | Upstream filters non-finite MRD edges | §1.1 and §2 step 3 filter `!is_finite` before sorting; oracle cases for inf cores and overflow |
| 3 | major | A kNN seed can sit below the kernel's `w*` | §4.4: per-round `+inf` reset with a full rewrite; seeding excluded unless a bitwise contract exists |
| 4 | major | ANN panics are not pre-validatable | §3.1 fallible `ann_preflight` with typed mapping (the consumed-`AnnPlan` design is superseded by r2 #2) |
| 5 | major | ANN is unvalidated at HDBSCAN widths | §3.3 campaign at k ∈ {15, 45, 60} (the typed `k_graph > 60` error is superseded by r2 #1: exact source) |
| 6 | major | `KnnGraph` inside `HdbscanSearch` conflicts with Phase 6 | §3.2 `MstGraph` + `knn_k` + `reproducible` in `HdbscanConfig`; §8 layering |
| 7 | minor | Ill-defined ANN quality gates | §3.3 finite-edge sums, equal bridge counts, structured data only |
| 8 | minor | Redundant progress guard; IR gate cannot be verified | Existing guard cited (§4.1); bitwise output gate with a dummy binding (§4.5) |
| 9 | minor | Missed callsites | §3.1 `cuda_backend` in moved tests; §3.3 schema/`validate_manifest`, `test_hdbscan.py:388`; §4.6 kwargs allowlist; one default rule |
| 10 | minor | Undocumented upstream divergences | §1.1 divergence list (positional self removal, small `m`) |

r2 verdict was **APPROVE_WITH_CHANGES**. All r1 findings were confirmed resolved, and the r2
findings are applied in revision 3:

| # | Sev | Finding | Resolution |
| --- | --- | --- | --- |
| 1 | major | The ANN width error breaks default large d>30 fits | §3.2: deterministic exact source when `k_graph > 60`, with its O(m²·d) cost documented; §3.3 `s=25` test |
| 2 | minor | Undefined `AnnPlan` boundary; nonexistent `Backend` variant | §3.1 option (a): in-op re-derivation plus a plan-equality test; `BackendUnavailable`; 481-484 reclassified as invariants |
| 3 | minor | Wrong Phase 6 citations | Corrected to :23, :30, :29, :39 |
| 4 | minor | The fixture exact-graph check needs the full graph | §3.3: separate `build_knn_graph` call comparing all `knn_k` columns |

r3 verdict: **APPROVE** (architectural status CLEAR). Its single P3 note, a duplicated preflight
citation and stale r1 rows, is applied above.

Revision 4's §4.8 received a focused read-only Architect review of ordering, validation,
associativity, buffer bounds, CPU/GPU decomposition and integration. It returned no concrete
issues. The existing destination-`n` error contract and absence of new cycle-validation errors
are explicit above. This is design review only, not compilation or backend execution evidence.

Revision 5 updates only the baseline assumptions and future verification commands after the
completed backend-feature removal. It does not change the algorithm designs or extend the
scope of prior Architect approvals. The build-contract matrix is `scripts/test_features.sh`;
there are no CPU-only/CUDA-only compilation lanes or missing-compiled-backend cases to add.
