# HDBSCAN implementation phase plan

This is an execution plan, not authorization to implement. It decomposes the pinned `fast_hdbscan==0.3.2` port across runnable, gated phases, and uses detailed stage semantics in [hdbscan_algo_impl.md](hdbscan_algo_impl.md) when the shorter overview [hdbscan_impl.md](hdbscan_impl.md) is ambiguous. The in-scope first release is exact Euclidean, unweighted Borůvka HDBSCAN with CPU/CUDA backends and a Python estimator.

## Target, invariants, and exclusions

### First-release scope

- Rust workspace targets: `fast-cubecl-common`, `fast-hdbscan-core`, `fast-hdbscan`, existing `fast-umap`, and existing `fast-umap-python`; Python package remains `fast_umap` and native module remains `fast_umap._fast_umap`.
- Public behavior: `HDBSCAN`/`FittedHDBSCAN`; labels, probabilities, single-linkage and condensed trees, sorted MST, finite-row-indexed neighbors and core distances, finite-index mapping, and `dbscan_clustering(epsilon)`.
- Euclidean exact MST via `algorithm="boruvka"`; extraction options `min_cluster_size`, optional `min_samples`, `eom`/`leaf`, `allow_single_cluster`, optional maximum size, epsilon and persistence selection; non-finite rows are excluded from fitting and remapped as noise.
- Backend selection follows current UMAP semantics: `Auto`, `Cpu`, `Cuda`; explicit backend selection never falls back. HDBSCAN does not import ANN, fuzzy graph, layout, spectral initialization, or model-state functionality.

### Explicit exclusions

Do not implement in these phases: Kruskal/kNN graph MST; sample weights; precomputed/sparse metrics; cannot-link or semi-supervised constraints; non-Euclidean metrics; `PLSCAN`, branch detection, layer clusters; distribution/package renaming; ANN reuse for HDBSCAN. The D>30 upstream public-fit approximate path is deliberately not replicated: use exact pinned internal `parallel_boruvka(..., reproducible=True)` as fixture oracle at every dimension.

### Cross-phase contracts

- Keep `fast-hdbscan-core` backend-free (no Burn/CubeCL, `faer`, or `rand`). Shared algorithm-neutral backend infrastructure is in `fast-cubecl-common`; algorithm-specific backend operations stay in `fast-hdbscan` to respect CubeCL extension/orphan boundaries. All internal Cargo edges use `default-features = false`; backend feature forwarding is per crate.
- Features are f32; pair distances and Borůvka candidate/core search use squared f32, feature accumulation in index order. Core distances are f32 square roots; MST output weight is `(weight_sq as f64).sqrt()`. Linkage and lambda calculation use f64 distance, lambda is stored f32. Stability accumulation and membership output are f32 with the exact mixed-precision rules in S9–S12 of the algorithm document. Validate row count ≤ `i32::MAX`; use deterministic total ordering and smaller original index for port-side distance ties. Exact/near tie outputs may differ from upstream traversal-order choices; assess these cases by documented ARI and probability/MST tolerances, not claims of bitwise identity.
- Shared exact-neighbor extraction MUST preserve UMAP's existing `f32::MAX` sentinel, overflow behavior, outputs and caller contract. HDBSCAN's squared mode uses `+inf` initialization so overflowed finite distances remain infinity; infinity candidates must win an equal-infinity tie against the n sentinel. Remaining missing-neighbor sentinel index `n` decodes to -1. Do not change UMAP behavior to fix the HDBSCAN contract.
- KD tree uses host-built implicit heap, flat uploaded buffers, deterministic `(feature, original index)` partition; traversal stack bound is checked. Lower-bound and component pruning are strict `>` so equality candidates are not pruned. Component-best scratch is reset to +inf before every enabled launch and never survives a Borůvka round. The brute-force implementation is the correctness oracle. Deterministic tie rules do not promise same tree/orientation as upstream.
- Handle input validation before device upload. Empty raw input, zero features, ragged rows, invalid config, row/index or allocation overflow are typed errors. NaN/inf rows are not errors. After finite split, **zero finite rows is InvalidInput before any degenerate handling**; one finite row returns all-noise labels/probabilities, empty trees/MST, neighbors of shape `(1, 0)` and infinite core distance. Two finite rows use the ordinary pipeline. Whenever `n_finite <= min_samples`, cores/MRD weights are infinite and labels are all noise. Non-finite remap affects labels/probabilities and trees only: MST, neighbors and core distances remain finite-row indexed and `finite_index` maps them.
- `dbscan_clustering` rejects NaN epsilon with `HdbscanError::InvalidConfig`; otherwise uses the linkage-cut semantics, including `epsilon <= 0` yielding singleton components and `epsilon=+inf` joining finite-distance rows. Its Rust return is therefore `Result<Vec<i32>, HdbscanError>` (the overview's bare `Vec<i32>` sketch is incomplete).
- Python rejects unsupported upstream parameters with `NotImplementedError`, never silently ignores them; uses upstream type checks for floating epsilon/persistence (including integer epsilon rejection), maps core errors to `ValueError` (invalid config/input), `RuntimeError` (backend unavailable/numerical), and `OverflowError` (overflow). Inputs cast to f32; finite f64 values that overflow f32 raise `ValueError`.

## Delivery phases

Each phase is independently buildable/testable before its dependent phase begins. Source stage coverage is exhaustive: S1–S2 phase 1; S3 phase 3; S4 phase 4; S5–S7 phases 2–3; S8–S12 phase 1; S13 phase 1; S14 phase 1. Estimator orchestration lands phase 3 and is extended at phase 5. Fixture generation is phase 0.5.

Execute in this order: **0 → 0.5 → 1 → 2 → 3 → 4 → 5**. Phase 1 consumes upstream MSTs, not a device implementation; phase 2 supplies the independent host oracle; phase 3 supplies the first end-to-end Rust estimator; phase 4 optimizes that working estimator without replacing its oracle; phase 5 exposes the verified implementation to Python. Each exit gate is a prerequisite, not a substitute for later end-to-end checks.

### Phase 0 — isolate shared infrastructure; preserve UMAP

**Scope and sequence**
1. Add `crates/fast-cubecl-common` for only neutral backend choice/error, CUDA discovery/device resolution, `legal_cube_dim`, `NeighborOps` and exact Euclidean neighbors. Move the relevant implementations/tests from `crates/fast-umap/src/{standard_umap/backend_choice.rs,cuda_backend.rs}` and device resolution in `standard_umap/estimator.rs`; retain facade-specific error mapping in each facade.
2. Extract exact-neighbor and squared-distance-block operations from `StandardUmapOps` into common. Keep approximate-neighbor/ANN kernels and orchestration in `fast-umap`; update all dispatch call sites and fusion/kernel metadata. UMAP continues legacy mode (`squared=false`, `f32::MAX` slot initialization); HDBSCAN will request squared mode with +inf sentinel.
3. Forward CPU/CUDA/tracing features per the architecture in the source plan. Add feature-isolation checks so backend-free cores do not acquire runtime dependencies.

**Infrastructure boundary checklist**
- Common owns `backend_choice.rs`, `cuda_backend.rs`, `device.rs`, `error.rs`, `backend.rs`, `neighbors.rs`, and exact-neighbor kernels. Keep the `NeighborOps` trait and its CubeBackend implementation together; expose `legal_cube_dim` for both facades.
- Move `selected_device`/`resolve_backend_choice` and their direct tests. Preserve Auto's rule: CPU is selected only after discovery confirms CUDA unavailable, never after CUDA execution fails.
- Common `BackendChoice::validate` returns `BackendError`; UMAP/HDBSCAN facade wrappers explicitly map errors with local `map_err`, not `From<BackendError>` implementations in backend-free cores. Preserve UMAP's existing error variants/messages and Python invalid-choice/unavailable mappings. This is a deliberate method-signature change, not a compatibility shim.
- Update every exact-neighbor/squared-distance dispatch caller, fusion registration, example and test. Retain `standard_umap/neighbors.rs`, `approximate_neighbors`, `ann_sampled.rs`, `candidate_width` and all ANN orchestration in UMAP.
- Extend `scripts/test_features.sh` for isolated CPU, CUDA, combined features, backend-free cores and the expected facade no-backend compile diagnostic. Keep Burn/CubeCL requirements centralized in workspace dependencies.

| Package | Default features | Feature contract |
|---|---|---|
| `fast-cubecl-common` | none | declares `cpu`, `cuda`, `tracing`; owns shared runtime enables |
| `fast-hdbscan-core` | none | `tracing` only; backend-free |
| `fast-hdbscan` | `cuda,cpu` | forwards backends to common; `tracing`; no-backend compile guard |
| `fast-umap` | `cuda,cpu` | existing features plus backend forwarding to common |
| `fast-umap-python` | `cuda,cpu` | forwards `cpu`/`cuda`/`tracing` to both facades |

Dependency gate: Python → UMAP → UMAP core/common, and Python → HDBSCAN → HDBSCAN core/common. HDBSCAN must not depend on the UMAP facade or transitively pull in spectral `faer`.

**Deliverables and focused tests**
- Common API and tests for `BackendChoice`, strict device resolution, exact-neighbor shape/tie/sentinel behavior and launch blocking.
- UMAP facade mapping tests and unchanged existing integration/fixture results; verify ordinary and overflow-sensitive neighbor cases before/after extraction.
- Update manifests, `Cargo.lock`, scripts/guidance only as needed for the new crate. No `pyproject.toml` cache-key edits.

**Exit gate**
UMAP observable outputs and fixtures unchanged on supported inputs, CPU and Python lanes; feature-isolation checks pass. Compare existing CPU benchmark before/after and investigate regression; make no performance claim absent evidence. Do not begin HDBSCAN phases until this gate is met.

### Phase 0.5 — pinned reference fixtures

**Scope and sequence**
1. Add `crates/fast-hdbscan/tests/fixtures/generate_fast_hdbscan.py`, pinned to `fast_hdbscan==0.3.2` and source commit `43c573ba61043a6ad6f10b396656bfaa7aaeead8`.
2. For every dimension including 2D, 10D and 50D, call pinned internal `parallel_boruvka(..., reproducible=True)` directly, then the pinned upstream tree, extraction and probability routines. Never use public `fit` as the exact MST oracle: upstream public fit switches to approximate pynndescent/Kruskal for D>30. Save inputs NPZ; JSON metadata records package/version, commit, entrypoint, reproducible flag, dimension and complete parameters; retain MST f64 weights and structured condensed-tree records plus linkage, labels and probabilities.
3. Add fixtures for blobs, moons, varying density, duplicates/ties, single cluster, all noise, `n < min_samples`, `n=1,2`, and mixed/all nonfinite rows. Sweep min_samples/min_cluster_size, leaf/EOM, single-cluster option, epsilon, persistence, and max size. Record upstream outlier order for remap checks.
4. Add the pinned reference package to `pyproject.toml` development dependencies and update `uv.lock` with uv. Store neighbors, core distances, finite-index maps and DBSCAN-cut outputs alongside tree/label fixtures so later tests do not recompute their oracle from Rust output.
5. Separate ordinary exact-MST fixtures from validation/degenerate cases. Split finite rows before calling the internal MST entrypoint; zero finite rows record the expected validation error, and one finite row records the degenerate contract rather than invoking an invalid MST calculation. Record upstream's one-finite-plus-outliers remap crash as a deliberate port divergence, not a fabricated upstream successful tree result.
6. Classify exact/near-tie datasets in metadata. Non-tie classification must cover candidate, merge and cluster-selection comparisons, not only gaps between MST weights. Include finite features whose squared distances overflow, missing-neighbor padding, and persistence thresholds exactly on branch boundaries.

**Deliverables and focused tests**
Fixture generation is reproducible from the pinned environment; metadata and array shapes/dtypes are validated by loader smoke tests (future target `test_fixtures`). Core integration tests will read shared fixtures through the relative path `../fast-hdbscan/tests/fixtures` from the core crate root; use existing workspace `serde_json` and `zip` as core dev dependencies for metadata/NPZ.

**Exit gate**
All fixtures regenerate with complete provenance, including direct internal MST oracle for every dimension; fixture loader rejects missing/inconsistent metadata. This phase runs before host pipeline parity work.

### Phase 1 — backend-free host pipeline (S1, S2, S8–S14)

**Scope and order**
1. Create `crates/fast-hdbscan-core/src/{lib.rs,config.rs,error.rs,finite.rs}`. Implement config validation and resolution, flatten/input checks, finite split, zero-finite error, one-finite degenerate result, raw-row remapping. Define errors `InvalidConfig`, `InvalidInput`, `BackendUnavailable`, `Numerical`, `Overflow` without backend dependency.
2. Implement `dsu.rs` rank union with first-argument root on rank ties and path compression; test representative behavior and argument order.
3. Implement `linkage.rs` S8 from externally supplied sorted MST; preserve larger component id first, f64 distance, sizes and exact row order.
4. Implement `condense.rs` S9 and optional persistence simplification S10. Preserve BFS left-before-right order, row order, f32-rounded hierarchy sizes vs unrounded integer threshold, lambda `1/d` f64 stored f32 (zero distance→+inf, infinite distance→0), eliminated-point ordering/size, sibling pair behavior and forward float accumulation semantics.
5. Implement `select.rs` S11 for EOM/leaf, optional max size, allow-single-cluster and epsilon ancestor search. Use iterative postorder for deep trees; preserve selected insertion order and floating comparisons.
6. Implement `labels.rs` S12 membership and S14 dbscan cut. Preserve selected-id ordering, DSU union order, special single-root assignment, point-row maximum lambda, and membership f64 division then f32 storage. S14 returns `Result` for NaN.
7. Use a test-only hook taking upstream `SortedMstEdge` fixtures to exercise S8–S12 without depending on a Rust MST implementation. Remap output trees with recorded upstream outlier ordering; separately test deterministic ascending Rust outlier ordering.

**Host implementation details to pin before phase 2**
- Define shared `SortedMstEdge`, `LinkageRow` and `CondensedTree` data types in phase 1 so fixture-driven linkage builds without a later MST implementation. Phase 2 adds squared working edges and producers of the existing finalized edge type. Condensed tree stores SoA `parent`, `child`, `lambda` and `child_size` arrays; integer counts are converted to f32 where upstream arithmetic requires it.
- Config defaults: minimum cluster size 5; minimum samples `None` resolves to that size; EOM; single cluster false; maximum size `None`; epsilon/persistence 0.0. Require positive minimum sizes and finite nonnegative selection thresholds. Keep backend choice outside the backend-free config.
- Linkage uses new node ids starting at `n` over `2n-1` slots, with the larger root id first. Condensation uses BFS, root id `n`, and consecutive sibling cluster pairs; `max_cluster_size` is a selection constraint, not a condensation constraint.
- Persistence simplification preserves forward last-write-wins lambda scatter and reverse sibling-pair processing. Compute branch persistence in f32 before promotion for comparison to f64 threshold; replace repeated skipped-node sums with an equivalent prefix sum.
- EOM scores accumulate f32 in row order; descendant score totals accumulate f64. Preserve upstream root size `n-1`, f32-rounded sizes compared to unrounded maximum-size thresholds, and insertion order before epsilon search. Implement iterative postorder plus top-down ancestor suppression; compare it with a literal selection implementation on generated trees.
- Leaf selection returns no clusters for an empty cluster tree; `allow_single_cluster` does not affect leaf selection. Epsilon search runs only for more than one selected cluster and positive epsilon, preserving upstream processing order.
- Labels skip point unions at lambda zero and apply the root-cluster membership threshold exactly. Membership maxima use point rows only, noise strength is zero, and infinite point lambda or zero cluster maximum gives strength one.
- Non-finite remap prepends condensed outlier rows at lambda zero and reproduces upstream's noncanonical infinite-distance linkage chaining. With one finite row, leave trees empty. Rust outlier ordering is ascending; normalize trees or inject recorded upstream order only in tests.
- DBSCAN cut unions only rows with `distance < epsilon`, uses `min_samples.unwrap_or(min_cluster_size)` as size threshold, and numbers by ascending DSU representative. Test equality at cut distance, negative/zero/infinite cut, NaN rejection, and `min_samples=1` outlier singleton labels after remap.

**Deliverables and focused tests**
Modules: `config.rs`, `error.rs`, `finite.rs`, `dsu.rs`, `linkage.rs`, `condense.rs`, `select.rs`, `labels.rs`; integration tests and fixture loader under `crates/fast-hdbscan-core/tests/`. Exact fixture comparisons for linkage, condensed tree, labels and probabilities; persistence threshold equality; EOM/leaf and option sweeps; deep-chain stack safety; finite-row scatter and tree remap; DBSCAN epsilon sweep both pre/post remap. Invalid config/input boundaries include empty, zero features, ragged, zeros, negative/nonfinite selection values, row-size overflow, NaN cut, zero finite rows, and one finite row.

**Exit gate**
Every host tree/label/probability result is exact versus fixtures when supplied the fixture MST and required outlier order; error and degenerate contracts pass. The backend-free crate builds/tests with no backend feature. No GPU dependency is introduced.

### Phase 2 — independent host exact MST oracle (S5–S7 reference)

**Scope and order**
1. Add host `mst.rs` helpers for squared `MstEdge`, candidate order and finalization into the phase-1 sorted f64 edge type; exact all-pairs mutual-reachability Borůvka using brute-force candidates and DSU. Compute core distances independently by host exact kNN. This is a test/reference implementation, not a production fallback; reusable merge/finalization helpers are production host code.
2. Implement/validate kNN initialization semantics against core and neighbor data; apply in point order, skip absent neighbors, preserve rank union and deterministic tie ordering.
3. Finalize squared f32 edge weights by f64 sqrt and sort by `(weight, src, dst)`. Validate exactly `m-1` edges for `m>=2` and valid acyclic spanning structure.

**Deliverables and focused tests**
Host oracle tests compare MST weight multiset with pinned oracle at relative tolerance 1e-5 and validate candidate/edge invariants, duplicates, ties and infinite-core cases. Expose it only as test support unless a production need is explicitly established.

**Exit gate**
Host oracle independently reproduces oracle MST weight multiset and feeds phase-1 host stages; it has no backend dependency. Phase 3 compares device candidate minima against this oracle.

### Phase 3 — facade, brute-force device Borůvka, estimator (S3, S5–S7)

**Scope and order**
1. Create `crates/fast-hdbscan` facade and `HdbscanOps` in `backend.rs` with its CubeBackend implementation/kernels in `src/kernels/`; add explicit local backend error conversions. Add required `HdbscanConfig`/estimator-facing Rust API and exports.
2. Implement S3 `core_distances` via common exact-neighbor squared mode with self excluded and `k=min(min_samples,n-1)`. Core is zero for min_samples=1, +inf when `n<=min_samples`, otherwise last requested neighbor squared distance; output core distances sqrt f32. Pad neighbor ids with -1 and distances +inf. Do not claim neighbor-list identity with upstream on duplicate points: our self-exclusion contract differs there while core radii match.
3. Implement S5 initialization and S6a point components; phase 3 has no KD node vectors/tree.
4. Implement tiled brute-force S6b MRD squared candidates, deterministic `(weight,index)` tie break, device-limit row blocking, and tile reduction. Infinity candidates must be selected by tie arm against sentinel `n`; validate CubeCL infinity literal on CPU and CUDA, passing scalar infinity if needed.
5. Implement S6c host per-component winners in first appearance order, strict `<` preserving first source row ties, union in that order, and zero-progress `Numerical` error. S7 converts/sorts MST. Then S8–S13 estimator orchestration using phase-1 core functions.
6. Implement strict `BackendChoice` behavior, CPU and CUDA, and backend-specific feature forwarding without adding these operations to `StandardUmapOps`.

**Device API and orchestration contract**
- Freeze `HdbscanOps::boruvka_candidates(data, core_sq, point_components, tree)` before implementing kernels. Inputs are immutable tensors; outputs are `[n]` f32 squared weights and `[n]` i32 destinations. `tree=None` selects brute force. Plan `DeviceKdTree` as a real fusion input with `ExtensionType` and metadata, not an opaque host struct; the populated tree path lands in phase 4.
- One tiled launch computes winners for `(query row, reference tile)`; a second reduces tiles. Check `rows_in_block * ceil(n/tile)` scratch sizing and all padded `n * min_samples` outputs with checked arithmetic and device allocation limits. Shared launch limits apply to both runtimes.
- Preserve the distinction between every-row exact candidates in brute force and only per-component exact minima when KD pruning is enabled. Ignore individual missing candidates, but error if any unfinished component has no outgoing candidate. Verify component-count reduction and exactly `m-1` final MST edges.
- Initialize from kNN in point order: first nonmissing neighbor with `core_i >= core_j`, weight `max(core_i, knn_sq)`. With minimum samples 1, cores are zero but the chosen edge retains its squared pair distance. Skip the round loop if initialization already connects every point.
- Allocate/upload only after validation and finite/degenerate handling. Upload a compact matrix only when rows were excluded; retain the caller's finite feature values unchanged. Expose typed config, fitted arrays, finite-index mapping, backend and fallible DBSCAN-cut accessors from the facade.

**Deliverables and focused tests**
`crates/fast-hdbscan/src/{lib.rs,backend.rs,estimator.rs,kernels/{mod.rs,boruvka.rs}}`; fixture end-to-end tests on CPU; core-distance host brute-force comparisons for min_samples `{1,n-1,n,n+5}`; squared overflow and all-infinite core tests; S5 acyclic forest; per-component candidate minima and host MST parity; no-progress path; explicit backend failure mapping. CUDA tests run only when usable hardware exists and are required with `FAST_UMAP_REQUIRE_CUDA=1` in CUDA lane. Near-tie tests use documented ARI and tolerance rules.

**Exit gate**
CPU end-to-end fixture labels/probabilities/trees pass (exact host stage, MST weights relative tolerance 1e-5, probabilities 1e-5; labels equal up to permutation/ARI=1 only where relevant decisions are separated from ties). CUDA parity uses the same rules; unavailable CUDA is reported unverified, never a passing skip. Exact MST works at all dimensions.

### Phase 4 — host-built KD tree and KD Borůvka fast path (S4, optimized S6)

**Scope and order**
1. Add backend-free `kdtree.rs`: leaf size 40, implicit heap, flat indices/ranges/leaf flags/bounds, iterative build, deterministic `(feature value, original index)` partition, validated nonempty leaf ranges and max depth.
2. Upload the tree and implement `box_lower_bound` and device traversal with fixed `MAX_KD_LEVELS+1` stack; near-first order, left-first on equal bound. Prune only on strict `>`; prune homogeneous own-component nodes. Test each bound against brute-force point-box calculations.
3. Add component-bound pruning only as a separately tested optional kernel policy, with implementation-owned scratch reset to +inf before every launch/round. It is disabled unless runtime-specific benchmarks demonstrate value; stale bounds are correctness failures. A missing candidate for an entire component is `Numerical`.
4. Add KD-tree exact kNN using the same traversal skeleton with a k-heap, replacing the phase-3 quadratic S3 path in the low-dimensional dispatch. Reuse the existing kNN-initialized first round; verify identical core radii, padding and MST invariants. Retain brute force as the exact high-dimensional path and test oracle, not as a retry after kernel errors. Select a low-dimensional cutoff through measured benchmarks, starting investigation at 30 dimensions; it is a performance dispatch only and cannot alter non-tie MST semantics.
5. Refresh point roots and bottom-up homogeneous-node component labels every round; upload the tree once and update component buffers. Check nonempty leaves, `MAX_KD_LEVELS=32` and `n_levels+1 <= STACK_LEN` before launch. Strict equality must remain visitable even when every candidate weight is infinite.
6. Treat device component-winner reduction and reduced host readback as optional, separately benchmarked work, not a prerequisite. If implemented, preserve first-source tie ordering and check atomic capabilities on each runtime. Do not add packed-u64 atomics or point reordering without their own evidence and invariant tests.

**Deliverables and focused tests**
Tree invariants: every row exactly once in leaves, bounds contain all points, depth within fixed stack, deterministic repeated build. KD vs brute-force per-component minima, CPU/CUDA parity away from ties, traversal tie cases, pruning equality, multiple Borůvka rounds with increasing winning weights (scratch reset). Criterion targets `hdbscan_cpu`/`hdbscan_cuda` are planned future targets; compare with upstream only after correctness. Benchmark tile sizes 128/256/512 as targeted investigation, not a performance promise.

**Exit gate**
KD path matches phase-3 brute-force MST weight multiset and downstream results under the same tie/tolerance policy on fixtures and adversarial tree cases. Any enabled optimization passes exact component-minimum checks. Benchmark data supports chosen cutoff; absent data, retain conservative exact brute-force selection.

### Phase 5 — PyO3 API and estimator (Python surface)

**Scope and order**
1. Extend `crates/fast-umap-python/src/lib.rs`, add `src/hdbscan.rs`; keep a single native module. Add `fit_hdbscan` and frozen fitted result accessors, translating Rust errors to specified Python classes.
2. Add `python/fast_umap/hdbscan.py` estimator and export `HDBSCAN` in `python/fast_umap/__init__.py`. Match only in-scope upstream parameters plus `backend="auto"` and `device_index=0`. Expose `labels_`, `probabilities_`, structured condensed tree dtype, linkage/MST arrays, and `dbscan_clustering(eps)`.
3. Reject unsupported parameters explicitly with `NotImplementedError` (including sample weights, non-Euclidean metric, Kruskal/knn_k, cannot-link, semi-supervised); do not accept-and-ignore. Match Python numeric type checks and f64→f32 overflow behavior. Preserve UMAP Python API and error mapping.
4. Add Python parity and integration tests, including `HDBSCAN(...).fit_predict(UMAP(...).fit_transform(X))`, nonfinite rows, output dtypes/shapes, error mapping, unsupported args and backend choice.

**Binding details**
- Split the existing UMAP binding source into `src/umap.rs` as needed, with registration in `lib.rs` and new HDBSCAN code in `src/hdbscan.rs`; avoid changing existing UMAP behavior or introducing a second extension module.
- Implement `fit`, `fit_predict`, fitted-state checks and native fitted-result lifetime/ownership. Accept `reproducible` but document that it has no effect because the port is always deterministic; this is the sole explicitly permitted ignored option.
- Labels expose NumPy `intp`; Rust probabilities remain f32, while Python mirrors upstream's f64 probability widening when non-finite rows are remapped. Preserve upstream structured condensed-tree field names/dtypes; expose raw linkage/MST arrays rather than plot wrapper objects. Expose neighbors/core distances and `finite_index` with their finite-row indexing documented.
- Parameterize unsupported-option tests over sample weights, non-Euclidean metric, Kruskal, `knn_k`, cannot-link options and semi-supervised mode. Distinguish upstream floating scalar type validation from nonnegative/finite value validation; integer epsilon is invalid even when zero.

**Deliverables and focused tests**
Future test target `python/tests/test_hdbscan.py` and Rust binding tests are planned; existing Python command patterns apply. Tests assert labels up to permutation/ARI according to tie classification, probabilities 1e-5, structured tree fields/dtypes, and Python exceptions for every invalid/unsupported input category.

**Exit gate**
CPU pytest parity, UMAP→HDBSCAN and error/dtype tests pass; CUDA Python parity is required only in a configured CUDA lane and must not be represented as verified otherwise. Existing UMAP Python tests remain unchanged and green.

### Phase 6+ — separate plans only

No work in this plan. Future separate proposals, in order: (1) `algorithm="kruskal"` with `knn_k`, using exact KD-tree `parallel_tree_query` rather than ANN and accounting for +inf bridges; (2) sample weights after resolving upstream squared/unsquared behavior; (3) precomputed sparse metrics, cannot-link, semi-supervised BC; (4) `PLSCAN`, branch detection/subclusters, layer clusters; (5) non-Euclidean metrics. Each needs its own semantics, API and acceptance plan. No compatibility aliases or speculative performance features are authorized here.

## Verification command ledger

Commands below are grounded in current repository conventions. **Existing commands** can be run now; **future targets/commands** depend on phase-created packages/tests and are not executable until those phases land. This plan does not claim any test command has been run.

### Existing verification commands

- Existing UMAP CPU integration suite: `cargo test -p fast-umap --no-default-features --features cpu`.
- Existing backend feature matrix and UMAP regression suites: `bash scripts/test_features.sh` (includes current `fast-umap-core` tests and feature checks; extend it when common/HDBSCAN exist).
- Existing CPU benchmark command pattern is documented in `AGENTS.md`: `cargo bench -p fast-umap --no-default-features --features cpu --bench standard_umap_cpu` (baseline and post-refactor comparison only).
- Existing Python UMAP pattern: `uv run maturin develop --release --no-default-features --features cpu && uv run --no-sync pytest python/tests/test_standard_*.py`.

### Planned future focused commands (after corresponding phases)

- Phase 0: `cargo check -p fast-cubecl-common --no-default-features` for backend-free feature compilation, and `cargo test -p fast-cubecl-common --no-default-features --features cpu` for shared CPU operations; existing UMAP CPU suite and `bash scripts/test_features.sh` after script update; compare UMAP CPU benchmark above.
- Phase 0.5/1: `cargo test -p fast-hdbscan-core` and `cargo test -p fast-hdbscan-core --test test_fixtures` (fixture-loader target name is proposed).
- Phase 2: `cargo test -p fast-hdbscan-core --test test_mst_oracle` (proposed target).
- Phase 3: `cargo test -p fast-hdbscan --no-default-features --features cpu`; CUDA lane `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --no-default-features --features cuda` (proposed crate/test target; run only with usable CUDA).
- Phase 4: `cargo test -p fast-hdbscan --no-default-features --features cpu --test test_kdtree` and `cargo bench -p fast-hdbscan --no-default-features --features cpu --bench hdbscan_cpu` (proposed targets); corresponding CUDA bench is proposed and requires CUDA.
- Phase 5: `uv run maturin develop --release --no-default-features --features cpu && uv run --no-sync pytest python/tests/test_hdbscan.py` (proposed test file).

## Cross-phase risk and release gates

| Risk | Required evidence / containment |
|---|---|
| Shared refactor changes UMAP | Before/after fixtures, overflow-sensitive neighbor tests, CPU/Python regressions, feature matrix and CPU benchmark; block phase 0 on unexplained changes |
| Precision changes downstream selection | Exact host-stage fixtures on supplied MST; S3 core-distance comparison at 1e-6 relative tolerance; end-to-end MST/probability checks at 1e-5 with explicit infinity handling |
| Ties mistaken for regressions or hidden behind ARI | Separate annotated tie fixtures; require labels up to permutation and ARI=1 away from all relevant near ties; record ARI and changed decisions on tie cases rather than silently relaxing every fixture |
| Infinity causes stalled Borůvka | Real infinity candidates win against sentinel, every component has an outgoing edge, component count decreases, and final tree has exactly `m-1` edges; compare infinity to infinity explicitly, not by subtracting infinities |
| KD traversal prunes valid edges | Bounds/stack invariants, strict pruning, component-minimum oracle, multi-round scratch-reset tests and runtime-specific parity; do not enable speculative pruning policies |
| Quadratic stages overwhelm device memory/time | Checked row/tile/neighbor allocation sizes and bounded scratch; stage timings and Criterion/upstream comparisons before choosing dispatch cutoff or claiming scalability |
| CUDA is skipped and reported as tested | New hardware tests must honor `FAST_UMAP_REQUIRE_CUDA=1` by failing when unavailable; record unavailable lanes as unverified, and require an actual CUDA run for CUDA release claims |
| Python surface silently drifts | Existing UMAP tests plus HDBSCAN parameter/error/dtype/finite-remap tests and the composed UMAP→HDBSCAN scenario |

At the end of each implementation phase, retain commands, pass/fail results, fixture provenance and outstanding hardware limitations with its review evidence. Run repository format/lint checks once over the resulting changes, using `cargo fmt --all -- --check`, the updated CPU feature matrix, and applicable Python Ruff/type checks. Final delivery requires all phase gates, documented deliberate divergences, crate/backend usage guidance, and measured benchmarks before any speed claim. This documentation task itself does not execute those future implementation gates.

## Conflicts resolved from source sketches

1. The short API sketch gives `dbscan_clustering(epsilon) -> Vec<i32>`, while detailed S14 requires NaN to return `InvalidConfig`; use `Result<Vec<i32>, HdbscanError>`.
2. The short validation list mentions empty input, but S2 explicitly requires `n_finite == 0` to error before degenerate handling. Thus all-nonfinite non-empty input is also `InvalidInput`; only one finite row gets the degenerate result.
3. Keep MST, neighbors and core distances indexed only by finite-row coordinates after S13 remapping; scatter labels/probabilities and remap the two trees, with `finite_index` exposed. Do not remap these distance/index arrays to raw coordinates.
4. The phase overview's “core pipeline” ordering is clarified by exact stage contracts: S3 is device exact-neighbor work and belongs with the phase-3 facade; S4 KD-tree is phase 4; phase 1 host tests consume supplied upstream MSTs.
5. D>30 fixture generation uses direct reproducible `parallel_boruvka`, never public fit's approximate path. Exact MST weight parity is tolerance-based; exact tie topology is not promised.
