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

| Package              | Default features | Feature contract                                                 |
| -------------------- | ---------------- | ---------------------------------------------------------------- |
| `fast-cubecl-common` | none             | declares `cpu`, `cuda`, `tracing`; owns shared runtime enables   |
| `fast-hdbscan-core`  | none             | `tracing` only; backend-free                                     |
| `fast-hdbscan`       | `cuda,cpu`       | forwards backends to common; `tracing`; no-backend compile guard |
| `fast-umap`          | `cuda,cpu`       | existing features plus backend forwarding to common              |
| `fast-umap-python`   | `cuda,cpu`       | forwards `cpu`/`cuda`/`tracing` to both facades                  |

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

**Verification receipt**
Core no-default and `test-support,tracing` suites pass; focused core Clippy with `-D warnings`, workspace CPU Clippy and `scripts/test_features.sh` pass. Generator `uv run --no-sync ... --check` verified 18 cases with the expected overflow `RuntimeWarning`. CUDA runtime is unverified. Implemented algorithms are backend-free host algorithms only; the oracle is non-default test support, not a fallback. No facade estimator or Python surface is implemented.
