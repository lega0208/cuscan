# HDBSCAN implementation phase plan

This is an execution plan, not authorization to implement. It decomposes the pinned `fast_hdbscan==0.3.2` port across runnable, gated phases, and uses detailed stage semantics in [hdbscan_algo_impl.md](hdbscan_algo_impl.md) when the shorter overview [hdbscan_impl.md](hdbscan_impl.md) is ambiguous. The in-scope first release is exact Euclidean, unweighted Borůvka HDBSCAN with CPU/CUDA backends and a Python estimator.

## Target, invariants, and exclusions

### First-release scope

- Rust workspace targets: `fast-cubecl-common`, `fast-hdbscan-core`, `fast-hdbscan`, existing `fast-umap`, and existing `fast-umap-python`; Python package remains `fast_umap` and native module remains `fast_umap._fast_umap`.
- Public behavior: `HDBSCAN`/`FittedHDBSCAN`; labels, probabilities, single-linkage and condensed trees, sorted MST, finite-row-indexed neighbors and core distances, finite-index mapping, and `dbscan_clustering(epsilon)`.
- Euclidean exact MST via `algorithm="boruvka"`; extraction options `min_cluster_size`, optional `min_samples`, `eom`/`leaf`, `allow_single_cluster`, optional maximum size, epsilon and persistence selection; non-finite rows are excluded from fitting and remapped as noise.
- Backend selection follows current UMAP semantics: `Auto`, `Cpu`, `Cuda`; explicit backend selection never falls back. HDBSCAN imports common ANN only for the kNN-graph route (`parallel_boruvka_plan.md`). It does not import the fuzzy graph, layout, spectral initialization, or model-state functionality.

### Explicit exclusions

Do not implement in these phases: complete-graph Kruskal (Phase 6); sample weights; precomputed/sparse metrics; cannot-link or semi-supervised constraints; non-Euclidean metrics; `PLSCAN`, branch detection, layer clusters; distribution/package renaming. The D>30 kNN-graph route is implemented under `parallel_boruvka_plan.md`, not in these phases. Complete-graph fixture oracles still use exact pinned internal `parallel_boruvka(..., reproducible=True)`.

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

Execute in this order: **0 → 0.5 → 1 → 2 → 3 → 4 → 5**. Phase 1 consumes upstream MSTs, not a device implementation; phase 2 supplies the independent host oracle; phase 3 supplies the first end-to-end Rust estimator; phase 4 optimizes that working estimator without replacing its oracle; phase 5 exposes the verified implementation to Python. Each exit gate is a prerequisite, not a substitute for later end-to-end checks. Completed phases are moved to [hdbscan_plan_phases_completed.md](hdbscan_plan_phases_completed.md) when completed.

### Phase 1 — backend-free host pipeline (S1, S2, S8–S14) — completed

Completed scope, contracts, deliverables and exit gate: see [Phase 1 in the completed phases document](hdbscan_plan_phases_completed.md#phase-1--backend-free-host-pipeline-s1-s2-s8s14).

### Phase 2 — independent host exact MST oracle (S5–S7 reference) — completed

Completed scope, contracts, deliverables and exit gate: see [Phase 2 in the completed phases document](hdbscan_plan_phases_completed.md#phase-2--independent-host-exact-mst-oracle-s5s7-reference).

### Phase 3 — facade, brute-force device Borůvka, estimator (S3, S5–S7) — completed

Completed scope, contracts, deliverables and exit gate: see [Phase 3 in the completed phases document](hdbscan_plan_phases_completed.md#phase-3--facade-bruteforce-device-boruvka-estimator-s3-s5s7).

### Phase 4 — host-built KD tree and KD Borůvka fast path (S4, optimized S6) — completed

Completed scope, contracts, deliverables and exit gate: see [Phase 4 in the completed phases document](hdbscan_plan_phases_completed.md#phase-4--hostbuilt-kd-tree-and-kd-boruvka-fast-path-s4-optimized-s6).

### Phase 5 — PyO3 API and estimator (Python surface) — completed

Completed scope, contracts, deliverables and exit gate: see [Phase 5 in the completed phases document](hdbscan_plan_phases_completed.md#phase-5--pyO3-api-and-estimator-python-surface).

### Phase 6+ — separate plans only

No work in this plan. Future separate proposals, in order: (1) complete-graph `algorithm="kruskal"` at ≤30 features (the kNN-graph route is already implemented under `parallel_boruvka_plan.md`); (2) sample weights after resolving upstream squared/unsquared behavior; (3) precomputed sparse metrics, cannot-link, semi-supervised BC; (4) `PLSCAN`, branch detection/subclusters, layer clusters; (5) non-Euclidean metrics. Each needs its own semantics, API and acceptance plan. No compatibility aliases or speculative performance features are authorized here.

## Verification command ledger

Commands below are grounded in current repository conventions. **Existing commands** can be run now; **future targets/commands** depend on phase-created packages/tests and are not executable until those phases land. This plan does not claim any test command has been run.

### Existing verification commands

- Existing UMAP CPU integration suite: `cargo test -p fast-umap --no-default-features --features cpu`.
- Existing backend feature matrix and UMAP regression suites: `bash scripts/test_features.sh` (includes current `fast-umap-core` tests and feature checks; extend it when common/HDBSCAN exist).
- Existing CPU benchmark command pattern is documented in `AGENTS.md`: `cargo bench -p fast-umap --no-default-features --features cpu --bench standard_umap_cpu` (baseline and post-refactor comparison only).
- Existing Python UMAP pattern: `uv run maturin develop --release --no-default-features --features cpu && uv run --no-sync pytest python/tests/test_standard_*.py`.

### Focused phase commands (Phase 5 remains future work)

- Phase 0: `cargo check -p fast-cubecl-common --no-default-features` for backend-free feature compilation, and `cargo test -p fast-cubecl-common --no-default-features --features cpu` for shared CPU operations; existing UMAP CPU suite and `bash scripts/test_features.sh` after script update; compare UMAP CPU benchmark above.
- Phase 0.5/1: `cargo test -p fast-hdbscan-core` and `cargo test -p fast-hdbscan-core --test test_fixtures`.
- Phase 2: `cargo test -p fast-hdbscan-core --features test-support --test mst_oracle`.
- Phase 3: `cargo test -p fast-hdbscan --no-default-features --features cpu`; CUDA lane `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --no-default-features --features cuda` (run only with usable CUDA).
- Phase 4: host invariants `cargo test -p fast-hdbscan-core --test kdtree`; device search `cargo test -p fast-hdbscan --no-default-features --features cpu --test test_kdtree_search`; full estimator parity is included in the Phase 3 facade commands. Benchmarks: `cargo bench -p fast-hdbscan --no-default-features --features cpu --bench hdbscan_cpu` and required CUDA `FAST_UMAP_REQUIRE_CUDA=1 cargo bench -p fast-hdbscan --no-default-features --features cuda --bench hdbscan_cuda`. The README records the measured grid and conservative dispatch policy; the durable ledger retains command/output evidence.
- Phase 5 CPU-only: `uv run maturin develop --release --no-default-features --features cpu && uv run --no-sync pytest python/tests/test_hdbscan.py python/tests/test_standard_*.py`. Combined build: `uv run maturin develop --release --no-default-features --features cpu,cuda`, then `FAST_UMAP_TEST_BACKEND=cpu uv run --no-sync pytest python/tests/test_hdbscan.py python/tests/test_standard_*.py` and, without rebuilding, `FAST_UMAP_TEST_BACKEND=cuda FAST_UMAP_REQUIRE_CUDA=1 uv run --no-sync pytest python/tests/test_hdbscan.py`.

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
