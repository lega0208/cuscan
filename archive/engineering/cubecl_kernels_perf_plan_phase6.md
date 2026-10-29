# Phase 6 detailed plan: CPU leaf-tiled SoA kernel, vectorized across candidates, with leaf-ordered queries

This document details Phase 6 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md) (F4 and the "leaf-ordered queries" part of opportunity 1 in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups"). It builds on accepted Phase 3 CPU code and Phase 5 CUDA geometry; Phase 4 was rejected. Legacy source line numbers refer to `24adb89` and are navigation hints only; inspect symbols in actual current source before implementation.

**Status:** execution authorized and implementation completed under the valid-builder-tree contract; acceptance code **C16** (SoA candidates, block64 leaf-ordered queries, cap16). All 11 experimental and 11 final Step 3 gates passed, with required CUDA on designated gates; final 20k/100k pipelines exited 0 with labels equal. Independent review receipts are maintained separately in the ultragoal ledger. See [accepted retry results](hdbscan_perf_report.md#phase-6-valid-tree-retry-results-c16-accepted) and `artifacts/cubecl_kernels_perf/phase6_valid_trees/{acceptance_summary,width_selection,experiment_gates,final_clean_gates,integration_clean,cold_start,rayon_baseline,codegen_evidence}.json`. The first experiment's malformed-tree rejection remains historical and unchanged in `artifacts/cubecl_kernels_perf/phase6/`; it is not retry proof. The steps and fresh-session handoff below record the approved execution design, not outstanding implementation authorization.

## Approved retry contract and fresh-session handoff

Current evidence index: [`summary.json`](artifacts/cubecl_kernels_perf/phase6_valid_trees/summary.json) derives `accept`/C16 with no unmet acceptance criteria, cold first-minus-warm differences, Rayon comparisons and pipeline label proof. Selected C's algorithm is unchanged from measurement; unselected branches were removed, and final-source verification is not acceptance timing. `final_clean_gates.json` records all 11 passing gates; `integration_clean.json` records 20k/100k exit0 and `pipeline_clean_{20000,100000}/benchmark.json` both report labels equal. Initial introduced clippy warnings were corrected; `logs/final_clean_clippy.log` retains only pre-existing seed-grouping/oracle-range-loop and dependency future warnings. Earlier gates/logs are retained; no warnings are claimed suppressed. Existing unrelated ignored hardware-campaign tests remain in `all_cuda`; final KD suite is 16 passed/0 ignored, not skipped required CUDA coverage. README/AGENTS were searched and need no CPU schedule/buffer-enumeration update (rationale in summary). Independent review receipts are maintained separately in the ultragoal ledger.

- Support only finite KD-search data and valid trees produced by `fast_hdbscan_core::KdTree::build`, uploaded unchanged by the estimator. This is after the estimator's existing finite-row compaction; public HDBSCAN nonfinite-row handling and original-row remapping remain unchanged. The builder rejects nonfinite input, starts with the permutation `0..n`, and partitions ranges. Low-level callers must supply equivalent valid tree/data pairs: permutation indices, partitioning reachable leaf ranges within `0..n`, valid reachable topology and levels, and enclosing bounds. Document this precondition; do not add runtime scans, readbacks or tree-content validation. Existing shape/dtype validation and checked sizing remain.
- No ownership map, ownership launch, absent-row tail pass or corresponding memory term exists in the retry. A permutation supplies exactly one query writer per original ID. Malformed trees, non-permutation indices, out-of-range candidates/ranges and injected NaNs are outside Phase 6 acceptance; physical-buffer clamping is not a semantic guarantee.
- On restart inspect `git status`, current HEAD and diffs; never revert or commit user work. Original production baseline is `d030b7eb0a1f2a4edccc41fea01671ca9b16f092`; production `kdtree.rs`, `kernels/mod.rs` and `estimator.rs` were restored to that baseline. At this amendment the known HEAD is `d8faeab5d0e825e4d93c6c1478e35b4d1f15721a` (`phase 6 failure`), which tracks the retained host example and Rayon dev-dependency. Preserve that existing tool and inspect its actual future status; do not require future HEAD equality or assume uncommitted changes.
- Original ultragoal G001 in `.gjc/_session-01a179fc-c1f5-7571-9e53-cb62663f9e8c/ultragoal/` completed as an authorized rejection, not an accepted optimization. Implementation requires a separate user execution request and new goal; do not reopen or combine that run. A new session may reuse a goal ID such as G001; identity is session-scoped.
- Use a new evidence directory, e.g. `artifacts/cubecl_kernels_perf/phase6_valid_trees/`, throughout this plan (abbreviated `retry/` below). Never overwrite historical `phase6/`. Inspect its `experiment.diff`, `provenance.json`, `summary.json`, `final_gates.json`, `integration.json`, `rayon_baseline.json`, `qa_boundary_report.json` and failure IR/logs only as historical evidence. Do not blindly reapply the diff: it contains obsolete ownership/malformed assumptions.
- Establish fresh provenance against actual working source and preserve the accepted CUDA64 baseline binary before rebuilding. Prior probes found widths `[16,8,4,2,1]`, 15 concurrent CPU units, and initial cap 16; no width was selected. Block size 64 remains fixed. No P/O/C acceptance timings exist; prior partial normal-case passes are not full revised validation proof. Run all original required gates again against the new implementation.
- The retained host `--verify` reports zero ID mismatches but many distance-bit mismatches in primary cells. Ordered unfused Rust versus observed LLVM `fmuladd` is a supported explanation, not an exact oracle. Keep this reporting-only comparison honest.
- Ignored historical artifacts may be absent in a fresh checkout. Do not infer tests or timings from missing files, blindly reconstruct evidence, or treat old claims as new validation. Establish fresh evidence from verified tracked source; record missing evidence and any newly generated input provenance explicitly.

### Affected source and test map

Use symbols rather than legacy line numbers; new retry symbols below are planned, not existing implementations.

| Path                                                                                 | Responsibility / symbols                                                                                                                                                                                                                                                                                |
| ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/fast-hdbscan-core/src/kdtree.rs`                                             | `KdTree::build`, `LEAF_SIZE`, `MAX_KD_LEVELS`: builder contract, no production algorithm change                                                                                                                                                                                                         |
| `crates/fast-hdbscan-core/tests/kdtree.rs`                                           | Extend `assert_invariants`, `build_handles_leaf_size_boundaries_and_heap_layout`, `build_is_deterministic_with_duplicate_and_equal_feature_values`; retain `build_rejects_invalid_dimensions_and_non_finite_values`                                                                                     |
| `crates/fast-hdbscan/src/backend.rs`                                                 | Document valid data/tree precondition on `DeviceKdTree` and `HdbscanOps::exact_neighbors_kd`; no runtime scan                                                                                                                                                                                           |
| `crates/fast-hdbscan/src/kernels/kdtree.rs`                                          | Existing `exact_neighbors_kd`, `launch_kd_neighbors`, `validate_tree`, `kd_launch_geometry`, `kd_padded_bytes`, `kd_neighbors_kernel`, frozen `kd_neighbors_reference_kernel`; planned plan/route, CPU/pack kernels, pack sizing and all-arm/all-width harness via `check_kd_neighbors_match_reference` |
| `crates/fast-hdbscan/src/kernels/mod.rs`                                             | Export planned `kd_neighbor_plan` / `kd_neighbor_pack_bytes` for sizing; preserve `HdbscanOps` dispatch                                                                                                                                                                                                 |
| `crates/fast-hdbscan/src/estimator.rs`                                               | `core_search_memory`, `check_device_allocation`, `core_search_memory_accounts_for_bounded_and_full_routes`: account for actual packs; `upload_kd_tree` stays unchanged                                                                                                                                  |
| `crates/fast-hdbscan/tests/test_kdtree_search.rs`                                    | Preserve `check_kd_oracle`, `check_kd_oracle_high_dim`, `check_small_neighbor_boundaries`, `check_permuted_tree_row_oracles`, overflow cases and CPU/required-CUDA wrappers                                                                                                                             |
| `crates/fast-hdbscan/examples/kd_host_baseline.rs`, `crates/fast-hdbscan/Cargo.toml` | Retain tracked reporting tool and Rayon dev-dependency                                                                                                                                                                                                                                                  |

## Objective

Reduce CPU (CubeCL LLVM) time of the KD exact neighbor search behind HDBSCAN `core_distances`, with neighbor IDs, `f32::to_bits` distances and per-row diagnostics bit-identical to the current kernel, by:

- **P (packed candidates):** scanning leaf candidates from a transposed, position-tiled SoA copy, so that one `Vector<f32, W>` lane is one candidate and each lane accumulates its own features `0..d` in order (no sum is reordered);
- **O (leaf-ordered queries):** replacing cyclic row assignment with block-cyclic assignment over **leaf-ordered** positions of `tree.indices`, so that consecutive queries of one unit share traversal paths and hot leaf tiles;
- measuring P and O separately and combined (**C**), as the report's ordering experiment could not separate them;
- measuring a plain Rust/Rayon host traversal to expose CubeCL CPU runtime overhead.

CUDA production is out of scope: the CUDA route keeps the Phase 3 `kd_neighbors_kernel` unchanged.

Measured CPU starting point (Phase 4 pooled baseline, core stage, includes packing): 20k×16 212 ms, 50k×16 1095 ms, 20k×30 518 ms, 50k×30 3103 ms, 20k×8 29.5 ms, 20k×2 3.3 ms. The 16D query scans about 74% of all pairs, so candidate distance arithmetic dominates; this is what P vectorizes.

## Ordering relative to Phase 5

Phase 5 (CUDA launch-geometry sweep) is implemented; the user selected a fixed 64-thread CUDA neighbor preference by explicit override of the default regression screen. Unrelated KD launches and the frozen reference retain 32, and CPU policy remains unchanged. Phase 6 must start from the post-override Phase 5 baseline, preserve that CUDA neighbor geometry, and run its own Step 3 gate. The CPU baseline remains the accepted Phase 3 kernel; Phase 5 does not change its algorithm or launch policy. Phase 6's CPU decision must not silently retune the accepted CUDA geometry.

## Changes from the master plan

1. **Position-tiled SoA.** Tile the position axis of `tree.indices` globally: tile `t` holds `[tW, tW + W)`, as `soa[t][feature][lane]`, with `ceil(n/W)` tiles. A valid leaf `[start,end)` scans tiles `floor(start/W)..ceil(end/W)` and masks lanes outside its range. Sizing depends only on `n,d,W`, without prefix sums or readback. Keep this simplest approved experiment, not a per-leaf redesign. Misaligned starts cost masked lanes: at leaf size about 60, roughly +17% lane work at W=16 and +5% at W=8; measure this through width selection.
2. **Block-cyclic leaf order, not one contiguous chunk per unit.** One contiguous chunk of `⌈n/U⌉` positions per unit (`U` = units, 15 on this host) maximizes locality but fixes the load split at launch; traversal cost varies across regions (Gaussian tails scan more), and the CPU kernel has no work stealing. Units instead take blocks of `KD_QUERY_BLOCK = 64` consecutive positions (about one leaf: 20k rows give ~78-point leaves, 50k ~49), assigned block-cyclically. The value is predeclared, not tuned.
3. **Exclusive output writers by precondition.** The valid tree's indices are a permutation, so visiting each position once gives each original row exactly one writer for its output-backed heap. No ownership machinery is needed.
4. **Vector admission screen.** Per tile, OR the in-range lane comparisons `distance_lane <= worst_distance` at tile start. If true, check lanes in order with the exact Phase 1 admission test against the current worst registers. Including the self lane in the screen is conservative. This is not a NaN-input guarantee.
5. **Reuse of Phase 3 traversal code.** Node traversal (`padded_box_lower_bound`, padded bounds, register query, the `d > 32` runtime chunk path) is reused unchanged, so pruning decisions are produced by the same expressions. The padded data copy is kept because the query and the runtime-path bound evaluation read it.
6. **Width selection with a cap.** `W` is the first value of `client.io_optimized_vector_sizes(4)` not above `KD_SOA_MAX_LANES`. The cap is chosen by a predeclared selection batch (Step 4a) that is separate from the acceptance batches; the initial cap is 16.
7. **Three variant configurations, preserved baseline.** P, O and C are comptime configurations of one new CPU kernel (`kd_neighbors_cpu_kernel`); B is the preserved original baseline binary, not a configuration of that kernel. CUDA production source stays untouched during measurement. O copies the Phase 3 candidate scan verbatim; Step 6 removes unselected configurations and permits only the separately gated O-only fold. The frozen reference always stays unchanged.

## Non-goals

- Algorithm changes to `kd_neighbors_kernel` (CUDA production), `kd_candidates_kernel` (Borůvka), `kd_neighbors_reference_kernel`, `box_lower_bound`, `DeviceKdTree`, `HdbscanOps`, `upload_kd_tree`, the host `KdTree` builder and fusion metadata. Builder invariant/failure tests and low-level precondition documentation are required; builder production changes are not. Step 6 permits only the separately gated O query-order fold.
- CUDA launch geometry (Phase 5), CUDA plane kernel (Phase 7), register top-k (rejected Phase 2), node records (rejected Phase 4).
- Routing thresholds and the >30-feature kNN/ANN path.
- Any change to the per-pair arithmetic or feature order, traversal order, pruning, child ordering, admission predicate, heap layout, heapsort or diagnostics semantics.
- Shared memory or barriers (`PLANE_DIM` is 1 on the CPU runtime; no on-chip benefit).

## Invariants

1. Output IDs and distance bits are identical to the frozen reference (`kd_neighbors_reference_kernel`, `kdtree.rs:420–621`) and thus to Phase 3 production, for every harness case, every selectable `W`, and both query orders.
2. Per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) are identical. `leaf_positions_scanned` counts real positions `end − start` of each scanned leaf with `start < end`, never masked lanes.
3. Each `(query, candidate)` distance is `0.0 + Σ_{f=0}^{d−1} (q_f − c_f)²` accumulated in order `f = 0..d−1` in one lane. The register path uses exactly `d` features; the runtime path (`d > 32`) also uses exactly `d` (the SoA has no feature padding).
4. Original IDs everywhere: self-exclusion, tie keys and output rows use `indices` values; tiles and positions are memory layout only.
5. Each output row is written by exactly one unit, once per call, because valid indices are a permutation.
6. Error contract: `validate_tree` runs first. New buffers get checked sizing, the existing `u32` element-count assertion pattern of `pad_rows` (`:1154–1157`), inclusion in `core_search_memory`, and `check_device_allocation` calls.
7. CUDA route selection and production source are unchanged during measurement; the CPU kernel is never launched on a non-CPU client. Only Step 6's O-only gated fold permits a production-source change, without changing CUDA geometry or semantics. The frozen reference always remains unchanged.

## Equivalence argument

Fix a query row. The traversal is a sequence of events: pop, prune test, leaf scan, child bound evaluation, push. Child bounds and the root bound are computed by the unchanged Phase 3 helpers from the same padded inputs, so they are bit-identical. Within a leaf, the scalar kernel visits positions `start..end` in increasing order and, for each `candidate ≠ row`, applies `distance < worst || (distance == worst && candidate < worst_index)`, then replaces the root, sifts and refreshes the worst registers. The SoA kernel visits tiles in increasing order and, inside a tile, lanes in increasing order; lanes outside `[start, end)` are masked, so the in-range lanes enumerate exactly `start..end` in the same order. Each lane's distance is the same ordered sum of the same `f32` values (SoA values are bit copies of `data[indices[p]·d + f]`, and the query scalar is the same value Phase 3 holds in its register vector), computed by the same operations in a different SIMD position; this is bit-identical if both kernels make the same FMA-contraction choice, which Step 3 tests and Step 4d inspects. The tile screen skips a tile only if every in-range lane has `distance > worst_at_tile_start` or is NaN; because `worst` never increases, the scalar kernel would admit none of those lanes either. Admissions, heap contents and worst registers therefore evolve identically; by induction pruning, pushes and diagnostics are identical, and the final heapsort (copied verbatim) produces identical rows. Query order only changes which unit computes a row and when; rows are independent and ownership is exclusive (Invariant 5), so the output is identical for any schedule.

## Step 0: Baseline and prerequisites

1. Inspect actual HEAD and working changes as specified in the handoff. Record source hashes and diff hash in `retry/provenance.json`, with `rustc -V`, CPU model, `nproc`, CubeCL `num_cpu_cores`, `hardware.load_width` and full `io_optimized_vector_sizes(4)` list.
2. Preserve the existing accepted baseline executable and its SHA-256 before any rebuild. Build and archive a fresh baseline from verified actual baseline source (`cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages`) as `retry/bin/profile_hdbscan_stages.baseline`, recording source/binary hashes and explaining differences. Preserve CUDA neighbor64, unrelated32 and frozen-reference32 policies.
3. **API probe** (a throwaway `#[cfg(test)]` kernel, removed before the experiment diff is final; record outcomes in `provenance.json`). CubeCL 0.11.1 has `ClientProperties::io_optimized_vector_sizes` (`cubecl-runtime-0.11.1/src/client.rs:1786`) and `trait Size` with `value()` (`cubecl-core-0.11.1/src/frontend/element/base.rs:1006`); generic `N: Size` kernels take a `VectorSize` launch argument (`cubecl-core-0.11.1/src/runtime_tests` `kernel_decode`). Confirm on CPU, for `N ∈` every value of the host list:
   - a `#[cube(launch)]` kernel generic over `F: Float, I: Int, N: Size` with `#[define(F, I)]` and a `Tensor<Vector<F, N>>` argument created by `TensorArg::from_raw_parts(handle, [1], [vectors])`, as in `padded_vector_view` (`:1176–1189`);
   - `#[unroll] for lane in 0..N::value()` with `vector.extract(lane)`;
   - a scalar-to-`Vector<F, N>` broadcast constructor, element-wise `-`, `*`, `+=`;
   - optionally, a vector `<=` producing a lane-wise boolean vector whose lanes can be extracted. If absent, the screen uses scalar comparisons on extracted lanes (semantically identical).
   - a way to dump the CPU-generated MLIR/LLVM IR for one kernel (CubeCL compilation logging in 0.11.1). If none exists, record that; Step 4d then reports "not available".
4. Determine from CubeCL CPU runtime source whether the units of one cube execute concurrently. This does not change the design (which inherits the existing one-cube `num_cpu_cores`-unit geometry), but the report must state what the block-cyclic split actually distributes.

## Step 1: Harness

The existing `check_kd_neighbors_match_reference` compares production `launch_kd_neighbors` against `exact_neighbors_kd_reference` on IDs, distance bits and diagnostics, with diagnostics on and off, across `d ∈ {1,2,3,4,5,8,12,16,20,24,28,29,30,32,33,40}`, duplicates, grids, overflow inputs and `n ∈ {1,3,4,9,257,512,2000}`, on CPU and required CUDA. The large-`k` sweep (`k ∈ 1..=17, 24, 30`) applies at `d ∈ {2,16,30}`; the other feature sizes use `k ∈ {1,5,16,17}`, except `d ∈ {33,40}`, which use `k=5`. Keep all cases and the existing CUDA width and max-cubes coverage.

Changes and additions:

1. **Plan-parameterized entry.** Thread `KdNeighborPlan` through the existing `launch_kd_neighbors_impl`, alongside its existing `#[cfg(test)] preferred_width` and `max_cubes_override` parameters. Production `launch_kd_neighbors` supplies `kd_neighbor_plan(&client)`; a test entry `launch_kd_neighbors_with_plan` supplies an explicit plan without bypassing that implementation. The harness runs, on CPU, every case for: the production plan; `LeafSoa { lanes: w }` × `{Cyclic, LeafBlocks}` for **every** `w` in `client.io_optimized_vector_sizes(4)` (including `1`); and `PaddedRows × LeafBlocks`. CUDA keeps the existing legal-width sweep `[32,64,128]` and the separate `max_cubes_override=Some(2)` test, all with `PaddedRows × Cyclic`, plus an assertion that `kd_neighbor_plan` selects that plan on CUDA. CPU plans ignore `preferred_width` and retain the existing one-cube CPU geometry. Record the CPU harness wall time; if it exceeds 10 minutes in release mode, keep every `w` for the existing per-feature small-`k` cases but run the large-`k` sweep at `d ∈ {2,16,30}` only at the production `w` and `w = 1`, and say so in the log.
2. **Valid special inputs**, driven through the same comparison: `n ∈ {1,4,9,15,17,33,257}` at `d ∈ {2,16}`, covering short and partial tiles. Build trees from the actual finite input, including duplicates, ties, signed zero and large finite values whose squared distances can overflow. Do not inject NaNs or mutate trees into malformed fixtures.
3. **Builder invariant tests** in the core builder tests: assert indices are exactly a permutation of `0..n`; reachable leaf ranges cover `0..n` once with no gaps or overlaps; child ranges partition each internal range; every reachable child exists and topology terminates at leaves; bounds enclose all assigned points and child bounds; level counts obey builder limits and traversal stack capacity. Cover tiny trees, leaf/level boundaries, uneven partitions, duplicate-heavy data and all supported feature sizes. Retain meaningful builder failure tests for nonfinite input, invalid shape and size/level limits. These tests establish the precondition, not runtime content validation.
4. **Pack unit tests** (`check_pack_leaf_soa`, CPU only): for `n=37`, `d ∈ {1,3,30,33}`, every host `w`, and finite values including `-0.0`, use builder-produced permutation indices. Read back and assert by `to_bits` that `(t·d+f)·w+lane` equals `data[indices[tw+lane]·d+f]` for real positions and positive zero otherwise; slot IDs equal `indices[p]` or padding sentinel `n`.
5. **Sizing unit tests** for `kd_neighbor_pack_bytes` (Step 2e), including overflow on each new term, and an updated `core_search_memory_accounts_for_bounded_and_full_routes` (`estimator.rs:1467–1471`) case for the CPU plan.
6. **Direct launch test.** `kd_neighbors_enabled_diagnostics_match_disabled_launch` (`:1704–1830`) launches `kd_neighbors_kernel` directly and stays unchanged (the kernel is untouched). Add an analogous enabled/disabled test for `kd_neighbors_cpu_kernel` through `launch_kd_neighbors_with_plan` at every host `w`.
7. Integration tests (`tests/test_kdtree_search.rs`, host oracles including `check_permuted_tree_row_oracles`) must pass unchanged on CPU (now the SoA route) and CUDA.

## Step 2: Design

### 2a. Plan and route

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KdCandidateLayout { PaddedRows, LeafSoa { lanes: usize } }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KdQueryOrder { Cyclic, LeafBlocks }
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct KdNeighborPlan { layout: KdCandidateLayout, order: KdQueryOrder }

/// Chosen once per call from client properties.
pub(crate) fn kd_neighbor_plan(client: &cubecl::client::Client) -> KdNeighborPlan;
```

- Non-CPU client (`hardware.num_cpu_cores.is_none()`, the same discriminator as `kd_launch_geometry`): `PaddedRows × Cyclic`, dispatched to the unchanged `kd_neighbors_kernel`.
- CPU client: the arm under measurement, fixed by one private `const KD_CPU_ARM` (P = `LeafSoa × Cyclic`, O = `PaddedRows × LeafBlocks`, C = `LeafSoa × LeafBlocks`), dispatched to `kd_neighbors_cpu_kernel`. Arm binaries differ only in this constant (and `KD_SOA_MAX_LANES` in Step 4a); each one-line diff and binary hash is recorded. `lanes` = first value of `client.io_optimized_vector_sizes(4)` that is `≤ KD_SOA_MAX_LANES`.
- After Step 5, `KD_CPU_ARM` is removed and the CPU plan is fixed to the selected arm (Step 6).

The estimator uses the same function (through the CubeCL client of the selected device, as `allocation_limit` does at `estimator.rs:150–167`), so sizing and dispatch cannot disagree.

### 2b. Pack kernels (CPU route only)

```rust
#[cube(launch)]
fn pack_leaf_soa_kernel<F: Float, I: Int>(
    data_v: &Tensor<KdVector<F>>, // pad_rows(data) view, read as data_v[c * d_chunks + f / 4]
    indices: &Tensor<I>,
    soa: &mut Tensor<F>,     // 1D scalar, tiles * d * lanes
    slot_ids: &mut Tensor<I>,// 1D, tiles * lanes
    n: u32, d: u32, d_chunks: u32, lanes: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
);
```

Grid-stride over `tiles · d · lanes` output elements: `tile = e / (d·lanes)`, `f = (e / lanes) % d`, `lane = e % lanes`, `p = tile·lanes + lane`. If `p < n`: `c = u32::cast_from(indices[p])`, `soa[e] = data_v[c · d_chunks + f / 4].extract(f % 4)` (the **padded vector view**, read with the same index expression as Phase 3 `point_distance`, `kdtree.rs:148`), and the `f == 0` element also writes `slot_ids[p] = indices[p]`; else `soa[e] = 0.0`, and the `f == 0` element writes `slot_ids[p] = n`. In both branches `slot_ids` is written only by the `f == 0` element, so each slot has exactly one writer (CPU units run concurrently; see Step 0.4). Pure copies, no arithmetic.

All real positions and candidate IDs are in bounds under the valid-tree precondition. The padded vector gather is an ordinary data copy, not a malformed-index emulation.

Views: `soa` as `Tensor<Vector<F, N>>` with `tiles · d` vectors via a `soa_vector_view(tensor, tiles, d, lanes)` helper following `padded_vector_view` (`:1176–1189`), with the same length assertion and SAFETY comment.

### 2c. CPU kernel

```rust
#[cube(launch)]
fn kd_neighbors_cpu_kernel<F: Float, I: Int, N: Size>(
    data_v: &Tensor<KdVector<F>>,            // pad_rows(data): query and runtime-path bounds
    data: &Tensor<F>,                        // [n, d]: runtime-path query scalars
    soa: &Tensor<Vector<F, N>>,              // LeafSoa only (1-vector dummy otherwise)
    slot_ids: &Tensor<I>,                    // LeafSoa only
    indices: &Tensor<I>, starts: &Tensor<I>, ends: &Tensor<I>, is_leaf: &Tensor<I>,
    bounds_min_v: &Tensor<KdVector<F>>, bounds_max_v: &Tensor<KdVector<F>>,
    n: u32, k: u32, d: u32, d_chunks: u32,
    #[comptime] reg_features: usize,
    #[comptime] leaf_soa: bool,
    #[comptime] leaf_blocks: bool,
    #[comptime] diagnostics_enabled: bool,
    out_indices: &mut Tensor<I>, out_distances: &mut Tensor<F>, diagnostics: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
);
```

For `PaddedRows`, launch with `N = 1` (unused). Dummy buffers are small allocations; they are not counted as pack memory beyond their constant size.

**Work loop (one call site for the per-row body).**

- `Cyclic`: work items `w = ABSOLUTE_POS_X, w += units` over `0..n`; `row = w`.
- `LeafBlocks`: work items over `0..n` in blocks of `KD_QUERY_BLOCK` assigned block-cyclically by unit; process `row = indices[w]` exactly once per position.

**Per-row body** (`#[cube] fn kd_search_row`, inlined once): a verbatim copy of the Phase 3 row body (`:651–857`: output init, worst registers, stacks, register query load, root bound, pop/prune/children, heapsort, diagnostics), with only the leaf scan replaced when `leaf_soa`:

```text
if start < end:
    if diagnostics: leaf_positions_scanned += end - start
    tile_end = end                           // valid ranges satisfy end <= n
    for tile in start / W .. (tile_end + W - 1) / W:
        acc = broadcast(0.0)
        for f in 0..d:                       // #[unroll] for f < reg_features; runtime loop if reg_features == 0
            q = query[f / 4].extract(f % 4)  // register path; data[row * d + f] on the runtime path
            delta = broadcast(q) - soa[tile * d + f]
            acc += delta * delta
        base = tile * W
        hit = false
        #[unroll] for lane in 0..W:
            p = base + lane
            if p >= start && p < tile_end && acc.extract(lane) <= worst_distance: hit = true
        if hit:
            #[unroll] for lane in 0..W:
                p = base + lane
                if p >= start && p < tile_end:
                    candidate = slot_ids[p]
                    distance = acc.extract(lane)
                    if candidate != row && (distance < worst_distance
                            || (distance == worst_distance && candidate < worst_index)):
                        replace root, sift, refresh worst registers   // verbatim :704–749
```

There is no scalar out-of-range tail: valid leaf ranges end at or before `n`.

The `PaddedRows` path keeps the Phase 3 position loop (`:687–752`) verbatim. The heap replacement and heapsort are copied, not shared with `kd_neighbors_kernel`, so CUDA code does not change; this duplication is temporary and Phase 8 removes it (or Step 6 folds it, below).

### 2d. Host launch

In the existing `launch_kd_neighbors_impl`, with the plan threaded through both production and test entries, after the existing `validate_tree`, output allocation and `pad_rows` of data and bounds:

- `PaddedRows × Cyclic`: the existing `kd_neighbors_kernel::launch`, preserving production geometry and the test-only `preferred_width` and `max_cubes_override` controls.
- CPU plans: `pack_leaf_soa` if `leaf_soa`, then `kd_neighbors_cpu_kernel::launch` with `kd_launch_geometry(&client, n)` (unchanged CPU geometry: one cube, `num_cpu_cores` units) and `VectorSize` argument `lanes`. O has no additional pack launch. `exact_neighbors_kd` and diagnostics emission are unchanged.

### 2e. Sizing and allocation checks

```rust
/// Device bytes of all per-call KD neighbor packs for `plan`.
pub(crate) fn kd_neighbor_pack_bytes(plan: KdNeighborPlan, rows: usize, nodes: usize, features: usize)
    -> Option<usize>;
```

= `kd_padded_bytes(rows, nodes, features)` (all plans) plus, for `LeafSoa { lanes }`, `tiles · lanes · (features + 1) · 4`, with `tiles = ceil(rows/lanes)` (SoA floats plus slot IDs). All checked. O adds no proportional storage. For `rows=1000, nodes=63, features=5, lanes=16`, C/P: `(1000+126)·8·4 + 63·16·6·4 = 36,032 + 24,192 = 60,224` bytes.

Estimator: `kd_pack_bytes` comes from `kd_neighbor_pack_bytes(kd_neighbor_plan(client), …)`. Add `check_device_allocation` for SoA (`tiles·lanes·features` f32) and slot IDs (`tiles·lanes` i32) only when used. Assert element counts fit `u32`, as `pad_rows` does. No ownership allocation or sizing term exists.

### 2f. Rust/Rayon host baseline (measurement example)

The example and Rayon dev-dependency are existing tracked code at known HEAD d8faeab; preserve them and check actual status on restart, rather than recreating or discarding them. The specification below describes the retained tool, not a new exact oracle. Its `--verify` reports distance-bit mismatches despite matching IDs; ordered Rust arithmetic is unfused while baseline diagnostic LLVM contains `fmuladd`. Report those results without weakening device bitwise gates or treating the host comparison as an acceptance gate.

Retain `crates/fast-hdbscan/examples/kd_host_baseline.rs` and its `rayon = { workspace = true }` dev-dependency. Usage: `kd_host_baseline INPUT ROWS FEATURES [--k 5] [--runs 5] [--threads T] [--verify]`. It builds the tree outside timing and times scalar ordered-f32 traversal with Rayon over output rows in a pool of `T` threads (default CubeCL core count). JSON reports run times and median; `--verify` compares IDs and distance bits against the public CPU operation. Mismatches are reported findings, not acceptance gates.

## Step 3: Correctness verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-hdbscan-core
cargo test -p fast-hdbscan --release --lib kdtree -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib kdtree -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib estimator -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --test test_kdtree_search -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --features tracing \
  --example profile_hdbscan_stages -- --nocapture
cargo build -p fast-hdbscan --release --example kd_host_baseline
uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py
```

**Gate.** This is the experimental matrix (all arms); after Step 6 the final-gate matrix applies. Require all gates to pass without hardware skips and every valid-tree case to match IDs, distance bits and diagnostics for every plan and host `w`. Run the plan-parameterized harness on measurement source; each arm binary additionally passes the tracing example test built with its `KD_CPU_ARM`. Archive fresh logs under `retry/logs/`, never historical `phase6/`.

**Mismatch triage.** Never change the reference kernel or tests to match.

- Distances differ, IDs/diagnostics follow: FMA contraction or operation-order difference between the vector and scalar sums. Inspect the generated IR (Step 0.3). Fix by expressing the vector accumulation so it matches the scalar kernel's contraction (e.g. the same `delta * delta` then `+=` shape the scalar code uses), never by changing the scalar kernel. **Width policy:** every width in the host list must pass; there is no width-exclusion mechanism. If any width stays unfixed, P and C are rejected (O is unaffected).
- Distances equal, diagnostics differ: tile range or lane mask error, or the screen skipped an admissible lane.
- Some rows uninitialized or differing only under `LeafBlocks`: block assignment or permutation-position scheduling error.

**End to end.** Final labels must be exactly equal across all arms within every benchmark cell (Step 4).

## Step 4: Performance measurement

All CPU. Verify these exact historical input paths and SHA-256 values from `phase6/provenance.json` before reuse; they do not all reside in `inputs/`:

| Cell   | Input path                                                            | SHA-256                                                            |
| ------ | --------------------------------------------------------------------- | ------------------------------------------------------------------ |
| 20k×16 | `artifacts/hdbscan_core_investigation/baseline_20000/raw/input.f32le` | `b2df07faa30a6814fffb398b140b2a8edcc8ab88fbde802cc85dcce3b659925e` |
| 50k×16 | `artifacts/hdbscan_core_investigation/baseline_50000/raw/input.f32le` | `99e4275cee1a443d0ac2b2e3434e1b81a86b5726d596f13e1455a1d9f1722a81` |
| 20k×30 | `artifacts/hdbscan_core_investigation/dim30.f32le`                    | `a7f6fa44daa73c5c75e55a56f57c383db64043a2409180a3c4d12af1848d935e` |
| 50k×30 | `artifacts/cubecl_kernels_perf/inputs/input_50000_30.f32le`           | `7e17b48290ed5ea9335caf4602f35d8526cd3c24d9336c5e493d549bba498e2e` |
| 20k×8  | `artifacts/hdbscan_core_investigation/dim8.f32le`                     | `6dc2e889f243d4d075b1f6a75e8d18d7cabb02c1466998e385b70c7a8e2be1f5` |
| 20k×2  | `artifacts/hdbscan_core_investigation/dim2.f32le`                     | `66235e50a1f68995e1fe39f8914f1551bafaa6a2ebb34c0b0ea9c8b7ef173239` |

Command: `<binary> <input.f32le> <rows> <features> --backend cpu --runs 3 --min-samples 5`. Each process reports core-stage and `core_neighbor_kernel` medians (including all packing), full-fit median and labels. Missing/mismatched files require fresh recorded input provenance, not an unsupported claim of historical comparability.

Runner: copy `artifacts/cubecl_kernels_perf/phase4/run_alternating.py`. Allowed changes, and only these: binary/input paths, an arm list (`--arms`) and per-cell order pattern, `--backend cpu` only, a `--batch` label (`S`, `A`, `B`), and a fresh output directory (still refusing to overwrite). Record the diff against the Phase 4 copy.

### 4a. Width selection (separate, before acceptance)

Arms: C built with `KD_SOA_MAX_LANES` set to each of the three widest values of the host list, mapped before any run as `X` = widest, `Y` = second, `Z` = third (expected 16, 8, 4 on AVX-512; mapping recorded in Step 0). Cells 20k×16 and 20k×30, one batch `S` of six blocks forming a Williams design (each arm twice in each position, each ordered adjacent pair once per three blocks): `XYZ | YZX | ZXY | ZYX | XZY | YXZ` (six processes per arm). Rule, predeclared: start from the widest; select a narrower width only if its core median is at least 2% lower than the currently selected width's in **both** cells. The selected cap is fixed for all later steps. Batch `S` data is reported but never pooled into acceptance statistics.

### 4b. Isolation and acceptance timing (H2)

- Arms: **B** (baseline binary), **P**, **O**, **C** (variant source at the Step 4a cap, differing only in `KD_CPU_ARM`).
- Cells: primary 20k×16, 50k×16, 20k×30, 50k×30; guards 20k×8 and 20k×2 (smallest work, where packing weighs most).
- Per cell and batch, four blocks of four processes forming a 4×4 Williams square (every arm once in every position; every ordered adjacent pair once). Batch `A` = `B,P,C,O | P,O,B,C | O,C,P,B | C,B,O,P`. Batch `B` = the 16-process list of `A` reversed = `P,O,B,C | B,P,C,O | C,B,O,P | O,C,P,B`, which balances temporal direction. The runner encodes both lists literally. Cells serial. Two predeclared batches, `A` then `B`, each over the full matrix. No further batches; neither batch is selected.
- Aggregation per cell and arm `X ∈ {P,O,C}`: pool the eight process medians per arm; `Δ_X = median(X)/median(B) − 1`. Per-batch `Δ_X` is reported too: the master H2 says to compare within a batch, and pooling two predeclared, direction-balanced batches follows the Phase 4 precedent; this is recorded as a deviation. Paired screen, fixed before measurement: in each block, the ratio `X/B` of that block's two processes regardless of position; eight ratios per arm and cell, reported with min–max. The screen is a consistency check, not a significance test.
- Component attribution (reporting-only, derived): `median(C)/median(P)` (effect of O given P) and `median(C)/median(O)` (effect of P given O) per cell.

### 4c. Rayon host baseline (reporting-only)

After batch B: three processes of `kd_host_baseline --runs 5 --verify` per primary cell, `--threads` = CubeCL `num_cpu_cores`. Report the median traversal time next to the pooled `core_neighbor_kernel` medians of B and the selected arm, and the verify result. The ratio is an implementation-level comparison (runtime, scheduling, scalar vs vector codegen and buffer handling all differ), not an isolated measure of runtime overhead and not an acceptance input.

### 4d. Codegen evidence (reporting-only)

Known diagnostic mechanism: build with `--features tracing,cubecl/pliron-dump` and set `CUBECL_DEBUG_PLIRON=<output-directory>` when executing. CPU output includes `llvm.ll` and optimized `llvm.opt.ll`. Isolate diagnostic builds using a separate `CARGO_TARGET_DIR` (or preserve and hash all timed binaries first); diagnostic executables are never timing executables. Dump selected CPU arm and baseline at d=16 under `retry/generated/`; report `<W x float>` candidate loads/arithmetic and FMA/contraction in both. Reconfirm availability on actual dependency versions; record failures honestly, not as missing arithmetic evidence proved absent.

### 4e. JIT/cold start (reporting-only)

Copy `phase4/cold_start.py` unchanged except paths and backend restricted to CPU; d ∈ {2, 16, 30}; baseline and selected-arm binaries. Report first-fit medians and first-minus-warm using the same binary's pooled H2 full-fit medians (sources recorded). P/C add one compiled variant per `(d,W,order)` plus the SoA pack kernel; O adds only the query-order variant. There is no ownership compilation/launch cost.

### 4f. Final pipeline validation

With final production code after Step 6, run `bash scripts/analyze_hdbscan_stages.sh --rows 20000` and `bash scripts/analyze_hdbscan_stages.sh --rows 100000`; archive `retry/logs/analyze_*.log`. Require actual CPU/CUDA execution and exact label equality. Existing CUDA kernel filters need no change because kernel names are unchanged; CPU timing is span-based. These are single runs, not interleaved comparisons.

## Step 5: Decision rule

Thresholds (Phase 4 effect sizes, consistency scaled to eight pairs): **gain** = pooled `Δ ≤ −2%` and at least 7 of 8 paired ratios below 1; **regression** = pooled `Δ ≥ +3%` and at least 7 of 8 above 1; anything between is "no effect".

An arm `X ∈ {P, O, C}` **passes** by default when:

1. every Step 3 gate passes for the source and `X`'s binary, and labels are equal in all processes;
2. at least three of the four primary cells show a gain;
3. no primary or guard cell shows a regression.

**Selection.** If no arm passes, reject Phase 6. Otherwise let `G` be the passing arm with the lowest geometric mean of pooled primary core medians. Among passing arms within +2% of `G` in every primary cell (including `G`), select the simplest (simplicity `O < P < C`). This is one comparison against `G`, not iterated. Correctness criterion (1) is mandatory. Criteria (2)–(3) and the selection rule are the default guide, not absolute vetoes: make an evidence-backed judgement call when timings are noisy or the trade-off is reasonable, weighing both batches, paired ratios, absolute and relative effects, guard-cell variability, memory (`kd_neighbor_pack_bytes` increase) and JIT cost. A threshold crossing alone does not establish benefit or harm, and "no effect" does not establish equivalence. This discretion does not authorize extra batches or selecting samples.

**Failure handling.**

- Fail of (1): a correctness defect; fix it, never relax tests. If unfixable for an arm, that arm cannot be selected.
- A guard regression only at 20k×2 with sub-millisecond absolute cost (Phase 3 precedent of non-replicating 2D noise) may be accepted only with a documented judgement; it is not silently ignored.
- **Document the judgement** in `summary.json` and the Phase 6 results in `hdbscan_perf_report.md`: decision, selected arm and `W`, each unmet criterion, supporting and contrary measurements (variability, absolute and relative costs), and the trade-off and why it is reasonable. Distinguish measured gains from hypotheses.
- **Rejection** reverts only Phase 6 production code (plan/route types, CPU kernel, pack kernels, views, sizing and allocation-check changes, tests that only exercise removed code). The `kd_host_baseline` example is retained as a measurement tool (it does not touch production code). Keep the evidence directory and an `experiment.diff`; record the negative result with numbers. Unrelated working-tree changes are preserved.

## Step 6: Finalization of the selected arm

- **C or P:** remove `KD_CPU_ARM`; fix the CPU plan to selected `(LeafSoa { lanes }, order)` with Step 4a cap. Remove unselected branches (`PaddedRows` from the CPU kernel; `LeafBlocks` if P). Keep every host `w` in the harness. No ownership code or storage is introduced or retained.
- **O:** do not ship a duplicated per-query kernel. Fold only block-cyclic leaf order into `kd_neighbors_kernel` as a `#[comptime] leaf_blocks` flag (CPU `true`, CUDA `false`); delete `kd_neighbors_cpu_kernel` and the SoA pack. Preserve CUDA64 and unrelated/frozen32. Archive final-source/binary hashes and apply: (a) final-matrix Step 3 with required CUDA; (b) CUDA two batches `A`,`B` of `B,V,V,B,B,V` on four primary cells plus 20k×2, unchanged Phase 4 thresholds (six pairs), H3 at 20k×16/30; any CUDA regression, register increase or local-memory increase blocks the fold; (c) CPU two batches of `B,V,V,B,B,V`, folded versus baseline, all six cells, Phase 4 thresholds, requiring ≥3 primary gains and no cell regression. Reject O if (b) or (c) fails, rather than shipping a duplicate. The frozen reference stays untouched.
- **Final-gate matrix.** After finalization, Step 1 item 1 shrinks to the plans the final source still contains: the production plan on CPU at every host `w` (C/P) or the production plan alone (O), plus `PaddedRows × Cyclic` on CUDA with the existing legal-width sweep and max-cubes test preserved. Removed arms are not kept behind `#[cfg(test)]`; their equivalence evidence is the measurement-source Step 3 logs, which are retained. Re-run Step 3 with this matrix on the final source, then Step 4f.

## Step 7: Deliverables

- Code: plan/route, CPU kernel, pack kernels and helpers, sizing and allocation checks, tests, `kd_host_baseline` example and its dev-dependency.
- `retry/`: fresh provenance (including API probes and units concurrency), hashed binaries and per-arm diffs, runner/diff, batch `S`/`A`/`B` records, Rayon JSON, generated IR, cold-start JSON, logs and `summary.json` with per-cell/per-arm results and decision. Historical `phase6/` remains untouched.
- `hdbscan_perf_report.md`: "Phase 6 results", scoped to the measured CPU cells on this host, including negative or partial results.
- `cubecl_kernels_perf_plan.md` Phase 6: status line, link to this plan, valid-tree precondition, position-tiled SoA, block-cyclic order and selected `W` cap.
- `AGENTS.md`/`README.md`: re-check with a search at completion. Update only where they describe KD traversal, device buffers or resource sizing (none currently enumerate the KD CPU schedule); list the new example only if a README section enumerates examples.

## Risks and mitigations

| Risk                                                       | Mitigation                                                                                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Vector vs scalar FMA contraction changes distance bits     | Bit-equality at every host `w` on CPU; Step 4d IR inspection; triage procedure; any unfixed `w` rejects P/C                          |
| Lane waste at misaligned leaf starts (est. +17% at W = 16) | Quantified up front; Step 4a selects `W` empirically; per-leaf layout recorded as a later experiment                                 |
| Load imbalance from leaf order                             | Block-cyclic `KD_QUERY_BLOCK = 64`; isolation arm O measures order alone                                                             |
| Caller violates valid-tree precondition                    | Document precondition; test builder invariants and rejection of invalid builder inputs, not runtime scans                            |
| JIT cost of `(d, W, order)` variants on CPU LLVM           | Only the selected `W`/order compiles in production; cold start reported (4e)                                                         |
| Extra memory (SoA plus slot IDs)                           | Checked sizing and allocation limits; reported in results; O has no added proportional storage                                       |
| Small cells dominated by packing                           | 20k×2 and 20k×8 guards; regression screen                                                                                            |
| Units of one CPU cube not concurrent as assumed            | Step 0.4 establishes the facts; design inherits the existing geometry either way                                                     |
| Duplicate heap code in two kernels                         | Copied verbatim to keep CUDA untouched; O folds into the CUDA kernel source only with CUDA verification; Phase 8 removes duplication |
| Gains specific to Gaussian 16D/30D inputs                  | Scope stated in results; no generality claim                                                                                         |
