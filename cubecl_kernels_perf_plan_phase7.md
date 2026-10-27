# Phase 7 detailed plan: CUDA plane-cooperative leaf-tile kernel

This document details Phase 7 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md) (F5 in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups"). It applies the accepted Phase 6 CPU ideas, leaf-ordered packed candidates and leaf-ordered query blocks, to the CUDA kernel. On the CPU, vector lanes are candidates. On CUDA, plane lanes are queries and each staged candidate is broadcast to the whole plane. It builds on accepted Phases 1, 3, 5 (CUDA neighbor width 64 by override) and 6 (CPU C16). Phases 2 and 4 were rejected. Source references are symbols in current `crates/fast-hdbscan/src/kernels/kdtree.rs`; inspect actual source before implementation.

**Status:** planned. It has not been implemented, measured or authorized for execution. Implementation needs a separate user execution request.

## Motivation and evidence

- **Crossover.** After Phase 6, the user observed CPU C16 overtaking CUDA in `core_distances` at roughly 300k rows. Phase 6 pooled core medians are 0.038 s (20k×16) and 0.246 s (50k×16) on CPU. No archived 300k comparison exists; Step 0 tests the reported crossover using baseline B only.
- **Weak pruning.** At 20k, a query scans 74.1% of all pairs at 16D and 96.5% at 30D (report, "Features" table). The search is close to O(n²) with traversal overhead. Its cost depends mainly on how cheaply each `(query, candidate)` pair gets its operands.
- **Current CUDA kernel.** `kd_neighbors_kernel` runs one query per thread, with incoherent per-thread traversal, and gathers candidates through `indices`. The baseline counters were 70–79% long-scoreboard stalls.
- **Cache hypothesis (unverified).** The RTX 3080 L2 is 5 MB. It holds padded rows only up to about 80k rows at 16D (64 B per row) and about 40k at 30D (128 B per row). The 9800X3D has 96 MB of L3. Step 0 examines baseline cache behavior through 300k to test this hypothesis; counter correlations support rather than prove causation. Nothing below depends on that hypothesis being true.
- **Ordering alone is weak on CUDA.** Spatial reordering gave CUDA core −8.9% at 20k and −2.3% at 50k, with overlapping ranges (report, "Spatial ordering"). Neighboring lanes touch the same leaves but diverge on pruning, so their loads still issue separately. A CUDA-only port of Phase 6's leaf-block order is therefore not an arm here.
- **Mechanism.** A plane stages each visited candidate chunk once with coalesced loads, and all `PLANE_DIM` lanes read it from shared memory as a broadcast. Global candidate traffic falls from once per lane to once per plane, about 32×. This is the CUDA counterpart of C16, which uses each SIMD candidate load for 16 pair computations.

## Invariants

All master-plan invariants apply unchanged. These are the ones this phase depends on most:

1. **Bit-identical IDs and distances.** Each pair distance is the f32 sum over features `0..d` in order, computed by the existing `chunk_distance` on `KdVector` chunks. Staged candidates are bit copies of padded rows, so the operands and the expression match the current kernel. Contraction is still tested, not assumed (Invariant 5).
2. **Visit order is free.** The tile traversal changes the order of nodes, leaves and candidates. The result is still the k smallest `(distance², original ID)` keys with pruning `lower <= worst`, evaluated per lane.
3. **Original IDs.** Self-exclusion, tie keys and output rows use `indices[position]`. Leaf order is only a memory layout and a query schedule. Public inputs and downstream arrays are never permuted.
4. **Valid-tree contract** (Phase 6). Finite data and builder-produced trees, or equivalent low-level inputs, are required. `indices` is a permutation, so each original row has exactly one writer. No runtime content scan, ownership map or tree validation is added. Existing shape/dtype validation (`validate_tree`) and checked sizing stay.
5. **Error contract.** New buffers use `checked_*` sizing and are counted in `kd_neighbor_pack_bytes` and `core_search_memory`. Device allocation checks cover the new pack.

## Changes from the master plan's Phase 7

1. **Independent planes with `sync_plane()`, not a cube-uniform stack with `sync_cube()`.** Each plane has its own shared slice and its own uniform traversal. Planes in a cube never synchronize with each other. This gets past the 16-blocks-per-SM limit without requiring cube-wide traversal agreement, and avoids a barrier under masked control flow. CubeCL 0.11.1 provides `sync_plane` (`cubecl-core/src/frontend/synchronization.rs:35`), which lowers to `__syncwarp()` on CUDA (`cubecl-cpp/src/cuda/dialect.rs:69`). The CUDA runtime enables `Plane::Ops` and enables `Plane::Sync` from arch 70 (`cubecl-cuda/src/runtime.rs:233–243, 363`). The RTX 3080 is sm_86.
2. **Phase 1 output-backed heap, not register top-k.** Phase 2 was rejected. The tile kernel copies the Phase 1 admission, sift-down and final heapsort from `kd_neighbors_kernel` verbatim, including the cached `worst_distance`/`worst_index` registers.
3. **Leaf-ordered padded rows (AoS), not SoA.** My earlier recommendation was the Phase 6 SoA pack at tile width 32. AoS works better on CUDA:
   - a chunk of `PLANE_DIM` consecutive positions is one contiguous range of `PLANE_DIM · d_chunks` `KdVector`s, so a flat lane-strided copy into shared memory is fully coalesced;
   - shared reads are broadcasts in either layout;
   - the compute loop keeps the exact Phase 3 `chunk_distance` expression on `KdVector` operands, which keeps the arithmetic closest to the current kernel.

   The CPU C16 SoA pack is not reused.

4. **Flat control arm.** A comptime `traverse = false` configuration of the same kernel scans all positions in staged chunks, with no tree. It shows whether the tree still pays for itself on GPU at 16–30D. It is also the strictest correctness check: it does no pruning, so a mismatch against the reference would expose inexact pruning in the reference or baseline. Stop and report such a mismatch; do not relax any test.
5. **Dimension gate selected in advance.** Union traversal visits a leaf when any lane needs it. At 2D (0.6% scan) the union can cost several times the per-query work. A selection batch, kept separate from acceptance batches, sets the smallest dimension that uses the tile kernel. Lower dimensions keep the per-query kernel.
6. **Bounded experimental row counts.** Initial evidence gathering and hypothesis testing in Step 0 use baseline B at 200k and 300k as well as smaller sizes. This is the only exception: all experimental-arm benchmarks, selection, acceptance, post-change counters and reporting runs are limited to 20k, 50k and 100k. Script options support the larger baseline workloads. At 100k the padded candidate rows already exceed the nominal L2 capacity at both 16D and 30D. Improvements are plausible at these sizes, not guaranteed: staging overhead and cache residency can make speedups size-dependent. Acceptance and rejection apply to the experimental measured range only; baseline large-size evidence does not establish new-arm performance there. Do not expand experimental scope to rescue a failed phase.
7. **Still deferred.** Double-buffered staging (`memcpy_async`) and the Borůvka `kd_candidates_kernel`.

## Non-goals

- CPU production code. `kd_neighbor_plan` keeps returning C16 (`LeafSoa`, `LeafBlocks`) on CPU, and `kd_neighbors_cpu_kernel` is unchanged. CPU plane size is 1, so the tile plan cannot be selected there.
- `kd_candidates_kernel`, the >30-feature kNN/ANN route, and routing-threshold changes.
- Unrelated KD launches and the frozen `kd_neighbors_reference_kernel` (both at width 32).
- Reordering feature sums, `plane_sum`, or matmul-form distances (rejected under Invariant 1).

## Affected source and test map

New symbols are planned, not existing implementations.

| Path                                                                                                             | Responsibility / symbols                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `crates/fast-hdbscan/src/kernels/kdtree.rs`                                                                      | Existing `KdNeighborPlan`, `KdCandidateLayout`, `KdQueryOrder`, `kd_neighbor_plan`, `kd_neighbor_pack_bytes`, `launch_kd_neighbors_impl`, `kd_neighbors_kernel`, `chunk_distance`, `padded_box_lower_bound`, `load_query`, `pad_rows`, `kd_launch_geometry_with_width`, `launch_count`. Planned: `KdCandidateLayout::LeafRows`, `KdQueryOrder::PlaneTiles { planes_per_cube }`, `pack_leaf_rows_kernel`, `kd_neighbors_tile_kernel`, tile geometry and shared-memory sizing, and harness additions to `check_kd_neighbors_match_reference` and the CUDA wrappers (`kd_neighbors_match_frozen_reference_cuda`, `kd_neighbors_capped_grid_stride_matches_reference_cuda`) |
| `crates/fast-hdbscan/src/kernels/mod.rs`                                                                         | Exports for plan/sizing; preserve `HdbscanOps` dispatch                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `crates/fast-hdbscan/src/estimator.rs`                                                                           | `core_search_memory` call site (`kd_neighbor_plan`, `kd_neighbor_pack_bytes`, around line 1102) and its sizing test (around line 1497): account for the CUDA tile pack                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `crates/fast-hdbscan/tests/test_kdtree_search.rs`                                                                | Keep `check_kd_oracle`, `check_kd_oracle_high_dim`, `check_small_neighbor_boundaries`, `check_permuted_tree_row_oracles`, the overflow cases and the CPU/required-CUDA wrappers passing                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `analyze/hdbscan_stages_common.py`, `analyze/profile_hdbscan_kernels.py`, `analyze/tests/test_hdbscan_stages.py` | Route-aware kernel families: the KdTree neighbor family must resolve to the kernel actually dispatched (`kd_neighbors_tile_kernel` or `kd_neighbors_kernel`), plus `pack_leaf_rows_kernel`                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `analyze/benchmark_hdbscan_stages.py`, `analyze/profile_hdbscan_counters.py`                                     | `--rows` choices are currently `20000/50000/100000`; add 200k and 300k with test updates                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |

## Kernel design

### Plan and routing

`kd_neighbor_plan` gains the inputs it needs (client properties and `features`). It selects `LeafRows × PlaneTiles { planes_per_cube }` only when all of these hold:

- the client is not CPU (`num_cpu_cores.is_none()`);
- `features.plane` contains both `Plane::Ops` and `Plane::Sync`;
- `plane_size_min == plane_size_max`, which is the tile width `T` (32 on CUDA);
- `features ≤ KD_REGISTER_QUERY_MAX_FEATURES` (32), so the query is held in registers (`reg_features > 0`);
- `features ≥` the gate selected in Step 3;
- the shared-memory requirement fits `max_shared_memory_size`.

Otherwise CUDA keeps `PaddedRows × Cyclic` (the per-query kernel at width 64). Plan selection is deterministic and has no runtime fallback after launch. `kd_neighbor_pack_bytes` must use the same plan as the launch. The estimator call site passes `features`.

### Pack: `pack_leaf_rows_kernel`

- **Output.** `leaf_rows[p · d_chunks + c]` for positions `p ∈ 0..⌈n/T⌉·T`, as `KdVector` chunks.
- **Contents.** Real positions hold the zero-padded row `indices[p]`, copied bitwise from unpadded `data`, with `+0.0` tail lanes as in `pad_rows`. Positions `p ≥ n` are all zero.
- **No slot-ID buffer.** The original ID of position `p` is `tree.indices[p]`, which is already leaf-ordered. Lanes mask `p ≥ n`.
- **No `data_padded`.** The tile path loads each query from `leaf_rows[p]`. Bounds evaluation with `reg_features > 0` reads only the register query and the padded bounds, so `pad_rows(data)` is skipped. `pad_rows` stays for `bounds_min`/`bounds_max`.
- **Sizing.** `⌈n/T⌉ · T · 4⌈d/4⌉ · 4` bytes plus the existing padded-bounds bytes, all checked. Tests cover equality with `pad_rows(data)` gathered by `indices`, the zero tail, and n values that are not multiples of `T`.

### Schedule

- **Geometry.** Cube dim `= planes_per_cube · T`, with plane index `UNIT_POS / PLANE_DIM` and lane `UNIT_POS_PLANE`.
- **Tiles.** Global plane `g` owns positions `[g·T, g·T + T)`. Tiles are assigned grid-stride over `⌈n/T⌉` and bounded by `launch_count` limits. A capped-grid test forces multiple tiles per plane.
- **Lanes.** Lane `p` is active iff `p < n`. Its query row is `indices[p]`. It loads its query from `leaf_rows` with `load_query` semantics.
- **Inactive lanes** still copy data, synchronize and take part in every plane operation. They vote `false`, contribute `+∞` to `plane_min`, and write nothing.

### Traversal (`traverse = true`)

**Stacks.** The node stack and bound stack are per-lane `Array`s of `KD_STACK_LEN`, as in the current kernel:

- node contents and stack index are plane-uniform;
- bounds are per lane;
- depth bound is unchanged: at most two pushes per level, one pop.

A shared node stack is an optional later step. Adopt it only if H3 shows that local-memory traffic from the stacks matters.

**Pop.** `own = active && bound_stack[top] <= worst_distance`, and `visit = plane_any(own)`. The plane visits the node if `visit` is true. When the node is visited, the count is incremented on every lane (diagnostics).

**Leaf.** For each chunk `[s, min(s + T, end))` of the leaf range:

1. Copy the chunk cooperatively with `len = min(T, end - s)`. Lane `l` copies elements `l, l + T, …` only while the element index is `< len · d_chunks`; no global load is issued for other slots. Lane `j < len` also copies `indices[s + j]` into a shared ID slot. Only candidates below `len` are consumed. Global pack padding does not imply leaf alignment.
2. `sync_plane()`.
3. If `own`, the lane loops over the staged candidates in order:
   - skip the candidate if its ID equals its own row;
   - compute the distance from shared chunks with `chunk_distance`, in the same chunk/lane order as `point_distance`'s register path;
   - admit and sift exactly as in Phase 1.
4. `sync_plane()` before the next copy overwrites the buffer (write-after-read).

The copy and both syncs run in plane-uniform control flow. Only the per-lane compute loop is masked.

**Internal node.** Every lane computes `left_bound` and `right_bound` with `padded_box_lower_bound`. All lanes read the same bounds addresses, so these loads are broadcasts.

- `need_l = plane_any(active && left_bound <= worst)`, and `need_r` likewise.
- Push order uses `plane_min` over the bounds of lanes that need each child (masked lanes contribute `+∞`): the smaller plane-min is pushed last, so it is popped first. Ties pop left first, matching the existing `left_bound <= right_bound` rule.
- Each lane stores its own bound alongside the uniform node.

**Output.** After traversal, each active lane runs the existing in-place heapsort on its row and writes diagnostics. Each heap row has one writer by permutation.

### Flat control (`traverse = false`)

Same tiles, staging, admission and output. Chunks cover positions `0..n` in order. No tree node or bound is read.

### Diagnostics

Meanings change in this kernel. Document them in `emit_kd_diagnostics` output and in the report:

- `nodes_popped` counts tile-uniform pops and is equal for all lanes of a tile;
- `leaf_positions_scanned` counts positions the lane actually evaluated (`own`);
- under `traverse = false`, `nodes_popped = 0` and `leaf_positions_scanned = n`.

Diagnostics-enabled and diagnostics-disabled runs must give identical IDs and distances.

### Shared memory and resources

- **Per plane:** `T · d_chunks · 16` bytes of candidate chunks (≤ 4096 at `d_chunks = 8`) plus `T · 4` bytes of IDs.
- **Per cube:** `planes_per_cube` times that. Shared lengths are comptime from `d_chunks` (derived from comptime `reg_features`), `T` and `planes_per_cube`.
- **Host check:** the requirement must fit `max_shared_memory_size`. Step 0 confirms the exact 0.11.1 shared-memory constructor and indexing API with a probe before kernel work.

## Steps

### Step 0: Provenance, baseline crossover and hypothesis evidence (no production change)

1. **Provenance.** Record `git status`, HEAD and the diff, and never revert user work. Archive the current release/tracing `profile_hdbscan_stages` binary as baseline **B** with its SHA-256. Use the evidence directory `artifacts/cubecl_kernels_perf/phase7/`.
2. **Inputs.** Generate or reuse 20k, 50k, 100k, 200k and 300k inputs at 16D and 30D with the same generator, parameters and seed as existing cells (`make_input(rows, features, 1.0, 1.0, 42, …)`). Record hashes in `inputs.json`. Extend the benchmark and counter tools' `--rows` choices to include 200k/300k and test argument acceptance. Larger inputs are for Step 0 baseline evidence only.
3. **CPU/CUDA crossover.** Run B on CPU and CUDA at 100k, 200k and 300k × 16D/30D, three processes each, with the CPU and CUDA processes alternating. Report core and full-fit medians, CPU thread count, observed crossover (or its absence) and labels equal. This is initial evidence only, not an acceptance input.
4. **Scan fraction.** Run B with `FAST_HDBSCAN_KD_DIAGNOSTICS=1` at 50k, 100k, 200k and 300k × 16D/30D, and report mean nodes popped and leaf positions scanned per n². This tests whether baseline pruning improves with n.
5. **L2 hypothesis.** Run H3 on B's `kd_neighbors_kernel` at 50k, 100k, 200k and 300k × 16D. Add the metrics `lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum`, `l1tex__t_sector_hit_rate.pct`, `l1tex` sectors per request, long-scoreboard stall share, achieved occupancy, and registers/local memory. Falling L2 hit rate and rising DRAM bytes per scanned pair support the hypothesis within this range only. Record the outcome either way. A refutation does not cancel the phase, because the tile kernel also removes divergent gathers, but the report must not attribute gains to L2.
6. **API probes.** These are throwaway test kernels, not production code. Run each on CUDA:
   - per-plane shared slices indexed by plane;
   - `sync_plane` inside plane-uniform loops, with the generated source checked for `__syncwarp()` and no `__syncthreads()`;
   - `plane_any` and `plane_min` over f32 with `+∞` masking;
   - `UNIT_POS_PLANE` and `PLANE_DIM` values at cube dims 32, 64 and 128.

   Record the generated-source mechanism used. If a required API is missing, stop and report.

### Step 1: Implementation (experimental configurations)

- **Code:**
  - `pack_leaf_rows_kernel`;
  - `kd_neighbors_tile_kernel` with comptime `traverse`, `planes_per_cube`, `T`, `reg_features` and `diagnostics_enabled`;
  - plan, sizing and estimator accounting;
  - test-only plan overrides, so every configuration runs from tests and measurement binaries.
- **Measurement binaries:** one per arm. **T{1,2,4}** is the tile tree kernel with 1, 2 or 4 planes per cube; **F{p}** is the flat control. CPU code stays byte-identical to B; Step 2 checks this with a diff.
- **Production files** stay at B until Step 6 finalization. Measurement uses archived, hashed binaries with per-arm diffs.

### Step 2: Correctness gates (H1, required CUDA)

All gates run with `FAST_UMAP_REQUIRE_CUDA=1`; CUDA tests must not be skipped.

1. **Frozen-reference equality.** IDs and `f32::to_bits` distances of every arm (T1, T2, T4 and F) must equal `kd_neighbors_reference_kernel` across this matrix:
   - `d ∈ {1, 2, 3, 4, 5, 8, 16, 29, 30, 32}`, with `d = 32` exercising the largest register path;
   - `k ∈ {1, 5, 16, 17}`;
   - `n ∈ {257, 1000, 4109}` plus `n ≡ 1` and `n ≡ 31 (mod 32)`;
   - duplicate rows, exact ties, and leaves longer than `T` (multi-chunk), including the unaligned final leaf at `n = 4109` whose last chunk begins at 4108; assert bounded loads and consumption below `len`;
   - default and capped grid strides.
2. **Diagnostics.** Outputs with diagnostics disabled must equal the enabled outputs. Check the diagnostics invariants: popped counts are tile-uniform, `scanned ≤ n`, and the flat arm reports `scanned = n`.
3. **Pack.** Bit equality with gathered `pad_rows` output, zero tail, and checked-overflow sizing cases.
4. **Routing.** The tile plan is never selected on CPU, for `d > 32`, below the gate, or when shared memory does not fit. C16 CPU outputs are unchanged.
5. **Existing suites.** `cargo test -p fast-hdbscan-core`, then `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture`, then `cargo test -p fast-hdbscan`. Also `cargo clippy --workspace --all-targets` and `cargo fmt --all -- --check`.

Any mismatch blocks timing. A mismatch only in F means pruning is inexact; report it as a contract finding.

### Step 3: Selection batch S (separate from acceptance)

- **Design.** Three measured fits per process; use their median as one observation. For plane-count selection, run blocks `(B,T1,T2,T4,F1,F2,F4)`, `(T1,T2,T4,F1,F2,F4,B)`, `(T2,T4,F1,F2,F4,B,T1)`, followed by those three blocks in reverse block order and reversed within each block. This gives six process medians per arm and cell. Never count fits within one process as independent observations.
- **Planes per cube.** Evaluate at 50k×16 and 50k×30. Separately for T and F, choose the smallest `planes_per_cube` whose pooled core median is within 2% of that family's best arm in both cells. If none qualifies in both cells, stop and report inconclusive selection rather than inventing a tie-break.
- **Dimension gates.** Evaluate selected T and F separately at 20k×{2, 8, 16}, using blocks `(B,T,F)`, `(T,F,B)`, `(F,B,T)`, then their reverse block order with each block reversed. For each arm, its gate is the smallest `d` among {2, 8, 16} for which its pooled core median is less than 3% slower than B at that dimension and every larger tested dimension. An arm failing at 16D is ineligible; if both fail, stop. The assumption that this gate generalizes between tested dimensions and above 16D is explicit, not measured evidence. Each arm retains its own gate through acceptance and final routing.

Selection results are reported separately and are never pooled into acceptance.

### Step 4: Acceptance batches A and B (H2), plus H3

- **Arms:** B, the selected T, the selected F.
- **Primary cells (CUDA core):** 50k×16, 50k×30, 100k×16, 100k×30.
- **Guard cells:** 20k×16, 20k×30, plus 20k×2 and 20k×8 for each arm when that dimension is at or above its own gate. Below-gate cells exercise baseline routing and do not count as gains.
- **Batches.** Batch A has blocks `(B,T,F)`, `(T,F,B)`, `(F,B,T)`, `(B,F,T)`. Batch B reverses A's block order and reverses each block: `(T,F,B)`, `(T,B,F)`, `(B,F,T)`, `(F,T,B)`. Each block launches one process per arm, with three measured fits reduced to one process median. Thus each arm has exactly eight process medians per cell. Pair each variant with B within the same block for eight ratios. Pooled delta is `100 · (median(variant's eight process medians) / median(B's eight process medians) - 1)`. Validate counts and block identities in the runner before applying thresholds. Labels must be equal within each comparison cell. If only one variant remains eligible after S, retain these blocks as a fixed measurement design but do not accept the ineligible variant.
- **Thresholds (Phase 6 rules):** a gain is a pooled `Δ ≤ −2%` with at least 7 of 8 paired ratios below 1; a regression is a pooled `Δ ≥ +3%` with at least 7 of 8 above 1; anything between is "no effect".
- **H3 counters (explanatory, not vetoes):** at 20k×16, 50k×16 and 100k×16, the Step 0 metrics plus shared-memory bytes per block and branch efficiency.
- **Reporting only:**
  - repeat the CPU C16 versus selected-CUDA comparison from Step 0.3 at 100k only;
  - first-fit cold-start medians for B and the selected arm at 20k×16 and 20k×30, noting that the on-disk cache is not cleared.

### Step 5: Decision rule

- **Eligibility.** An arm is eligible with gains in at least 3 of 4 primary cells, no primary regression, and no guard regression. The 20k cells may show "no effect".
- **Selection.** Select the eligible arm with the lower geometric mean over primary cells. If both are eligible and F is not at least 2% faster in every primary cell, prefer T, because its cost does not grow as n² when pruning improves.
- **Rejection.** If neither arm is eligible, reject the phase and keep the per-query kernel. Record the negative result.
- **Judgement calls.** The master plan's judgement policy applies. Document any departure, with supporting and contrary evidence, in the phase summary and in `hdbscan_perf_report.md`.

### Step 6: Finalization

- **On accept:**
  - remove the unselected arm and the test-only arm overrides;
  - fix the selected arm's `planes_per_cube` and its own gate as constants;
  - keep `kd_neighbors_kernel` as the non-plane, `d > 32` and below-gate CUDA path, and keep the frozen reference unchanged.
- **Final gates.** Re-run Step 2 on the final source, then the integration runs `bash analyze_hdbscan_stages.sh --rows 20000` and `bash analyze_hdbscan_stages.sh --rows 100000`. Both require real CPU and CUDA execution, exit 0 and exact label equality. Update the route-aware kernel families in `analyze/` beforehand so CUDA kernel timing finds the dispatched kernel; keep `uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py` passing.
- **On reject:** remove all experimental production code. Keep diffs, binaries' hashes and evidence under `artifacts/cubecl_kernels_perf/phase7/`.

### Step 7: Deliverables

- **Code:** pack kernel, tile kernel, plan/routing, sizing and allocation checks, tests, and `analyze/` route-family and rows updates.
- **Evidence:** provenance, inputs, Step 0 baseline crossover/hypothesis evidence through 300k, scan/counter results, API probes, per-arm diffs and hashes, S/A/B records, H3, cold start, gate logs and `summary.json` with per-cell results and the decision. Distinguish baseline-only 200k/300k evidence from experimental-arm results capped at 100k; new-arm performance above 100k is unmeasured.
- **Docs:**
  - `hdbscan_perf_report.md`, "Phase 7 results", scoped to the measured cells on this host, including negative results;
  - Phase 7 status in `cubecl_kernels_perf_plan.md`;
  - `AGENTS.md` and `README.md`, only where they describe KD traversal, device buffers or resource sizing (check with a search).

## Risks and mitigations

| Risk                                                                   | Mitigation                                                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Union traversal visits many more leaves at low d                       | Selection-batch dimension gate; per-query kernel below it                                               |
| Shared-memory distance contracts differently from the global-load path | Same `chunk_distance` expression; H1 bit-equality on CUDA at every `d`                                  |
| Barrier under divergent control flow                                   | `sync_plane` only in plane-uniform loops; generated-source check (Step 0.6); inactive lanes participate |
| Register growth and lower occupancy                                    | H3 registers/local memory; `planes_per_cube` selection; stacks stay per-lane local first                |
| Pruning not exact, so visit order would matter                         | Flat arm F checks against the frozen reference; mismatch stops the phase                                |
| CPU tests never exercise plane logic                                   | All plane gates require CUDA; routing tests assert the plan is never selected on CPU                    |
| Gains limited to these Gaussian 16D/30D workloads on one GPU           | Report scope explicitly; no general platform claims                                                     |
| Profiling tools miss the new kernel                                    | Route-aware families updated and tested before final integration runs                                   |
