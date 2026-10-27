# Phase 7 detailed plan: CUDA plane-cooperative leaf-tile kernel

This document details Phase 7 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md) (F5 in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups"). It applies the accepted Phase 6 CPU ideas, leaf-ordered packed candidates and leaf-ordered query blocks, to the CUDA kernel. On the CPU, vector lanes are candidates. On CUDA, plane lanes are queries and each staged candidate is broadcast to the whole plane. It builds on accepted Phases 1, 3, 5 (CUDA neighbor width 64 by override) and 6 (CPU C16). Phases 2 and 4 were rejected. Source references are symbols in current `crates/fast-hdbscan/src/kernels/kdtree.rs`; inspect actual source before implementation.

**Status:** Step 0 completed (baseline evidence only; see [phase7_step0_report.md](phase7_step0_report.md)). Steps 1–7 are planned, not implemented or measured, and need a separate user execution request. This revision incorporates the Step 0 recommendations and the user's 2026-10-27 decisions:
- experimental timing at 50k×16 and 300k×16 only, with no guard cells;
- gains at 300k take priority over regressions at 50k (Step 5);
- the accepted arm replaces the per-query kernel outright, with no row-count routing threshold.

## Motivation and evidence

- **Crossover.** After Phase 6, the user observed CPU C16 overtaking CUDA in `core_distances` at roughly 300k rows. Phase 6 pooled core medians are 0.038 s (20k×16) and 0.246 s (50k×16) on CPU. Step 0 confirmed it with baseline B: CUDA core loses between 200k and 300k at 16D (300k: CPU 4.20 s vs CUDA 4.69 s) and between 100k and 200k at 30D (300k: 16.30 s vs 25.86 s). Full fit still favors CUDA except at 300k×30 (33.72 s vs 35.40 s), because CUDA MST is faster. CPU per-pair cost is flat with n; CUDA per-pair cost roughly doubles above 100k (16D) and 50k (30D).
- **Weak pruning.** At 20k, a query scans 74.1% of all pairs at 16D and 96.5% at 30D (report, "Features" table). Step 0 shows pruning improves with n at 16D (scan/n² 0.74 → 0.33 from 20k to 300k) but only slowly at 30D (0.97 → 0.78). The search stays close to O(n²) with traversal overhead, most strongly at 30D. Its cost depends mainly on how cheaply each `(query, candidate)` pair gets its operands.
- **Current CUDA kernel.** `kd_neighbors_kernel` runs one query per thread, with incoherent per-thread traversal, and gathers candidates through `indices`. The baseline counters were 70–79% long-scoreboard stalls.
- **Cache hypothesis (supported by Step 0, correlational).** The RTX 3080 L2 is 5 MB; the 9800X3D has 96 MB of L3. On B's `kd_neighbors_kernel` the L2 hit rate falls 99.8% → 98.2% → 75.7% → 38.2% → 29.2% (16D, 20k–300k) and 99.6% → 82.2% → 49.3% (30D, 20k–100k). DRAM bytes per scanned pair rise from 0.09 to 65 at 16D (about one padded row per pair at 300k), while requested L2 sectors per pair stay constant. Loads are uncoalesced (L1 sectors/request ≈ active lanes). At 50k×16 the L2 still hits 98% but L1 throughput is 82% of peak, so that cell is L1/LSU-bound, not DRAM-bound. Nothing below depends on causation, and gains may be attributed to DRAM traffic only together with post-change counters.
- **Ordering alone is weak on CUDA.** Spatial reordering gave CUDA core −8.9% at 20k and −2.3% at 50k, with overlapping ranges (report, "Spatial ordering"). Neighboring lanes touch the same leaves but diverge on pruning, so their loads still issue separately. A CUDA-only port of Phase 6's leaf-block order is therefore not an arm here.
- **Mechanism.** A plane stages each visited candidate chunk once with coalesced loads, and all `PLANE_DIM` lanes read it from shared memory as a broadcast. Global candidate traffic per evaluated pair falls by about 32 × lane utilization: Step 0's tile simulation gives about 12–24× at 16D and 25–31× at 30D. DRAM traffic falls from about one row per pair to one row per staged candidate per tile. Compute is roughly unchanged: simulated tile lane utilization (37–74% at 16D) is close to the baseline's measured SIMT efficiency (27–39%), so the bet is entirely on memory. This is the CUDA counterpart of C16, which uses each SIMD candidate load for 16 pair computations.

## Invariants

All master-plan invariants apply unchanged. These are the ones this phase depends on most:

1. **Bit-identical IDs and distances.** Each pair distance is the f32 sum over features `0..d` in order, computed by the existing `chunk_distance` on `KdVector` chunks. Staged candidates are bit copies of padded rows, so the operands and the expression match the current kernel. Contraction is still tested, not assumed (Invariant 5).
2. **Visit order is free.** The tile traversal changes the order of nodes, leaves and candidates. The result is still the k smallest `(distance², original ID)` keys with pruning `lower <= worst`, evaluated per lane.
3. **Original IDs.** Self-exclusion, tie keys and output rows use `indices[position]`. Leaf order is only a memory layout and a query schedule. Public inputs and downstream arrays are never permuted.
4. **Valid-tree contract** (Phase 6). Finite data and builder-produced trees, or equivalent low-level inputs, are required. `indices` is a permutation, so each original row has exactly one writer. No runtime content scan, ownership map or tree validation is added. Existing shape/dtype validation (`validate_tree`) and checked sizing stay.
5. **Error contract.** New buffers use `checked_*` sizing and are counted in `kd_neighbor_pack_bytes` and `core_search_memory`. Device allocation checks cover the new pack.

## Changes from the master plan's Phase 7

1. **Independent planes with `sync_plane()`, not a cube-uniform stack with `sync_cube()`.** Each plane has its own shared slice and its own uniform traversal. Planes in a cube never synchronize with each other. This gets past the 16-blocks-per-SM limit without requiring cube-wide traversal agreement, and avoids a barrier under masked control flow. CubeCL 0.11.1 provides `sync_plane` (`cubecl-core/src/frontend/synchronization.rs:35`). This build compiles CUDA through `cubecl-llvm` NVPTX, not the cubecl-cpp dialect: `sync_plane` lowers to `llvm.nvvm.bar.warp.sync(i32 -1)` and `sync_cube` to `llvm.nvvm.barrier.cta.sync.aligned.all` (`cubecl-llvm-0.11.1/src/nvptx/synchronization.rs`), confirmed by the Step 0.6 probes. `plane_min` lowers to `activemask` plus a `shfl.sync.bfly` butterfly, and `plane_any` to `activemask` plus `vote.any.sync`. Their results cover only the lanes active at the call site, so every plane operation must be called at a converged, plane-uniform point, never inside `if own` or `if active`. The CUDA runtime enables `Plane::Ops` and enables `Plane::Sync` from arch 70 (`cubecl-cuda/src/runtime.rs:233–243, 363`). The RTX 3080 is sm_86.
2. **Phase 1 output-backed heap, not register top-k.** Phase 2 was rejected. The tile kernel copies the Phase 1 admission, sift-down and final heapsort from `kd_neighbors_kernel` verbatim, including the cached `worst_distance`/`worst_index` registers.
3. **Leaf-ordered padded rows (AoS), not SoA.** My earlier recommendation was the Phase 6 SoA pack at tile width 32. AoS works better on CUDA:
   - a chunk of `PLANE_DIM` consecutive positions is one contiguous range of `PLANE_DIM · d_chunks` `KdVector`s, so a flat lane-strided copy into shared memory is fully coalesced;
   - shared reads are broadcasts in either layout;
   - the compute loop keeps the exact Phase 3 `chunk_distance` expression on `KdVector` operands, which keeps the arithmetic closest to the current kernel.

   The CPU C16 SoA pack is not reused.

4. **Flat control arm.** A comptime `traverse = false` configuration of the same kernel scans all positions in staged chunks, with no tree. It shows whether the tree still pays for itself on GPU at 16–30D. It is also the strictest correctness check: it does no pruning, so a mismatch against the reference would expose inexact pruning in the reference or baseline. Stop and report such a mismatch; do not relax any test.
5. **Dimension gate selected in advance.** Union traversal visits a leaf when any lane needs it. Step 0's tile simulation at 20k gives staged work of 1.6× (2D) and 1.7× (8D) the baseline warp maximum, with 25–32% lane utilization, so the predicted gate is 16 for both arms. A selection batch, kept separate from acceptance batches, sets the smallest dimension that uses the tile kernel for T. F's gate is fixed at 16 without measurement, because at 2D/8D it stages every position while only 0.6–17% are useful. Lower dimensions keep the per-query kernel.
6. **Experimental cells: 50k and 300k only (user decision, 2026-10-27).** 300k is where the motivating crossover lies and where Step 0 measured the strongest cache degradation, so experimental arms are measured there to show how they scale. 50k is the smaller reference size. Acceptance timing is 16D only, and there are no guard cells. To keep run time down, experimental arms are not timed at 20k, 100k, 200k or 30D. 30D and other dimensions at or above the gate are covered by the Step 2 correctness gates but have no performance measurement. Step 0 data at the other sizes remains baseline-only evidence. Speedups may be size-dependent through staging overhead and cache residency. Acceptance and rejection apply to the measured cells only; performance between or above them is unmeasured. Do not add further sizes to rescue a failed phase.
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
| `analyze/benchmark_hdbscan_stages.py`, `analyze/profile_hdbscan_counters.py`                                     | Done in Step 0: `--rows` accepts 200k and 300k (`EXTENDED_ROWS`) with tests. The kernel, summary and shell tools still stop at 100k. They need no change unless a final integration run at 300k is added                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |

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
- **Leaf sizes.** `LEAF_SIZE = 40` is a minimum. With the builder's level formula, production leaves hold 40–79 points (Step 0: 78–79 at 20k, 48–49 at 50k–200k, 73–74 at 300k). Every leaf therefore takes 2–3 chunks at T = 32, with about 76–82% slot utilization, and multi-chunk staging is the main path. T-chunk staging is the default. A whole-leaf variant (80 slots per plane: about 10.6 KB at `d_chunks = 8`, 5.4 KB at 4) is an optional later step, only if H3 shows that staging or `sync_plane` overhead matters, and only after weighing its occupancy cost.
- **Per cube:** `planes_per_cube` times that. Shared lengths are comptime from `d_chunks` (derived from comptime `reg_features`), `T` and `planes_per_cube`.
- **Host check:** the requirement must fit `max_shared_memory_size`. Step 0 confirms the exact 0.11.1 shared-memory constructor and indexing API with a probe before kernel work.

## Steps

### Step 0: Provenance, baseline crossover and hypothesis evidence (no production change)

1. **Provenance.** Record `git status`, HEAD and the diff, and never revert user work. Archive the current release/tracing `profile_hdbscan_stages` binary as baseline **B** with its SHA-256. Use the evidence directory `artifacts/cubecl_kernels_perf/phase7/`.
2. **Inputs.** Generate or reuse 20k, 50k, 100k, 200k and 300k inputs at 16D and 30D with the same generator, parameters and seed as existing cells (`make_input(rows, features, 1.0, 1.0, 42, …)`). Record hashes in `inputs.json`. Extend the benchmark and counter tools' `--rows` choices to include 200k/300k and test argument acceptance. Experimental arms use the 50k×16 and 300k×16 inputs, plus 300k×2 and 300k×8 generated in Step 3 for T's gate. All other inputs are Step 0 evidence only.
3. **CPU/CUDA crossover.** Run B on CPU and CUDA at 100k, 200k and 300k × 16D/30D, three processes each, with the CPU and CUDA processes alternating. Report core and full-fit medians, CPU thread count, observed crossover (or its absence) and labels equal. This is initial evidence only, not an acceptance input.
4. **Scan fraction.** Run B with `FAST_HDBSCAN_KD_DIAGNOSTICS=1` at 50k, 100k, 200k and 300k × 16D/30D, and report mean nodes popped and leaf positions scanned per n². This tests whether baseline pruning improves with n.
5. **L2 hypothesis.** Run H3 on B's `kd_neighbors_kernel` at 50k, 100k, 200k and 300k × 16D. Add the metrics `lts__t_sector_hit_rate.pct`, `dram__bytes_read.sum`, `l1tex__t_sector_hit_rate.pct`, `l1tex` sectors per request, long-scoreboard stall share, achieved occupancy, and registers/local memory. Falling L2 hit rate and rising DRAM bytes per scanned pair support the hypothesis within this range only. Record the outcome either way. A refutation does not cancel the phase, because the tile kernel also removes divergent gathers, but the report must not attribute gains to L2.
6. **API probes.** These are throwaway test kernels, not production code. Run each on CUDA:
   - per-plane shared slices indexed by plane;
   - `sync_plane` inside plane-uniform loops, with the generated LLVM IR (`CUBECL_DEBUG_LOG=<file>`) checked for `llvm.nvvm.bar.warp.sync` and no `barrier.cta` (originally written as `__syncwarp()`/`__syncthreads()`, which assumed the cubecl-cpp backend);
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
   - duplicate rows, exact ties, and leaves longer than `T` (multi-chunk), including F's unaligned final global chunk at `n = 4109`, which begins at 4108, and a T case whose final leaf ends mid-chunk (T chunks are leaf-relative); assert bounded loads and consumption below `len`;
   - default and capped grid strides.
2. **Diagnostics.** Outputs with diagnostics disabled must equal the enabled outputs. Check the diagnostics invariants: popped counts are tile-uniform, `scanned ≤ n`, and the flat arm reports `scanned = n`.
3. **Pack.** Bit equality with gathered `pad_rows` output, zero tail, and checked-overflow sizing cases.
4. **Generated code.** With `CUBECL_DEBUG_LOG=<file>`, the tile kernel's LLVM IR contains `llvm.nvvm.bar.warp.sync` and no `barrier.cta`, as in the Step 0.6 probes.
5. **Routing.** The tile plan is never selected on CPU, for `d > 32`, below the gate, or when shared memory does not fit. C16 CPU outputs are unchanged.
6. **Existing suites.** `cargo test -p fast-hdbscan-core`, then `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture`, then `cargo test -p fast-hdbscan`. Also `cargo clippy --workspace --all-targets` and `cargo fmt --all -- --check`.

Any mismatch blocks timing. A mismatch only in F means pruning is inexact; report it as a contract finding.

### Step 3: Selection batch S (separate from acceptance)

- **Design.** Three measured fits per process; use their median as one observation. For plane-count selection, run blocks `(B,T1,T2,T4,F1,F2,F4)`, `(T1,T2,T4,F1,F2,F4,B)`, `(T2,T4,F1,F2,F4,B,T1)`, followed by those three blocks in reverse block order and reversed within each block. This gives six process medians per arm and cell. Never count fits within one process as independent observations.
- **Planes per cube.** Evaluate at 50k×16 and 300k×16. Separately for T and F, choose the smallest `planes_per_cube` whose pooled core median at 300k×16 is within 2% of that family's best arm. Larger sizes take priority, so 50k×16 breaks ties only among candidates that qualify at 300k. Choosing the smallest qualifying count always yields a selection.
- **Dimension gates.** F's gate is fixed at 16 (Changes §5). This is the only experimental timing outside 16D, and it exists so the replacement does not silently regress low-dimensional data. Evaluate selected T against B at 300k×{2, 8, 16}, using blocks `(B,T)`, `(T,B)`, `(B,T)`, then their reverse block order with each block reversed. Generate the 300k×2 and 300k×8 inputs with the Step 0.2 generator and record their hashes. T's gate is the smallest `d` among {2, 8, 16} for which its pooled core median at 300k is less than 3% slower than B at that dimension and every larger tested dimension. Record Step 0's prediction (gate = 16) before running. T failing at 300k×16 makes T ineligible. The assumption that the gate generalizes between tested dimensions and above 16D is explicit, not measured evidence. Each arm retains its own gate through acceptance and final routing.

Selection results are reported separately and are never pooled into acceptance.

### Step 4: Acceptance batches A and B (H2), plus H3

- **Arms:** B, the selected T, the selected F.
- **Primary cells (CUDA core):** 50k×16 and 300k×16. 300k×16 is the **priority cell**.
- **Guard cells:** none (user decision).
- **Cost.** A B process at 300k×16 takes about 15 s wall time (Step 0), so each batch takes roughly 12 B process-minutes at 300k plus the arms. Run both cells with the same block design; do not shorten them.
- **Batches.** Batch A has blocks `(B,T,F)`, `(T,F,B)`, `(F,B,T)`, `(B,F,T)`. Batch B reverses A's block order and reverses each block: `(T,F,B)`, `(T,B,F)`, `(B,F,T)`, `(F,T,B)`. Each block launches one process per arm, with three measured fits reduced to one process median. Thus each arm has exactly eight process medians per cell. Pair each variant with B within the same block for eight ratios. Pooled delta is `100 · (median(variant's eight process medians) / median(B's eight process medians) - 1)`. Validate counts and block identities in the runner before applying thresholds. Labels must be equal within each comparison cell. If only one variant remains eligible after S, retain these blocks as a fixed measurement design but do not accept the ineligible variant.
- **Thresholds (Phase 6 rules):** a gain is a pooled `Δ ≤ −2%` with at least 7 of 8 paired ratios below 1; a regression is a pooled `Δ ≥ +3%` with at least 7 of 8 above 1; anything between is "no effect".
- **H3 counters (explanatory, not vetoes):** at 50k×16 and 300k×16, for B and each tile arm.
  - Collect the Step 0.5 metric set (`run_l2_counters.py`): L2/L1 hit rates, `dram__bytes_read.sum`, `lts__t_sectors_srcunit_tex_op_read.sum`, sectors/request, threads per instruction, long-scoreboard share, occupancy, registers and local memory. Add shared-memory bytes per block and branch efficiency.
  - Normalize B by leaf positions scanned and the tile arms by staged positions × 32.
  - At 50k×16, attribute gains to L1/LSU traffic, not L2 (the baseline L2 hit rate is 98%).
- **Reporting only:**
  - repeat the CPU C16 versus selected-CUDA comparison from Step 0.3 at 300k×16 to show whether the 16D crossover moves;
  - first-fit cold-start medians for B and the selected arm at 50k×16, noting that the on-disk cache is not cleared.

### Step 5: Decision rule

**Size priority (user decision, 2026-10-27).** Gains at larger sizes take priority over regressions at smaller sizes. Correctness gates (Step 2) are never subject to this priority.

- **Large-size override.** An arm is eligible if 300k×16 shows a **gain of at least about 10%**: pooled `Δ ≤ −10%` with at least 7 of 8 paired ratios below 1. A 50k×16 regression is then ignored for eligibility, but reported with its magnitude. Treat an observed Δ just above −10% (for example −9.5%) as a judgement call under the master plan's policy rather than an automatic pass or fail, and document it.
- **Standard eligibility (when the override does not apply).** 300k×16 is a gain (pooled `Δ ≤ −2%`, at least 7 of 8 paired ratios below 1) and 50k×16 is not a regression.
- **Severe smaller-size regression.** If the selected arm's 50k×16 core regression is severe enough to question outright replacement, stop before finalization and report it to the user, who decides. This is a judgement call with no preset threshold. It does not reintroduce a row-count gate.
- **Selection.** Among eligible arms, select the one with the lower pooled median at 300k×16. If those are within 2%, use 50k×16. If that also ties within 2%, prefer F, which is simpler, reads no tree and has no traversal divergence. Step 0's simulation shows that T stages 88–100% of what F stages at 16/30D through 300k, so T's former tie-break ("its cost does not grow as n²") is not supported by evidence.
- **Outright replacement, no row-count gate (user decision).** The accepted arm replaces `kd_neighbors_kernel` for every row count at or above its dimension gate. No n-based routing threshold is added. `kd_neighbors_kernel` remains only for the non-plane, `d > 32` and below-dimension-gate paths.
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
- **Evidence:** provenance, inputs, Step 0 baseline crossover/hypothesis evidence through 300k, scan/counter results, API probes, per-arm diffs and hashes, S/A/B records, H3, cold start, gate logs and `summary.json` with per-cell results and the decision. Distinguish baseline-only Step 0 evidence (20k–300k, 16D/30D) from experimental-arm timing at 50k×16 and 300k×16 plus T's gate cells at 300k×{2, 8, 16}. New-arm performance at other sizes and at 30D is unmeasured. State explicitly which eligibility path (large-size override or standard) was used, and report any ignored 50k×16 regression.
- **Docs:**
  - `hdbscan_perf_report.md`, "Phase 7 results", scoped to the measured cells on this host, including negative results;
  - Phase 7 status in `cubecl_kernels_perf_plan.md`;
  - `AGENTS.md` and `README.md`, only where they describe KD traversal, device buffers or resource sizing (check with a search).

## Risks and mitigations

| Risk                                                                   | Mitigation                                                                                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Union traversal visits many more leaves at low d                       | Selection-batch dimension gate; per-query kernel below it                                               |
| Tile union stages nearly all leaves at 16/30D, so T has no work advantage over F | Both arms measured; selection by 300k×16 median, ties prefer F                              |
| Accepted arm regresses at small n                                      | Outright replacement by user decision; 50k×16 regressions reported; a severe one stops for a user decision; no n-based gate |
| Unmeasured cells (20k, 100k, 200k, 30D) regress after replacement     | Correctness at all `d` via Step 2; performance there stated as unmeasured; no guard cells by user decision |
| Shared-memory distance contracts differently from the global-load path | Same `chunk_distance` expression; H1 bit-equality on CUDA at every `d`                                  |
| Barrier under divergent control flow                                   | `sync_plane` and plane ops only at converged, plane-uniform points; Step 2 IR check (`bar.warp.sync` present, no `barrier.cta`); inactive lanes participate |
| Register growth and lower occupancy                                    | H3 registers/local memory; `planes_per_cube` selection; stacks stay per-lane local first                |
| Pruning not exact, so visit order would matter                         | Flat arm F checks against the frozen reference; mismatch stops the phase                                |
| CPU tests never exercise plane logic                                   | All plane gates require CUDA; routing tests assert the plan is never selected on CPU                    |
| Gains limited to these Gaussian 16D/30D workloads on one GPU           | Report scope explicitly; no general platform claims                                                     |
| Profiling tools miss the new kernel                                    | Route-aware families updated and tested before final integration runs                                   |
