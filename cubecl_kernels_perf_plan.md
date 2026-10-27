# Plan: CubeCL kernel optimizations for HDBSCAN `core_distances`

## Goal

Reduce the time spent in the KD-tree exact neighbor search behind HDBSCAN `core_distances` on CPU (CubeCL LLVM) and CUDA. Neighbor IDs and squared distances must stay bit-identical to the current kernel. This is the only part of the stage that matters: it accounts for more than 99% of the CPU core stage and about 99% of the CUDA one.

Source of the findings: [hdbscan_perf_report.md](hdbscan_perf_report.md), in particular the "Follow-ups: CubeCL kernel patterns" section. Phase 1 has been implemented and measured as recorded below. Phase 2 was measured and rejected. Phase 3 was implemented, measured and accepted; Phase 4 was implemented, measured and rejected. Phase 5 was measured and accepted at width 64 by explicit user override of the default regression screen. Phase 6's historical malformed-tree attempt was rejected; the separately authorized valid-tree retry is implemented and accepted as C16. Independent review receipts are maintained separately in the ultragoal ledger. Phase 7 onward remain unimplemented and unmeasured.

## Baseline facts this plan relies on

- **Route:** complete-graph search resolves to KD-tree for at least 256 finite rows and at most 30 features (`crates/fast-hdbscan/src/estimator.rs:124–133`). Above 30 features the kNN/ANN path applies instead, and this plan does not cover it.
- **Kernel:** `kd_neighbors_kernel` and `box_lower_bound` in `crates/fast-hdbscan/src/kernels/kdtree.rs`. CPU uses one legalized cube with `num_cpu_cores` units and cyclic query assignment; the CUDA neighbor launch prefers 64 threads after the Phase 5 override, one query per unit. Unrelated KD launches and the frozen reference retain the baseline 32-thread CUDA preference. Actual dimensions and grid counts remain device-legalized and bounded.
- **Host:** `exact_neighbors_kd` (`kdtree.rs:586–636`) is called from `core_distances` (`estimator.rs:384–547`). The tree is uploaded by `upload_kd_tree` (`estimator.rs:344`) into `DeviceKdTree` (`crates/fast-hdbscan/src/backend.rs:11–21`), which is a fusion `ExtensionType` that `boruvka_candidates_kd` also consumes.
- **Tree:** balanced implicit layout (`crates/fast-hdbscan-core/src/kdtree.rs:52–128`), with children of node `i` at `2i+1` and `2i+2`. `LEAF_SIZE = 40` determines the level count; a leaf holds roughly 40–80 points. Every leaf is a contiguous range of `tree.indices`.
- **Measured baseline (16D, k5, RTX 3080 / Ryzen 9800X3D):**
  - Core time: 20k is 0.272 s CPU / 0.074 s CUDA; 100k is 4.66 s CPU / 1.18 s CUDA.
  - Each query scans 74% of all pairs.
  - CUDA counters: 71 registers per thread, 33% theoretical and 18.5% achieved occupancy (the 16-blocks/SM limit binds), 70–79% long-scoreboard stalls.
- **Pinned versions:** Burn 0.22.1, CubeCL 0.11.1 (from `Cargo.lock`). `Vector<T, Const<N>>`, `Shared::new_slice`, and `plane_any`/`plane_min`/`plane_ballot`/`plane_broadcast` exist in 0.11.1. For the in-repo pattern of fixed-width vector tensors with `#[unroll]` lane extraction, see `crates/fast-hdbscan/src/kernels/component_reduce.rs`.

## Invariants (apply to every phase)

1. **Bit-identical output.** Neighbor IDs and squared distances must match the current kernel bit for bit. Each `(query, candidate)` distance is the f32 sum over `feature = 0..d` taken in that sequential order. Never reorder this sum: no horizontal vector reductions over features, no `plane_sum` over features, no `|q|²+|r|²−2q·r`.
2. **Visit order is free.** The result is the k smallest `(distance², original ID)` keys under a total order, with pruning `lower <= worst`. Node, leaf, candidate and query visit order may therefore change freely; this is what allows reordering, tiling and vectorizing across candidates or queries.
3. **Zero padding is neutral.** Padding features to a multiple of 4 with `0.0` in both points, and `min = max = 0` in bounds, adds `+0.0` terms and leaves every sum bit-identical. Because `d <= 30` on this route, padded widths are limited to `d_chunks ∈ 1..=8` vectors of 4.
4. **Original IDs everywhere.** Self-exclusion, tie keys and output rows use original row IDs. Packed or leaf-ordered positions are only a memory layout. Never permute public input or downstream arrays (the whole-input permutation probe changed final partitions; see the report).
5. **FMA contraction.** Vector and scalar code may contract differently. Bit-equality must be tested against the current kernel, never assumed.
6. **Error contract.** Preserve the existing assertion and validation behavior (`validate_tree`, overflow checks). New buffers get `checked_*` sizing and must be counted in the existing allocation and resource sizing (`CoreSearchMemory`, device allocation checks).

## Reference harness (built before any optimization)

- **H1. Golden-output test.** In `crates/fast-hdbscan/tests/test_kdtree_search.rs`, add a test that runs both the current kernel and the variant under test and asserts bit-equality of IDs and of `f32::to_bits` distances, for:
  - feature counts `d ∈ {1, 2, 3, 4, 5, 16, 29, 30}`;
  - `k ∈ {1, 5, 16, 17}`, where 17 exercises the large-k fallback path;
  - inputs with duplicate rows, exact distance ties, `n` just above 256, and sizes that are not multiples of the vector width or `PLANE_DIM`;
  - both CPU and CUDA, with CUDA mandatory under `FAST_UMAP_REQUIRE_CUDA=1`.

  Keep the current kernel callable from tests (a `#[cfg(test)]` reference entry point) until the final phase removes it.
- **H2. Interleaved benchmark protocol.** Use the existing `analyze/benchmark_hdbscan_stages.py` with `profile_hdbscan_stages`, at 20k and 50k rows, 16D and 30D, k5, alternating baseline and variant processes in `O,V,V,O,O,V` order (the method in the report). Record core-stage and full-fit medians and ranges, and compare only within a batch.
- **H3. CUDA counter capture.** Use `analyze/profile_hdbscan_counters.py` and add these metrics: registers per thread, local-memory load/store (spills and stacks), `l1tex` sectors per request, long-scoreboard stall share, and achieved/theoretical occupancy.

## Phases

The phases run in order. Each phase must pass H1 on CPU and CUDA, then be measured with H2 (plus H3 on CUDA), before the next phase starts. Performance acceptance and rejection rules are decision guidelines, not absolute vetoes, especially for potentially noisy timings. Normally, a phase without a demonstrated benefit is reverted and recorded as a negative result rather than stacked under later phases. Where the evidence supports a reasonable trade-off, make a judgement call: weigh timing variability, absolute and relative effects, backend/workload scope, counters, and implementation/resource costs. Document any departure from the stated rules, the supporting and contrary evidence, the cost accepted and why the trade-off is worthwhile in the phase summary and `hdbscan_perf_report.md`. Do not describe an inconclusive timing as a proven gain or selectively discard unfavorable samples. Correctness, error-contract invariants and required CPU/CUDA verification remain mandatory; this discretion does not relax them or retroactively change recorded phase decisions.

### Phase 1: Register-cached heap root (F2a), both backends

**Status:** implemented and accepted; CPU/CUDA bitwise and diagnostics gates pass. Eight-cell interleaved timing rules pass (CUDA gain); cache retained. Complete H3 counters captured, including sampled-stall share and theoretical occupancy. Evidence: `artifacts/cubecl_kernels_perf/phase1/summary.json` and report "Phase 1 results".

**Change.** In `kd_neighbors_kernel`, keep `worst_distance` and `worst_index` in registers:
- Candidate admission (`kdtree.rs:281–285`) and child pushes (`:339–359`) compare against the registers.
- After an admission and sift, reload the root once to refresh them.
- Global `out_*` is touched only on admission.

**Why.** This removes 2–3 dependent global loads per scanned candidate and up to 3 per internal node. Semantics do not change.

**Accept when** H1 is bit-identical and CPU or CUDA core time improves in H2 without the other backend regressing beyond the batch's noise range.

### Phase 2: Register top-k for small k (F2b), both backends

**Status:** measured and rejected; no register threshold retained. The experiment used an exact `#[comptime] k` specialization for each `k = 1..=16`, with sorted register-list insertion, and the output-backed Phase 1 heap otherwise. This replaces the earlier proposed `k_max ∈ {8,16}` bucket design; see the detailed plan's rationale. CPU/CUDA equivalence passed, but mandatory rules 3, 4 and 6 failed. No threshold revision was attempted. Production remains the accepted Phase 1 root-cache heap; extended equivalence/profiling harness retained. See [Phase 2 results](hdbscan_perf_report.md#phase-2-results-register-top-k-experiment-rejected) and [`summary.json`](artifacts/cubecl_kernels_perf/phase2/summary.json).

**Experimental design (not retained).** Fully unrolled sorted insertion of `(distance, id)` pairs into an exact-comptime-`k` list for each `k = 1..=16`; write results once at row completion. The unchanged output-backed heap handles `k > 16`. This design was tested, not accepted as a production optimization.

### Phase 3: Padded vector data and register-resident query (F1), both backends

**Status:** implemented and accepted. Detailed plan: [cubecl_kernels_perf_plan_phase3.md](cubecl_kernels_perf_plan_phase3.md). Bit-identical IDs, distances and diagnostics on CPU and CUDA; core stage −20% to −29% (CPU) and −62% to −70% (CUDA) in the primary cells. Two deviations from the literal decision rule are recorded: one 20k×2 CUDA campaign regression that did not replicate, and +0.16% local-load sectors at 2D (traversal stack, not spills). See [Phase 3 results](hdbscan_perf_report.md#phase-3-results-padded-vector-data-and-register-resident-query) and [`summary.json`](artifacts/cubecl_kernels_perf/phase3/summary.json).

**Change (as implemented).**
1. **Per-call pack, not `DeviceKdTree` fields.** `launch_kd_neighbors` packs `data`, `bounds_min` and `bounds_max` with `pad_rows` into contiguous 1D f32 buffers of `rows · 4⌈d/4⌉` values with `+0.0` tail lanes, and passes them as `Tensor<Vector<f32, Const<4>>>` views. `DeviceKdTree`, `HdbscanOps::exact_neighbors_kd` and `boruvka_candidates_kd` are unchanged.
2. **One layout for every `d`.** The kernel takes runtime `d_chunks` and an exact `#[comptime] reg_features`: for `d ≤ KD_REGISTER_QUERY_MAX_FEATURES = 32` the query is loaded once into registers and the register path sums exactly `d` real lanes (pads skipped); for `d > 32` (`reg_features = 0`) a runtime chunk loop reads the query from global memory and also adds the neutral pad lanes. The exact feature count replaced an initial `reg_chunks` specialization after the 20k×2 guard cell (Step 5 rule).
3. Per-lane arithmetic and feature order are unchanged; vectors are load containers only (`extract(lane)` with comptime lanes).
4. Sizing: `(n + 2·nodes) · 4⌈d/4⌉ · 4` bytes are added to the KD branch of `core_search_memory`, with three 1D `check_device_allocation` calls.

**Final register threshold.** 32 features. Registers per thread at 20k: 34 (2D), 106 (16D), 128 (30D), against 75 for the baseline; 30D sits exactly at the 128-register budget of the 32-thread geometry.

### Phase 4: Compact node records (F3), both backends

**Status:** implemented, measured and rejected. Detailed plan: [cubecl_kernels_perf_plan_phase4.md](cubecl_kernels_perf_plan_phase4.md). [Phase 4 results](hdbscan_perf_report.md#phase-4-results-compact-node-layout-rejected) and `artifacts/cubecl_kernels_perf/phase4/summary.json` preserve the fixed two-batch evidence. Production remains Phase 3: no primary cell meets the gain screen, and 20k×16 CPU regresses 3.40% with all six paired ratios above 1.

**Change.**
- Replace the separate `starts`/`ends`/`is_leaf` reads in the KD neighbor path with one `Vector<u32, Const<4>>` record per node: `(start, end, is_leaf, 0)`.
- The rejected experiment interleaved padded bounds per node as `[min chunks | max chunks]`, using `pack_node_bounds_kernel` at the per-call pack site, not fields in `DeviceKdTree`. Phase 3 production still uses separate `pad_rows` bounds.
- Keep the leaf flag as data, including non-0/1 flags and arbitrary early leaves in public `DeviceKdTree` inputs.
- `first_leaf` was not pursued: the flag occupies an otherwise unused lane in the `u32` record, so removing it saves no record load and introduces a host proof obligation. Both the record and interleave experiment were removed after rejection.

**Accept when** H1 passes and H2 shows a gain, using the detailed plan's default performance criteria. This phase is expected to be minor: noisy or mixed timing evidence is not an automatic rejection when a reasonable, evidence-backed trade-off justifies acceptance. Apply the judgement and documentation policy above and the detailed plan's Step 5; otherwise revert and record the result.

### Phase 5: Launch-geometry sweep (CUDA)

**Status:** measured and accepted at **64 threads by explicit user override** on 2026-10-26; CPU policy is unchanged. The experiment uses the accepted Phase 3 baseline, not cumulative Phases 1–4. Detailed protocol: [cubecl_kernels_perf_plan_phase5.md](cubecl_kernels_perf_plan_phase5.md). [Phase 5 results](hdbscan_perf_report.md#phase-5-results-cuda-neighbor-launch-geometry-64-selected-by-override) and `artifacts/cubecl_kernels_perf/phase5/pooled_summary.json` preserve both complete timing matrices. The default screen rejected both candidates; the override prioritizes larger practical workloads: 64 reduces 50k CUDA core time by 19.35% at 16D and 9.95% at 30D, accepting the 20k×30 cost of +1.443 ms (+3.62% core, +1.37% full fit). Width 128 remains rejected. No extra selection samples or workload-specific dispatch were added.

**Implementation.** A single fixed neighbor-only CUDA preference of 64, legalized against actual device limits; other KD launches and the frozen reference remain at 32, and CPU policy remains unchanged. Observed registers per thread at widths 32/64/128: 16D 106/64/50; 30D 128/72/72. Earlier occupancy estimates were hypotheses, not universal resource ceilings. Frozen-reference and capped grid-stride tests remain; experiment binaries, diffs and the original rejection evidence are preserved, with the superseding policy in `artifacts/cubecl_kernels_perf/phase5/override_64.json`.

Choose the setting from H2/H3 at 20k and 50k. The 20k case has only ~625 warps of work, so larger cubes are not expected to help it. Keep `legal_cube_dim` and the `max_cube_count` limits.

**Accept when** the chosen geometry beats 32 threads in an interleaved batch at 50k and does not regress 20k beyond noise. Otherwise keep 32.

### Phase 6: CPU leaf-packed SoA kernel, vectorized across candidates (F4)

**Status:** rejected at the mandatory correctness gate, before acceptance timing; no selected arm or width cap. Detailed plan: [cubecl_kernels_perf_plan_phase6.md](cubecl_kernels_perf_plan_phase6.md); [results](hdbscan_perf_report.md#phase-6-results-correctness-rejection-before-acceptance-timing). The experiment used position-tiled SoA, block-cyclic query blocks of 64 and first-occurrence row ownership, with P/O/C isolation configurations and initial cap 16. Every host width (16/8/4/2/1) passed the prerequisite vector probe. New malformed-input cases exposed a baseline/reference contradiction, so all experimental production code was removed; the full diff and evidence remain in `artifacts/cubecl_kernels_perf/phase6/`. The Rust/Rayon measurement example is retained. No CPU speedup or regression is established, and Phase 5 CUDA geometry remains unchanged.

**Current status — accepted valid-tree retry: C16.** Execution separately authorized and implementation completed. Independent review receipts are maintained separately in the ultragoal ledger. Finite data and valid builder-produced trees (or equivalent low-level preconditions) are required; estimator upload/finite-row remapping is unchanged. No runtime content scan or ownership machinery exists. All 11 experimental and 11 final Step 3 gates passed, with required CUDA on designated gates; final 20k/100k scripts exited 0 and both benchmark summaries report labels equal. P and C have four primary gains/no regressions; O has no primary gains. G=C, and P fails +2% near-best at 20k×16, selecting C16 by the exact rule. Width-selection S is separate from pooled A/B acceptance. Verified host properties: 15 cores, load width512, widths [16,8,4,2,1]. Existing Rust/Rayon tool, CUDA neighbor64, unrelated32 and frozen reference are retained. See [retry results](hdbscan_perf_report.md#phase-6-valid-tree-retry-results-c16-accepted) for timings, memory, first-fit/cache caveat, host bit mismatches and observed vector/FMA IR; new evidence is exclusively `artifacts/cubecl_kernels_perf/phase6_valid_trees/`. The preceding rejection is historical, not current implementation status.

**Change.** A separate CPU kernel behind the same `exact_neighbors_kd` host signature, chosen once from client properties:
Current evidence index: `artifacts/cubecl_kernels_perf/phase6_valid_trees/summary.json` derives acceptance, cold first-minus-warm, host comparison and pipeline label proof. Selected C's algorithm is unchanged from measurement after unselected branch removal. `final_clean_gates.json` records all 11 passing gates; `integration_clean.json` records 20k/100k exit0 and `pipeline_clean_{20000,100000}/benchmark.json` both report labels equal. Introduced clippy warnings were corrected; `logs/final_clean_clippy.log` retains only pre-existing seed-grouping/oracle-range-loop and dependency future warnings, not suppressed warnings. Earlier gates/logs remain retained. Independent review receipts are maintained separately in the ultragoal ledger. Existing unrelated ignored campaign tests are not required KD CUDA skips (final KD:16 passed/0 ignored). README/AGENTS lack affected CPU schedule/buffer enumeration, so remain unchanged with rationale recorded in the summary.

The existing host tool/dependency are tracked at known HEAD `d8faeab5d0e825e4d93c6c1478e35b4d1f15721a` (`phase 6 failure`); `d030b7eb0a1f2a4edccc41fea01671ca9b16f092` identifies the original production baseline, not a required future HEAD. Recheck actual status on restart. If ignored historical artifacts are missing, establish fresh evidence from tracked source, never infer prior tests/timings. The detailed handoff records exact six input paths/hashes, code/test symbol map and isolated LLVM diagnostic capture. B remains the preserved original binary; only P/O/C share the new CPU kernel. Frozen reference stays unchanged; CUDA production source stays untouched during measurement except the separately gated Step 6 O-only finalization fold.

1. **Layout.** Tile the position axis of `tree.indices` globally as `soa[tile][feature][lane]`, masking positions outside each leaf range.
   - `W` comes from `client.io_optimized_vector_sizes(4)` and is passed as comptime.
   - Padded slots are masked out explicitly.
   - Carry a parallel `slot → original ID` array.
2. **Distances.** Each `Vector<f32, W>` lane is a different candidate and accumulates features `0..d` in order (Invariant 1 holds without a feature-axis tail).
3. **Admission.** Compare a whole vector against the broadcast `worst`, then do unrolled per-lane scalar checks and insert using the original ID.
4. **Query order.** Units take block-cyclic blocks of 64 leaf-ordered positions over `0..n` and write to `out[original_row]`. Valid permutation indices guarantee one writer per row without ownership machinery.
5. **Isolation.** Benchmark the packed candidates (steps 1–3) separately from query ordering (step 4), then combined. The report's ordering experiment could not separate these effects.
6. **No shared memory or barriers** in this kernel; `PLANE_DIM` is 1 on the CPU runtime.

**Also measure** the retained Rust/Rayon host traversal as reporting-only. Its primary-cell verification has zero ID mismatches but many distance-bit mismatches (unfused Rust versus observed LLVM `fmuladd`); it is not an exact oracle or an isolated runtime-overhead measurement.

**Accept when** H1 passes at every `W` the host can select, not only the widest, and H2 shows a CPU gain over Phases 1–4 including packing cost.

### Phase 7: CUDA plane-cooperative leaf-tile kernel (F5)

**Status:** Step 0 baseline evidence completed ([phase7_step0_report.md](phase7_step0_report.md)); kernel steps planned, not implemented or authorized. User decisions (2026-10-27): experimental timing only at 50k×16 and 300k×16 with no guard cells; a 300k gain of at least about 10% overrides 50k regressions; the accepted arm replaces the per-query kernel outright, with no row-count threshold. Detailed plan: [cubecl_kernels_perf_plan_phase7.md](cubecl_kernels_perf_plan_phase7.md). It supersedes points 1, 2 and 3 below: independent planes using `sync_plane`, the Phase 1 heap instead of register top-k, and leaf-ordered padded rows (AoS). It also adds a flat brute-force control arm, a dimension gate and measurements at 100k–300k rows.

**Change.** A separate CUDA kernel, gated on `features.plane` and the queried `plane_size_min`, behind the same host signature. The per-query kernel remains the non-plane path.
1. **Tiles.** One plane owns `PLANE_DIM` consecutive leaf-ordered queries. Each lane keeps its own query in registers (Phase 3) and its own register top-k (Phase 2).
2. **Traversal.** Use a cube-uniform traversal stack in `Shared`:
   - each lane computes `lower(q, node) <= worst(q)`;
   - `plane_any` decides whether the tile visits the node;
   - children are ordered by `plane_min` of the lower bounds.

   Control flow is uniform, so `sync_cube()` is never inside a masked branch.
3. **Candidate staging.** Stage leaf candidates in `Shared` in chunks of `PLANE_DIM × d_chunks` vectors (≤ 4 KB at 32×8×16 B):
   - each lane loads one contiguous padded candidate row from leaf-packed storage;
   - `sync_cube()`, then each lane computes in-order distances against every staged candidate (broadcast reads);
   - `sync_cube()` again before the next overwrite (write-after-read).
4. **Output.** Each lane writes its sorted top-k to `out[original_row]` once.
5. **Diagnostics.** Counters now mean tile visits; document this in the diagnostics output.
6. **Resources.** Size `Shared` against `max_shared_memory_size` on the host. Try multiple planes per cube to clear the 16-blocks/SM limit.
7. **Later.** Double-buffering via `memcpy_async` is deferred until its API is verified in CubeCL 0.11.1.

**Accept when** H1 passes on CUDA, which requires a GPU run because CPU tests never execute plane logic. H2/H3 must show a CUDA gain over the best of Phases 1–5 at both 20k and 50k, and full-fit CPU/CUDA label equality must be preserved.

### Phase 8: Cleanup and documentation

- Remove superseded kernel variants and the `#[cfg(test)]` reference entry point once its role is covered by the host oracle tests. No dead aliases.
- Update `AGENTS.md`/`README.md` wherever they describe KD traversal, the device-tree layout or resource sizing.
- Append the measured results, including negative ones, to `hdbscan_perf_report.md`.

## Out of scope / rejected

- **Feature-order-changing reductions** (`plane_sum`, vector horizontal sums, matmul-form distances). Rejected under Invariant 1 by decision.
- **Core extraction or readback fusion.** The measured share is negligible.
- **Atomics or single-pass inter-cube merging.** Each query's top-k belongs to one unit or plane, so no cross-cube merge exists.
- **Whole-input permutation.** It changes final partitions.
- **`kd_candidates_kernel` (Borůvka) and the >30-feature kNN/ANN path.** Phase 7's structure may carry over to `kd_candidates_kernel` later, subject to its component-bound atomics.
- **Routing threshold changes.** No evidence supports them.

## Verification commands (per phase)

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-hdbscan-core
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture
cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages
uv run --no-sync python analyze/benchmark_hdbscan_stages.py \
  --binary target/release/examples/profile_hdbscan_stages \
  --rows 20000 --run-dir artifacts/cubecl_kernels_perf/<phase>_20000 --timeout 120
```

Repeat the benchmark at 50k, and on CUDA phases run `profile_hdbscan_counters.py`. Use a fresh run directory per batch.

## Risks

| Risk | Mitigation |
| --- | --- |
| FMA contraction or codegen changes distance bits | H1 bit-equality on both backends at every phase |
| Register growth (Phases 2, 3, 7) causes spills or lower occupancy | H3 registers/local-memory metrics; Phase 5 sweep |
| Comptime variant count increases JIT time (CPU LLVM especially) | Limit to `d_chunks ≤ 8` × `k_max ∈ {8, 16}` (× `W` on CPU); measure first-fit compile time separately from warm fits |
| `DeviceKdTree` extension breaks fusion or Borůvka paths | Update all constructors; run the full `fast-hdbscan` test suite each phase |
| Plane path untested on CPU | Required-CUDA tests; per-query kernel stays the non-plane path |
| Gains don't hold on other distributions | Report the 16D/30D Gaussian scope explicitly; add a uniform/duplicate-heavy run before claiming generality |
