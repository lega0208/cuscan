# Phase 5 detailed plan: tune CUDA KD-query launch geometry

This document details Phase 5 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md). **Status: implemented and measured; 64 selected by explicit user override on 2026-10-26.** Both larger widths failed the default regression screen; the user accepts the 64-thread 20k×30 cost (+1.443 ms core, +3.62%; full fit +1.37%) in exchange for the measured 50k CUDA gains (−19.35% core at 16D, −9.95% at 30D; full fit −5.77%/−2.72%). Larger row counts are the intended practical workload; these measurements do not establish gains at every larger size. Width 128 remains rejected. Phase 2 and Phase 4 were rejected. The experiment baseline is the accepted Phase 3 kernel, not cumulative Phases 1–4. The protocol below is preserved as the original experiment contract, not a claim that its default screen passed. Measured results, resources and validation are recorded in [the performance report](hdbscan_perf_report.md#phase-5-results-cuda-neighbor-launch-geometry-64-selected-by-override) and `artifacts/cubecl_kernels_perf/phase5/`; `override_64.json` supersedes the original `decision.json` policy without rewriting raw evidence.

## Objective and scope

Compare CUDA cube widths 32, 64 and 128 for `kd_neighbors_kernel`. Retain 32 unless larger geometry has a reproducible benefit at 50k without meaningful 20k regressions. Larger cubes can alter block residency, scheduling and resource allocation; they do not reduce the algorithm's work. Occupancy estimates in the master plan are hypotheses, not measurements on the current Phase 3 kernel.

Only the neighbor launch geometry changes. Preserve CPU geometry, padded vector packing, exact-feature register-query specialization, traversal, top-k admission, feature order, diagnostics and output layout. Do not revive Phase 2 register top-k or Phase 4 compact nodes. Do not change `kd_candidates_kernel`, routing thresholds, public APIs, allocation sizing, the frozen reference, query ordering, or plane cooperation. Phase 6 and Phase 7 are separate experiments.

## Pre-experiment implementation and file-level plan

The relevant implementation is `crates/fast-hdbscan/src/kernels/kdtree.rs`:

- `KD_CUBE_DIM` is currently `CubeDim::new_1d(32)`.
- `kd_launch_geometry` selects one CPU cube using `hardware.num_cpu_cores`, legalized against client limits. CUDA uses `KD_CUBE_DIM` and `hardware.max_cube_count.0`.
- `launch_count` computes `min(ceil(rows / dim.x), max_cubes)` with checked conversion to `u32`.
- `launch_kd_neighbors` packs data and both bounds, then calls the geometry helper immediately before launching `kd_neighbors_kernel`.
- The kernel advances rows using a grid stride; bounded cube count must continue to cover all rows.

Implement a private preferred-width parameter at the neighbor launch boundary, using the same legality and count calculation as today. Keep the CPU branch byte-for-byte equivalent in behavior. Inspect every caller of `kd_launch_geometry` before modifying its signature: geometry shared with unrelated launches must not silently inherit the experimental neighbor width. Keep the reference launch fixed at baseline geometry, so production and reference cannot share an accidental coverage defect.

Use private test hooks to exercise each requested width directly. For timing, preserve separate release binaries built with fixed preferred widths 32, 64 and 128. Do not introduce a public setting, environment-variable parser, permanent sweep dispatcher or workload autotuner. After selection, retain only the selected production width and useful legality/coverage tests; remove experiment-only production branches. Preserve experiment diffs and binaries in the evidence directory.

Keep unrelated geometry consumers at baseline: candidates, reset, padding, the frozen reference, the existing helper near `kdtree.rs:1763` and the direct `KD_CUBE_DIM` box-lower-bound test near `:1972`. Add explicit neighbor-only hooks rather than globally changing the shared constant.

## Geometry legality and coverage contract

For each CUDA request `w` in `{32, 64, 128}`:

1. Obtain actual dimensions with `fast_cubecl_common::legal_cube_dim(client, CubeDim::new_1d(w))`. The helper in `crates/fast-cubecl-common/src/kernels/mod.rs` bounds each axis and total units against `hardware.max_cube_dim` and `hardware.max_units_per_cube`.
2. Record requested width and actual `(x, y, z)`, hardware limits, plane-size properties and `hardware.max_cube_count`. A legalized width is not necessarily the requested width. Deduplicate identical actual geometries; do not report a clamped 64 request as a measured 64-thread launch.
3. Require positive usable dimensions and positive CUDA x cube capacity. Preserve the existing assertion when the device reports no legal KD launch capacity. Do not substitute an unchecked launch or silently bypass preflight.
4. Set `count_x = min(ceil(rows / actual_x), max_cube_count.x)` and `count_y = count_z = 1`. Keep arithmetic bounded and conversions checked. Derive the row stride from the actual launched grid, never the preferred width.
5. Preserve grid-stride processing when the cube count is capped. For every nonempty validated input each row has exactly one owner, and all rows are reached, including partial cubes and multiple stride rounds. Retain existing empty-input validation rather than manufacturing a zero-work GPU launch.
6. CPU continues to use legalized `num_cpu_cores.max(1)` and one cube. Never replace its width with the CUDA candidate.

Host dimension legality does not guarantee that a particular compiled kernel fits device resources. Compile and execute all eligible CUDA variants, capture launch errors and compiler resource metadata, and mark a resource-infeasible candidate ineligible. Do not clamp it again based on inferred register capacity or label it a successful benchmark. Registers per thread, register-allocation granularity, blocks per SM and local spills must be observed, not inferred from nominal width. In particular, Phase 3's measured 128-register 30D case is not a universal register ceiling or proof that 128-thread cubes are infeasible.

## Correctness invariants and test hooks

Each query remains an independent unit with the same ordered operations and output row. Changing ownership across CUDA blocks cannot legitimately change neighbor IDs, distance bits, traversal counts or labels. No barriers, shared traversal state or cross-query reductions are added.

Extend the existing `check_kd_neighbors_match_reference` harness to run the production neighbor launcher at each distinct legal CUDA width while retaining the frozen `kd_neighbors_reference_kernel` and `exact_neighbors_kd_reference` unchanged. Preserve all existing cases rather than replacing them with a small smoke test. Check IDs exactly, distances with `f32::to_bits`, and both per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) exactly. Check diagnostics-enabled and disabled launches give the same outputs. CPU reference coverage remains required despite unchanged CPU policy.

Add focused geometry tests for preferred widths 32/64/128, per-axis and total-unit clamping, duplicate legalized widths, positive capacity, count caps, checked conversions, and CPU independence. Include row counts around each legal width (`w-1`, `w`, `w+1`, plus 1) and around the capped launch capacity. Use a scoped test-only maximum-cube override or pure coverage calculation for synthetic small caps; do not require impossibly large allocations or mutate process-global settings across parallel tests. At least one device test must exercise multiple grid-stride rounds, not merely assert a host count formula.

Retain coverage for register and runtime query paths: features 2, 16 and 30, tail dimensions and dimensions above 32 from the existing harness; k boundaries including 1, 8, 16 and 17; ties, duplicate points, small/root-leaf trees and partial final cubes. Exercise each candidate through the same internal launcher used by production, not a separately implemented kernel. A legal width unavailable on the test GPU must be recorded as unavailable rather than claimed tested.

Future implementation verification commands (not run for this planning document):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-hdbscan-core
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib kdtree -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --test test_kdtree_search -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture
cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages
uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py
```

A CUDA claim requires an actual `FAST_UMAP_REQUIRE_CUDA=1` run, not a CPU pass or optional-device skip. Any mismatch blocks timing-based acceptance; diagnose it without changing the oracle, tolerating distance differences, or excluding the failing input. Verify complete baseline/variant label equality for every performance cell as well as low-level kernel equivalence. MST/oracle expectations remain unchanged.

## Reproducible experiment preparation

Create future evidence under `artifacts/cubecl_kernels_perf/phase5/`; this plan itself creates no experiment artifacts. Record baseline commit, complete relevant working-tree diff and hash, compiler/package versions, build flags, device identity, driver, queried hardware properties, input hashes, binary SHA-256 values and exact commands in provenance. Preserve unrelated user changes; a baseline requirement is not permission to reset the worktree.

Build and preserve a fresh Phase 3 baseline and fixed-width 64/128 variants using the same release/tracing configuration. Confirm the 32 variant matches the baseline source except for the experiment hook. Hash binaries after copying them to immutable evidence paths and before each campaign. Record each binary's requested and actual geometry for every cell. Do not let a subsequent build overwrite the binary being measured.

Reuse `artifacts/cubecl_kernels_perf/phase3/run_alternating.py` and its established process-median semantics. Preserve a copy and diff any changes. Permitted campaign changes are binary/input paths, width labels, cell selection, batch labels, explicit candidate ordering and exclusive per-batch output directories. Reuse existing deterministic inputs, checking hashes rather than regenerating convenient workloads. Keep backend, min-samples, run counts, synchronization, warmup and diagnostics settings identical across arms. Correctness runs may enable diagnostics; timing runs must use the same declared diagnostics policy for all arms and must not silently mix instrumentation states.

## Timing matrix and fixed comparison protocol

Primary matrix: rows `{20000, 50000}` × features `{16, 30}` × backend `{cpu, cuda}`: eight cells per comparison. Use the same min-samples 5 and three measured fits per process as the Phase 3 campaign:

```text
<immutable-binary> <input.f32le> <rows> <features> --backend <cpu|cuda> --runs 3 --min-samples 5 --labels-output <unique-labels-path>
```

Compare each distinct legal nonbaseline CUDA geometry separately against 32, including CPU controls of its binary. CPU controls detect unintended dispatch/build differences; they are not a claimed CPU optimization. Declare candidate and cell order before measurements. Run cells serially, with no competing benchmark, test, NCU or compilation process.

For each candidate and cell, batch A uses `B,V,V,B,B,V`, yielding three independent process medians per arm. After completing the entire declared candidate/cell matrix, repeat the entire matrix as batch B, reversing candidate order to reduce systematic ordering bias. Run both batches regardless of the first result. No optional third batch, early stopping, favorable subset selection, or post-hoc width added after inspecting results.

Report core-stage, neighbor-scope and full-fit timings separately, including median, min–max and absolute milliseconds. Neighbor scope includes packing and launch overhead; kernel-only duration is not an end-to-end substitute. Validate labels in all processes. Timing failures and missing cells remain visible; a failed process is not a zero-duration sample.

For each cell, pool the six baseline process medians and six variant process medians from both batches. Define `delta = median(V) / median(B) - 1`. Also report the six adjacent-pair V/B ratios: positions 1–2, 3–4 and 5–6 within each batch, always expressed variant over baseline regardless of temporal order. Preserve both batch summaries and every raw process result. Treat process medians, not the individual fits within a process, as replication units.

## CUDA resource and scheduler evidence

Use existing Phase 3 counter tooling and the route-aware analysis harness. Check the actual `kd_neighbors_kernel` match and geometry in profiler output; do not profile Boruvka or a similarly named reference kernel by accident. Profile baseline and every eligible candidate at 20k/50k and 16D/30D. Keep counter collection separate from timing and preserve exact commands, tool versions, launch selection, raw reports and parsing outputs.

Required readings: actual threads per block and grid size; registers per thread and available register/block metadata; theoretical and achieved occupancy; active blocks/warps and limiting resource when available; local-memory load/store sectors; long-scoreboard stalls; global load requests/sectors; kernel duration. Record unsupported metrics explicitly rather than inventing zeroes. Single matching launch or replay readings describe that selected launch, not a full-fit average. Use a consistent selected launch/warmup policy across variants.

Interpret counters as explanations. Larger blocks may reduce resident blocks or total scheduling flexibility even when nominal occupancy improves. Local-memory traffic may reflect compiler layout or spills and needs investigation; a register count alone cannot establish a spill. The arithmetic and traversal are invariant, so changed diagnostics are a correctness defect, not a performance explanation. Packing kernels are unchanged; any pack timing drift is not evidence for geometry-specific packing improvements. Use a same-fit timeline when reporting pack share, never divide unrelated profiler durations by benchmark medians.

Record first-fit/warmup timing in fresh processes separately for each binary under the same documented cache state. Existing Phase 3 cold-start tooling does not clear compilation caches or isolate kernel compilation; describe it accordingly. Cold-start differences and JIT specialization costs are reporting items, not grounds to select warm samples.

## Predeclared selection rule

Core-stage process medians drive all gain/regression screens, CPU controls and the geometric-mean width tie-break below. Neighbor-scope timing is explanatory, not a substitute selection series. Apply the same regression screen independently to full-fit process medians in every cell: a practical full-fit regression vetoes default acceptance even when core time improves. Include both series in any explicitly justified exception. Timing arms all use `kd_diagnostics_enabled = false`; diagnostics comparisons run separately.

Use the fixed two-batch pooled statistics. A practical gain is `delta <= -2%` with at least five of six paired ratios below 1. A practical regression is `delta >= +3%` with at least five of six ratios above 1. These screens describe consistency, not formal statistical significance; intermediate results do not prove equivalence.

A candidate is eligible only if all correctness and legality checks pass on its actual geometry, the complete matrix and provenance are available, and no unresolved launch/resource failure remains. Default acceptance requires:

1. A gain in at least one of the two 50k CUDA primary cells, with no regression in the other.
2. No regression in either 20k CUDA cell or any CPU control.
3. Full label equality and exact low-level outputs/diagnostics at all tested widths.
4. A defensible resource explanation; investigate new local-memory traffic, occupancy collapse or unexpectedly changed register counts before accepting.

If both 64 and 128 qualify, compare their geometric mean V/B ratio across the four CUDA primary cells, using each candidate's own interleaved baseline. Prefer the smaller mean; if within one percentage point, prefer the smaller width. Do not introduce dimension-specific or row-count-specific dispatch from this small matrix. A single selected CUDA width must justify itself across the matrix. Preserve 32 when neither candidate has a defensible benefit or comparisons are inconclusive.

Performance thresholds are default decision aids, not permission to hide trade-offs. An exception requires an explicit evidence-backed rationale reporting each unmet criterion, both batches, absolute costs, contrary cells and resource costs. Correctness is never discretionary. Any direct 64-versus-128 follow-up is a separately planned experiment, not an unreported extension used to choose a favorable result.

## Failure handling and deliverables

After removing experiment-only production hooks and selecting the winning width (or restoring 32), run final integration validation on that exact cleaned source:

```sh
bash scripts/analyze_hdbscan_stages.sh --rows 20000
bash scripts/analyze_hdbscan_stages.sh --rows 100000
```

Both campaigns must complete strict CPU and real CUDA execution, exact CPU/CUDA label checks, route-aware kernel/counter captures and successful summaries. Preserve run directories, raw logs, source/binary/input hashes and exact commands. Compilation alone, skipped CUDA or incomplete profiler outputs do not satisfy this gate. Resolve failures before declaring implementation complete. These are integration gates, not additional selection samples; do not use them to retune the width. Follow the script's scale-specific limits, including runtime JIT. Repeat source-sensitive correctness gates on the cleaned implementation.

- Correctness failure: stop acceptance, inspect row ownership/stride, actual-versus-requested dimensions, reference independence and dispatch. Never weaken equality tests.
- Illegal/clamped geometry: report actual dimensions, deduplicate identical experiments and retain legal baseline behavior. Resource compile/launch failure makes that candidate ineligible pending a real fix.
- No gain or inconsistent evidence: retain 32 and record a negative/inconclusive result. Do not pursue extra batches until a preferred answer appears.
- Acceptance: retain only the winning CUDA preference, unchanged CPU policy and useful tests. Rejection: remove experiment-only production hooks while preserving unrelated work and evidence.

Implementation deliverables will include the minimal neighbor-only geometry change and tests, immutable hashed binaries and diffs, provenance, raw logs/timings/counters, pooled and per-batch summaries, correctness evidence and a machine-readable decision summary. Append measured results, including rejection, to `hdbscan_perf_report.md` and update the master plan's status. Update runtime guidance only where actual behavior changes; no public configuration or memory-layout documentation change is expected. Do not claim Phase 5 complete merely because this planning document exists.
