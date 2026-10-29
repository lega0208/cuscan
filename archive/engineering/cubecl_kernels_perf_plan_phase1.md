# Phase 1 detailed plan: register-cached heap root in `kd_neighbors_kernel`

This document details Phase 1 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md). Phase 1 also builds the shared harness pieces H1–H3, because every later phase depends on them. The source analysis behind it is in [hdbscan_perf_report.md](hdbscan_perf_report.md), under "Follow-ups: CubeCL kernel patterns", F2.

**Status:** implemented and accepted. CPU/CUDA golden, diagnostics and host-oracle checks pass; timing decision rules pass and cache is retained. Complete H3 evidence is captured, including sampled-stall share and theoretical occupancy. See `artifacts/cubecl_kernels_perf/phase1/summary.json` and the report's "Phase 1 results".

The plan was written against revision `2c3fc35fa2b2492f519c0adcb544779357f46dd8` with a clean production tree. Line numbers refer to that revision; re-read them before editing.

## Objective

Remove redundant global/L1 loads of the output-backed max-heap root from the KD neighbor traversal, without changing any traversal decision or output bit.

The kernel today reads the current k-th-best key `(out_distances[root], out_indices[root])` from global memory:

| Site (`crates/fast-hdbscan/src/kernels/kdtree.rs`) | Reads of the root per event |
| --- | --- |
| `:262` `let worst = out_distances[row * width]`, once per node pop | 1 per pop |
| `:282–284`, the admission test for each scanned candidate | 2, plus 1 more on distance equality |
| `:339–359`, child-push tests for each internal node | 2 |

At 20k×16 this is roughly 14,821 scanned positions and about 404 pops per query, so about 30k avoidable dependent loads per query. Under the visit-order invariant, the root changes **only** during admission and the final heapsort. A register copy that is refreshed after each admission is therefore exactly equivalent.

**Expected effect:** fewer L1TEX dependencies per candidate on CUDA, where long-scoreboard stalls are 70–79%, and fewer loads on CPU. This is a hypothesis to measure, not a promised speedup. On CPU the root is likely L1-resident already, so the gain may be small.

## Non-goals

- `kd_candidates_kernel` (Borůvka). It already keeps `best`/`best_index` in registers.
- Replacing the heap (Phase 2), query registers or padding (Phase 3), node records (Phase 4), and launch geometry (Phase 5).
- Any change to `box_lower_bound`, the traversal order, the diagnostics' meaning, the host API, `DeviceKdTree`, or the estimator.

## Invariants for this phase

1. Neighbor IDs and squared distances are **bit-identical** to the unmodified kernel on CPU and CUDA: compare `i32` IDs and `f32::to_bits` distances.
2. The per-row diagnostics `nodes_popped` and `leaf_positions_scanned` are **identical** to the unmodified kernel. Every pruning decision is unchanged, so the counters must match exactly. This is a stronger check than output equality.
3. The register pair always equals `(out_distances[row*width], out_indices[row*width])` between admissions. It is initialized to the same `(inf, n)` sentinel written to every heap slot.
4. Assertions, validation and the error contract are unchanged.

## Step 0: Preserve the baseline

1. Record `git rev-parse HEAD` and confirm that `git diff --stat crates/` is empty.
2. Build the baseline profiler binary:

   ```sh
   cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages
   ```

3. Copy it to `artifacts/cubecl_kernels_perf/phase1/bin/profile_hdbscan_stages.baseline`. Record its SHA-256 and the toolchain identity (`rustc -Vv`, `nvidia-smi` driver line) in `artifacts/cubecl_kernels_perf/phase1/provenance.json`. Use `analyze/preserve_build.py` if its interface fits; otherwise use a plain `cp` plus `sha256sum`.
4. Benchmark inputs:
   - 16D: reuse `artifacts/hdbscan_core_investigation/baseline_{20000,50000}/raw/input.f32le`.
   - 30D at 20k and 50k: regenerate with the generator used by `artifacts/hdbscan_core_investigation/run_sensitivity.py` (same parameters, seed 42, 30 features) into `artifacts/cubecl_kernels_perf/inputs/`. Record each input's SHA-256.

## Step 1: Golden harness (H1), landed before the kernel edit

### 1a. Frozen reference kernel (crate-internal, test-only)

In `crates/fast-hdbscan/src/kernels/kdtree.rs`:

- Copy the current `kd_neighbors_kernel` verbatim as `#[cfg(test)] #[cube(launch)] fn kd_neighbors_reference_kernel<F: Float, I: Int>(…)`, with the same signature.
- Add `#[cfg(test)] fn exact_neighbors_kd_reference(data, tree, k, diagnostics: bool) -> (ids, distances, diagnostics)`. It mirrors `exact_neighbors_kd` (`:586–636`): it reuses `into_contiguous`, `into_contiguous_tree`, `validate_tree`, `kd_launch_geometry` and the allocators. Unlike the production path, it returns the diagnostics tensor instead of emitting it.
- Add a matching `#[cfg(test)]` helper that launches the **production** kernel with an explicit `diagnostics_enabled` flag. The current entry point reads `FAST_HDBSCAN_KD_DIAGNOSTICS` via `kd_diagnostics_enabled()`, and test code must not depend on process environment. Factor the launch body of `exact_neighbors_kd` into a private `launch_kd_neighbors(data, tree, k, diagnostics_enabled) -> (ids, distances, diagnostics)`. `exact_neighbors_kd` calls it with `kd_diagnostics_enabled()` and emits diagnostics as today; tests call it directly. This is a refactor only, with no behavior change.

The reference kernel exists only until Phase 8, which removes it once later phases no longer need a kernel-vs-kernel oracle.

### 1b. Kernel-vs-reference tests (`kdtree.rs` `mod tests`)

Add a deterministic input generator without new dependencies, for example a SplitMix64/LCG mapped to f32 in [-4, 4) plus cluster offsets. Then add `check_kd_neighbors_match_reference(device)`, run for each case below:

| Case | Purpose |
| --- | --- |
| n = 2,000; d ∈ {1, 2, 3, 4, 5, 16, 29, 30}; k ∈ {1, 5, 16, 17} | width/tail coverage for later phases; small/large k |
| duplicate-heavy: 2,000 rows from 50 distinct points, d = 16, k ∈ {5, 17} | equal distances and the ID tie-break path at `:284` |
| integer-grid data (exactly representable distances), d = 3, k = 5 | many exact ties |
| n = 257 (just above the route threshold), d = 16, k = 5 | partial last cube, small tree |
| n = 9 and n = 1 with k > n − 1 | sentinel `(inf, n)` slots stay in the output |

For every case, assert:
- ids equal;
- `distances.map(f32::to_bits)` equal;
- the diagnostics vectors equal per row.

Run each case with `diagnostics = true` on both kernels, and once with `false` on the production kernel to check its outputs. Add `#[test]` wrappers for CPU and CUDA following the existing pattern: the CUDA wrapper skips only when `try_device_count() == 0` and `FAST_UMAP_REQUIRE_CUDA` is unset (see `kdtree.rs` `box_lower_bound_matches_point_box_oracle_cuda`). Keep sizes modest; the CPU runtime runs these at JIT plus scalar speed.

### 1c. Host-oracle test at realistic dimensionality (`tests/test_kdtree_search.rs`)

Add `check_kd_oracle_high_dim(device)` for d ∈ {16, 30}, n = 1,000, k = 5. It reuses the existing host-oracle construction (`check_kd_oracle`, `:128–148`): sequential f32 sum, then sort by `total_cmp` and then ID. Compare `to_bits`, not `==`. Add CPU and CUDA wrappers.

Run 1b and 1c against the **unmodified** kernel first:
- 1b must pass trivially, since the two kernels are identical.
- If 1c fails on CUDA, the cause is a pre-existing host/device bit difference (for example FMA contraction at 16D). Record it in the phase notes. Restrict 1c to the backends where it passes, document that restriction in a test comment, and rely on 1b for CUDA. Do not loosen comparisons, and do not treat this as a Phase 1 regression.

Commit point: the harness lands on its own, with all of its tests green against the unmodified kernel. Commit only on the user's instruction.

## Step 2: Kernel change

Edit `kd_neighbors_kernel` only (`kdtree.rs:221–422`). The sketch below is uncompiled.

```rust
// After the slot initialization loop (:245–248):
let root = row * width;
let mut worst_distance = F::new(f32::INFINITY);
let mut worst_index = n; // usize; matches the `I::cast_from(n)` sentinel

// Node pop (:261–264): replace the global read.
let node = stack[stack_len] as usize;
let lower = bound_stack[stack_len];
if lower <= worst_distance {
    …
    // Candidate admission (:281–287):
    if distance < worst_distance
        || (distance == worst_distance && candidate < worst_index)
    {
        out_distances[root] = distance;
        out_indices[root] = I::cast_from(candidate);
        // sift-down loop unchanged (:288–330)
        …
        // Refresh the cache from the new root: exactly one load pair per admission.
        worst_distance = out_distances[root];
        worst_index = u32::cast_from(out_indices[root]) as usize;
    }
    …
    // Child pushes (:339–359): compare against `worst_distance`
    // instead of `out_distances[row * width]` (four sites).
}
```

Details to get right:
- **Declaration form.** Declare the cache as `let mut` with `F::new(f32::INFINITY)` and a runtime `n`, the same way `best`/`best_index` are declared in `kd_candidates_kernel` (`:104–110`). That precedent compiles without `.runtime()`. If the 0.11.1 frontend rejects reassigning a literal-initialized variable, use `.runtime()` on the literal; do not restructure the logic.
- **Sentinel type.** `worst_index` must compare as `usize` exactly as `worst_index` does at `:282` today, `u32::cast_from(out_indices[root]) as usize`, so the tie comparison `candidate < worst_index` keeps its semantics, including the `n` sentinel.
- **Ordering of the refresh.** It must come **after** the sift loop. Before the sift, the root holds the new candidate, which is not necessarily the maximum. Refreshing before the sift would cache the wrong key.
- **`k == 1`.** The sift loop condition `parent < width / 2` is false, so the root is the only slot. The refresh then reloads the value just written, which is correct.
- **Untouched code.** The final heapsort (`:368–418`) is unchanged. It runs after traversal and does not use the cache.
- **Diagnostics.** The increments are untouched.
- Remove the old local `let root = row * width;` at `:281` in favor of the hoisted `root`. Keep `width` usage identical.

No host-code, launch-geometry, allocation or API changes are needed.

## Step 3: Correctness verification

Run these from the repository root, in order:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py
cargo test -p fast-hdbscan-core
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib kdtree -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --test test_kdtree_search -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --features tracing \
  --example profile_hdbscan_stages -- --nocapture
```

Gate: every test passes on CPU and CUDA with no hardware skips, and 1b shows identical diagnostics. Store the logs in `artifacts/cubecl_kernels_perf/phase1/logs/`. If any 1b case differs, the change is wrong by construction. Fix it; never adjust the test.

End-to-end semantic check: compare the labels emitted by every Step 4 benchmark process against the baseline on the same input and backend. They must be **equal** (not ARI-close), because core outputs are bit-identical and nothing else changed. The profiler emits labels for its final measured fit only; this check does not claim label capture for the other fits.

## Step 4: Performance measurement (H2, H3)

### 4a. Interleaved timing (H2)

- **Binaries:** the baseline (Step 0) and the variant, built the same way and copied to `artifacts/cubecl_kernels_perf/phase1/bin/profile_hdbscan_stages.variant` with its SHA-256 recorded.
- **Native command form** (from `artifacts/hdbscan_core_investigation/alternating50.json`):

  ```sh
  <binary> <input.f32le> <rows> <features> --backend {cpu|cuda} --runs 3 --min-samples 5
  ```

- **Matrix:** {20k, 50k} × {16D, 30D} × {cpu, cuda}: all eight cells required by the parent H2 contract.
- **Order:** for each cell, run processes in `B,V,V,B,B,V` order (6 processes, 3 measured fits each after the binary's own warmup). Run cells serially; never run CPU and CUDA lanes concurrently.
- **Runner:** adapt the alternation loop from `artifacts/hdbscan_core_investigation/run_experiments.py` into `artifacts/cubecl_kernels_perf/phase1/run_alternating.py`, writing to its own output paths. It records commands, stdout/stderr, profile JSON and labels for each process.
- **Recorded per cell:** the median and [min–max] of the core stage (`core_distances`) and of the neighbor operation+sync scope, the full-fit median, and the label-equality result.
- **Statistical unit:** first take the median of each process's three measured fits for each timing scope; use the three process medians per binary for the per-cell median and [min–max] and every Step 5 comparison. Preserve all fit samples and their process grouping in `summary.json`; within-process fits are not independent trials.

### 4b. CUDA counters (H3)

- Extend `analyze/profile_hdbscan_counters.py`: add a CLI option for the binary label and add these metrics to its `--metrics` list:
  - registers/thread (`launch__registers_per_thread`);
  - local memory traffic (`l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum`, `…_op_st.sum`);
  - global load sectors/request (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` divided by `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum`);
  - `smsp__pcsamp_warps_issue_stalled_long_scoreboard` share;
  - achieved/theoretical occupancy.

  Verify the metric names against `ncu --query-metrics` for Nsight Compute 2026.2.1 before relying on them. Keep the existing metrics so results stay comparable with the report.
- Extend the parser's metric-specific unit/type normalization, not just the metric list: accept validated nonnegative sector/request/sample counts (including zero), positive registers/thread, and the existing duration/percentage types. Define sectors/request as global-load sectors divided by global-load requests; define long-scoreboard share as its sampled stall count divided by the corresponding total sampled-warp count, using compatible queried metrics. Record the raw numerator, denominator and units. A zero denominator produces an explicitly unavailable ratio with a reason, never a fabricated zero or an exception; unavailable required evidence must be resolved or reported as a blocked H3 capture.
- Update success and failure CSV fixtures in `analyze/tests/test_hdbscan_stages.py` for the complete required metric set, valid zero counts, zero denominators, missing metrics, malformed values and incompatible units. Keep strict missing-metric validation. Step 3 includes these tests.
- Capture 20k×16 and 50k×16 for both binaries, in fresh run directories under `artifacts/cubecl_kernels_perf/phase1/ncu_{baseline,variant}_{rows}/`. Before capture, run `analyze/benchmark_hdbscan_stages.py` with the corresponding preserved binary and matching workload into each directory to provision `raw/manifest.json`, `raw/input.f32le` and `benchmark.json`. Verify identity, input hash and `profiles.cuda.resolved_search_per_fit` satisfy the counter tool's existing contract. Include binary label and SHA-256 in capture provenance and validate them against the selected binary. These serial provisioning fits are not substitutes for the interleaved H2 samples.

### 4c. Expected observations (falsifiable)

- Global load requests per kernel drop. Long-scoreboard share drops or stays flat.
- Registers/thread may rise by 1–2. It must stay at or below about 128 for the current 32-thread geometry to keep its occupancy.
- If registers rise and time does not improve, record it; do not tune around it in this phase.

## Step 5: Decision rule

Accept Phase 1 when **all** of the following hold:
1. Every Step 3 gate passes and labels are equal end-to-end.
2. In at least one of CPU and CUDA, the variant's core-stage median is lower than the baseline's at both 20k×16 and 50k×16, with the variant's max below the baseline's median in at least one of those cells. In other words, the ranges show a separation, not just a shift in medians.
3. In every one of the eight cells, the variant's core-stage median minus the baseline's core-stage median is no larger than the baseline's core-stage max minus min, using process medians as defined in Step 4a.

If (2) fails on both backends, revert the kernel edit and keep the harness. Record the negative result with its numbers in `hdbscan_perf_report.md`. Then proceed to Phase 2, which subsumes the root cache because a register top-k makes the root a register by construction.

## Step 6: Deliverables

- The kernel edit plus the harness (reference kernel, `launch_kd_neighbors` refactor, and the 1b/1c tests).
- `artifacts/cubecl_kernels_perf/phase1/` containing provenance, binaries plus hashes, logs, alternation records, ncu CSVs, and a `summary.json` with the per-cell medians/ranges and the decision.
- A short "Phase 1 results" subsection appended to `hdbscan_perf_report.md`. It states what was measured, on which backend, and whether the decision rule passed, without generalizing beyond the measured cells.
- A Phase 1 status line updated in `cubecl_kernels_perf_plan.md`.
- No `AGENTS.md`/`README.md` change: the behavior, API and layout are unchanged.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Refresh placed before the sift caches a non-maximum root, silently weakening pruning or admitting wrongly | Diagnostics-equality in 1b catches any pruning change; output equality catches any admission change |
| `usize`/`I` sentinel mismatch changes tie behavior at the `n` sentinel | 1b cases with k > n − 1 and the duplicate-heavy ties |
| CubeCL 0.11.1 frontend rejects reassigning the literal-initialized cache | Use `.runtime()` on the initializer; follow the `kd_candidates_kernel` precedent |
| Pre-existing CUDA/host FMA bit differences surface in the new 1c test | Run 1c on the unmodified kernel first; document and scope it; 1b remains the CUDA gate |
| Gain is within noise (likely on CPU) | The decision rule requires range separation; record negative results instead of stacking them |
| Benchmark drift between batches | Compare only within the interleaved batch; serial cells; same inputs and hashes |
