# Phase 2 detailed plan: register top-k for small k in `kd_neighbors_kernel`

This document details Phase 2 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md). It depends on the Phase 1 harness (`cubecl_kernels_perf_plan_phase1.md`, Step 1), but not on the Phase 1 kernel edit being accepted. The source analysis is in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups", F2 (second part) and opportunity 5.

**Status:** measured and rejected. The exact-comptime-`k` experiment passed output/diagnostic equivalence but failed mandatory decision rules 3, 4 and 6; the register-top-k kernel was reverted, the accepted Phase 1 root-cache heap remains in production, and the extended equivalence/profiling harness is retained. Evidence: [`artifacts/cubecl_kernels_perf/phase2/summary.json`](artifacts/cubecl_kernels_perf/phase2/summary.json), [`cold_start.json`](artifacts/cubecl_kernels_perf/phase2/cold_start.json), [`provenance.json`](artifacts/cubecl_kernels_perf/phase2/provenance.json), and [`measured_kernel.patch`](artifacts/cubecl_kernels_perf/phase2/measured_kernel.patch); full measured outcome: [hdbscan_perf_report.md](hdbscan_perf_report.md#phase-2-results-register-top-k-experiment-rejected). The instructions below are retained as the historical experimental design; they are not an implementation specification for production.

## Objective

For small neighbor widths, replace the output-backed global max-heap with a sorted list held entirely in registers. Output and pruning must stay bit-identical. This removes:

- every global read-modify-write of the heap during traversal: the admission writes plus the sift-down's loads and stores at `kdtree.rs:286–329`;
- the slot initialization pass at `:245–248`;
- the final in-place heapsort at `:368–418`.

The result is written to `out_indices`/`out_distances` once per query, in ascending order. Widths above the register threshold keep the heap kernel, including the Phase 1 cache if accepted.

**Expected effect:** fewer global memory operations and dependent loads per admission, and no heapsort. This is a hypothesis to measure. The report measured that k5 → k30 costs only +21% CPU / +17% CUDA core time, and that increase is mixed with weaker pruning. Heap traffic is therefore **not** demonstrated to be dominant, and a small or null result is a plausible outcome.

## Change from the master plan

The master plan proposed `k_max ∈ {8, 16}` buckets with a runtime `k`. This plan uses an **exact comptime `k`** for `1 ≤ k ≤ 16` instead:

- With a runtime `k` below a comptime `k_max`, the worst entry sits at runtime index `k − 1`. That dynamic index would push the list into local memory, defeating the purpose. Masked updates over `k_max` slots would avoid that, but do more work.
- An exact comptime `k` makes every list index comptime, enabling scalarization under `#[unroll]`; register residency must still be verified in generated code and counters.
- A fit uses one `k`, so a process JIT-compiles only one variant. The cost of more variants is borne by tests and by the first fit for each new `k`, which Step 4 measures.

Update the master plan's Phase 2 text when this phase completes.

## Non-goals

- `kd_candidates_kernel` (Borůvka), whose candidate is already in registers.
- Query registers or padding (Phase 3), node records (Phase 4), launch geometry (Phase 5).
- Changing the host API, `DeviceKdTree`, `core_distances`, or how `k` is computed in the estimator.
- Tuning the threshold beyond the measurements in Step 4.

## Invariants

1. Output IDs and `f32::to_bits` distances are bit-identical to the reference kernel for every `k`, including sentinel slots `(inf, n)` when `k > n − 1`.
2. Per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) are identical to the reference kernel. The list's last element equals the heap root at every step, so every pruning decision is unchanged.
3. The ordering key is the lexicographic `(distance, original ID)`, identical to the heap's admission test (`:283–284`) and the heapsort's ascending order.
4. Leaves are disjoint and each node is popped at most once, so every candidate ID is offered at most once per query. Equal keys can only arise between sentinels, which are never offered.
5. Inputs are validated finite, but distances can overflow to `+inf` (covered by `check_overflow_distances` in `tests/test_kdtree_search.rs`). `(inf, j)` with `j < n` must outrank the sentinel `(inf, n)` exactly as the heap's `distance == root && candidate < root_index` test does. NaN is impossible: finite inputs give finite or `±inf` deltas, and the squares are `≥ 0` or `+inf`.

## Equivalence argument

Let `S_t` be the set of the k smallest keys among the sentinels and the candidates offered by step `t`.

- The current heap stores `S_t` with its maximum at the root.
- The proposed list stores `S_t` sorted ascending, with its maximum in slot `k − 1`.
- Both admit a candidate if and only if its key is strictly less than `max(S_t)`.
- Both use `max(S_t).distance` for pruning.

By induction over the offered sequence, which traversal produces identically when every decision is identical, the two hold the same `S_t` and make the same decisions. At the end, the heapsort emits `S_T` ascending, and the list already is `S_T` ascending. Distances are the same `f32` values, copied without arithmetic.

## Step 0: Baseline and prerequisites

1. The Phase 1 harness (reference kernel, `launch_kd_neighbors` refactor, and the 1b/1c tests) is on the branch and green.
2. Preserve the current production binary, with Phase 1's decision applied, as `artifacts/cubecl_kernels_perf/phase2/bin/profile_hdbscan_stages.baseline`, with its SHA-256 and provenance. This is the Phase 2 baseline: the comparison is against the best accepted state, not the original commit.
3. Confirm the CubeCL 0.11.1 APIs this phase uses. All are present in the registry source:
   - `Array::<T>::new(#[comptime] length: usize)` (`frontend/container/array/base.rs:45`);
   - `select(condition, then, or_else)` (`frontend/operation/branch.rs:14`);
   - `#[comptime] … : usize` kernel parameters (precedent: `component_reduce.rs:226`);
   - `#[unroll]` over a comptime range (precedent: `component_reduce.rs:53`).

## Step 1: Extend the harness

In `kdtree.rs` `mod tests`, extend `check_kd_neighbors_match_reference`. The register kernel is reached through the production `launch_kd_neighbors`, so every case runs whichever path the host dispatch selects.

| Case | Purpose |
| --- | --- |
| n = 2,000, d ∈ {2, 16, 30}, every k in 1..=17, plus k ∈ {24, 30} | each comptime variant; the 16/17 threshold edge; the heap path |
| duplicate-heavy (50 distinct points × 40), d = 16, k ∈ {1, 5, 16} | equal distances, ID tie order inside the list |
| integer grid, d = 3, k ∈ {5, 16} | many exact ties |
| overflow input (values near `f32::MAX`, as in `check_overflow_distances`), k ∈ {1, 5, 16} | `(inf, j)` vs sentinel `(inf, n)` |
| n = 1, k = 1; n = 4, k ∈ {3, 5, 16}; n = 9, k = 16 | sentinel slots remain in the list and output |
| n = 257, d = 16, k = 5 | partial last cube |

Assert equal IDs, equal distance bits and equal per-row diagnostics against `exact_neighbors_kd_reference`, on CPU and on CUDA (mandatory under `FAST_UMAP_REQUIRE_CUDA=1`). Keep CPU sizes modest, because each new `k` triggers a JIT compile. If the CPU test time becomes excessive, run the full `1..=17` sweep at n = 300 and the large-n cases only for k ∈ {1, 5, 16, 17}.

Add one host-dispatch unit test: the `k` threshold selects the register path for `k ≤ KD_REGISTER_TOPK_MAX` and the heap path otherwise. Assert on the selection function, not on timing.

### Measurement harness prerequisites

Extend the existing provisioning and profiling tools rather than relabeling fixed-k5 captures:

- Parameterize `min_samples` in `analyze/benchmark_hdbscan_stages.py` and `analyze/profile_hdbscan_counters.py`, propagate it into native arguments and manifest identity, and reject mismatches between provisioned workload, requested counters and benchmark configuration.
- If Step 2a chooses a separate kernel, adapt the selector in `analyze/hdbscan_stages_common.py` and its consumers to the actual dispatched neighbor kernel; reject a missing or wrong-path capture.
- Add supported global-store request metrics alongside global-load metrics, updating parsing and fixtures in `analyze/tests/test_hdbscan_stages.py`. Test k16 argument/manifest propagation, configuration mismatch rejection, kernel selection and store-metric parsing.
- Preserve fresh output checks and binary/input/configuration identity validation. Counter captures remain conditional first-warmup-launch samples, not whole-fit averages; record that scope with each result.

## Step 2: Kernel design

### 2a. Code structure

The traversal (stack, box bounds, pruning, child ordering, diagnostics) must stay single-sourced, while the top-k container varies. Put the comptime branch only in three small `#[cube]` helpers, never in the traversal body:

```rust
// uncompiled sketch
const KD_REGISTER_TOPK_MAX: usize = 16;

#[cube]
fn topk_worst<F: Float>(… state …, #[comptime] reg_k: usize) -> (F, usize);
#[cube]
fn topk_offer<F: Float, I: Int>(… state …, distance: F, candidate: usize, #[comptime] reg_k: usize);
#[cube]
fn topk_finish<F: Float, I: Int>(… state …, out_indices, out_distances, row, width, #[comptime] reg_k: usize);
```

- `reg_k == 0` selects the existing output-backed heap: initialization, admission with sift and the Phase 1 cache, and the final heapsort. It moves into the helpers unchanged.
- `reg_k ∈ 1..=16` selects the register list.
- Add `#[comptime] reg_k: usize` to `kd_neighbors_kernel`. The runtime `k` stays the width for the heap path and must equal `reg_k` when `reg_k > 0`. Add a host assert, not a kernel check.

The helpers need state (the list arrays). Construct it in the kernel as `Array::<F>::new(comptime![max(reg_k, 1)])` and `Array::<u32>::new(…)`, and pass it by `&mut`. If the 0.11.1 frontend cannot pass `Array` by `&mut` into `#[cube]` functions or reject a length-1 placeholder, write the register variant as a separate `kd_neighbors_topk_kernel` and extract the shared traversal pieces (child push, leaf scan bounds) into `#[cube]` functions. Decide by compiling; record the choice in the phase notes.

### 2b. Register list

State: `list_d: Array<F>` and `list_i: Array<u32>` of length `reg_k`. Every index is comptime under `#[unroll]`, which allows scalarization. Initialize every slot to the sentinel `(inf, n as u32)` with an unrolled loop.

**Worst:** `(list_d[reg_k − 1], list_i[reg_k − 1])`, a comptime index. This replaces the Phase 1 cache entirely; there is no separate register pair.

**Admission test:** unchanged form, against the worst slot:

```rust
let wd = list_d[reg_k - 1];
let wi = list_i[reg_k - 1] as usize;
if distance < wd || (distance == wd && candidate < wi) { insert(distance, candidate) }
```

**Insertion:** a forward carry pass. It is correct for an ascending list, and every index is comptime:

```rust
// uncompiled sketch
let mut cd = distance;
let mut ci = candidate as u32;
#[unroll]
for s in 0..reg_k {
    let sd = list_d[s];
    let si = list_i[s];
    let less = cd < sd || (cd == sd && ci < si);
    list_d[s] = select(less, cd, sd);
    list_i[s] = select(less, ci, si);
    cd = select(less, sd, cd);
    ci = select(less, si, ci);
}
// The final carry is the evicted previous worst; it is dropped.
```

After the admission test passes, the incoming key is smaller than slot `reg_k − 1`. The carry pass places it at its sorted position, shifts the larger keys up by one, and drops the old worst. `select` is intended to avoid per-slot branches, but branchless lowering is not guaranteed by the API; inspect generated code. If 0.11.1's `select` is not available for the tuple element types used, use `if less { swap }` with comptime indices and verify scalarization. Cost: `reg_k` compare/select steps per admission, and admissions are rare once the list fills.

**Comparison types:** compare IDs as `u32` and convert to `usize` only for the `candidate < wi` admission test, so the semantics match `:282–284`. The sentinel `n` fits in `u32`, because `validate_tree` asserts `rows` fits `i32`.

**Finish:** an unrolled `for s in 0..reg_k`, writing `out_distances[row * reg_k + s] = list_d[s]` and `out_indices[row * reg_k + s] = I::cast_from(list_i[s])`. There is no initialization pass and no heapsort.

### 2c. Traversal edits

These apply to both variants through the helpers:

- Node pop (`:262`) and child pushes (`:339–359`) use `topk_worst(...).0` in place of `out_distances[row * width]`, or in place of the Phase 1 register if Phase 1 landed.
- The leaf candidate loop (`:281–330`) becomes `topk_offer(...)`.
- After traversal, `topk_finish(...)` replaces the heapsort.
- Diagnostics increments are untouched.

### 2d. Host dispatch

In `launch_kd_neighbors` (from the Phase 1 refactor), pass `reg_k = k` when `1 ≤ k ≤ KD_REGISTER_TOPK_MAX`, otherwise 0. `k > 0` is already asserted at `:595`. Keep the selection in one small function so the dispatch test can check it. Output allocation, launch geometry and diagnostics emission are unchanged.

## Step 3: Correctness verification

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

Gate: everything passes on CPU and CUDA with no hardware skips. The Step 1 cases show identical outputs **and** diagnostics for every `k`. Logs go to `artifacts/cubecl_kernels_perf/phase2/logs/`. A diagnostics mismatch means the worst slot diverged from the heap root, so the implementation is wrong. Fix the kernel, never the test.

End-to-end: final labels from the baseline and variant binaries must be exactly equal for every benchmark process in Step 4.

## Step 4: Performance measurement

### 4a. Interleaved timing (H2)

- Use the Phase 1 runner, copied to `artifacts/cubecl_kernels_perf/phase2/run_alternating.py` with Phase 2 paths, and the native command form `<binary> <input.f32le> <rows> <features> --backend {cpu|cuda} --runs 3 --min-samples <k>`.
- Matrix:
  - {20k×16, 50k×16} × k ∈ {5, 16, 17} × {cpu, cuda}. The k=17 cells check that the heap path did not regress; they should be neutral.
  - {20k×30, 50k×30} × k5 × {cpu, cuda}, preserving all eight inherited H2 k5 cells.
  - Confirm that `--min-samples m` yields neighbor width `k = m` on the KD route (`estimator.rs:285`, `k = min_samples.min(n − 1)`) before labeling cells by k. If the width differs, label cells by the actual width.
- Order: `B,V,V,B,B,V` per cell, cells serial, the same inputs and hashes as Phase 1.
- Record core-stage and neighbor-scope medians and [min–max], the full-fit median, and label equality.

### 4b. CUDA counters (H3)

Use the Phase 1 extension of `analyze/profile_hdbscan_counters.py` at 20k and 50k, k ∈ {5, 16}, for both binaries. Required readings:

- **Registers/thread.** Expect growth of about `2·k` (plus carries). At k=16, more than about 128 would cut occupancy at the 32-thread geometry (16 blocks/SM × 32 threads = 512 threads ⇒ 128 registers each on 64K).
- **Local-memory load/store sectors.** These must not increase versus the baseline. An increase requires generated-code inspection: check list scalarization, unrolling, dynamic indices and spills, while distinguishing the traversal stack's local arrays from list storage.
- Global load/store requests (should drop), long-scoreboard share, and achieved occupancy.

### 4c. JIT/cold-start cost

For k ∈ {5, 16}, on CPU and CUDA, measure the first fit in a fresh process (including JIT) for both binaries using the profiler's existing `warmup_seconds` field. Even `--runs 1` performs an excluded warmup before the measured fit: `fit_seconds[0]` is not cold-start timing. Record cache conditions consistently. Report the first-fit overhead for a new `k`, not isolated compilation time. It must stay small relative to one fit at 20k, otherwise reconsider the variant scheme (for example, buckets with masked updates).

## Step 5: Decision rule

Accept Phase 2 when:

1. Every Step 3 gate passes and labels are equal end-to-end.
2. The core-stage median improves at both 20k×16 and 50k×16 for k=5 on at least one backend, with range separation (the variant's max is below the baseline's median in at least one of those cells).
3. k=16 does not regress on either backend beyond the baseline spread, and local-memory traffic does not increase.
4. k=17 (heap path) does not regress beyond the baseline spread.
5. The cold-start cost per new `k` is reported and judged acceptable.
6. Every mandatory Step 4a cell, including both 30D sizes and both backends at k5, satisfies `variant core-stage median − baseline core-stage median ≤ baseline max − baseline min`. Compute medians and ranges over process medians, as in Phase 1. This both-backend non-regression bound is independent of the winning-backend requirement in (2).

**Threshold adjustment:** if k=5 wins but k=16 regresses (registers or spills), lower `KD_REGISTER_TOPK_MAX` to the largest measured `k` that does not regress, then re-run the affected cells. Do not add new mechanisms.

If any mandatory acceptance condition fails, do not retain the kernel change. A threshold revision may be retained only after remeasuring all affected mandatory timing, counter, cold-start and correctness cells and satisfying every gate. Otherwise revert only this phase's kernel change, keep the extended harness, and record the negative result with its numbers in `hdbscan_perf_report.md` before proceeding to Phase 3.

## Step 6: Deliverables

- The kernel change (helpers or a separate kernel, chosen per 2a), host dispatch and its unit test, and the extended harness.
- `artifacts/cubecl_kernels_perf/phase2/` containing provenance, binaries plus hashes, logs, alternation records, ncu CSVs, cold-start measurements, and a `summary.json` with the per-cell results and the decision.
- A "Phase 2 results" subsection appended to `hdbscan_perf_report.md`, scoped to the measured cells and backends.
- `cubecl_kernels_perf_plan.md` Phase 2 updated: the exact-comptime-`k` design, the final threshold and the status.
- No `AGENTS.md`/`README.md` change: the API, layout and semantics are unchanged.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| List not scalarized (local memory), so slower than the heap | ncu local-memory sectors must not rise; all indices comptime under `#[unroll]`; inspect generated code if the counters rise |
| Register growth at k=16 lowers occupancy or spills | Register/spill readings; lower the threshold per the decision rule |
| Off-by-one in the carry pass (wrong eviction or ordering) | Exhaustive `k` 1..=17 bit-equality plus diagnostics equality |
| `(inf, j)` vs sentinel ordering differs from the heap | Overflow-input cases at k ∈ {1, 5, 16} |
| CubeCL frontend limits (`&mut Array` in helpers, length-1 placeholder, `select` types) | Fallback per 2a/2b: separate kernel or `if`-swap; decided by compiling |
| JIT cost of per-`k` variants hurts cold starts or test time | Step 4c measurement; reduced CPU sweep sizes; bucketing only if the cost is unacceptable |
| Comptime branch makes the traversal harder to maintain | Branch confined to three helpers; traversal body single-sourced |
| Gain within noise (heap not dominant per the report's k-sensitivity) | Range-separation rule; record negative results; do not stack under Phase 3 |
