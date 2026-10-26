# Phase 3 detailed plan: padded vector data and register-resident query in `kd_neighbors_kernel`

This document details Phase 3 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md). It depends on the Phase 1 harness (`cubecl_kernels_perf_plan_phase1.md`, Step 1). It does not depend on Phase 1 or Phase 2 being accepted; it applies on top of whichever top-k container is in production. The source analysis is in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups", F1.

**Status:** plan only. Nothing here has been compiled, run or measured. Line numbers refer to revision `2c3fc35fa2b2492f519c0adcb544779357f46dd8`; Phases 1–2 will shift them, so re-read before editing.

## Objective

Cut the load instructions in the two inner loops of the KD neighbor search while keeping every arithmetic operation, and its order, unchanged:

- **Point distances** (`kdtree.rs:274–280`): today each candidate issues `2·d` scalar loads. The query row `data[row*d + f]` is reloaded for every candidate, and the candidate row is read with 4-byte loads at an arbitrary alignment.
- **Box lower bounds** (`box_lower_bound`, `:19–39`, called at `:252, 336, 337`): today each call issues `3·d` scalar loads, again including the query.

After this phase:

- The query is loaded once per row into registers.
- Candidate rows and node bounds are read as 16-byte `Vector<f32, Const<4>>` loads from zero-padded copies.

| Per-visit load instructions | Today | Phase 3 (`c = ⌈d/4⌉`) | 16D | 30D |
| --- | --- | --- | --- | --- |
| Candidate distance | `2d` scalar | `c` vector | 32 → 4 | 60 → 8 |
| One box bound | `3d` scalar | `2c` vector | 48 → 8 | 90 → 16 |

These are instruction counts from source reading, not measured traffic. The query loads mostly hit L1 today, so the expected gain comes from:

- fewer issued load instructions;
- fewer dependent load chains, which matter because the report measured 70–79% long-scoreboard stalls on CUDA;
- the scan's share of the core stage: 74% at 16D and 96.5% at 30D.

The size of the gain is a hypothesis to measure. On CPU, LLVM may already keep the query in registers after unrolling, so the CPU result may be smaller.

## Changes from the master plan

1. **Pack per call inside `exact_neighbors_kd`, not in `DeviceKdTree`.** The master plan proposed adding the padded buffers to `DeviceKdTree`. Source reading argues against that:
   - `DeviceKdTree` is public (`lib.rs:8`). Integration tests build it directly (`tests/test_kdtree_search.rs:27`, `tests/test_component_rounds.rs:18`), and it is a fusion `ExtensionType`. Adding fields changes the public API and every constructor.
   - The padded data is derived from `data`, which `exact_neighbors_kd` already receives separately. Storing a padded copy in the tree would create a second source of truth that callers could desynchronize.
   - `core_distances` calls `exact_neighbors_kd` once per fit (`estimator.rs:405`). A per-call pack is a single memory-bound pass of `O(n·d + nodes·d)`, against a search that scans many leaf candidates per query. It is measured explicitly in Step 4.
   - Phases 4, 6 and 7 change or add layouts. With a single pack site inside the `CubeBackend` implementation, each of those is a local change.

   Revisit this only if Borůvka (`boruvka_candidates_kd`, called once per round) adopts the padded layout. Repeated per-round packing would then justify a persistent buffer.
2. **One layout for every feature count.** The kernel reads only the padded vector tensors. The register-resident query applies when `d ≤ KD_REGISTER_QUERY_MAX_FEATURES = 32` (8 chunks), which covers the whole Auto route (`d ≤ 30`, `estimator.rs:127`). An explicit `HdbscanSearch::KdTree` with `d > 32` (`estimator.rs:131`) uses the same kernel with a runtime chunk loop and the query read from global memory. This mirrors Phase 2's `reg_k = 0` convention. The old scalar `data`/`bounds_min`/`bounds_max` arguments leave `kd_neighbors_kernel` entirely.

Update the master plan's Phase 3 text when this phase completes.

## Non-goals

- `kd_candidates_kernel` (Borůvka) and the scalar `box_lower_bound` it uses. Both stay unchanged.
- The public `DeviceKdTree`, the `HdbscanOps::exact_neighbors_kd` signature, and its fusion metadata.
- Node-record compaction and bound interleaving (Phase 4), launch geometry (Phase 5), and CPU SoA or plane kernels (Phases 6–7).
- Any reordering of the per-pair feature sum. There are no horizontal vector sums and no lane-wise accumulation.

## Invariants

1. Output IDs and `f32::to_bits` distances are bit-identical to the Phase 1 reference kernel for every `d` and `k`.
2. Per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) are identical to the reference kernel. Bounds are bit-identical, so every pruning and child-ordering decision is unchanged.
3. **Per-lane expressions are unchanged.** For each feature `f` in increasing order:
   - distance: `delta = q_f − r_f; distance += delta * delta`, with the query first, as at `:277–279`;
   - box bound: `delta = max(max(min_f − q_f, q_f − max_f), 0); sum += delta * delta`, as at `:34–35`.

   Vectors serve only as load containers. The arithmetic is scalar, on lanes taken with `extract(lane)` (`#[comptime]` lane index), in feature order.
4. **Padding sits at the tail and is neutral.** Lanes `d..4c` of the last chunk hold `+0.0` in the data, query, min and max copies. Their terms come after every real term:
   - distance pad term: `delta = 0 − 0 = +0`, so the term is `+0·+0 = +0`;
   - box pad term: `delta = max(max(+0, +0), +0) = +0`, so the term is `+0`.

   Each sum starts at `+0.0` and adds only terms that are `≥ +0` or `+inf`, so it is never `−0`. Then `s + (+0) = s` bit-exactly, including for `s = +inf`. If the compiler contracts the multiply-add, `fma(+0, +0, s) = s` exactly as well. NaN cannot arise: inputs are validated finite.
5. The pack copies `f32` values without arithmetic. Its source indexing (`data[row*d + f]`, `bounds_min[node*d + f]`) is identical to the current kernel's, so it inherits exactly the addressing the current kernel relies on.

## Equivalence argument

For each `(query, candidate)` pair, the reference computes `s_0 = +0` and `s_{f+1} = s_f + δ_f²` for `f = 0..d−1`, with `δ_f = data[q][f] − data[c][f]`. The variant reads the same `f32` values: the register query and the padded copies are bit copies. It evaluates the same expression for `f = 0..d−1` in the same order, then adds `4c − d` pad terms, which leave `s` unchanged by Invariant 4. The same holds for each box bound.

All distances and bounds are therefore bit-identical, so every comparison, push and admission is identical, and by induction over the traversal, so are the outputs and diagnostics.

The one gap in this argument is FMA contraction. Whether the compiler fuses `s + δ*δ` is a codegen decision, and the surrounding IR changes (vector loads, extracts). Matching source expressions make identical contraction likely, but do not guarantee it. The bit-equality harness at every tested `d`, on CPU and CUDA, is the gate. Step 3 says what to do on a mismatch.

## Step 0: Baseline and prerequisites

1. The Phase 1 harness is on the branch and green.
2. Preserve the current production binary, with the Phase 1–2 decisions applied, as `artifacts/cubecl_kernels_perf/phase3/bin/profile_hdbscan_stages.baseline`, with its SHA-256 and provenance.
3. CubeCL 0.11.1 APIs used. These were checked in the registry source (`cubecl-core-0.11.1/src/frontend`) and the repo; none has been compiled in this kernel yet:
   - `Vector<P: Scalar, N: Size>` (`container/vector/base.rs:13`). `Float: Numeric: Scalar` (`element/float.rs:19`, `element/numeric.rs:14`), so `Vector<F, Const<4>>` type-checks in principle.
   - `Vector::extract(self, #[comptime] index: usize)` (`base.rs:93`).
   - The `#[unroll]` loop index is accepted as a comptime lane index. Precedent: `component_reduce.rs:53–81` (`insert(lane, …)`).
   - `Array::<T: CubePrimitive>::new(#[comptime] length)`. `Vector` is a `CubePrimitive` (`base.rs:319`).
   - `Tensor<Vector<u32, Const<4>>>` kernel arguments, launched through a 1D `TensorArg::from_raw_parts(handle, [1], [vectors])` view. Precedent: `component_reduce.rs:539–543`, with CPU and CUDA probe tests at `:1078–1125`. These also provide evidence that runtime buffer bases are 16-byte aligned on both backends.

## Step 1: Extend the harness

In `kdtree.rs` `mod tests`, extend `check_kd_neighbors_match_reference`. Every case runs the production `launch_kd_neighbors` (pack plus dispatch) against `exact_neighbors_kd_reference`, which keeps the old scalar layout. Compare IDs, `to_bits` distances and per-row diagnostics.

| Case | Purpose |
| --- | --- |
| n = 2,000; d ∈ {1, 2, 3, 4, 5, 8, 16, 29, 30, 32}; k ∈ {1, 5, 16, 17} | every register chunk count 1..=8; tails of 3/2/1/0 pad lanes; the 32-feature edge; both top-k paths if Phase 2 is accepted |
| n = 2,000; d ∈ {33, 40}; k = 5 | non-register path (`reg_chunks = 0`), runtime chunk loop including a padded tail |
| duplicate-heavy (50 distinct points × 40), d ∈ {3, 16}, k ∈ {5, 17} | ties under the padded layout |
| integer grid, d ∈ {2, 3}, k = 5 | exact ties with pad lanes present |
| overflow input (values near `f32::MAX`, as in `check_overflow_distances`), d ∈ {1, 2, 3, 5}, k ∈ {1, 5} | `+inf` partial sums followed by pad terms (Invariant 4) |
| n = 257, d = 16, k = 5; small n with k > n − 1 | partial last cube; sentinel slots |

Further tests:

- **Pack unit test.** For `rows × d` with d ∈ {1, 4, 5, 30, 33}, read back the padded buffer. Each real lane must equal its source `to_bits`, and each pad lane must equal `0x0000_0000` exactly (`+0`, not `−0`). Run the same test for the bounds pack.
- **Dispatch unit test.** `reg_chunks(d)` returns `⌈d/4⌉` for `1 ≤ d ≤ 32`, and `0` for `d > 32`.
- **Memory-plan unit test.** Extend `core_search_memory_accounts_for_bounded_and_full_routes` (`estimator.rs:1436`) to assert the added KD pack bytes (Step 2f).
- **Integration tests.** The existing `tests/test_kdtree_search.rs` cases, including the host-oracle test at d ∈ {16, 30} from Phase 1 Step 1c, must pass unchanged. They exercise the public entry point, so they also cover the pack.

## Step 2: Design

### 2a. Pack kernel and host helper

```rust
#[cube(launch)]
fn pad_rows_kernel<F: Float>(
    source: &Tensor<F>,          // [rows, d], read as source[row * d + f]
    padded: &mut Tensor<F>,      // 1D scalar view, rows * 4c elements
    rows: u32, d: u32, padded_width: u32,
    #[define(F)] _dtype: ElemType,
);
```

- One unit handles one element of the padded buffer, in a grid-stride loop. It writes `source[row*d + f]` for `f < d`, otherwise `F::new(0.0)`.
- Host helper: `fn pad_rows(source: &CubeTensor, rows: usize, d: usize) -> CubeTensor`.
  - It allocates a contiguous 1D `[rows * 4c]` f32 buffer through `client.empty`, the same pattern as `allocate_diagnostics` (`:645–663`). A 1D shape avoids CUDA row pitch.
  - It launches the kernel with `kd_launch_geometry`-style legal dims.
- `exact_neighbors_kd`, after `validate_tree`, calls `pad_rows` three times: once for `data` (`n` rows) and once each for `bounds_min` and `bounds_max` (`nodes` rows).
- All work is on the same client and stream as the search launch, so no extra synchronization is needed.

### 2b. Vector views

Pass the three padded buffers as `Tensor<Vector<F, Const<4>>>` through `TensorArg::from_raw_parts(handle, [1].into(), [rows * c].into())`. Copy the `SAFETY` reasoning from `component_reduce.rs:539–540`: each contiguous buffer has exactly `vectors × 4` scalar slots, and the view uses unit vector stride. Index the row chunk as `padded[row * c + chunk]`.

Inside the kernel, the row count comes from scalar arguments `row_count`/`n` and `d_chunks: u32`, never from the shapes of the vector views.

### 2c. Kernel signature

```rust
fn kd_neighbors_kernel<F: Float, I: Int>(
    data_v: &Tensor<Vector<F, Const<4>>>,
    indices, starts, ends, is_leaf,                     // unchanged
    bounds_min_v: &Tensor<Vector<F, Const<4>>>,
    bounds_max_v: &Tensor<Vector<F, Const<4>>>,
    row_count: u32, n: u32, k: u32, d_chunks: u32,
    #[comptime] reg_chunks: usize,                      // 1..=8, or 0 = runtime path
    /* Phase 2 #[comptime] reg_k if accepted */
    #[comptime] diagnostics_enabled: bool,
    out_indices, out_distances, diagnostics,
    #[define(F, I)] _dtypes: [ElemType; 2],
)
```

`n` becomes an explicit argument because `data.shape(0)` is no longer available. The check `local_row < data.shape(0)` at `:241` becomes `local_row < n`.

### 2d. Helpers

Confine the comptime branch to three small `#[cube]` helpers, as in Phase 2. The traversal body stays single-sourced.

```rust
#[cube] fn load_query<F: Float>(data_v, row, query: &mut Array<Vector<F, Const<4>>>, #[comptime] reg_chunks: usize);
#[cube] fn point_distance<F: Float>(data_v, query: &Array<…>, row, candidate, d_chunks, #[comptime] reg_chunks: usize) -> F;
#[cube] fn padded_box_lower_bound<F: Float>(bounds_min_v, bounds_max_v, data_v, query: &Array<…>, row, node, d_chunks, #[comptime] reg_chunks: usize) -> F;
```

- **Register path (`reg_chunks > 0`).**
  - `query` is `Array::new(reg_chunks)`. `load_query` fills it under `#[unroll] for c in 0..reg_chunks`.
  - `point_distance` first issues all candidate chunk loads into a local `Array` of length `reg_chunks`, so loads are independent and issued before any arithmetic. Then, under `#[unroll] for c` and `#[unroll] for lane in 0..4`, it does `let delta = query[c].extract(lane) - cand[c].extract(lane); distance += delta * delta;`.
  - `padded_box_lower_bound` loads `min`/`max` chunks the same way and applies Invariant 3's expression per lane.
  - Every array index and lane is comptime, so the arrays should scalarize. Step 4b checks this.
- **Runtime path (`reg_chunks == 0`).**
  - `query` is a length-1 placeholder and is never read.
  - The helpers loop `for c in 0..d_chunks` at runtime, load `data_v[row * d_chunks + c]` alongside the candidate or bound chunk, and run the same unrolled 4-lane body.
- **Order.** In both paths, all 4 lanes of every chunk are summed, so the order is real features `0..d−1` followed by the tail pads (Invariant 4).
- **Fallback.** The pinned frontend might reject `&mut Array<Vector<…>>` in a helper, an `Array` of `Vector`, or the length-1 placeholder. In that case:
  - first fallback: hold the query as `Array<F>` of `4·reg_chunks` scalars, filled from extracted lanes;
  - second fallback: a separate register-path kernel that shares the top-k helpers.

  Compiling decides between these, not the plan.

### 2e. Traversal edits

- Call `load_query` once per row, after the top-k initialization.
- Replace the root bound at `:252` and the child bounds at `:336–337` with `padded_box_lower_bound`.
- Replace the inner feature loop at `:274–280` with `point_distance`.
- Pruning, child ordering, pushes, top-k admission and diagnostics are unchanged.
- The scalar `box_lower_bound` and `box_lower_bound_test_kernel` stay for `kd_candidates_kernel`.

### 2f. Host dispatch and resource sizing

- In `launch_kd_neighbors`, compute `d_chunks = d.div_ceil(4)` and set `reg_chunks = if d <= 32 { d_chunks } else { 0 }`. Put this in one small function for the dispatch test, with named constants `KD_VECTOR_LANES = 4` and `KD_REGISTER_QUERY_MAX_FEATURES = 32`.
- The Phase 1 reference path keeps the old scalar launch and does not pack.
- **Sizing.** The pack adds `(n + 2·nodes) · 4c · 4` live device bytes during the core stage. For example, 100k rows at 30 features gives `c = 8`, so `n · 128 B = 12.8 MB`, plus the bounds.
  - Compute these bytes next to `tree_bytes` (`estimator.rs:1071–1097`) with checked arithmetic.
  - Add a parameter for them to `core_search_memory` (`:277`). The KD branch at `:310–313` adds them.
  - Add three `check_device_allocation` calls next to the existing tree checks (`:1111–1141`). The buffers are 1D, so they are not row-pitched.
  - The overflow subjects follow the existing wording, for example `"KD-tree padded data allocation"`.

## Step 3: Correctness verification

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets
cargo test -p fast-hdbscan-core
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib kdtree -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --lib estimator -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --test test_kdtree_search -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --features tracing \
  --example profile_hdbscan_stages -- --nocapture
```

**Gate.** Everything passes on CPU and CUDA with no hardware skips. Every Step 1 case shows identical outputs **and** diagnostics. Logs go to `artifacts/cubecl_kernels_perf/phase3/logs/`.

**On a bit mismatch:**
1. Locate the first differing pair and feature count.
2. If the mismatch appears only with `d % 4 ≠ 0`, suspect the pack or the pad order.
3. If it appears at every `d` on one backend, suspect FMA contraction. Compare the generated code of the reference and variant kernels. CubeCL can print the CUDA source; for CPU, use the LLVM IR dump, if the runtime exposes one.
4. Fix it by restoring the reference's contraction behavior in the variant. The reference kernel and the tests are never changed to match.

**End to end.** Final labels from the baseline and variant binaries must be exactly equal for every benchmark process in Step 4.

## Step 4: Performance measurement

### 4a. Interleaved timing (H2)

- Copy the Phase 1 runner to `artifacts/cubecl_kernels_perf/phase3/run_alternating.py` and use the native command form `<binary> <input.f32le> <rows> <features> --backend {cpu|cuda} --runs 3 --min-samples 5`.
- Matrix, each cell on {cpu, cuda}:
  - 20k×16 and 50k×16: `baseline_{20000,50000}/raw/input.f32le`. These are the primary cells.
  - 20k×30: `artifacts/hdbscan_core_investigation/dim30.f32le` (20k rows). This is the widest register query.
  - 20k×8 and 20k×2: `dim8.f32le`, `dim2.f32le` (20k rows each). These are regression guards: at d = 2, half the lane arithmetic is padding, and the vector load is 16 B instead of 8 B. Low-dimensional HDBSCAN, such as clustering a UMAP embedding, is a common use case.
  - If Phase 2 is accepted: 20k×30 at `--min-samples 16`, the register-budget corner (Step 4b).
- Order: `B,V,V,B,B,V` per cell, cells serial, with input hashes recorded.
- Record core-stage and neighbor-scope medians and [min–max], the full-fit median, and label equality. Pack time is inside the measured neighbor scope, so every reported gain is net of packing.

### 4b. CUDA counters (H3)

Run the Phase 1 extension of `analyze/profile_hdbscan_counters.py` on 20k×16, 20k×30 and 20k×2 for both binaries. Required readings:

- **Registers per thread.** Expect up to `4c` more (32 at 30D), plus the Phase 2 list if accepted. Above about 128, occupancy drops at the 32-thread geometry (16 blocks/SM × 32 threads ⇒ 128 registers each on 64K).
- **Local-memory load and store sectors.** These must not increase. An increase means the query or candidate array did not scalarize.
- **Global load requests and sectors per request.** Requests should drop roughly per the Objective table.
- **Long-scoreboard stall share and achieved occupancy.**
- **Pack kernel duration** (nsys/ncu kernel time), reported separately as a share of the core stage. On CPU it is only visible inside the neighbor scope; that is acceptable, because gains are already net of it.

### 4c. JIT/cold-start cost

Measure the first fit in a fresh process for d ∈ {2, 16, 30}, on CPU and CUDA, for both binaries. Report the extra compile time for the new `reg_chunks` variant plus the pack kernel. A fit compiles one `(reg_chunks, reg_k)` combination.

## Step 5: Decision rule

Accept Phase 3 when:

1. Every Step 3 gate passes and labels are equal end-to-end.
2. The core-stage median improves at 20k×16, 50k×16 and 20k×30 on at least one backend, with range separation: the variant's max is below the baseline's median in at least two of those three cells.
3. 20k×2 and 20k×8 do not regress on either backend beyond the baseline spread.
4. Local-memory sectors do not increase in any profiled cell.
5. Cold-start cost is reported and judged acceptable.

**If (2) holds but (3) fails at d = 2,** the wasted pad lanes are the likely cause. Switch the register path from comptime `reg_chunks` to an exact comptime feature count, so the unrolled loop skips pad lanes: `f / 4` and `f % 4` must then be comptime, which needs a compile check. Re-run the affected cells. Do not add a second layout.

**If (4) fails,** fix scalarization first. This is a defect, not a tuning result.

If (2) fails on both backends, revert the kernel, pack and sizing changes, but keep the extended harness. Record the negative result with its numbers in `hdbscan_perf_report.md`, then proceed to Phase 4. Phase 4 must then plan its bounds layout without a pack site.

## Step 6: Deliverables

- The pack kernel and helper, the kernel and helper changes, dispatch and its unit test, the memory-plan change and its test, and the extended harness.
- `artifacts/cubecl_kernels_perf/phase3/` containing provenance, binaries plus hashes, logs, alternation records, ncu/nsys CSVs, cold-start measurements, and a `summary.json` with the per-cell results and the decision.
- A "Phase 3 results" subsection appended to `hdbscan_perf_report.md`, scoped to the measured cells and backends.
- `cubecl_kernels_perf_plan.md` Phase 3 updated with the per-call pack, the unified layout, the final register threshold, and the status. Phase 4's text must then say that interleaved bounds are produced by the same pack site, not stored in `DeviceKdTree`.
- No `AGENTS.md`/`README.md` change: the public API, `DeviceKdTree` and the semantics are unchanged, and neither document enumerates KD device buffers. Re-check both with a search at completion.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| FMA contraction differs between the reference and variant | Bit-equality at every tested `d` on both backends; Step 3 mismatch procedure; never relax the test |
| Pad lanes not `+0`, or pads not at the tail | Pack unit test checks exact bits; odd-`d` and overflow cases in the harness |
| Query or candidate arrays spill to local memory | ncu local-memory sectors; comptime indices only; scalar-array fallback (2d) |
| Register growth at 30D (plus Phase 2's list) lowers occupancy | 4b readings at the 20k×30 corner; Phase 5 geometry sweep; `KD_REGISTER_QUERY_MAX_FEATURES` can be lowered, with the runtime path covering the rest |
| d = 2 regresses (pad waste, wider loads) | Dedicated 20k×2 cell; exact-feature-count variant per Step 5 |
| Vector view misuse (alignment, length) | Precedent from `component_reduce` views and probes; lengths from scalar arguments only; harness covers n = 257 and small n |
| Pack cost eats the gain on small inputs | Gains measured net of pack; pack duration reported separately on CUDA |
| Memory plan understates live bytes | Checked arithmetic plus a unit test for the added term; three allocation checks |
| CubeCL frontend limits (`Array<Vector>`, `&mut Array` in helpers, placeholder length) | Fallbacks in 2d, decided by compiling |
| Gain within noise (CPU LLVM may already hoist the query) | Range-separation rule; record negative results; do not stack under Phase 4 |
