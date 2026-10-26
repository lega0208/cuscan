# Phase 4 detailed plan: compact node records and interleaved node bounds in `kd_neighbors_kernel`

This document details Phase 4 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md) (F3 in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups"). It builds on the accepted Phase 3 production kernel (commit `3e885a7`): the per-call pack site in `launch_kd_neighbors`, the padded `Vector<f32, Const<4>>` views and the exact-comptime register query. Line numbers refer to `3e885a7`.

**Status:** planned; not implemented or measured.

## Objective

Reduce the number of separate global load instructions and distinct memory regions touched per **node visit** of the KD neighbor traversal, without changing any arithmetic.

Current per-visit node traffic (`crates/fast-hdbscan/src/kernels/kdtree.rs`), with `c = ⌈d/4⌉`:

| Event | Today (Phase 3) | Phase 4 |
| --- | --- | --- |
| Pop surviving `lower <= worst` (`:683–684`) | `is_leaf[node]` (4 B scalar) | `node_records[node]` (one 16 B vector) |
| Pop, leaf (`:685–686`) | `+ starts[node]`, `+ ends[node]` (two more dependent 4 B loads from two more arrays) | none extra (same record) |
| Internal node, two child bounds (`:756–775`) | `2c` vectors from `bounds_min_v` + `2c` from `bounds_max_v`: four regions in two arrays | `4c` vectors from one contiguous span of `node_bounds_v` (left `[min|max]`, then right `[min|max]`) |
| Root bound (`:663`) | `c` + `c` vectors, two arrays | `2c` contiguous vectors |

Leaf scan traffic (`indices[position]` and `c` candidate vectors per scanned position) is unchanged and dominates: each leaf holds roughly 40–80 points, and the 16D query scans 74% of all pairs. This phase is therefore **expected to be minor** (master plan). Its value is in the long-scoreboard chain at each pop (`is_leaf → starts/ends → indices → data`, now `record → indices → data`) and in locality of sibling bounds (16D: the two siblings' bounds become one contiguous 256 B span instead of four 64 B pieces in two arrays; the exact cache-line count depends on allocation alignment and is a hypothesis, not a guarantee). The gain is a hypothesis to measure; a negative result is an acceptable outcome.

## Changes from the master plan

1. **Record element type is `u32`, not `I`.** The record is `Vector<u32, Const<4>>` = `(start, end, is_leaf, 0)`. `u32` four-lane tensor views have compiled and run on CPU and CUDA (`component_reduce.rs:36–40, 365–366`, probe tests there). The kernel already converts with `u32::cast_from(starts[node])` (`:685–686`), so the pack performs exactly that conversion once; the kernel's semantics do not change.
2. **No `first_leaf` scalar.** The flag rides in an otherwise unused lane of a record that is loaded anyway, so replacing it buys no load and would add a host proof obligation. Source reading of the builder (`fast-hdbscan-core/src/kdtree.rs:52–128`) shows that builder trees only make the root an early leaf (`n_levels = 1` for `n ≤ 80`), but `DeviceKdTree` is public and `validate_tree` (`kdtree.rs:887–929`) does not check structure, so arbitrary flag patterns remain legal input. The `first_leaf` option is recorded as not pursued.
3. **No batched sibling-bound load.** Issuing all `4c` sibling vectors before arithmetic would hold `4c` vector registers (32 at 30D). Phase 3 measured 128 registers per thread at 20k×30, exactly the 32-thread budget, so this risks spills. The two `padded_box_lower_bound` calls stay sequential (each loads `2c` vectors); adjacency provides line locality only. A batched helper is a possible later experiment, not part of this phase.

## Non-goals

- `kd_candidates_kernel` (Borůvka), `box_lower_bound`, `box_lower_bound_test_kernel`, `kd_neighbors_reference_kernel`. They keep the scalar `DeviceKdTree` fields.
- The public `DeviceKdTree`, `HdbscanOps::exact_neighbors_kd`, fusion metadata, `upload_kd_tree`, and the host `KdTree` builder.
- Data packing (`pad_rows` for `data` is unchanged), register thresholds, top-k container, launch geometry (Phase 5), leaf SoA/plane kernels (Phases 6–7).
- Any change to traversal order, pruning, child ordering, admission, diagnostics, or the per-feature arithmetic.

## Invariants

1. Output IDs and `f32::to_bits` distances are bit-identical to the frozen reference kernel (`kd_neighbors_reference_kernel`), and therefore to Phase 3 production, for every case of the existing harness.
2. Per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) are identical.
3. **Same values, same arithmetic.** Every bound lane read through `node_bounds_v` is a bit copy of the value Phase 3 read through `bounds_min_v`/`bounds_max_v` at the same `(node, feature)`; pad lanes are `+0.0` (Phase 3 Invariant 4 unchanged). Every `start`/`end` read from a record equals `u32::cast_from(starts[node])`/`u32::cast_from(ends[node])`; the leaf test `record.extract(2) != 0` is true exactly when `is_leaf[node] != I::cast_from(0)`.
4. Only memory layout and load instructions change. The `chunk_box_bound`/`chunk_distance` expressions, their lane and chunk order, and the comptime `reg_features` dispatch are untouched.
5. Error contract preserved: `validate_tree` runs first; new buffers get checked sizing, are counted in `core_search_memory`, and get `check_device_allocation` calls.

## Equivalence argument

The pack kernels copy values without arithmetic (the record conversion is the same `u32::cast_from` the kernel performs today; the flag lane is a cast whose zero/non-zero status equals the `I` comparison because the cast of an `i32` to `u32` is bit-preserving). The traversal reads, at every pop and every child-bound evaluation, the same scalar values as Phase 3 and feeds them to the same expressions in the same order. All bounds, distances, comparisons, pushes and admissions are therefore identical by induction over the traversal. Unlike Phase 3, no new floating-point expression or vector extract pattern is introduced in the arithmetic, so FMA contraction risk is limited to the compiler re-scheduling around different address computations; the bit-equality harness on both backends remains the gate.

## Step 0: Baseline and prerequisites

1. Tree clean at `3e885a7` plus this plan; record `git rev-parse HEAD` and the working-tree diff hash in `artifacts/cubecl_kernels_perf/phase4/provenance.json`.
2. Build the Phase 3 production binary (`cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages`) and preserve it as `artifacts/cubecl_kernels_perf/phase4/bin/profile_hdbscan_stages.baseline` with SHA-256 and provenance (commit, `rustc -V`, `nvidia-smi` driver line). Reusing the Phase 3 `bin/` variant is allowed only if its SHA-256 matches a fresh build of `3e885a7`; otherwise rebuild.
3. CubeCL 0.11.1 APIs used, all with compiled precedent in this repository: `Tensor<Vector<u32, Const<4>>>` kernel arguments and `TensorArg::from_raw_parts(handle, [1], [vectors])` views (`component_reduce.rs:539–543`; `padded_vector_view`, `kdtree.rs:1176–1189`); `Vector::extract` with a comptime lane (`chunk_distance`, `:94–107`); writes of scalar elements into a 1D scalar tensor whose handle is later viewed as vectors (`pad_rows_kernel`, `:55–76`).

## Step 1: Harness

The existing `check_kd_neighbors_match_reference` (`kdtree.rs:1420–1564`) already compares production `launch_kd_neighbors` (pack plus dispatch) against `exact_neighbors_kd_reference` for IDs, distance bits and diagnostics across `d ∈ {1,2,3,4,5,8,12,16,20,24,28,29,30,32,33,40}`, `k ∈ 1..=17, 24, 30`, duplicates, grids, overflow inputs, `n ∈ {1, 4, 9, 257}`, on CPU and required CUDA. It covers the new layout without change, including root-only trees (`n ≤ 9`, `n_levels = 1`, root is a leaf) and the runtime chunk path (`d > 32`). Keep it unchanged.

Add:

1. **Record pack unit test** (`check_pack_node_records`, CPU and CUDA variants following `check_pad_rows`, `:1609–1681`). Upload hand-written `starts`, `ends`, `is_leaf` (`i32`) with `nodes = 7` and values including `0`, `i32::MAX`, a non-0/1 flag (`2`), and leaves at non-final levels. Read back `nodes · 4` `u32` values and assert lane 0 = `starts as u32`, lane 1 = `ends as u32`, lane 2 = `is_leaf as u32`, lane 3 = `0`, for every node.
2. **Interleaved bounds pack unit test** (`check_pack_node_bounds`, CPU and CUDA). For `nodes = 5` and `d ∈ {1, 4, 5, 30, 33}`, with distinct min/max values including `-0.0`, read back `nodes · 2 · W` floats (`W = 4c`) and assert by `to_bits`: position `node·2W + f` = `min[node·d + f]`, `node·2W + W + f` = `max[node·d + f]` for `f < d`, and every pad lane in either half = `0x0000_0000`.
3. **Hand-built early-leaf tree equivalence.** One extra case driven through both `launch_kd_neighbors` and `exact_neighbors_kd_reference` with a `DeviceKdTree` built from `KdTree::build` on `n = 200, d = 3` (3 levels) and then edited on the host so that one internal node at level 1 is flagged as a leaf (its range still covers its subtree's points; bounds unchanged). Assert identical IDs, distance bits and diagnostics. This exercises a leaf flag outside the last level, which builder trees never produce, so the flag lane is tested as data, not assumed.
4. **Direct-launch test update.** `kd_neighbors_enabled_diagnostics_match_disabled_launch` (`kdtree.rs:1704–1830`) launches `kd_neighbors_kernel` directly (`:1773–1797`) with the five arguments this phase removes, after `pad_rows` on the bounds (`:1767–1768`). It is the only launch site besides production (`:1068`). Pack once with `pack_node_records`/`pack_node_bounds` outside the launch closure, pass the two new views, and keep its enabled/disabled comparison unchanged. Fold the bounds sub-block of `check_pad_rows` (`:1638–1659`) into `check_pack_node_bounds`, since `pad_rows` no longer packs bounds.
5. **Record flag semantics in traversal.** Extend the record pack test with a negative flag (`-1`), and in the hand-built early-leaf case (item 3) set the edited leaf's flag to `2` and one other leaf's to `-1`, so the zero/non-zero argument is exercised by the traversal, not only by the pack.
6. **Sizing unit tests.** Update `kd_padded_bytes_counts_data_and_both_bounds` (`:1603–1607`) to the renamed function and new formula (Step 2d), including an overflow case on the record term, and update `core_search_memory_accounts_for_bounded_and_full_routes` (`estimator.rs:1467–1471`).
7. The integration tests (`tests/test_kdtree_search.rs`, host-oracle at d ∈ {16, 30}) must pass unchanged; the lib suite passes with only the updates listed above.

## Step 2: Design

### 2a. Pack kernels

```rust
#[cube(launch)]
fn pack_node_records_kernel<I: Int>(
    starts: &Tensor<I>, ends: &Tensor<I>, is_leaf: &Tensor<I>,
    records: &mut Tensor<u32>,          // 1D scalar view, nodes * 4
    nodes: u32,
    #[define(I)] _dtype: ElemType,
);

#[cube(launch)]
fn pack_node_bounds_kernel<F: Float>(
    bounds_min: &Tensor<F>, bounds_max: &Tensor<F>,   // [nodes, d], read as [node * d + f]
    packed: &mut Tensor<F>,             // 1D scalar view, nodes * 2 * padded_width
    nodes: u32, d: u32, padded_width: u32,
    #[define(F)] _dtype: ElemType,
);
```

- Both use the `pad_rows_kernel` grid-stride pattern (`:63–75`), one unit per output scalar.
- Records: `slot = position % 4`; lanes 0/1/2 write `u32::cast_from(starts|ends|is_leaf[node])`; lane 3 writes `0`.
- Bounds: `half = (position / padded_width) % 2`, `node = position / (2·padded_width)`, `f = position % padded_width`; write `bounds_min`/`bounds_max[node·d + f]` for `f < d`, else `F::new(0.0)`. Source indexing equals the current pack's, so it inherits the addressing the current kernel relies on (contiguous `[nodes, d]` after `into_contiguous_tree`, `:931–943`).
- Host helpers `pack_node_records(tree) -> CubeTensor` (`DType::U32`, length `nodes·4`) and `pack_node_bounds(tree, nodes, d) -> CubeTensor` (`DType::F32`, length `nodes·2W`), allocated through `client.empty` like `allocate_float`, launched with `kd_launch_geometry(&client, len)`, `u32::try_from` on lengths, same client/stream as the search. If `allocate_float` has no `u32` counterpart, add a minimal `allocate_u32` next to it rather than reinterpreting an `i32` allocation.
- The two `pad_rows` calls for `bounds_min`/`bounds_max` (`:1065–1066`) are replaced by one `pack_node_bounds` call. `pad_rows` stays for `data`.

### 2b. Views

- `node_records`: `TensorArg::from_raw_parts(handle, [1], [nodes])` as `Tensor<Vector<u32, Const<4>>>`, with the `SAFETY` reasoning of `padded_vector_view` (fresh contiguous allocation of exactly `nodes·4` scalars, unit vector stride, all accesses bounded by validated node indices). Add a `record_vector_view(tensor, nodes)` helper asserting the `[nodes·4]` shape, analogous to `padded_vector_view`.
- `node_bounds_v`: `padded_vector_view(&node_bounds, 2 * nodes, d_chunks)` — the buffer is exactly `2·nodes` padded rows (min row, max row per node), so the existing helper and its length assertion apply unchanged.

### 2c. Kernel signature and traversal edits

```rust
fn kd_neighbors_kernel<F: Float, I: Int>(
    data_v: &Tensor<KdVector<F>>,
    indices: &Tensor<I>,
    node_records: &Tensor<Vector<u32, Const<4>>>,
    node_bounds_v: &Tensor<KdVector<F>>,
    row_count: u32, n: u32, k: u32, d_chunks: u32,
    #[comptime] reg_features: usize,
    #[comptime] diagnostics_enabled: bool,
    out_indices, out_distances, diagnostics,
    #[define(F, I)] _dtypes: [ElemType; 2],
)
```

- `starts`, `ends`, `is_leaf`, `bounds_min_v`, `bounds_max_v` leave the kernel argument list.
- At pop (`:684–686`): `let record = node_records[node]; if record.extract(2) != 0 { let start = record.extract(0) as usize; let end = record.extract(1) as usize; … }`. The record load happens only after the `lower <= worst_distance` test, exactly where `is_leaf` is read today, so pruned pops still issue no node load.
- `padded_box_lower_bound` (`:173–215`) takes `node_bounds_v` instead of `bounds_min_v`/`bounds_max_v`. Register path: `minima[chunk] = node_bounds_v[node·2c + chunk]`, `maxima[chunk] = node_bounds_v[node·2c + c + chunk]`. Runtime path (`d > 32`): the same indices inside the runtime `for chunk in 0..d_chunks` loop. Arithmetic and call sites (root `:663`, children `:756–775`) unchanged.
- If the frontend rejects `u32` lanes compared to a literal or cast to `usize`, use `u32::cast_from`-style explicit conversions; the compiler decides, not the plan.

### 2d. Sizing and allocation checks

- Rename `kd_padded_bytes` to `kd_pack_bytes(rows, nodes, features)` (callers: `estimator.rs:1101–1104`, `mod.rs:8`, tests) and compute, all checked:
  `(rows·W + 2·nodes·W)·4 + nodes·4·4`.
  The data and bound terms are unchanged in value (Phase 3 already counted `(n + 2·nodes)·W·4`); the record term is new. Update its doc comment and the comment at `estimator.rs:315`.
- `estimator.rs:1149–1167`: replace the two padded-bounds checks with one `check_device_allocation(nodes·2·W, 4, …, "KD-tree packed bounds allocation")`, keep `"KD-tree padded data allocation"`, and add `check_device_allocation(nodes·4, 4, …, "KD-tree node record allocation")`. Element counts use `checked_mul` with the subject as the overflow message.
- `core_search_memory` is unchanged in shape; its `kd_pack_bytes` argument now includes the record term. Update the unit-test expectation: 1024 rows, 63 nodes, 5 features → `(1024 + 126)·8·4 + 63·16 = 37_808`.

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
uv run --no-sync pytest analyze/tests/test_hdbscan_stages.py
```

**Gate.** All pass on CPU and CUDA with no hardware skips; every harness case has identical IDs, distance bits and diagnostics. Logs go to `artifacts/cubecl_kernels_perf/phase4/logs/`. On a mismatch, follow the Phase 3 Step 3 procedure; additionally, a mismatch in diagnostics or IDs with matching distances points at the record pack (offsets or flag lane), a mismatch in pruning counts at the bounds interleave offsets. The reference kernel and tests are never changed to match.

**End to end.** Final labels of baseline and variant binaries must be exactly equal for every benchmark process in Step 4.

## Step 4: Performance measurement

### 4a. Interleaved timing (H2)

- Copy `artifacts/cubecl_kernels_perf/phase3/run_alternating.py` to `artifacts/cubecl_kernels_perf/phase4/`. It currently writes one exclusive output directory and runs the whole matrix (`:31–37`). Allowed changes, and only these: binary/input path constants, a `--cells` selector, a `--batch` label, and a fresh per-batch output directory (still refusing to overwrite). Record the diff against the Phase 3 copy. Same command form: `<binary> <input.f32le> <rows> <features> --backend {cpu|cuda} --runs 3 --min-samples 5`.
- Cells, each on {cpu, cuda}, identical inputs (with recorded hashes) to Phase 3:
  - primary: 20k×16, 50k×16, 20k×30, 50k×30;
  - guards: 20k×8, 20k×2. At d = 2, `c = 1`, so the record load (16 B) replaces one 4 B flag load per internal pop while the bound span shrinks least; this is where a regression would show.
- Order `B,V,V,B,B,V` per cell, cells serial; medians are process medians as in Phase 1 Step 4a. Record core-stage and neighbor-scope medians and [min–max], full-fit median, and label equality. Pack time is inside the neighbor scope, so results are net of packing.
- **Fixed replication.** Because the expected effect is small and Phase 3 saw a non-replicating 20k×2 CUDA false regression (bimodal 1.5–2.8 ms; `hdbscan_perf_report.md` Phase 3 results), every cell on both backends runs **two independent batches** (batch A, then after the full matrix, batch B), predeclared, regardless of the first result. No further batches. Both batches are reported; neither is selected.
- **Aggregation.** For each cell and backend, pool the 3+3 process medians per arm from A and B (6 per arm). The cell effect is `Δ = median(V) / median(B) − 1` over the pooled values. Uncertainty is assessed at the interleaved-block level: each `B,V,V,B,B,V` batch yields three adjacent baseline/variant pairs (positions 1–2, 3–4, 5–6), each expressed as the ratio variant/baseline regardless of temporal order; the cell's six ratios are reported with their min–max. The 5-of-6 rule below is a practical consistency screen, not a formal significance test.

### 4b. CUDA counters (H3)

Copy `artifacts/cubecl_kernels_perf/phase3/profile_counters.py` (`:35–40` filters `kd_neighbors_kernel` and `pad_rows_kernel`). Allowed change: add `pack_node_records_kernel` and `pack_node_bounds_kernel` to the variant kernel filter; record the diff. Run it on 20k×16, 20k×30 and 20k×2 for both binaries. As in Phase 3, the NCU counters describe the first matching warmup launch of each kernel (single-launch, conditional readings), not whole-fit averages. Required readings for `kd_neighbors_kernel`:

- global load requests (`l1tex__t_requests_pipe_lsu_mem_global_op_ld`) and sectors per request (gated by Step 5 rule 4);
- registers per thread — report; must not exceed 128 at 30D;
- local-memory load/store sectors — must not increase;
- long-scoreboard stall share, achieved and theoretical occupancy;
- pack cost: total GPU time of all pack launches (`pad_rows_kernel` for data, both new pack kernels) and of `kd_neighbors_kernel`, taken from one nsys timeline of the same fit, reported as pack share of that fit's KD neighbor GPU time. Never divide a replayed NCU duration by an unrelated fit median. This is reporting-only, since H2 gains are already net of packing.

### 4c. JIT/cold-start

Measure first fit in a fresh process for d ∈ {2, 16, 30}, CPU and CUDA, both binaries, with the Phase 3 `cold_start.py` (copied, unchanged except paths). It measures whole warmup fits and does not clear compilation caches, so it cannot isolate per-kernel compile cost. The result is **reporting-only**: the script's warmup-fit medians, plus a first-fit-minus-warm-fit delta computed explicitly from the warmup and the H2 warm-fit samples of the same binary (sources recorded), variant against baseline, under the same cache state. It is not an acceptance gate.

## Step 5: Decision rule

All rules use the pooled two-batch statistics of Step 4a. Predeclared thresholds: **gain** means pooled `Δ ≤ −2%` and at least 5 of 6 paired ratios below 1; **regression** means pooled `Δ ≥ +3%` and at least 5 of 6 paired ratios above 1. Anything between is "no effect".

The following are the default acceptance criteria, not absolute performance vetoes. Correctness criterion (1) remains mandatory. For performance criteria (2)–(4), make a judgement call when the evidence supports a reasonable trade-off, especially when timings are potentially noisy. Consider both batches and paired ratios, absolute as well as relative timing differences, backend/workload scope, counter evidence and implementation/resource costs. A threshold crossing alone does not establish a meaningful benefit or harm, and "no effect" does not establish equivalence.

Accept Phase 4 by default when all hold:

1. Every Step 3 gate passes and labels are equal end to end.
2. On at least one backend, at least three of the four primary cells show a gain.
3. No primary or guard cell on either backend shows a regression.
4. In every profiled cell, `kd_neighbors_kernel` global load requests strictly decrease, local-memory sectors do not increase, and registers per thread do not exceed 128.

Cold-start (4c) and pack share (4b) are reported but do not gate.

**Failure handling.**
- Fail of (1): a correctness defect; fix it, never relax tests. If unfixable, reject.
- Fail of (4) by local-memory increase or registers above 128: investigate as a scalarization or spill defect first (comptime indices, record lane extraction). If unresolved, normally reject; retain only with an explicit, evidence-backed explanation of why the measured benefit warrants the resource cost. A register reading at 128 with no local-memory increase passes, and its occupancy interaction is deferred to the Phase 5 sweep. The Step 4b counter limits are subject to this same performance judgement, not independent absolute vetoes.
- Fail of (2) or (3), or (4) by load requests: normally reject, but assess whether noise or a reasonable workload/backend trade-off justifies retention. An inconclusive result ("no effect") is not an automatic rejection, nor a proven speedup. Use the fixed two-batch evidence; this discretion does not authorize extra batches until a favorable result appears or selection of favorable samples.
- **Document the judgement.** In `summary.json` and the Phase 4 results in `hdbscan_perf_report.md`, state accept/reject, each unmet criterion, supporting and contrary measurements (including variability and absolute/relative costs), the trade-off accepted and why it is reasonable. Distinguish measured gains from hypotheses or unresolved uncertainty. Without a defensible benefit, reject rather than stacking an unsupported optimization under Phase 5.
- **Rejection** reverts only Phase 4 production code: the kernel, pack kernels and helpers, views, sizing and allocation-check changes, and the tests that only exercise removed code. Unrelated working-tree changes are preserved; the Step 0 clean-tree requirement is not permission to discard user work. Keep the evidence directory, record the negative result with numbers in `hdbscan_perf_report.md`, and proceed to Phase 5 on the Phase 3 kernel.

## Step 6: Deliverables

- Pack kernels and helpers, kernel and helper changes, sizing changes, new and updated tests.
- `artifacts/cubecl_kernels_perf/phase4/`: provenance, binaries plus hashes, logs, alternation records, counter CSVs/JSON, cold-start measurements, `summary.json` with per-cell results and the decision.
- A "Phase 4 results" subsection in `hdbscan_perf_report.md`, scoped to measured cells and backends, including the negative result if rejected.
- `cubecl_kernels_perf_plan.md` Phase 4: status line, link to this plan, the `u32` record and no-`first_leaf` decisions, and `:81` corrected from "(`pad_rows`)" to `pack_node_bounds_kernel` if accepted.
- No `AGENTS.md`/`README.md` change expected (public API, `DeviceKdTree` and semantics unchanged; neither enumerates KD device buffers). Re-check with a search at completion.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Interleave offset or record lane error | Exact-bit pack unit tests; full equivalence harness compares diagnostics, which catch any pruning or leaf-range change |
| Leaf flag semantics drift (non-0/1 flags, early leaves) | Flag copied as data; pack test with flag `2`; hand-built early-leaf equivalence case |
| Record load wider than the flag load hurts low-d | 20k×2 guard cell and no-regression rule (3) |
| Codegen changes around new address math alter contraction | Bit-equality harness on both backends; arithmetic code untouched |
| Register growth at 30D from record lanes | H3 registers at 20k×30; rule (4) |
| Extra pack launch and memory | Gains net of packing; pack duration reported; sizing counted and checked |
| Effect within noise | Predeclared two-batch pooling with paired-ratio thresholds; inconclusive is not a proven gain or an automatic rejection; document the evidence-backed judgement under Step 5 |
| Guard-cell noise (Phase 3 20k×2 precedent) forces a false revert | Two fixed batches for every cell; regression screen requires both a +3% pooled effect and consistent paired ratios; weigh variability and absolute cost before deciding, and document any trade-off |
