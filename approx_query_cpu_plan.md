# CPU leaf-SoA kernel for HDBSCAN `approximate_predict` KD queries

Status: draft for architect/critic review. No code has changed.

## 1. Goal

Use the same leaf-tiled SoA candidate scan as fit's CPU KD KNN (C16, `kd_neighbors_cpu_kernel`) for the
CPU KD searches in `approximate_predict`. CUDA routing doesn't change.

Numerical contract (single, used everywhere in this plan): the SoA route is exact under its own
arithmetic. It returns the true k nearest by `(distance², reference ID)` with `(inf, n)` sentinels,
where distance² is computed by C16's per-feature summation. On inputs whose squared sums are exactly
representable (quantized fixtures) it is bitwise identical to today's `PaddedRows` route and to a host
oracle. On arbitrary inputs, FMA contraction may differ between the vector and scalar paths, so
distances may differ by rounding, and at near-ties neighbor IDs and therefore downstream
labels/probabilities/connectors may differ from the previous release. This is the same contract
CPU-vs-CUDA prediction already has. It is deliberately permitted and documented (§3.4, §7), not
hidden. Fit's CPU core distances already use C16 arithmetic, so CPU prediction's core phase now
matches fit's arithmetic.

Non-goals:
- CUDA query routing. F2 is exhaustive over the training set, which is the wrong trade for external
  query batches. CUDA stays on `kd_neighbors_kernel`.
- Changing fit's KNN path or the Borůvka paths.
- A persistent (cross-call) device reference. `CallReference` is still built per call and released on
  return (prediction.rs:1155).
- Changes to serialization or schema. The SoA pack is derived device state that only exists for the
  duration of a call.

## 2. Current state (verified in source)

- `HdbscanOps::prepare_kd_reference(data, tree) -> DeviceKdReference` (backend.rs:120-148) runs once
  per prediction call (`CallReference::upload`, prediction.rs:1163). It stores `padded_data`
  (`rows × 4⌈d/4⌉`), padded bounds, `indices/starts/ends/is_leaf`, and the sizes. Its fusion `meta`
  closure computes every output shape from `data`/`tree` specs alone. Fusion is enabled in the
  workspace (Cargo.toml:37).
- `exact_neighbors_kd_query(query, reference, k, exclude_self, query_offset)` →
  `launch_kd_query` (kdtree.rs:1953) asserts the fixed `kd_query_plan() == PaddedRows/Cyclic`
  (kdtree.rs:1825, 2000) and launches `kd_neighbors_kernel`.
- Prediction makes two kinds of query calls, both with `exclude_self = false`:
  - the cold core phase: reference rows `start..end` against the reference, width `s`
    (prediction.rs:1281);
  - the query phase: external queries, width `k = candidate_width(s, n)` (prediction.rs:1396).
- C16 (`kd_neighbors_cpu_kernel`, kdtree.rs:1306-1597):
  - Each unit handles `KD_QUERY_BLOCK = 64` consecutive positions in leaf order
    (`row = indices[local_row]`) and runs its own pruned traversal: it is not a shared block traversal.
  - Query values and box bounds come from `data_v` (padded rows), or from compact `data` above 32
    features.
  - Candidates come only from `soa` (`slots × d`, lane-tiled per `lanes` positions) and `slot_ids`
    (`indices[position]`, or `n` for pad slots).
  - Self exclusion is hard-wired as `candidate != row`.
  - `pack_leaf_soa` (kdtree.rs:2382) builds `soa/slot_ids` from padded data and `indices`.
  - Lanes come from `client.io_optimized_vector_sizes(4)` capped at `KD_SOA_MAX_LANES = 16`
    (kdtree.rs:58-61).
- Reference sizing: `kd_reference_pack_bytes(rows, nodes, features) = kd_padded_bytes`. It feeds
  `reference_device_bytes` (prediction.rs:507-586), which is used by `plan_prediction`, the
  core-only plan (prediction.rs:771) and `CallReference::upload`.
- Distance arithmetic: both kernels add squared differences one feature at a time, in order
  (`chunk_distance`, kdtree.rs:300-315; C16 loops, kdtree.rs:1383-1398). The tests still quantize
  inputs "so the host oracle is bit-comparable regardless of device contraction" (kdtree.rs:3181).
  So bitwise equality between the scalar and vector paths on arbitrary data is not guaranteed under
  LLVM FMA contraction. Exactness is proven on quantized data, as the existing tests already do.

## 3. Design

### 3.1 Query plan becomes per client

Replace `const fn kd_query_plan()` with `kd_query_plan(client, features) -> KdNeighborPlan`:
- CPU client (`num_cpu_cores.is_some()`): `LeafSoa { lanes: kd_cpu_soa_lanes(client) }`,
  `order: Cyclic`. Queries run in input order; see Phase B for leaf order.
- Otherwise: `PaddedRows / Cyclic`, unchanged.

Extract `kd_cpu_soa_lanes(client) -> usize` from `kd_neighbor_plan` (kdtree.rs:58-61) so that fit and
predict share the same lane selection. `features` is accepted so this has the same shape as
`kd_neighbor_plan` and stays a single routing point. Today it doesn't affect CPU routing.

### 3.2 The reference owns its candidate pack, keyed by layout

Change `DeviceKdReference`:

| field | `lanes == 0` (PaddedRows) | `lanes > 0` (LeafSoa) |
|---|---|---|
| `candidates: FloatTensor` (replaces `padded_data`) | padded rows, `rows × 4⌈d/4⌉` | SoA, `kd_soa_slots(rows, lanes) × d` |
| `slot_ids: IntTensor` (new) | `indices` (same handle, no allocation) | `kd_soa_slots(rows, lanes)` packed IDs, pad = `rows` |
| `lanes: usize` (new) | 0 | lanes |

Rationale:
- The SoA kernel never reads padded reference rows: it reads queries from the query buffer and
  candidates from the SoA. Keeping both would add `rows × 4⌈d/4⌉ × 4` bytes that are never read. With
  the replacement, the CPU candidate pack costs `slots × (d+1) × 4` instead of `rows × 4⌈d/4⌉ × 4`.
  That is not always smaller: lane rounding, the ID array and `d mod 4 == 0` can make it larger (for
  example `rows=17, lanes=16, d=3`: 512 bytes versus 272). Sizing always uses the exact formula. The
  justification is avoiding an unread allocation, not a guaranteed saving.
- `slot_ids == indices` in PaddedRows mode keeps every field a real tensor. There are no placeholder
  allocations and the struct doesn't need to be an enum. The extension-type derive and the fusion
  metadata stay plain structs.
- `padded_min` / `padded_max` stay. Both kernels use them.

`prepare_kd_reference(data, tree, lanes: u32)`:
- Gains an explicit `lanes` argument. The fusion `meta` closure has no client, so the caller chooses
  the layout and `meta` computes shapes from `lanes`:
  - `candidates`: `rows·width` if `lanes == 0`, else `slots·d`;
  - `slot_ids`: `tree.indices` spec if `lanes == 0`, else `[slots]` I32.
- Implementation:
  - `lanes == 0`: today's padding.
  - `lanes > 0`: assert the CPU client and that `lanes ∈ io_optimized_vector_sizes(4)`, as in
    kdtree.rs:2204-2211. Pad rows into a temporary, call `pack_leaf_soa`, and drop the temporary
    before returning. The temporary is a short-lived peak; see 3.5.
- `validate_reference` checks the shapes of `candidates` and `slot_ids` for each layout.

The caller (`CallReference::upload`) computes the plan once from
`device.as_dispatch() → DispatchDevice::Cube(d) → d.client()` (the pattern at estimator.rs:1701-1703)
and passes `lanes`. The same plan is threaded into sizing (3.5).

### 3.3 One CPU kernel for fit and query

Make `kd_neighbors_cpu_kernel` cover both cases instead of duplicating it:
- `data_v` → `query_v` (padded queries) and `data` → `query` (compact queries, used above 32
  features).
- New runtime `query_offset: u32`, comptime `leaf_order: bool` and comptime `exclude_self: bool`.
- `row = if leaf_order { indices[local_row] } else { local_row }`, with the guard
  `local_row < row_count`. The current guard `local_row < n` is only correct when
  `row_count == n`.
- Exclusion: `!exclude_self || candidate != query_offset + row`, the same rule as
  `kd_neighbors_kernel` (kdtree.rs:856, 907).
- The sentinel stays `n` (reference rows). Outputs and diagnostics are indexed by query `row`.

Callers:
- Fit (`launch_kd_neighbors_impl`, kdtree.rs:2213-2248) passes `query_v = data_v = padded data`,
  `query = data`, `leaf_order = true`, `exclude_self = true`, `query_offset = 0`. This is the same
  computation as before.
- Query (`launch_kd_query`) dispatches on `reference.lanes`:
  - 0: existing `kd_neighbors_kernel`.
  - `> 0`: the CPU kernel with the padded query, compact query, `reference.candidates` (via
    `soa_vector_view`), `reference.slot_ids`, tree arrays and bounds, `leaf_order = false`, and the
    caller's `exclude_self` / `query_offset`.
- Launch geometry is `kd_launch_geometry(&client, rows)`, as in fit. `preferred_width` and
  `max_cubes_override` keep working only for the PaddedRows path. The SoA path gets its own test hook
  to force a single cube and strided blocks: `max_cubes_override = Some(1)` goes through
  `kd_launch_geometry_with_width`.

### 3.4 Op documentation

- backend.rs:160-166 currently says "Always uses bounds-pruned per-query traversal". Change it to say
  that CPU uses per-query pruned traversal over a leaf-SoA candidate pack and other devices use padded
  rows, and that the results contract is unchanged.
- Update the `DeviceKdReference` docs (backend.rs:33-40) for the new fields.
- Document the deliberate Rust API break: `HdbscanOps` is publicly re-exported
  (lib.rs:9), and `prepare_kd_reference` gains `lanes` while `DeviceKdReference.padded_data` becomes
  `candidates` plus `slot_ids`/`lanes`. The op docs state that `lanes = 0` selects padded rows on any
  device, and that `lanes > 0` requires a CPU client and a width from
  `io_optimized_vector_sizes(4)` that is ≤ 16 (otherwise it panics, like the other precondition
  asserts). There is no shim. The Python API doesn't change.
- State the numerical contract from §1 in the op docs and in a README note on prediction.
- Update the AGENTS.md architecture paragraph: approximate_predict CPU KD queries use the C16 SoA
  scan; CUDA queries stay per-query.

### 3.5 Sizing and accounting (checked, before upload)

- `kd_reference_pack_bytes(plan, rows, nodes, features)`:
  - PaddedRows: unchanged.
  - LeafSoa: `2·nodes·width·4 + slots·(d+1)·4`.
- New `kd_reference_pack_peak_bytes(plan, rows, nodes, features)` for the transient prepare peak.
  While `pack_leaf_soa` runs, the temporary padded rows (`rows·width·4`) are live alongside the final
  pack.
- `reference_device_bytes` takes the plan:
  - `plan_prediction` and the core-only plan (prediction.rs:650, 771) use the steady pack bytes for
    the phase live sums. They add a separate "prediction reference prepare" live set
    (geometry + tree + peak pack), and `device_peak_bytes` takes the max over it too.
  - `CallReference::upload` validates the same way.
- Per-allocation checks match fit (estimator.rs:1245-1260): SoA elements and slots must fit `u32`
  and pass `check_device_allocation` against the limit. SoA tensors and the temporary padded rows are
  1-D, so `row_pitched` doesn't apply to them (as for the existing padded pack); only the 2-D compact
  geometry uses it.
- `kd_query_pack_bytes` (the per-batch query copy, outputs and diagnostics) doesn't change: the SoA
  query path still pads the query batch and reads the compact query, which is the input tensor and is
  already counted as `query_source`.
- `PredictionPlan.device_peak_bytes` changes on CPU. Update the expected values in its tests
  (prediction.rs:1998-2015 and the overflow tests at 2055-2060), deriving them from the CPU plan
  rather than hard-coding a lane width.
- `plan_prediction` gains a `KdNeighborPlan` parameter. Both preflights resolve it from the target:
  the full prediction preflight (prediction.rs:1314, reached by the `doc(hidden)`
  `FittedHDBSCAN::prediction_plan`, estimator.rs:1459, through the crate-private helper at
  prediction.rs:1327) and the conversion/core-only preflight (prediction.rs:1140). The test helpers pass an explicit plan so that both layouts are sized.

## 4. Phases

### Phase A: SoA query route (input order)

1. `kd_cpu_soa_lanes` and the per-client `kd_query_plan`. Update the route test (kdtree.rs:3169-3178):
   CPU → LeafSoa, CUDA → PaddedRows.
2. `DeviceKdReference` fields, `prepare_kd_reference(…, lanes)` in backend.rs (trait and fusion
   meta), the `CubeBackend` forwarding impl in kernels/mod.rs:73-77, the new helper re-exports in
   kernels/mod.rs:10-13 (`kd_query_plan`, `kd_reference_pack_peak_bytes`), and `validate_reference`.
3. Generalize the CPU kernel (3.3). Re-point fit's launch.
4. `launch_kd_query` dispatch.
5. Sizing (3.5) and threading through prediction.rs.
6. Docs (3.4).

### Phase B: optional leaf-ordered queries (separate gate; only if Phase A's evidence warrants it)

- **Core phase.** Its queries are reference rows, so their leaf order is already known (`indices`).
  Within a core batch `start..end`, the launcher could take a permutation instead of `leaf_order`.
  The batch rows are a contiguous compact-ID range, not a leaf range, so locality needs either the
  batch defined over leaf positions with a scatter on readback, or one call over all rows in leaf
  order.
- **Query phase.** Host-descend each query to its leaf (`KdTree` on the host), sort a permutation,
  upload the permuted batch and unpermute on readback.
- Accept only if Phase B is measurably better than Phase A on the gate in §6. Outputs are identical
  by construction because results are per query.

Decision: Phase B isn't planned work. It is written down here so Phase A's interface (comptime
`leaf_order` and a row-mapping input) doesn't block it.

## 5. Tests (focused, CPU and CUDA where applicable)

kdtree.rs unit tests:
- `check_query_routes`, extended: on CPU, for `features ∈ {2, 16, 30, 33}` (covering register and
  runtime paths and `d % 4 ∈ {0, 1, 2}`), quantized inputs, `k = 9`:
  - the SoA query route matches `brute_query` bitwise;
  - the SoA route matches the PaddedRows route (forced with `lanes = 0` on the CPU client) bitwise;
  - every legal lane width in `io_optimized_vector_sizes(4)` is exercised;
  - diagnostics on and off;
  - a single-cube cap (strided blocks);
  - `rows < KD_QUERY_BLOCK`, `rows` not a multiple of 64, and `rows = 1`;
  - query rows > reference rows (for example 203 queries against 71 references, and multiple queries
    against `n = 1`), with a non-multiple-of-64 tail and a forced single cube, comparing every output
    row and the enabled diagnostics with the oracle. This is the case that exposes the old
    `local_row < n` guard;
  - `k > reference rows`: sentinel `(inf, n)` padding.
- Self exclusion on the SoA route with `query_offset = 40`: matches fit's self-query slice bitwise
  (the existing assertion, now on SoA).
- Self exclusion with identical coordinates at different IDs: never excluded by value.
- Fit regression: `kd_cpu_enabled_diagnostics_match_disabled_launch` and the existing fit
  neighbor/oracle tests still pass with the generalized kernel.
- Sizing: `kd_reference_pack_bytes` and peak bytes for both layouts, overflow → `None`, and
  `lanes = 0` versus SoA formulas, including small or unaligned references where SoA is larger
  (`rows=17, lanes=16, d=3`).
- Fused metadata: through `Dispatch`, for both layouts and with rows that need lane rounding, assert
  that the fused `DeviceKdReference` shapes and ordinary fields equal the backend result (Burn doesn't
  check meta against the result).
- Non-quantized behavior: on splitmix data, the SoA route's IDs match a host f64 oracle except at
  documented near-ties (|Δdistance²| within a few ULPs). This pins the §1 contract rather than
  asserting bitwise equality.
- `prepare_kd_reference` panics on `lanes > 0` with a non-CPU client, and on a lane width that isn't
  legal.
- The CUDA route test (`kd_query_pruned_routes_cuda`) still selects PaddedRows and passes under
  `FAST_UMAP_REQUIRE_CUDA=1`.

Integration (`tests/test_kdtree_search.rs`, `tests/test_prediction.rs`):
- The public `prepare_kd_reference` call site (test_kdtree_search.rs:685) is updated to pass `lanes`.
  Test both `0` and the CPU lane width on CPU.
- The existing prediction oracle, batching (`1024 + 1024 + 5` split), mixed-nonfinite and
  CPU-vs-CUDA tests all pass unchanged. The CPU-vs-CUDA test now compares the SoA route against the
  CUDA per-query route.
- The prediction.rs plan tests are updated per 3.5.

Python: `uv run --no-sync pytest python/tests/test_hdbscan*prediction*.py python/tests/test_hdbscan_predictor*.py`
after `maturin develop --release`. No API change is expected.

## 6. Verification and performance gate

Correctness (must pass):
```sh
cargo test -p fast-hdbscan
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan kd_query -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_prediction -- --nocapture
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
```
Known failures that already exist (a parquet fixture, a flaky CUDA timing test, Ruff formatting) are
reported, not hidden.

Performance (judgment-based, following earlier phases):
- **Tool.** `examples/profile_hdbscan_prediction.rs` (requires `tracing`), on CPU, comparing the
  route before and after on the same binary inputs. The baseline is the current source built with
  the same profile.
- **Matrix.** Rows ∈ {20k, 100k}, features ∈ {2, 8, 16, 33}, queries ∈ {1k, rows}, min_samples 5.
- **Metrics.**
  - cold and warm seconds;
  - trace spans `prediction_kd_search` (query phase) and the core-phase search, so the kernel gain is
    separated from the prepare/pack cost in `prediction_reference_upload_prepack`;
  - `planned_device_peak_bytes` (already printed by the example from `PredictionPlan.device_peak_bytes`).
  - for the §1 contract, the count of label/connector differences versus the baseline on each
    non-quantized shape, reported (not required to be zero).
- **Accept** if the KD search spans improve at most shapes, there's no material regression in
  end-to-end warm time at small query counts (the pack cost must be paid back), and the outputs are
  equal (the example already asserts cold == warm). If small-query calls regress, record it and
  decide. Do not hide it behind a query-count gate unless the evidence supports one; any gate would
  be a pre-launch routing rule documented in AGENTS.md.
- **Artifacts.** `artifacts/approx_query_cpu/{baseline,soa}/*.json` and a short summary in
  `hdbscan_perf_report.md`.
- No performance claim is made without these artifacts.

## 7. Risks

- **FMA contraction.** Vector and scalar paths may contract differently on non-quantized data, so
  predicted core distances or candidate tie-breaks could differ in the last ULP from the old CPU route
  (and from CUDA, as they already may). This is acceptable: the contract is exact
  `(distance², ID)` order under the kernel's own arithmetic, and the cross-backend tests already
  tolerate it through quantized fixtures. Release notes should say "CPU prediction distances may
  differ by rounding from the previous release".
- **Fusion metadata drift.** If `meta` and the implementation disagree on shapes, fusion breaks. A
  test runs prediction through `Dispatch` (fusion) on CPU, which the existing prediction tests
  already do, and the shapes are asserted in `validate_reference`.
- **Prepare peak memory.** The temporary padded rows add to the peak. They're accounted for in 3.5.
  Packing straight from compact data would remove them; this is deferred because
  `pack_leaf_soa_kernel` reads `KdVector` chunks.
- **Fit regression from generalizing the kernel.** This is covered by the existing fit tests, the
  diagnostics-equality test and the HDBSCAN stage analysis: `bash analyze_hdbscan_stages.sh --rows 20000`
  for feedback, then the required final `bash analyze_hdbscan_stages.sh --rows 100000`. CPU must show
  no regression against the Phase 7 artifacts. CUDA is expected to be unchanged because its branch
  isn't touched. The comptime flags compile to the same code for
  fit.

## 8. Decisions and rationale

1. **CPU uses LeafSoa; CUDA is unchanged.** The SoA scan's gain is lane vectorization on CPU. On
   CUDA, F2 is exhaustive and the per-query kernel is already the accepted choice for external
   queries.
2. **The SoA lives in the call-local reference, not per batch.** It's packed once per call and reused
   by every core and query batch. Packing per batch would cost `O(n·d)` per batch and dominate small
   batches.
3. **The SoA replaces the padded rows instead of being added beside them.** The SoA kernel never
   reads padded reference rows. Replacing them avoids an allocation that is never read. Total size
   changes with the shape and is sized exactly.
4. **`slot_ids == indices` in PaddedRows mode.** Every field is a real tensor, with no placeholder or
   enum in a fusion extension type.
5. **`lanes` is an explicit argument to `prepare_kd_reference`.** Fusion `meta` can't see the client,
   so the layout has to be chosen by the caller, and the same value drives sizing. This avoids hidden
   routing inside the op.
6. **One generalized CPU kernel.** It avoids two copies of about 300 lines of traversal and heap code
   that could drift apart. Fit's behavior is pinned by comptime flags.
7. **Input query order first; leaf order is Phase B behind its own gate.** It's the smallest change
   that works end to end. Locality is measured, not assumed.
8. **The numerical contract is permitted and documented.** It is not equivalence on arbitrary
   inputs. Preserving scalar arithmetic would throw away the vectorization that is the point of the
   change. Exactness is proven on quantized data, and non-quantized differences are measured and
   reported. This follows the project's convention for
   contraction-sensitive comparisons. There is no runtime fallback or recovery, and routing is decided
   before launch only.
