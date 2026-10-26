# Phase 6 detailed plan: CPU leaf-tiled SoA kernel, vectorized across candidates, with leaf-ordered queries

This document details Phase 6 of [cubecl_kernels_perf_plan.md](cubecl_kernels_perf_plan.md) (F4 and the "leaf-ordered queries" part of opportunity 1 in [hdbscan_perf_report.md](hdbscan_perf_report.md), "Follow-ups"). It builds on the accepted Phase 3 production kernel; Phase 4 was rejected and removed. Line numbers refer to HEAD `24adb89` (Phase 3 production plus planning documents and the Phase 4 negative result).

**Status:** planned; not implemented or measured.

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

1. **Position-tiled SoA instead of per-leaf padded blocks.** The master plan stores `leaf_soa[leaf][feature][slot]` with each leaf padded to `W`. That needs a per-leaf block offset (a prefix sum over leaf sizes) and a total size that depends on tree contents. `DeviceKdTree` is public, and `validate_tree` (`kdtree.rs:887–929`) checks only shapes and dtypes, so leaf ranges may overlap or be early leaves (the Phase 4 policy treats flags and ranges as data). The total would then be bounded only by `nodes · ⌈n/W⌉` tiles, or would require a device readback before allocation, and could not be pre-planned in `core_search_memory` from host shapes. This plan instead tiles the **position axis** of `tree.indices` globally: tile `t` holds positions `[tW, tW + W)`, laid out as `soa[t][feature][lane]` with `ceil(n/W)` tiles. A leaf `[start, end)` scans tiles `⌊start/W⌋ .. ⌈end/W⌉` and masks lanes outside `[start, end)`. Size depends only on `n`, `d`, `W`; every public tree is legal; no prefix sum, readback or flag dependence. Cost: misaligned leaf starts add masked lanes. For a leaf of `L` points with uniformly distributed start offset, the expected number of tiles is `(L + W − 1)/W`, against `⌈L/W⌉` for per-leaf padding: at `L ≈ 60`, about 4.69 vs 4 tiles at `W = 16` (+17% lane work) and 8.38 vs 8 at `W = 8` (+5%). This cost is real and is one input to the Step 4a width selection; a per-leaf layout is recorded as a possible later experiment if Phase 6 is accepted and lane waste is shown to matter, not part of this phase.
2. **Block-cyclic leaf order, not one contiguous chunk per unit.** One contiguous chunk of `⌈n/U⌉` positions per unit (`U` = units, 15 on this host) maximizes locality but fixes the load split at launch; traversal cost varies across regions (Gaussian tails scan more), and the CPU kernel has no work stealing. Units instead take blocks of `KD_QUERY_BLOCK = 64` consecutive positions (about one leaf: 20k rows give ~78-point leaves, 50k ~49), assigned block-cyclically. The value is predeclared, not tuned.
3. **Exclusive row ownership for leaf-ordered queries.** The kernel keeps the output-backed top-k heap (Phase 1), so each output row must be owned by exactly one unit. Iterating rows through `tree.indices` gives that only if `indices` is a permutation, which `validate_tree` does not check (`:924` checks length only). Today, a non-permutation `indices` still yields a deterministic result for every row. To preserve this, a single-unit pack kernel writes `row_owner[row]` = the first position holding `row`, or `n` if `row` is absent. A position processes its row only if it is the owner; a tail pass then processes absent rows (`row_owner[row] == n`). Every row is processed exactly once by one unit, for any `indices`. Values outside `[0, n)` behave as in the current kernel for candidate reads and are skipped as query positions (their rows are covered by the tail pass).
4. **Vector admission screen as an OR of lane comparisons.** "Compare a whole vector against the broadcast `worst`" is implemented as: per tile, `any` over in-range lanes of `distance_lane <= worst_distance` (worst at tile start). Only if true are lanes checked one by one, in lane order, with the exact Phase 1 admission test against the **current** worst registers. A horizontal minimum is not used: if data contains NaN (the public op does not reject it), a NaN-propagating minimum would skip admissible lanes. The self lane is not excluded from the screen (that would need the slot ID loads); including it is conservative.
5. **Reuse of Phase 3 traversal code.** Node traversal (`padded_box_lower_bound`, padded bounds, register query, the `d > 32` runtime chunk path) is reused unchanged, so pruning decisions are produced by the same expressions. The padded data copy is kept because the query and the runtime-path bound evaluation read it.
6. **Width selection with a cap.** `W` is the first value of `client.io_optimized_vector_sizes(4)` not above `KD_SOA_MAX_LANES`. The cap is chosen by a predeclared selection batch (Step 4a) that is separate from the acceptance batches; the initial cap is 16.
7. **All four arms in one kernel.** P, O and C are comptime configurations of one new CPU kernel (`kd_neighbors_cpu_kernel`), so Phase 3's CUDA kernel source stays untouched during measurement. O uses a verbatim copy of the Phase 3 per-candidate scan. Step 6 removes unselected configurations.

## Non-goals

- `kd_neighbors_kernel` (CUDA production), `kd_candidates_kernel` (Borůvka), `kd_neighbors_reference_kernel`, `box_lower_bound`, `DeviceKdTree`, `HdbscanOps`, `upload_kd_tree`, the host `KdTree` builder, fusion metadata.
- CUDA launch geometry (Phase 5), CUDA plane kernel (Phase 7), register top-k (rejected Phase 2), node records (rejected Phase 4).
- Routing thresholds and the >30-feature kNN/ANN path.
- Any change to the per-pair arithmetic or feature order, traversal order, pruning, child ordering, admission predicate, heap layout, heapsort or diagnostics semantics.
- Shared memory or barriers (`PLANE_DIM` is 1 on the CPU runtime; no on-chip benefit).

## Invariants

1. Output IDs and distance bits are identical to the frozen reference (`kd_neighbors_reference_kernel`, `kdtree.rs:420–621`) and thus to Phase 3 production, for every harness case, every selectable `W`, and both query orders.
2. Per-row diagnostics (`nodes_popped`, `leaf_positions_scanned`) are identical. `leaf_positions_scanned` counts real positions `end − start` of each scanned leaf with `start < end`, never masked lanes.
3. Each `(query, candidate)` distance is `0.0 + Σ_{f=0}^{d−1} (q_f − c_f)²` accumulated in order `f = 0..d−1` in one lane. The register path uses exactly `d` features; the runtime path (`d > 32`) also uses exactly `d` (the SoA has no feature padding).
4. Original IDs everywhere: self-exclusion, tie keys and output rows use `indices` values; tiles and positions are memory layout only.
5. Each output row is written by exactly one unit, once per call (row ownership, item 3 above).
6. Error contract: `validate_tree` runs first. New buffers get checked sizing, the existing `u32` element-count assertion pattern of `pad_rows` (`:1154–1157`), inclusion in `core_search_memory`, and `check_device_allocation` calls.
7. CUDA route selection and kernel are unchanged; the CPU kernel is never launched on a non-CPU client.

## Equivalence argument

Fix a query row. The traversal is a sequence of events: pop, prune test, leaf scan, child bound evaluation, push. Child bounds and the root bound are computed by the unchanged Phase 3 helpers from the same padded inputs, so they are bit-identical. Within a leaf, the scalar kernel visits positions `start..end` in increasing order and, for each `candidate ≠ row`, applies `distance < worst || (distance == worst && candidate < worst_index)`, then replaces the root, sifts and refreshes the worst registers. The SoA kernel visits tiles in increasing order and, inside a tile, lanes in increasing order; lanes outside `[start, end)` are masked, so the in-range lanes enumerate exactly `start..end` in the same order. Each lane's distance is the same ordered sum of the same `f32` values (SoA values are bit copies of `data[indices[p]·d + f]`, and the query scalar is the same value Phase 3 holds in its register vector), computed by the same operations in a different SIMD position; this is bit-identical if both kernels make the same FMA-contraction choice, which Step 3 tests and Step 4d inspects. The tile screen skips a tile only if every in-range lane has `distance > worst_at_tile_start` or is NaN; because `worst` never increases, the scalar kernel would admit none of those lanes either. Admissions, heap contents and worst registers therefore evolve identically; by induction pruning, pushes and diagnostics are identical, and the final heapsort (copied verbatim) produces identical rows. Query order only changes which unit computes a row and when; rows are independent and ownership is exclusive (Invariant 5), so the output is identical for any schedule.

## Step 0: Baseline and prerequisites

1. Start from a clean tree at the then-current HEAD; record `git rev-parse HEAD` and the working-tree diff hash in `artifacts/cubecl_kernels_perf/phase6/provenance.json`, with `rustc -V`, CPU model, `nproc`, the CubeCL-reported `num_cpu_cores`, `hardware.load_width` and the full `io_optimized_vector_sizes(4)` list on this host.
2. Build the baseline (`cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages`) and preserve it as `phase6/bin/profile_hdbscan_stages.baseline` with SHA-256. If HEAD is still `24adb89`-equivalent in source, the hash should equal the Phase 4 baseline hash `14ca624f…`; record match or mismatch (a mismatch is not an error after Phase 5 lands, but must be explained).
3. **API probe** (a throwaway `#[cfg(test)]` kernel, removed before the experiment diff is final; record outcomes in `provenance.json`). CubeCL 0.11.1 has `ClientProperties::io_optimized_vector_sizes` (`cubecl-runtime-0.11.1/src/client.rs:1786`) and `trait Size` with `value()` (`cubecl-core-0.11.1/src/frontend/element/base.rs:1006`); generic `N: Size` kernels take a `VectorSize` launch argument (`cubecl-core-0.11.1/src/runtime_tests` `kernel_decode`). Confirm on CPU, for `N ∈` every value of the host list:
   - a `#[cube(launch)]` kernel generic over `F: Float, I: Int, N: Size` with `#[define(F, I)]` and a `Tensor<Vector<F, N>>` argument created by `TensorArg::from_raw_parts(handle, [1], [vectors])`, as in `padded_vector_view` (`:1176–1189`);
   - `#[unroll] for lane in 0..N::value()` with `vector.extract(lane)`;
   - a scalar-to-`Vector<F, N>` broadcast constructor, element-wise `-`, `*`, `+=`;
   - optionally, a vector `<=` producing a lane-wise boolean vector whose lanes can be extracted. If absent, the screen uses scalar comparisons on extracted lanes (semantically identical).
   - a way to dump the CPU-generated MLIR/LLVM IR for one kernel (CubeCL compilation logging in 0.11.1). If none exists, record that; Step 4d then reports "not available".
4. Determine from CubeCL CPU runtime source whether the units of one cube execute concurrently. This does not change the design (which inherits the existing one-cube `num_cpu_cores`-unit geometry), but the report must state what the block-cyclic split actually distributes.

## Step 1: Harness

The existing `check_kd_neighbors_match_reference` (`kdtree.rs:1420–1564`) compares production `launch_kd_neighbors` against `exact_neighbors_kd_reference` on IDs, distance bits and diagnostics, with diagnostics on and off, across `d ∈ {1,2,3,4,5,8,12,16,20,24,28,29,30,32,33,40}`, `k ∈ 1..=17, 24, 30`, duplicates, grids, overflow inputs and `n ∈ {1,3,4,9,257,512,2000}`, on CPU and required CUDA. Keep all cases.

Changes and additions:

1. **Plan-parameterized entry.** Split `launch_kd_neighbors` into `launch_kd_neighbors(data, tree, k, diagnostics)` = `launch_kd_neighbors_with_plan(…, kd_neighbor_plan(&client))`. The harness runs, on CPU, every case for: the production plan; `LeafSoa { lanes: w }` × `{Cyclic, LeafBlocks}` for **every** `w` in `client.io_optimized_vector_sizes(4)` (including `1`); and `PaddedRows × LeafBlocks`. On CUDA it runs the production plan only, plus an assertion that `kd_neighbor_plan` is `PaddedRows × Cyclic` on the CUDA client. Record the CPU harness wall time; if it exceeds 10 minutes in release mode, keep every `w` for the full `d`/`k` list but run the large-`k` sweep (`k ∈ 1..=17, 24, 30`) only at the production `w` and `w = 1`, and say so in the log.
2. **New special inputs**, driven through the same comparison:
   - `n` not a multiple of any `w` and `n < w`: `n ∈ {1, 4, 9, 15, 17, 33, 257}` at `d ∈ {2, 16}`;
   - `"nan"`: `n = 300, d ∈ {3, 16}`, a few NaN feature values in non-query rows (tests the screen's NaN rule against the reference; the builder rejects non-finite values, so build the tree from a NaN-free copy and upload the NaN data separately, as the public op permits).
3. **Hand-built trees** (host-edited `KdTree::build` output, uploaded through a variant of `make_neighbor_inputs`):
   - early leaf: on `n = 200, d = 3`, flag one level-1 internal node as a leaf with flag `2` and another leaf with flag `−1` (Phase 4 precedent);
   - non-permutation `indices`: on `n = 300, d = 4`, overwrite one position with a duplicate of another row (so one row is absent and one appears twice). The reference kernel defines the expected output; this exercises ownership and the tail pass;
   - empty leaf: one unused heap slot (`0..0`) flagged as a reachable leaf;
   - out-of-range candidate inside a **scanned** leaf: on `n = 300, d ∈ {3, 5, 16}`, set one position of a leaf that every query reaches (make it the root-level leaf on a tiny tree and also one deep leaf) to `n + 3`, asserting identical output against the reference (exercises clamped padded-vector gather; d = 3 and 5 make vector vs scalar clamping differ);
   - `end > n`: one leaf's `end` set to `n + 5` (exercises the scalar tail loop and clamped `indices` reads).
4. **Pack unit tests** (`check_pack_leaf_soa`, `check_row_owner`; CPU only, since the kernels are CPU-only):
   - SoA: for `n = 37`, `d ∈ {1, 3, 30, 33}`, every host `w`, values including `-0.0` and a permuted `indices`; read back and assert by `to_bits` that element `(t·d + f)·w + lane` equals `data[indices[tw+lane]·d + f]` for `tw+lane < n`, and `0x0000_0000` otherwise; slot IDs equal `indices[p]` or `n` for padding;
   - ownership: a permutation, a duplicate, an absent row and an out-of-range value (`n + 3`), checking first-occurrence ownership and `n` for absent rows.
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

**Malformed-index semantics.** CubeCL checked indexing clamps out-of-bounds reads to the last buffer element (`cubecl-core-0.11.1/src/io.rs:34–35, 75–78`), not to zero. Phase 3 reads candidates as whole padded vectors, so an out-of-range ID (e.g. `n + 3`) yields the last *vector* of `data_v`, i.e. the last row's real features; a scalar gather from `data` would instead repeat the last scalar. Gathering through `data_v` with the identical vector index reproduces Phase 3 bit for bit, by construction. The pack kernel takes `data_v` (not `data`) for this reason.

```rust
#[cube(launch)]
fn kd_row_owner_kernel<I: Int>(
    indices: &Tensor<I>, row_owner: &mut Tensor<I>, n: u32,
    #[define(I)] _dtype: ElemType,
);
```

Launched with one cube of one unit (CPU only; sequential `O(n)`): set every `row_owner[r] = n`, then for `p in 0..n`, `r = indices[p]`; if `0 ≤ r < n` and `row_owner[r] == n`, write `row_owner[r] = p`. Only allocated and launched for `LeafBlocks`.

Views: `soa` as `Tensor<Vector<F, N>>` with `tiles · d` vectors via a `soa_vector_view(tensor, tiles, d, lanes)` helper following `padded_vector_view` (`:1176–1189`), with the same length assertion and SAFETY comment.

### 2c. CPU kernel

```rust
#[cube(launch)]
fn kd_neighbors_cpu_kernel<F: Float, I: Int, N: Size>(
    data_v: &Tensor<KdVector<F>>,            // pad_rows(data): query, runtime-path bounds, tail loop
    data: &Tensor<F>,                        // [n, d]: runtime-path query scalars
    soa: &Tensor<Vector<F, N>>,              // LeafSoa only (1-vector dummy otherwise)
    slot_ids: &Tensor<I>,                    // LeafSoa only
    row_owner: &Tensor<I>,                   // LeafBlocks only (1-element dummy otherwise)
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
- `LeafBlocks`: work items over `0..2n` in blocks of `KD_QUERY_BLOCK` assigned block-cyclically by unit. For `w < n`: `r = indices[w]`; process `row = r` iff `0 ≤ r < n` and `row_owner[r] == w`. For `w ≥ n`: `row = w − n`; process iff `row_owner[row] == n` (absent rows; normally none, one cheap read each).

**Per-row body** (`#[cube] fn kd_search_row`, inlined once): a verbatim copy of the Phase 3 row body (`:651–857`: output init, worst registers, stacks, register query load, root bound, pop/prune/children, heapsort, diagnostics), with only the leaf scan replaced when `leaf_soa`:

```text
if start < end:
    if diagnostics: leaf_positions_scanned += end - start
    tile_end = min(end, n)                   // positions < n come from the SoA
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
    for position in max(start, n) .. end:    // malformed `end > n` only: verbatim Phase 3
        Phase 3 scalar position body (:691–751), whose clamped `indices[position]`
        and `data_v` reads define the semantics
```

Positions `≥ n` never touch the SoA, so the clamped `indices[n−1]` reads of Phase 3 are reproduced exactly; for builder trees (`end ≤ n`) the tail loop is empty.

The `PaddedRows` path keeps the Phase 3 position loop (`:687–752`) verbatim. The heap replacement and heapsort are copied, not shared with `kd_neighbors_kernel`, so CUDA code does not change; this duplication is temporary and Phase 8 removes it (or Step 6 folds it, below).

### 2d. Host launch

In `launch_kd_neighbors_with_plan`, after the existing `validate_tree`, output allocation and `pad_rows` of data and bounds (`:1050–1066`):

- `PaddedRows × Cyclic`: the existing `kd_neighbors_kernel::launch` (`:1068–1092`), unchanged.
- CPU plans: `pack_leaf_soa` (if `leaf_soa`), `kd_row_owner` (if `leaf_blocks`), then `kd_neighbors_cpu_kernel::launch` with `kd_launch_geometry(&client, n)` (unchanged CPU geometry: one cube, `num_cpu_cores` units) and the `VectorSize` argument `lanes`. `exact_neighbors_kd` and diagnostics emission (`:1024–1038`) are unchanged.

### 2e. Sizing and allocation checks

```rust
/// Device bytes of all per-call KD neighbor packs for `plan`.
pub(crate) fn kd_neighbor_pack_bytes(plan: KdNeighborPlan, rows: usize, nodes: usize, features: usize)
    -> Option<usize>;
```

= `kd_padded_bytes(rows, nodes, features)` (Phase 3 term, all plans) `+` for `LeafSoa { lanes }`: `tiles · lanes · (features + 1) · 4` with `tiles = ⌈rows/lanes⌉` (SoA floats plus slot IDs) `+` for `LeafBlocks`: `rows · 4`. All checked. Example: `rows = 1000, nodes = 63, features = 5, lanes = 16`, C plan: `(1000 + 126)·8·4 + 63·16·6·4 + 1000·4 = 36,032 + 24,192 + 4,000 = 64,224`.

Estimator (`estimator.rs:1101–1104`): `kd_pack_bytes` comes from `kd_neighbor_pack_bytes(kd_neighbor_plan(client), …)`. Allocation checks (`:1149–1167`): add 1D `check_device_allocation` calls for the SoA (`tiles·lanes·features` f32), slot IDs (`tiles·lanes` i32) and owner map (`rows` i32) when the plan uses them. The host functions assert that the SoA element count fits `u32`, as `pad_rows` does.

### 2f. Rust/Rayon host baseline (measurement example)

Add `crates/fast-hdbscan/examples/kd_host_baseline.rs` with `rayon = { workspace = true }` as a dev-dependency of `fast-hdbscan`. Usage: `kd_host_baseline INPUT ROWS FEATURES [--k 5] [--runs 5] [--threads T] [--verify]`. It builds the tree with `fast_hdbscan_core::KdTree::build` (not timed), then times a scalar host port of the Phase 3 traversal (same stack, child order, bounds, admission, heap and heapsort; distances as sequential `f32` sums) with `rayon` `par_chunks_mut` over output rows, in a pool of `T` threads (default: CubeCL `num_cpu_cores`). It prints JSON with per-run times and the median. `--verify` runs `HdbscanOps::exact_neighbors_kd` once on the CPU backend through the public API (as `tests/test_kdtree_search.rs` does) and reports ID and distance-bit equality. Rust does not contract FMA by default; a mismatch is a reported finding, not a gate.

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

**Gate.** This is the experimental matrix (all arms); after Step 6 the final-gate matrix defined there applies instead. All pass with no hardware skips; every harness case is identical in IDs, distance bits and diagnostics for every plan and every host `w`. Because the harness exercises all arms through the plan parameter, it is run once on the measurement source; each arm binary additionally passes the `profile_hdbscan_stages` example test built with its `KD_CPU_ARM` value. Logs go to `artifacts/cubecl_kernels_perf/phase6/logs/`.

**Mismatch triage.** Never change the reference kernel or tests to match.
- Distances differ, IDs/diagnostics follow: FMA contraction or operation-order difference between the vector and scalar sums. Inspect the generated IR (Step 0.3). Fix by expressing the vector accumulation so it matches the scalar kernel's contraction (e.g. the same `delta * delta` then `+=` shape the scalar code uses), never by changing the scalar kernel. **Width policy:** every width in the host list must pass; there is no width-exclusion mechanism. If any width stays unfixed, P and C are rejected (O is unaffected).
- Distances equal, diagnostics differ: tile range or lane mask error, or the screen skipped an admissible lane.
- Some rows uninitialized or differing only under `LeafBlocks`: ownership/tail-pass error.

**End to end.** Final labels must be exactly equal across all arms within every benchmark cell (Step 4).

## Step 4: Performance measurement

All CPU, same inputs and hashes as Phases 3–4 (`artifacts/cubecl_kernels_perf/inputs/`; 16D/8D/2D cells use the existing derivation recorded in Phase 3 provenance). Command form unchanged: `<binary> <input.f32le> <rows> <features> --backend cpu --runs 3 --min-samples 5`. Each process reports core-stage and `core_neighbor_kernel` medians (the latter includes all packing), full-fit median and labels.

Runner: copy `artifacts/cubecl_kernels_perf/phase4/run_alternating.py`. Allowed changes, and only these: binary/input paths, an arm list (`--arms`) and per-cell order pattern, `--backend cpu` only, a `--batch` label (`S`, `A`, `B`), and a fresh output directory (still refusing to overwrite). Record the diff against the Phase 4 copy.

### 4a. Width selection (separate, before acceptance)

Arms: C built with `KD_SOA_MAX_LANES` set to each of the three widest values of the host list, mapped before any run as `X` = widest, `Y` = second, `Z` = third (expected 16, 8, 4 on AVX-512; mapping recorded in Step 0). Cells 20k×16 and 20k×30, one batch `S` of six blocks forming a Williams design (each arm twice in each position, each ordered adjacent pair once per three blocks): `XYZ | YZX | ZXY | ZYX | XZY | YXZ` (six processes per arm). Rule, predeclared: start from the widest; select a narrower width only if its core median is at least 2% lower than the currently selected width's in **both** cells. The selected cap is fixed for all later steps. Batch `S` data is reported but never pooled into acceptance statistics.

### 4b. Isolation and acceptance timing (H2)

- Arms: **B** (baseline binary), **P**, **O**, **C** (variant source at the Step 4a cap, differing only in `KD_CPU_ARM`).
- Cells: primary 20k×16, 50k×16, 20k×30, 50k×30; guards 20k×8 and 20k×2 (smallest work, where pack and owner launches weigh most).
- Per cell and batch, four blocks of four processes forming a 4×4 Williams square (every arm once in every position; every ordered adjacent pair once). Batch `A` = `B,P,C,O | P,O,B,C | O,C,P,B | C,B,O,P`. Batch `B` = the 16-process list of `A` reversed = `P,O,B,C | B,P,C,O | C,B,O,P | O,C,P,B`, which balances temporal direction. The runner encodes both lists literally. Cells serial. Two predeclared batches, `A` then `B`, each over the full matrix. No further batches; neither batch is selected.
- Aggregation per cell and arm `X ∈ {P,O,C}`: pool the eight process medians per arm; `Δ_X = median(X)/median(B) − 1`. Per-batch `Δ_X` is reported too: the master H2 says to compare within a batch, and pooling two predeclared, direction-balanced batches follows the Phase 4 precedent; this is recorded as a deviation. Paired screen, fixed before measurement: in each block, the ratio `X/B` of that block's two processes regardless of position; eight ratios per arm and cell, reported with min–max. The screen is a consistency check, not a significance test.
- Component attribution (reporting-only, derived): `median(C)/median(P)` (effect of O given P) and `median(C)/median(O)` (effect of P given O) per cell.

### 4c. Rayon host baseline (reporting-only)

After batch B: three processes of `kd_host_baseline --runs 5 --verify` per primary cell, `--threads` = CubeCL `num_cpu_cores`. Report the median traversal time next to the pooled `core_neighbor_kernel` medians of B and the selected arm, and the verify result. The ratio is an implementation-level comparison (runtime, scheduling, scalar vs vector codegen and buffer handling all differ), not an isolated measure of runtime overhead and not an acceptance input.

### 4d. Codegen evidence (reporting-only)

If Step 0.3 found an IR dump mechanism: dump `kd_neighbors_cpu_kernel` for the selected arm at d = 16 and the Phase 3 kernel on CPU at d = 16. Record whether the candidate feature loop uses `<W x float>` loads and arithmetic, and whether either kernel emits `fma` (or contraction flags) for the distance sum. Store under `phase6/generated/`.

### 4e. JIT/cold start (reporting-only)

Copy `phase4/cold_start.py` unchanged except paths and backend restricted to CPU; d ∈ {2, 16, 30}; baseline and selected-arm binaries. Report first-fit medians and first-minus-warm using the same binary's pooled H2 full-fit medians (sources recorded). The new kernel adds one compiled variant per `(d, W, order)` plus two pack kernels; this is where a JIT cost would show.

### 4f. Final pipeline validation

With the final production code (after Step 6): `bash analyze_hdbscan_stages.sh --rows 20000` and `bash analyze_hdbscan_stages.sh --rows 100000`, logs in `phase6/logs/analyze_*.log`. CPU/CUDA labels must agree. CUDA kernel filters (`analyze/hdbscan_stages_common.py:42`, `profile_hdbscan_kernels.py:138–139`) need no change because the CUDA kernel names are unchanged; CPU stage timing is span-based. These are single runs, not interleaved comparisons.

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

- **C or P:** remove `KD_CPU_ARM`; the CPU plan is fixed to the selected `(LeafSoa { lanes }, order)` with the Step 4a cap. Remove the unselected comptime branches (`PaddedRows` path from the CPU kernel; `LeafBlocks`, `row_owner` kernel, owner buffer and their sizing term if P). Keep the harness at every host `w`.
- **O:** do not ship a duplicated per-query kernel. Fold block-cyclic leaf order and the owner map into `kd_neighbors_kernel` as a `#[comptime] leaf_blocks` flag (CPU `true`, CUDA `false`), and delete `kd_neighbors_cpu_kernel` and the SoA pack. The fold changes both the measured CPU implementation and the CUDA kernel source, so it has its own predeclared protocol, with final-source and binary hashes archived: (a) the final-matrix Step 3 gate with required CUDA; (b) CUDA: two batches `A`, `B` of `B,V,V,B,B,V` on the four primary cells plus 20k×2, judged with the unchanged Phase 4 thresholds (six pairs), plus H3 at 20k×16 and 20k×30; any CUDA regression, register increase or local-memory increase blocks the fold; (c) CPU: two batches of `B,V,V,B,B,V`, folded binary against baseline, on all six cells, Phase 4 thresholds; the fold is accepted only if ≥3 of 4 primary cells still gain and no cell regresses. If (b) or (c) fails, O is rejected rather than shipped as a duplicate.
- **Final-gate matrix.** After finalization, Step 1 item 1 shrinks to the plans the final source still contains: the production plan on CPU at every host `w` (C/P) or the production plan alone (O), plus the CUDA production plan. Removed arms are not kept behind `#[cfg(test)]`; their equivalence evidence is the measurement-source Step 3 logs, which are retained. Re-run Step 3 with this matrix on the final source, then Step 4f.

## Step 7: Deliverables

- Code: plan/route, CPU kernel, pack kernels and helpers, sizing and allocation checks, tests, `kd_host_baseline` example and its dev-dependency.
- `artifacts/cubecl_kernels_perf/phase6/`: provenance (including API probe outcomes and units-concurrency finding), binaries with hashes and per-arm diffs, runner and its diff, batch `S`/`A`/`B` records, Rayon baseline JSON, generated IR (if available), cold-start JSON, logs, `summary.json` with per-cell, per-arm results and the decision.
- `hdbscan_perf_report.md`: "Phase 6 results", scoped to the measured CPU cells on this host, including negative or partial results.
- `cubecl_kernels_perf_plan.md` Phase 6: status line, link to this plan, and the recorded design changes (position-tiled SoA, block-cyclic order, ownership map, selected `W` cap).
- `AGENTS.md`/`README.md`: re-check with a search at completion. Update only where they describe KD traversal, device buffers or resource sizing (none currently enumerate the KD CPU schedule); list the new example only if a README section enumerates examples.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Vector vs scalar FMA contraction changes distance bits | Bit-equality at every host `w` on CPU; Step 4d IR inspection; triage procedure; any unfixed `w` rejects P/C |
| Lane waste at misaligned leaf starts (est. +17% at W = 16) | Quantified up front; Step 4a selects `W` empirically; per-leaf layout recorded as a later experiment |
| Load imbalance from leaf order | Block-cyclic `KD_QUERY_BLOCK = 64`; isolation arm O measures order alone |
| Non-permutation public `indices` races the output-backed heap | Owner map plus tail pass; dedicated harness case |
| NaN in public input breaks the screen | OR-of-comparisons screen; dedicated harness case |
| JIT cost of `(d, W, order)` variants on CPU LLVM | Only the selected `W`/order compiles in production; cold start reported (4e) |
| Extra memory (SoA ≈ one extra copy of the data, plus owner map) | Counted in `core_search_memory` and allocation checks; reported in the results |
| Small cells dominated by pack/owner launches | 20k×2 and 20k×8 guards; regression screen |
| Units of one CPU cube not concurrent as assumed | Step 0.4 establishes the facts; design inherits the existing geometry either way |
| Duplicate heap code in two kernels | Copied verbatim to keep CUDA untouched; O folds into the CUDA kernel source only with CUDA verification; Phase 8 removes duplication |
| Gains specific to Gaussian 16D/30D inputs | Scope stated in results; no generality claim |
