# HDBSCAN Performance Investigation

Scope: `fast-hdbscan` (facade, CubeCL kernels) and `fast-hdbscan-core` (host extraction). This builds on
`core_dist_perf.md`, `core_dist_refactor_plan.md` and `parallel_boruvka_plan.md`; their findings and the
rejected KD experiments (local heap, query caches, CUDA cube sizes 16/64/128) are not repeated here.

**Evidence labels**

- **[measured]**: timed with `profile_hdbscan_stages` (release, `--features tracing`). The backend is named
  every time. These are warm fits, the median of repeated runs.
- **[simulated]**: exact Python/numpy (f64) model of the Borůvka algorithm, kept in `/tmp/hdb/sim_boruvka.py`. It
  is not the device code.
- **[proposal]**: not implemented, compiled or run.

Hardware: RTX 3080 10 GiB (driver 617.14) for CUDA; the CPU runtime is CubeCL LLVM with 15 workers. Timings
from the CPU runtime say nothing about GPU behaviour, and the reverse is also true.

Data: synthetic Gaussian blobs (20 centers) and uniform data in `/tmp/hdb/`. Defaults apply
(`min_samples = 5`).

## 0. Correctness hazards (read first)

The proposals below change which rows and nodes get searched. Before any pruning lands, these points must
hold:

1. **Exactness under ties.** `merge_components` (`fast-hdbscan-core/src/mst.rs:78`) takes the first
   source per component with strict `<`. Per-point kernels choose the minimum target index among equal
   weights. Every prune has to use a strict test (`lower_bound > bound`), never `>=`, or a tied
   lower-index edge can be dropped and the output MST will change. Any skipped row needs a sentinel
   (`+inf` weight with target `n`), and `merge_components` has to treat target `n` as "no candidate"
   instead of indexing out of bounds.
2. **f32 vs f64.** The simulation used f64. Candidate reuse is exact in f32 only when the reused weight is
   the same stored f32 value. Never recompute it with a different operation order: KD and BF compute
   distances differently, so they can differ in the last ulp.
3. **`reproducible`.** Python accepts `_reproducible` and then drops it
   (`crates/fast-umap-python/src/hdbscan.rs:215`). Every proposal here is deterministic. None of them needs
   the non-reproducible shared-bound route in `parallel_boruvka_plan.md` §4.
4. **CUDA layout aliasing** does not apply to HDBSCAN. Buffers that new kernels update in place (the
   proposed per-row best/target state) must still be unaliased.

## 1. Auto routing sends most real workloads to the slow path [measured]

`HdbscanSearch::Auto` (`estimator.rs:36-70`) picks KD only for 256..=2048 rows with d ≤ 30, or for
2049..=4096 rows with d ≤ 10, and min_samples ≤ 32 in both cases. **Above 4096 rows, everything goes to brute
force.** The range was set from a campaign that stopped at 4096 rows.

Warm fit seconds:

| Case | CPU BF | CPU KD | CUDA BF | CUDA KD |
|---|---:|---:|---:|---:|
| blobs 5k×2 | 0.060 | 0.0095 | 0.040 | 0.009 |
| blobs 5k×10 | 0.085 | 0.023 | 0.047 | 0.023 |
| blobs 10k×2 | 0.25 | 0.015 | 0.12 | 0.014 |
| blobs 50k×2 | 6.1 | 0.065 | — | — |
| blobs 20k×30 | ≈2.7 | 0.77 | 0.95 | 0.17 |
| blobs 10k×50 | 1.19 | 0.31 | 0.30 | 0.087 |
| blobs 10k×100 | 2.3 | 0.77 | 0.50 | 0.16 |
| blobs 200k×2 | — | 0.27 | — | 0.154 |
| uniform 10k×20 | 0.44 | 0.48 | 0.18 | 0.059 |
| uniform 10k×50 | 1.03 | 1.06 | 0.29 | 0.109 |

KD was **never measurably slower** than BF in any case above 4096 rows. On CPU with uniform data at d=20/50
it ties within noise. Everywhere else it is 3–100× faster, on both backends. At 50k×2 CPU, BF's Borůvka
alone takes 5.6 s (6 rounds of about 0.9 s), because every round recomputes all n² distances
(`kernels/boruvka.rs`).

**Recommendation (highest value, smallest change):** use KD as the Auto default for n > 4096 on both
backends, at least for d ≤ 100 with min_samples ≤ 32. Keep BF for n < 256. Re-run the Criterion campaign
over n ∈ {8k, 32k, 128k} × d ∈ {2, 10, 30, 50, 100} × {blobs, uniform}, then update the doc comment and the
`resolve` tests (`estimator.rs:1001-1033`). Clustered structure is where KD is weakest relative to BF, so
the campaign should also include real fixtures, not only the synthetic data used here. Also check the
min_samples > 32 limit: the KD heap lives in global output, so a larger k costs bandwidth but stays
correct.

## 2. Borůvka dominates the KD fit, and late rounds cost the most [measured]

| Case | core | mst | per-round seconds |
|---|---:|---:|---|
| blobs 50k×10, CPU KD | 0.11 | 0.80 | 0.10, 0.08, 0.18, 0.22, 0.20 |
| blobs 50k×10, CUDA KD | 0.013 | 0.16 | — |

MST takes 80–90 % of KD fit time. Later rounds are *more* expensive even though there are far fewer
components. With few large components, the per-row "nearest other-component" radius grows. The KD lower
bound `max(box_dist, core_row)` then prunes less, and each query visits more of the tree. Each round still
searches all n rows, including rows whose answer cannot have changed.

### 2a. Exact per-row candidate reuse [simulated → proposal]

Components only merge, so for row i the candidate set "points outside comp(i)" only shrinks, and its best
weight is non-decreasing. If last round's target `q_i` is still outside comp(i), then `(w_i, q_i)` is
*still exactly* the best edge for row i. That includes the min-index tie rule: `q_i` was the minimum index
among the ties in a superset. So row i can skip its search completely.

Round 1 has a similar certificate. If some kNN neighbor j of i (within the min_samples list) is in another
component and `core_j ≤ core_i`, then `w = core_i`. That equals the lower bound, so the weight is certified
without a search. The target still needs the min-index tie check, so certify the weight and let the
component reduction take the candidate. Alternatively, only use this certificate as an upper bound (see 2b).

### 2b. Exact per-component upper bound [simulated → proposal]

The reused rows give each component an upper bound `U_c = min w_i` over its reused rows. A non-reused row
whose lower bound (last round's `w_i`, or `core_i` in round 1) is **strictly** greater than `U_c` cannot win
its component, so it can skip its search and write the sentinel. Rows that are still searched can also use
`U_c` as the initial KD pruning radius, which cuts traversal even for them. This differs from the
`parallel_boruvka_plan.md` §4 shared bound: here the bound is fixed before the launch and is
deterministic.

Simulated fraction of rows that would still need a full search (exact, verified by assertion against a full
recomputation every round):

| Data (10k rows) | round 1 | round 2 | round 3 | round 4 | round 5 |
|---|---:|---:|---:|---:|---:|
| blobs d=2 (797→3 comps) | 0.355 | 0.133 | 0.197 | 0.135 | 0.217 |
| uniform d=20 (321→6 comps) | 0.058 | 0.013 | 0.002 | — | — |
| blobs d=50 (142→7 comps) | 0.031 | 0.760 | 0.596 | — | — |

For d=2 blobs and uniform data, 65–99 % of row searches can be skipped exactly. Blobs at d=50 are the
counterexample: after round 1 the clusters merge into a few large components, the previous targets move
inside those components, and reuse falls to 2–9 %. **The gain depends on the data and must be measured on
the device.** Expected cost per round: one O(n) "is q_i still foreign" pass and one O(n) segmented min. Both
can run on the device, and they replace a full query on every skipped row.

Implementation notes [proposal]:

- Keep `best_w`/`best_q` resident on the device between rounds. The host already rebuilds
  `point_components` each round, so the reuse test needs one extra small kernel or a host pass over data that
  is already read back.
- The active-row compaction (a list of rows that still need a search) should feed the KD candidate kernel.
  Then skipped rows cost no cube slots at all, which matters on CUDA where each 32-wide cube is one query
  group. Masking inside the kernel would leave the warps diverged and much of the win would be lost.
- Brute-force Borůvka gets the same benefit, and more of it, since its per-row cost is O(n·d).

### 2c. Tighter KD lower bound with a per-node min core [proposal]

Inside a node, the mutual-reachability lower bound is `max(box_dist, core_row, min_core(node))`. The tree
currently uses only `max(box_dist, core_row)`. Storing `min_core` per node costs n_nodes floats. Core
distances are fixed after the kNN stage, so these values are computed once per fit. Inside dense clusters
the gain is small, but it helps against sparse, high-core regions that late rounds search. Also, inside leaf
scans, skip candidate j as soon as `core_j > best` (strict). That skips the distance computation and helps
most at d ≥ 30.

## 3. CUDA brute-force kNN is under-occupied [measured + code]

`core_distances` runs queries in batches of `NEIGHBOR_BATCH_ROWS = 256` (`estimator.rs:24`). The common
`exact_kernel` assigns one unit per query and uses at most 4 reference groups
(`fast-cubecl-common/src/neighbors.rs`: `REFERENCE_TILE = 128`, `MAX_REFERENCE_GROUPS = 4`), so a batch
launches about 1024 threads. That is under one wave on an RTX 3080 with 68 SMs. Each batch also pays a
readback and a synchronize. Measured result: on CUDA, the 50k×2 BF core stage took **2.2 s, which is slower
than CPU BF's 0.59 s**.

Recommendation [proposal]: size the batch from the scratch planner the code already has (the 32 MiB common
limit allows thousands of rows at small k), and keep the output on the device until the final concatenation.
If Auto switches to KD as in §1, this path matters mainly for n < 256 and for explicit `BruteForce`, so
treat it as second priority.

## 4. Host-side items (minor) [code review]

- **Per-round O(n) host work** in the Borůvka driver: computing labels, building `node_components`, the
  uploads and the candidate readback. At 200k×2 KD (6 rounds) this is measurable but not dominant. If
  §2a/§2b move state onto the device, this traffic shrinks as a side effect.
- **`merge_components` uses `HashMap<i32, usize>`** (`mst.rs:96`). Component ids are DSU roots in
  `0..n`, so a `Vec<u32>` indexed by root with a sentinel avoids hashing. The order of first appearance (and
  with it the tie semantics) is unchanged.
- `condense.rs:180` and `labels.rs:27,123,183` use `HashMap` keyed by dense node or cluster indices. They are
  linear and not hot in any measured case. Change them only together with the item above.
- **CPU runtime, BF Borůvka:** the reference tile decomposition is strided for a GPU. On the CPU runtime,
  cubes become work items, so a cache-blocked contiguous tile would suit it better. This only matters if BF
  stays reachable for large n after §1.

## 5. Suggested order

1. Retune `Auto` toward KD for n > 4096 (§1). This is a configuration and test change with a 3–100× effect
   on measured cases.
2. Device-resident per-row best/target with exact reuse plus the component bound, and active-row compaction
   for the KD candidate kernel (§2a/§2b). Gate it on a device-measured win across blobs and uniform data,
   including the d=50 blob case where reuse is poor.
3. Per-node `min_core` and the `core_j > best` leaf skip (§2c).
4. Batch sizing for CUDA brute-force kNN (§3) and the host cleanups (§4).

Every proposal must keep the MST bit-identical to the current output, compared with the `test-support`
exact MST oracle in `fast-hdbscan-core`. Each one also needs CPU and CUDA runs under
`FAST_UMAP_REQUIRE_CUDA=1` before it counts as validated.
