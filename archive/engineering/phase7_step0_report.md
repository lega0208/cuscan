# Phase 7 Step 0 report: baseline crossover, pruning, cache and API evidence

This report covers Step 0 of [cubecl_kernels_perf_plan_phase7.md](cubecl_kernels_perf_plan_phase7.md). Step 0 changes no production code. It measures only baseline **B** and runs throwaway probes and simulations. All evidence is in [`artifacts/cubecl_kernels_perf/phase7/`](artifacts/cubecl_kernels_perf/phase7/), and `step0_summary.json` there collects every number quoted below. The scope is one host (Ryzen 7 9800X3D under WSL2, RTX 3080 10 GB, driver 617.14) and the fixed Gaussian generator workload at k = 5. None of this is a general platform claim.

## Conclusions

1. **The crossover is real, and it starts earlier at 30D.**
   - **Core distances:** CPU C16 overtakes CUDA between 200k and 300k at 16D (300k: 4.20 s vs 4.69 s) and between 100k and 200k at 30D (200k: 8.17 s vs 10.53 s; 300k: 16.30 s vs 25.86 s).
   - **Full fit:** CUDA still wins everywhere except 300k×30 (CPU 33.72 s vs CUDA 35.40 s), because CUDA MST is 1.2–1.9× faster (1.5–1.9× from 200k).
   - **Labels:** equal in all six cells.
2. **CUDA's cost per scanned pair roughly doubles once the padded data no longer fits in L2.**
   - **CUDA:** at 16D the warm cost is flat at about 0.07 ns/pair through 100k, then rises to 0.130 ns at 200k and 0.160 ns at 300k. At 30D it rises from about 0.11 ns at 20k/50k to 0.18, 0.34 and 0.37 ns at 100k/200k/300k.
   - **CPU:** per-pair cost stays flat (0.14–0.17 ns at 16D, 0.23–0.26 ns at 30D).
   - The crossover therefore comes from CUDA per-pair cost degrading, not from the CPU improving with n.
3. **The L2 hypothesis is supported in this range.** Correlation does not prove causation, but the signals line up:
   - **L2 hit rate:** at 16D it falls 99.8% → 98.2% → 75.7% → 38.2% → 29.2% from 20k to 300k. At 30D it falls 99.6% → 82.2% → 49.3% from 20k to 100k.
   - **DRAM bytes read per scanned pair:** at 16D they rise 0.09 → 1.6 → 22.3 → 60.2 → 65.4. At 300k that is about one 64 B padded row per pair, so L2 provides almost no reuse.
   - **L2 sectors requested per pair are constant** (2.0–3.0 at 16D, 3.9–5.1 at 30D). The kernel requests the same traffic per pair; what changes is where that traffic is served from.
   - **Timing:** the cliff starts when the padded rows reach 6.4 MB (100k×16, 50k×30), compared with the 5 MB L2.
4. **Pruning improves with n at 16D, but only slowly at 30D.**
   - **Leaf positions scanned / n²:** at 16D 0.741 / 0.546 / 0.442 / 0.339 / 0.326 for 20k / 50k / 100k / 200k / 300k. At 30D 0.965 / 0.898 / 0.841 / 0.782 / 0.779.
   - So the "≈ O(n²)" premise holds well at 30D and weakens at 16D. Even at 300k×16, a query still scans a third of all points.
5. **Tile-union simulation (extra analysis, not in the plan).** It reports staged positions per tile as a fraction of n and lane utilization of staged candidates. The headline result is that **a 32-query tile visits nearly every leaf at 16D and 30D**:
   - **16D:** a tile stages 99.9% / 98.2% / 94.4% / 89.2% / 88.1% of all positions at 20k–300k.
   - **30D:** 99.6–100%.
   - **Consequence:** the tree arm T is close to the flat arm F in staged work through 300k, and T adds traversal on top. The plan's reason for preferring T, that its cost does not grow as n², is not supported in the measured range.
   - The simulator reproduces device diagnostics exactly at 20k, and to within ≤12 pops / ≤439 scanned positions out of 10⁸–10¹⁰ above that. Simulated per-lane results match the baseline bit for bit in every checked tile (0 mismatches).
6. **The memory mechanism is the whole bet; the compute side is a wash.**
   - **Compute is similar:** simulated tile lane utilization at 16D (37–74%) is close to the measured baseline SIMT efficiency (8.6–12.4 active threads per issued instruction, i.e. 27–39%).
   - **What changes is memory:** global/L1 candidate traffic per evaluated pair falls by roughly 32 × lane utilization (about 12× at 300k×16, about 25–31× at 30D). DRAM traffic falls from one row per pair to one row per staged candidate per tile.
   - **Limit:** at 20k everything is L2-resident, so the only memory gain there is lower L1/LSU traffic.
7. **Every required API exists, but the plan's lowering citations are wrong.**
   - **Correct lowering:** CubeCL 0.11.1 CUDA compiles through `cubecl-llvm` NVPTX, not the cubecl-cpp dialect. `sync_plane` lowers to `llvm.nvvm.bar.warp.sync(-1)`. `plane_min`/`plane_any` use `activemask` plus `shfl.sync`/`vote.any.sync`.
   - **Shared memory** is dynamic `addrspace(3)`.
   - **Probes:** all API probes pass at cube dims 32/64/128.
8. **Leaves are 40–79 points, not ≤ 40.** `LEAF_SIZE = 40` is a minimum: the level formula yields leaves of 78–79 (20k), 48–49 (50k–200k) and 73–74 (300k) points. Every production leaf therefore takes 2–3 staging chunks at T = 32.

## 0.1 Provenance

- **Source:** HEAD `f208885c817605768fe7ab3a489260801b982ff1`, clean tree, empty `initial.diff`.
- **Toolchain and hardware:** rustc 1.99.0-nightly; 15 logical CPUs visible to WSL; RTX 3080 with 68 SMs and 10,240 MiB; ncu 2026.2.1.
- **Baseline B:** `bin/profile_hdbscan_stages.B` (release + tracing). SHA-256 `769a306b541c2e49201ba7240b04033a9dd5218ef4ac06db914eb7912c2edcab`, byte-identical to Phase 6 `final_clean`. Every runner checks this hash before it executes.

## 0.2 Inputs and tooling

- **Inputs:** `make_inputs.py` generated 20k/50k/100k/200k/300k × 16/30 with `make_input(rows, d, 1.0, 1.0, 42)`. Hashes are in `inputs.json`; the four 20k/50k cells match the Phase 6 hashes.
- **Tool changes:** `analyze/hdbscan_stages_common.py` gained `EXTENDED_ROWS`. `benchmark_hdbscan_stages.py` (`benchmark()` validation and argparse) and `profile_hdbscan_counters.py` (argparse) accept 200k/300k.
- **Tests:** the benchmark parametrization was extended, and rejection and acceptance tests were added. `pytest analyze/tests/test_hdbscan_stages.py` passes (100), and `ruff check`/`ruff format --check` on `analyze/` are clean.
- **Unchanged tools:** `profile_hdbscan_kernels.py`, `summarize_hdbscan_stages.py` and `scripts/analyze_hdbscan_stages.sh` still accept only up to 100k. The plan did not require changing them.
- **Separate Step 0 scripts:** the benchmark tool hardcodes 16 features, and the counter tool's fixed metric set lacks the L2/DRAM metrics. Steps 0.3–0.5 therefore use separate runners (`run_crossover.py`, `run_scan_fraction.py`, `run_l2_counters.py`) instead of modifying production analysis tools.

## 0.3 CPU/CUDA crossover (B)

Each cell ran six processes alternating CPU/CUDA (three per backend), with the starting backend alternating between cells. Each process ran `--runs 3 --min-cluster-size 5 --min-samples 5 --search auto`, and `CUBECL_*` variables were stripped. CPU runs used the CubeCL LLVM runtime with 15 units. Values are medians of three process medians, in seconds.

| Cell    | Core CPU / CUDA | CPU/CUDA core | MST CPU / CUDA | Full fit CPU / CUDA | CPU/CUDA fit | Labels |
| ------- | --------------- | ------------: | -------------- | ------------------- | -----------: | ------ |
| 100k×16 | 0.767 / 0.302   |          2.54 | 0.510 / 0.423  | 1.357 / 0.821       |         1.65 | equal  |
| 200k×16 | 2.336 / 1.762   |          1.33 | 1.542 / 1.041  | 4.082 / 3.027       |         1.35 | equal  |
| 300k×16 | 4.199 / 4.689   |      **0.90** | 3.452 / 1.967  | 8.018 / 6.997       |         1.15 | equal  |
| 100k×30 | 2.187 / 1.511   |          1.45 | 2.341 / 1.345  | 4.709 / 3.045       |         1.55 | equal  |
| 200k×30 | 8.173 / 10.532  |      **0.78** | 7.105 / 4.308  | 15.686 / 15.200     |         1.03 | equal  |
| 300k×30 | 16.300 / 25.861 |      **0.63** | 16.765 / 8.810 | 33.717 / 35.401     |     **0.95** | equal  |

- **Spread:** process medians spread ≤ 2% except for one 100k×16 CPU process at 0.90 s (others 0.73–0.77 s). Tree build is identical across backends (host).
- **Core crossover:** between 200k and 300k at 16D and between 100k and 200k at 30D.
- **Full-fit crossover:** only 300k×30 in this matrix.
- **Status:** this is initial evidence, not an acceptance input.

Warm CUDA ns per scanned pair (core seconds / diagnostic leaf positions scanned):

|          |    20k |    50k |  100k |  200k |  300k |
| -------- | -----: | -----: | ----: | ----: | ----: |
| CUDA 16D | 0.072ᵃ | 0.067ᵃ | 0.068 | 0.130 | 0.160 |
| CUDA 30D | 0.107ᵃ | 0.109ᵃ | 0.180 | 0.337 | 0.369 |
| CPU 16D  |      — |      — | 0.174 | 0.172 | 0.143 |
| CPU 30D  |      — |      — | 0.260 | 0.261 | 0.232 |

ᵃ These come from Phase 5 width-64 CUDA core medians, which use the same CUDA kernel but a different measurement campaign. Use them as trend context only.

## 0.4 Scan fraction (B, CUDA diagnostics)

- **Method:** `FAST_HDBSCAN_KD_DIAGNOSTICS=1`, one warmup plus one measured fit per cell.
- **Determinism:** warmup and measured diagnostics were identical in every cell.
- **Cross-check:** the 20k cells reproduce the historical totals exactly (8,089,849 pops / 296,417,695 positions at 16D).
- **Raw data:** per-row arrays are kept gzipped in `scan_fraction_cuda/`.

| Cell    | Mean nodes popped | Scanned / n² | Per-query scanned min / median / max | Queries scanning all |
| ------- | ----------------: | -----------: | ------------------------------------ | -------------------: |
| 20k×16  |             404.5 |        0.741 | 4,453 / 15,237 / 20,000              |                0.11% |
| 50k×16  |           1,257.1 |        0.546 | 7,270 / 26,692 / 49,462              |                    0 |
| 100k×16 |           2,090.3 |        0.442 | 10,443 / 42,599 / 96,140             |                    0 |
| 200k×16 |           3,316.3 |        0.339 | 14,413 / 64,415 / 189,856            |                    0 |
| 300k×16 |           3,172.9 |        0.326 | 20,484 / 93,053 / 288,860            |                    0 |
| 20k×30  |             498.5 |        0.965 | 7,734 / 19,688 / 20,000              |                23.9% |
| 50k×30  |           1,896.6 |        0.898 | 14,596 / 46,535 / 50,000             |                 1.3% |
| 100k×30 |           3,603.5 |        0.841 | 24,117 / 87,256 / 100,000            |                0.08% |
| 200k×30 |           6,814.3 |        0.782 | 41,880 / 162,093 / 200,000           |               0.003% |
| 300k×30 |           6,752.9 |        0.779 | 70,235 / 241,768 / 300,000           |               0.002% |

- **Trend:** pruning improves with n, strongly at 16D and weakly at 30D.
- **200k → 300k:** pops/query drop at both dimensions, because both sizes have 13 levels and 300k leaves are 73–74 points instead of 48–49.
- **Variability:** per-query work varies by more than 10× within a cell. In the baseline, a warp of consecutive original rows pays for its slowest lane. In the simulator, the per-warp maximum is 0.61–1.0 n against a mean of 0.33–0.97 n.

## 0.5 L2 hypothesis (Nsight Compute on B's `kd_neighbors_kernel`)

**Method.** `run_l2_counters.py` captured the first `kd_neighbors_kernel` launch (the warmup fit) with `--print-units base` and default Nsight replay. Defaults flush caches per pass and lock clocks to base, so durations are not warm timings.

**Cells.** The required cells are 50k–300k × 16D. As cheap extras, the run also covers 20k×16 and 20k–100k × 30D.

**Normalization.** "Per pair" divides by the diagnostic leaf positions scanned for the same cell.

| Cell    | Padded rows | L2 hit | L1 hit | DRAM B/pair | L2 rd sectors/pair | L1 sectors/request | Long-scoreboard | Achieved occ. | Threads/inst | ns/pair (base clk) |
| ------- | ----------: | -----: | -----: | ----------: | -----------------: | -----------------: | --------------: | ------------: | -----------: | -----------------: |
| 20k×16  |     1.28 MB |  99.8% |  60.4% |        0.09 |               2.05 |               11.8 |             67% |         18.8% |         12.4 |              0.061 |
| 50k×16  |      3.2 MB |  98.2% |  52.1% |        1.60 |               2.66 |               10.1 |             76% |         46.1% |         10.3 |              0.051 |
| 100k×16 |      6.4 MB |  75.7% |  50.2% |        22.3 |               2.79 |                9.4 |             81% |         50.9% |          9.6 |              0.069 |
| 200k×16 |     12.8 MB |  38.2% |  47.7% |        60.2 |               2.95 |                8.6 |             94% |         61.9% |          8.7 |              0.134 |
| 300k×16 |     19.2 MB |  29.2% |  48.3% |        65.4 |               2.81 |                8.5 |             95% |         63.5% |          8.6 |              0.164 |
| 20k×30  |     2.56 MB |  99.6% |  55.8% |        0.26 |               3.86 |               15.0 |             75% |         18.9% |         16.5 |              0.092 |
| 50k×30  |      6.4 MB |  82.2% |  50.2% |        28.1 |               4.86 |               14.3 |             88% |         46.9% |         14.7 |              0.088 |
| 100k×30 |     12.8 MB |  49.3% |  48.0% |        85.4 |               5.14 |               13.6 |             95% |         51.2% |         13.8 |              0.193 |

**Resources.**

- **Registers:** 64/thread at 16D and 72 at 30D (width 64). The register occupancy limit is 16 / 14 blocks per SM, so theoretical occupancy is 66.7% / 58.3%.
- **20k:** launch-size limited (313 blocks on 68 SMs), so achieved occupancy is about 19%.
- **Local memory:** traffic from per-lane stacks is 1.1–2.3% of global load sectors (for example 2.75 × 10⁹ vs about 1.57 × 10¹¹ at 300k×16). It is not significant.
- **Branch uniformity:** `smsp__sass_average_branch_targets_threads_uniform` is about 98%.

**Interpretation.**

- **Hypothesis supported, as correlation.** L2 hit rate falls and DRAM bytes per pair rise sharply exactly when the padded rows exceed the 5 MB L2. Warm and base-clock ns/pair roughly double over the same range. Requested L2 sectors per pair stay constant. Long-scoreboard stalls climb to 94–95%, while SM and L1 throughput fall (300k×16: SM 9.9%, L1 28.9%). At large n the kernel waits on DRAM latency.
- **No coalescing.** L1 sectors/request ≈ active threads/instruction, i.e. one sector per active lane. This directly confirms the uncoalesced gather the plan describes.
- **Two regimes.** At 50k×16, L1 throughput is 82% of peak while L2 still hits 98%, so the bottleneck there is the LSU/L1 path, not DRAM. The tile kernel addresses both regimes, through staged broadcasts and through 32× fewer candidate fetches. The 20k cells have neither DRAM nor launch-size headroom.

## 0.6 API probes

All probes passed on CUDA. Details are in `logs/api_probe.log`; the archived test source is `api_probe.rs` (removed from the tree afterwards), with ncu cross-checks in `logs/api_probe_ncu.csv`.

- **Device properties:** `Plane::Ops` and `Plane::Sync` both present; `plane_size_min = plane_size_max = 32`; `max_shared_memory_size = 101,376`; `max_units_per_cube = 1024`.
- **Per-plane shared slices** indexed by `UNIT_POS / PLANE_DIM`, with (planes, chunks) ∈ {(1,8), (2,4), (4,8), (4,1)}.
  - Lane-strided copies of `Vector<f32,4>` are bit-exact, including subnormals and −0.0.
  - The probe includes plane-uniform but cube-divergent round counts under `sync_plane`.
  - SASS performs real `ld.shared`/`st.shared` (270 shared loads, matching 4 planes × 8 chunks). Dynamic shared memory is 16,896 B and static is 0.
- **`plane_min` (f32) and `plane_any`** with masked `+∞`/`false` at cube dims 32/64/128 are correct, including all-inactive planes.
- **`UNIT_POS_PLANE` and `PLANE_DIM`** are correct at cube dims 32/64/128.
- **Generated-code mechanism:** `CUBECL_DEBUG_LOG=<file>` dumps the generated LLVM IR (`logs/api_probe_cubecl_generated.log`).
  - `sync_plane` → `llvm.nvvm.bar.warp.sync(i32 -1)` (`cubecl-llvm-0.11.1/src/nvptx/synchronization.rs`). No `barrier.cta` appears in the probe kernels; `sync_cube` would lower to `llvm.nvvm.barrier.cta.sync.aligned.all`.
  - `plane_min` → `activemask` plus a butterfly of `shfl.sync.bfly.i32` with `llvm.minnum.f32`; `plane_any` → `activemask` plus `vote.any.sync`. The results cover only the lanes active at the call site, so the call sites must be converged.

## Extra analysis: tile-union simulation

**Purpose.** The plan defers the T/F and dimension-gate questions to timing batches. Their outcomes depend heavily on how many leaves a 32-query tile visits, and that can be computed host-side.

**Tool.** `phase7_tile_sim.rs` (binary `bin/phase7_tile_sim`) builds the production `fast_hdbscan_core::KdTree`. It then replays `kd_neighbors_kernel` per query: same push order, `lower <= worst` pruning, `(distance², id)` admission and self-exclusion, with sequential f32 sums.

**Tile model.** The plan's tile traversal is simulated over 32 consecutive leaf-ordered positions: `plane_any` visits, `plane_min` push order, per-lane bounds and own-lane evaluation.

**Validation.**

- Baseline totals equal the device diagnostics exactly at 20k and to within ≤ 12 pops / ≤ 439 positions elsewhere. The residual is FMA in device bounds/distances versus unfused host arithmetic.
- 64–625 tiles per cell were checked lane-by-lane against per-query results (IDs and distance bits): zero mismatches. This supports Invariant 2 (visit order is free) on these data.

| Cell    | Leaf size | Baseline scan/n | Warp-max/n | Tile staged/n | Lane util. | Staged ÷ warp-max | Staged ÷ mean | Tile pops (vs query pops) | Chunks/tile |
| ------- | --------- | --------------: | ---------: | ------------: | ---------: | ----------------: | ------------: | ------------------------- | ----------: |
| 20k×2   | 78–79     |           0.006 |      0.014 |         0.023 |        32% |              1.62 |          3.59 | 28 (18)                   |          17 |
| 20k×8   | 78–79     |           0.175 |      0.411 |         0.716 |        25% |              1.74 |          4.10 | 402 (125)                 |         550 |
| 20k×16  | 78–79     |           0.741 |      0.973 |         0.999 |        74% |              1.03 |          1.35 | 511 (404)                 |         767 |
| 50k×16  | 48–49     |           0.546 |      0.859 |         0.982 |        56% |              1.14 |          1.80 | 2,025 (1,257)             |       2,012 |
| 100k×16 | 48–49     |           0.442 |      0.749 |         0.944 |        47% |              1.26 |          2.14 | 3,935 (2,090)             |       3,866 |
| 200k×16 | 48–49     |           0.339 |      0.628 |         0.892 |        38% |              1.42 |          2.63 | 7,564 (3,316)             |       7,309 |
| 300k×16 | 73–74     |           0.326 |      0.606 |         0.881 |        37% |              1.45 |          2.70 | 7,474 (3,173)             |      10,823 |
| 20k×30  | 78–79     |           0.965 |      1.000 |         1.000 |        97% |              1.00 |          1.04 | 511 (499)                 |         768 |
| 50k×30  | 48–49     |           0.898 |      0.998 |         1.000 |        90% |              1.00 |          1.11 | 2,047 (1,897)             |       2,048 |
| 100k×30 | 48–49     |           0.841 |      0.990 |         0.999 |        84% |              1.01 |          1.19 | 4,092 (3,604)             |       4,090 |
| 200k×30 | 48–49     |           0.782 |      0.979 |         0.996 |        79% |              1.02 |          1.27 | 8,173 (6,814)             |       8,162 |

How to read the columns:

- **Warp-max/n:** the baseline cost floor for a SIMT warp of 32 consecutive original rows (the slowest lane).
- **Staged:** positions copied per tile.
- **Lane utilization:** own-lane evaluations ÷ (staged × 32).
- **Model limits:** this is a work model. It does not capture divergence inside the baseline traversal, which the measured 27–51% threads/instruction shows is worse than warp-max implies. It also does not model memory timing.

**Findings.**

- **16D and 30D:** the tree arm stages 88–100% of what the flat arm stages, and also pays roughly n/T × 2 bound evaluations and local stack traffic per tile. F, with no tree reads, is likely at least as fast as T at 30D at every size and at 16D through 100k. T's staging advantage reaches about 12% only at 200k–300k×16, outside the experimental range.
- **2D and 8D:** the union inflates staged work 1.6–1.7× over the baseline warp-max (3.6–4.1× over the mean), with 25–32% lane utilization. F would stage 100% against 0.6–17% useful, which is hopeless. The simulation predicts that both gates land at 16, unless memory savings exceed about 1.7× at 20k, where data is L2-resident.
- **Union inflation grows with n at 16D** (staged ÷ warp-max 1.03 → 1.45). The tile kernel's compute disadvantage widens with n while its memory advantage also widens. The measured range cannot separate the two.

## Impact on the plan's assumptions

| Plan assumption                                                       | Step 0 finding                                                                                 | Effect                                                                                               |
| --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Crossover near 300k (user observation)                                | Confirmed: 16D core between 200k and 300k; 30D between 100k and 200k; full fit only at 300k×30 | Motivation holds, and is stronger at 30D                                                             |
| "Near O(n²) with traversal overhead" (20k numbers)                    | 16D scan falls to 0.33 at 300k; 30D stays ≥ 0.78                                               | Holds for 30D. At 16D the per-query kernel's useful work shrinks, but the tile union does not (0.88) |
| L2 holds padded rows to about 80k (16D) / 40k (30D)                   | 98% hit at 50k×16, 76% at 100k×16; 82% at 50k×30, 49% at 100k×30                               | Thresholds confirmed within the resolution of the grid                                               |
| L2 hypothesis "unverified"                                            | Supported (correlational): hit-rate cliff, DRAM/pair ×40–700, ns/pair ×2.4                     | Gains at ≥ 100k may be attributed to reduced DRAM traffic only together with post-change H3 counters |
| Global candidate traffic falls about 32×                              | Real factor ≈ 32 × lane utilization: about 15–24× at 16D (≤ 100k), about 27–31× at 30D         | Lower expected traffic reduction at 16D. The direction is unchanged                                  |
| "Prefer T because its cost does not grow as n²"                       | T stages 88–100% of n through 300k at 16/30D                                                   | Rationale not supported in the measured range                                                        |
| Union visits many more leaves at low d                                | 2D/8D: staged ÷ warp-max 1.6–1.7, lane util 25–32%                                             | Confirmed. The gate will very likely be 16 for both arms                                             |
| Leaves longer than T are an edge case to test                         | Every production leaf is 40–79 points (2–3 chunks)                                             | Multi-chunk is the main path. Chunk slot utilization is about 76–82%                                 |
| `sync_plane` → `__syncwarp()` via `cubecl-cpp/src/cuda/dialect.rs:69` | Lowers via cubecl-llvm NVPTX to `llvm.nvvm.bar.warp.sync(-1)`; no cpp dialect in this build    | Citation and the Step 0.6 check criterion are wrong; the design is unaffected                        |
| Per-lane local stacks are acceptable first                            | Baseline local traffic ≈ 2% of global sectors                                                  | Supported                                                                                            |
| Experimental arms capped at 100k                                      | The crossover regime is at 200k–300k; the 100k cells sit at the start of the cache cliff       | Acceptance can show gains, but cannot show whether the crossover moves                               |

## Recommended plan changes

1. **Fix the lowering facts (Changes §1, Step 0.6, risk table).**
   - Replace the `__syncwarp()`/`cubecl-cpp` citation with: `sync_plane` → `llvm.nvvm.bar.warp.sync(i32 -1)` in `cubecl-llvm-0.11.1/src/nvptx/synchronization.rs`.
   - Define the generated-code check as: `CUBECL_DEBUG_LOG` IR contains `bar.warp.sync` and no `barrier.cta` in the tile kernel.
   - Add that `plane_min`/`plane_any` compute over `activemask`. They must be called at converged, plane-uniform points, never inside `if own` or `if active`. Inactive lanes must reach the call and contribute `+∞`/`false`, which the plan already requires.
2. **Rebalance T versus F (Step 5).**
   - Drop "prefer T because its cost does not grow as n²" as a tie-break. Select purely by geometric mean, or prefer F on a tie: it is simpler, reads no tree and has no traversal divergence.
   - Keep T as an arm, because 200k–300k×16 shows a 12% staging advantage that could grow with better pruning.
   - Record the simulation's prediction (F ≈ T or faster at 16/30D ≤ 100k) before Step 3, so the selection batch can test it.
3. **Treat multi-chunk leaves as the main path (Traversal, Shared memory, Step 2).**
   - Note that production leaves are 40–79 points (`LEAF_SIZE` is a minimum), i.e. 2–3 chunks with 76–82% slot utilization.
   - Optionally add a whole-leaf staging variant (80 slots per plane: 10.6 KB at `d_chunks = 8`, 5.4 KB at 4, including IDs). Weigh its shared-memory/occupancy cost against fewer syncs only if H3 shows `sync_plane`/staging overhead. Keep T = 32 chunking as the default.
   - The `n = 4109` "last chunk begins at 4108" case applies to F's global chunks only. T's chunks are leaf-relative (64 leaves of about 64 points at that n), so add a T case whose final leaf ends mid-chunk.
4. **Dimension gate (Step 3).** Keep the measured gate selection, but record the predicted outcome: gate = 16 for both arms at 20k. If 8D unexpectedly passes for T, report it against the 1.74× staged ÷ warp-max work inflation. F at 2D/8D can be skipped as structurally hopeless (100% staged against 0.6–17% useful), which saves selection time. The plan's rule can stand if the user prefers measurement over prediction.
5. **H3 attribution (Step 4).**
   - Add L2 hit rate, `dram__bytes_read.sum` per scanned pair, `lts__t_sectors_srcunit_tex_op_read.sum` and threads/instruction to the post-change counters at 50k/100k×16, plus 100k×30, where the baseline L2 hit is 49%.
   - Normalize by staged positions × 32 for the new kernels.
   - At 50k×16, attribute any gain to L1/LSU traffic (L2 hit 98%), not to L2.
6. **Resolved (2026-10-27):** the user re-added 300k as an experimental cell with size-priority acceptance. The plan now measures every arm at 300k. Original recommendation, kept for the record: **Reporting-only large-n check (Step 4, needs user approval).** The motivating crossover lies at 200k–300k, outside the 100k experimental cap. Consider one reporting-only, non-acceptance measurement: selected arm vs B vs CPU C16 at 200k/300k × 16/30, three processes each. Otherwise the phase cannot say whether the crossover moved. This contradicts Changes §6 as written, so it requires an explicit user decision; without it, the report must state that the crossover question remains open.
7. **Guard expectations (Step 4).** At 20k neither the DRAM regime nor launch-size headroom applies: about 19% achieved occupancy from only 313 cubes, and the tile kernel keeps the same warp count. Expect "no effect" or a small regression there. The existing guard thresholds stand.
8. **Analysis tooling.** The Phase 7 runners kept 200k/300k out of `profile_hdbscan_kernels.py`, `summarize_hdbscan_stages.py` and `scripts/analyze_hdbscan_stages.sh`. If item 6 is approved, extend those choices with tests as well. Otherwise no change is needed.

## Reproduction

From `artifacts/cubecl_kernels_perf/phase7/`:

- `uv run --no-sync python make_inputs.py`
- `uv run --no-sync python run_crossover.py` (about 40 minutes)
- `uv run --no-sync python run_scan_fraction.py cuda`
- `uv run --no-sync python run_l2_counters.py`
- `bin/phase7_tile_sim INPUT ROWS FEATURES 5 --check-tiles N`. Build it as a standalone crate that depends on `crates/fast-hdbscan-core`; the source is `phase7_tile_sim.rs`.

Each runner checks the B and input SHA-256 hashes before running and refuses to overwrite existing output directories. The tile simulator's source and binary hashes are in `step0_summary.json`.
