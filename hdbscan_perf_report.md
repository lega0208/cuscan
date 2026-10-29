# HDBSCAN core-distance performance investigation

## Conclusions

1. **Optimize the KD neighbor traversal, not core extraction or readback.** The current 16-feature Auto route is KD-tree on both CPU and CUDA. At 100k, core distances take **4.657 s CPU / 1.177 s CUDA**, respectively 88% / 65% of the traced fit. More than 99% of the CPU core stage and about 99% of the CUDA core stage is the synchronized neighbor-search scope.
2. **Pruning effectiveness is the dominant algorithmic warning.** At 20k x 16, each query scans an average 14,821 leaf positions: **74.1% of the all-pairs domain**. At 30 features it scans 96.5%; at 2 features only 0.63%. KD is not inherently subquadratic on these workloads.
3. **The best supported CPU opportunity is spatial locality with original IDs preserved.** Alternating input-order experiments reduce core time by 17% at 20k and 22% at 50k. This is evidence of sensitivity, not a delivered implementation. The experiment simultaneously changes query and candidate layout/order; those effects remain unseparated.
4. **CUDA needs latency hiding and a better access pattern, not merely fewer launches.** Current KD uses one neighbor launch per fit, not the tens of thousands of brute-force launches in the older timing notes. Counters show substantial long-scoreboard stalls, low active warps and high L1/TEX utilization. Larger blocks and cooperative leaf evaluation are concrete prototypes to test, not measured speedups.
5. **Do not deploy the experimental whole-input permutation.** After undoing it, public core distances and neighbor IDs agree exactly at 20k and 50k on both backends, but final clustering partitions differ. Internal scheduling/layout must preserve original IDs throughout MST and extraction.

The initial source investigation did not change production code; later kernel experiments and their outcomes are recorded in the phase sections below. The source investigation itself delivered a plan, measurements, semantic checks and ranked optimization designs, not an optimized kernel.

## Scope, provenance and methodology

The source-only plan was written first: [hdbscan_perf_investigation.md](hdbscan_perf_investigation.md). Experiments followed it, adding lower-dimensional controls and a semantic probe when input-order sensitivity appeared.

- Revision: `2c3fc35fa2b2492f519c0adcb544779357f46dd8`. Initial working tree clean; production-source diff remains empty. Benchmark manifests say dirty because the requested plan and generated evidence were untracked.
- Burn **0.22.1**, CubeCL **0.11.1**, CubeK **0.3.1** from `Cargo.lock`. Installed CubeCL 0.11.1 CPU threadpool source was inspected, rather than assuming the skill's 0.11.2 examples apply unchanged.
- CPU: AMD Ryzen 7 9800X3D, 15 logical processors exposed to WSL2. CUDA: **NVIDIA GeForce RTX 3080, 10,240 MiB**, driver **617.14**. Both backends actually executed.
- Release tracing example, one excluded warmup and three measured fits per process. CPU/CUDA runs and experiments were serial, not competing benchmark lanes. Native scopes synchronize the neighbor operation. They include dispatch, allocation and synchronization; they are not isolated device-kernel times.
- Fixed standard input: 8 Gaussian centers, 16 features, center/noise scales 1, seed 42, contiguous f32. `min_samples=5`, `min_cluster_size=5`, default graph/search Auto. At 20k, 19,138 points are noise: this is a heavily overlapping/high-noise workload, not a benchmark of well-separated clusters.
- Iteration used 20k, confirmation used 50k; 100k was used only for final baseline/scaling validation. All recorded benchmark/profile processes completed without timeout.
- Nsight Systems **2026.1.3** measures device launches in one measured fit. Nsight Compute **2026.2.1** samples the first matching, excluded-warmup launch with replay; its counters are representative, not averages over measured fits. Profiler timing is kept separate from baseline wall timing.

Evidence root: [`artifacts/hdbscan_core_investigation/`](artifacts/hdbscan_core_investigation/). It contains raw input, labels, traces, stdout/stderr, Nsight databases/CSV, scripts, `provenance.json`, `evidence_summary.json`, and the permutation semantic probe. These are local untracked artifacts; the two requested Markdown files are the primary deliverables. Provenance includes exact source/binary SHA-256 values, compiler/tool identities and an explicitly post-experiment source receipt.

## What this stage actually does

Relevant implementation:

| Source | Responsibility |
| --- | --- |
| `crates/fast-hdbscan/src/estimator.rs:124–133` | Resolve complete-graph neighbor search: KD for >=256 finite rows and <=30 features |
| `estimator.rs:384–547` | Exact neighbor search, device core extraction, host neighbor/distance transfer and seed-core assembly |
| `estimator.rs:549–572` | Consume compact neighbor IDs/distances for MST seeding |
| `crates/fast-hdbscan/src/kernels/kdtree.rs:221–422` | Single-unit-per-query near-first traversal, output-backed max-heap and final heapsort |
| `kdtree.rs:429–447,586–635` | CPU/CUDA launch geometry and allocations |
| `crates/fast-hdbscan-core/src/kdtree.rs:52–125` | Balanced implicit tree, widest-axis partitions and deterministic original-ID tie keys |
| `crates/fast-cubecl-common/src/kernels/neighbors.rs` | Exact brute-force CPU path and squared CUDA partial-list/merge path |

The KD kernel reads contiguous row-major feature vectors through `tree.indices`; points belonging to a leaf are not packed into a contiguous data region. Query threads follow input order. Every thread maintains private 33-entry node/bound stacks, computes scalar feature distances and child-box bounds, and updates a heap in its global output row. CPU uses one cube with reported CPU-core-count units and cyclic query assignment. CUDA uses 32-thread blocks, one query per thread.

The tree's `LEAF_SIZE=40` determines level count, not a strict maximum population. At 20k the tree has 511 nodes and 256 leaves, averaging about 78 points per leaf. Traversing 404 nodes and scanning 14,821 positions per query therefore represents weak pruning, not a tiny residual search.

Core extraction selects squared-distance column `min_samples-1`; special cases produce zero for `min_samples==1` and infinity for `min_samples>=n`. The host needs all compact neighbors/distances for MST seeds and public neighbor output. Replacing this stage with just a kth-distance reduction would remove required work, not optimize equivalent behavior.

Tree construction and upload are outside `core_distances`. Extraction is an asynchronous tensor slice; realization may be charged to a later readback. Fine timings are disjoint wall scopes nested inside the inclusive core stage, not additive to it. Above 30 features, default graph routing instead selects the separate kNN/ANN path, so these findings must not be generalized to that path.

## Baseline scaling

Median seconds, three measured fits after warmup:

| Rows | CPU fit | CPU core [min–max] | CUDA fit | CUDA core [min–max] |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 0.3430 | 0.2724 [0.2624–0.2943] | 0.1625 | 0.07421 [0.07402–0.07521] |
| 50,000 | 1.5949 | 1.3936 [1.2900–1.3983] | 0.6626 | 0.38625 [0.38427–0.38705] |
| 100,000 | 5.2866 | 4.6565 [4.3844–4.7280] | 1.8090 | 1.17654 [1.17548–1.19866] |

Increasing rows fivefold increases core time approximately 17.1x CPU / 15.9x CUDA. This is poor scaling consistent with substantial scan work; three sizes on one distribution do not establish an asymptotic complexity law.

Fine-stage medians in milliseconds:

| Rows/backend | Neighbor operation + sync | Extraction enqueue | Neighbor readback/conversion | Final host assembly |
| --- | ---: | ---: | ---: | ---: |
| 20k CPU | 271.344 | 0.024 | 0.556 | 0.020 |
| 20k CUDA | 73.318 | 0.012 | 0.477 | 0.018 |
| 100k CPU | 4654.905 | 0.026 | 0.878 | 0.082 |
| 100k CUDA | 1169.090 | 0.026 | 6.653 | 0.082 |

Independent medians need not sum exactly to the inclusive median; span-boundary overhead also remains. At 100k/k5, compact readback is only 4 MB of IDs plus distances. Transfer/extraction optimization cannot materially address multi-second traversal time on this workload.

The benchmark runner verified **exact CPU/CUDA final-label equality at all three sizes**.

## Discriminating experiments

### KD versus the current exact brute-force comparator

20k, same generator/configuration, median core and full-fit seconds:

| Features/backend | KD core | Brute core | KD fit | Brute fit |
| --- | ---: | ---: | ---: | ---: |
| 16 CPU | 0.2724 | 0.3140 | 0.3430 | 1.3818 |
| 16 CUDA | 0.07421 | 0.54023 | 0.1625 | 0.7557 |
| 30 CPU | 0.6425 | 0.6112 | — | 2.9368 |
| 30 CUDA | 0.10605 | 0.76074 | — | 1.2456 |

CPU repeat baseline at 16D was 0.2898 s core; its advantage over brute force is modest and variable. At 30D the brute-force core is only about 5% faster in this single comparison, not sufficient evidence to change routing. Forced search also changes **MST search**, making full-fit ratios unsuitable as isolated core-distance speedups. KD/brute final labels agree for both feature counts and both backends.

CUDA's approximately 7x KD advantage over the current brute comparator does **not** mean KD prunes well. Brute batches 256 queries, does a transaction per batch and launches partial/merge kernels. A well-tiled brute implementation remains a different, unmeasured comparator. Source and measurements do not justify making brute force the current default.

### Dimensionality and k sensitivity

20k, Auto/KD, k5; diagnostic work counts collected separately from timing:

| Features | CPU core (s) | CUDA core (s) | Mean nodes popped / 511 | Mean leaf positions | Leaf positions / n² |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.00239 | 0.00182 | 18.3 | 126 | 0.63% |
| 8 | 0.03710 | 0.01746 | 124.6 | 3,490 | 17.45% |
| 16 | 0.27241 | 0.07421 | 404.5 | 14,821 | 74.10% |
| 30 | 0.64249 | 0.10605 | 498.5 | 19,301 | 96.50% |

These use the same generator parameters/seed at different feature counts, not nested feature subsets. Low-dimensional pruning works extremely well; the 16D result is not universal. CPU/CUDA labels match exactly for each dimensional workload.

At 16D, k5 -> k15 -> k30 core medians are **0.272 -> 0.322 -> 0.330 s CPU** and **0.0742 -> 0.0798 -> 0.0869 s CUDA**. k30 scans 86.69% of leaf positions and pops 458.3 nodes/query, versus 74.10% and 404.5 at k5. CPU/CUDA diagnostic totals agree exactly for k5 and k30. k changes clustering semantics; label equality with k5 is neither expected nor required.

This does not isolate heap cost: a larger k also weakens pruning. A sixfold heap width increase produces a much smaller timing increase, so the global heap is **not demonstrated to be the main bottleneck**. Instrument admissions/sift steps or benchmark a local-heap prototype before prioritizing it above traversal/locality.

Splitting original diagnostic leaf work by `row % 15` yields max/mean work **1.012 at k5**, **1.007 at k30**. This is a work-count proxy, not per-thread elapsed time or a direct runtime-core-count capture. It does not support a large CPU load-imbalance problem. Installed CubeCL 0.11.1 `threadpool/mod.rs:90–106` submits one task per unit; adding CPU cube count alone does not create more unit tasks.

### Spatial ordering: supported CPU opportunity, limited CUDA result

The experiment recursively sorts by widest-axis medians, packs the same points in that order and runs the unchanged estimator. It changes both query coherence and candidate memory/within-leaf ordering. It does not isolate either mechanism.

A follow-up alternated original/spatial processes in `O,S,S,O,O,S` order. Each process had its own excluded warmup and three measured fits: nine samples/order/backend, across three process groups. Samples within a process are correlated; these are not nine independent statistical trials.

Core-stage medians and ranges in milliseconds:

| Rows/backend | Original | Spatial | Median reduction |
| --- | ---: | ---: | ---: |
| 20k CPU | 266.47 [256.50–285.61] | 221.53 [216.70–228.83] | 16.9% |
| 50k CPU | 1356.07 [1244.84–1444.37] | 1054.72 [1031.02–1169.99] | 22.2% |
| 20k CUDA | 61.72 [61.06–63.56] | 56.22 [55.65–57.47] | 8.9% |
| 50k CUDA | 319.73 [313.65–331.33] | 312.24 [297.59–329.89] | 2.3%, overlapping ranges |

The initial unpaired 50k CUDA ordering comparison was actually slower (0.3999 vs 0.3862 s). The paired confirmation shows only a small, noisy difference. **A robust CUDA ordering win at 50k is not established.** Later baseline timings also differ from the initial scaling campaign; compare candidates within their interleaved batch, not against whichever historical baseline gives the largest ratio.

At 20k, original and ordered diagnostics have **identical total nodes popped and leaf positions scanned**: 8,089,849 and 296,417,695. The timing difference is not explained by reduced aggregate traversal. Cache locality, lane coherence and heap-admission ordering remain possible mechanisms.

Permutation construction plus materializing ordered input took **25.6 ms at 20k / 59.7 ms at 50k**, excluding file IO. The CUDA 20k core saving is only 5.5 ms, less than that preparation cost. CPU savings exceed this Python preparation cost, but neither comparison is an end-to-end, output-preserving implementation benchmark. A production design can reuse the already-built KD permutation rather than rebuild it in Python.

**Semantic probe:** on CPU and CUDA at both 20k and 50k, after undoing row permutation and remapping neighbor IDs, there are **zero mismatched neighbor slots and zero bitwise mismatches in public Euclidean core distances**. This establishes stage-output equality for these inputs, not equality of every internal squared-distance bit.

Nevertheless final partitions have ARI **0.99997165 at 20k** and **0.75024177 at 50k** versus the original order. CPU/CUDA agree for the ordered input. These comparisons and exact source-label paths are recorded in [`partition_comparison.json`](artifacts/hdbscan_core_investigation/partition_comparison.json). Since public core/neighbor outputs match, the discrepancy is downstream of those observable outputs, not evidence of a core-search error. The precise downstream tie/order mechanism was not localized. Therefore whole-input reordering is **rejected as a semantics-preserving production optimization**. Keep original IDs and all downstream indexing intact in any internal-layout prototype.

## CUDA profiling: what the counters do and do not show

Nsight Systems sees **one** KD-neighbor launch per measured fit:

| Rows | Grid x block | Neighbor device time |
| ---: | --- | ---: |
| 20k | 625 x 32 | 74.07 ms |
| 50k | 1563 x 32 | 387.57 ms |
| 100k | 3125 x 32 | 1184.95 ms |

Representative Nsight Compute percentages:

| Rows | Active warps | SM throughput | DRAM throughput | L1/TEX throughput | Long-scoreboard stalls |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | 18.53 | 22.30 | 1.36 | 83.68 | 69.77 |
| 50k | 27.89 | 24.37 | 4.84 | 80.82 | 74.27 |
| 100k | 31.68 | 27.22 | 32.73 | 86.52 | 79.07 |

At 20k, additional LaunchStats/Occupancy capture reports **71 registers/thread**, a **16-block/SM limit**, **33.33% theoretical occupancy**, **18.53% achieved occupancy**, and **0.57 waves/SM**. One-warp blocks encounter the block residency limit well before the warp limit. At 20k, the total of only 625 warps also limits device-wide parallelism; increasing block size alone does not create more query work or warps. At larger sizes, grouping multiple planes per block could raise the block-limit occupancy ceiling, subject to register allocation and scheduling.

Together, these measurements support a memory-dependency/latency-hiding investigation. They do **not** prove a single cause:

- High L1/TEX and scoreboard stalls are not direct measurements of coalescing, cache hit rate or stack spilling. Sector/request and local-memory spill metrics were not collected.
- Low DRAM throughput at 20k does not imply kernel-launch overhead; one long kernel performs the search. DRAM utilization also rises substantially at 100k.
- Packing rows into leaf order helps within-thread/cache locality but does not automatically make different CUDA lanes load adjacent addresses. Query coherence or a cooperative candidate mapping is needed to address that separately.

The old [hdbscan_timing_findings.md](hdbscan_timing_findings.md) explicitly describes Auto resolving to brute force. Its launch-count diagnosis is historical and **does not describe the current default stage**.

## Ranked optimization opportunities

All proposed kernel changes below are **unimplemented/unbenchmarked**. Priorities reflect the evidence, not promised speedups.

### 1. CPU: packed leaf storage and internally ordered queries

Maintain `packed[position, feature] = data[tree.indices[position], feature]`, plus original-ID mapping. Query via a work-order permutation, but write output row `original_query_id` and compare heap/tie keys using **original candidate IDs**, never packed positions. Retain existing tree bounds and original-index downstream arrays. Do not permute estimator input/public finite-row indexing.

Start with packed candidates only, then query ordering only, then combine: the current experiment cannot separate them. Keep the CPU grid bounded; compare cyclic and contiguous query chunks only after layout is coherent. Extra packed input is 6.4 MB at 100k x 16, plus mapping/work-order arrays. Include packing/allocation/upload cost in fit comparisons and update resource sizing before delivery. Reuse existing KD construction; do not add a second tree builder.

**Evidence:** repeatable CPU sensitivity, equal public core/neighbor outputs, unchanged aggregate search counts at 20k. **Risk:** original-ID ties, self exclusion, output scatter and downstream component/MST indexing. No guaranteed equivalent speedup from this design has been measured.

### 2. CPU first, CUDA separately: reuse query values and vectorize feature IO

Prototype fixed/specialized feature-width paths that load query features once per query and reuse them in point and box distances. Select legal vector widths from the actual pinned API/device and dense layout; preserve explicit scalar tails for dimensions such as 3 or 30. Keep a generic scalar path for arbitrary features.

On CPU, SIMD is a credible target for contiguous feature arithmetic; compare a plain Rust/Rayon exact baseline over host-resident data to expose JIT/buffer/scheduling costs. On CUDA, test narrower vector loads independently: wider live vectors can worsen an already 71-register/thread kernel. Preserve sequential scalar accumulation when bitwise distance/tie behavior requires it; an unordered vector or plane sum is not automatically numerically equivalent. Generated-code inspection is needed to establish which loads LLVM/CUDA already hoist.

**Evidence:** hundreds of millions of scalar point comparisons and source-level repeated query/bounds accesses. **Limit:** no SIMD/cache-reuse variant was compiled or benchmarked here.

### 3. CUDA: multi-plane blocks, then cooperative leaf evaluation

First sweep 32/64/128-thread blocks without changing query semantics, at 20k and 50k. Measure theoretical/achieved occupancy, registers, spills, device time and full fit. Do not assume a block-size increase helps the underfilled 20k case.

If memory dependencies remain dominant, prototype a subgroup/coherent query-tile formulation: parallelize candidate rows within a leaf, compute deterministic `(distance, original ID)` partial top-k lists, then merge explicitly. Preserve scalar feature reduction order or define/test numerical tolerances before changing it. This offers actual extra work-level parallelism rather than merely repackaging the same warps. Cache/stage leaf feature blocks only where multiple queries can reuse them; not as unconditional shared-memory overhead.

Plane features/width must be queried, not assumed 32 for every backend. CUDA cooperative code should remain distinct from CPU's barrier-free per-query formulation. Use separate dispatches for any cross-cube combining; no counter-only publication/spin protocol.

**Evidence:** occupancy ceiling, 70–79% long-scoreboard stalls, weak pruning. **Risk:** register/shared-memory pressure, deterministic merging and GPU-specific synchronization. CPU correctness alone cannot validate this design.

### 4. Both: optimize bounds/search strategy for weak-pruning regimes

Measure a leaf-size/depth sweep: smaller leaves mean more node-bound work but fewer unnecessary point scans. At 16–30D either tradeoff can lose. A cheap split-plane lower bound is safe only as an actual lower bound; a looser bound preserves correctness but may increase visits. Replacing tight boxes or incrementally updating bounds needs a proof against the actual tight-box representation, not an assumption that only one coordinate changes.

Compare against a **new tiled exact distance + top-k** baseline when scan fractions approach 75–97%. Reuse reference tiles and keep only bounded partial top-k lists; do not materialize an n² distance tensor. On CUDA, increase useful query/reference parallelism and reduce per-256-row dispatch/readback cycles. Keep shared neighbor functionality in `fast-cubecl-common`, not a dependency on UMAP.

**Evidence:** dimensionality sweep and near parity of CPU KD/brute core at 16–30D. **Limit:** no evidence supports changing the current routing thresholds yet; dimensionality alone cannot predict pruning, and current CUDA brute is much worse.

### 5. Both: local small-k heaps, only after isolating their share

Prototype comptime small-k local arrays and a cached heap root, writing sorted IDs/distances once. Retain a dynamic-width path for unrestricted `min_samples`. Measure admissions, sift depth, registers and spills. This is a plausible reduction in global heap traffic, but k sensitivity does not establish it as the leading cost and CUDA's existing private stacks/register demand make indiscriminate caching risky.

### Low priority / rejected as primary fixes

- KD readback or core-extraction fusion: tiny measured share; all seed neighbors are still needed.
- More CPU cubes or dynamic scheduling to cure presumed imbalance: no supporting work-count evidence.
- Launch reduction for the current KD neighbor kernel: already one launch.
- Host Burn sort/all-pairs tensor composition as a presumed GPU primitive: it neither supplies the KD access pattern nor removes n² storage/selection costs. A custom access-pattern kernel is justified here, but new scan/sort APIs must be verified in the pinned dependencies rather than invented.

## Follow-ups: CubeCL kernel patterns for `core_distances`

**Status: source analysis only.** I compiled, ran, profiled and inspected generated code for none of the items below. They refine opportunities 1, 2, 3 and 5 above using CubeCL kernel-engineering practice. The analysis read `kd_neighbors_kernel` and `box_lower_bound` (`crates/fast-hdbscan/src/kernels/kdtree.rs:19–39, 221–447`), `core_distances` (`estimator.rs:384–547`), tree construction (`fast-hdbscan-core/src/kdtree.rs:52–128`) and the pinned CubeCL **0.11.1** frontend. The 0.11.1 source provides `Vector`, `Shared::new_slice`, and `plane_any`, `plane_min`, `plane_ballot`, `plane_broadcast` and `plane_shuffle`. The repo already uses fixed-width `Tensor<Vector<u32, Const<4>>>` buffers with `#[unroll]` lane extraction in `kernels/component_reduce.rs`; new vector code should follow that pattern.

### Invariants every variant must keep

- **The result is independent of visit order, but distance bits are not guaranteed.** The heap keeps the k smallest `(squared distance, original candidate ID)` keys under a total order, and pruning uses `lower <= worst`. Changing the order in which nodes, leaves or candidates are visited therefore cannot change the output, provided each pairwise distance has the same bits. That makes query reordering, tile traversal and candidate vectorization semantically safe. Reordering the sum over **features** is not safe, because exact-equality tie-breaks and the bitwise host-oracle tests depend on the sequential `feature = 0..d` f32 accumulation. Plane/vector horizontal sums over features are excluded unless a tolerance contract is adopted and documented. FMA contraction can also differ between the scalar and vector code paths. Check bitwise equality against the current kernel; do not assume it.
- **Zero-padding features is bitwise-neutral.** If both points pad unused features with `0.0`, each padded term adds `(0 - 0)^2 = +0.0` to a non-negative partial sum. Padded box features with `min = max = point = 0` likewise add `+0.0`. The KD route only accepts `d <= 30`, so padding to a multiple of 4 gives at most **8 `Vector<f32, 4>` chunks**. That bounded set of comptime specializations covers every KD-routed input.
- Keep original IDs for self-exclusion, tie keys and output rows (see opportunity 1 and the permutation probe). Packed positions are a memory layout only.

### Source-level traffic accounting (why latency, not FLOPs, dominates)

The counts below are global/L1 loads per scanned leaf position at 16D, read from the source. They are not measured:

| Load | Count | Needed? |
| --- | ---: | --- |
| `indices[position]` | 1 | yes; it is the gather that the packed layout removes |
| query features `data[row*d + f]` | 16 | **no**: the values are the same for the whole query |
| candidate features | 16 | yes |
| heap root `out_distances[root]`, `out_indices[root]` | 2 (+1 on equality) | **no**: the value changes only on admission |

About half of the loads in the leaf loop are therefore redundant. Each internal-node pop adds `is_leaf[node]`, two child lower bounds of `16 query + 32 bound` loads each, and up to three more reads of the global heap root (`kdtree.rs:339–359`). The feature loop bound is the runtime `data.shape(1)`, so the loop cannot be unrolled or kept in registers. The generated code was not inspected. Unless it marks the buffers no-alias, stores into `out_distances`/`out_indices` can prevent the compiler from hoisting `data` reads across heap updates. On CUDA, all of these loads are L1TEX dependencies, and that is consistent with the measured 70–79% long-scoreboard stalls. Dynamically indexed `stack`/`bound_stack` arrays normally live in local memory, which is also L1TEX. Treat this as a hypothesis until `local_load`/spill and sectors-per-request counters are collected.

### F1. Both backends: register-resident query, comptime feature width (lowest risk)

Add a `#[comptime] d_chunks: usize` (1..=8) specialization over a zero-padded `Tensor<Vector<f32, Const<4>>>` copy of the data, padded bounds and an unpadded original-ID view. Load the query once into `d_chunks` vectors and use it for both `box_lower_bound` and the point distances. Inside each chunk, accumulate lanes in feature order with `#[unroll]` `extract(lane)`, so 128-bit loads do not reorder the sum. Keep the current generic kernel for the scalar/test path or remove it once the padded path covers all KD inputs. KD inputs are already restricted to `d <= 30`, so the generic kernel may become dead code.

Register budget on the RTX 3080 (sm_86: 64K registers, 16 blocks/SM, 1536 threads/SM). At 32-thread blocks, the 16-block cap limits residency to 512 threads/SM. That allows up to about 128 registers/thread **with no occupancy loss**, so 16–32 extra query registers on top of the measured 71 cost nothing at the current geometry. Larger blocks change this. At 71 registers (allocated as 72 per thread, 2,304 per warp), 64-thread blocks reach about 28 warps (~58% theoretical occupancy). At about 104 registers, the ceiling falls to about 19 warps (~40%). Sweep F1 and the block size together, not independently. The `cubecl-cpu` backend runs each unit as a host task, so register-resident vectors map directly to SIMD registers there.

### F2. Both backends: cache the heap root and use a register top-k

This step is cheaper than the full local-heap redesign in opportunity 5. Keep `worst_distance`/`worst_index` in registers, compare candidates and child bounds against the registers, and touch the output heap only on admission. After a sift, the root is the new worst entry. This removes two dependent global loads per scanned candidate and up to three per internal node without changing semantics.

Next, for comptime `K <= 16` (current benchmarks use k5), replace the heap with a sorted register list that uses unrolled compare-and-shift insertion keyed by `(distance, id)`. Write the list to `out_*[original_row]` once at the end, which also removes the final heapsort. Every index into the list is comptime, so it stays in registers. A heap with dynamic sift indices would go to local memory. Keep the existing output-backed heap as the fallback for large/unrestricted `min_samples`. Measure registers and spills on CUDA, because K=16 adds 32 live values.

### F3. Both backends: compact node records and implicit tree arithmetic

Each node pop reads `is_leaf`, `starts` and `ends` from three separate buffers, and the bounds live in two `[nodes, d]` arrays. Pack `(start, end, is_leaf, pad)` into one `Vector<u32, Const<4>>` record, which is a single 16-byte load. Interleave the padded bounds per node as `[min chunks | max chunks]`, so the box bound for a node reads one contiguous span. Sibling children `2i+1`/`2i+2` are then adjacent as well. For KD-routed inputs (`n >= 256`), `LEAF_SIZE = 40` means every internal node holds at least 2 points, so leaves are exactly the last level. Derive this from the builder, not by assumption. The tree must still support the small-`n` early leaves at `kdtree.rs:107`, so keep the flag unless a validated `first_leaf` scalar replaces it on the KD route.

### F4. CPU: SoA leaf blocks, vectorized across candidates

This extends opportunity 1 (packed leaves). Store each leaf as a transposed block `leaf_soa[leaf][feature][slot]`, with the slot count padded to the selected width `W`. Take `W` from `client.io_optimized_vector_sizes(4)` at setup, not from `load_width / 32`, and pass it as comptime. Each `Vector<f32, W>` lane is a different candidate and accumulates its own features `0..d` in order. This vectorizes the distance computation **without reordering any pair's sum**, so the bitwise invariant holds. It also needs no feature-axis tail for `d = 3` or `30`. Tails go on the candidate axis. Either give padded slots an explicit validity mask, or give them ID `n` with an infinite distance. A sentinel is not admitted under the current `(distance, id)` comparisons, but an explicit mask is clearer. For admission, run one vector compare against the broadcast `worst`, then do an unrolled per-lane scalar check. Once the heap fills, admissions should be rare relative to the 14,821 positions scanned per query, so most vectors would take the cheap reject path. This is expected, not measured; the admission instrumentation from opportunity 5 would confirm it. Packing costs `n * d_padded * 4` bytes, or 12.8 MB at 100k x 32-padded. Charge the packing build to the fit.

Also replace cyclic query assignment (`local_row += CUBE_DIM * CUBE_COUNT`) with contiguous chunks of **leaf-ordered** queries, iterating `tree.indices` and writing to `out[original_row]`. The skill's CPU guidance is "one unit per contiguous slice of queries; sorting queries still helps the cache." Consecutive queries then share their traversal paths and hot leaf blocks in each core's L1/L2. Benchmark this separately from F4, as opportunity 1 requires. Shared memory has no on-chip advantage on the CPU runtime, and `PLANE_DIM` is 1, so the CPU kernel should stay barrier-free and per-query.

### F5. CUDA: plane-cooperative leaf-tile traversal (largest change)

The current kernel runs 32 incoherent per-thread traversals per warp. The data shows three related costs: divergent stacks, uncoalesced gathers through `indices` and repeated L1 misses on the same leaves. At a 74–97% scan fraction, almost every warp lane visits most leaves anyway. The proposed **separate kernel**, gated on `features.plane` and the queried `plane_size_min`, works as follows:

1. **Tile:** one plane (one cube of one plane at first) owns `PLANE_DIM` consecutive queries in leaf order, i.e. consecutive positions in `tree.indices`. Each lane keeps its own query in registers (F1) and its own register top-k (F2).
2. **Uniform traversal:** the stack is cube-uniform and held in `Shared`. For each node, every lane computes its own `lower(q, node) <= worst(q)`, and `plane_any` decides whether to visit. Children are ordered by the plane-min lower bound (`plane_min`). Control flow and barriers are uniform, so `sync_cube()` is never masked. Some extra leaves are visited, which costs time but does not change results (see the visit-order invariant).
3. **Staged candidates:** copy each visited leaf into `Shared` in chunks of `PLANE_DIM` candidates × `d_chunks` vectors (32 × 8 × 16 B = 4 KB). Each lane loads one candidate's contiguous padded row, so a warp reads contiguous leaf-packed memory. Synchronize, then each lane loops over the staged candidates. Every lane reads the same shared address, which is a broadcast with no bank conflicts, and accumulates in feature order. Synchronize again before the next chunk overwrites the buffer, to avoid a write-after-read race. Each staged candidate is reused `PLANE_DIM` times instead of being re-gathered per lane. Double-buffering with `memcpy_async`/barriers is a later step and needs the 0.11.1 API checked first.
4. **Output:** each lane writes its sorted top-k to `out[original_row]` once. Scattered row writes happen once per query, so they are negligible.

This kernel needs a cooperative launch geometry, which can include several planes per cube to clear the 16-block/SM cap. Size `Shared` against `max_shared_memory_size`. It also needs a GPU test, because CPU tests run with `PLANE_DIM = 1` and never exercise the cooperative logic. No cross-cube communication is required. Diagnostics change meaning in this mode (tile visits rather than per-query visits) and must be documented as such. If F5 wins, the same structure applies to `kd_candidates_kernel` in Borůvka. That kernel is outside `core_distances`, and its component-bound atomics add constraints.

### F6. Rejected or low-value patterns for this stage

- **Matmul-style distances (`|q|^2 + |r|^2 - 2 q·r`, CubeK matmul).** Rejected for the exact path. The form changes rounding, causes cancellation near ties and breaks bitwise oracle/tie semantics. It is acceptable only as an approximate or pre-filter path with exact rescoring.
- **`plane_sum`/vector horizontal sums over features.** These reorder the accumulation (see invariants). They are also unnecessary, because vectorizing across candidates (F4) or across queries (F5) keeps the order.
- **Shared memory or barriers in the CPU kernel.** No on-chip benefit, and barriers create a risk of spin cliffs on the CPU runtime.
- **Atomics or single-pass inter-cube merging.** Each query's top-k belongs to one unit or one plane, so nothing needs publishing across cubes.
- **Core extraction or readback fusion.** Already ruled out above. The slice is asynchronous, and readback is under 1% of the stage.

### Suggested order and acceptance

F2 (root caching only) → F1 → F3 are local edits to the existing kernel and apply to both backends. Benchmark each one in interleaved batches. F4 on CPU and F5 on CUDA are new kernels behind the existing `exact_neighbors_kd` host signature, chosen once per client from device properties. For every variant, the acceptance checks are:

- bitwise-equal neighbor IDs and distances against the current kernel and the host oracle, including ties, duplicates, `d` in {1, 3, 4, 16, 30} and padded tails;
- tests at every vector width the host can select, not only the CPU's widest;
- a required-CUDA run for any plane path;
- registers and spills from Nsight Compute;
- core-stage and full-fit timing at 20k and 50k against an interleaved baseline.

## Verification, limitations and reproduction

Executed with required CUDA, no hardware skips:

```sh
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --features tracing \
  --test test_kdtree_search -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --release --features tracing \
  --example profile_hdbscan_stages -- --nocapture
```

**4/4 KD tests and 7/7 profiler tests passed.** KD tests cover exact host-oracle comparisons, ties, indirection/permuted trees, padding/small boundaries and overflow cases on CPU and CUDA. Profiler tests cover parser/sample boundaries and real CPU/CUDA span counts. Raw logs: `kdtree_tests.log`, `profile_tests.log`. Cargo warns that upstream `burn-cubecl` and `burn-cubecl-fusion` contain future-incompatible code; these warnings were not suppressed.

Baseline reproduction (use a fresh run directory):

```sh
cargo build -p fast-hdbscan --release --features tracing --example profile_hdbscan_stages
uv run --no-sync python analyze/benchmark_hdbscan_stages.py \
  --binary target/release/examples/profile_hdbscan_stages \
  --rows 20000 --run-dir artifacts/hdbscan_core_investigation/NEW_RUN --timeout 120
uv run --no-sync python analyze/profile_hdbscan_kernels.py \
  --binary target/release/examples/profile_hdbscan_stages \
  --rows 20000 --run-dir artifacts/hdbscan_core_investigation/NEW_RUN --timeout 120
uv run --no-sync python analyze/profile_hdbscan_counters.py \
  --binary target/release/examples/profile_hdbscan_stages --binary-label current \
  --rows 20000 --run-dir artifacts/hdbscan_core_investigation/NEW_RUN --timeout 120
```

The component scripts were used directly rather than the all-in-one shell driver to allow longer bounded process deadlines and independent targeted experiments. `run_experiments.py`, `run_sensitivity.py`, `experiments.json`, `sensitivity.json`, `alternating.json`, `alternating50.json`, and `dimension30_comparator.json` preserve configurations/raw samples; alternating records include the exact native commands. `dump_core_outputs.rs`, `probe_build.jsonl`, `probe_build_command.json`, `compare_permutation.py` and `permutation_semantics.json` preserve the public-output semantic check. Run harnesses only in a fresh evidence directory or after adapting their output paths; unlike the benchmark runner, the small exploratory helpers overwrite their own named outputs.

Limitations: one host/GPU; synthetic Gaussian workloads, no real datasets; no uniform/duplicate-heavy performance campaign; no prototype kernel speedups; no spill/coalescing counters; no full-workspace test claim. The 100k results validate current-baseline scaling and backend agreement, not a hypothetical optimized implementation. The downstream permutation-sensitive clustering mechanism remains unresolved and is not presented as a diagnosed defect.

**Recommended first implementation experiment:** an original-ID-preserving packed-candidate path on CPU, benchmarked independently from query reordering; on CUDA, a launch-resource sweep followed by cooperative leaf evaluation only if counters justify it. Validate exact neighbors/core outputs first, then full MST/clustering outputs and total fit cost.

## Phase 1 results

Register-cached KD heap root implemented against `2c3fc35fa2b2492f519c0adcb544779357f46dd8`, on Ryzen 9800X3D / RTX 3080, Rust/Burn/CubeCL identities in `artifacts/cubecl_kernels_perf/phase1/provenance.json`. Baseline and variant release tracing binaries are preserved with SHA-256. Phase 2 was later measured and rejected as documented below; no later optimization phase was accepted.

Each cell ran serial B,V,V,B,B,V processes, three measured fits after warmup per process. Values below are **core-stage seconds**, median [min–max] of three process medians per binary, not nine independent fits. `summary.json` retains all fit samples, process grouping, neighbor-operation-plus-sync and full-fit distributions. Every one of the 48 processes' final-fit label vectors matches its cell's baseline exactly.

| Rows / dimensions / backend | Baseline | Variant |
| --- | --- | --- |
| 20k / 16 / CPU | 0.274719 [0.273299–0.283516] | 0.276343 [0.274420–0.278630] |
| 20k / 16 / CUDA | 0.062015 [0.061393–0.063803] | 0.060967 [0.059383–0.062536] |
| 20k / 30 / CPU | 0.642068 [0.613602–0.669521] | 0.641285 [0.638788–0.674744] |
| 20k / 30 / CUDA | 0.086227 [0.085676–0.086515] | 0.082826 [0.082734–0.083026] |
| 50k / 16 / CPU | 1.302675 [1.270986–1.318149] | 1.336367 [1.325411–1.339078] |
| 50k / 16 / CUDA | 0.317111 [0.314820–0.317113] | 0.267245 [0.267160–0.270133] |
| 50k / 30 / CPU | 3.641410 [3.586260–3.656764] | 3.618289 [3.597900–3.664402] |
| 50k / 30 / CUDA | 0.803226 [0.784221–0.807848] | 0.592493 [0.590486–0.594431] |

**Decision:** retain the root cache. CUDA core medians improve at both required 16D scales, with the 50k variant maximum below the baseline median. All eight median regressions are bounded by their baseline process-median range. CPU is not a demonstrated speedup; its 50k×16 median increases about 2.6%, within the plan's noise bound. No generalization beyond these cells is warranted.

**Correctness:** the frozen test-only reference is verbatim HEAD kernel; production and reference were tested before the edit. Required-CUDA golden tests compare IDs, distance bits and per-row traversal diagnostics across the complete planned matrix, including ties and padding sentinels; diagnostics-disabled outputs also match. High-dimensional random host-oracle data exposed an existing one-bit difference on **both** CPU and CUDA before optimization (retained failing log). The strict host-oracle fixture therefore uses exactly representable bounded integer data at 16D/30D; no tolerance or backend exclusion was introduced. Random inputs remain covered by the frozen-kernel comparison.

**H3:** four fresh baseline/variant 20k/50k×16 captures retain raw CSV, matching input/run/route identities and selected binary hashes. All required evidence was collected with Nsight Compute 2026.2.1. Global-load request totals decline 866,955,776→819,526,249 at 20k and 4,982,334,074→4,712,480,345 at 50k. Sectors/request rise slightly, 12.117→12.223 and 10.160→10.242. Allocated registers/thread rise **71→75**, below the phase's approximate128 ceiling. Local-load sectors rise 7,796,729→7,799,494 and 71,764,852→71,864,591; local-store sectors remain 11,047,614 and 97,001,130. Existing stack/local traffic is not claimed eliminated.

Sampled long-scoreboard share (queried stalled-warp numerator divided by `smsp__pcsamp_sample_count`, with raw units/counts retained) falls **68.85→59.59%** at 20k and **71.85→39.69%** at 50k. Achieved occupancy is 18.53→17.83% and 27.88→30.39%; theoretical occupancy stays **33.33%**. These are first matching **warmup** launch captures, not whole-fit averages. Sampling/replay perturb execution; these shares are not the existing non-sampled stall percentage. Strict fixtures cover required missing metrics, malformed values/units, nonnegative integer counts including zero, positive registers, and zero-denominator unavailable ratios.

The first capability assessment was wrong: broad query output omitted specialized launch/occupancy/warp-sampling/source collections. Full raw queries and command/status/tool/device identity are now retained in `metric_queries/`; they confirm the exact `.sum` counters, allocated-register metric, occupancy and compatible sampled numerator/denominator. Initial average-only captures and the earlier parser failure remain archived for audit, but are superseded by the complete exact-metric captures above. No unavailable metric is silently replaced by a different ratio.

Evidence root: `artifacts/cubecl_kernels_perf/phase1/`; the initial fractional-parser failure and original random host-oracle failure are retained, not suppressed. Final verification after rollback is recorded in [`artifacts/cubecl_kernels_perf/phase2/logs/`](artifacts/cubecl_kernels_perf/phase2/logs/): 90 Python tests passed; the optimized campaign unittest passed; required-CUDA HDBSCAN suites passed 95 tests across suites, with 8 pre-existing manual/campaign ignores and no runtime CUDA skips; tracing example tests, `cargo fmt --all -- --check`, and clippy completed successfully (clippy emitted pre-existing warnings). The logs do not claim the ignored tests ran. Production and frozen-reference source prefix are byte-identical to HEAD. The final strict 100k pipeline passed at [`artifacts/hdbscan_stages/run-100000-y6BpkRky/`](artifacts/hdbscan_stages/run-100000-y6BpkRky), including exact CPU/CUDA label equality and retained Nsight Systems and Nsight Compute evidence (`raw/nsys_trace.json`, `cuda_kernels.json`, `cuda_counters.json`, and raw reports). README/AGENTS are unchanged because public behavior, API and layout are unchanged.

## Phase 2 results: register top-k experiment rejected

The Phase 2 exact-comptime-`k` register-list implementation was measured against the accepted Phase 1 root-cache heap on CPU and CUDA, Ryzen 9800X3D / RTX 3080. This is a bounded experiment, not evidence that register top-k is generally slower or faster. Provenance, baseline/variant binaries and hashes, inputs, cell samples, counter reports, cold-start records and the experimental kernel patch are retained under [`artifacts/cubecl_kernels_perf/phase2/`](artifacts/cubecl_kernels_perf/phase2/). The per-cell summary is [`summary.json`](artifacts/cubecl_kernels_perf/phase2/summary.json); cold first-fit data is [`cold_start.json`](artifacts/cubecl_kernels_perf/phase2/cold_start.json).

For each cell the runner alternated baseline and variant processes (B,V,V,B,B,V); each process supplied three fits, and the table reports the median and range of **the three process medians**, not nine independent fits. Core values are milliseconds; full-fit column gives baseline/variant process medians (ms). All 16 cells had exact final-label equality.

| Rows × dimensions | k | Backend | Core baseline median [min–max] | Core variant median [min–max] | Full-fit medians B/V |
| --- | ---: | --- | ---: | ---: | ---: |
| 20k × 16 | 5 | CPU | 270.705 [267.604–271.666] | 268.543 [268.180–274.717] | 325.331 / 324.036 |
| 20k × 16 | 5 | CUDA | 58.789 [58.515–59.043] | 58.650 [58.355–58.660] | 127.110 / 126.770 |
| 20k × 16 | 16 | CPU | 306.414 [296.754–311.963] | 299.044 [280.161–303.360] | 334.985 / 327.136 |
| 20k × 16 | 16 | CUDA | 64.646 [64.492–64.772] | 64.390 [63.753–68.834] | 112.466 / 111.557 |
| 20k × 16 | 17 | CPU | 311.776 [307.957–312.357] | 298.248 [297.433–334.982] | 339.594 / 325.678 |
| 20k × 16 | 17 | CUDA | 70.907 [70.840–70.974] | 71.144 [70.843–71.196] | 121.175 / 121.095 |
| 20k × 30 | 5 | CPU | 610.241 [602.808–626.535] | 625.997 [612.916–638.545] | 798.420 / 812.246 |
| 20k × 30 | 5 | CUDA | 82.653 [82.345–82.679] | 82.619 [82.046–83.069] | 230.645 / 231.120 |
| 50k × 16 | 5 | CPU | 1,346.111 [1,335.766–1,354.702] | 1,366.055 [1,317.155–1,366.723] | 1,546.423 / 1,550.468 |
| 50k × 16 | 5 | CUDA | 267.796 [265.624–269.913] | 266.673 [265.045–269.068] | 497.184 / 492.086 |
| 50k × 16 | 16 | CPU | 1,563.672 [1,534.052–1,588.237] | 1,629.554 [1,540.535–1,766.095] | 1,641.018 / 1,706.054 |
| 50k × 16 | 16 | CUDA | 332.303 [330.151–333.515] | 330.854 [328.737–333.103] | 464.235 / 459.508 |
| 50k × 16 | 17 | CPU | 1,560.856 [1,536.756–1,575.685] | 1,569.391 [1,555.950–1,600.088] | 1,635.497 / 1,641.426 |
| 50k × 16 | 17 | CUDA | 313.971 [312.612–314.763] | 312.544 [311.042–313.776] | 440.588 / 438.318 |
| 50k × 30 | 5 | CPU | 3,565.519 [3,505.034–3,712.549] | 3,621.818 [3,616.564–3,624.905] | 4,363.573 / 4,427.335 |
| 50k × 30 | 5 | CUDA | 565.500 [564.302–566.026] | 566.232 [563.122–567.094] | 1,157.294 / 1,172.186 |

**Decision: reject the register-top-k kernel; retain no threshold.** Rule 2's winning-backend test passed on CUDA for k5, but the mandatory gates still failed: rule 3 (k16 non-regression/local memory), rule 4 (k17 heap-path non-regression), and rule 6 (every-cell non-regression). In particular, k5 CPU 50k×16 increased **19.944 ms**, exceeding the baseline process-median spread of **18.936 ms**; k16 CPU 50k×16 increased **65.882 ms**, exceeding its **54.185 ms** spread; and unchanged-heap k17 CUDA 20k×16 increased **0.237 ms**, beyond its **0.134 ms** spread. The candidate also had slightly more local-load sectors in every captured CUDA counter pair, although local stores were unchanged. Thus the counter non-increase requirement failed; “no ptxas spills” does not waive it. CUDA k5 gains do not compensate for independent mandatory failures.

CUDA counter captures at 20k/50k, k5/k16, are first matching kernel launches from the excluded warmup fit—not measured-fit averages or all traversal/Borůvka rounds. The experiment used Nsight Compute defaults; replay mode is not specified in the evidence. Register counts were 72 (k5) and 94 (k16), versus 75 baseline. Local-load sectors rose from 7,799,102 to 7,801,167 (20k k5), 7,796,479 to 7,797,966 (20k k16), 71,864,604 to 71,887,586 (50k k5), and 79,875,335 to 79,888,523 (50k k16). Local-store sectors were unchanged in all four comparisons. Global-load request counts declined slightly, but this is not sufficient to accept the variant. Preserved PTX shows scalarized carry/select (`selp`) chains, not vector loads; the traversal stack remains 264 bytes and ptxas reports zero spill loads/stores. These observations describe these captures only.

Fresh-process first-fit evidence, in seconds, includes JIT and is reported as `warmup_seconds`; it is not `fit_seconds[0]` and is not isolated compilation time. Representative 20k×16 medians (baseline/variant) were k5 CPU 0.484638/0.476386, k5 CUDA 0.291954/0.310396, k16 CPU 0.481654/0.485021 and k16 CUDA 0.264032/0.276482. At 50k×16, k5 CPU was 1.687443/1.715660 and CUDA 0.658902/0.644374; k16 CPU 1.806149/1.776133 and CUDA 0.619718/0.618982. All observations and ranges, including k17 and 30D cells, remain in `cold_start.json`. CUDA first-fit variability is substantial in some variant cells; no broad cold-start or compilation-cost claim is made. Runtime cache defaults were unchanged.

**Final post-rollback verification:** [`review_fix_python.log`](artifacts/cubecl_kernels_perf/phase2/logs/review_fix_python.log) records 91 Python tests passed; [`final_runner.stderr`](artifacts/cubecl_kernels_perf/phase2/logs/final_runner.stderr) records the optimized campaign unittest. [`review_fix_hdbscan_full.log`](artifacts/cubecl_kernels_perf/phase2/logs/review_fix_hdbscan_full.log) records required-CUDA HDBSCAN tests, with 95 passes across suites, 8 pre-existing manual/campaign ignores, and no runtime CUDA skips. [`review_fix_profiler.log`](artifacts/cubecl_kernels_perf/phase2/logs/review_fix_profiler.log) records tracing example tests; `review_fix_clippy.stderr` and `review_fix_100000.stderr` preserve clippy and pipeline diagnostics. Formatting check and clippy succeeded; clippy emitted pre-existing warnings. The eight ignored tests were not run. The production and frozen-reference source prefix is byte-identical to HEAD. The final strict 100k pipeline passed at [`artifacts/hdbscan_stages/run-100000-LiDYNAjq/`](artifacts/hdbscan_stages/run-100000-LiDYNAjq), with exact CPU/CUDA label equality and retained Nsight Systems and Nsight Compute evidence (`raw/nsys_trace.json`, `cuda_kernels.json`, `cuda_counters.json`, and raw reports). The production kernel remains the accepted Phase 1 root-cache heap, and the extended equivalence/profiling harness is retained. No threshold revision was attempted, and Phase 3 was not implemented.

Boundary review also closed a configuration-identity gap: the fixed-k5 Nsight Systems consumer now rejects non-k5 provisioned manifests before profiler execution instead of recording a k5 capture under a k16 identity. Counter provisioning remains parameterized for k16. A regression test covers rejection before launch/report creation; inherited ordinary KD feature dimensions 1, 3, 4, 5 and 29 remain covered alongside the new exhaustive width sweep.


## Phase 3 results: padded vector data and register-resident query

**Decision: accepted**, with two recorded deviations from the literal Step 5 rule (below). Plan: [cubecl_kernels_perf_plan_phase3.md](cubecl_kernels_perf_plan_phase3.md). Evidence: [`artifacts/cubecl_kernels_perf/phase3/`](artifacts/cubecl_kernels_perf/phase3/) (`summary.json`, `provenance.json`, `alternation_records.json`, `cuda_counters.json`, `cold_start.json`, `logs/`, `generated/`). Scope: the measured cells and backends below on this host (RTX 3080, Ryzen 7 9800X3D), k = 5.

**Change.** `launch_kd_neighbors` packs the data and both bound arrays per call into 1D zero-padded f32 buffers (`pad_rows`) and passes them to `kd_neighbors_kernel` as `Vector<f32, Const<4>>` views. For `d ≤ 32` the query is loaded once per row into registers and every candidate/box evaluation issues `⌈d/4⌉` (resp. `2⌈d/4⌉`) 16-byte loads; the register path is specialized on the exact comptime feature count, so pad lanes are skipped. `d > 32` uses a runtime chunk loop over the same layout. Arithmetic per lane and its feature order are unchanged. `DeviceKdTree`, the public API and Borůvka are unchanged. Sizing adds `(n + 2·nodes)·4⌈d/4⌉·4` bytes to the KD core-search plan.

**Correctness.** With `FAST_UMAP_REQUIRE_CUDA=1`, every Step 3 command passed with no hardware skips. The extended harness (d ∈ 1–5, 8, 12, 16, 20, 24, 28–30, 32, 33, 40; k ∈ 1, 5, 16, 17 and more; duplicates, integer grids, `f32::MAX` overflow inputs, n = 257 and tiny n) is bit-identical to the frozen reference kernel in IDs, distance bits and per-row diagnostics on CPU and CUDA. The pack test checks `+0` pad bits exactly. Final labels were equal in every benchmark process. The copied runner is byte-identical to the Phase 1 runner.

**Timing** (process medians of 3 measured fits; order B,V,V,B,B,V; `alternation_records.json`):

| Cell | Backend | Core baseline, ms [min–max] | Core variant, ms [min–max] | Change | Full fit B → V, ms | Labels equal |
| --- | --- | --- | --- | --- | --- | --- |
| 20k×16 | CPU | 292.5 [290.7–295.2] | 206.7 [203.1–211.5] | -29.3% | 351.6 → 264.7 | yes |
| 20k×16 | CUDA | 66.2 [65.8–66.4] | 19.7 [19.5–20.0] | -70.3% | 142.3 → 95.9 | yes |
| 50k×16 | CPU | 1431.9 [1420.5–1443.7] | 1048.3 [1045.3–1083.6] | -26.8% | 1634.5 → 1260.1 | yes |
| 50k×16 | CUDA | 304.7 [300.6–306.7] | 99.5 [97.2–100.2] | -67.3% | 564.4 → 356.1 | yes |
| 50k×30 | CPU | 3964.0 [3911.3–4019.8] | 2999.7 [2990.5–3000.1] | -24.3% | 4889.2 → 3854.7 | yes |
| 50k×30 | CUDA | 639.9 [634.5–640.1] | 241.3 [240.3–242.1] | -62.3% | 1312.7 → 900.2 | yes |
| 20k×30 | CPU | 652.6 [646.3–681.4] | 489.8 [488.3–516.7] | -25.0% | 847.6 → 682.6 | yes |
| 20k×30 | CUDA | 93.2 [91.5–94.0] | 34.3 [34.3–35.6] | -63.2% | 259.2 → 199.2 | yes |
| 20k×8 | CPU | 38.7 [35.1–39.9] | 29.5 [27.2–35.1] | -23.8% | 64.7 → 55.0 | yes |
| 20k×8 | CUDA | 13.3 [11.7–14.6] | 10.3 [8.8–10.7] | -22.3% | 57.1 → 54.0 | yes |
| 20k×2 | CPU | 3.9 [3.2–4.0] | 3.4 [3.3–3.8] | -10.7% | 21.7 → 21.5 | yes |
| 20k×2 | CUDA | 1.6 [1.4–1.8] | 1.9 [1.7–1.9] | +17.0% | 17.1 → 16.8 | yes |

All four primary cells improve on both backends, and the variant maximum is below the baseline median in 4/4 cells on each backend. 50k×30 does not regress.

**Regression guards.** 20k×8 improves on both backends. At 20k×2 on CUDA the core stage is about 1.5–2.8 ms with bimodal per-fit samples. The first campaign (a `reg_chunks` specialization, kept in `attempt1_reg_chunks/`) and the main campaign (exact feature count) both put the variant median above the baseline maximum (by 0.58 ms and 0.06 ms; +37% and +17% versus the baseline medians). A replication of the guard cells with identical binaries (`artifacts/cubecl_kernels_perf/phase3_d2_replicate/`) showed no regression beyond spread: 1.7 ms variant vs 1.8 ms baseline core, 17.5 vs 18.1 ms full fit. Deviation 1: rule 3 was not met in two of three campaigns at 20k×2 CUDA. The effect is sub-millisecond and did not replicate.

**CUDA counters** (ncu, first `kd_neighbors_kernel` launch = warmup fit, 20k rows):

| d | Registers B → V | Global load requests B → V | Local load sectors B → V | Local store sectors B → V | Kernel time B → V |
| --- | --- | --- | --- | --- | --- |
| 16 | 75 → 106 | 8.20e8 → 1.28e8 | 7,798,943 → 7,785,545 | 11,047,614 → 11,045,114 | 82.0 → 24.0 ms |
| 30 | 75 → 128 | 1.48e9 → 2.24e8 | 6,307,809 → 6,302,296 | 9,255,624 → 9,253,124 | 113.8 → 36.1 ms |
| 2 | 75 → 34 | 3.49e6 → 2.04e6 | 136,300 → 136,520 | 332,412 → 329,912 | 0.455 → 0.400 ms |

Global load requests fall about 6.4× (16D) and 6.6× (30D), in line with the source-level count. Deviation 2: 2D local-load sectors rise by 220 (+0.16%), while local stores fall in every cell. The generated kernel (`generated/kd_neighbors_kernel_cuda_d2.ll`) has only the two 33-entry traversal-stack allocas. Query and candidate arrays are scalarized into `align 16` `<4 x float>` global loads, so the rise is stack traffic, not a scalarization failure. Long-scoreboard stall share rises (61 → 67% at 16D, 61 → 74% at 30D) because fewer, wider loads remain on the critical path. Achieved occupancy is unchanged (about 18–19%). The data `pad_rows_kernel` launch takes 9 µs (16D) and 16 µs (30D), under 0.1% of the CUDA core stage. On CPU, packing is inside the measured neighbor scope, so the gains above are net of it.

**Cold start** (`cold_start.json`, first fit including runtime JIT, median of 3 fresh processes): the variant is not slower in any of d ∈ 2, 16, 30 on CPU or CUDA (−0.001 to −0.075 s). The extra JIT for the pack kernel and the specialization is smaller than the faster first fit.

**Final pipeline validation.** `bash analyze_hdbscan_stages.sh --rows 20000` and `--rows 100000` both passed with the variant (logs in `artifacts/cubecl_kernels_perf/phase3/logs/analyze_*.log`). At 100k×16 the CUDA `core_neighbor_kernel` median is 381.7 ms (1.18 s in the baseline facts), and the CPU median is 3.34 s (4.66 s). The three `pad_rows_kernel` launches total 43.8 µs. CPU/CUDA labels agree exactly. These are single runs, not interleaved comparisons.

## Phase 4 results: compact node layout rejected

**Decision: reject; retain Phase 3 production.** Compact `u32` node records and interleaved min/max bounds were implemented and tested, then removed under the detailed plan's rejection rule. The leaf flag remained data (including `2`, `-1` and early leaves); `first_leaf` and batched sibling loads were not pursued. Arithmetic, the frozen reference and the existing equivalence harness were unchanged. The measured implementation remains in `artifacts/cubecl_kernels_perf/phase4/experiment.diff` and the preserved variant binary.

**Provenance/method.** Clean HEAD `f407b4b` differs from `3e885a7` only in planning documents. Fresh baseline SHA-256 `14ca624faa0fb72074d582c5cf16bdeeb2ea4ee61a9b8e51f1f2d47608906c62` matches the Phase 3 variant. Inputs retain Phase 3 hashes. Two predeclared serial matrices (A then B), each B,V,V,B,B,V per cell, give six process medians per arm; each process measures three fits. No further timing batches were run. All 144 process labels agree within their benchmark cells. Full per-arm ranges, both batches, six adjacent-pair ratios, core/neighbor/full-fit samples and medians are retained in `summary.json` and `batch_{A,B}/`. The following core-stage timings include packing, not isolated kernel speedups.

| Cell | Backend | Pooled core B → V ms | Pooled Δ | Six paired V/B range | Screen |
| --- | --- | --- | --- | --- | --- |
| 20k×16 | CPU | 212.374 → 219.583 | +3.40% | 1.0175–1.0403 | regression |
| 20k×16 | CUDA | 20.854 → 20.140 | −3.42% | 0.9567–1.1322 | no effect |
| 50k×16 | CPU | 1095.460 → 1126.597 | +2.84% | 0.9872–1.0497 | no effect |
| 50k×16 | CUDA | 101.519 → 99.902 | −1.59% | 0.9757–0.9991 | no effect |
| 20k×30 | CPU | 517.719 → 522.466 | +0.92% | 0.9953–1.0545 | no effect |
| 20k×30 | CUDA | 36.551 → 36.127 | −1.16% | 0.9434–1.0367 | no effect |
| 50k×30 | CPU | 3103.270 → 3095.563 | −0.25% | 0.9774–1.0233 | no effect |
| 50k×30 | CUDA | 247.964 → 246.917 | −0.42% | 0.9907–1.0541 | no effect |
| 20k×8 | CPU | 29.544 → 28.834 | −2.41% | 0.8247–1.2323 | no effect |
| 20k×8 | CUDA | 9.134 → 9.155 | +0.23% | 0.9665–1.0399 | no effect |
| 20k×2 | CPU | 3.261 → 3.402 | +4.31% | 0.8499–1.7660 | no effect |
| 20k×2 | CUDA | 2.064 → 2.036 | −1.36% | 0.6833–1.2596 | no effect |

**Acceptance criteria and judgement.** (1) Correctness passed on CPU and required CUDA, including exact-bit pack tests, early-leaf traversal and diagnostics; benchmark labels agree. (2) Failed: zero primary cells on either backend meet the −2% plus 5-of-6 gain screen. (3) Failed: 20k×16 CPU regresses +3.395% (+7.209 ms), with all six pairs above 1; neighbor scope also regresses +3.371%. (4) Global requests decrease in all profiled cells and registers remain within 128, but 16D local-load sectors increase by 279 (+0.00358%). No exception is taken.

CUDA point estimates favor the variant, but 20k×16 has only four favorable pairs and opposing batch effects. All 50k×16 CUDA pairs improve, but the pooled core effect is only −1.59%. CPU 16D costs are repeatable; CPU 30D effects are small/mixed. Guard cells, especially 2D, are noisy, not evidence of equivalence or reliable regression. These measurements do not justify extra record allocation/packing and eight extra registers at 16D/2D. Without a defensible net benefit, reject rather than stacking an unsupported optimization under Phase 5.

**Conditional CUDA counters.** NCU first matching warmup launches, not whole-fit averages; raw CSVs and all metrics, including sectors/request, long-scoreboard stall share and achieved/theoretical occupancy, are in `cuda_counters.json` and `counters/`.

| d | Registers B → V | Global requests B → V | Local load sectors B → V | Local store sectors B → V |
| --- | --- | --- | --- | --- |
| 16 | 106 → 114 | 128,376,127 → 127,771,285 | 7,785,712 → 7,785,991 | 11,045,114 → 11,045,114 |
| 30 | 128 → 128 | 223,896,822 → 223,305,948 | 6,302,618 → 6,302,366 | 9,253,124 → 9,253,124 |
| 2 | 34 → 42 | 2,042,570 → 2,031,136 | 136,560 → 136,558 | 329,912 → 329,912 |

The tiny 16D local-load increase, unchanged stores and registers below budget do not establish a new major spill regime; nevertheless the strict default fails. No demonstrated benefit warrants retaining or further tuning the resource cost.

**Aligned pack share.** One Nsight Systems timeline per binary/cell, restricted to its single measured fit, sums all pack launches and the KD neighbor launch. Pack/(pack+KD) is 0.0620% → 0.0665% at 16D, 0.0496% → 0.0523% at 30D, and 1.4688% → 1.6041% at 2D. `pack_share_raw.json` and `timelines/` preserve commands, fit windows, individual launches and SQLite/report files. These reporting-only shares never divide replayed NCU durations by unrelated medians.

**Cold first fits**, median of three fresh-process whole warmup fits, caches not cleared. These include runtime JIT and cannot isolate per-kernel compilation cost. First-minus-warm uses the same binary's pooled H2 full-fit medians, with sources in `summary.json`.

| d | Backend | First fit B → V ms | First-minus-warm B → V ms |
| --- | --- | --- | --- |
| 2 | CPU | 187.342 → 211.863 | 164.536 → 190.191 |
| 2 | CUDA | 294.668 → 295.050 | 276.903 → 277.305 |
| 16 | CPU | 500.900 → 484.319 | 226.034 → 206.237 |
| 16 | CUDA | 374.310 → 385.136 | 276.651 → 287.589 |
| 30 | CPU | 934.917 → 929.152 | 217.641 → 193.953 |
| 30 | CUDA | 495.150 → 511.054 | 290.279 → 306.280 |

**Verification/rollback.** Variant Step 3 logs are in `variant_logs/` and `variant_verification.json`: formatting, workspace clippy, core tests, required-CUDA KD/estimator/integration/full HDBSCAN suites, tracing example and 91 Python profiling tests passed. Existing manual ignored tests remain ignored; hardware skips are not counted as validation. An initial test-only nonexistent `DeviceKdTree::clone` compile error was corrected before the successful campaign. Only the three owned implementation/test source files were restored byte-for-byte to Phase 3, preserving unrelated work. Final restored-source gates passed again (`logs/`, `verification.json`), as did strict 20k and 100k analysis pipelines with CPU/CUDA label equality and real Nsight captures. Existing Burn future-incompatibility notices and clippy warnings were not suppressed. README/AGENTS are unchanged because public API/semantics are unchanged and neither enumerates these internal buffers.

Evidence root: `artifacts/cubecl_kernels_perf/phase4/`. `summary.json` records each unmet criterion, supporting/contrary evidence and the reject judgement. This is a completed negative experiment, not an accepted optimization.

## Phase 5 results: CUDA neighbor launch geometry 64 selected by override

**Current decision: select 64 threads by explicit user override (2026-10-26).** The default screen originally rejected both larger widths. The user prioritizes higher-row-count practical workloads and accepts the small absolute 20k cost in exchange for the measured 50k gains. This is an explicit performance-policy exception, not a passing default screen or relaxed correctness requirement. Only the CUDA neighbor preference changes to 64; the accepted Phase 3 algorithm, CPU geometry, unrelated KD launches and 32-thread frozen reference are unchanged. Width 128 remains rejected. No extra timing batch, dimension dispatch or follow-up width was used. Original measurements and `decision.json` remain historical evidence; `override_64.json` records the superseding decision and its validation.

**Method/provenance.** Fresh baseline from committed `0fd27dfdddfe96e4be1e9b00bcb2b31e5e21a85e` has SHA-256 `14ca624faa0fb72074d582c5cf16bdeeb2ea4ee61a9b8e51f1f2d47608906c62`, identical to the accepted Phase 3 binary. Fixed release/tracing 32/64/128 binaries, source snapshots/diffs, versions, device identity, commands and unchanged deterministic input hashes are preserved in `artifacts/cubecl_kernels_perf/phase5/`. CUDA requests legalize to actual widths 32/64/128 on the RTX 3080; all execute successfully. Timing diagnostics are disabled uniformly. Matrix A runs 64 then 128; matrix B reverses that order. Each cell uses B,V,V,B,B,V, three measured fits per process: six independent process medians per arm pooled across batches. All 192 timing processes complete with equal labels within each comparison cell. `batch_{A,B}_width_{64,128}/` retains raw samples, process medians, labels, traces and separate batch summaries; `pooled_summary.json` retains ranges, absolute milliseconds, neighbor-scope series and all six adjacent-pair ratios.

| Cell | Backend | Core B → 64 ms | Core Δ64 | Core B → 128 ms | Core Δ128 | Full-fit Δ64 / Δ128 |
| --- | --- | --- | --- | --- | --- | --- |
| 20k×16 | CPU | 220.794 → 217.966 | −1.28% | 218.264 → 218.019 | −0.11% | −0.36% / +0.21% |
| 20k×16 | CUDA | 22.419 → 21.339 | −4.82% | 23.591 → 24.220 | +2.67% | −0.71% / +2.35% |
| 50k×16 | CPU | 1102.689 → 1125.238 | +2.04% | 1115.099 → 1103.159 | −1.07% | +1.73% / −0.56% |
| 50k×16 | CUDA | 112.822 → 90.996 | −19.35% | 113.507 → 91.474 | −19.41% | −5.77% / −5.36% |
| 50k×30 | CPU | 3107.531 → 3139.487 | +1.03% | 3100.292 → 3125.252 | +0.81% | +0.23% / −0.83% |
| 50k×30 | CUDA | 271.985 → 244.910 | −9.95% | 271.411 → 246.698 | −9.10% | −2.72% / −1.88% |
| 20k×30 | CPU | 526.542 → 529.314 | +0.53% | 524.932 → 547.259 | +4.25% | −0.58% / +2.23% |
| 20k×30 | CUDA | 39.811 → 41.254 | +3.62% | 39.805 → 46.470 | +16.74% | +1.37% / +3.66% |

**Default-screen result and override rationale.** Width 64 meets both 50k CUDA gain screens (six favorable pairs each): core savings are 21.826 ms (16D, −19.35%) and 27.075 ms (30D, −9.95%), with full-fit deltas −5.77% and −2.72%. Its sole failed default acceptance criterion is the 20k×30 CUDA core regression: +1.443 ms / +3.62%, five unfavorable pairs; full fit increases +1.37% but does not meet the full-fit regression screen. Its 20k×16 CUDA core improves −4.82%, and no CPU control or full-fit cell meets the regression screen. The user explicitly accepts this trade-off because larger workloads dominate practical use and small workloads already finish quickly. This does not extrapolate measured gains beyond the matrix. Both batches, all paired ratios, ranges and absolute full-fit/neighbor timings remain available in the original pooled and per-batch summaries; no unfavorable cells are dropped.

For 64, pooled full-fit medians are 405.384 → 381.983 ms at 50k×16 (−23.400 ms), 1042.820 → 1014.499 ms at 50k×30 (−28.320 ms), and 225.720 → 228.811 ms at 20k×30 (+3.092 ms). The contrary 20k×30 cell is not uniform across batches: core/full-fit deltas are +1.92%/+0.12% in A and +4.21%/+2.79% in B. Both batches favor 64 at 50k: 16D core/full-fit −19.95%/−5.69% (A), −18.74%/−6.03% (B); 30D −10.50%/−2.90% (A), −9.41%/−2.53% (B). The policy accepts the pooled small-workload cost without claiming batch equivalence or adding favorable samples.

Width 128 also meets both 50k gains, but 20k×30 CUDA core regresses +6.665 ms / +16.74% (six unfavorable pairs), its full fit independently vetoes acceptance (+3.66%, five or more unfavorable pairs), and its 20k×30 CPU control regresses +22.327 ms / +4.25% (five unfavorable pairs). It remains rejected. CPU changes are controls, not optimizations or evidence of CUDA causality. Other intermediate outcomes do not prove equivalence. Neither candidate qualifies under the original default screen, so its automatic width tie-break is not invoked; 64 is the user's explicit policy choice.

**Resources.** `cuda_counters.json`, `occupancy.json` and raw `counters/` capture every 20k/50k × 16D/30D × baseline/64/128 cell. NCU selects the first matching production neighbor launch (warmup), not the reference or Boruvka; replay durations are not full-fit averages. Requested and observed block/grid sizes match ceil(rows/width). Registers change with actual launch geometry: 16D is 106 / 64 / 50 and 30D is 128 / 72 / 72 (32 / 64 / 128 threads). These observed compiler choices refute a universal 128-register-per-thread assumption. Larger cubes change resource allocation and residency rather than reducing traversal. Local stores remain unchanged; local loads change slightly (20k×16: 7,785,088 / 7,787,022 / 7,787,251 sectors), which alone does not establish spills. Exact diagnostics remain equal. The independent occupancy captures preserve register/block limits, theoretical/achieved active warps and occupancy; counter captures preserve long-scoreboard stalls, global requests/sectors and duration. No pack-share claim is made from unrelated timings. Scheduling/resource improvements at 50k do not erase measured 20k regressions.

**Fresh-process first fits.** `cold_start.json` and raw `cold_start/` retain three first fits per binary/backend/cell, including the fixed 32 hook binary. Compilation caches are not cleared; these are whole warmup fits including runtime JIT, not isolated compilation times, and are not selection samples.

**Original 32-thread cleaned-source validation.** Formatting, workspace/all-target clippy, core tests, required-CUDA KD library/reference tests, KD integration tests, the complete release HDBSCAN suite, the tracing example build and Python analysis tests pass (`verification.json`, `logs/`). All three legal CUDA widths pass exact IDs, distance bits and both diagnostics against the unchanged frozen reference, including diagnostics-disabled parity and forced two-cube multi-stride device launches. Synthetic legality tests cover axis/total clamps and duplicate legalized candidates; CPU geometry remains independent. Existing ignored manual tests are not claimed executed. Burn/CubeCL dependency future-incompatibility warnings remain visible; no warning is suppressed. These original receipts are not substitutes for validation of the subsequently selected 64-thread production source.

Both original 32-thread cleaned-source integration campaigns pass with real CPU/CUDA execution, exact label equality, route-aware Nsight timelines/counters and successful summaries: `artifacts/hdbscan_stages/run-20000-1iVawgUl/` and `artifacts/hdbscan_stages/run-100000-Q03VFEM9/`. The 100k process walls are 18.229 s CPU and 5.926 s CUDA under the harness's scale-specific limits. These are integration gates, not selection samples. Original source/binary/input hashes, commands and links are in `decision.json` and `integration.json`. The override changes only the CUDA neighbor preference to 64; no public setting, routing, allocation, feature ordering, query ordering, traversal or frozen oracle changes.

**Counter limitations.** Raw LaunchStats/Occupancy/SchedulerStats sections supplement the strict metric-parser captures. Register-allocation granularity and a separate compiler spill count are not exposed by these captures and remain unavailable, not zero. Stack size and local load/store sectors are observed; they do not by themselves separate traversal stack traffic from spills. No resource-infeasible candidate or unsupported-metric value is treated as a successful zero-duration measurement.

**64-thread override validation.** The final selected production source passes formatting, workspace/all-target clippy, required-CUDA KD library/reference and KD integration tests, and the full release HDBSCAN suite. Strict analysis at 20k and 100k also passes with real CPU/CUDA execution, exact CPU/CUDA label equality and complete route-aware kernel/counter summaries: `artifacts/hdbscan_stages/run-20000-6V62WfWu/` and `artifacts/hdbscan_stages/run-100000-glVnz4Ck/`. Commands, exit statuses and raw logs are in `artifacts/cubecl_kernels_perf/phase5/override_64/verification.json`; the superseding decision and final hashes are in `override_64.json`. These are validation runs, not extra timing-selection batches or proof of an interleaved 100k speedup. The default test launcher uses the same 64-thread constant and shared legalization/count helper as production; candidates, padding and frozen reference still use the separate baseline 32 preference. Existing dependency warnings remain visible.

## Phase 6 results: correctness rejection before acceptance timing

**Decision: reject at Step 3, not a performance rejection.** The position-tiled SoA CPU kernel, P/O/C configurations, block-cyclic query order, ownership pack, sizing/allocation integration and expanded harness were implemented experimentally. The new malformed-tree requirements expose a contradiction already present between the unchanged Phase 3 production kernel and the frozen scalar reference. Step 5 makes correctness mandatory and permits rejection/removal; no arm is eligible under the stated contract. Production kernel, estimator and export files were restored byte-for-byte to baseline HEAD `d030b7eb0a1f2a4edccc41fea01671ca9b16f092`, preserving the Phase 5 CUDA neighbor preference of 64. `experiment.diff` retains the entire attempt and failing fixtures. This is not evidence that packed candidates or leaf ordering are slow or incorrect on valid builder trees.

**Provenance and prerequisite results.** `artifacts/cubecl_kernels_perf/phase6/provenance.json` records the initially clean diff hash, compiler/CPU, input hashes and preserved tracing baseline binary SHA-256 `65de332bfd60b95d6ba0553f54cf9b30be81427795ed031ad544f082039ff6ea`. Its difference from the Phase 4 binary is expected after Phase 5. This host reports 15 CubeCL CPU cores, load width 512 and `io_optimized_vector_sizes(4) = [16, 8, 4, 2, 1]`. The generic vector tensor/broadcast/extract/subtract/multiply/accumulate probe passes every width. CPU runtime source creates and schedules one task per unit, so block-cyclic work is distributed across concurrent units. The initial cap was 16; width selection was not run and no cap was selected.

**The blocking observations, not relaxed tests:**

- In `logs/baseline_malformed_cpu.log`, the additional first CPU test plan is the unchanged `PaddedRows × Cyclic` route. At `n=4, d=3, k=5`, a root leaf containing candidate ID `n+3 = 7` yields row-0 IDs `[2, 3, 7, 1, 4]`, versus frozen-reference `[2, 3, 1, 7, 4]`. Thus even the baseline fails the new reference-equality requirement. The archived source identifies the baseline-first ordering and dispatch; it is not a result attributed solely to C.
- With a reachable leaf ending at `n+5` (`n=300, d=3`), the frozen reference faults before variant comparison: CPU SIGSEGV (`logs/triage_cpu.log`) and CUDA event status 700 (`logs/triage_cuda.log`). The isolated required-CUDA reproduction fails identically. The fault is not a hardware skip.
- All earlier CPU cases in those runs complete their all-arm/all-width ID, distance-bit, per-row diagnostic and diagnostics-disabled comparisons, including NaNs, non-permutation ownership, early and empty leaves. Pack, ownership and checked-sizing tests also pass in the experimental log. These partial successes do **not** make the complete experimental gate pass.

The plan assumes out-of-range tensor accesses deterministically clamp to the last *logical* element. Dependency source instead derives checked buffer lengths from physical handles, which may contain upload padding. Furthermore, scalar `data[candidate*d+f]` and four-lane padded-vector gathers have different out-of-range address domains even with exact-sized buffers. Diagnostic `cubecl/pliron-dump` builds preserve generated IR under `generated/triage/`; the inspected frozen-reference NVPTX IR has direct indices/candidate GEP/load sequences without an intervening bounds clamp. These facts invalidate the assumed safe oracle; the precise compiler/runtime fault mechanism is not established here. Neither the reference, fixtures nor dependency implementation was modified to manufacture a passing result. Resuming optimization requires an explicit correction of malformed-input semantics/oracle scope, and potentially a separate bounds-safety change.

**Unperformed acceptance work.** Width batch S, isolation batches A/B, their paired/pooled statistics, per-arm accepted binaries, selected-arm codegen/cold-start comparison and selection/fold protocols were not run because Step 3 failed. There are no measured P/O/C gains, guard regressions, accepted width, or implied zero costs. Prepared runners and `summary.json` mark these as not run, rather than fabricating acceptance data.

**Retained host measurement tool.** `kd_host_baseline` implements scalar ordered-f32 traversal with Rayon, validated input/CLI, JSON run medians and `--verify` against the public CPU operation. Three independent processes per primary cell, five timed traversals each, 15 threads, were run after rejection as reporting-only measurements (not acceptance samples). Tree construction, allocation, pool creation, one warmup and verification are excluded; output initialization and scheduling are included. `rayon_baseline.json` retains every result:

| Cell | Median of process medians, ms | ID mismatches | Distance-bit mismatches / outputs |
|---|---:|---:|---:|
| 20k×16 | 279.967 | 0 | 31,111 / 100,000 |
| 50k×16 | 1478.868 | 0 | 79,019 / 250,000 |
| 20k×30 | 662.593 | 0 | 33,674 / 100,000 |
| 50k×30 | 4312.856 | 0 | 81,495 / 250,000 |

The host verification honestly reports non-equal distance bits; this is **not** an exact replacement oracle. Rust uses separate multiplication/addition, while the unchanged CPU kernel's optimized diagnostic LLVM IR contains `llvm.fmuladd` in the ordered sums (`generated/baseline_cpu16/`). That supports contraction as an explanation, not a proof of every mismatch. No FMA change was made to hide the comparison. Fifteen additional black-box boundary/CLI cases pass, including ties, overflow distances, `k > n`, dimensions above 32, invalid options, nonfinite and truncated input (`host_baseline_boundary_report.json`). Timings conflate runtime, scheduling, codegen and buffer handling; they do not isolate CubeCL overhead or establish a Phase 6 speedup.

**Final retained-source verification.** `final_gates.json` and full `logs/final_*.log` record passing formatting, workspace/all-target clippy, core tests, CPU and required-CUDA KD library tests, estimator and KD integration tests, the full release HDBSCAN suite, tracing-example tests, host-example build and 91 analysis pytest cases. Existing manual/observational ignored tests and dependency/clippy warnings remain visible and are not claimed executed or fixed. Final strict pipeline runs at 20k and 100k pass with actual CPU/CUDA execution, exact label equality and kernel/counter summaries: `artifacts/hdbscan_stages/run-20000-SgDOYUAT/` and `artifacts/hdbscan_stages/run-100000-8cBgENde/` (`integration.json`). These prove the retained baseline/tool integration, not correctness of the rejected malformed cases or performance of a selected CPU arm. `AGENTS.md` and `README.md` were rechecked; neither enumerates this CPU schedule, its removed buffers, or a list requiring the new measurement example, so both remain unchanged.

### Phase 6 approved forward contract: valid builder trees only

The preceding rejection and measurements remain historical facts. The user subsequently approved narrowing the retry to finite data and valid builder-produced trees, with equivalent valid-tree preconditions documented for low-level callers. The builder rejects nonfinite input, initializes indices as `0..n` and partitions ranges; the estimator uploads that tree unchanged. No runtime tree-content scan, readback or validation is to be added. Malformed/NaN/non-permutation fixtures and physical-buffer clamp assumptions are not revised acceptance requirements. Ownership maps, owner launches, absent-row passes and owner memory terms are removed from the prospective design, sizing, timing, cold-start and cleanup protocols; a permutation supplies exactly one query writer per original row.

The revised [detailed plan](cubecl_kernels_perf_plan_phase6.md) retains position-tiled SoA, fixed block64 query order, P/O/C isolation, ordered arithmetic, checked sizing, balanced width/performance protocols and rejection options. It requires builder tests for permutation, range coverage/partition, reachable topology, enclosing bounds and level limits, meaningful builder input failures, and full valid-tree bitwise/all-width/all-arm/diagnostics coverage. Preserve Phase 5 CUDA neighbor64, unrelated32 and frozen kernels. Prior partial normal-case passes are not this revised validation proof; rerun all required original gates, including actual required CUDA and final 20k/100k pipelines, against any new implementation.

This amendment authorizes documents only, not execution. Original G001 is complete as an authorized rejection, not an accepted optimization; a new implementation needs a separate user execution request/new goal. Known HEAD at amendment is `d8faeab5d0e825e4d93c6c1478e35b4d1f15721a` (`phase 6 failure`), with the retained host example/Rayon dependency tracked. Inspect actual future HEAD/status and preserve user work, rather than assuming uncommitted tool changes or requiring HEAD equality. Original production baseline is `d030b7eb0a1f2a4edccc41fea01671ca9b16f092`; production `kdtree.rs`, `kernels/mod.rs` and `estimator.rs` were restored to it. Verify actual source anew. Historical `phase6/` retains `experiment.diff`, provenance, summaries, gates, integration, Rayon and QA boundary reports, and failure IR/logs. Do not blindly reapply that obsolete diff or overwrite evidence. If ignored artifacts are missing, do not assume tests/timing evidence or reconstruct unsupported claims: establish fresh evidence from tracked source. Use a fresh directory such as `phase6_valid_trees/`, establish provenance and preserve the accepted CUDA64 baseline binary before rebuilding. The detailed plan lists exact six input paths/hashes, affected code/test symbols and isolated diagnostic-build instructions. Width probe `[16,8,4,2,1]` and 15 units are prior findings, not a selected cap; initial cap16 was never selected, and no P/O/C timing exists. The host tool remains reporting-only, not a bitwise oracle. No retry gains, implementation or new gate results are claimed here.

## Phase 6 valid-tree retry results: C16 accepted

**Status:** execution authorized, implementation completed, acceptance code **C16** (SoA candidates plus leaf-block queries, cap16). Independent review receipts are maintained separately in the ultragoal ledger. The historical malformed-tree rejection above remains unchanged and is not new acceptance evidence.

**Scope and evidence:** all retry evidence is in [`artifacts/cubecl_kernels_perf/phase6_valid_trees/`](artifacts/cubecl_kernels_perf/phase6_valid_trees/). Finite data and unchanged builder-produced valid trees (or equivalent low-level permutation, leaf-partition, topology/levels and enclosing-bounds preconditions) are required. Public finite-row compaction/remapping is unchanged; no runtime content scan, ownership map or absent-row pass is added. These fixed k5 workloads on this Ryzen 9800X3D/CubeCL LLVM host do not establish general workload/platform speedups. CUDA neighbor64 and unrelated/frozen32 are preserved.

### Retry width and selection

`provenance.json` directly queried **15 CPU cores/units**, `hardware.load_width=512`, widths **[16,8,4,2,1]**; all widths passed the typed vector probe. Source concurrency evidence is not measured simultaneous occupancy. Batch **S**, reported separately and never pooled into acceptance, chose cap16 (`width_selection.json`): core medians at 20k×16 / 20k×30 were X16 **0.038808 / 0.093683 s**, Y8 **0.041840 / 0.099793 s**, Z4 **0.055731 / 0.138602 s**. Y/X ratios 1.07815 / 1.06522 and Z/X 1.43609 / 1.47948 fail the narrower-width ≥2% improvement in both cells rule.

`acceptance_summary.json` pools the two predeclared direction-balanced A/B batches (eight process medians per arm/cell). P (SoA only) and C each have **four primary core gains and no regression cells**; O (ordering only) has zero primary gains and is ineligible. **G=C**, primary geometric mean 0.152794381 s versus P 0.153396097 s. P fails the +2% per-cell near-best rule at 20k×16: 0.039347995/0.038212973 = **1.02970**. The exact default rule therefore selects C16 without override. The summary's pending-external-gates field is a timing-stage record, not a correctness waiver.

### Retry warm measurements

[`summary.json`](artifacts/cubecl_kernels_perf/phase6_valid_trees/summary.json) derives the acceptance decision, first-minus-warm differences, Rayon comparisons and pipeline label proof from the underlying receipts. Its decision is `accept`, with no unmet acceptance criteria. Independent review receipts are maintained separately in the ultragoal ledger. Acceptance samples use archived measurement binaries. Finalization removes unselected branches without changing selected C's algorithm; final-source verification is separate and is not pooled into acceptance timing.

Seconds; B baseline, P packed/cyclic, O padded/leaf-block, C packed/leaf-block. Δ compares pooled medians to B; paired ratios are median [min,max]. All-arm ranges and per-batch effects for every metric remain in `acceptance_summary.json`.

| Cell | Core B / P / O / C (s) | C pooled Δ; batch A / B Δ | C paired core ratio [range] |
| --- | --- | --- | --- |
| 20k×16 | .203527 / .039348 / .201191 / .038213 | −81.22%; −81.35% / −80.48% | .1881 [.1799,.2132] |
| 50k×16 | 1.044663 / .245888 / 1.022576 / .246394 | −76.41%; −76.41% / −76.72% | .2345 [.2192,.2554] |
| 20k×30 | .488497 / .092948 / .488041 / .093392 | −80.88%; −80.77% / −80.94% | .1916 [.1765,.2314] |
| 50k×30 | 2.879559 / .615681 / 2.855542 / .619836 | −78.47%; −78.74% / −78.04% | .2135 [.2059,.2270] |
| 20k×8 | .028158 / .011078 / .026245 / .010659 | −62.14%; −63.76% / −60.35% | .3822 [.2850,.4317] |
| 20k×2 | .003473 / .003375 / .003228 / .003459 | −.41%; −.58% / −6.56% | .9557 [.6136,1.2507] |

| Cell | Synchronized neighbor B → C | Full fit B → C; pooled Δ | Full-fit batch A / B Δ; paired median [range] |
| --- | --- | --- | --- |
| 20k×16 | .202454 → .037243 | .260531 → .097523; −62.57% | −63.01% / −62.15%; .3744 [.3574,.3995] |
| 50k×16 | 1.043565 → .245341 | 1.241712 → .448006; −63.92% | −63.81% / −64.44%; .3588 [.3245,.3854] |
| 20k×30 | .487422 → .092482 | .677496 → .282036; −58.37% | −58.93% / −57.36%; .4140 [.4003,.4787] |
| 50k×30 | 2.878350 → .616742 | 3.711688 → 1.420763; −61.72% | −61.89% / −61.68%; .3800 [.3722,.4137] |
| 20k×8 | .027032 → .009656 | .053880 → .034656; −35.68% | −33.89% / −37.12%; .6423 [.5672,.7384] |
| 20k×2 | .002605 → .001986 | .022062 → .021778; −1.29% | +1.62% / −4.00%; .9716 [.8642,1.0885] |

Packing is included in core timing. Primary C core gains are −76.41% to −81.22%; 20k×8 is also a gain. Noisy 20k×2 core/full-fit results are **no effect**, not equivalence or demonstrated speedup.

### Retry memory, host comparator and first fit

Additional C16 device memory beyond existing padded data/bounds is `ceil(n/16)·16·(d+1)·4` bytes (SoA plus slot IDs), with checked sizing/allocation checks: 20k/50k×16 **1,360,000 / 3,400,000 bytes**; 20k/50k×30 **2,480,000 / 6,200,000 bytes**; 20k×8/2 **720,000 / 240,000 bytes**. No ownership allocation exists.

The existing `kd_host_baseline` Rust/Rayon tool and dev-dependency are retained. Reporting-only `rayon_baseline.json`: 15 threads, one warmup, five runs/process, three processes. Median-of-process medians are **.286200 / 1.459307 / .633810 / 4.277110 s** for 20k×16, 50k×16, 20k×30, 50k×30. Verification has zero ID mismatches but **31,111 / 79,019 / 33,674 / 81,495 distance-bit mismatches**, respectively. It is neither a bitwise oracle nor an isolated runtime-overhead measurement. Ordered unfused Rust versus observed LLVM FMA is a supported explanation, not proof of equality.

`cold_start.json`: first-fit medians from three fresh processes, including runtime JIT, B→C **.189786→.252638 s** at 20k×2 (+.062852 s), **.423338→.345365 s** at 20k×16 (−.077973 s), **.887035→.560610 s** at 20k×30 (−.326424 s). Warm C full-fit medians above are .021778 / .097523 / .282036 s. **On-disk compilation cache was not cleared**: fresh-process first fits are not guaranteed cache-cold compilations.

### Retry gates and observed code generation

`summary.json` derives B/C first-minus-warm full-fit differences of **.167724/.230859 s** at 20k×2, **.162807/.247842 s** at 20k×16 and **.209539/.278575 s** at 20k×30. These are differences across measurement groups, not isolated compilation costs.

All **11 experimental and 11 final-clean Step 3 gates passed**, with required CUDA on designated CUDA gates (`experiment_gates.json`, `final_clean_gates.json`). Experimental coverage includes all arms/host widths; final-clean coverage includes selected production at every host width and retained CUDA coverage. Bitwise IDs/distances and diagnostics are checked under the valid-tree contract. `integration_clean.json` records final `analyze_hdbscan_stages.sh` exits **0** at **20k and 100k**; both `pipeline_clean_20000/benchmark.json` and `pipeline_clean_100000/benchmark.json` have `correctness.labels_equal=true` (checked directly). Earlier `final_gates.json`, `integration.json`, pipeline outputs and logs are retained as earlier receipts, not substituted for this clean-boundary verification.

The full `all_cuda` campaign retains pre-existing unrelated ignored hardware-campaign tests; these are not required KD CUDA runtime skips. The final KD suite reports **16 passed, 0 ignored**. Initial final clippy introduced warnings despite exit0; these were corrected before the clean rerun. `logs/final_clean_clippy.log` retains only pre-existing seed-literal grouping, integration-oracle range-loop and dependency future-incompatibility warnings; no introduced warning remains and no warning suppression is claimed. `summary.json` records that README/AGENTS were searched: neither enumerates the changed CPU query schedule or resource buffers, so no guidance update was needed; mandatory 20k/100k guidance remains applicable.

`codegen_evidence.json` links hashed optimized LLVM IR from reporting-only 257-row d16 diagnostic probes, not timing binaries. Selected IR has **16 `load <16 x float>`**, **16 `fsub <16 x float>`**, **17 `llvm.fmuladd.v16f32`** matches; baseline has zero of these and **33 scalar `llvm.fmuladd.f32`** matches versus selected **17**. These textual counts include declarations, not dynamic instruction counts or assembly/ISA proof. Observed vector FMA supports candidate-axis vectorization but independently proves neither correctness nor the entire speedup. Independent review receipts are maintained separately in the ultragoal ledger.
## Phase 7 results: CUDA flat plane kernel F2 accepted

**Decision:** accept **F2**, fixed **two planes per cube**, dimension gate **16**, **no row-count gate**. CPU C16 is unchanged. CUDA retains the per-query kernel below 16 features, above 32, on non-plane devices and with insufficient launch/shared-memory resources. This is pre-launch routing, not execution-error recovery. F scans all candidates, staging leaf-ordered padded rows for plane broadcasts, retaining original IDs/output rows, the output-backed heap and ordered scalar feature arithmetic. Experimental arms/measurement overrides are removed in finalization; below-gate forcing is test-only.

Evidence: `artifacts/cubecl_kernels_perf/phase7/summary.json`, `experiments/{selection-planes,selection-gate,acceptance,cold,crossover}/observations.json`, and `arm_counters/summary.json`. Experimental H1 and existing release/debug gates passed. **Final retained-source verification and strict real-CPU/CUDA 20k/100k integration passed** (receipts below). Independent review status/receipts are tracked separately in the Ultragoal ledger. Experimental gates are not substituted for final-source proof.

### Acceptance A/B

Four interleaved blocks in A and four reversed blocks in B yield **eight process medians per arm/cell**, each reducing three measured fits. Pairing is within block; fits within one process are not independent observations. Core includes packing. No guard cells were run. Values below are seconds: median [minimum, maximum] across process medians, not confidence intervals. Exact labels agree in each comparison cell.

| Cell | Arm | Processes | Core seconds | Full-fit seconds |
| --- | --- | ---: | --- | --- |
| 50k×16 | B | 8 | .081274975 [.079497385,.081589532] | .344502836 [.338933851,.350434888] |
| 50k×16 | T2 | 8 | .020770730 [.020288903,.021904879] | .282018535 [.277249858,.301995144] |
| 50k×16 | F2 | 8 | .019142247 [.018906253,.020449292] | .282762217 [.279065369,.290641309] |
| 300k×16 | B | 8 | 5.458147029 [5.406012669,5.505541137] | 8.148663046 [8.083295341,8.214601698] |
| 300k×16 | T2 | 8 | .488467727 [.480982227,.493652909] | 3.182441613 [3.128015232,3.209343435] |
| 300k×16 | F2 | 8 | .476285891 [.468212442,.480946523] | 3.171553079 [3.155536712,3.218120805] |

F2 core Δ: **−76.44755% / −91.27385%**, with **8/8 paired ratios below 1** at 50k/300k. T2 also gains: **−74.44388% / −91.05067%**, 8/8 each. F2 is accepted by the **large-size override** and also satisfies **standard eligibility**. No 50k regression is ignored and there is **no judgment departure**. Full-fit F2 Δ: **−17.92166% / −61.07885%**; T2: **−18.13753% / −60.94523%**. These are scoped full-fit results, not isolated neighbor-kernel speedups.

### Selection S, separate from acceptance

S uses six process medians per arm/cell, each from three measured fits, **never pooled with A/B**. At 300k×16, the smallest plane count within 2% of its family's best selects **T2 and F2**. Core medians at 50k / 300k: B .080915468 / 5.473774395; T1 .027557338 / .621755854; T2 .020418030 / .485891982; T4 .020969542 / .490142895; F1 .023531956 / .568501482; F2 .018763580 / .476576222; F4 .019995508 / .500840791 seconds. The finalists are within 2% at 300k; F2 wins the 50k comparison by more than 2%.

T gate selection uses separate six-process groups at 300k×2/8/16: B core medians .011764237/.311471664/5.494453881 s; T2 .019636370/.071913957/.489031505 s. Thus T's selected gate is **8**, not the predicted 16; 2D fails. **T2 is not retained. F's fixed gate is 16**, not inherited from T. Raw S core/full-fit medians and ranges are tabulated below.

| S planes cell | Arm | Processes | Core seconds | Full-fit seconds |
| --- | --- | ---: | --- | --- |
| 50000×16 | B | 6 | .080915468 [.079364360,.082346767] | .344954763 [.334696385,.353197455] |
| 50000×16 | F1 | 6 | .023531956 [.022177004,.024165243] | .282584847 [.278557993,.286498762] |
| 50000×16 | F2 | 6 | .018763580 [.018331581,.019365160] | .277636557 [.271683756,.280519361] |
| 50000×16 | F4 | 6 | .019995508 [.019621459,.021091858] | .280526630 [.278251591,.286034875] |
| 50000×16 | T1 | 6 | .027557338 [.026900096,.028034268] | .288563656 [.281932998,.302113999] |
| 50000×16 | T2 | 6 | .020418030 [.020027070,.021780674] | .281473519 [.276166356,.284225358] |
| 50000×16 | T4 | 6 | .020969542 [.020357486,.021761711] | .282009726 [.275594258,.289706449] |
| 300000×16 | B | 6 | 5.473774395 [5.445902474,5.481680337] | 8.147657364 [8.125146821,8.189191907] |
| 300000×16 | F1 | 6 | .568501482 [.565934748,.569244336] | 3.245857948 [3.237674011,3.259044396] |
| 300000×16 | F2 | 6 | .476576222 [.475044167,.480683666] | 3.151467626 [3.149414248,3.190279835] |
| 300000×16 | F4 | 6 | .500840791 [.495101261,.504383897] | 3.181553553 [3.160859505,3.196365693] |
| 300000×16 | T1 | 6 | .621755854 [.618925151,.624836905] | 3.299158210 [3.273621430,3.308667082] |
| 300000×16 | T2 | 6 | .485891982 [.483647490,.491700598] | 3.152823881 [3.142802097,3.168609457] |
| 300000×16 | T4 | 6 | .490142895 [.485945593,.490798814] | 3.148546950 [3.139007290,3.164783162] |

| S gate cell | Arm | Processes | Core seconds | Full-fit seconds |
| --- | --- | ---: | --- | --- |
| 300000×2 | B | 6 | .011764237 [.011310006,.012632526] | .200977907 [.196996622,.204125473] |
| 300000×2 | T2 | 6 | .019636370 [.018960290,.020064999] | .210581808 [.206515663,.216234925] |
| 300000×8 | B | 6 | .311471664 [.309335683,.313060663] | .777289958 [.769647988,.782020752] |
| 300000×8 | T2 | 6 | .071913957 [.070629623,.073216825] | .539097727 [.538067510,.544797206] |
| 300000×16 | B | 6 | 5.494453881 [5.475897269,5.530640062] | 8.182252261 [8.157842485,8.236976192] |
| 300000×16 | T2 | 6 | .489031505 [.486256728,.490875794] | 3.176813347 [3.173110644,3.183297610] |

### H3: all seven arms × two cells

Counters sample the **first warmup neighbor launch**, not full-fit timing. Separate matching-input/arm/geometry diagnostics supply emitted counts, not simulation or acceptance instrumentation. Nsight replay defaults flush caches and use base clocks. B's denominator is **leaf positions scanned**; tile denominators are **summed staged positions ×32 exactly once**, representing **nominal broadcast opportunities**, not active-lane evaluated pairs. F reports zero node pops and n scanned positions per query including self; staging is n·ceil(n/32).

In the table: DRAM is raw read bytes, B/op and L2 sectors/op use those denominators, L1 s/r is global-load sectors/request, occupancy is achieved active-warps %, LS is long-scoreboard stall %, local is load/store sectors, shared is dynamic bytes/block (static zero throughout). Raw names, units and provenance remain in the counter summary.

| Arm | Rows | Denominator | DRAM bytes | B/op | L2 sectors/op | L1 s/r | L2 hit % | L1 throughput % | Registers | Occupancy % | LS % | Local ld/st sectors | Shared bytes | Branch uniform % | Threads/instruction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| B | 50000 | 1364757656 | 2406746240 | 1.7635 | 2.6703 | 10.1028 | 98.11 | 81.07 | 64 | 46.05 | 77.27 | 71821777/96994880 | 0 | 98.01 | 10.32 |
| B | 300000 | 29377727608 | 1933837973504 | 65.8267 | 2.8270 | 8.4880 | 29.64 | 28.72 | 64 | 63.41 | 94.68 | 1195111370/1554076006 | 0 | 98.77 | 8.57 |
| T1 | 50000 | 2456784320 | 257896192 | .1050 | .0781 | 3.9816 | 96.10 | 35.94 | 114 | 24.86 | 14.12 | 25312674/25716696 | 2176 | 97.76 | 19.76 |
| T1 | 300000 | 79270631488 | 119892487680 | 1.5124 | .0754 | 4.8576 | 44.41 | 43.32 | 114 | 31.21 | 17.02 | 560563512/562963512 | 2176 | 97.79 | 14.37 |
| T2 | 50000 | 2456784320 | 418921856 | .1705 | .0846 | 3.9945 | 94.77 | 50.55 | 63 | 42.07 | 15.51 | 25312674/25716696 | 4352 | 97.76 | 19.78 |
| T2 | 300000 | 79270631488 | 128738542720 | 1.6240 | .0782 | 4.8820 | 42.19 | 57.74 | 63 | 60.14 | 15.71 | 560563512/562963512 | 4352 | 97.79 | 14.44 |
| T4 | 50000 | 2456784320 | 544766208 | .2217 | .0859 | 3.9945 | 93.98 | 49.94 | 56 | 43.06 | 16.10 | 31700358/25722948 | 8704 | 97.76 | 19.79 |
| T4 | 300000 | 79270631488 | 132947671424 | 1.6771 | .0796 | 4.8821 | 41.18 | 58.46 | 56 | 66.83 | 15.25 | 705670544/563001012 | 8704 | 97.79 | 14.44 |
| F1 | 50000 | 2500800000 | 97200384 | .0389 | .0441 | 6.8189 | 101.69 | 39.12 | 63 | 24.75 | 9.03 | 0/0 | 2176 | 99.22 | 31.11 |
| F1 | 300000 | 90000000000 | 3905547392 | .0434 | .0377 | 10.5505 | 98.75 | 49.06 | 63 | 32.26 | 6.77 | 0/0 | 2176 | 99.82 | 31.78 |
| F2 | 50000 | 2500800000 | 91820032 | .0367 | .0678 | 6.8161 | 98.45 | 50.15 | 62 | 39.65 | 9.17 | 0/0 | 4352 | 99.22 | 31.12 |
| F2 | 300000 | 90000000000 | 8390150144 | .0932 | .0558 | 10.5490 | 94.27 | 60.16 | 62 | 63.16 | 4.76 | 0/0 | 4352 | 99.82 | 31.79 |
| F4 | 50000 | 2500800000 | 77178496 | .0309 | .0641 | 6.8160 | 99.17 | 47.82 | 48 | 41.54 | 9.53 | 0/0 | 8704 | 99.22 | 31.18 |
| F4 | 300000 | 90000000000 | 5368952704 | .0597 | .0457 | 10.5489 | 96.58 | 59.08 | 48 | 77.49 | 5.12 | 0/0 | 8704 | 99.82 | 31.80 |

F1's reported 50k L2 hit rate **101.69%** is retained as a profiler/replay anomaly, not a physical hit probability. At **50k**, B already has **98.11% L2 hits**: gains are consistent with reduced **L1/LSU traffic**, not curing L2 misses. F2 reduces raw DRAM reads 2,406,746,240→91,820,032 bytes, L1 throughput 81.07→50.15% and long-scoreboard stalls 77.27→9.17%. At **300k**, B/F2 L2 hits are 29.64/94.27%, DRAM reads 1,933,837,973,504/8,390,150,144 bytes and stalls 94.68/4.76%. This supports improved access/reuse on this workload, not universal causation. F2 occupancy is 63.16% versus B 63.41% at 300k, so increased achieved occupancy is not the explanation. F has zero measured local sectors; T retains stack traffic.

Additional H3 metrics: sampled-stall share is long-scoreboard samples / all PC samples, distinct from the active-warp stall percentage above.

| Arm | Rows | L1 hit % | Raw L2 read sectors | Sampled long-scoreboard share % |
| --- | ---: | ---: | ---: | ---: |
| B | 50000 | 52.06 | 3644256214 | 76.14 |
| B | 300000 | 48.27 | 83051656826 | 95.13 |
| T1 | 50000 | 29.67 | 191783768 | 14.74 |
| T1 | 300000 | 17.29 | 5977954147 | 18.03 |
| T2 | 50000 | 19.06 | 207776570 | 16.04 |
| T2 | 300000 | 11.73 | 6199803424 | 16.82 |
| T4 | 50000 | 19.51 | 211092322 | 16.87 |
| T4 | 300000 | 11.62 | 6310018125 | 16.28 |
| F1 | 50000 | 49.51 | 110253381 | 9.88 |
| F1 | 300000 | 47.34 | 3394683171 | 7.65 |
| F2 | 50000 | 23.12 | 169638777 | 10.02 |
| F2 | 300000 | 23.28 | 5024922622 | 6.24 |
| F4 | 50000 | 26.54 | 160392434 | 10.23 |
| F4 | 300000 | 35.92 | 4115254260 | 6.38 |

### Reporting-only first fit and CPU crossover

Cold reporting is **50k×16 only**; CPU C16 versus F2 is **300k×16 only**. Each uses three processes per arm. On-disk caches were **not cleared**. Fresh-process first fits include initialization/JIT/cache effects, but are neither guaranteed cache-cold compilations nor clean JIT isolation. Summary medians: 50k B/F2 first fit .511248672/.453236648 s, warm core .081349122/.019258328, warm fit .352412340/.283256005; 300k CPU/F2 first fit 9.861194983/3.333289634, warm core 4.568857219/.471445421, warm fit 8.856450496/3.170908501 s. Ranges are tabulated below. F2 beats CPU in both core and full fit in this repeat; it does not locate a universal new crossover.

| Reporting cell | Arm | Processes | First-fit seconds | Warm core seconds | Warm full-fit seconds |
| --- | --- | ---: | --- | --- | --- |
| 50000×16 | B | 3 | .511248672 [.509199486,.542216513] | .081349122 [.080881877,.082980450] | .352412340 [.345315310,.353396557] |
| 50000×16 | F2 | 3 | .453236648 [.438009840,.453861062] | .019258328 [.019119082,.019780093] | .283256005 [.274036447,.289352386] |
| 300000×16 | CPU | 3 | 9.861194983 [9.381330220,10.188934254] | 4.568857219 [4.529079635,4.762274275] | 8.856450496 [8.740564941,8.934927530] |
| 300000×16 | F2 | 3 | 3.333289634 [3.304519547,3.334004207] | .471445421 [.467081494,.481710093] | 3.170908501 [3.136429184,3.191483266] |

Exact labels digests agree across arms within each S planes, S gate, acceptance, cold and crossover cell; this is stage-output evidence for these inputs, not a universal clustering claim.

### Scope and final verification

Fixed synthetic Gaussian k5/min-cluster-size5 measurements on this **RTX 3080 / Ryzen 9800X3D** host/software stack only. Experimental performance covers **50k×16, 300k×16**, plus T gate tests at **300k×2/8**. Step 0's 20k–300k 16D/30D baseline is not new-arm evidence. New-arm **20k/100k/200k/30D acceptance timing is unmeasured**; final integration checks correctness, not additional acceptance timing. Generalization between dimensions and above 16 is an assumption, not timing evidence.

**Final-source verification passed.** Under `artifacts/cubecl_kernels_perf/phase7/`, `final_gates/results.json` records exit0 for core tests, required-CUDA release and default debug suites, workspace/all-target clippy, formatting checks and Python analysis/campaign tests (145 passed). `final_gates/ir.json` records 44 tile variants with warp synchronization present throughout and no CTA barrier. `summary.json` records final verification passed. These receipts cover retained-source gates rather than reusing experimental-arm results.

`final_integration.json` records both required commands, `bash analyze_hdbscan_stages.sh --rows 20000` and `--rows 100000`, exiting **0** with real CPU/CUDA execution and exact `labels_equal=true`: `artifacts/hdbscan_stages/run-20000-bgHeQylR` and `artifacts/hdbscan_stages/run-100000-xQPWOUox`. Both counter-family receipts identify **`kd_neighbors_tile_kernel`**. These integration runs establish correctness/tool integration, not new acceptance timing cells. Independent review status/receipts are tracked separately in the Ultragoal ledger.

## CPU approximate-prediction leaf-SoA (Phase A)

**Accepted; Phase B remains excluded.** Evidence is in
`artifacts/approx_query_cpu/{baseline,soa}/`, `summary.json`, `provenance.json`,
`small_query_repeats.json`, `fit_regression.json` and `stage_validation.json`.
The baseline is pristine revision `63af3234b04cab7441120d07210273cdef111bb1`,
with only label/connector JSON output added to the same tracing-enabled release
profiler. Both binaries read identical non-quantized float32 matrices: eight
separated Gaussian clusters, seed `20261029+n+d`, min_samples 5. The final matrix
ran sequentially without competing builds/tests, with one cold and three warm
calls per process. Persistent JIT caches were not cleared; cold is the first
prediction after fit, not guaranteed cache-cold compilation. Raw traces separate
rank-5 core searches, rank-10 external searches and reference upload/preparation.
`summarize.py` regenerates input hashes and the trace-separated summary.

Warm end-to-end / warm query-search speedups (baseline divided by SoA):

| Reference rows | Features | 1k queries | All-row queries |
| ---: | ---: | ---: | ---: |
| 20k | 2 | 0.69× / 0.64× | 1.02× / 1.00× |
| 20k | 8 | 1.11× / 1.39× | 1.36× / 1.51× |
| 20k | 16 | 1.13× / 2.34× | 1.90× / 2.06× |
| 20k | 33 | 1.74× / 2.86× | 2.99× / 3.30× |
| 100k | 2 | 1.08× / 1.36× | 1.00× / 0.98× |
| 100k | 8 | 1.27× / 1.86× | 1.72× / 1.87× |
| 100k | 16 | 1.92× / 3.54× | 3.18× / 3.39× |
| 100k | 33 | 3.45× / 5.64× | 4.76× / 5.00× |

The apparent 20k×2/1k regression was investigated with **seven additional
alternating process pairs**, each retaining three warm calls and raw traces.
Median-of-process-medians was **1.950 ms baseline versus 1.998 ms SoA** (+2.4%,
0.047 ms), with overlapping process samples (baseline 1.524–2.322 ms, SoA
1.750–2.404 ms). This does not support a material small-query regression or a
query-count routing gate. Search improves in most shapes; narrow 2D searches are
roughly neutral. **Cold prediction regresses in 8/16 cases**, including both query
counts for 20k×2, 20k×8, 20k×16 and 100k×2. Examples are 20k×8/1k
**0.075 → 0.128 s**, 20k×8/all **0.121 → 0.152 s**, and 20k×16/1k
**0.111 → 0.147 s**. Cold core-search spans regress in four of eight reference
shapes: 20k×{2,8,16} and 100k×2. The first rank-5 batch at 20k×2/1k costs
approximately **106 ms SoA versus 42 ms baseline**, while the remaining core
batches total about 13 ms in each. This is consistent with extra first-launch/JIT
cost, but persistent caches were not cleared and these traces do not isolate JIT
causally. The warm-time acceptance gate does not erase these cold-call tradeoffs.
The routing remains CPU SoA for all shapes, with no fallback. Larger-feature
cold/core gains and preparation costs remain explicit in `summary.json`.

All **16 cases had zero label and connector differences** against baseline;
each profiler invocation independently checked cold/warm labels, strengths and
connectors. This measures these inputs only: contraction-sensitive near-tie
differences remain permitted. Planned device peaks increase, not decrease: for
example 100k×16/1k rises **14,521,860 → 21,113,856 bytes**, and 20k×33/1k rises
**6,246,308 → 8,610,272 bytes**, including the temporary padded preparation rows.

Both required stage analyses passed with real CPU/CUDA execution and exact
labels: `run-20000-aCNH51me` and `run-100000-DBaYLcaS`. The latter measured CPU
fit **1.306 s**, CUDA **0.589 s**, and retained CUDA's `kd_neighbors_tile_kernel`
fit family. The historical Phase 7 integration directories cited above are not
present in this checkout, so they are not claimed as a fresh timing comparison.
Instead, three alternating baseline/SoA process pairs on the exact final-stage
100k×16 input, with three measured fits after one excluded warmup each, gave CPU
fit medians-of-process-medians **1.346 s versus 1.326 s**. This supplies a current
same-input fit-regression check against the accepted pre-change C16 source.

### Requested 300k×16 comparison, 10k queries

Using the same release binaries, seeded eight-cluster workload and min_samples 5,
CPU prediction measured **11.337 → 2.327 s cold (4.87×)** and
**384.975 → 83.511 ms warm median (4.61×, three calls)**. Labels and connectors
had zero differences across all 10k queries; each binary checked cold/warm
strengths too. Planned device peak increased **42,040,964 → 62,227,968 bytes**.
These are one process per arm, not confidence intervals. Raw outputs/traces and
input SHA-256 are retained under `artifacts/approx_query_cpu/300k/`.

## CPU prediction leaf-coherent scheduling (Phase B)

**Decision: reject both production scheduling arms.** Phase A's CPU leaf-SoA
candidate pack and input-order queries remain the production baseline. This is
a negative experiment, not a delivered prediction speedup. CUDA and fit kernels
are unchanged. The approved contract is [approx_query_cpu_plan_phaseB.md](approx_query_cpu_plan_phaseB.md).

Evidence is under `artifacts/approx_query_cpu_phaseB/`: frozen baseline binaries,
four experimental source diffs and binary hashes, stored float32 inputs and hashes,
raw process outputs/traces, `summary.json`, `correctness.json`, `decisions.json`,
and the reproducible `run_experiment.py` summarizer. The 21 cells cover the 16
historical clustered shapes, 300k×16/10k scaling, and uniform/overlapping 100k×{2,16}
controls. Every candidate has three initial and seven repeated processes, each
paired with A in alternating order; each process has three warm calls. Process
medians, not individual warm calls, are the statistical units. Shared JIT caches
were retained; first prediction is not asserted cache-cold.

All **1,260 exact comparisons** agree on labels, raw connectors, f64 probability
bits and compact-ID core bits. In-crate device tests additionally compare ordered
candidate IDs and squared-distance bits after inverse mapping. The frozen Phase A
product source, rebuilt with only exact-output profiler instrumentation, also
matches A exactly. Original frozen-binary timing and label/connector parity runs
are retained separately; timing noise is visible rather than treated as proof of
equal latency.

### Independent decisions

- **C, cold-core leaf gathering: reject.** Paired cold median improves about 9.9%
  (238 ms) at 300k×16/10k and about 10% at clustered 100k×16/1k. However, repeated
  overlapping 100k×2 and ×16 cold calls regress about 3.0% (9.6 ms) and 1.5%
  (43.5 ms), and uniform 100k×16 regresses about 1.0% (31.7 ms). Unconditional
  retention is not justified across relevant workloads. No speculative shape gate
  is introduced.
- **Q, external-query affinity buckets: reject.** Key construction, unstable sort,
  gathering/upload and output scatter are included in end-to-end measurements.
  Representative full-query warm regressions are about 9.8% at clustered
  100k×33, 11.1% at uniform 100k×2, and 10.1% at overlapping 100k×2. Search-only
  locality benefits do not justify these total costs.
- **CQ: reject.** Combining the arms does not remove external scheduling costs;
  independent core gains do not constitute combined acceptance.

`summary.json` records paired ratios and absolute deltas for every cell, planned
host/device peaks, batch sizes, and cold-core/cold-external/warm-external span
totals. Spans are inclusive and must not be summed across nested scopes. Plans
are logical live-buffer bounds, not RSS measurements. A corrects the shared
transactional output-column and f64 math-scratch accounting omissions; its host
peak increase versus frozen Phase A is accounting, not a new physical allocation.

The user explicitly reduced any further performance rerun to two representative
cases, one paired process each with three warm calls. The completed full matrix
is preserved; final-source evidence is deliberately narrower and does not claim
a new full-matrix replication.

Final-source Rust core/facade tests, strict CUDA prediction/KD tests, Clippy,
formatting, release Python extension build and 156 Python prediction/predictor
tests passed. Both native stage analyses passed on CPU/CUDA at 20k and 100k:
`run-20000-AMHHE8hJ` and `run-100000-nppzZfX2`. Existing warnings and ignored tests
remain visible. Final/A cold smoke checks were 115.3/114.1 ms at 20k×2/1k and
458.1/460.7 ms at 100k×16/1k, with bit-exact predictions and cores. These single
pairs establish no statistical speedup. Details and final binary hash are in
`final_gates/{verification,performance}.json`. Production contains no experimental
order constants or rejected leaf routes; alternatives exist only in unit tests.
