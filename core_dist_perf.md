# Initial hypotheses (superseded by measured results): why the KD-tree kernels might trail Numba

Oct 21, 2026

Three things likely account for most of the gap: Borůvka pruning that never shares a bound across a component (numba does), the CubeCL CPU runtime's cube-by-cube execution, and per-thread state that lives in bounds-checked global memory on CUDA. Impact ranks are estimates from reading the code; the node-visit count in the first section tells you which half of the list matters for your data.

Initial proposed quick wins (unvalidated hypotheses): tree-ordered rows (3), stacked bounds (7), `launch_unchecked` (8), early exit once best ≤ core (9). See the measured dispositions below; not all proposals were retained.

## Measure first

Before changing kernels, confirm the 100×/240× is kernel time, then find out whether the gap is more work or slower work.

- **Exclude compilation.** The first launch of each kernel JIT-compiles it (NVRTC on CUDA, MLIR on the CPU runtime). fast_hdbscan caches its compiled numba functions on disk by default ([README](https://github.com/TutteInstitute/fast_hdbscan/)), so its timings usually exclude compiling. Warm up once, then time.
- **Time at a sync point.** Burn launches are asynchronous; only `into_data`, `into_scalar` and `sync` wait for the device ([Burn Book](https://burn.dev/books/burn/performance/good-practices/asynchronous-execution.html)). Sync before and after each kernel call when profiling.
- **Split the time.** Time `exact_neighbors_kd` and `boruvka_candidates_kd` separately, count Borůvka rounds, and time the driver work between rounds.
- **Count nodes visited.** Add a debug output of nodes popped and leaf points scanned per row, and log the same in numba. Much higher counts mean causes 1 and 9 dominate; similar counts mean per-node cost (causes 3–8) is the problem.

## Initial hypotheses (superseded by measured results)

Ordered by estimated impact across both targets; the CPU and CUDA columns show where each one bites.

| # | Cause | Where | CPU impact | CUDA impact | Fix | Effort |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | No shared per-component pruning bound; round 1 not seeded from k-NN | `kd_candidates_kernel`, driver | Very high | Very high | Per-component best via atomic min; skip rows and candidates by it; seed from k-NN edges | Medium |
| 2 | Dispatch geometry: 32-unit cubes run one after another on CPU, each a fork-join of 32 uneven traversals | Host launch config | Very high | — | Size the cube to the core count and loop rows inside each unit | Low |
| 3 | Adjacent threads query unrelated rows, so warps diverge | Both kernels | Low | High | Map threads to rows in tree order | Low |
| 4 | k-NN heap kept in global output tensors | `kd_neighbors_kernel` | Medium | High | Comptime-k local array, write once | Medium |
| 5 | Query point and shapes re-read per feature, per node | Both kernels | Medium | High | Comptime-d local query, unrolled loops | Low |
| 6 | Candidates gathered through the `indices` permutation | Both kernels | Low–medium | High | Store data, core and labels in tree order | Medium |
| 7 | Each box bound computed twice | Both kernels | Medium | Medium | Stack (node, bound) pairs; prune at push | Low |
| 8 | Bounds-checked `launch` | Both kernels | Medium | Low–medium | `launch_unchecked` after host validation | Low |
| 9 | Min-index tie rule scans the whole core-distance ball | `kd_candidates_kernel` | Data-dependent | Data-dependent | Stop once best ≤ core | Low |
| 10 | 64-bit `usize` index arithmetic | Both kernels | Low | Low–medium (verify) | `u32` indices | Low |
| 11 | Leaf size and cube size untuned | Tree build, launch | Low–medium | Medium | Larger GPU leaves; sweep 64–128 units | Low |
| 12 | Host work between Borůvka rounds (not in snippet) | Driver | Unknown | Unknown | One sync per round; batch reads; stay on device | Varies |

## Causes 1–3

### 1. No shared per-component pruning bound

The numba Borůvka keeps one bound per component, shared by every point in it ([source](https://evoc.readthedocs.io/en/latest/_modules/evoc/boruvka.html)). It abandons a subtree when the box bound or the query's own core distance exceeds that bound, and skips leaf candidates whose core distance is already worse. Your kernel starts each row at `best = ∞` and learns only from its own leaves, so every point in a large component repeats most of the search.

Numba also seeds Borůvka from the k-NN graph it built for core distances, merging many components before the first tree query. If your driver skips that step, round 1 (n singleton components, the most expensive round) runs in full.

Sketch for f32 (atomic API names vary by CubeCL version):

```rust
// component_best: one u32 per label, reset to f32::INFINITY.to_bits() each round.
// Non-negative f32 values sort the same as their bit patterns, so an atomic u32 min works.
let comp_bound = f32::from_bits(Atomic::load(&component_best[component]));
if core <= comp_bound {
    // traversal: prune a node when lower > min(best, comp_bound), re-reading comp_bound per leaf
    // leaf loop: test a candidate only if core_sq[candidate] <= comp_bound
    if weight < best {
        best = weight;
        best_index = candidate;
        Atomic::min(&component_best[component], weight.to_bits());
    }
}
```

Prune with strict `>` so the component's true minimum edge survives. A shared bound makes *which* point reports an equal-weight edge order-dependent; numba offers a block-synchronous "reproducible" mode for that, and you can mimic it by reading the bound from the previous launch chunk only. Labels are union-find roots below n, so an n-sized buffer indexes them.

### 2. The dispatch geometry serializes the CPU into 32-wide steps

On the CPU runtime, a cube's units are spread across cores and cubes run as a loop ([CubeCL docs](https://docs.rs/crate/cubecl/0.11.0-pre.2)). Both kernels launch `KD_CUBE_DIM = 32` units per cube and about n / 32 cubes, one row per unit. The runtime is doing what that geometry asks, but the geometry is a GPU shape that maps badly to a CPU.

That shape causes three problems on CPU:

- **Capped parallelism.** At most 32 rows are in flight at any moment. On machines with more than 32 cores, the rest get no work.
- **A barrier every 32 rows.** Each cube must finish before the next starts. Traversal lengths vary widely, so most cores idle while the slowest unit in each cube finishes.
- **Per-cube overhead.** About n / 32 dispatches per launch, each paying the fork-join cost.

Numba's `prange` has none of these. All rows go to all cores in one parallel loop, with a single join at the end.

The fix is to size the dispatch for the device and loop over rows inside each unit. On CPU, launch one cube with about as many units as cores. The cubecl-cpu scheduler starts one worker per core and adds more when a cube needs them ([dispatcher source](https://docs.rs/crate/cubecl-cpu/0.11.0-pre.1/source/src/compute/threadpool/scheduler/dispatcher.rs)), so more units than cores only oversubscribes. On GPU, keep many cubes. The same strided loop serves both:

```rust
let stride = (CUBE_COUNT_X * CUBE_DIM_X) as usize;
let mut local_row = ABSOLUTE_POS_X as usize;
while local_row < row_count as usize {
    let row = local_row + row_offset as usize;
    // existing per-row body
    local_row += stride;
}
```

Striding gives each unit an even mix of cheap and expensive rows, because rows that are close in tree order tend to cost about the same. If imbalance remains, have each unit claim small chunks of rows from an atomic counter instead, which approximates numba's dynamic scheduling. On the host, choose the cube dim and cube count per device rather than using `KD_CUBE_DIM` everywhere, and check what `legal_cube_dim` allows on CPU.

### 3. Neighbouring GPU threads walk unrelated parts of the tree

Thread i handles row i, and row order has no spatial meaning. The 32 threads of a warp take different branches and pop different nodes, and the warp runs as long as its slowest thread.

Map threads to rows in tree order:

```rust
let row = u32::cast_from(indices[local_row + row_offset as usize]) as usize;
```

Points in the same leaf then follow nearly the same traversal and the warp stays converged. If that isn't enough, the standard GPU design is leaf-blocked: one cube per query leaf, a shared traversal using the query leaf's box, and the cube's threads scanning each candidate leaf together with coalesced loads.

## Causes 4–6

### 4. The k-NN heap lives in global memory

Every node pop reads `out_distances[row * width]`, and each sift-down does several global reads and writes, all bounds-checked. On a GPU each is a round trip to L2 or DRAM per comparison; numba's heap stays in L1.

Make `k` a comptime parameter, keep the heap in a local `Array`, and write the sorted result once at the end. For small k (up to about 16), a sorted insertion list with `#[unroll]` loops avoids dynamic indexing and can stay in registers.

```rust
#[cube(launch_unchecked)]
fn kd_neighbors_kernel<F: Float, I: Int>(/* … */, #[comptime] k: usize, #[comptime] d: usize) {
    let mut best_d = Array::<F>::new(k);
    let mut best_i = Array::<u32>::new(k);
    #[unroll]
    for s in 0..k {
        best_d[s] = F::new(f32::INFINITY);
        best_i[s] = u32::cast_from(n);
    }
    // traversal prunes against best_d[k - 1], the current worst
    #[unroll]
    for s in 0..k {
        out_distances[row * k + s] = best_d[s];
        out_indices[row * k + s] = I::cast_from(best_i[s]);
    }
}
```

This compiles one kernel per (k, d) pair, which is fine for the one or two values you use.

### 5. The query point and shapes are re-read for every node and candidate

`box_lower_bound` and the leaf distance loop read `data[row * d + feature]` from global memory each time, and `data.shape(1)` is a metadata read on every call. That is 3d global loads per node bound where 2d would do.

Load the query once into a local `Array<F>` of comptime length d and unroll the feature loops. When d is a multiple of 4, vectorized loads cut instruction count further; the CubeCL book reports about 3× from vectorizing its reduction example ([CubeCL Book](https://burn.dev/books/cubecl/getting-started/vectorized_reduction.html)).

### 6. Candidates are gathered through the index permutation

Leaf scans read `indices[position]`, then `data[candidate * d …]`, `components[candidate]` and `core_sq[candidate]` at scattered addresses. On a GPU each is a separate memory transaction per thread.

Store `data` and `core_sq` permuted into tree order once per build, and compute each round's labels in tree order too. Leaf ranges become contiguous, threads scanning the same leaf (after cause 3) share cache lines, and `indices` is needed only to translate the final result index.

## Causes 7–9

### 7. Every box bound is computed twice

Expanding an internal node computes both children's bounds to pick the order; popping each child computes its bound again. That doubles the dominant per-node cost, which is d-length work per call.

Push the bound with the node (a second `Array<F>` stack beside the node stack). On pop, compare the stored bound with the current best, which may have tightened since. Skip pushing a child whose bound already exceeds best. Numba passes the bound down its recursion the same way ([source](https://evoc.readthedocs.io/en/latest/_modules/evoc/boruvka.html)).

### 8. Kernels are launched with bounds checks

`#[cube(launch)]` produces a checked kernel: every tensor index gets a range check. CubeCL's default mode enforces checks on standard launches and skips them only on explicitly unchecked ones ([BoundsCheckMode](https://docs.rs/zenforks-cubecl/latest/cubecl/config/compilation/enum.BoundsCheckMode.html)).

Your host code already validates shapes, dtypes and stack depth, so switch to `#[cube(launch_unchecked)]` and call it inside `unsafe`. The kernel must then never index out of range or loop forever ([safety contract](https://docs.rs/cubecl-std/0.11.0-pre.1/cubecl_std/throughput/memory_direct/memory_direct_throughput/fn.launch_unchecked.html)). Run tests in the `Validate` mode, which keeps checks and flags out-of-bounds access in unchecked kernels.

### 9. Exact tie-breaking forces a scan of the whole core-distance ball

Mutual-reachability weights are floored at the query's core distance, and so are node lower bounds. Once a row finds an edge of weight exactly `core`, which is common, nothing can beat it. The `<=` prune and min-index tie rule still make the kernel visit every node whose bound equals `core`.

Stop the traversal once `best <= core`. You give up "smallest index among equal weights", but without the shared bound from cause 1 results stay deterministic, because each thread's traversal order is fixed. Numba's leaf test replaces only on a strictly smaller weight, so it makes no min-index promise either.

## Causes 10–12

### 10. 64-bit index arithmetic

Indices are computed as `usize`, with `u32::cast_from(…) as usize` round trips. If your CubeCL version lowers `usize` to 64-bit on CUDA, each index multiply-add costs several 32-bit instructions. Inspect the generated CUDA source; if it shows 64-bit math, switch to `u32` indices, since your row counts already fit in `i32`.

### 11. Leaf size and cube size are untuned

GPUs favour larger leaves than CPU trees, around 64–256 points: fewer internal nodes to traverse and longer coalesced scans once cause 6 is fixed. A cube of 32 units is one warp; 64–128 often hides latency better, but the stack and heap arrays add register pressure, so measure.

Burn's built-in kernels pick configurations through autotune with an on-disk cache ([Burn Book](https://burn.dev/books/burn/performance/good-practices/kernel-selection.html)). Custom kernels like these don't get that, so sweep leaf size and cube size by hand.

### 12. The host loop between Borůvka rounds

The driver isn't in the snippet, but every readback is a full device stall. Check that each round syncs at most once, and batch the reads you need with a `Transaction` ([Burn Book](https://burn.dev/books/burn/performance/good-practices/asynchronous-execution.html)). Keep the tree, data and label buffers on the device across rounds rather than re-uploading them.

## Sources

- [Burn Book: Performance good practices](https://burn.dev/books/burn/performance/good-practices/index.html)
- [Burn Book: Asynchronous execution](https://burn.dev/books/burn/performance/good-practices/asynchronous-execution.html)
- [Burn Book: Kernel selection](https://burn.dev/books/burn/performance/good-practices/kernel-selection.html)
- [CubeCL Book: Vectorized reduction](https://burn.dev/books/cubecl/getting-started/vectorized_reduction.html)
- [CubeCL README: execution model on CPU and GPU](https://docs.rs/crate/cubecl/0.11.0-pre.2)
- [CubeCL `BoundsCheckMode` (docs from a published fork)](https://docs.rs/zenforks-cubecl/latest/cubecl/config/compilation/enum.BoundsCheckMode.html)
- [CubeCL `launch_unchecked` safety contract](https://docs.rs/cubecl-std/0.11.0-pre.1/cubecl_std/throughput/memory_direct/memory_direct_throughput/fn.launch_unchecked.html)
- [Tutte Institute numba Borůvka (EVoC copy of fast_hdbscan's design)](https://evoc.readthedocs.io/en/latest/_modules/evoc/boruvka.html)
- [fast_hdbscan README](https://github.com/TutteInstitute/fast_hdbscan/)

## Execution results (2026-10-21)

The ranked causes above are the initial user hypotheses, not measured findings; this results section supersedes their performance claims. The initial statement that Borůvka round one was unseeded is incorrect: the current driver seeds from k-NN edges. Measurements compare the untouched baseline in `baseline/verified-results.json` with the repaired implementation in `evidence/repaired-quality.json` and all four repaired stage records. The Python comparator is `fast_hdbscan==0.3.2`; its fused Numba `core_distances_and_mst` time is end-to-end reference work, not an isolated MST or core-distance kernel.

### Measurement scope and results

Input: `comparison/embeddings_2d.parquet`, 2D float32, seed 42, `min_cluster_size=5`, `min_samples=5`, EOM selection, `allow_single_cluster=false`. The actual sample sizes are 10,000 and 272,383; no million-row result or substitute is claimed. Each baseline and repaired record has one excluded warmup and one measured fit, so entries are individual observations, not medians or stable estimates. Ratios below are explicitly **reference time / native time**. A ratio above 1 means the native fit was faster in this observation; below 1 means it was slower.

| Rows | Backend | Baseline native / reference (s) | Repaired native / reference (s) | Repaired reference/native |
| ---: | :--- | ---: | ---: | ---: |
| 10,000 | CPU | 0.015404594 / 0.018981560 | 0.017938040 / 0.024252719 | 1.35× |
| 10,000 | CUDA | 0.017095532 / 0.018755669 | 0.022477085 / 0.020474241 | 0.91× |
| 272,383 | CPU | 0.484315618 / 0.805651548 | 0.467865100 / 0.809748053 | 1.73× |
| 272,383 | CUDA | 0.285030994 / 0.782407032 | 0.244727063 / 0.821624321 | 3.36× |

The repaired native observations are slower than baseline at 10k (+16.4% CPU, +31.5% CUDA) and faster at 272,383 (-3.4% CPU, -14.1% CUDA). These are single samples and do not establish a stable speedup or regression. The original 10k CPU regression exceeding 5% was investigated with `evidence/small-cpu-regression-three.json`: three same-condition observations were 0.016450692, 0.016208174, and 0.016244030 s. Their small spread indicates measurement variability but does not erase the repaired 0.017938040 s single observation or guarantee a speedup. Cached bounds avoid repeated bound arithmetic; CPU-legal striding reduces the number of sequentially scheduled cubes; retained diagnostics support observability, testability, and review. These are concrete engineering reasons for retaining the changes, not proof of performance benefit.

Repaired native stage records show `Cpu` and `Cuda { device_index: 0 }`; reference metadata records 15 Numba threads using `tbb`, 15 native CPU cores, and affinity count 15. Reference stage `core_distances_and_mst` is fused and timed as part of the reference fit. Native Borůvka timings are nested inclusive measurements: candidate spans include node-label generation/upload, kernel work, synchronization, and readbacks; driver-and-seed includes k-NN seeding, labels, synchronization, and host merges. The profiled native timing includes tracing synchronization and readbacks; it is not an uninstrumented fit time and does not isolate kernel time. No OOM occurred in these runs. The 0.216481089, 0.308323776, and 0.294061617 s CUDA observations belong to the earlier P2 stack experiment, not to the repaired final timing set, and are not used to infer final performance.

The repaired build's small-input regression was separately investigated on both backends, with the profiler binary SHA-256 and exact argv in `evidence/repaired-small-regression-three.json`. CPU samples were 0.016855996, 0.016214782, 0.015636654 s; CUDA samples were 0.017827074, 0.019198723, 0.018129279 s. Against the baseline samples these range from +1.5% to +9.4% CPU and +4.3% to +12.3% CUDA. Thus the original +16.4%/+31.5% observations are not stable estimates, but a small-input cost remains plausible. Disposition: accept this bounded workload-specific cost for the measured final-size improvement, reduced repeated bound arithmetic, legal CPU scheduling, and improved diagnostic/test coverage. No across-the-board speedup is claimed. The earlier CPU three-sample investigation is historical, not proof about this repaired binary.

### Quality and traversal diagnostics

For each size, repaired CPU and CUDA labels have identical canonical i32le digests to the corresponding **pre-change native** labels: 10k `488553e9…c3f5e7`; 272,383 `da684901…f40af7`. The full digests and source paths are in `evidence/repaired-quality.json`. This establishes native-label identity across baseline/repaired outputs. No reference ARI was calculated for this report; the ARI fields in the evidence file are not used here to claim reference agreement.

The first six diagnostic entries in `evidence/diagnostics-original-cpu.log` are the initial `kd_neighbors` call followed by five `kd_candidates` rounds from the repeated warmup trace. Each entry lists total (min–max per-row):

| Trace entry | Operation | Nodes popped | Leaf positions scanned |
| ---: | :--- | ---: | ---: |
| 1 | `kd_neighbors` | 161,638 (15–37) | 1,066,088 (78–392) |
| 2 | `kd_candidates`, round 1 | 171,216 (15–43) | 1,285,977 (78–391) |
| 3 | `kd_candidates`, round 2 | 194,096 (15–43) | 1,791,177 (78–547) |
| 4 | `kd_candidates`, round 3 | 232,836 (15–49) | 2,520,010 (78–702) |
| 5 | `kd_candidates`, round 4 | 281,334 (15–63) | 3,084,944 (78–858) |
| 6 | `kd_candidates`, round 5 | 260,742 (15–63) | 2,241,143 (78–625) |

Equivalent Numba inner-traversal counters are unavailable: the pinned reference uses a fused JIT path with no supported hook for these inner counters. Instrumentation was not attempted by rewriting or altering the reference implementation. The native diagnostic switch is tracing-only, controlled by `FAST_HDBSCAN_KD_DIAGNOSTICS=1`, and CPU tests cover both enabled and disabled behavior.

### Disposition of all twelve initial hypotheses

| # | Disposition | Evidence / scope |
| ---: | :--- | :--- |
| 1 | **Not adopted.** | The driver already seeds Borůvka from k-NN edges, so the unseeded-round premise was false. A shared per-component pruning bound remains unimplemented; tied-edge/publication correctness was not established. The diagnostic counts above do not provide a native-vs-Numba work comparison. |
| 2 | **Retained: CPU striding only.** | CPU-legal row striding remains to reduce sequential cube scheduling. No isolated ablation proves a performance gain. |
| 3 | **Rejected and reverted.** | Tree-order queries were reverted because the public tree's index array was not validated as a true permutation and could alias outputs. Natural original row ownership is restored; no O(n) output readback or public-API redesign was introduced. |
| 4 | **Not adopted.** | Local k-NN storage/register specialization remains conditional; no generated-code, register-spill, or memory-traffic evidence established safe net benefit. |
| 5 | **Not adopted.** | Query-local storage/vectorization remains conditional; no generated-code/spill proof or controlled feature-loop measurement established net benefit. |
| 6 | **Not adopted.** | Persistent tree-ordered data/core/label storage was not adopted; memory-footprint and benefit evidence were not established. |
| 7 | **Retained: cached bounds.** | Cached node bounds/stacks remain, avoiding recomputation of bounds. This is an engineering rationale, not an isolated causal performance result. |
| 8 | **Rejected.** | Unchecked launch was not adopted because host validation does not establish the complete indexing/safety proof required by its contract. |
| 9 | **Not adopted.** | Early exit at `best <= core` would alter the per-row tied-endpoint/min-index behavior. The diagnostic trace is native-only and supplies no Numba comparison; no semantics change was justified. |
| 10 | **Not adopted.** | Index-width narrowing remains conditional; no generated-code evidence showed costly 64-bit index arithmetic on this workload. |
| 11 | **GPU cap proposal rejected; other tuning not adopted.** | GPU cube cap 32 was rejected. No controlled leaf-size/launch-geometry sweep established a net benefit without adverse stack/register effects. |
| 12 | **Partly retained; otherwise not adopted.** | Core-distance upload remains hoisted in the driver. Readback batching and persistent device-resident tree/data/labels were not adopted without measured benefit evidence; nested timings do not isolate batching. |

The zero-progress guard was removed as duplicate: the core merge routine already errors on zero merges. The diagnostics switch is an environment-controlled, tracing-only facility and enabled/disabled CPU tests were added. These dispositions do not imply a performance gain. The earlier P2 CUDA stack experiment is not the repaired result set. Reviews are not represented as final approval; the boundary reviews recorded request-changes/blocking findings, and no final sign-off is claimed here.

Evidence: `.gjc/_session-01a1606f-39b2-7094-8207-39df78b4c165/ultragoal/baseline/verified-results.json`; `.gjc/_session-01a1606f-39b2-7094-8207-39df78b4c165/ultragoal/evidence/repaired-quality.json`; and `evidence/repaired-{10000,272383}-{cpu,cuda}/stages.json`, `evidence/small-cpu-regression-three.json`, and `evidence/diagnostics-original-cpu.log` under that session directory. Baseline was captured by the untouched Rust baseline runner under the pinned workspace lock; the Python reference remained `fast_hdbscan==0.3.2`.
