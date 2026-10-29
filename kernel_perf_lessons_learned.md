# Kernel performance lessons learned

## The central lesson

**Optimize how work reaches the machine, not just how much mathematical work the algorithm performs.** The largest wins here came from removing repeated dependent loads, choosing a SIMD axis with contiguous data, and replacing mostly ineffective tree traversal with a regular, cooperative scan. Fewer visited candidates, higher occupancy, and fewer source-level loads are useful hypotheses—not optimization objectives by themselves.

This synthesis reviews [the experiment plan](cubecl_kernels_perf_plan.md) and the [report's follow-ups and subsequent results](hdbscan_perf_report.md#follow-ups-cubecl-kernel-patterns-for-core_distances), emphasizing accepted Phases 3, 6, 7 and 1. The original F1–F5 proposals are hypotheses; the later phase results establish what was actually retained. For example, Phase 7 retained an output-backed heap, not the proposed register top-k.

### What actually worked

All percentages below are **reductions in packing-inclusive core-stage time against each experiment's own baseline**, not independent gains against one common baseline. They must not be added or multiplied into a claimed cumulative speedup.

| Experiment | Retained change | Measured result | Generalizable lesson |
| --- | --- | --- | --- |
| [Phase 3](hdbscan_perf_report.md#phase-3-results-padded-vector-data-and-register-resident-query) | Padded four-wide loads, query cached in registers, exact feature-count specialization | Primary cells: CPU about 24–29% faster; CUDA about 62–70% faster | Hoist invariant operands and make memory operations/compiler-visible shapes explicit |
| [Phase 6 valid-tree retry](hdbscan_perf_report.md#phase-6-valid-tree-retry-results-c16-accepted) | CPU candidate-axis SoA SIMD, width cap 16, leaf-block queries (C16) | Primary cells: about 76–81% faster | Lay out data along the independent SIMD axis; isolate layout from scheduling |
| [Phase 7](hdbscan_perf_report.md#phase-7-results-cuda-flat-plane-kernel-f2-accepted) | CUDA flat scan, plane-local staging/broadcast, two planes per cube (F2) | 50k×16: 76.45% faster; 300k×16: 91.27% faster | Regularity and explicit reuse can beat pruning when traversal skips little work |
| [Phase 1](hdbscan_perf_report.md#phase-1-results) | Register-cached heap root, refreshed only after admission | 50k CUDA: about 15.7% faster at 16D and 26.2% at 30D; CPU gain not demonstrated | Cache the small, frequently read decision state rather than redesigning all state |

These are measurements on one Ryzen 9800X3D / RTX 3080 system and a fixed synthetic workload family, primarily k5. They demonstrate patterns worth testing elsewhere, not portable speedup promises.

## 1. Identify the expensive work before choosing an optimization

### Pattern to look for: a large named stage hiding one dominant operation

Initially, synchronized neighbor search accounted for more than 99% of the CPU core stage and about 99% on CUDA. Extraction and compact-output readback were tiny by comparison. Optimizing either could not remove seconds of traversal time.

**Apply:** establish a timing hierarchy: end-to-end operation → stage → synchronized device operation → kernel. Check actual dispatch and launch counts. A historical implementation's many launches are not evidence that the current implementation is launch-bound; this KD path already used one neighbor launch per fit.

Measure asynchronous operations at a boundary that includes completion. Otherwise an enqueue may look cheap while its work is charged to readback or a later synchronization. Do not sum nested inclusive spans or expect independent medians to add exactly.

### Pattern to look for: an algorithm doing almost all the work it was meant to avoid

Diagnostics showed each query scanning about 74% of candidates at 16D and 96.5% at 30D, but only 0.63% at 2D. The tree was paying for bounds, branches, stacks and gathers while eliminating little work in the higher-dimensional cases.

**Apply:** instrument work, not just elapsed time: candidates examined, nodes visited, rejected work, reuse opportunities and useful output. Compare these counts with a simple dense-work baseline. When pruning is weak, test a regular scan or blocked algorithm; when pruning is strong, preserve the sparse path.

The old brute-force comparator being slower did not disprove this opportunity: it had its own batching and partial/merge overheads. Compare implementations and access patterns, not just algorithm names.

## 2. Cache values according to how often they change

### Pattern to look for: a read on every iteration, a write only on rare events

Phase 1's heap root was consulted for every candidate and for child pruning, but changed only when a candidate was admitted and the heap repaired. Explicitly retaining `(worst_distance, worst_id)` in registers removed repeated dependent global loads; the cache was refreshed after the mutation.

**Apply:** separate the common read-only decision path from the uncommon update path. Cache the minimal sufficient state and define its refresh point precisely. Similar opportunities occur with thresholds, current minima/maxima, queue boundaries and convergence state.

Phase 3 applied the same principle at a different lifetime: query features remain constant throughout a query's traversal, so load them once and reuse them in candidate distances and box bounds. Source-level repetition may survive compilation when runtime indexing, alias uncertainty or intervening stores prevent hoisting. Confirm the emitted code rather than assuming the compiler fixes it.

**Do not extrapolate to “put everything in registers.”** Phase 2's full register top-k was rejected despite some CUDA wins. Its insertion/select chains, resource costs and regressions made it a worse trade than caching just the root. Register storage saves memory traffic but can add instructions, extend live ranges and alter code generation even on nominally unchanged paths.

## 3. Choose the parallel axis before choosing the layout

### Pattern to look for: nearby SIMD lanes gather unrelated addresses

The original CPU candidate access followed `indices[position]` into row-major data. Phase 6 packed position tiles as:

```text
soa[tile][feature][lane]
```

Each SIMD lane represented a different candidate. For one feature, the kernel loaded a contiguous vector and updated independent lane accumulators. It advanced features in their original order.

**Apply:** identify an independent outer dimension—candidates, queries, records or cells—and transpose/block storage so adjacent lanes consume adjacent values. Keep tails and masks on that dimension. A feature count of 3 or 30 need not prevent wide SIMD when lanes represent candidates rather than features.

Phase 3 used vectors differently: four-wide vectors were **load containers**, with lanes extracted and accumulated in scalar feature order. Phase 6 used vectors for **independent arithmetic across candidates**. These are distinct techniques; vector-shaped source code alone does not establish SIMD arithmetic.

### Pattern to look for: a proposed transformation combines several explanations

The initial input-order probe changed both candidate locality and query scheduling. It could show sensitivity, not identify the cause.

Phase 6 separated baseline, packed-only (P), ordering-only (O), and combined (C) arms. Packed-only and combined had large primary gains; ordering-only had no qualifying primary gains. C16 was selected, but the evidence attributes the dominant benefit to packing/vectorization—not query reordering alone. The combined arm was not fastest in every cell.

**Apply:** use ablation experiments to distinguish layout, scheduling and their interaction. Keep public row identity separate from packed position. Leaf-ordered scheduling can write results back to original rows without permuting the public input.

## 4. Trade irregular work avoidance for regular data reuse

### Pattern to look for: many workers repeatedly fetch overlapping data while following different control paths

CUDA's original per-query traversals combined candidate gathers, private stacks and dependent memory operations. When most queries examine most candidates anyway, separate traversals duplicate traffic and obstruct reuse.

Phase 7 staged leaf-ordered padded candidate rows in plane-local shared storage. Query lanes reused staged candidates through broadcasts, retaining their own ordered distance accumulation and heap. The accepted flat arm removed traversal altogether. Two independent planes shared a cube, with plane-scoped synchronization.

**Apply:** give a cooperative group a reusable data tile, load it collectively, and amortize those loads across several independent consumers. Synchronize before consumption and before overwrite. Match synchronization scope to ownership: independent planes need not wait on one another through a cube-wide barrier. Keep collective operations convergent and test on real cooperative hardware.

The decision is about total cost:

```text
traversal + bounds + scattered candidate loads + stack traffic
    versus
regular extra candidate evaluations + staging + synchronization
```

More arithmetic can be faster when it buys predictable control flow and much lower data movement. Phase 7's flat scan had zero measured local-memory sectors, whereas the traversal arm retained stack traffic. At 300k×16, baseline/F2 DRAM reads were approximately 1.93 TB / 8.39 GB in the profiled launches, with much lower long-scoreboard stalls. These support the reuse explanation, without isolating every cause.

**Bound the claim:** Phase 7 acceptance timing covered 50k×16 and 300k×16, not new-arm 30D timings. The retained feature gate of 16 is not a universally established crossover. The unretained traversal arm's 2D regression is evidence against indiscriminate cooperative routing, not a measurement of F2 at 2D. The flat scan remains quadratic; improved constants and reuse do not establish scalable performance beyond the measured sizes. Well-separated data with stronger pruning could favor a different choice.

## 5. Spend resources where they remove the bottleneck

### Pattern to look for: low occupancy or high register counts being treated as verdicts

Phase 3 increased registers per CUDA thread from 75 to 106 at 16D and to 128 at 30D, while substantially improving time. Achieved occupancy stayed around 18–19%. The extra registers removed far more expensive loads.

In Phase 7 at 300k, F2 achieved occupancy was about 63.16%, versus 63.41% for baseline. The large speedup was not explained by higher achieved occupancy. Four-plane variants could have higher occupancy without being faster than two-plane variants.

**Apply:** treat registers, occupancy, shared memory and block limits as constraints in a cost model. Tune launch geometry after meaningful kernel changes; compiler register allocation itself can change with geometry. Benchmark a small legal sweep, not a setting inferred solely from theoretical occupancy. Select the simplest near-best configuration rather than maximizing a proxy.

Likewise, use backend-specific mechanisms behind a stable operation: CPU candidate SIMD needs no GPU-style shared-memory/barrier machinery; CUDA can benefit from plane cooperation. A common API does not require identical execution strategies.

## 6. Use profiling to test explanations—not to replace timing

A useful investigation joins three kinds of evidence:

| Question | Evidence to collect | Interpretation trap |
| --- | --- | --- |
| Where does application time go? | Synchronized stage timings, full-fit timings, launch timeline | Enqueue time is not execution time; overlapping spans are not additive |
| Is the algorithm avoiding useful work? | Nodes, candidate positions, pruning fraction | Diagnostic counts may change meaning between implementations |
| Are loads redundant or inefficient? | Global requests, sectors/request, L1/L2 traffic, raw DRAM bytes | Cache hit rate alone says little about request volume or dependencies |
| Is execution waiting on memory? | Long-scoreboard stalls, occupancy, active lanes | A stall percentage is a fraction, not absolute stalled time |
| What did the compiler actually emit? | Optimized IR/PTX/assembly, register and local-memory evidence | Source vectors do not prove vector instructions; local traffic is not necessarily spills |
| Does the optimization pay for itself? | Packing-inclusive stage and end-to-end times, first-fit measurements, peak buffers | Kernel-only wins can be erased by preparation, JIT or memory costs |

Concrete examples:

- **Phase 3:** global-load requests fell about 6.4× at 16D, yet long-scoreboard stall share rose from about 61% to 67% as the kernel became much faster. The remaining loads represented a larger fraction of a smaller runtime. A rising stall share did not invalidate the gain.
- **Phase 7 at 50k:** baseline already had about 98% L2 hits. That did not make memory access cheap: F2 still reduced traffic and latency pressure substantially. An L2 hit does not remove L1/LSU request processing or dependent load chains.
- **Phase 6:** diagnostic LLVM IR contained wide vector loads and vector FMA, supporting candidate-axis vectorization. Textual IR counts included declarations and were not dynamic instruction counts or proof of the entire speedup.
- **Phase 7:** profiler output included an impossible 101.69% L2 hit rate for one arm. Preserve and flag anomalies; do not turn them into physical explanations.

Counter definitions matter. Phase 7's tile denominator was nominal broadcast opportunities, not active-lane evaluated pairs; baseline used scanned leaf positions. Compare raw totals alongside normalized metrics and document the denominator. Sampled-stall shares and active-warp stall percentages are also different quantities.

Profiler launches here were generally first warmup launches under replay, not measured-fit averages. Use matching binaries, inputs and geometries, but keep profiler timing separate from unprofiled acceptance timing. Preserve unavailable counters as unavailable, not zero; a missing spill count does not prove absence of spills.

## 7. Make semantic invariants an optimization tool

The key freedom was **visit-order independence with feature-order dependence**. Keeping the smallest `(distance², original ID)` keys allowed candidate/query reordering while preserving ties. Each individual distance still required the original sequential f32 feature accumulation.

**Apply:** write down what may move and what must remain fixed before redesigning the kernel:

- Vectorize across independent candidates or queries instead of reducing features in a new order.
- Use padded/vector loads without changing arithmetic order; test padding bits and contraction behavior.
- Carry original IDs through every pack for self-exclusion, tie-breaking and output placement.
- Test final consumers, not only local outputs. The whole-input permutation probe preserved mapped-back neighbor/core outputs but changed final clustering partitions.
- Compare bitwise against the original implementation on random data, and use a valid independent oracle where available. Ordered Rust arithmetic was not automatically bit-equivalent to LLVM-generated FMA; the Rayon comparator had many distance-bit mismatches and could not isolate runtime overhead.

Phase 6's first attempt failed before performance acceptance because malformed-tree requirements exposed invalid reference/bounds assumptions, including faults. The accepted retry followed an explicitly approved valid-tree contract. The lesson is **establish and verify legitimate preconditions**, not weaken tests until a fast kernel passes. Equivalent low-level callers must satisfy those preconditions; do not assume physical-buffer padding gives safe logical bounds behavior.

## 8. Make performance decisions reproducible and proportional

**Use controlled comparisons.** Alternate baseline/variant processes, reverse ordering in a second batch, avoid competing benchmark work, and retain input/source/binary identities. Analyze process-level observations; three fits in one process are not three independent experiments. Report ranges and paired effects, not only the best median.

**Separate tuning from confirmation.** Phase 6 width selection and Phase 7 plane selection were kept separate from acceptance batches. Otherwise the samples that chose a winner also overstate confidence in it.

**Measure preparation and deployment costs.** Packing was included in accepted stage timing. A useful break-even model is preparation cost versus savings per use multiplied by the number of uses; a repack can pay for a large fit but lose on a tiny query batch. Count additional live buffers, transient peaks and checked allocation sizes. Report first fit separately from warm operation: Phase 6's 20k×2 first fit regressed by about 63 ms despite strong higher-dimensional warm gains. Fresh processes with persistent caches intact do not establish cache-cold compilation cost.

**Use controls that could falsify the hypothesis.** Include strongly pruned low-dimensional cases, awkward tails, large k, ties, duplicates and every selectable vector width. Phase 3's small CUDA 2D regression did not replicate; the report explicitly retained that caveat. Phase 6's noisy 2D results were “no effect,” not proof of equivalence or improvement. Phase 7 had no acceptance guard cells, so its scope must remain narrower.

**Reject elegant changes that do not pay.** Phase 4's compact node records/interleaved bounds produced no qualifying primary gain and regressed one CPU cell. Phase 2's register top-k also failed acceptance. Fewer arrays, fewer loads in source, or a more elaborate kernel are not sufficient reasons to retain complexity.

**Re-profile after every major win.** At 50k×16, Phase 7 cut core time by 76.45% but full-fit time by only 17.92%; other stages now limited the benefit. Apply Amdahl's law to the current implementation, not the original bottleneck map. Final integration and cleaned-source verification establish correctness/tool compatibility; they are not additional controlled speedup measurements.

## Practical checklist

1. Locate the dominant synchronized operation and confirm actual routing/launches.
2. Count useful work, skipped work and repeated accesses; identify dependence chains.
3. Define semantic freedoms, numerical order, ownership and valid-input preconditions.
4. Cache small slow-changing state before attempting a complete state redesign.
5. Choose an independent SIMD/cooperative axis and pack data around it.
6. When pruning is weak, test regular tiling/scanning against traversal.
7. Isolate layout, scheduling and geometry changes; inspect emitted code.
8. Confirm with interleaved, packing-inclusive timing, full-operation results and scoped counters.
9. Account for first fit, memory peaks, guard workloads and all selectable hardware paths.
10. Retain only justified trade-offs, preserve negative evidence, and profile the new bottleneck.
