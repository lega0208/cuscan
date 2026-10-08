# Stage 2 research panel — discussion record

## Charter and evidence

Date: October 8, 2026. Chair: main agent. Members: architect (`3-PanelArchitect`) and critic (`4-PanelCritic`). This is preparation for a future implementation plan, **not that plan**, implementation approval, or a performance experiment.

**Scope: standard, non-parametric UMAP only. Parametric/neural implementations are excluded entirely, including follow-up topics.** Foreground: CUDA neighbor search, approximate transform queries and graph preparation. The validated Stage 1 16-batch atomic layout remains unchanged.

Both members independently reviewed the three requested findings documents and relevant source/book material. The chair independently inspected exact orchestration/kernels, ANN proposal generation, query admission/readback, graph statistics and the standard Flex path. Positions were challenged through chair/member exchanges and an architect-to-critic IRC exchange; the architect's final reconciliation supersedes its initial paper. This is a synthesis of the discussions, not a verbatim transcript.

Evidence labels below:
- **Source:** inspected repository/book behavior, not measured runtime cost.
- **Theory:** a derivation with stated assumptions, not a speedup prediction.
- **Existing measurement:** attributed to the [Stage 1 conclusion](artifacts/perf_optimization_stage1_conclusion_report.md), not remeasured here.
- **User report:** compute ~0.26%, DRAM ~0.23%, L1/L2 throughput <2.5%; capture provenance remains unknown.
- **Hypothesis:** an idea requiring future validation.

Seed references: [Burn release findings](burn_release_22_findings.md), [Burn book findings](references/burn_book_findings.md), [CubeCL book findings](references/cubecl_book_findings.md). Topic statuses and future discussion queue: [stage2_panel_topics.md](stage2_panel_topics.md).

## T1 — Neighbor-search mathematical parallelism and quality

### Description

What is the maximum mathematically feasible parallelism in exact fit/query, approximate fit and approximate query, and what must change in current implementations without degrading quality?

### Discussion summary and member positions

- **Architect:** current CUDA exact search exposes one lane per query row, at most 64 rows per operation; pair distances and partial top-k selection offer much more independent work. ANN's reverse discovery repeats a full candidate-array scan for each row. Approximate queries have independent outer rows but dependent inner traversal.
- **Critic:** counting independent arithmetic terms is not a hardware concurrency or speedup bound. Parallel selection needs a proof; ANN iterations require snapshots, whereas capped query traversal is order-sensitive. Recall floors must not excuse regression in a supposedly equivalent rewrite.
- **Chair:** separate independent tasks, algorithm work/span, bounded-memory scheduling and hardware residency. Initially preserve each pair's sequential float32 feature fold; make reassociation a separate numerical decision. Both members accepted these distinctions.
- **Resolved challenge:** ANN proposal arrival order need not be canonical under fixed finite distance keys and distinct-ID top-W selection. Query admission/pop/queue order does need preservation. Complete reverse adjacency is required, not a fixed-width reverse cap.

### Outcome: theoretical answer

Let Q be query rows, R reference rows, D features, k output neighbors, N fit rows, W ANN retained width, and d_i the number of source rows whose candidate list contains row i.

**There is no single algorithm- and hardware-independent maximum parallelism.** Work/span describes an algorithm's average available parallelism; instantaneous independent tasks describe a particular stage. Neither is a resident-thread count. An inefficient algorithm can create extra parallel work without improving time-to-solution.

| Path | Independent work available | Dependencies that remain |
|---|---|---|
| Exact rectangular search | Q×R pair distances, excluding aligned self pairs where applicable; Q×R×D independent squared-difference terms if feature summation may be reassociated | Each pair's reduction; selecting/merging k neighbors for each query |
| Exact fit | N(N−1) directed non-self distances; symmetry permits N(N−1)/2 unique unordered distances in mathematical arithmetic | Both endpoints need their own top-k; symmetry adds routing/ownership work and needs numerical validation |
| ANN expansion | Rows and known proposals inside one snapshot iteration; proposal distances can run independently before selection | Next iteration consumes the previous retained lists; reverse-index construction precedes its use |
| Approximate query | Q independent query traversals; distances to canonically admitted neighbors of a selected vertex can be evaluated concurrently | Seed generation, next-pop choice, seen-set admission, visit cutoff and queue updates define the traversal |

**Exact work/span, an achievable construction rather than an optimality claim:** the current scalar scan has worst-case work O(QR(D+k)) and per-row span O(R(D+k)), including insertion. Split each reference set into P disjoint partitions. Compute each pair with the unchanged serial feature fold; each partition retains local top-k; merge sorted lists in a binary tree. With serial list merges, per-row span is O(ceil(R/P)(D+k) + k log P), and work is O(R(D+k) + Pk). This assumes sufficient processors and identical finite keys; it does not account for real memory/launch costs. At P=R, pair evaluation offers QR tasks, but storing every singleton simultaneously defeats bounded memory. Stream partitions/waves instead.

Allowing feature-tree reductions changes the distance span from O(D) toward O(log D), but changes float32 reassociation. Parallel sorting/selection can also shorten merge span; there is no claim that the construction above is the theoretical optimum. The relevant upper opportunity is pair/feature independence, constrained by the chosen arithmetic and selection contract.

At N=5,000 and D=32, directed non-self fit exposes 24,995,000 pair tasks and 799,840,000 feature terms. These are **terms, not floating-point operation counts or simultaneous GPU threads**. The current exact wrapper dispatches at most 64 query lanes per launch (two 32-lane cubes for a full block). Scheduling/overlap and occupancy must be profiled.

**ANN derivation:** a full valid W-wide snapshot proposes, per row, W current candidates, W² forward neighbor-of-neighbor occurrences, and d_i×W reverse-list occurrences. Thus P_i ≤ W + W² + d_iW before invalid/self/duplicate rejection. Since sum(d_i)=NW for unique valid outgoing lists, aggregate proposal occurrences are NW + 2NW². Distinct eligible proposals per row cannot exceed N−1. Hubs can have d_i up to N−1; averages do not establish balanced work.

The current reverse membership discovery alone costs O(N²W) comparisons per expanding iteration. Complete reverse adjacency contains O(NW) entries; count/prefix/scatter removes that discovery rescan but **does not remove** reverse-list expansion, proposal deduplication, distance evaluation or selection. Iterations remain dependent. No numeric exact/ANN crossover is established.

**Approximate query:** there is no general QR-distance parallelism while preserving the existing best-first search. If H vertices are popped, their selection/commit sequence forms a dependency chain. Feature work and admitted edge distances within one expansion are opportunities; bulk frontier expansion is a different algorithm unless equivalence is proven.

### Quality-preservation obligations

1. Preserve global-index aligned self exclusion, query offsets, distinct IDs, squared-distance/index tie ordering before final sqrt, seeded starts, sentinels and error semantics.
2. Local top-k merging is lossless: any candidate excluded from a partition's top-k has at least k better keys in that partition, so cannot enter global top-k. Deduplicate IDs for overlapping proposal lists before applying this argument.
3. ANN top-W is set-based under fixed keys: after any proposal prefix, the retained list is its W smallest distinct eligible keys. The worst retained key never increases; an evicted candidate cannot re-enter with the same key. This permits proposal reordering and unordered complete reverse adjacency. Preserve input snapshots and separate next outputs; no in-place cross-row updates.
4. Keys must remain identically computed, finite and representable under existing sentinel behavior. Finite input alone does not prevent squared-distance overflow. Sequential source-level summation is an initial safeguard, not proof against compiler FMA/fast-math changes; compare generated behavior and outputs.
5. For query, admit unseen edge IDs in existing edge-slot order, deduplicate by first occurrence, truncate to remaining visit budget, and commit queue operations in original order. Parallel distances must not change the seen set or next pop.
6. A semantics-preserving candidate should reproduce its own backend's neighbor IDs/distances and diagnostics, not merely clear 90% recall. A reassociated or algorithm-changing candidate needs an explicitly agreed numerical contract plus downstream gates.

### Open questions and evidence

- Is strict current-backend distance/ID equivalence mandatory for all Stage 2 candidates, or only the first equivalence-preserving track? What contract admits near-tie changes?
- How skewed are reverse degrees and how many proposals are duplicates? Is distance or list maintenance dominant after removing discovery scans?
- How much pair/partition parallelism fits the scratch/register budget for target D and k?
- Can within-expansion query distance batching preserve diagnostics and statistics as well as selected neighbors?

Sources: [exact orchestration](src/standard_umap/neighbors.rs), [exact kernels](src/kernels/standard_umap/mod.rs), [ANN consider/step](src/kernels/standard_umap/ann.rs), [query kernel](src/kernels/standard_umap/query.rs). Numerical admission stays a future discussion (F1), not a silent relaxation.

## T2 — Cooperative distance/selection and useful cache locality

### Description

How can CubeCL parallelism, coalescing, shared-memory staging, vectorization and Burn/CubeK facilities make useful data reuse possible without oversized scratch or changed neighbor quality?

### Discussion summary and member positions

- **Architect:** reuse reference tiles across queries, map feature loads to adjacent lanes, retain top-k thresholds locally, and distinguish hardware locality from compilation/autotuning caches.
- **Critic:** cache throughput is not cache effectiveness; cooperative reductions, GEMM distance identities and reduced precision change keys. Extra staging/barriers/selection state can cause spills and outweigh reuse.
- **Chair:** accept tiling and loading ideas as hypotheses; preserve serial component accumulation first. Reject claims that nominal L2 fit proves residency or that larger launch dimensions necessarily improve runtime.

### Outcome

Candidate research directions, not selected implementations:
- Larger bounded query batches and/or reference partitioning to expose more useful pair work; the 64-row cap is not itself a CUDA memory invariant.
- A query tile Bq×D and reference tile Br×D staged for reuse; adjacent feature loading can be coalesced even if each pair's accumulation remains serial. Compare staging cost against existing reference broadcasts/cache behavior.
- Local top-k plus deterministic merging, with explicit owner per output and partial list. Avoid concurrent ordinary writes to the same list.
- Vectorized **loads** with serial component accumulation are distinct from vectorized/plane **reductions**. Test alignment, odd feature counts, tails and noncontiguous conversion costs.
- CubeK tile machinery is a discovery lead, not a demonstrated ready-made neighbor/top-k operator or automatic optimization of our handwritten kernels. Verify locked-version APIs and suitability before a future design relies on it.

Memory accounting must include inputs/outputs, scratch, partial selections and allocator peak. Exact CUDA output is O(Bqk); cooperative staging can require O((Bq+Br)D), optional O(BqBr) pair tiles, and O(BqPk) partition lists. Stream/reduce waves to bound partial lists. Complete ANN reverse entries remain O(NW), even with high-degree rows. Approximate query scratch scales with active queries times candidate/visit capacity. A full per-query N-sized seen bitmap adds O(QN) state and is not a default bounded-search choice.

A fit input of 5,000×32 float32 values is 640,000 bytes; that size alone says nothing conclusive about L2 residency, working set or bandwidth limits. Account for ANN/query scratch, replacement traffic and other kernels. Measure sectors/request, cache hit/miss behavior, repeated loads, spills and elapsed time.

Separate three mechanisms:
1. Hardware L1/L2 and shared-memory data reuse: affects memory access during kernels.
2. CubeCL Environment compiled-kernel/autotune caches: principally startup/repeated tuning behavior, not elimination of distance work.
3. Allocator pools: reuse allocations; reserved memory is not live memory or whole-process VRAM.

GEMM `||q||² + ||r||² − 2q·r`, TF32 and feature-tree reductions remain conditional numerical research. Cancellation can flip near ties or produce negative near-zero squared distances before sqrt. Clamping alone does not establish correctness. Tensor-core availability does not establish suitability or speedup.

### Open questions and evidence

- Which Bq/Br/partition shapes fit shared memory and register budgets without spilling, across small/odd D and larger k?
- Do irregular ANN proposals benefit from grouping/pair dedup more than rectangular tiling, including grouping overhead?
- Does local selection state or load latency dominate once work is reduced?
- Which CubeCL plane/shared-memory primitives and CubeK components are actually available through the locked Burn backend?

Sources: [CubeCL topology/reduction](references/cubecl-book/getting-started/parallel_reduction.md), [vectorization](references/cubecl-book/core-features/vectorization.md), [CubeCL findings](references/cubecl_book_findings.md), exact/ANN/query sources from T1. No tile size, cache hit target or gain percentage was selected.

## T3 — Redundant work and Burn/CubeCL execution boundaries

### Description

Which repeated work can be shared or removed, and which stages must remain ordered despite fusion, asynchronous execution, pooling or graph replay?

### Discussion summary and member positions

- **Architect:** reverse indexing, duplicate-distance rejection, query validation and host readback are concrete structural opportunities. Burn extension metadata does not optimize custom-kernel internals.
- **Critic:** preserve the complete CUDA proposal set, diagnostics and mutation ownership. A graph rescan may be a scaling problem without being today's largest measured cost. More stages can be beneficial when they remove much more work.
- **Chair:** prioritize evidenced search dominance, not a guessed ranking of every loop. Global statistics can first be computed once using the identical serial fold, isolating work removal from numerical reassociation.

### Outcome

| Source observation | Research direction | Contract/cost question |
|---|---|---|
| ANN discovers reverse sources by scanning N×W entries for every row | Count/prefix/scatter complete reverse adjacency per snapshot | No reverse-degree cap or dropped sources; expand each source's full CUDA candidate list; measure construction and hub imbalance |
| `consider` computes distance before checking retained-ID duplication | Reject retained duplicates earlier; investigate valid distance reuse | Rejecting unchanged duplicate IDs is equivalent under fixed keys; benefit depends on rejection rate and extra scans |
| `smooth_row` recomputes identical global fold over N×k for each row | Compute global statistic once, row sums locally | Same fold/order can isolate work reduction; preserve rho==0 global-mean fallback and sigma floors; tree reduction is separate |
| Query validates entire fitted graph per query | Validate once per operation with shared invalid flag | Preserve invalid-index range, `stats == -1` and `UmapError::Numerical`; persistence/mutation ownership must be explicit |
| Approximate transform reads stats for every block | Investigate batching diagnostic reads | Do not lose errors, move work outside timed scope, or silently alter externally required error timing |
| Exact host wrapper slices/launches/cats ≤64-row blocks | Backend-appropriate bounded orchestration | CUDA emits B×k; Flex materializes B×R; measure launches, contiguity and concatenation costs together |

Custom `StandardUmapOps` operations are opaque to ordinary Fusion optimization. Lazy output metadata is not a request to fuse their internals. Explicit candidates are not automatically tuned just because Burn autotuning is enabled. Preserve backend registration and exact metadata.

Separate ordered launches can be the correct implementation of global count/prefix/scatter or partial-selection dependencies. A workgroup barrier is not a grid-wide barrier. Preserve ANN ping-pong snapshots, scratch/output ownership, atomic scatter requirements and unaliased mutable buffers. Fewer launches is not a universal objective.

Graph replay and workspace reuse are conditional, lower-priority questions until attribution justifies them. Replay requires stable captured storage, stream/lifetime ownership, correct mutable parameters and resetting warmup side effects; it cannot remove reverse rescans. Pool usage/reporting should precede claims that allocation requests are physical allocation overhead. Layout copy removal is not reopened: Stage 1 measured only ~0.067 ms in its diagnostic captures.

### Open questions and evidence

- Is graph validation owned per query operation, per immutable fitted graph lifetime, or persistence boundary? Which mutation/lifetime proof supports amortization?
- How much query time is spent in block readbacks versus device traversal?
- Can retained distances be reused without changing distance precision or candidate ownership?
- Are repeated graph count/scatter edge evaluations worth storing, given additional traffic and scratch? This is a secondary hypothesis, not a demonstrated hotspot.

Sources: ANN/query/graph kernels; [transform orchestration, lines 429–465](src/standard_umap/estimator.rs#L429-L465); [custom operation chapter, lines 25–44](references/burn-book/advanced/backend-extension/custom-cubecl-kernel.md#L25-L44); [asynchronous execution](references/burn-book/performance/good-practices/asynchronous-execution.md); [release replay/memory discussion](burn_release_22_findings.md).

## T4 — Measurement interpretation, goals and readiness

### Description

What do the reported low utilization/throughput percentages mean, and what evidence/goals should constrain a future Stage 2 plan?

### Discussion summary and member positions

- **Architect:** launch geometry, serial loops, readbacks and stalls are plausible explanations; obtain per-stage timings and Nsight attribution.
- **Critic:** promotion requires matched end-to-end gains and quality, not occupancy or cache percentages. Distinguish profiler/tuning traffic from UMAP work.
- **Chair:** withdrew interim statements equating low throughput with proven latency bounds, L2 residency, fixed occupancy ceilings, predicted vectorization gains or numeric crossover. Both members accepted the evidence boundary; these statements are not panel conclusions.

### Outcome

The user's ~0.26% compute / ~0.23% DRAM / <2.5% cache throughput are **user-reported diagnostics with unknown kernel/range and denominator**, not proof of a specific bottleneck. Possible explanations include underfilled grids, dependent work, low lane efficiency, stalls, host gaps or the sampled range. Cache throughput is not hit rate; removing redundant traffic may lower it while improving performance. The goal is faster useful work at maintained quality, not maximizing every percentage.

Existing Stage 1 evidence supports foregrounding search: diagnostic totals over two fits were ~1.31 s exact search, ~2.80 s approximate steps and ~0.31 s layout per mode. These are instrumented stage totals, not warm product timings. Its reported 3.58% occupancy sample was a **layout batch**; do not assign it to neighbor kernels. Probe kernels are autotuning, not UMAP algorithm work.

Preparation goals:
1. **Performance:** synchronized end-to-end fit **and transform**, with exact/approximate reported separately; paired repeated uninstrumented runs with distributions. Preserve the existing paired both-fit-mode criterion for general promotions; any deliberately mode-specific rule needs explicit agreement, not a hidden gate change.
2. **Quality:** preserve hard fit/frozen-transform trustworthiness gaps ≤0.02, approximate recall above the existing 90% gate, immutable transform references, input buffers, seed/self/duplicate/tie/error contracts. Approximate original-baseline tracking remains a warning, not a new hard failure. Equivalent rewrites should show no neighbor/recall change against their own backend, rather than spending recall headroom.
3. **Memory/scaling:** state a bounded scratch/peak budget, measure live/reserved/process VRAM separately, and test mixed-size as well as fixed-size workloads.
4. **Attribution:** synchronized product time plus device-stage windows; close lazy windows appropriately with flush-enabled profiling or reads and account for pending work. Flush is not synchronization. Collect launch geometry, active warps/SMs, lane activity, registers/spills, stall reasons, memory sectors and cache hit/miss behavior; use Systems for launch gaps and host readbacks.
5. **Reproducibility:** features/lockfile, compiler/runtime/GPU, N/D/k/Q, seeds, initialization, instrumentation and cache identity. Separate cold process/cache, persistent-cache warm process and warmed in-process runs; exclude startup throughput probes from algorithm attribution, not from honestly reported cold-start cost.

No new numeric speed target is selected. Stage 1's 5,000×32, 20-epoch warmed fit medians (0.828 s exact / 1.551 s approximate) remain inherited controls, not fresh measurements or universal deployment targets.

### Open questions / planning inputs

- Which exact kernels/ranges produced the user's percentages, with which Nsight metric names, replay/capture settings and device?
- What N/D/k/Q envelope matters, and what GPU-vs-Python comparison is the actual product goal? Small-shape control alone cannot establish crossover.
- What strict neighbor numerical contract and peak scratch budget are required?
- What paired timing noise and mode-specific acceptance rules are appropriate for future experiments?
- Which existing fixtures cover adversarial near-ties, overflow, skewed reverse degrees and query visit truncation? Later verification must exercise actual CUDA, not silently skip hardware.

Outcome: sufficiently grounded to define future research questions and candidate families; **not enough evidence to choose a winning kernel, launch configuration, numeric policy or implementation plan**. F1–F3 retain the unresolved decision topics.

## T5 — Standard-UMAP backend and search-path scope

### Description

This topic arose when the panel discovered different standard Flex/CUDA reverse-proposal semantics. Prevent a semantic mismatch from being mistaken for an optimization regression or a portable reference.

### Discussion summary and member positions

- **Architect:** shared exact orchestration imposes a small block choice on different backend memory behavior. CUDA ANN and generic/Flex ANN do not generate identical reverse proposals.
- **Critic:** standard Flex can validate shared self/tie/dedup invariants, but its ANN expansion is not an oracle for identical CUDA proposal sets. GPU launch/race checks need actual CUDA execution.
- **Chair:** foreground standard CUDA fit/query/graph; document standard backend contracts without expanding into algorithm unification. The user's explicit exclusion removes all parametric/neural work from the agenda.

### Outcome

Source: generic standard ANN builds per-slot scatter-Assign reverse proposals consisting of source IDs ([standard ANN, lines 120–154](src/standard_umap/ann.rs#L120-L154)); CUDA finds sources containing the target and proposes each source's **whole candidate list** ([CUDA ANN, lines 177–200](src/kernels/standard_umap/ann.rs#L177-L200)). Assignment collisions and differing proposals prevent assumed expanded-ANN equality. A complete CUDA reverse index must reproduce CUDA's candidate universe, not replace it with Flex's.

Source: standard Flex exact search materializes a bounded query×reference squared-distance block ([Flex, lines 147–155](src/flex_backend.rs#L147-L155)); the CUDA exact kernel retains query×k outputs without that matrix. Preserve memory bounds, not necessarily a shared 64-row scheduling choice.

No backend unification, WGPU enablement, algorithm-selection cutoff change or additional implementation is selected. Foreground scope is standard CUDA exact/approximate fit and transform plus graph preparation; standard Flex remains relevant for shared contracts and bounded orchestration. All cross-backend equivalence claims must identify the precise operation/contract. Keep the validated layout frozen.

### Open questions and evidence

- Is current CUDA reverse proposal semantics the intended standard-CUDA contract? Until decided otherwise, preserve it.
- How should future bounded orchestration express each standard backend's actual temporary memory costs?
- Which shared tests establish invariant parity, and which tests must be backend-specific?

Sources: standard generic ANN, CUDA ANN, Flex exact operation and backend dispatch. Broader standard-backend design is queued as F4 only if a future plan crosses that boundary; excluded implementations are not follow-up work.

## Round 1 — Chair's closing synthesis

The strongest directions are **more independent pair/proposal work, complete reverse indexing, removal of repeated global/validation work, and bounded cooperative memory reuse**. None is a measured speedup or committed design. ANN proposal selection can be reordered under a fixed-key proof; approximate query traversal cannot simply become a parallel frontier. A lower cache-throughput percentage after eliminating waste is not failure.

No blocking disagreement remains on these principles. Empirical priorities, strict numerical admission, workload envelope, validation ownership and backend-specific boundaries remain deliberately unresolved and visible in the topic register. This panel round is concluded; those questions belong to future discussions before relevant implementation choices.

Verification boundary: only the two requested panel documents were changed. No product source, dependency, test or benchmark configuration was changed; no CUDA/benchmark/quality tests were run or claimed. Document links/structure and repository diff are checked separately by the chair.

## Continuing panel — queue policy and corrections

The user continued the panel with F1–F4 and authorized dynamic additions. Architect and critic resumed independent research, exchanged findings via IRC, and received chair challenges. F5/F6 were added from unresolved work-balance and capability questions; their exchange added F7, a concrete host-boundary inventory. Baseline-comparison mechanics stay inside F1; index feasibility stays inside F3; query-statistics semantics stay inside F4. Missing captures and deployment requirements are dependencies, not endlessly recycled discussion topics.

Corrections to carry forward:
- No standard `FittedUMAP` persistence API was found. Round 1's deserialized-graph premise was unsupported; the serialization edge tests encountered are excluded from this panel. F4 addresses actual private fitted state and public primitive callers, not a hypothetical load feature.
- Existing float-vector equality tests are numerical `assert_eq!`, not a stored bit-pattern baseline. Identical source folds do not prove identical generated arithmetic.
- Query seeds can retain overflowed distances and final selection can emit valid IDs with infinite distances. The claim that overflow universally yields `-1`/maximum-float outputs was withdrawn. Exact/ANN sentinel insertion behaves differently. Finite input alone is insufficient for finite squared-distance keys.
- Exact-rescoring only a numerically generated shortlist cannot recover a true neighbor already pruned from that shortlist. Precision admission needs completeness evidence or must accept a changed search contract explicitly.
- Positive query statistics remain documented output semantics even when the estimator consumes only the invalid sentinel. Readback is a synchronization boundary, not a demonstrated dominant cost.

## F1 — Numerical equivalence and algorithm-change admission

### Description

Separate scheduling/work-removal equivalence from reassociated numerics and algorithm changes, and determine what evidence can actually verify each claim without silently lowering quality.

### Discussion summary and member positions

- **Architect:** independent test review distinguishes same-backend repeatability from the `1e-5` host-distance fixture tolerance; FMA/code generation can change across launch variants. Proposed explicit admission tiers and cancellation stress cases.
- **Critic:** no existing old-versus-new neighbor/statistics baseline exists; run-to-run equality within one binary cannot establish cross-change equivalence. Tie-aware recall tolerates ID substitutions and cannot arbitrate deterministic neighbor parity. Validation error timing belongs in the contract.
- **Chair:** adopt distinct research classes, not a new user-approved precision policy. Reject final-shortlist rescoring as sufficient proof of exactness, any estimated cancellation error as a universal bound, and treating the original-baseline tracking warning as a new hard gate. Extreme-value behavior is preserve-and-characterize, not authorization to add rejection or clamping.

### Outcome

Recommended research classification:

| Class | Examples | Required evidence before an equivalence/admission claim |
|---|---|---|
| E: intended equivalence | Block/partition geometry, complete reverse indexing, retained-ID duplicate rejection, intra-call validation sharing | Differential current-backend IDs, distance bit patterns, statistics, graph outputs and diagnostics on identical inputs/seeds/compiler/device; repeated-run determinism and downstream inherited gates |
| N: changed arithmetic | Feature-tree sums, vector partial sums, GEMM identity, reduced precision/fast math | Explicit supported domain, distance/tie/error policy, old/new array comparisons, cancellation/near-tie/overflow tests, graph and frozen-transform checks; no blanket tolerance inferred from the existing fixture |
| A: changed search/API behavior | Candidate caps/sampling, iteration or traversal changes, externally changed validation boundary | Explicit candidate/traversal/budget/error changes, per-run recall and quality evidence, determinism/statistics behavior and matched product measurements; no equivalence label |

These are panel recommendations, **not approval to implement N/A**. E is an intent requiring differential proof, not a promise that keeping the same loop is sufficient. For finite representable squared-distance keys the top-k/set proofs from T1 apply; NaNs, overflow, signed zeros, sentinel exhaustion and compiler contraction require their own coverage. Do not silently change extreme-input outputs or introduce a new error contract in a performance rewrite.

Verification comparator direction: obtain old/new arrays or stable bit-pattern digests with full arrays available for diagnosis, tied to backend, locked dependency/toolchain/build, dataset hash, shape and seed. Existing tests compare repeated runs of one implementation; the fit-only profiler exports IDs and embeddings, not distance/statistics baselines. Define the comparator before claiming an E candidate preserved them. No golden infrastructure was built in this discussion.

Test/research matrix should include equal-distance lower-ID ties on actual CUDA, duplicate feature rows versus duplicate neighbor IDs, block offsets/self exclusion, odd feature tails, skewed proposal duplicates, query mid-expansion visit truncation, invalid graph markers and large finite values. Same-query results across block partitioning must keep global offsets and seeds.

Existing acceptance controls stay distinct: host-distance absolute tolerance `1e-5` is a fixture check; benchmark tie-aware recall counts unique returned IDs within the exact k-th radius plus `1e-6` relative / `1e-12` slack. Neither permits arbitrary ID drift. Recall is ≥0.90 **per run**, trustworthiness is within 0.02 per CUDA fit versus the Python mean, and frozen-transform gaps remain ≤0.02. Keep the existing tracking warning a warning; report numeric drift rather than hide it behind aggregate passes.

Exact final rescoring is useful only for candidates actually retained. To claim exact nearest neighbors after approximate arithmetic pruning, establish shortlist completeness, an interval/error-bound scheme that includes ambiguous candidates, or an exact verification pass. Otherwise it is N/A, not exact E. No shortlist size or precision tolerance was chosen.

### Open questions and evidence

- The user has not admitted changed arithmetic/search semantics or a new numeric tolerance. An E-first research stance is the panel recommendation; N/A remain conditional.
- Full differential baseline fixtures/digests and extreme-input characterization are missing evidence, not implemented tests.
- The end-to-end frozen layout has accepted nondeterminism; distinguish deterministic neighbor/graph outputs from embedding quality gates rather than demanding byte-identical final embeddings.

Sources: [ANN tests](tests/test_standard_ann.rs), [standard quality tests](tests/test_standard_quality.rs), [recall metric](scripts/compare_umap_spans.py#L181-L191), [query seeding and final selection](src/kernels/standard_umap/query.rs), [exact and ANN sentinel comparisons](src/kernels/standard_umap/mod.rs), [ANN consider](src/kernels/standard_umap/ann.rs).

## F2 — Capture provenance and transform attribution

### Description

Determine what existing captures can establish, define missing provenance, and avoid replacing useful research with a guessed interpretation of the user's percentages.

### Discussion summary and member positions

- **Architect:** the actual harness default selects a layout kernel, while the example records host spans and fits only. If that exact invocation produced the user's capture, the percentages would describe layout; that conditional is not provenance.
- **Critic:** search-kernel NCU evidence and transform attribution are missing; startup throughput probes must not be mistaken for algorithm work or excluded dishonestly from cold-start cost.
- **Chair:** preserve the conditional only. Do not call layout the leading diagnosis without the user's capture command/range; low metrics do not establish a bottleneck. Separate instrumentation evidence from uninstrumented acceptance timings.

### Outcome

Source-confirmed capture limitations:
- `compare_umap_spans.py` defaults `--ncu-kernel` to `.*layout_vertex.*kernel.*`, `--set basic`, skip 16 matching launches and capture one sample; it runs the approximate fit mode for this NCU pass.
- Its Systems pass runs one warmup plus one timed fit per mode, explaining two-fit aggregate device totals. These are not five-run warmed product medians.
- `profile_standard_cuda.rs --trace` describes host spans, not GPU durations, and the executable runs fits only. Existing artifacts do not attribute approximate transform traversal or transform graph costs.

Required provenance record for future captures: exact command/kernel filter and matching launch ordinal; kernel name/grid/block/shape/duration; device/runtime/compiler/features; metric names and denominators; NCU replay mode and counter set; clock/power state where controlled; warm/cold cache identity; algorithm mode/seed/budget; and whether startup compilation/autotune probes were included. A percent copied from a summary is not sufficient.

Future attribution questions: exact search launch waves and dependency stalls; ANN reverse discovery versus proposal/distance/selection work; approximate-query validation, seen scans, queue operations and per-block host reads; graph preparation and host boundary costs. Use Systems for host gaps/copy/synchronization intervals and Compute for representative algorithm kernels. Changing a profiler regex alone is insufficient: the layout-specific skip count can miss or misidentify search samples.

Build/compilation, input generation, diagnostic graph extraction and quality calculations must remain separately identified. Record cold product startup honestly; exclude tuning probes from **algorithm stage attribution**, not from any end-to-end cold-start number that includes them. Read/flush-close lazy profiling windows correctly; flush dispatches pending work and is not host synchronization.

### Open questions and evidence

- User capture files/command and exact metric identities remain unavailable. The panel cannot reconstruct them from reported percentages.
- Search NCU and separate fit-once/transform-repeated captures remain future measurement dependencies. No transform cost or utilization improvement is inferred here.
- A future capture must verify selected launch ordinals after any launch-geometry change rather than inherit the layout skip rule.

Sources: [capture options](scripts/compare_umap_spans.py#L275-L296), [default filter](scripts/compare_umap_spans.py#L320-L337), [profile orchestration](scripts/compare_umap_spans.py#L599-L633), [fit-only example](examples/profile_standard_cuda.rs), [Stage 1 capture record](artifacts/stage1-restored-control-20.json).

## F3 — Workload envelope, memory budget and crossover goals

### Description

Recommend a bounded research ladder and byte-accounting method without inventing a deployment requirement, empirical crossover, GPU scratch ceiling or full Cartesian benchmark campaign.

### Discussion summary and member positions

- **Architect:** retain the acceptance controls and vary one shape axis; calculate query/ANN/exact/CSR costs separately. Larger launch batches may consume much less memory than a full distance matrix.
- **Critic:** large resource tests are not timing/quality evidence. Keep per-run acceptance gates, matched baseline controls and deadline/incomplete-run rules; measure pool reservations separately from live operands.
- **Chair:** recommend a sparse ladder, conditionally extend it within an explicit resource/time budget, and reject both blind all-pairs truth at very large N and unsupported exact/ANN crossover forecasts. CSR entries alone are not its total working storage; proposal occurrences are not O(NW).

### Outcome

Suggested research ladder, **not a user-approved target or executed benchmark**:
- Preserve existing controls: 512×16/30 epochs, 1,024×32/100 epochs, 5,000×32/20 epochs, k=15, both explicitly forced search modes. Keep existing Random timing and separate Spectral frozen-quality controls identifiable.
- Row scaling candidates: 20k and 100k, with existing 20k-Flex/60k-CUDA resource fixtures as resource-only context. A 1M rung is conditional on checked memory/runtime feasibility, not a default exact-search run.
- Separate feature/k perturbations: D=31/33 for vector tails, D=128 for longer folds, k=5/15/50; add very high D only for an actual workload question.
- Separate transform Q=1/63/64/65/5,000 to expose launch/block boundaries and fit-once/repeated-transform behavior. Sweep approximate queue control separately from defaults, keeping valid caps and Exact's required default 4.0.
- Include controlled clustered, duplicate/tied and hub/skew cases. An external dataset is optional only when it addresses a stated distribution question; no download or full cross-product sweep is implied.

Full truth and downstream fixture comparisons remain required on feasible controls. At larger N, sampled rows with **streamed exact reference search** can assess ANN/query recall without an N×N distance allocation; report sampled coverage/uncertainty and do not call it full-graph parity. Sampling for diagnostics must not become sampling of the candidate's search itself. Trustworthiness computation can have quadratic work/storage too: budget the evaluator independently and do not silently substitute a different quality statistic.

Logical f32/i32 bytes, excluding allocator padding, pool reservation, input clones/conversions and transient backend/compiler resources:

| Object | Bytes / qualifications |
|---|---|
| Fit input | 4ND |
| Exact ID+distance output | 8Qk; a CUDA B-row operation emits 8Bk, whereas Flex distance tile costs 4BR |
| Proposed current/next ANN ID+distance pairs | 16NW logical workspace; actual old-buffer/deferred lifetimes and initial unused-distance allocation need separate accounting |
| Reverse adjacency source IDs + offsets | 4NW + 4(N+1) if 32-bit offsets are sufficient; counts/cursors/scan temporary storage and overflow-safe wider offsets add cost |
| Query scratch + outputs + stats for B active rows | B(12C + 8V + 8k + 8), from three candidate arrays, two seen arrays, ID/distance outputs and two i32 stats |
| Query input block staging | Additional 4BD; full transform input/output/retained block handles also count |
| Graph head/tail/weight COO arrays | At capacity 2Nk, 24Nk bytes for those three arrays only; membership/rho/sigma/clock/CSR/RNG storage is additional |

At the 5k control, transform k=15, C=60 and V=780, B=64 gives exactly **453,632 bytes** for the listed query arrays (~443 KiB), plus 8,192 input-block bytes at D=32. For Q=5,000 active queries the same arrays would be 35,440,000 bytes (~33.8 MiB). These are logical allocation calculations, not measured live VRAM. Fit k is 14 nonself at the control, not transform's 15; use actual widths when budgeting. ANN W=32 gives 2,560,000 bytes for paired current/next ID+distance buffers.

All prefix/entry products and byte multiplications must be checked separately from row-index bounds; N≤i32::MAX does not make NW or 2Nk fit in an i32 prefix. Cube launch counts, candidate/visit caps and graph COO indices impose additional representability limits. Bounded streaming, not a blanket prohibition on every superlinear optional tile, is the criterion.

Keep matched pre-change binaries/data hashes/seeds/configurations, paired repetitions, honest distributions and existing at-least-five-run controls. Avoid favorable reruns or post-hoc noise thresholds. Python's unseeded parallel comparison is not a strict deterministic parity oracle. A general promotion retains the inherited both-mode comparison; any future deliberately mode-specific acceptance needs explicit agreement. Missed GPU/Python targets and incomplete timed-out runs remain misses/incomplete.

Current Auto cutoff is implementation-relative; source records historical calibration around its 4,096-row boundary. No new crossover or cutoff was established. After a future admitted search change, remeasure rather than inherit or extrapolate it; recalibration is not selected implementation work here.

### Open questions and evidence

- User deployment N/D/k/Q distribution, latency goal and usable memory/runtime budget remain unspecified. The ladder is a recommendation, not that specification.
- No universal byte ceiling or degree/duplicate-rate distribution was measured. Allocation and pool accounting must be checked under mixed shapes and pending work.
- Large-N exact truth and quality evaluation need a feasibility statement; sampled evidence cannot prove bitwise equality of all rows.

Sources: [query budget formula](src/standard_umap/config.rs#L399-L426), [query allocations](src/kernels/standard_umap/query.rs#L292-L369), [ANN allocations](src/kernels/standard_umap/ann.rs#L247-L361), [standard quality/resource fixtures](tests/test_standard_quality.rs), [acceptance harness](scripts/compare_umap_spans.py), [recorded control](artifacts/stage1-restored-control-20.json).

## F4 — Standard backend boundaries and validation ownership

### Description

Resolve actual validation/statistics obligations and standard-backend semantic boundaries without designing nonexistent persistence or a broad new API.

### Discussion summary and member positions

- **Architect:** standard fitted graph is private, transform takes `&self`, and fit already checks a fuzzy-graph marker before returning a model. Lifetime validity is plausible for estimator-owned graphs; direct backend callers are a different boundary.
- **Critic:** direct `approximate_query` documents invalid `[-1,-1]` stats and must retain that contract. With source verification, withdrew the claim that Exact fits never validate. No standard persistence premise remains.
- **Chair:** recommend sharing validation **inside the same primitive operation first**; distinguish that per-block call from a whole transform. Preserve positive stats magnitudes and externally observable error boundary. Lifetime amortization remains a conditional architectural question, not automatic permission to remove primitive validation.

### Outcome

Preserve CUDA ANN's full source-list reverse proposals as the current optimization baseline. Flex's source-only scatter proposals remain a different implementation; use shared invariants/host distances, not presumed expanded-ANN array equality. No algorithm unification or new standard backend is selected.

Fit already validates invalid neighbor indices through `fuzzy_graph` and its `-2` marker, in both search modes; invalid fit graphs fail before returning `FittedUMAP`. The fitted graph is private and no standard save/load or injection API was found. Prior references to deserialized standard models are removed from the active agenda, not kept as a speculative future feature.

The public backend `approximate_query` still accepts caller-supplied graph primitives. It returns `[peak_candidates, unique_evaluations]` for every query and `[-1,-1]` for invalid graph indices. This is a documented semantic output, not disposable telemetry merely because the estimator checks only `-1`. Preserve magnitude/capacity/truncation behavior as well as IDs and distances for E; change/redefine it only under an explicit changed contract.

Validation sharing has distinct levels:
1. **Per primitive call:** check the immutable supplied graph once, propagate the flag to all query rows, preserve current output/stat/error behavior. This is the panel's conservative research direction.
2. **Per estimator transform:** the host currently invokes the primitive per ≤64-query block; level 1 alone still validates per block. Whole-transform sharing must express trusted state/ownership without silently weakening the public primitive's caller contract.
3. **Per fitted lifetime:** existing fit validation and private immutable state make this worth researching, but any trusted path must establish provenance, cloning/alias safety and error boundary. It is not a rationale for accepting arbitrary invalid graphs in the primitive.

No validate-op signature, trust token, cached flag, alternate backend path or persistence API was designed/approved. No extra fit-time validation was selected. Externally moving an error from transform to fit (or changing direct primitive behavior) is behavior-changing; eliminating repeated scans **within the same call** can be E when diagnostics and statistics are demonstrated unchanged. Batching reads may change when deferred device failures surface: state and verify the fallible boundary rather than remove it.

Backend-specific tiling should reflect actual scratch costs: Flex's B×R tile versus CUDA's B×k outputs. This is a direction for a later bounded orchestration design, not a proposal to couple budgets to an estimator CUDA special case or create extra public tuning controls. Exact queue_size=4.0 validation remains in force; an approximate query optimization must not silently make Exact budget-sensitive.

### Open questions and evidence

- Lifetime validation amortization needs an ownership/provenance proof; choosing a concrete API is future implementation planning, not this panel's deliverable.
- Direct invalid-graph tests and differential positive statistics should cover actual CUDA, not only Flex or silent hardware skips.
- Which read batching preserves documented errors while improving product time is unmeasured; changing the return boundary cannot be labeled pure scheduling without justification.

Sources: [fitted model fields and transform](src/standard_umap/estimator.rs#L331-L385), [fit marker check](src/standard_umap/estimator.rs#L176-L188), [backend query contract](src/backend.rs#L121-L134), [CUDA query graph validation](src/kernels/standard_umap/query.rs#L43-L51), [standard query tests](tests/test_standard_query.rs), [backend-specific source differences](src/standard_umap/ann.rs).

## F5 — Skew-aware complete reverse joins and streamed selection

### Description

Complete reverse indexing removes discovery rescans but introduces irregular work. How can it preserve all CUDA proposals while exposing hub-row parallelism and bounding intermediate selection storage?

### Discussion summary and member positions

- **Architect:** derive per-row proposal counts, partition heavy rows into bounded intervals, compute distinct local top-W and merge deterministically. Prefix and partial-list costs must be checked, not hidden under O(NW) adjacency storage.
- **Critic:** preserve full source-list semantics, reject degree/source caps and deduplicate before truncation. Shared fixed-width selection state must handle W>32, not assume a warp-sized candidate list.
- **Chair:** distinguish a fixed **tile size**, with every tile visited, from a fixed **tile count** that drops sources. The former is complete, not truncation. Stream waves/partial merges rather than retain every proposal or every partial list globally. Correct aggregate proposal counts to include own and forward lists.

### Outcome

For valid unique W-wide snapshots, target i has at most `P_i = W + W² + d_iW` proposal occurrences; `sum(d_i)=NW`, so `sum(P_i)=NW+2NW²`. Invalid/sentinel entries lower actual counts. A reverse CSR has at most NW source entries, but expanding their full lists costs up to NW² occurrences. Skew changes the maximum row span and work distribution, not this aggregate bound. A hub can have d_i=N−1; correctness cannot rely on average degree or discard sources above an average.

Complete work-balancing research direction:
1. Partition row/source/slot intervals into bounded chunks and distribute heavy-row chunks across tasks; enumerate every interval exactly as required by the candidate universe.
2. Each chunk retains the W smallest **distinct eligible IDs** under identical squared-distance/index keys. Retained-ID duplicate rejection can precede distance calculation; broader proposal deduplication has its own cost/storage.
3. Merge partials/waves with distinct-ID top-W selection and a single unambiguous output owner. A candidate excluded locally has W distinct better keys, so cannot enter global top-W. Taking W occurrences first and only then deduplicating is not lossless.
4. Preserve previous-iteration snapshots, next-output ownership and arithmetic consistency. Same ID must have the same key in every chunk; mixing arithmetic paths invalidates the proof. A sorted `(distance,id)` merge can remove adjacent duplicates only under that fixed-key condition.

No source/degree sampling, approximate cap, in-place iterative updates, early stopping or candidate-width change is selected. Nondeterministic reverse scatter order does not require sorting for this set-determined selection, but downstream graph COO ordering remains a separate preserved contract.

At N=5k/W=32 the occurrence upper bound is 10.4 million. Materializing one i32/f32 pair for each would cost **83.2 MB**, before reverse indices, selection temporaries or allocator overhead. This is a calculation, not an observed allocation. Bounded chunks plus immediate/wave merging avoid making that materialization a prerequisite. If A active chunk partials are retained, their ID/distance bytes are 8AW; all-partials-at-once can still be large. The final per-row selection remains 8NW bytes; counts, cursors, scan workspaces and partial-merge lifetimes also count.

Offsets over **proposal occurrences** can overflow even when reverse-entry offsets and row IDs fit. At N=1M/W=32, the occurrence bound is 2.08 billion, close to signed-32-bit capacity. Check all sums/products and launch limits; no 64-bit atomic/scan lowering is assumed. An explicitly bounded wave scheme need not materialize a global occurrence stream at all.

W is `max(k, min(4k,32))` capped by available rows, so it grows beyond 32 when k>32 (e.g. k=50). One-lane-per-entry or static shared-state designs must not silently truncate those configurations. Plane width is not a candidate-width limit.

### Open questions and evidence

- Actual reverse-degree skew, duplicate fraction, atomic hot-tail contention, and proposal versus selection time remain unmeasured. They determine efficient task granularity, not whether completeness is optional.
- Chunk/wave size, partial merge strategy and prefix dtype require resource/code-generation evidence before implementation selection.
- Reusing unordered-pair distances or repeated-list values is conditional on consistent computed keys and bounded cache/storage; no full N×N distance cache is proposed.

Sources: [candidate width](src/standard_umap/ann.rs), [CUDA proposal generation](src/kernels/standard_umap/ann.rs#L146-L200), [existing graph count/prefix/scatter mechanisms](src/kernels/standard_umap/graph.rs), T1's fixed-key proof and F3's byte accounting.

## F6 — Cooperative primitives and specialization limits

### Description

Verify actual locked-version building blocks and their scope, rather than infer capability from a book table of contents or assume CubeK supplies a ready-made exact neighbor selector.

### Discussion summary and member positions

- **Architect:** inspected installed CubeCL 0.11.0 shared storage, plane and synchronization source; collective participation and resource specialization are correctness obligations, not just tuning.
- **Critic:** source confirms conditional cross-cube publish/atomic handoff via `sync_storage`; the earlier blanket statement that every dependency requires a separate launch was too strong. Scalar min cannot implement a lexicographic key by itself.
- **Chair:** accept API expressibility, distinguish a device-scope fence/handoff from a global rendezvous, and require visibility **and forward-progress** proof for a single-launch alternative. Separate launches remain the conservative grid-dependency mechanism, not a categorical API limitation. No speculative universal fallback or backend enablement is mandated.

### Outcome

Verified source-level inventory in installed `cubecl-core-0.11.0`:

| Primitive | Supported research use | Obligation / non-claim |
|---|---|---|
| `Shared<[T]>::new_slice/new_aligned_slice` | Query/reference tiles or bounded selection state | Compile-time lengths; storage initially uninitialized. Initialize all read slots, respect alignment and shared-memory budget |
| `sync_cube` | Shared writes visible before cooperative reads/reuse | Every cube unit participates in uniform control; tail lanes cannot exit before a barrier |
| `plane_broadcast`, shuffle/XOR/up/down | Lane-owned key/value exchange | Broadcast source index is constant; edge shuffle reads are indeterminate and need masking; valid source participation required |
| Plane sum/min/max, inclusive/exclusive scan, all/any/ballot | Integer counts/compaction and key-selection building blocks | Runtime plane width and active lanes; integer overflow checks; float sum reassociation is class N. Ballot returns four u32 words, not necessarily one mask |
| `sync_plane` | Plane-local synchronization where supported | Source explicitly warns not every target supports it; no whole-cube/device barrier implication |
| `sync_storage` | Conditional storage publication before atomic handoff | Device-scope release/acquire only when runtime `device_memory_scope` support says so; otherwise only cube-scope guarantees |
| Compile-time flags/unrolling | A finite set of shape/capability-specialized variants | More generated kernels/cache entries, compile cost, register pressure and instruction size; not automatic autotuning or free speed |

`sync_storage` does **not** make all cubes resident or provide a grid-wide rendezvous. Producer publication and atomic announcement can express a handoff, but a spin-waiting consumer can block a nonresident producer in an oversubscribed launch. A single-launch scan/iteration dependency requires a valid progress/scheduling algorithm, proper atomic ordering and uniform local barriers, not simply a fence. Multi-launch count/prefix/scatter and ANN snapshot iteration boundaries remain safe research defaults; a supported/proven alternative is not ruled out by the book's missing barrier chapter.

For selection, `plane_min(distance)` returns a scalar, not the matching lower-ID argmin. Pairwise lexicographic shuffle comparisons or distance-min followed by eligible-ID selection are mechanisms to investigate. A u64 key packing nonnegative finite f32 bits plus ID can encode the order only with correct zero/sentinel treatment and verified dtype/backend lowering; it is not a claim that scalar f32 can encode every `(distance,id)` pair. No packing design was compiled or admitted.

Book/runtime plane constants must be interpreted correctly: `PLANE_DIM` is actual runtime width, `UNIT_POS_PLANE` is lane position, and unit/cube indices are distinct. A candidate launch has to establish real plane feature support, not silently assume 32 for all backends. CUDA's foreground scope does not authorize changing standard Flex or adding another backend.

The resolved stack is Burn/burn-cubecl 0.22.0 and CubeCL 0.11.0; the manifest uses ordinary version constraints, while the lock fixes current versions. The inspected CubeK 0.3.0 exports feature-gated tile/reduce/matmul families; that export inventory proves neither an exact distinct top-k operator nor suitability for irregular ANN/query gathers. Primitive operations above are already in the dependency stack; no dependency update is needed merely to discuss them. Check concrete library contracts before building a custom common primitive, but do not claim transitive presence is a usable public API.

### Open questions and evidence

- Kernel compilation through the actual Burn CUDA extension, generated arithmetic, capabilities, register/spill/shared usage and performance remain untested. Source existence is not compiled or speed evidence.
- Any chosen specialization must include tail masks/collective participation, k/W/D limits and deterministic selection tests, not just the 5k×32 happy path.
- Candidate registration/tune-key/cold-cache behavior needs measurement if autotuning is later used; no variant set or autotuner was selected here.

Primary dependency sources inspected: `cubecl-core-0.11.0/src/frontend/plane.rs`, `frontend/synchronization.rs`, `frontend/container/shared_memory.rs` in the local Cargo registry; `cubek-0.3.0/src/lib.rs`. Repository references: [locked dependency record](Cargo.lock), [plane constants](references/cubecl-book/core-features/constants.md), [feature querying](references/cubecl-book/core-features/features.md), [specialization](references/cubecl-book/core-features/comptime.md), [reduction example](references/cubecl-book/getting-started/parallel_reduction.md).

## F7 — Host ingress/egress and synchronization inventory

### Description

This topic arose from member exchange: enumerate actual host copies, device transfers and readback boundaries so compute/cache hypotheses are evaluated against complete product time rather than launch-only snippets.

### Discussion summary and member positions

- **Architect:** input flattening, graph diagnostics, initialization, RNG setup and stable edge scheduling have real host boundaries. Random timing controls differ from Spectral fixture paths.
- **Critic:** first-column RNG reads and batched diagnostic transfers are possible movement hypotheses, but deferred error boundaries and public stats must stay intact. Kernel handle cloning is not data transfer.
- **Chair:** verify the full inventory, including initial full-query construction followed by approximate block reconstruction. Do not call any boundary dominant without attribution, treat pre/post-pruning maxima as different values, and do not reopen frozen layout/RNG/schedule algorithms through an inventory discussion.

### Outcome

Source-level boundary inventory (not a measured cost ranking):

| Path / boundary | Inspected behavior | Timing/contract implication |
|---|---|---|
| Fit input | `matrix_tensor` flattens caller row vectors and constructs the device tensor | Packing/device ingress belongs inside current Rust fit timing; external file/data generation does not |
| Fit graph validation/maximum | Reads one head marker, then graph maximum weight after schedule preparation | Explicit fallible checks; do not drop or move errors outside the operation |
| Spectral-only initialization | Reads graph heads/tails/weights for host initialization, uploads initial embedding | Not present in Random Stage 1 timing controls; distinguish init method in comparisons |
| Optimizer RNG setup | Reads initial embedding, uses only each row's first component converted f32→f64 bits plus seeded wrapping offsets, uploads i64 state | Fit initial values originate on host; transform initial values are device-computed. Any movement reduction must preserve exact state arithmetic |
| Frozen edge schedule setup | Reads heads/weights, plus tails for shared fit, builds stable host CSR and uploads offsets/edge IDs | Real round trip outside kernel arithmetic; stable order, pruning/validation and layout behavior remain frozen |
| Final embedding | Fit/transform materialize host output | Synchronizes useful product output; not removable from an end-to-end result claim |
| Fit diagnostic graph | Optional tracing-only method reads back a copy after timed fit in profiling example | Diagnostic extraction/file writes are separate from fit timer; do not omit actual search used by fit |
| Approximate transform queries | Full query tensor constructed before search dispatch, then CPU row slices independently packed into block tensors; result blocks concatenated | Repeated host packing is source-visible. Whether an unused initial full-query device upload executes or is elided requires backend/lazy attribution |
| Approximate query stats | Every block reads stats and maps `-1` to `Numerical` | A synchronization/fallible boundary; magnitude and invalid semantics remain public primitive outputs |
| Transform graph and initialization | Reads graph marker and maximum before schedule preparation, then maximum after pruning | Maxima refer to different logical arrays; merging them requires semantic evidence, not a duplicate-scalar assumption |
| Exact block orchestration | Slices/clones/dispatches/cats blocks with offsets | Cloned handles/views are not inherently full copies; inspect contiguity/materialization/copy events |

Repeated Spectral graph downloads can overlap logically with later edge-schedule reads, but weights/pruning/validation stages differ; identical-data reuse must be proven. RNG first-column-only readback or use of still-available host initial values in fit is a conditional movement direction, not an approved change. Moving stable CSR computation to device is not automatically E: prove edge ordering, integer widths, invalid-endpoint/weight errors and clock/RNG/buffer ownership. Mere host/device movement can change rounding or when deferred failures surface.

The frozen layout kernel deliberately receives unaliased mutable clocks; handle clones in epoch calls do not authorize eliminating required de-aliasing. No epoch fusion, replay, layout schedule, RNG sequence or atomic update redesign is selected. This discussion can inform attribution while leaving the validated 16-batch path untouched.

Product evidence should include fit-once and repeated transforms with already-fitted reference data, plus named phases for packing/ingress, search, graph, host checks, initialization/schedule and final readback. Uninstrumented product timings and instrumented/copy traces answer different questions. An apparently short GPU kernel cannot establish short transform latency; conversely the existence of a readback alone cannot establish an expensive stall.

### Open questions and evidence

- Copy events/bytes and CPU packing time are unmeasured; the unused full-query tensor's actual upload behavior depends on the backend execution path.
- Batched or reduced reads need differential state/diagnostic evidence and an explicit error-boundary statement. No fallible boundary is removed by this panel.
- Layout-related host preparation is inventoried, not reopened for implementation. Relevant attribution must justify any separately scoped future proposal.

Sources: [fit graph/init boundaries](src/standard_umap/estimator.rs#L155-L264), [query block orchestration](src/standard_umap/estimator.rs#L399-L465), [transform graph/init boundaries](src/standard_umap/estimator.rs#L489-L581), [input packing](src/standard_umap/estimator.rs#L608-L611), [RNG setup](src/standard_umap/estimator.rs#L699-L723), [stable edge-schedule round trip](src/standard_umap/estimator.rs#L806-L850), [timer and diagnostic separation](examples/profile_standard_cuda.rs#L94-L120).

## Continuing panel — Final reconciliation and disposition

**Exchange record:** earlier F5–F7 IRC attempts encountered unavailable peers; those failures do not establish consensus. Both members were then resumed together for the chair-corrected final deliberation. Architect and critic each report a delivered live IRC message, a full peer reply, explicit mutual assent and no remaining dissent. Final receipts: `3-PanelArchitect-resume-159e6e7e482c5ee2` and `4-PanelCritic-resume-159e6e7e486c5ee3`. Their earlier conflicting statements are superseded by the recorded corrections below.

- **F5:** critic withdraws the claim that fixed-size tiles inherently truncate. Every tile must be streamed; degree/fixed-count caps remain excluded. Distinct local top-W precedes truncation, merged with deduplication under consistent keys; partials and wide prefixes have explicit workspace/overflow obligations.
- **F6:** architect withdraws the blanket separate-launch necessity; critic withdraws the mandatory portability-fallback proposal. Conditional storage publication can express a handoff, not a global barrier or progress proof. Separate launches remain the safe default; no compiled cooperative selection or speed gain is claimed.
- **F7:** critic withdraws automatic E classification for all host/device movement. Stable CSR order, RNG bits/wrapping state, errors and public stats must be preserved; pre/post-pruning maxima differ. Source-visible CPU packing duplication does not prove redundant PCIe traffic. Frozen layout/RNG/schedule remain untouched.
- **F1–F4:** prior corrections remain: numeric equality is not a bit-pattern golden; final-only rescoring cannot recover pruned winners; Inf seeds can survive; positive stats are public; standard persistence is not present. No precision/search-policy/error-boundary admission or new hard warning gate is granted.

Both members identify **no genuinely new independent topic**. The T1–T5/F1–F7 discussion queue is exhausted; capture provenance, deployment budgets, differential baselines, code-generation/resource evidence and transfer attribution remain explicit dependencies rather than endlessly recycled topics. The architect's final `WATCH / COMMENT` reflects those unresolved evidence requirements, not an implementation veto or approval.

This closes the research panel, **not implementation planning**. Preserve caller-visible semantics and the validated uncached 16-batch atomic layout. Any later implementation selection needs separately scoped authorization and evidence; arithmetic/search-policy changes require explicit admission. Only the two requested panel records were edited. No product code, dependencies, tests or benchmarks were changed; no experiments, product tests, formatters or performance gates were run. There are no measured speedup or new quality-pass claims.
