# Stage 2 — prioritized implementation phases

## Purpose and scope

Replace loose experimentation with targeted implementation phases for standard, non-parametric UMAP. CUDA neighbor search, approximate transform queries, and graph preparation are the foreground scope. Parametric/neural work is excluded entirely.

This roadmap draws on [panel_topics.md](panel_topics.md) and [panel_discussions.md](panel_discussions.md). The panel identified structural opportunities, not measured speedups or selected kernel designs. Phase ordering is a recommendation; prerequisite evidence and the user's workload priorities can justify changing it before planning.

## Phase workflow and ownership

Each phase follows this sequence:

1. **Resolve prerequisites.** Answer the phase-specific research questions and settle the required scope, semantics, resource budgets, and acceptance criteria. Research informs an implementation plan; it is not an open-ended candidate promotion process.
2. **Plan the implementation.** Define the feature(s), affected boundaries, tests, benchmark invocation, and completion conditions. Keep the plan focused on the phase's high-level goal. Obtain implementation authorization separately from this roadmap.
3. **Implement the feature(s).** Complete the agreed implementation and directly affected callsites, tests, and runtime guidance. Do not leave experimental alternatives, unfinished paths, or placeholders as the deliverable.
4. **Run final tests and verification.** Resolve implementation defects and complete the agreed correctness, quality, and resource checks. Missing required CUDA coverage or failed required checks must be reported as incomplete, not successful verification.
5. **Run the benchmark script.** Benchmark the finished implementation against the matched baseline using the agreed workloads, quality checks, and timing conditions. Keep changes in place and report the results without favorable reruns or post-hoc acceptance thresholds.
6. **Hand the decision to the user.** The agent's phase work ends when implementation is complete, all required final tests/verification are successful, and the benchmark has run with usable results. The user decides whether to **revert**, **refine**, or **commit and move on**, based on quality and speed criteria. A refinement is a separately directed continuation; do not automatically start it or the next phase.

There is no agent-owned promote/defer/reject gate. The agent does not automatically revert, commit, or replace the implementation based on benchmark results. A speed shortfall is reported for the user's decision. A correctness or required quality-check failure remains a verification failure and cannot be presented as a successfully completed phase; resolve it or report the blocker explicitly. A timed-out or failed benchmark is incomplete, not a valid result.

### Completion handoff

For each implementation phase, report:

- Implemented features and the relevant diff/files.
- Exact test and verification commands, results, and any limitations.
- Benchmark command, baseline identity, workload/build/device identity, and result artifact paths.
- Paired timing distributions, quality results, relevant resource measurements, and whether agreed criteria were met.
- Known limitations and remaining concerns, without taking the user's revert/refine/commit decision.

## Priority order

| Phase | High-level goal | Priority rationale |
|---|---|---|
| 0 | Establish implementation and benchmark readiness | Set concrete requirements and trustworthy comparison evidence |
| 1 | Eliminate ANN reverse-discovery rescans | Remove identified asymptotically redundant work before parallelizing it |
| 2 | Expose bounded exact-search pair parallelism | Address narrow, serial search execution |
| 3 | Eliminate repeated graph-preparation statistics | Implement a focused work-removal improvement |
| 4 | Parallelize remaining ANN proposal processing | Optimize residual work after reverse discovery is replaced |
| 5 | Eliminate repeated approximate-query validation | Improve transform without changing traversal semantics |
| 6 | Reduce measured search-boundary overhead | Address demonstrated packing, orchestration, and synchronization costs |

Priority is not technical dependency. Phases 2, 3, and 5 do not require Phase 1's implementation. Phase 4 depends on the accepted Phase 1 implementation and fresh residual-cost attribution. Do not automatically advance while the user's decision on the preceding implementation is pending.

## Phase 0 — implementation and benchmark readiness

**Goal:** Establish the decisions, baseline evidence, and benchmark tooling needed to plan the first implementation phase. This is a bounded preparation phase, not a performance optimization.

### Prerequisites and decisions before planning

- Identify relevant N/D/k, transform Q, initialization, fit frequency, and repeated-transform workloads.
- Specify the target GPU, usable memory budget, runtime budget, and quality/speed criteria.
- Agree on an equivalence-preserving first track: the recommendation is E-first, as defined in panel F1. Changed arithmetic or search/API semantics require explicit admission.
- Decide how meaningful speed changes are judged under timing noise. Preserve the inherited both-fit-mode comparison for general improvements unless the user explicitly chooses a mode-specific criterion.
- Inventory existing benchmark and differential-test support. Recover the original utilization capture provenance where available; missing historical captures need not block fresh, correctly attributed measurements.

### High-level steps

1. Establish matched baseline binaries, datasets, seeds, build/runtime identities, and cold/warm conditions.
2. Define synchronized, uninstrumented fit and transform comparisons, with exact/approximate modes reported separately.
3. Obtain enough attribution to distinguish search, graph preparation, packing, launch gaps, and readbacks for the first phase.
4. Establish comparisons for neighbor IDs, distance bits, query statistics, graph outputs, diagnostics, and errors.
5. Select a sparse workload ladder and budget the quality evaluator independently from the implementation.
6. Ensure the benchmark script supports the agreed product timings and quality checks. Identify any necessary harness changes explicitly in the plan.

Start with existing controls. Add targeted boundary cases: odd D, k/W greater than 32, query-block boundaries, ties, duplicates, hubs, and overflow-prone finite inputs. Avoid a full Cartesian sweep.

### Constraints and non-goals

- Utilization, occupancy, and cache throughput are diagnostics, not success targets.
- Do not invent a universal VRAM ceiling or numeric speedup threshold.
- Do not require every panel unknown to be resolved before planning the first focused implementation.
- No search, graph, or layout optimization in this preparation phase.

**Completion:** Agreed first-phase requirements, matched baseline evidence, usable benchmark/comparison tooling, and sufficient attribution to choose the first implementation goal. Verify any tooling changes and run its baseline benchmark. The user settles outstanding workload and acceptance choices before implementation planning proceeds.

## Phase 1 — eliminate ANN reverse-discovery rescans

**Goal:** Replace repeated reverse-source discovery while preserving the existing CUDA proposal universe.

### Prerequisites and decisions before planning

- Attribute reverse-discovery cost on relevant ANN workloads.
- Characterize reverse-degree distributions, hub extremes, construction cost, and peak workspace.
- Check count/prefix representability and launch limits.
- Confirm that every reverse source and its full candidate list must remain included.

### High-level implementation steps

1. Build complete reverse adjacency per iteration snapshot; count/prefix/scatter is the conservative starting direction for planning.
2. Retain existing proposal evaluation and selection initially.
3. Add differential iteration-output and downstream graph coverage, including hubs and invalid/sentinel entries.
4. Verify workspace accounting and benchmark construction cost, scaling, and product-time effects.

### Constraints and non-goals

- Preserve previous-iteration snapshots and separate next outputs.
- No reverse-degree caps, source sampling, candidate-width changes, or iteration changes.
- Preserve CUDA full-source-list proposals. Flex ANN is not an identical-output oracle.
- Account for counts, cursors, scans, and buffer lifetimes, not just adjacency entries.
- Exclude cooperative proposal selection and broad distance caching.

**Completion evidence:** Complete, bounded reverse discovery; successful equivalence and final verification; finished-implementation benchmark results. Removing O(N²W) discovery does not remove O(NW²) proposal expansion.

## Phase 2 — expose bounded exact-search pair parallelism

**Goal:** Increase useful parallelism across query/reference pairs without changing distance arithmetic.

### Prerequisites and decisions before planning

- Attribute serial distance scans, selection, and host orchestration.
- Determine feasible partition and partial-list budgets across target D/k.
- Establish generated-arithmetic and deterministic squared-distance/index merge requirements.
- Verify relevant locked-version primitives and actual CUDA-extension capabilities before relying on them in the design.

### High-level implementation steps

1. Partition reference work into bounded pieces.
2. Preserve each pair's serial feature accumulation.
3. Retain local top-k and merge under the existing squared-distance/index ordering.
4. Stream partitions or waves to bound workspace.
5. Include shared staging or vectorized loads only if prerequisite evidence supports them and the plan explicitly includes them. Keep their correctness and cost independently attributable.

### Constraints and non-goals

- Preserve global offsets, aligned self exclusion, distinct IDs, ties, sentinels, and errors.
- Assign each output and partial list an unambiguous owner.
- No unbounded distance matrix, feature-tree reduction, GEMM identity, TF32, reduced precision, or symmetry-routing redesign.
- Do not impose CUDA batching assumptions on Flex, whose temporary-memory costs differ.

**Completion evidence:** Successful differential and final verification across controls, tails, and larger k; bounded resource use; fit/query product benchmarks. Larger grids or higher occupancy alone do not demonstrate success.

## Phase 3 — eliminate repeated graph-preparation statistics

**Goal:** Compute the common smooth-kNN global statistic once instead of repeating its full fold per row.

### Prerequisites and decisions before planning

- Attribute the repeated global fold's contribution to graph-preparation time.
- Establish the exact current fold/order and arithmetic.
- Identify dependent rho-zero fallback, sigma-floor, graph-ordering, and error behavior.

### High-level implementation steps

1. Compute the shared statistic once using the existing fold/order.
2. Retain row-local calculations and existing graph construction semantics.
3. Add comparisons of intermediate values and final graphs using identical neighbor inputs.
4. Verify the completed change and benchmark graph-stage and end-to-end effects.

### Constraints and non-goals

- No reassociated reduction or changed fuzzy-graph policy.
- Preserve graph diagnostics and ordering.
- Do not move stable edge scheduling, CSR preparation, or RNG setup.
- Do not modify layout preparation or execution.

**Completion evidence:** Successful numerical/graph verification and benchmark results for the finished work-removal implementation.

## Phase 4 — parallelize remaining ANN proposal processing

**Goal:** Distribute residual proposal work, including hub rows, without dropping candidates.

### Prerequisites and decisions before planning

- Use the user-accepted Phase 1 implementation as the baseline.
- Attribute residual distance evaluation, duplicate handling, and selection costs.
- Characterize duplicate rates and worst-row spans.
- Select bounded chunk/wave and merge requirements, including register, shared-memory, spill, and synchronization limits.
- Establish that each occurrence of the same ID uses the same computed key.

### High-level implementation steps

1. Implement early retained-ID duplicate rejection if supported by the agreed scope and cost evidence.
2. Partition complete proposal work into bounded chunks, including heavy rows.
3. Retain local distinct-ID top-W selections.
4. Merge partials/waves with one output owner.
5. Include memory reuse only where its grouping/staging cost is justified by prerequisite evidence.
6. Verify ordinary and skewed cases, then benchmark the completed implementation.

### Constraints and non-goals

- Visit every required chunk; fixed tile size must not become fixed tile count.
- Deduplicate before local truncation.
- Handle W>32; plane width is not candidate capacity.
- Preserve snapshots and fixed-key selection semantics.
- Check occurrence products separately from adjacency-prefix limits.
- No sampling, early stopping, changed width, or full N×N distance cache.
- Prefer ordered launches unless a planned single-launch alternative proves visibility and forward progress.

**Completion evidence:** Successful completeness, equivalence, resource, and final verification; benchmark results covering ordinary and skewed inputs.

## Phase 5 — eliminate repeated approximate-query validation

**Goal:** Share graph validation within a public query-primitive call without weakening its contract.

### Prerequisites and decisions before planning

- Obtain fit-once/repeated-transform attribution for validation, traversal, and block readbacks.
- Specify positive statistics, invalid statistics, and current fallible boundaries.
- Establish how one validation result propagates while preserving every row's outputs.

### High-level implementation steps

1. Share validation within one primitive invocation.
2. Add differential coverage for IDs, distances, positive statistics, invalid graphs, and estimator errors.
3. Complete final verification and benchmark small and block-boundary Q values as well as relevant larger transforms.

### Constraints and non-goals

- Public primitive callers can supply invalid graphs; preserve validation for them.
- Per-call sharing still repeats across host blocks; report that limitation.
- No fitted-lifetime trust bypass or new trusted-state API.
- No bulk-frontier traversal.
- Preserve admission order, first-occurrence deduplication, visit truncation, queue commits, and next-pop choice.
- Do not discard positive statistics because the estimator only checks invalid markers.

**Completion evidence:** Successful output/stat/error verification and finished-implementation transform benchmark results.

Within-expansion distance batching is outside this phase. It requires a separately scoped follow-up only if traversal attribution justifies it and the user directs that work.

## Phase 6 — reduce measured search-boundary overhead

**Goal:** Remove demonstrated packing, orchestration, or synchronization overhead around search.

### Prerequisites and decisions before planning

- Determine which source-visible packing incurs CPU cost or actual device transfer.
- Identify which slices/clones materialize data.
- Attribute block launches, concatenation, and statistics-read latency.
- Establish ownership, memory, public-result, and error-boundary requirements for the selected change.

### High-level implementation steps

1. Rank attributed boundary costs and select a narrow scope before planning.
2. Implement the agreed change: query packing, backend-appropriate batching, concatenation, or diagnostic-read batching.
3. Verify ownership, memory peaks, statistics, and deferred-error behavior.
4. Benchmark synchronized end-to-end results for the finished implementation.

### Constraints and non-goals

- Duplicate CPU packing does not prove duplicate PCIe traffic.
- Handle cloning does not prove copying.
- Keep useful final host output inside product timing.
- Preserve public statistics and fallible boundaries.
- Exclude layout scheduling, RNG, replay, epoch fusion, and mutable-buffer de-aliasing changes.

**Completion evidence:** Successful final verification and measured boundary/product results, not merely fewer source-level calls.

## Shared constraints and benchmark requirements

- **Frozen implementation:** preserve the validated uncached 16-batch atomic layout, schedules, RNG behavior, and ownership requirements.
- **Equivalence first:** unchanged source folds are not proof. Compare actual current-backend outputs and generated behavior. Characterize ties, overflow, signed zeros, sentinels, and compiler contraction without silently changing their behavior.
- **Quality:** retain existing per-run approximate recall ≥0.90 and fit/frozen-transform trustworthiness gap limits ≤0.02. Equivalent changes must not spend recall headroom. Existing tracking warnings remain warnings. Do not require byte-identical final embeddings from the accepted nondeterministic layout.
- **Timing:** use matched paired runs, existing at-least-five-run controls, distributions, and honest cold/warm accounting. Keep instrumentation timings separate from uninstrumented acceptance results.
- **Resources:** account for live, peak, reserved, and process memory distinctly. Check products/prefixes and launch limits. Large-workload truth and quality evaluation require their own feasibility budget; sampled diagnostics are not full-output parity.
- **Benchmark tooling:** identify the exact script and invocation in each phase's plan. Reuse existing tooling where suitable; extend it explicitly when transform, statistics, or other required comparisons are unsupported. Do not silently substitute fit-only measurements for transform results.
- **No automatic decision:** leave the implementation in place after the completion handoff. The user chooses revert, refinement, or commit and progression. Do not commit, revert, or start the next phase without that direction.
- **Excluded work:** parametric/neural implementations, backend unification/enablement, Auto-cutoff recalibration, and unadmitted numerical/search-policy changes.

ANN reverse discovery is the recommended first implementation goal because it removes redundant work. Choose exact-search parallelism first instead if Phase 0 establishes that workload relevance, attribution, or reverse-index resource costs favor it. This is a pre-planning priority decision, not a claim that either implementation already has a measured speed advantage.
