# HDBSCAN core-distance GPU refactor plan

## Decision and scope

Refactor the pipeline boundary before replacing its search algorithms. Keep exact neighbor semantics, host-ordered MST unions, and eager fitted host outputs. Retain squared cores on the device; use compact neighbor storage and explicit bounded batches; then optimize brute-force selection and KD traversal independently.

This is a design, not an implementation or a speedup claim. No proposed kernel has been compiled or run. Source/API inspection used Burn 0.22.1, CubeCL/cubecl-core 0.11.1, and CubeK 0.3.1 from Cargo.lock. The skill examples target 0.11.2, so their compatibility with this lock must be checked before adopting code. No dependency upgrade is required.

Primary scope: `crates/fast-hdbscan/src/{estimator.rs,backend.rs,kernels/}` and focused tests/benchmarks. Shared exact-search improvements belong in `fast-cubecl-common`; HDBSCAN-specific core rules and seed logic do not. Keep both cores backend-free. Do not move clustering extraction, finite-row remapping, or union-find onto the device in this work.

## Current behavior and hazards

- `estimator.rs:200-226`: brute force launches at most 64 queries per batch and reads IDs and distances separately in every iteration. This bounds device memory but also limits parallel query work and introduces repeated host synchronization boundaries.
- `estimator.rs:265-328`: CoreDistances owns only host vectors. KD results are immediately downloaded; both routes allocate padded `[n,min_samples]` IDs/distances and copy compact results into them.
- `estimator.rs:342-355,643-645`: host kNN seeding needs squared distances; fit additionally clones public neighbors. Squared cores are uploaded again if Boruvka needs further rounds.
- `fast-cubecl-common/src/kernels/neighbors.rs:17-74`: each query is one unit scanning every reference and insertion-shifting its output-backed sorted list. Work is O(n²(d+k)) in the worst case, not a cooperative tiled top-k algorithm.
- `kernels/kdtree.rs:175-375`: KD already uses a flat tree, iterative near-first traversal, cached node bounds, and an output-backed max-heap. It is not recursive host code awaiting a mechanical GPU port.
- `kernels/kdtree.rs:383-400`: CPU-aware row striding and legal launch geometry already exist. Preserve them rather than reintroducing GPU-sized CPU cubes.
- `fast-hdbscan-core/src/mst.rs:21-72,75-145`: seed ordering and component tie handling are observable contracts. An atomic union or unordered edge queue is not an equivalent replacement.

`core_dist_perf.md:229-244` records prior dispositions: tree-order query writes lacked a validated permutation contract, unchecked launches lacked a safety proof, and early termination at `best <= core` changed tied endpoints. Do not repeat those changes as unconditional quick wins. Its timings are historical, workload-specific observations, not fresh measurements for this plan.

## Semantics that must remain unchanged

Let n be the finite-row count, s the resolved min_samples, and k = min(s,n-1), for n > 1.

1. Search excludes self by original finite-row identity, including offset query batches. Sort neighbors lexicographically by `(squared_distance, finite_row_id)`.
2. Internal computation uses squared f32 distances. Preserve the existing feature accumulation order initially; do not substitute the cancellation-prone norm/matmul distance identity.
3. Core squared distance is infinity when s >= n, zero when s = 1, and the distance at column s-1 otherwise. The infinity branch has precedence. Do not reinterpret s as an index including self.
4. Public neighbors remain flat, row-major `[n,s]`, with -1 in unfilled columns. Backend missing-neighbor sentinel n is distinct from public -1. A real ID with infinite distance is valid, not padding.
5. Public core distances remain f32 square roots of squared cores. MST finalization retains its existing f64-root behavior.
6. One finite row bypasses search: empty neighbors and an infinite core, including s = 1. Preserve existing invalid/empty/all-nonfinite input behavior and finite-to-raw remapping.
7. Host seeding visits sources in increasing order and chooses the first neighbor with `core_i >= core_dst`. It stops there even when that union is redundant; it must not search for a later cross-component neighbor.
8. Per-point Boruvka ties use destination ID. Host component reduction uses strict weight improvement and first-source appearance order. Preserve both, including infinite-weight real edges.

## Buffer and ownership design

Use separate device computation and host materialization responsibilities, not one struct mixing implicit transfers into accessors.

| Buffer | Producer | Consumers | Layout/lifetime |
| --- | --- | --- | --- |
| finite input | existing upload | search, Boruvka | f32 `[n,d]`, retained through MST |
| compact neighbor IDs | search | host seeding, fitted output | i32 `[q,k]` per output batch; never `[n,s]` on device |
| compact neighbor distances | search | core extraction, host seeding | f32 `[q,k]`; release after seed use |
| squared cores | core extraction | Boruvka, host materialization | f32 `[n]`, one writer per row; retained through MST |
| partial top-k pairs | brute-force tile pass | merge passes | separate ID/distance arrays; bounded scratch, ping-pong |
| host compact results | batched readback | existing init_from_knn | width k, not s; discard distances after seeding |
| public neighbor vector | host finalization | FittedHDBSCAN | `[n,s]`, -1 tail padding; move into fitted result |

A private computation result should expose retained device cores and explicit host seed data. Prefer ordinary tensors/structs and the existing HdbscanOps/backend-extension pattern, not a new execution framework.

Keep the existing input layout initially. Separate ID/distance arrays are useful for independent consumers; this does not imply every tree field should become SoA. Layout experiments require access-pattern evidence.

### Memory policy

Do not replace bounded brute-force outputs with an unconditional full `[n,k]` allocation. That would reject workloads currently supported with small device batches.

- Derive q and reference tile counts from checked per-buffer sizes, alignment, scratch requirements and device launch limits. Profile several legal batch sizes rather than hard-coding 64 as a performance policy.
- Use a full compact output only when the selected capacity policy permits it; otherwise use bounded output batches on the same selected backend. This is partitioning, not CPU/CUDA execution fallback.
- Check arithmetic, per-allocation limits and the total planned live working set before allocating. Account for CUDA pitch where tensors actually use it; existing neighbor kernels allocate contiguous outputs.
- Preserve bounded host staging for large outputs. Initially the host needs O(nk) compact IDs/distances plus O(ns) public IDs; never pad host squared distances to s. Avoid duplicating IDs when k = s by transferring ownership after seeding.
- Reused scratch buffers carry explicit logical counts/offsets. Capacity is not a valid logical length. Define non-aliasing for ping-pong passes and initialize every slot read later.

The current KD operation allocates full `[n,k]` outputs. Phase 2 may retain that existing allocation contract; bounded KD outputs require an explicit query-range/original-row-ID kernel interface, not slicing downloaded results. Add that interface only as a separately tested extension when capacity requirements justify it. The required bounded-output preservation above applies to the already-blocked brute-force route.

## Implementation sequence

### 1. Freeze contracts and capture a baseline

Files: existing estimator/KD/candidate tests, core MST tests, HDBSCAN benchmark and profiling harnesses.

- Extend independent-oracle tests across forced BruteForce and KdTree, rather than relying only on Auto.
- Capture warmed baseline search, transfer, host padding/seed, Boruvka and complete-fit timings; record peak live buffers, batch/launch counts and transfer bytes.
- Separate uninstrumented fit timing from tracing runs with their extra synchronization. Record backend/device, input distribution, n/d/s and JIT/cold-start cost separately.

Exit: exact expected outputs and reproducible baseline workloads exist before changing ownership or arithmetic.

### 2. Refactor residency and host materialization without changing search

Files: estimator.rs, backend.rs if needed, a small HDBSCAN core-extraction kernel and its registration, estimator tests.

- Replace the host-only CoreDistances boundary with retained device core tensors plus compact host seed results.
- Extract core values using an existing tensor select/slice/fill when it preserves offsets and layout without costly copies. Otherwise add a simple map kernel: one row owns one core output; implement the s boundary rules explicitly.
- For KD, keep compact device outputs long enough to extract cores. For brute force, extract each batch into its disjoint segment of the full core vector before releasing batch outputs.
- Batch IDs and squared-distance reads with Burn `Transaction::register(...).try_execute()`, mapping execution/conversion failures into HdbscanError. Batch per-round Boruvka weight/index reads similarly. This is one explicit transaction boundary, not a promise of one physical DMA or zero synchronization.
- Call existing init_from_knn with compact width k. Construct only the required public padded IDs and move them into FittedHDBSCAN after seeding; eliminate padded knn_sq and redundant full-vector clones.
- Pass the retained device squared-core tensor into spanning_tree; delete the host-to-device reconstruction. Host DSU and KD node-label refresh remain unchanged.
- Update allocation preflight to match the actual lifetimes and selected batching policy.

Exit: outputs and host seed order match the baseline; no squared-core reupload; no `[n,s]` device allocation; no padded host distance vector. Full compact distance readback still exists and must be reported honestly.

### 3. Replace brute-force serial selection with bounded parallel top-k

Files: common neighbor kernel/orchestration and tests; HDBSCAN estimator sizing/callsite as required. Keep public shared NeighborOps semantics unchanged for UMAP.

Proposed CUDA algorithm, not existing library functionality:

1. Tile query rows and reference rows. A cube processes one query tile/reference tile pair; units own independent distance pairs, with coalesced feature loads or shared reference tiles where beneficial.
2. Each reference tile emits at most k lexicographically smallest `(distance²,id)` pairs per query. For small k, implement bounded local selection; do not globally sort an n² distance matrix.
3. Merge partial lists hierarchically in separate dispatches, retaining only k pairs per query. Top-k of the union of per-tile top-k sets is exact under the same total ordering.
4. When all reference tiles cannot fit in scratch, process bounded reference groups and merge their results into a running top-k. No host reads inside reference-group processing. Extract cores and read output only at the query-batch boundary.

Choose explicit tile sizes and a bounded family of small-k capacities after profiling. Preserve a runtime-k implementation for arbitrary supported requests rather than truncating to a convenient maximum. Small partial tiles use neutral `(infinity,n)` pairs; real infinite-distance IDs win ties against n. Self exclusion uses global query offsets.

For the CPU runtime use a barrier-free row/block decomposition, one legal set of worker units and sequential local selection. It need not use CUDA shared-memory/plane algorithms. Dispatch variants from device properties, not scattered per-kernel guesses.

Start with the existing direct squared-distance arithmetic. Test shared-operation squared and unsquared modes, cross-query inputs and offsets; optimization must not change UMAP's stricter unsquared overflow/width behavior. If the optimized path is squared-only, retain the independently required unsquared implementation explicitly rather than silently changing it.

Exit: oracle-identical IDs/ties and correct distances; scratch is bounded by the capacity policy, not n²; benchmark shows where the new decomposition wins, including small CPU workloads.

### 4. Optimize KD storage and query coherence independently

Files: kdtree.rs and KD tests; core tree builder/validation only if the measured layout change requires it.

- First test a small bounded family of local heap capacities for small k. Keep a general runtime-k variant. Inspect generated CUDA/register and spill behavior: `Array` does not guarantee register residency, especially with dynamic heap indexing and existing traversal stacks.
- Benchmark local query caching and feature/vector specialization separately. Preserve feature tails, squared accumulation order and bounded JIT variant count.
- A tree-order query schedule is allowed only after proving the schedule is a permutation. Do not use unvalidated public tree.indices as output ownership. The safe design is a validated internal schedule that maps each query to exactly one original-row output; retain natural row ownership for operations accepting arbitrary public trees.
- Only if gather traffic dominates, test tree-ordered point storage with explicit forward/inverse maps. Preserve original IDs for self exclusion, tie comparisons, output order and component/core gathers. Include reorder memory and build cost in the measurement.
- Sweep legal CUDA cube/leaf sizes separately from CPU configuration. Keep near-first traversal, cached bounds and strict tie-safe pruning.

Exit: each retained optimization has its own correctness tests and controlled ablation. Do not retain a more complex layout solely because it looks GPU-friendly.

## Follow-on boundary, not required for the first refactor

If full squared-distance readback is still significant, move only per-row seed selection to the device: after all cores are ready, scan each ordered neighbor row for the first `core_i >= core_dst`, emit destination and `max(core_i,distance²)`, then apply those records on the host in increasing source order with the existing DSU behavior. This reduces seed payload from O(nk) distances to O(n), but public neighbor IDs still require O(nk) readback.

This extension needs an explicit bounded-memory design: streamed neighbor rows may no longer be resident when all cores become available. Compare retaining device pages against host staging/reupload or recomputation; do not claim distance-transfer elimination for the streamed case without accounting for that cost. Introduce a narrow host seed-record helper with an oracle equivalence test if this extension is selected. Do not smuggle in a concurrent GPU union-find redesign.

Also defer GPU tree construction, Morton/radix pipelines, device node-component aggregation, component-wide pruning atomics and work queues until profiling establishes their need. There is no demonstrated need for a device-wide scan/sort in the primary design; do not invent a CubeCL API for one. The skill's u32 primitives would require explicit adaptation and pinned-version tests.

## CubeCL implementation rules

- Follow pinned 0.11.1 runtime launch tests and existing CubeTensor/TensorArg host wrappers, not remembered 0.11.2 signatures. Implement new operations in the owning crate with correct fusion metadata.
- Query shared-memory, unit, cube-count and plane capabilities. Never assume a 32-lane plane. Unsupported plane operations select an explicit non-plane implementation before launch.
- Mask inactive lanes; every unit reaches every cube barrier. Cross-cube merges require separate dispatches, not sync_cube or publication through an unrelated atomic counter.
- Cap multidimensional grids or use bounded grid-stride work ownership. Every logical output has exactly one writer. Validate arithmetic/index conversions and logical offset ranges.
- Keep checked launches. Do not add launch_unchecked without a separate complete memory-safety argument.
- Preserve strict backend selection and errors; capacity partitioning never authorizes switching runtimes after execution failure.

## Verification and acceptance

### Correctness matrix

- n = 0/1/2 handling; s = 1, n-1, n, n+5; default min_samples resolution and invalid config behavior.
- Duplicate/all-equal coordinates, equal distances, ties across different reference tiles and merge levels, signed zero, finite features whose squared distances overflow to infinity, mixed nonfinite input rows.
- Query batches around old 64-row boundaries and new tile/cube/vector boundaries; nonzero self offsets; forced multi-group merges; partial final tiles; deliberately tiny capacity limits in pure sizing tests.
- Forced CPU and CUDA variants against the independent core oracle and brute-force exact search. Check exact IDs, padding and deterministic seed/MST endpoints, not only clustering quality. Use justified float tolerances only when arithmetic changes intentionally.
- Validate heap order, partial-list merge invariants, original/permuted ID roundtrips and unique output ownership if those variants are added.
- Existing estimator/route fixtures must retain MST, linkage, condensed tree, labels, probabilities and finite-row mapping behavior. Preserve numerical/error tests; do not weaken assertions to admit an optimization.

### Commands during implementation

Run focused tests per changed behavior, then the union once at completion:

```sh
cargo test -p fast-hdbscan-core --features test-support
cargo test -p fast-hdbscan --no-default-features --features cpu
cargo test -p fast-cubecl-common --no-default-features --features cpu
cargo test -p fast-umap --no-default-features --features cpu
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --no-default-features --features fast-umap/cpu,fast-hdbscan/cpu,fast-umap-python/cpu
```

For CUDA, run HDBSCAN and common suites with `--no-default-features --features cuda` on an available NVIDIA device, and verify the changed tests actually executed. Existing HDBSCAN tests may skip unavailable CUDA; do not count skips as coverage. Common's `FAST_UMAP_REQUIRE_CUDA=1` can require common CUDA execution. Plane/cooperative variants need actual CUDA tests, not just CPU-runtime success. Run `scripts/test_features.sh` if features/backend wiring changes. Check the skill examples against 0.11.1 before copying their primitive implementations, if any are needed.

### Performance matrix

Use existing hdbscan_cpu/hdbscan_cuda benches and profile_hdbscan_stages. Include n near Auto boundaries and representative larger n; d = 2/10/30/50; small and larger s, duplicates, coherent and shuffled distributions. Test forced routes so unchanged Auto policy cannot hide regressions. Include the historical 10k and 272,383-row dataset when available, without treating it as the entire workload.

Measure cold/JIT separately and repeated warmed samples with synchronization at measurement boundaries. Report distribution/uncertainty, search-only latency, transfer/seed cost, full-fit latency, scratch/live memory, launch counts and bytes transferred. Compare each change against the same-condition baseline and the simplest correct implementation. For KD, include tree/reorder build costs and spill/occupancy evidence. For brute force, include all merge dispatches and bounded-batch overhead. Do not equate fewer bytes or better asymptotics with a measured speedup.

A phase is accepted only when exactness and memory bounds hold on both supported backends and measurements justify retaining its complexity. Protect every case in the existing benchmark grid on CPU and CUDA, with both forced BruteForce and KdTree routes: warmed full-fit regression must not exceed 10% per case against the original same-condition baseline. Retained complexity requires a repeatable benefit beyond measurement noise; overlapping uncertainty is inconclusive, not a win. Use measured backend/workload dispatch splits to avoid regressions, without changing Auto routing thresholds absent separate cross-route evidence. Larger and historical datasets provide additional evidence rather than replacing the protected grid.

Preserve completed original baseline measurements; different candidate sample counts do not require rerunning that baseline. During iteration use a fixed representative subset covering input sizes, dimensions and min_samples on both backends and forced routes, with 30 samples in one campaign. Repeat borderline comparisons rather than repeating every case. Run the full protected grid once at integrated acceptance; collect only missing original baseline cases before changing the implementation. Report incomplete or noisy evidence honestly.

Evaluate every listed KD optimization that profiling identifies as relevant, with independent experiments and combined-candidate ablations. Record profiling-based non-relevance or rejection explicitly. Retaining no KD variant is acceptable; this does not permit skipping relevant experiments or substituting residency-only delivery for the required bounded brute-force outcome.

## Documentation and delivery

Update existing API comments for internal/public widths and sentinel distinctions, plus singleton behavior where currently implicit. Update existing README/runtime guidance only if user-visible routing, memory limits or supported behavior changes; this plan does not propose such changes. Keep dependency features and backend permissions unchanged.

Deliver phases as independently reviewable changes: ownership/readback first, brute-force selection second, measured KD variants third. Remove superseded host copies and implementations rather than leaving dead compatibility paths. Keep the user's existing core_dist_perf.md untouched.

### KD experiment disposition

The fixed eight-case subset was evaluated with 30 warmed samples on CPU and actual CUDA. Independent local-heap, neighbor-query-cache and candidate-query-cache experiments, their combination, and an optimistic reordered-input proxy were compared with natural traversal. Fresh controls exposed timing drift, so old-reference improvements alone were not attributed to an optimization. All experimental paths passed the focused CPU/CUDA contract suites before performance evaluation.

| Family | Disposition and evidence |
| --- | --- |
| Bounded local heap | Rejected: case-local CUDA gains, fixture regressions, increased register/local-memory cost; no robust general benefit. |
| Neighbor query cache | Rejected: mostly noise or regressions and increased local-memory traffic. |
| Candidate query cache | Rejected: isolated small-case gains; measured local loads rose from approximately 1.57 MB to 209 MB in the profiled case. |
| Heap plus both caches | Rejected: one fixture regressed approximately 11.8% against the fresh control. |
| CUDA cube sizes 16/64/128 versus 32 | Keep 32: larger cubes regressed larger cases; 128 regressed n2048/d30 by approximately 28%. Size 16 offered only marginal gains. |
| Host leaf sizes 20/80 versus 40 | Keep 40: 20 regressed larger CUDA cases by approximately 10%; 80 was mixed or inconclusive. An unchanged single-leaf case also moved substantially, identifying drift rather than a tree-size benefit. |
| Validated internal query schedule | Rejected after independent evaluation and a borderline rerun: the initial fixture gain did not repeat; n4096/d10 improved approximately 3.6–5.9%, but small-case timings remained unfavorable. Identity schedules already bypassed indirection, and nonidentity schedules reused the uploaded tree index buffer without another upload. The additional implementation was not justified by robust cross-case evidence. |
| Tree-ordered reference storage | Not pursued after the explicit attribution gate: the optimistic proxy added only approximately 1.2% full-fit benefit over scheduling at n4096/d10 and no net combined neighbor/candidate span benefit; at n2048/d30 it was slower. High L2 hit rates alone were not treated as evidence against gather latency. |
| Coherence/storage combinations | The optimistic combined proxy and the standalone validated schedule were measured. The attribution gate found no incremental storage kernel benefit warranting a production combination. |

The storage attribution used ten measured stage-profile runs after an excluded warmup, with benchmark-equivalent synthetic data and the pinned blob fixture. Boruvka round counts were unchanged. At n4096/d10, scheduling versus the reordered proxy took 13.878 versus 13.708 ms full fit, but combined core-neighbor and candidate spans were 6.321 versus 6.557 ms. At n2048/d30 the corresponding fits were 9.163 versus 9.252 ms. The proxy relabels input rows and excludes reorder cost; it is neither production correctness evidence nor a shippable optimization.

No KD optimization is retained. Temporary switches, alternate paths and the benchmark-only proxy are removed; the public arbitrary-tree path retains natural-row ownership. The durable ledger records experiment distributions, profiler output, failed compilation and repairs, and review receipts. Integrated full-grid acceptance and final review are separate requirements; the KD disposition alone does not establish them.

### Integrated verification evidence

The final implementation passed the full common and HDBSCAN suites on CPU and required actual CUDA, both backend-free core suites (including the independent HDBSCAN test-support oracle), and the full UMAP suites on CPU and CUDA. Formatting checks and workspace CPU Clippy passed. Newly introduced Clippy warnings were repaired; existing KD-test/UMAP CUDA-test warnings and Burn future-incompatibility notices were not suppressed. UMAP's existing opt-in seeded-drift diagnostic remains ignored by its normal suite; it is not counted as coverage.

One final protected-grid campaign measured all 38 datasets, both forced routes and both backends, with 30 warmed samples per fit. The original baseline remained byte-for-byte unchanged. CUDA's 76 observed mean comparisons were all below the 10% cap. The initial CPU campaign had eleven over-cap means with strongly unstable distributions; these failed observations remain in the evidence. A bounded check revisited only those comparisons plus one borderline comparison, without source or baseline changes. The sole remaining borderline comparison received one longer, still-30-sample measurement.

All 76 final observed CPU mean comparisons are within the cap. The worst is n4096/d10/min_samples32, forced KD: +9.69%, with a 95% interval of [+7.96%, +11.68%]. This is not an improvement, and the confidence upper bound is not below 10%; review must not turn the observed-mean result into a stronger statistical claim. CUDA's worst observed mean is -2.73%. Full distributions, the initial CPU failures, selective checks and baseline digest are preserved in `target/core-distance-acceptance-evidence.json` and the durable ledger references.

The historical 10k/272,383-row input `comparison/embeddings_2d.parquet` is absent from the current checkout. Prior-session historical reports exist but are not substituted for current-state measurements. README/runtime guidance and dependency features remain unchanged because user-visible routing, backend permissions and supported semantics did not change. Boundary-cohort and terminal-review receipts belong to the durable ledger and are required independently of this measurement record.

### Retrospective baseline evidence recovery

The completion audit found that the original warmed Criterion capture did not establish the planned cold/JIT, granular transfer/live-buffer and complete-output snapshots. Those missing observations were recovered **after implementation**, from original source `d4762aa2fc39ced947e2519b77ff2e862096c92a` and the final source in isolated worktrees. They are retrospective original-source measurements, not authentic earlier captures; the original 912 Criterion files and their digest remain unchanged.

`target/core-distance-recovery-evidence/captures.json` records 24 serial profiling processes: synthetic n64/d50/min_samples32, n2048/d30/min_samples32 and n4096/d10/min_samples5, both forced routes, CPU and actual CUDA, original and final implementations. Each process records one cold first fit and three subsequent warm fits, complete Chrome traces, synchronized search/readback/host-padding/seed/candidate stages and allocator boundary samples. These instrumented timings include observation overhead and do not replace the uninstrumented 30-sample acceptance distributions.

Every cold process uses a unique CubeCL environment. `jit-compilation-evidence.json` preserves 12 CUDA environment stores and 55 actual native compilation records, all within their first fit. CUDA driver/hardware caches were not cleared. LLVM's compilation cache is process-local; CPU first-fit timings include setup/JIT, but isolated LLVM compiler duration is not measured. Cold costs are not uniformly improved: the n4096/d10 CPU brute-force first fits were approximately 161 ms original and 358 ms final. No cold-start non-regression is claimed; the acceptance cap applies to warmed full fits.

`semantic-comparison.json` contains 100 three-way exact comparisons (300 snapshots): untouched original estimator, instrumented original and final implementation, across both backends/routes. Complete fitted-object dumps and explicit IEEE float-bit vectors match without normalization, covering representative inputs, singleton/two-row cases, duplicates, overflow, min_samples boundaries, the 257-row batch tail and finite-row remapping. All 24 timed-fit snapshots also match their independent semantic captures. Build/source patches, hashes and rejected cross-worktree Cargo artifact reuse are recorded in the recovery provenance files.

`recovery-accounting.json` validates all 96 fit partitions, per-batch stage sums/counts, per-round readbacks, allocator samples and source-predicted payloads. `recovery-comparison.json` contains all 12 original/final pairs, including unfavorable cold and instrumented-warm observations. Compact readback payload remains 8*n*k; original neighbor/candidate reads are separate, final reads are paired transactions, and the original 4*n core upload occurs once only when Boruvka rounds are needed, versus zero final core uploads.

`launch-observer-evidence.json` adds 24 separate, untimed native launch-observer fits after one excluded warmup each. Every source-derived neighbor/candidate launch count agrees with the observed count (`launch-model-cross-check.json`); framework kernels are preserved separately rather than omitted from total launches. For example, n4096/d10 CUDA brute-force issued 204 total launches originally and 717 finally: grouped parallel search trades more, shorter launches for lower warmed fit time. These are issued-launch observations on successful normal fits, not compiler-record counts or new acceptance benchmarks.

Measured allocator boundary maxima are samples, not guaranteed global live-memory or physical VRAM peaks. Source-derived phase/buffer bounds and logical payload accounting are separate from those observations; logical payloads are not physical PCIe traffic. Grouped scratch's 8*B*k*(G+2) formula is logical storage excluding alignment; the actual source sums aligned allocations and enforces its 32 MiB/per-page limits. The recovery analysis and durable ledger preserve these limitations rather than manufacturing unavailable peak or compiler measurements.
