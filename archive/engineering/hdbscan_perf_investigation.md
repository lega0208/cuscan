# HDBSCAN core-distance performance investigation

## Scope and pre-experiment assessment

This plan is written from source inspection only, before running experiments or interpreting existing profiling results. Investigate CPU (CubeCL LLVM) and CUDA without changing production algorithms. Deliver measured findings and ranked implementation opportunities in `hdbscan_perf_report.md`; do not claim unimplemented optimizations as speedups.

Pinned dependencies: Burn 0.22.1, CubeCL 0.11.1, CubeK 0.3.1 (`Cargo.lock`). Skill examples target CubeCL 0.11.2, so repository and installed 0.11.1 source are authoritative.

### Actual stage contract

- `estimator.rs:384–547` computes exact, self-excluding squared-distance neighbors, retains host IDs/distances for MST seeding, extracts a device core-distance vector and builds a host core vector. This stage is not just a kth-element reduction.
- For complete graphs, Auto chooses KD-tree at >=256 finite rows and <=30 features (`estimator.rs:28–32,124–133`). Higher-dimensional default graphs use the separate kNN/ANN path, not necessarily this stage. The standard 16-feature campaign therefore exercises KD-tree, not the old brute-force hotspot described in `hdbscan_timing_findings.md`.
- KD construction/upload are separate top-level stages. KD search launches once for all rows; host readback is a single transaction. Core extraction is a tensor slice, potentially realized later. A tracing neighbor span includes explicit synchronization; readback spans include transfer, conversion and any deferred extraction work. They are not pure kernel counters.
- Brute force batches queries in groups of 256 and reads both outputs after every group. CPU uses common's serial-per-query exact kernel. Squared CUDA uses grouped partial top-k lists and merge dispatches (`fast-cubecl-common/src/kernels/neighbors.rs`). This route is a useful exact comparator but not the current default for the campaign.
- KD neighbors (`kernels/kdtree.rs:221–422`) assigns one query to one unit, traverses near-first with 33-entry private node/bound stacks, calculates scalar feature loops and maintains a max-heap directly in global output memory, then heapsorts it. Both pruning and heap admission preserve distance/ID ties. `kd_launch_geometry` uses one CPU cube with one unit per reported core; CUDA uses 32-unit cubes. CPU queries are assigned cyclically, not contiguous chunks.

## Hypotheses and discriminating experiments

1. **Neighbor traversal dominates extraction/transfer.** Measure inclusive core time and disjoint fine spans; use CUDA Systems device totals and kernel counts. Reject transfer-first optimization if its measured fraction is negligible. Do not count nested spans twice.
2. **KD pruning effectiveness is workload-dependent.** Compare Auto/KD with forced brute force on the identical 20,000 x 16 input, same `min_samples=5`. Separately enable existing `FAST_HDBSCAN_KD_DIAGNOSTICS=1` counters to quantify node visits and leaf positions versus n²; exclude diagnostic timings from normal performance comparisons.
3. **CUDA scalar traversal has poor latency hiding/coherence.** Capture representative neighbor-kernel occupancy, SM, DRAM, L1/TEX and stall metrics using existing Nsight scripts. Low DRAM utilization alone is not proof of launch overhead or uncoalesced access. Count launches before recommending batching. Candidate implementations: multiple planes per cube, coherent query ordering, packed/reordered leaf data, or cooperative feature/candidate evaluation. Changes require GPU-specific correctness tests.
4. **Query ordering can improve CPU cache behavior and GPU traversal coherence.** Compare original shuffled input with a deterministic spatially coherent ordering of the same points at 20k. Record the permutation, preprocessing time and input hash. Undo permutation for label partition comparison. This is an input-order sensitivity experiment, not a drop-in implementation or guaranteed net speedup; a production implementation should reorder internal work while preserving original candidate IDs and public outputs.
5. **Small-k global heaps and scalar feature loops are optimization targets.** Sweep `min_samples` (5 versus 15/30 as useful) at 20k to separate selection pressure from fixed work; diagnostics distinguish increased traversal from increased heap work. Candidate small-k compile-time local arrays and feature-vector loads must preserve deterministic ties, +infinity behavior, feature tails and reduction-order tolerances. No inferred benefit without a measured prototype.
6. **CPU runtime scheduling is already bounded, but locality/SIMD remain open.** Read installed CubeCL 0.11.1 scheduling code. Compare ordering sensitivity and core/kernel proportions. Recommend contiguous query chunks, SIMD feature loops and a host Rust/Rayon exact-search baseline as separate prototypes; do not assume GPU launch geometry translates to CPU parallelism.

## Execution protocol

1. Inspect available tooling/hardware and record revision, dirty files, binary hash, dependency versions, CPU/GPU/driver identity and commands. Build the release tracing example once. Preserve all raw outputs in a fresh `artifacts/hdbscan_core_investigation/` directory.
2. Start with 20k: one excluded warmup and three measured fits for CPU and CUDA, serially. Use the existing deterministic 8-center Gaussian generator (16 features, noise/center scale 1, seed 42), min cluster size/sample count 5. Existing benchmark runner validates exact CPU/CUDA label equality and writes input/provenance. Use a bounded timeout that includes warmup and all repetitions; report timeouts as incomplete, never as performance samples.
3. Profile CUDA launch counts and representative counters separately from baseline wall timings. Existing scripts accept the native benchmark identity and resolved route. Profiling instrumentation/replay timings must not replace unprofiled stage timings.
4. Run the targeted exact-route, ordering and k-sensitivity comparisons at 20k. Retain raw JSON/trace/labels, compare partitions as appropriate, and summarize medians with raw ranges. Keep original baseline conditions fixed; avoid simultaneous CPU/GPU benchmark processes.
5. Use 50k to confirm main findings. Use 100k only for final validation of conclusions when feasible, not for iterative tuning. If a comparator is already clearly slower, avoid unnecessary expensive repetitions at larger sizes.
6. Run focused existing neighbor/oracle tests on CPU and required CUDA, plus profiling parser tests where practical. Performance label agreement is not an exact-neighbor correctness oracle. No new production algorithm is delivered in this investigation.

## Correctness constraints for proposed changes

Preserve finite-row indexing/remapping, self exclusion, ascending `(squared distance, original ID)` order, +infinity and padding behavior, ties/duplicate points, `min_samples == 1`, `min_samples >= n`, and public neighbor widths. Preserve all compact neighbors needed by MST seeds, not merely a core scalar. Use actual tree index indirection; only prune bounds strictly beyond the admissible worst distance. Retain bounded stacks and checked allocations. Any SIMD/cooperative path needs nonmultiple feature/query tails and backend-specific execution coverage. Do not introduce cross-cube publication/spin protocols.

## Report acceptance

Provide route/source map; measured CPU/CUDA core/fine timings and scaling; kernel/diagnostic evidence; ranked opportunities with exact proposed decomposition/layout, likely scope and correctness risks; explicit negative results; raw artifact paths and reproducible commands. Separate measured facts, source-backed explanations, and untested hypotheses. Do not extrapolate the fixed 16-D Gaussian workload to all HDBSCAN data or claim production speedups without an implementation comparison.
