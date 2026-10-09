# ANN kernel versus PyNNDescent: performance analysis

## Conclusion

The main problem is redundant work combined with a serial GPU implementation—not Python being inherently faster. PyNNDescent's hot paths are Numba-compiled native code.

This investigation did not modify source files. It compared the current CUDA implementation with installed PyNNDescent 0.6.0, inspected existing Phase 1 profiling and iteration artifacts, and ran warmed Python construction benchmarks on the saved campaign dataset.

## Timing evidence

Workload: **16,384 × 16 float32 inputs, 25 neighbors**. The Python benchmarks used UMAP's settings: 11 random-projection trees, `max_candidates=60`, a maximum of 14 iterations, Euclidean distance, and seed 42. Each configuration had one untimed construction followed by three timed constructions. Index preparation for querying was not included.

| Configuration | Median construction time | Actual expansion iterations |
|---|---:|---:|
| PyNNDescent, 15 threads, tree initialization | 135 ms | 2 |
| PyNNDescent, 1 thread, tree initialization | 615 ms | 3 |
| PyNNDescent, 15 threads, without tree initialization | 240 ms | 4 |
| CUDA ANN, existing Phase 1 profile | approximately 528 ms GPU execution per build | 14, plus initialization |

Python timed samples, in seconds:

- 15 threads, trees: `0.131372`, `0.134686`, `0.136179`.
- 1 thread, trees: `0.614746`, `0.620273`, `0.605919`.
- 15 threads, no trees: `0.244260`, `0.236291`, `0.240325`.

The CUDA estimate comes from two builds in the existing Nsight profile: 30 `ann_step` launches, two initialization launches, two output launches, and 28 sets of reverse-CSR construction kernels. Their summed execution time is approximately 1.057 seconds, or 528 ms per build.

This suggests approximately **4× slower construction than multithreaded PyNNDescent**, but not slower than the single-threaded version. It is not a fresh matched CPU/GPU benchmark: CUDA timing is historical, profiler-derived device execution and excludes host overhead, whereas Python timing is fresh construction wall time. Neighbor semantics also differ: the CUDA estimator requests 24 non-self neighbors, retains 32 internally, and then adds self; Python requests 25 neighbors.

The original campaign records full CUDA fit and transform timings, but does not record Python ANN construction times. Full fit timings must not be interpreted as ANN-only timings.

## Why the CUDA kernel loses

### 1. It continues after convergence

`artifacts/stage2-phase1/identity.json` records identical neighbor-ID and distance hashes for iterations 8 through 14. Nevertheless, the host loop in `src/kernels/standard_umap/ann.rs:422` always executes all configured iterations.

PyNNDescent counts successful graph updates and stops when progress falls below its threshold (`pynndescent_.py:296–320`). In the warmed campaign-matched multithreaded benchmark, it stopped after two expansion iterations rather than fourteen.

The repeated hashes establish wasted iterations for the recorded CUDA workload; they do not establish a universal convergence point for other datasets.

### 2. It repeatedly evaluates old and duplicate proposals

`consider()` in `src/kernels/standard_umap/ann.rs:106–151`:

1. Computes the complete feature distance for a valid, non-self proposal.
2. Scans all 32 retained neighbors.
3. Only then rejects duplicate IDs.
4. Shifts the sorted list when inserting a candidate.

Each subsequent iteration clears the output list and reconstructs it. Previous candidate distances are not supplied to `ann_step`, so even retained neighbors have their distances recomputed.

PyNNDescent preserves graph distances and separates new candidates from old candidates. Its candidate joins evaluate new–new and new–old pairs, avoiding old–old comparisons. A computed pair distance can update both endpoints (`utils.py:550–658,700–733`). This does not mean PyNNDescent eliminates every repeated pair across different joins.

For CUDA's internal width of 32, each expansion has approximately:

```text
32 current proposals
+ 32² forward proposals
+ 32 × average reverse degree
≈ 2,080 proposal calls per row
```

With full, unique candidate lists, average reverse degree is 32. Including initialization, the workload therefore generates approximately **478 million proposal calls** across fourteen expansions, before self/invalid filtering. Accepted and rejected duplicate proposals both incur distance computation unless excluded by the initial self/validity check.

### 3. Work within each row is serial

`ann_step` assigns one GPU thread per row (`ann.rs:230–288`). That thread serially performs all proposal evaluations, feature reductions, duplicate/selection scans, and insertion shifts.

There is no cooperative parallelism across features or proposals. Neighbor-list state is stored in global tensors. Neighboring threads access different rows with strided addresses, while candidate feature loads are indirect. Variable reverse degrees create unequal work across lanes.

These are source-visible architectural disadvantages. The available profile does not contain hardware counters sufficient to attribute their individual costs or prove a particular occupancy or bandwidth bottleneck.

### 4. Selection scans too much

CUDA scans the entire retained list even when a proposal cannot improve it, and maintains sorted order through shifts.

PyNNDescent uses a persistent max-heap (`utils.py:483–533`):

- Reject against the worst retained distance immediately.
- Check duplicates only for potentially useful insertions.
- Restore heap order rather than shift a sorted array.

CUDA's selection cost is especially wasteful when most proposals are duplicates or cannot improve an already-converged neighborhood.

### 5. Random initialization creates additional discovery work

CUDA starts with uniformly sampled random neighbors. PyNNDescent initializes from random-projection-tree leaves and fills remaining slots randomly.

The controlled Python comparison supports the value of that initialization on this clustered dataset: disabling trees increased warmed construction from **135 ms to 240 ms**, and expansion iterations from **2 to 4**. This measures the net effect within PyNNDescent, not the expected speedup from adding trees to CUDA.

## Reverse-CSR construction is no longer the dominant cost

The Phase 1 profile attributes approximately **99% of ANN kernel execution time to `ann_step`**. Reverse zero/count/prefix/scatter together contribute approximately **1%**.

The old reverse-discovery rescans have already been replaced by complete reverse adjacency. Further optimizing reverse-CSR construction would miss the dominant remaining cost for this workload.

## Separate approximate-transform problem

Graph construction and approximate query execution are different kernels. The approximate-query kernel has additional structural problems:

- `src/kernels/standard_umap/query.rs` launches one-thread blocks.
- Every query independently validates the entire training graph (`query.rs:43–51`).
- Distance evaluation, seen-ID searches, and candidate selection are serial within the query.

`approximate_query_kernel` accounts for **52.6% of GPU execution time** in the recorded approximate campaign, versus 9.0% for `ann_step`. Those percentages include other campaign work, warmup, and diagnostic exports; they are not a pure fit-only breakdown.

## Optimization priority

1. Eliminate converged iterations and repeated old/duplicate distance work.
2. Parallelize distance evaluation across features or proposals.
3. Replace full-list selection with efficient threshold rejection and retained-state selection.
4. Evaluate informed initialization, measuring both initialization cost and downstream savings.
5. Treat approximate-query launch geometry and repeated graph validation as a separate optimization target.

Early stopping, candidate sampling, or a different initialization can change the algorithm's outputs and require quality validation. They are not automatically bit-preserving optimizations. Even selection and arithmetic changes need explicit tie-breaking and numerical-equivalence checks.

The present construction kernel is closer to repeated exhaustive neighborhood expansion than PyNNDescent's incremental NN-descent algorithm. Its GPU placement does not compensate for the extra work and serial execution structure.

## Evidence sources

- CUDA construction: `src/kernels/standard_umap/ann.rs`.
- Internal candidate width: `src/standard_umap/ann.rs:11–15`.
- Estimator iteration policy and non-self neighbor handling: `src/standard_umap/estimator.rs:132–155`.
- CUDA query execution: `src/kernels/standard_umap/query.rs`.
- Installed PyNNDescent: `.venv/lib/python3.14/site-packages/pynndescent/pynndescent_.py` and `utils.py`.
- Installed UMAP PyNNDescent configuration: `.venv/lib/python3.14/site-packages/umap/umap_.py:327–343`.
- Dataset: `artifacts/stage2-phase1/phase1-candidate-stage2-48uxsust/train.f32le`.
- CUDA profile totals: `artifacts/stage2-phase1/phase1-candidate-stage2-48uxsust/nsys-stats-approximate.csv`.
- Iteration hashes: `artifacts/stage2-phase1/identity.json`.
- Campaign workload/provenance: `artifacts/stage2-phase1/phase1-candidate.json`.

Python measurements were executed with `uv run --no-sync python` during the investigation; the timed samples above were captured from command output, not from a saved benchmark artifact.
