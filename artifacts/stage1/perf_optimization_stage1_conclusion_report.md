# Stage 1 Performance Optimization — Conclusion

## Executive conclusion

Stage 1 substantially accelerated Rust CUDA **standard (non-parametric) UMAP** without relaxing the required quality checks. The final implementation is the **uncached, 16-batch atomic layout**, not the fastest experimental kernel. The run is complete, with clean final reviews and an independent terminal audit. Parametric neural training, Flex/WGPU optimization and neighbor/graph algorithm changes were outside scope.

**CUDA is still much slower than parallel Python on the measured workload.** The target of beating Python by 10% was missed and explicitly reported; completion does not mean every performance target was achieved.

## Final result and measurement context

The acceptance workload was 5,000 float32 samples × 32 features, 20 epochs, Euclidean distance, 15 neighbors, two output dimensions and random initialization. Results are medians of five timed fits after one warmup. CUDA ran on an RTX 3080; Python used umap-learn 0.5.12 with 15 CPU/Numba threads and unseeded parallel execution. CUDA timings include synchronization and embedding readback.

| Fit mode | Original Rust CUDA | Final Rust CUDA | Rust speedup | Parallel Python |
|---|---:|---:|---:|---:|
| Exact | 3.501 s | **0.828 s** | **4.23×** | 0.051 s |
| Approximate | 4.221 s | **1.551 s** | **2.72×** | 0.071 s |

Final CUDA remains approximately **16.1× / 22.0× slower than Python**, respectively. These are fit timings, not transform-speed results. Earlier 10-epoch measurements are historical evidence, not the denominator for these speedups.

The complete acceptance/profile suite finished in **101.5 seconds**, within its 600-second deadline. All four small-workload cases improved over the original baseline; their final times ranged from 0.130 to 1.173 seconds.

## Quality and reliability

- Every final primary CUDA fit met the hard neighborhood-preservation comparison: trustworthiness within 0.02 of the corresponding Python mean. Final mean trustworthiness was approximately 0.942 in both modes.
- Separate frozen fixtures use 200 fit epochs and 66 transform epochs. Their gaps were **0.001397 / 0.003647**, below the 0.02 limit.
- Approximate fit-neighbor recall was about **99.76%**; transform-query recall was **99.93%**, comfortably above 90%. Fit recall uses 14 non-self neighbors; query recall uses 15.
- Approximate-mode tracking against the original Rust mean minus half a sample standard deviation remains a **warning only**, as agreed. It is not a failed hard quality gate.
- Verification included 53 explicitly required-CUDA tests, 16 Python harness tests, and the portable Rust suite. Deadline handling and duplicate-neighbor recall accounting were corrected and regression-tested.

Concurrent GPU updates remain nondeterministic. Passing these fixtures is evidence for the tested workloads, not a universal guarantee or exact reproduction of serial coordinates.

## Experiment takeaways

**Parallel layout updates delivered the main gain.** Sixteen ordered batches retain more controlled update ordering while sacrificing some speed. Per-edge scheduling, random-state ownership, unchanged transform reference points and retained input buffers remain protected.

**Low-epoch quality alone was an incomplete screening rule.** Snapshot/Jacobi quality rose from roughly 0.70–0.72 at 20 epochs to 0.93–0.94 at 50–100, with nearly flat total fit time. A hybrid using live atomic updates in one quarter of epochs raised 20-epoch quality to roughly 0.938–0.940. These are useful research results, but neither candidate received final transform-fixture acceptance.

**Fit success did not guarantee transform quality.** Single-launch atomics reached 0.661 / 1.400 seconds in a five-run fit suite, but subsequently failed the frozen transform limit with a **0.027933** gap. Cached-16 and distance-2 variants also failed, at **0.026118 / 0.020051**. Their failures and code were preserved; the qualified 16-batch control was restored rather than promoting a favorable rerun.

**Smaller overhead did not necessarily improve the product.** Copy reuse removed copy launches but failed the paired, both-mode improvement criterion and was reverted. CUDA graph replay is supported by CubeCL, but a safe fixed-workspace implementation and credible paired gains were not established. No replay speedup is claimed.

## What Nsight established

Nsight Systems shows that **neighbor search is now the larger cost**, particularly approximate search. In the final diagnostic captures covering two fits, exact-neighbor kernels totaled about **1.31 seconds**, approximate-neighbor steps **2.80 seconds**, and layout kernels about **0.31 seconds** per mode. Layout-copy kernels totaled only about **0.067 milliseconds**; further copy removal is not a meaningful priority.

Nsight Compute sampled an active layout batch at about **142 microseconds**, with just **3.58% achieved occupancy** and five blocks of 64 threads. The small batches underfill the GPU, exposing an update-ordering/quality versus parallelism tradeoff—not proof that simply increasing occupancy is safe.

Profiler captures include instrumentation and startup/autotuning activity. Their kernel totals explain bottlenecks; they must not be substituted for steady-state fit timings. The trace's throughput-test kernels are autotuning, not UMAP optimization work.

## Recommended next stage

1. **Keep the validated 16-batch implementation as the baseline.** Preserve the hard fit/transform checks and negative evidence when testing faster schedules.
2. **Scope neighbor search and graph preparation as Stage 2.** Layout-only changes cannot plausibly close the current Python gap. Identify whether algorithm choice, repeated work or GPU execution dominates before tuning kernels.
3. **Benchmark larger, representative datasets.** Establish where CUDA becomes useful, with matched modes, end-to-end timings, repeated runs and quality checks. Do not extrapolate this small synthetic workload to all workloads.
4. **Continue promising scheduling research separately.** Higher-epoch Jacobi/hybrid recovery merits investigation, but promotion requires frozen transform checks and repeatable gains. Revisit graph replay only after defining safe buffer lifetimes and measuring its actual benefit.

## Evidence and deliverables

The authoritative final measurement is `artifacts/stage1-restored-control-20.json`; the original reference is `artifacts/stage1-serial-20.json`. Recovery, hybrid and paired-copy results are separate `stage1-*` artifacts. `artifacts/layout-experiments/manifest.json` records preserved implementations and reproduction paths. Historical progress notes contain superseded tentative selections; the manifest's final selection is `final_control`.

The accompanying [offline chart dashboard](perf_optimization_stage1_experiment_results.html) separates acceptance results, historical screens and failures, and provides source data.
