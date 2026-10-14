# Performance investigation: 15k versus 50k

## Conclusion

**The 50k regression is predominantly host-side spectral initialization, not ANN. Duplicate handling is the trigger in this dataset, and the Rust spectral solver handles the resulting graph much less efficiently than SciPy.**

The comparison also gives the implementations different fitting problems: `unique=True` only for umap-learn (`plot_comparison.py:85–90`). At 50k, Python fits 41,744 distinct vectors; fast_umap fits all 50,000 observations. Deduplicating before fast_umap reduces the measured full fit from **88.79 s to 7.74 s**. Switching only to random initialization reduces it to **5.84 s**, but does not repair the quality difference (see [geometry report](geometry_quality_findings.md)).

Spectral initialization **cannot occur before ANN**: it consumes the fuzzy graph produced by ANN. Asynchronous CUDA dispatch and incomplete stage logging can make the wait look like unlabelled preparation before ANN.

## Experiment and provenance

- Checkout: `14421bd` (`ANN performance greatly increased for CUDA (possibly others)`); no estimator, kernel, or comparison-script changes.
- Input: `comparison/feedback_clustered.parquet`, sampled using the existing `load_sample`, Polars seed 42, separately at 15,000 and 50,000 rows; 128 float32 features.
- Hardware: Ryzen 7 9800X3D, RTX 3080 10 GB, WSL2. NVIDIA-SMI reported driver 615.78.02 / CUDA UMD 13.4. Other desktop GPU activity was present; this was not an isolated benchmark host.
- Python reference: installed umap-learn **0.5.12**. Native Python module: `python/fast_umap/_fast_umap.abi3.so`.
- Full-fit settings: Euclidean, 15 neighbors, 2 components, `min_dist=0`, spectral unless specified, `random_state=None`, default epochs (200 for every measured fit). fast_umap uses CUDA; Python uses its default parallel execution. The collector omits verbose printing.
- Separate stage diagnostic: existing release `profile_standard_cuda`, approximate search, spectral initialization, seed 42, **one epoch**, no warmups, **min_dist=0.1**. These traces locate the bottleneck; they are not the plot's full-fit benchmark. The profiler was rebuilt with `cargo build --release --no-default-features --features cuda,tracing --example profile_standard_cuda`, and the raw-50k result was reproduced after rebuilding.
- Evidence and runnable collectors: `../artifacts/comparison-investigation/`.

## Measurements

### Full Python API fits matching the plot's main settings

| Fit | Input rows | Rows actually fitted | Seconds |
|---|---:|---:|---:|
| umap-learn, unique=True | 15,000 | 13,312 | 10.97 |
| fast_umap, raw, spectral | 15,000 | 15,000 | 6.71 |
| umap-learn, unique=True | 50,000 | 41,744 | 12.01 |
| fast_umap, raw, spectral | 50,000 | 50,000 | 88.79 |
| fast_umap, raw, random | 50,000 | 50,000 | 5.84 |
| fast_umap, explicitly deduplicated, spectral | 50,000 | 41,744 | 7.74 |
| umap-learn, unique=False | 50,000 | 50,000 | 19.60 |

Explicit fast deduplication uses `np.unique(X, axis=0, return_inverse=True)` and maps the fitted embedding back to original rows. Its fit timer excludes that external preprocessing; measured 50k `np.unique` alone took approximately 0.05 s in the initial input inspection. Python's internal deduplication is included in its timer. File loading, plotting, and imports are excluded throughout.

These are single process-first fits, not warmed repeated throughput measurements. They reproduce the reported reversal but do not establish general performance parity.

### Native stage isolation

| One-epoch spectral diagnostic | Full fit | Host initialization | Spectral subroutine |
|---|---:|---:|---:|
| Raw 15k, initial executable | 6.15 s | 1.56 s | 1.49 s |
| Raw 50k, initial executable | 96.55 s | 84.14 s | 83.81 s |
| Raw 50k, rebuilt executable | 86.38 s | 83.10 s | 82.75 s |
| Deduplicated 50k → 41,744 rows, rebuilt executable | 7.61 s | 2.92 s | 2.65 s |

The rebuilt raw fit spends **96%** of total time in initialization. Reducing the row count by only 16.5% through deduplication reduces spectral time by about **31×**. This is a graph-conditioning/convergence issue, not ordinary linear growth with sample count.

The initial 15k→50k spectral time increases 56× for 3.33× more observations. Do **not** interpret two different graph realizations as proof of a general cubic algorithmic complexity.

Python's measured `spectral_layout` takes **0.09 s at 15k unique**, **0.27 s at 50k unique**, and **7.50 s at raw 50k**. Duplicate removal helps Python too, but its raw spectral path is still substantially faster than Rust's. These are different ANN graphs, so this is not an identical-matrix solver benchmark. Python's `simplicial_set_embedding` timer contains spectral and layout work; nested stage durations must not be summed.

## Why the Rust spectral path is vulnerable

`src/standard_umap/init.rs` implements a custom bounded restarted Lanczos solver:

- Basis capped at 64 vectors; restart retains one Ritz vector, not the accumulated subspace (`:11–16`, `:690–729`).
- Each output mode is solved separately with deflation (`:495–619`).
- Two full reorthogonalization passes per iteration (`:763–770`), with host f64 vector arithmetic and sequential projected EVD (`:804`).
- Absolute residual threshold `1e-7` and an application budget proportional to `5*n` per mode (`:12–16`, `:500–504`). Budget exhaustion has a warning and random fallback; **no such warning appeared in these runs**.
- Normalized edge factors recompute a square root on every sparse operator application (`:133–144`).

Installed umap-learn's `umap/spectral.py` uses SciPy `eigsh` at these sizes, with `ncv=max(2*k+1, sqrt(n))`, `tol=1e-4`, and `maxiter=5*n`. For two output dimensions it requests three modes together; its basis allowance is roughly 122/223 vectors for a connected raw 15k/50k graph, or 115/204 for the measured deduplicated graphs. Disconnected layouts solve their components separately. Its tolerance is not the same stopping test as Rust's absolute residual, so their numeric values are not directly comparable.

The duplicate-heavy graph contains disconnected small groups and likely weak bottlenecks in its large component. That is consistent with slow restarted convergence. **Per-mode iteration counts, eigenvalues and residual histories were not recorded**, so a particular eigengap or restart count is a hypothesis, not a measured result.

There is also an expensive dense component-centroid operator for more than four components (`init.rs:204–260`, `:364–371`): it recomputes pairwise centroid affinities per application. The exported raw ANN graph has only four components, and the pruned weighted initialization graph was not exported. This path is therefore a potential risk, not an established cause of the observed 83 seconds.

## Why stage order looked misleading

Actual fit order (`src/standard_umap/estimator.rs`):

1. Validation, host copies and training-tensor upload (`:64–87`).
2. ANN (`:114–148`); Auto selects approximate above 4096 rows (`:21`, `:322–328`). Both sizes use ANN, with iteration caps 14 and 16 respectively.
3. Fuzzy graph (`:149–175`).
4. Blocking graph-marker and maximum-weight readbacks, schedule construction, then full graph readback for spectral (`:176–243`).
5. **Host spectral initialization** (`:244–261`).
6. Initialization upload, curve/RNG preparation, CUDA layout, final embedding readback (`:262–290`).

The `neighbor_search` and `fuzzy_graph` tracing spans primarily time host dispatch, not completion. Initial raw-50k trace: these spans close in fractions of a millisecond, while about **12.19 s** of deferred ANN/graph/JIT/readback work lands before `initialization`. At 15k that gap is about **4.36 s**. The rebuilt 50k fit has a smaller non-spectral remainder, illustrating cache/compilation effects.

Conversely, spectral's 83-second host span really is host work. Layout also executes asynchronously: the trace's tiny `optimization` host duration is **not** the GPU layout duration. A full stage breakdown needs synchronized measurement or CUDA events/Nsight, not sums of these host spans.

## Recommended changes, in priority order

1. **Make the comparison symmetric.** Deduplicate once and feed the same matrix to both, mapping results back for plotting; also separately compare both on raw observations. Do not silently change production duplicate semantics to disguise the solver problem.
2. **Fix spectral convergence/scaling at its source.** Evaluate an established implicitly/thick-restarted or block sparse eigensolver, matching reference tolerance and convergence checks. Precompute normalized sparse weights. Compare on identical frozen graphs, including this duplicate-heavy graph; preserve explicit convergence diagnostics.
3. **Repair profiling attribution.** Report synchronized ANN, fuzzy graph, graph export, spectral, layout and compilation/warmup separately. Do not charge pending GPU work to a host-stage label.
4. Add a 50k × 128 duplicate-heavy regression case with both runtime and quality acceptance; existing 15k-scale ANN success did not cover this spectral failure mode.

`init='random'` is useful as a diagnostic, not the proposed final fix: its raw-data quality remained below the deduplicated comparison.

## Reproduction and limitations

From the repository root, with the existing release extension available:

```sh
uv run --no-sync python artifacts/comparison-investigation/diagnose.py python-unique 50000
uv run --no-sync python artifacts/comparison-investigation/diagnose.py fast-spectral 50000
uv run --no-sync python artifacts/comparison-investigation/diagnose.py fast-random 50000
uv run --no-sync python artifacts/comparison-investigation/diagnose.py fast-unique 50000
uv run --no-sync python artifacts/comparison-investigation/diagnose.py python-all 50000
```

Collectors consume the frozen `sample-15000.f32le` / `sample-50000.f32le` matrices generated by `plot_comparison.load_sample`. Stage traces and JSON results are retained alongside them. Native profiler invocations used its documented CLI, e.g.:

```sh
target/release/examples/profile_standard_cuda \
  artifacts/comparison-investigation/sample-50000.f32le 50000 128 1 \
  artifacts/comparison-investigation/profile-50000-rebuilt \
  --search approximate --warmups 0 --runs 1 --init spectral \
  --trace artifacts/comparison-investigation/trace-50000-rebuilt.json
```

No product fixes were applied. Python `random_state=None` is stochastic/parallel; native None resolves to seed 42 but CUDA updates can still be order-dependent. No multi-seed confidence intervals, GPU event profile, or identical-graph solver comparison was performed. The release profiler build succeeded with upstream future-incompatibility warnings for `burn-cubecl` and `burn-cubecl-fusion` 0.22.0; no broad test suite was run for this investigation-only change.
