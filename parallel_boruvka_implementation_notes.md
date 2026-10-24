# Parallel Borůvka: M0/M1 implementation notes

## Decisions and findings
- M0 passes on LLVM CPU and required CUDA (RTX 3080), using CubeCL 0.11.1 `Atomic<i32>::load`/`fetch_min` and device-side f32↔i32 reinterpretation. No CAS fallback is needed. Tests cover both zeros, the smallest subnormal, 1, f32::MAX, infinity, shared-address contention across cubes, and reset bounds.
- Important deviation: `value + 0.0` did **not** canonicalize -0 on the CPU JIT (observed bits `0x80000000`). Canonicalize that exact integer bit pattern to zero instead; both runtimes pass. Future M5 must retain this rule rather than trusting fast-math arithmetic.
- M1 adds backend-free `knn_kruskal` with f64 cores/MRD after widening squared f32 distances, ordinal-stable tie decisions, final `sort_mst`, and minimum-component-ID infinity bridges. `KnnGraph.rows` is explicit so empty/singleton/zero-width graphs are representable.
- Validation rejects malformed IDs/rows, decreasing valid distances, and NaN/negative distances (including padding). Padding needs no particular sentinel distance; missing ranked neighbors give infinite cores, except min_samples≤1 gives zero cores. Size arithmetic and allocation byte bounds are checked; no new error variant or recoverable-OOM promise.
- Fixed `merge_components` first-component-appearance slot order. Explicit candidate presence distinguishes pruned placeholders from real infinity candidates; later strict-less tie behavior and no-progress errors are preserved.
- The test-support oracle constructs exact neighbors, Kruskal, DSU and bridges independently of production graph helpers. Tests compare exact graph-route edges/core bits, complete-graph weight multisets, ordinal ties, invalid inputs, disconnected forests, padding, overflow, and sample-count boundaries.

## Verification
All passed:
- `cargo test -p fast-hdbscan-core --features test-support` — 57 tests.
- `cargo test -p fast-hdbscan-core` — backend-free/default-feature coverage.
- `cargo test -p fast-cubecl-common --test atomic_bounds m0_atomic_cpu_` and `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common --test atomic_bounds m0_atomic_cuda_` — one real runtime test each, not skipped.
- `FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-cubecl-common`.
- `cargo test -p fast-hdbscan --test test_kdtree_search kd_traversals_match_exact_host_oracles_cpu` — 1 test; `cargo test -p fast-hdbscan --test test_estimator _cpu` — 4 tests.
- `cargo fmt --all -- --check`; Clippy with `-D warnings` for core/all-targets/test-support and common's atomic_bounds target.
- `cargo tree -p fast-hdbscan-core --no-default-features -e normal` — no normal dependencies.

Current manifests enable CPU/CUDA unconditionally; old `--features cpu/cuda` guidance is stale. Cargo reports existing future-incompatibility warnings in burn-cubecl/burn-cubecl-fusion 0.22.1; these were not suppressed.

## Deferred / untouched
M2–M8, estimator routing/configuration, ANN movement, device winner reduction, and performance tuning remain out of scope. M0 proves primitives only, not future KD-bound pruning correctness or performance. Unrelated `comparison/hdbscan_stages.html` changes are preserved. The consensus ralplan is the refined plan; no further planning rewrite was performed.

## M2 execution: implemented, acceptance blocked

- M2–M6 execution was subsequently approved against `pending-approval.md` rev 3 / plan rev 5. M2 is implemented uncommitted; M3, M5, M4 and M6 have not started because the approved milestone gates are serial.
- ANN kernels and candidate-width sizing moved to common. `NeighborOps` owns approximate search with a comptime squared-output flag; UMAP passes false. The estimator uses typed `ann_preflight`, sharing its checked plan derivation with the operation. Input, workspace and capability validation tests moved with the implementation. No HDBSCAN-to-UMAP dependency was added.
- Before relocation, `capture_cpu` and required `capture_cuda` passed. Each backend captured 30,720 bytes of IDs, 30,720 bytes of distance bits and 4,096 bytes of embedding bits under `.gjc/_session-01a169da-98fb-76d2-bad4-f28715c9381e/baselines/m2`. These files have not been replaced.
- Passed: approved all-targets cargo check; common tests with and without required CUDA; UMAP ANN/quality suites with and without required CUDA; full `cargo test -p fast-umap` with and without required CUDA. The baseline comparisons reach the embedding assertion, so standalone ANN IDs and distance bits match B0 on both backends.
- **Blocking:** the embedding-bit comparison fails on both backends. This is reproducible with the original pre-relocation executable itself, not just the relocated build. Original binary `target/debug/build/fast-umap/da5e0ca659936690/out/test_ann_bit_baseline-da5e0ca659936690 compare_cpu --ignored --exact --nocapture` produced first embedding word `0x4094487d` versus captured `0x4003fba4`; original required CUDA produced `0x41089952` versus `0x40c16beb`. Repeated relocated CPU runs also differ from each other. All use the same baseline-directory environment variable.
- Unchanged `layout.rs` explicitly promises quality rather than coordinate equality. Its atomic parallel SGD at 512 rows does not reproduce a captured embedding bitwise. Architect review `8-M2Review` confirms a gate-spec defect and requests a user decision under the approved stop-on-deviation rule. No acceptance threshold has been relaxed, no baseline recaptured, and M2 is not complete.
- **D15 approved by the user:** retain bitwise standalone ANN checks, additionally capture/check ANN with estimator parameters (`k=14`, 9 iterations at 512 rows), and use existing backend quality gates for the nondeterministic embedding. The original B0 files remain untouched. A pristine detached HEAD reference worktree at `target/m2-ann-baseline` provides the additional pre-relocation estimator-parameter capture. Those captures and comparisons are pending; no completion is claimed. Changing product layout semantics merely to chase one B0 sample is not justified. Pause critic `9-PauseCritic` confirmed the approval supersedes the human-only blocker, not the remaining M2 gates.
- The approved Python default-feature build exposed an existing gate prerequisite: required CUDA's GIL test calls `_timed_transform`, compiled only with `tracing`. The default build passed CPU (20 passed, 2 CUDA-only skips), but CUDA had 21 passed and that missing-method failure. Building with `--features tracing` then passed CPU (20 passed, 2 CUDA-only skips) and required CUDA (22 passed). The missing diagnostic was not hidden or skipped.
- Performance campaigns and M3–M6 correctness remain pending. Existing upstream Burn future-incompatibility warnings remain visible.

### M2 D15 gate result

The additional estimator-parameter captures passed on pristine HEAD `5a6fc72` on CPU and required CUDA; each captured ID and distance artifact is 28,672 bytes. All four comparisons (standalone B0 and estimator parameters, CPU and required CUDA) passed against relocated code. The original B0 artifacts were not overwritten. The full amended M2 gate then passed in order: all-targets cargo check; common tests ordinary and required CUDA; UMAP ANN/quality ordinary and required CUDA; four bitwise comparisons; full UMAP tests ordinary and required CUDA; tracing-enabled Python release build; CPU pytest (20 passed, two CUDA-only skips); required CUDA pytest (22 passed). Touched-package clippy with `-D warnings` and rustfmt check also passed. The tracing feature is necessary for the existing GIL diagnostic test, not for backend availability. Targeted-verification evidence is recorded in the session ledger; cumulative boundary review remains deferred until M6.

## M3 exact graph routing

- Added `MstGraph::{Auto, Complete, KnnGraph}`, `knn_k`, and `reproducible` configuration. Auto resolves above 30 features to KnnGraph; search-policy resolution runs only for Complete. Graph width follows upstream's `min(max(knn_k or max(3s,15), s), m-1)` rule. Some(0) is invalid even when the graph ignores the knob.
- KnnGraph uses exact common squared neighbors and host `knn_kruskal` at every shape in this milestone, including 4097 rows and widths 15/75. Public neighbors retain min_samples columns with padding. Diagnostic graph buffers use finite-row indexing, unlike the raw-indexed fitted outputs. The hidden exact comparator and graph/source accessors are available for M4.
- Checked graph ID/distance buffers, public neighbors, candidate tuples and sort workspace, including combined byte accounting. This is overflow/allocation-limit validation, not a physical-RAM guarantee. Common's documented `(n,+inf)` missing-neighbor sentinel alone is normalized to graph padding.
- Added independent-oracle CPU/CUDA tests for graph routing, overrides, sample-count/width boundaries, duplicates, disconnected components, overflow distances, small finite sets and remapping. Existing Complete blobs_d50 tests remain explicit Complete.
- Generated blobs_d50_knn from the identical blobs_d50 input. The pinned upstream graph's entire width is checked against exact nonself neighbors before upstream MST/extraction; provenance includes pynndescent and graph-source signatures/hashes. Regeneration of all 19 fixtures verifies. Host fixture oracle tests now select independent graph Kruskal for graph metadata instead of comparing its infinity bridges with a complete MST. No fixture assertion was weakened. Legacy blobs_d50 remains excluded from default-route parity and listed as a known upstream divergence; moons_leaf remains covered.
- Passed: all-targets checks for core/facade/Python extension; core tests with/without test-support; estimator tests (14) ordinary/required CUDA; full facade tests ordinary/required CUDA; fixture generator provenance, generation and --check; rustfmt and touched Python Ruff checks. Evidence: artifact83 (focused graph tests), artifact84 (generation), artifact88 (full Rust gates), artifact90 (final fixture/Python gates).
- Tracing-enabled release extension passed Python HDBSCAN CPU (48 passed, 12 skips: CUDA-only comparisons plus the non-CUDA-host test) and required CUDA (59 passed, one non-CUDA-host skip). The new default graph parity test executed and passed independently on both backends. Expected moons_leaf informational warnings remain visible.
- Extra strict all-targets clippy exposed three existing warnings: kdtree.rs constant chunks_exact and the alternate benchmark entry functions run_cpu/run_cuda being unused in the other benchmark binary (artifact89). These were not suppressed. They are outside the M3 acceptance commands; final integration must retain honest diagnostics. No M5/M4 performance or correctness is claimed here.

## M5 shared KD bounds

- Implemented non-strict atomic component bounds, reset to infinity immediately before every production traversal. Only candidate-kernel weights can lower a bound; kNN distances never seed it. Signed-zero canonicalisation uses integer bits, as established by M0. First-source ties, real infinite candidates and the shared `(inf, -1)` sentinel preserve host merge ordering. Brute-force and graph routes ignore the reproducibility flag.
- **D16, representation adaptation:** Burn 0.22.1 rejects mutable references in generated Dispatch/Fusion extensions. The public operation therefore takes `reproducible: bool`; private backend-local `KdBoundScratch` owns the unaliased allocation, length n or one dummy element. Its sole reset is in the backend wrapper, not the estimator. Pool allocation may reuse storage; explicit poison/reused-root tests also reuse the exact same allocation. Architect review found this preserves the accepted behaviour and section 6 gate; no threshold changed. Owned-tensor round-tripping was rejected because of Fusion aliasing risk.
- Production-path CPU/CUDA tests cover zero-poison reset, first-source component winners, bound equality with reported component minima, reused-root two-round reset with increasing outgoing weights, and length-one reproducible specialization. Shared non-winning row candidates may be heavier or missing and schedule-dependent; tests compare exact component winners, not all shared rows. Full fitted MST endpoint/weight-bit and output equality covers five fixtures, seeded tie sets and an Auto-KD-region dataset.
- **D17, conservative lint repair:** removed the unused alternate CPU/CUDA benchmark wrapper from each binary by sharing one entry function; backend-specific discovery now lives in the CUDA entry. Both tiny entry files consequently changed despite the original unchanged-entry expectation. Benchmark selection, timing configuration and CUDA requirement semantics are unchanged. Constant-size chunks use `as_chunks`; no warnings were suppressed.
- Passed final M5 sequence (artifact112): strict touched-package all-targets clippy, M0 atomic tests ordinary/required CUDA, all kernel unit tests ordinary/required CUDA, candidate/KD/estimator suites ordinary/required CUDA (18 estimator tests; two observational diagnostics ignored in normal runs), and rustfmt. Architect advisory review: CLEAR/APPROVE, no blocking findings. Shared-mode output documentation incorporates its clarification. The unused dummy still receives the one-word reset; no unmeasured optimization was applied.

### Observational timings

Both approved Criterion commands ran, with 100 samples and 95% confidence, on LLVM CPU and required CUDA (artifacts102/103). Table entries are measured median point estimates in milliseconds, not confidence bounds. These measurements preceded the semantically equivalent benchmark-entry and nested-condition lint repairs; they are not a claimed post-repair performance remeasurement. No default changed from performance evidence.

| Backend | Shape | s | Shared | Reproducible |
|---|---|---:|---:|---:|
| cpu | 1024×10 | 16 | 3.8451 | 4.2466 |
| cpu | 1024×10 | 32 | 3.2143 | 3.7188 |
| cpu | 1024×30 | 16 | 4.6678 | 4.9718 |
| cpu | 1024×30 | 32 | 4.8856 | 5.2566 |
| cpu | 2048×10 | 16 | 4.6660 | 6.1142 |
| cpu | 2048×10 | 32 | 4.6110 | 6.1690 |
| cpu | 2048×30 | 16 | 9.0866 | 11.0473 |
| cpu | 2048×30 | 32 | 9.6850 | 11.0374 |
| cuda | 1024×10 | 16 | 4.1181 | 4.5030 |
| cuda | 1024×10 | 32 | 3.7777 | 3.8321 |
| cuda | 1024×30 | 16 | 4.5708 | 4.5181 |
| cuda | 1024×30 | 32 | 5.1302 | 5.2352 |
| cuda | 2048×10 | 16 | 4.5799 | 4.7824 |
| cuda | 2048×10 | 32 | 4.9609 | 5.2306 |
| cuda | 2048×30 | 16 | 6.8499 | 6.8737 |
| cuda | 2048×30 | 32 | 7.6923 | 7.5395 |

### Observational traversal counters

The explicitly enabled tracing diagnostics ran separately on the existing `deterministic_rows` test dataset, **not** the timing harness's synthetic dataset. Counts sum only candidate traversals across all Borůvka rounds; neighbor-search counts are excluded. These are one observed interleaving per representative configuration, not deterministic shared-work promises. Both ignored measurement tests actually executed (artifact107); raw logs and machine-readable totals remain under `target/kd_bound_diagnostics*`.

| Backend | Shape | s | Reproducible | Rounds | Nodes popped | Leaf positions scanned |
|---|---|---:|---|---:|---:|---:|
| cpu | 1024×10 | 16 | false | 3 | 51220 | 1680000 |
| cpu | 1024×10 | 16 | true | 3 | 94186 | 3097664 |
| cpu | 1024×10 | 32 | false | 2 | 9326 | 296000 |
| cpu | 1024×10 | 32 | true | 2 | 63430 | 1980544 |
| cpu | 1024×30 | 16 | false | 1 | 2642 | 84352 |
| cpu | 1024×30 | 16 | true | 1 | 31744 | 1048576 |
| cpu | 1024×30 | 32 | false | 1 | 1570 | 51456 |
| cpu | 1024×30 | 32 | true | 1 | 31744 | 1048576 |
| cuda | 1024×10 | 16 | false | 3 | 65056 | 1835328 |
| cuda | 1024×10 | 16 | true | 3 | 94186 | 3097664 |
| cuda | 1024×10 | 32 | false | 2 | 23385 | 346304 |
| cuda | 1024×10 | 32 | true | 2 | 63430 | 1980544 |
| cuda | 1024×30 | 16 | false | 1 | 10514 | 122624 |
| cuda | 1024×30 | 16 | true | 1 | 31744 | 1048576 |
| cuda | 1024×30 | 32 | false | 1 | 10360 | 116992 |
| cuda | 1024×30 | 32 | true | 1 | 31744 | 1048576 |

## M4 ANN source and measured width gate

- Auto selects shared ANN only for finite-row count >4096 and effective graph width <=60; otherwise it selects exact neighbors. Forced `KnnSource::Ann` remains the campaign comparator independent of that production limit. Preflight errors are mapped explicitly; squared distances, seed 42 and rounded log2(row-count) iterations (minimum five) are used.
- **D18 (campaign wording reconciliation):** the approved M4 bullet says `k_graph = k`, but the established M3 width contract enforces `max(requested_knn_k, min_samples)`, capped at rows minus one. With min_samples=25, requested widths 15/45/60 therefore exercise effective widths **25/45/60**, not 15/45/60. Both values are retained in every record. No min_samples value or quality threshold was weakened.
- Release CPU, required CUDA and paired comparison passed (artifact139). Each backend has 18 cases: two datasets (6000×50 and 6000×40), three seeds, three requested widths. All six per-backend mean-recall gates passed; CUDA's lowest mean was 0.9998888888888889. Every same-backend repeat matched neighbor IDs, distance bits, labels and MST endpoint/weight bits. Per-case ARI was 1.0; maximum finite-MST-sum relative error was 0.000007070700677829524 on either backend. Bridges matched exact; paired CPU/CUDA ARI gates passed. **ANN_VALIDATED_MAX_WIDTH remains 60**, with no fallback needed.
- Raw JSON is under `crates/fast-hdbscan/.gjc/_session-01a169da-98fb-76d2-bad4-f28715c9381e/campaign/{cpu,cuda}.json`: Cargo runs integration tests with the crate directory as cwd, so the approved relative environment path resolves there. Each record includes labels, hashes, repeats, allocation accounting, one excluded warmup and three timed fits. Median fit times ranged 197.47–607.27 ms on CPU and 43.91–131.03 ms on CUDA. These are observational end-to-end fit times, not speedup or physical peak-memory claims.
- **D19 (conservative ownership repair):** the first M4 estimator run failed intermittently at `scratch.bounds.can_mut()` (artifact135). An isolated and full required-CUDA rerun passed (artifact136), but this was not treated as resolution. Architect review traced CubeCL 0.11.1 asynchronous fresh-allocation initialization: runtime-internal handle clones can temporarily exceed `can_mut`'s Arc-count threshold. The private per-call scratch has no caller-visible aliases. Removed the invalid runtime ownership assertion, retaining private structural ownership, dtype/length checks and the sole pre-traversal reset. No synchronization, retry, copy or execution fallback was added. Required-CUDA estimator and poisoned/reused-bound kernel tests passed after repair (artifacts138/144).
- The ordinary ARI known-pair test passed (artifact138). Strict touched-crate all-target clippy, ordinary and required-CUDA estimator suites (20 passed, two observational diagnostics ignored each), required-CUDA kernel tests (seven passed), and workspace format check passed (artifact144). Campaign lint repairs make nonfinite metric rejection explicit without relaxing gates. Final cumulative boundary review and integration remain M6 obligations.

## M6 Python plumbing and integration gates

- Native and wrapper validation accept `knn_k=None` or a positive integer, reject 0/-1/bool/float with ValueError, and forward `knn_k` and `reproducible`. The existing generic-kwargs native stub needed no signature change. Reproducible remains false by default; true/false bitwise tests now exercise actual complete-graph KD regions at 256×2, 512×12 and 2048×30 rather than only tiny fixtures.
- Release tracing extension build passed. Python HDBSCAN plus Standard UMAP suites passed on CPU (75 passed, 14 expected skips) and required CUDA (88 passed, one existing skip), artifact166. Both lanes emitted the existing informational moons_leaf upstream-divergence warning; it was not suppressed.
- The feature script, workspace all-target checks without defaults and with tracing, cubecl-ir-dump check, workspace clippy, required-CUDA full HDBSCAN/common suites and core test-support suite passed, artifact167. The script also ran backend-free core tests and ten Standard UMAP CPU integration suites. This is in addition to, not a replacement for, M2 bit-baseline and paired M4 evidence.
- **D20 (conservative gate repair):** global Ruff initially found pre-existing import/line-length failures in tracked `analyze/run_recovery.py` and formatting in Python code fences of `hdbscan_algo_impl.md`. Applied import sorting, removed unused sys, wrapped identical string values and ran the formatter; no algorithm or execution change. Compared parsed non-import AST against HEAD and required exact equality. The recovery program itself was not executed. No lint rules were weakened or files excluded. Repository runtime guidance also no longer incorrectly says there is no Python HDBSCAN API.
- Global Ruff check/format check, ty check, fixture regeneration check (19 cases) and workspace format check passed after repair. Fixture generation retains its expected float-overflow warning. Review/terminal receipts are separate durable-ledger obligations; test success does not substitute for those reviews.
