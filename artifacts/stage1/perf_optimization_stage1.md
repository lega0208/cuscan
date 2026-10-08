# CUDA Standard UMAP Stage 1 Performance Optimization

**Status: consensus approved; pending separate execution approval.** Planning only: no product edits, tests, formatters, builds, profilers or performance experiments were run. Locked IDs: `artifact:plan`, `surface:cuda-standard-umap`, `integration:nsight`, `constraint:planning-only`.

## Summary

Plan race-safe parallel layout for CUDA standard non-parametric UMAP, then consider orchestration candidates only after layout and only on measured benefit. Preserve the confirmed workload, exact/approx modes, Python baseline, speed and quality limits, regression cases, profiling requirements, timeout behavior, and separate execution approval. Include `examples/profile_standard_cuda.rs`: the comparison script invokes it and it currently hardcodes Exact.

Evidence: `src/kernels/standard_umap/layout.rs` currently executes one work item serially per epoch; per-head RNG state is three words and shared by edges with the same head. `src/standard_umap/estimator.rs` supports Exact/Approximate/Auto; native Approximate uses `StandardUmapOps::approximate_neighbors` with `max(round(log2(n)),5)` iterations. Fit/transform use unaliased mutable buffers and host materialization. The comparison script fixes exact search and seeded single-thread Python, limits rows to 16–4095, and measures one fit after warmup under 115 seconds; its spans are host timings, not GPU duration. The CUDA example fixes Exact/seed 42 and warms before tracing. Existing fixture quality uses mean-gap 0.02 and recall 0.90; fit recall is non-self width 14 and query recall width 15.

Prior actual CUDA runs on RTX 3080 are known. The WSL Microsoft display adapter is not evidence CUDA is unavailable; recheck current CUDA device/driver and Nsight readiness at approved preflight. No valid repeated 5,000×32/10 comparison against parallel Python is established by the existing harness. Historical results are not predictions; claim no unmeasured gain.

## Intent Reconciliation

No material open items. The user clarified that **every one of the five approximate runs must individually reach recall ≥0.90 for both Python and CUDA**, at non-self width 14. Mean or pooled recall cannot substitute for this gate.

The original specification's workload, parallel Python baseline, independent exact/approximate speed gates, quality/regression limits, Nsight requirements, conditional orchestration, five-minute combined deadline and user-directed timeout recovery are unchanged. Layout scheduling, diagnostic graph capture and internal backend integration are engineering choices, not new user requirements. No Flex/WGPU/parametric optimization or unconditional general-purpose diagnostic API is added. Execution remains separately pending approval.

Consensus evidence lives under `.gjc/_session-01a1174e-e58b-77be-bd90-fb689a3d3422/plans/ralplan/01a1174e-e58b-77be-bd90-fb689a3d3422/`:

- `stage-05-revision.md`: SHA-256 `1cdddc047de0ab121688534d8d8d916dba3eb80263e3f387dd655767fe73c559`.
- `stage-05-architect.md`: **CLEAR / APPROVE**, reviewing that revision.
- `stage-05-critic.md`: **OKAY**, reviewing the same revision and counter-reviewing the Architect.
- `stage-05-post-interview.md`: **reconciled-clean** against the original specification and `stage-02-intent.md`.

## Top Decision Drivers

1. Race-free ownership and correctness for fit and bipartite transform.
2. Fair end-to-end timing including neighbor generation and host readback.
3. Repeated unprofiled medians decide performance; diagnostic evidence cannot expand scope.

## Principles

- CUDA coordinates may be nondeterministic; quality is required, deterministic output is not.
- No data races or concurrent non-atomic embedding read/write; preserve buffer ownership.
- No serial compatibility fallback. Infeasible design or failed gate triggers review/revision.
- Evaluate orchestration candidates separately after layout; retain only demonstrated benefit.

## ADR — Parallel Layout and RNG

**Decision:** Use an atomics-free vertex-owner gather as the design to validate. Once per fit/transform, before building head CSR and explicit reverse/tail CSR, compact to `weight > 0` edges, retain each edge's original ID for clock/mask indexing, and preserve stable relative ordering within each head so per-head RNG consumes events in the existing order. Do not infer or rely on fuzzy-graph symmetry. Use one lane per vertex, retain existing per-head RNG, and have each lane sequentially apply that vertex's positive/negative events while other vertices are read from an immutable old-epoch embedding. Write each vertex only to its uniquely owned next-epoch row. For fit (`move_other=true`, shared embeddings), gather both outgoing-head and incoming-tail incidences so tail-side positive contributions are applied too. Negative events update heads only. For transform, update query/head rows only and read fitted training tails directly as fixed, immutable data.

Clock safety is a separate ordered phase, not a concurrent read convention: one lane owns each edge's clocks, computes an immutable per-epoch positive-fire/negative-count mask, then the vertex gather consumes that mask after phase completion. Preserve the current clock recurrence, dtype and arithmetic/rounding (including f32 recurrence where used), conditional negative-count/advance order, and unique edge writer. No gather lane reads a clock while it is being mutated. Each head owner consumes its current per-head RNG sequentially for that head's firing edges. Preserve self-feedback by using the lane's evolving local row against immutable old other rows; when a negative sample names the owner itself, use its current local row (self-distance remains zero). In fit, write next rows to a second embedding buffer and swap after each epoch; in transform, retain the fitted tail buffer unchanged and do not swap it.

**Drivers:** Unique state ownership; preserve within-row SGD feedback where feasible; avoid depending on unverified floating atomics; keep stage scope bounded.

**Alternatives considered:**

- **A — Vertex-owner gather, chosen:** No atomics, per-head RNG reuse, self-row feedback and unique writes. Costs head/reverse CSR and masks; may suffer from only `n` lanes, hub imbalance and stale other-row reads.
- **B — Snapshot plus atomic reduction:** Broad edge parallelism, but adds scratch/atomics/launches and stale-self quality risk. Proceed only if atomic support and quality are demonstrated.
- **C — Edge-color batches plus reduction:** May reduce positive contention, but requires coloring/barriers and still handles negative events. Also atomic-dependent unless a safe owner reduction is provided.

B/C are revision choices, not runtime fallback paths.

**Why chosen:** Existing per-head ownership already matches negative updates; the serial kernel moves tail only for positive `move_other`, while arbitrary negative endpoints are read but not written. Vertex ownership avoids racy writes without needing graph symmetry or new per-edge RNG. It is a proposal to validate, not a performance claim.

**Consequences:** Other-row values are epoch-snapshot reads, while each owner sees its own sequential updates; outputs may differ from serial order. CSR/mask preparation adds work and achievable CUDA occupancy is uncertain. Quality and end-to-end performance remain gates.

**Follow-ups/checkpoints:** Confirm CubeCL/Burn support for required indexing, gathers, buffer swap and ordered phase dispatch. Validate compaction preserves original edge IDs and stable per-head order, scratch bounds and exact clock recurrence. Assess per-owner degree/max-degree imbalance and small-case latency as an early viability screen, not a new acceptance gate, plus quality before the full benchmark. If unsafe/unsupported or quality fails, stop for design review; do not ship serial fallback.

### Statistics race checkpoint

The current kernel increments shared positive/negative `stats` counters. Trace actual consumers before parallelization. If unused, remove the obsolete shared writes/tensor as a correctness cleanup without claiming a speed benefit. If consumed, use uniquely owned per-vertex counts with a separate reduction or another proven race-free scheme. Never leave concurrent shared increments.

## Orchestration Options

**A, chosen:** After layout, test buffer reuse, stats removal (if not resolved as a required correctness fix), and CUDA graph replay independently. Keep a candidate only when repeated paired post-layout medians improve **both** exact and approximate modes versus matched post-layout control, beyond observed paired-run variation, with quality and small-case gates intact. No clear benefit or incomplete evidence means discard. Verify pinned dependency support before graph replay.

**B:** Bundle candidates with layout. Rejected because it hides attribution and contradicts measurement-led scope.

## Scope and File-Level Changes

In scope: CUDA standard UMAP; identical float32 5,000×32 input, 10 epochs, Euclidean, 15 neighbors, 2 components, random initialization; explicit exact/approximate modes; parallel Python; small regressions 512×16/30 and 1024×32/100; existing fixture quality/recall; Nsight Systems and bounded Nsight Compute.

Out of scope: Flex/WGPU/parametric optimization, estimator defaults/cutoff, deterministic CUDA output, CPU fallback, widening neighbor/graph kernels, or changing a gate/workload/deadline to pass.

| Target | Planned responsibility |
|---|---|
| **Coherent six-file backend integration unit:** `src/backend.rs`, `src/flex_backend.rs`, `src/kernels/mod.rs`, `src/standard_umap/layout.rs`, `src/kernels/standard_umap/layout.rs`, `src/standard_umap/estimator.rs` | Change the `StandardUmapOps::optimize_epoch` signature and all adapters/call sites atomically for CSR/mask inputs and phase outputs. Update `backend.rs` Fusion metadata with the interface change. Implement CUDA phases and preserve valid Flex compilation/semantics. This is an explicit six-source-file logical integration unit because the interface cannot be changed consistently in fewer; do not add a prepare/mask API solely to meet a file-count limit. |
| `tests/test_standard_layout.rs`, `tests/test_standard_quality.rs` | Focused ownership/clock/finite-output coverage and required quality gates; do not assert coordinate equality. |
| `src/standard_umap/estimator.rs`, `examples/profile_standard_cuda.rs`, `scripts/compare_umap_spans.py` | Narrow diagnostic path and harness: feature-gated, `#[doc(hidden)]` public read-only accessor on `FittedUMAP` for actual `fit_graph` IDs. The stored graph is `Tensor<2, Int>` on device; read back after the fit timer stops. Reuse non-default `tracing` feature if suitable; no estimator behavior change or unconditional general-purpose API. |
| `perf_optimization_stage1.md` | This consensus plan; implementation remains pending separate approval. |

**Delegation boundary:** Each delegated task must name and touch no more than five explicit files. The six-file interface change is one coherent integration unit; establish its interface contract before sequencing dependent tasks, and have the parent verify the joined six-file union once. Do not split the interface into inconsistent interim product states or create an extra API just to satisfy the per-task bound.

**Recall evidence checkpoint:** Before accepting recall, establish that the narrow accessor observes each actual native fit's neighbor IDs at non-self width 14, without changing estimator behavior. Python recall likewise uses each actual fitted estimator's graph. Compare to exact neighbors with matching configuration; never substitute a separate helper/search configuration. If the feature-gated accessor proves infeasible, pause for reviewer decision before claiming Q-03.

## Sequencing and Bounded Slices

The six-file backend interface integration is the explicit atomic-unit exception to any whole-unit five-file count; individual delegated tasks remain bounded to at most five files.

1. **Preflight after execution approval:** Recheck CUDA device/driver, Nsight versions/permissions, Python `umap-learn==0.5.12`, CPU/thread configuration. Prior RTX 3080 runs are known; current WSL display-adapter wording does not imply CUDA failure. Build release CUDA/tracing companion separately from runtime budget.
2. **Harness first:** Script + example; one warmup and five timed runs per implementation/mode. Python uses `random_state=None`, all available threads (`n_jobs=-1`); exact Python builds `precomputed_knn` inside each timed fit; approximate explicitly sets `force_approximation_algorithm=True`. CUDA selects Exact or native Approximate, not Auto; report native fit width 14 and iteration rule. Identical input/configuration; preserve warmup before tracing.
3. **Pre-layout baseline:** Capture valid exact/approx primary distributions, matched CUDA small-case baselines, quality/recall and bounded profiler diagnostics. Incomplete runs are not baselines or passes.
4. **Layout integration and early exit:** Agree the six-file interface contract, then coordinate the coherent backend integration unit with dependent estimator/harness tasks. Compact positive-weight CSR and reverse CSR once per fit/transform, preserving original IDs/stable per-head ordering. Before the full post-change suite, verify required backend capabilities, degree/max-degree imbalance and small-case latency screen; then run the focused CUDA quality exit: existing fixture Q-01 plus a representative primary trustworthiness spot-check in each search mode against the corresponding Python mean. The small-case screen is an early viability signal, not a replacement for H-03; the trustworthiness spot-check is an early stop/revise screen, not a replacement for Q-02's five-run gate. Failure or unsupported required primitive means revise/stop for review before spending the full suite. The parent verifies the joined six-file integration once.
5. **Full acceptance:** Required-CUDA fixture plus five primary runs/mode/implementation and small regressions. Capture actual-fit graphs outside fit timing. Every approximate run for both implementations must pass per-run recall; every CUDA embedding must pass its per-run comparison to the Python mode mean.
6. **Conditional orchestration:** Only after layout passes, test each candidate separately. Require complete paired evidence in both modes and all quality/regression gates intact. No mandatory runtime artifacts beyond concise evidence needed to substantiate gates.
7. **Handoff:** This root plan is delivered after consensus. Execution still requires separate approval.

## Acceptance Criteria

- **Harness H-01:** Exact and Approximate separately forced for identical primary input/settings. Exact Python neighbor generation is within timed fit; Python Approximate explicitly forces the algorithm (`force_approximation_algorithm=True`). CUDA Approximate uses native `StandardUmapOps` and its iteration rule; no Auto cutoff or invented external feature requirement.
- **Harness H-02:** One warmup plus five timed fits per implementation/mode; report samples and medians. Include neighbor generation and host embedding readback; exclude input generation, profiler overhead and quality calculation from each fit duration. Python baseline is unseeded parallel/all threads. Record effective configs/thread/tool versions; do not double-count nested host spans.
- **Profiling P-01:** Systems diagnoses timeline, launches, synchronization/utilization; Compute inspects one bounded representative vertex/event kernel. Host enqueue spans are not GPU runtime.
- **Layout L-01 / RNG R-01:** Unique per-edge clock/mask writer, per-head RNG owner, sequential own-row updates, immutable old-other reads, unique next-row writes; explicit incoming-positive handling, heads-only negatives, fit buffer swap and transform fixed tails. Nondeterminism allowed; no CUDA deterministic-output requirement.
- **Quality Q-01:** Existing required-CUDA fixture fit and transform trustworthiness absolute mean gaps each ≤0.02.
- **Quality Q-02:** Each of five CUDA primary embeddings per mode is within 0.02 trustworthiness of the same-mode five-Python-run mean.
- **Quality Q-03:** Every one of five approximate fit runs for **each** Python and CUDA has recall ≥0.90 against exact neighbors at non-self width 14, using actual fit neighbor IDs and matching configuration. No pooled/mean substitute.
- **Quality Q-04:** Existing transform-query recall remains width 15 and ≥0.90; distinguish from fit width 14.
- **Regression H-03:** Each small CUDA median ≤1.10× matched pre-layout median.
- **Performance H-04:** CUDA end-to-end median ≤0.90× parallel Python median separately in exact and approximate. Miss means report/follow-up, not expanded neighbor/graph scope.
- **Orchestration O-01:** Candidate retained only with repeated paired post-layout median improvement in both modes beyond observed variation and all quality/regression gates intact; no gain or incomplete evidence means discard.
- **Timeout T-01:** One 300-second combined benchmark/profile budget includes warmups, repetitions, quality/recall diagnostics and profiler overhead; build separate. On expiry stop, mark incomplete/no-pass, ask user to change parameters or extend timeout with recommendation. No automatic adjustment.

## Verification and Profiling Templates

No verification commands were run. Suite CLI options and example profiler arguments below are planned interfaces, **not current CLI claims**:

```sh
cargo build --release --no-default-features --features cuda,tracing --example profile_standard_cuda
uv run --no-sync python scripts/compare_umap_spans.py --suite stage1 --runs 5 --warmups 1 --deadline-seconds 300 --profile-nsys --profile-ncu --output /tmp/stage1-pre.json
FAST_UMAP_REQUIRE_CUDA=1 cargo test --no-default-features --features cuda --test test_standard_quality -- --nocapture
uv run --no-sync python scripts/compare_umap_spans.py --suite stage1 --runs 5 --warmups 1 --deadline-seconds 300 --baseline /tmp/stage1-pre.json --profile-nsys --profile-ncu --output /tmp/stage1-post.json
```

Profiler switches vary by installed version; these are verification-needed templates, no execution authorized. Check installed help before use. Launch actual profiler directly on the built application, not Cargo/shell wrapper; enforce the shared deadline and retain bounded kernel filtering:

```sh
nsys profile --trace=cuda,nvtx,osrt --sample=none --force-overwrite=true --output=/tmp/stage1-nsys target/release/examples/profile_standard_cuda INPUT 5000 32 10 TRACE --search approximate --profile-only
ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*layout_vertex_gather_kernel.*' --launch-count 1 --set basic --force-overwrite --export /tmp/stage1-ncu target/release/examples/profile_standard_cuda INPUT 5000 32 10 TRACE --search approximate --profile-only
```

Confirm the generated kernel name from Systems; if the filter matches none, stop and refine it, never remove it/profile all kernels. Profiler values diagnose only, not acceptance fit timing.

## Escalation / Risk Gate

Stop for missing current CUDA/profiler access, unsupported CSR/mask/phase capability, unsafe clock/embedding/stat handling, infeasible diagnostic accessor, incomplete baseline, quality failure, or timeout. Do not widen scope or alter gates. A five-minute timeout yields incomplete evidence and a user choice between parameter change and timeout extension. Retain only temporary, minimal measurements and actual-fit neighbor evidence required by gates.
