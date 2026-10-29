# Phase B: bounded leaf-coherent CPU HDBSCAN prediction queries

Status: implementation plan; no product implementation authorized by this document.
Revision: 3. Architect APPROVE; critic APPROVE / OK. No outstanding review issues.

## 1. Objective and acceptance boundary

Improve CPU `approximate_predict` locality by scheduling spatially nearby queries
consecutively over Phase A's accepted leaf-SoA candidate pack. Treat cold training-core
queries and external queries as independent optimizations, then measure their combination.
The accepted Phase A source, not the pre-SoA release, is the performance and correctness baseline.

This is the standalone successor to the deferred Phase B in `approx_query_cpu_plan.md`.
Do not reopen Phase A or count its measured speedups as Phase B gains. The Phase A ledger
has a final aggregate completion receipt for G004 in
`.gjc/_session-01a187c9-c215-72be-bcc9-0bf0adce9aa6/ultragoal/goals.json`.
That supersedes the original plan header's stale reference to pending completion receipts;
this plan does not mutate that historical ledger.

Deliverables, if implementation is separately authorized:

1. Bounded CPU core gathering/scattering in leaf-position order.
2. Bounded CPU external-query ordering, gathering and output restoration.
3. Checked allocation/live-set planning for every added buffer, including predictor conversion.
4. Exact Phase A equivalence tests, CPU/CUDA integration, and reproducible ablation evidence.
5. A judgment-based retain/reject decision for each independent arm; no mandatory performance win.

Non-goals:
- No CUDA query reordering, CUDA kernel changes, fit kernel changes, or F2 reuse for queries.
- No new public option, `HdbscanOps` signature, device row-map operand or serialization field.
- No persistent device references, persistent query permutations, or new cached tree metadata.
- No global external-query sort, unbounded all-reference launch, scaling, approximate search,
  candidate pruning change, arithmetic change, or execution-error fallback.
- No promise to eliminate Phase A's first-launch costs; attribution requires measurement.

## 2. Evidence and current source contracts

Read these symbols again before implementation; line numbers are deliberately not API contracts.

| Source | Relevant contract |
| --- | --- |
| `crates/fast-hdbscan/src/prediction.rs` | `compute_cores` currently slices `call.data` in compact-ID order, searches inclusive-self at width `s`, and appends f64 core distances. |
| same | `approximate_predict` validates/flat-packs queries, obtains the host tree, uploads one call-local reference, computes missing cores under the initialization lock, then processes bounded external batches at `candidate_width(s,n) = 2*s`, checked to be at most `n_finite`. |
| same | `CallReference::query` returns candidate IDs and squared distances in query-buffer row order. Both prediction phases pass `exclude_self = false`; offsets do not affect this mode. |
| same | `Target::query_plan`, `plan_batch_rows`, `plan_prediction`, `plan_core_preparation`, `reference_device_bytes`, and checked allocation helpers own preflight and live-set accounting. |
| `crates/fast-hdbscan/src/kernels/kdtree.rs` | `kd_query_plan` selects CPU LeafSoa, otherwise PaddedRows. `launch_kd_query` passes `leaf_order = false`; each row is independent and outputs are indexed by query row. Fit's `leaf_order = true` is a separate launch contract. |
| `crates/fast-hdbscan-core/src/kdtree.rs` | `KdTree` has implicit-heap children, `indices`, node ranges/leaf flags and bounds. It stores **no split axis or split threshold**. `indices` maps leaf positions to compact finite reference IDs. |
| `crates/fast-hdbscan-core/src/prediction.rs` | `prediction_core_from_squared_rows` computes rank `s-1`; `PredictionHierarchy::predict_squared_batch` appends `PredictionColumns` with public column vectors. Connecting points are already raw training IDs. |

`hdbscan_perf_report.md`, section “CPU approximate-prediction leaf-SoA (Phase A)”,
provides motivation, not proof of ordering benefit: warm search improved on most shapes,
2D was roughly neutral, and cold prediction regressed in 8/16 measured cells. The first
20k×2 core batch was much slower than later batches, but JIT was not causally isolated.
Extra host gathering can erase locality gains. Phase B therefore starts as an ablation,
not an assumed production improvement.

## 3. Architecture: reorder host batches, not the device API

Keep the existing independent per-query kernel unchanged. Feed it reordered compact query
buffers, then restore host results. A packed query row is an execution position, never a
reference ID. Candidate IDs retain the reference's compact identity throughout.

Use a private `QueryOrder { Input, Leaf }` parameter on internal core/external routines.
During experimentation production callers pass two temporary private constants,
`CORE_ORDER` and `EXTERNAL_ORDER`; planning receives the same values. Build four source
variants differing only in those constants, preserving every diff and binary hash.
The example and integration tests are separate crates: do not expose private controls via
public/hidden-public APIs, CLI routing flags or environment hooks. In-crate unit tests can
force all four parameter combinations in one binary. Remove experimental constants and
rejected production paths at finalization; retain only test-scoped forced alternatives.
The final profiler identifies retained routing through its provenance manifest, not a new API.

Production CPU eligibility is derived from the already-resolved target's LeafSoa query
plan, not from the user's requested Auto value. CUDA remains input-order/PaddedRows.
A rejected arm leaves its accepted Phase A behavior intact; this is an evidence-backed
implementation choice, not an execution-time fallback.

Resolve the two effective orders once alongside `Target::query_plan` in preflight, forcing
both to Input for non-LeafSoa targets. Thread those exact values into planning, conversion
preparation and execution; do not independently rediscover ordering in `compute_cores`.
Use the same resolver in the conversion-only preflight.

### 3.1 Cold training-core arm

Change the CPU branch of `compute_cores` to also receive the canonical host tree:

1. Keep `s == 1` returning canonical positive-zero cores without gathers or searches.
2. Fallibly reserve a checked length-`n` f64 destination, indexed by compact reference ID,
   and initialize it to NaN so an incomplete scatter cannot look like a valid zero core.
3. Iterate **leaf positions**, `start..end`, using the already-preflighted `core_batch`.
4. For each position `p`, let `id = tree.indices[p]`; copy the exact f32 feature row
   `state.geometry[id*d..(id+1)*d]` into one contiguous host batch.
5. Upload `[end-start, d]`, then call the existing query operation with width `s`,
   `exclude_self = false`, `query_offset = 0`. Never pass a leaf position as a self ID.
6. Preserve existing candidate-ID validation and `prediction_core_from_squared_rows`.
   Scatter `batch_cores[j]` to `cores[tree.indices[start+j]]`.
7. Reject any remaining NaN with `HdbscanError::Numerical` before hierarchy validation,
   then validate the complete core vector before publication, retaining the current lock,
   double-check and failure semantics. No partly filled vector enters the cache.

The CUDA branch keeps its current contiguous device slice and append behavior.
A small private helper may share upload/gather logic with the external arm, but should
not hide row-identity or buffer-lifetime rules behind a generic scheduling framework.

Pass the host tree through **every** `compute_cores` caller, including predictor conversion
(preparing cores for portable predictors). Audit these callers with content search rather
than assuming `approximate_predict` is the only path. Loaded predictors with ready cores
must not recompute them. `s == 1` and no-cluster-tree shortcuts remain cheap.

This deliberately adds CPU core host copy/upload work instead of introducing a gather
kernel, device permutation or full-reference reordered copy. Measure it explicitly; reject
the arm if cold latency does not justify it.

### 3.2 External-query arm: leaf-affinity buckets within each existing batch

Keep batch boundaries in original query order. For each validated batch of `b` rows:

1. Compute a deterministic leaf-affinity key for every original batch-local row.
2. Store one `Vec<(usize, usize)>` of `(leaf_start, original_batch_row)` pairs.
3. `sort_unstable_by_key` on the complete pair. Original row is an explicit unique tie-break,
   so no stable-sort scratch allocation or dependence on sort stability is needed.
4. Gather exact f32 row values in pair order into the existing batch upload buffer.
5. Search with width `k`, `exclude_self = false`, `query_offset = 0`, unchanged arithmetic.
6. Run hierarchy math into a temporary `PredictionColumns` in packed order.
7. Restore each triple to `columns[start + original_batch_row]`. Scatter predictions, not
   the much larger candidate matrices. Do not remap candidate IDs or connecting-point IDs.
8. Drop tuple storage, batch prediction columns and readbacks before the next batch.

For the CPU reordered branch, allocate final columns at checked length `m` once; use
fallible reservation and initialize storage before indexed scatter (probabilities to NaN).
Verify all batch lengths before writing. Each index is filled exactly once by construction
of the permutation. Reject remaining NaNs as an internal numerical error before returning.
The final published result contains no unfilled entries. The nonreordered branch can keep
its current append path; both must preserve existing error contracts.

A batch-local order limits sorting/memory to O(b), preserves public batching limits and
avoids storing an O(m) inverse map. It sacrifices cross-batch locality intentionally.
A global sort is a separate future proposal, not an implementation escape hatch.

### 3.3 Deterministic leaf affinity without changing `KdTree`

Do not pretend the tree contains split planes. Use its existing child bounds:

- Begin at the nonempty root. At each internal node inspect the two nonempty children
  `2*node+1` and `2*node+2`; choose the smaller squared box lower bound.
- Compute the key-only lower bound in host f64, feature order, converting f32 values
  before subtraction. This avoids f32 overflow on large finite queries. Choose the left
  child on equality. Builder trees guarantee both children exist and are nonempty;
  debug-assert that invariant instead of adding unreachable one-child recovery.
- Stop at a leaf and return its `starts[node]` leaf-position key. Bound descent by
  `tree.n_levels`. Use builder/import-validated trees; add no runtime whole-tree scan.
- This is **greedy leaf affinity**, not reconstruction of the original median partition,
  a nearest-leaf proof, or a search filter. Queries outside all boxes remain valid.
- Actual neighbor search still starts at the root and considers the same candidates with
  the same pruning/ties. Key arithmetic cannot change a neighbor distance.

Implement this private host helper in facade `prediction.rs`: it is prediction scheduling,
not a new general-purpose public core API. O(b*d*levels) key work and O(b log b) tuple
sorting are explicit costs. Degenerate/overlapping bounds may place every query in one
bucket and provide little benefit; ties preserve input order.

## 4. Numerical and identity invariants

Phase B is stricter than Phase A's old scalar-versus-vector comparison: it changes only
row scheduling under the **same** SoA arithmetic. After restoration require bitwise equal
CPU candidate squared distances, identical ordered candidate IDs, f64 core/probability
bits, labels and connectors against Phase A on the same device/build.

- Quantized and ordinary non-quantized inputs both require equality. Do not excuse
  differences as FMA drift: no arithmetic change is planned.
- Keep the same k, feature order, candidate pack, ID tie-break and inclusive-self rule.
- Query-buffer row, leaf position, compact reference ID and raw training ID are distinct.
  Only compact ID indexes cores; only original query row indexes public output.
- Equal feature rows are different observations. Duplicate queries still produce separate
  correctly positioned results. Duplicate training rows must not trigger value-based exclusion.
- Arbitrary scheduling must not change overflow/sentinel handling, error kinds or
  success/failure outcomes. Reordering may change which failing row is reported first;
  do not require identical row-specific error messages.
- CUDA low-level self-exclusion/offset semantics and fit leaf-order semantics are unchanged.
  Do not generalize this prediction-only gather path to `exclude_self = true`.

Any equivalence failure blocks performance acceptance until its cause is fixed. A change
that needs altered arithmetic or kernel scheduling contracts requires revising this plan.

## 5. Checked allocation and live-buffer accounting

Extend the existing helpers; do not create a competing planner. `PredictionPlan` keeps
its public fields. Resolve the same CPU ordering decisions for planning and execution.
Preflight before tree build/upload/core initialization, including `prediction_plan()` and
predictor conversion. Use checked multiplication/addition, `checked_allocation_bytes`,
`checked_live_sum`, tensor pitch/alignment checks and typed errors as already established.

Let Bc/Bq be preflighted batch sizes, d feature count, n reference rows, R existing steady
device-reference bytes, Pc/Pq existing `kd_query_pack_bytes`, and T(b,d) the checked,
pitch-aware compact query tensor allocation. Do not substitute logical bytes for T.

### Device live sets

- Preparation remains `reference_prepare_bytes`.
- CPU reordered core: **R + T(Bc,d) + Pc**, replacing R + Pc. The old compact reference
  `call.data` still lives inside R; the gathered source is additional, not an alias.
- External query: R + T(Bq,d) + Pq, unchanged; reordered gathering replaces the existing
  compact query upload, not a second simultaneous source.
- CUDA/core-input arm remains R + Pc. Apply added per-tensor page/alignment checks,
  not only total byte checks. Extend batch feasibility if new tensor constraints require it.
- Ensure materialized source queries remain counted until fused execution/readback finishes.

### Host live sets

Use current input/output/tree, upload-clone, core-publication and readback terms. Add:

- CPU reordered core: one gathered `Bc*d*size_of::<f32>()` buffer and explicitly
  `Bc*size_of::<f64>()` batch cores, alongside the length-n destination, its published-copy
  allowance and candidate readbacks. Include these terms in both preflights.
  Reuse `tree.indices[start..end]`: no new O(n) ID/permutation array.
- External reordered: `Bq*size_of::<(usize,usize)>()` tuples plus
  **`2*Bq*(size_of::<i32>()*2 + size_of::<f64>())`** for packed output columns and
  `predict_squared_batch`'s internal transactional `local` columns, which overlap on append.
  Also count its `k*size_of::<f64>()` per-row distance scratch. The gathered query copy
  replaces the existing Bq*d f32 copy. Final m-row outputs remain in base terms.
- The nonreordered external arm also needs one Bq-row internal `local` output payload
  and k f64 scratch in its host total. Correct these common accounting omissions for every
  arm rather than making the baseline look artificially smaller. Do not change core math.
- Sort is in-place unstable with a total key; no heap scratch array. Account host tuple
  size on the actual target, not an assumed 32-bit index size.
- Account simultaneously live TensorData/readback conversion copies conservatively using
  existing conventions. Make actual drop points agree with the accounting table.
- Predictor conversion's `plan_core_preparation` must include gathered host/device sources
  and batch f64 results even though it returns only a batch count, not `PredictionPlan`.
- Preserve the documented host-plan boundary (owned call buffers/cache tree, not arbitrary
  caller allocations or allocator-reserved/JIT memory); report planned peaks, not measured RSS.

Use fallible reservation for new host buffers and existing typed allocation errors. Overflow,
per-allocation page violations and aggregate live-set overflow must fail before cold work.
Do not add a user memory-budget option. Keep bounded batching rather than choosing an
all-row launch to evade scatter logic.

## 6. Implementation sequence and concrete file ownership

### Step 0 — freeze baseline and instrument the experiment

- Verify Phase A completion and preserve a tracing release baseline binary from the current
  accepted source. Record revision, full source diff/hash, compiler/dependencies, CPU/client
  vector widths, runtime/JIT-cache policy and profiler binary SHA-256.
- Extend `crates/fast-hdbscan/examples/profile_hdbscan_prediction.rs` JSON outputs for
  exact comparison (including probability bits). Arm selection is compile-time private
  constants in four builds, recorded by the experiment manifest; no profiler routing API.
- Add tracing spans for affinity-key generation, sorting, gathering/upload and output scatter;
  retain core/external search/readback/math separation. End-to-end clocks must include all
  these costs. Preserve the existing baseline path during experimentation.

### Step 1 — implement bounded host ordering helpers and their unit tests

- `crates/fast-hdbscan/src/prediction.rs`: private affinity, tuple ordering, checked row-gather
  and output-scatter helpers. Keep functions small and identities explicit.
- Unit-test keys/permutations/gathers/scatters without starting a device. No new tree fields.

### Step 2 — integrate core and external arms plus planning atomically

- Same source: CPU core leaf-position batches and f64 scatter; external batch-local order
  and prediction scatter; update all core-preparation callers and typed preflight.
- Add private explicit order parameters and temporary build constants as in §3; no
  persistent state, public controls or schema changes.
- Update plan unit tests to mirror precise live sets and allocation-limit branches.
- Keep `backend.rs`, `kernels/kdtree.rs` and `kernels/mod.rs` unchanged unless a genuine
  source-discovered defect requires a separately reviewed plan revision.

### Step 3 — correctness and integration

- In-crate `prediction.rs` tests force all four arms, row-identity and batch-tail comparisons
  through private parameters in one binary; use fresh runtimes for cold core comparisons.
- `crates/fast-hdbscan/tests/test_prediction.rs`: public row-identity, batching and CPU/CUDA
  checks under the compiled production routing, without reaching private order controls.
- `crates/fast-hdbscan/tests/test_predictor_state.rs`: conversion and loaded-ready predictors.
- Existing Python prediction/predictor tests remain public-contract coverage; add a focused
  regression in the existing matching test file only if Rust coverage misses binding behavior.
- No serialization changes are expected; verify portable state remains identical under core ordering.

### Step 4 — ablate, select, and remove experimental production paths

Run §8 only after §7 correctness. Retain independently justified arms; remove rejected
production code and experiment-only controls. Preserve experiments as artifacts, not dead
runtime aliases. Repeat gates on the exact final retained source and binary.

### Step 5 — document accepted routing or rejection

- Update this file's implementation outcome and link artifacts without rewriting approval history.
- Add a Phase B section to `hdbscan_perf_report.md`, distinguishing key/sort/gather/scatter
  costs, cold/warm outcomes, host/device peaks and independent arm decisions.
- Update the `AGENTS.md` architecture paragraph and README prediction guidance only for
  actual retained behavior. No new public configuration or numerical caveat is expected.
- Leave Phase A's implementation/evidence intact; its deferred Phase B section can link here.

## 7. Correctness matrix and verification commands

### Host helper and planner tests

- One-leaf trees, multiple levels, duplicate/all-equal points, overlapping bounds, keys tied
  across children, queries outside boxes, huge finite f32 magnitudes, signed zero.
- Deterministic total pair order; every original row appears once; gather preserves f32 bits;
  scatter restores deliberately shuffled unique triples and duplicates without changing IDs.
- Batches 1, 63, 64, 65, 1024 and tail splits such as 1024+1024+5. In-crate unit tests use
  private explicit batch-limit parameters on internal planning/execution helpers; production
  wrappers always pass the existing 1024 cap. Exercise complete private prediction flow
  with small limits, without a public/hidden-public knob or integration-crate hook.
- Exact added host/device bytes, pitch/alignment, tuple width, conversion-only preflight,
  `s == 1`, overflow, one-row feasibility and aggregate live-set overflow. Assert rejection
  occurs before upload/core counters change, preserving failure and lock-retry semantics.

### CPU equivalence and public integration

In in-crate tests, compare all four arms with input/input Phase A behavior on the same
initialized reference for candidate searches and fresh cold runtimes for core publication:
- Quantized and splitmix/non-quantized inputs; d in {2,8,16,30,33}; small/nonmultiple-of-64
  row counts and query count greater than reference count. Include repeated rows and exact ties.
- Compare candidate IDs and distance bits after inverse mapping in internal device tests,
  not merely final labels (which can hide wrong core/candidate mappings).
- Cold core values compared in compact-ID order; external labels/connectors/probability bits
  in input order. Test full rows and forced batch tails for both s=1 and ordinary s.
- Mixed nonfinite training rows validate compact-to-raw connector mapping. Invalid query
  validation remains unchanged; no-cluster-tree path performs no ordering/upload/core work.
- Concurrent calls/cache status, injected core failure/retry, predictor conversion/export and
  import with ready cores retain existing behavior. No reordered buffers become persistent.
- CUDA strict tests retain PaddedRows and input order. CPU-versus-CUDA tests keep their
  existing tolerance/fixture contract; do not require cross-backend bitwise equality.

Commands for separately authorized implementation (not executed during plan drafting):

```sh
cargo test -p fast-hdbscan-core
cargo test -p fast-hdbscan
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan --test test_prediction -- --nocapture
FAST_UMAP_REQUIRE_CUDA=1 cargo test -p fast-hdbscan kd_query -- --nocapture
cargo clippy --workspace --all-targets
cargo fmt --all -- --check
uv run maturin develop --release
uv run --no-sync pytest python/tests/test_hdbscan*prediction*.py python/tests/test_hdbscan_predictor*.py
bash analyze_hdbscan_stages.sh --rows 20000
bash analyze_hdbscan_stages.sh --rows 100000
```

Run focused new unit tests first. Run formatting once after the union of source changes,
then the checks above. If Python files change, also run Ruff and ty for the affected scope.
Required stage analyses use real strict CPU/CUDA execution, not compilation or skipped
hardware tests. Kernels/fit are intentionally unchanged, but the shared crate and profiler
still require final integration evidence. Record baseline failures separately, without hiding
new failures behind historical mentions of fixture, timing or Ruff problems.

## 8. Performance experiment and retention gate

### Arms and workloads

Four experimental builds differing only in the two private order constants:
A=input core/input external; C=leaf core/input external; Q=input core/leaf external;
CQ=both. Use identical toolchain, RUSTFLAGS, features and lockfile; retain source diff/hash
and binary SHA-256 for each variant. Cross-check A against the frozen Phase A binary as
a prerequisite to attributing any gain. Alternate each variant and A binaries within the
same measurement session, not one historical baseline run against later candidate runs.
The frozen-vs-A parity check covers output bits and timing, not equal host-plan peaks:
the shared internal-column/scratch accounting correction intentionally raises A's planned
host peak. Report that delta as corrected accounting, not new physical allocation.

Primary matrix matches Phase A: n in {20k,100k}, d in {2,8,16,33}, m in {1k,n},
min_samples=5, identical stored float32 eight-cluster inputs and hashes. Add the existing
300k×16/10k workload as a scaling case. Add shuffled uniform data and heavily overlapping
or duplicate clusters at representative 100k×{2,16} shapes to expose useless bucketing
and unfavorable core coherence; report those separately from the historical matrix.

Run sequentially, without builds/tests competing for CPU. Alternate arm process order,
with at least three independent processes per arm/cell and three warm calls per process;
use the same fixed inputs and fit seed. Cold is first prediction after fit; warm has cached
cores. Do not call process warm calls independent statistical trials. Repeat marginal or
regressing cells with at least seven alternating process pairs, as in Phase A.

Report medians-of-process-medians, individual process samples, absolute deltas and ratios.
Record JIT-cache policy; first-call is not necessarily cache-cold. Do not clear shared caches
silently. A separate controlled cache experiment is needed before attributing time to JIT.

### Metrics and decisions

For every arm/cell retain:
- Cold/warm end-to-end time; cold core search and core gather/upload; external search,
  key construction, sort, gather/upload, hierarchy math, scatter and readback spans.
- Planned host and device peaks and batch sizes. Logical plans are not RSS measurements.
- Exact output and core comparisons, including probability bits; no silent tolerance changes.
- Full binary/source/input provenance and raw traces, not just a summary speedup.

C must justify **cold** end-to-end cost; warm calls do not run this arm. Q must justify
**warm** end-to-end time including sorting/restoration and disclose its cold impact.
CQ must demonstrate no adverse interaction, not inherit independent gains by assumption.

Retain an arm only when repeatable end-to-end improvements across relevant workloads
justify copy/sort overhead and any regressions. Search-only gains are insufficient. Use
judgment, not a fabricated universal percentage threshold: report both absolute and
relative regressions, repeat noisy cases, and write an explicit accept/reject rationale.
An unresolved material regression blocks unconditional retention. Default remedy is to
reject the arm, not add speculative shape gates. Any evidence-backed pre-launch gate
requires a documented amendment and another architect/critic review before finalization.
CUDA behavior and fit must not materially regress. No performance claim without artifacts.

Store under `artifacts/approx_query_cpu_phaseB/`:
`provenance.json`, `inputs/`, `baseline/`, `arms/{A,C,Q,CQ}/`, `summary.json`,
`repeats/`, `correctness.json`, `final_gates/`, and a reproducible summary script.
Record final source/binary hashes separately from experimental hashes. After selection,
repeat correctness and representative performance on final retained source; if dispatch or
accounting changed in cleanup, rerun the full affected matrix. Both arms may be rejected:
a documented rejection with preserved evidence is a valid experiment outcome, not a
claim that a speedup was delivered.

## 9. Review and completion criteria

Planning completion requires independent architect and critic review of this exact file,
with blockers incorporated and both returning OK/APPROVE on the revised plan. Record
review iterations below. Plan approval is not authorization to edit product code.

Implementation completion, separately authorized, requires exact correctness, typed
preflight tests, final CPU/CUDA/Python verification, final-source performance evidence,
experimental-path cleanup, documentation and explicit independent arm decisions.
No commits, pushes, worktree changes or implementation occur as part of this planning request.

### Planning review record

- Revision 1 architect (`7-PhaseBArchitect`): REVISE. Resolved unimplementable private
  profiler controls with explicit private parameters/four build variants; moved forced-arm
  tests in-crate; included both transactional output copies and f64 math scratch in host
  planning. Also added NaN scatter-hole checks, precise error equivalence and builder-only
  descent invariants.
- Revision 2 architect (`7-PhaseBArchitect`, pass 2): APPROVE, no blockers.
- Critic (`6-PhaseBCritic`, pass 1): APPROVE / OKAY, no required changes. Optional
  clarifications covered NaN fill, in-crate batch-limit access, checked candidate width and
  shared order resolution. Revision 3 incorporates these and the architect's accounting-parity
  caveat.
- Revision 3 architect (`7-PhaseBArchitect`, pass 3): APPROVE / CLEAR; no blockers or
  remaining caveats. Receipt: `agent://7-PhaseBArchitect`.
- Revision 3 critic (`6-PhaseBCritic`, pass 2): APPROVE / OK; no blockers, optional
  suggestions or required changes. Receipt: `agent://6-PhaseBCritic`.
- After these final approvals, only status/review receipts changed. Technical plan unchanged;
  implementation remains separately authorized work.

## 10. Decisions and rationale

1. **Host gather/scatter instead of a device permutation API.** The existing per-query SoA
   kernel already provides the required arithmetic and independence. Additional kernel/fusion
   contracts are unnecessary for the smallest end-to-end experiment.
2. **Core batches cover leaf positions, scatter to compact IDs.** This uses the known reference
   order without launching all rows or confusing a leaf position with observation identity.
3. **External order stays batch-local.** Bounded memory and existing batching contracts matter
   more than speculative cross-batch reuse. A global sort is explicitly outside this plan.
4. **Greedy child-box affinity, not invented split metadata.** Existing trees lack split planes;
   their bounds provide deterministic spatial buckets without cache/schema changes.
5. **Restore final predictions, not candidate matrices.** Copying three output columns costs
   less memory than materializing inverse-permuted k-wide candidate matrices.
6. **Bitwise same-backend equivalence.** Scheduling is the only intended change; Phase A's
   scalar/vector arithmetic exception does not authorize drift here.
7. **Account extra core sources and host tuples explicitly.** Packing is not free, and the
   existing reference stays live. Conversion must satisfy the same checked resource contract.
8. **Independent arm retention with end-to-end gates.** Cold cores and warm external queries
   have different economics; either or both can be rejected without undoing Phase A.
