# Stage 2 research panel — topics

## Purpose and boundaries

Chair: main agent. Members: architect (`3-PanelArchitect`) and critic (`4-PanelCritic`). Prepare questions, directions and decision criteria **before implementation planning**. No product changes or new performance experiments in this panel.

Focus: standard, non-parametric UMAP neighbor search and graph preparation; useful GPU parallelism and memory locality through Burn 0.22 / CubeCL 0.11. Preserve the validated Stage 1 16-batch atomic layout and fit/frozen-transform quality contracts. **Parametric/neural implementations are explicitly excluded by the user: ignore them entirely, including follow-up topics.** Other standard-UMAP backends are considered only to identify contracts and scope, not to claim automatic portability or speedups.

Evidence seeds: [release findings](burn_release_22_findings.md), [Burn book findings](references/burn_book_findings.md), [CubeCL book findings](references/cubecl_book_findings.md), [Stage 1 conclusion](artifacts/perf_optimization_stage1_conclusion_report.md).

## Topic register

| ID | Topic | Status | Questions |
|---|---|---|---|
| T1 | Neighbor-search mathematical parallelism and quality | Discussed | Exact pair/feature opportunities and achievable work/span derived; fixed-key ANN selection can reorder, capped query traversal cannot. Numerical admission remains F1. |
| T2 | Cooperative distance/selection and useful cache locality | Discussed | Bounded tiles, coalesced/vector loads, shared reuse and partial top-k are hypotheses; no kernel configuration selected. Hardware caches, compilation caches and pools distinguished. |
| T3 | Remove redundant work; respect Burn/CubeCL execution boundaries | Discussed | Complete reverse adjacency, duplicate rejection, global-statistic sharing and validation hoisting identified; ordered stages and ownership preserved. Attribution remains F2/F3. |
| T4 | Measurement interpretation, goals and pre-planning readiness | Discussed | Utilization is diagnostic, not an objective or bottleneck proof. Existing quality/product gates retained; missing capture and workload evidence recorded as dependencies. |
| T5 | Standard-UMAP backend and search-path scope | Discussed; added during panel | CUDA/Flex ANN proposals differ; compare invariant contracts, not assumed identical iterations. CUDA is foreground; backend-boundary choices remain F4. |

## Current queue

**No pending discussion topics.** Both members completed live peer exchange and assented to the chair's corrected F5–F7 resolutions; neither identified another independent topic. This is **not an implementation task list**; no implementation plan or experiment has been started. Missing empirical evidence is recorded as a research dependency, not endlessly requeued as a new discussion.

### Continuing topic register

| ID | Topic | Status / trigger | Questions and required evidence |
|---|---|---|---|
| F1 | Numerical equivalence and algorithm-change admission | Discussed | E/N/A research classes recommended; differential baselines required. Same-loop arithmetic and final-shortlist rescoring do not prove exact parity. No new tolerance or hard warning gate admitted. |
| F2 | Capture provenance and transform attribution | Discussed | Harness defaults target layout and fit only; user-capture provenance and transform attribution remain explicit measurement dependencies, not inferred bottlenecks. |
| F3 | Workload envelope, memory budget and crossover goals | Discussed | Sparse proposed ladder, explicit logical byte formulas and index limits recorded; no deployment target, universal VRAM cap or new crossover inferred. |
| F4 | Standard backend boundaries and graph validation ownership | Discussed | Preserve CUDA proposals and public positive/invalid stats. Per-primitive validation sharing recommended; fitted-lifetime trust needs proof. Standard persistence premise withdrawn. |
| F5 | Skew-aware complete reverse joins and streamed selection | Discussed; added dynamically | Complete reverse indices and streamed distinct top-W merges recommended as research directions; all fixed-size tiles visited, no degree/source caps. Counts, skew, peak workspace and prefix limits explicit. |
| F6 | Cooperative primitives and specialization limits | Discussed; added dynamically | Installed plane/shared/barrier APIs verified at source. Conditional storage handoff is not grid rendezvous/progress proof; no compilation, variant set or reusable CubeK top-k claimed. |
| F7 | Host ingress/egress and synchronization inventory | Discussed; added from member exchange | Fit/transform host boundaries recorded, Random/Spectral and product/diagnostic separated. Source-visible packing duplication is not measured redundant PCIe transfer; frozen layout untouched. |

### Evidence dependencies — not queued discussions

- Capture provenance, metric definitions and search/transform-specific traces are missing; utilization percentages alone do not identify a bottleneck (F2/F7).
- Deployment N/D/k/Q, latency objectives and usable memory/runtime budgets are unspecified; workload ladder and byte formulas are proposals/calculations, not a measured crossover (F3).
- Old/new differential IDs, distance bits, positive/invalid stats and graph diagnostics on matched builds/devices remain required for an equivalence claim. Extreme finite inputs and shortlist completeness need characterization; no N/A admission exists (F1/F4).
- Degree/duplicate distributions, bounded workspace, generated kernels/resources and synchronization/progress evidence remain unmeasured. Source API availability does not prove compilation or performance (F5/F6).
- Actual copy elision, readback latency and stable CSR/RNG/error-boundary equivalence need attribution/proof before choosing any host-boundary change (F7).

## Agenda disposition

- Numerical hazards from T2 were consolidated into T1's quality contract and F1, not treated as an unapproved precision optimization.
- Transform profiling and autotune-probe discipline raised by the critic were merged into T4/F2 rather than duplicate topics.
- Backend semantic divergence became T5 and F4 so the search discussion did not drift into backend unification.
- Layout scheduling/replay redesign, automatic cutoff recalibration and backend enablement are not selected work. Revisit only with relevant evidence and a separately scoped discussion.
- Parametric/neural material encountered in the seed references was excluded, not retained as a future topic.

Panel outcome: T1–T5 and F1–F7 discussed with architect/critic/chair challenges and reconciliation. See the corresponding sections in [stage2_panel_discussions.md](stage2_panel_discussions.md). The substantive discussion queue is exhausted; empirical unknowns remain explicit. No speedups were measured or claimed, and no implementation is approved by these records.

## Management rules

Each discussion records description, member positions, exchange/challenges, outcome, evidence and open questions in [stage2_panel_discussions.md](stage2_panel_discussions.md). Add substantive new topics here rather than letting an active discussion drift. Keep concluded topics for traceability; merge duplicates explicitly. A discussed topic may retain unanswered empirical questions. Research conclusions are not implementation approval or speedup claims.
