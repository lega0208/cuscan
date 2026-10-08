# Burn 0.22 — Findings for fast-umap

Reviewed: October 8, 2026. Primary source: [Burn 0.22.0 release blog](https://tracel.ai/blog/release-0.22.0/), published October 6, 2026.

## 1. Executive assessment

**The release is most useful to us as an explanation of the runtime and a toolbox for Stage 2, not as an upgrade that will automatically close the Python performance gap.** The current `Cargo.toml` already requests Burn/burn-cubecl 0.22.0 and CubeCL 0.11.0; `Cargo.lock` resolves those released versions. Dynamic tensors, backend extensions, runtime autodiff, and Burnpack persistence are already present in our source. Older repository guidance describing exact prerelease pins is stale relative to these manifests; the current version requirements are not exact `=` pins.

The [Stage 1 conclusion](artifacts/perf_optimization_stage1_conclusion_report.md) establishes the baseline:

- Standard, non-parametric CUDA UMAP uses the validated **uncached, 16-batch atomic layout**.
- At 5,000 samples × 32 features and 20 epochs, synchronized end-to-end fits take **0.828 s exact / 1.551 s approximate**, still about **16.1× / 22.0× slower** than parallel Python.
- Neighbor search now dominates the diagnostic GPU traces. Layout copies account for only about **0.067 ms** in those captures; copy removal is not a meaningful priority.
- Faster layout candidates failed frozen transform checks. Neither higher occupancy nor fewer launches is sufficient evidence for promotion.

The strongest follow-ups are to eliminate repeated neighbor/graph work, improve parallel distance evaluation, and use Burn's timing/memory/cache utilities to measure the result. Graph replay remains a conditional research item. Parametric neural training has separate opportunities, but was outside Stage 1.

### Relevance and priority

| Topic | Relevance to our work | Recommended priority |
|---|---|---|
| Custom-operation boundaries and CubeK tiles | Explains why enabling fusion alone does not optimize our handwritten search kernels; suggests reusable building blocks for cooperative distance kernels | High: Stage 2 kernel/algorithm investigation |
| Device timing and adaptive autotuning | Separate device work from host orchestration and startup probes; improve experimental attribution | High: measurement infrastructure |
| Adaptive memory pools and used/reserved reporting | Larger datasets, ANN scratch, parametric training, allocator behavior | High for scaling diagnostics; speed gains unproven |
| CubeCL Environment/cache bundles | Reproducible cold/warm measurements and reduced deployment startup | Medium; not a steady-state algorithmic speedup |
| CUDA/HIP graph replay | Repeated launch sequences, particularly layout epochs | Conditional; requires stable buffers and correct changing state |
| Backend-free public API and faster rebuilding | Shorter experiment iteration and simpler multi-backend applications | Already adopted; build-time benefit, not fit-time evidence |
| LoRA/QLoRA, parameter groups | Future adaptation of fitted parametric models | Low for current standard UMAP optimization |
| ONNX export and remote compute | Parametric inference deployment and access to other GPUs | Separate product/deployment work |

**Evidence convention:** release claims below are attributed to the blog or migration guide. Repository observations identify inspected source. Proposed experiments are hypotheses; no new performance measurements or code changes were made for this review.

## 2. Custom kernels, fusion, and compiler foundations

### 2.1 Backend extensions remain the right integration point

The blog describes execution as **Tensor → Bridge → Dispatch → Backend**. Application tensors/models no longer need backend type parameters, but backend implementations and custom operations still do. The bridge erases concrete representation dependencies; dispatch selects the backend at runtime.

The `#[backend_extension(...)]` macro connects extension traits to dispatch and, with output metadata such as `#[fusion(dtype = ..., shape = ...)]`, registers custom operations for lazy Fusion execution. A custom backward implementation remains possible.

**Important general explanation:** custom operations are fusion boundaries by default. Neighboring operations may fuse on either side, but fusion through the custom operation requires a dedicated fusion optimization. Output-shape metadata is not an instruction to merge or optimize its internal kernel launches.

Our [backend extensions](src/backend.rs) already follow this model:

- `UmapOps` supplies differentiable pairwise-distance/k-NN operations and a handwritten autodiff adapter.
- `StandardUmapOps` separately supplies forward-only bounded search, fuzzy graph preparation, and layout operations. Standard UMAP must not accidentally route through the differentiable full-pairwise path.
- Both CUDA and WGPU feature definitions enable `burn/fusion`, but the standard extension's Cube/Fusion registration is CUDA-gated. The blog's WGPU capabilities do not establish WGPU support or speedups for our standard estimator.

**Implication:** investigate the internals of `StandardUmapOps` implementations directly. Enabling fusion or declaring output metadata is not a substitute for reducing repeated scans, parallelizing a distance reduction, or changing scratch organization.

### 2.2 CubeK tiles and Pliron

The release explains CubeK's tile abstraction as reusable logic for loading, staging, computing, and storing data across layouts, precisions, and hardware. It is already used for tiled matrix multiplication, attention, and interpolation. CubeCL has moved compiler infrastructure to Pliron; its LLVM path now includes AMD and NVIDIA GPU targets in addition to CPU.

For our search kernels, cooperative feature reductions, reusable query/reference tiles, and improved memory access are plausible research directions. The blog does **not** provide a ready-made UMAP/top-k kernel or demonstrate automatic vectorization of our scalar loops. Compiler/LLVM availability also does not mean our current CUDA device automatically selects a different compilation path.

Any tiled or alternative distance formulation must preserve self exclusion, `(distance, index)` ordering, and quality gates. Changing float32 reduction order or using a matrix-product distance identity can alter near ties and suffer cancellation; benchmark numerical behavior as well as throughput. Avoid replacing bounded search with an unrestricted full N×N distance allocation.

### 2.3 Concrete Stage 2 targets visible in our source

These are implementation observations, not profiler-based attribution of each loop's cost:

| Source and observed pattern | Experiment motivated by the release/runtime discussion | Constraints |
|---|---|---|
| [`standard_umap/neighbors.rs`](src/standard_umap/neighbors.rs) dispatches query blocks of at most 64 rows. [`kernels/standard_umap/mod.rs`](src/kernels/standard_umap/mod.rs), `exact_neighbors_kernel`, assigns one execution unit per query row and serially scans reference rows/features with sorted top-k insertion; launch width is 32 | Test larger bounded blocks and cooperative/tiled distance evaluation. At a full 64-row block, the current launch has only two 32-unit cubes | Preserve bounded memory, tie ordering, aligned self exclusion, and both fit/query semantics; generated code and profiling must establish actual utilization |
| [`ann.rs`](src/kernels/standard_umap/ann.rs), `ann_step`, scans all source candidate lists to discover reverse joins for each target row | Build explicit reverse adjacency and compare its construction/storage cost against repeated scans | Current reverse-membership checks scale as O(N² × candidate width) per expansion step; preserve proposal semantics and recall |
| `ann.rs`, `consider`, computes feature distance before rejecting an already retained candidate | Test earlier duplicate rejection and/or reuse of candidate distances | Changing traversal or retained-list semantics can affect quality; measure rejection rate and net cost |
| [`graph.rs`](src/kernels/standard_umap/graph.rs), `smooth_row`, scans every N×k distance for each row to obtain global and row sums | Compute the global statistic once and row statistics separately | Preserve smooth-kNN/rho/sigma behavior, including the global-mean fallback; reduction-order changes require parity tests |
| [`query.rs`](src/kernels/standard_umap/query.rs), `approximate_query_kernel`, validates the entire reference graph per query and uses a one-unit cube | Validate once per operation and investigate cooperative query processing | Preserve invalid-index diagnostics, bounded candidate/visit capacities, seeded traversal, and frozen transform acceptance |

These repeated broad scans are better initial candidates than removing the already negligible layout copies. Pools, caches, and replay cannot remove their algorithmic work.

## 3. Timing, autotuning, and cache discipline

### 3.1 Device timing is useful, but lazy execution changes the window

The blog announces expanded Burn/CubeCL profiling, including CUDA/HIP device timing. The installed Burn 0.22 API provides:

| API | Meaning and caveat |
|---|---|
| `Device::sync()` | Waits for pending operations; returns an execution error on failure |
| `Device::flush()` | Dispatches queued work without waiting for completion; **not** a synchronization substitute |
| `Device::profile(...)` | Returns output plus a `ProfileDuration`; resolve the duration after execution |
| `Device::profile_with(...)` and `ProfileOptions::flush` | Can close a timing window over lazily buffered work |

The device documentation warns that previously pending work can fall inside a timing window, while work held back by fusion can fall outside it. A window over lazy operations alone can report no time. End with an appropriate read or use flush-enabled profiling, and account for prior pending work.

**Application:** add device-time diagnostics for exact search, ANN, graph preparation, and layout, alongside existing host spans. Keep synchronized wall-clock fit/transform timings as the product metric: device time does not capture all host preparation, dispatch, transfers, or embedding readback. Nsight remains necessary for kernel behavior, launch gaps, occupancy, and memory analysis.

### 3.2 Adaptive autotuning is not automatic tuning of every custom kernel

CubeCL's adaptive scheduler uses measured compute and memory throughput to estimate a performance target. It can stop when a candidate reaches that target, discard weak candidates early, and adapt measurement effort.

The root manifest enables Burn's `autotune` feature. However, our inspected neighbor/graph wrappers use explicit launch dimensions and do not themselves register candidate sets with an autotuner. The release does not establish that these handwritten choices will automatically be optimized.

**Application:** if we introduce kernel variants, explicitly determine how they enter tuning and how shape/hardware keys distinguish them. Treat initialization/throughput probes as tuning work, not UMAP work. This reinforces Stage 1's warning about throughput-test kernels in Nsight traces.

### 3.3 CubeCL Environment makes caches a controllable experimental variable

Compiled kernels and tuning results were cached before 0.22. The new Environment groups them in a named environment that can be warmed, bundled, and shipped to compatible targets. It also unifies runtime configuration/cache storage across native, WebAssembly, and `no_std` deployments.

Verified CubeCL 0.11 API entry points, re-exported under `cubecl`:

- `cubecl::environment::activate(name)` selects a named environment.
- `environment::set_root(path)` controls its storage location; `environment::load(file)` mounts a shipped environment database where supported.
- `environment::namespaces()` reports what has been warmed.
- `environment::bundle().save(path, format)` exports the active environment where native cache support is enabled.

This is a global namespace API, **not an `Environment` value passed to each fit**. Installed source warns that changing environments resets caches; on CUDA/HIP, compiled modules from old environments can remain resident because unloading queued code is unsafe. Switching per request can grow resident module memory. Select an environment at startup, not inside the fit/epoch loop.

**Application:** report three distinct conditions: cold process/cache, fresh process with prewarmed persistent caches, and warmed in-process fits. Record device, software versions, feature set, shapes, initialization, and cache identity. Treat compatibility and cache-hit behavior as things to verify, not promises that an arbitrary GPU can reuse a bundle. Cache preparation improves startup predictability; it does not eliminate steady-state neighbor-search computation.

## 4. Adaptive memory pools and scaling

CubeCL already reused memory through pools. New adaptive pools adjust page sizes as workload allocations change, release outdated empty pages, can relocate live allocations to release pages earlier, and separate frequently changing small allocations. The blog also describes workload allocation statistics collected during a dry run as part of this work's motivation; it does not supply a fast-umap dry-run integration recipe.

Burn's installed `Device` exposes:

- `memory_pool_usage()` and `memory_pool_report()` for allocator usage and dynamic-pool diagnostics; unsupported backends may return `None`.
- `memory_cleanup()` to request cleanup, without a guarantee that anything will be freed.
- `memory_persistent_allocations(input, func)` for persistent allocation mode; this is not by itself a proof of graph-replay safety.

**Relevant distinctions:** used memory is not reserved memory, allocator reports are not whole-process VRAM, and `client.empty(...)` is an allocation request rather than proof of a fresh physical device allocation. Likewise, `into_contiguous` may copy a noncontiguous input but need not copy one already contiguous.

Our ANN wrapper requests next index/distance buffers each iteration. Query and graph kernels also request scratch arrays. Measure pool reuse, peak live tensors, used/reserved memory, and process VRAM before proposing workspace reuse. A safe ping-pong workspace may help, but no benefit is established here. Preserve input ownership and avoid aliasing mutable state.

For larger datasets, record startup peak separately from warmed peak and compare mixed-size repeated workloads as well as fixed-size fits. The blog's memory charts exclude temporary startup bandwidth-probe allocations; those exclusions matter when comparing our deployment memory budget.

## 5. Graph capture/replay: available, but not a drop-in epoch wrapper

### 5.1 What the release promises

CUDA/HIP graph replay reuses a launch sequence with less repeated launch preparation. WGPU has a software graph retaining resolved pipelines/bindings. Replay attacks host/launch overhead, not the neighbor algorithm's complexity, irregular gathers, or layout's quality-sensitive update ordering.

Three distinct uses of “graph” must remain separate:

1. **Runtime execution capture/replay:** repeated device launches.
2. **Computational graph capture for ONNX export:** records model operations/weights for export.
3. **Remote fusion graph replay:** sends a repeated operation graph once and replays it by server-side ID.

None is the UMAP fuzzy neighbor graph.

### 5.2 Verified high-level Burn replay contract

The installed `burn::tensor::capture` creates a `Graph`; its `replay()` method is **unsafe**. It reuses the exact device buffers touched during capture. Callers must keep all captured tensors live through execution, prevent concurrent use, and refresh/read buffers on the capture stream. `Graph::is_hardware()` distinguishes a hardware graph from closure re-execution.

The implementation performs **three warmup closure executions before capture**, despite its introductory documentation summarizing warmup more loosely. Warmup therefore has real effects on an in-place stateful algorithm. Unsupported or failed hardware capture can fall back to re-running the closure; merely using the API does not establish hardware replay or a speedup. The blog's WGPU software-graph support should not be equated with `is_hardware() == true` on this high-level hardware-oriented API.

### 5.3 Why our layout needs a designed workspace

[`estimator.rs::optimize`](src/standard_umap/estimator.rs) passes host `epoch` and `alpha` arguments into a custom operation and evolves embedding, clocks, and RNG state. The CUDA kernel updates operands in place. A recorded launch sequence will not rerun Rust code to recompute host scalar arguments on every hardware replay.

A valid experiment must define:

- Stable storage for embeddings, clocks, RNG, graph/schedule inputs, outputs, and scratch; keep mutable buffers unaliased.
- How epoch/learning-rate state changes between replays: device-resident mutable state, explicitly updated graph parameters where supported, or another verified design.
- How warmup/capture mutations are reset so measured training executes the intended schedule.
- Stream ordering, memory-lifetime guarantees, and immutable transform reference points.
- Capture/setup cost and amortization, plus explicit detection of real replay versus fallback execution.

Stage 1 already found replay support but did not establish a safe fixed workspace or credible paired gains. Preserve that negative/unfinished evidence. Prioritize neighbor/graph work; revisit replay only after timing demonstrates meaningful launch overhead. Retain the validated 16-batch order and all fit/transform gates when comparing replay to normal execution.

## 6. Parametric UMAP and deployment findings

### 6.1 Faster rebuilds and runtime autodiff

Removing backend generics reduced the blog's median release rebuilds from **28.42 to 4.57 s** for its CNN and **14.73 to 1.00 s** for its transformer. These are model-edit rebuild measurements, not clean-build or UMAP execution speedups.

We already use `Tensor<2>`, non-generic neural modules, and `Device` selection. Parametric fitting enables `.autodiff()`; inference uses `model.valid()` and `.without_autodiff()`. The blog emphasizes that enabling autodiff context does not make every tensor require gradients; runtime checks replace some former compile-time preconditions. Keep custom backward tests and inference/training separation when tuning this path.

### 6.2 Fusion and neural performance claims need workload matching

The release improves tensor-layout handling to avoid unnecessary copies and allow more fusion after an unfusable operation. That may benefit parametric training's ordinary tensor expressions around custom operations. Its convolution gradients, transposed-convolution kernels, and dedicated BatchNorm backward are not direct matches for our Linear/ReLU network.

The blog reports about **80% higher CNN training throughput**, warmed CNN VRAM **956 → 486 MiB**, and transformer VRAM **3,486 → 2,868 MiB**. These were FP32 CUDA training benchmarks on an **RTX 4050 Laptop GPU**, three runs/version, **50 warmup + 500 measured steps**. They are not predictions for standard UMAP, our RTX 3080, or our parametric architecture. Since this checkout already uses 0.22, do not count release-level improvements as additional future gains.

### 6.3 LoRA, QLoRA, and parameter groups

`Module::apply_lora(Lora::new(rank, alpha).set_param_group(...))` freezes base floating parameters and attaches adapters to selected two-dimensional weights. `ParamGroup` selection, `unfreeze_group`, and grouped optimizer/scheduler configurations permit targeted fine-tuning without replacing the forward model.

This could support adapting a fitted **parametric** embedding network to new data while training fewer parameters. It does nothing for standard UMAP's coordinate SGD and is not automatically faster for a small MLP trained from scratch.

QLoRA reduces frozen-weight storage, but the blog explicitly warns that training still constructs dense effective weights, may retain them for input gradients, and computes full weight-shaped gradients before propagation to adapters. Packed weights plus adapters are not an adequate peak-memory estimate. FP8/software conversion support also does not imply accelerated arithmetic for every operation/hardware. Precision experiments must test embedding quality and gradients, not only storage size.

### 6.4 ONNX export

`burn_onnx::export::OnnxExporter::new().export(&model, input, Model::forward)?.save(...)` records a forward pass and lowers its graph/weights to ONNX. Export coverage is early and limited; Burn support or successful ONNX import does not establish exportability.

Potential use: deploy the fitted parametric Linear/ReLU model through another inference runtime and benchmark transform latency/throughput. Verify operators, input-shape behavior, and same-input numerical parity. This does not export standard UMAP's search/layout estimator as a neural model, and it is not runtime launch replay.

### 6.5 Remote compute

Burn Remote now primarily uses authenticated/encrypted Iroh connections, preferring direct connections with relay support. A server can expose multiple devices. Fusion can send a repeated operation graph once for server-side replay; tensor transfers between Iroh servers can avoid a client round trip. WebSocket remains available for trusted networks.

The blog illustrates `RemoteHost::iroh(...).with_credential(...)`, `Device::remote_options(&host).init()?`, and transferring an inference model to the remote device. This could support parametric training/inference on other GPUs, but network traffic and synchronization can dominate small fits. Our custom backend extensions are not automatically proven serializable/executable remotely; verify server registration and operation transport before treating standard UMAP as supported.

## 7. Persistence, features, and toolchain

The release's migration guide requires **Rust 1.95**. Burn itself has no default execution backend; the application must enable relevant constructors. Our root feature defaults remain `cuda,cpu,verbose`, while `--no-default-features` retains the explicitly enabled Flex backend. Python bindings depend on the root library, so backend defaults are inherited rather than independently selected in the binding manifest.

With Burn defaults disabled, optimizers require the `optim` feature; our `parametric` feature already enables `burn/optim`. The guide also notes opt-in tracing, extraction of linear algebra to the `linalg` extension, Candle removal, and NdArray/LibTorch deprecation. Flex remains our real CPU training backend; the blog does not demonstrate a Flex UMAP speedup.

Burnpack is now the common checkpoint format for models, optimizers, and schedulers. `model.save_file(path)` and `model.try_load_file(path)` replace recorder/precision-setting arguments; the latter handles loading errors. Our [`parametric_umap/serialize.rs`](src/parametric_umap/serialize.rs) already uses these APIs.

Legacy MessagePack/binary/JSON recorder files need genuine conversion, not renaming. Model-parameter conversion does not automatically preserve legacy optimizer/scheduler state. This is deployment/checkpoint information, not a recommendation to add compatibility layers to our current persistence implementation.

Dataset access and more high-level training APIs are now fallible, and training outcomes separate failures from user interruption. Our library owns a synchronous custom training loop rather than Burn's high-level learner; trace actual consumers before changing cancellation/error behavior based on learner APIs.

## 8. Recommended experimental sequence and promotion rules

1. **Preserve the baseline:** unchanged validated 16-batch layout, current authoritative Stage 1 measurements, and failed-candidate artifacts.
2. **Attribute Stage 2 costs:** neighbor/search/graph device windows plus synchronized end-to-end fits; separate host preparation, reads, launch overhead, and cold-cache/tuning activity.
3. **Remove repeated work first:** ANN reverse joins/duplicate distance work, graph global-statistic rescans, and per-query graph validation. Compare algorithmic savings against extra scratch/construction launches.
4. **Increase useful search parallelism:** bounded exact-search block sizing and cooperative/tiled feature work. Measure generated code, memory traffic, and occupancy; do not assume occupancy alone predicts speed.
5. **Measure memory and startup separately:** used/reserved pools, process VRAM, startup probes, cache state, and larger representative datasets. Consider cache bundles for deployment, not as a claim of faster warm kernels.
6. **Revisit replay conditionally:** only with verified stable-buffer/state handling and measured launch overhead large enough to justify it. Run parametric/deployment investigations as separate workloads.

For promotion, retain the hard fit and frozen-transform trustworthiness comparisons, approximate neighbor recall, self/duplicate/tie handling, error paths, and immutable transform reference points. Stage 1's primary quality comparison uses a 0.02 trustworthiness gap limit; approximate recall must remain above the existing 90% gate. Keep approximate baseline tracking as its agreed warning rather than silently converting it into a hard gate. Require repeatable paired end-to-end improvements in both measured modes, not just a faster kernel or favorable rerun.

No release claim establishes that these changes will close the present Python gap. Larger datasets and matched fit/transform workloads are necessary to establish where CUDA is actually beneficial.

## 9. Sources and verification boundaries

### Primary and supplementary upstream references

- [Burn 0.22 release blog](https://tracel.ai/blog/release-0.22.0/): primary release claims, explanations, benchmark conditions, and examples.
- [Burn 0.22 migration guide](https://burn.dev/books/burn/migrating-to-0.22.html): toolchain, features, autodiff, fallibility, and checkpoint caveats. This live guide may evolve independently of the release.
- [Burn v0.22.0 release notes](https://github.com/tracel-ai/burn/releases/tag/v0.22.0): supplementary change inventory.
- [Burn tensor 0.22.0 device source](https://docs.rs/crate/burn-tensor/0.22.0/source/src/device.rs): timing, synchronization, memory, and dtype capability APIs.
- [Burn tensor 0.22.0 execution graph source](https://docs.rs/crate/burn-tensor/0.22.0/source/src/tensor/api/graph.rs): capture implementation and unsafe replay contract.
- [CubeCL Environment 0.11.0 source](https://docs.rs/crate/cubecl-environment/0.11.0/source/src/environment.rs): named/global environment semantics, switching warnings, and bundles.

API details were checked against the locally installed released crate sources corresponding to the versioned source links above; no API examples were compiled. The blog links to evolving upstream examples/PRs, which are useful follow-up references but are not proof that an API is present in a particular installed version.

### Repository evidence

- [Stage 1 conclusion](artifacts/perf_optimization_stage1_conclusion_report.md), [final measurements](artifacts/stage1-restored-control-20.json), and [experiment manifest](artifacts/layout-experiments/manifest.json). Numbers in this document are taken from the conclusion report, not remeasured or independently recomputed from those artifacts.
- [Root manifest](Cargo.toml), [lockfile](Cargo.lock), and [Python binding manifest](bindings/python/Cargo.toml).
- [Backend extensions](src/backend.rs), [exact-search orchestration](src/standard_umap/neighbors.rs), [exact CUDA kernels](src/kernels/standard_umap/mod.rs), [ANN](src/kernels/standard_umap/ann.rs), [query search](src/kernels/standard_umap/query.rs), [graph preparation](src/kernels/standard_umap/graph.rs), and [estimator/epoch orchestration](src/standard_umap/estimator.rs).
- [Parametric estimator](src/parametric_umap/mod.rs) and [persistence](src/parametric_umap/serialize.rs).

This is a documentation/research review only. No product source, dependency, or benchmark configuration was modified; no new CUDA, Rust, Python, export, or remote-execution tests were run. Suggested speedups, capture eligibility, cache portability, and deployment support remain experiments to verify.
