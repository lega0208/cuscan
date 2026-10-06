# ParametricUMAP Python parity report

## Verdict and scope

**Full algorithmic and mathematical parity is not present.** The primary Rust implementation now has density-adaptive fuzzy memberships, fuzzy symmetrization, self-inclusive neighborhoods, and integer-weighted positive sampling. Reference-labelled fixtures cover graph-helper mathematics on identical neighbor lists and float32 pruning/counts. Cross-entropy, attraction/repulsion aggregation, negative sampling, finite-pass training schedules, and optimization remain different. Matching graph mathematics does not make the objectives equivalent.

Original audited checkout: Rust `fast-umap` 1.6.0, HEAD `6b68d5e59cbc2eac86c0cc5081c23e2cef5517ce`; installed `umap-learn` **0.5.12**. Original numeric source citations below describe that audit revision. The original audit did not change source; subsequent approved primary-route remediations are identified explicitly below. Unrelated findings remain historical audit findings, not newly reverified parity claims.

Primary Rust route: `Umap::fit` / `fit_with_signal` / `fit_with_progress` → `train_sparse` (`src/lib.rs:272-383,389-465`). The deprecated `UMAP` route and feature-gated `CpuUmap` are separately identified below; their behavior must not be attributed to the primary route.

### Reference-file shorthand

All Python shorthand below resolves to these exact installed files:

- **P** = `.venv/lib/python3.14/site-packages/umap/parametric_umap.py`
- **U** = `.venv/lib/python3.14/site-packages/umap/umap_.py`
- **D** = `.venv/lib/python3.14/site-packages/umap/distances.py`
- **UT** = `.venv/lib/python3.14/site-packages/umap/utils.py`

Inherited graph construction must be included: `ParametricUMAP` subclasses `UMAP` and delegates fitting to it (`P:45,98,194-206`). Comparing only the neural-network file would miss the central UMAP mathematics.

Dependency citations use **BN** = `/home/marc/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-nn-0.22.0-pre.4/src/modules/linear.rs`, and **BO** = `/home/marc/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/burn-optim-0.22.0-pre.4/src/optim/adam.rs`. These identify the actual pinned Burn implementation, rather than inferring defaults from Rust call sites.

## Evidence and verification

This is a source-level algorithm audit with targeted numerical checks, **not an end-to-end embedding-quality benchmark**:

1. Three independent read-only reviews covered graph mathematics, training, and model/API capabilities; their findings were consolidated and checked against the current sources.
2. `cargo test --no-default-features --test test_core_python_prereqs`: **4 tests passed**. These exercise the current Flex CPU fit/transform, reproducibility, and input/configuration behavior; they do not assert Python parity.
3. The installed Python `convert_distance_to_log_probability`, `compute_cross_entropy`, and `get_graph_elements` functions were extracted using Python AST and executed with NumPy equivalents of the required Keras operations. Their algebra and weighted edge repetition were checked by assertions. No replacement training implementation was used.
4. The actual current Rust `fit_ab` function was extracted, compiled with `rustc`, and run; its results were compared with the AST-executed installed Python `find_ab_params` using SciPy's `curve_fit`.

Subsequent graph/count remediation verification: 6 Rust sparse helper tests, 7 with
`nn-descent` (including its actual 10K-sample preparation route), and 6 with `pca`.
The CPU API suite passed all nine tests: eight primary API/configuration regressions
and one seeded dense-trainer control-isolation regression. Seven weight-only
serialization tests also passed. The
release-built Python extension passed all 22 CPU tests; the separate WGPU smoke
test failed on this WSL adapter with `No tile size is available for the problem`.
That hardware-specific failure is not converted to a skip or GPU correctness claim.
The reference fixture generator was rerun successfully. Build warnings remain:
Burn/CubeCL future incompatibilities and unavailable `patchelf` for wheel rpath.

**Limitations:** TensorFlow and Keras are not installed in this environment, so the complete `ParametricUMAP` class cannot be instantiated. No TensorFlow training run, gradient comparison, GPU comparison, trustworthiness comparison, or statistical quality ranking was performed. Effects below follow from source/math, not measured quality gains or regressions. Keras dependency defaults were corroborated with its [Dense documentation](https://keras.io/api/layers/core_layers/dense/) and [Adam documentation](https://keras.io/api/optimizers/adam/); these are supplementary documentation, not proof of a locally installed Keras version.

## Common ground — but not proof of parity

Both primary implementations learn an encoder and transform unseen samples by a forward pass. Both support arbitrary positive embedding dimensions, use ReLU hidden layers and a linear final layer, use Euclidean embedding distances, and start from Adam with learning rate 0.001 and betas 0.9/0.999. Default `n_neighbors=15`, `min_dist=0.1`, `spread=1`, embedding dimension 2, and negative sample rate 5 agree (`src/train/config.rs:174-177,215-220,335-353,419-428`; `U:1665-1692`; `P:136,804-814,1145`). Both compute the nominal kernel `q(d)=1/(1+a*d^(2b))`, but Python subsequently applies a sigmoid to **log(q)**, which changes the effective loss. See difference 10.

## Differences

Labels: **Core mismatch** changes the default mathematics; **Defect** is misleading or inconsistent Rust behavior; **Tradeoff** is a deliberate-looking implementation choice whose quality is not established; **Capability gap** is missing optional functionality. These labels describe observed behavior, not author intent.

### Graph and high-dimensional geometry

### 1. Mandatory feature standardization changes the input geometry — Addressed in primary route

- **Original finding:** At the audited revision above, every feature was population-z-scored with `std + 1e-6` before graph construction and encoder training, with statistics reused for transforms. This was unconditional, not controlled by `graph.normalized`.
- **Current Rust:** `src/train/train_sparse.rs::sparse_inputs` passes caller-supplied values to both neighbor construction and encoder training. `Umap::finalize` and `FittedUmap::transform_to_tensor` forward inputs without feature scaling; `FittedUmap` no longer stores feature means or standard deviations. Newly fitted and loaded weights therefore use the same input contract.
- **Python:** `U:2367-2379,2550-2555,2624-2646`; `P:333-375,874-884,288-292`. The graph and encoder receive input X without equivalent automatic scaling.
- **Remediation:** Removed automatic feature scaling from the primary Rust and delegated Python paths. Callers own any optional preprocessing and must apply it consistently before fit and transform. Feature units now affect geometry directly; `graph.normalized` remains a separate distance setting.
- **Verification:** Raw-space nearest-neighbor and exact encoder-input assertions use unequal feature scales and a constant column with value `7`; direct model-forward comparisons cover finalization and transform. Restoring feature scaling in a temporary mutation made the nearest-neighbor regression fail (neighbor `1` instead of `2`); the mutation was removed. Rust persistence tests compare probe outputs before and after save/load. `cargo test --no-default-features` passed (64 tests; 10 ignored); two `nn-descent` sparse-preparation regressions passed, including the actual approximate branch at its 10,000-sample threshold. After rebuilding the native extension, `uv run --no-sync pytest python/tests/test_umap.py -k 'not gpu_smoke'` passed (13 tests; one hardware test deselected), including a real CPU constant-column preservation regression. The GPU smoke selector was corrected to `wgpu` and its skip handling narrowed to missing-adapter errors; attempting the hardware test aborted inside the host graphics driver, so GPU behavior is not verified. These checks establish the CPU preprocessing contract, not complete Python algorithm parity.
- **Notes:** Deprecated routes and shared scaling helpers remain unchanged. Previously saved weights trained on internally standardized inputs still need their original external preprocessing or retraining; there is no migration. Other numbered findings retain their original audit context and have not been re-audited here. Python's optional correlation-loss z-scoring is not graph/encoder preprocessing.

### 2. Local-connectivity radii, bandwidth search, and fuzzy memberships — Addressed in primary route

- **Current Rust:** `sparse_inputs` retains distances; `smooth_knn_dist` computes rho interpolation, 64-step sigma search, and row/global sigma floors. `directed_memberships` distinguishes explicit self from duplicate zero-distance observations and skips missing index -1. Primary Rust/Python expose finite `local_connectivity >= 0` and `bandwidth > 0` (defaults 1).
- **Python:** `U:153-254,420-435,578-590`. Computes local radius rho and searches sigma so that the local membership sum approximates `log2(k)*bandwidth`; memberships are 1 inside rho and `exp(-(d-rho)/sigma)` outside it.
- **Evidence:** `tests/fixtures/umap_learn_0.5.12_fuzzy_graph.json` is generated from installed helpers with NumPy 2.5.3. Rust fixtures compare rho/sigma, directed memberships, and symmetric weights on identical supplied neighbor lists, including duplicates, fractional connectivity, bandwidth, floors, and -1 skipping. This does not establish approximate-neighbor identity or embedding-quality parity.

### 3. Fuzzy union/intersection of directed neighborhoods — Addressed in primary route

- **Current Rust:** `directed_memberships` coalesces repeated coordinates; `symmetrize` applies the reference formula, removes zeros, and retains one coordinate in each direction. `set_op_mix_ratio` is exposed in Rust/Python, defaults to 1, and must be finite in [0,1].
- **Python:** `U:593-603` combines membership matrix A with its transpose using `mix*(A+Aᵀ-A∘Aᵀ)+(1-mix)*(A∘Aᵀ)`; the default is fuzzy union.
- **Evidence:** Fixed asymmetric/mutual graph fixtures cover mix 0, 0.5, and 1, reverse-weight symmetry, self zeroing, and duplicate-coordinate coalescing. Pure intersection with no surviving positive mass fails explicitly instead of falling back to union.

### 4. Weight-based pruning and integer positive distribution — Addressed; schedule remains different

- **Current Rust:** `repetition_counts` prunes strictly below `max(w)/E` and computes `floor(E*w)` with float32 operations using configured epochs E. `EdgeSampler` uses checked u64 counts/prefixes and fresh independent replacement draws, `min(50_000,sum(counts))` per epoch, with probability `count/sum(counts)`. It never expands the repeated-edge dataset; zero mass and overflow are explicit errors. Early stopping does not recompute counts.
- **Python:** `P:618-634,905-923` sums duplicate COO entries, prunes `w < max(w)/n_epochs`, and repeats each edge `int(n_epochs*w)` times before shuffling.
- **Evidence:** Versioned fixtures and boundary tests pin `[1,0.2,0.001]`, E=10 -> `[10,2,0]`, threshold equality/adjacent float32 values, and `[0.7]`, E=10 -> 7 (widening before multiplication incorrectly yields 6). Ticket boundaries and seeded probability tests cover exact count distribution. Replacement batches do not reproduce Python's finite shuffled-pass exposure, optimizer-step count, or scheduling.

### 5. Self-inclusive `n_neighbors` — Addressed dependency; public semantic break

- **Current Rust:** Primary `sparse_inputs` requests k-1 non-self candidates from unchanged shared search helpers, validates row lengths/identities/distances, and prepends explicit `(sample_id,0)` at slot zero. Primary validation requires k>=2 and n>k. This is a deliberate breaking change: k=15 now selects 14 other samples, not 15. Deprecated dense and shared search conventions remain unchanged.
- **Python:** `UT:14-35`; `U:192,223,426-427,1784-1785`. Its ordinary within-dataset neighbor list includes self; self membership is zero and smooth-kNN processing skips the first distance. Requires `n_neighbors>=2`.
- **Evidence:** Exact and actual threshold-triggered NN-descent preparation tests assert explicit self layout, distance retention and valid non-self neighbors. Duplicate zero-distance observations remain distinct by identity. ANN tests assert validity, not identical search results or seeded ANN parity.

### 6. Advertised graph metrics and distance options are ignored — Defect

- **Rust:** `src/train/config.rs:93-118,198-220` advertises Manhattan, cosine, Minkowski, `minkowski_p`, and normalized distances. `src/lib.rs:337,350-351` forwards them, but `src/train/train_sparse.rs:154-175` always uses Euclidean distances or Euclidean NN-descent.
- **Python:** `U:2550-2555,2624-2646` uses the selected input metric and keyword arguments; example formulas are `D:109-120,172-187,607-621`.
- **Effects:** Non-Euclidean Rust settings silently produce the wrong graph relative to the requested metric (negative). The Python binding separately rejects non-Euclidean metrics (`python/fast_umap/umap_.py:59-61`), but this does not protect native Rust callers.
- **Remediation:** Implement metric dispatch in every active graph route, or reject/remove unsupported settings. Do not wire the existing public metric helpers in without correcting difference 33.
- **Notes:** `graph.normalized` does not disable difference 1 and does not normalize distances in this route.

### 7. No precomputed-distance, precomputed-neighbor, or callable-metric workflow — Capability gap

- **Rust:** `src/lib.rs:272-305,323-358`; `src/train/config.rs:93-105`. Fit accepts feature rows only and computes neighbors internally.
- **Python:** `P:149-206,335-336` accepts separate precomputed distances while retaining X to train the encoder; `U:1669-1670,1695-1707,2005-2055` exposes input-metric and precomputed-kNN configuration.
- **Effects:** Rust cannot separate encoder inputs from graph geometry, reuse externally constructed neighbors, or train against specialized/custom distance data (negative capability/usability).
- **Remediation:** Introduce validated graph/neighbor inputs separate from feature tensors, with a clear requirement to retain encoder X.
- **Notes:** Optional functionality; its absence alone does not invalidate a deliberately Euclidean-only implementation. Accepting a precomputed distance matrix as encoder features would not reproduce Python's API or math.

### 8. Quadratic exact neighbor preparation remains the default at large n — Tradeoff

- **Rust:** `src/train/train_sparse.rs:135-175`; `src/train/get_distance_by_metric.rs:36-47`; `Cargo.toml:121-130`. Without opt-in `nn-descent`, materializes and transfers the full n×n matrix. Approximate search, when enabled, starts at n=10,000.
- **Python:** `U:2550-2555,2624-2646,328-345`. Normally uses exact pairwise search below 4096 samples and PyNNDescent above that (subject to its force/override options).
- **Effects:** Rust's exact search can produce better neighbor recall (positive), but incurs quadratic memory, transfers, and work (negative). A 50,000×50,000 f32 matrix alone is about 10 GB; temporary allocations add more.
- **Remediation:** Choose approximate search by default at a validated threshold or use bounded-memory tiled exact search.
- **Notes:** Sparse edge optimization does not make graph preparation sparse. Exact-versus-approximate search is not itself a mathematical defect; neither implementation guarantees identical approximate neighborhoods.

### 9. Optional Rust NN-descent has different initialization, convergence, and reproducibility — Tradeoff/defect

- **Rust:** `src/nn_descent.rs:28-43,64-79,84-140`; `src/train/train_sparse.rs:135-162`. Unseeded thread RNG, a full shuffled n−1 candidate list per point, fixed 12 iterations, closest/first candidate subsets, and convergence counts that omit candidate-to-candidate insertions.
- **Python:** `U:328-345,2624-2646` configures PyNNDescent with `random_state`, data-sized iteration/tree counts, RP-forest initialization, and bounded candidates.
- **Effects:** Rust's advertised seed does not determine approximate neighborhoods (negative). Initialization is quadratic in total work despite the large-data intent. Recall/convergence can differ; lower quality is plausible but was not benchmarked.
- **Remediation:** Pass the configured seed, sample initial neighbors without constructing full candidate lists, and use a complete, tested NN-descent implementation/convergence criterion.
- **Notes:** The unused parallel `updates` pass also wastes work. The passing small-data Rust reproducibility test does not cover this feature/threshold route.

### Objective, sampling, and optimization

### 10. The actual Python cross-entropy uses a different effective kernel — Core mismatch

- **Rust:** `src/train/train_sparse.rs:370-392`; `src/train/mod.rs:255-277`. With `u=a*d^(2b)`, positive loss is `log(1+u)` and negative loss is `log((1+u)/u)`, before clamps.
- **Python:** `P:701-720,723-763,1232-1254`. Computes `z=log(q)=-log1p(u)` and then uses z as a **logit** in `log_sigmoid`. Effective probability is `sigmoid(log(q))=q/(1+q)=1/(2+u)`, not q. Positive loss is `log(2+u)`; negative loss is `log((2+u)/(1+u))`.
- **Effects:** Different attraction and repulsion gradients, not merely a constant offset. For u>0, Rust positive derivative is `1/(1+u)` versus Python `1/(2+u)`; Rust negative derivative is `-1/(u*(1+u))` versus Python `-1/((1+u)*(2+u))`. Rust repulsion is especially stronger near overlap, until clamping removes its gradient. Layouts and convergence differ; neither quality direction was measured.
- **Remediation:** For parity with this installed implementation, use `softplus(-z)` and `softplus(z)` with z=`-log1p(u)` and the same aggregation/sampling. For textbook q-based UMAP cross-entropy instead, retain that distinct objective and explicitly state it is not this reference's behavior.
- **Notes:** Numerically checked using the installed helper bodies. At a=b=d=1, Python positive/negative losses are `1.098612289 / 0.405465108`; Rust formulas give `0.693147181 / 0.693147181`. At d=0, Python scalar losses both equal log(2). This reference quirk must not be silently corrected while claiming parity.

### 11. Separate positive/negative means change their relative weight — Core mismatch

- **Rust:** `src/train/train_sparse.rs:376-392` computes `mean(pos)+repulsion_strength*mean(neg)`; negative count is `max(n_pos*neg_rate,n)` (`src/train/train_sparse.rs:188-191`).
- **Python:** `P:1237-1254` takes one mean over B positives and B*r negatives. With equal repulsion strength, the aggregate negative-versus-positive group weight is r:1.
- **Effects:** Rust's default groups have weight 1:1 rather than Python's 5:1. Increasing Rust `neg_sample_rate` changes sample coverage/noise, not the expected total repulsion coefficient; the source documentation suggesting stronger repulsion is misleading (`src/train/config.rs:299-305`). Absolute loss scale also differs.
- **Remediation:** Aggregate the sum of positive and negative per-pair losses over the combined pair count, with the intended strength multiplier, instead of adding separate means.
- **Notes:** Isolated at d=1, a=b=1, r=5: Python combined mean is `0.520989638`, whereas Rust's separate means give `1.386294361`. This example includes difference 10; it is not a trained-model quality comparison.

### 12. Negative endpoints follow different distributions — Core mismatch

- **Rust:** `src/train/train_sparse.rs:222-232` draws both endpoints uniformly from all samples, excluding self but not known neighbors.
- **Python:** `P:1209-1221` repeats current positive heads r times and shuffles repeated positive tails within the minibatch. Does not exclude self or positive neighbors.
- **Effects:** Python negatives follow the edge-endpoint marginals, generally weighted by fuzzy graph degree; Rust negatives follow uniform vertex marginals. Points in different density/degree regions receive different relative repulsion. Excluding self removes uninformative negatives (potential positive), but violates literal parity.
- **Remediation:** Reproduce the within-batch repeated-tail shuffle for reference parity.
- **Notes:** Python's shuffle is a finite permutation, not independent sampling. True neighbors can be negatives on both sides; neither implements an exclusively non-edge sampler in the primary route.

### 13. Fresh edge/negative samples — Addressed consequence of weighted integration

- **Current Rust:** The cached 16-batch cycle is removed. Each epoch advances one seeded stream through fresh integer-weighted positives and fresh uniform non-self negatives. Index tensor uploads are now per-epoch rather than cached.
- **Python:** `P:925-936,1213-1221` continuously shuffles its repeated edge dataset and draws a new negative permutation during loss evaluation.
- **Limits:** Fresh draws remove the fixed-pool coverage restriction but are not the reference finite-pass schedule or negative permutation law. CPU tests verify seeded repeatability, state advancement, bounded positives, and non-self negatives; no throughput or GPU correctness claim follows.

### 14. Epochs, batches, and optimization budgets have incompatible meanings — Core mismatch

- **Rust:** `src/lib.rs:314,339,402,424`; `src/train/train_sparse.rs:349-350,414-416,494-495,548`; `src/train/config.rs:339-340`. Embeds all n points and takes one Adam step per loop. Configured batch size is not used. With no early stop, the loop runs indices 0 through `epochs` inclusive: default 100 means **101 steps**.
- **Python:** `P:624-629,910-917,397-398,430-436`. `n_epochs` controls edge multiplicities (default 500 at n<=10,000, else 200), not Keras epochs. Default batch size is min(n,1000). Runs `loss_report_frequency*n_training_epochs` Keras epochs, normally 10, each with `floor(n_expanded_edges/batch_size/10)` steps.
- **Effects:** Different stochastic gradient noise, number of updates, edge passes, and memory usage (negative for parity). Rust's full embedding pass may efficiently reuse endpoints, but full-graph activations remain in memory. Equal epoch settings are not comparable budgets.
- **Remediation:** Define edge-minibatch training, wire batch size, separate edge exposure from reporting epochs, and fix the inclusive loop boundary. Compare total edge exposure/optimizer steps explicitly.
- **Notes:** Python's integer rounding can also yield zero `steps_per_epoch` on tiny graphs. No claim that arbitrary small datasets work in the full reference was tested.

### 15. Rust clamps introduce loss plateaus and different overlap behavior — Core mismatch

- **Rust:** `src/train/train_sparse.rs:373-385` clamps d² to 1e-8 and probabilities to at least 1e-6; dense equivalent at `src/train/mod.rs:257-270`.
- **Python:** `P:720,751-759,1224-1235` uses norm, `log1p`, and stable `log_sigmoid`; its declared `EPS=1e-4` is not used by the helper body.
- **Effects:** Rust caps per-pair logarithmic losses around 13.8155 and produces zero-gradient regions inside clamps, including sufficiently distant positives and nearly overlapping pairs (negative for literal parity; positive for bounding values). Python's effective negative scalar loss is bounded by log(2), for a different reason.
- **Remediation:** Use the reference stable log-space objective, then separately test differentiation at zero and very small distances; do not add undocumented hard clamps to make tests pass.
- **Notes:** Finite scalar loss does not guarantee finite derivative through `norm` or a fractional power at zero. Python runtime gradients were not executed, so overlap-gradient safety is not certified on either backend.

### 16. Adam clipping, regularization, epsilon value, and epsilon convention differ — Core mismatch

- **Rust:** `src/train/train_sparse.rs:311-315`; `src/train/mod.rs:202-205`; `src/train/config.rs:341-344`. Uses Burn Adam with default L2-style `penalty=1e-5`, no configured gradient clipping. `BO:29-31,75-79,194-209` uses epsilon 1e-5 and adds `epsilon*sqrt(1-beta2^t)` to the uncorrected second-moment denominator.
- **Python:** `P:136,1145` explicitly uses Adam(0.001, clipvalue=4.0), without default weight decay. Documented Keras epsilon is 1e-7, added directly to sqrt(v); corroborating [Keras source v3.15.1, lines 107-141](https://github.com/keras-team/keras/blob/v3.15.1/keras/src/optimizers/adam.py#L107-L141).
- **Effects:** Regularization shrinks parameters; missing clipping and a different denominator change optimization, especially for small gradients (negative for parity). Clipping matters in conjunction with differences 10–11. Rust does expose its actual learning rate, whereas base Python UMAP's `learning_rate` is not the parametric optimizer's rate.
- **Remediation:** Match clipping-by-value and default penalty zero; verify the exact Adam update, including epsilon placement. Merely setting Burn epsilon to 1e-7 does not reproduce Keras's epsilon-hat convention.
- **Notes:** Learning rate and betas agree by default. Keras is absent locally; the supplementary source/documentation confirms its convention, not a pinned installed Keras build. Test controlled gradient sequences before claiming optimizer parity.

### 17. Kernel parameter fitting can differ dramatically at small spread — Core mismatch

- **Rust:** `src/train/config.rs:15-82`; `src/lib.rs:333-334`. Midpoint grid, coarse a/b grid, then 100 rounds of multiplicative 2% coordinate refinement.
- **Python:** `U:1393-1408,2382-2386`. Inclusive `linspace(0,3*spread,300)` and SciPy nonlinear least squares; supports explicit a/b overrides.
- **Effects:** Default fit differs slightly, but valid small-spread settings can differ by orders of magnitude, changing the embedding-distance scale and kernel shape (negative). Rust cannot bypass fitting through the primary UmapConfig with known reference a/b values.
- **Remediation:** Use the same sample grid and a well-conditioned nonlinear least-squares fit (including scale normalization); support explicit a/b and validate across min_dist/spread ranges.
- **Notes:** Actual compiled Rust/reference helper results:

  | min_dist | spread | Rust a | Rust b | Python a | Python b |
  |---:|---:|---:|---:|---:|---:|
  | 0.1 | 1 | 1.568 | 0.9 | 1.5769435 | 0.89506088 |
  | 0.5 | 1 | 0.5824 | 1.3247999 | 0.58303002 | 1.334167 |
  | 0.1 | 0.1 | 200.86755 | 1.5200511 | 830.00522 | 1.9292376 |
  | 0.01 | 0.01 | 219.12976 | 0.6836876 | 5991745.3 | 1.9292368 |

  SciPy emitted divide-by-zero/overflow warnings during some small-scale fit trials; these warnings were not suppressed. Listed fits are observed outputs, not universal high-precision ground truth.

### 18. Default encoder has one hidden layer instead of three — Tradeoff

- **Rust:** `src/train/config.rs:419-428`; `src/lib.rs:323-328`; `src/model.rs:23-49,59-73`. Default UmapConfig overrides the model builder with hidden_sizes=[100].
- **Python:** `P:804-814`. Flatten → 100 ReLU → 100 ReLU → 100 ReLU → linear embedding.
- **Effects:** Rust has less capacity and less computation (mixed); learned mappings and inductive biases differ. Shallower does not automatically mean worse on every dataset.
- **Remediation:** Use [100,100,100] for a default-parity contract; users can already select those widths explicitly.
- **Notes:** The lower-level Rust builder defaults to three hidden layers (`src/model.rs:112-120`), but that is not the primary API's effective default. Both have linear, unconstrained outputs.

### 19. Parameter initialization differs — Core mismatch

- **Rust:** `src/model.rs:29-42`; `BN:21-24,99-105`. Burn defaults to KaimingUniform with gain 1/sqrt(3) and uses the initializer for biases too.
- **Python:** `P:809-812` leaves Dense initializers at Keras defaults: Glorot-uniform kernels and zero biases, as documented by Keras Dense.
- **Effects:** Initial embedding scale, activation distributions, and early gradients differ (negative for controlled equivalence; no established quality ranking). Identical numerical seeds do not imply identical weights across frameworks.
- **Remediation:** Match initializer distributions and zero biases; for deterministic mathematical comparisons, import identical parameter tensors rather than comparing same-seed generators.
- **Notes:** Backend RNG algorithms and tensor layouts remain independent even after distribution matching.

### 20. Optional PCA warm-start adds an extra optimization phase — Tradeoff

- **Rust:** `src/train/train_sparse.rs:261-308`; `src/pca.rs:15-119`. With the pca feature, computes centered covariance/power-iteration PCA targets and trains the encoder against them for 10 MSE steps at learning rate 0.01 before manifold training.
- **Python:** `P:379-395,766-831`. Actual parametric path sets `init_embedding=None` and does not train against PCA or spectral coordinates.
- **Effects:** May improve initialization or global organization (possible positive), but changes objective history and optimization budget (negative for parity). Rust PCA is approximate and uses fixed initial eigenvectors.
- **Remediation:** Disable the feature for reference comparisons; make warm-start an explicit fit option rather than an implicit consequence of feature compilation.
- **Notes:** `P:643-698` contains a spectral/random embedding helper, but it is not invoked by the actual parametric fit route. Base `init='spectral'` must not be mistaken for spectral initialization of this encoder.

### 21. Best-checkpoint restoration and loss reporting differ — Tradeoff

- **Rust:** `src/train/train_sparse.rs:395-404,419,464-470,573-574`. Reads loss every 5 loops, repeats stale loss values between reads, and restores an automatically selected best model. The checkpoint is post-step, while the comparison loss is pre-step.
- **Python:** `P:430-445`. By default returns the final encoder; custom callbacks can change this through `keras_fit_kwargs`.
- **Effects:** Rust may return earlier parameters rather than final ones, potentially avoiding later deterioration (positive), but comparisons select on noisy sampled losses and do not match reference behavior (negative). Reported loss/history cannot be treated as independent measurements every epoch.
- **Remediation:** Make restoration optional, evaluate checkpoints consistently on a fixed/evaluation sample, and mark unsampled history values rather than duplicating stale measurements.
- **Notes:** Patience advances in readback increments. Returning a model with a lower sampled training loss is not proof of a better embedding.

### 22. Configuration semantics differ and some advertised knobs are no-ops — Defect/extension

- **Rust:** `src/train/config.rs:268-282,335-351`; `src/train/train_sparse.rs:108,376-392`; `src/lib.rs:314,339`. Applies `repulsion_strength`, but ignores `loss_reduction` and configured batch size in the primary sparse route.
- **Python:** `P:1188,1204,1248-1254,136`. `_umap_loss` is called without forwarding base `self.repulsion_strength`, so its default 1.0 applies. Base `learning_rate` is likewise not wired to the hardcoded parametric Adam optimizer.
- **Effects:** Rust strength can actually tune repulsion (positive capability), while the same-named Python base parameter does not do that here. Rust Sum/Mean selection and batch-size settings promise behavior they do not deliver (negative usability).
- **Remediation:** Remove no-op Rust knobs or implement them; document parameter semantics based on the active route. Do not equate Python base-class arguments with optimizer controls without tracing their use.
- **Notes:** Matching non-default `repulsion_strength` by passing the same number on both sides is not a valid parity experiment.

### Optional learning capabilities and inputs

### 23. Labels are plotting annotations, not supervised targets — Capability gap

- **Rust:** `src/lib.rs:257-258,272`; `src/train/train_sparse.rs:520-527`. Optional string labels are passed to charting, not graph construction or loss.
- **Python:** `P:149-206`; `U:2692-2805`. y can modify the data graph through categorical/continuous target relationships and target-weight intersection.
- **Effects:** No supervised/semi-supervised dimensionality reduction (negative). Rust labels do not constrain same-label proximity. The Python binding explicitly rejects y (`python/fast_umap/umap_.py:90-92`).
- **Remediation:** Add target-graph construction/intersection if supervision is required, or keep the limitation explicit and distinguish plot labels from training targets.
- **Notes:** A label-specific neural loss alone would not necessarily reproduce Python's supervised graph semantics.

### 24. Reconstruction, autoencoder regularization, and inverse transform are absent — Capability gap

- **Rust:** `src/model.rs:23-49`; `src/train/train_sparse.rs:370-392`; `src/lib.rs:524-615`. Encoder only; no reconstruction term or inverse mapping.
- **Python:** `P:294-312,816-829,1150-1155,1171-1178,1284-1288,400-421`. Optional decoder, weighted default BCE-from-logits reconstruction, optional reconstruction validation; gradients reach the encoder only when `autoencoder_loss=True`. Otherwise reconstruction sees stop-gradient embeddings. Inverse transform uses the decoder when enabled and inherited inverse transform otherwise.
- **Effects:** No generative decoding, learned inverse mapping, or reconstruction-driven regularization (negative capability). Simpler/lighter default training is positive, and default Python reconstruction is off.
- **Remediation:** Add an optional decoder and weighted reconstruction objective with the exact gradient-stop semantics and validation path; decide separately whether inherited nonparametric inverse mapping is required.
- **Notes:** An MSE decoder would be a new capability, not exact parity with the default BCE. Default unsupervised fits are unaffected by absence of an off-by-default term.

### 25. Global-correlation regularization is absent — Capability gap

- **Rust:** `src/train/train_sparse.rs:370-392` has only local attraction/repulsion; no correlation option in `src/train/config.rs:231-353`.
- **Python:** `P:888-889,1190-1192,1256-1282`. Optional negative Pearson correlation of consecutive head-row distances in globally z-scored/clipped input and embedding arrays, with tiny embedding-distance jitter.
- **Effects:** Rust cannot use this local/global-structure tradeoff (negative capability); no default mismatch because Python's weight is zero.
- **Remediation:** Add the optional term with the reference centering, clipping, jitter, batch-head selection, and weighting.
- **Notes:** This is not a full all-pairs stress loss. Replacing it with all-pairs distance MSE changes the algorithm.

### 26. No landmarks or continued fitting of the same encoder — Capability gap

- **Rust:** `src/lib.rs:272-330,514-522`. Fit consumes the unfitted object and builds a fresh encoder; FittedUmap exposes no continued training or landmark input.
- **Python:** `P:149-206,177-185,351-357,379-395,437-442,512-573,1102-1108,1290-1309`. Reuses its existing model on refit, accepts NaN-masked landmark targets, can retain previous samples and their current transformed positions, and accumulates history. Default landmark loss is mean `sqrt(sum((pred-target)^2)+1e-10)`, weighted; empty landmark batches contribute zero.
- **Effects:** Rust cannot anchor coordinates or adapt a learned embedding while preserving selected old positions (negative capability). Separate fits restart rather than update.
- **Remediation:** Add explicit continued fitting with model/optimizer ownership and masked L2 landmark constraints. Preserve or reset optimizer state deliberately.
- **Notes:** MSE landmarks are not the same as the reference mean distance. The reference landmark path uses TensorFlow control flow; cross-backend portability is not established here.

### 27. Custom architectures and shaped encoder inputs are missing — Capability gap

- **Rust:** `src/lib.rs:272-305,323-330`; `src/model.rs:23-49,83-86`. Dense row input and a configurable-width MLP only.
- **Python:** `P:48-61,138-147,338-344,804-814`. Accepts custom encoders/decoders, checks encoder output width, and reshapes flattened data to configured dims for image/convolutional models.
- **Effects:** Rust cannot express convolutional/domain-specific encoders or custom network regularization/callback patterns (negative capability). It can still accept flattened image features as ordinary vectors.
- **Remediation:** Introduce a bounded encoder interface if custom models are a requirement, and specify how flattened graph features map to encoder tensor shapes.
- **Notes:** `U:2372-2378` accepts CSR for base graph fitting, but full CSR ingestion into this TensorFlow edge dataset was not executed and should not be advertised as verified end-to-end sparse neural-input support. Rust's API does not offer CSR at all.

### 28. Transform has no inference minibatching — Tradeoff/usability gap

- **Rust:** `src/lib.rs:583-615`. Normalizes all supplied rows and forwards one tensor; no batch-size argument. Empty input has a special empty-tensor result.
- **Python:** `P:272-292`. Uses encoder.predict with a selectable batch size, defaulting to its training batch size.
- **Effects:** Similar deterministic mapping in principle, but Rust may run out of memory on large inference sets (negative); a single tensor minimizes per-batch overhead (positive).
- **Remediation:** Add bounded/chunked inference using the same stored preprocessing statistics and preserve row ordering.
- **Notes:** For the current ordinary MLP, changing inference batches should not change the mathematical mapping. Custom batch-sensitive encoders would require their own semantics.

### 29. Validation and normalization robustness differ — Defect/usability gap

- **Rust:** `src/lib.rs:131-175,309-310`; `src/utils.rs:212-235`; `src/train/get_distance_by_metric.rs:42-47`. Checks dimensions but not numeric finiteness or manifold/optimizer ranges; invalid fit input becomes a panic. Variance uses `E[x²]-E[x]²` in caller precision, which is cancellation-prone and can become negative before sqrt. NaN distance comparisons fall back to Equal.
- **Python:** `U:1754-1785,2367-2378,2456-2470`. Validates key ranges and by default checks finite X; oversized neighbor counts are warned/truncated rather than always rejected.
- **Effects:** Rust can accept settings or nonfinite data that later corrupt the graph/loss; f32 large-offset features can destabilize normalization (negative). Strict n>k is predictable but makes small datasets less convenient than truncation.
- **Remediation:** Use stable f64/two-pass or Welford statistics, validate finite inputs and parameter domains, and return structured errors. Specify whether oversized k should be rejected or adjusted.
- **Notes:** The Python binding has extra input guards (`python/fast_umap/umap_.py:79-88`); those are not native Rust guarantees. Python itself does not guarantee successful neural training for every graph that passes base validation.

### Persistence, alternate routes, and public mathematical helpers

### 30. Save/load loses preprocessing and changes future transforms — Defect

- **Rust:** `src/lib.rs:640-645,678-707`. Saves weights only; load requires external architecture/configuration, uses empty cached embedding, and resets means/stds to 0/1.
- **Python:** `P:458-510,981-1025`. Saves encoder, optional decoder, full model, and estimator pickle; loads the encoder and estimator used by transform.
- **Effects:** Rust predictions on raw inputs after reload generally differ from predictions before save, because training standardization is lost (negative output/usability). Supplying the same model configuration cannot recover missing statistics.
- **Remediation:** Persist the preprocessing statistics and architecture/configuration together with weights, and test raw-input transform equality across the round trip.
- **Notes:** Reference persistence is not perfect either: save writes `parametric_model.keras` (`P:475-479`) but load searches for `parametric_model` without that extension (`P:1019-1022`). Therefore full training-model/optimizer restoration must not be assumed. Encoder-based inference restoration is a narrower supported comparison. Rust's f64 result containers also do not imply f64 learned coordinates; `src/utils.rs:127-145` extracts f32 values.

### 31. Deprecated dense route optimizes a third objective and skips transform scaling — Defect/alternate route

- **Rust:** `src/lib.rs:763-816,825-835`; `src/train/mod.rs:117,141-191,249-277,322-328,415-417`. Deprecated UMAP::fit calls dense training. Direct dense train calls construct kNN within fixed contiguous batches and repel every within-batch non-neighbor; legacy fit uses a single full-data batch. Legacy transform forwards raw data despite standardized training. Dense checkpoints use `/tmp/{name}.burnpack` (`src/train/mod.rs:85`).
- **Python:** `U:2550-2676`; `P:359-375,905-936,1209-1254,272-292`. Global fuzzy graph, edge minibatches, sampled negatives, and matching encoder-input semantics.
- **Effects:** Different results from Python and the primary Rust sparse route. Direct smaller batches lose cross-batch neighbors; legacy transforms have inconsistent scaling (negative). Shared checkpoint names can interfere across concurrent dense runs.
- **Remediation:** Remove the obsolete parallel implementation and route callers through the corrected primary path; preserve preprocessing in every transform path and avoid shared checkpoint filenames.
- **Notes:** Custom Euclidean-distance autodiff (`src/autodiff_ops.rs:37-76`) matters to dense embedding-distance gradients, not the primary sparse loss, which uses standard select/subtract/square/sum operations. Custom backward registration is not evidence that primary parity is achieved.

### 32. Feature-gated CpuUmap is not UMAP at all — Defect/alternate route

- **Rust:** `src/lib.rs:103-104`; `src/cpu_backend.rs:23-45,63-77,104-105,114-123`. Public CPU-specific fit returns the first two standardized features multiplied by 10, always two dimensions; no neighbor graph or learning. Its transform panics.
- **Python:** `P:333-447,272-292,804-814`. Trains a learned encoder and can transform new data.
- **Effects:** CPU-specific API does not provide manifold learning or parametric projection and ignores the requested embedding dimension (negative). Its source calls it a demonstration/placeholder; that does not make it a parity-capable backend.
- **Remediation:** Remove this placeholder API or route it to the working Flex CPU primary implementation.
- **Notes:** **Do not confuse this with Umap on Flex CPU.** The latter is the real parametric sparse route tested above and works without the gpu feature; it is not this projection stub.

### 33. Public distance helpers do not implement their advertised formulas — Defect outside primary route

- **Rust:** `src/distances.rs:13-39,65-90,111-120,162-181,229-249`, exported by `src/lib.rs:107`.
- **Python:** `D:49-58,109-120,172-187,607-621` supplies actual Euclidean, Manhattan, Minkowski, and cosine dissimilarity formulas.
- **Difference:** Rust Euclidean helper triangularizes squared distances and slices the first column, rather than returning Euclidean distances; Manhattan returns row L1 norms instead of differences between samples; cosine returns similarity to row 0, not `1-cos`, with no zero-norm handling; Minkowski uses an integer/rounded root exponent (`ceil(1/p)`) instead of 1/p. Its kNN helper triangularizes and sums top-k squared values instead of exposing nearest-neighbor Euclidean distances.
- **Effects:** Users of the public helpers get incompatible or incorrect distance values (negative). Reusing them to implement difference 6 would introduce more defects.
- **Remediation:** Delete unused misleading helpers or implement explicit pairwise/reference-row contracts with correct formulas and numeric fixtures against D.
- **Notes:** These helpers are **not used by primary train_sparse**, which uses the separate Euclidean pairwise kernel. Their defects must not be mistaken for the actual primary Euclidean formula. Both shape contracts and numeric outputs need tests, not just successful tensor construction.

### 34. Disconnected vertices, duplicate handling, and nonfinite output visibility differ — Capability gap/defect

- **Rust:** `src/train/train_sparse.rs:177-185` assigns k positive neighbors per point without disconnection pruning; `src/utils.rs:139-145` silently converts NaN embedding coordinates to zero. No unique-row graph option in `src/train/config.rs:385-428`.
- **Python:** `U:2419-2453,2584-2587,2651-2655,2841-2847` offers unique-row fitting, disconnection-distance pruning, and explicitly marks zero-degree graph vertices with NaN fit coordinates. `U:422-423` skips missing-neighbor indices.
- **Effects:** Duplicate frequencies and zero-degree semantics can change the fit. Rust can make numerical failures look like legitimate points at the origin (negative). It does not guarantee finite output merely because NaNs are zeroed: infinities are not converted.
- **Remediation:** Preserve or explicitly report nonfinite results rather than hiding them; add missing-neighbor/disconnection semantics and deduplication only if required by the graph contract.
- **Notes:** For default unbounded Euclidean distance, disconnection cutoff is generally inactive. Python's post-fit NaN marking can make `embedding_` differ from a subsequent encoder.transform(X) on disconnected samples; this is a reference semantic quirk, not evidence of a different encoder.

## Important reference-side limitations

The target is the installed implementation, not an idealized textbook algorithm. Several inherited options do not confer the capability their names suggest:

- `output_metric` does not change `P:1224-1230`, which uses Euclidean norm. A non-Euclidean parametric output-loss implementation is **not** an established Python capability missing from Rust.
- `init_embedding_from_graph` exists (`P:643-698`) but is not used by the parametric route, which passes None (`P:379-392`).
- Base `repulsion_strength` and base `learning_rate` are not forwarded into the actual parametric loss/optimizer (difference 22).
- The parametric override returns empty auxiliary data (`P:445-447`); it does not implement densMAP auxiliary radii. Inherited `output_dens` tries to consume `rad_orig`/`rad_emb` (`U:2848-2850`). Do not claim densMAP/output-density parity merely by exposing those inherited arguments.
- Python's default network has a linear output (`P:812`), whereas its optional PyTorch export helper applies a final ReLU (`P:1340-1349`). Exported output can therefore differ from Keras inference. Rust's lack of that export helper is not a core mathematical mismatch with Keras training.
- Full-model load has the filename discrepancy documented in difference 30.

## Remediation priority and acceptance criteria

1. **Choose the parity contract:** exact installed 0.5.12 behavior versus textbook q-based UMAP cross-entropy. These are different contracts; explicitly choose before changing the loss.
2. **Correct the default pipeline:** graph memberships/set operations, weighted edge exposure, loss/aggregation, negative distribution, minibatches, parameter fit, initialization, and optimizer update. Address all together in controlled tests; fixing only the displayed kernel formula is insufficient.
3. **Fix silent correctness/usability defects:** reject unimplemented metric knobs, persist preprocessing, validate finite inputs, remove/fix obsolete dense and CPU-placeholder APIs, and stop converting numerical failures to zero.
4. **Add optional capabilities only when needed:** target graphs, reconstruction, landmarks, correlation, precomputed graph data, and custom encoders. These gaps do not affect Python's default off-by-default losses.

A meaningful parity test suite should compare, in order: controlled raw/preprocessed inputs; exact kNN/self conventions; rho/sigma and symmetric fuzzy weights; pruning and edge multiplicities; fixed positive/negative pair losses and gradients over overlap/near/far distances; controlled Adam gradient-sequence updates; and same-weight encoder/round-trip transforms. Only then compare trained embeddings statistically (neighbor retention/trustworthiness and downstream utility across seeds). Coordinate equality from two independently initialized runs is not a sensible acceptance test: translations, rotations, approximate neighbor search, RNGs, and optimizer trajectories can differ even for equivalent mathematical objectives.
