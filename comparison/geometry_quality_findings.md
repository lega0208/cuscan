# Geometry and quality investigation

## Conclusion

**The largest observed geometry difference is caused by asymmetric duplicate handling in the comparison, not evidence of a broken CUDA attraction/repulsion formula.** Python gets `unique=True`; fast_umap fits duplicate observations as distinct vertices. This materially changes neighborhoods, graph connectivity, repulsion frequency, and the embedding.

The controlled 50k runs support this directly:

- Deduplicating before fast_umap recovers approximately **84% of the measured trustworthiness gap** against Python's deduplicated fit.
- Turning Python's deduplication **off** produces similarly extended peripheral geometry and slightly *lower* measured trustworthiness than fast_umap on raw observations.
- Changing native initialization to random does **not** materially repair raw-data quality, despite removing the performance bottleneck.

There remains a smaller quality difference on matched deduplicated inputs. It is not resolved by this investigation, and one stochastic run is not enough to establish its significance. See the [performance report](performance_findings.md) for the independently confirmed slow spectral path.

## Evidence and comparison contract

Checkout `14421bd`; umap-learn 0.5.12; CUDA on RTX 3080; 128-dimensional float32 vectors sampled by `comparison/plot_comparison.py:30–51`, seed 42. Full fits use 15 neighbors, Euclidean distance, two dimensions, `min_dist=0`, default 200 epochs, `random_state=None`. No estimator, kernel or plot-script changes were made.

Measured arrays, fit JSON records and diagnostic scripts are in `../artifacts/comparison-investigation/`. All saved embeddings were checked for expected shape and finite values. The collector changes only the controlled settings described below; fit imports, dataset loading and plotting are outside timers.

Important harness differences:

| Setting | Python | fast_umap |
|---|---|---|
| Duplicate treatment | `unique=True`, results mapped back | All observations fitted independently |
| `random_state=None` | Fresh RNG, parallel stochastic layout | Falls back to seed 42; CUDA atomic ordering remains nondeterministic |
| Default epochs at these sizes | 200 | 200 |
| Initialization | Spectral | Spectral |
| Plot axis limits | Independent autorange | Independent autorange |

`plot_comparison.py:85–90` establishes the duplicate mismatch. Native seed fallback is in `src/standard_umap/estimator.rs:22–23,74`. `--device` is accepted but ignored by the plot's hardcoded `backend='cuda'`; this is a harness defect, not the cause of these CUDA results.

## 1. Duplicate-heavy inputs explain the microclusters

| Sample | Distinct vectors | Excess duplicate rows | Largest duplicate multiplicity | Groups with ≥15 copies | Rows in those groups |
|---|---:|---:|---:|---:|---:|
| 15k | 13,312 | 1,688 (11.3%) | 71 | 10 | 265 |
| 50k | 41,744 | 8,256 (16.5%) | 233 | 114 | 3,246 |

With `n_neighbors=15`, the fitted graph has a self slot plus **14 non-self slots**. Groups with at least 15 copies can fill all non-self slots with zero-distance copies. This leaves no requirement for outgoing links to distinct content. Incoming edges may still connect a group to the main graph, so not every such group must be disconnected.

The fuzzy graph assigns zero-distance neighbors weight one (`src/kernels/standard_umap/graph.rs:299–313`). Repeated vertices can therefore create internally strong groups with few external connections. Uniform negative sampling also samples observations rather than unique feature vectors, changing the mass assigned to repeated content.

### Connectivity measurements

The exported native **ANN** graph, symmetrized and analyzed on the host:

- 15k: one connected component, all 15,000 observations.
- 50k: **four components**, sizes **49,923, 36, 22, 19**.
- The three small 50k components contain only **two, two, and one distinct feature vectors**, respectively. They are concretely duplicate-dominated disconnected groups, not merely disconnected-looking scatterplot islands.

Python's fitted, pruned **fuzzy** graph:

- `unique=True`, 15k and 50k: one component each, with 13,312 and 41,744 vertices respectively.
- `unique=False`, 50k: **seven components**, largest 49,888 vertices; no singleton components.

These graph types are not identical: the native export is pre-fuzzy ANN, whereas Python's graph is weighted/pruned. Do not infer an exact graph-component parity result from their counts. Nevertheless, native ANN disconnections cannot be connected by later symmetrization or pruning, and both raw-input estimators demonstrate duplicate-associated fragmentation. The native weighted initialization graph was not exported.

UMAP does not determine a unique global arrangement of disconnected components. Peripheral islands can move substantially across seeds or solvers without representing a different local neighborhood structure.

## 2. Controlled neighborhood-quality measurements

We evaluate a fixed sample of **2,000 distinct feature vectors** at each size, using one representative original row per vector. Every method at that size uses the same selected representatives. This avoids counting thousands of tied duplicates as independent quality evidence.

Metrics:

- **Trustworthiness@15**: scikit-learn's Euclidean rank-based score on the 2,000-vector subset, higher is better.
- **Neighbor overlap@15**: mean intersection fraction of input-space and embedding-space 15-neighbor lists *within that same subset*, higher is better.

These are subsampled local metrics, not full-50k trustworthiness, semantic label accuracy, or proof of identical global geometry. Evaluation seed is 20261014; fitting remains as configured by the original plot.

### 50k sample

| Method | Duplicate treatment | Initialization | Trustworthiness@15 | Neighbor overlap@15 |
|---|---|---|---:|---:|
| Python | Unique | Spectral | **0.8684** | **0.3039** |
| fast_umap | Raw | Spectral | 0.8333 | 0.2589 |
| fast_umap | Raw | Random | 0.8368 | 0.2583 |
| fast_umap | Explicitly unique, then mapped back | Spectral | **0.8628** | **0.2943** |
| Python | Raw (`unique=False`) | Spectral | 0.8241 | 0.2526 |

Deduplicating native input closes 0.0295 of the original 0.0351 trustworthiness gap (approximately 84%), and approximately 79% of the neighbor-overlap gap. This is a much stronger explanation than assuming an incorrect negative-sampling coefficient from the plot alone.

Python on raw observations resembles native raw quality, and scores slightly worse in this single draw. That does not establish native superiority: Python fitting is unseeded and the ANN graphs/layout trajectories differ.

### 15k sample

| Method | Trustworthiness@15 | Neighbor overlap@15 |
|---|---:|---:|
| Python, unique | 0.8907 | 0.3242 |
| fast_umap, raw spectral | 0.8769 | 0.3093 |

The raw-vs-unique gap is smaller at 15k, consistent with fewer large duplicate groups. These samples are independently drawn and are not guaranteed to be nested.

## 3. Apparent central compression is partly an axis/outlier effect

The plot enforces square coordinate aspect ratios, **not common numerical axis ranges** (`plot_comparison.py:133–135`). Farther peripheral groups expand a panel's autorange, making its main body look tighter on screen.

Radial extent from each embedding's coordinate-wise median, 50k:

| Method | Median radius | 95th-percentile radius | Maximum radius | Median / maximum |
|---|---:|---:|---:|---:|
| Python unique | 4.66 | 9.46 | 11.29 | 0.412 |
| Native raw spectral | 4.81 | 14.71 | 20.61 | 0.233 |
| Native raw random | 5.05 | 15.10 | 21.35 | 0.236 |
| Native unique spectral | 4.57 | 8.48 | 12.00 | 0.381 |
| Python raw spectral | 4.84 | 14.84 | 20.73 | 0.234 |

The native raw median radius is **not smaller** than Python unique's. Its outer extent is nearly twice as large, however, and its median occupies a much smaller fraction of that extent. Python raw reproduces almost the same ratio. Native deduplication brings the distribution substantially closer to Python unique.

These radial quantities are descriptive, not invariant to every permissible embedding rotation or a substitute for density/topology analysis. They support an outlier-driven visual compression effect; they do not prove every central neighborhood has the same density.

In Python unique and explicitly deduplicated native fits, identical input rows map to exactly the same coordinate. In raw fits they can move separately: native raw duplicate displacement from the group's first representative has median 0.0015, 95th percentile 0.0392, but maximum **20.78** coordinate units. Python raw's maximum is **37.13**. Rare extreme duplicate separation is therefore not uniquely a CUDA phenomenon. Tied neighbors can select different subsets of a large duplicate group.

For visualization, include full-range and robust-percentile views, display numeric ranges, and show isolated components separately. Avoid treating absolute position, orientation, or one autoranged screenshot as a quality score.

## 4. Is ANN quality failing at 50k?

We checked the saved native ANN lists against brute-force Euclidean neighbors for **1,000 reproducibly selected query rows** per size, using the full input matrix as the search pool. Each returned neighbor within the exact 14th non-self distance, plus `1e-5` numerical tolerance, counts as correct; this is a **tie-aware distance recall**, not strict index recall. Exact duplicate ties otherwise make arbitrary neighbor IDs look wrong.

| Sample | Overall recall@14 | Queries with duplicated vectors | Queries with nonduplicated vectors |
|---|---:|---:|---:|
| 15k | 0.9819 | 0.9988 | 0.9795 |
| 50k | 0.9671 | 0.9967 | 0.9591 |

Recall declines somewhat at 50k, but this is **not an ANN collapse**. The duplicate-heavy queries are especially accurate under the distance criterion: accurately retrieving only copies is precisely how a duplicate group can lose external connectivity.

Native NN-Descent differs from PyNNDescent in candidate initialization and width, so the remaining deduplicated quality gap may still include ANN differences. This check does not compare both libraries' graph membership weights, and no full exact-neighbor fit was run. Existing ANN acceptance fixtures are smaller/lower-dimensional than this 50k × 128 input (`tests/test_standard_quality.rs:403–508`, `tests/fixtures/ann_campaign_reference.json`).

## 5. What source inspection did and did not find

The forward SGD formulas and major defaults inspected against installed umap-learn 0.5.12 do not reveal a gross attraction/repulsion scaling error:

- `min_dist=0`, `spread=1` curve is covered by `tests/test_standard_umap.rs:107–131`, with reference coefficients approximately `a=1.93281`, `b=0.79049`; native fitting is in `src/standard_umap/config.rs:108–201`.
- Default epochs, graph-pruning threshold and `max_weight/weight` sampling intervals follow the reference rules (`estimator.rs:189–201`, `:652–697`). Every full fit here exceeds 10k actual fitted rows and uses 200 epochs.
- Attraction/repulsion coefficients, clipping, self-negative handling and sampling clocks were traced in `src/kernels/standard_umap/layout.rs:47–59,116–128,189–220` against reference `umap/layouts.py`.
- Spectral preprocessing follows pruned graph → max-absolute scaling plus noise → per-axis min/max scaling (`init.rs:63–73`, `:879–943`), matching the reference sequence.

This is **not** a claim of bitwise or complete training parity. CUDA uses order-dependent atomic updates, neighbor/RNG sequences differ, and initialization uses a different eigensolver. None of the existing tests was rerun solely to claim formula equivalence.

Spectral localization around weakly connected duplicate groups remains plausible, especially given the independently measured solver slowdown. However, final raw random and spectral metrics are nearly the same. That argues against spectral initialization alone being the primary cause of the final geometry difference. Initial eigenvectors/coordinates and residuals were not captured, and no fallback warning was observed.

## Recommendations and remaining verification

1. **Correct the experiment contract first:** fit both estimators on one explicitly deduplicated matrix and map back identically; also retain a separate raw-observation comparison. Keep the production duplicate contract explicit.
2. Add repeated fixed-seed quality runs on this real 128-dimensional regime, with `min_dist=0`, large duplicate groups and at least 40–50k observations. Report uncertainty rather than demanding identical scatterplots.
3. Measure both ANN graphs on matched unique inputs, including tie-aware recall, fuzzy weights and connected components. If a significant residual gap persists, freeze one graph and isolate initialization from layout.
4. Fix the spectral performance issue without accepting worse neighborhood quality as the tradeoff. Random initialization removed the delay but left the duplicate-induced quality gap intact.
5. Improve the plot's evaluation views and honor `--device`; no plot changes were applied as part of this findings-only request.

Recompute the retained embedding metrics with:

```sh
uv run --no-sync python artifacts/comparison-investigation/quality.py
```

The script uses the saved embeddings and ANN exports from the performance investigation. No external data was uploaded. Findings apply to these local samples and measured runs; there are no multi-seed confidence intervals, supervised topic-label quality results, initial spectral localization measurements, or full-reference training-parity claims.
