# HDBSCAN Algorithm Port (stage by stage)

Companion to `hdbscan_impl.md` (crates, API, phases, test strategy). This document walks the
upstream `fast_hdbscan` 0.3.2 pipeline (commit `43c573ba61043a6ad6f10b396656bfaa7aaeead8`) for
the in-scope configuration: Euclidean, no sample weights, `algorithm="boruvka"`, no
constraints, no semi-supervision. For each stage it gives the upstream behavior, the parts of the
Python code that fix the semantics, and how the stage is ported.

Upstream paths are relative to `fast_hdbscan/`. Rust paths are relative to `crates/`.

## Pipeline overview

```text
 X: n×d f32 (raw)
   │ S1 validate config + input
   │ S2 split finite rows, handle n_finite < 2
   ▼
 X_f: m×d f32 ───────────────────────────── device (Burn/CubeCL) ─────────────┐
   │ S3 kNN → core distances² (f32), neighbors                                 │
   │ S4 KD-tree build (host, phase 4 only)                                     │
   │ S5 Boruvka init from kNN (host DSU)                                       │
   │ S6 Boruvka rounds: candidate kernel (device) → component winners → DSU    │
   ▼                                                                           │
 MST (m-1 edges, squared f32 → f64 sqrt) ─────────────────── host ◄─────────────┘
   │ S7 finalize + lexsort MST
   │ S8 single-linkage tree
   │ S9 condensed tree
   │ S10 persistence simplification (optional)
   │ S11 cluster tree + selection (EOM / leaf, epsilon search)
   │ S12 labels + membership strengths
   │ S13 remap non-finite rows into labels / trees
   ▼
 FittedHDBSCAN ── S14 dbscan_clustering(epsilon) on the single-linkage tree
```

| Stage | Crate / module | Runs on |
|---|---|---|
| S1, S2, S13 | `fast-hdbscan-core/src/{config.rs, finite.rs}` | host |
| S3 | `fast-cubecl-common` `NeighborOps::exact_neighbors` | device |
| S4 | `fast-hdbscan-core/src/kdtree.rs` | host |
| S5, S7 | `fast-hdbscan-core/src/{dsu.rs, mst.rs}` | host |
| S6 | `fast-hdbscan/src/kernels/boruvka.rs` (`HdbscanOps`) + `mst.rs` | device + host |
| S8 | `fast-hdbscan-core/src/linkage.rs` | host |
| S9, S10 | `fast-hdbscan-core/src/condense.rs` | host |
| S11 | `fast-hdbscan-core/src/select.rs` | host |
| S12, S14 | `fast-hdbscan-core/src/labels.rs` | host |
| orchestration | `fast-hdbscan/src/estimator.rs` | host |

Everything after S7 is O(n log n) or O(n) sequential tree work, which is what upstream does in
Numba as well. Only the distance-heavy stages (S3, S6) go to the device.

## Numerical and indexing conventions

These conventions decide whether host stages reproduce upstream exactly when given the same
MST, so they hold everywhere in the port.

| Quantity | Upstream type and origin | Port |
|---|---|---|
| Features | `build_kdtree` casts to `float32` (`numba_kdtree.py:481`) | `f32` input. The Python wrapper casts `float64` to `float32` and raises `ValueError` when a finite `float64` value overflows `f32` |
| Pair distance | squared `float32` `rdist` (`numba_kdtree.py:539`) | squared `f32`, accumulated feature by feature in index order |
| Core distance | squared `float32` during search; output `np.sqrt` in `float32` (`boruvka.py:647`) | squared `f32` internally; `f32::sqrt` on output |
| MST weight | squared `float32` widened to `float64`, then `np.sqrt` in `float64` (`boruvka.py:646`) | `(w_sq as f64).sqrt()` |
| Linkage distance | `float64` | `f64` |
| λ | `1/d` computed in `float64`, stored as `float32` (`cluster_trees.py:224,244`) | `hdbscan_lambda(d: f64) -> f32` |
| Condensed `child_size` | `float32` | `u32` in memory; converted to `f32` wherever upstream does arithmetic or comparisons on it (exact for counts up to 2^24) |
| Stability scores | `float32` dict accumulation in row order (`cluster_trees.py:657-669`) | `f32`, same order |
| Membership strength | `float32` (`cluster_trees.py:1013`) | `f32` |
| Point/node ids | `intp`; device index tensors `int32` | `u32` on host, `i32` tensors on device; `n ≤ i32::MAX` validated in S1 |

Determinism: every reduction breaks ties by a total order. Distances compare with
`f32::total_cmp`/`f64::total_cmp`, which is valid because NaN is excluded in S1/S2. The second
key is always the smaller index. Upstream breaks ties by traversal order, which depends on the
KD-tree. On inputs with exact distance ties our trees can therefore legitimately differ, and the
parity tests treat tie datasets separately (`hdbscan_impl.md`, Testing strategy).

---

## S1. Configuration and input validation

**Upstream.** `fast_hdbscan()` (`hdbscan.py:254-275`) rejects non-integer or non-positive
`min_samples`/`min_cluster_size` and non-`np.floating` or negative epsilon/persistence. An
invalid `cluster_selection_method` raises `ValueError` only later, during selection
(`hdbscan.py:558`). `min_samples=None` resolves to `min_cluster_size` (`hdbscan.py:308`).
`max_cluster_size` defaults to `np.inf`.

```python
if (
    (not (np.issubdtype(type(min_samples), np.integer) or min_samples is None))
    or not np.issubdtype(type(min_cluster_size), np.integer)
    or (min_samples is not None and min_samples <= 0)
    or min_cluster_size <= 0
):
    raise ValueError("Min samples and min cluster size must be positive integers!")
```

**Port.** Types handle most of this in Rust: `ClusterSelectionMethod` is an enum and
`max_cluster_size: Option<usize>` stands for `np.inf`. `HdbscanConfig::validate` covers the rest:

```rust
impl HdbscanConfig {
    pub fn validate(&self) -> Result<(), HdbscanError> {
        if self.min_cluster_size == 0 {
            return Err(HdbscanError::InvalidConfig("min_cluster_size must be positive".into()));
        }
        if self.min_samples == Some(0) {
            return Err(HdbscanError::InvalidConfig("min_samples must be positive".into()));
        }
        for (name, value) in [
            ("cluster_selection_epsilon", self.cluster_selection_epsilon),
            ("cluster_selection_persistence", self.cluster_selection_persistence),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(HdbscanError::InvalidConfig(format!(
                    "{name} must be finite and non-negative"
                )));
            }
        }
        Ok(())
    }

    pub fn resolved_min_samples(&self) -> usize {
        self.min_samples.unwrap_or(self.min_cluster_size)
    }
}
```

Input checks, done once in the estimator before any upload: non-empty, `d > 0`, rectangular
rows, `n ≤ i32::MAX`, and `n·d` fits `usize` and the device allocation limits. NaN/±inf are
*not* errors. S2 handles them. The Python wrapper repeats upstream's type checks
(`np.issubdtype(type(eps), np.floating)`), so `cluster_selection_epsilon=0` (an `int`) raises
`ValueError` as it does upstream, and it raises `NotImplementedError` for out-of-scope
parameters (`sample_weight`, `metric != "euclidean"`, `algorithm="kruskal"`, `knn_k`,
`cannot_link*`, `semi_supervised=True`). It accepts `reproducible` and ignores it, because the
port is always deterministic.

Divergence: upstream accepts `epsilon=np.inf`. The port rejects it, as `hdbscan_impl.md`
specifies.

## S2. Finite-row split and degenerate inputs

**Upstream.** `HDBSCAN.fit` (`hdbscan.py:763-784`) keeps only rows with all features finite. It
records `finite_index` and the `outliers` list, runs the pipeline on the clean rows and remaps
afterwards (S13). Inside `fast_hdbscan()`, fewer than two rows short-circuits
(`hdbscan.py:277-304`): all labels `-1`, probabilities `0`, empty trees and MST,
`neighbors` of shape `(n, 0)`, `core_distances = inf`.

```python
self._all_finite = np.all(np.isfinite(X))
if ~self._all_finite:
    finite_index = np.where(np.isfinite(X).sum(axis=1) == X.shape[1])[0]
    clean_data = X[finite_index]
    internal_to_raw = {x: y for x, y in zip(range(len(finite_index)), finite_index)}
    outliers = list(set(range(X.shape[0])) - set(finite_index))
```

**Port.** `fast-hdbscan-core/src/finite.rs`:

```rust
pub struct FiniteSplit {
    /// internal row → raw row, ascending (`finite_index`).
    pub finite_index: Vec<u32>,
    /// raw rows with any non-finite feature, ascending (`outliers`).
    pub outliers: Vec<u32>,
}

pub fn split_finite(data: &[f32], n_features: usize) -> FiniteSplit {
    let mut split = FiniteSplit { finite_index: Vec::new(), outliers: Vec::new() };
    for (row, values) in data.chunks_exact(n_features).enumerate() {
        let target = if values.iter().all(|v| v.is_finite()) {
            &mut split.finite_index
        } else {
            &mut split.outliers
        };
        target.push(row as u32);
    }
    split
}
```

`outliers` is sorted ascending. Upstream builds its list from a set difference
(`list(set(range(n)) - set(finite_index))`), which iterates in hash-table order. That order is
often, but not always, ascending: a quick CPython check found unsorted lists in 76 of 200 random
cases. The order shows up in S13 (outlier rows appended to the trees) and so in the
representative-based numbering of `dbscan_clustering` after remapping. Sorting is a deliberate
determinism divergence. Remap fixtures record upstream's `outliers` order, and the remap tests
either feed that order in through a test-only entry point or normalize both trees before
comparing. The compacted matrix
upload happens only when `outliers` is non-empty. Otherwise the caller's buffer or tensor is
used directly, as UMAP does.

Degenerate cases:

- `n_finite == 0`: return `HdbscanError::InvalidInput`. Upstream's `check_array` raises on an
  empty clean matrix.
- `n_finite == 1`: return the `fast_hdbscan` degenerate result for the finite part, then remap
  labels/probabilities only. The trees stay empty. Upstream crashes in this case
  (`remap_condensed_tree` reads `tree[0]` of an empty tree, `hdbscan.py:144`). We document the
  divergence and test it.
- `n_finite == 2`, or `n_finite ≤ min_samples`: no special case. S3 produces infinite core
  distances and the rest of the pipeline handles them (see S3).

## S3. Core distances (exact kNN)

**Upstream.** With no weights (`boruvka.py:580-601`) it queries the KD-tree for
`k = min_samples + 1` neighbors *including self*, in squared space (`output_rdist=True`). The
core distance is the last column. For `min_samples == 1`, core distances are zero and `k = 2`
is used only to seed S5. Missing neighbors (`k > n`) stay at the `inf`/`-1` initialization of
`parallel_tree_query` (`numba_kdtree.py:775-778`). The returned neighbors drop column 0
(`neighbors[:, 1:]`) and the returned core distances are `np.sqrt`.

```python
if min_samples > 1:
    distances, neighbors = parallel_tree_query(tree, tree.data, k=min_samples + 1, output_rdist=True)
    core_distances = distances.T[-1]
else:
    core_distances = np.zeros(tree.data.shape[0], dtype=np.float32)
    distances, neighbors = parallel_tree_query(tree, tree.data, k=2, output_rdist=True)
...
return all_edges, neighbors[:, 1:], np.sqrt(core_distances)
```

When `n ≤ min_samples` every core distance is `inf`. Every MRD edge is then `inf`, the whole
hierarchy collapses at λ = 0, and all labels are `-1`. This was checked against 0.3.2 with
`n=8, min_samples=10`: the MST has 7 `inf` edges and labels are all `-1`.

**Port.** Reuse the exact-neighbor kernel (`NeighborOps::exact_neighbors`, currently
`fast-umap/src/kernels/standard_umap/mod.rs:25-86`) with `exclude_self = true` and
`k = min_samples`. Excluding self and asking for `min_samples` neighbors gives the same k-th
distance as upstream's `min_samples + 1` including self, including on duplicates (distance 0
either way). The kernel currently takes the `sqrt` (`mod.rs:81-84`). The extraction adds a
`squared: u32` launch argument so HDBSCAN reads squared `f32` values without a lossy square/sqrt
round trip. UMAP passes false and preserves its `f32::MAX` initialization/output behavior;
HDBSCAN uses true and `+inf` initialization:

```rust
// fast-cubecl-common/src/kernels/neighbors.rs (moved kernel, one new flag)
#[cube(launch)]
fn exact_neighbors_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    k: u32,
    exclude_self: u32,
    query_offset: u32,
    squared: u32,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    // HDBSCAN mode uses (+inf, reference_rows); UMAP mode retains its old sentinel.
    // … unchanged insertion-sorted top-k with (distance, index) tie-break …
    if squared == 0 {
        for slot in 0..k_usize {
            let output = row * k_usize + slot;
            distances[output] = F::sqrt(distances[output]);
        }
    }
}
```

UMAP keeps passing `squared = false` through its legacy-equivalent sentinel behavior, so outputs
including overflow cases remain unchanged.

HDBSCAN mode uses `+inf` instead of UMAP's existing `f32::MAX` (`mod.rs:44`) slot initialization.
With `f32::MAX`, a squared distance that overflows to `inf` (finite features around 1e19 or
larger) can never displace the sentinel. The slot then keeps index `reference_rows`, and the
core distance becomes the finite `f32::MAX`, which is wrong. With `+inf` and the existing
`(distance, index)` tie arm, `inf` candidates fill the slots and the core distance is `inf`, as
upstream computes it. Upstream leaves those slots at `-1` (strict `d < heap_p[0]`), so the
`neighbors` output can differ on overflowing inputs. That is documented, not replicated for
HDBSCAN. Any sentinel index that remains (`k` larger than the available references) decodes to
`-1` on the host. UMAP retains its previous behavior in all cases.

Host orchestration in `fast-hdbscan/src/estimator.rs`:

```rust
struct CoreDistances {
    /// Squared core distance per point, `f32::INFINITY` when `n ≤ min_samples`.
    core_sq: Vec<f32>,
    /// `n × min(min_samples, n - 1)` neighbor ids padded with -1 to `min_samples` columns.
    neighbors: Vec<i32>,
    /// Squared kNN distances, same layout, padded with +inf.
    knn_sq: Vec<f32>,
}

fn core_distances(x: &Tensor<2>, min_samples: usize) -> Result<CoreDistances, HdbscanError> {
    let n = x.dims()[0];
    let k = min_samples.min(n - 1); // n ≥ 2 after S2
    let (idx, dist_sq) = exact_neighbors_blocked(x, k, /*exclude_self=*/ true, /*squared=*/ true)?;
    let core_sq = if min_samples >= n {
        vec![f32::INFINITY; n]
    } else if min_samples == 1 {
        vec![0.0; n]
    } else {
        last_column(&dist_sq, k)
    };
    Ok(CoreDistances { core_sq, neighbors: pad(idx, k, min_samples, -1), knn_sq: pad(dist_sq, k, min_samples, f32::INFINITY) })
}
```

`exact_neighbors_blocked` reuses UMAP's row-block orchestration (`query_offset`) so that output
size stays within device limits. For `min_samples == 1` the kernel runs with `k = 1`. That
gives the nearest neighbor, which S5 needs. It matches upstream's `k=2` query minus self.

Divergences to note in docs and tests:

- The `neighbors` attribute: upstream drops column 0 on the assumption that it is self. With
  duplicate points, upstream's column 0 can be a duplicate, and self then appears in the
  output. Ours never contains self. Core distances agree regardless.
- Padding: upstream pads missing neighbors with `-1` (index) and `inf` (distance), and so do we.

Performance: `exact_neighbors` is O(n²·d) with one unit per query. It is the same kernel UMAP
uses for small inputs. A KD-tree kNN kernel (S4 traversal with a k-heap instead of a single
best) is phase 4 work and shares the traversal skeleton from S6b.

## S4. KD-tree build (host, phase 4)

**Upstream.** `build_kdtree(data, leaf_size=40)` (`numba_kdtree.py:478-521`) builds an implicit
binary heap. The tree has `n_levels = floor(log2(max(1, (n-1)/leaf_size))) + 1` levels and
`n_nodes = 2^n_levels - 1` nodes. Node `i` has children `2i+1` and `2i+2`. Each node stores
`idx_start`, `idx_end` and `is_leaf`, plus per-feature bounding boxes (`node_bounds[0]` = min,
`node_bounds[1]` = max). A split takes the dimension of maximum spread and partitions with
introselect at `n_mid = n_points // 2` (`numba_kdtree.py:426-473`).

```python
if 2 * node + 1 >= is_leaf_array.shape[0]:
    is_leaf_array[node] = True
elif idx_end - idx_start < 2:
    is_leaf_array[node] = True
else:
    axis = _find_node_split_dim(data, idx_array, idx_start, idx_end)
    _introselect(data, idx_array, axis, idx_start, idx_end, idx_start + n_mid)
    _recursive_build_tree(..., idx_start, idx_start + n_mid, 2 * node + 1)
    _recursive_build_tree(..., idx_start + n_mid, idx_end, 2 * node + 2)
```

**Port.** `fast-hdbscan-core/src/kdtree.rs`, which is backend-free and uploaded as flat
buffers. The build is iterative (explicit stack, no recursion). The partition uses
`select_nth_unstable_by` with key `(value, original index)`, so the tree is deterministic.
Introselect's order is not part of upstream's contract.

```rust
pub const LEAF_SIZE: usize = 40;
/// Upper bound on `n_levels` for n ≤ i32::MAX; sizes the device traversal stack.
pub const MAX_KD_LEVELS: usize = 32;

pub struct KdTree {
    pub n_levels: u32,
    pub idx_array: Vec<i32>,     // permutation of 0..n
    pub idx_start: Vec<i32>,     // per node
    pub idx_end: Vec<i32>,
    pub is_leaf: Vec<u32>,       // 0/1, u32 for device upload
    pub bounds: Vec<f32>,        // [2][n_nodes][d]: min then max
}

pub fn build_kdtree(data: &[f32], n: usize, d: usize) -> KdTree {
    let n_levels = (((n.saturating_sub(1)) as f64 / LEAF_SIZE as f64).max(1.0).log2() as u32) + 1;
    let n_nodes = (1usize << n_levels) - 1;
    let mut tree = KdTree::allocate(n, d, n_levels, n_nodes);
    let mut stack = vec![(0usize, 0usize, n)];
    while let Some((node, start, end)) = stack.pop() {
        tree.init_node(data, d, node, start, end); // bounds over idx_array[start..end]
        if 2 * node + 1 >= n_nodes || end - start < 2 {
            tree.is_leaf[node] = 1;
            continue;
        }
        let axis = tree.max_spread_dim(data, d, start, end);
        let mid = start + (end - start) / 2;
        tree.idx_array[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a as usize * d + axis]
                .total_cmp(&data[b as usize * d + axis])
                .then(a.cmp(&b))
        });
        stack.push((2 * node + 2, mid, end));
        stack.push((2 * node + 1, start, mid));
    }
    tree
}
```

`n_levels ≤ MAX_KD_LEVELS` is asserted. With `n ≤ i32::MAX` and a leaf size of 40, `n_levels`
is at most 27. The point-to-box lower bound mirrors `point_to_node_lower_bound_rdist`
(`numba_kdtree.py:566-575`) and is computed on the device (S6b).

Phase 3 skips S4 entirely: the brute-force candidate kernel needs no tree.

## S5. Boruvka initialization from kNN

**Upstream.** `initialize_boruvka_from_knn` (`boruvka.py:486-524`): for each point `i` it scans
kNN columns `j ≥ 1` and takes the first neighbor `k` with `core[i] >= core[k]`. The edge
weight is `max(core[i], d²(i,k))`. For `min_samples > 1`, every queried neighbor lies within
`i`'s core radius, so the weight is `core[i]`, the smallest MRD any edge from `i` can have. For
`min_samples == 1`, `core = 0` and the edge is the nearest-neighbor edge with weight
`d²(i,k)`, which is again minimal. Either way, the edge is a valid Boruvka edge for `i`'s
singleton component. The edges are applied
sequentially in point order and merged with `ds_union_by_rank` when their endpoints are in
different components. Weights are squared. A point whose neighbors all have larger core
distances gets no initial edge.

```python
for i in numba.prange(knn_indices.shape[0]):
    for j in range(1, knn_indices.shape[1]):
        k = np.int32(knn_indices[i, j])
        if core_distances[i] >= core_distances[k]:
            edge_weight = max(core_distances[i], knn_distances[i, j])
            component_edges[i] = (np.float64(i), np.float64(k), np.float64(edge_weight))
            break
...
for edge in component_edges:
    if edge[0] < 0:
        continue
    from_component = ds_find(disjoint_set, np.int32(edge[0]))
    to_component = ds_find(disjoint_set, np.int32(edge[1]))
    if from_component != to_component:
        result[result_idx] = (edge[0], edge[1], edge[2]); result_idx += 1
        ds_union_by_rank(disjoint_set, from_component, to_component)
```

**Port.** Host code in `fast-hdbscan-core/src/mst.rs`. Our neighbor rows exclude self, so they
start at column 0. Padded `-1` neighbors are skipped. Upstream instead indexes
`core_distances[-1]`, a Numba wraparound that only happens in the all-`inf` case and only affects
which `inf` edges are chosen.

```rust
pub struct MstEdge { pub src: u32, pub dst: u32, pub weight_sq: f32 }

pub fn init_from_knn(cd: &CoreDistances, k: usize, dsu: &mut RankDsu) -> Vec<MstEdge> {
    let n = cd.core_sq.len();
    let mut edges = Vec::with_capacity(n);
    for i in 0..n {
        let core_i = cd.core_sq[i];
        let row = i * k;
        let pick = (0..k).find_map(|j| {
            let nb = cd.neighbors[row + j];
            (nb >= 0 && core_i >= cd.core_sq[nb as usize])
                .then(|| (nb as u32, core_i.max(cd.knn_sq[row + j])))
        });
        if let Some((dst, w)) = pick {
            let (a, b) = (dsu.find(i as u32), dsu.find(dst));
            if a != b {
                edges.push(MstEdge { src: i as u32, dst, weight_sq: w });
                dsu.union_roots(a, b);
            }
        }
    }
    edges
}
```

`RankDsu` (`dsu.rs`) reproduces `ds_union_by_rank` exactly: when ranks tie, the *first*
argument's root becomes the parent. This matters in S12 and S14, where representative ids
decide label assignment and numbering. Upstream's `ds_find` tuple assignment
(`disjoint_set.py:24`) never actually compresses paths (it writes `parent[p] = parent[p]`).
Path compression does not change roots, so our `find` uses path halving and stays
observationally identical.

```rust
pub struct RankDsu { parent: Vec<u32>, rank: Vec<u8> }
impl RankDsu {
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = p;
        }
        x
    }
    /// `ds_union_by_rank(x, y)` on roots: x wins ties.
    pub fn union_roots(&mut self, mut x: u32, mut y: u32) {
        if x == y { return; }
        if self.rank[x as usize] < self.rank[y as usize] { std::mem::swap(&mut x, &mut y); }
        self.parent[y as usize] = x;
        if self.rank[x as usize] == self.rank[y as usize] { self.rank[x as usize] += 1; }
    }
    pub fn union(&mut self, x: u32, y: u32) { let (a, b) = (self.find(x), self.find(y)); self.union_roots(a, b) }
}
```

When S5 has merged everything into one component (common when `min_samples` is large relative
to cluster sizes), S6 does not run.

## S6. Boruvka rounds

**Upstream.** `parallel_boruvka` loop (`boruvka.py:604-640`):

```python
n_components = len(np.unique(point_components))
while n_components > 1:
    candidate_distances, candidate_indices = boruvka_tree_query_reproducible(
        tree, node_components, point_components, core_distances, block_size)
    new_edges = merge_components(components_disjoint_set, candidate_indices,
                                 candidate_distances, point_components)
    n_components -= len(new_edges)
    update_component_vectors(tree, components_disjoint_set, node_components, point_components)
    all_edges = np.vstack((all_edges, new_edges))
```

A round has three parts. S6a refreshes the component labels, S6b searches candidates per point
on the device, and S6c reduces the candidates to one winner per component and merges on the
host.

### S6a. Component vectors

**Upstream.** `update_component_vectors` (`boruvka.py:94-124`) sets `point_components[i] =
find(i)` for every point. It then walks the KD nodes bottom-up: a node gets its component label
when every point in it (leaf) or both children (internal) share one component. Otherwise the
node keeps its old value, which upstream initializes to `-1` and never resets.

```python
for i in range(tree.idx_start.shape[0] - 1, -1, -1):
    if is_leaf:
        candidate_component = point_components[tree.idx_array[idx_start]]
        for j in range(idx_start + 1, idx_end):
            if point_components[tree.idx_array[j]] != candidate_component:
                break
        else:
            node_components[i] = candidate_component
    else:
        if node_components[left] == node_components[right]:
            node_components[i] = node_components[left]
```

Labels never go stale even though nodes are never reset. Components only grow, so a
homogeneous node stays homogeneous, and the bottom-up pass rewrites its label with the current
root every round. A node whose check fails has never been homogeneous and still holds `-1`.
An internal node with two `-1` children gets `-1` again, which is harmless. A faithful port is
therefore safe.

**Port.** Host code, O(n + n_nodes) per round, uploaded as two `i32` tensors. The
brute-force kernel (phase 3) needs only `point_components`.

```rust
pub fn update_component_vectors(
    dsu: &mut RankDsu, tree: Option<&KdTree>,
    point_components: &mut [i32], node_components: &mut [i32],
) {
    for (i, c) in point_components.iter_mut().enumerate() { *c = dsu.find(i as u32) as i32; }
    let Some(tree) = tree else { return };
    for node in (0..tree.idx_start.len()).rev() {
        if tree.is_leaf[node] == 1 {
            let (s, e) = (tree.idx_start[node] as usize, tree.idx_end[node] as usize);
            let first = point_components[tree.idx_array[s] as usize];
            if tree.idx_array[s + 1..e].iter().all(|&p| point_components[p as usize] == first) {
                node_components[node] = first;
            }
        } else {
            let (l, r) = (2 * node + 1, 2 * node + 2);
            if node_components[l] == node_components[r] { node_components[node] = node_components[l]; }
        }
    }
}
```

Upstream leaves leaves empty only when `n < 2`, and S2 excludes that. A leaf with
`idx_start == idx_end` would make the port index out of bounds, so `KdTree::validate` asserts
`idx_end > idx_start` for every leaf.

### S6b. Candidate search kernels (`HdbscanOps`)

**Upstream.** `component_aware_query_recursion` (`boruvka.py:154-287`) runs for each point `i`
and finds the nearest point (in MRD²) of a different component. It prunes with three tests:
the box lower bound against the point's current best (1a); the box lower bound, or `core[i]`,
against a shared per-component best (1b); and nodes that are homogeneous in `i`'s own component
(1c). The leaf update is guarded by `core[idx] < component_best`.

```python
if dist_lower_bound > heap_p[0]:
    return
elif dist_lower_bound > component_nearest_neighbor_dist[0] or current_core_distance > component_nearest_neighbor_dist[0]:
    return
elif node_components[node] == current_component:
    return
elif is_leaf:
    for i in range(idx_start, idx_end):
        idx = tree.idx_array[i]
        if point_components[idx] != current_component and core_distances[idx] < component_nearest_neighbor_dist[0]:
            d = max(rdist(point, tree.data[idx]), current_core_distance, core_distances[idx])
            if d < heap_p[0]:
                heap_p[0] = d; heap_i[0] = idx
                if d < component_nearest_neighbor_dist[0]:
                    component_nearest_neighbor_dist[0] = d
else:
    # visit nearer child first
```

Because of the component-level pruning (1b), a single point's result may be non-optimal or
missing (`inf`/`-1`). Only the per-component minimum is exact. Reproducible mode
(`boruvka.py:405-471`) makes the shared bound race-free by processing points in blocks.

**Port contract.** Freeze one signature before implementing Phase 3. The trait accepts only immutable
data inputs and the optional tree; component pruning is an implementation-internal KD-backend
optimization, not a fusion-op argument. The CubeBackend implementation allocates its atomic
per-root scratch and resets it before each launch when pruning is enabled. The brute-force variant
does not use this scratch. `DeviceKdTree` is a CubeCL fusion input type with its required
`ExtensionType` derive and kernel/fusion metadata implemented alongside this trait (not a
host-only opaque value):

```rust
// fast-hdbscan/src/backend.rs
#[backend_extension(…)]
pub trait HdbscanOps: Backend {
    /// For each row i, the lexicographically smallest (mrd², j) over j with
    /// components[j] != components[i], where mrd² = max(‖x_i − x_j‖², core_i, core_j).
    /// Returns (distance_sq: [n] f32, neighbor: [n] i32).
    /// No-candidate index sentinel is n on device; orchestration decodes it to -1.
    /// KD implementation may use internal component pruning; only component minima are consumed
    /// by the exact Boruvka merge.
    fn boruvka_candidates(
        data: FloatTensor<Self>,         // [n, d] f32
        core_sq: FloatTensor<Self>,      // [n]
        point_components: IntTensor<Self>, // [n] i32
        tree: Option<DeviceKdTree<Self>>,  // None → brute force
    ) -> (FloatTensor<Self>, IntTensor<Self>);
}
```

`DeviceKdTree` holds uploaded `idx_array`, `idx_start`, `idx_end`, `is_leaf`, `bounds`, and
per-round `node_components`. It is reused across rounds: `node_components` and
`point_components` change. Every device kernel uses `n` (not `-1`) as
the no-neighbor index sentinel; host readback converts exactly `n` to `-1` before merge logic.

**Variant 1: brute-force tiles (phase 3, also the device oracle).** One unit per
`(query row, reference tile)` writes the tile winner. A second launch reduces the
`ceil(n/T)` tile winners per row. Temporary storage is `n·ceil(n/T)` pairs, chunked over query
row blocks to stay within the allocation limit. There is no pruning, so every row's answer is
exact.

```rust
#[cube(launch)]
fn mrd_tile_candidates<F: Float>(
    data: &Tensor<F>,
    core_sq: &Tensor<F>,
    components: &Tensor<i32>,
    row_offset: u32,
    tile: u32,
    out_dist: &mut Tensor<F>,   // [rows_in_block, tiles]
    out_index: &mut Tensor<i32>,
    #[define(F)] _dtype: ElemType,
) {
    let local_row = ABSOLUTE_POS_X as usize;
    let tile_id = ABSOLUTE_POS_Y as usize;
    let n = data.shape(0);
    let d = data.shape(1);
    let tiles = out_dist.shape(1);
    let i = row_offset as usize + local_row;
    if i < n && local_row < out_dist.shape(0) && tile_id < tiles {
        let core_i = core_sq[i];
        let comp_i = components[i];
        let mut best = F::new(f32::INFINITY);
        let mut best_j = n; // sentinel; host decodes n to -1
        let start = tile_id * tile as usize;
        for q in 0..tile as usize {
            let j = start + q;
            if j < n && components[j] != comp_i {
                let mut r = F::new(0.0);
                for f in 0..d {
                    let diff = data[i * d + f] - data[j * d + f];
                    r += diff * diff;
                }
                r = F::max(F::max(r, core_i), core_sq[j]);
                if r < best || (r == best && j < best_j) {
                    best = r;
                    best_j = j;
                }
            }
        }
        out_dist[local_row * tiles + tile_id] = best;
        out_index[local_row * tiles + tile_id] = best_j as i32;
    }
}

#[cube(launch)]
fn reduce_tile_candidates<F: Float>(
    tile_dist: &Tensor<F>, tile_index: &Tensor<i32>,
    out_dist: &mut Tensor<F>, out_index: &mut Tensor<i32>, row_offset: u32,
    #[define(F)] _dtype: ElemType,
) {
    let local_row = ABSOLUTE_POS_X as usize;
    let tiles = tile_dist.shape(1);
    if local_row < tile_dist.shape(0) {
        let mut best = tile_dist[local_row * tiles];
        let mut best_j = tile_index[local_row * tiles];
        for t in 1..tiles {
            let dist = tile_dist[local_row * tiles + t];
            let j = tile_index[local_row * tiles + t];
            if dist < best || (dist == best && j < best_j) { best = dist; best_j = j; }
        }
        let i = row_offset as usize + local_row;
        out_dist[i] = best;
        out_index[i] = best_j;
    }
}
```

The `inf`-core case (`n ≤ min_samples`) needs care. Every candidate has `r = inf`, and
`inf < inf` is false, so the strict-less test alone would never pick one. The `r == best && j <
best_j` arm accepts the first `inf` candidate because `best_j` starts at the `n` sentinel. So
every row still gets a real candidate and the MST is a valid spanning tree with `inf` weights,
matching upstream's weight multiset. The CubeCL `f32::INFINITY` literal must be checked on both
runtimes. If either rejects it, pass `inf` as a scalar argument, as the architect report does.

Launch shape: `CubeDim::new_2d(16, 16)` clamped by `legal_cube_dim`. Tile `T = 256` is
benchmarked against 128 and 512 on both runtimes. Cost is O(n²·d) per round with
⌈log₂ n⌉ rounds at most, which is acceptable for the oracle and n up to ~10⁵ on CUDA.

**Variant 2: KD-tree traversal (phase 4).** One unit per query row walks the host-built tree
with a private stack of node ids. The stack length is the compile-time constant
`MAX_KD_LEVELS + 1` (near-first DFS keeps at most one pending sibling per level). Bounds are
recomputed when a node is popped instead of being stored on the stack.

```rust
#[cube(launch)]
fn kd_boruvka_candidates<F: Float>(
    data: &Tensor<F>, core_sq: &Tensor<F>, point_components: &Tensor<i32>,
    idx_array: &Tensor<i32>, idx_start: &Tensor<i32>, idx_end: &Tensor<i32>,
    is_leaf: &Tensor<u32>, bounds: &Tensor<F>, node_components: &Tensor<i32>,
    component_bound: &mut Tensor<Atomic<u32>>, // f32 bits per component root, ordered as u32
    use_component_bound: u32,
    out_dist: &mut Tensor<F>, out_index: &mut Tensor<i32>,
    #[define(F)] _dtype: ElemType,
) {
    let i = ABSOLUTE_POS_X as usize;
    let n = data.shape(0);
    let d = data.shape(1);
    let n_nodes = idx_start.shape(0);
    if i < n {
        let comp_i = point_components[i];
        let core_i = core_sq[i];
        let mut best = F::new(f32::INFINITY);
        let mut best_j = n;
        let mut stack = Array::<u32>::new(STACK_LEN); // comptime const = MAX_KD_LEVELS + 1
        stack[0] = 0;
        let mut sp = 1usize;
        while sp > 0 {
            sp -= 1;
            let node = stack[sp] as usize;
            let lb = F::max(box_lower_bound(bounds, n_nodes, d, node, data, i), core_i);
            // 1a: strict >, so equal-distance candidates are still visited (deterministic ties).
            let mut skip = lb > best || node_components[node] == comp_i;
            if use_component_bound == 1 && !skip {
                // 1b: read the shared bound with an atomic op, not an ordinary load.
                let bound = F::reinterpret(component_bound[comp_i as usize].fetch_min(0x7f80_0000u32));
                skip = lb > bound;
            }
            if !skip {
                if is_leaf[node] == 1 {
                    for p in idx_start[node] as usize..idx_end[node] as usize {
                        let j = idx_array[p] as usize;
                        if point_components[j] != comp_i {
                            let r = F::max(F::max(sq_dist(data, d, i, j), core_i), core_sq[j]);
                            if r < best || (r == best && j < best_j) { best = r; best_j = j; }
                        }
                    }
                    if use_component_bound == 1 && best_j < n {
                        component_bound[comp_i as usize].fetch_min(u32::reinterpret(best));
                    }
                } else {
                    let (l, r) = (2 * node + 1, 2 * node + 2);
                    let (lb_l, lb_r) = (box_lower_bound(bounds, n_nodes, d, l, data, i),
                                         box_lower_bound(bounds, n_nodes, d, r, data, i));
                    // push far first so near is popped next (upstream: left first on ties)
                    if lb_l <= lb_r { stack[sp] = r as u32; stack[sp + 1] = l as u32; }
                    else            { stack[sp] = l as u32; stack[sp + 1] = r as u32; }
                    sp += 2;
                }
            }
        }
        out_dist[i] = best;
        out_index[i] = best_j as i32;
    }
}
```

The CubeBackend implementation owns the `component_bound` scratch shown in the kernel signature;
it is not a trait input, is never mutated through a fusion-registered input, and is allocated and
reset internally immediately before each candidate launch. `use_component_bound` is likewise an
implementation-internal policy (disabled until benchmarks justify enabling it).

Notes for implementation:

- **Stack bound.** Near-first DFS pushes two nodes per internal visit and pops one, so the
  depth stays ≤ `n_levels + 1`. The host asserts `n_levels + 1 ≤ STACK_LEN`.
- **Component bound lifetime.** When the internal pruning policy is enabled, before every
  candidate launch the implementation resets all `n` scratch slots to `0x7f80_0000` (+inf bits)
  in a separate launch that completes first. A bound must never carry over between rounds: after
  a DSU merge, the edge that set it may be internal to the merged component, and a stale bound
  would prune every valid outgoing edge. `fetch_min(+inf)` is a read and does not reset anything.
  The test covers multiple rounds whose winning weights increase from round to round.
- **Component bound.** This is an optimization only, enabled per runtime once benchmarks justify
  it. With it, a row may stop before reaching its true best. That is fine, because S6c consumes
  only per-component minima. Two rules keep the per-component minimum exact and deterministic.
  Pruning is strict (`>`), so every candidate that ties the component winner is still visited by
  the row that owns it. And the bound is stored as `u32` bits of a non-negative, non-NaN `f32`,
  where integer order equals float order. `+0.0` is normalized and `-0.0` cannot occur because
  distances are sums of squares. CubeCL 0.11 also has `Atomic<u64>::fetch_min`
  (`cubecl-core-0.11.0/src/frontend/element/atomic.rs`, lowered for CUDA in
  `cubecl-cpp-0.11.0/src/cuda/atomic.rs`). A packed 64-bit `(dist bits, row)` winner would
  allow a one-pass device reduction in S6c. That is a later performance option, gated on a
  runtime capability check on both runtimes. The u32 bound is enough for pruning.
- **Leaf guard.** Upstream's `core[idx] < component_best` guard is redundant with the MRD check
  and is not ported. Leaving it out changes no component minimum.
- **Unvisited rows.** With pruning, a row can end with `best_j == n` (decoded to `-1`). S6c
  ignores such rows. A component with every row at `-1` would mean the pruning is wrong, and is
  reported as `HdbscanError::Numerical`, not silently skipped.
- **Data layout.** Rows stay in original order. Permuting rows into leaf order is a later
  locality optimization and must keep original ids for ties and outputs.

Both variants share `sq_dist`/`box_lower_bound` `#[cube]` helpers. Compare per-component minima
against the host oracle using the documented relative weight tolerance; require equal winners and
labels only away from numerical ties. CPU/CUDA parity follows `hdbscan_impl.md`'s explicit ARI
and probability rules rather than bitwise identity.

### S6c. Component winners and merge

**Upstream.** `merge_components` (`boruvka.py:35-78`) reduces the candidates per component in
point order. The strict `<` keeps the *first* (smallest `i`) row on a tie. Insertion order of
the component dict is the order in which components first appear. Each winner edge is applied
with `find` + `union_by_rank` and skipped when its endpoints are already joined. The edge is
oriented `(i, candidate[i])`.

**Port.** Host code, O(n) per round. The candidate tensors (`n` f32 + `n` i32) are read back in
one `into_data` per round. A device-side two-pass atomic reduction (u32 `fetch_min` on distance
bits, then on the source index among equal distances) is a phase 4 optimization that cuts the
readback to `n_components` entries. It is not needed for correctness.

```rust
pub fn merge_components(
    dsu: &mut RankDsu, point_components: &[i32],
    cand_dist: &[f32], cand_index: &[i32], edges: &mut Vec<MstEdge>,
) -> Result<usize, HdbscanError> {
    // winner per component root, in first-appearance order (dict insertion order upstream)
    let mut slot: HashMap<i32, usize> = HashMap::new();
    let mut winners: Vec<(u32, u32, f32)> = Vec::new();
    for i in 0..point_components.len() {
        let (j, w) = (cand_index[i], cand_dist[i]);
        if j < 0 { continue; }
        match slot.entry(point_components[i]) {
            Entry::Vacant(v) => { v.insert(winners.len()); winners.push((i as u32, j as u32, w)); }
            Entry::Occupied(o) => { let s = &mut winners[*o.get()]; if w < s.2 { *s = (i as u32, j as u32, w); } }
        }
    }
    let before = edges.len();
    for (src, dst, w) in winners {
        let (a, b) = (dsu.find(src), dsu.find(dst));
        if a != b { edges.push(MstEdge { src, dst, weight_sq: w }); dsu.union_roots(a, b); }
    }
    let added = edges.len() - before;
    if added == 0 {
        return Err(HdbscanError::Numerical("Boruvka round made no progress".into()));
    }
    Ok(added)
}
```

The round loop subtracts `added` from the component count and ends at 1. The zero-progress
check catches kernel bugs, which would otherwise loop forever. Upstream has no such check
(`boruvka.py:610`). The order of points-with-candidate differs from upstream only on rows we
skip with `-1`. Upstream would insert those too, with the `inf`/`-1` placeholder that its
`np.isnan` guard does not filter. That is a latent upstream bug and does not affect valid
inputs.

## S7. MST finalization and ordering

**Upstream.** Weights become `sqrt` in `float64` (`boruvka.py:646`). `clusters_from_spanning_tree`
then sorts with `np.lexsort((dst, src, weight))` (`hdbscan.py:511-519`), i.e. by weight, then
`src`, then `dst`. The sort is stable.

**Port.**

```rust
pub struct SortedMstEdge { pub src: u32, pub dst: u32, pub weight: f64 }

pub fn finalize_mst(edges: Vec<MstEdge>) -> Vec<SortedMstEdge> {
    let mut out: Vec<_> = edges.into_iter()
        .map(|e| SortedMstEdge { src: e.src, dst: e.dst, weight: (e.weight_sq as f64).sqrt() })
        .collect();
    out.sort_by(|a, b| a.weight.total_cmp(&b.weight).then(a.src.cmp(&b.src)).then(a.dst.cmp(&b.dst)));
    out
}
```

`debug_assert!(out.len() == n - 1)`. The estimator returns this sorted MST, which matches
upstream's `minimum_spanning_tree_` (it stores `sorted_mst`, `hdbscan.py:574`). Edge
*orientation* follows the `(query row, candidate)` rule from S5/S6c. On tie datasets the
orientation can differ from upstream, which can change the linkage row order. That is covered by
the tie-dataset tests.

Phase 1 test hook: host stages S8–S13 take a `&[SortedMstEdge]`, so the fixtures can feed in
upstream's MST directly and get exact equality.

## S8. Single-linkage tree

**Upstream.** `mst_to_linkage_tree` (`cluster_trees.py:66-95`) runs a union-find over
`2n-1` slots with `next = n`. For each sorted edge it finds both roots. It emits
`[max(root_l, root_r), min(root_l, root_r), weight, size_l + size_r]` (note: larger id
first) and then links both roots to the new node `next++`. `linkage_merge_find` compresses the
path fully to the root.

```python
left_component = linkage_merge_find(linkage_merge, left)
right_component = linkage_merge_find(linkage_merge, right)
if left_component > right_component:
    result[index][0] = left_component; result[index][1] = right_component
else:
    result[index][1] = left_component; result[index][0] = right_component
result[index][2] = delta
result[index][3] = linkage_merge.size[left_component] + linkage_merge.size[right_component]
linkage_merge_join(linkage_merge, left_component, right_component)
```

**Port.** `fast-hdbscan-core/src/linkage.rs`:

```rust
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkageRow { pub left: u32, pub right: u32, pub distance: f64, pub size: u32 }

pub fn mst_to_linkage_tree(mst: &[SortedMstEdge]) -> Vec<LinkageRow> {
    let n = mst.len() + 1;
    let mut parent: Vec<u32> = vec![u32::MAX; 2 * n - 1]; // u32::MAX ≙ upstream -1
    let mut size: Vec<u32> = (0..2 * n - 1).map(|i| u32::from(i < n)).collect();
    let mut next = n as u32;
    let find = |parent: &mut [u32], mut x: u32| {
        let start = x;
        while parent[x as usize] != u32::MAX && parent[x as usize] != x { x = parent[x as usize]; }
        parent[x as usize] = x;
        let mut r = start;
        while parent[r as usize] != x { let nx = parent[r as usize]; parent[r as usize] = x; r = nx; }
        x
    };
    mst.iter().map(|e| {
        let (a, b) = (find(&mut parent, e.src), find(&mut parent, e.dst));
        let row = LinkageRow { left: a.max(b), right: a.min(b), distance: e.weight,
                               size: size[a as usize] + size[b as usize] };
        size[next as usize] = row.size;
        parent[a as usize] = next;
        parent[b as usize] = next;
        next += 1;
        row
    }).collect()
}
```

The tree is exactly determined by the sorted MST, so it is a pure function to test against
upstream fixtures.

## S9. Condensed tree

**Upstream.** `condense_tree` (`cluster_trees.py:206-352`) walks the linkage hierarchy in BFS
order from `root = 2(n-1)`. The root is relabelled `n`, and new cluster labels start at
`n + 1`. For each internal node it computes λ = `1/d` (or `inf` at `d = 0`) and the child
counts. Then:

1. Left child is a point and right count ≥ `mcs`: emit `(parent, left, λ, 1)` and the right
   child inherits the parent label.
2. Left count < `mcs` ≤ right count: right inherits the label. Every point under the left
   child falls out at λ, and its internal nodes are marked `ignore`.
3. The mirror image of case 2.
4. Both < `mcs`: all points under both children fall out at λ.
5. Both > `max_cluster_size`: both children inherit. This branch is dead in the HDBSCAN path,
   because `clusters_from_spanning_tree` never passes `max_cluster_size` here
   (`hdbscan.py:526-528`).
6. Otherwise it is a true split: two new cluster labels with sizes `left_count` and
   `right_count`.

```python
for node in node_list:
    if ignore[node] or node < num_points:
        continue
    parent_node = relabel[node]
    l, r, d, _ = hierarchy[node - num_points]
    lambda_value = hdbscan_lambda_value(d)
    left_count = hierarchy[left - num_points, 3] if left >= num_points else sample_weights[left]
    ...
    if left < num_points and right_count >= min_cluster_size: ...
    elif left_count < min_cluster_size and right_count >= min_cluster_size: ...
```

`eliminate_branch` emits the branch's points in BFS order (`bfs_from_hierarchy`). The output
row order (parents before children, with points and clusters interleaved in processing order)
is relied on by S10–S12.

**Port.** `fast-hdbscan-core/src/condense.rs`. A direct translation with:

- `CondensedTree { parent: Vec<u32>, child: Vec<u32>, lambda: Vec<f32>, child_size: Vec<u32> }`
  (SoA, the same order as upstream).
- An iterative BFS that produces the same node order. `bfs_from_hierarchy` is level-order
  with left before right, built in a reusable buffer.
- `hdbscan_lambda(d: f64) -> f32 { if d > 0.0 { (1.0 / d) as f32 } else { f32::INFINITY } }`.
  `1/inf = 0` handles the infinite-core case.
- Counts are rounded to `f32` as upstream does (`np.float32(hierarchy[…, 3])`). The integer
  threshold is *not*: Numba promotes the mixed `f32`/`int64` comparison to `f64`. The port
  compares `(count as f32) as f64` against `min_cluster_size as f64`. For n ≤ 2^24 this equals
  the plain integer comparison. Above that it reproduces upstream's rounded counts without
  wrongly rounding the threshold.
- Case 5 is omitted, because no caller passes `max_cluster_size` into condensation. Adding it
  later is local.

```rust
pub fn condense_tree(h: &[LinkageRow], min_cluster_size: usize) -> CondensedTree {
    let n = h.len() + 1;
    let root = 2 * h.len();
    let mcs = min_cluster_size as f64;
    // f32-rounded count (upstream's float32 sizes), promoted to f64 for the comparison.
    let count = |c: usize| (if c >= n { h[c - n].size as f32 } else { 1.0f32 }) as f64;
    let mut relabel = vec![0u32; root + 1];
    relabel[root] = n as u32;
    let mut ignore = vec![false; root + 1];
    let mut next_label = n as u32 + 1;
    let mut out = CondensedTree::with_capacity(root);
    let mut bfs = Vec::new();
    for node in bfs_from_hierarchy(h, root, n) {
        if ignore[node] || node < n { continue; }
        let parent = relabel[node];
        let row = h[node - n];
        let (l, r, lam) = (row.left as usize, row.right as usize, hdbscan_lambda(row.distance));
        let (lc, rc) = (count(l), count(r));
        if l < n && rc >= mcs {
            relabel[r] = parent;
            out.push(parent, l as u32, lam, 1);
        } else if lc < mcs && rc >= mcs {
            relabel[r] = parent;
            eliminate_branch(h, n, l, parent, lam, &mut out, &mut ignore, &mut bfs);
        } else if lc >= mcs && rc < mcs {
            relabel[l] = parent;
            eliminate_branch(h, n, r, parent, lam, &mut out, &mut ignore, &mut bfs);
        } else if lc < mcs && rc < mcs {
            eliminate_branch(h, n, l, parent, lam, &mut out, &mut ignore, &mut bfs);
            eliminate_branch(h, n, r, parent, lam, &mut out, &mut ignore, &mut bfs);
        } else {
            relabel[l] = next_label;
            out.push(parent, next_label, lam, lc as u32);
            relabel[r] = next_label + 1;
            out.push(parent, next_label + 1, lam, rc as u32);
            next_label += 2;
        }
    }
    out
}
```

Upstream's `eliminate_branch` for a single point (`branch_node < num_points`) leaves
`sizes[idx]` at its initial `1`. For multi-point branches, the rows' sizes stay at the initial
`np.ones`. Every eliminated row therefore has `child_size = 1`, and the port pushes 1.

Upstream initializes `parents = np.ones(root)` but only returns `[:idx]`, so this is not
observable.

## S10. Persistence simplification (optional)

**Upstream.** `simplify_hierarchy(condensed_tree, persistence)` (`cluster_trees.py:775-831`)
runs only when `persistence > 0` and the tree is non-empty (`hdbscan.py:529-532`). The cluster
tree (rows with `child_size > 1`) consists of consecutive sibling pairs. Walking the pairs from
the last one back, a split is kept when `min(birth_l, birth_r) - death ≥ persistence`, where
`birth` is the maximum λ inside a cluster, propagated upwards. Otherwise the less persistent
child is merged into the parent. `lambda_map` records the death λ overrides, and a relabelling
pass renumbers the remaining clusters consecutively.

```python
for idx in range(cluster_tree.parent.shape[0] - 1, 0, -2):
    parent = cluster_tree.parent[idx]
    children = cluster_tree.child[idx - 1 : idx + 1]
    death = cluster_tree.lambda_val[idx]
    births = max_births[children - n_points]
    max_births[parent - n_points] = max(max_births[parent - n_points], births.max())
    if (births.min() - death) >= persistence_threshold:
        continue
    sibling_idx = np.int64(births[0] <= births[1])
    if leaf_indicator[node_indices[sibling_idx]]:
        leaf_indicator[parent - n_points] = True
    else:
        lambda_map[children[1 - sibling_idx]] = death
    parent_map[node_indices] = parent
...
for i, parent in enumerate(parent_map):
    n_skipped[i] = parent != (i + n_points)
    parent_map[i] = parent - n_skipped[: parent - n_points].sum()
```

**Port.** A verbatim translation in `condense.rs::simplify_hierarchy`, with these points pinned:

- The `max_births[parent - n_points] = lambda_val` scatter happens in row order, so the last
  write wins. Keep the same order (assign in a forward loop).
- `births.min() - death` is computed in `f32` (both operands are `f32`, and `persistence` is
  compared after promotion to `f64`, as NumPy does when comparing an `f32` scalar with a Python
  float). The port computes `((b_min - death) as f64) >= persistence`.
- The O(n²) `n_skipped[:k].sum()` becomes a prefix sum, `skipped_before[k]`. The result is
  identical.
- Upstream mutates the arrays in place and masks them. The port builds a new `CondensedTree`.
- The pair assumption (`children = child[idx-1..=idx]`) holds because S9 always emits cluster
  children in pairs. `debug_assert!` that the cluster-tree length is even.

## S11. Cluster tree and selection

**Upstream.** `cluster_tree_from_condensed_tree` is the `child_size > 1` mask
(`cluster_trees.py:673`). The selection methods are:

- **EOM** (`extract_eom_clusters`, `cluster_trees.py:725-771`). Node scores are
  `Σ λ·size` over child rows minus `λ_birth·size` (`score_condensed_tree_nodes`,
  `:657-669`). Node sizes come from the cluster tree, with the root sized `root - 1 = n - 1`.
  `eom_recursion` selects a node when `max(score, 0) ≥ Σ child results` and
  `size ≤ max_cluster_size`, and unselects its whole subtree. `allow_single_cluster` decides
  whether recursion starts at the root or at the root's children. An empty cluster tree gives
  no clusters.
- **Leaf** (`cluster_tree_leaves`, `:370-374`): cluster-tree nodes that are never parents.
  An empty cluster tree gives no clusters. `allow_single_cluster` has no effect here, as
  upstream.
- **Epsilon** (`cluster_epsilon_search`, `:835-863`). Runs only when more than one cluster is
  selected and `ε > 0`. For each selected cluster whose death ε (`1/λ_birth`) is below the
  minimum, it climbs (`traverse_upwards`) to the first ancestor with death ε ≥ ε_min, or to the
  root. All segments of that branch are marked processed.

```python
def eom_recursion(node, cluster_tree, node_scores, node_sizes, selected_clusters, max_cluster_size):
    current_score = max(node_scores[node], 0.0)
    current_size = node_sizes[node]
    children = cluster_tree.child[cluster_tree.parent == node]
    child_score_total = 0.0
    for child_node in children:
        child_score_total += eom_recursion(child_node, ...)
    if child_score_total > current_score or current_size > max_cluster_size:
        return child_score_total
    else:
        selected_clusters[node] = True
        unselect_below_node(node, cluster_tree, selected_clusters)
        return current_score
```

**Port.** `fast-hdbscan-core/src/select.rs`.

- Cluster ids are contiguous in `n..=max_id`. Use dense `Vec`s indexed by `id - n`, not maps:
  `scores: Vec<f32>`, `sizes: Vec<f32>`, `selected: Vec<bool>`, plus a CSR of children built once
  from the cluster tree. Child order follows row order, which matches
  `cluster_tree.child[cluster_tree.parent == node]`.
- **Scores**: `f32` accumulation in row order, matching the Numba dict (`np.float32` values,
  `lambda_val[i] * child_size[i]` as `f32 × f32`). Note the special case: the `child_size > 1`
  assignment *overwrites* (`result[child] = -λ·size`). That is safe only because a cluster's
  own birth row comes before its children's rows. The port asserts this ordering in debug
  builds.
- **EOM recursion** becomes an explicit post-order stack (cluster trees can be O(n) deep on
  chain-like data). `child_score_total` is accumulated as `f64`, because Numba's `0.0` literal is
  `float64`, and the comparisons `child_score_total > current_score` happen in `f64`, where
  `current_score = max(f32_score, 0.0)` is promoted. The fixtures confirm this promotion.
  `unselect_below_node` is not ported as repeated subtree clears. Every selection made at a
  node unselects its whole subtree, and the post-order visits ancestors later, so the final
  set is exactly the nodes marked "chosen" in post-order that have no chosen ancestor. The
  port records `chosen` during the post-order pass and resolves it in one top-down pass. That
  is O(n) and gives the same set. It is tested against a literal port on random trees.
- **Returned order**: upstream returns `selected_clusters` in dict insertion order
  (`node_scores` keys: root first, then children in row order). Only `np.sort(clusters)` is
  used downstream for numbering, but epsilon search iterates in the given order, and its
  `processed` set makes the result order-dependent. The port therefore returns selected ids in
  the same insertion order (root, then cluster-tree rows in order).
- **max_cluster_size**: `None` means never limit. `Some(m)` compares
  `(size_f32 as f64) > m as f64`. The threshold is never rounded to `f32` (see S9).
- **Epsilon search**: `searchsorted(child, cluster)` works because the cluster tree's children
  are increasing. Port it as `binary_search`, with `debug_assert!` that the children are sorted.
  `traverse_upwards` becomes a loop. `segments_in_branch` becomes a BFS over the parent-sorted
  CSR. `λ == 0` gives a death ε of `+inf`, as upstream does.

```rust
pub fn select_clusters(ct: &CondensedTree, n: usize, cfg: &SelectionConfig) -> Vec<u32> {
    let cluster_tree = ct.mask(|i| ct.child_size[i] > 1);
    let mut selected = match cfg.method {
        ClusterSelectionMethod::Eom => extract_eom(ct, &cluster_tree, n, cfg.max_cluster_size, cfg.allow_single_cluster),
        ClusterSelectionMethod::Leaf if cluster_tree.is_empty() => Vec::new(),
        ClusterSelectionMethod::Leaf => cluster_tree_leaves(&cluster_tree, n),
    };
    if selected.len() > 1 && cfg.epsilon > 0.0 {
        selected = cluster_epsilon_search(&selected, &cluster_tree, cfg.epsilon);
    }
    selected
}
```

`extract_eom` with `allow_single_cluster = false` runs the recursion over the root's children
only when `len(node_scores) > 1`. That is always true when the cluster tree is non-empty, but the
guard is kept anyway.

## S12. Labels and membership strengths

**Upstream.** `get_cluster_label_vector` (`cluster_trees.py:945-996`):

```python
cluster_label_map = {c: n for n, c in enumerate(np.sort(clusters))}
disjoint_set = ds_rank_create(max(tree.parent.max() + 1, tree.child.max() + 1))
for n in range(tree.parent.shape[0]):
    child = tree.child[n]; parent = tree.parent[n]
    if child not in clusters:
        if child < n_samples and tree.lambda_val[n] == 0.0:
            continue
        ds_union_by_rank(disjoint_set, parent, child)
for n in range(root_cluster):
    cluster = ds_find(disjoint_set, n)
    if cluster < root_cluster:
        result[n] = -1
    elif cluster == root_cluster:
        if len(clusters) == 1:
            max_lambda = tree.lambda_val[tree.parent == cluster].max()
            cur_lambda = tree.lambda_val[tree.child == n]
            if cluster_selection_epsilon > 0.0:
                result[n] = label if cur_lambda >= 1 / cluster_selection_epsilon else -1
            elif cur_lambda >= max_lambda:
                result[n] = label
            else:
                result[n] = -1
        else:
            result[n] = -1
    else:
        result[n] = cluster_label_map[cluster]
```

`get_point_membership_strength_vector` (`:1012-1031`) sets, for each labelled point,
`min(λ_point, λ_max(cluster)) / λ_max(cluster)`. The value is 1 when `λ_max == 0` or `λ_point`
is not finite. `λ_max` is the maximum λ over the cluster's *point* rows only (`child_size == 1`,
`max_lambdas`). Unlabelled points get 0.

**Port.** `fast-hdbscan-core/src/labels.rs`.

- Use `RankDsu` from S5 with upstream's argument order (`union(parent, child)`). The tie rule
  makes the parent's root win when ranks are equal, and representatives decide the result, so
  the DSU semantics must be exact. Upstream's `ds_union_by_rank(…, x, y)` finds both roots
  first, and `RankDsu::union` does the same.
- `clusters` membership uses a `Vec<bool>` over ids.
- For the single-cluster root branch, `max_lambda` over `parent == root` rows and each point's
  `cur_lambda` come from one pass building `point_lambda[point]` (the row where `child == point`;
  every point appears exactly once as a child). `1 / eps` is computed in `f64`, and
  `cur_lambda` is promoted to `f64` for the comparison.
- Labels are `i32`. Upstream returns `intp`, and the Python wrapper casts to `np.intp`.
- Membership: dense `deaths: Vec<f32>` per selected id. The values originate from `f32`, so
  storing them as `f32` is exact. The division is done in `f64` and the result stored as `f32`.
  Output is `Vec<f32>`.

```rust
pub fn membership_strengths(ct: &CondensedTree, selected_sorted: &[u32], labels: &[i32], n: usize) -> Vec<f32> {
    let mut deaths = vec![0.0f32; selected_sorted.len()];
    let pos = |c: u32| selected_sorted.binary_search(&c).ok();
    for i in 0..ct.len() {
        if ct.child_size[i] == 1 {
            if let Some(k) = pos(ct.parent[i]) { deaths[k] = deaths[k].max(ct.lambda[i]); }
        }
    }
    let mut out = vec![0.0f32; labels.len()];
    for i in 0..ct.len() {
        let p = ct.child[i] as usize;
        if p >= n || labels[p] < 0 { continue; }
        let max_l = deaths[labels[p] as usize];
        let l = ct.lambda[i];
        out[p] = if max_l == 0.0 || !l.is_finite() {
            1.0
        } else {
            ((l.min(max_l) as f64) / (max_l as f64)) as f32
        };
    }
    out
}
```

Upstream's `max_lambdas` initializes with the Python float `0.0` and takes `max` against `f32`
values. The dict values are therefore `float64`, but every stored value comes from an `f32`.
The division happens in `f64` (`min(float32, float64) / float64`), and the result is stored into
a `float32` array. The snippet above does the same. The fixture tests pin this down to the bit.

Empty case (empty condensed tree or no selected clusters): labels are all `-1` and strengths
all 0. The port returns this directly, without reaching the `tree.parent.min()` call that
upstream makes on the tree.

## S13. Remap non-finite rows

**Upstream.** `remap_condensed_tree` and `remap_single_linkage_tree` (`hdbscan.py:119-196`),
plus the label and probability scatter (`hdbscan.py:806-820`):

```python
# condensed tree
child = internal_to_raw[child] if child < finite_count else child + outlier_count
tree[i] = (parent + outlier_count, child, lambda_val, child_size)
root = tree[0][0]
outlier rows = [(root, outlier, 0, 1) for outlier in outliers]   # prepended
# single-linkage tree
left/right = internal_to_raw[x] if x < finite_count else x + outlier_count
last_cluster_id = tree[-1][0:2].max(); last_cluster_size = tree[-1][3]
for outlier in outliers:
    append (outlier, last_cluster_id + 1, inf, last_cluster_size + 1)
    last_cluster_id += 1; last_cluster_size += 1
```

The MST, `neighbors` and `core_distances` stay in *internal* (finite-row) indexing upstream.
Labels and probabilities are scattered back to raw rows, with `-1`/`0` for outliers.

**Port.** `finite.rs::remap`, a verbatim translation that applies only when `outliers` is
non-empty. The single-linkage remap reproduces upstream's `last_cluster_id + 1` chaining
exactly, even though it does not yield a canonical linkage matrix. `dbscan_clustering` (S14) on
the remapped tree is the observable consumer, and it must match upstream. Probabilities widen to
`f64` only in the Python wrapper, as upstream (`np.zeros(X.shape[0])` is `float64`) does. The
Rust API keeps `f32`. The MST, neighbors and core distances are documented as finite-row indexed
and expose `finite_index` so callers can map them.

Degenerate `n_finite == 1` (S2): labels and probabilities are scattered, and trees are left
empty without outlier rows.

## S14. `dbscan_clustering(epsilon)`

**Upstream.** `get_cluster_labelling_at_cut(single_linkage_tree, cut, min_samples or
min_cluster_size)` (`cluster_trees.py:907-941`). It unions every linkage row with
`distance < cut` into its node id (`n + i`), takes each point's representative, and drops
clusters smaller than the threshold. Labels are numbered by ascending representative id
(`np.unique`).

```python
cluster = num_points
for i in range(linkage_tree.shape[0]):
    if linkage_tree[i, 2] < cut:
        ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 0]), cluster)
        ds_union_by_rank(disjoint_set, np.intp(linkage_tree[i, 1]), cluster)
    cluster += 1
...
for cluster in np.unique(result):
    cluster_label_map[cluster] = -1 if cluster_size[cluster] < min_cluster_size else next_label
```

**Port.** `labels.rs::dbscan_cut`, which runs `RankDsu` over `2·rows + 1` slots with the same
union argument order and assigns labels by ascending representative. It operates on the stored
(remapped) single-linkage tree. The threshold is `min_samples.unwrap_or(min_cluster_size)`. Note
that with `min_samples == 1`, upstream labels each outlier row as its own cluster, and we
reproduce that. `epsilon` must not be NaN (`HdbscanError::InvalidConfig`). Any other value
keeps upstream semantics. `epsilon ≤ 0` unions nothing, so all points are singletons.
`epsilon = inf` unions every finite-distance row.

---

## Orchestration (estimator)

```rust
impl HDBSCAN {
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<FittedHDBSCAN, HdbscanError> {
        let cfg = &self.config;                                         // S1 (validated in new)
        let host = flatten_validate(data)?;
        let split = split_finite(&host, d);                             // S2
        let finite = compact_and_upload(&host, &split, &device)?;
        if split.finite_index.len() < 2 { return Ok(degenerate(&split, d)); }
        let ms = cfg.resolved_min_samples();
        let cd = core_distances(&finite, ms)?;                          // S3
        let tree = use_kdtree(d).then(|| build_kdtree(&host_f, m, d)); // S4 (phase 4)
        let mut dsu = RankDsu::new(m);
        let mut edges = init_from_knn(&cd, ms, &mut dsu);               // S5
        let mut comps = count_components(&mut dsu);
        let device_tree = tree.as_ref().map(|t| upload(t, &device));
        while comps > 1 {                                               // S6
            update_component_vectors(&mut dsu, tree.as_ref(), &mut pc, &mut nc);
            // The CubeBackend implementation owns/reset its optional KD-pruning scratch; it is
            // deliberately not an HdbscanOps input.
            let (dist, idx) = boruvka_candidates(&finite, &core_t, upload(&pc), device_tree.with(&nc));
            comps -= merge_components(&mut dsu, &pc, &read(dist), &read(idx), &mut edges)?;
        }
        let mst = finalize_mst(edges);                                  // S7
        let slt = mst_to_linkage_tree(&mst);                            // S8
        let mut ct = condense_tree(&slt, cfg.min_cluster_size);         // S9
        if cfg.cluster_selection_persistence > 0.0 && !ct.is_empty() {
            ct = simplify_hierarchy(ct, cfg.cluster_selection_persistence); // S10
        }
        let selected = select_clusters(&ct, m, &cfg.selection());       // S11
        let labels = cluster_label_vector(&ct, &selected, cfg.cluster_selection_epsilon, m); // S12
        let probs = membership_strengths(&ct, &sorted(&selected), &labels, m);
        Ok(remap(split, labels, probs, slt, ct, mst, cd))               // S13
    }
}
```

The KD path (`use_kdtree`) is enabled once phase 4 lands, for `d ≤ KD_MAX_DIM`. The cutoff is
picked by benchmark. Upstream uses 30 as the switch to its approximate path, so 30 is the
starting value. Above the cutoff, brute force is used. The result is the same exact MST either
way, so the switch is a pure performance choice and never changes output on non-tie data.

## Per-stage test matrix

| Stage | Test | Oracle |
|---|---|---|
| S1 | each invalid field / input shape → specific `HdbscanError` | — |
| S2 | NaN/inf rows, all non-finite (error), one finite row | upstream labels |
| S3 | core distances² vs host brute force; `min_samples ∈ {1, n-1, n, n+5}` | host + fixture `core_distances` (1e-6 rel) |
| S4 | every point in exactly one leaf; bounds contain points; depth ≤ `MAX_KD_LEVELS` | invariants |
| S5 | each init edge has weight `max(core_i, knn_sq)` (= `core_i` when `min_samples > 1`); forest is acyclic | invariants + oracle MST |
| S6 | per-component minima match host within weight tolerance; brute force and KD agree away from ties; CPU/CUDA use stated parity rules | host Boruvka/Prim oracle |
| S7 | sorted weight multiset equals upstream MST | fixture (1e-5 rel) |
| S8–S12 | fed with upstream's sorted MST: linkage, condensed, labels, probabilities exactly equal | fixtures (exact) |
| S10 | persistence sweep including thresholds equal to a branch persistence | fixtures |
| S11 | `leaf`/`eom`, `allow_single_cluster`, `max_cluster_size`, ε sweep; deep chain (stack safety) | fixtures |
| S13 | remapped condensed/linkage trees equal upstream's `_condensed_tree`/`_single_linkage_tree` given upstream's recorded `outliers` order | fixtures |
| S14 | `dbscan_clustering(ε)` sweep, including after remap | fixtures |
| e2e | blobs/moons/varying density, d ∈ {2, 10, 50}; duplicates; UMAP→HDBSCAN | ARI = 1, probs 1e-5 |
