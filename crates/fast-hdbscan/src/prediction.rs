//! Fitted-model state ownership and bounded exact-KD `approximate_predict` orchestration.
//!
//! # Canonical state and views
//!
//! Every successful fit captures one immutable [`FittedState`]: the authoritative,
//! backend-free host record of the complete fitted model (configuration, requested and
//! resolved fitting choices, all public outputs and diagnostics, owned compact finite
//! training geometry and raw-row mapping) plus the prediction hierarchy derived from the
//! final labels/condensed tree. The public mutable fields of
//! [`crate::FittedHDBSCAN`] are detached output views; mutating them affects neither
//! prediction nor this canonical record.
//!
//! Runtime concerns (selected runtime backend, initialization locks, `Arc` sharing) live
//! in [`PredictionRuntime`], separate from canonical data. Host KD tree and prediction-core
//! payloads are canonical values published into the runtime cache as immutable `Arc`s.
//!
//! # Future persistence boundary (not implemented)
//!
//! A future save prepares a complete state from [`FittedState`] plus a validated host KD
//! tree and the prediction-core status: `Ready { rank: s, values }` for nonempty cluster
//! trees (rejecting unsupported `2*s > n_finite` widths before cold work), or
//! `NotRequired(NoClusterTree)` for shortcut models, always with a valid tree (singleton
//! included). Derived hierarchy lookup tables are reconstructed on load, never persisted
//! as a second authority. Load selects the runtime backend independently of the training
//! backend recorded here as provenance. No file I/O or public save/load API exists yet.
//!
//! # Lazy cache and locking contract
//!
//! Lock order: `init` (serializes fallible cold initialization) then `cache` (held only
//! briefly to read or publish `Arc` payloads). No helper acquires `init` recursively and
//! no lock is held during warm searches. Tree build failure publishes nothing; core
//! failure keeps an already-published tree and leaves cores `Uninitialized` for retry.

use std::sync::{Arc, Mutex, MutexGuard};

use burn::{
    backend::Dispatch,
    tensor::{Device, Int, Tensor, TensorData, Transaction},
};
use fast_cubecl_common::{BackendChoice, selected_device};
use fast_hdbscan_core::{
    HdbscanConfig, HdbscanError, KdTree, MstGraph, PredictionColumns, PredictionHierarchy,
    condense::CondensedTree,
    error::Result,
    kdtree::{LEAF_SIZE, MAX_KD_LEVELS},
    linkage::{LinkageRow, SortedMstEdge},
    prediction::{candidate_width, prediction_core_from_squared_rows},
};

use crate::backend::{DeviceKdReference, HdbscanOps};
use crate::estimator::{
    DeviceAllocationLimit, HdbscanSearch, KnnDiagnostics, allocation_limit,
    check_device_allocation, check_device_tensor_allocation, checked_allocation_bytes,
    map_backend_error, tensor_allocation_bytes, upload_kd_tree,
};

/// Maximum query rows per bounded prediction batch before device-limit planning.
pub(crate) const PREDICTION_BATCH_ROWS: usize = 1024;

/// Result of [`crate::FittedHDBSCAN::approximate_predict`], in query order.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionResult {
    /// Existing fitted cluster IDs, or `-1` for noise.
    pub labels: Vec<i32>,
    /// Membership strengths in `[0, 1]`.
    pub probabilities: Vec<f64>,
    /// Original raw training-row ID of the selected mutual-reachability connector;
    /// `-1` only for the no-cluster-tree shortcut.
    pub connecting_points: Vec<i32>,
    /// Reference empty-cluster-tree shortcut; lets the Python bridge emit its warning.
    pub no_cluster_tree: bool,
}

/// Authoritative immutable fit/load snapshot of the complete fitted model.
#[derive(Debug)]
pub(crate) struct FittedState {
    pub(crate) config: HdbscanConfig,
    /// Concrete backend that fitted the model (provenance, not runtime selection).
    pub(crate) training_backend: BackendChoice,
    pub(crate) labels: Vec<i32>,
    pub(crate) probabilities: Vec<f32>,
    pub(crate) linkage: Vec<LinkageRow>,
    pub(crate) condensed: CondensedTree,
    pub(crate) mst: Vec<SortedMstEdge>,
    pub(crate) neighbors: Vec<i32>,
    pub(crate) core_distances: Vec<f32>,
    /// Ascending compact-to-raw row mapping.
    pub(crate) finite_index: Vec<u32>,
    pub(crate) n_features: usize,
    pub(crate) raw_n: usize,
    pub(crate) search: HdbscanSearch,
    pub(crate) resolved_graph: MstGraph,
    pub(crate) knn_diagnostics: Option<KnnDiagnostics>,
    /// Owned compact finite row-major float32 training coordinates.
    pub(crate) geometry: Vec<f32>,
    /// Resolved `min_samples` (`s`) used for prediction ranks.
    pub(crate) min_samples: usize,
    /// Validated prediction metadata derived from final labels/condensed tree.
    pub(crate) hierarchy: PredictionHierarchy,
}

/// Fitted outputs moved into [`FittedState::new`].
pub(crate) struct FittedOutputs {
    pub(crate) config: HdbscanConfig,
    pub(crate) training_backend: BackendChoice,
    pub(crate) labels: Vec<i32>,
    pub(crate) probabilities: Vec<f32>,
    pub(crate) linkage: Vec<LinkageRow>,
    pub(crate) condensed: CondensedTree,
    pub(crate) mst: Vec<SortedMstEdge>,
    pub(crate) neighbors: Vec<i32>,
    pub(crate) core_distances: Vec<f32>,
    pub(crate) finite_index: Vec<u32>,
    pub(crate) n_features: usize,
    pub(crate) raw_n: usize,
    pub(crate) search: HdbscanSearch,
    pub(crate) resolved_graph: MstGraph,
    pub(crate) knn_diagnostics: Option<KnnDiagnostics>,
    pub(crate) geometry: Vec<f32>,
}

impl FittedState {
    pub(crate) fn new(outputs: FittedOutputs) -> Result<Self> {
        let n = outputs.finite_index.len();
        let expected = n
            .checked_mul(outputs.n_features)
            .ok_or(HdbscanError::Overflow(
                "retained training geometry elements",
            ))?;
        if n == 0 || outputs.geometry.len() != expected {
            return Err(HdbscanError::Numerical(
                "retained training geometry does not match finite rows".into(),
            ));
        }
        if outputs.geometry.iter().any(|value| !value.is_finite()) {
            return Err(HdbscanError::Numerical(
                "retained training geometry contains nonfinite values".into(),
            ));
        }
        if outputs.labels.len() != outputs.raw_n {
            return Err(HdbscanError::Numerical(
                "fitted labels do not match raw row count".into(),
            ));
        }
        let hierarchy =
            PredictionHierarchy::new(&outputs.condensed, &outputs.labels, &outputs.finite_index)?;
        Ok(Self {
            min_samples: outputs.config.resolved_min_samples(),
            config: outputs.config,
            training_backend: outputs.training_backend,
            labels: outputs.labels,
            probabilities: outputs.probabilities,
            linkage: outputs.linkage,
            condensed: outputs.condensed,
            mst: outputs.mst,
            neighbors: outputs.neighbors,
            core_distances: outputs.core_distances,
            finite_index: outputs.finite_index,
            n_features: outputs.n_features,
            raw_n: outputs.raw_n,
            search: outputs.search,
            resolved_graph: outputs.resolved_graph,
            knn_diagnostics: outputs.knn_diagnostics,
            geometry: outputs.geometry,
            hierarchy,
        })
    }

    pub(crate) fn n_finite(&self) -> usize {
        self.finite_index.len()
    }

    /// Canonical status of the prediction-core cache before any initialization.
    fn initial_core_status(&self) -> CoreStatus {
        if self.hierarchy.no_cluster_tree() {
            CoreStatus::NotRequired(NotRequiredReason::NoClusterTree)
        } else {
            CoreStatus::Uninitialized
        }
    }

    /// Persistent host bytes of the canonical snapshot, from actual allocations
    /// (excluding the hierarchy, tree and cores which are reported separately).
    pub(crate) fn canonical_host_bytes(&self) -> Option<usize> {
        let condensed = self
            .condensed
            .parent
            .len()
            .checked_mul(2 * std::mem::size_of::<usize>() + 2 * std::mem::size_of::<f32>())?;
        let knn = self
            .knn_diagnostics
            .as_ref()
            .map_or(Some(0), |diagnostics| {
                diagnostics
                    .neighbors
                    .len()
                    .checked_mul(4)?
                    .checked_add(diagnostics.distances_sq.len().checked_mul(4)?)
            })?;
        [
            self.labels.len().checked_mul(4)?,
            self.probabilities.len().checked_mul(4)?,
            self.linkage
                .len()
                .checked_mul(std::mem::size_of::<LinkageRow>())?,
            condensed,
            self.mst
                .len()
                .checked_mul(std::mem::size_of::<SortedMstEdge>())?,
            self.neighbors.len().checked_mul(4)?,
            self.core_distances.len().checked_mul(4)?,
            self.finite_index.len().checked_mul(4)?,
            knn,
            self.geometry.len().checked_mul(4)?,
        ]
        .into_iter()
        .try_fold(0usize, |sum, bytes| sum.checked_add(bytes))
    }
}

/// Persistent host bytes of a host KD tree, from its actual arrays.
pub(crate) fn host_tree_bytes(tree: &KdTree) -> Option<usize> {
    [
        tree.indices.len(),
        tree.starts.len(),
        tree.ends.len(),
        tree.is_leaf.len(),
        tree.bounds_min.len(),
        tree.bounds_max.len(),
    ]
    .into_iter()
    .try_fold(0usize, |sum, elements| {
        sum.checked_add(elements.checked_mul(4)?)
    })
}

/// Reason a prediction-core vector is not required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotRequiredReason {
    /// No condensed cluster edges: prediction is the all-noise shortcut.
    NoClusterTree,
}

/// Canonical prediction-core cache status.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CoreStatus {
    NotRequired(NotRequiredReason),
    Uninitialized,
    /// Inclusive-self Euclidean distance at rank `rank - 1` for every compact training row.
    Ready {
        rank: usize,
        values: Arc<[f64]>,
    },
}

#[derive(Debug)]
struct CacheState {
    tree: Option<Arc<KdTree>>,
    cores: CoreStatus,
}

/// Runtime owner: concrete runtime backend and synchronized lazy host caches.
#[derive(Debug)]
pub(crate) struct PredictionRuntime {
    backend: BackendChoice,
    init: Mutex<()>,
    cache: Mutex<CacheState>,
}

/// Snapshot of runtime cache readiness, for diagnostics and tests.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionCacheStatus {
    pub tree_ready: bool,
    /// Actual host bytes of the retained/published KD tree (0 when absent).
    pub tree_bytes: usize,
    /// Actual host bytes of the canonical snapshot excluding tree, cores and hierarchy.
    pub canonical_bytes: usize,
    /// Actual retained host bytes of the prediction hierarchy metadata.
    pub hierarchy_bytes: usize,
    pub cores: PredictionCoreStatus,
}

/// Public diagnostic view of [`CoreStatus`].
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq)]
pub enum PredictionCoreStatus {
    NotRequiredNoClusterTree,
    Uninitialized,
    Ready { rank: usize, len: usize },
}

fn poisoned(subject: &str) -> HdbscanError {
    HdbscanError::Numerical(format!("prediction {subject} lock was poisoned"))
}

impl PredictionRuntime {
    /// `tree` is the fit-created host tree when the fitting route built one.
    pub(crate) fn new(state: &FittedState, backend: BackendChoice, tree: Option<KdTree>) -> Self {
        Self {
            backend,
            init: Mutex::new(()),
            cache: Mutex::new(CacheState {
                tree: tree.map(Arc::new),
                cores: state.initial_core_status(),
            }),
        }
    }

    pub(crate) fn backend(&self) -> BackendChoice {
        self.backend
    }

    fn cache(&self) -> Result<MutexGuard<'_, CacheState>> {
        self.cache.lock().map_err(|_| poisoned("cache"))
    }

    fn init(&self) -> Result<MutexGuard<'_, ()>> {
        self.init.lock().map_err(|_| poisoned("initialization"))
    }

    pub(crate) fn status(&self, state: &FittedState) -> Result<PredictionCacheStatus> {
        let cache = self.cache()?;
        let tree_bytes = match &cache.tree {
            Some(tree) => {
                host_tree_bytes(tree).ok_or(HdbscanError::Overflow("KD-tree host bytes"))?
            }
            None => 0,
        };
        Ok(PredictionCacheStatus {
            tree_ready: cache.tree.is_some(),
            tree_bytes,
            canonical_bytes: state
                .canonical_host_bytes()
                .ok_or(HdbscanError::Overflow("canonical fitted host bytes"))?,
            hierarchy_bytes: state.hierarchy.retained_bytes(),
            cores: match &cache.cores {
                CoreStatus::NotRequired(NotRequiredReason::NoClusterTree) => {
                    PredictionCoreStatus::NotRequiredNoClusterTree
                }
                CoreStatus::Uninitialized => PredictionCoreStatus::Uninitialized,
                CoreStatus::Ready { rank, values } => PredictionCoreStatus::Ready {
                    rank: *rank,
                    len: values.len(),
                },
            },
        })
    }

    /// Returns the published tree, building and publishing it under `init` when absent.
    /// The caller must not hold `init`.
    fn tree(&self, state: &FittedState) -> Result<Arc<KdTree>> {
        if let Some(tree) = self.cache()?.tree.clone() {
            return Ok(tree);
        }
        let _init = self.init()?;
        self.tree_locked(state)
    }

    /// Tree initialization while the caller holds `init`.
    fn tree_locked(&self, state: &FittedState) -> Result<Arc<KdTree>> {
        if let Some(tree) = self.cache()?.tree.clone() {
            return Ok(tree);
        }
        #[cfg(test)]
        tests::maybe_fail(tests::Injection::Tree)?;
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("prediction_tree_build").entered();
        let tree = Arc::new(KdTree::build(
            &state.geometry,
            state.n_finite(),
            state.n_features,
        )?);
        self.cache()?.tree = Some(Arc::clone(&tree));
        Ok(tree)
    }

    fn ready_cores(&self) -> Result<Option<Arc<[f64]>>> {
        Ok(match &self.cache()?.cores {
            CoreStatus::Ready { values, .. } => Some(Arc::clone(values)),
            _ => None,
        })
    }
}

/// Validate a query matrix against the fitted width; returns flattened row-major values.
fn validate_queries(input: &[Vec<f32>], n_features: usize) -> Result<Vec<f32>> {
    if input.is_empty() {
        return Err(HdbscanError::InvalidInput(
            "prediction input must contain at least one row".into(),
        ));
    }
    if n_features == 0 {
        return Err(HdbscanError::InvalidInput(
            "prediction input must contain at least one feature".into(),
        ));
    }
    if input.iter().any(|row| row.len() != n_features) {
        return Err(HdbscanError::InvalidInput(format!(
            "prediction input rows must all have exactly {n_features} features"
        )));
    }
    if input.len() > i32::MAX as usize {
        return Err(HdbscanError::InvalidInput(
            "prediction row count exceeds i32".into(),
        ));
    }
    let elements = input
        .len()
        .checked_mul(n_features)
        .ok_or(HdbscanError::Overflow("prediction input element count"))?;
    checked_allocation_bytes(elements, 4, None, "prediction input allocation")?;
    let mut values = Vec::with_capacity(elements);
    for row in input {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(HdbscanError::InvalidInput(
                "prediction input values must all be finite".into(),
            ));
        }
        values.extend_from_slice(row);
    }
    Ok(values)
}

/// Checked bounded batch plan for `rows` query rows against `k` candidate slots.
///
/// Only size overflows halve the batch; backend and pitch errors are returned at once.
pub(crate) fn plan_batch_rows(
    rows: usize,
    k: usize,
    features: usize,
    limit: DeviceAllocationLimit,
    row_pitched: bool,
) -> Result<usize> {
    let mut batch = rows.clamp(1, PREDICTION_BATCH_ROWS);
    loop {
        let fits = check_device_tensor_allocation(
            &[batch, k],
            ELEMENT_BYTES,
            limit,
            row_pitched,
            "prediction neighbor output allocation",
        )
        .and_then(|()| {
            check_device_tensor_allocation(
                &[batch, features],
                ELEMENT_BYTES,
                limit,
                row_pitched,
                "prediction query allocation",
            )
        })
        .and_then(|()| {
            crate::kernels::kd_query_pack_bytes(batch, features, k)
                .ok_or(HdbscanError::Overflow("prediction query pack bytes"))
                .and_then(|bytes| check_device_allocation(bytes, 1, limit, "prediction query pack"))
        });
        match fits {
            Ok(()) => return Ok(batch),
            Err(HdbscanError::Overflow(_)) if batch > 1 => batch /= 2,
            Err(error) => return Err(error),
        }
    }
}

/// Element size of every prediction device and host array (`f32` and `i32`).
const ELEMENT_BYTES: usize = 4;

/// Host bytes of one query output row: label `i32`, strength `f64` and connector `i32`.
const OUTPUT_ROW_BYTES: usize = 4 + 8 + 4;

/// Implicit-heap level count `KdTree::build` derives for `rows` points (same formula).
fn planned_tree_levels(rows: usize) -> Result<usize> {
    let quotient = (rows.saturating_sub(1) / LEAF_SIZE).max(1);
    let levels = (usize::BITS - quotient.leading_zeros()) as usize;
    if levels > MAX_KD_LEVELS {
        return Err(HdbscanError::Overflow("KD-tree level count"));
    }
    Ok(levels)
}

/// Node count of the `levels`-level implicit heap that `KdTree::build` allocates.
fn planned_tree_nodes(levels: usize) -> Result<usize> {
    let nodes = u32::try_from(levels)
        .ok()
        .and_then(|shift| 1usize.checked_shl(shift))
        .and_then(|width| width.checked_sub(1))
        .ok_or(HdbscanError::Overflow("KD-tree node count"))?;
    if nodes > i32::MAX as usize {
        return Err(HdbscanError::Overflow("KD-tree node index"));
    }
    Ok(nodes)
}

/// Checked device bytes of one prepared KD reference: row-pitched data and bounds, the KD index,
/// the four node arrays and the packed copy. Every allocation is checked against the page limit and
/// the live sum is checked against `isize::MAX`.
///
/// Planning passes planned shapes and upload passes the actual tree shapes through this one
/// function, so both agree on what fits.
fn reference_device_bytes(
    rows: usize,
    index_len: usize,
    nodes: usize,
    features: usize,
    limit: DeviceAllocationLimit,
    row_pitched: bool,
) -> Result<usize> {
    let padded = crate::kernels::kd_padded_width(features)
        .ok_or(HdbscanError::Overflow("prediction padded width"))?;
    for (padded_rows, subject) in [
        (rows, "prediction padded reference allocation"),
        (nodes, "prediction padded minimum-bounds allocation"),
        (nodes, "prediction padded maximum-bounds allocation"),
    ] {
        let elements = padded_rows
            .checked_mul(padded)
            .ok_or(HdbscanError::Overflow(subject))?;
        check_device_allocation(elements, ELEMENT_BYTES, limit, subject)?;
    }
    let data = tensor_allocation_bytes(
        &[rows, features],
        ELEMENT_BYTES,
        limit,
        row_pitched,
        "prediction reference allocation",
    )?;
    let index = checked_allocation_bytes(
        index_len,
        ELEMENT_BYTES,
        Some(limit),
        "prediction KD index allocation",
    )?;
    let starts = checked_allocation_bytes(
        nodes,
        ELEMENT_BYTES,
        Some(limit),
        "prediction KD start allocation",
    )?;
    let ends = checked_allocation_bytes(
        nodes,
        ELEMENT_BYTES,
        Some(limit),
        "prediction KD end allocation",
    )?;
    let leaves = checked_allocation_bytes(
        nodes,
        ELEMENT_BYTES,
        Some(limit),
        "prediction KD leaf allocation",
    )?;
    let components = checked_allocation_bytes(
        nodes,
        ELEMENT_BYTES,
        Some(limit),
        "prediction KD node-component allocation",
    )?;
    let bounds_min = tensor_allocation_bytes(
        &[nodes, features],
        ELEMENT_BYTES,
        limit,
        row_pitched,
        "prediction KD minimum bounds",
    )?;
    let bounds_max = tensor_allocation_bytes(
        &[nodes, features],
        ELEMENT_BYTES,
        limit,
        row_pitched,
        "prediction KD maximum bounds",
    )?;
    let pack = crate::kernels::kd_reference_pack_bytes(rows, nodes, features)
        .ok_or(HdbscanError::Overflow("prediction reference pack bytes"))?;
    checked_live_sum(
        &[
            data, index, starts, ends, leaves, components, bounds_min, bounds_max, pack,
        ],
        "prediction reference live bytes",
    )
}

/// Checked total of allocations live together; totals beyond `isize::MAX` are rejected.
fn checked_live_sum(terms: &[usize], subject: &'static str) -> Result<usize> {
    let total = terms
        .iter()
        .try_fold(0usize, |sum, &term| sum.checked_add(term))
        .ok_or(HdbscanError::Overflow(subject))?;
    if total > isize::MAX as usize {
        return Err(HdbscanError::Overflow(subject));
    }
    Ok(total)
}

/// Checked sizing of one prediction call: batch sizes and the planned peak live bytes.
///
/// Planned values, not measurements: each peak is the largest checked live-set total over the call's
/// phases. Totals guard overflow and the Rust allocation limit only; they never promise available
/// device or host memory.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PredictionPlan {
    /// Query rows per device batch.
    pub query_batch: usize,
    /// Reference rows per cold prediction-core batch.
    pub core_batch: usize,
    /// Largest planned device live set over the cold core and warm query phases, in bytes.
    pub device_peak_bytes: usize,
    /// Largest planned host live set over upload, cold core and warm query phases, in bytes.
    pub host_peak_bytes: usize,
}

/// Checks every sizing decision of one call before tree build, upload or core search.
///
/// Shapes come from the same formulas `KdTree::build` and `CallReference::upload` use. Each phase's
/// live set is summed with checked arithmetic: the cache tree is live throughout, and upload clones
/// of geometry and tree overlap it.
fn plan_prediction(
    queries: usize,
    reference: usize,
    features: usize,
    s: usize,
    limit: DeviceAllocationLimit,
    row_pitched: bool,
) -> Result<PredictionPlan> {
    if queries == 0 || queries > i32::MAX as usize {
        return Err(HdbscanError::InvalidInput(
            "prediction row count must be in 1..=i32::MAX".into(),
        ));
    }
    if reference == 0 || reference > i32::MAX as usize || features == 0 {
        return Err(HdbscanError::InvalidInput(
            "prediction reference rows must be in 1..=i32::MAX and features must be positive"
                .into(),
        ));
    }
    let k = candidate_width(s, reference)?;
    let levels = planned_tree_levels(reference)?;
    let nodes = planned_tree_nodes(levels)?;
    let query_batch = plan_batch_rows(queries, k, features, limit, row_pitched)?;
    let core_batch = plan_batch_rows(reference, s, features, limit, row_pitched)?;

    // Device reference: the same checked sizing `CallReference::upload` applies to the actual tree.
    let device_reference =
        reference_device_bytes(reference, reference, nodes, features, limit, row_pitched)?;
    let core_pack = crate::kernels::kd_query_pack_bytes(core_batch, features, s)
        .ok_or(HdbscanError::Overflow("prediction core pack bytes"))?;
    let core_device = checked_live_sum(
        &[device_reference, core_pack],
        "prediction core device bytes",
    )?;
    let query_source = tensor_allocation_bytes(
        &[query_batch, features],
        ELEMENT_BYTES,
        limit,
        row_pitched,
        "prediction query allocation",
    )?;
    let query_pack = crate::kernels::kd_query_pack_bytes(query_batch, features, k)
        .ok_or(HdbscanError::Overflow("prediction query pack bytes"))?;
    let query_device = checked_live_sum(
        &[device_reference, query_source, query_pack],
        "prediction query device bytes",
    )?;

    // Host: input, outputs and the cache tree; cold upload clones geometry and the tree again.
    let input_elements = queries
        .checked_mul(features)
        .ok_or(HdbscanError::Overflow("prediction input allocation"))?;
    let input = checked_allocation_bytes(
        input_elements,
        ELEMENT_BYTES,
        None,
        "prediction input allocation",
    )?;
    let outputs = checked_allocation_bytes(
        queries,
        OUTPUT_ROW_BYTES,
        None,
        "prediction output allocation",
    )?;
    let bounds_elements = nodes
        .checked_mul(features)
        .ok_or(HdbscanError::Overflow("KD-tree host bytes"))?;
    let tree_elements = reference
        .checked_add(
            nodes
                .checked_mul(3)
                .ok_or(HdbscanError::Overflow("KD-tree host bytes"))?,
        )
        .and_then(|sum| sum.checked_add(bounds_elements.checked_mul(2)?))
        .ok_or(HdbscanError::Overflow("KD-tree host bytes"))?;
    let tree_host =
        checked_allocation_bytes(tree_elements, ELEMENT_BYTES, None, "KD-tree host bytes")?;
    let base_host = checked_live_sum(&[input, outputs, tree_host], "prediction live host bytes")?;
    let geometry = checked_allocation_bytes(
        reference
            .checked_mul(features)
            .ok_or(HdbscanError::Overflow("prediction upload geometry clone"))?,
        ELEMENT_BYTES,
        None,
        "prediction upload geometry clone",
    )?;
    let upload_host = checked_live_sum(
        &[base_host, geometry, tree_host],
        "prediction upload host bytes",
    )?;

    // Cold cores: the f64 vector plus its published copy, and candidate readbacks of ids and
    // distances.
    let cores_host = checked_allocation_bytes(reference, 16, None, "prediction core allocation")?;
    let core_readback_slots = core_batch
        .checked_mul(s)
        .ok_or(HdbscanError::Overflow("prediction core readback"))?;
    let core_readback =
        checked_allocation_bytes(core_readback_slots, 16, None, "prediction core readback")?;
    let core_host = checked_live_sum(
        &[base_host, cores_host, core_readback],
        "prediction core host bytes",
    )?;

    // Warm batch: published cores, the query copy and candidate readbacks.
    let cores = checked_allocation_bytes(reference, 8, None, "prediction cores")?;
    let query_host = checked_allocation_bytes(
        query_batch
            .checked_mul(features)
            .ok_or(HdbscanError::Overflow("prediction query batch copy"))?,
        ELEMENT_BYTES,
        None,
        "prediction query batch copy",
    )?;
    let candidate_readback = checked_allocation_bytes(
        query_batch
            .checked_mul(k)
            .ok_or(HdbscanError::Overflow("prediction candidate readback"))?,
        16,
        None,
        "prediction candidate readback",
    )?;
    let query_host_live = checked_live_sum(
        &[base_host, cores, query_host, candidate_readback],
        "prediction query host bytes",
    )?;

    Ok(PredictionPlan {
        query_batch,
        core_batch,
        device_peak_bytes: core_device.max(query_device),
        host_peak_bytes: upload_host.max(core_host).max(query_host_live),
    })
}

/// Strict runtime device plus its allocation limits, resolved before any cold work.
struct Target {
    device: Device,
    limit: DeviceAllocationLimit,
    row_pitched: bool,
}

impl Target {
    fn resolve(backend: BackendChoice) -> Result<Self> {
        let (device, actual) = selected_device(backend).map_err(map_backend_error)?;
        if actual != backend {
            return Err(HdbscanError::BackendUnavailable(format!(
                "prediction runtime backend {backend:?} resolved to {actual:?}"
            )));
        }
        Ok(Self {
            limit: allocation_limit(&device)?,
            row_pitched: matches!(backend, BackendChoice::Cuda { .. }),
            device,
        })
    }
}

/// Call-local device reference: one upload per prediction call, released on return.
struct CallReference {
    device: Device,
    data: Tensor<2>,
    reference: DeviceKdReference<Dispatch>,
}

impl CallReference {
    fn upload(state: &FittedState, tree: &KdTree, target: &Target) -> Result<Self> {
        let device = target.device.clone();
        let limit = target.limit;
        let row_pitched = target.row_pitched;
        let n = state.n_finite();
        let features = state.n_features;
        // Validates the actual tree arrays with the same reference sizing that planning uses.
        reference_device_bytes(
            n,
            tree.indices.len(),
            tree.starts.len(),
            features,
            limit,
            row_pitched,
        )?;
        #[cfg(test)]
        tests::count_upload();
        #[cfg(feature = "tracing")]
        let _span = tracing::info_span!("prediction_reference_upload_prepack").entered();
        let data = Tensor::<2>::from_data(
            TensorData::new(state.geometry.clone(), [n, features]),
            &device,
        );
        let device_tree = upload_kd_tree(tree, &device);
        let reference = <Dispatch as HdbscanOps>::prepare_kd_reference(
            data.clone().into_dispatch(),
            device_tree,
        );
        #[cfg(feature = "tracing")]
        device.sync().map_err(|error| {
            HdbscanError::Numerical(format!("profiling synchronization failed: {error}"))
        })?;
        Ok(Self {
            device,
            data,
            reference,
        })
    }

    /// Exact KD query; returns row-major compact IDs and squared distances.
    fn query(
        &self,
        query: Tensor<2>,
        k: usize,
        exclude_self: bool,
        query_offset: usize,
    ) -> Result<(Vec<i32>, Vec<f32>)> {
        let rows = query.dims()[0];
        let width = u32::try_from(k).map_err(|_| HdbscanError::Overflow("prediction width"))?;
        #[cfg(feature = "tracing")]
        let search_span = tracing::info_span!("prediction_kd_search", rows, k).entered();
        let offset = u32::try_from(query_offset)
            .map_err(|_| HdbscanError::Overflow("prediction query offset"))?;
        let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd_query(
            query.into_dispatch(),
            self.reference.clone(),
            width,
            exclude_self,
            offset,
        );
        let ids = Tensor::<2, Int>::from_dispatch(ids);
        let distances = Tensor::<2>::from_dispatch(distances);
        if ids.dims() != [rows, k] || distances.dims() != [rows, k] {
            return Err(HdbscanError::Numerical(
                "prediction KD query returned wrong shapes".into(),
            ));
        }
        #[cfg(feature = "tracing")]
        {
            self.device.sync().map_err(|error| {
                HdbscanError::Numerical(format!("profiling synchronization failed: {error}"))
            })?;
            drop(search_span);
        }
        #[cfg(feature = "tracing")]
        let _readback_span = tracing::info_span!("prediction_readback", rows).entered();
        let [id_data, distance_data] = Transaction::default()
            .register(ids)
            .register(distances)
            .try_execute()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?
            .try_into()
            .map_err(|_| {
                HdbscanError::Numerical("prediction readback returned wrong arity".into())
            })?;
        let ids = id_data
            .try_to_vec::<i32>()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
        let distances = distance_data
            .try_to_vec::<f32>()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
        if ids.len() != rows * k || distances.len() != rows * k {
            return Err(HdbscanError::Numerical(
                "prediction readback returned wrong lengths".into(),
            ));
        }
        Ok((ids, distances))
    }
}

/// Compute validated inclusive-self prediction cores (`k = s`) through the call reference.
/// `batch` is the preflighted core batch row count.
fn compute_cores(state: &FittedState, call: &CallReference, batch: usize) -> Result<Vec<f64>> {
    #[cfg(test)]
    tests::maybe_fail(tests::Injection::Cores)?;
    #[cfg(feature = "tracing")]
    let _span = tracing::info_span!("prediction_training_cores").entered();
    let n = state.n_finite();
    let s = state.min_samples;
    if s == 1 {
        return Ok(vec![0.0; n]);
    }
    let mut cores = Vec::with_capacity(n);
    for start in (0..n).step_by(batch) {
        let end = start.saturating_add(batch).min(n);
        let query = call.data.clone().slice([start..end, 0..state.n_features]);
        let (ids, distances) = call.query(query, s, false, start)?;
        if ids.iter().any(|&id| id < 0 || id as usize >= n) {
            return Err(HdbscanError::Numerical(
                "prediction core search returned a missing or sentinel neighbor".into(),
            ));
        }
        let batch_cores = prediction_core_from_squared_rows(&distances, s)?;
        if batch_cores.len() != end - start {
            return Err(HdbscanError::Numerical(
                "prediction core batch returned wrong length".into(),
            ));
        }
        cores.extend(batch_cores);
    }
    state.hierarchy.validate_prediction_core(&cores)?;
    Ok(cores)
}

/// Preflights every sizing decision of `m` query rows before tree construction, upload or core
/// search: output columns, the strict runtime device and its planned batches and peaks.
fn preflight(
    state: &FittedState,
    runtime: &PredictionRuntime,
    m: usize,
) -> Result<(Target, PredictionPlan)> {
    for (elements, size, subject) in [
        (m, 4, "prediction label allocation"),
        (m, 8, "prediction strength allocation"),
        (m, 4, "prediction connector allocation"),
    ] {
        checked_allocation_bytes(elements, size, None, subject)?;
    }
    let target = Target::resolve(runtime.backend())?;
    let plan = plan_prediction(
        m,
        state.n_finite(),
        state.n_features,
        state.min_samples,
        target.limit,
        target.row_pitched,
    )?;
    Ok((target, plan))
}

/// Checked plan `approximate_predict` would use for `input`; `None` for the no-cluster-tree
/// shortcut, which plans no device work.
pub(crate) fn prediction_plan(
    state: &FittedState,
    runtime: &PredictionRuntime,
    input: &[Vec<f32>],
) -> Result<Option<PredictionPlan>> {
    validate_queries(input, state.n_features)?;
    if state.hierarchy.no_cluster_tree() {
        return Ok(None);
    }
    preflight(state, runtime, input.len()).map(|(_, plan)| Some(plan))
}

/// Bounded strict-backend exact-KD `approximate_predict` over canonical state.
pub(crate) fn approximate_predict(
    state: &FittedState,
    runtime: &PredictionRuntime,
    input: &[Vec<f32>],
) -> Result<PredictionResult> {
    let values = validate_queries(input, state.n_features)?;
    let m = input.len();
    if state.hierarchy.no_cluster_tree() {
        let shortcut = PredictionColumns::no_cluster_tree(m);
        return Ok(PredictionResult {
            labels: shortcut.labels,
            probabilities: shortcut.probabilities,
            connecting_points: shortcut.connecting_points,
            no_cluster_tree: true,
        });
    }
    let n = state.n_finite();
    let s = state.min_samples;
    let k = candidate_width(s, n)?;
    let (target, plan) = preflight(state, runtime, m)?;
    let features = state.n_features;
    let PredictionPlan {
        query_batch: batch,
        core_batch,
        ..
    } = plan;
    let tree = runtime.tree(state)?;
    let call = CallReference::upload(state, &tree, &target)?;
    let cores = match runtime.ready_cores()? {
        Some(cores) => cores,
        None => {
            let _init = runtime.init()?;
            match runtime.ready_cores()? {
                Some(cores) => cores,
                None => {
                    let cores: Arc<[f64]> = compute_cores(state, &call, core_batch)?.into();
                    runtime.cache()?.cores = CoreStatus::Ready {
                        rank: s,
                        values: Arc::clone(&cores),
                    };
                    cores
                }
            }
        }
    };

    let mut columns = PredictionColumns::default();
    for start in (0..m).step_by(batch) {
        let end = start.saturating_add(batch).min(m);
        let query = Tensor::<2>::from_data(
            TensorData::new(
                values[start * features..end * features].to_vec(),
                [end - start, features],
            ),
            &call.device,
        );
        let (ids, distances) = call.query(query, k, false, 0)?;
        #[cfg(feature = "tracing")]
        let _math_span = tracing::info_span!("prediction_candidate_math").entered();
        state
            .hierarchy
            .predict_squared_batch(&ids, &distances, &cores, s, &mut columns)?;
    }
    drop(call);
    if columns.len() != m {
        return Err(HdbscanError::Numerical(
            "prediction returned wrong row count".into(),
        ));
    }
    Ok(PredictionResult {
        labels: columns.labels,
        probabilities: columns.probabilities,
        connecting_points: columns.connecting_points,
        no_cluster_tree: false,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::Cell;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub(crate) enum Injection {
        Tree,
        Cores,
    }

    thread_local! {
        static FAIL: Cell<Option<Injection>> = const { Cell::new(None) };
        static UPLOADS: Cell<usize> = const { Cell::new(0) };
    }

    pub(crate) fn maybe_fail(point: Injection) -> Result<()> {
        if FAIL.with(|fail| fail.get()) == Some(point) {
            return Err(HdbscanError::Numerical(
                "injected prediction failure".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn count_upload() {
        UPLOADS.with(|uploads| uploads.set(uploads.get() + 1));
    }

    fn uploads() -> usize {
        UPLOADS.with(Cell::get)
    }

    fn inject(point: Option<Injection>) {
        FAIL.with(|fail| fail.set(point));
    }

    fn blobs(rows_per_group: usize) -> Vec<Vec<f32>> {
        (0..rows_per_group)
            .flat_map(|i| {
                let jitter = (i % 7) as f32 * 0.03 + (i / 7) as f32 * 0.011;
                [
                    vec![jitter, 0.5 * jitter],
                    vec![50.0 + jitter, 50.0 - jitter],
                ]
            })
            .collect()
    }

    fn fit(search: crate::HdbscanSearch) -> crate::FittedHDBSCAN {
        let config = HdbscanConfig {
            min_cluster_size: 5,
            min_samples: Some(3),
            ..HdbscanConfig::default()
        };
        crate::HDBSCAN::new(config, BackendChoice::Cpu)
            .with_search(search)
            .fit(&blobs(30))
            .unwrap()
    }

    #[test]
    fn canonical_state_holds_complete_outputs_and_statuses() {
        let model = fit(crate::HdbscanSearch::BruteForce);
        let state = &model.state;
        assert_eq!(state.labels, model.labels);
        assert_eq!(state.probabilities, model.probabilities);
        assert_eq!(state.linkage.len(), model.linkage.len());
        assert_eq!(state.condensed.parent, model.condensed.parent);
        assert_eq!(state.mst.len(), model.mst.len());
        assert_eq!(state.neighbors, model.neighbors);
        assert_eq!(state.core_distances, model.core_distances);
        assert_eq!(state.finite_index, model.finite_index);
        assert_eq!(state.geometry.len(), 60 * 2);
        assert_eq!(state.min_samples, 3);
        assert!(!state.hierarchy.no_cluster_tree());
        assert!(state.canonical_host_bytes().unwrap() >= state.geometry.len() * 4);
        let status = model.prediction_cache_status().unwrap();
        assert!(!status.tree_ready, "brute-force fit builds the tree lazily");
        assert_eq!(status.cores, PredictionCoreStatus::Uninitialized);
        assert_eq!(model.runtime.backend(), BackendChoice::Cpu);
    }

    #[test]
    fn kd_fit_retains_host_tree() {
        let model = fit(crate::HdbscanSearch::KdTree);
        let status = model.prediction_cache_status().unwrap();
        assert!(status.tree_ready);
        let tree = model.runtime.cache().unwrap().tree.clone().unwrap();
        assert!(host_tree_bytes(&tree).unwrap() > 0);
        assert_eq!(
            *tree,
            KdTree::build(&model.state.geometry, 60, 2).unwrap(),
            "retained tree is the fit-created builder tree"
        );
    }

    #[test]
    fn injected_tree_failure_publishes_nothing_and_retry_succeeds() {
        let model = fit(crate::HdbscanSearch::BruteForce);
        inject(Some(Injection::Tree));
        let error = model.approximate_predict(&[vec![0.1, 0.1]]).unwrap_err();
        inject(None);
        assert!(matches!(error, HdbscanError::Numerical(_)));
        let status = model.prediction_cache_status().unwrap();
        assert!(!status.tree_ready);
        assert_eq!(status.cores, PredictionCoreStatus::Uninitialized);
        let result = model.approximate_predict(&[vec![0.1, 0.1]]).unwrap();
        assert_eq!(result.labels[0], model.labels[0]);
    }

    #[test]
    fn injected_core_failure_keeps_tree_and_allows_retry() {
        let model = fit(crate::HdbscanSearch::BruteForce);
        let labels_before = model.state.labels.clone();
        inject(Some(Injection::Cores));
        assert!(model.approximate_predict(&[vec![0.1, 0.1]]).is_err());
        inject(None);
        let status = model.prediction_cache_status().unwrap();
        assert!(status.tree_ready, "valid tree survives core failure");
        assert_eq!(status.cores, PredictionCoreStatus::Uninitialized);
        assert_eq!(model.state.labels, labels_before);
        model.approximate_predict(&[vec![0.1, 0.1]]).unwrap();
        assert_eq!(
            model.prediction_cache_status().unwrap().cores,
            PredictionCoreStatus::Ready { rank: 3, len: 60 }
        );
    }

    #[test]
    fn one_upload_per_call_across_cold_cores_and_batches() {
        let model = fit(crate::HdbscanSearch::BruteForce);
        let queries: Vec<Vec<f32>> = (0..PREDICTION_BATCH_ROWS + 3)
            .map(|i| vec![(i % 5) as f32 * 0.01, 0.0])
            .collect();
        let before = uploads();
        model.approximate_predict(&queries).unwrap();
        assert_eq!(uploads() - before, 1, "cold call uploads once");
        let before = uploads();
        model.approximate_predict(&queries).unwrap();
        assert_eq!(uploads() - before, 1, "warm call uploads once");
    }

    #[test]
    fn no_cluster_tree_shortcut_skips_width_and_cold_work() {
        // Three collinear points with s=50 > n: no cluster tree, so the width is never checked.
        let config = HdbscanConfig {
            min_cluster_size: 5,
            min_samples: Some(50),
            ..HdbscanConfig::default()
        };
        let model = crate::HDBSCAN::new(config, BackendChoice::Cpu)
            .fit(&[vec![0.0], vec![1.0], vec![2.0]])
            .unwrap();
        assert!(model.state.hierarchy.no_cluster_tree());
        assert_eq!(
            model.prediction_cache_status().unwrap().cores,
            PredictionCoreStatus::NotRequiredNoClusterTree
        );
        let before = uploads();
        let result = model.approximate_predict(&[vec![0.5]]).unwrap();
        assert_eq!(
            result,
            PredictionResult {
                labels: vec![-1],
                probabilities: vec![0.0],
                connecting_points: vec![-1],
                no_cluster_tree: true,
            }
        );
        assert_eq!(uploads(), before);
        assert!(!model.prediction_cache_status().unwrap().tree_ready);
    }

    #[test]
    fn cluster_tree_width_violation_rejects_before_cold_work() {
        let mut model = fit(crate::HdbscanSearch::BruteForce);
        assert!(!model.state.hierarchy.no_cluster_tree());
        // Fitting cannot yield a nonempty tree with 2*s > n, so the state width is raised directly.
        let n = model.state.n_finite();
        Arc::get_mut(&mut model.state).unwrap().min_samples = n / 2 + 1;
        let before = uploads();
        let error = model.approximate_predict(&[vec![0.0, 0.0]]).unwrap_err();
        assert!(matches!(error, HdbscanError::InvalidInput(_)));
        assert_eq!(uploads(), before);
        let status = model.prediction_cache_status().unwrap();
        assert!(!status.tree_ready);
        assert_eq!(status.cores, PredictionCoreStatus::Uninitialized);
    }

    #[test]
    fn batch_plan_halves_under_small_device_limits() {
        let limit = DeviceAllocationLimit {
            max_page_bytes: 4096,
            alignment: 16,
        };
        let rows = plan_batch_rows(10_000, 8, 2, limit, false).unwrap();
        assert!(rows >= 1 && rows * 8 * 4 <= 4096);
        let tiny = DeviceAllocationLimit {
            max_page_bytes: 16,
            alignment: 16,
        };
        assert!(plan_batch_rows(10, 1 << 20, 2, tiny, false).is_err());
    }
}

#[cfg(test)]
mod resource_plan_tests {
    use super::*;

    const UNBOUNDED: DeviceAllocationLimit = DeviceAllocationLimit {
        max_page_bytes: usize::MAX,
        alignment: 256,
    };

    #[test]
    fn tree_shape_matches_kd_build() {
        for rows in [1usize, 40, 41, 81, 1000] {
            let data: Vec<f32> = (0..rows).map(|i| i as f32).collect();
            let tree = KdTree::build(&data, rows, 1).unwrap();
            let levels = planned_tree_levels(rows).unwrap();
            assert_eq!(tree.n_levels, levels, "rows={rows}");
            assert_eq!(
                tree.starts.len(),
                planned_tree_nodes(levels).unwrap(),
                "rows={rows}"
            );
        }
    }

    #[test]
    fn tree_levels_and_node_index_bounds() {
        assert_eq!(planned_tree_levels(i32::MAX as usize).unwrap(), 26);
        // 32 levels is the deepest the stack bound admits, but its heap exceeds the i32 node index.
        assert_eq!(
            planned_tree_levels(40 * (1 << 31) + 1).unwrap(),
            MAX_KD_LEVELS
        );
        assert!(planned_tree_nodes(MAX_KD_LEVELS).is_err());
        assert!(planned_tree_levels(40 * (1 << 32) + 1).is_err());
        assert!(planned_tree_levels(usize::MAX).is_err());
        assert_eq!(planned_tree_nodes(31).unwrap(), (1 << 31) - 1);
    }

    #[test]
    fn rank_and_row_bounds_reject_before_sizing() {
        for (queries, reference, s) in [
            (1, 5, 3),
            (1, 6, 0),
            (0, 6, 3),
            (i32::MAX as usize + 1, 6, 3),
        ] {
            assert!(
                matches!(
                    plan_prediction(queries, reference, 2, s, UNBOUNDED, false),
                    Err(HdbscanError::InvalidInput(_))
                ),
                "queries={queries} reference={reference} s={s}"
            );
        }
        let plan = plan_prediction(37, 90, 2, 3, UNBOUNDED, false).unwrap();
        assert_eq!((plan.query_batch, plan.core_batch), (37, 90));
    }

    fn limit_with(alignment: usize) -> DeviceAllocationLimit {
        DeviceAllocationLimit {
            max_page_bytes: usize::MAX,
            alignment,
        }
    }

    #[test]
    fn pitch_rounds_rows_to_alignment() {
        use crate::estimator::tensor_raw_bytes;
        let pitched = |rows, width| {
            tensor_raw_bytes(&[rows, width], ELEMENT_BYTES, UNBOUNDED, true, "t").unwrap()
        };
        let flat = |rows, width| {
            tensor_raw_bytes(&[rows, width], ELEMENT_BYTES, UNBOUNDED, false, "t").unwrap()
        };
        assert_eq!(flat(5, 3), 5 * 12);
        assert_eq!(pitched(5, 3), 5 * 16); // 12-byte rows pitch to 16.
        assert_eq!(pitched(7, 1), 7 * 4); // width 1 keeps element pitch.
        assert_eq!(pitched(5, 40), 5 * 256); // 160-byte rows pitch to 256.
        assert_eq!(pitched(3, 100), 3 * 512); // 400-byte rows ceil to 512 at 256 alignment.
        assert_eq!(
            tensor_allocation_bytes(&[5, 3], ELEMENT_BYTES, UNBOUNDED, true, "t").unwrap(),
            256
        ); // 80 bytes pad to 256.
        for (rows, width) in [(5, 3), (7, 1), (5, 40), (3, 100)] {
            assert!(pitched(rows, width) >= flat(rows, width));
        }
    }

    #[test]
    fn small_alignment_pitched_path_is_a_typed_error() {
        use crate::estimator::tensor_raw_bytes;
        for alignment in 1..16 {
            let limit = limit_with(alignment);
            // Width-1 tensors keep element pitch and never consult the alignment.
            assert_eq!(
                tensor_raw_bytes(&[7, 1], ELEMENT_BYTES, limit, true, "t").unwrap(),
                7 * 4,
                "alignment={alignment}"
            );
            // Wider pitched rows need the 16-byte minimum; the error is typed, not a panic.
            assert!(
                matches!(
                    tensor_raw_bytes(&[5, 3], ELEMENT_BYTES, limit, true, "t"),
                    Err(HdbscanError::BackendUnavailable(_))
                ),
                "alignment={alignment}"
            );
            assert!(
                matches!(
                    reference_device_bytes(5, 5, 9, 3, limit, true),
                    Err(HdbscanError::BackendUnavailable(_))
                ),
                "alignment={alignment}"
            );
            // Flat tensors never pitch, so the alignment is irrelevant to them.
            assert!(
                reference_device_bytes(5, 5, 9, 3, limit, false).is_ok(),
                "alignment={alignment}"
            );
        }
        assert!(matches!(
            plan_prediction(10, 30, 3, 3, limit_with(8), true),
            Err(HdbscanError::BackendUnavailable(_))
        ));
        assert!(matches!(
            plan_batch_rows(10, 3, 2, limit_with(8), true),
            Err(HdbscanError::BackendUnavailable(_))
        ));
        // Alignment 16 is the smallest pitched alignment the rule can express.
        assert_eq!(
            crate::estimator::tensor_raw_bytes(&[5, 3], ELEMENT_BYTES, limit_with(16), true, "t")
                .unwrap(),
            5 * 16
        );
    }

    #[test]
    fn plan_and_upload_agree_on_reference_sizing() {
        // Planned shapes must size exactly like the actual tree `KdTree::build` produces.
        for (rows, features) in [(41usize, 2usize), (81, 3), (1000, 5)] {
            let data: Vec<f32> = (0..rows * features).map(|i| i as f32).collect();
            let tree = KdTree::build(&data, rows, features).unwrap();
            let levels = planned_tree_levels(rows).unwrap();
            let nodes = planned_tree_nodes(levels).unwrap();
            for limit in [UNBOUNDED, limit_with(16), limit_with(256)] {
                for pitched in [false, true] {
                    assert_eq!(
                        reference_device_bytes(
                            rows,
                            tree.indices.len(),
                            tree.starts.len(),
                            features,
                            limit,
                            pitched
                        )
                        .unwrap(),
                        reference_device_bytes(rows, rows, nodes, features, limit, pitched)
                            .unwrap(),
                        "rows={rows} features={features} pitched={pitched}"
                    );
                }
            }
        }
    }

    #[test]
    fn plan_peaks_are_the_checked_helper_totals() {
        let (reference, features, s, queries) = (90usize, 2usize, 3usize, 37usize);
        let limit = limit_with(16);
        let plan = plan_prediction(queries, reference, features, s, limit, true).unwrap();
        let k = candidate_width(s, reference).unwrap();
        let nodes = planned_tree_nodes(planned_tree_levels(reference).unwrap()).unwrap();
        let device_reference =
            reference_device_bytes(reference, reference, nodes, features, limit, true).unwrap();
        let core = device_reference
            + crate::kernels::kd_query_pack_bytes(plan.core_batch, features, s).unwrap();
        let query = device_reference
            + tensor_allocation_bytes(
                &[plan.query_batch, features],
                ELEMENT_BYTES,
                limit,
                true,
                "t",
            )
            .unwrap()
            + crate::kernels::kd_query_pack_bytes(plan.query_batch, features, k).unwrap();
        assert_eq!(plan.device_peak_bytes, core.max(query));
        assert_eq!(
            plan.query_batch,
            plan_batch_rows(queries, k, features, limit, true).unwrap()
        );
        assert_eq!(
            plan.core_batch,
            plan_batch_rows(reference, s, features, limit, true).unwrap()
        );
        assert!(plan.host_peak_bytes >= reference * features * ELEMENT_BYTES);
    }

    #[test]
    fn device_tensor_rejects_page_and_overflow() {
        // 800 bytes pad to 1024; 1200 bytes pad to 1280 and exceed the page.
        let small = DeviceAllocationLimit {
            max_page_bytes: 1024,
            alignment: 256,
        };
        let bytes = |rows, width, limit, pitched| {
            tensor_allocation_bytes(&[rows, width], ELEMENT_BYTES, limit, pitched, "t")
        };
        assert!(bytes(100, 3, small, false).is_err());
        assert_eq!(bytes(100, 2, small, false).unwrap(), 1024);
        assert!(bytes(usize::MAX, 1, UNBOUNDED, false).is_err());
        assert!(bytes(1 << 62, 1, UNBOUNDED, true).is_err());
    }

    #[test]
    fn live_sums_reject_overlap_of_individually_valid_allocations() {
        let half = isize::MAX as usize / 2 + 1;
        assert!(checked_live_sum(&[half], "t").is_ok());
        assert!(checked_live_sum(&[half, half], "t").is_err());
        assert_eq!(checked_live_sum(&[3, 4, 5], "t").unwrap(), 12);
    }

    #[test]
    fn overlapping_reference_live_set_rejected_before_cold_work() {
        // Data and packed reference each fit one allocation, but together they exceed isize::MAX.
        let rows = i32::MAX as usize;
        assert!(matches!(
            plan_prediction(1, rows, 1 << 29, 1, UNBOUNDED, false),
            Err(HdbscanError::Overflow("prediction reference live bytes"))
        ));
        // Halving features keeps the same live set inside isize::MAX.
        assert!(plan_prediction(1, rows, 1 << 28, 1, UNBOUNDED, false).is_ok());
    }
}
