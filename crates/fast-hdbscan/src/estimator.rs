//! Exact HDBSCAN estimator orchestration.

use burn::{
    backend::{Dispatch, DispatchDevice},
    tensor::{Device, Int, Tensor, TensorData, Transaction},
};
use fast_cubecl_common::{
    AnnError, BackendChoice, ann_preflight, backend::NeighborOps, neighbors as common_neighbors,
    selected_device,
};
use fast_hdbscan_core::{
    HdbscanConfig, HdbscanError, HostClustering, MstGraph,
    condense::CondensedTree,
    dsu::RankDsu,
    error::Result,
    finite::{
        compact_rows, flatten_input, remap_condensed, remap_linkage, scatter_labels,
        scatter_probabilities, split_finite,
    },
    knn_kruskal::{KnnGraph, knn_kruskal},
    labels::dbscan_cut,
    linkage::{LinkageRow, SortedMstEdge},
    mst::{ComponentReductionPlan, finalize_mst, init_from_knn, merge_component_winners},
};

use crate::backend::{DeviceKdTree, HdbscanOps};

pub(crate) const NEIGHBOR_BATCH_ROWS: usize = 256;
const KNN_GRAPH_EXACT_MAX_ROWS: usize = 4096;

/// Comparator policy for kNN graph construction.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnSource {
    Exact,
    /// Explicit ANN comparator, independent of the production width limit.
    Ann,
    Auto,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnnNeighborSource {
    Exact,
    Approximate,
}

/// Graph evidence in finite-row indexing. Distances are squared Euclidean.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct KnnDiagnostics {
    pub source: KnnNeighborSource,
    pub neighbors: Vec<i32>,
    pub distances_sq: Vec<f32>,
    pub k: usize,
    pub bridges: usize,
    pub finite_weight_sum: f64,
    pub allocation_bytes: usize,
}

pub(crate) fn resolve_graph(requested: MstGraph, features: usize) -> MstGraph {
    match requested {
        MstGraph::Auto if features > 30 => MstGraph::KnnGraph,
        MstGraph::Auto => MstGraph::Complete,
        graph => graph,
    }
}

pub(crate) fn graph_width(n: usize, min_samples: usize, knn_k: Option<usize>) -> Result<usize> {
    let width = match knn_k {
        Some(width) => width,
        None => min_samples
            .checked_mul(3)
            .ok_or(HdbscanError::Overflow("default kNN graph width"))?
            .max(15),
    };
    Ok(width.max(min_samples).min(n.saturating_sub(1)))
}

/// Whether the kNN graph over `n` finite rows uses ANN for the given source policy.
pub(crate) fn knn_uses_ann(source: KnnSource, n: usize) -> bool {
    match source {
        KnnSource::Exact => false,
        KnnSource::Ann => true,
        KnnSource::Auto => n > KNN_GRAPH_EXACT_MAX_ROWS,
    }
}

pub(crate) fn map_ann_error(error: AnnError) -> HdbscanError {
    match error {
        AnnError::InvalidInput(message) => HdbscanError::InvalidInput(message),
        AnnError::Capability(message) => HdbscanError::BackendUnavailable(message),
        AnnError::Workspace(message) => HdbscanError::Numerical(message),
        AnnError::Overflow(_) => HdbscanError::Overflow("ANN workspace"),
    }
}

#[cfg(feature = "tracing")]
fn profile_sync(device: &Device) -> Result<()> {
    device.sync().map_err(|error| {
        HdbscanError::Numerical(format!("profiling synchronization failed: {error}"))
    })
}

/// Search algorithm used by the estimator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HdbscanSearch {
    /// Select KD-tree at 256..=2048 finite rows, up to 30 features, and min_samples <=32;
    /// at 2049..=4096 rows, use the stricter 10-feature limit.
    /// This conservative region follows the 38-case fixture/synthetic Criterion campaign
    /// (100 samples, 95% confidence, three independent runs per CPU/CUDA case; evidence
    /// recorded in artifact151). The 2049..=4096 range is a low-dimensional/min_samples
    /// interpolation policy, not a separately measured region or speed guarantee.
    #[default]
    Auto,
    /// Exact common squared-distance neighbors and brute-force Borůvka candidates.
    BruteForce,
    /// Exact KD-tree neighbors and Borůvka candidate traversal.
    KdTree,
}

impl HdbscanSearch {
    fn resolve(
        self,
        finite_rows: usize,
        n_features: usize,
        min_samples: usize,
        backend: BackendChoice,
    ) -> Self {
        match self {
            Self::Auto => match backend {
                BackendChoice::Cpu | BackendChoice::Cuda { .. }
                    if (256..=2048).contains(&finite_rows)
                        && n_features <= 30
                        && min_samples <= 32 =>
                {
                    Self::KdTree
                }
                BackendChoice::Cpu | BackendChoice::Cuda { .. }
                    if (2049..=4096).contains(&finite_rows)
                        && n_features <= 10
                        && min_samples <= 32 =>
                {
                    Self::KdTree
                }
                BackendChoice::Cpu | BackendChoice::Cuda { .. } | BackendChoice::Auto => {
                    Self::BruteForce
                }
            },
            route => route,
        }
    }
}

fn map_backend_error(error: fast_cubecl_common::BackendError) -> HdbscanError {
    match error {
        fast_cubecl_common::BackendError::Unavailable(message) => {
            HdbscanError::BackendUnavailable(message)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DeviceAllocationLimit {
    max_page_bytes: usize,
    alignment: usize,
}

fn allocation_limit(device: &Device) -> Result<DeviceAllocationLimit> {
    let DispatchDevice::Cube(cube_device) = device.as_dispatch();
    let client = cube_device.client();
    let memory = &client.properties().memory;
    let max_page_bytes = usize::try_from(memory.max_page_size)
        .map_err(|_| HdbscanError::Overflow("device maximum allocation size"))?;
    let alignment = usize::try_from(memory.alignment)
        .map_err(|_| HdbscanError::Overflow("device allocation alignment"))?;
    if max_page_bytes == 0 || alignment == 0 {
        return Err(HdbscanError::BackendUnavailable(
            "selected device reports invalid CubeCL allocation properties".into(),
        ));
    }
    Ok(DeviceAllocationLimit {
        max_page_bytes,
        alignment,
    })
}

fn checked_allocation_bytes(
    elements: usize,
    element_size: usize,
    limit: Option<DeviceAllocationLimit>,
    subject: &'static str,
) -> Result<usize> {
    let bytes = elements
        .checked_mul(element_size)
        .ok_or(HdbscanError::Overflow(subject))?;
    if bytes > isize::MAX as usize {
        return Err(HdbscanError::Overflow(subject));
    }
    let Some(limit) = limit else {
        return Ok(bytes);
    };
    let padded_bytes = bytes
        .checked_add(limit.alignment - 1)
        .ok_or(HdbscanError::Overflow(subject))?
        / limit.alignment
        * limit.alignment;
    if padded_bytes > limit.max_page_bytes {
        return Err(HdbscanError::Overflow(subject));
    }
    Ok(padded_bytes)
}

fn check_device_allocation(
    elements: usize,
    element_size: usize,
    limit: DeviceAllocationLimit,
    subject: &'static str,
) -> Result<()> {
    checked_allocation_bytes(elements, element_size, Some(limit), subject).map(|_| ())
}

fn check_device_tensor_allocation(
    shape: &[usize],
    element_size: usize,
    limit: DeviceAllocationLimit,
    row_pitched: bool,
    subject: &'static str,
) -> Result<()> {
    let mut elements = 1usize;
    for dimension in shape {
        elements = elements
            .checked_mul(*dimension)
            .ok_or(HdbscanError::Overflow(subject))?;
    }
    let bytes = if row_pitched && shape.len() > 1 {
        let width = *shape.last().expect("rank is greater than one");
        let height = shape[..shape.len() - 1]
            .iter()
            .try_fold(1usize, |rows, dimension| rows.checked_mul(*dimension))
            .ok_or(HdbscanError::Overflow(subject))?;
        let row_bytes = width
            .checked_mul(element_size)
            .ok_or(HdbscanError::Overflow(subject))?;
        // Matches CubeCL's CUDA PitchedMemoryLayoutPolicy::optimal_align heuristic.
        let pitch_alignment = if width == 1 {
            element_size
        } else {
            row_bytes
                .checked_next_power_of_two()
                .ok_or(HdbscanError::Overflow(subject))?
                .clamp(16, limit.alignment)
        };
        let pitch = row_bytes
            .checked_add(pitch_alignment - 1)
            .ok_or(HdbscanError::Overflow(subject))?
            / pitch_alignment
            * pitch_alignment;
        height
            .checked_mul(pitch)
            .ok_or(HdbscanError::Overflow(subject))?
    } else {
        elements
            .checked_mul(element_size)
            .ok_or(HdbscanError::Overflow(subject))?
    };
    checked_allocation_bytes(1, bytes, Some(limit), subject).map(|_| ())
}

fn checked_brute_force_scratch(
    query_rows: usize,
    reference_rows: usize,
    k: usize,
    limit: DeviceAllocationLimit,
) -> Result<usize> {
    common_neighbors::bounded_neighbor_scratch(
        query_rows,
        reference_rows,
        k,
        4,
        limit.max_page_bytes,
        limit.alignment,
    )
    .map(|plan| plan.live_bytes)
    .ok_or(HdbscanError::Overflow(
        "bounded brute-force neighbor scratch",
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CoreSearchMemory {
    device_bytes: usize,
    host_bytes: usize,
}

fn core_search_memory(
    n: usize,
    features: usize,
    min_samples: usize,
    search: HdbscanSearch,
    tree_bytes: usize,
    brute_force_scratch_bytes: usize,
) -> Result<CoreSearchMemory> {
    let k = min_samples.min(n.saturating_sub(1));
    let q = n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS);
    let input = n
        .checked_mul(features)
        .and_then(|v| v.checked_mul(4))
        .ok_or(HdbscanError::Overflow("core search input live bytes"))?;
    let core = n
        .checked_mul(4)
        .ok_or(HdbscanError::Overflow("core search core live bytes"))?;
    let compact = n
        .checked_mul(k)
        .and_then(|v| v.checked_mul(8))
        .ok_or(HdbscanError::Overflow("core search compact live bytes"))?;
    // Host IDs/distances coexist with the fitted public ID array during padded output construction.
    let public = n
        .checked_mul(min_samples)
        .and_then(|v| v.checked_mul(4))
        .ok_or(HdbscanError::Overflow(
            "core search public neighbor live bytes",
        ))?;
    let host_bytes = compact
        .checked_add(public)
        .and_then(|v| v.checked_add(core))
        .ok_or(HdbscanError::Overflow("core search host live bytes"))?;
    let device_bytes = match search {
        HdbscanSearch::KdTree => input
            .checked_add(core)
            .and_then(|v| v.checked_add(compact))
            .and_then(|v| v.checked_add(tree_bytes)),
        HdbscanSearch::BruteForce | HdbscanSearch::Auto => {
            // Input/core, bounded query view, batch ID/distance outputs, extracted core and
            // the final concatenated core coexist at the most conservative cat boundary.
            let batch = q
                .checked_mul(k)
                .and_then(|v| v.checked_mul(8))
                .and_then(|v| v.checked_add(q.checked_mul(features)?.checked_mul(4)?))
                .and_then(|v| v.checked_add(q.checked_mul(8)?));
            input
                .checked_add(core)
                .and_then(|v| v.checked_add(n.checked_mul(4)?))
                .and_then(|v| v.checked_add(batch?))
                .and_then(|v| v.checked_add(brute_force_scratch_bytes))
        }
    }
    .ok_or(HdbscanError::Overflow("core search device live bytes"))?;
    Ok(CoreSearchMemory {
        device_bytes,
        host_bytes,
    })
}

fn tensor_f32(values: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn tensor_i32(values: Vec<i32>, shape: [usize; 1], device: &Device) -> Tensor<1, Int> {
    Tensor::from_data(TensorData::new(values, shape), device)
}

fn upload_kd_tree(tree: &fast_hdbscan_core::KdTree, device: &Device) -> DeviceKdTree<Dispatch> {
    let nodes = tree.starts.len();
    let n_features = tree.n_features;
    DeviceKdTree {
        indices: tensor_i32(tree.indices.clone(), [tree.indices.len()], device).into_dispatch(),
        starts: tensor_i32(tree.starts.clone(), [nodes], device).into_dispatch(),
        ends: tensor_i32(tree.ends.clone(), [nodes], device).into_dispatch(),
        is_leaf: tensor_i32(tree.is_leaf.clone(), [nodes], device).into_dispatch(),
        bounds_min: tensor_f32(tree.bounds_min.clone(), [nodes, n_features], device)
            .into_dispatch(),
        bounds_max: tensor_f32(tree.bounds_max.clone(), [nodes, n_features], device)
            .into_dispatch(),
        node_components: tensor_i32(vec![-1; nodes], [nodes], device).into_dispatch(),
        n_levels: tree.n_levels,
        n_features,
    }
}

fn clone_device_kd_tree(tree: &DeviceKdTree<Dispatch>) -> DeviceKdTree<Dispatch> {
    DeviceKdTree {
        indices: tree.indices.clone(),
        starts: tree.starts.clone(),
        ends: tree.ends.clone(),
        is_leaf: tree.is_leaf.clone(),
        bounds_min: tree.bounds_min.clone(),
        bounds_max: tree.bounds_max.clone(),
        node_components: tree.node_components.clone(),
        n_levels: tree.n_levels,
        n_features: tree.n_features,
    }
}

struct CoreDistances {
    device_core_sq: Tensor<1>,
    seed_core_sq: Vec<f32>,
    neighbors: Vec<i32>,
    knn_sq: Vec<f32>,
    k: usize,
}

fn core_distances(
    data: &Tensor<2>,
    min_samples: usize,
    k: usize,
    search: HdbscanSearch,
    tree: Option<&DeviceKdTree<Dispatch>>,
) -> Result<CoreDistances> {
    let n = data.dims()[0];
    checked_allocation_bytes(
        n.checked_mul(k)
            .ok_or(HdbscanError::Overflow("compact neighbor elements"))?,
        8,
        None,
        "compact host seed allocation",
    )?;
    let k_width = u32::try_from(k).map_err(|_| HdbscanError::Overflow("neighbor width"))?;
    let (short_ids, short_distances, device_core_sq) = match search {
        HdbscanSearch::KdTree => {
            let tree = tree.expect("KD-tree route uploads the tree before neighbor search");
            #[cfg(feature = "tracing")]
            let neighbor_span = tracing::info_span!("core_neighbor_kernel").entered();
            let (ids, distances) = <Dispatch as HdbscanOps>::exact_neighbors_kd(
                data.clone().into_dispatch(),
                clone_device_kd_tree(tree),
                k_width,
            );
            #[cfg(feature = "tracing")]
            profile_sync(&data.device())?;
            #[cfg(feature = "tracing")]
            drop(neighbor_span);
            let ids = Tensor::<2, Int>::from_dispatch(ids);
            let distances = Tensor::<2>::from_dispatch(distances);
            #[cfg(feature = "tracing")]
            let extraction_span = tracing::info_span!("core_extraction").entered();
            let core = if min_samples >= n {
                Tensor::full([n], f32::INFINITY, &data.device())
            } else if min_samples == 1 {
                Tensor::zeros([n], &data.device())
            } else {
                distances
                    .clone()
                    .slice([0..n, (min_samples - 1)..min_samples])
                    .reshape([n])
            };
            #[cfg(feature = "tracing")]
            drop(extraction_span);
            #[cfg(feature = "tracing")]
            let readback_span = tracing::info_span!("core_neighbor_readback").entered();
            let [id_data, distance_data] = Transaction::default()
                .register(ids)
                .register(distances.clone())
                .try_execute()
                .map_err(|error| HdbscanError::Numerical(error.to_string()))?
                .try_into()
                .map_err(|_| {
                    HdbscanError::Numerical("neighbor readback returned wrong arity".into())
                })?;
            let ids = id_data
                .try_to_vec::<i32>()
                .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
            let distances_host = distance_data
                .try_to_vec::<f32>()
                .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
            #[cfg(feature = "tracing")]
            drop(readback_span);
            (ids, distances_host, core)
        }
        HdbscanSearch::BruteForce | HdbscanSearch::Auto => {
            let block_rows = n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS);
            let mut ids = Vec::with_capacity(
                n.checked_mul(k)
                    .ok_or(HdbscanError::Overflow("compact neighbor elements"))?,
            );
            let mut distances_host = Vec::with_capacity(ids.capacity());
            let mut core_blocks = Vec::new();
            for start in (0..n).step_by(block_rows) {
                let end = start.saturating_add(block_rows).min(n);
                let query = data.clone().slice([start..end, 0..data.dims()[1]]);
                #[cfg(feature = "tracing")]
                let neighbor_span = tracing::info_span!("core_neighbor_kernel").entered();
                let (block_ids, block_distances) = common_neighbors::exact_neighbors(
                    query,
                    data.clone(),
                    k_width,
                    true,
                    u32::try_from(start).map_err(|_| HdbscanError::Overflow("query row offset"))?,
                    true,
                );
                #[cfg(feature = "tracing")]
                profile_sync(&data.device())?;
                #[cfg(feature = "tracing")]
                drop(neighbor_span);
                let block_rows_count = end - start;
                #[cfg(feature = "tracing")]
                let extraction_span = tracing::info_span!("core_extraction").entered();
                let core = if min_samples >= n {
                    Tensor::full([block_rows_count], f32::INFINITY, &data.device())
                } else if min_samples == 1 {
                    Tensor::zeros([block_rows_count], &data.device())
                } else {
                    block_distances
                        .clone()
                        .slice([0..block_rows_count, (min_samples - 1)..min_samples])
                        .reshape([block_rows_count])
                };
                core_blocks.push(core);
                #[cfg(feature = "tracing")]
                drop(extraction_span);
                #[cfg(feature = "tracing")]
                let readback_span = tracing::info_span!("core_neighbor_readback").entered();
                let [id_data, distance_data] = Transaction::default()
                    .register(block_ids)
                    .register(block_distances)
                    .try_execute()
                    .map_err(|error| HdbscanError::Numerical(error.to_string()))?
                    .try_into()
                    .map_err(|_| {
                        HdbscanError::Numerical("neighbor readback returned wrong arity".into())
                    })?;
                ids.extend(
                    id_data
                        .try_to_vec::<i32>()
                        .map_err(|error| HdbscanError::Numerical(error.to_string()))?,
                );
                distances_host.extend(
                    distance_data
                        .try_to_vec::<f32>()
                        .map_err(|error| HdbscanError::Numerical(error.to_string()))?,
                );
                #[cfg(feature = "tracing")]
                drop(readback_span);
            }
            (ids, distances_host, Tensor::cat(core_blocks, 0))
        }
    };
    #[cfg(feature = "tracing")]
    let assembly_span = tracing::info_span!("core_final_assembly_readback").entered();
    checked_allocation_bytes(short_ids.len(), 4, None, "compact neighbor ID allocation")?;
    checked_allocation_bytes(
        short_distances.len(),
        4,
        None,
        "compact neighbor distance allocation",
    )?;
    let seed_core_sq = if min_samples >= n {
        vec![f32::INFINITY; n]
    } else if min_samples == 1 {
        vec![0.0; n]
    } else {
        (0..n)
            .map(|row| short_distances[row * k + min_samples - 1])
            .collect()
    };
    let result = CoreDistances {
        device_core_sq,
        seed_core_sq,
        neighbors: short_ids,
        knn_sq: short_distances,
        k,
    };
    #[cfg(feature = "tracing")]
    drop(assembly_span);
    Ok(result)
}

fn spanning_tree(
    data: &Tensor<2>,
    core: &mut CoreDistances,
    search: HdbscanSearch,
    tree: Option<&fast_hdbscan_core::KdTree>,
    device_tree: Option<&DeviceKdTree<Dispatch>>,
    reproducible: bool,
) -> Result<Vec<SortedMstEdge>> {
    let n = data.dims()[0];
    let mut dsu = RankDsu::new(n);
    #[cfg(feature = "tracing")]
    let _boruvka_span = tracing::info_span!("boruvka_total").entered();
    #[cfg(feature = "tracing")]
    let seed_span = tracing::info_span!("boruvka_seed_init").entered();
    let mut edges = init_from_knn(
        &core.seed_core_sq,
        &core.neighbors,
        &core.knn_sq,
        core.k,
        &mut dsu,
    )?;
    #[cfg(feature = "tracing")]
    drop(seed_span);
    drop(std::mem::take(&mut core.knn_sq));
    let mut component_count = (0..n)
        .map(|point| dsu.find(point))
        .collect::<std::collections::HashSet<_>>()
        .len();
    while component_count > 1 {
        #[cfg(feature = "tracing")]
        let _round_span = tracing::info_span!("boruvka_round", component_count).entered();
        #[cfg(feature = "tracing")]
        let metadata_span = tracing::info_span!("boruvka_metadata_prepare_upload").entered();
        let point_components: Vec<i32> = (0..n)
            .map(|point| i32::try_from(dsu.find(point)).expect("validated row count fits i32"))
            .collect();
        let plan = ComponentReductionPlan::build(
            &point_components,
            crate::kernels::COMPONENT_REDUCTION_TILE_WIDTH,
        )?;
        let c = plan.offsets.len() - 1;
        let p = plan.tile_starts.len();
        let limit = allocation_limit(&data.device())?;
        let mut workspace = 0usize;
        for (elements, subject) in [
            (n, "component members"),
            (p, "component tile starts"),
            (p, "component tile ends"),
            (c + 1, "component partial offsets"),
            (p, "component partial weights"),
            (p, "component partial sources"),
            (p, "component partial errors"),
            (c, "component winner weights"),
            (
                c.checked_mul(5)
                    .ok_or(HdbscanError::Overflow("winner metadata"))?,
                "component winner metadata",
            ),
        ] {
            let bytes = checked_allocation_bytes(elements, 4, Some(limit), subject)?;
            workspace = workspace
                .checked_add(bytes)
                .ok_or(HdbscanError::Overflow("component reduction workspace"))?;
        }
        checked_allocation_bytes(workspace, 1, None, "component reduction workspace")?;
        let core_tensor = core.device_core_sq.clone();
        let component_tensor = Tensor::<1, Int>::from_data(
            TensorData::new(point_components.clone(), [n]),
            &data.device(),
        );
        let members = tensor_i32(plan.members.clone(), [n], &data.device()).into_dispatch();
        let tile_starts = tensor_i32(plan.tile_starts.clone(), [p], &data.device()).into_dispatch();
        let tile_ends = tensor_i32(plan.tile_ends.clone(), [p], &data.device()).into_dispatch();
        let partial_offsets =
            tensor_i32(plan.partial_offsets.clone(), [c + 1], &data.device()).into_dispatch();
        let round_tree = if search == HdbscanSearch::KdTree {
            let host_tree = tree.expect("KD-tree route retains host tree for fresh node labels");
            let mut round_tree = clone_device_kd_tree(
                device_tree.expect("KD-tree route uploads the tree before Boruvka"),
            );
            round_tree.node_components = tensor_i32(
                host_tree.node_components(&point_components)?,
                [host_tree.starts.len()],
                &data.device(),
            )
            .into_dispatch();
            Some(round_tree)
        } else {
            None
        };
        #[cfg(feature = "tracing")]
        profile_sync(&data.device())?;
        #[cfg(feature = "tracing")]
        drop(metadata_span);
        #[cfg(feature = "tracing")]
        let candidate_span = tracing::info_span!("boruvka_candidate_kernel").entered();
        let (candidate_distances, candidate_indices) = match search {
            HdbscanSearch::KdTree => {
                let (weights, destinations) = <Dispatch as HdbscanOps>::boruvka_candidates_kd(
                    data.clone().into_dispatch(),
                    core_tensor.into_dispatch(),
                    component_tensor.into_dispatch(),
                    round_tree.expect("KD-tree metadata prepared for this round"),
                    reproducible,
                );
                (weights, destinations)
            }
            HdbscanSearch::BruteForce | HdbscanSearch::Auto => {
                <Dispatch as HdbscanOps>::boruvka_candidates(
                    data.clone().into_dispatch(),
                    core_tensor.into_dispatch(),
                    component_tensor.into_dispatch(),
                )
            }
        };
        #[cfg(feature = "tracing")]
        profile_sync(&data.device())?;
        #[cfg(feature = "tracing")]
        drop(candidate_span);
        #[cfg(feature = "tracing")]
        let reduction_span = tracing::info_span!("boruvka_component_reduction").entered();
        let (winner_weights, winner_metadata) = <Dispatch as HdbscanOps>::component_reduce(
            candidate_distances,
            candidate_indices,
            members,
            tile_starts,
            tile_ends,
            partial_offsets,
            u32::try_from(c).map_err(|_| HdbscanError::Overflow("component count"))?,
        );
        #[cfg(feature = "tracing")]
        profile_sync(&data.device())?;
        #[cfg(feature = "tracing")]
        drop(reduction_span);
        #[cfg(feature = "tracing")]
        let readback_span = tracing::info_span!("boruvka_winner_readback").entered();
        let winner_weights = Tensor::<1>::from_dispatch(winner_weights);
        let winner_metadata = Tensor::<2, Int>::from_dispatch(winner_metadata);
        if winner_weights.dims() != [c] || winner_metadata.dims() != [c, 5] {
            return Err(HdbscanError::Numerical(
                "component reduction returned wrong shapes".into(),
            ));
        }
        let [distance_data, index_data] = Transaction::default()
            .register(winner_weights)
            .register(winner_metadata)
            .try_execute()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?
            .try_into()
            .map_err(|_| HdbscanError::Numerical("winner readback returned wrong arity".into()))?;
        let winner_weights = distance_data
            .try_to_vec::<f32>()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
        let winner_metadata = index_data
            .try_to_vec::<i32>()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?;
        #[cfg(feature = "tracing")]
        drop(readback_span);
        #[cfg(feature = "tracing")]
        let merge_span = tracing::info_span!("boruvka_host_merge").entered();
        merge_component_winners(
            &mut dsu,
            &plan,
            &winner_weights,
            &winner_metadata,
            &mut edges,
        )?;
        component_count = (0..n)
            .map(|point| dsu.find(point))
            .collect::<std::collections::HashSet<_>>()
            .len();
        #[cfg(feature = "tracing")]
        drop(merge_span);
    }
    if edges.len() != n - 1 {
        return Err(HdbscanError::Numerical(
            "Boruvka produced an incomplete spanning tree".into(),
        ));
    }
    #[cfg(feature = "tracing")]
    let finalize_span = tracing::info_span!("boruvka_finalize").entered();
    let result = finalize_mst(edges);
    #[cfg(feature = "tracing")]
    drop(finalize_span);
    Ok(result)
}

fn public_neighbors(ids: &[i32], n: usize, width: usize, min_samples: usize) -> Result<Vec<i32>> {
    let slots = n
        .checked_mul(min_samples)
        .ok_or(HdbscanError::Overflow("public neighbor output elements"))?;
    checked_allocation_bytes(slots, 4, None, "public neighbor ID allocation")?;
    let mut public = vec![-1; slots];
    let copied = width.min(min_samples);
    for row in 0..n {
        public[row * min_samples..row * min_samples + copied]
            .copy_from_slice(&ids[row * width..row * width + copied]);
    }
    Ok(public)
}

fn knn_spanning_tree(
    data: &Tensor<2>,
    min_samples: usize,
    k: usize,
    approximate: bool,
) -> Result<(fast_hdbscan_core::knn_kruskal::KnnMst, KnnDiagnostics)> {
    let n = data.dims()[0];
    let slots = n
        .checked_mul(k)
        .ok_or(HdbscanError::Overflow("kNN graph elements"))?;
    // Check each buffer and their combined accounting before any graph allocation.
    // Sort workspace conservatively reserves another full candidate tuple array.
    let mut allocation_bytes = 0usize;
    for (elements, size, subject) in [
        (slots, 4, "kNN graph IDs"),
        (slots, 4, "kNN graph distances"),
        (
            n.checked_mul(min_samples)
                .ok_or(HdbscanError::Overflow("public neighbors"))?,
            4,
            "public neighbor IDs",
        ),
        (
            slots,
            std::mem::size_of::<(f64, usize, usize, usize)>(),
            "kNN candidate tuples",
        ),
        (
            slots,
            std::mem::size_of::<(f64, usize, usize, usize)>(),
            "kNN candidate sort workspace",
        ),
    ] {
        let bytes = checked_allocation_bytes(elements, size, None, subject)?;
        allocation_bytes = allocation_bytes
            .checked_add(bytes)
            .ok_or(HdbscanError::Overflow("kNN graph live allocation bytes"))?;
    }
    checked_allocation_bytes(allocation_bytes, 1, None, "kNN graph live allocation bytes")?;
    let (mut ids, distances_sq) = if approximate {
        ann_preflight(&data.device(), n, data.dims()[1], k).map_err(map_ann_error)?;
        let width = u32::try_from(k).map_err(|_| HdbscanError::Overflow("ANN neighbor width"))?;
        let iterations = (n as f64).log2().round().max(5.0) as u32;
        let (ids, distances) = <Dispatch as NeighborOps>::approximate_neighbors(
            data.clone().into_dispatch(),
            width,
            42,
            iterations,
            true,
        );
        let [id_data, distance_data] = Transaction::default()
            .register(Tensor::<2, Int>::from_dispatch(ids))
            .register(Tensor::<2>::from_dispatch(distances))
            .try_execute()
            .map_err(|error| HdbscanError::Numerical(error.to_string()))?
            .try_into()
            .map_err(|_| HdbscanError::Numerical("ANN readback returned wrong arity".into()))?;
        (
            id_data
                .try_to_vec::<i32>()
                .map_err(|error| HdbscanError::Numerical(error.to_string()))?,
            distance_data
                .try_to_vec::<f32>()
                .map_err(|error| HdbscanError::Numerical(error.to_string()))?,
        )
    } else {
        let core = core_distances(data, min_samples, k, HdbscanSearch::BruteForce, None)?;
        (core.neighbors, core.knn_sq)
    };
    for (id, distance) in ids.iter_mut().zip(&distances_sq) {
        // Common exact search represents a missing nonfinite neighbor by (n, +inf).
        // Convert only that documented sentinel, never repair invalid graph records.
        if *id == n as i32 && *distance == f32::INFINITY {
            *id = -1;
        }
    }
    let result = knn_kruskal(
        KnnGraph {
            ids: &ids,
            dist_sq: &distances_sq,
            rows: n,
            k,
        },
        min_samples,
    )?;
    let diagnostics = KnnDiagnostics {
        source: if approximate {
            KnnNeighborSource::Approximate
        } else {
            KnnNeighborSource::Exact
        },
        neighbors: ids,
        distances_sq,
        k,
        bridges: result.bridges,
        finite_weight_sum: result
            .edges
            .iter()
            .filter(|edge| edge.weight.is_finite())
            .map(|edge| edge.weight)
            .sum(),
        allocation_bytes,
    };
    Ok((result, diagnostics))
}

/// Force the kNN graph with a controlled neighbor source for exact comparisons.
#[doc(hidden)]
pub fn fit_knn_graph_with_source(
    data: &[Vec<f32>],
    mut config: HdbscanConfig,
    backend: BackendChoice,
    source: KnnSource,
) -> Result<FittedHDBSCAN> {
    config.mst_graph = MstGraph::KnnGraph;
    HDBSCAN::new(config, backend).fit_with_source(data, source)
}

/// HDBSCAN estimator using complete or kNN mutual-reachability graphs.
#[derive(Debug, Clone)]
pub struct HDBSCAN {
    config: HdbscanConfig,
    backend: BackendChoice,
    search: HdbscanSearch,
}

impl HDBSCAN {
    pub fn new(config: HdbscanConfig, backend: BackendChoice) -> Self {
        Self {
            config,
            backend,
            search: HdbscanSearch::Auto,
        }
    }

    pub fn with_search(mut self, search: HdbscanSearch) -> Self {
        self.search = search;
        self
    }

    pub fn fit(&self, input: &[Vec<f32>]) -> Result<FittedHDBSCAN> {
        self.fit_with_source(input, KnnSource::Auto)
    }

    fn fit_with_source(&self, input: &[Vec<f32>], source: KnnSource) -> Result<FittedHDBSCAN> {
        #[cfg(feature = "tracing")]
        let _input_validation_span = tracing::info_span!("input_validation").entered();
        self.config.validate()?;
        if input.is_empty() {
            return Err(HdbscanError::InvalidInput(
                "input must contain at least one row".into(),
            ));
        }
        let features = input[0].len();
        if input.iter().any(|row| row.len() != features) {
            return Err(HdbscanError::InvalidInput(
                "input rows must all have the same number of features".into(),
            ));
        }
        fast_hdbscan_core::finite::validate_dims(input.len(), features)?;
        let input_elements = input
            .len()
            .checked_mul(features)
            .ok_or(HdbscanError::Overflow("input element count"))?;
        checked_allocation_bytes(input_elements, 4, None, "input allocation")?;
        let (raw, raw_n, features) = flatten_input(input)?;
        let split = split_finite(&raw, features)?;
        let n = split.finite_index.len();
        #[cfg(feature = "tracing")]
        drop(_input_validation_span);
        let min_samples = self.config.resolved_min_samples();
        #[cfg(feature = "tracing")]
        let _backend_setup_span = tracing::info_span!("backend_setup").entered();
        let (device, actual_backend) = selected_device(self.backend).map_err(map_backend_error)?;
        let allocation_limit = allocation_limit(&device)?;
        let plan = crate::resources::plan_fit(n, features, &self.config, actual_backend, source)?;
        #[cfg(feature = "tracing")]
        crate::resources::emit_advisory(&plan.estimate);
        let resolved_graph = plan.graph;
        let neighbor_width = plan.width;
        let resolved_search = if resolved_graph == MstGraph::Complete {
            self.search
                .resolve(n, features, min_samples, actual_backend)
        } else {
            HdbscanSearch::BruteForce
        };
        let cuda_row_pitch = matches!(actual_backend, BackendChoice::Cuda { .. });
        let brute_force_scratch_bytes =
            if n > 1 && resolved_search == HdbscanSearch::BruteForce && cuda_row_pitch {
                checked_brute_force_scratch(
                    n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS),
                    n,
                    neighbor_width,
                    allocation_limit,
                )?
            } else {
                0
            };
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        drop(_backend_setup_span);
        #[cfg(feature = "tracing")]
        let _input_preparation_span = tracing::info_span!("input_preparation", rows = n).entered();
        let mut neighbors = Vec::new();
        let mut core_values = vec![f32::INFINITY; n];
        let mut mst = Vec::new();
        let mut linkage = Vec::new();
        let mut condensed = CondensedTree::default();
        let mut labels = vec![-1; raw_n];
        let mut probabilities = vec![0.0; raw_n];
        let mut knn_diagnostics = None;
        #[cfg(feature = "tracing")]
        let result_finalization_span;

        if n > 1 {
            let finite_values = if split.outliers.is_empty() {
                raw
            } else {
                compact_rows(&raw, features, &split.finite_index)?
            };
            check_device_tensor_allocation(
                &[n, features],
                4,
                allocation_limit,
                cuda_row_pitch,
                "finite input allocation",
            )?;
            let sorted_mst = if resolved_graph == MstGraph::KnnGraph {
                let device_rows = n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS);
                for subject in ["device graph IDs", "device graph distances"] {
                    check_device_tensor_allocation(
                        &[device_rows, neighbor_width],
                        4,
                        allocation_limit,
                        cuda_row_pitch,
                        subject,
                    )?;
                }
                let data = tensor_f32(finite_values, [n, features], &device);
                let (graph_mst, diagnostics) =
                    knn_spanning_tree(&data, min_samples, neighbor_width, plan.approximate)?;
                neighbors =
                    public_neighbors(&diagnostics.neighbors, n, diagnostics.k, min_samples)?;
                core_values = graph_mst.core.iter().map(|&value| value as f32).collect();
                knn_diagnostics = Some(diagnostics);
                #[cfg(feature = "tracing")]
                drop(_input_preparation_span);
                graph_mst.edges
            } else {
                check_device_allocation(n, 4, allocation_limit, "core-distance allocation")?;
                check_device_allocation(n, 4, allocation_limit, "component-label allocation")?;
                let neighbor_slots = n
                    .checked_mul(min_samples)
                    .ok_or(HdbscanError::Overflow("neighbor output elements"))?;
                checked_allocation_bytes(neighbor_slots, 4, None, "public neighbor ID allocation")?;
                let sizing = crate::kernels::validate_boruvka_candidate_sizing(n)?;
                let neighbor_device_rows = if resolved_search == HdbscanSearch::KdTree {
                    n
                } else {
                    n.saturating_sub(1).clamp(1, NEIGHBOR_BATCH_ROWS)
                };
                let neighbor_device_elements = neighbor_device_rows
                    .checked_mul(neighbor_width)
                    .ok_or(HdbscanError::Overflow("device neighbor output elements"))?;
                check_device_allocation(
                    neighbor_device_elements,
                    4,
                    allocation_limit,
                    "device neighbor index allocation",
                )?;
                check_device_allocation(
                    neighbor_device_elements,
                    4,
                    allocation_limit,
                    "device neighbor distance allocation",
                )?;
                let scratch_elements = sizing
                    .block_rows
                    .checked_mul(sizing.tiles)
                    .ok_or(HdbscanError::Overflow("Boruvka candidate scratch elements"))?;
                if resolved_search == HdbscanSearch::BruteForce {
                    check_device_allocation(
                        scratch_elements,
                        4,
                        allocation_limit,
                        "Boruvka candidate weight scratch allocation",
                    )?;
                    check_device_allocation(
                        scratch_elements,
                        4,
                        allocation_limit,
                        "Boruvka candidate index scratch allocation",
                    )?;
                }
                check_device_allocation(
                    n,
                    4,
                    allocation_limit,
                    "Boruvka candidate weight output allocation",
                )?;
                check_device_allocation(
                    n,
                    4,
                    allocation_limit,
                    "Boruvka candidate index output allocation",
                )?;
                #[cfg(feature = "tracing")]
                drop(_input_preparation_span);

                #[cfg(feature = "tracing")]
                let _tree_build_span = tracing::info_span!("tree_build", rows = n).entered();
                let host_tree = if resolved_search == HdbscanSearch::KdTree {
                    Some(fast_hdbscan_core::KdTree::build(
                        &finite_values,
                        n,
                        features,
                    )?)
                } else {
                    None
                };

                let tree_bytes = host_tree.as_ref().map_or(Ok(0usize), |tree| {
                    [
                        tree.indices.len(),
                        tree.starts.len(),
                        tree.ends.len(),
                        tree.is_leaf.len(),
                    ]
                    .into_iter()
                    .try_fold(0usize, |sum, elements| {
                        elements
                            .checked_mul(4)
                            .and_then(|bytes| sum.checked_add(bytes))
                            .ok_or(HdbscanError::Overflow("KD-tree live bytes"))
                    })
                    .and_then(|bytes| {
                        let bounds = tree
                            .starts
                            .len()
                            .checked_mul(tree.n_features)
                            .and_then(|elements| elements.checked_mul(8))
                            .ok_or(HdbscanError::Overflow("KD-tree bounds live bytes"))?;
                        bytes
                            .checked_add(bounds)
                            .and_then(|v| v.checked_add(tree.starts.len().checked_mul(4)?))
                            .ok_or(HdbscanError::Overflow("KD-tree live bytes"))
                    })
                })?;
                let memory = core_search_memory(
                    n,
                    features,
                    min_samples,
                    resolved_search,
                    tree_bytes,
                    brute_force_scratch_bytes,
                )?;
                if memory.device_bytes > isize::MAX as usize
                    || memory.host_bytes > isize::MAX as usize
                {
                    return Err(HdbscanError::Overflow("core search live-memory plan"));
                }
                if let Some(tree) = &host_tree {
                    for (elements, element_size, subject) in [
                        (tree.indices.len(), 4, "KD-tree index allocation"),
                        (tree.starts.len(), 4, "KD-tree start allocation"),
                        (tree.ends.len(), 4, "KD-tree end allocation"),
                        (tree.is_leaf.len(), 4, "KD-tree leaf allocation"),
                    ] {
                        check_device_allocation(elements, element_size, allocation_limit, subject)?;
                    }
                    check_device_tensor_allocation(
                        &[tree.starts.len(), tree.n_features],
                        4,
                        allocation_limit,
                        cuda_row_pitch,
                        "KD-tree minimum-bounds allocation",
                    )?;
                    check_device_tensor_allocation(
                        &[tree.starts.len(), tree.n_features],
                        4,
                        allocation_limit,
                        cuda_row_pitch,
                        "KD-tree maximum-bounds allocation",
                    )?;
                    // Per-round node-component labels are uploaded separately from the static tree.
                    check_device_allocation(
                        tree.starts.len(),
                        4,
                        allocation_limit,
                        "KD-tree node-component allocation",
                    )?;
                }
                #[cfg(feature = "tracing")]
                drop(_tree_build_span);

                #[cfg(feature = "tracing")]
                let _input_upload_span = tracing::info_span!("input_upload", rows = n).entered();
                let data = tensor_f32(finite_values, [n, features], &device);
                let device_tree = host_tree.as_ref().map(|tree| upload_kd_tree(tree, &device));
                #[cfg(feature = "tracing")]
                profile_sync(&device)?;
                #[cfg(feature = "tracing")]
                drop(_input_upload_span);

                #[cfg(feature = "tracing")]
                let _core_distances_span =
                    tracing::info_span!("core_distances", rows = n).entered();
                let mut core = core_distances(
                    &data,
                    min_samples,
                    neighbor_width,
                    resolved_search,
                    device_tree.as_ref(),
                )?;
                core_values = core.seed_core_sq.iter().map(|value| value.sqrt()).collect();
                #[cfg(feature = "tracing")]
                profile_sync(&device)?;
                #[cfg(feature = "tracing")]
                drop(_core_distances_span);

                #[cfg(feature = "tracing")]
                let _mst_span =
                    tracing::info_span!("mst", rows = n, search = ?resolved_search).entered();
                let sorted_mst = spanning_tree(
                    &data,
                    &mut core,
                    resolved_search,
                    host_tree.as_ref(),
                    device_tree.as_ref(),
                    self.config.reproducible,
                )?;
                if core.k == min_samples {
                    neighbors = core.neighbors;
                } else {
                    let slots = n
                        .checked_mul(min_samples)
                        .ok_or(HdbscanError::Overflow("public neighbor output elements"))?;
                    checked_allocation_bytes(slots, 4, None, "public neighbor output allocation")?;
                    let mut public = vec![-1; slots];
                    for row in 0..n {
                        public[row * min_samples..row * min_samples + core.k]
                            .copy_from_slice(&core.neighbors[row * core.k..(row + 1) * core.k]);
                    }
                    neighbors = public;
                }
                #[cfg(feature = "tracing")]
                profile_sync(&device)?;
                #[cfg(feature = "tracing")]
                drop(_mst_span);
                sorted_mst
            };
            #[cfg(feature = "tracing")]
            let _cluster_span = tracing::info_span!("cluster_extraction", rows = n).entered();
            let host = fast_hdbscan_core::clusters_from_spanning_tree(&sorted_mst, &self.config)?;
            #[cfg(feature = "tracing")]
            drop(_cluster_span);
            #[cfg(feature = "tracing")]
            {
                result_finalization_span =
                    tracing::info_span!("result_finalization", rows = n).entered();
            }
            linkage = if split.outliers.is_empty() {
                host.linkage.clone()
            } else {
                remap_linkage(&host.linkage, &split.finite_index, &split.outliers)?
            };
            labels = scatter_labels(&host.labels, &split.finite_index, raw_n)?;
            probabilities = scatter_probabilities(&host.probabilities, &split.finite_index, raw_n)?;
            condensed = if split.outliers.is_empty() {
                host.condensed.clone()
            } else {
                remap_condensed(&host.condensed, &split.finite_index, &split.outliers)?
            };
            mst = host.sorted_mst;
        } else {
            #[cfg(feature = "tracing")]
            drop(_input_preparation_span);
            #[cfg(feature = "tracing")]
            {
                result_finalization_span =
                    tracing::info_span!("result_finalization", rows = n).entered();
            }
            let host = HostClustering::one_finite();
            labels = scatter_labels(&host.labels, &split.finite_index, raw_n)?;
            probabilities = scatter_probabilities(&host.probabilities, &split.finite_index, raw_n)?;
        }

        let fitted = FittedHDBSCAN {
            config: self.config.clone(),
            backend: actual_backend,
            labels,
            probabilities,
            linkage,
            condensed,
            mst,
            neighbors,
            core_distances: core_values,
            finite_index: split.finite_index,
            n_features: features,
            raw_n,
            search: resolved_search,
            resolved_graph,
            knn_diagnostics,
        };
        #[cfg(feature = "tracing")]
        drop(result_finalization_span);
        Ok(fitted)
    }
}

/// Fitted estimator results. Neighbor and core-distance rows use finite-row indexing;
/// labels, probabilities, linkage and condensed-tree node ids are in raw-row space.
#[derive(Debug, Clone)]
pub struct FittedHDBSCAN {
    pub config: HdbscanConfig,
    pub backend: BackendChoice,
    pub labels: Vec<i32>,
    pub probabilities: Vec<f32>,
    pub linkage: Vec<LinkageRow>,
    pub condensed: CondensedTree,
    pub mst: Vec<SortedMstEdge>,
    /// Row-major finite-row neighbor ids padded to `resolved_min_samples()` with -1.
    pub neighbors: Vec<i32>,
    /// Euclidean core distances in finite-row order.
    pub core_distances: Vec<f32>,
    pub finite_index: Vec<u32>,
    n_features: usize,
    raw_n: usize,
    search: HdbscanSearch,
    resolved_graph: MstGraph,
    knn_diagnostics: Option<KnnDiagnostics>,
}

impl FittedHDBSCAN {
    /// The requested graph policy, including `Auto` when that was requested.
    pub fn mst_graph(&self) -> MstGraph {
        self.config.mst_graph
    }

    pub fn resolved_mst_graph(&self) -> MstGraph {
        self.resolved_graph
    }

    #[doc(hidden)]
    pub fn knn_diagnostics(&self) -> Option<&KnnDiagnostics> {
        self.knn_diagnostics.as_ref()
    }

    pub fn dbscan_clustering(&self, epsilon: f64) -> Result<Vec<i32>> {
        let labels = dbscan_cut(&self.linkage, epsilon, self.config.resolved_min_samples())?;
        if self.finite_index.len() == self.raw_n {
            return Ok(labels);
        }
        if self.finite_index.len() <= 1 {
            return Ok(vec![-1; self.raw_n]);
        }
        // `remap_linkage` has already expanded the tree with an infinite-distance outlier chain.
        Ok(labels)
    }

    pub fn n_features(&self) -> usize {
        self.n_features
    }

    pub fn config(&self) -> &HdbscanConfig {
        &self.config
    }
    pub fn backend(&self) -> BackendChoice {
        self.backend
    }
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn linkage(&self) -> &[LinkageRow] {
        &self.linkage
    }
    pub fn condensed_tree(&self) -> &CondensedTree {
        &self.condensed
    }
    pub fn minimum_spanning_tree(&self) -> &[SortedMstEdge] {
        &self.mst
    }
    pub fn neighbors(&self) -> &[i32] {
        &self.neighbors
    }
    pub fn core_distances(&self) -> &[f32] {
        &self.core_distances
    }
    pub fn finite_index(&self) -> &[u32] {
        &self.finite_index
    }
    pub fn search(&self) -> HdbscanSearch {
        self.search
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;

    #[test]
    fn auto_search_uses_the_bounded_dispatch_region() {
        for backend in [BackendChoice::Cpu, BackendChoice::Cuda { device_index: 0 }] {
            assert_eq!(
                HdbscanSearch::Auto.resolve(255, 2, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(256, 2, 5, backend),
                HdbscanSearch::KdTree
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(2048, 30, 32, backend),
                HdbscanSearch::KdTree
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(2048, 31, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(2049, 10, 32, backend),
                HdbscanSearch::KdTree
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(2049, 11, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(2049, 30, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(4096, 10, 32, backend),
                HdbscanSearch::KdTree
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(4097, 2, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(4096, 11, 5, backend),
                HdbscanSearch::BruteForce
            );
            assert_eq!(
                HdbscanSearch::Auto.resolve(4096, 2, 33, backend),
                HdbscanSearch::BruteForce
            );
        }
    }

    #[test]
    fn forced_search_routes_are_independent_of_auto_cutoff() {
        for backend in [BackendChoice::Cpu, BackendChoice::Cuda { device_index: 0 }] {
            for (rows, features, min_samples) in [(1, 50, 100), (255, 31, 33), (4096, 2, 5)] {
                assert_eq!(
                    HdbscanSearch::BruteForce.resolve(rows, features, min_samples, backend),
                    HdbscanSearch::BruteForce,
                );
                assert_eq!(
                    HdbscanSearch::KdTree.resolve(rows, features, min_samples, backend),
                    HdbscanSearch::KdTree,
                );
            }
        }
    }

    #[test]
    fn device_allocation_checks_include_alignment_and_runtime_limit() {
        let limit = DeviceAllocationLimit {
            max_page_bytes: 16,
            alignment: 8,
        };
        assert_eq!(
            checked_allocation_bytes(3, 5, Some(limit), "test").unwrap(),
            16
        );
        assert!(checked_allocation_bytes(4, 5, Some(limit), "test").is_err());
        assert!(checked_allocation_bytes(usize::MAX, 2, Some(limit), "test").is_err());
        assert_eq!(
            checked_allocation_bytes(usize::MAX, 2, None, "test").unwrap_err(),
            HdbscanError::Overflow("test")
        );

        let pitched_limit = DeviceAllocationLimit {
            max_page_bytes: 64,
            alignment: 32,
        };
        assert!(check_device_tensor_allocation(&[2, 5], 4, pitched_limit, true, "test").is_ok());
        let too_small_pitched_limit = DeviceAllocationLimit {
            max_page_bytes: 16,
            alignment: 32,
        };
        assert!(
            check_device_tensor_allocation(&[2, 5], 4, too_small_pitched_limit, true, "test")
                .is_err()
        );
    }

    #[test]
    fn core_search_memory_accounts_for_bounded_and_full_routes() {
        let brute = core_search_memory(1024, 8, 5, HdbscanSearch::BruteForce, 0, 4096).unwrap();
        let cpu_brute = core_search_memory(1024, 8, 5, HdbscanSearch::BruteForce, 0, 0).unwrap();
        let kd = core_search_memory(1024, 8, 5, HdbscanSearch::KdTree, 4096, 0).unwrap();
        assert_eq!(brute.device_bytes, 65_536);
        assert_eq!(cpu_brute.device_bytes, 61_440);
        assert_eq!(kd.device_bytes, 81_920);
        assert!(brute.device_bytes < kd.device_bytes);
        assert_eq!(brute.host_bytes, kd.host_bytes);
        assert!(core_search_memory(usize::MAX, 8, 5, HdbscanSearch::BruteForce, 0, 0).is_err());
    }

    #[test]
    fn impossible_neighbor_scratch_is_typed_overflow_before_allocation() {
        let limit = DeviceAllocationLimit {
            max_page_bytes: 128,
            alignment: 64,
        };
        assert_eq!(
            checked_brute_force_scratch(256, 1024, 100, limit).unwrap_err(),
            HdbscanError::Overflow("bounded brute-force neighbor scratch"),
        );
    }

    #[test]
    fn selected_cpu_reports_allocation_properties_without_tensor_upload() {
        let (device, _) = selected_device(BackendChoice::Cpu).unwrap();
        let limit = allocation_limit(&device).unwrap();
        assert!(limit.max_page_bytes > 0);
        assert!(limit.alignment > 0);
    }
}
