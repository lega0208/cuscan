use burn::{
    backend::{Dispatch, tensor::IntTensor},
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::backend::StandardUmapOps;
use cuscan_common::{AnnError, ann_preflight};
use umap_core::modern_layout::{BufferLifetime, BufferSpace, LayoutOperation, LiveBuffer};

use super::{
    Embedding, Init, InitRescale, NeighborSearch, QueryBudget, Result, UMAPConfig, UmapError,
    approximate_query_budget, fit_curve, fit_neighbor_widths,
    query::{approximate_query_scratch_layout, query_output_allocation},
    query_neighbor_width, validate_input, validate_search_settings,
};

/// Auto uses exact search for small data and bounded NN-Descent above this row count.
///
/// Measured with `examples/standard_auto_cutoff.rs` (32 features, release, RTX 3080 /
/// Ryzen 7 9800X3D): CUDA exact beats approximate up to 4k rows (0.46 s vs 0.53 s) and
/// loses from 8k (1.78 s vs 0.93 s). The shared cutoff follows that CUDA crossover;
/// it has not been recalibrated for the CubeCL CPU runtime.
pub const AUTO_EXACT_MAX_ROWS: usize = 4096;
// A reproducible default when the caller leaves random_state unset.
const DEFAULT_RANDOM_SEED: u64 = 42;
const APPROXIMATE_QUERY_BLOCK: usize = 64;

#[cfg(feature = "tracing")]
fn profile_sync(device: &Device) -> Result<()> {
    device
        .sync()
        .map_err(|error| UmapError::Numerical(format!("profiling synchronization failed: {error}")))
}

fn distinct_training_rows(data: &[Vec<f32>]) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    let mut indices = HashMap::<Vec<u32>, usize>::new();
    indices
        .try_reserve(data.len())
        .map_err(|error| UmapError::Memory(format!("unique index: {error}")))?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(data.len())
        .map_err(|error| UmapError::Memory(format!("unique rows: {error}")))?;
    let mut inverse = Vec::new();
    inverse
        .try_reserve_exact(data.len())
        .map_err(|error| UmapError::Memory(format!("unique inverse: {error}")))?;
    for row in data {
        let mut key = Vec::new();
        key.try_reserve_exact(row.len())
            .map_err(|error| UmapError::Memory(format!("unique key: {error}")))?;
        key.extend(
            row.iter()
                .map(|&value| if value == 0.0 { 0 } else { value.to_bits() }),
        );
        let index = if let Some(&index) = indices.get(&key) {
            index
        } else {
            let mut copy = Vec::new();
            copy.try_reserve_exact(row.len())
                .map_err(|error| UmapError::Memory(format!("unique row: {error}")))?;
            copy.extend_from_slice(row);
            let index = rows.len();
            rows.push(copy);
            indices.insert(key, index);
            index
        };
        inverse.push(index);
    }
    if rows.len() < 2 {
        return Err(UmapError::InvalidInput(
            "unique fitting requires at least two distinct rows".into(),
        ));
    }
    Ok((rows, inverse))
}

/// Standard, nonparametric UMAP estimator on the selected backend.
#[derive(Debug, Clone)]
pub struct UMAP {
    config: UMAPConfig,
    backend: crate::standard_umap::BackendChoice,
    search: NeighborSearch,
}

impl UMAP {
    pub fn new(
        config: UMAPConfig,
        backend: crate::standard_umap::BackendChoice,
        search: NeighborSearch,
    ) -> Self {
        Self {
            config,
            backend,
            search,
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip_all,
            fields(
                rows = data.len(),
                features = data.first().map_or(0, Vec::len),
                neighbors = self.config.n_neighbors,
                components = self.config.n_components,
                backend = ?self.backend,
                search = ?self.search,
                init = ?self.config.init,
                epochs = self.config.n_epochs.unwrap_or(0),
            )
        )
    )]
    pub fn fit(&self, data: &[Vec<f32>]) -> Result<FittedUMAP> {
        self.config.validate()?;
        let (original_n, features) = validate_input(data, None)?;
        original_n
            .checked_mul(self.config.n_components)
            .ok_or(UmapError::Overflow("original embedding shape"))?;
        if self.config.unique {
            let bytes = super::modern_runtime::allocation_bytes;
            let row = bytes(features, 1, size_of::<f32>())?;
            // HashMap reserves at least 7/8 of a power-of-two bucket table. Two
            // buckets per entry, rounded up, conservatively includes controls.
            let buckets = original_n
                .checked_mul(2)
                .and_then(usize::checked_next_power_of_two)
                .ok_or(UmapError::Overflow("unique hash table"))?
                .max(8);
            let table = bytes(buckets, 1, size_of::<(Vec<u32>, usize)>() + 1)?
                .checked_add(16)
                .ok_or(UmapError::Overflow("unique hash table"))?;
            let allocations = [
                ("original input rows", row, original_n),
                (
                    "original input row metadata",
                    bytes(original_n, 1, size_of::<Vec<f32>>())?,
                    1,
                ),
                ("unique keys", row, original_n),
                ("unique transient key", row, 1),
                ("distinct training rows", row, original_n),
                (
                    "distinct training row metadata",
                    bytes(original_n, 1, size_of::<Vec<f32>>())?,
                    1,
                ),
                (
                    "unique inverse mapping",
                    bytes(original_n, 1, size_of::<usize>())?,
                    1,
                ),
                ("unique hash table", table, 1),
            ];
            super::modern_runtime::validate_host_allocations(&allocations)?;
        }
        // Validate every original value before canonicalizing or dropping rows.
        let (distinct, original_to_training) = if self.config.unique {
            let (rows, inverse) = distinct_training_rows(data)?;
            (Some(rows), Some(inverse))
        } else {
            (None, None)
        };
        let data = distinct.as_deref().unwrap_or(data);
        let n = data.len();
        n.checked_mul(features)
            .ok_or(UmapError::Overflow("input shape"))?;
        if n > i32::MAX as usize {
            return Err(UmapError::Overflow("neighbor row indices"));
        }
        let search = self.resolve_search(n);
        validate_search_settings(self.search, self.config.transform_queue_size, Some(search))?;
        let seed = self.config.random_state.unwrap_or(DEFAULT_RANDOM_SEED);
        let widths = fit_neighbor_widths(n, self.config.n_neighbors)?;
        if n > 1 && widths.nonself > 0 {
            let capacity = n
                .checked_mul(widths.self_inclusive)
                .and_then(|value| value.checked_mul(2))
                .ok_or(UmapError::Overflow("fuzzy graph capacity"))?;
            if capacity > i32::MAX as usize {
                return Err(UmapError::Overflow("fuzzy graph COO indices"));
            }
            validate_layout_work(capacity, self.config.negative_sample_rate)?;
        }
        let (device, backend) = selected_device(self.backend)?;
        let ann_plan = if search == NeighborSearch::Approximate && n > 1 && widths.nonself > 0 {
            let budget = cuscan_common::memory::resolve_memory_budget(
                &device,
                self.config.memory_limit_bytes,
            )
            .map_err(|error| UmapError::BackendUnavailable(error.to_string()))?;
            Some(
                ann_preflight(&device, n, features, widths.nonself, &budget)
                    .map_err(map_ann_error)?,
            )
        } else {
            None
        };
        let mut modern_buffers = Vec::new();
        if self.config.optimizer != super::Optimizer::Compatibility && n > 1 && widths.nonself > 0 {
            let size = super::modern_runtime::allocation_bytes;
            let matrix = |rows, columns| {
                super::modern_runtime::matrix_allocation_bytes(&device, rows, columns, 4)
            };
            modern_buffers
                .try_reserve_exact(192)
                .map_err(|e| UmapError::Memory(format!("modern plan entries: {e}")))?;
            modern_buffers.push(LiveBuffer {
                name: "retained training features",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Both,
                bytes: matrix(n, features)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "retained fit graph",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Both,
                bytes: matrix(n, widths.nonself)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "original input rows",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::RepeatedBoth(original_n),
                bytes: size(features, 1, 4)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "original input row metadata",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Both,
                bytes: size(original_n, 1, size_of::<Vec<f32>>())?,
            });
            modern_buffers.push(LiveBuffer {
                name: "fit output rows",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::FutureResult(n),
                bytes: size(self.config.n_components, 1, 4)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "fit output row metadata",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::FutureResult(1),
                bytes: size(n, 1, size_of::<Vec<f32>>())?,
            });
            modern_buffers.push(LiveBuffer {
                name: "training feature upload staging",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Build,
                bytes: size(n, features, 4)?,
            });
            if matches!(
                device.as_dispatch(),
                burn::backend::DispatchDevice::Cube(burn_cubecl::CubeDevice::Cuda(_))
            ) {
                modern_buffers.push(LiveBuffer {
                    name: "training feature pinned upload staging",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Build,
                    bytes: size(n, features, 4)?,
                });
            }
            modern_buffers.push(LiveBuffer {
                name: "initialization flat upload",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Build,
                bytes: size(n, self.config.n_components, 4)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "neighbor distances",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: matrix(n, widths.nonself)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "self-inclusive neighbor indices",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: matrix(n, widths.self_inclusive)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "self-inclusive neighbor distances",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: matrix(n, widths.self_inclusive)?,
            });
            let capacity = size(n, widths.self_inclusive, 2)?;
            if self.config.init == Init::Spectral {
                modern_buffers.extend(umap_core::init::spectral_solver_buffers(
                    n,
                    self.config.n_components,
                    features,
                    capacity,
                )?);
            }
            if search == NeighborSearch::Exact {
                super::modern_runtime::exact_search_buffers(
                    &mut modern_buffers,
                    &device,
                    n,
                    n,
                    features,
                    widths.nonself,
                )?;
            }
            if self.config.init == Init::Spectral
                || self.config.init_rescale == InitRescale::GraphEdge
            {
                for name in [
                    "initialization graph head readback",
                    "initialization graph tail readback",
                    "initialization graph weight readback",
                ] {
                    modern_buffers.push(LiveBuffer {
                        name,
                        space: BufferSpace::Host,
                        lifetime: BufferLifetime::Build,
                        bytes: size(capacity, 1, 4)?,
                    });
                }
                for (name, element) in [
                    ("initialization compact heads", size_of::<usize>()),
                    ("initialization compact tails", size_of::<usize>()),
                    ("initialization compact weights", size_of::<f32>()),
                ] {
                    modern_buffers.push(LiveBuffer {
                        name,
                        space: BufferSpace::Host,
                        lifetime: BufferLifetime::Build,
                        bytes: size(capacity, 1, element)?,
                    });
                }
            }
            for name in [
                "graph rho",
                "graph sigma",
                "graph counts",
                "graph offsets",
                "graph cursors",
            ] {
                modern_buffers.push(LiveBuffer {
                    name,
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: size(n, 1, 4)?,
                });
            }
            modern_buffers.push(LiveBuffer {
                name: "graph directed weights",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: size(n, widths.self_inclusive, 4)?,
            });
            if self.config.init.resolve() != Init::Recursive {
                for name in ["classic positive intervals", "classic negative intervals"] {
                    modern_buffers.push(LiveBuffer {
                        name,
                        space: BufferSpace::Device,
                        lifetime: BufferLifetime::Build,
                        bytes: size(capacity, 1, 8)?,
                    });
                }
                modern_buffers.push(LiveBuffer {
                    name: "classic RNG",
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: matrix(n, 3)?,
                });
            }
            if self.config.unique {
                modern_buffers.push(LiveBuffer {
                    name: "distinct training rows",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::RepeatedBoth(n),
                    bytes: size(features, 1, 4)?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "distinct training row metadata",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes: size(original_n, 1, size_of::<Vec<f32>>())?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "unique inverse mapping",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes: size(original_n, 1, size_of::<usize>())?,
                });
            }
            super::modern_runtime::preflight(
                &device,
                super::modern_runtime::LayoutAdmissionStage::ConstructionBoundsOnly,
                LayoutOperation::Fit,
                n,
                n,
                self.config.n_components,
                features,
                original_n,
                capacity,
                capacity,
                &self.config,
                &modern_buffers,
                ann_plan.as_ref(),
            )?;
        }
        let training_data = matrix_tensor(data, &device);
        let n_components = self.config.n_components;
        n.checked_mul(n_components)
            .ok_or(UmapError::Overflow("embedding shape"))?;

        if n == 1 {
            let embedding = vec![vec![0.0; n_components]];
            let embedding_tensor = matrix_tensor(&embedding, &device);
            return Ok(FittedUMAP {
                config: self.config.clone(),
                backend,
                requested_search: self.search,
                search,
                n_features: features,
                training_data,
                fit_graph: None,
                embedding,
                original_to_training,
                expanded_embedding: OnceLock::new(),
                embedding_tensor,
                a: 0.0,
                b: 0.0,
            });
        }
        if widths.nonself == 0 {
            return Err(UmapError::InvalidInput(
                "fit neighbor graph is empty for two samples".into(),
            ));
        }

        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        let _neighbors_span = tracing::info_span!(
            "neighbor_search",
            rows = n,
            neighbors = widths.nonself,
            search = ?search,
        )
        .entered();
        let (nonself_indices, nonself_distances) = match search {
            NeighborSearch::Exact => {
                let neighbors = super::neighbors::exact_neighbors(
                    training_data.clone(),
                    training_data.clone(),
                    widths.nonself,
                    true,
                )?;
                (neighbors.indices, neighbors.distances)
            }
            NeighborSearch::Approximate => {
                let plan = ann_plan
                    .as_ref()
                    .ok_or_else(|| UmapError::Numerical("ANN plan unavailable".into()))?;
                let iters = (n as f64).log2().round().max(5.0) as u32;
                cuscan_common::neighbors::approximate_neighbors(
                    training_data.clone(),
                    plan,
                    seed,
                    iters,
                    false,
                )
                .map_err(map_ann_error)?
            }
            NeighborSearch::Auto => unreachable!("search is resolved before dispatch"),
        };
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        drop(_neighbors_span);
        let rows = Tensor::<1, Int>::arange(0..n as i64, &device)
            .cast(burn::tensor::IntDType::I32)
            .reshape([n, 1]);
        let fit_graph = nonself_indices.clone();
        let indices = Tensor::<2, Int>::cat(vec![rows, nonself_indices], 1);
        let distances = Tensor::<2>::cat(
            vec![Tensor::<2>::zeros([n, 1], &device), nonself_distances],
            1,
        );
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        let _graph_span =
            tracing::info_span!("fuzzy_graph", rows = n, neighbors = widths.self_inclusive)
                .entered();
        let (heads, tails, weights, _, _, _) = <Dispatch as StandardUmapOps>::fuzzy_graph(
            indices.into_dispatch(),
            distances.into_dispatch(),
            self.config.local_connectivity,
            1.0,
            self.config.set_op_mix_ratio,
            false,
            u32::try_from(n).map_err(|_| UmapError::Overflow("reference count"))?,
        );
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        drop(_graph_span);
        let heads: Tensor<1, Int> = Tensor::from_dispatch(heads);
        let tails: Tensor<1, Int> = Tensor::from_dispatch(tails);
        let weights: Tensor<1> = Tensor::from_dispatch(weights);
        let graph_marker = heads
            .clone()
            .slice(0..1)
            .to_data()
            .try_to_vec::<i32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not inspect fuzzy graph marker: {error}"))
            })?;
        if graph_marker.first() == Some(&-2) {
            return Err(UmapError::Numerical(
                "fuzzy graph contains missing or invalid neighbor indices (for example, a squared feature distance overflowed float32)".into(),
            ));
        }
        let fit_graph = ValidatedFitGraph(fit_graph);
        let default_epochs = if n <= 10_000 { 500 } else { 200 };
        let requested_epochs = self.config.n_epochs.unwrap_or(default_epochs);
        let n_epochs = u32::try_from(requested_epochs)
            .map_err(|_| UmapError::Overflow("optimization epochs"))?;
        let recursive = self.config.init.resolve() == Init::Recursive;
        let recursive_modern =
            recursive && self.config.optimizer != super::Optimizer::Compatibility;
        #[cfg(feature = "tracing")]
        let _schedule_span = tracing::info_span!("schedule", rows = n).entered();
        let maximum = weights.clone().max_dim(0);
        let (heads, tails, weights, clocks) = if recursive_modern {
            let divisor = if n_epochs > 10 {
                n_epochs as f64
            } else {
                default_epochs as f64
            };
            let threshold = maximum.clone().div_scalar(divisor);
            let pruned = weights
                .clone()
                .mask_fill(weights.clone().lower(threshold), 0.0);
            (heads, tails, pruned, None)
        } else {
            let (heads, tails, weights, positive, next_positive, negative, next_negative) =
                prepare_schedule(
                    heads,
                    tails,
                    weights,
                    maximum.clone(),
                    n_epochs,
                    default_epochs,
                    self.config.negative_sample_rate,
                )?;
            (
                heads,
                tails,
                weights,
                Some((positive, next_positive, negative, next_negative)),
            )
        };

        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        drop(_schedule_span);

        // Read after scheduling, as before. The maximum is the pre-prune value;
        // pruning keeps it (divisor >= 1), so this equals the post-prune maximum.
        validate_graph_maximum(
            maximum,
            "could not inspect fuzzy graph weights",
            "fuzzy graph is empty after positive-edge pruning",
        )?;

        #[cfg(feature = "tracing")]
        let _export_span = tracing::info_span!("spectral_graph_export", rows = n).entered();
        let (init_heads, init_tails, init_weights) = if recursive
            || self.config.init == Init::Spectral
            || self.config.init_rescale == InitRescale::GraphEdge
        {
            // Export peak: 12*C readback bytes overlap E*(2*sizeof(usize)+4)
            // compact directed bytes. Raw arrays die here, before core scratch;
            // compact edges die after initialize, before upload/layout.
            let readback_bytes = weights.dims()[0]
                .checked_mul(2 * size_of::<i32>() + size_of::<f32>())
                .ok_or(UmapError::Overflow("initialization graph readback"))?;
            if recursive {
                let edge_bytes = weights.dims()[0]
                    .checked_mul(
                        3 * size_of::<usize>()
                            + 2 * size_of::<f32>()
                            + size_of::<(usize, usize, f32)>(),
                    )
                    .ok_or(UmapError::Overflow("recursive export edges"))?;
                let offset_bytes = n
                    .checked_add(1)
                    .and_then(|rows| rows.checked_mul(size_of::<usize>()))
                    .ok_or(UmapError::Overflow("recursive export offsets"))?;
                let csr_build = edge_bytes
                    .checked_add(offset_bytes)
                    .ok_or(UmapError::Overflow("recursive export CSR build"))?;
                let readback_peak = weights.dims()[0]
                    .checked_mul(2 * size_of::<usize>() + size_of::<f32>())
                    .and_then(|bytes| bytes.checked_add(readback_bytes))
                    .ok_or(UmapError::Overflow("recursive export readback peak"))?;
                admit_recursive_layout(
                    &device,
                    "recursive export",
                    csr_build.max(readback_peak),
                    0,
                )?;
            }
            let host_heads = heads.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!(
                    "could not read initialization graph heads: {error}"
                ))
            })?;
            let host_tails = tails.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!(
                    "could not read initialization graph tails: {error}"
                ))
            })?;
            let host_weights = weights.to_data().try_to_vec::<f32>().map_err(|error| {
                UmapError::Numerical(format!(
                    "could not read initialization graph weights: {error}"
                ))
            })?;
            let positive_edges = host_weights.iter().filter(|&&weight| weight > 0.0).count();
            #[cfg(feature = "tracing")]
            tracing::debug!(target: "umap::init_export", positive_edges, "exported pruned initialization graph");
            positive_edges
                .checked_mul(2 * size_of::<usize>() + size_of::<f32>())
                .and_then(|bytes| bytes.checked_add(readback_bytes))
                .ok_or(UmapError::Overflow("initialization graph export peak"))?;
            let mut valid_heads = Vec::new();
            let mut valid_tails = Vec::new();
            let mut valid_weights = Vec::new();
            valid_heads
                .try_reserve_exact(positive_edges)
                .map_err(|error| {
                    UmapError::Memory(format!(
                        "could not allocate initialization graph heads: {error}"
                    ))
                })?;
            valid_tails
                .try_reserve_exact(positive_edges)
                .map_err(|error| {
                    UmapError::Memory(format!(
                        "could not allocate initialization graph tails: {error}"
                    ))
                })?;
            valid_weights
                .try_reserve_exact(positive_edges)
                .map_err(|error| {
                    UmapError::Memory(format!(
                        "could not allocate initialization graph weights: {error}"
                    ))
                })?;
            for ((head, tail), weight) in host_heads.into_iter().zip(host_tails).zip(host_weights) {
                if !weight.is_finite() || weight < 0.0 {
                    return Err(UmapError::Numerical(
                        "invalid initialization graph weight".into(),
                    ));
                }
                if weight > 0.0 {
                    let head = usize::try_from(head).map_err(|_| {
                        UmapError::Numerical("invalid initialization graph head".into())
                    })?;
                    let tail = usize::try_from(tail).map_err(|_| {
                        UmapError::Numerical("invalid initialization graph tail".into())
                    })?;
                    if head >= n || tail >= n || head == tail {
                        return Err(UmapError::Numerical(
                            "invalid initialization graph endpoints".into(),
                        ));
                    }
                    valid_heads.push(head);
                    valid_tails.push(tail);
                    valid_weights.push(weight);
                }
            }
            (valid_heads, valid_tails, valid_weights)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let legacy_graph = if recursive_modern {
            drop((heads, tails, weights));
            None
        } else {
            Some((heads, tails, weights))
        };
        #[cfg(feature = "tracing")]
        drop(_export_span);
        #[cfg(feature = "tracing")]
        let _initialization_span = tracing::info_span!(
            "initialization",
            rows = n,
            components = n_components,
            method = ?self.config.init,
        )
        .entered();
        use umap_core::{
            modern_layout::{ModernLayoutParams, compact_graph_from_csr},
            recursive_init,
        };
        let (a, b) = fit_curve(self.config.min_dist, self.config.spread)?;
        let mut recursive_graph = None;
        let flat = if recursive {
            let graph = recursive_init::graph::HostCsr::from_coo(
                n,
                &init_heads,
                &init_tails,
                &init_weights,
            )?;
            let feature_bytes = n
                .checked_mul(features)
                .and_then(|count| count.checked_mul(size_of::<f32>()))
                .ok_or(UmapError::Overflow("recursive features"))?;
            admit_recursive_layout(&device, "recursive features", feature_bytes, 0)?;
            let mut feature_values = Vec::new();
            feature_values
                .try_reserve_exact(
                    n.checked_mul(features)
                        .ok_or(UmapError::Overflow("recursive features"))?,
                )
                .map_err(|error| UmapError::Memory(error.to_string()))?;
            feature_values.extend(data.iter().flatten().copied());
            let size = super::modern_runtime::allocation_bytes;
            let matrix = |rows, columns| {
                super::modern_runtime::matrix_allocation_bytes(&device, rows, columns, 4)
            };
            let mut retained = Vec::new();
            retained
                .try_reserve_exact(10)
                .map_err(|error| UmapError::Memory(error.to_string()))?;
            for (name, space, bytes) in [
                (
                    "retained training features",
                    BufferSpace::Device,
                    matrix(n, features)?,
                ),
                (
                    "retained fit graph",
                    BufferSpace::Device,
                    matrix(n, widths.nonself)?,
                ),
                (
                    "original input rows",
                    BufferSpace::Host,
                    size(original_n, features, 4)?,
                ),
                (
                    "original input row metadata",
                    BufferSpace::Host,
                    size(original_n, 1, size_of::<Vec<f32>>())?,
                ),
                (
                    "recursive exported heads",
                    BufferSpace::Host,
                    size(init_heads.capacity(), 1, size_of::<usize>())?,
                ),
                (
                    "recursive exported tails",
                    BufferSpace::Host,
                    size(init_tails.capacity(), 1, size_of::<usize>())?,
                ),
                (
                    "recursive exported weights",
                    BufferSpace::Host,
                    size(init_weights.capacity(), 1, size_of::<f32>())?,
                ),
            ] {
                retained.push(LiveBuffer {
                    name,
                    space,
                    lifetime: BufferLifetime::Both,
                    bytes,
                });
            }
            if let Some(rows) = &distinct {
                let row_bytes = rows
                    .iter()
                    .try_fold(0usize, |sum, row| {
                        sum.checked_add(row.capacity().checked_mul(size_of::<f32>())?)
                    })
                    .ok_or(UmapError::Overflow("distinct training rows"))?;
                let bytes = row_bytes
                    .checked_add(size(rows.capacity(), 1, size_of::<Vec<f32>>())?)
                    .and_then(|bytes| {
                        bytes.checked_add(
                            original_to_training
                                .as_ref()?
                                .capacity()
                                .checked_mul(size_of::<usize>())?,
                        )
                    })
                    .ok_or(UmapError::Overflow("distinct training rows"))?;
                retained.push(LiveBuffer {
                    name: "distinct rows and inverse",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes,
                });
            }
            if let Some((heads, _, _)) = &legacy_graph {
                let bytes = size(heads.dims()[0], 7, size_of::<f32>())?;
                retained.push(LiveBuffer {
                    name: "legacy COO and clocks",
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Both,
                    bytes,
                });
            }
            let mut values = recursive_driver(
                &graph,
                &feature_values,
                features,
                (a, b),
                &self.config,
                seed,
                &device,
                &retained,
            )?;
            drop(feature_values);
            let scratch = if self.config.init_rescale == InitRescale::GraphEdge {
                init_heads.len().checked_mul(size_of::<(usize, f64, f64)>())
            } else {
                n.checked_add(n_components)
                    .and_then(|count| count.checked_mul(size_of::<f64>()))
            }
            .ok_or(UmapError::Overflow("recursive finish scratch"))?;
            let upload = n
                .checked_mul(n_components)
                .and_then(|count| count.checked_mul(size_of::<f32>()))
                .ok_or(UmapError::Overflow("recursive initialization upload"))?;
            let device_upload = super::modern_runtime::matrix_allocation_bytes(
                &device,
                n,
                n_components,
                size_of::<f32>(),
            )?;
            admit_recursive_layout(
                &device,
                "recursive finish",
                scratch.max(upload),
                device_upload,
            )?;
            recursive_init::finish(
                &mut values,
                n,
                &init_heads,
                &init_tails,
                &init_weights,
                &self.config,
            )?;
            recursive_graph = Some(graph);
            values
        } else {
            super::initialize(
                data,
                &init_heads,
                &init_tails,
                &init_weights,
                &self.config,
                seed,
            )?
        };
        drop((init_heads, init_tails, init_weights));
        #[cfg(feature = "tracing")]
        drop(_initialization_span);
        let rng = if recursive_modern {
            None
        } else {
            if recursive {
                let mut host = n
                    .checked_mul(3 * size_of::<i64>())
                    .ok_or(UmapError::Overflow("recursive optimizer RNG"))?;
                if device != Device::cpu() {
                    host = host
                        .checked_mul(2)
                        .ok_or(UmapError::Overflow("recursive optimizer RNG upload"))?;
                }
                let device_bytes = super::modern_runtime::matrix_allocation_bytes(
                    &device,
                    n,
                    3,
                    size_of::<i64>(),
                )?;
                admit_recursive_layout(&device, "recursive optimizer RNG", host, device_bytes)?;
            }
            Some(optimizer_rng_from_first_coordinates(
                flat.chunks_exact(n_components).map(|row| row[0]),
                n,
                seed,
                &device,
            )?)
        };
        let initial = Tensor::<2>::from_data(TensorData::new(flat, [n, n_components]), &device);
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        let _optimization_span =
            tracing::info_span!("optimization", rows = n, epochs = n_epochs).entered();
        let embedding_tensor = if self.config.optimizer == super::Optimizer::Compatibility {
            let (heads, tails, weights) = legacy_graph.expect("compatibility graph");
            let (positive, next_positive, negative, next_negative) =
                clocks.expect("compatibility clocks");
            optimize(
                initial.clone(),
                initial,
                heads,
                tails,
                weights,
                positive,
                next_positive,
                negative,
                next_negative,
                rng.expect("compatibility RNG"),
                n_epochs,
                a,
                b,
                self.config.learning_rate,
                self.config.repulsion_strength,
                self.config.negative_sample_rate,
                true,
                true,
            )?
        } else if let Some(graph) = recursive_graph {
            if n_epochs == 0 {
                initial
            } else {
                let params = ModernLayoutParams::main(
                    &self.config,
                    n_epochs,
                    a,
                    b,
                    self.config.learning_rate,
                    true,
                    seed,
                );
                admit_recursive_layout(
                    &device,
                    "recursive main compact",
                    recursive_compact_bytes(n, graph.indices.len())?,
                    0,
                )?;
                let compact = compact_graph_from_csr(
                    &graph,
                    n_epochs as usize,
                    self.config.negative_sample_rate as f32,
                )?;
                drop(graph);
                super::modern_runtime::optimize_compact(
                    initial.clone(),
                    initial,
                    compact,
                    &params,
                    LayoutOperation::Fit,
                    features,
                    original_n,
                    &modern_buffers,
                )?
            }
        } else {
            let (heads, tails, weights) = legacy_graph.expect("modern graph");
            drop((clocks, rng));
            super::modern_runtime::optimize(
                initial.clone(),
                initial,
                heads,
                tails,
                weights,
                n_epochs,
                a,
                b,
                self.config.learning_rate,
                &self.config,
                LayoutOperation::Fit,
                seed,
                features,
                original_n,
                &modern_buffers,
            )?
        };
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
        #[cfg(feature = "tracing")]
        drop(_optimization_span);
        let embedding = tensor_embedding(&embedding_tensor)?;
        Ok(FittedUMAP {
            config: self.config.clone(),
            backend,
            requested_search: self.search,
            search,
            n_features: features,
            training_data,
            fit_graph: Some(fit_graph),
            embedding,
            original_to_training,
            expanded_embedding: OnceLock::new(),
            embedding_tensor,
            a,
            b,
        })
    }

    /// Fit the model and return it together with the training embedding.
    ///
    /// Mirrors Python `UMAP.fit_transform`, which fits a reusable model and returns
    /// `model.embedding_`: the returned embedding is the one stored on the model.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip_all,
            fields(rows = data.len(), features = data.first().map_or(0, Vec::len))
        )
    )]
    pub fn fit_transform(&self, data: &[Vec<f32>]) -> Result<(FittedUMAP, Embedding)> {
        let fitted = self.fit(data)?;
        let embedding = fitted.embedding().clone();
        Ok((fitted, embedding))
    }

    fn resolve_search(&self, n: usize) -> NeighborSearch {
        match self.search {
            NeighborSearch::Auto if n <= AUTO_EXACT_MAX_ROWS => NeighborSearch::Exact,
            NeighborSearch::Auto => NeighborSearch::Approximate,
            mode => mode,
        }
    }
}

/// A fitted standard UMAP model supporting bipartite transform.
#[derive(Debug, Clone)]
pub struct FittedUMAP {
    config: UMAPConfig,
    backend: crate::standard_umap::BackendChoice,
    requested_search: NeighborSearch,
    search: NeighborSearch,
    n_features: usize,
    training_data: Tensor<2>,
    fit_graph: Option<ValidatedFitGraph>,
    embedding: Embedding,
    original_to_training: Option<Vec<usize>>,
    expanded_embedding: OnceLock<Embedding>,
    embedding_tensor: Tensor<2>,
    a: f32,
    b: f32,
}

/// A fit graph whose indices were validated by the fit fuzzy-graph marker.
#[derive(Debug, Clone)]
struct ValidatedFitGraph(Tensor<2, Int>);

impl ValidatedFitGraph {
    fn clone_dispatch(&self) -> IntTensor<Dispatch> {
        self.0.clone().into_dispatch()
    }
}

impl FittedUMAP {
    /// Export an exact, storage-independent snapshot of fitted state.
    pub fn export_state(&self) -> Result<super::ModelState> {
        let [training_rows, n_features] = self.training_data.dims();
        let [embedding_rows, n_components] = self.embedding_tensor.dims();
        let training_features = self
            .training_data
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|e| UmapError::Numerical(format!("could not read training features: {e}")))?;
        // Exact transforms search the retained features directly; only approximate
        // transforms consume the fitted neighbor graph.
        let query_graph = (self.search == NeighborSearch::Approximate)
            .then_some(self.fit_graph.as_ref())
            .flatten()
            .map(|graph| {
                graph
                    .0
                    .to_data()
                    .try_to_vec::<i32>()
                    .map_err(|e| UmapError::Numerical(format!("could not read fit graph: {e}")))
            })
            .transpose()?;
        Ok(super::ModelState {
            config: self.config.clone(),
            requested_search: self.requested_search,
            resolved_search: self.search,
            a: self.a,
            b: self.b,
            training_features,
            training_rows,
            n_features,
            embedding: self
                .embedding_tensor
                .to_data()
                .try_to_vec::<f32>()
                .map_err(|e| {
                    UmapError::Numerical(format!("could not read fitted embedding: {e}"))
                })?,
            embedding_rows,
            n_components,
            query_graph,
            query_graph_width: if self.search == NeighborSearch::Approximate {
                self.fit_graph.as_ref().map_or(0, |graph| graph.0.dims()[1])
            } else {
                0
            },
            original_to_training: self.original_to_training.clone(),
        })
    }

    /// Validate and reconstruct a fitted model on the requested execution backend.
    pub fn from_state(
        state: super::ModelState,
        backend: crate::standard_umap::BackendChoice,
    ) -> Result<Self> {
        state.config.validate()?;
        if state.training_rows == 0
            || state.n_features == 0
            || state.embedding_rows != state.training_rows
            || state.n_components == 0
            || state.n_components != state.config.n_components
        {
            return Err(UmapError::InvalidInput(
                "model state dimensions are inconsistent".into(),
            ));
        }
        let feature_count = state
            .training_rows
            .checked_mul(state.n_features)
            .ok_or(UmapError::Overflow("training feature shape"))?;
        let embedding_count = state
            .embedding_rows
            .checked_mul(state.n_components)
            .ok_or(UmapError::Overflow("embedding shape"))?;
        if state.training_features.len() != feature_count
            || state.embedding.len() != embedding_count
        {
            return Err(UmapError::InvalidInput(
                "model state array lengths do not match dimensions".into(),
            ));
        }
        if state.training_rows > i32::MAX as usize {
            return Err(UmapError::Overflow("neighbor row indices"));
        }
        if state
            .training_features
            .iter()
            .chain(&state.embedding)
            .any(|v| !v.is_finite())
            || !state.a.is_finite()
            || !state.b.is_finite()
        {
            return Err(UmapError::InvalidInput(
                "model state contains non-finite values".into(),
            ));
        }
        if state.requested_search == NeighborSearch::Auto {
            let expected = if state.training_rows <= AUTO_EXACT_MAX_ROWS {
                NeighborSearch::Exact
            } else {
                NeighborSearch::Approximate
            };
            if state.resolved_search != expected {
                return Err(UmapError::InvalidConfig(
                    "Auto search resolution is inconsistent with retained row count".into(),
                ));
            }
        } else if state.requested_search != state.resolved_search {
            return Err(UmapError::InvalidConfig(
                "requested and resolved search modes are inconsistent".into(),
            ));
        }
        state
            .config
            .validate_search(state.requested_search, Some(state.resolved_search))?;
        if state.training_rows == 1 {
            if state.query_graph.is_some()
                || state.query_graph_width != 0
                || state.a != 0.0
                || state.b != 0.0
            {
                return Err(UmapError::InvalidInput(
                    "singleton model state must have no graph and a zero curve".into(),
                ));
            }
        } else {
            if state.a <= 0.0 || state.b <= 0.0 {
                return Err(UmapError::InvalidInput(
                    "nonsingleton model state requires positive curve coefficients".into(),
                ));
            }
            if state.resolved_search == NeighborSearch::Approximate {
                let expected_width =
                    fit_neighbor_widths(state.training_rows, state.config.n_neighbors)?.nonself;
                if expected_width == 0 {
                    return Err(UmapError::InvalidInput(
                        "approximate fit graph width must be greater than zero".into(),
                    ));
                }
                let graph = state.query_graph.as_ref().ok_or_else(|| {
                    UmapError::InvalidInput(
                        "fit query graph is required for multiple training rows".into(),
                    )
                })?;
                if state.query_graph_width != expected_width
                    || graph.len()
                        != state
                            .training_rows
                            .checked_mul(expected_width)
                            .ok_or(UmapError::Overflow("fit graph shape"))?
                {
                    return Err(UmapError::InvalidInput(
                        "fit query graph dimensions are inconsistent".into(),
                    ));
                }
                for (row, chunk) in graph.chunks_exact(expected_width).enumerate() {
                    if chunk
                        .iter()
                        .any(|&i| i < 0 || i as usize >= state.training_rows || i as usize == row)
                    {
                        return Err(UmapError::InvalidInput(
                            "fit query graph contains an invalid neighbor index".into(),
                        ));
                    }
                    let mut neighbors = std::collections::HashSet::with_capacity(expected_width);
                    if chunk.iter().any(|&i| !neighbors.insert(i)) {
                        return Err(UmapError::InvalidInput(
                            "fit query graph contains duplicate neighbors".into(),
                        ));
                    }
                }
            } else if state.query_graph.is_some() || state.query_graph_width != 0 {
                return Err(UmapError::InvalidInput(
                    "exact-search model state must not contain an approximate query graph".into(),
                ));
            }
        }
        match (&state.original_to_training, state.config.unique) {
            (Some(mapping), true) => {
                if mapping.len() < state.training_rows
                    || mapping.iter().any(|&i| i >= state.training_rows)
                {
                    return Err(UmapError::InvalidInput(
                        "inverse mapping length or index is invalid".into(),
                    ));
                }
                let mut seen = vec![false; state.training_rows];
                for &i in mapping {
                    seen[i] = true;
                }
                if seen.iter().any(|v| !v) {
                    return Err(UmapError::InvalidInput(
                        "inverse mapping does not cover every retained row".into(),
                    ));
                }
                let mut unique = std::collections::HashSet::with_capacity(state.training_rows);
                for row in state.training_features.chunks_exact(state.n_features) {
                    let key: Vec<u32> = row
                        .iter()
                        .map(|v| if *v == 0.0 { 0 } else { v.to_bits() })
                        .collect();
                    if !unique.insert(key) {
                        return Err(UmapError::InvalidInput(
                            "unique model contains duplicate retained rows".into(),
                        ));
                    }
                }
            }
            (None, false) => {}
            _ => {
                return Err(UmapError::InvalidInput(
                    "inverse mapping presence disagrees with unique configuration".into(),
                ));
            }
        }
        let (device, selected_backend) = selected_device(backend)?;
        let training_data = Tensor::<2>::from_data(
            TensorData::new(
                state.training_features.clone(),
                [state.training_rows, state.n_features],
            ),
            &device,
        );
        let embedding_tensor = Tensor::<2>::from_data(
            TensorData::new(
                state.embedding.clone(),
                [state.embedding_rows, state.n_components],
            ),
            &device,
        );
        let embedding = state
            .embedding
            .chunks_exact(state.n_components)
            .map(<[f32]>::to_vec)
            .collect();
        let fit_graph = state.query_graph.map(|indices| {
            ValidatedFitGraph(Tensor::<2, Int>::from_data(
                TensorData::new(indices, [state.training_rows, state.query_graph_width]),
                &device,
            ))
        });
        Ok(Self {
            config: state.config,
            backend: selected_backend,
            requested_search: state.requested_search,
            search: state.resolved_search,
            n_features: state.n_features,
            training_data,
            fit_graph,
            embedding,
            original_to_training: state.original_to_training,
            expanded_embedding: OnceLock::new(),
            embedding_tensor,
            a: state.a,
            b: state.b,
        })
    }

    pub fn embedding(&self) -> &Embedding {
        match &self.original_to_training {
            Some(inverse) => self.expanded_embedding.get_or_init(|| {
                inverse
                    .iter()
                    .map(|&index| self.embedding[index].clone())
                    .collect()
            }),
            None => &self.embedding,
        }
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Diagnostic read-only copy of the fit's non-self neighbor ids, row-major
    /// `[rows, width]`; reads back from the device. Not part of the stable API.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn diagnostic_fit_graph(&self) -> Result<(Vec<i32>, usize)> {
        let Some(fit_graph) = &self.fit_graph else {
            return Ok((Vec::new(), 0));
        };
        let width = fit_graph.0.dims()[1];
        let ids =
            fit_graph.0.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!("could not read fit graph: {error}"))
            })?;
        Ok((ids, width))
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            skip_all,
            fields(
                rows = data.len(),
                features = data.first().map_or(0, Vec::len),
                training_rows = self.training_data.dims()[0],
                neighbors = self.config.n_neighbors,
                components = self.embedding_tensor.dims()[1],
                backend = ?self.backend,
                search = ?self.search,
                epochs = self.config.n_epochs.unwrap_or(0),
            )
        )
    )]
    pub fn transform(&self, data: &[Vec<f32>]) -> Result<Embedding> {
        self.transform_impl(data, None)
    }

    /// Transform owned row-major input without nested row allocations.
    #[doc(hidden)]
    pub fn transform_flat(
        &self,
        data: Vec<f32>,
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f32>> {
        if n_rows == 0 {
            return Err(UmapError::InvalidInput(
                "input must contain at least one row".into(),
            ));
        }
        if n_features == 0 {
            return Err(UmapError::InvalidInput(
                "input must contain at least one feature".into(),
            ));
        }
        if n_features != self.n_features {
            return Err(UmapError::InvalidInput(format!(
                "input has {n_features} features; expected {}",
                self.n_features
            )));
        }
        let elements = n_rows
            .checked_mul(n_features)
            .ok_or(UmapError::Overflow("transform input shape"))?;
        if data.len() != elements {
            return Err(UmapError::InvalidInput(format!(
                "flat input has {} values; expected {elements}",
                data.len()
            )));
        }
        if let Some(position) = data.iter().position(|value| !value.is_finite()) {
            return Err(UmapError::InvalidInput(format!(
                "value at row {}, feature {} is not finite",
                position / n_features,
                position % n_features
            )));
        }
        self.transform_flat_impl(data, n_rows, n_features, false, None)
    }

    /// Return the transform and the actual training-neighbor IDs used by it.
    ///
    /// Diagnostic readbacks are deliberately separate from timed transforms.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn diagnostic_transform(&self, data: &[Vec<f32>]) -> Result<(Embedding, Vec<i32>)> {
        let mut indices = Vec::new();
        let embedding = self.transform_impl(data, Some(&mut indices))?;
        Ok((embedding, indices))
    }

    /// Measurement-only stage replay, search entry (Phase 7 D6).
    ///
    /// Runs cuscan's query search, graph, pruning, initialization, schedule and
    /// layout, returning every stage boundary. `budget` replaces the
    /// queue-derived approximate budget (C/V override); it must satisfy
    /// `k <= candidates <= visits <= n_fitted` and requires approximate search.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn replay_transform_search(
        &self,
        data: &[Vec<f32>],
        budget: Option<QueryBudget>,
    ) -> Result<TransformReplay> {
        let (q, features) = validate_input(data, Some(self.n_features))?;
        if budget.is_some() {
            let n = self.training_data.dims()[0];
            self.query_budget(n, query_neighbor_width(n, self.config.n_neighbors)?, budget)?;
        }
        let count = q
            .checked_mul(features)
            .ok_or(UmapError::Overflow("replay input shape"))?;
        let modern_buffers = self.prepare_transform(q, features, true, budget)?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(count)
            .map_err(|error| UmapError::Memory(format!("replay input: {error}")))?;
        flat.extend(data.iter().flat_map(|row| row.iter().copied()));
        let mut replay = TransformReplay::default();
        let (indices, distances) = self.transform_neighbors(
            flat,
            q,
            features,
            budget,
            None,
            Some(&mut replay.query_stats),
        )?;
        self.replay_from_neighbors(indices, distances, &modern_buffers, replay)
    }

    /// Measurement-only stage replay, graph entry (Phase 7 D6).
    ///
    /// Takes injected query kNN (`rows * k` training IDs and distances,
    /// row-major, `k = query_neighbor_width(n_fitted, n_neighbors)`) and runs
    /// cuscan's graph, pruning, initialization, schedule and layout.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn replay_transform_graph(
        &self,
        rows: usize,
        indices: &[i32],
        distances: &[f32],
    ) -> Result<TransformReplay> {
        let n = self.training_data.dims()[0];
        let k = query_neighbor_width(n, self.config.n_neighbors)?;
        let count = rows
            .checked_mul(k)
            .ok_or(UmapError::Overflow("replay neighbor shape"))?;
        if rows == 0 || indices.len() != count || distances.len() != count {
            return Err(UmapError::InvalidInput(format!(
                "injected neighbors need {rows} nonzero rows of {k} IDs and distances"
            )));
        }
        if let Some(position) = indices.iter().position(|&id| id < 0 || id as usize >= n) {
            return Err(UmapError::InvalidInput(format!(
                "injected neighbor ID at row {}, slot {} is outside [0, {n})",
                position / k,
                position % k
            )));
        }
        let limit = f32::MAX.sqrt();
        if let Some(position) = distances
            .iter()
            .position(|&value| !(0.0..=limit).contains(&value))
        {
            return Err(UmapError::InvalidInput(format!(
                "injected neighbor distance at row {}, slot {} is not in [0, sqrt(f32::MAX)]",
                position / k,
                position % k
            )));
        }
        let modern_buffers = self.prepare_transform(rows, self.n_features, false, None)?;
        let device = self.training_data.device();
        let indices =
            Tensor::<2, Int>::from_data(TensorData::new(indices.to_vec(), [rows, k]), &device)
                .cast(burn::tensor::IntDType::I32);
        let distances =
            Tensor::<2>::from_data(TensorData::new(distances.to_vec(), [rows, k]), &device);
        self.replay_from_neighbors(
            indices,
            distances,
            &modern_buffers,
            TransformReplay::default(),
        )
    }

    #[cfg(feature = "tracing")]
    fn replay_from_neighbors(
        &self,
        indices: Tensor<2, Int>,
        distances: Tensor<2>,
        modern_buffers: &[LiveBuffer],
        mut replay: TransformReplay,
    ) -> Result<TransformReplay> {
        replay.neighbor_indices = read_ints(indices.clone(), "replay neighbor IDs")?;
        replay.neighbor_distances = read_floats(distances.clone(), "replay neighbor distances")?;
        let embedding =
            self.transform_from_neighbors(indices, distances, modern_buffers, Some(&mut replay))?;
        replay.embedding = embedding;
        Ok(replay)
    }

    /// Measurement-only stage replay, layout entry (Phase 7 D6).
    ///
    /// Takes an injected ordered pruned edge stream (query heads in `[0, rows)`,
    /// training tails, non-negative weights; COO order for Compatibility, any
    /// order for modern optimizers, which build sorted CSR) and an injected
    /// `[rows, n_components]` initial placement. Runs only cuscan's schedule
    /// (pruning against the stream's own maximum) and layout, and returns the
    /// schedule arrays and final embedding. Graph fields stay empty.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn replay_transform_layout(
        &self,
        rows: usize,
        heads: &[i32],
        tails: &[i32],
        weights: &[f32],
        initial: &[f32],
    ) -> Result<TransformReplay> {
        let n = self.training_data.dims()[0];
        let components = self.config.n_components;
        let edges = weights.len();
        if rows == 0 || edges == 0 || heads.len() != edges || tails.len() != edges {
            return Err(UmapError::InvalidInput(
                "injected edge stream needs equal, nonzero head, tail and weight counts".into(),
            ));
        }
        if i32::try_from(edges).is_err() || i32::try_from(rows).is_err() {
            return Err(UmapError::Overflow("injected edge stream indices"));
        }
        if heads.iter().any(|&head| head < 0 || head as usize >= rows)
            || tails.iter().any(|&tail| tail < 0 || tail as usize >= n)
        {
            return Err(UmapError::InvalidInput(
                "injected edge endpoints are outside the query or training rows".into(),
            ));
        }
        if weights
            .iter()
            .any(|&weight| !weight.is_finite() || weight < 0.0)
        {
            return Err(UmapError::InvalidInput(
                "injected edge weights must be finite and non-negative".into(),
            ));
        }
        let placement = rows
            .checked_mul(components)
            .ok_or(UmapError::Overflow("injected initial placement"))?;
        if initial.len() != placement || initial.iter().any(|value| !value.is_finite()) {
            return Err(UmapError::InvalidInput(format!(
                "injected initial placement needs {placement} finite values"
            )));
        }
        validate_layout_work(edges, self.config.negative_sample_rate)?;
        let modern_buffers = self.prepare_transform(rows, self.n_features, false, None)?;
        let device = self.training_data.device();
        let int = |values: &[i32]| {
            Tensor::<1, Int>::from_data(TensorData::new(values.to_vec(), [edges]), &device)
                .cast(burn::tensor::IntDType::I32)
        };
        let heads = int(heads);
        let tails = int(tails);
        let weights = Tensor::<1>::from_data(TensorData::new(weights.to_vec(), [edges]), &device);
        let maximum = weights.clone().max_dim(0);
        validate_graph_maximum(
            maximum.clone(),
            "could not inspect injected transform graph",
            "injected transform graph has no positive edge",
        )?;
        let initial = Tensor::<2>::from_data(
            TensorData::new(initial.to_vec(), [rows, components]),
            &device,
        );
        let mut replay = TransformReplay::default();
        let embedding = self.transform_layout(
            initial,
            heads,
            tails,
            weights,
            maximum,
            &modern_buffers,
            Some(&mut replay),
        )?;
        replay.embedding = embedding;
        Ok(replay)
    }

    fn transform_impl(
        &self,
        data: &[Vec<f32>],
        diagnostic_indices: Option<&mut Vec<i32>>,
    ) -> Result<Embedding> {
        let (q, query_features) = validate_input(data, Some(self.n_features))?;
        let count = q
            .checked_mul(query_features)
            .ok_or(UmapError::Overflow("transform input shape"))?;
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or(UmapError::Overflow("transform input bytes"))?;
        if bytes > isize::MAX as usize {
            return Err(UmapError::Capacity(
                "transform input allocation exceeds host limit".into(),
            ));
        }
        self.prepare_transform(q, query_features, true, None)?;
        let mut flat = Vec::new();
        flat.try_reserve_exact(count)
            .map_err(|error| UmapError::Memory(format!("transform input: {error}")))?;
        flat.extend(data.iter().flat_map(|row| row.iter().copied()));
        let values = self.transform_flat_impl(flat, q, query_features, true, diagnostic_indices)?;
        Ok(values
            .chunks_exact(self.embedding_tensor.dims()[1])
            .map(|row| row.to_vec())
            .collect())
    }

    fn prepare_transform(
        &self,
        q: usize,
        query_features: usize,
        nested: bool,
        budget: Option<QueryBudget>,
    ) -> Result<Vec<LiveBuffer>> {
        if self.training_data.dims()[0] == 1 {
            return Err(UmapError::InvalidInput(
                "Transform unavailable when model was fit with only a single data sample.".into(),
            ));
        }
        let device = self.training_data.device();
        let n = self.training_data.dims()[0];
        let k = query_neighbor_width(n, self.config.n_neighbors)?;
        let graph_capacity = q
            .checked_mul(k)
            .and_then(|value| value.checked_mul(2))
            .ok_or(UmapError::Overflow("transform graph capacity"))?;
        if graph_capacity > i32::MAX as usize {
            return Err(UmapError::Overflow("transform graph COO indices"));
        }
        validate_layout_work(graph_capacity, self.config.negative_sample_rate)?;
        let mut modern_buffers = Vec::new();
        if self.config.optimizer != super::Optimizer::Compatibility {
            let size = super::modern_runtime::allocation_bytes;
            let matrix = |rows, columns| {
                super::modern_runtime::matrix_allocation_bytes(&device, rows, columns, 4)
            };
            modern_buffers
                .try_reserve_exact(192)
                .map_err(|e| UmapError::Memory(format!("modern plan entries: {e}")))?;
            modern_buffers.push(LiveBuffer {
                name: "retained training features",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Both,
                bytes: matrix(n, self.n_features)?,
            });
            if let Some(graph) = &self.fit_graph {
                modern_buffers.push(LiveBuffer {
                    name: "retained fit graph",
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Both,
                    bytes: matrix(n, graph.0.dims()[1])?,
                });
            }
            modern_buffers.push(LiveBuffer {
                name: "retained host embedding rows",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::RepeatedBoth(n),
                bytes: size(self.config.n_components, 1, 4)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "retained host embedding row metadata",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Both,
                bytes: size(n, 1, size_of::<Vec<f32>>())?,
            });
            modern_buffers.push(LiveBuffer {
                name: "query upload",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: matrix(q, query_features)?,
            });
            modern_buffers.push(LiveBuffer {
                name: "query input",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Build,
                bytes: size(q, query_features, 4)?,
            });
            if matches!(
                device.as_dispatch(),
                burn::backend::DispatchDevice::Cube(burn_cubecl::CubeDevice::Cuda(_))
            ) {
                modern_buffers.push(LiveBuffer {
                    name: "query pinned upload staging",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Build,
                    bytes: size(q, query_features, 4)?,
                });
            }
            if nested {
                modern_buffers.push(LiveBuffer {
                    name: "nested query rows",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::RepeatedBoth(q),
                    bytes: size(query_features, 1, 4)?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "nested query metadata",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes: size(q, 1, size_of::<Vec<f32>>())?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "nested transform output rows",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::FutureResult(q),
                    bytes: size(self.config.n_components, 1, 4)?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "nested transform output metadata",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::FutureResult(1),
                    bytes: size(q, 1, size_of::<Vec<f32>>())?,
                });
            }
            for name in ["graph rho", "graph sigma"] {
                modern_buffers.push(LiveBuffer {
                    name,
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: size(q, 1, 4)?,
                });
            }
            let nodes = q
                .checked_add(n)
                .ok_or(UmapError::Overflow("transform graph nodes"))?;
            i32::try_from(nodes).map_err(|_| UmapError::Overflow("transform graph nodes"))?;
            for name in ["graph counts", "graph offsets", "graph cursors"] {
                modern_buffers.push(LiveBuffer {
                    name,
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: size(nodes, 1, 4)?,
                });
            }
            for name in [
                "query indices",
                "query distances",
                "directed weights",
                "normalized weights",
            ] {
                modern_buffers.push(LiveBuffer {
                    name,
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: matrix(q, k)?,
                });
            }
            if self.search == NeighborSearch::Exact {
                super::modern_runtime::exact_search_buffers(
                    &mut modern_buffers,
                    &device,
                    q,
                    n,
                    query_features,
                    k,
                )?;
            }
            for name in ["classic positive intervals", "classic negative intervals"] {
                modern_buffers.push(LiveBuffer {
                    name,
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: size(graph_capacity, 1, 8)?,
                });
            }
            modern_buffers.push(LiveBuffer {
                name: "classic RNG",
                space: BufferSpace::Device,
                lifetime: BufferLifetime::Build,
                bytes: matrix(q, 3)?,
            });
            if self.search == NeighborSearch::Approximate {
                let budget = self.query_budget(n, k, budget)?;
                let graph = self.fit_graph.as_ref().ok_or_else(|| {
                    UmapError::Numerical("validated fit graph unavailable".into())
                })?;
                let chunk = approximate_query_scratch_layout(
                    q,
                    budget.candidates,
                    budget.visits,
                    graph.0.dims()[1],
                )?
                .max_chunk_rows;
                for name in ["candidate ids", "candidate distances", "candidate expanded"] {
                    modern_buffers.push(LiveBuffer {
                        name,
                        space: BufferSpace::Device,
                        lifetime: BufferLifetime::Build,
                        bytes: matrix(chunk, budget.candidates)?,
                    });
                }
                for name in ["visited ids", "visited distances"] {
                    modern_buffers.push(LiveBuffer {
                        name,
                        space: BufferSpace::Device,
                        lifetime: BufferLifetime::Build,
                        bytes: matrix(chunk, budget.visits)?,
                    });
                }
                modern_buffers.push(LiveBuffer {
                    name: "query statistics",
                    space: BufferSpace::Device,
                    lifetime: BufferLifetime::Build,
                    bytes: matrix(q, 2)?,
                });
            }
            if let Some(inverse) = &self.original_to_training {
                modern_buffers.push(LiveBuffer {
                    name: "unique inverse mapping",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes: size(inverse.len(), 1, size_of::<usize>())?,
                });
            }
            if let Some(expanded) = self.expanded_embedding.get() {
                modern_buffers.push(LiveBuffer {
                    name: "expanded host embedding rows",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::RepeatedBoth(expanded.len()),
                    bytes: size(self.config.n_components, 1, 4)?,
                });
                modern_buffers.push(LiveBuffer {
                    name: "expanded host embedding row metadata",
                    space: BufferSpace::Host,
                    lifetime: BufferLifetime::Both,
                    bytes: size(expanded.len(), 1, size_of::<Vec<f32>>())?,
                });
            }
            super::modern_runtime::preflight(
                &device,
                super::modern_runtime::LayoutAdmissionStage::ConstructionBoundsOnly,
                LayoutOperation::Transform,
                q,
                n,
                self.config.n_components,
                self.n_features,
                q,
                graph_capacity,
                graph_capacity,
                &self.config,
                &modern_buffers,
                None,
            )?;
        }
        Ok(modern_buffers)
    }

    fn transform_flat_impl(
        &self,
        data: Vec<f32>,
        q: usize,
        query_features: usize,
        nested: bool,
        diagnostic_indices: Option<&mut Vec<i32>>,
    ) -> Result<Vec<f32>> {
        let modern_buffers = self.prepare_transform(q, query_features, nested, None)?;
        let (indices, distances) =
            self.transform_neighbors(data, q, query_features, None, diagnostic_indices, None)?;
        self.transform_from_neighbors(indices, distances, &modern_buffers, None)
    }

    /// The approximate-search budget: the configured one, or a measurement override.
    fn query_budget(&self, n: usize, k: usize, budget: Option<QueryBudget>) -> Result<QueryBudget> {
        match budget {
            None => approximate_query_budget(n, k, self.config.transform_queue_size),
            Some(budget) => {
                if self.search != NeighborSearch::Approximate {
                    return Err(UmapError::InvalidInput(
                        "a query budget override requires approximate search".into(),
                    ));
                }
                if budget.candidates < k || budget.visits < budget.candidates || budget.visits > n {
                    return Err(UmapError::InvalidInput(format!(
                        "query budget override must satisfy {k} <= candidates <= visits <= {n}"
                    )));
                }
                Ok(budget)
            }
        }
    }

    /// Transform query kNN: training-neighbor IDs and distances, `[q, k]`.
    fn transform_neighbors(
        &self,
        data: Vec<f32>,
        q: usize,
        query_features: usize,
        budget: Option<QueryBudget>,
        diagnostic_indices: Option<&mut Vec<i32>>,
        mut diagnostic_stats: Option<&mut Vec<i32>>,
    ) -> Result<(Tensor<2, Int>, Tensor<2>)> {
        let device = self.training_data.device();
        let n = self.training_data.dims()[0];
        let k = query_neighbor_width(n, self.config.n_neighbors)?;
        #[cfg(feature = "tracing")]
        let _neighbors_span = tracing::info_span!(
            "neighbor_search",
            query_rows = q,
            training_rows = n,
            neighbors = k,
            search = ?self.search,
        )
        .entered();
        let (indices, distances) = match self.search {
            NeighborSearch::Exact => {
                let result = super::neighbors::exact_neighbors(
                    Tensor::from_data(TensorData::new(data, [q, query_features]), &device),
                    self.training_data.clone(),
                    k,
                    false,
                )?;
                (result.indices, result.distances)
            }
            NeighborSearch::Approximate => {
                let budget = self.query_budget(n, k, budget)?;
                let k_u32 =
                    u32::try_from(k).map_err(|_| UmapError::Overflow("transform query width"))?;
                let candidates = u32::try_from(budget.candidates)
                    .map_err(|_| UmapError::Overflow("transform candidate budget"))?;
                let visits = u32::try_from(budget.visits)
                    .map_err(|_| UmapError::Overflow("transform visit budget"))?;
                if matches!(
                    self.backend,
                    crate::standard_umap::BackendChoice::Cpu
                        | crate::standard_umap::BackendChoice::Cuda { .. }
                ) {
                    u32::try_from(q).map_err(|_| UmapError::Overflow("transform query rows"))?;
                    let fit_graph = self.fit_graph.as_ref().ok_or_else(|| {
                        UmapError::Numerical("validated fit graph unavailable".into())
                    })?;
                    approximate_query_scratch_layout(
                        q,
                        budget.candidates,
                        budget.visits,
                        fit_graph.0.dims()[1],
                    )?;
                    query_output_allocation(q, k)?;

                    let (indices, distances, stats) =
                        <Dispatch as StandardUmapOps>::approximate_query_prevalidated(
                            Tensor::<2>::from_data(
                                TensorData::new(data, [q, query_features]),
                                &device,
                            )
                            .into_dispatch(),
                            self.training_data.clone().into_dispatch(),
                            fit_graph.clone_dispatch(),
                            k_u32,
                            candidates,
                            visits,
                            self.config.transform_seed,
                            0,
                        );
                    let stats = Tensor::from_dispatch(stats);
                    if let Some(output) = diagnostic_stats.as_deref_mut() {
                        read_query_stats(&stats, output)?;
                    }
                    check_query_stats(stats)?;
                    (
                        Tensor::<2, Int>::from_dispatch(indices),
                        Tensor::<2>::from_dispatch(distances),
                    )
                } else {
                    let mut all_indices = Vec::new();
                    let mut all_distances = Vec::new();
                    for start in (0..q).step_by(APPROXIMATE_QUERY_BLOCK) {
                        let end = (start + APPROXIMATE_QUERY_BLOCK).min(q);
                        let block = Tensor::<2>::from_data(
                            TensorData::new(
                                data[start * query_features..end * query_features].to_vec(),
                                [end - start, query_features],
                            ),
                            &device,
                        );
                        let graph = self
                            .fit_graph
                            .as_ref()
                            .ok_or_else(|| {
                                UmapError::Numerical("validated fit graph unavailable".into())
                            })?
                            .clone_dispatch();
                        let (indices, distances, stats) =
                            <Dispatch as StandardUmapOps>::approximate_query_prevalidated(
                                block.into_dispatch(),
                                self.training_data.clone().into_dispatch(),
                                graph,
                                k_u32,
                                candidates,
                                visits,
                                self.config.transform_seed,
                                u64::try_from(start)
                                    .map_err(|_| UmapError::Overflow("query row offset"))?,
                            );
                        let stats = Tensor::from_dispatch(stats);
                        if let Some(output) = diagnostic_stats.as_deref_mut() {
                            read_query_stats(&stats, output)?;
                        }
                        check_query_stats(stats)?;
                        all_indices.push(Tensor::<2, Int>::from_dispatch(indices));
                        all_distances.push(Tensor::<2>::from_dispatch(distances));
                    }
                    (Tensor::cat(all_indices, 0), Tensor::cat(all_distances, 0))
                }
            }
            NeighborSearch::Auto => unreachable!("fitted search mode is resolved"),
        };
        if let Some(output) = diagnostic_indices {
            *output = indices.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!(
                    "could not read query neighbor diagnostics: {error}"
                ))
            })?;
        }
        #[cfg(feature = "tracing")]
        drop(_neighbors_span);
        Ok((indices, distances))
    }

    /// Transform graph, pruning and initialization from query kNN, then layout.
    fn transform_from_neighbors(
        &self,
        indices: Tensor<2, Int>,
        distances: Tensor<2>,
        modern_buffers: &[LiveBuffer],
        mut capture: Option<&mut TransformReplay>,
    ) -> Result<Vec<f32>> {
        let q = indices.dims()[0];
        let n = self.training_data.dims()[0];
        let local_connectivity = (self.config.local_connectivity - 1.0).max(0.0);
        #[cfg(feature = "tracing")]
        let _graph_span =
            tracing::info_span!("fuzzy_graph", rows = q, neighbors = indices.dims()[1]).entered();
        let (heads, tails, weights, rho, sigma, directed) =
            <Dispatch as StandardUmapOps>::fuzzy_graph(
                indices.clone().into_dispatch(),
                distances.into_dispatch(),
                local_connectivity,
                1.0,
                self.config.set_op_mix_ratio,
                true,
                u32::try_from(n).map_err(|_| UmapError::Overflow("reference count"))?,
            );
        #[cfg(feature = "tracing")]
        drop(_graph_span);
        let heads: Tensor<1, Int> = Tensor::from_dispatch(heads);
        let tails: Tensor<1, Int> = Tensor::from_dispatch(tails);
        let weights: Tensor<1> = Tensor::from_dispatch(weights);
        let directed: Tensor<2> = Tensor::from_dispatch(directed);
        check_graph_marker(&heads)?;
        if let Some(replay) = capture.as_deref_mut() {
            replay.rho = read_floats(Tensor::<1>::from_dispatch(rho), "replay rho")?;
            replay.sigma = read_floats(Tensor::<1>::from_dispatch(sigma), "replay sigma")?;
            replay.memberships = read_floats(directed.clone(), "replay memberships")?;
            replay.graph_heads = read_ints(heads.clone(), "replay graph heads")?;
            replay.graph_tails = read_ints(tails.clone(), "replay graph tails")?;
            replay.graph_weights = read_floats(weights.clone(), "replay graph weights")?;
        } else {
            drop((rho, sigma));
        }
        let maximum = weights.clone().max_dim(0);
        validate_graph_maximum(
            maximum.clone(),
            "could not inspect transform graph",
            "transform fuzzy graph is empty after positive-edge pruning",
        )?;
        let sums = directed.clone().sum_dim(1).reshape([q, 1]);
        let normalized = directed
            .clone()
            .div(sums.clone().mask_fill(sums.equal_elem(0.0), 1.0));
        let initial = weighted_initialization(
            &self.embedding_tensor,
            &indices,
            &normalized,
            &directed,
            super::query::TRANSFORM_SCRATCH_BUDGET_BYTES,
        )?;
        drop((indices, normalized));
        self.transform_layout(
            initial,
            heads,
            tails,
            weights,
            maximum,
            modern_buffers,
            capture,
        )
    }

    /// Transform schedule and layout from a COO edge stream (query heads,
    /// training tails) and an initial placement. `maximum` is the validated
    /// pre-prune maximum of `weights`.
    #[allow(clippy::too_many_arguments)]
    fn transform_layout(
        &self,
        initial: Tensor<2>,
        heads: Tensor<1, Int>,
        tails: Tensor<1, Int>,
        weights: Tensor<1>,
        maximum: Tensor<1>,
        modern_buffers: &[LiveBuffer],
        capture: Option<&mut TransformReplay>,
    ) -> Result<Vec<f32>> {
        let q = initial.dims()[0];
        let n = self.training_data.dims()[0];
        let device = self.training_data.device();
        let n_epochs = self
            .config
            .n_epochs
            .map(|epochs| epochs / 3)
            .unwrap_or(if q <= 10_000 { 100 } else { 30 });
        let n_epochs_u32 = u32::try_from(n_epochs)
            .map_err(|_| UmapError::Overflow("transform optimization epochs"))?;
        // Unlike fit, upstream transform prunes using the actual epoch count,
        // even when it is at most ten (or zero).
        let (heads, tails, weights, positive, next_positive, negative, next_negative) =
            prepare_schedule(
                heads,
                tails,
                weights,
                maximum,
                n_epochs_u32,
                n_epochs,
                self.config.negative_sample_rate,
            )?;
        if let Some(replay) = capture {
            replay.epochs = n_epochs_u32;
            replay.initial_alpha = self.config.learning_rate / 4.0;
            replay.initial = read_floats(initial.clone(), "replay initial placement")?;
            replay.edge_heads = read_ints(heads.clone(), "replay edge heads")?;
            replay.edge_tails = read_ints(tails.clone(), "replay edge tails")?;
            replay.edge_weights = read_floats(weights.clone(), "replay edge weights")?;
            if self.config.optimizer == super::Optimizer::Compatibility {
                replay.periods = read_f64(positive.clone(), "replay periods")?;
                replay.next_positive = read_f64(next_positive.clone(), "replay positive clocks")?;
                replay.negative_periods = read_f64(negative.clone(), "replay negative periods")?;
                replay.next_negative = read_f64(next_negative.clone(), "replay negative clocks")?;
            } else if let Some(graph) = super::modern_runtime::diagnostic_compact_schedule(
                q,
                n,
                heads.clone(),
                tails.clone(),
                weights.clone(),
                n_epochs_u32,
                &self.config,
                LayoutOperation::Transform,
            )? {
                replay.csr_offsets = graph.offsets;
                replay.csr_indices = graph.indices;
                replay.csr_weights = graph.weights;
                replay.periods = graph.periods;
                replay.negative_periods = graph.negative_periods;
                replay.next_positive = graph.next_positive;
                replay.next_negative = graph.next_negative;
            }
        }
        let transform_seed = self.config.transform_seed;
        let first_coordinates = initial
            .clone()
            .slice_dim(1, 0..1)
            .reshape([q])
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not initialize optimizer RNG: {error}"))
            })?;
        let rng = optimizer_rng_from_first_coordinates(
            first_coordinates.into_iter(),
            q,
            transform_seed,
            &device,
        )?;
        #[cfg(feature = "tracing")]
        let _optimization_span =
            tracing::info_span!("optimization", rows = q, epochs = n_epochs_u32).entered();
        let embedding = if self.config.optimizer == super::Optimizer::Compatibility {
            optimize(
                initial,
                self.embedding_tensor.clone(),
                heads,
                tails,
                weights,
                positive,
                next_positive,
                negative,
                next_negative,
                rng,
                n_epochs_u32,
                self.a,
                self.b,
                self.config.learning_rate / 4.0,
                self.config.repulsion_strength,
                self.config.negative_sample_rate,
                false,
                false,
            )?
        } else {
            drop((positive, next_positive, negative, next_negative, rng));
            super::modern_runtime::optimize(
                initial,
                self.embedding_tensor.clone(),
                heads,
                tails,
                weights,
                n_epochs_u32,
                self.a,
                self.b,
                self.config.learning_rate / 4.0,
                &self.config,
                LayoutOperation::Transform,
                transform_seed,
                self.n_features,
                q,
                modern_buffers,
            )?
        };
        #[cfg(feature = "tracing")]
        drop(_optimization_span);
        embedding
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| UmapError::Numerical(format!("could not read embedding: {error}")))
    }
}

/// Measurement-only stage-replay output (Phase 7 D6). Matrices are row-major.
///
/// Graph fields are empty for the layout entry. Schedule arrays are the ones
/// the selected optimizer consumes: for Compatibility they align with the COO
/// edge stream (pruned zero-weight slots carry `maximum / 1` intervals and are
/// never sampled) and the CSR fields are empty; for modern optimizers they
/// align with the compact CSR built from the positive edges, and all schedule
/// fields are empty at zero epochs.
#[doc(hidden)]
#[derive(Debug, Clone, Default)]
#[cfg_attr(not(feature = "tracing"), allow(dead_code))]
pub struct TransformReplay {
    /// Query neighbor IDs and distances used, `[q, k]`.
    pub neighbor_indices: Vec<i32>,
    pub neighbor_distances: Vec<f32>,
    /// Approximate-search `[peak_candidates, unique_evaluations]`, `[q, 2]`;
    /// empty for exact search and injected neighbors.
    pub query_stats: Vec<i32>,
    /// Per-query smooth-kNN rho and sigma, `[q]`.
    pub rho: Vec<f32>,
    pub sigma: Vec<f32>,
    /// Raw directed memberships per neighbor slot, `[q, k]`.
    pub memberships: Vec<f32>,
    /// Coalesced COO graph before pruning, capacity `2 * q * k`, zero-padded.
    pub graph_heads: Vec<i32>,
    pub graph_tails: Vec<i32>,
    pub graph_weights: Vec<f32>,
    /// Pruned COO edge stream in consumption order (query heads, training tails).
    pub edge_heads: Vec<i32>,
    pub edge_tails: Vec<i32>,
    pub edge_weights: Vec<f32>,
    /// Transform epochs and initial learning rate.
    pub epochs: u32,
    pub initial_alpha: f32,
    /// Modern compact CSR (`offsets` has `q + 1` entries).
    pub csr_offsets: Vec<u32>,
    pub csr_indices: Vec<u32>,
    pub csr_weights: Vec<f32>,
    /// Positive and negative periods and their initial clocks.
    pub periods: Vec<f64>,
    pub negative_periods: Vec<f64>,
    pub next_positive: Vec<f64>,
    pub next_negative: Vec<f64>,
    /// Initial placement and final embedding, `[q, n_components]`.
    pub initial: Vec<f32>,
    pub embedding: Vec<f32>,
}

fn read_floats<const D: usize>(tensor: Tensor<D>, what: &str) -> Result<Vec<f32>> {
    tensor
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|error| UmapError::Numerical(format!("could not read {what}: {error}")))
}

fn read_f64(tensor: Tensor<1>, what: &str) -> Result<Vec<f64>> {
    tensor
        .to_data()
        .try_to_vec::<f64>()
        .map_err(|error| UmapError::Numerical(format!("could not read {what}: {error}")))
}

fn read_ints<const D: usize>(tensor: Tensor<D, Int>, what: &str) -> Result<Vec<i32>> {
    tensor
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|error| UmapError::Numerical(format!("could not read {what}: {error}")))
}

fn read_query_stats(stats: &Tensor<2, Int>, output: &mut Vec<i32>) -> Result<()> {
    let values = read_ints(stats.clone(), "query statistics")?;
    output
        .try_reserve(values.len())
        .map_err(|error| UmapError::Memory(format!("query statistics: {error}")))?;
    output.extend(values);
    Ok(())
}

fn weighted_initialization_chunk_rows(
    query_rows: usize,
    k: usize,
    components: usize,
    scratch_limit_bytes: usize,
) -> Result<usize> {
    // Include weighted-mean buffers, host copies of IDs and memberships,
    // copy IDs and masks on host and device, and the gathered replacement.
    let row_words = k
        .checked_mul(components)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(k))
        .and_then(|value| k.checked_mul(4).and_then(|extra| value.checked_add(extra)))
        .and_then(|value| value.checked_add(6))
        .and_then(|value| {
            components
                .checked_mul(8)
                .and_then(|extra| value.checked_add(extra))
        })
        .ok_or(UmapError::Overflow("transform initialization scratch"))?;
    let row_bytes = row_words
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or(UmapError::Overflow("transform initialization scratch"))?;
    let chunk_rows = scratch_limit_bytes / row_bytes;
    if chunk_rows == 0 {
        return Err(UmapError::InvalidInput(
            "one transform initialization row exceeds the 256 MiB scratch budget".into(),
        ));
    }
    query_rows
        .checked_mul(components)
        .and_then(|value| value.checked_mul(std::mem::size_of::<f32>()))
        .ok_or(UmapError::Overflow("transform initialization output"))?;
    chunk_rows
        .min(query_rows)
        .checked_mul(k)
        .ok_or(UmapError::Overflow("transform initialization scratch"))?;
    Ok(chunk_rows)
}

fn weighted_initialization(
    embedding: &Tensor<2>,
    indices: &Tensor<2, Int>,
    normalized: &Tensor<2>,
    directed: &Tensor<2>,
    scratch_limit_bytes: usize,
) -> Result<Tensor<2>> {
    let [query_rows, k] = indices.dims();
    let components = embedding.dims()[1];
    let chunk_rows =
        weighted_initialization_chunk_rows(query_rows, k, components, scratch_limit_bytes)?;
    let device = embedding.device();
    let mut chunks = Vec::new();
    for start in (0..query_rows).step_by(chunk_rows) {
        let rows = (query_rows - start).min(chunk_rows);
        let end = start + rows;
        let host_ids = read_ints(
            indices.clone().slice_dim(0, start..end),
            "initialization indices",
        )?;
        let host_weights = read_floats(
            directed.clone().slice_dim(0, start..end),
            "initialization memberships",
        )?;
        let mut copy_ids = Vec::new();
        let mut copy_flags = Vec::new();
        copy_ids
            .try_reserve_exact(rows)
            .map_err(|e| UmapError::Memory(format!("initialization copy indices: {e}")))?;
        copy_flags
            .try_reserve_exact(rows)
            .map_err(|e| UmapError::Memory(format!("initialization copy flags: {e}")))?;
        for (ids, weights) in host_ids.chunks_exact(k).zip(host_weights.chunks_exact(k)) {
            let mut selected: Option<i32> = None;
            let mut positive = false;
            for slot in 0..k {
                let id = ids[slot];
                if id < 0 || ids[..slot].contains(&id) {
                    continue;
                }
                let mut weight = 0.0f32;
                for other in 0..k {
                    if ids[other] == id {
                        weight += weights[other];
                    }
                }
                positive |= weight > 0.0;
                if weight == 1.0 && selected.is_none_or(|previous| id < previous) {
                    selected = Some(id);
                }
            }
            if !positive {
                return Err(UmapError::Numerical(
                    "transform initialization row has no positive memberships".into(),
                ));
            }
            copy_ids.push(selected.unwrap_or(0));
            copy_flags.push(i32::from(selected.is_some()));
        }
        drop((host_ids, host_weights));
        let index_count = rows
            .checked_mul(k)
            .ok_or(UmapError::Overflow("transform initialization scratch"))?;
        let ids = indices
            .clone()
            .slice_dim(0, start..end)
            .reshape([index_count]);
        let neighbors = embedding
            .clone()
            .select(0, ids)
            .reshape([rows, k, components]);
        // One broadcast product [rows, k, components]. Slots are then added in
        // left-to-right order from an initial zero, as before; a parallel
        // reduction would change float32 rounding and the RNG seed bits.
        let weights = normalized
            .clone()
            .slice_dim(0, start..end)
            .unsqueeze_dim::<3>(2);
        let products = neighbors.mul(weights);
        let mut sum = Tensor::<2>::zeros([rows, components], &device);
        for slot in 0..k {
            let values = products
                .clone()
                .slice_dim(1, slot..slot + 1)
                .reshape([rows, components]);
            sum = sum.add(values);
        }
        let copy_ids = Tensor::<1, Int>::from_data(TensorData::new(copy_ids, [rows]), &device);
        let copy_mask =
            Tensor::<2, Int>::from_data(TensorData::new(copy_flags, [rows, 1]), &device)
                .equal_elem(1)
                .repeat_dim(1, components);
        let copies = embedding.clone().select(0, copy_ids);
        chunks.push(sum.mask_where(copy_mask, copies));
    }
    Ok(Tensor::cat(chunks, 0))
}

#[doc(hidden)]
pub fn profile_recursive_driver(
    graph: &umap_core::recursive_init::graph::HostCsr,
    data: &[f32],
    features: usize,
    backend: super::BackendChoice,
    seed: u64,
) -> Result<Vec<f32>> {
    let config = UMAPConfig::default();
    let curve = fit_curve(config.min_dist, config.spread)?;
    recursive_driver(
        graph,
        data,
        features,
        curve,
        &config,
        seed,
        &selected_device(backend)?.0,
        &[],
    )
}

#[allow(clippy::too_many_arguments)]
fn recursive_driver(
    graph: &umap_core::recursive_init::graph::HostCsr,
    data: &[f32],
    features: usize,
    curve: (f32, f32),
    config: &UMAPConfig,
    seed: u64,
    device: &Device,
    retained: &[LiveBuffer],
) -> Result<Vec<f32>> {
    use umap_core::{
        modern_layout::{ModernLayoutParams, compact_graph_from_csr},
        recursive_init,
    };
    let d = config.n_components;
    recursive_init::run(
        graph,
        data,
        features,
        d,
        curve,
        &config.recursive_init,
        seed,
        |stage, bytes| admit_recursive_layout(device, stage, bytes, 0),
        |coarse, values, level, level_seed, recursive_owners| {
            let params = ModernLayoutParams::recursive_level(level, coarse.n, level_seed);
            let host_bytes = recursive_compact_bytes(coarse.n, coarse.indices.len())?;
            let coordinate_bytes =
                super::modern_runtime::matrix_allocation_bytes(device, coarse.n, d, 4)?;
            admit_recursive_layout(
                device,
                "recursive coarse layout",
                host_bytes,
                coordinate_bytes,
            )?;
            let compact = compact_graph_from_csr(coarse, level.epochs as usize, 1.0)?;
            let coords = Tensor::<2>::from_data(TensorData::new(values, [coarse.n, d]), device);
            let mut caller = Vec::new();
            caller
                .try_reserve_exact(
                    retained
                        .len()
                        .checked_add(2)
                        .ok_or(UmapError::Overflow("recursive owner table"))?,
                )
                .map_err(|error| UmapError::Memory(error.to_string()))?;
            caller.extend_from_slice(retained);
            caller.push(LiveBuffer {
                name: "recursive feature input",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Both,
                bytes: data
                    .len()
                    .checked_mul(size_of::<f32>())
                    .ok_or(UmapError::Overflow("recursive feature input"))?,
            });
            caller.push(LiveBuffer {
                name: "recursive level owners",
                space: BufferSpace::Host,
                lifetime: BufferLifetime::Both,
                bytes: recursive_owners,
            });
            let optimized = super::modern_runtime::optimize_compact(
                coords.clone(),
                coords,
                compact,
                &params,
                LayoutOperation::Fit,
                features,
                coarse.n,
                &caller,
            )?;
            let readback_bytes = coarse
                .n
                .checked_mul(d)
                .and_then(|count| count.checked_mul(2 * size_of::<f32>()))
                .ok_or(UmapError::Overflow("recursive layout readback"))?;
            admit_recursive_layout(device, "recursive layout readback", readback_bytes, 0)?;
            optimized.to_data().try_to_vec::<f32>().map_err(|error| {
                UmapError::Numerical(format!("recursive layout readback: {error}"))
            })
        },
    )
}

fn recursive_compact_bytes(rows: usize, edges: usize) -> Result<usize> {
    const EDGE_BYTES: usize = size_of::<u32>() + size_of::<f32>() + 4 * size_of::<f64>();
    rows.checked_add(1)
        .and_then(|offsets| offsets.checked_mul(size_of::<u32>()))
        .and_then(|offsets| {
            edges
                .checked_mul(EDGE_BYTES)
                .and_then(|edges| offsets.checked_add(edges))
        })
        .ok_or(UmapError::Overflow("recursive compact graph"))
}

#[cfg(test)]
type RecursiveBudget = (&'static str, u64, u64);
#[cfg(test)]
type RecursiveAdmission = (&'static str, usize, usize);
#[cfg(test)]
thread_local! {
    /// Overrides (host, device) headroom for one named recursive admission.
    static RECURSIVE_BUDGET: std::cell::Cell<Option<RecursiveBudget>> =
        const { std::cell::Cell::new(None) };
    /// Records every recursive admission as (stage, host bytes, device bytes).
    static RECURSIVE_ADMISSIONS: std::cell::RefCell<Option<Vec<RecursiveAdmission>>> =
        const { std::cell::RefCell::new(None) };
}

fn admit_recursive_layout(
    device: &Device,
    stage: &'static str,
    host_bytes: usize,
    device_bytes: usize,
) -> Result<()> {
    super::modern_runtime::check_injected_name(stage)?;
    #[cfg(test)]
    RECURSIVE_ADMISSIONS.with(|log| {
        if let Some(log) = log.borrow_mut().as_mut() {
            log.push((stage, host_bytes, device_bytes));
        }
    });
    let map_error = super::backend_choice::map_backend_error;
    let cpu = *device == Device::cpu();
    #[cfg(test)]
    let budget = RECURSIVE_BUDGET
        .with(|slot| slot.get())
        .filter(|(name, _, _)| *name == stage)
        .map(|(_, host, headroom)| (host, headroom));
    #[cfg(not(test))]
    let budget: Option<(u64, u64)> = None;
    let (host_budget, device_budget) = if let Some((host, headroom)) = budget {
        // CPU host and runtime memory share one budget.
        (if cpu { headroom } else { host }, headroom)
    } else {
        let device_budget = cuscan_common::resolve_memory_budget(device, None)
            .map_err(map_error)?
            .usable_bytes(0);
        let host_budget = if cpu {
            device_budget
        } else {
            cuscan_common::host_usable_memory().map_err(map_error)?
        };
        (host_budget, device_budget)
    };
    let host_needed = if cpu {
        host_bytes
            .checked_add(device_bytes)
            .ok_or(UmapError::Overflow("recursive layout staging"))?
    } else {
        host_bytes
    };
    if host_needed as u64 > host_budget || (!cpu && device_bytes as u64 > device_budget) {
        return Err(UmapError::Capacity(
            "recursive layout staging exceeds available memory".into(),
        ));
    }
    Ok(())
}

fn selected_device(
    choice: crate::standard_umap::BackendChoice,
) -> Result<(Device, crate::standard_umap::BackendChoice)> {
    cuscan_common::selected_device(choice).map_err(super::backend_choice::map_backend_error)
}

fn matrix_tensor(data: &[Vec<f32>], device: &Device) -> Tensor<2> {
    let columns = data.first().map_or(0, Vec::len);
    let values = data.iter().flat_map(|row| row.iter().copied()).collect();
    Tensor::from_data(TensorData::new(values, [data.len(), columns]), device)
}

fn check_graph_marker(heads: &Tensor<1, Int>) -> Result<()> {
    let marker = heads
        .clone()
        .slice(0..1)
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|error| {
            UmapError::Numerical(format!("could not inspect fuzzy graph marker: {error}"))
        })?;
    if marker.first() == Some(&-2) {
        Err(UmapError::Numerical(
            "fuzzy graph contains missing or invalid neighbor indices (for example, a squared feature distance overflowed float32)".into(),
        ))
    } else {
        Ok(())
    }
}

fn validate_layout_work(edge_capacity: usize, negative_sample_rate: usize) -> Result<()> {
    let negative_events = if negative_sample_rate == 0 {
        0
    } else {
        let bound = (2.0 * negative_sample_rate as f64 + 1.0).ceil();
        if !bound.is_finite() || bound > i32::MAX as f64 {
            return Err(UmapError::InvalidConfig(
                "negative_sample_rate exceeds the optimizer limit".into(),
            ));
        }
        bound as usize
    };
    edge_capacity
        .checked_mul(negative_events)
        .ok_or(UmapError::Overflow("negative sampling work"))?;
    Ok(())
}

/// Rejects approximate-query stats containing the `-1` invalid-neighbor marker.
/// The reduction runs on the device; only one Bool scalar is read back.
fn check_query_stats(stats: Tensor<2, Int>) -> Result<()> {
    let has_invalid = stats
        .equal_scalar(-1)
        .any()
        .try_into_scalar::<bool>()
        .map_err(|error| {
            UmapError::Numerical(format!(
                "could not inspect approximate-query stats: {error}"
            ))
        })?;
    if has_invalid {
        return Err(UmapError::Numerical(
            "approximate query encountered an invalid fitted neighbor graph".into(),
        ));
    }
    Ok(())
}

/// Reads and validates a singleton maximum at the caller's validation point.
fn validate_graph_maximum(maximum: Tensor<1>, context: &str, empty_message: &str) -> Result<()> {
    let value = maximum
        .try_into_scalar::<f32>()
        .map_err(|error| UmapError::Numerical(format!("{context}: {error}")))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(UmapError::InvalidInput(empty_message.into()));
    }
    Ok(())
}

#[allow(clippy::type_complexity)]
/// `maximum` is the pre-prune maximum of `weights`. Pruning zeroes weights below
/// `maximum / divisor`. Fit uses a positive fallback divisor; transform uses its
/// actual epoch count, including zero (which prunes every edge). Callers validate
/// the pre-prune maximum once.
#[allow(clippy::too_many_arguments)]
fn prepare_schedule(
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    maximum: Tensor<1>,
    n_epochs: u32,
    default_epochs: usize,
    negative_sample_rate: usize,
) -> Result<(
    Tensor<1, Int>,
    Tensor<1, Int>,
    Tensor<1>,
    Tensor<1>,
    Tensor<1>,
    Tensor<1>,
    Tensor<1>,
)> {
    let divisor = if n_epochs > 10 {
        n_epochs as f64
    } else {
        default_epochs as f64
    };
    let threshold = maximum.clone().div_scalar(divisor);
    let pruned = weights
        .clone()
        .mask_fill(weights.clone().lower(threshold), 0.0);
    let safe = pruned
        .clone()
        .mask_fill(pruned.clone().equal_elem(0.0), 1.0);
    let intervals = maximum
        .cast(burn::tensor::FloatDType::F64)
        .div(safe.cast(burn::tensor::FloatDType::F64));
    let positive = intervals.clone();
    let next_positive = intervals.clone();
    let negative = intervals.div_scalar(negative_sample_rate.max(1) as f64);
    let next_negative = negative.clone();
    Ok((
        heads,
        tails,
        pruned,
        positive,
        next_positive,
        negative,
        next_negative,
    ))
}

fn optimizer_rng_from_first_coordinates(
    values: impl Iterator<Item = f32>,
    rows: usize,
    seed: u64,
    device: &Device,
) -> Result<Tensor<2, Int>> {
    let mut random = StdRng::seed_from_u64(seed);
    let base = [
        random.random_range(i32::MIN..i32::MAX) as i64,
        random.random_range(i32::MIN..i32::MAX) as i64,
        random.random_range(i32::MIN..i32::MAX) as i64,
    ];
    let count = rows
        .checked_mul(3)
        .ok_or(UmapError::Overflow("optimizer RNG state"))?;
    let mut state = Vec::new();
    state
        .try_reserve_exact(count)
        .map_err(|error| UmapError::Memory(format!("optimizer RNG state: {error}")))?;
    for value in values {
        let offset = f64::from(value).to_bits() as i64;
        for value in base {
            state.push(value.wrapping_add(offset));
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(state, [rows, 3]),
        (device, DType::I64),
    ))
}

// The frozen transform oracle retains the original full-embedding readback.
#[cfg(test)]
fn optimizer_rng(embedding: &Tensor<2>, seed: u64, device: &Device) -> Result<Tensor<2, Int>> {
    let values = embedding.to_data().try_to_vec::<f32>().map_err(|error| {
        UmapError::Numerical(format!("could not initialize optimizer RNG: {error}"))
    })?;
    let n = embedding.dims()[0];
    let mut random = StdRng::seed_from_u64(seed);
    let base = [
        random.random_range(i32::MIN..i32::MAX) as i64,
        random.random_range(i32::MIN..i32::MAX) as i64,
        random.random_range(i32::MIN..i32::MAX) as i64,
    ];
    let mut state = Vec::with_capacity(
        n.checked_mul(3)
            .ok_or(UmapError::Overflow("optimizer RNG state"))?,
    );
    for row in 0..n {
        let offset = f64::from(values[row * embedding.dims()[1]]).to_bits() as i64;
        for value in base {
            state.push(value.wrapping_add(offset));
        }
    }
    Ok(Tensor::from_data(
        TensorData::new(state, [n, 3]),
        (device, DType::I64),
    ))
}

#[allow(clippy::too_many_arguments)]
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        skip_all,
        fields(epochs = n_epochs, edges = heads.dims()[0], dimensions = head.dims()[1])
    )
)]
fn optimize(
    mut head: Tensor<2>,
    mut tail: Tensor<2>,
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
    positive: Tensor<1>,
    mut next_positive: Tensor<1>,
    negative: Tensor<1>,
    mut next_negative: Tensor<1>,
    mut rng: Tensor<2, Int>,
    n_epochs: u32,
    a: f32,
    b: f32,
    learning_rate: f32,
    gamma: f32,
    negative_sample_rate: usize,
    move_other: bool,
    shared: bool,
) -> Result<Tensor<2>> {
    // The device op updates clocks and RNG state in place; give each mutable
    // clock its own buffer so clocks cloned from their intervals cannot
    // observe each other's writes.
    next_positive = next_positive.add_scalar(0.0);
    next_negative = next_negative.add_scalar(0.0);
    let device = head.device();
    let (head_offsets, head_edges) = edge_schedule(
        &heads,
        &tails,
        &weights,
        head.dims()[0],
        tail.dims()[0],
        shared,
        &device,
    )?;
    let mut alpha = learning_rate;
    for epoch in 0..n_epochs {
        #[cfg(feature = "tracing")]
        let _epoch_span = tracing::trace_span!("layout_epoch", epoch, alpha).entered();
        let (new_head, new_tail, new_next_positive, new_next_negative, new_rng) =
            <Dispatch as StandardUmapOps>::optimize_epoch(
                head.clone().into_dispatch(),
                tail.clone().into_dispatch(),
                heads.clone().into_dispatch(),
                tails.clone().into_dispatch(),
                weights.clone().into_dispatch(),
                positive.clone().into_dispatch(),
                next_positive.clone().into_dispatch(),
                negative.clone().into_dispatch(),
                next_negative.clone().into_dispatch(),
                rng.clone().into_dispatch(),
                head_offsets.clone().into_dispatch(),
                head_edges.clone().into_dispatch(),
                epoch,
                a,
                b,
                gamma,
                alpha,
                negative_sample_rate as f32,
                move_other,
                shared,
            );
        head = Tensor::from_dispatch(new_head);
        tail = Tensor::from_dispatch(new_tail);
        next_positive = Tensor::from_dispatch(new_next_positive);
        next_negative = Tensor::from_dispatch(new_next_negative);
        rng = Tensor::from_dispatch(new_rng);
        alpha = learning_rate * (1.0 - epoch as f32 / n_epochs.max(1) as f32);
    }
    // Atomic position bindings are read-only to CubeCL's failure tracker. The
    // vertex kernel also writes RNG, so retain and check every output handle.
    let head_dispatch = head.clone().into_dispatch();
    let positive_dispatch = next_positive.into_dispatch();
    let negative_dispatch = next_negative.into_dispatch();
    let rng_dispatch = rng.into_dispatch();
    use burn::backend::{BackendTensor, DispatchTensorKind};
    let (
        DispatchTensorKind::Cube(BackendTensor::Float(head_raw)),
        DispatchTensorKind::Cube(BackendTensor::Float(positive_raw)),
        DispatchTensorKind::Cube(BackendTensor::Float(negative_raw)),
        DispatchTensorKind::Cube(BackendTensor::Int(rng_raw)),
    ) = (
        head_dispatch.kind,
        positive_dispatch.kind,
        negative_dispatch.kind,
        rng_dispatch.kind,
    )
    else {
        return Err(UmapError::Numerical("unsupported layout backend".into()));
    };
    let head_raw = head_raw
        .client
        .clone()
        .resolve_tensor_float::<burn_cubecl::CubeBackend>(head_raw);
    let positive_raw = positive_raw
        .client
        .clone()
        .resolve_tensor_float::<burn_cubecl::CubeBackend>(positive_raw);
    let negative_raw = negative_raw
        .client
        .clone()
        .resolve_tensor_float::<burn_cubecl::CubeBackend>(negative_raw);
    let rng_raw = rng_raw
        .client
        .clone()
        .resolve_tensor_int::<burn_cubecl::CubeBackend>(rng_raw);
    burn::cubecl::future::block_on(head_raw.client.sync_buffers([
        &head_raw.handle,
        &positive_raw.handle,
        &negative_raw.handle,
        &rng_raw.handle,
    ]))
    .map_err(|error| UmapError::Numerical(format!("layout execution failed: {error}")))?;
    Ok(head)
}

/// Build stable head CSR for positive-weight edges.
fn edge_schedule(
    heads: &Tensor<1, Int>,
    tails: &Tensor<1, Int>,
    weights: &Tensor<1>,
    head_rows: usize,
    tail_rows: usize,
    shared: bool,
    device: &Device,
) -> Result<(Tensor<1, Int>, Tensor<1, Int>)> {
    if heads.dims()[0] != weights.dims()[0] || heads.dims()[0] != tails.dims()[0] {
        return Err(UmapError::Numerical(
            "layout edge arrays have different lengths".into(),
        ));
    }
    if !shared {
        let head_rows_u32 =
            u32::try_from(head_rows).map_err(|_| UmapError::Overflow("layout head offsets"))?;
        i32::try_from(head_rows).map_err(|_| UmapError::Overflow("layout head offsets"))?;
        let edges = heads.dims()[0];
        i32::try_from(edges).map_err(|_| UmapError::Overflow("layout edge id"))?;
        // Include both scratch and fixed-capacity outputs in this conservative bound.
        let words = head_rows
            .checked_mul(3)
            .and_then(|value| value.checked_add(3))
            .and_then(|value| value.checked_add(edges.max(1)))
            .ok_or(UmapError::Overflow("transform CSR allocation"))?;
        let bytes = words
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or(UmapError::Overflow("transform CSR allocation"))?;
        if bytes > super::query::TRANSFORM_SCRATCH_BUDGET_BYTES {
            return Err(UmapError::InvalidInput(
                "transform CSR allocation exceeds the 256 MiB scratch budget".into(),
            ));
        }
        let (offsets, ids) = <Dispatch as StandardUmapOps>::head_csr(
            heads.clone().into_dispatch(),
            weights.clone().into_dispatch(),
            head_rows_u32,
        );
        let offsets = Tensor::<1, Int>::from_dispatch(offsets);
        let marker = offsets
            .clone()
            .slice(0..1)
            .to_data()
            .try_to_vec::<i32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not read layout CSR marker: {error}"))
            })?[0];
        match marker {
            0 => return Ok((offsets, Tensor::from_dispatch(ids))),
            -2 => {
                return Err(UmapError::Numerical(
                    "layout edge endpoint out of range".into(),
                ));
            }
            -3 => {
                return Err(UmapError::Numerical(
                    "layout edge has invalid weight".into(),
                ));
            }
            _ => return Err(UmapError::Numerical("invalid layout CSR marker".into())),
        }
    }
    let heads = heads
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|error| UmapError::Numerical(format!("could not read layout edges: {error}")))?;
    let weights = weights
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|error| UmapError::Numerical(format!("could not read layout weights: {error}")))?;
    let (offsets, ids) = build_head_csr(&heads, &weights, head_rows)?;
    if shared {
        let tails = tails.to_data().try_to_vec::<i32>().map_err(|error| {
            UmapError::Numerical(format!("could not read layout edges: {error}"))
        })?;
        for (&tail, &weight) in tails.iter().zip(&weights) {
            let tail = usize::try_from(tail)
                .ok()
                .filter(|&row| row < tail_rows)
                .ok_or_else(|| UmapError::Numerical("layout edge endpoint out of range".into()))?;
            if weight > 0.0 && tail >= head_rows {
                return Err(UmapError::Numerical(
                    "shared layout endpoint out of range".into(),
                ));
            }
        }
    }
    let len = ids.len();
    Ok((
        Tensor::from_data(TensorData::new(offsets, [head_rows + 1]), device),
        Tensor::from_data(TensorData::new(ids, [len]), device),
    ))
}

fn build_head_csr(
    heads: &[i32],
    weights: &[f32],
    head_rows: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    if heads.len() != weights.len() {
        return Err(UmapError::Numerical(
            "layout edge arrays have different lengths".into(),
        ));
    }
    let offset_len = head_rows
        .checked_add(1)
        .ok_or(UmapError::Overflow("layout head offsets"))?;
    let mut counts = vec![0usize; head_rows];
    for (&head, &weight) in heads.iter().zip(weights) {
        let row = usize::try_from(head)
            .ok()
            .filter(|&row| row < head_rows)
            .ok_or_else(|| UmapError::Numerical("layout edge endpoint out of range".into()))?;
        if !weight.is_finite() || weight < 0.0 {
            return Err(UmapError::Numerical(
                "layout edge has invalid weight".into(),
            ));
        }
        if weight > 0.0 {
            counts[row] = counts[row]
                .checked_add(1)
                .ok_or(UmapError::Overflow("layout edge count"))?;
        }
    }
    let mut offsets = vec![0i32; offset_len];
    for row in 0..head_rows {
        let next = (offsets[row] as usize)
            .checked_add(counts[row])
            .ok_or(UmapError::Overflow("layout edge count"))?;
        offsets[row + 1] =
            i32::try_from(next).map_err(|_| UmapError::Overflow("layout edge count"))?;
    }
    let total = offsets[head_rows] as usize;
    let mut cursor = offsets[..head_rows].to_vec();
    // Keep the tensor non-empty when there are no positive edges.
    let mut ids = vec![1i32; total.max(1)];
    for (edge, (&head, &weight)) in heads.iter().zip(weights).enumerate() {
        if weight > 0.0 {
            let edge = i32::try_from(edge).map_err(|_| UmapError::Overflow("layout edge id"))?;
            let row = head as usize;
            let slot = &mut cursor[row];
            ids[*slot as usize] = edge;
            *slot += 1;
        }
    }
    Ok((offsets, ids))
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(skip_all, fields(rows = tensor.dims()[0], components = tensor.dims()[1]))
)]
fn tensor_embedding(tensor: &Tensor<2>) -> Result<Embedding> {
    let [rows, columns] = tensor.dims();
    let values = tensor
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|error| UmapError::Numerical(format!("could not read embedding: {error}")))?;
    Ok(values
        .chunks_exact(columns)
        .take(rows)
        .map(|row| row.to_vec())
        .collect())
}

/// Maps ANN preflight failures onto the UMAP error contract. Capacity rejections keep the
/// precise ANN message (quantity, value, limit and shape).
pub(super) fn map_ann_error(error: AnnError) -> UmapError {
    match error {
        AnnError::InvalidInput(message) => UmapError::InvalidInput(message),
        AnnError::Capability(message) => UmapError::BackendUnavailable(message),
        AnnError::Workspace(message) => UmapError::Numerical(message),
        AnnError::Overflow(message) => UmapError::Capacity(message),
        AnnError::Memory { message, .. } => UmapError::Memory(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestration_rejects_injected_allocations_before_fit_and_transform() {
        let data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32, (i as f32).sin()]).collect();
        let original = data.clone();
        let config = UMAPConfig {
            optimizer: super::super::Optimizer::Momentum,
            n_neighbors: 4,
            n_epochs: Some(2),
            init: Init::Random,
            negative_selection_range: 7,
            unique: true,
            ..Default::default()
        };
        let estimator = UMAP::new(
            config,
            super::super::BackendChoice::Cpu,
            NeighborSearch::Exact,
        );
        super::super::modern_runtime::inject_allocation_failure(Some("unique keys"));
        let error = estimator.fit(&data).unwrap_err();
        super::super::modern_runtime::inject_allocation_failure(None);
        assert!(
            matches!(error, UmapError::Capacity(ref message) if message == "injected allocation failure: unique keys")
        );
        assert_eq!(data, original);

        super::super::modern_runtime::inject_allocation_failure(Some("exact block indices"));
        let error = estimator.fit(&data).unwrap_err();
        super::super::modern_runtime::inject_allocation_failure(None);
        assert!(
            matches!(error, UmapError::Capacity(ref message) if message == "injected allocation failure: exact block indices")
        );
        assert_eq!(data, original);

        for name in ["source ranks", "keys_a"] {
            super::super::modern_runtime::inject_allocation_failure(Some(name));
            let error = estimator.fit(&data).unwrap_err();
            super::super::modern_runtime::inject_allocation_failure(None);
            assert!(matches!(error, UmapError::Capacity(ref message)
                if message == &format!("injected allocation failure: {name}")));
            assert_eq!(data, original);
        }

        let fitted = estimator.fit(&data).unwrap();
        let before = fitted.embedding().clone();
        super::super::modern_runtime::inject_allocation_failure(Some("nested query metadata"));
        let error = fitted.transform(&data).unwrap_err();
        super::super::modern_runtime::inject_allocation_failure(None);
        assert!(
            matches!(error, UmapError::Capacity(ref message) if message == "injected allocation failure: nested query metadata")
        );
        assert_eq!(fitted.embedding(), &before);
        assert_eq!(data, original);
        super::super::modern_runtime::inject_allocation_failure(Some("query projections"));
        let error = fitted.transform(&data[..1]).unwrap_err();
        super::super::modern_runtime::inject_allocation_failure(None);
        assert!(matches!(error, UmapError::Capacity(ref message)
            if message == "injected allocation failure: query projections"));
        assert_eq!(fitted.embedding(), &before);
    }

    type Admission = (&'static str, usize, usize);

    fn recursive_backends() -> Vec<super::super::BackendChoice> {
        let mut backends = vec![super::super::BackendChoice::Cpu];
        if crate::cuda_backend::device(0).is_some() {
            backends.push(super::super::BackendChoice::Cuda { device_index: 0 });
        } else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
        }
        backends
    }

    fn recursive_estimator(
        backend: super::super::BackendChoice,
        optimizer: super::super::Optimizer,
        n_neighbors: usize,
        threshold: usize,
    ) -> UMAP {
        UMAP::new(
            UMAPConfig {
                optimizer,
                init: Init::Recursive,
                recursive_init: super::super::RecursiveInitConfig {
                    base_init_threshold: threshold,
                    n_embedding_epochs: 3,
                    ..Default::default()
                },
                n_neighbors,
                n_epochs: Some(3),
                random_state: Some(7),
                ..Default::default()
            },
            backend,
            NeighborSearch::Exact,
        )
    }

    fn recorded<T>(run: impl FnOnce() -> T) -> (T, Vec<Admission>) {
        RECURSIVE_ADMISSIONS.with(|log| *log.borrow_mut() = Some(Vec::new()));
        let result = run();
        let log = RECURSIVE_ADMISSIONS
            .with(|log| log.borrow_mut().take())
            .expect("admission log");
        (result, log)
    }

    fn with_budget<T>(
        stage: &'static str,
        host: usize,
        device: usize,
        run: impl FnOnce() -> T,
    ) -> T {
        RECURSIVE_BUDGET.with(|slot| slot.set(Some((stage, host as u64, device as u64))));
        let result = run();
        RECURSIVE_BUDGET.with(|slot| slot.set(None));
        result
    }

    fn multi_level_data() -> Vec<Vec<f32>> {
        (0..40)
            .map(|i| {
                let t = i as f32 * 0.31;
                vec![
                    t.cos() * (1.0 + 0.05 * t),
                    t.sin() * (1.0 + 0.05 * t),
                    0.1 * t,
                ]
            })
            .collect()
    }

    fn stages(log: &[Admission]) -> Vec<&'static str> {
        let mut names: Vec<_> = log.iter().map(|(name, _, _)| *name).collect();
        names.dedup();
        names
    }

    /// Injects a failure at each named recursive stage observed in a successful
    /// recorded fit and checks the explicit error precedes that admission.
    fn assert_named_injections(estimator: &UMAP, data: &[Vec<f32>], required: &[&'static str]) {
        let original = data.to_vec();
        let (fitted, log) = recorded(|| estimator.fit(data));
        fitted.unwrap();
        let observed = stages(&log);
        for stage in required {
            assert!(
                observed.contains(stage),
                "{stage} missing from {observed:?}"
            );
        }
        for stage in observed {
            super::super::modern_runtime::inject_allocation_failure(Some(stage));
            let (error, log) = recorded(|| estimator.fit(data));
            super::super::modern_runtime::inject_allocation_failure(None);
            let error = error.unwrap_err();
            assert!(
                matches!(error, UmapError::Capacity(ref message)
                    if message == &format!("injected allocation failure: {stage}")),
                "{stage}: {error:?}"
            );
            assert!(log.iter().all(|(name, _, _)| *name != stage), "{stage}");
            assert_eq!(data, original.as_slice());
        }
    }

    #[test]
    fn recursive_named_injections_multi_level() {
        let data = multi_level_data();
        for backend in recursive_backends() {
            assert_named_injections(
                &recursive_estimator(backend, super::super::Optimizer::Adam, 4, 4),
                &data,
                &[
                    "recursive export",
                    "recursive features",
                    "recursive anchor",
                    "recursive labels",
                    "recursive outliers",
                    "recursive remap",
                    "recursive count",
                    "recursive aggregate",
                    "recursive transition",
                    "recursive coarse anchor",
                    "recursive base",
                    "recursive coarse layout",
                    "recursive layout readback",
                    "recursive prolongation",
                    "recursive Procrustes",
                    "recursive finish",
                    "recursive main compact",
                ],
            );
        }
    }

    #[test]
    fn recursive_named_injections_compatibility_rng() {
        let data = multi_level_data();
        for backend in recursive_backends() {
            let estimator =
                recursive_estimator(backend, super::super::Optimizer::Compatibility, 4, 4);
            assert_named_injections(
                &estimator,
                &data,
                &[
                    "recursive export",
                    "recursive anchor",
                    "recursive coarse layout",
                    "recursive finish",
                    "recursive optimizer RNG",
                ],
            );
            // Compatibility keeps the legacy graph and never builds a main compact graph.
            let (_, log) = recorded(|| estimator.fit(&data).unwrap());
            assert!(
                stages(&log)
                    .iter()
                    .all(|name| *name != "recursive main compact")
            );
            let (host, device) = log
                .iter()
                .find(|(name, _, _)| *name == "recursive optimizer RNG")
                .map(|&(_, host, device)| (host, device))
                .unwrap();
            let cpu = backend == super::super::BackendChoice::Cpu;
            let exact = if cpu { host + device } else { host.max(device) };
            with_budget(
                "recursive optimizer RNG",
                host,
                if cpu { exact } else { device },
                || {
                    estimator.fit(&data).unwrap();
                },
            );
            let tight = if cpu {
                (host, exact - 1)
            } else {
                (host, device - 1)
            };
            let error = with_budget("recursive optimizer RNG", tight.0, tight.1, || {
                estimator.fit(&data).unwrap_err()
            });
            assert!(matches!(error, UmapError::Capacity(_)), "{error:?}");
        }
    }

    #[test]
    fn recursive_named_injections_base_only_and_three_rows() {
        for backend in recursive_backends() {
            // n <= T: base-only, no coarse or level work.
            let data: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32, (i as f32).sin()]).collect();
            let estimator = recursive_estimator(backend, super::super::Optimizer::Adam, 3, 8);
            assert_named_injections(&estimator, &data, &["recursive anchor", "recursive base"]);
            let (_, log) = recorded(|| estimator.fit(&data).unwrap());
            assert!(
                stages(&log)
                    .iter()
                    .all(|name| *name != "recursive labels" && *name != "recursive coarse layout")
            );

            // n = 3, T = 2 gives p1 = 2 > floor(n / 2).
            let data = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![2.5, 0.5]];
            let estimator = recursive_estimator(backend, super::super::Optimizer::Adam, 2, 2);
            assert_named_injections(
                &estimator,
                &data,
                &["recursive anchor", "recursive labels", "recursive base"],
            );
            let (_, log) = recorded(|| estimator.fit(&data).unwrap());
            let anchor = log
                .iter()
                .find(|(name, _, _)| *name == "recursive anchor")
                .unwrap()
                .1;
            with_budget("recursive anchor", anchor, anchor, || {
                estimator.fit(&data).unwrap()
            });
            let error = with_budget("recursive anchor", anchor - 1, anchor - 1, || {
                estimator.fit(&data).unwrap_err()
            });
            assert!(matches!(error, UmapError::Capacity(_)), "{error:?}");
        }
    }

    #[test]
    fn recursive_main_compact_tight_budget_includes_u32_offsets() {
        let data = multi_level_data();
        let n = data.len();
        for backend in recursive_backends() {
            let estimator = recursive_estimator(backend, super::super::Optimizer::Adam, 4, 4);
            let (_, log) = recorded(|| estimator.fit(&data).unwrap());
            let main: Vec<_> = log
                .iter()
                .filter(|(name, _, _)| *name == "recursive main compact")
                .collect();
            assert_eq!(main.len(), 1);
            let &(_, bytes, device) = main[0];
            assert_eq!(device, 0);
            let offsets = (n + 1) * size_of::<u32>();
            let edges = bytes - offsets;
            const EDGE_BYTES: usize = size_of::<u32>() + size_of::<f32>() + 4 * size_of::<f64>();
            assert_eq!(edges % EDGE_BYTES, 0);
            assert!(edges > 0);
            assert_eq!(
                bytes,
                recursive_compact_bytes(n, edges / EDGE_BYTES).unwrap()
            );
            with_budget("recursive main compact", bytes, bytes, || {
                estimator.fit(&data).unwrap();
            });
            for budget in [bytes - 1, edges] {
                let error = with_budget("recursive main compact", budget, budget, || {
                    estimator.fit(&data).unwrap_err()
                });
                assert!(
                    matches!(error, UmapError::Capacity(ref message)
                        if message == "recursive layout staging exceeds available memory"),
                    "{budget}: {error:?}"
                );
            }
        }
    }

    #[test]
    fn recursive_coarse_layout_tight_budget_unified_and_separate() {
        let data = multi_level_data();
        for backend in recursive_backends() {
            let estimator = recursive_estimator(backend, super::super::Optimizer::Adam, 4, 4);
            let (_, log) = recorded(|| estimator.fit(&data).unwrap());
            let coarse: Vec<_> = log
                .iter()
                .filter(|(name, _, _)| *name == "recursive coarse layout")
                .map(|&(_, host, device)| (host, device))
                .collect();
            assert!(!coarse.is_empty());
            let stage = "recursive coarse layout";
            if backend == super::super::BackendChoice::Cpu {
                // CPU host staging and device coordinates share one budget.
                let unified = coarse.iter().map(|(h, d)| h + d).max().unwrap();
                with_budget(stage, 0, unified, || estimator.fit(&data).unwrap());
                let error = with_budget(stage, usize::MAX >> 1, unified - 1, || {
                    estimator.fit(&data).unwrap_err()
                });
                assert!(matches!(error, UmapError::Capacity(_)), "{error:?}");
            } else {
                let host = coarse.iter().map(|(h, _)| *h).max().unwrap();
                let device = coarse.iter().map(|(_, d)| *d).max().unwrap();
                // Separate spaces: neither is charged for the other's bytes.
                with_budget(stage, host, device, || estimator.fit(&data).unwrap());
                for (h, d) in [(host - 1, usize::MAX >> 1), (usize::MAX >> 1, device - 1)] {
                    let error = with_budget(stage, h, d, || estimator.fit(&data).unwrap_err());
                    assert!(matches!(error, UmapError::Capacity(_)), "{error:?}");
                }
            }
        }
    }

    #[test]
    fn recursive_empty_coarse_graph_skips_layout_admission() {
        // Two distant pairs coarsen to two isolated labels: the coarse graph is
        // empty, so its hypothetical layout must never be admitted or rejected.
        let graph = umap_core::recursive_init::graph::HostCsr::from_coo(
            4,
            &[0, 1, 2, 3],
            &[2, 3, 0, 1],
            &[1.0; 4],
        )
        .unwrap();
        let data = [0.0, 0.1, 100.0, 100.1];
        let config = UMAPConfig {
            optimizer: umap_core::Optimizer::Compatibility,
            init: Init::Spectral,
            recursive_init: umap_core::RecursiveInitConfig {
                base_init_threshold: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        for backend in recursive_backends() {
            let device = selected_device(backend).unwrap().0;
            let (embedding, log) = recorded(|| {
                with_budget("recursive coarse layout", 0, 0, || {
                    recursive_driver(&graph, &data, 1, (1.0, 1.0), &config, 11, &device, &[])
                })
            });
            let embedding = embedding.unwrap();
            let names = stages(&log);
            assert!(names.contains(&"recursive transition"), "{names:?}");
            assert!(names.contains(&"recursive prolongation"), "{names:?}");
            assert!(!names.contains(&"recursive coarse layout"), "{names:?}");
            assert!(!names.contains(&"recursive layout readback"), "{names:?}");
            assert_eq!(embedding.len(), 4 * config.n_components);
            assert!(embedding.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn modern_reference_windows_allow_raw_ranges_below_cardinality() {
        for optimizer in [
            super::super::Optimizer::Adam,
            super::super::Optimizer::Momentum,
        ] {
            for range in [1, 17, 200_000] {
                let config = UMAPConfig {
                    optimizer,
                    negative_selection_range: range,
                    n_neighbors: 3,
                    n_epochs: Some(3),
                    init: Init::Random,
                    ..Default::default()
                };
                let data: Vec<Vec<f32>> =
                    (0..24).map(|i| vec![i as f32, (i as f32).sin()]).collect();
                let fitted = UMAP::new(
                    config,
                    super::super::BackendChoice::Cpu,
                    NeighborSearch::Exact,
                )
                .fit(&data)
                .unwrap();
                let before = fitted.embedding().clone();
                assert_eq!(fitted.transform(&data).unwrap().len(), data.len());
                assert_eq!(fitted.embedding(), &before);
            }
        }
    }

    #[test]
    fn modern_fit_transform_is_seeded_and_keeps_reference_fixed() {
        let data: Vec<Vec<f32>> = (0..12)
            .map(|i| vec![i as f32 * 0.2, (i as f32 * 0.7).sin()])
            .collect();
        for optimizer in [
            super::super::Optimizer::Adam,
            super::super::Optimizer::Momentum,
        ] {
            for dimensions in [16, 48] {
                let config = UMAPConfig {
                    optimizer,
                    n_components: dimensions,
                    n_neighbors: 4,
                    n_epochs: Some(12),
                    init: Init::Random,
                    random_state: Some(17),
                    negative_selection_range: 12,
                    ..Default::default()
                };
                let mut initial_config = config.clone();
                initial_config.n_epochs = Some(0);
                let initial = UMAP::new(
                    initial_config,
                    super::super::BackendChoice::Cpu,
                    NeighborSearch::Exact,
                )
                .fit(&data)
                .unwrap();
                let estimator = UMAP::new(
                    config,
                    super::super::BackendChoice::Cpu,
                    NeighborSearch::Exact,
                );
                let fitted = estimator.fit(&data).unwrap();
                let repeated = estimator.fit(&data).unwrap();
                assert_eq!(fitted.embedding(), repeated.embedding());
                assert_ne!(fitted.embedding(), initial.embedding());
                assert!(fitted.embedding().iter().flatten().all(|x| x.is_finite()));
                let before = fitted.embedding().clone();
                let transformed = fitted.transform(&data).unwrap();
                assert!(transformed.iter().flatten().all(|x| x.is_finite()));
                assert_eq!(&before, fitted.embedding());
                assert_eq!(transformed, fitted.transform(&data).unwrap());
            }
        }
    }

    #[test]
    fn modern_unique_windows_use_distinct_rows() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![0.0], vec![1.0]];
        for optimizer in [
            super::super::Optimizer::Adam,
            super::super::Optimizer::Momentum,
        ] {
            let config = UMAPConfig {
                optimizer,
                unique: true,
                n_neighbors: 2,
                n_epochs: Some(4),
                init: Init::Random,
                negative_selection_range: 1,
                ..Default::default()
            };
            let fitted = UMAP::new(
                config,
                super::super::BackendChoice::Cpu,
                NeighborSearch::Exact,
            )
            .fit(&data)
            .unwrap();
            assert_eq!(fitted.embedding()[0], fitted.embedding()[3]);
            assert_eq!(fitted.embedding()[1], fitted.embedding()[4]);
        }
    }

    #[test]
    fn modern_persisted_windows_use_reference_cardinality() {
        let data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32, (i as f32).sin()]).collect();
        let fitted = UMAP::new(
            UMAPConfig {
                n_neighbors: 4,
                n_epochs: Some(3),
                init: Init::Random,
                optimizer: crate::standard_umap::Optimizer::Compatibility,
                ..Default::default()
            },
            super::super::BackendChoice::Cpu,
            NeighborSearch::Exact,
        )
        .fit(&data)
        .unwrap();
        let mut state = fitted.export_state().unwrap();
        state.config.optimizer = super::super::Optimizer::Momentum;
        state.config.negative_selection_range = 11;
        let loaded = FittedUMAP::from_state(state, super::super::BackendChoice::Cpu).unwrap();
        assert_eq!(loaded.transform(&data[..1]).unwrap().len(), 1);
    }

    #[cfg(feature = "tracing")]
    #[test]
    fn initialization_downloads_graph_only_when_consumed() {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use tracing_subscriber::{Layer, layer::SubscriberExt};
        struct Exports(Arc<AtomicUsize>);
        impl<S: tracing::Subscriber> Layer<S> for Exports {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event.metadata().target() == "umap::init_export" {
                    self.0.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        let count = Arc::new(AtomicUsize::new(0));
        let subscriber = tracing_subscriber::registry().with(Exports(count.clone()));
        let _guard = tracing::subscriber::set_default(subscriber);
        let data: Vec<Vec<f32>> = (0..16).map(|i| vec![i as f32, (i as f32).sin()]).collect();
        for init in [Init::Random, Init::Spectral] {
            for mode in [
                InitRescale::Isotropic,
                InitRescale::PerAxis,
                InitRescale::GraphEdge,
            ] {
                count.store(0, Ordering::SeqCst);
                UMAP::new(
                    UMAPConfig {
                        init,
                        init_rescale: mode,
                        n_neighbors: 4,
                        optimizer: crate::standard_umap::Optimizer::Compatibility,
                        n_epochs: Some(0),
                        ..Default::default()
                    },
                    super::super::BackendChoice::Cpu,
                    NeighborSearch::Exact,
                )
                .fit(&data)
                .unwrap();
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    usize::from(init == Init::Spectral || mode == InitRescale::GraphEdge)
                );
            }
        }
    }

    #[test]
    fn modern_fit_initialization_matches_core_on_identical_pruned_graph() {
        let mut backends = vec![super::super::BackendChoice::Cpu];
        if crate::cuda_backend::device(0).is_some() {
            backends.push(super::super::BackendChoice::Cuda { device_index: 0 });
        } else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
        }
        let data: Vec<Vec<f32>> = (0..24)
            .map(|i| vec![i as f32 * 0.2, (i as f32 * 0.3).sin()])
            .collect();
        for backend in backends {
            let (device, _) = selected_device(backend).unwrap();
            let input = matrix_tensor(&data, &device);
            let neighbors =
                super::super::neighbors::exact_neighbors(input.clone(), input, 5, true).unwrap();
            let rows = Tensor::<1, Int>::arange(0..24, &device)
                .cast(burn::tensor::IntDType::I32)
                .reshape([24, 1]);
            let indices = Tensor::<2, Int>::cat(vec![rows, neighbors.indices], 1);
            let distances = Tensor::<2>::cat(
                vec![Tensor::<2>::zeros([24, 1], &device), neighbors.distances],
                1,
            );
            let (heads, tails, weights, _, _, _) = <Dispatch as StandardUmapOps>::fuzzy_graph(
                indices.into_dispatch(),
                distances.into_dispatch(),
                1.0,
                1.0,
                1.0,
                false,
                24,
            );
            let heads = Tensor::<1, Int>::from_dispatch(heads);
            let tails = Tensor::<1, Int>::from_dispatch(tails);
            let weights = Tensor::<1>::from_dispatch(weights);
            let maximum = weights.clone().max_dim(0);
            let (heads, tails, weights, ..) =
                prepare_schedule(heads, tails, weights, maximum, 0, 500, 5).unwrap();
            let heads = heads.to_data().try_to_vec::<i32>().unwrap();
            let tails = tails.to_data().try_to_vec::<i32>().unwrap();
            let weights = weights.to_data().try_to_vec::<f32>().unwrap();
            let mut h = Vec::new();
            let mut t = Vec::new();
            let mut w = Vec::new();
            for ((head, tail), weight) in heads.into_iter().zip(tails).zip(weights) {
                if weight > 0.0 {
                    h.push(head as usize);
                    t.push(tail as usize);
                    w.push(weight);
                }
            }
            assert!(
                h.iter()
                    .zip(&t)
                    .zip(h.iter().zip(&t).skip(1))
                    .all(|((a, b), (c, d))| (a, b) < (c, d))
            );
            for init in [Init::Random, Init::Spectral] {
                for mode in [
                    InitRescale::Isotropic,
                    InitRescale::PerAxis,
                    InitRescale::GraphEdge,
                ] {
                    let config = UMAPConfig {
                        init,
                        init_rescale: mode,
                        init_scale_factor: 1.25,
                        optimizer: crate::standard_umap::Optimizer::Compatibility,
                        n_neighbors: 6,
                        n_epochs: Some(0),
                        random_state: Some(19),
                        ..Default::default()
                    };
                    let expected =
                        super::super::initialize(&data, &h, &t, &w, &config, 19).unwrap();
                    let fit = UMAP::new(config, backend, NeighborSearch::Exact)
                        .fit(&data)
                        .unwrap();
                    assert_eq!(
                        fit.embedding()
                            .iter()
                            .flatten()
                            .copied()
                            .collect::<Vec<_>>(),
                        expected,
                        "{backend:?} {init:?} {mode:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn initialization_pruning_preserves_threshold_order_and_caller_aliases() {
        let mut devices = vec![Device::cpu()];
        if let Some(device) = crate::cuda_backend::device(0) {
            devices.push(device);
        } else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
        }
        for device in devices {
            for epochs in [0, 10, 11] {
                // max=11: epochs 0/10 use the default divisor 22; epoch11
                // uses divisor 11. Equality survives; its immediate predecessor
                // must not. Padding remains a device slot, preserving edge IDs.
                let threshold: f32 = if epochs > 10 { 1.0 } else { 0.5 };
                let values = vec![
                    11.0,
                    threshold,
                    f32::from_bits(threshold.to_bits() - 1),
                    0.0,
                ];
                let heads = Tensor::<1, Int>::from_data(
                    TensorData::new(vec![0_i32, 0, 1, 0], [4]),
                    &device,
                );
                let tails = Tensor::<1, Int>::from_data(
                    TensorData::new(vec![1_i32, 2, 2, 0], [4]),
                    &device,
                );
                let weights = Tensor::<1>::from_data(TensorData::new(values.clone(), [4]), &device);
                let aliases = (heads.clone(), tails.clone(), weights.clone());
                let before_heads = aliases.0.to_data().try_to_vec::<i32>().unwrap();
                let before_tails = aliases.1.to_data().try_to_vec::<i32>().unwrap();
                let before_weights = aliases.2.to_data().try_to_vec::<f32>().unwrap();
                let maximum = weights.clone().max_dim(0);
                let (h, t, w, positive, next_positive, negative, next_negative) =
                    prepare_schedule(heads, tails, weights, maximum, epochs, 22, 5).unwrap();
                assert_eq!(h.to_data().try_to_vec::<i32>().unwrap(), before_heads);
                assert_eq!(t.to_data().try_to_vec::<i32>().unwrap(), before_tails);
                assert_eq!(
                    w.to_data().try_to_vec::<f32>().unwrap(),
                    vec![11.0, threshold, 0.0, 0.0]
                );
                assert_eq!(
                    aliases.0.to_data().try_to_vec::<i32>().unwrap(),
                    before_heads
                );
                assert_eq!(
                    aliases.1.to_data().try_to_vec::<i32>().unwrap(),
                    before_tails
                );
                assert_eq!(
                    aliases.2.to_data().try_to_vec::<f32>().unwrap(),
                    before_weights
                );
                let positive = positive.to_data().try_to_vec::<f64>().unwrap();
                assert_eq!(positive, vec![1.0, 11.0 / f64::from(threshold), 11.0, 11.0]);
                assert_eq!(
                    next_positive.to_data().try_to_vec::<f64>().unwrap(),
                    positive
                );
                let negative = negative.to_data().try_to_vec::<f64>().unwrap();
                assert_eq!(
                    negative,
                    positive.iter().map(|v| v / 5.0).collect::<Vec<_>>()
                );
                assert_eq!(
                    next_negative.to_data().try_to_vec::<f64>().unwrap(),
                    negative
                );
            }
        }
    }

    #[test]
    fn ann_capacity_error_keeps_precise_message() {
        let message =
            "ANN input copy buffer elements 4296000000 exceeds u32 addressing maximum 4294967295";
        let error = map_ann_error(AnnError::Overflow(message.into()));
        assert_eq!(error, UmapError::Capacity(message.into()));
        assert_eq!(error.to_string(), message);
    }

    #[test]
    fn overflow_display_does_not_claim_usize() {
        let text = UmapError::Overflow("layout edge id").to_string();
        assert_eq!(text, "layout edge id exceeds the supported integer range");
        assert!(!text.contains("usize"));
    }

    #[test]
    fn layout_launch_failure_surfaces_for_fit_and_transform() {
        use std::sync::atomic::Ordering;
        struct InjectionGuard(u32);
        impl InjectionGuard {
            fn new(alpha: f32) -> Self {
                Self(
                    crate::kernels::INJECT_BAD_LAYOUT_LAUNCH
                        .swap(alpha.to_bits(), Ordering::SeqCst),
                )
            }
        }
        impl Drop for InjectionGuard {
            fn drop(&mut self) {
                crate::kernels::INJECT_BAD_LAYOUT_LAUNCH.store(self.0, Ordering::SeqCst);
            }
        }
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        let data: Vec<Vec<f32>> = (0..12).map(|i| vec![i as f32, (i as f32).sin()]).collect();
        let config = UMAPConfig {
            n_neighbors: 4,
            n_epochs: Some(3),
            init: Init::Random,
            random_state: Some(31),
            learning_rate: 0.987_654_3,
            optimizer: crate::standard_umap::Optimizer::Compatibility,
            ..Default::default()
        };
        let estimator = UMAP::new(
            config,
            crate::standard_umap::BackendChoice::Cuda { device_index: 0 },
            NeighborSearch::Exact,
        );
        let fitted = estimator.fit(&data).unwrap();
        let guard = InjectionGuard::new(0.987_654_3 / 4.0);
        let transform = fitted.transform(&[vec![0.1, 0.0], vec![5.1, 0.0]]);
        drop(guard);
        let guard = InjectionGuard::new(0.987_654_3);
        let fit = estimator.fit(&data);
        drop(guard);
        assert!(
            matches!(transform, Err(UmapError::Numerical(ref message)) if message.contains("layout execution failed")),
            "transform did not report a layout failure: {transform:?}"
        );
        assert!(
            matches!(fit, Err(UmapError::Numerical(ref message)) if message.contains("layout execution failed")),
            "fit did not report a layout failure"
        );
    }

    #[test]
    fn transform_flat_preserves_validation_errors() {
        let backend = crate::standard_umap::BackendChoice::Cpu;
        let data = (0..16)
            .map(|row| vec![row as f32, row as f32 * 0.25])
            .collect::<Vec<_>>();
        let fitted = UMAP::new(
            UMAPConfig {
                n_neighbors: 3,
                n_epochs: Some(0),
                init: Init::Random,
                optimizer: crate::standard_umap::Optimizer::Compatibility,
                ..Default::default()
            },
            backend,
            NeighborSearch::Exact,
        )
        .fit(&data)
        .unwrap();
        for (nested, flat, rows, features) in [
            (vec![], vec![], 0, 2),
            (vec![vec![]], vec![], 1, 0),
            (vec![vec![1.0]], vec![1.0], 1, 1),
            (vec![vec![1.0, f32::NAN]], vec![1.0, f32::NAN], 1, 2),
            (
                vec![vec![1.0, 2.0], vec![f32::INFINITY, 0.0]],
                vec![1.0, 2.0, f32::INFINITY, 0.0],
                2,
                2,
            ),
        ] {
            assert_eq!(
                fitted.transform_flat(flat, rows, features).unwrap_err(),
                fitted.transform(&nested).unwrap_err()
            );
        }
        assert!(matches!(
            fitted.transform_flat(vec![1.0], 1, 2),
            Err(UmapError::InvalidInput(_))
        ));
        assert!(matches!(
            fitted.transform_flat(vec![], usize::MAX, 2),
            Err(UmapError::Overflow("transform input shape"))
        ));
    }

    fn weighted_initialization_inputs(device: &Device) -> (Tensor<2>, Tensor<2, Int>, Tensor<2>) {
        let embedding = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    -0.125, 1.25, 2.5, -3.75, 4.125, 5.5, -6.25, 7.75, 8.5, -9.125,
                ],
                [5, 2],
            ),
            device,
        );
        let indices = Tensor::<2, Int>::from_data(
            TensorData::new(
                vec![
                    0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0,
                ],
                [7, 3],
            ),
            device,
        );
        let normalized = Tensor::<2>::from_data(
            TensorData::new(
                vec![
                    0.125, 0.25, 0.625, 0.5, 0.125, 0.375, 0.25, 0.625, 0.125, 0.375, 0.25, 0.375,
                    0.625, 0.125, 0.25, 0.125, 0.625, 0.25, 0.5, 0.25, 0.25,
                ],
                [7, 3],
            ),
            device,
        );
        (embedding, indices, normalized)
    }

    fn original_weighted_initialization(
        embedding: &Tensor<2>,
        indices: &Tensor<2, Int>,
        normalized: &Tensor<2>,
    ) -> Tensor<2> {
        let [query_rows, k] = indices.dims();
        let components = embedding.dims()[1];
        let mut initial = Tensor::<2>::zeros([query_rows, components], &embedding.device());
        for slot in 0..k {
            let ids = indices
                .clone()
                .slice_dim(1, slot..slot + 1)
                .reshape([query_rows]);
            let neighbors = embedding.clone().select(0, ids);
            let weight = normalized
                .clone()
                .slice_dim(1, slot..slot + 1)
                .repeat_dim(1, components);
            initial = initial.add(neighbors.mul(weight));
        }
        initial
    }

    fn assert_chunked_weighted_initialization_matches_original(
        device: &Device,
        scratch_limit_bytes: usize,
    ) {
        let (embedding, indices, normalized) = weighted_initialization_inputs(device);
        let expected = original_weighted_initialization(&embedding, &indices, &normalized);
        let actual = weighted_initialization(
            &embedding,
            &indices,
            &normalized,
            &normalized,
            scratch_limit_bytes,
        )
        .unwrap();
        let expected_values = expected.to_data().try_to_vec::<f32>().unwrap();
        let actual_values = actual.to_data().try_to_vec::<f32>().unwrap();
        assert_eq!(actual_values.len(), expected_values.len());
        // The broadcast product may round differently from the per-slot repeat;
        // require finite values within a float32 tolerance and report drift.
        let mut max_abs = 0.0f32;
        let mut sum_abs = 0.0f64;
        let mut bit_mismatches = 0usize;
        for (&actual, &expected) in actual_values.iter().zip(&expected_values) {
            assert!(
                actual.is_finite(),
                "non-finite weighted initialization value"
            );
            let delta = (actual - expected).abs();
            max_abs = max_abs.max(delta);
            sum_abs += f64::from(delta);
            if actual.to_bits() != expected.to_bits() {
                bit_mismatches += 1;
            }
            assert!(
                delta <= 1e-5 + 1e-5 * expected.abs(),
                "weighted initialization drift {delta} exceeds tolerance for expected {expected}"
            );
        }
        eprintln!(
            "weighted_initialization drift: max_abs={max_abs} mean_abs={} bit_mismatches={bit_mismatches}",
            sum_abs / actual_values.len() as f64
        );

        let query_rows = indices.dims()[0];
        let components = embedding.dims()[1];
        let expected_first = expected_values
            .iter()
            .step_by(components)
            .copied()
            .collect::<Vec<_>>();
        let actual_first = actual_values
            .iter()
            .step_by(components)
            .copied()
            .collect::<Vec<_>>();
        for (actual, expected) in actual_first.iter().zip(&expected_first) {
            assert!((actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs());
        }
        let expected_rng = optimizer_rng_from_first_coordinates(
            expected_values.iter().step_by(components).copied(),
            query_rows,
            42,
            device,
        )
        .unwrap()
        .to_data()
        .try_to_vec::<i64>()
        .unwrap();
        let actual_rng = optimizer_rng_from_first_coordinates(
            actual_values.iter().step_by(components).copied(),
            query_rows,
            42,
            device,
        )
        .unwrap()
        .to_data()
        .try_to_vec::<i64>()
        .unwrap();
        // RNG seeds are derived from the first coordinate's float bits, so a
        // float32 rounding change may legitimately alter them. Report divergence
        // instead of requiring equality.
        let rng_mismatches = actual_rng
            .iter()
            .zip(&expected_rng)
            .filter(|(a, e)| a != e)
            .count();
        eprintln!(
            "weighted_initialization rng mismatches: {rng_mismatches}/{}",
            expected_rng.len()
        );
        assert_eq!(actual_rng.len(), expected_rng.len());
    }

    fn estimator(search: NeighborSearch) -> UMAP {
        UMAP::new(
            UMAPConfig {
                optimizer: crate::standard_umap::Optimizer::Compatibility,
                init: Init::Spectral,
                ..UMAPConfig::default()
            },
            crate::standard_umap::BackendChoice::Cpu,
            search,
        )
    }

    #[test]
    fn resolve_search_uses_exact_up_to_the_auto_cutoff() {
        let estimator = estimator(NeighborSearch::Auto);
        assert_eq!(estimator.resolve_search(0), NeighborSearch::Exact);
        assert_eq!(
            estimator.resolve_search(AUTO_EXACT_MAX_ROWS),
            NeighborSearch::Exact
        );
    }

    #[test]
    fn resolve_search_uses_approximate_above_the_auto_cutoff() {
        let estimator = estimator(NeighborSearch::Auto);
        assert_eq!(
            estimator.resolve_search(AUTO_EXACT_MAX_ROWS + 1),
            NeighborSearch::Approximate
        );
        assert_eq!(
            estimator.resolve_search(usize::MAX),
            NeighborSearch::Approximate
        );
    }

    #[test]
    fn resolve_search_passes_explicit_modes_through() {
        for mode in [NeighborSearch::Exact, NeighborSearch::Approximate] {
            let estimator = estimator(mode);
            assert_eq!(estimator.resolve_search(0), mode);
            assert_eq!(estimator.resolve_search(AUTO_EXACT_MAX_ROWS + 1), mode);
        }
    }

    #[test]
    fn head_csr_preserves_original_edge_order_within_each_head() {
        let (offsets, edge_ids) =
            build_head_csr(&[2, 0, 2, 1, 0, 2], &[1.0, 0.0, 2.0, 1.0, 3.0, 0.0], 4).unwrap();
        assert_eq!(offsets, vec![0, 1, 2, 4, 4]);
        assert_eq!(edge_ids, vec![4, 3, 0, 2]);
    }

    #[test]
    fn first_coordinate_rng_matches_original_full_embedding_bits() {
        let device = Device::cpu();
        let values = vec![-0.0f32, 99.0, 1.25, -17.0, -3.5, 0.0];
        let embedding = Tensor::<2>::from_data(TensorData::new(values.clone(), [3, 2]), &device);
        for seed in [0, 42, u64::MAX] {
            let original = optimizer_rng(&embedding, seed, &device)
                .unwrap()
                .to_data()
                .try_to_vec::<i64>()
                .unwrap();
            let reduced = optimizer_rng_from_first_coordinates(
                values.as_chunks::<2>().0.iter().map(|row| row[0]),
                3,
                seed,
                &device,
            )
            .unwrap()
            .to_data()
            .try_to_vec::<i64>()
            .unwrap();
            assert_eq!(reduced, original);
            assert_ne!(&reduced[0..3], &reduced[3..6]);
        }
    }

    #[test]
    fn weighted_initialization_scratch_accounting_checks_limits_and_overflow() {
        const ROW_BYTES: usize = (2 * 3 * 2 + 3 * 5 + 6 + 8 * 2) * std::mem::size_of::<f32>();
        assert_eq!(
            weighted_initialization_chunk_rows(7, 3, 2, ROW_BYTES),
            Ok(1)
        );
        assert_eq!(
            weighted_initialization_chunk_rows(7, 3, 2, ROW_BYTES * 2),
            Ok(2)
        );
        assert_eq!(
            weighted_initialization_chunk_rows(7, 3, 2, ROW_BYTES - 1),
            Err(UmapError::InvalidInput(
                "one transform initialization row exceeds the 256 MiB scratch budget".into()
            ))
        );
        assert_eq!(
            weighted_initialization_chunk_rows(1, usize::MAX, 2, usize::MAX),
            Err(UmapError::Overflow("transform initialization scratch"))
        );
        assert_eq!(
            weighted_initialization_chunk_rows(usize::MAX, 1, 1, 128),
            Err(UmapError::Overflow("transform initialization output"))
        );
        const TWO_NEIGHBOR_ROW_BYTES: usize =
            (2 * 2 * 8 + 5 * 2 + 6 + 8 * 8) * std::mem::size_of::<f32>();
        assert_eq!(
            weighted_initialization_chunk_rows(7, 2, 8, TWO_NEIGHBOR_ROW_BYTES * 2),
            Ok(2)
        );
        assert_eq!(
            weighted_initialization_chunk_rows(7, 2, 8, TWO_NEIGHBOR_ROW_BYTES - 1),
            Err(UmapError::InvalidInput(
                "one transform initialization row exceeds the 256 MiB scratch budget".into()
            ))
        );
    }

    #[test]
    fn chunked_weighted_initialization_matches_original_on_cpu() {
        assert_chunked_weighted_initialization_matches_original(&Device::cpu(), 400);
    }

    #[test]
    fn cuda_chunked_weighted_initialization_matches_original_with_small_limit() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        // Q=7 and two rows per chunk leaves a one-row final chunk.
        assert_eq!(weighted_initialization_chunk_rows(7, 3, 2, 400), Ok(2));
        assert_chunked_weighted_initialization_matches_original(&device, 400);
    }

    fn assert_copy_initialization(device: &Device) {
        let embedding = Tensor::<2>::from_data(
            TensorData::new(vec![10.0, 20.0, 30.0, 40.0], [4, 1]),
            device,
        );
        // Unsorted IDs: the first row has two exact memberships, the second
        // coalesces duplicate slots to one, and the third is a weighted mean.
        let indices = Tensor::<2, Int>::from_data(
            TensorData::new(vec![3, 1, 0, 2, 2, 0, 3, 1, 0], [3, 3]),
            device,
        );
        let raw = Tensor::<2>::from_data(
            TensorData::new(
                vec![1.0, 1.0, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5, 0.25],
                [3, 3],
            ),
            device,
        );
        let sums = raw.clone().sum_dim(1);
        let normalized = raw.clone().div(sums);
        let actual = weighted_initialization(&embedding, &indices, &normalized, &raw, 1024)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(actual[0], 20.0);
        assert_eq!(actual[1], 30.0);
        // Upstream S3 uses rtol=1e-4 and atol=1e-5.
        assert!((actual[2] - 22.5).abs() <= 1e-5 + 1e-4 * 22.5);

        let empty = Tensor::<2>::zeros([1, 3], device);
        let one_row = indices.slice_dim(0, 0..1);
        assert!(matches!(
            weighted_initialization(&embedding, &one_row, &empty, &empty, 1024),
            Err(UmapError::Numerical(_))
        ));
    }

    #[test]
    fn copy_initialization_on_cpu() {
        assert_copy_initialization(&Device::cpu());
    }

    #[test]
    fn copy_initialization_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        assert_copy_initialization(&device);
    }

    fn singleton_maximum(value: f32, device: &Device) -> Tensor<1> {
        Tensor::<1>::from_data(TensorData::new(vec![value], [1]), device)
    }

    fn assert_positive_maximum_rejected(device: &Device) {
        for (context, message) in [
            (
                "could not inspect fuzzy graph weights",
                "fuzzy graph is empty after positive-edge pruning",
            ),
            (
                "could not inspect transform graph",
                "transform fuzzy graph is empty after positive-edge pruning",
            ),
        ] {
            for value in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert_eq!(
                    validate_graph_maximum(singleton_maximum(value, device), context, message),
                    Err(UmapError::InvalidInput(message.into())),
                    "maximum {value} must be rejected",
                );
            }
            assert_eq!(
                validate_graph_maximum(singleton_maximum(2.5, device), context, message),
                Ok(()),
            );
        }

        // Exercise the fit ordering with a constructed zero-weight graph: schedule
        // first, then read the retained maximum and preserve the fit error text.
        let weights = Tensor::<1>::zeros([2], device);
        let maximum = weights.clone().max_dim(0);
        let ids = Tensor::<1, Int>::zeros([2], device);
        let (_, _, pruned, _, _, _, _) =
            prepare_schedule(ids.clone(), ids, weights, maximum.clone(), 20, 500, 5).unwrap();
        assert_eq!(pruned.to_data().try_to_vec::<f32>().unwrap(), vec![0.0; 2]);
        assert_eq!(
            validate_graph_maximum(
                maximum,
                "could not inspect fuzzy graph weights",
                "fuzzy graph is empty after positive-edge pruning",
            ),
            Err(UmapError::InvalidInput(
                "fuzzy graph is empty after positive-edge pruning".into()
            )),
        );
    }

    #[test]
    fn positive_maximum_validates_singleton_on_cpu() {
        assert_positive_maximum_rejected(&Device::cpu());
    }

    #[test]
    fn positive_maximum_validates_singleton_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        assert_positive_maximum_rejected(&device);
    }

    fn assert_query_stats_detects_invalid_marker(device: &Device) {
        let valid = Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 2, 3], [2, 2]), device);
        assert!(check_query_stats(valid).is_ok());
        // Non-leading position: the marker is not in the first element.
        let nonleading =
            Tensor::<2, Int>::from_data(TensorData::new(vec![0, 1, 2, -1], [2, 2]), device);
        assert_eq!(
            check_query_stats(nonleading),
            Err(UmapError::Numerical(
                "approximate query encountered an invalid fitted neighbor graph".into()
            ))
        );
    }

    #[test]
    fn query_stats_bool_scalar_detects_nonleading_marker_on_cpu() {
        assert_query_stats_detects_invalid_marker(&Device::cpu());
    }

    #[test]
    fn query_stats_bool_scalar_detects_nonleading_marker_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        assert_query_stats_detects_invalid_marker(&device);
    }

    #[test]
    fn head_csr_keeps_empty_graph_tensor_nonempty() {
        let (offsets, edge_ids) = build_head_csr(&[0, 1], &[0.0, 0.0], 3).unwrap();
        assert_eq!(offsets, vec![0, 0, 0, 0]);
        assert_eq!(edge_ids, vec![1]);
    }

    fn device_csr_matches_host(device: &Device) {
        let cases = [
            (vec![2, 0, 2, 1, 0, 2], vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0]),
            (
                vec![0, 0, 1, 1, 2, 2, 0, 0],
                vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0],
            ),
            (vec![0, 1, 0], vec![0.0, 0.0, 0.0]),
        ];
        for (heads, weights) in cases {
            let (expected_offsets, expected_ids) = build_head_csr(&heads, &weights, 4).unwrap();
            let edges = heads.len();
            for _ in 0..3 {
                let (offsets, ids) = <Dispatch as StandardUmapOps>::head_csr(
                    Tensor::<1, Int>::from_data(TensorData::new(heads.clone(), [edges]), device)
                        .into_dispatch(),
                    Tensor::<1>::from_data(TensorData::new(weights.clone(), [edges]), device)
                        .into_dispatch(),
                    4,
                );
                let offsets = Tensor::<1, Int>::from_dispatch(offsets)
                    .to_data()
                    .try_to_vec::<i32>()
                    .unwrap();
                let ids = Tensor::<1, Int>::from_dispatch(ids)
                    .to_data()
                    .try_to_vec::<i32>()
                    .unwrap();
                assert_eq!(offsets, expected_offsets);
                assert_eq!(&ids[..expected_ids.len()], expected_ids);
            }
        }
        for (heads, weights, marker) in [
            (vec![-1, 0], vec![1.0, 1.0], -2),
            (vec![4, 0], vec![0.0, 1.0], -2),
            (vec![0, 1], vec![f32::NAN, 1.0], -3),
            (vec![0, 1], vec![f32::INFINITY, 1.0], -3),
            (vec![0, 1], vec![f32::NEG_INFINITY, 1.0], -3),
            (vec![0, 1], vec![-1.0, 1.0], -3),
            (vec![0, -1], vec![-1.0, 1.0], -3),
            (vec![-1, 0], vec![1.0, -1.0], -2),
        ] {
            let (offsets, _) = <Dispatch as StandardUmapOps>::head_csr(
                Tensor::<1, Int>::from_data(TensorData::new(heads, [2]), device).into_dispatch(),
                Tensor::<1>::from_data(TensorData::new(weights, [2]), device).into_dispatch(),
                4,
            );
            let offsets = Tensor::<1, Int>::from_dispatch(offsets)
                .to_data()
                .try_to_vec::<i32>()
                .unwrap();
            assert_eq!(offsets[0], marker);
        }
    }

    #[test]
    fn head_csr_device_matches_host_on_cpu() {
        device_csr_matches_host(&Device::cpu());
    }

    #[test]
    fn head_csr_device_matches_host_on_cuda() {
        if let Some(device) = crate::cuda_backend::device(0) {
            device_csr_matches_host(&device);
        } else {
            assert!(
                std::env::var_os("CUSCAN_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
        }
    }

    #[test]
    fn head_csr_rejects_invalid_endpoints_weights_and_shapes() {
        assert!(build_head_csr(&[-1], &[1.0], 1).is_err());
        assert!(build_head_csr(&[1], &[1.0], 1).is_err());
        assert!(build_head_csr(&[0], &[f32::NAN], 1).is_err());
        assert!(build_head_csr(&[0], &[-1.0], 1).is_err());
        assert!(build_head_csr(&[0], &[], 1).is_err());
    }

    #[test]
    fn transform_graph_marker_errors_checked_per_call() {
        let device = Device::cpu();
        let valid_heads = Tensor::<1, Int>::from_data(TensorData::new(vec![0], [1]), &device);
        let invalid_heads = Tensor::<1, Int>::from_data(TensorData::new(vec![-2, 0], [2]), &device);
        let expected = Err(UmapError::Numerical(
            "fuzzy graph contains missing or invalid neighbor indices (for example, a squared feature distance overflowed float32)".into(),
        ));

        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
        assert_eq!(check_graph_marker(&invalid_heads), expected);
        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
        assert_eq!(check_graph_marker(&invalid_heads), expected);
        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
    }
}
