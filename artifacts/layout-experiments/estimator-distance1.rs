use burn::{
    backend::Dispatch,
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::backend::StandardUmapOps;

use super::{
    Embedding, Init, NeighborSearch, Result, UMAPConfig, UmapError, approximate_query_budget,
    fit_curve, fit_neighbor_widths, query_neighbor_width, validate_backend_choice, validate_input,
    validate_search_settings,
};

/// Auto uses exact search for small data and bounded NN-Descent above this row count.
///
/// Measured with `examples/standard_auto_cutoff.rs` (32 features, release, RTX 3080 /
/// Ryzen 7 9800X3D): CUDA exact beats approximate up to 4k rows (0.46 s vs 0.53 s) and
/// loses from 8k (1.78 s vs 0.93 s). On Flex, exact stays faster through 16k rows, but
/// its blocked search grows quadratically, so one shared cutoff follows the CUDA crossover.
pub const AUTO_EXACT_MAX_ROWS: usize = 4096;
// A reproducible default when the caller leaves random_state unset.
const DEFAULT_RANDOM_SEED: u64 = 42;
const APPROXIMATE_QUERY_BLOCK: usize = 64;

/// Standard, nonparametric UMAP estimator on an explicitly selected backend.
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
        validate_backend_choice(self.backend)?;
        let (n, features) = validate_input(data, None)?;
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
        let device = selected_device(self.backend)?;
        let training_data = matrix_tensor(data, &device);
        let n_components = self.config.n_components;
        n.checked_mul(n_components)
            .ok_or(UmapError::Overflow("embedding shape"))?;

        if n == 1 {
            let embedding = vec![vec![0.0; n_components]];
            let embedding_tensor = matrix_tensor(&embedding, &device);
            return Ok(FittedUMAP {
                config: self.config.clone(),
                backend: self.backend,
                search,
                n_features: features,
                training_data,
                fit_graph: Tensor::<2, Int>::zeros([1, 0], &device),
                embedding,
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
                let iters = (n as f64).log2().round().max(5.0) as u32;
                let (indices, distances) = <Dispatch as StandardUmapOps>::approximate_neighbors(
                    training_data.clone().into_dispatch(),
                    widths.nonself as u32,
                    seed,
                    iters,
                );
                (
                    Tensor::from_dispatch(indices),
                    Tensor::from_dispatch(distances),
                )
            }
            NeighborSearch::Auto => unreachable!("search is resolved before dispatch"),
        };
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
                "fuzzy graph contains invalid neighbor indices".into(),
            ));
        }
        let default_epochs = if n <= 10_000 { 500 } else { 200 };
        let requested_epochs = self.config.n_epochs.unwrap_or(default_epochs);
        let n_epochs = u32::try_from(requested_epochs)
            .map_err(|_| UmapError::Overflow("optimization epochs"))?;
        let (heads, tails, weights, positive, next_positive, negative, next_negative) =
            prepare_schedule(
                heads,
                tails,
                weights,
                n_epochs,
                default_epochs,
                self.config.negative_sample_rate,
            )?;

        let max_weight = weights
            .clone()
            .max_dim(0)
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not inspect fuzzy graph weights: {error}"))
            })?
            .into_iter()
            .next()
            .unwrap_or(0.0);
        if !max_weight.is_finite() || max_weight <= 0.0 {
            return Err(UmapError::InvalidInput(
                "fuzzy graph is empty after positive-edge pruning".into(),
            ));
        }

        let (init_heads, init_tails, init_weights) = if self.config.init == Init::Spectral {
            let host_heads = heads.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!("could not read spectral graph heads: {error}"))
            })?;
            let host_tails = tails.to_data().try_to_vec::<i32>().map_err(|error| {
                UmapError::Numerical(format!("could not read spectral graph tails: {error}"))
            })?;
            let host_weights = weights.to_data().try_to_vec::<f32>().map_err(|error| {
                UmapError::Numerical(format!("could not read spectral graph weights: {error}"))
            })?;
            let mut valid_heads = Vec::new();
            let mut valid_tails = Vec::new();
            let mut valid_weights = Vec::new();
            for ((head, tail), weight) in host_heads.into_iter().zip(host_tails).zip(host_weights) {
                if weight > 0.0 {
                    valid_heads.push(head as usize);
                    valid_tails.push(tail as usize);
                    valid_weights.push(weight);
                }
            }
            (valid_heads, valid_tails, valid_weights)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        #[cfg(feature = "tracing")]
        let _initialization_span = tracing::info_span!(
            "initialization",
            rows = n,
            components = n_components,
            method = ?self.config.init,
        )
        .entered();
        let flat = super::initialize(
            data,
            &init_heads,
            &init_tails,
            &init_weights,
            &self.config,
            seed,
        )?;
        #[cfg(feature = "tracing")]
        drop(_initialization_span);
        let initial = Tensor::<2>::from_data(TensorData::new(flat, [n, n_components]), &device);
        let (a, b) = fit_curve(self.config.min_dist, self.config.spread)?;
        let rng = optimizer_rng(&initial, seed, &device)?;
        #[cfg(feature = "tracing")]
        let _optimization_span =
            tracing::info_span!("optimization", rows = n, epochs = n_epochs).entered();
        let embedding_tensor = optimize(
            initial.clone(),
            initial,
            heads,
            tails,
            weights,
            positive,
            next_positive,
            negative,
            next_negative,
            rng,
            n_epochs,
            a,
            b,
            self.config.learning_rate,
            self.config.repulsion_strength,
            self.config.negative_sample_rate,
            true,
            true,
        )?;
        #[cfg(feature = "tracing")]
        drop(_optimization_span);
        let embedding = tensor_embedding(&embedding_tensor)?;
        Ok(FittedUMAP {
            config: self.config.clone(),
            backend: self.backend,
            search,
            n_features: features,
            training_data,
            fit_graph,
            embedding,
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
        let embedding = fitted.embedding.clone();
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
    search: NeighborSearch,
    n_features: usize,
    training_data: Tensor<2>,
    fit_graph: Tensor<2, Int>,
    embedding: Embedding,
    embedding_tensor: Tensor<2>,
    a: f32,
    b: f32,
}

impl FittedUMAP {
    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Diagnostic read-only copy of the fit's non-self neighbor ids, row-major
    /// `[rows, width]`; reads back from the device. Not part of the stable API.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    pub fn diagnostic_fit_graph(&self) -> Result<(Vec<i32>, usize)> {
        let width = self.fit_graph.dims()[1];
        let ids = self
            .fit_graph
            .to_data()
            .try_to_vec::<i32>()
            .map_err(|error| UmapError::Numerical(format!("could not read fit graph: {error}")))?;
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
        let (q, query_features) = validate_input(data, Some(self.n_features))?;
        q.checked_mul(query_features)
            .ok_or(UmapError::Overflow("transform input shape"))?;
        if self.training_data.dims()[0] == 1 {
            return Err(UmapError::InvalidInput(
                "Transform unavailable when model was fit with only a single data sample.".into(),
            ));
        }
        let device = selected_device(self.backend)?;
        if device != self.training_data.device() {
            return Err(UmapError::BackendUnavailable(
                "fitted tensors are not on the selected backend device".into(),
            ));
        }
        let queries = matrix_tensor(data, &device);
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
                    queries,
                    self.training_data.clone(),
                    k,
                    false,
                )?;
                (result.indices, result.distances)
            }
            NeighborSearch::Approximate => {
                let budget = approximate_query_budget(n, k, self.config.transform_queue_size)?;
                let mut all_indices = Vec::new();
                let mut all_distances = Vec::new();
                for start in (0..q).step_by(APPROXIMATE_QUERY_BLOCK) {
                    let end = (start + APPROXIMATE_QUERY_BLOCK).min(q);
                    let block = matrix_tensor(&data[start..end], &device);
                    let (indices, distances, stats) =
                        <Dispatch as StandardUmapOps>::approximate_query(
                            block.into_dispatch(),
                            self.training_data.clone().into_dispatch(),
                            self.fit_graph.clone().into_dispatch(),
                            k as u32,
                            budget.candidates as u32,
                            budget.visits as u32,
                            self.config.transform_seed,
                            start as u64,
                        );
                    let stats: Tensor<2, Int> = Tensor::from_dispatch(stats);
                    if stats
                        .to_data()
                        .try_to_vec::<i32>()
                        .map_err(|error| {
                            UmapError::Numerical(format!(
                                "could not inspect approximate-query stats: {error}"
                            ))
                        })?
                        .contains(&-1)
                    {
                        return Err(UmapError::Numerical(
                            "approximate query encountered an invalid fitted neighbor graph".into(),
                        ));
                    }
                    all_indices.push(Tensor::<2, Int>::from_dispatch(indices));
                    all_distances.push(Tensor::<2>::from_dispatch(distances));
                }
                (Tensor::cat(all_indices, 0), Tensor::cat(all_distances, 0))
            }
            NeighborSearch::Auto => unreachable!("fitted search mode is resolved"),
        };
        #[cfg(feature = "tracing")]
        drop(_neighbors_span);
        let local_connectivity = (self.config.local_connectivity - 1.0).max(0.0);
        #[cfg(feature = "tracing")]
        let _graph_span = tracing::info_span!("fuzzy_graph", rows = q, neighbors = k).entered();
        let (heads, tails, weights, _, _, directed) = <Dispatch as StandardUmapOps>::fuzzy_graph(
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
        let n_epochs = self
            .config
            .n_epochs
            .map(|epochs| epochs / 3)
            .unwrap_or(if q <= 10_000 { 100 } else { 30 });
        let n_epochs_u32 = u32::try_from(n_epochs)
            .map_err(|_| UmapError::Overflow("transform optimization epochs"))?;
        let default_epochs = if q <= 10_000 { 100 } else { 30 };
        let max_weight = weights.clone().max_dim(0);
        if max_weight
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not inspect transform graph: {error}"))
            })?
            .first()
            .copied()
            .unwrap_or(0.0)
            <= 0.0
        {
            return Err(UmapError::InvalidInput(
                "transform fuzzy graph is empty after positive-edge pruning".into(),
            ));
        }
        let sums = directed.clone().sum_dim(1).reshape([q, 1]);
        let normalized = directed.div(sums.clone().mask_fill(sums.equal_elem(0.0), 1.0));
        let k = indices.dims()[1];
        let components = self.embedding_tensor.dims()[1];
        let mut initial = Tensor::<2>::zeros([q, components], &device);
        for slot in 0..k {
            let ids = indices.clone().slice_dim(1, slot..slot + 1).reshape([q]);
            let neighbors = self.embedding_tensor.clone().select(0, ids);
            let weight = normalized
                .clone()
                .slice_dim(1, slot..slot + 1)
                .repeat_dim(1, components);
            initial = initial.add(neighbors.mul(weight));
        }
        let (heads, tails, weights, positive, next_positive, negative, next_negative) =
            prepare_schedule(
                heads,
                tails,
                weights,
                n_epochs_u32,
                default_epochs,
                self.config.negative_sample_rate,
            )?;
        if weights
            .clone()
            .max_dim(0)
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| {
                UmapError::Numerical(format!("could not inspect transform graph: {error}"))
            })?
            .first()
            .copied()
            .unwrap_or(0.0)
            <= 0.0
        {
            return Err(UmapError::InvalidInput(
                "transform fuzzy graph is empty after positive-edge pruning".into(),
            ));
        }
        let transform_seed = self.config.transform_seed;
        let rng = optimizer_rng(&initial, transform_seed, &device)?;
        #[cfg(feature = "tracing")]
        let _optimization_span =
            tracing::info_span!("optimization", rows = q, epochs = n_epochs_u32).entered();
        let embedding = optimize(
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
        )?;
        #[cfg(feature = "tracing")]
        drop(_optimization_span);
        tensor_embedding(&embedding)
    }
}

fn selected_device(choice: crate::standard_umap::BackendChoice) -> Result<Device> {
    match choice {
        crate::standard_umap::BackendChoice::Flex => Ok(Device::flex()),
        crate::standard_umap::BackendChoice::Cuda { device_index } => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda_backend::device(device_index).ok_or_else(|| {
                    UmapError::BackendUnavailable(format!(
                        "CUDA device index {device_index} is unavailable"
                    ))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = device_index;
                Err(UmapError::BackendUnavailable(
                    "CUDA backend requires the `cuda` feature".into(),
                ))
            }
        }
    }
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
            "fuzzy graph contains invalid neighbor indices".into(),
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

#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
fn prepare_schedule(
    heads: Tensor<1, Int>,
    tails: Tensor<1, Int>,
    weights: Tensor<1>,
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
    let maximum = weights.clone().max_dim(0);
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
    let (head_offsets, head_edges, head_groups, group_count) =
        edge_schedule(&heads, &tails, &weights, head.dims()[0], tail.dims()[0], shared, &device)?;
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
                head_groups.clone().into_dispatch(),
                group_count,
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
    Ok(head)
}

/// Build stable head CSR and a deterministic vertex coloring for positive
/// shared-support edges. Coloring separates adjacent heads, but is not a
/// positive-write-set independence guarantee: distinct heads may still write
/// the same shared tail, so parallel coordinate updates require atomics.
fn edge_schedule(
    heads: &Tensor<1, Int>,
    tails: &Tensor<1, Int>,
    weights: &Tensor<1>,
    head_rows: usize,
    tail_rows: usize,
    shared: bool,
    device: &Device,
) -> Result<(Tensor<1, Int>, Tensor<1, Int>, Tensor<1, Int>, u32)> {
    let heads = heads
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|error| UmapError::Numerical(format!("could not read layout edges: {error}")))?;
    let tails = tails
        .to_data()
        .try_to_vec::<i32>()
        .map_err(|error| UmapError::Numerical(format!("could not read layout edges: {error}")))?;
    let weights = weights
        .to_data()
        .try_to_vec::<f32>()
        .map_err(|error| UmapError::Numerical(format!("could not read layout weights: {error}")))?;
    if heads.len() != tails.len() || heads.len() != weights.len() {
        return Err(UmapError::Numerical("layout edge arrays have different lengths".into()));
    }
    let mut offsets = vec![0i32; head_rows + 1];
    let mut adjacency = if shared { vec![Vec::new(); head_rows] } else { Vec::new() };
    for ((&head, &tail), &weight) in heads.iter().zip(&tails).zip(&weights) {
        let head = usize::try_from(head)
            .ok()
            .filter(|&row| row < head_rows)
            .ok_or_else(|| UmapError::Numerical("layout edge endpoint out of range".into()))?;
        let tail = usize::try_from(tail)
            .ok()
            .filter(|&row| row < tail_rows)
            .ok_or_else(|| UmapError::Numerical("layout edge endpoint out of range".into()))?;
        if weight > 0.0 {
            offsets[head + 1] += 1;
            if shared {
                if tail >= head_rows {
                    return Err(UmapError::Numerical("shared layout endpoint out of range".into()));
                }
                if head != tail {
                    adjacency[head].push(tail);
                    adjacency[tail].push(head);
                }
            }
        }
    }
    for row in 0..head_rows {
        offsets[row + 1] += offsets[row];
    }
    let total = offsets[head_rows] as usize;
    let mut cursor = offsets.clone();
    // Keep a non-empty buffer; offsets bound every read.
    let mut ids = vec![0i32; total.max(1)];
    for (edge, (&row, &weight)) in heads.iter().zip(&weights).enumerate() {
        if weight > 0.0 {
            let slot = &mut cursor[row as usize];
            ids[*slot as usize] =
                i32::try_from(edge).map_err(|_| UmapError::Overflow("layout edge id"))?;
            *slot += 1;
        }
    }
    let (groups, group_count) = if shared {
        let (groups, count) = greedy_distance1_coloring(&adjacency)?;
        (groups, count)
    } else {
        (vec![0i32; head_rows], 1)
    };
    let len = ids.len();
    Ok((
        Tensor::from_data(TensorData::new(offsets, [head_rows + 1]), device),
        Tensor::from_data(TensorData::new(ids, [len]), device),
        Tensor::from_data(TensorData::new(groups, [head_rows]), device),
        group_count,
    ))
}

/// Greedy smallest-available-color assignment in ascending original vertex id.
fn greedy_distance1_coloring(adjacency: &[Vec<usize>]) -> Result<(Vec<i32>, u32)> {
    let mut colors = vec![0i32; adjacency.len()];
    let mut used = vec![false; adjacency.len()];
    let mut color_count = 0usize;
    for vertex in 0..adjacency.len() {
        for &neighbor in &adjacency[vertex] {
            if neighbor >= adjacency.len() {
                return Err(UmapError::Numerical("layout coloring neighbor out of range".into()));
            }
            if neighbor < vertex {
                used[colors[neighbor] as usize] = true;
            }
        }
        let color = used.iter().position(|&taken| !taken).unwrap_or(used.len());
        let color = i32::try_from(color).map_err(|_| UmapError::Overflow("layout color"))?;
        colors[vertex] = color;
        color_count = color_count.max(color as usize + 1);
        for &neighbor in &adjacency[vertex] {
            if neighbor < vertex {
                used[colors[neighbor] as usize] = false;
            }
        }
    }
    let count = u32::try_from(color_count).map_err(|_| UmapError::Overflow("layout group count"))?;
    Ok((colors, count.max(1)))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn estimator(search: NeighborSearch) -> UMAP {
        UMAP::new(
            UMAPConfig::default(),
            crate::standard_umap::BackendChoice::Flex,
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
    fn distance1_coloring_separates_adjacent_vertices_and_leaves_empty_rows() {
        let adjacency = vec![vec![1], vec![0, 2], vec![1], vec![], vec![]];
        let (colors, count) = greedy_distance1_coloring(&adjacency).unwrap();
        assert_ne!(colors[0], colors[1]);
        assert_ne!(colors[1], colors[2]);
        assert_eq!(colors[0], colors[2]);
        assert_eq!(colors[3], 0);
        assert_eq!(colors[4], 0);
        assert_eq!(count, 2);
    }

    #[test]
    fn distance1_coloring_handles_self_and_duplicate_edges() {
        // Self loops impose no constraint; duplicate adjacency entries do not
        // change the deterministic smallest-available-color result.
        let adjacency = vec![vec![0, 1, 1], vec![0, 0]];
        let (colors, count) = greedy_distance1_coloring(&adjacency).unwrap();
        assert_ne!(colors[0], colors[1]);
        assert_eq!(colors, vec![0, 1]);
        assert_eq!(count, 2);
    }

    #[test]
    fn distance1_coloring_keeps_disconnected_components_deterministic() {
        let adjacency = vec![vec![1], vec![0], vec![3], vec![2], vec![]];
        let (colors, count) = greedy_distance1_coloring(&adjacency).unwrap();
        assert_eq!(colors, vec![0, 1, 0, 1, 0]);
        assert_eq!(count, 2);
    }
}
