use burn::{
    backend::{Dispatch, tensor::IntTensor},
    tensor::{DType, Device, Int, Tensor, TensorData},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use std::sync::OnceLock;

use crate::backend::StandardUmapOps;

use super::{
    Embedding, Init, NeighborSearch, Result, UMAPConfig, UmapError, approximate_query_budget,
    fit_curve, fit_neighbor_widths,
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
    let mut indices = HashMap::<Vec<u32>, usize>::with_capacity(data.len());
    let mut rows = Vec::new();
    let mut inverse = Vec::with_capacity(data.len());
    for row in data {
        let key = row
            .iter()
            .map(|&value| if value == 0.0 { 0 } else { value.to_bits() })
            .collect();
        let index = *indices.entry(key).or_insert_with(|| {
            let index = rows.len();
            rows.push(row.clone());
            index
        });
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
                "fuzzy graph contains invalid neighbor indices".into(),
            ));
        }
        let fit_graph = ValidatedFitGraph(fit_graph);
        let default_epochs = if n <= 10_000 { 500 } else { 200 };
        let requested_epochs = self.config.n_epochs.unwrap_or(default_epochs);
        let n_epochs = u32::try_from(requested_epochs)
            .map_err(|_| UmapError::Overflow("optimization epochs"))?;
        #[cfg(feature = "tracing")]
        let _schedule_span = tracing::info_span!("schedule", rows = n).entered();
        let maximum = weights.clone().max_dim(0);
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
        drop(_export_span);
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
        let rng = optimizer_rng_from_first_coordinates(
            flat.chunks_exact(n_components).map(|row| row[0]),
            n,
            seed,
            &device,
        )?;
        let initial = Tensor::<2>::from_data(TensorData::new(flat, [n, n_components]), &device);
        let (a, b) = fit_curve(self.config.min_dist, self.config.spread)?;
        #[cfg(feature = "tracing")]
        profile_sync(&device)?;
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

#[cfg(all(test, feature = "tracing"))]
struct OriginalFitState {
    config: UMAPConfig,
    backend: crate::standard_umap::BackendChoice,
    search: NeighborSearch,
    n_features: usize,
    training_data: Tensor<2>,
    fit_graph: Tensor<2, Int>,
    embedding_tensor: Tensor<2>,
    a: f32,
    b: f32,
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
            .then(|| self.fit_graph.as_ref())
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

    #[cfg(all(test, feature = "tracing"))]
    fn original_fit_state(&self) -> OriginalFitState {
        let fit_graph = match &self.fit_graph {
            Some(graph) => graph.0.clone(),
            None => Tensor::<2, Int>::zeros([1, 0], &self.training_data.device()),
        };
        OriginalFitState {
            config: self.config.clone(),
            backend: self.backend,
            search: self.search,
            n_features: self.n_features,
            training_data: self.training_data.clone(),
            fit_graph,
            embedding_tensor: self.embedding_tensor.clone(),
            a: self.a,
            b: self.b,
        }
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
        self.transform_flat_impl(data, n_rows, n_features, None)
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

    fn transform_impl(
        &self,
        data: &[Vec<f32>],
        diagnostic_indices: Option<&mut Vec<i32>>,
    ) -> Result<Embedding> {
        let (q, query_features) = validate_input(data, Some(self.n_features))?;
        q.checked_mul(query_features)
            .ok_or(UmapError::Overflow("transform input shape"))?;
        let flat = data.iter().flat_map(|row| row.iter().copied()).collect();
        let values = self.transform_flat_impl(flat, q, query_features, diagnostic_indices)?;
        Ok(values
            .chunks_exact(self.embedding_tensor.dims()[1])
            .map(|row| row.to_vec())
            .collect())
    }

    fn transform_flat_impl(
        &self,
        data: Vec<f32>,
        q: usize,
        query_features: usize,
        diagnostic_indices: Option<&mut Vec<i32>>,
    ) -> Result<Vec<f32>> {
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
                let budget = approximate_query_budget(n, k, self.config.transform_queue_size)?;
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
                    check_query_stats(Tensor::from_dispatch(stats))?;
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
                        check_query_stats(Tensor::from_dispatch(stats))?;
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
        let maximum = weights.clone().max_dim(0);
        validate_graph_maximum(
            maximum.clone(),
            "could not inspect transform graph",
            "transform fuzzy graph is empty after positive-edge pruning",
        )?;
        let sums = directed.clone().sum_dim(1).reshape([q, 1]);
        let normalized = directed.div(sums.clone().mask_fill(sums.equal_elem(0.0), 1.0));
        let initial = weighted_initialization(
            &self.embedding_tensor,
            &indices,
            &normalized,
            super::query::TRANSFORM_SCRATCH_BUDGET_BYTES,
        )?;
        let (heads, tails, weights, positive, next_positive, negative, next_negative) =
            prepare_schedule(
                heads,
                tails,
                weights,
                maximum,
                n_epochs_u32,
                default_epochs,
                self.config.negative_sample_rate,
            )?;
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
        embedding
            .to_data()
            .try_to_vec::<f32>()
            .map_err(|error| UmapError::Numerical(format!("could not read embedding: {error}")))
    }
}

fn weighted_initialization_chunk_rows(
    query_rows: usize,
    k: usize,
    components: usize,
    scratch_limit_bytes: usize,
) -> Result<usize> {
    // Include gather/reshape storage, indices, the broadcast product, the live
    // slot, and old and new accumulator tensors, including k=2.
    let row_words = k
        .checked_mul(components)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(k))
        .and_then(|value| {
            components
                .checked_mul(4)
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
        chunks.push(sum);
    }
    Ok(Tensor::cat(chunks, 0))
}

fn selected_device(
    choice: crate::standard_umap::BackendChoice,
) -> Result<(Device, crate::standard_umap::BackendChoice)> {
    fast_cubecl_common::selected_device(choice, cfg!(feature = "cpu"), cfg!(feature = "cuda"))
        .map_err(super::backend_choice::map_backend_error)
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
/// `maximum / divisor` with `divisor >= 1`, so a finite positive maximum always
/// survives and remains the post-prune maximum. Callers validate it once.
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
    let mut state = Vec::with_capacity(
        rows.checked_mul(3)
            .ok_or(UmapError::Overflow("optimizer RNG state"))?,
    );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_flat_preserves_validation_errors() {
        #[cfg(feature = "cpu")]
        let backend = crate::standard_umap::BackendChoice::Cpu;
        #[cfg(all(not(feature = "cpu"), feature = "cuda"))]
        let backend = crate::standard_umap::BackendChoice::Cuda { device_index: 0 };
        let data = (0..16)
            .map(|row| vec![row as f32, row as f32 * 0.25])
            .collect::<Vec<_>>();
        let fitted = UMAP::new(
            UMAPConfig {
                n_neighbors: 3,
                n_epochs: Some(0),
                init: Init::Random,
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
        let actual =
            weighted_initialization(&embedding, &indices, &normalized, scratch_limit_bytes)
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
            UMAPConfig::default(),
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
    #[cfg(feature = "cpu")]
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
                values.chunks_exact(2).map(|row| row[0]),
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
        const ROW_BYTES: usize = (2 * 3 * 2 + 3 + 4 * 2) * std::mem::size_of::<f32>();
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
            weighted_initialization_chunk_rows(usize::MAX, 1, 1, 32),
            Err(UmapError::Overflow("transform initialization output"))
        );
        const TWO_NEIGHBOR_ROW_BYTES: usize = (2 * 2 * 8 + 2 + 4 * 8) * std::mem::size_of::<f32>();
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
    #[cfg(feature = "cpu")]
    fn chunked_weighted_initialization_matches_original_on_cpu() {
        assert_chunked_weighted_initialization_matches_original(&Device::cpu(), 200);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_chunked_weighted_initialization_matches_original_with_small_limit() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        };
        // Q=7 and two rows per chunk leaves a one-row final chunk.
        assert_eq!(weighted_initialization_chunk_rows(7, 3, 2, 200), Ok(2));
        assert_chunked_weighted_initialization_matches_original(&device, 200);
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
    #[cfg(feature = "cpu")]
    fn positive_maximum_validates_singleton_on_cpu() {
        assert_positive_maximum_rejected(&Device::cpu());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn positive_maximum_validates_singleton_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
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
    #[cfg(feature = "cpu")]
    fn query_stats_bool_scalar_detects_nonleading_marker_on_cpu() {
        assert_query_stats_detects_invalid_marker(&Device::cpu());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn query_stats_bool_scalar_detects_nonleading_marker_on_cuda() {
        let Some(device) = crate::cuda_backend::device(0) else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
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
    #[cfg(feature = "cpu")]
    fn head_csr_device_matches_host_on_cpu() {
        device_csr_matches_host(&Device::cpu());
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn head_csr_device_matches_host_on_cuda() {
        if let Some(device) = crate::cuda_backend::device(0) {
            device_csr_matches_host(&device);
        } else {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
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
    #[cfg(feature = "cpu")]
    fn transform_graph_marker_errors_checked_per_call() {
        let device = Device::cpu();
        let valid_heads = Tensor::<1, Int>::from_data(TensorData::new(vec![0], [1]), &device);
        let invalid_heads = Tensor::<1, Int>::from_data(TensorData::new(vec![-2, 0], [2]), &device);
        let expected = Err(UmapError::Numerical(
            "fuzzy graph contains invalid neighbor indices".into(),
        ));

        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
        assert_eq!(check_graph_marker(&invalid_heads), expected);
        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
        assert_eq!(check_graph_marker(&invalid_heads), expected);
        assert_eq!(check_graph_marker(&valid_heads), Ok(()));
    }
}
