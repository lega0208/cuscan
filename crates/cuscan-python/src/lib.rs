use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyModule};
use umap::{
    BackendChoice, FittedUMAP as RustFittedUMAP, HubSelection, Init, InitRescale, NeighborSearch,
    Optimizer, RecursiveInitConfig, UMAP, UMAPConfig, UmapError,
};

pub(crate) mod hdbscan;

#[cfg(feature = "tracing")]
type DiagnosticTransformArrays<'py> = (Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>);

/// Flat 1-D NumPy arrays for every replay field; callers reshape `[q, k]`
/// fields with `k = len(neighbor_indices) // q` and placements with
/// `n_components`.
// A macro because the replay type is not nameable outside the hook methods.
#[cfg(feature = "tracing")]
macro_rules! replay_dict {
    ($py:expr, $replay:expr) => {{
        let py = $py;
        let replay = $replay;
        let dict = PyDict::new(py);
        dict.set_item("neighbor_indices", replay.neighbor_indices.into_pyarray(py))?;
        dict.set_item(
            "neighbor_distances",
            replay.neighbor_distances.into_pyarray(py),
        )?;
        dict.set_item("query_stats", replay.query_stats.into_pyarray(py))?;
        dict.set_item("rho", replay.rho.into_pyarray(py))?;
        dict.set_item("sigma", replay.sigma.into_pyarray(py))?;
        dict.set_item("memberships", replay.memberships.into_pyarray(py))?;
        dict.set_item("graph_heads", replay.graph_heads.into_pyarray(py))?;
        dict.set_item("graph_tails", replay.graph_tails.into_pyarray(py))?;
        dict.set_item("graph_weights", replay.graph_weights.into_pyarray(py))?;
        dict.set_item("edge_heads", replay.edge_heads.into_pyarray(py))?;
        dict.set_item("edge_tails", replay.edge_tails.into_pyarray(py))?;
        dict.set_item("edge_weights", replay.edge_weights.into_pyarray(py))?;
        dict.set_item("epochs", replay.epochs)?;
        dict.set_item("initial_alpha", replay.initial_alpha)?;
        dict.set_item("csr_offsets", replay.csr_offsets.into_pyarray(py))?;
        dict.set_item("csr_indices", replay.csr_indices.into_pyarray(py))?;
        dict.set_item("csr_weights", replay.csr_weights.into_pyarray(py))?;
        dict.set_item("periods", replay.periods.into_pyarray(py))?;
        dict.set_item("negative_periods", replay.negative_periods.into_pyarray(py))?;
        dict.set_item("next_positive", replay.next_positive.into_pyarray(py))?;
        dict.set_item("next_negative", replay.next_negative.into_pyarray(py))?;
        dict.set_item("initial", replay.initial.into_pyarray(py))?;
        dict.set_item("embedding", replay.embedding.into_pyarray(py))?;
        PyResult::Ok(dict)
    }};
}

#[pyclass(frozen, unsendable)]
struct FittedUMAP {
    inner: RustFittedUMAP,
}

#[pymethods]
impl FittedUMAP {
    /// Return `(arrays, metadata)` containing independent NumPy snapshots.
    fn export_state<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let state = py
            .detach(|| self.inner.export_state())
            .map_err(map_umap_error)?;
        let arrays = PyDict::new(py);
        let features = Array2::from_shape_vec(
            (state.training_rows, state.n_features),
            state.training_features,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let embedding =
            Array2::from_shape_vec((state.embedding_rows, state.n_components), state.embedding)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        arrays.set_item("training_features", features.into_pyarray(py))?;
        arrays.set_item("embedding", embedding.into_pyarray(py))?;
        arrays.set_item(
            "query_graph",
            state.query_graph.map(|v| {
                Array2::from_shape_vec((state.training_rows, state.query_graph_width), v)
                    .expect("validated graph shape")
                    .into_pyarray(py)
            }),
        )?;
        arrays.set_item(
            "original_to_training",
            state.original_to_training.map(|v| {
                v.into_iter()
                    .map(|i| i as u64)
                    .collect::<Vec<_>>()
                    .into_pyarray(py)
            }),
        )?;
        let metadata = PyDict::new(py);
        let c = state.config;
        metadata.set_item("n_components", c.n_components)?;
        metadata.set_item("n_neighbors", c.n_neighbors)?;
        metadata.set_item("min_dist", c.min_dist)?;
        metadata.set_item("spread", c.spread)?;
        metadata.set_item("local_connectivity", c.local_connectivity)?;
        metadata.set_item("set_op_mix_ratio", c.set_op_mix_ratio)?;
        metadata.set_item("n_epochs", c.n_epochs)?;
        metadata.set_item("learning_rate", c.learning_rate)?;
        metadata.set_item("negative_sample_rate", c.negative_sample_rate)?;
        metadata.set_item("repulsion_strength", c.repulsion_strength)?;
        metadata.set_item("optimizer", optimizer_name(c.optimizer))?;
        metadata.set_item("negative_selection_range", c.negative_selection_range)?;
        metadata.set_item("negative_sample_scale", c.negative_sample_scale)?;
        metadata.set_item("exclude_graph_neighbors", c.exclude_graph_neighbors)?;
        metadata.set_item(
            "negative_sample_scale_adaptation_samples",
            c.negative_sample_scale_adaptation_samples,
        )?;
        metadata.set_item("init_scale_factor", c.init_scale_factor)?;
        metadata.set_item(
            "init_rescale",
            match c.init_rescale {
                InitRescale::Isotropic => "isotropic",
                InitRescale::PerAxis => "per_axis",
                InitRescale::GraphEdge => "graph_edge",
            },
        )?;
        metadata.set_item(
            "init",
            match c.init {
                Init::Random => "random",
                Init::Spectral => "spectral",
                Init::Recursive => "recursive",
                Init::Auto => "auto",
            },
        )?;
        metadata.set_item(
            "recursive_coarsening_ratio",
            c.recursive_init.coarsening_ratio,
        )?;
        metadata.set_item(
            "recursive_base_init_threshold",
            c.recursive_init.base_init_threshold,
        )?;
        metadata.set_item(
            "recursive_n_embedding_epochs",
            c.recursive_init.n_embedding_epochs,
        )?;
        metadata.set_item(
            "recursive_hub_selection",
            hub_selection_name(c.recursive_init.hub_selection),
        )?;
        metadata.set_item(
            "recursive_anchor_sample_size",
            c.recursive_init.anchor_sample_size,
        )?;
        metadata.set_item("random_state", c.random_state)?;
        metadata.set_item("transform_seed", c.transform_seed)?;
        metadata.set_item("transform_queue_size", c.transform_queue_size)?;
        metadata.set_item("unique", c.unique)?;
        metadata.set_item("memory_limit_bytes", c.memory_limit_bytes)?;
        metadata.set_item("requested_search", search_name(state.requested_search))?;
        metadata.set_item("resolved_search", search_name(state.resolved_search))?;
        metadata.set_item("a", state.a)?;
        metadata.set_item("b", state.b)?;
        metadata.set_item("training_rows", state.training_rows)?;
        metadata.set_item("n_features", state.n_features)?;
        metadata.set_item("query_graph_width", state.query_graph_width)?;
        Ok((arrays, metadata))
    }

    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let embedding = self.inner.embedding();
        let rows = embedding.len();
        let columns = embedding.first().map_or(0, Vec::len);
        let values = embedding.iter().flatten().copied().collect();
        Array2::from_shape_vec((rows, columns), values)
            .expect("embedding is rectangular")
            .into_pyarray(py)
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        self.transform_owned(py, x, false).map(|(array, _)| array)
    }

    /// Timed diagnostic for verifying GIL release around the native transform.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    fn _timed_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, u64, u64)> {
        let (array, timing) = self.transform_owned(py, x, true)?;
        let (start_ns, end_ns) = timing.expect("timed transform captures native timestamps");
        Ok((array, start_ns, end_ns))
    }

    /// Untimed diagnostic exposing the actual neighbor IDs used by transform.
    #[cfg(feature = "tracing")]
    fn _diagnostic_transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<DiagnosticTransformArrays<'py>> {
        let data: Vec<Vec<f32>> = x
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        let (embedding, indices) = self
            .inner
            .diagnostic_transform(&data)
            .map_err(map_umap_error)?;
        let rows = embedding.len();
        let columns = self.inner.embedding()[0].len();
        let width = indices.len() / rows;
        let values = embedding.into_iter().flatten().collect();
        Ok((
            Array2::from_shape_vec((rows, columns), values)
                .expect("transform output is rectangular")
                .into_pyarray(py),
            Array2::from_shape_vec((rows, width), indices)
                .expect("neighbor diagnostics are rectangular")
                .into_pyarray(py),
        ))
    }

    /// Measurement-only stage replay (Phase 7 D6), search entry. `candidates`
    /// and `visits` override the approximate budget together (C/V sweep).
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    #[pyo3(signature = (x, candidates=None, visits=None))]
    fn _replay_transform_search<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        candidates: Option<usize>,
        visits: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let budget = match (candidates, visits) {
            (None, None) => None,
            (Some(candidates), Some(visits)) => {
                Some(umap::standard_umap::QueryBudget { candidates, visits })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "candidates and visits must be given together",
                ));
            }
        };
        let data: Vec<Vec<f32>> = x
            .as_array()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();
        drop(x);
        let replay = py
            .detach(|| self.inner.replay_transform_search(&data, budget))
            .map_err(map_umap_error)?;
        replay_dict!(py, replay)
    }

    /// Measurement-only stage replay (Phase 7 D6), graph entry from injected
    /// `[rows, k]` query neighbor IDs and distances.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    fn _replay_transform_graph<'py>(
        &self,
        py: Python<'py>,
        indices: PyReadonlyArray2<'py, i32>,
        distances: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if indices.shape() != distances.shape() {
            return Err(PyValueError::new_err(
                "indices and distances must have the same shape",
            ));
        }
        let rows = indices.shape()[0];
        let indices = indices.as_array().iter().copied().collect::<Vec<_>>();
        let distances = distances.as_array().iter().copied().collect::<Vec<_>>();
        let replay = py
            .detach(|| {
                self.inner
                    .replay_transform_graph(rows, &indices, &distances)
            })
            .map_err(map_umap_error)?;
        replay_dict!(py, replay)
    }

    /// Measurement-only stage replay (Phase 7 D6), layout entry from an
    /// injected ordered pruned edge stream and `[rows, n_components]` placement.
    #[cfg(feature = "tracing")]
    #[doc(hidden)]
    fn _replay_transform_layout<'py>(
        &self,
        py: Python<'py>,
        heads: PyReadonlyArray1<'py, i32>,
        tails: PyReadonlyArray1<'py, i32>,
        weights: PyReadonlyArray1<'py, f32>,
        initial: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let rows = initial.shape()[0];
        let heads = heads.as_array().iter().copied().collect::<Vec<_>>();
        let tails = tails.as_array().iter().copied().collect::<Vec<_>>();
        let weights = weights.as_array().iter().copied().collect::<Vec<_>>();
        let initial = initial.as_array().iter().copied().collect::<Vec<_>>();
        let replay = py
            .detach(|| {
                self.inner
                    .replay_transform_layout(rows, &heads, &tails, &weights, &initial)
            })
            .map_err(map_umap_error)?;
        replay_dict!(py, replay)
    }
}

type TransformOutput<'py> = (Bound<'py, PyArray2<f32>>, Option<(u64, u64)>);

impl FittedUMAP {
    fn transform_owned<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        timed: bool,
    ) -> PyResult<TransformOutput<'py>> {
        let shape = x.shape();
        let (n_rows, n_features) = (shape[0], shape[1]);
        let data = if x.is_c_contiguous() {
            x.as_slice()
                .expect("C-contiguous input has a slice")
                .to_vec()
        } else {
            x.as_array()
                .as_standard_layout()
                .into_owned()
                .into_raw_vec_and_offset()
                .0
        };
        drop(x);

        let n_components = self.inner.embedding()[0].len();
        let inner = &self.inner;
        let (result, timing) = py.detach(move || {
            let start_ns = timed.then(unix_time_ns);
            let result = inner.transform_flat(data, n_rows, n_features);
            let timing = start_ns.map(|start_ns| (start_ns, unix_time_ns()));
            (result, timing)
        });
        let values = result.map_err(map_umap_error)?;
        let array = Array2::from_shape_vec((n_rows, n_components), values)
            .expect("flat transform output shape is validated by Rust");
        Ok((array.into_pyarray(py), timing))
    }
}

fn unix_time_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock is after the Unix epoch")
        .as_nanos()
        .try_into()
        .expect("native diagnostic timestamp fits u64 nanoseconds")
}

fn map_umap_error(error: UmapError) -> PyErr {
    let message = error.to_string();
    match error {
        UmapError::InvalidConfig(_) | UmapError::InvalidInput(_) => PyValueError::new_err(message),
        UmapError::Memory(_) => PyMemoryError::new_err(message),
        UmapError::BackendUnavailable(_)
        | UmapError::Numerical(_)
        | UmapError::SpectralConvergence(_)
        | UmapError::Overflow(_)
        | UmapError::Capacity(_) => PyRuntimeError::new_err(message),
    }
}

fn search_name(search: NeighborSearch) -> &'static str {
    match search {
        NeighborSearch::Auto => "auto",
        NeighborSearch::Exact => "exact",
        NeighborSearch::Approximate => "approximate",
    }
}

fn parse_search(value: &str) -> PyResult<NeighborSearch> {
    match value {
        "auto" => Ok(NeighborSearch::Auto),
        "exact" => Ok(NeighborSearch::Exact),
        "approximate" => Ok(NeighborSearch::Approximate),
        _ => Err(PyValueError::new_err("invalid neighbor search mode")),
    }
}

fn optimizer_name(value: Optimizer) -> &'static str {
    match value {
        Optimizer::Compatibility => "compatibility",
        Optimizer::Adam => "adam",
        Optimizer::Momentum => "momentum",
    }
}

fn parse_optimizer(value: &str) -> PyResult<Optimizer> {
    match value {
        "compatibility" => Ok(Optimizer::Compatibility),
        "adam" => Ok(Optimizer::Adam),
        "momentum" => Ok(Optimizer::Momentum),
        _ => Err(PyValueError::new_err(
            "optimizer must be 'compatibility', 'adam', or 'momentum'",
        )),
    }
}

fn hub_selection_name(value: HubSelection) -> &'static str {
    match value {
        HubSelection::Degree => "degree",
        HubSelection::Diverse => "diverse",
        HubSelection::Random => "random",
    }
}

fn parse_hub_selection(value: &str) -> PyResult<HubSelection> {
    match value {
        "degree" => Ok(HubSelection::Degree),
        "diverse" => Ok(HubSelection::Diverse),
        "random" => Ok(HubSelection::Random),
        _ => Err(PyValueError::new_err(
            "hub_selection must be 'degree', 'diverse', or 'random'",
        )),
    }
}

struct RecursiveInteger(usize);

impl<'a, 'py> FromPyObject<'a, 'py> for RecursiveInteger {
    type Error = PyErr;

    fn extract(value: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let rejected = || {
            PyValueError::new_err("recursive option must be a non-negative integer fitting usize")
        };
        if !is_numeric_scalar(&value, false)? {
            return Err(rejected());
        }
        value.extract::<usize>().map(Self).map_err(|_| rejected())
    }
}

struct NegativeSelectionRange(usize);

fn is_numeric_scalar(value: &Bound<'_, PyAny>, allow_float: bool) -> PyResult<bool> {
    let numpy = PyModule::import(value.py(), "numpy")?;
    if value.is_instance_of::<PyBool>() || value.is_instance(&numpy.getattr("bool_")?)? {
        return Ok(false);
    }
    Ok(value.is_instance_of::<PyInt>()
        || value.is_instance(&numpy.getattr("integer")?)?
        || (allow_float
            && (value.is_instance_of::<PyFloat>()
                || value.is_instance(&numpy.getattr("floating")?)?)))
}

impl<'a, 'py> FromPyObject<'a, 'py> for NegativeSelectionRange {
    type Error = PyErr;

    fn extract(value: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let rejected = || {
            PyValueError::new_err(
                "negative_selection_range must be a positive integer that fits in i32",
            )
        };
        if !is_numeric_scalar(&value, false)? {
            return Err(rejected());
        }
        match value.extract::<usize>() {
            Ok(range) if range > 0 && i32::try_from(range).is_ok() => Ok(Self(range)),
            _ => Err(rejected()),
        }
    }
}

struct AdaptationSamples(usize);

impl<'a, 'py> FromPyObject<'a, 'py> for AdaptationSamples {
    type Error = PyErr;

    fn extract(value: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let rejected = || {
            PyValueError::new_err(
                "negative_sample_scale_adaptation_samples must be a nonnegative integer that fits in i32",
            )
        };
        if !is_numeric_scalar(&value, false)? {
            return Err(rejected());
        }
        match value.extract::<usize>() {
            Ok(samples) if i32::try_from(samples).is_ok() => Ok(Self(samples)),
            _ => Err(rejected()),
        }
    }
}

struct StrictBool(bool);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictBool {
    type Error = PyErr;

    fn extract(value: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let numpy = PyModule::import(value.py(), "numpy")?;
        if value.is_instance_of::<PyBool>() || value.is_instance(&numpy.getattr("bool_")?)? {
            return Ok(Self(value.is_truthy()?));
        }
        Err(PyValueError::new_err(
            "exclude_graph_neighbors must be a boolean",
        ))
    }
}

fn optional_negative_sample_scale(value: &Bound<'_, PyAny>) -> PyResult<Option<f32>> {
    if value.is_none() {
        return Ok(None);
    }
    let rejected = || {
        PyValueError::new_err(
            "negative_sample_scale must be a finite nonnegative number representable as f32 or None",
        )
    };
    if !is_numeric_scalar(value, true)? {
        return Err(rejected());
    }
    let scale = value.extract::<f64>().map_err(|_| rejected())?;
    let narrowed = scale as f32;
    if !scale.is_finite()
        || scale < 0.0
        || !narrowed.is_finite()
        || (scale != 0.0 && narrowed == 0.0)
    {
        return Err(rejected());
    }
    Ok(Some(narrowed))
}

#[pyfunction(signature = (arrays, metadata, backend="auto", device_index=0))]
#[allow(clippy::too_many_arguments)]
fn load(
    py: Python<'_>,
    arrays: &Bound<'_, PyDict>,
    metadata: &Bound<'_, PyDict>,
    backend: &str,
    device_index: usize,
) -> PyResult<FittedUMAP> {
    macro_rules! get {
        ($key:literal) => {
            metadata
                .get_item($key)?
                .ok_or_else(|| PyValueError::new_err(concat!("missing metadata key: ", $key)))?
        };
    }
    let training_features: PyReadonlyArray2<'_, f32> = arrays
        .get_item("training_features")?
        .ok_or_else(|| PyValueError::new_err("missing array: training_features"))?
        .extract()?;
    let embedding: PyReadonlyArray2<'_, f32> = arrays
        .get_item("embedding")?
        .ok_or_else(|| PyValueError::new_err("missing array: embedding"))?
        .extract()?;
    let query_graph: Option<PyReadonlyArray2<'_, i32>> = arrays
        .get_item("query_graph")?
        .ok_or_else(|| PyValueError::new_err("missing array: query_graph"))?
        .extract()?;
    let mapping: Option<PyReadonlyArray1<'_, u64>> = arrays
        .get_item("original_to_training")?
        .ok_or_else(|| PyValueError::new_err("missing array: original_to_training"))?
        .extract()?;
    let original_to_training = mapping
        .map(|values| {
            values
                .as_array()
                .iter()
                .map(|&v| {
                    usize::try_from(v)
                        .map_err(|_| PyValueError::new_err("inverse mapping value exceeds usize"))
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .transpose()?;
    let init = match get!("init").extract::<String>()?.as_str() {
        "random" => Init::Random,
        "spectral" => Init::Spectral,
        "recursive" => Init::Recursive,
        "auto" => Init::Auto,
        _ => return Err(PyValueError::new_err("invalid init")),
    };
    let requested_search = parse_search(&get!("requested_search").extract::<String>()?)?;
    let resolved_search = parse_search(&get!("resolved_search").extract::<String>()?)?;
    let config = UMAPConfig {
        n_components: get!("n_components").extract()?,
        n_neighbors: get!("n_neighbors").extract()?,
        min_dist: get!("min_dist").extract()?,
        spread: get!("spread").extract()?,
        local_connectivity: get!("local_connectivity").extract()?,
        set_op_mix_ratio: get!("set_op_mix_ratio").extract()?,
        n_epochs: get!("n_epochs").extract()?,
        learning_rate: get!("learning_rate").extract()?,
        negative_sample_rate: get!("negative_sample_rate").extract()?,
        repulsion_strength: get!("repulsion_strength").extract()?,
        optimizer: parse_optimizer(&get!("optimizer").extract::<String>()?)?,
        negative_selection_range: get!("negative_selection_range")
            .extract::<NegativeSelectionRange>()?
            .0,
        negative_sample_scale: optional_negative_sample_scale(&get!("negative_sample_scale"))?,
        exclude_graph_neighbors: get!("exclude_graph_neighbors").extract::<StrictBool>()?.0,
        negative_sample_scale_adaptation_samples: get!("negative_sample_scale_adaptation_samples")
            .extract::<AdaptationSamples>()?
            .0,
        init,
        recursive_init: RecursiveInitConfig {
            coarsening_ratio: get!("recursive_coarsening_ratio")
                .extract::<RecursiveInteger>()?
                .0,
            base_init_threshold: get!("recursive_base_init_threshold")
                .extract::<RecursiveInteger>()?
                .0,
            n_embedding_epochs: get!("recursive_n_embedding_epochs")
                .extract::<RecursiveInteger>()?
                .0,
            hub_selection: parse_hub_selection(
                &get!("recursive_hub_selection").extract::<String>()?,
            )?,
            anchor_sample_size: get!("recursive_anchor_sample_size")
                .extract::<RecursiveInteger>()?
                .0,
        },
        init_rescale: parse_init_rescale(&get!("init_rescale").extract::<String>()?)?,
        init_scale_factor: get!("init_scale_factor").extract()?,
        random_state: get!("random_state").extract()?,
        transform_seed: get!("transform_seed").extract()?,
        transform_queue_size: get!("transform_queue_size").extract()?,
        unique: get!("unique").extract()?,
        memory_limit_bytes: optional_memory_limit(&get!("memory_limit_bytes"))?,
    };
    config.validate().map_err(map_umap_error)?;
    let fshape = training_features.shape();
    let eshape = embedding.shape();
    if get!("training_rows").extract::<usize>()? != fshape[0]
        || get!("n_features").extract::<usize>()? != fshape[1]
        || get!("n_components").extract::<usize>()? != eshape[1]
        || eshape[0] != fshape[0]
    {
        return Err(PyValueError::new_err(
            "metadata dimensions do not match fitted arrays",
        ));
    }
    let features = training_features.as_array().iter().copied().collect();
    let embedding_values = embedding.as_array().iter().copied().collect();
    let (graph_values, graph_width) = if let Some(graph) = query_graph {
        if graph.shape()[0] != fshape[0]
            || get!("query_graph_width").extract::<usize>()? != graph.shape()[1]
        {
            return Err(PyValueError::new_err(
                "metadata dimensions do not match query graph",
            ));
        }
        (
            Some(graph.as_array().iter().copied().collect()),
            graph.shape()[1],
        )
    } else {
        (None, 0)
    };
    let state = umap::standard_umap::ModelState {
        config,
        requested_search,
        resolved_search,
        a: get!("a").extract()?,
        b: get!("b").extract()?,
        training_features: features,
        training_rows: fshape[0],
        n_features: fshape[1],
        embedding: embedding_values,
        embedding_rows: eshape[0],
        n_components: eshape[1],
        query_graph: graph_values,
        query_graph_width: graph_width,
        original_to_training,
    };
    let backend = match backend {
        "auto" => BackendChoice::Auto,
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index },
        _ => {
            return Err(PyValueError::new_err(
                "backend must be 'auto', 'cpu' or 'cuda'",
            ));
        }
    };
    py.detach(move || RustFittedUMAP::from_state(state, backend))
        .map(|inner| FittedUMAP { inner })
        .map_err(map_umap_error)
}

#[pyfunction(text_signature = "(x, *, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0, local_connectivity=1.0, set_op_mix_ratio=1.0, n_epochs=None, learning_rate=1.0, negative_sample_rate=5, repulsion_strength=1.0, init='auto', random_state=None, transform_seed=42, transform_queue_size=4.0, backend='auto', device_index=0, neighbor_search='auto', unique=False, memory_limit_bytes=None, init_rescale='isotropic', init_scale_factor=1.0, optimizer='adam', negative_selection_range=200000, negative_sample_scale=None, exclude_graph_neighbors=False, negative_sample_scale_adaptation_samples=128, recursive_coarsening_ratio=4, recursive_base_init_threshold=1024, recursive_n_embedding_epochs=50, recursive_hub_selection='degree', recursive_anchor_sample_size=16384)", signature = (
  x, *, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0,
  local_connectivity=1.0, set_op_mix_ratio=1.0, n_epochs=None,
  learning_rate=1.0, negative_sample_rate=5, repulsion_strength=1.0,
  init="auto", random_state=None, transform_seed=42,
  transform_queue_size=4.0, backend="auto", device_index=0,
  neighbor_search="auto", unique=false, memory_limit_bytes=None,
  init_rescale="isotropic", init_scale_factor=1.0,
  optimizer="adam", negative_selection_range=NegativeSelectionRange(200_000),
  negative_sample_scale=None, exclude_graph_neighbors=StrictBool(false),
  negative_sample_scale_adaptation_samples=AdaptationSamples(128),
  recursive_coarsening_ratio=RecursiveInteger(4), recursive_base_init_threshold=RecursiveInteger(1024),
  recursive_n_embedding_epochs=RecursiveInteger(50), recursive_hub_selection="degree",
  recursive_anchor_sample_size=RecursiveInteger(16384)))]
#[allow(clippy::too_many_arguments)]
fn fit_umap(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    local_connectivity: f32,
    set_op_mix_ratio: f32,
    n_epochs: Option<usize>,
    learning_rate: f32,
    negative_sample_rate: usize,
    repulsion_strength: f32,
    init: &str,
    random_state: Option<u64>,
    transform_seed: u64,
    transform_queue_size: f64,
    backend: &str,
    device_index: usize,
    neighbor_search: &str,
    unique: bool,
    memory_limit_bytes: Option<&Bound<'_, PyAny>>,
    init_rescale: &str,
    init_scale_factor: f32,
    optimizer: &str,
    negative_selection_range: NegativeSelectionRange,
    negative_sample_scale: Option<&Bound<'_, PyAny>>,
    exclude_graph_neighbors: StrictBool,
    negative_sample_scale_adaptation_samples: AdaptationSamples,
    recursive_coarsening_ratio: RecursiveInteger,
    recursive_base_init_threshold: RecursiveInteger,
    recursive_n_embedding_epochs: RecursiveInteger,
    recursive_hub_selection: &str,
    recursive_anchor_sample_size: RecursiveInteger,
) -> PyResult<FittedUMAP> {
    let memory_limit_bytes = memory_limit_bytes
        .map(optional_memory_limit)
        .transpose()?
        .flatten();
    let init = match init {
        "spectral" => Init::Spectral,
        "random" => Init::Random,
        "recursive" => Init::Recursive,
        "auto" => Init::Auto,
        _ => {
            return Err(PyValueError::new_err(
                "init must be 'spectral', 'random', 'recursive', or 'auto'",
            ));
        }
    };
    let backend = match backend {
        "auto" => BackendChoice::Auto,
        "cpu" => BackendChoice::Cpu,
        "cuda" => BackendChoice::Cuda { device_index },
        _ => {
            return Err(PyValueError::new_err(
                "backend must be 'auto', 'cpu' or 'cuda'",
            ));
        }
    };
    let search = match neighbor_search {
        "auto" => NeighborSearch::Auto,
        "exact" => NeighborSearch::Exact,
        "approximate" => NeighborSearch::Approximate,
        _ => {
            return Err(PyValueError::new_err(
                "neighbor_search must be 'auto', 'exact' or 'approximate'",
            ));
        }
    };
    let config = UMAPConfig {
        n_components,
        n_neighbors,
        min_dist,
        spread,
        local_connectivity,
        set_op_mix_ratio,
        n_epochs,
        learning_rate,
        negative_sample_rate,
        repulsion_strength,
        optimizer: parse_optimizer(optimizer)?,
        negative_selection_range: negative_selection_range.0,
        negative_sample_scale: negative_sample_scale
            .map(optional_negative_sample_scale)
            .transpose()?
            .flatten(),
        exclude_graph_neighbors: exclude_graph_neighbors.0,
        negative_sample_scale_adaptation_samples: negative_sample_scale_adaptation_samples.0,
        init,
        recursive_init: RecursiveInitConfig {
            coarsening_ratio: recursive_coarsening_ratio.0,
            base_init_threshold: recursive_base_init_threshold.0,
            n_embedding_epochs: recursive_n_embedding_epochs.0,
            hub_selection: parse_hub_selection(recursive_hub_selection)?,
            anchor_sample_size: recursive_anchor_sample_size.0,
        },
        init_rescale: parse_init_rescale(init_rescale)?,
        init_scale_factor,
        random_state,
        transform_seed,
        transform_queue_size,
        unique,
        memory_limit_bytes,
    };
    config.validate().map_err(map_umap_error)?;
    let data: Vec<Vec<f32>> = x
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    py.detach(|| UMAP::new(config, backend, search).fit(&data))
        .map(|inner| FittedUMAP { inner })
        .map_err(map_umap_error)
}

fn parse_init_rescale(value: &str) -> PyResult<InitRescale> {
    match value {
        "isotropic" => Ok(InitRescale::Isotropic),
        "per_axis" => Ok(InitRescale::PerAxis),
        "graph_edge" => Ok(InitRescale::GraphEdge),
        _ => Err(PyValueError::new_err(
            "init_rescale must be 'isotropic', 'per_axis', or 'graph_edge'",
        )),
    }
}

pub(crate) fn optional_memory_limit(value: &Bound<'_, PyAny>) -> PyResult<Option<u64>> {
    if value.is_none() {
        return Ok(None);
    }
    let rejected =
        || PyValueError::new_err("memory_limit_bytes must be a positive integer or None");
    if value.is_instance_of::<PyBool>() || value.is_instance_of::<PyFloat>() {
        return Err(rejected());
    }
    match value.extract::<u64>() {
        Ok(parsed) if parsed > 0 => Ok(Some(parsed)),
        _ => Err(rejected()),
    }
}

#[pymodule]
fn _cuscan(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<FittedUMAP>()?;
    module.add_class::<hdbscan::FittedHDBSCAN>()?;
    module.add_class::<hdbscan::HDBSCANPredictor>()?;
    module.add_function(wrap_pyfunction!(fit_umap, module)?)?;
    module.add_function(wrap_pyfunction!(hdbscan::fit_hdbscan, module)?)?;
    module.add_function(wrap_pyfunction!(hdbscan::load_hdbscan_predictor, module)?)?;
    module.add_function(wrap_pyfunction!(
        hdbscan::estimate_hdbscan_resources,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(hdbscan::hdbscan_cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    Ok(())
}

#[cfg(test)]
mod umap_error_tests {
    use super::*;

    #[test]
    fn umap_capacity_maps_to_runtime_error_with_unchanged_message() {
        Python::initialize();
        Python::attach(|py| {
            let message = "ANN input copy buffer elements 4296000000 exceeds u32 addressing maximum 4294967295";
            let error = map_umap_error(UmapError::Capacity(message.into()));
            assert!(error.is_instance_of::<PyRuntimeError>(py));
            assert_eq!(error.value(py).to_string(), message);
        });
    }

    #[test]
    fn umap_memory_maps_to_memory_error_with_unchanged_message() {
        Python::initialize();
        Python::attach(|py| {
            let message = "memory limit exceeded";
            let error = map_umap_error(UmapError::Memory(message.into()));
            assert!(error.is_instance_of::<PyMemoryError>(py));
            assert_eq!(error.value(py).to_string(), message);
        });
    }

    #[test]
    fn memory_limit_requires_a_positive_integer_or_none() {
        Python::initialize();
        Python::attach(|py| {
            let values = PyDict::new(py);
            values.set_item("none", py.None()).unwrap();
            values.set_item("positive", u64::MAX).unwrap();
            values.set_item("zero", 0).unwrap();
            values.set_item("negative", -1).unwrap();
            values.set_item("bool", true).unwrap();
            values.set_item("float", 1.0).unwrap();
            assert_eq!(
                optional_memory_limit(&values.get_item("none").unwrap().unwrap()).unwrap(),
                None
            );
            assert_eq!(
                optional_memory_limit(&values.get_item("positive").unwrap().unwrap()).unwrap(),
                Some(u64::MAX)
            );
            for key in ["zero", "negative", "bool", "float"] {
                let error =
                    optional_memory_limit(&values.get_item(key).unwrap().unwrap()).unwrap_err();
                assert!(error.is_instance_of::<PyValueError>(py));
            }
        });
    }
}
