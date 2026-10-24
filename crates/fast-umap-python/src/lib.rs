use fast_umap::{
    BackendChoice, FittedUMAP as RustFittedUMAP, Init, NeighborSearch, UMAP, UMAPConfig, UmapError,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

pub(crate) mod hdbscan;

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
        metadata.set_item(
            "init",
            match c.init {
                Init::Random => "random",
                Init::Spectral => "spectral",
            },
        )?;
        metadata.set_item("random_state", c.random_state)?;
        metadata.set_item("transform_seed", c.transform_seed)?;
        metadata.set_item("transform_queue_size", c.transform_queue_size)?;
        metadata.set_item("unique", c.unique)?;
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
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<i32>>)> {
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
        UmapError::BackendUnavailable(_)
        | UmapError::Numerical(_)
        | UmapError::SpectralConvergence(_)
        | UmapError::Overflow(_) => PyRuntimeError::new_err(message),
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
        init,
        random_state: get!("random_state").extract()?,
        transform_seed: get!("transform_seed").extract()?,
        transform_queue_size: get!("transform_queue_size").extract()?,
        unique: get!("unique").extract()?,
    };
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
    let state = fast_umap::standard_umap::ModelState {
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

#[pyfunction(signature = (
  x, *, n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0,
  local_connectivity=1.0, set_op_mix_ratio=1.0, n_epochs=None,
  learning_rate=1.0, negative_sample_rate=5, repulsion_strength=1.0,
  init="spectral", random_state=None, transform_seed=42,
  transform_queue_size=4.0, backend="auto", device_index=0,
  neighbor_search="auto", unique=false))]
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
) -> PyResult<FittedUMAP> {
    let init = match init {
        "spectral" => Init::Spectral,
        "random" => Init::Random,
        _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
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
        init,
        random_state,
        transform_seed,
        transform_queue_size,
        unique,
    };
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

#[pymodule]
fn _fast_umap(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<FittedUMAP>()?;
    module.add_class::<hdbscan::FittedHDBSCAN>()?;
    module.add_function(wrap_pyfunction!(fit_umap, module)?)?;
    module.add_function(wrap_pyfunction!(hdbscan::fit_hdbscan, module)?)?;
    module.add_function(wrap_pyfunction!(
        hdbscan::estimate_hdbscan_resources,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(hdbscan::hdbscan_cuda_available, module)?)?;
    module.add_function(wrap_pyfunction!(load, module)?)?;
    Ok(())
}
