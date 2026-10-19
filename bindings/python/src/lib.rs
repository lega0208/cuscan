use fast_umap::{
    BackendChoice, FittedUMAP as RustFittedUMAP, Init, NeighborSearch, UMAP, UMAPConfig, UmapError,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass(frozen, unsendable)]
struct FittedUMAP {
    inner: RustFittedUMAP,
}

#[pymethods]
impl FittedUMAP {
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

impl FittedUMAP {
    fn transform_owned<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        timed: bool,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Option<(u64, u64)>)> {
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
    module.add_function(wrap_pyfunction!(fit_umap, module)?)?;
    Ok(())
}
