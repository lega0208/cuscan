use fast_hdbscan::{
    BackendChoice, ClusterSelectionMethod, FittedHDBSCAN as RustFittedHDBSCAN, HDBSCAN,
    HDBSCANPredictor as RustHDBSCANPredictor, HdbscanConfig, HdbscanError, HdbscanPredictorState,
    MstGraph, PredictionCoresState, PredictionHierarchyState, estimate_fit_resources,
};
use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyUserWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyTuple};

#[pyclass(frozen, unsendable)]
pub(crate) struct FittedHDBSCAN {
    inner: RustFittedHDBSCAN,
}

#[pymethods]
impl FittedHDBSCAN {
    fn to_predictor(&self, py: Python<'_>) -> PyResult<HDBSCANPredictor> {
        py.detach(|| self.inner.to_predictor())
            .map(|inner| HDBSCANPredictor { inner })
            .map_err(map_hdbscan_error)
    }

    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner
            .labels()
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[getter]
    fn probabilities<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.probabilities().to_vec().into_pyarray(py)
    }

    #[getter]
    fn condensed_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let tree = self.inner.condensed_tree();
        let parents = tree
            .parent
            .iter()
            .map(|&value| {
                i64::try_from(value)
                    .map_err(|_| PyOverflowError::new_err("tree node id exceeds np.intp"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let children = tree
            .child
            .iter()
            .map(|&value| {
                i64::try_from(value)
                    .map_err(|_| PyOverflowError::new_err("tree node id exceeds np.intp"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let arrays = (
            parents.into_pyarray(py),
            children.into_pyarray(py),
            tree.lambda.to_vec().into_pyarray(py),
            tree.child_size.to_vec().into_pyarray(py),
        );
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("names", "parent,child,lambda_val,child_size")?;
        kwargs.set_item("formats", "i8,i8,f4,f4")?;
        py.import("numpy")?
            .getattr("rec")?
            .getattr("fromarrays")?
            .call((arrays,), Some(&kwargs))
    }

    #[getter]
    fn single_linkage_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self.inner.linkage();
        let mut values = Vec::with_capacity(rows.len().saturating_mul(4));
        for row in rows {
            values.extend([
                row.left as f64,
                row.right as f64,
                row.distance,
                row.size as f64,
            ]);
        }
        Array2::from_shape_vec((rows.len(), 4), values)
            .map(|array| array.into_pyarray(py))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn minimum_spanning_tree<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let edges = self.inner.minimum_spanning_tree();
        let mut values = Vec::with_capacity(edges.len().saturating_mul(3));
        for edge in edges {
            values.extend([f64::from(edge.src), f64::from(edge.dst), edge.weight]);
        }
        Array2::from_shape_vec((edges.len(), 3), values)
            .map(|array| array.into_pyarray(py))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn neighbors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n_rows = self.inner.finite_index().len();
        // Fitting rejects zero finite rows before creating a fitted model.
        let width = self.inner.neighbors().len() / n_rows;
        let values = self
            .inner
            .neighbors()
            .iter()
            .map(|&value| i64::from(value))
            .collect();
        Array2::from_shape_vec((n_rows, width), values)
            .map(|array| array.into_pyarray(py))
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    #[getter]
    fn core_distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.core_distances().to_vec().into_pyarray(py)
    }

    #[getter]
    fn finite_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner
            .finite_index()
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[pyo3(signature = (eps))]
    fn dbscan_clustering<'py>(
        &self,
        py: Python<'py>,
        eps: f64,
    ) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let labels = py
            .detach(|| self.inner.dbscan_clustering(eps))
            .map_err(map_hdbscan_error)?;
        Ok(labels
            .into_iter()
            .map(i64::from)
            .collect::<Vec<_>>()
            .into_pyarray(py))
    }

    /// Predicts fixed fitted clusters for new float32 rows (original hdbscan `approximate_predict`).
    ///
    /// Returns `(labels, strengths)` or, with `return_connecting_points=True`,
    /// `(labels, strengths, connecting_points)`. Labels and connectors are int64 (`np.intp`);
    /// strengths are float64. A connector is the selected mutual-reachability neighbor, as an
    /// original training-row index; it is -1 only on the no-cluster-tree shortcut, which also
    /// emits exactly one `UserWarning` (at `stacklevel` relative to this call).
    ///
    /// The flag must be a Python `bool`; `None` and other truthy/falsy objects are rejected.
    #[pyo3(
        signature = (x, return_connecting_points = StrictBool(false), *, stacklevel = 1),
        text_signature = "($self, x, return_connecting_points=False, *, stacklevel=1)"
    )]
    fn approximate_predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        return_connecting_points: StrictBool,
        stacklevel: i32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let StrictBool(return_connecting_points) = return_connecting_points;
        if stacklevel < 1 {
            return Err(PyValueError::new_err(
                "stacklevel must be a positive integer",
            ));
        }
        let view = x.as_array();
        let (rows, features) = view.dim();
        if rows == 0 || features == 0 {
            return Err(PyValueError::new_err(
                "X must contain at least one row and one feature",
            ));
        }
        if features != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {features} features, but the model was fitted with {}",
                self.inner.n_features()
            )));
        }
        // Native sizing limits: element count and raw row IDs must stay representable.
        rows.checked_mul(features)
            .ok_or_else(|| PyOverflowError::new_err("X element count overflows usize"))?;
        if i32::try_from(rows).is_err() {
            return Err(PyOverflowError::new_err("X row count exceeds i32::MAX"));
        }
        if view.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("X must contain only finite values"));
        }
        // Owned rows: the borrowed NumPy view must not be touched while detached.
        let input = view
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let result = py
            .detach(|| self.inner.approximate_predict(&input))
            .map_err(map_hdbscan_error)?;
        prediction_output(py, result, return_connecting_points, stacklevel)
    }
}

#[pyclass(frozen, unsendable)]
pub(crate) struct HDBSCANPredictor {
    inner: RustHDBSCANPredictor,
}

#[pymethods]
impl HDBSCANPredictor {
    #[getter]
    fn n_features(&self) -> usize {
        self.inner.n_features()
    }

    #[pyo3(
        signature = (x, return_connecting_points = StrictBool(false), *, stacklevel = 1),
        text_signature = "($self, x, return_connecting_points=False, *, stacklevel=1)"
    )]
    fn approximate_predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
        return_connecting_points: StrictBool,
        stacklevel: i32,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let StrictBool(return_connecting_points) = return_connecting_points;
        if stacklevel < 1 {
            return Err(PyValueError::new_err(
                "stacklevel must be a positive integer",
            ));
        }
        let view = x.as_array();
        let (rows, features) = view.dim();
        if rows == 0 || features == 0 {
            return Err(PyValueError::new_err(
                "X must contain at least one row and one feature",
            ));
        }
        if features != self.inner.n_features() {
            return Err(PyValueError::new_err(format!(
                "X has {features} features, but the model was fitted with {}",
                self.inner.n_features()
            )));
        }
        rows.checked_mul(features)
            .ok_or_else(|| PyOverflowError::new_err("X element count overflows usize"))?;
        if i32::try_from(rows).is_err() {
            return Err(PyOverflowError::new_err("X row count exceeds i32::MAX"));
        }
        if view.iter().any(|value| !value.is_finite()) {
            return Err(PyValueError::new_err("X must contain only finite values"));
        }
        let input = view
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        let result = py
            .detach(|| self.inner.approximate_predict(&input))
            .map_err(map_hdbscan_error)?;
        prediction_output(py, result, return_connecting_points, stacklevel)
    }

    fn export_state<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
        let state = py
            .detach(|| self.inner.export_state())
            .map_err(map_hdbscan_error)?;
        let arrays = PyDict::new(py);
        let n_finite = state.hierarchy.finite_index.len();
        let cluster_nodes = state.hierarchy.node_ids.len();
        let (cores_status, cores_rank) = match &state.prediction_cores {
            PredictionCoresState::NotRequiredNoClusterTree => ("not_required_no_cluster_tree", 0),
            PredictionCoresState::Ready { rank, .. } => ("ready", *rank),
        };
        let hierarchy = state.hierarchy;
        let (
            geometry,
            finite_index,
            node_ids,
            node_parent_indices,
            node_birth_lambdas,
            node_labels,
            node_max_lambdas,
            point_node_indices,
            point_departure_lambdas,
            prediction_cores,
        ) = match state.prediction_cores {
            PredictionCoresState::NotRequiredNoClusterTree => {
                (None, None, None, None, None, None, None, None, None, None)
            }
            PredictionCoresState::Ready { values, .. } => (
                Some(state.geometry),
                Some(hierarchy.finite_index),
                Some(hierarchy.node_ids),
                Some(hierarchy.node_parent_indices),
                Some(hierarchy.node_birth_lambdas),
                Some(hierarchy.node_labels),
                Some(hierarchy.node_max_lambdas),
                Some(hierarchy.point_node_indices),
                Some(hierarchy.point_departure_lambdas),
                Some(values),
            ),
        };
        arrays.set_item("geometry", geometry.map(|v| v.into_pyarray(py)))?;
        arrays.set_item("finite_index", finite_index.map(|v| v.into_pyarray(py)))?;
        arrays.set_item("node_ids", node_ids.map(|v| v.into_pyarray(py)))?;
        arrays.set_item(
            "node_parent_indices",
            node_parent_indices.map(|v| v.into_pyarray(py)),
        )?;
        arrays.set_item(
            "node_birth_lambdas",
            node_birth_lambdas.map(|v| v.into_pyarray(py)),
        )?;
        arrays.set_item("node_labels", node_labels.map(|v| v.into_pyarray(py)))?;
        arrays.set_item(
            "node_max_lambdas",
            node_max_lambdas.map(|v| v.into_pyarray(py)),
        )?;
        arrays.set_item(
            "point_node_indices",
            point_node_indices.map(|v| v.into_pyarray(py)),
        )?;
        arrays.set_item(
            "point_departure_lambdas",
            point_departure_lambdas.map(|v| v.into_pyarray(py)),
        )?;
        arrays.set_item(
            "prediction_cores",
            prediction_cores.map(|v| v.into_pyarray(py)),
        )?;
        let metadata = PyDict::new(py);
        metadata.set_item("n_features", state.n_features)?;
        metadata.set_item("n_finite", n_finite)?;
        metadata.set_item("min_samples", state.min_samples)?;
        metadata.set_item("cluster_nodes", cluster_nodes)?;
        metadata.set_item("cores_status", cores_status)?;
        metadata.set_item("cores_rank", cores_rank)?;
        Ok((arrays, metadata))
    }
}

const PREDICTOR_ARRAY_KEYS: &[&str] = &[
    "geometry",
    "finite_index",
    "node_ids",
    "node_parent_indices",
    "node_birth_lambdas",
    "node_labels",
    "node_max_lambdas",
    "point_node_indices",
    "point_departure_lambdas",
    "prediction_cores",
];
const PREDICTOR_METADATA_KEYS: &[&str] = &[
    "n_features",
    "n_finite",
    "min_samples",
    "cluster_nodes",
    "cores_status",
    "cores_rank",
];

#[pyfunction(signature = (arrays, metadata, backend="auto", device_index=StrictDeviceIndex(0)))]
pub(crate) fn load_hdbscan_predictor(
    py: Python<'_>,
    arrays: &Bound<'_, PyDict>,
    metadata: &Bound<'_, PyDict>,
    backend: &str,
    device_index: StrictDeviceIndex,
) -> PyResult<HDBSCANPredictor> {
    let StrictDeviceIndex(device_index) = device_index;
    if device_index > isize::MAX as usize {
        return Err(PyOverflowError::new_err(
            "device_index exceeds addressable range",
        ));
    }
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
    require_exact_keys(arrays, PREDICTOR_ARRAY_KEYS, "arrays")?;
    require_exact_keys(metadata, PREDICTOR_METADATA_KEYS, "metadata")?;
    let scalar = |key: &str| -> PyResult<usize> {
        let value = metadata.get_item(key)?.expect("exact keys validated");
        if !value.get_type().is(py.get_type::<PyInt>()) {
            return Err(PyValueError::new_err(format!(
                "metadata '{key}' must be a nonnegative integer"
            )));
        }
        value
            .extract::<usize>()
            .map_err(|_| PyValueError::new_err(format!("metadata '{key}' is out of range")))
    };
    let n_features = scalar("n_features")?;
    let n_finite = scalar("n_finite")?;
    let min_samples = scalar("min_samples")?;
    let cluster_nodes = scalar("cluster_nodes")?;
    let cores_rank = scalar("cores_rank")?;
    for (key, value) in [
        ("n_features", n_features),
        ("n_finite", n_finite),
        ("min_samples", min_samples),
        ("cluster_nodes", cluster_nodes),
        ("cores_rank", cores_rank),
    ] {
        if value > isize::MAX as usize {
            return Err(PyOverflowError::new_err(format!(
                "metadata '{key}' exceeds addressable dimensions"
            )));
        }
    }
    let status = metadata
        .get_item("cores_status")?
        .expect("exact keys validated");
    if !status.get_type().is(py.get_type::<pyo3::types::PyString>()) {
        return Err(PyValueError::new_err(
            "metadata 'cores_status' must be a string",
        ));
    }
    let status = status
        .extract::<String>()
        .map_err(|_| PyValueError::new_err("metadata 'cores_status' must be a string"))?;
    if n_features == 0 || min_samples == 0 {
        return Err(PyValueError::new_err(
            "predictor feature count and min_samples must be positive",
        ));
    }
    let geometry_len = n_finite.checked_mul(n_features).ok_or_else(|| {
        PyOverflowError::new_err("predictor geometry element count overflows usize")
    })?;
    let mut total_bytes =
        checked_array_bytes(geometry_len, std::mem::size_of::<f32>(), "geometry")?;
    let clustered = match status.as_str() {
        "ready" => {
            if n_finite == 0 || cluster_nodes == 0 || cores_rank != min_samples {
                return Err(PyValueError::new_err(
                    "ready prediction cores have inconsistent dimensions or rank",
                ));
            }
            true
        }
        "not_required_no_cluster_tree" => {
            if n_finite != 0 || cluster_nodes != 0 || cores_rank != 0 {
                return Err(PyValueError::new_err(
                    "no-cluster-tree metadata is not canonical",
                ));
            }
            false
        }
        _ => {
            return Err(PyValueError::new_err(
                "metadata 'cores_status' has an unsupported value",
            ));
        }
    };
    if !clustered {
        for key in PREDICTOR_ARRAY_KEYS {
            if !dict_value(arrays, key)?.is_none() {
                return Err(PyValueError::new_err(
                    "no-cluster-tree state requires all arrays to be None",
                ));
            }
        }
        let state = HdbscanPredictorState {
            n_features,
            min_samples,
            geometry: Vec::new(),
            hierarchy: PredictionHierarchyState {
                finite_index: Vec::new(),
                node_ids: Vec::new(),
                node_parent_indices: Vec::new(),
                node_birth_lambdas: Vec::new(),
                node_labels: Vec::new(),
                node_max_lambdas: Vec::new(),
                point_node_indices: Vec::new(),
                point_departure_lambdas: Vec::new(),
            },
            prediction_cores: PredictionCoresState::NotRequiredNoClusterTree,
        };
        return import_predictor_state(py, state, backend);
    }
    if i32::try_from(n_finite).is_err() {
        return Err(PyOverflowError::new_err("n_finite exceeds i32::MAX"));
    }
    for (key, count, width) in [
        ("finite_index", n_finite, std::mem::size_of::<u32>()),
        ("node_ids", cluster_nodes, std::mem::size_of::<u64>()),
        (
            "node_parent_indices",
            cluster_nodes,
            std::mem::size_of::<u64>(),
        ),
        (
            "node_birth_lambdas",
            cluster_nodes,
            std::mem::size_of::<f64>(),
        ),
        ("node_labels", cluster_nodes, std::mem::size_of::<i32>()),
        (
            "node_max_lambdas",
            cluster_nodes,
            std::mem::size_of::<f64>(),
        ),
        ("point_node_indices", n_finite, std::mem::size_of::<u64>()),
        (
            "point_departure_lambdas",
            n_finite,
            std::mem::size_of::<f64>(),
        ),
        ("prediction_cores", n_finite, std::mem::size_of::<f64>()),
    ] {
        total_bytes = total_bytes
            .checked_add(checked_array_bytes(count, width, key)?)
            .ok_or_else(|| {
                PyOverflowError::new_err("predictor arrays total byte count overflows usize")
            })?;
    }
    if total_bytes > isize::MAX as usize {
        return Err(PyOverflowError::new_err(
            "predictor arrays exceed addressable size",
        ));
    }
    let array_len = |key: &str| -> PyResult<()> {
        let value = dict_value(arrays, key)?;
        if value.is_none() {
            return Err(PyValueError::new_err(format!(
                "clustered predictor requires array '{key}'"
            )));
        }
        Ok(())
    };
    for key in PREDICTOR_ARRAY_KEYS {
        array_len(key)?;
    }
    macro_rules! copy_array {
        ($key:literal, $ty:ty, $expected:expr) => {{
            let value = dict_value(arrays, $key)?;
            let array = value.extract::<PyReadonlyArray1<'_, $ty>>().map_err(|_| {
                PyValueError::new_err(concat!(
                    "array '",
                    $key,
                    "' must be one-dimensional with the exact dtype"
                ))
            })?;
            if array.shape()[0] != $expected {
                return Err(PyValueError::new_err(format!(
                    "array '{}' has an inconsistent length",
                    $key
                )));
            }
            array.as_array().iter().copied().collect::<Vec<$ty>>()
        }};
    }
    let geometry = copy_array!("geometry", f32, geometry_len);
    let finite_index = copy_array!("finite_index", u32, n_finite);
    let node_ids = copy_array!("node_ids", u64, cluster_nodes);
    let node_parent_indices = copy_array!("node_parent_indices", u64, cluster_nodes);
    let node_birth_lambdas = copy_array!("node_birth_lambdas", f64, cluster_nodes);
    let node_labels = copy_array!("node_labels", i32, cluster_nodes);
    let node_max_lambdas = copy_array!("node_max_lambdas", f64, cluster_nodes);
    let point_node_indices = copy_array!("point_node_indices", u64, n_finite);
    let point_departure_lambdas = copy_array!("point_departure_lambdas", f64, n_finite);
    let prediction_cores = copy_array!("prediction_cores", f64, n_finite);
    let state = HdbscanPredictorState {
        n_features,
        min_samples,
        geometry,
        hierarchy: PredictionHierarchyState {
            finite_index,
            node_ids,
            node_parent_indices,
            node_birth_lambdas,
            node_labels,
            node_max_lambdas,
            point_node_indices,
            point_departure_lambdas,
        },
        prediction_cores: PredictionCoresState::Ready {
            rank: cores_rank,
            values: prediction_cores,
        },
    };
    import_predictor_state(py, state, backend)
}

fn dict_value<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing array '{key}'")))
}

fn checked_array_bytes(count: usize, element_size: usize, name: &str) -> PyResult<usize> {
    let bytes = count.checked_mul(element_size).ok_or_else(|| {
        PyOverflowError::new_err(format!("array '{name}' byte count overflows usize"))
    })?;
    if bytes > isize::MAX as usize {
        return Err(PyOverflowError::new_err(format!(
            "array '{name}' exceeds addressable size"
        )));
    }
    Ok(bytes)
}

fn require_exact_keys(dict: &Bound<'_, PyDict>, expected: &[&str], name: &str) -> PyResult<()> {
    if dict.len() != expected.len()
        || dict.iter().any(|(key, _)| {
            key.extract::<String>()
                .map_or(true, |key| !expected.contains(&key.as_str()))
        })
    {
        return Err(PyValueError::new_err(format!(
            "{name} must contain exactly the documented keys"
        )));
    }
    Ok(())
}

fn import_predictor_state(
    py: Python<'_>,
    state: HdbscanPredictorState,
    backend: BackendChoice,
) -> PyResult<HDBSCANPredictor> {
    py.detach(move || RustHDBSCANPredictor::from_state(state, backend))
        .map(|inner| HDBSCANPredictor { inner })
        .map_err(map_hdbscan_error)
}

fn prediction_output<'py>(
    py: Python<'py>,
    result: fast_hdbscan::PredictionResult,
    return_connecting_points: bool,
    stacklevel: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let labels = result
        .labels
        .iter()
        .map(|&value| i64::from(value))
        .collect::<Vec<_>>()
        .into_pyarray(py);
    let probabilities = result.probabilities.into_pyarray(py);
    let tuple = if return_connecting_points {
        let connectors = result
            .connecting_points
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>()
            .into_pyarray(py);
        PyTuple::new(
            py,
            [
                labels.into_any(),
                probabilities.into_any(),
                connectors.into_any(),
            ],
        )?
    } else {
        PyTuple::new(py, [labels.into_any(), probabilities.into_any()])?
    };
    if result.no_cluster_tree {
        PyErr::warn(
            py,
            py.get_type::<PyUserWarning>().as_any(),
            c"Clusterer does not have any defined clusters, new data will be automatically predicted as noise.",
            stacklevel,
        )?;
    }
    Ok(tuple)
}

/// Exactly a Python `bool`: rejects `None`, integers and other truthy objects with `ValueError`.
struct StrictBool(bool);

/// Exact nonnegative Python int; rejects bool and integer-like objects.
pub(crate) struct StrictDeviceIndex(usize);

impl<'a, 'py> FromPyObject<'a, 'py> for StrictDeviceIndex {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if !obj.get_type().is(obj.py().get_type::<PyInt>()) {
            return Err(PyValueError::new_err(
                "device_index must be a nonnegative integer",
            ));
        }
        obj.extract::<usize>()
            .map(StrictDeviceIndex)
            .map_err(|_| PyValueError::new_err("device_index must be a nonnegative integer"))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for StrictBool {
    type Error = PyErr;

    fn extract(obj: pyo3::Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        obj.cast::<PyBool>()
            .map(|flag| StrictBool(flag.is_true()))
            .map_err(|_| PyValueError::new_err("return_connecting_points must be a boolean"))
    }
}

/// Parses the native config and backend shared by fit and the resource estimate bridge.
fn parse_hdbscan_kwargs(
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<(HdbscanConfig, BackendChoice)> {
    let kwargs = kwargs.ok_or_else(|| PyTypeError::new_err("keyword arguments are required"))?;
    const ALLOWED: &[&str] = &[
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "allow_single_cluster",
        "max_cluster_size",
        "cluster_selection_epsilon",
        "cluster_selection_persistence",
        "backend",
        "device_index",
        "reproducible",
        "knn_k",
    ];
    for (key, _) in kwargs.iter() {
        let key = key.extract::<String>()?;
        if !ALLOWED.contains(&key.as_str()) {
            return Err(PyTypeError::new_err(format!(
                "fit_hdbscan() got an unexpected keyword argument '{key}'"
            )));
        }
    }
    let min_cluster_size = kwargs
        .get_item("min_cluster_size")?
        .map(|value| value.extract::<usize>())
        .transpose()?
        .unwrap_or(5);
    let min_samples = kwargs
        .get_item("min_samples")?
        .map(|value| value.extract::<Option<usize>>())
        .transpose()?
        .unwrap_or(None);
    let cluster_selection_method = kwargs
        .get_item("cluster_selection_method")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "eom".to_owned());
    let allow_single_cluster = kwargs
        .get_item("allow_single_cluster")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    let max_cluster_size = kwargs
        .get_item("max_cluster_size")?
        .map(|value| value.extract::<Option<usize>>())
        .transpose()?
        .unwrap_or(None);
    let cluster_selection_epsilon = kwargs
        .get_item("cluster_selection_epsilon")?
        .map(|value| value.extract::<f64>())
        .transpose()?
        .unwrap_or(0.0);
    let cluster_selection_persistence = kwargs
        .get_item("cluster_selection_persistence")?
        .map(|value| value.extract::<f64>())
        .transpose()?
        .unwrap_or(0.0);
    let backend = kwargs
        .get_item("backend")?
        .map(|value| value.extract::<String>())
        .transpose()?
        .unwrap_or_else(|| "auto".to_owned());
    let device_index = kwargs
        .get_item("device_index")?
        .map(|value| value.extract::<usize>())
        .transpose()?
        .unwrap_or(0);
    let reproducible = kwargs
        .get_item("reproducible")?
        .map(|value| value.extract::<bool>())
        .transpose()?
        .unwrap_or(false);
    let knn_k = kwargs
        .get_item("knn_k")?
        .map(|value| optional_positive_integer(&value, "knn_k"))
        .transpose()?
        .unwrap_or(None);
    let backend = match backend {
        ref backend if backend == "auto" => BackendChoice::Auto,
        ref backend if backend == "cpu" => BackendChoice::Cpu,
        ref backend if backend == "cuda" => BackendChoice::Cuda { device_index },
        _ => {
            return Err(PyValueError::new_err(
                "backend must be 'auto', 'cpu' or 'cuda'",
            ));
        }
    };
    let selection_method = match cluster_selection_method.as_str() {
        "eom" => ClusterSelectionMethod::Eom,
        "leaf" => ClusterSelectionMethod::Leaf,
        _ => {
            return Err(PyValueError::new_err(
                "cluster_selection_method must be 'eom' or 'leaf'",
            ));
        }
    };
    let config = HdbscanConfig {
        min_cluster_size,
        min_samples,
        cluster_selection_method: selection_method,
        allow_single_cluster,
        max_cluster_size,
        cluster_selection_epsilon,
        cluster_selection_persistence,
        knn_k,
        reproducible,
        ..Default::default()
    };
    Ok((config, backend))
}

#[pyfunction(signature = (x, **kwargs))]
pub(crate) fn fit_hdbscan(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<FittedHDBSCAN> {
    let (config, backend) = parse_hdbscan_kwargs(kwargs)?;
    let input = x
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    py.detach(|| HDBSCAN::new(config, backend).fit(&input))
        .map(|inner| FittedHDBSCAN { inner })
        .map_err(map_hdbscan_error)
}

/// Pure resource estimate for a fit shape. Uses the same native config and backend parsing as
/// `fit_hdbscan`; it neither scans data nor initializes CUDA.
#[pyfunction(signature = (n_finite, features, **kwargs))]
pub(crate) fn estimate_hdbscan_resources<'py>(
    py: Python<'py>,
    n_finite: usize,
    features: usize,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let (config, backend) = parse_hdbscan_kwargs(kwargs)?;
    let estimate =
        estimate_fit_resources(n_finite, features, &config, backend).map_err(map_hdbscan_error)?;
    let result = PyDict::new(py);
    let graph = match estimate.resolved_graph {
        MstGraph::Complete => "complete",
        MstGraph::KnnGraph => "knn",
        // `resolve_graph` never returns Auto; reaching this is a planner bug, reported as such.
        MstGraph::Auto => {
            return Err(PyRuntimeError::new_err(
                "internal error: resource planner returned unresolved MstGraph::Auto",
            ));
        }
    };
    result.set_item("resolved_graph", graph)?;
    let knn = match &estimate.knn {
        None => py.None(),
        Some(knn) => {
            let entry = PyDict::new(py);
            entry.set_item("source", format!("{:?}", knn.source).to_lowercase())?;
            // effective_width: production graph width. retained_width: candidate width kept by
            // the source (ANN may exceed effective_width). Both are reported separately.
            entry.set_item("effective_width", knn.effective_width)?;
            entry.set_item("retained_width", knn.retained_width)?;
            entry.set_item("finite_rows", knn.finite_rows)?;
            entry.set_item("min_samples", knn.min_samples)?;
            entry.set_item("directed_edges", knn.directed_edges)?;
            entry.set_item("density", knn.density)?;
            let scenarios = PyList::empty(py);
            for scenario in &knn.scenarios {
                let item = PyDict::new(py);
                item.set_item("backend", format!("{:?}", scenario.backend).to_lowercase())?;
                item.set_item("host_peak_bytes", scenario.host_peak_bytes)?;
                item.set_item("device_peak_bytes", scenario.device_peak_bytes)?;
                scenarios.append(item)?;
            }
            entry.set_item("scenarios", scenarios)?;
            match &knn.advisory {
                None => entry.set_item("advisory", py.None())?,
                Some(advisory) => {
                    let item = PyDict::new(py);
                    item.set_item("threshold_bytes", advisory.threshold_bytes)?;
                    let triggers = PyList::empty(py);
                    for backend in &advisory.triggers {
                        triggers.append(format!("{backend:?}").to_lowercase())?;
                    }
                    item.set_item("triggers", triggers)?;
                    item.set_item("width_floor_applies", advisory.width_floor_applies)?;
                    entry.set_item("advisory", item)?;
                }
            }
            entry.into_any().unbind()
        }
    };
    result.set_item("knn", knn)?;
    Ok(result)
}

/// Accepts `None` or a strictly positive Python integer; rejects bool, float, zero and negatives.
fn optional_positive_integer(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<usize>> {
    if value.is_none() {
        return Ok(None);
    }
    let rejected = || PyValueError::new_err(format!("{name} must be a positive integer or None"));
    if value.is_instance_of::<PyBool>() || value.is_instance_of::<PyFloat>() {
        return Err(rejected());
    }
    match value.extract::<usize>() {
        Ok(parsed) if parsed > 0 => Ok(Some(parsed)),
        _ => Err(rejected()),
    }
}

#[pyfunction]
pub(crate) fn hdbscan_cuda_available(py: Python<'_>) -> bool {
    py.detach(|| {
        HDBSCAN::new(
            HdbscanConfig::default(),
            BackendChoice::Cuda { device_index: 0 },
        )
        .fit(&[vec![0.0_f32]])
        .is_ok()
    })
}

fn map_hdbscan_error(error: HdbscanError) -> PyErr {
    let message = error.to_string();
    match error {
        HdbscanError::InvalidConfig(_) | HdbscanError::InvalidInput(_) => {
            PyValueError::new_err(message)
        }
        HdbscanError::BackendUnavailable(_) | HdbscanError::Numerical(_) => {
            PyRuntimeError::new_err(message)
        }
        HdbscanError::Overflow(_) => PyOverflowError::new_err(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_each_hdbscan_error_to_the_documented_python_type() {
        Python::initialize();
        Python::attach(|py| {
            assert!(
                map_hdbscan_error(HdbscanError::InvalidConfig("bad config".into()))
                    .is_instance_of::<PyValueError>(py)
            );
            assert!(
                map_hdbscan_error(HdbscanError::InvalidInput("bad input".into()))
                    .is_instance_of::<PyValueError>(py)
            );
            assert!(
                map_hdbscan_error(HdbscanError::BackendUnavailable("no backend".into()))
                    .is_instance_of::<PyRuntimeError>(py)
            );
            assert!(
                map_hdbscan_error(HdbscanError::Numerical("numerical".into()))
                    .is_instance_of::<PyRuntimeError>(py)
            );
            assert!(
                map_hdbscan_error(HdbscanError::Overflow("test"))
                    .is_instance_of::<PyOverflowError>(py)
            );
        });
    }
}
