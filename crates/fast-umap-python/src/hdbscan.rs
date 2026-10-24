use fast_hdbscan::{
    BackendChoice, ClusterSelectionMethod, FittedHDBSCAN as RustFittedHDBSCAN, HDBSCAN,
    HdbscanConfig, HdbscanError,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyOverflowError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat};

#[pyclass(frozen, unsendable)]
pub(crate) struct FittedHDBSCAN {
    inner: RustFittedHDBSCAN,
}

#[pymethods]
impl FittedHDBSCAN {
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
}

#[pyfunction(signature = (x, **kwargs))]
pub(crate) fn fit_hdbscan(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<FittedHDBSCAN> {
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
