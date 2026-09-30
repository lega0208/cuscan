use std::sync::{Arc, Mutex};

use burn::backend::{Autodiff, NdArray};
use crossbeam_channel::unbounded;
use cubecl::wgpu::WgpuRuntime;
use fast_umap::{
    validate_fit_data, FittedUmap, GraphParams, ManifoldParams, Metric, OptimizationParams, Umap,
    UmapConfig,
};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

type GpuBackend = Autodiff<burn::backend::wgpu::CubeBackend<WgpuRuntime, f32, i32, u32>>;
type CpuBackend = Autodiff<NdArray<f32>>;

enum Inner {
    Gpu(FittedUmap<GpuBackend>),
    Cpu(FittedUmap<CpuBackend>),
}

#[pyclass(frozen, unsendable)]
struct FittedModel {
    inner: Mutex<Option<Inner>>,
    device: String,
}

#[pymethods]
impl FittedModel {
    #[getter]
    fn device(&self) -> &str {
        &self.device
    }

    #[getter]
    fn n_features(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("fitted model lock poisoned"))?;
        Ok(
            match inner
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("fitted model unavailable"))?
            {
                Inner::Gpu(model) => model.n_features(),
                Inner::Cpu(model) => model.n_features(),
            },
        )
    }

    #[getter]
    fn embedding<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("fitted model lock poisoned"))?;
        let embedding = match inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("fitted model unavailable"))?
        {
            Inner::Gpu(model) => model.embedding(),
            Inner::Cpu(model) => model.embedding(),
        };
        let rows = embedding.len();
        let columns = embedding.first().map_or(0, Vec::len);
        let values = embedding.iter().flatten().map(|&v| v as f32).collect();
        Ok(Array2::from_shape_vec((rows, columns), values)
            .expect("embedding is rectangular")
            .into_pyarray(py))
    }

    fn transform<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let array = x.as_array();
        let (rows, columns) = array.dim();
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("fitted model lock poisoned"))?;
        let model = inner
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("fitted model unavailable"))?;
        let expected = match &model {
            Inner::Gpu(model) => model.n_features(),
            Inner::Cpu(model) => model.n_features(),
        };
        let data: Vec<Vec<f64>> = array
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|&v| v as f64).collect())
            .collect();
        if columns != expected {
            *inner = Some(model);
            return Err(PyValueError::new_err(format!(
                "expected {expected} features, got {columns}"
            )));
        }
        let (model, output) = py.detach(move || {
            let output = match &model {
                Inner::Gpu(model) => model.transform(data),
                Inner::Cpu(model) => model.transform(data),
            };
            (model, output)
        });
        *inner = Some(model);
        let cols = output.first().map_or_else(
            || match inner.as_ref().expect("model restored") {
                Inner::Gpu(model) => model.config().n_components,
                Inner::Cpu(model) => model.config().n_components,
            },
            Vec::len,
        );
        let values = output.into_iter().flatten().map(|v| v as f32).collect();
        let shape = if rows == 0 { (0, cols) } else { (rows, cols) };
        Ok(Array2::from_shape_vec(shape, values)
            .expect("transform output is rectangular")
            .into_pyarray(py))
    }
}

fn fit_impl<B: fast_umap::backend::AutodiffBackend>(
    x: Vec<Vec<f32>>,
    config: UmapConfig,
) -> PyResult<FittedUmap<B>>
where
    B: burn::tensor::backend::AutodiffBackend,
    f32: num::Float + num::FromPrimitive + burn::tensor::Element,
{
    validate_fit_data(&config, &x).map_err(PyValueError::new_err)?;
    let (exit_tx, exit_rx) = unbounded();
    let interrupt = Arc::new(Mutex::new(None));
    let callback_interrupt = Arc::clone(&interrupt);
    let callback = Box::new(move |_progress: fast_umap::EpochProgress| {
        if let Err(error) = Python::attach(|py| py.check_signals()) {
            if let Ok(mut stored) = callback_interrupt.lock() {
                *stored = Some(error);
            }
            let _ = exit_tx.send(());
        }
    });
    let fitted = Umap::<B>::new(config).fit_with_progress(x, None, exit_rx, callback);
    if let Some(error) = interrupt
        .lock()
        .map_err(|_| PyRuntimeError::new_err("interrupt lock poisoned"))?
        .take()
    {
        return Err(error);
    }
    Ok(fitted)
}

#[pyfunction(signature = (x, *, device="gpu", n_components=2, n_neighbors=15, min_dist=0.1, spread=1.0, hidden_sizes=vec![100], n_epochs=100, learning_rate=0.001, repulsion_strength=1.0, negative_sample_rate=5, patience=None, timeout=None, seed=9999, verbose=false))]
#[allow(clippy::too_many_arguments)]
fn fit(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f32>,
    device: &str,
    n_components: usize,
    n_neighbors: usize,
    min_dist: f32,
    spread: f32,
    hidden_sizes: Vec<usize>,
    n_epochs: usize,
    learning_rate: f64,
    repulsion_strength: f32,
    negative_sample_rate: usize,
    patience: Option<i32>,
    timeout: Option<u64>,
    seed: u64,
    verbose: bool,
) -> PyResult<FittedModel> {
    let x = x.as_array();
    let data: Vec<Vec<f32>> = x.rows().into_iter().map(|row| row.to_vec()).collect();
    let mut config = UmapConfig::default();
    config.n_components = n_components;
    config.hidden_sizes = hidden_sizes;
    config.graph = GraphParams {
        n_neighbors,
        metric: Metric::Euclidean,
        ..GraphParams::default()
    };
    config.manifold = ManifoldParams { min_dist, spread };
    config.optimization = OptimizationParams {
        seed,
        n_epochs,
        learning_rate,
        repulsion_strength,
        neg_sample_rate: negative_sample_rate,
        patience,
        timeout,
        verbose,
        ..OptimizationParams::default()
    };
    let model = match device {
        "gpu" => Inner::Gpu(py.detach(|| fit_impl::<GpuBackend>(data, config))?),
        "cpu" => Inner::Cpu(py.detach(|| fit_impl::<CpuBackend>(data, config))?),
        _ => return Err(PyValueError::new_err("device must be 'gpu' or 'cpu'")),
    };
    Ok(FittedModel {
        inner: Mutex::new(Some(model)),
        device: device.to_owned(),
    })
}

#[pymodule]
fn _fast_umap(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_class::<FittedModel>()?;
    module.add_function(wrap_pyfunction!(fit, module)?)?;
    Ok(())
}
