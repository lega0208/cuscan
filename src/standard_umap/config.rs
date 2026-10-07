use super::{Result, UmapError};

/// Configuration for the nonparametric UMAP fit and transform pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct UMAPConfig {
    pub n_components: usize,
    pub n_neighbors: usize,
    pub min_dist: f32,
    pub spread: f32,
    pub local_connectivity: f32,
    pub set_op_mix_ratio: f32,
    pub n_epochs: Option<usize>,
    pub learning_rate: f32,
    pub negative_sample_rate: usize,
    pub repulsion_strength: f32,
    pub init: Init,
    pub random_state: Option<u64>,
    pub transform_seed: u64,
    pub transform_queue_size: f64,
}

impl Default for UMAPConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 15,
            min_dist: 0.1,
            spread: 1.0,
            local_connectivity: 1.0,
            set_op_mix_ratio: 1.0,
            n_epochs: None,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            repulsion_strength: 1.0,
            init: Init::Spectral,
            random_state: None,
            transform_seed: 42,
            transform_queue_size: 4.0,
        }
    }
}

impl UMAPConfig {
    /// Validate controls that do not depend on a particular input matrix.
    pub fn validate(&self) -> Result<()> {
        if self.n_components == 0 {
            return Err(UmapError::InvalidConfig(
                "n_components must be greater than zero".into(),
            ));
        }
        if self.n_neighbors < 2 {
            return Err(UmapError::InvalidConfig(
                "n_neighbors must be at least two".into(),
            ));
        }
        if !self.min_dist.is_finite() || self.min_dist < 0.0 {
            return Err(UmapError::InvalidConfig(
                "min_dist must be finite and nonnegative".into(),
            ));
        }
        if !self.spread.is_finite() || self.spread <= 0.0 {
            return Err(UmapError::InvalidConfig(
                "spread must be finite and greater than zero".into(),
            ));
        }
        if self.min_dist > self.spread {
            return Err(UmapError::InvalidConfig(
                "min_dist must not exceed spread".into(),
            ));
        }
        if !self.local_connectivity.is_finite() || self.local_connectivity < 0.0 {
            return Err(UmapError::InvalidConfig(
                "local_connectivity must be finite and nonnegative".into(),
            ));
        }
        if !self.set_op_mix_ratio.is_finite() || !(0.0..=1.0).contains(&self.set_op_mix_ratio) {
            return Err(UmapError::InvalidConfig(
                "set_op_mix_ratio must be between zero and one".into(),
            ));
        }
        if !self.learning_rate.is_finite() || self.learning_rate < 0.0 {
            return Err(UmapError::InvalidConfig(
                "learning_rate must be finite and nonnegative".into(),
            ));
        }
        if !self.repulsion_strength.is_finite() || self.repulsion_strength < 0.0 {
            return Err(UmapError::InvalidConfig(
                "repulsion_strength must be finite and nonnegative".into(),
            ));
        }
        validate_queue_size(self.transform_queue_size)
    }

    /// Validate exact/approximate applicability after Auto has been resolved.
    pub fn validate_search(
        &self,
        search: NeighborSearch,
        resolved_auto: Option<NeighborSearch>,
    ) -> Result<()> {
        validate_search_settings(search, self.transform_queue_size, resolved_auto)
    }
}

/// Fit the low-dimensional UMAP curve to the pinned 0.5.12 offset exponential target.
///
/// This uses the reference's 300 inclusive samples over `[0, 3 * spread]` and
/// minimizes unweighted squared residuals for `1 / (1 + a * x^(2b))`.
pub fn fit_curve(min_dist: f32, spread: f32) -> Result<(f32, f32)> {
    if !min_dist.is_finite() || min_dist < 0.0 {
        return Err(UmapError::InvalidConfig(
            "min_dist must be finite and nonnegative".into(),
        ));
    }
    if !spread.is_finite() || spread <= 0.0 {
        return Err(UmapError::InvalidConfig(
            "spread must be finite and greater than zero".into(),
        ));
    }

    let mut samples = [(0.0_f64, 0.0_f64); 300];
    for (index, (x, target)) in samples.iter_mut().enumerate() {
        let x_value = 3.0 * f64::from(spread) * index as f64 / 299.0;
        *x = x_value;
        *target = if x_value < f64::from(min_dist) {
            1.0
        } else {
            (-(x_value - f64::from(min_dist)) / f64::from(spread)).exp()
        };
    }

    // Optimize log(a), log(b) so every accepted iterate remains in the
    // positive parameter domain required by the reference curve.
    let (mut log_a, mut log_b) = (0.0_f64, 0.0_f64);
    let mut damping = 1.0e-3_f64;
    let mut objective = curve_objective(&samples, log_a, log_b)?;
    let mut converged = false;

    for _ in 0..500 {
        let (h_aa, h_ab, h_bb, g_a, g_b) = curve_normal_equations(&samples, log_a, log_b)?;
        let gradient_norm = g_a.abs().max(g_b.abs());
        // At the least-squares minimum, objective changes can fall below f64
        // resolution before the analytic gradient reaches machine-scale zero.
        // This threshold is still well below the precision of the f32 result.
        if gradient_norm <= 1.0e-8 {
            converged = true;
            break;
        }

        let damped_aa = h_aa + damping * h_aa.max(1.0);
        let damped_bb = h_bb + damping * h_bb.max(1.0);
        let determinant = damped_aa * damped_bb - h_ab * h_ab;
        if !determinant.is_finite() || determinant <= 0.0 {
            damping *= 10.0;
            if !damping.is_finite() || damping > 1.0e30 {
                break;
            }
            continue;
        }

        let delta_a = (damped_bb * g_a - h_ab * g_b) / determinant;
        let delta_b = (damped_aa * g_b - h_ab * g_a) / determinant;
        if !delta_a.is_finite() || !delta_b.is_finite() {
            return Err(UmapError::Numerical(
                "curve fitting produced a non-finite parameter step".into(),
            ));
        }
        let candidate_a = log_a - delta_a;
        let candidate_b = log_b - delta_b;
        let candidate_objective = curve_objective(&samples, candidate_a, candidate_b)?;
        if candidate_objective < objective {
            log_a = candidate_a;
            log_b = candidate_b;
            objective = candidate_objective;
            damping = (damping / 3.0).max(1.0e-15);
            if delta_a.abs().max(delta_b.abs()) <= 1.0e-11 {
                converged = true;
                break;
            }
        } else {
            damping *= 10.0;
            if !damping.is_finite() || damping > 1.0e30 {
                break;
            }
        }
    }

    let a = log_a.exp();
    let b = log_b.exp();
    if !converged || !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err(UmapError::Numerical(
            "curve fitting failed to converge to finite positive coefficients".into(),
        ));
    }
    let coefficients = (a as f32, b as f32);
    if !coefficients.0.is_finite() || !coefficients.1.is_finite() {
        return Err(UmapError::Numerical(
            "curve fitting coefficients do not fit in f32".into(),
        ));
    }
    Ok(coefficients)
}

fn curve_objective(samples: &[(f64, f64); 300], log_a: f64, log_b: f64) -> Result<f64> {
    let a = log_a.exp();
    let b = log_b.exp();
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Ok(f64::INFINITY);
    }
    let mut sum = 0.0;
    for &(x, target) in samples {
        let prediction = curve_value(x, a, b);
        let residual = prediction - target;
        sum += residual * residual;
    }
    if sum.is_finite() {
        Ok(sum)
    } else {
        Err(UmapError::Numerical(
            "curve fitting objective became non-finite".into(),
        ))
    }
}

fn curve_normal_equations(
    samples: &[(f64, f64); 300],
    log_a: f64,
    log_b: f64,
) -> Result<(f64, f64, f64, f64, f64)> {
    let a = log_a.exp();
    let b = log_b.exp();
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err(UmapError::Numerical(
            "curve fitting reached non-finite parameters".into(),
        ));
    }
    let mut h_aa = 0.0;
    let mut h_ab = 0.0;
    let mut h_bb = 0.0;
    let mut g_a = 0.0;
    let mut g_b = 0.0;
    for &(x, target) in samples {
        let prediction = curve_value(x, a, b);
        let residual = prediction - target;
        let (jacobian_a, jacobian_b) = if x == 0.0 {
            (0.0, 0.0)
        } else {
            let powered = (2.0 * b * x.ln()).exp();
            let scaled = a * powered;
            let derivative = -scaled / (1.0 + scaled).powi(2);
            (derivative, derivative * 2.0 * b * x.ln())
        };
        h_aa += jacobian_a * jacobian_a;
        h_ab += jacobian_a * jacobian_b;
        h_bb += jacobian_b * jacobian_b;
        g_a += jacobian_a * residual;
        g_b += jacobian_b * residual;
    }
    if [h_aa, h_ab, h_bb, g_a, g_b]
        .iter()
        .all(|value| value.is_finite())
    {
        Ok((h_aa, h_ab, h_bb, g_a, g_b))
    } else {
        Err(UmapError::Numerical(
            "curve fitting normal equations became non-finite".into(),
        ))
    }
}

fn curve_value(x: f64, a: f64, b: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        1.0 / (1.0 + a * (2.0 * b * x.ln()).exp())
    }
}

/// Select the compute backend. WGPU is deliberately not a supported choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendChoice {
    Flex,
    Cuda { device_index: usize },
}

impl BackendChoice {
    /// Check feature availability and, for CUDA, the requested visible device.
    pub fn validate(self) -> Result<()> {
        validate_backend_choice(self)
    }
}

/// Select the neighbor-search implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeighborSearch {
    Exact,
    Approximate,
    Auto,
}

/// Initialization strategy for the eventual embedding optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Init {
    Random,
    Spectral,
}

/// Validate input shape, optional fitted feature width, and every feature value.
pub fn validate_input(
    data: &[Vec<f32>],
    expected_features: Option<usize>,
) -> Result<(usize, usize)> {
    let first = data
        .first()
        .ok_or_else(|| UmapError::InvalidInput("input must contain at least one row".into()))?;
    let n_features = first.len();
    if n_features == 0 {
        return Err(UmapError::InvalidInput(
            "input must contain at least one feature".into(),
        ));
    }
    if let Some(expected) = expected_features {
        if n_features != expected {
            return Err(UmapError::InvalidInput(format!(
                "input has {n_features} features; expected {expected}"
            )));
        }
    }
    for (row_index, row) in data.iter().enumerate() {
        if row.len() != n_features {
            return Err(UmapError::InvalidInput(format!(
                "row {row_index} has {} features; expected {n_features}",
                row.len()
            )));
        }
        if let Some(column_index) = row.iter().position(|value| !value.is_finite()) {
            return Err(UmapError::InvalidInput(format!(
                "value at row {row_index}, feature {column_index} is not finite"
            )));
        }
    }
    Ok((data.len(), n_features))
}

/// Self-inclusive fit width and its non-self neighbor width after Python-compatible capping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FitNeighborWidths {
    pub self_inclusive: usize,
    pub nonself: usize,
}

/// Compute fit widths. One row is the reference-supported zero-neighbor special case.
pub fn fit_neighbor_widths(n_samples: usize, requested: usize) -> Result<FitNeighborWidths> {
    if n_samples == 0 {
        return Err(UmapError::InvalidInput(
            "n_samples must be greater than zero".into(),
        ));
    }
    if requested == 0 {
        return Err(UmapError::InvalidConfig(
            "n_neighbors must be greater than zero".into(),
        ));
    }
    if n_samples == 1 {
        return Ok(FitNeighborWidths {
            self_inclusive: 1,
            nonself: 0,
        });
    }
    let self_inclusive = requested.min(n_samples - 1);
    Ok(FitNeighborWidths {
        self_inclusive,
        nonself: self_inclusive - 1,
    })
}

/// Compute bipartite transform width (there is no query self slot).
pub fn query_neighbor_width(n_fitted_samples: usize, requested: usize) -> Result<usize> {
    if n_fitted_samples == 0 {
        return Err(UmapError::InvalidInput(
            "fitted data must contain at least one row".into(),
        ));
    }
    if requested == 0 {
        return Err(UmapError::InvalidConfig(
            "n_neighbors must be greater than zero".into(),
        ));
    }
    Ok(requested.min(n_fitted_samples))
}

/// Upper bounds on approximate transform distance evaluations and retained candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryBudget {
    pub candidates: usize,
    pub visits: usize,
}

/// Compute checked approximate-query budgets before applying the fitted-row caps.
pub fn approximate_query_budget(
    n_fitted_samples: usize,
    query_width: usize,
    queue_size: f64,
) -> Result<QueryBudget> {
    if n_fitted_samples == 0 || query_width == 0 || query_width > n_fitted_samples {
        return Err(UmapError::InvalidInput(
            "query width must be between one and the fitted row count".into(),
        ));
    }
    validate_queue_size(queue_size)?;
    let raw_candidates = queue_size * query_width as f64;
    if !raw_candidates.is_finite() || raw_candidates.ceil() >= usize::MAX as f64 {
        return Err(UmapError::Overflow("transform candidate budget"));
    }
    let scaled_candidates = raw_candidates.ceil() as usize;
    let candidates = n_fitted_samples.min(query_width.max(scaled_candidates));
    let log_factor = (n_fitted_samples.max(2) as f64).log2().ceil();
    if !log_factor.is_finite() || log_factor >= usize::MAX as f64 {
        return Err(UmapError::Overflow("transform visit multiplier"));
    }
    let visits_before_cap = candidates
        .checked_mul(log_factor as usize)
        .ok_or(UmapError::Overflow("transform visit budget"))?;
    Ok(QueryBudget {
        candidates,
        visits: n_fitted_samples.min(visits_before_cap),
    })
}

/// Validate the queue control, including exact-mode applicability.
pub fn validate_search_settings(
    search: NeighborSearch,
    queue_size: f64,
    resolved_auto: Option<NeighborSearch>,
) -> Result<()> {
    validate_queue_size(queue_size)?;
    let effective = match search {
        NeighborSearch::Exact => NeighborSearch::Exact,
        NeighborSearch::Approximate => NeighborSearch::Approximate,
        NeighborSearch::Auto => match resolved_auto {
            Some(mode @ (NeighborSearch::Exact | NeighborSearch::Approximate)) => mode,
            _ => {
                return Err(UmapError::InvalidConfig(
                    "Auto search requires a resolved Exact or Approximate mode".into(),
                ))
            }
        },
    };
    if effective == NeighborSearch::Exact && queue_size != 4.0 {
        return Err(UmapError::InvalidConfig(
            "custom transform_queue_size is inapplicable to exact search".into(),
        ));
    }
    Ok(())
}

fn validate_queue_size(queue_size: f64) -> Result<()> {
    if !queue_size.is_finite() || queue_size <= 0.0 {
        return Err(UmapError::InvalidConfig(
            "transform_queue_size must be finite and greater than zero".into(),
        ));
    }
    Ok(())
}

/// Validate explicit backend availability without silently substituting another backend.
pub fn validate_backend_choice(choice: BackendChoice) -> Result<()> {
    match choice {
        BackendChoice::Flex => Ok(()),
        BackendChoice::Cuda { device_index } => {
            #[cfg(feature = "cuda")]
            {
                let count = crate::cuda_backend::device_count();
                if device_index >= count {
                    return Err(UmapError::BackendUnavailable(format!("CUDA device index {device_index} is unavailable; {count} device(s) detected")));
                }
                Ok(())
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
