use serde::{Deserialize, Serialize};
use std::fmt;

/// Distance metric used to construct the high-dimensional k-NN graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Metric {
    Euclidean,
    EuclideanKNN,
    Manhattan,
    Cosine,
    Minkowski,
}

impl From<&str> for Metric {
    fn from(value: &str) -> Self {
        match value.to_lowercase().as_str() {
            "euclidean" => Self::Euclidean,
            "euclideanknn" | "euclidean_knn" => Self::EuclideanKNN,
            "manhattan" => Self::Manhattan,
            "cosine" => Self::Cosine,
            "minkowski" => Self::Minkowski,
            _ => panic!("Invalid metric type: {value}"),
        }
    }
}

impl fmt::Display for Metric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Euclidean => write!(f, "Euclidean"),
            Self::EuclideanKNN => write!(f, "Euclidean KNN"),
            Self::Manhattan => write!(f, "Manhattan"),
            Self::Cosine => write!(f, "cosine"),
            Self::Minkowski => write!(f, "minkowski"),
        }
    }
}

/// Configuration for manifold shape and embedding space properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifoldParams {
    pub min_dist: f32,
    pub spread: f32,
}

impl Default for ManifoldParams {
    fn default() -> Self {
        Self {
            min_dist: 0.1,
            spread: 1.0,
        }
    }
}

/// Configuration for k-nearest-neighbor graph construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphParams {
    pub n_neighbors: usize,
    pub local_connectivity: f64,
    pub bandwidth: f64,
    pub set_op_mix_ratio: f64,
    pub metric: Metric,
    pub normalized: bool,
    pub minkowski_p: f64,
}

impl Default for GraphParams {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            local_connectivity: 1.0,
            bandwidth: 1.0,
            set_op_mix_ratio: 1.0,
            metric: Metric::Euclidean,
            normalized: true,
            minkowski_p: 1.0,
        }
    }
}

/// Fit the UMAP kernel parameters `a` and `b` from `min_dist` and `spread`.
pub fn fit_ab(min_dist: f32, spread: f32) -> (f32, f32) {
    let n = 300;
    let x_max = 3.0 * spread;
    let xs: Vec<f32> = (0..n)
        .map(|i| (i as f32 + 0.5) / n as f32 * x_max)
        .collect();
    let ys: Vec<f32> = xs
        .iter()
        .map(|&x| {
            if x <= min_dist {
                1.0
            } else {
                (-(x - min_dist) / spread).exp()
            }
        })
        .collect();
    let residual = |a: f32, b: f32| -> f32 {
        xs.iter()
            .zip(ys.iter())
            .map(|(&x, &y)| {
                let pred = 1.0 / (1.0 + a * x.powf(2.0 * b));
                (pred - y) * (pred - y)
            })
            .sum::<f32>()
    };
    let mut best_a = 1.0f32;
    let mut best_b = 1.0f32;
    let mut best_err = f32::INFINITY;
    for ai in 1..=80 {
        let a = ai as f32 * 0.08;
        for bi in 1..=50 {
            let b = bi as f32 * 0.06;
            let err = residual(a, b);
            if err < best_err {
                best_err = err;
                best_a = a;
                best_b = b;
            }
        }
    }
    for _ in 0..100 {
        let step_a = best_a * 0.02;
        let step_b = best_b * 0.02;
        for &da in &[-step_a, 0.0, step_a] {
            for &db in &[-step_b, 0.0, step_b] {
                let a = (best_a + da).max(1e-4);
                let b = (best_b + db).max(1e-4);
                let err = residual(a, b);
                if err < best_err {
                    best_err = err;
                    best_a = a;
                    best_b = b;
                }
            }
        }
    }
    (best_a, best_b)
}
