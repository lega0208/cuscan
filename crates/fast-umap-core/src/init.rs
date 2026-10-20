use crate::{Init, Result, UMAPConfig, UmapError};
use faer::dyn_stack::MemStack;
use faer::linalg::evd::{self, ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
use faer::{Mat, Par, Spec};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::{BTreeMap, VecDeque};
use std::mem::MaybeUninit;

const RESIDUAL_TOLERANCE: f64 = 1.0e-6;
const ORTHOGONALITY_TOLERANCE: f64 = 1.0e-8;

#[path = "block_spectral.rs"]
mod block_spectral;

/// Construct the initialization embedding in row-major order.
///
/// The sparse graph is interpreted as weighted undirected affinity. If an edge
/// is supplied in both directions, the larger accumulated directed weight is
/// used, matching the symmetric graph expected by spectral initialization.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(
        skip_all,
        fields(
            rows = data.len(),
            edges = heads.len(),
            components = config.n_components,
            method = ?config.init,
        )
    )
)]
pub fn initialize(
    data: &[Vec<f32>],
    heads: &[usize],
    tails: &[usize],
    weights: &[f32],
    config: &UMAPConfig,
    seed: u64,
) -> Result<Vec<f32>> {
    config.validate()?;
    let (n, _) = crate::validate_input(data, None)?;
    n.checked_mul(config.n_components)
        .ok_or(UmapError::Overflow("embedding shape"))?;
    if heads.len() != tails.len() || heads.len() != weights.len() {
        return Err(UmapError::InvalidInput(
            "spectral graph head, tail, and weight arrays must have equal lengths".into(),
        ));
    }
    let graph = SparseGraph::from_edges(n, heads, tails, weights)?;
    if n == 1 {
        let output_size = n
            .checked_mul(config.n_components)
            .ok_or(UmapError::Overflow("embedding shape"))?;
        return Ok(vec![0.0; output_size]);
    }
    let mut rng = StdRng::seed_from_u64(seed);

    let mut embedding = match config.init {
        Init::Random => random_embedding(n, config.n_components, &mut rng)?,
        Init::Spectral => spectral_embedding(data, &graph, config.n_components, &mut rng)?,
    };
    if config.init == Init::Spectral {
        noisy_scale_coordinates(&mut embedding, n, config.n_components, &mut rng)?;
    }
    minmax_scale(&mut embedding, n, config.n_components)?;
    Ok(embedding)
}

#[derive(Debug)]
struct SparseGraph {
    rows: Vec<Vec<(usize, f64)>>,
    degrees: Vec<f64>,
    normalized: NormalizedCsr,
}

#[derive(Debug)]
struct NormalizedCsr {
    row_offsets: Vec<usize>,
    columns: Vec<usize>,
    values: Vec<f64>,
}

impl NormalizedCsr {
    fn from_rows(rows: &[Vec<(usize, f64)>], degrees: &[f64]) -> Result<Self> {
        let mut row_offsets = Vec::with_capacity(rows.len() + 1);
        let mut columns = Vec::new();
        let mut values = Vec::new();
        row_offsets.push(0);
        for (i, row) in rows.iter().enumerate() {
            if !degrees[i].is_finite() || (degrees[i] <= 0.0 && !row.is_empty()) {
                return Err(numerical("spectral graph has a zero or non-finite degree"));
            }
            for &(j, weight) in row {
                let normalized = weight / (degrees[i].sqrt() * degrees[j].sqrt());
                if !weight.is_finite()
                    || weight <= 0.0
                    || !normalized.is_finite()
                    || normalized <= 0.0
                {
                    return Err(numerical(
                        "spectral graph has an invalid normalized edge weight",
                    ));
                }
                columns.push(j);
                values.push(normalized);
            }
            row_offsets.push(columns.len());
        }
        Ok(Self {
            row_offsets,
            columns,
            values,
        })
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..x.len() {
            let mut value = x[i];
            for edge in self.row_offsets[i]..self.row_offsets[i + 1] {
                value += self.values[edge] * x[self.columns[edge]];
            }
            out[i] = value;
        }
    }
}

impl SparseGraph {
    fn from_edges(n: usize, heads: &[usize], tails: &[usize], weights: &[f32]) -> Result<Self> {
        if heads.len() != tails.len() || heads.len() != weights.len() {
            return Err(UmapError::InvalidInput(
                "spectral graph COO arrays have different lengths".into(),
            ));
        }
        let mut directed = BTreeMap::<(usize, usize), f64>::new();
        for (edge, ((&head, &tail), &weight)) in heads.iter().zip(tails).zip(weights).enumerate() {
            if head >= n || tail >= n || head == tail {
                return Err(UmapError::InvalidInput(format!(
                    "spectral graph edge {edge} has invalid endpoints ({head}, {tail}) for {n} rows"
                )));
            }
            if !weight.is_finite() || weight <= 0.0 {
                return Err(UmapError::InvalidInput(format!(
                    "spectral graph edge {edge} has invalid weight {weight}"
                )));
            }
            *directed.entry((head, tail)).or_default() += f64::from(weight);
        }

        let mut undirected = BTreeMap::<(usize, usize), f64>::new();
        for (&(head, tail), &weight) in &directed {
            let pair = (head.min(tail), head.max(tail));
            let reverse = directed.get(&(tail, head)).copied().unwrap_or(0.0);
            let symmetric_weight = weight.max(reverse);
            undirected
                .entry(pair)
                .and_modify(|current| *current = current.max(symmetric_weight))
                .or_insert(symmetric_weight);
        }
        let mut rows = vec![Vec::new(); n];
        for ((left, right), weight) in undirected {
            rows[left].push((right, weight));
            rows[right].push((left, weight));
        }
        for row in &mut rows {
            row.sort_unstable_by_key(|&(neighbor, _)| neighbor);
        }
        let degrees = rows
            .iter()
            .map(|row| row.iter().map(|&(_, weight)| weight).sum::<f64>())
            .collect::<Vec<f64>>();
        if degrees.iter().any(|degree| !degree.is_finite()) {
            return Err(numerical("spectral graph degree overflowed"));
        }
        let normalized = NormalizedCsr::from_rows(&rows, &degrees)?;
        Ok(Self {
            rows,
            degrees,
            normalized,
        })
    }

    fn n(&self) -> usize {
        self.rows.len()
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        self.normalized.shifted_apply(x, out);
    }

    fn components(&self) -> Vec<Vec<usize>> {
        let mut seen = vec![false; self.n()];
        let mut components = Vec::new();
        for start in 0..self.n() {
            if seen[start] {
                continue;
            }
            seen[start] = true;
            let mut queue = VecDeque::from([start]);
            let mut component = Vec::new();
            while let Some(vertex) = queue.pop_front() {
                component.push(vertex);
                for &(neighbor, _) in &self.rows[vertex] {
                    if !seen[neighbor] {
                        seen[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
            component.sort_unstable();
            components.push(component);
        }
        components
    }

    fn induced(&self, vertices: &[usize]) -> Result<Self> {
        let mut local = vec![usize::MAX; self.n()];
        for (index, &vertex) in vertices.iter().enumerate() {
            local[vertex] = index;
        }
        let mut rows = vec![Vec::new(); vertices.len()];
        for (index, &vertex) in vertices.iter().enumerate() {
            for &(neighbor, weight) in &self.rows[vertex] {
                let mapped = local[neighbor];
                if mapped != usize::MAX {
                    rows[index].push((mapped, weight));
                }
            }
            rows[index].sort_unstable_by_key(|&(neighbor, _)| neighbor);
        }
        let degrees = rows
            .iter()
            .map(|row| row.iter().map(|&(_, weight)| weight).sum::<f64>())
            .collect::<Vec<f64>>();
        if degrees.iter().any(|degree| !degree.is_finite()) {
            return Err(numerical("induced graph degree overflowed"));
        }
        let normalized = NormalizedCsr::from_rows(&rows, &degrees)?;
        Ok(Self {
            rows,
            degrees,
            normalized,
        })
    }
}

#[derive(Debug)]
struct CentroidGraph {
    centroids: Vec<Vec<f64>>,
    degrees: Vec<f64>,
    normalized: NormalizedCsr,
}

impl CentroidGraph {
    fn new(data: &[Vec<f32>], components: &[Vec<usize>]) -> Result<Self> {
        let n_features = data[0].len();
        let centroids = components
            .iter()
            .map(|component| {
                (0..n_features)
                    .map(|feature| {
                        component
                            .iter()
                            .map(|&vertex| f64::from(data[vertex][feature]))
                            .sum::<f64>()
                            / component.len() as f64
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut degrees = vec![0.0; components.len()];
        for i in 0..components.len() {
            for j in 0..components.len() {
                degrees[i] += centroid_affinity(&centroids[i], &centroids[j]);
            }
        }
        if degrees
            .iter()
            .any(|degree| !degree.is_finite() || *degree <= 0.0)
        {
            return Err(numerical("component-centroid affinity has invalid degree"));
        }
        for i in 0..components.len() {
            if components.len() > 1
                && (0..components.len())
                    .all(|j| i == j || centroid_affinity(&centroids[i], &centroids[j]) == 0.0)
            {
                return Err(numerical(
                    "component-centroid affinities underflowed and split the complete affinity operator",
                ));
            }
        }
        let n = components.len();
        let capacity = n
            .checked_mul(n)
            .ok_or(UmapError::Overflow("centroid affinity matrix"))?;
        let mut row_offsets = Vec::with_capacity(n + 1);
        let mut columns = Vec::with_capacity(capacity);
        let mut values = Vec::with_capacity(capacity);
        row_offsets.push(0);
        for i in 0..n {
            for j in 0..n {
                let value = centroid_affinity(&centroids[i], &centroids[j])
                    / (degrees[i].sqrt() * degrees[j].sqrt());
                if !value.is_finite() || value < 0.0 {
                    return Err(numerical(
                        "component-centroid affinity has invalid normalized weight",
                    ));
                }
                if value == 0.0 {
                    continue;
                }
                columns.push(j);
                values.push(value);
            }
            row_offsets.push(columns.len());
        }
        Ok(Self {
            centroids,
            degrees,
            normalized: NormalizedCsr {
                row_offsets,
                columns,
                values,
            },
        })
    }

    fn n(&self) -> usize {
        self.centroids.len()
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        self.normalized.shifted_apply(x, out);
    }
}

fn centroid_affinity(left: &[f64], right: &[f64]) -> f64 {
    let squared_distance = left
        .iter()
        .zip(right)
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>();
    (-squared_distance).exp()
}

#[derive(Clone, Copy, Debug)]
enum BaseOperator<'a> {
    Sparse(&'a SparseGraph),
    Centroids(&'a CentroidGraph),
}

impl BaseOperator<'_> {
    fn n(&self) -> usize {
        match self {
            Self::Sparse(graph) => graph.n(),
            Self::Centroids(graph) => graph.n(),
        }
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        match self {
            Self::Sparse(graph) => graph.shifted_apply(x, out),
            Self::Centroids(graph) => graph.shifted_apply(x, out),
        }
    }

    fn degrees(&self) -> &[f64] {
        match self {
            Self::Sparse(graph) => &graph.degrees,
            Self::Centroids(graph) => &graph.degrees,
        }
    }
}

#[cfg_attr(
    feature = "tracing",
    tracing::instrument(skip_all, fields(rows = graph.n(), dimensions))
)]
fn spectral_embedding(
    data: &[Vec<f32>],
    graph: &SparseGraph,
    dimensions: usize,
    rng: &mut StdRng,
) -> Result<Vec<f32>> {
    let components = graph.components();
    if components.len() == 1 {
        if graph.n()
            <= dimensions
                .checked_add(1)
                .ok_or(UmapError::Overflow("spectral mode count"))?
        {
            return Err(numerical(format!(
                "connected graph with {} vertices cannot provide {} spectral dimensions",
                graph.n(),
                dimensions
            )));
        }
        let modes = modes_with_source_fallback(
            BaseOperator::Sparse(graph),
            dimensions,
            rng,
            "connected graph",
        )?;
        let output_size = graph
            .n()
            .checked_mul(dimensions)
            .ok_or(UmapError::Overflow("embedding shape"))?;
        let mut embedding = vec![0.0; output_size];
        for row in 0..graph.n() {
            for dim in 0..dimensions {
                embedding[row * dimensions + dim] = modes[dim][row] as f32;
            }
        }
        return Ok(embedding);
    }

    let twice_dimensions = dimensions
        .checked_mul(2)
        .ok_or(UmapError::Overflow("component threshold"))?;
    let centers = if components.len() <= twice_dimensions {
        signed_axis_centers(components.len(), dimensions)
    } else {
        centroid_layout(data, &components, dimensions, rng)?
    };
    let output_size = graph
        .n()
        .checked_mul(dimensions)
        .ok_or(UmapError::Overflow("embedding shape"))?;
    let mut output = vec![0.0_f32; output_size];
    for (component_index, vertices) in components.iter().enumerate() {
        let center = &centers[component_index];
        let radius = nearest_center_distance(component_index, &centers)? / 2.0;
        let small_component_threshold = dimensions
            .checked_mul(2)
            .ok_or(UmapError::Overflow("component threshold"))?;
        let small_dimension_threshold = dimensions
            .checked_add(1)
            .ok_or(UmapError::Overflow("component threshold"))?;
        if vertices.len() < small_component_threshold || vertices.len() <= small_dimension_threshold
        {
            for &vertex in vertices {
                for dim in 0..dimensions {
                    output[vertex * dimensions + dim] =
                        (center[dim] + rng.random_range(-radius..radius)) as f32;
                }
            }
        } else {
            let local_graph = graph.induced(vertices)?;
            let modes = modes_with_source_fallback(
                BaseOperator::Sparse(&local_graph),
                dimensions,
                rng,
                "disconnected component",
            )?;
            let max_abs = modes
                .iter()
                .flat_map(|mode| mode.iter())
                .map(|value| value.abs())
                .fold(0.0_f64, f64::max);
            if !max_abs.is_finite() || max_abs == 0.0 {
                return Err(numerical(format!(
                    "component {component_index} spectral embedding has zero or non-finite span"
                )));
            }
            let expansion = radius / max_abs;
            for (local_row, &vertex) in vertices.iter().enumerate() {
                for dim in 0..dimensions {
                    output[vertex * dimensions + dim] =
                        (center[dim] + expansion * modes[dim][local_row]) as f32;
                }
            }
        }
    }
    Ok(output)
}

fn signed_axis_centers(count: usize, dimensions: usize) -> Vec<Vec<f64>> {
    let axes = count.div_ceil(2);
    (0..count)
        .map(|index| {
            let axis = index % axes;
            let sign = if index < axes { 1.0 } else { -1.0 };
            let mut center = vec![0.0; dimensions];
            center[axis] = sign;
            center
        })
        .collect()
}

fn nearest_center_distance(index: usize, centers: &[Vec<f64>]) -> Result<f64> {
    let nearest = centers
        .iter()
        .enumerate()
        .filter(|(other, _)| *other != index)
        .map(|(_, center)| {
            centers[index]
                .iter()
                .zip(center)
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .filter(|distance| *distance > 0.0 && distance.is_finite())
        .fold(f64::INFINITY, f64::min);
    if !nearest.is_finite() {
        return Err(numerical(format!(
            "component {index} has no distinct finite neighboring component center"
        )));
    }
    Ok(nearest)
}

fn centroid_layout(
    data: &[Vec<f32>],
    components: &[Vec<usize>],
    dimensions: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>> {
    let centroid_graph = CentroidGraph::new(data, components)?;
    let laplacian_modes = modes_with_source_fallback(
        BaseOperator::Centroids(&centroid_graph),
        dimensions,
        rng,
        "component centroid affinity",
    )?;
    let mut layout = vec![vec![0.0; dimensions]; components.len()];
    for component in 0..components.len() {
        for dim in 0..dimensions {
            layout[component][dim] = laplacian_modes[dim][component];
        }
    }
    let max_value = layout
        .iter()
        .flatten()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max_value.is_finite() || max_value == 0.0 {
        return Err(numerical(
            "centroid spectral layout has zero or non-finite maximum",
        ));
    }
    for value in layout.iter_mut().flatten() {
        *value /= max_value;
    }
    Ok(layout)
}

fn top_nontrivial_modes(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
) -> Result<Vec<Vec<f64>>> {
    solve_spectral(base, dimensions, rng).map(|solution| solution.modes)
}

struct SpectralSolution {
    modes: Vec<Vec<f64>>,
    #[cfg(all(test, feature = "tracing"))]
    diagnostics: Option<block_spectral::Diagnostics>,
    #[cfg(all(test, feature = "tracing"))]
    eigenvalues: Vec<f64>,
    #[cfg(all(test, feature = "tracing"))]
    eigenvectors: Vec<Vec<f64>>,
    #[cfg(all(test, feature = "tracing"))]
    raw_eigenvalues: Vec<f64>,
    #[cfg(all(test, feature = "tracing"))]
    raw_eigenvectors: Vec<Vec<f64>>,
    #[cfg(all(test, feature = "tracing"))]
    n_converged: usize,
    #[cfg(all(test, feature = "tracing"))]
    max_dim: usize,
    #[cfg(all(test, feature = "tracing"))]
    max_residual: f64,
    #[cfg(all(test, feature = "tracing"))]
    orthogonality: f64,
    #[cfg(all(test, feature = "tracing"))]
    dense: bool,
    #[cfg(all(test, feature = "tracing"))]
    requested: usize,
}

fn solve_spectral(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
) -> Result<SpectralSolution> {
    let n = base.n();
    let k = dimensions
        .checked_add(1)
        .ok_or(UmapError::Overflow("spectral mode count"))?;
    if n < 2 || k > n - 1 {
        return Err(numerical(format!(
            "spectral graph with {n} vertices cannot provide {k} requested eigenpairs"
        )));
    }
    if base
        .degrees()
        .iter()
        .any(|degree| !degree.is_finite() || *degree <= 0.0)
    {
        return Err(numerical("spectral graph has a zero or non-finite degree"));
    }
    let m = 64_usize.max(
        k.checked_mul(2)
            .ok_or(UmapError::Overflow("spectral max dimension"))?,
    );
    let cutoff = 64_usize.max(k * 2).max(m);
    let dense = n <= cutoff;
    let max_dim = m;
    let solve_count = if dense { n } else { k };
    let mut values = vec![0.0; if dense { n } else { solve_count }];
    let mut vectors = Mat::zeros(n, values.len());
    let par = Par::Seq;
    #[cfg(all(test, feature = "tracing"))]
    let mut diagnostics = None;
    let n_converged = if dense {
        let mut matrix = Mat::zeros(n, n);
        for col in 0..n {
            let mut basis = vec![0.0; n];
            basis[col] = 1.0;
            let mut applied = vec![0.0; n];
            base.shifted_apply(&basis, &mut applied);
            for row in 0..n {
                matrix[(row, col)] = applied[row];
            }
        }
        let params = Spec::<evd::SelfAdjointEvdParams, f64>::default();
        let req = self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, params);
        let size = req
            .size_bytes()
            .checked_add(req.align_bytes())
            .ok_or_else(|| numerical("dense EVD scratch size overflow"))?;
        let mut scratch = vec![MaybeUninit::<u8>::uninit(); size];
        let mut eigenvalue_matrix = Mat::zeros(n, n);
        self_adjoint_evd(
            matrix.as_ref(),
            eigenvalue_matrix.as_mut().diagonal_mut(),
            Some(vectors.as_mut()),
            par,
            MemStack::new(&mut scratch),
            params,
        )
        .map_err(|error| {
            UmapError::Numerical(format!("dense self-adjoint EVD failed: {error:?}"))
        })?;
        for i in 0..n {
            values[i] = eigenvalue_matrix[(i, i)];
        }
        n
    } else {
        if max_dim >= n {
            return Err(numerical(
                "sparse spectral block dimension must be smaller than n",
            ));
        }
        let solution = block_spectral::solve(base, k, max_dim, rng)?;
        convergence_status(solution.pairs.len(), k)?;
        #[cfg(all(test, feature = "tracing"))]
        {
            diagnostics = Some(solution.diagnostics);
        }
        for (col, (value, vector)) in solution.pairs.into_iter().enumerate() {
            values[col] = value;
            vectors.col_as_slice_mut(col).copy_from_slice(&vector);
        }
        k
    };
    let count = n_converged.min(values.len());
    let mut pairs = (0..count)
        .map(|i| {
            (
                values[i],
                (0..n).map(|j| vectors[(j, i)]).collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
    validate_raw_eigenpairs(base, &pairs)?;
    #[cfg(all(test, feature = "tracing"))]
    let raw_eigenvalues = pairs.iter().map(|pair| pair.0).collect::<Vec<_>>();
    #[cfg(all(test, feature = "tracing"))]
    let raw_eigenvectors = pairs.iter().map(|pair| pair.1.clone()).collect::<Vec<_>>();
    let trivial = base.degrees().iter().map(|d| d.sqrt()).collect::<Vec<_>>();
    let trivial_norm = l2_norm(&trivial);
    if !trivial_norm.is_finite() || trivial_norm == 0.0 {
        return Err(numerical(
            "locked trivial eigenvector has zero or non-finite norm",
        ));
    }
    let mut retained = Vec::new();
    for (value, vector) in pairs {
        let overlap = dot(&vector, &trivial).abs() / trivial_norm;
        if !overlap.is_finite() {
            return Err(numerical(
                "non-finite overlap with locked trivial eigenvector",
            ));
        }
        if overlap > 1.0 - 1.0e-6 {
            continue;
        }
        retained.push((value, vector));
    }
    if retained.len() < dimensions {
        return Err(UmapError::SpectralConvergence(format!(
            "only {} nontrivial eigenpairs available; requested {dimensions}",
            retained.len()
        )));
    }
    let mut max_residual: f64 = 0.0;
    let mut orthogonality: f64 = 0.0;
    for (i, (value, vector)) in retained.iter().enumerate() {
        let mut applied = vec![0.0; n];
        base.shifted_apply(vector, &mut applied);
        let residual = l2_norm(
            &applied
                .iter()
                .zip(vector)
                .map(|(a, v)| a - value * v)
                .collect::<Vec<_>>(),
        ) / (2.0 - value).abs().max(1.0);
        max_residual = max_residual.max(residual);
        for (j, (_, other)) in retained.iter().enumerate() {
            let error = dot(vector, other) - if i == j { 1.0 } else { 0.0 };
            orthogonality = orthogonality.max(error.abs());
        }
    }
    if !max_residual.is_finite() {
        return Err(numerical(format!(
            "independent spectral residual is non-finite: {max_residual}"
        )));
    }
    if max_residual > RESIDUAL_TOLERANCE {
        return Err(UmapError::SpectralConvergence(format!(
            "independent residual {max_residual} exceeds {RESIDUAL_TOLERANCE}"
        )));
    }
    if !orthogonality.is_finite() {
        return Err(numerical(format!(
            "independent spectral orthogonality error is non-finite: {orthogonality}"
        )));
    }
    if orthogonality > ORTHOGONALITY_TOLERANCE {
        return Err(UmapError::SpectralConvergence(format!(
            "orthogonality error {orthogonality} exceeds {ORTHOGONALITY_TOLERANCE}"
        )));
    }
    #[cfg(all(test, feature = "tracing"))]
    let sorted_nontrivial = retained.iter().map(|pair| pair.0).collect::<Vec<_>>();
    let modes = retained
        .iter()
        .take(dimensions)
        .map(|pair| pair.1.clone())
        .collect();
    #[cfg(feature = "tracing")]
    tracing::debug!(
        n_converged,
        requested = k,
        max_dim,
        max_residual,
        orthogonality,
        dense,
        "spectral solver diagnostics"
    );
    Ok(SpectralSolution {
        modes,
        #[cfg(all(test, feature = "tracing"))]
        diagnostics,
        #[cfg(all(test, feature = "tracing"))]
        eigenvalues: sorted_nontrivial,
        #[cfg(all(test, feature = "tracing"))]
        eigenvectors: retained.into_iter().map(|pair| pair.1).collect(),
        #[cfg(all(test, feature = "tracing"))]
        raw_eigenvalues,
        #[cfg(all(test, feature = "tracing"))]
        raw_eigenvectors,
        #[cfg(all(test, feature = "tracing"))]
        n_converged,
        #[cfg(all(test, feature = "tracing"))]
        max_dim,
        #[cfg(all(test, feature = "tracing"))]
        max_residual,
        #[cfg(all(test, feature = "tracing"))]
        orthogonality,
        #[cfg(all(test, feature = "tracing"))]
        dense,
        #[cfg(all(test, feature = "tracing"))]
        requested: k,
    })
}

fn validate_raw_eigenpairs(base: BaseOperator<'_>, pairs: &[(f64, Vec<f64>)]) -> Result<()> {
    let n = base.n();
    if pairs.is_empty() {
        return Err(numerical("spectral eigensolver returned no eigenpairs"));
    }
    let mut norms = Vec::with_capacity(pairs.len());
    let mut max_residual = (0.0_f64, 0_usize);
    let mut max_diagonal = (0.0_f64, 0_usize);
    let mut max_offdiagonal = (0.0_f64, 0_usize, 0_usize);
    for (i, (eigenvalue, vector)) in pairs.iter().enumerate() {
        if !eigenvalue.is_finite() {
            return Err(numerical(format!(
                "raw eigenpair {i} has non-finite eigenvalue {eigenvalue}"
            )));
        }
        if *eigenvalue < -1.0e-12 || *eigenvalue > 2.0 + 1.0e-12 {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue {eigenvalue} is outside the shifted [0,2] spectrum"
            )));
        }
        if vector.len() != n {
            return Err(numerical(format!(
                "raw eigenpair {i} has vector length {}, expected {n}",
                vector.len()
            )));
        }
        if let Some((coordinate, value)) = vector
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} has non-finite coordinate {coordinate}={value}"
            )));
        }
        let norm = l2_norm(vector);
        if !norm.is_finite() {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} has non-finite norm {norm}"
            )));
        }
        if norm == 0.0 {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} has zero norm"
            )));
        }
        norms.push(norm);
        let mut applied = vec![0.0; n];
        base.shifted_apply(vector, &mut applied);
        if let Some((coordinate, value)) = applied
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} operator result at coordinate {coordinate} is non-finite: {value}"
            )));
        }
        let residual = l2_norm(
            &applied
                .iter()
                .zip(vector)
                .map(|(actual, value)| actual - eigenvalue * value)
                .collect::<Vec<_>>(),
        ) / (2.0 - eigenvalue).abs().max(1.0);
        if !residual.is_finite() {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} has non-finite residual {residual}"
            )));
        }
        if residual > max_residual.0 {
            max_residual = (residual, i);
        }
        let diagonal_error = (dot(vector, vector) - 1.0).abs();
        if !diagonal_error.is_finite() {
            return Err(numerical(format!(
                "raw eigenpair {i} eigenvalue={eigenvalue} has non-finite diagonal Gram error {diagonal_error}"
            )));
        }
        if diagonal_error > max_diagonal.0 {
            max_diagonal = (diagonal_error, i);
        }
        for (j, (other_eigenvalue, other_vector)) in pairs.iter().take(i).enumerate() {
            let gram = dot(vector, other_vector);
            if !gram.is_finite() {
                return Err(numerical(format!(
                    "raw Gram entry ({j},{i}) for eigenvalues ({},{eigenvalue}) is non-finite: {gram}",
                    other_eigenvalue
                )));
            }
            if gram.abs() > max_offdiagonal.0 {
                max_offdiagonal = (gram.abs(), j, i);
            }
        }
    }
    let min_norm = norms.iter().copied().fold(f64::INFINITY, f64::min);
    let max_norm = norms.iter().copied().fold(0.0_f64, f64::max);
    if max_residual.0 > RESIDUAL_TOLERANCE
        || max_diagonal.0 > ORTHOGONALITY_TOLERANCE
        || max_offdiagonal.0 > ORTHOGONALITY_TOLERANCE
    {
        let (r, ri) = max_residual;
        let (d, di) = max_diagonal;
        let (o, oi, oj) = max_offdiagonal;
        return Err(UmapError::SpectralConvergence(format!(
            "raw eigenpair validation failed across {} pairs: max_residual={r} at pair {ri} eigenvalue={} (limit {RESIDUAL_TOLERANCE}); min_norm={min_norm}; max_norm={max_norm}; max_diagonal_gram_error={d} at pair {di} eigenvalue={} (limit {ORTHOGONALITY_TOLERANCE}); max_offdiagonal_dot_abs={o} at pair indices i={oi},j={oj} with shifted eigenvalues lambda_i={},lambda_j={} (limit {ORTHOGONALITY_TOLERANCE})",
            pairs.len(),
            pairs[ri].0,
            pairs[di].0,
            pairs[oi].0,
            pairs[oj].0
        )));
    }
    Ok(())
}

/// Private transport used only by the ignored spectral spike test.
#[cfg(all(test, feature = "tracing"))]
#[derive(serde::Serialize)]
struct SpectralProbeResult {
    production_eigenvalues: Vec<f64>,
    production_eigenvectors: Vec<Vec<f64>>,
    eigenvalues: Vec<f64>,
    eigenvectors: Vec<Vec<f64>>,
    n_converged: usize,
    diagnostic_n_converged: usize,
    requested: usize,
    diagnostic_requested: usize,
    max_dim: usize,
    production_dense: bool,
    diagnostic_dense: bool,
    max_residual: f64,
    orthogonality: f64,
    eigengap: Option<f64>,
    cluster_start: usize,
    cluster_end: usize,
    fallback: bool,
    production_diagnostics: Option<block_spectral::Diagnostics>,
    diagnostic_diagnostics: Option<block_spectral::Diagnostics>,
}

#[cfg(all(test, feature = "tracing"))]
fn spectral_probe(
    n: usize,
    heads: &[usize],
    tails: &[usize],
    weights: &[f32],
    dimensions: usize,
    seed: u64,
) -> Result<SpectralProbeResult> {
    if dimensions == 0 || dimensions >= n.saturating_sub(1) {
        return Err(numerical(
            "spectral probe dimensions must be positive and less than n-1",
        ));
    }
    let graph = SparseGraph::from_edges(n, heads, tails, weights)?;
    // First solve the exact production request. Preserve its routing diagnostics:
    // in particular dims=40,n=83 stays sparse with k=41 and M=82.
    let production = solve_spectral(
        BaseOperator::Sparse(&graph),
        dimensions,
        &mut StdRng::seed_from_u64(seed),
    )?;
    let production_dense = production.dense;
    let production_max_dim = production.max_dim;
    let production_requested = production.requested;
    let production_n_converged = production.n_converged;
    let production_diagnostics = production.diagnostics;
    let production_eigenvalues = production.raw_eigenvalues;
    let production_eigenvectors = production.raw_eigenvectors;

    // Supplement only the diagnostic subspace. An increased request can cross
    // the dense cutoff; that route is reported separately and is not production.
    let mut diagnostic_modes = dimensions.saturating_add(1).min(n.saturating_sub(2));
    let solution = loop {
        let solution = solve_spectral(
            BaseOperator::Sparse(&graph),
            diagnostic_modes,
            &mut StdRng::seed_from_u64(seed),
        )?;
        let boundary_index = dimensions - 1;
        let boundary = solution.eigenvalues[boundary_index];
        let last = solution.eigenvalues.len() - 1;
        let separated = last > boundary_index
            && (boundary - solution.eigenvalues[last]).abs() / boundary.abs().max(1.0) > 1.0e-6;
        if separated || solution.eigenvalues.len() == n - 1 || diagnostic_modes >= n - 2 {
            break solution;
        }
        diagnostic_modes = (diagnostic_modes + 8).min(n - 2);
    };
    let boundary_index = dimensions - 1;
    let boundary = solution.eigenvalues[boundary_index];
    let cluster_start = (0..=boundary_index)
        .rev()
        .find(|&index| {
            (solution.eigenvalues[index] - boundary).abs() / boundary.abs().max(1.0) > 1.0e-6
        })
        .map_or(0, |index| index + 1);
    let cluster_end = (boundary_index + 1..solution.eigenvalues.len())
        .find(|&index| {
            (solution.eigenvalues[index] - boundary).abs() / boundary.abs().max(1.0) > 1.0e-6
        })
        .unwrap_or(solution.eigenvalues.len());
    let gap = if dimensions < solution.eigenvalues.len() {
        Some(
            (solution.eigenvalues[dimensions - 1] - solution.eigenvalues[dimensions]).abs()
                / solution.eigenvalues[dimensions - 1].abs().max(1.0),
        )
    } else {
        None
    };
    let mut eigenvalues = solution.eigenvalues[..dimensions].to_vec();
    eigenvalues.extend_from_slice(&solution.eigenvalues[dimensions..cluster_end]);
    let mut eigenvectors = solution.eigenvectors[..dimensions].to_vec();
    eigenvectors.extend_from_slice(&solution.eigenvectors[dimensions..cluster_end]);
    let diagnostic_diagnostics = solution.diagnostics;
    Ok(SpectralProbeResult {
        production_eigenvalues,
        production_eigenvectors,
        eigenvalues,
        eigenvectors,
        n_converged: production_n_converged,
        diagnostic_n_converged: solution.n_converged,
        requested: production_requested,
        diagnostic_requested: solution.requested,
        max_dim: production_max_dim,
        production_dense,
        diagnostic_dense: solution.dense,
        max_residual: solution.max_residual,
        orthogonality: solution.orthogonality,
        eigengap: gap,
        cluster_start,
        cluster_end,
        fallback: false,
        production_diagnostics,
        diagnostic_diagnostics,
    })
}

fn modes_with_source_fallback(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
    context: &str,
) -> Result<Vec<Vec<f64>>> {
    modes_with_source_fallback_and_warning(base, dimensions, rng, context, |message| {
        eprintln!("[fast-umap] WARNING: {message}");
    })
}

fn modes_with_source_fallback_and_warning(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
    context: &str,
    mut warn: impl FnMut(&str),
) -> Result<Vec<Vec<f64>>> {
    let result = top_nontrivial_modes(base, dimensions, rng);
    fallback_on_spectral_convergence(result, base.n(), dimensions, rng, context, &mut warn)
}

fn fallback_on_spectral_convergence(
    result: Result<Vec<Vec<f64>>>,
    row_count: usize,
    dimensions: usize,
    rng: &mut StdRng,
    context: &str,
    mut warn: impl FnMut(&str),
) -> Result<Vec<Vec<f64>>> {
    match result {
        Ok(modes) => Ok(modes),
        Err(UmapError::SpectralConvergence(reason)) => {
            #[cfg(feature = "tracing")]
            tracing::warn!(spectral_fallback = true, context = %context, reason = %reason, "spectral solver used seeded random modes");
            warn(&format!(
                "spectral initialization did not converge for {context} ({reason}); using seeded uniform initialization"
            ));
            Ok(random_modes(row_count, dimensions, rng))
        }
        Err(error) => Err(error),
    }
}

fn random_modes(rows: usize, dimensions: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    (0..dimensions)
        .map(|_| {
            (0..rows)
                .map(|_| rng.random_range(-10.0_f64..10.0_f64))
                .collect()
        })
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
}

fn convergence_status(actual: usize, requested: usize) -> Result<()> {
    if actual < requested {
        Err(UmapError::SpectralConvergence(format!(
            "spectral solver converged {actual} of {requested} requested eigenpairs"
        )))
    } else {
        Ok(())
    }
}

fn l2_norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

fn gaussian(rng: &mut StdRng) -> f64 {
    let first = rng.random_range(f64::MIN_POSITIVE..1.0);
    let second = rng.random_range(0.0..1.0);
    (-2.0 * first.ln()).sqrt() * (std::f64::consts::TAU * second).cos()
}

fn random_embedding(n: usize, dimensions: usize, rng: &mut StdRng) -> Result<Vec<f32>> {
    let size = n
        .checked_mul(dimensions)
        .ok_or(UmapError::Overflow("embedding shape"))?;
    Ok((0..size)
        .map(|_| rng.random_range(-10.0_f32..10.0_f32))
        .collect())
}

fn noisy_scale_coordinates(
    embedding: &mut [f32],
    n: usize,
    dimensions: usize,
    rng: &mut StdRng,
) -> Result<()> {
    let max_abs = embedding
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);
    if !max_abs.is_finite() || max_abs == 0.0 {
        return Err(numerical(
            "spectral embedding has zero or non-finite global span",
        ));
    }
    for value in embedding.iter_mut() {
        let noise = gaussian(rng) * 1.0e-4;
        *value = *value / max_abs * 10.0 + noise as f32;
    }
    let expected = n
        .checked_mul(dimensions)
        .ok_or(UmapError::Overflow("embedding shape"))?;
    if embedding.len() != expected || embedding.iter().any(|value| !value.is_finite()) {
        return Err(numerical(
            "scaled spectral coordinates are non-finite or malformed",
        ));
    }
    Ok(())
}

fn minmax_scale(embedding: &mut [f32], n: usize, dimensions: usize) -> Result<()> {
    if embedding.len()
        != n.checked_mul(dimensions)
            .ok_or(UmapError::Overflow("embedding shape"))?
    {
        return Err(UmapError::InvalidInput(
            "embedding storage does not match its row-major shape".into(),
        ));
    }
    for dimension in 0..dimensions {
        let mut minimum = f32::INFINITY;
        let mut maximum = f32::NEG_INFINITY;
        for row in 0..n {
            let value = embedding[row * dimensions + dimension];
            if !value.is_finite() {
                return Err(numerical(format!(
                    "embedding coordinate ({row}, {dimension}) is non-finite"
                )));
            }
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
        let span = maximum - minimum;
        if !span.is_finite() || span <= 0.0 {
            return Err(numerical(format!(
                "embedding dimension {dimension} has zero or non-finite span"
            )));
        }
        for row in 0..n {
            embedding[row * dimensions + dimension] =
                10.0 * (embedding[row * dimensions + dimension] - minimum) / span;
        }
    }
    Ok(())
}

fn numerical(message: impl Into<String>) -> UmapError {
    UmapError::Numerical(message.into())
}

#[cfg(test)]
mod solver_tests {
    use super::*;

    #[test]
    fn dense_solver_returns_the_analytic_path_mode() {
        let graph = SparseGraph::from_edges(3, &[0, 1], &[1, 2], &[1.0, 1.0]).unwrap();
        let modes = top_nontrivial_modes(
            BaseOperator::Sparse(&graph),
            1,
            &mut StdRng::seed_from_u64(17),
        )
        .unwrap();
        let mut applied = vec![0.0; 3];
        graph.shifted_apply(&modes[0], &mut applied);
        let residual = l2_norm(
            &applied
                .iter()
                .zip(&modes[0])
                .map(|(a, v)| a - v)
                .collect::<Vec<_>>(),
        );
        assert!(residual < RESIDUAL_TOLERANCE);
        assert!((dot(&modes[0], &modes[0]) - 1.0).abs() < ORTHOGONALITY_TOLERANCE);
    }

    #[test]
    fn zero_coordinate_span_is_an_explicit_numerical_error() {
        assert!(matches!(
            minmax_scale(&mut [2.0, 2.0], 2, 1),
            Err(UmapError::Numerical(_))
        ));
    }

    #[test]
    fn fewer_converged_eigenpairs_is_spectral_convergence() {
        assert!(matches!(
            convergence_status(2, 3),
            Err(UmapError::SpectralConvergence(_))
        ));
    }

    #[test]
    fn only_spectral_convergence_uses_seeded_random_fallback() {
        let mut first_warning = String::new();
        let first = fallback_on_spectral_convergence(
            Err(UmapError::SpectralConvergence(
                "injected short convergence".into(),
            )),
            8,
            2,
            &mut StdRng::seed_from_u64(917),
            "injected test",
            |message| first_warning = message.to_owned(),
        )
        .unwrap();
        assert!(first_warning.contains("did not converge"));
        let mut repeat_warning = String::new();
        let repeat = fallback_on_spectral_convergence(
            Err(UmapError::SpectralConvergence(
                "injected short convergence".into(),
            )),
            8,
            2,
            &mut StdRng::seed_from_u64(917),
            "injected test",
            |message| repeat_warning = message.to_owned(),
        )
        .unwrap();
        assert_eq!(first, repeat);
        assert_eq!(first_warning, repeat_warning);

        let mut warned = false;
        assert!(matches!(
            fallback_on_spectral_convergence(
                Err(UmapError::Numerical("invalid operator".into())),
                8,
                2,
                &mut StdRng::seed_from_u64(918),
                "error classification",
                |_| warned = true,
            ),
            Err(UmapError::Numerical(_))
        ));
        assert!(!warned);
    }

    #[test]
    fn invariant_error_propagates_without_random_fallback() {
        let error = UmapError::Numerical("non-finite spectral Ritz value".into());
        let mut rng = StdRng::seed_from_u64(919);
        let mut untouched_rng = StdRng::seed_from_u64(919);
        let mut warned = false;

        let result = fallback_on_spectral_convergence(
            Err(error.clone()),
            8,
            2,
            &mut rng,
            "invariant error regression",
            |_| warned = true,
        );

        assert_eq!(result, Err(error));
        assert!(!warned);
        assert_eq!(rng.random::<u64>(), untouched_rng.random::<u64>());
    }

    #[test]
    fn nonfinite_weight_is_rejected_before_solver() {
        assert!(matches!(
            SparseGraph::from_edges(3, &[0], &[1], &[f32::NAN]),
            Err(UmapError::InvalidInput(_))
        ));
        assert!(matches!(
            SparseGraph::from_edges(3, &[0], &[1], &[f32::INFINITY]),
            Err(UmapError::InvalidInput(_))
        ));
    }

    #[cfg(feature = "tracing")]
    #[test]
    #[ignore = "external spectral probe; request and response paths supplied by harness"]
    fn external_spectral_probe() {
        use std::time::Instant;

        let request_path = std::env::var_os("FAST_UMAP_SPECTRAL_REQUEST")
            .expect("FAST_UMAP_SPECTRAL_REQUEST must name a graph JSON file");
        let response_path = std::env::var_os("FAST_UMAP_SPECTRAL_RESPONSE")
            .expect("FAST_UMAP_SPECTRAL_RESPONSE must name a result JSON file");
        let input: serde_json::Value = serde_json::from_slice(
            &std::fs::read(request_path).expect("read spectral graph request"),
        )
        .expect("parse spectral graph request JSON");
        let n = input["n"].as_u64().expect("request n") as usize;
        let dimensions = input["dimensions"].as_u64().expect("request dimensions") as usize;
        let seed = input["seed"].as_u64().expect("request seed");
        let heads = input["heads"]
            .as_array()
            .expect("request heads")
            .iter()
            .map(|value| value.as_u64().expect("COO head") as usize)
            .collect::<Vec<_>>();
        let tails = input["tails"]
            .as_array()
            .expect("request tails")
            .iter()
            .map(|value| value.as_u64().expect("COO tail") as usize)
            .collect::<Vec<_>>();
        let weights = input["weights"]
            .as_array()
            .expect("request weights")
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .map(|value| value as f32)
                    .or_else(|| value.as_str().and_then(|value| value.parse::<f32>().ok()))
                    .expect("COO weight")
            })
            .collect::<Vec<_>>();
        if input["runtime"].as_bool() == Some(true) {
            let graph =
                SparseGraph::from_edges(n, &heads, &tails, &weights).expect("valid runtime graph");
            assert_eq!(
                graph.components().len(),
                1,
                "runtime spectral graph must be connected"
            );
            let mut warmup_seconds = 0.0;
            let mut seconds = Vec::new();
            let mut diagnostics = Vec::with_capacity(6);
            for run in 0..6 {
                // Unseeded runtime is distinct from the paired-seed quality
                // protocol. Entropy acquisition is outside the solver clock.
                let mut rng = StdRng::seed_from_u64(rand::rng().random());
                let started = Instant::now();
                let solution = solve_spectral(BaseOperator::Sparse(&graph), dimensions, &mut rng)
                    .expect("runtime spectral solver must not fail or fall back");
                let elapsed = started.elapsed().as_secs_f64();
                diagnostics.push(solution.diagnostics);
                std::hint::black_box(solution.modes);
                if run == 0 {
                    warmup_seconds = elapsed;
                } else {
                    seconds.push(elapsed);
                }
            }
            std::fs::write(
                response_path,
                serde_json::to_vec_pretty(&serde_json::json!({
                    "runtime": true, "fallback": false, "warmups": 1, "runs": 5,
                    "warmup_seconds": warmup_seconds, "seconds": seconds,
                    "diagnostics": diagnostics,
                    "n": n, "dimensions": dimensions, "unseeded": true,
                }))
                .expect("serialize runtime spectral result"),
            )
            .expect("write runtime spectral result");
            return;
        }
        let started = Instant::now();
        let output = match spectral_probe(n, &heads, &tails, &weights, dimensions, seed) {
            Ok(result) => {
                let mut output = serde_json::to_value(result).expect("serialize spectral result");
                output["n"] = n.into();
                output["dimensions"] = dimensions.into();
                output["seed"] = seed.into();
                output["elapsed_seconds"] = started.elapsed().as_secs_f64().into();
                output["fallback"] = false.into();
                output
            }
            Err(UmapError::InvalidInput(reason)) => serde_json::json!({
                "n": n,
                "dimensions": dimensions,
                "seed": seed,
                "expected_rejection": true,
                "error": reason,
                "fallback": false,
                "elapsed_seconds": started.elapsed().as_secs_f64(),
            }),
            Err(error) => panic!("strict spectral probe failed: {error}"),
        };
        std::fs::write(
            response_path,
            serde_json::to_vec_pretty(&output).expect("serialize spectral result JSON"),
        )
        .expect("write spectral result JSON");
    }
}
