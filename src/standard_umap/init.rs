use super::{Init, Result, UMAPConfig, UmapError};
use faer::dyn_stack::MemStack;
use faer::linalg::evd::{self, ComputeEigenvectors, self_adjoint_evd, self_adjoint_evd_scratch};
use faer::{Mat, Par, Spec};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use std::collections::{BTreeMap, VecDeque};
use std::mem::MaybeUninit;

// Bound basis storage and projected EVD size independently of operator size.
const MAX_KRYLOV_BASIS: usize = 64;
const MAX_LANCZOS_OPERATOR_APPLICATIONS_PER_VERTEX: usize = 5;
const MAX_REORTHOGONALIZATION_PASSES: usize = 2;
const LANCZOS_BREAKDOWN_TOLERANCE: f64 = 1.0e-13;
const RESIDUAL_TOLERANCE: f64 = 1.0e-7;
const ORTHOGONALITY_TOLERANCE: f64 = 1.0e-7;

// faer 0.24.4's partial self-adjoint solver can loop without a bound on exact
// Lanczos breakdown. The matrix-free restarted driver below instead bounds
// operator applications and uses faer only for its <=64-dimensional projection.

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
    let (n, _) = super::validate_input(data, None)?;
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
}

impl SparseGraph {
    fn from_edges(n: usize, heads: &[usize], tails: &[usize], weights: &[f32]) -> Result<Self> {
        let mut directed = BTreeMap::<(usize, usize), f64>::new();
        for (edge, ((&head, &tail), &weight)) in heads.iter().zip(tails).zip(weights).enumerate() {
            if head >= n || tail >= n || head == tail {
                return Err(UmapError::InvalidInput(format!(
                    "spectral graph edge {edge} has invalid endpoints ({head}, {tail}) for {n} rows"
                )));
            }
            if !weight.is_finite() || weight < 0.0 {
                return Err(UmapError::InvalidInput(format!(
                    "spectral graph edge {edge} has invalid weight {weight}"
                )));
            }
            if weight > 0.0 {
                *directed.entry((head, tail)).or_default() += f64::from(weight);
            }
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
        Ok(Self { rows, degrees })
    }

    fn n(&self) -> usize {
        self.rows.len()
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.n() {
            if self.degrees[i] == 0.0 {
                out[i] = 2.0 * x[i];
                continue;
            }
            let mut value = x[i];
            for &(j, weight) in &self.rows[i] {
                value += weight * x[j] / (self.degrees[i] * self.degrees[j]).sqrt();
            }
            out[i] = value;
        }
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
        Ok(Self { rows, degrees })
    }
}

#[derive(Debug)]
struct CentroidGraph {
    centroids: Vec<Vec<f64>>,
    degrees: Vec<f64>,
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
        Ok(Self { centroids, degrees })
    }

    fn n(&self) -> usize {
        self.centroids.len()
    }

    fn shifted_apply(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.n() {
            let mut value = x[i];
            for (j, &x_j) in x.iter().enumerate().take(self.n()) {
                let weight = centroid_affinity(&self.centroids[i], &self.centroids[j]);
                value += weight * x_j / (self.degrees[i] * self.degrees[j]).sqrt();
            }
            out[i] = value;
        }
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

/// Matrix-free rank-one deflation around either
/// sparse graph or complete centroid-affinity normalized adjacency.
#[derive(Debug)]
struct SolverOperator<'a> {
    base: BaseOperator<'a>,
    deflated: Vec<(f64, Vec<f64>)>,
}

impl SolverOperator<'_> {
    fn apply_vector(&self, input: &[f64], output: &mut [f64]) {
        assert_eq!(input.len(), self.base.n());
        assert_eq!(output.len(), self.base.n());
        self.base.shifted_apply(input, output);
        for (eigenvalue, vector) in &self.deflated {
            let projection = dot(vector, input);
            for (value, &basis) in output.iter_mut().zip(vector) {
                *value -= eigenvalue * basis * projection;
            }
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
    let budget = base
        .n()
        .checked_mul(MAX_LANCZOS_OPERATOR_APPLICATIONS_PER_VERTEX)
        .ok_or(UmapError::Overflow("Lanczos application budget"))?;
    top_nontrivial_modes_with_budget(base, dimensions, rng, budget)
}

fn top_nontrivial_modes_with_budget(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
    application_budget: usize,
) -> Result<Vec<Vec<f64>>> {
    let base_n = base.n();
    if dimensions == 0 || dimensions >= base_n {
        return Err(numerical(format!(
            "spectral graph with {base_n} vertices cannot supply {dimensions} nontrivial modes"
        )));
    }
    if base
        .degrees()
        .iter()
        .any(|degree| !degree.is_finite() || *degree <= 0.0)
    {
        return Err(numerical("spectral graph has a zero or non-finite degree"));
    }

    let dimension = base_n;
    let deflated_capacity = dimensions
        .checked_add(1)
        .ok_or(UmapError::Overflow("deflated eigenspace count"))?;
    let mut deflated = Vec::with_capacity(deflated_capacity);
    let mut trivial = vec![0.0; dimension];
    let norm = base.degrees().iter().sum::<f64>().sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(numerical(
            "normalized-Laplacian trivial mode has invalid norm",
        ));
    }
    for (i, slot) in trivial.iter_mut().enumerate().take(base_n) {
        *slot = base.degrees()[i].sqrt() / norm;
    }
    deflated.push((2.0, trivial));

    let mut modes = Vec::with_capacity(dimensions);

    for mode_index in 0..dimensions {
        let mut start = vec![0.0; dimension];
        for value in &mut start {
            *value = gaussian(rng);
        }
        orthogonalize(
            &mut start,
            deflated.iter().map(|(_, vector)| vector.as_slice()),
        );
        let start_norm = l2_norm(&start);
        if !start_norm.is_finite() || start_norm <= 1.0e-12 {
            return Err(numerical(format!(
                "spectral mode {mode_index} has no independent deterministic start vector"
            )));
        }
        for value in &mut start {
            *value /= start_norm;
        }
        let operator = SolverOperator {
            base,
            deflated: deflated.clone(),
        };
        let (eigenvalue, mut vector) =
            bounded_lanczos_largest(&operator, start, application_budget)?;
        if !eigenvalue.is_finite() {
            return Err(numerical(format!(
                "spectral mode {mode_index} eigenvalue is non-finite"
            )));
        }
        let vector_norm = l2_norm(&vector);
        if !vector_norm.is_finite() || (vector_norm - 1.0).abs() > ORTHOGONALITY_TOLERANCE {
            return Err(numerical(format!(
                "spectral mode {mode_index} eigenvector norm is {vector_norm}"
            )));
        }
        orthogonalize(
            &mut vector,
            deflated.iter().map(|(_, previous)| previous.as_slice()),
        );
        let vector_norm = l2_norm(&vector);
        if !vector_norm.is_finite() || vector_norm <= 1.0e-12 {
            return Err(numerical(format!(
                "spectral mode {mode_index} collapsed against an existing eigenspace"
            )));
        }
        for value in &mut vector {
            *value /= vector_norm;
        }
        let mut applied = vec![0.0; base_n];
        base.shifted_apply(&vector, &mut applied);
        let residual = applied
            .iter()
            .zip(&vector)
            .map(|(&actual, &value)| (actual - eigenvalue * value).powi(2))
            .sum::<f64>();
        let residual = residual.sqrt();
        if !residual.is_finite() || residual > RESIDUAL_TOLERANCE {
            return Err(UmapError::SpectralConvergence(format!(
                "spectral mode {mode_index} residual {residual} exceeds {RESIDUAL_TOLERANCE} after reorthogonalization"
            )));
        }
        for (previous_index, (_, previous)) in deflated.iter().enumerate() {
            let overlap = dot(&vector, previous).abs();
            if overlap > ORTHOGONALITY_TOLERANCE {
                return Err(numerical(format!(
                    "spectral mode {mode_index} overlaps mode {previous_index} by {overlap}"
                )));
            }
        }
        deflated.push((eigenvalue, vector.clone()));
        modes.push(vector[..base_n].to_vec());
    }
    Ok(modes)
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

#[cfg(test)]
fn modes_with_source_fallback_and_budget_and_warning(
    base: BaseOperator<'_>,
    dimensions: usize,
    rng: &mut StdRng,
    budget: usize,
    context: &str,
    mut warn: impl FnMut(&str),
) -> Result<Vec<Vec<f64>>> {
    let result = top_nontrivial_modes_with_budget(base, dimensions, rng, budget);
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

/// Bounded, fully reorthogonalized matrix-free Lanczos followed by faer's dense
/// EVD only on the small symmetric projected matrix. Work/storage are
/// O(N*k + k²), with fixed iteration/reorthogonalization bounds; breakdown
/// terminates the Krylov sequence instead of entering an unbounded solver loop.
fn bounded_lanczos_largest(
    operator: &SolverOperator<'_>,
    start: Vec<f64>,
    total_budget: usize,
) -> Result<(f64, Vec<f64>)> {
    let n = operator.base.n();
    if n == 0 {
        return Err(numerical("Lanczos operator has zero dimension"));
    }
    let mut remaining_budget = total_budget;
    let mut start = start;
    let mut last_residual = f64::INFINITY;
    let mut last_breakdown = false;

    while remaining_budget > 0 {
        let cycle_limit = MAX_KRYLOV_BASIS.min(n).min(remaining_budget);
        let (eigenvalue, vector, residual, breakdown) =
            lanczos_ritz_cycle(operator, start, cycle_limit)?;
        remaining_budget -= cycle_limit;
        if !eigenvalue.is_finite() {
            return Err(numerical("Lanczos Ritz value is non-finite"));
        }
        if residual <= RESIDUAL_TOLERANCE {
            return Ok((eigenvalue, vector));
        }
        last_residual = residual;
        last_breakdown = breakdown;
        if breakdown {
            break;
        }
        start = vector;
    }
    let status = if last_breakdown {
        "invariant Krylov subspace exhausted"
    } else {
        "bounded restarted-Lanczos application budget exhausted"
    };
    Err(UmapError::SpectralConvergence(format!(
        "largest Ritz pair residual {last_residual} after {status}; budget {total_budget} operator applications"
    )))
}

fn lanczos_ritz_cycle(
    operator: &SolverOperator<'_>,
    start: Vec<f64>,
    max_iterations: usize,
) -> Result<(f64, Vec<f64>, f64, bool)> {
    let n = operator.base.n();
    let mut basis = Vec::<Vec<f64>>::with_capacity(max_iterations);
    let mut diagonal = Vec::<f64>::with_capacity(max_iterations);
    let mut off_diagonal = Vec::<f64>::with_capacity(max_iterations.saturating_sub(1));
    basis.push(start);
    let mut applied = vec![0.0; n];
    let mut breakdown = false;

    for iteration in 0..max_iterations {
        operator.apply_vector(&basis[iteration], &mut applied);
        if iteration > 0 {
            let beta = off_diagonal[iteration - 1];
            for (value, &previous) in applied.iter_mut().zip(&basis[iteration - 1]) {
                *value -= beta * previous;
            }
        }
        let alpha = dot(&basis[iteration], &applied);
        if !alpha.is_finite() {
            return Err(numerical(format!(
                "Lanczos alpha {iteration} is non-finite"
            )));
        }
        diagonal.push(alpha);
        for (value, &current) in applied.iter_mut().zip(&basis[iteration]) {
            *value -= alpha * current;
        }
        for _ in 0..MAX_REORTHOGONALIZATION_PASSES {
            for previous in &basis {
                let projection = dot(&applied, previous);
                for (value, &basis_value) in applied.iter_mut().zip(previous) {
                    *value -= projection * basis_value;
                }
            }
        }
        let beta = l2_norm(&applied);
        if !beta.is_finite() {
            return Err(numerical(format!("Lanczos beta {iteration} is non-finite")));
        }
        if beta <= LANCZOS_BREAKDOWN_TOLERANCE {
            breakdown = true;
            break;
        }
        if iteration + 1 == max_iterations {
            break;
        }
        off_diagonal.push(beta);
        for value in &mut applied {
            *value /= beta;
        }
        basis.push(applied.clone());
    }

    let projected_n = diagonal.len();
    if projected_n == 0 {
        return Err(numerical("Lanczos produced an empty projected operator"));
    }
    let mut projected = Mat::zeros(projected_n, projected_n);
    for i in 0..projected_n {
        projected[(i, i)] = diagonal[i];
        if i + 1 < projected_n {
            let beta = off_diagonal[i];
            projected[(i, i + 1)] = beta;
            projected[(i + 1, i)] = beta;
        }
    }

    let params = Spec::<evd::SelfAdjointEvdParams, f64>::default();
    let par = Par::Seq;
    let scratch_req =
        self_adjoint_evd_scratch::<f64>(projected_n, ComputeEigenvectors::Yes, par, params);
    let scratch_size = scratch_req
        .size_bytes()
        .checked_add(scratch_req.align_bytes())
        .ok_or_else(|| numerical("projected EVD scratch size overflowed"))?;
    let mut scratch = vec![MaybeUninit::<u8>::uninit(); scratch_size];
    let mut eigenvalues = Mat::zeros(projected_n, projected_n);
    let mut eigenvectors = Mat::zeros(projected_n, projected_n);
    self_adjoint_evd(
        projected.as_ref(),
        eigenvalues.as_mut().diagonal_mut(),
        Some(eigenvectors.as_mut()),
        par,
        MemStack::new(&mut scratch),
        params,
    )
    .map_err(|error| {
        UmapError::SpectralConvergence(format!("projected self-adjoint EVD failed: {error:?}"))
    })?;

    let top = projected_n - 1;
    let eigenvalue = eigenvalues[(top, top)];
    let mut vector = vec![0.0; n];
    for column in 0..projected_n {
        let coefficient = eigenvectors[(column, top)];
        for row in 0..n {
            vector[row] += coefficient * basis[column][row];
        }
    }
    let mut exact = vec![0.0; n];
    operator.apply_vector(&vector, &mut exact);
    let residual = l2_norm(
        &exact
            .iter()
            .zip(&vector)
            .map(|(&actual, &value)| actual - eigenvalue * value)
            .collect::<Vec<_>>(),
    );
    Ok((eigenvalue, vector, residual, breakdown))
}

fn orthogonalize<'a>(vector: &mut [f64], against: impl Iterator<Item = &'a [f64]>) {
    for basis in against {
        let projection = dot(vector, basis);
        for (value, &basis_value) in vector.iter_mut().zip(basis) {
            *value -= projection * basis_value;
        }
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(&a, &b)| a * b).sum()
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
    fn bounded_lanczos_returns_analytic_path_and_repeated_cycle_modes() {
        let path = SparseGraph::from_edges(3, &[0, 1], &[1, 2], &[1.0, 1.0]).unwrap();
        let path_modes = top_nontrivial_modes(
            BaseOperator::Sparse(&path),
            1,
            &mut StdRng::seed_from_u64(17),
        )
        .unwrap();
        let mut path_applied = vec![0.0; 3];
        path.shifted_apply(&path_modes[0], &mut path_applied);
        let path_residual = path_applied
            .iter()
            .zip(&path_modes[0])
            .map(|(&actual, &value)| (actual - value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(path_residual < RESIDUAL_TOLERANCE);
        assert!((dot(&path_modes[0], &path_modes[0]) - 1.0).abs() < ORTHOGONALITY_TOLERANCE);

        let n = 12;
        let heads = (0..n).collect::<Vec<_>>();
        let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
        let weights = vec![1.0; n];
        let cycle = SparseGraph::from_edges(n, &heads, &tails, &weights).unwrap();
        let modes = top_nontrivial_modes(
            BaseOperator::Sparse(&cycle),
            2,
            &mut StdRng::seed_from_u64(73),
        )
        .unwrap();
        let expected = 1.0 + (2.0 * std::f64::consts::PI / n as f64).cos();
        let mut applied = vec![0.0; n];
        for mode in &modes {
            cycle.shifted_apply(mode, &mut applied);
            let residual = applied
                .iter()
                .zip(mode)
                .map(|(&actual, &value)| (actual - expected * value).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                residual < RESIDUAL_TOLERANCE,
                "cycle mode residual {residual}"
            );
        }
        assert!(dot(&modes[0], &modes[1]).abs() < ORTHOGONALITY_TOLERANCE);
    }

    #[test]
    fn zero_coordinate_span_is_an_explicit_numerical_error() {
        assert!(matches!(
            minmax_scale(&mut [2.0, 2.0], 2, 1),
            Err(UmapError::Numerical(_))
        ));
    }

    #[test]
    fn forced_solver_budget_failure_warns_and_uses_deterministic_random_modes() {
        let n = 12;
        let heads = (0..n).collect::<Vec<_>>();
        let tails = (0..n).map(|i| (i + 1) % n).collect::<Vec<_>>();
        let weights = vec![1.0; n];
        let graph = SparseGraph::from_edges(n, &heads, &tails, &weights).unwrap();
        assert!(matches!(
            top_nontrivial_modes_with_budget(
                BaseOperator::Sparse(&graph),
                2,
                &mut StdRng::seed_from_u64(916),
                0,
            ),
            Err(UmapError::SpectralConvergence(_))
        ));
        let mut first_rng = StdRng::seed_from_u64(917);
        let mut first_warning = String::new();
        let first = modes_with_source_fallback_and_budget_and_warning(
            BaseOperator::Sparse(&graph),
            2,
            &mut first_rng,
            0,
            "forced solver-budget test",
            |message| first_warning = message.to_owned(),
        )
        .unwrap();
        assert!(first_warning.contains("did not converge"));
        assert!(first_warning.contains("seeded uniform initialization"));

        let mut repeated_rng = StdRng::seed_from_u64(917);
        let mut repeated_warning = String::new();
        let repeated = modes_with_source_fallback_and_budget_and_warning(
            BaseOperator::Sparse(&graph),
            2,
            &mut repeated_rng,
            0,
            "forced solver-budget test",
            |message| repeated_warning = message.to_owned(),
        )
        .unwrap();
        assert_eq!(first, repeated);
        assert_eq!(first_warning, repeated_warning);
        assert!(
            first
                .iter()
                .flatten()
                .all(|value| (-10.0..10.0).contains(value))
        );

        let mut warned = false;
        assert!(matches!(
            fallback_on_spectral_convergence(
                Err(UmapError::Numerical("invalid operator".into())),
                n,
                2,
                &mut StdRng::seed_from_u64(918),
                "error classification test",
                |_| warned = true,
            ),
            Err(UmapError::Numerical(_))
        ));
        assert!(
            !warned,
            "non-convergence fallback must not catch numerical errors"
        );
    }
}
