//! Private cached block thick-restart Krylov solver. Iteration budgets are
//! numerical limits, not wall-clock guarantees.
//! The Krylov kernels are deliberately sequential because measured Rayon
//! dispatch overhead exceeded the work performed per call.

use super::{BaseOperator, Result, UmapError, gaussian};
use faer::linalg::matmul::dot::inner_prod;
use faer::linalg::matmul::matmul;
use faer::{Accum, ColRef, Conj, Mat, MatRef, Par, Side};
use rand::rngs::StdRng;

const BLOCK_SIZE: usize = 2;
const MAX_CYCLES: usize = 300;
const MAX_RANDOM_REDRAWS: usize = 3;
const RANK_THRESHOLD: f64 = 1.0e-12;
const DGKS_ETA: f64 = std::f64::consts::FRAC_1_SQRT_2;
const RITZ_TOLERANCE: f64 = 1.0e-10;
const CLUSTER_RELATIVE_GAP: f64 = 1.0e-6;
// Bound scratch space while computing Ritz residual norms.
const RESIDUAL_CHUNK: usize = 4096;

#[cfg_attr(all(test, feature = "tracing"), derive(serde::Serialize))]
#[derive(Clone, Debug, Default)]
pub(super) struct Diagnostics {
    pub(super) cycles: usize,
    pub(super) column_applies: usize,
    pub(super) augmentations: usize,
    pub(super) reorthogonalizations: usize,
    pub(super) skipped_reorthogonalizations: usize,
    pub(super) confirmation_failures: usize,
    pub(super) basis_width: usize,
    pub(super) keep: usize,
    pub(super) cluster_end: usize,
    pub(super) max_residual: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) apply_seconds: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) orthogonalization_seconds: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) restart_seconds: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) eig_seconds: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) rayleigh_seconds: f64,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) confirmation_seconds: f64,
}

pub(super) struct Solution {
    pub(super) pairs: Vec<(f64, Vec<f64>)>,
    #[cfg(all(test, feature = "tracing"))]
    pub(super) diagnostics: Diagnostics,
}

// Pack input columns into contiguous row blocks so each sparse-neighbor gather
// reads all vector coordinates contiguously. Rows retain CSR accumulation order.
fn apply_into(
    base: BaseOperator<'_>,
    input: &Mat<f64>,
    packed_input: &mut Vec<f64>,
    packed_output: &mut Vec<f64>,
    output: &mut Mat<f64>,
) {
    let graph = match base {
        BaseOperator::Sparse(graph) => &graph.normalized,
        BaseOperator::Centroids(graph) => &graph.normalized,
    };
    let n = input.nrows();
    let width = input.ncols();
    let len = n * width;
    packed_input.resize(len, 0.0);
    packed_output.resize(len, 0.0);
    for col in 0..width {
        for (row, value) in input.col_as_slice(col).iter().enumerate() {
            packed_input[row * width + col] = *value;
        }
    }
    let packed_input = &packed_input[..];
    match width {
        1 => apply_rows::<1>(graph, packed_input, packed_output, 0),
        2 => apply_rows::<2>(graph, packed_input, packed_output, 0),
        3 => apply_rows::<3>(graph, packed_input, packed_output, 0),
        4 => apply_rows::<4>(graph, packed_input, packed_output, 0),
        _ => apply_rows_dynamic(graph, packed_input, packed_output, 0, width),
    }
    output.resize_with(n, width, |_, _| 0.0);
    for col in 0..width {
        for (row, value) in output.col_as_slice_mut(col).iter_mut().enumerate() {
            *value = packed_output[row * width + col];
        }
    }
}

fn apply_rows<const W: usize>(
    graph: &super::NormalizedCsr,
    input: &[f64],
    output_rows: &mut [f64],
    first_row: usize,
) {
    let (rows, _) = output_rows.as_chunks_mut::<W>();
    for (offset, output) in rows.iter_mut().enumerate() {
        let row = first_row + offset;
        let mut sum: [f64; W] = input[row * W..row * W + W].try_into().expect("row width");
        let start = graph.row_offsets[row];
        let end = graph.row_offsets[row + 1];
        for (&column, &weight) in graph.columns[start..end]
            .iter()
            .zip(&graph.values[start..end])
        {
            let source: &[f64; W] = input[column * W..column * W + W]
                .try_into()
                .expect("row width");
            for col in 0..W {
                sum[col] += weight * source[col];
            }
        }
        output.copy_from_slice(&sum);
    }
}

fn apply_rows_dynamic(
    graph: &super::NormalizedCsr,
    input: &[f64],
    output_rows: &mut [f64],
    first_row: usize,
    width: usize,
) {
    for (offset, output) in output_rows.chunks_exact_mut(width).enumerate() {
        let row = first_row + offset;
        output.copy_from_slice(&input[row * width..row * width + width]);
        for edge in graph.row_offsets[row]..graph.row_offsets[row + 1] {
            let source = &input[graph.columns[edge] * width..graph.columns[edge] * width + width];
            let weight = graph.values[edge];
            for (target, value) in output.iter_mut().zip(source) {
                *target += weight * value;
            }
        }
    }
}

fn apply_counted(
    base: BaseOperator<'_>,
    input: &Mat<f64>,
    packed_input: &mut Vec<f64>,
    packed_output: &mut Vec<f64>,
    output: &mut Mat<f64>,
    stats: &mut Diagnostics,
) {
    #[cfg(all(test, feature = "tracing"))]
    let started = std::time::Instant::now();
    apply_into(base, input, packed_input, packed_output, output);
    stats.column_applies += input.ncols();
    #[cfg(all(test, feature = "tracing"))]
    {
        stats.apply_seconds += started.elapsed().as_secs_f64();
    }
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    inner_prod(
        ColRef::from_slice(left).transpose(),
        Conj::No,
        ColRef::from_slice(right),
        Conj::No,
    )
}

fn l2_norm(values: &[f64]) -> f64 {
    ColRef::from_slice(values).norm_l2()
}

fn projection_coefficients(basis: &Mat<f64>, candidate: &Mat<f64>) -> Mat<f64> {
    let mut coefficients = Mat::zeros(basis.ncols(), candidate.ncols());
    matmul(
        coefficients.as_mut(),
        Accum::Replace,
        basis.as_ref().transpose(),
        candidate.as_ref(),
        1.0,
        Par::Seq,
    );
    coefficients
}

fn project_against_matrix(candidate: &mut Mat<f64>, basis: &Mat<f64>) -> Mat<f64> {
    let coefficients = projection_coefficients(basis, candidate);
    matmul(
        candidate.as_mut(),
        Accum::Add,
        basis.as_ref(),
        coefficients.as_ref(),
        -1.0,
        Par::Seq,
    );
    coefficients
}

fn project_block(
    candidates: &mut Mat<f64>,
    trivial: &Mat<f64>,
    basis: &Mat<f64>,
    force_reorthogonalization: bool,
    stats: &mut Diagnostics,
) -> Mat<f64> {
    let norms_before = (0..candidates.ncols())
        .map(|col| l2_norm(candidates.col_as_slice(col)))
        .collect::<Vec<_>>();
    let _ = project_against_matrix(candidates, trivial);
    // q0 is excluded from H; the coefficients against the basis are accumulated
    // across exactly the CGS passes used to project these candidates.
    let mut coefficients = project_against_matrix(candidates, basis);
    let needs_reorthogonalization = force_reorthogonalization
        || (0..candidates.ncols())
            .any(|col| l2_norm(candidates.col_as_slice(col)) < DGKS_ETA * norms_before[col]);
    if needs_reorthogonalization {
        let _ = project_against_matrix(candidates, trivial);
        let second_pass = project_against_matrix(candidates, basis);
        for col in 0..coefficients.ncols() {
            for row in 0..coefficients.nrows() {
                coefficients[(row, col)] += second_pass[(row, col)];
            }
        }
        stats.reorthogonalizations += 1;
    } else {
        stats.skipped_reorthogonalizations += 1;
    }
    coefficients
}

fn modified_gram_schmidt_twice(candidate: &mut Mat<f64>, accepted: &Mat<f64>, count: usize) {
    for _ in 0..2 {
        for col in 0..count {
            let coefficient = dot(accepted.col_as_slice(col), candidate.col_as_slice(0));
            for (target, value) in candidate
                .col_as_slice_mut(0)
                .iter_mut()
                .zip(accepted.col_as_slice(col))
            {
                *target -= coefficient * value;
            }
        }
    }
}

fn append_columns(left: &mut Mat<f64>, right: &Mat<f64>) {
    let first_col = left.ncols();
    left.resize_with(left.nrows(), first_col + right.ncols(), |row, col| {
        right[(row, col - first_col)]
    });
}

fn append_rayleigh_columns(rayleigh: &mut Mat<f64>, q: &Mat<f64>, aq: &Mat<f64>, old_width: usize) {
    let width = q.ncols();
    let added = width - old_width;
    let applied = Mat::from_fn(aq.nrows(), added, |row, col| aq[(row, old_width + col)]);
    let mut block = Mat::zeros(width, added);
    matmul(
        block.as_mut(),
        Accum::Replace,
        q.as_ref().transpose(),
        applied.as_ref(),
        1.0,
        Par::Seq,
    );

    rayleigh.resize_with(width, width, |_, _| 0.0);
    for col in old_width..width {
        for row in 0..old_width {
            let value = block[(row, col - old_width)];
            rayleigh[(row, col)] = value;
            rayleigh[(col, row)] = value;
        }
    }
    for row in old_width..width {
        for col in old_width..width {
            let value = (block[(row, col - old_width)] + block[(col, row - old_width)]) * 0.5;
            rayleigh[(row, col)] = value;
            rayleigh[(col, row)] = value;
        }
    }
}

fn set_rayleigh_columns_from_coefficients(
    rayleigh: &mut Mat<f64>,
    coefficients: &Mat<f64>,
    block_start: usize,
) {
    // For expansion candidates W = A Q_block, these are Q^T W = H[:, block].
    let width = coefficients.nrows();
    let block_width = coefficients.ncols();
    rayleigh.resize_with(width, width, |_, _| 0.0);

    for col in 0..block_width {
        for row in 0..block_start {
            let value = coefficients[(row, col)];
            rayleigh[(row, block_start + col)] = value;
            rayleigh[(block_start + col, row)] = value;
        }
    }
    for row in 0..block_width {
        for col in 0..block_width {
            let value = (coefficients[(block_start + row, col)]
                + coefficients[(block_start + col, row)])
                * 0.5;
            rayleigh[(block_start + row, block_start + col)] = value;
        }
    }
}

fn symmetrize(matrix: &mut Mat<f64>) {
    for row in 0..matrix.nrows() {
        for col in 0..row {
            let value = (matrix[(row, col)] + matrix[(col, row)]) * 0.5;
            matrix[(row, col)] = value;
            matrix[(col, row)] = value;
        }
    }
}

fn rotate_rayleigh(rayleigh: &Mat<f64>, rotation: &Mat<f64>) -> Mat<f64> {
    let mut intermediate = Mat::zeros(rayleigh.nrows(), rotation.ncols());
    matmul(
        intermediate.as_mut(),
        Accum::Replace,
        rayleigh.as_ref(),
        rotation.as_ref(),
        1.0,
        Par::Seq,
    );
    let mut rotated = Mat::zeros(rotation.ncols(), rotation.ncols());
    matmul(
        rotated.as_mut(),
        Accum::Replace,
        rotation.as_ref().transpose(),
        intermediate.as_ref(),
        1.0,
        Par::Seq,
    );
    symmetrize(&mut rotated);
    rotated
}

// DGKS-conditional block CGS against q0 and the existing basis, then within-block MGS2.
// A deficient column is replaced by at most three fresh projected Gaussians.
fn orthogonalize_candidates(
    candidates: &Mat<f64>,
    trivial: &Mat<f64>,
    basis: &Mat<f64>,
    max_columns: usize,
    rng: &mut StdRng,
    stats: &mut Diagnostics,
) -> Result<Mat<f64>> {
    orthogonalize_candidates_with_coefficients(
        candidates,
        trivial,
        basis,
        max_columns,
        rng,
        stats,
        false,
    )
    .map(|(accepted, _)| accepted)
}

fn orthogonalize_candidates_with_coefficients(
    candidates: &Mat<f64>,
    trivial: &Mat<f64>,
    basis: &Mat<f64>,
    max_columns: usize,
    rng: &mut StdRng,
    stats: &mut Diagnostics,
    capture_all_candidate_coefficients: bool,
) -> Result<(Mat<f64>, Mat<f64>)> {
    if candidates.nrows() == 0
        || candidates.nrows() != trivial.nrows()
        || trivial.ncols() != 1
        || basis.nrows() != candidates.nrows()
        || candidates.ncols() == 0
        || max_columns == 0
    {
        return Err(UmapError::Numerical(
            "malformed Krylov orthogonalization dimensions".into(),
        ));
    }
    if candidates
        .col_as_slice(0)
        .iter()
        .chain((1..candidates.ncols()).flat_map(|col| candidates.col_as_slice(col)))
        .any(|value| !value.is_finite())
    {
        return Err(UmapError::Numerical(
            "non-finite Krylov candidate block".into(),
        ));
    }
    let count = candidates.ncols().min(max_columns);
    let mut projected = Mat::from_fn(candidates.nrows(), count, |row, col| candidates[(row, col)]);
    let coefficients = project_block(&mut projected, trivial, basis, false, stats);
    let coefficients = if capture_all_candidate_coefficients && count < candidates.ncols() {
        // Preserve the expansion block's DGKS decision for accepted columns;
        // project any unconsumed tail separately only to populate its H columns.
        let mut unconsumed = Mat::from_fn(
            candidates.nrows(),
            candidates.ncols() - count,
            |row, col| candidates[(row, count + col)],
        );
        let mut ignored_stats = Diagnostics::default();
        let unconsumed_coefficients =
            project_block(&mut unconsumed, trivial, basis, false, &mut ignored_stats);
        Mat::from_fn(basis.ncols(), candidates.ncols(), |row, col| {
            if col < count {
                coefficients[(row, col)]
            } else {
                unconsumed_coefficients[(row, col - count)]
            }
        })
    } else {
        coefficients
    };
    let mut accepted = Mat::zeros(candidates.nrows(), count);
    for col in 0..count {
        let mut candidate = Mat::from_fn(candidates.nrows(), 1, |row, _| projected[(row, col)]);
        modified_gram_schmidt_twice(&mut candidate, &accepted, col);
        let mut norm = l2_norm(candidate.col_as_slice(0));
        if !norm.is_finite() {
            return Err(UmapError::Numerical(
                "non-finite norm during Krylov orthogonalization".into(),
            ));
        }
        let mut redraws = 0;
        while norm <= RANK_THRESHOLD {
            if redraws == MAX_RANDOM_REDRAWS {
                return Err(UmapError::SpectralConvergence(format!(
                    "Krylov candidate {col} remained rank deficient after {MAX_RANDOM_REDRAWS} Gaussian redraws"
                )));
            }
            redraws += 1;
            stats.augmentations += 1;
            for row in 0..candidate.nrows() {
                candidate[(row, 0)] = gaussian(rng);
            }
            if candidate
                .col_as_slice(0)
                .iter()
                .any(|value| !value.is_finite())
            {
                return Err(UmapError::Numerical(
                    "non-finite Gaussian Krylov augmentation".into(),
                ));
            }
            let _ = project_block(&mut candidate, trivial, basis, true, stats);
            modified_gram_schmidt_twice(&mut candidate, &accepted, col);
            norm = l2_norm(candidate.col_as_slice(0));
            if !norm.is_finite() {
                return Err(UmapError::Numerical(
                    "non-finite augmented Krylov candidate norm".into(),
                ));
            }
        }
        for (target, value) in accepted
            .col_as_slice_mut(col)
            .iter_mut()
            .zip(candidate.col_as_slice(0))
        {
            *target = value / norm;
        }
    }
    Ok((accepted, coefficients))
}

fn descending_eigenvectors(eigenvectors: MatRef<'_, f64>) -> Mat<f64> {
    let width = eigenvectors.ncols();
    Mat::from_fn(width, width, |row, col| {
        eigenvectors[(row, width - 1 - col)]
    })
}

fn leading_columns(matrix: &Mat<f64>, count: usize) -> Mat<f64> {
    Mat::from_fn(matrix.nrows(), count, |row, col| matrix[(row, col)])
}

fn rotate(basis: &Mat<f64>, rotation: &Mat<f64>) -> Mat<f64> {
    let mut output = Mat::zeros(basis.nrows(), rotation.ncols());
    rotate_into(basis, rotation, &mut output);
    output
}

fn rotate_into(basis: &Mat<f64>, rotation: &Mat<f64>, output: &mut Mat<f64>) {
    output.resize_with(basis.nrows(), rotation.ncols(), |_, _| 0.0);
    matmul(
        output.as_mut(),
        Accum::Replace,
        basis.as_ref(),
        rotation.as_ref(),
        1.0,
        Par::Seq,
    );
}

fn residual_norm(vector: &[f64], applied: &[f64], value: f64) -> f64 {
    let mut residual = [0.0; RESIDUAL_CHUNK];
    let mut squared_norm = 0.0;
    for start in (0..vector.len()).step_by(RESIDUAL_CHUNK) {
        let end = (start + RESIDUAL_CHUNK).min(vector.len());
        for index in start..end {
            residual[index - start] = applied[index] - value * vector[index];
        }
        let chunk_norm = l2_norm(&residual[..end - start]);
        squared_norm += chunk_norm * chunk_norm;
    }
    squared_norm.sqrt()
}

struct ApplyBuffers<'a> {
    packed_input: &'a mut Vec<f64>,
    packed_output: &'a mut Vec<f64>,
    applied: &'a mut Mat<f64>,
}

fn confirm(
    base: BaseOperator<'_>,
    trivial: &[f64],
    vectors: &Mat<f64>,
    values: &[f64],
    count: usize,
    buffers: &mut ApplyBuffers<'_>,
    stats: &mut Diagnostics,
) -> Result<bool> {
    if trivial.len() != base.n()
        || vectors.nrows() != base.n()
        || count > vectors.ncols()
        || count > values.len()
    {
        return Err(UmapError::Numerical(
            "malformed fresh Ritz confirmation dimensions".into(),
        ));
    }
    #[cfg(all(test, feature = "tracing"))]
    let started = std::time::Instant::now();
    let tested = vectors;
    apply_into(
        base,
        tested,
        buffers.packed_input,
        buffers.packed_output,
        buffers.applied,
    );
    stats.column_applies += count;
    let mut max_residual = 0.0_f64;
    let mut valid = true;
    for (col, &value) in values.iter().take(count).enumerate() {
        let vector = tested.col_as_slice(col);
        let ax = buffers.applied.col_as_slice(col);
        if vector
            .iter()
            .chain(ax)
            .any(|coordinate| !coordinate.is_finite())
            || !value.is_finite()
        {
            return Err(UmapError::Numerical(
                "non-finite value during fresh Ritz confirmation".into(),
            ));
        }
        let residual = residual_norm(vector, ax, value);
        if !residual.is_finite() {
            return Err(UmapError::Numerical(
                "non-finite fresh Ritz residual".into(),
            ));
        }
        max_residual = max_residual.max(residual);
        valid &= residual <= RITZ_TOLERANCE;
        let trivial_gram = dot(trivial, vector);
        if !trivial_gram.is_finite() {
            return Err(UmapError::Numerical(
                "non-finite locked-trivial Gram entry during Ritz confirmation".into(),
            ));
        }
        valid &= trivial_gram.abs() <= 1.0e-12;
        for other in 0..count {
            let gram = dot(vector, tested.col_as_slice(other));
            if !gram.is_finite() {
                return Err(UmapError::Numerical(
                    "non-finite Ritz Gram entry during confirmation".into(),
                ));
            }
            valid &= (gram - if col == other { 1.0 } else { 0.0 }).abs() <= 1.0e-12;
        }
    }
    stats.max_residual = max_residual;
    if !valid {
        stats.confirmation_failures += 1;
    }
    #[cfg(all(test, feature = "tracing"))]
    {
        stats.confirmation_seconds += started.elapsed().as_secs_f64();
    }
    Ok(valid)
}

pub(super) fn solve(
    base: BaseOperator<'_>,
    requested: usize,
    max_dim: usize,
    rng: &mut StdRng,
) -> Result<Solution> {
    let n = base.n();
    if requested < 2 {
        return Err(UmapError::Numerical(
            "spectral solver requires a trivial mode and at least one requested mode".into(),
        ));
    }
    if requested > n {
        return Err(UmapError::Numerical(format!(
            "spectral solver requested {requested} eigenpairs for {n} vertices"
        )));
    }
    let wanted = requested - 1;
    let max_basis = max_dim.min(n.saturating_sub(1 + BLOCK_SIZE));
    let minimum_basis = wanted
        .checked_add(2 * BLOCK_SIZE)
        .ok_or(UmapError::Overflow("spectral minimum block basis"))?;
    if max_basis < minimum_basis {
        return Err(UmapError::Numerical(format!(
            "spectral max basis {max_basis} cannot reserve {minimum_basis} required columns"
        )));
    }
    let mut trivial = base
        .degrees()
        .iter()
        .map(|degree| degree.sqrt())
        .collect::<Vec<_>>();
    if trivial.iter().any(|value| !value.is_finite()) {
        return Err(UmapError::Numerical(
            "non-finite locked trivial eigenvector".into(),
        ));
    }
    let trivial_norm = l2_norm(&trivial);
    if !trivial_norm.is_finite() || trivial_norm == 0.0 {
        return Err(UmapError::Numerical(
            "zero or non-finite locked trivial eigenvector".into(),
        ));
    }
    for value in &mut trivial {
        *value /= trivial_norm;
    }
    let trivial_column = Mat::from_fn(n, 1, |row, _| trivial[row]);

    let mut stats = Diagnostics::default();
    let initial = Mat::from_fn(n, BLOCK_SIZE, |_, _| gaussian(rng));
    #[cfg(all(test, feature = "tracing"))]
    let started = std::time::Instant::now();
    let mut q = orthogonalize_candidates(
        &initial,
        &trivial_column,
        &Mat::zeros(n, 0),
        BLOCK_SIZE,
        rng,
        &mut stats,
    )?;
    q.reserve(n, max_basis);
    let mut aq = Mat::zeros(n, 0);
    aq.reserve(n, max_basis);
    let mut q_rotation_scratch = Mat::zeros(n, 0);
    q_rotation_scratch.reserve(n, max_basis);
    let mut aq_rotation_scratch = Mat::zeros(n, 0);
    aq_rotation_scratch.reserve(n, max_basis);
    let mut applied_scratch = Mat::zeros(n, 0);
    applied_scratch.reserve(n, BLOCK_SIZE);
    let mut rayleigh = Mat::zeros(0, 0);
    let mut packed_input = Vec::with_capacity(n * BLOCK_SIZE);
    let mut packed_output = Vec::with_capacity(n * BLOCK_SIZE);
    #[cfg(all(test, feature = "tracing"))]
    {
        stats.orthogonalization_seconds += started.elapsed().as_secs_f64();
    }
    apply_counted(
        base,
        &q,
        &mut packed_input,
        &mut packed_output,
        &mut applied_scratch,
        &mut stats,
    );
    append_columns(&mut aq, &applied_scratch);
    #[cfg(all(test, feature = "tracing"))]
    let rayleigh_started = std::time::Instant::now();
    append_rayleigh_columns(&mut rayleigh, &q, &aq, 0);
    #[cfg(all(test, feature = "tracing"))]
    {
        stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
    }
    let mut frontier_start = 0;
    let mut frontier_width = BLOCK_SIZE;
    let mut final_block_rayleigh_explicit = true;

    while q.ncols() < max_basis {
        let remaining = max_basis - q.ncols();
        let next_width = remaining.min(BLOCK_SIZE);
        let augmentations_before = stats.augmentations;
        let frontier = Mat::from_fn(n, frontier_width, |row, col| {
            aq[(row, frontier_start + col)]
        });
        #[cfg(all(test, feature = "tracing"))]
        let started = std::time::Instant::now();
        let (next, coefficients) = orthogonalize_candidates_with_coefficients(
            &frontier,
            &trivial_column,
            &q,
            next_width,
            rng,
            &mut stats,
            true,
        )?;
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.orthogonalization_seconds += started.elapsed().as_secs_f64();
        }
        set_rayleigh_columns_from_coefficients(&mut rayleigh, &coefficients, frontier_start);
        frontier_start = q.ncols();
        frontier_width = next.ncols();
        let old_width = q.ncols();
        append_columns(&mut q, &next);
        apply_counted(
            base,
            &next,
            &mut packed_input,
            &mut packed_output,
            &mut applied_scratch,
            &mut stats,
        );
        append_columns(&mut aq, &applied_scratch);
        final_block_rayleigh_explicit = stats.augmentations != augmentations_before;
        if final_block_rayleigh_explicit {
            // A Gaussian redraw was not formed from the previous A-image block,
            // so compute this appended block's H columns explicitly.
            #[cfg(all(test, feature = "tracing"))]
            let rayleigh_started = std::time::Instant::now();
            append_rayleigh_columns(&mut rayleigh, &q, &aq, old_width);
            #[cfg(all(test, feature = "tracing"))]
            {
                stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
            }
        }
    }
    if !final_block_rayleigh_explicit {
        // This terminal block has no following expansion to provide its H columns.
        let final_block_start = q.ncols() - frontier_width;
        #[cfg(all(test, feature = "tracing"))]
        let rayleigh_started = std::time::Instant::now();
        append_rayleigh_columns(&mut rayleigh, &q, &aq, final_block_start);
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
        }
    }

    for cycle in 0..MAX_CYCLES {
        stats.cycles = cycle + 1;
        stats.basis_width = q.ncols();
        #[cfg(all(test, feature = "tracing"))]
        let started = std::time::Instant::now();
        let projected = &rayleigh;
        if projected.nrows() != q.ncols() || projected.ncols() != q.ncols() {
            return Err(UmapError::Numerical(
                "malformed full Rayleigh projection dimensions".into(),
            ));
        }
        if (0..q.ncols()).any(|col| {
            projected
                .col_as_slice(col)
                .iter()
                .any(|value| !value.is_finite())
        }) {
            return Err(UmapError::Numerical(
                "non-finite full Rayleigh projection".into(),
            ));
        }
        let eigen = projected.self_adjoint_eigen(Side::Lower).map_err(|error| {
            UmapError::Numerical(format!("projected spectral EVD failed: {error:?}"))
        })?;
        let width = q.ncols();
        if eigen.S().dim() != width || eigen.U().nrows() != width || eigen.U().ncols() != width {
            return Err(UmapError::Numerical(
                "malformed projected spectral EVD dimensions".into(),
            ));
        }
        let values = (0..width)
            .map(|index| eigen.S()[width - 1 - index])
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(UmapError::Numerical(
                "non-finite spectral Ritz value".into(),
            ));
        }
        if values[width - 1] < -1.0e-12 || values[0] > 2.0 + 1.0e-12 {
            return Err(UmapError::Numerical(
                "spectral Ritz values outside the shifted [0,2] spectrum".into(),
            ));
        }
        let rotation = descending_eigenvectors(eigen.U());
        let boundary = values[wanted - 1];
        let cluster_end = (wanted..width)
            .find(|&index| {
                (values[index] - boundary).abs() > CLUSTER_RELATIVE_GAP * boundary.abs().max(1.0)
            })
            .unwrap_or(width);
        stats.cluster_end = cluster_end;
        let residual_search_end = cluster_end.saturating_add(BLOCK_SIZE).min(width);
        let ritz_rotation = leading_columns(&rotation, residual_search_end);
        let ritz_vectors = rotate(&q, &ritz_rotation);
        let ritz_applied = rotate(&aq, &ritz_rotation);
        let mut residual_vectors = Mat::zeros(n, residual_search_end);
        let residuals = (0..residual_search_end)
            .map(|col| {
                let vector = ritz_vectors.col_as_slice(col);
                let applied = ritz_applied.col_as_slice(col);
                for row in 0..n {
                    residual_vectors[(row, col)] = applied[row] - values[col] * vector[row];
                }
                l2_norm(residual_vectors.col_as_slice(col))
            })
            .collect::<Vec<_>>();
        if residuals.iter().any(|residual| !residual.is_finite()) {
            return Err(UmapError::Numerical(
                "non-finite cached spectral Ritz residual".into(),
            ));
        }
        let cached_max_residual = residuals[..cluster_end].iter().copied().fold(0.0, f64::max);
        stats.max_residual = cached_max_residual;
        let preferred_keep = cluster_end
            .saturating_add(BLOCK_SIZE)
            .max(wanted + (max_basis - wanted) / 3);
        let keep = preferred_keep.min(max_basis - BLOCK_SIZE);
        stats.keep = keep;
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.eig_seconds += started.elapsed().as_secs_f64();
        }

        if cluster_end < width && cached_max_residual <= RITZ_TOLERANCE {
            #[cfg(all(test, feature = "tracing"))]
            let started = std::time::Instant::now();
            let vectors = leading_columns(&ritz_vectors, cluster_end);
            #[cfg(all(test, feature = "tracing"))]
            let confirmation_before = stats.confirmation_seconds;
            let confirmed = confirm(
                base,
                &trivial,
                &vectors,
                &values,
                cluster_end,
                &mut ApplyBuffers {
                    packed_input: &mut packed_input,
                    packed_output: &mut packed_output,
                    applied: &mut applied_scratch,
                },
                &mut stats,
            )?;
            #[cfg(all(test, feature = "tracing"))]
            {
                let confirmation_elapsed = stats.confirmation_seconds - confirmation_before;
                stats.eig_seconds += started.elapsed().as_secs_f64() - confirmation_elapsed;
            }
            if confirmed {
                let mut applied_trivial = vec![0.0; n];
                base.shifted_apply(&trivial, &mut applied_trivial);
                let trivial_value = dot(&trivial, &applied_trivial);
                if !trivial_value.is_finite() {
                    return Err(UmapError::Numerical(
                        "non-finite locked trivial eigenvalue".into(),
                    ));
                }
                let mut pairs = vec![(trivial_value, trivial)];
                for (col, &value) in values.iter().take(wanted).enumerate() {
                    pairs.push((value, vectors.col_as_slice(col).to_vec()));
                }
                #[cfg(feature = "tracing")]
                tracing::debug!(
                    cycles = stats.cycles,
                    basis_width = stats.basis_width,
                    keep = stats.keep,
                    cluster_end = stats.cluster_end,
                    max_residual = stats.max_residual,
                    column_applies = stats.column_applies,
                    augmentations = stats.augmentations,
                    reorthogonalizations = stats.reorthogonalizations,
                    skipped_reorthogonalizations = stats.skipped_reorthogonalizations,
                    confirmation_failures = stats.confirmation_failures,
                    "cached block spectral solver converged"
                );
                return Ok(Solution {
                    pairs,
                    #[cfg(all(test, feature = "tracing"))]
                    diagnostics: stats,
                });
            }
        }

        if cycle + 1 == MAX_CYCLES {
            return Err(UmapError::SpectralConvergence(format!(
                "cached block Krylov solver exhausted {MAX_CYCLES} cycles: residual={cached_max_residual}, cluster_end={cluster_end}, width={width}, basis={max_basis}"
            )));
        }

        #[cfg(all(test, feature = "tracing"))]
        let started = std::time::Instant::now();
        let keep_rotation = leading_columns(&rotation, keep);
        rotate_into(&q, &keep_rotation, &mut q_rotation_scratch);
        rotate_into(&aq, &keep_rotation, &mut aq_rotation_scratch);
        std::mem::swap(&mut q, &mut q_rotation_scratch);
        std::mem::swap(&mut aq, &mut aq_rotation_scratch);
        #[cfg(all(test, feature = "tracing"))]
        let rayleigh_started = std::time::Instant::now();
        rayleigh = rotate_rayleigh(&rayleigh, &keep_rotation);
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
        }
        let restart_width = BLOCK_SIZE;
        let mut residual_block = Mat::zeros(n, restart_width);
        let mut residual_count = 0;
        for (col, &residual) in residuals.iter().take(residual_search_end).enumerate() {
            if residual > RITZ_TOLERANCE && residual_count < restart_width {
                residual_block
                    .col_as_slice_mut(residual_count)
                    .copy_from_slice(residual_vectors.col_as_slice(col));
                residual_count += 1;
            }
        }
        for col in residual_count..restart_width {
            for row in 0..n {
                residual_block[(row, col)] = gaussian(rng);
            }
            stats.augmentations += 1;
        }
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.restart_seconds += started.elapsed().as_secs_f64();
        }
        #[cfg(all(test, feature = "tracing"))]
        let started = std::time::Instant::now();
        let frontier = orthogonalize_candidates(
            &residual_block,
            &trivial_column,
            &q,
            restart_width,
            rng,
            &mut stats,
        )?;
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.orthogonalization_seconds += started.elapsed().as_secs_f64();
        }
        append_columns(&mut q, &frontier);
        apply_counted(
            base,
            &frontier,
            &mut packed_input,
            &mut packed_output,
            &mut applied_scratch,
            &mut stats,
        );
        append_columns(&mut aq, &applied_scratch);
        // Restart candidates are Ritz residuals or Gaussian fills, not prior A-images.
        #[cfg(all(test, feature = "tracing"))]
        let rayleigh_started = std::time::Instant::now();
        append_rayleigh_columns(&mut rayleigh, &q, &aq, keep);
        #[cfg(all(test, feature = "tracing"))]
        {
            stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
        }
        frontier_start = keep;
        frontier_width = frontier.ncols();
        let mut final_block_rayleigh_explicit = true;

        while q.ncols() < max_basis {
            let remaining = max_basis - q.ncols();
            let next_width = remaining.min(BLOCK_SIZE);
            let augmentations_before = stats.augmentations;
            let expansion = Mat::from_fn(n, frontier_width, |row, col| {
                aq[(row, frontier_start + col)]
            });
            #[cfg(all(test, feature = "tracing"))]
            let started = std::time::Instant::now();
            let (next, coefficients) = orthogonalize_candidates_with_coefficients(
                &expansion,
                &trivial_column,
                &q,
                next_width,
                rng,
                &mut stats,
                true,
            )?;
            #[cfg(all(test, feature = "tracing"))]
            {
                stats.orthogonalization_seconds += started.elapsed().as_secs_f64();
            }
            set_rayleigh_columns_from_coefficients(&mut rayleigh, &coefficients, frontier_start);
            frontier_start = q.ncols();
            frontier_width = next.ncols();
            let old_width = q.ncols();
            append_columns(&mut q, &next);
            apply_counted(
                base,
                &next,
                &mut packed_input,
                &mut packed_output,
                &mut applied_scratch,
                &mut stats,
            );
            append_columns(&mut aq, &applied_scratch);
            final_block_rayleigh_explicit = stats.augmentations != augmentations_before;
            if final_block_rayleigh_explicit {
                // Redrawn columns were not formed from the previous A-image block.
                #[cfg(all(test, feature = "tracing"))]
                let rayleigh_started = std::time::Instant::now();
                append_rayleigh_columns(&mut rayleigh, &q, &aq, old_width);
                #[cfg(all(test, feature = "tracing"))]
                {
                    stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
                }
            }
        }
        if !final_block_rayleigh_explicit {
            // No later expansion will provide this terminal block's H columns.
            let final_block_start = q.ncols() - frontier_width;
            #[cfg(all(test, feature = "tracing"))]
            let rayleigh_started = std::time::Instant::now();
            append_rayleigh_columns(&mut rayleigh, &q, &aq, final_block_start);
            #[cfg(all(test, feature = "tracing"))]
            {
                stats.rayleigh_seconds += rayleigh_started.elapsed().as_secs_f64();
            }
        }
    }
    unreachable!("bounded spectral cycle loop returns or errors")
}

#[cfg(test)]
mod tests {
    use super::super::SparseGraph;
    use super::*;
    use rand::SeedableRng;

    fn trivial(graph: &SparseGraph) -> Vec<f64> {
        let mut q0 = graph
            .degrees
            .iter()
            .map(|degree| degree.sqrt())
            .collect::<Vec<_>>();
        let norm = l2_norm(&q0);
        for value in &mut q0 {
            *value /= norm;
        }
        q0
    }

    fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
        let mut sum = 0.0;
        let mut correction = 0.0;
        for value in values {
            let next = sum + value;
            correction += if sum.abs() >= value.abs() {
                (sum - next) + value
            } else {
                (value - next) + sum
            };
            sum = next;
        }
        sum + correction
    }

    #[test]
    fn faer_simd_dot_and_norm_cover_tails_offsets_and_cancellation() {
        for length in [1, 3, 7, 17, 65, 4099] {
            let mut aligned_dot_bits = None;
            let mut aligned_norm_bits = None;
            for left_offset in 0..4 {
                for right_offset in 0..4 {
                    let mut left_storage = vec![0.0; length + 4];
                    let mut right_storage = vec![0.0; length + 4];
                    for index in 0..length {
                        left_storage[left_offset + index] = match index % 7 {
                            0 => 1.0e8,
                            1 => -1.0e8,
                            2 => 1.0,
                            3 => -1.0,
                            4 => 1.0e-8,
                            5 => -2.0e-8,
                            _ => 0.25,
                        };
                        right_storage[right_offset + index] = 1.0 + (index % 13) as f64 * 1.0e-4;
                    }
                    let left = &left_storage[left_offset..left_offset + length];
                    let right = &right_storage[right_offset..right_offset + length];
                    let expected_dot =
                        compensated_sum(left.iter().zip(right).map(|(&a, &b)| a * b));
                    let absolute_products = left
                        .iter()
                        .zip(right)
                        .map(|(&a, &b)| (a * b).abs())
                        .sum::<f64>();
                    let actual_dot = dot(left, right);
                    assert!(
                        (actual_dot - expected_dot).abs() <= 2.0e-14 * absolute_products.max(1.0),
                        "dot mismatch at length {length} and offsets {left_offset}/{right_offset}"
                    );
                    assert_eq!(actual_dot.to_bits(), dot(left, right).to_bits());
                    if let Some(expected_bits) = aligned_dot_bits {
                        assert_eq!(actual_dot.to_bits(), expected_bits);
                    } else {
                        aligned_dot_bits = Some(actual_dot.to_bits());
                    }

                    let expected_norm =
                        compensated_sum(left.iter().map(|value| value * value)).sqrt();
                    let actual_norm = l2_norm(left);
                    assert!(
                        (actual_norm - expected_norm).abs() <= 3.0e-14 * expected_norm.max(1.0),
                        "norm mismatch at length {length} and offset {left_offset}"
                    );
                    assert_eq!(actual_norm.to_bits(), l2_norm(left).to_bits());
                    if let Some(expected_bits) = aligned_norm_bits {
                        assert_eq!(actual_norm.to_bits(), expected_bits);
                    } else {
                        aligned_norm_bits = Some(actual_norm.to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn projection_coefficients_match_scalar_reference_and_repeat_exactly() {
        let n = 4099;
        let basis = Mat::from_fn(n, 5, |row, col| {
            1.0 + ((row * 17 + col * 11) % 101) as f64 * 1.0e-4
        });
        let candidates = Mat::from_fn(n, 2, |row, col| {
            0.5 + ((row * 7 + col * 19) % 97) as f64 * 1.0e-4
        });
        let first = projection_coefficients(&basis, &candidates);
        let second = projection_coefficients(&basis, &candidates);
        for candidate_col in 0..candidates.ncols() {
            for basis_col in 0..basis.ncols() {
                let expected = (0..n)
                    .map(|row| basis[(row, basis_col)] * candidates[(row, candidate_col)])
                    .sum::<f64>();
                let actual = first[(basis_col, candidate_col)];
                assert!(
                    (actual - expected).abs() <= 1.0e-12 * expected.abs(),
                    "projection coefficient mismatch at basis column {basis_col}, candidate column {candidate_col}: {actual} vs {expected}"
                );
                assert_eq!(
                    actual.to_bits(),
                    second[(basis_col, candidate_col)].to_bits()
                );
            }
        }
    }

    #[test]
    fn rank_breakdown_redraws_and_keeps_locked_trivial_complement() {
        let n = 16;
        let q0 = vec![1.0 / (n as f64).sqrt(); n];
        let q0_matrix = Mat::from_fn(n, 1, |row, _| q0[row]);
        let candidates = Mat::from_fn(n, BLOCK_SIZE, |row, _| row as f64);
        let mut rng = StdRng::seed_from_u64(71);
        let mut stats = Diagnostics::default();
        let q = orthogonalize_candidates(
            &candidates,
            &q0_matrix,
            &Mat::zeros(n, 0),
            BLOCK_SIZE,
            &mut rng,
            &mut stats,
        )
        .unwrap();
        assert!(stats.augmentations > 0);
        for col in 0..BLOCK_SIZE {
            assert!(dot(&q0, q.col_as_slice(col)).abs() < 1.0e-12);
            for other in 0..BLOCK_SIZE {
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!(
                    (dot(q.col_as_slice(col), q.col_as_slice(other)) - expected).abs() < 1.0e-12
                );
            }
        }
    }

    #[test]
    fn dgks_reorthogonalization_keeps_random_basis_orthogonal() {
        let n = 4099;
        let mut rng = StdRng::seed_from_u64(401);
        let mut q0 = (0..n).map(|_| gaussian(&mut rng)).collect::<Vec<_>>();
        let q0_norm = l2_norm(&q0);
        for value in &mut q0 {
            *value /= q0_norm;
        }
        let trivial = Mat::from_fn(n, 1, |row, _| q0[row]);
        let seed = Mat::from_fn(n, 3, |_, _| gaussian(&mut rng));
        let basis = orthogonalize_candidates(
            &seed,
            &trivial,
            &Mat::zeros(n, 0),
            3,
            &mut rng,
            &mut Diagnostics::default(),
        )
        .unwrap();
        let candidates = Mat::from_fn(n, 2, |row, col| {
            if col == 0 {
                basis[(row, 0)] + gaussian(&mut rng) * 1.0e-9
            } else {
                gaussian(&mut rng)
            }
        });
        let mut stats = Diagnostics::default();
        let q = orthogonalize_candidates(&candidates, &trivial, &basis, 2, &mut rng, &mut stats)
            .unwrap();

        assert!(stats.reorthogonalizations > 0);
        for col in 0..basis.ncols() + q.ncols() {
            let vector = if col < basis.ncols() {
                basis.col_as_slice(col)
            } else {
                q.col_as_slice(col - basis.ncols())
            };
            assert!(dot(&q0, vector).abs() <= 1.0e-12);
            for other in 0..basis.ncols() + q.ncols() {
                let other_vector = if other < basis.ncols() {
                    basis.col_as_slice(other)
                } else {
                    q.col_as_slice(other - basis.ncols())
                };
                let expected = if col == other { 1.0 } else { 0.0 };
                assert!((dot(vector, other_vector) - expected).abs() <= 1.0e-12);
            }
        }
    }

    #[test]
    fn nonfinite_candidate_fails_immediately() {
        let candidates = Mat::from_fn(8, 1, |row, _| if row == 3 { f64::NAN } else { 0.5 });
        let q0 = Mat::from_fn(8, 1, |_, _| 1.0 / 8.0_f64.sqrt());
        let mut rng = StdRng::seed_from_u64(3);
        assert!(
            orthogonalize_candidates(
                &candidates,
                &q0,
                &Mat::zeros(8, 0),
                1,
                &mut rng,
                &mut Diagnostics::default(),
            )
            .is_err()
        );
    }

    #[test]
    fn cached_and_fresh_apply_agree_on_small_connected_path_and_cycle() {
        let n = 32;
        let path = SparseGraph::from_edges(
            n,
            &(0..n - 1).collect::<Vec<_>>(),
            &(1..n).collect::<Vec<_>>(),
            &vec![1.0; n - 1],
        )
        .unwrap();
        let cycle = SparseGraph::from_edges(
            n,
            &(0..n).collect::<Vec<_>>(),
            &(1..=n).map(|row| row % n).collect::<Vec<_>>(),
            &vec![1.0; n],
        )
        .unwrap();
        for graph in [&path, &cycle] {
            let base = BaseOperator::Sparse(graph);
            let q0 = trivial(graph);
            let q0_matrix = Mat::from_fn(n, 1, |row, _| q0[row]);
            let mut rng = StdRng::seed_from_u64(29);
            let candidates = Mat::from_fn(n, BLOCK_SIZE, |_, _| gaussian(&mut rng));
            let q = orthogonalize_candidates(
                &candidates,
                &q0_matrix,
                &Mat::zeros(n, 0),
                BLOCK_SIZE,
                &mut rng,
                &mut Diagnostics::default(),
            )
            .unwrap();
            let mut packed_input = Vec::with_capacity(n * BLOCK_SIZE);
            let mut packed_output = Vec::with_capacity(n * BLOCK_SIZE);
            let mut cached = Mat::zeros(n, 0);
            apply_into(base, &q, &mut packed_input, &mut packed_output, &mut cached);
            for col in 0..BLOCK_SIZE {
                let mut fresh = vec![0.0; n];
                base.shifted_apply(q.col_as_slice(col), &mut fresh);
                let cached_residual =
                    residual_norm(q.col_as_slice(col), cached.col_as_slice(col), 1.0);
                let fresh_residual = residual_norm(q.col_as_slice(col), &fresh, 1.0);
                assert!((cached_residual - fresh_residual).abs() <= 1.0e-14);
                for row in 0..n {
                    assert!((cached[(row, col)] - fresh[row]).abs() <= 1.0e-14);
                }
            }
        }
    }

    #[test]
    fn incremental_rayleigh_matches_fresh_projection_through_restart() {
        let n = 32;
        let graph = SparseGraph::from_edges(
            n,
            &(0..n).collect::<Vec<_>>(),
            &(1..=n).map(|row| row % n).collect::<Vec<_>>(),
            &vec![1.0; n],
        )
        .unwrap();
        let base = BaseOperator::Sparse(&graph);
        let q0 = trivial(&graph);
        let trivial = Mat::from_fn(n, 1, |row, _| q0[row]);
        let mut rng = StdRng::seed_from_u64(812);
        let mut stats = Diagnostics::default();
        let initial = Mat::from_fn(n, BLOCK_SIZE, |_, _| gaussian(&mut rng));
        let mut q = orthogonalize_candidates(
            &initial,
            &trivial,
            &Mat::zeros(n, 0),
            BLOCK_SIZE,
            &mut rng,
            &mut stats,
        )
        .unwrap();
        let mut packed_input = Vec::with_capacity(n * BLOCK_SIZE);
        let mut packed_output = Vec::with_capacity(n * BLOCK_SIZE);
        let mut aq = Mat::zeros(n, 0);
        let mut applied = Mat::zeros(n, 0);
        apply_into(
            base,
            &q,
            &mut packed_input,
            &mut packed_output,
            &mut applied,
        );
        append_columns(&mut aq, &applied);
        let mut rayleigh = Mat::zeros(0, 0);
        append_rayleigh_columns(&mut rayleigh, &q, &aq, 0);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);

        for _ in 0..3 {
            let frontier_start = aq.ncols() - BLOCK_SIZE;
            let frontier = Mat::from_fn(n, BLOCK_SIZE, |row, col| aq[(row, frontier_start + col)]);
            let (next, coefficients) = orthogonalize_candidates_with_coefficients(
                &frontier, &trivial, &q, BLOCK_SIZE, &mut rng, &mut stats, true,
            )
            .unwrap();
            set_rayleigh_columns_from_coefficients(&mut rayleigh, &coefficients, frontier_start);
            assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);
            append_columns(&mut q, &next);
            apply_into(
                base,
                &next,
                &mut packed_input,
                &mut packed_output,
                &mut applied,
            );
            append_columns(&mut aq, &applied);
        }
        let final_block_start = q.ncols() - BLOCK_SIZE;
        append_rayleigh_columns(&mut rayleigh, &q, &aq, final_block_start);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);

        let eigen = rayleigh.self_adjoint_eigen(Side::Lower).unwrap();
        let rotation = descending_eigenvectors(eigen.U());
        let keep_rotation = leading_columns(&rotation, 4);
        q = rotate(&q, &keep_rotation);
        aq = rotate(&aq, &keep_rotation);
        rayleigh = rotate_rayleigh(&rayleigh, &keep_rotation);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);

        let restart = Mat::from_fn(n, BLOCK_SIZE, |_, _| gaussian(&mut rng));
        let frontier =
            orthogonalize_candidates(&restart, &trivial, &q, BLOCK_SIZE, &mut rng, &mut stats)
                .unwrap();
        append_columns(&mut q, &frontier);
        apply_into(
            base,
            &frontier,
            &mut packed_input,
            &mut packed_output,
            &mut applied,
        );
        append_columns(&mut aq, &applied);
        append_rayleigh_columns(&mut rayleigh, &q, &aq, 4);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);

        let frontier_start = 4;
        let expansion = Mat::from_fn(n, BLOCK_SIZE, |row, col| aq[(row, frontier_start + col)]);
        let (next, coefficients) = orthogonalize_candidates_with_coefficients(
            &expansion, &trivial, &q, BLOCK_SIZE, &mut rng, &mut stats, true,
        )
        .unwrap();
        set_rayleigh_columns_from_coefficients(&mut rayleigh, &coefficients, frontier_start);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);
        let old_width = q.ncols();
        append_columns(&mut q, &next);
        apply_into(
            base,
            &next,
            &mut packed_input,
            &mut packed_output,
            &mut applied,
        );
        append_columns(&mut aq, &applied);
        append_rayleigh_columns(&mut rayleigh, &q, &aq, old_width);
        assert_rayleigh_matches_fresh(&q, &aq, &rayleigh);
    }

    fn assert_rayleigh_matches_fresh(q: &Mat<f64>, aq: &Mat<f64>, rayleigh: &Mat<f64>) {
        let mut fresh = Mat::zeros(q.ncols(), q.ncols());
        matmul(
            fresh.as_mut(),
            Accum::Replace,
            q.as_ref().transpose(),
            aq.as_ref(),
            1.0,
            Par::Seq,
        );
        symmetrize(&mut fresh);
        let mut max_difference = 0.0_f64;
        for row in 0..fresh.nrows() {
            for col in 0..fresh.ncols() {
                max_difference =
                    max_difference.max((fresh[(row, col)] - rayleigh[(row, col)]).abs());
            }
        }
        assert!(
            max_difference <= 1.0e-12,
            "Rayleigh cache drift: {max_difference}"
        );
    }

    #[test]
    fn fixed_seed_and_sequential_gemm_are_deterministic() {
        let n = 24;
        let q0 = vec![1.0 / (n as f64).sqrt(); n];
        let q0_matrix = Mat::from_fn(n, 1, |row, _| q0[row]);
        let build = || {
            let mut rng = StdRng::seed_from_u64(927);
            let candidates = Mat::from_fn(n, BLOCK_SIZE, |_, _| gaussian(&mut rng));
            orthogonalize_candidates(
                &candidates,
                &q0_matrix,
                &Mat::zeros(n, 0),
                BLOCK_SIZE,
                &mut rng,
                &mut Diagnostics::default(),
            )
            .unwrap()
        };
        let first = build();
        let second = build();
        for col in 0..BLOCK_SIZE {
            assert_eq!(first.col_as_slice(col), second.col_as_slice(col));
        }
    }
}
