//! Host-side geometry for recursive initialization.

use super::graph::HostCsr;
use crate::{Result, UmapError};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::matmul::matmul;
use faer::linalg::{householder, qr::no_pivoting::factor, svd};
use faer::{Accum, Conj, Mat, MatRef, Par, Spec};
use rand::RngExt;
use rand::rngs::StdRng;

/// Row-major coordinates; only rows selected by `mask` contain anchor data.
pub(crate) struct Anchor {
    pub(crate) values: Vec<f32>,
    pub(crate) mask: Vec<bool>,
    pub(crate) dimensions: usize,
}

fn zeros<T: Default + Clone>(len: usize, name: &'static str) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| UmapError::Memory(format!("cannot allocate {name}")))?;
    v.resize(len, T::default());
    Ok(v)
}

fn size(a: usize, b: usize, name: &'static str) -> Result<usize> {
    a.checked_mul(b).ok_or(UmapError::Overflow(name))
}

// faer 0.24 pads each f64 matrix column to a 64-byte boundary.
fn matrix_bytes(rows: usize, cols: usize) -> Result<usize> {
    let padded = rows
        .checked_add(7)
        .ok_or(UmapError::Overflow("geometry matrix"))?
        / 8
        * 8;
    let bytes = size(size(padded, cols, "geometry matrix")?, 8, "geometry matrix")?;
    if bytes > isize::MAX as usize {
        return Err(UmapError::Overflow("geometry matrix"));
    }
    Ok(bytes)
}

fn sum_bytes(parts: &[usize]) -> Result<usize> {
    parts.iter().try_fold(0usize, |a, &b| {
        a.checked_add(b)
            .ok_or(UmapError::Overflow("geometry allocations"))
    })
}

fn scratch_bytes(req: faer::dyn_stack::StackReq) -> Result<usize> {
    if req.align_bytes() == 0 || req.size_bytes() > isize::MAX as usize {
        return Err(UmapError::Overflow("geometry scratch"));
    }
    Ok(req.size_bytes())
}

fn svd_sizes(m: usize, n: usize) -> Result<usize> {
    let k = m.min(n);
    let u = matrix_bytes(m, k)?;
    let v = matrix_bytes(n, k)?;
    let diagonal = size(k, 8, "SVD diagonal")?;
    let scratch = std::panic::catch_unwind(|| {
        svd::svd_scratch::<f64>(
            m,
            n,
            svd::ComputeSvdVectors::Thin,
            svd::ComputeSvdVectors::Thin,
            Par::Seq,
            Spec::default(),
        )
    })
    .map_err(|_| UmapError::Overflow("SVD scratch"))?;
    sum_bytes(&[u, v, diagonal, scratch_bytes(scratch)?])
}

/// Peak live anchor-stage allocations, excluding caller-owned input data.
pub(crate) fn anchor_allocation_bytes(
    n: usize,
    features: usize,
    dimensions: usize,
    sample_size: usize,
) -> Result<usize> {
    let s = n.min(sample_size);
    let m = s.max(features);
    let p = s.min(features);
    let width = dimensions
        .checked_add(8)
        .ok_or(UmapError::Overflow("anchor oversamples"))?;
    let k = m.min(width);
    let block = factor::recommended_block_size::<f64>(m, width);
    let qr_coeff = matrix_bytes(block, k)?;
    let qr_scratch = std::panic::catch_unwind(|| {
        factor::qr_in_place_scratch::<f64>(m, width, block, Par::Seq, Spec::default())
    })
    .map_err(|_| UmapError::Overflow("QR scratch"))?;
    let householder_scratch = std::panic::catch_unwind(|| {
        householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
            m, block, k,
        )
    })
    .map_err(|_| UmapError::Overflow("Householder scratch"))?;
    let retained = sum_bytes(&[
        size(s, size_of::<usize>(), "anchor sample")?,
        n,
        size(features, 8, "anchor means")?,
        matrix_bytes(m, p)?,
        matrix_bytes(p, width)?,
        matrix_bytes(m, width)?,
        matrix_bytes(p, width)?,
    ])?;
    let q = matrix_bytes(m, k)?;
    let qr_peak = sum_bytes(&[
        retained,
        matrix_bytes(m, width)?,
        qr_coeff,
        scratch_bytes(qr_scratch)?,
        q,
        scratch_bytes(householder_scratch)?,
    ])?;
    let after_q = sum_bytes(&[retained, q, matrix_bytes(k, p)?])?;
    let svd_peak = sum_bytes(&[after_q, svd_sizes(k, p)?])?;
    let r = k.min(p);
    let final_peak = sum_bytes(&[
        after_q,
        matrix_bytes(k, r)?,
        matrix_bytes(p, r)?,
        size(r, 8, "SVD diagonal")?,
        matrix_bytes(m, r)?,
        size(
            size(n, dimensions, "anchor coordinates")?,
            4,
            "anchor coordinates",
        )?,
    ])?;
    Ok(qr_peak.max(svd_peak).max(final_peak))
}

/// Peak live Procrustes allocations, excluding the layout and anchor.
pub(crate) fn procrustes_allocation_bytes(d: usize) -> Result<usize> {
    let mat = matrix_bytes(d, d)?;
    let means = size(d, 16, "procrustes means")?;
    let svd_peak = sum_bytes(&[means, mat, svd_sizes(d, d)?])?;
    let rotation_peak = sum_bytes(&[
        means,
        mat,
        mat,
        mat,
        mat,
        mat,
        mat,
        size(d, 4, "procrustes row")?,
    ])?;
    Ok(svd_peak.max(rotation_peak))
}

fn matrix(rows: usize, cols: usize, name: &'static str) -> Result<Mat<f64>> {
    let mut result = Mat::new();
    result
        .try_reserve(rows, cols)
        .map_err(|_| UmapError::Memory(format!("cannot allocate {name}")))?;
    result.resize_with(rows, cols, |_, _| 0.0);
    Ok(result)
}

fn product(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Result<Mat<f64>> {
    assert_eq!(a.ncols(), b.nrows());
    let mut result = matrix(a.nrows(), b.ncols(), "geometry product")?;
    // faer 0.24.4 uses allocation-free nano-gemm for products of at most
    // 16^3 scalar operations; larger products can allocate an infallible
    // thread-local packing buffer even with Par::Seq. Keep each call below
    // that threshold while retaining its SIMD kernels. The output matrix is
    // the only product allocation and is already included in stage sizing.
    const TILE: usize = 16;
    for col in (0..b.ncols()).step_by(TILE) {
        let cols = (b.ncols() - col).min(TILE);
        for row in (0..a.nrows()).step_by(TILE) {
            let rows = (a.nrows() - row).min(TILE);
            for inner in (0..a.ncols()).step_by(TILE) {
                let depth = (a.ncols() - inner).min(TILE);
                matmul(
                    result.as_mut().submatrix_mut(row, col, rows, cols),
                    if inner == 0 {
                        Accum::Replace
                    } else {
                        Accum::Add
                    },
                    a.submatrix(row, inner, rows, depth),
                    b.submatrix(inner, col, depth, cols),
                    1.0,
                    Par::Seq,
                );
            }
        }
    }
    Ok(result)
}

// The low-level faer calls explicitly use sequential execution, independently of
// the global rayon pool. Thin factors avoid quadratic allocations in row count.
fn thin_svd(a: &Mat<f64>) -> Result<(Mat<f64>, Vec<f64>, Mat<f64>)> {
    let (m, n) = a.shape();
    let k = m.min(n);
    let mut u = matrix(m, k, "SVD left factor")?;
    let mut s = zeros::<f64>(k, "SVD singular values")?;
    let mut v = matrix(n, k, "SVD right factor")?;
    let req = svd::svd_scratch::<f64>(
        m,
        n,
        svd::ComputeSvdVectors::Thin,
        svd::ComputeSvdVectors::Thin,
        Par::Seq,
        Spec::default(),
    );
    let mut buffer = MemBuffer::try_new(req)
        .map_err(|_| UmapError::Memory("cannot allocate SVD scratch".into()))?;
    svd::svd(
        a.as_ref(),
        faer::ColMut::from_slice_mut(&mut s).as_diagonal_mut(),
        Some(u.as_mut()),
        Some(v.as_mut()),
        Par::Seq,
        MemStack::new(&mut buffer),
        Spec::default(),
    )
    .map_err(|e| UmapError::Numerical(format!("recursive SVD failed: {e:?}")))?;
    Ok((u, s, v))
}

fn thin_q(mut y: Mat<f64>) -> Result<Mat<f64>> {
    let (m, n) = y.shape();
    let k = m.min(n);
    let block = factor::recommended_block_size::<f64>(m, n);
    let mut coeff = matrix(block, k, "QR coefficients")?;
    let req = factor::qr_in_place_scratch::<f64>(m, n, block, Par::Seq, Spec::default());
    let mut buffer = MemBuffer::try_new(req)
        .map_err(|_| UmapError::Memory("cannot allocate QR scratch".into()))?;
    factor::qr_in_place(
        y.as_mut(),
        coeff.as_mut(),
        Par::Seq,
        MemStack::new(&mut buffer),
        Spec::default(),
    );
    let mut q = matrix(m, k, "QR factor")?;
    for i in 0..k {
        q[(i, i)] = 1.0;
    }
    let req = householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(
        m, block, k,
    );
    let mut buffer = MemBuffer::try_new(req)
        .map_err(|_| UmapError::Memory("cannot allocate Householder scratch".into()))?;
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        y.as_ref(),
        coeff.as_ref(),
        Conj::No,
        q.as_mut(),
        Par::Seq,
        MemStack::new(&mut buffer),
    );
    Ok(q)
}

/// Construct a sampled PCA anchor. `data` is a row-major n by features matrix.
pub(crate) fn anchor_reference(
    data: &[f32],
    n: usize,
    features: usize,
    dimensions: usize,
    sample_size: usize,
    rng: &mut StdRng,
) -> Result<Anchor> {
    if n == 0
        || features == 0
        || dimensions == 0
        || sample_size == 0
        || data.len() != size(n, features, "anchor input")?
    {
        return Err(UmapError::InvalidInput(
            "invalid recursive anchor shape".into(),
        ));
    }
    let s = n.min(sample_size);
    let mut sample = zeros::<usize>(s, "anchor sample")?;
    for (i, row) in sample.iter_mut().enumerate() {
        *row = i;
    }
    for i in s..n {
        let j = rng.random_range(0..=i);
        if j < s {
            sample[j] = i;
        }
    }
    sample.sort_unstable();
    let mut mask = zeros::<bool>(n, "anchor mask")?;
    for &row in &sample {
        mask[row] = true;
    }
    let mut means = zeros::<f64>(features, "anchor means")?;
    for &row in &sample {
        for j in 0..features {
            means[j] += f64::from(data[row * features + j]);
        }
    }
    for x in &mut means {
        *x /= s as f64;
    }
    let transposed = s < features;
    let (m, p) = if transposed {
        (features, s)
    } else {
        (s, features)
    };
    let mut matrix = self::matrix(m, p, "centered anchor sample")?;
    for i in 0..m {
        for j in 0..p {
            let (row, col) = if transposed { (j, i) } else { (i, j) };
            matrix[(i, j)] = f64::from(data[sample[row] * features + col]) - means[col];
        }
    }
    let width = dimensions
        .checked_add(8)
        .ok_or(UmapError::Overflow("anchor oversamples"))?;
    let mut omega = self::matrix(p, width, "anchor Gaussian matrix")?;
    let mut spare = None;
    for i in 0..p {
        for j in 0..width {
            let value = match spare.take() {
                Some(x) => x,
                None => {
                    let (x, y) = super::gaussian_pair(rng);
                    spare = Some(y);
                    x
                }
            };
            omega[(i, j)] = value;
        }
    }
    // Power iteration without intermediate normalization: Y = M(Mᵀ(MΩ)).
    let first = product(matrix.as_ref(), omega.as_ref())?;
    let power = product(matrix.as_ref().transpose(), first.as_ref())?;
    let q = thin_q(product(matrix.as_ref(), power.as_ref())?)?;
    let b = product(q.as_ref().transpose(), matrix.as_ref())?;
    let (u, singular, v) = thin_svd(&b)?;
    let rank = dimensions.min(s).min(features);
    let projected = product(q.as_ref(), u.as_ref())?;
    let mut values = zeros::<f32>(
        size(n, dimensions, "anchor coordinates")?,
        "anchor coordinates",
    )?;
    for axis in 0..rank {
        // sklearn svd_flip chooses the largest-magnitude entry in the left
        // singular vector before a transpose, and the right one after it.
        let len = if transposed { s } else { m };
        let mut pivot = 0;
        for i in 1..len {
            let x = if transposed {
                v[(i, axis)]
            } else {
                projected[(i, axis)]
            };
            let best = if transposed {
                v[(pivot, axis)]
            } else {
                projected[(pivot, axis)]
            };
            if x.abs() > best.abs() {
                pivot = i;
            }
        }
        let pivot_value = if transposed {
            v[(pivot, axis)]
        } else {
            projected[(pivot, axis)]
        };
        let sign = if pivot_value < 0.0 { -1.0 } else { 1.0 };
        let sigma = singular[axis];
        for (i, &row) in sample.iter().enumerate() {
            let coordinate = if transposed {
                v[(i, axis)]
            } else {
                projected[(i, axis)]
            };
            values[row * dimensions + axis] = (sign * sigma * coordinate) as f32;
            if !values[row * dimensions + axis].is_finite() {
                return Err(UmapError::Numerical(
                    "non-finite recursive PCA score".into(),
                ));
            }
        }
        let min = sample
            .iter()
            .map(|&r| values[r * dimensions + axis])
            .fold(f32::INFINITY, f32::min);
        let mut span = 0.0f32;
        for &row in &sample {
            let x = &mut values[row * dimensions + axis];
            *x -= min;
            span = span.max(*x);
        }
        if !span.is_finite() {
            return Err(UmapError::Numerical(
                "non-finite recursive anchor span".into(),
            ));
        }
        if span == 0.0 {
            span = 1.0;
        }
        for &row in &sample {
            let x = &mut values[row * dimensions + axis];
            *x /= span;
            *x *= 10.0;
        }
    }
    for &row in &sample {
        for axis in rank..dimensions {
            values[row * dimensions + axis] = rng.random_range(0.0f32..10.0);
        }
    }
    Ok(Anchor {
        values,
        mask,
        dimensions,
    })
}

/// Average sampled members of each part, retaining a mask for empty parts.
pub(crate) fn coarse_anchor(anchor: &Anchor, labels: &[i32], parts: usize) -> Result<Anchor> {
    let d = anchor.dimensions;
    let mut sums = zeros::<f64>(size(parts, d, "coarse anchor")?, "coarse anchor sums")?;
    let mut counts = zeros::<usize>(parts, "coarse anchor counts")?;
    for (row, &label) in labels.iter().enumerate() {
        let part = label as usize;
        if anchor.mask[row] {
            counts[part] += 1;
            for j in 0..d {
                sums[part * d + j] += f64::from(anchor.values[row * d + j]);
            }
        }
    }
    let mut values = zeros::<f32>(sums.len(), "coarse anchor")?;
    let mut mask = zeros::<bool>(parts, "coarse anchor mask")?;
    for p in 0..parts {
        if counts[p] > 0 {
            mask[p] = true;
            for j in 0..d {
                values[p * d + j] = (sums[p * d + j] / counts[p] as f64) as f32;
            }
        }
    }
    Ok(Anchor {
        values,
        mask,
        dimensions: d,
    })
}

/// Fill unsampled rows and apply the depth-dependent robust scaling.
pub(crate) fn base_case(graph: &HostCsr, anchor: &Anchor, depth: usize) -> Result<Vec<f32>> {
    let n = graph.n;
    let d = anchor.dimensions;
    let mut result = zeros::<f64>(size(n, d, "base layout")?, "base layout")?;
    let mut mean = zeros::<f64>(d, "anchor mean")?;
    let count = anchor.mask.iter().filter(|&&x| x).count();
    if count == 0 {
        return Err(UmapError::InvalidInput(
            "recursive anchor has no sampled rows".into(),
        ));
    }
    for row in 0..n {
        if anchor.mask[row] {
            for (j, row_mean) in mean.iter_mut().enumerate().take(d) {
                *row_mean += f64::from(anchor.values[row * d + j]);
            }
        }
    }
    for x in &mut mean {
        *x /= count as f64;
    }
    for row in 0..n {
        if anchor.mask[row] {
            for j in 0..d {
                result[row * d + j] = f64::from(anchor.values[row * d + j]);
            }
            continue;
        }
        let mut total = 0.0;
        for edge in graph.indptr[row]..graph.indptr[row + 1] {
            let neighbor = graph.indices[edge];
            if anchor.mask[neighbor] {
                let w = f64::from(graph.weights[edge]);
                total += w;
                for j in 0..d {
                    result[row * d + j] += w * f64::from(anchor.values[neighbor * d + j]);
                }
            }
        }
        for j in 0..d {
            result[row * d + j] = if total > 0.0 {
                result[row * d + j] / total
            } else {
                mean[j]
            };
        }
    }
    let mut sorted = zeros::<f64>(n, "base quantiles")?;
    let scale = (n as f64).log10() * 3.0 * ((depth + 1) as f64).log2();
    for j in 0..d {
        let avg = (0..n).map(|i| result[i * d + j]).sum::<f64>() / n as f64;
        for i in 0..n {
            result[i * d + j] -= avg;
            sorted[i] = result[i * d + j];
        }
        sorted.sort_unstable_by(f64::total_cmp);
        let quantile = |q: f64| {
            let pos = q * (n - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = pos.ceil() as usize;
            sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
        };
        let spread = quantile(0.95) - quantile(0.05);
        let multiplier = scale / if spread == 0.0 { 1.0 } else { spread };
        for i in 0..n {
            result[i * d + j] *= multiplier;
        }
    }
    let mut out = zeros::<f32>(result.len(), "base layout output")?;
    for (x, y) in out.iter_mut().zip(result) {
        *x = y as f32;
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err(UmapError::Numerical(
            "non-finite recursive base layout".into(),
        ));
    }
    Ok(out)
}

/// Prolongate a coarse layout and subtract its column means.
pub(crate) fn prolongate(
    graph: &HostCsr,
    labels: &[i32],
    coarse: &[f32],
    d: usize,
) -> Result<Vec<f32>> {
    let n = graph.n;
    let mut result = zeros::<f64>(size(n, d, "prolongation")?, "prolongation")?;
    for row in 0..n {
        let mut total = 0.0;
        for edge in graph.indptr[row]..graph.indptr[row + 1] {
            let w = f64::from(graph.weights[edge]);
            total += w;
            let p = labels[graph.indices[edge]] as usize;
            for j in 0..d {
                result[row * d + j] += w * f64::from(coarse[p * d + j]);
            }
        }
        for j in 0..d {
            result[row * d + j] = 0.5
                * (if total > 0.0 {
                    result[row * d + j] / total
                } else {
                    0.0
                } + f64::from(coarse[labels[row] as usize * d + j]));
        }
    }
    let mut out = zeros::<f32>(result.len(), "prolongation output")?;
    for j in 0..d {
        let mean = (0..n).map(|i| result[i * d + j]).sum::<f64>() / n as f64;
        for i in 0..n {
            out[i * d + j] = (result[i * d + j] - mean) as f32;
        }
    }
    if !out.iter().all(|x| x.is_finite()) {
        return Err(UmapError::Numerical(
            "non-finite recursive prolongation".into(),
        ));
    }
    Ok(out)
}

/// Rotate all layout rows to align sampled rows to the anchor, without reflection.
pub(crate) fn procrustes_align(layout: &mut [f32], anchor: &Anchor) -> Result<()> {
    let d = anchor.dimensions;
    let n = anchor.mask.len();
    let count = anchor.mask.iter().filter(|&&x| x).count();
    if count == 0 {
        return Err(UmapError::InvalidInput(
            "recursive anchor has no sampled rows".into(),
        ));
    }
    let mut a_mean = zeros::<f64>(d, "procrustes anchor mean")?;
    let mut x_mean = zeros::<f64>(d, "procrustes layout mean")?;
    for i in 0..n {
        if anchor.mask[i] {
            for j in 0..d {
                a_mean[j] += f64::from(anchor.values[i * d + j]);
                x_mean[j] += f64::from(layout[i * d + j]);
            }
        }
    }
    for j in 0..d {
        a_mean[j] /= count as f64;
        x_mean[j] /= count as f64;
    }
    let mut a_norm = 0.0;
    let mut x_norm = 0.0;
    for i in 0..n {
        if anchor.mask[i] {
            for j in 0..d {
                a_norm += (f64::from(anchor.values[i * d + j]) - a_mean[j]).powi(2);
                x_norm += (f64::from(layout[i * d + j]) - x_mean[j]).powi(2);
            }
        }
    }
    if a_norm == 0.0 || x_norm == 0.0 {
        return Ok(());
    }
    let scale = (a_norm * x_norm).sqrt();
    let mut covariance = matrix(d, d, "procrustes covariance")?;
    for j in 0..d {
        for k in 0..d {
            covariance[(j, k)] = {
                (0..n)
                    .filter(|&i| anchor.mask[i])
                    .map(|i| {
                        (f64::from(layout[i * d + j]) - x_mean[j])
                            * (f64::from(anchor.values[i * d + k]) - a_mean[k])
                    })
                    .sum::<f64>()
                    / scale
            };
        }
    }
    let (mut u, _, v) = thin_svd(&covariance)?;
    let mut rotation = product(u.as_ref(), v.as_ref().transpose())?;
    // Pivoted elimination determines the sign without changing the rotation.
    let mut work = matrix(d, d, "procrustes determinant")?;
    work.as_mut().copy_from(rotation.as_ref());
    let mut determinant = 1.0;
    for j in 0..d {
        let mut pivot = j;
        for i in j + 1..d {
            if work[(i, j)].abs() > work[(pivot, j)].abs() {
                pivot = i;
            }
        }
        if pivot != j {
            for k in j..d {
                let t = work[(j, k)];
                work[(j, k)] = work[(pivot, k)];
                work[(pivot, k)] = t;
            }
            determinant = -determinant;
        }
        let diag = work[(j, j)];
        determinant *= diag;
        if diag != 0.0 {
            for i in j + 1..d {
                let factor = work[(i, j)] / diag;
                for k in j + 1..d {
                    work[(i, k)] -= factor * work[(j, k)];
                }
            }
        }
    }
    if determinant < 0.0 {
        for i in 0..d {
            u[(i, d - 1)] = -u[(i, d - 1)];
        }
        rotation = product(u.as_ref(), v.as_ref().transpose())?;
    }
    let mut row = zeros::<f32>(d, "procrustes row")?;
    for i in 0..n {
        for j in 0..d {
            row[j] = (0..d)
                .map(|k| f64::from(layout[i * d + k]) * rotation[(k, j)])
                .sum::<f64>() as f32;
        }
        if !row.iter().all(|x| x.is_finite()) {
            return Err(UmapError::Numerical("non-finite recursive rotation".into()));
        }
        layout[i * d..(i + 1) * d].copy_from_slice(&row);
    }
    Ok(())
}

#[cfg(test)]
mod sizing_tests {
    use super::*;

    #[test]
    fn tiled_product_handles_transposes_and_partial_tiles() {
        let mut left = matrix(35, 19, "test").unwrap();
        let mut right = matrix(23, 19, "test").unwrap();
        for j in 0..19 {
            for i in 0..35 {
                left[(i, j)] = (i as f64 - j as f64) / 7.0;
            }
            for i in 0..23 {
                right[(i, j)] = (i as f64 + j as f64) / 11.0;
            }
        }
        let actual = product(left.as_ref(), right.as_ref().transpose()).unwrap();
        for j in 0..23 {
            for i in 0..35 {
                let expected: f64 = (0..19).map(|k| left[(i, k)] * right[(j, k)]).sum();
                assert!((actual[(i, j)] - expected).abs() < 1e-10);
            }
        }
        let empty = product(
            matrix(3, 0, "test").unwrap().as_ref(),
            matrix(0, 2, "test").unwrap().as_ref(),
        )
        .unwrap();
        assert_eq!(empty[(2, 1)], 0.0);
    }

    #[test]
    fn matrix_sizing_accounts_for_faer_column_padding() {
        for (rows, cols) in [(1, 3), (8, 2), (9, 4)] {
            let mat = matrix(rows, cols, "test").unwrap();
            assert_eq!(
                matrix_bytes(rows, cols).unwrap(),
                mat.as_ref().col_stride() as usize * cols * 8
            );
        }
    }

    #[test]
    fn geometry_sizing_checks_overflow_and_both_anchor_orientations() {
        assert!(anchor_allocation_bytes(32, 9, 2, 16).unwrap() > 32 * 2 * 4);
        assert!(anchor_allocation_bytes(32, 31, 2, 8).unwrap() > 32 * 2 * 4);
        assert!(anchor_allocation_bytes(usize::MAX, 3, 2, 1).is_err());
        assert!(anchor_allocation_bytes(8, 3, usize::MAX, 4).is_err());
        assert!(procrustes_allocation_bytes(usize::MAX).is_err());
    }
}
