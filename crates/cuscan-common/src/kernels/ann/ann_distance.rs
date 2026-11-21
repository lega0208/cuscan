//! Canonical R8 squared Euclidean distance used by every ANN distance path.

#[cfg(test)]
#[path = "ann_distance_tests.rs"]
mod tests;

use cubecl::prelude::*;

/// Accumulate features in eight independent chains and reduce them in the
/// fixed R8 tree. Width one uses scalar chains; widths four/eight load vector
/// chunks and preserve the same per-lane R8 order.
// Keep compile-time width specialization separate from runtime chunk parity.
#[allow(clippy::if_same_then_else)]
#[cube]
pub(super) fn squared_l2_r8<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right: usize,
    features: usize,
    #[comptime] vector_width: usize,
) -> F {
    let mut s0 = F::new(0.0f32);
    let mut s1 = F::new(0.0f32);
    let mut s2 = F::new(0.0f32);
    let mut s3 = F::new(0.0f32);
    let mut s4 = F::new(0.0f32);
    let mut s5 = F::new(0.0f32);
    let mut s6 = F::new(0.0f32);
    let mut s7 = F::new(0.0f32);
    if comptime![vector_width == 1] {
        let chunks = features / 8;
        for chunk in 0..chunks {
            let base = chunk * 8;
            let d0 = data[left * features + base].extract(0usize)
                - data[right * features + base].extract(0usize);
            let d1 = data[left * features + base + 1].extract(0usize)
                - data[right * features + base + 1].extract(0usize);
            let d2 = data[left * features + base + 2].extract(0usize)
                - data[right * features + base + 2].extract(0usize);
            let d3 = data[left * features + base + 3].extract(0usize)
                - data[right * features + base + 3].extract(0usize);
            let d4 = data[left * features + base + 4].extract(0usize)
                - data[right * features + base + 4].extract(0usize);
            let d5 = data[left * features + base + 5].extract(0usize)
                - data[right * features + base + 5].extract(0usize);
            let d6 = data[left * features + base + 6].extract(0usize)
                - data[right * features + base + 6].extract(0usize);
            let d7 = data[left * features + base + 7].extract(0usize)
                - data[right * features + base + 7].extract(0usize);
            s0 = fma(d0, d0, s0);
            s1 = fma(d1, d1, s1);
            s2 = fma(d2, d2, s2);
            s3 = fma(d3, d3, s3);
            s4 = fma(d4, d4, s4);
            s5 = fma(d5, d5, s5);
            s6 = fma(d6, d6, s6);
            s7 = fma(d7, d7, s7);
        }
        let tail = chunks * 8;
        if tail < features {
            let d = data[left * features + tail].extract(0usize)
                - data[right * features + tail].extract(0usize);
            s0 = fma(d, d, s0);
        }
        if tail + 1 < features {
            let d = data[left * features + tail + 1].extract(0usize)
                - data[right * features + tail + 1].extract(0usize);
            s1 = fma(d, d, s1);
        }
        if tail + 2 < features {
            let d = data[left * features + tail + 2].extract(0usize)
                - data[right * features + tail + 2].extract(0usize);
            s2 = fma(d, d, s2);
        }
        if tail + 3 < features {
            let d = data[left * features + tail + 3].extract(0usize)
                - data[right * features + tail + 3].extract(0usize);
            s3 = fma(d, d, s3);
        }
        if tail + 4 < features {
            let d = data[left * features + tail + 4].extract(0usize)
                - data[right * features + tail + 4].extract(0usize);
            s4 = fma(d, d, s4);
        }
        if tail + 5 < features {
            let d = data[left * features + tail + 5].extract(0usize)
                - data[right * features + tail + 5].extract(0usize);
            s5 = fma(d, d, s5);
        }
        if tail + 6 < features {
            let d = data[left * features + tail + 6].extract(0usize)
                - data[right * features + tail + 6].extract(0usize);
            s6 = fma(d, d, s6);
        }
    } else {
        let lanes = vector_width;
        let vector_count = features / lanes;
        let mut acc0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut acc1 = Vector::<F, N>::new(F::new(0.0f32));
        let row_vectors = features / lanes;
        for chunk in 0..vector_count {
            let delta = data[left * row_vectors + chunk] - data[right * row_vectors + chunk];
            if comptime![vector_width == 8] {
                acc0 = fma(delta, delta, acc0);
            } else if chunk % 2 == 0 {
                acc0 = fma(delta, delta, acc0);
            } else {
                acc1 = fma(delta, delta, acc1);
            }
        }
        if comptime![vector_width == 8] {
            s0 = acc0.extract(0usize);
            s1 = acc0.extract(1usize);
            s2 = acc0.extract(2usize);
            s3 = acc0.extract(3usize);
            s4 = acc0.extract(4usize);
            s5 = acc0.extract(5usize);
            s6 = acc0.extract(6usize);
            s7 = acc0.extract(7usize);
        } else {
            s0 = acc0.extract(0usize);
            s1 = acc0.extract(1usize);
            s2 = acc0.extract(2usize);
            s3 = acc0.extract(3usize);
            s4 = acc1.extract(0usize);
            s5 = acc1.extract(1usize);
            s6 = acc1.extract(2usize);
            s7 = acc1.extract(3usize);
        }
    }
    ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
}

/// Four independent fixed-R8 distances. Width one deliberately repeats the
/// scalar pair kernel; vector widths share the left-hand vector loads.
// Keep compile-time width specialization separate from runtime chunk parity.
#[allow(clippy::if_same_then_else)]
#[cube]
pub(super) fn squared_l2_r8_x4<F: Float, N: Size>(
    data: &Tensor<Vector<F, N>>,
    left: usize,
    right0: usize,
    right1: usize,
    right2: usize,
    right3: usize,
    features: usize,
    #[comptime] vector_width: usize,
) -> (F, F, F, F) {
    if comptime![vector_width == 1] {
        (
            squared_l2_r8::<F, N>(data, left, right0, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right1, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right2, features, vector_width),
            squared_l2_r8::<F, N>(data, left, right3, features, vector_width),
        )
    } else {
        let mut a0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a2 = Vector::<F, N>::new(F::new(0.0f32));
        let mut a3 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b0 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b1 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b2 = Vector::<F, N>::new(F::new(0.0f32));
        let mut b3 = Vector::<F, N>::new(F::new(0.0f32));
        let row_vectors = features / vector_width;
        for chunk in 0..row_vectors {
            let value = data[left * row_vectors + chunk];
            let d0 = value - data[right0 * row_vectors + chunk];
            let d1 = value - data[right1 * row_vectors + chunk];
            let d2 = value - data[right2 * row_vectors + chunk];
            let d3 = value - data[right3 * row_vectors + chunk];
            if comptime![vector_width == 8] {
                a0 = fma(d0, d0, a0);
                a1 = fma(d1, d1, a1);
                a2 = fma(d2, d2, a2);
                a3 = fma(d3, d3, a3);
            } else if chunk % 2 == 0 {
                a0 = fma(d0, d0, a0);
                a1 = fma(d1, d1, a1);
                a2 = fma(d2, d2, a2);
                a3 = fma(d3, d3, a3);
            } else {
                b0 = fma(d0, d0, b0);
                b1 = fma(d1, d1, b1);
                b2 = fma(d2, d2, b2);
                b3 = fma(d3, d3, b3);
            }
        }
        (
            reduce_r8_vectors::<F, N>(a0, b0, vector_width),
            reduce_r8_vectors::<F, N>(a1, b1, vector_width),
            reduce_r8_vectors::<F, N>(a2, b2, vector_width),
            reduce_r8_vectors::<F, N>(a3, b3, vector_width),
        )
    }
}

#[cube]
fn reduce_r8_vectors<F: Float, N: Size>(
    a: Vector<F, N>,
    b: Vector<F, N>,
    #[comptime] width: usize,
) -> F {
    let low = (a.extract(0usize) + a.extract(1usize)) + (a.extract(2usize) + a.extract(3usize));
    if comptime![width == 8] {
        low + ((a.extract(4usize) + a.extract(5usize)) + (a.extract(6usize) + a.extract(7usize)))
    } else {
        low + ((b.extract(0usize) + b.extract(1usize)) + (b.extract(2usize) + b.extract(3usize)))
    }
}

#[cfg(test)]
pub(super) fn host_squared_l2_r8(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sums = [0.0f32; 8];
    for (feature, (&left, &right)) in a.iter().zip(b).enumerate() {
        let difference = left - right;
        sums[feature % 8] = difference.mul_add(difference, sums[feature % 8]);
    }
    ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]))
}
