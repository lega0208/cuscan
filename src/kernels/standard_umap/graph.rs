use burn::{
    backend::{
        TensorMetadata,
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
    },
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

const GLOBAL_SUM_CHUNK_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq)]
struct SmoothThresholds {
    floor_connectivity: u32,
    fraction: f32,
    minimum_positive: u32,
    interpolate: bool,
}

fn smooth_thresholds(local_connectivity: f32, k: usize) -> SmoothThresholds {
    let floor_connectivity = local_connectivity as usize;
    let fraction = local_connectivity - floor_connectivity as f32;
    let minimum_positive = (local_connectivity as usize).saturating_add(
        if local_connectivity > floor_connectivity as f32 {
            1
        } else {
            0
        },
    );
    // k is bounded by the checked COO capacity (<= i32::MAX / 2), so k + 1
    // fits in u32. Any threshold above k is behaviorally equivalent: no row
    // can meet it, and rank searches cannot select a rank beyond the row.
    let above_width = k + 1;
    SmoothThresholds {
        floor_connectivity: u32::try_from(floor_connectivity.min(above_width)).unwrap(),
        fraction,
        minimum_positive: u32::try_from(minimum_positive.min(above_width)).unwrap(),
        interpolate: fraction > 1e-5f32,
    }
}

#[cfg(test)]
mod tests {
    use super::{SmoothThresholds, smooth_thresholds};

    #[test]
    fn smooth_thresholds_preserve_f32_boundary_expressions() {
        for (local_connectivity, expected) in [
            (
                0.0,
                SmoothThresholds {
                    floor_connectivity: 0,
                    fraction: 0.0,
                    minimum_positive: 0,
                    interpolate: false,
                },
            ),
            (
                1.0,
                SmoothThresholds {
                    floor_connectivity: 1,
                    fraction: 0.0,
                    minimum_positive: 1,
                    interpolate: false,
                },
            ),
            (
                1.5,
                SmoothThresholds {
                    floor_connectivity: 1,
                    fraction: 0.5,
                    minimum_positive: 2,
                    interpolate: true,
                },
            ),
            (
                1.00001,
                SmoothThresholds {
                    floor_connectivity: 1,
                    fraction: 1.00001 - 1.0,
                    minimum_positive: 2,
                    interpolate: true,
                },
            ),
        ] {
            assert_eq!(smooth_thresholds(local_connectivity, 4), expected);
        }
        assert_eq!(
            smooth_thresholds(f32::MAX, 4),
            SmoothThresholds {
                floor_connectivity: 5,
                fraction: f32::MAX - (usize::MAX as f32),
                minimum_positive: 5,
                interpolate: true,
            }
        );
    }
}

#[cube]
fn smooth_row<F: Float>(
    distances: &Tensor<F>,
    global_mean: &Tensor<F>,
    row: usize,
    k: usize,
    floor_connectivity: u32,
    fraction: f32,
    minimum_positive: u32,
    interpolate: u32,
    target_sum: f32,
) -> (F, F) {
    let mut positive_count = 0usize;
    let mut positive_max = F::new(0.0f32);
    for slot in 0..k {
        let distance = distances[row * k + slot];
        if distance > F::new(0.0f32) {
            positive_count += 1;
            if distance > positive_max {
                positive_max = distance;
            }
        }
    }

    let mut rho = F::new(0.0f32);
    if positive_count > 0 {
        if floor_connectivity == 0u32 {
            let mut first = positive_max;
            for slot in 0..k {
                let distance = distances[row * k + slot];
                if distance > F::new(0.0f32) && distance < first {
                    first = distance;
                }
            }
            rho = first * F::cast_from(fraction);
        } else {
            let lo_rank = (floor_connectivity - 1u32) as usize;
            let hi_rank = floor_connectivity as usize;
            let mut before = positive_max;
            let mut after = positive_max;
            for candidate_slot in 0..k {
                let candidate = distances[row * k + candidate_slot];
                if candidate > F::new(0.0f32) {
                    let mut less = 0usize;
                    let mut equal = 0usize;
                    for other_slot in 0..k {
                        let other = distances[row * k + other_slot];
                        if other > F::new(0.0f32) && other < candidate {
                            less += 1;
                        } else if other > F::new(0.0f32) && other == candidate {
                            equal += 1;
                        }
                    }
                    if less <= lo_rank && lo_rank < less + equal {
                        before = candidate;
                    }
                    if less <= hi_rank && hi_rank < less + equal {
                        after = candidate;
                    }
                }
            }
            if interpolate != 0u32 {
                rho = before + (after - before) * F::cast_from(fraction);
            } else {
                rho = before;
            }
        }
        if positive_count < minimum_positive as usize {
            rho = positive_max;
        }
    }

    let mut row_sum = F::new(0.0f32);
    for slot in 0..k {
        row_sum += distances[row * k + slot];
    }
    let floor_mean = if rho > F::new(0.0f32) {
        row_sum / F::cast_from(k as u32)
    } else {
        F::cast_from(global_mean[0])
    };
    let mut lo = F::new(0.0f32);
    let mut hi = F::new(3.402_823_5e38_f32);
    let mut mid = F::new(1.0f32);
    let target = F::cast_from(target_sum);
    for _ in 0..64 {
        let mut sum = F::new(0.0f32);
        for slot in 1..k {
            let delta = distances[row * k + slot] - rho;
            let safe_mid = if mid == F::new(0.0f32) {
                F::new(1.0f32)
            } else {
                mid
            };
            let clamped_delta = if delta < F::new(0.0f32) {
                F::new(0.0f32)
            } else {
                delta
            };
            let mut value = (-clamped_delta / safe_mid).exp();
            if delta <= F::new(0.0f32) {
                value = F::new(1.0f32);
            }
            sum += value;
        }
        let error = F::abs(sum - target);
        if error < F::new(1e-5f32) {
            break;
        }
        if sum > target {
            hi = mid;
            mid = (lo + hi) * F::new(0.5f32);
        } else if hi == F::new(3.402_823_5e38_f32) {
            lo = mid;
            mid *= F::new(2.0f32);
        } else {
            lo = mid;
            mid = (lo + hi) * F::new(0.5f32);
        }
    }
    let sigma_floor = floor_mean * F::new(1e-3f32);
    if mid < sigma_floor {
        mid = sigma_floor;
    }
    (rho, mid)
}

#[cube(launch)]
fn fuzzy_graph_global_sum_partial_kernel<F: Float>(
    distances: &Tensor<F>,
    partials: &mut Tensor<F>,
    element_count: u32,
    #[define(F)] _dtype: ElemType,
) {
    let chunk = ABSOLUTE_POS_X as usize;
    if chunk < partials.shape(0) {
        let start = chunk * GLOBAL_SUM_CHUNK_SIZE;
        let mut end = start + GLOBAL_SUM_CHUNK_SIZE;
        if end > element_count as usize {
            end = element_count as usize;
        }
        let mut sum = F::new(0.0f32);
        for index in start..end {
            sum += distances[index];
        }
        partials[chunk] = sum;
    }
}

#[cube(launch)]
fn fuzzy_graph_global_mean_finalize_kernel<F: Float>(
    partials: &Tensor<F>,
    mean: &mut Tensor<F>,
    element_count: u32,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0u32 {
        let mut sum = F::new(0.0f32);
        for index in 0..partials.shape(0) {
            sum += partials[index];
        }
        mean[0] = sum / F::cast_from(element_count);
    }
}

/// Evaluate the edge contribution of `indices[row, slot]` against the already
/// computed directed memberships. Returns the two directed entries
/// `(head, tail)` of the emitted edge plus its weight, or a direction count of
/// zero when this slot emits nothing. Duplicate slots are coalesced (only the
/// first occurrence emits, summing every duplicate slot's membership), and the
/// reverse membership is read directly from row `tail`'s K slots. Bipartite
/// edges keep the coalesced forward membership unchanged; the set-op blend
/// applies only to the fit graph, whose reciprocal membership is non-zero.
#[cube]
fn evaluate_edge<F: Float, I: Int>(
    indices: &Tensor<I>,
    directed: &Tensor<F>,
    row: usize,
    slot: usize,
    k: usize,
    mix: f32,
    bipartite: u32,
    n_reference: u32,
) -> (I, I, I, I, F, u32) {
    let mut head_first = I::cast_from(0i32);
    let mut tail_first = I::cast_from(0i32);
    let mut head_second = I::cast_from(0i32);
    let mut tail_second = I::cast_from(0i32);
    let mut fuzzy = F::new(0.0f32);
    let mut directions = 0u32;
    let tail = indices[row * k + slot];
    if tail >= I::cast_from(0u32)
        && tail < I::cast_from(n_reference)
        && (bipartite != 0 || tail != I::cast_from(row as u32))
    {
        let mut first = true;
        let mut forward = F::new(0.0f32);
        for other_slot in 0..k {
            if indices[row * k + other_slot] == tail {
                if other_slot < slot {
                    first = false;
                }
                forward += directed[row * k + other_slot];
            }
        }
        if first {
            let mut reverse = F::new(0.0f32);
            let mut has_reverse = false;
            if bipartite == 0 {
                let tail_row = u32::cast_from(tail) as usize;
                for reverse_slot in 0..k {
                    if indices[tail_row * k + reverse_slot] == I::cast_from(row as u32) {
                        reverse += directed[tail_row * k + reverse_slot];
                        has_reverse = true;
                    }
                }
            }
            let row_index = I::cast_from(row as u32);
            if !has_reverse || row_index <= tail {
                if bipartite != 0 {
                    // The transform graph keeps directed forward memberships
                    // unchanged: set_op_mix_ratio only blends the reciprocal
                    // fit memberships, and the bipartite reverse membership is
                    // fixed at zero.
                    fuzzy = forward;
                } else {
                    let intersection = forward * reverse;
                    fuzzy = F::cast_from(mix) * (forward + reverse - intersection)
                        + F::cast_from(1.0f32 - mix) * intersection;
                }
                if fuzzy > F::new(0.0f32) {
                    let low = if bipartite != 0 || row_index < tail {
                        row_index
                    } else {
                        tail
                    };
                    let high = if bipartite != 0 || row_index < tail {
                        tail
                    } else {
                        row_index
                    };
                    head_first = low;
                    tail_first = high;
                    head_second = high;
                    tail_second = low;
                    directions = if bipartite != 0 { 1u32 } else { 2u32 };
                }
            }
        }
    }
    (
        head_first,
        tail_first,
        head_second,
        tail_second,
        fuzzy,
        directions,
    )
}

#[cube(launch)]
fn fuzzy_graph_zero_kernel<F: Float, I: Int>(
    heads: &mut Tensor<I>,
    tails: &mut Tensor<I>,
    weights: &mut Tensor<F>,
    counts: &mut Tensor<I>,
    invalid_flag: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let position = ABSOLUTE_POS_X as usize;
    let edge_capacity = heads.shape(0);
    let node_count = counts.shape(0);
    if position < edge_capacity {
        heads[position] = I::cast_from(0i32);
        tails[position] = I::cast_from(0i32);
        weights[position] = F::new(0.0f32);
    }
    if position < node_count {
        counts[position] = I::cast_from(0i32);
    }
    if position == 0 {
        invalid_flag[0] = I::cast_from(0i32);
    }
}

#[cube(launch)]
fn fuzzy_graph_row_kernel<F: Float, I: Int>(
    indices: &Tensor<I>,
    distances: &Tensor<F>,
    global_mean: &Tensor<F>,
    rho_out: &mut Tensor<F>,
    sigma_out: &mut Tensor<F>,
    directed_out: &mut Tensor<F>,
    invalid_flag: &mut Tensor<Atomic<I>>,
    floor_connectivity: u32,
    fraction: f32,
    minimum_positive: u32,
    interpolate: u32,
    target_sum: f32,
    bipartite: u32,
    n_reference: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = indices.shape(0);
    let k = indices.shape(1);
    if row < n {
        let (rho, sigma) = smooth_row(
            distances,
            global_mean,
            row,
            k,
            floor_connectivity,
            fraction,
            minimum_positive,
            interpolate,
            target_sum,
        );
        rho_out[row] = rho;
        sigma_out[row] = sigma;
        let mut invalid = false;
        for slot in 0..k {
            let index = indices[row * k + slot];
            let distance = distances[row * k + slot];
            let delta = distance - rho;
            let safe_sigma = if sigma == F::new(0.0f32) {
                F::new(1.0f32)
            } else {
                sigma
            };
            let clamped_delta = if delta < F::new(0.0f32) {
                F::new(0.0f32)
            } else {
                delta
            };
            let mut value = (-clamped_delta / safe_sigma).exp();
            if distance <= rho || sigma == F::new(0.0f32) {
                value = F::new(1.0f32);
            }
            if index < I::cast_from(0u32)
                || index >= I::cast_from(n_reference)
                || (bipartite == 0 && index == I::cast_from(row as u32))
            {
                value = F::new(0.0f32);
            }
            directed_out[row * k + slot] = value;
            if index < I::cast_from(-1i32) || index >= I::cast_from(n_reference) {
                invalid = true;
            }
        }
        if invalid {
            invalid_flag[0].store(I::cast_from(1i32));
        }
    }
}

#[cube(launch)]
fn fuzzy_graph_count_kernel<F: Float, I: Int>(
    indices: &Tensor<I>,
    directed: &Tensor<F>,
    counts: &mut Tensor<Atomic<I>>,
    mix: f32,
    bipartite: u32,
    n_reference: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = indices.shape(0);
    let k = indices.shape(1);
    if row < n {
        for slot in 0..k {
            let (head_first, _tail_first, head_second, _tail_second, _fuzzy, directions) =
                evaluate_edge(indices, directed, row, slot, k, mix, bipartite, n_reference);
            if directions > 0u32 {
                counts[u32::cast_from(head_first) as usize].fetch_add(I::cast_from(1i32));
                if directions > 1u32 {
                    counts[u32::cast_from(head_second) as usize].fetch_add(I::cast_from(1i32));
                }
            }
        }
    }
}

#[cube(launch)]
fn fuzzy_graph_prefix_kernel<I: Int>(
    counts: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let node_count = counts.shape(0);
        let mut running = I::cast_from(0i32);
        for node in 0..node_count {
            offsets[node] = running;
            cursors[node].store(running);
            running += counts[node];
        }
    }
}

#[cube(launch)]
fn fuzzy_graph_scatter_kernel<F: Float, I: Int>(
    indices: &Tensor<I>,
    directed: &Tensor<F>,
    cursors: &mut Tensor<Atomic<I>>,
    heads: &mut Tensor<I>,
    tails: &mut Tensor<I>,
    weights: &mut Tensor<F>,
    mix: f32,
    bipartite: u32,
    n_reference: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = indices.shape(0);
    let k = indices.shape(1);
    if row < n {
        for slot in 0..k {
            let (head_first, tail_first, head_second, tail_second, fuzzy, directions) =
                evaluate_edge(indices, directed, row, slot, k, mix, bipartite, n_reference);
            if directions > 0u32 {
                let position_first = u32::cast_from(
                    cursors[u32::cast_from(head_first) as usize].fetch_add(I::cast_from(1i32)),
                ) as usize;
                heads[position_first] = head_first;
                tails[position_first] = tail_first;
                weights[position_first] = fuzzy;
                if directions > 1u32 {
                    let position_second = u32::cast_from(
                        cursors[u32::cast_from(head_second) as usize].fetch_add(I::cast_from(1i32)),
                    ) as usize;
                    heads[position_second] = head_second;
                    tails[position_second] = tail_second;
                    weights[position_second] = fuzzy;
                }
            }
        }
    }
}

#[cube(launch)]
fn fuzzy_graph_sort_kernel<F: Float, I: Int>(
    counts: &Tensor<I>,
    offsets: &Tensor<I>,
    heads: &mut Tensor<I>,
    tails: &mut Tensor<I>,
    weights: &mut Tensor<F>,
    invalid_flag: &Tensor<Atomic<I>>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let head = ABSOLUTE_POS_X as usize;
    let node_count = counts.shape(0);
    if head < node_count {
        let start = u32::cast_from(offsets[head]) as usize;
        let end = start + u32::cast_from(counts[head]) as usize;
        // Tails are unique per head after coalescing, so insertion sort by tail
        // yields the same deterministic order regardless of the atomic scatter.
        let mut position = start + 1;
        while position < end {
            let head_value = heads[position];
            let tail_value = tails[position];
            let weight_value = weights[position];
            let mut insert = position;
            while insert > start {
                let previous = insert - 1;
                if tails[previous] < tail_value {
                    break;
                }
                heads[insert] = heads[previous];
                tails[insert] = tails[previous];
                weights[insert] = weights[previous];
                insert -= 1;
            }
            heads[insert] = head_value;
            tails[insert] = tail_value;
            weights[insert] = weight_value;
            position += 1;
        }
        if head == 0 && invalid_flag[0].load() != I::cast_from(0i32) {
            heads[0] = I::cast_from(-2i32);
            tails[0] = I::cast_from(-2i32);
            weights[0] = F::new(-1.0f32);
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn fuzzy_graph(
    indices: IntTensor<CubeBackend>,
    distances: FloatTensor<CubeBackend>,
    local_connectivity: f32,
    bandwidth: f32,
    mix: f32,
    bipartite: bool,
    n_reference: u32,
) -> (
    IntTensor<CubeBackend>,
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
) {
    let indices = into_contiguous(indices);
    let distances = into_contiguous(distances);
    let [n, k] = indices.shape().dims::<2>();
    assert!(n > 0 && k > 0, "neighbor tensors must be non-empty");
    assert_eq!(
        distances.shape().dims::<2>(),
        [n, k],
        "neighbor distance shape mismatch"
    );
    assert_eq!(
        distances.dtype,
        DType::F32,
        "fuzzy graph requires f32 distances"
    );
    assert_eq!(
        indices.dtype,
        DType::I32,
        "fuzzy graph requires i32 neighbor indices"
    );
    assert!(local_connectivity.is_finite() && local_connectivity >= 0.0);
    assert!(bandwidth.is_finite() && bandwidth > 0.0);
    assert!(mix.is_finite() && (0.0..=1.0).contains(&mix));
    let n_reference = usize::try_from(n_reference).expect("reference count exceeds usize");
    assert!(n_reference > 0 && n_reference <= i32::MAX as usize);
    if !bipartite {
        assert_eq!(
            n, n_reference,
            "fit graph reference count must equal query count"
        );
    }
    let node_count = if bipartite {
        n.checked_add(n_reference)
            .expect("bipartite node count overflows usize")
    } else {
        n
    };
    assert!(
        node_count <= i32::MAX as usize,
        "graph node ids exceed i32 range"
    );
    let capacity = n
        .checked_mul(k)
        .and_then(|value| value.checked_mul(2))
        .expect("fuzzy graph capacity overflows usize");
    assert!(
        capacity <= i32::MAX as usize,
        "fuzzy graph COO indices exceed i32 range"
    );
    let n_u32 = u32::try_from(n).expect("query row count exceeds CubeCL range");
    let element_count = n
        .checked_mul(k)
        .expect("fuzzy graph distance count overflows usize");
    let element_count_u32 =
        u32::try_from(element_count).expect("fuzzy graph distance count exceeds CubeCL range");
    let partial_count = element_count.div_ceil(GLOBAL_SUM_CHUNK_SIZE).max(1);
    let partial_count_u32 =
        u32::try_from(partial_count).expect("global reduction partial count exceeds CubeCL range");
    let node_u32 = u32::try_from(node_count).expect("graph node count exceeds CubeCL range");
    let index_bytes = capacity
        .checked_mul(DType::I32.size())
        .expect("graph index allocation overflows usize");
    let weight_bytes = capacity
        .checked_mul(distances.dtype.size())
        .expect("graph weight allocation overflows usize");
    let row_bytes = n
        .checked_mul(distances.dtype.size())
        .expect("graph row-stat allocation overflows usize");
    let directed_bytes = n
        .checked_mul(k)
        .and_then(|value| value.checked_mul(distances.dtype.size()))
        .expect("directed membership allocation overflows usize");
    let partial_bytes = partial_count
        .checked_mul(DType::F32.size())
        .expect("global reduction partial allocation overflows usize");
    let mean_bytes = DType::F32.size();
    let node_index_bytes = node_count
        .checked_mul(DType::I32.size())
        .expect("graph node scratch allocation overflows usize");
    let client = indices.client.clone();
    let cube_dim = crate::kernels::legal_cube_dim(&client, CubeDim::new_1d(32));
    let device = indices.device.clone();
    let dtypes = [
        dtype_to_storage_type(distances.dtype),
        dtype_to_storage_type(indices.dtype),
    ];
    let index_dtype = dtype_to_storage_type(indices.dtype);
    let heads = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([capacity]),
        client.empty(index_bytes),
        DType::I32,
    );
    let tails = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([capacity]),
        client.empty(index_bytes),
        DType::I32,
    );
    let weights = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([capacity]),
        client.empty(weight_bytes),
        distances.dtype,
    );
    let rho = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(row_bytes),
        distances.dtype,
    );
    let sigma = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(row_bytes),
        distances.dtype,
    );
    let directed = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n, k]),
        client.empty(directed_bytes),
        distances.dtype,
    );
    let partials = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([partial_count]),
        client.empty(partial_bytes),
        DType::F32,
    );
    let global_mean = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([1]),
        client.empty(mean_bytes),
        DType::F32,
    );
    let counts = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([node_count]),
        client.empty(node_index_bytes),
        DType::I32,
    );
    let offsets = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([node_count]),
        client.empty(node_index_bytes),
        DType::I32,
    );
    let cursors = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([node_count]),
        client.empty(node_index_bytes),
        DType::I32,
    );
    let invalid_flag = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([1]),
        client.empty(DType::I32.size()),
        DType::I32,
    );
    let zero_units =
        u32::try_from(capacity.max(node_count)).expect("graph zero-init exceeds CubeCL range");
    fuzzy_graph_zero_kernel::launch(
        &client,
        CubeCount::Static(zero_units.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        heads.clone().into_tensor_arg(),
        tails.clone().into_tensor_arg(),
        weights.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        invalid_flag.clone().into_tensor_arg(),
        dtypes,
    );
    let reduction_dim = crate::kernels::legal_cube_dim(&client, CubeDim::new_1d(32));
    fuzzy_graph_global_sum_partial_kernel::launch(
        &client,
        CubeCount::Static(partial_count_u32.div_ceil(reduction_dim.x), 1, 1),
        reduction_dim,
        distances.clone().into_tensor_arg(),
        partials.clone().into_tensor_arg(),
        element_count_u32,
        dtype_to_storage_type(distances.dtype),
    );
    fuzzy_graph_global_mean_finalize_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        partials.into_tensor_arg(),
        global_mean.clone().into_tensor_arg(),
        element_count_u32,
        dtype_to_storage_type(distances.dtype),
    );
    let thresholds = smooth_thresholds(local_connectivity, k);
    fuzzy_graph_row_kernel::launch(
        &client,
        CubeCount::Static(n_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        indices.clone().into_tensor_arg(),
        distances.clone().into_tensor_arg(),
        global_mean.into_tensor_arg(),
        rho.clone().into_tensor_arg(),
        sigma.clone().into_tensor_arg(),
        directed.clone().into_tensor_arg(),
        invalid_flag.clone().into_tensor_arg(),
        thresholds.floor_connectivity,
        thresholds.fraction,
        thresholds.minimum_positive,
        thresholds.interpolate as u32,
        (k as f32).log2() * bandwidth,
        bipartite as u32,
        u32::try_from(n_reference).expect("reference count exceeds CubeCL range"),
        dtypes,
    );
    fuzzy_graph_count_kernel::launch(
        &client,
        CubeCount::Static(n_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        indices.clone().into_tensor_arg(),
        directed.clone().into_tensor_arg(),
        counts.clone().into_tensor_arg(),
        mix,
        bipartite as u32,
        u32::try_from(n_reference).expect("reference count exceeds CubeCL range"),
        dtypes,
    );
    fuzzy_graph_prefix_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        counts.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        cursors.clone().into_tensor_arg(),
        index_dtype,
    );
    fuzzy_graph_scatter_kernel::launch(
        &client,
        CubeCount::Static(n_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        indices.into_tensor_arg(),
        directed.clone().into_tensor_arg(),
        cursors.clone().into_tensor_arg(),
        heads.clone().into_tensor_arg(),
        tails.clone().into_tensor_arg(),
        weights.clone().into_tensor_arg(),
        mix,
        bipartite as u32,
        u32::try_from(n_reference).expect("reference count exceeds CubeCL range"),
        dtypes,
    );
    fuzzy_graph_sort_kernel::launch(
        &client,
        CubeCount::Static(node_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        counts.into_tensor_arg(),
        offsets.into_tensor_arg(),
        heads.clone().into_tensor_arg(),
        tails.clone().into_tensor_arg(),
        weights.clone().into_tensor_arg(),
        invalid_flag.into_tensor_arg(),
        dtypes,
    );
    (heads, tails, weights, rho, sigma, directed)
}

#[cube(launch)]
fn head_csr_zero_kernel<I: Int>(
    counts: &mut Tensor<I>,
    ids: &mut Tensor<I>,
    invalid: &mut Tensor<I>,
    edge_capacity: u32,
    #[define(I)] _dtype: ElemType,
) {
    let position = ABSOLUTE_POS_X as usize;
    if position < counts.shape(0) {
        counts[position] = I::cast_from(0i32);
    }
    if position < ids.shape(0) {
        ids[position] = I::cast_from(1i32);
    }
    if position < 2 {
        invalid[position] = I::cast_from(edge_capacity);
    }
}

#[cube(launch)]
fn head_csr_count_kernel<F: Float, I: Int>(
    heads: &Tensor<I>,
    weights: &Tensor<F>,
    counts: &mut Tensor<Atomic<I>>,
    invalid: &mut Tensor<Atomic<I>>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let edge = ABSOLUTE_POS_X as usize;
    if edge < heads.shape(0) {
        let head = heads[edge];
        let weight = weights[edge];
        let head_valid = head >= I::cast_from(0i32) && head < I::cast_from(counts.shape(0) as u32);
        let weight_valid = weight >= F::new(0.0f32) && weight <= F::new(f32::MAX);
        if !head_valid {
            invalid[0].fetch_min(I::cast_from(edge as u32));
        }
        if !weight_valid {
            invalid[1].fetch_min(I::cast_from(edge as u32));
        }
        if head_valid && weight_valid && weight > F::new(0.0f32) {
            let row = u32::cast_from(head) as usize;
            counts[row].fetch_add(I::cast_from(1i32));
        }
    }
}

#[cube(launch)]
fn head_csr_prefix_kernel<I: Int>(
    counts: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let head_rows = counts.shape(0);
        let mut running = I::cast_from(0i32);
        for head in 0..head_rows {
            offsets[head] = running;
            cursors[head].store(running);
            running += counts[head];
        }
        offsets[head_rows] = running;
    }
}

#[cube(launch)]
fn head_csr_scatter_kernel<F: Float, I: Int>(
    heads: &Tensor<I>,
    weights: &Tensor<F>,
    cursors: &mut Tensor<Atomic<I>>,
    ids: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let edge = ABSOLUTE_POS_X as usize;
    if edge < heads.shape(0) {
        let head = heads[edge];
        let weight = weights[edge];
        let head_valid = head >= I::cast_from(0i32) && head < I::cast_from(cursors.shape(0) as u32);
        let weight_valid = weight >= F::new(0.0f32) && weight <= F::new(f32::MAX);
        if head_valid && weight_valid && weight > F::new(0.0f32) {
            let row = u32::cast_from(head) as usize;
            let position = u32::cast_from(cursors[row].fetch_add(I::cast_from(1i32))) as usize;
            // Count and scatter use the same validity/filter predicates, so
            // the cursor cannot advance beyond this fixed-capacity output.
            ids[position] = I::cast_from(edge as u32);
        }
    }
}

#[cube(launch)]
fn head_csr_sort_kernel<I: Int>(
    offsets: &Tensor<I>,
    ids: &mut Tensor<I>,
    #[define(I)] _dtype: ElemType,
) {
    let head = ABSOLUTE_POS_X as usize;
    let head_rows = offsets.shape(0) - 1;
    if head < head_rows {
        let start = u32::cast_from(offsets[head]) as usize;
        let end = u32::cast_from(offsets[head + 1]) as usize;
        let mut position = start + 1;
        while position < end {
            let edge_id = ids[position];
            let mut insert = position;
            while insert > start {
                let previous = insert - 1;
                if ids[previous] < edge_id {
                    break;
                }
                ids[insert] = ids[previous];
                insert -= 1;
            }
            ids[insert] = edge_id;
            position += 1;
        }
    }
}

#[cube(launch)]
fn head_csr_marker_kernel<I: Int>(
    invalid: &Tensor<Atomic<I>>,
    offsets: &mut Tensor<I>,
    edge_capacity: u32,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let first_bad_head = invalid[0].load();
        let first_bad_weight = invalid[1].load();
        let capacity = I::cast_from(edge_capacity);
        if first_bad_head < capacity || first_bad_weight < capacity {
            offsets[0] = if first_bad_head <= first_bad_weight {
                I::cast_from(-2i32)
            } else {
                I::cast_from(-3i32)
            };
        }
    }
}

pub(crate) fn head_csr(
    heads: IntTensor<CubeBackend>,
    weights: FloatTensor<CubeBackend>,
    head_rows: u32,
) -> (IntTensor<CubeBackend>, IntTensor<CubeBackend>) {
    let heads = into_contiguous(heads);
    let weights = into_contiguous(weights);
    let [edge_count] = heads.shape().dims::<1>();
    assert_eq!(
        weights.shape().dims::<1>(),
        [edge_count],
        "layout edge arrays have different lengths"
    );
    assert_eq!(heads.dtype, DType::I32, "layout edge heads must be i32");
    assert_eq!(weights.dtype, DType::F32, "layout edge weights must be f32");
    assert_eq!(
        heads.device, weights.device,
        "layout edge arrays must share a device"
    );
    let head_rows = usize::try_from(head_rows).expect("layout head rows exceed usize");
    assert!(head_rows > 0, "layout head rows must be positive");
    i32::try_from(head_rows).expect("layout head rows exceed i32");
    let head_rows_u32 = u32::try_from(head_rows).expect("layout head rows exceed u32");
    let edge_capacity = u32::try_from(edge_count).expect("layout edge count exceeds u32");
    i32::try_from(edge_count).expect("layout edge ids exceed i32");
    let offsets_len = head_rows
        .checked_add(1)
        .expect("layout head offsets overflow usize");
    let _offsets_len_u32 = u32::try_from(offsets_len).expect("layout offsets exceed CubeCL range");
    let ids_len = edge_count.max(1);
    let ids_len_u32 = u32::try_from(ids_len).expect("layout ids exceed CubeCL range");

    let index_size = DType::I32.size();
    let counts_bytes = head_rows
        .checked_mul(index_size)
        .expect("head CSR count allocation overflows usize");
    let cursors_bytes = head_rows
        .checked_mul(index_size)
        .expect("head CSR cursor allocation overflows usize");
    let invalid_bytes = 2usize
        .checked_mul(index_size)
        .expect("head CSR validation allocation overflows usize");
    let offsets_bytes = offsets_len
        .checked_mul(index_size)
        .expect("head CSR offsets allocation overflows usize");
    let ids_bytes = ids_len
        .checked_mul(index_size)
        .expect("head CSR ids allocation overflows usize");
    let scratch_bytes = counts_bytes
        .checked_add(cursors_bytes)
        .and_then(|bytes| bytes.checked_add(invalid_bytes))
        .expect("head CSR scratch allocation overflows usize");
    let output_bytes = offsets_bytes
        .checked_add(ids_bytes)
        .expect("head CSR output allocation overflows usize");
    let allocation_bytes = scratch_bytes
        .checked_add(output_bytes)
        .expect("head CSR total allocation overflows usize");
    assert!(
        allocation_bytes <= 256 * 1024 * 1024,
        "head CSR allocations exceed 256 MiB"
    );

    let client = heads.client.clone();
    let cube_dim = crate::kernels::legal_cube_dim(&client, CubeDim::new_1d(32));
    let device = heads.device.clone();
    let index_dtype = dtype_to_storage_type(heads.dtype);
    let weight_dtype = dtype_to_storage_type(weights.dtype);
    let dtypes = [weight_dtype, index_dtype];
    let counts = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([head_rows]),
        client.empty(counts_bytes),
        DType::I32,
    );
    let cursors = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([head_rows]),
        client.empty(cursors_bytes),
        DType::I32,
    );
    let offsets = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([offsets_len]),
        client.empty(offsets_bytes),
        DType::I32,
    );
    let ids = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([ids_len]),
        client.empty(ids_bytes),
        DType::I32,
    );
    let invalid = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([2]),
        client.empty(invalid_bytes),
        DType::I32,
    );

    let zero_units = head_rows_u32.max(ids_len_u32).max(2);
    head_csr_zero_kernel::launch(
        &client,
        CubeCount::Static(zero_units.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        counts.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),
        invalid.clone().into_tensor_arg(),
        edge_capacity,
        index_dtype,
    );
    if edge_count > 0 {
        head_csr_count_kernel::launch(
            &client,
            CubeCount::Static(edge_capacity.div_ceil(cube_dim.x), 1, 1),
            cube_dim,
            heads.clone().into_tensor_arg(),
            weights.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            invalid.clone().into_tensor_arg(),
            dtypes,
        );
    }
    head_csr_prefix_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        counts.clone().into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        cursors.clone().into_tensor_arg(),
        index_dtype,
    );
    if edge_count > 0 {
        head_csr_scatter_kernel::launch(
            &client,
            CubeCount::Static(edge_capacity.div_ceil(cube_dim.x), 1, 1),
            cube_dim,
            heads.into_tensor_arg(),
            weights.into_tensor_arg(),
            cursors.into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            dtypes,
        );
    }
    head_csr_sort_kernel::launch(
        &client,
        CubeCount::Static(head_rows_u32.div_ceil(cube_dim.x), 1, 1),
        cube_dim,
        offsets.clone().into_tensor_arg(),
        ids.clone().into_tensor_arg(),
        index_dtype,
    );
    head_csr_marker_kernel::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_1d(1),
        invalid.into_tensor_arg(),
        offsets.clone().into_tensor_arg(),
        edge_capacity,
        index_dtype,
    );
    (offsets, ids)
}
