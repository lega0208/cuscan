use super::legal_cube_dim;
use crate::neighbors::{MAX_REFERENCE_GROUPS, REFERENCE_TILE, bounded_neighbor_scratch};
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

const DISTANCE_TILE: CubeDim = CubeDim::new_2d(16, 16);
const NEIGHBOR_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

#[cube(launch)]
fn exact_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    k: u32,
    exclude_self: u32,
    query_offset: u32,
    squared: u32,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let nr = reference.shape(0);
    let d = query.shape(1);
    let kw = k as usize;
    if row < query.shape(0) {
        for slot in 0..kw {
            indices[row * kw + slot] = I::cast_from(nr);
            distances[row * kw + slot] = if squared != 0 {
                F::new(f32::INFINITY)
            } else {
                F::new(f32::MAX)
            };
        }
        for r in 0..nr {
            if exclude_self == 0 || r != query_offset as usize + row {
                let mut distance = F::new(0.0f32);
                for f in 0..d {
                    let delta = query[row * d + f] - reference[r * d + f];
                    distance += delta * delta;
                }
                let worst = row * kw + kw - 1;
                let wi = u32::cast_from(indices[worst]) as usize;
                if distance < distances[worst] || (distance == distances[worst] && r < wi) {
                    let mut slot = kw - 1;
                    while slot > 0 {
                        let prev = row * kw + slot - 1;
                        let pi = u32::cast_from(indices[prev]) as usize;
                        if distance < distances[prev] || (distance == distances[prev] && r < pi) {
                            indices[row * kw + slot] = indices[prev];
                            distances[row * kw + slot] = distances[prev];
                            slot -= 1;
                        } else {
                            break;
                        }
                    }
                    indices[row * kw + slot] = I::cast_from(r);
                    distances[row * kw + slot] = distance;
                }
            }
        }
        if squared == 0 {
            for slot in 0..kw {
                let at = row * kw + slot;
                distances[at] = F::sqrt(distances[at]);
            }
        }
    }
}

#[cube(launch)]
fn partial_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    k: u32,
    exclude_self: u32,
    query_offset: u32,
    query_start: u32,
    qrows: u32,
    group_count: u32,
    group_capacity: u32,
    reference_start: u32,
    reference_end: u32,
    squared: u32,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let logical = ABSOLUTE_POS_X as usize;
    let row = logical / group_count as usize;
    let group = logical % group_count as usize;
    let nr = reference.shape(0);
    let d = query.shape(1);
    let kw = k as usize;
    if row < qrows as usize {
        let out_base = (row * group_capacity as usize + group) * kw;
        for slot in 0..kw {
            indices[out_base + slot] = I::cast_from(nr);
            distances[out_base + slot] = if squared != 0 {
                F::new(f32::INFINITY)
            } else {
                F::new(f32::MAX)
            };
        }
        let q = query_start as usize + row;
        let group_start = reference_start as usize + group * REFERENCE_TILE;
        let tile_len = if group_count == 1 && reference_end as usize == nr {
            nr - group_start
        } else {
            (reference_end as usize)
                .saturating_sub(group_start)
                .min(REFERENCE_TILE)
        };
        for tile_r in 0..tile_len {
            let r = group_start + tile_r;
            if r < nr
                && r < reference_end as usize
                && (exclude_self == 0 || r != query_offset as usize + q)
            {
                let mut distance = F::new(0.0f32);
                for f in 0..d {
                    let delta = query[q * d + f] - reference[r * d + f];
                    distance += delta * delta;
                }
                let worst = out_base + kw - 1;
                let wi = u32::cast_from(indices[worst]) as usize;
                if distance < distances[worst] || (distance == distances[worst] && r < wi) {
                    let mut slot = kw - 1;
                    while slot > 0 {
                        let prev = out_base + slot - 1;
                        let pi = u32::cast_from(indices[prev]) as usize;
                        if distance < distances[prev] || (distance == distances[prev] && r < pi) {
                            indices[out_base + slot] = indices[prev];
                            distances[out_base + slot] = distances[prev];
                            slot -= 1;
                        } else {
                            break;
                        }
                    }
                    indices[out_base + slot] = I::cast_from(r);
                    distances[out_base + slot] = distance;
                }
            }
        }
    }
}

#[cube(launch)]
fn merge_kernel<F: Float, I: Int>(
    previous_indices: &Tensor<I>,
    previous_distances: &Tensor<F>,
    partial_indices: &Tensor<I>,
    partial_distances: &Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    k: u32,
    qrows: u32,
    previous_valid: u32,
    group_count: u32,
    partial_group: u32,
    sentinel: u32,
    squared: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let kw = k as usize;
    if row < qrows as usize {
        let partial_base = (row * group_count as usize + partial_group as usize) * kw;
        let mut left = 0usize;
        let mut right = 0usize;
        for out in 0..kw {
            let mut left_id = sentinel;
            let mut left_d = if squared != 0 {
                F::new(f32::INFINITY)
            } else {
                F::new(f32::MAX)
            };
            if previous_valid != 0 && left < kw {
                left_id = u32::cast_from(previous_indices[row * kw + left]);
                left_d = previous_distances[row * kw + left];
            }
            let mut right_id = sentinel;
            let mut right_d = if squared != 0 {
                F::new(f32::INFINITY)
            } else {
                F::new(f32::MAX)
            };
            if right < kw {
                right_id = u32::cast_from(partial_indices[partial_base + right]);
                right_d = partial_distances[partial_base + right];
            }
            if right < kw
                && (left >= kw
                    || previous_valid == 0
                    || right_d < left_d
                    || (right_d == left_d && right_id < left_id))
            {
                output_indices[row * kw + out] = I::cast_from(right_id);
                output_distances[row * kw + out] = right_d;
                right += 1;
            } else {
                output_indices[row * kw + out] = I::cast_from(left_id);
                output_distances[row * kw + out] = left_d;
                left += 1;
            }
        }
    }
}

#[cube(launch)]
fn copy_rows_kernel<F: Float, I: Int>(
    source_indices: &Tensor<I>,
    source_distances: &Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    k: u32,
    query_start: u32,
    squared: u32,
    sentinel: u32,
    qrows: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let kw = k as usize;
    if row < qrows as usize {
        for slot in 0..kw {
            let at = row * kw + slot;
            let id = u32::cast_from(source_indices[at]);
            let distance = source_distances[at];
            output_indices[(query_start as usize + row) * kw + slot] = I::cast_from(id);
            output_distances[(query_start as usize + row) * kw + slot] =
                if squared == 0 && id < sentinel {
                    F::sqrt(distance)
                } else if squared == 0 && id == sentinel {
                    F::sqrt(F::new(f32::MAX))
                } else {
                    distance
                };
        }
    }
}

#[cube(launch)]
fn distance_kernel<F: Float>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    output: &mut Tensor<F>,
    #[define(F)] _dtype: ElemType,
) {
    let q = ABSOLUTE_POS_X as usize;
    let r = ABSOLUTE_POS_Y as usize;
    let d = query.shape(1);
    if q < query.shape(0) && r < reference.shape(0) {
        let mut sum = F::new(0.0f32);
        for f in 0..d {
            let delta = query[q * d + f] - reference[r * d + f];
            sum += delta * delta;
        }
        output[q * reference.shape(0) + r] = sum;
    }
}

pub(super) fn exact_neighbors(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    k: u32,
    exclude_self: bool,
    query_offset: u32,
    squared: bool,
) -> (IntTensor<CubeBackend>, FloatTensor<CubeBackend>) {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [qr, qf] = query.shape().dims::<2>();
    let [rr, rf] = reference.shape().dims::<2>();
    assert!(qr > 0 && rr > 0, "neighbor inputs must be non-empty");
    assert!(qf > 0, "neighbor feature dimension must be non-empty");
    assert_eq!(qf, rf, "query and reference feature dimensions must match");
    assert_eq!(
        query.dtype, reference.dtype,
        "query and reference dtype must match"
    );
    assert!(k > 0, "neighbor count must be positive");
    let qr32 = u32::try_from(qr).expect("query row count exceeds u32");
    let rr32 = u32::try_from(rr).expect("reference row count exceeds u32");
    i32::try_from(rr).expect("reference indices exceed i32");
    let end = query_offset
        .checked_add(qr32)
        .expect("query offset plus block rows exceeds u32");
    if exclude_self {
        assert!(
            end <= rr32,
            "self-excluded query indices must be valid reference indices"
        );
    }
    if !squared {
        assert!(
            k <= rr32 - exclude_self as u32,
            "neighbor count exceeds available references"
        );
    }
    let n = qr
        .checked_mul(k as usize)
        .expect("neighbor output dimensions overflow usize");
    let ib = n
        .checked_mul(DType::I32.size())
        .expect("neighbor index allocation overflows usize");
    let db = n
        .checked_mul(query.dtype.size())
        .expect("neighbor distance allocation overflows usize");
    let client = query.client.clone();
    let dim = legal_cube_dim(&client, NEIGHBOR_CUBE_DIM);
    let memory = &client.properties().memory;
    let max_page =
        usize::try_from(memory.max_page_size).expect("device maximum allocation exceeds usize");
    let alignment = usize::try_from(memory.alignment).expect("device alignment exceeds usize");
    assert!(
        max_page > 0 && alignment > 0,
        "device reports invalid allocation limits"
    );
    let aligned_bytes = |bytes: usize| {
        bytes
            .checked_add(alignment - 1)
            .expect("neighbor allocation alignment overflow")
            / alignment
            * alignment
    };
    assert!(
        aligned_bytes(ib) <= max_page,
        "neighbor index output exceeds device allocation limit"
    );
    assert!(
        aligned_bytes(db) <= max_page,
        "neighbor distance output exceeds device allocation limit"
    );
    let ids = CubeTensor::new_contiguous(
        client.clone(),
        query.device.clone(),
        Shape::new([qr, k as usize]),
        client.empty(ib),
        DType::I32,
    );
    let ds = CubeTensor::new_contiguous(
        client.clone(),
        query.device.clone(),
        Shape::new([qr, k as usize]),
        client.empty(db),
        query.dtype,
    );
    let cpu_route = client.properties().hardware.num_cpu_cores.is_some();
    if cpu_route || !squared {
        exact_kernel::launch(
            &client,
            CubeCount::Static(qr32.div_ceil(dim.x), 1, 1),
            dim,
            query.into_tensor_arg(),
            reference.into_tensor_arg(),
            k,
            exclude_self as u32,
            query_offset,
            squared as u32,
            ids.clone().into_tensor_arg(),
            ds.clone().into_tensor_arg(),
            [
                dtype_to_storage_type(ds.dtype),
                dtype_to_storage_type(ids.dtype),
            ],
        );
        return (ids, ds);
    }
    let group_count = rr.div_ceil(REFERENCE_TILE).min(MAX_REFERENCE_GROUPS);
    let max_cubes = client.properties().hardware.max_cube_count.0;
    assert!(
        max_cubes > 0,
        "device reports no legal neighbor launch capacity"
    );
    let max_group_rows = (dim.x as usize)
        .checked_mul(max_cubes as usize)
        .expect("neighbor launch capacity overflows usize")
        / group_count;
    assert!(
        max_group_rows > 0,
        "device cannot launch one reference group"
    );
    let scratch =
        bounded_neighbor_scratch(qr, rr, k as usize, query.dtype.size(), max_page, alignment)
            .expect("bounded neighbor scratch sizing invariant violated");
    let qgroup = scratch.query_rows.min(max_group_rows);
    let running_elements = qgroup
        .checked_mul(k as usize)
        .expect("neighbor running scratch elements overflow");
    let partial_elements = running_elements
        .checked_mul(group_count)
        .expect("neighbor partial scratch elements overflow");
    let running_index_bytes = aligned_bytes(
        running_elements
            .checked_mul(DType::I32.size())
            .expect("running index bytes overflow"),
    );
    let running_distance_bytes = aligned_bytes(
        running_elements
            .checked_mul(query.dtype.size())
            .expect("running distance bytes overflow"),
    );
    let partial_index_bytes = aligned_bytes(
        partial_elements
            .checked_mul(DType::I32.size())
            .expect("partial index bytes overflow"),
    );
    let partial_distance_bytes = aligned_bytes(
        partial_elements
            .checked_mul(query.dtype.size())
            .expect("partial distance bytes overflow"),
    );
    assert!(
        running_index_bytes <= max_page
            && running_distance_bytes <= max_page
            && partial_index_bytes <= max_page
            && partial_distance_bytes <= max_page
    );
    let make_running_ids = || {
        CubeTensor::new_contiguous(
            client.clone(),
            query.device.clone(),
            Shape::new([qgroup, k as usize]),
            client.empty(running_index_bytes),
            DType::I32,
        )
    };
    let make_running_ds = || {
        CubeTensor::new_contiguous(
            client.clone(),
            query.device.clone(),
            Shape::new([qgroup, k as usize]),
            client.empty(running_distance_bytes),
            query.dtype,
        )
    };
    let make_partial_ids = || {
        CubeTensor::new_contiguous(
            client.clone(),
            query.device.clone(),
            Shape::new([qgroup, group_count * k as usize]),
            client.empty(partial_index_bytes),
            DType::I32,
        )
    };
    let make_partial_ds = || {
        CubeTensor::new_contiguous(
            client.clone(),
            query.device.clone(),
            Shape::new([qgroup, group_count * k as usize]),
            client.empty(partial_distance_bytes),
            query.dtype,
        )
    };
    let partial_ids = make_partial_ids();
    let partial_ds = make_partial_ds();
    let mut running_ids = make_running_ids();
    let mut running_ds = make_running_ds();
    let mut merged_ids = make_running_ids();
    let mut merged_ds = make_running_ds();
    for qstart in (0..qr).step_by(qgroup) {
        let qrows = (qr - qstart).min(qgroup);
        let qrows32 = u32::try_from(qrows).expect("query group exceeds u32");
        let mut reference_start = 0usize;
        let mut previous_valid = false;
        while reference_start < rr {
            let active_groups = (rr - reference_start)
                .div_ceil(REFERENCE_TILE)
                .min(group_count);
            let active_groups32 =
                u32::try_from(active_groups).expect("reference group count exceeds u32");
            partial_kernel::launch(
                &client,
                CubeCount::Static(
                    qrows32
                        .checked_mul(active_groups32)
                        .expect("partial launch count overflow")
                        .div_ceil(dim.x),
                    1,
                    1,
                ),
                dim,
                query.clone().into_tensor_arg(),
                reference.clone().into_tensor_arg(),
                k,
                exclude_self as u32,
                query_offset,
                u32::try_from(qstart).expect("query offset exceeds u32"),
                qrows32,
                active_groups32,
                u32::try_from(group_count).expect("reference group capacity exceeds u32"),
                u32::try_from(reference_start).expect("reference offset exceeds u32"),
                u32::try_from((reference_start + active_groups * REFERENCE_TILE).min(rr))
                    .expect("reference end exceeds u32"),
                squared as u32,
                partial_ids.clone().into_tensor_arg(),
                partial_ds.clone().into_tensor_arg(),
                [
                    dtype_to_storage_type(query.dtype),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            for group in 0..active_groups {
                merge_kernel::launch(
                    &client,
                    CubeCount::Static(qrows32.div_ceil(dim.x), 1, 1),
                    dim,
                    running_ids.clone().into_tensor_arg(),
                    running_ds.clone().into_tensor_arg(),
                    partial_ids.clone().into_tensor_arg(),
                    partial_ds.clone().into_tensor_arg(),
                    merged_ids.clone().into_tensor_arg(),
                    merged_ds.clone().into_tensor_arg(),
                    k,
                    qrows32,
                    previous_valid as u32,
                    u32::try_from(group_count).unwrap(),
                    u32::try_from(group).unwrap(),
                    rr32,
                    squared as u32,
                    [
                        dtype_to_storage_type(query.dtype),
                        dtype_to_storage_type(DType::I32),
                    ],
                );
                std::mem::swap(&mut running_ids, &mut merged_ids);
                std::mem::swap(&mut running_ds, &mut merged_ds);
                previous_valid = true;
            }
            reference_start += active_groups * REFERENCE_TILE;
        }
        copy_rows_kernel::launch(
            &client,
            CubeCount::Static(qrows32.div_ceil(dim.x), 1, 1),
            dim,
            running_ids.clone().into_tensor_arg(),
            running_ds.clone().into_tensor_arg(),
            ids.clone().into_tensor_arg(),
            ds.clone().into_tensor_arg(),
            k,
            u32::try_from(qstart).expect("query output offset exceeds u32"),
            squared as u32,
            rr32,
            qrows32,
            [
                dtype_to_storage_type(query.dtype),
                dtype_to_storage_type(DType::I32),
            ],
        );
    }
    (ids, ds)
}

pub(super) fn squared_euclidean_distance_block(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
) -> FloatTensor<CubeBackend> {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let [qr, qf] = query.shape().dims::<2>();
    let [rr, rf] = reference.shape().dims::<2>();
    assert!(qr > 0 && rr > 0, "distance blocks must be non-empty");
    assert!(qf > 0, "distance feature dimension must be non-empty");
    assert_eq!(qf, rf, "query and reference feature dimensions must match");
    let n = qr
        .checked_mul(rr)
        .expect("squared-distance output dimensions overflow usize");
    let bytes = n
        .checked_mul(query.dtype.size())
        .expect("squared-distance output allocation overflows usize");
    let qr32 = u32::try_from(qr).expect("query block row count exceeds u32");
    let rr32 = u32::try_from(rr).expect("reference row count exceeds u32");
    let out = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([qr, rr]),
        query.client.empty(bytes),
        query.dtype,
    );
    let client = query.client.clone();
    let dim = legal_cube_dim(&client, DISTANCE_TILE);
    distance_kernel::launch(
        &client,
        CubeCount::Static(qr32.div_ceil(dim.x), rr32.div_ceil(dim.y), 1),
        dim,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        dtype_to_storage_type(out.dtype),
    );
    out
}

#[cfg(test)]
mod sizing_tests {
    use super::{MAX_REFERENCE_GROUPS, REFERENCE_TILE, bounded_neighbor_scratch};

    #[test]
    fn scratch_sizing_respects_runtime_page_alignment_and_total_budget() {
        assert_eq!(
            bounded_neighbor_scratch(100, 512, 10, 4, 1 << 20, 256)
                .unwrap()
                .query_rows,
            100
        );
        let plan = bounded_neighbor_scratch(10_000, 512, 1_000, 4, 64_000, 256).unwrap();
        let rows = plan.query_rows;
        assert!(rows < 10_000);
        let partial_ids = (rows * 1_000 * 4 * 4).div_ceil(256) * 256;
        let partial_distances = partial_ids;
        let running_ids = (rows * 1_000 * 4).div_ceil(256) * 256;
        let running_distances = running_ids;
        assert!(partial_ids <= 64_000 && partial_distances <= 64_000);
        assert!(running_ids <= 64_000 && running_distances <= 64_000);
        assert!(
            partial_ids + partial_distances + 2 * (running_ids + running_distances)
                <= 32 * 1024 * 1024
        );
        let query_group =
            bounded_neighbor_scratch(701, 763, 1_300, 4, 64 * 1024 * 1024, 256).unwrap();
        assert!(query_group.query_rows < 701);
        assert!(query_group.live_bytes <= 32 * 1024 * 1024);
        assert_eq!(1163_usize.div_ceil(REFERENCE_TILE), 10);
        assert_eq!(10_usize.div_ceil(MAX_REFERENCE_GROUPS), 3);
    }

    #[test]
    fn scratch_sizing_returns_none_for_impossible_single_row() {
        assert!(bounded_neighbor_scratch(10, 512, 100, 4, 128, 64).is_none());
    }
}
