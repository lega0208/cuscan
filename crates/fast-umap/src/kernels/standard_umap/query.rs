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
use fast_umap_core::query::{
    approximate_query_scratch_layout, query_chunk_rows, query_output_allocation,
};

const QUERY_CUBE_DIM: CubeDim = CubeDim::new_1d(1);

fn query_cpu_chunk_geometry(
    rows: u32,
    cores: u32,
    max_cube_dim_x: u32,
    max_units_per_cube: u32,
    max_cube_count: u32,
) -> (CubeCount, CubeDim) {
    assert!(rows > 0, "query chunk row count must be nonzero");
    let dim_x = rows
        .min(cores)
        .min(max_cube_dim_x)
        .min(max_units_per_cube)
        .max(1);
    let dim = CubeDim::new_1d(dim_x);
    let count = rows.div_ceil(dim_x);
    assert!(
        count <= max_cube_count,
        "query chunk requires x cube count {count}; maximum is {max_cube_count}"
    );
    (CubeCount::Static(count, 1, 1), dim)
}

#[cube(launch)]
fn query_flag_zero_kernel<I: Int>(flag: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    if ABSOLUTE_POS_X == 0 {
        flag[0] = I::cast_from(0u32);
    }
}

#[cube(launch)]
fn query_validate_graph_kernel<I: Int>(
    graph: &Tensor<I>,
    reference_rows: u32,
    row_start: u32,
    invalid_flag: &mut Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    let row = row_start as usize + ABSOLUTE_POS_X as usize;
    let reference_count = graph.shape(0);
    let graph_width = graph.shape(1);
    if row < reference_count {
        for slot in 0..graph_width {
            let edge = i32::cast_from(graph[row * graph_width + slot]);
            if edge < -1 || edge >= reference_rows as i32 {
                invalid_flag[0].fetch_max(I::cast_from(1u32));
            }
        }
    }
}

#[cube(launch)]
fn approximate_query_kernel<F: Float, I: Int>(
    query: &Tensor<F>,
    reference: &Tensor<F>,
    graph: &Tensor<I>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u32,
    query_offset: u32,
    row_start: u32,
    chunk_rows: u32,
    invalid_flag: ComptimeOption<&Tensor<I>>,
    #[comptime] checked: bool,
    candidate_ids: &mut Tensor<I>,
    candidate_distances: &mut Tensor<F>,
    candidate_expanded: &mut Tensor<I>,
    seen_ids: &mut Tensor<I>,
    seen_distances: &mut Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    stats: &mut Tensor<I>,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let local_row = ABSOLUTE_POS_X as usize;
    let row = row_start as usize + local_row;
    let query_rows = query.shape(0);
    let reference_rows = reference.shape(0);
    let features = query.shape(1);
    let graph_width = graph.shape(1);
    let c = candidate_capacity as usize;
    let v = visit_capacity as usize;
    let k_usize = k as usize;
    if local_row < chunk_rows as usize && row < query_rows {
        let mut invalid_graph = false;
        if checked {
            invalid_graph = i32::cast_from(invalid_flag.unwrap()[0]) != 0;
        }
        let candidate_base = local_row * c;
        let seen_base = local_row * v;
        for slot in 0..c {
            candidate_ids[candidate_base + slot] = I::cast_from(-1i32);
            candidate_distances[candidate_base + slot] = F::new(3.402_823_5e38_f32);
            candidate_expanded[candidate_base + slot] = I::cast_from(0u32);
        }
        for slot in 0..v {
            seen_ids[seen_base + slot] = I::cast_from(-1i32);
            seen_distances[seen_base + slot] = F::new(3.402_823_5e38_f32);
        }
        let n_u32 = reference_rows as u32;
        let query_key = (query_offset + row as u32) % n_u32;
        let rng_seed = (seed + query_key) % n_u32;
        let mut rng_state = (rng_seed + 1u32) as i32;
        if rng_state == 2147483647i32 {
            rng_state = 1i32;
        }
        // Floyd's algorithm samples C distinct initial vertices, with one
        // seeded Park-Miller draw for each j in [N-C, N).
        for slot in 0..c {
            let j = reference_rows - c + slot;
            let hi = rng_state / 127773i32;
            let lo = rng_state - hi * 127773i32;
            let test = 16807i32 * lo - 2836i32 * hi;
            rng_state = if test <= 0i32 {
                test + 2147483647i32
            } else {
                test
            };
            let draw = (rng_state - 1i32) as u32 % (j as u32 + 1u32);
            let mut collision = false;
            for previous in 0..slot {
                if u32::cast_from(candidate_ids[candidate_base + previous]) == draw {
                    collision = true;
                }
            }
            let start = if collision { j as u32 } else { draw };
            let mut distance = F::new(0.0f32);
            for feature in 0..features {
                let difference = query[row * features + feature]
                    - reference[start as usize * features + feature];
                distance += difference * difference;
            }
            candidate_ids[candidate_base + slot] = I::cast_from(start);
            candidate_distances[candidate_base + slot] = distance;
            seen_ids[seen_base + slot] = I::cast_from(start);
            seen_distances[seen_base + slot] = distance;
            candidate_expanded[candidate_base + slot] = I::cast_from(0u32);
        }
        let mut evaluated = c;
        let mut peak_queue = c;
        if !invalid_graph {
            let mut searching = true;
            while searching && evaluated < v {
                let mut best_slot = c;
                let mut best_id = reference_rows;
                let mut best_distance = F::new(3.402_823_5e38_f32);
                for slot in 0..c {
                    let candidate_id =
                        u32::cast_from(candidate_ids[candidate_base + slot]) as usize;
                    if candidate_id < reference_rows
                        && candidate_expanded[candidate_base + slot] == I::cast_from(0u32)
                    {
                        let distance = candidate_distances[candidate_base + slot];
                        if best_slot == c
                            || distance < best_distance
                            || (distance == best_distance && candidate_id < best_id)
                        {
                            best_slot = slot;
                            best_id = candidate_id;
                            best_distance = distance;
                        }
                    }
                }
                if best_slot == c {
                    searching = false;
                } else {
                    candidate_expanded[candidate_base + best_slot] = I::cast_from(1u32);
                    for edge_slot in 0..graph_width {
                        if evaluated < v {
                            let raw_id = i32::cast_from(graph[best_id * graph_width + edge_slot]);
                            if raw_id >= 0 && raw_id < reference_rows as i32 {
                                let neighbor = raw_id as usize;
                                let mut already_seen = false;
                                for seen_slot in 0..evaluated {
                                    if u32::cast_from(seen_ids[seen_base + seen_slot]) as usize
                                        == neighbor
                                    {
                                        already_seen = true;
                                    }
                                }
                                if !already_seen {
                                    let mut distance = F::new(0.0f32);
                                    for feature in 0..features {
                                        let difference = query[row * features + feature]
                                            - reference[neighbor * features + feature];
                                        distance += difference * difference;
                                    }
                                    seen_ids[seen_base + evaluated] = I::cast_from(neighbor);
                                    seen_distances[seen_base + evaluated] = distance;
                                    evaluated += 1;
                                    let mut insertion = c;
                                    let mut worst_slot = c;
                                    let mut worst_distance = F::new(-1.0f32);
                                    let mut worst_id = 0usize;
                                    let mut active_count = 0usize;
                                    for slot in 0..c {
                                        let queued =
                                            u32::cast_from(candidate_ids[candidate_base + slot])
                                                as usize;
                                        if queued < reference_rows
                                            && candidate_expanded[candidate_base + slot]
                                                == I::cast_from(0u32)
                                        {
                                            active_count += 1;
                                            if insertion == c {
                                                insertion = slot;
                                            }
                                            let queued_distance =
                                                candidate_distances[candidate_base + slot];
                                            if worst_slot == c
                                                || queued_distance > worst_distance
                                                || (queued_distance == worst_distance
                                                    && queued > worst_id)
                                            {
                                                worst_slot = slot;
                                                worst_distance = queued_distance;
                                                worst_id = queued;
                                            }
                                        } else if insertion == c {
                                            insertion = slot;
                                        }
                                    }
                                    if insertion == c
                                        && worst_slot < c
                                        && (distance < worst_distance
                                            || (distance == worst_distance && neighbor < worst_id))
                                    {
                                        insertion = worst_slot;
                                    }
                                    if insertion < c {
                                        candidate_ids[candidate_base + insertion] =
                                            I::cast_from(neighbor);
                                        candidate_distances[candidate_base + insertion] = distance;
                                        candidate_expanded[candidate_base + insertion] =
                                            I::cast_from(0u32);
                                    }
                                    let retained = if active_count < c {
                                        active_count + 1
                                    } else {
                                        active_count
                                    };
                                    if retained > peak_queue {
                                        peak_queue = retained;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        for output_slot in 0..k_usize {
            let mut best = v;
            let mut best_id = reference_rows;
            let mut best_distance = F::new(3.402_823_5e38_f32);
            for seen_slot in 0..evaluated {
                let id = u32::cast_from(seen_ids[seen_base + seen_slot]) as usize;
                let distance = seen_distances[seen_base + seen_slot];
                if id < reference_rows
                    && (best == v
                        || distance < best_distance
                        || (distance == best_distance && id < best_id))
                {
                    best = seen_slot;
                    best_id = id;
                    best_distance = distance;
                }
            }
            let output = row * k_usize + output_slot;
            if best < v {
                output_indices[output] = I::cast_from(best_id);
                output_distances[output] = F::sqrt(best_distance);
                seen_ids[seen_base + best] = I::cast_from(reference_rows);
            } else {
                output_indices[output] = I::cast_from(-1i32);
                output_distances[output] = F::new(3.402_823_5e38_f32);
            }
        }
        stats[row * 2] = I::cast_from(if invalid_graph {
            -1i32
        } else {
            peak_queue as i32
        });
        stats[row * 2 + 1] = I::cast_from(if invalid_graph {
            -1i32
        } else {
            evaluated as i32
        });
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn approximate_query(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    graph: IntTensor<CubeBackend>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    approximate_query_impl(
        query,
        reference,
        graph,
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        query_offset,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn approximate_query_prevalidated(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    graph: IntTensor<CubeBackend>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    approximate_query_impl(
        query,
        reference,
        graph,
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        query_offset,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn approximate_query_impl(
    query: FloatTensor<CubeBackend>,
    reference: FloatTensor<CubeBackend>,
    graph: IntTensor<CubeBackend>,
    k: u32,
    candidate_capacity: u32,
    visit_capacity: u32,
    seed: u64,
    query_offset: u64,
    checked: bool,
) -> (
    IntTensor<CubeBackend>,
    FloatTensor<CubeBackend>,
    IntTensor<CubeBackend>,
) {
    let query = into_contiguous(query);
    let reference = into_contiguous(reference);
    let graph = into_contiguous(graph);
    let [query_rows, features] = query.shape().dims::<2>();
    let [reference_rows, reference_features] = reference.shape().dims::<2>();
    let [graph_rows, graph_width] = graph.shape().dims::<2>();
    assert!(query_rows > 0 && reference_rows > 0 && features > 0 && graph_width > 0);
    assert_eq!(features, reference_features);
    assert_eq!(reference_rows, graph_rows);
    assert_eq!(query.dtype, DType::F32);
    assert_eq!(reference.dtype, DType::F32);
    assert_eq!(graph.dtype, DType::I32);
    assert!(k > 0);
    assert!(candidate_capacity > 0);
    assert!(visit_capacity > 0);
    assert!(candidate_capacity >= k && visit_capacity >= candidate_capacity);
    let _query_rows_u32 = u32::try_from(query_rows).expect("query block exceeds u32");
    let reference_rows_u32 = u32::try_from(reference_rows).expect("reference count exceeds u32");
    i32::try_from(reference_rows).expect("reference indices exceed i32");
    let offset = u32::try_from(query_offset % u64::from(reference_rows_u32))
        .expect("query offset conversion failed");
    let seed = u32::try_from(seed % u64::from(reference_rows_u32)).expect("seed conversion failed");
    query_rows
        .checked_mul(features)
        .expect("query input shape overflows usize");
    reference_rows
        .checked_mul(features)
        .expect("reference input shape overflows usize");
    reference_rows
        .checked_mul(graph_width)
        .expect("query graph shape overflows usize");
    let k_usize = usize::try_from(k).expect("query width conversion failed");
    let candidate_capacity_usize =
        usize::try_from(candidate_capacity).expect("candidate capacity conversion failed");
    let visit_capacity_usize =
        usize::try_from(visit_capacity).expect("visit capacity conversion failed");
    assert!(k_usize <= reference_rows);
    assert!(candidate_capacity_usize <= reference_rows);
    assert!(visit_capacity_usize <= reference_rows);
    let output_allocation = query_output_allocation(query_rows, k_usize)
        .expect("query output sizes must be checked before backend invocation");
    let max_chunk_rows = approximate_query_scratch_layout(
        query_rows,
        candidate_capacity_usize,
        visit_capacity_usize,
        graph_width,
    )
    .expect("query scratch must pass estimator preflight")
    .max_chunk_rows;
    let candidate_len = max_chunk_rows
        .checked_mul(candidate_capacity_usize)
        .expect("candidate buffer size overflows usize");
    let visit_len = max_chunk_rows
        .checked_mul(visit_capacity_usize)
        .expect("visited buffer size overflows usize");
    let index_bytes = |elements: usize| {
        elements
            .checked_mul(DType::I32.size())
            .expect("query index allocation overflows usize")
    };
    let float_bytes = |elements: usize| {
        elements
            .checked_mul(query.dtype.size())
            .expect("query float allocation overflows usize")
    };
    let candidate_ids = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([candidate_len]),
        query.client.empty(index_bytes(candidate_len)),
        DType::I32,
    );
    let candidate_distances = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([candidate_len]),
        query.client.empty(float_bytes(candidate_len)),
        query.dtype,
    );
    let candidate_expanded = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([candidate_len]),
        query.client.empty(index_bytes(candidate_len)),
        DType::I32,
    );
    let seen_ids = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([visit_len]),
        query.client.empty(index_bytes(visit_len)),
        DType::I32,
    );
    let seen_distances = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([visit_len]),
        query.client.empty(float_bytes(visit_len)),
        query.dtype,
    );
    let indices = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, k_usize]),
        query.client.empty(output_allocation.output_indices_bytes),
        DType::I32,
    );
    let distances = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, k_usize]),
        query.client.empty(output_allocation.output_distances_bytes),
        query.dtype,
    );
    let stats = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, 2]),
        query.client.empty(output_allocation.stats_bytes),
        DType::I32,
    );
    let client = query.client.clone();
    let query_storage = dtype_to_storage_type(query.dtype);
    let graph_storage = dtype_to_storage_type(graph.dtype);
    let invalid_flag = if checked {
        Some(CubeTensor::new_contiguous(
            client.clone(),
            query.device.clone(),
            Shape::new([1]),
            client.empty(DType::I32.size()),
            DType::I32,
        ))
    } else {
        None
    };
    if checked {
        let validation_dim = fast_cubecl_common::legal_cube_dim(&client, CubeDim::new_1d(32));
        let validation_rows_per_cube = usize::try_from(validation_dim.num_elems())
            .expect("validation cube size exceeds usize");
        let max_validation_cubes = usize::try_from(client.properties().hardware.max_cube_count.0)
            .expect("maximum validation cube count exceeds usize");
        let max_validation_rows = max_validation_cubes
            .checked_mul(validation_rows_per_cube)
            .expect("maximum validation segment size overflows usize");
        assert!(max_validation_rows > 0);
        let invalid_flag_tensor = invalid_flag.as_ref().unwrap();
        query_flag_zero_kernel::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            invalid_flag_tensor.clone().into_tensor_arg(),
            graph_storage,
        );
        let mut validation_row_start = 0usize;
        while validation_row_start < reference_rows {
            let rows = (reference_rows - validation_row_start).min(max_validation_rows);
            let validation_cubes = rows.div_ceil(validation_rows_per_cube);
            query_validate_graph_kernel::launch(
                &client,
                CubeCount::Static(
                    u32::try_from(validation_cubes).expect("validation cube count exceeds u32"),
                    1,
                    1,
                ),
                validation_dim,
                graph.clone().into_tensor_arg(),
                reference_rows_u32,
                u32::try_from(validation_row_start).expect("validation row start exceeds u32"),
                invalid_flag_tensor.clone().into_tensor_arg(),
                graph_storage,
            );
            validation_row_start += rows;
        }
    }
    let mut row_start = 0;
    while row_start < query_rows {
        let rows = query_chunk_rows(query_rows, row_start, max_chunk_rows);
        let row_start_u32 = u32::try_from(row_start).expect("query row start exceeds u32");
        let rows_u32 = u32::try_from(rows).expect("query chunk exceeds u32");
        let (cube_count, cube_dim) = if client.name() == "cpu" {
            let cores = std::thread::available_parallelism()
                .map_or(1, |cores| u32::try_from(cores.get()).unwrap_or(u32::MAX));
            let hardware = &client.properties().hardware;
            query_cpu_chunk_geometry(
                rows_u32,
                cores,
                hardware.max_cube_dim.0,
                hardware.max_units_per_cube,
                hardware.max_cube_count.0,
            )
        } else {
            assert!(
                rows_u32 <= client.properties().hardware.max_cube_count.0,
                "query chunk requires x cube count {rows_u32}; maximum is {}",
                client.properties().hardware.max_cube_count.0
            );
            (CubeCount::Static(rows_u32, 1, 1), QUERY_CUBE_DIM)
        };
        approximate_query_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            query.clone().into_tensor_arg(),
            reference.clone().into_tensor_arg(),
            graph.clone().into_tensor_arg(),
            k,
            candidate_capacity,
            visit_capacity,
            seed,
            offset,
            row_start_u32,
            rows_u32,
            match invalid_flag.as_ref() {
                Some(flag) => ComptimeOptionArgs::Some(flag.clone().into_tensor_arg()),
                None => ComptimeOptionArgs::None,
            },
            checked,
            candidate_ids.clone().into_tensor_arg(),
            candidate_distances.clone().into_tensor_arg(),
            candidate_expanded.clone().into_tensor_arg(),
            seen_ids.clone().into_tensor_arg(),
            seen_distances.clone().into_tensor_arg(),
            indices.clone().into_tensor_arg(),
            distances.clone().into_tensor_arg(),
            stats.clone().into_tensor_arg(),
            [query_storage, graph_storage],
        );
        row_start += rows;
    }
    (indices, distances, stats)
}

#[cfg(test)]
mod tests {
    use super::{CubeCount, query_cpu_chunk_geometry};

    fn geometry(rows: u32, cores: u32, max_x: u32, max_units: u32, max_cubes: u32) -> (u32, u32) {
        let (CubeCount::Static(count, 1, 1), dim) =
            query_cpu_chunk_geometry(rows, cores, max_x, max_units, max_cubes)
        else {
            panic!("query geometry must use a static one-dimensional count");
        };
        (count, dim.x)
    }

    #[test]
    fn query_cpu_chunk_geometry_caps_width_and_ceil_counts() {
        assert_eq!(geometry(1023, 15, 32, 32, 1023), (69, 15));
        assert_eq!(geometry(4, 15, 32, 32, 4), (1, 4));
        assert_eq!(geometry(23, 32, 32, 32, 23), (1, 23));
        assert_eq!(geometry(23, 32, 7, 32, 23), (4, 7));
        assert_eq!(geometry(23, 32, 32, 5, 23), (5, 5));
        assert_eq!(geometry(23, 6, 7, 5, 23), (5, 5));
    }

    #[test]
    fn query_cpu_chunk_geometry_rejects_cube_count_overflow() {
        assert!(std::panic::catch_unwind(|| geometry(23, 7, 7, 7, 3)).is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn query_cpu_chunk_geometry_reports_runtime_limits() {
        let client = burn_cubecl::CubeDevice::cpu().unwrap().client();
        let hardware = &client.properties().hardware;
        let cores = u32::try_from(std::thread::available_parallelism().unwrap().get()).unwrap();
        for rows in [1023, 4] {
            let (count, dim) = query_cpu_chunk_geometry(
                rows,
                cores,
                hardware.max_cube_dim.0,
                hardware.max_units_per_cube,
                hardware.max_cube_count.0,
            );
            eprintln!("query runtime rows={rows} cores={cores} count={count:?} dim={dim:?}");
            assert!(dim.x <= rows && dim.x <= cores);
        }
    }
}
