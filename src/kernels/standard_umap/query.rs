use burn::{
    backend::{
        cubecl::dtype_to_storage_type,
        tensor::{FloatTensor, IntTensor},
        TensorMetadata,
    },
    tensor::{DType, Shape},
};
use burn_cubecl::{kernel::into_contiguous, tensor::CubeTensor, CubeBackend};
use cubecl::prelude::*;

const QUERY_CUBE_DIM: CubeDim = CubeDim::new_1d(1);

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
    let row = ABSOLUTE_POS_X as usize;
    let query_rows = query.shape(0);
    let reference_rows = reference.shape(0);
    let features = query.shape(1);
    let graph_width = graph.shape(1);
    let c = candidate_capacity as usize;
    let v = visit_capacity as usize;
    let k_usize = k as usize;
    if row < query_rows {
        let mut invalid_graph = false;
        for node in 0..reference_rows {
            for slot in 0..graph_width {
                let edge = i32::cast_from(graph[node * graph_width + slot]);
                if edge < -1 || edge >= reference_rows as i32 {
                    invalid_graph = true;
                }
            }
        }
        let candidate_base = row * c;
        let seen_base = row * v;
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
    assert!(k > 0 && k as usize <= reference_rows);
    assert!(candidate_capacity > 0 && candidate_capacity as usize <= reference_rows);
    assert!(visit_capacity > 0 && visit_capacity as usize <= reference_rows);
    assert!(candidate_capacity >= k && visit_capacity >= candidate_capacity);
    let query_rows_u32 = u32::try_from(query_rows).expect("query block exceeds u32");
    let reference_rows_u32 = u32::try_from(reference_rows).expect("reference count exceeds u32");
    i32::try_from(reference_rows).expect("reference indices exceed i32");
    let offset = u32::try_from(query_offset % u64::from(reference_rows_u32))
        .expect("query offset conversion failed");
    let seed = u32::try_from(seed % u64::from(reference_rows_u32)).expect("seed conversion failed");
    let output_len = query_rows
        .checked_mul(k as usize)
        .expect("query output size overflows usize");
    let candidate_len = query_rows
        .checked_mul(candidate_capacity as usize)
        .expect("candidate buffer size overflows usize");
    let visit_len = query_rows
        .checked_mul(visit_capacity as usize)
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
        Shape::new([query_rows, k as usize]),
        query.client.empty(index_bytes(output_len)),
        DType::I32,
    );
    let distances = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, k as usize]),
        query.client.empty(float_bytes(output_len)),
        query.dtype,
    );
    let stats = CubeTensor::new_contiguous(
        query.client.clone(),
        query.device.clone(),
        Shape::new([query_rows, 2]),
        query.client.empty(index_bytes(
            query_rows
                .checked_mul(2)
                .expect("query stats allocation overflows usize"),
        )),
        DType::I32,
    );
    let client = query.client.clone();
    let query_storage = dtype_to_storage_type(query.dtype);
    let graph_storage = dtype_to_storage_type(graph.dtype);
    approximate_query_kernel::launch(
        &client,
        CubeCount::Static(query_rows_u32, 1, 1),
        QUERY_CUBE_DIM,
        query.into_tensor_arg(),
        reference.into_tensor_arg(),
        graph.into_tensor_arg(),
        k,
        candidate_capacity,
        visit_capacity,
        seed,
        offset,
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
    (indices, distances, stats)
}
