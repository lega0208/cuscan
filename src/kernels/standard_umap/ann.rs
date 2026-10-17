//! Direct CubeCL NN-Descent kernels for supported standard-UMAP runtimes.

use burn::{
    backend::{TensorMetadata, cubecl::dtype_to_storage_type, tensor::FloatTensor},
    cubecl::prelude::*,
    tensor::{DType, Shape},
};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};

use crate::standard_umap::ann::candidate_width;

const ANN_CUBE_DIM: CubeDim = CubeDim::new_1d(32);

fn ann_launch_geometry(
    client: &cubecl::client::Client,
    feature_size: usize,
    n_rows: u32,
) -> (bool, CubeCount, CubeDim) {
    let cpu_merge = client.name() == "cpu";
    let hardware = &client.properties().hardware;
    if !cpu_merge {
        validate_cooperative_ann_launch(hardware, client.features(), feature_size, n_rows);
    }
    ann_launch_geometry_with_limits(
        cpu_merge,
        (
            hardware.max_cube_dim.0,
            hardware.max_cube_dim.1,
            hardware.max_cube_dim.2,
        ),
        hardware.max_units_per_cube,
        hardware.max_cube_count.0,
        std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .try_into()
            .unwrap_or(u32::MAX),
        n_rows,
    )
}

fn ann_launch_geometry_with_limits(
    cpu_merge: bool,
    max_cube_dim: (u32, u32, u32),
    max_units_per_cube: u32,
    max_cubes: u32,
    cores: u32,
    n_rows: u32,
) -> (bool, CubeCount, CubeDim) {
    if cpu_merge {
        assert!(n_rows > 0, "ANN row count must be nonzero");
        let dim_x = n_rows
            .min(cores)
            .min(max_cube_dim.0)
            .min(max_units_per_cube)
            .max(1);
        let dim = CubeDim::new_1d(dim_x);
        let count = ann_independent_cube_count(n_rows, dim.x, max_cubes);
        (true, CubeCount::Static(count, 1, 1), dim)
    } else {
        assert!(
            n_rows <= max_cubes,
            "ANN cooperative cube count exceeds hardware limit"
        );
        (false, CubeCount::Static(n_rows, 1, 1), ANN_CUBE_DIM)
    }
}

fn validate_cooperative_ann_launch(
    hardware: &cubecl::ir::HardwareProperties,
    features: &cubecl::features::Features,
    feature_size: usize,
    n_rows: u32,
) {
    use cubecl::features::Plane;

    assert!(
        features.plane.contains(Plane::Ops),
        "cooperative ANN requires Plane::Ops for plane_any; observed plane features {:?}",
        features.plane
    );
    // plane_any is whole-cube; the guaranteed plane must cover all 32 tiled units.
    // A wider plane is valid because inactive lanes contribute false.
    assert!(
        hardware.plane_size_min >= ANN_CUBE_DIM.x,
        "cooperative ANN requires guaranteed plane width >= {}; observed min={}, max={}",
        ANN_CUBE_DIM.x,
        hardware.plane_size_min,
        hardware.plane_size_max
    );
    assert!(
        hardware.max_units_per_cube >= ANN_CUBE_DIM.x && hardware.max_cube_dim.0 >= ANN_CUBE_DIM.x,
        "cooperative ANN requires 32 units and x dimension; observed units={}, max_cube_dim={:?}",
        hardware.max_units_per_cube,
        hardware.max_cube_dim
    );
    let shared_bytes = 32 * (2 * DType::I32.size() + feature_size) + DType::I32.size();
    // This is a lower bound: compiler padding/alignment and internal usage are
    // excluded; CubeCL's launch-time shared-memory checks remain authoritative.
    assert!(
        shared_bytes <= hardware.max_shared_memory_size,
        "cooperative ANN requires at least {shared_bytes} shared bytes (lower bound); observed {}",
        hardware.max_shared_memory_size
    );
    assert!(
        hardware.max_cube_count.0 >= n_rows,
        "cooperative ANN requires x cube count >= {n_rows}; observed {}",
        hardware.max_cube_count.0
    );
}

fn ann_independent_launch(client: &cubecl::client::Client, units: u32) -> (CubeCount, CubeDim) {
    let dim = crate::kernels::legal_cube_dim(client, CubeDim::new_1d(ANN_CUBE_DIM.x));
    let count =
        ann_independent_cube_count(units, dim.x, client.properties().hardware.max_cube_count.0);
    (CubeCount::Static(count, 1, 1), dim)
}

fn ann_independent_cube_count(units: u32, units_per_cube: u32, max_cubes: u32) -> u32 {
    assert!(
        units_per_cube > 0,
        "ANN independent launch dimension must be nonzero"
    );
    let count = units.div_ceil(units_per_cube);
    assert!(
        count <= max_cubes,
        "ANN independent launch requires x cube count {count}; maximum is {max_cubes}"
    );
    count
}

fn ann_should_stop(new_count: i32, n: usize, width: usize) -> bool {
    let capacity = n
        .checked_mul(width)
        .expect("ANN candidate elements overflow usize");
    assert!(capacity <= i32::MAX as usize, "ANN count exceeds i32");
    assert!(
        new_count >= 0 && (new_count as usize) <= capacity,
        "ANN new-membership count is outside candidate capacity"
    );
    (new_count as u64) * 1000 <= capacity as u64
}

#[cube]
fn reverse_first_hit<I: Int>(
    candidates: &Tensor<I>,
    source: usize,
    slot: usize,
    n: usize,
    width: usize,
) -> bool {
    let target = u32::cast_from(candidates[source * width + slot]) as usize;
    let mut first = target < n;
    if first {
        for previous in 0..slot {
            if candidates[source * width + previous] == candidates[source * width + slot] {
                first = false;
            }
        }
    }
    first
}

#[cube(launch)]
fn reverse_zero<I: Int>(counts: &mut Tensor<I>, #[define(I)] _dtype: ElemType) {
    let target = ABSOLUTE_POS_X as usize;
    if target < counts.shape(0) {
        counts[target] = I::cast_from(0i32);
    }
}

#[cube(launch)]
fn reverse_count<I: Int>(
    candidates: &Tensor<I>,
    counts: &mut Tensor<Atomic<I>>,
    n: u32,
    width: u32,
    #[define(I)] _dtype: ElemType,
) {
    let source = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if source < n {
        for slot in 0..width {
            if reverse_first_hit(candidates, source, slot, n, width) {
                let target = u32::cast_from(candidates[source * width + slot]) as usize;
                counts[target].fetch_add(I::cast_from(1i32));
            }
        }
    }
}

#[cube(launch)]
fn reverse_prefix<I: Int>(
    counts: &Tensor<I>,
    offsets: &mut Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    #[define(I)] _dtype: ElemType,
) {
    if ABSOLUTE_POS_X == 0 {
        let n = counts.shape(0);
        let mut running = I::cast_from(0i32);
        offsets[0] = running;
        for target in 0..n {
            cursors[target].store(running);
            running += counts[target];
            offsets[target + 1] = running;
        }
    }
}

#[cube(launch)]
fn reverse_scatter<I: Int>(
    candidates: &Tensor<I>,
    cursors: &mut Tensor<Atomic<I>>,
    sources: &mut Tensor<I>,
    n: u32,
    width: u32,
    #[define(I)] _dtype: ElemType,
) {
    let source = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if source < n {
        for slot in 0..width {
            if reverse_first_hit(candidates, source, slot, n, width) {
                let target = u32::cast_from(candidates[source * width + slot]) as usize;
                let position =
                    u32::cast_from(cursors[target].fetch_add(I::cast_from(1i32))) as usize;
                sources[position] = I::cast_from(source as i32);
            }
        }
    }
}

#[cube(launch)]
fn ann_init<I: Int>(
    indices: &mut Tensor<I>,
    n: u32,
    width: u32,
    seed_base: u32,
    #[define(I)] _dtype: ElemType,
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = n as usize;
    let width = width as usize;
    if row < n {
        let rng_modulus = I::cast_from(2_147_483_647i32);
        let seed_base = I::cast_from(seed_base);
        let row_value = I::cast_from(row as i32);
        let mut state = if row_value >= rng_modulus - seed_base {
            row_value - (rng_modulus - seed_base)
        } else {
            row_value + seed_base
        };
        if state == I::cast_from(0i32) {
            state = I::cast_from(1i32);
        }
        for column in 0..width {
            let hi = state / I::cast_from(127_773i32);
            let lo = state - hi * I::cast_from(127_773i32);
            let test = lo * I::cast_from(16_807i32) - hi * I::cast_from(2_836i32);
            state = if test <= I::cast_from(0i32) {
                test + I::cast_from(2_147_483_647i32)
            } else {
                test
            };
            let t = state - I::cast_from(1i32);
            let j = n - 1 - width + column;
            let bound = I::cast_from((j + 1) as i32);
            let draw = t - (t / bound) * bound;
            let mut collision = false;
            for previous in 0..column {
                let previous_id = I::cast_from(indices[row * width + previous]);
                let previous_raw = if previous_id > row_value {
                    previous_id - I::cast_from(1i32)
                } else {
                    previous_id
                };
                if previous_raw == draw {
                    collision = true;
                }
            }
            let chosen = if collision {
                I::cast_from(j as i32)
            } else {
                draw
            };
            let candidate = if chosen >= row_value {
                chosen + I::cast_from(1i32)
            } else {
                chosen
            };
            indices[row * width + column] = candidate;
        }
    }
}

#[cube]
fn cooperative_merge_tile<F: Float, I: Int>(
    n: usize,
    row: usize,
    width: usize,
    proposal: I,
    proposal_squared: F,
    proposal_valid: u32,
    toggle: u32,
    ids_a: &mut Tensor<I>,
    distances_a: &mut Tensor<F>,
    ids_b: &mut Tensor<I>,
    distances_b: &mut Tensor<F>,
    proposal_ids: &mut Shared<[I]>,
    proposal_distances: &mut Shared<[F]>,
    proposal_valids: &mut Shared<[I]>,
    row_base: usize,
) -> u32 {
    let lane = UNIT_POS_X as usize;
    proposal_ids[lane] = proposal;
    proposal_distances[lane] = proposal_squared;
    proposal_valids[lane] = I::cast_from(proposal_valid as i32);
    sync_cube();

    let sentinel = I::cast_from(n as i32);
    let max_distance = F::new(3.402_823_5e38_f32);
    let proposal_id = proposal_ids[lane];
    let proposal_index = u32::cast_from(proposal_id) as usize;
    let mut proposal_novel: bool = proposal_valids[lane] != I::cast_from(0i32)
        && proposal_index < n
        && proposal_index != row
        && proposal_distances[lane] <= max_distance;
    let mut old_slot = 0usize;
    while proposal_novel && old_slot < width {
        let old_id = if toggle == 0 {
            ids_a[row_base + old_slot]
        } else {
            ids_b[row_base + old_slot]
        };
        if old_id == proposal_id {
            proposal_novel = false;
        }
        old_slot += 1;
    }
    let mut previous_lane = 0usize;
    while proposal_novel && previous_lane < lane {
        if proposal_valids[previous_lane] != I::cast_from(0i32)
            && proposal_ids[previous_lane] == proposal_id
        {
            proposal_novel = false;
        }
        previous_lane += 1;
    }
    sync_cube();
    proposal_valids[lane] = I::cast_from(if proposal_novel { 1i32 } else { 0i32 });
    sync_cube();

    let any_novel: bool = plane_any(proposal_novel);
    if any_novel {
        let mut position = lane;
        while position < width {
            if toggle == 0 {
                ids_b[row_base + position] = sentinel;
                distances_b[row_base + position] = max_distance;
            } else {
                ids_a[row_base + position] = sentinel;
                distances_a[row_base + position] = max_distance;
            }
            position += 32;
        }
        sync_cube();

        // Existing candidates are already unique and lex-sorted. A candidate's
        // merged rank is its old rank plus the number of novel tile entries before it.
        position = lane;
        while position < width {
            let id = if toggle == 0 {
                ids_a[row_base + position]
            } else {
                ids_b[row_base + position]
            };
            let distance = if toggle == 0 {
                distances_a[row_base + position]
            } else {
                distances_b[row_base + position]
            };
            let id_index = u32::cast_from(id) as usize;
            if id_index < n && distance <= max_distance {
                let mut rank = position;
                for other_lane in 0..32 {
                    if proposal_valids[other_lane] != I::cast_from(0i32) {
                        let other_id = proposal_ids[other_lane];
                        let other_distance = proposal_distances[other_lane];
                        let other_index = u32::cast_from(other_id) as usize;
                        if other_distance < distance
                            || (other_distance == distance && other_index < id_index)
                        {
                            rank += 1;
                        }
                    }
                }
                if rank < width {
                    if toggle == 0 {
                        ids_b[row_base + rank] = id;
                        distances_b[row_base + rank] = distance;
                    } else {
                        ids_a[row_base + rank] = id;
                        distances_a[row_base + rank] = distance;
                    }
                }
            }
            position += 32;
        }

        // Each lane contributes at most one unique proposal; duplicate ids are
        // owned by their lowest lane and proposals already present in the list vanish.
        let id = proposal_ids[lane];
        let distance = proposal_distances[lane];
        let id_index = u32::cast_from(id) as usize;
        if proposal_valids[lane] != I::cast_from(0i32) {
            let mut rank = 0usize;
            let mut old_slot = 0usize;
            while old_slot < width {
                let old_id = if toggle == 0 {
                    ids_a[row_base + old_slot]
                } else {
                    ids_b[row_base + old_slot]
                };
                let old_distance = if toggle == 0 {
                    distances_a[row_base + old_slot]
                } else {
                    distances_b[row_base + old_slot]
                };
                let old_index = u32::cast_from(old_id) as usize;
                if old_index < n
                    && (old_distance < distance
                        || (old_distance == distance && old_index < id_index))
                {
                    rank += 1;
                }
                old_slot += 1;
            }
            for other_lane in 0..32 {
                if proposal_valids[other_lane] != I::cast_from(0i32) {
                    let other_id = proposal_ids[other_lane];
                    let other_distance = proposal_distances[other_lane];
                    let other_index = u32::cast_from(other_id) as usize;
                    if other_distance < distance
                        || (other_distance == distance && other_index < id_index)
                    {
                        rank += 1;
                    }
                }
            }
            if rank < width {
                if toggle == 0 {
                    ids_b[row_base + rank] = id;
                    distances_b[row_base + rank] = distance;
                } else {
                    ids_a[row_base + rank] = id;
                    distances_a[row_base + rank] = distance;
                }
            }
        }
        sync_cube();
        toggle ^ 1u32
    } else {
        toggle
    }
}

#[cube]
fn sequential_merge_candidate<F: Float, I: Int>(
    n: usize,
    row: usize,
    width: usize,
    proposal: I,
    proposal_squared: F,
    proposal_valid: bool,
    indices: &mut Tensor<I>,
    distances: &mut Tensor<F>,
    row_base: usize,
) {
    let max_distance = F::new(3.402_823_5e38_f32);
    let proposal_index = u32::cast_from(proposal) as usize;
    if proposal_valid
        && proposal_index < n
        && proposal_index != row
        && proposal_squared <= max_distance
    {
        let mut duplicate = false;
        for slot in 0..width {
            if indices[row_base + slot] == proposal {
                duplicate = true;
            }
        }
        if !duplicate {
            let mut rank = 0usize;
            for slot in 0..width {
                let old_id = indices[row_base + slot];
                let old_index = u32::cast_from(old_id) as usize;
                if old_index < n
                    && (distances[row_base + slot] < proposal_squared
                        || (distances[row_base + slot] == proposal_squared
                            && old_index < proposal_index))
                {
                    rank += 1;
                }
            }
            if rank < width {
                let mut position = width;
                while position > rank + 1 {
                    indices[row_base + position - 1] = indices[row_base + position - 2];
                    distances[row_base + position - 1] = distances[row_base + position - 2];
                    position -= 1;
                }
                indices[row_base + rank] = proposal;
                distances[row_base + rank] = proposal_squared;
            }
        }
    }
}

#[cube(launch)]
fn ann_step<F: Float, I: Int>(
    data: &Tensor<F>,
    candidates: &Tensor<I>,
    candidate_distances: &Tensor<F>,
    candidate_retained: &Tensor<I>,
    reverse_offsets: &Tensor<I>,
    reverse_sources: &Tensor<I>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    workspace_indices: &mut Tensor<I>,
    workspace_distances: &mut Tensor<F>,
    output_retained: &mut Tensor<I>,
    new_count: &mut Tensor<Atomic<I>>,
    width: u32,
    do_expand: u32,
    #[comptime] cpu_merge: bool,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = if comptime![cpu_merge] {
        (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize
    } else {
        CUBE_POS_X as usize
    };
    let lane = UNIT_POS_X as usize;
    let n = data.shape(0);
    let features = data.shape(1);
    let width = width as usize;
    if row < n {
        if comptime![cpu_merge] {
            let base = row * width;
            let max_distance = F::new(3.402_823_5e38_f32);
            for slot in 0..width {
                output_indices[base + slot] = I::cast_from(n as i32);
                output_distances[base + slot] = max_distance;
            }

            // This CPU path processes a whole row in one unit: no plane
            // operations, shared memory, or cross-lane synchronization.
            for proposal_position in 0..width {
                let proposal = candidates[base + proposal_position];
                let mut proposal_squared = max_distance;
                if do_expand != 0 {
                    proposal_squared = candidate_distances[base + proposal_position];
                } else {
                    let candidate_index = u32::cast_from(proposal) as usize;
                    if candidate_index < n && candidate_index != row {
                        proposal_squared = F::new(0.0f32);
                        for feature in 0..features {
                            let delta = data[row * features + feature]
                                - data[candidate_index * features + feature];
                            proposal_squared += delta * delta;
                        }
                    }
                }
                sequential_merge_candidate::<F, I>(
                    n,
                    row,
                    width,
                    proposal,
                    proposal_squared,
                    true,
                    output_indices,
                    output_distances,
                    base,
                );
            }

            if do_expand != 0 {
                for outer in 0..width {
                    let neighbor = u32::cast_from(candidates[base + outer]) as usize;
                    if neighbor < n {
                        let neighbor_retained =
                            candidate_retained[base + outer] != I::cast_from(0i32);
                        for inner in 0..width {
                            let inner_retained =
                                candidate_retained[neighbor * width + inner] != I::cast_from(0i32);
                            if !neighbor_retained || !inner_retained {
                                let proposal = candidates[neighbor * width + inner];
                                let candidate_index = u32::cast_from(proposal) as usize;
                                let mut proposal_squared = max_distance;
                                if candidate_index < n && candidate_index != row {
                                    proposal_squared = F::new(0.0f32);
                                    for feature in 0..features {
                                        let delta = data[row * features + feature]
                                            - data[candidate_index * features + feature];
                                        proposal_squared += delta * delta;
                                    }
                                }
                                sequential_merge_candidate::<F, I>(
                                    n,
                                    row,
                                    width,
                                    proposal,
                                    proposal_squared,
                                    true,
                                    output_indices,
                                    output_distances,
                                    base,
                                );
                            }
                        }
                    }
                }

                let start = u32::cast_from(reverse_offsets[row]) as usize;
                let end = u32::cast_from(reverse_offsets[row + 1]) as usize;
                for position in start..end {
                    let source = u32::cast_from(reverse_sources[position]) as usize;
                    let source_base = source * width;
                    let mut row_slot = width;
                    for source_slot in 0..width {
                        if row_slot == width
                            && candidates[source_base + source_slot] == I::cast_from(row as i32)
                        {
                            row_slot = source_slot;
                        }
                    }
                    let row_retained = row_slot < width
                        && candidate_retained[source_base + row_slot] != I::cast_from(0i32);
                    for reverse_slot in 0..width {
                        if !row_retained
                            || candidate_retained[source_base + reverse_slot] == I::cast_from(0i32)
                        {
                            let proposal = candidates[source_base + reverse_slot];
                            let candidate_index = u32::cast_from(proposal) as usize;
                            let mut proposal_squared = max_distance;
                            if candidate_index < n && candidate_index != row {
                                proposal_squared = F::new(0.0f32);
                                for feature in 0..features {
                                    let delta = data[row * features + feature]
                                        - data[candidate_index * features + feature];
                                    proposal_squared += delta * delta;
                                }
                            }
                            sequential_merge_candidate::<F, I>(
                                n,
                                row,
                                width,
                                proposal,
                                proposal_squared,
                                true,
                                output_indices,
                                output_distances,
                                base,
                            );
                        }
                    }
                }
            }

            let mut row_new = I::cast_from(0i32);
            for output_slot in 0..width {
                let output_id = output_indices[base + output_slot];
                let output_index = u32::cast_from(output_id) as usize;
                let mut retained = false;
                if do_expand != 0 && output_index < n {
                    for previous_slot in 0..width {
                        if candidates[base + previous_slot] == output_id {
                            retained = true;
                        }
                    }
                }
                output_retained[base + output_slot] =
                    I::cast_from(if retained { 1i32 } else { 0i32 });
                if do_expand != 0 && output_index < n && !retained {
                    row_new += I::cast_from(1i32);
                }
            }
            if row_new > I::cast_from(0i32) {
                new_count[0].fetch_add(row_new);
            }
        } else {
            let base = row * width;
            let mut proposal_ids = Shared::<[I]>::new_slice(32usize);
            let mut proposal_distances = Shared::<[F]>::new_slice(32usize);
            let mut proposal_valids = Shared::<[I]>::new_slice(32usize);
            let mut reverse_skip = Shared::<[I]>::new_slice(1usize);
            let mut toggle = 0u32;
            let mut slot = lane;
            while slot < width {
                output_indices[base + slot] = I::cast_from(n as i32);
                output_distances[base + slot] = F::new(3.402_823_5e38_f32);
                slot += 32;
            }
            sync_cube();

            // Own plus forward proposals form a rectangular, lane-parallel stream.
            let own_and_forward = if do_expand != 0 {
                width + width * width
            } else {
                width
            };
            let mut tile = 0usize;
            while tile * 32 < own_and_forward {
                let proposal_position = tile * 32 + lane;
                let mut proposal = I::cast_from(n as i32);
                let mut proposal_squared = F::new(3.402_823_5e38_f32);
                let mut valid = false;
                let mut reuse = false;
                if proposal_position < width {
                    proposal = candidates[base + proposal_position];
                    valid = true;
                    reuse = do_expand != 0;
                    if reuse {
                        proposal_squared = candidate_distances[base + proposal_position];
                    }
                } else if proposal_position < own_and_forward {
                    let forward_position = proposal_position - width;
                    let outer = forward_position / width;
                    let inner = forward_position - outer * width;
                    let neighbor = u32::cast_from(candidates[base + outer]) as usize;
                    if neighbor < n {
                        let neighbor_retained =
                            candidate_retained[base + outer] != I::cast_from(0i32);
                        let inner_retained =
                            candidate_retained[neighbor * width + inner] != I::cast_from(0i32);
                        if !neighbor_retained || !inner_retained {
                            proposal = candidates[neighbor * width + inner];
                            valid = true;
                        }
                    }
                }
                if valid && !reuse {
                    let candidate_index = u32::cast_from(proposal) as usize;
                    if candidate_index < n && candidate_index != row {
                        proposal_squared = F::new(0.0f32);
                        for feature in 0..features {
                            let delta = data[row * features + feature]
                                - data[candidate_index * features + feature];
                            proposal_squared += delta * delta;
                        }
                    }
                }
                toggle = cooperative_merge_tile::<F, I>(
                    n,
                    row,
                    width,
                    proposal,
                    proposal_squared,
                    if valid { 1u32 } else { 0u32 },
                    toggle,
                    output_indices,
                    output_distances,
                    workspace_indices,
                    workspace_distances,
                    &mut proposal_ids,
                    &mut proposal_distances,
                    &mut proposal_valids,
                    base,
                );
                tile += 1;
            }

            if do_expand != 0 {
                let start = u32::cast_from(reverse_offsets[row]) as usize;
                let end = u32::cast_from(reverse_offsets[row + 1]) as usize;
                for position in start..end {
                    let source = u32::cast_from(reverse_sources[position]) as usize;
                    let source_base = source * width;
                    if lane == 0 {
                        let mut row_slot = width;
                        for source_slot in 0..width {
                            if row_slot == width
                                && candidates[source_base + source_slot] == I::cast_from(row as i32)
                            {
                                row_slot = source_slot;
                            }
                        }
                        reverse_skip[0] = I::cast_from(
                            if row_slot < width
                                && candidate_retained[source_base + row_slot] != I::cast_from(0i32)
                            {
                                1i32
                            } else {
                                0i32
                            },
                        );
                    }
                    sync_cube();
                    let row_retained = reverse_skip[0] != I::cast_from(0i32);
                    let mut reverse_tile = 0usize;
                    while reverse_tile * 32 < width {
                        let reverse_slot = reverse_tile * 32 + lane;
                        let mut proposal = I::cast_from(n as i32);
                        let mut valid = false;
                        if reverse_slot < width
                            && (!row_retained
                                || candidate_retained[source_base + reverse_slot]
                                    == I::cast_from(0i32))
                        {
                            proposal = candidates[source_base + reverse_slot];
                            valid = true;
                        }
                        let mut proposal_squared = F::new(3.402_823_5e38_f32);
                        if valid {
                            let candidate_index = u32::cast_from(proposal) as usize;
                            if candidate_index < n && candidate_index != row {
                                proposal_squared = F::new(0.0f32);
                                for feature in 0..features {
                                    let delta = data[row * features + feature]
                                        - data[candidate_index * features + feature];
                                    proposal_squared += delta * delta;
                                }
                            }
                        }
                        toggle = cooperative_merge_tile::<F, I>(
                            n,
                            row,
                            width,
                            proposal,
                            proposal_squared,
                            if valid { 1u32 } else { 0u32 },
                            toggle,
                            output_indices,
                            output_distances,
                            workspace_indices,
                            workspace_distances,
                            &mut proposal_ids,
                            &mut proposal_distances,
                            &mut proposal_valids,
                            base,
                        );
                        reverse_tile += 1;
                    }
                }
            }

            sync_cube();
            if toggle != 0 {
                slot = lane;
                while slot < width {
                    output_indices[base + slot] = workspace_indices[base + slot];
                    output_distances[base + slot] = workspace_distances[base + slot];
                    slot += 32;
                }
                sync_cube();
            }
            if lane == 0 {
                let mut row_new = I::cast_from(0i32);
                for output_slot in 0..width {
                    let output_id = output_indices[base + output_slot];
                    let output_index = u32::cast_from(output_id) as usize;
                    let mut retained = false;
                    if do_expand != 0 && output_index < n {
                        for previous_slot in 0..width {
                            if candidates[base + previous_slot] == output_id {
                                retained = true;
                            }
                        }
                    }
                    output_retained[base + output_slot] =
                        I::cast_from(if retained { 1i32 } else { 0i32 });
                    if do_expand != 0 && output_index < n && !retained {
                        row_new += I::cast_from(1i32);
                    }
                }
                if row_new > I::cast_from(0i32) {
                    new_count[0].fetch_add(row_new);
                }
            }
        }
    }
}

#[cube(launch)]
fn ann_output<F: Float, I: Int>(
    indices: &Tensor<I>,
    squared_distances: &Tensor<F>,
    output_indices: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    width: u32,
    k: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let row = ABSOLUTE_POS_X as usize;
    let n = indices.shape(0);
    let width = width as usize;
    let k = k as usize;
    if row < n {
        for slot in 0..k {
            let source = row * width + slot;
            let output = row * k + slot;
            output_indices[output] = indices[source];
            output_distances[output] = F::sqrt(squared_distances[source]);
        }
    }
}

/// Approximate neighbors on CUDA. Candidate lists and all feature-distance
/// evaluations and deterministic candidate seeding reside on CubeCL.
pub(crate) fn approximate_neighbors(
    data: CubeTensor,
    k: u32,
    seed: u64,
    max_iterations: u32,
) -> (CubeTensor, CubeTensor) {
    let debug_timing = std::env::var_os("FAST_UMAP_ANN_DEBUG").is_some();
    let data = into_contiguous(data);
    let [n, features] = data.shape().dims::<2>();
    assert!(
        n >= 2 && features > 0,
        "ANN input requires at least two rows and one feature"
    );
    assert_eq!(data.dtype, DType::F32, "ANN requires float32 features");
    assert!(n <= i32::MAX as usize, "ANN row indices exceed i32");
    let k = usize::try_from(k).expect("ANN width exceeds usize");
    assert!(k > 0 && k < n, "ANN neighbor width must be in 1..N");
    let width = candidate_width(n, k);
    let elements = n
        .checked_mul(width)
        .expect("ANN candidate elements overflow usize");
    let offset_len = n
        .checked_add(1)
        .expect("ANN reverse offsets length overflows usize");
    assert!(
        elements <= i32::MAX as usize,
        "ANN candidate source capacity and reverse CSR prefix exceed i32"
    );
    let index_bytes = elements
        .checked_mul(DType::I32.size())
        .expect("ANN index storage overflows usize");
    let distance_bytes = elements
        .checked_mul(DType::F32.size())
        .expect("ANN distance storage overflows usize");
    let reverse_count_bytes = n
        .checked_mul(DType::I32.size())
        .expect("ANN reverse count allocation overflows usize");
    let reverse_offset_bytes = offset_len
        .checked_mul(DType::I32.size())
        .expect("ANN reverse offset allocation overflows usize");
    let n_u32 = u32::try_from(n).expect("ANN row count exceeds CubeCL range");
    let width_u32 = u32::try_from(width).expect("ANN candidate width exceeds CubeCL range");
    // CPU geometry is row-parallel; cooperative CUDA geometry is validated, never clamped.
    let (cpu_merge, ann_step_count, ann_step_dim) =
        ann_launch_geometry(&data.client, data.dtype.size(), n_u32);
    let _feature_count_u32 =
        u32::try_from(features).expect("ANN feature count exceeds CubeCL range");
    let client = data.client.clone();
    let device = data.device.clone();
    let reverse_counts = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(reverse_count_bytes),
        DType::I32,
    );
    let reverse_offsets = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([offset_len]),
        client.empty(reverse_offset_bytes),
        DType::I32,
    );
    let reverse_cursors = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([n]),
        client.empty(reverse_count_bytes),
        DType::I32,
    );
    let reverse_sources = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::new([elements]),
        client.empty(index_bytes),
        DType::I32,
    );
    let mut candidates = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    let (helper_count, helper_dim) = ann_independent_launch(&data.client, n_u32);
    let init_start = debug_timing.then(std::time::Instant::now);
    ann_init::launch(
        &data.client,
        helper_count,
        helper_dim,
        candidates.clone().into_tensor_arg(),
        u32::try_from(n).expect("ANN row count exceeds CubeCL range"),
        u32::try_from(width).expect("ANN candidate width exceeds CubeCL range"),
        (seed % 2_147_483_646u64) as u32 + 1,
        dtype_to_storage_type(DType::I32),
    );
    if let Some(start) = init_start {
        let _ = data.client.read_one_unchecked(candidates.handle.clone());
        eprintln!(
            "ann_debug init_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    let mut candidate_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        client.empty(distance_bytes),
        DType::F32,
    );
    let mut next_candidates = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    let mut next_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(distance_bytes),
        DType::F32,
    );
    let workspace_indices = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    let workspace_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(distance_bytes),
        DType::F32,
    );
    let mut candidate_retained = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    let mut next_retained = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, width]),
        data.client.empty(index_bytes),
        DType::I32,
    );
    let new_count = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([1]),
        data.client.empty(DType::I32.size()),
        DType::I32,
    );
    for iteration in 0..=max_iterations {
        let helper_start = debug_timing.then(std::time::Instant::now);
        if iteration > 0 {
            let (helper_count, helper_dim) = ann_independent_launch(&data.client, n_u32);
            reverse_zero::launch(
                &data.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                new_count.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            reverse_zero::launch(
                &data.client,
                helper_count.clone(),
                helper_dim,
                reverse_counts.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            reverse_count::launch(
                &data.client,
                helper_count.clone(),
                helper_dim,
                candidates.clone().into_tensor_arg(),
                reverse_counts.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                dtype_to_storage_type(DType::I32),
            );
            reverse_prefix::launch(
                &data.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                reverse_counts.clone().into_tensor_arg(),
                reverse_offsets.clone().into_tensor_arg(),
                reverse_cursors.clone().into_tensor_arg(),
                dtype_to_storage_type(DType::I32),
            );
            reverse_scatter::launch(
                &data.client,
                helper_count,
                helper_dim,
                candidates.clone().into_tensor_arg(),
                reverse_cursors.clone().into_tensor_arg(),
                reverse_sources.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                dtype_to_storage_type(DType::I32),
            );
        }
        if let Some(start) = helper_start {
            let _ = data
                .client
                .read_one_unchecked(reverse_counts.handle.clone());
            let helper_ms = if iteration > 0 {
                start.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };
            let step_start = std::time::Instant::now();
            ann_step::launch(
                &data.client,
                ann_step_count.clone(),
                ann_step_dim,
                data.clone().into_tensor_arg(),
                candidates.clone().into_tensor_arg(),
                candidate_distances.clone().into_tensor_arg(),
                candidate_retained.clone().into_tensor_arg(),
                reverse_offsets.clone().into_tensor_arg(),
                reverse_sources.clone().into_tensor_arg(),
                next_candidates.clone().into_tensor_arg(),
                next_distances.clone().into_tensor_arg(),
                workspace_indices.clone().into_tensor_arg(),
                workspace_distances.clone().into_tensor_arg(),
                next_retained.clone().into_tensor_arg(),
                new_count.clone().into_tensor_arg(),
                width_u32,
                u32::from(iteration > 0),
                cpu_merge,
                [
                    dtype_to_storage_type(data.dtype),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            let _ = data
                .client
                .read_one_unchecked(next_candidates.handle.clone());
            let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
            let readback_start = std::time::Instant::now();
            std::mem::swap(&mut candidates, &mut next_candidates);
            std::mem::swap(&mut candidate_distances, &mut next_distances);
            std::mem::swap(&mut candidate_retained, &mut next_retained);
            if iteration > 0 {
                let bytes = data.client.read_one_unchecked(new_count.handle.clone());
                let new_ids = i32::from_ne_bytes(bytes[..4].try_into().unwrap());
                let stop = ann_should_stop(new_ids, n, width);
                eprintln!(
                    "ann_debug iteration={iteration} new_count={new_ids} helper_ms={helper_ms:.3} step_ms={step_ms:.3} readback_ms={:.3} stop={stop}",
                    readback_start.elapsed().as_secs_f64() * 1000.0
                );
                if stop {
                    break;
                }
            } else {
                eprintln!(
                    "ann_debug iteration={iteration} new_count=NA helper_ms={helper_ms:.3} step_ms={step_ms:.3} readback_ms=NA stop=false"
                );
            }
        } else {
            ann_step::launch(
                &data.client,
                ann_step_count.clone(),
                ann_step_dim,
                data.clone().into_tensor_arg(),
                candidates.clone().into_tensor_arg(),
                candidate_distances.clone().into_tensor_arg(),
                candidate_retained.clone().into_tensor_arg(),
                reverse_offsets.clone().into_tensor_arg(),
                reverse_sources.clone().into_tensor_arg(),
                next_candidates.clone().into_tensor_arg(),
                next_distances.clone().into_tensor_arg(),
                workspace_indices.clone().into_tensor_arg(),
                workspace_distances.clone().into_tensor_arg(),
                next_retained.clone().into_tensor_arg(),
                new_count.clone().into_tensor_arg(),
                width_u32,
                u32::from(iteration > 0),
                cpu_merge,
                [
                    dtype_to_storage_type(data.dtype),
                    dtype_to_storage_type(DType::I32),
                ],
            );
            std::mem::swap(&mut candidates, &mut next_candidates);
            std::mem::swap(&mut candidate_distances, &mut next_distances);
            std::mem::swap(&mut candidate_retained, &mut next_retained);
            if iteration > 0 {
                let bytes = data.client.read_one_unchecked(new_count.handle.clone());
                let new_ids = i32::from_ne_bytes(bytes[..4].try_into().unwrap());
                if ann_should_stop(new_ids, n, width) {
                    break;
                }
            }
        }
    }
    let output_elements = n
        .checked_mul(k)
        .expect("ANN output dimensions overflow usize");
    let output_indices = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, k]),
        data.client.empty(
            output_elements
                .checked_mul(DType::I32.size())
                .expect("ANN output allocation overflows usize"),
        ),
        DType::I32,
    );
    let output_distances = CubeTensor::new_contiguous(
        data.client.clone(),
        data.device.clone(),
        Shape::new([n, k]),
        data.client.empty(
            output_elements
                .checked_mul(DType::F32.size())
                .expect("ANN output allocation overflows usize"),
        ),
        DType::F32,
    );
    let (helper_count, helper_dim) = ann_independent_launch(&data.client, n_u32);
    let output_start = debug_timing.then(std::time::Instant::now);
    ann_output::launch(
        &data.client,
        helper_count,
        helper_dim,
        candidates.into_tensor_arg(),
        candidate_distances.into_tensor_arg(),
        output_indices.clone().into_tensor_arg(),
        output_distances.clone().into_tensor_arg(),
        width_u32,
        u32::try_from(k).expect("ANN width exceeds CubeCL range"),
        [
            dtype_to_storage_type(DType::F32),
            dtype_to_storage_type(DType::I32),
        ],
    );
    if let Some(start) = output_start {
        let _ = data
            .client
            .read_one_unchecked(output_indices.handle.clone());
        eprintln!(
            "ann_debug output_ms={:.3}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    (output_indices, output_distances)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_cubecl::CubeDevice;

    #[cfg(feature = "cpu")]
    #[test]
    fn ann_cpu_runtime_identity_matches_pinned_capability_and_reports_topology() {
        let raw_device = CubeDevice::cpu().expect("CPU runtime is enabled for this test");
        let client = raw_device.client();
        let cores = client.properties().hardware.num_cpu_cores;
        println!(
            "ANN runtime identity: name={}, num_cpu_cores={cores:?}",
            client.name()
        );
        assert_eq!(client.name() == "cpu", cores.is_some());
        #[cfg(feature = "cuda")]
        if crate::cuda_backend::device(0).is_some() {
            let cuda_client = CubeDevice::cuda(0).expect("observed CUDA device").client();
            println!("ANN CUDA runtime identity: name={}", cuda_client.name());
            assert_ne!(cuda_client.name(), "cpu");
        }
    }

    #[test]
    fn ann_launch_geometry_uses_legal_cpu_width_and_ceil_count() {
        let limits = (8, 1, 1);
        for (rows, cores, units, expected_dim, expected_count) in
            [(67, 6, 8, 6, 12), (67, 16, 8, 8, 9), (3, 16, 8, 3, 1)]
        {
            let (cpu, count, dim) =
                ann_launch_geometry_with_limits(true, limits, units, 16, cores, rows);
            assert!(cpu);
            assert_eq!((dim.x, dim.y, dim.z), (expected_dim, 1, 1));
            assert!(matches!(count, CubeCount::Static(n, 1, 1) if n == expected_count));
        }
        let (cpu, count, dim) = ann_launch_geometry_with_limits(false, limits, 8, 67, 16, 67);
        assert!(!cpu);
        assert!(matches!(count, CubeCount::Static(n, 1, 1) if n == 67));
        assert_eq!((dim.x, dim.y, dim.z), (32, 1, 1));
        assert!(
            std::panic::catch_unwind(|| {
                ann_launch_geometry_with_limits(true, limits, 8, 2, 4, 67)
            })
            .is_err()
        );
    }

    #[test]
    fn ann_stop_threshold_uses_inclusive_integer_equivalent() {
        assert!(ann_should_stop(0, 1, 1));
        assert!(!ann_should_stop(1, 1, 999));
        assert!(ann_should_stop(1, 1, 1000));
        assert!(!ann_should_stop(2, 1, 1000));
        assert!(ann_should_stop(1, 1, 1999));
        assert!(!ann_should_stop(2, 1, 1999));
        assert!(ann_should_stop(524, 16_384, 32));
        assert!(!ann_should_stop(525, 16_384, 32));
    }

    #[test]
    fn ann_independent_cube_count_rejects_excess_cube_count() {
        assert_eq!(ann_independent_cube_count(67, 32, 3), 3);
        assert!(std::panic::catch_unwind(|| ann_independent_cube_count(67, 32, 2)).is_err());
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_ann_merge_preserves_integer_membership_and_lexicographic_ties() {
        let n = 19usize;
        let raw_device = CubeDevice::cpu().expect("CPU runtime is enabled for this test");
        let client = raw_device.client();
        let data = raw_tensor(
            &vec![0.0f32; n],
            Shape::new([n, 1]),
            DType::F32,
            &raw_device,
            &client,
        );
        let (indices, distances) = approximate_neighbors(data, 7, 0x5179, 4);
        let ids = read_i32(&indices);
        let distances = read_f32(&distances);

        assert_eq!(ids.len(), n * 7);
        assert_eq!(distances, vec![0.0; n * 7]);
        for row in 0..n {
            let row_ids = &ids[row * 7..(row + 1) * 7];
            assert!(row_ids.iter().all(|&id| id >= 0 && id < n as i32));
            assert!(!row_ids.contains(&(row as i32)));
            assert!(row_ids.windows(2).all(|pair| pair[0] < pair[1]));
        }
        assert_eq!(
            ids,
            read_i32(
                &approximate_neighbors(
                    raw_tensor(
                        &vec![0.0f32; n],
                        Shape::new([n, 1]),
                        DType::F32,
                        &raw_device,
                        &client,
                    ),
                    7,
                    0x5179,
                    4,
                )
                .0
            )
        );
    }

    fn raw_tensor<T: burn::tensor::Element>(
        data: &[T],
        shape: Shape,
        dtype: DType,
        device: &CubeDevice,
        client: &cubecl::client::Client,
    ) -> CubeTensor {
        let tensor_data = burn::tensor::TensorData::new(data.to_vec(), shape.clone());
        assert_eq!(tensor_data.dtype(), dtype);
        let (bytes, _, _) = tensor_data.into_parts();
        let handle = client.create(bytes);
        CubeTensor::new_contiguous(client.clone(), device.clone(), shape, handle, dtype)
    }

    fn read_i32(tensor: &CubeTensor) -> Vec<i32> {
        let bytes = tensor.client.read_one_unchecked(tensor.handle.clone());
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn read_f32(tensor: &CubeTensor) -> Vec<f32> {
        let bytes = tensor.client.read_one_unchecked(tensor.handle.clone());
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cooperative_preflight_checks_plane_and_dtype_aware_shared_memory() {
        let raw_device = CubeDevice::cpu().expect("CPU runtime is enabled for this test");
        let mut hardware = raw_device.client().properties().hardware.clone();
        hardware.plane_size_min = 32;
        hardware.plane_size_max = 64;
        hardware.max_units_per_cube = 32;
        hardware.max_cube_dim.0 = 32;
        hardware.max_cube_count.0 = 16;
        hardware.max_shared_memory_size = 516;
        let mut features = cubecl::features::Features::default();
        features.plane.insert(cubecl::features::Plane::Ops);
        validate_cooperative_ann_launch(&hardware, &features, DType::F32.size(), 16);
        validate_cooperative_ann_launch(&hardware, &features, DType::F64.size(), 16);
        let mut wide_plane = hardware.clone();
        wide_plane.plane_size_min = 64;
        validate_cooperative_ann_launch(&wide_plane, &features, DType::F64.size(), 16);
        let mut dtype_boundary = hardware.clone();
        dtype_boundary.max_shared_memory_size = 388;
        validate_cooperative_ann_launch(&dtype_boundary, &features, DType::F32.size(), 16);
        dtype_boundary.max_shared_memory_size = 515;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&dtype_boundary, &features, DType::F64.size(), 16)
            })
            .is_err()
        );

        let mut no_plane_ops = features.clone();
        no_plane_ops.plane.clear();
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&hardware, &no_plane_ops, 4, 16)
            })
            .is_err()
        );
        let mut narrow_plane = hardware.clone();
        narrow_plane.plane_size_min = 31;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&narrow_plane, &features, 4, 16)
            })
            .is_err()
        );
        let mut small_cube = hardware.clone();
        small_cube.max_units_per_cube = 31;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&small_cube, &features, 4, 16)
            })
            .is_err()
        );
        let mut small_axis = hardware.clone();
        small_axis.max_cube_dim.0 = 31;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&small_axis, &features, 4, 16)
            })
            .is_err()
        );
        let mut little_shared = hardware.clone();
        little_shared.max_shared_memory_size = 387;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&little_shared, &features, DType::F32.size(), 16)
            })
            .is_err()
        );
        let mut few_cubes = hardware;
        few_cubes.max_cube_count.0 = 15;
        assert!(
            std::panic::catch_unwind(|| {
                validate_cooperative_ann_launch(&few_cubes, &features, 4, 16)
            })
            .is_err()
        );
    }

    #[cfg(any(feature = "cpu", feature = "cuda"))]
    fn old_rescan_step(
        points: &[f32],
        candidates: &[i32],
        n: usize,
        features: usize,
        width: usize,
    ) -> (Vec<i32>, Vec<f32>) {
        let mut ids = vec![n as i32; n * width];
        let mut distances = vec![f32::MAX; n * width];
        for row in 0..n {
            let mut consider_candidate = |candidate: i32| {
                if candidate < 0 || candidate as usize >= n || candidate as usize == row {
                    return;
                }
                let candidate = candidate as usize;
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta =
                        points[row * features + feature] - points[candidate * features + feature];
                    squared += delta * delta;
                }
                let base = row * width;
                let mut position = width;
                let mut duplicate = false;
                for slot in 0..width {
                    let old = ids[base + slot];
                    duplicate |= old == candidate as i32;
                    if position == width
                        && (squared < distances[base + slot]
                            || (squared == distances[base + slot] && candidate < old as usize))
                    {
                        position = slot;
                    }
                }
                if !duplicate && position < width {
                    for slot in (position + 1..width).rev() {
                        ids[base + slot] = ids[base + slot - 1];
                        distances[base + slot] = distances[base + slot - 1];
                    }
                    ids[base + position] = candidate as i32;
                    distances[base + position] = squared;
                }
            };
            for slot in 0..width {
                consider_candidate(candidates[row * width + slot]);
            }
            for slot in 0..width {
                let neighbor = candidates[row * width + slot];
                if neighbor >= 0 && (neighbor as usize) < n {
                    for other in 0..width {
                        consider_candidate(candidates[neighbor as usize * width + other]);
                    }
                }
            }
            for source in 0..n {
                if (0..width).any(|slot| candidates[source * width + slot] == row as i32) {
                    for slot in 0..width {
                        consider_candidate(candidates[source * width + slot]);
                    }
                }
            }
        }
        (ids, distances)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers_on_device(
            CubeDevice::cuda(0).expect("CUDA device was just detected"),
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers_cpu() {
        reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers_on_device(
            CubeDevice::cpu().expect("CPU runtime is enabled for this test"),
        );
    }

    #[cfg(any(feature = "cpu", feature = "cuda"))]
    fn reverse_csr_matches_host_gate_and_is_complete_across_reused_buffers_on_device(
        raw_device: CubeDevice,
    ) {
        let client = raw_device.client();
        let n = 67usize;
        let width = 13usize;
        let elements = n * width;
        let count_bytes = n * DType::I32.size();
        let offset_bytes = (n + 1) * DType::I32.size();
        let counts = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let offsets = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n + 1]),
            client.empty(offset_bytes),
            DType::I32,
        );
        let cursors = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let sources = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([elements]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let mut state = 0x5eed_1234u32;
        let mut matrices = Vec::new();
        let mut hub = vec![-1i32; elements];
        for source in 0..n {
            hub[source * width] = 0;
            hub[source * width + 1] = 0;
            hub[source * width + 2] = (source % n) as i32;
            hub[source * width + 3] = n as i32;
            hub[source * width + 4] = (n + 9) as i32;
            hub[source * width + 5] = -2;
        }
        matrices.push(hub);
        let mut random = vec![0i32; elements];
        for value in &mut random {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *value = match state % 8 {
                0 => -1,
                1 => -2,
                2 => n as i32,
                3 => (n + 17) as i32,
                _ => ((state >> 8) as usize % n) as i32,
            };
        }
        matrices.push(random);
        let mut changed_support = vec![-1i32; elements];
        for source in 0..n {
            for slot in 0..width {
                if (source + slot) % 7 == 0 {
                    changed_support[source * width + slot] =
                        ((source * 11 + slot * 3 + 1) % n) as i32;
                }
            }
        }
        matrices.push(changed_support);

        let launch_dtype = dtype_to_storage_type(DType::I32);
        let n_u32 = u32::try_from(n).unwrap();
        let width_u32 = u32::try_from(width).unwrap();
        let (helper_count, helper_dim) = ann_independent_launch(&client, n_u32);
        for (iteration, matrix) in matrices.into_iter().enumerate() {
            let candidates = raw_tensor(
                &matrix,
                Shape::new([elements]),
                DType::I32,
                &raw_device,
                &client,
            );
            reverse_zero::launch(
                &client,
                helper_count.clone(),
                helper_dim,
                counts.clone().into_tensor_arg(),
                launch_dtype,
            );
            reverse_count::launch(
                &client,
                helper_count.clone(),
                helper_dim,
                candidates.clone().into_tensor_arg(),
                counts.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                launch_dtype,
            );
            reverse_prefix::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                counts.clone().into_tensor_arg(),
                offsets.clone().into_tensor_arg(),
                cursors.clone().into_tensor_arg(),
                launch_dtype,
            );
            reverse_scatter::launch(
                &client,
                helper_count.clone(),
                helper_dim,
                candidates.into_tensor_arg(),
                cursors.clone().into_tensor_arg(),
                sources.clone().into_tensor_arg(),
                n_u32,
                width_u32,
                launch_dtype,
            );

            let count_values = read_i32(&counts);
            let offset_values = read_i32(&offsets);
            let cursor_values = read_i32(&cursors);
            let source_values = read_i32(&sources);
            let mut expected = vec![Vec::<i32>::new(); n];
            for source in 0..n {
                for slot in 0..width {
                    let target = matrix[source * width + slot];
                    if target >= 0
                        && (target as usize) < n
                        && !matrix[source * width..source * width + slot].contains(&target)
                    {
                        expected[target as usize].push(source as i32);
                    }
                }
            }
            let count_total: usize = count_values.iter().map(|&count| count as usize).sum();
            assert_eq!(
                count_total, offset_values[n] as usize,
                "prefix total, iteration {iteration}"
            );
            for target in 0..n {
                let start = offset_values[target] as usize;
                let end = offset_values[target + 1] as usize;
                assert_eq!(
                    count_values[target] as usize,
                    end - start,
                    "count, target {target}, iteration {iteration}"
                );
                assert_eq!(
                    cursor_values[target],
                    offset_values[target + 1],
                    "cursor completion, target {target}, iteration {iteration}"
                );
                let mut actual = source_values[start..end].to_vec();
                actual.sort_unstable();
                expected[target].sort_unstable();
                assert_eq!(
                    actual, expected[target],
                    "source set, target {target}, iteration {iteration}"
                );
            }
            if iteration == 0 {
                assert_eq!(
                    count_values[0], n as i32,
                    "fixture must exercise uncapped hub indegree"
                );
            }
        }
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates_cpu() {
        ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates_on_device(
            CubeDevice::cpu().expect("CPU runtime is enabled for this test"),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates_on_device(
            CubeDevice::cuda(0).expect("CUDA device was just detected"),
        );
    }

    #[cfg(any(feature = "cpu", feature = "cuda"))]
    fn ann_step_reverse_matches_old_rescan_for_injected_duplicate_and_invalid_candidates_on_device(
        raw_device: CubeDevice,
    ) {
        let n = 9usize;
        let width = 6usize;
        let features = 2usize;
        let points = (0..n)
            .flat_map(|row| [row as f32 * 0.25, ((row * 5) % 7) as f32 * 0.5])
            .collect::<Vec<_>>();
        let client = raw_device.client();
        let data = raw_tensor(
            &points,
            Shape::new([n, features]),
            DType::F32,
            &raw_device,
            &client,
        );
        let mut candidates = vec![n as i32; n * width];
        for source in 0..n {
            let base = source * width;
            candidates[base..base + width].copy_from_slice(&[
                0,
                0,
                ((source + 2) % n) as i32,
                -1,
                (n + 4) as i32,
                if source % 2 == 0 { 3 } else { -2 },
            ]);
        }
        // Make row zero an uncapped reverse hub while retaining duplicate and invalid slots.
        for source in 0..n {
            candidates[source * width] = 0;
        }
        let candidate_tensor = raw_tensor(
            &candidates,
            Shape::new([n, width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let candidate_squared = candidates
            .iter()
            .enumerate()
            .map(|(position, &candidate)| {
                let row = position / width;
                if candidate < 0 || candidate as usize >= n || candidate as usize == row {
                    return f32::MAX;
                }
                let mut squared = 0.0f32;
                for feature in 0..features {
                    let delta = points[row * features + feature]
                        - points[candidate as usize * features + feature];
                    squared += delta * delta;
                }
                squared
            })
            .collect::<Vec<_>>();
        let candidate_distance_tensor = raw_tensor(
            &candidate_squared,
            Shape::new([n, width]),
            DType::F32,
            &raw_device,
            &client,
        );
        let candidate_retained_tensor = raw_tensor(
            &vec![0i32; n * width],
            Shape::new([n, width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let elements = n * width;
        let count_bytes = n * DType::I32.size();
        let counts = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let offsets = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n + 1]),
            client.empty((n + 1) * DType::I32.size()),
            DType::I32,
        );
        let cursors = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n]),
            client.empty(count_bytes),
            DType::I32,
        );
        let sources = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([elements]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let n_u32 = u32::try_from(n).unwrap();
        let width_u32 = u32::try_from(width).unwrap();
        let (helper_count, helper_dim) = ann_independent_launch(&client, n_u32);
        let dtype = dtype_to_storage_type(DType::I32);
        reverse_zero::launch(
            &client,
            helper_count.clone(),
            helper_dim,
            counts.clone().into_tensor_arg(),
            dtype,
        );
        reverse_count::launch(
            &client,
            helper_count.clone(),
            helper_dim,
            candidate_tensor.clone().into_tensor_arg(),
            counts.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype,
        );
        reverse_prefix::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            counts.clone().into_tensor_arg(),
            offsets.clone().into_tensor_arg(),
            cursors.clone().into_tensor_arg(),
            dtype,
        );
        reverse_scatter::launch(
            &client,
            helper_count.clone(),
            helper_dim,
            candidate_tensor.clone().into_tensor_arg(),
            cursors.clone().into_tensor_arg(),
            sources.clone().into_tensor_arg(),
            n_u32,
            width_u32,
            dtype,
        );
        let output_ids = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let output_distances = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::F32.size()),
            DType::F32,
        );
        let workspace_ids = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let workspace_distances = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::F32.size()),
            DType::F32,
        );
        let output_retained = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(elements * DType::I32.size()),
            DType::I32,
        );
        let new_count = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([1]),
            client.empty(DType::I32.size()),
            DType::I32,
        );
        reverse_zero::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            new_count.clone().into_tensor_arg(),
            dtype,
        );
        let (cpu_merge, step_count, step_dim) =
            ann_launch_geometry(&client, data.dtype.size(), n_u32);
        ann_step::launch(
            &client,
            step_count,
            step_dim,
            data.clone().into_tensor_arg(),
            candidate_tensor.into_tensor_arg(),
            candidate_distance_tensor.into_tensor_arg(),
            candidate_retained_tensor.into_tensor_arg(),
            offsets.into_tensor_arg(),
            sources.into_tensor_arg(),
            output_ids.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            workspace_ids.into_tensor_arg(),
            workspace_distances.into_tensor_arg(),
            output_retained.clone().into_tensor_arg(),
            new_count.clone().into_tensor_arg(),
            width_u32,
            1,
            cpu_merge,
            [dtype_to_storage_type(DType::F32), dtype],
        );
        let actual_ids = read_i32(&output_ids);
        let actual_distances = read_f32(&output_distances);
        let mut expected_new_count = 0i32;
        let mut expected_retained = vec![0i32; elements];
        for row in 0..n {
            let input = &candidates[row * width..(row + 1) * width];
            for slot in 0..width {
                let output = actual_ids[row * width + slot];
                if output >= 0 && (output as usize) < n {
                    if input.contains(&output) {
                        expected_retained[row * width + slot] = 1;
                    } else {
                        expected_new_count += 1;
                    }
                }
            }
        }
        assert_eq!(
            read_i32(&new_count),
            [expected_new_count],
            "expansion count is valid output membership absent from the input row"
        );
        assert_eq!(read_i32(&output_retained), expected_retained);
        let expected = old_rescan_step(&points, &candidates, n, features, width);
        assert_eq!(actual_ids, expected.0);
        assert_eq!(
            actual_distances
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .1
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn ann_step_evaluates_both_mixed_retention_forward_pairs_cpu() {
        ann_step_evaluates_both_mixed_retention_forward_pairs_on_device(
            CubeDevice::cpu().expect("CPU runtime is enabled for this test"),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ann_step_evaluates_both_mixed_retention_forward_pairs_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        ann_step_evaluates_both_mixed_retention_forward_pairs_on_device(
            CubeDevice::cuda(0).expect("CUDA device was just detected"),
        );
    }

    #[cfg(any(feature = "cpu", feature = "cuda"))]
    fn ann_step_evaluates_both_mixed_retention_forward_pairs_on_device(raw_device: CubeDevice) {
        let n = 7usize;
        let width = 2usize;
        let features = 1usize;
        let n_u32 = u32::try_from(n).unwrap();
        let points = [0.0f32, 10.0, 11.0, 0.1, 50.0, -0.1, 60.0];
        let mut candidates = vec![n as i32; n * width];
        candidates[0..2].copy_from_slice(&[1, 2]);
        candidates[2..4].copy_from_slice(&[3, 4]);
        candidates[4..6].copy_from_slice(&[5, 6]);
        let mut retained = vec![0i32; n * width];
        retained[0..2].copy_from_slice(&[1, 0]); // retained s=1, new s=2
        retained[2..4].copy_from_slice(&[0, 1]); // new c=3, retained c=4
        retained[4..6].copy_from_slice(&[1, 0]); // retained c=5, new c=6

        let client = raw_device.client();
        let data = raw_tensor(
            &points,
            Shape::new([n, features]),
            DType::F32,
            &raw_device,
            &client,
        );
        let candidate_tensor = raw_tensor(
            &candidates,
            Shape::new([n, width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let candidate_squared = candidates
            .iter()
            .enumerate()
            .map(|(position, &candidate)| {
                let row = position / width;
                if candidate < 0 || candidate as usize >= n || candidate as usize == row {
                    return f32::MAX;
                }
                let delta = points[row] - points[candidate as usize];
                delta * delta
            })
            .collect::<Vec<_>>();
        let candidate_distances = raw_tensor(
            &candidate_squared,
            Shape::new([n, width]),
            DType::F32,
            &raw_device,
            &client,
        );
        let retained_tensor = raw_tensor(
            &retained,
            Shape::new([n, width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let offsets = raw_tensor(
            &vec![0i32; n + 1],
            Shape::new([n + 1]),
            DType::I32,
            &raw_device,
            &client,
        );
        let sources = raw_tensor(
            &vec![0i32; n * width],
            Shape::new([n * width]),
            DType::I32,
            &raw_device,
            &client,
        );
        let output_ids = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(n * width * DType::I32.size()),
            DType::I32,
        );
        let output_distances = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(n * width * DType::F32.size()),
            DType::F32,
        );
        let workspace_ids = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(n * width * DType::I32.size()),
            DType::I32,
        );
        let workspace_distances = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(n * width * DType::F32.size()),
            DType::F32,
        );
        let output_retained = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([n, width]),
            client.empty(n * width * DType::I32.size()),
            DType::I32,
        );
        let new_count = CubeTensor::new_contiguous(
            client.clone(),
            raw_device.clone(),
            Shape::new([1]),
            client.empty(DType::I32.size()),
            DType::I32,
        );
        let dtype = dtype_to_storage_type(DType::I32);
        reverse_zero::launch(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            new_count.clone().into_tensor_arg(),
            dtype,
        );
        let (cpu_merge, step_count, step_dim) =
            ann_launch_geometry(&client, data.dtype.size(), n_u32);
        ann_step::launch(
            &client,
            step_count,
            step_dim,
            data.into_tensor_arg(),
            candidate_tensor.into_tensor_arg(),
            candidate_distances.into_tensor_arg(),
            retained_tensor.into_tensor_arg(),
            offsets.into_tensor_arg(),
            sources.into_tensor_arg(),
            output_ids.clone().into_tensor_arg(),
            output_distances.clone().into_tensor_arg(),
            workspace_ids.into_tensor_arg(),
            workspace_distances.into_tensor_arg(),
            output_retained.clone().into_tensor_arg(),
            new_count.clone().into_tensor_arg(),
            u32::try_from(width).unwrap(),
            1,
            cpu_merge,
            [dtype_to_storage_type(DType::F32), dtype],
        );

        // Full own-list-first join for row zero: [1,2] plus row 1's [3,4]
        // and row 2's [5,6]. Both mixed retained/new directions are needed
        // to produce the two closest results, 3 and 5.
        let mut proposals = candidates[..width].to_vec();
        for neighbor in [1usize, 2] {
            proposals.extend_from_slice(&candidates[neighbor * width..(neighbor + 1) * width]);
        }
        let mut expected = proposals
            .into_iter()
            .map(|candidate| {
                let delta = points[0] - points[candidate as usize];
                (delta * delta, candidate)
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| left.0.total_cmp(&right.0).then(left.1.cmp(&right.1)));
        let actual_ids = read_i32(&output_ids);
        let actual_distances = read_f32(&output_distances);
        assert_eq!(
            &actual_ids[..width],
            &expected
                .iter()
                .take(width)
                .map(|entry| entry.1)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual_distances[..width]
                .iter()
                .map(|distance| distance.to_bits())
                .collect::<Vec<_>>(),
            expected
                .iter()
                .take(width)
                .map(|entry| entry.0.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(read_i32(&new_count), [2]);
        assert_eq!(&read_i32(&output_retained)[..width], &[0, 0]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn ann_step_device_counter_locks_inclusive_stop_boundary_cpu() {
        ann_step_device_counter_locks_inclusive_stop_boundary_on_device(
            CubeDevice::cpu().expect("CPU runtime is enabled for this test"),
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn ann_step_device_counter_locks_inclusive_stop_boundary_cuda() {
        if crate::cuda_backend::device(0).is_none() {
            assert!(
                std::env::var_os("FAST_UMAP_REQUIRE_CUDA").is_none(),
                "CUDA is required"
            );
            return;
        }
        ann_step_device_counter_locks_inclusive_stop_boundary_on_device(
            CubeDevice::cuda(0).expect("CUDA device was just detected"),
        );
    }

    #[cfg(any(feature = "cpu", feature = "cuda"))]
    fn ann_step_device_counter_locks_inclusive_stop_boundary_on_device(raw_device: CubeDevice) {
        let n = 16_384usize;
        let width = 32usize;
        let elements = n * width;
        let affected_ids = (0..width as i32).collect::<Vec<_>>();
        let mut affected_retained = vec![1i32; width];
        affected_retained[0] = 0;
        let points = vec![0.0f32; n];
        let mut candidates = vec![0i32; elements];
        let mut retained = vec![1i32; elements];
        for row in 0..n {
            let base = row * width;
            let mut slot = 0;
            for id in 0..=width {
                if id != row {
                    candidates[base + slot] = id as i32;
                    slot += 1;
                    if slot == width {
                        break;
                    }
                }
            }
        }
        // The forward source proposes zero to each affected row. The newly
        // inserted zero displaces the largest old ID, 32, from the full row.
        candidates[width..2 * width].copy_from_slice(&[
            0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ]);

        let client = raw_device.client();
        let n_u32 = u32::try_from(n).unwrap();
        let data = raw_tensor(
            &points,
            Shape::new([n, 1]),
            DType::F32,
            &raw_device,
            &client,
        );
        let candidate_distances = raw_tensor(
            &vec![0.0f32; elements],
            Shape::new([n, width]),
            DType::F32,
            &raw_device,
            &client,
        );
        let offsets = raw_tensor(
            &vec![0i32; n + 1],
            Shape::new([n + 1]),
            DType::I32,
            &raw_device,
            &client,
        );
        let reverse_sources =
            raw_tensor(&[0i32], Shape::new([1]), DType::I32, &raw_device, &client);
        let dtype = dtype_to_storage_type(DType::I32);
        let width_u32 = u32::try_from(width).unwrap();
        for target_new in [524usize, 525] {
            let affected_start = 64usize;
            let affected_end = affected_start + target_new;
            for row in affected_start..affected_end {
                let base = row * width;
                candidates[base..base + width]
                    .copy_from_slice(&(1..=width as i32).collect::<Vec<_>>());
                retained[base] = 0;
            }
            let candidate_tensor = raw_tensor(
                &candidates,
                Shape::new([n, width]),
                DType::I32,
                &raw_device,
                &client,
            );
            let candidate_retained = raw_tensor(
                &retained,
                Shape::new([n, width]),
                DType::I32,
                &raw_device,
                &client,
            );
            let output_ids = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([n, width]),
                client.empty(elements * DType::I32.size()),
                DType::I32,
            );
            let output_distances = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([n, width]),
                client.empty(elements * DType::F32.size()),
                DType::F32,
            );
            let workspace_ids = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([n, width]),
                client.empty(elements * DType::I32.size()),
                DType::I32,
            );
            let workspace_distances = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([n, width]),
                client.empty(elements * DType::F32.size()),
                DType::F32,
            );
            let output_retained = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([n, width]),
                client.empty(elements * DType::I32.size()),
                DType::I32,
            );
            let new_count = CubeTensor::new_contiguous(
                client.clone(),
                raw_device.clone(),
                Shape::new([1]),
                client.empty(DType::I32.size()),
                DType::I32,
            );

            reverse_zero::launch(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(1),
                new_count.clone().into_tensor_arg(),
                dtype,
            );
            let (cpu_merge, step_count, step_dim) =
                ann_launch_geometry(&client, data.dtype.size(), n_u32);
            ann_step::launch(
                &client,
                step_count,
                step_dim,
                data.clone().into_tensor_arg(),
                candidate_tensor.into_tensor_arg(),
                candidate_distances.clone().into_tensor_arg(),
                candidate_retained.into_tensor_arg(),
                offsets.clone().into_tensor_arg(),
                reverse_sources.clone().into_tensor_arg(),
                output_ids.clone().into_tensor_arg(),
                output_distances.into_tensor_arg(),
                workspace_ids.into_tensor_arg(),
                workspace_distances.into_tensor_arg(),
                output_retained.clone().into_tensor_arg(),
                new_count.clone().into_tensor_arg(),
                width_u32,
                1,
                cpu_merge,
                [dtype_to_storage_type(DType::F32), dtype],
            );

            let actual_new = read_i32(&new_count)[0];
            let stop = ann_should_stop(actual_new, n, width);
            println!("ANN boundary n={n} width={width} actualNEW={actual_new} stop={stop}");
            assert_eq!(actual_new, target_new as i32);
            assert_eq!(stop, target_new == 524);

            let actual_ids = read_i32(&output_ids);
            let actual_retained = read_i32(&output_retained);
            for row in 0..n {
                let base = row * width;
                if (affected_start..affected_end).contains(&row) {
                    assert_eq!(&actual_ids[base..base + width], affected_ids.as_slice());
                    assert_eq!(
                        &actual_retained[base..base + width],
                        affected_retained.as_slice()
                    );
                } else {
                    assert_eq!(
                        &actual_ids[base..base + width],
                        &candidates[base..base + width],
                        "unaffected row {row} must retain its lexicographically smallest neighbors"
                    );
                    assert!(
                        actual_retained[base..base + width]
                            .iter()
                            .all(|&flag| flag == 1)
                    );
                }
            }
            assert!(actual_retained.iter().all(|&flag| flag == 0 || flag == 1));
        }
    }
}
