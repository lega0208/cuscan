//! CPU owner-route join: bounded chunk-local generation followed by row-owned apply.

#[cfg(test)]
#[path = "ann_owner_tests.rs"]
mod tests;

use cubecl::prelude::*;

use super::super::ann_distance::{squared_l2_r8, squared_l2_r8_x4};
use super::super::orchestration::OWNER_D_MAX as D_MAX;
use super::{KEEP_LEFT, KEEP_RIGHT, PRODUCED, cpu_row_apply, endpoint_mask};

// The comptime forest specialization must remain separate from the runtime test.
#[allow(clippy::if_same_then_else)]
#[cube]
fn pool_candidate<I: Int>(
    sampled_ids: &Tensor<I>,
    base: usize,
    capacity: usize,
    new_count: usize,
    i: usize,
    index: usize,
    #[comptime] forest: bool,
) -> usize {
    let new_remaining = new_count - i - 1;
    let slot = if comptime![forest] {
        i + 1 + index
    } else if index < new_remaining {
        i + 1 + index
    } else {
        capacity + index - new_remaining
    };
    u32::cast_from(sampled_ids[base + slot]) as usize
}

#[cube]
fn append_pair<F: Float, I: Int>(
    left: usize,
    right: usize,
    distance: F,
    n: usize,
    rows_per_partition: usize,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    record_rows: &mut Tensor<I>,
    record_ids: &mut Tensor<I>,
    record_distances: &mut Tensor<F>,
    counts: &mut Array<u32>,
    cursor: usize,
) -> (usize, u32) {
    let mask = endpoint_mask::<F, I>(left, right, distance, n, row_worst_ids, row_worst_distances);
    let mut next = cursor;
    if mask & KEEP_LEFT != 0 {
        record_rows[next] = I::cast_from(left as u32);
        record_ids[next] = I::cast_from(right as u32);
        record_distances[next] = distance;
        let partition = left / rows_per_partition;
        counts[partition] += 1;
        next += 1;
    }
    if mask & KEEP_RIGHT != 0 {
        record_rows[next] = I::cast_from(right as u32);
        record_ids[next] = I::cast_from(left as u32);
        record_distances[next] = distance;
        let partition = right / rows_per_partition;
        counts[partition] += 1;
        next += 1;
    }
    let produced = if mask & PRODUCED != 0 { 2u32 } else { 0u32 };
    (next, produced)
}

/// The host bounds each block to its staging capacity and supplies monotone
/// center offsets. A chunk owns exactly twice its slot interval; no two chunks
/// write the same record, order, or offsets locations.
#[cube(launch)]
pub(super) fn owner_join_generate<F: Float, I: Int, P: Int, V: Size>(
    data: &Tensor<Vector<F, V>>,
    sampled_ids: &Tensor<I>,
    pool_sizes: &Tensor<I>,
    center_offsets: &Tensor<P>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    record_rows: &mut Tensor<I>,
    record_ids: &mut Tensor<I>,
    record_distances: &mut Tensor<F>,
    record_order: &mut Tensor<I>,
    chunk_offsets: &mut Tensor<I>,
    profile_counts: &mut Tensor<Atomic<u64>>,
    n: u32,
    capacity: u32,
    features: u32,
    c0: u32,
    c1: u32,
    partitions: u32,
    #[comptime] ch: usize,
    #[comptime] vector_width: usize,
    #[comptime] forest: bool,
    #[comptime] profile: bool,
    #[define(F, I, P)] _dtypes: [ElemType; 3],
) {
    let chunk = ABSOLUTE_POS_X as usize;
    let start = c0 as usize + chunk * ch;
    let end = (start + ch).min(c1 as usize);
    if start < c1 as usize {
        let n = n as usize;
        let capacity = capacity as usize;
        let partitions = partitions as usize;
        let rows_per_partition = n.div_ceil(partitions);
        // Subtract in the wide offset type before narrowing to the tile index.
        let region_start =
            2 * u32::cast_from(center_offsets[start] - center_offsets[c0 as usize]) as usize;
        let mut cursor = region_start;
        let mut produced = 0u64;
        let mut counts = Array::<u32>::new(D_MAX);
        for d in 0..D_MAX {
            counts[d] = 0;
        }
        for center in start..end {
            let (base, new_count, old_count) = if comptime![forest] {
                let first = u32::cast_from(pool_sizes[center]) as usize;
                let last = u32::cast_from(pool_sizes[center + 1]) as usize;
                (first, last - first, 0usize)
            } else {
                (
                    center * capacity * 2,
                    u32::cast_from(pool_sizes[center * 2]) as usize,
                    u32::cast_from(pool_sizes[center * 2 + 1]) as usize,
                )
            };
            for i in 0..new_count {
                let left = u32::cast_from(sampled_ids[base + i]) as usize;
                let candidates = new_count - i - 1 + old_count;
                let mut j = 0usize;
                while j + 4 <= candidates {
                    let r0 =
                        pool_candidate::<I>(sampled_ids, base, capacity, new_count, i, j, forest);
                    let r1 = pool_candidate::<I>(
                        sampled_ids,
                        base,
                        capacity,
                        new_count,
                        i,
                        j + 1,
                        forest,
                    );
                    let r2 = pool_candidate::<I>(
                        sampled_ids,
                        base,
                        capacity,
                        new_count,
                        i,
                        j + 2,
                        forest,
                    );
                    let r3 = pool_candidate::<I>(
                        sampled_ids,
                        base,
                        capacity,
                        new_count,
                        i,
                        j + 3,
                        forest,
                    );
                    // Invalid pool entries must never address data, even on checked launches.
                    if left < n {
                        let s0 = if r0 < n { r0 } else { left };
                        let s1 = if r1 < n { r1 } else { left };
                        let s2 = if r2 < n { r2 } else { left };
                        let s3 = if r3 < n { r3 } else { left };
                        let (d0, d1, d2, d3) = squared_l2_r8_x4::<F, V>(
                            data,
                            left,
                            s0,
                            s1,
                            s2,
                            s3,
                            features as usize,
                            vector_width,
                        );
                        let (next0, p0) = append_pair::<F, I>(
                            left,
                            r0,
                            d0,
                            n,
                            rows_per_partition,
                            row_worst_ids,
                            row_worst_distances,
                            record_rows,
                            record_ids,
                            record_distances,
                            &mut counts,
                            cursor,
                        );
                        let (next1, p1) = append_pair::<F, I>(
                            left,
                            r1,
                            d1,
                            n,
                            rows_per_partition,
                            row_worst_ids,
                            row_worst_distances,
                            record_rows,
                            record_ids,
                            record_distances,
                            &mut counts,
                            next0,
                        );
                        let (next2, p2) = append_pair::<F, I>(
                            left,
                            r2,
                            d2,
                            n,
                            rows_per_partition,
                            row_worst_ids,
                            row_worst_distances,
                            record_rows,
                            record_ids,
                            record_distances,
                            &mut counts,
                            next1,
                        );
                        let (next3, p3) = append_pair::<F, I>(
                            left,
                            r3,
                            d3,
                            n,
                            rows_per_partition,
                            row_worst_ids,
                            row_worst_distances,
                            record_rows,
                            record_ids,
                            record_distances,
                            &mut counts,
                            next2,
                        );
                        cursor = next3;
                        if comptime![profile] {
                            produced += (p0 + p1 + p2 + p3) as u64;
                        }
                    }
                    j += 4;
                }
                while j < candidates {
                    let right =
                        pool_candidate::<I>(sampled_ids, base, capacity, new_count, i, j, forest);
                    if left < n && right < n && left != right {
                        let distance = squared_l2_r8::<F, V>(
                            data,
                            left,
                            right,
                            features as usize,
                            vector_width,
                        );
                        let (next, p) = append_pair::<F, I>(
                            left,
                            right,
                            distance,
                            n,
                            rows_per_partition,
                            row_worst_ids,
                            row_worst_distances,
                            record_rows,
                            record_ids,
                            record_distances,
                            &mut counts,
                            cursor,
                        );
                        cursor = next;
                        if comptime![profile] {
                            produced += p as u64;
                        }
                    }
                    j += 1;
                }
            }
        }
        let offset_base = chunk * (D_MAX + 1);
        let mut prefix = region_start;
        for d in 0..partitions {
            chunk_offsets[offset_base + d] = I::cast_from(prefix as u32);
            let count = counts[d];
            counts[d] = prefix as u32;
            prefix += count as usize;
        }
        chunk_offsets[offset_base + partitions] = I::cast_from(prefix as u32);
        for record in region_start..cursor {
            let row = u32::cast_from(record_rows[record]) as usize;
            let d = row / rows_per_partition;
            let position = counts[d] as usize;
            record_order[position] = I::cast_from(record as u32);
            counts[d] += 1;
        }
        if comptime![profile] {
            profile_counts[0].fetch_add(produced);
            profile_counts[1].fetch_add((cursor - region_start) as u64);
        }
    }
}

/// Each partition exclusively owns its rows, and consumes chunk slices in order.
#[cube(launch)]
pub(super) fn owner_join_apply<F: Float, I: Int>(
    record_rows: &Tensor<I>,
    record_ids: &Tensor<I>,
    record_distances: &Tensor<F>,
    record_order: &Tensor<I>,
    chunk_offsets: &Tensor<I>,
    output_ids: &mut Tensor<I>,
    output_distances: &mut Tensor<F>,
    row_worst_ids: &mut Tensor<I>,
    row_worst_distances: &mut Tensor<F>,
    n: u32,
    width: u32,
    chunk_count: u32,
    partitions: u32,
    #[define(F, I)] _dtypes: [ElemType; 2],
) {
    let partition = ABSOLUTE_POS_X as usize;
    if partition < partitions as usize {
        for chunk in 0..chunk_count as usize {
            let base = chunk * (D_MAX + 1);
            let start = u32::cast_from(chunk_offsets[base + partition]) as usize;
            let end = u32::cast_from(chunk_offsets[base + partition + 1]) as usize;
            for position in start..end {
                let record = u32::cast_from(record_order[position]) as usize;
                let row = u32::cast_from(record_rows[record]) as usize;
                let (worst_id, worst_distance) = cpu_row_apply::<F, I>(
                    row,
                    n as usize,
                    width as usize,
                    record_ids[record],
                    record_distances[record],
                    output_ids,
                    output_distances,
                    row_worst_ids[row],
                    row_worst_distances[row],
                );
                row_worst_ids[row] = worst_id;
                row_worst_distances[row] = worst_distance;
            }
        }
    }
}
