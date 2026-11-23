//! Plane-cooperative pair decoding and distance staging.

use cubecl::prelude::*;

use super::ann_distance::{squared_l2_r8, squared_l2_r8_group, triangle_decode};
use super::ann_sampled::endpoint_mask;

#[cube]
fn prefix_search<P: Int>(prefix: &Tensor<P>, value: P, first: u32, last: u32) -> u32 {
    let mut low = first;
    let mut high = last + 1;
    while high - low > 1 {
        let middle = low + (high - low) / 2;
        if prefix[middle as usize] <= value {
            low = middle;
        } else {
            high = middle;
        }
    }
    low
}

#[cube]
fn plane_bounds<P: Int>(prefix: &Tensor<P>, slot_offset: u64, slots: u32) -> (u32, u32) {
    let lane = UNIT_POS_X;
    let base = CUBE_POS_X * 32;
    let last_lane = (slots - 1 - base).min(31);
    let last_index = prefix.shape(0) as u32 - 2;
    let mut first = 0u32;
    let mut last = 0u32;
    if lane == 0 {
        first = prefix_search(
            prefix,
            P::cast_from(slot_offset) + P::cast_from(base),
            0,
            last_index,
        );
    }
    if lane == last_lane {
        last = prefix_search(
            prefix,
            P::cast_from(slot_offset) + P::cast_from(base + last_lane),
            0,
            last_index,
        );
    }
    (plane_broadcast(first, 0u32), plane_shuffle(last, last_lane))
}

#[cube]
fn plane_distance<F: Float, Lanes: Size>(
    data: &Tensor<Vector<F, Lanes>>,
    left: u32,
    right: u32,
    valid: bool,
    features: u32,
    #[comptime] vector_width: usize,
    #[comptime] group: usize,
    #[comptime] binding_width: usize,
    #[comptime] unroll: usize,
) -> F {
    let lane = UNIT_POS_X;
    let mut result = F::new(0.0f32);
    if comptime![group == 1] {
        result = squared_l2_r8::<F, Lanes>(
            data,
            left as usize,
            right as usize,
            features as usize,
            vector_width,
        );
    } else {
        let groups = comptime![32 / group] as u32;
        for round in 0..comptime![group] {
            let source = round as u32 * groups + lane / comptime![group] as u32;
            let group_left = plane_shuffle(left, source);
            let group_right = plane_shuffle(right, source);
            let mut valid_value = 0u32;
            if valid {
                valid_value = 1;
            }
            let group_valid = plane_shuffle(valid_value, source) != 0;
            let value = squared_l2_r8_group::<F, Lanes>(
                data,
                group_left as usize,
                group_right as usize,
                features as usize,
                group_valid,
                group,
                binding_width,
                unroll,
            );
            let own = plane_shuffle(value, (lane % groups) * comptime![group] as u32);
            if round as u32 == lane / groups {
                result = own;
            }
        }
    }
    result
}

#[cube(launch)]
pub(super) fn forest_pair_slots_plane_kernel<F: Float, I: Int, P: Int, Lanes: Size>(
    data: &Tensor<Vector<F, Lanes>>,
    forest_rows: &Tensor<I>,
    leaf_rows: &Tensor<I>,
    pair_prefix: &Tensor<P>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    n: u32,
    features: u32,
    slot_offset: u64,
    slots: u32,
    #[comptime] vector_width: usize,
    #[comptime] group: usize,
    #[comptime] binding_width: usize,
    #[comptime] decode: bool,
    #[comptime] unroll: usize,
    #[define(F, I, P)] _dtypes: [ElemType; 3],
) {
    let slot = ABSOLUTE_POS_X;
    let active = slot < slots;
    let mut first = 0u32;
    let mut last = pair_prefix.shape(0) as u32 - 2;
    if comptime![decode] {
        let (lower, upper) = plane_bounds(pair_prefix, slot_offset, slots);
        first = lower;
        last = upper;
    }
    let mut left = 0u32;
    let mut right = 0u32;
    let mut valid = false;
    if active {
        let global = P::cast_from(slot_offset) + P::cast_from(slot);
        let leaf = prefix_search(pair_prefix, global, first, last) as usize;
        valid = global < pair_prefix[pair_prefix.shape(0) - 1];
        if valid {
            let mut local = u32::cast_from(global - pair_prefix[leaf]);
            let start = u32::cast_from(leaf_rows[leaf]);
            let length = u32::cast_from(leaf_rows[leaf + 1]) - start;
            let mut i = 0u32;
            let j: u32;
            if comptime![decode] {
                let (decoded_i, decoded_j) = triangle_decode(local, length);
                i = decoded_i;
                j = decoded_j;
            } else {
                while local >= length - i - 1 {
                    local -= length - i - 1;
                    i += 1;
                }
                j = i + 1 + local;
            }
            left = u32::cast_from(forest_rows[(start + i) as usize]);
            right = u32::cast_from(forest_rows[(start + j) as usize]);
        }
    }
    let mut load_left = 0u32;
    let mut load_right = 0u32;
    if valid {
        load_left = left;
        load_right = right;
    }
    let result = plane_distance::<F, Lanes>(
        data,
        load_left,
        load_right,
        valid,
        features,
        vector_width,
        group,
        binding_width,
        unroll,
    );
    if active {
        let mut distance = F::new(0.0f32);
        let mut mask = 0i32;
        if valid {
            distance = result;
            mask = endpoint_mask::<F, I>(
                left as usize,
                right as usize,
                distance,
                n as usize,
                row_worst_ids,
                row_worst_distances,
            );
        }
        pair_left[slot as usize] = I::cast_from(left);
        pair_right[slot as usize] = I::cast_from(right);
        pair_distances[slot as usize] = distance;
        pair_valid[slot as usize] = I::cast_from(mask);
    }
}

#[allow(clippy::manual_checked_ops)]
#[cube(launch)]
pub(super) fn pair_slots_plane_kernel<F: Float, I: Int, P: Int, Lanes: Size>(
    data: &Tensor<Vector<F, Lanes>>,
    sampled_ids: &Tensor<I>,
    pool_sizes: &Tensor<I>,
    center_offsets: &Tensor<P>,
    row_worst_ids: &Tensor<I>,
    row_worst_distances: &Tensor<F>,
    pair_left: &mut Tensor<I>,
    pair_right: &mut Tensor<I>,
    pair_distances: &mut Tensor<F>,
    pair_valid: &mut Tensor<I>,
    n: u32,
    features: u32,
    slot_offset: u64,
    slots: u32,
    capacity: u32,
    #[comptime] vector_width: usize,
    #[comptime] group: usize,
    #[comptime] binding_width: usize,
    #[comptime] decode: bool,
    #[comptime] unroll: usize,
    #[define(F, I, P)] _dtypes: [ElemType; 3],
) {
    let slot = ABSOLUTE_POS_X;
    let active = slot < slots;
    let mut first = 0u32;
    let mut last = center_offsets.shape(0) as u32 - 2;
    if comptime![decode] {
        let (lower, upper) = plane_bounds(center_offsets, slot_offset, slots);
        first = lower;
        last = upper;
    }
    let mut left = n;
    let mut right = n;
    let mut valid = false;
    if active {
        let flattened = P::cast_from(slot_offset) + P::cast_from(slot);
        let center = prefix_search(center_offsets, flattened, first, last) as usize;
        let mut local = u32::cast_from(flattened - center_offsets[center]);
        let new_count = u32::cast_from(pool_sizes[center * 2]);
        let old_count = u32::cast_from(pool_sizes[center * 2 + 1]);
        let base = center * capacity as usize * 2;
        let triangle = new_count * new_count.saturating_sub(1) / 2;
        if local < triangle {
            let mut i = 0u32;
            let mut j = 0u32;
            if comptime![decode] {
                let (decoded_i, decoded_j) = triangle_decode(local, new_count);
                i = decoded_i;
                j = decoded_j;
            } else {
                let mut done = false;
                while !done {
                    let row_len = new_count - 1 - i;
                    if local < row_len {
                        j = i + 1 + local;
                        done = true;
                    } else {
                        local -= row_len;
                        i += 1;
                    }
                }
            }
            left = u32::cast_from(sampled_ids[base + i as usize]);
            right = u32::cast_from(sampled_ids[base + j as usize]);
        } else if old_count > 0 {
            let cross = local - triangle;
            left = u32::cast_from(sampled_ids[base + (cross / old_count) as usize]);
            right = u32::cast_from(
                sampled_ids[base + capacity as usize + (cross % old_count) as usize],
            );
        }
        valid = left < n && right < n && left != right;
    }
    let mut load_left = 0u32;
    let mut load_right = 0u32;
    if valid {
        load_left = left;
        load_right = right;
    }
    let result = plane_distance::<F, Lanes>(
        data,
        load_left,
        load_right,
        valid,
        features,
        vector_width,
        group,
        binding_width,
        unroll,
    );
    if active {
        let mut distance = F::new(0.0f32);
        if valid {
            distance = result;
        }
        let mask = endpoint_mask::<F, I>(
            left as usize,
            right as usize,
            distance,
            n as usize,
            row_worst_ids,
            row_worst_distances,
        );
        pair_left[slot as usize] = I::cast_from(left);
        pair_right[slot as usize] = I::cast_from(right);
        pair_distances[slot as usize] = distance;
        pair_valid[slot as usize] = I::cast_from(mask);
    }
}
