//! Deterministic host graph primitives for recursive initialization.
//!
//! Inputs are canonical, square CSR graphs with sorted columns. Labels passed to
//! aggregation are compact non-negative indices in `0..parts`.

use crate::{HubSelection, Result, UmapError};
use rayon::prelude::*;

fn reserve<T>(count: usize) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|e| UmapError::Memory(e.to_string()))?;
    Ok(values)
}

fn filled<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut values = reserve(count)?;
    values.resize(count, value);
    Ok(values)
}

#[derive(Clone, Debug, PartialEq)]
pub struct HostCsr {
    pub n: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

impl HostCsr {
    /// Validate canonical CSR and symmetric support required by recursive admission.
    pub fn validate(&self) -> Result<()> {
        let invalid = || UmapError::InvalidInput("invalid recursive CSR graph".into());
        if self.n.checked_add(1) != Some(self.indptr.len())
            || self.indptr.first() != Some(&0)
            || self.indices.len() != self.weights.len()
            || self.indptr.last() != Some(&self.indices.len())
            || self
                .indptr
                .windows(2)
                .any(|pair| pair[0] > pair[1] || pair[1] > self.indices.len())
        {
            return Err(invalid());
        }
        for row in 0..self.n {
            let range = self.indptr[row]..self.indptr[row + 1];
            let columns = &self.indices[range.clone()];
            if columns.windows(2).any(|pair| pair[0] >= pair[1])
                || columns.iter().any(|&col| col >= self.n || col == row)
                || self.weights[range]
                    .iter()
                    .any(|&w| !w.is_finite() || w <= 0.0)
            {
                return Err(invalid());
            }
        }
        for row in 0..self.n {
            for &col in &self.indices[self.indptr[row]..self.indptr[row + 1]] {
                if self.indices[self.indptr[col]..self.indptr[col + 1]]
                    .binary_search(&row)
                    .is_err()
                {
                    return Err(invalid());
                }
            }
        }
        Ok(())
    }

    /// Build a canonical CSR from the positive pruned fit graph.
    pub fn from_coo(n: usize, heads: &[usize], tails: &[usize], weights: &[f32]) -> Result<Self> {
        if heads.len() != weights.len() || tails.len() != weights.len() {
            return Err(UmapError::InvalidInput(
                "recursive COO lengths differ".into(),
            ));
        }
        let mut entries = reserve(weights.len())?;
        for (ordinal, ((&head, &tail), &weight)) in heads.iter().zip(tails).zip(weights).enumerate()
        {
            if head >= n || tail >= n || head == tail || !weight.is_finite() || weight <= 0.0 {
                return Err(UmapError::InvalidInput(
                    "invalid recursive COO entry".into(),
                ));
            }
            entries.push(ordinal);
        }
        entries.sort_unstable_by_key(|&ordinal| (heads[ordinal], tails[ordinal], ordinal));
        let offsets = n
            .checked_add(1)
            .ok_or(UmapError::Overflow("recursive CSR offsets"))?;
        let mut csr = Self {
            n,
            indptr: reserve(offsets)?,
            indices: reserve(weights.len())?,
            weights: reserve(weights.len())?,
        };
        csr.indptr.resize(offsets, 0);
        let mut previous = None;
        for ordinal in entries {
            let (head, tail, weight) = (heads[ordinal], tails[ordinal], weights[ordinal]);
            if previous == Some((head, tail)) {
                let last = csr.weights.last_mut().expect("previous entry exists");
                *last += weight;
                if !last.is_finite() {
                    return Err(UmapError::Numerical(
                        "recursive COO duplicate sum is non-finite".into(),
                    ));
                }
                continue;
            }
            previous = Some((head, tail));
            csr.indptr[head + 1] += 1;
            csr.indices.push(tail);
            csr.weights.push(weight);
        }
        for row in 0..n {
            csr.indptr[row + 1] += csr.indptr[row];
        }
        Ok(csr)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SummedCsr {
    pub n: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<usize>,
    pub weights: Vec<f64>,
}

// SplitMix64 provides independent, reproducible per-index fallback values.
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

pub fn initialize_labels(
    graph: &HostCsr,
    parts: usize,
    method: HubSelection,
    seed: u64,
) -> Result<Vec<i32>> {
    let n = graph.n;
    assert!(parts <= n && parts <= i32::MAX as usize);
    let mut labels = filled(n, -1)?;
    if parts == 0 {
        return Ok(labels);
    }
    if matches!(method, HubSelection::Random) {
        for label in 0..parts {
            labels[(mix(seed.wrapping_add(label as u64)) % n as u64) as usize] = label as i32;
        }
        return Ok(labels);
    }
    let mut degrees = reserve(n)?;
    for row in 0..n {
        let range = graph.indptr[row]..graph.indptr[row + 1];
        degrees.push(graph.weights[range].iter().copied().sum::<f32>());
    }
    let mut order = reserve(n)?;
    order.extend(0..n);
    order.sort_unstable_by(|&a, &b| degrees[a].total_cmp(&degrees[b]).then(a.cmp(&b)));
    if matches!(method, HubSelection::Degree) {
        for (label, &row) in order[n - parts..].iter().enumerate() {
            labels[row] = label as i32;
        }
    } else {
        let mut blocked = filled(n, false)?;
        let mut next = 0;
        for &row in order.iter().rev() {
            if !blocked[row] {
                labels[row] = next as i32;
                next += 1;
                blocked[row] = true;
                for &neighbor in &graph.indices[graph.indptr[row]..graph.indptr[row + 1]] {
                    blocked[neighbor] = true;
                }
                if next == parts {
                    return Ok(labels);
                }
            }
        }
        for &row in order.iter().rev() {
            if labels[row] < 0 {
                labels[row] = next as i32;
                next += 1;
                if next == parts {
                    break;
                }
            }
        }
    }
    Ok(labels)
}

/// Jacobi propagation. `share` selects the deeper-level 10% rule.
pub fn propagate_labels(graph: &HostCsr, labels: &mut [i32], share: bool) -> Result<usize> {
    let mut previous = labels.iter().filter(|&&x| x < 0).count();
    let mut next = reserve(labels.len())?;
    next.extend_from_slice(labels);
    for sweep in 0..100 {
        next.par_iter_mut()
            .enumerate()
            .try_for_each(|(row, output)| -> Result<()> {
                if labels[row] >= 0 {
                    return Ok(());
                }
                let mut votes = reserve(graph.indptr[row + 1] - graph.indptr[row])?;
                let mut total = 0.0f64;
                for edge in graph.indptr[row]..graph.indptr[row + 1] {
                    let label = labels[graph.indices[edge]];
                    let weight = graph.weights[edge];
                    total += weight as f64;
                    if label >= 0 {
                        votes.push((label, edge, weight));
                    }
                }
                votes.sort_unstable_by_key(|vote| (vote.0, vote.1));
                let mut grouped = reserve(votes.len())?;
                let mut i = 0;
                while i < votes.len() {
                    let (label, first, mut sum) = votes[i];
                    i += 1;
                    while i < votes.len() && votes[i].0 == label {
                        sum += votes[i].2;
                        i += 1;
                    }
                    grouped.push((first, label, sum));
                }
                grouped.sort_unstable_by_key(|vote| vote.0);
                let mut maximum = if share { 0.1 * total } else { 1.0 };
                for (_, label, vote) in grouped {
                    if vote as f64 >= maximum {
                        maximum = vote as f64;
                        *output = label;
                    }
                }
                Ok(())
            })?;
        labels.copy_from_slice(&next);
        if sweep % 5 == 0 {
            let remaining = labels.iter().filter(|&&x| x < 0).count();
            if remaining == 0 || remaining == previous {
                return Ok(sweep + 1);
            }
            previous = remaining;
        }
    }
    Ok(100)
}

pub fn label_outliers(graph: &HostCsr, labels: &mut [i32], seed: u64) -> Result<()> {
    let max_degree = graph
        .indptr
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .max()
        .unwrap_or(0);
    let capacity = max_degree
        .checked_mul(100)
        .and_then(|n| n.checked_add(1))
        .ok_or(UmapError::Overflow("recursive outlier stack"))?;
    let count = labels
        .iter()
        .copied()
        .max()
        .unwrap_or(-1)
        .saturating_add(1)
        .max(1) as u64;
    for row in 0..labels.len() {
        if labels[row] >= 0 {
            continue;
        }
        let mut stack = reserve(capacity)?;
        stack.push(row);
        for _ in 0..100 {
            let Some(node) = stack.pop() else { break };
            for &neighbor in &graph.indices[graph.indptr[node]..graph.indptr[node + 1]] {
                if labels[neighbor] >= 0 {
                    labels[row] = labels[neighbor];
                    break;
                }
                stack.push(neighbor);
            }
            if labels[row] >= 0 {
                break;
            }
        }
        if labels[row] < 0 {
            labels[row] = (mix(seed.wrapping_add(row as u64)) % count) as i32;
        }
    }
    Ok(())
}

pub fn remap_labels(labels: &mut [i32]) -> Result<usize> {
    let mut unique = reserve(labels.len())?;
    unique.extend(labels.iter().copied().filter(|&x| x >= 0));
    unique.sort_unstable();
    unique.dedup();
    let mut next = unique.len();
    for label in labels {
        if *label < 0 {
            *label = next as i32;
            next += 1;
        } else {
            *label = unique.binary_search(label).unwrap() as i32;
        }
    }
    Ok(next)
}

fn grouped_members(labels: &[i32], parts: usize) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut starts = filled(
        parts
            .checked_add(1)
            .ok_or(UmapError::Overflow("recursive parts"))?,
        0usize,
    )?;
    for &label in labels {
        starts[label as usize + 1] += 1;
    }
    for i in 0..parts {
        starts[i + 1] += starts[i];
    }
    let mut cursor = reserve(parts)?;
    cursor.extend_from_slice(&starts[..parts]);
    let mut members = filled(labels.len(), 0usize)?;
    for (row, &label) in labels.iter().enumerate() {
        let part = label as usize;
        members[cursor[part]] = row;
        cursor[part] += 1;
    }
    Ok((starts, members))
}

/// Count the distinct coarse targets in each part before allocating output CSR.
pub fn count_coarse_edges(
    graph: &HostCsr,
    summed: Option<&SummedCsr>,
    labels: &[i32],
    parts: usize,
) -> Result<Vec<usize>> {
    let (indptr, indices) = match summed {
        Some(s) => (&s.indptr, &s.indices),
        None => (&graph.indptr, &graph.indices),
    };
    let (starts, members) = grouped_members(labels, parts)?;
    let mut counts = filled(parts, 0usize)?;
    counts
        .par_iter_mut()
        .enumerate()
        .try_for_each(|(part, count)| -> Result<()> {
            let degree =
                members[starts[part]..starts[part + 1]]
                    .iter()
                    .try_fold(0usize, |sum, &row| {
                        sum.checked_add(indptr[row + 1] - indptr[row])
                            .ok_or(UmapError::Overflow("recursive part degree"))
                    })?;
            let mut entries = reserve(degree)?;
            for &row in &members[starts[part]..starts[part + 1]] {
                for edge in indptr[row]..indptr[row + 1] {
                    entries.push(labels[indices[edge]]);
                }
            }
            entries.sort_unstable();
            entries.dedup();
            *count = entries.len();
            Ok(())
        })?;
    Ok(counts)
}

/// Aggregate the previous summed graph, or the original f32 graph at level one.
/// Rows are grouped in input order and contributions stably sorted by target.
pub fn aggregate_graph(
    graph: &HostCsr,
    summed: Option<&SummedCsr>,
    labels: &[i32],
    parts: usize,
) -> Result<SummedCsr> {
    let counts = count_coarse_edges(graph, summed, labels, parts)?;
    aggregate_graph_counted(graph, summed, labels, parts, &counts)
}

pub(crate) fn aggregate_graph_counted(
    graph: &HostCsr,
    summed: Option<&SummedCsr>,
    labels: &[i32],
    parts: usize,
    counts: &[usize],
) -> Result<SummedCsr> {
    let (offsets, indices) = match summed {
        Some(s) => (&s.indptr, &s.indices),
        None => (&graph.indptr, &graph.indices),
    };
    let (starts, members) = grouped_members(labels, parts)?;
    let mut rows: Vec<Vec<(usize, f64)>> = filled(parts, Vec::new())?;
    rows.par_iter_mut()
        .enumerate()
        .try_for_each(|(part, output)| -> Result<()> {
            let degree =
                members[starts[part]..starts[part + 1]]
                    .iter()
                    .try_fold(0usize, |sum, &row| {
                        sum.checked_add(offsets[row + 1] - offsets[row])
                            .ok_or(UmapError::Overflow("recursive part degree"))
                    })?;
            let mut entries = reserve(degree)?;
            for &row in &members[starts[part]..starts[part + 1]] {
                for edge in offsets[row]..offsets[row + 1] {
                    let weight = match summed {
                        Some(s) => s.weights[edge],
                        None => graph.weights[edge] as f64,
                    };
                    entries.push((labels[indices[edge]] as usize, edge, weight));
                }
            }
            entries.sort_unstable_by_key(|entry| (entry.0, entry.1));
            output
                .try_reserve_exact(counts[part])
                .map_err(|e| UmapError::Memory(e.to_string()))?;
            for (target, _, weight) in entries {
                if let Some((_, sum)) = output.last_mut().filter(|entry| entry.0 == target) {
                    *sum += weight;
                } else {
                    output.push((target, weight));
                }
            }
            Ok(())
        })?;
    let total = rows.iter().try_fold(0usize, |sum, row| {
        sum.checked_add(row.len())
            .ok_or(UmapError::Overflow("recursive coarse edges"))
    })?;
    let mut result = SummedCsr {
        n: parts,
        indptr: reserve(
            parts
                .checked_add(1)
                .ok_or(UmapError::Overflow("recursive parts"))?,
        )?,
        indices: reserve(total)?,
        weights: reserve(total)?,
    };
    result.indptr.push(0);
    for row in rows {
        for (target, weight) in row {
            result.indices.push(target);
            result.weights.push(weight);
        }
        result.indptr.push(result.indices.len());
    }
    Ok(result)
}

/// Drop internal weights, normalize directed rows, then take the fuzzy union.
pub fn transition_graph(summed: &SummedCsr) -> Result<HostCsr> {
    let n = summed.n;
    let mut totals = filled(n, 0.0f64)?;
    totals.par_iter_mut().enumerate().for_each(|(row, total)| {
        *total = (summed.indptr[row]..summed.indptr[row + 1])
            .filter(|&edge| summed.indices[edge] != row)
            .map(|edge| summed.weights[edge])
            .sum();
    });
    let capacity = summed
        .indices
        .len()
        .checked_mul(2)
        .ok_or(UmapError::Overflow("recursive transition support"))?;
    let mut support = reserve(capacity)?;
    for row in 0..n {
        let range = summed.indptr[row]..summed.indptr[row + 1];
        if totals[row] <= 0.0 {
            continue;
        }
        for edge in range {
            let col = summed.indices[edge];
            let weight = summed.weights[edge];
            if col == row || weight == 0.0 {
                continue;
            }
            support.push((row, col));
            support.push((col, row));
        }
    }
    support.sort_unstable();
    support.dedup();
    let mut weights = filled(support.len(), 0.0f32)?;
    weights
        .par_iter_mut()
        .zip(support.par_iter())
        .for_each(|(output, &(row, col))| {
            let directed = |from: usize, to: usize| {
                if totals[from] <= 0.0 {
                    return 0.0;
                }
                let start = summed.indptr[from];
                let end = summed.indptr[from + 1];
                match summed.indices[start..end].binary_search(&to) {
                    Ok(offset) => summed.weights[start + offset] / totals[from],
                    Err(_) => 0.0,
                }
            };
            let forward = directed(row, col);
            let reverse = directed(col, row);
            *output = (forward + reverse - forward * reverse).clamp(0.0, 1.0) as f32;
        });
    let mut result = HostCsr {
        n,
        indptr: filled(
            n.checked_add(1)
                .ok_or(UmapError::Overflow("recursive CSR offsets"))?,
            0usize,
        )?,
        indices: reserve(support.len())?,
        weights: reserve(support.len())?,
    };
    for ((row, col), weight) in support.into_iter().zip(weights) {
        if weight != 0.0 {
            result.indptr[row + 1] += 1;
            result.indices.push(col);
            result.weights.push(weight);
        }
    }
    for row in 0..n {
        result.indptr[row + 1] += result.indptr[row];
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coo_merges_duplicates_in_input_order_and_rejects_invalid_entries() {
        let graph = HostCsr::from_coo(
            3,
            &[1, 0, 0, 0, 0],
            &[0, 1, 2, 1, 1],
            &[2.0, 1.0, 3.0, 1.0e-7, 1.0e-7],
        )
        .unwrap();
        assert_eq!(graph.indptr, vec![0, 2, 3, 3]);
        assert_eq!(graph.indices, vec![1, 2, 0]);
        assert_eq!(graph.weights[0], (1.0f32 + 1.0e-7) + 1.0e-7);
        assert!(HostCsr::from_coo(1, &[0], &[0], &[1.0]).is_err());
        assert!(HostCsr::from_coo(2, &[0, 0], &[1, 1], &[f32::MAX, f32::MAX]).is_err());
        assert!(HostCsr::from_coo(2, &[0], &[2], &[1.0]).is_err());
        assert!(HostCsr::from_coo(2, &[0], &[1], &[f32::NAN]).is_err());
    }

    #[test]
    fn csr_validation_rejects_malformed_and_asymmetric_graphs() {
        let valid = HostCsr::from_coo(3, &[0, 1], &[1, 0], &[1.0, 2.0]).unwrap();
        assert!(valid.validate().is_ok());
        assert!(
            HostCsr::from_coo(0, &[], &[], &[])
                .unwrap()
                .validate()
                .is_ok()
        );
        let mut bad = valid.clone();
        bad.indptr = vec![0, 2, 1, 2];
        assert!(bad.validate().is_err());
        bad.indptr = vec![0, 1, 2];
        assert!(bad.validate().is_err());
        bad = valid.clone();
        bad.indices[0] = 3;
        assert!(bad.validate().is_err());
        bad = valid.clone();
        bad.indices[0] = 0;
        assert!(bad.validate().is_err());
        bad = valid.clone();
        bad.weights[0] = f32::INFINITY;
        assert!(bad.validate().is_err());
        bad = HostCsr::from_coo(3, &[0], &[1], &[1.0]).unwrap();
        assert!(bad.validate().is_err());
        bad = HostCsr {
            n: 3,
            indptr: vec![0, 2, 2, 2],
            indices: vec![2, 1],
            weights: vec![1.0, 1.0],
        };
        assert!(bad.validate().is_err());
        bad.indices[1] = 2;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn votes_ignore_unlabeled_and_break_ties_in_first_occurrence_order() {
        let graph = HostCsr {
            n: 4,
            indptr: vec![0, 3, 3, 3, 3],
            indices: vec![1, 2, 3],
            weights: vec![1.0, 1.0, 9.0],
        };
        let mut labels = vec![-1, 7, 4, -1];
        propagate_labels(&graph, &mut labels, false).unwrap();
        assert_eq!(labels[0], 4);
        let mut labels = vec![-1, 7, 4, -1];
        propagate_labels(&graph, &mut labels, true).unwrap();
        assert_eq!(labels[0], -1);
    }

    #[test]
    fn outliers_are_sequential_and_remap_is_ordered() {
        let graph = HostCsr {
            n: 4,
            indptr: vec![0, 1, 2, 2, 2],
            indices: vec![2, 0],
            weights: vec![1.0, 1.0],
        };
        let mut labels = vec![-1, -1, 9, 2];
        label_outliers(&graph, &mut labels, 123).unwrap();
        assert_eq!(labels, vec![9, 9, 9, 2]);
        assert_eq!(remap_labels(&mut labels).unwrap(), 2);
        assert_eq!(labels, vec![1, 1, 1, 0]);
    }

    #[test]
    fn remap_assigns_unlabeled_rows_after_existing_parts() {
        assert_eq!(remap_labels(&mut []).unwrap(), 0);
        let mut labels = [-1, 8, -1, 3, 8];
        assert_eq!(remap_labels(&mut labels).unwrap(), 4);
        assert_eq!(labels, [2, 1, 3, 0, 1]);
        let mut labels = [-1; 3];
        assert_eq!(remap_labels(&mut labels).unwrap(), 3);
        assert_eq!(labels, [0, 1, 2]);
    }

    #[test]
    fn aggregation_preserves_internal_edges_and_transition_isolates() {
        let graph = HostCsr {
            n: 4,
            indptr: vec![0, 2, 4, 5, 5],
            indices: vec![1, 2, 0, 2, 0],
            weights: vec![0.5, 0.25, 0.5, 0.75, 0.25],
        };
        let sum = aggregate_graph(&graph, None, &[0, 0, 1, 2], 3).unwrap();
        assert_eq!(sum.indptr, vec![0, 2, 3, 3]);
        assert_eq!(sum.indices, vec![0, 1, 0]);
        assert_eq!(sum.weights, vec![1.0, 1.0, 0.25]);
        let transition = transition_graph(&sum).unwrap();
        assert_eq!(transition.indptr, vec![0, 1, 2, 2]);
        assert_eq!(transition.indices, vec![1, 0]);
        assert_eq!(transition.weights, vec![1.0, 1.0]);
    }

    #[test]
    fn hub_order_and_diverse_fill_are_deterministic() {
        let graph = HostCsr {
            n: 3,
            indptr: vec![0, 2, 3, 4],
            indices: vec![1, 2, 0, 0],
            weights: vec![1.0, 1.0, 1.0, 0.5],
        };
        assert_eq!(
            initialize_labels(&graph, 2, HubSelection::Degree, 0).unwrap(),
            vec![1, 0, -1]
        );
        assert_eq!(
            initialize_labels(&graph, 2, HubSelection::Diverse, 0).unwrap(),
            vec![0, 1, -1]
        );
    }

    #[test]
    fn asymmetric_transition_and_parallel_graph_steps_are_deterministic() {
        let graph = HostCsr {
            n: 4,
            indptr: vec![0, 2, 3, 5, 5],
            indices: vec![1, 2, 2, 0, 1],
            weights: vec![0.25, 0.75, 0.5, 0.2, 0.8],
        };
        let run = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut labels = vec![0, -1, 1, -1];
                    let sweeps = propagate_labels(&graph, &mut labels, false).unwrap();
                    let partition = [0, 0, 1, 2];
                    let counts = count_coarse_edges(&graph, None, &partition, 3).unwrap();
                    let sum = aggregate_graph(&graph, None, &partition, 3).unwrap();
                    let transition = transition_graph(&sum).unwrap();
                    (labels, sweeps, counts, sum, transition)
                })
        };
        let one = run(1);
        assert_eq!(one, run(4));
        assert_eq!(one.2, vec![2, 1, 0]);
        assert_eq!(one.4.indptr, vec![0, 1, 2, 2]);
        assert_eq!(one.4.indices, vec![1, 0]);
        assert_eq!(one.4.weights, vec![1.0, 1.0]);
        let directed = SummedCsr {
            n: 3,
            indptr: vec![0, 2, 3, 3],
            indices: vec![1, 2, 0],
            weights: vec![1.0, 3.0, 2.0],
        };
        let transition = transition_graph(&directed).unwrap();
        assert_eq!(transition.indptr, vec![0, 2, 3, 4]);
        assert_eq!(transition.indices, vec![1, 2, 0, 0]);
        assert_eq!(transition.weights, vec![1.0, 0.75, 1.0, 0.75]);
    }
}
