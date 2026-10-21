//! S12 cluster labels and membership strengths, plus S14 DBSCAN cuts.

use std::collections::HashMap;

use crate::condense::CondensedTree;
use crate::dsu::RankDsu;
use crate::error::{HdbscanError, Result};
use crate::linkage::LinkageRow;

/// Assign point labels from selected cluster ids in a condensed tree.
///
/// `clusters` retains duplicate ids: sorting and enumerating it intentionally gives duplicate
/// keys the last label, matching the upstream dictionary construction.
pub fn get_cluster_label_vector(
    tree: &CondensedTree,
    clusters: &[usize],
    cluster_selection_epsilon: f64,
    n_points: usize,
) -> Vec<i32> {
    let mut labels = vec![-1; n_points];
    if tree.is_empty() || clusters.is_empty() || n_points == 0 {
        return labels;
    }

    let mut sorted_clusters = clusters.to_vec();
    sorted_clusters.sort_unstable();
    let label_map: HashMap<usize, i32> = sorted_clusters
        .into_iter()
        .enumerate()
        .map(|(label, cluster)| (cluster, label as i32))
        .collect();
    let mut selected = vec![
        false;
        tree.parent
            .iter()
            .chain(&tree.child)
            .copied()
            .max()
            .unwrap_or(0)
            + 1
    ];
    for &cluster in clusters {
        if let Some(slot) = selected.get_mut(cluster) {
            *slot = true;
        }
    }

    let slots = tree
        .parent
        .iter()
        .chain(&tree.child)
        .copied()
        .max()
        .unwrap_or(0)
        .max(n_points.saturating_sub(1))
        + 1;
    let mut dsu = RankDsu::new(slots);
    let mut point_lambda = vec![0.0f32; n_points];
    let mut root_max_lambda = 0.0f32;
    let root = n_points;
    for index in 0..tree.len() {
        let parent = tree.parent[index];
        let child = tree.child[index];
        let lambda = tree.lambda[index];
        if parent == root {
            root_max_lambda = root_max_lambda.max(lambda);
        }
        if child < n_points {
            point_lambda[child] = lambda;
        }
        if !selected.get(child).copied().unwrap_or(false) {
            if child < n_points && lambda == 0.0 {
                continue;
            }
            if parent < slots && child < slots {
                dsu.union(parent, child);
            }
        }
    }

    let single_root = clusters
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .len()
        == 1
        && clusters[0] == root;
    for point in 0..n_points {
        let representative = dsu.find(point);
        if representative < root {
            continue;
        }
        if representative == root {
            if single_root {
                let lambda = point_lambda[point] as f64;
                let include = if cluster_selection_epsilon > 0.0 {
                    lambda >= 1.0 / cluster_selection_epsilon
                } else {
                    lambda >= root_max_lambda as f64
                };
                if include {
                    labels[point] = label_map.get(&root).copied().unwrap_or(-1);
                }
            }
        } else {
            labels[point] = label_map.get(&representative).copied().unwrap_or(-1);
        }
    }
    labels
}

/// Compute point membership strengths using point rows only to determine cluster maxima.
pub fn membership_strengths(
    tree: &CondensedTree,
    selected_sorted: &[usize],
    labels: &[i32],
    n_points: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; labels.len()];
    if tree.is_empty() || selected_sorted.is_empty() {
        return output;
    }
    let mut max_lambdas = HashMap::<usize, f64>::new();
    for index in 0..tree.len() {
        if tree.child_size[index] == 1.0
            && selected_sorted.binary_search(&tree.parent[index]).is_ok()
        {
            let lambda = tree.lambda[index] as f64;
            let maximum = max_lambdas.entry(tree.parent[index]).or_default();
            *maximum = maximum.max(lambda);
        }
    }
    for index in 0..tree.len() {
        let point = tree.child[index];
        if point >= n_points || point >= labels.len() || labels[point] < 0 {
            continue;
        }
        let cluster_label = labels[point] as usize;
        let Some(&cluster) = selected_sorted.get(cluster_label) else {
            continue;
        };
        let maximum = max_lambdas.get(&cluster).copied().unwrap_or(0.0);
        let lambda = tree.lambda[index];
        output[point] = if maximum == 0.0 || !lambda.is_finite() {
            1.0
        } else {
            ((f64::from(lambda).min(maximum)) / maximum) as f32
        };
    }
    output
}

/// Label connected components at a strict single-linkage distance cut.
pub fn dbscan_cut(linkage: &[LinkageRow], cut: f64, min_size: usize) -> Result<Vec<i32>> {
    if cut.is_nan() {
        return Err(HdbscanError::InvalidConfig(
            "DBSCAN cut must not be NaN".into(),
        ));
    }
    let n_points = linkage
        .len()
        .checked_add(1)
        .ok_or(HdbscanError::Overflow("DBSCAN point count"))?;
    let slots = linkage
        .len()
        .checked_mul(2)
        .and_then(|n| n.checked_add(1))
        .ok_or(HdbscanError::Overflow("DBSCAN disjoint-set size"))?;
    let mut dsu = RankDsu::new(slots);
    for (index, row) in linkage.iter().enumerate() {
        if row.distance < cut {
            let cluster = n_points + index;
            if row.left >= slots || row.right >= slots {
                return Err(HdbscanError::InvalidInput(format!(
                    "linkage row {index} has an out-of-range node"
                )));
            }
            dsu.union(row.left, cluster);
            dsu.union(row.right, cluster);
        }
    }
    let mut representatives = Vec::with_capacity(n_points);
    let mut sizes = HashMap::<usize, usize>::new();
    for point in 0..n_points {
        let representative = dsu.find(point);
        representatives.push(representative);
        *sizes.entry(representative).or_default() += 1;
    }
    let mut ordered = sizes.keys().copied().collect::<Vec<_>>();
    ordered.sort_unstable();
    let mut next_label = 0i32;
    let label_map = ordered
        .into_iter()
        .map(|representative| {
            let label = if sizes[&representative] < min_size {
                -1
            } else {
                let label = next_label;
                next_label += 1;
                label
            };
            (representative, label)
        })
        .collect::<HashMap<_, _>>();
    Ok(representatives
        .into_iter()
        .map(|rep| label_map[&rep])
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(
        parent: &[usize],
        child: &[usize],
        lambda: &[f32],
        child_size: &[f32],
    ) -> CondensedTree {
        CondensedTree {
            parent: parent.into(),
            child: child.into(),
            lambda: lambda.into(),
            child_size: child_size.into(),
        }
    }

    #[test]
    fn duplicate_selected_ids_use_last_sorted_label() {
        let ct = tree(
            &[2, 2, 3, 3],
            &[3, 4, 0, 1],
            &[1., 1., 2., 2.],
            &[2., 2., 1., 1.],
        );
        assert_eq!(get_cluster_label_vector(&ct, &[3, 3], 0.0, 2), vec![1, 1]);
    }

    #[test]
    fn membership_uses_point_maximum_and_f64_division() {
        let ct = tree(&[3, 3, 3], &[0, 1, 4], &[1., 3., 100.], &[1., 1., 2.]);
        assert_eq!(
            membership_strengths(&ct, &[3, 3], &[1, 1], 2),
            vec![1.0 / 3.0, 1.0]
        );
    }

    #[test]
    fn dbscan_cut_is_strict_and_rejects_nan() {
        let linkage = [
            LinkageRow {
                left: 1,
                right: 0,
                distance: 1.0,
                size: 2,
            },
            LinkageRow {
                left: 3,
                right: 2,
                distance: 2.0,
                size: 3,
            },
        ];
        assert_eq!(dbscan_cut(&linkage, 1.0, 1).unwrap(), vec![0, 1, 2]);
        assert_eq!(dbscan_cut(&linkage, 2.1, 2).unwrap(), vec![0, 0, 0]);
        assert!(matches!(
            dbscan_cut(&linkage, f64::NAN, 1),
            Err(HdbscanError::InvalidConfig(_))
        ));
    }
}
