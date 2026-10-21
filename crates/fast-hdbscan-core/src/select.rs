//! S11 cluster selection and pinned epsilon search.

use std::collections::HashSet;

use crate::argsort::argsort;
use crate::condense::CondensedTree;
use crate::config::{ClusterSelectionMethod, HdbscanConfig};

/// Select cluster ids from the condensed tree using the configured method and epsilon.
pub fn select_clusters(
    tree: &CondensedTree,
    n_points: usize,
    config: &HdbscanConfig,
) -> Vec<usize> {
    let cluster_tree = tree.cluster_tree();
    let mut selected = match config.cluster_selection_method {
        ClusterSelectionMethod::Eom => extract_eom(
            tree,
            &cluster_tree,
            n_points,
            config.max_cluster_size,
            config.allow_single_cluster,
        ),
        ClusterSelectionMethod::Leaf => leaves(&cluster_tree, n_points),
    };
    if selected.len() > 1 && config.cluster_selection_epsilon > 0.0 {
        selected = epsilon_search(&selected, &cluster_tree, config.cluster_selection_epsilon);
    }
    selected
}

fn extract_eom(
    condensed: &CondensedTree,
    tree: &CondensedTree,
    n_points: usize,
    max_size: Option<usize>,
    allow_root: bool,
) -> Vec<usize> {
    if tree.is_empty() {
        return Vec::new();
    }
    let width = tree
        .child
        .iter()
        .chain(&tree.parent)
        .copied()
        .max()
        .unwrap()
        - n_points
        + 1;
    let mut scores = vec![0.0f32; width];
    let mut sizes = vec![0.0f32; width];
    let mut children = vec![Vec::new(); width];
    let mut born = vec![false; width];
    born[0] = true;
    let root = n_points;
    sizes[0] = n_points.saturating_sub(1) as f32;
    let mut insertion_order = vec![root];
    for i in 0..condensed.len() {
        let parent = condensed.parent[i];
        let child = condensed.child[i];
        let score = condensed.lambda[i] * condensed.child_size[i];
        let pi = parent - n_points;
        debug_assert!(
            born[pi],
            "cluster parent must be born before its child rows"
        );
        if condensed.child_size[i] > 1.0 {
            let ci = child - n_points;
            scores[ci] = -score;
            sizes[ci] = condensed.child_size[i];
            if !born[ci] {
                insertion_order.push(child);
            }
            born[ci] = true;
            children[pi].push(child);
        }
        scores[pi] += score;
    }

    let roots = if allow_root {
        vec![root]
    } else {
        children[0].clone()
    };
    let mut chosen = vec![false; width];
    let mut value = vec![0.0f64; width];
    for &start in &roots {
        let mut stack = vec![(start, false)];
        while let Some((node, visited)) = stack.pop() {
            let index = node - n_points;
            if !visited {
                stack.push((node, true));
                for &child in children[index].iter().rev() {
                    stack.push((child, false));
                }
            } else {
                let mut child_total = 0.0f64;
                for &child in &children[index] {
                    child_total += value[child - n_points];
                }
                // Python max(s, 0.0) preserves a NaN in its first argument.
                let current = (if 0.0 > scores[index] {
                    0.0
                } else {
                    scores[index]
                }) as f64;
                let too_large =
                    max_size.is_some_and(|limit| f64::from(sizes[index]) > limit as f64);
                if child_total > current || too_large {
                    value[index] = child_total;
                } else {
                    chosen[index] = true;
                    value[index] = current;
                }
            }
        }
    }
    let mut selected = vec![false; width];
    let mut stack: Vec<_> = roots.into_iter().rev().map(|node| (node, false)).collect();
    while let Some((node, ancestor_selected)) = stack.pop() {
        let index = node - n_points;
        let selected_here = !ancestor_selected && chosen[index];
        selected[index] = selected_here;
        for &child in children[index].iter().rev() {
            stack.push((child, ancestor_selected || selected_here));
        }
    }
    insertion_order
        .into_iter()
        .filter(|&node| selected[node - n_points])
        .collect()
}

fn leaves(tree: &CondensedTree, n_points: usize) -> Vec<usize> {
    if tree.is_empty() {
        return Vec::new();
    }
    let mut parents = HashSet::with_capacity(tree.len());
    for &parent in &tree.parent {
        parents.insert(parent);
    }
    let mut result: Vec<_> = tree
        .child
        .iter()
        .copied()
        .filter(|child| !parents.contains(child))
        .collect();
    result.sort_unstable();
    debug_assert!(result.iter().all(|&node| node >= n_points));
    result
}

fn epsilon_search(selected: &[usize], tree: &CondensedTree, epsilon: f64) -> Vec<usize> {
    debug_assert!(tree.child.windows(2).all(|pair| pair[0] <= pair[1]));
    let order = argsort(&tree.parent);
    let parents: Vec<_> = order.iter().map(|&i| tree.parent[i]).collect();
    let children: Vec<_> = order.iter().map(|&i| tree.child[i]).collect();
    let root = *tree.parent.iter().min().unwrap();
    let mut processed = HashSet::new();
    let mut output = Vec::with_capacity(selected.len());
    for &cluster in selected {
        let idx = tree.child.partition_point(|&child| child < cluster);
        let lambda = tree.lambda.get(idx).copied().unwrap_or(0.0);
        let death_eps = if lambda > 0.0 {
            1.0 / f64::from(lambda)
        } else {
            f64::INFINITY
        };
        if death_eps < epsilon {
            if !processed.contains(&cluster) {
                let ancestor = traverse_upwards(cluster, epsilon, root, tree);
                output.push(ancestor);
                processed.extend(segments_in_branch(ancestor, &parents, &children));
            }
        } else {
            output.push(cluster);
        }
    }
    output
}

fn traverse_upwards(node: usize, epsilon: f64, root: usize, tree: &CondensedTree) -> usize {
    let mut segment = node;
    loop {
        let Some(index) = tree.child.iter().position(|&child| child == segment) else {
            return root;
        };
        let parent = tree.parent[index];
        if parent == root {
            return root;
        }
        let idx = tree.child.partition_point(|&child| child < parent);
        let lambda = tree.lambda.get(idx).copied().unwrap_or(0.0);
        let death_eps = if lambda > 0.0 {
            1.0 / f64::from(lambda)
        } else {
            f64::INFINITY
        };
        if death_eps >= epsilon {
            return parent;
        }
        segment = parent;
    }
}

fn segments_in_branch(segment: usize, parents: &[usize], children: &[usize]) -> HashSet<usize> {
    let mut result = HashSet::new();
    let mut pending = vec![segment];
    while !pending.is_empty() {
        let mut next = Vec::new();
        for node in pending.drain(..) {
            if !result.insert(node) {
                continue;
            }
            let idx = parents.partition_point(|&parent| parent < node);
            if idx < parents.len() {
                next.push(children[idx]);
                if let Some(&foreign_child) = children.get(idx + 1) {
                    next.push(foreign_child);
                }
            }
        }
        pending = next;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_selection_is_sorted_and_empty_safe() {
        let tree = CondensedTree {
            parent: vec![4, 4, 5, 5],
            child: vec![5, 6, 7, 8],
            lambda: vec![1.0; 4],
            child_size: vec![2.0; 4],
        };
        assert_eq!(leaves(&tree, 4), vec![6, 7, 8]);
        assert!(leaves(&CondensedTree::default(), 0).is_empty());
    }

    #[test]
    fn pr72_branch_keeps_foreign_children() {
        let parents = [5, 5, 7, 7];
        let children = [6, 7, 8, 9];
        let result = segments_in_branch(6, &parents, &children);
        assert_eq!(result, [6, 8, 9].into_iter().collect());
    }

    #[test]
    fn eom_handles_a_deep_chain_iteratively() {
        let n = 100_002;
        let mut tree = CondensedTree::default();
        for i in 0..100_000 {
            let parent = n + i;
            tree.parent.extend([parent, parent]);
            tree.child.extend([parent + 1, i]);
            tree.lambda.extend([1.0, 1.0]);
            tree.child_size.extend([(n - i - 1) as f32, 1.0]);
        }
        tree.parent.extend([n + 100_000, n + 100_000]);
        tree.child.extend([100_000, 100_001]);
        tree.lambda.extend([2.0, 2.0]);
        tree.child_size.extend([1.0, 1.0]);
        assert!(!extract_eom(&tree, &tree.cluster_tree(), n, None, true).is_empty());
    }

    #[test]
    fn iterative_eom_matches_literal_recursion_on_generated_trees() {
        fn recurse(
            node: usize,
            n: usize,
            tree: &CondensedTree,
            scores: &[f32],
            sizes: &[f32],
            max_size: f64,
            selected: &mut [bool],
        ) -> f64 {
            let children: Vec<_> = tree
                .parent
                .iter()
                .zip(&tree.child)
                .filter_map(|(&parent, &child)| (parent == node).then_some(child))
                .collect();
            let mut total = 0.0;
            for &child in &children {
                total += recurse(child, n, tree, scores, sizes, max_size, selected);
            }
            let value = if 0.0 > scores[node - n] {
                0.0
            } else {
                scores[node - n]
            } as f64;
            if total > value || f64::from(sizes[node - n]) > max_size {
                return total;
            }
            selected[node - n] = true;
            let mut pending = children;
            while let Some(child) = pending.pop() {
                selected[child - n] = false;
                pending.extend(
                    tree.parent
                        .iter()
                        .zip(&tree.child)
                        .filter_map(|(&parent, &next)| (parent == child).then_some(next)),
                );
            }
            value
        }
        for seed in 0..32usize {
            // A full binary cluster tree in BFS order, with two point rows per leaf.
            let n = 16;
            let mut full = CondensedTree::default();
            for node in 0..15 {
                let parent = n + node;
                let lambda = 1.0 + ((node * 17 + seed * 7) % 11) as f32 * 0.25;
                if node < 7 {
                    for child in [node * 2 + 1, node * 2 + 2] {
                        full.parent.push(parent);
                        full.child.push(n + child);
                        full.lambda.push(lambda);
                        full.child_size.push((16 >> ((child + 1).ilog2())) as f32);
                    }
                } else {
                    for point in [(node - 7) * 2, (node - 7) * 2 + 1] {
                        full.parent.push(parent);
                        full.child.push(point);
                        full.lambda.push(lambda + 4.0);
                        full.child_size.push(1.0);
                    }
                }
            }
            let clusters = full.cluster_tree();
            let mut scores = vec![0.0f32; 15];
            let mut sizes = vec![0.0f32; 15];
            sizes[0] = (n - 1) as f32;
            for i in 0..full.len() {
                let amount = full.lambda[i] * full.child_size[i];
                if full.child_size[i] > 1.0 {
                    scores[full.child[i] - n] = -amount;
                    sizes[full.child[i] - n] = full.child_size[i];
                }
                scores[full.parent[i] - n] += amount;
            }
            for allow_root in [false, true] {
                for maximum in [None, Some(5)] {
                    let mut expected = vec![false; 15];
                    let roots = if allow_root {
                        vec![n]
                    } else {
                        vec![n + 1, n + 2]
                    };
                    for root in roots {
                        recurse(
                            root,
                            n,
                            &clusters,
                            &scores,
                            &sizes,
                            maximum.map_or(f64::INFINITY, |x| x as f64),
                            &mut expected,
                        );
                    }
                    let expected: Vec<_> = expected
                        .iter()
                        .enumerate()
                        .filter_map(|(node, &chosen)| chosen.then_some(n + node))
                        .collect();
                    assert_eq!(
                        extract_eom(&full, &clusters, n, maximum, allow_root),
                        expected,
                        "seed={seed}, root={allow_root}, max={maximum:?}"
                    );
                }
            }
        }
    }
}
