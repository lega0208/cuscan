//! S9 tree condensation (`condense_tree`) and S10 persistence simplification
//! (`simplify_hierarchy`).

use std::collections::HashMap;

use crate::linkage::LinkageRow;

/// Condensed tree in structure-of-arrays form. Ids below `n_points` are points; the root is
/// `n_points` and cluster ids increase in BFS order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CondensedTree {
    pub parent: Vec<usize>,
    pub child: Vec<usize>,
    pub lambda: Vec<f32>,
    pub child_size: Vec<f32>,
}

impl CondensedTree {
    pub fn len(&self) -> usize {
        self.parent.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }

    fn push(&mut self, parent: usize, child: usize, lambda: f32, child_size: f32) {
        self.parent.push(parent);
        self.child.push(child);
        self.lambda.push(lambda);
        self.child_size.push(child_size);
    }

    /// Rows whose `child_size > 1` (`cluster_tree_from_condensed_tree`).
    pub fn cluster_tree(&self) -> CondensedTree {
        self.masked(|index| self.child_size[index] > 1.0)
    }

    fn masked(&self, keep: impl Fn(usize) -> bool) -> CondensedTree {
        let mut out = CondensedTree::default();
        for index in 0..self.len() {
            if keep(index) {
                out.push(
                    self.parent[index],
                    self.child[index],
                    self.lambda[index],
                    self.child_size[index],
                );
            }
        }
        out
    }
}

/// `hdbscan_lambda_value`: `1/d` in f64 stored as f32; zero distance gives `+inf`.
pub fn lambda_value(distance: f64) -> f32 {
    if distance > 0.0 {
        (1.0 / distance) as f32
    } else {
        f32::INFINITY
    }
}

/// Breadth-first node order from `root`, left child before right child.
fn bfs_from_hierarchy(hierarchy: &[LinkageRow], root: usize, num_points: usize) -> Vec<usize> {
    let mut result = vec![root];
    let mut start = 0;
    while start < result.len() {
        let end = result.len();
        for position in start..end {
            let node = result[position];
            if node >= num_points {
                let row = &hierarchy[node - num_points];
                result.push(row.left);
                result.push(row.right);
            }
        }
        start = end;
    }
    result
}

/// Condense a single-linkage tree with `min_cluster_size` (S9).
///
/// Upstream passes `max_cluster_size = inf` here, so its "both children too large" branch is
/// unreachable and not ported; maximum size only constrains selection.
pub fn condense_tree(hierarchy: &[LinkageRow], min_cluster_size: usize) -> CondensedTree {
    let mut tree = CondensedTree::default();
    if hierarchy.is_empty() {
        return tree;
    }
    let num_points = hierarchy.len() + 1;
    let root = 2 * hierarchy.len();
    let min_size = min_cluster_size as f64;
    let mut next_label = num_points + 1;
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = num_points;
    let mut ignore = vec![false; root + 1];

    let count = |node: usize| -> f32 {
        if node >= num_points {
            hierarchy[node - num_points].size as f32
        } else {
            1.0
        }
    };
    let eliminate = |tree: &mut CondensedTree, ignore: &mut [bool], branch, parent, lambda| {
        if branch < num_points {
            tree.push(parent, branch, lambda, 1.0);
        } else {
            for sub in bfs_from_hierarchy(hierarchy, branch, num_points) {
                if sub < num_points {
                    tree.push(parent, sub, lambda, 1.0);
                } else {
                    ignore[sub] = true;
                }
            }
        }
    };

    for node in bfs_from_hierarchy(hierarchy, root, num_points) {
        if ignore[node] || node < num_points {
            continue;
        }
        let parent = relabel[node];
        let row = hierarchy[node - num_points];
        let (left, right) = (row.left, row.right);
        let lambda = lambda_value(row.distance);
        let left_count = count(left);
        let right_count = count(right);
        let left_small = (left_count as f64) < min_size;
        let right_small = (right_count as f64) < min_size;

        if left < num_points && !right_small {
            relabel[right] = parent;
            tree.push(parent, left, lambda, 1.0);
        } else if left_small && !right_small {
            relabel[right] = parent;
            eliminate(&mut tree, &mut ignore, left, parent, lambda);
        } else if !left_small && right_small {
            relabel[left] = parent;
            eliminate(&mut tree, &mut ignore, right, parent, lambda);
        } else if left_small && right_small {
            eliminate(&mut tree, &mut ignore, left, parent, lambda);
            eliminate(&mut tree, &mut ignore, right, parent, lambda);
        } else {
            relabel[left] = next_label;
            tree.push(parent, next_label, lambda, left_count);
            next_label += 1;
            relabel[right] = next_label;
            tree.push(parent, next_label, lambda, right_count);
            next_label += 1;
        }
    }
    tree
}

/// Remove clusters whose persistence is below `threshold` (S10, `simplify_hierarchy`).
///
/// Returns the tree unchanged when it has no cluster rows (upstream would index an empty
/// cluster tree).
pub fn simplify_hierarchy(tree: CondensedTree, threshold: f64) -> CondensedTree {
    let cluster_tree = tree.cluster_tree();
    if tree.is_empty() || cluster_tree.is_empty() {
        return tree;
    }
    let n_points = tree.parent[0];
    let n_nodes = cluster_tree.child[cluster_tree.len() - 1] + 1;
    let width = n_nodes - n_points;

    let mut leaf = vec![true; width];
    for &parent in &cluster_tree.parent {
        leaf[parent - n_points] = false;
    }
    let mut max_births = vec![0.0f32; width];
    for (&parent, &lambda) in tree.parent.iter().zip(&tree.lambda) {
        max_births[parent - n_points] = lambda;
    }
    let mut parent_map: Vec<usize> = (n_points..n_nodes).collect();
    let mut lambda_map: HashMap<usize, f32> = HashMap::new();

    let mut idx = cluster_tree.len() - 1;
    while idx > 0 {
        let parent = cluster_tree.parent[idx];
        let children = [cluster_tree.child[idx - 1], cluster_tree.child[idx]];
        let death = cluster_tree.lambda[idx];
        let nodes = [children[0] - n_points, children[1] - n_points];
        let births = [max_births[nodes[0]], max_births[nodes[1]]];
        let birth_max = if births[1] > births[0] {
            births[1]
        } else {
            births[0]
        };
        let birth_min = if births[1] < births[0] {
            births[1]
        } else {
            births[0]
        };
        let slot = parent - n_points;
        if birth_max > max_births[slot] {
            max_births[slot] = birth_max;
        }
        // Upstream continues when persistence is >= threshold. Negating that literal
        // comparison (rather than spelling it as `<`) preserves NaN behavior: NaN >= t
        // is false, so the split is simplified.
        if !matches!(
            f64::from(birth_min - death).partial_cmp(&threshold),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        ) {
            let sibling = usize::from(births[0] <= births[1]);
            if leaf[nodes[sibling]] {
                leaf[slot] = true;
            } else {
                lambda_map.insert(children[1 - sibling], death);
            }
            parent_map[nodes[0]] = parent;
            parent_map[nodes[1]] = parent;
        }
        if idx < 2 {
            break;
        }
        idx -= 2;
    }

    for i in 0..width {
        parent_map[i] = parent_map[parent_map[i] - n_points];
    }
    let mut skipped = vec![false; width];
    let mut prefix = vec![0usize; width + 1];
    for i in 0..width {
        let parent = parent_map[i];
        debug_assert!(parent >= n_points);
        debug_assert!(parent - n_points <= i);
        skipped[i] = parent != i + n_points;
        prefix[i + 1] = prefix[i] + usize::from(skipped[i]);
        parent_map[i] = parent - prefix[parent - n_points];
    }

    let mut out = CondensedTree::default();
    for index in 0..tree.len() {
        let (parent, child) = (tree.parent[index], tree.child[index]);
        if skipped[child.saturating_sub(n_points)] {
            continue;
        }
        let lambda = lambda_map
            .get(&parent)
            .copied()
            .unwrap_or(tree.lambda[index]);
        let child = if child >= n_points {
            parent_map[child - n_points]
        } else {
            child
        };
        out.push(
            parent_map[parent - n_points],
            child,
            lambda,
            tree.child_size[index],
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_persistence_is_simplified() {
        let hierarchy = [
            LinkageRow {
                left: 1,
                right: 0,
                distance: 0.0,
                size: 2,
            },
            LinkageRow {
                left: 3,
                right: 2,
                distance: 0.0,
                size: 2,
            },
            LinkageRow {
                left: 5,
                right: 4,
                distance: 0.0,
                size: 4,
            },
        ];
        let tree = condense_tree(&hierarchy, 2);
        let out = simplify_hierarchy(tree.clone(), 1.0);
        assert_ne!(out, tree);
    }

    #[test]
    fn parent_prefix_bound_holds_for_deep_chain() {
        let n = 3;
        let mut hierarchy = Vec::new();
        // A valid, highly unbalanced linkage tree: each merge adds one new point to
        // the growing component. Condensation gives the simplifier genuine BFS ids.
        for point in 1..100 {
            let previous = if point == 1 { 0 } else { n + point - 2 };
            hierarchy.push(LinkageRow {
                left: point,
                right: previous,
                distance: point as f64,
                size: point + 1,
            });
        }
        let tree = condense_tree(&hierarchy, 2);
        let out = simplify_hierarchy(tree, f64::INFINITY);
        assert!(!out.is_empty());
    }
}
