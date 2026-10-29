//! Backend-free `approximate_predict` hierarchy metadata and candidate math.
//!
//! This ports original hdbscan's `PredictionData` reconstruction and
//! `_find_cluster_and_probability` from the *final* fitted condensed tree and labels.
//! Neighbor search is the caller's job: callers supply ordered exact candidate lists
//! (`(squared distance, compact id)` order, self included) and prediction core
//! distances; this module validates them and performs all f64 hierarchy math.
//!
//! Hierarchy ids stay raw (post finite-row remapping) ids. Only point attachments are
//! indexed by compact finite row.

use std::collections::HashMap;

use crate::condense::CondensedTree;
use crate::error::{HdbscanError, Result};

/// Prediction for one query row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointPrediction {
    /// Existing fitted label, or `-1` for noise.
    pub label: i32,
    /// Reference prediction strength in `[0, 1]`.
    pub probability: f64,
    /// Raw training-row id of the mutual-reachability connector, or `-1` only for the
    /// no-cluster-tree shortcut.
    pub connecting_point: i32,
}

/// Column-wise prediction results in query order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PredictionColumns {
    pub labels: Vec<i32>,
    pub probabilities: Vec<f64>,
    pub connecting_points: Vec<i32>,
}

impl PredictionColumns {
    /// Reference no-cluster-tree shortcut: all noise, zero strength, `-1` connectors.
    pub fn no_cluster_tree(rows: usize) -> Self {
        Self {
            labels: vec![-1; rows],
            probabilities: vec![0.0; rows],
            connecting_points: vec![-1; rows],
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    fn reserve(&mut self, rows: usize) {
        self.labels.reserve(rows);
        self.probabilities.reserve(rows);
        self.connecting_points.reserve(rows);
    }

    fn push(&mut self, prediction: PointPrediction) {
        self.labels.push(prediction.label);
        self.probabilities.push(prediction.probability);
        self.connecting_points.push(prediction.connecting_point);
    }
}

/// Checked external-query candidate width `2*s`, requiring `2*s <= n_finite`.
///
/// Only models with a nonempty cluster tree need this; check it before any cold work.
pub fn candidate_width(min_samples: usize, n_finite: usize) -> Result<usize> {
    if min_samples == 0 {
        return Err(HdbscanError::InvalidInput(
            "min_samples must be positive".into(),
        ));
    }
    let width = min_samples
        .checked_mul(2)
        .ok_or(HdbscanError::Overflow("prediction candidate width"))?;
    if width > n_finite {
        return Err(HdbscanError::InvalidInput(format!(
            "approximate_predict needs 2*min_samples = {width} neighbors but the model has only {n_finite} finite training points"
        )));
    }
    Ok(width)
}

/// Prediction training core distances from row-major inclusive-self kNN rows of width `s`.
///
/// Each row is a `k = s` query of a training point against the training set *including
/// itself*, ordered by squared distance; the core is the Euclidean distance at rank `s-1`.
/// For `s == 1` every core is zero, so callers may skip the search and pass
/// `vec![0.0; n]` directly; this function still accepts width-1 rows.
pub fn prediction_core_from_squared_rows(squared: &[f32], min_samples: usize) -> Result<Vec<f64>> {
    if min_samples == 0 {
        return Err(HdbscanError::InvalidInput(
            "min_samples must be positive".into(),
        ));
    }
    if !squared.len().is_multiple_of(min_samples) {
        return Err(HdbscanError::InvalidInput(format!(
            "prediction core rows have {} values, not a multiple of min_samples = {min_samples}",
            squared.len()
        )));
    }
    let mut cores = Vec::with_capacity(squared.len() / min_samples);
    for row in squared.chunks_exact(min_samples) {
        let value = row[min_samples - 1];
        let core = if min_samples == 1 {
            0.0
        } else {
            euclidean_from_squared(value)?
        };
        cores.push(core);
    }
    Ok(cores)
}

fn euclidean_from_squared(value: f32) -> Result<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(HdbscanError::Numerical(format!(
            "neighbor search returned invalid squared distance {value}"
        )));
    }
    Ok(f64::from(value).sqrt())
}

#[derive(Debug, Clone, Copy)]
struct ClusterNode {
    /// Raw hierarchy id.
    id: usize,
    /// Parent node index and birth lambda of this cluster, `None` for the root.
    parent: Option<(usize, f64)>,
    /// Propagated selected label, `-1` when unmapped.
    label: i32,
    /// Propagated selected-anchor maximum lambda.
    max_lambda: f64,
}

#[derive(Debug, Clone, Copy)]
struct PointAttachment {
    node: usize,
    departure_lambda: f64,
}

/// Immutable validated hierarchy metadata for `approximate_predict`.
#[derive(Debug, Clone)]
pub struct PredictionHierarchy {
    finite_index: Vec<u32>,
    /// Empty for the no-cluster-tree state.
    nodes: Vec<ClusterNode>,
    /// Indexed by compact finite row; empty for the no-cluster-tree state.
    points: Vec<PointAttachment>,
    root: usize,
}

/// Portable, storage-independent columns for a validated prediction hierarchy.
///
/// The all-empty value represents the no-cluster-tree shortcut. Clustered values use
/// node index 0 as the root and `u64::MAX` as its parent sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionHierarchyState {
    /// Ascending raw training-row ids for compact finite rows.
    pub finite_index: Vec<u32>,
    /// Raw hierarchy ids, with the root at index 0.
    pub node_ids: Vec<u64>,
    /// Parent node index; `u64::MAX` marks the single root at node index 0.
    pub node_parent_indices: Vec<u64>,
    /// Birth lambda on the parent edge; root stores canonical +0.0.
    pub node_birth_lambdas: Vec<f64>,
    /// Propagated fitted label (`-1` means unmapped/no selected cluster).
    pub node_labels: Vec<i32>,
    /// Propagated selected-anchor maximum lambda.
    pub node_max_lambdas: Vec<f64>,
    /// Hierarchy node index attached to each compact finite row.
    pub point_node_indices: Vec<u64>,
    /// Departure lambda for each compact finite row.
    pub point_departure_lambdas: Vec<f64>,
}

impl PredictionHierarchy {
    /// Export the validated hierarchy into portable, owned columns.
    pub fn export_state(&self) -> Result<PredictionHierarchyState> {
        if self.no_cluster_tree() {
            return Ok(PredictionHierarchyState {
                finite_index: Vec::new(),
                node_ids: Vec::new(),
                node_parent_indices: Vec::new(),
                node_birth_lambdas: Vec::new(),
                node_labels: Vec::new(),
                node_max_lambdas: Vec::new(),
                point_node_indices: Vec::new(),
                point_departure_lambdas: Vec::new(),
            });
        }
        let mut finite_index = Vec::new();
        reserve_column(
            &mut finite_index,
            self.finite_index.len(),
            "portable finite index",
        )?;
        finite_index.extend_from_slice(&self.finite_index);
        let mut state = PredictionHierarchyState {
            finite_index,
            node_ids: Vec::new(),
            node_parent_indices: Vec::new(),
            node_birth_lambdas: Vec::new(),
            node_labels: Vec::new(),
            node_max_lambdas: Vec::new(),
            point_node_indices: Vec::new(),
            point_departure_lambdas: Vec::new(),
        };
        reserve_column(
            &mut state.node_ids,
            self.nodes.len(),
            "portable hierarchy nodes",
        )?;
        reserve_column(
            &mut state.node_parent_indices,
            self.nodes.len(),
            "portable hierarchy parents",
        )?;
        reserve_column(
            &mut state.node_birth_lambdas,
            self.nodes.len(),
            "portable hierarchy births",
        )?;
        reserve_column(
            &mut state.node_labels,
            self.nodes.len(),
            "portable hierarchy labels",
        )?;
        reserve_column(
            &mut state.node_max_lambdas,
            self.nodes.len(),
            "portable hierarchy maxima",
        )?;
        reserve_column(
            &mut state.point_node_indices,
            self.points.len(),
            "portable hierarchy point nodes",
        )?;
        reserve_column(
            &mut state.point_departure_lambdas,
            self.points.len(),
            "portable hierarchy point departures",
        )?;
        for node in &self.nodes {
            state.node_ids.push(
                u64::try_from(node.id)
                    .map_err(|_| HdbscanError::Overflow("portable hierarchy node id"))?,
            );
            let (parent, birth) = match node.parent {
                Some((parent, birth)) => (
                    u64::try_from(parent)
                        .map_err(|_| HdbscanError::Overflow("portable hierarchy parent index"))?,
                    birth,
                ),
                None => (u64::MAX, 0.0),
            };
            state.node_parent_indices.push(parent);
            state.node_birth_lambdas.push(birth);
            state.node_labels.push(node.label);
            state.node_max_lambdas.push(node.max_lambda);
        }
        for point in &self.points {
            state.point_node_indices.push(
                u64::try_from(point.node)
                    .map_err(|_| HdbscanError::Overflow("portable hierarchy point node index"))?,
            );
            state.point_departure_lambdas.push(point.departure_lambda);
        }
        Ok(state)
    }

    /// Validate and import portable hierarchy columns.
    ///
    /// Every column is treated as untrusted and checked before private hierarchy metadata
    /// is constructed. An all-empty state imports the no-cluster-tree shortcut.
    pub fn from_state(state: PredictionHierarchyState) -> Result<Self> {
        let PredictionHierarchyState {
            finite_index,
            node_ids,
            node_parent_indices,
            node_birth_lambdas,
            node_labels,
            node_max_lambdas,
            point_node_indices,
            point_departure_lambdas,
        } = state;
        let columns_empty = [
            finite_index.is_empty(),
            node_ids.is_empty(),
            node_parent_indices.is_empty(),
            node_birth_lambdas.is_empty(),
            node_labels.is_empty(),
            node_max_lambdas.is_empty(),
            point_node_indices.is_empty(),
            point_departure_lambdas.is_empty(),
        ];
        if columns_empty.iter().all(|&empty| empty) {
            return Ok(Self {
                finite_index,
                nodes: Vec::new(),
                points: Vec::new(),
                root: 0,
            });
        }
        if columns_empty.iter().any(|&empty| empty) {
            return Err(invalid(
                "portable prediction hierarchy has a partially empty column set",
            ));
        }

        let n_finite = finite_index.len();
        let n_nodes = node_ids.len();
        if n_finite > i32::MAX as usize || n_nodes > i32::MAX as usize {
            return Err(HdbscanError::Overflow(
                "portable prediction hierarchy count",
            ));
        }
        checked_column_bytes::<u32>(n_finite, "portable finite index")?;
        checked_column_bytes::<u64>(n_nodes, "portable hierarchy node")?;
        checked_column_bytes::<f64>(n_nodes, "portable hierarchy node")?;
        checked_column_bytes::<i32>(n_nodes, "portable hierarchy node")?;
        checked_column_bytes::<u64>(n_finite, "portable hierarchy point")?;
        checked_column_bytes::<f64>(n_finite, "portable hierarchy point")?;
        if node_parent_indices.len() != n_nodes
            || node_birth_lambdas.len() != n_nodes
            || node_labels.len() != n_nodes
            || node_max_lambdas.len() != n_nodes
            || point_node_indices.len() != n_finite
            || point_departure_lambdas.len() != n_finite
        {
            return Err(invalid(
                "portable prediction hierarchy columns have inconsistent lengths",
            ));
        }
        if finite_index.iter().any(|&raw| raw > i32::MAX as u32)
            || finite_index.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(invalid(
                "portable finite index must be strictly ascending and fit i32",
            ));
        }

        let mut ids = Vec::new();
        reserve_column(&mut ids, n_nodes, "portable hierarchy ids")?;
        let mut unique_ids = std::collections::HashSet::new();
        unique_ids
            .try_reserve(n_nodes)
            .map_err(|_| invalid("unable to reserve portable hierarchy id validation"))?;
        for &id in &node_ids {
            let host_id = usize::try_from(id)
                .map_err(|_| invalid("portable hierarchy node id does not fit usize"))?;
            if !unique_ids.insert(host_id) {
                return Err(invalid("portable hierarchy node ids are not unique"));
            }
            ids.push(host_id);
        }
        if ids[0] != *ids.iter().min().expect("nonempty node ids") {
            return Err(invalid(
                "portable hierarchy root is not the minimum node id",
            ));
        }
        if node_parent_indices[0] != u64::MAX
            || node_birth_lambdas[0].to_bits() != 0.0_f64.to_bits()
        {
            return Err(invalid(
                "portable hierarchy root sentinel or birth lambda is not canonical",
            ));
        }
        let mut parents = Vec::new();
        reserve_column(&mut parents, n_nodes, "portable hierarchy parents")?;
        parents.push(None);
        for index in 1..n_nodes {
            let parent = usize::try_from(node_parent_indices[index])
                .map_err(|_| invalid("portable hierarchy parent index does not fit usize"))?;
            if parent >= n_nodes
                || parent == index
                || ids[parent] >= ids[index]
                || ids[index] <= ids[0]
            {
                return Err(invalid("portable hierarchy parent graph is invalid"));
            }
            parents.push(Some(parent));
        }

        let mut labels = std::collections::HashSet::new();
        labels
            .try_reserve(n_nodes)
            .map_err(|_| invalid("unable to reserve portable hierarchy label validation"))?;
        if node_labels[0] >= 0 {
            labels.insert(node_labels[0]);
        }
        for index in 0..n_nodes {
            let birth = node_birth_lambdas[index];
            let maximum = node_max_lambdas[index];
            if birth.is_nan()
                || birth < 0.0
                || maximum.is_nan()
                || maximum < 0.0
                || node_labels[index] < -1
            {
                return Err(invalid(
                    "portable hierarchy contains an invalid lambda or label",
                ));
            }
            if index == 0 {
                if node_labels[index] == -1 && maximum.to_bits() != 0.0_f64.to_bits() {
                    return Err(invalid(
                        "unmapped portable hierarchy root must have canonical zero maximum",
                    ));
                }
            } else {
                let parent = parents[index].expect("non-root parent checked");
                if node_labels[index] == node_labels[parent] {
                    if maximum.to_bits() != node_max_lambdas[parent].to_bits() {
                        return Err(invalid(
                            "inherited portable hierarchy maximum differs from its parent",
                        ));
                    }
                } else if node_labels[index] < 0 || !labels.insert(node_labels[index]) {
                    return Err(invalid(
                        "portable hierarchy anchor labels are invalid or duplicated",
                    ));
                }
            }
        }

        let mut nodes = Vec::new();
        reserve_column(&mut nodes, n_nodes, "validated hierarchy nodes")?;
        for index in 0..n_nodes {
            nodes.push(ClusterNode {
                id: ids[index],
                parent: parents[index].map(|parent| (parent, node_birth_lambdas[index])),
                label: node_labels[index],
                max_lambda: node_max_lambdas[index],
            });
        }
        let mut points = Vec::new();
        reserve_column(&mut points, n_finite, "validated hierarchy points")?;
        for (node, &departure_lambda) in point_node_indices.iter().zip(&point_departure_lambdas) {
            let node = usize::try_from(*node)
                .map_err(|_| invalid("portable point node index does not fit usize"))?;
            if node >= n_nodes || departure_lambda.is_nan() || departure_lambda < 0.0 {
                return Err(invalid("portable point attachment is invalid"));
            }
            points.push(PointAttachment {
                node,
                departure_lambda,
            });
        }
        Ok(Self {
            finite_index,
            nodes,
            points,
            root: 0,
        })
    }

    /// Build metadata from the final raw-row condensed tree, raw fitted labels
    /// (`labels.len()` is the raw row count) and the ascending finite-to-raw mapping.
    ///
    /// Point rows have `child < raw_n`; cluster edges have `child_size > 1`.
    pub fn new(tree: &CondensedTree, labels: &[i32], finite_index: &[u32]) -> Result<Self> {
        let raw_n = labels.len();
        if raw_n > i32::MAX as usize {
            return Err(HdbscanError::Overflow("raw point index"));
        }
        validate_finite_index(finite_index, raw_n)?;
        validate_labels(labels, finite_index)?;
        let rows = tree.len();
        if tree.child.len() != rows || tree.lambda.len() != rows || tree.child_size.len() != rows {
            return Err(invalid("condensed tree columns have inconsistent lengths"));
        }

        // Row-level validation. Cluster edges are collected afterwards with an exact capacity.
        for row in 0..rows {
            let (parent, child) = (tree.parent[row], tree.child[row]);
            let (lambda, size) = (tree.lambda[row], tree.child_size[row]);
            if parent < raw_n {
                return Err(invalid(format!(
                    "condensed tree row {row} has point id {parent} as a parent"
                )));
            }
            if lambda.is_nan() || lambda < 0.0 {
                return Err(invalid(format!(
                    "condensed tree row {row} has invalid lambda {lambda}"
                )));
            }
            if !size.is_finite() || size < 1.0 {
                return Err(invalid(format!(
                    "condensed tree row {row} has invalid child size {size}"
                )));
            }
            let is_point = child < raw_n;
            let is_cluster = size > 1.0;
            if is_point == is_cluster {
                return Err(invalid(format!(
                    "condensed tree row {row} mixes point/cluster child {child} with size {size}"
                )));
            }
            if is_cluster && child <= parent {
                return Err(invalid(format!(
                    "cluster edge {parent} -> {child} does not increase the cluster id"
                )));
            }
        }

        // Validated rows: `child_size > 1` exactly marks cluster edges, so the count is exact.
        let cluster_count = tree.child_size.iter().filter(|&&size| size > 1.0).count();
        let mut cluster_rows = Vec::with_capacity(cluster_count);
        cluster_rows.extend((0..rows).filter(|&row| tree.child_size[row] > 1.0));

        if cluster_rows.is_empty() {
            // Root-only trees: a single parent, each finite point exactly once, nothing else.
            // The empty tree is the valid one-finite-row/no-split state.
            if rows == 0 && finite_index.len() >= 2 {
                return Err(invalid(
                    "condensed tree has no rows for several finite training points",
                ));
            }
            if rows > 0 {
                let root = tree.parent[0];
                if tree.parent.iter().any(|&parent| parent != root) {
                    return Err(invalid("root-only condensed tree has several parents"));
                }
                // Injected non-finite rows are never connectors; `seen` covers finite rows only.
                let mut seen = vec![false; finite_index.len()];
                for &child in &tree.child {
                    let Ok(position) =
                        finite_index.binary_search_by(|&raw| (raw as usize).cmp(&child))
                    else {
                        continue;
                    };
                    if std::mem::replace(&mut seen[position], true) {
                        return Err(invalid(format!(
                            "training point {child} has more than one condensed tree row"
                        )));
                    }
                }
                if let Some(&raw) = finite_index
                    .iter()
                    .zip(&seen)
                    .find_map(|(raw, &covered)| (!covered).then_some(raw))
                {
                    return Err(invalid(format!(
                        "finite training point {raw} has no condensed tree row"
                    )));
                }
            }
            return Ok(Self {
                finite_index: finite_index.to_vec(),
                nodes: Vec::new(),
                points: Vec::new(),
                root: 0,
            });
        }

        // Cluster nodes: root is the minimum cluster-edge parent.
        let root_id = cluster_rows
            .iter()
            .map(|&row| tree.parent[row])
            .min()
            .ok_or_else(|| invalid("cluster tree unexpectedly empty"))?;
        let mut node_of: HashMap<usize, usize> = HashMap::with_capacity(cluster_rows.len() + 1);
        let mut nodes = Vec::with_capacity(cluster_rows.len() + 1);
        node_of.insert(root_id, 0);
        nodes.push(ClusterNode {
            id: root_id,
            parent: None,
            label: -1,
            max_lambda: 0.0,
        });
        for &row in &cluster_rows {
            let child = tree.child[row];
            if node_of.insert(child, nodes.len()).is_some() {
                return Err(invalid(format!(
                    "cluster {child} has more than one parent edge or is the root"
                )));
            }
            nodes.push(ClusterNode {
                id: child,
                parent: None,
                label: -1,
                max_lambda: 0.0,
            });
        }
        for (offset, &row) in cluster_rows.iter().enumerate() {
            let parent = tree.parent[row];
            let parent_node = *node_of.get(&parent).ok_or_else(|| {
                invalid(format!(
                    "cluster parent {parent} is neither the root nor a child cluster"
                ))
            })?;
            nodes[offset + 1].parent = Some((parent_node, f64::from(tree.lambda[row])));
        }

        // Maximum lambda over all outgoing rows, point attachments and label anchors.
        let mut max_lambda = vec![0.0_f64; nodes.len()];
        let mut seen = vec![false; nodes.len()];
        let mut attachment: HashMap<usize, PointAttachment> =
            HashMap::with_capacity(finite_index.len());
        // Distinct anchored labels map injectively to nodes on success, so `nodes.len()` is exact.
        let mut anchors: HashMap<i32, usize> = HashMap::with_capacity(nodes.len());
        for row in 0..rows {
            let parent = tree.parent[row];
            let node = *node_of.get(&parent).ok_or_else(|| {
                invalid(format!(
                    "condensed tree parent {parent} is not part of the cluster tree"
                ))
            })?;
            let lambda = f64::from(tree.lambda[row]);
            if !seen[node] || lambda > max_lambda[node] {
                max_lambda[node] = lambda;
                seen[node] = true;
            }
            let child = tree.child[row];
            // Injected non-finite rows are never connectors; keep attachments finite-only so
            // the map stays within `finite_index.len()` (the modeled capacity).
            if finite_index
                .binary_search_by(|&raw| (raw as usize).cmp(&child))
                .is_err()
            {
                continue;
            }
            if attachment
                .insert(
                    child,
                    PointAttachment {
                        node,
                        departure_lambda: lambda,
                    },
                )
                .is_some()
            {
                return Err(invalid(format!(
                    "training point {child} has more than one condensed tree row"
                )));
            }
            let label = labels[child];
            if label >= 0 {
                let anchor = anchors.entry(label).or_insert(parent);
                if parent < *anchor {
                    *anchor = parent;
                }
            }
        }
        let mut points = Vec::with_capacity(finite_index.len());
        for &raw in finite_index {
            let point = attachment.get(&(raw as usize)).ok_or_else(|| {
                invalid(format!(
                    "finite training point {raw} has no condensed tree row"
                ))
            })?;
            points.push(*point);
        }

        // Anchor -> actual fitted label.
        let mut is_anchor: Vec<Option<i32>> = vec![None; nodes.len()];
        for (&label, &anchor) in &anchors {
            let node = node_of[&anchor];
            if let Some(other) = is_anchor[node].replace(label) {
                return Err(invalid(format!(
                    "fitted labels {other} and {label} share selected cluster {anchor}"
                )));
            }
        }
        // Iterative top-down traversal: a node's own anchor wins over inherited values,
        // matching the reference's ascending-anchor subtree overwrite (descendant anchors
        // have larger ids than their ancestors).
        let mut child_start = vec![0usize; nodes.len() + 1];
        for node in &nodes {
            if let Some((parent, _)) = node.parent {
                child_start[parent + 1] += 1;
            }
        }
        for index in 0..nodes.len() {
            child_start[index + 1] += child_start[index];
        }
        let mut fill = child_start.clone();
        let mut children = vec![0usize; nodes.len().saturating_sub(1)];
        for (index, node) in nodes.iter().enumerate() {
            if let Some((parent, _)) = node.parent {
                children[fill[parent]] = index;
                fill[parent] += 1;
            }
        }
        // Each node is pushed at most once (one parent edge each), so `nodes.len()` is exact.
        let mut stack = Vec::with_capacity(nodes.len());
        stack.push(0usize);
        let mut visited = 0usize;
        while let Some(node) = stack.pop() {
            visited += 1;
            stack.extend_from_slice(&children[child_start[node]..child_start[node + 1]]);
            if let Some(label) = is_anchor[node] {
                nodes[node].label = label;
                nodes[node].max_lambda = max_lambda[node];
            } else if let Some((parent, _)) = nodes[node].parent {
                nodes[node].label = nodes[parent].label;
                nodes[node].max_lambda = nodes[parent].max_lambda;
            }
        }
        if visited != nodes.len() {
            return Err(invalid("cluster tree is not connected to its root"));
        }

        Ok(Self {
            finite_index: finite_index.to_vec(),
            nodes,
            points,
            root: 0,
        })
    }

    /// Checked upper bound of retained heap bytes for `n_finite` points and `cluster_nodes`
    /// cluster nodes (cluster edges + 1 root, or 0 for the no-cluster state), using the
    /// actual private element sizes. Excludes build-time transients and caller-owned cores.
    pub fn allocation_bytes_bound(n_finite: usize, cluster_nodes: usize) -> Option<usize> {
        let index = n_finite.checked_mul(std::mem::size_of::<u32>())?;
        let points = n_finite.checked_mul(std::mem::size_of::<PointAttachment>())?;
        let nodes = cluster_nodes.checked_mul(std::mem::size_of::<ClusterNode>())?;
        index.checked_add(points)?.checked_add(nodes)
    }

    /// Actual retained heap bytes (by capacity) of this metadata.
    pub fn retained_bytes(&self) -> usize {
        self.finite_index.capacity() * std::mem::size_of::<u32>()
            + self.points.capacity() * std::mem::size_of::<PointAttachment>()
            + self.nodes.capacity() * std::mem::size_of::<ClusterNode>()
    }

    /// Number of retained cluster nodes (0 for the no-cluster-tree state).
    pub fn cluster_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// True when the tree has no `child_size > 1` rows: the reference all-noise shortcut.
    pub fn no_cluster_tree(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Number of compact finite training points (valid candidate ids are `0..n_finite`).
    pub fn n_finite(&self) -> usize {
        self.finite_index.len()
    }

    /// Ascending compact-to-raw training row mapping.
    pub fn finite_index(&self) -> &[u32] {
        &self.finite_index
    }

    /// Validate a full prediction core vector: one finite nonnegative value per finite row.
    pub fn validate_prediction_core(&self, prediction_core: &[f64]) -> Result<()> {
        if prediction_core.len() != self.n_finite() {
            return Err(invalid(format!(
                "prediction core has {} values for {} finite training points",
                prediction_core.len(),
                self.n_finite()
            )));
        }
        if let Some(value) = prediction_core
            .iter()
            .find(|value| !value.is_finite() || **value < 0.0)
        {
            return Err(HdbscanError::Numerical(format!(
                "invalid prediction core distance {value}"
            )));
        }
        Ok(())
    }

    /// Predict one query from its ordered candidate list of width `2*min_samples`.
    ///
    /// `ids` are compact training ids; `distances` are Euclidean f64 distances in the same
    /// order. The first minimum mutual-reachability candidate wins.
    pub fn predict_candidates(
        &self,
        ids: &[i32],
        distances: &[f64],
        prediction_core: &[f64],
        min_samples: usize,
    ) -> Result<PointPrediction> {
        let width = self.require_tree_and_width(min_samples)?;
        if prediction_core.len() != self.n_finite() {
            return Err(invalid(format!(
                "prediction core has {} values for {} finite training points",
                prediction_core.len(),
                self.n_finite()
            )));
        }
        self.predict_row(ids, distances, prediction_core, min_samples, width)
    }

    /// Predict a batch of row-major candidate rows of width `2*min_samples` produced by an
    /// exact squared-distance search, appending to `out` only if every row succeeds.
    ///
    /// Only the core values actually referenced are checked; validate the full vector
    /// once with [`Self::validate_prediction_core`] when it is built.
    pub fn predict_squared_batch(
        &self,
        ids: &[i32],
        squared: &[f32],
        prediction_core: &[f64],
        min_samples: usize,
        out: &mut PredictionColumns,
    ) -> Result<()> {
        let width = self.require_tree_and_width(min_samples)?;
        if prediction_core.len() != self.n_finite() {
            return Err(invalid(format!(
                "prediction core has {} values for {} finite training points",
                prediction_core.len(),
                self.n_finite()
            )));
        }
        if ids.len() != squared.len() || !ids.len().is_multiple_of(width) {
            return Err(invalid(format!(
                "candidate batch has {} ids and {} distances for width {width}",
                ids.len(),
                squared.len()
            )));
        }
        let rows = ids.len() / width;
        let mut local = PredictionColumns::default();
        local.reserve(rows);
        let mut distances = vec![0.0_f64; width];
        for (row_ids, row_squared) in ids.chunks_exact(width).zip(squared.chunks_exact(width)) {
            for (distance, &value) in distances.iter_mut().zip(row_squared) {
                *distance = euclidean_from_squared(value)?;
            }
            local.push(self.predict_row(
                row_ids,
                &distances,
                prediction_core,
                min_samples,
                width,
            )?);
        }
        out.reserve(rows);
        out.labels.extend_from_slice(&local.labels);
        out.probabilities.extend_from_slice(&local.probabilities);
        out.connecting_points
            .extend_from_slice(&local.connecting_points);
        Ok(())
    }

    fn require_tree_and_width(&self, min_samples: usize) -> Result<usize> {
        if self.no_cluster_tree() {
            return Err(invalid(
                "model has no cluster tree; use the no-cluster-tree shortcut",
            ));
        }
        candidate_width(min_samples, self.n_finite())
    }

    fn predict_row(
        &self,
        ids: &[i32],
        distances: &[f64],
        prediction_core: &[f64],
        min_samples: usize,
        width: usize,
    ) -> Result<PointPrediction> {
        if ids.len() != width || distances.len() != width {
            return Err(invalid(format!(
                "candidate row has {} ids and {} distances, expected {width}",
                ids.len(),
                distances.len()
            )));
        }
        if let Some(value) = distances
            .iter()
            .find(|value| !value.is_finite() || **value < 0.0)
        {
            return Err(HdbscanError::Numerical(format!(
                "invalid candidate distance {value}"
            )));
        }
        let query_core = distances[min_samples];
        let mut best: Option<(usize, f64)> = None;
        for (position, (&id, &distance)) in ids.iter().zip(distances).enumerate() {
            let compact = usize::try_from(id)
                .ok()
                .filter(|&id| id < self.n_finite())
                .ok_or_else(|| HdbscanError::Numerical(format!("invalid candidate id {id}")))?;
            let core = prediction_core[compact];
            if !core.is_finite() || core < 0.0 {
                return Err(HdbscanError::Numerical(format!(
                    "invalid prediction core distance {core}"
                )));
            }
            let mrd = core.max(query_core).max(distance);
            if best.is_none_or(|(_, current)| mrd < current) {
                best = Some((position, mrd));
            }
        }
        let (position, mrd) = best.ok_or_else(|| invalid("empty candidate row"))?;
        let connector = ids[position] as usize;
        let lambda = if mrd > 0.0 { 1.0 / mrd } else { f64::MAX };

        let point = self.points[connector];
        let mut cluster = point.node;
        if point.departure_lambda > lambda {
            let root_id = self.nodes[self.root].id;
            while self.nodes[cluster].id > root_id {
                match self.nodes[cluster].parent {
                    Some((parent, birth)) if birth >= lambda => cluster = parent,
                    _ => break,
                }
            }
        }
        let node = self.nodes[cluster];
        let (label, probability) = if node.label < 0 {
            (-1, 0.0)
        } else if node.max_lambda > 0.0 {
            (node.label, node.max_lambda.min(lambda) / node.max_lambda)
        } else {
            (node.label, 1.0)
        };
        Ok(PointPrediction {
            label,
            probability,
            connecting_point: self.finite_index[connector] as i32,
        })
    }
}

fn checked_column_bytes<T>(count: usize, name: &'static str) -> Result<usize> {
    count
        .checked_mul(std::mem::size_of::<T>())
        .ok_or(HdbscanError::Overflow(name))
}

fn reserve_column<T>(column: &mut Vec<T>, count: usize, name: &'static str) -> Result<()> {
    checked_column_bytes::<T>(count, name)?;
    column
        .try_reserve_exact(count)
        .map_err(|_| invalid(format!("unable to reserve {name} column")))
}

fn invalid(message: impl Into<String>) -> HdbscanError {
    HdbscanError::InvalidInput(message.into())
}

fn validate_finite_index(finite_index: &[u32], raw_n: usize) -> Result<()> {
    if finite_index.is_empty() {
        return Err(invalid(
            "prediction requires at least one finite training point",
        ));
    }
    for pair in finite_index.windows(2) {
        if pair[0] >= pair[1] {
            return Err(invalid("finite index must be strictly ascending"));
        }
    }
    if finite_index[finite_index.len() - 1] as usize >= raw_n {
        return Err(invalid("finite index refers past the raw row count"));
    }
    Ok(())
}

fn validate_labels(labels: &[i32], finite_index: &[u32]) -> Result<()> {
    let mut finite = finite_index.iter().peekable();
    for (raw, &label) in labels.iter().enumerate() {
        let is_finite = finite.next_if(|&&index| index as usize == raw).is_some();
        if label < -1 || (!is_finite && label != -1) {
            return Err(invalid(format!(
                "invalid fitted label {label} for row {raw}"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod portable_state_tests {
    use super::*;

    fn clustered_state() -> PredictionHierarchyState {
        PredictionHierarchyState {
            finite_index: vec![3, 9],
            node_ids: vec![10, 11],
            node_parent_indices: vec![u64::MAX, 0],
            node_birth_lambdas: vec![0.0, 1.0],
            node_labels: vec![-1, 4],
            node_max_lambdas: vec![0.0, 4.0],
            point_node_indices: vec![1, 1],
            point_departure_lambdas: vec![0.1, 0.2],
        }
    }

    fn three_node_state() -> PredictionHierarchyState {
        PredictionHierarchyState {
            finite_index: vec![3, 9],
            node_ids: vec![10, 11, 12],
            node_parent_indices: vec![u64::MAX, 0, 0],
            node_birth_lambdas: vec![0.0, 1.0, 1.5],
            node_labels: vec![-1, 4, 5],
            node_max_lambdas: vec![0.0, 4.0, 5.0],
            point_node_indices: vec![1, 2],
            point_departure_lambdas: vec![0.1, 0.2],
        }
    }

    #[test]
    fn portable_state_round_trip_preserves_prediction_authority() {
        let hierarchy = PredictionHierarchy::from_state(clustered_state()).unwrap();
        let exported = hierarchy.export_state().unwrap();
        assert_eq!(exported, clustered_state());

        let prediction = hierarchy
            .predict_candidates(&[0, 1], &[1.0, 2.0], &[0.0, 0.0], 1)
            .unwrap();
        assert_eq!(prediction.label, 4);
        assert_eq!(prediction.probability, 0.125);
        assert_eq!(prediction.connecting_point, 3);
    }

    #[test]
    fn portable_state_empty_shortcut_is_canonical() {
        let empty = PredictionHierarchyState {
            finite_index: Vec::new(),
            node_ids: Vec::new(),
            node_parent_indices: Vec::new(),
            node_birth_lambdas: Vec::new(),
            node_labels: Vec::new(),
            node_max_lambdas: Vec::new(),
            point_node_indices: Vec::new(),
            point_departure_lambdas: Vec::new(),
        };
        let hierarchy = PredictionHierarchy::from_state(empty.clone()).unwrap();
        assert!(hierarchy.no_cluster_tree());
        assert_eq!(hierarchy.export_state().unwrap(), empty);

        // Root-only fitted trees retain a finite map in memory but must export canonically empty.
        let root_only = CondensedTree {
            parent: vec![2, 2],
            child: vec![0, 1],
            lambda: vec![1.0, 2.0],
            child_size: vec![1.0, 1.0],
        };
        let hierarchy = PredictionHierarchy::new(&root_only, &[-1, -1], &[0, 1]).unwrap();
        assert!(hierarchy.no_cluster_tree());
        assert_eq!(hierarchy.export_state().unwrap(), empty);
    }

    #[test]
    fn portable_state_rejects_malformed_columns_and_hierarchy_invariants() {
        let mut state = clustered_state();
        state.node_labels.pop();
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.finite_index[1] = state.finite_index[0];
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.node_parent_indices[1] = 1;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.node_birth_lambdas[1] = f64::NAN;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.node_labels[1] = -1;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.node_ids.push(12);
        state.node_parent_indices.push(0);
        state.node_birth_lambdas.push(1.0);
        state.node_labels.push(4);
        state.node_max_lambdas.push(4.0);
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.point_node_indices[0] = 2;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.node_parent_indices[0] = 0;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.point_departure_lambdas[0] = f64::NEG_INFINITY;
        assert!(PredictionHierarchy::from_state(state).is_err());

        let mut state = clustered_state();
        state.point_departure_lambdas[0] = f64::INFINITY;
        assert!(PredictionHierarchy::from_state(state.clone()).is_ok());
        assert_eq!(
            PredictionHierarchy::from_state(state.clone())
                .unwrap()
                .export_state()
                .unwrap(),
            state
        );
    }

    #[test]
    fn portable_state_rejects_table_of_invalid_values() {
        enum InvalidField {
            DuplicateNodeIds,
            ParentIdDoesNotDescend,
            InheritedMaximumBitsDiffer,
            DuplicateAnchors,
            BirthLambda(f64),
            MaximumLambda(f64),
            DepartureLambda(f64),
            OutOfRangeParent,
            MultipleRootSentinels,
            RawIdTooLarge,
            RawIdsOutOfOrder,
        }

        let cases = [
            ("duplicate node ids", InvalidField::DuplicateNodeIds),
            (
                "parent id does not descend",
                InvalidField::ParentIdDoesNotDescend,
            ),
            (
                "inherited maximum has different bits",
                InvalidField::InheritedMaximumBitsDiffer,
            ),
            ("duplicate anchor labels", InvalidField::DuplicateAnchors),
            ("NaN birth lambda", InvalidField::BirthLambda(f64::NAN)),
            (
                "negative-infinity birth lambda",
                InvalidField::BirthLambda(f64::NEG_INFINITY),
            ),
            ("negative birth lambda", InvalidField::BirthLambda(-1.0)),
            ("NaN maximum lambda", InvalidField::MaximumLambda(f64::NAN)),
            (
                "negative-infinity maximum lambda",
                InvalidField::MaximumLambda(f64::NEG_INFINITY),
            ),
            ("negative maximum lambda", InvalidField::MaximumLambda(-1.0)),
            (
                "NaN departure lambda",
                InvalidField::DepartureLambda(f64::NAN),
            ),
            (
                "negative-infinity departure lambda",
                InvalidField::DepartureLambda(f64::NEG_INFINITY),
            ),
            (
                "negative departure lambda",
                InvalidField::DepartureLambda(-1.0),
            ),
            ("out-of-range parent", InvalidField::OutOfRangeParent),
            (
                "multiple root sentinels",
                InvalidField::MultipleRootSentinels,
            ),
            ("raw id exceeds i32", InvalidField::RawIdTooLarge),
            ("raw ids out of order", InvalidField::RawIdsOutOfOrder),
        ];

        for (case, field) in cases {
            let mut state = clustered_state();
            match field {
                InvalidField::DuplicateNodeIds => {
                    state.node_ids[1] = state.node_ids[0];
                }
                InvalidField::ParentIdDoesNotDescend => {
                    state = three_node_state();
                    state.node_ids = vec![10, 12, 11];
                    state.node_parent_indices[1] = 2;
                    state.node_parent_indices[2] = 0;
                    state.node_labels = vec![-1, 4, 5];
                    state.node_max_lambdas = vec![0.0, 4.0, 5.0];
                    state.node_birth_lambdas = vec![0.0, 1.0, 1.5];
                    state.point_node_indices = vec![1, 2];
                    state.point_departure_lambdas = vec![0.1, 0.2];
                    // Node 2 (id 11) now has node 1 (id 12) as parent.
                    state.node_parent_indices[2] = 1;
                }
                InvalidField::InheritedMaximumBitsDiffer => {
                    state.node_labels = vec![4, 4];
                    state.node_max_lambdas = vec![0.0, -0.0];
                }
                InvalidField::DuplicateAnchors => {
                    state = three_node_state();
                    state.node_labels = vec![-1, 4, 4];
                    state.node_max_lambdas = vec![0.0, 4.0, 4.0];
                }
                InvalidField::BirthLambda(value) => state.node_birth_lambdas[1] = value,
                InvalidField::MaximumLambda(value) => state.node_max_lambdas[1] = value,
                InvalidField::DepartureLambda(value) => state.point_departure_lambdas[0] = value,
                InvalidField::OutOfRangeParent => state.node_parent_indices[1] = 2,
                InvalidField::MultipleRootSentinels => state.node_parent_indices[1] = u64::MAX,
                InvalidField::RawIdTooLarge => {
                    state.finite_index[1] = i32::MAX as u32 + 1;
                }
                InvalidField::RawIdsOutOfOrder => state.finite_index = vec![9, 3],
            }
            assert!(
                PredictionHierarchy::from_state(state).is_err(),
                "expected portable hierarchy import to reject {case}"
            );
        }
    }
}
