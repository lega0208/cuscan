"""Generate HDBSCAN ``approximate_predict`` reference fixtures.

The oracle is the original ``hdbscan`` package (``hdbscan.prediction``), never
this repository's ``generate_fast_hdbscan.py`` fitting oracle nor the separate
``fast_hdbscan`` package. Training inputs are rounded to float32 and promoted
to float64 before reaching the reference so that preprocessing differences are
not confused with prediction differences.

Three layers are recorded:

* ``controlled``: hand-built condensed trees with fixed candidate arrays and
  core distances, evaluated with original private helpers. These pin exact
  branch behavior (MRD first-minimum, zero MRD, departure/birth boundaries,
  infinite maxima, unmapped clusters, selected ancestors without point rows).
* ``fitted``: original estimators fit with ``prediction_data=True``. Each case
  records the fitted raw hierarchy, labels, prediction cores, candidate arrays,
  label maps/maxima and expected predictions from both the estimator and the
  test-only same-hierarchy adapter (they must agree).
* ``core_rank`` / ``width``: core-rank conventions and width failures.

Usage: ``python generate_hdbscan_prediction.py [--check]``. ``--check``
regenerates in memory and fails if the stored JSON differs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import hdbscan
import hdbscan.prediction as hp
import numpy as np
from hdbscan.plots import CondensedTree

FIXTURE_PATH = Path(__file__).with_name("hdbscan_prediction.json")
FORMAT_VERSION = 1
RAW_TREE_DTYPE = np.dtype(
    [("parent", np.intp), ("child", np.intp), ("lambda_val", np.float64), ("child_size", np.intp)]
)


# --------------------------------------------------------------------------
# JSON encoding helpers: non-finite floats are encoded as strings.


def enc_float(value: float) -> float | str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def dec_float(value: float | str) -> float:
    return float(value)


def enc_floats(values: Any) -> list[float | str]:
    return [enc_float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def enc_matrix(values: Any) -> list[list[float | str]]:
    array = np.asarray(values, dtype=np.float64)
    return [enc_floats(row) for row in array]


def enc_tree(raw_tree: np.ndarray) -> dict[str, list]:
    return {
        "parent": [int(v) for v in raw_tree["parent"]],
        "child": [int(v) for v in raw_tree["child"]],
        "lambda_val": enc_floats(raw_tree["lambda_val"]),
        "child_size": [int(v) for v in raw_tree["child_size"]],
    }


def dec_tree(tree: dict[str, list]) -> np.ndarray:
    raw = np.empty(len(tree["parent"]), dtype=RAW_TREE_DTYPE)
    raw["parent"] = tree["parent"]
    raw["child"] = tree["child"]
    raw["lambda_val"] = [dec_float(v) for v in tree["lambda_val"]]
    raw["child_size"] = tree["child_size"]
    return raw


def enc_map(mapping: dict) -> dict[str, Any]:
    return {
        str(int(k)): (enc_float(v) if isinstance(v, (float, np.floating)) else int(v))
        for k, v in sorted(mapping.items(), key=lambda item: int(item[0]))
    }


# --------------------------------------------------------------------------
# Reference identity.


def reference_identity() -> dict[str, str]:
    source = Path(hp.__file__).read_bytes()
    return {
        "package": "hdbscan",
        "version": importlib.metadata.version("hdbscan"),
        "prediction_py_sha256": hashlib.sha256(source).hexdigest(),
        "numpy": np.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
    }


# --------------------------------------------------------------------------
# Test-only same-hierarchy adapter (shared semantics with the Python tests).


def same_hierarchy_model(
    data: np.ndarray, raw_tree: np.ndarray, labels: np.ndarray, min_samples: int
) -> types.SimpleNamespace:
    """Original PredictionData over a supplied raw tree and labels.

    ``data`` must be compact finite training coordinates whose row IDs match the
    point children of ``raw_tree``; ``labels`` is indexed by those IDs.
    """
    raw_tree = np.asarray(raw_tree).astype(RAW_TREE_DTYPE)
    labels = np.asarray(labels, dtype=np.intp)
    condensed = CondensedTree(raw_tree, labels)
    prediction_data = hp.PredictionData(
        np.asarray(data, dtype=np.float64), condensed, min_samples, tree_type="kdtree"
    )
    return types.SimpleNamespace(
        prediction_data_=prediction_data,
        condensed_tree_=condensed,
        min_samples=min_samples,
        min_cluster_size=min_samples,
    )


def rounded(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def tie_annotations(distances: np.ndarray, indices: np.ndarray, cores: np.ndarray, s: int) -> list:
    notes = []
    for row, (dist, idx) in enumerate(zip(distances, indices, strict=True)):
        mrd = np.maximum(np.maximum(cores[idx], dist[s]), dist)
        euclid_tie = bool(np.unique(dist).size != dist.size)
        minimal = mrd == mrd.min()
        mrd_min_tie = bool(np.count_nonzero(minimal) > 1)
        # The connector depends on candidate order only when the first MRD
        # minimum shares its Euclidean distance with another minimal candidate.
        first = int(np.argmax(minimal))
        order_sensitive = bool(np.count_nonzero(minimal & (dist == dist[first])) > 1)
        if euclid_tie or mrd_min_tie:
            notes.append(
                {
                    "query": row,
                    "distance_tie": euclid_tie,
                    "mrd_minimum_tie": mrd_min_tie,
                    "connector_order_sensitive": order_sensitive,
                }
            )
    return notes


# --------------------------------------------------------------------------
# Controlled hierarchy layer.


def tree_rows(rows: list[tuple[int, int, float, int]]) -> np.ndarray:
    return np.array(rows, dtype=RAW_TREE_DTYPE)


def controlled_trees() -> dict[str, dict[str, Any]]:
    inf = float("inf")
    # Points 0..8, root 9. Cluster 10: points 0-3. Cluster 11: point 4 plus
    # cluster children 12 (points 5,6) and 13 (points 7,8). Labels 0 and 1.
    nested = tree_rows(
        [
            (9, 10, 1.0, 4),
            (9, 11, 1.0, 5),
            (10, 0, 2.0, 1),
            (10, 1, 2.0, 1),
            (10, 2, 3.0, 1),
            (10, 3, 4.0, 1),
            (11, 4, 1.5, 1),
            (11, 12, 2.0, 2),
            (11, 13, 2.0, 2),
            (12, 5, 3.0, 1),
            (12, 6, 5.0, 1),
            (13, 7, 3.0, 1),
            (13, 8, 6.0, 1),
        ]
    )
    nested_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Infinite maximum: cluster 10 contains exact duplicates (lambda inf).
    infinite = tree_rows(
        [
            (8, 9, 1.0, 4),
            (8, 10, 1.0, 4),
            (9, 0, 2.0, 1),
            (9, 1, 2.0, 1),
            (9, 2, 3.0, 1),
            (9, 3, 4.0, 1),
            (10, 4, inf, 1),
            (10, 5, inf, 1),
            (10, 6, inf, 1),
            (10, 7, inf, 1),
        ]
    )
    infinite_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    # Selected ancestor 11 has no direct point rows; original positional
    # selection anchors label 1 at child segment 12 and leaves sibling 13
    # unmapped.
    anchorless = tree_rows(
        [
            (9, 10, 1.0, 4),
            (9, 11, 1.0, 5),
            (10, 0, 2.0, 1),
            (10, 1, 2.0, 1),
            (10, 2, 3.0, 1),
            (10, 3, 4.0, 1),
            (11, 12, 2.0, 2),
            (11, 13, 2.0, 3),
            (12, 4, 3.0, 1),
            (12, 5, 5.0, 1),
            (13, 6, 3.0, 1),
            (13, 7, 4.0, 1),
            (13, 8, 6.0, 1),
        ]
    )
    anchorless_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1]
    return {
        "nested": {"raw_tree": nested, "labels": nested_labels},
        "infinite_max": {"raw_tree": infinite, "labels": infinite_labels},
        "anchorless_selected_ancestor": {"raw_tree": anchorless, "labels": anchorless_labels},
    }


def lam_query(neighbor: int, lam: float) -> dict[str, Any]:
    """s=1 width-2 candidate list whose query core sets MRD to 1/lam."""
    d = 1.0 / lam
    return {"indices": [neighbor, neighbor], "distances": [d * 0.5, d]}


def controlled_queries() -> list[dict[str, Any]]:
    big = 1.0e6
    q = []

    def add(name, tree, s, cores, indices, distances, note):
        q.append(
            {
                "name": name,
                "tree": tree,
                "min_samples": s,
                "core_distances": cores,
                "indices": indices,
                "distances": distances,
                "note": note,
            }
        )

    zero9 = [0.0] * 9
    zero8 = [0.0] * 8
    for name, nb, lam, note in [
        ("departure_equal", 0, 2.0, "neighbor row lambda == query lambda: no ascent"),
        ("departure_less", 0, 3.0, "neighbor row lambda < query lambda: no ascent, strength 0.75"),
        ("departure_greater_birth_less", 0, 1.25, "ascend check stops: birth 1 < 1.25"),
        (
            "birth_equal_root",
            0,
            1.0,
            "birth == lambda ascends to root: unmapped noise with connector",
        ),
        ("birth_greater_root", 0, 0.8, "birth > lambda ascends to root"),
        ("descendant_stay", 5, 2.5, "leaf 12 inherits label 1 and ancestor max 2"),
        ("descendant_birth_equal", 5, 2.0, "birth 2 == lambda ascends to selected 11"),
        ("descendant_ascend_selected", 5, 1.5, "ascends to 11, strength 0.75"),
        ("descendant_to_root", 5, 0.5, "ascends through 11 to root"),
    ]:
        c = lam_query(nb, lam)
        add(name, "nested", 1, zero9, c["indices"], c["distances"], note)
    add(
        "zero_mrd",
        "nested",
        1,
        zero9,
        [3, 2],
        [0.0, 0.0],
        "zero MRD yields lambda f64::MAX; strength min(max,MAX)/max = 1",
    )
    add(
        "mrd_tie_first_minimum",
        "nested",
        1,
        zero9,
        [5, 0],
        [0.4, 0.5],
        "equal MRD 0.5 with different Euclidean distances picks first candidate (5), not lowest ID",
    )
    add(
        "neighbor_core_dominates",
        "nested",
        1,
        [1.0] + [0.0] * 8,
        [0, 5],
        [0.1, 0.2],
        "neighbor core 1.0 raises candidate 0 MRD; candidate 5 wins",
    )
    add(
        "query_core_rank",
        "nested",
        2,
        zero9,
        [0, 5, 1, 6],
        [0.1, 0.2, 0.3, big],
        "query core is distances[s]=0.3 (not the last candidate); MRD [0.3,0.3,0.3,big] -> first",
    )
    add(
        "raw_distance_dominates",
        "nested",
        1,
        zero9,
        [5, 0],
        [0.4, 0.4 + 1e-9],
        "candidate distances dominate; untied first minimum",
    )
    add(
        "infinite_max_zero_mrd",
        "infinite_max",
        1,
        zero8,
        [4, 5],
        [0.0, 0.0],
        "duplicate cluster max inf; lambda f64::MAX; MAX/inf == 0",
    )
    add(
        "infinite_max_finite",
        "infinite_max",
        1,
        zero8,
        [4, 4],
        [0.1, 0.2],
        "finite lambda 5 divided by inf max == 0",
    )
    add(
        "anchorless_mapped_child",
        "anchorless_selected_ancestor",
        1,
        zero9,
        [4, 4],
        [0.1, 0.2],
        "child anchor 12 maps to label 1",
    )
    add(
        "anchorless_unmapped_sibling",
        "anchorless_selected_ancestor",
        1,
        zero9,
        [6, 6],
        [0.1, 0.2],
        "sibling 13 unmapped by original positional selection -> -1, connector kept",
    )
    add(
        "anchorless_ascend_to_selected",
        "anchorless_selected_ancestor",
        1,
        zero9,
        [6, 6],
        [0.5, 0.6],
        "lambda 1/0.6 < birth 2 ascends 13->11; 11 unmapped in original -> -1",
    )
    return q


def build_controlled() -> dict[str, Any]:
    trees = controlled_trees()
    out_trees = {}
    models = {}
    for name, spec in trees.items():
        raw = spec["raw_tree"]
        labels = np.asarray(spec["labels"], dtype=np.intp)
        n = labels.size
        data = np.arange(n, dtype=np.float64)[:, None] * 10.0
        model = same_hierarchy_model(data, raw, labels, 1)
        pd = model.prediction_data_
        models[name] = model
        out_trees[name] = {
            "raw_tree": enc_tree(raw),
            "labels": labels.tolist(),
            "selected_clusters": sorted(int(c) for c in model.condensed_tree_._select_clusters()),
            "cluster_map": enc_map(pd.cluster_map),
            "max_lambdas": enc_map({k: float(v) for k, v in pd.max_lambdas.items()}),
            "cluster_tree_rows": int(pd.cluster_tree.shape[0]),
        }
    queries = []
    for query in controlled_queries():
        model = models[query["tree"]]
        pd = model.prediction_data_
        cores = np.asarray(query["core_distances"], dtype=np.float64)
        idx = np.asarray(query["indices"], dtype=np.intp)
        dist = np.asarray(query["distances"], dtype=np.float64)
        s = query["min_samples"]
        assert idx.size == 2 * s
        neighbor, lam = hp._find_neighbor_and_lambda(idx, dist, cores, s)
        label, prob, connector = hp._find_cluster_and_probability(
            model.condensed_tree_,
            pd.cluster_tree,
            idx,
            dist,
            cores,
            pd.cluster_map,
            pd.max_lambdas,
            s,
        )
        assert connector == neighbor
        queries.append(
            {
                **query,
                "core_distances": enc_floats(cores),
                "distances": enc_floats(dist),
                "expected": {
                    "neighbor": int(neighbor),
                    "lambda": enc_float(lam),
                    "label": int(label),
                    "strength": enc_float(prob),
                },
            }
        )
    return {"trees": out_trees, "queries": queries}


# --------------------------------------------------------------------------
# Fitted layer.


def fitted_specs() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20261027)
    specs = []

    group_a = np.array([0.0, 0.13, 0.29, 0.41, 0.58, 0.72, 0.87, 1.03, 1.16, 1.34])
    group_b = group_a * 1.07 + 20.0
    specs.append(
        {
            "id": "separated_groups_1d",
            "train": np.concatenate([group_a, group_b])[:, None],
            "queries": np.array([[0.51], [0.95], [20.6], [21.27], [10.4], [500.0], [0.29]]),
            "params": {"min_cluster_size": 5, "min_samples": 3},
            "independent_fit": True,
        }
    )

    centers = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 12.0]])
    blobs = np.concatenate([c + rng.normal(scale=0.8, size=(60, 2)) for c in centers])
    blob_queries = np.concatenate(
        [c + rng.normal(scale=0.6, size=(5, 2)) for c in centers]
        + [np.array([[6.0, 6.0], [40.0, -40.0]]), blobs[[3, 75, 140]]]
    )
    specs.append(
        {
            "id": "blobs_2d",
            "train": blobs,
            "queries": blob_queries,
            "params": {"min_cluster_size": 10, "min_samples": 5},
            "independent_fit": True,
        }
    )

    dense = rng.normal(scale=0.3, size=(40, 3))
    sparse = rng.normal(scale=2.5, size=(40, 3)) + np.array([15.0, 0.0, 0.0])
    specs.append(
        {
            "id": "varying_density_3d",
            "train": np.concatenate([dense, sparse]),
            "queries": np.concatenate(
                [
                    rng.normal(scale=0.3, size=(4, 3)),
                    rng.normal(scale=2.5, size=(4, 3)) + np.array([15.0, 0.0, 0.0]),
                    np.array([[7.5, 0.0, 0.0], [-30.0, 30.0, 30.0]]),
                ]
            ),
            "params": {"min_cluster_size": 8, "min_samples": 4},
            "independent_fit": True,
        }
    )

    dup = np.concatenate([np.full((8, 1), 5.0), group_a[:, None] + 40.0])
    specs.append(
        {
            "id": "duplicates_infinite_max",
            "train": dup,
            "queries": np.array([[5.0], [5.05], [40.6], [100.0]]),
            "params": {"min_cluster_size": 4, "min_samples": 2},
            "independent_fit": True,
            "note": "duplicate group has max lambda inf; exact duplicate query has strength 0",
        }
    )

    specs.append(
        {
            "id": "root_only_single_cluster",
            "train": (np.linspace(0.0, 1.0, 20) ** 1.1)[:, None],
            "queries": np.array([[0.5], [50.0]]),
            "params": {"min_cluster_size": 4, "min_samples": 2, "allow_single_cluster": True},
            "independent_fit": True,
            "note": "root-only selection: empty cluster tree shortcut despite training cluster 0",
        }
    )

    specs.append(
        {
            "id": "no_cluster_tree",
            "train": np.array([[0.0], [0.11], [0.23], [0.37], [0.52], [0.68], [0.81], [0.97]]),
            "queries": np.array([[0.3], [9.0]]),
            "params": {"min_cluster_size": 6, "min_samples": 2},
            "independent_fit": True,
            "note": "no split survives min_cluster_size: empty cluster tree shortcut",
        }
    )

    mixed = np.concatenate([group_a, group_b])[:, None]
    raw_mixed = np.full((mixed.shape[0] + 4, 1), np.nan)
    nan_rows = [0, 6, 13, 23]
    finite_rows = [i for i in range(raw_mixed.shape[0]) if i not in nan_rows]
    raw_mixed[finite_rows] = mixed
    raw_mixed[13, 0] = np.inf
    specs.append(
        {
            "id": "mixed_nonfinite_training",
            "train": raw_mixed,
            "queries": np.array([[0.51], [21.27], [500.0], [0.0]]),
            "params": {"min_cluster_size": 5, "min_samples": 3},
            "independent_fit": True,
            "note": "extension: compact finite rows; connectors translated to raw row IDs",
        }
    )
    return specs


def fit_reference(train_raw: np.ndarray, params: dict[str, Any]):
    finite = np.isfinite(train_raw).all(axis=1)
    finite_index = np.flatnonzero(finite)
    compact = rounded(train_raw[finite])
    model = hdbscan.HDBSCAN(prediction_data=True, **params).fit(compact)
    return model, compact, finite_index


def build_fitted_case(spec: dict[str, Any]) -> dict[str, Any]:
    params = spec["params"]
    s = params.get("min_samples") or params["min_cluster_size"]
    train_raw = np.asarray(spec["train"], dtype=np.float64)
    train32 = train_raw.astype(np.float32)
    queries = rounded(spec["queries"])
    model, compact, finite_index = fit_reference(train32.astype(np.float64), params)
    pd = model.prediction_data_
    raw_tree = np.asarray(model.condensed_tree_._raw_tree)
    labels = np.asarray(model.labels_, dtype=np.intp)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = hp.approximate_predict(model, queries, return_connecting_points=True)
    est_warnings = [str(w.message) for w in caught]

    adapter = same_hierarchy_model(compact, raw_tree, labels, s)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        same = hp.approximate_predict(adapter, queries, return_connecting_points=True)
    for a, b in zip(est, same, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    empty_tree = pd.cluster_tree.shape[0] == 0
    case: dict[str, Any] = {
        "id": spec["id"],
        "note": spec.get("note", ""),
        "params": params,
        "resolved_min_samples": int(s),
        "train_float32": enc_matrix(train32),
        "finite_index": finite_index.tolist(),
        "queries_float32": enc_matrix(queries),
        "raw_tree": enc_tree(raw_tree),
        "labels_compact": labels.tolist(),
        "selected_clusters": sorted(int(c) for c in model.condensed_tree_._select_clusters()),
        "cluster_map": enc_map(pd.cluster_map),
        "max_lambdas": enc_map({k: float(v) for k, v in pd.max_lambdas.items()}),
        "cluster_tree_rows": int(pd.cluster_tree.shape[0]),
        "no_cluster_tree": bool(empty_tree),
        "prediction_core_distances": enc_floats(pd.core_distances),
        "estimator_warnings": est_warnings,
        "independent_fit": bool(spec["independent_fit"]),
    }
    if not empty_tree:
        dist, idx = pd.tree.query(queries, k=2 * s)
        case["candidate_indices_compact"] = idx.tolist()
        case["candidate_distances"] = enc_matrix(dist)
        case["tie_annotations"] = tie_annotations(dist, idx, pd.core_distances, s)
    labels_out, strengths, connectors = (np.asarray(v) for v in est)
    raw_connectors = np.where(connectors >= 0, finite_index[np.maximum(connectors, 0)], -1)
    case["expected"] = {
        "labels": labels_out.astype(np.int64).tolist(),
        "strengths": enc_floats(strengths),
        "connecting_points_compact": connectors.astype(np.int64).tolist(),
        "connecting_points_raw": raw_connectors.astype(np.int64).tolist(),
        "same_hierarchy_adapter_matches_estimator": True,
    }
    return case


# --------------------------------------------------------------------------
# Core-rank and width layer.


def build_core_rank() -> dict[str, Any]:
    from sklearn.neighbors import KDTree

    train = np.array([[0.0], [1.0], [3.0], [10.0]])
    s = 2
    tree = KDTree(train)
    inclusive = tree.query(train, k=s)[0][:, -1]
    nonself = tree.query(train, k=s + 1)[0][:, -1]
    query = np.array([[2.0]])
    qdist = tree.query(query, k=2 * s)[0][0]
    return {
        "train": enc_matrix(train),
        "min_samples": s,
        "prediction_core_inclusive_self": enc_floats(inclusive),
        "fit_core_nonself": enc_floats(nonself),
        "query": enc_matrix(query),
        "query_candidate_distances": enc_floats(qdist),
        "query_core": enc_float(qdist[s]),
        "note": (
            "training prediction core = inclusive-self rank s (nonself s-1); "
            "query core = distances[s]"
        ),
    }


def build_width() -> list[dict[str, Any]]:
    base = np.array([[0.0], [0.1], [0.25], [0.4], [5.0], [5.12], [5.3], [5.41]])
    out = []
    # Two separated groups: nonempty cluster tree, so width is checked.
    for n, s, mcs in [(8, 4, 4), (8, 5, 4), (8, 9, 4)]:
        record: dict[str, Any] = {"n_finite": n, "min_samples": s, "min_cluster_size": mcs}
        try:
            model = hdbscan.HDBSCAN(prediction_data=False, min_cluster_size=mcs, min_samples=1).fit(
                base[:n]
            )
            raw_tree = np.asarray(model.condensed_tree_._raw_tree)
            adapter = same_hierarchy_model(base[:n], raw_tree, model.labels_, s)
            record["cluster_tree_rows"] = int(adapter.prediction_data_.cluster_tree.shape[0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hp.approximate_predict(adapter, np.array([[0.2]]))
            record["reference_outcome"] = "ok"
        except ValueError as error:
            record["reference_outcome"] = "ValueError"
            record["reference_message"] = str(error)
        record["project_outcome"] = "ok" if 2 * s <= n else "ValueError"
        out.append(record)
    return out


def generate() -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return {
            "format_version": FORMAT_VERSION,
            "reference": reference_identity(),
            "float_encoding": "JSON numbers; non-finite as strings 'inf'/'-inf'/'nan'; "
            "training/query coordinates are float32 values promoted to float64",
            "core_rank": build_core_rank(),
            "width": build_width(),
            "controlled": build_controlled(),
            "fitted": [build_fitted_case(spec) for spec in fitted_specs()],
        }


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = dumps(generate())
    if args.check:
        if FIXTURE_PATH.read_text() != text:
            print("hdbscan_prediction.json is stale", file=sys.stderr)
            return 1
        return 0
    FIXTURE_PATH.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
