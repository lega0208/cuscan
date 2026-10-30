#!/usr/bin/env python3
"""Generate bounded reference fixtures from the pinned fast_hdbscan internals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import tempfile
import tomllib
from pathlib import Path

import numpy as np
from sklearn.datasets import make_blobs, make_moons

ROOT = Path(__file__).resolve().parents[4]
FIXTURES = Path(__file__).resolve().parent
MANIFEST = FIXTURES / "manifest.json"
SCHEMA = FIXTURES / "schema.json"
COMMIT = "43c573ba61043a6ad6f10b396656bfaa7aaeead8"
EXPECTED = {
    "version": "0.3.2",
    "repository": "https://github.com/TutteInstitute/fast_hdbscan",
    "sdist": "sha256:248e09202eda04da4b84dd819de3cc714469912205bb447ad4c7136db580f205",
    "wheel": "sha256:619b42963935e6c3ab1e51de7895dd35260f3ef9ebe4627ff4317073fe51a667",
}
RELEASE_BLOBS = {
    "boruvka.py": "747d89e477e715ac9afabe56379f8782dc97bca0",
    "cluster_trees.py": "0455190d04ec2ac6db6a51798d1a5fbdbe0522c7",
    "hdbscan.py": "9c5e42a8982133081fcb1e9673bc4ade0c69dba9",
    "numba_kdtree.py": "d6c9f57706345dc538948edc2eb824b54f861187",
}
COMPARISON_COMMIT_BLOBS = {
    "boruvka.py": "747d89e477e715ac9afabe56379f8782dc97bca0",
    "cluster_trees.py": "66adcdb6424db913c918d52756bb7065f85e66a5",
    "hdbscan.py": "4483f870ef092004b416376e7f5935127b1c434c",
    "numba_kdtree.py": "414553a5e98b033c0daaf37c5efc421fba13d0f7",
}


KNN_GRAPH_BLOBS = {
    "nndescent.py": "a4353f25f43606da6e35fe08678f360172940ba4",
    "kruskal.py": "0a5c7df222aad488d929773181af516c51dafa63",
    "precomputed.py": "da6b7e52e5f0fb2b854596504f79bacd3c10f7d3",
}
KNN_GRAPH_RANDOM_STATE = 42


def knn_graph_sources() -> dict:
    import fast_hdbscan.kruskal as kruskal
    import fast_hdbscan.nndescent as nndescent
    import fast_hdbscan.precomputed as precomputed

    modules = {"nndescent.py": nndescent, "kruskal.py": kruskal, "precomputed.py": precomputed}
    for filename, expected_blob in KNN_GRAPH_BLOBS.items():
        source = Path(modules[filename].__file__).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        if git_blob != expected_blob:
            raise RuntimeError(f"installed {filename} differs from the verified knn graph source")
    return {
        "pynndescent_version": importlib.metadata.version("pynndescent"),
        "modules": KNN_GRAPH_BLOBS,
        "build_knn_graph": str(inspect.signature(nndescent.build_knn_graph)),
        "compute_mst_from_knn_graph": str(inspect.signature(nndescent.compute_mst_from_knn_graph)),
        "random_state": KNN_GRAPH_RANDOM_STATE,
    }


def provenance() -> dict:
    from fast_hdbscan import boruvka, cluster_trees, hdbscan

    dist = importlib.metadata.distribution("fast_hdbscan")
    project_urls = dist.metadata.get_all("Project-URL") or []
    if dist.version != EXPECTED["version"]:
        raise RuntimeError(f"expected fast_hdbscan {EXPECTED['version']}, found {dist.version}")
    if not any(EXPECTED["repository"] in item for item in project_urls):
        raise RuntimeError("installed distribution does not declare the pinned upstream repository")
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = next((p for p in lock["package"] if p["name"] == "fast-hdbscan"), None)
    if locked is None or locked["version"] != EXPECTED["version"]:
        raise RuntimeError("uv.lock is missing the pinned fast-hdbscan release")
    if locked.get("sdist", {}).get("hash") != EXPECTED["sdist"]:
        raise RuntimeError("fast_hdbscan sdist digest in uv.lock differs from pinned artifact")
    wheels = {wheel["hash"] for wheel in locked.get("wheels", [])}
    if EXPECTED["wheel"] not in wheels:
        raise RuntimeError("fast_hdbscan wheel digest in uv.lock differs from pinned artifact")
    signature = inspect.signature(boruvka.parallel_boruvka)
    required = {"tree", "n_threads", "min_samples", "sample_weights", "reproducible"}
    if not required.issubset(signature.parameters):
        raise RuntimeError(f"parallel_boruvka signature mismatch: {signature}")
    expected_symbols = {
        boruvka: ("parallel_boruvka",),
        hdbscan: (
            "mst_to_linkage_tree",
            "condense_tree",
            "extract_eom_clusters",
            "get_point_membership_strength_vector",
            "get_cluster_labelling_at_cut",
        ),
        cluster_trees: ("simplify_hierarchy",),
    }
    for module, names in expected_symbols.items():
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise RuntimeError(f"{module.__name__} missing pinned internal entrypoints: {missing}")
    modules = {Path(module.__file__).name: module for module in (boruvka, cluster_trees, hdbscan)}
    import fast_hdbscan.numba_kdtree as numba_kdtree

    modules["numba_kdtree.py"] = numba_kdtree
    for filename, expected_blob in RELEASE_BLOBS.items():
        source = Path(modules[filename].__file__).read_bytes()
        git_blob = hashlib.sha1(f"blob {len(source)}\0".encode() + source).hexdigest()
        if git_blob != expected_blob:
            raise RuntimeError(
                f"installed {filename} does not match the published 0.3.2 release: {git_blob}"
            )
    return {
        "package": "fast_hdbscan",
        "version": dist.version,
        "repository": EXPECTED["repository"],
        "comparison_commit": COMMIT,
        "entrypoint": "fast_hdbscan.boruvka.parallel_boruvka",
        "reproducible": True,
        "dimensions": None,
        "release_source_evidence": (
            "Installed source files match the published 0.3.2 release Git blob IDs; "
            "uv.lock pins release wheel and sdist SHA-256 hashes."
        ),
        "release_source_blobs": RELEASE_BLOBS,
        "knn_graph_sources": knn_graph_sources(),
        "comparison_commit_blob_ids": COMPARISON_COMMIT_BLOBS,
        "artifact_hashes": {"sdist": EXPECTED["sdist"], "wheel": EXPECTED["wheel"]},
        "modules": {
            name: Path(module.__file__).name
            for name, module in [
                ("boruvka", boruvka),
                ("hdbscan", hdbscan),
                ("cluster_trees", cluster_trees),
            ]
        },
        "signatures": {
            "parallel_boruvka": str(signature),
            "mst_to_linkage_tree": str(inspect.signature(hdbscan.mst_to_linkage_tree)),
            "condense_tree": str(inspect.signature(hdbscan.condense_tree)),
            "simplify_hierarchy": str(inspect.signature(cluster_trees.simplify_hierarchy)),
            "get_cluster_labelling_at_cut": str(
                inspect.signature(hdbscan.get_cluster_labelling_at_cut)
            ),
        },
    }


def input_for(case: dict) -> np.ndarray:
    rng = np.random.default_rng(case["seed"])
    n, d, kind = case["n"], case["dimensions"], case["dataset"]
    if kind == "moons":
        xy, _ = make_moons(n_samples=n, noise=0.04, random_state=case["seed"])
        x = np.zeros((n, d), dtype=np.float32)
        x[:, :2] = xy
    elif kind == "blobs":
        x, _ = make_blobs(
            n_samples=n, n_features=d, centers=3, cluster_std=0.35, random_state=case["seed"]
        )
    elif kind == "varying_density":
        x, _ = make_blobs(
            n_samples=n,
            n_features=d,
            centers=3,
            cluster_std=[0.08, 0.3, 0.8],
            random_state=case["seed"],
        )
    elif kind == "hierarchical_blobs":
        centers = np.array([[-10.0, 0.0], [-9.0, 0.0], [10.0, 0.0], [11.0, 0.0]])
        x = np.concatenate(
            [rng.normal(center, 0.08, size=(n // 4, 2)) for center in centers], axis=0
        )
    elif kind == "duplicates":
        x = np.repeat(
            np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
            n // 4,
            axis=0,
        )
    elif kind == "single_cluster":
        x = rng.normal(0.0, 0.15, (n, d))
    elif kind == "all_noise":
        x = np.arange(n * d, dtype=np.float64).reshape(n, d) * 1000
    elif kind == "two_points":
        x = np.zeros((n, d), dtype=np.float32)
        x[-1, :] = 1
    elif kind == "one_point":
        x = np.zeros((n, d), dtype=np.float32)
    elif kind in ("mixed_nonfinite", "one_finite", "all_nonfinite"):
        x = np.full((n, d), np.nan, dtype=np.float32)
        if kind == "mixed_nonfinite":
            x[: n - 3] = rng.normal(size=(n - 3, d))
        if kind == "one_finite":
            x[0] = 0
    elif kind == "overflow":
        x = np.zeros((n, d), dtype=np.float32)
        x[:, 0] = np.linspace(0, 3e19, n)
    else:
        raise ValueError(f"unsupported dataset {kind}")
    x = np.asarray(x, dtype=np.float32)
    if x.shape != (n, d):
        raise ValueError(f"dataset {kind} produced unexpected shape {x.shape}")
    return x


def verify_issue_repros(manifest: dict) -> list[dict]:
    from fast_hdbscan.cluster_trees import segments_in_branch

    results = []
    for repro in manifest.get("upstream_issue_repros", []):
        observed = sorted(
            int(value)
            for value in segments_in_branch(
                np.asarray(repro["parents"], dtype=np.int64),
                np.asarray(repro["children"], dtype=np.int64),
                np.int64(repro["segment"]),
            )
        )
        if observed != repro["release_observed"]:
            raise RuntimeError(f"{repro['id']} release result changed: {observed}")
        results.append(
            {
                "id": repro["id"],
                "release_observed": observed,
                "post_release_fix_expected": repro["post_release_fix_expected"],
                "issue": repro["issue"],
            }
        )
    return results


def resolve_persistence_boundary(condensed, n_points: int) -> dict:
    cluster_rows = np.flatnonzero(condensed.child_size > 1)
    parents = condensed.parent[cluster_rows]
    children = condensed.child[cluster_rows]
    lambdas = condensed.lambda_val[cluster_rows]
    node_count = int(max(np.max(parents), np.max(children)) - n_points + 1)
    max_births = np.zeros(node_count, dtype=np.float32)
    for parent, value in zip(condensed.parent, condensed.lambda_val, strict=True):
        max_births[int(parent) - n_points] = np.float32(value)
    boundaries = []
    for i in range(len(children) - 1, 0, -2):
        parent = int(parents[i])
        child_ids = children[i - 1 : i + 1] - n_points
        births = max_births[child_ids]
        death = np.float32(lambdas[i])
        # simplify_hierarchy stores both births and death in float32; NumPy
        # scalar subtraction below therefore reproduces the threshold exactly.
        persistence = np.float32(np.min(births) - death)
        max_births[parent - n_points] = np.maximum(max_births[parent - n_points], np.max(births))
        if persistence > 0 and np.isfinite(persistence):
            boundaries.append(
                {
                    "threshold": float(persistence),
                    "birth_min": float(np.min(births)),
                    "death": float(death),
                    "dtype": "float32",
                    "rule": ">=",
                }
            )
    if boundaries:
        return min(boundaries, key=lambda boundary: boundary["threshold"])
    raise RuntimeError("could not measure a positive persistence equality boundary")


def tie_evidence(x: np.ndarray, core: np.ndarray, mst: np.ndarray, condensed, params: dict) -> dict:
    # The overflow fixture deliberately produces infinite float32 distances.
    # Preserve those values without emitting an expected overflow warning.
    with np.errstate(over="ignore"):
        squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2, dtype=np.float32)
    i, j = np.triu_indices(len(x), 1)
    euclidean = np.sqrt(squared, dtype=np.float32)
    candidate_weights = np.maximum(euclidean[i, j], np.maximum(core[i], core[j]))

    def classify(values: np.ndarray) -> str:
        values = np.sort(np.asarray(values, dtype=np.float64))
        if len(values) < 2:
            return "none"
        finite = values[np.isfinite(values)]
        if np.any(finite[1:] == finite[:-1]):
            return "exact"
        gaps = np.diff(finite)
        scale = np.maximum(1.0, np.maximum(np.abs(finite[:-1]), np.abs(finite[1:])))
        if np.any(gaps <= 1e-6 * scale):
            return "near"
        # Two or more nonfinite values can compare equal without forming a finite tie.
        return "nonfinite" if len(values) - len(finite) > 1 else "none"

    candidate_class = classify(candidate_weights)
    merge_class = classify(mst[:, 2]) if len(mst) else "none"
    from fast_hdbscan.cluster_trees import (
        cluster_tree_from_condensed_tree,
        eom_recursion,
        score_condensed_tree_nodes,
    )
    from numba import typed, types

    tree = cluster_tree_from_condensed_tree(condensed)
    scores = score_condensed_tree_nodes(condensed)
    comparisons = []
    if params["selection"] == "eom" and len(tree.parent):
        root = int(np.min(tree.parent))
        sizes = typed.Dict.empty(types.int64, types.float32)
        for child, size in zip(tree.child, tree.child_size, strict=True):
            sizes[np.int64(child)] = np.float32(size)
        sizes[np.int64(root)] = np.float32(root - 1)
        selected = typed.Dict.empty(types.int64, types.boolean)
        for node in scores:
            selected[node] = False
        maximum = np.inf if params["max_cluster_size"] is None else params["max_cluster_size"]
        for node in scores:
            if node == root and not params["allow_single_cluster"]:
                continue
            # Invoke the actual pinned recursion, rather than implementing selection.
            child_total = sum(
                float(eom_recursion(child, tree, scores, sizes, selected, maximum))
                for child in tree.child[tree.parent == node]
            )
            comparisons.append((max(float(scores[node]), 0.0), child_total))
            if np.isfinite(maximum):
                comparisons.append((float(sizes[node]), float(maximum)))
    if params["epsilon"] > 0:
        comparisons.extend(
            (1.0 / float(value) if value > 0 else np.inf, params["epsilon"])
            for value in tree.lambda_val
        )
    exact = any(a == b and np.isfinite(a) for a, b in comparisons)
    near = any(
        np.isfinite(a) and np.isfinite(b) and 0 < abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))
        for a, b in comparisons
    )
    nonfinite = any(not np.isfinite(a) or not np.isfinite(b) for a, b in comparisons)
    classification = "exact" if exact else "near" if near else "nonfinite" if nonfinite else "none"
    return {
        "candidate": candidate_class,
        "merge": merge_class,
        "cluster_selection": {
            "method": params["selection"],
            "comparison": classification,
            "numeric_comparison_count": len(comparisons),
            "leaf_selection_is_categorical": params["selection"] == "leaf",
        },
    }


def knn_graph_mst(
    x: np.ndarray, min_samples: int, knn_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from fast_hdbscan import nndescent

    _, knn_indices, _ = nndescent.build_knn_graph(x, knn_k, random_state=KNN_GRAPH_RANDOM_STATE)
    if len(np.unique(x, axis=0)) != len(x):
        raise RuntimeError("knn graph fixtures must not contain duplicate rows")
    squared = np.sum((x[:, None, :].astype(np.float64) - x[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(squared, np.inf)
    exact = np.argsort(squared, axis=1, kind="stable")[:, :knn_k]
    if not np.array_equal(knn_indices, exact):
        raise RuntimeError("pynndescent knn graph differs from the exact non-self kNN oracle")
    mst, neighbors, core = nndescent.compute_mst_from_knn_graph(
        x, min_samples, knn_k=knn_k, random_state=KNN_GRAPH_RANDOM_STATE
    )
    if not np.array_equal(neighbors, knn_indices[:, :min_samples]):
        raise RuntimeError("compute_mst_from_knn_graph neighbors differ from the validated graph")
    return mst, neighbors, core


def create_case(case: dict, reference: dict) -> tuple[dict, dict]:
    from fast_hdbscan import boruvka, hdbscan
    from numba import get_num_threads, set_num_threads

    # Exact-tie MST endpoints depend on Numba thread scheduling; one thread keeps the stored
    # sorted MST (src, dst, weight) reproducible.
    set_num_threads(1)
    x = input_for(case)
    finite_index = np.flatnonzero(np.isfinite(x).all(axis=1)).astype(np.int64)
    clean = np.ascontiguousarray(x[finite_index], dtype=np.float32)
    params = dict(case["params"])
    persistence = params["persistence"]
    status, reason = case["status"], None
    arrays = {"input": x, "finite_index": finite_index}
    metadata = {
        "schema_version": 1,
        "case_id": case["id"],
        "status": status,
        "reference": reference,
        "parameters": params,
        "input_shape": list(x.shape),
        "finite_index": finite_index.tolist(),
        "tie_classification": [],
        "divergence_classification": case["divergence"],
        "reason": None,
        "arrays": {},
        "tie_evidence": {
            "candidate": "unavailable",
            "merge": "unavailable",
            "cluster_selection": "unavailable",
        },
    }
    if status == "validation_failure":
        reason = "zero finite rows are rejected before invoking the upstream MST routine"
    elif status == "port_contract":
        reason = (
            "single-finite-row arrays are the intended port contract, "
            "not upstream algorithm output; "
            "the isolated empty-linkage remap probe below records published release behavior"
        )
        arrays["outlier_order"] = np.asarray(
            list(set(range(x.shape[0])) - set(finite_index.tolist())), dtype=np.int64
        )
        if clean.shape[0] == 1:
            empty_linkage = np.empty((0, 4), dtype=np.float64)
            try:
                hdbscan.remap_single_linkage_tree(
                    empty_linkage.copy(),
                    dict(enumerate(finite_index.tolist())),
                    arrays["outlier_order"].tolist(),
                )
            except IndexError as exc:
                reason += f"; isolated remap probe empty (0,4) linkage raises IndexError: {exc}"
            else:
                raise RuntimeError("expected release empty-linkage remap probe to raise IndexError")
            arrays.update(
                {
                    "mst_weights": np.empty(0, dtype=np.float64),
                    "mst": np.empty((0, 3), dtype=np.float64),
                    "linkage": empty_linkage,
                    "condensed_tree": np.empty(
                        0,
                        dtype=[
                            ("parent", np.intp),
                            ("child", np.intp),
                            ("lambda_val", float),
                            ("child_size", np.float32),
                        ],
                    ),
                    "labels": np.full(x.shape[0], -1, dtype=np.int64),
                    "probabilities": np.zeros(x.shape[0], dtype=np.float32),
                    "neighbors": np.empty((1, 0), dtype=np.int64),
                    "core_distances": np.full(1, np.inf, dtype=np.float32),
                    "dbscan_labels": np.full(x.shape[0], -1, dtype=np.int64),
                }
            )
    else:
        if clean.shape[0] < 2:
            raise RuntimeError(
                f"{case['id']} must not invoke Boruvka with fewer than two finite rows"
            )
        if params.get("mst_graph") == "knn_graph":
            mst, neighbors, core = knn_graph_mst(clean, params["min_samples"], params["knn_k"])
        else:
            tree = hdbscan.build_kdtree(clean)
            mst, neighbors, core = boruvka.parallel_boruvka(
                tree,
                get_num_threads(),
                min_samples=params["min_samples"],
                sample_weights=np.zeros(1, dtype=np.float32),
                reproducible=True,
            )
        labels, probabilities, linkage, condensed, sorted_mst = hdbscan.clusters_from_spanning_tree(
            mst,
            min_cluster_size=params["min_cluster_size"],
            cluster_selection_method=params["selection"],
            max_cluster_size=np.inf
            if params["max_cluster_size"] is None
            else params["max_cluster_size"],
            allow_single_cluster=params["allow_single_cluster"],
            cluster_selection_epsilon=params["epsilon"],
            cluster_selection_persistence=0.0
            if persistence == "measured_boundary"
            else persistence,
        )
        if persistence == "measured_boundary":
            from fast_hdbscan.cluster_trees import simplify_hierarchy

            boundary = resolve_persistence_boundary(condensed, clean.shape[0])
            persistence = boundary["threshold"]
            metadata["persistence_boundary_evidence"] = boundary
            arrays["boundary_input"] = hdbscan.to_numpy_rec_array(condensed).copy()
            for name, threshold in (
                ("below", float(np.nextafter(np.float32(persistence), np.float32(-np.inf)))),
                ("equal", persistence),
                ("above", float(np.nextafter(np.float32(persistence), np.float32(np.inf)))),
            ):
                probe_tree = type(condensed)(*(column.copy() for column in condensed))
                arrays[f"boundary_{name}"] = hdbscan.to_numpy_rec_array(
                    simplify_hierarchy(probe_tree, threshold)
                )
            if not equal_array(arrays["boundary_below"], arrays["boundary_equal"]):
                raise RuntimeError("persistence equality did not retain the boundary branch")
            if equal_array(arrays["boundary_equal"], arrays["boundary_above"]):
                raise RuntimeError("persistence above-boundary probe did not remove a branch")
            labels, probabilities, linkage, condensed, sorted_mst = (
                hdbscan.clusters_from_spanning_tree(
                    mst,
                    min_cluster_size=params["min_cluster_size"],
                    cluster_selection_method=params["selection"],
                    max_cluster_size=np.inf
                    if params["max_cluster_size"] is None
                    else params["max_cluster_size"],
                    allow_single_cluster=params["allow_single_cluster"],
                    cluster_selection_epsilon=params["epsilon"],
                    cluster_selection_persistence=persistence,
                )
            )
            metadata["parameters"]["persistence"] = persistence
        outliers = list(set(range(x.shape[0])) - set(finite_index.tolist()))
        internal_to_raw = {i: int(raw) for i, raw in enumerate(finite_index)}
        raw_labels = np.full(x.shape[0], -1, dtype=np.int64)
        raw_probabilities = np.zeros(x.shape[0], dtype=np.float32)
        raw_labels[finite_index] = labels
        raw_probabilities[finite_index] = probabilities
        if outliers:
            remapped_condensed = hdbscan.remap_condensed_tree(
                hdbscan.to_numpy_rec_array(condensed), internal_to_raw, outliers
            )
            remapped_linkage = hdbscan.remap_single_linkage_tree(
                linkage.copy(), internal_to_raw, outliers
            )
        else:
            remapped_condensed = hdbscan.to_numpy_rec_array(condensed).copy()
            remapped_linkage = linkage.copy()
        if clean.shape[0] != x.shape[0]:
            labels, probabilities = raw_labels, raw_probabilities
        evidence = tie_evidence(clean, core, mst, condensed, params)
        metadata["tie_evidence"] = evidence
        metadata["tie_classification"] = sorted(
            key
            for key, value in evidence.items()
            if (value.get("comparison") if isinstance(value, dict) else value) in ("exact", "near")
        )
        dbscan = hdbscan.get_cluster_labelling_at_cut(
            linkage, params["epsilon"], params["min_samples"]
        )
        raw_dbscan = np.full(x.shape[0], -1, dtype=np.int64)
        raw_dbscan[finite_index] = dbscan
        remapped_dbscan = hdbscan.get_cluster_labelling_at_cut(
            remapped_linkage, params["epsilon"], params["min_samples"]
        )
        arrays.update(
            {
                "mst_weights": np.asarray(sorted_mst[:, 2], dtype=np.float64),
                "mst": np.asarray(sorted_mst, dtype=np.float64),
                "linkage": np.asarray(linkage, dtype=np.float64),
                "condensed_tree": np.asarray(hdbscan.to_numpy_rec_array(condensed)),
                "remapped_condensed_tree": np.asarray(remapped_condensed),
                "remapped_linkage": np.asarray(remapped_linkage, dtype=np.float64),
                "labels": np.asarray(labels, dtype=np.int64),
                "probabilities": np.asarray(probabilities, dtype=np.float32),
                "neighbors": np.asarray(neighbors, dtype=np.int64),
                "core_distances": np.asarray(core, dtype=np.float32),
                "dbscan_labels": raw_dbscan,
                "remapped_dbscan_labels": np.asarray(remapped_dbscan, dtype=np.int64),
                "outlier_order": np.asarray(outliers, dtype=np.int64),
            }
        )
        if status == "reference":
            metadata["status"] = status
    if case["divergence"] == "overflow_neighbor_divergence":
        from fast_hdbscan.numba_kdtree import parallel_tree_query

        probe = np.array([[-2e19, 0.0], [2e19, 0.0]], dtype=np.float32)
        squared_distances, probe_neighbors = parallel_tree_query(
            hdbscan.build_kdtree(probe), probe, 2, True
        )
        if not np.array_equal(probe_neighbors, [[0, -1], [1, -1]]):
            raise RuntimeError("published overflow-neighbor behavior changed")
        if not np.array_equal(squared_distances, [[0.0, np.inf], [0.0, np.inf]]):
            raise RuntimeError("published squared-overflow distances changed")
        arrays.update(
            {
                "overflow_probe_input": probe,
                "overflow_probe_squared_distances": squared_distances,
                "overflow_probe_neighbors": probe_neighbors,
            }
        )
        metadata["overflow_neighbor_probe"] = {
            "entrypoint": "fast_hdbscan.numba_kdtree.parallel_tree_query",
            "output_rdist": True,
            "self_included": True,
            "release_observation": "infinite-distance real neighbors are replaced by -1 padding",
            "intended_common_indices": [[0, 1], [1, 0]],
            "intended_contract_source": "port contract, not upstream output",
        }
    metadata["reason"] = reason
    metadata["reference"].update({"dimensions": int(x.shape[1]), "shape": list(x.shape)})
    for key, value in arrays.items():
        dtype = (
            [{"name": field[0], "dtype": field[1]} for field in value.dtype.descr]
            if value.dtype.fields
            else value.dtype.str
        )
        metadata["arrays"][key] = {"dtype": dtype, "shape": list(value.shape)}
    return arrays, metadata


def write_cases(directory: Path, manifest: dict, reference: dict) -> None:
    for case in manifest["cases"]:
        arrays, metadata = create_case(case, reference)
        np.savez(directory / f"{case['id']}.npz", **arrays)
        (directory / f"{case['id']}.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )


def equal_array(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.fields:
        return all(np.array_equal(a[name], b[name], equal_nan=True) for name in a.dtype.names or ())
    return np.array_equal(a, b, equal_nan=True)


def array_mismatch(name: str, expected: np.ndarray, actual: np.ndarray) -> list[str]:
    """Compare topology exactly and allow bounded drift only in derived floats."""
    if expected.dtype != actual.dtype or expected.shape != actual.shape:
        return [
            f"{name}: expected dtype={expected.dtype}, shape={expected.shape}; "
            f"actual dtype={actual.dtype}, shape={actual.shape}"
        ]
    if expected.dtype.fields:
        return [
            detail
            for field in expected.dtype.names or ()
            for detail in array_mismatch(f"{name}.{field}", expected[field], actual[field])
        ]
    if name in {"mst", "linkage", "remapped_linkage"}:
        return [
            detail
            for column in range(expected.shape[1])
            for detail in array_mismatch(
                f"{name}[{column}]", expected[:, column], actual[:, column]
            )
        ]
    equal = expected == actual
    if np.issubdtype(expected.dtype, np.inexact):
        equal = equal | (np.isnan(expected) & np.isnan(actual))
        numerical = name in {
            "mst_weights",
            "core_distances",
            "probabilities",
            "mst[2]",
            "linkage[2]",
            "remapped_linkage[2]",
            "condensed_tree.lambda_val",
            "remapped_condensed_tree.lambda_val",
            "boundary_input.lambda_val",
            "boundary_below.lambda_val",
            "boundary_equal.lambda_val",
            "boundary_above.lambda_val",
            "overflow_probe_squared_distances",
        }
        if numerical:
            # The relative bound matches the existing core-distance contract;
            # the small absolute floor handles values near zero. Only finite values
            # are approximate: NaNs and signed infinities must match exactly.
            finite = np.isfinite(expected) & np.isfinite(actual)
            equal = equal | (finite & np.isclose(expected, actual, rtol=1e-6, atol=1e-7))
    if np.all(equal):
        return []
    index = tuple(int(i) for i in np.argwhere(~equal)[0])
    return [
        f"{name}: {np.count_nonzero(~equal)} differing values; first at {index}: "
        f"expected={expected[index]!r}, actual={actual[index]!r}"
    ]


def metadata_mismatch(expected: object, actual: object, path: str = "") -> list[str]:
    """Keep metadata exact except explicitly named derived boundary evidence."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        details = [f"{path}: missing={missing}, extra={extra}"] if missing or extra else []
        for key in sorted(set(expected) & set(actual)):
            child = f"{path}.{key}" if path else key
            details.extend(metadata_mismatch(expected[key], actual[key], child))
        return details
    if (
        path
        in {
            "persistence_boundary_evidence.threshold",
            "persistence_boundary_evidence.birth_min",
            "persistence_boundary_evidence.death",
        }
        and type(expected) is float
        and type(actual) is float
    ):
        if (
            np.isfinite(expected)
            and np.isfinite(actual)
            and np.isclose(expected, actual, rtol=1e-6, atol=1e-7)
        ):
            return []
    if type(expected) is not type(actual) or expected != actual:
        return [f"{path}: expected={expected!r}, actual={actual!r}"]
    return []


def validate_manifest(manifest: dict, schema: dict) -> None:
    if manifest["schema_version"] != schema["schema_version"]:
        raise RuntimeError("manifest/schema version mismatch")
    required_reference = schema.get("manifest_reference_required")
    if not isinstance(required_reference, list) or not required_reference:
        raise RuntimeError("schema manifest_reference_required must be a nonempty array")
    if any(not isinstance(key, str) or not key for key in required_reference):
        raise RuntimeError("schema manifest_reference_required entries must be nonempty strings")
    if len(required_reference) != len(set(required_reference)):
        raise RuntimeError("schema manifest_reference_required contains duplicate keys")
    reference = manifest.get("reference")
    if not isinstance(reference, dict):
        raise RuntimeError("manifest reference must be an object")
    for key in required_reference:
        if key not in reference:
            raise RuntimeError(f"manifest reference missing required key {key}")
    if set(reference) != set(required_reference):
        raise RuntimeError("manifest reference has unknown keys")
    cases = manifest["cases"]
    ids = [case["id"] for case in cases]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("manifest needs unique nonempty case identifiers")
    for key, field in (("datasets", "dataset"), ("dimensions", "dimensions")):
        if set(manifest["coverage"][key]) != {case[field] for case in cases}:
            raise RuntimeError(f"manifest {key} coverage does not match its cases")
    if {case["dimensions"] for case in cases} != {2, 10, 50}:
        raise RuntimeError("the exact reference must cover dimensions 2/10/50")
    if {case["params"]["selection"] for case in cases} != {"leaf", "eom"}:
        raise RuntimeError("manifest must cover leaf and EOM extraction")
    if not any(case["n"] < case["params"]["min_samples"] for case in cases):
        raise RuntimeError("manifest lacks insufficient-neighbor coverage")
    if not {1, 2}.issubset({case["n"] for case in cases}):
        raise RuntimeError("manifest lacks singleton/two-row boundaries")
    if not any(
        case["params"]["selection"] == "eom"
        and case["params"]["max_cluster_size"] is not None
        and case["params"]["epsilon"] > 0
        and case["params"]["allow_single_cluster"]
        for case in cases
    ):
        raise RuntimeError("manifest lacks interacting EOM/max-size/epsilon/single-cluster options")
    for case in cases:
        if case["status"] not in schema["case_statuses"]:
            raise RuntimeError(f"unknown case status: {case['status']}")
        if case["n"] <= 0 or case["dimensions"] <= 0:
            raise RuntimeError("fixture dimensions must be positive")
        if case["params"].get("mst_graph") == "knn_graph":
            expected_width = min(max(3 * case["params"]["min_samples"], 15), case["n"] - 1)
            if case["params"]["knn_k"] != expected_width:
                raise RuntimeError(f"{case['id']} knn_k differs from the default graph width")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-provenance", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    proof = provenance()
    manifest = json.loads(MANIFEST.read_text())
    schema = json.loads(SCHEMA.read_text())
    validate_manifest(manifest, schema)
    for key, expected in manifest["reference"].items():
        if proof.get(key) != expected:
            raise RuntimeError(f"manifest reference {key} differs from the verified release")
    proof["known_issue_repros"] = verify_issue_repros(manifest)
    if args.verify_provenance:
        print(json.dumps(proof, indent=2, sort_keys=True))
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix="fast-hdbscan-fixture-check-") as tmp:
            temp = Path(tmp)
            write_cases(temp, manifest, proof)
            for case in manifest["cases"]:
                stem = case["id"]
                with (
                    np.load(FIXTURES / f"{stem}.npz", allow_pickle=False) as old,
                    np.load(temp / f"{stem}.npz", allow_pickle=False) as new,
                ):
                    missing = sorted(set(old.files) - set(new.files))
                    extra = sorted(set(new.files) - set(old.files))
                    details = [
                        detail
                        for key in sorted(set(old.files) & set(new.files))
                        for detail in array_mismatch(key, old[key], new[key])
                    ]
                    if missing or extra or details:
                        raise RuntimeError(
                            f"array mismatch while regenerating {stem}: "
                            f"missing={missing}, extra={extra}\n" + "\n".join(details)
                        )
                old_meta = json.loads((FIXTURES / f"{stem}.json").read_text())
                new_meta = json.loads((temp / f"{stem}.json").read_text())
                metadata_details = metadata_mismatch(old_meta, new_meta)
                if metadata_details:
                    raise RuntimeError(
                        f"semantic metadata mismatch while regenerating {stem}:\n"
                        + "\n".join(metadata_details)
                    )
            print(f"verified regeneration of {len(manifest['cases'])} cases")
        return 0
    write_cases(FIXTURES, manifest, proof)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
