#!/usr/bin/env python3
"""Generate pinned umap-learn 0.5.12 fuzzy-graph reference fixtures."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from umap.umap_ import compute_membership_strengths, smooth_knn_dist

REFERENCE_VERSION = "0.5.12"
OUTPUT = Path(__file__).with_name("umap_learn_0.5.12_fuzzy_graph.json")

# Every row has an explicit self identity at slot zero. Neighbor arrays are
# deliberately fixed so this oracle covers graph math, not search or training.
GRAPH_INPUTS = [
    {
        "name": "irregular_fractional_bandwidth",
        "n": 4,
        "k": 4,
        "indices": [0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0],
        "distances": [
            0.0,
            0.15,
            0.8,
            1.7,
            0.0,
            0.15,
            0.55,
            1.2,
            0.0,
            0.35,
            0.8,
            1.5,
            0.0,
            0.55,
            1.2,
            1.7,
        ],
        "local_connectivity": 1.5,
        "bandwidth": 1.25,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
    {
        "name": "duplicate_and_all_identical",
        "n": 3,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 2, 2, 0, 1],
        "distances": [0.0] * 9,
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
    {
        "name": "duplicate_neighbor_coordinates",
        "n": 3,
        "k": 3,
        "indices": [0, 1, 1, 1, 0, 2, 2, 0, 1],
        "distances": [0.0, 0.1, 0.2, 0.0, 0.2, 0.7, 0.0, 0.4, 0.8],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 0.5,
        "epochs": 10,
    },
    {
        "name": "fractional_and_zero_connectivity",
        "n": 3,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 2, 2, 0, 1],
        "distances": [0.0, 0.0, 0.4, 0.0, 0.2, 0.9, 0.0, 0.4, 0.8],
        "local_connectivity": 0.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
    {
        "name": "insufficient_positive_distances",
        "n": 4,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2],
        "distances": [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "local_connectivity": 2.0,
        "bandwidth": 0.75,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
    {
        "name": "sigma_floor_and_minus_one_skip",
        "n": 3,
        "k": 3,
        "indices": [0, 1, -1, 1, 0, 2, 2, 0, 1],
        "distances": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
    {
        "name": "asymmetric_mix_zero",
        "n": 4,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2],
        "distances": [0.0, 0.2, 0.7, 0.0, 0.3, 0.8, 0.0, 0.4, 0.9, 0.0, 0.5, 1.0],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 0.0,
        "epochs": 10,
    },
    {
        "name": "asymmetric_mix_half",
        "n": 4,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2],
        "distances": [0.0, 0.2, 0.7, 0.0, 0.3, 0.8, 0.0, 0.4, 0.9, 0.0, 0.5, 1.0],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 0.5,
        "epochs": 10,
    },
    {
        "name": "asymmetric_mix_one",
        "n": 4,
        "k": 3,
        "indices": [0, 1, 2, 1, 0, 3, 2, 0, 3, 3, 1, 2],
        "distances": [0.0, 0.2, 0.7, 0.0, 0.3, 0.8, 0.0, 0.4, 0.9, 0.0, 0.5, 1.0],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 1.0,
        "epochs": 10,
    },
]


def fuzzy_graph(case: dict[str, object]) -> dict[str, object]:
    n = int(case["n"])
    k = int(case["k"])
    indices = np.asarray(case["indices"], dtype=np.int32).reshape(n, k)
    distances = np.asarray(case["distances"], dtype=np.float32).reshape(n, k)
    local_connectivity = float(case["local_connectivity"])
    bandwidth = float(case["bandwidth"])
    mix_ratio = float(case["set_op_mix_ratio"])

    sigmas, rhos = smooth_knn_dist(
        distances,
        float(k),
        local_connectivity=local_connectivity,
        bandwidth=bandwidth,
    )
    rows, cols, values, _ = compute_membership_strengths(
        indices,
        distances,
        sigmas,
        rhos,
        return_dists=False,
        bipartite=False,
    )
    directed_sparse = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    directed_sparse.eliminate_zeros()
    transpose = directed_sparse.transpose().tocsr()
    product = directed_sparse.multiply(transpose)
    symmetric_sparse = (
        mix_ratio * (directed_sparse + transpose - product) + (1.0 - mix_ratio) * product
    )
    symmetric_sparse.eliminate_zeros()

    weights32 = symmetric_sparse.data.astype(np.float32, copy=False)
    threshold = np.float32(weights32.max() / np.float32(int(case["epochs"])))
    counts = np.floor(weights32 * np.float32(int(case["epochs"]))).astype(np.int64)
    counts[weights32 < threshold] = 0
    dense_counts = np.zeros((n, n), dtype=np.int64)
    coo = symmetric_sparse.tocoo()
    dense_counts[coo.row, coo.col] = counts

    result = dict(case)
    result.update(
        rho=rhos.astype(np.float32).tolist(),
        sigma=sigmas.astype(np.float32).tolist(),
        directed=directed_sparse.toarray().astype(np.float32).ravel().tolist(),
        symmetric=symmetric_sparse.toarray().astype(np.float32).ravel().tolist(),
        counts=dense_counts.ravel().tolist(),
    )
    return result


def count_case(name: str, weights: list[float], epochs: int) -> dict[str, object]:
    values = np.asarray(weights, dtype=np.float32)
    threshold = np.float32(values.max() / np.float32(epochs))
    counts = np.floor(values * np.float32(epochs)).astype(np.int64)
    counts[values < threshold] = 0
    return {
        "name": name,
        "weights": values.tolist(),
        "epochs": epochs,
        "counts": counts.tolist(),
        "threshold": float(threshold),
    }


def main() -> None:
    actual_version = version("umap-learn")
    if actual_version != REFERENCE_VERSION:
        raise RuntimeError(f"expected umap-learn {REFERENCE_VERSION}, found {actual_version}")

    count_cases = [
        count_case("float32_point_seven_times_ten", [0.7], 10),
        count_case("prune_and_floor", [1.0, 0.2, 0.001], 10),
    ]
    boundary = np.float32(0.1)
    count_cases.append(
        count_case(
            "strict_threshold_adjacent_float32",
            [
                1.0,
                float(np.nextafter(boundary, np.float32(-np.inf))),
                float(boundary),
                float(np.nextafter(boundary, np.float32(np.inf))),
            ],
            10,
        )
    )

    fixture = {
        "reference_version": actual_version,
        "numpy_version": np.__version__,
        "cases": [fuzzy_graph(case) for case in GRAPH_INPUTS],
        "count_cases": count_cases,
    }
    OUTPUT.write_text(json.dumps(fixture, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
