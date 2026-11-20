#!/usr/bin/env -S uv run --no-sync
"""Score fixed Rust ANN graphs with the gate's tie-aware recall convention."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_matrix(path: Path, rows: int, features: int) -> tuple[np.ndarray, dict[str, Any]]:
    expected = rows * features * np.dtype("<f4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes; expected {expected}")
    matrix = np.memmap(path, mode="r", dtype="<f4", shape=(rows, features))
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} contains non-finite values")
    return matrix, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": rows,
        "features": features,
        "dtype": "f32le",
    }


def _read_ids(path: Path, rows: int, width: int) -> tuple[np.ndarray, dict[str, Any]]:
    expected = rows * width * np.dtype("<i4").itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path} has {path.stat().st_size} bytes; expected {expected}")
    ids = np.memmap(path, mode="r", dtype="<i4", shape=(rows, width))
    if np.any(ids < 0) or np.any(ids >= rows):
        raise ValueError(f"{path} contains an out-of-range neighbor ID")
    return ids, {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "rows": rows,
        "width": width,
        "dtype": "i32le",
    }


def _sample_rows(path: Path | None, rows: int) -> np.ndarray:
    if path is None:
        return np.arange(rows, dtype=np.int64)
    if path.stat().st_size % 8:
        raise ValueError(f"{path} has an invalid u64le byte length")
    selected = np.fromfile(path, dtype="<u8").astype(np.int64)
    if not len(selected) or np.any(selected < 0) or np.any(selected >= rows):
        raise ValueError(f"{path} has no valid scoring rows")
    return selected


def tie_aware_recall(
    matrix: np.ndarray,
    actual: np.ndarray,
    k: int,
    rows: np.ndarray,
    *,
    exclude_self: bool,
    reference_chunk_size: int = 4096,
) -> float:
    """Reproduce test_standard_quality::tie_aware_recall, including unique hits."""
    n = matrix.shape[0]
    if actual.ndim != 2 or actual.shape[0] != n or actual.shape[1] < k or k <= 0:
        raise ValueError("invalid k or neighbor graph dimensions")
    hits = 0
    for row in rows:
        query = np.asarray(matrix[row], dtype=np.float64)

        def distances(points: np.ndarray, query: np.ndarray) -> np.ndarray:
            total = np.zeros(len(points), dtype=np.float64)
            for feature, coordinate in enumerate(query):
                delta = coordinate - points[:, feature].astype(np.float64)
                total += delta * delta
            return np.sqrt(total)

        nearest = np.empty(k, dtype=np.float64)
        nearest.fill(np.inf)
        # Keep the k smallest Euclidean distances, with the same f64 arithmetic and
        # sqrt cutoff as the Rust comparator. Self is excluded only for fit recall.
        for start in range(0, n, reference_chunk_size):
            end = min(start + reference_chunk_size, n)
            distance = distances(matrix[start:end], query)
            if exclude_self and start <= row < end:
                distance[row - start] = np.inf
            nearest = np.partition(np.concatenate((nearest, distance)), k - 1)[:k]
        boundary = float(np.max(nearest))
        seen: set[int] = set()
        candidates = actual[row, :k]
        candidate_distances = distances(matrix[candidates], query)
        for neighbor, distance in zip(candidates, candidate_distances, strict=True):
            neighbor_id = int(neighbor)
            if neighbor_id in seen:
                continue
            seen.add(neighbor_id)
            if exclude_self and neighbor_id == row:
                raise ValueError(f"fit graph contains self at row {row}")
            hits += int(distance <= boundary)
    return hits / (len(rows) * k)


def _fit_rinnd(matrix: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    import umap.umap_ as umap_module

    version = importlib.metadata.version("umap-learn")
    if not version.startswith("0.6."):
        raise RuntimeError(f"fixed recall reference requires umap-learn 0.6.x, found {version}")
    if umap_module.RINND is None:
        raise RuntimeError("umap-learn did not bind RINND")
    original_rinnd = umap_module.RINND
    original_nnd = umap_module.NNDescent
    calls = {"rinnd": 0, "pynndescent": 0}
    graph_only_calls: list[bool] = []

    def count_rinnd(*args: Any, **kwargs: Any) -> Any:
        calls["rinnd"] += 1
        graph_only_calls.append(kwargs.get("graph_only") is True)
        return original_rinnd(*args, **kwargs)

    def reject_pynndescent(*args: Any, **kwargs: Any) -> Any:
        calls["pynndescent"] += 1
        raise RuntimeError("fixed recall reference unexpectedly dispatched to PyNNDescent")

    umap_module.RINND = count_rinnd
    vars(umap_module)["NNDescent"] = reject_pynndescent
    try:
        ids, _, index = umap_module.nearest_neighbors(
            np.asarray(matrix),
            k + 1,
            "euclidean",
            {},
            False,
            random_state=seed,
            n_jobs=-1,
            low_memory=True,
            verbose=False,
            nearest_neighbor_backend="rinnd",
            compatibility_layout=False,
            graph_only=True,
        )
    finally:
        umap_module.RINND = original_rinnd
        umap_module.NNDescent = original_nnd
    if index is not None:
        raise RuntimeError("graph-only reference returned a search index")
    ids = np.asarray(ids, dtype=np.int32)
    if ids.ndim != 2 or ids.shape[0] != len(matrix) or ids.shape[1] < k:
        raise RuntimeError(f"unexpected RINND graph shape {ids.shape}")
    nonself = np.empty((len(matrix), k), dtype=np.int32)
    for row, candidates in enumerate(ids):
        candidates = candidates[candidates != row]
        if len(candidates) < k:
            raise RuntimeError(f"RINND row {row} returned fewer than {k} non-self IDs")
        nonself[row] = candidates[:k]
    if calls["rinnd"] < 1 or calls["pynndescent"] or not all(graph_only_calls):
        raise RuntimeError(f"reference dispatch was not proven: {calls}")
    return nonself, {
        "umap_learn_version": version,
        "rinnd_construction_calls": calls["rinnd"],
        "graph_only_calls": len(graph_only_calls),
        "all_graph_only": all(graph_only_calls),
        "pynndescent_calls": calls["pynndescent"],
        "nearest_neighbor_backend": "rinnd",
        "graph_only": True,
        "n_neighbors": k + 1,
        "random_state": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="exported f32le fixture")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--rust-ids", type=Path, required=True, help="exported i32le test graph")
    parser.add_argument("--width", type=int, required=True, help="exported graph width")
    parser.add_argument("--k", type=int, required=True, help="non-self recall width")
    parser.add_argument("--sample-rows", type=Path, help="optional u64le row indices")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--rust-recall", type=float, help="printed Rust score for P0 parity check")
    parser.add_argument("--out", type=Path, required=True, help="new receipt JSON path")
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite existing receipt: {args.out}")
    if args.width < args.k or args.k <= 0 or args.rows <= args.k:
        parser.error("require rows > k > 0 and width >= k")
    matrix, matrix_info = _read_matrix(args.input, args.rows, args.features)
    rust_ids, rust_info = _read_ids(args.rust_ids, args.rows, args.width)
    rows = _sample_rows(args.sample_rows, args.rows)
    rust_recall = tie_aware_recall(matrix, rust_ids, args.k, rows, exclude_self=True)
    parity = None
    if args.rust_recall is not None:
        parity = {
            "reported": args.rust_recall,
            "scored": rust_recall,
            "passed": abs(args.rust_recall - rust_recall) <= 5e-10,
        }
        if not parity["passed"]:
            raise RuntimeError(f"P0 scorer parity failed: {parity}")
    rinnd_ids, dispatch = _fit_rinnd(matrix, args.k, args.seed)
    rinnd_recall = tie_aware_recall(matrix, rinnd_ids, args.k, rows, exclude_self=True)
    receipt = {
        "schema": "ann-rinnd-fixed-recall-v1",
        "input": matrix_info,
        "rust_graph": rust_info,
        "scoring_rows": {
            "count": len(rows),
            "sha256": hashlib.sha256(np.asarray(rows, dtype="<u8").tobytes()).hexdigest(),
            "path": str(args.sample_rows.resolve()) if args.sample_rows else None,
        },
        "scoring": {
            "k_nonself": args.k,
            "tie_rule": "unique IDs at or below the exact k-th non-self distance count as hits",
            "exclude_self": True,
            "rust_p0": rust_recall,
            "rinnd": rinnd_recall,
            "p0_parity": parity,
        },
        "reference_dispatch": dispatch,
        "driver": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "python": platform.python_version(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps(receipt, allow_nan=False))


if __name__ == "__main__":
    main()
