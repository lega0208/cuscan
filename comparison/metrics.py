"""Fixed-representative quality metrics for embedding comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

EVALUATION_SEED = 20261014
N_NEIGHBORS = 15
N_REPRESENTATIVES = 2000


def unique_rows(x: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.intp], NDArray[np.intp]]:
    """Return lexicographically unique rows plus first and inverse row mappings."""
    matrix = _finite_matrix(x, "x")
    values, first, inverse = np.unique(matrix, axis=0, return_index=True, return_inverse=True)
    return values.astype(np.float32, copy=False), first, inverse


def representative_indices(
    unique_count: int, *, seed: int = EVALUATION_SEED, count: int = N_REPRESENTATIVES
) -> NDArray[np.intp]:
    """Choose a deterministic subset of unique-row indices."""
    if unique_count < 1:
        raise ValueError("unique_count must be positive")
    if count < 1 or count > unique_count:
        raise ValueError("representative count must be between 1 and unique_count")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return np.random.default_rng(seed).choice(unique_count, count, replace=False)


def embedding_metrics(
    reference: ArrayLike,
    embedding: ArrayLike,
    *,
    neighbors: int = N_NEIGHBORS,
) -> dict[str, float]:
    """Measure trustworthiness and exact neighbor overlap for paired points."""
    x = _finite_matrix(reference, "reference")
    y = _finite_matrix(embedding, "embedding")
    if x.shape[0] != y.shape[0]:
        raise ValueError("reference and embedding must have the same number of rows")
    if neighbors < 1 or x.shape[0] <= 2 * neighbors:
        raise ValueError("row count must be greater than twice neighbors")

    trust = trustworthiness(x, y, n_neighbors=neighbors)
    reference_neighbors = (
        NearestNeighbors(n_neighbors=neighbors + 1, algorithm="brute", metric="euclidean")
        .fit(x)
        .kneighbors(x, return_distance=False)[:, 1:]
    )
    embedding_neighbors = (
        NearestNeighbors(n_neighbors=neighbors + 1, algorithm="brute", metric="euclidean")
        .fit(y)
        .kneighbors(y, return_distance=False)[:, 1:]
    )
    overlap = np.mean(
        [
            np.intersect1d(left, right, assume_unique=False).size / neighbors
            for left, right in zip(reference_neighbors, embedding_neighbors, strict=True)
        ]
    )
    return {"trustworthiness_at_15": float(trust), "neighbor_overlap_at_15": float(overlap)}


def _finite_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric 2-D matrix") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty 2-D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix
