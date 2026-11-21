"""Exact, deterministic quality metrics for the UMAP 0.6 campaign."""

import hashlib
import multiprocessing
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any
from zipfile import BadZipFile

import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import KDTree

_CHUNK = 64
_METRICS = ("trustworthiness", "transform_trustworthiness", "preservation", "preservation_qr")


def _array(value, name):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _ids(ids, n, size=None):
    result = np.asarray(ids)
    if result.ndim != 1 or not np.issubdtype(result.dtype, np.integer):
        raise ValueError("query_ids must be a one-dimensional integer array")
    if (size is not None and len(result) != size) or not len(result):
        raise ValueError("query_ids has the wrong length or is empty")
    if np.any(result < 0) or np.any(result >= n):
        raise ValueError("query_ids out of range")
    return result.astype(np.int64, copy=False)


def _k(k, n, *, trust=False):
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    if k <= 0 or k > n or (trust and 2 * k >= n):
        raise ValueError("invalid k for the reference size")


_TASK: Callable[[int], Any] | None = None
_IN_WORKER = False


def _run_task(start):
    if _TASK is None:
        raise RuntimeError("worker started without a task")
    return _TASK(start)


def _mark_worker(task):
    global _IN_WORKER, _TASK
    _IN_WORKER = True
    _TASK = task


def _parallel(task, starts):
    """Map `task` over chunk starts in spawned processes, preserving order."""
    starts = list(starts)
    workers = min(os.cpu_count() or 1, len(starts))
    if _IN_WORKER or workers <= 1:
        return [task(start) for start in starts]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=_mark_worker, initargs=(task,)
    ) as pool:
        return list(pool.map(_run_task, starts))


def _neighbors_chunk(start, reference, queries, k, excluded, tree, indices):
    stop = min(start + _CHUNK, len(queries))
    block = queries[start:stop]
    if tree is not None:
        # Query one extra for self, then recover every boundary tie.
        count = k + (excluded is not None)
        d, _ = tree.query(block, k=count)
        radii = np.nextafter(d[:, -1], np.inf)
        candidate_sets = tree.query_radius(block, r=radii)
    chunk_ids = np.empty((stop - start, k), dtype=np.int64)
    chunk_distances = np.empty((stop - start, k), dtype=np.float64)
    for offset, q in enumerate(block):
        candidates = indices if tree is None else candidate_sets[offset]
        delta = reference[candidates] - q
        squared = np.einsum("ij,ij->i", delta, delta)
        if excluded is not None:
            squared[candidates == excluded[start + offset]] = np.inf
        boundary = np.partition(squared, k - 1)[k - 1]
        below = np.flatnonzero(squared < boundary)
        tied = np.flatnonzero(squared == boundary)
        remaining = k - len(below)
        if len(tied) > remaining:
            tied = tied[np.argpartition(candidates[tied], remaining - 1)[:remaining]]
        selected = np.concatenate((below, tied))
        order = selected[np.lexsort((candidates[selected], squared[selected]))]
        chunk_ids[offset] = candidates[order]
        chunk_distances[offset] = np.sqrt(squared[order])
    return chunk_ids, chunk_distances


def _neighbors(reference, queries, k, excluded=None, tree=None):
    n = len(reference)
    _k(k, n - (excluded is not None))
    ids = np.empty((len(queries), k), dtype=np.int64)
    distances = np.empty((len(queries), k), dtype=np.float64)
    indices = np.arange(n)
    if tree is None and reference.shape[1] == 2:
        tree = KDTree(reference)

    chunk = partial(
        _neighbors_chunk,
        reference=reference,
        queries=queries,
        k=k,
        excluded=excluded,
        tree=tree,
        indices=indices,
    )

    starts = range(0, len(queries), _CHUNK)
    for start, (chunk_ids, chunk_distances) in zip(starts, _parallel(chunk, starts), strict=True):
        ids[start : start + len(chunk_ids)] = chunk_ids
        distances[start : start + len(chunk_ids)] = chunk_distances
    return ids, distances


def exact_neighbors(reference, queries=None, *, k=15, query_ids=None):
    """Return Euclidean neighbors, breaking equal distances by reference row ID."""
    reference = _array(reference, "reference")
    if queries is None:
        queries = reference
        if query_ids is not None:
            raise ValueError("query_ids is implicit when queries is None")
        excluded = np.arange(len(reference))
    else:
        queries = _array(queries, "queries")
        if queries.shape[1] != reference.shape[1]:
            raise ValueError("reference and query dimensions differ")
        excluded = None if query_ids is None else _ids(query_ids, len(reference), len(queries))
    return _neighbors(reference, queries, k, excluded)


def _fit_data(x, y, k, query_ids):
    x, y = _array(x, "x"), _array(y, "y")
    if len(x) != len(y):
        raise ValueError("x and y must have equal row counts")
    _k(k, len(x), trust=True)
    q = np.arange(len(x)) if query_ids is None else _ids(query_ids, len(x))
    if len(np.unique(q)) != len(q):
        raise ValueError("query_ids must be unique")
    return x, y, q


def _ranks(squared, ordered, ids):
    """One-based ranks of `ids` under (distance, row ID) ordering, without a full sort."""
    values = squared[ids]
    below = np.searchsorted(ordered, values, side="left")
    equal = np.searchsorted(ordered, values, side="right") - below
    ranks = below + 1
    for i in np.flatnonzero(equal > 1):
        ranks[i] += np.count_nonzero(squared[: ids[i]] == values[i])
    return ranks


def _fit_scores_chunk(start, x, embeddings, q, k, trees):
    rows = q[start : start + _CHUNK]
    neighbors = {
        name: _neighbors(y, y[rows], k, rows, trees[name])[0] for name, y in embeddings.items()
    }
    chunk_penalties = {name: np.empty(len(rows)) for name in embeddings}
    chunk_preservation = {name: np.empty(len(rows)) for name in embeddings}
    for offset, row in enumerate(rows):
        delta = x - x[row]
        squared = np.einsum("ij,ij->i", delta, delta)
        squared[row] = np.inf
        ordered = np.sort(squared)
        for name, ids in neighbors.items():
            selected_ranks = _ranks(squared, ordered, ids[offset])
            chunk_penalties[name][offset] = np.maximum(selected_ranks - k, 0).sum()
            chunk_preservation[name][offset] = np.count_nonzero(selected_ranks <= k) / k
    return chunk_penalties, chunk_preservation


def _fit_scores_many(x, embeddings, q, k):
    if not embeddings:
        return {}
    n = len(x)
    trees = {name: KDTree(y) if y.shape[1] == 2 else None for name, y in embeddings.items()}
    penalties = {name: np.empty(len(q), dtype=np.float64) for name in embeddings}
    preservation = {name: np.empty(len(q), dtype=np.float64) for name in embeddings}

    chunk = partial(_fit_scores_chunk, x=x, embeddings=embeddings, q=q, k=k, trees=trees)

    starts = range(0, len(q), _CHUNK)
    for start, (chunk_penalties, chunk_preservation) in zip(
        starts, _parallel(chunk, starts), strict=True
    ):
        for name in embeddings:
            stop = start + len(chunk_penalties[name])
            penalties[name][start:stop] = chunk_penalties[name]
            preservation[name][start:stop] = chunk_preservation[name]
    result = {}
    for name in embeddings:
        contributions = 1.0 - 2.0 * penalties[name] / (k * (2 * n - 3 * k - 1))
        se = float(np.std(contributions, ddof=1) / np.sqrt(len(q))) if len(q) > 1 else 0.0
        result[name] = (float(np.mean(contributions)), se, float(np.mean(preservation[name])))
    return result


def _fit_scores(x, y, q, k):
    return _fit_scores_many(x, {"embedding": y}, q, k)["embedding"]


def trustworthiness(x, y, *, k=15, query_ids=None):
    x, y, q = _fit_data(x, y, k, query_ids)
    value, se, _ = _fit_scores(x, y, q, k)
    return {"value": value, "standard_error": se}


def diagnostics(x, y, *, labels=None, pair_count=20_000, seed=20261014):
    """Return fixed-pair and optional class-centroid rank correlations and scale."""
    x, y = _array(x, "x"), _array(y, "y")
    if len(x) != len(y) or len(x) < 2 or pair_count <= 0:
        raise ValueError("diagnostics require equal row counts, at least two rows and pairs")
    rng = np.random.default_rng(seed)
    first = rng.integers(0, len(x), size=pair_count)
    second = rng.integers(0, len(x) - 1, size=pair_count)
    second += second >= first
    dx = np.linalg.norm(x[first] - x[second], axis=1)
    dy = np.linalg.norm(y[first] - y[second], axis=1)
    result = {
        "pair_spearman": float(spearmanr(dx, dy).statistic),
        "axis_std": np.std(y, axis=0).tolist(),
        "bounding_box": {"min": np.min(y, axis=0).tolist(), "max": np.max(y, axis=0).tolist()},
    }
    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != len(x):
            raise ValueError("labels must have one entry per row")
        classes, inverse = np.unique(labels, return_inverse=True)
        if len(classes) < 3:
            raise ValueError("centroid correlation requires at least three classes")
        counts = np.bincount(inverse)
        cx = np.zeros((len(classes), x.shape[1]))
        cy = np.zeros((len(classes), y.shape[1]))
        np.add.at(cx, inverse, x)
        np.add.at(cy, inverse, y)
        cx /= counts[:, None]
        cy /= counts[:, None]
        a, b = np.triu_indices(len(classes), 1)
        result["centroid_spearman"] = float(
            spearmanr(
                np.linalg.norm(cx[a] - cx[b], axis=1), np.linalg.norm(cy[a] - cy[b], axis=1)
            ).statistic
        )
    return result


def score_fit(x, y, *, query_ids=None, k=15):
    return score_fits(x, {"embedding": y}, query_ids=query_ids, k=k)["embedding"]


def score_fits(x, embeddings: dict[str, np.ndarray], *, query_ids=None, k=15) -> dict[str, dict]:
    """Score all embeddings while computing each input-space rank only once."""
    x = _array(x, "x")
    _k(k, len(x), trust=True)
    q = np.arange(len(x)) if query_ids is None else _ids(query_ids, len(x))
    if len(np.unique(q)) != len(q):
        raise ValueError("query_ids must be unique")
    validated = {}
    for name, embedding in embeddings.items():
        y = _array(embedding, "embedding")
        if len(x) != len(y):
            raise ValueError("x and embeddings must have equal row counts")
        validated[name] = y
    scores = _fit_scores_many(x, validated, q, k)
    return {
        name: {
            "trustworthiness": value,
            "trustworthiness_se": se,
            "preservation": preservation,
            "diagnostics": diagnostics(x, validated[name]),
        }
        for name, (value, se, preservation) in scores.items()
    }


def score_transform(train, fit, queries, embedding, *, k=15):
    return score_transforms(train, {"embedding": fit}, queries, {"embedding": embedding}, k=k)[
        "embedding"
    ]


def score_transforms(
    train, fits: dict[str, np.ndarray], queries, embeddings: dict[str, np.ndarray], *, k=15
) -> dict[str, dict]:
    """Score transform runs, sharing query-to-training neighbors and input ranks."""
    train, queries = _array(train, "train"), _array(queries, "queries")
    if train.shape[1] != queries.shape[1]:
        raise ValueError("reference and query dimensions differ")
    if fits.keys() != embeddings.keys():
        raise ValueError("fits and embeddings must have identical keys")
    _k(k, len(queries), trust=True)
    _k(k, len(train))
    validated_fits, validated_embeddings = {}, {}
    for name in fits:
        fit = _array(fits[name], "fit")
        embedding = _array(embeddings[name], "embedding")
        if len(train) != len(fit) or len(queries) != len(embedding):
            raise ValueError("reference or query row counts differ")
        if fit.shape[1] != embedding.shape[1]:
            raise ValueError("reference and query dimensions differ")
        validated_fits[name], validated_embeddings[name] = fit, embedding
    if not fits:
        return {}
    scores = _fit_scores_many(queries, validated_embeddings, np.arange(len(queries)), k)
    input_ids, _ = _neighbors(train, queries, k)
    result = {}
    for name in fits:
        embedded_ids, _ = _neighbors(validated_fits[name], validated_embeddings[name], k)
        overlap = [
            np.intersect1d(a, b, assume_unique=True).size / k
            for a, b in zip(input_ids, embedded_ids, strict=True)
        ]
        value, se, _ = scores[name]
        result[name] = {
            "trustworthiness": value,
            "trustworthiness_se": se,
            "preservation_qr": float(np.mean(overlap)),
        }
    return result


def knn_recall(reference, query_ids, exact_distances, approximate_ids):
    """Tie-inclusive recall for upstream self-inclusive kNN rows."""
    reference = _array(reference, "reference")
    q = _ids(query_ids, len(reference))
    exact_distances = np.asarray(exact_distances, dtype=np.float64)
    approximate_ids = np.asarray(approximate_ids)
    if (
        exact_distances.ndim != 2
        or approximate_ids.ndim != 2
        or approximate_ids.shape[1] < 2
        or exact_distances.shape[1] < approximate_ids.shape[1] - 1
        or len(approximate_ids) != len(q)
        or len(q) != len(exact_distances)
    ):
        raise ValueError("neighbor arrays must have matching query-by-k shapes")
    if (
        not np.issubdtype(approximate_ids.dtype, np.integer)
        or np.any(approximate_ids < -1)
        or np.any(approximate_ids >= len(reference))
    ):
        raise ValueError("invalid approximate neighbor IDs")
    if not np.isfinite(exact_distances).all() or np.any(exact_distances < 0):
        raise ValueError("invalid exact distances")
    if np.any(np.diff(exact_distances, axis=1) < 0):
        raise ValueError("exact distances must be sorted")
    if np.any((approximate_ids[:, 0] != q) & (approximate_ids[:, 0] != -1)):
        raise ValueError("first approximate neighbor must be self or missing")
    neighbor_count = approximate_ids.shape[1] - 1
    hits = 0
    for i, row in enumerate(q):
        ids = approximate_ids[i]
        valid = ids[(ids >= 0) & (ids != row)]
        valid = np.unique(valid)
        d = np.linalg.norm(reference[valid] - reference[row], axis=1)
        count = np.count_nonzero(d <= np.nextafter(exact_distances[i, neighbor_count - 1], np.inf))
        hits += min(neighbor_count, int(count))
    return {
        "recall": hits / (len(q) * neighbor_count),
        "missing": int(np.count_nonzero(approximate_ids == -1)),
        "convention": "self-inclusive",
        "neighbor_count": neighbor_count,
    }


QUERY_TIE_RULE = "distance,training_id"


def content_sha256(values) -> str:
    """SHA-256 of an array's shape and little-endian float32 content, in row order."""
    array = np.ascontiguousarray(values, dtype="<f4")
    digest = hashlib.sha256(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def query_recall(train, queries, approximate_ids, *, k=15, exact_ids=None):
    """Recall@k of transform query neighbor IDs against exact training neighbors.

    Exact neighbors use ``_neighbors`` (distance, then training ID), the same rule for both
    libraries. Negative IDs are missing entries; duplicates count once.
    """
    train, queries = _array(train, "train"), _array(queries, "queries")
    if train.shape[1] != queries.shape[1]:
        raise ValueError("reference and query dimensions differ")
    _k(k, len(train))
    approximate_ids = np.asarray(approximate_ids)
    if (
        approximate_ids.ndim != 2
        or approximate_ids.shape != (len(queries), k)
        or not np.issubdtype(approximate_ids.dtype, np.integer)
    ):
        raise ValueError("approximate_ids must be an integer query-by-k array")
    if np.any(approximate_ids >= len(train)):
        raise ValueError("approximate neighbor IDs out of range")
    if exact_ids is None:
        exact_ids, _ = _neighbors(train, queries, k)
    exact_ids = np.asarray(exact_ids)
    if exact_ids.shape != (len(queries), k) or not np.issubdtype(exact_ids.dtype, np.integer):
        raise ValueError("exact_ids must be an integer query-by-k array")
    per_query = np.empty(len(queries), dtype=np.float64)
    for row, (found, exact) in enumerate(zip(approximate_ids, exact_ids, strict=True)):
        valid = np.unique(found[found >= 0])
        per_query[row] = np.intersect1d(valid, exact, assume_unique=False).size / k
    return {
        "recall": float(np.mean(per_query)),
        "per_query": per_query,
        "missing": int(np.count_nonzero(approximate_ids < 0)),
        "k": k,
        "tie_rule": QUERY_TIE_RULE,
    }


def _cache_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_ids(n):
    return (
        np.arange(n)
        if n <= 20_000
        else np.random.default_rng(20261014).choice(n, 5000, replace=False)
    )


def verify_cache(root: Path, name: str) -> list[str]:
    """Return cache integrity errors, including missing or untracked caches."""
    from . import datasets

    root = Path(root)
    path = root / name / "eval.npz"
    sidecar = path.with_suffix(".npz.sha256")
    try:
        train, _, _ = datasets.load_fixture(root, name)
        fixture_hash = datasets.fixture_hash(root, name)
        expected_hash = sidecar.read_text(encoding="ascii").strip()
        if len(expected_hash) != 64 or expected_hash != _cache_digest(path):
            raise ValueError("cache SHA256 mismatch")
        n = len(train)
        q = _evaluation_ids(n)
        with np.load(path, allow_pickle=False) as cache:
            expected = {"fixture_hash", "query_ids", "neighbor_ids", "neighbor_distances"}
            if name == "mix_50k":
                expected |= {"precomputed_ids", "precomputed_distances"}
            if set(cache.files) != expected:
                raise ValueError("unexpected cache fields")
            if cache["fixture_hash"].shape != () or cache["fixture_hash"].item() != fixture_hash:
                raise ValueError("cache fixture hash mismatch")
            if not np.array_equal(cache["query_ids"], q):
                raise ValueError("cache query IDs do not match the evaluation sample")
            for prefix, rows in (("neighbor", q), ("precomputed", np.arange(n))):
                if prefix == "precomputed" and name != "mix_50k":
                    continue
                ids = cache[prefix + "_ids"]
                distances = cache[prefix + "_distances"]
                if ids.shape != (len(rows), 15) or distances.shape != ids.shape:
                    raise ValueError(f"invalid {prefix} array shapes")
                if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or np.any(ids >= n):
                    raise ValueError(f"invalid {prefix} IDs")
                if (
                    not np.issubdtype(distances.dtype, np.floating)
                    or not np.isfinite(distances).all()
                    or np.any(distances < 0)
                ):
                    raise ValueError(f"invalid {prefix} distances")
                if np.any(np.diff(distances, axis=1) < 0):
                    raise ValueError(f"unsorted {prefix} distances")
                if np.any(np.diff(np.sort(ids, axis=1), axis=1) == 0):
                    raise ValueError(f"duplicate {prefix} IDs")
                if prefix == "neighbor" and np.any(ids == rows[:, None]):
                    raise ValueError("evaluation neighbors include self")
                if prefix == "precomputed" and (
                    np.any(ids[:, 0] != rows) or np.any(distances[:, 0] != 0)
                ):
                    raise ValueError("precomputed neighbors must start with self")
    except (OSError, ValueError, KeyError, TypeError, BadZipFile, EOFError) as error:
        return [f"{path}: {error}"]
    return []


def build_cache(root: Path, name: str) -> Path:
    """Build fixture-keyed exact evaluation neighbors and the 50k precomputed graph."""
    from . import datasets

    root = Path(root)
    path = root / name / "eval.npz"
    sidecar = path.with_suffix(".npz.sha256")
    if path.exists() or sidecar.exists():
        errors = verify_cache(root, name)
        if errors:
            raise ValueError(
                "refusing to overwrite invalid or untracked cache: " + "; ".join(errors)
            )
        return path
    train, _, _ = datasets.load_fixture(root, name)
    fixture_hash = datasets.fixture_hash(root, name)
    n = len(train)
    q = _evaluation_ids(n)
    ids, distances = exact_neighbors(train, train[q], k=15, query_ids=q)
    contents = {
        "fixture_hash": np.array(fixture_hash),
        "query_ids": q,
        "neighbor_ids": ids,
        "neighbor_distances": distances,
    }
    if name == "mix_50k":
        other_ids, other_distances = exact_neighbors(train, k=14)
        contents["precomputed_ids"] = np.column_stack((np.arange(n), other_ids))
        contents["precomputed_distances"] = np.column_stack((np.zeros(n), other_distances))
    with path.open("xb") as stream:
        np.savez(stream, allow_pickle=False, **contents)
    with sidecar.open("x", encoding="ascii") as stream:
        stream.write(_cache_digest(path) + "\n")
    return path


def acceptance_margins(records: list[dict]) -> dict:
    """Pool per-configuration sample variances, independently per fixture and metric."""
    groups = {}
    for record in records:
        fixture, config = record["fixture"], record["configuration"]
        for metric in _METRICS:
            if metric in record:
                value = float(record[metric])
                if not np.isfinite(value):
                    raise ValueError("metric values must be finite")
                groups.setdefault((fixture, metric), {}).setdefault(config, []).append(value)
    result = {}
    for (fixture, metric), configs in groups.items():
        if any(len(values) < 2 for values in configs.values()):
            raise ValueError("each configuration requires at least two seed values")
        variance = np.mean([np.var(values, ddof=1) for values in configs.values()])
        floor = 0.01 if "trustworthiness" in metric else 0.02
        result.setdefault(fixture, {})[metric] = max(floor, 2 * float(np.sqrt(variance)))
    return result


def acceptance_pass(candidate, upstream, margin) -> bool:
    candidate, upstream, margin = float(candidate), float(upstream), float(margin)
    if not np.isfinite([candidate, upstream, margin]).all() or margin < 0:
        raise ValueError("values must be finite and margin nonnegative")
    return candidate >= upstream - margin
