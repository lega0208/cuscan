"""Isolated upstream fits; never imports cuscan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
from comparison.umap_reference import reference_kwargs, require_reference

from .datasets import load_fixture
from .spec import CONFIGS, FIXTURES, warmup_plan


def embedding_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def validate_embedding(values: np.ndarray, rows: int, components: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (rows, components) or not np.isfinite(values).all():
        raise ValueError(f"Invalid embedding: {values.shape}, expected {(rows, components)} finite")
    return values


STAGE_HOOKS = (
    "smooth_knn_dist",
    "compute_membership_strengths",
    "init_graph_transform",
    "make_epochs_per_sample",
)


def _store(capture: dict[str, Any], key: str, value: Any) -> None:
    if key in capture:
        raise RuntimeError(f"Upstream stage {key!r} was called more than once in one transform")
    capture[key] = value


def _copy(values: Any) -> np.ndarray:
    array = np.array(values, copy=True)
    if array.dtype == object:
        raise RuntimeError("Stage capture produced an object array")
    return array


def install_stage_hooks(upstream: Any, holder: list) -> None:
    """Wrap the module globals that ``UMAP.transform`` resolves at call time.

    The wrappers are pass-through and record only while ``holder[0]`` is a dict, so
    fits and the default transforms are unaffected.
    """
    smooth = upstream.smooth_knn_dist
    memberships = upstream.compute_membership_strengths
    initialize = upstream.init_graph_transform
    schedule = upstream.make_epochs_per_sample

    def smooth_knn_dist(distances, k, *positional, **keywords):
        sigmas, rhos = smooth(distances, k, *positional, **keywords)
        capture = holder[0]
        if capture is not None:
            _store(capture, "smooth_k", float(k))
            _store(capture, "smooth_keywords", {key: float(v) for key, v in keywords.items()})
            _store(capture, "sigmas", _copy(sigmas))
            _store(capture, "rhos", _copy(rhos))
        return sigmas, rhos

    def compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos, *rest, **keywords):
        capture = holder[0]
        if capture is not None:
            _store(capture, "knn_ids", _copy(knn_indices))
            _store(capture, "knn_distances", _copy(knn_dists))
            _store(capture, "membership_bipartite", bool(keywords.get("bipartite", False)))
        result = memberships(knn_indices, knn_dists, sigmas, rhos, *rest, **keywords)
        if capture is not None:
            rows, cols, vals, dists = result
            _store(capture, "membership_rows", _copy(rows))
            _store(capture, "membership_cols", _copy(cols))
            _store(capture, "membership_vals", _copy(vals))
            if dists is not None:
                _store(capture, "membership_dists", _copy(dists))
        return result

    def init_graph_transform(graph, embedding):
        result = initialize(graph, embedding)
        capture = holder[0]
        if capture is not None:
            _store(capture, "init_csr_indptr", _copy(graph.indptr))
            _store(capture, "init_csr_indices", _copy(graph.indices))
            _store(capture, "init_csr_data", _copy(graph.data))
            _store(capture, "initial_placement", _copy(result))
        return result

    def make_epochs_per_sample(weights, n_epochs):
        result = schedule(weights, n_epochs)
        capture = holder[0]
        if capture is not None:
            _store(capture, "schedule_weights", _copy(weights))
            _store(capture, "schedule_n_epochs", int(n_epochs))
        return result

    hooks = {
        "smooth_knn_dist": smooth_knn_dist,
        "compute_membership_strengths": compute_membership_strengths,
        "init_graph_transform": init_graph_transform,
        "make_epochs_per_sample": make_epochs_per_sample,
    }
    for name in STAGE_HOOKS:
        setattr(upstream, name, hooks[name])


def capture_layout_call(capture: dict[str, Any], positional: tuple, keywords: dict) -> None:
    """Record the arguments ``optimize_layout_euclidean`` consumes, before it mutates them."""
    names = (
        "head_embedding",
        "tail_embedding",
        "head",
        "tail",
        "n_epochs",
        "n_vertices",
        "epochs_per_sample",
        "a",
        "b",
        "rng_state",
        "gamma",
        "initial_alpha",
        "negative_sample_rate",
        "parallel",
    )
    bound = dict(zip(names[: len(positional)], positional, strict=True))
    overlap = set(bound) & set(keywords)
    if overlap:
        raise RuntimeError(f"Duplicate layout arguments: {sorted(overlap)}")
    bound.update(keywords)
    _store(
        capture,
        "layout",
        {
            key: (_copy(value) if isinstance(value, np.ndarray) else value)
            for key, value in bound.items()
            if key not in ("tqdm_kwds", "densmap_kwds", "random_state", "verbose")
        },
    )


def stage_arrays(
    capture: dict[str, Any], final: np.ndarray, layouts: Any
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Assemble the replay inputs in the order upstream's optimizer consumes them."""
    missing = [
        key
        for key in (
            "sigmas",
            "knn_ids",
            "membership_vals",
            "initial_placement",
            "layout",
            "schedule_weights",
        )
        if key not in capture
    ]
    if missing:
        raise RuntimeError(f"Upstream transform skipped observed stages: {missing}")
    layout = capture["layout"]
    optimizer = str(layout.get("optimizer", "adam"))
    n_epochs = layout["n_epochs"]
    if isinstance(n_epochs, list):
        raise RuntimeError("Transform layout received an epoch list")
    n_epochs = int(n_epochs)
    if n_epochs != capture["schedule_n_epochs"]:
        raise RuntimeError("Schedule and layout epoch counts differ")
    epochs_per_sample = layout["epochs_per_sample"]
    weights = capture["schedule_weights"]
    if optimizer == "compatibility":
        heads = np.asarray(layout["head"])
        tails = np.asarray(layout["tail"])
        indptr = None
    else:
        indptr = np.asarray(layout["csr_indptr"])
        tails = np.asarray(layout["csr_indices"])
        heads = np.repeat(np.arange(len(indptr) - 1, dtype=tails.dtype), np.diff(indptr))
        csr_data = layout.get("csr_data")
        if csr_data is None or not np.array_equal(csr_data, weights):
            raise RuntimeError("Modern CSR data differs from the scheduled weights")
    if not (len(heads) == len(tails) == len(weights) == len(epochs_per_sample)):
        raise RuntimeError("Consumed edge stream and schedule lengths differ")
    # Replicates layouts.optimize_layout_euclidean (0.6.0 lines 1806-1808, 1840-1842):
    # these local periods and clocks are not returned by upstream.
    epochs_per_negative = epochs_per_sample / layout["negative_sample_rate"]
    next_negative = epochs_per_negative.copy()
    next_sample = epochs_per_sample.copy()
    if optimizer != "compatibility":
        epochs_per_negative *= 1.5
        next_negative *= 1.5
    good_initialization = bool(layout.get("good_initialization", False))
    alpha_schedule = layouts._create_alpha_schedule(
        optimizer, n_epochs, layout["initial_alpha"], good_initialization
    )
    arrays = {
        "knn_ids": capture["knn_ids"],
        "knn_distances": capture["knn_distances"],
        "sigmas": capture["sigmas"],
        "rhos": capture["rhos"],
        "membership_rows": capture["membership_rows"],
        "membership_cols": capture["membership_cols"],
        "membership_vals": capture["membership_vals"],
        "init_csr_indptr": capture["init_csr_indptr"],
        "init_csr_indices": capture["init_csr_indices"],
        "init_csr_data": capture["init_csr_data"],
        "initial_placement": capture["initial_placement"],
        "layout_head_embedding": layout["head_embedding"],
        "edge_heads": heads,
        "edge_tails": tails,
        "edge_weights": weights,
        "epochs_per_sample": epochs_per_sample,
        "epochs_per_negative_sample": epochs_per_negative,
        "epoch_of_next_sample": next_sample,
        "epoch_of_next_negative_sample": next_negative,
        "alpha_schedule": np.asarray(alpha_schedule),
        "rng_state": layout["rng_state"],
        "final_embedding": final,
    }
    if "membership_dists" in capture:
        arrays["membership_dists"] = capture["membership_dists"]
    if "query_ids_raw" in capture:
        arrays["query_ids_raw"] = capture["query_ids_raw"]
        arrays["query_distances_raw"] = capture["query_distances_raw"]
    if indptr is not None:
        arrays["edge_csr_indptr"] = indptr
    if not np.array_equal(layout["head_embedding"], capture["initial_placement"], equal_nan=True):
        raise RuntimeError("Layout did not start from the captured initial placement")
    metadata = {
        "optimizer": optimizer,
        "edge_order": "coo" if optimizer == "compatibility" else "sorted_csr",
        "n_epochs": n_epochs,
        "n_vertices": int(layout["n_vertices"]),
        "a": float(layout["a"]),
        "b": float(layout["b"]),
        "gamma": float(layout["gamma"]),
        "initial_alpha": float(layout["initial_alpha"]),
        "negative_sample_rate": float(layout["negative_sample_rate"]),
        "parallel": bool(layout["parallel"]),
        "good_initialization": good_initialization,
        "move_other": bool(layout.get("move_other", False)),
        "negative_selection_range": layout.get("negative_selection_range"),
        "negative_sample_scale": layout.get("negative_sample_scale"),
        "exclude_graph_neighbors": layout.get("exclude_graph_neighbors"),
        "smooth_k": capture["smooth_k"],
        "smooth_keywords": capture["smooth_keywords"],
        "membership_bipartite": capture["membership_bipartite"],
        "edges": len(weights),
        "derived_not_intercepted": [
            "epochs_per_negative_sample",
            "epoch_of_next_sample",
            "epoch_of_next_negative_sample",
            "alpha_schedule",
            "edge_heads" if indptr is not None else None,
        ],
    }
    metadata["derived_not_intercepted"] = [
        name for name in metadata["derived_not_intercepted"] if name is not None
    ]
    return arrays, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, choices=FIXTURES)
    parser.add_argument("--config", required=True, choices=CONFIGS)
    parser.add_argument("--seeds", required=True, type=int, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/umap06"))
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--precomputed-knn", action="store_true")
    parser.add_argument("--record-query-knn", action="store_true")
    parser.add_argument(
        "--stage-embedding",
        type=Path,
        help="Opt-in D6 replay capture: cuscan fit embedding (.npy) substituted for embedding_",
    )
    parser.add_argument("--stage-batches", type=int, nargs="+")
    parser.add_argument(
        "--stage-transform-epochs",
        type=int,
        nargs="+",
        help="Transform epochs to capture; sets n_epochs=3*E for the capture only",
    )
    parser.add_argument("--expected-ab", type=float, nargs=2, metavar=("A", "B"))
    parser.add_argument("--ab-rtol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.precomputed_knn and args.record_query_knn:
        parser.error("--record-query-knn requires an upstream search index")
    staged = args.stage_embedding is not None
    if not staged and (
        args.stage_batches or args.stage_transform_epochs or args.expected_ab is not None
    ):
        parser.error("stage options require --stage-embedding")
    if staged and args.precomputed_knn:
        parser.error("--stage-embedding requires an upstream search index")
    if staged and len(args.seeds) != 1:
        parser.error("--stage-embedding substitutes one fit embedding; pass exactly one seed")
    if args.stage_transform_epochs and min(args.stage_transform_epochs) < 1:
        parser.error("--stage-transform-epochs must be >= 1")
    require_reference()
    import umap
    import umap.layouts as layouts
    import umap.umap_ as upstream

    observed_epochs: list[int] = []
    optimize_layout = upstream.optimize_layout_euclidean
    stage_capture: list = [None]

    def observe_layout(*positional, **keywords):
        epochs = positional[4] if len(positional) > 4 else keywords["n_epochs"]
        observed_epochs.append(int(np.max(epochs)))
        if stage_capture[0] is not None:
            capture_layout_call(stage_capture[0], positional, keywords)
        return optimize_layout(*positional, **keywords)

    upstream.optimize_layout_euclidean = cast(Any, observe_layout)
    stage_embedding = None
    if staged:
        install_stage_hooks(upstream, stage_capture)
        stage_embedding = np.load(args.stage_embedding, allow_pickle=False)

    args.out.mkdir(parents=True, exist_ok=False)
    train, batches, _ = load_fixture(args.data, args.fixture)
    if stage_embedding is not None:
        stage_embedding = validate_embedding(stage_embedding, len(train), args.components)
        unknown = sorted(set(args.stage_batches or ()) - set(batches))
        if unknown:
            parser.error(f"Unknown stage batches {unknown}; available {sorted(batches)}")
    config = CONFIGS[args.config]
    warmup = warmup_plan(FIXTURES[args.fixture])

    def model(seed: int, precomputed: tuple | None = None):
        options = reference_kwargs(
            optimizer=config.optimizer,
            init=config.init,
            n_neighbors=15,
            n_components=args.components,
            min_dist=0.1,
            spread=1.0,
            learning_rate=1.0,
            negative_sample_rate=5,
            repulsion_strength=1.0,
            metric="euclidean",
            random_state=seed,
            transform_seed=seed,
            low_memory=True,
            verbose=False,
            n_jobs=-1,
        )
        if precomputed is not None:
            options["precomputed_knn"] = precomputed
        return umap.UMAP(**options)

    warm_start = time.perf_counter()
    with warnings.catch_warnings(record=True) as warm_warnings:
        warnings.simplefilter("always")
        for rows in warmup["fit_rows"]:
            reducer = model(args.seeds[0]).fit(train[:rows])
            for count in warmup["transform_rows"]:
                # Repeat verified input rows only for compilation; these are not measured queries.
                queries = np.resize(train[: min(len(train), count)], (count, train.shape[1]))
                reducer.transform(queries)
    warm_seconds = time.perf_counter() - warm_start
    precomputed = None
    if args.precomputed_knn:
        with np.load(args.data / args.fixture / "eval.npz", allow_pickle=False) as cache:
            precomputed = (cache["precomputed_ids"], cache["precomputed_distances"], None)
    for seed in args.seeds:
        destination = args.out / str(seed)
        destination.mkdir(exist_ok=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reducer = model(seed, precomputed)
            observed_epochs.clear()
            start = time.perf_counter()
            fit = validate_embedding(reducer.fit_transform(train), len(train), args.components)
            fit_seconds = time.perf_counter() - start
            if not observed_epochs:
                raise RuntimeError("Upstream fit did not call the observed layout entry point")
            fit_epochs = observed_epochs[-1]
            fit_layout_epochs = list(observed_epochs)
            np.save(destination / "fit.npy", fit, allow_pickle=False)
            hashes = {"fit": embedding_hash(fit)}
            transform_seconds = {}
            transform_epochs = {}
            if not args.precomputed_knn:
                for count, queries in batches.items():
                    observed_epochs.clear()
                    start = time.perf_counter()
                    embedding = validate_embedding(
                        reducer.transform(queries), count, args.components
                    )
                    transform_seconds[str(count)] = time.perf_counter() - start
                    if not observed_epochs:
                        raise RuntimeError("Upstream transform did not call observed layout")
                    transform_epochs[str(count)] = observed_epochs[-1]
                    np.save(destination / f"transform_{count}.npy", embedding, allow_pickle=False)
                    hashes[f"transform_{count}"] = embedding_hash(embedding)
                    if args.record_query_knn:
                        index = getattr(reducer, "_knn_search_index", None)
                        if index is None:
                            raise RuntimeError("Query kNN recording requires an approximate index")
                        epsilon = 0.24 if index._angular_trees else 0.12
                        ids, distances = index.query(queries, reducer.n_neighbors, epsilon=epsilon)
                        if (
                            ids.shape != (count, reducer.n_neighbors)
                            or distances.shape != ids.shape
                        ):
                            raise RuntimeError("Unexpected upstream query kNN shape")
                        np.savez(
                            destination / f"query_knn_{count}.npz", ids=ids, distances=distances
                        )
            stages = None
            if stage_embedding is not None:
                stages = record_stages(
                    reducer,
                    stage_embedding,
                    {count: batches[count] for count in (args.stage_batches or sorted(batches))},
                    args.stage_transform_epochs or [None],
                    destination / "stages",
                    stage_capture,
                    layouts,
                    args,
                )
        warning_records = [str(item.message) for item in [*warm_warnings, *caught]]
        for message in warning_records:
            lowered = message.lower()
            if (
                "fallback" in lowered
                or "falling back" in lowered
                or "spectral initialisation failed" in lowered
            ):
                raise RuntimeError(f"Reference initialization fallback: {message}")
        knn = getattr(reducer, "_knn_indices", None)
        if knn is not None:
            np.save(destination / "knn_indices.npy", knn, allow_pickle=False)
        params = reducer.get_params()
        params.pop("precomputed_knn", None)
        record = {
            "role": "upstream",
            "fixture": args.fixture,
            "config": args.config,
            "seed": seed,
            "components": args.components,
            "precomputed_knn": args.precomputed_knn,
            "params": params,
            "resolved_optimizer": reducer._optimizer,
            "resolved_init": reducer._init,
            "epochs": fit_epochs,
            "fit_layout_epochs": fit_layout_epochs,
            "transform_epochs": transform_epochs,
            "fit_seconds": fit_seconds,
            "transform_seconds": transform_seconds,
            "warmup": warmup,
            "warmup_seconds": warm_seconds,
            "warnings": warning_records,
            "hashes": hashes,
            "threads": {
                key: os.environ.get(key)
                for key in (
                    "NUMBA_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "cpu_count": os.cpu_count(),
        }
        if stages is not None:
            record["stages"] = stages
        (destination / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def record_stages(
    reducer: Any,
    embedding: np.ndarray,
    batches: dict[int, np.ndarray],
    transform_epochs: list[int | None],
    out: Path,
    holder: list,
    layouts: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Re-run transforms on cuscan's fit embedding and archive every consumed stage input."""
    a, b = float(reducer._a), float(reducer._b)
    ab_check = None
    if args.expected_ab is not None:
        expected = np.asarray(args.expected_ab, dtype=np.float64)
        close = bool(np.allclose([a, b], expected, rtol=args.ab_rtol, atol=0.0))
        ab_check = {"expected": expected.tolist(), "rtol": args.ab_rtol, "matched": close}
        if not close:
            raise RuntimeError(f"Upstream a/b {(a, b)} differ from cuscan {tuple(expected)}")
    index = getattr(reducer, "_knn_search_index", None)
    if index is None or getattr(reducer, "_small_data", False):
        raise RuntimeError("Stage capture requires the approximate upstream search index")
    out.mkdir(exist_ok=False)
    fitted_embedding = reducer.embedding_
    fitted_epochs = reducer.n_epochs
    query = index.query

    def observed_query(*positional, **keywords):
        ids, distances = query(*positional, **keywords)
        capture = holder[0]
        if capture is not None:
            _store(capture, "query_ids_raw", _copy(ids))
            _store(capture, "query_distances_raw", _copy(distances))
            _store(capture, "query_epsilon", float(keywords.get("epsilon", float("nan"))))
        return ids, distances

    runs = []
    reducer.embedding_ = embedding
    index.query = observed_query
    try:
        for count, queries in batches.items():
            for epochs in transform_epochs:
                reducer.n_epochs = fitted_epochs if epochs is None else 3 * epochs
                holder[0] = {}
                try:
                    final = validate_embedding(reducer.transform(queries), count, args.components)
                    capture = holder[0]
                finally:
                    holder[0] = None
                arrays, metadata = stage_arrays(capture, final, layouts)
                if epochs is not None and metadata["n_epochs"] != epochs:
                    raise RuntimeError("Upstream transform epochs differ from the request")
                name = f"transform_{count}_e{metadata['n_epochs']}"
                path = out / f"{name}.npz"
                if path.exists():
                    raise RuntimeError(f"Duplicate stage capture {name}")
                # The dynamic array names are not NumPy's allow_pickle parameter.
                np.savez(path, **cast(dict[str, Any], arrays))
                runs.append(
                    {
                        "name": name,
                        "file": f"stages/{name}.npz",
                        "rows": count,
                        "query_hash": embedding_hash(queries),
                        "requested_transform_epochs": epochs,
                        "query_epsilon": capture.get("query_epsilon"),
                        "disconnection_distance": (
                            float(reducer._disconnection_distance)
                            if np.isfinite(reducer._disconnection_distance)
                            else None
                        ),
                        **metadata,
                        "arrays": {
                            key: {
                                "dtype": str(value.dtype),
                                "shape": list(value.shape),
                                "sha256": hashlib.sha256(
                                    np.ascontiguousarray(value).tobytes()
                                ).hexdigest(),
                            }
                            for key, value in arrays.items()
                        },
                    }
                )
    finally:
        reducer.embedding_ = fitted_embedding
        reducer.n_epochs = fitted_epochs
        del index.query
    return {
        "schema": "umap06.upstream_stages.v1",
        "embedding_source": str(args.stage_embedding),
        "embedding_hash": embedding_hash(embedding),
        "train_hash": embedding_hash(reducer._raw_data),
        "a": a,
        "b": b,
        "min_dist": reducer.min_dist,
        "spread": reducer.spread,
        "ab_check": ab_check,
        "runs": runs,
    }


if __name__ == "__main__":
    main()
