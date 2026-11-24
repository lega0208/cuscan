"""Strict-backend cuscan wall baselines; never imports umap."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ImportError:
    resource = None

from .datasets import load_fixture
from .spec import FIXTURES


def checked(values, rows: int, components: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (rows, components) or not np.isfinite(result).all():
        raise ValueError("Invalid cuscan embedding shape or non-finite values")
    return result


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=FIXTURES, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--init", choices=("spectral", "random", "recursive", "auto"))
    parser.add_argument(
        "--init-rescale", choices=("isotropic", "per_axis", "graph_edge"), default="isotropic"
    )
    parser.add_argument("--optimizer", choices=("compatibility", "adam", "momentum"), default=None)
    parser.add_argument(
        "--library-defaults",
        action="store_true",
        help="omit optimizer and init so the library defaults apply (record config 'default')",
    )
    parser.add_argument("--negative-selection-range", type=int, default=200000)
    parser.add_argument("--negative-sample-scale", type=float, default=None)
    parser.add_argument("--exclude-graph-neighbors", choices=("true", "false"), default="false")
    parser.add_argument("--negative-sample-scale-adaptation-samples", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/umap06"))
    parser.add_argument("--components", type=int, default=2)
    parser.add_argument("--neighbor-search", choices=("auto", "exact"), default="auto")
    transform_options = parser.add_mutually_exclusive_group()
    transform_options.add_argument("--skip-transform", action="store_true")
    transform_options.add_argument("--transform-self", action="store_true")
    parser.add_argument("--transform-batches", nargs="+", type=int)
    parser.add_argument("--skip-repeat-fit", action="store_true")
    args = parser.parse_args()
    algorithm: dict[str, Any] = {}
    if args.library_defaults:
        if args.init is not None or args.optimizer is not None:
            parser.error("--library-defaults is mutually exclusive with --init and --optimizer")
    else:
        if args.init is None:
            parser.error("--init is required unless --library-defaults is given")
        if args.optimizer is None:
            args.optimizer = "compatibility"
        algorithm.update(optimizer=args.optimizer, init=args.init)
    from cuscan import UMAP

    args.out.mkdir(parents=True, exist_ok=False)
    train, batches, _ = load_fixture(args.data, args.fixture)
    if args.transform_batches is not None:
        if unknown := set(args.transform_batches) - batches.keys():
            parser.error(f"unavailable transform batches: {sorted(unknown)}")
        batches = {count: batches[count] for count in args.transform_batches}
    settings = {
        "negative_selection_range": args.negative_selection_range,
        "negative_sample_scale": args.negative_sample_scale,
        "exclude_graph_neighbors": args.exclude_graph_neighbors == "true",
        "negative_sample_scale_adaptation_samples": args.negative_sample_scale_adaptation_samples,
    }
    first_hash = None
    for position, seed in enumerate(args.seeds):
        destination = args.out / str(seed)
        destination.mkdir(exist_ok=False)
        model = UMAP(
            backend=args.backend,
            neighbor_search=args.neighbor_search,
            n_neighbors=15,
            n_components=args.components,
            min_dist=0.1,
            spread=1.0,
            learning_rate=1.0,
            negative_sample_rate=5,
            repulsion_strength=1.0,
            **algorithm,
            **settings,
            init_rescale=args.init_rescale,
            random_state=seed,
            transform_seed=seed,
        )
        start = time.perf_counter()
        fit = checked(model.fit_transform(train), len(train), args.components)
        fit_seconds = time.perf_counter() - start
        np.save(destination / "fit.npy", fit, allow_pickle=False)
        _, metadata = model._model.export_state()
        hashes = {"fit": digest(fit)}
        if position == 0:
            first_hash = hashes["fit"]
        transforms = {}
        if not args.skip_transform and (
            args.neighbor_search != "exact" or args.fixture != "mix_50k"
        ):
            for count, queries in batches.items():
                samples = []
                baseline = None
                for _ in range(4):
                    start = time.perf_counter()
                    values = checked(model.transform(queries), count, args.components)
                    samples.append(time.perf_counter() - start)
                    if baseline is not None and digest(values) != digest(baseline):
                        raise RuntimeError("Cold and warm transforms are not bit-identical")
                    baseline = values
                np.save(destination / f"transform_{count}.npy", baseline, allow_pickle=False)
                hashes[f"transform_{count}"] = digest(baseline)
                if (
                    os.environ.get("CUSCAN_UMAP06_RECORD_QUERY_KNN") == "1"
                    and metadata["resolved_search"] == "approximate"
                ):
                    diagnostic, ids = model._model._diagnostic_transform(queries)
                    if digest(diagnostic) != hashes[f"transform_{count}"]:
                        raise RuntimeError("Diagnostic and timed transform are not bit-identical")
                    ids = np.ascontiguousarray(ids, dtype="<i8")
                    key = f"query_neighbors_{count}"
                    np.save(destination / f"{key}.npy", ids, allow_pickle=False)
                    hashes[key] = hashlib.sha256(ids.tobytes()).hexdigest()
                transforms[str(count)] = {"cold": samples[0], "warm": samples[1:]}
        if args.transform_self:
            start = time.perf_counter()
            values = checked(model.transform(train), len(train), args.components)
            transforms["self"] = {"cold": time.perf_counter() - start, "warm": []}
            np.save(destination / "transform_self.npy", values, allow_pickle=False)
            hashes["transform_self"] = digest(values)
        record = {
            "role": "cuscan",
            "fixture": args.fixture,
            "backend": args.backend,
            "config": (
                "default"
                if args.library_defaults
                else f"{'classic' if args.optimizer == 'compatibility' else args.optimizer}"
                f"_{args.init}"
            ),
            "library_defaults": args.library_defaults,
            "init": metadata["init"],
            "optimizer": metadata["optimizer"],
            "seed": seed,
            "components": args.components,
            "precomputed_knn": args.neighbor_search == "exact",
            "neighbor_search": args.neighbor_search,
            "init_rescale": model.init_rescale,
            "epochs": 500 if len(train) <= 10000 else 200,
            "fit_seconds": fit_seconds,
            "fit_temperature": "cold" if position == 0 else "warm",
            "transform_seconds": transforms,
            "hashes": hashes,
            "settings": {name: getattr(model, name) for name in settings},
            "peak_rss_kib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource is not None else None
            ),
        }
        if not args.skip_repeat_fit and position == len(args.seeds) - 1:
            repeat_model = UMAP(
                backend=args.backend,
                neighbor_search=args.neighbor_search,
                n_neighbors=15,
                n_components=args.components,
                min_dist=0.1,
                spread=1.0,
                learning_rate=1.0,
                negative_sample_rate=5,
                repulsion_strength=1.0,
                **algorithm,
                **settings,
                init_rescale=args.init_rescale,
                random_state=args.seeds[0],
                transform_seed=args.seeds[0],
            )
            repeat = checked(repeat_model.fit_transform(train), len(train), args.components)
            record["in_process_reproducibility"] = {
                "seed": args.seeds[0],
                "first_hash": first_hash,
                "repeat_hash": digest(repeat),
                "identical": first_hash == digest(repeat),
            }
        (destination / "run.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
