#!/usr/bin/env -S uv run --no-sync
"""Capture the same-host S1 RINND graph-only build reference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from ann_rinnd_bench import make_s1, sha256
from umap_reference import reference_kwargs, require_reference


class _CapturedCall(Exception):
    pass


class _UnexpectedPyNNDescent(RuntimeError):
    pass


def capture_call(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
    import umap
    import umap.umap_ as umap_module

    original_rinnd = umap_module.RINND
    original_nnd = umap_module.NNDescent
    if original_rinnd is None:
        raise RuntimeError("umap-learn did not bind RINND")
    calls = {"rinnd": 0, "pynndescent": 0}
    captured: dict[str, Any] = {}

    def intercept(*args: Any, **kwargs: Any) -> None:
        calls["rinnd"] += 1
        if len(args) != 1 or not isinstance(args[0], np.ndarray):
            raise RuntimeError("unexpected RINND positional arguments")
        captured.update(data=args[0], kwargs=kwargs.copy())
        raise _CapturedCall

    def reject_pynndescent(*args: Any, **kwargs: Any) -> None:
        calls["pynndescent"] += 1
        raise _UnexpectedPyNNDescent("PyNNDescent fallback was invoked")

    umap_module.RINND = intercept
    umap_module.NNDescent = reject_pynndescent
    try:
        options = reference_kwargs(
            optimizer="adam",
            init="recursive",
            n_neighbors=30,
            random_state=None,
            metric="euclidean",
            n_jobs=-1,
            nearest_neighbor_backend="rinnd",
            force_approximation_algorithm=True,
        )
        try:
            umap.UMAP(**options).fit(matrix)
        except _CapturedCall:
            pass
        else:
            raise RuntimeError("UMAP fit did not invoke RINND")
    finally:
        umap_module.RINND = original_rinnd
        umap_module.NNDescent = original_nnd

    if calls != {"rinnd": 1, "pynndescent": 0}:
        raise RuntimeError(f"unexpected backend dispatch: {calls}")
    data = captured["data"]
    kwargs = captured["kwargs"]
    if data.shape != matrix.shape or not np.shares_memory(data, matrix):
        raise RuntimeError("UMAP changed the intercepted input matrix")
    if kwargs.get("n_jobs") != -1 or kwargs.get("n_neighbors") != 30:
        raise RuntimeError(f"unexpected RINND thread or neighbor settings: {kwargs}")
    return data, kwargs, calls


def checked_stats(index: Any) -> dict[str, Any]:
    stats = dict(index.build_stats)
    vectors = (
        "update_generation_seconds",
        "update_application_seconds",
        "candidate_seconds",
        "updates",
    )
    scalars = (
        "leaf_initialization_seconds",
        "forest_seconds",
        "nn_descent_seconds",
        "total_seconds",
    )
    result = {key: stats[key] for key in (*vectors, *scalars)}
    count = len(result["updates"])
    if count == 0 or any(len(result[key]) != count for key in vectors):
        raise ValueError("RINND build statistics have inconsistent iteration counts")
    for key in (*vectors[:-1], *scalars):
        values = result[key] if key in vectors else [result[key]]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"invalid RINND statistic: {key}")
    if any(not isinstance(value, int) or value < 0 for value in result["updates"]):
        raise ValueError("invalid RINND update counts")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite receipt: {args.out}")
    if "RAYON_NUM_THREADS" in os.environ:
        raise RuntimeError("RAYON_NUM_THREADS must be unset")
    cpu_count = os.cpu_count()
    affinity_count = len(os.sched_getaffinity(0))

    require_reference()
    import rinnd
    import rinnd.rinnd as rinnd_extension
    import umap.umap_ as umap_module

    with tempfile.TemporaryDirectory(prefix="rinnd-s1-") as directory:
        input_path = Path(directory) / "s1.f32le"
        input_info = make_s1(input_path)
        input_info["path"] = "temporary generated input (removed after run)"
        matrix = np.fromfile(input_path, dtype="<f4").reshape(250_000, 256)
        data, original_kwargs, calls = capture_call(matrix)
        # Graph-only mode is the sole intentional change to the intercepted call.
        kwargs = dict(original_kwargs, graph_only=True)

        # Warm-up and timing surround construction only; reading build_stats is outside the timer.
        warmup = rinnd.RINND(data, **kwargs)
        warmup_stats = checked_stats(warmup)
        del warmup
        runs = []
        for _ in range(2):
            started = time.perf_counter()
            index = rinnd.RINND(data, **kwargs)
            wall_seconds = time.perf_counter() - started
            runs.append({"wall_seconds": wall_seconds, "build_stats": checked_stats(index)})
            del index

    def identity(path: str) -> dict[str, str]:
        resolved = Path(path).resolve()
        return {"path": str(resolved), "sha256": sha256(resolved)}

    receipt = {
        "schema": "rinnd-build-stats-v1",
        "input": input_info,
        "dispatch": {
            "calls": calls,
            "estimator_random_state": None,
            "estimator_n_neighbors": 30,
            "intercepted_kwargs": original_kwargs,
            "graph_only_kwargs": kwargs,
        },
        "threads": {
            "rayon_num_threads": None,
            "os_cpu_count": cpu_count,
            "affinity_cpu_count": affinity_count,
            "rinnd_n_jobs": kwargs["n_jobs"],
            "expected_logical_cpus": 16,
            "observed_counts_match_expected": cpu_count == 16 and affinity_count == 16,
            "default_pool_threads": None,
            "note": "n_jobs=-1 requests Rust available_parallelism; pool size is not exposed",
        },
        "provenance": {
            "driver": identity(__file__),
            "input_writer": identity(str(Path(__file__).with_name("ann_rinnd_bench.py"))),
            "umap_module": identity(umap_module.__file__),
            "rinnd_extension": identity(rinnd_extension.__file__),
            "umap_learn_version": importlib.metadata.version("umap-learn"),
            "rinnd_version": importlib.metadata.version("rinnd"),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "warmup_build_stats": warmup_stats,
        "runs": runs,
        "g1_reference_s": statistics.median(
            run["build_stats"]["update_generation_seconds"][0]
            + run["build_stats"]["update_application_seconds"][0]
            for run in runs
        ),
        "g1_generation_s": statistics.median(
            run["build_stats"]["update_generation_seconds"][0] for run in runs
        ),
        "g2_reference_s": statistics.median(run["wall_seconds"] for run in runs),
    }
    serialized = json.dumps(receipt, indent=2, allow_nan=False, default=lambda value: value.item())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(serialized + "\n")


if __name__ == "__main__":
    main()
