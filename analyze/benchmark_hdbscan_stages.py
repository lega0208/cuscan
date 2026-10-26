#!/usr/bin/env python3
"""Generate the fixed HDBSCAN workload and run serial CPU/CUDA stage benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from hdbscan_stages_common import atomic_json, require_fields, run_command

ROOT = Path(__file__).resolve().parents[1]
FEATURES = 16
MEASURED_RUNS = 3
STAGES = (
    "input_validation",
    "backend_setup",
    "input_preparation",
    "tree_build",
    "input_upload",
    "core_distances",
    "mst",
    "cluster_extraction",
    "result_finalization",
)
CORE_STAGES = (
    "core_neighbor_kernel",
    "core_extraction",
    "core_neighbor_readback",
    "core_final_assembly_readback",
)
BORUVKA_STAGES = (
    "boruvka_metadata_prepare_upload",
    "boruvka_candidate_kernel",
    "boruvka_component_reduction",
    "boruvka_winner_readback",
    "boruvka_host_merge",
)


def _positive_finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{context} must be finite and nonnegative")
    return result


def _validate_profile(profile: Any, backend: str, *, runs: int = MEASURED_RUNS) -> dict[str, Any]:
    required = (
        "fit_seconds",
        "warmup_seconds",
        "boruvka_samples",
        "core_distance_samples",
        "stage_seconds",
        "cluster_counts",
        "noise_counts",
        "backend",
        "search",
        "resolved_search_per_fit",
    )
    require_fields(profile, required, f"{backend} profile")
    fits = profile["fit_seconds"]
    if not isinstance(fits, list) or len(fits) != runs:
        raise ValueError(f"{backend} profile must contain exactly {runs} fit samples")
    for index, sample in enumerate(fits):
        _positive_finite(sample, f"{backend} fit_seconds[{index}]")
    _positive_finite(profile["warmup_seconds"], f"{backend} warmup_seconds")
    if not isinstance(profile["stage_seconds"], dict) or set(profile["stage_seconds"]) != set(
        STAGES
    ):
        raise ValueError(f"{backend} profile has missing or unexpected top-level stage names")
    for stage in STAGES:
        values = profile["stage_seconds"][stage]
        if not isinstance(values, list) or len(values) != runs:
            raise ValueError(f"{backend} stage {stage} must have {runs} measured samples")
        for index, value in enumerate(values):
            _positive_finite(value, f"{backend} stage {stage}[{index}]")

    core_samples = profile["core_distance_samples"]
    if not isinstance(core_samples, list) or len(core_samples) != runs:
        raise ValueError(f"{backend} core-distance profiles must have {runs} measured samples")
    for fit_index, sample in enumerate(core_samples):
        require_fields(sample, ("stage_seconds", "scope"), f"{backend} core sample {fit_index}")
        stages = sample["stage_seconds"]
        if not isinstance(stages, dict) or set(stages) != set(CORE_STAGES):
            raise ValueError(f"{backend} core sample {fit_index} has invalid subphase names")
        for stage in CORE_STAGES:
            values = stages[stage]
            if not isinstance(values, list):
                raise ValueError(f"{backend} core subphase {stage} must be a duration list")
            for index, value in enumerate(values):
                _positive_finite(value, f"{backend} core sample {fit_index} {stage}[{index}]")

    boruvka_samples = profile["boruvka_samples"]
    if not isinstance(boruvka_samples, list) or len(boruvka_samples) != runs:
        raise ValueError(f"{backend} Boruvka profiles must have {runs} measured samples")
    for fit_index, sample in enumerate(boruvka_samples):
        require_fields(
            sample,
            (
                "total_seconds",
                "round_count",
                "round_seconds",
                "round_stage_seconds",
                "component_counts_per_round",
                "seed_init_seconds",
                "finalize_seconds",
                "seed_and_round_overhead_seconds",
                "scope",
            ),
            f"{backend} Boruvka sample {fit_index}",
        )
        rounds = sample["round_count"]
        components = sample["component_counts_per_round"]
        round_times = sample["round_seconds"]
        if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 0:
            raise ValueError(f"{backend} Boruvka sample {fit_index} has invalid round_count")
        if not isinstance(components, list) or len(components) != rounds:
            raise ValueError(
                f"{backend} Boruvka sample {fit_index} component counts do not align with rounds"
            )
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in components
        ):
            raise ValueError(f"{backend} Boruvka sample {fit_index} has invalid component counts")
        if not isinstance(round_times, list) or len(round_times) != rounds:
            raise ValueError(
                f"{backend} Boruvka sample {fit_index} round times do not align with rounds"
            )
        for index, value in enumerate(round_times):
            _positive_finite(value, f"{backend} Boruvka round {index}")
        round_stages = sample["round_stage_seconds"]
        if not isinstance(round_stages, dict) or set(round_stages) != set(BORUVKA_STAGES):
            raise ValueError(
                f"{backend} Boruvka sample {fit_index} has invalid round subphase names"
            )
        for stage in BORUVKA_STAGES:
            values = round_stages[stage]
            if not isinstance(values, list) or len(values) != rounds:
                raise ValueError(f"{backend} Boruvka {stage} does not align with rounds")
            for index, value in enumerate(values):
                _positive_finite(value, f"{backend} Boruvka {stage}[{index}]")
        for field in ("total_seconds", "seed_and_round_overhead_seconds"):
            _positive_finite(sample[field], f"{backend} Boruvka {field}")
        for field in ("seed_init_seconds", "finalize_seconds"):
            values = sample[field]
            if not isinstance(values, list):
                raise ValueError(f"{backend} Boruvka {field} must be a duration list")
            for index, value in enumerate(values):
                _positive_finite(value, f"{backend} Boruvka {field}[{index}]")

    for field in ("cluster_counts", "noise_counts", "resolved_search_per_fit"):
        values = profile[field]
        if not isinstance(values, list) or len(values) != runs:
            raise ValueError(f"{backend} profile {field} must contain {runs} measured values")
    for field in ("cluster_counts", "noise_counts"):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in profile[field]
        ):
            raise ValueError(f"{backend} profile {field} contains an invalid count")
    if any(not isinstance(value, str) or not value for value in profile["resolved_search_per_fit"]):
        raise ValueError(f"{backend} resolved search values must be nonempty strings")
    backend_name = profile["backend"]
    if backend == "cpu" and backend_name != "Cpu":
        raise ValueError(f"CPU command returned unexpected backend identity {backend_name!r}")
    if backend == "cuda" and (
        not isinstance(backend_name, str) or not backend_name.startswith("Cuda")
    ):
        raise ValueError(f"CUDA command returned unexpected backend identity {backend_name!r}")
    if profile["search"] != "Auto":
        raise ValueError(f"{backend} profile did not use requested automatic search")

    profile["medians"] = {
        "fit_seconds": statistics.median(fits),
        "stage_seconds": {
            stage: statistics.median(profile["stage_seconds"][stage]) for stage in STAGES
        },
    }
    return profile


def _load_profile(path: Path, backend: str) -> dict[str, Any]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read {backend} stdout {path}: {error}") from error
    candidates = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "fit_seconds" in value and "stage_seconds" in value:
            candidates.append(value)
    if len(candidates) != 1:
        raise ValueError(f"{backend} child must emit exactly one native profile JSON object")
    return _validate_profile(candidates[0], backend)


def _repository_provenance() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty}


def _tool_version(executable: str, flag: str) -> dict[str, Any]:
    path = shutil.which(executable)
    if path is None:
        return {"available": False, "version": None}
    try:
        result = subprocess.run(
            [path, flag], capture_output=True, text=True, check=False, timeout=5
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "version": None, "error": str(error)}
    text = (result.stdout or result.stderr).strip()
    return {
        "available": result.returncode == 0 and bool(text),
        "version": text if result.returncode == 0 and text else None,
    }


def _generate_input(path: Path, rows: int) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from comparison.compare_hdbscan_stages import make_input

    make_input(rows, FEATURES, 1.0, 1.0, 42, path)
    expected_size = rows * FEATURES * 4
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"generated input has {path.stat().st_size} bytes; expected {expected_size}"
        )


def benchmark(
    binary: Path, rows: int, run_dir: Path, timeout: float, min_samples: int = 5
) -> dict[str, Any]:
    if rows not in (20_000, 50_000, 100_000):
        raise ValueError("rows must be 20000, 50000, or 100000")
    if isinstance(min_samples, bool) or not isinstance(min_samples, int) or min_samples <= 0:
        raise ValueError("min_samples must be a positive integer")
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"profiling binary is not an executable file: {binary}")
    run_dir = run_dir.resolve()
    raw = run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    manifest_path = raw / "manifest.json"
    report_path = run_dir / "benchmark.json"
    if manifest_path.exists() or report_path.exists():
        raise FileExistsError(
            f"run directory already contains benchmark identity/report: {run_dir}"
        )

    input_path = raw / "input.f32le"
    if input_path.exists():
        raise FileExistsError(f"run directory already contains generated input: {input_path}")
    _generate_input(input_path, rows)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    workload = {
        "rows": rows,
        "features": FEATURES,
        "config": {
            "min_samples": min_samples,
            "min_cluster_size": 5,
            "search": "auto",
            "centers": 8,
            "center_scale": 1.0,
            "noise": 1.0,
            "seed": 42,
        },
    }
    provenance = {
        "repository": _repository_provenance(),
        "generator": {
            "path": "comparison/compare_hdbscan_stages.py::make_input",
            "numpy_version": np.__version__,
            "description": (
                "8 Gaussian centers, shuffled round-robin labels, normal noise; "
                "contiguous little-endian float32"
            ),
        },
        "python_version": platform.python_version(),
        "device_identity": {
            "available": False,
            "value": None,
            "note": "native profile reports backend but does not expose CPU or GPU model identity",
        },
        "tools": {
            "cargo": _tool_version("cargo", "--version"),
            "rustc": _tool_version("rustc", "--version"),
            "uv": _tool_version("uv", "--version"),
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "workload": workload,
        "input_sha256": digest,
        "provenance": provenance,
    }
    atomic_json(manifest_path, manifest)

    profiles: dict[str, Any] = {}
    for backend in ("cpu", "cuda"):
        labels_path = raw / f"{backend}_labels.json"
        trace_path = raw / f"{backend}_trace.json"
        stdout_path = raw / f"{backend}.stdout"
        stderr_path = raw / f"{backend}.stderr"
        argv = [
            str(binary),
            str(input_path),
            str(rows),
            str(FEATURES),
            "--backend",
            backend,
            "--runs",
            str(MEASURED_RUNS),
            "--min-cluster-size",
            "5",
            "--min-samples",
            str(min_samples),
            "--search",
            "auto",
            "--labels-output",
            str(labels_path),
            "--trace-output",
            str(trace_path),
        ]
        started = time.monotonic()
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            command = run_command(argv, timeout=timeout, stdout=stdout, stderr=stderr)
        command["wall_seconds"] = time.monotonic() - started
        if command["timed_out"]:
            raise TimeoutError(
                f"{backend} benchmark timed out after {timeout:g}s; "
                f"logs: {stdout_path}, {stderr_path}"
            )
        if command["exit_status"] != 0:
            raise RuntimeError(
                f"{backend} benchmark exited {command['exit_status']}; "
                f"see {stdout_path} and {stderr_path}"
            )
        profile = _load_profile(stdout_path, backend)
        if not labels_path.is_file() or not trace_path.is_file():
            raise ValueError(f"{backend} benchmark did not produce labels and trace outputs")
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {backend} trace output {trace_path}: {error}") from error
        if not isinstance(trace, list):
            raise ValueError(f"{backend} trace output must be a JSON event array")
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid {backend} label output {labels_path}: {error}") from error
        if (
            not isinstance(labels, list)
            or len(labels) != rows
            or any(isinstance(label, bool) or not isinstance(label, int) for label in labels)
        ):
            raise ValueError(f"{backend} labels must contain exactly {rows} integer labels")
        profile["process_wall_seconds"] = command["wall_seconds"]
        profile["command_provenance"] = command
        profiles[backend] = profile
        if backend == "cpu":
            cpu_labels = labels
        else:
            if labels != cpu_labels:
                mismatch = next(
                    index
                    for index, (cpu_label, cuda_label) in enumerate(
                        zip(cpu_labels, labels, strict=True)
                    )
                    if cpu_label != cuda_label
                )
                raise ValueError(
                    f"CPU/CUDA labels differ at row {mismatch}: "
                    f"{cpu_labels[mismatch]} != {labels[mismatch]}"
                )

    report = {
        "schema_version": manifest["schema_version"],
        "run_id": manifest["run_id"],
        "workload": workload,
        "input_sha256": digest,
        "provenance": provenance,
        "profiles": profiles,
        "correctness": {"labels_equal": True, "rows_compared": rows},
    }
    atomic_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int, choices=(20_000, 50_000, 100_000))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("--min-samples", type=int, default=5)
    args = parser.parse_args()
    try:
        benchmark(args.binary.resolve(), args.rows, args.run_dir, args.timeout, args.min_samples)
    except (OSError, RuntimeError, TimeoutError, ValueError) as error:
        print(f"benchmark_hdbscan_stages: {error}", file=sys.stderr)
        return 1
    print(f"wrote {args.run_dir.resolve() / 'benchmark.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
