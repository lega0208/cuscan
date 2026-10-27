#!/usr/bin/env python3
"""Validate and summarize the three offline HDBSCAN stage reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from hdbscan_stages_common import observed_kernel_families, uniform_route

SCHEMA_VERSION = 1
EXPECTED_CONFIG = {
    "features": 16,
    "config": {
        "min_samples": 5,
        "min_cluster_size": 5,
        "search": "auto",
        "centers": 8,
        "center_scale": 1.0,
        "noise": 1.0,
        "seed": 42,
    },
}
CORE_SUBPHASES = (
    "core_neighbor_kernel",
    "core_extraction",
    "core_neighbor_readback",
    "core_final_assembly_readback",
)
ROUND_SUBPHASES = (
    "boruvka_metadata_prepare_upload",
    "boruvka_candidate_kernel",
    "boruvka_component_reduction",
    "boruvka_winner_readback",
    "boruvka_host_merge",
)
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
COUNTER_METRICS = {
    "occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex": "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "lg_throttle": "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "long_scoreboard": "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
}
ALL_METRICS = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    *COUNTER_METRICS.values(),
)


class ReportError(ValueError):
    """Raised when a report is missing or violates the canonical contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def _object(value: Any, where: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{where} must be a JSON object")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{where} must be a number",
    )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ReportError(f"{where} must be a finite number") from exc
    _require(math.isfinite(result), f"{where} must be finite")
    _require(
        result > 0 if positive else result >= 0,
        f"{where} must be {'positive' if positive else 'nonnegative'}",
    )
    return result


def _load(path: str | Path, label: str) -> dict[str, Any]:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {label} report {path}: {exc}") from exc
    return _object(value, label)


def _validate_identity(reports: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    baseline: dict[str, Any] | None = None
    for label, report in reports:
        for key in ("schema_version", "run_id", "workload", "input_sha256", "provenance"):
            _require(key in report, f"{label} report is missing {key}")
        _require(
            isinstance(report["schema_version"], int)
            and not isinstance(report["schema_version"], bool)
            and report["schema_version"] == SCHEMA_VERSION,
            f"{label} schema_version must be {SCHEMA_VERSION}",
        )
        _require(
            isinstance(report["run_id"], str) and report["run_id"],
            f"{label} run_id must be a nonempty string",
        )
        workload = _object(report["workload"], f"{label}.workload")
        _require(
            isinstance(report["input_sha256"], str)
            and len(report["input_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in report["input_sha256"].lower()),
            f"{label}.input_sha256 must be a SHA-256 hex digest",
        )
        _object(report["provenance"], f"{label}.provenance")
        identity = {
            key: report[key]
            for key in ("schema_version", "run_id", "workload", "input_sha256", "provenance")
        }
        if baseline is None:
            baseline = identity
        else:
            for key, expected in baseline.items():
                _require(identity[key] == expected, f"{label} identity mismatch for {key}")
        _require(
            workload.get("features") == EXPECTED_CONFIG["features"],
            f"{label}.workload.features must be 16",
        )
        config = _object(workload.get("config"), f"{label}.workload.config")
        for key, expected in EXPECTED_CONFIG["config"].items():
            _require(
                config.get(key) == expected, f"{label}.workload.config.{key} must be {expected!r}"
            )
        rows = workload.get("rows")
        _require(
            isinstance(rows, int) and not isinstance(rows, bool) and rows in (20000, 50000, 100000),
            f"{label}.workload.rows must be 20000, 50000, or 100000",
        )
    assert baseline is not None
    return reports[0][1]


def _aligned_samples(values: Any, expected_len: int, where: str) -> list[float]:
    _require(
        isinstance(values, list) and len(values) == expected_len,
        f"{where} must contain {expected_len} aligned samples",
    )
    return [_number(value, f"{where}[{index}]") for index, value in enumerate(values)]


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _percent(numerator: float, denominator: float) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def _validate_profile(name: str, profile: Any, sample_count: int = 3) -> dict[str, Any]:
    profile = _object(profile, f"profiles.{name}")
    fits = _aligned_samples(
        profile.get("fit_seconds"), sample_count, f"profiles.{name}.fit_seconds"
    )
    if sample_count == 3:
        _number(profile.get("warmup_seconds"), f"profiles.{name}.warmup_seconds")
        _number(profile.get("process_wall_seconds"), f"profiles.{name}.process_wall_seconds")
        command = _object(profile.get("command_provenance"), f"profiles.{name}.command_provenance")
        _require(
            command.get("exit_status") == 0 and command.get("timed_out") is False,
            f"profiles.{name} benchmark command did not complete successfully",
        )
        _number(
            command.get("timeout_seconds"),
            f"profiles.{name}.command_provenance.timeout_seconds",
            positive=True,
        )
        _number(
            command.get("elapsed_seconds"), f"profiles.{name}.command_provenance.elapsed_seconds"
        )
        _require(
            isinstance(command.get("argv"), list)
            and command["argv"]
            and all(isinstance(arg, str) for arg in command["argv"]),
            f"profiles.{name}.command_provenance.argv must be nonempty argv",
        )
    if "warmup_seconds" in profile:
        _number(profile["warmup_seconds"], f"profiles.{name}.warmup_seconds")
    backend = profile.get("backend")
    expected_backend = "Cpu" if name == "cpu" else "Cuda"
    _require(
        isinstance(backend, str)
        and (
            backend == "Cpu"
            if expected_backend == "Cpu"
            else re.fullmatch(r"Cuda \{ device_index: \d+ \}", backend) is not None
        ),
        f"profiles.{name}.backend must identify {expected_backend}",
    )
    _require(profile.get("search") == "Auto", f"profiles.{name}.search must be Auto")
    stages = _object(profile.get("stage_seconds"), f"profiles.{name}.stage_seconds")
    _require(
        set(stages) == set(STAGES),
        f"profiles.{name}.stage_seconds must contain the canonical stages",
    )
    stage_values = {
        key: _aligned_samples(value, sample_count, f"profiles.{name}.stage_seconds.{key}")
        for key, value in stages.items()
    }
    core = profile.get("core_distance_samples")
    boruvka = profile.get("boruvka_samples")
    _require(
        isinstance(core, list) and len(core) == sample_count,
        f"profiles.{name}.core_distance_samples must contain {sample_count} fits",
    )
    _require(
        isinstance(boruvka, list) and len(boruvka) == sample_count,
        f"profiles.{name}.boruvka_samples must contain {sample_count} fits",
    )
    for fit_index, record in enumerate(core):
        record = _object(record, f"profiles.{name}.core_distance_samples[{fit_index}]")
        _require(
            isinstance(record.get("scope"), str) and record["scope"],
            f"profiles.{name} core sample {fit_index} needs a scope note",
        )
        subphases = _object(
            record.get("stage_seconds"),
            f"profiles.{name}.core_distance_samples[{fit_index}].stage_seconds",
        )
        _require(
            set(subphases) == set(CORE_SUBPHASES),
            f"profiles.{name} core subphase names are not canonical",
        )
        for key in CORE_SUBPHASES:
            _require(key in subphases, f"profiles.{name} core sample missing {key}")
            _require(isinstance(subphases[key], list), f"profiles.{name} core {key} must be a list")
            for index, duration in enumerate(subphases[key]):
                _number(duration, f"profiles.{name}.core[{fit_index}].{key}[{index}]")
    for fit_index, record in enumerate(boruvka):
        record = _object(record, f"profiles.{name}.boruvka_samples[{fit_index}]")
        _require(
            isinstance(record.get("scope"), str) and record["scope"],
            f"profiles.{name} Boruvka sample {fit_index} needs a scope note",
        )
        rounds = record.get("round_count")
        _require(
            isinstance(rounds, int) and not isinstance(rounds, bool) and rounds >= 0,
            f"profiles.{name}.boruvka[{fit_index}].round_count must be nonnegative",
        )
        round_seconds = _aligned_samples(
            record.get("round_seconds"),
            rounds,
            f"profiles.{name}.boruvka[{fit_index}].round_seconds",
        )
        counts = record.get("component_counts_per_round")
        _require(
            isinstance(counts, list) and len(counts) == rounds,
            f"profiles.{name}.boruvka[{fit_index}].component_counts_per_round "
            "must align with rounds",
        )
        _require(
            all(
                isinstance(count, int) and not isinstance(count, bool) and count >= 0
                for count in counts
            ),
            f"profiles.{name}.boruvka[{fit_index}] component counts must be nonnegative integers",
        )
        round_stages = _object(
            record.get("round_stage_seconds"),
            f"profiles.{name}.boruvka[{fit_index}].round_stage_seconds",
        )
        _require(
            set(round_stages) == set(ROUND_SUBPHASES),
            f"profiles.{name} Boruvka subphase names are not canonical",
        )
        for key in ROUND_SUBPHASES:
            _aligned_samples(
                round_stages.get(key), rounds, f"profiles.{name}.boruvka[{fit_index}].{key}"
            )
        _number(record.get("total_seconds"), f"profiles.{name}.boruvka[{fit_index}].total_seconds")
        _number(
            record.get("seed_and_round_overhead_seconds"),
            f"profiles.{name}.boruvka[{fit_index}].seed_and_round_overhead_seconds",
        )
        for key in ("seed_init_seconds", "finalize_seconds"):
            values = record.get(key)
            _require(
                isinstance(values, list),
                f"profiles.{name}.boruvka[{fit_index}].{key} must be a list",
            )
            for index, value in enumerate(values):
                _number(value, f"profiles.{name}.boruvka[{fit_index}].{key}[{index}]")
        _require(len(round_seconds) == len(counts), f"profiles.{name} round metadata mismatch")
    for field in ("cluster_counts", "noise_counts", "resolved_search_per_fit"):
        values = profile.get(field)
        _require(
            isinstance(values, list) and len(values) == sample_count,
            f"profiles.{name}.{field} must align with measured fits",
        )
    for field in ("cluster_counts", "noise_counts"):
        _require(
            all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in profile[field]
            ),
            f"profiles.{name}.{field} must contain nonnegative integers",
        )
    _require(
        all(isinstance(route, str) and route for route in profile["resolved_search_per_fit"]),
        f"profiles.{name}.resolved_search_per_fit must contain route names",
    )
    if sample_count == 3:
        medians = _object(profile.get("medians"), f"profiles.{name}.medians")
        _require(
            math.isclose(
                _number(medians.get("fit_seconds"), f"profiles.{name}.medians.fit_seconds"),
                _median(fits),
                rel_tol=1e-9,
            ),
            f"profiles.{name}.fit_seconds median is inconsistent",
        )
        stage_medians = _object(
            medians.get("stage_seconds"), f"profiles.{name}.medians.stage_seconds"
        )
        _require(
            set(stage_medians) == set(STAGES),
            f"profiles.{name}.medians.stage_seconds is incomplete",
        )
        for key, values in stage_values.items():
            _require(
                math.isclose(
                    _number(stage_medians[key], f"profiles.{name}.medians.stage_seconds.{key}"),
                    _median(values),
                    rel_tol=1e-9,
                ),
                f"profiles.{name}.stage_seconds.{key} median is inconsistent",
            )
    return {"fit": fits, "stages": stage_values, "core": core, "boruvka": boruvka, "raw": profile}


def _metric_value(record: dict[str, Any], metric: str, where: str) -> tuple[float, str]:
    metrics = _object(record.get("metrics"), f"{where}.metrics")
    _require(metric in metrics, f"{where} missing required counter metric {metric}")
    entry = _object(metrics[metric], f"{where}.{metric}")
    _require(
        entry.get("requested") is True and entry.get("observed") is True,
        f"{where}.{metric} was not both requested and observed",
    )
    raw_value = _number(entry.get("value"), f"{where}.{metric}.value")
    raw_unit = entry.get("unit")
    normalized_value = _number(entry.get("normalized_value"), f"{where}.{metric}.normalized_value")
    normalized_unit = entry.get("normalized_unit")
    _require(
        isinstance(raw_unit, str) and isinstance(normalized_unit, str),
        f"{where}.{metric} units must be strings",
    )

    if metric == "gpu__time_duration.sum":
        time_factors = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}
        _require(raw_unit in time_factors, f"{where}.{metric} has unsupported unit {raw_unit!r}")
        expected_value, expected_unit = raw_value * time_factors[raw_unit], "s"
    elif metric in ("launch__grid_size", "launch__block_size"):
        _require(raw_unit in ("", "#"), f"{where}.{metric} has unsupported unit {raw_unit!r}")
        _require(
            raw_value.is_integer() and raw_value > 0,
            f"{where}.{metric} must be a positive integer count",
        )
        expected_value, expected_unit = raw_value, "count"
    else:
        _require(
            raw_unit in ("%", "percent"),
            f"{where}.{metric} has unsupported unit {raw_unit!r}",
        )
        expected_value, expected_unit = raw_value, "%"
    _require(
        normalized_unit == expected_unit,
        f"{where}.{metric} normalized unit must be {expected_unit!r}",
    )
    _require(
        math.isclose(normalized_value, expected_value, rel_tol=1e-9, abs_tol=1e-15),
        f"{where}.{metric} normalized value is inconsistent with its raw value and unit",
    )
    if metric in ("launch__grid_size", "launch__block_size"):
        _require(
            normalized_value.is_integer() and normalized_value > 0,
            f"{where}.{metric} normalized count must be a positive integer",
        )
    return normalized_value, normalized_unit


def _validate_kernels(
    report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], float]:
    profiled = _object(report.get("profiled_fit"), "cuda_kernels.profiled_fit")
    window = _object(profiled.get("window"), "cuda_kernels.profiled_fit.window")
    for key in ("start_unix_ns", "end_unix_ns", "start_ns", "end_ns"):
        _require(
            isinstance(window.get(key), int) and not isinstance(window[key], bool),
            f"profiled_fit.window.{key} must be an integer",
        )
    _require(
        window["end_unix_ns"] > window["start_unix_ns"] and window["end_ns"] > window["start_ns"],
        "profiled_fit window must be ordered",
    )
    _require(
        window["end_unix_ns"] - window["start_unix_ns"] == window["end_ns"] - window["start_ns"],
        "profiled_fit window clock alignments disagree",
    )
    native = _validate_profile(
        "cuda_kernels.profiled_fit.native_profile", profiled.get("native_profile"), 1
    )
    kernels = profiled.get("kernels")
    _require(isinstance(kernels, list) and kernels, "cuda_kernels.kernels must be a nonempty array")
    clean = []
    for index, item in enumerate(kernels):
        item = _object(item, f"cuda_kernels.kernels[{index}]")
        name = item.get("name")
        _require(
            isinstance(name, str) and name, f"cuda_kernels.kernels[{index}] needs a kernel name"
        )
        _require(
            isinstance(item.get("family"), str) and item["family"],
            f"cuda_kernels.kernels[{index}] needs a family",
        )
        launches = item.get("launch_count")
        _require(
            isinstance(launches, int) and not isinstance(launches, bool) and launches > 0,
            f"cuda_kernels.kernels[{index}].launch_count must be positive",
        )
        duration = _number(
            item.get("device_seconds"), f"cuda_kernels.kernels[{index}].device_seconds"
        )
        _number(item.get("mean_seconds"), f"cuda_kernels.kernels[{index}].mean_seconds")
        _require(
            math.isclose(duration / launches, item["mean_seconds"], rel_tol=1e-9),
            f"cuda_kernels.kernels[{index}] mean duration is inconsistent",
        )
        geometries = item.get("geometry")
        launch_records = item.get("launches")
        _require(
            isinstance(geometries, list) and geometries,
            f"cuda_kernels.kernels[{index}].geometry must be nonempty",
        )
        _require(
            isinstance(launch_records, list) and len(launch_records) == launches,
            f"cuda_kernels.kernels[{index}] launch records must match launch_count",
        )
        _require(
            sum(_object(g, "kernel geometry").get("count", 0) for g in geometries) == launches,
            f"cuda_kernels.kernels[{index}] geometry counts do not match launches",
        )
        for launch in launch_records:
            launch = _object(launch, "kernel launch")
            start = _number(launch.get("start_ns"), "kernel launch start_ns")
            end = _number(launch.get("end_ns"), "kernel launch end_ns")
            duration_ns = _number(launch.get("duration_ns"), "kernel launch duration_ns")
            _require(
                end > start and math.isclose(end - start, duration_ns),
                "kernel launch interval is inconsistent",
            )
            _require(
                start >= window["start_ns"] and end <= window["end_ns"],
                "reported kernel launch is outside measured-fit window",
            )
        geometry_counts = 0
        for geometry in geometries:
            geometry = _object(geometry, "kernel geometry")
            for key in ("block", "grid"):
                values = geometry.get(key)
                _require(
                    isinstance(values, list)
                    and len(values) == 3
                    and all(isinstance(value, int) and value > 0 for value in values),
                    f"kernel geometry {key} must contain three positive dimensions",
                )
            count = geometry.get("count")
            _require(
                isinstance(count, int) and not isinstance(count, bool) and count > 0,
                "kernel geometry count must be positive",
            )
            geometry_counts += count
        _require(
            geometry_counts == launches,
            f"cuda_kernels.kernels[{index}] geometry counts do not match launches",
        )
        launch_sum = (
            sum(_number(item["duration_ns"], "kernel launch duration") for item in launch_records)
            / 1e9
        )
        _require(
            math.isclose(duration, launch_sum, rel_tol=1e-9),
            f"cuda_kernels.kernels[{index}] device time does not match launch durations",
        )
        clean.append(
            {
                "name": name,
                "family": item["family"],
                "launches": launches,
                "seconds": duration,
            }
        )
    total = _number(profiled.get("total_device_seconds"), "profiled_fit.total_device_seconds")
    _require(
        math.isclose(total, sum(row["seconds"] for row in clean), rel_tol=1e-9),
        "profiled_fit total device time does not match kernel families",
    )
    _require(
        isinstance(profiled.get("boundary_straddling_count"), int)
        and profiled["boundary_straddling_count"] >= 0,
        "profiled_fit.boundary_straddling_count must be nonnegative",
    )
    _require(
        isinstance(profiled.get("overlapping_activity"), bool),
        "profiled_fit.overlapping_activity must be boolean",
    )
    capture = _object(report.get("capture"), "cuda_kernels.capture")
    for key in ("exit_status", "timeout_seconds", "timed_out", "elapsed_seconds"):
        _require(key in capture, f"cuda_kernels.capture missing {key}")
    _require(
        capture["exit_status"] == 0 and capture["timed_out"] is False,
        "Nsight Systems capture did not complete successfully",
    )
    wall = _number(capture["elapsed_seconds"], "cuda_kernels.capture.elapsed_seconds")
    _require(wall > 0, "Nsight Systems capture elapsed time must be positive")
    export = _object(report.get("export"), "cuda_kernels.export")
    _require(
        export.get("exit_status") == 0 and export.get("timed_out") is False,
        "Nsight SQLite export did not complete successfully",
    )
    _number(export.get("elapsed_seconds"), "cuda_kernels.export.elapsed_seconds")
    _require(
        isinstance(export.get("argv"), list) and export["argv"],
        "cuda_kernels.export.argv must be nonempty",
    )
    return sorted(clean, key=lambda item: item["seconds"], reverse=True), native, profiled, wall


def _validate_counters(
    report: dict[str, Any], route: str, families: tuple[str, str]
) -> dict[str, dict[str, float]]:
    _require(
        report.get("route") == route,
        f"cuda_counters.route must match the benchmark CUDA route {route}",
    )
    by_family = _object(report.get("kernels"), "cuda_counters.kernels")
    captures = _object(report.get("captures"), "cuda_counters.captures")
    output: dict[str, dict[str, float]] = {}
    for family in families:
        _require(family in by_family, f"cuda_counters missing {family} record")
        record = _object(by_family[family], f"cuda_counters.{family}")
        capture = _object(captures.get(family), f"cuda_counters.captures.{family}")
        _require(
            capture.get("exit_status") == 0 and capture.get("timed_out") is False,
            f"cuda_counters.{family} capture did not complete successfully",
        )
        _number(
            capture.get("timeout_seconds"),
            f"cuda_counters.captures.{family}.timeout_seconds",
            positive=True,
        )
        _number(capture.get("elapsed_seconds"), f"cuda_counters.captures.{family}.elapsed_seconds")
        _require(
            isinstance(capture.get("argv"), list) and capture["argv"],
            f"cuda_counters.captures.{family}.argv must be nonempty",
        )
        for key in ("kernel_name", "launch_id", "geometry", "command", "sampling_scope"):
            _require(key in record, f"cuda_counters.{family} missing {key}")
        _require(
            isinstance(record["kernel_name"], str) and record["kernel_name"],
            f"cuda_counters.{family}.kernel_name must be nonempty",
        )
        geometry = _object(record["geometry"], f"cuda_counters.{family}.geometry")
        for dimension in ("block", "grid"):
            _require(
                isinstance(geometry.get(dimension), str) and geometry[dimension],
                f"cuda_counters.{family}.geometry.{dimension} must be present",
            )
        _require(
            isinstance(record["command"], list)
            and all(isinstance(arg, str) for arg in record["command"]),
            f"cuda_counters.{family}.command must be argv",
        )
        _require(
            isinstance(record["sampling_scope"], str) and record["sampling_scope"],
            f"cuda_counters.{family}.sampling_scope must be nonempty",
        )
        _require(
            family in record["kernel_name"],
            f"cuda_counters.{family} selected a kernel from another family",
        )
        values: dict[str, float] = {}
        for metric in ALL_METRICS:
            value, unit = _metric_value(record, metric, f"cuda_counters.{family}")
            if metric in COUNTER_METRICS.values():
                short = next(k for k, v in COUNTER_METRICS.items() if v == metric)
                values[short] = value
                _require(
                    unit == "%" and value <= 100.0,
                    f"cuda_counters.{family}.{metric} must normalize to percent in [0,100]",
                )
            elif metric == "gpu__time_duration.sum":
                _require(unit == "s", f"cuda_counters.{family}.{metric} must normalize to seconds")
            else:
                _require(
                    unit == "count", f"cuda_counters.{family}.{metric} must normalize to count"
                )
        output[family] = values
    return output


def _fmt_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1e6:.1f} µs"
    if value < 1:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.3f} s"


def _share(values: list[float], totals: list[float]) -> float:
    _require(
        len(values) == len(totals) and len(values) > 0, "cannot calculate unaligned phase shares"
    )
    return statistics.mean(
        _percent(value, total) for value, total in zip(values, totals, strict=True)
    )


def summarize_reports(
    benchmark: dict[str, Any], kernels: dict[str, Any], counters: dict[str, Any]
) -> str:
    """Validate report objects and return a concise, evidence-bounded summary."""
    identity = _validate_identity(
        [("benchmark", benchmark), ("cuda_kernels", kernels), ("cuda_counters", counters)]
    )
    profiles = _object(benchmark.get("profiles"), "benchmark.profiles")
    cpu = _validate_profile("cpu", profiles.get("cpu"))
    cuda = _validate_profile("cuda", profiles.get("cuda"))
    correctness = _object(benchmark.get("correctness"), "benchmark.correctness")
    _require(
        isinstance(correctness.get("labels_equal"), bool),
        "benchmark.correctness.labels_equal must be boolean",
    )
    _require(
        correctness["labels_equal"],
        "CPU/CUDA final labels differ; refusing to summarize an incorrect run",
    )
    workload = identity["workload"]
    _require(
        correctness.get("rows_compared") == workload["rows"],
        "benchmark.correctness.rows_compared must equal workload.rows",
    )
    kernel_rows, profiled_native, profiled_fit, profiled_wall = _validate_kernels(kernels)
    rows = workload["rows"]
    search = profiles["cuda"].get("resolved_search_per_fit")
    _require(
        isinstance(search, list)
        and len(search) == 3
        and all(isinstance(item, str) and item for item in search),
        "profiles.cuda.resolved_search_per_fit must contain three route strings",
    )
    try:
        route = uniform_route(search, "profiles.cuda.resolved_search_per_fit")
    except ValueError as error:
        raise ReportError(str(error)) from error
    binary = counters.get("binary", {})
    if route == "KdTree":
        binary = _object(binary, "cuda_counters.binary")
    families, family_evidence = observed_kernel_families(
        route,
        kernels,
        benchmark,
        {key: binary.get(key) for key in ("path", "sha256")},
    )
    if route == "KdTree":
        _require(
            counters.get("family_evidence") == family_evidence,
            "cuda_counters family evidence does not match cuda_kernels",
        )
    counter_values = _validate_counters(counters, route, families)

    lines = [
        f"HDBSCAN stage analysis — {rows:,} rows x {workload['features']} features",
        f"Input SHA-256: {identity['input_sha256'][:12]}…; CUDA route: {', '.join(search)}",
        "Scope: three measured fits per backend after one excluded warmup; phase spans are native "
        "host wall-clock scopes.",
        "",
        "Backend   median fit     core distances   MST             warmup       process wall",
    ]
    for name, profile in (("CPU", cpu), ("CUDA", cuda)):
        warmup_text = _fmt_seconds(_number(profile["raw"]["warmup_seconds"], name + ".warmup"))
        process_text = _fmt_seconds(
            _number(profile["raw"]["process_wall_seconds"], name + ".process")
        )
        lines.append(
            f"{name:<8} {_fmt_seconds(_median(profile['fit'])):>12} "
            f"{_fmt_seconds(_median(profile['stages']['core_distances'])):>16} "
            f"{_fmt_seconds(_median(profile['stages']['mst'])):>15} "
            f"{warmup_text:>12} {process_text:>14}"
        )
    profiled_fit_seconds = profiled_native["fit"][0]
    profiled_mst_seconds = profiled_native["stages"]["mst"][0]
    device_total = _number(
        profiled_fit["total_device_seconds"], "profiled_fit.total_device_seconds"
    )
    lines.append(
        "Nsight Systems measured-fit scope: "
        f"fit {_fmt_seconds(profiled_fit_seconds)}, MST {_fmt_seconds(profiled_mst_seconds)}, "
        f"kernel device activity {_fmt_seconds(device_total)}, "
        f"capture process {_fmt_seconds(profiled_wall)}."
    )
    if profiled_fit["overlapping_activity"]:
        lines.append("  Kernel activity overlaps; summed device duration is not elapsed GPU time.")
    if profiled_fit["boundary_straddling_count"]:
        lines.append(
            f"  {profiled_fit['boundary_straddling_count']} CUDA activities straddled the "
            "measured-fit boundary and were excluded."
        )
    lines.append("Phase shares (mean of per-fit phase/fit ratios; not sums of medians):")
    for name, profile in (("CPU", cpu), ("CUDA", cuda)):
        lines.append(
            f"  {name}: core distances "
            f"{_share(profile['stages']['core_distances'], profile['fit']):.1f}%, "
            f"MST {_share(profile['stages']['mst'], profile['fit']):.1f}% of fit"
        )
        mst_totals = profile["stages"]["mst"]
        core_parts: dict[str, list[float]] = {key: [] for key in CORE_SUBPHASES}
        round_parts: dict[str, list[float]] = {key: [] for key in ROUND_SUBPHASES}
        component_paths: list[str] = []
        for i, record in enumerate(profile["core"]):
            for key in CORE_SUBPHASES:
                core_parts[key].append(
                    sum(_number(x, f"{name}.core[{i}].{key}") for x in record["stage_seconds"][key])
                )
        for i, record in enumerate(profile["boruvka"]):
            for key in ROUND_SUBPHASES:
                round_parts[key].append(
                    sum(
                        _number(x, f"{name}.boruvka[{i}].{key}")
                        for x in record["round_stage_seconds"][key]
                    )
                )
            counts = record["component_counts_per_round"]
            component_paths.append(
                "→".join(str(value) for value in counts) if counts else "no candidate rounds"
            )
        dominant_core = max(core_parts, key=lambda key: statistics.median(core_parts[key]))
        dominant_round = max(round_parts, key=lambda key: statistics.median(round_parts[key]))
        dominant_core_seconds = _fmt_seconds(_median(core_parts[dominant_core]))
        lines.append(
            f"  {name} dominant core subphase: {dominant_core} "
            f"({dominant_core_seconds} median across fit scopes)"
        )
        lines.append(f"  {name} Boruvka component trajectory by fit: {'; '.join(component_paths)}")
        candidate = round_parts["boruvka_candidate_kernel"]
        reduction = round_parts["boruvka_component_reduction"]
        host_merge = round_parts["boruvka_host_merge"]
        lines.append(
            f"  {name} Boruvka subphases per MST: candidate "
            f"{_share(candidate, mst_totals):.1f}%, reduction "
            f"{_share(reduction, mst_totals):.1f}%, host merge "
            f"{_share(host_merge, mst_totals):.1f}%; largest median subphase {dominant_round}"
        )

    lines.extend(
        [
            "",
            "CUDA kernel families (summed device activity; overlapping launches need not add "
            "to elapsed time):",
        ]
    )
    for row in kernel_rows[:8]:
        lines.append(
            f"  {row['family']}: {row['name']} — {_fmt_seconds(row['seconds'])} device, "
            f"{row['launches']} launches"
        )
    shown_names = {row["name"] for row in kernel_rows[:8]}
    for row in kernel_rows[8:]:
        if row["family"] == "unknown" and row["name"] not in shown_names:
            lines.append(
                f"  unknown: {row['name']} — {_fmt_seconds(row['seconds'])} device, "
                f"{row['launches']} launches"
            )
    lines.append(
        "Representative NCU counters (first matching warmup launch; not measured-fit averages):"
    )
    if family_evidence:
        lines.append(f"  Selected neighbor family from cuda_kernels.json: {families[0]}")
    for family, values in counter_values.items():
        lines.append(
            f"  {family}: occupancy {values['occupancy']:.1f}%, SM {values['sm']:.1f}%, "
            f"L1/TEX {values['l1tex']:.1f}%, DRAM {values['dram']:.1f}%, "
            f"LG throttle {values['lg_throttle']:.1f}%, "
            f"long scoreboard {values['long_scoreboard']:.1f}%"
        )

    conclusions: list[str] = []
    candidate_share = _share(round_parts["boruvka_candidate_kernel"], cuda["stages"]["mst"])
    if candidate_share > 90.0:
        conclusions.append(
            f"Candidate generation accounts for {candidate_share:.1f}% of aligned CUDA MST "
            "phase time; it dominates this run's measured MST scope."
        )
    else:
        winner = max(
            ROUND_SUBPHASES, key=lambda key: _share(round_parts[key], cuda["stages"]["mst"])
        )
        conclusions.append(
            f"The largest aligned CUDA Boruvka subphase is {winner} "
            f"({_share(round_parts[winner], cuda['stages']['mst']):.1f}% of MST), "
            "not candidate generation alone."
        )
    neighbor_family = families[0]
    partial = counter_values[neighbor_family]
    if partial["occupancy"] < 10 and partial["sm"] < 10:
        conclusions.append(
            f"{neighbor_family} shows low representative occupancy "
            f"({partial['occupancy']:.1f}%) and SM throughput ({partial['sm']:.1f}%); "
            "this is a launch-specific hypothesis, not a whole-run bottleneck proof."
        )
    local_pressure = [
        (family, values)
        for family, values in counter_values.items()
        if values["l1tex"] > 80 and values["lg_throttle"] > 50 and values["dram"] < 50
    ]
    if local_pressure:
        family, values = local_pressure[0]
        conclusions.append(
            f"{family} counters are consistent with possible local load/store pressure "
            f"(L1/TEX {values['l1tex']:.1f}%, LG throttle {values['lg_throttle']:.1f}%, "
            f"DRAM {values['dram']:.1f}%); counters do not establish causality."
        )
    if len(conclusions) < 2:
        top = max(kernel_rows, key=lambda row: row["seconds"])
        conclusions.append(
            f"Largest reported kernel family is {top['name']} at "
            f"{_fmt_seconds(top['seconds'])} summed device activity; "
            "overlap means this is not elapsed time."
        )
    lines.extend(["", "Conditional conclusions:", *(f"  - {line}" for line in conclusions[:4])])
    lines.append(
        "Correctness: CPU/CUDA labels agree exactly for this fixed synthetic input; this does not "
        "establish external-data clustering quality."
    )
    lines.append(
        "Limitations: synchronized phase timings are not device-only durations; kernel sums differ "
        "from wall time. Nsight synchronization, replay, and cache behavior perturb captures; "
        "GPU counters cannot diagnose CPU instruction bottlenecks."
    )
    return "\n".join(lines)


def summarize_paths(
    benchmark_path: str | Path, kernels_path: str | Path, counters_path: str | Path
) -> str:
    return summarize_reports(
        _load(benchmark_path, "benchmark"),
        _load(kernels_path, "cuda_kernels"),
        _load(counters_path, "cuda_counters"),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, help="benchmark.json report")
    parser.add_argument("--kernels", required=True, help="cuda_kernels.json report")
    parser.add_argument("--counters", required=True, help="cuda_counters.json report")
    args = parser.parse_args(argv)
    try:
        print(summarize_paths(args.benchmark, args.kernels, args.counters))
    except ReportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
