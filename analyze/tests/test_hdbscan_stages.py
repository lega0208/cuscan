from __future__ import annotations

import csv
import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
common_tools = importlib.import_module("analyze.hdbscan_stages_common")
summary_tools = importlib.import_module("analyze.summarize_hdbscan_stages")
kernel_tools = importlib.import_module("profile_hdbscan_kernels")
counter_tools = importlib.import_module("profile_hdbscan_counters")
benchmark_tools = importlib.import_module("benchmark_hdbscan_stages")
atomic_json = common_tools.atomic_json
identity = common_tools.identity
require_fields = common_tools.require_fields
run_command = common_tools.run_command
validate_min_samples = common_tools.validate_min_samples
CORE_SUBPHASES = summary_tools.CORE_SUBPHASES
COUNTER_METRICS = summary_tools.COUNTER_METRICS
ROUND_SUBPHASES = summary_tools.ROUND_SUBPHASES
summarize_paths = summary_tools.summarize_paths
summarize_reports = summary_tools.summarize_reports

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "analyze_hdbscan_stages.sh"
SUMMARY_SCRIPT = ROOT / "analyze" / "summarize_hdbscan_stages.py"
BASH = shutil.which("bash") or "/bin/bash"


def fake_tools(
    tmp_path: Path,
    *,
    missing: str | None = None,
    uv_status: int = 0,
    uv_failure: str | None = None,
) -> tuple[Path, Path]:
    bin_dir = tmp_path / "fake tools"
    bin_dir.mkdir()
    log = tmp_path / "commands.log"
    for name in ("cargo", "uv", "nsys", "ncu"):
        if name == missing:
            continue
        code = uv_status if name == "uv" else 0
        executable = bin_dir / name
        executable.write_text(
            f"#!{sys.executable}\n"
            "import pathlib, sys\n"
            f"pathlib.Path({str(log)!r}).open('a').write("
            f"{name!r} + ' ' + repr(sys.argv[1:]) + '\\n')\n"
            f"sys.exit({code} if {uv_failure!r} is None or {uv_failure!r} "
            "in ' '.join(sys.argv) else 0)\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
    for utility in ("dirname", "mkdir"):
        target = shutil.which(utility)
        if target is not None:
            (bin_dir / utility).symlink_to(target)
    return bin_dir, log


def shell_only_path(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "shell utilities"
    bin_dir.mkdir()
    for utility in ("dirname", "mkdir"):
        target = shutil.which(utility)
        if target is not None:
            (bin_dir / utility).symlink_to(target)
    return bin_dir


def invoke_shell(
    *args: str, cwd: Path, path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [BASH, str(SCRIPT), *args],
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_shell_help_is_successful_without_dependencies(tmp_path: Path) -> None:
    result = invoke_shell("--help", cwd=tmp_path, path=str(shell_only_path(tmp_path)))
    assert result.returncode == 0
    assert "--rows 20000|50000|100000" in result.stdout
    assert "--output-dir NEW_DIRECTORY" in result.stdout


@pytest.mark.parametrize(
    "rows, native_timeout, profile_timeout",
    [("20000", "10", "10"), ("50000", "240", "90"), ("100000", "240", "90")],
)
def test_shell_cli_accepts_supported_rows_and_resolves_paths_from_foreign_cwd(
    tmp_path: Path, rows: str, native_timeout: str, profile_timeout: str
) -> None:
    bin_dir, log = fake_tools(tmp_path)
    output = tmp_path / "evidence with spaces" / f"rows-{rows}"
    result = invoke_shell(
        "--rows", rows, "--output-dir", str(output), cwd=tmp_path, path=str(bin_dir)
    )
    assert result.returncode == 0, result.stderr
    assert output.is_dir()
    assert (output / "raw").is_dir()
    assert str(output) in result.stdout
    commands = log.read_text(encoding="utf-8")
    assert "cargo" in commands
    assert "benchmark_hdbscan_stages.py" in commands
    assert "--rows', '" + rows in commands
    assert "--timeout', '" + native_timeout in commands
    assert "profile_hdbscan_kernels.py" in commands
    assert "profile_hdbscan_counters.py" in commands
    assert "--min-samples', '5" in commands
    assert "--timeout', '" + profile_timeout in commands
    assert "summarize_hdbscan_stages.py" in commands


def test_shell_rejects_existing_output_directory_without_modifying_it(tmp_path: Path) -> None:
    bin_dir, _ = fake_tools(tmp_path)
    output = tmp_path / "already exists"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    result = invoke_shell("--output-dir", str(output), cwd=tmp_path, path=str(bin_dir))
    assert result.returncode != 0
    assert "already exists" in result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_shell_rejects_unknown_rows_and_flags_before_preflight(tmp_path: Path) -> None:
    shell_path = str(shell_only_path(tmp_path))
    result = invoke_shell("--rows", "30000", cwd=tmp_path, path=shell_path)
    assert result.returncode != 0
    assert "--rows must be 20000, 50000 or 100000" in result.stderr
    unknown = invoke_shell("--nonsense", cwd=tmp_path, path=shell_path)
    assert unknown.returncode != 0
    assert "Unknown argument" in unknown.stderr


def test_shell_reports_missing_dependency(tmp_path: Path) -> None:
    bin_dir, _ = fake_tools(tmp_path, missing="nsys")
    result = invoke_shell("--output-dir", str(tmp_path / "out"), cwd=tmp_path, path=str(bin_dir))
    assert result.returncode != 0
    assert "Required tool not found: nsys" in result.stderr
    assert not (tmp_path / "out").exists()


def test_shell_propagates_phase_failure_without_success_summary(tmp_path: Path) -> None:
    bin_dir, log = fake_tools(tmp_path, uv_status=17)
    output = tmp_path / "failed run"
    result = invoke_shell("--output-dir", str(output), cwd=tmp_path, path=str(bin_dir))
    assert result.returncode == 17
    assert "Phase failed" in result.stderr
    assert "No matching" not in result.stdout
    assert log.read_text(encoding="utf-8").count("uv ") == 1
    assert (output / "raw").is_dir()


@pytest.mark.parametrize(
    "failing_phase",
    ["profile_hdbscan_kernels.py", "profile_hdbscan_counters.py"],
)
def test_shell_stops_on_profiler_failure_without_running_later_phases(
    tmp_path: Path, failing_phase: str
) -> None:
    bin_dir, log = fake_tools(tmp_path, uv_failure=failing_phase, uv_status=19)
    output = tmp_path / "profiler failure"
    result = invoke_shell("--output-dir", str(output), cwd=tmp_path, path=str(bin_dir))
    assert result.returncode == 19
    assert "Phase failed" in result.stderr
    commands = log.read_text(encoding="utf-8")
    assert failing_phase in commands
    if "profile_hdbscan_kernels.py" in failing_phase:
        assert "profile_hdbscan_counters.py" not in commands
    assert "summarize_hdbscan_stages.py" not in commands


def manifest() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "synthetic-run",
        "workload": {"rows": 20_000, "features": 16},
        "input_sha256": "a" * 64,
        "provenance": {"source": "synthetic test"},
    }


def synthetic_reports(
    candidate_seconds: float = 0.95,
    route: str = "BruteForce",
    neighbor_family: str = "kd_neighbors_kernel",
) -> tuple[dict[str, Any], ...]:
    families = common_tools.route_kernel_families(route)
    if route == "KdTree":
        families = (neighbor_family, "kd_candidates_kernel")
    identity_fields = {
        "schema_version": 1,
        "run_id": "synthetic-run",
        "workload": {
            "rows": 20_000,
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
        },
        "input_sha256": "b" * 64,
        "provenance": {"source": "synthetic test"},
    }
    profiles = {}
    for backend in ("cpu", "cuda"):
        profiles[backend] = {
            "fit_seconds": [2.0, 2.1, 1.9],
            "warmup_seconds": 1.2,
            "process_wall_seconds": 8.0,
            "command_provenance": {
                "argv": ["synthetic", backend],
                "exit_status": 0,
                "timed_out": False,
                "timeout_seconds": 10,
                "elapsed_seconds": 8.0,
            },
            "medians": {
                "fit_seconds": 2.0,
                "stage_seconds": {
                    stage: 0.8 if stage == "core_distances" else 1.0 if stage == "mst" else 0.01
                    for stage in summary_tools.STAGES
                },
            },
            "stage_seconds": {
                stage: (
                    [0.8, 0.9, 0.7]
                    if stage == "core_distances"
                    else [1.0, 1.0, 1.0]
                    if stage == "mst"
                    else [0.01, 0.01, 0.01]
                )
                for stage in summary_tools.STAGES
            },
            "core_distance_samples": [
                {
                    "stage_seconds": {
                        key: [0.5 if key == "core_extraction" else 0.1] for key in CORE_SUBPHASES
                    },
                    "scope": "one measured fit",
                }
                for _ in range(3)
            ],
            "boruvka_samples": [
                {
                    "round_count": 1,
                    "round_seconds": [1.0],
                    "component_counts_per_round": [4],
                    "round_stage_seconds": {
                        key: [
                            candidate_seconds
                            if key == "boruvka_candidate_kernel"
                            else (
                                0.8
                                if candidate_seconds < 0.2 and key == "boruvka_host_merge"
                                else (0.2 - candidate_seconds) / (len(ROUND_SUBPHASES) - 2)
                                if candidate_seconds < 0.2
                                else (1.0 - candidate_seconds) / (len(ROUND_SUBPHASES) - 1)
                            )
                        ]
                        for key in ROUND_SUBPHASES
                    },
                    "total_seconds": 1.0,
                    "seed_and_round_overhead_seconds": 0.1,
                    "seed_init_seconds": [0.05],
                    "finalize_seconds": [0.05],
                    "scope": "one measured fit",
                }
                for _ in range(3)
            ],
            "cluster_counts": [4, 4, 4],
            "noise_counts": [2, 2, 2],
            "backend": "Cpu" if backend == "cpu" else "Cuda { device_index: 0 }",
            "search": "Auto",
            "resolved_search_per_fit": [route] * 3,
        }
    benchmark = {
        **identity_fields,
        "profiles": profiles,
        "correctness": {"labels_equal": True, "rows_compared": 20_000},
    }
    native = synthetic_native_profile()
    native["fit_seconds"] = native["fit_seconds"][:1]
    native["backend"] = "Cuda { device_index: 0 }"
    native["stage_seconds"] = {key: values[:1] for key, values in native["stage_seconds"].items()}
    native["core_distance_samples"] = native["core_distance_samples"][:1]
    native["boruvka_samples"] = native["boruvka_samples"][:1]
    for key in ("cluster_counts", "noise_counts", "resolved_search_per_fit"):
        native[key] = native[key][:1]
    kernels = {
        **identity_fields,
        "capture": {
            "exit_status": 0,
            "timeout_seconds": 10,
            "timed_out": False,
            "elapsed_seconds": 3.0,
        },
        "export": {
            "argv": ["nsys", "export"],
            "exit_status": 0,
            "timeout_seconds": 60,
            "timed_out": False,
            "elapsed_seconds": 1.0,
        },
        "profiled_fit": {
            "window": {
                "start_unix_ns": 1100,
                "end_unix_ns": 1200,
                "start_ns": 100,
                "end_ns": 200,
            },
            "native_profile": native,
            "kernels": [
                {
                    "name": "tile_candidates_variant_abc",
                    "family": "tile_candidates",
                    "launch_count": 2,
                    "device_seconds": 40e-9,
                    "mean_seconds": 20e-9,
                    "geometry": [{"block": [256, 1, 1], "grid": [32, 1, 1], "count": 2}],
                    "launches": [
                        {"start_ns": 110, "end_ns": 130, "duration_ns": 20},
                        {"start_ns": 120, "end_ns": 140, "duration_ns": 20},
                    ],
                },
                {
                    "name": "unknown_future_kernel",
                    "family": "unknown",
                    "launch_count": 1,
                    "device_seconds": 10e-9,
                    "mean_seconds": 10e-9,
                    "geometry": [{"block": [8, 1, 1], "grid": [1, 1, 1], "count": 1}],
                    "launches": [{"start_ns": 150, "end_ns": 160, "duration_ns": 10}],
                },
            ],
            "total_device_seconds": 50e-9,
            "boundary_straddling_count": 1,
            "overlapping_activity": True,
        },
    }
    all_metrics = summary_tools.ALL_METRICS
    counters = {
        **identity_fields,
        "route": route,
        "captures": {
            family: {
                "argv": ["ncu", "--kernel-name", f"regex:{family}"],
                "exit_status": 0,
                "timeout_seconds": 10,
                "timed_out": False,
                "elapsed_seconds": 2.0,
            }
            for family in families
        },
        "kernels": {
            family: {
                "kernel_name": f"{family}_synthetic_hash",
                "launch_id": "1",
                "geometry": {"block": "(32, 1, 1)", "grid": "(8, 1, 1)"},
                "command": ["ncu", "--kernel-name", family],
                "sampling_scope": "first matching warmup launch",
                "metrics": {
                    metric: {
                        "requested": True,
                        "observed": True,
                        "value": (
                            5.0
                            if metric in COUNTER_METRICS.values()
                            else 3_000_000_000.0
                            if metric == "gpu__time_duration.sum"
                            else 3.0
                        ),
                        "unit": (
                            "%"
                            if metric in COUNTER_METRICS.values()
                            else "ns"
                            if metric == "gpu__time_duration.sum"
                            else ""
                        ),
                        "normalized_value": 5.0 if metric in COUNTER_METRICS.values() else 3.0,
                        "normalized_unit": (
                            "%"
                            if metric in COUNTER_METRICS.values()
                            else "s"
                            if metric == "gpu__time_duration.sum"
                            else "count"
                        ),
                    }
                    for metric in all_metrics
                },
            }
            for family in families
        },
    }
    if route == "KdTree":
        binary = {"path": "/synthetic/profile", "sha256": "b" * 64}
        kernels["binary"] = binary
        counters["binary"] = {**binary, "label": "current"}
        native["resolved_search_per_fit"] = [route]
        item = kernels["profiled_fit"]["kernels"][0]
        item["family"] = neighbor_family
        item["name"] = f"{neighbor_family}_synthetic_hash"
        _, evidence = common_tools.observed_kernel_families(route, kernels, benchmark, binary)
        counters["family_evidence"] = evidence
    return benchmark, kernels, counters


@pytest.mark.parametrize("rows", [20_000, 50_000, 100_000])
def test_summary_accepts_supported_row_counts(rows: int) -> None:
    reports = synthetic_reports()
    for report in reports:
        report["workload"]["rows"] = rows
    reports[0]["correctness"]["rows_compared"] = rows
    result = summarize_reports(*reports)
    assert result.startswith(f"HDBSCAN stage analysis — {rows:,} rows")


def test_summary_selects_counter_families_from_kd_route() -> None:
    benchmark, kernels, counters = synthetic_reports(route="KdTree")
    assert set(counters["kernels"]) == {"kd_neighbors_kernel", "kd_candidates_kernel"}
    for family in counters["kernels"]:
        counters["kernels"][family]["metrics"][COUNTER_METRICS["occupancy"]]["normalized_value"] = 1
        counters["kernels"][family]["metrics"][COUNTER_METRICS["occupancy"]]["value"] = 1
    result = summarize_reports(benchmark, kernels, counters)
    assert "CUDA route: KdTree, KdTree, KdTree" in result


def test_summary_rejects_counter_route_mismatch_and_mixed_routes() -> None:
    benchmark, kernels, counters = synthetic_reports(route="KdTree")
    counters["route"] = "BruteForce"
    with pytest.raises(ValueError, match=r"cuda_counters\.route must match"):
        summarize_reports(benchmark, kernels, counters)
    benchmark, kernels, counters = synthetic_reports()
    benchmark["profiles"]["cuda"]["resolved_search_per_fit"] = ["BruteForce", "KdTree", "KdTree"]
    with pytest.raises(ValueError, match="one consistent resolved search route"):
        summarize_reports(benchmark, kernels, counters)


def test_route_kernel_families_cover_both_routes_and_reject_unknown() -> None:
    assert common_tools.route_kernel_families("BruteForce") == ("partial_kernel", "tile_candidates")
    assert common_tools.route_kernel_families("KdTree") == (
        "kd_neighbors_kernel",
        "kd_candidates_kernel",
    )
    with pytest.raises(ValueError, match="unsupported resolved search route"):
        common_tools.route_kernel_families("Auto")


@pytest.mark.parametrize("family", ["kd_neighbors_kernel", "kd_neighbors_tile_kernel"])
def test_counter_kernel_selection_uses_dispatched_kd_neighbor_kernel(family: str) -> None:
    benchmark, kernels, counters = synthetic_reports(route="KdTree", neighbor_family=family)
    families, evidence = common_tools.observed_kernel_families(
        "KdTree", kernels, benchmark, kernels["binary"]
    )
    assert families == (family, "kd_candidates_kernel")
    assert evidence["neighbor_family"] == family
    assert f"Selected neighbor family from cuda_kernels.json: {family}" in summarize_reports(
        benchmark, kernels, counters
    )


@pytest.mark.parametrize(
    "failure",
    ["missing", "ambiguous", "identity", "provenance", "binary", "route", "capture", "export"],
)
def test_observed_kd_families_reject_invalid_evidence(failure: str) -> None:
    benchmark, kernels, _ = synthetic_reports(route="KdTree")
    if failure == "missing":
        kernels["profiled_fit"]["kernels"] = []
    elif failure == "ambiguous":
        kernels["profiled_fit"]["kernels"].append(
            {
                "family": "kd_neighbors_tile_kernel",
                "name": "kd_neighbors_tile_kernel_hash",
                "launch_count": 1,
            }
        )
    elif failure == "identity":
        kernels["run_id"] = "wrong-run"
    elif failure == "provenance":
        kernels["provenance"] = {"source": "wrong-source"}
    elif failure == "binary":
        kernels["binary"] = {"path": "/wrong", "sha256": "c" * 64}
    elif failure in ("capture", "export"):
        kernels[failure]["exit_status"] = 1
    else:
        kernels["profiled_fit"]["native_profile"]["resolved_search_per_fit"] = ["BruteForce"]
    with pytest.raises(ValueError, match="cuda_kernels"):
        common_tools.observed_kernel_families(
            "KdTree",
            kernels,
            benchmark,
            {"path": "/synthetic/profile", "sha256": "b" * 64},
        )


def test_observed_kd_families_require_kernel_report() -> None:
    with pytest.raises(ValueError, match="cuda_kernels must be a JSON object"):
        common_tools.observed_kernel_families("KdTree", None, {}, {})


def test_summary_rejects_counter_family_evidence_mismatch() -> None:
    reports = synthetic_reports(route="KdTree")
    reports[2]["family_evidence"]["neighbor_family"] = "kd_neighbors_tile_kernel"
    with pytest.raises(ValueError, match="family evidence"):
        summarize_reports(*reports)


@pytest.mark.parametrize("mismatch_source", ["manifest", "benchmark"])
def test_counter_min_samples_rejects_configuration_mismatch(mismatch_source: str) -> None:
    manifest_data = {"workload": {"config": {"min_samples": 16}}}
    benchmark_data = {"workload": {"config": {"min_samples": 16}}}
    target = manifest_data if mismatch_source == "manifest" else benchmark_data
    target["workload"]["config"]["min_samples"] = 5
    with pytest.raises(ValueError, match=f"does not match {mismatch_source}"):
        validate_min_samples(manifest_data, benchmark_data, 16)


@pytest.mark.parametrize("configured", [True, "16", 16.0, 0, -1])
@pytest.mark.parametrize("source", ["manifest", "benchmark"])
def test_counter_min_samples_rejects_malformed_configuration(configured, source: str) -> None:
    manifest_data = {"workload": {"config": {"min_samples": 16}}}
    benchmark_data = {"workload": {"config": {"min_samples": 16}}}
    target = manifest_data if source == "manifest" else benchmark_data
    target["workload"]["config"]["min_samples"] = configured
    with pytest.raises(ValueError, match=f"{source} configuration has invalid min_samples"):
        validate_min_samples(manifest_data, benchmark_data, 16)


def test_summary_accepts_three_synthetic_reports_and_preserves_caveats() -> None:
    benchmark, kernels, counters = synthetic_reports()
    result = summarize_reports(benchmark, kernels, counters)
    assert result.startswith("HDBSCAN stage analysis — 20,000 rows")
    assert "three measured fits per backend after one excluded warmup" in result
    assert "Candidate generation accounts for" in result
    assert "unknown_future_kernel" in result
    assert "Nsight synchronization, replay, and cache behavior perturb captures" in result
    assert "does not establish external-data clustering quality" in result


@pytest.mark.parametrize(
    "report_index,mutate,diagnostic",
    [
        (1, lambda report: report.update(run_id="other"), "identity mismatch"),
        (2, lambda report: report.update(input_sha256="c" * 64), "identity mismatch"),
        (1, lambda report: report.update(schema_version=2), "schema_version"),
        (0, lambda report: report["workload"].update(rows=30_000), "workload.rows"),
    ],
)
def test_summary_rejects_incompatible_identity(report_index, mutate, diagnostic) -> None:
    reports = list(synthetic_reports())
    mutate(reports[report_index])
    with pytest.raises(ValueError, match=diagnostic):
        summarize_reports(*reports)


@pytest.mark.parametrize(
    "profile_name,backend",
    [("cpu", "Cuda { device_index: 0 }"), ("cuda", "Cpu")],
)
def test_summary_rejects_profile_backend_mismatch(profile_name: str, backend: str) -> None:
    reports = list(synthetic_reports())
    reports[0]["profiles"][profile_name]["backend"] = backend
    with pytest.raises(ValueError, match="backend must identify"):
        summarize_reports(*reports)


@pytest.mark.parametrize("corruption", [float("nan"), float("inf"), -0.1])
def test_summary_rejects_nonfinite_or_negative_durations(corruption: float) -> None:
    reports = list(synthetic_reports())
    reports[0]["profiles"]["cpu"]["fit_seconds"][1] = corruption
    with pytest.raises(ValueError):
        summarize_reports(*reports)


def test_summary_changes_conclusion_when_bottleneck_changes() -> None:
    candidate_reports = synthetic_reports(candidate_seconds=0.95)
    host_reports = synthetic_reports(candidate_seconds=0.1)
    candidate_summary = summarize_reports(*candidate_reports)
    host_summary = summarize_reports(*host_reports)
    assert "Candidate generation accounts for" in candidate_summary
    assert "largest aligned CUDA Boruvka subphase is boruvka_host_merge" in host_summary


def test_summary_rejects_label_mismatch_and_missing_counter_metric() -> None:
    reports = list(synthetic_reports())
    reports[0]["correctness"]["labels_equal"] = False
    with pytest.raises(ValueError, match="labels differ"):
        summarize_reports(*reports)
    reports = list(synthetic_reports())
    del reports[2]["kernels"]["partial_kernel"]["metrics"][COUNTER_METRICS["occupancy"]]
    with pytest.raises(ValueError, match="missing required counter metric"):
        summarize_reports(*reports)


@pytest.mark.parametrize(
    "metric,field,value,diagnostic",
    [
        (
            "gpu__time_duration.sum",
            "value",
            1.0,
            "normalized value is inconsistent",
        ),
        ("launch__grid_size", "value", 1.5, "positive integer count"),
    ],
)
def test_summary_rejects_inconsistent_raw_counter_units_and_fractional_geometry(
    metric: str, field: str, value: float, diagnostic: str
) -> None:
    reports = list(synthetic_reports())
    reports[2]["kernels"]["partial_kernel"]["metrics"][metric][field] = value
    if metric == "launch__grid_size":
        reports[2]["kernels"]["partial_kernel"]["metrics"][metric]["normalized_value"] = value
    with pytest.raises(ValueError, match=diagnostic):
        summarize_reports(*reports)


def test_summary_paths_and_cli_are_offline_and_report_invalid_inputs(tmp_path: Path) -> None:
    reports = synthetic_reports()
    paths = [tmp_path / f"report-{index}.json" for index in range(3)]
    for path, report in zip(paths, reports, strict=True):
        path.write_text(json.dumps(report), encoding="utf-8")
    assert "Input SHA-256" in summarize_paths(*paths)
    result = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--benchmark",
            str(paths[0]),
            "--kernels",
            str(paths[1]),
            "--counters",
            str(paths[2]),
        ],
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Input SHA-256" in result.stdout
    paths[2].write_text("not json", encoding="utf-8")
    bad = subprocess.run(
        [
            sys.executable,
            str(SUMMARY_SCRIPT),
            "--benchmark",
            str(paths[0]),
            "--kernels",
            str(paths[1]),
            "--counters",
            str(paths[2]),
        ],
        cwd=tmp_path,
        env={"PATH": "", "PYTHONPATH": str(ROOT)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert bad.returncode == 2
    assert "cannot read cuda_counters report" in bad.stderr


def test_marker_parser_requires_one_ordered_profile_pair() -> None:
    assert kernel_tools.parse_markers(
        "warmup marker\nprofile_run_start run=0 unix_ns=100\nprofile_run_end run=0 unix_ns=250\n"
    ) == (100, 250)
    for stderr in (
        "",
        "profile_run_start run=0 unix_ns=100",
        "profile_run_end run=0 unix_ns=250\nprofile_run_start run=0 unix_ns=100",
        "profile_run_start run=0 unix_ns=100\nprofile_run_end run=1 unix_ns=250",
        "profile_run_start run=0 unix_ns=100\nprofile_run_end run=0 unix_ns=100",
    ):
        with pytest.raises(ValueError):
            kernel_tools.parse_markers(stderr)


def test_stdout_profile_parser_selects_only_one_schema_candidate() -> None:
    profile = single_fit_profile()
    profile["backend"] = "Cuda { device_index: 0 }"
    profile = benchmark_tools._validate_profile(profile, "cuda", runs=1)
    assert kernel_tools.select_profile(f"nsys message\n{json.dumps(profile)}\n") == profile
    with pytest.raises(ValueError, match="found 0"):
        kernel_tools.select_profile("nsys says no JSON here")
    with pytest.raises(ValueError, match="found 2"):
        kernel_tools.select_profile(f"{json.dumps(profile)}\n{json.dumps(profile)}")
    incomplete = {"fit_seconds": [1], "stage_seconds": {}}
    with pytest.raises(ValueError, match="found 0"):
        kernel_tools.select_profile(json.dumps(incomplete))
    wrong_backend = single_fit_profile()
    wrong_backend["backend"] = "Cpu"
    with pytest.raises(ValueError, match="backend"):
        kernel_tools.select_profile(json.dumps(wrong_backend))


def synthetic_nsys_db(path: Path, *, missing_table: bool = False) -> None:
    import sqlite3

    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)")
        db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (1000)")
        db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
        db.executemany(
            "INSERT INTO StringIds VALUES (?, ?)",
            [(1, "partial_kernel_hash"), (2, "tile_candidates_hash"), (3, "future_kernel")],
        )
        if not missing_table:
            db.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL ("
                "start INTEGER, end INTEGER, demangledName INTEGER, "
                "gridX INTEGER, gridY INTEGER, gridZ INTEGER, "
                "blockX INTEGER, blockY INTEGER, blockZ INTEGER)"
            )
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (100, 120, 1, 1, 1, 1, 32, 1, 1),
                    (110, 140, 1, 1, 1, 1, 32, 1, 1),
                    (150, 170, 2, 2, 1, 1, 256, 1, 1),
                    (90, 110, 3, 1, 1, 1, 8, 1, 1),
                    (190, 210, 3, 1, 1, 1, 8, 1, 1),
                ],
            )


def test_sqlite_kernel_aggregation_aligns_window_excludes_warmup_and_groups_geometry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "synthetic.sqlite"
    synthetic_nsys_db(path)
    result = kernel_tools.aggregate_kernels(path, 1100, 1200)
    assert result["window"] == {
        "start_unix_ns": 1100,
        "end_unix_ns": 1200,
        "start_ns": 100,
        "end_ns": 200,
    }
    assert result["boundary_straddling_count"] == 2
    assert result["overlapping_activity"] is True
    assert result["total_device_seconds"] == pytest.approx(70e-9)
    kernels = {record["name"]: record for record in result["kernels"]}
    assert kernels["partial_kernel_hash"]["launch_count"] == 2
    assert kernels["partial_kernel_hash"]["family"] == "partial_kernel"
    assert kernels["tile_candidates_hash"]["launch_count"] == 1
    assert "future_kernel" not in kernels


@pytest.mark.parametrize("family", ["pack_leaf_rows_kernel", "kd_neighbors_tile_kernel"])
def test_sqlite_kernel_aggregation_recognizes_phase7_families(
    tmp_path: Path,
    family: str,
) -> None:
    import sqlite3

    path = tmp_path / "phase7.sqlite"
    synthetic_nsys_db(path)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE StringIds SET value = ? WHERE id = 1", (f"{family}_hash",))
    result = kernel_tools.aggregate_kernels(path, 1100, 1200)
    matching = [row for row in result["kernels"] if row["family"] == family]
    assert len(matching) == 1
    assert matching[0]["launch_count"] == 2


@pytest.mark.parametrize(
    "start,end,diagnostic",
    [(0, 1100, "invalid measured interval"), (900, 1000, "cannot be aligned")],
)
def test_sqlite_aggregation_rejects_invalid_alignment(
    tmp_path: Path, start: int, end: int, diagnostic: str
) -> None:
    path = tmp_path / "synthetic.sqlite"
    synthetic_nsys_db(path)
    with pytest.raises(ValueError, match=diagnostic):
        kernel_tools.aggregate_kernels(path, start, end)


def test_sqlite_aggregation_rejects_missing_schema_and_empty_window(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    synthetic_nsys_db(missing, missing_table=True)
    with pytest.raises(ValueError, match="missing tables"):
        kernel_tools.aggregate_kernels(missing, 1100, 1200)
    empty = tmp_path / "empty.sqlite"
    synthetic_nsys_db(empty)
    with pytest.raises(ValueError, match="no kernels fully contained"):
        kernel_tools.aggregate_kernels(empty, 3000, 4000)


def ncu_csv(
    *,
    family: str = "tile_candidates",
    launch_id: str = "7",
    metric_overrides: dict[str, tuple[str, str]] | None = None,
) -> str:
    import csv

    metric_overrides = metric_overrides or {}
    metrics = (
        ("gpu__time_duration.sum", "ns", "360,215,648"),
        ("launch__grid_size", "", "32,768"),
        ("launch__block_size", "", "256"),
        ("launch__registers_per_thread", "register/thread", "48"),
        ("launch__occupancy_limit_warps", "block", "4"),
        ("l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", "sector", "0"),
        ("l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum", "sector", "4"),
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", "sector", "12"),
        ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", "", "3"),
        ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum", "sector", "8"),
        ("l1tex__t_requests_pipe_lsu_mem_global_op_st.sum", "", "2"),
        ("smsp__pcsamp_warps_issue_stalled_long_scoreboard", "warp", "2"),
        ("smsp__pcsamp_sample_count", "", "8"),
        ("sm__warps_active.avg.pct_of_peak_sustained_active", "%", "99.18"),
        ("sm__throughput.avg.pct_of_peak_sustained_elapsed", "%", "10.89"),
        ("dram__throughput.avg.pct_of_peak_sustained_elapsed", "%", "18.71"),
        ("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", "%", "94.34"),
        ("sm__maximum_warps_per_active_cycle_pct", "%", "42.5"),
        ("smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct", "%", "71.79"),
        ("smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct", "%", "24.72"),
    )
    header = [
        "ID",
        "Process ID",
        "Process Name",
        "Host Name",
        "Kernel Name",
        "Context",
        "Stream",
        "Block Size",
        "Grid Size",
        "Device",
        "CC",
        "Section Name",
        "Metric Name",
        "Metric Unit",
        "Metric Value",
    ]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["==PROF== fake preamble"])
    writer.writerow(header)
    for metric, default_unit, default_value in metrics:
        unit, value = metric_overrides.get(metric, (default_unit, default_value))
        writer.writerow(
            [
                launch_id,
                "123",
                "profile",
                "host",
                f"{family}_f32_hash",
                "1",
                "14",
                "(256, 1, 1)",
                "(32768, 1, 1)",
                "0",
                "8.6",
                "metrics",
                metric,
                unit,
                value,
            ]
        )
    return output.getvalue()


def test_ncu_parser_handles_preamble_quotes_numbers_and_normalizes_units() -> None:
    result = counter_tools.parse_ncu_csv(ncu_csv(), "tile_candidates")
    assert result["kernel_name"] == "tile_candidates_f32_hash"
    assert result["launch_id"] == "7"
    assert result["geometry"] == {"block": "(256, 1, 1)", "grid": "(32768, 1, 1)"}
    duration = result["metrics"]["gpu__time_duration.sum"]
    assert duration["normalized_value"] == pytest.approx(0.360215648)
    assert duration["normalized_unit"] == "s"
    assert result["metrics"]["launch__grid_size"]["normalized_value"] == 32768
    assert result["metrics"]["launch__grid_size"]["normalized_unit"] == "count"
    assert (
        result["metrics"]["l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum"]["normalized_value"] == 0
    )
    registers = result["metrics"]["launch__registers_per_thread"]
    assert registers["normalized_value"] == 48
    assert registers["normalized_unit"] == "register/thread"
    ratio = result["ratios"]["global_load_sectors_per_request"]
    assert ratio["status"] == "available"
    assert ratio["value"] == 4
    assert ratio["unit"] == "sector/request"
    store_ratio = result["ratios"]["global_store_sectors_per_request"]
    assert store_ratio["status"] == "available"
    assert store_ratio["value"] == 4
    assert store_ratio["unit"] == "sector/request"
    assert ratio["numerator"] == {
        "metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "raw_value": 12,
        "raw_unit": "sector",
        "value": 12,
        "unit": "sector",
    }
    assert ratio["denominator"] == {
        "metric": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "raw_value": 3,
        "raw_unit": "",
        "value": 3,
        "unit": "request",
    }
    sample_ratio = result["ratios"]["long_scoreboard_share"]
    assert sample_ratio["value"] == 0.25
    assert sample_ratio["status"] == "available"
    assert sample_ratio["unavailable_reason"] is None
    assert sample_ratio["numerator"]["metric"] == (
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard"
    )
    assert sample_ratio["denominator"] == {
        "metric": "smsp__pcsamp_sample_count",
        "raw_value": 8,
        "raw_unit": "",
        "value": 8,
        "unit": "sampled warp",
    }
    theoretical = result["ratios"]["theoretical_occupancy"]
    assert theoretical["status"] == "available"
    assert theoretical["value"] == 42.5
    assert theoretical["achieved"]["value"] == 99.18
    assert theoretical["theoretical"]["metric"] == "sm__maximum_warps_per_active_cycle_pct"
    assert theoretical["source"] == "Nsight Compute Occupancy.section header metric"


def test_ncu_parser_zero_denominators_are_explicitly_unavailable() -> None:
    result = counter_tools.parse_ncu_csv(
        ncu_csv(
            metric_overrides={
                "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": ("", "0"),
                "smsp__pcsamp_sample_count": ("", "0"),
            }
        ),
        "tile_candidates",
    )
    ratio = result["ratios"]["global_load_sectors_per_request"]
    assert ratio["value"] is None
    assert ratio["status"] == "unavailable"
    assert ratio["unavailable_reason"] == "zero denominator"
    assert ratio["denominator"]["value"] == 0
    sample_ratio = result["ratios"]["long_scoreboard_share"]
    assert sample_ratio["value"] is None
    assert sample_ratio["status"] == "unavailable"
    assert sample_ratio["unavailable_reason"] == "zero denominator"


def test_ncu_parser_rejects_missing_unknown_or_ambiguous_metric_records() -> None:
    text = ncu_csv()
    with pytest.raises(ValueError, match="no Nsight Compute launch matched"):
        counter_tools.parse_ncu_csv(text, "partial_kernel")
    with pytest.raises(ValueError, match="header was not found"):
        counter_tools.parse_ncu_csv("unstructured output", "tile_candidates")
    missing = "\n".join(line for line in text.splitlines() if "gpu__time_duration.sum" not in line)
    with pytest.raises(ValueError, match=r"gpu__time_duration\.sum"):
        counter_tools.parse_ncu_csv(missing, "tile_candidates")
    lines = text.splitlines()
    duration_row = next(i for i, line in enumerate(lines) if "gpu__time_duration.sum" in line)
    fields = next(csv.reader([lines[duration_row]]))
    fields[0] = "8"
    output = io.StringIO()
    csv.writer(output).writerow(fields)
    duplicate_launch = text + output.getvalue()
    with pytest.raises(ValueError, match="ambiguous Nsight Compute capture"):
        counter_tools.parse_ncu_csv(duplicate_launch, "tile_candidates")


def test_ncu_capture_rejects_preexisting_trace_without_modifying_it(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    family = "kd_neighbors_kernel"
    trace = raw / f"ncu_{family}.trace.json"
    trace.write_text("preserve existing trace\n", encoding="utf-8")
    expected = (
        raw / f"ncu_{family}.csv",
        raw / f"ncu_{family}.stdout",
        raw / f"ncu_{family}.stderr",
        trace,
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite profiler evidence"):
        counter_tools._require_fresh_capture_paths(raw, family)
    assert trace.read_text(encoding="utf-8") == "preserve existing trace\n"
    assert not any(path.exists() for path in expected[:3])


def test_nsys_kernel_profiler_rejects_k16_manifest_before_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "k16 workload"
    raw = run_dir / "raw"
    raw.mkdir(parents=True)
    run_manifest = manifest()
    run_manifest["workload"] = {"rows": 20_000, "features": 16, "config": {"min_samples": 16}}
    (raw / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    commands: list[list[str]] = []
    monkeypatch.setattr(kernel_tools, "run_command", lambda argv, **kwargs: commands.append(argv))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_hdbscan_kernels.py",
            "--binary",
            str(tmp_path / "profile-binary"),
            "--rows",
            "20000",
            "--run-dir",
            str(run_dir),
            "--timeout",
            "10",
        ],
    )

    with pytest.raises(ValueError, match="supports only min_samples=5"):
        kernel_tools.main()

    assert commands == []
    assert not (run_dir / "cuda_kernels.json").exists()


@pytest.mark.parametrize(
    "metric",
    [
        "launch__occupancy_limit_warps",
        "launch__registers_per_thread",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
        "smsp__pcsamp_warps_issue_stalled_long_scoreboard",
        "smsp__pcsamp_sample_count",
        "sm__maximum_warps_per_active_cycle_pct",
    ],
)
def test_ncu_parser_rejects_missing_expanded_metrics(metric: str) -> None:
    text = ncu_csv()
    missing = "\n".join(line for line in text.splitlines() if metric not in line)
    with pytest.raises(ValueError, match="expected exactly one"):
        counter_tools.parse_ncu_csv(missing, "tile_candidates")


def test_ncu_parser_rejects_bad_values_duplicate_metrics_and_geometry(tmp_path: Path) -> None:
    text = ncu_csv()
    malformed = text.replace("360,215,648", "not-a-number")
    with pytest.raises(ValueError, match="invalid numeric"):
        counter_tools.parse_ncu_csv(malformed, "tile_candidates")
    fractional_sector = ncu_csv(
        metric_overrides={"l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum": ("sector", "1.5")}
    )
    with pytest.raises(ValueError, match="nonnegative integer"):
        counter_tools.parse_ncu_csv(fractional_sector, "tile_candidates")
    duplicate_metric = (
        text + next(line for line in text.splitlines() if "gpu__time_duration.sum" in line) + "\n"
    )
    with pytest.raises(ValueError, match=r"exactly one gpu__time_duration\.sum"):
        counter_tools.parse_ncu_csv(duplicate_metric, "tile_candidates")
    geometry = text.replace('"(256, 1, 1)","(32768, 1, 1)"', '"(128, 1, 1)","(32768, 1, 1)"', 1)
    with pytest.raises(ValueError, match="inconsistent geometry"):
        counter_tools.parse_ncu_csv(geometry, "tile_candidates")
    with pytest.raises(ValueError, match="unsupported unit"):
        counter_tools.normalize_metric("gpu__time_duration.sum", 12, "ticks")
    with pytest.raises(ValueError, match="positive integer"):
        counter_tools.normalize_metric("launch__grid_size", 1.5, "")
    with pytest.raises(ValueError, match="unsupported unit"):
        counter_tools.normalize_metric("launch__block_size", 32, "threads")
    assert counter_tools.normalize_metric(
        "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 0, "sector"
    ) == (0, "sector")
    with pytest.raises(ValueError, match="unsupported unit"):
        counter_tools.normalize_metric(
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 1, "request"
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        counter_tools.normalize_metric(
            "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum", 1.5, "sector"
        )
    assert counter_tools.normalize_metric(
        "launch__registers_per_thread", 72, "register/thread"
    ) == (72, "register/thread")
    with pytest.raises(ValueError, match="positive integer"):
        counter_tools.normalize_metric("launch__registers_per_thread", 0, "register/thread")


def synthetic_native_profile(backend: str = "Cpu") -> dict[str, Any]:
    return {
        "fit_seconds": [2.0, 2.1, 1.9],
        "warmup_seconds": 1.2,
        "stage_seconds": {stage: [0.2, 0.3, 0.4] for stage in benchmark_tools.STAGES},
        "core_distance_samples": [
            {
                "stage_seconds": {stage: [0.1] for stage in benchmark_tools.CORE_STAGES},
                "scope": "one fit",
            }
            for _ in range(3)
        ],
        "boruvka_samples": [
            {
                "total_seconds": 1.0,
                "round_count": 1,
                "round_seconds": [0.8],
                "round_stage_seconds": {stage: [0.16] for stage in benchmark_tools.BORUVKA_STAGES},
                "component_counts_per_round": [5],
                "seed_init_seconds": [0.1],
                "finalize_seconds": [0.1],
                "seed_and_round_overhead_seconds": 0.2,
                "scope": "one fit",
            }
            for _ in range(3)
        ],
        "cluster_counts": [4, 4, 4],
        "noise_counts": [2, 2, 2],
        "backend": backend,
        "search": "Auto",
        "resolved_search_per_fit": ["BruteForce"] * 3,
    }


def single_fit_profile() -> dict[str, Any]:
    profile = synthetic_native_profile()
    profile["fit_seconds"] = profile["fit_seconds"][:1]
    profile["stage_seconds"] = {
        name: values[:1] for name, values in profile["stage_seconds"].items()
    }
    for name in (
        "core_distance_samples",
        "boruvka_samples",
        "cluster_counts",
        "noise_counts",
        "resolved_search_per_fit",
    ):
        profile[name] = profile[name][:1]
    return profile


def test_native_profile_parser_excludes_warmup_and_computes_sample_medians() -> None:
    profile = synthetic_native_profile()
    parsed = benchmark_tools._validate_profile(profile, "cpu")
    assert parsed["medians"]["fit_seconds"] == 2.0
    assert parsed["medians"]["stage_seconds"]["mst"] == 0.3
    text = "profiler banner\n" + json.dumps(synthetic_native_profile()) + "\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native.stdout"
        path.write_text(text, encoding="utf-8")
        loaded = benchmark_tools._load_profile(path, "cpu")
    assert loaded["fit_seconds"] == [2.0, 2.1, 1.9]
    assert loaded["warmup_seconds"] == 1.2


@pytest.mark.parametrize(
    "mutate,diagnostic",
    [
        (lambda profile: profile.update(fit_seconds=[1.0]), "exactly 3 fit samples"),
        (lambda profile: profile.update(search="BruteForce"), "requested automatic search"),
        (lambda profile: profile.update(backend="Cuda"), "unexpected backend identity"),
        (
            lambda profile: profile["boruvka_samples"][0].update(component_counts_per_round=[]),
            "component counts do not align",
        ),
        (
            lambda profile: profile["stage_seconds"].update(mst=[0.1, float("inf"), 0.1]),
            "finite and nonnegative",
        ),
    ],
)
def test_native_profile_parser_rejects_invalid_samples_and_identity(mutate, diagnostic) -> None:
    profile = synthetic_native_profile()
    mutate(profile)
    with pytest.raises(ValueError, match=diagnostic):
        benchmark_tools._validate_profile(profile, "cpu")


def test_native_profile_parser_rejects_ambiguous_stdout_profiles(tmp_path: Path) -> None:
    profile = json.dumps(synthetic_native_profile())
    path = tmp_path / "mixed.stdout"
    path.write_text(f"banner\n{profile}\n{profile}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one native profile"):
        benchmark_tools._load_profile(path, "cpu")


def test_native_profile_parser_rejects_backend_mismatch() -> None:
    with pytest.raises(ValueError, match="unexpected backend identity"):
        benchmark_tools._validate_profile(synthetic_native_profile("Cuda(0)"), "cpu")
    with pytest.raises(ValueError, match="unexpected backend identity"):
        benchmark_tools._validate_profile(synthetic_native_profile("Cpu"), "cuda")


def test_deterministic_input_matches_existing_generator_shape_bytes_and_hash(
    tmp_path: Path,
) -> None:
    from comparison.compare_hdbscan_stages import make_input

    generated = tmp_path / "generated.f32le"
    reference = tmp_path / "reference.f32le"
    benchmark_tools._generate_input(generated, 256)
    make_input(256, 16, 1.0, 1.0, 42, reference)
    generated_bytes = generated.read_bytes()
    reference_bytes = reference.read_bytes()
    assert generated_bytes == reference_bytes
    assert len(generated_bytes) == 256 * 16 * 4
    assert (
        __import__("hashlib").sha256(generated_bytes).digest()
        == __import__("hashlib").sha256(reference_bytes).digest()
    )


def run_fake_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    run_dir: Path,
    *,
    mismatch: bool = False,
    failure_backend: str | None = None,
    rows: int = 20_000,
    min_samples: int = 5,
) -> dict[str, Any]:
    binary = run_dir / "profile-binary"
    binary.parent.mkdir(parents=True, exist_ok=True)
    binary.write_text("synthetic executable", encoding="utf-8")
    binary.chmod(0o755)

    def write_input(path: Path, rows: int) -> None:
        path.write_bytes(b"\0" * (rows * benchmark_tools.FEATURES * 4))

    monkeypatch.setattr(benchmark_tools, "_generate_input", write_input)
    monkeypatch.setattr(benchmark_tools, "_tool_version", lambda *args: {"available": False})
    monkeypatch.setattr(benchmark_tools, "_repository_provenance", lambda: {"revision": "test"})
    calls: list[list[str]] = []

    def fake_run_command(argv, *, timeout, stdout, stderr):
        args = list(argv)
        calls.append(args)
        backend = args[args.index("--backend") + 1]
        if backend == failure_backend:
            return {
                "argv": args,
                "exit_status": 9,
                "timeout_seconds": timeout,
                "timed_out": False,
                "elapsed_seconds": 0.01,
            }
        stdout.write(json.dumps(synthetic_native_profile("Cpu" if backend == "cpu" else "Cuda(0)")))
        labels_path = Path(args[args.index("--labels-output") + 1])
        trace_path = Path(args[args.index("--trace-output") + 1])
        labels = [0] * rows
        if mismatch and backend == "cuda":
            labels[9] = 1
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        trace_path.write_text("[]", encoding="utf-8")
        return {
            "argv": args,
            "exit_status": 0,
            "timeout_seconds": timeout,
            "timed_out": False,
            "elapsed_seconds": 0.01,
        }

    monkeypatch.setattr(benchmark_tools, "run_command", fake_run_command)
    report = benchmark_tools.benchmark(binary, rows, run_dir, 10, min_samples)
    assert [args[args.index("--backend") + 1] for args in calls] == ["cpu", "cuda"]
    return report


@pytest.mark.parametrize("rows", [20_000, 50_000, 100_000, 200_000, 300_000])
def test_benchmark_runs_cpu_then_cuda_and_persists_valid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: int
) -> None:
    run_dir = tmp_path / "benchmark success"
    report = run_fake_benchmark(monkeypatch, run_dir, rows=rows)
    persisted = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert report["correctness"] == {"labels_equal": True, "rows_compared": rows}
    assert report["profiles"]["cpu"]["fit_seconds"] == [2.0, 2.1, 1.9]
    assert report["profiles"]["cuda"]["backend"] == "Cuda(0)"


def test_benchmark_rejects_unsupported_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rows must be one of"):
        benchmark_tools.benchmark(tmp_path / "missing", 400_000, tmp_path / "run", 10)


@pytest.mark.parametrize("script", ["benchmark_hdbscan_stages.py", "profile_hdbscan_counters.py"])
@pytest.mark.parametrize("rows", ["200000", "300000", "400000"])
def test_standalone_tools_accept_extended_rows(tmp_path: Path, script: str, rows: str) -> None:
    argv = [sys.executable, str(ROOT / "analyze" / script), "--binary", str(tmp_path / "none")]
    if script == "profile_hdbscan_counters.py":
        argv += ["--binary-label", "current", "--min-samples", "5"]
    argv += ["--rows", rows, "--run-dir", str(tmp_path / "run"), "--timeout", "1"]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path, check=False)
    assert result.returncode != 0
    if rows == "400000":
        assert result.returncode == 2
        assert "invalid choice" in result.stderr
    else:
        assert result.returncode == 1
        assert "invalid choice" not in result.stderr


def test_benchmark_provisions_k16_and_records_manifest_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = run_fake_benchmark(monkeypatch, tmp_path / "k16", min_samples=16)
    assert report["workload"]["config"]["min_samples"] == 16
    for backend in ("cpu", "cuda"):
        argv = report["profiles"][backend]["command_provenance"]["argv"]
        assert argv[argv.index("--min-samples") + 1] == "16"


@pytest.mark.parametrize(
    "mismatch,failure_backend,diagnostic",
    [(True, None, "labels differ at row 9"), (False, "cuda", "benchmark exited 9")],
)
def test_benchmark_failures_never_write_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mismatch: bool,
    failure_backend: str | None,
    diagnostic: str,
) -> None:
    run_dir = tmp_path / "benchmark failed"
    with pytest.raises((ValueError, RuntimeError), match=diagnostic):
        run_fake_benchmark(
            monkeypatch,
            run_dir,
            mismatch=mismatch,
            failure_backend=failure_backend,
        )
    assert not (run_dir / "benchmark.json").exists()


def test_identity_reads_only_valid_shared_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    manifest_path = run_dir / "raw" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
    result = identity(run_dir)
    assert result == manifest()
    for changed in (
        {**manifest(), "schema_version": 2},
        {**manifest(), "run_id": None},
        {**manifest(), "input_sha256": "not-a-digest"},
        {**manifest(), "workload": []},
        {key: value for key, value in manifest().items() if key != "run_id"},
    ):
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError):
            identity(run_dir)


def test_identity_reports_missing_and_malformed_manifest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read run identity manifest"):
        identity(tmp_path)
    path = tmp_path / "raw" / "manifest.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read run identity manifest"):
        identity(tmp_path)


def test_atomic_json_preserves_previous_report_on_serialization_error(tmp_path: Path) -> None:
    path = tmp_path / "reports" / "report.json"
    atomic_json(path, {"prior": True})
    with pytest.raises(ValueError):
        atomic_json(path, {"invalid": float("nan")})
    assert json.loads(path.read_text(encoding="utf-8")) == {"prior": True}
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("value,fields", [(None, ("x",)), ({}, ("x",))])
def test_require_fields_rejects_nonobjects_and_missing_fields(value, fields) -> None:
    with pytest.raises(ValueError):
        require_fields(value, fields, "synthetic report")


def test_run_command_records_success_and_nonzero_exit(tmp_path: Path) -> None:
    for return_code in (0, 7):
        stdout_path = tmp_path / f"stdout-{return_code}.txt"
        stderr_path = tmp_path / f"stderr-{return_code}.txt"
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = run_command(
                [
                    sys.executable,
                    "-c",
                    "import sys; print('out'); print('err', file=sys.stderr); "
                    f"sys.exit({return_code})",
                ],
                timeout=3,
                stdout=stdout,
                stderr=stderr,
            )
        assert result["exit_status"] == return_code
        assert result["timed_out"] is False
        assert result["timeout_seconds"] == 3
        assert result["elapsed_seconds"] >= 0
        assert stdout_path.read_text(encoding="utf-8").strip() == "out"
        assert stderr_path.read_text(encoding="utf-8").strip() == "err"


def test_run_command_rejects_invalid_executable_and_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="argv"):
        run_command([], timeout=1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with pytest.raises(ValueError, match="timeout"):
        run_command(
            [sys.executable, "-c", "pass"],
            timeout=0,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def process_exited(proc_stat: Path) -> bool:
    try:
        return proc_stat.read_text(encoding="utf-8").split()[2] == "Z"
    except FileNotFoundError:
        # Reaping can remove the entry at any point, including during the read.
        return True


@pytest.mark.parametrize("state", ["missing", "Z", "S"])
def test_process_exited_handles_reaping(tmp_path: Path, state: str) -> None:
    proc_stat = tmp_path / "stat"
    if state != "missing":
        proc_stat.write_text(f"123 (python) {state} 0", encoding="utf-8")
    assert process_exited(proc_stat) is (state != "S")


def test_run_command_timeout_kills_process_group_and_reaps_child(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child.pid"
    child_code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    parent_code = (
        "import pathlib, signal, subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', "
        + repr(child_code)
        + "]); pathlib.Path(sys.argv[1]).write_text(str(child.pid)); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    )
    with (
        (tmp_path / "runner.out").open("w", encoding="utf-8") as stdout,
        (tmp_path / "runner.err").open("w", encoding="utf-8") as stderr,
    ):
        result = run_command(
            [sys.executable, "-c", parent_code, str(child_pid_file)],
            timeout=0.2,
            stdout=stdout,
            stderr=stderr,
        )
    assert result["timed_out"] is True
    assert result["exit_status"] is not None
    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    proc_stat = Path(f"/proc/{child_pid}/stat")
    deadline = time.monotonic() + 2
    while not process_exited(proc_stat) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert process_exited(proc_stat)


def test_run_command_interrupt_stops_and_reaps_process_group(tmp_path: Path, monkeypatch) -> None:
    # Exercise cleanup on an exception from waiting without delivering a signal to pytest.
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    real_wait = proc.wait
    should_interrupt = True

    def interrupt_once(timeout=None):
        nonlocal should_interrupt
        if should_interrupt:
            should_interrupt = False
            raise KeyboardInterrupt
        return real_wait(timeout=timeout)

    proc.wait = interrupt_once
    monkeypatch.setattr(common_tools.subprocess, "Popen", lambda *args, **kwargs: proc)
    with pytest.raises(KeyboardInterrupt):
        run_command(
            [sys.executable, "-c", "pass"],
            timeout=1,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    assert proc.poll() is not None
