from __future__ import annotations

import copy
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "compare_umap_spans", Path(__file__).resolve().parents[2] / "scripts/compare_umap_spans.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_deadline_rejects_elapsed_budget() -> None:
    deadline = module.Deadline(1.0)
    deadline.ends_at = 0.0
    with pytest.raises(module.DeadlineExceeded, match="deadline expired"):
        deadline.remaining()


@pytest.mark.parametrize("seconds", [0, -1])
def test_deadline_requires_positive_budget(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive"):
        module.Deadline(seconds)


def test_worker_timeout_is_reported_as_incomplete_deadline() -> None:
    with pytest.raises(module.DeadlineExceeded, match="during worker execution") as error:
        module.run_with_deadline(
            [sys.executable, "-c", "import time; time.sleep(30)"], module.Deadline(0.1)
        )
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_supervised_check_false_returns_nonzero_process_result() -> None:
    result = module.run_supervised(
        [sys.executable, "-c", "raise SystemExit(7)"], module.Deadline(2), check=False
    )
    assert result.returncode == 7


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_supervisor_kills_process_group_on_deadline(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child-pid"
    code = (
        "import subprocess,time, pathlib; p=subprocess.Popen(['python','-c',"
        '"import os,time; os.setsid(); time.sleep(30)"]); '
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    )
    with pytest.raises(module.DeadlineExceeded, match="supervised worker"):
        module.run_supervised([sys.executable, "-c", code], module.Deadline(0.3))
    child_pid = int(child_pid_file.read_text())
    for _ in range(20):
        if not Path(f"/proc/{child_pid}").exists():
            break
        __import__("time").sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_supervisor_kills_descendants_resisting_term_after_success(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child-pid"
    code = (
        "import signal,subprocess,pathlib; "
        "p=subprocess.Popen(['python','-c',\"import os,signal,time; os.setsid(); "
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"]); '
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid))"
    )
    completed = module.run_supervised([sys.executable, "-c", code], module.Deadline(3))
    assert completed.returncode == 0
    child_pid = int(child_pid_file.read_text())
    for _ in range(20):
        if not Path(f"/proc/{child_pid}").exists():
            break
        __import__("time").sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_supervisor_kills_tree_on_unexpected_telemetry_error(tmp_path: Path) -> None:
    child_pid_file = tmp_path / "child-pid"
    code = (
        "import subprocess,pathlib,time; "
        "p=subprocess.Popen(['sleep','30'],stdin=subprocess.DEVNULL,"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
    )

    class BrokenTelemetry:
        def sample(self, root_pid, deadline):
            while not child_pid_file.exists():
                __import__("time").sleep(min(0.01, deadline.remaining()))
            raise RuntimeError("unexpected telemetry failure")

    with pytest.raises(RuntimeError, match="unexpected telemetry"):
        module.run_supervised(
            [sys.executable, "-c", code], module.Deadline(3), telemetry=BrokenTelemetry()
        )
    child_pid = int(child_pid_file.read_text())
    for _ in range(20):
        if not Path(f"/proc/{child_pid}").exists():
            break
        __import__("time").sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


def test_cuda_command_matches_rust_positional_contract(tmp_path: Path) -> None:
    binary, train, out, query = (
        tmp_path / name for name in ("binary", "train.f32le", "out", "query.f32le")
    )
    command = module.stage2_cuda_command(
        binary, train, 16384, 16, 100, out, "exact", 47, 0, 1, 25, "spectral", [(query, 1024)]
    )
    assert command[:6] == [str(binary), str(train), "16384", "16", "100", str(out)]
    assert command[6:] == [
        "--search",
        "exact",
        "--warmups",
        "0",
        "--runs",
        "1",
        "--neighbors",
        "25",
        "--min-dist",
        "0.0",
        "--init",
        "spectral",
        "--seed",
        "47",
        "--query",
        str(query),
        "--query-rows",
        "1024",
    ]


@pytest.mark.parametrize("failure_mode", ["none", "truncated", "subprocess"])
def test_benchmark_cli_mocked_campaign_exercises_complete_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_mode: str
) -> None:
    binary = tmp_path / "cuda-binary"
    binary.write_bytes(b"mock binary")
    output_path = tmp_path / "phase0.json"
    calls: list[list[str]] = []
    reference_calls = []
    cuda_calls = []
    worker_calls = []

    def fake_supervised(command, deadline, *, env=None, telemetry=None, check=True):
        calls.append(command)
        if command[0] == "git":
            if command[1:3] == ["rev-parse", "HEAD"]:
                stdout = "test-head\n"
            else:
                stdout = ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

        if command[0] == "/mock/nsys":
            if command[1] == "profile":
                return subprocess.CompletedProcess(command, 0, "profile-ok", "")
            if command[1] == "export":
                sqlite_out = Path(command[command.index("-o") + 1])
                with sqlite3.connect(sqlite_out) as connection:
                    connection.execute(
                        "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS "
                        "(start INTEGER, globalPid INTEGER, deviceId INTEGER, "
                        "bytes INTEGER, memoryOperationType INTEGER)"
                    )
                    connection.executemany(
                        "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES (?, ?, ?, ?, ?)",
                        [(1, 100, 0, 4096, 0), (2, 100, 0, 4096, 1)],
                    )
                return subprocess.CompletedProcess(command, 0, "export-ok", "")
            return subprocess.CompletedProcess(command, 0, "FULL,NSYS,CSV\n", "stats-stderr")

        if command[0] == str(binary):
            cuda_calls.append(command)
            if failure_mode == "subprocess" and command[command.index("--warmups") + 1] == "0":
                raise subprocess.CalledProcessError(
                    17, command, "captured stdout", "distinctive stderr marker"
                )
            out_dir = Path(command[5])
            rows, query_rows = int(command[2]), 32
            np.zeros((rows, 2), dtype="<f4").tofile(out_dir / "embedding_0.f32le")
            np.tile(np.arange(25, dtype="<i4"), (rows, 1)).tofile(out_dir / "graph_0.i32le")
            query_paths = [
                Path(command[index + 1])
                for index, value in enumerate(command[:-1])
                if value == "--query"
            ]
            for batch, _ in enumerate(query_paths):
                np.zeros((query_rows, 2), dtype="<f4").tofile(
                    out_dir / f"transform_0_{batch}.f32le"
                )
                np.tile(np.arange(25, dtype="<i4"), (query_rows, 1)).tofile(
                    out_dir / f"query_graph_0_{batch}.i32le"
                )
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "fit_seconds": [1.0],
                        "transform_seconds": [[0.1, 0.2, 0.3]],
                        "query_graph_width": 25,
                    }
                ),
                "",
            )

        payload = json.loads(command[command.index("--_stage2_worker") + 1])
        worker_calls.append(payload.get("kind", "reference"))
        if payload.get("kind") == "generate":
            result = module._stage2_worker(payload)
        elif payload.get("kind") == "provenance":
            result = {
                "dependencies": {"umap-learn": "0.5.12"},
                "python": "test",
                "platform": "test",
                "gpu": {"status": "unavailable"},
                "source": {
                    "working_head": "test-head",
                    "rustc": "rustc test",
                    "binary_sha256": module.sha256_file(binary),
                    "script_sha256": "script",
                    "lockfile_sha256": "lock",
                    "harness_test_sha256": {},
                    "harness_diff_path": str(tmp_path / "harness.diff"),
                    "harness_diff_sha256": "diff",
                    "cuda_toolkit_nvcc_version": None,
                },
            }
            Path(payload["artifact_directory"], "harness.diff").write_text("mock diff")
        elif payload.get("kind") == "cuda-quality":
            diagnostic_path = Path(payload["diagnostic_path"])
            if failure_mode == "truncated":
                diagnostic_path.write_text('{"broken":', encoding="utf-8")
                raise module.DeadlineExceeded("simulated evaluator deadline")
            result = {
                "fit": {"trustworthiness": 0.95, "recall": 0.95},
                "transform": [{"trustworthiness": 0.95, "recall": 0.86} for _ in range(3)],
                "reverse_graph_diagnostics": {"observed_valid_edge_count": 10},
            }
            diagnostic_path.write_text(json.dumps(result["reverse_graph_diagnostics"]))
        else:
            reference_calls.append(payload)
            result = {
                "fit_trustworthiness": 0.95,
                "fit_recall": 0.95,
                "transform": [{"trustworthiness": 0.95} for _ in range(3)],
                "reference": {"version": "0.5.12", "mode": payload["mode"]},
            }
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(module, "run_supervised", fake_supervised)
    monkeypatch.setattr(
        module.shutil, "which", lambda name: "/mock/nsys" if name == "nsys" else None
    )
    status = module._stage2_cli(
        [
            "--suite",
            "stage2",
            "--rows",
            "32",
            "--features",
            "2",
            "--epochs",
            "10",
            "--query-rows",
            "32",
            "--warmups",
            "1",
            "--deadline-seconds",
            "90",
            "--binary",
            str(binary),
            "--output",
            str(output_path),
            "--profile-nsys",
        ]
    )

    report = json.loads(output_path.read_text())
    assert status == (1 if failure_mode != "none" else 0)
    assert worker_calls.count("generate") == 1
    assert worker_calls.count("provenance") == 1
    expected_references = 6 if failure_mode == "none" else 1
    assert len(reference_calls) == expected_references, report.get("incomplete_reason")
    if failure_mode in ("truncated", "subprocess"):
        # The reference receipt is committed before its first CUDA repetition starts.
        completed_references = report["completed_references"]
        assert len(completed_references) == 1
        assert len({(row["pair"], row["mode"], row["seed"]) for row in completed_references}) == 1
    if failure_mode == "truncated":
        assert completed_references[0]["mode"] in module.STAGE2_MODES
        assert report["incomplete_reason"] == "simulated evaluator deadline"
        assert report["reverse_graph_diagnostics"][0]["recovery_error"]
        assert report["coverage"]["profiles"] == "not requested"
        assert not any(command[0] == "git" for command in calls)
        return
    if failure_mode == "subprocess":
        assert report["completed_references"][0]["mode"] in module.STAGE2_MODES
        failure = report["failures"][0]
        details = json.loads(Path(failure["artifact"]).read_text())
        assert details["returncode"] == 17
        assert "distinctive stderr marker" in details["stderr"]
        assert details["stdout"] == "captured stdout"
        assert not any(command[0] == "git" for command in calls)
        return
    assert len(cuda_calls) == 16  # Four warmups and twelve timed baseline repetitions.
    assert sum("1" == call[call.index("--warmups") + 1] for call in cuda_calls) == 4
    assert sum("0" == call[call.index("--warmups") + 1] for call in cuda_calls) == 12
    assert all(payload["neighbors"] == 25 for payload in reference_calls)
    completed_references = report["completed_references"]
    assert len(completed_references) == 6
    assert report["coverage"]["quality_runs"] == 6
    assert len(completed_references) == report["coverage"]["quality_runs"]
    assert len({(row["pair"], row["mode"], row["seed"]) for row in completed_references}) == 6
    assert all(
        call[1:5] == [str(Path(report["raw_artifact_directory"]) / "train.f32le"), "32", "2", "10"]
        for call in cuda_calls
    )
    assert report["configuration"]["neighbors"] == 25
    assert report["configuration"]["paired_runs"] == 3
    assert report["configuration"]["fit_recall_threshold"] == 0.90
    assert report["configuration"]["query_recall_threshold"] == 0.87
    assert report["coverage"]["completed_baseline_repetitions"] == 12
    assert report["coverage"]["baseline_repetitions_required_per_mode"] == 6
    assert report["coverage"]["full_baseline_pair_coverage"] is True
    assert not any(command[0] == "git" for command in calls)
    assert "frozen_product_baseline" not in report["provenance"]["source_identity"]
    assert "frozen_product_src_tree" not in report["provenance"]["source_identity"]
    assert report["coverage"]["profiles"]["status"] == "captured"
    assert report["quality"]["exact"]["all_runs_pass"] is True
    assert report["quality"]["approximate"]["all_runs_pass"] is True
    assert report["resource_status"] == "pass"
    assert report["warnings"]
    assert all(item["kind"] == "query_recall_below_limit" for item in report["warnings"])
    assert report["timings"]
    assert report["paired_timing_differences"]
    assert all(call[0] == "/mock/nsys" for call in calls if call[0] == "/mock/nsys")
    profile_commands = [call for call in calls if call[:2] == ["/mock/nsys", "profile"]]
    assert all("--trace=cuda,nvtx,osrt" in call for call in profile_commands)
    assert all("--cuda-memory-usage=true" in call for call in profile_commands)
    profile_records = report["coverage"]["profiles"]["runs"]
    assert set(profile_records) == set(module.STAGE2_MODES)
    for profile in profile_records.values():
        assert Path(profile["stats_csv"]).read_text() == "FULL,NSYS,CSV\n"
        assert Path(profile["host_trace"]).name.startswith("host-trace-")
        assert profile["workload"] == "same fit plus three transforms as measured runs"
        assert profile["memory_evidence"]["status"] == "pass"


def test_input_generation_is_deterministic_and_balanced(tmp_path: Path) -> None:
    def generate(folder: Path):
        folder.mkdir()
        return module._stage2_worker(
            {
                "kind": "generate",
                "rows": 32,
                "features": 4,
                "query_rows": 16,
                "seeds": [43, 44, 45],
                "train": str(folder / "train.f32le"),
                "query_paths": [str(folder / f"query-{seed}.f32le") for seed in (43, 44, 45)],
            }
        )

    first, second = generate(tmp_path / "first"), generate(tmp_path / "second")
    assert first["train_sha256"] == second["train_sha256"]
    assert [row["sha256"] for row in first["queries"]] == [
        row["sha256"] for row in second["queries"]
    ]
    assert first["train_label_counts"] == [4] * 8
    assert all(max(row["label_counts"]) - min(row["label_counts"]) <= 1 for row in first["queries"])


def test_resource_telemetry_incomplete_ancestry_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "process_descendants",
        lambda pid: {"pids": [pid], "ancestry": {str(pid): [pid]}, "complete": False},
    )

    def unexpected_query(*args, **kwargs):
        pytest.fail("incomplete ancestry must not be attributed")

    monkeypatch.setattr(module, "_nvidia_smi", unexpected_query)
    telemetry = module.GpuProcessTelemetry()
    telemetry.sample(42, module.Deadline(2))
    assert telemetry.evidence["status"] == "unavailable"
    assert telemetry.evidence["sampled_max_bytes"] is None
    assert "process attribution is incomplete" in telemetry.evidence["explanation"]


def test_resource_telemetry_unavailable_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi missing in WSL")

    monkeypatch.setattr(
        module,
        "process_descendants",
        lambda pid: {"pids": [pid], "ancestry": {str(pid): [pid]}, "complete": True},
    )
    monkeypatch.setattr(module, "_nvidia_smi", unavailable)
    telemetry = module.GpuProcessTelemetry()
    telemetry.sample(42, module.Deadline(2))
    assert telemetry.evidence["status"] == "unavailable"
    assert telemetry.evidence["sampled_max_bytes"] is None
    assert "nvidia-smi missing in WSL" in telemetry.evidence["explanation"]


def test_resource_telemetry_timeout_skips_sample_without_disabling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = [0]

    def slow_then_answer(args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise subprocess.TimeoutExpired(["nvidia-smi"], 0.05)
        return subprocess.CompletedProcess(["nvidia-smi", *args], 0, "", "")

    monkeypatch.setattr(
        module,
        "process_descendants",
        lambda pid: {"pids": [pid], "ancestry": {str(pid): [pid]}, "complete": True},
    )
    monkeypatch.setattr(module, "_nvidia_smi", slow_then_answer)
    telemetry = module.GpuProcessTelemetry()
    telemetry.sample(42, module.Deadline(2))
    assert telemetry.evidence["timed_out_samples"] == 1
    assert telemetry.evidence["status"] == "pending"
    telemetry.sample(42, module.Deadline(2))
    assert calls[0] == 2
    assert telemetry.evidence["status"] == "no_attributable_process_seen"


def test_nsys_memory_evidence_tracks_interleaved_pid_device_allocations(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "memory.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS "
            "(start INTEGER, globalPid INTEGER, deviceId INTEGER, bytes INTEGER, "
            "memoryOperationType INTEGER)"
        )
        connection.executemany(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES (?, ?, ?, ?, ?)",
            [
                (1, 10, 0, 100, 0),
                (2, 20, 0, 200, 0),
                (3, 10, 0, 50, 0),
                (4, 20, 0, 50, 1),
                (5, 10, 0, 120, 1),
                (6, 20, 0, 10, 0),
            ],
        )
    evidence = module.nsys_cuda_memory_evidence(sqlite_path, 250)
    assert evidence["nsys_peak_live_allocation_bytes"] == 200
    assert evidence["allocation_event_count"] == 4
    assert evidence["free_event_count"] == 2
    assert evidence["per_pid_device_peaks"] == [
        {"globalPid": 10, "deviceId": 0, "peak_live_allocation_bytes": 150},
        {"globalPid": 20, "deviceId": 0, "peak_live_allocation_bytes": 200},
    ]
    assert evidence["pass"] is True
    assert module.nsys_cuda_memory_evidence(sqlite_path, 200)["status"] == "pass"
    assert module.nsys_cuda_memory_evidence(sqlite_path, 199)["status"] == "fail"


def test_nsys_memory_evidence_unknown_operation_is_incomplete(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "unknown.sqlite"
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS "
            "(start INTEGER, globalPid INTEGER, deviceId INTEGER, bytes INTEGER, "
            "memoryOperationType INTEGER)"
        )
        connection.executemany(
            "INSERT INTO CUDA_GPU_MEMORY_USAGE_EVENTS VALUES (?, ?, ?, ?, ?)",
            [(1, 10, 0, 100, 0), (2, 10, 0, 100, 2)],
        )
    evidence = module.nsys_cuda_memory_evidence(sqlite_path, module.STAGE2_GPU_CAP)
    assert evidence["status"] == "incomplete"
    assert evidence["pass"] is False
    assert evidence["unknown_event_count"] == 1
    assert evidence["nsys_peak_live_allocation_bytes"] is None


@pytest.mark.parametrize(
    ("second", "expected", "peak"),
    [
        ({"status": "pass", "nsys_peak_live_allocation_bytes": 300}, "pass", 300),
        ({"status": "fail", "nsys_peak_live_allocation_bytes": 900}, "fail", 900),
        ({"status": "incomplete", "nsys_peak_live_allocation_bytes": None}, "incomplete", None),
    ],
)
def test_finalize_resource_evidence_requires_every_mode(
    second: dict, expected: str, peak: int | None
) -> None:
    first, other = module.STAGE2_MODES
    evidence = {
        "nsys_modes": {
            first: {"status": "pass", "nsys_peak_live_allocation_bytes": 200},
            other: second,
        },
        "authoritative_lifetime_peak_bytes": None,
    }
    telemetry = module.GpuProcessTelemetry()
    assert module.finalize_resource_evidence(evidence, telemetry) == expected
    assert evidence["authoritative_lifetime_peak_bytes"] == peak
    assert evidence["supplementary_nvidia_smi"] is telemetry.evidence


def test_finalize_resource_evidence_missing_mode_is_incomplete() -> None:
    evidence = {
        "nsys_modes": {
            module.STAGE2_MODES[0]: {"status": "pass", "nsys_peak_live_allocation_bytes": 1}
        },
        "authoritative_lifetime_peak_bytes": None,
    }
    status = module.finalize_resource_evidence(evidence, module.GpuProcessTelemetry())
    assert status == "incomplete"
    assert evidence["authoritative_lifetime_peak_bytes"] is None


def test_nsys_memory_evidence_missing_table_is_incomplete(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "missing.sqlite"
    sqlite3.connect(sqlite_path).close()
    evidence = module.nsys_cuda_memory_evidence(sqlite_path, module.STAGE2_GPU_CAP)
    assert evidence["status"] == "incomplete"
    assert evidence["pass"] is False
    assert "missing" in evidence["explanation"]
    empty_path = tmp_path / "empty.sqlite"
    with sqlite3.connect(empty_path) as connection:
        connection.execute(
            "CREATE TABLE CUDA_GPU_MEMORY_USAGE_EVENTS "
            "(start INTEGER, globalPid INTEGER, deviceId INTEGER, bytes INTEGER, "
            "memoryOperationType INTEGER)"
        )
    empty = module.nsys_cuda_memory_evidence(empty_path, module.STAGE2_GPU_CAP)
    assert empty["status"] == "incomplete"
    assert "No CUDA allocation events" in empty["explanation"]


def test_resource_sample_unattributed_process_is_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        module,
        "_nvidia_smi",
        lambda args, **kwargs: subprocess.CompletedProcess(
            ["nvidia-smi", *args], 0, "999, 10, GPU-foreign\n", ""
        ),
    )
    monkeypatch.setattr(
        module,
        "process_descendants",
        lambda pid: {"pids": [pid], "ancestry": {str(pid): [pid]}, "complete": True},
    )
    telemetry = module.GpuProcessTelemetry()
    telemetry.sample(123, module.Deadline(2))
    assert telemetry.evidence["status"] == "no_attributable_process_seen"
    assert telemetry.evidence["instantaneous_process_bytes"] is None
    assert telemetry.evidence["sampled_max_bytes"] is None
    assert telemetry.evidence["readings"][0]["timestamp_utc"].endswith("+00:00")


def test_gpu_provenance_distinguishes_driver_compatibility_from_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def query(args, **kwargs):
        if args == ["--query-gpu=name,uuid,driver_version", "--format=csv,noheader"]:
            return subprocess.CompletedProcess([], 0, "RTX 3080, GPU-uuid, 550.54\n", "")
        if args == ["--version"]:
            return subprocess.CompletedProcess([], 0, "NVIDIA-SMI test version\n", "")
        return subprocess.CompletedProcess(
            [], 0, "| NVIDIA-SMI 550 Driver Version: 550 CUDA Version: 12.4 |\n", ""
        )

    monkeypatch.setattr(module, "_nvidia_smi", query)
    gpu = module.gpu_device_provenance()
    assert gpu["status"] == "available"
    assert gpu["devices"][0]["name"] == "RTX 3080"
    assert gpu["devices"][0]["driver_cuda_compatibility_version"] == "12.4"
    assert gpu["devices"][0]["cuda_runtime_version"] is None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_over_cap_sample_kills_supervised_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = [None]
    samples = [0]

    def nvidia(args, **kwargs):
        if args == ["--version"]:
            return subprocess.CompletedProcess(["nvidia-smi"], 0, "NVIDIA-SMI test\n", "")
        samples[0] += 1
        if samples[0] == 1:
            return subprocess.CompletedProcess(["nvidia-smi", *args], 0, "", "")
        return subprocess.CompletedProcess(
            ["nvidia-smi", *args], 0, f"{root[0]}, 7169, GPU-test\n", ""
        )

    monkeypatch.setattr(module, "_nvidia_smi", nvidia)
    original_descendants = module.process_descendants

    def ancestry(pid):
        root[0] = pid
        return original_descendants(pid)

    monkeypatch.setattr(module, "process_descendants", ancestry)
    telemetry = module.GpuProcessTelemetry()
    child_pid_file = tmp_path / "over-cap-child-pid"
    with pytest.raises(module.ResourceLimitExceeded, match="exceeds"):
        module.run_supervised(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess,time,pathlib; p=subprocess.Popen(['python','-c',"
                    '"import os,time; os.setsid(); time.sleep(30)"], '
                    "stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, "
                    "stderr=subprocess.DEVNULL); "
                    f"pathlib.Path({str(child_pid_file)!r}).write_text(str(p.pid)); time.sleep(30)"
                ),
            ],
            module.Deadline(2),
            telemetry=telemetry,
        )
    assert telemetry.evidence["status"] == "violation"
    assert telemetry.evidence["sampled_max_bytes"] > module.STAGE2_GPU_CAP
    assert telemetry.evidence["violation"]["processes"][0]["gpu_uuid"] == "GPU-test"
    child_pid = int(child_pid_file.read_text())
    for _ in range(20):
        if not Path(f"/proc/{child_pid}").exists():
            break
        __import__("time").sleep(0.05)
    assert not Path(f"/proc/{child_pid}").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_profile_worker_timeout_is_interruptible() -> None:
    with pytest.raises(module.DeadlineExceeded):
        module.run_supervised(
            [sys.executable, "-c", "import time; time.sleep(30)"], module.Deadline(0.1)
        )


def test_aggregation_checks_full_run_batch_coverage_without_speed_gate() -> None:
    references = {
        mode: [
            {
                "fit_trustworthiness": 0.95,
                "transform": [{"trustworthiness": 0.94 + batch / 100} for batch in range(3)],
            }
        ]
        for mode in module.STAGE2_MODES
    }
    records = []
    for mode in module.STAGE2_MODES:
        for side in ("A", "B"):
            records.append(
                {
                    "mode": mode,
                    "pair": 0,
                    "side": side,
                    "reference": {},
                    "fit_trustworthiness": 0.95,
                    "fit_recall": 0.91,
                    "fit_seconds": [999999.0],
                    "transform_total_seconds": 999999.0,
                    "transform": [
                        {
                            "batch": batch,
                            "seconds": 999999.0,
                            "trustworthiness": 0.94 + batch / 100,
                            "recall": 0.91,
                        }
                        for batch in range(3)
                    ],
                }
            )
    aggregate = module.stage2_aggregate(records, references, 1)
    assert aggregate["full_coverage"] is True
    assert aggregate["speed_gates"] == []
    assert all(result["all_runs_pass"] for result in aggregate["quality"].values())
    assert all(
        "trustworthiness_gap_from_reference_mean" in batch
        for result in aggregate["quality"].values()
        for row in result["runs"]
        for batch in row["transform"]
    )
    missing = module.stage2_aggregate(records[:-1], references, 1)
    assert missing["full_coverage"] is False


def test_fit_and_query_recall_thresholds_are_distinct() -> None:
    references = {
        mode: [
            {
                "fit_trustworthiness": 0.95,
                "transform": [{"trustworthiness": 0.95} for _ in range(3)],
            }
        ]
        for mode in module.STAGE2_MODES
    }

    def records(query_recall: float) -> list[dict]:
        return [
            {
                "mode": mode,
                "pair": 0,
                "side": side,
                "reference": {},
                "fit_trustworthiness": 0.95,
                "fit_recall": 0.90,
                "fit_seconds": [1.0],
                "transform_total_seconds": 0.3,
                "transform": [
                    {
                        "batch": batch,
                        "seed": batch + 43,
                        "seconds": 0.1,
                        "trustworthiness": 0.95,
                        "recall": query_recall,
                    }
                    for batch in range(3)
                ],
            }
            for mode in module.STAGE2_MODES
            for side in ("A", "B")
        ]

    at_threshold = module.stage2_aggregate(records(0.87), references, 1)
    assert all(result["all_runs_pass"] for result in at_threshold["quality"].values())
    below_query_threshold = module.stage2_aggregate(records(0.869), references, 1)
    assert all(result["all_runs_pass"] for result in below_query_threshold["quality"].values())
    assert all(
        result["query_recall_warning"] for result in below_query_threshold["quality"].values()
    )
    assert below_query_threshold["warnings"] == [
        {
            "kind": "query_recall_below_limit",
            "mode": mode,
            "pair": 0,
            "side": side,
            "seed": batch + 43,
            "recall": 0.869,
            "limit": 0.87,
        }
        for mode in module.STAGE2_MODES
        for side in ("A", "B")
        for batch in range(3)
    ]
    below_fit_threshold = records(0.87)
    for row in below_fit_threshold:
        row["fit_recall"] = 0.899
    fit_failure = module.stage2_aggregate(below_fit_threshold, references, 1)
    assert not any(result["all_runs_pass"] for result in fit_failure["quality"].values())


def test_reverse_graph_diagnostics_report_hubs_representability_and_workspace() -> None:
    graph = np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int32)
    result = module.reverse_graph_diagnostics(graph, rows=4)
    assert result["observed_valid_edge_count"] == 8
    assert result["reverse_indegree"]["max"] == 8
    assert result["reverse_indegree"]["hub_thresholds"]["count_greater_than_2x_mean"] == 1
    estimate = result["reverse_index_workspace_estimate_bytes"]
    assert estimate["counts_u32"] == 16
    assert estimate["cursors_u32"] == 16
    assert estimate["prefix_u64"] == 40
    assert estimate["reverse_ids_i32"] == 32
    assert estimate["concurrent_peak_estimate"] == 104
    assert "not measured allocations" in result["evidence_kind"]


def test_reverse_graph_diagnostics_reject_invalid_id() -> None:
    with pytest.raises(ValueError, match="invalid neighbor IDs"):
        module.reverse_graph_diagnostics(np.array([[0, 3]], dtype=np.int32), rows=3)


def test_query_recall_uses_training_ids_and_full_denominator() -> None:
    query = np.array([[0.0], [10.0]])
    train = np.array([[0.0], [1.0], [9.0], [11.0]])
    # Duplicate IDs count once; missing slots remain in the fixed denominator.
    actual = np.array([[0, 0, 3], [2, 3, -1]])
    assert module.stage2_query_recall(query, train, actual, 2) == pytest.approx(3 / 6)


def test_cuda_quality_worker_validates_raw_ids_and_pads_short_fit_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, features = 32, 2
    train = np.random.default_rng(3).normal(size=(rows, features)).astype("<f4")
    query = np.random.default_rng(4).normal(size=(rows, features)).astype("<f4")
    train_path, query_path = tmp_path / "train.f32le", tmp_path / "query.f32le"
    train.tofile(train_path)
    query.tofile(query_path)
    fit_embedding_path = tmp_path / "fit-embedding.f32le"
    fit_embedding = np.random.default_rng(5).normal(size=(rows, 2)).astype("<f4")
    fit_embedding.tofile(fit_embedding_path)
    fit_graph = np.full((rows, 2), -1, dtype="<i4")
    fit_graph[:, 0] = np.arange(rows)
    fit_graph[:, 1] = (np.arange(rows) + 1) % rows
    fit_graph_path = tmp_path / "fit-graph.i32le"
    fit_graph.tofile(fit_graph_path)
    transform_embedding_path = tmp_path / "transform.f32le"
    np.zeros((rows, 2), dtype="<f4").tofile(transform_embedding_path)
    query_graph_path = tmp_path / "query-graph.i32le"
    np.tile(np.arange(25, dtype="<i4"), (rows, 1)).tofile(query_graph_path)
    graph_widths = []
    original_recall = module.stage2_recall

    def record_recall(matrix, actual, neighbors):
        graph_widths.append(actual.shape[1])
        return original_recall(matrix, actual, neighbors)

    monkeypatch.setattr(module, "stage2_recall", record_recall)
    payload = {
        "kind": "cuda-quality",
        "train": str(train_path),
        "queries": [str(query_path)],
        "rows": rows,
        "features": features,
        "query_rows": rows,
        "neighbors": 25,
        "fit_embedding": str(fit_embedding_path),
        "fit_graph": str(fit_graph_path),
        "transform_embeddings": [str(transform_embedding_path)],
        "query_graphs": [str(query_graph_path)],
        "graph_width": 25,
    }
    result = module._stage2_worker(payload)
    assert graph_widths == [24]
    assert result["fit"]["recall"] >= 0

    fit_graph[0, 1] = -2
    fit_graph.tofile(fit_graph_path)
    with pytest.raises(ValueError, match="invalid raw neighbor IDs"):
        module._stage2_worker(payload)


def test_recall_rejects_invalid_ids() -> None:
    with pytest.raises(ValueError, match="invalid IDs"):
        module.stage2_recall(np.zeros((3, 2)), np.array([[3], [0], [1]]), 1)


def test_expiry_during_final_diagnostic_cannot_mark_suite_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sklearn.manifold

    clock = [0.0]
    calls = [0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "PRIMARY", (32, 2, 20))
    monkeypatch.setattr(module, "SMALL", ())

    def trustworthiness(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 4:
            clock[0] = 601.0
        return 0.95

    monkeypatch.setattr(sklearn.manifold, "trustworthiness", trustworthiness)
    embedding = np.zeros((32, 2), dtype="<f4")
    graph = (np.arange(32)[:, None] + np.arange(1, 15)[None, :]) % 32

    def worker(command, deadline):
        deadline.remaining()
        if "--_python-worker" in command:
            destination = Path(command[-1])
            np.save(destination / "embedding_0.npy", embedding)
            np.save(destination / "knn_0.npy", graph)
        else:
            destination = Path(command[5])
            destination.mkdir()
            embedding.tofile(destination / "embedding_0.f32le")
            graph.astype("<i4").tofile(destination / "graph_0.i32le")
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"fit_seconds": [0.1], "fit_graph_width": 14}), ""
        )

    monkeypatch.setattr(module, "run_with_deadline", worker)
    binary = tmp_path / "mock-binary"
    binary.touch()
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["suite", "--binary", str(binary), "--runs", "1", "--output", str(output)],
    )
    assert module._cli() == 1
    result = json.loads(output.read_text())
    assert calls[0] == 4
    assert result["complete"] is False
    assert "deadline expired" in result["incomplete_reason"]


def test_nonself_filters_self_and_missing_neighbors() -> None:
    actual = np.array([[0, -1, 2, 1], [2, 1, -1, 0], [-1, 0, 2, 1]])
    np.testing.assert_array_equal(module.nonself(actual, 2), [[2, 1], [2, 0], [0, 1]])
    with pytest.raises(ValueError, match="fewer than 3"):
        module.nonself(actual, 3)


def test_recall_counts_distinct_in_radius_neighbors_only() -> None:
    matrix = np.array([[0.0], [1.0], [10.0], [2.0]])
    actual = np.array([[1, 1, 2], [0, 0, 2], [1, 1, 0], [1, 1, 2]])
    # Repeated near neighbors must not disguise the distinct far neighbors.
    assert module.tie_aware_recall(matrix, actual, np.ones(4)) == pytest.approx(3 / 12)


def test_recall_accepts_alternate_neighbors_at_exact_distance_ties() -> None:
    matrix = np.array([[0.0], [-1.0], [1.0]])
    actual = np.array([[2], [0], [0]])
    assert module.tie_aware_recall(matrix, actual, np.ones(3)) == 1.0


@pytest.fixture
def reference() -> dict:
    return {
        "complete": True,
        "configuration": {
            "primary": {"rows": 5000, "features": 32, "epochs": 20},
            "neighbors": 15,
            "components": 2,
            "init": "random",
            "metric": "euclidean",
            "input": "seeded clustered float32 (8 centers), little-endian row-major",
        },
        "quality": {
            "exact": {"cuda_trustworthiness": [0.941, 0.943]},
            "approximate": {"cuda_trustworthiness": [0.942, 0.943]},
        },
    }


def test_reference_accepts_matched_complete_workload(tmp_path: Path, reference: dict) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    assert module.load_quality_reference(path) == reference


@pytest.mark.parametrize(
    ("field", "value"),
    [("epochs", 10), ("rows", 1024), ("features", 16)],
)
def test_reference_rejects_mismatched_workloads(
    tmp_path: Path, reference: dict, field: str, value: int
) -> None:
    reference["configuration"]["primary"][field] = value
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="primary workload"):
        module.load_quality_reference(path)


@pytest.mark.parametrize("bad_samples", [[0.94], [True, 0.94], [float("nan"), 0.94]])
def test_reference_rejects_unusable_samples(
    tmp_path: Path, reference: dict, bad_samples: list
) -> None:
    reference["quality"]["approximate"]["cuda_trustworthiness"] = bad_samples
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="samples"):
        module.load_quality_reference(path)


def test_reference_rejects_incomplete_or_missing_metadata(tmp_path: Path, reference: dict) -> None:
    missing_metric = copy.deepcopy(reference)
    del missing_metric["configuration"]["metric"]
    for bad in [dict(reference, complete=False), {}, missing_metric]:
        path = tmp_path / "reference.json"
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            module.load_quality_reference(path)


# Run when upgrading umap-learn/pynndescent or changing reference-index construction:
# CUSCAN_TEST_REFERENCE_COMPAT=1 uv run --no-sync pytest \
#     python/tests/test_compare_umap_spans.py \
#     -k test_umap_exact_reference_index_transforms_and_missing_index_fails
# This checks upstream assumptions used by our reference helper, not native UMAP correctness.
@pytest.mark.skipif(
    os.environ.get("CUSCAN_TEST_REFERENCE_COMPAT") != "1",
    reason="upstream reference compatibility check; set CUSCAN_TEST_REFERENCE_COMPAT=1 to run",
)
def test_umap_exact_reference_index_transforms_and_missing_index_fails() -> None:
    import umap
    from pynndescent import NNDescent
    from sklearn.neighbors import NearestNeighbors

    if umap.__version__ != "0.5.12":
        pytest.skip("real compatibility fixture requires pinned umap-learn 0.5.12")
    training = np.random.default_rng(11).normal(size=(80, 5)).astype(np.float32)
    queries = np.random.default_rng(12).normal(size=(12, 5)).astype(np.float32)
    estimator = module._stage2_reference_fit(training, "exact", seed=5, epochs=20, neighbors=25)
    assert estimator.n_neighbors == 25
    assert estimator.init == "spectral"
    assert estimator.min_dist == 0.0
    assert estimator.metric == "euclidean"
    assert estimator.random_state == 5
    assert estimator.transform_seed == 5
    assert estimator.force_approximation_algorithm is True
    assert estimator.transform(queries).shape == (12, 2)

    distances, indices = NearestNeighbors(n_neighbors=12).fit(training).kneighbors(training)
    index = NNDescent(training, n_neighbors=12, random_state=5)
    estimator = umap.UMAP(
        n_neighbors=12,
        n_components=2,
        n_epochs=20,
        init="random",
        random_state=5,
        transform_seed=5,
        precomputed_knn=(indices, distances, index),
    ).fit(training)
    assert estimator.n_neighbors == 12
    assert estimator.init == "random"
    assert estimator.random_state == 5
    assert estimator.transform_seed == 5
    # umap-learn enables its transform/index path internally with precomputed_knn.
    assert estimator.force_approximation_algorithm is True
    assert estimator.transform(queries).shape == (12, 2)

    without_index = umap.UMAP(
        n_neighbors=12,
        n_components=2,
        n_epochs=20,
        init="random",
        random_state=5,
        precomputed_knn=(indices, distances, None),
    ).fit(training)
    with pytest.raises(NotImplementedError):
        without_index.transform(queries)
