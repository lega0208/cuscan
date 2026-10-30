#!/usr/bin/env -S uv run --script
"""Stage-1 benchmark suite: umap-learn 0.5.12 (parallel) vs cuscan CUDA standard UMAP.

Run from the repository root with ``uv run --no-sync python scripts/compare_umap_spans.py``.
Build the companion first (build time is excluded from the runtime budget):
``cargo build -p umap --release --features tracing
--example profile_standard_cuda``.

Primary workload: identical clustered float32 5000x32 input, 20 epochs, Euclidean,
15 neighbors, 2 components, random init, explicitly forced Exact and Approximate
search. Each implementation/mode runs one warmup plus N timed fits. Fit durations
include neighbor generation and host embedding readback and exclude input generation,
profiler overhead and quality calculation. Python is unseeded and uses all threads.
Small CUDA regressions: 512x16/30 and 1024x32/100 in both search modes.

Use ``--binary PATH`` to select the benchmark executable. To capture the pre-change
Rust serial implementation as a tracking reference, run
Use the suite command with ``--binary /path/to/pre-change/profile_standard_cuda``
and ``--output artifacts/serial-reference.json``
and pass the complete suite JSON later with
``--quality-reference PATH``. It is distinct from ``--baseline``, which only
provides the existing small/speed comparison gates. Tracking requires at least
two finite CUDA trustworthiness samples per mode and does not replace Q-02.

One global deadline (default 600 seconds) covers warmups, repetitions,
quality/recall diagnostics and
profilers. On expiry the run stops and is reported incomplete (no pass); the
deadline is never adjusted automatically.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import datetime
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_standard_cuda"
PRIMARY = (5000, 32, 20)
SMALL = ((512, 16, 30), (1024, 32, 100))
MODES = ("exact", "approximate")
NEIGHBORS = 15
FIT_WIDTH = NEIGHBORS - 1
TRUST_GAP = 0.025
RECALL_LIMIT = 0.90
SMALL_RATIO_LIMIT = 1.10
SPEED_RATIO_LIMIT = 0.90
STAGE2_QUERY_SEEDS = (43, 44, 45)
STAGE2_MODES = ("exact", "approximate")
STAGE2_NEIGHBORS = 25
STAGE2_TRUST_NEIGHBORS = 15
STAGE2_FIT_RECALL_LIMIT = 0.90
STAGE2_QUERY_RECALL_LIMIT = 0.87
STAGE2_PAIRED_RUNS = 3
STAGE2_GPU_CAP = 7 * 2**30
NSYS_MEMORY_METHOD = (
    "nsys --cuda-memory-usage allocation tracking (excludes CUDA context/module overhead)"
)
RESOURCE_POLL_SECONDS = 0.1
_SUBREAPER_ENABLED = False


def _enable_child_subreaper() -> bool:
    """Ask Linux to reparent orphaned worker descendants here, enabling detached cleanup."""
    global _SUBREAPER_ENABLED
    if _SUBREAPER_ENABLED:
        return True
    if not sys.platform.startswith("linux"):
        return False
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        return False
    _SUBREAPER_ENABLED = True
    return True


def _direct_children() -> set[int]:
    parents, _ = _process_parents()
    return {pid for pid, parent in parents.items() if parent == os.getpid()}


class ResourceLimitExceeded(RuntimeError):
    """An attributable CUDA process exceeded the Stage2 process-memory ceiling."""


def _process_parents() -> tuple[dict[int, int], bool]:
    """Return PID -> PPID from Linux procfs; inaccessible/exited processes are omitted."""
    parents: dict[int, int] = {}
    started = time.monotonic()
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return parents, False
    complete = True
    for entry in entries:
        if len(parents) >= 8192 or time.monotonic() - started >= 0.025:
            complete = False
            break
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
        except (OSError, ValueError, IndexError):
            continue
    return parents, complete


def process_descendants(root_pid: int) -> dict[str, Any]:
    started = time.monotonic()
    parents, complete = _process_parents()
    lineage: dict[int, list[int]] = {root_pid: [root_pid]}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if pid not in lineage and parent in lineage:
                lineage[pid] = [*lineage[parent], pid]
                changed = True
            if len(lineage) >= 4096 or time.monotonic() - started >= 0.05:
                break
        if len(lineage) >= 4096 or time.monotonic() - started >= 0.05:
            break
    complete = complete and len(lineage) < 4096 and time.monotonic() - started < 0.05
    return {
        "pids": sorted(lineage),
        "ancestry": {str(pid): chain for pid, chain in lineage.items()},
        "complete": complete,
    }


def _nvidia_smi(args: list[str], timeout: float = 0.08) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=timeout, check=True
    )


def gpu_device_provenance() -> dict[str, Any]:
    """Query device, driver and CUDA runtime identity without inventing missing values."""
    try:
        result = _nvidia_smi(
            ["--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], timeout=1.0
        )
        devices = [row for row in csv.reader(result.stdout.splitlines()) if row]
        details = _nvidia_smi([], timeout=1.0).stdout
        version = _nvidia_smi(["--version"], timeout=1.0).stdout.splitlines()[0]
        runtime_line = next(
            (line for line in details.splitlines() if "CUDA Version:" in line), None
        )
        compatibility_version = (
            runtime_line.split("CUDA Version:", 1)[1].strip().split()[0] if runtime_line else None
        )
        return {
            "status": "available",
            "api": "nvidia-smi",
            "version": version,
            "devices": [
                {
                    "name": row[0].strip(),
                    "uuid": row[1].strip(),
                    "driver_version": row[2].strip(),
                    "driver_cuda_compatibility_version": compatibility_version,
                    "cuda_runtime_version": None,
                    "cuda_runtime_explanation": (
                        "nvidia-smi does not identify the CUDA runtime loaded by this process."
                    ),
                }
                for row in devices
                if len(row) >= 3
            ],
        }
    except (OSError, subprocess.SubprocessError, IndexError) as error:
        return {
            "status": "unavailable",
            "api": "nvidia-smi",
            "devices": None,
            "cuda_runtime_version": None,
            "cuda_runtime_explanation": (
                "Unavailable because nvidia-smi device provenance could not be queried."
            ),
            "reason": (
                f"nvidia-smi device provenance unavailable (including WSL/no-device cases): {error}"
            ),
        }


class GpuProcessTelemetry:
    """Best-effort 100 ms nvidia-smi process attribution; sampled peaks are not lifetime peaks."""

    def __init__(self) -> None:
        self.evidence: dict[str, Any] = {
            "cap_bytes": STAGE2_GPU_CAP,
            "method": "nvidia-smi --query-compute-apps",
            "telemetry_api": "nvidia-smi",
            "telemetry_version": None,
            "units": "bytes",
            "telemetry_source_memory_unit": "MiB, converted to bytes using 1024^2",
            "sample_interval_seconds": RESOURCE_POLL_SECONDS,
            "readings": [],
            "instantaneous_process_bytes": None,
            "sampled_max_bytes": None,
            "timed_out_samples": 0,
            "process_tree_attribution": ("PID ancestry via /proc; GPU process PID from nvidia-smi"),
            "status": "pending",
            "explanation": None,
        }
        self.evidence["explanation"] = (
            "Telemetry version will be read from supervised device provenance."
        )

    def sample(self, root_pid: int, deadline: Deadline) -> None:
        if self.evidence["status"] == "unavailable":
            return
        ancestry = process_descendants(root_pid)
        if not ancestry["complete"]:
            self.evidence.update(
                status="unavailable",
                explanation=(
                    "/proc process ancestry snapshot unavailable or "
                    "exceeded its bounded scan budget; "
                    "process attribution is incomplete."
                ),
            )
            return
        try:
            remaining = deadline.remaining()
            result = _nvidia_smi(
                [
                    "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ],
                timeout=min(RESOURCE_POLL_SECONDS / 2, remaining),
            )
            rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
            self.evidence["telemetry_version"] = (
                self.evidence["telemetry_version"] or "nvidia-smi (version unavailable)"
            )
            process_rows = []
            total = 0
            known_pids = set(ancestry["pids"])
            for row in rows:
                if len(row) < 3:
                    continue
                try:
                    pid, mib = int(row[0].strip()), int(row[1].strip())
                except ValueError:
                    continue
                if pid in known_pids:
                    used = mib * 1024 * 1024
                    total += used
                    process_rows.append(
                        {
                            "pid": pid,
                            "ancestry": ancestry["ancestry"].get(str(pid), [pid]),
                            "gpu_uuid": row[2].strip(),
                            "used_bytes": used,
                        }
                    )
            reading = {
                "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "monotonic_seconds": time.monotonic(),
                "root_pid": root_pid,
                "telemetry_api": self.evidence["telemetry_api"],
                "telemetry_version": self.evidence["telemetry_version"],
                "tracked_pids": ancestry["pids"],
                "processes": process_rows,
                "attributable_process_bytes": total if process_rows else None,
                "units": "bytes",
            }
            self.evidence["readings"].append(reading)
            self.evidence["instantaneous_process_bytes"] = total if process_rows else None
            if process_rows:
                previous = self.evidence["sampled_max_bytes"] or 0
                self.evidence["sampled_max_bytes"] = max(previous, total)
                self.evidence["status"] = "sampled_attributable"
                self.evidence["explanation"] = (
                    "Observed nvidia-smi process samples; transient peaks "
                    "between samples are not covered."
                )
                if total > self.evidence["cap_bytes"]:
                    self.evidence["status"] = "violation"
                    self.evidence["violation"] = {
                        "timestamp_utc": reading["timestamp_utc"],
                        "observed_bytes": total,
                        "cap_bytes": self.evidence["cap_bytes"],
                        "processes": process_rows,
                    }
                    raise ResourceLimitExceeded(
                        f"attributable CUDA process-tree GPU memory {total} "
                        f"exceeds {self.evidence['cap_bytes']} bytes"
                    )
            elif not self.evidence["readings"][:-1]:
                self.evidence["status"] = "no_attributable_process_seen"
                self.evidence["explanation"] = (
                    "nvidia-smi answered but no CUDA process PID was "
                    "attributable to the supervised worker tree."
                )
        except ResourceLimitExceeded:
            raise
        except subprocess.TimeoutExpired:
            # A slow query skips one sample; later samples still arm the over-cap kill.
            self.evidence["timed_out_samples"] += 1
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            self.evidence["status"] = "unavailable"
            self.evidence["explanation"] = f"nvidia-smi process telemetry unavailable: {error}"


def stage2_cuda_command(
    binary: Path,
    input_path: Path,
    rows: int,
    features: int,
    epochs: int,
    output_dir: Path,
    mode: str,
    seed: int,
    warmups: int,
    runs: int,
    neighbors: int,
    init: str,
    queries: list[tuple[Path, int]],
) -> list[str]:
    """Construct the Rust example's positional INPUT ROWS FEATURES EPOCHS OUT_DIR CLI."""
    command = [
        str(binary),
        str(input_path),
        str(rows),
        str(features),
        str(epochs),
        str(output_dir),
        "--search",
        mode,
        "--warmups",
        str(warmups),
        "--runs",
        str(runs),
        "--neighbors",
        str(neighbors),
        "--min-dist",
        "0.0",
        "--init",
        init,
        "--seed",
        str(seed),
    ]
    for path, query_rows in queries:
        command.extend(["--query", str(path), "--query-rows", str(query_rows)])
    return command


def stage2_aggregate(
    records: list[dict[str, Any]], references: dict[str, list[dict[str, Any]]], paired_runs: int
) -> dict[str, Any]:
    """Aggregate complete matched quality and timing evidence; timing never gates quality."""
    quality: dict[str, Any] = {}
    timings: dict[str, Any] = {}
    diffs = []
    warnings = []
    full_coverage = True
    for mode in STAGE2_MODES:
        mode_rows = [row for row in records if row["mode"] == mode]
        refs = references[mode]
        fit_mean = statistics.fmean(row["fit_trustworthiness"] for row in refs)
        batch_means = [
            statistics.fmean(row["transform"][batch]["trustworthiness"] for row in refs)
            for batch in range(3)
        ]
        for row in mode_rows:
            row["reference_fit_trustworthiness_mean"] = fit_mean
            row["fit_trustworthiness_gap_from_reference_mean"] = abs(
                row["fit_trustworthiness"] - fit_mean
            )
            for batch_row in row["transform"]:
                batch_mean = batch_means[batch_row["batch"]]
                batch_row["reference_trustworthiness_mean"] = batch_mean
                batch_row["trustworthiness_gap_from_reference_mean"] = abs(
                    batch_row["trustworthiness"] - batch_mean
                )
            row["quality_pass"] = (
                row["fit_trustworthiness_gap_from_reference_mean"] <= TRUST_GAP
                and row["fit_recall"] >= STAGE2_FIT_RECALL_LIMIT
                and len(row["transform"]) == 3
                and all(
                    batch_row["trustworthiness_gap_from_reference_mean"] <= TRUST_GAP
                    for batch_row in row["transform"]
                )
            )
            for batch_row in row["transform"]:
                if batch_row["recall"] < STAGE2_QUERY_RECALL_LIMIT:
                    warnings.append(
                        {
                            "kind": "query_recall_below_limit",
                            "mode": mode,
                            "pair": row["pair"],
                            "side": row["side"],
                            "seed": batch_row["seed"],
                            "recall": batch_row["recall"],
                            "limit": STAGE2_QUERY_RECALL_LIMIT,
                        }
                    )
        expected = {(pair, side) for pair in range(paired_runs) for side in ("A", "B")}
        present = {(row["pair"], row["side"]) for row in mode_rows}
        full_coverage &= present == expected and all(
            len(row["transform"]) == 3 for row in mode_rows
        )
        quality[mode] = {
            "reference_fit_trustworthiness_mean": fit_mean,
            "reference_transform_trustworthiness_means": batch_means,
            "all_runs_pass": len(mode_rows) == paired_runs * 2
            and all(row["quality_pass"] for row in mode_rows),
            "query_recall_warning": any(item["mode"] == mode for item in warnings),
            "runs": [
                {
                    "pair": row["pair"],
                    "side": row["side"],
                    "reference": row["reference"],
                    "fit_trustworthiness": row["fit_trustworthiness"],
                    "fit_recall": row["fit_recall"],
                    "transform": row["transform"],
                    "pass": row["quality_pass"],
                }
                for row in mode_rows
            ],
        }
        fit = [float(row["fit_seconds"][0]) for row in mode_rows]
        timings[mode] = {
            "fit_seconds": fit,
            "fit_median_seconds": statistics.median(fit) if fit else None,
            "transform_by_batch_seconds": [
                [
                    row["transform"][batch]["seconds"]
                    for row in mode_rows
                    if len(row["transform"]) > batch
                ]
                for batch in range(3)
            ],
            "transform_total_seconds": [row["transform_total_seconds"] for row in mode_rows],
            "warmup_policy": "one warmup for each mode/side before timed pairs",
        }
        for pair in range(paired_runs):
            a = next((row for row in mode_rows if row["pair"] == pair and row["side"] == "A"), None)
            b = next((row for row in mode_rows if row["pair"] == pair and row["side"] == "B"), None)
            if a is None or b is None:
                full_coverage = False
            else:
                diffs.append(
                    {
                        "pair": pair,
                        "mode": mode,
                        "side_a_minus_b_fit_seconds": float(a["fit_seconds"][0])
                        - float(b["fit_seconds"][0]),
                    }
                )
    return {
        "quality": quality,
        "timings": timings,
        "paired_timing_differences": diffs,
        "full_coverage": full_coverage,
        "speed_gates": [],
        "warnings": warnings,
    }


def nsys_memory_incomplete(cap_bytes: int, explanation: str) -> dict[str, Any]:
    """Nsight memory evidence that cannot establish a lifetime peak."""
    return {
        "nsys_peak_live_allocation_bytes": None,
        "allocation_event_count": 0,
        "free_event_count": 0,
        "unknown_event_count": 0,
        "per_pid_device_peaks": [],
        "method": NSYS_MEMORY_METHOD,
        "cap_bytes": cap_bytes,
        "pass": False,
        "status": "incomplete",
        "explanation": explanation,
    }


def nsys_cuda_memory_evidence(sqlite_path: Path, cap_bytes: int) -> dict[str, Any]:
    """Compute CUDA allocation lifetime peaks from Nsight Systems' exported event table.

    Operation type 0 is an allocation and 1 a free. Any other operation type makes the
    evidence incomplete, because its effect on live bytes is unknown.
    """
    try:
        with contextlib.closing(sqlite3.connect(sqlite_path)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(CUDA_GPU_MEMORY_USAGE_EVENTS)")
            }
            required = {"start", "globalPid", "deviceId", "bytes", "memoryOperationType"}
            if not required.issubset(columns):
                return nsys_memory_incomplete(
                    cap_bytes, "CUDA_GPU_MEMORY_USAGE_EVENTS table or required columns missing."
                )
            rows = connection.execute(
                "SELECT globalPid, deviceId, bytes, memoryOperationType "
                "FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
            ).fetchall()
    except (OSError, sqlite3.Error) as error:
        return nsys_memory_incomplete(
            cap_bytes, f"Nsight Systems SQLite memory evidence unavailable: {error}"
        )
    live: dict[tuple[int, int], int] = {}
    peaks: dict[tuple[int, int], int] = {}
    counts = {0: 0, 1: 0}
    unknown = 0
    for pid, device, size, operation in rows:
        key = (int(pid), int(device))
        if operation == 0:
            live[key] = live.get(key, 0) + int(size)
        elif operation == 1:
            live[key] = live.get(key, 0) - int(size)
        else:
            unknown += 1
            continue
        counts[operation] += 1
        peaks[key] = max(peaks.get(key, 0), live[key])
    evidence = nsys_memory_incomplete(cap_bytes, "")
    evidence.update(
        allocation_event_count=counts[0], free_event_count=counts[1], unknown_event_count=unknown
    )
    if counts[0] == 0:
        evidence["explanation"] = "No CUDA allocation events were exported."
        return evidence
    if unknown:
        evidence["explanation"] = f"{unknown} memory events have an unknown operation type."
        return evidence
    del evidence["explanation"]
    evidence["per_pid_device_peaks"] = [
        {"globalPid": pid, "deviceId": device, "peak_live_allocation_bytes": peak}
        for (pid, device), peak in sorted(peaks.items())
    ]
    peak = max(peaks.values())
    evidence["nsys_peak_live_allocation_bytes"] = peak
    evidence["pass"] = peak <= cap_bytes
    evidence["status"] = "pass" if evidence["pass"] else "fail"
    return evidence


def finalize_resource_evidence(evidence: dict[str, Any], telemetry: GpuProcessTelemetry) -> str:
    """Attach supplementary telemetry and derive the resource status from Nsight evidence.

    Every mode must have Nsight memory evidence that passes; any failing mode fails.
    """
    evidence["supplementary_nvidia_smi"] = telemetry.evidence
    modes = [evidence["nsys_modes"].get(mode, {}) for mode in STAGE2_MODES]
    peaks = [item.get("nsys_peak_live_allocation_bytes") for item in modes]
    if all(peak is not None for peak in peaks):
        evidence["authoritative_lifetime_peak_bytes"] = max(peaks)
    if any(item.get("status") == "fail" for item in modes):
        return "fail"
    if all(item.get("status") == "pass" for item in modes):
        return "pass"
    return "incomplete"


class DeadlineExceeded(TimeoutError):
    """The global benchmark deadline expired."""


class Deadline:
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.ends_at = time.monotonic() + seconds

    def remaining(self) -> float:
        value = self.ends_at - time.monotonic()
        if value <= 0:
            raise DeadlineExceeded("global runtime deadline expired")
        return value


def run_with_deadline(
    command: list[str], deadline: Deadline, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
            timeout=deadline.remaining(),
        )
    except subprocess.TimeoutExpired as error:
        raise DeadlineExceeded("global runtime deadline expired during worker execution") from error


def run_supervised(
    command: list[str],
    deadline: Deadline,
    *,
    env: dict[str, str] | None = None,
    telemetry: GpuProcessTelemetry | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a killable process group, so timeout cleanup includes worker descendants."""
    subreaper = _enable_child_subreaper()
    if sys.platform.startswith("linux") and not subreaper:
        raise OSError("Linux child-subreaper setup failed; detached worker cleanup is unavailable")
    baseline_children = _direct_children() if subreaper else set()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    seen_descendants: set[int] = set()
    adopted_roots: set[int] = set()

    def refresh_descendants() -> None:
        snapshot = process_descendants(process.pid)
        seen_descendants.update(snapshot["pids"])
        parents, _ = _process_parents()
        if subreaper:
            adopted = {
                pid
                for pid, parent in parents.items()
                if parent == os.getpid() and pid not in baseline_children and pid != process.pid
            }
            adopted_roots.update(adopted)
            seen_descendants.update(adopted)
        for pid in tuple(adopted_roots):
            tree = process_descendants(pid)
            seen_descendants.update(tree["pids"])

    def kill_tree(*, graceful: bool) -> None:
        if graceful:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        for pid in tuple(seen_descendants):
            if pid == process.pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.wait()
        if subreaper:
            parents, _ = _process_parents()
            seen_descendants.update(
                pid
                for pid, parent in parents.items()
                if parent == os.getpid() and pid not in baseline_children and pid != process.pid
            )
            for pid in tuple(seen_descendants):
                if pid == process.pid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            reap_until = time.monotonic() + 0.5
            for pid in tuple(seen_descendants):
                if pid == process.pid:
                    continue
                while True:
                    try:
                        waited, _ = os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        break
                    if waited == pid:
                        break
                    if time.monotonic() >= reap_until:
                        break
                    time.sleep(0.01)

    stdout = stderr = ""
    try:
        while True:
            refresh_descendants()
            if process.poll() is not None:
                kill_tree(graceful=False)
                stdout, stderr = process.communicate(timeout=0.2)
                break
            if telemetry is not None:
                telemetry.sample(process.pid, deadline)
            remaining = deadline.remaining()
            polling_wait = (
                RESOURCE_POLL_SECONDS / 2 if telemetry is not None else RESOURCE_POLL_SECONDS
            )
            try:
                stdout, stderr = process.communicate(timeout=min(polling_wait, remaining))
                break
            except subprocess.TimeoutExpired as timeout_error:
                stdout = timeout_error.stdout or stdout
                stderr = timeout_error.stderr or stderr
                continue
    except BaseException as error:
        kill_tree(graceful=isinstance(error, DeadlineExceeded))
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        if isinstance(error, ResourceLimitExceeded):
            raise
        if not isinstance(error, DeadlineExceeded):
            raise
        raise DeadlineExceeded(
            "global runtime deadline expired during supervised worker execution"
        ) from error
    refresh_descendants()
    kill_tree(graceful=True)
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, stdout, stderr)
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage2_trustworthiness(matrix: Any, embedding: Any) -> float:
    from sklearn.manifold import trustworthiness

    return float(trustworthiness(matrix, embedding, n_neighbors=STAGE2_TRUST_NEIGHBORS))


def stage2_recall(matrix: Any, actual: Any, neighbors: int) -> float:
    """Recall with deterministic ID tie-breaking, unique IDs and inclusive radius ties."""
    import numpy as np
    from sklearn.metrics import pairwise_distances_chunked

    actual = np.asarray(actual)
    if actual.ndim != 2 or actual.shape[0] != matrix.shape[0]:
        raise ValueError("neighbor IDs must cover every input row")
    if np.any((actual < -1) | (actual >= matrix.shape[0])):
        raise ValueError("neighbor output contains invalid IDs")
    exact_k = min(neighbors, matrix.shape[0] - 1)
    hits = 0
    offset = 0
    for block in pairwise_distances_chunked(matrix, matrix, metric="euclidean", working_memory=64):
        local_rows = np.arange(block.shape[0])
        block[local_rows, offset + local_rows] = np.inf
        cutoffs = np.partition(block, exact_k - 1, axis=1)[:, exact_k - 1]
        for local_row, ids in enumerate(actual[offset : offset + block.shape[0]]):
            seen: set[int] = set()
            for index in ids:
                index = int(index)
                if index >= 0 and index != offset + local_row and index not in seen:
                    seen.add(index)
                    hits += block[local_row, index] <= cutoffs[local_row]
        offset += block.shape[0]
    denominator = matrix.shape[0] * actual.shape[1]
    return float(hits / denominator) if denominator else 0.0


def stage2_query_recall(queries: Any, training: Any, actual: Any, neighbors: int) -> float:
    """Query-to-training recall; query rows are not training/self IDs."""
    import numpy as np
    from sklearn.metrics import pairwise_distances_chunked

    actual = np.asarray(actual)
    if actual.ndim != 2 or actual.shape[0] != queries.shape[0]:
        raise ValueError("query neighbor IDs must cover every query row")
    if np.any((actual < -1) | (actual >= training.shape[0])):
        raise ValueError("query graph contains invalid training IDs")
    k = min(neighbors, training.shape[0])
    hits = 0
    offset = 0
    for block in pairwise_distances_chunked(
        queries, training, metric="euclidean", working_memory=64
    ):
        cutoffs = np.partition(block, k - 1, axis=1)[:, k - 1]
        for local_row, ids in enumerate(actual[offset : offset + block.shape[0]]):
            seen: set[int] = set()
            for index in ids:
                index = int(index)
                if index >= 0 and index not in seen:
                    seen.add(index)
                    hits += block[local_row, index] <= cutoffs[local_row]
        offset += block.shape[0]
    return float(hits / (actual.shape[0] * actual.shape[1]))


def reverse_graph_diagnostics(graph: Any, rows: int) -> dict[str, Any]:
    """Describe observed reverse degrees and explicit memory estimates for a CSR-like index."""
    import numpy as np

    graph = np.asarray(graph)
    if graph.ndim != 2:
        raise ValueError("fit graph must be a matrix")
    if np.any((graph < -1) | (graph >= rows)):
        raise ValueError("fit graph contains invalid neighbor IDs")
    ids = graph[graph >= 0].astype(np.int64, copy=False)
    degrees = np.bincount(ids, minlength=rows).astype(np.uint64, copy=False)
    edge_count = int(ids.size)
    max_u32 = (1 << 32) - 1
    max_i32 = (1 << 31) - 1
    max_u64 = (1 << 64) - 1
    representability = {
        "node_count_u32": rows <= max_u32,
        "edge_count_u32": edge_count <= max_u32,
        "node_ids_i32": rows <= max_i32 + 1,
        "prefix_sum_u64": edge_count <= max_u64,
    }
    counts_bytes = rows * 4
    cursors_bytes = rows * 4
    prefix_bytes = (rows + 1) * 8
    reverse_ids_bytes = edge_count * 4
    workspace = counts_bytes + cursors_bytes + prefix_bytes + reverse_ids_bytes
    workspace_representable = all(
        value <= max_u64
        for value in (counts_bytes, cursors_bytes, prefix_bytes, reverse_ids_bytes, workspace)
    )
    mean = float(degrees.mean()) if rows else 0.0
    std = float(degrees.std()) if rows else 0.0
    return {
        "evidence_kind": (
            "observed graph degrees; construction sizes are estimates, not measured allocations"
        ),
        "graph_shape": list(graph.shape),
        "observed_valid_edge_count": edge_count,
        "reverse_indegree": {
            "mean": mean,
            "stddev": std,
            "p50": float(np.percentile(degrees, 50)),
            "p90": float(np.percentile(degrees, 90)),
            "p95": float(np.percentile(degrees, 95)),
            "p99": float(np.percentile(degrees, 99)),
            "max": int(degrees.max(initial=0)),
            "hub_thresholds": {
                "greater_than_2x_mean": 2.0 * mean,
                "count_greater_than_2x_mean": int(np.count_nonzero(degrees > 2.0 * mean)),
                "greater_than_mean_plus_3_stddev": mean + 3.0 * std,
                "count_greater_than_mean_plus_3_stddev": int(
                    np.count_nonzero(degrees > mean + 3.0 * std)
                ),
            },
        },
        "representability": representability | {"workspace_bytes_u64": workspace_representable},
        "reverse_index_workspace_estimate_bytes": {
            "counts_u32": counts_bytes,
            "cursors_u32": cursors_bytes,
            "prefix_u64": prefix_bytes,
            "reverse_ids_i32": reverse_ids_bytes,
            "concurrent_peak_estimate": workspace if workspace_representable else None,
            "assumptions": (
                "counts, mutable cursors, prefix offsets, and all reverse IDs "
                "coexist during construction"
            ),
            "source": ("computed from observed exported fit graph; not allocator telemetry"),
        },
    }


def stage2_nonself_padded(indices: Any, width: int) -> Any:
    """Validate raw neighbor IDs, drop self IDs and pad short rows with -1 sentinels."""
    import numpy as np

    indices = np.asarray(indices)
    if indices.ndim != 2:
        raise ValueError("fit graph must be a two-dimensional ID matrix")
    rows, _ = indices.shape
    if np.any((indices < -1) | (indices >= rows)):
        raise ValueError("fit graph contains invalid raw neighbor IDs")
    result = np.full((rows, width), -1, dtype=np.int64)
    for row in range(rows):
        selected = [int(value) for value in indices[row] if value >= 0 and value != row]
        count = min(len(selected), width)
        result[row, :count] = selected[:count]
    return result


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stage2_reference_fit(matrix: Any, mode: str, seed: int, epochs: int, neighbors: int) -> Any:
    import umap
    from pynndescent import NNDescent
    from sklearn.neighbors import NearestNeighbors

    if umap.__version__ != "0.5.12":
        raise ValueError(f"Stage2 requires umap-learn 0.5.12, got {umap.__version__}")
    n_neighbors = neighbors
    index = NNDescent(matrix, n_neighbors=n_neighbors, metric="euclidean", random_state=seed)
    if mode == "exact":
        distances, indices = (
            NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(matrix).kneighbors(matrix)
        )
        precomputed = (indices, distances, index)
    else:
        precomputed = (None, None, None)
    estimator = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=2,
        n_epochs=epochs,
        init="spectral",
        min_dist=0.0,
        random_state=seed,
        transform_seed=seed,
        metric="euclidean",
        n_jobs=-1,
        force_approximation_algorithm=mode == "approximate",
        precomputed_knn=precomputed,
        verbose=False,
    )
    estimator.fit(matrix)
    if estimator._small_data:
        raise ValueError("reference did not use indexed transform search")
    if (
        estimator.n_neighbors != n_neighbors
        or estimator.init != "spectral"
        or estimator.min_dist != 0.0
        or estimator.metric != "euclidean"
        or estimator.random_state != seed
        or estimator.transform_seed != seed
    ):
        raise ValueError(
            "umap-learn reference fitted settings differ from the requested Stage2 contract"
        )
    return estimator


def _stage2_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """One isolated reference quality run (fit and three held-out transforms)."""
    import numpy as np

    if payload.get("kind") == "generate":
        train_rng = np.random.default_rng(42)
        centers = train_rng.normal(0.0, 5.0, size=(8, payload["features"])).astype(np.float32)
        labels = np.arange(payload["rows"]) % 8
        train_rng.shuffle(labels)
        train = centers[labels] + train_rng.normal(
            0.0, 0.35, (payload["rows"], payload["features"])
        ).astype(np.float32)
        train.astype("<f4", copy=False).tofile(payload["train"])
        queries = []
        for seed, path in zip(payload["seeds"], payload["query_paths"], strict=True):
            query_rng = np.random.default_rng(seed)
            labels = np.arange(payload["query_rows"]) % 8
            query_rng.shuffle(labels)
            query = centers[labels] + query_rng.normal(
                0.0, 0.35, (payload["query_rows"], payload["features"])
            ).astype(np.float32)
            query.astype("<f4", copy=False).tofile(path)
            queries.append(
                {
                    "path": path,
                    "sha256": sha256_file(Path(path)),
                    "label_counts": np.bincount(labels, minlength=8).tolist(),
                    "labels_sha256": hashlib.sha256(labels.astype("<i8").tobytes()).hexdigest(),
                }
            )
        return {
            "train_sha256": sha256_file(Path(payload["train"])),
            "train_label_counts": np.bincount(np.arange(payload["rows"]) % 8, minlength=8).tolist(),
            "queries": queries,
            "generation": {
                "train_seed": 42,
                "center_scale": 5.0,
                "noise": 0.35,
                "query_seeds": payload["seeds"],
                "query_centers": "shared train centers",
                "query_labels": "balanced and shuffled per seed",
            },
        }
    if payload.get("kind") == "provenance":
        import importlib.metadata
        import shutil

        names = ("numpy", "scikit-learn", "umap-learn", "pynndescent", "numba", "scipy")
        root = Path(payload["root"])
        harness = [Path(item) for item in payload["harness_paths"]]
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        rustc = subprocess.run(
            ["rustc", "--version"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", *map(str, harness)],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        diff_path = Path(payload["artifact_directory"]) / "harness.diff"
        diff_path.write_text(diff, encoding="utf-8")
        nvcc = shutil.which("nvcc")
        toolkit = None
        if nvcc:
            toolkit = subprocess.run(
                [nvcc, "--version"], cwd=root, check=True, capture_output=True, text=True
            ).stdout
        return {
            "dependencies": {name: importlib.metadata.version(name) for name in names},
            "python": platform.python_version(),
            "platform": platform.platform(),
            "gpu": gpu_device_provenance(),
            "source": {
                "working_head": head,
                "rustc": rustc,
                "binary_sha256": sha256_file(Path(payload["binary"])),
                "script_sha256": sha256_file(root / "scripts/compare_umap_spans.py"),
                "lockfile_sha256": sha256_file(root / "Cargo.lock"),
                "harness_test_sha256": {
                    str(path): sha256_file(root / path)
                    for path in harness
                    if (root / path).is_file()
                },
                "harness_diff_path": str(diff_path),
                "harness_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                "cuda_toolkit_nvcc_version": toolkit,
            },
        }

    if payload.get("kind") == "cuda-quality":
        train = np.fromfile(payload["train"], dtype="<f4").reshape(
            payload["rows"], payload["features"]
        )
        fit_embedding = np.fromfile(payload["fit_embedding"], dtype="<f4").reshape(
            payload["rows"], 2
        )
        raw_fit_graph = np.fromfile(payload["fit_graph"], dtype="<i4").reshape(payload["rows"], -1)
        if np.any((raw_fit_graph < -1) | (raw_fit_graph >= payload["rows"])):
            raise ValueError("fit graph contains invalid raw neighbor IDs")
        fit_width = payload["neighbors"] - 1
        fit_graph = stage2_nonself_padded(raw_fit_graph, fit_width)
        reverse_diagnostics = reverse_graph_diagnostics(fit_graph, payload["rows"])
        if payload.get("diagnostic_path"):
            atomic_write_json(Path(payload["diagnostic_path"]), reverse_diagnostics)
        fit_scores = {
            "trustworthiness": stage2_trustworthiness(train, fit_embedding),
            "recall": stage2_recall(train, fit_graph, fit_width),
        }
        if not all(math.isfinite(value) for value in fit_scores.values()):
            raise ValueError("fit quality evaluator produced a non-finite result")
        transform_scores = []
        for batch, query_path in enumerate(payload["queries"]):
            query = np.fromfile(query_path, dtype="<f4").reshape(
                payload["query_rows"], payload["features"]
            )
            embedding = np.fromfile(payload["transform_embeddings"][batch], dtype="<f4").reshape(
                payload["query_rows"], 2
            )
            graph = np.fromfile(payload["query_graphs"][batch], dtype="<i4").reshape(
                payload["query_rows"], payload["graph_width"]
            )
            scores = {
                "trustworthiness": stage2_trustworthiness(query, embedding),
                "recall": stage2_query_recall(query, train, graph, payload["neighbors"]),
            }
            if not all(math.isfinite(value) for value in scores.values()):
                raise ValueError(
                    f"query-batch {batch} quality evaluator produced a non-finite result"
                )
            transform_scores.append(scores)
        return {
            "fit": fit_scores,
            "transform": transform_scores,
            "reverse_graph_diagnostics": reverse_diagnostics,
        }

    train = np.fromfile(payload["train"], dtype="<f4").reshape(payload["rows"], payload["features"])
    queries = [
        np.fromfile(path, dtype="<f4").reshape(payload["query_rows"], payload["features"])
        for path in payload["queries"]
    ]
    mode, seed = payload["mode"], payload["seed"]
    model = _stage2_reference_fit(train, mode, seed, payload["epochs"], payload["neighbors"])
    fit = np.asarray(model.embedding_, dtype=np.float32)
    fit_neighbors = model._knn_indices
    result: dict[str, Any] = {
        "fit_trustworthiness": stage2_trustworthiness(train, fit),
        "fit_recall": stage2_recall(
            train, nonself(fit_neighbors, payload["neighbors"] - 1), payload["neighbors"] - 1
        ),
        "transform": [],
        "reference": {
            "package": "umap-learn",
            "version": "0.5.12",
            "index": "pynndescent.NNDescent",
            "index_version": __import__("pynndescent").__version__,
            "mode": mode,
            "random_state": seed,
            "transform_seed": seed,
            "n_neighbors": payload["neighbors"],
            "n_components": 2,
            "n_epochs": payload["epochs"],
            "init": "spectral",
            "min_dist": 0.0,
            "metric": "euclidean",
            "n_jobs": -1,
            "requested_force_approximation_algorithm": mode == "approximate",
            "fit_neighbor_source": "sklearn_exact_precomputed"
            if mode == "exact"
            else "umap_approximate_search",
            "post_fit_force_approximation_algorithm": bool(model.force_approximation_algorithm),
            "exact_mode_fit_knn": mode == "exact",
            "trustworthiness_n_neighbors": STAGE2_TRUST_NEIGHBORS,
            "trustworthiness_population": "all fit rows or current query batch only",
            "recall_truth": "euclidean input-space exact kth-distance cutoff",
        },
    }
    if not math.isfinite(result["fit_trustworthiness"]) or not math.isfinite(result["fit_recall"]):
        raise ValueError("pinned reference fit quality evaluator produced a non-finite result")
    for query in queries:
        embedding = np.asarray(model.transform(query), dtype=np.float32)
        score = stage2_trustworthiness(query, embedding)
        if not math.isfinite(score):
            raise ValueError(
                "pinned reference transform quality evaluator produced a non-finite result"
            )
        result["transform"].append({"trustworthiness": score})
    return result


def _stage2_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Bounded Stage2 standard UMAP baseline campaign")
    parser.add_argument("--suite", choices=["stage2"], required=True)
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--neighbors", type=int, default=STAGE2_NEIGHBORS)
    parser.add_argument("--init", choices=["spectral"], default="spectral")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--query-rows", type=int, default=1024)
    parser.add_argument("--query-seeds", default=",".join(map(str, STAGE2_QUERY_SEEDS)))
    parser.add_argument(
        "--paired-runs",
        type=int,
        default=STAGE2_PAIRED_RUNS,
        help="matched A/B baseline repetitions per mode",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=900.0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--_stage2_worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._stage2_worker:
        print(json.dumps(_stage2_worker(json.loads(args._stage2_worker))))
        return 0
    if args.output is None:
        parser.error("--output is required for a Stage2 campaign")
    if (
        args.rows <= 2 * STAGE2_TRUST_NEIGHBORS
        or args.features < 1
        or args.neighbors != STAGE2_NEIGHBORS
        or args.query_rows <= 2 * STAGE2_TRUST_NEIGHBORS
        or args.epochs < 1
    ):
        parser.error(
            f"Stage2 requires rows/query-rows>{2 * STAGE2_TRUST_NEIGHBORS}, features>=1, epochs>=1 "
            f"and neighbors={STAGE2_NEIGHBORS}"
        )
    if args.paired_runs < STAGE2_PAIRED_RUNS or args.warmups != 1 or args.deadline_seconds <= 0:
        parser.error(
            f"Stage2 requires at least {STAGE2_PAIRED_RUNS} paired runs, a positive deadline, "
            "and exactly one warmup"
        )
    seeds = tuple(int(item) for item in args.query_seeds.split(","))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        parser.error("Stage2 requires exactly three distinct query seeds")
    binary = args.binary.resolve()
    if not binary.is_file():
        parser.error(f"CUDA executable not found: {binary}")
    deadline = Deadline(args.deadline_seconds)
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_artifacts = Path(
        tempfile.mkdtemp(prefix=f"{args.output.stem}-stage2-", dir=args.output.parent.resolve())
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    output: dict[str, Any] = {
        "suite": "stage2",
        "complete": False,
        "configuration": configuration
        | {
            "binary": str(binary),
            "query_seeds": list(seeds),
            "fit_recall_threshold": STAGE2_FIT_RECALL_LIMIT,
            "query_recall_threshold": STAGE2_QUERY_RECALL_LIMIT,
            "trustworthiness_gap_threshold": TRUST_GAP,
            "trustworthiness_n_neighbors": STAGE2_TRUST_NEIGHBORS,
        },
        "pairs": [],
        "warnings": [],
        "reverse_graph_diagnostics": [],
        "quality": {},
        "resource_status": "incomplete",
        "resource_evidence": {
            "cap_bytes": STAGE2_GPU_CAP,
            "units": "bytes",
            "method": NSYS_MEMORY_METHOD,
            "authoritative_lifetime_peak_bytes": None,
            "nsys_modes": {},
            "supplementary_nvidia_smi": None,
            "explanation": (
                "Nsight Systems CUDA allocation tracking is required for an authoritative "
                "lifetime peak; nvidia-smi polling is supplementary only."
            ),
        },
        "provenance": {
            "binary": str(binary),
            "binary_sha256": None,
            "input_digests": {},
            "source_identity": None,
        },
        "coverage": {
            "modes": list(STAGE2_MODES),
            "pairs_required_per_mode": args.paired_runs,
            "baseline_repetitions_required_per_mode": 2 * args.paired_runs,
            "completed_pairs": 0,
            "completed_baseline_repetitions": 0,
            "quality_runs": 0,
            "profiles": "not requested",
        },
        "raw_artifact_directory": str(raw_artifacts),
    }

    def emit() -> None:
        output["elapsed_seconds"] = time.monotonic() - started
        atomic_write_json(args.output, output)

    try:
        with contextlib.nullcontext(str(raw_artifacts)) as temp_name:
            temp = Path(temp_name)
            telemetry = GpuProcessTelemetry()
            output["resource_evidence"]["supplementary_nvidia_smi"] = telemetry.evidence
            train_path = temp / "train.f32le"
            query_paths = [temp / f"query-{seed}.f32le" for seed in seeds]
            generated = last_json(
                run_supervised(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--suite",
                        "stage2",
                        "--_stage2_worker",
                        json.dumps(
                            {
                                "kind": "generate",
                                "rows": args.rows,
                                "features": args.features,
                                "query_rows": args.query_rows,
                                "seeds": list(seeds),
                                "train": str(train_path),
                                "query_paths": [str(path) for path in query_paths],
                            }
                        ),
                    ],
                    deadline,
                ).stdout,
                require_fit_seconds=False,
            )
            output["provenance"]["input_digests"] = {
                train_path.name: generated["train_sha256"],
                **{Path(item["path"]).name: item["sha256"] for item in generated["queries"]},
            }
            output["provenance"]["generation"] = generated["generation"] | {
                "train_label_counts": generated["train_label_counts"],
                "query_label_counts": {
                    Path(item["path"]).name: item["label_counts"] for item in generated["queries"]
                },
                "query_label_digests": {
                    Path(item["path"]).name: item["labels_sha256"] for item in generated["queries"]
                },
            }
            harness_paths = [
                Path("crates/umap/examples/profile_standard_cuda.rs"),
                Path("crates/umap/tests/test_standard_ann.rs"),
                Path("crates/umap/tests/test_standard_graph.rs"),
                Path("crates/umap/tests/test_standard_quality.rs"),
                Path("python/tests/test_compare_umap_spans.py"),
            ]
            manifest_payload = {
                "kind": "provenance",
                "root": str(ROOT),
                "harness_paths": [str(path) for path in harness_paths],
                "artifact_directory": str(raw_artifacts),
                "binary": str(binary),
            }
            manifest = last_json(
                run_supervised(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--suite",
                        "stage2",
                        "--_stage2_worker",
                        json.dumps(manifest_payload),
                    ],
                    deadline,
                ).stdout,
                require_fit_seconds=False,
            )
            output["provenance"].update(manifest)
            telemetry.evidence["telemetry_version"] = manifest.get("gpu", {}).get("version")
            if telemetry.evidence["telemetry_version"] is None:
                telemetry.evidence["explanation"] = (
                    "nvidia-smi version unavailable from supervised GPU provenance."
                )
            output["provenance"]["source_identity"] = manifest["source"] | {
                "features": ["cuda", "tracing"],
            }
            output["provenance"]["binary_sha256"] = manifest["source"]["binary_sha256"]
            reference_scores: dict[str, list[dict[str, Any]]] = {mode: [] for mode in STAGE2_MODES}
            queries = [(path, args.query_rows) for path in query_paths]
            # One isolated warmup for each mode/side; measured repetitions use warmups=0.
            for mode in STAGE2_MODES:
                for side in ("A", "B"):
                    warm_dir = temp / f"warmup-{mode}-{side}"
                    warm_dir.mkdir()
                    warm_command = stage2_cuda_command(
                        binary,
                        train_path,
                        args.rows,
                        args.features,
                        args.epochs,
                        warm_dir,
                        mode,
                        42,
                        args.warmups,
                        1,
                        args.neighbors,
                        args.init,
                        queries,
                    )
                    run_supervised(warm_command, deadline, telemetry=telemetry)
                    output["resource_evidence"]["supplementary_nvidia_smi"] = telemetry.evidence
            for pair in range(args.paired_runs):
                seed = 42 + pair
                mode_order = STAGE2_MODES if pair % 2 == 0 else tuple(reversed(STAGE2_MODES))
                for mode in mode_order:
                    ref_payload = {
                        "train": str(train_path),
                        "queries": [str(path) for path in query_paths],
                        "rows": args.rows,
                        "features": args.features,
                        "query_rows": args.query_rows,
                        "epochs": args.epochs,
                        "neighbors": args.neighbors,
                        "mode": mode,
                        "seed": seed,
                    }
                    reference = last_json(
                        run_supervised(
                            [
                                sys.executable,
                                str(Path(__file__).resolve()),
                                "--suite",
                                "stage2",
                                "--_stage2_worker",
                                json.dumps(ref_payload),
                            ],
                            deadline,
                        ).stdout,
                        require_fit_seconds=False,
                    )
                    output["coverage"]["quality_runs"] += 1
                    reference_scores[mode].append(reference)
                    output.setdefault("completed_references", []).append(
                        {"pair": pair, "mode": mode, "seed": seed, "reference": reference}
                    )
                    emit()
                    side_order = (
                        ("A", "B")
                        if (pair + (0 if mode == "exact" else 1)) % 2 == 0
                        else ("B", "A")
                    )
                    for side in side_order:
                        cuda_dir = temp / f"cuda-{pair}-{mode}-{side}"
                        cuda_dir.mkdir()
                        command = stage2_cuda_command(
                            binary,
                            train_path,
                            args.rows,
                            args.features,
                            args.epochs,
                            cuda_dir,
                            mode,
                            seed,
                            0,
                            1,
                            args.neighbors,
                            args.init,
                            queries,
                        )
                        result = last_json(
                            run_supervised(command, deadline, telemetry=telemetry).stdout,
                            require_fit_seconds=False,
                        )
                        output["resource_evidence"]["supplementary_nvidia_smi"] = telemetry.evidence
                        transform_times, width = (
                            result.get("transform_seconds"),
                            result.get("query_graph_width"),
                        )
                        if (
                            not isinstance(transform_times, list)
                            or len(transform_times) != 1
                            or len(transform_times[0]) != 3
                            or not isinstance(width, int)
                            or width != args.neighbors
                        ):
                            raise ValueError(
                                "Rust Stage2 output lacks transform_seconds[run][batch] "
                                "or query_graph_width"
                            )
                        fit_times = result.get("fit_seconds")
                        if (
                            not isinstance(fit_times, list)
                            or len(fit_times) != 1
                            or not math.isfinite(float(fit_times[0]))
                            or float(fit_times[0]) < 0
                            or any(
                                not math.isfinite(float(value)) or float(value) < 0
                                for value in transform_times[0]
                            )
                        ):
                            raise ValueError(
                                "Rust Stage2 output lacks a finite single-run fit_seconds value"
                            )
                        quality_payload = {
                            "kind": "cuda-quality",
                            "train": str(train_path),
                            "queries": [str(path) for path in query_paths],
                            "rows": args.rows,
                            "features": args.features,
                            "query_rows": args.query_rows,
                            "neighbors": args.neighbors,
                            "fit_embedding": str(cuda_dir / "embedding_0.f32le"),
                            "fit_graph": str(cuda_dir / "graph_0.i32le"),
                            "transform_embeddings": [
                                str(cuda_dir / f"transform_0_{batch}.f32le") for batch in range(3)
                            ],
                            "query_graphs": [
                                str(cuda_dir / f"query_graph_0_{batch}.i32le") for batch in range(3)
                            ],
                            "diagnostic_path": str(
                                raw_artifacts / f"reverse-{pair}-{mode}-{side}.json"
                            ),
                            "graph_width": width,
                        }
                        measured = last_json(
                            run_supervised(
                                [
                                    sys.executable,
                                    str(Path(__file__).resolve()),
                                    "--suite",
                                    "stage2",
                                    "--_stage2_worker",
                                    json.dumps(quality_payload),
                                ],
                                deadline,
                            ).stdout,
                            require_fit_seconds=False,
                        )
                        per_batch = []
                        for batch in range(3):
                            ref_batch = reference["transform"][batch]
                            score = measured["transform"][batch]["trustworthiness"]
                            per_batch.append(
                                {
                                    "batch": batch,
                                    "seed": seeds[batch],
                                    "seconds": float(transform_times[0][batch]),
                                    "trustworthiness": score,
                                    "recall": measured["transform"][batch]["recall"],
                                    "reference_trustworthiness": ref_batch["trustworthiness"],
                                    "reference_trustworthiness_mean": None,
                                }
                            )
                        output["pairs"].append(
                            {
                                "pair": pair,
                                "mode": mode,
                                "side": side,
                                "order": len(output["pairs"]),
                                "mode_order": list(mode_order),
                                "side_order": list(side_order),
                                "seed": seed,
                                "cuda": result,
                                "reference": reference,
                                "fit_trustworthiness": measured["fit"]["trustworthiness"],
                                "fit_seconds": result["fit_seconds"],
                                "reference_fit_trustworthiness": reference["fit_trustworthiness"],
                                "fit_recall": measured["fit"]["recall"],
                                "transform": per_batch,
                                "reverse_graph_diagnostics": measured["reverse_graph_diagnostics"],
                                "transform_total_seconds": sum(row["seconds"] for row in per_batch),
                            }
                        )
                        output["coverage"]["completed_baseline_repetitions"] += 1
                        output["coverage"]["completed_pairs"] += int(side == side_order[-1])
                        emit()
                        deadline.remaining()
            if args.profile_nsys:
                output["coverage"]["profiles"] = {"status": "running", "runs": {}}
                output["resource_evidence"]["nsys_modes"] = {}
                nsys = shutil.which("nsys")
                if nsys is None:
                    output["coverage"]["profiles"] = {
                        "status": "unavailable",
                        "reason": "nsys executable not found",
                    }
                    for mode in STAGE2_MODES:
                        output["resource_evidence"]["nsys_modes"][mode] = nsys_memory_incomplete(
                            STAGE2_GPU_CAP, "nsys executable not found"
                        )
                else:
                    profile_root = Path(args.output).resolve().parent / f"{args.output.stem}-nsys"
                    profile_root.mkdir(parents=True, exist_ok=True)
                    for mode in STAGE2_MODES:
                        profile_out = profile_root / mode
                        (temp / f"profile-{mode}").mkdir()
                        profile_command = stage2_cuda_command(
                            binary,
                            train_path,
                            args.rows,
                            args.features,
                            args.epochs,
                            temp / f"profile-{mode}",
                            mode,
                            42,
                            args.warmups,
                            1,
                            args.neighbors,
                            args.init,
                            queries,
                        )
                        trace_path = raw_artifacts / f"host-trace-{mode}.json"
                        profile_command.extend(["--trace", str(trace_path)])
                        profile = run_supervised(
                            [
                                nsys,
                                "profile",
                                "--force-overwrite=true",
                                "--trace=cuda,nvtx,osrt",
                                "--cuda-memory-usage=true",
                                "--output",
                                str(profile_out),
                                *profile_command,
                            ],
                            deadline,
                            telemetry=telemetry,
                        )
                        output["resource_evidence"]["supplementary_nvidia_smi"] = telemetry.evidence
                        sqlite_out = Path(str(profile_out) + ".sqlite")
                        try:
                            run_supervised(
                                [
                                    nsys,
                                    "export",
                                    "--type",
                                    "sqlite",
                                    "--force-overwrite=true",
                                    "-o",
                                    str(sqlite_out),
                                    str(profile_out) + ".nsys-rep",
                                ],
                                deadline,
                            )
                            memory_evidence = nsys_cuda_memory_evidence(sqlite_out, STAGE2_GPU_CAP)
                        except (subprocess.SubprocessError, OSError) as error:
                            memory_evidence = nsys_memory_incomplete(
                                STAGE2_GPU_CAP, f"Nsight Systems export failed: {error}"
                            )
                        output["resource_evidence"]["nsys_modes"][mode] = memory_evidence
                        stats = run_supervised(
                            [
                                nsys,
                                "stats",
                                "--report",
                                "cuda_gpu_kern_sum,cuda_api_sum",
                                "--format",
                                "csv",
                                "--force-export=true",
                                str(profile_out) + ".nsys-rep",
                            ],
                            deadline,
                        )
                        stats_path = raw_artifacts / f"nsys-stats-{mode}.csv"
                        stats_path.write_text(stats.stdout, encoding="utf-8")
                        stats_stderr_path = raw_artifacts / f"nsys-stats-{mode}.stderr.txt"
                        stats_stderr_path.write_text(stats.stderr, encoding="utf-8")
                        output["coverage"]["profiles"]["runs"][mode] = {
                            "report": str(profile_out) + ".nsys-rep",
                            "stdout": profile.stdout[-2000:],
                            "stderr": profile.stderr[-2000:],
                            "host_trace": str(trace_path),
                            "stats_csv": str(stats_path),
                            "stats_csv_sha256": sha256_file(stats_path),
                            "stats_stderr": str(stats_stderr_path),
                            "sqlite": str(sqlite_out),
                            "memory_evidence": memory_evidence,
                            "workload": "same fit plus three transforms as measured runs",
                        }
                    output["coverage"]["profiles"]["status"] = "captured"
            aggregate = stage2_aggregate(output["pairs"], reference_scores, args.paired_runs)
            output.update(
                {
                    key: aggregate[key]
                    for key in ("quality", "timings", "paired_timing_differences", "warnings")
                }
            )
            output["coverage"]["full_baseline_pair_coverage"] = aggregate["full_coverage"]
            output["coverage"]["speed_gates"] = aggregate["speed_gates"]
            output["resource_status"] = finalize_resource_evidence(
                output["resource_evidence"], telemetry
            )
            output["complete"] = (
                output["coverage"]["completed_pairs"] == args.paired_runs * 2
                and aggregate["full_coverage"]
                and all(v["all_runs_pass"] for v in output["quality"].values())
            )
            if not output["complete"]:
                output["incomplete_reason"] = "quality failed or required pair coverage missing"
            elif output["resource_status"] == "fail":
                output["complete"] = False
                output["incomplete_reason"] = "Nsight Systems GPU lifetime memory cap exceeded"
            elif output["resource_status"] != "pass":
                output["complete"] = False
                output["incomplete_reason"] = "GPU lifetime memory evidence unavailable"
            else:
                output["incomplete_reason"] = None
    except (
        DeadlineExceeded,
        ResourceLimitExceeded,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
        IndexError,
        KeyError,
        TypeError,
        statistics.StatisticsError,
    ) as error:
        output["incomplete_reason"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            failure_path = raw_artifacts / f"failure-{len(output.get('failures', [])):03d}.json"
            failure = {
                "argv": error.cmd,
                "returncode": error.returncode,
                "stdout": error.stdout,
                "stderr": error.stderr,
            }
            failure_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            output.setdefault("failures", []).append({"artifact": str(failure_path)})
        if "telemetry" in locals():
            output["resource_status"] = finalize_resource_evidence(
                output["resource_evidence"], telemetry
            )
    finally:
        recovered = []
        for path in sorted(raw_artifacts.glob("reverse-*.json")):
            try:
                diagnostic = json.loads(path.read_text(encoding="utf-8"))
                recovered.append({"artifact": str(path), "diagnostics": diagnostic})
            except (OSError, json.JSONDecodeError) as error:
                recovered.append({"artifact": str(path), "recovery_error": str(error)})
        output["reverse_graph_diagnostics"] = recovered
        emit()
    return 0 if output["complete"] else 1


def last_json(text: str, *, require_fit_seconds: bool = True) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and (not require_fit_seconds or "fit_seconds" in parsed):
            return parsed
    raise ValueError("worker did not emit its result JSON")


def make_clustered_input(rows: int, features: int, destination: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    centers = rng.normal(0.0, 5.0, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % len(centers)
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, 0.35, size=(rows, features)).astype(np.float32)
    matrix.astype("<f4", copy=False).tofile(destination)


def _python_worker(
    input_path: Path, rows: int, features: int, epochs: int, mode: str, runs: int, out: Path
) -> dict[str, Any]:
    import numba
    import numpy as np
    import umap
    from sklearn.neighbors import NearestNeighbors

    if umap.__version__ != "0.5.12":
        raise ValueError("This harness requires umap-learn 0.5.12")
    matrix = np.fromfile(input_path, dtype="<f4").reshape(rows, features)

    def fit_once() -> Any:
        precomputed = None
        if mode == "exact":
            search = NearestNeighbors(n_neighbors=NEIGHBORS, n_jobs=-1).fit(matrix)
            distances, indices = search.kneighbors(matrix)
            precomputed = (indices, distances, None)
        estimator = umap.UMAP(
            n_neighbors=NEIGHBORS,
            n_components=2,
            n_epochs=epochs,
            init="random",
            min_dist=0.0,
            random_state=None,
            metric="euclidean",
            force_approximation_algorithm=mode == "approximate",
            precomputed_knn=precomputed if precomputed is not None else (None, None, None),
            n_jobs=-1,
            verbose=False,
        )
        estimator.fit(matrix)
        return estimator

    fit_once()  # warmup (numba JIT)
    fit_seconds = []
    for run in range(runs):
        start = time.perf_counter()
        estimator = fit_once()
        fit_seconds.append(time.perf_counter() - start)
        np.save(out / f"embedding_{run}.npy", np.asarray(estimator.embedding_, dtype=np.float32))
        np.save(out / f"knn_{run}.npy", np.asarray(estimator._knn_indices))
    return {
        "fit_seconds": fit_seconds,
        "umap_learn_version": umap.__version__,
        "numba_threads": numba.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "config": {
            "random_state": None,
            "n_jobs": -1,
            "min_dist": 0.0,
            "force_approximation_algorithm": mode == "approximate",
            "exact_precomputed_knn_inside_timed_fit": mode == "exact",
        },
    }


def nonself(indices: Any, width: int) -> Any:
    import numpy as np

    rows = indices.shape[0]
    result = np.empty((rows, width), dtype=np.int64)
    for row in range(rows):
        ids = [int(i) for i in indices[row] if int(i) != row and int(i) >= 0]
        if len(ids) < width:
            raise ValueError(f"row {row} has fewer than {width} non-self neighbors")
        result[row] = ids[:width]
    return result


def tie_aware_recall(matrix: Any, actual: Any, exact_kth: Any) -> float:
    """Fraction of actual non-self ids no farther than the exact k-th non-self distance."""
    import numpy as np

    rows = np.arange(actual.shape[0])[:, None]
    distances = np.linalg.norm(matrix[actual] - matrix[rows], axis=2)
    hits = 0
    for row in range(actual.shape[0]):
        _, unique_positions = np.unique(actual[row], return_index=True)
        hits += int(np.sum(distances[row, unique_positions] <= exact_kth[row] * (1 + 1e-6) + 1e-12))
    return hits / (actual.shape[0] * actual.shape[1])


def median(values: list[float]) -> float:
    return statistics.median(values)


def load_quality_reference(path: Path) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or reference.get("complete") is not True:
        raise ValueError("quality reference must be a complete suite JSON")
    configuration = reference.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("quality reference is missing configuration metadata")
    primary = configuration.get("primary")
    expected_primary = {"rows": PRIMARY[0], "features": PRIMARY[1], "epochs": PRIMARY[2]}
    if primary != expected_primary:
        raise ValueError(f"quality reference primary workload {primary!r} != {expected_primary!r}")
    expected_metadata = {
        "neighbors": NEIGHBORS,
        "components": 2,
        "init": "random",
        "metric": "euclidean",
        "input": "seeded clustered float32 (8 centers), little-endian row-major",
    }
    for key, expected in expected_metadata.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"quality reference {key} metadata {configuration.get(key)!r} != {expected!r}"
            )
    quality = reference.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("quality reference is missing quality results")
    for mode in MODES:
        mode_quality = quality.get(mode)
        if not isinstance(mode_quality, dict):
            raise ValueError(f"quality reference is missing {mode} quality results")
        values = mode_quality.get("cuda_trustworthiness")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(
                f"quality reference {mode} requires at least two CUDA trustworthiness samples"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                f"quality reference {mode} CUDA trustworthiness samples must be finite"
            )
    return reference


def nsys_kernels(binary_args: list[str], tag: str, temp: Path, deadline: Deadline) -> dict:
    report = temp / f"nsys-{tag}"
    run_with_deadline(
        [
            "nsys",
            "profile",
            "--trace=cuda,osrt",
            "--sample=none",
            "--force-overwrite=true",
            f"--output={report}",
            *binary_args,
        ],
        deadline,
    )
    stats = run_with_deadline(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum",
            "--format",
            "csv",
            "--force-export=true",
            str(report) + ".nsys-rep",
        ],
        deadline,
    )
    return {"report": str(report) + ".nsys-rep", "summary_csv": stats.stdout[-6000:]}


def ncu_kernel(binary_args: list[str], regex: str, temp: Path, deadline: Deadline) -> dict:
    report = temp / "ncu"
    profiled = run_with_deadline(
        [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{regex}",
            # The selected sixteen-batch fit has no fired edges in epoch zero.
            "--launch-skip",
            "16",
            "--launch-count",
            "1",
            "--set",
            "basic",
            "--force-overwrite",
            "--export",
            str(report),
            *binary_args,
        ],
        deadline,
        check=False,
    )
    if "ERR_NVGPUCTRPERM" in profiled.stdout + profiled.stderr:
        # Driver-level counter access is a host setting only the user can change.
        return {"status": "blocked", "reason": "ERR_NVGPUCTRPERM: GPU performance counters denied"}
    if profiled.returncode != 0:
        raise subprocess.CalledProcessError(
            profiled.returncode, profiled.args, profiled.stdout, profiled.stderr
        )
    details = run_with_deadline(
        ["ncu", "--import", str(report) + ".ncu-rep", "--csv", "--page", "details"], deadline
    )
    if not details.stdout.strip():
        raise ValueError(f"ncu kernel filter {regex!r} matched no kernel; refine it")
    return {
        "report": str(report) + ".ncu-rep",
        "matching_launches_skipped": 16,
        "details_csv": details.stdout,
    }


def _source_identity(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError(f"source root is not a directory: {source_root}")
    root = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if Path(root).resolve() != source_root:
        raise ValueError(
            f"source root must be a git worktree root: {source_root} (detected {root})"
        )
    commit = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    patch = subprocess.run(
        ["git", "-C", root, "diff", "HEAD", "--binary"], capture_output=True, check=True
    ).stdout
    untracked = subprocess.run(
        ["git", "-C", root, "ls-files", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    untracked_digest = hashlib.sha256()
    for relative in untracked.split(b"\0"):
        if relative:
            path = Path(root) / os.fsdecode(relative)
            untracked_digest.update(relative + b"\0" + path.read_bytes())
    identity = hashlib.sha256(
        commit.encode()
        + b"\0"
        + status.encode()
        + b"\0"
        + patch
        + b"\0"
        + untracked_digest.digest()
    ).hexdigest()
    return {
        "repository": root,
        "commit": commit,
        "working_tree_status": status,
        "working_tree_status_sha256": hashlib.sha256(status.encode()).hexdigest(),
        "tracked_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "untracked_files_sha256": untracked_digest.hexdigest(),
        "source_tree_identity_sha256": identity,
    }


def _ann_worker_identity(binary: Path) -> dict[str, Any]:
    binary = binary.resolve()
    return {"path": str(binary), "sha256": sha256_file(binary)}


class AnnConstructionWorker:
    def __init__(
        self,
        binary: Path,
        data: Path,
        rows: int,
        features: int,
        device_index: int,
        compile_log: Path,
    ):
        compile_log.parent.mkdir(parents=True, exist_ok=True)
        compile_log.unlink(missing_ok=True)
        self.compile_log = compile_log.resolve()
        self.config_dir = self.compile_log.parent / f"{self.compile_log.stem}-worker"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.config_dir / "cubecl.toml"
        config_path.write_text(
            "[profiling.logger]\n"
            'level = "disabled"\n'
            "\n"
            "[compilation.logger]\n"
            'level = "full"\n'
            f"file = {json.dumps(str(self.compile_log))}\n",
            encoding="utf-8",
        )
        child_env = os.environ.copy()
        child_env.pop("CUBECL_DEBUG_LOG", None)
        child_env.pop("CUBECL_DEBUG_OPTION", None)
        self.process = subprocess.Popen(
            [
                str(binary.resolve()),
                str(data.resolve()),
                str(rows),
                str(features),
                str(device_index),
            ],
            cwd=self.config_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
        )

    def _stable_log_snapshot(self) -> dict[str, Any]:
        stable_observations = 0
        previous: tuple[int, int] | None = None
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                content = self.compile_log.read_bytes()
            except FileNotFoundError:
                content = b""
            current = (
                len(content),
                sum(line == b"[START_KERNEL_COMPILATION]" for line in content.splitlines()),
            )
            if current == previous:
                stable_observations += 1
            else:
                stable_observations = 0
            if stable_observations >= 4:
                return {
                    "bytes": current[0],
                    "compilation_records": current[1],
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "stable_observations": stable_observations + 1,
                    "poll_interval_ms": 25,
                }
            previous = current
            time.sleep(0.025)
        raise TimeoutError(f"CubeCL compile log did not stabilize: {self.compile_log}")

    def request(
        self, op: str, seed: int, k: int, cap: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        assert self.process.stdin is not None and self.process.stdout is not None
        before = self._stable_log_snapshot()
        self.process.stdin.write(json.dumps({"op": op, "seed": seed, "k": k, "cap": cap}) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("ANN worker closed stdout before replying")
        value = json.loads(line)
        if value.get("ok") is not True:
            raise RuntimeError(f"ANN worker error: {value}")
        after = self._stable_log_snapshot()
        return value, before, after

    def close(self) -> None:
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        if self.process.returncode:
            error = self.process.stderr.read() if self.process.stderr else ""
            raise RuntimeError(f"ANN worker exit {self.process.returncode}: {error[-4000:]}")


def _validate_ann_graph(ids: Any, distances: Any, rows: int, k: int) -> tuple[Any, Any]:
    import numpy as np

    ids, distances = np.asarray(ids), np.asarray(distances)
    if ids.ndim != 2 or distances.shape != ids.shape or ids.shape[0] != rows:
        raise ValueError(f"invalid neighbor graph shapes: {ids.shape}, {distances.shape}")
    clean_ids = np.empty((rows, k), dtype="<i4")
    clean_distances = np.empty((rows, k), dtype="<f4")
    for row in range(rows):
        keep = ids[row] != row
        row_ids, row_distances = ids[row][keep], distances[row][keep]
        if (
            row_ids.size != k
            or np.any(row_ids < 0)
            or np.any(row_ids >= rows)
            or np.unique(row_ids).size != k
            or np.any(row_ids == row)
        ):
            raise ValueError(
                f"row {row} does not contain exactly {k} unique valid nonself neighbors"
            )
        if not np.all(np.isfinite(row_distances)) or np.any(row_distances < 0):
            raise ValueError(f"row {row} contains invalid distances")
        if np.any(row_distances[1:] < row_distances[:-1]):
            raise ValueError(f"row {row} contains descending materialized distances")
        clean_ids[row], clean_distances[row] = row_ids, row_distances
    return clean_ids, clean_distances


class AnnGraphValidationTests(unittest.TestCase):
    def test_equal_rounded_sqrt_distances_do_not_require_id_tie_break(self) -> None:
        import numpy as np

        squared = np.array(
            [1.0, np.nextafter(np.float32(1.0), np.float32(np.inf))], dtype=np.float32
        )
        distances = np.sqrt(squared, dtype=np.float32)
        self.assertNotEqual(squared[0], squared[1])
        self.assertEqual(distances[0], distances[1])
        graph_ids = [[2, 1], [2, 0], [1, 0]]
        graph_distances = np.tile(distances, (3, 1))
        ids, validated_distances = _validate_ann_graph(graph_ids, graph_distances, rows=3, k=2)
        np.testing.assert_array_equal(ids, graph_ids)
        np.testing.assert_array_equal(validated_distances, graph_distances)

    def test_descending_materialized_distances_are_rejected(self) -> None:
        import numpy as np

        with self.assertRaisesRegex(ValueError, "descending materialized distances"):
            _validate_ann_graph(
                [[1, 2], [0, 2], [0, 1]],
                [[1.0, np.nextafter(np.float32(1.0), np.float32(0.0))], [1.0, 2.0], [1.0, 2.0]],
                rows=3,
                k=2,
            )


def _numba_dispatcher_snapshot(numba: Any) -> dict[str, Any]:
    """Snapshot all currently-live Numba dispatchers without entering build timing."""
    import gc

    from numba.core.registry import CPUDispatcher

    dispatchers: dict[int, tuple[str, ...]] = {}
    dispatcher_refs: dict[int, Any] = {}
    for value in gc.get_objects():
        try:
            if isinstance(value, CPUDispatcher):
                dispatchers[id(value)] = tuple(
                    sorted(repr(signature) for signature in value.signatures)
                )
                dispatcher_refs[id(value)] = value
        except (ReferenceError, RuntimeError):
            continue
    flattened = [
        f"{identity}:{signature}"
        for identity, signatures in dispatchers.items()
        for signature in signatures
    ]
    flattened.sort()
    signature_bytes = "\n".join(flattened).encode()
    return {
        "live_dispatcher_count": len(dispatchers),
        "compiled_signature_count": sum(len(signatures) for signatures in dispatchers.values()),
        "dispatcher_ids": dispatchers,
        "dispatcher_refs": dispatcher_refs,
        "signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
    }


def _numba_compile_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old, new = before["dispatcher_ids"], after["dispatcher_ids"]
    new_dispatchers = sorted(set(new) - set(old))
    lost_dispatchers = sorted(set(old) - set(new))
    added_signatures = [
        {"dispatcher_id": identity, "signature": signature}
        for identity in set(old) | set(new)
        for signature in sorted(set(new.get(identity, ())) - set(old.get(identity, ())))
    ]
    return {
        "dispatcher_count_before": before["live_dispatcher_count"],
        "dispatcher_count_after": after["live_dispatcher_count"],
        "new_dispatchers": len(new_dispatchers),
        "vanished_dispatchers": len(lost_dispatchers),
        "compiled_signature_count_before": before["compiled_signature_count"],
        "compiled_signature_count_after": after["compiled_signature_count"],
        "new_compiled_signatures": added_signatures,
        "compile_event_count": len(added_signatures),
        "compile_contaminated": bool(added_signatures or new_dispatchers or lost_dispatchers),
        "before_signature_sha256": before["signature_sha256"],
        "after_signature_sha256": after["signature_sha256"],
        "method": (
            "live numba.core.registry.CPUDispatcher identities plus public signatures snapshots "
            "before and after each constructor; snapshots run outside timed interval"
        ),
    }


def _ann_construction_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Interleaved CUDA ANN construction campaign")
    parser.add_argument("--suite", choices=["ann-construction"], required=True)
    parser.add_argument("--baseline-worker", type=Path, required=True)
    parser.add_argument("--candidate-worker", type=Path, required=True)
    parser.add_argument("--baseline-source-root", type=Path, required=True)
    parser.add_argument("--candidate-source-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--features", type=int, required=True)
    parser.add_argument("--k", type=int, required=True, help="nonself neighbor count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args(argv)
    if (
        args.rows <= 25
        or args.features < 1
        or args.k != 24
        or args.k >= args.rows
        or args.device_index < 0
    ):
        parser.error("campaign requires rows > 25, features > 0 and k=24")
    try:
        source_identities = {
            "baseline": _source_identity(args.baseline_source_root),
            "candidate": _source_identity(args.candidate_source_root),
        }
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(f"invalid source root: {error}")
    data = args.data.resolve()
    if data.stat().st_size != args.rows * args.features * 4:
        parser.error("f32le input length does not match rows * features * 4")

    for name in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "15"
    import numpy as np
    import pynndescent
    from pynndescent import NNDescent

    numba = __import__("numba")
    numba.set_num_threads(15)
    matrix = np.fromfile(data, dtype="<f4").reshape(args.rows, args.features)
    if pynndescent.__version__ != "0.6.0":
        parser.error(f"requires PyNNDescent 0.6.0, found {pynndescent.__version__}")
    configuration = {
        "rows": args.rows,
        "features": args.features,
        "metric": "euclidean",
        "k_nonself": 24,
        "n_neighbors_including_self": 25,
        "seed": 42,
        "cuda_cap": 14,
        "cuda_device_index": args.device_index,
        "pynndescent": {
            "version": pynndescent.__version__,
            "n_jobs": 15,
            "n_trees": 11,
            "max_candidates": 60,
            "n_iters": 14,
            "n_neighbors": 25,
            "random_state": 42,
            "delta": 0.001,
        },
        "warmups_per_implementation": 1,
        "timed_builds_per_implementation": 10,
        "timer": (
            "CUDA host input through upload, construction, sync and materialized host "
            "IDs/distances; PyNNDescent constructor including trees plus neighbor_graph read; "
            "disk IO excluded"
        ),
        "ipc": (
            "persistent Rust process; stdin/stdout JSONL {op:warmup|build,seed,k,cap}; "
            "serial requests"
        ),
        "order": "ten trials; baseline/candidate/PyNNDescent rotated one-at-a-time; no overlap",
    }
    output: dict[str, Any] = {
        "schema": "cuscan.ann-construction-campaign.v1",
        "complete": False,
        "protocol_admissible": False,
        "configuration": configuration,
        "provenance": {
            "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "data": {"path": str(data), "sha256": sha256_file(data), "bytes": data.stat().st_size},
            "harness": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "workers": {
                "baseline": {
                    **_ann_worker_identity(args.baseline_worker),
                    "source": source_identities["baseline"],
                },
                "candidate": {
                    **_ann_worker_identity(args.candidate_worker),
                    "source": source_identities["candidate"],
                },
            },
            "source_roots": source_identities,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": sys.version,
            },
            "dependencies": {
                name: __import__("importlib.metadata", fromlist=["version"]).version(name)
                for name in ("numpy", "pynndescent", "numba", "scipy", "scikit-learn")
            },
            "gpu": gpu_device_provenance(),
        },
        "compile_contamination": {
            "status": "cubecl_log_and_numba_dispatcher_snapshots",
            "cubecl": {
                "record_marker": "[START_KERNEL_COMPILATION]",
                "configuration": (
                    "Each worker runs from an isolated per-worker cwd containing cubecl.toml with "
                    "[compilation.logger] level='full' file=<dedicated log> and "
                    "[profiling.logger] level='disabled'. CUBECL_DEBUG_LOG and "
                    "CUBECL_DEBUG_OPTION are removed from child environment."
                ),
                "record_cost": (
                    "CubeCL emits records from the compilation/load path only; "
                    "`CompiledKernel` Full formatting and any C++ source formatting occur only "
                    "on new compilation. "
                    "Profiling is independently disabled, so no kernel profiling/synchronization "
                    "instrumentation is enabled."
                ),
                "delivery": (
                    "The source `ServerLogger` sends compilation messages through an async "
                    "channel; "
                    "the file sink writes one complete record and flushes on each logger write. "
                    "This harness waits for five identical file-size/event-count observations "
                    "25ms apart after each response; this is a quiet-period barrier, not a "
                    "synchronous CubeCL acknowledgement."
                ),
                "logs": {},
            },
            "pynndescent": {
                "method": (
                    "live Numba CPUDispatcher identity and public signatures/count snapshots "
                    "before and after every constructor call, performed outside "
                    "the constructor timer"
                ),
                "proof_scope": (
                    "No newly visible Numba dispatcher/signature in the timed call proves the "
                    "observed dispatcher set was post-warmup; Numba exposes no general public "
                    "compilation event counter here, so this is explicitly a dispatcher-snapshot "
                    "signal, not a claim about transient unseen dispatchers."
                ),
            },
            "rejected_samples": [],
            "strategy": (
                "One in-process warmup per implementation. Reject and rerun a CUDA sample if its "
                "stabilized per-worker CubeCL log gains compilation markers; reject and rerun "
                "PyNNDescent if its live dispatcher count or compiled signatures change. Keep "
                "every rejected sample and its evidence in this object."
            ),
        },
        "warmups": {},
        "samples": [],
        "medians_ms": {},
        "rejections": [],
    }

    def persist() -> None:
        out = args.output.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    workers: dict[str, AnnConstructionWorker] = {}
    try:
        log_dir = args.output.resolve().with_suffix("").with_name(args.output.stem + "-cubecl-logs")
        workers["baseline"] = AnnConstructionWorker(
            args.baseline_worker,
            data,
            args.rows,
            args.features,
            args.device_index,
            log_dir / "baseline.log",
        )
        workers["candidate"] = AnnConstructionWorker(
            args.candidate_worker,
            data,
            args.rows,
            args.features,
            args.device_index,
            log_dir / "candidate.log",
        )
        for name in ("baseline", "candidate"):
            result, before_log, after_log = workers[name].request("warmup", 42, 24, 14)
            output["warmups"][name] = {
                **{
                    key: result[key]
                    for key in (
                        "elapsed_ms",
                        "ids_hash_fnv1a64",
                        "distances_hash_fnv1a64",
                        "compile_signal",
                    )
                },
                "cubecl_compile_records_during_warmup": after_log["compilation_records"]
                - before_log["compilation_records"],
                "cubecl_log_snapshot": after_log,
            }
            output["compile_contamination"]["cubecl"]["logs"][name] = {
                "path": str(workers[name].compile_log),
                "configuration_path": str(workers[name].config_dir / "cubecl.toml"),
                "configuration_sha256": sha256_file(workers[name].config_dir / "cubecl.toml"),
                "profiling_level": "disabled",
                "compilation_level": "full",
                "sha256_after_warmup": after_log["sha256"],
                "warmup_compile_records": after_log["compilation_records"]
                - before_log["compilation_records"],
            }

        def py_build() -> tuple[float, Any, Any]:
            start = time.perf_counter()
            index = NNDescent(
                matrix,
                n_neighbors=25,
                n_trees=11,
                max_candidates=60,
                n_iters=14,
                random_state=42,
                n_jobs=15,
                delta=0.001,
                metric="euclidean",
            )
            graph = index.neighbor_graph
            elapsed = (time.perf_counter() - start) * 1000.0
            ids, distances = _validate_ann_graph(graph[0], graph[1], args.rows, 24)
            return elapsed, ids, distances

        py_before = _numba_dispatcher_snapshot(numba)
        py_ms, py_ids, py_distances = py_build()
        py_after = _numba_dispatcher_snapshot(numba)
        py_warm_compile = _numba_compile_delta(py_before, py_after)
        output["warmups"]["pynndescent"] = {
            "elapsed_ms": py_ms,
            "ids_sha256": hashlib.sha256(py_ids.tobytes()).hexdigest(),
            "distances_sha256": hashlib.sha256(py_distances.tobytes()).hexdigest(),
            "numba_dispatcher_snapshot": py_warm_compile,
        }
        accepted = {"baseline": 0, "candidate": 0, "pynndescent": 0}
        for trial in range(10):
            labels = ["baseline", "candidate", "pynndescent"]
            order = labels[trial % 3 :] + labels[: trial % 3]
            for order_index, name in enumerate(order):
                attempts = 0
                while accepted[name] <= trial:
                    attempts += 1
                    if attempts > 20:
                        raise RuntimeError(
                            f"{name} continued compiling after warmup for 20 consecutive attempts"
                        )
                    sample: dict[str, Any] = {
                        "trial": trial,
                        "order_index": order_index,
                        "attempt": attempts - 1,
                        "accepted_index": accepted[name],
                        "implementation": name,
                    }
                    if name == "pynndescent":
                        before = _numba_dispatcher_snapshot(numba)
                        elapsed, ids, distances = py_build()
                        after = _numba_dispatcher_snapshot(numba)
                        compile_delta = _numba_compile_delta(before, after)
                        sample.update(
                            {
                                "elapsed_ms": elapsed,
                                "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
                                "distances_sha256": hashlib.sha256(distances.tobytes()).hexdigest(),
                                "numba_dispatcher_snapshot": compile_delta,
                                "compile_event_count": compile_delta["compile_event_count"],
                                "compile_contaminated": compile_delta["compile_contaminated"],
                            }
                        )
                    else:
                        result, before_log, after_log = workers[name].request("build", 42, 24, 14)
                        event_delta = (
                            after_log["compilation_records"] - before_log["compilation_records"]
                        )
                        sample.update(
                            {
                                "elapsed_ms": result["elapsed_ms"],
                                "ids_hash_fnv1a64": result["ids_hash_fnv1a64"],
                                "distances_hash_fnv1a64": result["distances_hash_fnv1a64"],
                                "cubecl_log_before": before_log,
                                "cubecl_log_after": after_log,
                                "compile_event_count": event_delta,
                                "compile_contaminated": event_delta > 0,
                            }
                        )
                    if sample["compile_contaminated"]:
                        sample["rejected"] = True
                        output["compile_contamination"]["rejected_samples"].append(sample)
                        output["rejections"].append(
                            {
                                "trial": trial,
                                "implementation": name,
                                "attempt": attempts - 1,
                                "elapsed_ms": sample["elapsed_ms"],
                                "compile_event_count": sample["compile_event_count"],
                                "evidence": sample.get(
                                    "cubecl_log_after", sample.get("numba_dispatcher_snapshot")
                                ),
                            }
                        )
                    else:
                        sample["rejected"] = False
                        output["samples"].append(sample)
                        accepted[name] += 1
                    persist()
        for name, worker in workers.items():
            final_log = worker._stable_log_snapshot()
            output["compile_contamination"]["cubecl"]["logs"][name].update(
                {
                    "final_sha256": final_log["sha256"],
                    "final_bytes": final_log["bytes"],
                    "total_compilation_records": final_log["compilation_records"],
                }
            )
        output["medians_ms"] = {
            name: statistics.median(
                sample["elapsed_ms"]
                for sample in output["samples"]
                if sample["implementation"] == name
            )
            for name in ("baseline", "candidate", "pynndescent")
        }
        output["complete"] = True
        output["protocol_admissible"] = all(
            len([sample for sample in output["samples"] if sample["implementation"] == name]) == 10
            and all(
                sample["compile_event_count"] == 0
                for sample in output["samples"]
                if sample["implementation"] == name
            )
            for name in ("baseline", "candidate", "pynndescent")
        )
        expected_order = []
        labels = ["baseline", "candidate", "pynndescent"]
        for trial in range(10):
            order = labels[trial % 3 :] + labels[: trial % 3]
            expected_order.extend((trial, name) for name in order)
        actual_order = [(sample["trial"], sample["implementation"]) for sample in output["samples"]]
        if actual_order != expected_order:
            raise RuntimeError(
                "accepted construction samples do not match the interleaved schedule"
            )
        output["compile_contamination"]["status"] = (
            "zero_timed_compile_events_observed"
            if output["protocol_admissible"]
            else "compile_exclusion_not_established"
        )
        persist()
    except BaseException as error:
        output["error"] = f"{type(error).__name__}: {error}"
        persist()
        raise
    finally:
        for worker in workers.values():
            worker.close()
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _cli() -> int:
    if (
        "--suite" in sys.argv
        and sys.argv.index("--suite") + 1 < len(sys.argv)
        and sys.argv[sys.argv.index("--suite") + 1] == "stage2"
    ):
        return _stage2_cli(sys.argv[1:])
    if (
        "--suite" in sys.argv
        and sys.argv.index("--suite") + 1 < len(sys.argv)
        and sys.argv[sys.argv.index("--suite") + 1] == "ann-construction"
    ):
        return _ann_construction_cli(sys.argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["stage1"], default="stage1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=600.0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, help="write the full suite JSON")
    parser.add_argument("--baseline", type=Path, help="pre-change suite JSON for H-03 ratios")
    parser.add_argument(
        "--quality-reference",
        type=Path,
        help="complete pre-change serial suite JSON for per-mode CUDA quality tracking",
    )
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--profile-ncu", action="store_true")
    parser.add_argument("--ncu-kernel", default=".*layout_vertex.*kernel.*")
    parser.add_argument("--profile-dir", type=Path, help="keep profiler reports here")
    parser.add_argument("--_python-worker", nargs=7, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._python_worker:
        input_path, rows, features, epochs, mode, runs, out = args._python_worker
        result = _python_worker(
            Path(input_path), int(rows), int(features), int(epochs), mode, int(runs), Path(out)
        )
        print(json.dumps(result))
        return 0
    if args.runs < 1 or args.warmups < 1:
        parser.error("runs and warmups must be positive")
    quality_reference = None
    if args.quality_reference:
        try:
            quality_reference = load_quality_reference(args.quality_reference)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(f"invalid --quality-reference: {error}")
    binary = args.binary.resolve()
    if not binary.is_file():
        print(
            f"CUDA profiler executable not found: {binary}\nBuild it first:\n"
            "  cargo build -p umap --release --features tracing "
            "--example profile_standard_cuda",
            file=sys.stderr,
        )
        return 2

    import numpy as np
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    deadline = Deadline(args.deadline_seconds)
    start = time.monotonic()
    output: dict[str, Any] = {
        "configuration": {
            "primary": {"rows": PRIMARY[0], "features": PRIMARY[1], "epochs": PRIMARY[2]},
            "small": [{"rows": r, "features": f, "epochs": e} for r, f, e in SMALL],
            "neighbors": NEIGHBORS,
            "components": 2,
            "init": "random",
            "min_dist": 0.0,
            "metric": "euclidean",
            "warmups": args.warmups,
            "runs": args.runs,
            "deadline_seconds": args.deadline_seconds,
            "input": "seeded clustered float32 (8 centers), little-endian row-major",
            "cuda_seeds": "42+run (warmups 1000+i)",
        },
        "complete": False,
    }

    def emit() -> None:
        output["elapsed_seconds"] = time.monotonic() - start
        rendered = json.dumps(output, indent=2, sort_keys=True)
        if args.output:
            args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        print(rendered)

    with tempfile.TemporaryDirectory(prefix="stage1-") as temporary:
        temp = Path(temporary)
        profile_dir = args.profile_dir.resolve() if args.profile_dir else temp
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows, features, epochs = PRIMARY
            primary_input = temp / "primary.f32le"
            make_clustered_input(rows, features, primary_input)
            matrix = np.fromfile(primary_input, dtype="<f4").reshape(rows, features)
            primary: dict[str, Any] = {}
            for mode in MODES:
                py_dir = temp / f"py-{mode}"
                py_dir.mkdir()
                python = last_json(
                    run_with_deadline(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--_python-worker",
                            str(primary_input),
                            str(rows),
                            str(features),
                            str(epochs),
                            mode,
                            str(args.runs),
                            str(py_dir),
                        ],
                        deadline,
                    ).stdout
                )
                cuda_dir = temp / f"cuda-{mode}"
                cuda = last_json(
                    run_with_deadline(
                        [
                            str(binary),
                            str(primary_input),
                            str(rows),
                            str(features),
                            str(epochs),
                            str(cuda_dir),
                            "--search",
                            mode,
                            "--min-dist",
                            "0.0",
                            "--warmups",
                            str(args.warmups),
                            "--runs",
                            str(args.runs),
                        ],
                        deadline,
                    ).stdout
                )
                if cuda["fit_graph_width"] != FIT_WIDTH:
                    raise ValueError(f"CUDA fit graph width {cuda['fit_graph_width']} != 14")
                primary[mode] = {"python": python, "cuda": cuda, "dirs": (py_dir, cuda_dir)}
            output["primary"] = {
                mode: {
                    "python": {k: v for k, v in data["python"].items()},
                    "cuda": data["cuda"],
                    "python_median_seconds": median(data["python"]["fit_seconds"]),
                    "cuda_median_seconds": median(data["cuda"]["fit_seconds"]),
                }
                for mode, data in primary.items()
            }

            small: dict[str, Any] = {}
            for r, f, e in SMALL:
                small_input = temp / f"small-{r}.f32le"
                make_clustered_input(r, f, small_input)
                for mode in MODES:
                    result = last_json(
                        run_with_deadline(
                            [
                                str(binary),
                                str(small_input),
                                str(r),
                                str(f),
                                str(e),
                                str(temp / f"small-{r}-{mode}"),
                                "--search",
                                mode,
                                "--min-dist",
                                "0.0",
                                "--warmups",
                                str(args.warmups),
                                "--runs",
                                str(args.runs),
                            ],
                            deadline,
                        ).stdout
                    )
                    small[f"{r}x{f}/{e}/{mode}"] = {
                        "fit_seconds": result["fit_seconds"],
                        "median_seconds": median(result["fit_seconds"]),
                    }
            output["small_cuda"] = small

            deadline.remaining()
            exact = NearestNeighbors(n_neighbors=NEIGHBORS).fit(matrix)
            exact_dist, exact_idx = exact.kneighbors(matrix)
            exact_kth = np.array(
                [
                    sorted(
                        d for d, i in zip(exact_dist[row], exact_idx[row], strict=True) if i != row
                    )[FIT_WIDTH - 1]
                    for row in range(rows)
                ]
            )
            quality: dict[str, Any] = {}
            for mode, data in primary.items():
                py_dir, cuda_dir = data["dirs"]
                py_trust, cuda_trust, py_recall, cuda_recall = [], [], [], []
                for run in range(args.runs):
                    deadline.remaining()
                    py_emb = np.load(py_dir / f"embedding_{run}.npy")
                    cuda_emb = np.fromfile(cuda_dir / f"embedding_{run}.f32le", "<f4").reshape(
                        rows, 2
                    )
                    py_trust.append(float(trustworthiness(matrix, py_emb, n_neighbors=NEIGHBORS)))
                    cuda_trust.append(
                        float(trustworthiness(matrix, cuda_emb, n_neighbors=NEIGHBORS))
                    )
                    if mode == "approximate":
                        py_graph = nonself(np.load(py_dir / f"knn_{run}.npy"), FIT_WIDTH)
                        cuda_graph = np.fromfile(cuda_dir / f"graph_{run}.i32le", "<i4").reshape(
                            rows, FIT_WIDTH
                        )
                        py_recall.append(tie_aware_recall(matrix, py_graph, exact_kth))
                        cuda_recall.append(tie_aware_recall(matrix, cuda_graph, exact_kth))
                python_mean = statistics.fmean(py_trust)
                quality[mode] = {
                    "python_trustworthiness": py_trust,
                    "python_trustworthiness_mean": python_mean,
                    "cuda_trustworthiness": cuda_trust,
                    "Q-02_each_cuda_within_0.025_of_python_mean": all(
                        abs(t - python_mean) <= TRUST_GAP for t in cuda_trust
                    ),
                }
                if mode == "approximate":
                    quality[mode].update(
                        {
                            "python_fit_recall_at_14": py_recall,
                            "cuda_fit_recall_at_14": cuda_recall,
                            "Q-03_every_run_recall_ge_0.90": all(
                                r >= RECALL_LIMIT for r in py_recall + cuda_recall
                            ),
                        }
                    )
            output["quality"] = quality

            if quality_reference is not None:
                reference_quality = quality_reference["quality"]
                tracking_gates: dict[str, bool] = {}
                for mode in MODES:
                    reference_samples = reference_quality[mode]["cuda_trustworthiness"]
                    candidate_samples = quality[mode]["cuda_trustworthiness"]
                    if any(not math.isfinite(value) for value in candidate_samples):
                        raise ValueError(
                            f"candidate {mode} CUDA trustworthiness samples must be finite"
                        )
                    reference_mean = statistics.fmean(reference_samples)
                    reference_std = statistics.stdev(reference_samples)
                    threshold = reference_mean - 0.5 * reference_std
                    candidate_mean = statistics.fmean(candidate_samples)
                    passed = candidate_mean >= threshold
                    quality[mode]["tracking_reference"] = {
                        "sample_count": len(reference_samples),
                        "mean": reference_mean,
                        "sample_std": reference_std,
                        "threshold_mean_minus_0.5_sample_std": threshold,
                        "candidate_cuda_mean": candidate_mean,
                        "pass": passed,
                    }
                    tracking_gates[f"Rust-reference_{mode}_mean_ge_reference_minus_0.5std"] = passed
                output["quality_tracking"] = {
                    "policy": "warning_only",
                    "checks": tracking_gates,
                    "warnings": [name for name, passed in tracking_gates.items() if not passed],
                }

            gates: dict[str, Any] = {
                f"H-04_{mode}_cuda_median_le_0.90x_python": output["primary"][mode][
                    "cuda_median_seconds"
                ]
                <= SPEED_RATIO_LIMIT * output["primary"][mode]["python_median_seconds"]
                for mode in MODES
            }
            for mode in MODES:
                gates[f"H-04_{mode}_ratio_cuda_over_python"] = (
                    output["primary"][mode]["cuda_median_seconds"]
                    / output["primary"][mode]["python_median_seconds"]
                )
            if args.baseline:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                ratios = {
                    key: value["median_seconds"] / baseline["small_cuda"][key]["median_seconds"]
                    for key, value in small.items()
                }
                gates["H-03_small_ratio_vs_baseline"] = ratios
                gates["H-03_pass"] = all(r <= SMALL_RATIO_LIMIT for r in ratios.values())
                gates["primary_cuda_ratio_vs_baseline"] = {
                    mode: output["primary"][mode]["cuda_median_seconds"]
                    / baseline["primary"][mode]["cuda_median_seconds"]
                    for mode in MODES
                }
            output["gates"] = gates

            profiles: dict[str, Any] = {}
            primary_args = [str(binary), str(primary_input), *map(str, PRIMARY)]
            if args.profile_nsys:
                for mode in MODES:
                    profiles[f"nsys_{mode}"] = nsys_kernels(
                        [
                            *primary_args,
                            str(temp / f"nsys-{mode}-out"),
                            "--search",
                            mode,
                            "--min-dist",
                            "0.0",
                            "--warmups",
                            "1",
                            "--runs",
                            "1",
                        ],
                        mode,
                        profile_dir,
                        deadline,
                    )
            if args.profile_ncu:
                profiles["ncu_approximate"] = ncu_kernel(
                    [
                        *primary_args,
                        str(temp / "ncu-out"),
                        "--search",
                        "approximate",
                        "--min-dist",
                        "0.0",
                        "--warmups",
                        "1",
                        "--runs",
                        "1",
                    ],
                    args.ncu_kernel,
                    profile_dir,
                    deadline,
                )
            output["profiles"] = profiles
            deadline.remaining()
            output["complete"] = True
        except DeadlineExceeded as error:
            output["incomplete_reason"] = (
                f"{error}; results are incomplete and not a pass. Change parameters or extend "
                "the timeout (recommended: extend --deadline-seconds) and rerun."
            )
            emit()
            return 1
        except subprocess.CalledProcessError as error:
            output["incomplete_reason"] = f"worker failed: {error}\n{error.stderr[-4000:]}"
            emit()
            return 1
    emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
