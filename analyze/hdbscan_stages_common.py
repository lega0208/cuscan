"""Shared identity, JSON, and subprocess utilities for HDBSCAN stage analysis."""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO

# Standalone tools include Phase 7 experiments at 50k/300k and baseline investigations
# at 200k/300k; the integrated campaign script stays at 20k/50k/100k.
EXTENDED_ROWS = (20_000, 50_000, 100_000, 200_000, 300_000)


def identity(run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the shared identity written to ``raw/manifest.json``."""
    path = Path(run_dir) / "raw" / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read run identity manifest {path}: {error}") from error
    required = ("schema_version", "run_id", "workload", "input_sha256", "provenance")
    require_fields(data, required, f"identity manifest {path}")
    if data["schema_version"] != 1 or not isinstance(data["run_id"], str):
        raise ValueError(f"invalid schema_version or run_id in {path}")
    if not isinstance(data["workload"], dict) or not isinstance(data["provenance"], dict):
        raise ValueError(f"invalid workload or provenance in {path}")
    digest = data["input_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError(f"invalid input_sha256 in {path}")
    return {key: data[key] for key in required}


# Representative counter kernels (neighbor search, Boruvka candidates) per resolved search route.
ROUTE_KERNEL_FAMILIES: dict[str, tuple[str, str]] = {
    "BruteForce": ("partial_kernel", "tile_candidates"),
    "KdTree": ("kd_neighbors_kernel", "kd_candidates_kernel"),
}


def route_kernel_families(route: Any) -> tuple[str, str]:
    """Return counter kernel families for a resolved route, rejecting unknown routes."""
    if not isinstance(route, str) or route not in ROUTE_KERNEL_FAMILIES:
        raise ValueError(f"unsupported resolved search route {route!r}")
    return ROUTE_KERNEL_FAMILIES[route]


def uniform_route(routes: Any, context: str) -> str:
    """Return the single resolved route shared by every fit in ``routes``."""
    if not isinstance(routes, list) or not routes or len(set(map(str, routes))) != 1:
        raise ValueError(f"{context} must contain one consistent resolved search route")
    route = routes[0]
    route_kernel_families(route)
    return route


def observed_kernel_families(
    route: str, report: Any, expected_identity: dict[str, Any], binary: Any
) -> tuple[tuple[str, str], dict[str, Any]]:
    """Select KD counters strictly from the same binary's measured Nsight capture."""
    if route != "KdTree":
        return route_kernel_families(route), {}
    require_fields(report, ["profiled_fit", "capture", "export", "binary"], "cuda_kernels")
    for field in ("schema_version", "run_id", "workload", "input_sha256", "provenance"):
        if report.get(field) != expected_identity.get(field):
            raise ValueError(f"cuda_kernels identity mismatch for {field}")
    if (
        not isinstance(binary, dict)
        or not isinstance(binary.get("path"), str)
        or not binary["path"]
        or not isinstance(binary.get("sha256"), str)
        or len(binary["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in binary["sha256"])
        or report["binary"] != binary
    ):
        raise ValueError("cuda_kernels binary provenance mismatch")
    capture = report["capture"]
    if not isinstance(capture, dict) or (
        capture.get("exit_status") != 0 or capture.get("timed_out") is not False
    ):
        raise ValueError("cuda_kernels capture did not complete successfully")
    export = report["export"]
    if not isinstance(export, dict) or (
        export.get("exit_status") != 0 or export.get("timed_out") is not False
    ):
        raise ValueError("cuda_kernels export did not complete successfully")
    profiled = report["profiled_fit"]
    require_fields(profiled, ["native_profile", "kernels"], "cuda_kernels.profiled_fit")
    native = profiled["native_profile"]
    if not isinstance(native, dict) or native.get("resolved_search_per_fit") != [route]:
        raise ValueError("cuda_kernels measured route does not match benchmark route")
    rows = profiled["kernels"]
    if not isinstance(rows, list):
        raise ValueError("cuda_kernels measured kernels must be an array")
    choices = {"kd_neighbors_kernel", "kd_neighbors_tile_kernel"}
    observed = [row for row in rows if isinstance(row, dict) and row.get("family") in choices]
    families = {row["family"] for row in observed}
    if len(families) != 1:
        raise ValueError(
            "cuda_kernels needs exactly one observed KD neighbor family (missing or ambiguous)"
        )
    for row in observed:
        count = row.get("launch_count")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not isinstance(row.get("name"), str)
            or row["family"] not in row["name"]
        ):
            raise ValueError("cuda_kernels KD family lacks valid measured launch evidence")
    family = next(iter(families))
    evidence = {
        "source": "cuda_kernels.json",
        "neighbor_family": family,
        "kernel_names": sorted({row["name"] for row in observed}),
        "launch_count": sum(row["launch_count"] for row in observed),
    }
    return (family, "kd_candidates_kernel"), evidence


def validate_min_samples(identity_data: Any, benchmark: Any, requested: int) -> None:
    """Require counter-requested min_samples to match both provisioned identities."""
    if isinstance(requested, bool) or not isinstance(requested, int) or requested <= 0:
        raise ValueError("min_samples must be a positive integer")
    for source, report in (("manifest", identity_data), ("benchmark", benchmark)):
        try:
            configured = report["workload"]["config"]["min_samples"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{source} configuration lacks min_samples") from error
        if isinstance(configured, bool) or not isinstance(configured, int) or configured <= 0:
            raise ValueError(f"{source} configuration has invalid min_samples")
        if configured != requested:
            raise ValueError(f"requested min_samples does not match {source} configuration")


def atomic_json(path: str | os.PathLike[str], data: Any) -> None:
    """Atomically persist JSON, leaving an existing report intact on failure."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            json.dump(data, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def require_fields(data: Any, fields: tuple[str, ...] | list[str], context: str) -> None:
    """Raise a contextual error when a JSON object lacks required fields."""
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a JSON object")
    missing = [field for field in fields if field not in data]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _signal_process_group(process: subprocess.Popen[str], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _raise_termination(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"received signal {signum}")


def _stop_process_group(process: subprocess.Popen[str], grace_seconds: float = 2.0) -> None:
    _signal_process_group(process, signal.SIGTERM)
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        try:
            process.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    # The leader can exit while a profiler/application grandchild remains alive.
    _signal_process_group(process, signal.SIGKILL)
    process.wait()


def run_command(
    argv: list[str] | tuple[str, ...],
    *,
    timeout: float,
    stdout: TextIO,
    stderr: TextIO,
) -> dict[str, Any]:
    """Run a command in a new process group and reap its tree on timeout/interruption.

    The returned record includes an exit status even for nonzero commands; callers
    decide whether that status is acceptable. Timeout is enforced over the full child.
    """
    command = [os.fspath(arg) for arg in argv]
    if not command or not command[0]:
        raise ValueError("argv must contain an executable")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number of seconds")
    started = time.monotonic()
    previous_term_handler = signal.signal(signal.SIGTERM, _raise_termination)
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(process)
            returncode = process.returncode
        if returncode != 0 and not timed_out:
            # A failed launcher may leave profiler/application grandchildren behind.
            _stop_process_group(process)
    except BaseException:
        if process is not None:
            _stop_process_group(process)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term_handler)
    return {
        "argv": command,
        "exit_status": returncode,
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
    }
