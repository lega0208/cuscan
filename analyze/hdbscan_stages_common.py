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
