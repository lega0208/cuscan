#!/usr/bin/env python3
"""Capture and attribute CUDA kernels for one HDBSCAN measured fit with Nsight Systems."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

from benchmark_hdbscan_stages import _validate_profile
from hdbscan_stages_common import atomic_json, identity, run_command

MARKER = re.compile(r"profile_run_(start|end)\s+run=(\d+)\s+unix_ns=(\d+)")


def parse_markers(stderr: str) -> tuple[int, int]:
    markers = [(m.group(1), int(m.group(2)), int(m.group(3))) for m in MARKER.finditer(stderr)]
    if len(markers) != 2 or [(m[0], m[1]) for m in markers] != [("start", 0), ("end", 0)]:
        raise ValueError("expected exactly one ordered profile_run_start/end run=0 marker pair")
    start, end = markers[0][2], markers[1][2]
    if start <= 0 or end <= start:
        raise ValueError("profile marker timestamps are invalid or unordered")
    return start, end


def select_profile(stdout: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    profiles: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", stdout):
        try:
            value, _ = decoder.raw_decode(stdout[match.start() :])
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and {
                "fit_seconds",
                "stage_seconds",
                "warmup_seconds",
                "backend",
                "boruvka_samples",
                "core_distance_samples",
            }
            <= value.keys()
        ):
            profiles.append(value)
    if len(profiles) != 1:
        raise ValueError(f"expected exactly one native profile JSON object, found {len(profiles)}")
    for key in ("fit_seconds", "stage_seconds", "boruvka_samples", "warmup_seconds"):
        if key not in profiles[0]:
            raise ValueError(f"native profile is missing {key}")
    profile = profiles[0]
    if not isinstance(profile["fit_seconds"], list) or len(profile["fit_seconds"]) != 1:
        raise ValueError("profiled capture must contain exactly one measured fit")
    if not isinstance(profile["stage_seconds"], dict) or not profile["stage_seconds"]:
        raise ValueError("native profile must contain stage timing samples")
    if not isinstance(profile["boruvka_samples"], list) or len(profile["boruvka_samples"]) != 1:
        raise ValueError("profiled capture must contain one Boruvka sample")
    if (
        not isinstance(profile.get("core_distance_samples"), list)
        or len(profile["core_distance_samples"]) != 1
    ):
        raise ValueError("profiled capture must contain one core-distance sample")
    for name, samples in profile["stage_seconds"].items():
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(f"profiled stage {name} must contain exactly one sample")
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v < float("inf")
            for v in samples
        ):
            raise ValueError(f"profiled stage {name} contains an invalid duration")
    for name in ("fit_seconds", "warmup_seconds"):
        values = profile[name] if isinstance(profile[name], list) else [profile[name]]
        if any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 < v < float("inf")
            for v in values
        ):
            raise ValueError(f"native profile contains invalid {name}")
    return _validate_profile(profile, "cuda", runs=1)


def aggregate_kernels(db_path: str | Path, start_ns: int, end_ns: int) -> dict[str, Any]:
    if start_ns <= 0 or end_ns <= start_ns:
        raise ValueError("invalid measured interval")
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"TARGET_INFO_SESSION_START_TIME", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
        if not required <= tables:
            raise ValueError(
                f"Nsight SQLite missing tables: {', '.join(sorted(required - tables))}"
            )
        epoch_rows = db.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME").fetchall()
        if len(epoch_rows) != 1 or epoch_rows[0][0] is None:
            raise ValueError("Nsight SQLite must contain one session start epoch")
        epoch = int(epoch_rows[0][0])
        columns = {r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        needed = {
            "start",
            "end",
            "demangledName",
            "gridX",
            "gridY",
            "gridZ",
            "blockX",
            "blockY",
            "blockZ",
        }
        if not needed <= columns:
            raise ValueError("Nsight kernel table has incompatible columns")
        rows = db.execute(
            "SELECT k.start,k.end,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName "
            "ORDER BY k.start"
        ).fetchall()
    win_start, win_end = start_ns - epoch, end_ns - epoch
    if win_start < 0 or win_end <= win_start:
        raise ValueError("profile markers cannot be aligned to Nsight session clock")
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    straddling = 0
    for s, e, name, gx, gy, gz, bx, by, bz in rows:
        if e <= s:
            raise ValueError("Nsight kernel activity has invalid duration")
        if s < win_end and e > win_start and not (s >= win_start and e <= win_end):
            straddling += 1
        if s < win_start or e > win_end:
            continue
        geometry = {"block": [bx, by, bz], "grid": [gx, gy, gz]}
        known_families = (
            "partial_kernel",
            "tile_candidates",
            "kd_neighbors_kernel",
            "kd_candidates_kernel",
            "reset_component_bounds",
            "stage_pass",
            "tile_pass",
            "reduce_tiles",
            "finish_pass",
            "merge_kernel",
            "copy_rows_kernel",
            "slice_kernel",
            "elemwise_fuse",
        )
        family = next((f for f in known_families if str(name).startswith(f)), "unknown")
        key = str(name)
        item = selected.setdefault(
            key,
            {
                "name": str(name),
                "family": family,
                "launch_count": 0,
                "device_seconds": 0.0,
                "geometry": [],
                "launches": [],
            },
        )
        duration = e - s
        item["launch_count"] += 1
        item["device_seconds"] += duration / 1e9
        existing = next(
            (
                entry
                for entry in item["geometry"]
                if entry["block"] == geometry["block"] and entry["grid"] == geometry["grid"]
            ),
            None,
        )
        if existing is None:
            item["geometry"].append({**geometry, "count": 1})
        else:
            existing["count"] += 1
        item["launches"].append({"start_ns": s, "end_ns": e, "duration_ns": duration})
    if not selected:
        raise ValueError("no kernels fully contained in the measured-fit interval")
    kernels = list(selected.values())
    for item in kernels:
        item["mean_seconds"] = item["device_seconds"] / item["launch_count"]
    intervals = sorted((x["start_ns"], x["end_ns"]) for k in kernels for x in k["launches"])
    overlap = any(right[0] < left[1] for left, right in pairwise(intervals))
    return {
        "window": {
            "start_unix_ns": start_ns,
            "end_unix_ns": end_ns,
            "start_ns": win_start,
            "end_ns": win_end,
        },
        "kernels": kernels,
        "total_device_seconds": sum(k["device_seconds"] for k in kernels),
        "boundary_straddling_count": straddling,
        "overlapping_activity": overlap,
    }


def _version(executable: str, raw: Path) -> dict[str, Any]:
    stdout_path, stderr_path = (
        raw / f"{executable}_version.stdout",
        raw / f"{executable}_version.stderr",
    )
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        record = run_command([executable, "--version"], timeout=10, stdout=stdout, stderr=stderr)
    if record["timed_out"] or record["exit_status"] != 0:
        raise RuntimeError(f"unable to query {executable} version")
    return {
        "command": record["argv"],
        "stdout": stdout_path.read_text(encoding="utf-8").strip(),
        "stderr": stderr_path.read_text(encoding="utf-8").strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=int, choices=(20000, 100000))
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    args = parser.parse_args()
    raw = args.run_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    ident = identity(args.run_dir)
    if ident["workload"].get("rows") != args.rows:
        raise ValueError("--rows does not match the run manifest workload")
    report_path = args.run_dir / "cuda_kernels.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report {report_path}")
    inp, trace = raw / "input.f32le", raw / "nsys_trace.json"
    rep, sqlite_path = raw / "cuda.nsys-rep", raw / "cuda.sqlite"
    stdout_path, stderr_path = raw / "nsys.stdout", raw / "nsys.stderr"
    if not inp.is_file() or hashlib.sha256(inp.read_bytes()).hexdigest() != ident["input_sha256"]:
        raise ValueError("raw input is missing or does not match the run manifest SHA-256")
    version_paths = (raw / "nsys_version.stdout", raw / "nsys_version.stderr")
    stale = [
        p for p in (rep, sqlite_path, stdout_path, stderr_path, trace, *version_paths) if p.exists()
    ]
    if stale:
        raise FileExistsError(f"refusing to overwrite profiler evidence: {stale[0]}")
    nsys_version = _version("nsys", raw)
    command = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda",
        "--sample=none",
        "--cpuctxsw=none",
        "-o",
        str(rep.with_suffix("")),
        str(args.binary),
        str(inp),
        str(args.rows),
        "16",
        "--backend",
        "cuda",
        "--runs",
        "1",
        "--min-cluster-size",
        "5",
        "--min-samples",
        "5",
        "--search",
        "auto",
        "--trace-output",
        str(trace),
    ]
    with (
        stdout_path.open("w", encoding="utf-8") as out,
        stderr_path.open("w", encoding="utf-8") as err,
    ):
        provenance = run_command(command, timeout=args.timeout, stdout=out, stderr=err)
    if provenance["timed_out"] or provenance["exit_status"] != 0:
        raise RuntimeError(f"Nsight Systems capture failed; diagnostics: {stderr_path}")
    marker_pair = parse_markers(stderr_path.read_text(encoding="utf-8"))
    profile = select_profile(stdout_path.read_text(encoding="utf-8"))
    export = [
        "nsys",
        "export",
        "--type=sqlite",
        "--force-overwrite=true",
        "--output",
        str(sqlite_path),
        str(rep),
    ]
    with (
        (raw / "nsys_export.stdout").open("w", encoding="utf-8") as out,
        (raw / "nsys_export.stderr").open("w", encoding="utf-8") as err,
    ):
        export_provenance = run_command(export, timeout=60, stdout=out, stderr=err)
    if export_provenance["timed_out"] or export_provenance["exit_status"] != 0:
        raise RuntimeError("Nsight SQLite export failed")
    attribution = aggregate_kernels(sqlite_path, *marker_pair)
    atomic_json(
        report_path,
        {
            **ident,
            "profiled_fit": {**attribution, "native_profile": profile},
            "capture": provenance,
            "export": export_provenance,
            "profiler_tools": {"nsys": nsys_version},
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        print(f"profile_hdbscan_kernels: {error}", file=sys.stderr)
        raise SystemExit(1) from error
