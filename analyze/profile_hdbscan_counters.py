#!/usr/bin/env python3
"""Capture representative HDBSCAN CUDA kernel counters with Nsight Compute."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import sys
from pathlib import Path
from typing import Any

from hdbscan_stages_common import atomic_json, identity, run_command

METRICS = (
    "gpu__time_duration.sum",
    "launch__grid_size",
    "launch__block_size",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
)


def _number(value: str) -> float:
    cleaned = value.strip().replace(",", "")
    try:
        result = float(cleaned)
    except ValueError as error:
        raise ValueError(f"invalid numeric Nsight metric value {value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"invalid numeric Nsight metric value {value!r}")
    return result


def normalize_metric(name: str, value: float, unit: str) -> tuple[float, str]:
    unit = unit.strip()
    if name == "gpu__time_duration.sum":
        factors = {
            "ns": (1e-9, "s"),
            "us": (1e-6, "s"),
            "µs": (1e-6, "s"),
            "ms": (1e-3, "s"),
            "s": (1.0, "s"),
        }
        if unit not in factors:
            raise ValueError(f"unsupported unit {unit!r} for {name}")
        factor, normalized = factors[unit]
        return value * factor, normalized
    if name.startswith("launch__"):
        if unit not in ("", "#", "count"):
            raise ValueError(f"unsupported unit {unit!r} for {name}")
        if value <= 0 or not value.is_integer():
            raise ValueError(f"{name} must be a positive integer")
        return value, "count"
    if unit not in ("%", "percent"):
        raise ValueError(f"unsupported unit {unit!r} for {name}")
    if value > 100:
        raise ValueError(f"percentage metric {name} exceeds 100%")
    return value, "%"


def parse_ncu_csv(text: str, kernel_family: str) -> dict[str, Any]:
    lines = text.splitlines()
    header_index = next(
        (
            i
            for i, line in enumerate(lines)
            if len(next(csv.reader([line]))) >= 15
            and next(csv.reader([line]))[0].strip().strip('"') == "ID"
            and "Metric Name" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute CSV header was not found")
    records = list(
        csv.DictReader(io.StringIO("\n".join(lines[header_index:])), skipinitialspace=True)
    )
    records = [r for r in records if kernel_family in (r.get("Kernel Name") or "")]
    if not records:
        raise ValueError(f"no Nsight Compute launch matched {kernel_family}")
    ids = {(r.get("ID", "").strip(), r.get("Kernel Name", "").strip()) for r in records}
    if len(ids) != 1:
        raise ValueError(
            f"ambiguous Nsight Compute capture for {kernel_family}: {len(ids)} launch records"
        )
    launch_id, kernel_name = next(iter(ids))
    if not launch_id or not kernel_name:
        raise ValueError(f"{kernel_family} record lacks kernel name or launch ID")
    by_metric: dict[str, list[dict[str, str]]] = {}
    for record in records:
        name = (record.get("Metric Name") or "").strip()
        if name in METRICS:
            by_metric.setdefault(name, []).append(record)
    parsed: dict[str, Any] = {}
    for name in METRICS:
        found = by_metric.get(name, [])
        if len(found) != 1:
            raise ValueError(
                f"expected exactly one {name} record for {kernel_family}, found {len(found)}"
            )
        rec = found[0]
        value = _number(rec.get("Metric Value") or "")
        unit = (rec.get("Metric Unit") or "").strip()
        normalized, normalized_unit = normalize_metric(name, value, unit)
        parsed[name] = {
            "requested": True,
            "observed": True,
            "value": value,
            "unit": unit,
            "normalized_value": normalized,
            "normalized_unit": normalized_unit,
        }
    geometry = {"block": records[0].get("Block Size"), "grid": records[0].get("Grid Size")}
    if not geometry["block"] or not geometry["grid"]:
        raise ValueError(f"{kernel_family} record lacks launch geometry")
    if any(
        (r.get("Block Size"), r.get("Grid Size"))
        != (records[0].get("Block Size"), records[0].get("Grid Size"))
        for r in records
    ):
        raise ValueError(f"inconsistent geometry for {kernel_family} launch {launch_id}")
    return {
        "kernel_name": kernel_name,
        "launch_id": launch_id,
        "geometry": geometry,
        "metrics": parsed,
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
    report_path = args.run_dir / "cuda_counters.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report {report_path}")
    input_path = raw / "input.f32le"
    if ident["workload"].get("rows") != args.rows:
        raise ValueError("--rows does not match the run manifest workload")
    if (
        not input_path.is_file()
        or hashlib.sha256(input_path.read_bytes()).hexdigest() != ident["input_sha256"]
    ):
        raise ValueError("raw input is missing or does not match the run manifest SHA-256")
    reports: dict[str, Any] = {}
    captures: dict[str, Any] = {}
    version_stdout, version_stderr = raw / "ncu_version.stdout", raw / "ncu_version.stderr"
    if version_stdout.exists() or version_stderr.exists():
        raise FileExistsError("refusing to overwrite existing Nsight Compute version evidence")
    with (
        version_stdout.open("w", encoding="utf-8") as stdout,
        version_stderr.open("w", encoding="utf-8") as stderr,
    ):
        version_record = run_command(["ncu", "--version"], timeout=10, stdout=stdout, stderr=stderr)
    if version_record["timed_out"] or version_record["exit_status"] != 0:
        raise RuntimeError("unable to query Nsight Compute version")
    ncu_version = {
        "command": version_record["argv"],
        "stdout": version_stdout.read_text(encoding="utf-8").strip(),
        "stderr": version_stderr.read_text(encoding="utf-8").strip(),
    }
    for family in ("partial_kernel", "tile_candidates"):
        log = raw / f"ncu_{family}.csv"
        stdout = raw / f"ncu_{family}.stdout"
        stderr = raw / f"ncu_{family}.stderr"
        stale = [path for path in (log, stdout, stderr) if path.exists()]
        if stale:
            raise FileExistsError(f"refusing to overwrite profiler evidence: {stale[0]}")
        command = [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name",
            f"regex:{family}",
            "--launch-count",
            "1",
            "--metrics",
            ",".join(METRICS),
            "--csv",
            "--log-file",
            str(log),
            str(args.binary),
            str(input_path),
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
            str(raw / f"ncu_{family}.trace.json"),
        ]
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            provenance = run_command(command, timeout=args.timeout, stdout=out, stderr=err)
        if provenance["timed_out"] or provenance["exit_status"] != 0:
            raise RuntimeError(f"Nsight Compute capture failed for {family}; diagnostics: {stderr}")
        record = parse_ncu_csv(log.read_text(encoding="utf-8", errors="replace"), family)
        record["command"] = provenance["argv"]
        record["sampling_scope"] = (
            "First matching kernel launch, intentionally the excluded warmup fit; "
            "not the measured-fit "
            "average or all Boruvka rounds. Replay/cache behavior follows Nsight Compute defaults; "
            "the exact replay mode is unspecified."
        )
        reports[family] = record
        captures[family] = provenance
    atomic_json(
        report_path,
        {
            **ident,
            "kernels": reports,
            "captures": captures,
            "profiler_tools": {"ncu": ncu_version},
            "caveats": [
                "Representative first matching warmup launch only.",
                "Counter collection can perturb execution; values are not measured-fit averages.",
            ],
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"profile_hdbscan_counters: {error}", file=sys.stderr)
        raise SystemExit(1) from error
