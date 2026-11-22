#!/usr/bin/env -S uv run --no-sync
"""Nsight Systems / Nsight Compute harness for the CUDA ANN pair kernels (rinnd item 4).

One invocation is one profiled worker process for one arm and fixture:

* ``nsys``: one warm-up and two measured builds inside process-wide NVTX ranges
  ``ann:warmup:0``, ``ann:build:1`` and ``ann:build:2``. Kernel records are attributed to builds
  by launch-API timestamp inside the range and joined to launches through ``correlationId``.
* ``ncu``: a single build (the warm-up) with the first launch of one pair kernel profiled.

Identity: ``--identity intake`` (H only) requires all builds of the process to agree and adds
the fixture's entry to the immutable ``head-graphs.json``; ``--identity reference`` checks every
response against it and stops the worker before the measured builds on a warm-up mismatch.
Every binary needs a ``cuscan-native-provenance-v1`` receipt. Output paths are never reused.
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from comparison import native_provenance
    from comparison.ann_rinnd_bench import (
        DEFAULT_DATA,
        IDENTITY_SCHEMA,
        PROFILE_FIELD,
        ROOT,
        IdentityMismatch,
        check_identity,
        fixture_inputs,
        load_identity_reference,
        sha256,
    )
else:
    import native_provenance
    from ann_rinnd_bench import (
        DEFAULT_DATA,
        IDENTITY_SCHEMA,
        PROFILE_FIELD,
        ROOT,
        IdentityMismatch,
        check_identity,
        fixture_inputs,
        load_identity_reference,
        sha256,
    )

SCHEMA = "cuscan-ann-cuda-kernels-v1"
FIXTURES = {"R2": {"k": 29, "seed": 11}, "S3": {"k": 29, "seed": 11}}
CAP = 60
RANGE_NAMES = ("ann:warmup:0", "ann:build:1", "ann:build:2")
# Ordered: the first matching prefix wins, so `forest_pair_slots` precedes `pair_slots`.
KERNEL_GROUPS = (
    ("forest_pair_slots", "forest_pair_slots_kernel"),
    ("pair_slots", "pair_slots_kernel"),
    ("endpoint", "endpoint_"),
    ("sampled_pool", "sampled_pool_kernel"),
)
PAIR_GROUPS = ("forest_pair_slots", "pair_slots")
NCU_KERNELS = {
    "forest_pair_slots": "^forest_pair_slots_kernel_.*$",
    "pair_slots": "^pair_slots_kernel_.*$",
}
NCU_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "WarpStateStats",
    "Occupancy",
    "LaunchStats",
    "SourceCounters",
)
NCU_EXTRA_METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
)
NCU_REPORTED = {
    "duration_us": "gpu__time_duration.sum",
    "sectors_ld": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "requests_ld": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "l1_hit_rate_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_rate_pct": "lts__t_sector_hit_rate.pct",
    "l2_throughput_pct": "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "registers_per_thread": "launch__registers_per_thread",
    "spill_instructions": "sass__inst_executed_register_spilling",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "theoretical_occupancy_pct": "sm__maximum_warps_per_active_cycle_pct",
}
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"


class Inconclusive(RuntimeError):
    """The trace violates an attribution invariant; recorded, never re-run."""


def tool_version(*command: str) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() or completed.stderr.strip() or None


def device_name(backend: str) -> str | None:
    index = backend.removeprefix("cuda:")
    return tool_version(
        "nvidia-smi", f"--id={index}", "--query-gpu=name,driver_version", "--format=csv,noheader"
    )


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


class ProfiledWorker:
    """A worker under a profiler launcher; stdout lines that are not JSON are profiler chatter."""

    def __init__(
        self,
        launcher: list[str],
        binary: Path,
        input_info: dict[str, Any],
        backend: str,
        log_dir: Path,
    ) -> None:
        log_dir.mkdir(parents=True, exist_ok=False)
        self.stderr_path = log_dir / "worker.stderr.log"
        self.stdout_noise_path = log_dir / "launcher.stdout.log"
        self.stderr_stream = self.stderr_path.open("x", encoding="utf-8")
        self.noise = self.stdout_noise_path.open("x", encoding="utf-8")
        env = os.environ.copy()
        env["CUSCAN_ANN_PROFILE"] = "1"
        env["CUSCAN_REQUIRE_CUDA"] = "1"
        command = [
            *launcher,
            str(binary),
            input_info["path"],
            str(input_info["rows"]),
            str(input_info["features"]),
            backend,
        ]
        self.command_line = command
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_stream,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
        )
        self.offsets = [0]

    def command(self, payload: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"worker exited before responding (status={self.process.poll()})"
                )
            if line.startswith("{"):
                break
            self.noise.write(line)
        response = json.loads(line)
        self.stderr_stream.flush()
        self.offsets.append(self.stderr_path.stat().st_size)
        return response

    def close(self) -> int:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.close()
        self.noise.write(self.process.stdout.read())
        status = self.process.wait()
        self.stderr_stream.close()
        self.noise.close()
        return status

    def profile_lines(self) -> list[list[dict[str, Any]]]:
        """ANN_PROFILE lines per command, split at the stderr offsets recorded per response."""
        data = self.stderr_path.read_bytes()
        result = []
        for start, end in zip(self.offsets, self.offsets[1:], strict=False):
            lines = data[start:end].decode("utf-8", errors="replace").splitlines()
            result.append(
                [
                    {"raw": line, "fields": dict(PROFILE_FIELD.findall(line))}
                    for line in lines
                    if "ANN_PROFILE" in line
                ]
            )
        return result


def classify(name: str) -> str:
    for group, prefix in KERNEL_GROUPS:
        if name.startswith(prefix):
            return group
    return "other"


def attribute_trace(database: Path) -> dict[str, Any]:
    """Attribute kernel device time to NVTX build ranges by launch timestamp."""
    connection = sqlite3.connect(database)
    tables = {row[0] for row in connection.execute("select name from sqlite_master")}
    for table in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"):
        if table not in tables:
            raise Inconclusive(f"trace lacks table {table}")
    ranges = connection.execute(
        "select coalesce(e.text, s.value), e.start, e.end from NVTX_EVENTS e "
        "left join StringIds s on s.id = e.textId "
        "where coalesce(e.text, s.value) like 'ann:%' order by e.start"
    ).fetchall()
    names = tuple(row[0] for row in ranges)
    if names != RANGE_NAMES:
        raise Inconclusive(f"NVTX ranges are {names}, expected {RANGE_NAMES}")
    for (_, start, end), (_, next_start, _) in itertools.pairwise(ranges):
        if end is None or start is None or not start < end <= next_start:
            raise Inconclusive("NVTX ranges overlap or are unterminated")
    if ranges[-1][2] is None or ranges[-1][1] >= ranges[-1][2]:
        raise Inconclusive("last NVTX range is unterminated")
    api_tables = [
        table
        for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER")
        if table in tables
    ]
    launches: list[tuple[int, int]] = []
    for table in api_tables:
        launches.extend(
            connection.execute(
                f"select a.correlationId, a.start from {table} a "
                "join StringIds s on s.id = a.nameId where s.value like '%LaunchKernel%' "
                "or s.value like '%LaunchCooperativeKernel%'"
            ).fetchall()
        )
    by_correlation: dict[int, list[int]] = {}
    for correlation, start in launches:
        by_correlation.setdefault(correlation, []).append(start)
    range_of_launch: dict[int, int] = {}
    for correlation, starts in by_correlation.items():
        if len(starts) != 1:
            raise Inconclusive(f"correlationId {correlation} has {len(starts)} launch records")
        containing = [
            index for index, (_, start, end) in enumerate(ranges) if start <= starts[0] <= end
        ]
        if len(containing) != 1:
            raise Inconclusive(f"launch {correlation} lies in {len(containing)} ranges")
        range_of_launch[correlation] = containing[0]
    kernels = connection.execute(
        "select k.correlationId, k.end - k.start, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName"
    ).fetchall()
    groups = [group for group, _ in KERNEL_GROUPS] + ["other"]
    per_range = [
        {"device_ns": dict.fromkeys(groups, 0), "launches": dict.fromkeys(groups, 0)}
        for _ in ranges
    ]
    names_seen: dict[str, set[str]] = {group: set() for group in groups}
    for correlation, duration, name in kernels:
        if correlation not in range_of_launch:
            raise Inconclusive(f"kernel {name} ({correlation}) joins no launch record")
        group = classify(name)
        bucket = per_range[range_of_launch[correlation]]
        bucket["device_ns"][group] += duration
        bucket["launches"][group] += 1
        names_seen[group].add(name)
    for index, bucket in enumerate(per_range):
        for group in PAIR_GROUPS:
            if bucket["launches"][group] == 0:
                raise Inconclusive(f"range {RANGE_NAMES[index]} has no {group} launch")
    return {
        "ranges": [
            {"name": name, "start_ns": start, "end_ns": end, **per_range[index]}
            for index, (name, start, end) in enumerate(ranges)
        ],
        "kernel_names": {group: sorted(values) for group, values in names_seen.items()},
        "launch_records": len(by_correlation),
        "kernel_records": len(kernels),
        "launch_api_tables": api_tables,
    }


def write_intake(
    reference_path: Path,
    fixture: str,
    settings: dict[str, Any],
    responses: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    hashes = {(r["ids_hash_fnv1a64"], r["distances_hash_fnv1a64"]) for r in responses}
    if len(hashes) != 1:
        raise IdentityMismatch(f"K1: intake builds disagree: {sorted(hashes)}")
    (ids_hash, distances_hash) = hashes.pop()
    reference: dict[str, Any]
    if reference_path.exists():
        reference = load_identity_reference(reference_path)
    else:
        reference = {"schema": IDENTITY_SCHEMA, "fixtures": {}}
    if fixture in reference["fixtures"]:
        raise FileExistsError(f"identity reference already has {fixture}: {reference_path}")
    entry = {
        **settings,
        "distance_mode": "unsquared",
        "binary_sha256": provenance["binary_sha256"],
        "source_sha256": provenance["source_sha256"],
        "base_sha": provenance["base_sha"],
        "ids_hash_fnv1a64": ids_hash,
        "distances_hash_fnv1a64": distances_hash,
        "builds_agreeing": len(responses),
    }
    reference["fixtures"][fixture] = entry
    temporary = reference_path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(reference, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(reference_path)
    return entry


def build_payload(op: str, settings: dict[str, Any], slot_policy: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "op": op,
        "k": settings["k_nonself"],
        "seed": settings["seed"],
        "cap": settings["cap"],
        "squared": settings["squared"],
    }
    if slot_policy is not None:
        payload["slot_policy"] = slot_policy
    return payload


def run_nsys(
    args: argparse.Namespace,
    out: Path,
    input_info: dict[str, Any],
    settings: dict[str, Any],
    provenance: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    trace = out / "trace"
    launcher = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "-o",
        str(trace),
    ]
    worker = ProfiledWorker(launcher, args.binary, input_info, settings["backend"], out / "logs")
    responses: list[dict[str, Any]] = []
    k1: str | None = None
    try:
        for op in ("warmup", "build", "build"):
            response = worker.command(build_payload(op, settings, args.slot_policy))
            responses.append(response)
            if not response.get("ok"):
                raise RuntimeError(f"worker failed: {response.get('error')}")
            if args.identity == "reference":
                check_identity(reference, fixture=args.fixture, response=response, **settings)
    except IdentityMismatch as error:
        k1 = str(error)
    finally:
        status = worker.close()
    result: dict[str, Any] = {
        "launcher": worker.command_line,
        "launcher_status": status,
        "responses": responses,
        "profile_lines": worker.profile_lines(),
    }
    if k1 is not None:
        result["status"] = "k1"
        result["k1_event"] = k1
        return result
    if status != 0:
        raise RuntimeError(f"nsys exited with status {status}")
    if args.identity == "intake":
        try:
            result["identity_entry"] = write_intake(
                args.reference, args.fixture, settings, responses, provenance
            )
        except IdentityMismatch as error:
            result["status"] = "k1"
            result["k1_event"] = str(error)
            return result
    report = trace.with_suffix(".nsys-rep")
    database = trace.with_suffix(".sqlite")
    subprocess.run(
        ["nsys", "export", "--type", "sqlite", "-o", str(database), str(report)],
        check=True,
        capture_output=True,
    )
    try:
        attribution = attribute_trace(database)
    except Inconclusive as error:
        result["status"] = "inconclusive"
        result["inconclusive_reason"] = str(error)
        return result
    builds: list[dict[str, Any]] = []
    for index in (1, 2):
        device = attribution["ranges"][index]["device_ns"]
        ann_ms = responses[index]["ann_ms"]
        pair_ms = sum(device[group] for group in PAIR_GROUPS) / 1e6
        builds.append(
            {
                "range": RANGE_NAMES[index],
                "ann_ms": ann_ms,
                "device_ms": {group: value / 1e6 for group, value in device.items()},
                "pair_kernels_ms": pair_ms,
                "pair_share": pair_ms / ann_ms,
            }
        )
    result["status"] = "ok"
    result["attribution"] = attribution
    result["measured_builds"] = builds
    result["median"] = {
        "ann_ms": statistics.median(build["ann_ms"] for build in builds),
        "pair_kernels_ms": statistics.median(build["pair_kernels_ms"] for build in builds),
        "pair_share": statistics.median(build["pair_share"] for build in builds),
        "device_ms": {
            group: statistics.median(build["device_ms"][group] for build in builds)
            for group in builds[0]["device_ms"]
        },
    }
    return result


def parse_ncu_raw(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return []
    header = rows[0]
    # Row 1 holds units; each later row is one profiled launch.
    return [dict(zip(header, row, strict=False)) for row in rows[2:] if row]


def number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def run_ncu(
    args: argparse.Namespace,
    out: Path,
    input_info: dict[str, Any],
    settings: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if shutil.which("ncu") is None:
        return {"status": "ncu missing"}
    pattern = NCU_KERNELS[args.kernel]
    report = out / f"{args.kernel}.ncu-rep"
    launcher = [
        "ncu",
        "--target-processes",
        "all",
        "--kernel-name-base",
        "function",
        "--kernel-name",
        f"regex:{pattern}",
        "--launch-skip",
        "0",
        "--launch-count",
        "1",
        *(item for section in NCU_SECTIONS for item in ("--section", section)),
        "--metrics",
        ",".join(NCU_EXTRA_METRICS),
        "--export",
        str(report),
    ]
    worker = ProfiledWorker(launcher, args.binary, input_info, settings["backend"], out / "logs")
    result: dict[str, Any] = {"launcher": worker.command_line, "kernel_regex": pattern}
    try:
        response = worker.command(build_payload("warmup", settings, args.slot_policy))
    finally:
        status = worker.close()
    result["launcher_status"] = status
    result["responses"] = [response]
    result["profile_lines"] = worker.profile_lines()
    chatter = worker.stdout_noise_path.read_text(encoding="utf-8") + worker.stderr_path.read_text(
        encoding="utf-8"
    )
    if "ERR_NVGPUCTRPERM" in chatter:
        result["status"] = "ncu counters unavailable"
        return result
    if not response.get("ok"):
        raise RuntimeError(f"worker failed: {response.get('error')}")
    try:
        check_identity(reference, fixture=args.fixture, response=response, **settings)
    except IdentityMismatch as error:
        result["status"] = "k1"
        result["k1_event"] = str(error)
        return result
    actual_report = report if report.exists() else report.with_suffix(".ncu-repz")
    if status != 0 or not actual_report.exists():
        result["status"] = "ncu missing"
        result["detail"] = f"ncu exited {status}; report present={actual_report.exists()}"
        return result
    raw = subprocess.run(
        ["ncu", "--import", str(actual_report), "--page", "raw", "--csv"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (out / "raw.csv").write_text(raw, encoding="utf-8")
    launches = parse_ncu_raw(raw)
    regex = re.compile(pattern)
    names = [launch.get("Kernel Name", "") for launch in launches]
    if len(launches) != 1 or regex.fullmatch(names[0]) is None:
        result["status"] = "inconclusive"
        result["inconclusive_reason"] = f"expected one {args.kernel} launch, got {names}"
        return result
    launch = launches[0]
    metrics = {key: number(launch.get(metric)) for key, metric in NCU_REPORTED.items()}
    sectors, requests = metrics["sectors_ld"], metrics["requests_ld"]
    metrics["sectors_per_request"] = (
        sectors / requests if sectors is not None and requests else None
    )
    stalls = {
        name.removeprefix(STALL_PREFIX).removesuffix(STALL_SUFFIX): number(value)
        for name, value in launch.items()
        if name.startswith(STALL_PREFIX) and name.endswith(STALL_SUFFIX)
    }
    top = sorted(
        ((name, value) for name, value in stalls.items() if value is not None),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    memory, compute = metrics["memory_throughput_pct"], metrics["compute_throughput_pct"]
    memory_stalls = {"long_scoreboard", "lg_throttle", "mio_throttle", "drain", "tex_throttle"}
    if memory is None or compute is None or not top:
        bound = "unclassified (missing SOL or stall metrics)"
    elif memory >= compute or top[0][0] in memory_stalls:
        bound = "latency/memory"
    else:
        bound = "issue/ALU"
    result["status"] = "ok"
    result["kernel_function"] = names[0]
    mangled = parse_ncu_raw(
        subprocess.run(
            [
                "ncu",
                "--import",
                str(actual_report),
                "--page",
                "raw",
                "--csv",
                "--print-kernel-base",
                "mangled",
                "--metrics",
                "gpu__time_duration.sum",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    result["mangled_name"] = mangled[0].get("Kernel Name") if len(mangled) == 1 else None
    result["metrics"] = metrics
    result["top_stalls_per_issue"] = top
    result["bound"] = bound
    result["bound_rule"] = (
        "latency/memory when SOL memory % >= compute % or the top per-issue stall is a "
        "memory-dependency stall; issue/ALU otherwise"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("nsys", "ncu"), required=True)
    parser.add_argument("--arm", required=True, help="arm label, for example H")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--fixture", choices=tuple(FIXTURES), required=True)
    parser.add_argument("--identity", choices=("intake", "reference"), required=True)
    parser.add_argument("--reference", type=Path, required=True, help="head-graphs.json")
    parser.add_argument("--kernel", choices=tuple(NCU_KERNELS), help="ncu mode only")
    parser.add_argument("--slot-policy", help="worker slot_policy override")
    parser.add_argument("--backend", default="cuda:0")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.reference = args.reference.resolve()
    if not args.backend.startswith("cuda:"):
        raise ValueError("the pair-kernel harness profiles CUDA only")
    if args.mode == "ncu" and (args.kernel is None or args.identity != "reference"):
        raise ValueError("ncu mode needs --kernel and --identity reference")
    if args.mode == "nsys" and args.kernel is not None:
        raise ValueError("--kernel is only valid in ncu mode")
    if args.mode == "nsys" and shutil.which("nsys") is None:
        raise FileNotFoundError("nsys is required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    provenance = native_provenance.load_receipt(args.provenance, args.binary)
    reference = None
    if args.identity == "reference":
        reference = load_identity_reference(args.reference)
        reference["k1_event_path"] = str(out / "k1-event.json")
    input_info = fixture_inputs(out, args.data.resolve(), (args.fixture,))[args.fixture]
    fixture = FIXTURES[args.fixture]
    settings = {
        "backend": args.backend,
        "input_sha256": input_info["sha256"],
        "k_nonself": fixture["k"],
        "seed": fixture["seed"],
        "cap": CAP,
        "squared": False,
    }
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": args.mode,
        "arm": args.arm,
        "fixture": args.fixture,
        "kernel": args.kernel,
        "identity_mode": args.identity,
        "identity_reference": str(args.reference),
        "slot_policy_override": args.slot_policy,
        "binary": str(args.binary),
        "binary_sha256": provenance["binary_sha256"],
        "provenance_path": str(args.provenance.resolve()),
        "provenance": provenance,
        "input": input_info,
        "settings": settings,
        "device": device_name(args.backend),
        "nsys_version": tool_version("nsys", "--version"),
        "ncu_version": tool_version("ncu", "--version"),
        "worker_env": {"CUSCAN_ANN_PROFILE": "1", "CUSCAN_REQUIRE_CUDA": "1"},
        "harness_sha256": {
            path.name: sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("ann_rinnd_bench.py"),
                Path(__file__).resolve().with_name("native_provenance.py"),
            )
        },
    }
    if args.mode == "nsys":
        receipt.update(run_nsys(args, out, input_info, settings, provenance, reference))
    else:
        receipt.update(run_ncu(args, out, input_info, settings, reference))
    # The input matrix is reproducible from its recorded generator and hash.
    Path(input_info["path"]).unlink()
    write_new(out / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "receipt": str(out / "receipt.json")}))
    return 0 if receipt["status"] in ("ok", "ncu counters unavailable", "ncu missing") else 2


if __name__ == "__main__":
    sys.exit(main())
