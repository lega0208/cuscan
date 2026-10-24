import datetime
import hashlib
import json
import os
import statistics
import struct
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
TOKEN = str(time.time_ns())


def relative(path):
    return str(path.relative_to(REPO))


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, environment, stdout, stderr):
    with stdout.open("wb") as out, stderr.open("wb") as err:
        result = subprocess.run(
            [str(x) for x in command],
            cwd=REPO,
            env={**os.environ, **environment},
            stdout=out,
            stderr=err,
        )
    if result.returncode:
        raise RuntimeError(f"capture failed ({result.returncode}): {command}; stderr={stderr}")


for directory in ("profiles", "semantics"):
    (ROOT / directory).mkdir(exist_ok=True)

captures = []
profile_cases = [
    ("synthetic_n64_d50", 64, 50, 32),
    ("synthetic_n2048_d30", 2048, 30, 32),
    ("synthetic_n4096_d10", 4096, 10, 5),
]
for name, n, d, ms in profile_cases:
    for backend in ("cpu", "cuda"):
        for search in ("brute-force", "kd-tree"):
            for variant in ("original", "current"):
                stem = f"{name}_ms{ms}_{backend}_{search}_{variant}"
                prefix = ROOT / "profiles" / stem
                output = prefix.with_suffix(".json")
                trace = prefix.with_suffix(".trace.json")
                snapshot = prefix.with_suffix(".snapshot")
                env = {
                    "CUBECL_ENVIRONMENT": f"g006-{TOKEN}-{stem}",
                    "CUBECL_ENVIRONMENT_RECORDS": "basic",
                }
                for flag in (
                    "CUBECL_DEBUG_LOG",
                    "CUBECL_DEBUG_OPTION",
                    "FAST_HDBSCAN_KD_DIAGNOSTICS",
                ):
                    if flag in os.environ:
                        raise RuntimeError(f"unexpected inherited instrumentation setting: {flag}")
                command = [
                    ROOT / "bin" / f"{variant}-instrumented",
                    ROOT / "inputs" / f"{name}.raw",
                    n,
                    d,
                    "--backend",
                    backend,
                    "--runs",
                    3,
                    "--min-cluster-size",
                    5,
                    "--min-samples",
                    ms,
                    "--search",
                    search,
                    "--snapshot-output",
                    snapshot,
                    "--trace-output",
                    trace,
                ]
                started = datetime.datetime.now(datetime.timezone.utc).isoformat()
                run(command, env, output, prefix.with_suffix(".stderr"))
                profile = json.loads(output.read_text())
                assert len(profile["fit_seconds"]) == 3
                assert len(profile["recovery_stage_seconds"]["warm_fits"]) == 3
                assert profile["recovery_stage_seconds"]["cold_first_fit"]["memory_events"]
                row = {
                    "variant": variant,
                    "backend": backend,
                    "search": search,
                    "case": {
                        "name": name,
                        "n": n,
                        "d": d,
                        "min_samples": ms,
                        "min_cluster_size": 5,
                    },
                    "profile_json": relative(output),
                    "trace_json": relative(trace),
                    "snapshot": relative(snapshot),
                    "env": env,
                    "started_at": started,
                    "command": [str(x) for x in command],
                    "sha256": {relative(path): sha(path) for path in (output, trace, snapshot)},
                }
                captures.append(row)
                save_json(
                    ROOT / "captures.json",
                    {
                        "provenance": {
                            "retrospective": True,
                            "original_head": "d4762aa2fc39ced947e2519b77ff2e862096c92a",
                            "cold": (
                                "First fit in a fresh process with a unique CubeCL environment; "
                                "includes backend/context setup and CubeCL JIT. CUDA driver PTX "
                                "cache and hardware caches are not cleared."
                            ),
                            "warm": (
                                "Three subsequent fits in the same process; "
                                "tracing/synchronization/allocator-observation overhead remains. "
                                "Not a replacement for "
                                "immutable uninstrumented Criterion distributions."
                            ),
                            "records": (
                                "basic; execution capture is Chrome tracing; physical PCIe bytes/"
                                "true peak VRAM are not measured"
                            ),
                        },
                        "captures": captures,
                    },
                )
                print(
                    "PROFILE",
                    stem,
                    "cold",
                    profile["cold_first_fit_seconds"],
                    "warm_median",
                    statistics.median(profile["fit_seconds"]),
                    flush=True,
                )

# Additional semantic inputs deliberately include non-finite rows, which the stage CLI rejects.
small = {
    "singleton_n1_d2": [(0.0, 0.0)],
    "two_n2_d2": [(0.0, 0.0), (3.0, 4.0)],
    "finite_remap_n7_d2": [
        (0.0, 0.0),
        (float("nan"), 2.0),
        (1.0, 1.0),
        (float("inf"), 0.0),
        (-1.0, -1.0),
        (2.0, -2.0),
        (-float("inf"), 3.0),
    ],
}
for name, matrix in small.items():
    (ROOT / "inputs" / f"{name}.raw").write_bytes(
        b"".join(struct.pack("<f", value) for row in matrix for value in row)
    )
semantic_cases = (
    profile_cases
    + [("synthetic_n257_d2", 257, 2, ms) for ms in (1, 32, 256, 257, 262)]
    + [("duplicates_n9_d2", 9, 2, ms) for ms in (1, 8, 9, 14)]
    + [("overflow_n3_d2", 3, 2, ms) for ms in (1, 2, 3, 8)]
    + [("singleton_n1_d2", 1, 2, ms) for ms in (1, 6)]
    + [("two_n2_d2", 2, 2, ms) for ms in (1, 2, 7)]
    + [("finite_remap_n7_d2", 7, 2, ms) for ms in (1, 3, 4, 9)]
)
comparisons = []
for name, n, d, ms in semantic_cases:
    for backend in ("cpu", "cuda"):
        for search in ("brute-force", "kd-tree"):
            snapshots = {}
            for variant in ("original-pristine", "original-instrumented", "current-instrumented"):
                stem = f"{name}_ms{ms}_{backend}_{search}_{variant}"
                snapshot = ROOT / "semantics" / f"{stem}.snapshot"
                env = {
                    "CUBECL_ENVIRONMENT": f"g006-semantics-{TOKEN}-{variant}-{backend}",
                    "CUBECL_ENVIRONMENT_RECORDS": "basic",
                }
                command = [
                    ROOT / "bin" / f"{variant}-semantics",
                    ROOT / "inputs" / f"{name}.raw",
                    n,
                    d,
                    ms,
                    backend,
                    search,
                    5 if (name, n, d, ms) in profile_cases else 2,
                ]
                run(command, env, snapshot, snapshot.with_suffix(".stderr"))
                snapshots[variant] = {
                    "path": relative(snapshot),
                    "sha256": sha(snapshot),
                    "bytes": snapshot.stat().st_size,
                    "env": env,
                }
            equal = len({record["sha256"] for record in snapshots.values()}) == 1
            comparisons.append(
                {
                    "case": name,
                    "n": n,
                    "d": d,
                    "min_samples": ms,
                    "backend": backend,
                    "search": search,
                    "equal_pristine_instrumented_current": equal,
                    "snapshots": snapshots,
                }
            )
            save_json(
                ROOT / "semantic-comparison.json",
                {
                    "retrospective": True,
                    "snapshot_format": (
                        "Complete fitted Debug, plus explicit f32/f64 IEEE bit vectors for core "
                        "distances, probabilities, linkage/MST weights, condensed lambda/child "
                        "sizes. No output normalization or tolerance."
                    ),
                    "comparisons": comparisons,
                    "all_equal": all(
                        row["equal_pristine_instrumented_current"] for row in comparisons
                    ),
                    "expected_comparison_count": len(semantic_cases) * 4,
                },
            )
            if not equal:
                raise RuntimeError(
                    f"exact semantic mismatch: {name} ms{ms} {backend} {search}; "
                    "preserved raw snapshots"
                )
    print("SEMANTICS", name, ms, "all variants/backends/routes exact", flush=True)
print(
    "COMPLETE",
    len(captures),
    "profiles;",
    len(comparisons),
    "three-way exact semantic comparisons",
    flush=True,
)
