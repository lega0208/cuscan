"""Capture bounded Phase 0 ANN runs and external memory observations."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path


def gpu_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return [
        [int(value.strip()) for value in line.split(",")] for line in result.stdout.splitlines()
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["cpu", "cuda"])
    parser.add_argument("rows", type=int)
    parser.add_argument("features", type=int)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=float, default=540)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--slots", type=int, default=4194304)
    parser.add_argument("--budgeted", action="store_true")
    parser.add_argument(
        "--executable", type=Path, default=Path("target/release/examples/ann_tile_probe")
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.slots <= 0:
        parser.error("--slots must be positive")
    executable = args.executable.resolve()
    with executable.open("rb") as binary:
        executable_sha256 = hashlib.file_digest(binary, "sha256").hexdigest()
        executable_mtime_ns = os.fstat(binary.fileno()).st_mtime_ns
    args.output.mkdir(parents=True, exist_ok=False)
    command = [
        str(executable),
        args.backend,
        str(args.rows),
        str(args.features),
        str(args.repeats),
        "budget" if args.budgeted else str(args.slots),
    ]
    before = gpu_snapshot() if args.backend == "cuda" else None
    samples = []
    peak_rss = 0
    started = time.monotonic()
    timed_out = False
    with (args.output / "raw.log").open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUSCAN_ANN_PROFILE": "1", "CUSCAN_REQUIRE_CUDA": "1"},
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > args.timeout:
                    timed_out = True
                    process.kill()
                    break
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text()
                except (FileNotFoundError, ProcessLookupError):
                    # The process can die between poll() and the read; a stale
                    # cached dentry fails the read itself with ESRCH.
                    status = ""
                for line in status.splitlines():
                    if line.startswith("VmHWM:"):
                        peak_rss = max(peak_rss, int(line.split()[1]) * 1024)
                if args.backend == "cuda":
                    samples.append({"seconds": elapsed, "gpus": gpu_snapshot()})
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                process.kill()
            returncode = process.wait(timeout=10)
    result = {
        "command": command,
        "executable_sha256": executable_sha256,
        "executable_mtime_ns": executable_mtime_ns,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": time.monotonic() - started,
        "host_peak_rss_bytes_sampled": peak_rss,
        "gpu_before_input_upload": before,
        "samples": samples,
        "limitations": (
            "RSS and GPU memory sampled, may miss transient peaks; GPU includes other processes. "
            "First pass is reported separately by probe. Host timer excludes input "
            "generation/upload. No spill/residency claim."
        ),
    }
    (args.output / "capture.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}))
    if returncode or timed_out:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
