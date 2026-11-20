"""Run Phase 5 campaign F using prebuilt release tracing examples.

Run from the repository root: python -m comparison.umap06.phase5_profile --out DIR
The output directory must be new. This script never builds either example.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .phase5_probes import ROOT, git, product_sha256, recorded_build_provenance, sha256
from .spec import FIXTURES

ANN_MEMORY = re.compile(r"ANN needs (\d+) bytes(?: of device memory)? but (\d+) are available")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new campaign directory")
    parser.add_argument("--data", type=Path, default=ROOT / "data/umap06")
    parser.add_argument("--build-provenance", type=Path, help="JSON provenance for prebuilt inputs")
    parser.add_argument(
        "--profile-binary",
        type=Path,
        default=ROOT / "target/release/examples/profile_standard_stages",
    )
    parser.add_argument(
        "--synthetic-binary",
        type=Path,
        default=ROOT / "target/release/examples/profile_recursive_init",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    data = args.data.resolve()
    standard = args.profile_binary.resolve()
    synthetic = args.synthetic_binary.resolve()
    for path in (standard, synthetic, data / "manifest.json"):
        if not path.is_file():
            parser.error(f"required prebuilt binary or input manifest not found: {path}")

    sources = [
        ROOT / "umap_06_phase5_plan.md",
        ROOT / "umap_06_port_plan.md",
        ROOT / "umap_06_phase5_measurement_gaps.md",
        data / "manifest.json",
        *sorted((ROOT / "comparison/umap06").glob("*.py")),
    ]
    provenance = {
        "git_revision": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "input_sha256": {str(path): sha256(path) for path in sources},
        "product_sha256": product_sha256(),
        "build_provenance": recorded_build_provenance(args.build_provenance),
        "binaries": {
            "standard": {"path": str(standard), "sha256": sha256(standard)},
            "synthetic": {"path": str(synthetic), "sha256": sha256(synthetic)},
        },
        "python": sys.version,
    }
    out.mkdir(parents=True, exist_ok=False)
    ledger = out / "campaign.jsonl"
    items = []

    def profile(fixture: str, backend: str, init: str, runs: int, *, attempt=False):
        trace = out / f"{fixture}_{backend}_{init}.trace.json"
        spec = FIXTURES[fixture]
        command = [
            str(standard),
            str(data / fixture / "train.f32"),
            str(spec.rows),
            str(spec.features),
            "200",
            "--backend",
            backend,
            "--search",
            "approximate",
            "--init",
            init,
            "--optimizer",
            "adam",
            "--seed",
            "11",
            "--runs",
            str(runs),
            "--trace",
            str(trace),
        ]
        # timeout remains in the campaign process group so the outer deadline kills both.
        wrapped = (
            ["timeout", "--foreground", "--signal=TERM", "--kill-after=2s", "60s", *command]
            if attempt
            else command
        )
        items.append(
            {
                "kind": "10m_attempt" if attempt else "fit",
                "fixture": fixture,
                "backend": backend,
                "init": init,
                "runs": runs,
                "trace": str(trace),
                "command": wrapped,
                "profile_command": command,
            }
        )

    def synthetic_profile(backend: str):
        trace = out / f"synthetic_10m_{backend}.trace.json"
        items.append(
            {
                "kind": "synthetic_10m",
                "fixture": "mix_10m",
                "backend": backend,
                "init": "recursive",
                "runs": 1,
                "trace": str(trace),
                "command": [
                    str(synthetic),
                    "--synthetic-10m",
                    str(data / "mix_10m/train.f32"),
                    "--backend",
                    backend,
                    "--runs",
                    "1",
                    "--trace",
                    str(trace),
                ],
            }
        )

    profile("mix_1m", "cuda", "recursive", 2)
    profile("mix_1m", "cuda", "spectral", 2)
    profile("mix_10m", "cuda", "recursive", 1, attempt=True)
    synthetic_profile("cuda")
    profile("mix_1m", "cpu", "recursive", 1)
    profile("mix_10m", "cpu", "recursive", 1, attempt=True)
    synthetic_profile("cpu")
    for fixture in ("mix_200k", "mix_50k"):
        for backend in ("cuda", "cpu"):
            profile(fixture, backend, "recursive", 1)
    profile("mix_1m", "cpu", "spectral", 1)

    expired = False
    attempts = {}
    for number, item in enumerate(items):
        if expired:
            item["outcome"] = "unrun_deadline"
            continue
        if item["kind"] == "synthetic_10m" and attempts.get(item["backend"]) != "ann_admission":
            item["outcome"] = "not_run_without_ann_admission"
            continue
        binary = synthetic if item["kind"] == "synthetic_10m" else standard
        binary_key = "synthetic" if item["kind"] == "synthetic_10m" else "standard"
        if sha256(binary) != provenance["binaries"][binary_key]["sha256"]:
            item["outcome"] = "binary_changed"
            expired = True
            continue
        wrapper = [
            sys.executable,
            "-m",
            "comparison.umap06.campaign",
            "run",
            "--ledger",
            str(ledger),
            "--budget-seconds",
            "1200",
            "--reserve-seconds",
            "30",
            "--",
            *item["command"],
        ]
        log_path = out / f"command_{number:03d}.log"
        item["log"] = str(log_path)
        with log_path.open("x", encoding="utf-8") as log:
            subprocess.run(wrapper, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        entries = (
            [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            if ledger.is_file()
            else []
        )
        if len(entries) < 2 or entries[-1].get("argv") != item["command"]:
            item["outcome"] = "wrapper_error"
            expired = True
            continue
        entry = entries[-1]
        item["ledger_entry"] = entry
        item["outcome"] = entry["outcome"]
        expired = item["outcome"] == "deadline"
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        item["log_sha256"] = sha256(log_path)
        if sha256(binary) != provenance["binaries"][binary_key]["sha256"]:
            item["outcome"] = "binary_changed"
            expired = True
            continue
        if item["kind"] == "10m_attempt":
            match = ANN_MEMORY.search(log_text)
            if entry["outcome"] == "failed" and entry["exit_status"] == 124:
                item["outcome"] = "attempt_timeout_60s"
            elif entry["outcome"] == "failed" and match and int(match[1]) > int(match[2]):
                item["outcome"] = "ann_admission"
                item["ann_required_bytes"] = int(match[1])
                item["ann_available_bytes"] = int(match[2])
            attempts[item["backend"]] = item["outcome"]
        if entry["outcome"] == "success":
            trace = Path(item["trace"])
            records = []
            for line in log_text.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and (
                    "fit_seconds" in value or "recursive_seconds" in value
                ):
                    records.append(value)
            if not trace.is_file() or len(records) != 1:
                item["outcome"] = "missing_trace_or_timing"
            else:
                item["trace_sha256"] = sha256(trace)
                item["timing"] = records[0]
                if item["kind"] == "synthetic_10m" and (
                    records[0].get("workload") != "synthetic-10m"
                    or records[0].get("end_to_end_fit") is not False
                    or records[0].get("connected_components") != 1
                ):
                    item["outcome"] = "invalid_synthetic_record"
    report = subprocess.run(
        [sys.executable, "-m", "comparison.umap06.campaign", "report", "--ledger", str(ledger)],
        cwd=ROOT,
        check=False,
    )
    manifest = {
        "campaign": "F",
        "provenance": provenance,
        "items": items,
        "campaign_report_exit_status": report.returncode,
        "limitations": [
            "Binary hashes do not prove the binaries were built from the recorded source "
            "or with tracing.",
            "Small-fixture spectral pairs from campaign P are cross-source controls, "
            "not same-source timing.",
            "Synthetic 10M profiles are not end-to-end fit measurements.",
            "Standard profile warm runs use successive seeds starting at 11.",
        ],
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    acceptable = {"success", "ann_admission", "not_run_without_ann_admission"}
    if report.returncode or any(item["outcome"] not in acceptable for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
