"""Run the capped Phase 5 predecessor (P) or regression (R) probes.

Run from the harness checkout with the extension under test installed in the
current Python environment. P also needs the release tracing profile binary.
Each invocation requires a fresh output directory and uses one campaign ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .spec import FIXTURES

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def product_sha256() -> dict[str, str]:
    """Hash the current product tree, including nonignored untracked sources."""
    output = subprocess.run(
        [
            "git",
            "ls-files",
            "-c",
            "-o",
            "--exclude-standard",
            "-z",
            "--",
            "crates",
            "python",
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "uv.lock",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    names = sorted(set(os.fsdecode(name) for name in output.split(b"\0") if name))
    excluded_dirs = {"target", "__pycache__", ".pytest_cache", ".ruff_cache"}
    excluded_suffixes = {".pyc", ".pyo", ".so", ".dylib", ".dll"}
    return {
        name: sha256(ROOT / name)
        for name in names
        if not excluded_dirs.intersection(Path(name).parts)
        and Path(name).suffix not in excluded_suffixes
        and (ROOT / name).is_file()
    }


def recorded_build_provenance(path: Path | None) -> dict | None:
    """Record an explicit provenance input without claiming it proves a build."""
    if path is None:
        return None
    path = path.resolve()
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "record": json.loads(content),
    }


def probe_matrix() -> list[dict]:
    probes = []

    def add(fixture, optimizer, init="spectral", **options):
        probes.append({"fixture": fixture, "optimizer": optimizer, "init": init, **options})

    for fixture in ("blobs_1k", "curve_20k", "mix_50k"):
        for optimizer in ("adam", "momentum"):
            options = {}
            if fixture == "mix_50k":
                options["negative_selection_range"] = 10000
            if fixture == "curve_20k":
                options["transform_self"] = True
            add(fixture, optimizer, **options)
    for optimizer in ("adam", "momentum"):
        variants = [
            {"negative_sample_scale": 0.5},
            *(
                [{"negative_sample_scale_adaptation_samples": value} for value in (0, 64)]
                if optimizer == "adam"
                else []
            ),
            {"exclude_graph_neighbors": True},
        ]
        for variant in variants:
            add("mix_50k", optimizer, negative_selection_range=10000, **variant)
    for fixture in ("digits", "curve_20k"):
        for init in ("spectral", "random"):
            add(fixture, "compatibility", init)
    add("curve_20k", "compatibility", "random", init_rescale="graph_edge")
    for components in (16, 48):
        add("digits", "adam", components=components)
    add("blobs_1k", "adam", "recursive")
    add("curve_20k", "adam", "recursive", transform_self=True)
    add("mix_50k", "adam", "recursive", negative_selection_range=10000)
    add("curve_20k", "momentum", "recursive")
    add("curve_20k", "adam", "auto")
    for optimizer in ("adam", "momentum"):
        add("mix_1m", optimizer)
    add("mix_1m", "adam", "recursive")
    return probes


def worker_command(probe: dict, backend: str, out: Path, data: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "comparison.umap06.cuscan_worker",
        "--fixture",
        probe["fixture"],
        "--backend",
        backend,
        *(
            ("--library-defaults",)
            if probe.get("library_defaults")
            else ("--init", probe["init"], "--optimizer", probe["optimizer"])
        ),
        "--seeds",
        "11",
        "--out",
        str(out),
        "--data",
        str(data),
        "--skip-repeat-fit",
    ]
    batches = FIXTURES[probe["fixture"]].batches
    if (
        probe["fixture"] == "mix_1m"
        or (
            probe["fixture"] == "mix_50k"
            and any(
                key in probe
                for key in (
                    "negative_sample_scale",
                    "negative_sample_scale_adaptation_samples",
                    "exclude_graph_neighbors",
                )
            )
        )
        or probe.get("optimizer") == "compatibility"
        or "components" in probe
    ):
        batches = batches[:1]
    command.extend(("--transform-batches", *(str(batch) for batch in batches)))
    for key in (
        "negative_selection_range",
        "negative_sample_scale",
        "negative_sample_scale_adaptation_samples",
        "exclude_graph_neighbors",
        "init_rescale",
        "components",
    ):
        if key in probe:
            value = probe[key]
            command.extend(("--" + key.replace("_", "-"), str(value).lower()))
    if probe.get("transform_self"):
        command.append("--transform-self")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=("P", "R"))
    parser.add_argument("--out", type=Path, required=True, help="new campaign directory")
    parser.add_argument("--data", type=Path, default=ROOT / "data/umap06")
    parser.add_argument("--build-provenance", type=Path, help="JSON provenance for prebuilt inputs")
    parser.add_argument(
        "--profile-binary",
        type=Path,
        default=ROOT / "target/release/examples/profile_standard_stages",
    )
    args = parser.parse_args()
    out = args.out.resolve()
    data = args.data.resolve()
    binary = args.profile_binary.resolve()
    spec = importlib.util.find_spec("cuscan._cuscan")
    if spec is None or not spec.origin:
        parser.error("the cuscan extension under test must be installed")
    extension = Path(spec.origin).resolve()
    if args.campaign == "P" and not binary.is_file():
        parser.error("P requires the prebuilt release tracing profile binary")
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
        "extension": {"path": str(extension), "sha256": sha256(extension)},
        "input_sha256": {str(path): sha256(path) for path in sources},
        "product_sha256": product_sha256(),
        "build_provenance": recorded_build_provenance(args.build_provenance),
        "profile_binary": (
            {"path": str(binary), "sha256": sha256(binary)} if args.campaign == "P" else None
        ),
        "python": sys.version,
    }
    out.mkdir(parents=True, exist_ok=False)
    ledger = out / "campaign.jsonl"
    items: list[dict[str, Any]] = []
    matrix = probe_matrix()
    # Profiles are before the 1M tier: the latter is always the final work.
    for index, probe in enumerate(matrix):
        if (
            probe["fixture"] == "mix_1m"
            and args.campaign == "P"
            and not any(item["kind"] == "profile" for item in items)
        ):
            for fixture in ("mix_200k", "mix_50k"):
                for backend in ("cuda", "cpu"):
                    trace = out / f"profile_{fixture}_{backend}.json"
                    command = [
                        str(binary),
                        str(data / fixture / "train.f32"),
                        str(FIXTURES[fixture].rows),
                        str(FIXTURES[fixture].features),
                        "200",
                        "--backend",
                        backend,
                        "--search",
                        "approximate",
                        "--init",
                        "spectral",
                        "--optimizer",
                        "adam",
                        "--seed",
                        "11",
                        "--runs",
                        "2",
                        "--trace",
                        str(trace),
                    ]
                    items.append(
                        {
                            "kind": "profile",
                            "fixture": fixture,
                            "backend": backend,
                            "command": command,
                            "trace": str(trace),
                        }
                    )
        for backend in ("cuda", "cpu"):
            repeats = 2 if args.campaign == "P" and probe["optimizer"] == "compatibility" else 1
            for replay in range(repeats):
                destination = out / "probes" / f"{index:02d}_{backend}_{replay}"
                items.append(
                    {
                        "kind": "probe",
                        "identity": probe,
                        "backend": backend,
                        "replay": replay,
                        "directory": str(destination),
                        "command": worker_command(probe, backend, destination, data),
                    }
                )
    expired = False
    for number, item in enumerate(items):
        if expired:
            item["outcome"] = "unrun_deadline"
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
        with (out / f"command_{number:03d}.log").open("x", encoding="utf-8") as log:
            subprocess.run(wrapper, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        entries = (
            [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
            if ledger.is_file()
            else []
        )
        if len(entries) < 2 or entries[-1].get("argv") != item["command"]:
            item["outcome"] = "wrapper_error"
            expired = True
        else:
            item["outcome"] = entries[-1]["outcome"]
            expired = item["outcome"] == "deadline"
        if item["kind"] == "probe" and item["outcome"] == "success":
            record = Path(item["directory"]) / "11/run.json"
            if record.is_file():
                item["record_sha256"] = sha256(record)
                item["hashes"] = json.loads(record.read_text(encoding="utf-8"))["hashes"]
            else:
                item["outcome"] = "missing_record"
        elif item["kind"] == "profile" and item["outcome"] == "success":
            trace = Path(item["trace"])
            if trace.is_file():
                item["trace_sha256"] = sha256(trace)
            else:
                item["outcome"] = "missing_trace"
    report = subprocess.run(
        [sys.executable, "-m", "comparison.umap06.campaign", "report", "--ledger", str(ledger)],
        cwd=ROOT,
        check=False,
    )
    replay_stability = []
    for item in items:
        if (
            item["kind"] != "probe"
            or item["replay"] != 0
            or item["identity"]["optimizer"] != "compatibility"
            or args.campaign != "P"
        ):
            continue
        other = next(
            (
                candidate
                for candidate in items
                if candidate["kind"] == "probe"
                and candidate["identity"] == item["identity"]
                and candidate["backend"] == item["backend"]
                and candidate["replay"] == 1
            ),
            None,
        )
        first, second = item.get("hashes"), other.get("hashes") if other else None
        replay_stability.append(
            {
                "identity": item["identity"],
                "backend": item["backend"],
                "hashes": (
                    {key: first[key] == second[key] for key in first.keys() & second.keys()}
                    if first is not None and second is not None
                    else None
                ),
            }
        )
    manifest = {
        "campaign": args.campaign,
        "provenance": provenance,
        "items": items,
        "classic_replay_stability": replay_stability,
        "campaign_report_exit_status": report.returncode,
        "limitations": [
            "The installed extension and profile binary must be built from the intended source; "
            "their hashes alone do not establish that relationship.",
            "Profile pairs on P are cross-source controls for candidate timings.",
        ],
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if report.returncode or any(item["outcome"] != "success" for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
