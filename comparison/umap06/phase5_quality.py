"""Run the capped Phase 5 quality campaign in a fresh output directory.

The extension must already be built and installed. Run from the harness checkout;
all fit, scoring and receipt commands share the campaign.py ledger.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .phase5_probes import ROOT, git, product_sha256, recorded_build_provenance, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new campaign directory")
    parser.add_argument("--data", type=Path, default=ROOT / "data/umap06")
    parser.add_argument("--build-provenance", type=Path, help="JSON provenance for prebuilt inputs")
    parser.add_argument("--reference", type=Path, default=ROOT / "comparison/umap06/reference")
    parser.add_argument("--upstream-reference", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = args.data.resolve()
    reference = args.reference.resolve()
    upstream = args.upstream_reference.resolve()
    if sha256(upstream) != args.upstream_sha256:
        parser.error("frozen upstream reference SHA-256 does not match")
    spec = importlib.util.find_spec("cuscan._cuscan")
    if spec is None or not spec.origin:
        parser.error("the cuscan extension under test must be installed")
    extension = Path(spec.origin).resolve()
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
        "upstream_reference": {"path": str(upstream), "sha256": args.upstream_sha256},
        "input_sha256": {str(path): sha256(path) for path in sources},
        "product_sha256": product_sha256(),
        "build_provenance": recorded_build_provenance(args.build_provenance),
        "python": sys.version,
    }
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "python"
    shutil.copytree(
        extension.parent, snapshot / "cuscan", ignore=shutil.ignore_patterns("__pycache__")
    )
    pinned_extension = snapshot / "cuscan" / extension.name
    if sha256(pinned_extension) != provenance["extension"]["sha256"]:
        raise RuntimeError("the extension changed during campaign setup")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(filter(None, (str(snapshot), os.environ.get("PYTHONPATH")))),
    )
    ledger = out / "campaign.jsonl"
    artifacts = out / "runs"
    base = [sys.executable, "-m", "comparison.umap06.run"]
    common = ["--data", str(data), "--artifacts", str(artifacts)]
    items: list[dict[str, Any]] = []

    def fit(fixture, backend, seeds, *, variant=None):
        command = [
            *base,
            "cuscan",
            *common,
            "--reference",
            str(reference),
            "--fixtures",
            fixture,
            "--backend",
            backend,
            "--seeds",
            *map(str, seeds),
            "--skip-transform",
        ]
        if variant is None:
            command += ["--phase5", "--scope", "halved-fit", "--configs", "adam_recursive"]
        else:
            command += ["--phase5-variants", "--configs", variant]
        items.append(
            {
                "kind": "fit",
                "fixture": fixture,
                "backend": backend,
                "seeds": seeds,
                "variant": variant or "adam_recursive",
                "command": command,
            }
        )

    def score(fixture):
        items.append(
            {
                "kind": "score",
                "fixture": fixture,
                "command": [
                    *base,
                    "score",
                    *common,
                    "--fixtures",
                    fixture,
                    "--fit-only",
                ],
            }
        )

    for fixture in ("mix_200k", "mix_50k", "curve_20k", "digits", "blobs_1k"):
        fit(fixture, "cuda", [11, 29])
        fit(fixture, "cpu", [11] if fixture == "mix_200k" else [11, 29])
        score(fixture)
    for backend, seed in (("cuda", 11), ("cpu", 11), ("cuda", 29)):
        fit("mix_1m", backend, [seed])
        score("mix_1m")
    # Produce the recursive-only receipt before spending any remaining time on variants.
    items.append(
        {
            "kind": "receipt",
            "command": [
                *base,
                "accept-phase5",
                *common,
                "--upstream-reference",
                str(upstream),
                "--upstream-sha256",
                args.upstream_sha256,
                "--scope",
                "halved-fit",
            ],
        }
    )
    for variant in ("momentum_recursive", "classic_recursive"):
        for backend in ("cuda", "cpu"):
            fit("mix_50k", backend, [11], variant=variant)
        score("mix_50k")
    # The variant selector requests both component counts in a single invocation.
    fit("mix_50k", "cuda", [11], variant="adam_recursive")
    score("mix_50k")

    expired = False
    for number, item in enumerate(items):
        if expired:
            item["outcome"] = "unrun_deadline"
            continue
        if item["kind"] == "fit" and not (artifacts / "extension.json").exists():
            item["command"].append("--record-extension")
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
            subprocess.run(
                wrapper, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
            )
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
    report = subprocess.run(
        [sys.executable, "-m", "comparison.umap06.campaign", "report", "--ledger", str(ledger)],
        cwd=ROOT,
        check=False,
    )
    manifest = {
        "campaign": "Q",
        "provenance": provenance,
        "items": items,
        "campaign_report_exit_status": report.returncode,
        "limitations": [
            "The installed extension hash does not establish that it was built from the "
            "recorded source.",
            "Transform metrics are not measured; halved seeds reduce statistical power.",
            "Variant fits are reporting-only and have no recursive upstream reference.",
            "A deadline before the receipt leaves it unrun; completed fits may also remain "
            "unscored.",
        ],
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if report.returncode or any(item["outcome"] != "success" for item in items):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
