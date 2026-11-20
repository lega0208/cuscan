"""Run source-bound Phase 6 measurement campaigns under the shared deadline."""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from comparison.umap06.phase5_probes import (
    ROOT,
    git,
    probe_matrix,
    product_sha256,
    recorded_build_provenance,
    sha256,
    worker_command,
)

PRODUCT_PATHS = ("crates", "python/cuscan", "Cargo.toml", "Cargo.lock", "pyproject.toml")
BACKENDS = ("cuda", "cpu")


def module(name, *args):
    return [sys.executable, "-m", f"comparison.umap06.{name}", *map(str, args)]


def edges(out, mode):
    for backend in reversed(BACKENDS) if mode == "explicit" else BACKENDS:
        result = out / f"edges_{backend}.json"
        yield {
            "kind": "edge_cases",
            "backend": backend,
            "command": module(
                "phase6_edge_cases", "--mode", mode, "--backend", backend, "--out", result
            ),
            "record": str(result),
        }


def probes(out, data, baseline=False):
    for index, identity in enumerate(probe_matrix()):
        for backend in BACKENDS:
            for replay in range(2 if baseline and identity["optimizer"] == "compatibility" else 1):
                dest = out / "probes" / f"{index:02d}_{backend}_{replay}"
                yield {
                    "kind": "probe",
                    "identity": identity,
                    "backend": backend,
                    "replay": replay,
                    "record": str(dest / "11/run.json"),
                    "command": worker_command(identity, backend, dest, data),
                }


def regression(out, data, baseline):
    expected = []
    for tier, large in (("A", False), ("B", True)):
        for item in probes(out, data):
            probe = item["identity"]
            if (probe["fixture"] == "mix_1m") != large:
                continue
            keys = ["fit"]
            command = item["command"]
            if "--transform-batches" in command and "--skip-transform" not in command:
                start = command.index("--transform-batches") + 1
                for arg in command[start:]:
                    if arg.startswith("--"):
                        break
                    keys.append(f"transform_{arg}")
            if probe.get("transform_self"):
                keys.append("transform_self")
            expected.append({"identity": probe, "backend": item["backend"], "hash_keys": keys})
            yield item
        matrix = out / f"expected_{tier}.json"
        dump(matrix, {"items": expected})
        yield {
            "kind": "comparison",
            "tier": tier,
            "command": module(
                "run",
                "compare-hashes",
                "--baseline",
                baseline,
                "--candidate",
                out,
                "--expected",
                matrix,
            ),
        }


def defaults(out, data):
    yield from edges(out, "library")
    variants = {
        "explicit": {"init": "recursive", "optimizer": "adam"},
        "auto": {"init": "auto", "optimizer": "adam"},
        "library": {"library_defaults": True},
    }
    for tier, fixtures in (("A", ("curve_20k", "mix_50k")), ("B", ("mix_1m",))):
        for fixture in fixtures:
            for backend in BACKENDS:
                for variant, options in variants.items():
                    dest = out / variant / f"{fixture}_{backend}"
                    command = worker_command({"fixture": fixture, **options}, backend, dest, data)
                    if fixture == "mix_1m":
                        command.append("--skip-transform")
                    yield {
                        "kind": "variant",
                        "variant": variant,
                        "fixture": fixture,
                        "backend": backend,
                        "seed": 11,
                        "tier": tier,
                        "record": str(dest / "11/run.json"),
                        "command": command,
                    }
        yield {
            "kind": "comparison",
            "tier": tier,
            "command": module(
                "run",
                "compare-variants",
                "--dir",
                out,
                "--manifest",
                out / "manifest.json",
                "--tier",
                tier,
            ),
        }


def dump(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def execute(out, manifest, env):
    path = out / "manifest.json"
    ledger = out / "campaign.jsonl"
    dump(path, manifest)
    expired = False
    for index, item in enumerate(manifest["items"]):
        if expired:
            item["outcome"] = "unrun_deadline"
            continue
        for binary in item.get("binaries", []):
            if sha256(Path(binary)) != manifest["provenance"]["binaries"][binary]:
                raise RuntimeError(f"prebuilt binary changed: {binary}")
        command = item["command"]
        for key in ("trace", "output_dir"):
            if key in item:
                Path(item[key]).parent.mkdir(parents=True, exist_ok=True)
        log = out / f"command_{index:03d}.log"
        child_env = {**env, **item.get("env", {})}
        if "ty" in command and command[-1] == "check":
            child_env.pop("PYTHONPATH", None)
        with log.open("x", encoding="utf-8") as stream:
            proc = subprocess.run(
                module(
                    "campaign",
                    "run",
                    "--ledger",
                    ledger,
                    "--budget-seconds",
                    1200,
                    "--reserve-seconds",
                    30,
                    "--",
                    *command,
                ),
                cwd=ROOT,
                env=child_env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        entries = ledger.read_text(encoding="utf-8").splitlines() if ledger.is_file() else []
        entry = json.loads(entries[-1]) if len(entries) > 1 else {}
        if entry.get("argv") != command:
            entry = {"outcome": "failed", "exit_status": proc.returncode}
        outcome = entry["outcome"]
        expired = outcome == "deadline"
        item.update(
            outcome="failed" if outcome == "launch_error" else outcome,
            exit_status=entry["exit_status"],
            log=str(log),
        )
        if item["kind"] == "receipt" and outcome in {"success", "failed"}:
            acceptance = Path(item["acceptance"])
            if not acceptance.is_file():
                lines = log.read_text(encoding="utf-8").splitlines()
                if lines:
                    reported = Path(lines[-1]) / "acceptance.json"
                    if reported.is_relative_to(acceptance.parent):
                        acceptance = reported
            if acceptance.is_file():
                saved = out / f"acceptance_{item['receipt']}.json"
                shutil.copy2(acceptance, saved)
                item["receipt_path"] = str(saved)
                item["receipt_sha256"] = sha256(saved)
                # Acceptance misses are measurements, not runner failures.
                item["outcome"] = "success"
        if outcome == "success" and "record" in item:
            record = Path(item["record"])
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
                item["record_sha256"] = sha256(record)
                if item["kind"] == "probe":
                    item["hashes"] = payload["hashes"]
            except (OSError, ValueError, KeyError) as exc:
                item.update(outcome="failed", error=str(exc))
        dump(path, manifest)
    report = subprocess.run(module("campaign", "report", "--ledger", ledger), cwd=ROOT, check=False)
    manifest["campaign_report_exit_status"] = report.returncode
    dump(path, manifest)
    return report.returncode or any(item["outcome"] == "failed" for item in manifest["items"])


def main():
    if sys.argv[1:2] == ["build"]:
        parser = argparse.ArgumentParser(description="Build and bind campaign executable outputs")
        parser.add_argument("--receipt", type=Path, required=True)
        parser.add_argument("--binary", type=Path, action="append", default=[])
        parser.add_argument("command", nargs=argparse.REMAINDER)
        args = parser.parse_args(sys.argv[2:])
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command or args.receipt.exists():
            parser.error("a build command and fresh receipt path are required")
        product = product_sha256()
        subprocess.run(command, cwd=ROOT, check=True)
        if product_sha256() != product:
            raise RuntimeError("product sources changed during build")
        spec = importlib.util.find_spec("cuscan._cuscan")
        if spec is None or spec.origin is None:
            raise RuntimeError("build did not provide an importable extension")
        outputs = [Path(spec.origin).resolve(), *(path.resolve() for path in args.binary)]
        dump(
            args.receipt,
            {
                "command": command,
                "product_sha256": product,
                "outputs": {str(path): sha256(path) for path in outputs},
            },
        )
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", choices=("P", "pin", "flip", "R", "D", "Q", "F", "V"))
    parser.add_argument("--predecessor", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--upstream-reference", type=Path, required=True)
    parser.add_argument("--upstream-sha256", required=True)
    parser.add_argument("--build-provenance", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--pin-trace", type=Path)
    parser.add_argument("--trace-allowlist", type=Path)
    parser.add_argument("--test-plan", type=Path, help="Prebuilt pin test commands as item JSON")
    parser.add_argument("--profile-binary", type=Path)
    parser.add_argument("--transform-binary", type=Path)
    parser.add_argument("--bench-cpu", type=Path)
    parser.add_argument("--bench-cuda", type=Path)
    args = parser.parse_args()
    out, data, reference, upstream = (
        path.resolve() for path in (args.out, args.data, args.reference, args.upstream_reference)
    )
    head = git("rev-parse", "HEAD")
    if args.campaign == "P" and (
        head != git("rev-parse", args.predecessor)
        or subprocess.run(
            ["git", "diff", "--quiet", head, "--", *PRODUCT_PATHS], cwd=ROOT, check=False
        ).returncode
        or git("ls-files", "--others", "--exclude-standard", "--", *PRODUCT_PATHS)
    ):
        parser.error("P6 requires the committed predecessor product tree")
    if sha256(upstream) != args.upstream_sha256:
        parser.error("frozen upstream reference SHA-256 does not match")
    if not reference.is_dir():
        parser.error("reference directory does not exist")
    spec = importlib.util.find_spec("cuscan._cuscan")
    if spec is None or spec.origin is None:
        parser.error("the cuscan extension under test must be installed")
    extension = Path(spec.origin).resolve()
    extension_hash = sha256(extension)
    product = product_sha256()
    manifest = {
        "campaign": args.campaign,
        "provenance": {
            "git_revision": head,
            "git_status_porcelain": git("status", "--porcelain"),
            "product_sha256": product,
            "harness_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in [
                    *sorted((ROOT / "comparison/umap06").glob("*.py")),
                    ROOT / "python/tests/test_umap06_harness.py",
                ]
            },
            "upstream_reference": {"path": str(upstream), "sha256": args.upstream_sha256},
            "fixture_manifest_sha256": sha256(data / "manifest.json"),
            "extension": {"path": str(extension), "sha256": extension_hash},
            "build_provenance": recorded_build_provenance(args.build_provenance),
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    if args.campaign == "P":
        items = [*edges(out, "explicit"), *probes(out, data, baseline=True)]
    elif args.campaign == "R":
        if args.baseline is None:
            parser.error("R6 requires --baseline")
        baseline = args.baseline.resolve()
        receipt = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
        previous = receipt["provenance"]
        current = manifest["provenance"]
        if (
            receipt["campaign"] != "P"
            or previous["git_revision"] != git("rev-parse", args.predecessor)
            or previous["fixture_manifest_sha256"] != current["fixture_manifest_sha256"]
            or previous["upstream_reference"]["sha256"] != args.upstream_sha256
        ):
            parser.error("R6 baseline is not P6 for this predecessor, fixture and reference")
        items = list(regression(out, data, baseline))
    elif args.campaign == "D":
        items = list(defaults(out, data))
    elif args.campaign in {"pin", "flip"}:
        if args.trace is None:
            parser.error("pin and flip require a prepared --trace")
        trace = args.trace.resolve()
        shards = json.loads(
            subprocess.check_output(module("phase6_trace", "shards", "--out", trace), cwd=ROOT)
        )
        items = []
        if args.campaign == "pin":
            if args.test_plan is None:
                parser.error("pin requires --test-plan")
            items.extend(json.loads(args.test_plan.read_text(encoding="utf-8")))
        items.extend(
            {"kind": "trace", "shard": row["id"], "command": row["argv"]} for row in shards
        )
        if args.campaign == "flip":
            if args.pin_trace is None or args.trace_allowlist is None:
                parser.error("flip requires --pin-trace and --trace-allowlist")
            items.append(
                {
                    "kind": "trace_diff",
                    "command": module(
                        "phase6_trace",
                        "diff",
                        "--pin",
                        args.pin_trace.resolve(),
                        "--flip",
                        trace,
                        "--allowlist",
                        args.trace_allowlist.resolve(),
                        "--report",
                        out / "trace_diff.json",
                    ),
                }
            )
    else:
        from comparison.umap06 import phase6_schedules

        if args.campaign == "Q":
            items = list(
                phase6_schedules.quality(out, data, reference, upstream, args.upstream_sha256)
            )
        elif args.campaign == "F":
            binaries = (args.profile_binary, args.transform_binary, args.bench_cpu, args.bench_cuda)
            if any(binary is None for binary in binaries):
                parser.error("F requires all four profiling and benchmark binaries")
            profile, transform, cpu, cuda = (binary.resolve() for binary in binaries)
            items = list(
                phase6_schedules.profiling(
                    out, data, profile, transform, {"cpu": cpu, "cuda": cuda}
                )
            )
        else:
            items = list(phase6_schedules.verification())
    manifest["items"] = items
    manifest["provenance"]["binaries"] = {
        binary: sha256(Path(binary)) for item in items for binary in item.get("binaries", [])
    }
    provenance = manifest["provenance"]["build_provenance"]
    if provenance is None:
        parser.error("build receipt is required")
    build = provenance["record"]
    required_outputs = {str(extension): extension_hash, **manifest["provenance"]["binaries"]}
    if build.get("product_sha256") != product or any(
        build.get("outputs", {}).get(path) != digest for path, digest in required_outputs.items()
    ):
        parser.error("build receipt does not bind this product tree and its executable outputs")
    snapshot = out / "python"
    shutil.copytree(
        extension.parent, snapshot / "cuscan", ignore=shutil.ignore_patterns("__pycache__")
    )
    if sha256(snapshot / "cuscan" / extension.name) != extension_hash:
        raise RuntimeError("the extension changed during campaign setup")
    if product_sha256() != product:
        raise RuntimeError("the product changed during campaign setup")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            filter(
                None,
                (
                    str(snapshot),
                    os.environ.get("PYTHONPATH"),
                ),
            )
        ),
    )
    raise SystemExit(execute(out, manifest, env))


if __name__ == "__main__":
    main()
