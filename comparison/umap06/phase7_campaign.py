"""Source-bound measurement campaigns for UMAP 0.6 Phase 7."""

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .datasets import load_fixture
from .phase5_probes import ROOT, git, product_sha256, recorded_build_provenance, sha256
from .phase6_campaign import dump, execute, module
from .quality import QUERY_TIE_RULE, content_sha256, query_recall
from .run import QUERY_REFERENCE_SCHEMA

P7_PRODUCT_CHANGES = frozenset(
    {
        "crates/cuscan-python/src/lib.rs",
        "crates/umap/Cargo.toml",
        "crates/umap/examples/profile_standard_cuda.rs",
        "crates/umap/examples/profile_transform.rs",
        "crates/umap/examples/transform_stages.rs",
        "crates/umap/src/standard_umap/estimator.rs",
        "crates/umap/src/standard_umap/modern_runtime.rs",
        "python/tests/test_umap06_harness.py",
    }
)
PRODUCT_ROOTS = ("crates", "python", "Cargo.toml", "Cargo.lock", "pyproject.toml", "uv.lock")


def paired_regression(predecessor, candidate, margin, *, recall=False, deciding=True):
    """Compare seed-keyed scores without imputing missing pairs."""
    seeds = sorted(predecessor.keys() & candidate.keys())
    missing = sorted(predecessor.keys() ^ candidate.keys())
    if not seeds:
        return {"status": "not measured", "seeds": [], "missing": missing}
    differences = [candidate[seed] - predecessor[seed] for seed in seeds]
    delta = statistics.mean(differences)
    deviation = statistics.stdev(differences) if len(differences) > 1 else None
    threshold = 0.01 if recall else max(margin, 2 * (deviation or 0))
    return {
        "status": "regressed" if deciding and delta < -threshold else "measured",
        "deciding": deciding,
        "seeds": seeds,
        "missing": missing,
        "d": delta,
        "s": deviation,
        "m": 0.01 if recall else margin,
        "threshold": threshold,
        "near_threshold": deciding and abs(delta + threshold) <= 0.25 * threshold,
    }


PERFORMANCE_CASES = (
    ("cuda", 20_000, 32_768, 32, 2),
    ("cuda", 20_000, 8_192, 256, 2),
    ("cuda", 100_000, 8_192, 32, 2),
    ("cpu", 20_000, 2_048, 32, 2),
    ("cpu", 20_000, 32_768, 32, 2),
    ("cpu", 20_000, 8_192, 32, 16),
)


def performance_snapshot_items(out, binary, *, targeted=False):
    for backend, train, query, features, components in PERFORMANCE_CASES:
        case = f"{backend}_{train}_{query}_{features}_{components}"
        path = out / "performance" / case
        yield {
            "kind": "performance_snapshot",
            "case": case,
            "binaries": [str(binary)],
            "output_dir": str(path / "profile"),
            "record": str(path / "snapshot" / "manifest.json"),
            "command": [
                str(binary),
                "--output-dir",
                str(path / "profile"),
                "--save-snapshot",
                str(path / "snapshot"),
                "--snapshot-only",
                "true",
                "--backend",
                backend,
                "--optimizer",
                "adam",
                "--train-rows",
                str(train),
                "--query-rows",
                str(query),
                "--features",
                str(features),
                "--components",
                str(components),
                "--neighbors",
                "15",
                "--epochs",
                "80" if targeted else "200",
                *(["--init", "recursive", "--warm-repeats", "2"] if targeted else []),
            ],
        }


def b7_items(out, baseline, predecessor_binary, candidate_binary, attributed_binary):
    """Schedule fresh processes on the same archived snapshot in P/C, C/P, P/C order."""
    for backend, train, query, features, components in PERFORMANCE_CASES:
        case = f"{backend}_{train}_{query}_{features}_{components}"
        snapshot = baseline / "performance" / case / "snapshot"
        for pair, order in enumerate(
            (
                ("predecessor", "candidate"),
                ("candidate", "predecessor"),
                ("predecessor", "candidate"),
            ),
            1,
        ):
            for arm in order:
                binary = predecessor_binary if arm == "predecessor" else candidate_binary
                destination = out / "wall" / case / f"pair{pair}_{arm}"
                yield b7_item(case, snapshot, destination, binary, arm, pair, False)
    for backend, train, query, features, components in PERFORMANCE_CASES:
        case = f"{backend}_{train}_{query}_{features}_{components}"
        snapshot = baseline / "performance" / case / "snapshot"
        destination = out / "attributed" / case
        yield b7_item(case, snapshot, destination, attributed_binary, "candidate", None, True)


def b7_item(case, snapshot, destination, binary, arm, pair, attributed):
    return {
        "kind": "b7_attributed" if attributed else "b7_wall",
        "case": case,
        "arm": arm,
        "pair": pair,
        "snapshot": str(snapshot),
        "binaries": [str(binary)],
        "output_dir": str(destination),
        "record": str(destination / "summary.json"),
        "command": [
            str(binary),
            "--backend",
            case.split("_", 1)[0],
            "--load-snapshot",
            str(snapshot),
            "--output-dir",
            str(destination),
            "--warm-repeats",
            "2",
            "--stage-barriers",
            str(attributed).lower(),
            "--paired-modes",
            str(attributed).lower(),
        ],
    }


def b7_snapshots(baseline, predecessor):
    """Bind every file, not only the manifest, to the archived P7-targeted item."""
    if predecessor.get("campaign") != "P7-targeted":
        raise ValueError("B7 requires a P7-targeted campaign")
    items = {
        item["case"]: item
        for item in predecessor["items"]
        if item["kind"] == "performance_snapshot"
    }
    snapshots = {}
    for backend, train, query, features, components in PERFORMANCE_CASES:
        case = f"{backend}_{train}_{query}_{features}_{components}"
        item = items.get(case, {})
        directory = baseline / "performance" / case / "snapshot"
        manifest_path = directory / "manifest.json"
        if item.get("outcome") != "success" or item.get("record_sha256") != sha256(manifest_path):
            raise ValueError(f"B7 snapshot is missing or changed: {case}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = manifest["parameters"]
        expected = {
            "backend": backend,
            "train_rows": train,
            "query_rows": query,
            "features": features,
            "components": components,
            "neighbors": 15,
            "epochs": 80,
            "optimizer": "adam",
            "initialization": "recursive",
            "search": "approximate",
        }
        if any(parameters.get(key) != value for key, value in expected.items()):
            raise ValueError(f"B7 snapshot parameters do not match: {case}")
        files = {str(path): sha256(path) for path in sorted(directory.iterdir()) if path.is_file()}
        snapshots[case] = files
    return snapshots


def interference_sample():
    """Best-effort host and GPU interference samples; unavailable is explicit."""
    sample = {"time": time.time(), "loadavg": os.getloadavg()}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        sample["gpu"] = {
            "exit_status": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        sample["gpu"] = {"unavailable": str(exc)}
    return sample


def run_b7(
    out,
    baseline,
    predecessor_binary,
    candidate_binary,
    attributed_binary,
    predecessor_receipt,
    candidate_receipt,
    attributed_receipt,
):
    baseline = baseline.resolve()
    predecessor = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    snapshots = b7_snapshots(baseline, predecessor)
    product = product_sha256()
    binaries = {}
    builds = {}
    for arm, binary, receipt, source in (
        (
            "predecessor",
            predecessor_binary,
            predecessor_receipt,
            predecessor["provenance"]["product_sha256"],
        ),
        ("candidate", candidate_binary, candidate_receipt, product),
        ("attributed", attributed_binary, attributed_receipt, product),
    ):
        binary = binary.resolve()
        build = recorded_build_provenance(receipt)
        digest = sha256(binary)
        if (
            build is None
            or build["record"].get("product_sha256") != source
            or build["record"].get("outputs", {}).get(str(binary)) != digest
        ):
            raise ValueError(f"B7 {arm} build receipt does not bind sources and binary")
        binaries[str(binary)] = digest
        builds[arm] = build
    items = list(
        b7_items(
            out,
            baseline,
            predecessor_binary.resolve(),
            candidate_binary.resolve(),
            attributed_binary.resolve(),
        )
    )
    manifest = {
        "campaign": "B7",
        "items": items,
        "unmeasured": ["250k hard-negative diagnosis: no P7-targeted snapshot"],
        "provenance": {
            "git_revision": git("rev-parse", "HEAD"),
            "git_status_porcelain": git("status", "--porcelain"),
            "product_sha256": product,
            "builds": builds,
            "binaries": binaries,
            "snapshots": snapshots,
            "predecessor_manifest": {
                "path": str(baseline / "manifest.json"),
                "sha256": sha256(baseline / "manifest.json"),
            },
            "harness_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((ROOT / "comparison/umap06").glob("*.py"))
            },
        },
    }
    out.mkdir(parents=True, exist_ok=False)
    return execute_targeted(out, manifest, dict(os.environ))


def fixed_fit_items(out, data, fixture, components_set, baseline=None):
    for components in components_set:
        for backend in ("cuda", "cpu"):
            for config in ("adam_spectral", "classic_spectral"):
                case = f"{fixture}_{components}_{backend}_{config}_11"
                record = out / "stages" / f"{case}.json"
                snapshot = (baseline or out) / "snapshots" / case
                yield {
                    "kind": "fixed_fit",
                    "case": case,
                    "record": str(record),
                    "command": module(
                        "phase7_stages",
                        "--data",
                        data,
                        "--fixture",
                        fixture,
                        "--components",
                        components,
                        "--backend",
                        backend,
                        "--config",
                        config,
                        "--seed",
                        11,
                        "--snapshot",
                        snapshot,
                        "--out",
                        record,
                        *(["--require-snapshot"] if baseline else []),
                    ),
                }


def q7e_items(out, data, baseline):
    """Measure both search graphs in one process on each archived P7 fit."""
    for fixture in ("curve_20k", "mix_50k"):
        for backend in ("cuda", "cpu"):
            case = f"{fixture}_2_{backend}_adam_spectral_11"
            record = out / "stages" / f"{case}.json"
            yield {
                "kind": "q7e_fixed_fit",
                "case": case,
                "record": str(record),
                "env": {"CUSCAN_REQUIRE_CUDA": "1"},
                "command": module(
                    "phase7_stages",
                    "--data",
                    data,
                    "--fixture",
                    fixture,
                    "--components",
                    2,
                    "--backend",
                    backend,
                    "--config",
                    "adam_spectral",
                    "--seed",
                    11,
                    "--snapshot",
                    baseline / "snapshots" / case,
                    "--out",
                    record,
                    "--require-snapshot",
                    "--q7e",
                ),
            }


def q7e_summary(manifest, query_reference=None, query_sha256=None):
    """Evaluate only independently evidenced seed-11 D4 comparisons; never select a winner."""
    missing = {mode: [] for mode in ("forward", "reverse")}
    missed = {mode: [] for mode in missing}
    observed = []
    measured = []
    try:
        reference = manifest["provenance"]["upstream_reference"]
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("quality reference hash mismatch")
        margins = json.loads(Path(reference["path"]).read_text())["acceptance_rule"][
            "transform_margins_by_batch"
        ]
        if not query_reference or not query_sha256 or sha256(Path(query_reference)) != query_sha256:
            raise ValueError("query reference hash mismatch or absent")
        query_data = json.loads(Path(query_reference).read_text())
        if query_data["schema"] != QUERY_REFERENCE_SCHEMA:
            raise ValueError("query reference schema mismatch")
        upstream = query_data["rows"]
    except (OSError, ValueError, KeyError, TypeError):
        margins, upstream = {}, []
        for mode in missing:
            missing[mode].append("verified frozen references")

    items = {
        item["case"]: item for item in manifest["items"] if item.get("kind") == "q7e_fixed_fit"
    }
    for fixture in ("curve_20k", "mix_50k"):
        for backend in ("cuda", "cpu"):
            case = f"{fixture}_2_{backend}_adam_spectral_11"
            try:
                item = items[case]
                record = Path(item["record"])
                if item.get("outcome") != "success" or sha256(record) != item.get("record_sha256"):
                    raise ValueError("record missing or changed")
                rows = json.loads(record.read_text())
                command = item["command"]
                snapshot = Path(command[command.index("--snapshot") + 1])
                snap = json.loads((snapshot / "manifest.json").read_text())
                if any(
                    sha256(snapshot / name) != snap[key]
                    for name, key in (("fit.h5", "model_sha256"), ("fit.npy", "embedding_sha256"))
                ):
                    raise ValueError("snapshot changed")
                identity = snap["identity"]
                if identity != {
                    "backend": backend,
                    "optimizer": "adam",
                    "init": "spectral",
                    "components": 2,
                    "seed": 11,
                }:
                    raise ValueError("snapshot identity mismatch")
                measured.append(case)
            except (OSError, ValueError, KeyError, TypeError, IndexError):
                for mode in missing:
                    missing[mode].append(f"{case}: record or snapshot integrity")
                continue
            for batch in (512, 4096):
                label = f"{case}/batch={batch}"
                refs = [
                    r
                    for r in upstream
                    if r.get("fixture") == fixture
                    and r.get("seed") == 11
                    and r.get("batch") == batch
                ]
                pair = {
                    mode: [r for r in rows if r.get("batch") == batch and r.get("mode") == mode]
                    for mode in missing
                }
                for mode in missing:
                    if len(pair[mode]) != 1 or len(pair["forward"]) != 1:
                        missing[mode].append(f"{label}: paired row")
                        continue
                    row, base = pair[mode][0], pair["forward"][0]
                    try:
                        ref = refs[0] if len(refs) == 1 else None
                        if (
                            not ref
                            or ref["n_neighbors"] != 15
                            or ref["epsilon"] != 0.12
                            or ref["tie_rule"] != QUERY_TIE_RULE
                        ):
                            raise ValueError("upstream identity")
                        if ref["train_sha256"] != snap["train_sha256"] or any(
                            r["query_sha256"] != ref["query_sha256"]
                            or r["train_sha256"] != ref["train_sha256"]
                            or r["tie_rule"] != QUERY_TIE_RULE
                            or r["model_sha256"] != snap["model_sha256"]
                            or r["embedding_sha256"] != snap["embedding_sha256"]
                            for r in (row, base)
                        ):
                            raise ValueError("content identity")
                        recall, floor = float(row["recall"]), float(ref["recall"]) - 0.01
                        if (
                            not np.isfinite([recall, floor]).all()
                            or not 0 <= recall <= 1
                            or not 0 <= float(ref["recall"]) <= 1
                        ):
                            raise ValueError("invalid recall")
                        observed.append(
                            {
                                "case": case,
                                "batch": batch,
                                "mode": mode,
                                "recall": recall,
                                "floor": floor,
                            }
                        )
                        if recall < floor:
                            missed[mode].append(f"{label}: recall below upstream floor")
                        base_recall = float(base["recall"])
                        if not np.isfinite(base_recall) or not 0 <= base_recall <= 1:
                            raise ValueError("invalid paired recall")
                        result = paired_regression(
                            {11: base_recall}, {11: recall}, 0.01, recall=True
                        )
                        observed.append(
                            {
                                "case": case,
                                "batch": batch,
                                "mode": mode,
                                "metric": "recall",
                                **result,
                            }
                        )
                        if result["status"] == "regressed":
                            missed[mode].append(f"{label}: recall regression")
                    except (KeyError, TypeError, ValueError):
                        missing[mode].append(f"{label}: upstream recall provenance")
                        continue
                    for metric, margin_key in (
                        ("trustworthiness", "transform_trustworthiness"),
                        ("preservation_qr", "preservation_qr"),
                    ):
                        if metric == "preservation_qr" and fixture.startswith("mix_"):
                            continue
                        try:
                            before, after = (
                                float(base["scores"][metric]),
                                float(row["scores"][metric]),
                            )
                            margin = float(margins[str(batch)][fixture][margin_key])
                            if not np.isfinite([before, after, margin]).all() or margin < 0:
                                raise ValueError("invalid quality")
                            result = paired_regression({11: before}, {11: after}, margin)
                            observed.append(
                                {
                                    "case": case,
                                    "batch": batch,
                                    "mode": mode,
                                    "metric": metric,
                                    **result,
                                }
                            )
                            if result["status"] == "regressed":
                                missed[mode].append(f"{label}: {metric} regression")
                        except (KeyError, TypeError, ValueError):
                            missing[mode].append(f"{label}: {metric} pair or margin")
                    before = row.get("fitted_state_before_sha256")
                    after = row.get("fitted_state_after_sha256")
                    if any(
                        not isinstance(value, str)
                        or len(value) != 64
                        or any(c not in "0123456789abcdef" for c in value)
                        for value in (before, after)
                    ):
                        missing[mode].append(f"{label}: fit invariance not recorded")
                    elif before != after:
                        missed[mode].append(f"{label}: fit invariance violated")
    eligible = [mode for mode in missing if not missing[mode] and not missed[mode]]
    return {
        "measured": measured,
        "observed": observed,
        "missing": missing,
        "missed": missed,
        "excluded": ["seed != 11", "mix_1m"],
        "unmeasured_variants": ["queue override", "7c variants"],
        "before_freeze_missing": ["compatibility row", "warm timing cases"],
        "eligible": eligible,
        "decision": "eligibility only; no candidate adopted",
    }


def quality_items(out, data, reference, tiers, *, targeted=False):
    for tier in tiers:
        yield {
            "kind": "quality_matrix",
            "tier": tier,
            "env": {"CUSCAN_UMAP06_RECORD_QUERY_KNN": "1", "CUSCAN_REQUIRE_CUDA": "1"},
            "command": module(
                "run",
                "cuscan",
                "--phase7",
                "--phase7-tier",
                tier,
                *(["--record-extension"] if tier == tiers[0] else []),
                *(
                    [
                        "--seeds",
                        11,
                        "--components",
                        2,
                        16,
                        "--fixtures",
                        "curve_20k",
                        "mix_50k",
                        "digits",
                        "blobs_1k",
                        "mix_200k",
                        "--no-child-timeout",
                    ]
                    if targeted
                    else []
                ),
                "--data",
                data,
                "--artifacts",
                out / "quality",
                "--reference",
                reference.parent,
            ),
        }
        yield {
            "kind": "quality_score",
            "tier": tier,
            "command": module(
                "run",
                "score",
                "--data",
                data,
                "--artifacts",
                out / "quality",
                "--reference",
                reference.parent,
            ),
        }


def upstream_query_items(out, data, fixtures):
    """Schedule one upstream search index per fixture and pinned seed."""
    for fixture in fixtures:
        for seed in (11,) if fixture == "mix_1m" else (11, 29):
            destination = out / "query" / fixture / str(seed)
            record = destination / "recall.json"
            yield {
                "kind": "upstream_query",
                "fixture": fixture,
                "seed": seed,
                "output_dir": str(destination),
                "command": module(
                    "upstream_worker",
                    "--fixture",
                    fixture,
                    "--config",
                    "classic_spectral",
                    "--seeds",
                    seed,
                    "--data",
                    data,
                    "--out",
                    destination,
                    "--record-query-knn",
                ),
            }
            yield {
                "kind": "upstream_recall",
                "fixture": fixture,
                "seed": seed,
                "record": str(record),
                "command": module(
                    "phase7_campaign", "score-query", data, fixture, seed, destination, record
                ),
            }


def pin_item(out, data, baseline, fixture, backend, config, capture=None):
    case = f"{fixture}_2_{backend}_{config}_11"
    record = out / "pins" / f"{case}.json"
    mode = "replay" if capture else "host"
    return {
        "kind": f"stage_pin_{mode}",
        "case": case,
        "record": str(record),
        "output_dir": str(record),
        "command": module(
            "phase7_pins",
            mode,
            "--data",
            data,
            "--fixture",
            fixture,
            "--backend",
            backend,
            "--config",
            config,
            "--seed",
            11,
            "--components",
            2,
            "--snapshot",
            baseline / "snapshots" / case,
            "--batch",
            512,
            "--out",
            record,
            *(["--capture", capture] if capture else []),
        ),
    }


def upstream_stage_items(out, data, baseline):
    """Use the exact predecessor fit embedding for every upstream replay."""
    for fixture in ("curve_20k", "mix_50k"):
        for backend in ("cuda", "cpu"):
            for config in ("adam_spectral", "classic_spectral"):
                case = f"{fixture}_2_{backend}_{config}_11"
                snapshot = baseline / "snapshots" / case
                destination = out / "upstream_stages" / case
                yield {
                    "kind": "upstream_stage",
                    "case": case,
                    "output_dir": str(destination),
                    "record": str(destination / "11" / "run.json"),
                    "command": module(
                        "phase7_campaign",
                        "capture-stage",
                        data,
                        fixture,
                        config,
                        snapshot,
                        destination,
                    ),
                }
                yield pin_item(out, data, baseline, fixture, backend, config, destination / "11")


def freeze_upstream_query_reference(out, destination):
    """Freeze completed upstream recall rows, without treating cut rows as zero."""
    if destination.exists():
        raise FileExistsError(destination)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    if manifest["campaign"] != "U7":
        raise ValueError("Query reference requires the U7 manifest")
    rows = []
    for item in manifest["items"]:
        if item["kind"] == "upstream_recall" and item["outcome"] == "success":
            record = Path(item["record"])
            if sha256(record) != item["record_sha256"]:
                raise ValueError(f"Upstream recall record changed after measurement: {record}")
            rows.extend(json.loads(record.read_text(encoding="utf-8")))
    dump(destination, {"schema": QUERY_REFERENCE_SCHEMA, "rows": rows})
    dump(out / "query_reference.json", {"path": str(destination), "sha256": sha256(destination)})


def score_upstream_query(data, fixture, seed, worker_out, record):
    """Run inside the U7 deadline: exact neighbor computation is measurement time."""
    train, batches, _ = load_fixture(data, fixture)
    train_digest = content_sha256(train)
    rows = []
    for count, queries in batches.items():
        with np.load(worker_out / str(seed) / f"query_knn_{count}.npz") as query:
            recall = query_recall(train, queries, query["ids"])["recall"]
        rows.append(
            {
                "fixture": fixture,
                "seed": seed,
                "batch": count,
                "n_neighbors": 15,
                "epsilon": 0.12,
                "tie_rule": QUERY_TIE_RULE,
                "train_sha256": train_digest,
                "query_sha256": content_sha256(queries),
                "recall": recall,
            }
        )
    dump(record, rows)


def execute_targeted(out, manifest, env):
    """Run without a deadline, recording a heartbeat while each child runs."""
    path = out / "manifest.json"
    dump(path, manifest)
    env = dict(env)
    env.pop("UMAP06_CAMPAIGN_DEADLINE", None)
    failed = False
    with (out / "campaign.jsonl").open("x", encoding="utf-8", buffering=1) as events:
        for index, item in enumerate(manifest["items"]):
            for binary in item.get("binaries", []):
                if sha256(Path(binary)) != manifest["provenance"]["binaries"][binary]:
                    raise RuntimeError(f"prebuilt binary changed: {binary}")
            if manifest["campaign"] == "B7":
                for snapshot_path, digest in manifest["provenance"]["snapshots"][
                    item["case"]
                ].items():
                    if sha256(Path(snapshot_path)) != digest:
                        raise RuntimeError(f"B7 snapshot changed: {snapshot_path}")
                item["interference_before"] = interference_sample()
            for key in ("trace", "output_dir"):
                if key in item:
                    Path(item[key]).parent.mkdir(parents=True, exist_ok=True)
            log = out / f"command_{index:03d}.log"
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.Popen(
                    item["command"],
                    cwd=ROOT,
                    env={**env, **item.get("env", {})},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                events.write(
                    json.dumps(
                        {"event": "start", "index": index, "pid": proc.pid, "time": time.time()}
                    )
                    + "\n"
                )
                while True:
                    try:
                        status = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        events.write(
                            json.dumps(
                                {
                                    "event": "heartbeat",
                                    "index": index,
                                    "pid": proc.pid,
                                    "time": time.time(),
                                }
                            )
                            + "\n"
                        )
            item.update(
                outcome="success" if status == 0 else "failed", exit_status=status, log=str(log)
            )
            if manifest["campaign"] == "B7":
                item["interference_after"] = interference_sample()
            if status == 0 and "record" in item:
                try:
                    record = Path(item["record"])
                    payload = json.loads(record.read_text(encoding="utf-8"))
                    if manifest["campaign"] == "B7":
                        expected_build = (
                            "tracing" if item["kind"] == "b7_attributed" else "wall_only"
                        )
                        if payload["run"]["build"] != expected_build:
                            raise ValueError(f"expected {expected_build} build")
                        params = payload["parameters"]
                        if (
                            params["fit_epochs"] != 80
                            or params["transform_epochs"] != 26
                            or params["warm_repeats"] != 2
                        ):
                            raise ValueError("unexpected B7 epoch or repeat parameters")
                    item["record_sha256"] = sha256(record)
                except (OSError, ValueError, KeyError) as exc:
                    item.update(outcome="failed", error=str(exc))
            failed |= item["outcome"] == "failed"
            events.write(
                json.dumps(
                    {
                        "event": "finish",
                        "index": index,
                        "time": time.time(),
                        "outcome": item["outcome"],
                        "exit_status": status,
                    }
                )
                + "\n"
            )
            dump(path, manifest)
    return int(failed)


def run_bound_campaign(
    name, out, data, reference, reference_sha256, build_receipt, items, *, uncapped=False
):
    """Bind the package and native commands before starting the shared deadline."""
    out, data, reference = (Path(path).resolve() for path in (out, data, reference))
    if sha256(reference) != reference_sha256:
        raise ValueError("Frozen upstream reference SHA-256 does not match")
    product = product_sha256()
    head = git("rev-parse", "HEAD")
    if name in ("P7", "P7-targeted"):
        changed = set(git("diff", "--name-only", "HEAD", "--", *PRODUCT_ROOTS).splitlines())
        changed.update(
            git("ls-files", "-o", "--exclude-standard", "--", *PRODUCT_ROOTS).splitlines()
        )
        if head != git("rev-parse", "957f053") or changed - P7_PRODUCT_CHANGES:
            raise ValueError("P7 must run from the Phase 6 predecessor plus the approved harness")
    build = recorded_build_provenance(build_receipt)
    spec = importlib.util.find_spec("cuscan._cuscan")
    if build is None or spec is None or spec.origin is None:
        raise ValueError("A recorded build and an importable cuscan extension are required")
    extension = Path(spec.origin).resolve()
    extension_hash = sha256(extension)
    items = list(items)
    binaries = {
        binary: sha256(Path(binary)) for item in items for binary in item.get("binaries", [])
    }
    record = build["record"]
    if record.get("product_sha256") != product or any(
        record.get("outputs", {}).get(path) != digest
        for path, digest in {str(extension): extension_hash, **binaries}.items()
    ):
        raise ValueError("Build provenance does not bind the product sources and executables")
    manifest = {
        "campaign": name,
        "items": items,
        "provenance": {
            "git_revision": head,
            "git_status_porcelain": git("status", "--porcelain"),
            "product_sha256": product,
            "build_provenance": build,
            "binaries": binaries,
            "harness_sha256": {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted((ROOT / "comparison/umap06").glob("*.py"))
            },
            "fixture_manifest_sha256": sha256(data / "manifest.json"),
            "upstream_reference": {"path": str(reference), "sha256": reference_sha256},
            "extension": {"path": str(extension), "sha256": extension_hash},
        },
    }
    if name == "P7-targeted":
        manifest["unmeasured"] = ["32-component upstream quality has no frozen reference"]
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / "python"
    shutil.copytree(
        extension.parent, snapshot / "cuscan", ignore=shutil.ignore_patterns("__pycache__")
    )
    if sha256(snapshot / "cuscan" / extension.name) != extension_hash:
        raise RuntimeError("The extension changed during campaign setup")
    if product_sha256() != product:
        raise RuntimeError("The product changed during campaign setup")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(snapshot), env.get("PYTHONPATH"))))
    return (
        execute_targeted(out, manifest, env)
        if name == "P7-targeted" or uncapped
        else execute(out, manifest, env)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    b7 = commands.add_parser("B7")
    for flag in (
        "out",
        "baseline",
        "predecessor-binary",
        "candidate-binary",
        "attributed-binary",
        "predecessor-receipt",
        "candidate-receipt",
        "attributed-receipt",
    ):
        b7.add_argument(f"--{flag}", type=Path, required=True)
    for name in ("P7", "P7-targeted", "U7", "Q7b", "Q7e"):
        campaign = commands.add_parser(name)
        for flag in ("out", "data", "reference", "build-receipt"):
            campaign.add_argument(f"--{flag}", type=Path, required=True)
        campaign.add_argument("--reference-sha256", required=True)
        if name in ("P7", "P7-targeted"):
            campaign.add_argument("--profile-binary", type=Path, required=True)
        else:
            campaign.add_argument("--baseline", type=Path, required=True)
            if name == "U7":
                campaign.add_argument("--query-reference", type=Path, required=True)
            if name == "Q7e":
                campaign.add_argument("--query-reference", type=Path, required=True)
                campaign.add_argument("--query-reference-sha256", required=True)
    score = commands.add_parser("score-query")
    score.add_argument("data", type=Path)
    score.add_argument("fixture")
    score.add_argument("seed", type=int)
    score.add_argument("worker_out", type=Path)
    score.add_argument("record", type=Path)
    capture = commands.add_parser("capture-stage")
    capture.add_argument("data", type=Path)
    capture.add_argument("fixture")
    capture.add_argument("config")
    capture.add_argument("snapshot", type=Path)
    capture.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "Q7e":
        parser.error("Q7e requires the archived reverse-graph source and matching tracing build")
    if args.command == "B7":
        raise SystemExit(
            run_b7(
                args.out.resolve(),
                args.baseline,
                args.predecessor_binary,
                args.candidate_binary,
                args.attributed_binary,
                args.predecessor_receipt,
                args.candidate_receipt,
                args.attributed_receipt,
            )
        )
    if args.command in ("P7", "P7-targeted", "U7", "Q7b", "Q7e"):
        out, data, reference = (path.resolve() for path in (args.out, args.data, args.reference))
        if args.command == "P7":
            items = [
                *fixed_fit_items(out, data, "curve_20k", (2,)),
                *fixed_fit_items(out, data, "mix_50k", (2, 16, 48)),
                *(
                    pin_item(out, data, out, fixture, backend, config)
                    for fixture in ("curve_20k", "mix_50k")
                    for backend in ("cuda", "cpu")
                    for config in ("adam_spectral", "classic_spectral")
                ),
                *quality_items(out, data, reference, (1,)),
                *performance_snapshot_items(out, args.profile_binary.resolve()),
                *fixed_fit_items(out, data, "mix_200k", (2,)),
                *fixed_fit_items(out, data, "mix_1m", (2,)),
            ]
        elif args.command == "P7-targeted":
            items = [
                *fixed_fit_items(out, data, "curve_20k", (2,)),
                *fixed_fit_items(out, data, "mix_50k", (2, 16, 32)),
                *(
                    pin_item(out, data, out, fixture, backend, config)
                    for fixture in ("curve_20k", "mix_50k")
                    for backend in ("cuda", "cpu")
                    for config in ("adam_spectral", "classic_spectral")
                ),
                *quality_items(out, data, reference, (1, 2, 3, 4), targeted=True),
                *performance_snapshot_items(out, args.profile_binary.resolve(), targeted=True),
                *fixed_fit_items(out, data, "mix_200k", (2,)),
            ]
        elif args.command == "Q7e":
            if sha256(args.query_reference) != args.query_reference_sha256:
                parser.error("Q7e query reference SHA-256 does not match")
            baseline = args.baseline.resolve()
            predecessor = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
            if predecessor["campaign"] != "P7-targeted" or predecessor["provenance"][
                "fixture_manifest_sha256"
            ] != sha256(data / "manifest.json"):
                parser.error("Q7e requires P7-targeted for the same fixture manifest")
            items = list(q7e_items(out, data, baseline))
        elif args.command == "Q7b":
            baseline = args.baseline.resolve()
            predecessor = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
            targeted = predecessor["campaign"] == "P7-targeted"
            if predecessor["campaign"] not in ("P7", "P7-targeted") or predecessor["provenance"][
                "fixture_manifest_sha256"
            ] != sha256(data / "manifest.json"):
                parser.error("Q7b requires P7 or P7-targeted for the same fixture manifest")
            completed = {
                item["case"]
                for item in predecessor["items"]
                if item["kind"] == "fixed_fit" and item["outcome"] == "success"
            }
            items = [
                item
                for fixture, widths in (
                    ("curve_20k", (2,)),
                    ("mix_50k", (2, 16, 32 if targeted else 48)),
                )
                for item in fixed_fit_items(out, data, fixture, widths, baseline)
                if item["case"] in completed
            ]
            if not items:
                parser.error("Q7b has no paired predecessor rows")
        else:
            baseline = args.baseline.resolve()
            query_reference = args.query_reference.resolve()
            if query_reference.exists() or not query_reference.parent.is_dir():
                parser.error("U7 requires a fresh query reference path in an existing directory")
            predecessor = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
            if predecessor["campaign"] != "P7" or predecessor["provenance"][
                "fixture_manifest_sha256"
            ] != sha256(data / "manifest.json"):
                parser.error("U7 requires P7 for the same fixture manifest")
            items = [
                *upstream_query_items(out, data, ("curve_20k", "mix_50k")),
                *upstream_stage_items(out, data, baseline),
                *upstream_query_items(out, data, ("mix_200k", "mix_1m")),
            ]
        status = run_bound_campaign(
            args.command, out, data, reference, args.reference_sha256, args.build_receipt, items
        )
        if args.command == "U7":
            freeze_upstream_query_reference(out, query_reference)
        if args.command == "Q7e":
            manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
            summary = q7e_summary(manifest, args.query_reference, args.query_reference_sha256)
            summary["query_reference"] = {
                "path": str(args.query_reference.resolve()),
                "sha256": args.query_reference_sha256,
            }
            dump(out / "q7e_summary.json", summary)
        raise SystemExit(status)
    if args.command == "score-query":
        score_upstream_query(args.data, args.fixture, args.seed, args.worker_out, args.record)
    else:
        manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
        embedding = args.snapshot / "fit.npy"
        if sha256(embedding) != manifest["embedding_sha256"]:
            raise ValueError("Predecessor embedding SHA-256 changed")
        os.execv(
            sys.executable,
            module(
                "upstream_worker",
                "--fixture",
                args.fixture,
                "--config",
                args.config,
                "--seeds",
                11,
                "--data",
                args.data,
                "--out",
                args.out,
                "--stage-embedding",
                embedding,
                "--expected-ab",
                manifest["a"],
                manifest["b"],
                "--stage-batches",
                512,
                "--stage-transform-epochs",
                5,
                10,
                11,
                100,
            ),
        )


if __name__ == "__main__":
    main()
