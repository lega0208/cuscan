"""Serial, evidence-preserving UMAP 0.6 comparison campaign runner.

Run from the repository root with ``python -m comparison.umap06.run``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
import tomllib
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets, quality
from .spec import CONFIGS, FIXTURES, Config, expected_runs

ROOT = Path(__file__).resolve().parents[2]
THREAD_KEYS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
MEASUREMENT_SOURCES = (
    "comparison/umap_reference.py",
    "comparison/umap06/spec.py",
    "comparison/umap06/datasets.py",
    "comparison/umap06/upstream_worker.py",
    "comparison/umap06/cuscan_worker.py",
)
SCORE_SOURCES = ("comparison/umap06/quality.py", "comparison/umap06/datasets.py")
FROZEN_UPSTREAM_WORKER_REVISION = "8e6823f83ded23b806ef38133145703e35466c7d"
FROZEN_UPSTREAM_WORKER_SHA256 = "ba3b17ebb447fdc56ccf440d0581509ce2e6b20185a02ba56ee593083a886140"
FROZEN_UPSTREAM_WORKER_FIXTURE = (
    ROOT
    / "comparison/umap06/reference"
    / f"upstream-worker-{FROZEN_UPSTREAM_WORKER_REVISION}.py.txt"
)


def phase7_frozen_worker_hash():
    """Verify the exact historical worker bytes used by the frozen reference."""
    try:
        data = FROZEN_UPSTREAM_WORKER_FIXTURE.read_bytes()
    except OSError as exc:
        raise ValueError("cannot read frozen upstream worker fixture") from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != FROZEN_UPSTREAM_WORKER_SHA256:
        raise ValueError("historical upstream worker sha256 mismatch")
    return digest


# Cuscan negative-sampling settings forwarded explicitly to workers and recorded per run.
DEFAULT_SETTINGS = {
    "negative_selection_range": 200000,
    "negative_sample_scale": None,
    "exclude_graph_neighbors": False,
    "negative_sample_scale_adaptation_samples": 128,
}
MODERN_SPECTRAL = ("adam_spectral", "momentum_spectral")
VARIANT_CONFIGS = {
    "classic_recursive": Config("compatibility", "recursive"),
    "momentum_recursive": Config("momentum", "recursive"),
}


def run_config(name):
    return CONFIGS[name] if name in CONFIGS else VARIANT_CONFIGS[name]


def normalize_settings(values):
    scale = values["negative_sample_scale"]
    exclude = values["exclude_graph_neighbors"]
    if not isinstance(exclude, bool):
        raise ValueError("exclude_graph_neighbors must be a boolean")
    if set(values) != set(DEFAULT_SETTINGS):
        raise ValueError(f"unexpected settings keys: {sorted(values)}")
    return {
        "negative_selection_range": int(values["negative_selection_range"]),
        "negative_sample_scale": None if scale is None else float(scale),
        "exclude_graph_neighbors": exclude,
        "negative_sample_scale_adaptation_samples": int(
            values["negative_sample_scale_adaptation_samples"]
        ),
    }


def requested_settings(args):
    """Settings selected on the command line; defaults match the worker defaults."""
    return normalize_settings(
        {
            "negative_selection_range": getattr(args, "negative_selection_range", 200000),
            "negative_sample_scale": getattr(args, "negative_sample_scale", None),
            "exclude_graph_neighbors": getattr(args, "exclude_graph_neighbors", "false") == "true",
            "negative_sample_scale_adaptation_samples": getattr(
                args, "negative_sample_scale_adaptation_samples", 128
            ),
        }
    )


def settings_arguments(settings):
    result = [
        "--negative-selection-range",
        str(settings["negative_selection_range"]),
        "--exclude-graph-neighbors",
        "true" if settings["exclude_graph_neighbors"] else "false",
        "--negative-sample-scale-adaptation-samples",
        str(settings["negative_sample_scale_adaptation_samples"]),
    ]
    if settings["negative_sample_scale"] is not None:
        result += ["--negative-sample-scale", repr(float(settings["negative_sample_scale"]))]
    return result


def record_settings(record):
    """Recorded run settings; runs predating explicit recording used the defaults."""
    if "settings" in record:
        return normalize_settings(record["settings"])
    return dict(DEFAULT_SETTINGS)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except OSError as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def sample():
    try:
        load = Path("/proc/loadavg").read_text(encoding="utf-8").strip()
    except OSError as exc:
        load = f"unavailable: {exc}"
    return {
        "loadavg": load,
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,driver_version,utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
    }


def provenance():
    sources = [
        ROOT / "comparison" / "umap_reference.py",
        *sorted((ROOT / "comparison" / "umap06").glob("*.py")),
    ]
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        cpu_model = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        )
    except OSError:
        cpu_model = platform.processor()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "nproc": command_output(["nproc"]),
        "threads": {k: os.environ.get(k) for k in THREAD_KEYS},
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "rustc": command_output(["rustc", "-Vv"]),
        "freeze": command_output(["uv", "pip", "freeze"]),
        "gpu": sample()["gpu"],
        "source_hashes": {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources},
    }


def wheel_provenance(lock_path=ROOT / "uv.lock"):
    lock_path = Path(lock_path)
    with lock_path.open("rb") as stream:
        lock = tomllib.load(stream)
    packages = [p for p in lock["package"] if p["name"] == "umap-learn"]
    if len(packages) != 1 or packages[0]["version"] != "0.6.0":
        raise ValueError("uv.lock must contain exactly umap-learn 0.6.0")
    wheels = [
        w
        for w in packages[0].get("wheels", [])
        if w["url"].endswith("/umap_learn-0.6.0-py3-none-any.whl")
    ]
    expected = "sha256:31f0f6ab31e95463a33404b5228921d24f4314f1607c9db405446ab0fad23663"
    if len(wheels) != 1 or wheels[0].get("hash") != expected:
        raise ValueError("uv.lock does not contain the expected umap-learn 0.6.0 wheel hash")
    return {
        "wheel_sha256": wheels[0]["hash"].removeprefix("sha256:"),
        "wheel_url": wheels[0]["url"],
        "lock_sha256": sha256(lock_path),
    }


def extension():
    spec = importlib.util.find_spec("cuscan._cuscan")
    if spec is None or not spec.origin:
        raise RuntimeError("Cannot locate the built cuscan extension")
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing extension: {path}")
    return {"path": str(path), "sha256": sha256(path)}


def require_extension(artifacts, establish=False):
    baseline = Path(artifacts) / "extension.json"
    current = extension()
    if establish:
        write_new(
            baseline,
            {
                "extension": current,
                "provenance": provenance(),
                "recorded_at": dt.datetime.now(dt.UTC).isoformat(),
            },
        )
    if not baseline.exists():
        raise RuntimeError(
            "Extension baseline missing: run cuscan with --record-extension deliberately"
        )
    expected = read_json(baseline)["extension"]
    if expected["sha256"] != current["sha256"]:
        raise RuntimeError(
            f"Extension hash mismatch: expected {expected['sha256']}, got {current['sha256']}"
        )
    return current


def new_run(root, role):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (
        dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + role
        + "-"
        + uuid.uuid4().hex[:8]
    )
    path.mkdir(exist_ok=False)
    return path


class CampaignTerminated(BaseException):
    pass


def campaign_term_handler(_signum, _frame):
    raise CampaignTerminated()


def child(command, destination, timeout, env=None, intent=None):
    if os.name != "posix":
        raise RuntimeError("UMAP 0.6 campaign workers require POSIX process groups")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    if intent is not None:
        write_new(destination / "intent.json", intent)
    before = sample()
    start = time.monotonic()
    timed_out = False
    deadline = os.environ.get("UMAP06_CAMPAIGN_DEADLINE")
    remaining = float(deadline) - time.time() if deadline is not None else None
    deadline_limited = remaining is not None and (timeout is None or remaining <= timeout)
    effective_timeout = timeout
    if remaining is not None:
        effective_timeout = remaining if timeout is None else min(timeout, remaining)
    previous_handler = signal.signal(signal.SIGTERM, campaign_term_handler)
    with (
        (destination / "stdout.txt").open("x", encoding="utf-8") as out,
        (destination / "stderr.txt").open("x", encoding="utf-8") as err,
    ):
        try:
            if effective_timeout is not None and effective_timeout <= 0:
                timed_out = True
                code = None
            else:
                process = subprocess.Popen(
                    command, stdout=out, stderr=err, env=env, cwd=ROOT, start_new_session=True
                )
                try:
                    code = process.wait(timeout=effective_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    code = None
                except BaseException:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
        except CampaignTerminated:
            timed_out = True
            deadline_limited = True
            code = None
        except OSError as exc:
            err.write(f"Unable to launch child: {exc}\n")
            code = -1
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
    record = {
        "command": command,
        "before": before,
        "after": sample(),
        "seconds": time.monotonic() - start,
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "outcome": "deadline" if timed_out and deadline_limited else None,
        "returncode": code,
        "environment": {k: (env or os.environ).get(k) for k in THREAD_KEYS},
    }
    write_new(destination / "child.json", record)
    return record


def identity(record):
    return (
        record["role"],
        record.get("backend"),
        record["fixture"],
        record["config"],
        int(record["seed"]),
        int(record["components"]),
        bool(record["precomputed_knn"]),
    )


def validate_run(path, record, skip_transform=False):
    fixture = FIXTURES[record["fixture"]]
    if record["seed"] not in fixture.seeds:
        raise ValueError(f"Unexpected seed in {path}")
    names = ["fit"]
    if not record["precomputed_knn"] and not skip_transform:
        names.extend(f"transform_{count}" for count in fixture.batches)
    for name in names:
        array = np.load(path.parent / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        rows = fixture.rows if name == "fit" else int(name.split("_")[1])
        if array.shape != (rows, record["components"]) or array.dtype != np.dtype("float32"):
            raise ValueError(f"Invalid embedding shape or dtype: {path.parent / name}")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite embedding: {path.parent / name}")
        digest = hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()
        if digest != record["hashes"].get(name):
            raise ValueError(f"Embedding hash mismatch: {path.parent / name}")
    expected_epochs = 500 if fixture.rows <= 10000 else 200
    if record["epochs"] != expected_epochs:
        raise ValueError(f"Unexpected epoch count: {path}")
    if not np.isfinite(record["fit_seconds"]) or record["fit_seconds"] <= 0:
        raise ValueError(f"Invalid fit timing: {path}")
    expected_batches = (
        set() if record["precomputed_knn"] or skip_transform else {str(n) for n in fixture.batches}
    )
    if set(record["transform_seconds"]) != expected_batches:
        raise ValueError(f"Missing transform timing: {path}")
    if record["role"] == "upstream":
        config = CONFIGS[record["config"]]
        params = record["params"]
        if (
            not record.get("fit_layout_epochs")
            or record["fit_layout_epochs"][-1] != record["epochs"]
            or params.get("n_epochs") is not None
        ):
            raise ValueError(f"Missing observed fit epochs or unexpected n_epochs: {path}")
        transform_epochs = {batch: 100 if int(batch) <= 10000 else 30 for batch in expected_batches}
        if record.get("transform_epochs") != transform_epochs:
            raise ValueError(f"Unexpected observed transform epochs: {path}")
        if (
            record["resolved_optimizer"] != config.optimizer
            or record["resolved_init"] != config.init
            or params["compatibility_layout"] is not False
            or params["nearest_neighbor_backend"] != "pynndescent"
        ):
            raise ValueError(f"Unexpected resolved upstream configuration: {path}")
        if any(not np.isfinite(t) or t <= 0 for t in record["transform_seconds"].values()):
            raise ValueError(f"Invalid upstream transform timings: {path}")
    elif record["role"] == "cuscan":
        if record["backend"] not in ("cpu", "cuda") or record["fit_temperature"] not in (
            "cold",
            "warm",
        ):
            raise ValueError(f"Invalid cuscan backend or fit temperature: {path}")
        for samples in record["transform_seconds"].values():
            if len(samples["warm"]) != 3 or any(
                not np.isfinite(t) or t <= 0 for t in [samples["cold"], *samples["warm"]]
            ):
                raise ValueError(f"Invalid cuscan transform timings: {path}")
    else:
        raise ValueError(f"Invalid run role: {path}")


def validate_invocation(path, record, data, extension_hash, source_hashes, fixture_hashes):
    invocation = read_json(path.parents[4] / "invocation.json")
    fixture = record["fixture"]
    if invocation.get("extension", {}).get("sha256") != extension_hash:
        raise ValueError(f"Extension mismatch in invocation: {path}")
    recorded = invocation.get("provenance", {}).get("source_hashes", {})
    if any(recorded.get(name) != source_hashes[name] for name in MEASUREMENT_SOURCES):
        raise ValueError(f"Source mismatch in invocation: {path}")
    if fixture not in fixture_hashes:
        fixture_hashes[fixture] = datasets.fixture_hash(data, fixture)
    if invocation.get("fixture_hashes", {}).get(fixture) != fixture_hashes[fixture]:
        raise ValueError(f"Fixture mismatch in invocation: {path}")
    receipt = read_json(path.parents[2] / "child.json")
    command = receipt["command"]
    parser = argparse.ArgumentParser(add_help=False)
    for flag in (
        "fixture",
        "config",
        "backend",
        "init",
        "components",
        "data",
        "out",
        "neighbor-search",
        "optimizer",
        "negative-selection-range",
        "negative-sample-scale",
        "exclude-graph-neighbors",
        "negative-sample-scale-adaptation-samples",
    ):
        parser.add_argument(f"--{flag}")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--precomputed-knn", action="store_true")
    parser.add_argument("--skip-transform", action="store_true")
    options, unknown = parser.parse_known_args(command[3:])
    role = record["role"]
    if (
        len(command) < 3
        or command[1:3] != ["-m", f"comparison.umap06.{role}_worker"]
        or unknown
        or options.fixture != fixture
        or int(options.components) != record["components"]
        or record["seed"] not in options.seeds
        or Path(options.out).resolve() != path.parents[1].resolve()
        or Path(options.data).resolve() != Path(data).resolve()
    ):
        raise ValueError(f"Worker command mismatch: {path}")
    if role == "upstream":
        if any(
            value is not None
            for value in (
                options.negative_selection_range,
                options.negative_sample_scale,
                options.exclude_graph_neighbors,
                options.negative_sample_scale_adaptation_samples,
            )
        ):
            raise ValueError(f"Upstream command carries cuscan settings: {path}")
        if (
            options.config != record["config"]
            or options.precomputed_knn != record["precomputed_knn"]
            or receipt["environment"].get("NUMBA_NUM_THREADS") != "15"
        ):
            raise ValueError(f"Upstream command or environment mismatch: {path}")
    elif (
        options.backend != record["backend"]
        or options.init != run_config(record["config"]).init
        or (options.optimizer or "compatibility") != run_config(record["config"]).optimizer
        or (options.neighbor_search == "exact") != record["precomputed_knn"]
        or (options.exclude_graph_neighbors or "false") not in ("true", "false")
    ):
        raise ValueError(f"Cuscan command mismatch: {path}")
    command_settings = normalize_settings(
        {
            "negative_selection_range": options.negative_selection_range or 200000,
            "negative_sample_scale": options.negative_sample_scale,
            "exclude_graph_neighbors": (options.exclude_graph_neighbors or "false") == "true",
            "negative_sample_scale_adaptation_samples": (
                options.negative_sample_scale_adaptation_samples or 128
            ),
        }
    )
    if command_settings != record_settings(record):
        raise ValueError(f"Cuscan settings mismatch between command and run: {path}")
    if options.skip_transform != invocation.get("skip_transform", False):
        raise ValueError(f"Transform selection mismatch: {path}")


def runs(root, data):
    result = {}
    current_extension = require_extension(root)["sha256"]
    source_hashes = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in [
            ROOT / "comparison" / "umap_reference.py",
            *sorted((ROOT / "comparison" / "umap06").glob("*.py")),
        ]
    }
    fixture_hashes = {}
    for path in sorted(Path(root).glob("*/children/*/output/*/run.json")):
        if not (path.parents[4] / "invocation.json").exists():
            continue
        record = read_json(path)
        child_path = path.parents[2] / "child.json"
        if not child_path.exists() or read_json(child_path).get("returncode") != 0:
            continue
        validate_invocation(path, record, data, current_extension, source_hashes, fixture_hashes)
        invocation = read_json(path.parents[4] / "invocation.json")
        if invocation.get("skip_transform", False):
            validate_run(path, record, skip_transform=True)
        else:
            validate_run(path, record)
        key = identity(record)
        if key in result:
            raise ValueError(
                f"Duplicate successful run identity: {key}; {result[key][0]} and {path}"
            )
        result[key] = (path, record)
    return result


def phase3_runs():
    classic = expected_runs("cuscan")
    modern = [
        dict(row, backend=backend)
        for row in expected_runs("upstream")
        if row["config"] in ("adam_spectral", "momentum_spectral") and row["fixture"] != "mix_1m"
        for backend in ("cpu", "cuda")
    ]
    return classic + modern


def phase4_runs():
    """Classic compatibility probes plus every frozen non-recursive modern spectral variant.

    Unlike Phase 3, ``mix_1m`` modern rows are required (plan Phase 4 quality section).
    """
    classic = expected_runs("cuscan")
    modern = [
        dict(row, backend=backend)
        for row in expected_runs("upstream")
        if row["config"] in MODERN_SPECTRAL
        for backend in ("cpu", "cuda")
    ]
    return classic + modern


def phase4_halved_runs():
    groups = {}
    for row in phase4_runs():
        key = (
            row["fixture"],
            row["config"],
            row["backend"],
            row["components"],
            row["precomputed_knn"],
        )
        groups.setdefault(key, []).append(row)
    selected = []
    odd = 0
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row["seed"])
        count = len(rows) // 2
        if len(rows) % 2:
            count += 1 if odd % 2 == 0 else 0
            odd += 1
        selected.extend(rows[:count])
    assert len(selected) * 2 == len(phase4_runs())
    return selected


def phase5_runs(halved=False):
    rows = [
        dict(row, backend=backend)
        for row in expected_runs("upstream")
        if row["config"] == "adam_recursive"
        for backend in ("cpu", "cuda")
    ]
    if not halved:
        return rows
    return [
        row
        for row in rows
        if row["seed"] in (11, 29)
        and not (
            row["backend"] == "cpu"
            and row["fixture"] in ("mix_200k", "mix_1m")
            and row["seed"] == 29
        )
    ]


PHASE7_TIER_FIXTURES = ("curve_20k", "mix_50k", "digits", "blobs_1k")
PHASE7_HALVED_COUNTS = {
    "mix_50k": 36,
    "blobs_1k": 20,
    "digits": 20,
    "curve_20k": 20,
    "mix_200k": 12,
    "mix_1m": 8,
}
PHASE7_RECALL_MARGIN = 0.01


def phase7_runs():
    """Every frozen upstream identity without precomputed kNN, on CPU and CUDA."""
    return [
        dict(row, backend=backend)
        for row in expected_runs("upstream")
        if not row["precomputed_knn"]
        for backend in ("cpu", "cuda")
    ]


def phase7_tier(row):
    """Plan 7 priority tier (1-5) of a Phase 7 row."""
    if row["fixture"] == "mix_1m":
        return 5
    if row["fixture"] == "mix_200k":
        return 4
    if row["components"] != 2:
        return 3
    return 1 if row["config"] in ("adam_recursive", "classic_spectral") else 2


def phase7_halved_runs():
    """The 116 pinned Phase 7 rows in tier order, CUDA before CPU within each tier.

    Seeds 11 and 29, except seed 11 only for mix_1m and CPU mix_200k.
    """
    rows = [
        row
        for row in phase7_runs()
        if row["seed"] in (11, 29)
        and not (
            row["seed"] == 29
            and (
                row["fixture"] == "mix_1m"
                or (row["fixture"] == "mix_200k" and row["backend"] == "cpu")
            )
        )
    ]
    order = {name: index for index, name in enumerate(PHASE7_TIER_FIXTURES)}
    rows.sort(
        key=lambda row: (
            phase7_tier(row),
            row["backend"] != "cuda",
            order.get(row["fixture"], len(order)),
            row["config"],
            row["components"],
            row["seed"],
        )
    )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["fixture"]] = counts.get(row["fixture"], 0) + 1
    assert len(rows) == 116 and counts == PHASE7_HALVED_COUNTS, counts
    assert len({identity(dict(row, role="cuscan")) for row in rows}) == len(rows)
    return rows


def selected_matrix(
    role,
    fixtures=None,
    configs=None,
    seeds=None,
    backend=None,
    phase3=False,
    phase4=False,
    phase5=False,
    phase5_variants=False,
    phase7=False,
    phase7_tier_only=None,
):
    if sum((phase3, phase4, phase5, phase5_variants, phase7)) > 1:
        raise ValueError("phase selections are mutually exclusive")
    if phase7_tier_only is not None and not phase7:
        raise ValueError("--phase7-tier requires --phase7")
    if role == "cuscan" and phase7:
        rows = [
            row
            for row in phase7_halved_runs()
            if phase7_tier_only is None or phase7_tier(row) == phase7_tier_only
        ]
    elif role == "cuscan" and phase5_variants:
        base = phase5_runs()
        rows: list[dict[str, Any]] = [
            dict(row, config=config) for row in base for config in VARIANT_CONFIGS
        ]
        rows += [dict(row, components=components) for row in base for components in (16, 48)]
    elif role == "cuscan" and phase5:
        rows = phase5_runs()
    elif role == "cuscan" and phase4:
        rows = phase4_runs()
    elif role == "cuscan" and phase3:
        rows = phase3_runs()
    else:
        rows = expected_runs(role, backend if role == "cuscan" else None)
    return [
        dict(row, role=role)
        for row in rows
        if (fixtures is None or row["fixture"] in fixtures)
        and (configs is None or row["config"] in configs)
        and (seeds is None or row["seed"] in seeds)
        and (backend is None or role != "cuscan" or row["backend"] == backend)
    ]


def matrix(args, role):
    extension_record = require_extension(args.artifacts, args.record_extension)
    from comparison.umap_reference import require_reference

    require_reference()
    existing = runs(args.artifacts, args.data)
    phase4 = getattr(args, "phase4", False)
    requested = selected_matrix(
        role,
        args.fixtures,
        args.configs,
        args.seeds,
        args.backend,
        getattr(args, "phase3", False),
        phase4,
        getattr(args, "phase5", False),
        getattr(args, "phase5_variants", False),
        getattr(args, "phase7", False),
        getattr(args, "phase7_tier", None),
    )
    if getattr(args, "components", None) is not None:
        requested = [row for row in requested if row["components"] in args.components]
    if getattr(args, "scope", None) == "halved-fit":
        if phase4:
            allowed = {identity(dict(row, role="cuscan")) for row in phase4_halved_runs()}
        elif getattr(args, "phase5", False):
            allowed = {identity(dict(row, role="cuscan")) for row in phase5_runs(halved=True)}
        else:
            raise ValueError("--scope halved-fit requires --phase4 or --phase5")
        requested = [row for row in requested if identity(row) in allowed]
    if not requested:
        raise ValueError("Selection contains no matrix identities")
    settings = requested_settings(args) if role == "cuscan" else None
    explicit_settings = role == "cuscan" and (phase4 or settings != DEFAULT_SETTINGS)
    for row in requested:
        previous = existing.get(identity(row))
        if previous is None or role != "cuscan":
            continue
        invocation = read_json(previous[0].parents[4] / "invocation.json")
        if invocation.get("skip_transform", False) != getattr(args, "skip_transform", False):
            raise ValueError(f"Existing run has different transform selection: {previous[0]}")
        if phase4 and "settings" not in previous[1]:
            raise ValueError(f"Existing run lacks recorded settings: {previous[0]}")
        if record_settings(previous[1]) != settings:
            raise ValueError(
                f"Existing run settings differ from requested settings: {previous[0]}; "
                "use a separate --artifacts directory per settings variant"
            )
    for name in {row["fixture"] for row in requested}:
        cache_errors = quality.verify_cache(args.data, name)
        if cache_errors:
            raise ValueError(f"Invalid evaluation cache for {name}: {cache_errors}")
    approved = {
        identity(item["scope"]): item
        for item in decisions(args.reference)
        if "role" in item["scope"]
    }
    missing = [
        r
        for r in requested
        if identity(r) not in existing
        and approved.get(identity(r), {}).get("option") not in ("drop", "omit_seed")
    ]
    failed = set()
    for intent_path in Path(args.artifacts).glob("*/children/*/intent.json"):
        child_path = intent_path.parent / "child.json"
        if not child_path.exists():
            raise RuntimeError(f"Unresolved interrupted attempt: {intent_path.parent}")
        outcome = read_json(child_path)
        if outcome.get("timed_out"):
            failed.update(identity(row) for row in read_json(intent_path)["identities"])
    for row in missing:
        if (
            identity(row) in failed
            and approved.get(identity(row), {}).get("option") != "raise_timeout"
        ):
            raise RuntimeError(
                f"Previous child failed for {identity(row)}; explicit decision required"
            )
    if role == "upstream":
        pilots = [r for r in selected_matrix(role, fixtures=["mix_1m"], seeds=[11])]
        for pilot in pilots:
            previous = existing.get(identity(pilot))
            if previous is not None:
                result = read_json(previous[0].parents[2] / "child.json")
                if (
                    result["seconds"] > 17280
                    and approved.get(identity(pilot), {}).get("option") != "raise_timeout"
                ):
                    raise RuntimeError(
                        f"1M pilot exceeded 4.8 hours: {previous[0]}; human decision required"
                    )
        if any(r["fixture"] == "mix_1m" and r["seed"] != 11 for r in missing):
            scheduled = {identity(r) for r in missing}
            absent = [
                r
                for r in pilots
                if identity(r) not in existing
                and identity(r) not in scheduled
                and approved.get(identity(r), {}).get("option") not in ("drop", "omit_seed")
            ]
            if absent:
                raise RuntimeError(f"All 1M seed-11 pilots must pass first; missing: {absent}")
    run = new_run(args.artifacts, role)
    write_new(
        run / "invocation.json",
        {
            "role": role,
            "requested": requested,
            "reused": [r for r in requested if identity(r) in existing],
            "omitted_by_decision": [
                r
                for r in requested
                if identity(r) not in existing
                and approved.get(identity(r), {}).get("option") in ("drop", "omit_seed")
            ],
            "provenance": provenance(),
            "extension": extension_record,
            **(
                {
                    "settings": settings,
                    "phase4": phase4,
                    "skip_transform": getattr(args, "skip_transform", False),
                    "no_child_timeout": getattr(args, "no_child_timeout", False),
                }
                if role == "cuscan"
                else {}
            ),
            "fixture_hashes": {
                name: datasets.fixture_hash(args.data, name)
                for name in sorted({r["fixture"] for r in requested})
            },
        },
    )
    groups = {}
    for row in missing:
        key = (
            row["fixture"],
            row["config"],
            row.get("backend"),
            row["components"],
            row["precomputed_knn"],
            row["seed"]
            if row["fixture"] == "mix_1m" and (role == "upstream" or getattr(args, "phase5", False))
            else None,
        )
        groups.setdefault(key, []).append(row["seed"])
    ordered = sorted(
        groups.items(),
        key=lambda entry: (
            role == "upstream" and entry[0][0] == "mix_1m" and entry[0][5] != 11,
            bool(getattr(args, "scope", None) and entry[0][0] == "mix_1m"),
            bool(getattr(args, "scope", None) and entry[0][4]),
            FIXTURES[entry[0][0]].rows,
            (entry[0][5] or 0),
            entry[0][2] != "cuda" if entry[0][0] == "mix_1m" else False,
        ),
    )
    if getattr(args, "phase7", False):
        ordered = list(groups.items())
    passed_pilots = (
        {
            identity(r)
            for r in pilots
            if identity(r) in existing
            or approved.get(identity(r), {}).get("option") in ("drop", "omit_seed")
        }
        if role == "upstream"
        else set()
    )
    for index, ((fixture, config, backend, components, exact, _), seeds) in enumerate(ordered):
        if role == "upstream" and fixture == "mix_1m" and 11 not in seeds:
            if any(identity(pilot) not in passed_pilots for pilot in pilots):
                raise RuntimeError("All 1M seed-11 pilots must pass before other 1M seeds")
        output = run / "children" / str(index) / "output"
        cmd = [
            sys.executable,
            "-m",
            f"comparison.umap06.{role}_worker",
            "--fixture",
            fixture,
            "--seeds",
            *map(str, seeds),
            "--out",
            str(output.resolve()),
            "--data",
            str(args.data.resolve()),
            "--components",
            str(components),
        ]
        if role == "upstream":
            cmd += ["--config", config]
            if exact:
                cmd.append("--precomputed-knn")
        else:
            cmd += [
                "--backend",
                backend,
                "--init",
                run_config(config).init,
                "--optimizer",
                run_config(config).optimizer,
            ]
            if exact:
                cmd += ["--neighbor-search", "exact"]
            if explicit_settings:
                cmd += settings_arguments(settings)
            if getattr(args, "skip_transform", False):
                cmd.append("--skip-transform")
        env = os.environ.copy()
        if role == "upstream":
            env["NUMBA_NUM_THREADS"] = "15"
        child_identities = [
            dict(
                role=role,
                fixture=fixture,
                config=config,
                seed=seed,
                components=components,
                precomputed_knn=exact,
                **({"backend": backend} if backend else {}),
            )
            for seed in seeds
        ]
        timeout = (
            (21600 if role == "upstream" else 21600 * len(seeds)) if fixture == "mix_1m" else 1800
        )
        for row in child_identities:
            decision = approved.get(identity(row), {})
            if decision.get("option") == "raise_timeout":
                timeout = max(timeout, int(decision["timeout_seconds"]))
        if getattr(args, "no_child_timeout", False):
            timeout = None
        result = child(cmd, output.parent, timeout, env, {"identities": child_identities})
        if result["returncode"] != 0:
            detail = (
                "; human decision required for timeout"
                if result["timed_out"]
                else "; investigate child error"
            )
            raise RuntimeError(f"Child failed: {output.parent}{detail}")
        if (
            role == "upstream"
            and fixture == "mix_1m"
            and 11 in seeds
            and result["seconds"] > 17280
            and approved.get(identity(child_identities[0]), {}).get("option") != "raise_timeout"
        ):
            raise RuntimeError(
                f"1M pilot exceeded 4.8 hours: {output.parent}; human decision required"
            )
        if role == "upstream" and fixture == "mix_1m" and 11 in seeds:
            passed_pilots.add(
                identity(
                    dict(
                        role=role,
                        fixture=fixture,
                        config=config,
                        seed=11,
                        components=components,
                        precomputed_knn=exact,
                    )
                )
            )
    print(run)


def generate(args):
    run = new_run(args.artifacts, "generate")
    write_new(
        run / "invocation.json", {"data": str(args.data.resolve()), "fixtures": args.fixtures}
    )
    datasets.generate(args.data, args.fixtures)
    for name in args.fixtures or list(FIXTURES):
        path = args.data / name / "eval.npz"
        if path.exists():
            errors = quality.verify_cache(args.data, name)
            if errors:
                raise ValueError(f"Invalid evaluation cache: {errors}")
        else:
            quality.build_cache(args.data, name)
    write_new(run / "result.json", {"fixtures": args.fixtures or list(FIXTURES)})
    print(run)


def determinism(args):
    from comparison.umap_reference import require_reference

    require_reference()
    available = runs(args.artifacts, args.data)
    run = new_run(args.artifacts, "determinism")
    results = []
    for fixture in ("blobs_1k", "digits", "mix_50k"):
        if args.fixtures and fixture not in args.fixtures:
            continue
        for config in CONFIGS:
            if args.configs and config not in args.configs:
                continue
            key = ("upstream", None, fixture, config, 11, 2, False)
            if key not in available:
                raise RuntimeError(f"Missing original run for determinism: {key}")
            baseline = available[key][1]["hashes"]
            for threads in (15, 1) if fixture == "digits" else (15,):
                destination = run / "children" / f"{fixture}-{config}-{threads}"
                env = os.environ.copy()
                env["NUMBA_NUM_THREADS"] = str(threads)
                command = [
                    sys.executable,
                    "-m",
                    "comparison.umap06.upstream_worker",
                    "--fixture",
                    fixture,
                    "--config",
                    config,
                    "--seeds",
                    "11",
                    "--out",
                    str((destination / "output").resolve()),
                    "--data",
                    str(args.data.resolve()),
                ]
                outcome = child(command, destination, 1800, env)
                if outcome["returncode"] != 0:
                    raise RuntimeError(f"Determinism child failed: {destination}")
                repeated = read_json(destination / "output" / "11" / "run.json")
                results.append(
                    {
                        "fixture": fixture,
                        "config": config,
                        "threads": threads,
                        "original": str(available[key][0].resolve()),
                        "repeated": str((destination / "output" / "11" / "run.json").resolve()),
                        "original_hashes": baseline,
                        "repeat_hashes": repeated["hashes"],
                        "identical": baseline == repeated["hashes"],
                    }
                )
    write_new(run / "determinism.json", {"results": results, "provenance": provenance()})
    print(run)


def stages(args):
    extension_record = require_extension(args.artifacts)
    run = new_run(args.artifacts, "stages")
    fixture_hashes = {
        name: datasets.fixture_hash(args.data, name) for name in ("mix_50k", "mix_200k", "mix_1m")
    }
    write_new(
        run / "provenance.json",
        {"build": provenance(), "extension": extension_record, "fixture_hashes": fixture_hashes},
    )
    build = child(
        [
            "soldr",
            "cargo",
            "build",
            "-p",
            "umap",
            "--release",
            "--features",
            "tracing",
            "--example",
            "profile_standard_stages",
        ],
        run / "build",
        21600,
    )
    if build["returncode"] != 0:
        raise RuntimeError(f"Stage build failed: {run / 'build'}")
    for fixture in ("mix_50k", "mix_200k", "mix_1m"):
        for backend in ("cpu", "cuda"):
            dest = run / "stages" / f"{fixture}-{backend}"
            cmd = [
                str(ROOT / "target/release/examples/profile_standard_stages"),
                str((args.data / fixture / "train.f32").resolve()),
                str(FIXTURES[fixture].rows),
                "32",
                "200",
                "--backend",
                backend,
                "--search",
                "approximate",
                "--init",
                "spectral",
                "--seed",
                "11",
                "--runs",
                "3",
                "--trace",
                str((dest / "trace.json").resolve()),
            ]
            result = child(cmd, dest, 21600)
            if result["returncode"] != 0:
                raise RuntimeError(f"Stage failed: {dest}")
    print(run)


def score(args):
    existing = runs(args.artifacts, args.data)
    run = new_run(args.artifacts, "score")
    current = scored_runs(args.artifacts, existing)
    source_provenance = provenance()
    for fixture in args.fixtures or list(FIXTURES):
        selected = [
            (path, record) for path, record in existing.values() if record["fixture"] == fixture
        ]
        if not selected:
            continue
        if all(identity(record) in current for _, record in selected):
            continue
        fixture_hash = datasets.fixture_hash(args.data, fixture)
        records = []
        train, batches, labels = datasets.load_fixture(args.data, fixture)
        cache_errors = quality.verify_cache(args.data, fixture)
        if cache_errors:
            raise ValueError(f"Invalid evaluation cache for {fixture}: {cache_errors}")
        with np.load(args.data / fixture / "eval.npz", allow_pickle=False) as cache:
            if str(cache["fixture_hash"]) != datasets.fixture_hash(args.data, fixture):
                raise ValueError(f"Stale evaluation cache for {fixture}")
            query_ids = cache["query_ids"].copy()
            distances = cache["neighbor_distances"].copy()
        embeddings = {
            str(path): np.load(path.parent / "fit.npy", mmap_mode="r", allow_pickle=False)
            for path, _ in selected
        }
        fit_scores = quality.score_fits(train, embeddings, query_ids=query_ids)
        transform_scores = {}
        for count, queries in () if args.fit_only else batches.items():
            transformed = {}
            for path, record in selected:
                if record["precomputed_knn"]:
                    continue
                transform = path.parent / f"transform_{count}.npy"
                transformed[str(path)] = np.load(transform, mmap_mode="r", allow_pickle=False)
            if transformed:
                fits = {name: embeddings[name] for name in transformed}
                transform_scores[str(count)] = quality.score_transforms(
                    train, fits, queries, transformed
                )
        for path, record in selected:
            fit = embeddings[str(path)]
            metrics = fit_scores[str(path)]
            metrics["diagnostics"] = quality.diagnostics(train, fit, labels=labels)
            transforms = {
                batch: values[str(path)]
                for batch, values in transform_scores.items()
                if str(path) in values
            }
            knn_path = path.parent / "knn_indices.npy"
            recall = None
            if knn_path.exists():
                neighbors = np.load(knn_path, mmap_mode="r", allow_pickle=False)
                recall = quality.knn_recall(train, query_ids, distances, neighbors[query_ids])
            elif record["role"] == "upstream" and len(train) >= 4096:
                raise ValueError(f"Missing upstream ANN neighbor IDs: {knn_path}")
            records.append(
                {
                    "run": str(path.resolve()),
                    "identity": record,
                    "fit": metrics,
                    "transforms": transforms,
                    "knn_recall": recall,
                }
            )
        write_new(
            run / f"scores-{fixture}.json",
            {
                "fixture": fixture,
                "fixture_hash": fixture_hash,
                "records": records,
                "fit_only": args.fit_only,
                "provenance": source_provenance,
            },
        )
    print(run)


def decisions(reference):
    """Read explicit machine-readable decision entries embedded as Markdown list items.

    Each item is a JSON object with date, trigger, observed, option and scope.
    A scope is an exact matrix identity, never a wildcard or an opaque note.
    """
    path = Path(reference) / "decisions.md"
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- ") and not line.startswith("- {"):
            raise ValueError("Decision list items must be explicit JSON objects")
        if not line.startswith("- {"):
            continue
        item = json.loads(line[2:])
        dt.date.fromisoformat(item["date"])
        for field in ("trigger", "observed", "option"):
            if field not in item or item[field] is None or item[field] == "":
                raise ValueError(f"Decision missing {field}: {line}")
        scope = item["scope"]
        if not isinstance(scope, dict):
            raise ValueError(f"Decision scope must be explicit: {line}")
        if "section" in scope and len(scope) == 1 and scope["section"]:
            result.append(item)
            continue
        if item["option"] not in ("drop", "omit_seed", "raise_timeout"):
            raise ValueError(f"Unknown matrix decision option: {line}")
        required = {"role", "fixture", "config", "seed", "components", "precomputed_knn"}
        if scope.get("role") == "cuscan":
            required.add("backend")
        if set(scope) != required:
            raise ValueError(f"Decision scope must name exactly one run: {line}")
        if identity(scope) not in {
            identity(row) for role in ("upstream", "cuscan") for row in selected_matrix(role)
        }:
            raise ValueError(f"Decision scope is not an exact matrix identity: {line}")
        default_timeout = 21600 if scope["fixture"] == "mix_1m" else 1800
        if item["option"] == "raise_timeout" and (
            not isinstance(item.get("timeout_seconds"), int)
            or item["timeout_seconds"] <= default_timeout
        ):
            raise ValueError(f"Timeout decision must exceed the standard timeout: {line}")
        result.append(item)
    scopes = [json.dumps(item["scope"], sort_keys=True) for item in result]
    if len(scopes) != len(set(scopes)):
        raise ValueError("Duplicate decision scopes")
    return result


def scored_runs(artifacts, available):
    result = {}
    current_hashes = {name: sha256(ROOT / name) for name in SCORE_SOURCES}
    paths = [
        *Path(artifacts).glob("*-score-*/scores.json"),
        *Path(artifacts).glob("*-score-*/scores-*.json"),
    ]
    for path in sorted(paths):
        document = read_json(path)
        hashes = document.get("provenance", {}).get("source_hashes", {})
        if any(hashes.get(name) != value for name, value in current_hashes.items()):
            continue
        records = document["records"]
        if "fixture" in document:
            fixture = document["fixture"]
            keys = [identity(record["identity"]) for record in records]
            if len(keys) != len(set(keys)) or any(key[2] != fixture for key in keys):
                raise ValueError(f"Invalid fixture score shard: {path}")
            if not keys:
                raise ValueError(f"Empty fixture score shard: {path}")
            if keys[0] in available:
                invocation = read_json(available[keys[0]][0].parents[4] / "invocation.json")
                if document.get("fixture_hash") != invocation["fixture_hashes"][fixture]:
                    raise ValueError(f"Fixture hash mismatch in score shard: {path}")
        for score_record in records:
            key = identity(score_record["identity"])
            if (
                key not in available
                or Path(score_record["run"]).resolve() != available[key][0].resolve()
            ):
                raise ValueError(f"Score refers to missing or mismatched run: {path}")
            if score_record["identity"] != available[key][1]:
                raise ValueError(f"Score refers to modified run metadata: {path}")
            fit = score_record["fit"]
            for name in ("trustworthiness", "trustworthiness_se", "preservation"):
                if not np.isfinite(fit[name]):
                    raise ValueError(f"Invalid score {name}: {path}")
            if not all(
                name in fit.get("diagnostics", {})
                for name in ("pair_spearman", "axis_std", "bounding_box")
            ):
                raise ValueError(f"Missing fit diagnostics: {path}")
            if (
                key[0] == "upstream"
                and FIXTURES[key[2]].rows >= 4096
                and score_record["knn_recall"] is None
            ):
                raise ValueError(f"Missing upstream kNN recall: {path}")
            expected = (
                set()
                if score_record["identity"]["precomputed_knn"]
                else {str(n) for n in FIXTURES[key[2]].batches}
            )
            if document.get("fit_only", False):
                expected = set()
            if set(score_record["transforms"]) != expected:
                raise ValueError(f"Missing transform scores: {path} {key}")
            for values in score_record["transforms"].values():
                if not all(
                    np.isfinite(values[name])
                    for name in ("trustworthiness", "trustworthiness_se", "preservation_qr")
                ):
                    raise ValueError(f"Invalid transform score: {path} {key}")
            if key in result and result[key] != score_record:
                raise ValueError(f"Conflicting scores for {key}")
            result[key] = score_record
    return result


def aggregate(records):
    groups = {}
    for score_record in records:
        run = score_record["identity"]
        group = (run["fixture"], run["config"], run["components"], run["precomputed_knn"])
        metrics = {name: score_record["fit"][name] for name in ("trustworthiness", "preservation")}
        for batch, values in score_record["transforms"].items():
            metrics[f"transform_{batch}_trustworthiness"] = values["trustworthiness"]
            metrics[f"transform_{batch}_preservation_qr"] = values["preservation_qr"]
        for name, value in metrics.items():
            groups.setdefault(group, {}).setdefault(name, []).append(value)
    result = {}
    for key, metrics in sorted(groups.items()):
        result.setdefault(key[1], []).append(
            {
                "fixture": key[0],
                "configuration": key[1],
                "components": key[2],
                "precomputed_knn": key[3],
                "metrics": {
                    name: {
                        "mean": statistics.mean(values),
                        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                        "seed_count": len(values),
                    }
                    for name, values in metrics.items()
                },
            }
        )
    return result


def determinism_evidence(artifacts, available):
    result = {}
    for path in sorted(Path(artifacts).glob("*-determinism-*/determinism.json")):
        for item in read_json(path)["results"]:
            key = ("upstream", None, item["fixture"], item["config"], 11, 2, False)
            if key not in available or item["original_hashes"] != available[key][1]["hashes"]:
                raise ValueError(f"Determinism original mismatch: {path}")
            repeat = Path(item["repeated"])
            repeated_record = read_json(repeat)
            validate_run(repeat, repeated_record)
            if identity(repeated_record) != key:
                raise ValueError(f"Determinism repeat identity mismatch: {repeat}")
            child_record = read_json(repeat.parents[2] / "child.json")
            if child_record["returncode"] != 0 or child_record["environment"][
                "NUMBA_NUM_THREADS"
            ] != str(item["threads"]):
                raise ValueError(f"Invalid determinism child: {repeat}")
            if repeated_record["hashes"] != item["repeat_hashes"] or item["identical"] != (
                item["original_hashes"] == item["repeat_hashes"]
            ):
                raise ValueError(f"Invalid determinism comparison: {path}")
            result[(item["fixture"], item["config"], item["threads"])] = item
    return result


def stage_evidence(artifacts):
    results = {}
    attempts = {}
    for directory in sorted(Path(artifacts).glob("*-stages-*/stages/*")):
        if not (directory / "child.json").exists():
            continue
        child_record = read_json(directory / "child.json")
        name, backend = directory.name.rsplit("-", 1)
        if name not in ("mix_50k", "mix_200k", "mix_1m") or backend not in (
            "cpu",
            "cuda",
        ):
            raise ValueError(f"Unexpected stage identity: {directory}")
        command = child_record["command"]
        expected_suffix = [
            "--backend",
            backend,
            "--search",
            "approximate",
            "--init",
            "spectral",
            "--seed",
            "11",
            "--runs",
            "3",
            "--trace",
        ]
        if (
            Path(command[0]).name != "profile_standard_stages"
            or Path(command[1]).name != "train.f32"
            or Path(command[1]).parent.name != name
            or command[2:5] != [str(FIXTURES[name].rows), "32", "200"]
            or command[5:-1] != expected_suffix
            or Path(command[-1]).resolve() != (directory / "trace.json").resolve()
        ):
            raise ValueError(f"Stage command mismatch: {directory}")
        stage_provenance = read_json(directory.parents[1] / "provenance.json")
        if stage_provenance["extension"]["sha256"] != require_extension(artifacts)["sha256"]:
            raise ValueError(f"Stage extension mismatch: {directory}")
        fixture_root = Path(command[1]).parents[1]
        if stage_provenance["fixture_hashes"][name] != datasets.fixture_hash(fixture_root, name):
            raise ValueError(f"Stage fixture mismatch: {directory}")
        build = read_json(directory.parents[1] / "build" / "child.json")
        if build["returncode"] != 0:
            raise ValueError(f"Missing successful tracing build: {directory}")
        error = (directory / "stderr.txt").read_text(encoding="utf-8")
        stdout = (directory / "stdout.txt").read_text(encoding="utf-8")
        if child_record["returncode"] == 0:
            output = json.loads(stdout)
            trace = read_json(directory / "trace.json")
            if not output or not trace:
                raise ValueError(f"Empty stage output: {directory}")
            if (
                not output.get("backend", "").split()
                or output["backend"].split()[0].lower() != backend
                or output.get("epochs") != 200
                or output.get("neighbors") != 15
                or len(output.get("fit_seconds", [])) != 3
            ):
                raise ValueError(f"Stage output mismatch: {directory}")
            if not output.get("stage_seconds") or any(
                not samples or any(value <= 0 for value in samples)
                for samples in output["stage_seconds"].values()
            ):
                raise ValueError(f"Missing or zero stage timing: {directory}")
        else:
            attempts.setdefault(directory.name, []).append(
                {
                    "child": child_record,
                    "stderr": error,
                    "directory": str(directory.resolve()),
                    "outcome": "failed",
                }
            )
            results.pop(directory.name, None)
            continue
        trace_digest = sha256(directory / "trace.json") if trace is not None else None
        attempt = {
            "child": child_record,
            "output": output,
            "trace_sha256": trace_digest,
            "stderr": error,
            "directory": str(directory.resolve()),
            "outcome": "success"
            if child_record["returncode"] == 0
            else "memory_or_planning_failure",
        }
        attempts.setdefault(directory.name, []).append(attempt)
        results[directory.name] = {**attempt, "attempts": list(attempts[directory.name])}
    return results


def summary_data(args):
    available = runs(args.artifacts, args.data)
    scores = scored_runs(args.artifacts, available)
    decision_list = decisions(args.reference)
    by_role = {
        role: [value for key, value in scores.items() if key[0] == role]
        for role in ("upstream", "cuscan")
    }
    upstream = aggregate(by_role["upstream"])
    margins_input = []
    batch_inputs = {}
    for record in by_role["upstream"]:
        run = record["identity"]
        if run["components"] != 2 or run["precomputed_knn"]:
            continue
        configuration = f"{run['config']}:{run['components']}:{run['precomputed_knn']}"
        margins_input.append(
            {
                "fixture": run["fixture"],
                "configuration": configuration,
                "trustworthiness": record["fit"]["trustworthiness"],
                "preservation": record["fit"]["preservation"],
            }
        )
        for batch, values in record["transforms"].items():
            batch_inputs.setdefault(batch, []).append(
                {
                    "fixture": run["fixture"],
                    "configuration": configuration,
                    "transform_trustworthiness": values["trustworthiness"],
                    "preservation_qr": values["preservation_qr"],
                }
            )

    def pooled(entries):
        counts = {}
        for entry in entries:
            for metric in set(entry) - {"fixture", "configuration"}:
                key = (entry["fixture"], entry["configuration"], metric)
                counts[key] = counts.get(key, 0) + 1
        eligible = []
        for entry in entries:
            filtered = {"fixture": entry["fixture"], "configuration": entry["configuration"]}
            filtered.update(
                {
                    metric: value
                    for metric, value in entry.items()
                    if metric not in filtered
                    and counts.get((entry["fixture"], entry["configuration"], metric), 0) >= 2
                }
            )
            if len(filtered) > 2:
                eligible.append(filtered)
        return quality.acceptance_margins(eligible) if eligible else {}

    margins = pooled(margins_input)
    transform_margins = {batch: pooled(entries) for batch, entries in batch_inputs.items()}
    stage_results = stage_evidence(args.artifacts)
    deterministic = list(determinism_evidence(args.artifacts, available).values())
    current_provenance = provenance()
    current_provenance.update(wheel_provenance())
    current_provenance["fixture_hashes"] = {
        name: datasets.fixture_hash(args.data, name)
        for name in FIXTURES
        if (args.data / name / "train.f32").exists()
    }
    baseline = {
        "schema": "cuscan-umap06-upstream-reference-v1",
        "provenance": current_provenance,
        "runs": by_role["upstream"],
        "aggregates": upstream,
        "configurations": {
            name: {
                "optimizer": config.optimizer,
                "init": config.init,
                "aggregates": upstream.get(name, []),
            }
            for name, config in CONFIGS.items()
        },
        "determinism": deterministic,
        "acceptance_rule": {
            "formula": "candidate >= upstream - margin",
            "margins": margins,
            "transform_margins_by_batch": transform_margins,
        },
        "decisions": decision_list,
    }
    devices: dict[str, dict[str, Any]] = {}
    for device in ("cpu", "cuda"):
        records = [
            record for record in by_role["cuscan"] if record["identity"]["backend"] == device
        ]
        timing_groups = {}
        for scored in records:
            record = scored["identity"]
            group = (
                record["fixture"],
                record["config"],
                record["components"],
                record["precomputed_knn"],
            )
            samples = timing_groups.setdefault(
                group, {"cold_fit": [], "warm_fit": [], "transforms": {}}
            )
            samples[f"{record['fit_temperature']}_fit"].append(record["fit_seconds"])
            for batch, values in record["transform_seconds"].items():
                transform = samples["transforms"].setdefault(batch, {"cold": [], "warm": []})
                transform["cold"].append(values["cold"])
                transform["warm"].extend(values["warm"])

        def with_median(values):
            return {"samples": values, "median": statistics.median(values) if values else None}

        timings = []
        for group, values in timing_groups.items():
            timings.append(
                {
                    "fixture": group[0],
                    "configuration": group[1],
                    "components": group[2],
                    "precomputed_knn": group[3],
                    "cold_fit": with_median(values["cold_fit"]),
                    "warm_fit": with_median(values["warm_fit"]),
                    "transforms": {
                        batch: {kind: with_median(samples) for kind, samples in value.items()}
                        for batch, value in values["transforms"].items()
                    },
                }
            )
        devices[device] = {"runs": records, "aggregates": aggregate(records), "timings": timings}
    agreement = []
    for key, (path, record) in available.items():
        role, backend, fixture, config, seed, components, exact = key
        other = (role, "cuda", fixture, config, seed, components, exact)
        if role != "cuscan" or backend != "cpu" or seed != 11 or other not in available:
            continue
        a = np.load(path.parent / "fit.npy", mmap_mode="r", allow_pickle=False)
        b = np.load(available[other][0].parent / "fit.npy", mmap_mode="r", allow_pickle=False)
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        differences = {}
        if key in scores and other in scores:
            differences = {
                metric: scores[key]["fit"][metric] - scores[other]["fit"][metric]
                for metric in ("trustworthiness", "preservation")
            }
            for batch, values in scores[key]["transforms"].items():
                for metric in ("trustworthiness", "preservation_qr"):
                    differences[f"transform_{batch}_{metric}"] = (
                        values[metric] - scores[other]["transforms"][batch][metric]
                    )
        agreement.append(
            {
                "identity": record,
                "max_abs_difference": float(np.max(delta)),
                "bit_identical_fraction": float(np.mean(a.view("u4") == b.view("u4"))),
                "metric_differences": differences,
            }
        )
    cuscan = {
        "schema": "cuscan-umap06-cuscan-baseline-v1",
        "backends": devices,
        "epoch_provenance": (
            "Cuscan worker epochs are calculated from default rules, not observed optimizer calls."
        ),
        "stage_attribution": stage_results,
        "cpu_cuda_agreement": agreement,
        "build_provenance": current_provenance,
        "extension": require_extension(args.artifacts),
        "decisions": decision_list,
    }
    comparisons = []
    upstream_index = {
        (item["fixture"], item["configuration"], item["components"], item["precomputed_knn"]): item
        for groups in upstream.values()
        for item in groups
    }
    for device, data in devices.items():
        for group in (item for groups in data["aggregates"].values() for item in groups):
            key = (
                group["fixture"],
                group["configuration"],
                group["components"],
                group["precomputed_knn"],
            )
            if key not in upstream_index:
                continue
            for metric, value in group["metrics"].items():
                other = upstream_index[key]["metrics"].get(metric)
                margin_key = (
                    "transform_trustworthiness"
                    if metric.startswith("transform_") and metric.endswith("trustworthiness")
                    else "preservation_qr"
                    if metric.startswith("transform_")
                    else metric
                )
                if metric.startswith("transform_"):
                    batch = metric.split("_", 2)[1]
                    margin = (
                        transform_margins.get(batch, {}).get(group["fixture"], {}).get(margin_key)
                    )
                else:
                    margin = margins.get(group["fixture"], {}).get(margin_key)
                if other is not None and margin is not None:
                    comparisons.append(
                        {
                            "backend": device,
                            "group": list(key),
                            "metric": metric,
                            "candidate": value["mean"],
                            "upstream": other["mean"],
                            "margin": margin,
                            "passes": quality.acceptance_pass(value["mean"], other["mean"], margin),
                        }
                    )
    cuscan["diagnostic_acceptance"] = comparisons
    return baseline, cuscan


def summarize(args):
    upstream, cuscan = summary_data(args)
    run = new_run(args.artifacts, "summarize")
    write_new(run / "upstream.json", upstream)
    write_new(run / "cuscan.json", cuscan)
    if args.freeze:
        errors = evidence_errors(args)
        if errors:
            raise RuntimeError("Cannot freeze an incomplete campaign: " + "; ".join(errors))
        revision = upstream["provenance"]["git_revision"]
        if revision["returncode"] != 0:
            raise RuntimeError("Cannot freeze without a git revision")
        rev = revision["stdout"].strip()
        paths = [args.reference / "upstream-0.6.0.json", args.reference / f"cuscan-{rev}.json"]
        if any(path.exists() for path in paths):
            raise FileExistsError("Frozen reference already exists; refusing overwrite")
        write_new(paths[0], upstream)
        write_new(paths[1], cuscan)
    print(run)


def phase2_acceptance(args, *, phase3=False, phase4=False, phase5=False, phase7=False):
    if sum((phase3, phase4, phase5, phase7)) > 1:
        raise ValueError("acceptance phases are mutually exclusive")
    phase = 7 if phase7 else 5 if phase5 else 4 if phase4 else 3 if phase3 else 2
    scoped = phase5 or phase7 or (phase4 and getattr(args, "scope", None) == "halved-fit")
    # Phases 4, 5 and 7 report evidence-scoped, incomplete-aware receipts.
    evidence_scoped = phase4 or phase5 or phase7
    result = {
        "schema": f"cuscan-umap06-phase{phase}-acceptance-v1",
        "upstream_sha256": args.upstream_sha256,
        "comparisons": [],
        "errors": [],
    }
    errors = result["errors"]
    incomplete = []
    scope_fixtures: set[str] = set()
    if evidence_scoped:
        result["incomplete"] = incomplete
    reference_path = Path(args.upstream_reference)
    try:
        digest = sha256(reference_path)
        if digest != args.upstream_sha256:
            raise ValueError("frozen upstream sha256 mismatch")
        reference = read_json(reference_path)
        if reference["schema"] != "cuscan-umap06-upstream-reference-v1":
            raise ValueError("unexpected upstream reference schema")
        frozen_provenance = reference["provenance"]
        wheel = wheel_provenance()
        if any(frozen_provenance.get(key) != value for key, value in wheel.items()):
            raise ValueError("upstream wheel provenance mismatch")
        for name in (
            "comparison/umap_reference.py",
            "comparison/umap06/spec.py",
            "comparison/umap06/datasets.py",
            "comparison/umap06/upstream_worker.py",
        ):
            expected_hash = (
                phase7_frozen_worker_hash()
                if phase7 and name == "comparison/umap06/upstream_worker.py"
                else sha256(ROOT / name)
            )
            if frozen_provenance["source_hashes"].get(name) != expected_hash:
                raise ValueError(f"upstream source provenance mismatch: {name}")
        for name, config in CONFIGS.items():
            entry = reference["configurations"][name]
            if (entry["optimizer"], entry["init"]) != (config.optimizer, config.init):
                raise ValueError(f"upstream configuration provenance mismatch: {name}")
            if entry["aggregates"] != reference["aggregates"].get(name, []):
                raise ValueError(f"upstream configuration aggregates mismatch: {name}")
        observed_configs = set()
        for record in reference["runs"]:
            run_record = record["identity"]
            config = CONFIGS[run_record["config"]]
            observed_configs.add(run_record["config"])
            params = run_record["params"]
            if (
                run_record["resolved_optimizer"] != config.optimizer
                or run_record["resolved_init"] != config.init
                or params["compatibility_layout"] is not False
                or params["nearest_neighbor_backend"] != "pynndescent"
            ):
                raise ValueError("upstream run configuration provenance mismatch")
        if observed_configs != set(CONFIGS):
            raise ValueError("missing upstream configuration run provenance")
        if phase7:
            revised_scope = getattr(args, "seed_11_no_mix_1m", False)
            if revised_scope and args.full_seeds:
                raise ValueError("seed-11-no-mix-1m and full-seeds are mutually exclusive")
            source_rows = phase7_runs() if args.full_seeds else phase7_halved_runs()
            if revised_scope:
                source_rows = [
                    row for row in source_rows if row["seed"] == 11 and row["fixture"] != "mix_1m"
                ]
        elif phase5:
            source_rows = phase5_runs(halved=not args.full_seeds)
        elif phase4:
            source_rows = phase4_halved_runs() if scoped else phase4_runs()
        elif phase3:
            source_rows = phase3_runs()
        else:
            source_rows = expected_runs("cuscan")
        expected_rows = [
            dict(row, role="cuscan")
            for row in source_rows
            if phase3 or evidence_scoped or row["config"] in ("classic_spectral", "classic_random")
        ]
        if evidence_scoped:
            # Evidence-scoped reporting: excluded fixtures are never evaluated and always
            # leave the report incomplete and not accepted. No fixture is exempt.
            required = sorted({row["fixture"] for row in expected_rows})
            evidence = getattr(args, "evidence_fixtures", None)
            scope = required if evidence is None else sorted(set(evidence))
            if set(scope) - set(required):
                raise ValueError(f"unknown evidence fixtures: {sorted(set(scope) - set(required))}")
            excluded = sorted(set(required) - set(scope))
            settings = requested_settings(args)
            result.update(
                required_fixtures=required,
                evidence_fixtures=scope,
                excluded_fixtures=excluded,
                settings=settings,
                known_exceptions=sorted(map(list, PHASE2_KNOWN_EXCEPTIONS)),
                scope=(
                    "seed-11-no-mix-1m"
                    if phase7 and revised_scope
                    else "full-seeds"
                    if (phase5 or phase7) and args.full_seeds
                    else "halved-fit"
                    if scoped
                    else "full"
                ),
                expected_identities=expected_rows,
            )
            scope_fixtures = set(scope)
            if excluded:
                incomplete.append(
                    f"fixtures not evaluated (evidence-scoped report only): {excluded}"
                )
            expected_rows = [row for row in expected_rows if row["fixture"] in scope]
        expected = {identity(row) for row in expected_rows}
        if not expected or len(expected) != len(expected_rows):
            raise ValueError("empty or duplicate expected run identities")
        for fixture in sorted({row["fixture"] for row in expected_rows}):
            if (
                datasets.fixture_hash(args.data, fixture)
                != frozen_provenance["fixture_hashes"][fixture]
            ):
                raise ValueError(f"frozen fixture hash mismatch: {fixture}")
            cache_errors = quality.verify_cache(args.data, fixture)
            if cache_errors:
                raise ValueError(f"evaluation cache {fixture}: {cache_errors}")
        fixture_errors = datasets.verify(args.data)
        if fixture_errors:
            raise ValueError(f"fixture validation: {fixture_errors}")
        require_extension(args.artifacts)
        available = runs(args.artifacts, args.data)
        actual = {
            key
            for key in available
            if key[0] == "cuscan" and (not evidence_scoped or key[2] in scope_fixtures)
        }
        if scoped:
            missing = expected - actual
            if missing:
                incomplete.append(f"missing run identities: {sorted(missing)}")
            actual &= expected
        elif actual != expected:
            raise ValueError(
                f"run identities mismatch: missing={sorted(expected - actual)} "
                f"extra={sorted(actual - expected)}"
            )
        for key in sorted(actual if scoped else expected):
            path, record = available[key]
            if evidence_scoped and "settings" not in record:
                raise ValueError(f"run lacks recorded settings: {path}")
            wanted = result["settings"] if evidence_scoped else DEFAULT_SETTINGS
            if record_settings(record) != wanted:
                raise ValueError(f"run settings differ from the evaluated settings: {path}")
        fit_only = bool(phase3 or phase4 or phase5 or args.fit_only)
        if evidence_scoped:
            unscored = set()
            result["fit_only"] = fit_only
        else:
            unscored = set(args.unscored_fixtures or ())
            result["unscored_fixtures"] = sorted(unscored)
            result["fit_only"] = fit_only
            result["exemption_reason"] = args.exemption_reason
            if unscored - {"mix_1m"}:
                raise ValueError("only mix_1m may be unscored (plan section 9.1 amendment)")
            if (unscored or args.fit_only) and not (args.exemption_reason or "").strip():
                raise ValueError("--unscored-fixtures and --fit-only require --exemption-reason")
        scored_expected = {key for key in expected if key[2] not in unscored}
        if not scored_expected:
            raise ValueError("every fixture is unscored")
        scores = scored_runs(args.artifacts, available)
        scored_actual = {
            key
            for key in scores
            if key[0] == "cuscan"
            and key[2] not in unscored
            and (not evidence_scoped or key[2] in scope_fixtures)
        }
        if scoped:
            missing_scores = scored_expected - scored_actual
            if missing_scores:
                incomplete.append(f"missing candidate scores: {sorted(missing_scores)}")
            groups = {}
            for key in scored_expected:
                groups.setdefault((key[1], key[2], key[3], key[5], key[6]), set()).add(key)
            scored_expected = set().union(
                *(keys for keys in groups.values() if keys <= scored_actual and keys <= actual)
            )
            result["not_measured"] = [
                list(group)
                for group, keys in groups.items()
                if not (keys <= scored_actual and keys <= actual)
            ]
            if not phase7:
                result["transform_metrics"] = "not measured"
        elif scored_actual != scored_expected:
            raise ValueError("missing or extra candidate scores")
        upstream = {}
        for config_name, groups in reference["aggregates"].items():
            for group in groups:
                key = (
                    group["fixture"],
                    group["configuration"],
                    group["components"],
                    group["precomputed_knn"],
                )
                if key in upstream or key[1] != config_name:
                    raise ValueError(f"duplicate or invalid upstream aggregate: {key}")
                upstream[key] = group
        rule = reference["acceptance_rule"]
        if phase5:
            reporting_variants: list[dict[str, Any]] = []
            result["reporting_variants"] = reporting_variants
            for backend in ("cpu", "cuda"):
                variant_scores = [
                    value
                    for key, value in scores.items()
                    if key[0] == "cuscan"
                    and key[1] == backend
                    and key[2] in scope_fixtures
                    and (key[3] in VARIANT_CONFIGS or (key[3] == "adam_recursive" and key[5] != 2))
                ]
                variant_groups = aggregate(variant_scores)
                for values in variant_groups.values():
                    for group in values:
                        spectral = {
                            "classic_recursive": "classic_spectral",
                            "momentum_recursive": "momentum_spectral",
                            "adam_recursive": "adam_spectral",
                        }[group["configuration"]]
                        ref_key = (
                            group["fixture"],
                            spectral,
                            group["components"],
                            group["precomputed_knn"],
                        )
                        if ref_key not in upstream:
                            ref_key = (group["fixture"], spectral, 2, False)
                        reporting_variants.append(
                            {
                                "backend": backend,
                                "group": group,
                                "reference_group": list(ref_key),
                                "reference_metrics": upstream[ref_key]["metrics"],
                                "gating": False,
                            }
                        )
        for backend in ("cpu", "cuda"):
            groups = aggregate(
                [scores[key] for key in sorted(scored_expected) if key[1] == backend]
            )
            for group in (group for values in groups.values() for group in values):
                key = (
                    group["fixture"],
                    group["configuration"],
                    group["components"],
                    group["precomputed_knn"],
                )
                frozen = upstream[key]
                count = len(FIXTURES[key[0]].seeds)
                candidate_count = sum(
                    1
                    for row in expected_rows
                    if row["backend"] == backend
                    and row["fixture"] == key[0]
                    and row["config"] == key[1]
                    and row["components"] == key[2]
                    and row["precomputed_knn"] == key[3]
                )
                metrics = [
                    ("trustworthiness", rule["margins"][key[0]]["trustworthiness"]),
                    ("preservation", rule["margins"][key[0]]["preservation"]),
                ]
                if not key[3] and (phase7 or not scoped):
                    for batch in FIXTURES[key[0]].batches:
                        margins = rule["transform_margins_by_batch"][str(batch)][key[0]]
                        metrics.extend(
                            [
                                (
                                    f"transform_{batch}_trustworthiness",
                                    margins["transform_trustworthiness"],
                                ),
                                (f"transform_{batch}_preservation_qr", margins["preservation_qr"]),
                            ]
                        )
                for metric, margin in metrics:
                    if phase7 and metric not in group["metrics"]:
                        # Scored without transforms: the target is unmeasured, never dropped.
                        result["comparisons"].append(
                            {
                                "backend": backend,
                                "group": list(key),
                                "metric": metric,
                                "margin": margin,
                                "target": "not measured",
                                "gating": False,
                            }
                        )
                        incomplete.append(
                            f"transform metric not measured: {backend} {key} {metric}"
                        )
                        continue
                    candidate = group["metrics"][metric]
                    baseline = frozen["metrics"][metric]
                    values = (candidate["mean"], baseline["mean"], margin)
                    if (
                        not all(
                            isinstance(v, (int, float))
                            and not isinstance(v, bool)
                            and np.isfinite(v)
                            for v in values
                        )
                        or margin < 0
                        or candidate["seed_count"] != candidate_count
                        or baseline["seed_count"] != count
                    ):
                        raise ValueError(
                            f"invalid metric, margin or seed count: {backend} {key} {metric}"
                        )
                    passed = quality.acceptance_pass(candidate["mean"], baseline["mean"], margin)
                    known = (backend, *key, metric) in PHASE2_KNOWN_EXCEPTIONS
                    gating = not (fit_only and metric.startswith("transform_")) and not known
                    result["comparisons"].append(
                        {
                            "backend": backend,
                            "group": list(key),
                            "metric": metric,
                            "candidate_mean": candidate["mean"],
                            "upstream_mean": baseline["mean"],
                            "candidate_seed_count": candidate["seed_count"],
                            "upstream_seed_count": baseline["seed_count"],
                            "margin": margin,
                            "passed": passed,
                            "gating": gating,
                            "known_exception": known,
                            **({"target": "met" if passed else "missed"} if phase7 else {}),
                        }
                    )
                    if gating and not passed:
                        errors.append(f"quality inequality failed: {backend} {key} {metric}")
        if phase7:
            result["query_recall"] = phase7_query_recall(
                args, available, actual, expected_rows, incomplete, errors
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError, IndexError) as exc:
        errors.append(f"validation: {exc}")
    try:
        if sha256(reference_path) != args.upstream_sha256:
            errors.append("frozen upstream reference changed or sha256 mismatch")
    except OSError as exc:
        errors.append(f"frozen upstream reference: {exc}")
    result["total_comparisons"] = len(result["comparisons"])
    if not result["total_comparisons"]:
        errors.append("no comparisons")
    if evidence_scoped:
        result["complete"] = not incomplete
        result["scoped_quality_passed"] = not errors
        result["passed"] = not errors and not incomplete
    else:
        result["passed"] = not errors
    return result


QUERY_REFERENCE_SCHEMA = "cuscan-umap06-upstream-query-reference-v1"


def query_neighbors_digest(values):
    """Hash recorded for a run's ``query_neighbors_<batch>.npy`` (little-endian int64)."""
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<i8").tobytes()).hexdigest()


def load_query_reference(path, expected_sha256):
    """Upstream query recall rows keyed by (fixture, seed, batch).

    Schema: {"schema": QUERY_REFERENCE_SCHEMA, "rows": [{"fixture", "seed", "batch",
    "n_neighbors": 15, "epsilon": 0.12, "tie_rule", "train_sha256", "query_sha256",
    "recall"}]}; hashes are ``quality.content_sha256`` of the fixture arrays.
    """
    if sha256(path) != expected_sha256:
        raise ValueError("upstream query reference sha256 mismatch")
    document = read_json(path)
    if document.get("schema") != QUERY_REFERENCE_SCHEMA:
        raise ValueError("unexpected upstream query reference schema")
    rows = {}
    for row in document["rows"]:
        key = (row["fixture"], int(row["seed"]), int(row["batch"]))
        recall = row["recall"]
        if (
            key in rows
            or row["n_neighbors"] != 15
            or row["epsilon"] != 0.12
            or isinstance(recall, bool)
            or not isinstance(recall, (int, float))
            or not 0 <= recall <= 1
        ):
            raise ValueError(f"duplicate or invalid upstream query reference row: {key}")
        rows[key] = row
    return rows


def phase7_query_recall(args, available, actual, expected_rows, incomplete, errors):
    """T3: mean recall@15 per approximate-search fixture and backend (batch/seed breakdown).

    cuscan recall comes from each run's ``query_neighbors_<batch>.npy``; a row is paired with
    an upstream row only when fixture, seed, batch, tie rule and content hashes match. Rows on
    either side without a pair are gaps; a target with no pairs is not measured.
    """
    # Auto resolves to exact search at or below 4096 training rows; recall applies above it.
    approximate = [
        row
        for row in expected_rows
        if FIXTURES[row["fixture"]].rows > 4096 and not row["precomputed_knn"]
    ]
    reference_path = getattr(args, "query_reference", None)
    reference_sha256 = getattr(args, "query_reference_sha256", None)
    reference = None
    if reference_path is not None:
        reference = load_query_reference(reference_path, reference_sha256)
    elif approximate:
        incomplete.append("upstream query reference not supplied; T3 not measured")
    content = {}
    exact = {}
    rows = []
    for row in approximate:
        key = identity(row)
        fixture = row["fixture"]
        for batch in FIXTURES[fixture].batches:
            entry = {
                "backend": row["backend"],
                "fixture": fixture,
                "config": row["config"],
                "components": row["components"],
                "seed": row["seed"],
                "batch": batch,
                "cuscan_recall": None,
                "upstream_recall": None,
                "paired": False,
            }
            rows.append(entry)
            if key not in actual:
                entry["reason"] = "run not measured"
                continue
            path, record = available[key]
            neighbors_path = path.parent / f"query_neighbors_{batch}.npy"
            if not neighbors_path.is_file():
                entry["reason"] = "query neighbor IDs not recorded"
                continue
            ids = np.load(neighbors_path, allow_pickle=False)
            if query_neighbors_digest(ids) != record["hashes"].get(f"query_neighbors_{batch}"):
                raise ValueError(f"query neighbor hash mismatch: {neighbors_path}")
            if fixture not in content:
                train, batches, _ = datasets.load_fixture(args.data, fixture)
                content[fixture] = (train, batches)
            train, batches = content[fixture]
            if (fixture, batch) not in exact:
                exact[(fixture, batch)] = quality.exact_neighbors(train, batches[batch], k=15)[0]
            measured = quality.query_recall(
                train, batches[batch], ids, k=15, exact_ids=exact[(fixture, batch)]
            )
            entry["cuscan_recall"] = measured["recall"]
            entry["missing"] = measured["missing"]
            entry["train_sha256"] = quality.content_sha256(train)
            entry["query_sha256"] = quality.content_sha256(batches[batch])
            if reference is None:
                entry["reason"] = "no upstream query reference"
                continue
            upstream = reference.get((fixture, row["seed"], batch))
            if upstream is None:
                entry["reason"] = "no upstream query row"
            elif (
                upstream["tie_rule"] != quality.QUERY_TIE_RULE
                or upstream["train_sha256"] != entry["train_sha256"]
                or upstream["query_sha256"] != entry["query_sha256"]
            ):
                entry["reason"] = "content hash or tie rule differs from upstream"
            else:
                entry["upstream_recall"] = float(upstream["recall"])
                entry["paired"] = True
    targets = []
    groups: dict[tuple, list[dict]] = {}
    for entry in rows:
        groups.setdefault((entry["backend"], entry["fixture"]), []).append(entry)

    def summary(entries):
        paired = [entry for entry in entries if entry["paired"]]
        result = {
            "paired_rows": len(paired),
            "expected_rows": len(entries),
            "margin": PHASE7_RECALL_MARGIN,
        }
        if not paired:
            result["target"] = "not measured"
            return result
        cuscan_mean = statistics.mean(entry["cuscan_recall"] for entry in paired)
        upstream_mean = statistics.mean(entry["upstream_recall"] for entry in paired)
        met = cuscan_mean >= upstream_mean - PHASE7_RECALL_MARGIN
        result.update(
            cuscan_mean=cuscan_mean,
            upstream_mean=upstream_mean,
            target="met" if met else "missed",
        )
        return result

    # T3 target: one mean per fixture and backend; batches and seeds are a breakdown.
    for (backend, fixture), entries in sorted(groups.items()):
        target = {"backend": backend, "fixture": fixture, **summary(entries)}
        target["by_batch"] = [
            {"batch": batch, **summary([e for e in entries if e["batch"] == batch])}
            for batch in sorted({e["batch"] for e in entries})
        ]
        target["by_seed"] = [
            {"seed": seed, **summary([e for e in entries if e["seed"] == seed])}
            for seed in sorted({e["seed"] for e in entries})
        ]
        if target["target"] == "missed":
            errors.append(f"query recall target missed: {backend} {fixture}")
        if target["paired_rows"] != target["expected_rows"]:
            incomplete.append(
                f"query recall partially measured: {backend} {fixture} "
                f"({target['paired_rows']}/{target['expected_rows']} rows paired)"
            )
        targets.append(target)
    cuscan_keys = {(row["fixture"], row["seed"], row["batch"]) for row in rows}
    upstream_unpaired = sorted(list(key) for key in (reference or {}) if key not in cuscan_keys)
    return {
        "k": 15,
        "tie_rule": quality.QUERY_TIE_RULE,
        "reference_sha256": reference_sha256,
        "rows": rows,
        "targets": targets,
        "upstream_unpaired": upstream_unpaired,
    }


# User decision 2026-11-16 (plan section 9.1 amendment): the CUDA random-init fit gap on
# curve_20k predates Phase 2 and arises after initialization; it is reported, not gating.
PHASE2_KNOWN_EXCEPTIONS = frozenset(
    {("cuda", "curve_20k", "classic_random", 2, False, "preservation")}
)


def accept_phase2(args):
    run = new_run(args.artifacts, "accept-phase2")
    result = phase2_acceptance(args)
    write_new(run / "acceptance.json", result)
    for error in result["errors"]:
        print(error, file=sys.stderr)
    print(run)
    if not result["passed"]:
        raise SystemExit(1)


def accept_phase3(args):
    run = new_run(args.artifacts, "accept-phase3")
    result = phase2_acceptance(args, phase3=True)
    write_new(run / "acceptance.json", result)
    for error in result["errors"]:
        print(error, file=sys.stderr)
    print(run)
    if not result["passed"]:
        raise SystemExit(1)


def accept_phase4(args):
    run = new_run(args.artifacts, "accept-phase4")
    result = phase2_acceptance(args, phase4=True)
    write_new(run / "acceptance.json", result)
    for error in [*result["errors"], *result["incomplete"]]:
        print(error, file=sys.stderr)
    print(run)
    if not result["passed"]:
        raise SystemExit(1)


def accept_phase5(args):
    run = new_run(args.artifacts, "accept-phase5")
    result = phase2_acceptance(args, phase5=True)
    write_new(run / "acceptance.json", result)
    print(run)
    if not result["passed"]:
        raise SystemExit(1)


def accept_phase7(args):
    if (args.query_reference is None) != (args.query_reference_sha256 is None):
        raise SystemExit("--query-reference and --query-reference-sha256 go together")
    run = new_run(args.artifacts, "accept-phase7")
    result = phase2_acceptance(args, phase7=True)
    write_new(run / "acceptance.json", result)
    for error in [*result["errors"], *result["incomplete"]]:
        print(error, file=sys.stderr)
    print(run)
    if not result["passed"]:
        raise SystemExit(1)


def hash_records(root):
    root = Path(root)
    manifest = root / "manifest.json"
    records = {}
    if manifest.exists() and "items" in (document := read_json(manifest)):
        for item in document["items"]:
            if item.get("kind") != "probe" or item.get("outcome") != "success":
                continue
            key = json.dumps([item["backend"], item["identity"]], sort_keys=True)
            records.setdefault(key, []).append(item["hashes"])
    else:
        for path in root.glob("*/children/*/output/*/run.json"):
            record = read_json(path)
            key = json.dumps(identity(record))
            records.setdefault(key, []).append(record["hashes"])
    return records


def side_items(root):
    """Return campaign-manifest probe items grouped by (backend, identity)."""
    manifest = Path(root) / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"missing campaign manifest: {manifest}")
    document = read_json(manifest)
    if not isinstance(document.get("items"), list):
        raise SystemExit(f"campaign manifest without items: {manifest}")
    items = {}
    for item in document["items"]:
        if item.get("kind") != "probe":
            continue
        key = json.dumps([item["backend"], item["identity"]], sort_keys=True)
        items.setdefault(key, []).append(item)
    return items


def side_state(items):
    """Split one identity's items into successful hashes and per-replay failure reasons."""
    successes = [item["hashes"] for item in items if item.get("outcome") == "success"]
    reasons = [
        {"replay": item.get("replay"), "outcome": item.get("outcome", "missing_outcome")}
        for item in items
        if item.get("outcome") != "success"
    ]
    if not items:
        reasons.append({"replay": None, "outcome": "absent"})
    return successes, reasons


def projected(hashes, keys):
    return {key: hashes.get(key) for key in keys}


def status_counts(rows):
    counts = dict.fromkeys(("matched", "mismatched", "unmeasured", "not_comparable"), 0)
    for row in rows:
        counts[row["status"]] += 1
    return counts


def compare_expected(args):
    """Compare the planned identity list of an expected manifest, not the records found.

    Expected manifest: {"items": [{"backend": str, "identity": {probe}, "hash_keys":
    ["fit", "transform_<batch>", ..., "transform_self"?]}, ...]} in planned order. Each
    side is a campaign output root whose manifest.json lists probe items with backend,
    identity, replay, outcome and (on success) hashes.
    """
    expected = read_json(args.expected)
    baseline, candidate = side_items(args.baseline), side_items(args.candidate)
    planned, rows = set(), []
    for entry in expected["items"]:
        backend, probe, keys = entry["backend"], entry["identity"], list(entry["hash_keys"])
        if not keys:
            raise SystemExit(f"expected item without hash keys: {entry}")
        key = json.dumps([backend, probe], sort_keys=True)
        if key in planned:
            raise SystemExit(f"duplicate expected item: {entry}")
        planned.add(key)
        left, left_reasons = side_state(baseline.get(key, []))
        right, right_reasons = side_state(candidate.get(key, []))
        classic = probe.get("optimizer") == "compatibility"
        row = {
            "backend": backend,
            "identity": probe,
            "hash_keys": keys,
            "classic": classic,
            "baseline": {"hashes": left, "unsuccessful": left_reasons},
            "candidate": {"hashes": right, "unsuccessful": right_reasons},
        }
        if not left or not right:
            row["status"] = "unmeasured"
            row["reason"] = {
                "baseline": left_reasons if not left else None,
                "candidate": right_reasons if not right else None,
            }
        elif classic and (len(left) < 2 or any(value != left[0] for value in left[1:])):
            row["status"] = "not_comparable"
            row["reason"] = "classic baseline lacks two successful identical replays"
        else:
            reference = projected(left[0], keys)
            problems = []
            for side, values in (("baseline", left), ("candidate", right)):
                for position, hashes in enumerate(values):
                    if missing := [name for name in keys if name not in hashes]:
                        problems.append(f"{side}[{position}] missing {missing}")
                    elif projected(hashes, keys) != reference:
                        differing = [name for name in keys if hashes[name] != reference[name]]
                        problems.append(f"{side}[{position}] differs on {differing}")
            row["status"] = "mismatched" if problems else "matched"
            if problems:
                row["reason"] = problems
        rows.append(row)
    unplanned = sorted((baseline.keys() | candidate.keys()) - planned)
    counts = status_counts(rows)
    report = {
        "mode": "expected",
        "expected": str(args.expected),
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "results": rows,
        "counts": counts,
        "unplanned": [json.loads(key) for key in unplanned],
        "passed": counts["mismatched"] == 0 and counts["matched"] > 0,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


VARIANTS = {"explicit": "adam_recursive", "auto": "adam_auto", "library": "default"}
VARIANT_KEY_FIELDS = (
    "backend",
    "fixture",
    "seed",
    "components",
    "neighbor_search",
    "precomputed_knn",
    "init_rescale",
    "epochs",
    "settings",
)


# Native metadata each D6 variant record must export: (init, optimizer, library_defaults).
VARIANT_METADATA = {
    "explicit": ("recursive", "adam", False),
    "auto": ("auto", "adam", False),
    "library": ("auto", "adam", True),
}
VARIANT_IDENTITY = ("backend", "fixture", "seed")


def planned_variant_hash_keys(item):
    """Required hash keys for a scheduled D6 variant item.

    Uses the item's explicit "hash_keys" when recorded; otherwise derives them from the
    scheduled worker argv: "fit" plus transform_<n> for each --transform-batches entry
    unless --skip-transform. Returns None when they cannot be established.
    """
    if "hash_keys" in item:
        keys = item["hash_keys"]
        if not isinstance(keys, list) or "fit" not in keys:
            return None
        return sorted(keys)
    command = item.get("command")
    if not isinstance(command, list) or "--transform-batches" not in command:
        return None
    keys = ["fit"]
    if "--skip-transform" not in command:
        start = command.index("--transform-batches") + 1
        for value in command[start:]:
            if value.startswith("--"):
                break
            keys.append(f"transform_{value}")
        if len(keys) == 1:
            return None
    if "--transform-self" in command:
        keys.append("transform_self")
    return sorted(keys)


def compare_variants(args):
    """Compare D6 variants <dir>/explicit, <dir>/auto and <dir>/library from a plan.

    --manifest (campaign manifest) is required: its items {"kind": "variant", "variant",
    "backend", "fixture", "seed", "outcome", "record", "record_sha256", "command" and
    optionally "hash_keys"} define every expected identity (backend, fixture, seed).
    Each identity must schedule all three variants with the same required hash keys;
    each variant needs a successful item whose record lies under its root, matches its
    recorded digest, carries the variant's config label and exported init/optimizer
    metadata, and has exactly the required hash keys. Records under the roots that no
    plan item references are errors.
    """
    root = Path(args.dir)
    errors = []
    if args.manifest is None:
        report = {
            "mode": "variants",
            "dir": str(root),
            "variants": VARIANTS,
            "results": [],
            "counts": status_counts([]),
            "errors": ["--manifest is required to define planned variant identities"],
            "passed": False,
        }
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    plan = {}
    referenced = set()
    for item in read_json(args.manifest).get("items", []):
        if item.get("kind") != "variant":
            continue
        if item.get("record") is not None:
            referenced.add(Path(item["record"]).resolve())
        if getattr(args, "tier", None) == "A" and item.get("tier") == "B":
            continue
        variant = item.get("variant")
        if variant not in VARIANTS or any(name not in item for name in VARIANT_IDENTITY):
            errors.append(f"malformed variant manifest item: {item}")
            continue
        key = json.dumps({name: item[name] for name in VARIANT_IDENTITY}, sort_keys=True)
        row = plan.setdefault(key, {})
        if variant in row:
            errors.append(f"duplicate planned {variant} item for {key}")
            continue
        row[variant] = item
    for variant in VARIANTS:
        for path in sorted((root / variant).rglob("run.json")):
            if path.resolve() not in referenced:
                errors.append(f"{path}: record not referenced by any planned variant item")
    if not plan:
        errors.append("manifest schedules no variant items")
    results = []
    for key in sorted(plan):
        items = plan[key]
        identity = json.loads(key)
        result = {"key": identity, "records": {}, "unsuccessful": {}}
        problems = []
        planned_keys = {}
        for variant in VARIANTS:
            item = items.get(variant)
            if item is None:
                result["unsuccessful"][variant] = "unplanned"
                problems.append(f"{variant} not scheduled")
                continue
            planned_keys[variant] = planned_variant_hash_keys(item)
            if planned_keys[variant] is None:
                problems.append(f"{variant} has no derivable planned hash keys")
        distinct = {json.dumps(value) for value in planned_keys.values() if value is not None}
        if len(distinct) > 1:
            problems.append(f"planned hash keys differ across variants: {planned_keys}")
        required = json.loads(distinct.pop()) if len(distinct) == 1 else None
        shared = {}
        for variant, label in VARIANTS.items():
            item = items.get(variant)
            if item is None:
                continue
            if item.get("outcome") != "success":
                result["unsuccessful"][variant] = item.get("outcome", "missing_outcome")
                continue
            path = Path(item.get("record") or "")
            if not item.get("record") or not path.is_file():
                result["unsuccessful"][variant] = "record_absent"
                continue
            if (root / variant).resolve() not in path.resolve().parents:
                problems.append(f"{variant} record {path} is outside {root / variant}")
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != item.get("record_sha256"):
                problems.append(f"{variant} record digest does not match manifest")
                continue
            record = read_json(path)
            init, optimizer, library_defaults = VARIANT_METADATA[variant]
            observed = {name: record.get(name) for name in VARIANT_IDENTITY}
            if observed != identity:
                problems.append(f"{variant} record identity {observed} is not planned {identity}")
            if record.get("config") != label:
                problems.append(f"{variant} config {record.get('config')!r} is not {label!r}")
            if (
                record.get("init"),
                record.get("optimizer"),
                record.get("library_defaults"),
            ) != (init, optimizer, library_defaults):
                problems.append(
                    f"{variant} metadata init={record.get('init')!r} "
                    f"optimizer={record.get('optimizer')!r} "
                    f"library_defaults={record.get('library_defaults')!r} is not "
                    f"{(init, optimizer, library_defaults)}"
                )
            hashes = record.get("hashes")
            if not isinstance(hashes, dict):
                problems.append(f"{variant} record has no hashes")
                hashes = {}
            if required is not None and sorted(hashes) != required:
                problems.append(f"{variant} hash keys {sorted(hashes)} are not planned {required}")
            shared[variant] = {name: record.get(name) for name in VARIANT_KEY_FIELDS}
            result["records"][variant] = {"path": str(path), "hashes": hashes}
        if len({json.dumps(value, sort_keys=True) for value in shared.values()}) > 1:
            problems.append(f"variant settings differ: {shared}")
        records = result["records"]
        if required is not None:
            result["hash_keys"] = required
        if problems:
            result["status"] = "mismatched"
            result["reason"] = problems
        elif missing := [name for name in VARIANTS if name not in records]:
            result["status"] = "unmeasured"
            result["reason"] = {name: result["unsuccessful"][name] for name in missing}
        else:
            reference = records["explicit"]["hashes"]
            differing = [
                f"{variant} differs on {name}"
                for variant, value in records.items()
                for name in required or ()
                if value["hashes"][name] != reference[name]
            ]
            result["status"] = "mismatched" if differing else "matched"
            if differing:
                result["reason"] = differing
        results.append(result)
    counts = status_counts(results)
    report = {
        "mode": "variants",
        "dir": str(root),
        "variants": VARIANTS,
        "results": results,
        "counts": counts,
        "errors": errors,
        "passed": not errors
        and counts["mismatched"] == 0
        and counts["unmeasured"] == 0
        and counts["matched"] == len(results) > 0,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def compare_hashes(args):
    if args.expected is not None:
        compare_expected(args)
        return
    baseline, candidate = hash_records(args.baseline), hash_records(args.candidate)
    results = []
    for key in sorted(baseline.keys() | candidate.keys()):
        left, right = baseline.get(key, []), candidate.get(key, [])
        probe = json.loads(key)
        stable = bool(left) and all(value == left[0] for value in left)
        equal = stable and bool(right) and all(value == left[0] for value in right)
        classic_unstable = (
            len(probe) == 2
            and isinstance(probe[1], dict)
            and probe[1].get("optimizer") == "compatibility"
            and len(left) >= 2
            and not stable
            and bool(right)
        )
        results.append(
            {
                "identity": probe,
                "baseline": left,
                "candidate": right,
                "baseline_stable": stable,
                "equal": equal,
                "status": "not_comparable"
                if classic_unstable
                else "matched"
                if equal
                else "unmatched",
            }
        )
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "results": results,
        "passed": any(row["equal"] for row in results)
        and all(row["status"] != "unmatched" for row in results),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def add_settings_arguments(command):
    command.add_argument("--negative-selection-range", type=int, default=200000)
    command.add_argument("--negative-sample-scale", type=float, default=None)
    command.add_argument("--exclude-graph-neighbors", choices=("true", "false"), default="false")
    command.add_argument("--negative-sample-scale-adaptation-samples", type=int, default=128)


def evidence_errors(args):
    errors = datasets.verify(args.data)
    for intent_path in Path(args.artifacts).glob("*/children/*/intent.json"):
        if not (intent_path.parent / "child.json").exists():
            errors.append(f"unresolved interrupted attempt: {intent_path.parent}")
    try:
        manifest = read_json(args.data / "manifest.json")
        for name in FIXTURES:
            if name not in manifest["fixtures"]:
                errors.append(f"missing fixture: {name}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"fixture manifest: {exc}")
    try:
        from comparison.umap_reference import require_reference

        require_reference()
    except Exception as exc:
        errors.append(f"environment: {exc}")
    try:
        require_extension(args.artifacts)
    except Exception as exc:
        errors.append(f"extension: {exc}")
    available = {}
    scores = {}
    decision_list = []
    try:
        decision_list = decisions(args.reference)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"decisions: {exc}")
    covered = {
        identity(item["scope"])
        for item in decision_list
        if "role" in item["scope"] and item["option"] in ("drop", "omit_seed")
    }
    try:
        available = runs(args.artifacts, args.data)
        for path, record in available.values():
            child_path = path.parents[2]
            child_data = read_json(child_path / "child.json")
            if (
                not (child_path / "stdout.txt").is_file()
                or not (child_path / "stderr.txt").is_file()
            ):
                errors.append(f"missing child output: {child_path}")
            if not all(key in child_data for key in ("before", "after", "command", "seconds")):
                errors.append(f"incomplete child evidence: {child_path}")
            fixture = record["fixture"]
            if (
                record["role"] == "upstream"
                and fixture == "mix_1m"
                and record["seed"] == 11
                and child_data["seconds"] > 17280
                and not any(
                    identity(item["scope"]) == identity(record)
                    and item["option"] == "raise_timeout"
                    for item in decision_list
                    if "role" in item["scope"]
                )
            ):
                errors.append(f"1M pilot exceeded budget without decision: {child_path}")
        expected = {
            identity(row) for role in ("upstream", "cuscan") for row in selected_matrix(role)
        }
        for extra in set(available) - expected:
            errors.append(f"unexpected run: {extra}")
        for role in ("upstream", "cuscan"):
            for row in selected_matrix(role):
                if identity(row) not in available and identity(row) not in covered:
                    errors.append(f"missing run: {identity(row)}")
        scores = scored_runs(args.artifacts, available)
        for key in available:
            if key not in scores:
                errors.append(f"missing score: {key}")
    except (OSError, ValueError, KeyError, RuntimeError, ImportError) as exc:
        errors.append(f"run records: {exc}")
    for fixture in FIXTURES:
        try:
            errors.extend(quality.verify_cache(args.data, fixture))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"evaluation cache {fixture}: {exc}")
    try:
        repeats = determinism_evidence(args.artifacts, available)
        for fixture in ("blobs_1k", "digits", "mix_50k"):
            for config in CONFIGS:
                for threads in (15, 1) if fixture == "digits" else (15,):
                    original = ("upstream", None, fixture, config, 11, 2, False)
                    if (fixture, config, threads) not in repeats and original not in covered:
                        errors.append(f"missing determinism: {(fixture, config, threads)}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"determinism: {exc}")
    try:
        stages_data = stage_evidence(args.artifacts)
        for fixture in ("mix_50k", "mix_200k", "mix_1m"):
            for backend in ("cpu", "cuda"):
                if f"{fixture}-{backend}" not in stages_data:
                    errors.append(f"missing stage evidence: {fixture}-{backend}")
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        errors.append(f"stage evidence: {exc}")
    cuscan_groups = {}
    for key, (_, record) in available.items():
        if key[0] == "cuscan":
            group = (key[1], key[2], key[3], key[5], key[6])
            cuscan_groups.setdefault(group, []).append(record)
    for group, records in cuscan_groups.items():
        first = next((r for r in records if r["seed"] == 11), None)
        repeats = [
            r["in_process_reproducibility"] for r in records if "in_process_reproducibility" in r
        ]
        if first is None or not any(
            r["seed"] == 11
            and r["first_hash"] == first["hashes"]["fit"]
            and r["identical"] == (r["first_hash"] == r["repeat_hash"])
            for r in repeats
        ):
            errors.append(f"missing in-process seed-11 reproducibility: {group}")
    return errors


def check(args):
    """Compare complete current evidence exactly; retired-fixture baselines cannot match."""
    run = new_run(args.artifacts, "check")
    errors = evidence_errors(args)
    try:
        expected_upstream, expected_cuscan = summary_data(args)
        upstream_path = args.reference / "upstream-0.6.0.json"
        paths = sorted(args.reference.glob("cuscan-*.json"))
        if len(paths) != 1:
            errors.append(f"expected exactly one cuscan reference, found {len(paths)}")
        upstream = read_json(upstream_path)
        for field in (
            "schema",
            "runs",
            "aggregates",
            "configurations",
            "determinism",
            "acceptance_rule",
            "decisions",
        ):
            if upstream.get(field) != expected_upstream[field]:
                errors.append(f"upstream reference mismatch: {field}")
        for field in (
            "wheel_sha256",
            "wheel_url",
            "lock_sha256",
            "fixture_hashes",
            "source_hashes",
        ):
            if upstream.get("provenance", {}).get(field) != expected_upstream["provenance"][field]:
                errors.append(f"upstream provenance mismatch: {field}")
        for field in ("python", "platform", "cpu_count", "threads", "freeze"):
            if field not in upstream.get("provenance", {}):
                errors.append(f"upstream reference missing provenance: {field}")
        if paths:
            cuscan = read_json(paths[0])
            for field in (
                "schema",
                "backends",
                "stage_attribution",
                "cpu_cuda_agreement",
                "diagnostic_acceptance",
                "extension",
                "decisions",
            ):
                if cuscan.get(field) != expected_cuscan[field]:
                    errors.append(f"cuscan reference mismatch: {field}")
            if not cuscan.get("build_provenance"):
                errors.append("cuscan reference missing build provenance")
            for field in ("rustc", "git_revision", "git_status", "fixture_hashes", "source_hashes"):
                if field not in cuscan.get("build_provenance", {}):
                    errors.append(f"cuscan reference missing build provenance: {field}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, ImportError) as exc:
        errors.append(f"reference validation: {exc}")
    for error in errors:
        print(error, file=sys.stderr)
    write_new(run / "result.json", {"errors": errors, "passed": not errors})
    if errors:
        raise SystemExit(1)
    print(run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, function in (
        ("generate", generate),
        ("upstream", lambda a: matrix(a, "upstream")),
        ("determinism", determinism),
        ("cuscan", lambda a: matrix(a, "cuscan")),
        ("stages", stages),
        ("score", score),
        ("summarize", summarize),
        ("check", check),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--data", type=Path, default=Path("data/umap06"))
        command.add_argument("--artifacts", type=Path, default=Path("artifacts/umap06_p0"))
        command.add_argument("--reference", type=Path, default=Path("comparison/umap06/reference"))
        if name in ("generate", "upstream", "cuscan", "score", "determinism"):
            command.add_argument("--fixtures", nargs="+", choices=FIXTURES)
        if name == "determinism":
            command.add_argument("--configs", nargs="+", choices=CONFIGS)
        if name == "score":
            command.add_argument("--fit-only", action="store_true")
        if name in ("upstream", "cuscan"):
            command.add_argument(
                "--configs",
                nargs="+",
                choices=(*CONFIGS, *VARIANT_CONFIGS) if name == "cuscan" else CONFIGS,
            )
            command.add_argument("--seeds", nargs="+", type=int)
            command.add_argument("--backend", choices=("cpu", "cuda"))
        if name in ("upstream", "cuscan"):
            command.add_argument("--record-extension", action="store_true")
        if name == "cuscan":
            command.add_argument("--phase3", action="store_true")
            command.add_argument("--phase4", action="store_true")
            command.add_argument("--phase5", action="store_true")
            command.add_argument("--phase5-variants", action="store_true")
            command.add_argument("--phase7", action="store_true")
            command.add_argument("--phase7-tier", type=int, choices=range(1, 6))
            command.add_argument("--components", nargs="+", type=int)
            command.add_argument("--scope", choices=("halved-fit",))
            command.add_argument("--skip-transform", action="store_true")
            command.add_argument("--no-child-timeout", action="store_true")
            add_settings_arguments(command)
        if name == "summarize":
            command.add_argument("--freeze", action="store_true")
        command.set_defaults(function=function)
    command = subparsers.add_parser("accept-phase2")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--upstream-reference", type=Path, required=True)
    command.add_argument("--upstream-sha256", required=True)
    command.add_argument(
        "--unscored-fixtures",
        nargs="+",
        choices=("mix_1m",),
        help="fixtures whose fits must exist but whose quality is not compared; recorded",
    )
    command.add_argument(
        "--exemption-reason",
        help="required with --unscored-fixtures or --fit-only; recorded verbatim in the receipt",
    )
    command.add_argument(
        "--fit-only",
        action="store_true",
        help="compare fit metrics only; transform metrics stay scored but do not gate; recorded",
    )
    command.set_defaults(function=accept_phase2)
    command = subparsers.add_parser("accept-phase3")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--upstream-reference", type=Path, required=True)
    command.add_argument("--upstream-sha256", required=True)
    command.add_argument("--unscored-fixtures", nargs="+", choices=("mix_1m",))
    command.add_argument("--exemption-reason")
    command.set_defaults(function=accept_phase3, fit_only=False)
    command = subparsers.add_parser("accept-phase4")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--upstream-reference", type=Path, required=True)
    command.add_argument("--upstream-sha256", required=True)
    command.add_argument(
        "--evidence-fixtures",
        nargs="+",
        choices=list(FIXTURES),
        help="evaluate only these fixtures; any omitted fixture leaves the receipt incomplete",
    )
    add_settings_arguments(command)
    command.add_argument("--scope", choices=("halved-fit",))
    command.set_defaults(function=accept_phase4, fit_only=False)
    command = subparsers.add_parser("accept-phase5")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--upstream-reference", type=Path, required=True)
    command.add_argument("--upstream-sha256", required=True)
    command.add_argument("--evidence-fixtures", nargs="+", choices=list(FIXTURES))
    command.add_argument("--scope", choices=("halved-fit",))
    command.add_argument("--full-seeds", action="store_true")
    add_settings_arguments(command)
    command.set_defaults(function=accept_phase5, fit_only=False)
    command = subparsers.add_parser("accept-phase7")
    command.add_argument("--data", type=Path, required=True)
    command.add_argument("--artifacts", type=Path, required=True)
    command.add_argument("--upstream-reference", type=Path, required=True)
    command.add_argument("--upstream-sha256", required=True)
    command.add_argument("--query-reference", type=Path, default=None)
    command.add_argument("--query-reference-sha256", default=None)
    command.add_argument("--evidence-fixtures", nargs="+", choices=list(FIXTURES))
    seed_scope = command.add_mutually_exclusive_group()
    seed_scope.add_argument("--full-seeds", action="store_true")
    seed_scope.add_argument("--seed-11-no-mix-1m", action="store_true")
    add_settings_arguments(command)
    command.set_defaults(function=accept_phase7, fit_only=False)
    command = subparsers.add_parser("compare-hashes")
    command.add_argument("--baseline", type=Path, required=True)
    command.add_argument("--candidate", type=Path, required=True)
    command.add_argument("--expected", type=Path, default=None)
    command.set_defaults(function=compare_hashes)
    command = subparsers.add_parser("compare-variants")
    command.add_argument("--dir", type=Path, required=True)
    command.add_argument("--manifest", type=Path, default=None)
    command.add_argument("--tier", choices=("A", "B"))
    command.set_defaults(function=compare_variants)
    args = parser.parse_args()
    if args.command in ("compare-hashes", "compare-variants"):
        args.function(args)
        return
    if os.name != "posix":
        parser.error("UMAP 0.6 campaign commands require POSIX file locking")
    import fcntl

    args.artifacts.mkdir(parents=True, exist_ok=True)
    with (args.artifacts / ".campaign.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another UMAP 0.6 campaign command is running") from exc
        args.function(args)


if __name__ == "__main__":
    main()
