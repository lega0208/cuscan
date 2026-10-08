#!/usr/bin/env -S uv run --script
"""Stage-1 benchmark suite: umap-learn 0.5.12 (parallel) vs fast-umap CUDA standard UMAP.

Run from the repository root with ``uv run --no-sync python scripts/compare_umap_spans.py``.
Build the companion first (build time is excluded from the runtime budget):
``cargo build --release --no-default-features --features cuda,tracing
--example profile_standard_cuda``.

Primary workload: identical clustered float32 5000x32 input, 20 epochs, Euclidean,
15 neighbors, 2 components, random init, explicitly forced Exact and Approximate
search. Each implementation/mode runs one warmup plus N timed fits. Fit durations
include neighbor generation and host embedding readback and exclude input generation,
profiler overhead and quality calculation. Python is unseeded and uses all threads.
Small CUDA regressions: 512x16/30 and 1024x32/100 in both search modes.

Use ``--binary PATH`` to select the benchmark executable. To capture the pre-change
Rust serial implementation as a tracking reference, run
Use the suite command with ``--binary /path/to/pre-change/profile_standard_cuda``
and ``--output artifacts/serial-reference.json``
and pass the complete suite JSON later with
``--quality-reference PATH``. It is distinct from ``--baseline``, which only
provides the existing small/speed comparison gates. Tracking requires at least
two finite CUDA trustworthiness samples per mode and does not replace Q-02.

One global deadline (default 600 seconds) covers warmups, repetitions,
quality/recall diagnostics and
profilers. On expiry the run stops and is reported incomplete (no pass); the
deadline is never adjusted automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_standard_cuda"
PRIMARY = (5000, 32, 20)
SMALL = ((512, 16, 30), (1024, 32, 100))
MODES = ("exact", "approximate")
NEIGHBORS = 15
FIT_WIDTH = NEIGHBORS - 1
TRUST_GAP = 0.02
RECALL_LIMIT = 0.90
SMALL_RATIO_LIMIT = 1.10
SPEED_RATIO_LIMIT = 0.90


class DeadlineExceeded(TimeoutError):
    """The global benchmark deadline expired."""


class Deadline:
    def __init__(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("deadline must be positive")
        self.ends_at = time.monotonic() + seconds

    def remaining(self) -> float:
        value = self.ends_at - time.monotonic()
        if value <= 0:
            raise DeadlineExceeded("global runtime deadline expired")
        return value


def run_with_deadline(
    command: list[str], deadline: Deadline, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=ROOT,
            check=check,
            capture_output=True,
            text=True,
            timeout=deadline.remaining(),
        )
    except subprocess.TimeoutExpired as error:
        raise DeadlineExceeded("global runtime deadline expired during worker execution") from error


def last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "fit_seconds" in parsed:
            return parsed
    raise ValueError("worker did not emit its result JSON")


def make_clustered_input(rows: int, features: int, destination: Path) -> None:
    import numpy as np

    rng = np.random.default_rng(42)
    centers = rng.normal(0.0, 5.0, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % len(centers)
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, 0.35, size=(rows, features)).astype(np.float32)
    matrix.astype("<f4", copy=False).tofile(destination)


def _python_worker(
    input_path: Path, rows: int, features: int, epochs: int, mode: str, runs: int, out: Path
) -> dict[str, Any]:
    import numba
    import numpy as np
    import umap
    from sklearn.neighbors import NearestNeighbors

    if umap.__version__ != "0.5.12":
        raise ValueError("This harness requires umap-learn 0.5.12")
    matrix = np.fromfile(input_path, dtype="<f4").reshape(rows, features)

    def fit_once() -> Any:
        precomputed = None
        if mode == "exact":
            search = NearestNeighbors(n_neighbors=NEIGHBORS, n_jobs=-1).fit(matrix)
            distances, indices = search.kneighbors(matrix)
            precomputed = (indices, distances, None)
        estimator = umap.UMAP(
            n_neighbors=NEIGHBORS,
            n_components=2,
            n_epochs=epochs,
            init="random",
            random_state=None,
            metric="euclidean",
            force_approximation_algorithm=mode == "approximate",
            precomputed_knn=precomputed if precomputed is not None else (None, None, None),
            n_jobs=-1,
            verbose=False,
        )
        estimator.fit(matrix)
        return estimator

    fit_once()  # warmup (numba JIT)
    fit_seconds = []
    for run in range(runs):
        start = time.perf_counter()
        estimator = fit_once()
        fit_seconds.append(time.perf_counter() - start)
        np.save(out / f"embedding_{run}.npy", np.asarray(estimator.embedding_, dtype=np.float32))
        np.save(out / f"knn_{run}.npy", np.asarray(estimator._knn_indices))
    return {
        "fit_seconds": fit_seconds,
        "umap_learn_version": umap.__version__,
        "numba_threads": numba.get_num_threads(),
        "cpu_count": os.cpu_count(),
        "config": {
            "random_state": None,
            "n_jobs": -1,
            "force_approximation_algorithm": mode == "approximate",
            "exact_precomputed_knn_inside_timed_fit": mode == "exact",
        },
    }


def nonself(indices: Any, width: int) -> Any:
    import numpy as np

    rows = indices.shape[0]
    result = np.empty((rows, width), dtype=np.int64)
    for row in range(rows):
        ids = [int(i) for i in indices[row] if int(i) != row and int(i) >= 0]
        if len(ids) < width:
            raise ValueError(f"row {row} has fewer than {width} non-self neighbors")
        result[row] = ids[:width]
    return result


def tie_aware_recall(matrix: Any, actual: Any, exact_kth: Any) -> float:
    """Fraction of actual non-self ids no farther than the exact k-th non-self distance."""
    import numpy as np

    rows = np.arange(actual.shape[0])[:, None]
    distances = np.linalg.norm(matrix[actual] - matrix[rows], axis=2)
    hits = 0
    for row in range(actual.shape[0]):
        _, unique_positions = np.unique(actual[row], return_index=True)
        hits += int(np.sum(distances[row, unique_positions] <= exact_kth[row] * (1 + 1e-6) + 1e-12))
    return hits / (actual.shape[0] * actual.shape[1])


def median(values: list[float]) -> float:
    return statistics.median(values)


def load_quality_reference(path: Path) -> dict[str, Any]:
    reference = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(reference, dict) or reference.get("complete") is not True:
        raise ValueError("quality reference must be a complete suite JSON")
    configuration = reference.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("quality reference is missing configuration metadata")
    primary = configuration.get("primary")
    expected_primary = {"rows": PRIMARY[0], "features": PRIMARY[1], "epochs": PRIMARY[2]}
    if primary != expected_primary:
        raise ValueError(f"quality reference primary workload {primary!r} != {expected_primary!r}")
    expected_metadata = {
        "neighbors": NEIGHBORS,
        "components": 2,
        "init": "random",
        "metric": "euclidean",
        "input": "seeded clustered float32 (8 centers), little-endian row-major",
    }
    for key, expected in expected_metadata.items():
        if configuration.get(key) != expected:
            raise ValueError(
                f"quality reference {key} metadata {configuration.get(key)!r} != {expected!r}"
            )
    quality = reference.get("quality")
    if not isinstance(quality, dict):
        raise ValueError("quality reference is missing quality results")
    for mode in MODES:
        mode_quality = quality.get(mode)
        if not isinstance(mode_quality, dict):
            raise ValueError(f"quality reference is missing {mode} quality results")
        values = mode_quality.get("cuda_trustworthiness")
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(
                f"quality reference {mode} requires at least two CUDA trustworthiness samples"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                f"quality reference {mode} CUDA trustworthiness samples must be finite"
            )
    return reference


def nsys_kernels(binary_args: list[str], tag: str, temp: Path, deadline: Deadline) -> dict:
    report = temp / f"nsys-{tag}"
    run_with_deadline(
        [
            "nsys",
            "profile",
            "--trace=cuda,osrt",
            "--sample=none",
            "--force-overwrite=true",
            f"--output={report}",
            *binary_args,
        ],
        deadline,
    )
    stats = run_with_deadline(
        [
            "nsys",
            "stats",
            "--report",
            "cuda_gpu_kern_sum,cuda_api_sum",
            "--format",
            "csv",
            "--force-export=true",
            str(report) + ".nsys-rep",
        ],
        deadline,
    )
    return {"report": str(report) + ".nsys-rep", "summary_csv": stats.stdout[-6000:]}


def ncu_kernel(binary_args: list[str], regex: str, temp: Path, deadline: Deadline) -> dict:
    report = temp / "ncu"
    profiled = run_with_deadline(
        [
            "ncu",
            "--target-processes",
            "all",
            "--kernel-name-base",
            "demangled",
            "--kernel-name",
            f"regex:{regex}",
            # The selected sixteen-batch fit has no fired edges in epoch zero.
            "--launch-skip",
            "16",
            "--launch-count",
            "1",
            "--set",
            "basic",
            "--force-overwrite",
            "--export",
            str(report),
            *binary_args,
        ],
        deadline,
        check=False,
    )
    if "ERR_NVGPUCTRPERM" in profiled.stdout + profiled.stderr:
        # Driver-level counter access is a host setting only the user can change.
        return {"status": "blocked", "reason": "ERR_NVGPUCTRPERM: GPU performance counters denied"}
    if profiled.returncode != 0:
        raise subprocess.CalledProcessError(
            profiled.returncode, profiled.args, profiled.stdout, profiled.stderr
        )
    details = run_with_deadline(
        ["ncu", "--import", str(report) + ".ncu-rep", "--csv", "--page", "details"], deadline
    )
    if not details.stdout.strip():
        raise ValueError(f"ncu kernel filter {regex!r} matched no kernel; refine it")
    return {
        "report": str(report) + ".ncu-rep",
        "matching_launches_skipped": 16,
        "details_csv": details.stdout,
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["stage1"], default="stage1")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--deadline-seconds", type=float, default=600.0)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, help="write the full suite JSON")
    parser.add_argument("--baseline", type=Path, help="pre-change suite JSON for H-03 ratios")
    parser.add_argument(
        "--quality-reference",
        type=Path,
        help="complete pre-change serial suite JSON for per-mode CUDA quality tracking",
    )
    parser.add_argument("--profile-nsys", action="store_true")
    parser.add_argument("--profile-ncu", action="store_true")
    parser.add_argument("--ncu-kernel", default=".*layout_vertex.*kernel.*")
    parser.add_argument("--profile-dir", type=Path, help="keep profiler reports here")
    parser.add_argument("--_python-worker", nargs=7, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._python_worker:
        input_path, rows, features, epochs, mode, runs, out = args._python_worker
        result = _python_worker(
            Path(input_path), int(rows), int(features), int(epochs), mode, int(runs), Path(out)
        )
        print(json.dumps(result))
        return 0
    if args.runs < 1 or args.warmups < 1:
        parser.error("runs and warmups must be positive")
    quality_reference = None
    if args.quality_reference:
        try:
            quality_reference = load_quality_reference(args.quality_reference)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            parser.error(f"invalid --quality-reference: {error}")
    binary = args.binary.resolve()
    if not binary.is_file():
        print(
            f"CUDA profiler executable not found: {binary}\nBuild it first:\n"
            "  cargo build --release --no-default-features --features cuda,tracing "
            "--example profile_standard_cuda",
            file=sys.stderr,
        )
        return 2

    import numpy as np
    from sklearn.manifold import trustworthiness
    from sklearn.neighbors import NearestNeighbors

    deadline = Deadline(args.deadline_seconds)
    start = time.monotonic()
    output: dict[str, Any] = {
        "configuration": {
            "primary": {"rows": PRIMARY[0], "features": PRIMARY[1], "epochs": PRIMARY[2]},
            "small": [{"rows": r, "features": f, "epochs": e} for r, f, e in SMALL],
            "neighbors": NEIGHBORS,
            "components": 2,
            "init": "random",
            "metric": "euclidean",
            "warmups": args.warmups,
            "runs": args.runs,
            "deadline_seconds": args.deadline_seconds,
            "input": "seeded clustered float32 (8 centers), little-endian row-major",
            "cuda_seeds": "42+run (warmups 1000+i)",
        },
        "complete": False,
    }

    def emit() -> None:
        output["elapsed_seconds"] = time.monotonic() - start
        rendered = json.dumps(output, indent=2, sort_keys=True)
        if args.output:
            args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
        print(rendered)

    with tempfile.TemporaryDirectory(prefix="stage1-") as temporary:
        temp = Path(temporary)
        profile_dir = args.profile_dir.resolve() if args.profile_dir else temp
        profile_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows, features, epochs = PRIMARY
            primary_input = temp / "primary.f32le"
            make_clustered_input(rows, features, primary_input)
            matrix = np.fromfile(primary_input, dtype="<f4").reshape(rows, features)
            primary: dict[str, Any] = {}
            for mode in MODES:
                py_dir = temp / f"py-{mode}"
                py_dir.mkdir()
                python = last_json(
                    run_with_deadline(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--_python-worker",
                            str(primary_input),
                            str(rows),
                            str(features),
                            str(epochs),
                            mode,
                            str(args.runs),
                            str(py_dir),
                        ],
                        deadline,
                    ).stdout
                )
                cuda_dir = temp / f"cuda-{mode}"
                cuda = last_json(
                    run_with_deadline(
                        [
                            str(binary),
                            str(primary_input),
                            str(rows),
                            str(features),
                            str(epochs),
                            str(cuda_dir),
                            "--search",
                            mode,
                            "--warmups",
                            str(args.warmups),
                            "--runs",
                            str(args.runs),
                        ],
                        deadline,
                    ).stdout
                )
                if cuda["fit_graph_width"] != FIT_WIDTH:
                    raise ValueError(f"CUDA fit graph width {cuda['fit_graph_width']} != 14")
                primary[mode] = {"python": python, "cuda": cuda, "dirs": (py_dir, cuda_dir)}
            output["primary"] = {
                mode: {
                    "python": {k: v for k, v in data["python"].items()},
                    "cuda": data["cuda"],
                    "python_median_seconds": median(data["python"]["fit_seconds"]),
                    "cuda_median_seconds": median(data["cuda"]["fit_seconds"]),
                }
                for mode, data in primary.items()
            }

            small: dict[str, Any] = {}
            for r, f, e in SMALL:
                small_input = temp / f"small-{r}.f32le"
                make_clustered_input(r, f, small_input)
                for mode in MODES:
                    result = last_json(
                        run_with_deadline(
                            [
                                str(binary),
                                str(small_input),
                                str(r),
                                str(f),
                                str(e),
                                str(temp / f"small-{r}-{mode}"),
                                "--search",
                                mode,
                                "--warmups",
                                str(args.warmups),
                                "--runs",
                                str(args.runs),
                            ],
                            deadline,
                        ).stdout
                    )
                    small[f"{r}x{f}/{e}/{mode}"] = {
                        "fit_seconds": result["fit_seconds"],
                        "median_seconds": median(result["fit_seconds"]),
                    }
            output["small_cuda"] = small

            deadline.remaining()
            exact = NearestNeighbors(n_neighbors=NEIGHBORS).fit(matrix)
            exact_dist, exact_idx = exact.kneighbors(matrix)
            exact_kth = np.array(
                [
                    sorted(
                        d for d, i in zip(exact_dist[row], exact_idx[row], strict=True) if i != row
                    )[FIT_WIDTH - 1]
                    for row in range(rows)
                ]
            )
            quality: dict[str, Any] = {}
            for mode, data in primary.items():
                py_dir, cuda_dir = data["dirs"]
                py_trust, cuda_trust, py_recall, cuda_recall = [], [], [], []
                for run in range(args.runs):
                    deadline.remaining()
                    py_emb = np.load(py_dir / f"embedding_{run}.npy")
                    cuda_emb = np.fromfile(cuda_dir / f"embedding_{run}.f32le", "<f4").reshape(
                        rows, 2
                    )
                    py_trust.append(float(trustworthiness(matrix, py_emb, n_neighbors=NEIGHBORS)))
                    cuda_trust.append(
                        float(trustworthiness(matrix, cuda_emb, n_neighbors=NEIGHBORS))
                    )
                    if mode == "approximate":
                        py_graph = nonself(np.load(py_dir / f"knn_{run}.npy"), FIT_WIDTH)
                        cuda_graph = np.fromfile(cuda_dir / f"graph_{run}.i32le", "<i4").reshape(
                            rows, FIT_WIDTH
                        )
                        py_recall.append(tie_aware_recall(matrix, py_graph, exact_kth))
                        cuda_recall.append(tie_aware_recall(matrix, cuda_graph, exact_kth))
                python_mean = statistics.fmean(py_trust)
                quality[mode] = {
                    "python_trustworthiness": py_trust,
                    "python_trustworthiness_mean": python_mean,
                    "cuda_trustworthiness": cuda_trust,
                    "Q-02_each_cuda_within_0.02_of_python_mean": all(
                        abs(t - python_mean) <= TRUST_GAP for t in cuda_trust
                    ),
                }
                if mode == "approximate":
                    quality[mode].update(
                        {
                            "python_fit_recall_at_14": py_recall,
                            "cuda_fit_recall_at_14": cuda_recall,
                            "Q-03_every_run_recall_ge_0.90": all(
                                r >= RECALL_LIMIT for r in py_recall + cuda_recall
                            ),
                        }
                    )
            output["quality"] = quality

            if quality_reference is not None:
                reference_quality = quality_reference["quality"]
                tracking_gates: dict[str, bool] = {}
                for mode in MODES:
                    reference_samples = reference_quality[mode]["cuda_trustworthiness"]
                    candidate_samples = quality[mode]["cuda_trustworthiness"]
                    if any(not math.isfinite(value) for value in candidate_samples):
                        raise ValueError(
                            f"candidate {mode} CUDA trustworthiness samples must be finite"
                        )
                    reference_mean = statistics.fmean(reference_samples)
                    reference_std = statistics.stdev(reference_samples)
                    threshold = reference_mean - 0.5 * reference_std
                    candidate_mean = statistics.fmean(candidate_samples)
                    passed = candidate_mean >= threshold
                    quality[mode]["tracking_reference"] = {
                        "sample_count": len(reference_samples),
                        "mean": reference_mean,
                        "sample_std": reference_std,
                        "threshold_mean_minus_0.5_sample_std": threshold,
                        "candidate_cuda_mean": candidate_mean,
                        "pass": passed,
                    }
                    tracking_gates[f"Rust-reference_{mode}_mean_ge_reference_minus_0.5std"] = passed
                output["quality_tracking"] = {
                    "policy": "warning_only",
                    "checks": tracking_gates,
                    "warnings": [name for name, passed in tracking_gates.items() if not passed],
                }

            gates: dict[str, Any] = {
                f"H-04_{mode}_cuda_median_le_0.90x_python": output["primary"][mode][
                    "cuda_median_seconds"
                ]
                <= SPEED_RATIO_LIMIT * output["primary"][mode]["python_median_seconds"]
                for mode in MODES
            }
            for mode in MODES:
                gates[f"H-04_{mode}_ratio_cuda_over_python"] = (
                    output["primary"][mode]["cuda_median_seconds"]
                    / output["primary"][mode]["python_median_seconds"]
                )
            if args.baseline:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
                ratios = {
                    key: value["median_seconds"] / baseline["small_cuda"][key]["median_seconds"]
                    for key, value in small.items()
                }
                gates["H-03_small_ratio_vs_baseline"] = ratios
                gates["H-03_pass"] = all(r <= SMALL_RATIO_LIMIT for r in ratios.values())
                gates["primary_cuda_ratio_vs_baseline"] = {
                    mode: output["primary"][mode]["cuda_median_seconds"]
                    / baseline["primary"][mode]["cuda_median_seconds"]
                    for mode in MODES
                }
            output["gates"] = gates

            profiles: dict[str, Any] = {}
            primary_args = [str(binary), str(primary_input), *map(str, PRIMARY)]
            if args.profile_nsys:
                for mode in MODES:
                    profiles[f"nsys_{mode}"] = nsys_kernels(
                        [
                            *primary_args,
                            str(temp / f"nsys-{mode}-out"),
                            "--search",
                            mode,
                            "--warmups",
                            "1",
                            "--runs",
                            "1",
                        ],
                        mode,
                        profile_dir,
                        deadline,
                    )
            if args.profile_ncu:
                profiles["ncu_approximate"] = ncu_kernel(
                    [
                        *primary_args,
                        str(temp / "ncu-out"),
                        "--search",
                        "approximate",
                        "--warmups",
                        "1",
                        "--runs",
                        "1",
                    ],
                    args.ncu_kernel,
                    profile_dir,
                    deadline,
                )
            output["profiles"] = profiles
            deadline.remaining()
            output["complete"] = True
        except DeadlineExceeded as error:
            output["incomplete_reason"] = (
                f"{error}; results are incomplete and not a pass. Change parameters or extend "
                "the timeout (recommended: extend --deadline-seconds) and rerun."
            )
            emit()
            return 1
        except subprocess.CalledProcessError as error:
            output["incomplete_reason"] = f"worker failed: {error}\n{error.stderr[-4000:]}"
            emit()
            return 1
    emit()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
