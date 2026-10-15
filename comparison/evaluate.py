"""Run reproducible UMAP comparisons and write JSON plus standalone HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

if __package__:
    from .metrics import (
        EVALUATION_SEED,
        N_NEIGHBORS,
        N_REPRESENTATIVES,
        embedding_metrics,
        representative_indices,
        unique_rows,
    )
else:
    from metrics import (
        EVALUATION_SEED,
        N_NEIGHBORS,
        N_REPRESENTATIVES,
        embedding_metrics,
        representative_indices,
        unique_rows,
    )

METHODS = ("rust-raw", "rust-unique", "python-raw", "python-unique")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="row-major float32 input file")
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--features", required=True, type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "wgpu"), default="cpu")
    parser.add_argument("--seeds", type=_parse_seeds, default=[EVALUATION_SEED])
    parser.add_argument("--representatives", type=int, default=N_REPRESENTATIVES)
    parser.add_argument(
        "--output", required=True, type=Path, help="JSON output path; HTML uses same stem"
    )
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=list(METHODS),
        help=(f"e.g. --methods={','.join(METHODS)}"),
    )
    parser.add_argument("--n-epochs", type=int, help="default UMAP epoch count when omitted")
    parser.add_argument(
        "--min-trustworthiness",
        type=float,
        help="minimum acceptable trustworthiness at 15 neighbors",
    )
    parser.add_argument(
        "--min-neighbor-overlap",
        type=float,
        help="minimum acceptable neighbor overlap at 15 neighbors",
    )
    parser.add_argument(
        "--max-runtime-seconds", type=float, help="maximum allowed runtime in seconds"
    )
    parser.add_argument(
        "--max-runtime-ratio", type=float, help="maximum allowed runtime ratio relative to baseline"
    )
    parser.add_argument(
        "--runtime-protocol", action="store_true", help="one warmup then five unseeded matched runs"
    )
    parser.add_argument(
        "--child-timeout-seconds",
        type=float,
        default=300.0,
        help="maximum allowed time for child processes in seconds",
    )
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.rows <= 0 or args.features <= 0:
        raise ValueError("--rows and --features must be positive")
    if not args.seeds or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must contain non-negative integers")
    if args.n_epochs is not None and args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive")
    if args.child_timeout_seconds <= 0:
        raise ValueError("--child-timeout-seconds must be positive")
    if args.representatives <= 0:
        raise ValueError("--representatives must be positive")
    if args.representatives <= 2 * N_NEIGHBORS:
        raise ValueError(f"--representatives must exceed {2 * N_NEIGHBORS} for trustworthiness@15")
    thresholds = {
        "trustworthiness_at_15": args.min_trustworthiness,
        "neighbor_overlap_at_15": args.min_neighbor_overlap,
        "full_runtime_seconds": args.max_runtime_seconds,
    }
    if args.max_runtime_ratio is not None and args.max_runtime_ratio < 0:
        raise ValueError("--max-runtime-ratio must be non-negative")
    if any(value is not None and value < 0 for value in thresholds.values()):
        raise ValueError("gate thresholds must be non-negative")

    data = args.input.read_bytes()
    expected_bytes = args.rows * args.features * np.dtype(np.float32).itemsize
    if len(data) != expected_bytes:
        raise ValueError(f"input has {len(data)} bytes; expected {expected_bytes}")
    x = np.frombuffer(data, dtype="<f4").reshape(args.rows, args.features).copy()
    if not np.isfinite(x).all():
        raise ValueError("input contains non-finite values")
    distinct, first, _ = unique_rows(x)
    if distinct.shape[0] <= N_NEIGHBORS:
        raise ValueError(f"input must contain at least {N_NEIGHBORS + 1} distinct rows")
    if args.representatives > distinct.shape[0]:
        raise ValueError("--representatives exceeds the number of distinct input rows")
    reps = representative_indices(distinct.shape[0], count=args.representatives)
    selected_methods = list(args.methods)
    if args.device == "wgpu":
        raise ValueError("the standard Rust UMAP API supports cpu and cuda, not wgpu")

    run_results: dict[str, list[dict[str, Any]]] = {method: [] for method in selected_methods}
    embeddings: dict[str, list[dict[str, Any]]] = {method: [] for method in selected_methods}
    failures: list[dict[str, Any]] = []
    for seed in args.seeds:
        for method in selected_methods:
            is_unique = method.endswith("unique")
            try:
                worker_result = _fit_in_child(
                    x,
                    method=method,
                    device=args.device,
                    seed=int(seed),
                    n_epochs=args.n_epochs,
                    timeout=args.child_timeout_seconds,
                )
                y = worker_result["embedding"]
                expected_rows = x.shape[0]
                if y.shape != (expected_rows, 2) or not np.isfinite(y).all():
                    raise ValueError(f"{method} returned invalid embedding shape/values: {y.shape}")
                full_time = worker_result["fit_seconds"]
                evaluation_y = y[first[reps]]
                metric_values = embedding_metrics(distinct[reps], evaluation_y)
                result = {
                    "seed": int(seed),
                    "metrics": metric_values,
                    "full_runtime_seconds": full_time,
                    "stage_times": {
                        "fit_seconds": worker_result["fit_seconds"],
                        "subprocess_wall_seconds": worker_result["process_seconds"],
                    },
                    "embedding_rows": int(y.shape[0]),
                    "unique_strategy": "native unique fit" if is_unique else None,
                    "warnings": worker_result["warnings"],
                    "stderr": worker_result["stderr"],
                }
                run_results[method].append(result)
                embeddings[method].append({"seed": int(seed), "values": y})
            except Exception as error:
                evidence = getattr(error, "evidence", {"error": str(error)})
                failures.append({"method": method, "seed": int(seed), **evidence})
                _persist_failure(args.output, failures, run_results)
                raise

    metrics_summary: dict[str, dict[str, dict[str, float]]] = {}
    gates: dict[str, dict[str, Any]] = {}
    for method, runs in run_results.items():
        metrics_summary[method] = {}
        for metric in ("trustworthiness_at_15", "neighbor_overlap_at_15"):
            vals = np.asarray([r["metrics"][metric] for r in runs], dtype=np.float64)
            metrics_summary[method][metric] = {
                "mean": float(np.mean(vals)),
                "sample_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "run_count": len(vals),
            }
        method_gates: dict[str, Any] = {}
        for metric, threshold in thresholds.items():
            if threshold is None:
                method_gates[metric] = {
                    "status": "pending_approval_not_evaluated",
                    "threshold": None,
                }
                continue
            observed = (
                float(np.mean([r["full_runtime_seconds"] for r in runs]))
                if metric == "full_runtime_seconds"
                else metrics_summary[method][metric]["mean"]
            )
            passed = (
                observed <= threshold if metric == "full_runtime_seconds" else observed >= threshold
            )
            method_gates[metric] = {
                "status": "pass" if passed else "fail",
                "threshold": threshold,
                "observed_mean": observed,
            }
        gates[method] = method_gates
    runtime_protocol = None
    if args.runtime_protocol or args.max_runtime_ratio is not None:
        try:
            runtime_protocol = _run_runtime_protocol(
                x, selected_methods, args.device, args.n_epochs, args.child_timeout_seconds
            )
        except Exception as error:
            failure = {
                "phase": "runtime_protocol",
                **getattr(error, "evidence", {"error": str(error)}),
            }
            failures.append(failure)
            _persist_failure(args.output, failures, run_results)
            raise
    if args.max_runtime_ratio is None:
        runtime_gate = {"status": "pending_approval_not_evaluated", "threshold": None}
    elif runtime_protocol is None:
        runtime_gate = {"status": "not_evaluated", "threshold": args.max_runtime_ratio}
    else:
        ratios = [entry["median_ratio"] for entry in runtime_protocol.values()]
        maximum = max(ratios)
        runtime_gate = {
            "status": "pass" if maximum <= args.max_runtime_ratio else "fail",
            "threshold": args.max_runtime_ratio,
            "worst_matched_median_ratio": maximum,
        }
    provenance = {
        "input_path": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "unique_input_sha256": hashlib.sha256(distinct.tobytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "scikit-learn", "umap-learn", "fast-umap", "plotly")
        },
    }
    report = {
        "provenance": provenance,
        "configuration": {
            "device": args.device,
            "device_routing": (
                "fast_umap uses the selected device; umap-learn reference runs on CPU"
            ),
            "seeds": [int(seed) for seed in args.seeds],
            "evaluation_seed": EVALUATION_SEED,
            "neighbors": N_NEIGHBORS,
            "representative_count": len(reps),
            "representative_indices_unique_order": reps.tolist(),
            "n_components": 2,
            "min_dist": 0.0,
            "n_epochs": args.n_epochs,
            "n_epochs_policy": "library default" if args.n_epochs is None else "explicit",
            "methods": selected_methods,
            "gate_thresholds": thresholds,
            "runtime_protocol": "one unseeded warmup plus five paired measured runs"
            if runtime_protocol is not None
            else None,
        },
        "input": {
            "raw_rows": int(x.shape[0]),
            "unique_rows": int(distinct.shape[0]),
            "features": int(x.shape[1]),
        },
        "runs": run_results,
        "uncertainty": metrics_summary,
        "gates": gates,
        "runtime_ratio_gate": runtime_gate,
        "runtime_protocol_results": runtime_protocol,
        "failures": failures,
        "timing_notes": (
            "full_runtime_seconds is child-measured model.fit_transform wall time, "
            "including native unique work; subprocess_wall_seconds includes imports, "
            "process startup, and file I/O."
        ),
    }
    _write_html(args.output.with_suffix(".html"), embeddings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _fit_in_child(
    matrix: np.ndarray,
    *,
    method: str,
    device: str,
    seed: int | None,
    n_epochs: int | None,
    timeout: float,
    fit_count: int = 1,
) -> dict[str, Any]:
    worker = r"""
import json, sys, time, warnings
import numpy as np
from pathlib import Path
source, target, method, device, seed, epochs, fit_count = sys.argv[1:]
x = np.load(source, allow_pickle=False)
is_rust = method.startswith("rust-")
settings = dict(
    n_components=2,
    n_neighbors=15,
    min_dist=0.0,
    init="spectral",
    transform_seed=42 if seed == "none" else int(seed),
    random_state=None if seed == "none" else int(seed),
    unique=method.endswith("unique"),
)
if epochs != "none": settings["n_epochs"] = int(epochs)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    if is_rust:
        from fast_umap import UMAP
        model = UMAP(
            **settings,
            backend="flex" if device == "cpu" else "cuda",
            neighbor_search="approximate",
        )
    else:
        import umap
        model = umap.UMAP(**settings, metric="euclidean", verbose=False,
                          n_jobs=1 if seed != "none" else -1)
    fit_seconds = []
    for _ in range(int(fit_count)):
        started = time.perf_counter()
        embedding = np.asarray(model.fit_transform(x), dtype=np.float32)
        fit_seconds.append(time.perf_counter() - started)
        if embedding.shape != (x.shape[0], 2) or not np.isfinite(embedding).all():
            raise ValueError("fit_transform returned invalid embedding shape or non-finite values")
np.save(target, embedding, allow_pickle=False)
warning_data = [
    {"category": warning.category.__name__, "message": str(warning.message)}
    for warning in caught
]
print(json.dumps({"fit_seconds": fit_seconds, "warnings": warning_data}))
"""
    with tempfile.TemporaryDirectory(prefix="umap-eval-") as temp:
        root = Path(temp)
        source, target = root / "input.npy", root / "embedding.npy"
        np.save(source, matrix, allow_pickle=False)
        command = [
            sys.executable,
            "-c",
            worker,
            str(source),
            str(target),
            method,
            device,
            "none" if seed is None else str(seed),
            "none" if n_epochs is None else str(n_epochs),
            str(fit_count),
        ]
        started = perf_counter()
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=timeout * fit_count, check=False
            )
        except subprocess.TimeoutExpired as error:
            elapsed = perf_counter() - started
            raise ChildFitError(
                f"{method} fit exceeded {timeout * fit_count:g}s timeout",
                {
                    "status": "timeout",
                    "timeout_seconds": timeout * fit_count,
                    "subprocess_wall_seconds": elapsed,
                    "stdout": _text(error.stdout),
                    "stderr": _text(error.stderr),
                },
            ) from error
        elapsed = perf_counter() - started
        if completed.returncode != 0:
            raise ChildFitError(
                f"{method} child failed with exit status {completed.returncode}",
                {
                    "status": "failed",
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
        try:
            metadata = json.loads(completed.stdout.strip().splitlines()[-1])
            embedding = np.load(target, allow_pickle=False)
        except (ValueError, IndexError, OSError) as error:
            raise ChildFitError(
                f"{method} child produced invalid output: {error}",
                {
                    "status": "invalid_output",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            ) from error
        warning_text = [f"{item['category']}: {item['message']}" for item in metadata["warnings"]]
        fallback_text = (completed.stderr + "\n" + "\n".join(warning_text)).lower()
        if (
            "using seeded uniform" in fallback_text
            or ("spectral initialization" in fallback_text and "failed" in fallback_text)
            or ("spectral initialisation" in fallback_text and "failed" in fallback_text)
        ):
            raise ChildFitError(
                f"{method} spectral initialization fallback detected",
                {
                    "status": "spectral_fallback",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                    "warnings": warning_text,
                },
            )
        fit_times = metadata["fit_seconds"]
        if (
            not isinstance(fit_times, list)
            or len(fit_times) != fit_count
            or any(
                not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0
                for value in fit_times
            )
        ):
            raise ChildFitError(
                f"{method} child produced invalid fit timing",
                {
                    "status": "invalid_output",
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                },
            )
        return {
            "embedding": embedding,
            "fit_seconds": fit_times[0] if fit_count == 1 else fit_times,
            "process_seconds": elapsed,
            "warnings": warning_text,
            "stderr": completed.stderr,
        }


class ChildFitError(RuntimeError):
    def __init__(self, message: str, evidence: dict[str, Any]) -> None:
        super().__init__(message)
        self.evidence = evidence


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode(errors="replace") if isinstance(value, bytes) else value


def _persist_failure(path: Path, failures: list[dict[str, Any]], runs: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": "failed", "runs": runs, "failures": failures}, indent=2) + "\n",
        encoding="utf-8",
    )


def _runtime_worker(
    x: np.ndarray, *, method: str, device: str, n_epochs: int | None, timeout: float
) -> dict[str, Any]:
    result = _fit_in_child(
        x,
        method=method,
        device=device,
        seed=None,
        n_epochs=n_epochs,
        timeout=timeout,
        fit_count=6,
    )
    return {"fit_seconds": result["fit_seconds"], "process_seconds": result["process_seconds"]}


def _run_runtime_protocol(
    x: np.ndarray, methods: list[str], device: str, n_epochs: int | None, timeout: float
) -> dict[str, Any]:
    available = set(methods)
    pairs = [
        (raw, raw.replace("rust-", "python-"))
        for raw in ("rust-raw", "rust-unique")
        if raw in available and raw.replace("rust-", "python-") in available
    ]
    if not pairs:
        raise ValueError("runtime protocol requires at least one matched Rust/Python method pair")
    results: dict[str, Any] = {}
    for rust_method, python_method in pairs:
        measurements: dict[str, list[float]] = {rust_method: [], python_method: []}
        protocol_runs: dict[str, dict[str, Any]] = {}
        for method in (rust_method, python_method):
            protocol_runs[method] = _runtime_worker(
                x, method=method, device=device, n_epochs=n_epochs, timeout=timeout
            )
            measurements[method] = protocol_runs[method]["fit_seconds"][1:]
        rust_median = float(np.median(measurements[rust_method]))
        python_median = float(np.median(measurements[python_method]))
        results[f"{rust_method}:{python_method}"] = {
            "warmup_runs": 1,
            "measured_runs": 5,
            "rust_warmup_seconds": protocol_runs[rust_method]["fit_seconds"][0],
            "python_warmup_seconds": protocol_runs[python_method]["fit_seconds"][0],
            "rust_median_seconds": rust_median,
            "python_median_seconds": python_median,
            "median_ratio": rust_median / python_median,
            "rust_samples_seconds": measurements[rust_method],
            "python_samples_seconds": measurements[python_method],
            "rust_process_seconds": protocol_runs[rust_method]["process_seconds"],
            "python_process_seconds": protocol_runs[python_method]["process_seconds"],
            "process_model": (
                "one fresh subprocess per method, one warmup plus five sequential fits; "
                "caches persist within each worker and are cold across workers; "
                "subprocess startup/import excluded from fit medians; "
                "Rust compilation/build is external and excluded"
            ),
            "native_unique": rust_method.endswith("unique"),
        }
    return results


def _write_html(path: Path, embeddings: dict[str, list[dict[str, Any]]]) -> None:
    figure = make_subplots(
        rows=1, cols=2, subplot_titles=("Full range", "Robust 1-99 percentile range")
    )
    all_values = np.concatenate(
        [item["values"] for items in embeddings.values() for item in items], axis=0
    )
    robust = np.quantile(all_values, [0.01, 0.99], axis=0)
    full = np.quantile(all_values, [0.0, 1.0], axis=0)
    for method, items in embeddings.items():
        for item in items:
            points = item["values"]
            label = f"{method} seed={item['seed']}"
            for col in (1, 2):
                figure.add_trace(
                    go.Scattergl(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        name=label,
                        legendgroup=label,
                        showlegend=col == 1,
                        marker={"size": 3, "opacity": 0.55},
                    ),
                    row=1,
                    col=col,
                )
    figure.update_xaxes(
        range=full[:, 0].tolist(),
        title_text=f"x full [{full[0, 0]:.6g}, {full[1, 0]:.6g}]",
        row=1,
        col=1,
    )
    figure.update_yaxes(
        range=full[:, 1].tolist(),
        title_text=f"y full [{full[0, 1]:.6g}, {full[1, 1]:.6g}]",
        row=1,
        col=1,
    )
    figure.update_xaxes(
        range=robust[:, 0].tolist(),
        title_text=f"x p01-p99 [{robust[0, 0]:.6g}, {robust[1, 0]:.6g}]",
        row=1,
        col=2,
    )
    figure.update_yaxes(
        range=robust[:, 1].tolist(),
        title_text=f"y p01-p99 [{robust[0, 1]:.6g}, {robust[1, 1]:.6g}]",
        row=1,
        col=2,
    )
    figure.update_layout(title="UMAP fixed-representative comparison", height=700)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pio.to_html(figure, include_plotlyjs=True, full_html=True), encoding="utf-8")


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("seeds must be comma-separated integers") from error
    if not seeds or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return seeds


def _parse_methods(value: str) -> list[str]:
    methods = value.split(",")
    if not methods or any(method not in METHODS for method in methods):
        raise argparse.ArgumentTypeError(
            f"methods must be comma-separated choices from {','.join(METHODS)}"
        )
    if len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("methods must not contain duplicates")
    return methods


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = evaluate(args)
    if report["runtime_ratio_gate"]["status"] == "fail" or any(
        gate["status"] == "fail" for gates in report["gates"].values() for gate in gates.values()
    ):
        raise SystemExit(
            "comparison failed an explicitly supplied gate; evidence is in the JSON report"
        )


if __name__ == "__main__":
    main()
