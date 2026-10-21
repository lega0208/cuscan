#!/usr/bin/env python3
"""Compare native fast-hdbscan and fast_hdbscan 0.3.2 stage timings.

Build the native companion before running (the backend must match --backend):

    cargo build -p fast-hdbscan --release --no-default-features \\
        --features cpu,tracing --example profile_hdbscan_stages

Then run, for example:

    uv run --no-sync python comparison/compare_hdbscan_stages.py --html hdbscan-stages.html

Input generation/loading and subprocess startup are outside fit timing. Rust stage spans
include host tree build/upload, core distances and MST in minimum_spanning_tree, then host
clustering in cluster_extraction. Python wraps the module functions called by HDBSCAN.fit.
For Euclidean inputs with more than 30 features, fast_hdbscan switches its MST path to
pynndescent approximate neighbors; otherwise it uses its KD-tree/Boruvka path. These are
not algorithmically equivalent to the native --search choice. One run is cold; when runs
are at least two, each implementation's comparison is the median of its warmed fits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_hdbscan_stages"
STAGES = ("minimum_spanning_tree", "cluster_extraction")
PYTHON_STAGE_FUNCTIONS = {
    "compute_minimum_spanning_tree": "minimum_spanning_tree",
    "clusters_from_spanning_tree": "cluster_extraction",
}


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def make_input(
    rows: int, features: int, center_scale: float, noise: float, seed: int, path: Path
) -> None:
    import numpy as np

    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, center_scale, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % len(centers)
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, noise, size=(rows, features)).astype(np.float32)
    matrix.astype("<f4", copy=False).tofile(path)


def load_parquet(path: Path, sample_size: int, cluster_on: str, seed: int) -> Any:
    import numpy as np
    import polars as pl

    column = "embedding_2d" if cluster_on == "2d" else "embedding"
    try:
        data = pl.scan_parquet(path).select(column).collect()
    except Exception as error:
        raise ValueError(f"could not read {column!r} from parquet {path}: {error}") from error
    if data.height == 0:
        raise ValueError("input parquet must contain rows")
    if sample_size < data.height:
        data = data.sample(n=sample_size, seed=seed, shuffle=True)
    try:
        matrix = np.asarray(data.get_column(column).to_list(), dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{column} must contain rectangular numeric vectors") from error
    matrix = validate_matrix(matrix)
    if cluster_on == "2d" and matrix.shape[1] != 2:
        raise ValueError("embedding_2d must have exactly two dimensions")
    return matrix


def validate_matrix(matrix: Any) -> Any:
    import numpy as np

    if (
        matrix.ndim != 2
        or matrix.shape[0] < 2
        or matrix.shape[1] < 1
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("input must be finite with at least 2 rows and 1 feature")
    return np.asarray(matrix, dtype="<f4", order="C")


def _python_worker(args: argparse.Namespace) -> None:
    import fast_hdbscan
    import fast_hdbscan.hdbscan as hdbscan_module
    import numpy as np

    try:
        version = importlib.metadata.version("fast-hdbscan")
    except importlib.metadata.PackageNotFoundError as error:
        raise SystemExit("fast-hdbscan==0.3.2 is required in this Python environment") from error
    if version != "0.3.2":
        raise SystemExit(f"requires fast-hdbscan==0.3.2, got {version}")
    matrix = np.fromfile(args.worker_input, dtype="<f4").reshape(args.rows, args.features)
    current: dict[str, float] = {}
    originals: dict[str, Any] = {}

    def wrap(stage: str, function: Any) -> Any:
        def timed(*call_args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return function(*call_args, **kwargs)
            finally:
                current[stage] = current.get(stage, 0.0) + time.perf_counter() - start

        return timed

    for name, stage in PYTHON_STAGE_FUNCTIONS.items():
        originals[name] = getattr(hdbscan_module, name)
        setattr(hdbscan_module, name, wrap(stage, originals[name]))

    fit_seconds: list[float] = []
    stage_seconds = {stage: [] for stage in STAGES}
    cluster_counts: list[int] = []
    noise_counts: list[int] = []
    try:
        for _ in range(args.runs):
            current.clear()
            start = time.perf_counter()
            estimator = fast_hdbscan.HDBSCAN(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples,
                cluster_selection_method=args.cluster_selection_method,
                allow_single_cluster=args.allow_single_cluster,
            ).fit(matrix)
            fit_seconds.append(time.perf_counter() - start)
            for stage in STAGES:
                if stage not in current:
                    raise RuntimeError(f"reference fit did not execute measured stage {stage}")
                stage_seconds[stage].append(current[stage])
            labels = estimator.labels_
            cluster_counts.append(len(set(labels.tolist()) - {-1}))
            noise_counts.append(int(np.count_nonzero(labels == -1)))
    finally:
        for name, function in originals.items():
            setattr(hdbscan_module, name, function)
    print(
        json.dumps(
            {
                "fit_seconds": fit_seconds,
                "stage_seconds": stage_seconds,
                "cluster_counts": cluster_counts,
                "noise_counts": noise_counts,
            }
        )
    )


def run(command: list[str], label: str) -> dict[str, Any]:
    print(f"running {label}: {' '.join(command)}", file=sys.stderr)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        raise SystemExit(f"{label} failed:\n{result.stdout}\n{result.stderr}")
    for line in reversed(result.stdout.splitlines()):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise SystemExit(f"{label} did not emit a JSON object on its final output line")


def summarize(result: dict[str, Any], indices: range) -> dict[str, float | None]:
    fit = result["fit_seconds"]
    if not indices:
        raise ValueError("no fits selected for summary")
    row: dict[str, float | None] = {"total": statistics.median(fit[i] for i in indices) * 1000}
    residuals = [fit[i] for i in indices]
    for stage in STAGES:
        values = result.get("stage_seconds", {}).get(stage)
        if values is None:
            row[stage] = None
            continue
        for position, index in enumerate(indices):
            residuals[position] -= values[index]
        row[stage] = statistics.median(values[i] for i in indices) * 1000
    row["other"] = statistics.median(residuals) * 1000
    return row


def comparison_rows(
    native: dict[str, float | None], reference: dict[str, float | None]
) -> list[dict[str, Any]]:
    rows = []
    for stage in (*STAGES, "other", "total"):
        value, other = native[stage], reference[stage]
        rows.append(
            {
                "stage": stage,
                "native_ms": value,
                "percent": None
                if value is None or native["total"] in (None, 0)
                else value / native["total"] * 100,
                "python_ms": other,
                "speedup": None
                if value is None or other is None or other <= 0
                else (1 - value / other) * 100,
            }
        )
    return rows


def fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:,.1f}{suffix}"


def print_table(title: str, rows: list[dict[str, Any]]) -> None:
    headers = ("Stage", "Native (ms)", "Native %", "Python (ms)", "Native faster (%)")
    body = [
        (
            r["stage"],
            fmt(r["native_ms"]),
            fmt(r["percent"], "%"),
            fmt(r["python_ms"]),
            fmt(r["speedup"], "%"),
        )
        for r in rows
    ]
    widths = [max(len(str(cell)) for cell in col) for col in zip(headers, *body, strict=False)]
    print(
        f"\n{title}\n"
        + "  ".join(
            h.ljust(w) if i == 0 else h.rjust(w)
            for i, (h, w) in enumerate(zip(headers, widths, strict=False))
        )
    )
    print("  ".join("-" * w for w in widths))
    for cells in body:
        if cells[0] == "total":
            print("  ".join("-" * w for w in widths))
        print(
            "  ".join(
                c.ljust(w) if i == 0 else c.rjust(w)
                for i, (c, w) in enumerate(zip(cells, widths, strict=False))
            )
        )


def write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Native stage share", "Native speedup (%)", "Stage timings (ms)"),
    )
    parts = [row for row in rows if row["stage"] != "total"]
    figure.add_trace(
        go.Bar(
            name="Native % of fit", x=[r["stage"] for r in parts], y=[r["percent"] for r in parts]
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name="Native faster (%)", x=[r["stage"] for r in rows], y=[r["speedup"] for r in rows]
        ),
        row=2,
        col=1,
    )
    for name, key in (("Native", "native_ms"), ("Python", "python_ms")):
        figure.add_trace(
            go.Bar(name=name, x=[r["stage"] for r in rows], y=[r[key] for r in rows]), row=3, col=1
        )
    figure.update_yaxes(title_text="%", row=1, col=1)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="ms", row=3, col=1)
    figure.update_layout(title=title, barmode="group", height=1100)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path, include_plotlyjs=True, full_html=True)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=positive_int, default=5000)
    parser.add_argument("--features", type=positive_int, default=16)
    parser.add_argument("--center-scale", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--input", type=Path, help="optional parquet with embedding_2d or embedding column"
    )
    parser.add_argument("--cluster-on", choices=("2d", "full"), default="2d")
    parser.add_argument("--sample-size", type=positive_int, default=1_000_000)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--rust-runs", type=positive_int, default=1)
    parser.add_argument("--python-runs", type=positive_int, default=1)
    parser.add_argument("--min-cluster-size", type=positive_int, default=5)
    parser.add_argument("--min-samples", type=positive_int)
    parser.add_argument("--cluster-selection-method", choices=("eom", "leaf"), default="eom")
    parser.add_argument("--allow-single-cluster", action="store_true")
    parser.add_argument("--search", choices=("auto", "brute-force", "kd-tree"), default="auto")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--html", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--_python-worker", dest="worker_input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-rows", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-features", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-runs", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_input:
        args.rows, args.features, args.runs = (
            args._worker_rows,
            args._worker_features,
            args._worker_runs,
        )
        _python_worker(args)
        return 0
    if args.min_cluster_size < 2:
        parser.error("--min-cluster-size must be at least 2")
    if not args.binary.is_file():
        parser.error(
            f"{args.binary} missing; build with: cargo build -p fast-hdbscan --release "
            f"--no-default-features --features {args.backend},tracing "
            "--example profile_hdbscan_stages"
        )

    import numpy as np

    with tempfile.TemporaryDirectory() as temp:
        input_path = Path(temp) / "input.f32le"
        if args.input:
            matrix = load_parquet(args.input, args.sample_size, args.cluster_on, args.seed)
        else:
            if args.rows < 2:
                parser.error("--rows must be at least 2")
            make_input(
                args.rows, args.features, args.center_scale, args.noise, args.seed, input_path
            )
            matrix = validate_matrix(
                np.fromfile(input_path, dtype="<f4").reshape(args.rows, args.features)
            )
        rows, features = matrix.shape
        matrix.tofile(input_path)
        digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if features > 30:
            print(
                "warning: fast_hdbscan uses pynndescent approximate neighbors above 30 features; "
                "native search is not equivalent",
                file=sys.stderr,
            )
        native_cmd = [
            str(args.binary),
            str(input_path),
            str(rows),
            str(features),
            "--backend",
            args.backend,
            "--runs",
            str(args.rust_runs),
            "--min-cluster-size",
            str(args.min_cluster_size),
            "--cluster-selection-method",
            args.cluster_selection_method,
            "--search",
            args.search,
        ]
        if args.min_samples is not None:
            native_cmd += ["--min-samples", str(args.min_samples)]
        if args.allow_single_cluster:
            native_cmd.append("--allow-single-cluster")
        native = run(native_cmd, "native")
        py_cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_python-worker",
            str(input_path),
            "--_worker-rows",
            str(rows),
            "--_worker-features",
            str(features),
            "--_worker-runs",
            str(args.python_runs),
            "--min-cluster-size",
            str(args.min_cluster_size),
            "--cluster-selection-method",
            args.cluster_selection_method,
        ]
        if args.min_samples is not None:
            py_cmd += ["--min-samples", str(args.min_samples)]
        if args.allow_single_cluster:
            py_cmd.append("--allow-single-cluster")
        reference = run(py_cmd, "fast_hdbscan")
    native_indices = range(1, args.rust_runs) if args.rust_runs >= 2 else range(1)
    python_indices = range(1, args.python_runs) if args.python_runs >= 2 else range(1)
    comparison = comparison_rows(
        summarize(native, native_indices), summarize(reference, python_indices)
    )
    search_ref = "pynndescent" if features > 30 else "kd-tree-boruvka"
    title = (
        f"fast-hdbscan {args.backend} vs fast_hdbscan 0.3.2: {rows:,}x{features}, "
        f"{args.cluster_selection_method}, native {args.search}, reference {search_ref}"
    )
    print(title)
    native_mode = "median warmed fits" if args.rust_runs >= 2 else "cold fit"
    python_mode = "median warmed fits" if args.python_runs >= 2 else "cold fit"
    print_table(f"Native: {native_mode}; Python: {python_mode}", comparison)
    if args.json:
        metadata = {
            "input_sha256": digest,
            "rows": rows,
            "features": features,
            "input": str(args.input) if args.input else "synthetic",
            "cluster_on": args.cluster_on if args.input else None,
            "sample_size": rows if args.input else None,
            "seed": args.seed,
            "center_scale": args.center_scale,
            "noise": args.noise,
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "cluster_selection_method": args.cluster_selection_method,
            "allow_single_cluster": args.allow_single_cluster,
            "backend": args.backend,
            "search": args.search,
            "reference_search": search_ref,
            "rust_runs": args.rust_runs,
            "python_runs": args.python_runs,
            "warm_semantics": (
                "run 0 cold; when runs >= 2 compare median of runs 1..N-1; per implementation"
            ),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {"metadata": metadata, "native": native, "python": reference, "rows": comparison},
                indent=2,
            )
        )
    if args.html:
        write_html(args.html, comparison, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
