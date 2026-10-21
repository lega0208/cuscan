#!/usr/bin/env python3
"""Compare native fast-hdbscan and fast_hdbscan 0.3.2 stage timings.

Build the native companion before running (the backend must match --backend):

    cargo build -p fast-hdbscan --release --no-default-features \\
        --features cpu,tracing --example profile_hdbscan_stages

Then run, for example:

    uv run --no-sync python comparison/compare_hdbscan_stages.py --html hdbscan-stages.html

Input generation/loading and estimator construction are outside fit timing. Native and
reference stage spans are exclusive. fast_hdbscan 0.3.2 fuses reference core distances
and MST in one Numba Boruvka call; above 30 features its pynndescent stage is reported as
one inclusive approximate-neighbors-and-MST span. These paths are not algorithmically
equivalent to native --search. Unattributed is fit minus measured spans (timer/bookkeeping
and uncaptured overhead), not measured work. One run is cold; for >=2 runs summaries use
the median of each implementation's warmed fits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_hdbscan_stages"
NATIVE_STAGES = (
    "input_validation",
    "backend_setup",
    "input_preparation",
    "tree_build",
    "input_upload",
    "core_distances",
    "mst",
    "cluster_extraction",
    "result_finalization",
)
REFERENCE_STAGES = (
    "input_validation",
    "mst_setup",
    "tree_build",
    "core_distances_and_mst",
    "approximate_neighbors_and_mst",
    "cluster_extraction",
    "result_finalization",
)
STAGES = (
    "input_validation",
    "backend_setup",
    "input_preparation",
    "mst_setup",
    "tree_build",
    "input_upload",
    "core_distances",
    "core_distances_and_mst",
    "mst",
    "approximate_neighbors_and_mst",
    "cluster_extraction",
    "result_finalization",
)
STAGE_DEFINITIONS = {
    "input_validation": (
        "Native validation; reference estimator-fit prefix and fast_hdbscan parameter validation, "
        "excluding measured children."
    ),
    "backend_setup": "Native backend initialization/setup.",
    "input_preparation": "Native host-side input preparation.",
    "mst_setup": (
        "Reference compute_minimum_spanning_tree dispatch and checks, exclusive of its children."
    ),
    "tree_build": "KD-tree construction (native or reference).",
    "input_upload": "Native host-to-backend input upload.",
    "core_distances": (
        "Native core-distance calculation; reference Boruvka fuses this with MST "
        "and is not separately measurable."
    ),
    "mst": "Native minimum spanning tree work; reference Boruvka fuses this with core distances.",
    "core_distances_and_mst": (
        "Reference parallel_boruvka fused Numba operation; "
        "not separable into core-distance and MST timings."
    ),
    "approximate_neighbors_and_mst": (
        "Reference compute_mst_from_knn_graph inclusive operation for >30 features."
    ),
    "cluster_extraction": "Cluster extraction from the spanning tree.",
    "result_finalization": (
        "Reference fit suffix after fast_hdbscan returns, including to_numpy_rec_array; "
        "native result finalization."
    ),
    "unattributed": (
        "Fit minus measured exclusive spans: timer/bookkeeping and uncaptured overhead, "
        "not measured work."
    ),
}


class ExclusiveTimer:
    """Accumulate exclusive named intervals; nested measures pause their parent."""

    def __init__(self, clock: Any = None):
        self.clock = clock or time.perf_counter
        self.seconds: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        self._stack: list[tuple[str, float]] = []

    def _pause(self) -> None:
        if self._stack:
            stage, started = self._stack[-1]
            now = self.clock()
            self.seconds[stage] = self.seconds.get(stage, 0.0) + now - started
            self._stack[-1] = (stage, now)

    def switch(self, stage: str | None) -> None:
        self._pause()
        if self._stack:
            self._stack.pop()
        if stage is not None:
            self._stack.append((stage, self.clock()))
            self.calls[stage] = self.calls.get(stage, 0) + 1

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        self._pause()
        self._stack.append((stage, self.clock()))
        self.calls[stage] = self.calls.get(stage, 0) + 1
        try:
            yield
        finally:
            self._pause()
            self._stack.pop()
            if self._stack:
                name, _ = self._stack[-1]
                self._stack[-1] = (name, self.clock())


@contextmanager
def measure_reference(
    module: Any, approximate_module: Any, timer: ExclusiveTimer
) -> Iterator[None]:
    """Instrument fast_hdbscan 0.3.2 callsites, restoring all patched hooks on exit."""
    originals: list[tuple[Any, str, Any]] = []

    def patch(target: Any, name: str, stage: str, *, required: bool = True) -> None:
        if not hasattr(target, name):
            if required:
                raise RuntimeError(f"required fast_hdbscan timing hook missing: {name}")
            return
        original = getattr(target, name)
        originals.append((target, name, original))

        def timed(*args: Any, **kwargs: Any) -> Any:
            with timer.measure(stage):
                return original(*args, **kwargs)

        setattr(target, name, timed)

    try:
        patch(module, "build_kdtree", "tree_build")
        patch(module, "parallel_boruvka", "core_distances_and_mst")
        patch(approximate_module, "compute_mst_from_knn_graph", "approximate_neighbors_and_mst")
        patch(module, "clusters_from_spanning_tree", "cluster_extraction")
        # Instrument the complete compute function as an exclusive outer dispatch frame.
        patch(module, "compute_minimum_spanning_tree", "mst_setup")
        yield
    finally:
        for target, name, original in reversed(originals):
            setattr(target, name, original)


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
    import fast_hdbscan.nndescent as approximate_module

    matrix = np.fromfile(args.worker_input, dtype="<f4").reshape(args.rows, args.features)
    timer = ExclusiveTimer()
    fit_seconds: list[float] = []
    stage_seconds: dict[str, list[float | None]] = {stage: [] for stage in STAGES}
    cluster_counts: list[int] = []
    noise_counts: list[int] = []
    original_fast_hdbscan = hdbscan_module.fast_hdbscan
    original_fit = fast_hdbscan.HDBSCAN.fit

    def timed_fast_hdbscan(*call_args: Any, **kwargs: Any) -> Any:
        timer.switch("input_validation")
        try:
            return original_fast_hdbscan(*call_args, **kwargs)
        finally:
            timer.switch("result_finalization")

    def timed_fit(estimator: Any, *call_args: Any, **kwargs: Any) -> Any:
        timer.switch("input_validation")
        try:
            return original_fit(estimator, *call_args, **kwargs)
        finally:
            timer.switch(None)

    hdbscan_module.fast_hdbscan = timed_fast_hdbscan
    fast_hdbscan.HDBSCAN.fit = timed_fit
    try:
        with measure_reference(hdbscan_module, approximate_module, timer):
            for _ in range(args.runs):
                timer.seconds.clear()
                timer.calls.clear()
                estimator = fast_hdbscan.HDBSCAN(
                    min_cluster_size=args.min_cluster_size,
                    min_samples=args.min_samples,
                    cluster_selection_method=args.cluster_selection_method,
                    allow_single_cluster=args.allow_single_cluster,
                )
                start = time.perf_counter()
                estimator.fit(matrix)
                fit_seconds.append(time.perf_counter() - start)
                high_dimensional = matrix.shape[1] > 30
                required = [
                    "input_validation",
                    "mst_setup",
                    "cluster_extraction",
                    "result_finalization",
                ]
                required += (
                    ["approximate_neighbors_and_mst"]
                    if high_dimensional
                    else ["tree_build", "core_distances_and_mst"]
                )
                for stage in required:
                    if not timer.calls.get(stage):
                        raise RuntimeError(f"reference fit did not execute measured stage {stage}")
                for stage in STAGES:
                    stage_seconds[stage].append(
                        timer.seconds.get(stage) if timer.calls.get(stage) else None
                    )
                labels = estimator.labels_
                cluster_counts.append(len(set(labels.tolist()) - {-1}))
                noise_counts.append(int(np.count_nonzero(labels == -1)))
    finally:
        hdbscan_module.fast_hdbscan = original_fast_hdbscan
        fast_hdbscan.HDBSCAN.fit = original_fit
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


def validate_native_stages(result: dict[str, Any]) -> None:
    stage_values = result.get("stage_seconds")
    if not isinstance(stage_values, dict):
        raise SystemExit(
            "native profiler output is missing stage_seconds; rebuild the native companion "
            "with tracing enabled"
        )
    missing = set(NATIVE_STAGES) - stage_values.keys()
    if missing:
        raise SystemExit(
            "native profiler output is missing required stages "
            f"{', '.join(sorted(missing))}; rebuild the native companion with tracing enabled"
        )


def summarize(result: dict[str, Any], indices: range) -> dict[str, float | None]:
    fit = result["fit_seconds"]
    if not indices:
        raise ValueError("no fits selected for summary")
    if not isinstance(fit, list) or not fit:
        raise ValueError("fit_seconds must be a non-empty list")
    for value in fit:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("fit_seconds contains a malformed, nonfinite, or negative duration")
    selected = list(indices)
    if any(index < 0 or index >= len(fit) for index in selected):
        raise ValueError("fit summary index is out of range")
    residuals = [float(fit[i]) for i in selected]
    stage_values: dict[str, list[float | None]] = result.get("stage_seconds", {})
    if not isinstance(stage_values, dict):
        raise ValueError("stage_seconds must be an object")
    unknown = stage_values.keys() - set(STAGES)
    if unknown:
        raise ValueError(
            f"stage_seconds contains unrecognized stages: {', '.join(sorted(unknown))}"
        )
    totals = [0.0 for _ in selected]
    row: dict[str, float | None] = {"total": statistics.median(residuals) * 1000}
    for stage in STAGES:
        values = stage_values.get(stage)
        if values is None:
            row[stage] = None
            continue
        if not isinstance(values, list) or len(values) != len(fit):
            raise ValueError(f"stage_seconds[{stage!r}] must align with fit_seconds")
        present = []
        for position, index in enumerate(selected):
            value = values[index]
            if value is None:
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"stage {stage!r} contains a malformed duration")
            duration = float(value)
            totals[position] += duration
            present.append(duration)
        row[stage] = statistics.median(present) * 1000 if present else None
    for position, fit_seconds in enumerate(residuals):
        tolerance = max(1e-9, fit_seconds * 1e-6)
        if totals[position] > fit_seconds + tolerance:
            raise ValueError("exclusive stage durations exceed fit duration")
        residuals[position] = fit_seconds - totals[position]
    row["unattributed"] = statistics.median(residuals) * 1000
    return row


def comparison_rows(
    native: dict[str, float | None], reference: dict[str, float | None]
) -> list[dict[str, Any]]:
    rows = []
    for stage in (*STAGES, "unattributed", "total"):
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
                if (
                    stage != "total"
                    and (stage not in NATIVE_STAGES or stage not in REFERENCE_STAGES)
                )
                or value is None
                or other is None
                or other <= 0
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
    figure.update_layout(
        title=title,
        barmode="group",
        height=1100,
        margin={"b": 250},
    )
    figure.add_annotation(
        text=(
            "Stages are exclusive. Reference Boruvka fuses core distances and MST; "
            ">30D uses approximate-neighbors+MST.<br>"
            "n/a means not separately applicable. Unattributed is timer/bookkeeping/uncaptured "
            "overhead, not measured work.<br>"
            "Native/reference stages differ; unmatched stages have no speedup."
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.22,
        showarrow=False,
        xanchor="left",
        align="left",
    )
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
        validate_native_stages(native)
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
            "stage_definitions": STAGE_DEFINITIONS,
            "timing_limitations": (
                "Exclusive spans; reference core_distances_and_mst is a fused Numba call; "
                "reference >30-feature approximate_neighbors_and_mst is inclusive; "
                "unattributed is timer/bookkeeping and uncaptured overhead, not measured work. "
                "n/a marks stages not separately applicable, and unmatched stages have no speedup."
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
