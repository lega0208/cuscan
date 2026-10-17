#!/usr/bin/env -S uv run --no-sync
"""Per-stage fit timing: fast-umap Standard UMAP (CPU by default) vs umap-learn 0.5.12.

Build the Rust companion first (build time is not measured):

    cargo build --release --no-default-features --features cpu,tracing \
        --example profile_standard_stages

Then run from the repository root:

    uv run --no-sync python comparison/compare_stages.py [--html stages.html]

Both implementations fit the same clustered float32 input (the generator from
scripts/compare_umap_spans.py with configurable center scale/noise) in a fresh subprocess
each. The defaults overlap the clusters so the k-NN graph is connected: with the original
well-separated clusters (--center-scale 5 --noise 0.35) fast-umap spectral init currently
fails with "component-centroid affinities underflowed"; use --init random for those.
Each process performs exactly one fit, so timings are cold: they include CubeCL kernel JIT
and numba compilation in that process (on-disk caches from earlier processes may still
apply).

Rust stages come from the library's `tracing` spans (with device barriers). Python
stages are measured by wrapping the umap-learn module functions that `UMAP.fit` calls:
nearest_neighbors, fuzzy_simplicial_set, make_epochs_per_sample, spectral_layout
(or the random init path, which is then counted in "other") and
optimize_layout_euclidean. "other" is total fit time minus the measured stages.
umap-learn has no separate spectral graph export step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target" / "release" / "examples" / "profile_standard_stages"
STAGES = (
    "neighbor_search",
    "fuzzy_graph",
    "schedule",
    "spectral_graph_export",
    "initialization",
    "optimization",
)
# umap-learn function -> shared stage name
PYTHON_STAGE_FUNCTIONS = {
    "nearest_neighbors": "neighbor_search",
    "fuzzy_simplicial_set": "fuzzy_graph",
    "make_epochs_per_sample": "schedule",
    "spectral_layout": "initialization",
    "optimize_layout_euclidean": "optimization",
}
SPEEDUP_TITLE = "Rust faster by (%)"


def _load_spans_helpers() -> Any:
    """Reuse the JSON/median helpers from scripts/compare_umap_spans.py."""
    path = ROOT / "scripts" / "compare_umap_spans.py"
    spec = importlib.util.spec_from_file_location("compare_umap_spans", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spans = _load_spans_helpers()


def make_clustered_input(
    rows: int, features: int, center_scale: float, noise: float, destination: Path
) -> None:
    """spans.make_clustered_input with configurable center scale and noise."""
    import numpy as np

    rng = np.random.default_rng(42)
    centers = rng.normal(0.0, center_scale, size=(8, features)).astype(np.float32)
    labels = np.arange(rows) % len(centers)
    rng.shuffle(labels)
    matrix = centers[labels] + rng.normal(0.0, noise, size=(rows, features)).astype(np.float32)
    matrix.astype("<f4", copy=False).tofile(destination)


def _python_worker(args: argparse.Namespace) -> None:
    import numpy as np
    import umap
    import umap.umap_ as umap_module
    from sklearn.neighbors import NearestNeighbors

    if umap.__version__ != "0.5.12":
        raise SystemExit(f"requires umap-learn 0.5.12, got {umap.__version__}")
    matrix = np.fromfile(args.input, dtype="<f4").reshape(args.rows, args.features)
    current: dict[str, float] = {}

    def timed(stage: str, function: Any) -> Any:
        def wrapper(*inner_args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                return function(*inner_args, **kwargs)
            finally:
                current[stage] = current.get(stage, 0.0) + time.perf_counter() - start

        return wrapper

    for name, stage in PYTHON_STAGE_FUNCTIONS.items():
        setattr(umap_module, name, timed(stage, getattr(umap_module, name)))

    start = time.perf_counter()
    precomputed: tuple[Any, Any, Any] = (None, None, None)
    if args.search == "exact":
        search_start = time.perf_counter()
        distances, indices = (
            NearestNeighbors(n_neighbors=args.neighbors, n_jobs=-1).fit(matrix).kneighbors(matrix)
        )
        current["neighbor_search"] = time.perf_counter() - search_start
        precomputed = (indices, distances, None)
    umap.UMAP(
        n_neighbors=args.neighbors,
        n_components=2,
        n_epochs=args.epochs,
        init=args.init,
        min_dist=args.min_dist,
        random_state=args.seed if args.seeded_python else None,
        metric="euclidean",
        n_jobs=1 if args.seeded_python else -1,
        force_approximation_algorithm=args.search == "approximate",
        precomputed_knn=precomputed,
        verbose=False,
    ).fit(matrix)
    fit_seconds = [time.perf_counter() - start]
    stage_seconds = {stage: [current.get(stage, 0.0)] for stage in PYTHON_STAGE_FUNCTIONS.values()}
    print(json.dumps({"fit_seconds": fit_seconds, "stage_seconds": stage_seconds}))


def _run(command: list[str], label: str) -> dict[str, Any]:
    print(f"running {label}: {' '.join(command)}", file=sys.stderr)
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise SystemExit(f"{label} failed:\n{completed.stdout}\n{completed.stderr}")
    return spans.last_json(completed.stdout)


def _summarize(result: dict[str, Any]) -> dict[str, float | None]:
    """Return {stage: ms} for the single cold fit, including other/total."""
    (total,) = result["fit_seconds"]
    row: dict[str, float | None] = {}
    for stage in STAGES:
        values = result["stage_seconds"].get(stage)
        row[stage] = None if values is None else values[0] * 1e3
    measured = sum(value for value in row.values() if value is not None)
    row["other"] = total * 1e3 - measured
    row["total"] = total * 1e3
    return row


def _speedup(rust: float | None, python: float | None) -> float | None:
    if rust is None or python is None or python <= 0.0:
        return None
    return -(rust / python - 1.0) * 100.0


def _rows(rust: dict[str, float | None], python: dict[str, float | None]) -> list[dict[str, Any]]:
    total = rust["total"]
    assert total is not None
    rows = []
    for stage in (*STAGES, "other", "total"):
        value = rust[stage]
        rows.append(
            {
                "stage": stage,
                "rust_ms": value,
                "percent": None if value is None else value / total * 100.0,
                "python_ms": python[stage],
                "speedup": _speedup(value, python[stage]),
            }
        )
    return rows


def _fmt(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:,.1f}{suffix}"


def _print_table(title: str, rows: list[dict[str, Any]]) -> None:
    headers = ("Stage", "Time (ms)", "% of total", "Python time (ms)", SPEEDUP_TITLE)
    body = [
        (
            row["stage"],
            _fmt(row["rust_ms"]),
            _fmt(row["percent"], "%"),
            _fmt(row["python_ms"]),
            _fmt(row["speedup"], "%"),
        )
        for row in rows
    ]
    widths = [
        max(len(str(cell)) for cell in column) for column in zip(headers, *body, strict=False)
    ]
    line = "  ".join("-" * width for width in widths)
    print(f"\n{title}")
    print(
        "  ".join(
            h.ljust(w) if i == 0 else h.rjust(w)
            for i, (h, w) in enumerate(zip(headers, widths, strict=False))
        )
    )
    print(line)
    for cells in body:
        if cells[0] == "total":
            print(line)
        print(
            "  ".join(
                c.ljust(w) if i == 0 else c.rjust(w)
                for i, (c, w) in enumerate(zip(cells, widths, strict=False))
            )
        )


def _write_html(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Rust: % of total fit time",
            f"{SPEEDUP_TITLE} (positive = Rust faster)",
            "Raw stage timings (ms)",
        ),
        vertical_spacing=0.1,
    )
    parts = [row for row in rows if row["stage"] != "total"]
    figure.add_trace(
        go.Bar(
            name="% of total",
            x=[r["stage"] for r in parts],
            y=[r["percent"] for r in parts],
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            name=SPEEDUP_TITLE,
            x=[r["stage"] for r in rows],
            y=[r["speedup"] for r in rows],
        ),
        row=2,
        col=1,
    )
    for implementation, key in (("Rust", "rust_ms"), ("Python", "python_ms")):
        figure.add_trace(
            go.Bar(name=implementation, x=[r["stage"] for r in rows], y=[r[key] for r in rows]),
            row=3,
            col=1,
        )
    figure.update_yaxes(title_text="%", row=1, col=1)
    figure.update_yaxes(title_text="%", row=2, col=1)
    figure.update_yaxes(title_text="ms", row=3, col=1)
    figure.update_layout(title=title, barmode="group", height=1300)
    figure.write_html(path)
    print(f"\nwrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200, help="200 = library default for n>10k")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--init", choices=["spectral", "random"], default="spectral")
    parser.add_argument("--search", choices=["approximate", "exact"], default="approximate")
    parser.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--center-scale", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument(
        "--seeded-python",
        action="store_true",
        help="seed umap-learn (forces single-threaded layout); default is unseeded, all threads",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--html", type=Path, help="write plotly bar charts to this HTML file")
    parser.add_argument("--json", type=Path, help="write raw per-run results to this JSON file")
    parser.add_argument("--_python-worker", dest="worker_input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_input is not None:
        args.input = args.worker_input
        _python_worker(args)
        return 0
    if not args.binary.exists():
        parser.error(
            f"{args.binary} missing; build with: cargo build --release --no-default-features "
            f"--features {args.backend},tracing --example profile_standard_stages"
        )

    with tempfile.TemporaryDirectory() as temp:
        input_path = Path(temp) / "input.f32le"
        make_clustered_input(args.rows, args.features, args.center_scale, args.noise, input_path)
        shared = ["--neighbors", str(args.neighbors), "--seed", str(args.seed)]
        shared += ["--min-dist", str(args.min_dist)]
        rust = _run(
            [
                str(args.binary),
                str(input_path),
                str(args.rows),
                str(args.features),
                str(args.epochs),
                *shared,
                "--runs",
                "1",
                "--backend",
                args.backend,
                "--search",
                args.search,
                "--init",
                args.init,
            ],
            "rust",
        )
        python_command = [sys.executable, str(Path(__file__).resolve())]
        python_command += ["--_python-worker", str(input_path), "--rows", str(args.rows)]
        python_command += ["--features", str(args.features), "--epochs", str(args.epochs)]
        python_command += ["--search", args.search, "--init", args.init, *shared]
        if args.seeded_python:
            python_command.append("--seeded-python")
        python = _run(python_command, "python")

    rows = _rows(_summarize(rust), _summarize(python))
    title = (
        f"fast-umap {args.backend} vs umap-learn 0.5.12: {args.rows:,}x{args.features}, "
        f"{args.epochs} epochs, k={args.neighbors}, {args.search} search, {args.init} init, "
        f"{os.cpu_count()} CPUs"
    )
    print(title)
    _print_table("Cold run (one fit per fresh process)", rows)
    if args.json:
        args.json.write_text(json.dumps({"rust": rust, "python": python, "rows": rows}, indent=2))
    if args.html:
        _write_html(args.html, rows, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
