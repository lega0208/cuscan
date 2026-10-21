import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "embeddings_2d.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "hdbscan_comparison.html"
SAMPLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference fast_hdbscan and fast_umap on one data sample."
    )
    parser.add_argument("--cluster-on", choices=("2d", "full"), default="2d")
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int)
    parser.add_argument("--cluster-selection-method", choices=("eom", "leaf"), default="eom")
    parser.add_argument("--allow-single-cluster", action="store_true")
    return parser.parse_args()


def _matrix(values: list[object], rows: int, column: str) -> np.ndarray:
    try:
        matrix = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must form a rectangular numeric matrix") from error
    if (
        matrix.ndim != 2
        or matrix.shape[0] != rows
        or matrix.shape[1] == 0
        or not np.issubdtype(matrix.dtype, np.number)
        or np.issubdtype(matrix.dtype, np.complexfloating)
    ):
        raise ValueError(f"{column} must form a nonempty rectangular numeric matrix")
    try:
        matrix = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{column} must form a rectangular numeric matrix") from error
    if not np.isfinite(matrix).all():
        raise ValueError(f"{column} must contain only finite values")
    return matrix


def load_sample(
    path: Path, sample_size: int, cluster_on: str = "2d"
) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if cluster_on not in ("2d", "full"):
        raise ValueError("--cluster-on must be '2d' or 'full'")

    columns = ["embedding_2d", "embedding"] if cluster_on == "full" else ["embedding_2d"]
    data = pl.scan_parquet(path).select(columns).collect()
    if data.height == 0:
        raise ValueError("input must contain at least one row")
    sample = (
        data
        if sample_size >= data.height
        else data.sample(n=sample_size, seed=SAMPLE_SEED, shuffle=True)
    )
    visualization = _matrix(
        sample.get_column("embedding_2d").to_list(), sample.height, "embedding_2d"
    )
    if visualization.shape[1] != 2:
        raise ValueError("embedding_2d must have exactly two dimensions")
    source_column = "embedding_2d" if cluster_on == "2d" else "embedding"
    clustering = (
        visualization
        if source_column == "embedding_2d"
        else _matrix(sample.get_column("embedding").to_list(), sample.height, "embedding")
    )
    return clustering, visualization


def build_figure(
    visualization: np.ndarray,
    reference_labels: np.ndarray,
    local_labels: np.ndarray,
) -> go.Figure:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("fast_hdbscan (reference)", "fast_umap (local)"),
    )
    palette = plotly.colors.qualitative.Plotly
    for column, (estimator, labels) in enumerate(
        (("reference", reference_labels), ("local", local_labels)), start=1
    ):
        for cluster in sorted(set(labels.tolist())):
            noise = cluster == -1
            mask = labels == cluster
            name = f"{estimator}: noise" if noise else f"{estimator}: cluster {cluster}"
            color = "#808080" if noise else palette[int(cluster) % len(palette)]
            figure.add_trace(
                go.Scattergl(
                    x=visualization[mask, 0],
                    y=visualization[mask, 1],
                    mode="markers",
                    name=name,
                    legendgroup=f"{estimator}-{cluster}",
                    marker={"color": color, "size": 4},
                ),
                row=1,
                col=column,
            )

    figure.update_xaxes(constrain="domain")
    figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    figure.update_layout(
        title="HDBSCAN comparison on shared 2D embedding",
        legend={
            "groupclick": "togglegroup",
            "title": {"text": "Estimator-specific clusters"},
        },
    )
    return figure


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.min_cluster_size < 2:
        raise ValueError("--min-cluster-size must be at least 2")
    if args.min_samples is not None and args.min_samples <= 0:
        raise ValueError("--min-samples must be positive when supplied")

    X, visualization = load_sample(args.input, args.sample_size, args.cluster_on)
    from fast_hdbscan import HDBSCAN as ReferenceHDBSCAN

    from fast_umap import HDBSCAN as LocalHDBSCAN

    parameters = {
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "cluster_selection_method": args.cluster_selection_method,
        "allow_single_cluster": args.allow_single_cluster,
    }
    print("starting fast_hdbscan (reference)")
    started = perf_counter()
    reference_labels = ReferenceHDBSCAN(**parameters).fit(X).labels_
    reference_seconds = perf_counter() - started

    print("starting fast_umap (local)")
    started = perf_counter()
    local_labels = LocalHDBSCAN(**parameters, backend=args.device).fit(X).labels_
    local_seconds = perf_counter() - started

    figure = build_figure(visualization, reference_labels, local_labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, full_html=True)
    for name, seconds, labels in (
        ("fast_hdbscan (reference)", reference_seconds, reference_labels),
        ("fast_umap (local)", local_seconds, local_labels),
    ):
        clusters = set(labels.tolist()) - {-1}
        noise_count = int(np.count_nonzero(labels == -1))
        print(f"{name} fit: {seconds:.2f}s; clusters: {len(clusters)}; noise: {noise_count}")
    print(f"Saved standalone HTML to {args.output.resolve()}")


if __name__ == "__main__":
    main()
