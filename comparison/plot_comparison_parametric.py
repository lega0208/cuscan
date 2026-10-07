import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import plotly.colors
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "feedback_clustered.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "umap_comparison.html"
SAMPLE_SEED = 42
N_NEIGHBORS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare umap-learn and fast_umap on one reproducible data sample."
    )
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cuda", "cpu", "wgpu"), default="cuda")
    parser.add_argument("--n-epochs", type=int)
    return parser.parse_args()


def load_sample(path: Path, sample_size: int) -> tuple[np.ndarray, list[str | None]]:
    if sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    data = pl.scan_parquet(path).select("topic_global", "embedding").collect()
    sample = data.sample(n=min(sample_size, data.height), seed=SAMPLE_SEED, shuffle=True)
    if sample.height <= N_NEIGHBORS:
        raise ValueError(
            f"Need at least {N_NEIGHBORS + 1} sampled rows for n_neighbors={N_NEIGHBORS}; "
            f"found {sample.height}"
        )

    labels = sample.get_column("topic_global").cast(pl.String).to_list()
    try:
        embeddings = np.asarray(sample.get_column("embedding").to_list(), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError("embeddings must form a rectangular numeric matrix") from error
    if embeddings.ndim != 2 or embeddings.shape[0] != sample.height:
        raise ValueError("embeddings must form a rectangular 2-D matrix with one row per sample")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must contain only finite values")
    return embeddings, labels


def validate_embedding(embedding: np.ndarray, expected_rows: int, method: str) -> None:
    if embedding.ndim != 2 or embedding.shape != (expected_rows, 2):
        raise ValueError(
            f"{method} returned an invalid embedding shape {embedding.shape}; "
            f"expected ({expected_rows}, 2)"
        )
    if not np.isfinite(embedding).all():
        raise ValueError(f"{method} returned non-finite embedding values")


def main() -> None:
    args = parse_args()
    X, labels = load_sample(args.input, args.sample_size)

    config = {
        "metric": "euclidean",
        "n_neighbors": N_NEIGHBORS,
        "n_components": 2,
        "random_state": None,
        "min_dist": 0.0,
        "verbose": True,
    }
    if args.n_epochs is not None:
        if args.n_epochs <= 0:
            raise ValueError("--n-epochs must be positive")
        config["n_epochs"] = args.n_epochs

    import umap

    from fast_umap import ParametricUMAP as FastUMAP

    umap_config = {**config, "unique": True}
    fast_config = {**config, "device": args.device, "n_epochs": 2500}

    print("starting umap-learn")
    started = perf_counter()
    umap_embedding = umap.UMAP(**umap_config).fit_transform(X)
    umap_seconds = perf_counter() - started
    validate_embedding(umap_embedding, X.shape[0], "umap-learn")

    print("starting fast_umap with n_epochs=2500")
    started = perf_counter()
    fast_embedding = FastUMAP(**fast_config).fit_transform(X)
    fast_seconds = perf_counter() - started
    validate_embedding(fast_embedding, X.shape[0], "fast_umap")

    categories = sorted(set(labels), key=lambda label: (label is not None, label or ""))
    palette = plotly.colors.qualitative.Plotly
    category_colors = {
        label: palette[index % len(palette)] for index, label in enumerate(categories)
    }
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("umap-learn", "fast_umap"),
    )
    for index, label in enumerate(categories):
        display_name = "(null)" if label is None else label
        mask = np.fromiter((value == label for value in labels), dtype=bool, count=len(labels))
        color = category_colors[label]
        for column, embedding in enumerate((umap_embedding, fast_embedding), start=1):
            figure.add_trace(
                go.Scattergl(
                    x=embedding[mask, 0],
                    y=embedding[mask, 1],
                    mode="markers",
                    name=display_name,
                    legendgroup=f"topic-{index}",
                    showlegend=column == 1,
                    marker={"color": color, "size": 4},
                ),
                row=1,
                col=column,
            )

    figure.update_xaxes(constrain="domain")
    figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)
    figure.update_layout(
        title="UMAP embedding comparison",
        legend={"groupclick": "togglegroup", "title": {"text": "topic_global"}},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(args.output, include_plotlyjs=True, full_html=True)

    print(f"umap-learn fit: {umap_seconds:.2f}s")
    print(f"fast_umap fit: {fast_seconds:.2f}s")
    print(f"Saved standalone HTML to {args.output.resolve()}")


if __name__ == "__main__":
    main()
