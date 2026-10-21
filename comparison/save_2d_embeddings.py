from pathlib import Path

import polars as pl

from fast_umap import UMAP

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "feedback_clustered.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "embeddings_2d.parquet"
SAMPLE_SEED = 42
N_NEIGHBORS = 30

config = {
    "n_neighbors": N_NEIGHBORS,
    "n_components": 2,
    "random_state": None,
    "min_dist": 0.0,
    "unique": True,
}

df = pl.read_parquet(DEFAULT_INPUT)

print("Computing 2D embeddings...")
embeddings_2d = UMAP(**config).fit_transform(df["embedding"].to_numpy())

print("Saving 2D embeddings...")
df.with_columns(embedding_2d=pl.Series(embeddings_2d)).write_parquet(DEFAULT_OUTPUT)
print("2D embeddings saved.")
