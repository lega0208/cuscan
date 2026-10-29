from datetime import datetime
from pathlib import Path

import polars as pl

from cuscan import UMAP

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "data.parquet"
DEFAULT_OUTPUT = SCRIPT_DIR / "embeddings.parquet"
SAMPLE_SEED = 42
N_NEIGHBORS = 30

config = {
    "n_neighbors": N_NEIGHBORS,
    "n_components": 2,
    "random_state": None,
    "min_dist": 0.0,
    "unique": True,
}

config_16d = {
    **config,
    "n_components": 16,
}

df = pl.read_parquet(DEFAULT_INPUT)

embeddings = df["embedding"].to_numpy()

start_2d = datetime.now()
print("Computing 2D embeddings...")
embeddings_2d = UMAP(**config).fit_transform(embeddings)
end_2d = datetime.now()
print(f"2D embeddings computed in {end_2d - start_2d}")

start_16d = datetime.now()
print("Computing 16D embeddings...")
embeddings_16d = UMAP(**config_16d).fit_transform(embeddings)
end_16d = datetime.now()
print(f"16D embeddings computed in {end_16d - start_16d}")

print("Saving embeddings...")
df.with_columns(
    embeddings=pl.Series(embeddings),
    embedding_2d=pl.Series(embeddings_2d),
    embedding_16d=pl.Series(embeddings_16d),
).write_parquet(DEFAULT_OUTPUT)
print("Embeddings saved.")
