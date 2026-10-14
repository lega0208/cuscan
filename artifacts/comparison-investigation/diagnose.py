"""Collect investigation evidence; no estimator source changes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument(
    "method", choices=["python-unique", "python-all", "fast-spectral", "fast-random", "fast-unique"]
)
parser.add_argument("rows", type=int)
args = parser.parse_args()
X = np.fromfile(ROOT / f"sample-{args.rows}.f32le", dtype=np.float32).reshape(args.rows, 128)
stages = {}
if args.method.startswith("python"):
    import umap
    import umap.umap_ as impl

    for name in (
        "nearest_neighbors",
        "fuzzy_simplicial_set",
        "spectral_layout",
        "simplicial_set_embedding",
    ):
        original = getattr(impl, name)

        def wrapped(*a, _original=original, _name=name, **kw):
            start = time.perf_counter()
            try:
                return _original(*a, **kw)
            finally:
                stages[_name] = stages.get(_name, 0) + time.perf_counter() - start

        setattr(impl, name, wrapped)
    estimator = umap.UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        random_state=None,
        unique=args.method == "python-unique",
    )
else:
    from fast_umap import UMAP

    estimator = UMAP(
        n_neighbors=15,
        n_components=2,
        min_dist=0.0,
        random_state=None,
        backend="cuda",
        init="random" if args.method == "fast-random" else "spectral",
    )
if args.method == "fast-unique":
    fit_X, inverse = np.unique(X, axis=0, return_inverse=True)
else:
    fit_X, inverse = X, None
start = time.perf_counter()
Y = estimator.fit_transform(fit_X)
elapsed = time.perf_counter() - start
if inverse is not None:
    Y = Y[inverse]
assert Y.shape == (len(X), 2) and np.isfinite(Y).all()
np.save(ROOT / f"{args.method}-{args.rows}.npy", Y)
actual_fit_rows = estimator.graph_.shape[0] if args.method.startswith("python") else len(fit_X)
result = {
    "method": args.method,
    "rows": len(X),
    "fit_rows": actual_fit_rows,
    "fit_seconds": elapsed,
    "stages_seconds": stages,
    "random_state": None,
    "min_dist": 0.0,
    "n_neighbors": 15,
    "n_epochs": None,
    "finite": True,
}
if args.method.startswith("python"):
    from scipy.sparse.csgraph import connected_components

    count, labels = connected_components(estimator.graph_)
    sizes = np.bincount(labels)
    result["graph_components"] = int(count)
    result["largest_graph_component"] = int(sizes.max())
    result["graph_nnz"] = int(estimator.graph_.nnz)
    result["graph_singletons"] = int((sizes == 1).sum())
(ROOT / f"{args.method}-{args.rows}.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
