"""Tie-aware ANN checks and scale-independent embedding diagnostics."""

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
output = {}
with threadpool_limits(limits=8):
    for n in (15000, 50000):
        X = np.fromfile(ROOT / f"sample-{n}.f32le", dtype=np.float32).reshape(n, 128)
        unique, first, inverse, counts = np.unique(
            X, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        graph = np.fromfile(ROOT / f"profile-{n}/graph_0.i32le", dtype=np.int32).reshape(n, 14)
        assert ((graph >= 0) & (graph < n)).all()
        G = csr_matrix(
            (np.ones(graph.size), (np.repeat(np.arange(n), 14), graph.ravel())), shape=(n, n)
        )
        components, component_labels = connected_components(G, directed=False)
        sizes = np.sort(np.bincount(component_labels))[::-1]
        rng = np.random.default_rng(20261014)
        query = rng.choice(n, 1000, replace=False)
        distances, indices = (
            NearestNeighbors(n_neighbors=15, algorithm="brute", metric="euclidean", n_jobs=8)
            .fit(X)
            .kneighbors(X[query])
        )
        threshold = distances[:, 14]
        actual = np.linalg.norm(X[graph[query]] - X[query, None], axis=2)
        # Count all tied neighbors at the kth distance as correct. IDs are arbitrary at ties.
        recall = (actual <= threshold[:, None] + 1e-5).mean(axis=1)
        repeated = counts[inverse[query]] > 1
        data = {
            "graph_components": int(components),
            "largest_component": int(sizes[0]),
            "component_sizes": sizes.tolist(),
            "duplicate_groups_at_least_15": int((counts >= 15).sum()),
            "rows_in_groups_at_least_15": int(counts[counts >= 15].sum()),
            "ann_queries": len(query),
            "tie_aware_recall_at_14": float(recall.mean()),
            "recall_duplicate_queries": float(recall[repeated].mean()),
            "recall_nonduplicate_queries": float(recall[~repeated].mean()),
            "embeddings": {},
        }
        chosen = rng.choice(len(unique), 2000, replace=False)
        subX = unique[chosen]
        ref_indices = (
            NearestNeighbors(n_neighbors=16, algorithm="brute")
            .fit(subX)
            .kneighbors(subX, return_distance=False)[:, 1:]
        )
        for method in (
            "python-unique",
            "fast-spectral",
            "fast-random",
            "fast-unique",
            "python-all",
        ):
            path = ROOT / f"{method}-{n}.npy"
            if not path.exists():
                continue
            Y = np.load(path)
            subY = Y[first[chosen]]
            tw = trustworthiness(subX, subY, n_neighbors=15)
            emb_indices = cKDTree(subY).query(subY, k=16)[1][:, 1:]
            overlap = np.mean(
                [len(set(a) & set(b)) / 15 for a, b in zip(ref_indices, emb_indices, strict=True)]
            )
            radius = np.linalg.norm(Y - np.median(Y, axis=0), axis=1)
            quantiles = np.quantile(radius, [0.5, 0.9, 0.95, 0.99, 1.0])
            # Duplicate displacements are descriptive, not a quality score.
            duplicate_distances = np.linalg.norm(Y - Y[first[inverse]], axis=1)
            repeated_rows = counts[inverse] > 1
            data["embeddings"][method] = {
                "trustworthiness_2000_unique_at_15": float(tw),
                "neighbor_overlap_2000_unique_at_15": float(overlap),
                "radius_quantiles_50_90_95_99_100": quantiles.tolist(),
                "radius_p95_over_max": float(quantiles[2] / quantiles[4]),
                "duplicate_displacement_p50_p95_max": np.quantile(
                    duplicate_distances[repeated_rows], [0.5, 0.95, 1]
                ).tolist(),
            }
        output[n] = data
(ROOT / "quality-metrics.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
