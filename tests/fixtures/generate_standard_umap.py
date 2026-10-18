#!/usr/bin/env python3
"""Generate frozen numerical, optimizer, and quality references from umap-learn 0.5.12."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from sklearn.manifold import trustworthiness
from umap import layouts
from umap.umap_ import compute_membership_strengths, make_epochs_per_sample, smooth_knn_dist

REFERENCE_VERSION = "0.5.12"
HERE = Path(__file__).resolve().parent
REFERENCE_JSON = HERE / "standard_umap_reference.json"
QUALITY_NPZ = HERE / "standard_umap_quality.npz"
SEEDS = [11, 29, 47, 71, 101]
N_NEIGHBORS = 15
N_COMPONENTS = 2
TRUST_K = 15
CAMPAIGN_SEEDS = [11, 29, 47, 71, 101]
CAMPAIGN_N_NEIGHBORS = 25
CAMPAIGN_TRUST_K = 15

# Explicit fixed rows isolate the reference graph math from neighbor search.
GRAPH_CASES = [
    {
        "name": "fractional_distances",
        "n": 4,
        "k": 4,
        "indices": [0, 1, 2, 3, 1, 0, 3, 2, 2, 3, 0, 1, 3, 2, 1, 0],
        "distances": [0, 0.15, 0.8, 1.7, 0, 0.15, 0.55, 1.2, 0, 0.35, 0.8, 1.5, 0, 0.55, 1.2, 1.7],
        "local_connectivity": 1.5,
        "bandwidth": 1.25,
        "set_op_mix_ratio": 0.5,
    },
    {
        "name": "duplicates_and_zero_distances",
        "n": 4,
        "k": 4,
        "indices": [0, 1, 2, 3, 1, 0, 2, 3, 2, 0, 1, 3, 3, 0, 1, 2],
        "distances": [0, 0, 0, 0.5, 0, 0, 0.2, 0.8, 0, 0, 0.2, 0.7, 0, 0.5, 0.7, 0.8],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 1.0,
    },
    {
        "name": "ties_and_disconnected_minus_one",
        "n": 4,
        "k": 3,
        "indices": [0, 1, -1, 1, 0, -1, 2, 3, -1, 3, 2, -1],
        "distances": [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        "local_connectivity": 1.0,
        "bandwidth": 1.0,
        "set_op_mix_ratio": 0.0,
    },
]


def source_graph(case: dict[str, object]) -> dict[str, object]:
    n, k = int(case["n"]), int(case["k"])
    indices = np.asarray(case["indices"], dtype=np.int32).reshape(n, k)
    distances = np.asarray(case["distances"], dtype=np.float32).reshape(n, k)
    sigmas, rhos = smooth_knn_dist(
        distances,
        float(k),
        local_connectivity=float(case["local_connectivity"]),
        bandwidth=float(case["bandwidth"]),
    )
    rows, cols, values, _ = compute_membership_strengths(
        indices, distances, sigmas, rhos, return_dists=False, bipartite=False
    )
    directed = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
    directed.eliminate_zeros()
    transpose = directed.transpose().tocsr()
    product = directed.multiply(transpose)
    mix = float(case["set_op_mix_ratio"])
    symmetric = mix * (directed + transpose - product) + (1.0 - mix) * product
    symmetric.eliminate_zeros()
    out = dict(case)
    out.update(
        rho=rhos.astype(np.float32).tolist(),
        sigma=sigmas.astype(np.float32).tolist(),
        directed=directed.toarray().astype(np.float32).ravel().tolist(),
        symmetric=symmetric.toarray().astype(np.float32).ravel().tolist(),
    )
    return out


def optimizer_probe(
    name: str,
    *,
    head: list[list[float]],
    tail: list[list[float]],
    edges: list[list[int]],
    weights: list[float],
    epochs: int,
    negative_rate: float,
    move_other: bool,
    seed: int,
    alias_embeddings: bool = False,
) -> dict[str, object]:
    """Run installed source kernel; wrappers record ordered prefix-event and epoch states."""
    edge_array = np.asarray(edges, dtype=np.int64)
    interval_array = np.asarray(
        make_epochs_per_sample(np.asarray(weights), epochs), dtype=np.float32
    )
    source_epoch = layouts._get_optimize_layout_euclidean_single_epoch_fn

    def execute(
        event_edges: np.ndarray,
    ) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
        head_array = np.asarray(head, dtype=np.float32).copy()
        tail_array = head_array if alias_embeddings else np.asarray(tail, dtype=np.float32).copy()
        rng_state = np.asarray([seed, seed + 1, seed + 2], dtype=np.int64)
        records: list[dict[str, object]] = []

        def capture_epoch(parallel: bool = False):
            epoch_fn = source_epoch(parallel)

            def run(*args):
                epoch_fn(*args)
                records.append(
                    {
                        "epoch": int(args[16]),
                        "head_embedding": head_array.tolist(),
                        "tail_embedding": tail_array.tolist(),
                        "epoch_of_next_sample": args[15].tolist(),
                        "epoch_of_next_negative_sample": args[14].tolist(),
                        "rng_state_per_sample": args[8].tolist(),
                        "alpha_used": float(args[12]),
                        # Source assigns alpha immediately after the epoch kernel.
                        "alpha_after_epoch": 0.8 * (1.0 - (float(args[16]) / float(epochs))),
                    }
                )

            return run

        with patch.object(layouts, "_get_optimize_layout_euclidean_single_epoch_fn", capture_epoch):
            layouts.optimize_layout_euclidean(
                head_array,
                tail_array,
                event_edges[:, 0],
                event_edges[:, 1],
                epochs,
                tail_array.shape[0],
                interval_array[: len(event_edges)],
                1.576943460405378,
                0.8950608781227859,
                rng_state,
                gamma=1.7,
                initial_alpha=0.8,
                negative_sample_rate=negative_rate,
                parallel=False,
                verbose=False,
                tqdm_kwds={"disable": True},
                move_other=move_other,
            )
        return records, head_array, tail_array, rng_state

    epoch_records, head_array, tail_array, rng_state = execute(edge_array)
    event_prefixes = []
    for edge_index in range(len(edge_array)):
        prefix_records, _, _, _ = execute(edge_array[: edge_index + 1])
        event_prefixes.append({"through_edge_index": edge_index, "epochs": prefix_records})
    return {
        "name": name,
        "head_initial": head,
        "tail_initial": tail,
        "edges_head_tail": edges,
        "positive_weights": weights,
        "epochs_per_sample": interval_array.tolist(),
        "negative_sample_rate": negative_rate,
        "n_epochs": epochs,
        "n_vertices": int(tail_array.shape[0]),
        "rng_state": rng_state.tolist(),
        "gamma": 1.7,
        "initial_alpha": 0.8,
        "move_other": move_other,
        "head_tail_share_storage": alias_embeddings,
        "epochs": epoch_records,
        "ordered_event_prefixes": event_prefixes,
        "final_head": head_array.tolist(),
        "final_tail": tail_array.tolist(),
    }


def make_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.RandomState(20261006)
    cases: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Four separated Gaussian components.
    centers = np.asarray([[-5, -5], [-5, 5], [5, -5], [5, 5]], dtype=np.float64)
    labels = np.arange(96) % 4
    train = centers[labels] + rng.normal(scale=0.32, size=(96, 2))
    test_labels = np.arange(40) % 4
    held = centers[test_labels] + rng.normal(scale=0.32, size=(40, 2))
    cases["clusters"] = train, held

    # Curved manifold with held-out points drawn from the same parameterized surface.
    def manifold(t: np.ndarray, noise_rng: np.random.RandomState) -> np.ndarray:
        radius = 1.0 + 0.18 * np.sin(5 * t)
        return np.column_stack(
            (radius * np.cos(t), radius * np.sin(t), 0.45 * t)
        ) + noise_rng.normal(scale=0.025, size=(len(t), 3))

    cases["manifold"] = (
        manifold(np.linspace(0, 4 * np.pi, 96, endpoint=False), rng),
        manifold(np.linspace(0.13, 4 * np.pi + 0.13, 40, endpoint=False), rng),
    )

    # Repeated exact points among otherwise distinct observations.
    unique = rng.normal(size=(64, 5))
    train = np.concatenate((unique, unique[:16]), axis=0)
    held_base = rng.normal(size=(40, 5))
    held = np.concatenate((held_base[:20], unique[:20]), axis=0)
    cases["duplicates"] = train, held

    # Integer lattice produces abundant equal-distance neighbor ties.
    lattice = np.asarray(
        [(x, y, (x + y) % 3) for x in range(8) for y in range(8)], dtype=np.float64
    )
    cases["ties"] = (
        lattice,
        np.asarray(
            [(x + 0.5, y + 0.5, (x + y) % 3 + 0.2) for x in range(5) for y in range(8)],
            dtype=np.float64,
        ),
    )

    # Explicitly disconnected components, separated far beyond within-component scales.
    left = rng.normal(loc=0.0, scale=0.2, size=(48, 4))
    right = rng.normal(loc=40.0, scale=0.2, size=(48, 4))
    held_left = rng.normal(loc=0.0, scale=0.2, size=(20, 4))
    held_right = rng.normal(loc=40.0, scale=0.2, size=(20, 4))
    cases["disconnected"] = np.concatenate((left, right)), np.concatenate((held_left, held_right))
    return cases


def run_quality_cases() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    import umap

    datasets = make_datasets()
    arrays: dict[str, np.ndarray] = {}
    scores: dict[str, object] = {}
    for case, (train, heldout) in datasets.items():
        if len(train) <= TRUST_K or len(heldout) <= TRUST_K:
            raise AssertionError(f"{case} is too small for trustworthiness@{TRUST_K}")
        arrays[f"{case}__train"] = train.astype(np.float64)
        arrays[f"{case}__heldout"] = heldout.astype(np.float64)
        case_scores = []
        for seed in SEEDS:
            reducer = umap.UMAP(
                n_neighbors=N_NEIGHBORS,
                n_components=N_COMPONENTS,
                min_dist=0.1,
                spread=1.0,
                local_connectivity=1.0,
                set_op_mix_ratio=1.0,
                n_epochs=200,
                learning_rate=1.0,
                negative_sample_rate=5,
                repulsion_strength=1.0,
                init="spectral",
                random_state=seed,
                transform_seed=seed,
                metric="euclidean",
                low_memory=True,
                n_jobs=1,
                verbose=False,
            )
            reducer.fit(train)
            fit_embedding = reducer.embedding_.astype(np.float32)
            transformed = reducer.transform(heldout).astype(np.float32)
            arrays[f"{case}__seed{seed}__fit"] = fit_embedding
            arrays[f"{case}__seed{seed}__transform"] = transformed
            case_scores.append(
                {
                    "seed": seed,
                    "fit_trustworthiness_at_15": float(
                        trustworthiness(train, fit_embedding, n_neighbors=TRUST_K)
                    ),
                    "transform_trustworthiness_at_15": float(
                        trustworthiness(heldout, transformed, n_neighbors=TRUST_K)
                    ),
                }
            )
        scores[case] = case_scores
    return scores, arrays


def exact_trustworthiness_bounded(original: np.ndarray, embedded: np.ndarray, k: int) -> float:
    """Exact sklearn-style trustworthiness using O(n + n*k) working memory."""
    n = len(original)
    if embedded.shape[0] != n or not 0 < k < n / 2:
        raise ValueError("trustworthiness requires equal rows and 0 < k < n/2")
    penalty = 0
    original64 = np.asarray(original, dtype=np.float64)
    embedded64 = np.asarray(embedded, dtype=np.float64)
    for row in range(n):
        original_dist = np.sum((original64 - original64[row]) ** 2, axis=1)
        embedded_dist = np.sum((embedded64 - embedded64[row]) ** 2, axis=1)
        original_dist[row] = np.inf
        embedded_dist[row] = np.inf
        # Stable row-index tie breaks are used consistently by Rust candidate scoring.
        original_order = np.lexsort((np.arange(n), original_dist))
        embedded_order = np.lexsort((np.arange(n), embedded_dist))[:k]
        ranks = np.empty(n, dtype=np.int32)
        ranks[original_order] = np.arange(1, n + 1, dtype=np.int32)
        penalty += int(np.maximum(ranks[embedded_order] - k, 0).sum())
    normalizer = n * k * (2 * n - 3 * k - 1)
    return 1.0 - 2.0 * penalty / normalizer


def run_campaign(input_path: Path, reference_path: Path, quality_path: Path) -> None:
    import umap

    actual_version = version("umap-learn")
    if actual_version != REFERENCE_VERSION:
        raise RuntimeError(f"expected umap-learn {REFERENCE_VERSION}, found {actual_version}")
    expected_values = 16384 * 16
    raw = input_path.read_bytes()
    if len(raw) != expected_values * 4:
        raise ValueError(f"expected {expected_values * 4} input bytes, found {len(raw)}")
    train = np.frombuffer(raw, dtype="<f4").reshape(16384, 16).copy()
    if not np.isfinite(train).all():
        raise ValueError("campaign training data contains non-finite values")

    arrays: dict[str, np.ndarray] = {"campaign__train": train}
    scores: list[dict[str, object]] = []
    for seed in CAMPAIGN_SEEDS:
        reducer = umap.UMAP(
            n_neighbors=CAMPAIGN_N_NEIGHBORS,
            n_components=N_COMPONENTS,
            min_dist=0.1,
            spread=1.0,
            local_connectivity=1.0,
            set_op_mix_ratio=1.0,
            n_epochs=200,
            learning_rate=1.0,
            negative_sample_rate=5,
            repulsion_strength=1.0,
            init="spectral",
            random_state=seed,
            transform_seed=seed,
            metric="euclidean",
            low_memory=True,
            n_jobs=1,
            verbose=False,
        )
        reducer.fit(train)
        embedding = reducer.embedding_.astype(np.float32)
        if embedding.shape != (len(train), N_COMPONENTS) or not np.isfinite(embedding).all():
            raise ValueError(f"invalid campaign reference embedding for seed {seed}")
        arrays[f"campaign__seed{seed}__fit"] = embedding
        scores.append(
            {
                "seed": seed,
                "fit_trustworthiness_at_15": exact_trustworthiness_bounded(
                    train, embedding, CAMPAIGN_TRUST_K
                ),
            }
        )

    result = {
        "schema": "fast-umap-ann-campaign-quality-v1",
        "reference": {
            "distribution": "umap-learn",
            "version": actual_version,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": version("scikit-learn"),
            "numba": version("numba"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "data": {
            "input_file": str(input_path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dtype": "little-endian float32",
            "shape": [16384, 16],
        },
        "controls": {
            "n_neighbors": CAMPAIGN_N_NEIGHBORS,
            "n_components": N_COMPONENTS,
            "seeds": CAMPAIGN_SEEDS,
            "trustworthiness_k": CAMPAIGN_TRUST_K,
            "fit": {
                "min_dist": 0.1,
                "spread": 1.0,
                "local_connectivity": 1.0,
                "set_op_mix_ratio": 1.0,
                "n_epochs": 200,
                "learning_rate": 1.0,
                "negative_sample_rate": 5,
                "repulsion_strength": 1.0,
                "init": "spectral",
                "metric": "euclidean",
                "n_jobs": 1,
                "low_memory": True,
            },
            "truth": (
                "Exact Euclidean neighbors, self excluded by row identity; "
                "ties at kth distance are inclusive."
            ),
            "trustworthiness": (
                "Exact Euclidean ranks, self excluded by row identity, ties ordered by row index; "
                "row-wise bounded-memory computation."
            ),
            "freeze": "Freeze campaign reference before evaluating kernel candidates.",
        },
        "quality": {"cases": ["campaign"], "scores": {"campaign": scores}},
    }
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    reference_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(quality_path, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign",
        action="store_true",
        help="generate only the isolated large campaign reference",
    )
    parser.add_argument("--input", type=Path, help="campaign raw little-endian float32 input")
    parser.add_argument("--output-reference", type=Path, help="campaign reference JSON output")
    parser.add_argument("--output-quality", type=Path, help="campaign quality NPZ output")
    args = parser.parse_args()
    if args.campaign:
        if not all((args.input, args.output_reference, args.output_quality)):
            parser.error("--campaign requires --input, --output-reference, and --output-quality")
        outputs = {args.output_reference.resolve(), args.output_quality.resolve()}
        protected = {REFERENCE_JSON.resolve(), QUALITY_NPZ.resolve()}
        if len(outputs) != 2 or outputs & protected:
            parser.error(
                "campaign outputs must be distinct and may not overwrite standard frozen fixtures"
            )
        run_campaign(args.input, args.output_reference, args.output_quality)
        return
    if any((args.input, args.output_reference, args.output_quality)):
        parser.error("campaign input/output controls require --campaign")
    actual_version = version("umap-learn")
    if actual_version != REFERENCE_VERSION:
        raise RuntimeError(f"expected umap-learn {REFERENCE_VERSION}, found {actual_version}")
    scores, quality_arrays = run_quality_cases()
    graph_cases = [source_graph(case) for case in GRAPH_CASES]
    optimizer_cases = [
        optimizer_probe(
            "fit_unequal_weights_live_events",
            head=[[0.0, 0.0], [1.2, -0.4], [-0.8, 0.7]],
            tail=[[0.0, 0.0], [1.2, -0.4], [-0.8, 0.7]],
            edges=[[0, 1], [0, 2]],
            weights=[2.5, 1.0],
            epochs=7,
            negative_rate=2.5,
            move_other=True,
            seed=41,
            alias_embeddings=True,
        ),
        optimizer_probe(
            "bipartite_equal_raw_indices_positive_distance",
            head=[[0.4, 0.1], [0.0, 3.0]],
            tail=[[0.0, 0.0], [1.1, 0.1], [2.0, 1.0], [3.0, 3.0]],
            edges=[[1, 1]],
            weights=[1.0],
            epochs=4,
            negative_rate=1.0,
            move_other=False,
            seed=73,
        ),
        optimizer_probe(
            "coincident_distinct_zero_distance",
            head=[[3.0, 3.0], [0.25, -0.25]],
            tail=[[0.25, -0.25], [0.25, -0.25]],
            edges=[[1, 1]],
            weights=[1.0],
            epochs=3,
            negative_rate=1.0,
            move_other=False,
            seed=97,
        ),
        optimizer_probe(
            "same_index_zero_distance_skip",
            head=[[0.0, 0.0]],
            tail=[[0.0, 0.0]],
            edges=[[0, 0]],
            weights=[1.0],
            epochs=3,
            negative_rate=1.0,
            move_other=False,
            seed=101,
        ),
        optimizer_probe(
            "clipped_gradient_and_frozen_transform_tail",
            head=[[0.0, 0.0]],
            tail=[[0.005, 0.0], [0.005, 0.005], [0.0, 0.0]],
            edges=[[0, 2]],
            weights=[1.0],
            epochs=3,
            negative_rate=1.0,
            move_other=False,
            seed=19,
        ),
    ]
    result = {
        "reference": {
            "distribution": "umap-learn",
            "version": actual_version,
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": version("scikit-learn"),
            "numba": version("numba"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "controls": {
            "n_neighbors": N_NEIGHBORS,
            "n_components": N_COMPONENTS,
            "seeds": SEEDS,
            "trustworthiness_k": TRUST_K,
            "graph_helper_tolerance": "abs(actual-expected) <= 1e-5 + 1e-5*abs(expected)",
            "fit": {
                "min_dist": 0.1,
                "spread": 1.0,
                "local_connectivity": 1.0,
                "set_op_mix_ratio": 1.0,
                "n_epochs": 200,
                "learning_rate": 1.0,
                "negative_sample_rate": 5,
                "repulsion_strength": 1.0,
                "init": "spectral",
                "metric": "euclidean",
                "n_jobs": 1,
            },
            "transform": {
                "transform_seed": "equal to case seed",
                "reference_call": "fitted_reference.transform(heldout)",
                "n_epochs": 66,
                "n_epochs_rule": (
                    "int(fit_n_epochs // 3) = int(200 // 3), from umap.umap_.UMAP.transform"
                ),
                "init": "weighted",
                "transform_queue_size": 4.0,
            },
        },
        "graph_cases": graph_cases,
        "source_basis": {
            "graph": (
                "umap.umap_.smooth_knn_dist and compute_membership_strengths; "
                "fuzzy set operation follows UMAP's sparse union/intersection expression."
            ),
            "positive_schedule": "umap.umap_.make_epochs_per_sample",
            "optimizer": (
                "umap.layouts.optimize_layout_euclidean with the installed "
                "_get_optimize_layout_euclidean_single_epoch_fn wrapped to record states "
                "after each epoch and ordered event prefix."
            ),
        },
        "optimizer_probes": optimizer_cases,
        "quality": {
            "cases": list(scores),
            "splits": {
                case: {
                    "train_rows": int(data[0].shape[0]),
                    "heldout_rows": int(data[1].shape[0]),
                    "features": int(data[0].shape[1]),
                }
                for case, data in make_datasets().items()
            },
            "scoring": {
                "trustworthiness": (
                    "sklearn.manifold.trustworthiness(X, embedding, n_neighbors=15); "
                    "report fit and held-out transform independently. For each surface, "
                    "compute the arithmetic mean of per-case/per-seed trustworthiness "
                    "scores for implementation and reference, then gate on "
                    "abs(implementation_mean-reference_mean) <= 0.025. This is intentionally "
                    "not the stricter mean of per-run absolute gaps."
                ),
                "recall": (
                    "Fit uses k=14 (= n_neighbors - 1; fit's kNN includes self), held-out "
                    "transform query uses k=15 (= n_neighbors; bipartite search has no self "
                    "slot). For each row, exact input-space neighbors are sorted by "
                    "(Euclidean distance, row index). Each unique returned neighbor earns "
                    "1/k if its distance is <= the exact kth distance (inclusive ties); "
                    "denominator remains k even when output is short. Average over all rows "
                    "across cases and seeds with equal row weight. Fit excludes self; query "
                    "searches training rows. No tie-breaking requirement within the "
                    "kth-distance boundary."
                ),
                "freeze": (
                    "Datasets, splits, controls, references, and score definitions are fixed "
                    "before implementation scoring; do not retune after observing results."
                ),
            },
            "scores": scores,
        },
    }
    REFERENCE_JSON.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    np.savez_compressed(QUALITY_NPZ, **quality_arrays)


if __name__ == "__main__":
    main()
