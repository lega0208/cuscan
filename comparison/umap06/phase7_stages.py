"""Fixed-fit query attribution for Phase 7."""

import argparse
import json
from pathlib import Path

import numpy as np

from .datasets import load_fixture
from .phase5_probes import sha256
from .phase6_campaign import dump
from .quality import content_sha256, exact_neighbors, query_recall, score_transforms
from .spec import CONFIGS


def models(snapshot, backend, train, config, components, seed, *, q7e=False):
    if q7e:
        raise RuntimeError(
            "Q7e replay requires the archived Phase 7 reverse-graph source "
            "and a matching tracing build"
        )
    from cuscan import UMAP, _cuscan

    identity = {
        "backend": backend,
        "optimizer": config.optimizer,
        "init": config.init,
        "components": components,
        "seed": seed,
    }
    if not snapshot.exists():
        model = UMAP(
            backend=backend,
            n_components=components,
            n_neighbors=15,
            min_dist=0.1,
            init=config.init,
            optimizer=config.optimizer,
            random_state=seed,
            transform_seed=seed,
        )
        model.fit(train)
        snapshot.mkdir(parents=True, exist_ok=False)
        model.save(snapshot / "fit.h5")
        np.save(snapshot / "fit.npy", model.embedding_, allow_pickle=False)
        _, metadata = model._model.export_state()
        dump(
            snapshot / "manifest.json",
            {
                "identity": identity,
                "model_sha256": sha256(snapshot / "fit.h5"),
                "embedding_sha256": sha256(snapshot / "fit.npy"),
                "train_sha256": content_sha256(train),
                "a": metadata["a"],
                "b": metadata["b"],
            },
        )
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["identity"] != identity
        or sha256(snapshot / "fit.h5") != manifest["model_sha256"]
        or sha256(snapshot / "fit.npy") != manifest["embedding_sha256"]
        or content_sha256(train) != manifest["train_sha256"]
    ):
        raise ValueError("Fixed-fit snapshot content has changed")
    fitted = UMAP.load(snapshot / "fit.h5", backend=backend)
    arrays, metadata = fitted._model.export_state()
    if metadata["resolved_search"] != "approximate":
        raise ValueError("Stage attribution requires an approximate-search snapshot")
    exact_arrays = {**arrays, "query_graph": None}
    exact_metadata = {
        **metadata,
        "resolved_search": "exact",
        "requested_search": "exact",
        "query_graph_width": 0,
        "transform_queue_size": 4.0,
    }
    yield "exact", _cuscan.load(exact_arrays, exact_metadata, backend), {}
    n = len(train)
    log = (max(n, 2) - 1).bit_length()
    for queue in (4, 16):
        native = _cuscan.load(arrays, {**metadata, "transform_queue_size": queue}, backend)
        yield f"queue_{queue}", native, {}
    for c, v in ((60, 60 * log), (60, 240 * log), (240, 240 * log)):
        yield f"C{c}_V{min(v, n)}", fitted._model, {"candidates": c, "visits": min(v, n)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--require-snapshot", action="store_true")
    parser.add_argument("--q7e", action="store_true")
    args = parser.parse_args()
    if args.q7e:
        parser.error(
            "Q7e replay requires the archived Phase 7 reverse-graph source "
            "and a matching tracing build"
        )
    if args.require_snapshot and not args.snapshot.is_dir():
        parser.error("predecessor snapshot is missing; refusing to refit")
    train, batches, _ = load_fixture(args.data, args.fixture)
    if args.batch is not None:
        batches = {args.batch: batches[args.batch]}
    projections = {count: {} for count in batches}
    neighbors = {count: {} for count in batches}
    for name, model, budget in models(
        args.snapshot,
        args.backend,
        train,
        CONFIGS[args.config],
        args.components,
        args.seed,
        q7e=args.q7e,
    ):
        for count, queries in batches.items():
            replay = model._replay_transform_search(queries, **budget)
            neighbors[count][name] = replay["neighbor_indices"].reshape(count, -1)
            projections[count][name] = replay["embedding"].reshape(count, args.components)
    fit = np.load(args.snapshot / "fit.npy", allow_pickle=False)
    results = []
    for count, queries in batches.items():
        exact_ids, _ = exact_neighbors(train, queries)
        scores = score_transforms(
            train, dict.fromkeys(projections[count], fit), queries, projections[count]
        )
        query_digest = content_sha256(queries)
        for name, ids in neighbors[count].items():
            recall = query_recall(train, queries, ids, exact_ids=exact_ids)
            results.append(
                {
                    "mode": name,
                    "batch": count,
                    "query_sha256": query_digest,
                    "recall": recall["recall"],
                    "scores": scores[name],
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(args.out, results)


if __name__ == "__main__":
    main()
