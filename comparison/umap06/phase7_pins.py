"""Phase 7 host stage pins and shared-input upstream replay."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from .datasets import load_fixture
from .phase5_probes import sha256
from .phase6_campaign import dump
from .quality import content_sha256, score_transforms
from .spec import CONFIGS

# These tolerances cover f32 device arithmetic, not different graph topology.
RTOL = 1e-4
ATOL = 1e-5


def compare(actual, expected, *, exact=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    result = {"actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    if actual.shape != expected.shape:
        return {**result, "matched": False, "reason": "shape mismatch"}
    if exact:
        return {**result, "matched": bool(np.array_equal(actual, expected))}
    finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {
        **result,
        "matched": finite and bool(np.allclose(actual, expected, rtol=RTOL, atol=ATOL)),
        "max_abs_error": float(difference.max()) if difference.size and finite else None,
        "rtol": RTOL,
        "atol": ATOL,
    }


def fixed_state(args, train):
    from cuscan import UMAP

    manifest = json.loads((args.snapshot / "manifest.json").read_text(encoding="utf-8"))
    identity = {
        "backend": args.backend,
        "optimizer": CONFIGS[args.config].optimizer,
        "init": CONFIGS[args.config].init,
        "components": args.components,
        "seed": args.seed,
    }
    if (
        manifest["identity"] != identity
        or sha256(args.snapshot / "fit.h5") != manifest["model_sha256"]
        or sha256(args.snapshot / "fit.npy") != manifest["embedding_sha256"]
        or content_sha256(train) != manifest["train_sha256"]
    ):
        raise ValueError("Fixed-fit snapshot content has changed")
    fitted = UMAP.load(args.snapshot / "fit.h5", backend=args.backend)
    arrays, metadata = fitted._model.export_state()
    fit = np.load(args.snapshot / "fit.npy", allow_pickle=False)
    fitted_embedding = arrays["embedding"]
    if fitted_embedding is None or not np.array_equal(fit, fitted_embedding):
        raise ValueError("Snapshot embedding differs from fitted state")
    return arrays, metadata, fit, manifest


def model_for_epochs(arrays, metadata, backend, epochs):
    from cuscan import _cuscan

    return _cuscan.load(arrays, {**metadata, "n_epochs": 3 * epochs}, backend)


def graph_oracle(ids, distances, fit):
    from umap.umap_ import compute_membership_strengths, init_graph_transform, smooth_knn_dist

    rows, width = ids.shape
    sigma, rho = smooth_knn_dist(distances, float(width), local_connectivity=0.0)
    heads, tails, weights, _ = compute_membership_strengths(
        ids, distances, sigma, rho, bipartite=True
    )
    graph = coo_matrix((weights, (heads, tails)), shape=(rows, len(fit))).tocsr()
    graph.eliminate_zeros()
    graph.sort_indices()
    return rho, sigma, weights, graph, init_graph_transform(graph, fit)


def consumed_edges(replay, optimizer):
    if optimizer == "compatibility":
        return replay["edge_heads"], replay["edge_tails"], replay["edge_weights"]
    offsets = np.asarray(replay["csr_offsets"])
    if not len(offsets):
        return np.array([], dtype=np.int32), replay["csr_indices"], replay["csr_weights"]
    return (
        np.repeat(np.arange(len(offsets) - 1), np.diff(offsets)),
        replay["csr_indices"],
        replay["csr_weights"],
    )


def schedule_comparison(
    replay,
    heads,
    tails,
    weights,
    periods,
    negative,
    next_positive,
    next_negative,
    optimizer,
    epochs,
    alpha,
):
    actual_heads, actual_tails, actual_weights = consumed_edges(replay, optimizer)
    checks = {
        "heads": compare(actual_heads, heads, exact=True),
        "tails": compare(actual_tails, tails, exact=True),
        "weights": compare(actual_weights, weights),
        "periods": compare(replay["periods"], periods),
        "negative_periods": compare(replay["negative_periods"], negative),
        "next_positive": compare(replay["next_positive"], next_positive),
        "next_negative": compare(replay["next_negative"], next_negative),
        "epochs": compare(replay["epochs"], epochs, exact=True),
        "initial_alpha": compare(replay["initial_alpha"], alpha),
    }
    return {"matched": all(item["matched"] for item in checks.values()), "checks": checks}


def host_schedule(replay, ids, memberships, optimizer, epochs, rate, alpha, n_train):
    from umap.umap_ import make_epochs_per_sample

    heads = np.repeat(np.arange(len(ids)), ids.shape[1])
    coo = coo_matrix((memberships, (heads, ids.ravel())), shape=(len(ids), n_train))
    coo.data[coo.data < coo.data.max() / epochs] = 0
    coo.eliminate_zeros()
    if optimizer != "compatibility":
        coo = coo.tocsr()
        coo.sort_indices()
        coo = coo.tocoo()
    weights = coo.data
    periods = make_epochs_per_sample(weights, epochs)
    negative = periods / rate * (1.0 if optimizer == "compatibility" else 1.5)
    return schedule_comparison(
        replay,
        coo.row,
        coo.col,
        weights,
        periods,
        negative,
        periods,
        negative,
        optimizer,
        epochs,
        alpha,
    )


def capture_arrays(root, run):
    path = (root / run["file"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Capture path escapes worker directory")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(run["arrays"]):
            raise ValueError("Capture array inventory differs from manifest")
        arrays = {name: archive[name] for name in archive.files}
    for name, value in arrays.items():
        spec = run["arrays"][name]
        digest = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        if (
            list(value.shape) != spec["shape"]
            or str(value.dtype) != spec["dtype"]
            or digest != spec["sha256"]
        ):
            raise ValueError(f"Capture array changed: {name}")
    return arrays


def capture_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "replay"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--components", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if (args.mode == "replay") != (args.capture is not None):
        parser.error("--capture is required for replay and forbidden for host")
    train, batches, _ = load_fixture(args.data, args.fixture)
    if args.batch is not None:
        batches = {args.batch: batches[args.batch]}
    arrays, metadata, fit, manifest = fixed_state(args, train)
    optimizer = CONFIGS[args.config].optimizer
    rows = []
    if args.mode == "host":
        from comparison.umap_reference import require_reference

        require_reference()
        for count, queries in batches.items():
            search_model = model_for_epochs(arrays, metadata, args.backend, 100)
            search = search_model._replay_transform_search(queries)
            ids = np.asarray(search["neighbor_indices"], dtype=np.int32).reshape(count, -1)
            distances = np.asarray(search["neighbor_distances"], dtype=np.float32).reshape(
                ids.shape
            )
            rho, sigma, memberships, _, initial = graph_oracle(ids, distances, fit)
            zero_model = model_for_epochs(arrays, metadata, args.backend, 0)
            zero = zero_model._replay_transform_graph(ids, distances)
            zero_initial = np.asarray(zero["initial"]).reshape(initial.shape)
            zero_output = np.asarray(zero["embedding"]).reshape(initial.shape)
            for epochs in (5, 10, 11, 100):
                model = model_for_epochs(arrays, metadata, args.backend, epochs)
                replay = model._replay_transform_graph(ids, distances)
                rows.append(
                    {
                        "batch": count,
                        "query_sha256": content_sha256(queries),
                        "epochs": epochs,
                        "graph": {
                            "rho": compare(replay["rho"], rho),
                            "sigma": compare(replay["sigma"], sigma),
                            "memberships": compare(replay["memberships"], memberships),
                        },
                        "initial": compare(
                            np.asarray(replay["initial"]).reshape(initial.shape), initial
                        ),
                        "zero_epoch_initial": compare(zero_initial, initial),
                        "zero_epoch_output": compare(zero_output, zero_initial),
                        "initial_csr_order": "ascending_column",
                        "schedule": host_schedule(
                            replay,
                            ids,
                            memberships,
                            optimizer,
                            epochs,
                            metadata["negative_sample_rate"],
                            metadata["learning_rate"] / 4.0,
                            len(fit),
                        ),
                    }
                )
    else:
        record = json.loads((args.capture / "run.json").read_text(encoding="utf-8"))
        stage = record["stages"]
        if (
            record["fixture"] != args.fixture
            or record["config"] != args.config
            or record["seed"] != args.seed
            or record["components"] != args.components
            or stage["schema"] != "umap06.upstream_stages.v1"
            or stage["embedding_hash"] != capture_hash(fit)
            or stage["train_hash"] != capture_hash(train)
            or not stage["ab_check"]
            or not stage["ab_check"]["matched"]
            or not np.allclose(
                [stage["a"], stage["b"]], [manifest["a"], manifest["b"]], rtol=1e-6, atol=0
            )
        ):
            raise ValueError("Upstream capture does not match snapshot identity")
        for run in stage["runs"]:
            count = run["rows"]
            if count not in batches:
                continue
            queries = batches[count]
            if run["query_hash"] != capture_hash(queries) or run["optimizer"] != optimizer:
                raise ValueError("Upstream query or optimizer identity mismatch")
            captured = capture_arrays(args.capture, run)
            epochs = run["n_epochs"]
            model = model_for_epochs(arrays, metadata, args.backend, epochs)
            ids = np.asarray(captured["knn_ids"], dtype=np.int32)
            distances = np.asarray(captured["knn_distances"], dtype=np.float32)
            graph = model._replay_transform_graph(ids, distances)
            layout = model._replay_transform_layout(
                np.asarray(captured["edge_heads"], dtype=np.int32),
                np.asarray(captured["edge_tails"], dtype=np.int32),
                np.asarray(captured["edge_weights"], dtype=np.float32),
                np.asarray(captured["initial_placement"], dtype=np.float32),
            )
            schedule = schedule_comparison(
                layout,
                captured["edge_heads"],
                captured["edge_tails"],
                captured["edge_weights"],
                captured["epochs_per_sample"],
                captured["epochs_per_negative_sample"],
                captured["epoch_of_next_sample"],
                captured["epoch_of_next_negative_sample"],
                optimizer,
                epochs,
                run["initial_alpha"],
            )
            parameters_match = (
                run["n_vertices"] == len(train)
                and not run["good_initialization"]
                and not run["move_other"]
                and run["negative_sample_rate"] == metadata["negative_sample_rate"]
                and np.isclose(run["gamma"], metadata["repulsion_strength"], rtol=RTOL, atol=ATOL)
            )
            graph_schedule = schedule_comparison(
                graph,
                captured["edge_heads"],
                captured["edge_tails"],
                captured["edge_weights"],
                captured["epochs_per_sample"],
                captured["epochs_per_negative_sample"],
                captured["epoch_of_next_sample"],
                captured["epoch_of_next_negative_sample"],
                optimizer,
                epochs,
                run["initial_alpha"],
            )
            scores = score_transforms(
                train,
                {"upstream": fit, "cuscan": fit},
                queries,
                {
                    "upstream": captured["final_embedding"],
                    "cuscan": np.asarray(layout["embedding"]).reshape(count, args.components),
                },
            )
            rows.append(
                {
                    "batch": count,
                    "epochs": epochs,
                    "capture": run["name"],
                    "query_sha256": content_sha256(queries),
                    "graph": {
                        "rho": compare(graph["rho"], captured["rhos"]),
                        "sigma": compare(graph["sigma"], captured["sigmas"]),
                        "memberships": compare(graph["memberships"], captured["membership_vals"]),
                    },
                    "initial": compare(
                        np.asarray(graph["initial"]).reshape(count, args.components),
                        captured["initial_placement"],
                    ),
                    "graph_schedule": graph_schedule,
                    "injected_schedule": schedule,
                    "layout_parameters_matched": bool(parameters_match),
                    "layout_attribution": "unattributed",
                    "layout_attribution_reason": "The hook does not expose its alpha schedule; "
                    "shared schedule equality cannot be fully verified.",
                    "scores": scores,
                }
            )
        if not rows:
            raise ValueError("No matching upstream stage captures")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(
        args.out,
        {
            "schema": "umap06.phase7_pins.v1",
            "mode": args.mode,
            "identity": manifest["identity"],
            "snapshot_model_sha256": manifest["model_sha256"],
            "missing_batches": sorted(set(batches) - {row["batch"] for row in rows}),
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
