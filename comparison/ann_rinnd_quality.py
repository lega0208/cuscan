#!/usr/bin/env -S uv run --no-sync
"""Fit and score the fixed R2 quality matrix for the rinnd campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

ROWS = 500_000
FEATURES = 256
N_NEIGHBORS = 30
SEED = 11
SCORE_K = 15


class _UnexpectedPyNNDescent(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedding_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def _verified_arm_provenance(path: Path, arm: str) -> dict[str, Any]:
    provenance = json.loads(path.read_text(encoding="utf-8"))
    if provenance.get("arm") != arm:
        raise ValueError(f"provenance arm does not match {arm}: {path}")
    for key in ("head", "base_sha", "tree_id"):
        value = provenance.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"provenance {key} must be a lowercase Git object ID: {path}")
    for key in (
        "product_diff_sha256",
        "2a_patch_sha256",
        "arm_patch_sha256",
        "binary_sha256",
        "extension_sha256",
    ):
        value = provenance.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"provenance {key} must be a lowercase SHA-256: {path}")
    if provenance["head"] != provenance["base_sha"]:
        raise ValueError(f"provenance head does not match base_sha: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "arm": arm,
        "head": provenance["head"],
        "product_diff_sha256": provenance["product_diff_sha256"],
        "base_sha": provenance["base_sha"],
        "tree_id": provenance["tree_id"],
        "2a_patch_sha256": provenance["2a_patch_sha256"],
        "arm_patch_sha256": provenance["arm_patch_sha256"],
        "binary_sha256": provenance["binary_sha256"],
        "extension_sha256": provenance["extension_sha256"],
    }


def _load_input(path: Path, expected_sha256: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    actual = _sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"input sha256 mismatch: expected {expected_sha256}, found {actual}")
    values = np.fromfile(path, dtype="<f4")
    if values.size != ROWS * FEATURES:
        raise ValueError(f"input has {values.size} values; expected {ROWS * FEATURES}")
    values = values.reshape(ROWS, FEATURES)
    if not np.isfinite(values).all():
        raise ValueError("input contains non-finite values")
    return values, {
        "path": str(path.resolve()),
        "rows": ROWS,
        "features": FEATURES,
        "dtype": "f32le",
        "sha256": actual,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_embedding(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (ROWS, 2) or not np.isfinite(array).all():
        raise ValueError(
            f"{name} embedding must be finite with shape ({ROWS}, 2), got {array.shape}"
        )
    return np.ascontiguousarray(array)


def _receipt_base(input_info: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "schema": "ann-rinnd-quality-v1",
        "role": role,
        "input": input_info,
        "matrix": {"n_neighbors": N_NEIGHBORS, "seed": SEED, "score_k": SCORE_K},
        "driver": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__))},
        "python": platform.python_version(),
    }


def _fit_reference(x: np.ndarray, input_info: dict[str, Any], out: Path) -> None:
    import umap
    import umap.umap_ as umap_module
    from comparison.umap_reference import REFERENCE_VERSION, reference_kwargs, require_reference

    require_reference()
    actual_version = importlib.metadata.version("umap-learn")
    if actual_version != REFERENCE_VERSION or not actual_version.startswith("0.6."):
        raise RuntimeError(f"reference requires umap-learn 0.6.x, found {actual_version}")
    rinnd_module = importlib.import_module("rinnd")
    rinnd_path = Path(rinnd_module.__file__).resolve()
    if not rinnd_path.is_file():
        raise RuntimeError(f"rinnd module origin is not a file: {rinnd_path}")

    original_rinnd = umap_module.RINND
    if original_rinnd is None:
        raise RuntimeError("umap-learn did not bind RINND")
    original_nnd = umap_module.NNDescent
    counters = {"rinnd": 0, "pynndescent": 0}

    def counting_rinnd(*args: Any, **kwargs: Any) -> Any:
        counters["rinnd"] += 1
        return original_rinnd(*args, **kwargs)

    def guarded_nnd(*args: Any, **kwargs: Any) -> Any:
        counters["pynndescent"] += 1
        raise _UnexpectedPyNNDescent("PyNNDescent fallback was invoked")

    umap_module.RINND = counting_rinnd
    vars(umap_module)["NNDescent"] = guarded_nnd

    def kwargs(backend: str) -> dict[str, Any]:
        return reference_kwargs(
            optimizer="adam",
            init="recursive",
            n_neighbors=N_NEIGHBORS,
            n_components=2,
            min_dist=0.1,
            spread=1.0,
            learning_rate=1.0,
            negative_sample_rate=5,
            repulsion_strength=1.0,
            random_state=SEED,
            transform_seed=SEED,
            metric="euclidean",
            low_memory=True,
            verbose=False,
            n_jobs=-1,
            nearest_neighbor_backend=backend,
            force_approximation_algorithm=True,
        )

    try:
        counters.update(rinnd=0, pynndescent=0)
        umap.UMAP(**kwargs("rinnd")).fit(x[:5_000])
        if counters["rinnd"] < 1 or counters["pynndescent"] != 0:
            raise RuntimeError(f"rinnd preflight dispatch failed: {counters}")
        preflight = {
            "rows": 5_000,
            "rinnd_construction_calls": counters["rinnd"],
            "pynndescent_calls": counters["pynndescent"],
        }

        counters.update(rinnd=0, pynndescent=0)
        try:
            umap.UMAP(**kwargs("pynndescent")).fit(x[:5_000])
        except _UnexpectedPyNNDescent as exc:
            if counters["pynndescent"] < 1:
                raise RuntimeError(
                    "forced-PyNNDescent negative control did not reach its guard"
                ) from exc
            negative_control = {
                "passed": True,
                "forced_backend": "pynndescent",
                "guard_calls": counters["pynndescent"],
            }
        else:
            raise RuntimeError(
                "forced-PyNNDescent negative control did not trip the fallback guard"
            )

        counters.update(rinnd=0, pynndescent=0)
        model = umap.UMAP(**kwargs("rinnd"))
        started = time.perf_counter()
        embedding = _validate_embedding(model.fit_transform(x), "rinnd")
        fit_seconds = time.perf_counter() - started
        if counters["rinnd"] < 1 or counters["pynndescent"] != 0:
            raise RuntimeError(f"rinnd fit dispatch failed: {counters}")
        if counters["rinnd"] < preflight["rinnd_construction_calls"]:
            raise RuntimeError("reference fit did not construct rinnd")
    finally:
        umap_module.RINND = original_rinnd
        umap_module.NNDescent = original_nnd

    out.mkdir(parents=True, exist_ok=False)
    embedding_path = out / "embedding.npy"
    np.save(embedding_path, embedding, allow_pickle=False)
    receipt = _receipt_base(input_info, "reference")
    receipt.update(
        {
            "arm": "R",
            "embedding": {
                "path": str(embedding_path.resolve()),
                "sha256": _embedding_sha256(embedding),
            },
            "fit_seconds": fit_seconds,
            "configuration": {
                "nearest_neighbor_backend": "rinnd",
                "force_approximation_algorithm": True,
                "optimizer": "adam",
                "init": "recursive",
                "n_jobs": -1,
                "metric": "euclidean",
                "n_components": 2,
                "min_dist": 0.1,
                "spread": 1.0,
                "learning_rate": 1.0,
                "negative_sample_rate": 5,
                "repulsion_strength": 1.0,
                "low_memory": True,
            },
            "dispatch": {
                "preflight": preflight,
                "fit_rinnd_construction_calls": counters["rinnd"],
                "fit_pynndescent_calls": counters["pynndescent"],
                "negative_control": negative_control,
                "umap_learn_version": actual_version,
                "rinnd_module_path": str(rinnd_path),
                "rinnd_module_sha256": _sha256(rinnd_path),
            },
            "software": {"umap_learn": actual_version},
        }
    )
    _write_json(out / "receipt.json", receipt)


def _fit_cuscan(
    arm: str,
    x: np.ndarray,
    input_info: dict[str, Any],
    out: Path,
    provenance_path: Path,
) -> None:
    import cuscan
    from cuscan import UMAP

    source = _verified_arm_provenance(provenance_path, arm)
    extension_path = Path(cuscan._cuscan.__file__).resolve()
    if not extension_path.is_file():
        raise RuntimeError(f"imported cuscan extension is not a file: {extension_path}")
    extension_sha256 = _sha256(extension_path)
    if extension_sha256 != source["extension_sha256"]:
        raise ValueError(
            "imported cuscan extension sha256 does not match source provenance: "
            f"expected {source['extension_sha256']}, found {extension_sha256} ({extension_path})"
        )

    model = UMAP(
        backend="cuda",
        neighbor_search="approximate",
        n_neighbors=N_NEIGHBORS,
        random_state=SEED,
    )
    if model.neighbor_search != "approximate":
        raise RuntimeError("cuscan approximate neighbor search was not selected")
    started = time.perf_counter()
    embedding = _validate_embedding(model.fit_transform(x), arm)
    fit_seconds = time.perf_counter() - started
    _, metadata = model._model.export_state()
    resolved_search = metadata.get("resolved_search")
    if resolved_search != "approximate":
        raise RuntimeError(f"cuscan resolved search to {resolved_search!r}, not 'approximate'")
    out.mkdir(parents=True, exist_ok=False)
    embedding_path = out / "embedding.npy"
    np.save(embedding_path, embedding, allow_pickle=False)
    receipt = _receipt_base(input_info, "arm")
    receipt.update(
        {
            "arm": arm,
            "embedding": {
                "path": str(embedding_path.resolve()),
                "sha256": _embedding_sha256(embedding),
            },
            "fit_seconds": fit_seconds,
            "configuration": {
                "backend": "cuda",
                "neighbor_search": "approximate",
                "resolved_search": resolved_search,
                "n_neighbors": N_NEIGHBORS,
                "random_state": SEED,
                "defaults": "otherwise (Adam, init=auto -> recursive)",
            },
            "source": {
                **source,
                "extension_path": str(extension_path),
                "imported_extension_sha256": extension_sha256,
            },
        }
    )
    _write_json(out / "receipt.json", receipt)


def _parse_embedding(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("embedding must be ARM=PATH")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("embedding must be ARM=PATH")
    return name, Path(path)


def _score(input_info: dict[str, Any], x: np.ndarray, specs: list[str], out: Path) -> None:
    from comparison.umap06 import quality

    embeddings: dict[str, np.ndarray] = {}
    fit_receipts: dict[str, dict[str, Any]] = {}
    for spec in specs:
        name, path = _parse_embedding(spec)
        if name in embeddings:
            raise ValueError(f"duplicate embedding arm: {name}")
        receipt_path = path.parent / "receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("input", {}).get("sha256") != input_info["sha256"]:
            raise ValueError(f"{name} receipt input hash does not match scoring input")
        if receipt.get("matrix") != {
            "n_neighbors": N_NEIGHBORS,
            "seed": SEED,
            "score_k": SCORE_K,
        }:
            raise ValueError(f"{name} receipt matrix does not match this campaign")
        values = _validate_embedding(np.load(path, allow_pickle=False), name)
        actual_embedding_hash = _embedding_sha256(values)
        if receipt.get("embedding", {}).get("sha256") != actual_embedding_hash:
            raise ValueError(f"{name} embedding hash does not match its fit receipt")
        embeddings[name] = values
        fit_receipts[name] = {
            "path": str(receipt_path.resolve()),
            "sha256": _sha256(receipt_path),
            "embedding_sha256": actual_embedding_hash,
            "fit_seconds": receipt.get("fit_seconds"),
        }
    if not embeddings:
        raise ValueError("at least one --embedding ARM=PATH is required")
    expected_arms = {"P0", "P1", "P2", "R"}
    if embeddings.keys() != expected_arms:
        raise ValueError(f"score all matrix arms exactly once; expected {sorted(expected_arms)}")
    query_ids = quality._evaluation_ids(len(x))
    scores = quality.score_fits(x, embeddings, query_ids=query_ids, k=SCORE_K)
    receipt = _receipt_base(input_info, "scores")
    receipt.update(
        {
            "arms": fit_receipts,
            "query_ids": query_ids.tolist(),
            "query_ids_sha256": hashlib.sha256(
                np.asarray(query_ids, dtype="<i8").tobytes()
            ).hexdigest(),
            "scoring": {"module": "comparison.umap06.quality.score_fits", "k": SCORE_K},
            "scores": scores,
        }
    )
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "receipt.json", receipt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reference", choices=("rinnd",), help="fit upstream umap-learn reference")
    mode.add_argument("--arm", choices=("P0", "P1", "P2"), help="fit one cuscan arm")
    mode.add_argument(
        "--score", action="store_true", help="score one or more previously fitted arms"
    )
    parser.add_argument("--input", type=Path, required=True, help="hashed R2 f32le input")
    parser.add_argument("--input-sha256", help="expected SHA-256 of the input")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="verified arm JSON with source tree/patch IDs and expected extension_sha256",
    )
    parser.add_argument("--out", type=Path, required=True, help="new output directory")
    parser.add_argument(
        "--embedding", action="append", default=[], help="scoring input ARM=embedding.npy"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    x, input_info = _load_input(args.input, args.input_sha256)
    if args.score:
        _score(input_info, x, args.embedding, args.out)
    elif args.reference:
        if args.provenance is not None:
            raise ValueError("--provenance is only valid when fitting a cuscan arm")
        _fit_reference(x, input_info, args.out)
    else:
        if args.provenance is None:
            raise ValueError("cuscan arm fits require --provenance")
        _fit_cuscan(args.arm, x, input_info, args.out, args.provenance)


if __name__ == "__main__":
    main()
