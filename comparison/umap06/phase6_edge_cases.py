"""Phase 6 edge cases of the default optimizer and init, through the public Python API.

``--mode explicit`` passes ``optimizer="adam", init="auto"``; ``--mode library`` passes
neither. Each case records the fit outcome and, when the fit succeeded, the transform
outcome separately. Data generators are deterministic and live here so the predecessor
(P6) and candidate (D6) runs use identical inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SEED = 7
TRANSFORM_SEED = 7


def digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f4").tobytes()).hexdigest()


def seeded_normal(rows: int, features: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, features)).astype(np.float32)


def separated_blobs() -> np.ndarray:
    """6 blobs x 40 rows x 8 features, centres ``blob * 50`` on axis ``blob % 8``, unit jitter.

    Same layout as ``separated_blobs()`` in ``test_standard_estimator.rs``; the jitter comes
    from NumPy's generator, so values are not bitwise equal to the Rust fixture.
    """
    jitter = np.random.default_rng(SEED).uniform(-1.0, 1.0, (240, 8))
    data = jitter.astype(np.float32)
    for row in range(240):
        blob = row // 40
        data[row, blob % 8] += np.float32(blob * 50.0)
    return data


def with_duplicates() -> np.ndarray:
    base = seeded_normal(200, 4, 101)
    return np.concatenate([base, base[:20]], axis=0)


def clustered(rows: int) -> np.ndarray:
    random = np.random.default_rng(1024)
    centres = random.normal(0.0, 10.0, (8, 6))
    labels = np.arange(rows) % 8
    return (centres[labels] + random.standard_normal((rows, 6))).astype(np.float32)


def case_list() -> list[tuple[str, Callable[[], np.ndarray], dict[str, Any]]]:
    exact = {"neighbor_search": "exact"}
    cases: list[tuple[str, Callable[[], np.ndarray], dict[str, Any]]] = [
        ("1_one_row", lambda: np.array([[2.0, 3.0]], dtype=np.float32), exact),
        (
            "2_two_rows_k8",
            lambda: np.array([[0.0], [1.0]], dtype=np.float32),
            {**exact, "n_neighbors": 8},
        ),
    ]
    for components in (1, 2, 3):
        cases.append(
            (
                f"3_connected_tiny_c{components}",
                lambda: np.array([[0.0], [1.0], [3.0]], dtype=np.float32),
                {**exact, "n_neighbors": 2, "n_components": components},
            )
        )
    for components in (1, 2):
        cases.append(
            (
                f"4_separated_blobs_c{components}",
                separated_blobs,
                {**exact, "n_components": components},
            )
        )
    cases += [
        ("5_anchor_padding_f2_c3", lambda: seeded_normal(300, 2, 5), {**exact, "n_components": 3}),
        ("6_zero_epochs", lambda: seeded_normal(300, 4, 6), {**exact, "n_epochs": 0}),
        ("7_unique_duplicates", with_duplicates, {**exact, "unique": True}),
        ("8_clustered_1024", lambda: clustered(1024), exact),
        ("8_clustered_1025_approx", lambda: clustered(1025), {"neighbor_search": "approximate"}),
        ("9_identical_50", lambda: np.ones((50, 3), dtype=np.float32), exact),
    ]
    return cases


def outcome(action: Callable[[], np.ndarray]) -> dict[str, Any]:
    try:
        values = np.asarray(action())
    except Exception as error:
        return {"status": "error", "type": type(error).__name__, "message": str(error)}
    return {
        "status": "ok",
        "hash": digest(values),
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
    }


def run_case(mode: str, backend: str, data: np.ndarray, kwargs: dict[str, Any]) -> dict[str, Any]:
    from cuscan import UMAP

    params: dict[str, Any] = {
        "random_state": SEED,
        "transform_seed": TRANSFORM_SEED,
        "backend": backend,
        **kwargs,
    }
    if mode == "explicit":
        params.update(optimizer="adam", init="auto")
    model = UMAP(**params)
    fit = outcome(lambda: model.fit(data).embedding_)
    record: dict[str, Any] = {
        "params": {key: value for key, value in params.items() if key != "backend"},
        "input_hash": digest(data),
        "input_shape": list(data.shape),
        "fit": fit,
    }
    if fit["status"] == "ok":
        query = data[: min(3, data.shape[0])]
        record["transform"] = outcome(lambda: model.transform(query))
    else:
        record["transform"] = {"status": "not_run", "reason": "fit failed"}
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("explicit", "library"), required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="*", help="case-name prefixes to run (default: all)")
    args = parser.parse_args()

    results: dict[str, Any] = {}
    for name, generator, kwargs in case_list():
        if args.cases and not any(name.startswith(prefix) for prefix in args.cases):
            continue
        results[name] = run_case(args.mode, args.backend, generator(), kwargs)
        print(
            f"{name}: fit={results[name]['fit']['status']} "
            f"transform={results[name]['transform']['status']}",
            file=sys.stderr,
        )

    document = {
        "schema": "umap06.phase6_edge_cases.v1",
        "mode": args.mode,
        "backend": args.backend,
        "python": platform.python_version(),
        "cases": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
