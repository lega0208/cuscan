"""Generate independent fixtures by executing the installed umap-learn 0.6 kernels."""

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

import numba
import numpy as np
import umap.layouts as layouts

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def provenance():
    from comparison.umap06.run import wheel_provenance

    dist = metadata.distribution("umap-learn")
    if dist.version != "0.6.0":
        raise RuntimeError(f"expected umap-learn 0.6.0, found {dist.version}")
    files = dist.files
    if files is None:
        raise RuntimeError("umap-learn distribution has no recorded files")
    entry = next(f for f in files if str(f) == "umap/layouts.py")
    path = Path(str(dist.locate_file(entry))).resolve()
    if (
        path != Path(layouts.__file__).resolve()
        or entry.hash is None
        or entry.hash.mode != "sha256"
    ):
        raise RuntimeError("layouts.py is not the recorded distribution source")
    digest = hashlib.sha256(path.read_bytes()).digest()
    if base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value:
        raise RuntimeError("layouts.py differs from distribution RECORD")
    return {"version": dist.version, "layouts_sha256": digest.hex(), **wheel_provenance()}


def case(rows, dim, operation, optimizer, gamma):
    refs = rows if operation == "fit" else rows + 3
    # Distinct, exactly reproducible f32 coordinates; rows 0 and 1 coincide.
    embedding = ((np.arange(refs * dim).reshape(refs, dim) % 37) - 18).astype(np.float32) / 32
    embedding[1] = embedding[0]
    head = embedding[:rows].copy()
    tail = head if operation == "fit" else embedding.copy()
    initial_head, initial_tail = head.copy(), tail.copy()
    active = [0, 2, rows // 2, rows - 1]
    targets = [1, 3, rows // 2 + 1, 0]
    indptr = np.zeros(rows + 1, dtype=np.int32)
    for row in active:
        indptr[row + 1] += 1
    indptr = np.cumsum(indptr, dtype=np.int32)
    indices = np.array(targets, dtype=np.int32)
    periods = np.array([1.0, 2.0, 1.0, 4.0], dtype=np.float64)
    negative_periods = np.array([0.5, 1.0, 0.5, 2.0], dtype=np.float64)
    next_positive = np.array([1.0, 5.0, 2.0, 2.0], dtype=np.float64)
    next_negative = np.array([0.5, 1.0, 1.0, 2.0], dtype=np.float64)
    initial_positive, initial_negative = next_positive.copy(), next_negative.copy()
    from_order = np.arange(rows, dtype=np.int32)
    to_order = np.arange(refs, dtype=np.int32)[::-1].copy()
    ranks = np.arange(rows, dtype=np.int32)
    updates = np.full((rows, dim), 0.0, dtype=np.float32)
    if optimizer == "momentum":
        updates[0, 0] = 0.03125
    initial_updates = updates.copy()
    first = np.zeros_like(head)
    second = np.zeros_like(head)
    n, alpha, momentum = 3, 0.2, 0.4
    small = rows <= 1024
    common = [head, tail, indptr, indices]
    if small:
        common.append(np.ones(len(indices), dtype=np.float32))
    common += [
        refs,
        periods,
        1.0,
        1.0,
        gamma,
        dim,
        alpha,
        negative_periods,
        next_negative,
        next_positive,
        n,
        updates,
    ]
    if optimizer == "adam":
        common += [first, second, 0.9, 0.99]
    common += [from_order, to_order]
    if not small:
        common += [ranks, rows // 2, refs, 0.75]
    common.append(False)  # Explicit b1 required by all four compiled signatures.
    kernel = getattr(
        layouts, f"optimize_{'small_' if small else ''}layout_euclidean_single_epoch_{optimizer}"
    )
    kernel(*common)
    raw_updates = updates.copy()
    layouts._finalize_update_buffer(updates, optimizer, momentum)
    return {
        "rows": rows,
        "refs": refs,
        "dim": dim,
        "operation": operation,
        "optimizer": optimizer,
        "gamma": gamma,
        "epoch": n,
        "alpha": alpha,
        "momentum": momentum,
        "scale": 0.75,
        "block": rows if small else rows // 2,
        "head": initial_head.tolist(),
        "tail": initial_tail.tolist(),
        "offsets": indptr.tolist(),
        "indices": indices.tolist(),
        "periods": periods.tolist(),
        "negative_periods": negative_periods.tolist(),
        "next_positive": initial_positive.tolist(),
        "next_negative": initial_negative.tolist(),
        "from_order": from_order.tolist(),
        "to_order": to_order.tolist(),
        "ranks": ranks.tolist(),
        "initial_updates": initial_updates.tolist(),
        "expected_head": head.tolist(),
        "expected_updates": raw_updates.tolist(),
        "expected_final_updates": updates.tolist(),
        "expected_first": first.tolist(),
        "expected_second": second.tolist(),
        "expected_positive": next_positive.tolist(),
        "expected_negative": next_negative.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "crates/umap-core/tests/fixtures/modern_layout_upstream.json",
    )
    args = parser.parse_args()
    numba.set_num_threads(1)
    schedules = []
    for optimizer in ("adam", "momentum"):
        for good in (False, True):
            for epochs in (1, 7, 16, 201):
                schedules.append(
                    {
                        "optimizer": optimizer,
                        "good": good,
                        "epochs": epochs,
                        "alpha": layouts._create_alpha_schedule(
                            optimizer, epochs, 1.0, good
                        ).tolist(),
                        "momentum": layouts._create_momentum_schedule(
                            optimizer, epochs, good
                        ).tolist(),
                    }
                )
    cases = [
        case(rows, dim, op, opt, gamma)
        for rows in (1024, 1025)
        for dim in (16, 48)
        for op in ("fit", "transform")
        for opt in ("adam", "momentum")
        for gamma in (0.0, 1.5)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"provenance": provenance(), "schedules": schedules, "cases": cases}, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
