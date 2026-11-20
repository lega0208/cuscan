"""Generate recursive substep fixtures from the pinned umap-learn 0.6.0 wheel.

Run from the repository root with `uv run --no-sync python
comparison/umap06/generate_recursive_oracle.py`. Inputs and outputs are JSON
arrays; CSR graphs have offsets, indices and float32 weights in row order.
"""

import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
import sys
from pathlib import Path

# Keep linear algebra and numba scheduling independent of the host core count.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numba  # noqa: E402
import numpy as np  # noqa: E402
import umap.label_prop as label_prop  # noqa: E402
import umap.layouts as layouts  # noqa: E402
import umap.umap_ as umap_module  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCES = {
    "umap/label_prop.py": (
        label_prop,
        "5447ce40fd99340161ec592807917fa55e31327cc719cb198be593012b656745",
    ),
    "umap/layouts.py": (
        layouts,
        "802c106f33bcfca071c96de3b20bc519cb26750571cee9f8f7e12a5339d37e70",
    ),
    "umap/umap_.py": (
        umap_module,
        "1441f803cf35a3412172c9d598f3ed6309e21dcfc4e416e4e677163faf05af88",
    ),
}


def provenance():
    from comparison.umap06.run import wheel_provenance

    dist = metadata.distribution("umap-learn")
    if dist.version != "0.6.0" or dist.files is None:
        raise RuntimeError("expected an installed umap-learn 0.6.0 distribution with RECORD")
    hashes = {}
    for name, (module, expected) in SOURCES.items():
        entry = next((item for item in dist.files if str(item) == name), None)
        if entry is None or entry.hash is None or entry.hash.mode != "sha256":
            raise RuntimeError(f"missing sha256 RECORD entry for {name}")
        path = Path(str(dist.locate_file(entry))).resolve()
        digest = hashlib.sha256(path.read_bytes()).digest()
        if (
            path != Path(module.__file__).resolve()
            or digest.hex() != expected
            or base64.urlsafe_b64encode(digest).rstrip(b"=").decode() != entry.hash.value
        ):
            raise RuntimeError(f"source does not match pinned wheel and RECORD: {name}")
        hashes[name] = digest.hex()
    return {
        "version": dist.version,
        "source_sha256": hashes,
        **wheel_provenance(),
        "numpy": np.__version__,
        "numba": numba.__version__,
    }


def graph(offsets, indices, weights):
    n = len(offsets) - 1
    return csr_matrix(
        (
            np.array(weights, dtype=np.float32),
            np.array(indices, dtype=np.int32),
            np.array(offsets, dtype=np.int32),
        ),
        shape=(n, n),
    )


def graph_json(g):
    return {"offsets": g.indptr.tolist(), "indices": g.indices.tolist(), "weights": g.data.tolist()}


def base_case(name, g, anchor, mask, depth):
    anchor = np.array(anchor, dtype=np.float32)
    mask = np.array(mask, dtype=np.bool_)
    filled = anchor.copy() if mask.all() else label_prop._fill_unsampled(g, anchor, mask)
    result = label_prop._base_case_layout(g, anchor, mask, depth)
    return {
        "name": name,
        "graph": graph_json(g),
        "anchor": anchor.tolist(),
        "sample_mask": mask.tolist(),
        "depth": depth,
        "filled": filled.tolist(),
        "result": result.tolist(),
    }


def expansion_case(name, g, labels, coarse):
    labels = np.array(labels, dtype=np.int32)
    coarse = np.array(coarse, dtype=np.float32)
    expanded = label_prop._expand_layout(g.indptr, g.indices, g.data, labels, coarse)
    centered = (expanded - expanded.mean(axis=0)).astype(np.float32)
    return {
        "name": name,
        "graph": graph_json(g),
        "labels": labels.tolist(),
        "coarse": coarse.tolist(),
        "expanded_f64": expanded.tolist(),
        "centered_f32": centered.tolist(),
    }


def alignment_case(name, anchor, embedding):
    anchor = np.ascontiguousarray(anchor, dtype=np.float32)
    embedding = np.ascontiguousarray(embedding, dtype=np.float32)
    rotation = label_prop.procrustes_align(anchor, embedding)
    return {
        "name": name,
        "anchor": anchor.tolist(),
        "embedding": embedding.tolist(),
        "rotation": rotation.tolist(),
        "aligned": (embedding @ rotation).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "crates/umap-core/tests/fixtures/recursive_oracle.json",
    )
    args = parser.parse_args()
    numba.set_num_threads(1)
    source = provenance()
    g = graph([0, 2, 4, 6, 7, 7], [1, 2, 0, 2, 0, 1, 4], [0.25, 0.75, 0.25, 0.5, 0.75, 0.5, 1.0])
    full = graph([0, 1, 2, 3, 4], [1, 0, 3, 2], [1.0, 1.0, 0.5, 0.5])
    payload = {
        "provenance": source,
        "base_cases": [
            base_case(
                "weighted_and_fallback", g, [[2, 4], [8, -2]], [True, False, True, False, False], 1
            ),
            base_case(
                "all_sampled_constant_axis",
                full,
                [[1, 7], [3, 7], [5, 7], [9, 7]],
                [True, True, True, True],
                3,
            ),
        ],
        "prolongation_cases": [
            expansion_case("weighted_and_isolated", g, [0, 1, 2, 1, 0], [[1, 2], [-3, 4], [5, -6]]),
            expansion_case(
                "fractional_weights",
                full,
                [0, 1, 1, 2],
                [[0.1, -0.3, 1.25], [2.5, 0.2, -4], [-1, 3, 0.75]],
            ),
        ],
        "procrustes_cases": [
            alignment_case(
                "rotation",
                [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                [[1, -1], [1, 1], [-1, 1], [-1, -1]],
            ),
            alignment_case(
                "reflection_corrected",
                [[0, 0], [2, 0], [0, 1], [1, 3]],
                [[0, 0], [-2, 0], [0, 1], [-1, 3]],
            ),
            alignment_case("collapsed", [[1, 2], [1, 2], [1, 2]], [[0, 0], [2, 1], [-1, 3]]),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
