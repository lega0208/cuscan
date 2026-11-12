from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from cuscan import HDBSCAN, UMAP, _cuscan


@pytest.mark.parametrize("estimator", [UMAP, HDBSCAN])
@pytest.mark.parametrize("value", [0, -1, True, np.bool_(False), 1.5, "1", 2**64])
def test_invalid_memory_limit(estimator: type, value: object) -> None:
    model = estimator(memory_limit_bytes=value)
    with pytest.raises(ValueError, match="memory_limit_bytes"):
        model.fit(np.ones((16, 3), dtype=np.float32))


@pytest.mark.parametrize("estimator", [UMAP, HDBSCAN])
@pytest.mark.parametrize("value", [np.int64(123), 2**64 - 1])
def test_memory_limit_forwarded(
    estimator: type, value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = []

    def capture(*args: object, **kwargs: object) -> None:
        received.append(kwargs["memory_limit_bytes"])
        raise RuntimeError("captured")

    name = "fit_umap" if estimator is UMAP else "estimate_hdbscan_resources"
    monkeypatch.setattr(_cuscan, name, capture)
    with pytest.raises(RuntimeError, match="captured"):
        estimator(memory_limit_bytes=value).fit(np.ones((16, 3), dtype=np.float32))
    assert received == [int(value)]
    assert type(received[0]) is int


def test_hdbscan_memory_limit_reaches_native_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr(_cuscan, "estimate_hdbscan_resources", lambda *a, **k: {"knn": None})

    def capture(*args: object, **kwargs: object) -> None:
        received.append(kwargs["memory_limit_bytes"])
        raise RuntimeError("captured")

    monkeypatch.setattr(_cuscan, "fit_hdbscan", capture)
    with pytest.raises(RuntimeError, match="captured"):
        HDBSCAN(memory_limit_bytes=123).fit(np.ones((16, 3), dtype=np.float32))
    assert received == [123]


@pytest.mark.parametrize(
    "backend",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                os.environ.get("CUSCAN_REQUIRE_CUDA") != "1", reason="CUDA validation only"
            ),
        ),
    ],
)
def test_umap_ann_memory_error(backend: Literal["cpu", "cuda"]) -> None:
    x = np.random.default_rng(42).normal(size=(64, 4)).astype(np.float32)
    with pytest.raises(MemoryError):
        UMAP(
            backend=backend, neighbor_search="approximate", init="random", memory_limit_bytes=1
        ).fit(x)


def test_hdbscan_ann_memory_error(hdbscan_backend: Literal["cpu", "cuda"]) -> None:
    x = np.random.default_rng(42).normal(size=(4097, 31)).astype(np.float32)
    with pytest.raises(MemoryError):
        HDBSCAN(backend=hdbscan_backend, memory_limit_bytes=1).fit(x)


@pytest.mark.parametrize("limit", [None, 2**64 - 1])
def test_umap_saved_memory_limit_snapshot(tmp_path: Path, limit: int | None) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    x = np.random.default_rng(42).normal(size=(32, 4)).astype(np.float32)
    model = UMAP(
        backend="cpu", neighbor_search="exact", init="random", n_epochs=10, memory_limit_bytes=limit
    ).fit(x)
    model.memory_limit_bytes = 1
    path = tmp_path / "model.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    assert restored.memory_limit_bytes == limit
    assert restored._fit_config["memory_limit_bytes"] == limit
    restored.save(tmp_path / "resaved.h5")
