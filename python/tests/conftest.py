from __future__ import annotations

import os
from typing import Literal

import pytest

from fast_umap import _fast_umap


@pytest.fixture(scope="session")
def hdbscan_backend() -> Literal["cpu", "cuda"]:
    backend = os.environ.get("FAST_UMAP_TEST_BACKEND", "cpu").lower()
    if backend not in {"cpu", "cuda"}:
        pytest.fail("FAST_UMAP_TEST_BACKEND must be 'cpu' or 'cuda'")
    if backend == "cuda" and not _fast_umap.hdbscan_cuda_available():
        if os.environ.get("FAST_UMAP_REQUIRE_CUDA") == "1":
            pytest.fail("CUDA was required but HDBSCAN CUDA is unavailable")
        pytest.skip("HDBSCAN CUDA is unavailable")
    return backend
