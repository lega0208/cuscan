import os
from typing import Any

import numpy as np
import pytest

import fast_umap
from fast_umap import UMAP


def blobs(n_per_cluster: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    centers = np.array([[-7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32)
    labels = np.repeat(np.arange(len(centers)), n_per_cluster)
    data = np.vstack([center + rng.normal(0, 0.22, (n_per_cluster, 3)) for center in centers])
    return data.astype(np.float32), labels


def estimator(**kwargs: object) -> UMAP:
    params: dict[str, Any] = {"n_neighbors": 10, "n_epochs": 100, "random_state": 19}
    params.update(kwargs)
    return UMAP(**params)


def test_exports_and_shape_dtype() -> None:
    assert "UMAP" in fast_umap.__all__
    x, _ = blobs()
    model = estimator()
    result = model.fit_transform(x)
    assert result.shape == (len(x), 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert model.embedding_.shape == result.shape


def test_random_state_determinism() -> None:
    x, _ = blobs(12)
    first = estimator().fit_transform(x)
    second = estimator().fit_transform(x)
    np.testing.assert_array_equal(first, second)


def test_blob_clusters_are_separated() -> None:
    x, labels = blobs()
    embedding = estimator().fit_transform(x)
    centers = np.stack([embedding[labels == group].mean(axis=0) for group in range(3)])
    within = np.mean(
        [
            np.linalg.norm(embedding[labels == group] - centers[group], axis=1).mean()
            for group in range(3)
        ]
    )
    between = np.mean(
        [np.linalg.norm(centers[i] - centers[j]) for i in range(3) for j in range(i + 1, 3)]
    )
    assert between > within


def test_transform_held_out_sample_near_cluster() -> None:
    x, _ = blobs()
    model = estimator().fit(x)
    query = np.array([[0.05, 7.05, 0.02]], dtype=np.float32)
    projected = model.transform(query)
    assert projected.shape == (1, 2)
    assert projected.dtype == np.float32
    assert np.linalg.norm(projected[0] - model.embedding_[24:48].mean(axis=0)) < np.linalg.norm(
        projected[0] - model.embedding_[:24].mean(axis=0)
    )


def test_singleton_is_zero_and_transform_error_is_preserved() -> None:
    model = estimator().fit(np.array([[2.0, 3.0]], dtype=np.float32))
    np.testing.assert_array_equal(model.embedding_, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(
        ValueError, match="Transform unavailable when model was fit with only a single data sample"
    ):
        model.transform(np.array([[2.0, 3.0]], dtype=np.float32))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_neighbors": 1}, "n_neighbors"),
        ({"min_dist": -0.1}, "min_dist"),
        ({"backend": "bogus"}, "backend"),
        ({"neighbor_search": "bogus"}, "neighbor_search"),
    ],
)
def test_invalid_parameters_raise_value_error(kwargs: dict[str, object], message: str) -> None:
    x, _ = blobs(12)
    with pytest.raises(ValueError, match=message):
        estimator(**kwargs).fit(x)


@pytest.mark.skipif(
    os.environ.get("FAST_UMAP_REQUIRE_CUDA") != "1",
    reason="CUDA required only in CUDA validation runs",
)
def test_cuda_backend() -> None:
    x, _ = blobs(12)
    result = estimator(backend="cuda").fit_transform(x)
    assert result.shape == (len(x), 2)
    assert np.isfinite(result).all()
