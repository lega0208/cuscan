from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from fast_umap import UMAP, NotFittedError


def blobs(
    n_per_cluster: int = 30, n_features: int = 5
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    rng = np.random.default_rng(17)
    centers = rng.normal(size=(3, n_features)).astype(np.float32) * 5
    labels = np.repeat(np.arange(3), n_per_cluster)
    data = np.concatenate(
        [center + rng.normal(scale=0.25, size=(n_per_cluster, n_features)) for center in centers]
    ).astype(np.float32)
    return data, labels


def estimator(**kwargs: object) -> UMAP:
    params: dict[str, Any] = {
        "n_neighbors": 8,
        "n_epochs": 20,
        "random_state": 17,
        "device": "cpu",
        "hidden_sizes": (16,),
    }
    params.update(kwargs)
    return UMAP(**params)


def test_fit_transform_shape_dtype_finite() -> None:
    x, _ = blobs()
    embedding = estimator().fit_transform(x)
    assert embedding.shape == (90, 2)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()


def test_fit_sets_learned_attributes_and_returns_self() -> None:
    x, _ = blobs()
    model = estimator()
    assert model.fit(x) is model
    assert model.embedding_.shape == (90, 2)
    assert model.n_features_in_ == 5


def test_transform_new_data_and_training_data() -> None:
    x, _ = blobs()
    model = estimator().fit(x)
    projected = model.transform(x[:7])
    assert projected.shape == (7, 2)
    assert projected.dtype == np.float32
    np.testing.assert_allclose(model.transform(x), model.embedding_, atol=1e-5)


def test_transform_empty_data() -> None:
    x, _ = blobs()
    model = estimator().fit(x)
    transformed = model.transform(np.empty((0, x.shape[1]), dtype=np.float32))
    assert transformed.shape == (0, model.n_components)
    assert transformed.dtype == np.float32


def test_transform_empty_data_with_wrong_feature_count() -> None:
    x, _ = blobs()
    model = estimator().fit(x)
    with pytest.raises(ValueError, match="features"):
        model.transform(np.empty((0, x.shape[1] - 1), dtype=np.float32))


def test_n_components_three() -> None:
    x, _ = blobs()
    assert estimator(n_components=3).fit_transform(x).shape == (90, 3)


def test_seed_determinism_and_variation() -> None:
    x, _ = blobs()
    first = estimator(random_state=111).fit_transform(x)
    same = estimator(random_state=111).fit_transform(x)
    other = estimator(random_state=112).fit_transform(x)
    np.testing.assert_array_equal(first, same)
    assert not np.array_equal(first, other)


def test_cluster_separation() -> None:
    x, labels = blobs()
    embedding = estimator().fit_transform(x)
    within: list[float] = []
    between: list[float] = []
    for i in range(len(x)):
        distances = np.linalg.norm(embedding[i] - embedding[i + 1 :], axis=1)
        for offset, distance in enumerate(distances, start=i + 1):
            (within if labels[i] == labels[offset] else between).append(float(distance))
    assert np.mean(within) < np.mean(between)


def test_validation_errors() -> None:
    x, _ = blobs()
    with pytest.raises(ValueError, match="2-D"):
        estimator().fit(x[0])
    invalid = x.copy()
    invalid[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        estimator().fit(invalid)
    with pytest.raises(ValueError, match="n_samples"):
        UMAP(n_neighbors=len(x), device="cpu").fit(x)
    with pytest.raises(ValueError, match="non-negative"):
        estimator(random_state=-1).fit(x)
    with pytest.raises(ValueError, match="metric"):
        estimator(metric="cosine").fit(x)
    with pytest.raises(NotFittedError):
        estimator().transform(x)
    with pytest.raises(ValueError, match="features"):
        estimator().fit(x).transform(x[:, :-1])
    with pytest.raises(NotImplementedError):
        estimator().fit(x, np.zeros(len(x)))


def test_float64_integer_and_noncontiguous_inputs() -> None:
    x, _ = blobs(n_features=6)
    for source in (
        x.astype(np.float64),
        (x * 10).astype(np.int64),
        np.asfortranarray(x),
        x[:, ::2],
    ):
        result = estimator().fit_transform(source)
        assert result.dtype == np.float32
        assert np.isfinite(result).all()


def test_params_repr_and_random_state_generators() -> None:
    model = estimator(n_components=3)
    clone_params = model.get_params()
    clone = UMAP().set_params(**clone_params)
    assert clone.get_params() == clone_params
    assert list(clone_params) == list(inspect.signature(UMAP.__init__).parameters)[1:]
    assert "n_components=3" in repr(model)
    params_model = UMAP()
    assert params_model.set_params(device="cpu") is params_model
    random_model = UMAP(random_state=np.random.RandomState(1))
    assert random_model._validate_params() == int(
        np.random.RandomState(1).randint(0, np.iinfo(np.int32).max)
    )
    generator_model = UMAP(random_state=np.random.default_rng(1))
    assert generator_model._validate_params() == int(
        np.random.default_rng(1).integers(0, np.iinfo(np.int32).max)
    )


def test_non_euclidean_metric_rejected() -> None:
    x, _ = blobs()
    with pytest.raises(ValueError, match="euclidean"):
        estimator(metric="manhattan").fit(x)


def test_transform_uses_training_standardization() -> None:
    x, _ = blobs()
    model = estimator().fit(x)
    transformed = model.transform(x)
    np.testing.assert_allclose(transformed, model.embedding_, atol=1e-5)


def test_gpu_smoke_when_adapter_available() -> None:
    x, _ = blobs(n_per_cluster=6)
    try:
        result = UMAP(n_neighbors=3, n_epochs=2, hidden_sizes=(4,), device="gpu").fit_transform(x)
    except BaseException as error:
        pytest.skip(f"GPU adapter unavailable or backend initialization failed: {error}")
    assert result.shape == (18, 2)
    assert np.isfinite(result).all()
