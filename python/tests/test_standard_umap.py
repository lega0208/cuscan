import os
import threading
import time
from itertools import pairwise
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


def test_unique_defaults_to_raw_training_behavior() -> None:
    x, _ = blobs(12)
    rows = np.vstack((x, x[0], x[3]))
    default = estimator().fit_transform(rows)
    explicit_raw = estimator(unique=False).fit_transform(rows)
    assert default.shape == (len(rows), 2)
    np.testing.assert_array_equal(default, explicit_raw)


def test_unique_expands_first_occurrence_embedding_and_transforms() -> None:
    x, _ = blobs(12)
    x[0, 0] = 0.0
    signed_zero_duplicate = x[0].copy()
    signed_zero_duplicate[0] = -0.0
    rows = np.vstack((x, x[5], signed_zero_duplicate, x[2], x[5]))

    first_occurrences: list[np.ndarray] = []
    row_to_unique: list[int] = []
    for row in rows:
        for index, distinct in enumerate(first_occurrences):
            if np.array_equal(row, distinct):
                row_to_unique.append(index)
                break
        else:
            row_to_unique.append(len(first_occurrences))
            first_occurrences.append(row)
    distinct_rows = np.stack(first_occurrences)

    unique_model = estimator(unique=True, init="random").fit(rows)
    reference_embedding = estimator(init="random").fit_transform(distinct_rows)
    np.testing.assert_array_equal(unique_model.embedding_, reference_embedding[row_to_unique])
    np.testing.assert_array_equal(unique_model.embedding_[-3], unique_model.embedding_[0])
    np.testing.assert_array_equal(unique_model.embedding_[-1], unique_model.embedding_[5])
    assert unique_model.embedding_.shape == (len(rows), 2)

    transformed = unique_model.transform(np.vstack((rows[1], rows[-2])))
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()


def test_unique_requires_two_distinct_rows_and_validates_before_deduplication() -> None:
    repeated = np.repeat(np.array([[1.0, 2.0]], dtype=np.float32), 4, axis=0)
    with pytest.raises(ValueError, match=r"distinct|unique|at least"):
        estimator(unique=True).fit(repeated)

    two_distinct = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        estimator(unique=True).fit(two_distinct)

    nonfinite = np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, 3.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="finite"):
        estimator(unique=True).fit(nonfinite)


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


def test_native_transform_accepts_contiguous_and_strided_inputs() -> None:
    x, _ = blobs(12)
    model = estimator().fit(x)
    native_model = model._model
    expected = native_model.transform(x)

    transposed = np.ascontiguousarray(x.T).T
    reverse_strided = np.ascontiguousarray(x[::-1, ::-1])[::-1, ::-1]
    assert not transposed.flags.c_contiguous
    assert not reverse_strided.flags.c_contiguous
    assert any(stride < 0 for stride in reverse_strided.strides)

    np.testing.assert_array_equal(native_model.transform(transposed), expected)
    np.testing.assert_array_equal(native_model.transform(reverse_strided), expected)


def test_native_transform_rejects_invalid_shape_and_nonfinite_values() -> None:
    x, _ = blobs(12)
    native_model = estimator().fit(x)._model

    with pytest.raises(ValueError, match="features"):
        native_model.transform(np.zeros((2, x.shape[1] + 1), dtype=np.float32))
    nonfinite = np.zeros((2, x.shape[1]), dtype=np.float32)
    nonfinite[1, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        native_model.transform(nonfinite)


def test_native_fitted_model_rejects_transform_from_another_thread() -> None:
    x, _ = blobs(12)
    native_model = estimator().fit(x)._model
    errors: list[BaseException] = []

    def transform_from_worker() -> None:
        try:
            native_model.transform(x[:1])
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=transform_from_worker)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert type(errors[0]).__name__ == "PanicException"
    assert type(errors[0]).__module__ == "pyo3_runtime"
    assert "FittedUMAP is unsendable, but sent to another thread" in str(errors[0])


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


@pytest.mark.skipif(
    os.environ.get("FAST_UMAP_REQUIRE_CUDA") != "1",
    reason="CUDA required only in CUDA validation runs",
)
def test_cuda_transform_releases_gil_for_at_least_half_second() -> None:
    x, _ = blobs(12)
    model = estimator(backend="cuda", init="random", neighbor_search="exact", n_epochs=200).fit(x)
    query = np.random.default_rng(23).normal(size=(8192, x.shape[1])).astype(np.float32)

    stop = threading.Event()
    started = threading.Event()
    counter_timestamps: list[int] = []

    def count_during_native_transform() -> None:
        while not stop.is_set():
            counter_timestamps.append(time.time_ns())
            started.set()
            time.sleep(0.001)

    counter = threading.Thread(target=count_during_native_transform)
    counter.start()
    assert started.wait(timeout=5)
    try:
        output, start_ns, end_ns = model._model._timed_transform(query)  # ty: ignore[unresolved-attribute]
    finally:
        stop.set()
        counter.join(timeout=5)

    assert not counter.is_alive()
    assert end_ns - start_ns >= 500_000_000, "native CUDA transform took less than 500 ms"
    during_transform = [stamp for stamp in counter_timestamps if start_ns <= stamp <= end_ns]
    print(
        f"native_transform_ns={end_ns - start_ns} "
        f"counter_ticks_inside_native={len(during_transform)}"
    )
    assert len(during_transform) >= 10, "fewer than ten Python counter ticks overlapped native work"
    assert all(later - earlier >= 1_000_000 for earlier, later in pairwise(during_transform)), (
        "Python counter ticks were less than 1 ms apart"
    )
    assert output.shape == (len(query), model.n_components)
    assert np.isfinite(output).all()
