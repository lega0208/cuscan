from __future__ import annotations

import gc
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fast_umap import _fast_umap

ARRAY_KEYS = {
    "geometry",
    "finite_index",
    "node_ids",
    "node_parent_indices",
    "node_birth_lambdas",
    "node_labels",
    "node_max_lambdas",
    "point_node_indices",
    "point_departure_lambdas",
    "prediction_cores",
}
METADATA_KEYS = {
    "n_features",
    "n_finite",
    "min_samples",
    "cluster_nodes",
    "cores_status",
    "cores_rank",
}


def clustered_native() -> Any:
    train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [8.0, 8.0],
            [8.1, 8.0],
            [8.0, 8.1],
            [8.1, 8.1],
        ],
        dtype=np.float32,
    )
    return _fast_umap.fit_hdbscan(
        train, min_cluster_size=3, min_samples=2, backend="cpu"
    ).to_predictor()


def shortcut_native() -> Any:
    train = np.array([[0.0], [10.0], [20.0], [30.0]], dtype=np.float32)
    return _fast_umap.fit_hdbscan(
        train, min_cluster_size=3, min_samples=2, backend="cpu"
    ).to_predictor()


def test_export_uses_exact_keys_shapes_dtypes_and_independent_snapshots() -> None:
    predictor = clustered_native()
    arrays, metadata = predictor.export_state()
    assert set(arrays) == ARRAY_KEYS
    assert set(metadata) == METADATA_KEYS
    expected_dtypes = {
        "geometry": np.dtype(np.float32),
        "finite_index": np.dtype(np.uint32),
        "node_ids": np.dtype(np.uint64),
        "node_parent_indices": np.dtype(np.uint64),
        "node_birth_lambdas": np.dtype(np.float64),
        "node_labels": np.dtype(np.int32),
        "node_max_lambdas": np.dtype(np.float64),
        "point_node_indices": np.dtype(np.uint64),
        "point_departure_lambdas": np.dtype(np.float64),
        "prediction_cores": np.dtype(np.float64),
    }
    for key, dtype in expected_dtypes.items():
        assert isinstance(arrays[key], np.ndarray)
        assert arrays[key].ndim == 1
        assert arrays[key].dtype == dtype
    assert metadata == {
        "n_features": 2,
        "n_finite": len(arrays["finite_index"]),
        "min_samples": 2,
        "cluster_nodes": len(arrays["node_ids"]),
        "cores_status": "ready",
        "cores_rank": 2,
    }
    before = predictor.approximate_predict(np.array([[0.05, 0.03]], np.float32), True)
    for value in arrays.values():
        value[...] = 0
    after = predictor.approximate_predict(np.array([[0.05, 0.03]], np.float32), True)
    for left, right in zip(before, after, strict=True):
        np.testing.assert_array_equal(left, right)


def test_native_loader_default_device_index_and_prediction_parity() -> None:
    predictor = clustered_native()
    arrays, metadata = predictor.export_state()
    restored = _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")
    query = np.array([[0.05, 0.03], [8.04, 8.02]], dtype=np.float32)
    expected = predictor.approximate_predict(query, True)
    actual = restored.approximate_predict(query, True)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_no_cluster_export_is_canonical_none_state() -> None:
    arrays, metadata = shortcut_native().export_state()
    assert set(arrays) == ARRAY_KEYS
    assert all(value is None for value in arrays.values())
    assert metadata == {
        "n_features": 1,
        "n_finite": 0,
        "min_samples": 2,
        "cluster_nodes": 0,
        "cores_status": "not_required_no_cluster_tree",
        "cores_rank": 0,
    }


def test_no_cluster_loader_rejects_non_none_arrays() -> None:
    arrays, metadata = shortcut_native().export_state()
    arrays["geometry"] = np.zeros(0, dtype=np.float32)
    with pytest.raises(ValueError, match="all arrays to be None"):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")


@pytest.mark.parametrize("side", ["arrays", "metadata"])
def test_loader_rejects_missing_and_extra_keys(side: str) -> None:
    arrays, metadata = clustered_native().export_state()
    target = arrays if side == "arrays" else metadata
    target.pop(next(iter(target)))
    with pytest.raises(ValueError, match="exactly"):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")
    arrays, metadata = clustered_native().export_state()
    target = arrays if side == "arrays" else metadata
    target["unexpected"] = None
    with pytest.raises(ValueError, match="exactly"):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("n_features", True),
        ("n_features", 2.0),
        ("n_features", np.int64(2)),
        ("n_features", -1),
        ("n_finite", False),
        ("n_finite", 8.0),
        ("n_finite", -1),
        ("min_samples", True),
        ("min_samples", 2.0),
        ("min_samples", -1),
        ("cluster_nodes", False),
        ("cluster_nodes", 1.0),
        ("cluster_nodes", -1),
        ("cores_rank", True),
        ("cores_rank", 2.0),
        ("cores_rank", -1),
        ("cores_status", True),
        ("cores_status", "READY"),
    ],
)
def test_loader_rejects_noncanonical_metadata_scalars(key: str, bad: object) -> None:
    arrays, metadata = clustered_native().export_state()
    metadata[key] = bad
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("geometry", np.zeros(16, dtype=np.float64)),
        ("geometry", np.zeros((8, 2), dtype=np.float32)),
        ("geometry", np.zeros(15, dtype=np.float32)),
        ("finite_index", np.zeros(8, dtype=np.int64)),
        ("finite_index", np.zeros((8, 1), dtype=np.uint32)),
        ("finite_index", np.zeros(7, dtype=np.uint32)),
        ("node_ids", np.zeros(1, dtype=np.int64)),
        ("node_parent_indices", np.zeros(1, dtype=np.uint32)),
        ("node_birth_lambdas", np.zeros(1, dtype=np.float32)),
        ("node_labels", np.zeros(1, dtype=np.int64)),
        ("node_max_lambdas", np.zeros(1, dtype=np.float32)),
        ("point_node_indices", np.zeros(7, dtype=np.uint64)),
        ("point_departure_lambdas", np.zeros((8, 1), dtype=np.float64)),
        ("prediction_cores", np.zeros(7, dtype=np.float64)),
    ],
)
def test_loader_rejects_array_dtype_rank_and_shape(key: str, replacement: np.ndarray) -> None:
    arrays, metadata = clustered_native().export_state()
    arrays[key] = replacement
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend="cpu")


@pytest.mark.parametrize("backend", ["invalid", "CPU", "", 1])
def test_loader_rejects_unsupported_backend(backend: Any) -> None:
    arrays, metadata = clustered_native().export_state()
    with pytest.raises((TypeError, ValueError)):
        _fast_umap.load_hdbscan_predictor(arrays, metadata, backend=backend)


@pytest.mark.parametrize("device_index", [True, False, -1, 1.0, "0"])
def test_loader_requires_nonnegative_builtin_integer_device_index(device_index: Any) -> None:
    arrays, metadata = clustered_native().export_state()
    with pytest.raises((TypeError, ValueError, OverflowError)):
        _fast_umap.load_hdbscan_predictor(
            arrays, metadata, backend="cpu", device_index=device_index
        )


def test_native_prediction_feature_width_and_bool_are_strict() -> None:
    predictor = clustered_native()
    with pytest.raises(ValueError, match="features"):
        predictor.approximate_predict(np.zeros((1, 1), dtype=np.float32))
    for flag in (None, 0, 1, np.bool_(True), "yes"):
        with pytest.raises((TypeError, ValueError), match="boolean"):
            predictor.approximate_predict(np.zeros((1, 2), dtype=np.float32), flag)


def test_shortcut_warning_count_and_stacklevel() -> None:
    predictor = shortcut_native()
    query = np.array([[4.0]], dtype=np.float32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = predictor.approximate_predict(query, True, stacklevel=1)
    matching = [
        warning
        for warning in caught
        if "does not have any defined clusters" in str(warning.message)
    ]
    assert len(matching) == 1
    assert Path(matching[0].filename).name == Path(__file__).name
    assert all(array.dtype == np.dtype(np.intp) for array in (result[0], result[2]))
    assert result[1].dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(result[0], -1)
    np.testing.assert_array_equal(result[1], 0.0)
    np.testing.assert_array_equal(result[2], -1)


def test_predictor_outlives_native_fitted_estimator() -> None:
    train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
            [8.0, 8.0],
            [8.1, 8.0],
            [8.0, 8.1],
            [8.1, 8.1],
        ],
        dtype=np.float32,
    )
    fitted = _fast_umap.fit_hdbscan(train, min_cluster_size=3, min_samples=2, backend="cpu")
    predictor = fitted.to_predictor()
    del fitted
    gc.collect()
    output = predictor.approximate_predict(np.array([[0.05, 0.03]], np.float32), True)
    assert len(output) == 3
    assert all(len(array) == 1 for array in output)
