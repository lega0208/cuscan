from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fast_umap
from fast_umap import HDBSCAN, HDBSCANPredictor

_FIXTURE_PATH = (
    Path(__file__).parents[2] / "crates/fast-hdbscan/tests/fixtures/hdbscan_prediction.json"
)
_FIXTURE = json.loads(_FIXTURE_PATH.read_text())


def train_data() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [3.0, 3.0], [3.1, 3.0], [3.0, 3.1]],
        dtype=np.float32,
    )


def test_public_type_is_lightweight_and_has_no_arbitrary_constructor() -> None:
    assert "HDBSCANPredictor" in fast_umap.__all__
    with pytest.raises(TypeError, match="created by"):
        HDBSCANPredictor()
    assert not hasattr(HDBSCANPredictor, "labels_")
    assert not hasattr(HDBSCANPredictor, "condensed_tree_")


def test_serialization_dependency_is_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("fast_umap._hdbscan_predictor_serialization_v1")

    def missing(_model: str) -> Any:
        raise ImportError("requires fast-umap[serialization]")

    monkeypatch.setattr(module._common, "_deps", missing)
    predictor = (
        HDBSCAN(backend="cpu", min_cluster_size=2, min_samples=1).fit(train_data()).to_predictor()
    )
    with pytest.raises(ImportError, match=r"fast-umap\[serialization\]"):
        predictor.save("not-written.h5")
    assert fast_umap.HDBSCANPredictor is HDBSCANPredictor


def test_predictor_save_exports_before_file_layer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    predictor = (
        HDBSCAN(backend="cpu", min_cluster_size=2, min_samples=1).fit(train_data()).to_predictor()
    )
    arrays, metadata = predictor._native_predictor.export_state()
    seen: list[tuple[dict[str, Any], dict[str, Any], bool]] = []
    module = importlib.import_module("fast_umap._hdbscan_predictor_serialization_v1")

    def save(path: Any, actual_arrays: Any, actual_metadata: Any, *, overwrite: bool) -> None:
        assert path == tmp_path / "state.h5"
        seen.append((actual_arrays, actual_metadata, overwrite))

    monkeypatch.setattr(module, "save_model", save)
    predictor.save(tmp_path / "state.h5", overwrite=True)
    assert len(seen) == 1
    assert seen[0][2] is True
    assert set(seen[0][0]) == set(arrays)
    assert seen[0][1] == metadata


def test_cpu_round_trip_is_exact_and_predictor_only(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    train = train_data()
    train_with_nonfinite = np.vstack(([[np.nan, 0.0]], train))
    fitted = HDBSCAN(backend="cpu", min_cluster_size=2, min_samples=1).fit(train_with_nonfinite)
    predictor = fitted.to_predictor()
    assert predictor.n_features_in_ == 2
    file_path = tmp_path / "prediction-only.h5"
    predictor.save(file_path)
    with h5py.File(file_path, "r") as file:
        assert set(file) == {"state"}
        assert file.attrs["model_kind"] == "hdbscan_predictor"
        state = file["state"]
        assert set(state) == {
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
        assert not {"labels", "probabilities", "condensed_tree", "core_distances"} & set(state)

    restored = HDBSCANPredictor.load(file_path, backend="cpu")
    assert type(restored) is HDBSCANPredictor
    assert restored.n_features_in_ == predictor.n_features_in_
    queries = np.array([[0.05, 0.03], [3.05, 3.02]], dtype=np.float32)
    for actual, expected in zip(
        restored.approximate_predict(queries, True),
        predictor.approximate_predict(queries, True),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert np.min(restored.approximate_predict(queries, True)[2]) >= 1
    assert not hasattr(restored, "labels_")
    assert not hasattr(restored, "probabilities_")
    assert not hasattr(restored, "_fitted_state")
    assert not hasattr(restored, "__dict__")


def test_shortcut_and_root_only_state_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    # No split survives min_cluster_size, so this persists canonical no-tree state.
    no_tree_case = next(case for case in _FIXTURE["fitted"] if case["id"] == "no_cluster_tree")
    train = np.asarray(no_tree_case["train_float32"], dtype=np.float32)
    predictor = HDBSCAN(backend="cpu", **no_tree_case["params"]).fit(train).to_predictor()
    path = tmp_path / "shortcut.h5"
    predictor.save(path)
    restored = HDBSCANPredictor.load(path, backend="cpu")
    with pytest.warns(UserWarning, match="does not have any defined clusters"):
        labels, strengths, connectors = restored.approximate_predict([[0.0]], True)
    np.testing.assert_array_equal(labels, [-1])
    np.testing.assert_array_equal(strengths, [0.0])
    np.testing.assert_array_equal(connectors, [-1])

    # A selected root-only cluster is still represented by the no-cluster shortcut.
    root_case = next(
        case for case in _FIXTURE["fitted"] if case["id"] == "root_only_single_cluster"
    )
    root_only_train = np.asarray(root_case["train_float32"], dtype=np.float32)
    root_only = HDBSCAN(backend="cpu", **root_case["params"]).fit(root_only_train).to_predictor()
    root_path = tmp_path / "root-only.h5"
    root_only.save(root_path)
    root_restored = HDBSCANPredictor.load(root_path, backend="cpu")
    with pytest.warns(UserWarning, match="does not have any defined clusters"):
        root_labels, root_strengths = root_restored.approximate_predict([[5.0]])
    np.testing.assert_array_equal(root_labels, [-1])
    np.testing.assert_array_equal(root_strengths, [0.0])


def test_load_backend_validation_is_keyword_only_and_precedes_file_access() -> None:
    load: Any = HDBSCANPredictor.load
    with pytest.raises(TypeError):
        load("missing.h5", "cpu")
    invalid_backends: tuple[Any, ...] = ("CPU", "invalid")
    for backend in invalid_backends:
        with pytest.raises(ValueError, match="backend"):
            load("missing.h5", backend=backend)
    invalid_indices: tuple[Any, ...] = (True, False, -1, 0.0, "0")
    for device_index in invalid_indices:
        with pytest.raises(ValueError, match="device_index"):
            load("missing.h5", device_index=device_index)


def test_save_refuses_overwrite_without_permission(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    pytest.importorskip("hdf5plugin")
    predictor = (
        HDBSCAN(backend="cpu", min_cluster_size=2, min_samples=1).fit(train_data()).to_predictor()
    )
    path = tmp_path / "state.h5"
    predictor.save(path)
    with pytest.raises(FileExistsError):
        predictor.save(path)
