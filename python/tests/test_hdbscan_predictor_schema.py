from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import fast_umap
from fast_umap import _hdbscan_predictor_serialization_v1 as serialization

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

NAMES = (
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
)


def no_tree_state() -> tuple[dict[str, None], dict[str, Any]]:
    arrays = dict.fromkeys(NAMES)
    metadata = {
        "n_features": 2,
        "n_finite": 0,
        "min_samples": 2,
        "cluster_nodes": 0,
        "cores_status": "not_required_no_cluster_tree",
        "cores_rank": 0,
    }
    return arrays, metadata


def test_no_tree_schema_is_minimal_and_exact(tmp_path: Path) -> None:
    arrays, metadata = no_tree_state()
    path = tmp_path / "predictor.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r") as file:
        assert set(file.keys()) == {"state"}
        assert file.attrs["model_kind"] == "hdbscan_predictor"
        state = file["state"]
        assert set(state.keys()) == set()
        assert set(state.attrs) == {
            "n_features",
            "n_finite",
            "min_samples",
            "cluster_nodes",
            "cores_rank",
            "cores_status",
            *(f"{name}_present" for name in NAMES),
        }
        assert all(not state.attrs[f"{name}_present"] for name in NAMES)
        assert state.attrs["cores_status"] == "not_required_no_cluster_tree"


def test_load_validates_backend_before_opening_file() -> None:
    with pytest.raises(ValueError, match="backend"):
        serialization.load_model(object, "missing.h5", backend="bogus")
    with pytest.raises(ValueError, match="device_index"):
        serialization.load_model(object, "missing.h5", device_index=True)


def test_no_tree_load_passes_native_minimal_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, metadata = no_tree_state()
    path = tmp_path / "predictor.h5"
    serialization.save_model(path, arrays, metadata)
    received: list[tuple[dict[str, Any], dict[str, Any], str, int]] = []

    class Native:
        pass

    class Predictor:
        @staticmethod
        def _from_native(native: Any) -> Any:
            return native

    def capture(a: dict[str, Any], m: dict[str, Any], backend: str, device_index: int) -> Native:
        received.append((a, m, backend, device_index))
        return Native()

    monkeypatch.setattr(fast_umap._fast_umap, "load_hdbscan_predictor", capture)
    restored = serialization.load_model(Predictor, path, backend="cpu", device_index=3)
    assert isinstance(restored, Native)
    loaded_arrays, loaded_metadata, backend, device_index = received[0]
    assert set(loaded_arrays) == set(NAMES)
    assert all(value is None for value in loaded_arrays.values())
    assert loaded_metadata == metadata
    assert (backend, device_index) == ("cpu", 3)


def test_wrong_model_kind_is_reported(tmp_path: Path) -> None:
    arrays, metadata = no_tree_state()
    path = tmp_path / "predictor.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        file.attrs["model_kind"] = "standard_umap"
    with pytest.raises(ValueError, match=r"model kind mismatch.*standard_umap.*hdbscan_predictor"):
        serialization.load_model(object, path)


def test_rejects_presence_mismatch_and_link_alias_before_native_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, metadata = no_tree_state()
    path = tmp_path / "predictor.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        file["state"].attrs["geometry_present"] = np.bool_(True)
    monkeypatch.setattr(
        fast_umap._fast_umap,
        "load_hdbscan_predictor",
        lambda *args, **kwargs: pytest.fail("native loader must not be called"),
    )
    with pytest.raises(ValueError, match="missing dataset geometry"):
        serialization.load_model(object, path)

    serialization.save_model(path, arrays, metadata, overwrite=True)
    with h5py.File(path, "r+") as file:
        file["state/alias"] = file["state"]
    with pytest.raises(ValueError, match="hard-link"):
        serialization.load_model(object, path)


def _clustered_state() -> tuple[Any, dict[str, Any], dict[str, Any]]:
    x = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [3.0, 3.0], [3.1, 3.0], [3.0, 3.1]],
        dtype=np.float32,
    )
    fitted = fast_umap._fast_umap.fit_hdbscan(x, backend="cpu", min_samples=1, min_cluster_size=2)
    predictor = fitted.to_predictor()
    arrays, metadata = predictor.export_state()
    return predictor, arrays, metadata


def test_rejects_scalar_limits_and_candidate_width_before_dataset_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arrays, metadata = no_tree_state()
    metadata["n_features"] = int(np.iinfo(np.intp).max) + 1
    with pytest.raises(ValueError, match="platform allocation"):
        serialization.save_model(tmp_path / "overflow.h5", arrays, metadata)

    arrays, metadata = no_tree_state()
    path = tmp_path / "no-tree-overflow.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        file["state"].attrs["n_features"] = np.uint64(int(np.iinfo(np.intp).max) + 1)
    monkeypatch.setattr(
        fast_umap._fast_umap,
        "load_hdbscan_predictor",
        lambda *args, **kwargs: pytest.fail("native loader must not be called"),
    )
    with pytest.raises(ValueError, match="platform allocation"):
        serialization.load_model(object, path)

    _, arrays, metadata = _clustered_state()
    path = tmp_path / "width.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        file["state"].attrs["min_samples"] = np.uint64(metadata["n_finite"])
        file["state"].attrs["cores_rank"] = np.uint64(metadata["n_finite"])

    def unexpected_dataset_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("payload/filter inspection must not start before candidate-width validation")

    monkeypatch.setattr(serialization._common, "_filters", unexpected_dataset_read)
    with pytest.raises(ValueError, match="candidate-width"):
        serialization.load_model(object, path)


def test_rejects_predictor_dataset_attributes_and_wrong_state_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, arrays, metadata = _clustered_state()
    path = tmp_path / "dataset-attrs.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        file["state/geometry"].attrs["unexpected"] = 1
    monkeypatch.setattr(
        fast_umap._fast_umap,
        "load_hdbscan_predictor",
        lambda *args, **kwargs: pytest.fail("native loader must not be called"),
    )
    with pytest.raises(ValueError, match="dataset attributes"):
        serialization.load_model(object, path)

    arrays, metadata = no_tree_state()
    serialization.save_model(path, arrays, metadata, overwrite=True)
    with h5py.File(path, "r+") as file:
        del file["state"]
        file.create_dataset("state", data=np.uint8(0))
    with pytest.raises(ValueError, match="state must be a group"):
        serialization.load_model(object, path)


def test_rejects_predictor_chunk_policy_mismatch(tmp_path: Path) -> None:
    _, arrays, metadata = _clustered_state()
    path = tmp_path / "chunks.h5"
    serialization.save_model(path, arrays, metadata)
    with h5py.File(path, "r+") as file:
        state = file["state"]
        dataset = state["geometry"]
        payload = dataset[...]
        del state["geometry"]
        state.create_dataset(
            "geometry",
            data=payload,
            dtype="<f4",
            chunks=(max(1, len(payload) - 1),),
            **serialization._deps()[1].Blosc2(
                cname="zstd", filters=serialization._deps()[1].Blosc2.SHUFFLE, clevel=1
            ),
        )
    with pytest.raises(ValueError, match="filter pipeline or chunks"):
        serialization.load_model(object, path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("dtype", "invalid predictor dataset dtype"),
        ("rank", "invalid predictor dataset dtype"),
        ("length", "invalid predictor dataset dtype"),
        ("virtual", "virtual and externally stored datasets"),
        ("external", "virtual and externally stored datasets"),
        ("wrong_codec", "invalid Blosc2 filter options"),
        ("wrong_shuffle", "invalid Blosc2 filter options"),
        ("wrong_clevel", "invalid Blosc2 filter options"),
        ("extra_filter", "invalid predictor filter pipeline or chunks"),
    ],
)
def test_rejects_malformed_predictor_dataset_without_reading_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    _, arrays, metadata = _clustered_state()
    path = tmp_path / f"{mutation}.h5"
    serialization.save_model(path, arrays, metadata)
    plugin = serialization._deps()[1]
    with h5py.File(path, "r+") as file:
        state = file["state"]
        original = state["geometry"]
        payload = original[...]
        del state["geometry"]
        length = len(payload)
        if mutation == "virtual":
            source_path = tmp_path / "virtual-source.h5"
            with h5py.File(source_path, "w") as source_file:
                source_file.create_dataset("geometry", data=payload, dtype="<f4")
            source = h5py.VirtualSource(str(source_path), "geometry", shape=(length,))
            layout = h5py.VirtualLayout(shape=(length,), dtype="<f4")
            layout[:] = source[:]
            state.create_virtual_dataset("geometry", layout)
        elif mutation == "external":
            state.create_dataset(
                "geometry",
                shape=(length,),
                dtype="<f4",
                external=[(str(tmp_path / "external.bin"), 0, h5py.h5f.UNLIMITED)],
            )
        elif mutation in ("dtype", "rank", "length"):
            if mutation == "dtype":
                data = payload.astype("<f8")
                dtype = "<f8"
            elif mutation == "rank":
                data = payload.reshape(length, 1)
                dtype = "<f4"
            else:
                data = payload[:-1]
                dtype = "<f4"
            chunks = (min(data.shape[0], 32_768), *data.shape[1:])
            state.create_dataset(
                "geometry",
                data=data,
                dtype=dtype,
                chunks=chunks,
                **plugin.Blosc2(cname="zstd", filters=plugin.Blosc2.SHUFFLE, clevel=1),
            )
        else:
            blosc_options: dict[str, Any] = {
                "cname": "zstd",
                "filters": plugin.Blosc2.SHUFFLE,
                "clevel": 1,
            }
            if mutation == "wrong_codec":
                blosc_options["cname"] = "lz4"
            elif mutation == "wrong_shuffle":
                blosc_options["filters"] = plugin.Blosc2.NOFILTER
            elif mutation == "wrong_clevel":
                blosc_options["clevel"] = 2
            dataset_options = dict(plugin.Blosc2(**blosc_options))
            if mutation == "extra_filter":
                dataset_options["fletcher32"] = True
            state.create_dataset(
                "geometry",
                data=payload,
                dtype="<f4",
                chunks=(min(length, 32_768),),
                **dataset_options,
            )

    def unexpected_native_load(*args: Any, **kwargs: Any) -> None:
        pytest.fail("malformed HDF5 dataset reached the native loader")

    def unexpected_payload_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("malformed HDF5 payload was decoded before schema rejection")

    monkeypatch.setattr(fast_umap._fast_umap, "load_hdbscan_predictor", unexpected_native_load)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", unexpected_payload_read)
    with pytest.raises(ValueError, match=message):
        serialization.load_model(object, path)


@pytest.mark.parametrize("name", ["n_features", "min_samples", "cores_rank"])
def test_save_rejects_invalid_no_tree_scalars(name: str, tmp_path: Path) -> None:
    arrays, metadata = no_tree_state()
    metadata[name] = 1 if name == "cores_rank" else 0
    with pytest.raises(ValueError):
        serialization.save_model(tmp_path / "bad.h5", arrays, metadata)


def test_native_predictor_round_trip_directly(tmp_path: Path) -> None:
    predictor, arrays, metadata = _clustered_state()
    path = tmp_path / "native.h5"
    serialization.save_model(path, arrays, metadata)

    class NativePredictor:
        @staticmethod
        def _from_native(native: Any) -> Any:
            return native

    restored = serialization.load_model(NativePredictor, path, backend="cpu")
    queries = np.array([[0.05, 0.02], [3.05, 3.02]], dtype=np.float32)
    for left, right in zip(
        predictor.approximate_predict(queries, True),
        restored.approximate_predict(queries, True),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
