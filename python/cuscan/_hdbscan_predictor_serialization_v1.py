from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import _serialization_common as _common

_ARRAYS = (
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
_DTYPES = {
    "geometry": "<f4",
    "finite_index": "<u4",
    "node_ids": "<u8",
    "node_parent_indices": "<u8",
    "node_birth_lambdas": "<f8",
    "node_labels": "<i4",
    "node_max_lambdas": "<f8",
    "point_node_indices": "<u8",
    "point_departure_lambdas": "<f8",
    "prediction_cores": "<f8",
}
_META = {"n_features", "n_finite", "min_samples", "cluster_nodes", "cores_status", "cores_rank"}
_MAX_INTP = int(np.iinfo(np.intp).max)


def _deps() -> tuple[Any, Any]:
    return _common._deps("HDBSCAN predictor")


def _uint(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"invalid predictor metadata: {name}")
    return int(value)


def _state(
    metadata: dict[str, Any], arrays: dict[str, Any], *, check_arrays: bool = True
) -> tuple[dict[str, Any], bool]:
    if set(metadata) != _META:
        raise ValueError("invalid predictor metadata fields")
    if set(arrays) != set(_ARRAYS):
        raise ValueError("invalid predictor array fields")
    values = {
        name: _uint(metadata[name], name)
        for name in ("n_features", "n_finite", "min_samples", "cluster_nodes", "cores_rank")
    }
    if any(value > _MAX_INTP for value in values.values()):
        raise ValueError("predictor dimensions exceed platform allocation limits")
    status = metadata["cores_status"]
    if not isinstance(status, str):
        raise ValueError("invalid predictor metadata: cores_status")
    values["cores_status"] = status
    no_tree = status == "not_required_no_cluster_tree"
    if status not in ("ready", "not_required_no_cluster_tree"):
        raise ValueError("invalid predictor core status")
    if values["n_features"] == 0 or values["min_samples"] == 0:
        raise ValueError("predictor feature count and min_samples must be positive")
    expected_presence = not no_tree
    for name in _ARRAYS:
        value = arrays[name]
        if (value is not None) != expected_presence:
            raise ValueError("predictor dataset presence disagrees with core status")
    if no_tree:
        if any(values[name] != 0 for name in ("n_finite", "cluster_nodes", "cores_rank")):
            raise ValueError("noncanonical no-cluster predictor metadata")
        return values, True
    if values["n_finite"] == 0 or values["cluster_nodes"] == 0:
        raise ValueError("clustered predictor dimensions must be positive")
    if (
        values["n_finite"] > np.iinfo(np.int32).max
        or values["cluster_nodes"] > np.iinfo(np.int32).max
    ):
        raise ValueError("predictor row and cluster counts exceed i32 limits")
    if values["cores_rank"] != values["min_samples"]:
        raise ValueError("predictor core rank disagrees with min_samples")
    if values["min_samples"] > values["n_finite"] // 2:
        raise ValueError("predictor min_samples exceeds candidate-width limit")
    lengths = {
        "geometry": values["n_finite"] * values["n_features"],
        "finite_index": values["n_finite"],
        "node_ids": values["cluster_nodes"],
        "node_parent_indices": values["cluster_nodes"],
        "node_birth_lambdas": values["cluster_nodes"],
        "node_labels": values["cluster_nodes"],
        "node_max_lambdas": values["cluster_nodes"],
        "point_node_indices": values["n_finite"],
        "point_departure_lambdas": values["n_finite"],
        "prediction_cores": values["n_finite"],
    }
    total = 0
    for name, length in lengths.items():
        if length <= 0 or length > _MAX_INTP:
            raise ValueError(f"invalid predictor dataset length: {name}")
        byte_count = length * np.dtype(_DTYPES[name]).itemsize
        if byte_count > _MAX_INTP or total > _MAX_INTP - byte_count:
            raise ValueError("predictor arrays exceed platform allocation limits")
        total += byte_count
    if check_arrays:
        for name, length in lengths.items():
            value = np.asarray(arrays[name])
            if value.ndim != 1 or value.size != length or value.dtype != np.dtype(_DTYPES[name]):
                raise ValueError(f"invalid predictor array dtype, rank or shape: {name}")
    return values, False


def save_model(
    path: str | os.PathLike[str],
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    values, _ = _state(metadata, arrays)

    def write(file: Any, _h5py: Any, plugin: Any) -> None:
        from . import _cuscan

        _common._write_root(file, _cuscan.__version__, "hdbscan_predictor")
        state = file.create_group("state")
        for name in ("n_features", "n_finite", "min_samples", "cluster_nodes", "cores_rank"):
            state.attrs[name] = np.uint64(values[name])
        state.attrs["cores_status"] = values["cores_status"]
        for name in _ARRAYS:
            value = arrays[name]
            state.attrs[f"{name}_present"] = np.bool_(value is not None)
            if value is not None:
                _common._create_dataset(state, name, value, _DTYPES[name], plugin)

    _common._publish(path, overwrite, write, "HDBSCAN predictor")


def load_model(
    cls: Any,
    path: str | os.PathLike[str],
    *,
    backend: str = "auto",
    device_index: int = 0,
) -> Any:
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu' or 'cuda'")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    h5py, _ = _deps()
    arrays: dict[str, Any] = {}
    with h5py.File(path, "r") as file:
        clevel = _common._validate_root(
            file, h5py, model_kind="hdbscan_predictor", groups={"state"}
        )
        if clevel != 1:
            raise ValueError("unsupported compression level")
        state = file["state"]
        if not isinstance(state, h5py.Group):
            raise ValueError("predictor state must be a group")
        expected_attrs = _META | {f"{name}_present" for name in _ARRAYS}
        if set(state.attrs) != expected_attrs or set(state.keys()) - set(_ARRAYS):
            raise ValueError("invalid predictor state attributes or datasets")
        metadata: dict[str, Any] = {}
        for name in ("n_features", "n_finite", "min_samples", "cluster_nodes", "cores_rank"):
            metadata[name] = _common._check_scalar(state, name, "<u8")
        metadata["cores_status"] = _common._check_text(state, "cores_status")
        presence: dict[str, bool] = {}
        for name in _ARRAYS:
            presence[name] = _common._check_scalar(state, f"{name}_present", "|b1")
            if presence[name] != (name in state):
                message = (
                    f"missing dataset {name}" if presence[name] else f"unexpected dataset {name}"
                )
                raise ValueError(message)
        metadata, no_tree = _state(
            metadata,
            {name: object() if present else None for name, present in presence.items()},
            check_arrays=False,
        )
        lengths = {}
        if not no_tree:
            lengths = {
                "geometry": metadata["n_finite"] * metadata["n_features"],
                "finite_index": metadata["n_finite"],
                "node_ids": metadata["cluster_nodes"],
                "node_parent_indices": metadata["cluster_nodes"],
                "node_birth_lambdas": metadata["cluster_nodes"],
                "node_labels": metadata["cluster_nodes"],
                "node_max_lambdas": metadata["cluster_nodes"],
                "point_node_indices": metadata["n_finite"],
                "point_departure_lambdas": metadata["n_finite"],
                "prediction_cores": metadata["n_finite"],
            }
        total = 0
        datasets: dict[str, Any] = {}
        for name, length in lengths.items():
            dataset = state[name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(f"invalid predictor dataset: {name}")
            if dataset.attrs:
                raise ValueError(f"unexpected predictor dataset attributes: {name}")
            if (
                dataset.dtype != np.dtype(_DTYPES[name])
                or dataset.ndim != 1
                or dataset.shape != (length,)
            ):
                raise ValueError(f"invalid predictor dataset dtype, rank or shape: {name}")
            if dataset.is_virtual or dataset.id.get_create_plist().get_external_count():
                raise ValueError("virtual and externally stored datasets are not permitted")
            plist = dataset.id.get_create_plist()
            expected_chunks = (min(length, _common._CHUNK_ROWS),)
            if plist.get_nfilters() != 1 or dataset.chunks != expected_chunks:
                raise ValueError(f"invalid predictor filter pipeline or chunks: {name}")
            _common._filters(dataset, clevel)
            byte_count = length * dataset.dtype.itemsize
            if byte_count > _MAX_INTP or total > _MAX_INTP - byte_count:
                raise ValueError("predictor arrays exceed platform allocation limits")
            total += byte_count
            datasets[name] = dataset
        for name in _ARRAYS:
            arrays[name] = datasets[name][...] if name in datasets else None
    # Check loaded arrays again for exact byte representation before crossing the native boundary.
    _state(metadata, arrays)
    from . import _cuscan

    native = _cuscan.load_hdbscan_predictor(
        arrays, metadata, backend=backend, device_index=device_index
    )
    return cls._from_native(native)
