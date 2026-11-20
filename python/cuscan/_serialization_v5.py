from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import _serialization_common as _common

DurabilityUncertainError = _common.DurabilityUncertainError
_FORMAT = _common._FORMAT
_VERSION = 5

_FLOATS = (
    "min_dist",
    "spread",
    "local_connectivity",
    "set_op_mix_ratio",
    "learning_rate",
    "repulsion_strength",
    "init_scale_factor",
)
_CONFIG = (
    "n_components",
    "n_neighbors",
    "min_dist",
    "spread",
    "local_connectivity",
    "set_op_mix_ratio",
    "learning_rate",
    "negative_sample_rate",
    "repulsion_strength",
    "init",
    "init_rescale",
    "init_scale_factor",
    "transform_seed",
    "transform_queue_size",
    "unique",
    "requested_search",
    "resolved_search",
    "training_rows",
    "n_features",
    "query_graph_width",
    "optimizer",
    "negative_selection_range",
    "exclude_graph_neighbors",
    "negative_sample_scale_adaptation_samples",
    "recursive_coarsening_ratio",
    "recursive_base_init_threshold",
    "recursive_n_embedding_epochs",
    "recursive_hub_selection",
    "recursive_anchor_sample_size",
)
_ARRAYS = ("training_features", "embedding", "query_graph", "original_to_training")
_CONSTRUCTOR = (
    "n_components",
    "n_neighbors",
    "min_dist",
    "spread",
    "local_connectivity",
    "set_op_mix_ratio",
    "n_epochs",
    "learning_rate",
    "negative_sample_rate",
    "repulsion_strength",
    "init",
    "init_rescale",
    "random_state",
    "transform_seed",
    "transform_queue_size",
    "unique",
    "optimizer",
    "negative_selection_range",
    "exclude_graph_neighbors",
    "negative_sample_scale_adaptation_samples",
)
_I32_MAX = 2**31 - 1


def _deps() -> tuple[Any, Any]:
    return _common._deps("UMAP")


def _native_f32(value: Any, name: str) -> np.float32:
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"invalid native float metadata: {name}")
    return result


def _validate_precision(metadata: dict[str, Any], precision: dict[str, Any]) -> None:
    for key in _FLOATS:
        expected_native = _native_f32(precision[key], key)
        if expected_native.view(np.uint32) != np.asarray(metadata[key], dtype="<f4").view(
            np.uint32
        ):
            raise ValueError(f"Python precision metadata disagrees with native config: {key}")
    if not np.isfinite(precision["init_scale_factor"]) or precision["init_scale_factor"] <= 0:
        raise ValueError("invalid Python precision init_scale_factor")
    scale = precision["negative_sample_scale"]
    native_scale = metadata["negative_sample_scale"]
    if (scale is None) != (native_scale is None):
        raise ValueError("Python negative_sample_scale disagrees with native config")
    if scale is not None:
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("invalid Python precision negative_sample_scale")
        rounded = _native_f32(scale, "negative_sample_scale")
        if (scale != 0 and rounded == 0) or rounded.view(np.uint32) != np.asarray(
            native_scale, dtype="<f4"
        ).view(np.uint32):
            raise ValueError(
                "Python precision metadata disagrees with native config: negative_sample_scale"
            )


def _validate_links(group: Any, h5py: Any) -> None:
    _common._validate_links(group, h5py)


def _check_scalar(group: Any, name: str, dtype: str) -> Any:
    return _common._check_scalar(group, name, dtype)


def _check_text(group: Any, name: str) -> str:
    return _common._check_text(group, name)


def _validate_config(config: Any, precision: Any, curve: Any) -> None:
    for name in (
        "n_components",
        "n_neighbors",
        "negative_sample_rate",
        "transform_seed",
        "n_epochs_value",
        "random_state_value",
        "memory_limit_bytes_value",
        "training_rows",
        "n_features",
        "query_graph_width",
        "negative_selection_range",
        "negative_sample_scale_adaptation_samples",
        "recursive_coarsening_ratio",
        "recursive_base_init_threshold",
        "recursive_n_embedding_epochs",
        "recursive_anchor_sample_size",
    ):
        _check_scalar(config, name, "<u8")
    for name in _FLOATS:
        _check_scalar(config, name, "<f4")
        _check_scalar(precision, name, "<f8")
    _check_scalar(config, "transform_queue_size", "<f8")
    _check_scalar(config, "negative_sample_scale_value", "<f4")
    _check_scalar(precision, "negative_sample_scale", "<f8")
    for name in (
        "unique",
        "exclude_graph_neighbors",
        "n_epochs_is_auto",
        "random_state_is_none",
        "memory_limit_bytes_is_none",
        "negative_sample_scale_is_auto",
    ):
        _check_scalar(config, name, "|b1")
    for name in (
        "init",
        "init_rescale",
        "requested_search",
        "resolved_search",
        "optimizer",
        "recursive_hub_selection",
    ):
        text = _check_text(config, name)
        options = {
            "init": {"random", "spectral", "recursive", "auto"},
            "init_rescale": {"isotropic", "per_axis", "graph_edge"},
            "requested_search": {"auto", "exact", "approximate"},
            "resolved_search": {"exact", "approximate"},
            "optimizer": {"compatibility", "adam", "momentum"},
            "recursive_hub_selection": {"degree", "diverse", "random"},
        }[name]
        if text not in options:
            raise ValueError(f"invalid enum value: {name}")
    factor = _attr(config, "init_scale_factor")
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError("invalid init_scale_factor")
    if not 0 < _attr(config, "negative_selection_range") <= 2**31 - 1:
        raise ValueError("invalid negative_selection_range")
    if not 0 <= _attr(config, "negative_sample_scale_adaptation_samples") <= _I32_MAX:
        raise ValueError("invalid negative_sample_scale_adaptation_samples")
    if _attr(config, "recursive_coarsening_ratio") not in (2, 3, 4):
        raise ValueError("invalid recursive_coarsening_ratio")
    if not 2 <= _attr(config, "recursive_base_init_threshold") <= _I32_MAX:
        raise ValueError("invalid recursive_base_init_threshold")
    if _attr(config, "recursive_n_embedding_epochs") < 1:
        raise ValueError("invalid recursive_n_embedding_epochs")
    if not 1 <= _attr(config, "recursive_anchor_sample_size") <= _I32_MAX:
        raise ValueError("invalid recursive_anchor_sample_size")
    scale = _attr(config, "negative_sample_scale_value")
    py_scale = _attr(precision, "negative_sample_scale")
    auto = _attr(config, "negative_sample_scale_is_auto")
    if not np.isfinite(scale) or scale < 0 or not np.isfinite(py_scale) or py_scale < 0:
        raise ValueError("invalid negative_sample_scale")
    if auto and (scale != 0 or py_scale != 0 or np.signbit(scale) or np.signbit(py_scale)):
        raise ValueError("noncanonical negative_sample_scale auto payload")
    for name in ("a", "b"):
        _check_scalar(curve, name, "<f4")


def _attr(group: Any, name: str) -> Any:
    return _common._attr(group, name)


def save_model(
    path: str | os.PathLike[str],
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    precision: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    def write(file: Any, h5py: Any, plugin: Any) -> None:
        from . import _cuscan

        for key in ("optimizer", "negative_selection_range"):
            if key not in metadata or metadata[key] != precision[key]:
                raise ValueError(f"Python metadata disagrees with native config: {key}")
        _common._write_root(file, _cuscan.__version__, "standard_umap", schema_version=_VERSION)
        config = file.create_group("config")
        for key in _CONFIG:
            if key not in metadata:
                raise ValueError(f"missing native metadata {key}")
            value = metadata[key]
            if key in _FLOATS:
                value = np.asarray(value, dtype="<f4")
            elif key == "transform_queue_size":
                value = np.asarray(value, dtype="<f8")
            elif key in ("unique", "exclude_graph_neighbors"):
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"invalid native bool metadata: {key}")
                value = np.bool_(value)
            elif key in (
                "negative_sample_scale_adaptation_samples",
                "recursive_coarsening_ratio",
                "recursive_base_init_threshold",
                "recursive_n_embedding_epochs",
                "recursive_anchor_sample_size",
            ):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not 0 <= int(value) <= (2**64 - 1)
                ):
                    raise ValueError(f"invalid native integer metadata: {key}")
                value = np.uint64(value)
            elif key in (
                "requested_search",
                "resolved_search",
                "init",
                "init_rescale",
                "optimizer",
                "recursive_hub_selection",
            ):
                if not isinstance(value, str):
                    raise ValueError(f"invalid native text metadata: {key}")
            else:
                value = np.uint64(value)
            config.attrs[key] = value
        config.attrs["n_epochs_is_auto"] = np.bool_(metadata["n_epochs"] is None)
        config.attrs["n_epochs_value"] = np.uint64(
            0 if metadata["n_epochs"] is None else metadata["n_epochs"]
        )
        config.attrs["random_state_is_none"] = np.bool_(metadata["random_state"] is None)
        config.attrs["random_state_value"] = np.uint64(
            0 if metadata["random_state"] is None else metadata["random_state"]
        )
        if metadata["memory_limit_bytes"] != precision["memory_limit_bytes"]:
            raise ValueError("Python memory limit disagrees with native config")
        config.attrs["memory_limit_bytes_is_none"] = np.bool_(
            metadata["memory_limit_bytes"] is None
        )
        config.attrs["memory_limit_bytes_value"] = np.uint64(
            0 if metadata["memory_limit_bytes"] is None else metadata["memory_limit_bytes"]
        )
        py_config = file.create_group("python_precision")
        for key in _FLOATS:
            py_config.attrs[key] = np.asarray(precision[key], dtype="<f8")
        scale = metadata["negative_sample_scale"]
        py_scale = precision["negative_sample_scale"]
        config.attrs["negative_sample_scale_is_auto"] = np.bool_(scale is None)
        config.attrs["negative_sample_scale_value"] = np.asarray(
            0 if scale is None else scale, dtype="<f4"
        )
        py_config.attrs["negative_sample_scale"] = np.asarray(
            0 if py_scale is None else py_scale, dtype="<f8"
        )
        curve = file.create_group("curve")
        curve.attrs["a"] = np.asarray(metadata["a"], dtype="<f4")
        curve.attrs["b"] = np.asarray(metadata["b"], dtype="<f4")
        _validate_config(config, py_config, curve)
        _validate_precision(metadata, precision)
        state = file.create_group("state")
        for name in _ARRAYS:
            value = arrays.get(name)
            if value is None:
                state.attrs[f"{name}_present"] = np.bool_(False)
                continue
            state.attrs[f"{name}_present"] = np.bool_(True)
            dtype = {
                "training_features": "<f4",
                "embedding": "<f4",
                "query_graph": "<i4",
                "original_to_training": "<u8",
            }[name]
            _common._create_dataset(state, name, value, dtype, plugin)

    _common._publish(path, overwrite, write, "UMAP")


def load_model(
    cls: type, path: str | os.PathLike[str], *, backend: str = "auto", device_index: int = 0
) -> Any:
    h5py, _ = _deps()
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu' or 'cuda'")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    with h5py.File(path, "r") as file:
        clevel = _common._validate_root(
            file,
            h5py,
            model_kind="standard_umap",
            groups={"config", "python_precision", "curve", "state"},
            schema_version=_VERSION,
        )
        config, precision, curve, state = (
            file[name] for name in ("config", "python_precision", "curve", "state")
        )
        if config.keys() or precision.keys() or curve.keys():
            raise ValueError("unexpected objects in metadata groups")
        _validate_config(config, precision, curve)
        expected_config = set(_CONFIG) | {
            "n_epochs_is_auto",
            "n_epochs_value",
            "random_state_is_none",
            "random_state_value",
            "memory_limit_bytes_is_none",
            "memory_limit_bytes_value",
            "negative_sample_scale_is_auto",
            "negative_sample_scale_value",
        }
        if (
            set(config.attrs) != expected_config
            or set(precision.attrs) != set(_FLOATS) | {"negative_sample_scale"}
            or set(curve.attrs) != {"a", "b"}
        ):
            raise ValueError("invalid schema attributes")
        metadata = {key: _attr(config, key) for key in _CONFIG}
        metadata["exclude_graph_neighbors"] = bool(metadata["exclude_graph_neighbors"])
        metadata["negative_sample_scale_adaptation_samples"] = int(
            metadata["negative_sample_scale_adaptation_samples"]
        )
        scale_is_auto = _attr(config, "negative_sample_scale_is_auto")
        metadata["negative_sample_scale"] = (
            None if scale_is_auto else float(_attr(config, "negative_sample_scale_value"))
        )
        metadata["n_epochs"] = (
            None if _attr(config, "n_epochs_is_auto") else int(_attr(config, "n_epochs_value"))
        )
        metadata["random_state"] = (
            None
            if _attr(config, "random_state_is_none")
            else int(_attr(config, "random_state_value"))
        )
        if (_attr(config, "n_epochs_is_auto") and _attr(config, "n_epochs_value") != 0) or (
            _attr(config, "random_state_is_none") and _attr(config, "random_state_value") != 0
        ):
            raise ValueError("noncanonical optional-value payload")
        memory_is_none = _attr(config, "memory_limit_bytes_is_none")
        memory_value = int(_attr(config, "memory_limit_bytes_value"))
        if (memory_is_none and memory_value != 0) or (not memory_is_none and memory_value == 0):
            raise ValueError("noncanonical memory_limit_bytes payload")
        metadata["memory_limit_bytes"] = None if memory_is_none else memory_value
        for key in ("a", "b"):
            metadata[key] = _attr(curve, key)
        precision_values: dict[str, float | None] = {
            key: float(_attr(precision, key)) for key in _FLOATS
        }
        precision_values["negative_sample_scale"] = (
            None if scale_is_auto else float(_attr(precision, "negative_sample_scale"))
        )
        _validate_precision(metadata, precision_values)
        arrays: dict[str, Any] = {}
        expected = {
            "training_features": ("<f4", 2),
            "embedding": ("<f4", 2),
            "query_graph": ("<i4", 2),
            "original_to_training": ("<u8", 1),
        }
        if set(state.attrs) != {f"{name}_present" for name in _ARRAYS}:
            raise ValueError("invalid fitted-state attributes")
        if set(state.keys()) - set(_ARRAYS):
            raise ValueError("unknown fitted-state objects")
        for name in _ARRAYS:
            present = _check_scalar(state, f"{name}_present", "|b1")
            if present:
                if name not in state:
                    raise ValueError(f"missing dataset {name}")
                dataset = state[name]
                dtype, rank = expected[name]
                if (
                    dataset.dtype != np.dtype(dtype)
                    or dataset.ndim != rank
                    or any(size <= 0 for size in dataset.shape)
                ):
                    raise ValueError(f"invalid dataset dtype, rank or shape: {name}")
                if dataset.is_virtual or dataset.id.get_create_plist().get_external_count():
                    raise ValueError("virtual and externally stored datasets are not permitted")
                if name == "training_features" and dataset.shape != (
                    int(metadata["training_rows"]),
                    int(metadata["n_features"]),
                ):
                    raise ValueError("training feature shape disagrees with metadata")
                if name == "embedding" and dataset.shape != (
                    int(metadata["training_rows"]),
                    int(metadata["n_components"]),
                ):
                    raise ValueError("embedding shape disagrees with metadata")
                if name == "query_graph" and dataset.shape != (
                    int(metadata["training_rows"]),
                    int(metadata["query_graph_width"]),
                ):
                    raise ValueError("query graph shape disagrees with metadata")
                _common._filters(dataset, clevel)
                arrays[name] = dataset[...]
            else:
                if name in state:
                    raise ValueError(f"unexpected dataset {name}")
                arrays[name] = None
    features, embedding = arrays["training_features"], arrays["embedding"]
    n, d = features.shape
    if (
        embedding.shape != (n, int(metadata["n_components"]))
        or n != int(metadata["training_rows"])
        or d != int(metadata["n_features"])
    ):
        raise ValueError("fitted array shapes disagree with metadata")
    graph = arrays["query_graph"]
    if (graph is None and int(metadata["query_graph_width"]) != 0) or (
        graph is not None and graph.shape != (n, int(metadata["query_graph_width"]))
    ):
        raise ValueError("query graph shape disagrees with metadata")
    mapping = arrays["original_to_training"]
    if mapping is not None and (
        mapping.size > np.iinfo(np.intp).max
        or np.any(mapping >= n)
        or np.any(mapping > np.iinfo(np.int32).max)
    ):
        raise ValueError("invalid inverse mapping value")
    from . import _cuscan

    native = _cuscan.load(arrays, metadata, backend=backend, device_index=device_index)
    constructor = {key: metadata[key] for key in _CONSTRUCTOR if key not in _FLOATS}
    constructor.update(precision_values)
    constructor.update(
        backend=backend,
        device_index=device_index,
        neighbor_search=metadata["requested_search"],
        memory_limit_bytes=None if memory_is_none else memory_value,
        recursive_coarsening_ratio=int(metadata["recursive_coarsening_ratio"]),
        recursive_init_kwds={
            "base_init_threshold": int(metadata["recursive_base_init_threshold"]),
            "n_embedding_epochs": int(metadata["recursive_n_embedding_epochs"]),
            "hub_selection": metadata["recursive_hub_selection"],
            "anchor_sample_size": int(metadata["recursive_anchor_sample_size"]),
        },
    )
    model = cls(**constructor)
    model._model = native
    model.embedding_ = native.embedding
    model.n_features_in_ = d
    model._fit_config = {
        **metadata,
        **precision_values,
        "memory_limit_bytes": model.memory_limit_bytes,
    }
    return model
