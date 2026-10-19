from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "fast_umap_fitted_model"
# Rows per HDF5 chunk; trailing dimensions are stored whole.
_CHUNK_ROWS = 32_768
_VERSION = 1


class DurabilityUncertainError(OSError):
    """The destination was published, but directory durability could not be confirmed."""


_FLOATS = (
    "min_dist",
    "spread",
    "local_connectivity",
    "set_op_mix_ratio",
    "learning_rate",
    "repulsion_strength",
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
    "transform_seed",
    "transform_queue_size",
    "unique",
    "requested_search",
    "resolved_search",
    "training_rows",
    "n_features",
    "query_graph_width",
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
    "random_state",
    "transform_seed",
    "transform_queue_size",
    "unique",
)


def _deps() -> tuple[Any, Any]:
    try:
        import h5py
        import hdf5plugin
    except ImportError as error:
        raise ImportError(
            "UMAP serialization requires fast-umap[serialization] (h5py and hdf5plugin)"
        ) from error
    return h5py, hdf5plugin


def _native_f32(value: Any, name: str) -> np.float32:
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"invalid native float metadata: {name}")
    return result


def _filters(dataset: Any, h5py: Any, clevel: int) -> None:
    plist = dataset.id.get_create_plist()
    found = False
    for index in range(plist.get_nfilters()):
        filter_id, _, values, _ = plist.get_filter(index)
        if filter_id == 32026:
            found = True
            # hdf5plugin.Blosc2 stores level, shuffle, and codec at indices 4, 5, and 6.
            if len(values) < 7 or values[4] != clevel or values[5] != 1 or values[6] != 5:
                raise ValueError("invalid Blosc2 filter options")
    if not found:
        raise ValueError("missing required Blosc2 filter")


def _validate_links(group: Any, h5py: Any) -> None:
    visited: set[int] = set()

    def visit(current: Any) -> None:
        identity = h5py.h5o.get_info(current.id).addr
        if identity in visited:
            raise ValueError("hard-link aliases and cycles are not permitted")
        visited.add(identity)
        if not isinstance(current, h5py.Group):
            return
        for name in current:
            link = current.get(name, getlink=True)
            if isinstance(link, (h5py.ExternalLink, h5py.SoftLink)):
                raise ValueError("external and soft links are not permitted")
            visit(current[name])

    visit(group)


def _check_scalar(group: Any, name: str, dtype: str) -> Any:
    if name not in group.attrs:
        raise ValueError(f"missing attribute {name}")
    value = group.attrs[name]
    if np.asarray(value).ndim != 0 or np.asarray(value).dtype != np.dtype(dtype):
        raise ValueError(f"invalid scalar attribute dtype or rank: {name}")
    return _attr(group, name)


def _check_text(group: Any, name: str) -> str:
    value = group.attrs.get(name)
    if not isinstance(value, (str, bytes)) or np.asarray(value).ndim != 0:
        raise ValueError(f"invalid text attribute: {name}")
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _validate_config(config: Any, precision: Any, curve: Any) -> None:
    for name in (
        "n_components",
        "n_neighbors",
        "negative_sample_rate",
        "transform_seed",
        "n_epochs_value",
        "random_state_value",
        "training_rows",
        "n_features",
        "query_graph_width",
    ):
        _check_scalar(config, name, "<u8")
    for name in _FLOATS:
        _check_scalar(config, name, "<f4")
        _check_scalar(precision, name, "<f8")
    _check_scalar(config, "transform_queue_size", "<f8")
    for name in ("unique", "n_epochs_is_auto", "random_state_is_none"):
        _check_scalar(config, name, "|b1")
    for name in ("init", "requested_search", "resolved_search"):
        text = _check_text(config, name)
        options = {
            "init": {"random", "spectral"},
            "requested_search": {"auto", "exact", "approximate"},
            "resolved_search": {"exact", "approximate"},
        }[name]
        if text not in options:
            raise ValueError(f"invalid enum value: {name}")
    for name in ("a", "b"):
        _check_scalar(curve, name, "<f4")


def _attr(group: Any, name: str) -> Any:
    if name not in group.attrs:
        raise ValueError(f"missing attribute {name}")
    value = group.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_model(
    path: str | os.PathLike[str],
    arrays: dict[str, Any],
    metadata: dict[str, Any],
    precision: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    h5py, plugin = _deps()
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    clevel = 1
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    os.close(fd)
    committed = False
    try:
        with h5py.File(temporary, "w") as file:
            from . import _fast_umap

            file.attrs.update(
                format=_FORMAT,
                schema_version=np.uint32(_VERSION),
                producer_version=_fast_umap.__version__,
                model_kind="standard_umap",
                filter_id=np.uint32(32026),
                codec="zstd",
                shuffle="shuffle",
                clevel=np.uint8(clevel),
            )
            config = file.create_group("config")
            for key in _CONFIG:
                if key not in metadata:
                    raise ValueError(f"missing native metadata {key}")
                value = metadata[key]
                if key in _FLOATS:
                    value = np.asarray(value, dtype="<f4")
                elif key == "transform_queue_size":
                    value = np.asarray(value, dtype="<f8")
                elif key in ("unique",):
                    value = np.bool_(value)
                elif key in ("requested_search", "resolved_search", "init"):
                    value = str(value)
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
            py_config = file.create_group("python_precision")
            for key in _FLOATS:
                py_config.attrs[key] = np.asarray(precision[key], dtype="<f8")
            curve = file.create_group("curve")
            curve.attrs["a"] = np.asarray(metadata["a"], dtype="<f4")
            curve.attrs["b"] = np.asarray(metadata["b"], dtype="<f4")
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
                value = np.asarray(value, dtype=dtype, order="C")
                if value.ndim == 0 or value.size == 0:
                    raise ValueError(f"invalid empty array {name}")
                chunks = (min(value.shape[0], _CHUNK_ROWS), *value.shape[1:])
                state.create_dataset(
                    name,
                    data=value,
                    dtype=dtype,
                    chunks=chunks,
                    **plugin.Blosc2(cname="zstd", filters=plugin.Blosc2.SHUFFLE, clevel=clevel),
                )
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        committed = True
        try:
            directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise DurabilityUncertainError(
                f"model is committed at {destination}, but directory durability is uncertain"
            ) from error
    finally:
        if not committed or os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_model(
    cls: type, path: str | os.PathLike[str], *, backend: str = "auto", device_index: int = 0
) -> Any:
    h5py, _ = _deps()
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError("backend must be 'auto', 'cpu' or 'cuda'")
    if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
        raise ValueError("device_index must be a non-negative integer")
    with h5py.File(path, "r") as file:
        _validate_links(file, h5py)
        if set(file.keys()) != {"config", "python_precision", "curve", "state"}:
            raise ValueError("invalid schema groups")
        root_attrs = {
            "format",
            "schema_version",
            "producer_version",
            "model_kind",
            "filter_id",
            "codec",
            "shuffle",
            "clevel",
        }
        if set(file.attrs) != root_attrs:
            raise ValueError("invalid root schema attributes")
        if _check_text(file, "producer_version") == "":
            raise ValueError("invalid producer_version")
        _check_scalar(file, "schema_version", "<u4")
        _check_scalar(file, "filter_id", "<u4")
        _check_scalar(file, "clevel", "|u1")
        for name in ("format", "model_kind", "codec", "shuffle"):
            _check_text(file, name)
        if _attr(file, "format") != _FORMAT or _attr(file, "model_kind") != "standard_umap":
            raise ValueError("not a fast_umap fitted model")
        if _attr(file, "schema_version") != _VERSION:
            raise ValueError("unsupported serialization schema version")
        if (_attr(file, "filter_id"), _attr(file, "codec"), _attr(file, "shuffle")) != (
            32026,
            "zstd",
            "shuffle",
        ):
            raise ValueError("unsupported compression filter")
        clevel = int(_attr(file, "clevel"))
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
        }
        if (
            set(config.attrs) != expected_config
            or set(precision.attrs) != set(_FLOATS)
            or set(curve.attrs) != {"a", "b"}
        ):
            raise ValueError("invalid schema attributes")
        metadata = {key: _attr(config, key) for key in _CONFIG}
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
        for key in ("a", "b"):
            metadata[key] = _attr(curve, key)
        precision_values = {key: float(_attr(precision, key)) for key in _FLOATS}
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
                _filters(dataset, h5py, clevel)
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
    for key in _FLOATS:
        expected_native = _native_f32(precision_values[key], key)
        if expected_native.view(np.uint32) != np.asarray(metadata[key], dtype="<f4").view(
            np.uint32
        ):
            raise ValueError(f"Python precision metadata disagrees with native config: {key}")
    from . import _fast_umap

    native = _fast_umap.load(arrays, metadata, backend=backend, device_index=device_index)
    constructor = {key: metadata[key] for key in _CONSTRUCTOR if key not in _FLOATS}
    constructor.update(precision_values)
    constructor.update(
        backend=backend, device_index=device_index, neighbor_search=metadata["requested_search"]
    )
    model = cls(**constructor)
    model._model = native
    model.embedding_ = native.embedding
    model.n_features_in_ = d
    model._fit_config = {**metadata, **precision_values}
    return model
