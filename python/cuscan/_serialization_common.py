from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_FORMAT = "cuscan_fitted_model"
_VERSION = 1
_FILTER_ID = 32026
_CHUNK_ROWS = 32_768


class DurabilityUncertainError(OSError):
    """The destination was published, but directory durability could not be confirmed."""


def _deps(model: str) -> tuple[Any, Any]:
    try:
        import h5py
        import hdf5plugin
    except ImportError as error:
        raise ImportError(
            f"{model} serialization requires cuscan[serialization] (h5py and hdf5plugin)"
        ) from error
    return h5py, hdf5plugin


def _filters(dataset: Any, clevel: int) -> None:
    plist = dataset.id.get_create_plist()
    found = False
    for index in range(plist.get_nfilters()):
        filter_id, _, values, _ = plist.get_filter(index)
        if filter_id == _FILTER_ID:
            found = True
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
        if isinstance(current, h5py.Group):
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


def _attr(group: Any, name: str) -> Any:
    if name not in group.attrs:
        raise ValueError(f"missing attribute {name}")
    value = group.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if isinstance(value, np.generic) else value


def _validate_root(file: Any, h5py: Any, *, model_kind: str, groups: set[str]) -> int:
    _validate_links(file, h5py)
    expected_attrs = {
        "format",
        "schema_version",
        "producer_version",
        "model_kind",
        "filter_id",
        "codec",
        "shuffle",
        "clevel",
    }
    if set(file.attrs) != expected_attrs:
        raise ValueError("invalid root schema attributes")
    if _check_text(file, "producer_version") == "":
        raise ValueError("invalid producer_version")
    _check_scalar(file, "schema_version", "<u4")
    _check_scalar(file, "filter_id", "<u4")
    _check_scalar(file, "clevel", "|u1")
    for name in ("format", "model_kind", "codec", "shuffle"):
        _check_text(file, name)
    found_kind = _attr(file, "model_kind")
    if found_kind != model_kind:
        raise ValueError(f"model kind mismatch: found {found_kind!r}, expected {model_kind!r}")
    if set(file.keys()) != groups:
        raise ValueError("invalid schema groups")
    if _attr(file, "format") != _FORMAT:
        raise ValueError("unsupported serialization format")
    if _attr(file, "schema_version") != _VERSION:
        raise ValueError("unsupported serialization schema version")
    if (_attr(file, "filter_id"), _attr(file, "codec"), _attr(file, "shuffle")) != (
        _FILTER_ID,
        "zstd",
        "shuffle",
    ):
        raise ValueError("unsupported compression filter")
    clevel = int(_attr(file, "clevel"))
    return clevel


def _write_root(file: Any, producer_version: str, model_kind: str) -> None:
    file.attrs.update(
        format=_FORMAT,
        schema_version=np.uint32(_VERSION),
        producer_version=producer_version,
        model_kind=model_kind,
        filter_id=np.uint32(_FILTER_ID),
        codec="zstd",
        shuffle="shuffle",
        clevel=np.uint8(1),
    )


def _create_dataset(group: Any, name: str, value: Any, dtype: str, plugin: Any) -> None:
    array = np.asarray(value, dtype=dtype, order="C")
    if array.ndim == 0 or array.size == 0:
        raise ValueError(f"invalid empty array {name}")
    chunks = (min(array.shape[0], _CHUNK_ROWS), *array.shape[1:])
    group.create_dataset(
        name,
        data=array,
        dtype=dtype,
        chunks=chunks,
        **plugin.Blosc2(cname="zstd", filters=plugin.Blosc2.SHUFFLE, clevel=1),
    )


def _sync_directory(parent: Path) -> None:
    # Windows cannot fsync a directory through Python's CRT descriptors. There,
    # publication is atomic and file data is flushed, but directory-entry
    # durability across a crash is not guaranteed.
    if os.name == "nt":
        return
    directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish(
    path: str | os.PathLike[str],
    overwrite: bool,
    writer: Callable[[Any, Any, Any], None],
    model: str,
) -> None:
    h5py, plugin = _deps(model)
    destination = Path(path)
    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(parent)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=parent)
    os.close(fd)
    try:
        with h5py.File(temporary, "w") as file:
            writer(file, h5py, plugin)
        # Windows _commit (used by os.fsync) requires a writable CRT descriptor;
        # the HDF5 VFD handle is not necessarily a Python CRT descriptor.
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
        try:
            _sync_directory(parent)
        except OSError as error:
            raise DurabilityUncertainError(
                f"model is committed at {destination}, but directory durability is uncertain"
            ) from error
    finally:
        if os.path.exists(temporary):
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
