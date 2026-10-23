from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

import fast_umap

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

UMAP = fast_umap.UMAP


def fitted(**kwargs: Any) -> UMAP:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    return UMAP(
        backend="cpu",
        neighbor_search="exact",
        init="random",
        learning_rate=0.0,
        n_epochs=20,
        **kwargs,
    ).fit(x)


def test_round_trip_keeps_native_state_and_fit_time_precision(tmp_path: Path) -> None:
    model = fitted(min_dist=0.123456789012345)
    transformed = model.transform(np.ones((3, 4), dtype=np.float32))
    model.min_dist = 0.8
    path = tmp_path / "model.h5"
    model.save(str(path))
    loaded = UMAP.load(str(path), backend="cpu")
    assert loaded.min_dist == 0.123456789012345
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    np.testing.assert_array_equal(loaded.transform(np.ones((3, 4), dtype=np.float32)), transformed)
    arrays, metadata = model._model.export_state()
    with h5py.File(path) as file:
        np.testing.assert_array_equal(
            file["state/training_features"][...], arrays["training_features"]
        )
        np.testing.assert_array_equal(file["state/embedding"][...], arrays["embedding"])
        assert metadata["requested_search"] == file["config"].attrs["requested_search"]
        dataset = file["state/training_features"]
        assert any(
            dataset.id.get_create_plist().get_filter(i)[0] == 32026
            for i in range(dataset.id.get_create_plist().get_nfilters())
        )


@pytest.mark.parametrize("search", ["exact", "approximate"])
def test_search_mode_round_trip(tmp_path: Path, search: Literal["exact", "approximate"]) -> None:
    rng = np.random.default_rng(8)
    x = rng.normal(size=(48, 3)).astype(np.float32)
    model = UMAP(
        backend="cpu", neighbor_search=search, init="random", n_epochs=15, learning_rate=0.0
    ).fit(x)
    arrays, metadata = model._model.export_state()
    path = tmp_path / f"{search}.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    restored_arrays, restored_metadata = restored._model.export_state()
    assert restored_metadata["requested_search"] == metadata["requested_search"]
    assert restored_metadata["resolved_search"] == metadata["resolved_search"]
    for name in ("training_features", "embedding", "query_graph", "original_to_training"):
        if arrays[name] is None:
            assert restored_arrays[name] is None
        else:
            np.testing.assert_array_equal(restored_arrays[name], arrays[name])


def test_singleton_round_trip_preserves_unavailable_transform(tmp_path: Path) -> None:
    model = UMAP(backend="cpu").fit(np.array([[2.0, 3.0]], dtype=np.float32))
    path = tmp_path / "singleton.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    np.testing.assert_array_equal(restored.embedding_, np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Transform unavailable"):
        restored.transform(np.array([[2.0, 3.0]], dtype=np.float32))


def test_unique_duplicate_rows_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(17)
    rows = rng.normal(size=(24, 3)).astype(np.float32)
    original = np.vstack((rows, rows[4], rows[0], rows[4]))
    model = UMAP(
        backend="cpu",
        unique=True,
        neighbor_search="exact",
        init="random",
        n_epochs=20,
        learning_rate=0.0,
    ).fit(original)
    path = tmp_path / "unique.h5"
    model.save(path)
    loaded = UMAP.load(path, backend="cpu")
    np.testing.assert_array_equal(loaded.embedding_, model.embedding_)
    query = original[[24, 25, 26]]
    np.testing.assert_array_equal(loaded.transform(query), model.transform(query))
    before, _ = model._model.export_state()
    after, _ = loaded._model.export_state()
    np.testing.assert_array_equal(after["original_to_training"], before["original_to_training"])


def test_independent_reader_loads_without_plugin_path(tmp_path: Path) -> None:
    path = tmp_path / "reader.h5"
    fitted().save(path)
    code = (
        "import h5py,hdf5plugin,sys; f=h5py.File(sys.argv[1]); "
        "import numpy as np; data=f['state/training_features'][...]; "
        "assert data.shape[0] == 30; assert np.isfinite(data).all(); "
        "assert np.any(data != 0); f.close()"
    )
    env = dict(__import__("os").environ)
    env.pop("HDF5_PLUGIN_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_device_index_forwarded_and_bool_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    path = tmp_path / "device.h5"
    fitted().save(path)
    real_import = fast_umap._fast_umap.load
    received: list[tuple[str, int]] = []

    def capture(arrays: Any, metadata: Any, backend: str, device_index: int) -> Any:
        received.append((backend, device_index))
        return real_import(arrays, metadata, backend="cpu", device_index=0)

    monkeypatch.setattr(fast_umap._fast_umap, "load", capture)
    UMAP.load(path, backend="cuda", device_index=7)
    assert received == [("cuda", 7)]
    with pytest.raises(ValueError, match="device_index"):
        UMAP.load(path, backend="cpu", device_index=True)


def test_cuda_load_forwards_device_index_strictly(tmp_path: Path) -> None:
    path = tmp_path / "cuda.h5"
    fitted().save(path)
    try:
        model = UMAP.load(path, backend="cuda", device_index=0)
    except RuntimeError as error:
        if "cuda" in str(error).lower() and any(
            phrase in str(error).lower()
            for phrase in (
                "unavailable",
                "no device",
            )
        ):
            pytest.skip(f"CUDA backend unavailable: {error}")
        raise
    assert model.device_index == 0
    with pytest.raises(RuntimeError):
        UMAP.load(path, backend="cuda", device_index=999)


def test_no_clobber_and_replace(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    fitted().save(str(path))
    with pytest.raises(FileExistsError):
        fitted().save(str(path))
    original = path.read_bytes()
    assert original
    fitted(min_dist=0.3).save(str(path), overwrite=True)
    assert UMAP.load(str(path), backend="cpu").min_dist == 0.3


def test_rejects_bad_version_and_external_links(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    fitted().save(str(path))
    with h5py.File(path, "r+") as file:
        file.attrs["schema_version"] = 99
    with pytest.raises(ValueError, match="version"):
        UMAP.load(str(path))
    with h5py.File(path, "w") as file:
        file["external"] = h5py.ExternalLink("other.h5", "/target")
    with pytest.raises(ValueError, match="links"):
        UMAP.load(str(path))


def test_rejects_corrupt_dataset_before_native_import(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    fitted().save(str(path))
    with h5py.File(path, "r+") as file:
        del file["state/embedding"]
    with pytest.raises(ValueError, match="missing dataset"):
        UMAP.load(str(path))


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_flag",
        "fixed_width",
        "optional_payload",
        "unknown_root",
        "unknown_state",
        "producer_missing",
        "producer_rank",
        "hardlink_cycle",
    ],
)
def test_rejects_malformed_schema_entries(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "malformed.h5"
    fitted().save(path)
    with h5py.File(path, "r+") as file:
        if mutation == "bool_flag":
            file["config"].attrs["n_epochs_is_auto"] = "False"
        elif mutation == "fixed_width":
            file["config"].attrs["n_components"] = np.uint32(2)
        elif mutation == "optional_payload":
            file["config"].attrs["random_state_value"] = np.uint64(22)
        elif mutation == "unknown_root":
            file.attrs["unexpected"] = 1
        elif mutation == "unknown_state":
            file["state"].create_dataset("unexpected", data=np.ones(1))
        elif mutation == "producer_missing":
            del file.attrs["producer_version"]
        elif mutation == "producer_rank":
            file.attrs["producer_version"] = np.array(["v1", "v2"], dtype="S2")
        else:
            file["state/unexpected"] = file["state"]
    with pytest.raises(ValueError):
        UMAP.load(path)


def test_invalid_filter_settings_rejected(tmp_path: Path) -> None:
    path = tmp_path / "filter.h5"
    fitted().save(path)
    with h5py.File(path, "r+") as file:
        file["config"].attrs["min_dist"] = np.float32(0.9)
        dataset = file["state/training_features"]
        del file["state/training_features"]
        value = dataset[...]
        file["state"].create_dataset(
            "training_features",
            data=value,
            chunks=True,
            **hdf5plugin.Blosc2(cname="lz4", filters=hdf5plugin.Blosc2.SHUFFLE, clevel=1),
        )
    with pytest.raises(ValueError, match=r"filter|Blosc"):
        UMAP.load(path)


def test_bad_precision_and_filter_options_rejected(tmp_path: Path) -> None:
    path = tmp_path / "model.h5"
    fitted().save(str(path))
    with h5py.File(path, "r+") as file:
        file["python_precision"].attrs["min_dist"] = 0.9
    with pytest.raises(ValueError, match="precision"):
        UMAP.load(str(path))


def test_optional_dependency_error_is_lazy() -> None:
    code = """import builtins
old = builtins.__import__
def blocked(name, *args, **kwargs):
    if name in ('h5py', 'hdf5plugin'):
        raise ImportError()
    return old(name, *args, **kwargs)
builtins.__import__ = blocked
from fast_umap._serialization_v1 import _deps
try:
    _deps()
except ImportError as error:
    assert 'serialization' in str(error)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_save_temp_cleanup_after_existing_destination(tmp_path: Path) -> None:
    path = tmp_path / "existing"
    path.write_text("preserve")
    with pytest.raises(FileExistsError):
        fitted().save(str(path))
    assert path.read_text() == "preserve"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["existing"]


def test_no_clobber_race_preserves_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_umap import _serialization_v1

    path = tmp_path / "raced.h5"
    actual_link = _serialization_v1.os.link

    def race(source: str, destination: str) -> None:
        Path(destination).write_text("racing writer")
        actual_link(source, destination)

    monkeypatch.setattr(_serialization_v1.os, "link", race)
    with pytest.raises(FileExistsError):
        fitted().save(str(path))
    assert path.read_text() == "racing writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raced.h5"]


@pytest.mark.parametrize("failure", ["write", "close", "fsync", "link", "replace"])
def test_precommit_failures_clean_temporary_and_preserve_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from fast_umap import _serialization_v1

    model = fitted()
    destination = tmp_path / "target.h5"
    overwrite = failure == "replace"
    if overwrite:
        destination.write_text("old destination")
    if failure == "write":
        monkeypatch.setattr(
            h5py.Group,
            "create_dataset",
            lambda *a, **k: (_ for _ in ()).throw(OSError("write failed")),
        )
    elif failure == "close":
        real_file = h5py.File

        class CloseFailure:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.file = real_file(*args, **kwargs)

            def __enter__(self) -> Any:
                return self.file.__enter__()

            def __exit__(self, *args: Any) -> None:
                self.file.__exit__(*args)
                raise OSError("close failed")

        monkeypatch.setattr(h5py, "File", CloseFailure)
    else:
        operation = {"fsync": "fsync", "link": "link", "replace": "replace"}[failure]

        def fail(*args: Any, **kwargs: Any) -> None:
            raise OSError(f"{failure} failed")

        monkeypatch.setattr(_serialization_v1.os, operation, fail)
    with pytest.raises(OSError):
        model.save(destination, overwrite=overwrite)
    if overwrite:
        assert destination.read_text() == "old destination"
    else:
        assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == (["target.h5"] if overwrite else [])


def test_postcommit_directory_fsync_failure_identifies_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from fast_umap import _serialization_v1

    path = tmp_path / "committed.h5"
    real_fsync = os.fsync
    calls = 0

    def fail_directory(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(_serialization_v1.os, "fsync", fail_directory)
    with pytest.raises(_serialization_v1.DurabilityUncertainError, match="committed") as error:
        fitted().save(path)
    assert isinstance(error.value.__cause__, OSError)
    assert path.is_file()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["committed.h5"]
