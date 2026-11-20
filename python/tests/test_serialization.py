from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest

import cuscan

h5py = pytest.importorskip("h5py")
hdf5plugin = pytest.importorskip("hdf5plugin")

UMAP = cuscan.UMAP


def fitted(**kwargs: Any) -> UMAP:
    rng = np.random.default_rng(5)
    x = rng.normal(size=(30, 4)).astype(np.float32)
    algorithm: dict[str, Any] = {"optimizer": "compatibility"}
    algorithm.update(kwargs)
    return UMAP(
        backend="cpu",
        neighbor_search="exact",
        init="random",
        learning_rate=0.0,
        n_epochs=20,
        **algorithm,
    ).fit(x)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["isotropic", "per_axis", "graph_edge"])
@pytest.mark.parametrize("search", ["exact", "approximate"])
def test_v5_rescale_snapshot_round_trip(
    tmp_path: Path,
    backend: Literal["cpu", "cuda"],
    mode: Literal["isotropic", "per_axis", "graph_edge"],
    search: Literal["exact", "approximate"],
) -> None:
    x = np.random.default_rng(13).normal(size=(48, 4)).astype(np.float32)
    x[-1] = x[0]
    factor = 3.123456789012345
    model = UMAP(
        backend=backend,
        neighbor_search=search,
        init="random",
        optimizer="compatibility",
        init_rescale=mode,
        init_scale_factor=factor,
        unique=True,
        random_state=11,
        n_epochs=15,
        learning_rate=0.0,
    )
    try:
        model.fit(x)
    except RuntimeError as error:
        if (
            backend == "cuda"
            and os.environ.get("CUSCAN_REQUIRE_CUDA") != "1"
            and (
                "cuda" in str(error).lower()
                and any(s in str(error).lower() for s in ("unavailable", "no device"))
            )
        ):
            pytest.skip(f"CUDA backend unavailable: {error}")
        raise
    before, metadata = model._model.export_state()
    transformed = model.transform(x[:3])
    model.init_rescale = "isotropic" if mode != "isotropic" else "graph_edge"
    model.init_scale_factor = 9.0
    path = tmp_path / "rescale.h5"
    model.save(path)
    restored = UMAP.load(path, backend=backend)
    after, loaded_metadata = restored._model.export_state()
    assert restored.init_rescale == mode
    assert restored.init_scale_factor == factor
    assert loaded_metadata["init_rescale"] == metadata["init_rescale"] == mode
    assert loaded_metadata["init_scale_factor"] == metadata["init_scale_factor"]
    for name in ("embedding", "training_features", "query_graph", "original_to_training"):
        if before[name] is None:
            assert after[name] is None
        else:
            np.testing.assert_array_equal(after[name], before[name])
    np.testing.assert_array_equal(restored.transform(x[:3]), transformed)
    with h5py.File(path) as file:
        assert file.attrs["schema_version"] == 5
        assert file["config"].attrs["init_rescale"] == mode
        assert file["python_precision"].attrs["init_scale_factor"] == factor


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "mutation",
    [
        "version_1",
        "version_2",
        "version_3",
        "version_4",
        "version_future",
        "missing_mode",
        "missing_factor",
        "missing_precision",
        "mode_type",
        "mode_invalid",
        "factor_type",
        "precision_type",
        "factor_zero",
        "factor_negative",
        "factor_nan",
        "factor_inf",
        "precision_mismatch",
        "precision_nan",
    ],
)
def test_v5_rejects_invalid_metadata_before_payload_or_native(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: Literal["cpu", "cuda"],
    mutation: str,
) -> None:
    path = tmp_path / "invalid.h5"
    fitted(init_scale_factor=3.123456789).save(path)
    with h5py.File(path, "r+") as file:
        config = file["config"].attrs
        precision = file["python_precision"].attrs
        if mutation.startswith("version_"):
            file.attrs.modify(
                "schema_version",
                np.uint32(
                    {
                        "version_1": 1,
                        "version_2": 2,
                        "version_3": 3,
                        "version_4": 4,
                        "version_future": 99,
                    }[mutation]
                ),
            )
        elif mutation == "missing_mode":
            del config["init_rescale"]
        elif mutation == "missing_factor":
            del config["init_scale_factor"]
        elif mutation == "missing_precision":
            del precision["init_scale_factor"]
        elif mutation == "mode_type":
            config["init_rescale"] = np.uint32(1)
        elif mutation == "mode_invalid":
            config["init_rescale"] = "unknown"
        elif mutation == "factor_type":
            config["init_scale_factor"] = np.float64(3.0)
        elif mutation == "precision_type":
            precision["init_scale_factor"] = np.float32(3.0)
        elif mutation.startswith("factor_"):
            config["init_scale_factor"] = np.float32(
                {
                    "factor_zero": 0,
                    "factor_negative": -1,
                    "factor_nan": np.nan,
                    "factor_inf": np.inf,
                }[mutation]
            )
        else:
            precision["init_scale_factor"] = np.float64(
                np.nan if mutation == "precision_nan" else 4.0
            )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("payload read or native construction occurred before schema rejection")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    monkeypatch.setattr(cuscan._cuscan, "load", forbidden)
    with pytest.raises(ValueError, match="version" if mutation.startswith("version_") else None):
        UMAP.load(path, backend=backend)


@pytest.mark.parametrize("init", ["random", "recursive", "auto"])
@pytest.mark.parametrize("hub", ["degree", "diverse", "random"])
def test_v5_recursive_settings_snapshot(
    tmp_path: Path, init: Literal["recursive", "auto"], hub: str
) -> None:
    x = np.random.default_rng(42).normal(size=(30, 4)).astype(np.float32)
    kwds = {
        "base_init_threshold": 32,
        "n_embedding_epochs": 3,
        "hub_selection": hub,
        "anchor_sample_size": 7,
    }
    model = UMAP(
        backend="cpu",
        neighbor_search="exact",
        init=init,
        optimizer="compatibility",
        n_epochs=2,
        recursive_coarsening_ratio=3,
        recursive_init_kwds=kwds,
    ).fit(x)
    model.recursive_coarsening_ratio = 4
    model.recursive_init_kwds = None
    path = tmp_path / "recursive.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    assert restored.init == init
    assert restored.recursive_coarsening_ratio == 3
    assert restored.recursive_init_kwds == kwds
    with h5py.File(path) as file:
        attrs = file["config"].attrs
        assert attrs["init"] == init
        assert attrs["recursive_hub_selection"] == hub
        for name, value in (
            ("coarsening_ratio", 3),
            ("base_init_threshold", 32),
            ("n_embedding_epochs", 3),
            ("anchor_sample_size", 7),
        ):
            assert attrs[f"recursive_{name}"].dtype == np.dtype("<u8")
            assert attrs[f"recursive_{name}"] == value


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("recursive_coarsening_ratio", np.uint64(1)),
        ("recursive_coarsening_ratio", np.uint64(5)),
        ("recursive_base_init_threshold", np.uint64(1)),
        ("recursive_base_init_threshold", np.uint64(2**31)),
        ("recursive_n_embedding_epochs", np.uint64(0)),
        ("recursive_anchor_sample_size", np.uint64(0)),
        ("recursive_anchor_sample_size", np.uint64(2**31)),
        ("recursive_hub_selection", "unknown"),
        ("recursive_coarsening_ratio", np.bool_(True)),
        ("recursive_n_embedding_epochs", np.float64(3)),
        ("recursive_hub_selection", None),
    ],
)
def test_v5_rejects_invalid_recursive_metadata_before_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, value: Any
) -> None:
    path = tmp_path / "invalid_recursive.h5"
    fitted().save(path)
    with h5py.File(path, "r+") as file:
        del file["config"].attrs[name]
        if value is not None:
            file["config"].attrs[name] = value

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("payload read or native construction occurred before schema rejection")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    monkeypatch.setattr(cuscan._cuscan, "load", forbidden)
    with pytest.raises(ValueError):
        UMAP.load(path, backend="cpu")


@pytest.mark.parametrize("optimizer", ["compatibility", "adam", "momentum"])
@pytest.mark.parametrize("scale", [None, 0.0, 1.23456789012345])
def test_optimizer_snapshot_round_trip(tmp_path: Path, optimizer: str, scale: float | None) -> None:
    model = fitted(optimizer=optimizer, negative_selection_range=100, negative_sample_scale=scale)
    query = np.random.default_rng(44).normal(size=(3, 4)).astype(np.float32)
    transformed = model.transform(query)
    model.optimizer = "compatibility"
    model.negative_selection_range = 2
    model.negative_sample_scale = 5.0
    path = tmp_path / "optimizer.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    assert restored.optimizer == optimizer
    assert restored.negative_selection_range == 100
    assert restored.negative_sample_scale == scale
    np.testing.assert_array_equal(restored.transform(query), transformed)
    _, metadata = restored._model.export_state()
    assert metadata["optimizer"] == optimizer
    assert metadata["negative_selection_range"] == 100
    assert metadata["negative_sample_scale"] == (None if scale is None else np.float32(scale))


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("optimizer", "invalid"),
        ("negative_selection_range", True),
        ("negative_selection_range", np.bool_(True)),
        ("negative_selection_range", 0),
        ("negative_selection_range", -1),
        ("negative_selection_range", 2**31),
        ("negative_selection_range", 1.0),
        ("negative_sample_scale", True),
        ("negative_sample_scale", np.bool_(True)),
        ("negative_sample_scale", -1.0),
        ("negative_sample_scale", float("nan")),
        ("negative_sample_scale", float("inf")),
        ("negative_sample_scale", 1e100),
        ("negative_sample_scale", 1e-100),
        ("exclude_graph_neighbors", 1),
        ("exclude_graph_neighbors", "true"),
        ("exclude_graph_neighbors", None),
        ("negative_sample_scale_adaptation_samples", True),
        ("negative_sample_scale_adaptation_samples", np.bool_(True)),
        ("negative_sample_scale_adaptation_samples", -1),
        ("negative_sample_scale_adaptation_samples", 2**31),
        ("negative_sample_scale_adaptation_samples", 1.0),
    ],
)
def test_native_load_rejects_bad_optimizer_metadata_before_cuda_creation(
    name: str, value: object
) -> None:
    arrays, metadata = fitted()._model.export_state()
    metadata[name] = value
    with pytest.raises(ValueError, match=name):
        cuscan._cuscan.load(arrays, metadata, backend="cuda", device_index=2**31 - 1)


@pytest.mark.parametrize(
    "name",
    [
        "optimizer",
        "negative_selection_range",
        "negative_sample_scale",
        "exclude_graph_neighbors",
        "negative_sample_scale_adaptation_samples",
    ],
)
def test_native_load_requires_optimizer_metadata_before_cuda_creation(name: str) -> None:
    arrays, metadata = fitted()._model.export_state()
    del metadata[name]
    with pytest.raises(ValueError, match=f"missing metadata key: {name}"):
        cuscan._cuscan.load(arrays, metadata, backend="cuda", device_index=2**31 - 1)


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_loaded_modern_hard_window_transform_preserves_reference(
    tmp_path: Path, optimizer: str
) -> None:
    path = tmp_path / "range.h5"
    fitted(optimizer=optimizer, negative_selection_range=30).save(path)
    with h5py.File(path, "r+") as file:
        file["config"].attrs.modify("negative_selection_range", np.uint64(29))
    restored = UMAP.load(path, backend="cpu")
    assert restored.negative_selection_range == 29
    _, metadata = restored._model.export_state()
    assert metadata["negative_selection_range"] == 29
    reference = restored.embedding_.copy()
    query = np.ones((1, 4), dtype=np.float32)
    first = restored.transform(query)
    assert first.shape == (1, 2)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(restored.transform(query), first)
    np.testing.assert_array_equal(restored.embedding_, reference)
    after, _ = restored._model.export_state()
    np.testing.assert_array_equal(after["embedding"], reference)


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_loaded_modern_small_query_large_reference_round_trip(
    tmp_path: Path, optimizer: Literal["adam", "momentum"]
) -> None:
    x = np.random.default_rng(71).normal(size=(1100, 4)).astype(np.float32)
    query = np.random.default_rng(72).normal(size=(7, 4)).astype(np.float32)
    model = UMAP(
        backend="cpu",
        neighbor_search="exact",
        init="random",
        n_epochs=10,
        random_state=73,
        transform_seed=74,
        optimizer=optimizer,
        negative_selection_range=100,
        exclude_graph_neighbors=True,
        negative_sample_scale_adaptation_samples=16,
    ).fit(x)
    expected = model.transform(query)
    path = tmp_path / "small_query_large_reference.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    assert restored.negative_selection_range == 100
    reference = restored.embedding_.copy()
    actual = restored.transform(query)
    assert np.all(np.isfinite(actual))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.transform(query), expected)
    np.testing.assert_array_equal(restored.embedding_, reference)


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_loaded_modern_equal_query_and_reference_rows_preserves_reference(
    tmp_path: Path, optimizer: Literal["adam", "momentum"]
) -> None:
    x = np.random.default_rng(17).normal(size=(30, 4)).astype(np.float32)
    query = np.random.default_rng(18).normal(size=x.shape).astype(np.float32)
    model = UMAP(
        backend="cpu",
        neighbor_search="exact",
        init="random",
        n_epochs=20,
        n_neighbors=10,
        random_state=19,
        transform_seed=23,
        optimizer=optimizer,
        negative_selection_range=len(x),
    ).fit(x)
    path = tmp_path / "equal_rows.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    before, _ = restored._model.export_state()
    assert before["embedding"] is not None
    reference = before["embedding"].copy()
    first = restored.transform(query)
    after, _ = restored._model.export_state()
    np.testing.assert_array_equal(after["embedding"], reference)
    np.testing.assert_array_equal(restored.embedding_, reference)
    np.testing.assert_array_equal(restored.transform(query), first)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_optimizer",
        "invalid_optimizer",
        "optimizer_type",
        "missing_range",
        "range_type",
        "range_zero",
        "range_large",
        "missing_auto",
        "auto_type",
        "missing_scale",
        "scale_type",
        "missing_precision",
        "precision_type",
        "scale_negative",
        "scale_nan",
        "precision_mismatch",
        "auto_payload",
        "auto_precision",
        "missing_exclude",
        "exclude_type",
        "exclude_int",
        "missing_samples",
        "samples_type",
        "samples_float",
        "samples_bool",
        "samples_large",
    ],
)
def test_optimizer_metadata_rejected_before_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = tmp_path / "invalid_optimizer.h5"
    fitted(negative_sample_scale=None if mutation.startswith("auto_") else 1.25).save(path)
    with h5py.File(path, "r+") as file:
        config = file["config"].attrs
        precision = file["python_precision"].attrs
        changes = {
            "missing_optimizer": (config, "optimizer", None),
            "invalid_optimizer": (config, "optimizer", "wrong"),
            "optimizer_type": (config, "optimizer", np.uint32(1)),
            "missing_range": (config, "negative_selection_range", None),
            "range_type": (config, "negative_selection_range", np.int64(1)),
            "range_zero": (config, "negative_selection_range", np.uint64(0)),
            "range_large": (config, "negative_selection_range", np.uint64(2**31)),
            "missing_auto": (config, "negative_sample_scale_is_auto", None),
            "auto_type": (config, "negative_sample_scale_is_auto", np.uint8(1)),
            "missing_scale": (config, "negative_sample_scale_value", None),
            "scale_type": (config, "negative_sample_scale_value", np.float64(1.25)),
            "missing_precision": (precision, "negative_sample_scale", None),
            "precision_type": (precision, "negative_sample_scale", np.float32(1.25)),
            "scale_negative": (config, "negative_sample_scale_value", np.float32(-1)),
            "scale_nan": (config, "negative_sample_scale_value", np.float32(np.nan)),
            "precision_mismatch": (precision, "negative_sample_scale", np.float64(2)),
            "auto_payload": (config, "negative_sample_scale_value", np.float32(1)),
            "auto_precision": (precision, "negative_sample_scale", np.float64(1)),
            "missing_exclude": (config, "exclude_graph_neighbors", None),
            "exclude_type": (config, "exclude_graph_neighbors", "false"),
            "exclude_int": (config, "exclude_graph_neighbors", np.uint8(0)),
            "missing_samples": (config, "negative_sample_scale_adaptation_samples", None),
            "samples_type": (config, "negative_sample_scale_adaptation_samples", np.int64(128)),
            "samples_float": (
                config,
                "negative_sample_scale_adaptation_samples",
                np.float64(128),
            ),
            "samples_bool": (config, "negative_sample_scale_adaptation_samples", np.bool_(True)),
            "samples_large": (
                config,
                "negative_sample_scale_adaptation_samples",
                np.uint64(2**31),
            ),
        }
        attrs, key, value = changes[mutation]
        if value is None:
            del attrs[key]
        else:
            attrs[key] = value

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("payload read or native construction occurred before schema rejection")

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbidden)
    monkeypatch.setattr(cuscan._cuscan, "load", forbidden)
    with pytest.raises(ValueError):
        UMAP.load(path, backend="cpu")


@pytest.mark.parametrize("optimizer", ["compatibility", "adam", "momentum"])
@pytest.mark.parametrize(("exclude", "samples"), [(False, 128), (True, 0), (True, 2**31 - 1)])
def test_phase4_settings_round_trip(
    tmp_path: Path, optimizer: str, exclude: bool, samples: int
) -> None:
    model = fitted(
        optimizer=optimizer,
        exclude_graph_neighbors=exclude,
        negative_sample_scale_adaptation_samples=samples,
    )
    query = np.random.default_rng(45).normal(size=(3, 4)).astype(np.float32)
    transformed = model.transform(query)
    model.exclude_graph_neighbors = not exclude
    model.negative_sample_scale_adaptation_samples = 7
    path = tmp_path / "phase4.h5"
    model.save(path)
    with h5py.File(path) as file:
        attrs = file["config"].attrs
        assert file.attrs["schema_version"] == 5
        assert attrs["exclude_graph_neighbors"].dtype == np.dtype("|b1")
        assert bool(attrs["exclude_graph_neighbors"]) is exclude
        assert attrs["negative_sample_scale_adaptation_samples"].dtype == np.dtype("<u8")
        assert attrs["negative_sample_scale_adaptation_samples"] == samples
    restored = UMAP.load(path, backend="cpu")
    assert restored.exclude_graph_neighbors is exclude
    assert restored.negative_sample_scale_adaptation_samples == samples
    assert type(restored.negative_sample_scale_adaptation_samples) is int
    assert restored._fit_config["exclude_graph_neighbors"] is exclude
    assert restored._fit_config["negative_sample_scale_adaptation_samples"] == samples
    np.testing.assert_array_equal(restored.transform(query), transformed)
    _, metadata = restored._model.export_state()
    assert metadata["exclude_graph_neighbors"] is exclude
    assert metadata["negative_sample_scale_adaptation_samples"] == samples


def test_phase4_settings_default_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "phase4_default.h5"
    fitted().save(path)
    restored = UMAP.load(path, backend="cpu")
    assert restored.exclude_graph_neighbors is False
    assert restored.negative_sample_scale_adaptation_samples == 128


def _round_trip_or_skip_cuda(
    tmp_path: Path, backend: Literal["cpu", "cuda"], name: str, **kwargs: Any
) -> None:
    x = np.random.default_rng(17).normal(size=(48, 4)).astype(np.float32)
    query = np.random.default_rng(18).normal(size=(5, 4)).astype(np.float32)
    model = UMAP(backend=backend, n_neighbors=10, n_epochs=20, random_state=7, **kwargs)
    try:
        model.fit(x)
    except RuntimeError as error:
        if (
            backend == "cuda"
            and os.environ.get("CUSCAN_REQUIRE_CUDA") != "1"
            and (
                "cuda" in str(error).lower()
                and any(s in str(error).lower() for s in ("unavailable", "no device"))
            )
        ):
            pytest.skip(f"CUDA backend unavailable: {error}")
        raise
    init = kwargs.get("init", "auto")
    optimizer = kwargs.get("optimizer", "adam")
    transformed = model.transform(query)
    path = tmp_path / f"{name}.h5"
    model.save(path)
    restored = UMAP.load(path, backend=backend)
    assert restored.init == init
    assert restored.optimizer == optimizer
    _, metadata = restored._model.export_state()
    assert metadata["init"] == init
    assert metadata["optimizer"] == optimizer
    with h5py.File(path) as file:
        assert file.attrs["schema_version"] == 5
        assert file["config"].attrs["init"] == init
    np.testing.assert_array_equal(restored.embedding_, model.embedding_)
    np.testing.assert_array_equal(restored.transform(query), transformed)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_v5_library_defaults_round_trip(tmp_path: Path, backend: Literal["cpu", "cuda"]) -> None:
    _round_trip_or_skip_cuda(tmp_path, backend, "library_defaults")


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_v5_explicit_previous_defaults_round_trip(
    tmp_path: Path, backend: Literal["cpu", "cuda"]
) -> None:
    _round_trip_or_skip_cuda(
        tmp_path, backend, "previous_defaults", optimizer="compatibility", init="spectral"
    )


@pytest.mark.parametrize("optimizer", ["adam", "momentum"])
def test_phase4_equal_rows_and_small_transform_after_load(
    tmp_path: Path, optimizer: Literal["adam", "momentum"]
) -> None:
    x = np.random.default_rng(31).normal(size=(30, 4)).astype(np.float32)
    query = np.random.default_rng(32).normal(size=x.shape).astype(np.float32)
    model = UMAP(
        backend="cpu",
        neighbor_search="exact",
        init="random",
        n_epochs=20,
        n_neighbors=10,
        random_state=19,
        transform_seed=23,
        optimizer=optimizer,
        negative_selection_range=len(x),
        exclude_graph_neighbors=True,
        negative_sample_scale_adaptation_samples=5,
    ).fit(x)
    expected_equal = model.transform(query)
    expected_small = model.transform(query[:2])
    path = tmp_path / "phase4_equal.h5"
    model.save(path)
    restored = UMAP.load(path, backend="cpu")
    np.testing.assert_array_equal(restored.transform(query), expected_equal)
    np.testing.assert_array_equal(restored.transform(query[:2]), expected_small)


def test_phase4_inconsistent_native_settings_rejected_on_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = fitted(exclude_graph_neighbors=True, negative_sample_scale_adaptation_samples=9)
    arrays, metadata = model._model.export_state()
    for key, value in (
        ("exclude_graph_neighbors", 1),
        ("negative_sample_scale_adaptation_samples", 2**31),
        ("negative_sample_scale_adaptation_samples", True),
    ):
        bad = dict(metadata, **{key: value})
        stub = type("Stub", (), {"export_state": lambda self, bad=bad: (arrays, bad)})()
        monkeypatch.setattr(model, "_model", stub)
        with pytest.raises(ValueError, match=key):
            model.save(tmp_path / f"{key}.h5")
        assert not (tmp_path / f"{key}.h5").exists()


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
        assert file.attrs["format"] == "cuscan_fitted_model"
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
        backend="cpu",
        neighbor_search=search,
        init="random",
        optimizer="compatibility",
        n_epochs=15,
        learning_rate=0.0,
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
    model = UMAP(backend="cpu", init="spectral", optimizer="compatibility").fit(
        np.array([[2.0, 3.0]], dtype=np.float32)
    )
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
        optimizer="compatibility",
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
    real_import = cuscan._cuscan.load
    received: list[tuple[str, int]] = []

    def capture(arrays: Any, metadata: Any, backend: str, device_index: int) -> Any:
        received.append((backend, device_index))
        return real_import(arrays, metadata, backend="cpu", device_index=0)

    monkeypatch.setattr(cuscan._cuscan, "load", capture)
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
        file.attrs["schema_version"] = np.uint32(99)
    with pytest.raises(ValueError, match="version"):
        UMAP.load(str(path))
    with h5py.File(path, "w") as file:
        file["external"] = h5py.ExternalLink("other.h5", "/target")
    with pytest.raises(ValueError, match="links"):
        UMAP.load(str(path))


@pytest.mark.parametrize("format_value", ["fast_umap_fitted_model", "invalid_format", 123])
def test_rejects_old_and_malformed_format(tmp_path: Path, format_value: Any) -> None:
    path = tmp_path / "model.h5"
    fitted().save(path)
    with h5py.File(path, "r+") as file:
        file.attrs["format"] = format_value
    with pytest.raises(ValueError, match="format"):
        UMAP.load(path, backend="cpu")


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
from cuscan._serialization_v5 import _deps
try:
    _deps()
except ImportError as error:
    assert 'cuscan[serialization]' in str(error)
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
    from cuscan import _serialization_v5

    path = tmp_path / "raced.h5"
    actual_link = _serialization_v5.os.link

    def race(source: str, destination: str) -> None:
        Path(destination).write_text("racing writer")
        actual_link(source, destination)

    monkeypatch.setattr(_serialization_v5.os, "link", race)
    with pytest.raises(FileExistsError):
        fitted().save(str(path))
    assert path.read_text() == "racing writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["raced.h5"]


@pytest.mark.parametrize("failure", ["write", "close", "fsync", "link", "replace"])
def test_precommit_failures_clean_temporary_and_preserve_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    from cuscan import _serialization_v5

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

        monkeypatch.setattr(_serialization_v5.os, operation, fail)
    with pytest.raises(OSError):
        model.save(destination, overwrite=overwrite)
    if overwrite:
        assert destination.read_text() == "old destination"
    else:
        assert not destination.exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == (["target.h5"] if overwrite else [])


def test_file_sync_uses_writable_descriptor_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuscan import _serialization_common

    path = tmp_path / "model.h5"
    real_fsync = os.fsync
    synced = False

    def check_file_sync(fd: int) -> None:
        nonlocal synced
        if not synced:
            assert not path.exists()
            assert os.write(fd, b"") == 0
            synced = True
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", check_file_sync)
    _serialization_common._publish(path, False, lambda file, *_: None, "test")
    assert synced
    with h5py.File(path, "r"):
        pass


def test_windows_directory_sync_does_not_use_crt_directory_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cuscan import _serialization_common

    def unexpected_open(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Windows directory must not be opened as a CRT descriptor")

    with monkeypatch.context() as patch:
        patch.setattr(os, "name", "nt")
        patch.setattr(os, "open", unexpected_open)
        _serialization_common._sync_directory(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="Windows cannot fsync directory CRT descriptors")
def test_postcommit_directory_fsync_failure_identifies_committed_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from cuscan import _serialization_v5

    path = tmp_path / "committed.h5"
    real_fsync = os.fsync
    calls = 0

    def fail_directory(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(_serialization_v5.os, "fsync", fail_directory)
    with pytest.raises(_serialization_v5.DurabilityUncertainError, match="committed") as error:
        fitted().save(path)
    assert isinstance(error.value.__cause__, OSError)
    assert path.is_file()
    assert sorted(item.name for item in tmp_path.iterdir()) == ["committed.h5"]
