import ast
import hashlib
import inspect
import json
import os
import threading
import time
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import cuscan
from cuscan import UMAP


def blobs(n_per_cluster: int = 24) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(31)
    centers = np.array([[-7.0, 0.0, 0.0], [0.0, 7.0, 0.0], [7.0, 0.0, 0.0]], dtype=np.float32)
    labels = np.repeat(np.arange(len(centers)), n_per_cluster)
    data = np.vstack([center + rng.normal(0, 0.22, (n_per_cluster, 3)) for center in centers])
    return data.astype(np.float32), labels


def estimator(**kwargs: object) -> UMAP:
    params: dict[str, Any] = {
        "n_neighbors": 10,
        "n_epochs": 100,
        "random_state": 19,
        "init": "spectral",
        "optimizer": "compatibility",
    }
    params.update(kwargs)
    return UMAP(**params)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("optimizer", "unknown"),
        ("optimizer", True),
        ("negative_selection_range", True),
        ("negative_selection_range", 0),
        ("negative_selection_range", 2**31),
        ("negative_selection_range", 1.5),
        ("negative_sample_scale", True),
        ("negative_sample_scale", -1),
        ("negative_sample_scale", float("nan")),
        ("negative_sample_scale", float("inf")),
        ("negative_sample_scale", 1e100),
        ("negative_sample_scale", 1e-100),
    ],
)
def test_optimizer_settings_reject_invalid_values(name: str, value: object) -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match=name):
        estimator(**{name: value}).fit(x)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("optimizer", "unknown"),
        ("negative_selection_range", True),
        ("negative_selection_range", False),
        ("negative_selection_range", np.bool_(True)),
        ("negative_selection_range", np.bool_(False)),
        ("negative_selection_range", 0),
        ("negative_selection_range", -1),
        ("negative_selection_range", 2**31),
        ("negative_selection_range", 1.0),
        ("negative_selection_range", np.float32(2)),
        ("negative_sample_scale", True),
        ("negative_sample_scale", False),
        ("negative_sample_scale", np.bool_(True)),
        ("negative_sample_scale", np.bool_(False)),
        ("negative_sample_scale", -0.5),
        ("negative_sample_scale", float("nan")),
        ("negative_sample_scale", float("inf")),
        ("negative_sample_scale", -float("inf")),
        ("negative_sample_scale", 1e100),
        ("negative_sample_scale", 1e-100),
    ],
)
def test_native_fit_rejects_optimizer_settings_before_cuda_creation(
    name: str, value: object
) -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match=name):
        cuscan._cuscan.fit_umap(
            x, backend="cuda", device_index=2**31 - 1, **{name: cast(Any, value)}
        )


@pytest.mark.parametrize("value", [True, np.bool_(True), 42])
def test_native_fit_rejects_nonstring_optimizer_before_cuda_creation(value: object) -> None:
    x, _ = blobs(4)
    with pytest.raises((TypeError, ValueError)):
        cuscan._cuscan.fit_umap(
            x, backend="cuda", device_index=2**31 - 1, init="spectral", optimizer=cast(Any, value)
        )


@pytest.mark.parametrize("optimizer", ["compatibility", "adam", "momentum"])
@pytest.mark.parametrize("scale", [None, 0.0, 1.23456789012345])
def test_optimizer_settings_forward_and_snapshot(
    monkeypatch: pytest.MonkeyPatch, optimizer: str, scale: float | None
) -> None:
    x, _ = blobs(4)
    seen: dict[str, Any] = {}

    class Native:
        embedding = np.zeros((len(x), 2), dtype=np.float32)

        def transform(self, values: np.ndarray) -> np.ndarray:
            return np.zeros((len(values), 2), dtype=np.float32)

    def fit_native(values: np.ndarray, **kwargs: Any) -> Native:
        seen.update(kwargs)
        return Native()

    monkeypatch.setattr(cuscan._cuscan, "fit_umap", fit_native)
    model = estimator(
        optimizer=optimizer, negative_selection_range=2**31 - 1, negative_sample_scale=scale
    ).fit(x)
    assert seen["optimizer"] == model._fit_config["optimizer"] == optimizer
    assert (
        seen["negative_selection_range"]
        == model._fit_config["negative_selection_range"]
        == 2**31 - 1
    )
    assert seen["negative_sample_scale"] == model._fit_config["negative_sample_scale"] == scale
    model.optimizer = cast(Any, "invalid")
    assert model.transform(x[:2]).shape == (2, 2)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("exclude_graph_neighbors", 1),
        ("exclude_graph_neighbors", 0),
        ("exclude_graph_neighbors", "true"),
        ("exclude_graph_neighbors", None),
        ("exclude_graph_neighbors", 1.0),
        ("negative_sample_scale_adaptation_samples", True),
        ("negative_sample_scale_adaptation_samples", False),
        ("negative_sample_scale_adaptation_samples", np.bool_(True)),
        ("negative_sample_scale_adaptation_samples", -1),
        ("negative_sample_scale_adaptation_samples", 2**31),
        ("negative_sample_scale_adaptation_samples", 2**64),
        ("negative_sample_scale_adaptation_samples", 1.0),
        ("negative_sample_scale_adaptation_samples", np.float32(2)),
        ("negative_sample_scale_adaptation_samples", None),
    ],
)
def test_phase4_settings_reject_invalid_values_in_wrapper_and_native(
    name: str, value: object
) -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match=name):
        estimator(backend="cuda", device_index=2**31 - 1, **{name: value}).fit(x)
    with pytest.raises(ValueError, match=name):
        cuscan._cuscan.fit_umap(
            x, backend="cuda", device_index=2**31 - 1, **{name: cast(Any, value)}
        )


@pytest.mark.parametrize("exclude", [False, True, np.bool_(True)])
@pytest.mark.parametrize("samples", [0, 128, np.int64(2**31 - 1)])
def test_phase4_settings_forward_and_snapshot(
    monkeypatch: pytest.MonkeyPatch, exclude: Any, samples: Any
) -> None:
    x, _ = blobs(4)
    seen: dict[str, Any] = {}

    class Native:
        embedding = np.zeros((len(x), 2), dtype=np.float32)

    def fit_native(values: np.ndarray, **kwargs: Any) -> Native:
        seen.update(kwargs)
        return Native()

    monkeypatch.setattr(cuscan._cuscan, "fit_umap", fit_native)
    model = estimator(
        exclude_graph_neighbors=exclude, negative_sample_scale_adaptation_samples=samples
    ).fit(x)
    assert type(seen["exclude_graph_neighbors"]) is bool
    assert seen["exclude_graph_neighbors"] == bool(exclude)
    assert type(seen["negative_sample_scale_adaptation_samples"]) is int
    assert seen["negative_sample_scale_adaptation_samples"] == int(samples)
    model.exclude_graph_neighbors = cast(Any, "invalid")
    model.negative_sample_scale_adaptation_samples = -5
    assert model._fit_config["exclude_graph_neighbors"] == bool(exclude)
    assert model._fit_config["negative_sample_scale_adaptation_samples"] == int(samples)


def test_phase4_settings_defaults() -> None:
    model = UMAP()
    assert model.init == "auto"
    assert model.optimizer == "adam"
    assert model.exclude_graph_neighbors is False
    assert model.negative_sample_scale_adaptation_samples == 128


EXPECTED_UMAP_DEFAULTS: dict[str, Any] = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "spread": 1.0,
    "local_connectivity": 1.0,
    "set_op_mix_ratio": 1.0,
    "n_epochs": None,
    "learning_rate": 1.0,
    "negative_sample_rate": 5,
    "repulsion_strength": 1.0,
    "init": "auto",
    "random_state": None,
    "transform_seed": 42,
    "transform_queue_size": 4.0,
    "backend": "auto",
    "device_index": 0,
    "neighbor_search": "auto",
    "unique": False,
    "memory_limit_bytes": None,
    "init_rescale": "isotropic",
    "init_scale_factor": 1.0,
    "optimizer": "adam",
    "negative_selection_range": 200_000,
    "negative_sample_scale": None,
    "exclude_graph_neighbors": False,
    "negative_sample_scale_adaptation_samples": 128,
    "recursive_coarsening_ratio": 4,
    "recursive_init_kwds": None,
}


def _fit_or_skip_cuda(backend: str, fit: Any) -> Any:
    try:
        return fit()
    except RuntimeError as error:
        if (
            backend == "cuda"
            and os.environ.get("CUSCAN_REQUIRE_CUDA") != "1"
            and ("no CUDA" in str(error) or "CUDA is unavailable" in str(error))
        ):
            pytest.skip(str(error))
        raise


def test_defaults_are_auto_and_adam() -> None:
    # UMAP has no sklearn get_params; the constructor attributes are its parameters.
    model = UMAP()
    params = {name: getattr(model, name) for name in EXPECTED_UMAP_DEFAULTS}
    assert params == EXPECTED_UMAP_DEFAULTS
    assert set(inspect.signature(UMAP).parameters) == set(EXPECTED_UMAP_DEFAULTS)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_default_fit_matches_explicit_adam_auto(backend: str) -> None:
    x, _ = blobs()
    options: dict[str, Any] = {
        "backend": backend,
        "n_neighbors": 10,
        "n_epochs": 30,
        "random_state": 19,
    }
    default = _fit_or_skip_cuda(backend, lambda: UMAP(**options).fit(x))
    explicit = UMAP(optimizer="adam", init="auto", **options).fit(x)
    assert default.init == "auto"
    assert default.optimizer == "adam"
    np.testing.assert_array_equal(default.embedding_, explicit.embedding_)
    np.testing.assert_array_equal(default.transform(x[:5]), explicit.transform(x[:5]))


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_native_default_arguments(backend: str) -> None:
    x, _ = blobs()
    options: dict[str, Any] = {
        "backend": backend,
        "n_neighbors": 10,
        "n_epochs": 30,
        "random_state": 19,
    }
    default = _fit_or_skip_cuda(backend, lambda: cuscan._cuscan.fit_umap(x, **options))
    explicit = cuscan._cuscan.fit_umap(x, init="auto", optimizer="adam", **options)
    _, metadata = default.export_state()
    assert metadata["init"] == "auto"
    assert metadata["optimizer"] == "adam"
    np.testing.assert_array_equal(default.embedding, explicit.embedding)


def test_default_surfaces_agree() -> None:
    wrapper = inspect.signature(UMAP).parameters
    assert wrapper["init"].default == "auto"
    assert wrapper["optimizer"].default == "adam"

    text_signature = getattr(cuscan._cuscan.fit_umap, "__text_signature__", None)
    assert text_signature is not None
    assert "init='auto'" in text_signature
    assert "optimizer='adam'" in text_signature
    native = inspect.signature(cuscan._cuscan.fit_umap).parameters
    assert native["init"].default == "auto"
    assert native["optimizer"].default == "adam"

    stub = Path(cuscan.__file__).with_name("_cuscan.pyi")
    tree = ast.parse(stub.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "fit_umap"
    )
    stub_defaults = {
        argument.arg: ast.literal_eval(default)
        for argument, default in zip(
            function.args.kwonlyargs, function.args.kw_defaults, strict=True
        )
        if default is not None
    }
    assert stub_defaults["init"] == "auto"
    assert stub_defaults["optimizer"] == "adam"


@pytest.mark.parametrize(("exclude", "samples"), [(False, 128), (True, 0), (True, 2**31 - 1)])
def test_phase4_settings_survive_native_export_and_load(exclude: bool, samples: int) -> None:
    x, _ = blobs(4)
    model = estimator(
        backend="cpu",
        init="random",
        learning_rate=0.0,
        exclude_graph_neighbors=exclude,
        negative_sample_scale_adaptation_samples=samples,
    ).fit(x)
    arrays, metadata = model._model.export_state()
    assert metadata["exclude_graph_neighbors"] is exclude
    assert metadata["negative_sample_scale_adaptation_samples"] == samples
    loaded = cuscan._cuscan.load(arrays, metadata, backend="cpu")
    _, reloaded = loaded.export_state()
    assert reloaded["exclude_graph_neighbors"] is exclude
    assert reloaded["negative_sample_scale_adaptation_samples"] == samples
    for key, bad in [
        ("exclude_graph_neighbors", 1),
        ("negative_sample_scale_adaptation_samples", 2**31),
        ("negative_sample_scale_adaptation_samples", True),
    ]:
        with pytest.raises(ValueError, match=key):
            cuscan._cuscan.load(arrays, {**metadata, key: bad}, backend="cpu")
    missing = dict(metadata)
    del missing["exclude_graph_neighbors"]
    with pytest.raises((KeyError, ValueError)):
        cuscan._cuscan.load(arrays, missing, backend="cpu")


def test_exports_and_shape_dtype() -> None:
    assert "UMAP" in cuscan.__all__
    x, _ = blobs()
    model = estimator()
    result = model.fit_transform(x)
    assert result.shape == (len(x), 2)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert model.embedding_.shape == result.shape


@pytest.mark.parametrize(
    ("input_kind", "expected_dtype"),
    [
        ("float16", np.float16),
        ("float32", np.float32),
        ("float64", np.float64),
        ("list", np.float32),
        ("int64", np.float32),
    ],
)
def test_output_dtype_matches_floating_input(input_kind: str, expected_dtype: type) -> None:
    x, _ = blobs(12)
    values = x.tolist() if input_kind == "list" else x.astype(input_kind)
    model = estimator(backend="cpu", init="random", learning_rate=0.0)
    embedding = model.fit_transform(values)
    query = values[:4]
    transformed = model.transform(query)

    assert embedding.dtype == expected_dtype
    assert transformed.dtype == expected_dtype
    assert model.embedding_.dtype == np.float32
    if input_kind == "float32":
        assert embedding is model.embedding_
    if input_kind == "float64":
        reference = estimator(backend="cpu", init="random", learning_rate=0.0)
        reference_embedding = reference.fit_transform(x)
        reference_transformed = reference.transform(x[:4])
        np.testing.assert_array_equal(embedding, reference_embedding.astype(np.float64))
        np.testing.assert_array_equal(transformed, reference_transformed.astype(np.float64))


@pytest.mark.parametrize("neighbor_search", ["exact", "approximate"])
def test_transform_rejects_overflow_without_poisoning_model(neighbor_search: str) -> None:
    x = np.random.default_rng(7).normal(size=(300, 4)).astype(np.float32)
    model = estimator(
        backend="cpu", neighbor_search=neighbor_search, init="random", learning_rate=0.0
    ).fit(x)
    query = np.vstack((x[:3], np.full((1, 4), 1e20, dtype=np.float32)))
    with pytest.raises(RuntimeError) as error:
        model.transform(query)
    assert str(error.value) == (
        "fuzzy graph contains missing or invalid neighbor indices "
        "(for example, a squared feature distance overflowed float32)"
    )
    result = model.transform(x[:3])
    assert result.shape == (3, 2)
    assert np.isfinite(result).all()


@pytest.mark.parametrize(("n_components", "spectral_fails"), [(2, True), (3, True), (1, False)])
def test_connected_tiny_graph_spectral_error_and_random_success(
    n_components: int, spectral_fails: bool
) -> None:
    # The nearest-neighbor edges between values 0--1 and 3--1 connect all vertices.
    x = np.array([[0.0], [1.0], [3.0]], dtype=np.float32)
    params: dict[str, Any] = {
        "n_components": n_components,
        "n_neighbors": 2,
        "backend": "cpu",
        "random_state": 7,
        "optimizer": "compatibility",
    }
    if spectral_fails:
        with pytest.raises(RuntimeError, match=r"n_components \+ 2"):
            UMAP(**params, init="spectral").fit(x)
        result = UMAP(**params, init="random").fit_transform(x)
    else:
        result = UMAP(**params, init="spectral").fit_transform(x)
    assert result.shape == (3, n_components)
    assert np.isfinite(result).all()


@pytest.mark.parametrize("backend", ["auto", "cpu"])
def test_backend_selectors_fit_transform_and_reject_flex(backend: str) -> None:
    x, _ = blobs(12)
    model = estimator(backend=backend, neighbor_search="exact").fit(x)
    assert model.embedding_.shape == (len(x), 2)
    assert np.isfinite(model.embedding_).all()
    transformed = model.transform(x[:3])
    assert transformed.shape == (3, 2)
    assert np.isfinite(transformed).all()

    with pytest.raises(ValueError, match="backend"):
        estimator(backend="flex").fit(x)


@pytest.mark.parametrize("neighbor_search", ["exact", "approximate"])
def test_cpu_fit_transform_and_input_validation(neighbor_search: str) -> None:
    x, _ = blobs(12)
    model = estimator(backend="cpu", neighbor_search=neighbor_search, init="random").fit(x)
    assert model.embedding_.shape == (len(x), 2)
    assert np.isfinite(model.embedding_).all()

    transformed = model.transform(x[:4])
    assert transformed.shape == (4, 2)
    assert np.isfinite(transformed).all()

    with pytest.raises(ValueError, match="finite"):
        model.transform(np.array([[np.nan, 0.0, 0.0]], dtype=np.float32))


def test_random_state_determinism() -> None:
    x, _ = blobs(12)
    first = estimator(init="random", learning_rate=0.0).fit_transform(x)
    second = estimator(init="random", learning_rate=0.0).fit_transform(x)
    np.testing.assert_array_equal(first, second)


def test_unique_defaults_to_raw_training_behavior() -> None:
    x, _ = blobs(12)
    rows = np.vstack((x, x[0], x[3]))
    default = estimator(learning_rate=0.0).fit_transform(rows)
    explicit_raw = estimator(unique=False, learning_rate=0.0).fit_transform(rows)
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

    unique_model = estimator(unique=True, init="random", learning_rate=0.0).fit(rows)
    reference_embedding = estimator(init="random", learning_rate=0.0).fit_transform(distinct_rows)
    np.testing.assert_array_equal(unique_model.embedding_, reference_embedding[row_to_unique])
    np.testing.assert_array_equal(unique_model.embedding_[-3], unique_model.embedding_[0])
    np.testing.assert_array_equal(unique_model.embedding_[-1], unique_model.embedding_[5])
    assert unique_model.embedding_.shape == (len(rows), 2)

    transformed = unique_model.transform(np.vstack((rows[1], rows[-2])))
    assert transformed.shape == (2, 2)
    assert np.isfinite(transformed).all()


def test_optimized_unique_layout_has_quality_and_exact_duplicate_identity() -> None:
    x, labels = blobs()
    rows = np.vstack((x, x[0], x[3]))
    labels = np.concatenate((labels, labels[[0, 3]]))
    embedding = estimator(unique=True, init="random").fit_transform(rows)

    np.testing.assert_array_equal(embedding[-2], embedding[0])
    np.testing.assert_array_equal(embedding[-1], embedding[3])
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
    os.environ.get("CUSCAN_REQUIRE_CUDA") != "1",
    reason="CUDA required only in CUDA validation runs",
)
def test_cuda_backend() -> None:
    x, _ = blobs(12)
    result = estimator(backend="cuda").fit_transform(x)
    assert result.shape == (len(x), 2)
    assert np.isfinite(result).all()


@pytest.mark.skipif(
    os.environ.get("CUSCAN_REQUIRE_CUDA") != "1",
    reason="CUDA required only in CUDA validation runs",
)
def test_cuda_transform_releases_gil_for_at_least_half_second() -> None:
    x, _ = blobs(12)
    model = estimator(backend="cuda", init="random", neighbor_search="exact", n_epochs=200).fit(x)
    # Keep native work above half a second even when kernels are already compiled.
    query = np.random.default_rng(23).normal(size=(131_072, x.shape[1])).astype(np.float32)

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
        output, start_ns, end_ns = model._model._timed_transform(query)
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


def test_spectral_init_on_separated_blobs_raises_actionable_error() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.0, 1.0, size=(240, 8)).astype(np.float32)
    for blob in range(6):
        x[blob * 40 : (blob + 1) * 40, blob % 8] += blob * 50.0
    with pytest.raises(RuntimeError, match="underflows exp"):
        estimator(
            n_components=1,
            n_neighbors=10,
            init="spectral",
            backend="cpu",
            random_state=7,
            n_epochs=None,
        ).fit(x)
    model = estimator(
        n_components=1, n_neighbors=10, init="random", backend="cpu", random_state=7, n_epochs=None
    )
    assert model.fit_transform(x).shape == (240, 1)


@pytest.mark.parametrize("factor", [True, np.bool_(False), "2", None, 0, -1, np.nan, np.inf])
def test_init_factor_rejects_invalid_python_values(factor: Any) -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match="init_scale_factor"):
        estimator(init_scale_factor=factor).fit(x)


@pytest.mark.parametrize("factor", [1e-100, 1e100])
def test_init_factor_native_float32_conversion_cannot_bypass_validation(factor: float) -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match="init_scale_factor"):
        estimator(init_scale_factor=factor).fit(x)
    with pytest.raises(ValueError, match="init_scale_factor"):
        cuscan._cuscan.fit_umap(
            x, backend="cpu", init="spectral", optimizer="compatibility", init_scale_factor=factor
        )


def test_init_rescale_unknown_mode_rejected_by_wrapper_and_native() -> None:
    x, _ = blobs(4)
    with pytest.raises(ValueError, match="init_rescale"):
        estimator(init_rescale="axis").fit(x)
    with pytest.raises(ValueError, match="init_rescale"):
        cuscan._cuscan.fit_umap(
            x,
            init="spectral",
            optimizer="compatibility",
            init_rescale="axis",  # ty: ignore[invalid-argument-type]
        )


def test_pinned_modern_rescale_oracle_provenance_and_outputs() -> None:
    import umap.umap_ as source
    from comparison.umap_reference import require_reference
    from scipy.sparse import coo_matrix

    require_reference()
    path = (
        Path(__file__).resolve().parents[2]
        / "crates/umap/tests/fixtures/standard_umap_reference.json"
    )
    oracle = json.loads(path.read_text())["init_rescale"]
    assert oracle["version"] == "0.6.0"
    assert hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() == oracle["source_sha256"]
    coordinates = np.asarray(oracle["coordinates"], dtype=np.float32)
    graph = (
        coo_matrix(
            (oracle["weights"], (oracle["heads"], oracle["tails"])), shape=(len(coordinates),) * 2
        )
        .astype(np.float32)
        .tocsr()
    )
    for mode, expected in oracle["outputs"].items():
        actual = source._rescale_embedding(
            coordinates,
            len(coordinates),
            oracle["repulsion_strength"],
            method=mode,
            scale_factor=oracle["init_scale_factor"],
            graph=graph,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_recursive_auto_matches_multilevel_adam_fit(backend: str) -> None:
    x = np.random.default_rng(11).normal(size=(64, 3)).astype(np.float32)
    options = dict(
        backend=backend,
        optimizer="adam",
        n_epochs=5,
        random_state=11,
        recursive_init_kwds={"base_init_threshold": 8, "n_embedding_epochs": 2},
    )
    try:
        recursive = estimator(init="recursive", **options).fit_transform(x)
    except RuntimeError as error:
        if (
            backend == "cuda"
            and os.environ.get("CUSCAN_REQUIRE_CUDA") != "1"
            and ("no CUDA" in str(error) or "CUDA is unavailable" in str(error))
        ):
            pytest.skip(str(error))
        raise
    auto = estimator(init="auto", **options).fit_transform(x)
    assert recursive.shape == (64, 2)
    assert np.isfinite(recursive).all()
    np.testing.assert_array_equal(recursive, auto)


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("init", ["random", "spectral"])
@pytest.mark.parametrize("mode", ["isotropic", "per_axis", "graph_edge"])
@pytest.mark.parametrize("search", ["exact", "approximate"])
def test_modern_initialization_controls_center_scale_and_repeat(
    backend: str, init: str, mode: str, search: str
) -> None:
    x = np.random.default_rng(31).normal(size=(40, 3)).astype(np.float32)
    params: dict[str, Any] = {
        "backend": backend,
        "init": init,
        "init_rescale": mode,
        "init_scale_factor": np.float64(1.25),
        "neighbor_search": search,
        "n_neighbors": 8,
        "learning_rate": 0.0,
        "n_epochs": 12,
    }
    try:
        model = estimator(**params).fit(x)
    except RuntimeError as error:
        if (
            backend == "cuda"
            and os.environ.get("CUSCAN_REQUIRE_CUDA") != "1"
            and ("no CUDA" in str(error) or "CUDA is unavailable" in str(error))
        ):
            pytest.skip(str(error))
        raise
    assert model.embedding_.shape == (40, 2)
    assert np.isfinite(model.embedding_).all()
    np.testing.assert_allclose(model.embedding_.mean(axis=0), 0.0, atol=1e-5)
    repeated = estimator(**params).fit(x)
    np.testing.assert_array_equal(model.embedding_, repeated.embedding_)
    doubled = estimator(**(params | {"init_scale_factor": 2.5})).fit(x)
    np.testing.assert_allclose(doubled.embedding_, 2 * model.embedding_, rtol=1e-5, atol=1e-5)
    _, metadata = model._model.export_state()
    assert metadata["init_rescale"] == mode
    assert metadata["init_scale_factor"] == np.float32(1.25)
