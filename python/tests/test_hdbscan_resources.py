from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from fast_umap import HDBSCAN, _fast_umap

GIB = 2**30
ROWS = 20_000
KNN_FEATURES = 64  # above the 30-feature Auto threshold, so Auto resolves to the kNN graph


def _knn_estimate(
    *,
    advisory: dict[str, Any] | None,
    width: int = 15,
    rows: int = 60,
    scenarios: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "resolved_graph": "knn",
        "knn": {
            "source": "exact",
            "finite_rows": rows,
            "min_samples": 5,
            "effective_width": width,
            "directed_edges": rows * width,
            "density": width / (rows - 1),
            "scenarios": scenarios
            or [
                {"backend": "cpu", "host_peak_bytes": GIB, "device_peak_bytes": 0},
                {"backend": "cuda", "host_peak_bytes": 2**20, "device_peak_bytes": GIB},
            ],
            "advisory": advisory,
        },
    }


def _advisory(*, floor: bool = False) -> dict[str, Any]:
    return {
        "threshold_bytes": GIB,
        "triggers": ["cpu", "cuda"],
        "width_floor_applies": floor,
    }


def _blobs(rows: int = 60, features: int = 3) -> np.ndarray:
    rng = np.random.default_rng(7)
    return np.vstack(
        [
            rng.normal(loc=0.0, scale=0.2, size=(rows // 2, features)),
            rng.normal(loc=5.0, scale=0.2, size=(rows - rows // 2, features)),
        ]
    ).astype(np.float32)


def _patch_estimate(monkeypatch: pytest.MonkeyPatch, estimate: dict[str, Any]) -> None:
    monkeypatch.setattr(
        _fast_umap, "estimate_hdbscan_resources", lambda *_args, **_kwargs: estimate
    )


def _advisories(record: Any) -> list[Any]:
    return [item for item in record if "kNN graph" in str(item.message)]


def _native_knn(estimate: dict[str, object]) -> dict[str, Any]:
    # The bridge returns a kNN dict for kNN-route shapes; assert that shape before reading fields.
    knn = estimate["knn"]
    assert isinstance(knn, dict)
    return knn


# Real bridge: pure arithmetic on shapes. No data arrays, no CUDA initialization.


def test_bridge_complete_route_has_no_knn_estimate() -> None:
    result = _fast_umap.estimate_hdbscan_resources(10**9, 3, backend="cpu", min_samples=5)
    assert result["resolved_graph"] == "complete"
    assert result["knn"] is None


def test_bridge_default_width_is_floored_by_min_samples() -> None:
    # Default request max(3 * min_samples, 15) = 75 for min_samples=25.
    knn = _native_knn(
        _fast_umap.estimate_hdbscan_resources(ROWS, KNN_FEATURES, backend="cpu", min_samples=25)
    )
    assert knn["effective_width"] == 75
    assert knn["min_samples"] == 25
    assert knn["finite_rows"] == ROWS
    assert knn["directed_edges"] == ROWS * 75


def test_bridge_explicit_wide_knn_k_is_reported_as_effective_width() -> None:
    knn = _native_knn(
        _fast_umap.estimate_hdbscan_resources(
            ROWS, KNN_FEATURES, backend="cpu", min_samples=5, knn_k=512
        )
    )
    assert knn["effective_width"] == 512
    assert knn["directed_edges"] == ROWS * 512


def test_bridge_knn_k_below_min_samples_is_raised_to_min_samples() -> None:
    knn = _native_knn(
        _fast_umap.estimate_hdbscan_resources(
            ROWS, KNN_FEATURES, backend="cpu", min_samples=25, knn_k=3
        )
    )
    assert knn["effective_width"] == 25


def test_bridge_width_is_capped_by_finite_row_count() -> None:
    knn = _native_knn(
        _fast_umap.estimate_hdbscan_resources(
            40, KNN_FEATURES, backend="cpu", min_samples=5, knn_k=512
        )
    )
    assert knn["effective_width"] == 39
    assert knn["directed_edges"] == 40 * 39


def test_bridge_auto_reports_conditional_cpu_and_cuda_scenarios() -> None:
    knn = _native_knn(
        _fast_umap.estimate_hdbscan_resources(
            ROWS, KNN_FEATURES, backend="auto", min_samples=5, knn_k=512
        )
    )
    assert {scenario["backend"] for scenario in knn["scenarios"]} == {"cpu", "cuda"}


def test_bridge_overflow_is_typed_and_allocates_nothing() -> None:
    # Input elements overflow usize before any allocation or graph routing.
    with pytest.raises(OverflowError):
        _fast_umap.estimate_hdbscan_resources(100, 2**63, backend="cpu", min_samples=5)


def test_bridge_rejects_rows_outside_device_index_range() -> None:
    with pytest.raises(ValueError, match=r"row count.*supported maximum"):
        _fast_umap.estimate_hdbscan_resources(2**40, KNN_FEATURES, backend="cpu", min_samples=5)


def test_bridge_rejects_invalid_kwargs_like_fit() -> None:
    with pytest.raises(TypeError):
        _fast_umap.estimate_hdbscan_resources(100, 3, not_a_parameter=1)
    with pytest.raises(ValueError):
        _fast_umap.estimate_hdbscan_resources(100, 3, knn_k=True)
    with pytest.raises(ValueError):
        _fast_umap.estimate_hdbscan_resources(100, 3, backend="gpu")


# Finite-row forwarding: the warning path sends the finite-row count, not the raw row count.


def test_finite_row_count_is_forwarded_to_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def record(*args: object, **kwargs: object) -> dict[str, Any]:
        calls.append((args, kwargs))
        return {"resolved_graph": "complete", "knn": None}

    monkeypatch.setattr(_fast_umap, "estimate_hdbscan_resources", record)
    data = _blobs(60, 3)
    data[0, 0] = np.nan
    data[1, 2] = np.inf
    HDBSCAN().fit(data)
    assert len(calls) == 1
    args, _ = calls[0]
    assert args[0] == 58
    assert args[1] == 3


# Python warning: at most one filterable UserWarning derived only from the native estimate.


def test_advisory_emits_one_user_warning_with_context(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    model = HDBSCAN(min_cluster_size=5, knn_k=15)
    with pytest.warns(UserWarning) as record:
        model.fit(_blobs())
    relevant = _advisories(record)
    assert len(relevant) == 1
    message = str(relevant[0].message)
    assert "60 finite rows" in message
    assert "effective width 15" in message
    assert "min_samples=5" in message
    assert "density" in message
    assert "GiB" in message
    assert "estimates, not total process usage" in message


def test_auto_backend_warning_says_scenarios_are_conditional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=15, backend="auto").fit(_blobs())
    message = str(_advisories(record)[0].message)
    assert "conditional" in message
    assert "not summed" in message
    assert "CPU host+runtime" in message
    assert "CUDA host" in message and "separate peaks" in message


def test_single_backend_warning_names_the_requested_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenarios = [{"backend": "cpu", "host_peak_bytes": GIB, "device_peak_bytes": 0}]
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory(), scenarios=scenarios))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=15, backend="cpu").fit(_blobs())
    message = str(_advisories(record)[0].message)
    assert "Backend 'cpu' was requested" in message
    assert "conditional" not in message


def test_floor_active_explains_min_samples_and_does_not_advise_lowering_knn_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory(floor=True)))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=2).fit(_blobs())
    message = str(_advisories(record)[0].message)
    assert "floor is in effect" in message
    assert "will not reduce it" in message
    assert "Consider lowering knn_k" not in message


def test_floor_inactive_advises_lowering_knn_k_with_min_samples_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory(floor=False)))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=15).fit(_blobs())
    message = str(_advisories(record)[0].message)
    assert "Consider lowering knn_k" in message
    assert "cannot be below min_samples=5" in message


def test_warning_points_at_fit_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=15).fit(_blobs())
    relevant = _advisories(record)
    assert len(relevant) == 1
    assert relevant[0].filename == __file__


def test_warning_points_at_fit_predict_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    with pytest.warns(UserWarning) as record:
        HDBSCAN(knn_k=15).fit_predict(_blobs())
    relevant = _advisories(record)
    assert len(relevant) == 1
    assert relevant[0].filename == __file__


def test_no_warning_below_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=None))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        HDBSCAN(knn_k=15).fit(_blobs())
    assert not _advisories(record)


def test_no_warning_on_complete_route(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, {"resolved_graph": "complete", "knn": None})
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        HDBSCAN().fit(_blobs())
    assert not _advisories(record)


def test_ignore_filter_suppresses_advisory(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("ignore")
        HDBSCAN(knn_k=15).fit(_blobs())
    assert not _advisories(record)


def test_error_filter_promotes_advisory_and_leaves_model_unfitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    model = HDBSCAN(knn_k=15)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="kNN graph"):
            model.fit(_blobs())
    with pytest.raises(ValueError, match="not fitted"):
        _ = model.labels_


def test_advisory_does_not_change_results_or_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    data = _blobs()
    expected_labels = HDBSCAN(min_cluster_size=5, knn_k=15).fit(data).labels_.copy()

    _patch_estimate(monkeypatch, _knn_estimate(advisory=_advisory()))
    model = HDBSCAN(min_cluster_size=5, knn_k=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(data)

    np.testing.assert_array_equal(model.labels_, expected_labels)
    assert model.knn_k == 15
    assert model.min_cluster_size == 5
    assert model.backend == "auto"
