from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

import cuscan
from cuscan import HDBSCAN, UMAP, _cuscan

FIXTURE_DIR = Path(__file__).parents[2] / "crates/hdbscan/tests/fixtures"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text())
PARITY_CASES = [
    case["id"]
    for case in MANIFEST["cases"]
    if case["status"] == "reference"
    and case["divergence"] == "none"
    and case["id"] not in {"blobs_d50", "blobs_d50_knn", "moons_leaf"}
]

# These case-specific assessments preserve the pinned Rust oracle as the gate.
# fast_hdbscan's Boruvka path switches to approximate neighbors above 30D;
# its public result is therefore diagnostic, not an exact-MST oracle. For
# moons_leaf, upstream and stored hierarchies have equivalent partitions at
# every measured distinct edge-weight batch, while leaf extraction differs by
# one membership; this is a fixture-scoped traversal/tie-sensitive assessment,
# not a claim about the cause of the divergence.
KNOWN_UPSTREAM_DIVERGENCES = {
    "blobs_d50": "upstream public estimator uses approximate neighbors above 30 dimensions",
    "moons_leaf": "measured leaf-extraction traversal/tie-sensitive membership difference",
}


def fixture_data(case_id: str) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    metadata = next(case for case in MANIFEST["cases"] if case["id"] == case_id)
    fixture_metadata = json.loads((FIXTURE_DIR / f"{case_id}.json").read_text())
    metadata["fixture"] = fixture_metadata
    with np.load(FIXTURE_DIR / f"{case_id}.npz") as arrays:
        x = arrays["input"].copy()
        expected = {key: arrays[key].copy() for key in arrays.files}
    return x, metadata, expected


def estimator_params(metadata: dict[str, Any]) -> dict[str, Any]:
    params = metadata["params"]
    return {
        "min_cluster_size": params["min_cluster_size"],
        "min_samples": params["min_samples"],
        "cluster_selection_method": params["selection"],
        "allow_single_cluster": params["allow_single_cluster"],
        "max_cluster_size": (
            float("inf") if params["max_cluster_size"] is None else params["max_cluster_size"]
        ),
        "cluster_selection_epsilon": params["epsilon"],
        "cluster_selection_persistence": (
            metadata["fixture"]["persistence_boundary_evidence"]["threshold"]
            if isinstance(params["persistence"], str)
            else params["persistence"]
        ),
    }


def estimator(backend: Literal["cpu", "cuda"], **kwargs: Any) -> HDBSCAN:
    return HDBSCAN(backend=backend, **kwargs)


def assert_close_with_infs(actual: np.ndarray, expected: np.ndarray, atol: float) -> None:
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    finite = np.isfinite(expected)
    np.testing.assert_allclose(actual[finite], expected[finite], atol=atol, rtol=0)


def partition_at(linkage: np.ndarray, threshold: float) -> frozenset[frozenset[int]]:
    n_leaves = len(linkage) + 1
    parent = list(range(2 * n_leaves - 1))

    def root(node: int) -> int:
        while parent[node] != node:
            node = parent[node]
        return node

    for index, row in enumerate(linkage):
        if row[2] > threshold:
            break
        parent[root(int(row[0]))] = n_leaves + index
        parent[root(int(row[1]))] = n_leaves + index
    components: dict[int, set[int]] = {}
    for leaf in range(n_leaves):
        components.setdefault(root(leaf), set()).add(leaf)
    return frozenset(frozenset(group) for group in components.values())


def label_assessment(reference: np.ndarray, actual: np.ndarray) -> dict[str, int | float]:
    return {
        "ari": float(adjusted_rand_score(reference, actual)),
        "changed_memberships": int(np.count_nonzero(reference != actual)),
        "reference_noise": int(np.count_nonzero(reference == -1)),
        "actual_noise": int(np.count_nonzero(actual == -1)),
    }


def align_cluster_labels(reference: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Match non-noise label IDs by maximum overlap, keeping noise fixed."""
    reference_ids = np.unique(reference[reference >= 0])
    actual_ids = np.unique(actual[actual >= 0])
    overlap = np.asarray(
        [
            [
                np.count_nonzero((reference == ref_id) & (actual == actual_id))
                for actual_id in actual_ids
            ]
            for ref_id in reference_ids
        ]
    )
    ref_indices, actual_indices = linear_sum_assignment(-overlap)
    mapping = {
        actual_ids[actual_index]: reference_ids[ref_index]
        for ref_index, actual_index in zip(ref_indices, actual_indices, strict=True)
    }
    aligned = np.full(actual.shape, -1, dtype=np.intp)
    for actual_id, reference_id in mapping.items():
        aligned[actual == actual_id] = reference_id
    return aligned


def raise_injected_native_error(*args: Any, **kwargs: Any) -> None:
    raise RuntimeError("injected")


@pytest.mark.parametrize(
    ("case_id", "probability_dtype", "min_samples"),
    [("blobs_d2", np.float32, 5), ("mixed_nonfinite", np.float64, 3)],
)
def test_output_dtype_and_shape_contract(
    case_id: str,
    probability_dtype: type[np.floating[Any]],
    min_samples: int,
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    assert "HDBSCAN" in cuscan.__all__
    x, _, _ = fixture_data(case_id)
    model = estimator(hdbscan_backend, min_cluster_size=min_samples, min_samples=min_samples).fit(x)
    finite_count = int(np.isfinite(x).all(axis=1).sum())
    assert model.labels_.dtype == np.dtype(np.intp)
    assert model.labels_.shape == (len(x),)
    assert model.probabilities_.dtype == probability_dtype
    assert model.probabilities_.shape == (len(x),)
    assert model.condensed_tree_.dtype == np.dtype(
        [
            ("parent", np.intp),
            ("child", np.intp),
            ("lambda_val", np.float64),
            ("child_size", np.float32),
        ]
    )
    assert model.single_linkage_tree_.dtype == np.float64
    assert model.single_linkage_tree_.shape[1:] == (4,)
    assert model.minimum_spanning_tree_.dtype == np.float64
    assert model.minimum_spanning_tree_.shape[1:] == (3,)
    assert model.neighbors_.dtype == np.dtype(np.intp)
    assert model.neighbors_.shape == (finite_count, min_samples)
    assert model.core_distances_.dtype == np.float32
    assert model.core_distances_.shape == (finite_count,)
    assert model.finite_index_.dtype == np.dtype(np.intp)
    assert model.finite_index_.shape == (finite_count,)
    cut = model.dbscan_clustering(0.0)
    assert cut.dtype == np.dtype(np.intp) and cut.shape == (len(x),)


def test_nonfinite_remap_preserves_raw_row_indexing(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, metadata, _ = fixture_data("mixed_nonfinite")
    model = estimator(hdbscan_backend, min_cluster_size=3, min_samples=3).fit(x)
    upstream = pytest.importorskip("fast_hdbscan")
    reference = upstream.HDBSCAN(**estimator_params(metadata)).fit(x)
    assert model.labels_.dtype == np.dtype(np.intp)
    np.testing.assert_array_equal(model.finite_index_, np.flatnonzero(np.isfinite(x).all(axis=1)))
    outliers = np.setdiff1d(np.arange(len(x)), model.finite_index_)
    assert np.all(model.labels_[outliers] == -1)
    assert np.all(model.probabilities_[outliers] == 0.0)
    assert model.neighbors_.shape == (len(model.finite_index_), 3)
    np.testing.assert_allclose(model.core_distances_, reference._core_distances, rtol=1e-6, atol=0)
    assert_close_with_infs(
        model.minimum_spanning_tree_[:, 2], reference._min_spanning_tree[:, 2], 1e-5
    )


@pytest.mark.parametrize("case_id", PARITY_CASES)
def test_phase_one_fixtures_match_upstream(
    case_id: str, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    upstream = pytest.importorskip("fast_hdbscan")
    x, metadata, expected = fixture_data(case_id)
    params = estimator_params(metadata)
    model = estimator(hdbscan_backend, **params).fit(x)
    reference = upstream.HDBSCAN(**params).fit(x)
    assert adjusted_rand_score(reference.labels_, model.labels_) == 1.0
    assert adjusted_rand_score(expected["labels"], model.labels_) == 1.0
    assert_close_with_infs(model.probabilities_, reference.probabilities_, 1e-5)
    assert_close_with_infs(model.probabilities_, expected["probabilities"], 1e-5)
    assert_close_with_infs(
        model.minimum_spanning_tree_[:, 2],
        reference._min_spanning_tree[:, 2],
        1e-5,
    )
    assert_close_with_infs(model.minimum_spanning_tree_[:, 2], expected["mst_weights"], 1e-5)
    np.testing.assert_allclose(model.core_distances_, expected["core_distances"], rtol=1e-6, atol=0)
    np.testing.assert_allclose(model.core_distances_, reference._core_distances, rtol=1e-6, atol=0)


@pytest.mark.parametrize("case_id", ["moons_leaf"])
def test_known_upstream_divergences_keep_exact_fixture_as_gate(
    case_id: str, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    upstream = pytest.importorskip("fast_hdbscan")
    x, metadata, expected = fixture_data(case_id)
    params = estimator_params(metadata)
    model = estimator(hdbscan_backend, **params).fit(x)

    # Phase 1 generated these arrays from the pinned upstream exact oracle; they remain
    # strict despite the unrelated public-upstream differences.
    assert adjusted_rand_score(expected["labels"], model.labels_) == 1.0
    assert_close_with_infs(model.probabilities_, expected["probabilities"], 1e-5)
    assert_close_with_infs(model.minimum_spanning_tree_[:, 2], expected["mst_weights"], 1e-5)
    np.testing.assert_allclose(model.core_distances_, expected["core_distances"], rtol=1e-6, atol=0)

    reference = upstream.HDBSCAN(**params).fit(x)

    upstream_mst = np.asarray(reference._min_spanning_tree, dtype=np.float64)
    upstream_cores = np.asarray(reference._core_distances, dtype=np.float64)
    upstream_labels = np.asarray(reference.labels_)
    diagnostics: dict[str, Any] = {
        "case": case_id,
        "reason": KNOWN_UPSTREAM_DIVERGENCES[case_id],
        "label_assessment": label_assessment(expected["labels"], upstream_labels),
        "core_distance_max_abs_delta": float(
            np.max(np.abs(expected["core_distances"] - upstream_cores))
        ),
        "mst_weight_max_abs_delta": float(
            np.max(np.abs(expected["mst_weights"] - upstream_mst[:, 2]))
        ),
    }
    if case_id == "moons_leaf":
        stored_linkage = np.asarray(expected["linkage"], dtype=np.float64)
        upstream_linkage = np.asarray(reference._single_linkage_tree, dtype=np.float64)
        all_thresholds = sorted(set(stored_linkage[:, 2]) | set(upstream_linkage[:, 2]))
        mismatched_thresholds = [
            threshold
            for threshold in all_thresholds
            if partition_at(stored_linkage, threshold) != partition_at(upstream_linkage, threshold)
        ]
        # On this fixture the complete hierarchy is equivalent at every
        # distinct edge-weight batch; the observed difference is at extraction.
        assert not mismatched_thresholds
        assert len(all_thresholds) == 28
        diagnostics["hierarchy_batches"] = {
            "stored_distinct_weight_batches": len(np.unique(stored_linkage[:, 2])),
            "upstream_distinct_weight_batches": len(np.unique(upstream_linkage[:, 2])),
            "checked_union_thresholds": len(all_thresholds),
            "mismatched_thresholds": mismatched_thresholds,
        }
        assert adjusted_rand_score(expected["labels"], upstream_labels) >= (
            0.9091063180643247 - 1e-12
        )
        assert_array_noise = np.array_equal(expected["labels"] == -1, upstream_labels == -1)
        assert assert_array_noise
        aligned_upstream = align_cluster_labels(expected["labels"], upstream_labels)
        changed_rows = np.flatnonzero(expected["labels"] != aligned_upstream)
        assert len(changed_rows) <= 1
        assert set(changed_rows).issubset({22})
        expected_clusters = np.unique(expected["labels"][expected["labels"] >= 0])
        upstream_clusters = np.unique(upstream_labels[upstream_labels >= 0])
        contingency = np.asarray(
            [
                [
                    np.count_nonzero(
                        (expected["labels"] == expected_cluster)
                        & (upstream_labels == upstream_cluster)
                    )
                    for upstream_cluster in upstream_clusters
                ]
                for expected_cluster in expected_clusters
            ]
        )
        diagnostics["cluster_overlap"] = contingency.tolist()
        assert_close_with_infs(model.minimum_spanning_tree_[:, 2], upstream_mst[:, 2], 1e-5)
        np.testing.assert_allclose(model.core_distances_, upstream_cores, rtol=1e-6, atol=0)
        # The measured 0.06672919 maximum is rounded upward to a narrowly
        # scoped 0.067 bound; this does not relax stored-oracle tolerances.
        assert np.max(np.abs(expected["probabilities"] - reference.probabilities_)) <= 0.067
        diagnostics["upstream_mst_label_assessment"] = label_assessment(
            expected["labels"], upstream_labels
        )
        diagnostics["leaf_membership_assessment"] = {
            "noise_mask_equal": bool(assert_array_noise),
            "changed_rows": changed_rows.tolist(),
            "probability_max_abs_delta": float(
                np.max(np.abs(expected["probabilities"] - reference.probabilities_))
            ),
        }
    warnings.warn(
        f"informational upstream comparison: {json.dumps(diagnostics, sort_keys=True)}",
        stacklevel=1,
    )


def test_knn_graph_default_route_matches_fixture(hdbscan_backend: Literal["cpu", "cuda"]) -> None:
    # Default config resolves this d=50 fixture to the KnnGraph route; the stored oracle is the
    # pynndescent kNN graph validated exactly against numpy during generation.
    x, metadata, expected = fixture_data("blobs_d50_knn")
    model = estimator(hdbscan_backend, **estimator_params(metadata)).fit(x)
    assert adjusted_rand_score(expected["labels"], model.labels_) == 1.0
    assert_close_with_infs(model.probabilities_, expected["probabilities"], 1e-5)
    assert_close_with_infs(model.minimum_spanning_tree_[:, 2], expected["mst_weights"], 1e-5)
    np.testing.assert_allclose(model.core_distances_, expected["core_distances"], rtol=1e-6, atol=0)


@pytest.mark.parametrize("case_id", ["duplicates_ties", "persistence_equal"])
def test_annotated_tie_fixtures_use_tie_safe_checks(
    case_id: str,
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    # These fixtures annotate candidate/merge ties and a selection boundary.
    # Do not impose untied traversal-order label comparisons on them.
    x, metadata, expected = fixture_data(case_id)
    model = estimator(hdbscan_backend, **estimator_params(metadata)).fit(x)
    assert model.minimum_spanning_tree_.shape == expected["mst"].shape
    assert_close_with_infs(model.minimum_spanning_tree_[:, 2], expected["mst_weights"], 1e-5)
    np.testing.assert_allclose(model.core_distances_, expected["core_distances"], rtol=1e-6, atol=0)
    upstream = pytest.importorskip("fast_hdbscan")
    reference = upstream.HDBSCAN(**estimator_params(metadata)).fit(x)
    assert_close_with_infs(
        model.minimum_spanning_tree_[:, 2], reference._min_spanning_tree[:, 2], 1e-5
    )
    np.testing.assert_allclose(model.core_distances_, reference._core_distances, rtol=1e-6, atol=0)
    assert model.labels_.shape == (len(x),)


@pytest.mark.parametrize("case_id", [*PARITY_CASES, "blobs_d50", "blobs_d50_knn", "moons_leaf"])
def test_cuda_and_cpu_match_on_phase_one_fixtures(
    case_id: str, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    if hdbscan_backend != "cuda":
        pytest.skip("cross-backend parity runs in the CUDA lane")
    x, metadata, _ = fixture_data(case_id)
    params = estimator_params(metadata)
    cpu = estimator("cpu", **params).fit(x)
    cuda = estimator("cuda", **params).fit(x)
    assert adjusted_rand_score(cpu.labels_, cuda.labels_) == 1.0
    assert_close_with_infs(cuda.probabilities_, cpu.probabilities_, 1e-5)
    assert_close_with_infs(
        cuda.minimum_spanning_tree_[:, 2], cpu.minimum_spanning_tree_[:, 2], 1e-5
    )
    np.testing.assert_allclose(cuda.core_distances_, cpu.core_distances_, rtol=1e-6, atol=0)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_cluster_size": 0}, ValueError),
        ({"min_samples": 0}, ValueError),
        ({"cluster_selection_method": "invalid"}, ValueError),
        ({"cluster_selection_epsilon": 0}, ValueError),
        ({"cluster_selection_persistence": 0}, ValueError),
        ({"metric": "manhattan"}, NotImplementedError),
        ({"algorithm": "kruskal"}, NotImplementedError),
        ({"cannot_link": object()}, NotImplementedError),
        ({"cannot_link_groups": np.array([0, 1])}, NotImplementedError),
        ({"semi_supervised": True}, NotImplementedError),
        ({"ss_algorithm": "bc_simple"}, NotImplementedError),
        ({"validate_cannot_link": False}, NotImplementedError),
        ({"metric_kwds": {"p": 2}}, NotImplementedError),
    ],
)
def test_parameter_validation(
    kwargs: dict[str, Any],
    error: type[Exception],
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    with pytest.raises(error):
        estimator(hdbscan_backend, **kwargs).fit(x)


@pytest.mark.parametrize("x", [np.ones(4), np.ones((2, 2, 2)), np.array([["a"]])])
def test_invalid_input_shape_and_type(
    x: np.ndarray, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    with pytest.raises(ValueError):
        estimator(hdbscan_backend).fit(x)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cluster_selection_epsilon": -0.1},
        {"cluster_selection_persistence": -0.1},
        {"cluster_selection_epsilon": float("inf")},
    ],
)
def test_parameter_value_ranges(
    kwargs: dict[str, Any], hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    with pytest.raises(ValueError):
        estimator(hdbscan_backend, **kwargs).fit(x)


def test_sample_weights_are_explicitly_unsupported(hdbscan_backend: Literal["cpu", "cuda"]) -> None:
    x, _, _ = fixture_data("blobs_d2")
    with pytest.raises(NotImplementedError):
        estimator(hdbscan_backend).fit(x, sample_weight=np.ones(len(x)))


def test_zero_finite_and_finite_overflow_rejected(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    with pytest.raises(ValueError):
        estimator(hdbscan_backend).fit(np.full((3, 2), np.nan))
    with pytest.raises(ValueError):
        estimator(hdbscan_backend).fit(np.array([[1e100, 0.0], [0.0, 0.0]]))
    model = estimator(hdbscan_backend).fit(np.array([[np.inf, 0.0], [0.0, 0.0], [1.0, 1.0]]))
    assert model.labels_[0] == -1


def test_single_finite_and_too_few_finite_contract(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    one = estimator(hdbscan_backend, min_samples=2).fit(np.array([[1.0, 2.0], [np.nan, 0.0]]))
    assert one.neighbors_.shape == (1, 0)
    assert np.isinf(one.core_distances_[0])
    few = estimator(hdbscan_backend, min_samples=4, min_cluster_size=4).fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 0.0]])
    )
    assert np.all(few.labels_ == -1)
    assert np.all(np.isinf(few.core_distances_))


def test_dbscan_cut_edges_and_unfitted_state(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    model = estimator(hdbscan_backend)
    with pytest.raises(ValueError):
        model.dbscan_clustering(0.2)
    x, _, _ = fixture_data("blobs_d2")
    singleton_model = estimator(hdbscan_backend, min_samples=1).fit(x)
    # Rust/upstream cuts filter components smaller than resolved min_samples;
    # one-neighbor cuts retain singleton components at nonpositive epsilon.
    zero_cut = singleton_model.dbscan_clustering(0.0)
    negative_cut = singleton_model.dbscan_clustering(-1.0)
    assert len(zero_cut) == len(x) and len(np.unique(zero_cut)) == len(x)
    assert len(negative_cut) == len(x) and len(np.unique(negative_cut)) == len(x)
    model.fit(x)
    assert np.all(model.dbscan_clustering(0.0) == -1)
    infinite_cut = singleton_model.dbscan_clustering(float("inf"))
    assert len(infinite_cut) == len(x)
    assert len(np.unique(infinite_cut)) == 1
    upstream = pytest.importorskip("fast_hdbscan")
    reference = upstream.HDBSCAN(min_cluster_size=5, min_samples=1).fit(x)
    np.testing.assert_array_equal(zero_cut, reference.dbscan_clustering(0.0))
    np.testing.assert_array_equal(negative_cut, reference.dbscan_clustering(-1.0))
    np.testing.assert_array_equal(infinite_cut, reference.dbscan_clustering(float("inf")))
    with pytest.raises(ValueError):
        model.dbscan_clustering(float("nan"))


def test_wrapper_native_failure_is_injected_on_called_module(
    monkeypatch: pytest.MonkeyPatch, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    import cuscan.hdbscan as hdbscan_module

    x, _, _ = fixture_data("blobs_d2")
    model = estimator(hdbscan_backend).fit(x)
    old_native = model._native
    monkeypatch.setattr(
        hdbscan_module._cuscan,
        "fit_hdbscan",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected")),
    )
    with pytest.raises(RuntimeError, match="injected"):
        model.fit(x)
    assert model._native is old_native


def test_returned_arrays_are_independent_copies(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    model = estimator(hdbscan_backend).fit(x)
    attrs = [
        "labels_",
        "probabilities_",
        "condensed_tree_",
        "single_linkage_tree_",
        "minimum_spanning_tree_",
        "neighbors_",
        "core_distances_",
        "finite_index_",
    ]
    for attr in attrs:
        first = getattr(model, attr)
        original = first.copy()
        if first.size:
            first.flat[0] = 0
        np.testing.assert_array_equal(getattr(model, attr), original)


@pytest.mark.parametrize(
    "failure",
    ["invalid", "unsupported", "overflow", "native"],
)
def test_failed_refit_preserves_fitted_state(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    model = estimator(hdbscan_backend).fit(x)
    before = {
        name: getattr(model, name).copy()
        for name in (
            "labels_",
            "probabilities_",
            "condensed_tree_",
            "single_linkage_tree_",
            "minimum_spanning_tree_",
            "neighbors_",
            "core_distances_",
            "finite_index_",
        )
    }
    cut_before = model.dbscan_clustering(0.2).copy()
    native = model._native
    refit_input = x
    if failure == "invalid":
        model.min_cluster_size = 0
        error: type[Exception] = ValueError
    elif failure == "unsupported":
        model.metric = "manhattan"
        error = NotImplementedError
    elif failure == "overflow":
        refit_input = np.array([[1e100, 0.0], [0.0, 0.0]])
        error = ValueError
    else:
        from cuscan.hdbscan import _cuscan as native_module

        monkeypatch.setattr(
            native_module,
            "fit_hdbscan",
            raise_injected_native_error,
        )
        error = RuntimeError
    with pytest.raises(error):
        model.fit(refit_input)
    assert model._native is native
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    np.testing.assert_array_equal(model.dbscan_clustering(0.2), cut_before)


def test_successful_refit_replaces_state(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    model = estimator(hdbscan_backend).fit(x)
    old_native = model._native
    model.fit(np.concatenate([x, x + 100.0]))
    assert model._native is not old_native
    assert model.labels_.shape == (2 * len(x),)


@pytest.fixture
def kd_region_data(request: pytest.FixtureRequest) -> np.ndarray:
    rows, dims = request.param
    rng = np.random.default_rng(20241023 + rows + dims)
    centers = rng.normal(scale=8.0, size=(4, dims))
    assignment = rng.integers(0, len(centers), size=rows)
    noise = rng.normal(scale=1.0, size=(rows, dims))
    return np.ascontiguousarray(centers[assignment] + noise, dtype=np.float32)


def assert_bitwise_equal(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("kd_region_data", "min_samples"),
    [
        ((256, 2), 5),
        ((512, 12), 5),
        ((2048, 30), 32),
    ],
    indirect=["kd_region_data"],
)
def test_reproducible_kd_route_is_bitwise_identical_to_default(
    kd_region_data: np.ndarray,
    min_samples: int,
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    ordinary = estimator(hdbscan_backend, min_samples=min_samples).fit(kd_region_data)
    reproducible = estimator(hdbscan_backend, min_samples=min_samples, reproducible=True).fit(
        kd_region_data
    )
    for name in (
        "labels_",
        "probabilities_",
        "condensed_tree_",
        "single_linkage_tree_",
        "minimum_spanning_tree_",
        "neighbors_",
        "core_distances_",
        "finite_index_",
    ):
        assert_bitwise_equal(getattr(ordinary, name), getattr(reproducible, name))


def test_knn_k_is_forwarded_to_knn_graph_route(hdbscan_backend: Literal["cpu", "cuda"]) -> None:
    # knn_k=15 is the stored width of the d=50 kNN-graph fixture; the explicit value must
    # reproduce the same strict fixture assertions as the default route.
    x, metadata, expected = fixture_data("blobs_d50_knn")
    model = estimator(hdbscan_backend, **estimator_params(metadata), knn_k=15).fit(x)
    assert adjusted_rand_score(expected["labels"], model.labels_) == 1.0
    assert_close_with_infs(model.probabilities_, expected["probabilities"], 1e-5)
    assert_close_with_infs(model.minimum_spanning_tree_[:, 2], expected["mst_weights"], 1e-5)
    np.testing.assert_allclose(model.core_distances_, expected["core_distances"], rtol=1e-6, atol=0)


@pytest.mark.parametrize("knn_k", [0, -1, True, 2.0])
def test_knn_k_rejects_non_positive_integers(
    knn_k: Any, hdbscan_backend: Literal["cpu", "cuda"]
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    with pytest.raises(ValueError):
        estimator(hdbscan_backend, knn_k=knn_k).fit(x)
    with pytest.raises(ValueError):
        _cuscan.fit_hdbscan(x, knn_k=knn_k)


def test_native_knn_k_accepts_none_and_positive_integers() -> None:
    x, _, _ = fixture_data("blobs_d2")
    assert _cuscan.fit_hdbscan(x, knn_k=None).labels.shape == (len(x),)
    assert _cuscan.fit_hdbscan(x, knn_k=3).labels.shape == (len(x),)


def test_explicit_cuda_does_not_fall_back(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    if _cuscan.hdbscan_cuda_available():
        pytest.skip("requires a build or host without CUDA support")
    x, _, _ = fixture_data("blobs_d2")
    with pytest.raises(RuntimeError):
        estimator("cuda").fit(x)


def test_composed_umap_then_hdbscan(
    hdbscan_backend: Literal["cpu", "cuda"],
) -> None:
    x, _, _ = fixture_data("blobs_d2")
    embedding = UMAP(
        n_neighbors=5,
        n_epochs=20,
        random_state=7,
        backend=hdbscan_backend,
    ).fit_transform(x)
    labels = estimator(hdbscan_backend).fit_predict(embedding)
    assert labels.shape == (len(x),)
    assert labels.dtype == np.dtype(np.intp)
