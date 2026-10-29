from __future__ import annotations

import os
import warnings
from typing import Any, Literal, TypedDict, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _cuscan


class _FittedArrays(TypedDict):
    labels: NDArray[np.intp]
    probabilities: NDArray[np.float32] | NDArray[np.float64]
    condensed_tree: NDArray
    single_linkage_tree: NDArray[np.float64]
    minimum_spanning_tree: NDArray[np.float64]
    neighbors: NDArray[np.intp]
    core_distances: NDArray[np.float32]
    finite_index: NDArray[np.intp]


class _FittedState(TypedDict):
    arrays: _FittedArrays
    n_features_in: int
    native: _cuscan.FittedHDBSCAN


def _approximate_predict(
    native: Any,
    n_features: int,
    X: ArrayLike,
    return_connecting_points: bool | np.bool_,
) -> (
    tuple[NDArray[np.intp], NDArray[np.float64]]
    | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
):
    if not isinstance(return_connecting_points, (bool, np.bool_)):
        raise ValueError("return_connecting_points must be a boolean")
    x, finite_rows = HDBSCAN._input_array(X)
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must contain at least one row and one feature")
    if not finite_rows.all():
        raise ValueError("X must contain only finite values")
    if x.shape[1] != n_features:
        raise ValueError(f"X has {x.shape[1]} features, but the model was fitted with {n_features}")
    # This helper adds one Python frame between the native warning and the caller.
    if bool(return_connecting_points):
        raw_labels, raw_strengths, raw_connectors = native.approximate_predict(
            x, True, stacklevel=3
        )
        return (
            np.array(raw_labels, dtype=np.intp, copy=True),
            np.array(raw_strengths, dtype=np.float64, copy=True),
            np.array(raw_connectors, dtype=np.intp, copy=True),
        )
    raw_labels, raw_strengths = native.approximate_predict(x, False, stacklevel=3)
    return (
        np.array(raw_labels, dtype=np.intp, copy=True),
        np.array(raw_strengths, dtype=np.float64, copy=True),
    )


class HDBSCANPredictor:
    """Lightweight detached predictor for an already-fitted HDBSCAN hierarchy."""

    __slots__ = ("_n_features_in", "_native_predictor")

    def __init__(self) -> None:
        raise TypeError(
            "HDBSCANPredictor instances are created by HDBSCAN.to_predictor() or load()"
        )

    @classmethod
    def _from_native(cls, native: _cuscan.HDBSCANPredictor) -> HDBSCANPredictor:
        predictor = cls.__new__(cls)
        predictor._native_predictor = native
        predictor._n_features_in = int(native.n_features)
        return predictor

    @property
    def n_features_in_(self) -> int:
        return self._n_features_in

    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: Literal[False] = ...
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]: ...
    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: Literal[True]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]: ...
    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: bool | np.bool_ = ...
    ) -> (
        tuple[NDArray[np.intp], NDArray[np.float64]]
        | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
    ): ...
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: bool | np.bool_ = False
    ) -> (
        tuple[NDArray[np.intp], NDArray[np.float64]]
        | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
    ):
        """Predict into the fixed hierarchy and return new arrays in query order."""
        return _approximate_predict(
            self._native_predictor,
            self._n_features_in,
            X,
            return_connecting_points,
        )

    def save(self, path: str | os.PathLike[str], overwrite: bool = False) -> None:
        """Save this ready prediction-only state to the optional HDF5 format."""
        arrays, metadata = self._native_predictor.export_state()
        from . import _hdbscan_predictor_serialization_v1 as serialization

        serialization.save_model(path, arrays, metadata, overwrite=overwrite)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        backend: Literal["auto", "cpu", "cuda"] = "auto",
        device_index: int = 0,
    ) -> HDBSCANPredictor:
        """Load a prediction-only HDF5 state with strict runtime backend selection."""
        from . import _hdbscan_predictor_serialization_v1 as serialization

        return serialization.load_model(cls, path, backend=backend, device_index=device_index)


_ArrayName = Literal[
    "labels",
    "probabilities",
    "condensed_tree",
    "single_linkage_tree",
    "minimum_spanning_tree",
    "neighbors",
    "core_distances",
    "finite_index",
]


class HDBSCAN:
    """Hierarchical density-based clustering backed by the native cuscan engine.

    ``reproducible=True`` gives the complete-graph KD route independent per-row bounds, so
    repeated fits use the same reduction order. It does not change the algorithm.

    ``knn_k`` sets the non-self neighbor width of the kNN graph route, used by the default
    ``Auto`` graph above 30 features. ``None`` uses ``max(3 * min_samples, 15)``; any width is
    raised to at least ``min_samples`` and capped at ``n_samples - 1``.

    Before a kNN-route fit, one ``UserWarning`` is emitted when the native estimate reaches
    1 GiB (2**30 bytes) of host or CUDA-device allocation. It is a usability advisory, not a
    safety boundary: fitting continues with the same settings, and warning filters can suppress
    or escalate it. The estimate is conditional on the backend (``Auto`` reports both CPU and
    CUDA scenarios) and does not measure available memory. Complete-graph fits emit no advisory.
    """

    _fitted_state: _FittedState

    def _fitted_array(self, name: _ArrayName) -> NDArray:
        if not hasattr(self, "_fitted_state"):
            raise ValueError("This HDBSCAN instance is not fitted yet")
        return np.array(self._fitted_state["arrays"][name], copy=True)

    @property
    def labels_(self) -> NDArray[np.intp]:
        return self._fitted_array("labels")

    @property
    def probabilities_(self) -> NDArray[np.float32] | NDArray[np.float64]:
        return self._fitted_array("probabilities")

    @property
    def condensed_tree_(self) -> NDArray:
        return self._fitted_array("condensed_tree")

    @property
    def single_linkage_tree_(self) -> NDArray[np.float64]:
        return self._fitted_array("single_linkage_tree")

    @property
    def minimum_spanning_tree_(self) -> NDArray[np.float64]:
        return self._fitted_array("minimum_spanning_tree")

    @property
    def neighbors_(self) -> NDArray[np.intp]:
        return self._fitted_array("neighbors")

    @property
    def core_distances_(self) -> NDArray[np.float32]:
        return self._fitted_array("core_distances")

    @property
    def finite_index_(self) -> NDArray[np.intp]:
        return self._fitted_array("finite_index")

    @property
    def n_features_in_(self) -> int:
        if not hasattr(self, "_fitted_state"):
            raise ValueError("This HDBSCAN instance is not fitted yet")
        return self._fitted_state["n_features_in"]

    @property
    def _native(self) -> _cuscan.FittedHDBSCAN:
        if not hasattr(self, "_fitted_state"):
            raise ValueError("This HDBSCAN instance is not fitted yet")
        return self._fitted_state["native"]

    def to_predictor(self) -> HDBSCANPredictor:
        """Prepare and return a detached prediction-only view of this fitted model.

        Conversion can synchronously prepare prediction cores on the fit-time
        backend and can fail before returning. The predictor shares no estimator
        wrapper or fit-output arrays and remains usable after this estimator is
        dropped or refitted.
        """
        return HDBSCANPredictor._from_native(self._native.to_predictor())

    def __init__(
        self,
        *,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
        allow_single_cluster: bool = False,
        max_cluster_size: int | float | None = None,
        cluster_selection_epsilon: float = 0.0,
        cluster_selection_persistence: float = 0.0,
        backend: Literal["auto", "cpu", "cuda"] = "auto",
        device_index: int = 0,
        reproducible: bool = False,
        semi_supervised: bool = False,
        ss_algorithm: str = "bc",
        metric: str = "euclidean",
        algorithm: str = "boruvka",
        knn_k: int | None = None,
        cannot_link: object | None = None,
        validate_cannot_link: bool = True,
        cannot_link_groups: object | None = None,
        metric_kwds: object | None = None,
    ) -> None:
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.max_cluster_size = max_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.cluster_selection_persistence = cluster_selection_persistence
        self.backend = backend
        self.device_index = device_index
        self.reproducible = reproducible
        self.semi_supervised = semi_supervised
        self.ss_algorithm = ss_algorithm
        self.metric = metric
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.cannot_link = cannot_link
        self.validate_cannot_link = validate_cannot_link
        self.cannot_link_groups = cannot_link_groups
        self.metric_kwds = metric_kwds

    @staticmethod
    def _input_array(X: ArrayLike) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        try:
            values = np.asarray(X)
        except (TypeError, ValueError) as error:
            raise ValueError("X must be a numeric 2-D array") from error
        if values.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
            values.dtype, np.complexfloating
        ):
            raise ValueError("X must be a numeric 2-D array")
        try:
            values64 = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("X must be a numeric 2-D array") from error

        finite_rows = np.isfinite(values64).all(axis=1)
        f32_max = np.finfo(np.float32).max
        finite_values = values64[np.isfinite(values64)]
        if np.any(np.abs(finite_values) > f32_max):
            raise ValueError("finite values in X must fit in float32")
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.ascontiguousarray(values64, dtype=np.float32)
        return result, finite_rows

    @staticmethod
    def _positive_integer(name: str, value: object, *, allow_none: bool = False) -> None:
        if allow_none and value is None:
            return
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            suffix = " or None" if allow_none else ""
            raise ValueError(f"{name} must be a positive integer{suffix}")
        if int(value) <= 0:
            suffix = " or None" if allow_none else ""
            raise ValueError(f"{name} must be a positive integer{suffix}")

    def _validate_parameters(self) -> None:
        self._positive_integer("min_cluster_size", self.min_cluster_size)
        self._positive_integer("min_samples", self.min_samples, allow_none=True)
        for name, value in (
            ("cluster_selection_epsilon", self.cluster_selection_epsilon),
            ("cluster_selection_persistence", self.cluster_selection_persistence),
        ):
            if not np.issubdtype(type(value), np.floating):
                raise ValueError(f"{name} must be a floating point number")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.cluster_selection_method not in ("eom", "leaf"):
            raise ValueError("cluster_selection_method must be 'eom' or 'leaf'")
        if not isinstance(self.allow_single_cluster, (bool, np.bool_)):
            raise ValueError("allow_single_cluster must be a boolean")
        if self.max_cluster_size is not None and not (
            isinstance(self.max_cluster_size, (float, np.floating))
            and np.isposinf(self.max_cluster_size)
        ):
            self._positive_integer("max_cluster_size", self.max_cluster_size)
        if self.backend not in ("auto", "cpu", "cuda"):
            raise ValueError("backend must be 'auto', 'cpu' or 'cuda'")
        if not np.issubdtype(type(self.device_index), np.integer) or self.device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        if not isinstance(self.reproducible, (bool, np.bool_)):
            raise ValueError("reproducible must be a boolean")
        self._positive_integer("knn_k", self.knn_k, allow_none=True)

        unsupported = (
            ("semi_supervised", self.semi_supervised, False),
            ("ss_algorithm", self.ss_algorithm, "bc"),
            ("metric", self.metric, "euclidean"),
            ("algorithm", self.algorithm, "boruvka"),
            ("cannot_link", self.cannot_link, None),
            ("validate_cannot_link", self.validate_cannot_link, True),
            ("cannot_link_groups", self.cannot_link_groups, None),
            ("metric_kwds", self.metric_kwds, None),
        )
        for name, value, default in unsupported:
            if default is None:
                is_default = value is None
            elif isinstance(default, bool):
                is_default = value is default
            else:
                is_default = value == default
            if not is_default:
                raise NotImplementedError(f"{name} is not supported")

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | float | None = None,
        **fit_params: object,
    ) -> HDBSCAN:
        return self._fit(X, y, sample_weight, fit_params)

    def _fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None,
        sample_weight: ArrayLike | float | None,
        fit_params: dict[str, object],
    ) -> HDBSCAN:
        if y is not None:
            raise NotImplementedError("y and semi-supervised fitting are not supported")
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported")
        if fit_params:
            name = next(iter(fit_params))
            raise NotImplementedError(f"fit parameter {name!r} is not supported")
        self._validate_parameters()
        x, finite_rows = self._input_array(X)
        if not np.any(finite_rows):
            raise ValueError("X must contain at least one fully finite row")

        native_kwargs = self._native_kwargs()
        estimate = _cuscan.estimate_hdbscan_resources(
            int(np.count_nonzero(finite_rows)),
            int(x.shape[1]),
            **native_kwargs,
        )
        self._warn_resource_advisory(estimate)
        fitted = _cuscan.fit_hdbscan(x, **native_kwargs)

        labels = np.array(fitted.labels, dtype=np.intp, copy=True)
        probabilities_raw = np.array(fitted.probabilities, dtype=np.float32, copy=True)
        if finite_rows.all():
            probabilities = probabilities_raw
        else:
            probabilities = np.zeros(x.shape[0], dtype=np.float64)
            probabilities[finite_rows] = probabilities_raw[finite_rows]
        condensed_tree = np.array(
            fitted.condensed_tree,
            dtype=[
                ("parent", np.intp),
                ("child", np.intp),
                ("lambda_val", np.float64),
                ("child_size", np.float32),
            ],
            copy=True,
        )
        single_linkage_tree = np.array(fitted.single_linkage_tree, dtype=np.float64, copy=True)
        minimum_spanning_tree = np.array(fitted.minimum_spanning_tree, dtype=np.float64, copy=True)
        neighbors = np.array(fitted.neighbors, dtype=np.intp, copy=True)
        core_distances = np.array(fitted.core_distances, dtype=np.float32, copy=True)
        finite_index = np.array(fitted.finite_index, dtype=np.intp, copy=True)

        # Publish only after native fitting and every derived attribute has succeeded.
        self._fitted_state = {
            "arrays": {
                "labels": labels,
                "probabilities": probabilities,
                "condensed_tree": condensed_tree,
                "single_linkage_tree": single_linkage_tree,
                "minimum_spanning_tree": minimum_spanning_tree,
                "neighbors": neighbors,
                "core_distances": core_distances,
                "finite_index": finite_index,
            },
            "n_features_in": x.shape[1],
            "native": fitted,
        }
        return self

    def _native_kwargs(self) -> dict[str, object]:
        max_cluster_size = self.max_cluster_size
        if max_cluster_size is not None and np.isposinf(max_cluster_size):
            max_cluster_size = None
        return {
            "min_cluster_size": int(self.min_cluster_size),
            "min_samples": None if self.min_samples is None else int(self.min_samples),
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": bool(self.allow_single_cluster),
            "max_cluster_size": None if max_cluster_size is None else int(max_cluster_size),
            "cluster_selection_epsilon": float(self.cluster_selection_epsilon),
            "cluster_selection_persistence": float(self.cluster_selection_persistence),
            "backend": self.backend,
            "device_index": int(self.device_index),
            "reproducible": bool(self.reproducible),
            "knn_k": None if self.knn_k is None else int(self.knn_k),
        }

    @staticmethod
    def _warn_resource_advisory(estimate: dict[str, object]) -> None:
        knn = estimate["knn"]
        if not isinstance(knn, dict) or knn["advisory"] is None:
            return
        advisory = knn["advisory"]
        gib = float(2**30)
        scenarios = knn["scenarios"]
        parts = []
        for scenario in scenarios:
            if scenario["backend"] == "cpu":
                parts.append(
                    f"CPU host+runtime {scenario['host_peak_bytes'] / gib:.2f} GiB "
                    "(overlapping host and runtime buffers)"
                )
            else:
                parts.append(
                    f"CUDA host {scenario['host_peak_bytes'] / gib:.2f} GiB and device "
                    f"{scenario['device_peak_bytes'] / gib:.2f} GiB (separate peaks)"
                )
        if len(scenarios) > 1:
            backend_note = (
                "Backend 'auto' is unresolved, so the CPU and CUDA figures are conditional "
                "alternatives: only the backend actually selected runs, and they are not summed."
            )
        else:
            backend_note = f"Backend {scenarios[0]['backend']!r} was requested."
        min_samples = knn["min_samples"]
        if advisory["width_floor_applies"]:
            advice = (
                f"knn_k is below min_samples={min_samples}, so that floor is in effect: the "
                f"effective width stays at {min_samples} and lowering knn_k further will not "
                "reduce it. Lowering min_samples changes clustering semantics and is not advised."
            )
        else:
            advice = (
                f"Consider lowering knn_k; the effective width cannot be below "
                f"min_samples={min_samples}."
            )
        # Frames: warnings.warn -> this method -> _fit -> fit/fit_predict -> caller.
        warnings.warn(
            f"HDBSCAN kNN graph: {knn['finite_rows']} finite rows, effective width "
            f"{knn['effective_width']} (min_samples={min_samples}), {knn['directed_edges']} "
            f"directed edges, density {knn['density']:.3g}. Estimated peak allocations "
            f"(estimates, not total process usage): {'; '.join(parts)}. {backend_note} "
            f"This may use substantial memory and take longer. {advice} "
            "Execution continues with your settings.",
            UserWarning,
            stacklevel=4,
        )

    def fit_predict(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        sample_weight: ArrayLike | float | None = None,
        **fit_params: object,
    ) -> NDArray[np.intp]:
        return self._fit(X, y, sample_weight, fit_params).labels_.copy()

    def dbscan_clustering(self, eps: float) -> NDArray[np.intp]:
        if not hasattr(self, "_fitted_state"):
            raise ValueError("This HDBSCAN instance is not fitted yet")
        if np.isnan(eps):
            raise ValueError("eps must not be NaN")
        return np.array(self._native.dbscan_clustering(float(eps)), dtype=np.intp, copy=True)

    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: Literal[False] = ...
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]: ...
    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: Literal[True]
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]: ...
    @overload
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: bool | np.bool_ = ...
    ) -> (
        tuple[NDArray[np.intp], NDArray[np.float64]]
        | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
    ): ...
    def approximate_predict(
        self, X: ArrayLike, return_connecting_points: bool | np.bool_ = False
    ) -> (
        tuple[NDArray[np.intp], NDArray[np.float64]]
        | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
    ):
        """Predict fitted cluster labels for new points (original hdbscan ``approximate_predict``).

        Clusters are fixed: labels are existing fitted label IDs (``-1`` is noise) and are never
        renumbered. Uses the options and backend chosen at fit time; constructor attributes
        changed after fitting have no effect. ``X`` must be a finite 2-D numeric array with
        ``n_features_in_`` columns whose values fit in float32; rows are never dropped.

        Returns ``(labels, strengths)`` or, with ``return_connecting_points=True``,
        ``(labels, strengths, connecting_points)`` as new arrays in query order. Labels and
        connectors are ``np.intp``; strengths are ``np.float64`` and need not equal
        ``probabilities_`` even for training coordinates. A connector is the selected
        mutual-reachability neighbor (not necessarily the Euclidean nearest point) as an
        original training-row index; noise keeps its valid connector. Only when the fitted
        cluster tree has no clusters are all outputs noise/``0``/``-1``, with one
        ``UserWarning``.
        """
        return _approximate_predict(
            self._native,
            self._fitted_state["n_features_in"],
            X,
            return_connecting_points,
        )
