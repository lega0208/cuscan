from __future__ import annotations

from os import PathLike
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _cuscan


def _output_dtype(X: ArrayLike) -> np.dtype:
    dtype = getattr(X, "dtype", None)
    if dtype is not None and np.issubdtype(dtype, np.floating):
        return np.dtype(dtype)
    return np.dtype(np.float32)


class UMAP:
    """Standard nonparametric UMAP estimator.

    Fit defaults to ``init="auto"`` and ``optimizer="adam"``. ``"auto"`` resolves
    to recursive initialization for every optimizer, and a default model's
    transform uses Adam. Pass ``optimizer="compatibility", init="spectral"`` for
    the previous defaults (classic SGD after spectral initialization).

    With ``init="spectral"``, fit raises ``RuntimeError`` when the k-NN graph has more than
    ``2 * n_components`` components and some component's centroid affinity to all others
    underflows to zero (umap-learn only warns there); use ``init="random"``, a larger
    ``n_neighbors``, or scaled features instead.
    For a connected graph, spectral initialization requires at least
    ``n_components + 2`` samples; use random initialization for smaller graphs.

    Computation and ``embedding_`` use float32. As in umap-learn 0.6.0,
    ``fit_transform`` and ``transform`` return the input's floating dtype, or
    float32 for non-floating inputs.

    ``optimizer="compatibility"`` selects classic SGD; ``"momentum"`` is opt-in.
    For N fitted references and range Q, modern layouts with H > 1024 source rows
    use projection windows of width min(N, max(200, Q)) when Q < N; H <= 1024
    samples globally with unit repulsion. None selects the automatic initial scale;
    transform uses fitted settings and keeps the reference embedding immutable.
    ``exclude_graph_neighbors`` (strict bool) and
    ``negative_sample_scale_adaptation_samples`` (int in ``[0, 2**31 - 1]``)
    are modern negative-sampling controls. Transform uses fitted settings.
    ``init_rescale`` centers each axis, then uses an isotropic robust span,
    per-axis robust spans, or the pruned graph's weighted-median edge length.
    ``init_scale_factor`` is a positive finite multiplier. Constant axes remain
    zero; an empty graph only centers in graph-edge mode. These fit-only controls
    never rescale transform results or loaded embeddings. Saved UMAP state uses
    schema v5; earlier UMAP files are rejected.

    ``memory_limit_bytes`` is an optional positive byte cap on the estimated ANN
    working set including the input, before the 64 MiB reserve. It is not a
    reservation or a budget for the entire fit; exact search is unaffected.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        local_connectivity: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        n_epochs: int | None = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        repulsion_strength: float = 1.0,
        init: Literal["spectral", "random", "recursive", "auto"] = "auto",
        random_state: int | None = None,
        transform_seed: int = 42,
        transform_queue_size: float = 4.0,
        backend: Literal["auto", "cpu", "cuda"] = "auto",
        device_index: int = 0,
        neighbor_search: Literal["auto", "exact", "approximate"] = "auto",
        unique: bool = False,
        memory_limit_bytes: int | None = None,
        init_rescale: Literal["isotropic", "per_axis", "graph_edge"] = "isotropic",
        init_scale_factor: float = 1.0,
        optimizer: Literal["compatibility", "adam", "momentum"] = "adam",
        negative_selection_range: int = 200_000,
        negative_sample_scale: float | None = None,
        exclude_graph_neighbors: bool = False,
        negative_sample_scale_adaptation_samples: int = 128,
        recursive_coarsening_ratio: int = 4,
        recursive_init_kwds: dict | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.local_connectivity = local_connectivity
        self.set_op_mix_ratio = set_op_mix_ratio
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.repulsion_strength = repulsion_strength
        self.init = init
        self.random_state = random_state
        self.transform_seed = transform_seed
        self.transform_queue_size = transform_queue_size
        self.backend = backend
        self.device_index = device_index
        self.neighbor_search = neighbor_search
        self.unique = unique
        self.memory_limit_bytes = memory_limit_bytes
        self.init_rescale = init_rescale
        self.init_scale_factor = init_scale_factor
        self.optimizer = optimizer
        self.negative_selection_range = negative_selection_range
        self.negative_sample_scale = negative_sample_scale
        self.exclude_graph_neighbors = exclude_graph_neighbors
        self.negative_sample_scale_adaptation_samples = negative_sample_scale_adaptation_samples
        self.recursive_coarsening_ratio = recursive_coarsening_ratio
        self.recursive_init_kwds = recursive_init_kwds

    @staticmethod
    def _input_array(X: ArrayLike) -> NDArray[np.float32]:
        try:
            array = np.asarray(X, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise ValueError("X must be a finite numeric 2-D array") from error
        if array.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if not np.isfinite(array).all():
            raise ValueError("X must contain only finite values")
        return np.ascontiguousarray(array)

    def fit(self, X: ArrayLike) -> UMAP:
        x = self._input_array(X)
        if not isinstance(self.init, str) or self.init not in (
            "spectral",
            "random",
            "recursive",
            "auto",
        ):
            raise ValueError("init must be 'spectral', 'random', 'recursive', or 'auto'")
        ratio = self.recursive_coarsening_ratio
        if (
            isinstance(ratio, (bool, np.bool_))
            or not isinstance(ratio, (int, np.integer))
            or int(ratio) not in (2, 3, 4)
        ):
            raise ValueError("recursive_coarsening_ratio must be 2, 3, or 4")
        supported = (
            "base_init_threshold",
            "n_embedding_epochs",
            "hub_selection",
            "anchor_sample_size",
        )
        kwds = self.recursive_init_kwds
        if kwds is not None and not isinstance(kwds, dict):
            raise ValueError("recursive_init_kwds must be a dict or None")
        kwds = {} if kwds is None else kwds
        if any(key not in supported for key in kwds):
            raise ValueError(f"recursive_init_kwds supports only: {', '.join(supported)}")
        recursive = {
            "base_init_threshold": kwds.get("base_init_threshold", 1024),
            "n_embedding_epochs": kwds.get("n_embedding_epochs", 50),
            "hub_selection": kwds.get("hub_selection", "degree"),
            "anchor_sample_size": kwds.get("anchor_sample_size", 16384),
        }
        for key, minimum, maximum in (
            ("base_init_threshold", 2, 2**31 - 1),
            ("n_embedding_epochs", 1, np.iinfo(np.uintp).max),
            ("anchor_sample_size", 1, 2**31 - 1),
        ):
            value = recursive[key]
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or not minimum <= int(value) <= maximum
            ):
                raise ValueError(f"{key} must be an integer in [{minimum}, {maximum}]")
            recursive[key] = int(value)
        if not isinstance(recursive["hub_selection"], str) or recursive["hub_selection"] not in (
            "degree",
            "diverse",
            "random",
        ):
            raise ValueError("hub_selection must be 'degree', 'diverse', or 'random'")
        if not isinstance(self.optimizer, str) or self.optimizer not in (
            "compatibility",
            "adam",
            "momentum",
        ):
            raise ValueError("optimizer must be 'compatibility', 'adam', or 'momentum'")
        if (
            isinstance(self.negative_selection_range, (bool, np.bool_))
            or not isinstance(self.negative_selection_range, (int, np.integer))
            or not 0 < int(self.negative_selection_range) <= 2**31 - 1
        ):
            raise ValueError("negative_selection_range must be a positive int32 integer")
        scale = self.negative_sample_scale
        if scale is not None:
            if isinstance(scale, (bool, np.bool_)) or not isinstance(
                scale, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    "negative_sample_scale must be a finite non-negative float or None"
                )
            try:
                scale_float = float(scale)
            except (OverflowError, ValueError):
                raise ValueError("negative_sample_scale must be representable as float32") from None
            if (
                not np.isfinite(scale_float)
                or scale_float < 0
                or scale_float > float(np.finfo(np.float32).max)
                or (scale_float != 0 and np.float32(scale_float) == 0)
            ):
                raise ValueError(
                    "negative_sample_scale must be a finite non-negative float32 value"
                )
        if not isinstance(self.exclude_graph_neighbors, (bool, np.bool_)):
            raise ValueError("exclude_graph_neighbors must be a boolean")
        adaptation_samples = self.negative_sample_scale_adaptation_samples
        if (
            isinstance(adaptation_samples, (bool, np.bool_))
            or not isinstance(adaptation_samples, (int, np.integer))
            or not 0 <= int(adaptation_samples) <= 2**31 - 1
        ):
            raise ValueError(
                "negative_sample_scale_adaptation_samples must be a non-negative int32 integer"
            )
        if self.init_rescale not in ("isotropic", "per_axis", "graph_edge"):
            raise ValueError("init_rescale must be 'isotropic', 'per_axis', or 'graph_edge'")
        if (
            isinstance(self.init_scale_factor, (bool, np.bool_))
            or not isinstance(self.init_scale_factor, (int, float, np.integer, np.floating))
            or not np.isfinite(self.init_scale_factor)
            or self.init_scale_factor <= 0.0
        ):
            raise ValueError("init_scale_factor must be finite and greater than zero")
        if self.random_state is not None and (
            not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None")
        if self.n_epochs is not None and (
            not isinstance(self.n_epochs, (int, np.integer)) or self.n_epochs <= 0
        ):
            raise ValueError("n_epochs must be a positive integer or None")
        if self.memory_limit_bytes is not None and (
            isinstance(self.memory_limit_bytes, (bool, np.bool_))
            or not isinstance(self.memory_limit_bytes, (int, np.integer))
            or not 0 < int(self.memory_limit_bytes) <= 2**64 - 1
        ):
            raise ValueError("memory_limit_bytes must be a positive uint64 integer or None")
        model = _cuscan.fit_umap(
            x,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            spread=self.spread,
            local_connectivity=self.local_connectivity,
            set_op_mix_ratio=self.set_op_mix_ratio,
            n_epochs=self.n_epochs,
            learning_rate=self.learning_rate,
            negative_sample_rate=self.negative_sample_rate,
            repulsion_strength=self.repulsion_strength,
            init=self.init,
            recursive_coarsening_ratio=int(ratio),
            recursive_base_init_threshold=recursive["base_init_threshold"],
            recursive_n_embedding_epochs=recursive["n_embedding_epochs"],
            recursive_hub_selection=recursive["hub_selection"],
            recursive_anchor_sample_size=recursive["anchor_sample_size"],
            init_rescale=self.init_rescale,
            init_scale_factor=self.init_scale_factor,
            optimizer=self.optimizer,
            negative_selection_range=int(self.negative_selection_range),
            negative_sample_scale=None if scale is None else scale_float,
            exclude_graph_neighbors=bool(self.exclude_graph_neighbors),
            negative_sample_scale_adaptation_samples=int(adaptation_samples),
            random_state=None if self.random_state is None else int(self.random_state),
            transform_seed=self.transform_seed,
            transform_queue_size=self.transform_queue_size,
            backend=self.backend,
            device_index=self.device_index,
            neighbor_search=self.neighbor_search,
            unique=self.unique,
            memory_limit_bytes=(
                None if self.memory_limit_bytes is None else int(self.memory_limit_bytes)
            ),
        )
        snapshot = {
            name: getattr(self, name)
            for name in (
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
                "recursive_coarsening_ratio",
                "init_rescale",
                "init_scale_factor",
                "optimizer",
                "negative_selection_range",
                "negative_sample_scale",
                "exclude_graph_neighbors",
                "negative_sample_scale_adaptation_samples",
                "random_state",
                "transform_seed",
                "transform_queue_size",
                "neighbor_search",
                "unique",
                "memory_limit_bytes",
            )
        }
        snapshot["recursive_coarsening_ratio"] = int(ratio)
        snapshot["recursive_init_kwds"] = (
            None if self.recursive_init_kwds is None else {key: recursive[key] for key in kwds}
        )
        self.embedding_ = model.embedding
        self.n_features_in_ = x.shape[1]
        self._model = model
        self._fit_config = snapshot
        return self

    def save(self, path: str | PathLike[str], overwrite: bool = False) -> None:
        if not hasattr(self, "_model"):
            raise ValueError("This UMAP instance is not fitted yet")
        from ._serialization_v5 import save_model

        arrays, metadata = self._model.export_state()
        save_model(path, arrays, metadata, self._fit_config, overwrite=overwrite)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        backend: Literal["auto", "cpu", "cuda"] = "auto",
        device_index: int = 0,
    ) -> UMAP:
        from ._serialization_v5 import load_model

        return load_model(cls, path, backend=backend, device_index=device_index)

    def fit_transform(self, X: ArrayLike) -> NDArray[np.floating]:
        dtype = _output_dtype(X)
        return self.fit(X).embedding_.astype(dtype, copy=False)

    def transform(self, X: ArrayLike) -> NDArray[np.floating]:
        if not hasattr(self, "_model"):
            raise ValueError("This UMAP instance is not fitted yet")
        dtype = _output_dtype(X)
        x = self._input_array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[1]}")
        return self._model.transform(x).astype(dtype, copy=False)
