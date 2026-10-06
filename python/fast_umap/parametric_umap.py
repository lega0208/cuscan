from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _fast_umap


class NotFittedError(ValueError, AttributeError):
    """Raised when a transform is requested before fitting the estimator."""


class ParametricUMAP:
    """Parametric UMAP estimator.

    The learned parametric model makes ``transform`` a deterministic forward pass.
    Embeddings are float32; supervised UMAP and inverse transforms are unsupported,
    and learning-rate semantics differ from umap-learn. A fitted model is bound to
    the thread that created it. Inputs are used without automatic feature scaling;
    if preprocessing is needed, apply the same caller-owned transformation before
    both ``fit`` and ``transform``.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        *,
        n_components: int = 2,
        metric: str = "euclidean",
        n_epochs: int | None = None,
        learning_rate: float = 0.001,
        local_connectivity: float = 1.0,
        bandwidth: float = 1.0,
        set_op_mix_ratio: float = 1.0,
        min_dist: float = 0.1,
        spread: float = 1.0,
        repulsion_strength: float = 1.0,
        negative_sample_rate: int = 5,
        random_state: int | np.random.RandomState | np.random.Generator | None = None,
        verbose: bool = False,
        device: Literal["cuda", "wgpu", "cpu"] = "cuda",
        hidden_sizes: tuple[int, ...] = (100, 100, 100),
        patience: int | None = None,
        timeout: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.local_connectivity = local_connectivity
        self.bandwidth = bandwidth
        self.set_op_mix_ratio = set_op_mix_ratio
        self.min_dist = min_dist
        self.spread = spread
        self.repulsion_strength = repulsion_strength
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state
        self.verbose = verbose
        self.device = device
        self.hidden_sizes = hidden_sizes
        self.patience = patience
        self.timeout = timeout

    def _validate_params(self) -> int:
        if self.n_neighbors < 2:
            raise ValueError("n_neighbors must be at least 2 (including self)")
        if self.n_epochs is not None and self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        for name, value in (
            ("local_connectivity", self.local_connectivity),
            ("bandwidth", self.bandwidth),
            ("set_op_mix_ratio", self.set_op_mix_ratio),
        ):
            if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(
                value
            ):
                raise ValueError(f"{name} must be finite")
        if self.local_connectivity < 0:
            raise ValueError("local_connectivity must be non-negative")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if not 0 <= self.set_op_mix_ratio <= 1:
            raise ValueError("set_op_mix_ratio must be between 0 and 1")
        if self.metric != "euclidean":
            raise ValueError("only metric='euclidean' is supported")
        if self.device not in ("cuda", "wgpu", "cpu"):
            raise ValueError("device must be 'cuda', 'wgpu', or 'cpu'")
        state = self.random_state
        if state is None:
            return 9999
        if isinstance(state, (int, np.integer)):
            seed = int(state)
            if seed < 0:
                raise ValueError("random_state must be non-negative")
            return seed
        if isinstance(state, np.random.RandomState):
            return int(state.randint(0, np.iinfo(np.int32).max))
        if isinstance(state, np.random.Generator):
            return int(state.integers(0, np.iinfo(np.int32).max))
        raise ValueError("random_state must be None, an integer, RandomState, or Generator")

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
        return np.ascontiguousarray(array, dtype=np.float32)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> ParametricUMAP:
        if y is not None:
            raise NotImplementedError("supervised UMAP is not supported")
        x = self._input_array(X)
        seed = self._validate_params()
        if x.shape[0] <= self.n_neighbors:
            raise ValueError("n_samples must be greater than n_neighbors")
        model = _fast_umap.fit_parametric_umap(
            x,
            device=self.device,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            local_connectivity=self.local_connectivity,
            bandwidth=self.bandwidth,
            set_op_mix_ratio=self.set_op_mix_ratio,
            min_dist=self.min_dist,
            spread=self.spread,
            hidden_sizes=list(self.hidden_sizes),
            n_epochs=100 if self.n_epochs is None else self.n_epochs,
            learning_rate=self.learning_rate,
            repulsion_strength=self.repulsion_strength,
            negative_sample_rate=self.negative_sample_rate,
            patience=self.patience,
            timeout=self.timeout,
            seed=seed,
            verbose=self.verbose,
        )
        self.embedding_ = model.embedding
        self.n_features_in_ = x.shape[1]
        self._model = model
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray[np.float32]:
        return self.fit(X, y).embedding_

    def transform(self, X: ArrayLike) -> NDArray[np.float32]:
        if not hasattr(self, "_model"):
            raise NotFittedError("This ParametricUMAP instance is not fitted yet")
        x = self._input_array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[1]}")
        return self._model.transform(x)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {name: getattr(self, name) for name in self._parameter_names()}

    def set_params(self, **params: Any) -> ParametricUMAP:
        unknown = params.keys() - self._parameter_names()
        if unknown:
            raise ValueError(f"invalid parameter(s): {', '.join(sorted(unknown))}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    @staticmethod
    def _parameter_names() -> tuple[str, ...]:
        return (
            "n_neighbors",
            "n_components",
            "metric",
            "n_epochs",
            "learning_rate",
            "local_connectivity",
            "bandwidth",
            "set_op_mix_ratio",
            "min_dist",
            "spread",
            "repulsion_strength",
            "negative_sample_rate",
            "random_state",
            "verbose",
            "device",
            "hidden_sizes",
            "patience",
            "timeout",
        )

    def __repr__(self) -> str:
        defaults = type(self)()
        params = self.get_params()
        non_defaults = [
            f"{name}={value!r}"
            for name, value in params.items()
            if value != getattr(defaults, name)
        ]
        return f"ParametricUMAP({', '.join(non_defaults)})"
