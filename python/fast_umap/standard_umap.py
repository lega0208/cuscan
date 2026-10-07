from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _fast_umap


class UMAP:
    """Standard nonparametric UMAP estimator."""

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
        init: Literal["spectral", "random"] = "spectral",
        random_state: int | None = None,
        transform_seed: int = 42,
        transform_queue_size: float = 4.0,
        backend: Literal["flex", "cuda"] = "flex",
        device_index: int = 0,
        neighbor_search: Literal["auto", "exact", "approximate"] = "auto",
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
        if self.random_state is not None and (
            not isinstance(self.random_state, (int, np.integer)) or self.random_state < 0
        ):
            raise ValueError("random_state must be a non-negative integer or None")
        if self.n_epochs is not None and (
            not isinstance(self.n_epochs, (int, np.integer)) or self.n_epochs <= 0
        ):
            raise ValueError("n_epochs must be a positive integer or None")
        model = _fast_umap.fit_umap(
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
            random_state=None if self.random_state is None else int(self.random_state),
            transform_seed=self.transform_seed,
            transform_queue_size=self.transform_queue_size,
            backend=self.backend,
            device_index=self.device_index,
            neighbor_search=self.neighbor_search,
        )
        self.embedding_ = model.embedding
        self.n_features_in_ = x.shape[1]
        self._model = model
        return self

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float32]:
        return self.fit(X).embedding_

    def transform(self, X: ArrayLike) -> NDArray[np.float32]:
        if not hasattr(self, "_model"):
            raise ValueError("This UMAP instance is not fitted yet")
        x = self._input_array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {x.shape[1]}")
        return self._model.transform(x)
