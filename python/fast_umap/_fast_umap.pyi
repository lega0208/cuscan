from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

__version__: str

class FittedUMAP:
    @property
    def embedding(self) -> NDArray[np.float32]: ...
    def transform(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...
    def export_state(self) -> tuple[dict[str, NDArray[np.generic] | None], dict[str, object]]: ...

class FittedHDBSCAN:
    @property
    def labels(self) -> NDArray[np.intp]: ...
    @property
    def probabilities(self) -> NDArray[np.float32]: ...
    @property
    def condensed_tree(self) -> NDArray[np.void]: ...
    @property
    def single_linkage_tree(self) -> NDArray[np.float64]: ...
    @property
    def minimum_spanning_tree(self) -> NDArray[np.float64]: ...
    @property
    def neighbors(self) -> NDArray[np.intp]: ...
    @property
    def core_distances(self) -> NDArray[np.float32]: ...
    @property
    def finite_index(self) -> NDArray[np.intp]: ...
    def dbscan_clustering(self, eps: float) -> NDArray[np.intp]: ...
    # Labels/connectors are int64 (np.intp), strengths float64. Connectors are original
    # training-row indices of the selected mutual-reachability neighbor; -1 only on the
    # no-cluster-tree shortcut, which emits one UserWarning at `stacklevel`.
    @overload
    def approximate_predict(
        self,
        x: NDArray[np.float32],
        return_connecting_points: Literal[False] = ...,
        *,
        stacklevel: int = ...,
    ) -> tuple[NDArray[np.intp], NDArray[np.float64]]: ...
    @overload
    def approximate_predict(
        self,
        x: NDArray[np.float32],
        return_connecting_points: Literal[True],
        *,
        stacklevel: int = ...,
    ) -> tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]: ...
    @overload
    def approximate_predict(
        self,
        x: NDArray[np.float32],
        return_connecting_points: bool = ...,
        *,
        stacklevel: int = ...,
    ) -> (
        tuple[NDArray[np.intp], NDArray[np.float64]]
        | tuple[NDArray[np.intp], NDArray[np.float64], NDArray[np.intp]]
    ): ...

def fit_hdbscan(
    x: NDArray[np.float32],
    **kwargs: object,
) -> FittedHDBSCAN: ...
def estimate_hdbscan_resources(
    n_finite: int,
    features: int,
    **kwargs: object,
) -> dict[str, object]: ...
def hdbscan_cuda_available() -> bool: ...
def fit_umap(
    x: NDArray[np.float32],
    *,
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
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    device_index: int = 0,
    neighbor_search: Literal["auto", "exact", "approximate"] = "auto",
    unique: bool = False,
) -> FittedUMAP: ...
def load(
    arrays: dict[str, NDArray[np.generic] | None],
    metadata: dict[str, object],
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    device_index: int = 0,
) -> FittedUMAP: ...
