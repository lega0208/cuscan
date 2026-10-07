from typing import Literal

import numpy as np
from numpy.typing import NDArray

__version__: str

class FittedParametricUMAP:
    @property
    def device(self) -> Literal["cuda", "wgpu", "cpu"]: ...
    @property
    def n_features(self) -> int: ...
    @property
    def embedding(self) -> NDArray[np.float32]: ...
    def transform(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

class FittedUMAP:
    @property
    def embedding(self) -> NDArray[np.float32]: ...
    def transform(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

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
    backend: Literal["flex", "cuda"] = "flex",
    device_index: int = 0,
    neighbor_search: Literal["auto", "exact", "approximate"] = "auto",
) -> FittedUMAP: ...
def fit_parametric_umap(
    x: NDArray[np.float32],
    *,
    device: Literal["cuda", "wgpu", "cpu"] = "cuda",
    n_components: int = 2,
    n_neighbors: int = 15,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    set_op_mix_ratio: float = 1.0,
    min_dist: float = 0.1,
    spread: float = 1.0,
    hidden_sizes: list[int] = [100],
    n_epochs: int = 100,
    learning_rate: float = 0.001,
    repulsion_strength: float = 1.0,
    negative_sample_rate: int = 5,
    patience: int | None = None,
    timeout: int | None = None,
    seed: int = 9999,
    verbose: bool = False,
) -> FittedParametricUMAP: ...
