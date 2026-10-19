from typing import Literal

import numpy as np
from numpy.typing import NDArray

__version__: str

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
    backend: Literal["auto", "cpu", "cuda"] = "auto",
    device_index: int = 0,
    neighbor_search: Literal["auto", "exact", "approximate"] = "auto",
    unique: bool = False,
) -> FittedUMAP: ...
