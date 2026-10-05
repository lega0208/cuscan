from typing import Literal

import numpy as np
from numpy.typing import NDArray

__version__: str

class FittedModel:
    @property
    def device(self) -> Literal["cuda", "wgpu", "cpu"]: ...
    @property
    def n_features(self) -> int: ...
    @property
    def embedding(self) -> NDArray[np.float32]: ...
    def transform(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

def fit(
    x: NDArray[np.float32],
    *,
    device: Literal["cuda", "wgpu", "cpu"] = "cuda",
    n_components: int = 2,
    n_neighbors: int = 15,
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
) -> FittedModel: ...
