from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from sc_exp_design.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_mmd,
    compute_wasserstein_distance,
    compute_min_max_mse,
    compute_cell_props
)
from sc_exp_design.types import TensorLike

__all__ = [
    "Metrics",
]


@dataclass(frozen=True)
class Metrics:
    r_squared: Callable[[TensorLike, TensorLike], float] = compute_r_squared
    energy_distance: Callable[[TensorLike, TensorLike], float] = compute_e_distance
    maximum_mean_discrepancy: Callable[[TensorLike, TensorLike], float] = compute_mmd
    wasserstein_distance: Callable[[TensorLike, TensorLike], float] = compute_wasserstein_distance
    sinkhorn_divergence: Callable[[TensorLike, TensorLike], float] = partial(compute_wasserstein_distance, method="sinkhorn")
