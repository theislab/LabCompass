from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sc_exp_design.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_sinkhorn_div,
)

__all__ = [
    "Metrics",
]


@dataclass(frozen=True)
class Metrics:
    r_squared: Callable[[np.ndarray, np.ndarray], np.ndarray] = compute_r_squared
    sinkhorn_divergence: Callable[[np.ndarray, np.ndarray], np.ndarray] = compute_sinkhorn_div
    energy_distance: Callable[[np.ndarray, np.ndarray], np.ndarray] = compute_e_distance
