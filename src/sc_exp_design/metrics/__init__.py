from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from sc_exp_design.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_mmd,
    compute_wasserstein_distance,
    compute_weighted_min_mse,
    compute_weighted_max_mse,
    compute_cell_props,
    compute_marginal_e_distance,
    compute_n_gen_knn_cell_props
)
from sc_exp_design.types import TensorLike

__all__ = [
    "Metrics",
]


@dataclass(frozen=True)
class Metrics:
<<<<<<< Updated upstream
    r_squared: Callable[[TensorLike, TensorLike], float] = compute_r_squared
    energy_distance: Callable[[TensorLike, TensorLike], float] = compute_e_distance
    maximum_mean_discrepancy: Callable[[TensorLike, TensorLike], float] = compute_mmd
    wasserstein_distance: Callable[[TensorLike, TensorLike], float] = compute_wasserstein_distance
    sinkhorn_divergence: Callable[[TensorLike, TensorLike], float] = partial(compute_wasserstein_distance, method="sinkhorn")
<<<<<<< HEAD
    min_max_mse: Callable[[TensorLike, TensorLike], float] = compute_min_max_mse
    cell_props: Callable[[TensorLike, TensorLike], float] = compute_cell_props
=======
=======
    weights: TensorLike
    r_squared: Callable[[TensorLike, TensorLike], float] = field(init=False)
    energy_distance: Callable[[TensorLike, TensorLike], float] = field(init=False)
    maximum_mean_discrepancy: Callable[[TensorLike, TensorLike], float] = field(init=False)
    wasserstein_distance: Callable[[TensorLike, TensorLike], float] = field(init=False)
    sinkhorn_divergence: Callable[[TensorLike, TensorLike], float] = field(init=False)

    average_min_mse: Callable[[TensorLike, TensorLike], float] = field(init=False)
    average_max_mse: Callable[[TensorLike, TensorLike], float] = field(init=False)

    average_weight_min_mse: Callable[[TensorLike, TensorLike], float] = field(init=False)
    average_weight_max_mse: Callable[[TensorLike, TensorLike], float] = field(init=False)

    neigh_composition: Callable[[TensorLike, TensorLike], float] = field(init=False)

    marginal_e_distance: Callable[[TensorLike, TensorLike], float] = field(init=False)
    marginal_weighted_e_distance: Callable[[TensorLike, TensorLike], float] = field(init=False)

    generational_homogeneity: Callable[[TensorLike, TensorLike], float] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'r_squared', compute_r_squared)
        object.__setattr__(self, 'energy_distance', compute_e_distance)
        object.__setattr__(self, 'maximum_mean_discrepancy', compute_mmd)
        object.__setattr__(self, 'wasserstein_distance', compute_wasserstein_distance)
        object.__setattr__(self, 'sinkhorn_divergence', partial(compute_wasserstein_distance, method="sinkhorn"))

        object.__setattr__(self, 'average_min_mse', compute_weighted_min_mse)
        object.__setattr__(self, 'average_max_mse', compute_weighted_max_mse)

        object.__setattr__(self, 'average_weight_min_mse',
                           partial(compute_weighted_min_mse, weights=self.weights))
        object.__setattr__(self, 'average_weight_max_mse',
                           partial(compute_weighted_max_mse, weights=self.weights))

        object.__setattr__(self, 'neigh_composition', compute_cell_props)

        object.__setattr__(self, 'marginal_e_distance', compute_marginal_e_distance)
        object.__setattr__(self, 'marginal_weighted_e_distance',
                           partial(compute_marginal_e_distance, weights=self.weights))

        object.__setattr__(self, 'generational_homogeneity', compute_n_gen_knn_cell_props)
>>>>>>> Stashed changes
>>>>>>> 77173f4 (added more metrics and adapted the callback)
