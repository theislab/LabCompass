from collections.abc import Sequence
from typing import Literal

import numpy as np
import pytest
import torch

from labcompass.metrics import (
    compute_cell_props,
    compute_e_distance,
    compute_min_max_mse,
    compute_mmd,
    compute_r_squared,
    compute_wasserstein_distance,
)


class TestMetrics:
    @pytest.mark.parametrize("gammas", [None, (0.1, 1.0, 2.0, 5.0, 100.0)])
    @pytest.mark.parametrize("k", [1, 20, 100])
    @pytest.mark.parametrize("n_iter", [10, 20, 50, 100])
    @pytest.mark.parametrize("method", ["exact", "sinkhorn"])
    def test_compute_metrics(
        self,
        gammas: Sequence[float] | None,
        k: int,
        n_iter: int,
        method: Literal["exact", "sinkhorn"]
    ) -> None:

        batch_size = 128
        num_samples = 250
        dim = 10

        pred = torch.linspace(0.1, 1.0, dim).repeat(batch_size, 1)
        target = torch.linspace(0.1, 1.0, dim).repeat(batch_size, 1)*2 + 1

        r2 = compute_r_squared(pred, target)
        e_distance = compute_e_distance(pred, target)
        mmd = compute_mmd(pred, target, gammas=gammas)
        min_mse, max_mse = compute_min_max_mse(pred.unsqueeze(0).repeat(num_samples, 1, 1), target)
        cell_props = compute_cell_props(pred, target, k=k, n_iter=n_iter)
        wasserstein_distance = compute_wasserstein_distance(pred, target, method=method)

        assert -50 < r2 <= 1.0
        assert e_distance >= 0.0
        assert mmd >= 0.0
        assert np.all((0.0 <= cell_props) & (cell_props <= 1.0))
        assert np.all(min_mse >= 0.0)
        assert np.all(max_mse >= 0.0)
        assert wasserstein_distance >= 0.0
