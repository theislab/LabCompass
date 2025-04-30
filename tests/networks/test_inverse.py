from collections.abc import Sequence
from typing import Any, Literal

import pytest
import torch

import sc_exp_design


optimal_condition = ...
perturbation_covariates = ...
perturbation_covariates_dims = ...
loss_fn = ...

class TestInverseNetworks:

    @pytest.mark.parametrize("is_discrete_dict", [
        {

        },
        {
        }
    ])
    @pytest.mark.parametrize("hard", [True, False])
    @pytest.mark.parametrize("use_prior", [True, False])
    def test_map(
        self,
        target_prediction_model: object,
        forward_model: object,
        prior: object,
        is_discrete_dict: dict[str, bool],
        hard: bool,
        use_prior: bool,
    ) -> None:
        """Test the map method of the inverse networks."""

        if not use_prior:
            prior = None

        # initialize the inverse networks
        map_inverse = sc_exp_design.networks.inverse.MAPConditionOptimizer(
            optimal_condition=optimal_condition,
            target_prediction_model=target_prediction_model,
            forward_model=forward_model,
            loss_fn=loss_fn,
            perturbation_covariates=perturbation_covariates,
            perturbation_covariates_dims=perturbation_covariates_dims,
            is_discrete_dict=is_discrete_dict,
            prior=prior,
            hard=hard,
        )
