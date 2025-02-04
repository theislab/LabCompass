import itertools
import logging
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sc_exp_design.constants import (
    PERTURBATION_PARAMS_KEYS,
    SOURCE_PARAMS_KEY,
    TARGET_PARAMS_KEY,
)
from sc_exp_design.networks.blocks import BaseModule, ConditionEncoder, MLPBlock
from sc_exp_design.networks.neural_noise_models import MLPGaussianNoiseModel, MLPNegBinNoiseModel

__all__ = ["PerturbationApproximatePosterior", "EndpointsApproximatePosterior"]


class BaseApproximatePosterior(BaseModule):
    """"""
    _raise_error_if_none: bool
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:

        super().__init__()
    
    def _get_noise_model(
        self,
        noise_model: Literal["gaussian", "neg_bin"] | None,
    ) -> BaseModule:
        """"""
        if noise_model is None and (not self._raise_error_if_none):
            return MLPBlock
        elif noise_model == "gaussian":
            return MLPGaussianNoiseModel
        elif noise_model == "neg_bin":
            return MLPNegBinNoiseModel
        else:
            msg = (
                f"{noise_model=} not supported, possible values `['gaussian', 'neg_bin']`" if self.raise_error_if_none else
                f"{noise_model=} not supported, possible values `['gaussian', 'neg_bin', None]`" 
            )
            raise NotImplementedError(msg)


class PerturbationApproximatePosterior(BaseApproximatePosterior):
    """"""
    _raise_error_if_none: bool = False
    def __init__(
        self,
        input_dim: int,
        freeze_grads: bool = True,
        target_output_dims: dict[str, int] | None = None,
        noise_models: dict[str, Literal["gaussian", "neg_bin"]] | None = None,
        covariate_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.freeze_grads = freeze_grads
        self.target_output_dims = target_output_dims
        self.noise_models = noise_models
        self.covariate_kwargs = covariate_kwargs

        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """"""
        pert_approximate_posterior = {}
        for covariate_id, output_dim in self.target_output_dims.items():
            # retrieving configuration for target covariates
            pert_covariate_noise_model = self.noise_models[covariate_id]
            pert_covariate_approximate_posterior_kwargs = self.covariate_kwargs[
                covariate_id
            ]
            # initializing the module
            pert_approximate_posterior_class = self._get_noise_model(pert_covariate_noise_model)
            cov_pert_approximate_posterior = pert_approximate_posterior_class(
                self.input_dim,
                output_dim,
                **pert_covariate_approximate_posterior_kwargs,
            )
            pert_approximate_posterior[covariate_id] = cov_pert_approximate_posterior
        self.pert_approximate_posterior = pert_approximate_posterior

    def to(
        self,
        device: torch.device,
    ) -> nn.Module:
        """"""
        self = super().to(device)
        pert_approximate_posterior = {}
        for pert_target_covariate_id, cov_pert_approximate_posterior in self.pert_approximate_posterior.items():
            pert_approximate_posterior[pert_target_covariate_id] = cov_pert_approximate_posterior.to(device)
        self.pert_approximate_posterior = pert_approximate_posterior
        return self

    def parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """"""
        parameters = [super().parameters()]
        for cov_decoder in self.pert_approximate_posterior.values():
            parameters.append(cov_decoder.parameters())
        parameters = itertools.chain(*parameters)
        return parameters

    def train(
        self,
        mode: bool = True
    ) -> nn.Module:
        """"""
        self = super().train(mode)
        pert_approximate_posterior = {}
        for pert_target_covariate_id, cov_pert_approximate_posterior in self.pert_approximate_posterior.items():
            pert_approximate_posterior[pert_target_covariate_id] = cov_pert_approximate_posterior.train(mode)
        self.pert_approximate_posterior = pert_approximate_posterior
        return self

    def eval(
        self,
    ) -> nn.Module:
        """"""
        self = super().eval()
        pert_approximate_posterior = {}
        for pert_target_covariate_id, cov_pert_approximate_posterior in self.pert_approximate_posterior.items():
            pert_approximate_posterior[pert_target_covariate_id] = cov_pert_approximate_posterior.eval()
        self.pert_approximate_posterior = pert_approximate_posterior
        return self

    def forward(
        self,
        input_pert_posterior: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        """"""
        pert_output_dict = {}
        pert_posterior_params_dict = {}
        for cov_id, cov_decoder in self.pert_approximate_posterior.items():
            # cloning to preserve the gradients
            input_pert_posterior = input_pert_posterior.clone()
            # freezing the grads
            if self.freeze_grads:
                input_pert_posterior = input_pert_posterior.detach()
            # forward pass on nn and storing the results
            pert_posterior_params = cov_decoder(input_pert_posterior)
            pert_posterior_params_dict[cov_id] = pert_posterior_params
        pert_output_dict[PERTURBATION_PARAMS_KEYS] = pert_posterior_params_dict
        return pert_output_dict


class EndpointsApproximatePosterior(BaseApproximatePosterior):
    """"""
    _raise_error_if_none: bool = True
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        freeze_grads: bool = True,
        src_noise_model: Literal["gaussian", "neg_bin"] = "gaussian",
        src_approximate_posterior_kwargs: dict[str, Any] | None = None,
        tgt_noise_model: Literal["gaussian", "neg_bin"] = "gaussian",
        tgt_approximate_posterior_kwargs: dict[str, Any] | None = None
    ) -> None:

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freeze_grads = freeze_grads
        self.src_noise_model = src_noise_model
        self.src_approximate_posterior_kwargs = src_approximate_posterior_kwargs
        self.tgt_noise_model = tgt_noise_model
        self.tgt_approximate_posterior_kwargs = tgt_approximate_posterior_kwargs

        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """"""
        # source
        src_approximate_posterior_class = self._get_noise_model(self.src_noise_model)
        self.src_approximate_posterior = src_approximate_posterior_class(
            self.input_dim,
            self.output_dim,
            **self.src_approximate_posterior_kwargs,
        )
        # target
        tgt_approximate_posterior_class = self._get_noise_model(self.tgt_noise_model)
        self.tgt_approximate_posterior = tgt_approximate_posterior_class(
            self.input_dim,
            self.output_dim,
            **self.tgt_approximate_posterior_kwargs,
        )
    
    def forward(
        self,
        input_tensor: Tensor,
    ) -> dict[str, Tensor]:
        """"""
        cond_vars_output_dict = {}
        # cloning to preserve the gradients
        src_input_tensor = input_tensor.clone()
        tgt_input_tensor = input_tensor.clone()
        # freezing the gradients
        if self.freeze_grads:
            src_input_tensor = src_input_tensor.detach()
            tgt_input_tensor = tgt_input_tensor.detach()
        # forward pass on nn
        src_posterior_params = self.src_approximate_posterior(input_tensor)
        tgt_posterior_params = self.tgt_approximate_posterior(input_tensor)
        # storing the results
        cond_vars_output_dict[SOURCE_PARAMS_KEY] = src_posterior_params
        cond_vars_output_dict[TARGET_PARAMS_KEY] = tgt_posterior_params
        return cond_vars_output_dict
