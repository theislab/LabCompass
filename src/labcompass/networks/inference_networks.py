import itertools
import logging
from collections.abc import Callable, Iterator
from typing import Any, Literal

import torch
from torch import Tensor, nn

from labcompass.constants import VFStepFields
from labcompass.networks.blocks import BaseModule, MLPBlock
from labcompass.networks.neural_noise_models import MLPGaussianNoiseModel, MLPNegBinNoiseModel

__all__ = ["PerturbationApproximatePosterior"]


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
        """
        Returns the appropriate noise model based on the provided string identifier.

        Args:
            noise_model (str | None): The type of noise model to be used. Can be 
                                    'gaussian', 'neg_bin', or None.

        Returns:
            BaseModule: A module corresponding to the specified noise model. 
                        If no model is provided, defaults to MLPBlock.

        Raises:
            NotImplementedError: If an unsupported noise model is provided.

        Notes:
            - If `noise_model` is None, and `self._raise_error_if_none` is False, the method will return `MLPBlock`.
            - If `noise_model` is "gaussian", it returns the `MLPGaussianNoiseModel`.
            - If `noise_model` is "neg_bin", it returns the `MLPNegBinNoiseModel`.
        """
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
    """
    A class that constructs perturbation-based approximate posteriors for multiple target covariates. 
    It allows for the specification of noise models, covariate-specific configurations, and supports 
    various training and evaluation modes.

    Attributes:
        _raise_error_if_none (bool): Flag indicating whether to raise an error if no noise model is provided.
        input_dim (int): The input dimension of the model.
        freeze_grads (bool): Flag to freeze gradients during training.
        target_output_dims (dict): A dictionary mapping covariate IDs to their respective output dimensions.
        noise_models (dict): A dictionary mapping covariate IDs to noise model types, such as "gaussian" or "neg_bin".
        covariate_kwargs (dict): A dictionary containing additional keyword arguments for covariate-specific configurations.
        pert_approximate_posterior (dict): A dictionary storing the initialized perturbation approximate posterior modules for each covariate.

    Methods:
        __init__: Initializes the PerturbationApproximatePosterior object.
        _init_modules: Initializes the perturbation approximate posterior modules for each covariate.
        to: Moves the model and its components to the specified device.
        parameters: Returns an iterator over the model parameters.
        train: Puts the model into training mode, including its perturbation components.
        eval: Puts the model into evaluation mode, including its perturbation components.
        forward: Performs a forward pass through the model and returns perturbation parameters for each covariate.
    """

    _raise_error_if_none: bool = False
    def __init__(
        self,
        input_dim: int,
        freeze_grads: bool = True,
        target_output_dims: dict[str, int] | None = None,
        noise_models: dict[str, Literal["gaussian", "neg_bin"]] | None = None,
        covariate_kwargs: dict[str, dict[str, Any]] | None = None,
        use_shared_representation: bool = False,
        latent_dim: int = 1024,
        encoder_mlp_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the PerturbationApproximatePosterior object.

        Args:
            input_dim (int): The input dimension of the model.
            freeze_grads (bool, optional): Flag to freeze gradients. Defaults to True.
            target_output_dims (dict, optional): A dictionary of target output dimensions. Defaults to None.
            noise_models (dict, optional): A dictionary of noise models for each covariate. Defaults to None.
            covariate_kwargs (dict, optional): A dictionary of covariate-specific configurations. Defaults to None.
        """

        super().__init__()
        self.input_dim = input_dim
        self.freeze_grads = freeze_grads
        self.target_output_dims = target_output_dims
        self.noise_models = noise_models
        self.covariate_kwargs = covariate_kwargs
        self.use_shared_representation = use_shared_representation
        self.latent_dim = latent_dim
        self.encoder_mlp_kwargs = encoder_mlp_kwargs  

        self._init_modules()

    @property
    def decoder_input_dim(
        self,
    ) -> None:
        """
        Returns the dimensionality of the decoder input based on whether shared representation is used.

        :return: Dimensionality of the decoder input.
        :rtype: int
        """
        if self.use_shared_representation:
            return self.latent_dim
        return self.input_dim

    def _init_modules(
        self,
    ) -> None:
        """
        Initializes the perturbation approximate posterior modules for each covariate.

        This method iterates through the target output dimensions and initializes the corresponding perturbation
        approximate posterior for each covariate using the provided noise models and covariate-specific arguments.

        Populates the `pert_approximate_posterior` attribute with the initialized modules for each covariate.

        Raises:
            KeyError: If a covariate ID in `target_output_dims` does not have a corresponding noise model or configuration.
        """
        modules = {}
        if self.use_shared_representation:
            modules["encoder"] = MLPBlock(
                self.input_dim,
                self.latent_dim,
                **self.encoder_mlp_kwargs,
            )
        for covariate_id, output_dim in self.target_output_dims.items():
            # retrieving configuration for target covariates
            pert_covariate_noise_model = self.noise_models[covariate_id]
            pert_covariate_approximate_posterior_kwargs = self.covariate_kwargs[
                covariate_id
            ]
            # initializing the module
            pert_approximate_posterior_class = self._get_noise_model(pert_covariate_noise_model)
            cov_pert_approximate_posterior = pert_approximate_posterior_class(
                self.decoder_input_dim,
                output_dim,
                **pert_covariate_approximate_posterior_kwargs,
            )
            modules[covariate_id] = cov_pert_approximate_posterior
        self.pert_approximate_posterior = nn.ModuleDict(modules)

    def forward(
        self,
        input_pert_posterior: Tensor,
    ) -> dict[str, dict[str, Tensor]]:
        """
        Performs a forward pass through the model and returns perturbation parameters for each covariate.

        Args:
            input_pert_posterior (Tensor): The input tensor representing the perturbation posterior.

        Returns:
            dict[str, dict[str, Tensor]]: A dictionary containing the perturbation parameters for each covariate.
        """
        # cloning to preserve the gradients
        input_pert_posterior = input_pert_posterior.clone()
        # freezing the grads
        if self.freeze_grads:
            input_pert_posterior = input_pert_posterior.detach()
        
        # optional encoder
        if self.use_shared_representation:
            input_pert_posterior = self.pert_approximate_posterior["encoder"](input_pert_posterior)

        # decoder for each target covariate
        pert_posterior_params_dict = {}
        for cov_id, cov_decoder in self.pert_approximate_posterior.items():
            if cov_id == "encoder":
                continue
            # forward pass on nn and storing the results
            pert_posterior_params = cov_decoder(input_pert_posterior)
            pert_posterior_params_dict[cov_id] = pert_posterior_params
        return pert_posterior_params_dict
