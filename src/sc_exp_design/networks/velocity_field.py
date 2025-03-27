import itertools
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from sc_exp_design.constants import VFStepFields
from sc_exp_design.networks.blocks import BaseModule, ConditionEncoder, MLPBlock
from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig
from sc_exp_design.networks.neural_noise_models import MLPGaussianNoiseModel, MLPNegBinNoiseModel
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior, EndpointsApproximatePosterior
from sc_exp_design.utils import sinusoidal_time_features

logger = logging.getLogger(__name__)

__all__ = ["NeuralVelocityField"]


class NeuralVelocityField(BaseModule):
    """
    A neural velocity field module for modeling continuous-time dynamics with neural networks.
    
    This class implements a velocity field using a neural network architecture, allowing for encoding
    time, state, and conditions to predict state transitions.
    """

    def __init__(
        self,
        config: NeuralVelocityFieldConfig,
    ) -> None:
        """
        Initialize the NeuralVelocityField.
        
        Args:
            flow_dim (int): Dimensionality of the flow field.
            config (NeuralVelocityFieldConfig): Configuration settings for the model.
        """
        super().__init__()
        self.config = config

        # initializing modules
        self._init_modules()

    def to(
        self,
        device: torch.device,
    ) -> nn.Module:
        """
        Moves the model and its components to the specified device.
        
        Args:
            device (torch.device): The target device (CPU/GPU).
        
        Returns:
            nn.Module: The model moved to the specified device.
        """
        self = super().to(device)
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.to(device)
        return self

    def parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """
        Returns an iterator over the model's parameters, including submodules.
        
        Returns:
            Iterator[nn.Parameter]: Model parameters.
        """
        parameters = [super().parameters()]
        if self.condition_encoder is not None:
            parameters.append(self.condition_encoder.parameters())
        parameters = itertools.chain(*parameters)
        return parameters

    def train(
        self,
        mode: bool = True
    ) -> nn.Module:
        """
        Sets the model to training mode.
        
        Args:
            mode (bool, optional): Whether to enable training mode. Defaults to True.
        
        Returns:
            nn.Module: The model in training mode.
        """
        self = super().train(mode)
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.train(mode)
        return self

    def eval(
        self,
    ) -> nn.Module:
        """
        Sets the model to evaluation mode.
        
        Returns:
            nn.Module: The model in evaluation mode.
        """
        self = super().eval()
        if self.config.learn_posterior_on_perts:
            self.pert_approximate_posterior = self.pert_approximate_posterior.eval()
        if self.condition_encoder is not None:
            self.condition_encoder = self.condition_encoder.eval()
        return self

    def _init_modules(
        self,
    ) -> None:
        """
        Initializes all necessary neural network modules including encoders, decoders, and inference models.
        """
        # state encoder
        self.x_encoder = None
        if self.config.encode_state:
            self.x_encoder = MLPBlock(
                self.config.flow_dim,
                self.config.state_encoder_output_dim,
                **self.config.state_encoder_mlp_kwargs,
            )
        # time encoder
        self.time_encoder = None
        if self.config.encode_time:
            self.time_encoder = MLPBlock(
                self.config.time_encoder_input_dim,
                self.config.time_encoder_output_dim,
                **self.config.time_encoder_mlp_kwargs,
            )
        # condition encoder
        self.condition_encoder = None
        if self.config.use_guidance and self.config.encode_conditions:
            self.condition_encoder = ConditionEncoder(
                latent_dim=self.config.perturbation_latent_dim,
                layers_before_pooling=self.config.perturbation_layers_before_pooling,
                covariates_not_pooled=self.config.perturbation_covariates_not_pooled,
                pooling=self.config.perturbation_pooling,
                pooling_kwargs=self.config.perturbation_pooling_kwargs,
                layers_after_pooling=self.config.perturbation_layers_after_pooling,
            )
        # optional source encoder
        if self.config.initialize_source_encoder:
            self.source_encoder = MLPBlock(
                self.config.flow_dim,
                self.config.source_latent_dim,
                **self.config.source_mlp_kwargs,
            )
        # decoder
        self.decoder = MLPBlock(
            self.config.joint_latent_dim,
            self.config.flow_dim,
            **self.config.decoder_mlp_kwargs
        )

    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """
        Forward pass through the neural velocity field model.
        
        Args:
            t (Tensor): Time input.
            xt (Tensor): State input.
            cond (dict[str, Tensor] | None, optional): Conditioning variables. Defaults to None.
            source (Tensor | None, optional): Source state for perturbation inference. Defaults to None.
            target (Tensor | None, optional): Target state for perturbation inference. Defaults to None.
        
        Returns:
            dict[str, Tensor]: Model output including velocity field and latent representations.
        """
        # encoding time
        t = torch.unsqueeze(t, dim=-1)
        t_latent = t
        if self.config.use_sinusoidal_time_features:
            t_latent = sinusoidal_time_features(
                t,
                num_freqs=self.config.time_features_num_freqs,
            )
        if self.config.encode_time:
            t_latent = self.time_encoder(t)
            
        # encoding conditions
        condition_latent = cond
        if self.config.use_guidance and self.config.encode_conditions:
            # sanity check (condition should be not None)
            msg = f""
            assert cond is not None, msg
            condition_latent = self.condition_encoder(cond)
            condition_original = torch.concatenate(list(cond.values()), dim=-1)
        elif self.config.use_guidance and (not self.config.encode_conditions):
            # sanity check (condition should be not None)
            msg = f""
            assert cond is not None, msg
            cond_values = [val for key, val in cond.items() if key in self.config.perturbation_layers_before_pooling]
            condition_latent = torch.concatenate(cond_values, dim=-1)
            condition_original = condition_latent
        
        # encoding states
        xt_latent = xt
        if self.config.encode_state:
            xt_latent = self.x_encoder(xt)

        # concatenating original and latent representations
        if self.config.use_guidance:
            # sanity check (condition should be not None)
            msg = f""
            assert cond is not None, msg
            latent_concat = torch.cat([t_latent, xt_latent, condition_latent], dim=-1)
            original_concat = torch.cat([t, xt, condition_original], dim=-1)
        else:
            latent_concat = torch.cat([t_latent, xt_latent], dim=-1)
            original_concat = torch.cat([t, xt], dim=-1)

        # encoding source
        if self.config.use_source_as_condition:
            msg = f""
            assert source is not None, msg
            source_latent = source
            if self.config.encode_source:
                source_latent = self.source_encoder(source)
            # concatenating to the input for the decoder
            original_concat = torch.cat([original_concat, source], dim=-1)
            latent_concat = torch.cat([latent_concat, source_latent], dim=-1)

        # forward pass on neural velocity field
        vf = self.decoder(latent_concat)
        # creating output dictionary
        output_dict = {VFStepFields.VF: vf, VFStepFields.LATENT_REPR: latent_concat, VFStepFields.LATENT_STATE: xt_latent}

        return output_dict

    def vf(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
    ) -> Tensor:
        """
        Computes the velocity field given time and state.
        
        Args:
            t (Tensor): Time input.
            xt (Tensor): State input.
            cond (dict[str, Tensor] | None, optional): Conditioning variables. Defaults to None.
        
        Returns:
            Tensor: Velocity field output.
        """
        return self.forward(t, xt, cond=cond, source=source)[VFStepFields.VF]

    def get_vf_fn(
        self,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """
        Returns a velocity field function.

        Args:
            cond (dict[str, Tensor] | None, optional): Conditioning variables. Defaults to None.
            gamma_fn (Callable[[Tensor, Tensor], Tensor] | None, optional): Function for computing diffusion coefficient. Defaults to None.
        
        Returns:
            Callable[[Tensor, Tensor], Tensor]: Velocity field function.
        """
        # sanity checks
        if self.config.use_source_as_condition:
            msg = f""
            assert source is not None, msg

        def vf_fn(
            t: Tensor,
            xt: Tensor,
        ) -> Tensor:
            """"""
            return self.vf(t, xt, cond=cond, source=source)

        return vf_fn

    def get_condition_embedding(
        self,
        cond: dict[str, Tensor],
    ) -> Tensor:
        """
        Computes the condition embedding.
        
        Args:
            cond (dict[str, Tensor]): Conditioning variables.
        
        Returns:
            Tensor: Condition embedding tensor.
        """
        # sanity check
        msg = f"No condition encoder associated to this Velocity Field (i.e.: {self.config.encode_conditions=})."
        assert self.config.encode_conditions, msg
        msg = f"The velocity field is in the unguided mode (i.e.: {self.config.use_guidance=})"
        assert self.config.use_guidance, msg
        # forward pass on condition encoder
        condition_latent = self.condition_encoder(cond)
        return condition_latent
