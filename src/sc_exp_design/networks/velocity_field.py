import itertools
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from sc_exp_design.constants import VFStepFields
from sc_exp_design.networks.blocks import BaseModule, ConditionEncoder, MLPBlock
from sc_exp_design.networks.config import NeuralVelocityFieldConfig
from sc_exp_design.networks.neural_noise_models import MLPGaussianNoiseModel, MLPNegBinNoiseModel
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior, EndpointsApproximatePosterior

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
        if self.config.learn_posterior_on_perts:
            self.pert_approximate_posterior = self.pert_approximate_posterior.to(device)
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
        if self.config.learn_posterior_on_perts:
            parameters.append(self.pert_approximate_posterior.parameters())
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
        if self.config.learn_posterior_on_perts:
            self.pert_approximate_posterior = self.pert_approximate_posterior.train(mode)
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
        # decoder
        self.decoder = MLPBlock(
            self.config.joint_latent_dim,
            self.config.flow_dim,
            **self.config.decoder_mlp_kwargs
        )
        # score
        self.score_decoder = None
        if self.config.learn_score_field:
            self.score_decoder = MLPBlock(
                self.config.joint_latent_dim,
                self.config.flow_dim,
                **self.config.score_mlp_kwargs,
            )
        # inference on conditioning vars 
        self.endpoints_approximate_posterior = None
        if self.config.learn_posterior_on_cond_vars:
            self.endpoints_approximate_posterior = EndpointsApproximatePosterior(
                self.config.cond_vars_input_dim,
                self.config.flow_dim,
                freeze_grads=self.config.endpoints_approximate_posterior_freeze_grads,
                src_noise_model=self.config.src_noise_model,
                src_approximate_posterior_kwargs=self.config.src_approximate_posterior_kwargs,
                tgt_noise_model=self.config.tgt_noise_model,
                tgt_approximate_posterior_kwargs=self.config.tgt_approximate_posterior_kwargs,
            )
        # inference on perturbations
        self.pert_approximate_posterior = None
        if self.config.learn_posterior_on_perts:
            self.pert_approximate_posterior = PerturbationApproximatePosterior(
                self.config.pert_input_dim,
                freeze_grads=self.config.pert_approximate_posterior_freeze_grads,
                target_output_dims=self.config.pert_target_covariates_output_dims,
                noise_models=self.config.pert_noise_model,
                covariate_kwargs=self.config.pert_approximate_posterior_kwargs,
            )
        # inference on latent perturbations
        self.latent_pert_approximate_posterior = None
        if self.config.learn_posterior_on_latent_perts:
            self.latent_pert_approximate_posterior = MLPGaussianNoiseModel(
                2 * self.config.flow_dim,
                self.condition_encoder.latent_dim,
                **self.config.latent_perts_approximate_posterior_kwargs,
            )

    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
        target: Tensor | None = None,
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

        # forward pass on neural velocity field
        vf = self.decoder(latent_concat)
        # creating output dictionary
        output_dict = {VFStepFields.VF: vf, VFStepFields.LATENT_REPR: latent_concat, VFStepFields.LATENT_STATE: xt_latent}

        # preparing the endpoints for inference on perturbation
        endpoints = None
        if self.config.learn_posterior_on_perts and (self.config.pert_approximate_posterior_input_type == "endpoints"):
            if (source is not None) and (target is not None):
                endpoints = torch.concatenate((source, target), dim=1)

        # one step prediction for inference on perturbation
        one_step_prediction = None
        if self.config.learn_posterior_on_perts and (self.config.pert_approximate_posterior_input_type == "one_step_prediction"):
            if source is not None:
                one_step_prediction = xt + (1 - t)*vf 
                one_step_prediction = torch.concatenate((source, one_step_prediction), dim=1)

        # forward pass on neural score field
        if self.config.learn_score_field:
            score_decoder_input = latent_concat.clone()
            if self.config.score_field_freeze_grads:
                score_decoder_input = score_decoder_input.detach()
            score = self.score_decoder(latent_concat)
            output_dict[VFStepFields.SCORE] = score
        if self.config.encode_time:
            output_dict[VFStepFields.LATENT_TIME] = t_latent
        if self.config.use_guidance and self.config.encode_conditions:
            output_dict[VFStepFields.LATENT_PERTURBATION] = condition_latent

        # optional inference on conditioning variables
        if self.config.learn_posterior_on_cond_vars:
            if self.config.endpoints_approximate_posterior_use_latent_repr:
                input_condition_var_posterior = latent_concat
            else:
                input_condition_var_posterior = original_concat
            cond_vars_output_dict = self.endpoints_approximate_posterior(input_condition_var_posterior)
            output_dict.update(cond_vars_output_dict)

        # optional inference on perturbations
        if self.config.learn_posterior_on_perts:
            pert_output_dict = {}
            # with endpoints
            if (endpoints is not None) and (self.config.pert_approximate_posterior_input_type == "endpoints"):
                pert_output_dict[VFStepFields.PERTURBATION_PARAMS] = self.pert_approximate_posterior(endpoints)
            # with original representation
            if (original_concat is not None) and (self.config.pert_approximate_posterior_input_type == "original"):
                pert_output_dict[VFStepFields.PERTURBATION_PARAMS] = self.pert_approximate_posterior(original_concat)
            # with latent representation
            if (latent_concat is not None) and (self.config.pert_approximate_posterior_input_type == "latent"):
                pert_output_dict[VFStepFields.PERTURBATION_PARAMS] = self.pert_approximate_posterior(latent_concat)
            # with one step prediction
            if (one_step_prediction is not None) and (self.config.pert_approximate_posterior_input_type == "one_step_prediction"):
                pert_output_dict[VFStepFields.PERTURBATION_PARAMS] = self.pert_approximate_posterior(one_step_prediction)
            output_dict.update(pert_output_dict)

        return output_dict

    def vf(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
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
        return self.forward(t, xt, cond=cond)[VFStepFields.VF]

    def score(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
    ) -> Tensor:
        """
        Computes the score given time and state.
        
        Args:
            t (Tensor): Time input.
            xt (Tensor): State input.
            cond (dict[str, Tensor] | None, optional): Conditioning variables. Defaults to None.
        
        Returns:
            Tensor: Score output.
        """
        msg = f"{self.config.learn_score_field=}, hence no score field was initialized"
        assert self.config.learn_score_field, msg
        return self.forward(t, xt, cond=cond)[VFStepFields.SCORE]

    def get_vf_fn(
        self,
        cond: dict[str, Tensor] | None = None,
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
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
        if gamma_fn is not None and (not self.config.learn_score_field):
            msg = f"You passed `gamma_fn` for computing the diffusion coefficient with {self.config.learn_score_field=}. Deterministic sampling is set, hence it will be ignored."
            logger.warning(msg)
        if self.config.learn_score_field and gamma_fn is None:
            msg = f"With {self.config.learn_score_field=} you should pass a `gamma_fn` to compute the diffusion coefficient, found `None`. Falling back to deterministic sampling by default."
            logger.warning(msg)

        def vf_fn(
            t: Tensor,
            xt: Tensor,
        ) -> Tensor:
            """"""
            vf = self.vf(t, xt, cond=cond)
            if self.config.learn_score_field and gamma_fn is None:
                score = self.score(t, xt, cond=cond)
                gamma = gamma_fn(t, xt)
                return vf + 0.5 * (gamma**2) * score
            return vf

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

    def get_latent_condition_inf_params(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes latent condition inference parameters.
        
        Args:
            source (Tensor): Source tensor.
            target (Tensor): Target tensor.
        
        Returns:
            Tensor: Inference parameters.
        """
        input_perts_recognition_model = torch.concatenate((source, target), dim=1)
        params = self.latent_pert_approximate_posterior(input_perts_recognition_model)
        return params

    def get_perts_inf_params(
        self,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes perturbation inference parameters.
        
        Args:
            source (Tensor): Source tensor.
            target (Tensor): Target tensor.
        
        Returns:
            Tensor: Perturbation inference parameters.
        """
        msg = f"This method is only available when `self.config.pert_approximate_posterior_input_type` is either `'enpoints'` or `'one_step_prediction'`, found {self.config.pert_approximate_posterior_input_type=}"
        assert self.config.pert_approximate_posterior_input_type in ["endpoints", "one_step_prediction"], msg
        endpoints = torch.concatenate((source, target), dim=1)
        pert_output_dict = self.pert_approximate_posterior(endpoints)
        return pert_output_dict
