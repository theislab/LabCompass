import itertools
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from sc_exp_design.constants import VFStepFields
from sc_exp_design.networks.blocks import BaseModule, ConditionEncoder, MLPBlock, ResnetBlock, FiLMBlock
from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig
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

    def _init_modules(
        self,
    ) -> None:
        """
        Initializes all necessary neural network modules including encoders, decoders, and inference models.
        """
        # state encoder
        modules = {} 
        if self.config.encode_state:
            modules["x_encoder"] = MLPBlock(
                self.config.flow_dim,
                self.config.state_encoder_output_dim,
                **self.config.state_encoder_mlp_kwargs,
            )
        else:
            modules["x_encoder"] = torch.nn.Identity()
        # time encoder
        if self.config.encode_time:
            modules["time_encoder"] = MLPBlock(
                self.config.time_encoder_input_dim,
                self.config.time_encoder_output_dim,
                **self.config.time_encoder_mlp_kwargs,
            )
        else:
            modules["time_encoder"] = torch.nn.Identity()
        # condition encoder
        if self.config.use_guidance and self.config.encode_conditions:
            modules["condition_encoder"] = ConditionEncoder(
                latent_dim=self.config.perturbation_latent_dim,
                layers_before_pooling=self.config.perturbation_layers_before_pooling,
                covariates_not_pooled=self.config.perturbation_covariates_not_pooled,
                pooling=self.config.perturbation_pooling,
                pooling_kwargs=self.config.perturbation_pooling_kwargs,
                layers_after_pooling=self.config.perturbation_layers_after_pooling,
            )
        # optional source encoder
        if self.config.initialize_source_encoder:
            modules["source_encoder"] = MLPBlock(
                self.config.flow_dim,
                self.config.source_latent_dim,
                **self.config.source_encoder_mlp_kwargs,
            )
        # ResNet
        self.resnet_blocks = None
        if self.config.conditioning_type == "resnet":            
            resnet_blocks = []
            for _ in range(self.config.n_resnet_blocks):
                resnet_blocks.append(
                    ResnetBlock(
                        self.config.state_encoder_output_dim, 
                        out_dim=None,  # dimensionality preserving 
                        dropout_prob=self.config.resnet_dropout_prob, 
                        embedding_dim=self.config.resnet_embedding_dim,
                        normalization=self.config.resnet_normalization
                    )
                ) 
            modules["resnet_blocks"] = nn.ModuleList(resnet_blocks)   
        #FiLM
        if self.config.conditioning_type == "film":
            modules["film_block"] = FiLMBlock(
                in_dim=(self.config.state_latent_dim + self.config.time_latent_dim),
                cond_dim=(self.config.perturbation_latent_dim + self.config.source_latent_dim))
        # Decoder
        modules["decoder"] = MLPBlock(
            self.config.decoder_input_dim,
            self.config.flow_dim,
            **self.config.decoder_mlp_kwargs
        )
        self.vf_modules = torch.nn.ModuleDict(modules)
        
    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
    ) -> Tensor:
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
                max_period=self.config.time_features_max_periods
            )
        if self.config.encode_time:
            t_latent = self.vf_modules["time_encoder"](t_latent)
            
        # encoding conditions
        condition_latent = cond
        if self.config.use_guidance and self.config.encode_conditions:
            # sanity check (condition should be not None)
            msg = f""
            assert cond is not None, msg
            condition_latent = self.vf_modules["condition_encoder"](cond)
            condition_latent = nn.functional.dropout(
                condition_latent,
                p=self.config.perturbation_output_dropout
            )
        elif self.config.use_guidance and (not self.config.encode_conditions):
            # sanity check (condition should be not None)
            msg = f""
            assert cond is not None, msg
            cond_values = [val for key, val in cond.items() if key in self.config.perturbation_layers_before_pooling]
            condition_latent = torch.concatenate(cond_values, dim=-1)


        
        # encoding states
        xt_latent = xt
        if self.config.encode_state:
            xt_latent = self.vf_modules["x_encoder"](xt)

        # concatenating original and latent representations
        if self.config.conditioning_type == "concatenation":
            if self.config.use_guidance:
                # sanity check (condition should be not None)
                msg = f""
                assert cond is not None, msg
                latent_concat = torch.cat([t_latent, xt_latent, condition_latent], dim=-1)
            else:
                latent_concat = torch.cat([t_latent, xt_latent], dim=-1)
        elif self.config.conditioning_type == "resnet":
            latent_concat = xt_latent
            if self.config.use_guidance:
                condition_concat = torch.cat([t_latent, condition_latent], dim=-1)  
            else:
                condition_concat = t_latent
        elif self.config.conditioning_type == "film":
            msg = f"FiLM is only possible with guidance"
            assert self.config.use_guidance, msg
            condition_concat = condition_latent 
            latent_concat = torch.cat([t_latent, xt_latent], dim=-1)
     
        # encoding source
        if self.config.use_source_as_condition:
            msg = f""
            assert source is not None, msg
            source_latent = source
            if self.config.encode_source:
                source_latent = self.vf_modules["source_encoder"](source)
            if self.config.conditioning_type == "concatenation":
                # concatenating to the input for the decoder
                latent_concat = torch.cat([latent_concat, source_latent], dim=-1)
            else:
                condition_concat = torch.cat([condition_concat, source_latent], dim=-1)  # concatenate
            
        # ResNet 
        if self.config.conditioning_type == "resnet":
            latent_initial_shape = latent_concat.shape
            condition_initial_shape = condition_concat.shape
            latent_concat = latent_concat.reshape(-1, latent_initial_shape[-1])
            condition_concat = condition_concat.reshape(-1, condition_initial_shape[-1])
            for block in self.vf_modules["resnet_blocks"]:
                latent_concat = block(latent_concat, condition_concat)
            latent_concat = latent_concat.reshape(*latent_initial_shape)
        # FiLM
        elif self.config.conditioning_type == "film":
            latent_concat = self.vf_modules["film_block"](latent_concat, condition_concat)

        # forward pass on neural velocity field
        return self.vf_modules["decoder"](latent_concat)

    def get_vf_fn(
        self,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
        cfg_guidance_strength: float = 1.0,
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
            # when using cfg
            if self.config.use_classifier_free_guidance:
                # get null condition token
                null_condition_token = self.get_null_condition_token(cond)
                # computing unguided and guided velocity fields
                vf_unguided = self.forward(t, xt, cond=null_condition_token, source=source)
                vf_guided = self.forward(t, xt, cond=cond, source=source)
                # computing the final velocity field
                vf = vf_unguided + cfg_guidance_strength * (vf_guided - vf_unguided)
                return vf
            # when not using cfg
            return self.forward(t, xt, cond=cond, source=source)

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
        condition_latent = self.vf_modules["condition_encoder"](cond)
        return condition_latent

    def get_null_condition_token(
        self,
        cond: dict[str, Tensor] | None,
    ) -> Tensor:
        """"""
        # when condition is None we simply return None
        if cond is None:
            return None
        # otherwise we need to replace each value 
        # of the dictionary with a null condition token
        cond_copy = {}
        for key, val in cond.items():
            cond_copy[key] = torch.ones_like(val)*self.config.null_condition_token
        return cond_copy
