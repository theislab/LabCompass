import itertools
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn

from labcompass.constants import VFStepFields
from labcompass.networks.blocks import BaseModule, ConditionEncoder, MLPBlock, ResnetBlock, FiLMBlock
from labcompass.config.velocity_field import NeuralVelocityFieldConfig
from labcompass.utils import sinusoidal_time_features

logger = logging.getLogger(__name__)

__all__ = ["NeuralVelocityField"]


class NeuralVelocityField(BaseModule):
    """A neural velocity field module for modeling continuous-time dynamics with neural networks.

    This class implements a velocity field using a neural network architecture, allowing for encoding
    time, state, and conditions to predict state transitions.

    :param config: Configuration object specifying the architecture of the velocity field module (which
        encoders to build for state, time, conditions and source, the conditioning strategy used to combine
        them, and the configuration of the final decoder), used to build the submodules collected in
        :attr:`NeuralVelocityField.vf_modules`.
    :type config: class:`NeuralVelocityFieldConfig`
    """

    def __init__(
        self,
        config: NeuralVelocityFieldConfig,
    ) -> None:
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
        """Computes the predicted velocity of the flow at state `xt` and time `t`.

        Encodes the time, state, perturbation conditions and (optionally) the source state according to the
        :class:`NeuralVelocityFieldConfig` passed at initialization, combines the resulting latent
        representations using the configured conditioning strategy (`"concatenation"`, `"resnet"` or
        `"film"`), and decodes them into the predicted velocity.

        :param t: Time steps at which the velocity is evaluated, of shape `(*batch_shape,)`.
        :type t: class:`Tensor`

        :param xt: State at time `t`, of shape `(*batch_shape, flow_dim)`.
        :type xt: class:`Tensor`

        :param cond: Dictionary mapping each perturbation covariate name to its corresponding conditioning
            tensor. Required when :attr:`NeuralVelocityFieldConfig.use_guidance` is `True`, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Source (e.g.: control) state used as an additional conditioning signal. Required when
            :attr:`NeuralVelocityFieldConfig.use_source_as_condition` is `True`, defaults to `None`.
        :type source: class:`Tensor | None`

        :return: The predicted velocity at `(t, xt)`, of shape `(*batch_shape, flow_dim)`.
        :rtype: class:`Tensor`
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
        """Builds a velocity field function of `(t, xt)` bound to fixed conditions and source state.

        The returned function evaluates :meth:`NeuralVelocityField.forward` at given `t` and `xt` values.
        When :attr:`NeuralVelocityFieldConfig.use_classifier_free_guidance` is `True`, it instead combines
        the unguided (null condition) and guided velocities as
        `vf_unguided + cfg_guidance_strength * (vf_guided - vf_unguided)`.

        :param cond: Dictionary mapping each perturbation covariate name to its corresponding conditioning
            tensor, held fixed across calls to the returned function. Required when
            :attr:`NeuralVelocityFieldConfig.use_guidance` is `True`, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Source (e.g.: control) state, held fixed across calls to the returned function.
            Required when :attr:`NeuralVelocityFieldConfig.use_source_as_condition` is `True`, defaults to
            `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free guidance term, only used when
            :attr:`NeuralVelocityFieldConfig.use_classifier_free_guidance` is `True`, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :return: A function mapping `(t, xt)` to the predicted velocity tensor of shape
            `(*batch_shape, flow_dim)`.
        :rtype: class:`Callable[[Tensor, Tensor], Tensor]`
        """
        # sanity checks
        if self.config.use_source_as_condition:
            msg = f""
            assert source is not None, msg

        def vf_fn(
            t: Tensor,
            xt: Tensor,
        ) -> Tensor:
            """Evaluates the (optionally classifier-free guided) velocity field at time `t` and state `xt`.

            :param t: Time steps, of shape `(*batch_shape,)`.
            :type t: class:`Tensor`

            :param xt: State at time `t`, of shape `(*batch_shape, flow_dim)`.
            :type xt: class:`Tensor`

            :return: The predicted velocity, of shape `(*batch_shape, flow_dim)`.
            :rtype: class:`Tensor`
            """
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
        """Builds the null (unconditional) counterpart of a condition dictionary for classifier-free guidance.

        :param cond: Dictionary mapping each perturbation covariate name to its corresponding conditioning
            tensor. If `None`, `None` is returned unchanged.
        :type cond: class:`dict[str, Tensor] | None`

        :return: `None` if `cond` is `None`, otherwise a dictionary with the same keys as `cond`, where every
            value is replaced with a tensor of the same shape filled with `self.config.null_condition_token`.
        :rtype: class:`dict[str, Tensor] | None`
        """
        # when condition is None we simply return None
        if cond is None:
            return None
        # otherwise we need to replace each value 
        # of the dictionary with a null condition token
        cond_copy = {}
        for key, val in cond.items():
            cond_copy[key] = torch.ones_like(val)*self.config.null_condition_token
        return cond_copy
