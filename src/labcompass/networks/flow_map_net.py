import itertools
from functools import partial
import logging
from collections.abc import Callable, Iterator

import torch
from torch import Tensor, nn
from torch.func import functional_call

from labcompass.constants import VFStepFields
from labcompass.networks.blocks import BaseModule, ConditionEncoder, MLPBlock, ResnetBlock, FiLMBlock
from labcompass.config.flow_map import NeuralFlowMapConfig
from labcompass.utils import sinusoidal_time_features

logger = logging.getLogger(__name__)

__all__ = ["NeuralFlowMap"]


class NeuralFlowMap(BaseModule):
    """A neural flow map module predicting a direct (possibly multi-step) transport map between two time points.

    Given a start time `s`, an end time `t` and a state `xs` at time `s`, this module predicts the state at
    time `t`, optionally guided by perturbation conditions and a source state, allowing few-step or one-step
    integration of the underlying dynamics without numerically solving an ODE.

    :param config: Configuration object specifying the architecture of the flow map module (which encoders
        to build for state, start/end time, conditions and source, the conditioning strategy used to combine
        them, the reparametrization applied to the decoder output, and the configuration of the final
        decoder), used to build the submodules collected in :attr:`NeuralFlowMap.vf_modules`.
    :type config: class:`NeuralFlowMapConfig`
    """

    def __init__(
        self,
        config: NeuralFlowMapConfig,
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
            modules["t_encoder"] = MLPBlock(
                self.config.time_encoder_input_dim,
                self.config.time_encoder_output_dim,
                **self.config.time_encoder_mlp_kwargs,
            )
            modules["s_encoder"] = MLPBlock(
                self.config.time_encoder_input_dim,
                self.config.time_encoder_output_dim,
                **self.config.time_encoder_mlp_kwargs,
            )
        else:
            modules["t_encoder"] = torch.nn.Identity()
            modules["s_encoder"] = torch.nn.Identity()
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
        s: Tensor,
        t: Tensor,
        xs: Tensor,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None,
    ) -> Tensor:
        """Predicts the state at time `t` given the state `xs` at time `s`.

        Encodes the start time `s`, end time `t`, state `xs`, perturbation conditions and (optionally) the
        source state according to the :class:`NeuralFlowMapConfig` passed at initialization, combines the
        resulting latent representations using the configured conditioning strategy (`"concatenation"`,
        `"resnet"` or `"film"`), and decodes them into the predicted state at time `t`. The raw decoder
        output is then combined with `xs` according to :attr:`NeuralFlowMapConfig.reparametrization_type`:

        - `"none"`: the decoder output is returned directly as the predicted state.
        - `"residual"`: the prediction is `xs + (t - s) * decoder_output`.
        - `"redisual-rescaled"`: the prediction is `(1 - (t - s)) * xs + (t - s) * decoder_output`.

        :param s: Start time(s) from which the state `xs` is transported, of shape `(*batch_shape,)`.
        :type s: class:`Tensor`

        :param t: End time(s) at which the state is predicted, of shape `(*batch_shape,)`.
        :type t: class:`Tensor`

        :param xs: State at time `s`, of shape `(*batch_shape, flow_dim)`.
        :type xs: class:`Tensor`

        :param cond: Dictionary mapping each perturbation covariate name to its corresponding conditioning
            tensor. Required when :attr:`NeuralFlowMapConfig.use_guidance` is `True`, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Source (e.g.: control) state used as an additional conditioning signal. Required when
            :attr:`NeuralFlowMapConfig.use_source_as_condition` is `True`, defaults to `None`.
        :type source: class:`Tensor | None`

        :return: The predicted state at time `t`, of shape `(*batch_shape, flow_dim)`.
        :rtype: class:`Tensor`

        :raises ValueError: If :attr:`NeuralFlowMapConfig.reparametrization_type` is not one of `"none"`,
            `"residual"` or `"redisual-rescaled"`.
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
            t_latent = self.vf_modules["t_encoder"](t_latent)
    
        # encoding time
        s = torch.unsqueeze(s, dim=-1)
        s_latent = s
        if self.config.use_sinusoidal_time_features:
            s_latent = sinusoidal_time_features(
                s,
                num_freqs=self.config.time_features_num_freqs,
                max_period=self.config.time_features_max_periods
            )
        if self.config.encode_time:
            s_latent = self.vf_modules["s_encoder"](s_latent)

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
        xt_latent = xs
        if self.config.encode_state:
            xt_latent = self.vf_modules["x_encoder"](xs)

        # concatenating original and latent representations
        if self.config.conditioning_type == "concatenation":
            if self.config.use_guidance:
                # sanity check (condition should be not None)
                msg = f""
                assert cond is not None, msg
                latent_concat = torch.cat([t_latent, s_latent, xt_latent, condition_latent], dim=-1)
            else:
                latent_concat = torch.cat([t_latent, s_latent, xt_latent], dim=-1)
        elif self.config.conditioning_type == "resnet":
            latent_concat = xt_latent
            if self.config.use_guidance:
                condition_concat = torch.cat([t_latent, s_latent, condition_latent], dim=-1)  
            else:
                condition_concat = torch.cat([t_latent, s_latent], dim=-1)
        elif self.config.conditioning_type == "film":
            msg = f"FiLM is only possible with guidance"
            assert self.config.use_guidance, msg
            condition_concat = condition_latent 
            latent_concat = torch.cat([t_latent, s_latent, xt_latent], dim=-1)
     
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
        res = self.vf_modules["decoder"](latent_concat)

        # handle reparametrization
        if self.config.reparametrization_type == "none":
            return res
        elif self.config.reparametrization_type == "residual":
            return xs  + (t - s)*res
        elif self.config.reparametrization_type == "redisual-rescaled":
            return (1 - (t - s))*xs  + (t - s)*res
        else:
            msg = f"{self.config.reparametrization_type} not supported"
            raise ValueError(msg)

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

    def get_map_fn(
        self,
        cond: dict[str, Tensor] | None = None,
        source: Tensor | None = None
    ) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """Builds a flow map function of `(s, t, xs)` bound to fixed conditions and source state.

        The returned function evaluates :meth:`NeuralFlowMap.forward` at given `s`, `t` and `xs` values.

        :param cond: Dictionary mapping each perturbation covariate name to its corresponding conditioning
            tensor, held fixed across calls to the returned function. Required when
            :attr:`NeuralFlowMapConfig.use_guidance` is `True`, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Source (e.g.: control) state, held fixed across calls to the returned function.
            Required when :attr:`NeuralFlowMapConfig.use_source_as_condition` is `True`, defaults to `None`.
        :type source: class:`Tensor | None`

        :return: A function mapping `(s, t, xs)` to the predicted state at time `t`, of shape
            `(*batch_shape, flow_dim)`.
        :rtype: class:`Callable[[Tensor, Tensor, Tensor], Tensor]`
        """
        def flow_map_fn(
            s: Tensor,
            t: Tensor,
            xs:Tensor
        ) -> Tensor:
            """Evaluates the flow map from time `s` to time `t` for state `xs`, given the enclosing `cond` and `source`."""
            return self.forward(s, t, xs, cond=cond, source=source)
        return flow_map_fn
