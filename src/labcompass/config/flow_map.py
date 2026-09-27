from dataclasses import dataclass
from typing import Literal

from labcompass.config.velocity_field import NeuralVelocityFieldConfig


@dataclass(slots=True)
class NeuralFlowMapConfig(NeuralVelocityFieldConfig):
    """Object for configuring :class:`NeuralFlowMap` objects.

    Extends :class:`NeuralVelocityFieldConfig` to configure a flow map, i.e.: a network that maps a state at a
    start time `s` directly to the corresponding state at an end time `t`, rather than a velocity field defined
    at a single time. As a consequence, the start time `s` and end time `t` are each independently encoded, so
    the joint latent space used by the decoder (see :attr:`NeuralFlowMapConfig.decoder_input_dim` and
    :attr:`NeuralFlowMapConfig.resnet_embedding_dim`) accounts for :attr:`NeuralVelocityFieldConfig.time_latent_dim`
    twice, once for each time.

    :param reparametrization_type: Controls how the raw decoder output `res` is turned into the predicted state at time `t`,
        given the known state `xs` at time `s`:
        - `"none"`: the decoder output `res` is returned directly as the predicted state.
        - `"residual"`: returns `xs + (t - s) * res`, i.e.: `res` is treated as a rate scaled by the time gap `(t - s)`
            and added as a residual to `xs`, so the map reduces to the identity when `s == t`.
        - `"redisual-rescaled"` (verbatim, as spelled in the code): returns `(1 - (t - s)) * xs + (t - s) * res`,
            a convex combination of `xs` and `res` weighted by `(t - s)`.
        Defaults to `"residual"`.
    :type reparametrization_type: class: `Literal["none", "residual", "redisual-rescaled"]`
    """

    reparametrization_type: Literal["none", "residual", "redisual-rescaled"] = "residual"

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """
        Collects the dimension of the joint latent space, used to infer the input dimension for the flow map decoder.
        It adds the correct dimensions for each different case. Unlike :attr:`NeuralVelocityFieldConfig.decoder_input_dim`,
        the time latent dimension is counted twice, once for the start time `s` and once for the end time `t`.

        :rtype: class: `int`
        """
        # concatenation state, conditions, source and time
        if self.conditioning_type == "concatenation":
            return self.state_latent_dim + self.time_latent_dim*2 + self.perturbation_latent_dim + self.source_latent_dim
        # resnet block
        elif self.conditioning_type == "resnet":
            return self.state_latent_dim
        # film block
        elif self.conditioning_type == "film":
            return self.state_latent_dim + self.time_latent_dim*2

    @property
    def resnet_embedding_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the residual network condition embedding."""
        return self.time_latent_dim*2 + self.perturbation_latent_dim + self.source_latent_dim
