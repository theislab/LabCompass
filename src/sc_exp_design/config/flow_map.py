from dataclasses import dataclass
from typing import Literal

from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig



@dataclass(slots=True)
class NeuralFlowMapConfig(NeuralVelocityFieldConfig):

    reparametrization_type: Literal["none", "residual", "redisual-rescaled"] = "residual"

    @property
    def decoder_input_dim(
        self,
    ) -> int:
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
