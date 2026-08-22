import logging

import torch
import numpy as np
from torch import Tensor

from labcompass.utils import match_shapes
from labcompass.constants import DataFields, LossFields
from labcompass.training.flow_matching import CFMTrainer
from labcompass.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "CFMTrainerWithScore",
]


class CFMTrainerWithScore(CFMTrainer):
    """"""

    def _train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # parsing batch dictionary
        target = batch[DataFields.TARGET_STATE]
        if self.has_controls:
            source = batch[DataFields.SOURCE_STATE]
            latent = source
            if self.generate_from_noise:
                latent = torch.randn_like(source)
        else:
            source = None
            msg = f""
            assert self.generate_from_noise, msg
            latent = self.noise_distribution(target.shape).to(target.device)

        # optional condition key
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]
        # handling the case of unconditional generation
        if self.velocity_field.config.use_classifier_free_guidance:
            if torch.rand(1).item() < self.cfg_prob_unconditional:
                condition = self.velocity_field.get_null_condition_token(condition)

        # retrieving batch size and ode time
        batch_size = target.shape[0]
        t = self.time_sampler((batch_size,), device=target.device)

        # computing flow and target velocity field
        xt = self.flow.compute_x_t(t, latent, target)
        ut = self.flow.compute_u_t(t, latent, target, xt)

        # forward pass on the neural vf
        vt, st = self.velocity_field(t, xt, condition, source=source)

        # computing coefficients of flow for score
        mu_t = self.flow.compute_mu_t(
            match_shapes(t, target),
            latent,
            target
        )
        sigma_t = self.flow.compute_sigma_t(t).unsqueeze(1)
        sigma_t = sigma_t**2

        # computing losses
        loss_vt = torch.nn.functional.mse_loss(vt, ut)
        loss_st = torch.nn.functional.mse_loss(sigma_t*st, -(xt - mu_t))
        loss = loss_vt + loss_st

        return loss, {LossFields.LOSS: loss.detach().cpu().item()}
