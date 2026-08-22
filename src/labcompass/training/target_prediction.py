from typing import Any, Literal

import torch
from torch import Tensor

from labcompass.constants import DataFields, LossFields, PredictionFields
from labcompass.networks.blocks import BaseModule
from labcompass.networks.neural_noise_models import MLPGaussianNoiseModel
from labcompass.training.base import BaseTrainer
from labcompass.training.callbacks import BaseCallBack
from labcompass.training.utils import compute_pert_inference_loss
from labcompass.types import TensorLike


__all__ = ["InverseModelTrainer"]


class TargetPredictionTrainer(BaseTrainer):
    """"""

    def __init__(
        self,
        target_prediction_model: BaseModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int | None = None,
        num_grad_accumulation_steps: int = 1,
        loss_fn_kwargs: dict[str, Any] | None = None
    ) -> None:
        """"""
        self.target_prediction_model = target_prediction_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.callbacks = callbacks
        self.grad_steps_log_interval = grad_steps_log_interval
        self.num_grad_accumulation_steps = num_grad_accumulation_steps 
        self.loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs

    @property
    def model(
        self,
    ) -> BaseModule:
        """"""
        return self.target_prediction_model

    @property
    def pert_cov_estimation_modes(
        self
    ) -> dict[str, Literal["isotropic", "anisotropic"] | None]:
        """"""
        covariance_estimation_modes = {}
        for covariate, covariate_approximate_posterior in self.target_prediction_model.pert_approximate_posterior.items():
            estimation_mode = None
            if isinstance(covariate_approximate_posterior, MLPGaussianNoiseModel):
                estimation_mode = covariate_approximate_posterior.cov_estimation_mode
            covariance_estimation_modes[covariate] = estimation_mode
        return covariance_estimation_modes

    def _train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # parsing batch dictonary
        states = batch[DataFields.STATE_DATA]
        targets = batch[DataFields.TARGET_CATEGORIES]
        # forward pass on the model
        predictions = self.target_prediction_model(states)
        # computing loss
        loss, log_dict = compute_pert_inference_loss(
            predictions,
            targets,
            self.target_prediction_model.noise_models,
            pert_cov_estimation_modes=self.pert_cov_estimation_modes,
            allow_noise_model_to_be_none=True,
            loss_fn_kwargs=self.loss_fn_kwargs
        )
        return loss, {LossFields.LOSS: loss.item(), **log_dict}
    
    def _validation_step(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # parsing batch dictonary
        states = batch[DataFields.STATE_DATA]
        targets = batch[DataFields.TARGET_CATEGORIES]
        predictions = self.target_prediction_model(states)
        return {
            k: {
                PredictionFields.PREDICTION_DATA: v.cpu().numpy(),
                DataFields.TARGET_STATE: targets[k].detach().cpu()
            } for k, v in predictions.items()
        }
