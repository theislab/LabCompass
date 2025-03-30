from collections.abc import Callable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from sc_exp_design.constants import DataFields, LossFields, VFStepFields
from sc_exp_design.data import SequentialDataLoader
from sc_exp_design.networks.blocks import BaseModule
from sc_exp_design.networks.neural_noise_models import MLPGaussianNoiseModel
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.training.callbacks import BaseCallBack
from sc_exp_design.training.utils import compute_pert_inference_loss
from sc_exp_design.types import TensorLike


__all__ = ["TargetPredictionTrainer", "InverseModelTrainer", ]


class TargetPredictionTrainer(BaseTrainer):
    """"""

    def __init__(
        self,
        target_prediction_model: BaseModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int | None = None
    ) -> None:
        """"""
        self.target_prediction_model = target_prediction_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.callbacks = callbacks
        self.grad_steps_log_interval = grad_steps_log_interval

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
            pert_cov_estimation_modes=self.pert_cov_estimation_modes, # TODO: we dont need it for the moment but we will need to pass it at some point.
            allow_noise_model_to_be_none=True,
        )
        return loss, {LossFields.LOSS: loss.item(), **log_dict}
    
    def _validation_step(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError


class InverseModelTrainer(BaseTrainer):
    """"""

    def __init__(
        self,
        inverse_model: BaseModule,
        forward_model: BaseModule,
        target_prediction_model: BaseModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int | None = None,
    ) -> None:
        """"""
        super().__init__()
        
        self.inverse_model = inverse_model
        self.forward_model = forward_model
        self.target_prediction_model = target_prediction_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.callbacks = callbacks
        self.grad_steps_log_interval = grad_steps_log_interval 

    @property
    def model(
        self,
    ) -> BaseModule:
        """"""
        return self.inverse_model

    def _train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # parsing batch dictonary
        source_states = batch[DataFields.STATE_DATA]

        loss, out_dict = self.inverse_model(source_states)    
        return loss, {LossFields.LOSS: loss.item(), **out_dict}
        
    def _validation_step(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError
