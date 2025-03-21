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
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.training.callbacks import CallBack
from sc_exp_design.training.utils import compute_pert_inference_loss
from sc_exp_design.types import TensorLike


__all__ = ["TargetPredictionTrainer", "InverseModelTrainer", ]


class TargetPredictionTrainer(BaseTrainer):
    """"""
    _require_solver_for_validation: bool = False

    def __init__(
        self,
        target_prediction_model: BaseModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 1000,
    ) -> None:
        """"""
        self.target_prediction_model = target_prediction_model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.callbacks = callbacks
        self.grad_step_interval_log = grad_step_interval_log

    @property
    def model(
        self,
    ) -> BaseModule:
        """"""
        return self.target_prediction_model

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
        loss = torch.zeros((), requires_grad=True)
        loss, log_dict = compute_pert_inference_loss(
            loss,
            predictions,
            targets,
            self.target_prediction_model.noise_models,
            pert_cov_estimation_modes=..., # TODO: we dont need it for the moment but we will need to pass it at some point.
            add_loss=True,
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
    _require_solver_for_validation: bool = False

    def __init__(
        self,
        inverse_model: BaseModule,
        forward_model: BaseModule,
        target_prediction_model: BaseModule,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 1000,
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
        self.grad_step_interval_log = grad_step_interval_log 

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
