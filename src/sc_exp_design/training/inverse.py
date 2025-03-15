from collections.abc import Callable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from sc_exp_design.constants import DataFields, LossFields
from sc_exp_design.data import SequentialDataLoader
from sc_exp_design.networks.blocks import BaseModule
from sc_exp_design.training.callbacks import CallBack
from sc_exp_design.types import TensorLike


__all__ = ["TargetPredictionTrainer", "InverseModelTrainer", ]


class TargetPredictionTrainer:
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

    def __train_step_(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # loss, log_dict = ..., ...
        # return loss, log_dict
        raise NotImplementedError
    
    def __validation_step_(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError


class InverseModelTrainer:
    """"""
    def __init__(
        self,
    ) -> None:
        """"""

    def __train_step_(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # loss, log_dict = ..., ...
        # return loss, log_dict
        raise NotImplementedError

    def __validation_step_(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError
