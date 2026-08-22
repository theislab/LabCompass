from typing import Any, Literal

import torch
from torch import Tensor

from labcompass.constants import DataFields, LossFields
from labcompass.networks.blocks import BaseModule
from labcompass.networks.neural_noise_models import MLPGaussianNoiseModel
from labcompass.training.base import BaseTrainer
from labcompass.training.callbacks import BaseCallBack
from labcompass.training.utils import compute_pert_inference_loss
from labcompass.types import TensorLike


__all__ = ["InverseModelTrainer", ]


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
        num_grad_accumulation_steps: int = 1,
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
        self.num_grad_accumulation_steps = num_grad_accumulation_steps 

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
