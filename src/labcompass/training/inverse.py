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
    """Trainer implementing the training loop for an inverse model.

    :param inverse_model: The model being trained; called as `self.inverse_model(source_states)` at each training step to compute the loss.
    :type inverse_model: class:`BaseModule`

    :param forward_model: Forward model instance associated with the trainer.
    :type forward_model: class:`BaseModule`

    :param target_prediction_model: Target-prediction model instance associated with the trainer.
    :type target_prediction_model: class:`BaseModule`

    :param optimizer: Optimizer used to update :attr:`inverse_model`'s parameters.
    :type optimizer: class:`torch.optim.Optimizer`

    :param lr_scheduler: Optional learning rate scheduler, defaults to `None`.
    :type lr_scheduler: class:`torch.optim.lr_scheduler.LRScheduler | None`

    :param lr_scheduler_step: When to step :attr:`lr_scheduler`, either after each gradient step (`"grad_step"`) or after each validation step (`"valid_step"`), defaults to `"grad_step"`.
    :type lr_scheduler_step: class:`Literal["grad_step", "valid_step"]`

    :param callbacks: Optional callbacks run during training, defaults to `None`.
    :type callbacks: class:`BaseCallBack | None`

    :param grad_steps_log_interval: Number of gradient steps after which :meth:`BaseTrainer.fit` updates the progress bar and runs the logging callbacks, defaults to `None`.
    :type grad_steps_log_interval: class:`int | None`

    :param num_grad_accumulation_steps: Number of gradient steps over which to accumulate gradients before stepping the optimizer, defaults to `1`.
    :type num_grad_accumulation_steps: class:`int`
    """

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
        """The model being optimized.

        :return: The wrapped inverse model, i.e. :attr:`inverse_model`.
        :rtype: class:`BaseModule`
        """
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
