from collections.abc import Callable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from sc_exp_design.constants import DataFields, LossFields, VFStepFields
from sc_exp_design.data import (
    TrainDataLoader,
    ValidationDataLoader,
)
from sc_exp_design.flows import BaseFlow
from sc_exp_design.networks import NeuralVelocityField
from sc_exp_design.ode import ODESolver
from sc_exp_design.training.callbacks import CallBack
from sc_exp_design.training.utils import (
    compute_cond_vars_inference_loss,
    compute_latent_perturbation_inference_loss,
    compute_pert_inference_loss,
)
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.types import TensorLike

__all__ = [
    "CFMTrainer",
]


class CFMTrainer(BaseTrainer):
    """"""
    _require_solver_for_validation: bool = True

    def __init__(
        self,
        velocity_field: NeuralVelocityField,
        flow: BaseFlow,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        time_sampler: Callable = torch.rand,
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 1000,
        num_time_steps: int = 100,
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        solver_kwargs: dict[str, Any] = None,
        posterior_on_cond_vars_update_step: int | None = None,
        posterior_on_perts_update_step: int | None = None,
        posterior_on_latent_perts_update_step: int | None = None,
        has_controls: bool = True,
        generate_from_noise: bool = False,
    ) -> None:
        """"""
        self.velocity_field = velocity_field
        self.flow = flow
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.time_sampler = time_sampler
        self.callbacks = callbacks
        self.grad_step_interval_log = grad_step_interval_log
        self.num_time_steps = num_time_steps
        self.gamma_fn = gamma_fn
        self.solver_kwargs = solver_kwargs
        self.posterior_on_cond_vars_update_step = posterior_on_cond_vars_update_step
        self.posterior_on_perts_update_step = posterior_on_perts_update_step
        self.posterior_on_latent_perts_update_step = posterior_on_latent_perts_update_step
        self.has_controls = has_controls
        self.generate_from_noise = generate_from_noise

    @property
    def model(
        self,
    ) -> NeuralVelocityField:
        """"""
        return self.velocity_field

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
            latent = torch.randn_like(target)
        # optional condition key
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]
        # retrieving batch size and ode time
        batch_size = source.shape[0]
        t = self.time_sampler((batch_size,), device=source.device)
        # computing flow and target velocity field
        xt = self.flow.compute_x_t(t, latent, target)
        ut = self.flow.compute_u_t(t, latent, target, xt)
        # forward pass on the neural vf
        vt_step = self.velocity_field(t, xt, condition, source=source)
        vt = vt_step[VFStepFields.VF]
        # computing losses
        loss = torch.nn.functional.mse_loss(vt, ut)
        return loss, {LossFields.LOSS: loss.detach().cpu().item()}

    def _validation_step(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # parsing batch dictionary
        source = batch[DataFields.SOURCE_STATE]
        target = batch[DataFields.TARGET_STATE]
        # optional condition key
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]
        # defining velocity function
        vf = self.velocity_field.get_vf_fn(condition, gamma_fn=self.gamma_fn)
        # initializing the sampler clss
        ode_sampler = ODESolver(
            vf,
            num_time_steps=self.num_time_steps,
            gamma_fn=self.gamma_fn,
            solver_kwargs=self.solver_kwargs,
        )
        predictions = ode_sampler.integrate(source)
        return predictions, target
