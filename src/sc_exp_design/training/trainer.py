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
from sc_exp_design.types import TensorLike

__all__ = [
    "CFMTrainer",
]


class CFMTrainer:
    """"""

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
        solver_class: ODESolver | None = None,
        num_time_steps: int = 100,
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        solver_kwargs: dict[str, Any] = None,
        posterior_on_cond_vars_update_step: int | None = None,
        posterior_on_perts_update_step: int | None = None,
        posterior_on_latent_perts_update_step: int | None = None,
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
        self.solver_class = solver_class
        self.num_time_steps = num_time_steps
        self.gamma_fn = gamma_fn
        self.solver_kwargs = solver_kwargs
        self.posterior_on_cond_vars_update_step = posterior_on_cond_vars_update_step
        self.posterior_on_perts_update_step = posterior_on_perts_update_step
        self.posterior_on_latent_perts_update_step = posterior_on_latent_perts_update_step

    def __train_step_(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """"""
        # parsing batch dictionary
        source = batch[DataFields.SOURCE_STATE]
        target = batch[DataFields.TARGET_STATE]
        condition = batch[DataFields.PERTURBATION_DATA]
        # retrieving batch size and ode time
        batch_size = source.shape[0]
        t = self.time_sampler((batch_size,), device=source.device)
        # computing flow and target velocity field
        xt = self.flow.compute_x_t(t, source, target)
        ut = self.flow.compute_u_t(t, source, target, xt)
        # forward pass on the neural vf
        vt_step = self.velocity_field(t, xt, condition, source=source, target=target)
        vt = vt_step[VFStepFields.VF]
        # computing losses
        vf_loss = torch.nn.functional.mse_loss(vt, ut)
        loss = vf_loss

        log_dict = {LossFields.VF_LOSS: vf_loss.detach().cpu()}
        # optional loss on the score field
        if self.velocity_field.config.learn_score_field:
            score_t = self.flow.compute_score_t(t, source, target, xt)
            st = vt_step[LossFields.SCORE]
            score_loss = torch.nn.functional.mse_loss(st, score_t)
            loss = loss + score_loss
            log_dict.update({LossFields.SCORE_LOSS: score_loss.detach().cpu()})
        # optional decoding on conditioning variables
        if self.velocity_field.config.learn_posterior_on_cond_vars:
            src_posterior_params = vt_step[VFStepFields.SOURCE_PARAMS]
            tgt_posterior_params = vt_step[VFStepFields.TARGET_PARAMS]

            # optionally stopping backpropagation of the loss
            add_loss = True
            if self.posterior_on_cond_vars_update_step is not None:
                add_loss = step_idx % self.posterior_on_cond_vars_update_step == 0

            # retrieving covariance estimation mode in case of gaussian noise model
            src_cov_estimation_mode = None
            tgt_cov_estimation_mode = None
            if self.velocity_field.config.src_noise_model == "gaussian":
                src_cov_estimation_mode = self.velocity_field.endpoints_approximate_posterior.src_approximate_posterior.cov_estimation_mode
            if self.velocity_field.config.tgt_noise_model == "gaussian":
                tgt_cov_estimation_mode = self.velocity_field.endpoints_approximate_posterior.tgt_approximate_posterior.cov_estimation_mode

            loss, cond_var_posterior_loss = compute_cond_vars_inference_loss(
                loss,
                source,
                target,
                src_posterior_params,
                tgt_posterior_params,
                self.velocity_field.config.src_noise_model,
                self.velocity_field.config.tgt_noise_model,
                src_cov_estimation_mode,
                tgt_cov_estimation_mode,
                add_loss=add_loss,
            )
            log_dict.update(cond_var_posterior_loss)

        # optional decoding on perturbations
        if self.velocity_field.config.learn_posterior_on_perts:
            pert_posterior_params = vt_step[VFStepFields.PERTURBATION_PARAMS]
            pert_target_rep = batch[DataFields.PERTURBATION_TARGET_REPR]

            # optionally stopping backpropagation of the loss
            add_loss = True
            if self.posterior_on_perts_update_step is not None:
                add_loss = step_idx % self.posterior_on_perts_update_step == 0

            loss, pert_posterior_loss = compute_pert_inference_loss(
                loss,
                pert_posterior_params,
                pert_target_rep,
                self.velocity_field.config.pert_noise_model,
                self.velocity_field.config.pert_cov_estimation_modes,
                add_loss=add_loss,
            )
            log_dict.update(pert_posterior_loss)
        # optional inference on perturbation latent state from endpoints
        if self.velocity_field.config.learn_posterior_on_latent_perts:
            latent_perturbation = vt_step[VFStepFields.LATENT_PERTURBATION]
            if self.velocity_field.config.latent_perts_posterior_freeze_grads:
                latent_perturbation = latent_perturbation.detach()

            # optionally stopping backpropagation of the loss
            add_loss = True
            if self.posterior_on_latent_perts_update_step is not None:
                add_loss = step_idx % self.posterior_on_latent_perts_update_step == 0

            params = self.velocity_field.get_latent_condition_inf_params(source, target)
            loss, latent_cond_inf_loss = compute_latent_perturbation_inference_loss(
                loss,
                params,
                latent_perturbation,
                self.velocity_field.latent_pert_approximate_posterior.cov_estimation_mode,
                add_loss=add_loss,
            )
            log_dict.update(latent_cond_inf_loss)
        # updating log dictionary with final loss value
        log_dict[LossFields.LOSS] = loss.detach().cpu().item()
        return loss, log_dict

    def __validation_step_(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # parsing batch dictionary
        source = batch[DataFields.SOURCE_STATE]
        target = batch[DataFields.TARGET_STATE]
        condition = batch[DataFields.PERTURBATION_DATA]
        # defining velocity function
        vf = self.velocity_field.get_vf_fn(condition, gamma_fn=self.gamma_fn)
        # initializing the sampler clss
        ode_sampler = self.solver_class(
            vf,
            num_time_steps=self.num_time_steps,
            gamma_fn=self.gamma_fn,
            solver_kwargs=self.solver_kwargs,
        )
        predictions = ode_sampler.integrate(source)
        return predictions, target

    def __train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> TensorLike:
        """"""
        # optimizations step
        self.velocity_field.train()
        self.optimizer.zero_grad()
        loss, log_dict = self.__train_step_(step_idx, batch)
        loss.backward()
        self.optimizer.step()
        # learning rate scheduler step
        if self.lr_scheduler_step == "grad_step" and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # running callbacks
        if self.callbacks is not None:
            self.callbacks.run_on_grad_step()
        return log_dict

    def __validation_step(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, TensorLike]:
        """"""
        self.velocity_field.eval()
        with torch.no_grad():
            val_preds, val_gt = self.__validation_step_(batch)
            val_preds = val_preds.cpu().numpy()
            val_gt = val_gt.cpu().numpy()
        # learning rate scheduler step
        if self.lr_scheduler_step == "valid_step" and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # running callbacks
        if self.callbacks is not None:
            self.callbacks.run_on_grad_step()
        return val_preds, val_gt

    def __update_logs(
        self,
        metrics: dict[str, float],
    ) -> None:
        """"""
        for metric_id, metric_val in metrics.items():
            if metric_id not in self.training_logs.keys():
                self.training_logs[metric_id] = []
            self.training_logs[metric_id].append(metric_val)

    def fit(
        self,
        num_training_steps: int,
        train_dataloader: TrainDataLoader,
        validation_dataloader: ValidationDataLoader | None = None,
        valid_freq: int | None = None,
    ) -> None:
        """"""

        self.training_logs = {LossFields.LOSS: []}

        iterator = range(num_training_steps)
        prog_bar = tqdm(iterator)

        do_validation = validation_dataloader is not None
        if do_validation and valid_freq is None:
            valid_freq = num_training_steps

        for grad_step in iterator:
            batch = train_dataloader.sample()
            log_dict = self.__train_step(grad_step, batch)
            self.__update_logs(log_dict)

            # updaring progress bar
            if (grad_step + 1) % self.grad_step_interval_log and grad_step > 0:
                prog_bar.set_description(f"Loss: {log_dict[LossFields.LOSS]:.4f}")
                prog_bar.update()

            # validation step
            if do_validation:
                if (grad_step + 1) % valid_freq == 0 and grad_step > 0:
                    # skipping if no dataloader provided
                    if validation_dataloader is None:
                        continue
                    # sanity check
                    msg = "To perform the validation step `self.solver_class` must be an instance of `ODESolver`, found `None`."
                    assert self.solver_class is not None, msg

                    batch = validation_dataloader.sample()
                    val_preds, val_gt = self.__validation_step(batch)

                    # computing metrics
                    if self.callbacks is not None:
                        metrics = self.callbacks.run_on_valid_step(val_preds, val_gt)
                        self.__update_logs(metrics)

    def plot_training_logs(
        self,
        figsize: Sequence[int] = (5, 3),
        show: bool = False,
    ) -> tuple[Figure, Axes]:
        """"""
        fig, axes = plt.subplots(1, len(self.training_logs), figsize=figsize)
        for idx, (loss_id, loss_history) in enumerate(self.training_logs.items()):
            axes[idx].set_title(loss_id)
            axes[idx].plot(loss_history)
        if show:
            fig.show()
        return fig, axes
