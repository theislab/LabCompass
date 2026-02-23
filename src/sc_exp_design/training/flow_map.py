import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import numpy as np
from torch import Tensor

from sc_exp_design.constants import DataFields, LossFields, PredictionFields

from sc_exp_design.config.flow_map import NeuralFlowMapConfig
from sc_exp_design.flows import BaseFlow
from sc_exp_design.networks import NeuralVelocityField
from sc_exp_design.networks.flow_map_net import NeuralFlowMap
from sc_exp_design.ode import push_forward
from sc_exp_design.training.callbacks import BaseCallBack
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "FlowMapTrainer",
]


class FlowMapTrainer(BaseTrainer):
    """"""

    def __init__(
        self,
        flow_map: NeuralFlowMapConfig,
        flow: BaseFlow,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        time_sampler: Callable = torch.rand,
        callbacks: BaseCallBack | None = None,
        grad_step_interval_log: int = 1000,
        num_time_steps: int = 100,
        solver_kwargs: dict[str, Any] = None,
        has_controls: bool = True,
        generate_from_noise: bool = False,
        noise_distribution: Callable[[Sequence[int]], Tensor] = torch.randn,
        grad_steps_log_interval: bool | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
        num_samples_per_validation_step: int | None = None,
        cfg_prob_unconditional: float = 0.1,
        validation_cfg_guidance_strength: float = 1.0,
        num_grad_accumulation_steps: int = 1,
        velocity_field: NeuralVelocityField | None = None,
        weight_fn: None | Callable = lambda s, t: 1.0,
    ) -> None:
        """"""
        self.flow_map = flow_map
        self.flow = flow
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.time_sampler = time_sampler
        self.callbacks = callbacks
        self.grad_step_interval_log = grad_step_interval_log
        self.num_time_steps = num_time_steps
        self.solver_kwargs = solver_kwargs
        self.has_controls = has_controls
        self.generate_from_noise = generate_from_noise
        self.noise_distribution = noise_distribution
        self.grad_steps_log_interval = grad_steps_log_interval
        self.device_id = device_id
        self.num_samples_per_validation_step = num_samples_per_validation_step
        self.cfg_prob_unconditional = cfg_prob_unconditional
        self.validation_cfg_guidance_strength = validation_cfg_guidance_strength
        self.num_grad_accumulation_steps = num_grad_accumulation_steps 
        self.velocity_field = velocity_field
        self.weight_fn = weight_fn

    @property
    def model(
        self,
    ) -> NeuralFlowMap:
        """"""
        return self.flow_map

    def _compute_loss_distillation(
            self,
            s,
            t,
            latent,
            target,
            condition,
            source,
        ):
            # sample ground truth interpolant
            xs = self.flow.compute_x_t(s, latent, target)
    
            # Get the functional function and the parameter dict
            # vf_fn, params = self.flow_map.get_map_fn(condition, source=source)

            # Now, include 'params' in the JVP transformation
            # map_fn = self.flow_map.get_map_fn(condition, source=source)
            # def jvp_wrapper(s_i, t_i, xs_i):
            #     # We want dX/dt, so we only provide a tangent for t_i (the 2nd arg after params)
            #     # We use a zero-tangent for params because we aren't deriving wrt weights here,
            #     # BUT keeping them in the closure of the functional_call allows backprop to find them.
            #     return torch.func.jvp(
            #         lambda s_val, t_val, x_val: map_fn(s_val, t_val, x_val),
            #         (s_i, t_i, xs_i),
            #         (torch.zeros_like(s_i), torch.ones_like(t_i), torch.zeros_like(xs_i))
            #     )

            # xts_hat, dXdt = torch.vmap(jvp_wrapper)(s, t, xs)

            # # forward pass on neural networks with jvp
            xts_hat, dXdt = torch.vmap(
                lambda s_i, t_i, xs_i: torch.func.jvp(
                    self.flow_map.get_map_fn(condition, source=source), 
                    (s_i, t_i, xs_i),
                    (torch.zeros_like(s_i), torch.ones_like(t_i), torch.zeros_like(xs_i)),
                )
            )(s, t, xs)
            vt = self.velocity_field(t, xts_hat, condition, source=source)

            # # Derivative with respect to time
            # def dXdt_single(x, s, t):
            #     return torch.func.jacrev(
            #         self.flow_map.get_map_fn(condition, source=source),
            #         argnums=1
            #     )(s, t, x)
    
            # # forward pass on neural networks with jacrev
            # xts_hat_jcrv = self.flow_map(s, t, xs, cond=condition, source=source)
            # dXdt_jcrv = torch.vmap(dXdt_single)(xs, s, t)

            # print(f"{xts_hat.shape=}, {dXdt.shape=}, {xts_hat_jcrv.shape=}, {dXdt_jcrv.shape=}")

            # errors
            # err_map = ((xts_hat_jcrv - xts_hat)**2).sum(-1).mean()
            # err_tang = ((dXdt_jcrv - dXdt)**2).sum(-1).mean()
            # print(f"{err_map=}, {err_tang=}")

            return torch.mean(self.weight_fn(s, t) * ((dXdt - vt)**2).sum(-1))

    def _compute_loss_end_to_end(
        self,
        s,
        t,
        latent,
        target,
        condition,
        source,
    ):
        # sample ground truth interpolant and compute corresponding velocity field
        xt = self.flow.compute_x_t(t, latent, target)
        ut = self.flow.compute_u_t(t, latent, target, xt)
    
        # forward pass on neural networks
        xst_hat = self.flow_map(t, s, xt, condition, source=source)
        _, dXdt = torch.vmap(
            lambda s_i, t_i, xs_i: torch.func.jvp(
                self.flow_map.get_map_fn(condition, source=source), 
                (s_i, t_i, xs_i),
                (torch.zeros_like(s_i), torch.ones_like(t_i), torch.zeros_like(xs_i)),
            )
        )(s, t, xst_hat)

        return torch.mean(self.weight_fn(s, t) * ((dXdt - ut)**2).sum(-1))

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
        if self.flow_map.config.use_classifier_free_guidance:
            if torch.rand(1).item() < self.cfg_prob_unconditional:
                condition = self.flow_map.get_null_condition_token(condition)

        # retrieving batch size and ode time
        batch_size = target.shape[0]
        s, t = self.time_sampler((batch_size,), device=target.device)

        # distillation
        if self.velocity_field is not None:
            loss = self._compute_loss_distillation(
                s,
                t,
                latent,
                target,
                condition,
                source,
            )
        else:
            loss = self._compute_loss_end_to_end(
                s,
                t,
                latent,
                target,
                condition,
                source,
            )
        return loss, {LossFields.LOSS: loss.detach().cpu().item()}

    def __validation_step(
        self,
        perturbation_batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        # handling source
        source = None
        if self.has_controls:
            source = perturbation_batch[DataFields.SOURCE_STATE]
        # retrieving target
        target = perturbation_batch[DataFields.TARGET_STATE]
        # optional condition key
        condition = None
        if DataFields.PERTURBATION_DATA in perturbation_batch.keys():
            condition = perturbation_batch[DataFields.PERTURBATION_DATA]
        # pushing forward the particles
        predictions = push_forward(
            self.flow_map,
            source,
            condition,
            self.generate_from_noise,
            self.noise_distribution,
            self.num_time_steps,
            self.solver_kwargs,
            self.device_id,
            return_trajectory=False,
            no_grad=True,
            num_samples=self.num_samples_per_validation_step,
            batch_size=target.shape[0],
            cfg_guidance_strength=self.validation_cfg_guidance_strength,
        )
        if self.num_samples_per_validation_step is None:
            return predictions, target
        # handling number of samples
        if self.num_samples_per_validation_step is not None:
            num_samples = self.num_samples_per_validation_step
            if not self.generate_from_noise:
                msg = f""
                logger.warning(msg)
                num_samples = 1
        else:
            num_samples = 1
        msg = f""
        assert isinstance(num_samples, int), msg
        # handling the shape of the target when we sample multiple predictions
        target = target.unsqueeze(0)
        target = target.repeat(num_samples, *(1 for _ in predictions.shape[1:]))
        return predictions.reshape(-1, predictions.shape[-1]), target.reshape(-1, target.shape[-1])

    def _validation_step(
        self,
        batch: dict[str, dict[str, TensorLike]],
    ) -> dict[str, dict[str, TensorLike]]:
        """"""
        # list to store all the results
        predictions = []
        targets = []
        
        # dictionary to store the results per perturbation
        predictions_dict = {}

        # iterating over the perturbations of the current batch
        for perturbation, perturbation_batch in batch.items():
            # performing validation step on single perturbation
            perturbation_predictions, perturbation_targets = self.__validation_step(perturbation_batch)

            # detaching predictions from graph and moving tensors to numpy
            perturbation_predictions = perturbation_predictions.cpu().numpy()
            perturbation_targets = perturbation_targets.cpu().numpy()

            # appending to the list of all results
            predictions.append(perturbation_predictions)
            targets.append(perturbation_targets)

            # storing the results to the output grouped per perturbation
            predictions_dict[perturbation] = {
                PredictionFields.PREDICTION_DATA: perturbation_predictions,
                DataFields.TARGET_STATE: perturbation_targets
            }
        
        # concatenating the results for all conditions
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # updating results dictionary with predictions concatenated over all conditions
        predictions_dict["all_conditions"] = {
            PredictionFields.PREDICTION_DATA: predictions,
            DataFields.TARGET_STATE: targets
        }

        return predictions_dict
