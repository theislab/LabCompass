import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch
import numpy as np
from torch import Tensor

from labcompass.constants import DataFields, LossFields, PredictionFields, VFStepFields
from labcompass.data import (
    BaseDataLoader,
    TrainDataLoader,
    ValidationDataLoader,
)
from labcompass.flows import BaseFlow
from labcompass.networks import NeuralVelocityField
from labcompass.ode import push_forward
from labcompass.training.callbacks import BaseCallBack
from labcompass.training.base import BaseTrainer
from labcompass.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "CFMTrainer",
]


class CFMTrainer(BaseTrainer):
    """Trainer implementing the training loop for the conditional flow-matching velocity field.

    :param velocity_field: The velocity field being trained; called on the flow's interpolated states to predict the target velocity.
    :type velocity_field: class:`NeuralVelocityField`

    :param flow: The flow used to sample interpolated states `xt` and target velocities `ut` between the source/latent state and the target state.
    :type flow: class:`BaseFlow`

    :param optimizer: Optimizer used to update :attr:`velocity_field`'s parameters.
    :type optimizer: class:`torch.optim.Optimizer`

    :param lr_scheduler: Optional learning rate scheduler, defaults to `None`.
    :type lr_scheduler: class:`torch.optim.lr_scheduler.LRScheduler | None`

    :param lr_scheduler_step: When to step :attr:`lr_scheduler`, either after each gradient step (`"grad_step"`) or after each validation step (`"valid_step"`), defaults to `"grad_step"`.
    :type lr_scheduler_step: class:`Literal["grad_step", "valid_step"]`

    :param time_sampler: Function used to sample the time steps at which the flow is evaluated during training, defaults to `torch.rand`.
    :type time_sampler: class:`Callable`

    :param callbacks: Optional callbacks run during training, defaults to `None`.
    :type callbacks: class:`BaseCallBack | None`

    :param grad_step_interval_log: Number of gradient steps between progress bar updates, defaults to `1000`.
    :type grad_step_interval_log: class:`int`

    :param num_time_steps: Number of discretization steps used when integrating the velocity field during validation, defaults to `100`.
    :type num_time_steps: class:`int`

    :param solver_kwargs: Dictionary of keyword arguments passed to :func:`push_forward` when integrating the dynamics during validation, defaults to `None`.
    :type solver_kwargs: class:`dict[str, Any] | None`

    :param has_controls: Whether the training data provides source/control states. When `False`, the latent state is instead drawn from :attr:`noise_distribution`, defaults to `True`.
    :type has_controls: class:`bool`

    :param generate_from_noise: Whether the flow's source is Gaussian noise rather than the batch's source state, defaults to `False`.
    :type generate_from_noise: class:`bool`

    :param noise_distribution: Function used to sample the latent state when :attr:`generate_from_noise` is `True` or when :attr:`has_controls` is `False`, defaults to `torch.randn`.
    :type noise_distribution: class:`Callable[[Sequence[int]], Tensor]`

    :param grad_steps_log_interval: Number of gradient steps after which :meth:`BaseTrainer.fit` updates the progress bar and runs the logging callbacks, defaults to `None`.
    :type grad_steps_log_interval: class:`bool | None`

    :param device_id: Identifier of the device used when pushing particles forward during validation, defaults to `"cuda"`.
    :type device_id: class:`Literal["cuda", "cpu"]`

    :param num_samples_per_validation_step: Number of samples generated per observation during validation. Only used when :attr:`generate_from_noise` is `True`, defaults to `None` in which case a single sample is generated.
    :type num_samples_per_validation_step: class:`int | None`

    :param cfg_prob_unconditional: Probability of replacing the condition with the null condition token during training, when the velocity field uses classifier-free guidance, defaults to `0.1`.
    :type cfg_prob_unconditional: class:`float`

    :param validation_cfg_guidance_strength: Strength of the classifier-free guidance term applied when pushing particles forward during validation, defaults to `1.0`.
    :type validation_cfg_guidance_strength: class:`float`

    :param num_grad_accumulation_steps: Number of gradient steps over which to accumulate gradients before stepping the optimizer, defaults to `1`.
    :type num_grad_accumulation_steps: class:`int`
    """

    def __init__(
        self,
        velocity_field: NeuralVelocityField,
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

    @property
    def model(
        self,
    ) -> NeuralVelocityField:
        """The model being optimized.

        :return: The wrapped velocity field, i.e. :attr:`velocity_field`.
        :rtype: class:`NeuralVelocityField`
        """
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
        vt = self.velocity_field(t, xt, condition, source=source)

        # computing losses
        loss = torch.nn.functional.mse_loss(vt, ut)
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
            self.velocity_field,
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
