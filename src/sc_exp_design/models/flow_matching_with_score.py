import logging
import os
from collections.abc import Mapping
from typing import Any, Literal

import torch

from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig

from sc_exp_design.data import TrainDataLoader, ValidationDataLoader
from sc_exp_design.models.base import BaseModel
from sc_exp_design.models.flow_matching import FlowMatching
from sc_exp_design.networks.velocity_field_with_score import NeuralVelocityFieldWithScore
from sc_exp_design.training import BaseCallBack
from sc_exp_design.training.flow_matching_with_score import CFMTrainerWithScore
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["FlowMatchingWithScore"]


class FlowMatchingWithScore(FlowMatching):

    def prepare_model(
        self,
        cvf_config: NeuralVelocityFieldConfig,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.0001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        num_time_steps: int = 100,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        :param cvf_config: Instance of :class:`sc_exp_design.networks.NeuralVelocityFieldConfig` used to initialize the
            :class:`sc_exp_design.networks.NeuralVelocityFIeld` object, then set as :attr:`FlowMatching.velocity_field` attribute.
        :type cvf_config: class: `sc_exp_design.networks.NeuralVelocityFieldConfig`

        :param optimizer_class: Optimizer used to update the model's weights during training. Should reference a class derived from
            :class:`torch.optim.Optimizer` and not an instance, defaults to :class:`torch.optim.AdamW`.
        :type optimizer_class: class:`torch.optim.Optimizer`

        :param lr_scheduler_class: Optional scheduler used to updated the learning rate during optimization. SHould reference a class
            detive from :class:`torch.optim.lr_scheduler.LRScheduler` and not an instance, defaults to `None`.
        :type lr_scheduler_class: class:`torch.optim.lr_scheduler.LRScheduler | None`

        :param optimizer_kwargs: Dictionary containing the keyword arguments used to initialize the :param:`optimizer_class`, defaults to `{"lr": 0.001}`.
        :type optimizer_kwargs: class:`dict[str, Any]`

        :param lr_scheduler_kwargs: Dictionary containing the keyword arguments used to initialize the :param:`lr_scheduler_class`, defaults to `None`.
        :type lr_scheduler_kwargs: class:`dict[str, Any] | None`

        :param lr_scheduler_step: (Optional) :class:`str` identifier indicating when to perform the learning rate scheduling step, if a :param:`lr_scheduler_class`
            is specified (otherwise it is ignored). When :param:`lr_scheduler_step` is `"grad_step"` the learning rate will be updated after each gradient step.
            Otherwise when set to `"epoch"`, the learning rate will be updated after each validation step, defaults to `"grad_step"`.
        :type lr_scheduler_step: class: `Literal["grad_step", "epoch"]`

        :param num_time_steps: Number of time steps which to integrate the dynamics over during inference, defaults to `100`.
        :type num_time_steps: class:`int`

        :param solver_kwargs: Dictionary containining the keyword arguments used to initialize the :param:`solver_class`, defaults to `None`.
        :type solver_kwargs: class:`dict[str, Any] | None`
        """
        if (not self.data_manager.has_controls) and cvf_config.use_source_as_condition:
            msg = "When no controls are available use_source_as_condition should be False."
            raise ValueError(msg)
        elif self.data_manager.has_controls and self.generate_from_noise and (not cvf_config.use_source_as_condition):
            msg = "When generating from noise you need to use source as conditions."
            raise ValueError(msg)
        elif (not self.data_manager.has_controls) and (not self.generate_from_noise):
            msg = f"When no controls are available you need to generate from noise."
            raise ValueError(msg)

        self.cvf_config = cvf_config
        
        # given a dimensionality and a configuration of hparams, initialize a flow model 
        self.velocity_field = NeuralVelocityFieldWithScore(
            config=self.cvf_config,
        )
        self.velocity_field = self.velocity_field.float()
        self.velocity_field = self.velocity_field.to(self.device)

        # optimizer and scheduler 
        self.optimizer = optimizer_class(
            self.velocity_field.parameters(),
            **optimizer_kwargs,
        )

        # initialize scheduler
        self.lr_scheduler = None
        self.lr_scheduler_step = None
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler_class is not None:
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **lr_scheduler_kwargs)
            self.lr_scheduler_step = lr_scheduler_step

        self.num_time_steps = num_time_steps
        self.solver_kwargs = solver_kwargs

    def train(
        self,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
        num_treatments_to_load: int | None = None,
        num_samples_per_validation_step: int | None = None,
        cfg_prob_unconditional: float = 0.1,
        validation_cfg_guidance_strength: float = 1.0,
        num_grad_accumulation_steps: int = 1,
        close_wandb_connection: bool = True,
    ) -> None:
        """Trains the model.

        :param num_training_steps: The number of steps which to train the model on, defaults to `500`.
        :type num_training_steps: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step, in case the
            :attr:`FlowMatching.validation_data` was initialized by using :method:`FlowMatching.prepare_validation_data`, defaults to `None`.
        :type valid_freq: class:`int | None`

        :param train_batch_size: The batch size used for sampling the training data, defaults to `1024`.
        :type train_batch_size: class:`int`

        :param validation_batch_size: The batch size for sampling the validation data, defaults to `512`
        :type validation_batch_size: class:`int`

        :param state_transforms: (Optional) transformations applied to the states before feeding them into the model.
            Should be an instance of a class derived from :class:`sc_exp_design.transforms.Transform` and provide at least the method :method:`state_transforms.transform`
            and optionally the method :method:`state_transforms.inverse_transform` in case of invertible transformations. Defaults to `None`.
        :type state_transforms: class:`Transforms`

        :param callbacks: (Optional) callbacks that will be called during training. Still work in progress, defaults to `None`.
        :type callbacks: class:`BaseCallBack`

        :param grad_step_interval_log: The number of gradient steps after which to update the progress bar, defaults to `100`.
        :type grad_step_interval_log: class:`int`

        :param num_treatments_to_load: Specifies the maximum number of unique treatments to be loaded in a single batch.
            Defaults to `None`, in which case all unique treatments are loaded.
        :type num_treatments_to_load: class: `int | None`

        :param num_samples_per_validation_step: Specifies the number of samples for each observation to be generated during the validation step.
            Only used when :attr: `self.generate_from_noise` is set to `True`, defaults to `None` in which case only one sample will be generated.
        :type num_samples_per_validation_step: class: `int | None`

        :param cfg_prob_unconditional: Probability of sampling the null condition token during the training of the velocitf field.
            Only used when :attr: `self.cvf_config.use_classifier_free_guidance` is set to `True`, defaults to `0.1`.
        :type cfg_prob_unconditional: class: `float`

        :param validation_cfg_guidance_strength: Strength of the guidance term during the validation step.
            Only used when :attr: `self.cvf_config.use_classifier_free_guidance` is set to `True`, defaults to `1.0`.
        :type validation_cfg_guidance_strength: class: `float`

        :param num_grad_accumulation_steps: The number of gradient steps which to accumulate the gradients over, defaults to `1`.
        :type num_grad_accumulation_steps: class:`int`
        """
        # sanity checks
        msg = "Data not initialized, run `prepare_data` before training the model"
        assert self.train_data is not None, msg
        msg = "Model not initialized, run `prepare_model` before training the model"
        assert self.velocity_field is not None, msg
        if self.cvf_config.use_classifier_free_guidance:
            msg = "The probability of sampling the null condition token must be less than 1 for classifier-free guidance"
            assert cfg_prob_unconditional < 1, msg

        # storing state transforms as attribute
        self.state_transforms = state_transforms

        # storing cfg arguments as attributes
        self.cfg_prob_unconditional = cfg_prob_unconditional
        self.validation_cfg_guidance_strength = validation_cfg_guidance_strength

        self.trainer = CFMTrainerWithScore(
            self.velocity_field,
            self.flow,
            self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_step=self.lr_scheduler_step,
            time_sampler=self.time_sampler,
            callbacks=callbacks,
            grad_steps_log_interval=grad_steps_log_interval,
            num_time_steps=self.num_time_steps,
            solver_kwargs=self.solver_kwargs,
            has_controls=self.data_manager.has_controls,
            generate_from_noise=self.generate_from_noise,
            noise_distribution=self.noise_distribution,
            device_id=self.device_id,
            num_samples_per_validation_step=num_samples_per_validation_step,
            cfg_prob_unconditional=self.cfg_prob_unconditional,
            validation_cfg_guidance_strength=self.validation_cfg_guidance_strength,
            num_grad_accumulation_steps=num_grad_accumulation_steps,
        )

        self.train_dataloader = TrainDataLoader(
            self.train_data,
            self.coupling,
            train_batch_size,
            state_transforms=self.state_transforms,
            device_id=self.device_id,
            has_controls=self.data_manager.has_controls,
        )

        self.validation_dataloader = None
        if self.validation_data is not None:
            self.validation_dataloader = ValidationDataLoader(
                self.validation_data,
                self.coupling,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
                has_controls=self.data_manager.has_controls,
            )

        self.trainer.fit(
            num_training_steps,
            self.train_dataloader,
            self.validation_dataloader,
            valid_freq,
            close_wandb_connection=close_wandb_connection
        )
