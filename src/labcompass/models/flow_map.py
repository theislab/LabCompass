import logging
import os
from collections.abc import Callable, Sequence, Mapping
from typing import Any, Literal


import torch

from labcompass.constants import DataFields
from labcompass.data import TrainDataLoader, ValidationDataLoader
from labcompass.ode.utils import get_initial_state_and_condition
from labcompass.training import BaseCallBack
from labcompass.training.flow_map import FlowMapTrainer
from labcompass.transforms import Transform
from labcompass.config.flow_map import NeuralFlowMapConfig
from labcompass.models import FlowMatching
from labcompass.networks import NeuralVelocityField
from labcompass.networks.flow_map_net import NeuralFlowMap


class FlowMap(FlowMatching):

    def __init__(
        self,
        flow_type: Literal["constant_noise", "encoding_decoding", "rectified", "variance_preserving"] = "rectified",
        flow_kwargs: dict[str, Any] | None = None,
        coupling_type: Literal["independent", "ot"] = "ot",
        coupling_kwargs: dict[str, Any] | None = None,
        time_sampler: Callable[[Sequence[int], Any], torch.Tensor] | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
        generate_from_noise: bool = False,
        noise_distribution: Callable[[Sequence[int]], torch.Tensor] = torch.randn,
    ) -> None:
        if time_sampler is None:
            def time_sampler(shape, **kwargs):
                s = torch.rand(shape, **kwargs)
                t = torch.rand(shape, **kwargs)
                return s, t
        super().__init__(
            flow_type=flow_type,
            flow_kwargs=flow_kwargs,
            coupling_type=coupling_type,
            coupling_kwargs=coupling_kwargs,
            time_sampler=time_sampler,
            device_id=device_id,
            generate_from_noise=generate_from_noise,
            noise_distribution=noise_distribution,
        )

    def prepare_model(
        self,
        cvf_config: NeuralFlowMapConfig,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.0001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        num_time_steps: int = 100,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        :param cvf_config: Instance of :class:`labcompass.networks.NeuralVelocityFieldConfig` used to initialize the
            :class:`labcompass.networks.NeuralVelocityFIeld` object, then set as :attr:`FlowMatching.velocity_field` attribute.
        :type cvf_config: class: `labcompass.networks.NeuralVelocityFieldConfig`

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
        self.flow_map = NeuralFlowMap(
            config=self.cvf_config,
        )
        self.flow_map = self.flow_map.float()
        self.flow_map = self.flow_map.to(self.device)

        # optimizer and scheduler 
        self.optimizer = optimizer_class(
            self.flow_map.parameters(),
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
        velocity_field: NeuralVelocityField | None = None,
        weight_fn: None | Callable = lambda s, t: 1.0,
        sample_groups: bool = False,
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
            Should be an instance of a class derived from :class:`labcompass.transforms.Transform` and provide at least the method :method:`state_transforms.transform`
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
        assert self.flow_map is not None, msg
        if self.cvf_config.use_classifier_free_guidance:
            msg = "The probability of sampling the null condition token must be less than 1 for classifier-free guidance"
            assert cfg_prob_unconditional < 1, msg

        # storing state transforms as attribute
        self.state_transforms = state_transforms

        # storing cfg arguments as attributes
        self.cfg_prob_unconditional = cfg_prob_unconditional
        self.validation_cfg_guidance_strength = validation_cfg_guidance_strength

        self.trainer = FlowMapTrainer(
            self.flow_map,
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
            velocity_field=velocity_field,
            weight_fn=weight_fn,
        )

        self.train_dataloader = TrainDataLoader(
            self.train_data,
            self.coupling,
            train_batch_size,
            state_transforms=self.state_transforms,
            device_id=self.device_id,
            has_controls=self.data_manager.has_controls,
            sample_groups=sample_groups,
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

    def predict(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        time_steps: torch.Tensor | None = None,
        num_steps: int = 2,
        return_trajectory: bool = False,
        no_grad: bool = True,
        num_samples: int | None = None,
        batch_size: int | None = None,
        num_time_steps: int | None = None,
        fix_noise: bool = False, 
    ) -> dict[str, torch.Tensor]:
        """Generates the predictions by integrating the dynamics with the learnt velocity field for a given initial condition

        :param batch: A batch of data containing both source point and conditions (guidance term).
        :type batch: class:`dict[str, Tensor | dict[str, Tensor]]`

        :param return_trajectory: Whether to return the whole trajectory at the given discretization points
        :type return_trajectory: class: `bool`

        :param num_samples: Specifies the number of samples for each observation to be generated.
            Only used when :attr: `self.generate_from_noise` is set to `True`, defaults to `None` in which case only one sample will be generated.
        :type num_samples : class: `int | None`

        :param batch_size: Specifies the number of observations to sample. Only used when :attr: `self.generate_from_noise` is `True` and neither
            source states nor conditions are passed in the :param: `batch`. In such cases, when :param: `batch_size` is not provided,
            it will be inferred from :attr: `self.validation_dataloader.batch_size` if present, otherwise from :attr: `self.training_dataloader.batch_size`
            Only used when :attr: `self.generate_from_noise` is set to `True`, defaults to `None` in which case only one sample will be generated.
        :type batch_size : class: `int | None`

        :param num_time_steps: Number of time steps which to integrate the dynamics over during inference.
            If provided, it will be used instead of :attr: `self.num_time_steps` . Defaults to `None`.
        :type num_time_steps: class:`int | None`
        
        :param fix_noise: Whether the noise  for the prediction is fixed and present in the batch as a source.
        :type fix_noise: class: `bool`

        :param solver_kwargs: Dictionary containining the keyword arguments used to initialize the :param:`solver_class`.
            If provided, it will be used instead of :attr: `self.solver_kwargs` . Defaults to `None`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param cfg_guidance_strength: Strength of the guidance term for sampling.
            Only used when :attr: `self.cvf_config.use_classifier_free_guidance` is set to `True`, defaults to `1.0`.
        :type cfg_guidance_strength: class: `float`
        
        :return: Tensor of shape `(batch_size, self.flow_dim)` if :param:`return_trajectory` is `False`, otherwise Tensor of shape `(batch_size, self.num_time_steps, self.flow_dim)`
        :rtype: class:`torch.Tensor`
        """
        # handling source
        source = None
        if self.data_manager.has_controls or fix_noise: 
            source = batch[DataFields.SOURCE_STATE]

        # handling conditions
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]

        # handling batch size
        if self.generate_from_noise:
            # inferring the batch size
            # first we will try from the data passed in the batch
            # when we have both source and condition we need to verify that
            # their first dimension coincides
            if (source is not None) and (condition is not None):
                # sanity check
                for condition_covariate, condition_data in condition.items():
                    msg = f""
                    assert condition_data.shape[:-1] == source.shape[:-1], msg
                batch_size = source.shape[:-1]
            # when we only have the source states (unconditional generation)
            # simply take its first dimension
            elif (source is not None):
                batch_size = source.shape[:-1]
            # when we only have the condition (no notion of control states)
            # we need to check that they all share the same batch size
            elif (condition is not None):
                ref_batch_size = list(condition.values())[0].shape[:-1]
                # sanity check
                for condition_covariate, condition_data in condition.items():
                    msg = f""
                    assert condition_data.shape[:-1] == ref_batch_size, msg
                batch_size = ref_batch_size
            # otherwise we retrieve it from the dataloaders
            if batch_size is None:
                # if it exists, infer it from validation dataloader
                # otherwise uses the train dataloader.
                if self.validation_dataloader is not None:
                    batch_size = self.validation_dataloader.batch_size
                else:
                    batch_size = self.train_dataloader.batch_size

        # handling discretization time steps
        if num_time_steps is None:
            num_time_steps = self.num_time_steps

        # pushing forward particles
        initial_state, condition = get_initial_state_and_condition(
            source,
            batch_size,
            num_samples,
            self.flow_map.config.flow_dim,
            condition,
            self.noise_distribution,
            self.device_id,
            self.generate_from_noise,
        )

        # get map fn
        map_fn = self.flow_map.get_map_fn(
            condition,
            source=source
        )

        # prepare time steps
        if time_steps is None:
            time_steps = torch.linspace(0.0, 1.0, num_steps+1)
        
        X_s = initial_state
        traj = [X_s]
        for idx, s in enumerate(time_steps[:-1]):
            t = time_steps[idx + 1]
            s_tensor = torch.ones([*initial_state.shape[:-1]], device=self.device).float()*s
            t_tensor = torch.ones([*initial_state.shape[:-1]], device=self.device).float()*t
            X_s = map_fn(s_tensor, t_tensor, X_s)
            traj.append(X_s)
        if return_trajectory:
            traj = torch.stack(traj, axis=0)    
            return traj
        return X_s
