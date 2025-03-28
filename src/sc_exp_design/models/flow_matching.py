import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from anndata import AnnData
from torch import Tensor

from sc_exp_design.constants import DataFields
from sc_exp_design.config.velocity_field import NeuralVelocityFieldConfig
from sc_exp_design.couplings import (
    IndependentCoupling,
    FixedCoupling,
    OTCoupling,
)
from sc_exp_design.data import DataManager, TrainDataLoader, ValidationDataLoader
from sc_exp_design.flows import (
    ConstantNoiseFlow,
    EncodingDecodingFlow,
    RectifiedFlow,
    VariancePreservingFlow,
)
from sc_exp_design.networks import NeuralVelocityField
from sc_exp_design.ode import ODESolver
from sc_exp_design.training import BaseCallBack, CFMTrainer
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["FlowMatching"]


class FlowMatching:
    """Initializes the :class:`FlowMatching` model.

    :param flow_type: The flow used to define the target dynamics. Should be a reference to
        a class derived from :class:`sc_exp_design.flows.BaseFlow` and not an instance.
        Such class has to provide the methods :method:`flow_type.compute_x_t`, :method:`flow_type.compute_u_t` and
        (optionally), :method:`flow_type.compute_score_t`, when we also want to lean the score
        (i.e.: :attr:`NeuralVelocityFieldConfig.learn_score_fiels` is `True`).
        Defaults to :class:`sc_exp_design.flows.ConstantNoiseFlow`.
    :type flow_type: class:`BaseFlow | None`

    :param flow_kwargs: Dictionary containing the keyword arguments to pass to :param:`flow_type` for its initialization.
        Refer to the :module:`sc_exp_design.flows` page for the available flows and their respective keyword arguments.
        Defaults to `None`.
    :type flow_kwargs: class:`dict[str, Any] | None`

    :param coupling_type: The coupling used to sample source and terminal states from the dataset. Should be a
        reference to a class derived from :class:`sc_exp_design.couplings.BaseCoupling` and not an instance.
        Such class has to provide the :method:`coupling_type.match_groups` that will be used by the dataloader
        to define the pairings during sampling. Defaults to  :class:`sc_exp_design.couplings.IndependentCoupling`
    :type coupling_type: class:`dict[str, Any] | Any`

    :param coupling_kwargs: Dictionary containing the keyword arguments to pass to :param:`coupling_type` for its initialization.
        Refer to the :module:`sc_exp_design.couplings` page for the available couplings and their respective keyword arguments.
        Defaults to `None`.
    :type coupling_kwargs: class:`dict[str, Any] | None`

    :param time_sampler: Function used to sample time steps during training, defaults to `torch.rand` (i.e.: Uniform sampling).
    :type time_sampler: class:`Callable[[Sequence[int], Any], Tensor]`

    :param device_id: The identifier for the device where to do the computations, defaults to `"cuda"`.
    :type device_id: class:`Literal["cuda", "cpu"]`
    
    :param generate_from_noise: Controls if the source samples are Gaussian (True) or control cells (False).
    :type num_training_steps: class:`bool`
    """

    def __init__(
        self,
        flow_type: Literal["constant_noise", "encoding_decoding", "rectified", "variance_preserving"] = "rectified",
        flow_kwargs: dict[str, Any] | None = None,
        coupling_type: Literal["fixed", "independent", "ot"] = "ot",
        coupling_kwargs: dict[str, Any] | None = None,
        time_sampler: Callable[[Sequence[int], Any], Tensor] = torch.rand,
        device_id: Literal["cuda", "cpu"] = "cuda",
        generate_from_noise: bool = False,
        noise_distribution: Callable[[Sequence[int]], Tensor] = torch.randn,
    ) -> None:
        # initialize the Flow model 
        if flow_type == "constant_noise":
            flow_class = ConstantNoiseFlow
        elif flow_type == "encoding_decoding":
            flow_class = EncodingDecodingFlow
        elif flow_type == "rectified":
            flow_class = RectifiedFlow
        elif flow_type == "variance_preserving":
            flow_class = VariancePreservingFlow
        else:
            msg = f""
            raise ValueError(msg)
        # setting optional flow kwargs
        if flow_kwargs is None:
            flow_kwargs = {}
        self.flow = flow_class(**flow_kwargs)

        # initialize the coupling logic 
        if coupling_type == "fixed":
            coupling_class = FixedCoupling
        elif coupling_type == "independent":
            coupling_class = IndependentCoupling
        elif coupling_type == "ot":
            coupling_class = OTCoupling
        else:
            msg = f""
            raise ValueError(msg)
        # setting optional coupling kwargs
        if coupling_kwargs is None:
            coupling_kwargs = {}
        self.coupling = coupling_class(**coupling_kwargs)

        self.time_sampler = time_sampler

        if generate_from_noise:
            msg = f""
            assert flow_type == "rectified", msg
        self.generate_from_noise = generate_from_noise
        self.noise_distribution = noise_distribution

        self.device_id = device_id
        self.device = torch.device(self.device_id)

        self.data_manager = None
        self.train_data = None
        self.validation_data = None

    def prepare_train_data(
        self,
        train_adata: AnnData,
        sample_rep: str | None = None,
        control_key: str | None = None,
        perturbations: str | Sequence[str] | None = None,
        perturbation_covariates: dict[str, str | Sequence[str]] | None = None,
        perturbation_reps: dict[str, str | Sequence[str]] | None = None,
        load_target_covariates: bool = False,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"]] | None = None,
        target_covariates_in_obsm: dict[str, bool] | None = None,
        target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        :param train_adata: An instance of :class:`anndata.AnnData` containing the training data.
        :type train_adata: class:`anndata.AnnData`

        :param sample_rep: A :class:`str` identifier indicating what key in :attr:`train_adata.layers` to look for the
            data representation. If `None`, it will use :attr:`train_adata.X`, defaults to `None`.
        :type sample_rep: class:`str | None`

        :param control_key: A :class:`str` identifier indicating what column in :attr:`train_adata.obs` to look for the
            binary variable representing whether an observation belongs to the control group or not, defaults to `None`.
        :type control_key: class:`str | None`

        :param perturbations: Either a :class:`str` or a :class:`Sequence[str]` indicating what column(s) in :attr:`train_adata.obs` to look
            for each modeled perturbation. Each perturbation should then have a key in either :param:`perturbation_covariates` or :param:`perturbation_reps`
            to be actually included in the data, otherwise a warning message is displayed and the passed perturbation is ignored, defaults to `None`.
        :type perturbations: class:`str | Sequence[str] | None`

        :param perturbation_covariates: A dictionary whose keys are :class:`str` identifiers for the modeled perturbations (i.e.: elements of :param:`perturbations`)
            and keys being either :class:`str` or :class:`Sequence[str]` indicating which keys in :param:`train_adata.obsm` where to look for the covariates
            associated to each perturbation. Such covariate will encode features of each perturbation that vary across the samples and thus might differ also
            across observation stimulated with the same perturbation, defaults to `None`.
        :param perturbation_covariates: class:`dict[str, str | Sequence[str]] | None`

        :param perturbation_reps: A dictionary whose keys are :class:`str` identifiers for the modeled perturbations (i.e.: elements of :param:`perturbations`)
            and keys being either :class:`str` or :class:`Sequence[str]` indicating which keys in :param:`train_adata.uns` where to look for the representation
            associated to each perturbation. Such representation will encode features of each perturbation that are constant across the samples (but do vary across
            the perturbations), defaults to `None`.
        :type perturbation_reps: class:`dict[str, str | Sequence[str]] | None`

        :param load_target_covariates: Whether to use some target representation for the perturbations which to perform inference on, defaults t o `False`
        :type load_target_covariates: class:`bool`

        :param target_covariates:
        :type target_covariates_in_obsm: class:`dict[str, bool] | None`

        :param target_covariates_in_obsm:
        :type target_covariates_in_obsm: class:`dict[str, bool] | None`

        :param target_covariates_kwargs:
        :type target_covariates_kwargs: class `dict[str, Any] | None`
        """
        has_controls = (control_key is not None)
        data_manager = DataManager(
            train_adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbations=perturbations,
            perturbation_covariates=perturbation_covariates,
            perturbation_reps=perturbation_reps,
            load_target_covariates=load_target_covariates,
            target_covariates=target_covariates,
            target_covariates_in_obsm=target_covariates_in_obsm,
            target_covariates_kwargs=target_covariates_kwargs,
            has_controls=has_controls,
        )
        train_data = data_manager.get_train_data(train_adata)

        self.data_manager = data_manager
        self.train_data = train_data
        self.has_controls = has_controls

    def prepare_validation_data(
        self,
        validation_adata: AnnData,
    ) -> None:
        """Prepares the data for validation and initializs the :attr:`FlowMatching.validation_data` attribute of the model.

        :param validation_adata: An instance of :class:`anndata.AnnData` containing the validation data.
        :type validation_adata: class:`anndata.AnnData`
        """
        validation_data = self.data_manager.get_train_data(validation_adata)
        self.validation_data = validation_data

    def prepare_model(
        self,
        cvf_config: NeuralVelocityFieldConfig,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        num_time_steps: int = 100,
        solver_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the model.

        :param flow_dim: The dimensionality of the flow.
        :type flow_dim: class:`int`

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
        if not self.has_controls:
            msg = f""
            assert not cvf_config.use_source_as_condition, msg
        else:
            if self.generate_from_noise:
                msg = f""
                assert cvf_config.use_source_as_condition, msg

        self.cvf_config = cvf_config
        
        # given a dimensionality and a configuration of hparams, initialize a flow model 
        self.velocity_field = NeuralVelocityField(
            config=self.cvf_config,
        )
        self.velocity_field = self.velocity_field.float()
        self.velocity_field = self.velocity_field.to(self.device)

        # optimizer and scheduler 
        self.optimizer = optimizer_class(
            self.velocity_field.parameters(),
            **optimizer_kwargs,
        )

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
        grad_step_interval_log: int = 100,
        num_treatments_to_load: int | None = None,
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

        :param posterior_on_cond_vars_update_step: The number of gradient steps after which to update the weights of
            the neural approximate posterior on the conditioning variables (endpoints), defaults to `None` (updating weights after each gradient step).
        :type posterior_on_cond_vars_update_step: class:`int | None`

        :param posterior_on_perts_update_step: The number of gradient steps after which to update the weights of
            the neural approximate posterior on the perturbation covariates, defaults to `None` (updating weights after each gradient step).
        :type posterior_on_perts_update_step: class:`int | None`

        :param posterior_on_latent_perts_update_step: The number of gradient steps after which to update the weights of
            the neural approximate posterior on the latent perturbation representation, defaults to `None` (updating weights after each gradient step).
        :type posterior_on_latent_perts_update_step: class:`int | None`

        :param gamma_fn: (Optional) function used to compute the diffusion coefficient in case we want to integrate the dynamics using an SDE and
            a drift adjusted by the score. In case :attr:`self.velocity_field.config.lean_score_field` is `False` it will be ignores, falling back to ODE sampling by
            default as from the original fromulation, defaults to `None`.
        :type gamma_fn: class:`Callable[[Tensor, Tensor], Tensor] | None`
        """
        # sanity checks
        msg = "Data not initialized, run `prepare_data` before training the model"
        assert self.train_data is not None, msg
        msg = "Model not initialized, run `prepare_model` before training the model"
        assert self.velocity_field is not None, msg

        # storing state transforms as attribute
        self.state_transforms = state_transforms

        self.trainer = CFMTrainer(
            self.velocity_field,
            self.flow,
            self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_step=self.lr_scheduler_step,
            time_sampler=self.time_sampler,
            callbacks=callbacks,
            grad_step_interval_log=grad_step_interval_log,
            num_time_steps=self.num_time_steps,
            solver_kwargs=self.solver_kwargs,
            has_controls=self.has_controls,
            generate_from_noise=self.generate_from_noise,
            noise_distribution=self.noise_distribution,
        )

        self.train_dataloader = TrainDataLoader(
            self.train_data,
            self.coupling,
            train_batch_size,
            state_transforms=self.state_transforms,
            device_id=self.device_id,
            has_controls=self.has_controls,
        )

        self.validation_dataloader = None
        if self.validation_data is not None:
            self.validation_dataloader = ValidationDataLoader(
                self.validation_data,
                self.coupling,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
                has_controls=self.has_controls,
                num_treatments_to_load=num_treatments_to_load
            )

        self.trainer.fit(
            num_training_steps,
            self.train_dataloader,
            self.validation_dataloader,
            valid_freq,
        )

    def predict(
        self,
        batch: dict[str, Tensor | dict[str, Tensor]],
        return_trajectory: bool = False,
        no_grad: bool = True,
        num_samples: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Tensor]:
        """Generates the predictions by integrating the dynamics with the learnt velocity field for a given initial condition

        :param batch: A batch of data containing both source point and conditions (guidance term).
        :type batch: class:`dict[str, Tensor | dict[str, Tensor]]`

        :param return_trajectory: Whether to return the whole trajectory at the given discretization points
        :type return_trajectory: class: `bool`

        :param gamma_fn: (Optional) function used to compute the diffusion coefficient in case we want to integrate the dynamics using an SDE and
            a drift adjusted by the score. In case :attr:`self.velocity_field.config.lean_score_field` is `False` it will be ignores, falling back to ODE sampling by
            default as from the original fromulation, defaults to `None`.
        :type gamma_fn: class:`Callable[[Tensor, Tensor], Tensor] | None`

        :return: Tensor of shape `(batch_size, self.flow_dim)` if :param:`return_trajectory` is `False`, otherwise Tensor of shape `(batch_size, self.num_time_steps, self.flow_dim)`
        :rtype: class:`torch.Tensor`
        """
        # handling source
        source = None
        if self.has_controls:
            source = batch[DataFields.SOURCE_STATE]

        # handling conditions
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]

        # handling batch size
        if source is not None:
            batch_size = source.shape[: -1]
        if batch_size is None:
            batch_size = (1, )
        
        if isinstance(batch_size, int):
            batch_size = (batch_size,)
        
        # handling number of samples
        if num_samples is not None:
            if not self.generate_from_noise:
                msg = f""
                logger.warning(msg)
                num_samples = 1
        else:
            num_samples = 1
        msg = f""
        assert isinstance(num_samples, int), msg
        if condition is not None:
            condition = {
                condition_covariate: condition_data.repeat(num_samples, *(1 for _ in condition_data.shape)).squeeze()
                for condition_covariate, condition_data in condition.items()
            }
        if source is not None:
            source = source.repeat(num_samples, *(1 for _ in source.shape)).squeeze()

        # handling latent state
        initial_state = source
        if self.generate_from_noise:
            initial_state = self.noise_distribution((num_samples, *batch_size, self.cvf_config.flow_dim)).squeeze().to(self.device)
        msg = f""
        assert initial_state is not None, msg

        # defining velocity function
        vf = self.velocity_field.get_vf_fn(condition, source=source)
        # initializing the sampler clss
        ode_solver = ODESolver(
            vf,
            num_time_steps=self.num_time_steps,
            solver_kwargs=self.solver_kwargs,
            device_id=self.device_id,
        )
        if no_grad:
            with torch.no_grad():
                predictions = ode_solver.integrate(initial_state, return_trajectory=return_trajectory)
        else:
            predictions = ode_solver.integrate(initial_state, return_trajectory=return_trajectory)
        return predictions
