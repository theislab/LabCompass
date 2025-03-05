import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import torch
from anndata import AnnData
from torch import Tensor

from sc_exp_design.constants import DataFields
from sc_exp_design.couplings import Coupling, IndependentCoupling
from sc_exp_design.data import DataManager, TrainDataLoader, ValidationDataLoader
from sc_exp_design.flows import BaseFlow, ConstantNoiseFlow
from sc_exp_design.networks import NeuralVelocityField, NeuralVelocityFieldConfig
from sc_exp_design.ode import ODESolver
from sc_exp_design.training import CallBack, CFMTrainer
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["FlowMatching"]


class FlowMatching:
    """Initializes the :class:`FlowMatching` model.

    :param flow_class: The flow used to define the target dynamics. Should be a reference to
        a class derived from :class:`sc_exp_design.flows.BaseFlow` and not an instance.
        Such class has to provide the methods :method:`flow_class.compute_x_t`, :method:`flow_class.compute_u_t` and
        (optionally), :method:`flow_class.compute_score_t`, when we also want to lean the score
        (i.e.: :attr:`NeuralVelocityFieldConfig.learn_score_fiels` is `True`).
        Defaults to :class:`sc_exp_design.flows.ConstantNoiseFlow`.
    :type flow_class: class:`BaseFlow | None`

    :param flow_kwargs: Dictionary containing the keyword arguments to pass to :param:`flow_class` for its initialization.
        Refer to the :module:`sc_exp_design.flows` page for the available flows and their respective keyword arguments.
        Defaults to `None`.
    :type flow_kwargs: class:`dict[str, Any] | None`

    :param coupling_class: The coupling used to sample source and terminal states from the dataset. Should be a
        reference to a class derived from :class:`sc_exp_design.couplings.BaseCoupling` and not an instance.
        Such class has to provide the :method:`coupling_class.match_groups` that will be used by the dataloader
        to define the pairings during sampling. Defaults to  :class:`sc_exp_design.couplings.IndependentCoupling`
    :type coupling_class: class:`dict[str, Any] | Any`

    :param coupling_kwargs: Dictionary containing the keyword arguments to pass to :param:`coupling_class` for its initialization.
        Refer to the :module:`sc_exp_design.couplings` page for the available couplings and their respective keyword arguments.
        Defaults to `None`.
    :type coupling_kwargs: class:`dict[str, Any] | None`

    :param time_sampler: Function used to sample time steps during training, defaults to `torch.rand` (i.e.: Uniform sampling).
    :type time_sampler: class:`Callable[[Sequence[int], Any], Tensor]`

    :param device_id: The identifier for the device where to do the computations, defaults to `"cuda"`.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        flow_class: BaseFlow | None = None,
        flow_kwargs: dict[str, Any] | None = None,
        coupling_class: Coupling | None = None,
        coupling_kwargs: dict[str, Any] | None = None,
        time_sampler: Callable[[Sequence[int], Any], Tensor] = torch.rand,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        # initialize the Flow model 
        if flow_class is None:
            flow_class = ConstantNoiseFlow
        if flow_kwargs is None:
            flow_kwargs = {}
        self.flow = flow_class(**flow_kwargs)

        # initialize the coupling logic 
        if coupling_class is None:
            coupling_class = IndependentCoupling
        if coupling_kwargs is None:
            coupling_kwargs = {}
        self.coupling = coupling_class(**coupling_kwargs)

        self.time_sampler = time_sampler
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
        use_perturbation_target_repr: bool = False,
        perturbation_target_covariates: dict[str, Literal["one_hot", "label", "identity"]] | None = None,
        perturbation_target_covariates_in_obsm: dict[str, bool] | None = None,
        perturbation_target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Prepares the data for training and initializes the :attr:`FlowMatching.data_manager` and :attr:`FlowMatching.train_data` attributes of the model.

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

        :param use_perturbation_target_repr: Whether to use some target representation for the perturbations which to perform inference on, defaults t o `False`
        :type use_perturbation_target_repr: class:`bool`

        :param perturbation_target_covariates:
        :type perturbation_target_covariates_in_obsm: class:`dict[str, bool] | None`

        :param perturbation_target_covariates_in_obsm:
        :type perturbation_target_covariates_in_obsm: class:`dict[str, bool] | None`

        :param perturbation_target_covariates_kwargs:
        :type perturbation_target_covariates_kwargs: class `dict[str, Any] | None`
        """
        data_manager = DataManager(
            sample_rep=sample_rep,
            control_key=control_key,
            perturbations=perturbations,
            perturbation_covariates=perturbation_covariates,
            perturbation_reps=perturbation_reps,
            use_perturbation_target_repr=use_perturbation_target_repr,
            perturbation_target_covariates=perturbation_target_covariates,
            perturbation_target_covariates_in_obsm=perturbation_target_covariates_in_obsm,
            perturbation_target_covariates_kwargs=perturbation_target_covariates_kwargs,
        )
        train_data = data_manager.get_train_data(train_adata)

        self.data_manager = data_manager
        self.train_data = train_data

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
        flow_dim: int,
        cvf_config: NeuralVelocityFieldConfig,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        solver_class: ODESolver | None = ODESolver,
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

        :param solver_class: Reference to a solver used to integrate the dynamics during inference (not an instance). Should provide the method
            :method:`integrate`, needed to simulate the dynamics over time for a given initial condition and guidance term, defaults to :class:`ODESolver`.
        :type solver_class: class:`ODESolver`

        :param num_time_steps: Number of time steps which to integrate the dynamics over during inference, defaults to `100`.
        :type num_time_steps: class:`int`

        :param solver_kwargs: Dictionary containining the keyword arguments used to initialize the :param:`solver_class`, defaults to `None`.
        :type solver_kwargs: class:`dict[str, Any] | None`
        """
        self.flow_dim = flow_dim
        self.cvf_config = cvf_config
        
        # given a dimensionality and a configuration of hparams, initialize a flow model 
        self.velocity_field = NeuralVelocityField(
            self.flow_dim,
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

        self.solver_class = solver_class
        self.num_time_steps = num_time_steps
        self.solver_kwargs = solver_kwargs

    def train(
        self,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 100,
        posterior_on_cond_vars_update_step: int | None = None,
        posterior_on_perts_update_step: int | None = None,
        posterior_on_latent_perts_update_step: int | None = None,
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
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
        :type callbacks: class:`CallBack`

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

        self.trainer = CFMTrainer(
            self.velocity_field,
            self.flow,
            self.optimizer,
            lr_scheduler=self.lr_scheduler,
            lr_scheduler_step=self.lr_scheduler_step,
            time_sampler=self.time_sampler,
            callbacks=callbacks,
            grad_step_interval_log=grad_step_interval_log,
            solver_class=self.solver_class,
            num_time_steps=self.num_time_steps,
            gamma_fn=gamma_fn,
            solver_kwargs=self.solver_kwargs,
            posterior_on_cond_vars_update_step=posterior_on_cond_vars_update_step,
            posterior_on_perts_update_step=posterior_on_perts_update_step,
            posterior_on_latent_perts_update_step=posterior_on_latent_perts_update_step,
        )

        self.train_dataloader = TrainDataLoader(
            self.train_data,
            self.coupling,
            train_batch_size,
            state_transforms,
            self.device_id,
        )

        self.validation_dataloader = None
        if self.validation_data is not None:
            self.validation_dataloader = ValidationDataLoader(
                self.validation_data,
                self.coupling,
                validation_batch_size,
                state_transforms,
                self.device_id,
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
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
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
        source = batch[DataFields.SOURCE_STATE]
        
        condition = None
        if DataFields.PERTURBATION_DATA in batch.keys():
            condition = batch[DataFields.PERTURBATION_DATA]

        # defining velocity function
        vf = self.velocity_field.get_vf_fn(condition, gamma_fn=gamma_fn)
        # initializing the sampler clss
        ode_sampler = self.solver_class(
            vf,
            gamma_fn=gamma_fn,
            num_time_steps=self.num_time_steps,
            solver_kwargs=self.solver_kwargs,
            device_id=self.device_id,
        )
        with torch.no_grad():
            predictions = ode_sampler.integrate(source, return_trajectory=return_trajectory)
        return predictions
