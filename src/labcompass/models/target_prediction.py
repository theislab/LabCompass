import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch
from anndata import AnnData

from labcompass.data.dataloaders import SequentialDataLoader, SequentialValDataLoader
from labcompass.data.datamanager import DataManager
from labcompass.models.base import BaseModel
from labcompass.networks.inference_networks import PerturbationApproximatePosterior
from labcompass.training import BaseCallBack, TargetPredictionTrainer
from labcompass.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["TargetPredictionModel"]


class TargetPredictionModel(BaseModel):
    """Initializes the :class:`TargetPredictionModel`, a supervised model that predicts target covariates
    directly from observed cell states.

    :param device_id: The identifier for the device where to do the computations, defaults to `"cuda"`.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        self.device_id = device_id
        self.device = torch.device(self.device_id)
        self.target_prediction_model = None
        self.target_prediction_model_trained = False
        self.validation_data = None

    def prepare_train_data(
        self,
        train_adata: AnnData,
        sample_rep: str | None = None,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None = None,
        target_covariates_in_obsm: dict[str, bool] | None = None,
        target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Prepares the training data using the :class:`DataManager` object.

        Refer to the :class:`DataManager` documentation for an explanation of each argument. Once the
        :class:`DataManager` is initialized, it calls the :meth:`DataManager.get_data` method to retrieve a
        structured representation of the analyzed dataset, together with its target covariates.

        :param train_adata: An instance of :class:`AnnData` containing the training data.
        :type train_adata: class:`AnnData`

        :param sample_rep: Optional key in the `obsm` attribute of `train_adata` where the cell state
            representation is held. If `None`, it is retrieved directly from `train_adata.X`. Defaults to `None`.
        :type sample_rep: class:`str | None`

        :param target_covariates: Dictionary mapping each target covariate to the encoding used to represent it
            (`"one_hot"`, `"label"`, or `"identity"`), or `None` to use the default encoding. Defaults to `None`.
        :type target_covariates: class:`dict[str, Literal["one_hot", "label", "identity"] | None] | None`

        :param target_covariates_in_obsm: Dictionary indicating, for each target covariate, whether it should be
            retrieved from `train_adata.obsm` rather than `train_adata.obs`. Defaults to `None`.
        :type target_covariates_in_obsm: class:`dict[str, bool] | None`

        :param target_covariates_kwargs: Dictionary containing additional keyword arguments used to retrieve the
            desired representation for each target covariate. Defaults to `None`.
        :type target_covariates_kwargs: class:`dict[str, Any] | None`
        """
        data_manager = DataManager(
            train_adata,
            sample_rep=sample_rep,
            load_target_covariates=True,
            target_covariates=target_covariates,
            target_covariates_in_obsm=target_covariates_in_obsm,
            target_covariates_kwargs=target_covariates_kwargs,
        )
        train_data = data_manager.get_data(train_adata)

        self.data_manager = data_manager
        self.train_data = train_data

    def prepare_validation_data(
        self,
        validation_adata: AnnData,
    ) -> None:
        """Prepares the data for validation and initializs the :attr:`FlowMatching.validation_data` attribute of the model.

        :param validation_adata: An instance of :class:`AnnData` containing the validation data.
            It should satisfy the same requirements as the one used to construct the training data.
        :type validation_adata: class:`AnnData`
        """
        validation_data = self.data_manager.get_data(validation_adata)
        self.validation_data =  validation_data

    def prepare_model(
        self,
        target_covariates: str | Sequence[str],
        target_covariates_dims: int | dict[str, int],
        target_covariates_noise_models: Literal["gaussian"] | dict[str, None | Literal["gaussian"]] | None = None,
        target_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        target_covariates_use_shared_representation: bool = False,
        target_covariates_latent_dim: int = 1024,
        target_covariates_encoder_mlp_kwargs: dict[str, Any] | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
    ) -> None:
        """Initializes the target-covariate prediction model, together with its optimizer and optional learning
        rate scheduler.

        :param target_covariates: The identifier(s) of the target covariate(s) to predict. If a single
            :class:`str` is given, it is wrapped into a one-element tuple.
        :type target_covariates: class:`str | Sequence[str]`

        :param target_covariates_dims: The output dimensionality of the predictor for each target covariate. If
            an :class:`int` is given, `target_covariates` must contain a single element and the dimensionality is
            applied to it.
        :type target_covariates_dims: class:`int | dict[str, int]`

        :param target_covariates_noise_models: The noise model used for the predictive distribution of each
            target covariate, either `"gaussian"` or `None` for a deterministic output. If a single
            :class:`str` is given, `target_covariates` must contain a single element. Defaults to `None`, in
            which case every covariate is assigned `None`.
        :type target_covariates_noise_models: class:`Literal["gaussian"] | dict[str, None | Literal["gaussian"]] | None`

        :param target_covariates_predictor_kwargs: Dictionary mapping each target covariate to the keyword
            arguments used to initialize its predictor network. Defaults to `None`, in which case an empty
            dictionary is used for every covariate.
        :type target_covariates_predictor_kwargs: class:`dict[str, dict[str, Any]] | None`

        :param target_covariates_use_shared_representation: Whether the target covariate predictors share a
            common encoder representation of the input state, defaults to `False`.
        :type target_covariates_use_shared_representation: class:`bool`

        :param target_covariates_latent_dim: Dimensionality of the (optionally shared) latent representation used
            by the target covariate predictors, defaults to `1024`.
        :type target_covariates_latent_dim: class:`int`

        :param target_covariates_encoder_mlp_kwargs: Dictionary containing the keyword arguments used to
            initialize the encoder MLP producing the (optionally shared) latent representation, defaults to
            `None`.
        :type target_covariates_encoder_mlp_kwargs: class:`dict[str, Any] | None`

        :param optimizer_class: Optimizer used to update the target prediction model's weights during training.
            Should reference a class derived from :class:`torch.optim.Optimizer` and not an instance, defaults to
            :class:`torch.optim.AdamW`.
        :type optimizer_class: class:`torch.optim.Optimizer`

        :param optimizer_kwargs: Dictionary containing the keyword arguments used to initialize the
            `optimizer_class`, defaults to `{"lr": 0.001}`.
        :type optimizer_kwargs: class:`dict[str, Any]`

        :param lr_scheduler_class: Optional scheduler used to update the learning rate during optimization. Should
            reference a class derived from :class:`torch.optim.lr_scheduler.LRScheduler` and not an instance,
            defaults to `None`.
        :type lr_scheduler_class: class:`torch.optim.lr_scheduler.LRScheduler | None`

        :param lr_scheduler_kwargs: Dictionary containing the keyword arguments used to initialize the
            `lr_scheduler_class`, defaults to `None`.
        :type lr_scheduler_kwargs: class:`dict[str, Any] | None`

        :param lr_scheduler_step: :class:`str` identifier indicating when to perform the learning rate scheduling
            step, if a `lr_scheduler_class` is specified (otherwise it is ignored). When `"grad_step"`, the
            learning rate is updated after each gradient step; when `"epoch"`, it is updated after each
            validation step. Defaults to `"grad_step"`.
        :type lr_scheduler_step: class:`Literal["grad_step", "epoch"]`
        """
        # preparing input with some sanity checks
        if isinstance(target_covariates, str):
            target_covariates = (target_covariates, )

        if isinstance(target_covariates_dims, int):
            msg = f"When `target_covariates_dims` is of type `int`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_dims = {target_covariates[0]: target_covariates_dims}

        if isinstance(target_covariates_noise_models, str):
            msg = f"When `target_covariates_noise_models` is of type `str`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_noise_models = {target_covariates[0]: target_covariates_noise_models}
        if target_covariates_noise_models is None:
            target_covariates_noise_models = dict.fromkeys(target_covariates)

        if target_covariates_predictor_kwargs is None:
            target_covariates_predictor_kwargs = {
                target_covariate: {} for target_covariate in target_covariates
            }

        # checking types
        msg = ""
        assert isinstance(target_covariates, Sequence), msg

        msg = ""
        assert isinstance(target_covariates_dims, dict), msg

        msg = ""
        assert isinstance(target_covariates_noise_models, dict), msg

        msg = ""
        assert isinstance(target_covariates_predictor_kwargs, dict), msg

        # storing the settings here as attributes
        self.target_covariates = target_covariates
        self.target_covariates_dims = target_covariates_dims
        self.target_covariates_noise_models = target_covariates_noise_models
        self.target_covariates_predictor_kwargs = target_covariates_predictor_kwargs
        self.target_covariates_use_shared_representation = target_covariates_use_shared_representation
        self.target_covariates_latent_dim = target_covariates_latent_dim
        self.target_covariates_encoder_mlp_kwargs = target_covariates_encoder_mlp_kwargs

        # initializing the predictor for each target covariate
        self.target_prediction_model = PerturbationApproximatePosterior(
            self.state_dim,
            freeze_grads=False,
            target_output_dims=self.target_covariates_dims,
            noise_models=self.target_covariates_noise_models,
            covariate_kwargs=self.target_covariates_predictor_kwargs,
            use_shared_representation=self.target_covariates_use_shared_representation,
            latent_dim=self.target_covariates_latent_dim,
            encoder_mlp_kwargs=self.target_covariates_encoder_mlp_kwargs,
        )
        self.target_prediction_model = self.target_prediction_model.float()
        self.target_prediction_model = self.target_prediction_model.to(self.device)

        # optimizer and scheduler
        self.target_prediction_optimizer = optimizer_class(
            self.target_prediction_model.parameters(),
            **optimizer_kwargs,
        )

        self.target_prediction_lr_scheduler = None
        self.target_prediction_lr_scheduler_step = None
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler_class is not None:
            self.target_prediction_lr_scheduler = lr_scheduler_class(self.target_prediction_optimizer, **lr_scheduler_kwargs)
            self.target_prediction_lr_scheduler_step = lr_scheduler_step

    def train(
        self,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
        loss_fn_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Trains the target-covariate prediction model prepared by :meth:`prepare_model`.

        :param num_training_steps: The number of steps which to train the model on, defaults to `500`.
        :type num_training_steps: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step, in case
            :attr:`TargetPredictionModel.validation_data` was initialized by using
            :meth:`TargetPredictionModel.prepare_validation_data`, defaults to `None`.
        :type valid_freq: class:`int | None`

        :param train_batch_size: The batch size used for sampling the training data, defaults to `1024`.
        :type train_batch_size: class:`int`

        :param validation_batch_size: The batch size used for sampling the validation data, defaults to `512`.
        :type validation_batch_size: class:`int`

        :param state_transforms: (Optional) transformations applied to the states before feeding them into the
            model. Should be an instance of a class derived from :class:`labcompass.transforms.Transform`,
            defaults to `None`.
        :type state_transforms: class:`Transform`

        :param callbacks: (Optional) callbacks that will be called during training, defaults to `None`.
        :type callbacks: class:`BaseCallBack`

        :param grad_steps_log_interval: The number of gradient steps after which to update the progress bar,
            defaults to `100`.
        :type grad_steps_log_interval: class:`int`

        :param loss_fn_kwargs: Dictionary containing additional keyword arguments forwarded to the loss function
            used by the trainer, defaults to `None`.
        :type loss_fn_kwargs: class:`dict[str, Any] | None`
        """
        # sanity checks
        msg = "You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        # initializing data loader
        self.target_predictor_train_dataloader = SequentialDataLoader(
            self.train_data,
            train_batch_size,
            state_transforms=state_transforms,
            device_id=self.device_id
        )

        # initialize trainer
        self.target_predictor_trainer = TargetPredictionTrainer(
            self.target_prediction_model,
            self.target_prediction_optimizer,
            lr_scheduler=self.target_prediction_lr_scheduler,
            lr_scheduler_step=self.target_prediction_lr_scheduler_step,
            callbacks=callbacks,
            grad_steps_log_interval=grad_steps_log_interval,
            loss_fn_kwargs=loss_fn_kwargs,
        )

        # optional validation data
        self.target_predictor_validation_dataloader = None
        if self.validation_data is not None:
            self.target_predictor_validation_dataloader = SequentialValDataLoader(
                self.validation_data,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )

        # fitting the trainer
        self.target_predictor_trainer.fit(
            num_training_steps,
            self.target_predictor_train_dataloader,
            self.target_predictor_validation_dataloader,
            valid_freq,
        )

        self.target_prediction_model_trained = True

    def predict(
        self,
        control_states: torch.Tensor,
        no_grad: bool = True,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predicts the target covariates for the given cell states using the trained target prediction model.

        :param control_states: A tensor of cell states for which to predict the target covariates.
        :type control_states: class:`torch.Tensor`

        :param no_grad: Whether to disable gradient tracking while computing the predictions, defaults to `True`.
        :type no_grad: class:`bool`

        :return: Dictionary mapping each target covariate to its predicted representation.
        :rtype: class:`dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]`
        """
        self.target_prediction_model.eval()
        if no_grad:
            with torch.no_grad():
                return self.target_prediction_model(control_states)
        else:
            return self.target_prediction_model(control_states)

    @property
    def state_dim(self):
        """The dimensionality of the cell state representation used by the training data.

        :return: The size of the last dimension of `self.train_data.state_data`.
        :rtype: class:`int`
        """
        return self.train_data.state_data.shape[-1]
