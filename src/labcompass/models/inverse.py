import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from labcompass.data.dataloaders import AnnotatedPerturbationData, SequentialDataLoader
from labcompass.models.base import BaseModel
from labcompass.models.inverse_utils import LangevinOptimizer
from labcompass.networks.blocks import BaseForwardModel, BaseModule
from labcompass.networks.inference_networks import PerturbationApproximatePosterior
from labcompass.networks.inverse import (
    LangevinSampler,
    MAPConditionOptimizer,
    NeuralInverseModel,
)
from labcompass.training import BaseCallBack, InverseModelTrainer, TargetPredictionTrainer
from labcompass.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["InverseModel"]


class InverseModel(BaseModel):
    """Initializes the :class:`InverseModel`, which solves the inverse problem of inferring the perturbation
    covariates that would drive a control cell state towards a desired target cell state.

    :param forward_model: A trained forward model (typically a fitted :class:`FlowMatching` instance) used to map
        control states and perturbation covariates to predicted post-perturbation states. It is expected to expose
        a `.predict()` method compatible with :class:`labcompass.networks.blocks.BaseForwardModel`, together with
        `cvf_config` and `train_data` attributes. Defaults to `None`.
    :type forward_model: class:`BaseForwardModel | None`

    :param state_dim: Dimensionality of the cell state space the inverse problem operates in. If `None`, it is
        inferred from `forward_model.cvf_config.flow_dim`. If both `state_dim` and `forward_model` are provided and
        disagree, a warning is logged and the value is overridden with `forward_model.cvf_config.flow_dim`.
        Defaults to `None`.
    :type state_dim: class:`int | None`

    :param inverse_method: String identifier selecting the algorithm used to solve the inverse problem: `"map"`
        uses :class:`labcompass.networks.inverse.MAPConditionOptimizer` to obtain a point estimate of the
        perturbation covariates via gradient-based MAP optimization, `"langevin"` uses
        :class:`labcompass.networks.inverse.LangevinSampler` to draw posterior samples of the perturbation
        covariates via Langevin dynamics (see :class:`labcompass.models.inverse_utils.LangevinOptimizer`), and
        `"neural"` uses :class:`labcompass.networks.inverse.NeuralInverseModel` to amortize inference with a
        neural network that predicts the perturbation covariates directly. Defaults to `"map"`.
    :type inverse_method: class:`Literal["map", "langevin", "neural"]`

    :param device_id: The identifier for the device where to do the computations, defaults to `"cuda"`.
    :type device_id: class:`Literal["cuda", "cpu"]`
    """

    def __init__(
        self,
        forward_model: BaseForwardModel | None = None,
        state_dim: int | None = None,
        inverse_method: Literal["map", "langevin", "neural"] = "map",
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        # sanity check on the input
        if state_dim is None:
            msg = ""
            assert forward_model is not None, msg
            state_dim = forward_model.cvf_config.flow_dim
        else:
            msg = ""
            assert isinstance(state_dim, int), msg
            # if we pass the forward model we take the dimensionality from there
            if (forward_model is not None) and state_dim != forward_model.cvf_config.flow_dim:
                msg = ""
                logger.warning(msg)
                state_dim = forward_model.cvf_config.flow_dim

        if inverse_method == "map":
            inverse_method_class = MAPConditionOptimizer
        elif inverse_method == "langevin":
            inverse_method_class = LangevinSampler
        elif inverse_method == "neural":
            inverse_method_class = NeuralInverseModel
        else:
            msg = ""
            raise ValueError(msg)

        self.forward_model = forward_model
        self.state_dim = state_dim

        self.inverse_method = inverse_method
        self.inverse_method_class = inverse_method_class

        self.device_id = device_id
        self.device = torch.device(self.device_id)

        self.target_prediction_model = None
        self.target_prediction_model_trained = False
        self.inverse_model = None

    def prepare_target_prediction_model(
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
        """Initializes the target-covariate prediction model, used to predict the target covariates from the
        states predicted by the forward model, together with its optimizer and optional learning rate scheduler.

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
            freeze_grads=False, # we want to backpropagate the gradients from its input
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

    def train_target_prediction_model(
        self,
        train_data: AnnotatedPerturbationData | None = None,
        validation_data: AnnotatedPerturbationData | None = None,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
    ) -> None:
        """Trains the target-covariate prediction model prepared by :meth:`prepare_target_prediction_model`.

        :param train_data: The data used to train the target prediction model. Must be an instance of
            :class:`AnnotatedPerturbationData` with `target_reprs` set. Defaults to `None`, in which case
            `self.forward_model.train_data` is used.
        :type train_data: class:`AnnotatedPerturbationData | None`

        :param validation_data: (Optional) data used to validate the target prediction model during training,
            with the same requirements as `train_data`. Defaults to `None`.
        :type validation_data: class:`AnnotatedPerturbationData | None`

        :param num_training_steps: The number of steps which to train the model on, defaults to `500`.
        :type num_training_steps: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step, only used when
            `validation_data` is provided, defaults to `None`.
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
        """
        # sanity checks
        msg = "You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        if train_data is None:
            msg = ""
            assert self.forward_model is not None, msg
            train_data = self.forward_model.train_data

        msg = ""
        assert isinstance(train_data, AnnotatedPerturbationData), msg

        msg = ""
        assert train_data.target_reprs is not None, msg

        # initializing data loader
        self.target_predictor_train_data = train_data
        self.target_predictor_train_dataloader = SequentialDataLoader(
            self.target_predictor_train_data,
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
        )

        # optional validation data
        self.target_predictor_validation_dataloader = None
        self.target_predictor_validation_data = validation_data
        if validation_data is not None:

            self.target_predictor_validation_dataloader = SequentialDataLoader(
                self.target_predictor_validation_data,
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

    def attach_target_prediction_model(
        self,
        other,
    ) -> None:
        """
        Copies all attributes from another InverseModel instance.
        """
        if not isinstance(other, InverseModel):
            msg = "Expected an instance of InverseModel"
            raise TypeError(msg)

        msg = ""
        assert other.target_prediction_model_trained, msg

        # Copy attributes set in prepare_target_prediction_model
        self.target_covariates = other.target_covariates
        self.target_covariates_dims = other.target_covariates_dims
        self.target_covariates_noise_models = other.target_covariates_noise_models
        self.target_covariates_predictor_kwargs = other.target_covariates_predictor_kwargs
        self.target_prediction_model = other.target_prediction_model
        self.target_prediction_optimizer = other.target_prediction_optimizer
        self.target_prediction_lr_scheduler = other.target_prediction_lr_scheduler
        self.target_prediction_lr_scheduler_step = other.target_prediction_lr_scheduler_step

        # Copy attributes set in train_target_prediction_model
        self.target_predictor_train_data = other.target_predictor_train_data
        self.target_predictor_train_dataloader = other.target_predictor_train_dataloader
        self.target_predictor_trainer = other.target_predictor_trainer
        self.target_predictor_validation_dataloader = other.target_predictor_validation_dataloader
        self.target_predictor_validation_data = other.target_predictor_validation_data
        self.target_prediction_model_trained = other.target_prediction_model_trained

    def prepare_inverse_model(
        self,
        optimal_condition: torch.Tensor | dict[str, torch.Tensor],
        loss_fn: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_covariates: str | Sequence[str],
        perturbation_covariates_dims: int | dict[str, int],
        is_discrete_dict: bool | dict[str, bool] | None = None,
        forward_model: BaseForwardModel | None = None,
        target_prediction_model: BaseModule | None = None,
        prior: dict[str, torch.distributions.Distribution] | None = None,
        prior_weight: float | None = None,
        hard: bool = False,
        perturbation_initializer: Callable[[Any], torch.Tensor] | dict[str, Callable[[Any], torch.Tensor]] | None = None,
        perturbation_non_linearities: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor] | dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] | None = None,
        perturbation_covariates_noise_models: Literal["gaussian"] | dict[str, None | Literal["gaussian"]] | None = None,
        perturbation_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        perturbation_covariates_use_shared_representation: bool = False,
        perturbation_covariates_latent_dim: int = 1024,
        perturbation_encoder_mlp_kwargs: dict[str, Any] | None = None,
        n_samples: int | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        **kwargs,
    ) -> None:
        """Initializes the inverse model (an instance of :class:`MAPConditionOptimizer`,
        :class:`LangevinSampler`, or :class:`NeuralInverseModel`, depending on `self.inverse_method`) that infers
        the perturbation covariates driving control cell states towards `optimal_condition`, together with its
        optimizer and optional learning rate scheduler.

        :param optimal_condition: The desired value(s) of the target covariate(s) that the inferred perturbation
            should produce. If a single :class:`torch.Tensor` is given, `self.target_covariates` must contain a
            single element and the tensor is associated with it.
        :type optimal_condition: class:`torch.Tensor | dict[str, torch.Tensor]`

        :param loss_fn: The loss function(s) comparing the target covariate(s) predicted from the perturbed state
            against `optimal_condition`. If a single :class:`Callable` is given, `self.target_covariates` must
            contain a single element and the function is associated with it.
        :type loss_fn: class:`dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`

        :param perturbation_covariates: The identifier(s) of the perturbation covariate(s) to infer.
        :type perturbation_covariates: class:`str | Sequence[str]`

        :param perturbation_covariates_dims: The dimensionality of the representation of each perturbation
            covariate. If an :class:`int` is given, `perturbation_covariates` must contain a single element.
        :type perturbation_covariates_dims: class:`int | dict[str, int]`

        :param is_discrete_dict: Whether each perturbation covariate is discrete (in which case it is optimized
            through a differentiable Gumbel-softmax relaxation) or continuous. If a single :class:`bool` is
            given, `perturbation_covariates` must contain a single element. Defaults to `None`, in which case
            every covariate is treated as continuous (`False`) and a warning is logged.
        :type is_discrete_dict: class:`bool | dict[str, bool] | None`

        :param forward_model: The forward model used to predict the perturbed cell states, overriding
            `self.forward_model` when provided. Defaults to `None`, in which case `self.forward_model` is used.
        :type forward_model: class:`BaseForwardModel | None`

        :param target_prediction_model: The model used to predict the target covariates from the perturbed cell
            states, overriding `self.target_prediction_model` when provided. Defaults to `None`, in which case
            `self.target_prediction_model` is used, which requires it to have already been trained via
            :meth:`train_target_prediction_model`.
        :type target_prediction_model: class:`BaseModule | None`

        :param prior: (Optional) dictionary mapping each perturbation covariate to a prior distribution (any
            object exposing a `log_prob` method, e.g. a :class:`torch.distributions.Distribution`) over it, used
            to add a negative log-prior regularization term to the loss. Defaults to `None`.
        :type prior: class:`dict[str, torch.distributions.Distribution] | None`

        :param prior_weight: The weight of the negative log-prior term in the loss, only used when `prior` is
            provided. Defaults to `None`, in which case it is set to `1.0` and a warning is logged.
        :type prior_weight: class:`float | None`

        :param hard: Whether discrete perturbation covariates use the hard (straight-through) Gumbel-softmax
            estimator rather than the soft relaxation, defaults to `False`.
        :type hard: class:`bool`

        :param perturbation_initializer: Function(s) used to initialize the optimized perturbation covariate
            tensor(s). If a single :class:`Callable` is given, `perturbation_covariates` must contain a single
            element. Defaults to `None`, in which case `torch.randn` is used for every covariate and a warning is
            logged.
        :type perturbation_initializer: class:`Callable[[Any], torch.Tensor] | dict[str, Callable[[Any], torch.Tensor]] | None`

        :param perturbation_non_linearities: Non-linearity(ies) applied to each continuous perturbation covariate
            before it is fed to the forward model. If a single :class:`torch.nn.Module` or :class:`Callable` is
            given, `perturbation_covariates` must contain a single element. Defaults to `None`, in which case
            :class:`torch.nn.Identity` is used for every covariate and a warning is logged.
        :type perturbation_non_linearities: class:`torch.nn.Module | Callable[[torch.Tensor], torch.Tensor] | dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] | None`

        :param perturbation_covariates_noise_models: The noise model used for the predictive distribution of each
            perturbation covariate, only used when `self.inverse_method` is `"neural"`. If a single :class:`str`
            is given, `perturbation_covariates` must contain a single element. Defaults to `None`, in which case
            every covariate is assigned `None` and a warning is logged.
        :type perturbation_covariates_noise_models: class:`Literal["gaussian"] | dict[str, None | Literal["gaussian"]] | None`

        :param perturbation_covariates_predictor_kwargs: Dictionary mapping each perturbation covariate to the
            keyword arguments used to initialize its predictor network, only used when `self.inverse_method` is
            `"neural"`. Defaults to `None`, in which case an empty dictionary is used for every covariate and a
            warning is logged.
        :type perturbation_covariates_predictor_kwargs: class:`dict[str, dict[str, Any]] | None`

        :param perturbation_covariates_use_shared_representation: Whether the perturbation covariate predictors
            share a common encoder representation of the input, only used when `self.inverse_method` is
            `"neural"`, defaults to `False`.
        :type perturbation_covariates_use_shared_representation: class:`bool`

        :param perturbation_covariates_latent_dim: Dimensionality of the (optionally shared) latent
            representation used by the perturbation covariate predictors, only used when `self.inverse_method` is
            `"neural"`, defaults to `1024`.
        :type perturbation_covariates_latent_dim: class:`int`

        :param perturbation_encoder_mlp_kwargs: Dictionary containing the keyword arguments used to initialize the
            encoder MLP producing the (optionally shared) latent representation, only used when
            `self.inverse_method` is `"neural"`. Defaults to `None`, in which case an empty dictionary is used and
            a warning is logged.
        :type perturbation_encoder_mlp_kwargs: class:`dict[str, Any] | None`

        :param n_samples: The number of perturbation covariate samples drawn per control state, only used when
            `self.inverse_method` is `"langevin"`. Defaults to `None`, in which case it is set to `1` and a
            warning is logged.
        :type n_samples: class:`int | None`

        :param optimizer_class: Optimizer used to update the inverse model's parameters during training. Should
            reference a class derived from :class:`torch.optim.Optimizer` and not an instance, defaults to
            :class:`torch.optim.AdamW`. When `self.inverse_method` is `"langevin"`, it is forced to
            :class:`labcompass.models.inverse_utils.LangevinOptimizer` (with a warning if a different class was
            passed), and its keyword arguments are instead built from the `eta` and `noise_scale` entries of
            `kwargs`.
        :type optimizer_class: class:`torch.optim.Optimizer`

        :param optimizer_kwargs: Dictionary containing the keyword arguments used to initialize `optimizer_class`,
            defaults to `{"lr": 0.001}`. Ignored when `self.inverse_method` is `"langevin"`.
        :type optimizer_kwargs: class:`dict[str, Any]`

        :param lr_scheduler_class: Optional scheduler used to update the learning rate during optimization. Should
            reference a class derived from :class:`torch.optim.lr_scheduler.LRScheduler` and not an instance,
            defaults to `None`. Not supported when `self.inverse_method` is `"langevin"`, in which case it is
            forced back to `None` and a warning is logged if provided.
        :type lr_scheduler_class: class:`torch.optim.lr_scheduler.LRScheduler | None`

        :param lr_scheduler_kwargs: Dictionary containing the keyword arguments used to initialize
            `lr_scheduler_class`, defaults to `None`.
        :type lr_scheduler_kwargs: class:`dict[str, Any] | None`

        :param lr_scheduler_step: :class:`str` identifier indicating when to perform the learning rate scheduling
            step, if a `lr_scheduler_class` is specified (otherwise it is ignored). When `"grad_step"`, the
            learning rate is updated after each gradient step; when `"epoch"`, it is updated after each
            validation step. Defaults to `"grad_step"`.
        :type lr_scheduler_step: class:`Literal["grad_step", "epoch"]`

        :param kwargs: Additional keyword arguments forwarded to the underlying inverse-method class
            (:class:`MAPConditionOptimizer`, :class:`LangevinSampler`, or :class:`NeuralInverseModel`). When
            `self.inverse_method` is `"langevin"`, the `eta` and `noise_scale` entries (if present) are also used
            to build `optimizer_kwargs`.
        :type kwargs: class:`Any`
        """
        # we need to have at least one trained target predictor
        if target_prediction_model is None:
            msg = "You need to have trained the target predictor model by calling `train_target_prediction_model`."
            assert self.target_prediction_model_trained, msg
            target_prediction_model = self.target_prediction_model

        # we need to have at least one forward model
        if forward_model is None:
            msg = ""
            assert self.forward_model is not None, msg
            forward_model = self.forward_model

        # preparing input with some sanity checks
        if isinstance(optimal_condition, torch.Tensor):
            msg = ""
            assert len(self.target_covariates) == 1, msg
            optimal_condition = {self.target_covariates[0]: optimal_condition}

        if isinstance(loss_fn, Callable):
            msg = ""
            assert len(self.target_covariates) == 1, msg
            loss_fn = {self.target_covariates[0]: loss_fn}

        if isinstance(perturbation_covariates, str):
            perturbation_covariates = (perturbation_covariates, )

        if isinstance(perturbation_covariates_dims, int):
            msg = f"When `perturbation_covariates_dims` is of type `int`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_covariates_dims = {perturbation_covariates[0]: perturbation_covariates_dims}

        if isinstance(is_discrete_dict, bool):
            msg = f"When `is_discrete_dict` is of type `bool`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            is_discrete_dict = {perturbation_covariates[0]: is_discrete_dict}
        if is_discrete_dict is None:
            msg = ""
            logger.warning(msg)
            is_discrete_dict = dict.fromkeys(perturbation_covariates, False)

        if isinstance(perturbation_initializer, Callable):
            msg = f"When `perturbation_initializer` is of type `Callable`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_initializer = {perturbation_covariates[0]: perturbation_initializer}
        if perturbation_initializer is None:
            msg = "No initialization passed, setting to normal by default."
            logger.warning(msg)
            perturbation_initializer = dict.fromkeys(perturbation_covariates, torch.randn)

        if isinstance(perturbation_non_linearities, Callable | torch.nn.Module):
            msg = f"When `perturbation_non_linearities` is of type `Callable | torch.nn.Module`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_non_linearities = {perturbation_covariates[0]: perturbation_non_linearities}
        if perturbation_non_linearities is None:
            msg = "No non-linearity passed, setting to identity by default."
            logger.warning(msg)
            perturbation_non_linearities = {perturbation_covariate: torch.nn.Identity() for perturbation_covariate in perturbation_covariates}

        if isinstance(perturbation_covariates_noise_models, str):
            msg = f"When `perturbation_non_linearities` is of type `str`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_covariates_noise_models = {perturbation_covariates[0]: perturbation_covariates_noise_models}
        if perturbation_covariates_noise_models is None:
            msg = ""
            logger.warning(msg)
            perturbation_covariates_noise_models = dict.fromkeys(perturbation_covariates)

        if perturbation_covariates_predictor_kwargs is None:
            msg = ""
            logger.warning(msg)
            perturbation_covariates_predictor_kwargs = {
                covariate: {} for covariate in perturbation_covariates
            }

        if perturbation_encoder_mlp_kwargs is None:
            msg = ""
            logger.warning(msg)
            perturbation_encoder_mlp_kwargs = {}

        # check types
        msg = f"`perturbation_covariates` nees to be a sequence of perturbation covatiate identifiers, found {type(perturbation_covariates)}"
        assert isinstance(perturbation_covariates, Sequence), msg

        msg = f"`perturbation_covariates_dims` needs to be a dictionary mapping each condition to its dimensionality, found {type(perturbation_covariates_dims)}"
        assert isinstance(perturbation_covariates_dims, dict), msg

        msg = ""
        assert isinstance(is_discrete_dict, dict), msg

        msg = ""
        assert isinstance(perturbation_initializer, dict), msg

        msg = ""
        assert isinstance(perturbation_non_linearities, dict), msg

        msg = ""
        assert isinstance(perturbation_covariates_noise_models, dict), msg

        msg = ""
        assert isinstance(perturbation_covariates_predictor_kwargs, dict), msg

        # we want all these dictionaries to share the same keys (i.e.: covariate ids) found in perturbation_covariates
        for perturbation_key in perturbation_covariates:

            msg = f"{perturbation_key=} not found in `perturbation_covariates_dims.keys()`, you need to specify a corresponding dimensionality."
            assert perturbation_key in perturbation_covariates_dims.keys(), msg

            msg = ""
            assert perturbation_key in is_discrete_dict.keys(), msg

            # when initializer is not passed for a covariate we set it to normal initialization
            msg = ""
            assert perturbation_key in perturbation_initializer.keys(), msg
            if perturbation_initializer[perturbation_key] is None:
                perturbation_initializer[perturbation_key] = torch.randn

            # when non linearity is not passed for a covariate we set it to identity
            msg = ""
            assert perturbation_key in perturbation_non_linearities.keys(), msg
            if perturbation_non_linearities[perturbation_key] is None:
                perturbation_non_linearities[perturbation_key] = torch.nn.Identity()

            msg = ""
            assert perturbation_key in perturbation_covariates_noise_models, msg

            # when non keyword settings are not passed for a covariate we set it to empty dictionary
            msg = ""
            assert perturbation_key in perturbation_covariates_predictor_kwargs, msg
            if perturbation_covariates_predictor_kwargs[perturbation_key] is None:
                perturbation_covariates_predictor_kwargs[perturbation_key] = {}

        # when we pass the prior on the perturbations
        if prior is not None:
            if prior_weight is None:
                msg = "`prior` was passed, but no `prior_weight` was given. Setting to 1.0 by default."
                logger.warning(msg)
                prior_weight = 1.0

        # when we use langevin we need to pass the number of samples
        if n_samples is None and self.inverse_method == "langevin":
            msg = ""
            logger.warning(msg)
            n_samples = 1

        # storing the attributes here
        self.optimal_condition = optimal_condition
        self.loss_fn = loss_fn
        self.perturbation_covariates = perturbation_covariates
        self.perturbation_covariates_dims = perturbation_covariates_dims
        self.is_discrete_dict = is_discrete_dict
        self.prior = prior
        self.prior_weight = prior_weight
        self.hard = hard
        self.perturbation_initializer = perturbation_initializer
        self.perturbation_non_linearities = perturbation_non_linearities
        self.perturbation_covariates_noise_models = perturbation_covariates_noise_models
        self.perturbation_covariates_predictor_kwargs = perturbation_covariates_predictor_kwargs
        self.perturbation_covariates_use_shared_representation = perturbation_covariates_use_shared_representation
        self.perturbation_covariates_latent_dim = perturbation_covariates_latent_dim
        self.perturbation_encoder_mlp_kwargs = perturbation_encoder_mlp_kwargs
        self.n_samples = n_samples

        # initializing the inverse model
        self.inverse_model = self.inverse_method_class(
            self.optimal_condition,
            self.target_prediction_model,
            forward_model,
            self.loss_fn,
            self.perturbation_covariates,
            self.perturbation_covariates_dims,
            self.is_discrete_dict,
            prior=self.prior,
            prior_weight=self.prior_weight,
            hard=self.hard,
            perturbation_initializer=self.perturbation_initializer,
            perturbation_non_linearities=self.perturbation_non_linearities,
            perturbation_covariates_noise_models=self.perturbation_covariates_noise_models,
            perturbation_covariates_predictor_kwargs=self.perturbation_covariates_predictor_kwargs,
            perturbation_covariates_use_shared_representation=self.perturbation_covariates_use_shared_representation,
            perturbation_covariates_latent_dim=self.perturbation_covariates_latent_dim,
            perturbation_encoder_mlp_kwargs=self.perturbation_encoder_mlp_kwargs,
            n_samples=self.n_samples,
            state_dim=self.state_dim,
            **kwargs
        )
        self.inverse_model = self.inverse_model.float()
        self.inverse_model = self.inverse_model.to(self.device)

        # optimizer and scheduler
        if self.inverse_method == "langevin":
            if optimizer_class is not LangevinOptimizer:
                msg = f"When {self.inverse_method=}, you need to pass `LangevinOptimizer` as the `optimizer_class` argument. Setting it for you."
                logger.warning(msg)
                optimizer_class = LangevinOptimizer
            # preparing optimizer keyword arguments
            optimizer_kwargs = {}
            if "eta" in kwargs.keys():
                optimizer_kwargs["eta"] = kwargs["eta"]
            if "noise_scale" in kwargs.keys():
                optimizer_kwargs["noise_scale"] = kwargs["noise_scale"]
            # no scheduler when using langevin
            if lr_scheduler_class is not None:
                msg = f"With {self.inverse_method=} the use of learning rate schedulers is not supported."
                logger.warning(msg)
                lr_scheduler_class = None

        self.inverse_model_optimizer = optimizer_class(
            self.inverse_model.parameters(),
            **optimizer_kwargs,
        )

        self.inverse_model_lr_scheduler = None
        self.inverse_model_lr_scheduler_step = None
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler_class is not None:
            self.inverse_model_lr_scheduler = lr_scheduler_class(self.inverse_model_optimizer, **lr_scheduler_kwargs)
            self.inverse_model_lr_scheduler_step = lr_scheduler_step

    def train_inverse_model(
        self,
        train_data: AnnotatedPerturbationData | None = None,
        validation_data: AnnotatedPerturbationData | None = None,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
    ) -> None:
        """Trains the inverse model prepared by :meth:`prepare_inverse_model` on control cell states.

        :param train_data: The data used to train the inverse model. Must be an instance of
            :class:`AnnotatedPerturbationData` with `target_reprs` set. Only the rows flagged as controls
            (`train_data.adata.obs[train_data.control_key]`) are used. Defaults to `None`, in which case
            `self.forward_model.train_data` is used.
        :type train_data: class:`AnnotatedPerturbationData | None`

        :param validation_data: (Optional) data used to validate the inverse model during training, with the same
            requirements as `train_data`; only its control rows are used. Defaults to `None`.
        :type validation_data: class:`AnnotatedPerturbationData | None`

        :param num_training_steps: The number of steps which to train the model on, defaults to `500`.
        :type num_training_steps: class:`int`

        :param valid_freq: The number of gradient steps after which to perform a validation step, only used when
            `validation_data` is provided, defaults to `None`.
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
        """
        # sanity checks
        msg = "You need to have instantitated the target predictor model by calling `prepare_inverse_model`"
        assert self.inverse_model is not None, msg

        if train_data is None:
            msg = ""
            assert self.forward_model is not None, msg

            msg = ""
            assert self.forward_model.train_data is not None, msg
            train_data = self.forward_model.train_data

        msg = ""
        assert isinstance(train_data, AnnotatedPerturbationData), msg

        msg = ""
        assert train_data.target_reprs is not None, msg

        # initialize trainer
        self.inverse_model_trainer = InverseModelTrainer(
            self.inverse_model,
            self.forward_model,
            self.target_prediction_model,
            self.inverse_model_optimizer,
            lr_scheduler=self.inverse_model_lr_scheduler,
            lr_scheduler_step=self.inverse_model_lr_scheduler_step,
            callbacks=callbacks,
            grad_steps_log_interval=grad_steps_log_interval,
        )

        # retrieving control indices
        control_idxs = np.argwhere(train_data.adata.obs[train_data.control_key].values == True)[:, 0]  # noqa: E712
        # initializing data loader with only control states
        self.inverse_model_train_data = train_data[control_idxs]
        self.inverse_model_train_dataloader = SequentialDataLoader(
            self.inverse_model_train_data,
            train_batch_size,
            state_transforms=state_transforms,
            device_id=self.device_id,
        )

        # optional validation data
        self.inverse_model_validation_dataloader = None
        if validation_data is not None:
            control_idxs = np.argwhere(validation_data.adata.obs[validation_data.control_key].values == True)[:, 0]  # noqa: E712
            self.inverse_model_validation_data = validation_data[control_idxs]
            self.inverse_model_validation_dataloader = SequentialDataLoader(
                self.inverse_model_train_data,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )

        # fitting the trainer
        self.inverse_model_trainer.fit(
            num_training_steps,
            self.inverse_model_train_dataloader,
            self.inverse_model_validation_dataloader,
            valid_freq,
        )

    def predict(
        self,
        control_states: torch.Tensor,
        return_loss: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Infers the perturbation covariates for the given control cell states using the trained inverse model.

        :param control_states: A tensor of control cell states for which to infer the perturbation covariates
            driving them towards `self.optimal_condition`.
        :type control_states: class:`torch.Tensor`

        :param return_loss: Whether to also return the loss computed by the inverse model, defaults to `False`.
        :type return_loss: class:`bool`

        :return: A dictionary with the source states, the inferred perturbation covariates, the forward model's
            predicted post-perturbation states, and the target covariates predicted from them, if `return_loss`
            is `False`. Otherwise, a tuple with the loss as first element and this dictionary as second element.
        :rtype: class:`dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]`
        """
        loss, out_dict = self.inverse_model(control_states)
        if return_loss:
            return loss, out_dict
        return out_dict
