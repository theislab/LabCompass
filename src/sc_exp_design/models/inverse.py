import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from sc_exp_design.data.dataloaders import TrainData, SequentialDataLoader
from sc_exp_design.models.flow_matching import FlowMatching
from sc_exp_design.networks.inverse import (
    BaseConditionOptimizer,
    MAPConditionOptimizer,
    LangevinSampler,
    NeuralInverseModel,
)
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior
from sc_exp_design.training import CallBack, TargetPredictionTrainer, InverseModelTrainer
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["InverseModel"]


class InverseModel:
    """"""
    def __init__(
        self,
        forward_model: FlowMatching | None = None,
        state_dim: int | None = None,
        inverse_method: Literal["map", "langevin", "neural"] = "map",
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        # sanity check on the input 
        if state_dim is None:
            msg = f""
            assert forward_model is not None, msg
            state_dim = forward_model.cvf_config.flow_dim
        else:
            msg = f""
            assert isinstance(state_dim, int), msg
            # if we pass the forward model we take the dimensionality from there
            if (forward_model is not None) and state_dim != forward_model.cvf_config.flow_dim:
                msg = f""
                logger.warning(msg)
                state_dim = forward_model.cvf_config.flow_dim

        self.forward_model = forward_model
        self.state_dim = state_dim

        if inverse_method == "map":
            inverse_method_class = MAPConditionOptimizer
        elif inverse_method == "langevin":
            inverse_method_class = LangevinSampler
        elif inverse_method == "neural":
            inverse_method_class = NeuralInverseModel
        else:
            msg = f""
            raise ValueError(msg)

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
        target_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
        target_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
    ) -> None:
        """"""
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
            target_covariates_noise_models = {target_covariate: None for target_covariate in target_covariates}

        # setting the optional keyword arguments to a dictionary when not passed
        if target_covariates_predictor_kwargs is None:
            target_covariates_predictor_kwargs = {
                target_covariate: {} for target_covariate in target_covariates
            }

        # storing the settings here as attributes
        self.target_covariates = target_covariates
        self.target_covariates_dims = target_covariates_dims
        self.target_covariates_noise_models = target_covariates_noise_models
        self.target_covariates_predictor_kwargs = target_covariates_predictor_kwargs

        # initializing the predictor for each target covariate
        self.target_prediction_model = PerturbationApproximatePosterior(
            self.state_dim,
            freeze_grads=False, # we want to backpropagate the gradients from its input
            target_output_dims=self.target_covariates_dims,
            noise_models=self.target_covariates_noise_models,
            covariate_kwargs=self.target_covariates_predictor_kwargs,
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
        train_data: TrainData | None = None,
        validation_data: TrainData | None = None,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 100,
    ) -> None:
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        if train_data is None:
            msg = f""
            assert self.forward_model is not None, msg
            train_data = self.forward_model.train_data

        msg = f""
        assert isinstance(train_data, TrainData), msg

        msg = f""
        assert train_data.target_perturbation_repr is not None, msg

        # initializing data loader
        self.train_data = train_data
        self.train_dataloader = SequentialDataLoader(
            self.train_data,
            train_batch_size,
            state_transforms=state_transforms,
            device_id=self.device_id
        )

        # initialize trainer
        self.trainer = TargetPredictionTrainer(
            self.target_prediction_model,
            self.target_prediction_optimizer,
            lr_scheduler=self.target_prediction_lr_scheduler,
            lr_scheduler_step=self.target_prediction_lr_scheduler_step,
            callbacks=callbacks,
            grad_step_interval_log=grad_step_interval_log,
        )

        # optional validation data
        self.validation_dataloader = None
        if validation_data is not None:
            self.validation_dataloader = SequentialDataLoader(
                validation_data,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )

        # fitting the trainer
        self.trainer.fit(
            num_training_steps,
            self.train_dataloader,
            self.validation_dataloader,
            valid_freq,
        )

        self.target_prediction_model_trained = True

    def prepare_inverse_model(
        self,
        optimal_condition: torch.Tensor | dict[str, torch.Tensor], 
        loss_fn: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] | Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cond_dim: int | dict[str, int],
        perturbation_representation_keys: Sequence[str],
        is_discrete_dict: bool | dict[str, bool],
        forward_model: torch.nn.Module | None = None, 
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = False,
        perturbation_initializer: Callable[[Any], torch.Tensor] | dict[str, Callable[[Any], torch.Tensor]] | None = None,
        perturbation_non_linearities: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor] | dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] | None = None,
        n_samples: int | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
        **kwargs,
    ) -> None:
        """"""
        # we need trained target predictor
        msg = f"You need to have trained the target predictor model by calling `train_target_prediction_model`."
        assert self.target_prediction_model_trained, msg

        # we need to have at least one forward model
        if forward_model is None:
            msg = f""
            assert self.forward_model is not None, msg
            forward_model = self.forward_model

        # preparing input with some sanity checks
        if isinstance(optimal_condition, torch.Tensor):
            msg = f""
            assert len(self.target_covariates) == 1, msg
            optimal_condition = {self.target_covariates[0]: optimal_condition}
        if isinstance(loss_fn, Callable):
            msg = f""
            assert len(self.target_covariates) == 1, msg
            loss_fn = {self.target_covariates[0]: loss_fn}
        if isinstance(perturbation_representation_keys, str):
            perturbation_representation_keys = (perturbation_representation_keys, )
        if isinstance(cond_dim, int):
            msg = f"When `cond_dim` is of type `int`, the respective perturbations should contain only one element, found {len(perturbation_representation_keys)}"
            assert len(perturbation_representation_keys) == 1, msg
            cond_dim = {perturbation_representation_keys[0]: cond_dim}
        if isinstance(is_discrete_dict, bool):
            msg = f"When `cond_dim` is of type `int`, the respective perturbations should contain only one element, found {len(perturbation_representation_keys)}"
            assert len(perturbation_representation_keys) == 1, msg
            is_discrete_dict = {perturbation_representation_keys[0]: is_discrete_dict}
        if isinstance(perturbation_initializer, Callable):
            msg = f"When `perturbation_initializer` is of type `Callable`, the respective perturbations should contain only one element, found {len(perturbation_representation_keys)}"
            assert len(perturbation_representation_keys) == 1, msg
            perturbation_initializer = {perturbation_representation_keys[0]: perturbation_initializer}
        if perturbation_initializer is None:
            perturbation_initializer = {perturbation_covariate:torch.randn for perturbation_covariate in perturbation_representation_keys}
        if isinstance(perturbation_non_linearities, Callable | torch.nn.Module):
            msg = f"When `perturbation_non_linearities` is of type `Callable | torch.nn.Module`, the respective perturbations should contain only one element, found {len(perturbation_representation_keys)}"
            assert len(perturbation_representation_keys) == 1, msg
            perturbation_non_linearities = {perturbation_representation_keys[0]: perturbation_non_linearities}
        if perturbation_non_linearities is None:
            perturbation_non_linearities = {perturbation_covariate: torch.nn.Identity() for perturbation_covariate in perturbation_representation_keys}

        msg = f"`cond_dim` needs to be a dictionary mapping each condition to its dimensionality, found {type(cond_dim)}"
        assert isinstance(cond_dim, dict), msg
        msg = f"`perturbation_representation_keys` nees to be a sequence of perturbation covatiate identifiers, found {type(perturbation_representation_keys)}"
        assert isinstance(perturbation_representation_keys, Sequence), msg
        msg = f""
        assert isinstance(is_discrete_dict, dict), msgs
        msg = f""
        assert isinstance(perturbation_initializer, dict), msg
        msg = f""
        assert isinstance(perturbation_non_linearities, dict), msg

        # we want all the keys to be in condition dim
        for perturbation_key in perturbation_representation_keys:
            msg = f"{perturbation_key=} not found in `cond_dim.keys()`, you need to specify a corresponding dimensionality."
            assert perturbation_key in cond_dim.keys(), msg
            msg = f""
            assert perturbation_key in is_discrete_dict.keys(), msg
            msg = f""
            assert perturbation_key in perturbation_initializer.keys(), msg
            if perturbation_initializer[perturbation_key] is None:
                perturbation_initializer[perturbation_key] = torch.randn
            msg = f""
            assert perturbation_key in perturbation_non_linearities.keys(), msg
            if perturbation_non_linearities[perturbation_key] is None:
                perturbation_non_linearities[perturbation_key] = torch.nn.Identity()

        # when we pass the prior on the perturbations        
        if prior is not None:
            if prior_weight is None:
                msg = f"`prior` is not None, but prior_weight was not passed. Setting to 1.0 by default."
                logger.warning(msg)
                prior_weight = 1.0
        
        # when we use langevin we need to pass the number of samples
        if n_samples is None and self.inverse_method == "langevin":
            msg = f""
            logger.warning(msg)
            n_samples = 1

        # storing the attributes here
        self.optimal_condition = optimal_condition
        self.loss_fn = loss_fn
        self.cond_dim = cond_dim
        self.perturbation_representation_keys = perturbation_representation_keys
        self.is_discrete_dict = is_discrete_dict
        self.prior = prior
        self.prior_weight = prior_weight
        self.hard = hard
        self.perturbation_initializer = perturbation_initializer
        self.perturbation_non_linearities = perturbation_non_linearities
        self.n_samples = n_samples

        # initializing the inverse model
        self.inverse_model = self.inverse_method_class(
            self.optimal_condition,
            self.target_prediction_model,
            forward_model,
            self.loss_fn,
            self.cond_dim,
            self.perturbation_representation_keys,
            self.is_discrete_dict,
            prior=self.prior,
            prior_weight=self.prior_weight,
            hard=self.hard,
            perturbation_initializer=self.perturbation_initializer,
            perturbation_non_linearities=self.perturbation_non_linearities,
            n_samples=self.n_samples,
            state_dim=self.state_dim,
            **kwargs
        )
        self.inverse_model = self.inverse_model.float()
        self.inverse_model = self.inverse_model.to(self.device)

        # optimizer and scheduler 
        if not self.inverse_model.training_free:
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
        train_data: TrainData | None = None,
        validation_data: TrainData | None = None,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: CallBack | None = None,
        grad_step_interval_log: int = 100,
    ) -> None:
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_inverse_model`"
        assert self.inverse_model is not None, msg

        msg = f""
        assert not self.inverse_model.training_free, msg

        if train_data is None:
            msg = f""
            assert self.forward_model is not None, msg
            train_data = self.forward_model.train_data

        msg = f""
        assert isinstance(train_data, TrainData), msg

        msg = f""
        assert train_data.target_perturbation_repr is not None, msg

        # initialize trainer
        self.inverse_model_trainer = InverseModelTrainer(
            self.inverse_model,
            self.forward_model,
            self.target_prediction_model,
            self.inverse_model_optimizer,
            lr_scheduler=self.inverse_model_lr_scheduler,
            lr_scheduler_step=self.inverse_model_lr_scheduler_step,
            callbacks=callbacks,
            grad_step_interval_log=grad_step_interval_log,
        )

        # retrieving control indices
        control_idxs = np.argwhere(train_data.adata.obs[train_data.control_key].values == True)[:, 0]
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
            control_idxs = np.argwhere(validation_data.adata.obs[validation_data.control_key].values == True)[:, 0]
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
    ) -> torch.Tensor:
        """"""
        