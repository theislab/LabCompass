import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import torch

from sc_exp_design.constants import DataFields
from sc_exp_design.data.dataloaders import AnnotatedPerturbationData, SequentialDataLoader
from sc_exp_design.models.flow_matching import FlowMatching
from sc_exp_design.models.inverse_utils import LangevinOptimizer
from sc_exp_design.networks.blocks import BaseModule, BaseForwardModel
from sc_exp_design.networks.inverse import (
    BaseConditionOptimizer,
    MAPConditionOptimizer,
    LangevinSampler,
    NeuralInverseModel,
)
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior
from sc_exp_design.models.base import BaseModel
from sc_exp_design.training import BaseCallBack, TargetPredictionTrainer, InverseModelTrainer
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["InverseModel"]


class InverseModel(BaseModel):
    """"""
    def __init__(
        self,
        forward_model: BaseForwardModel | None = None,
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

        if inverse_method == "map":
            inverse_method_class = MAPConditionOptimizer
        elif inverse_method == "langevin":
            inverse_method_class = LangevinSampler
        elif inverse_method == "neural":
            inverse_method_class = NeuralInverseModel
        else:
            msg = f""
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
        target_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
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

        if target_covariates_predictor_kwargs is None:
            target_covariates_predictor_kwargs = {
                target_covariate: {} for target_covariate in target_covariates
            }

        # checking types
        msg = f""
        assert isinstance(target_covariates, Sequence), msg

        msg = f""
        assert isinstance(target_covariates_dims, dict), msg

        msg = f""
        assert isinstance(target_covariates_noise_models, dict), msg

        msg = f""
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
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        if train_data is None:
            msg = f""
            assert self.forward_model is not None, msg
            train_data = self.forward_model.train_data

        msg = f""
        assert isinstance(train_data, AnnotatedPerturbationData), msg

        msg = f""
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

        msg = f""
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
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = False,
        perturbation_initializer: Callable[[Any], torch.Tensor] | dict[str, Callable[[Any], torch.Tensor]] | None = None,
        perturbation_non_linearities: torch.nn.Module | Callable[[torch.Tensor], torch.Tensor] | dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] | None = None,
        perturbation_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
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
        """"""
        # we need to have at least one trained target predictor
        if target_prediction_model is None:
            msg = f"You need to have trained the target predictor model by calling `train_target_prediction_model`."
            assert self.target_prediction_model_trained, msg
            target_prediction_model = self.target_prediction_model

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
            msg = f""
            logger.warning(msg)
            is_discrete_dict = {covariate: False for covariate in perturbation_covariates}

        if isinstance(perturbation_initializer, Callable):
            msg = f"When `perturbation_initializer` is of type `Callable`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_initializer = {perturbation_covariates[0]: perturbation_initializer}
        if perturbation_initializer is None:
            msg = f"No initialization passed, setting to normal by default."
            logger.warning(msg)
            perturbation_initializer = {perturbation_covariate:torch.randn for perturbation_covariate in perturbation_covariates}
        
        if isinstance(perturbation_non_linearities, Callable | torch.nn.Module):
            msg = f"When `perturbation_non_linearities` is of type `Callable | torch.nn.Module`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_non_linearities = {perturbation_covariates[0]: perturbation_non_linearities}
        if perturbation_non_linearities is None:
            msg = f"No non-linearity passed, setting to identity by default."
            logger.warning(msg)
            perturbation_non_linearities = {perturbation_covariate: torch.nn.Identity() for perturbation_covariate in perturbation_covariates}

        if isinstance(perturbation_covariates_noise_models, str):
            msg = f"When `perturbation_non_linearities` is of type `str`, the respective perturbations should contain only one element, found {len(perturbation_covariates)}"
            assert len(perturbation_covariates) == 1, msg
            perturbation_covariates_noise_models = {perturbation_covariates[0]: perturbation_covariates_noise_models}
        if perturbation_covariates_noise_models is None:
            msg = f""
            logger.warning(msg)
            perturbation_covariates_noise_models = {covariate: None for covariate in perturbation_covariates}
        
        if perturbation_covariates_predictor_kwargs is None:
            msg = f""
            logger.warning(msg)
            perturbation_covariates_predictor_kwargs = {
                covariate: {} for covariate in perturbation_covariates
            }

        if perturbation_encoder_mlp_kwargs is None:
            msg = f""
            logger.warning(msg)
            perturbation_encoder_mlp_kwargs = {}

        # when we pass the prior on the perturbations        
        if prior is not None:
            if prior_weight is None:
                msg = f"`prior` was passed, but no `prior_weight` was given. Setting to 1.0 by default."
                logger.warning(msg)
                prior_weight = 1.0

            if isinstance(prior, torch.distributions.Distribution):
                msg = f""
                assert len(perturbation_covariates) == 1, msg
                prior = {perturbation_covariates[0]: prior}

        # check types
        msg = f"`perturbation_covariates` nees to be a sequence of perturbation covatiate identifiers, found {type(perturbation_covariates)}"
        assert isinstance(perturbation_covariates, Sequence), msg

        msg = f"`perturbation_covariates_dims` needs to be a dictionary mapping each condition to its dimensionality, found {type(perturbation_covariates_dims)}"
        assert isinstance(perturbation_covariates_dims, dict), msg

        msg = f""
        assert isinstance(is_discrete_dict, dict), msg

        msg = f""
        assert isinstance(perturbation_initializer, dict), msg

        msg = f""
        assert isinstance(perturbation_non_linearities, dict), msg

        msg = f""
        assert isinstance(perturbation_covariates_noise_models, dict), msg

        msg = f""
        assert isinstance(perturbation_covariates_predictor_kwargs, dict), msg

        if prior is not None:
            msg = f""
            assert isinstance(prior, dict), msg

        # we want all these dictionaries to share the same keys (i.e.: covariate ids) found in perturbation_covariates
        for perturbation_key in perturbation_covariates:
            
            msg = f"{perturbation_key=} not found in `perturbation_covariates_dims.keys()`, you need to specify a corresponding dimensionality."
            assert perturbation_key in perturbation_covariates_dims.keys(), msg
            
            msg = f""
            assert perturbation_key in is_discrete_dict.keys(), msg
            
            # when initializer is not passed for a covariate we set it to normal initialization
            msg = f""
            assert perturbation_key in perturbation_initializer.keys(), msg
            if perturbation_initializer[perturbation_key] is None:
                perturbation_initializer[perturbation_key] = torch.randn

            # when non linearity is not passed for a covariate we set it to identity
            msg = f""
            assert perturbation_key in perturbation_non_linearities.keys(), msg
            if perturbation_non_linearities[perturbation_key] is None:
                perturbation_non_linearities[perturbation_key] = torch.nn.Identity()

            msg = f""
            assert perturbation_key in perturbation_covariates_noise_models, msg

            # when non keyword settings are not passed for a covariate we set it to empty dictionary
            msg = f""
            assert perturbation_key in perturbation_covariates_predictor_kwargs, msg
            if perturbation_covariates_predictor_kwargs[perturbation_key] is None:
                perturbation_covariates_predictor_kwargs[perturbation_key] = {}

            # when using prior we need to have the log_prob method
            if prior is not None:
                msg = f""
                assert perturbation_key in prior.keys(), msg
                if not hasattr(prior[perturbation_key], "log_prob"):
                    msg = f"Prior for {perturbation_key} does not have a `log_prob` method."
                    raise ValueError(msg)

        # when we use langevin we need to pass the number of samples
        if n_samples is None and self.inverse_method == "langevin":
            msg = f""
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
            if not optimizer_class is LangevinOptimizer:
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
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_inverse_model`"
        assert self.inverse_model is not None, msg

        if train_data is None:
            msg = f""
            assert self.forward_model is not None, msg

            msg = f""
            assert self.forward_model.train_data is not None, msg
            train_data = self.forward_model.train_data

        msg = f""
        assert isinstance(train_data, AnnotatedPerturbationData), msg

        msg = f""
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

        # retrieving control states if available
        inverse_model_train_data = None
        inverse_model_train_dataloader = None
        if train_data.has_controls:
            control_idxs = np.argwhere(train_data.adata.obs[train_data.control_key].values == True)[:, 0]
            inverse_model_train_data = train_data[control_idxs]
            # initializing data loader with only control states
            inverse_model_train_dataloader = SequentialDataLoader(
                inverse_model_train_data,
                train_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )
        self.inverse_model_train_data = inverse_model_train_data
        self.inverse_model_train_dataloader = inverse_model_train_dataloader

        # optional validation data
        inverse_model_validation_data = None
        inverse_model_validation_dataloader = None
        if validation_data is not None:
            if validation_data.has_controls:
                control_idxs = np.argwhere(validation_data.adata.obs[validation_data.control_key].values == True)[:, 0]
                inverse_model_validation_data = validation_data[control_idxs] 
                inverse_model_validation_dataloader = SequentialDataLoader(
                    inverse_model_validation_data,
                    validation_batch_size,
                    state_transforms=state_transforms,
                    device_id=self.device_id,
                )
        self.inverse_model_validation_data = inverse_model_validation_data
        self.inverse_model_validation_dataloader = inverse_model_validation_dataloader

        # fitting the trainer
        self.inverse_model_trainer.fit(
            num_training_steps,
            self.inverse_model_train_dataloader,
            self.inverse_model_validation_dataloader,
            valid_freq,
        )

    def predict(
        self,
        control_states: torch.Tensor | None,
        return_loss: bool = False,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """"""
        loss, out_dict = self.inverse_model(control_states)
        if return_loss:
            return loss, out_dict
        return out_dict
        