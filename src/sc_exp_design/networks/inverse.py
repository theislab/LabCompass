import abc
from collections.abc import Callable, Iterator, Sequence
import itertools
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sc_exp_design.networks.blocks import BaseModule, BaseForwardModel
from sc_exp_design.constants import DataFields, PredictionFields, ParamsFields
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior

__all__ = ["BaseConditionOptimizer", "MAPConditionOptimizer", "LangevinSampler", "NeuralInverseModel"]


class BaseConditionOptimizer(BaseModule):
    """"""

    def __init__(
        self, 
        optimal_condition: dict[str, torch.Tensor], 
        target_prediction_model: nn.Module,
        forward_model: BaseForwardModel,
        loss_fn: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        perturbation_covariates: Sequence[str],
        perturbation_covariates_dims: dict[str, int],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        tau: float = 1.0,
        eps: float = 1e-10,
        perturbation_initializer: dict[str, Callable[[Any], torch.Tensor]] = None,
        perturbation_non_linearities: dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] = None,
        device_id: Literal["cpu", "cuda"] = "cuda",
        **kwargs,
    ) -> None:
        """"""
        super().__init__()
    
        self.optimal_condition = optimal_condition
        self.target_prediction_model = target_prediction_model
        self.forward_model = forward_model
        self.perturbation_covariates_dims = perturbation_covariates_dims
        self.loss_fn = loss_fn
        self.perturbation_covariates = perturbation_covariates
        self.is_discrete_dict = is_discrete_dict
        self.prior = prior
        self.prior_weight = prior_weight
        self.hard = hard
        self.tau = tau
        self.eps = eps 
        self.perturbation_initializer = perturbation_initializer
        self.perturbation_non_linearities = perturbation_non_linearities
        self.device_id = device_id
        self.device = torch.device(self.device_id)

    def compute_loss(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        e_optimized: dict[str, nn.Parameter],
    ) -> torch.Tensor:
        """"""
        loss = torch.zeros((), requires_grad=True, device=list(target.values())[0].device)
        for covariate, loss_fn in self.loss_fn.items():
            loss = loss + loss_fn(pred[covariate], target[covariate])
        if self.prior:
            for key in self.prior:
                log_prior = self.prior[key].log_prob(e_optimized[key]).sum()
                loss = loss - self.prior_weight * log_prior
        return loss

    @abc.abstractmethod
    def forward(
            self,
        ) -> torch.Tensor:
        """"""
        raise NotImplementedError
    
    def get_optimized_e(
            self,
        ) -> Sequence[torch.nn.Parameter]:
        optimized_perturbation_data_tmp = [self.optimized_perturbation_data[pert].detach() for pert in self.optimized_perturbation_data]        
        return optimized_perturbation_data_tmp

    def differentiable_categorical(
            self,
            logits: torch.Tensor,
        ) -> torch.Tensor:
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + self.eps) + self.eps)

        # Gumbel-Softmax reparameterization
        y_soft = F.softmax((logits + gumbel_noise) / self.tau, dim=-1)

        if self.hard:
            # Convert to hard one-hot, but keep gradients
            y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            return y_hard + (y_soft - y_hard).detach()  # Keep gradients

        return y_soft


class MAPConditionOptimizer(BaseConditionOptimizer):
    """"""

    def __init__(
        self, 
        optimal_condition: dict[str, torch.Tensor], 
        target_prediction_model: nn.Module,
        forward_model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_covariates: Sequence[str],
        perturbation_covariates_dims: int | dict[str, int],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        tau: float = 1.0,
        eps: float = 1e-10,
        perturbation_initializer: dict[str, Callable[[Any], torch.Tensor]] | None = None,
        perturbation_non_linearities: dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] = None,
        device_id: Literal["cpu", "cuda"] = "cuda",
        **kwargs,
    ) -> None:
        """"""
        
        super().__init__(
            optimal_condition, 
            target_prediction_model,
            forward_model,
            loss_fn, 
            perturbation_covariates,
            perturbation_covariates_dims,
            is_discrete_dict,
            prior=prior,
            prior_weight=prior_weight,
            hard=hard,
            tau=tau,
            eps=eps,
            perturbation_initializer=perturbation_initializer,
            perturbation_non_linearities=perturbation_non_linearities,
            device_id=device_id,
        )
        
        # perturbation representation keys 
        self.perturbation_covariates = perturbation_covariates

        # initializing modules
        self._init_modules()

    def parameters(
        self,
    ) -> list[torch.nn.Parameter]:
        """"""
        return list(self.optimized_perturbation_data.values())

    def _init_modules(
        self,
    ) -> None:
        """"""
        self.optimized_perturbation_data = {}
        
        for perturbation_representation, perturbation_covariates_dims in self.perturbation_covariates_dims.items():
            covariate_initializer = self.perturbation_initializer[perturbation_representation]
            self.optimized_perturbation_data[perturbation_representation] = nn.Parameter(covariate_initializer(1, perturbation_covariates_dims, requires_grad=True, device=self.device_id))

    def forward(
            self,
            X_controls: torch.Tensor,
        ) -> torch.Tensor:
        # prepare batch information cellFlow           
        batch_dict = {
            DataFields.SOURCE_STATE: X_controls,
        }
        
        expanded_perturbation_data = {}
        for pert_key in self.optimized_perturbation_data:
            if not self.is_discrete_dict:
                non_linearity = self.perturbation_non_linearities[pert_key]
                expanded_perturbation_data[pert_key] = non_linearity(
                    self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1)
                ) 
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1))
            
        batch_dict[DataFields.PERTURBATION_DATA] = expanded_perturbation_data

        # pushing forward the particles 
        X_pert_pred = self.forward_model.predict(
            batch_dict,
            no_grad=False,
        )
        
        # simulation
        class_pred = self.target_prediction_model(X_pert_pred)
        
        # handling shape of optimal condition
        optimal_condition = {
            covariate: covariate_data.repeat(X_controls.shape[0], 1).to(X_controls.device) for covariate, covariate_data in self.optimal_condition.items()
        }

        # compute loss 
        loss = self.compute_loss(class_pred, optimal_condition, self.optimized_perturbation_data)

        # constructing step output dictionary
        out_dict = {
            DataFields.SOURCE_STATE: X_controls.detach().cpu(),
            DataFields.PERTURBATION_DATA: {
                covariate: covariate_data.detach().cpu() for covariate, covariate_data in expanded_perturbation_data.items()
            },
            PredictionFields.PREDICTION_DATA: X_pert_pred.detach().cpu(),
            PredictionFields.TARGET_PREDICTION_DATA: {
                covariate: covariate_pred.detach().cpu() for covariate, covariate_pred in class_pred.items()
            },
            PredictionFields.PREDICTED_PERTURBATION: {
                covariate: covariate_data.clone().detach().cpu() for covariate, covariate_data in self.optimized_perturbation_data.items()
            },
        }
        return loss, out_dict
    
    
class LangevinSampler(BaseConditionOptimizer):
    """"""

    def __init__(
        self,
        optimal_condition: dict[str, torch.Tensor], 
        target_prediction_model: nn.Module,
        forward_model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_covariates: Sequence[str],
        perturbation_covariates_dims: int | dict[str, int],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        tau: float = 1.0,
        eps: float = 1e-10,
        n_samples: int | None = None,
        perturbation_initializer: dict[str, Callable[[Any], torch.Tensor]] = None,
        perturbation_non_linearities: dict[str, torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]] = None,
        device_id: Literal["cpu", "cuda"] = "cuda",
        **kwargs,
    ) -> None:
        """"""
        
        super().__init__(
            optimal_condition, 
            target_prediction_model,
            forward_model,
            loss_fn, 
            perturbation_covariates,
            perturbation_covariates_dims,
            is_discrete_dict,
            prior=prior,
            prior_weight=prior_weight,
            hard=hard,
            tau=tau,
            eps=eps,
            perturbation_initializer=perturbation_initializer,
            perturbation_non_linearities=perturbation_non_linearities,
            device_id=device_id,
        )
            
        self.perturbation_covariates = perturbation_covariates 
        self.is_discrete_dict = is_discrete_dict
        self.n_samples = n_samples

        # initializing modules
        self._init_modules()

    def parameters(
        self,
    ) -> list[torch.nn.Parameter]:
        """"""
        return list(self.optimized_perturbation_data.values())

    def _init_modules(
        self,
    ) -> None:
        """"""
        # Optimized perturbation representation data 
        self.optimized_perturbation_data = {} 
        
        for perturbation_representation, perturbation_covariates_dims in self.perturbation_covariates_dims.items():
            covariate_initializer = self.perturbation_initializer[perturbation_representation]
            self.optimized_perturbation_data[perturbation_representation] = covariate_initializer(
                self.n_samples,
                perturbation_covariates_dims,
                requires_grad=True,
                device=self.device_id
            )

    def forward(
            self,
            X_controls: torch.Tensor,
        ) -> torch.Tensor:
        # Expand target  and controls
        target = {
            covariate: covariate_data.repeat(self.n_samples, X_controls.shape[0], 1).to(X_controls.device) for covariate, covariate_data in self.optimal_condition.items()
        }
        X_controls = X_controls.unsqueeze(0).expand(self.n_samples, -1, -1) 
        
        # prepare batch information cellFlow           
        batch_dict = {
            DataFields.SOURCE_STATE: X_controls,
        }
        expanded_perturbation_data = {}
        for pert_key in self.optimized_perturbation_data:
            if not self.is_discrete_dict:
                non_linearity = self.perturbation_non_linearities[pert_key]
                expanded_perturbation_data[pert_key] = non_linearity( 
                    self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[1], -1) 
                )
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[1], -1))
                       
        batch_dict[DataFields.PERTURBATION_DATA] = expanded_perturbation_data

        # pushing forward the particles 
        X_pert_pred = self.forward_model.predict(
            batch_dict,
            no_grad=False,
        )
        
        class_pred = self.target_prediction_model(X_pert_pred)
        loss = self.compute_loss(class_pred, target, self.optimized_perturbation_data)
        
        # constructing step output dictionary
        out_dict = {
            DataFields.SOURCE_STATE: X_controls.detach().cpu(),
            DataFields.PERTURBATION_DATA: {
                covariate: covariate_data.detach().cpu() for covariate, covariate_data in expanded_perturbation_data.items()
            },
            PredictionFields.PREDICTION_DATA: X_pert_pred.detach().cpu(),
            PredictionFields.TARGET_PREDICTION_DATA: {
                covariate: covariate_pred.detach().cpu() for covariate, covariate_pred in class_pred.items()
            },
            PredictionFields.PREDICTED_PERTURBATION: {
                covariate: covariate_data.clone().detach().cpu() for covariate, covariate_data in self.optimized_perturbation_data.items()
            },
        }

        return loss, out_dict


class NeuralInverseModel(BaseConditionOptimizer):
    """"""

    def __init__(
        self,
        optimal_condition: dict[str, torch.Tensor], 
        target_prediction_model: nn.Module,
        forward_model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        perturbation_covariates_dims: int | dict[str, int],
        perturbation_covariates: Sequence[str],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        tau: float = 1.0,
        eps: float = 1e-10,
        state_dim: int | None = None,
        perturbation_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
        perturbation_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        device_id: Literal["cpu", "cuda"] = "cuda",
        **kwargs,
    ) -> None:
        """"""

        super().__init__(
            optimal_condition, 
            target_prediction_model,
            forward_model,
            loss_fn, 
            perturbation_covariates_dims,
            perturbation_covariates,
            is_discrete_dict,
            prior=prior,
            prior_weight=prior_weight,
            hard=hard,
            tau=tau,
            eps=eps,
            device_id=device_id,
        )

        # preparing input with some sanity checks
        msg = f""
        assert perturbation_covariates is not None, msg
        if isinstance(perturbation_covariates, str):
            perturbation_covariates = (perturbation_covariates, )

        if perturbation_covariates_noise_models is None:
            perturbation_covariates_noise_models = {covariate: None for covariate in perturbation_covariates}

        # setting the optional keyword arguments to a dictionary when not passed
        if perturbation_covariates_predictor_kwargs is None:
            perturbation_covariates_predictor_kwargs = {
                covariate: {} for covariate in perturbation_covariates
            }
        msg = f""
        assert perturbation_covariates_noise_models is not None, msg
        msg = f""
        assert perturbation_covariates_predictor_kwargs is not None, msg
        msg = f""
        assert state_dim is not None, msg

        # setting additional attributes
        self.state_dim = state_dim
        self.perturbation_covariates_noise_models = perturbation_covariates_noise_models
        self.perturbation_covariates_predictor_kwargs = perturbation_covariates_predictor_kwargs

        # initializing modules
        self._init_modules()

    @property
    def input_dim(
        self,
    ) -> int:
        """"""
        input_dim = self.state_dim
        for optimal_covariate in self.optimal_condition.values():
            input_dim = input_dim + optimal_covariate.shape[0] 
        return input_dim

    def _init_modules(
        self,
    ) -> None:
        """"""
        # initializing the predictor for each target covariate
        self.perturbation_prediction_model = PerturbationApproximatePosterior(
            self.input_dim,
            freeze_grads=False, # we want to backpropagate the gradients from its input
            target_output_dims=self.perturbation_covariates_dims,
            noise_models=self.perturbation_covariates_noise_models,
            covariate_kwargs=self.perturbation_covariates_predictor_kwargs,
        )

    def parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """"""
        return self.perturbation_prediction_model.parameters()
    
    def to(
        self,
        device: torch.device,
    ) -> nn.Module:
        """"""
        self.perturbation_prediction_model = self.perturbation_prediction_model.to(device)
        return self

    def __prepare_perturbation_data(
        self,
        pert_data: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """"""
        perturbation_data = {}
        for covariate, covariate_data in pert_data.items():
            # retrieving covariate settings
            covariate_noise_model = self.perturbation_prediction_model.noise_models[covariate]
            is_discrete_covariate = self.is_discrete_dict[covariate]
            covariate_network = self.perturbation_prediction_model.pert_approximate_posterior[covariate]

            # when discrete
            if is_discrete_covariate:
                self.differentiable_categorical(covariate_params)
            
            # gaussian noise model
            elif covariate_noise_model == "gaussian":
                # parsing parameter dictionaries
                mean = covariate_data[ParamsFields.MEAN]
                covariance = covariate_data[ParamsFields.COVARIANCE]

                # reparametrization trick
                z = torch.randn_like(mean)
                if covariate_network.cov_estimation_mode == "isotropic":
                    covariate_data = mean + covariance*z
                elif covariate_network.cov_estimation_mode == "anisotropic":
                    covariate_data = ...
                    raise NotImplementedError

            # negative binomial noise model
            elif covariate_noise_model == "neg_bin":
                raise NotImplementedError
            
            # identity (keep as ise)
            elif covariate_noise_model is None:
                pass

            else:
                msg = f""
                raise ValueError(msg)
            
            perturbation_data[covariate] = covariate_data
        return perturbation_data

    def forward(
        self,
        control_states: torch.Tensor,
    ) -> torch.Tensor:
        """"""
        # handling the shape of the optimal condition
        target = {
            covariate: covariate_data.repeat(control_states.shape[0], 1).to(control_states.device)
            for covariate, covariate_data in self.optimal_condition.items()
        }
        # predicting the optimal perturbation
        input_tensor = torch.concatenate((control_states, *target.values()), dim=1)
        pert_params = self.perturbation_prediction_model(input_tensor)

        # preparing perturbation params
        pert_data = self.__prepare_perturbation_data(pert_params)

        # prepare batch information cellFlow           
        batch_dict = {
            DataFields.SOURCE_STATE: control_states,
            DataFields.PERTURBATION_DATA: pert_data,
        }

        # predict perturbation effetc
        x1_hat = self.forward_model.predict(
            batch_dict,
            no_grad=False,
        )

        # predicting target response
        class_pred = self.target_prediction_model(x1_hat)
        loss = self.compute_loss(class_pred, target, pert_data)

        # constructing step output dictionary
        out_dict = {
            DataFields.SOURCE_STATE: control_states.detach().cpu(),
            DataFields.PERTURBATION_DATA: {
                covariate: covariate_data.detach().cpu() for covariate, covariate_data in pert_data.items()
            },
            PredictionFields.PREDICTION_DATA: x1_hat.detach().cpu(),
            PredictionFields.TARGET_PREDICTION_DATA: {
                covariate: covariate_pred.detach().cpu() for covariate, covariate_pred in class_pred.items()
            },
            PredictionFields.PREDICTED_PERTURBATION: {
                covariate: torch.mean(covariate_data, dim=0).detach().cpu() for covariate, covariate_data in pert_data.items()
            }
        }
        return loss, out_dict
