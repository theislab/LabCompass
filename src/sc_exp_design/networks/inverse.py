import abc
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sc_exp_design.networks.blocks import BaseModule
from sc_exp_design.constants import DataFields, ParamsFields
from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior

__all__ = ["BaseConditionOptimizer", "MAPConditionOptimizer", "LangevinSampler", "NeuralInverseModel"]


class BaseConditionOptimizer(BaseModule):
    """"""
    training_free: bool

    def __init__(
        self, 
        optimal_condition: dict[str, torch.Tensor], 
        linear_classifier: nn.Module,
        perturbation_predictor: nn.Module,
        loss_fn: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        cond_dim: dict[str, int],
        perturbation_representation_keys: Sequence[str],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        **kwargs
    ) -> None:
        """"""
        super().__init__()
    
        self.optimal_condition = optimal_condition
        self.linear_classifier = linear_classifier
        self.perturbation_predictor = perturbation_predictor
        self.cond_dim = cond_dim
        self.loss_fn = loss_fn
        self.perturbation_representation_keys = perturbation_representation_keys
        self.is_discrete_dict = is_discrete_dict
        self.prior = prior
        self.prior_weight = prior_weight
        self.hard = hard


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

    @staticmethod    
    def differentiable_categorical(
            logits: torch.Tensor,
            tau: float = 1.0,
            hard: bool = False,
            eps: float = 1e-10
        ) -> torch.Tensor:
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + eps) + eps)

        # Gumbel-Softmax reparameterization
        y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)

        if hard:
            # Convert to hard one-hot, but keep gradients
            y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            return y_hard + (y_soft - y_hard).detach()  # Keep gradients

        return y_soft


class MAPConditionOptimizer(BaseConditionOptimizer):
    """"""
    training_free: bool = True

    def __init__(
        self, 
        optimal_condition: dict[str, torch.Tensor], 
        linear_classifier: nn.Module,
        perturbation_predictor: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cond_dim: int | dict[str, int],
        perturbation_representation_keys: Sequence[str],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        lr: float = 1e-1,
        tau: float = 1.0,
        **kwargs,
    ) -> None:
        """"""
        
        super().__init__(
            optimal_condition, 
            linear_classifier,
            perturbation_predictor,
            loss_fn, 
            cond_dim,
            perturbation_representation_keys,
            is_discrete_dict,
            prior,
            prior_weight,
            hard,
        )
        
        self.is_discrete_dict = is_discrete_dict
        self.tau = tau
        
        # perturbation representation keys 
        self.perturbation_representation_keys = perturbation_representation_keys

        # initializing modules
        self._init_modules()

        # Set up optimizer for the conditions 
        self.e_optimizer = optim.Adam([self.optimized_perturbation_data[perturbation_representation] for perturbation_representation in self.optimized_perturbation_data], 
                                      lr=lr)
    
    def _init_modules(
        self,
    ) -> None:
        """"""
        self.optimized_perturbation_data = {}
        
        for perturbation_representation, cond_dim in self.cond_dim.items():
            self.optimized_perturbation_data[perturbation_representation] = nn.Parameter(torch.randn(1, cond_dim))

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
                expanded_perturbation_data[pert_key] = self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1) 
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1))
            
        batch_dict[DataFields.PERTURBATION_DATA] = expanded_perturbation_data

        # pushing forward the particles 
        X_pert_pred = self.perturbation_predictor.predict(batch_dict,
                                                          no_grad=False)
        
        # simulation
        class_pred = self.linear_classifier(X_pert_pred)
        
        # compute loss 
        loss = self.compute_loss(class_pred, self.optimal_condition.repeat(class_pred.shape[0], 1), 
                                 self.optimized_perturbation_data)
        
        # optimize step 
        self.e_optimizer.zero_grad()
        loss.backward()
        self.e_optimizer.step()
        return loss
    
    
class LangevinSampler(BaseConditionOptimizer):
    """"""
    training_free: bool = True

    def __init__(
        self,
        optimal_condition: dict[str, torch.Tensor], 
        linear_classifier: nn.Module,
        perturbation_predictor: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cond_dim: int | dict[str, int],
        perturbation_representation_keys: Sequence[str],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        n_samples: int | None = None,
        eta: float = 1e-1,
        noise_scale: float = 1e-1, 
        tau: float = 1.0,
        **kwargs,
    ) -> None:
        """"""
        
        super().__init__(
            optimal_condition, 
            linear_classifier,
            perturbation_predictor,
            loss_fn, 
            cond_dim,
            perturbation_representation_keys,
            is_discrete_dict,
            prior,
            prior_weight,
            hard,
        )
            
        self.eta = eta
        self.noise_scale = noise_scale
        self.perturbation_representation_keys = perturbation_representation_keys 
        self.is_discrete_dict = is_discrete_dict
        self.n_samples = n_samples
        self.tau = tau

        # initializing modules
        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """"""
        # Optimized perturbation representation data 
        self.optimized_perturbation_data = {} 
        
        for perturbation_representation in self.perturbation_representation_keys:
            self.optimized_perturbation_data[perturbation_representation] = torch.randn(n_samples, cond_dim[perturbation_representation])

    def forward(
            self,
            X_controls: torch.Tensor,
        ) -> torch.Tensor:
        # Expand target  and controls
        target = self.optimal_condition.repeat(self.n_samples, X_controls.shape[0], -1)
        X_controls = X_controls.unsqueeze(0).expand(self.n_samples, -1, -1) 
        
        # prepare batch information cellFlow           
        batch_dict = {
            DataFields.SOURCE_STATE: X_controls,
        }
        expanded_perturbation_data = {}
        for pert_key in self.optimized_perturbation_data:
            if not self.is_discrete_dict:
                expanded_perturbation_data[pert_key] = self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[0], -1) 
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[0], -1))
                       
        batch_dict[DataFields.PERTURBATION_DATA] = expanded_perturbation_data
        
        # pushing forward the particles 
        X_pert_pred = self.perturbation_predictor.predict(
            batch_dict,
            no_grad=False,
        )
        
        class_pred = self.linear_classifier(X_pert_pred)
        loss = self.compute_loss(class_pred, target, self.optimized_perturbation_data)
        
        for pert in self.optimized_perturbation_data:
            grad = torch.autograd.grad(loss, self.optimized_perturbation_data[pert], create_graph=False, retain_graph=True)[0]
            
            # Langevin update
            noise = torch.randn_like(self.optimized_perturbation_data[pert]) * self.noise_scale
            self.optimized_perturbation_data[pert] -= (self.eta / 2) * grad + torch.sqrt(torch.tensor(self.eta)) * noise  # In-place update
            self.optimized_perturbation_data[pert].detach_()  # Remove gradients on the just updated element for memory efficiency 
            self.optimized_perturbation_data[pert].requires_grad_()  # Re-enable gradient tracking
        return loss


class NeuralInverseModel(BaseConditionOptimizer):
    """"""
    training_free: bool = False

    def __init__(
        self,
        optimal_condition: dict[str, torch.Tensor], 
        linear_classifier: nn.Module,
        perturbation_predictor: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        cond_dim: int | dict[str, int],
        perturbation_representation_keys: Sequence[str],
        is_discrete_dict: dict[str, bool],
        prior: torch.nn.Module | None = None,
        prior_weight: float | None = None,
        hard: bool = True,
        state_dim: int | None = None,
        perturbation_covariates: Sequence[str] | None = None,
        perturbation_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
        perturbation_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """"""

        super().__init__(
            optimal_condition, 
            linear_classifier,
            perturbation_predictor,
            loss_fn, 
            cond_dim,
            perturbation_representation_keys,
            is_discrete_dict,
            prior,
            prior_weight,
            hard,
        )

        # preparing input with some sanity checks
        msg = f""
        assert perturbation_representation_keys is not None, msg
        if isinstance(perturbation_representation_keys, str):
            perturbation_representation_keys = (perturbation_representation_keys, )

        if perturbation_covariates_noise_models is None:
            perturbation_covariates_noise_models = {covariate: None for covariate in perturbation_representation_keys}

        # setting the optional keyword arguments to a dictionary when not passed
        if perturbation_covariates_predictor_kwargs is None:
            perturbation_covariates_predictor_kwargs = {
                covariate: {} for covariate in perturbation_representation_keys
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
            target_output_dims=self.cond_dim,
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
            if covariate_noise_model == "gaussian":
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

            elif covariate_noise_model == "neg_bin":
                raise NotImplementedError
            
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
        x1_hat = self.perturbation_predictor.predict(
            batch_dict,
            no_grad=False,
        )

        # predicting target response
        class_pred = self.linear_classifier(x1_hat)
        loss = self.compute_loss(class_pred, target, pert_data)

        return loss
