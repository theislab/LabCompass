import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchdiffeq import odeint

import sc_exp_design

class BaseOptimizer(torch.nn.Module):
    """
    Base class for optimization methods using conditional vector fields.

    :param optimal_condition: The target condition for optimization.
    :type optimal_condition: torch.Tensor
    :param linear_classifier: The classifier used to predict outcomes based on perturbed data.
    :type linear_classifier: torch.nn.Module
    :param v_field: The trained velocity field used to model dynamics.
    :type v_field: torch.nn.Module
    :param cond_dim: Dimensionality of the condition vector.
    :type cond_dim: int
    :param prior: Optional prior distribution on the condition vector.
    :type prior: torch.distributions.Distribution | None
    :param prior_weight: Weight applied to the prior term in the loss function.
    :type prior_weight: float | None
    """
    def __init__(self, 
                 optimal_condition, 
                 linear_classifier, 
                 perturbation_predictor,
                 cond_dim, 
                 loss_fn, 
                 prior=None,
                 prior_weight=None):
        
        super().__init__()
    
        self.optimal_condition = optimal_condition
        self.linear_classifier = linear_classifier
        self.perturbation_predictor = perturbation_predictor
        self.cond_dim = cond_dim
        self.loss_fn = loss_fn
        
        self.prior = prior
        self.prior_weight = prior_weight if prior_weight else 1.0
        
    def compute_loss(self, pred, target, e_optimized):
        """
        Compute the loss function, including optional prior regularization.
        """
        loss = self.loss_fn(pred, target)
        if self.prior:
            for key in self.prior: 
                log_prior = self.prior[key].log_prob(e_optimized[key]).sum()
                loss = loss - self.prior_weight * log_prior
        return loss

    def forward(self):
        pass


class MAPConditionOptimizer(BaseOptimizer):
    """
    Performs Maximum A Posteriori (MAP) optimization to find the best condition vector.
    """
    def __init__(self, 
                 optimal_condition, 
                 linear_classifier, 
                 perturbation_predictor, 
                 cond_dim, 
                 loss_fn, 
                 perturbation_representation_keys, 
                 prior=None, 
                 prior_weight=None, 
                 lr=1e-1):
        
        super().__init__(optimal_condition, 
                         linear_classifier,
                         perturbation_predictor, 
                         cond_dim,
                         loss_fn, 
                         prior,
                         prior_weight)
        
        # Perturbation representation keys 
        self.perturbation_representation_keys = perturbation_representation_keys
        self.optimized_perturbation_data = {}
        
        for perturbation_representation in self.perturbation_representation_keys:
            self.optimized_perturbation_data[perturbation_representation] = nn.Parameter(torch.randn(1, cond_dim[perturbation_representation]))
                    
        # Set up optimizer for the conditions 
        self.e_optimizer = optim.Adam([self.optimized_perturbation_data[perturbation_representation] for perturbation_representation in self.optimized_perturbation_data], 
                                      lr=lr)
    
    def forward(self, X_controls):
        # prepare batch information cellFlow           
        batch_dict = {
            sc_exp_design.constants.DataFields.SOURCE_STATE: X_controls,
        }
        expanded_perturbation_data = {pert_key: self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1) for pert_key in self.optimized_perturbation_data}
        batch_dict[sc_exp_design.constants.DataFields.PERTURBATION_DATA] = expanded_perturbation_data

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

    def get_optimized_e(self):
        """
        Returns the optimized condition vector.
        """
        optimized_perturbation_data_tmp = [self.optimized_perturbation_data[pert].detach() for pert in self.optimized_perturbation_data]        
        return optimized_perturbation_data_tmp
    

class LangevinSampler(BaseOptimizer):
    """
    Performs Langevin dynamics sampling to estimate the optimal condition vector distribution.
    """
    def __init__(self, 
                 optimal_condition, 
                 linear_classifier, 
                 perturbation_predictor, 
                 cond_dim,
                 loss_fn, 
                 perturbation_representation_keys,
                 prior,
                 prior_weight,
                 n_samples,
                 eta=1e-1, 
                 noise_scale=1e-1):
        
        super().__init__(optimal_condition, 
                            linear_classifier,
                            perturbation_predictor, 
                            cond_dim,
                            loss_fn, 
                            prior,
                            prior_weight)
            
        self.eta = eta
        self.noise_scale = noise_scale
        self.perturbation_representation_keys = perturbation_representation_keys 
        self.n_samples = n_samples
        
        # Optimized perturbation representation data 
        self.optimized_perturbation_data = {} 
        
        for perturbation_representation in self.perturbation_representation_keys:
            self.optimized_perturbation_data[perturbation_representation] = torch.randn(n_samples, cond_dim[perturbation_representation])

    def forward(self, X_controls):
        # Expand target  and controls
        target = self.optimal_condition.repeat(self.n_samples, X_controls.shape[0], -1)
        X_controls = X_controls.unsqueeze(0).expand(self.n_samples, -1, -1) 
        
        # prepare batch information cellFlow           
        batch_dict = {
            sc_exp_design.constants.DataFields.SOURCE_STATE: X_controls,
        }
        expanded_perturbation_data = {pert_key: self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[0], -1) 
                                      for pert_key in self.optimized_perturbation_data}
        batch_dict[sc_exp_design.constants.DataFields.PERTURBATION_DATA] = expanded_perturbation_data
        
        # pushing forward the particles 
        X_pert_pred = self.perturbation_predictor.predict(batch_dict,
                                                          no_grad=False)
        
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