import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sc_exp_design

class BaseOptimizer(torch.nn.Module):
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
        loss = self.loss_fn(pred, target)
        if self.prior:
            for key in self.prior: 
                log_prior = self.prior[key].log_prob(e_optimized[key]).sum()
                loss = loss - self.prior_weight * log_prior
        return loss

    def forward(self):
        pass
    
    def get_optimized_e(self):
        optimized_perturbation_data_tmp = [self.optimized_perturbation_data[pert].detach() for pert in self.optimized_perturbation_data]        
        return optimized_perturbation_data_tmp
    
    def differentiable_categorical(self, logits, tau=1.0, hard=False):
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Gumbel-Softmax reparameterization
        y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)

        if hard:
            # Convert to hard one-hot, but keep gradients
            y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
            return y_hard + (y_soft - y_hard).detach()  # Keep gradients

        return y_soft


class MAPConditionOptimizer(BaseOptimizer):
    def __init__(self, 
                 optimal_condition, 
                 linear_classifier, 
                 perturbation_predictor, 
                 cond_dim, 
                 loss_fn, 
                 perturbation_representation_keys, 
                 is_discrete_dict, 
                 prior=None, 
                 prior_weight=None, 
                 lr=1e-1, 
                 tau=1.0, 
                 hard=True):
        
        super().__init__(optimal_condition, 
                         linear_classifier,
                         perturbation_predictor, 
                         cond_dim,
                         loss_fn, 
                         prior,
                         prior_weight)
        
        self.is_discrete_dict = is_discrete_dict
        self.tau = tau
        self.hard = hard
        
        # perturbation representation keys 
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
        
        expanded_perturbation_data = {}
        for pert_key in self.optimized_perturbation_data:
            if not self.is_discrete_dict:
                expanded_perturbation_data[pert_key] = self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1) 
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].expand(X_controls.shape[0], -1))
            
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
    
    
class LangevinSampler(BaseOptimizer):
    def __init__(self, 
                 optimal_condition, 
                 linear_classifier, 
                 perturbation_predictor, 
                 cond_dim,
                 loss_fn, 
                 perturbation_representation_keys,
                 is_discrete_dict, 
                 prior,
                 prior_weight,
                 n_samples,
                 eta=1e-1, 
                 noise_scale=1e-1, 
                 tau=1.0, 
                 hard=True):
        
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
        self.is_discrete_dict = is_discrete_dict
        self.n_samples = n_samples
        self.tau = tau
        self.hard = hard
        
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
        expanded_perturbation_data = {}
        for pert_key in self.optimized_perturbation_data:
            if not self.is_discrete_dict:
                expanded_perturbation_data[pert_key] = self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[0], -1) 
            else:
                expanded_perturbation_data[pert_key] = self.differentiable_categorical(self.optimized_perturbation_data[pert_key].unsqueeze(1).expand(-1, X_controls.shape[0], -1))
                       
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
    