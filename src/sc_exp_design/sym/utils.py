import numpy as np
import torch
import itertools

import numpy as np
import torch
from sc_exp_design.sym.gmm import AnnotatedGaussianMixtureModel
from sc_exp_design.utils import set_reproducibility

def generate_annotated_perturbation_data(
        sigma: float,
        d: int,
        U: int,
        n_cat: int,
        N0: int,
        Nu: int,
        mean_range: int = 5,
        linespace_width: int = 10,
        uniform_range: int = 5,
        seed: int | None = None,
    ) -> tuple[AnnotatedGaussianMixtureModel, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate observations
    """
    if seed is not None:
        set_reproducibility(seed)
        
    N = Nu*d + N0  # total number of samples
    cov = torch.eye(d)*sigma  # covariance matrix for perturbed distributions

    # Randomly draw the means 
    feature_range = np.linspace(-mean_range, mean_range, linespace_width)
    combinations = np.array(list(itertools.combinations(feature_range.tolist(), 2)))

    # Randomly select `n` combinations
    sampled_combinations = np.random.choice(len(combinations), U)
    sampled_combinations = combinations[sampled_combinations]

    mu_array = [torch.tensor(mu).float() for mu in sampled_combinations]
    params_array = [{"mean": mu, "cov": cov} for mu in mu_array]
    cat_logits_lm = torch.rand(d, n_cat)*uniform_range

    gmm = AnnotatedGaussianMixtureModel(params=params_array,
                       n_cat=n_cat,  
                       cat_logit_lm=cat_logits_lm) 

    perturbation_ids = torch.concatenate([torch.ones((Nu, ), dtype=int) * i for i in range(1, U+1)],
                                        dim=0)
    
    # Sample observations 
    target_samples, target_categories = gmm.sample(comps=(perturbation_ids - 1))  # sampling from the perturbed population
    source_samples = torch.randn((N0, d))*sigma  # sampling from the control population
    source_categories = gmm.sample_categories(source_samples)

    perturbation_ids = torch.concatenate(
        (torch.zeros((N0, ), dtype=int),  # id for no perturbation (control)
            perturbation_ids,  # rest of perturbations
        ), dim=0)
    
    # sample ids 
    states = torch.concatenate(
        (source_samples,  # source states
         target_samples,  # target states
        ), dim=0)
    
    # categories 
    categories = torch.concatenate(
        (source_categories, # source states
         target_categories, # target states
        ), dim=0)

    random_perm_idx = torch.randperm(states.shape[0])
    states = states[random_perm_idx].numpy()
    perturbation_ids = perturbation_ids[random_perm_idx].numpy()
    categories = categories[random_perm_idx].numpy()

    return gmm, states, perturbation_ids, categories
