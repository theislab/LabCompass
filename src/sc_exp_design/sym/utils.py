import numpy as np
import torch
import itertools

import numpy as np
import torch
from sc_exp_design.sym.gmm import AnnotatedGaussianMixtureModel
from sc_exp_design.utils import set_reproducibility

__all__ = ["generate_annotated_perturbation_data"]


def generate_annotated_perturbation_data(sigma, 
                                         d, 
                                         U, 
                                         n_cat, 
                                         N0, 
                                         Nu, 
                                         mean_range=5, 
                                         linespace_width=10, 
                                         uniform_range=5, 
                                         seed=None, 
                                         return_perturbation_representation=False,
                                         heteroskedastic=False,
                                         max_var=5.0,
                                         min_var=1e-4,
                                         ):
    """
    Generate annotated perturbation data using a Gaussian Mixture Model (GMM).

    Parameters:
    sigma (float): Standard deviation for the perturbation noise.
    d (int): Dimensionality of the feature space.
    U (int): Number of perturbation categories.
    n_cat (int): Number of categorical labels for the Gaussian Mixture Model.
    N0 (int): Number of samples for the control (unperturbed) population.
    Nu (int): Number of samples for each perturbed category.
    mean_range (float, optional): Range for selecting mean values of Gaussians. Default is 5.
    linespace_width (int, optional): Number of points in the linspace for mean selection. Default is 10.
    uniform_range (float, optional): Range for sampling category logits. Default is 5.
    seed (int, optional): Random seed for reproducibility. Default is None.

    Returns:
    tuple: (gmm, states, perturbation_ids, categories)
        - gmm (AnnotatedGaussianMixtureModel): The Gaussian Mixture Model instance.
        - states (numpy.ndarray): Array of sampled feature vectors.
        - perturbation_ids (numpy.ndarray): Array indicating perturbation category for each sample.
        - categories (numpy.ndarray): Array of categorical labels for each sample.
    """
    
    # if specified, set the seed for both torch and numpy
    if seed is not None:
        set_reproducibility(seed)
        
    N = Nu*d + N0  # total number of samples
    cov = torch.eye(d)*sigma  # covariance matrix for perturbed distributions

    # Collect mean perturbation shifts
    feature_range = np.linspace(-mean_range, mean_range, linespace_width)
    combinations = np.array(list(itertools.permutations(feature_range.tolist(), 2)))

    # Randomly select `n` combinations
    sampled_means = np.random.choice(len(combinations), U)  # sample combinations of dimension means 
    sampled_means = combinations[sampled_means]

    # sampling the variances when heteroskedastic == True
    trtms_covs = None
    if heteroskedastic:
        sigmas = [min_var + np.random.rand()*max_var for _ in range(U)]
        trtms_covs = [torch.eye(d)*sigma for sigma in sigmas]

    cov = torch.eye(d) * sigma  # covariance matrix for perturbed distributions

    # if not trtms covs if found, use control by default
    if trtms_covs is None:
        trtms_covs = [cov for _ in range(U)]

    mu_array = [torch.tensor(mu).float() for mu in sampled_means]
    params_array = [{"mean": mu, "cov": trtms_covs[idx]} for idx, mu in enumerate(mu_array)]
    cat_logits_lm = torch.rand(d, n_cat) * uniform_range  # logits defining class of interest

    gmm = AnnotatedGaussianMixtureModel(params=params_array,
                       n_cat=n_cat,  
                       cat_logit_lm=cat_logits_lm) 

    perturbation_ids = torch.concatenate([torch.ones((Nu,), dtype=int) * i for i in range(1, U+1)],
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
    
    if return_perturbation_representation:
        sampled_means = torch.cat([torch.zeros(1, d), 
                                   torch.tensor(sampled_means)], dim=0)
        perturbation_representation = sampled_means[perturbation_ids]

    if return_perturbation_representation:
        return gmm, states, perturbation_ids, categories, perturbation_representation
    else:
        return gmm, states, perturbation_ids, categories
