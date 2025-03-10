from collections.abc import Callable
from typing import Any, Literal

import anndata
import numpy as np
import torch
import itertools

import numpy as np
import torch

from sc_exp_design.types import TensorLike
from sc_exp_design.sym.gmm import AnnotatedGaussianMixtureModel, DoseResolvedAnnotatedGaussianMixtureModel
from sc_exp_design.utils import set_reproducibility

__all__ = ["get_annotated_perturbation_data"] 


def __generate_perturbation_data(
    sigma: float,
    d: int,
    U: int,
    n_cat: int,
    N0: int,
    Nu: int,
    mean_range: float = 5.0,
    linespace_width: int = 10,
    uniform_range: float = 5.0,
    seed: int | None = None,
    return_perturbation_representation: bool = False,
    homoskedastic: bool = True,
    covariance_type: Literal["isotropic", "anisotropic", "full_covariance"] = "isotropic",
    cov_prior: Callable[[Any], TensorLike] = np.random.rand,
    min_var: float = 1e-4,
    max_var: float = 5.0,
    dose_resolved: bool = False,
    dosage_prior: Callable[[Any], TensorLike] = torch.rand,
    interpolation_fn: Callable[[float, TensorLike, TensorLike], TensorLike] | None = None,
) -> dict[str, Any]:
    """"""

    # reproducibility
    if seed is not None:
        set_reproducibility(seed)

    # total number of samples
    N = N0 + d*Nu

    # mean for perturbations
    feature_range = np.linspace(-mean_range, mean_range, linespace_width)
    combinations = np.array(list(itertools.permutations(feature_range.tolist(), 2)))
    trtm_means = np.random.choice(len(combinations), U)  # sample combinations of dimension means 
    trtm_means = combinations[trtm_means]
    trtm_means = torch.from_numpy(trtm_means).float()

    # covariance matrix perturbed distribution
    if homoskedastic: # same covariance as control
        trtm_covs = [sigma*torch.eye(d) for _ in range(U)]
    elif covariance_type == "isotropic": # isotropic gaussians
        trtm_covs = [
            min_var + cov_prior()(max_var - min_var) for _ in range(U)
        ]
        trtm_covs = [
            sigma*torch.eye(d) for sigma in trtm_covs
        ]
    elif covariance_type == "anisotropic": # anisotropic gaussians
        trtm_covs = [
            min_var + cov_prior(d)(max_var - min_var) for _ in range(U)
        ]
        trtm_covs = [
            np.diag(cov) for cov in trtm_covs
        ]
    elif covariance_type == "full_covariance": # full covariance matrix
        trtm_covs = ...
        raise NotImplementedError
    else:
        msg = f"{covariance_type=} is not supported, choose among `[\"isotropic\", \"anisotropic\", \"full_covariance\"]`"
        raise ValueError

    # initializing the parameters
    trtm_params = [
        {
            "mean": trtm_means[u].float(),
            "cov": trtm_covs[u].float(),
        }  for u in range(U)
    ]

    # sampling the logits
    cat_logit_lm = torch.rand(d, n_cat) * uniform_range  # logits defining class of interest

    # initializing GMM
    if dose_resolved:
        gmm = DoseResolvedAnnotatedGaussianMixtureModel(
            params=trtm_params,
            n_cat=n_cat,
            cat_logit_lm=cat_logit_lm,
            dosage_prior=dosage_prior,
            interpolation_fn=interpolation_fn,
        )
    else:
        gmm = AnnotatedGaussianMixtureModel(
            params=trtm_params,
            n_cat=n_cat,
            cat_logit_lm=cat_logit_lm,
        )

    # handling perturbation identifiers
    trtm_perturbation_ids = torch.concatenate(
        [
            torch.ones((Nu,), dtype=int) * i for i in range(1, U+1)
        ],
        dim=0,
    )
    perturbation_ids = torch.concatenate(
        (
            torch.zeros((N0, ), dtype=int),
            trtm_perturbation_ids,
        ),
        dim=0,
    )

    # sampling the treatment data
    if dose_resolved:
        target_states, target_categories, target_dosages = gmm.sample(comps=(trtm_perturbation_ids - 1))
    else:
        target_states, target_categories = gmm.sample(comps=(trtm_perturbation_ids - 1))

    # sampling the control data
    source_states = torch.randn((N0, d))*sigma
    source_categories = gmm.sample_categories(source_states)

    # handling the states
    states = torch.concatenate(
        (
            source_states,
            target_states,
        ),
        dim=0,
    )

    # handling the categories
    categories = torch.concatenate(
        (
            source_categories,
            target_categories,
        ),
        dim=0,
    )

    # (optional) handling the perturbation representation
    if return_perturbation_representation:
        trtm_means = torch.concatenate(
            (
                torch.zeros(1, d),
                trtm_means,
            ),
            dim=0,
        ) 

    # (optional) hadling the dosages
    if dose_resolved:
        dosages = torch.concatenate(
            (
                torch.zeros((N0, )),
                target_dosages,
            ),
            dim=0,
        )

    # shuffling the data
    random_perm_idx = torch.randperm(states.shape[0])
    states = states[random_perm_idx].numpy()
    perturbation_ids = perturbation_ids[random_perm_idx].numpy()
    categories = categories[random_perm_idx].numpy()
    if return_perturbation_representation:
        trtm_means = trtm_means[perturbation_ids].numpy()
    if dose_resolved:
        dosages = dosages[random_perm_idx].numpy()
    
    # constructing output dictionary
    out = {
        "gmm": gmm,
        "states": states,
        "perturbation_ids": perturbation_ids,
        "categories": categories,
    }
    if return_perturbation_representation:
        out["treatment_means"] = trtm_means
    if dose_resolved:
        out["dosages"] = dosages
    return out


def get_annotated_perturbation_data(
    sigma: float,
    d: int,
    U: int,
    n_cat: int,
    N0: int,
    Nu: int,
    mean_range: float = 5.0,
    linespace_width: int = 10,
    uniform_range: float = 5.0,
    seed: int | None = None,
    return_perturbation_representation: bool = False,
    homoskedastic: bool = True,
    covariance_type: Literal["isotropic", "anisotropic", "full_covariance"] = "isotropic",
    cov_prior: Callable[[Any], TensorLike] = np.random.rand,
    min_var: float = 1e-4,
    max_var: float = 5.0,
    dose_resolved: bool = False,
    dosage_prior: Callable[[Any], TensorLike] = torch.rand,
    interpolation_fn: Callable[[float, TensorLike, TensorLike], TensorLike] | None = None,
    control_label: str = "control",
    treatment_label: str = "treatment",
    category_label: str = "cell_type",
) -> anndata.AnnData:
    """"""
    # generating data
    sym_dictionary = __generate_perturbation_data(
        sigma,
        d,
        U,
        n_cat,
        N0,
        Nu,
        mean_range=mean_range,
        linespace_width=linespace_width,
        uniform_range=uniform_range,
        seed=seed,
        return_perturbation_representation=return_perturbation_representation,
        homoskedastic=homoskedastic,
        covariance_type=covariance_type,
        cov_prior=cov_prior,
        min_var=min_var,
        max_var=max_var,
        dose_resolved=dose_resolved,
        dosage_prior=dosage_prior,
        interpolation_fn=interpolation_fn,
    )

    # parsing output dictionary
    gmm = sym_dictionary["gmm"]
    states = sym_dictionary["states"]
    perturbation_ids = sym_dictionary["perturbation_ids"]
    categories = sym_dictionary["categories"]
    if return_perturbation_representation:
        treatment_means = sym_dictionary["treatment_means"]
    if dose_resolved:
        dosages = sym_dictionary["dosages"]

    # annotating the perturbation data
    perturbation_ids_to_labels = {
        0: control_label,
        **{
            idx: f"{treatment_label}_{idx}" for idx in range(1, U + 1)
        }
    }
    perturbation_labels_to_ids = {v:np.array([k]) for k, v in perturbation_ids_to_labels.items()}
    perturbation_labels = np.vectorize(perturbation_ids_to_labels.get)(perturbation_ids)

    # annotating the category data
    category_ids_to_labels = {
        idx: f"{category_label}_{idx}" for idx in range(n_cat)
    }
    category_labels_to_ids = {v:np.array([k]) for k, v in category_ids_to_labels.items()}
    category_labels = np.vectorize(category_ids_to_labels.get)(categories)

    # retrieving perturbation shift
    perturbation_shift = {
        control_label: torch.zeros((d)),
        **{
            perturbation_ids_to_labels[(idx + 1)]: comp["mean"] for idx, comp in enumerate(gmm.params)
        }
    }

    # handling obs attribute of annotated data
    obs = {
        treatment_label: perturbation_labels,
        category_label: category_labels,
        control_label: (perturbation_labels=="control").astype(int),
    }
    if dose_resolved:
        obs["dose"] = dosages

    # handling uns attribute of annotated data
    uns = {
        f"{treatment_label}_labels": perturbation_labels_to_ids,
        f"{treatment_label}_shift": perturbation_shift,
        f"{category_label}_label": category_labels_to_ids,
    }

    # initializing annotated data
    adata = anndata.AnnData(
        X=states,
        obs=obs,
        uns=uns,
    )
    return adata
