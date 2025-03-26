from collections.abc import Callable, Sequence
from typing import Any, Literal

import anndata
import numpy as np
import torch
import itertools

from sc_exp_design.types import TensorLike
from sc_exp_design.sym.gmm import (
    AnnotatedGaussianMixtureModel,
    DoseResolvedAnnotatedGaussianMixtureModel,
    MultiAttributeAnnotatedGaussianMixtureModel,
)
from sc_exp_design.utils import set_reproducibility

__all__ = ["get_annotated_perturbation_data"] 


def __get_means(
    mean_range: float,
    linespace_width: int,
    U: int,
) -> torch.Tensor:
    """"""
    feature_range = np.linspace(-mean_range, mean_range, linespace_width)
    combinations = np.array(list(itertools.permutations(feature_range.tolist(), 2)))
    trtm_means = np.random.choice(len(combinations), U)  # sample combinations of dimension means 
    trtm_means = combinations[trtm_means]
    trtm_means = torch.from_numpy(trtm_means).float()
    return trtm_means


def __get_covariances(
    d: int,
    sigma: float,
    homoskedastic: bool,
    covariance_type: Literal["isotropic", "anisotropic", "full_covariance"],
    cov_prior: Callable[[Any], TensorLike],
    min_var: float,
    max_var: float,
    U: int | dict[int],
) -> Sequence[torch.Tensor]:
    """"""
    if homoskedastic: # same covariance as control
        return [sigma*torch.eye(d) for _ in range(U)]
    elif covariance_type == "isotropic": # isotropic gaussians
        trtm_covs = [
            min_var + cov_prior()*(max_var - min_var) for _ in range(U)
        ]
        return [
            sigma*torch.eye(d) for sigma in trtm_covs
        ]
    elif covariance_type == "anisotropic": # anisotropic gaussians
        trtm_covs = [
            min_var + cov_prior(d)*(max_var - min_var) for _ in range(U)
        ]
        return [
            torch.from_numpy(np.diag(cov)) for cov in trtm_covs
        ]
    elif covariance_type == "full_covariance": # full covariance matrix
        trtm_covs = ...
        raise NotImplementedError
    else:
        msg = f"{covariance_type=} is not supported, choose among `[\"isotropic\", \"anisotropic\", \"full_covariance\"]`"
        raise ValueError


def __generate_perturbation_data(
    sigma: float,
    d: int,
    U: int | dict[str, int],
    n_cat: int,
    N0: int,
    Nu: int,
    mean_range: float = 5.0,
    linespace_width: int = 10,
    uniform_range: float = 5.0,
    seed: int | None = None,
    return_perturbation_representation: bool = False,
    non_linearity: Callable[[TensorLike], TensorLike] | None = None,
    homoskedastic: bool = True,
    covariance_type: Literal["isotropic", "anisotropic", "full_covariance"] = "isotropic",
    cov_prior: Callable[[Any], TensorLike] = np.random.rand,
    min_var: float = 1e-4,
    max_var: float = 5.0,
    dose_resolved: bool = False,
    dosage_prior: Callable[[Any], TensorLike] = torch.rand,
    interpolation_fn: Callable[[float, TensorLike, TensorLike], TensorLike] | None = None,
    multi_attribute: bool = False,
) -> dict[str, Any]:
    """"""

    # reproducibility
    if seed is not None:
        set_reproducibility(seed)

    # sanity check on the input
    if multi_attribute:
        msg = f""
        assert isinstance(U, dict), msg
    else:
        msg = f""
        assert isinstance(U, int), msg

    # total number of samples
    N = N0 + d*Nu

    # mean for perturbations
    if multi_attribute:
        trtm_means = {
            covariate_label: __get_means(
                mean_range,
                linespace_width,
                u,
            ) for covariate_label, u in U.items()
        }
    else:
        trtm_means = __get_means(
            mean_range,
            linespace_width,
            U,
        )

    # covariance matrix perturbed distribution
    if multi_attribute:
        trtm_covs = {
            covariate_label: __get_covariances(
                d,
                sigma,
                homoskedastic,
                covariance_type,
                cov_prior,
                min_var,
                max_var,
                u,
            ) for covariate_label, u in U.items()
        }
    else:
        trtm_covs = __get_covariances(
            d,
            sigma,
            homoskedastic,
            covariance_type,
            cov_prior,
            min_var,
            max_var,
            U,
        )

    # initializing the parameters
    if multi_attribute:
        trtm_params = {
            covariate_label: [
                {
                    "mean": trtm_means[covariate_label][u].float(),
                    "cov": trtm_covs[covariate_label][u].float(),
                }  for u in range(U_covariate)
            ] for covariate_label, U_covariate in U.items()
        }
    else:
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
            multi_attribute=multi_attribute,
            non_linearity=non_linearity,
        )
    elif multi_attribute:
        gmm = MultiAttributeAnnotatedGaussianMixtureModel(
            params=trtm_params,
            n_cat=n_cat,
            cat_logit_lm=cat_logit_lm,
            non_linearity=non_linearity,
        )
    else:
        gmm = AnnotatedGaussianMixtureModel(
            params=trtm_params,
            n_cat=n_cat,
            cat_logit_lm=cat_logit_lm,
            non_linearity=non_linearity,
        )

    # handling perturbation identifiers
    if multi_attribute:
        trtm_perturbation_ids = {
            covariate_label: torch.concatenate(
                [
                    torch.ones((Nu,), dtype=int) * i for i in range(1, U_covariate+1)
                ],
                dim=0,
            ) for covariate_label, U_covariate in U.items()
        }
        # shuffling the perturbation ids to get combinations of treatments
        trtm_perturbation_ids = {
            covariate: covariate_perturbation_ids[torch.randperm(covariate_perturbation_ids.shape[0])]
            for covariate, covariate_perturbation_ids in trtm_perturbation_ids.items()
        }
        # concatenating with control labels
        perturbation_ids = {
            covariate_label: torch.concatenate(
                (
                    torch.zeros((N0, ), dtype=int),
                    covariate_perturbation_ids,
                )
            ) for covariate_label, covariate_perturbation_ids in trtm_perturbation_ids.items()
        }
    else:
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
    if multi_attribute:
        sampling_perturbation_ids = {
            covariate_label: trtm_perturbation_id - 1
            for covariate_label, trtm_perturbation_id in trtm_perturbation_ids.items()
        }
    else:
        sampling_perturbation_ids = trtm_perturbation_ids - 1

    if dose_resolved:
        target_states, target_categories, target_dosages = gmm.sample(comps=sampling_perturbation_ids)
    else:
        target_states, target_categories = gmm.sample(comps=sampling_perturbation_ids)

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
        if multi_attribute:
            trtm_means = {
                covariate_label: torch.concatenate(
                    (
                        torch.zeros(1, d),
                        trtm_mean,
                    ),
                    dim=0,
                ) for covariate_label, trtm_mean in trtm_means.items()
            }
        else:
            trtm_means = torch.concatenate(
                (
                    torch.zeros(1, d),
                    trtm_means,
                ),
                dim=0,
            ) 

    # (optional) hadling the dosages
    if dose_resolved:
        if multi_attribute:
            dosages = {
                covariate_label: torch.concatenate(
                    (
                        torch.zeros((N0, )),
                        target_dosage,
                    ),
                    dim=0,
                )
                for covariate_label, target_dosage in target_dosages.items()
            }
        else:
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
    categories = categories[random_perm_idx].numpy()
    if multi_attribute:
        perturbation_ids = {
            covariate_label: perturbation_id[random_perm_idx].numpy()
            for covariate_label, perturbation_id in perturbation_ids.items()
        }
        if return_perturbation_representation:
            trtm_means = {
                covariate_label: trtm_mean[perturbation_ids[covariate_label]].numpy()
                for covariate_label, trtm_mean in trtm_means.items()
            }
        if dose_resolved:
            dosages = {
                covariate_label: dosage[random_perm_idx]
                for covariate_label, dosage in dosages.items()
            }
    else:
        perturbation_ids = perturbation_ids[random_perm_idx].numpy()
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
    U: int | dict[str, int],
    n_cat: int,
    N0: int,
    Nu: int,
    mean_range: float = 5.0,
    linespace_width: int = 10,
    uniform_range: float = 5.0,
    seed: int | None = None,
    return_perturbation_representation: bool = False,
    non_linearity: Callable[[TensorLike], TensorLike] | None = None,
    homoskedastic: bool = True,
    covariance_type: Literal["isotropic", "anisotropic", "full_covariance"] = "isotropic",
    cov_prior: Callable[[Any], TensorLike] = np.random.rand,
    min_var: float = 1e-4,
    max_var: float = 5.0,
    dose_resolved: bool = False,
    dosage_prior: Callable[[Any], TensorLike] = torch.rand,
    interpolation_fn: Callable[[float, TensorLike, TensorLike], TensorLike] | None = None,
    multi_attribute: bool = False,
    control_label: str = "control",
    treatment_label: str = "treatment",
    category_label: str = "cell_type",
) -> tuple[anndata.AnnData, dict[str, Any]]:
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
        non_linearity=non_linearity,
        homoskedastic=homoskedastic,
        covariance_type=covariance_type,
        cov_prior=cov_prior,
        min_var=min_var,
        max_var=max_var,
        dose_resolved=dose_resolved,
        dosage_prior=dosage_prior,
        interpolation_fn=interpolation_fn,
        multi_attribute=multi_attribute,
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
    if multi_attribute:
        perturbation_ids_to_labels = {
            covariate_label: {
                0: control_label,
                **{
                    idx: f"{treatment_label}_{idx}" for idx in range(1, U_covariate + 1)
                }
            } for covariate_label, U_covariate in U.items()
        }
        perturbation_labels_to_ids = {
            covariate_label: {v:np.array([k]) for k, v in perturbation_ids_to_labels[covariate_label].items()}
            for covariate_label, perturbation_ids_to_label in perturbation_ids_to_labels.items()
        }
        perturbation_labels = {
            covariate_label: np.vectorize(perturbation_ids_to_label.get)(perturbation_ids[covariate_label])
            for covariate_label, perturbation_ids_to_label in perturbation_ids_to_labels.items()
        }
        # one hot encoding of the treatments
        perturbation_labels_one_hot = {
            covariate_label: {
                perturbation_label: np.zeros(U[covariate_label] + 1) for perturbation_label in perturbation_labels_to_ids[covariate_label].keys()#.keys()
            } for covariate_label in perturbation_labels.keys()
        }
        for perturbation_label in perturbation_labels_one_hot.values():
            for idx, perturbation_one_hot in enumerate((perturbation_label.values())):
                np.put(perturbation_one_hot, [idx], [1])
    else:
        perturbation_ids_to_labels = {
            0: control_label,
            **{
                idx: f"{treatment_label}_{idx}" for idx in range(1, U + 1)
            }
        }
        perturbation_labels_to_ids = {v:np.array([k]) for k, v in perturbation_ids_to_labels.items()}
        perturbation_labels = np.vectorize(perturbation_ids_to_labels.get)(perturbation_ids)
        # one hot encoding of the treatments
        perturbation_labels_one_hot = {
            str(perturbation_label): np.zeros(U + 1) for perturbation_label in perturbation_labels#.keys()
        }
        for idx, perturbation_one_hot in enumerate(perturbation_labels_one_hot.values()):
            np.put(perturbation_one_hot, [idx], [1])

    # annotating the category data
    category_ids_to_labels = {
        idx: f"{category_label}_{idx}" for idx in range(n_cat)
    }
    category_labels_to_ids = {v:np.array([k]) for k, v in category_ids_to_labels.items()}
    category_labels = np.vectorize(category_ids_to_labels.get)(categories)

    # retrieving perturbation shift
    if multi_attribute:
        perturbation_shift = {
            covariate_label: {
                control_label: torch.zeros((d)).numpy(),
                **{
                    perturbation_ids_to_label[(idx + 1)]: comp["mean"].numpy() for idx, comp in enumerate(gmm.parameters[covariate_label])
                } 
            } for covariate_label, perturbation_ids_to_label in perturbation_ids_to_labels.items()
        }
    else:
        perturbation_shift = {
            control_label: torch.zeros((d)).numpy(),
            **{
                perturbation_ids_to_labels[(idx + 1)]: comp["mean"].numpy() for idx, comp in enumerate(gmm.parameters)
            }
        }

    # handling control flag
    if multi_attribute:
        # zipping the perturbation labels for each covariate in a single tuple to check
        # if all of the covariates are equal to control
        zipped_perturbation_labels = list(zip(*list(perturbation_labels.values())))
        is_control = np.array([
            tuple(obs_perturbation_labels) == tuple([control_label]*len(obs_perturbation_labels))
            for obs_perturbation_labels in zipped_perturbation_labels
        ]).astype(int)
    else:
        is_control = (perturbation_labels==control_label).astype(int)

    if multi_attribute:
        # handling obs attribute of annotated data
        obs = {
            category_label: category_labels,
            control_label: is_control,
            **{
                covariate_label: covariate_perturbation_label
                for covariate_label, covariate_perturbation_label in perturbation_labels.items()
            }
        }
        if dose_resolved:
            obs.update(
                {
                    f"{covariate_label}_dose": dosage for covariate_label, dosage in dosages.items()
                }
            )

        # handling uns attribute of annotated data        
        uns = {
            f"{category_label}_label": category_labels_to_ids,
            **{
                f"{covariate_label}_shift": covariate_perturbation_shift
                for covariate_label, covariate_perturbation_shift in perturbation_shift.items()
            },
            **{
                f"{covariate_label}_label": covariate_label_id
                for covariate_label, covariate_label_id in perturbation_labels_to_ids.items()
            },
            **{
                f"{covariate_label}_one_hot": covariate_one_hot
                for covariate_label, covariate_one_hot in perturbation_labels_one_hot.items()
            }
        }
    else:
        obs = {
            treatment_label: perturbation_labels,
            category_label: category_labels,
            control_label: is_control,
        }
        if dose_resolved:
            obs["dose"] = dosages

        # handling uns attribute of annotated data
        uns = {
            f"{treatment_label}_labels": perturbation_labels_to_ids,
            f"{treatment_label}_shift": perturbation_shift,
            f"{treatment_label}_one_hot": perturbation_labels_one_hot,
            f"{category_label}_label": category_labels_to_ids,
        }

    # initializing annotated data
    adata = anndata.AnnData(
        X=states,
        obs=obs,
        uns=uns,
    )
    return adata, sym_dictionary
