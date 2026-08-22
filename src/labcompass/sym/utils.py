from collections.abc import Callable, Sequence
from typing import Any, Literal

import anndata
import numpy as np
import torch
import itertools

from labcompass.types import TensorLike
from labcompass.sym.gmm import (
    AnnotatedGaussianMixtureModel,
    DoseResolvedAnnotatedGaussianMixtureModel,
    MultiAttributeAnnotatedGaussianMixtureModel,
)
from labcompass.utils import set_reproducibility

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
    n_cat: int | dict[str, int],
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

    if isinstance(uniform_range, int | float):
        uniform_range = {cat_id: uniform_range for cat_id in n_cat.keys()}
    msg = f""
    assert isinstance(uniform_range, dict), msg

    if isinstance(non_linearity, Callable):
        non_linearity = {cat_id: non_linearity for cat_id in n_cat.keys()}

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
    cat_logit_lm = {cat_id: torch.rand(d, cat_dim) * uniform_range[cat_id] for cat_id, cat_dim in n_cat.items()}  # logits defining class of interest

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
    categories = {
        cat_id: torch.concatenate(
            (
                source_categories[cat_id],
                target_categories[cat_id],
            ),
            dim=0,
        ) for cat_id in source_categories.keys()
    }
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
    categories = {cat_id: category[random_perm_idx].numpy() for cat_id, category in categories.items()}
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
    n_cat: int | dict[str, int],
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
    """Generates a synthetic single-cell perturbation dataset from a Gaussian mixture model and wraps it in an :class:`AnnData` object.

    Control cells are sampled as isotropic Gaussian noise (scaled by :param:`sigma`) around the origin, while perturbed cells
    are sampled from Gaussian components (one per unique perturbation) whose means are drawn from a grid of `linespace_width`
    values in `[-mean_range, mean_range]`. Each cell is additionally assigned categorical labels (e.g. cell types) sampled from
    a linear-logit model of its state. When :param:`multi_attribute` is `True`, several perturbation covariates are generated
    simultaneously (one Gaussian mixture per covariate, combined additively); when :param:`dose_resolved` is `True`, a continuous
    dosage per cell interpolates its state between the control mean and the perturbed component mean.

    :param sigma: Standard deviation used to sample control cells (`torch.randn(...) * sigma`) and, when :param:`homoskedastic`
        is `True`, the fixed variance of every perturbed component's covariance (`sigma * eye(d)`).
    :type sigma: class:`float`

    :param d: Dimensionality of the generated cell states.
    :type d: class:`int`

    :param U: Number of unique perturbations (excluding control) to generate. When :param:`multi_attribute` is `True`, a
        dictionary mapping each covariate label to its own number of unique perturbations.
    :type U: class:`int | dict[str, int]`

    :param n_cat: Number of categorical labels to sample per cell. When an :class:`int` is passed, it is used as the number
        of categories for a single covariate named `"cell_type"`; otherwise a dictionary mapping each categorical covariate
        name to its number of categories.
    :type n_cat: class:`int | dict[str, int]`

    :param N0: Number of control cells to generate.
    :type N0: class:`int`

    :param Nu: Number of cells to generate for each unique perturbation.
    :type Nu: class:`int`

    :param mean_range: Half-width of the range `[-mean_range, mean_range]` from which the perturbation component means are
        drawn, defaults to `5.0`.
    :type mean_range: class:`float`

    :param linespace_width: Number of candidate values per dimension in the grid used to sample perturbation component means,
        defaults to `10`.
    :type linespace_width: class:`int`

    :param uniform_range: Scale applied to the randomly sampled logit matrix used to assign categorical labels from cell
        states, defaults to `5.0`.
    :type uniform_range: class:`float`

    :param seed: Random seed used to make the generation reproducible via `labcompass.utils.set_reproducibility`, defaults to
        `None` in which case no seed is set.
    :type seed: class:`int | None`

    :param return_perturbation_representation: Whether to include the perturbation mean vectors (`"treatment_means"`) in the
        returned dictionary, defaults to `False`.
    :type return_perturbation_representation: class:`bool`

    :param non_linearity: Non-linearity applied to the cell states before computing the categorical logits, defaults to `None`
        in which case the identity function is used.
    :type non_linearity: class:`Callable[[TensorLike], TensorLike] | None`

    :param homoskedastic: Whether every perturbed component shares the same covariance as the control distribution
        (`sigma * eye(d)`). When `False`, covariances are instead sampled according to :param:`covariance_type`, defaults to
        `True`.
    :type homoskedastic: class:`bool`

    :param covariance_type: The type of covariance matrix sampled for each perturbed component when :param:`homoskedastic` is
        `False`. `"full_covariance"` is not yet implemented, defaults to `"isotropic"`.
    :type covariance_type: class:`Literal["isotropic", "anisotropic", "full_covariance"]`

    :param cov_prior: Function used to sample the (unscaled) variances of the perturbed components' covariances, defaults to
        `np.random.rand`.
    :type cov_prior: class:`Callable[[Any], TensorLike]`

    :param min_var: Lower bound used to rescale the variances sampled by :param:`cov_prior`, defaults to `1e-4`.
    :type min_var: class:`float`

    :param max_var: Upper bound used to rescale the variances sampled by :param:`cov_prior`, defaults to `5.0`.
    :type max_var: class:`float`

    :param dose_resolved: Whether to additionally sample a continuous dosage per perturbed cell and interpolate its state
        between the control mean and the perturbed component mean, defaults to `False`.
    :type dose_resolved: class:`bool`

    :param dosage_prior: Function used to sample the per-cell dosages when :param:`dose_resolved` is `True`, defaults to
        `torch.rand`.
    :type dosage_prior: class:`Callable[[Any], TensorLike]`

    :param interpolation_fn: Function used to interpolate a cell's state between the control mean and the perturbed component
        mean given its dosage. Only used when :param:`dose_resolved` is `True`, defaults to `None` in which case linear
        interpolation `(1 - dose) * source + dose * target` is used.
    :type interpolation_fn: class:`Callable[[float, TensorLike, TensorLike], TensorLike] | None`

    :param multi_attribute: Whether to generate several perturbation covariates simultaneously, each with its own Gaussian
        mixture, defaults to `False`. When `True`, :param:`U` must be a dictionary.
    :type multi_attribute: class:`bool`

    :param control_label: Label assigned to control cells in the returned :class:`AnnData`, defaults to `"control"`.
    :type control_label: class:`str`

    :param treatment_label: Prefix used to build perturbation labels (e.g. `"treatment_1"`) and the corresponding `.obs`/`.uns`
        keys when :param:`multi_attribute` is `False`, defaults to `"treatment"`.
    :type treatment_label: class:`str`

    :param category_label: Unused when :param:`n_cat` is a dictionary; only relevant as the categorical covariate name when
        :param:`n_cat` is an :class:`int`, in which case `"cell_type"` is used instead, defaults to `"cell_type"`.
    :type category_label: class:`str`

    :return: A tuple `(adata, sym_dictionary)` where `adata` is an :class:`AnnData` with `.X` set to the generated cell states,
        `.obs` containing the perturbation/control labels and categorical labels (and dosages if :param:`dose_resolved` is
        `True`), and `.uns` containing the label-to-id mappings, one-hot encodings and perturbation mean shifts; `sym_dictionary`
        is the raw dictionary produced during generation (keys `"gmm"`, `"states"`, `"perturbation_ids"`, `"categories"`, and
        optionally `"treatment_means"` / `"dosages"`).
    :rtype: class:`tuple[anndata.AnnData, dict[str, Any]]`
    """

    if isinstance(n_cat, int):
        n_cat = {"cell_type": n_cat}
    msg = f""
    assert isinstance(n_cat, dict), msg

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
        cat_id: {
            idx: f"{cat_id}_{idx}" for idx in range(n_cat[cat_id])
        } for cat_id in categories.keys()
    }
    category_labels_to_ids = {
        cat_id: {
            v:np.array([k]) for k, v in category_ids_to_labels[cat_id].items()
        } for cat_id in categories.keys()
    }
    category_labels = {
        cat_id: np.vectorize(category_ids_to_labels[cat_id].get)(categories[cat_id]) for cat_id in categories.keys()
    }

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
            control_label: is_control,
            **{
                cat_id: cat_labels for cat_id, cat_labels in category_labels.items()
            },
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
            **{
                f"{cat_id}_label": cat_labels for cat_id, cat_labels in category_labels_to_ids.items()
            },
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
            control_label: is_control,
            **{
                cat_id: cat_labels for cat_id, cat_labels in category_labels.items()
            },

        }
        if dose_resolved:
            obs["dose"] = dosages

        # handling uns attribute of annotated data
        uns = {
            f"{treatment_label}_labels": perturbation_labels_to_ids,
            f"{treatment_label}_shift": perturbation_shift,
            f"{treatment_label}_one_hot": perturbation_labels_one_hot,
            **{
                f"{cat_id}_label": cat_labels for cat_id, cat_labels in category_labels_to_ids.items()
            },
        }

    # initializing annotated data
    adata = anndata.AnnData(
        X=states,
        obs=obs,
        uns=uns,
    )
    return adata, sym_dictionary
