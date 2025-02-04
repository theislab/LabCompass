from collections.abc import Sequence

import anndata
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

from sc_exp_design.sym import GaussianMixtureModel

__all__ = ["create_gmm_anndata", "create_gmm_anndata_with_shifts", "create_gmm_data"]


def create_gmm_anndata(
    num_ctrl_obs: int,
    num_trtm1_obs: int,
    num_trtm2_obs: int,
    mu_ctrl: Tensor = torch.tensor([0.0, 0.0]),
    cov_ctrl: Tensor = torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    mu_1: Tensor = torch.tensor([5.0, 5.0]),
    mu_2: Tensor = torch.tensor([-5.0, -5.0]),
    cov_trtm: Tensor = torch.tensor([[1.0, 0.5], [0.5, 1.0]]),
) -> anndata.AnnData:
    """"""
    # control distribution
    ctrl_dist = MultivariateNormal(mu_ctrl, cov_ctrl)
    params = [{"mean": mu_1, "cov": cov_trtm}, {"mean": mu_2, "cov": cov_trtm}]
    trtm_dist = GaussianMixtureModel(params)
    # defining the perturbations
    ctrl = -torch.ones((num_ctrl_obs,), dtype=torch.int16)  # encoding control with -1
    pert1 = torch.zeros((num_trtm1_obs,), dtype=torch.int16)
    pert2 = torch.ones((num_trtm2_obs,), dtype=torch.int16)
    perts = torch.concatenate((pert1, pert2))
    # sampling from control and treatments
    ctrl_obs = ctrl_dist.sample((num_ctrl_obs,))
    trtm_obs = trtm_dist.sample(comps=perts)
    # concatenating the data
    conditions = torch.concatenate((ctrl, perts))
    states = torch.concatenate((ctrl_obs, trtm_obs))
    # constructing obs dataframe
    obs = pd.DataFrame({"cond": conditions.tolist()})
    obs["control"] = obs["cond"] == -1
    obs["cond"] = obs["cond"].map({-1: "control", 0: "treatment_0", 1: "treatment_1"})
    # defining uns
    pert_rep_dict = {
        "control": np.array([0]),
        "treatment_0": np.array([1]),
        "treatment_1": np.array([2]),
    }
    # instantiating AnnData object
    adata = anndata.AnnData(X=states.numpy(), obs=obs, uns={"pert_rep_dict": pert_rep_dict})
    return adata


def create_gmm_anndata_with_shifts(
    num_ctrl_obs: int,
    num_trtm1_obs: int,
    num_trtm2_obs: int,
    var_ctrl: float = 2e-1,
    mu_1: Tensor = torch.tensor([1.0, 1.0]),
    mu_2: Tensor = torch.tensor([-1.0, -1.0]),
    cov_trtm1: Tensor = torch.tensor([[0.1, 0.0], [0.0, 0.1]]),
    cov_trtm2: Tensor = torch.tensor([[0.1, 0.0], [0.0, 0.1]]),
) -> anndata.AnnData:
    """"""
    # sampling perturbations
    perturbations_distributions = GaussianMixtureModel(
        params=[
            {
                "mean": mu_1,
                "cov": cov_trtm1,
            },
            {
                "mean": mu_2,
                "cov": cov_trtm2,
            },
        ]
    )

    treatments = torch.concatenate((torch.zeros((num_trtm1_obs)), torch.ones(num_trtm2_obs)), dim=0).int()

    perturbations = perturbations_distributions.sample(comps=treatments)

    controls = torch.zeros((num_ctrl_obs, 2))
    perturbations = torch.concatenate((controls, perturbations), dim=0)

    # # control flag
    is_control = torch.concatenate(
        (torch.ones((num_ctrl_obs)), torch.zeros(num_trtm1_obs + num_trtm2_obs)), dim=0
    ).numpy()

    # # treatment id
    treatments = torch.concatenate((-torch.ones((num_ctrl_obs)), treatments), dim=0).numpy()

    treatment_map = {
        -1: "control",
        0: "treatment_0",
        1: "treatment_1",
    }
    treatment_ids = np.vectorize(treatment_map.get)(treatments)

    # # sampling initial population from standard normal
    num_samples = num_ctrl_obs + num_trtm1_obs + num_trtm2_obs
    initial_population = torch.randn((num_samples, 2)) * var_ctrl

    # keeping the first 20_000 observations as control the rest as treatment
    perturbed_population = initial_population - perturbations

    # constructing input AnnData object

    obs = pd.DataFrame({"is_control": is_control, "cond": treatment_ids})

    obsm = {"shift": perturbations.numpy(), "original_state": initial_population.numpy()}

    uns = {
        "rep": {
            "control": np.array([0]),
            "treatment_0": np.array([1]),
            "treatment_1": np.array([2]),
        }
    }

    train_adata = anndata.AnnData(
        X=perturbed_population.numpy(),
        obs=obs,
        obsm=obsm,
        uns=uns,
    )
    return train_adata


def create_gmm_data(
    num_components: int = 8,
    num_train_control_obs: int = 30_000,
    num_train_treatment_obs: int = 80_000,
    num_test_control_obs: int = 5_000,
    num_test_treatment_obs: int = 20_000,
    base_params: Sequence[dict[str, Tensor]] = [
        {
            "mean": torch.tensor([5.0, 0]),
            "cov": torch.tensor([[0.5, 0], [0.0, 0.1]]),
        }
    ],
) -> dict[int : tuple[anndata.AnnData, anndata.AnnData]]:
    def rotate(params, theta):
        R_theta = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
        params = [
            {
                "mean": torch.matmul(R_theta, param["mean"]),
                "cov": torch.matmul(torch.matmul(R_theta, param["cov"]), R_theta.T),
            }
            for param in params
        ]
        return params

    def get_splits(adata, idx):
        trtm_id = lambda idx: f"treatment_{idx}"
        ood_adata = adata[adata.obs["cond"] == trtm_id(idx)]
        train_adata = adata[adata.obs["cond"] != trtm_id(idx)]
        return train_adata, ood_adata

    num_train_obs = num_components * num_train_treatments + num_train_controls

    # flags for treatment and control groups
    is_train_control = torch.concatenate(
        (torch.ones((num_train_controls,), dtype=int), torch.zeros((num_train_treatments * num_comps,), dtype=int)),
        dim=0,
    ).numpy()
    train_treatments = torch.concatenate(
        (
            torch.zeros((num_train_controls,), dtype=int),
            *[torch.ones((num_train_treatments,), dtype=int) * (idx + 1) for idx in range(num_comps)],
        ),
        dim=0,
    ).numpy()

    # flags for treatment and control groups
    is_test_control = torch.concatenate(
        (torch.ones((num_test_controls,), dtype=int), torch.zeros((num_test_treatments * num_comps,), dtype=int)), dim=0
    ).numpy()
    test_treatments = torch.concatenate(
        (
            torch.zeros((num_test_controls,), dtype=int),
            *[torch.ones((num_test_treatments,), dtype=int) * (idx + 1) for idx in range(num_comps)],
        ),
        dim=0,
    ).numpy()

    # sampling the perturbations and the states
    treatment_dist = sc_exp_design.sym.GaussianMixtureModel(params)
    train_perturbations = treatment_dist.sample(comps=(train_treatments[num_train_controls:] - 1))
    test_perturbations = treatment_dist.sample(comps=(test_treatments[num_test_controls:] - 1))

    train_perturbations = torch.concatenate((torch.zeros((num_train_controls, 2)), train_perturbations), dim=0).numpy()
    test_perturbations = torch.concatenate((torch.zeros((num_test_controls, 2)), test_perturbations), dim=0).numpy()

    train_initial_population = torch.randn((num_train_obs, 2)).numpy() * var_ctrl
    test_initial_population = torch.randn((num_test_obs, 2)).numpy() * var_ctrl

    train_perturbed_population = train_initial_population - train_perturbations
    test_perturbed_population = test_initial_population - test_perturbations

    # creating a treatment identifier
    treatment_map = {0: "control", **{(idx + 1): f"treatment_{idx+1}" for idx in range(num_comps)}}
    train_treatment_ids = np.vectorize(treatment_map.get)(train_treatments)
    test_treatment_ids = np.vectorize(treatment_map.get)(test_treatments)

    # creating anndata object
    train_obs = pd.DataFrame({"is_control": is_train_control, "cond": train_treatment_ids})
    test_obs = pd.DataFrame({"is_control": is_test_control, "cond": test_treatment_ids})

    train_obsm = {"shift": train_perturbations, "original_state": train_initial_population}
    test_obsm = {"shift": test_perturbations, "original_state": test_initial_population}
    uns = {"rep": {value: np.array([key]) for key, value in treatment_map.items()}}
    train_adata = anndata.AnnData(
        X=train_perturbed_population,
        obs=train_obs,
        obsm=train_obsm,
        uns=uns,
    )
    test_adata = anndata.AnnData(
        X=test_perturbed_population,
        obs=test_obs,
        obsm=test_obsm,
        uns=uns,
    )

    splits = {}
    for comp in range(num_components):
        train_adata, ood_adata = get_splits(train_adata, idx + 1)
        test_adata, _ = get_splits(test_adata, idx + 1)

        splits[idx + 1] = {"train": train_adata, "test": test_adata, "ood": ood_adata}
    return splits
