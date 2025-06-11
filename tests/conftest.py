from collections.abc import Sequence

import pytest

import anndata
import numpy as np

from sc_exp_design.sym import get_dummy_data


@pytest.fixture
def num_combinatorial_treatments() -> int:
    """"""
    return 3


@pytest.fixture
def num_targets() -> int:
    """"""
    return 2


@pytest.fixture
def dim_target_covariates() -> int:
    """"""
    return 8


@pytest.fixture
def num_unique_treatments() -> int:
    """"""
    return 5


@pytest.fixture
def num_unique_groups() -> int:
    """"""
    return 3


@pytest.fixture
def num_control_cells() -> int:
    """"""
    return 100


@pytest.fixture
def num_perturbed_cells() -> int:
    """"""
    return 150


@pytest.fixture
def num_genes() -> int:
    """"""
    return 200


@pytest.fixture
def num_perturbation_feats() -> int:
    """"""
    return 100


@pytest.fixture
def batch_size() -> int:
    """"""
    return 64


@pytest.fixture
def num_treatments_to_load() -> int:
    """"""
    return 2


@pytest.fixture
def num_unique_target_values() -> int:
    """"""
    return 7


@pytest.fixture
def perturbations_in_obsm() -> str:
    """"""
    return "treatment2"


@pytest.fixture
def target_covariates_in_obsm() -> str:
    """"""
    return "target3"


@pytest.fixture
def tot_perturbed_cells(
    num_unique_treatments: int,
    num_perturbed_cells: int,
) -> int:
    """"""
    return num_perturbed_cells*num_unique_treatments


@pytest.fixture
def num_cells(
    num_control_cells: int,
    tot_perturbed_cells: int,
) -> int:
    """"""
    return num_control_cells + tot_perturbed_cells


@pytest.fixture
def states(
    num_cells: int,
    num_genes: int,
) -> np.ndarray:
    """"""
    return np.ones((num_cells, num_genes))


@pytest.fixture
def treatment_labels(
    num_combinatorial_treatments: int,
) -> Sequence[str]:
    """"""
    return (f"treatment{u}" for u in range(num_combinatorial_treatments))


@pytest.fixture
def perturbations(
    num_cells: int,
    num_perturbation_feats: int,
    treatment_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """"""
    return {
        treatment: np.zeros((num_cells, num_perturbation_feats)) for treatment in treatment_labels
    }


@pytest.fixture
def target_labels(
    num_targets: int,
) -> Sequence[str]:
    """"""
    return (f"target{u}" for u in range(num_targets))


@pytest.fixture
def targets(
    num_cells: int,
    target_labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """"""
    return {
        target: np.zeros((num_cells, 1)) for target in target_labels
    }


@pytest.fixture
def target_covariate_in_obsm_data(
    num_cells: int,
    dim_target_covariates: int,
) -> np.ndarray:
    """"""
    return np.zeros((num_cells, dim_target_covariates))


@pytest.fixture
def adata(
    num_unique_treatments: int,
    num_unique_groups: int,
    num_control_cells: int,
    num_perturbed_cells: int,
    num_genes: int,
    num_unique_target_values: int,
    num_perturbation_feats: int,
    tot_perturbed_cells: int,
    num_cells: int,
    states: np.ndarray,
    dim_target_covariates: int,
    target_covariates_in_obsm: str,
    target_covariate_in_obsm_data: np.ndarray,
    perturbations_in_obsm: str,
) -> anndata.AnnData:
    """"""
    return get_dummy_data(
        num_unique_treatments,
        num_unique_groups,
        num_control_cells,
        num_perturbed_cells,
        num_genes,
        num_unique_target_values,
        num_perturbation_feats,
        tot_perturbed_cells,
        num_cells,
        states,
        dim_target_covariates,
        target_covariates_in_obsm,
        target_covariate_in_obsm_data,
        perturbations_in_obsm,
    )
