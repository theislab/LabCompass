from collections.abc import Sequence
from typing import Literal

import anndata
import pytest

import sc_exp_design
from sc_exp_design.constants import DataFields

from .base_data_test import BaseDataTest
from .utils import validate_batch, validate_parametrized_inputs


class TestDataLoaders(BaseDataTest):
    """"""

    def test_train_dataloader(
        self,
        adata: anndata.AnnData,
        batch_size: int,
        num_genes: int,
        target_covariates_in_obsm: str,
        sample_rep: None | str,
        control_key: None | str,
        perturbations: None | str | Sequence[str],
        perturbations_in_obsm: Sequence[str] | None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
        load_target_covariates: bool,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None,
        has_controls: bool,
    ) -> None:
        """"""

        # validating parametrized inputs
        (
            sample_rep,
            control_key,
            perturbations,
            perturbations_in_obsm, 
            perturbation_covariates,
            perturbation_reps,
            load_target_covariates,
            target_covariates,
            has_controls,
        ) = validate_parametrized_inputs(
            sample_rep,
            control_key,
            perturbations,
            perturbations_in_obsm, 
            perturbation_covariates,
            perturbation_reps,
            load_target_covariates,
            target_covariates,
            has_controls,
        )

        # initializing data manager
        data_manager = sc_exp_design.data.DataManager(
            adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbations=perturbations,
            perturbations_in_obsm=perturbations_in_obsm,
            perturbation_covariates=perturbation_covariates,
            perturbation_reps=perturbation_reps,
            load_target_covariates=load_target_covariates,
            target_covariates_in_obsm=target_covariates_in_obsm,
            target_covariates=target_covariates,
            has_controls=has_controls,
        )

        # retrieving data
        data = data_manager.get_data()

        # initialize coupling
        coupling = sc_exp_design.couplings.IndependentCoupling()

        # initialize transforms
        state_transforms = None

        # initializing trainig data loader
        train_dataloader = sc_exp_design.data.TrainDataLoader(
            data,
            coupling,
            batch_size,
            state_transforms,
            has_controls=has_controls,
        )

        # sampling batch of train data and validating it
        train_batch = train_dataloader.sample()
        validate_batch(
            train_batch,
            has_controls,
            batch_size,
            num_genes,
            perturbations,
            data,
        )
