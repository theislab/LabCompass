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

    @pytest.mark.parametrize("set_num_treatments_to_load", [True, False])
    def test_validation_dataloader(
        self,
        adata: anndata.AnnData,
        batch_size: int,
        num_genes: int,
        target_covariates_in_obsm: str,
        num_treatments_to_load: int,
        sample_rep: None | str,
        control_key: None | str,
        perturbations: None | str | Sequence[str],
        perturbations_in_obsm: Sequence[str] | None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
        load_target_covariates: bool,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None,
        has_controls: bool,
        set_num_treatments_to_load: bool,
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

        # initializing validation data loader
        validation_dataloader = sc_exp_design.data.ValidationDataLoader(
            data,
            coupling,
            batch_size,
            state_transforms,
            has_controls=has_controls,
            num_treatments_to_load=num_treatments_to_load if set_num_treatments_to_load else None,
        )

        # sampling batch of train data and validating it
        validation_batch = validation_dataloader.sample()

        # check that we have the correct perturbaation keys        
        seen_combinatorial_perturbations = data.seen_combinatorial_perturbations
        perturbations_with_rep = data.perturbations_with_rep 


        # when perturbations do not induce a group for OT
        if seen_combinatorial_perturbations is None:
            # check that we have the correct number of batches
            if len(validation_batch) > 1:
                msg = f"When {seen_combinatorial_perturbations=} the validation batch should have only one element. Found {len(validation_batch)}."
                raise ValueError(msg)

            # check that we have the correct key
            if data.perturbations_in_obsm is None:
                expected_key = "unconditional"
            else:
                # concatenate perturbation names
                if perturbations_with_rep is None:
                    msg = f"When {seen_combinatorial_perturbations=} and `data.perturbations_in_obsm` is not None, `perturbations_with_rep` should be not None. FOund None."
                    raise ValueError(msg)
                treatment = [perturbation for perturbation in data.perturbations_with_rep]
                expected_key = "_".join(treatment)

        # when we can construct groups
        else:
            # retrieve expected perturbation ids
            expected_keys = ("_".join(treatment) for treatment in seen_combinatorial_perturbations)

            if set_num_treatments_to_load:
                # check that we have the correct number of batches
                if len(validation_batch) != num_treatments_to_load:
                    msg = f"When {set_num_treatments_to_load=} the validation batch should have {num_treatments_to_load} element. Found {len(validation_batch)}."
                    raise ValueError(msg)

                # check that we have the correct keys
                for perturbation_key in validate_batch.keys():
                    if perturbation_key not in expected_keys:
                        msg = f"Perturbation identitfier {perturbation_key} not found in {expected_keys=}."
                        raise ValueError(msg)

            else:
                # check that we have the correct number of batches
                if len(validation_batch) != len(seen_combinatorial_perturbations):
                    msg = f"When {set_num_treatments_to_load=} the validation batch should have {len(seen_combinatorial_perturbations)} elements (one for each combionatorial perturbation). Found {len(validate_batch)}."
                    raise ValueError(msg)

                # check that we have the correct keys
                if tuple(validate_batch.keys()) != expected_keys:
                    msg = f"When {set_num_treatments_to_load=} the validation batch should have all the perturbation ids appearing in `seen_combinatorial_perturbations`."
                    raise ValueError(msg)

        # for each element verify that the batch dictionary is correct
        for perturbation, perturbation_batch in validate_batch.items():
            validate_batch(
                perturbation_batch,
                has_controls,
                batch_size,
                num_genes,
                perturbations,
                data,
            )
