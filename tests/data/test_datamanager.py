from collections.abc import Sequence
from typing import Literal

import anndata
import pytest
import torch

import sc_exp_design

from .base_data_test import BaseDataTest
from .utils import validate_batch, validate_parametrized_inputs


class TestDataManager(BaseDataTest):
    """"""

    def get_expected_num_perturbation_features(
        self,
        condition: str,
        perturbations_in_obsm: Sequence[str] | None, 
        num_perturbation_feats: int,
    ) -> int:
        # check whether the data is a representation or covariates
        is_repr = "repr" in condition
        if not is_repr:
            msg = f""
            assert "cov" in condition, msg
        # collect expected number of perturbation features
        expected_num_perturbation_features = 1
        if is_repr:
            # check whether we have perturbations in obsm
            if perturbations_in_obsm is not None:
                # check that the perturbation is in obsm
                is_pert_in_obsm = False
                for perturbation in perturbations_in_obsm:
                    if perturbation in condition:
                        is_pert_in_obsm = True
                if is_pert_in_obsm:
                    expected_num_perturbation_features = num_perturbation_feats
        return expected_num_perturbation_features

    @pytest.mark.parametrize("batch_data", [False, True])
    @pytest.mark.parametrize("treatments", [None, ]) # TODO: Add othe option to test
    def test_data_manager(
        self,
        adata: anndata.AnnData,
        num_genes: int,
        num_control_cells: int,
        num_perturbed_cells: int,
        tot_perturbed_cells: int,
        num_perturbation_feats: int,
        batch_size: int,
        target_covariates_in_obsm: str,
        sample_rep: None | str,
        control_key: None | str,
        perturbations: None | str | Sequence[str],
        perturbations_in_obsm: Sequence[str] | None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
        load_target_covariates: bool,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None,
        batch_data: bool,
        treatments: None,
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
        ) = validate_parametrized_inputs(
            sample_rep,
            control_key,
            perturbations,
            perturbations_in_obsm, 
            perturbation_covariates,
            perturbation_reps,
            load_target_covariates,
            target_covariates,
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
        )

        # retrieving data
        data = data_manager.get_data()

        # check that we have the correct perturbation keys        
        seen_combinatorial_perturbations = data.seen_combinations
        perturbations_with_rep = data.perturbations

        # when we should not have any perturbation with rep
        if perturbations is None:
            msg = f""
            assert perturbations_with_rep is None, msg
        if perturbation_reps is None:
            msg = f""
            assert perturbations_with_rep is None, msg

        # perturbation data
        perturbation_data = data.perturbation_data
        if perturbations is None:
            msg = f""
            assert perturbation_data is None, msg
        else:
            msg = f""
            assert perturbation_data is not None, msg

            # perturbation representations
            for perturbation in perturbations:
                reps = perturbation_reps[perturbation]
                if perturbations_in_obsm is not None:
                    if perturbation in perturbations_in_obsm:
                        msg = f""
                        assert f"repr_{perturbation}_{reps}" in perturbation_data.keys(), msg
                        continue
                else:    
                    for rep in reps:
                        msg = f""
                        assert f"repr_{perturbation}_{rep}" in perturbation_data.keys(), msg

                    if perturbation_covariates is not None:
                        if perturbation in perturbation_covariates.keys():
                            covs = perturbation_covariates[perturbation]

                            for cov in covs:
                                msg = f""
                                assert f"cov_{perturbation}_{cov}" in perturbation_data.keys(), msg
        
        # target data
        target_data = data.target_data
        if load_target_covariates:
            msg = f""
            assert target_data is not None, msg
            for target, target_rep in target_covariates.items():
                msg = f""
                assert target in target_data.keys(), msg
        
        # treatments
        treatment_data = data.get_treatments(batch_size=batch_size if batch_data else None)
        
        # state data
        msg = f""
        assert hasattr(treatment_data, "state_data"), msg

        # collect expected number of treatment cells
        expected_num_cells = batch_size
        if not batch_data:
            expected_num_cells = num_perturbed_cells
            if treatments is None:
                expected_num_cells = tot_perturbed_cells
            if not data_manager.has_controls:
                expected_num_cells = expected_num_cells + num_control_cells

        # define expected shape for treatment states
        expected_shape = (expected_num_cells, num_genes)

        # check shapes
        msg = f""
        assert treatment_data.state_data.shape == expected_shape, msg

        # perturbation data
        if perturbations is not None:
            msg = f""
            assert hasattr(treatment_data, "perturbation_data"), msg
            
            # check shapes
            for condition, condition_data in treatment_data.perturbation_data.items():
                # collect expected number of perturbation features
                expected_num_perturbation_features = self.get_expected_num_perturbation_features(
                    condition,
                    perturbations_in_obsm,
                    num_perturbation_feats
                )
                # define expected shape for treatment perturbation data
                expected_shape = (expected_num_cells, expected_num_perturbation_features)

                # check shapes
                msg = f"Test failed on {condition=}. Expected shape {expected_shape}, found {condition_data.shape}."
                assert condition_data.shape == expected_shape, msg
        
        # target data
        if load_target_covariates:
            msg = f""
            assert hasattr(treatment_data, "target_data"), msg
        
        # controls
        if data_manager.has_controls:
            control_data = data.get_controls(batch_size=batch_size if batch_data else None)

            # state data
            msg = f""
            assert hasattr(control_data, "state_data"), msg

            # collect expected number of treatment cells
            expected_num_cells = num_control_cells
            if batch_data:
                expected_num_cells = batch_size
            
            # define expected shape for control states           
            expected_shape = (expected_num_cells, num_genes)

            # check shapes
            msg = f""
            assert control_data.state_data.shape == expected_shape, msg

            # perturbation data
            if perturbations is not None:
                msg = f""
                assert hasattr(control_data, "perturbation_data"), msg

                # check shapes
                for condition, condition_data in control_data.perturbation_data.items():
                    # collect expected number of perturbation features
                    expected_num_perturbation_features = self.get_expected_num_perturbation_features(
                        condition,
                        perturbations_in_obsm,
                        num_perturbation_feats
                    )
                    # define expected shape for treatment perturbation data
                    expected_shape = (expected_num_cells, expected_num_perturbation_features)

                    # check shapes
                    msg = f"Test failed on {condition=}. Expected shape {expected_shape}, found {condition_data.shape}."
                    assert condition_data.shape == expected_shape, msg

            # target data
            if load_target_covariates:
                msg = f""
                assert hasattr(control_data, "target_data"), msg
