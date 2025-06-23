from collections.abc import Sequence
from typing import Any, Literal

import anndata
import numpy as np
import pytest

import sc_exp_design

INVALID_KEY = "unwanted_key"


class TestDataSchema:
    """"""

    @pytest.mark.parametrize("sample_rep", [None, "states", INVALID_KEY])
    def test_state_data_schema(
        self,
        adata: anndata.AnnData,
        num_cells: int,
        num_genes: int,
        sample_rep: None | str,
    ) -> None:
        """"""

        # when we expect an exception
        if sample_rep == INVALID_KEY:
            with pytest.raises(KeyError):
                state_data_schema = sc_exp_design.data.schemas.StateDataSchema(
                    adata,
                    sample_rep,
                )
        # this should initialize without problems
        else:
            state_data_schema = sc_exp_design.data.schemas.StateDataSchema(
                adata,
                sample_rep,
            )
        
        # retrieving the data
        state_data = state_data_schema.get_data()

        # check type
        if not isinstance(state_data, np.ndarray):
            msg = f""
            raise TypeError(msg)
        
        # check shape
        expected_shape = (num_cells, num_genes)
        if state_data.shape != expected_shape:
            msg = f""
            raise ValueError(msg)


    @pytest.mark.parametrize("perturbations", [("treatment0",), ("treatment0", "treatment1"), ("treatment0", "treatment1", "treatment2"), (INVALID_KEY,)])
    @pytest.mark.parametrize(
        "perturbation_covariates",
        [
            None,
            {"treatment0":("treatment0_dose", )},
            {"treatment0":("treatment0_dose", "treatment0_time")},
            {"treatment0": ("treatment0_dose", ), "treatment1": ("treatment1_dose", )},
            {"treatment0": ("treatment0_dose", "treatment0_time"), "treatment1": ("treatment1_dose", "treatment1_time")},
            {"treatment0": ("treatment0_dose", ), "treatment1": ("treatment1_dose", "treatment1_time")},
            {"treatment0": ("treatment0_dose", "treatment0_time"), "treatment1": ("treatment1_dose", )},
        ]
    )
    @pytest.mark.parametrize(
        "perturbation_reps",
        [
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_label", "treatment1_group"), "treatment2": ("treatment2_features",)},
            {"treatment0": ("treatment0_label", ), "treatment1": ("treatment1_label", "treatment1_group"), "treatment2": ("treatment2_features",)},
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_group", ), "treatment2": ("treatment2_features",)},
        ]
    )
    def test_perturbation_data_schema(
        self,
        adata: anndata.AnnData,
        perturbations_in_obsm: Sequence[str],
        num_cells: int,
        num_perturbation_feats: int,
        perturbations: None | str | Sequence[str],
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
    ) -> None:
        """"""
        # throwing error at schema initialization
        for perturbation in perturbations:
            # keyerror expected when
            # perturbation does not have associated representation
            # perturbation is not present in adata.obs
            if (perturbation not in perturbation_reps.keys()) or (perturbation == INVALID_KEY):
                with pytest.raises(KeyError):
                    perturbation_schema = sc_exp_design.data.schemas.PerturbationDataSchema(
                        adata,
                        perturbations,
                        perturbation_reps,
                        perturbations_in_obsm,
                        perturbation_covariates,
                    )
    
        # initializing schema
        perturbation_schema = sc_exp_design.data.schemas.PerturbationDataSchema(
            adata,
            perturbations,
            perturbation_reps,
            perturbations_in_obsm,
            perturbation_covariates,
        )
        # retrieving the data
        perturbation_data = perturbation_schema.get_data()

        # checking the returned data
        for perturbation in perturbations:
            # retrieving representations
            reps = perturbation_reps[perturbation]
            # when perturbation is in obsm
            if perturbation in perturbations_in_obsm:
                # retrieving representation and constructing
                # expected key to be found in data dictionary
                reps = reps[0]
                expected_key = f"{sc_exp_design.constants.DataFields.CONDITION_FEATS}_{perturbation}_{reps}"
                # checking that the key is present in the data dictionary
                assert expected_key in perturbation_data.keys()
                # check shape
                expected_shape = (num_cells, num_perturbation_feats)
                assert perturbation_data[expected_key].shape == expected_shape

            # otherwise we need to check both reps and covariates when provided
            else:
                # check representations
                for perturbation_rep in reps:
                    # constructing expected key
                    expected_key = f"{sc_exp_design.constants.DataFields.CONDITION_REP}_{perturbation}_{perturbation_rep}"
                    # checking that the key is present in the data dictionary
                    assert expected_key in perturbation_data.keys()
                    # check shape
                    expected_shape = (num_cells, 1)
                    assert perturbation_data[expected_key].shape == expected_shape

                # check covariates
                if perturbation_covariates is not None:
                    # retrieve covariates
                    covs = perturbation_covariates[perturbation]
                    for perturbation_cov in covs:
                        # constructing expected key
                        expected_key = f"{sc_exp_design.constants.DataFields.CONDITION_COV}_{perturbation}_{perturbation_cov}"
                        # checking that the key is present in the data dictionary
                        assert expected_key in perturbation_data.keys()
                        # check shape
                        expected_shape = (num_cells, 1)
                        assert perturbation_data[expected_key].shape == expected_shape

    @pytest.mark.parametrize(
        "target_covariates",
        [
            {"target0": "one_hot"},
            {"target0": "label"},
            {"target0": "one_hot", "target1": "one_hot"},
            {"target0": "label", "target1": "one_hot"},
            {"target0": "one_hot", "target1": "label"},
            {"target0": "label", "target1": "label"},
            {"target2": "identity"},
            {"target0": "one_hot", "target2": "identity",},
            {"target0": "label", "target2": "identity",},
            {"target0": "one_hot", "target1": "one_hot", "target2": "identity",},
            {"target0": "label", "target1": "one_hot", "target2": "identity",},
            {"target0": "one_hot", "target1": "label", "target2": "identity",},
            {"target0": "label", "target1": "label", "target2": "identity",},
            {"target3": "identity"},
            {INVALID_KEY: "label"},
        ]
    )
    def test_target_data_schema(
        self,
        adata: anndata.AnnData,
        target_covariates_in_obsm: str,
        num_cells: int,
        num_unique_target_values: int,
        dim_target_covariates: int,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"]],
    ) -> None:
        """"""
        # throwing error at schema initialization
        if INVALID_KEY in target_covariates.keys():
            with pytest.raises(KeyError):    
                target_schema = sc_exp_design.data.schemas.TargetDataSchema(
                    adata,
                    target_covariates,
                    target_covariates_in_obsm,
                    target_covariates_kwargs=None,
                )

        # initializing target data schema 
        target_schema = sc_exp_design.data.schemas.TargetDataSchema(
            adata,
            target_covariates,
            target_covariates_in_obsm,
            target_covariates_kwargs=None,
        )
        # retrieving target data
        target_data = target_schema.get_data()

        # iterating over the target covariates
        for target_covariate, target_covariate_repr in target_covariates.items():
            # check that we have the expected key
            assert target_covariate in target_data.keys()
            # check shape
            if target_covariate in target_covariates_in_obsm:
                expected_shape = (num_cells, dim_target_covariates)
            else:
                if target_covariate_repr == "label":
                    expected_shape = (num_cells, 1)
                elif target_covariate_repr == "one_hot":
                    expected_shape = (num_cells, num_unique_target_values)
                else:
                    expected_shape = (num_cells, 1)
            assert target_data[target_covariate].shape == expected_shape
