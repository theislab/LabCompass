from collections.abc import Sequence
from typing import Literal

import anndata
import pytest
import torch

import sc_exp_design


class TestDataManager:
    """"""
    @pytest.mark.parametrize("sample_rep", [None, "states"])
    @pytest.mark.parametrize("control_key", [None, "is_control"])
    @pytest.mark.parametrize("perturbations", [None, "treatment0", ("treatment0", "treatment1"), ("treatment0", "treatment1", "treatment2"), ])
    @pytest.mark.parametrize("perturbations_in_obsm", [None, "treatment2"])
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
            None, 
            {"treatment0":("treatment0_label", )},
            {"treatment0":("treatment0_label", "treatment0_group")},
            {"treatment0": ("treatment0_label", ), "treatment1": ("treatment1_label", )},
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_label", "treatment1_group")},
            {"treatment0": ("treatment0_label", ), "treatment1": ("treatment1_label", "treatment1_group")},
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_group", )},
        ]
    )
    @pytest.mark.parametrize("load_target_covariates", [False, True])
    @pytest.mark.parametrize(
        "target_covariates",
        [
            None,
            {"target0": "one_hot"},
            {"target0": "label"},
            {"target0": "identity"},
            {"target0": "one_hot", "target1": "one_hot"},
            {"target0": "label", "target1": "one_hot"},
            {"target0": "identity", "target1": "one_hot"},
            {"target0": "one_hot", "target1": "label"},
            {"target0": "label", "target1": "label"},
            {"target0": "identity", "target1": "label"},
            {"target0": "one_hot", "target1": "identity"},
            {"target0": "label", "target1": "identity"},
            {"target0": "identity", "target1": "identity"},
        ]
    )
    @pytest.mark.parametrize("has_controls", [True, False])
    def test_data_manager(
        self,
        adata: anndata.AnnData,
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

        # handling inputs
        if target_covariates is None:
            load_target_covariates = False
        
        # we need to be passing the representation
        if perturbation_reps is not None:
            if perturbations is not None:
                perturbations = tuple(perturbation for perturbation in perturbations if perturbation in perturbation_reps.keys())
        else:
            perturbations = None
        
        # when there are no controls
        if control_key is None:
            has_controls = False

        # when we are using perturbations in obsm, we need to ensure that
        # it appears in `perturbations`
        # also, we need to add the modeled features to the perturbation_reps dictionary
        if perturbations is None or ("treatment2" not in perturbations):
            perturbations_in_obsm = None
        if perturbations_in_obsm is not None:
            if perturbation_reps is None:
                perturbation_reps = {}
            # perturbation_reps["treatment2"] = "feats_treatment2_features"
            perturbation_reps["treatment2"] = "cov_treatment2_features" # TODO: change "cov" to "feats" once finished the viral notebooks

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
            target_covariates=target_covariates,
            has_controls=has_controls,
        )

        # testing perturbation with reps attribute
        perturbations_with_rep = data_manager.perturbations_with_rep

        # when we should not have any perturbation with rep
        if perturbations is None:
            msg = f"When `perturbations` is None also `perturbations_with_rep` should be None. Found {perturbations_with_rep}."
            assert perturbations_with_rep is None, msg
        if perturbation_reps is None:
            msg = f"When `perturbations_reps` is None also `perturbations_with_rep` should be None. Found {perturbations_with_rep}."
            assert perturbations_with_rep is None, msg
        
        # when we have only one perturbation
        if perturbations == ("treatment0", ):
            if perturbation_reps is not None:
                expected = {
                    "treatment0": ["control", "drug1", "drug2", "drug3", "drug4", "drug5"]
                }
                msg = f"Value Mismatch: Expected {expected} got {perturbations_with_rep}"
                assert perturbations_with_rep == expected, msg

        # when we have two perturbations
        if perturbations == ("treatment0", "treatment1"):
            if perturbation_reps is not None:
                expected = {
                    "treatment0": ["control", "drug1", "drug2", "drug3", "drug4", "drug5"],
                    "treatment1": ["control", "drug1", "drug2", "drug3", "drug4", "drug5"]
                }
                msg = f"Value Mismatch: Expected {expected} got {perturbations_with_rep}"
                assert perturbations_with_rep == expected, msg

        # when we have three perturbations
        if perturbations == ("treatment0", "treatment1", "treatment2"):
            if perturbation_reps is not None:
                expected = {
                    "treatment0": ["control", "drug1", "drug2", "drug3", "drug4", "drug5"],
                    "treatment1": ["control", "drug1", "drug2", "drug3", "drug4", "drug5"],
                    "treatment2": "feats_treatment2_treatment2_features",
                    "treatment2": "cov_treatment2_treatment2_features" # TODO: change "cov" to "feats" once finished the viral notebooks
                }
                msg = f"Value Mismatch: Expected {expected} got {perturbations_with_rep}"
                assert perturbations_with_rep == expected, msg
        # retrieving data
        data = data_manager.get_data()

        # perturbation data
        perturbation_data = data.perturbation_data

        if perturbations is None:
            msg = f"When `perturbations` is None also `perturbation_data` should be None. Found {perturbation_data}."
            assert perturbation_data is None, msg
        else:
            msg = f"When `perturbations` is passed also `perturbation_data` should be not None. Found None."
            assert perturbation_data is not None, msg

            # perturbation representations
            for perturbation in perturbations:
                reps = perturbation_reps[perturbation]

                for rep in reps:
                    expected_key = f"repr_{perturbation}_{rep}"
                    msg = f"Expected key {expected_key} not found in {perturbation_data.keys()=}"
                    assert expected_key in perturbation_data.keys(), msg
            
            # perturbation covariates
            for perturbation in perturbations:
                if perturbation_covariates is not None:
                    if perturbation in perturbation_covariates.keys():
                        covs = perturbation_covariates[perturbation]

                        for cov in covs:
                            expected_key = f"cov_{perturbation}_{cov}"        
                            msg = f"Expected key {expected_key} not found in {perturbation_data.keys()=}"
                            assert expected_key in perturbation_data.keys(), msg
        
        # target data
        target_data = data.target_reprs
        if load_target_covariates:
            msg = f""
            assert target_data is not None, msg
            for target, target_rep in target_covariates.items():
                msg = f""
                assert target in target_data.keys(), msg
        
        # treatments
        treatment_data = data.get_treatments()
        
        msg = f""
        assert "state_data" in treatment_data.keys()

        if perturbations is not None:
            msg = f""
            assert "condition" in treatment_data.keys()
        
        if load_target_covariates:
            msg = f""
            assert "target_data" in treatment_data.keys(), msg
        
        # controls
        if has_controls:
            control_data = data.get_controls()

            msg = f""
            assert "state_data" in control_data.keys()

            if perturbations is not None:
                msg = f""
                assert "condition" in control_data.keys()
            
            if load_target_covariates:
                msg = f""
                assert "target_data" in control_data.keys(), msg
