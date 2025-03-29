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
    @pytest.mark.parametrize("perturbations", [None, "treatment0", ("treatment0", "treatment1")])
    @pytest.mark.parametrize("perturbation_covariates",
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
    @pytest.mark.parametrize("perturbation_reps",
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
    @pytest.mark.parametrize("target_covariates",
        [
            None,
            {"target0": "one_hot"},
            {"target0": "label"},
            {"target0": "identity"},
            {"target0": None},
            {"target0": "one_hot", "target1": "one_hot"},
            {"target0": "label", "target1": "one_hot"},
            {"target0": "identity", "target1": "one_hot"},
            {"target0": None, "target1": "one_hot"},
            {"target0": "one_hot", "target1": "label"},
            {"target0": "label", "target1": "label"},
            {"target0": "identity", "target1": "label"},
            {"target0": None, "target1": "label"},
            {"target0": "one_hot", "target1": "identity"},
            {"target0": "label", "target1": "identity"},
            {"target0": "identity", "target1": "identity"},
            {"target0": None, "target1": "identity"},
            {"target0": "one_hot", "target1": None},
            {"target0": "label", "target1": None},
            {"target0": "identity", "target1": None},
            {"target0": None, "target1": None},
        ]
    )
    @pytest.mark.parametrize("has_controls", [True, False])
    def test_data_manager(
        self,
        adata: anndata.AnnData,
        sample_rep: None | str,
        control_key: None | str,
        perturbations: None | str | Sequence[str],
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

        # initializing data manager
        data_manager = sc_exp_design.data.DataManager(
            adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbations=perturbations,
            perturbation_covariates=perturbation_covariates,
            perturbation_reps=perturbation_reps,
            load_target_covariates=load_target_covariates,
            target_covariates=target_covariates,
            has_controls=has_controls,
        )

        # testing perturbation with reps attribute
        perturbations_with_rep = data_manager.perturbations_with_rep

        # when we should not have any perturbation with rep
        if perturbation_covariates is None:
            msg = f""
            assert perturbations_with_rep is None, msg
        if perturbation_reps is None:
            msg = f""
            assert perturbations_with_rep is None, msg
        
        # when we have only one perturbation
        if perturbations == ("treatment0", ):
            if perturbation_reps is not None:
                expected = {
                    "treatment0": "treatment0_label"
                }
                msg = f"Value Mismatch: Expected {expected} got {perturbations_with_rep}"
                assert perturbations_with_rep == expected, msg

        # when we have only one perturbation
        if perturbations == ("treatment0", "treatment1"):
            if perturbation_reps is not None:
                expected = {
                    "treatment0": "treatment0_label",
                    "treatment1": "treatment1_label"
                }
                msg = f"Value Mismatch: Expected {expected} got {perturbations_with_rep}"
                assert perturbations_with_rep == expected, msg
