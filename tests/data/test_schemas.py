from collections.abc import Sequence

import anndata
import numpy as np
import pytest

import sc_exp_design

class TestDataSchema:
    """"""

    @pytest.mark.parametrize("sample_rep", [None, "states", "invalid_key"])
    def test_state_data_schema(
        self,
        adata: anndata.AnnData,
        num_cells: int,
        num_genes: int,
        sample_rep: None | str,
    ) -> None:
        """"""

        # when we expect an exception
        if sample_rep == "invalid_key":
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


    @pytest.mark.parametrize("perturbations", [None, ("treatment0",), ("treatment0", "treatment1"), ("treatment0", "treatment1", "treatment2"), ])
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
            {"treatment2": ("treatment2_features",)}
        ]
    )
    def test_perturbation_data_schema(
        self,
        adata: anndata.AnnData,
        perturbations_in_obsm: Sequence[str] | None,
        perturbations: None | str | Sequence[str],
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
    ) -> None:
        """"""

        # prepare data

    def test_target_data_schema(
        self,
    ) -> None:
        """"""
