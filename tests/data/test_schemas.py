import anndata
import pytest

import sc_exp_design

class TestDataSchema:
    """"""

    @pytest.mark.parametrize("sample_rep", [None, "states", "invalid_key"])
    def test_state_data_schema_init(
        self,
        adata: anndata.AnnData,
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
