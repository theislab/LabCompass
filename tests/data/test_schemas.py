import anndata
import pytest

import sc_exp_design

class TestDataSchema:

    def test_perturbation_data_schema(
        self,
        adata: anndata.AnnData, 
    ) -> None:
        """"""
