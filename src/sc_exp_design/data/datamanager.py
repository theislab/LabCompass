import logging
from collections.abc import Sequence
from typing import Any, Literal

import anndata
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sc_exp_design.constants import (
    CONDITION_COV_KEY,
    CONDITION_REP_KEY,
    TARGET_CATEGORIES_KEY,
)
from sc_exp_design.data.data import TrainData
from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "DataManager",
]


class DataManager:
    """"""

    def __init__(
        self,
        adata: anndata.AnnData | None = None,
        sample_rep: str | dict[str] | None = None,
        control_key: str | None = None,
        perturbations: str | Sequence[str] | None = None,
        perturbation_covariates: dict[str, str | Sequence[str]] | None = None,
        perturbation_reps: dict[str, str | Sequence[str]] | None = None,
        use_perturbation_target_repr: bool = False,
        perturbation_target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None = None,
        perturbation_target_covariates_in_obsm: dict[str, bool] | None = None,
        perturbation_target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        `self.perturbations`:
            the individual perturbations modeled. need to be a key in `adata.obs`

        `self.perturbation_reps`:
            the representation in self.perturbation_covariate_reps
            are the same over cell with the same perturbation
            basically the resulting data will have one
            entry for each unique perturbation considered
            for example, for a given drug this could be given
            by their chemical representation or any other feature that
            is constant across cells for any perturbation

        `self.perturbation_covariates`:
            these will be the covariates associated to a given perturbation
            on a cell level basis and that can vary over the cells for the
            same perturbations. The resulting data will have one entry for each cell.
            for example, for a given drug this could be given
            by the dosage or the time of the treatment or any other feature
            associated to the current perturbation that can vary across cells

        """
        self.adata = adata
        self.sample_rep = sample_rep
        self.control_key = control_key

        # preparing the attributes
        if perturbations is not None:
            if isinstance(perturbations, str):
                perturbations = (perturbations,)
            for perturbation in perturbations:
                perturbation_found = False
                # cell level covariates
                if perturbation_covariates is not None:
                    if perturbation in perturbation_covariates.keys():
                        covariates = perturbation_covariates[perturbation]
                        if isinstance(covariates, str):
                            covariates = (covariates,)
                        perturbation_found = True
                    else:
                        covariates = ()
                    perturbation_covariates[perturbation] = covariates
                # perturbation level covariates
                if perturbation_reps is not None:
                    if perturbation in perturbation_reps.keys():
                        rep = perturbation_reps[perturbation]
                        if isinstance(rep, str):
                            rep = (rep,)
                        perturbation_found = True
                    else:
                        rep = ()
                    perturbation_reps[perturbation] = rep
                # warning if perturbation not found
                if not perturbation_found:
                    msg = f"{perturbation} in `self.perturbation` has neither any representation nor covariates associates, skipping."
                    logger.warning(msg)

        self.perturbations = perturbations
        self.perturbation_covariates = perturbation_covariates
        self.perturbation_reps = perturbation_reps
        self.use_perturbation_target_repr = use_perturbation_target_repr

        # when we need some target representation for the conditions
        if self.use_perturbation_target_repr:
            msg = f"With {self.use_perturbation_target_repr=} you need to specify the target covariate reprs in `perturbation_target_reprs`, `None` found"
            assert perturbation_target_covariates is not None, msg
            if perturbation_target_covariates_kwargs is None:
                perturbation_target_covariates_kwargs = {}
                for target_covariate, target_covariate_rep in perturbation_target_covariates.items():
                    perturbation_target_covariates_kwargs[target_covariate] = {}

            if perturbation_target_covariates_in_obsm is None:
                perturbation_target_covariates_in_obsm = ()

        self.perturbation_target_covariates = perturbation_target_covariates
        self.perturbation_target_covariates_in_obsm = perturbation_target_covariates_in_obsm
        self.perturbation_target_covariates_kwargs = perturbation_target_covariates_kwargs

    def __get_state_data(
        self,
        adata: anndata.AnnData,
    ) -> TensorLike:
        """"""
        if self.sample_rep is None:
            state_data = adata.X
        else:
            if self.sample_rep not in adata.layers.keys():
                msg = f"{self.sample_rep=} not found in `adata.layers` (Keys found: {list(adata.layers.keys())})"
                raise ValueError(msg)
            state_data = adata.layers[self.sample_rep]
        return state_data

    def __get_perturbation_data(
        self,
        adata: anndata.AnnData,
    ) -> dict[str, TensorLike]:
        """"""
        # retrieving perturbation data
        perturbation_data = {}
        # iterating over each perturbation covariate
        for perturbation in self.perturbations:
            # sanity check on the input AnnData
            if perturbation not in adata.obs.keys():
                msg = f"{perturbation} not found in `adata.obs.keys()`"
                raise ValueError(msg)
            # what perturbation was applied
            covariate_data = adata.obs[perturbation].values
            # optionally retrieving the representation of such covariate
            if self.perturbation_reps is not None:
                perturbation_covariate_rep = self.perturbation_reps[perturbation]
                for rep in perturbation_covariate_rep:
                    # sanity check on the input AnnData
                    if rep not in adata.uns.keys():
                        msg = f"{perturbation_covariate_rep} not found in `adata.uns.keys()`"
                        raise ValueError(msg)
                    # retrieving the covariates and their representations
                    covariate_reps_dict = adata.uns[rep]
                    # mapping each observation condition to their representation
                    covariate_reps = [covariate_reps_dict[covariate] for covariate in covariate_data]
                    covariate_reps = np.stack(covariate_reps, axis=0)
                    # storing the results
                    covariate_rep_key = f"{CONDITION_REP_KEY}_{perturbation}_{rep}"
                    perturbation_data[covariate_rep_key] = covariate_reps
            # loading perturbation covariates that are individual for each cell
            if self.perturbation_covariates is not None:
                perturbation_covariates = self.perturbation_covariates[perturbation]
                # iterating over each of such covariates associated to the current perturbation
                # which should be stored in the corresponding obsm field of the AnnData object
                for covariate in perturbation_covariates:
                    # sanity check on the input AnnData
                    if covariate not in adata.obsm.keys():
                        msg = f"{covariate=} not found in `self.adata.obsm.keys()`"
                        raise ValueError(msg)
                    # retrieving the covariate data
                    covariate_data = adata.obsm[covariate]
                    # storing the results
                    covariate_cov_key = f"{CONDITION_COV_KEY}_{perturbation}_{covariate}"
                    perturbation_data[covariate_cov_key] = covariate_data
            # storing them to the output dictionary
        return perturbation_data

    def __get_perturbation_target_rep_data(
        self,
        adata: anndata.AnnData,
    ) -> dict[str, TensorLike]:
        """"""
        out_dict = {}
        for condition_target_covariate, condition_target_covariate_rep in self.perturbation_target_covariates.items():
            if condition_target_covariate in self.perturbation_target_covariates_in_obsm:
                # sanity check
                msg = f"{condition_target_covariate} not found in `adata.obsm.keys()`"
                assert condition_target_covariate in adata.obsm.keys(), msg
                condition_target_covariate_data = adata.obsm[condition_target_covariate]
                out_dict[condition_target_covariate] = condition_target_covariate_data
            else:
                condition_target_covariate_kwargs = self.perturbation_target_covariates_kwargs[condition_target_covariate]
                # sanity check
                msg = f"{condition_target_covariate} not found in `adata.obs.columns`"
                assert condition_target_covariate in adata.obs.columns, msg

                covariate_target_rep = self.perturbation_target_covariates[condition_target_covariate]
                covariate_target_rep_kwargs = self.perturbation_target_covariates_kwargs[condition_target_covariate]

                covariate_data = adata.obs[[condition_target_covariate]].values

                if covariate_target_rep == "one_hot":
                    covariate_rep_encoder = OneHotEncoder(**covariate_target_rep_kwargs)
                    covariate_target_rep_data = covariate_rep_encoder.fit_transform(covariate_data).toarray()
                elif covariate_target_rep == "label":
                    covariate_rep_encoder = LabelEncoder()
                    if TARGET_CATEGORIES_KEY in covariate_target_rep_kwargs.keys():
                        target_categories = covariate_target_rep_kwargs[TARGET_CATEGORIES_KEY]
                        covariate_rep_encoder.fit(target_categories)
                        covariate_target_rep_data = covariate_rep_encoder.transform(covariate_data)
                    else:
                        covariate_target_rep_data = covariate_rep_encoder.fit_transform(covariate_data)
                else:
                    msg = f"{covariate_target_rep=} not currently supported (avaiable options are `['one_hot', 'label', 'identity']`)"
                    raise NotImplementedError(msg)

                out_dict[condition_target_covariate] = covariate_target_rep_data
        return out_dict

    def get_train_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> TrainData:
        """"""
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata
        state_data = self.__get_state_data(adata)
        perturbation_data = None
        if self.perturbations is not None:
            perturbation_data = self.__get_perturbation_data(adata)
        # condition target rep
        target_perturbation_repr = None
        if self.use_perturbation_target_repr:
            target_perturbation_repr = self.__get_perturbation_target_rep_data(adata)
        return TrainData(adata, self.control_key, state_data, perturbation_data, target_perturbation_repr)
