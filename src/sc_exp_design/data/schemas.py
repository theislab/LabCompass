import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import anndata
import numpy as np

from sc_exp_design.constants import DataFields
from sc_exp_design.transforms.utils import label_encode, one_hot_encode

logger = logging.getLogger(__name__)

__all__ = [
    "StateDataSchema",
    "PerturbationDataModel",
    "TargetDataSchema",
]


@dataclass
class BaseDataSchema(abc.ABC):
    """"""

    adata: anndata.AnnData

    def _resolve_adata(
        self,
        adata: anndata.AnnData | None = None,
    ) -> anndata.AnnData:
        """"""
        # handling adata
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata
        return adata

    @abc.abstractclassmethod
    def get_data(
        self,
    ) -> Any:
        """"""
        raise NotImplementedError


@dataclass
class StateDataSchema(BaseDataSchema):
    """"""

    adata: anndata.AnnData
    sample_rep: str | None

    def __post_init__(
        self,
    ) -> None:
        """"""
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self.sample_rep is not None:
            if self.sample_rep not in self.adata.obsm.keys():
                msg = f"{self.sample_rep=} not found in `adata.obsm` (Keys found: {list(adata.obsm.keys())})"
                raise KeyError(msg)

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> dict[str, np.ndarray]:
        """"""
        # handling adata
        adata = self._resolve_adata(adata)
        
        # retrieving the X attribute when no sample rep provided
        if self.sample_rep is None:
            state_data = adata.X
        else:
            state_data = adata.obsm[self.sample_rep]
        return state_data


@dataclass
class PerturbationDataSchema(BaseDataSchema):
    """"""

    adata: anndata.AnnData
    perturbations: str | Sequence[str] | None
    perturbations_in_obsm: str | Sequence[str] | None
    perturbation_covariates: dict[str, str | Sequence[str]] | None
    perturbation_reps: dict[str, str | Sequence[str]] | None

    def __post_init__(
        self,
    ) -> None:
        """"""
        # sanity check perturbations_in_obsm
        # when not provided, initialize empty sequence
        if self.perturbations_in_obsm is None:
            self.perturbations_in_obsm = ()
        else:
            # check compatibility
            # we should have both `perturbations` and `perturbation_reps`
            if self.perturbations is None:
                msg = f"When no perturbations are passed, also `perturbations_in_obsm` should be `None`."
                raise ValueError(msg)
            if self.perturbation_reps is None:
                msg = f"When no perturbations representations are passed, also `perturbations_in_obsm` should be `None`."
                raise ValueError(msg)

        # when only one perturbation is passed create a sequence with only one element
        if isinstance(self.perturbations_in_obsm, str):
            msg = f"Only one element provided in {self.perturbations_in_obsm=}. Setting it to a sequence."
            logger.info(msg)
            self.perturbations_in_obsm = (self.perturbations_in_obsm, )

        # we should check that it is a sequence of string identifiers
        if not isinstance(self.perturbations_in_obsm, Sequence):
            msg = f"`perturbations_in_obsm` should be a sequence of string perturbation identifiers, found {type(self.perturbations_in_obsm)}"
            raise TypeError(msg)

        # handling optional perturbations
        if self.perturbations is not None:
            # when only one perturbation is passed create a sequence with only one element
            if isinstance(self.perturbations, str):
                msg = f"Only one element provided in {self.perturbations=}. Setting it to a sequence."
                logger.info(msg)
                self.perturbations = (self.perturbations,)

            # we should check that it is a sequence of string identifiers
            if not isinstance(self.perturbations, Sequence):
                msg = f"`perturbations` should be a sequence of string perturbation identifiers, found {type(self.perturbations)}"
                raise TypeError(msg)

            # when we pass the perturbations we should pass their representations as well
            if self.perturbation_reps is None:
                msg = f"When passing perturbations, you need to provide their representation in `self.perturbation_reps`."
                raise ValueError(msg)

            # check each perturbation individually
            for perturbation in self.perturbations:
                # check that each perturbation identifiers is of the expected type    
                if not isinstance(perturbation, str):
                    msg = f"Perturbation {perturbation} is expected to be a string, found {type(perturbation)}"
                    raise TypeError(msg)

                # check that it appears in the rep dictionary                    
                if not perturbation in self.perturbation_reps.keys():
                    msg = f"Perturbation {perturbation} not found in `self.perturbation_reps`."
                    raise KeyError(msg)

                # retrieving the representations for the current perturbation
                reps = self.perturbation_reps[perturbation]

                # configuring covariate metadata
                if perturbation in self.perturbations_in_obsm:
                    # check if perturbation is in obsm
                    reps = self.__configure_covariate_metadata(reps, "obsm", allow_only_one_element=True)
                else:
                    # sanity check on the input AnnData
                    if perturbation not in self.adata.obs.keys():
                        msg = f"{perturbation} not found in `adata.obs.keys()`"
                        raise ValueError(msg)

                    # require that we have the same keys across all the representations of a given perturbation
                    reference_keys = list(self.adata.uns[reps[0]].keys())
                    for rep in reps:
                        keys = list(self.adata.uns[rep].keys())
                        if reference_keys != keys:
                            msg = f""
                            raise ValueError(msg)

                    # finally retrieve the representation
                    reps = self.__configure_covariate_metadata(reps, "uns")
                # store the parsed representations back in perturbation_reps
                self.perturbation_reps[perturbation] = reps

                # optionally retrieving the covariates for the current perturbation
                if self.perturbation_covariates is not None:
                    if perturbation in self.perturbation_covariates.keys():
                        # retrieving the covariates for the current perturbation
                        covariates = self.perturbation_covariates[perturbation]
                        covariates = self.__configure_covariate_metadata(covariates, "obsm")
                    # otherwise initialize with empty sequence
                    else:
                        covariates = ()
                    # store the parsed covariates back in the perturbation reps
                    self.perturbation_covariates[perturbation] = covariates

    def __configure_covariate_metadata(
        self,
        identifiers: Sequence[str] | str,
        adata_field_key: Literal["uns", "obs"],
        allow_only_one_element: bool = False
    ) -> Sequence[str] | str:
        """"""
        # retrieving adata field
        adata_field = getattr(self.adata, adata_field_key)

        # when only one identifier is passed create a sequence with only one element
        if isinstance(identifiers, str):
            msg = f"Only one element provided in {identifiers=}. Setting it to a sequence."
            logger.info(msg)
            identifiers = (identifiers, )
        
        # optionally check that we only have one identifier
        if allow_only_one_element:
            if isinstance(identifiers, Sequence):
                if len(identifiers) != 1:
                    msg = "When a perturbation is in .obsm, there should be only one representation."
                    raise ValueError(identifiers)

        # checking that the representations are of the correct type
        if not isinstance(identifiers, Sequence):
            msg = f" for perturbation {identifiers} should be a string representation identifier, found {type(identifiers)}."
            raise TypeError(msg)

        # checking that the representations are found in `self.adata.uns`
        for identifier in identifiers:
            if identifier not in adata_field:
                msg = f"Representation {identifier} not found in `self.adata.{adata_field_key}.keys()`."
                raise KeyError(msg)
        
        # optionally returning the only one element of the sequence
        if allow_only_one_element:
            return identifiers[0]
        return identifiers

    def __get_data(
        self,
        perturbation: str,
        adata: anndata.AnnData,
    ) -> dict[str, np.ndarray]:
        """"""
        # initializing output dictionary
        perturbation_data = {}

        # retrieving representation
        reps = self.perturbation_reps[perturbation]

        # when perturbation is in obsm
        if perturbation in self.perturbations_in_obsm:
            # storing data
            covariate_feats_key = f"{DataFields.CONDITION_FEATS}_{perturbation}_{reps}" 
            perturbation_data[covariate_feats_key] = adata.obsm[reps]
            return perturbation_data
        
        # storing representations
        covariate_data = adata.obs[perturbation].values
        # iterating over each representation
        for rep in reps:
            # retrieving the representation dict
            rep_dict = adata.uns[rep]
            # mapping each observation condition to their representation
            covariate_reps = [rep_dict[covariate] for covariate in covariate_data]
            covariate_reps = np.stack(covariate_reps, axis=0)
            # storing data
            covariate_rep_key = f"{DataFields.CONDITION_REP}_{perturbation}_{rep}"
            perturbation_data[covariate_rep_key] = covariate_reps
        
        # optionally retrieving additional covariates
        if self.perturbation_covariates is not None:
            # retrieving covariates
            covariates = self.perturbation_covariates[perturbation]
            # iterating over each covariate
            for covariate in covariates:
                # retrieving representation
                covariate_data = adata.obsm[covariate]
                # adding dimension when ndim == 1
                if covariate_data.ndim == 1:
                    covariate_data = covariate_data.reshape(-1, 1)
                # storing the results
                covariate_cov_key = f"{DataFields.CONDITION_COV}_{perturbation}_{covariate}"
                perturbation_data[covariate_cov_key] = covariate_data
        return perturbation_data

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> dict[str, np.ndarray]:
        """"""
        # handling adata
        adata = self._resolve_adata(adata)
        
        # raise error when we do not have perturbations
        if self.perturbations is None:
            msg = f"No perturbation passed, cannot retrieve the data."
            raise RuntimeError(msg)

        # otherwise retrieve the data for each perturbation
        data = {}
        # irerating over each perturbation
        for perturbation in self.perturbations:
            # retrieving data for current perturbation
            perturbation_data = self.__get_data(perturbation, adata)
            # updating data dictionary with data for current perturbation
            data.update(perturbation_data)
        return data 

    @property
    def allow_grouped_couplings(
        self,
    ) -> bool:
        """Flags indicating whether the data configuration allows for grouped coupling.

        Grouped couplings are not allowed in the following cases:
            1. Unconditional generation (i.e.: :attr: `self.perturbation` is `None`).
            2. No unique representations provided for the perturbations (i.e.: :attr: `self.perturbation_reps` is `None`).
            3. At least one perturbation is provided in the :attr: `self.perturbations_in_obsm`, in which case the
                perturbation value will be unique for each cell.
        """
        # no perturbation found
        if self.perturbations is None:
            return False
        # no representation found
        if self.perturbation_reps is None:
            return False
        # perturbation in obsm found
        if len(self.perturbations_in_obsm) > 0:
            return False
        return True

    @property
    def seen_combinations(
        self,
    ) -> Sequence[Sequence[str]] | None:
        """"""
        # no perturbation to group over
        if not self.allow_grouped_couplings:
            return None
        combs = self.adata.obs[[pert for pert in self.perturbations]].drop_duplicates().values.tolist()
        return [tuple(comb) for comb in combs]


@dataclass
class TargetDataSchema(BaseDataSchema):
    """"""

    adata: anndata.AnnData
    target_covariates: dict[str, Literal["one_hot", "label", "identity"]]
    target_covariates_in_obsm: Sequence[str] | None = None,
    target_covariates_kwargs: dict[str, Any] | None = None,

    def __post_init__(
        self,
    ) -> None:
        """"""

        # handling optional covariate keywargs
        if self.target_covariates_kwargs is None:
            self.target_covariates_kwargs = {}
            for target_covariate in self.target_covariates.keys():
                self.target_covariates_kwargs[target_covariate] = {}

        # handling optional covariates in obsm
        if self.target_covariates_in_obsm is None:
            self.target_covariates_in_obsm = ()

        # checking that each target covariate appears in the anndata object
        for target_covariate in self.target_covariates.keys():
            # when the target is in obsm
            if target_covariate in self.target_covariates_in_obsm:
                if not target_covariate in self.adata.obsm.keys():
                    msg = f"{target_covariate} not found in `adata.obsm.keys()`"
                    raise KeyError(msg)

            # otherwise we should find it in obs
            else:
                if target_covariate not in self.adata.obs.keys():
                    msg = f"{target_covariate} not found in `adata.obs.columns`"
                    raise KeyError(msg)

    def __get_data(
        self,
        target_covariate: str,
        adata: anndata.AnnData,
    ) -> np.ndarray:
        """"""

        # when target covariate is in obsm
        if target_covariate in self.target_covariates_in_obsm:
            # retrieving target data
            target_data = adata.obsm[target_covariate]
            # adding dimension when ndim == 1
            if target_data.ndim == 1:
                target_data = target_data.reshape(-1, 1)
            return target_data

        # retrieving target covariate representation
        target_covariate_reps = self.target_covariates[target_covariate]
        # retrieving the keywod argument to get the target representation
        covariate_target_rep_kwargs = self.target_covariates_kwargs[target_covariate]
        # Collect the condition target covariate from the adata.obs 
        covariate_data = adata.obs[[target_covariate]].values

        # one hot encoding
        if target_covariate_reps == "one_hot":
            covariate_data = one_hot_encode(covariate_data, covariate_target_rep_kwargs)

        # label encoding
        elif target_covariate_reps == "label":
            covariate_data = label_encode(covariate_data, covariate_target_rep_kwargs) 

        # no encoding
        elif target_covariate_reps == "identity":
            if covariate_data.ndim == 1:
                covariate_data = covariate_data.reshape(-1, 1)

        # value error otherwise
        else:
            msg = f"{target_covariate_reps=} not currently supported (avaiable options are `['one_hot', 'label', 'identity']`)"
            raise ValueError(msg)
        return covariate_data

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> dict[str, np.ndarray]:
        """"""
        # handling adata
        adata = self._resolve_adata(adata)
        
        # otherwise retrieve the data for each perturbation
        data = {}
        # irerating over each perturbation
        for target_covariate in self.target_covariates.keys():
            # retrieving data for current perturbation
            target_data = self.__get_data(target_covariate, adata)
            # updating data dictionary with data for current perturbation
            data[target_covariate] = target_data
        return data
