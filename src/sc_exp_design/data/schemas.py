import abc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import anndata
import numpy as np

from sc_exp_design.constants import DataFields
from sc_exp_design.data.container import BatchMixin
from sc_exp_design.transforms.utils import label_encode, one_hot_encode

logger = logging.getLogger(__name__)

__all__ = [
    "StateDataSchema",
    "PerturbationDataModel",
    "TargetDataSchema",
]


@dataclass
class BaseDataSchema(abc.ABC):
    """Base class for handling data schemas given an underlying :object: `anndata.AnnData`.
    
    :param adata: The annotated data object on which to validate and enforce the data schema.
    :type adata: class: `anndata.AnnData`
    """

    adata: anndata.AnnData

    def resolve_adata(
        self,
        adata: anndata.AnnData | None = None,
    ) -> anndata.AnnData:
        """Resolves optional extra annotated data arguments.

        When external annotated data are provided, it enforces and uses its schema over them.
        Otherwise returns the :attr: `self.adata` object by default.

        :param adata: Optional external annotated data on which to enforce the data schema, defaults to `None`.
        :type adata: class: `anndata.AnnData | None`
        """
        # handling adata
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata
        return adata

    @abc.abstractmethod
    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(
        self,
    ) -> Any:
        """Enforces the data schema and returned the compiled data."""
        raise NotImplementedError


@dataclass
class StateDataSchema(BaseDataSchema):
    """Schema for handling state data. Currently supports only unimodal states.

    :param adata: The annotated data object on which to validate and enforce the data schema.
    :type adata: class: `anndata.AnnData`

    :param sample_rep: Optional key indicating the location of the sample representation
        within the :attr: `self.adata.obsm` object. When not provided, it will automatically retrieve
        the :attr: `self.adata.X` to be used a state representation.
    :type sample_rep: class: `str | None`
    """

    adata: anndata.AnnData
    sample_rep: str | None

    def __post_init__(
        self,
    ) -> None:
        """Performs sanity checks on the annotated data object.
        
        It verifies that the :attr: `sample_rep` key appear in :attr: `self.adata.obsm` when provided.
        """
        self._validate_adata(self.adata)

    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        # when we provide the sample rep key it should appear in `self.adata.obsm`
        if self.sample_rep is not None:
            if self.sample_rep not in adata.obsm.keys():
                msg = f"{self.sample_rep=} not found in `adata.obsm` (Keys found: {list(adata.obsm.keys())})"
                raise KeyError(msg)

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> np.ndarray:
        """Enforces the data schema and returns the compiled state data.
        
        :param adata: Optional annotated data object on which to enforce the schema.
            When not provided, it will automatically use :attr: `self.adata`.
        :type adata: class: `anndata.AnnData | None`
        """
        # handling adata
        adata = self.resolve_adata(adata)
        
        # validating data
        self._validate_adata(adata)

        # retrieving the X attribute when no sample rep provided
        if self.sample_rep is None:
            state_data = adata.X
        else:
            state_data = adata.obsm[self.sample_rep]
        return state_data


@dataclass
class PerturbationDataSchema(BaseDataSchema):
    """Schema for handling perturbation data.
    
    :param adata: The annotated data object on which to validate and enforce the data schema.
    :type adata: class: `anndata.AnnData`

    :param perturbations: Optional identifiers for the modeled perturbations.
        When a single string identifier is passed, it will be coerced to a sequence.
        Each element of :param: `perturbations` is required to have an associated
        representation in :param: `perturbation_reps`.
        Every identifier in :param: `perturbations` that is not appearing in :param: `perturbations_in_obsm`,
        should be the name of a column in :attr: `self.adata.obs`.
    :type perturbations: class: `str | Sequence[str] | None`

    :param perturbations_in_obsm: Optional identifiers for the perturbations associated
        to a continuous representation. When a string identifier is passed, it will be coerced to
        a sequence with only one element. When `None`, it will be coerced to an empty sequence.
        Each element in :param: `perturbations_in_obsm` should also be appearing in :param: `perturbations`.
        All perturbations appearing here, will need to have at most one representation passed in :param: `perturbation_reps`.
        Such perturbation representations will have to be found inside :attr: `self.adata.obsm`.
        When these are provided, it is not possible to use grouped couplings (i.e.: Optimal Transport).
    :type perturbations_in_obsm: class: `str | Sequence[str] | None`

    :param perturbation_covariates: Optional dictionary mapping perturbations in :param: `perturbations`
        to their modeled covariates to be found in :attr: `self.adata.obsm`.
    :type perturbation_covariates: class: `dict[str, str | Sequence[str]] | None`

    :param perturbation_reps: Optional dictionary mapping perturbations to their respective representation.
        For perturbations appearing inside :param: `perturbations_in_obsm`, such representation is to be found
        in :attr: `self.adata.obsm`. It will map the remaining perturbations to their modeled representation
        to be found in :attr: `self.adata.uns`. For each perturbation in the keys of :param: `perturbation_reps`
        the provided value should map to a dictioanry field in :attr: `self.adata.uns`, whose keys will
        have to match the unique values appearing in the :attr: `self.adata.obs[perturbation]` column. 
    :type perturbation_reps: class: `dict[str, str | Sequence[str]] | None`
    """

    adata: anndata.AnnData
    perturbations: str | Sequence[str]
    perturbation_reps: dict[str, str | Sequence[str]]
    perturbations_in_obsm: str | Sequence[str] | None
    perturbation_covariates: dict[str, str | Sequence[str]] | None

    def __post_init__(
        self,
    ) -> None:
        """"""
        # validate args
        self._validate_args()
        # validate adata
        self._validate_adata(self.adata)

    def _validate_args(
        self,
    ) -> None:
        """Performs sanity checks on the configurations and the annotated data object."""
        # sanity check perturbations_in_obsm
        # when not provided, initialize empty sequence
        if self.perturbations_in_obsm is None:
            self.perturbations_in_obsm = ()

        # when only one perturbation is passed create a sequence with only one element
        if isinstance(self.perturbations_in_obsm, str):
            msg = f"Only one element provided in {self.perturbations_in_obsm=}. Setting it to a sequence."
            logger.info(msg)
            self.perturbations_in_obsm = (self.perturbations_in_obsm, )

        # we should check that it is a sequence of string identifiers
        if not isinstance(self.perturbations_in_obsm, Sequence):
            msg = f"`perturbations_in_obsm` should be a sequence of string perturbation identifiers, found {type(self.perturbations_in_obsm)}"
            raise TypeError(msg)

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
                reps = self.__configure_covariate_metadata(reps, allow_only_one_element=True)
            else:
                # sanity check on the input AnnData
                if perturbation not in self.adata.obs.keys():
                    msg = f"{perturbation} not found in `adata.obs.keys()`"
                    raise ValueError(msg)

                # finally retrieve the representation
                reps = self.__configure_covariate_metadata(reps)
            # store the parsed representations back in perturbation_reps
            self.perturbation_reps[perturbation] = reps

            # optionally retrieving the covariates for the current perturbation
            if self.perturbation_covariates is not None:
                if perturbation in self.perturbation_covariates.keys():
                    # retrieving the covariates for the current perturbation
                    covariates = self.perturbation_covariates[perturbation]
                    covariates = self.__configure_covariate_metadata(covariates)
                # otherwise initialize with empty sequence
                else:
                    covariates = ()
                # store the parsed covariates back in the perturbation reps
                self.perturbation_covariates[perturbation] = covariates

    def __configure_covariate_metadata(
        self,
        identifiers: Sequence[str] | str,
        allow_only_one_element: bool = False
    ) -> Sequence[str]:
        """Configures the covariates metadata and performs some additional sanity checks
        
        :param identifiers: Sequence of covariate identifiers for the current perturbation.
        :type identifiers: class: `Sequence[str] | str`

        :param adata_field_key: Key indicating the field in the annotated data object 
            where such identifiers are to be found.
        :type adata_field_key: class `Literal["uns", "obsm"]`

        :param allow_only_one_element: Whether to allow for only one identifier to be passed.
            This is needed for all the perturbations is :attr: `self.perturbations_in_obsm`.
        :type allow_only_one_element: class: `bool`
        """
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
        return identifiers

    def __validate_covariate_metadata(
        self,
        adata: anndata.AnnData,
        identifiers: Sequence[str] | str,
        adata_field_key: Literal["uns", "obsm"],
    ) -> None:
        """"""
        # retrieving adata field
        adata_field = getattr(adata, adata_field_key)
        # checking that the representations are found in adata_field_key
        for identifier in identifiers:
            if identifier not in adata_field:
                msg = f"Representation {identifier} not found in `adata.{adata_field_key}.keys()`."
                raise KeyError(msg)

    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """"""
        # check each perturbation individually
        for perturbation in self.perturbations:
            # retrieving the representations for the current perturbation
            reps = self.perturbation_reps[perturbation]
            # configuring covariate metadata
            if perturbation in self.perturbations_in_obsm:
                self.__validate_covariate_metadata(adata, reps, "obsm")
            else:
                # sanity check on the input AnnData
                if perturbation not in adata.obs.keys():
                    msg = f"{perturbation} not found in `adata.obs.keys()`"
                    raise ValueError(msg)
                # require that we have the same keys across all the representations of a given perturbation
                reference_keys = list(adata.uns[reps[0]].keys())
                for rep in reps:
                    keys = list(adata.uns[rep].keys())
                    if reference_keys != keys:
                        msg = f""
                        raise ValueError(msg)
                self.__validate_covariate_metadata(adata, reps, "uns")
            # optionally retrieving the covariates for the current perturbation
            if self.perturbation_covariates is not None:
                if perturbation in self.perturbation_covariates.keys():
                    # retrieving the covariates for the current perturbation
                    covariates = self.perturbation_covariates[perturbation]
                    self.__validate_covariate_metadata(adata, covariates, "obsm")

    def __get_data(
        self,
        perturbation: str,
        adata: anndata.AnnData,
    ) -> dict[str, np.ndarray]:
        """Retrieves the data for a given perturbation identifier.
        
        :param perturbation: Identifier for the current perturbation for which to retrieve the data.
        :type perturbation: class: `str`

        :param adata: Annotated data object on which to enforce the data schema for the current perturbation.
        :type adata: class: `anndata.AnnData`
        """
        # initializing output dictionary
        perturbation_data = {}

        # retrieving representation
        reps = self.perturbation_reps[perturbation]

        # when perturbation is in obsm
        if perturbation in self.perturbations_in_obsm:
            # storing data
            reps = reps[0]
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
    ) -> BatchMixin:
        """Enforces the data schema and returns the compiled perturbation data.
        
        :param adata: Optional annotated data object on which to enforce the schema.
            When not provided, it will automatically use :attr: `self.adata`.
        :type adata: class: `anndata.AnnData | None`
        """
        # handling adata
        adata = self.resolve_adata(adata)

        # validating data
        self._validate_adata(adata)

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
        return BatchMixin(data) 

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
        # perturbation in obsm found
        if len(self.perturbations_in_obsm) > 0:
            return False
        return True

    def get_seen_combinations(
        self,
        adata: anndata.AnnData,
    ) -> Sequence[Sequence[str]] | None:
        """Returns a sequence of unique perturbation combinations appearing in the data."""
        # no perturbation to group over
        if not self.allow_grouped_couplings:
            return None
        combs = adata.obs[[pert for pert in self.perturbations]].drop_duplicates().values.tolist()
        return [tuple(comb) for comb in combs]


@dataclass
class TargetDataSchema(BaseDataSchema):
    """Schema for handling target data

    :param target_covariates: Dictionary mapping each string identifier for the target covariates to be loaded to their
        target representation: This can be either `"label"` for loading labels, `"one_hot"` for loading one hot encoded
        vectors or `"identity"` to keep the retrieved representation as is.
        These covariates are to be found as keys of the :attr: `obs` attribute of :param: `adata`, unless appearing
        in :param: `target_covariates_in_obsm`, in which case their representation is to be retrieved from :attr: `obsm`.
        Defaults to `None`.
    :type target_covariates: class: `dict[str, Literal["one_hot", "label", "identity"]] | None`
    
    :param target_covariates_in_obsm: Optional sequence of string identifiers indicating the target covariates to be retrieved from
        the :attr: `obsm` attribute of :param: `adata`, defaults to `None`.
    :type target_covariates_in_obsm: class: `Sequence[str] | None`.

    :param target_covariates_kwargs: Optional keyword arguments used to retrieve the desired representation for the target covariates.
        Defaults to `None`.
    :type target_covariates_kwargs: class: `dict[str, Any]`
    """

    adata: anndata.AnnData
    target_covariates: dict[str, Literal["one_hot", "label", "identity"]]
    target_covariates_in_obsm: Sequence[str] | None = None,
    target_covariates_kwargs: dict[str, Any] | None = None,

    def __post_init__(
        self,
    ) -> None:
        """"""
        # validate args
        self._validate_args()
        # validate adata
        self._validate_adata(self.adata)

    def _validate_args(
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

    def _validate_adata(
            self,
            adata: anndata.AnnData,
        ) -> None:

        # checking that each target covariate appears in the anndata object
        for target_covariate in self.target_covariates.keys():
            # when the target is in obsm
            if target_covariate in self.target_covariates_in_obsm:
                if not target_covariate in adata.obsm.keys():
                    msg = f"{target_covariate} not found in `adata.obsm.keys()`"
                    raise KeyError(msg)

            # otherwise we should find it in obs
            else:
                if target_covariate not in adata.obs.keys():
                    msg = f"{target_covariate} not found in `adata.obs.columns`"
                    raise KeyError(msg)

    def __get_data(
        self,
        target_covariate: str,
        adata: anndata.AnnData,
    ) -> np.ndarray:
        """
        Retrieves the data for a given target covariate identifier identifier.
        
        :param target_covariate: Identifier for the current target covariate for which to retrieve the data.
        :type target_covariate: class: `str`

        :param adata: Annotated data object on which to enforce the data schema for the current perturbation.
        :type adata: class: `anndata.AnnData`
        """

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
    ) -> BatchMixin:
        """Enforces the data schema and returns the compiled perturbation data.
        
        :param adata: Optional annotated data object on which to enforce the schema.
            When not provided, it will automatically use :attr: `self.adata`.
        :type adata: class: `anndata.AnnData | None`
        """
        # handling adata
        adata = self.resolve_adata(adata)
        
        # validating data
        self._validate_adata(adata)

        # otherwise retrieve the data for each perturbation
        data = {}
        # irerating over each perturbation
        for target_covariate in self.target_covariates.keys():
            # retrieving data for current perturbation
            target_data = self.__get_data(target_covariate, adata)
            # updating data dictionary with data for current perturbation
            data[target_covariate] = target_data
        return BatchMixin(data)
