import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import anndata
import numpy as np

from labcompass.constants import DataFields
from labcompass.data.container import BatchMixin
from labcompass.transforms.utils import label_encode, one_hot_encode
from labcompass.utils import coerce_string_to_sequence

__all__ = [
    "StateDataSchema",
    "PerturbationDataSchema",
    "TargetDataSchema",
]


@dataclass
class BaseDataSchema(abc.ABC):
    """Base class for handling data schemas given an underlying :object: `anndata.AnnData`.
    
    :param adata: The annotated data object on which to validate and enforce the data schema.
    :type adata: class: `anndata.AnnData`
    """

    adata: anndata.AnnData

    @abc.abstractmethod
    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """Verifies the input :class: `anndata.AnnData`. This method should be overridden by derived classes.
        
        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_data(
        self,
    ) -> Any:
        """Enforces the data schema and returnes the compiled data."""
        raise NotImplementedError

    def __validate_covariates_metadata(
        self,
        adata: anndata.AnnData,
        identifier: str,
        adata_field_key: Literal["obs", "uns", "obsm"],
    ) -> None:
        """Checks that a given key identifier is present in a selected attribute of the input annotated data.

        When such key is not found in the target field, it raises a :class: `KeyError`.
        
        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`

        :param identifier: The string identifier for the key to be searched in :param: `adata`.
        :type identifier: class: `str`

        :param adata_field: The attribute of :param: `adata` checked.
        :type adata_field: class: `Literal["obs", "uns", "obsm"]`
        """
        # retrieving adata field
        adata_field = getattr(adata, adata_field_key)
        # checking that the representation is found in adata_field_key
        if identifier not in adata_field.keys():
            msg = f"Representation {identifier} not found in `adata.{adata_field_key}.keys()`."
            raise KeyError(msg)

    def _validate_covariates_metadata(
        self,
        adata: anndata.AnnData,
        identifiers: Sequence[str] | str,
        adata_field_key: Literal["obs", "uns", "obsm"],
    ) -> None:
        """Checks that a sequence of keys is present in a selected attribute of the input annotated data.

        When a key is not found in the target field, it raises a :class: `KeyError`. Coerces single string
        to a sequence then calls :method: `self.__validate_covariates_metadata` while iterating over it.
        
        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`

        :param identifiers: A sequence of string identifiers for the key to be searched in :param: `adata`.
            When a single string identifier, it will be coerced to a sequence.
        :type identifiers: class: `Sequence[str] | str`

        :param adata_field: The attribute of :param: `adata` checked.
        :type adata_field: class: `Literal["obs", "uns", "obsm"]`
        """
        # handling the case when identifier is a single string
        if isinstance(identifiers, str):
            identifiers = (identifiers, )
        # checking that the representations are found in adata_field_key
        for identifier in identifiers:
            self.__validate_covariates_metadata(adata, identifier, adata_field_key)
        
    def resolve_adata(
        self,
        adata: anndata.AnnData | None = None,
    ) -> anndata.AnnData:
        """Resolves optional extra annotated data arguments.

        When external annotated data are provided, it enforces and uses its schema over them.
        Otherwise returns the :attr: `self.adata` object by default.

        :param adata: Optional external annotated data on which to enforce the data schema, defaults to `None`
            in which case :attr: `self.adata` is returned.
        :type adata: class: `anndata.AnnData | None`
        """
        # handling adata
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata
        return adata


@dataclass
class StateDataSchema(BaseDataSchema):
    """Schema for handling state data. Currently supports only unimodal states.

    :param adata: The annotated data object on which to validate and enforce the data schema.
    :type adata: class: `anndata.AnnData`

    :param sample_rep: Optional key indicating the location of the sample representation
        within the :attr: `self.adata.obsm` object. When not provided, it will automatically retrieve
        the :attr: `self.adata.X` to be used as state representation.
    :type sample_rep: class: `str | None`
    """

    adata: anndata.AnnData
    sample_rep: str | None

    def __post_init__(
        self,
    ) -> None:
        """Performs sanity checks on the annotated data object.
        
        It verifies that the :attr: `sample_rep` key appears in :attr: `self.adata.obsm` when provided.
        """
        self._validate_adata(self.adata)

    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """Verifies the input :class: `anndata.AnnData`.

        When :attr: `self.sample_rep` is not `None`, checks that it is present in the keys of :attr: `self.adata.obsm`.

        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`
        """
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
        The remaining identifiers should instead be names of columns in :attr: `self.adata.obs`, which
        to retrieve the perturbations applied to each observation.
    :type perturbations: class: `str | Sequence[str] | None`

    :param perturbation_reps: Optional dictionary mapping perturbations to their respective representation.
        For perturbations appearing inside :param: `perturbations_in_obsm`, such representation is to be found
        in :attr: `self.adata.obsm`. It will map the remaining perturbations to their modeled representation
        to be found in :attr: `self.adata.uns`. For each perturbation in the keys of :param: `perturbation_reps`
        the provided value should map to a dictionary field in :attr: `self.adata.uns`, whose keys will
        have to match the unique values appearing in the :attr: `self.adata.obs[perturbation]` column.
    :type perturbation_reps: class: `dict[str, str | Sequence[str]] | None`

    :param perturbations_in_obsm: Optional identifiers for the perturbations associated
        to a continuous and observation-unique representation. When a string identifier is passed, it will be coerced to
        a sequence with only one element. When `None`, it will be coerced to an empty sequence.
        Each element in :param: `perturbations_in_obsm` should also be appearing in :param: `perturbations`.
        All perturbations appearing here, will need to have at most one representation passed in :param: `perturbation_reps`.
        Such perturbation representations will have to be found inside :attr: `self.adata.obsm`.
        When these are provided, it is not possible to use grouped couplings (i.e.: Optimal Transport).
    :type perturbations_in_obsm: class: `str | Sequence[str] | None`

    :param perturbation_covariates: Optional dictionary mapping perturbations in :param: `perturbations`
        to their modeled covariates to be found in :attr: `self.adata.obsm`.
        These will be the perturbation covariates that are associated to each perturbation on a cell-level basis, while still
        providing the possibility to use grouped couplings. 
        When passed, such covariates will need to be found inside :attr: `self.adata.obsm`.
    :type perturbation_covariates: class: `dict[str, str | Sequence[str]] | None`
    """

    adata: anndata.AnnData
    perturbations: str | Sequence[str]
    perturbation_reps: dict[str, str | Sequence[str]]
    perturbations_in_obsm: str | Sequence[str] | None
    perturbation_covariates: dict[str, str | Sequence[str]] | None

    def __post_init__(
        self,
    ) -> None:
        """Validates the arguments and :attr: `self.adata`.
        """
        # validate args
        self._validate_args()
        # validate adata
        self._validate_adata(self.adata)

    def _validate_perturbation_args(
        self,
        perturbation: str,
    ) -> None:
        """Validates the arguments associated to a given perturbation.

        For a given perturbation, it performs the following checks:
            * Checks that :param: `perturbation` is a string.

            * Checks that :param: `perturbation` appears inside :attr: `self.perturbation_reps`.

            * Ensures that the representations in :attr: `self.perturbation_reps[perturbation]`
                are a sequence of string identifiers as expected. For perturbations appearing in
                :attr: `self.perturbations_in_obsm`, it will raise an error when more than one
                representation is passed. For the remaining perturbations, it is possible to
                pass as many representations as desired.

            * When :attr: `self.perturbation_covariates` are passed, it ensures that it will contain
                each perturbation in :attr: `self.perturbations` as keys. For perturbations
                without any associated covariates, this will be set to an empty sequence.

        :param perturbation: The string identifier for the perturbation whose arguments to verify.
        :type perturbation: class: `str`
        """
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
            reps = coerce_string_to_sequence(reps, allow_only_one_element=True)
        else:
            # finally retrieve the representation
            reps = coerce_string_to_sequence(reps)
        # store the parsed representations back in perturbation_reps
        self.perturbation_reps[perturbation] = reps

        # optionally retrieving the covariates for the current perturbation
        if self.perturbation_covariates is not None:
            if perturbation in self.perturbation_covariates.keys():
                # retrieving the covariates for the current perturbation
                covariates = self.perturbation_covariates[perturbation]
                covariates = coerce_string_to_sequence(covariates)
            # otherwise initialize with empty sequence
            else:
                covariates = ()
            # store the parsed covariates back in the perturbation reps
            self.perturbation_covariates[perturbation] = covariates

    def _validate_perturbation_adata(
        self,
        adata: anndata.AnnData,
        perturbation: str,
    ) -> None:
        """Validates the arguments associated to a given perturbation against the input annotated data.

        For a given perturbation, it performs the following checks:
            * If the perturbation appears in :attr: `self.perturbations_in_obsm`,
                it checks that its representation appears in :attr: `adata.obsm`.

            * For the other perturbations, it verifies that they appear as column in :attr: `adata.obs`.

            * Checks that each representation passed in :attr: `self.perturbation_reps[perturbation]` 
                shares the same keys, which are then used to infer the unique values which to
                define groups for matched couplings on.

            * When :attr: `self.perturbation_covariates` are passed, it verifies that they appear in
                :attr: `adata.obsm` as expected.

        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData` 

        :param perturbation: The string identifier for the perturbation whose arguments to verify.
        :type perturbation: class: `str`
        """
        # retrieving the representations for the current perturbation
        reps = self.perturbation_reps[perturbation]
        # configuring covariate metadata
        if perturbation in self.perturbations_in_obsm:
            self._validate_covariates_metadata(adata, reps, "obsm")
        else:
            # sanity check on the input AnnData
            self._validate_covariates_metadata(adata, perturbation, "obs")
            # require that we have the same keys across all the representations of a given perturbation
            reference_keys = list(adata.uns[reps[0]].keys())
            for rep in reps:
                keys = list(adata.uns[rep].keys())
                if reference_keys != keys:
                    msg = f""
                    raise ValueError(msg)
            self._validate_covariates_metadata(adata, reps, "uns")
            # optionally retrieving the covariates for the current perturbation
            if self.perturbation_covariates is not None:
                if perturbation in self.perturbation_covariates.keys():
                    # retrieving the covariates for the current perturbation
                    covariates = self.perturbation_covariates[perturbation]
                    self._validate_covariates_metadata(adata, covariates, "obsm")

    def _get_perturbation_data(
        self,
        perturbation: str,
        adata: anndata.AnnData,
    ) -> dict[str, np.ndarray]:
        """Retrieves the data for a given perturbation identifier and returns it as dictionary of :class: `np.ndarray`.

        For a given perturbation it returns the data with keys in the following format:
            * For perturbations appearing in :attr: `self.perturbations_in_obsm`, the returned dictionary will
                contain only one mapping, whose key will be constructed as a underscore-separated string of the form
                `"features_[perturbation]_[perturbation_feature_id]"`, where `[perturbation]` is the name of the
                perturbation (i.e.: identifier passed in :attr: `self.perturbations`) and `[perturbation_feature_id]`
                is the name of the features associated to such perturbation (i.e: the value at :attr: `self.perturbation_reps[perturbation]`).

            * For the other perturbations, it will first retrieve the observation-level perturbation values from :attr: `adata.obs[perturbation]`.
                Then it will iterate over each representation in :attr: `self.perturbation_reps[perturbation]`.
                For each representation, it will use the dictionary in :attr: `adata.uns[representation]`
                (where `representation` is :attr: `self.perturbation_reps[perturbation]`) as lookup table.
                Such dictionary will have as keys the unique values of :attr: `adata.obs[perturbation]`, which will then be used
                to query the representation for each observation, then concatenates the resulting arrays over the first dimension.
                The output dictionary will then be updated to contain underscore-separated strings of the form `"repr_[perturbation]_[representation]"` as keys.

            * When perturbation covariates are passed in :attr: `self.perturbation_covariates`, it will retrieve each of them from :attr: `adata.obsm`.
                It will also make sure that they have a least two dimensions, by adding a dummy trailing dimension
                when the retrieved covariate are of shape `(adata.shape[0], )`.The perturbation covariates will be stored in the output dictionary
                inside keys constructing following the underscore-separeted pattern `"cov_[perturbation]_[covariate_name]"`.

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
            # retrieving the only representation
            reps = reps[0]
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

    def _validate_args(
        self,
    ) -> None:
        """Performs sanity checks on the configurations and updates the attributes for a valid configuration.
        """
        # configure perturbations in obsm
        self.perturbations_in_obsm = coerce_string_to_sequence(self.perturbations_in_obsm, allow_none=True)

        # configure perturbation 
        self.perturbations = coerce_string_to_sequence(self.perturbations)

        # check each perturbation individually
        for perturbation in self.perturbations:
            self._validate_perturbation_args(perturbation)

    def _validate_adata(
        self,
        adata: anndata.AnnData,
    ) -> None:
        """Validates the input annotated data.

        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`
        """
        # check each perturbation individually
        for perturbation in self.perturbations:
            self._validate_perturbation_adata(adata, perturbation)

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
            perturbation_data = self._get_perturbation_data(perturbation, adata)
            # updating data dictionary with data for current perturbation
            data.update(perturbation_data)
        return BatchMixin(data)

    def get_seen_combinations(
        self,
        adata: anndata.AnnData,
    ) -> Sequence[Sequence[str]] | None:
        """Returns a sequence of unique perturbation combinations appearing in the data.
        
        :param adata: The input annotated data which to retrieve the seen combinations from.
        :type adata: class: `anndata.AnnData` 
        """
        # no perturbation to group over
        if not self.allow_grouped_couplings:
            return None
        # get all unique values from the perturbation columns
        combs = adata.obs[[pert for pert in self.perturbations]].drop_duplicates().values.tolist()
        return [tuple(comb) for comb in combs]

    @property
    def allow_grouped_couplings(
        self,
    ) -> bool:
        """Flags indicating whether the data configuration allows for grouped coupling.

        Grouped couplings are not allowed when perturbations with unique representations are passed
        in :attr: `self.perturbations_in_obsm`.
        """
        # perturbation in obsm found
        if len(self.perturbations_in_obsm) > 0:
            return False
        return True


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
    target_covariates_in_obsm: Sequence[str] | None = None
    target_covariates_kwargs: dict[str, Any] | None = None

    def __post_init__(
        self,
    ) -> None:
        """Validates the arguments and :attr: `self.adata`.
        """
        # validate args
        self._validate_args()
        # validate adata
        self._validate_adata(self.adata)

    def _validate_args(
        self,
    ) -> None:
        """Performs sanity checks on the configurations and updates the attributes for a valid configuration.
        """

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
        """Validates the input annotated data.

        :param adata: The input annotated data to verify.
        :type adata: class: `anndata.AnnData`
        """
        # checking that each target covariate appears in the anndata object
        for target_covariate in self.target_covariates.keys():
            # when the target is in obsm
            if target_covariate in self.target_covariates_in_obsm:
                self._validate_covariates_metadata(adata, target_covariate, "obsm")
            # otherwise we should find it in obs
            else:
                self._validate_covariates_metadata(adata, target_covariate, "obs")

    def _get_data(
        self,
        target_covariate: str,
        adata: anndata.AnnData,
    ) -> np.ndarray:
        """
        Retrieves the data for a given target covariate identifier.
        
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
            target_data = self._get_data(target_covariate, adata)
            # updating data dictionary with data for current perturbation
            data[target_covariate] = target_data
        return BatchMixin(data)
