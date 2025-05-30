import logging
from collections.abc import Sequence
from typing import Any, Literal

import anndata
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sc_exp_design.constants import DataFields
from sc_exp_design.data.data import AnnotatedPerturbationData
from sc_exp_design.data.schemas import (
    StateDataSchema,
    PerturbationDataSchema,
    TargetDataSchema,
)
from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = [
    "DataManager",
]


class DataManager:
    """
    Class for managing perturbation-related data in single-cell experiments.
    """
    def __init__(
        self,
        adata: anndata.AnnData | None = None,
        sample_rep: str | None = None,
        control_key: str | None = None,
        perturbations: str | Sequence[str] | None = None,
        perturbations_in_obsm: str | Sequence[str] | None = None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None = None,
        perturbation_reps: dict[str, str | Sequence[str]] | None = None,
        load_target_covariates: bool = False,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None = None,
        target_covariates_in_obsm: Sequence[str] | None = None,
        target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Initializes the :class: `DataManager` object

        :param adata: The annotated data object which to retrieve the data to enforce the data model on.
        :type adata: class: `AnnData`

        :param sample_rep: Optional string identifier indicating the key in the :attr: `obsm` attribute of
            :param: `adata` where the state representation is held. If `None`, it will directly retrieve it from
            the :attr: `X` attribute of :param: `adata`, defaults to `None`.
        :type sample_rep: class: `str`

        :param control_key: Optional key in the :attr:`AnnData.obs` attribute of :param: `adata` where to retrieve
            the boolean flag indicating whether a given cells belongs to the control group or not. Should only be used
            when there exists some notion of control states, otherwise it would be preferable to simply generate from noise
            samples (check the :attr: `FlowMatching.generate_from_noise` attribute).
        :type control_key: class: `str | None`

        :param perturbations: Optional string identifiers for the perturbations whose effect we want to model.
            Each element of :param: `perturbations` should be present as a key inside `perturbation_reps`, otherwise it is
            simply ignored by the :class: `DataManager`. Defaults to `None`.
        :type perturbations: class: `str | Sequence[str] | None`

        :param perturbations_in_obsm: Optional sequence of modeled perturbation whose representation is to be retrieved from the :attr: `osbm` attribte of the :param: `adata`.
            These should be perturbations representated by some continuous and dense feature vector. When this is not `None`, it is not possible to use Optimal Transport Couplings
            Defaulst to `None`.
        :type perturbations_in_obsm: class: `str | Sequence[str] | None`

        :param perturbation_covariates: DIctionary mapping each perturbation in :param: `perturbations` to the set of its perturbation covariates.
            The perturbation covariates in :param: `perturbation_covariates` are associated to each perturbation on a cell level basis
            and are supposed to vary over the cells for a given perturbation. Such covariates are to be found in :attr: `adata.obsm`. 
            For example, for a given drug this could be given by the dosage or the time of the treatment application on a given cell.
            More generally, this can be given by any other feature associated to the current perturbation that can vary across cells.
            Defaults to `None`. 
        :type perturbation_covariates:

        :param perturbation_reps: Maps each perturbation in :param: `perturbations` to its target representation.
            The representations in :param:`perturbation_covariate_reps` are the same over cell with the same perturbation.
            For example, for a given drug this could be given by their chemical representation or any other feature that
            is constant across observations treated with the same perturbation.
            For all perturbations in :param: `perturbations` not appearing in :param: `perturbations_in_obsm`, 
            the values specified in :param: `perturbation_reps` should map to keys in :attr: `adata.uns` where to retrieve 
            the representations for their unique values. For the perturbations that instead appear in :param: `perturbations_in_obsm`,
            it should map to the corresponding representations to be found in :attr: `adata.obsm`. Defaults to `None`.
        :type perturbation_reps: class: `dict[str, str | Sequence[str]]`
 
        :param load_target_covariates: Flag indicating whether to load target covariates during the dataloading.
            This will represent the quantities that we want to optimize for by choosing the perturbations, defaults to `False`.
        :type load_target_covariates:
        
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

        # storing attributes
        self.adata = adata
        self.sample_rep = sample_rep
        self.control_key = control_key
        self.perturbations = perturbations
        self.perturbations_in_obsm = perturbations_in_obsm
        self.perturbation_covariates = perturbation_covariates
        self.perturbation_reps = perturbation_reps
        self.load_target_covariates = load_target_covariates
        self.target_covariates = target_covariates
        self.target_covariates_in_obsm = target_covariates_in_obsm
        self.target_covariates_kwargs = target_covariates_kwargs

        # initializing data schemas
        self.__init_schemas()

    def __init_schemas(
        self,
    ) -> None:
        """"""

        # state data schema
        self.state_data_schema = StateDataSchema(
            self.adata,
            self.sample_rep,
        )

        # perturbation data schema
        self.perturbation_data_schema = PerturbationDataSchema(
            self.adata,
            self.perturbations,
            self.perturbations_in_obsm,
            self.perturbation_covariates,
            self.perturbation_reps,
        )

        # target data schema
        self.target_data_schema = None
        if self.load_target_covariates:
            # sanity check
            if self.target_covariates is None:
                msg = f"With {self.load_target_covariates=} you need to specify the target covariate reprs in `target_covariates`, `None` found"
                raise ValueError(msg)
            self.target_data_schema = TargetDataSchema(
                self.adata,
                self.target_covariates,
                self.target_covariates_in_obsm,
                self.target_covariates_kwargs,
            )

    def get_data(
        self,
        adata: anndata.AnnData | None = None,
    ) -> AnnotatedPerturbationData:
        """
        :param adata: Annotated data object containing single-cell data.
        :type adata: class: `AnnData`

        :return: A structured object containing all necessary training inputs.
        :rtype: class: `AnnotatedPerturbationData`

        :raises ValueError: If both `adata` and `self.adata` are `None`.
        """
        # handling adata
        if adata is None and self.adata is None:
            msg = "Both `adata` and `self.adata` are None, you need to pass an `anndata.AnnData` object containing the data."
            raise ValueError(msg)
        elif adata is None:
            adata = self.adata

        # retrieving state data
        state_data = self.state_data_schema.get_data()

        # retrieving perturbation data
        perturbation_data = None
        if self.perturbations is not None:
            perturbation_data = self.perturbation_data_schema.get_data()

        # condition target representation
        target_data = None
        if self.load_target_covariates:
            target_data = self.target_data_schema.get_data()
        
        # constructing data object
        return AnnotatedPerturbationData(
            adata,
            self.control_key, 
            state_data,
            perturbation_data,
            target_data=target_data,
            seen_combinations=self.perturbation_data_schema.seen_combinations, 
            has_controls=self.has_controls,
            perturbations=self.perturbations,
        )

    @property
    def has_controls(
        self,
    ) -> bool:
        """Flag indicating whether a notion of control states applies to the current data.
        
        Automatically inferred by the presence of :attr: `self.control_key`    
        """
        if self.control_key is None:
            return False
        return True
