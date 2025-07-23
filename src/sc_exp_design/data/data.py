import abc
from collections.abc import Sequence
from itertools import product
from dataclasses import dataclass
from typing import Any

import anndata
import numpy as np

from sc_exp_design.constants import DataFields
from sc_exp_design.data.container import DataContainer, BatchMixin
from sc_exp_design.types import TensorLike

__all__ = [
    "AnnotatedPerturbationData",
]


@dataclass
class AnnotatedPerturbationData:
    """
    Data structure for annotated perturbation data.
    
    :param adata: The underlying annotated data object which to enforce the data model on.
        Should always be provided
    :type adata: class: `AnnData`

    :param control_key: Optional key in the :attr:`AnnData.obs` attribute of :param: `adata` where to retrieve
        the boolean flag indicating whether a given cells belongs to the control group or not. Should only be used
        when there exists some notion of control states, otherwise it would be preferable to simply generate from noise
        samples (check the :attr: `FlowMatching.generate_from_noise` attribute).
    :type control_key: class: `str | None`

    :param state_data: A tensor or an array containig the state data. Should always be provided
    :type state_data: class: `TensorLike`

    :param perturbation_data: Optional dictionary mapping each perturbation covariate to the corresponding data. In case
        of unconditional generation it will be `None`. Each key of such dictionary will represent an individual perturbation,
        while each value will be given by their numerical realization. A condition is then defined as a combination of values
        of such covariates for a given observation/cell.
    :type perturbation_data: class: `dict[str, TensorLike] | None`

    :param target_data: Optional dictionary mapping target covariates to be loaded in the case of inverse modeling
        to their representation. This will represent the quantities that we want to optimize for by choosing the perturbations, defaults to `None`.
    :type target_data:

    :param perturbations_with_rep: Optional dictionary mapping each perturbation covariate to its uniqua values. This is needed in the
        case of Optimal Transport couplings as we want to be able to sample a unique perturbation for each batch of target data, defaults to `None`.
        In the cases when :param: `perturbation_data` is `None`, it should be set to `None`. Similarly, it should be `None` in the case where it is not
        possible to use OT coupling, like for example when the perturbations are given by dense and continuous vectors of features (i.e.: when :param: `perturbations_in_obsm` is not `None`).
        Defaults to `None`.
    :type perturbations_with_rep: class: `dict[str, Sequence[str]] | None`

    :param has_controls: Flag indicating whether a notion of control states applies to the current data.
        When this is the case, the :param: `control_key` needs to be properly set. Defaults to `True`.
    :type has_controls: class: `bool`

    :param perturbations_in_obsm: Optional sequence of modeled perturbation whose representation is to be retrieved from the :attr: `osbm` attribte of the :param: `adata`.
        These should be perturbations representated by some continuous and dense feature vector. When this is not `None`, it is not possible to use Optimal Transport Couplings
        Defaulst to `None`.
    :type perturbations_in_obsm: class: `Sequence[str] | None`
    """
    
    adata: anndata.AnnData
    control_key: str | None
    state_data: TensorLike
    perturbation_data: BatchMixin | None
    target_data: BatchMixin | None = None
    seen_combinations: Sequence[Sequence[str]] | None = None
    has_controls: bool = True
    perturbations: Sequence[str] | None = None

    def __post_init__(
        self,
    ) -> None:
        """
        Registers the indices of control and treatment data for more efficient dataloading.
        """

        # pre-allocating attributes        
        self.control_idxs = None
        self.control_data = None

        self.treatment_idxs = None
        self.treatment_idxs_per_condition = None
        self.treatment_data = None

        # initializing data container
        self.data = DataContainer(
            self.state_data,
            self.perturbation_data,
            self.target_data,
        )

        # storing control data
        if self.has_controls:
            # sanity check
            msg = f""
            assert self.control_key is not None, msg

            # register indices and state data
            self.control_idxs = np.argwhere(self.adata.obs[self.control_key] == True)[:, 0]
            self.control_data = self.data[self.control_idxs]

            # storing perturbation data
            self.treatment_idxs = np.argwhere(self.adata.obs[self.control_key] == False)[:, 0]

        else:
            self.treatment_idxs = np.arange(len(self.adata))
        if self.perturbations is not None and self.seen_combinations is not None:
            self.treatment_idxs_per_condition = {
                DataFields.CONDITION_VALUES: self.treatment_idxs,
                **{
                    treatment: np.argwhere(self.adata.obs[[pert for pert in self.perturbations]] == treatment)[:, 0] 
                        for treatment in self.seen_combinations
                }
            }
        else:
            self.treatment_idxs_per_condition = {DataFields.CONDITION_VALUES: self.treatment_idxs}

    def __getitem__(
        self,
        idx: int | slice,
    ) -> "AnnotatedPerturbationData":
        """
        Durden method needed to slice the :class: `AnnotatedPerturbationData` object.

        Retrieves all the data from the original instance and returns a new instance of :class: `AnnotatedPerturbationData`.
        """
        # retrieving adata and states
        adata = self.adata[idx]
        state_data = self.state_data[idx]
        # retrieving optional data
        perturbation_data = None
        if self.perturbation_data is not None:
            perturbation_data = self.perturbation_data[idx]
        target_data = None
        if self.target_data is not None:
            target_data = self.target_data[idx]
        return AnnotatedPerturbationData(
            adata,
            self.control_key,
            state_data,
            perturbation_data=perturbation_data,
            target_data=target_data,
            seen_combinations=self.seen_combinations,
            perturbations_with_rep=self.perturbations_with_rep,
            has_controls=self.has_controls,
            perturbations=self.perturbations
        )
    
    def __len__(
        self,
    ) -> int:
        """
        Returns the number of observations present in the data.

        :rtype: class: `int`
        """
        return self.adata.shape[0]

    def _get_treatments(
        self,
        treatments: Sequence[str] | None = None,
    ) -> np.ndarray:
        """"""
        # case 0: Seen combinatorial perturbation is None. No specific treatment is to be retrieved.
        if self.seen_combinations is None:
            # sanity check: we should not pass the treatments
            msg = f""
            assert treatments is None, msg
            treatments = DataFields.CONDITION_VALUES
        # case 0: Seen combinatorial perturbation is not None.
        # treatment should be either None or be appering inside self.seen_combinatorial_perturbations
        else:
            # when no treatment is passed
            if treatments is None or len(treatments) == 0:
                treatments = DataFields.CONDITION_VALUES
            msg = f"{treatments=} not found in {self.treatment_idxs_per_condition.keys()=}"
            assert treatments in self.treatment_idxs_per_condition.keys(), msg

        # retrieving indices of current treatment and slicing data
        treatment_idxs = self.treatment_idxs_per_condition[treatments]
        return self.data[treatment_idxs]

    def get_controls(
        self,
        batch_size: int | None = None,
    ) -> DataContainer:
        """
        Retrieve control group data.
        
        Can only be called when :attr: `AnnotatedPerturbationData.has_controls` is `True` and :attr: `AnnotatedPerturbationData.control_key`
        is specified.

        :param batch_size: Number of samples to return. If `None`, all controls are returned, defaults to `None`.
        :type batch_size: class: `int | None`

        :return: Dictionary containing control state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # sanity check
        if self.has_controls:
            msg = f""
            assert self.control_key is not None, msg
        else:
            msg = "Controls are not available in this dataset (has_controls=False)."
            raise ValueError(msg)
        
        # collect control ids and features
        ctrl_data = self.control_data

        # collect batch subset of the observations 
        if batch_size is not None:
            batch_idxs = np.random.choice(np.arange(len(ctrl_data)), size=batch_size)

            ctrl_data = ctrl_data[batch_idxs]

        return self.data[batch_idxs]

    def get_treatments(
        self,
        batch_size: int | None = None,
        treatments: Sequence[str] | None = None,
    ) -> DataContainer:
        """
        Retrieve treatment group data.

        :param batch_size: Number of samples to return. If None, all treatments are returned. Deafults to `None`.
        :type batch_size: int | None

        :param treatments: The identifier for the treatments to be sampled in the current batch.
            Defaults to `None`, in which case all individual treatments could be sampled.
        :type treatment_ids: int | None

        :return: Dictionary containing treatment state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # retrieve indices
        trtm_data = self._get_treatments(treatments)
        
        # collect batch subset of the observations 
        if batch_size is not None:
            batch_idxs = np.random.choice(np.arange(len(trtm_data)), size=batch_size)

            trtm_data = trtm_data[batch_idxs]

        return trtm_data

    @property
    def allow_grouped_couplings(
        self,
    ) -> bool:
        """"""
        if self.seen_combinations is None:
            return False
        return True
