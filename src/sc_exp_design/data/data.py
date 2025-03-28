import abc
from collections.abc import Sequence
from itertools import product
from dataclasses import dataclass
from typing import Any

import anndata
import numpy as np

from sc_exp_design.constants import DataFields
from sc_exp_design.types import TensorLike

__all__ = [
    "BaseDataStruct",
    "AnnotatedPerturbationData",
]


class BaseDataStruct(abc.ABC):
    """
    Abstract base class for data structures used in modeling perturbations and controls.
    """
    
    @abc.abstractmethod
    def get_controls(
        self,
        *args,
        **kwargs,
    ) -> Any:
        """"""
        raise NotImplementedError


@dataclass
class AnnotatedPerturbationData(BaseDataStruct):
    """
    Data structure for training data containing control and perturbation information.
    """
    
    adata: anndata.AnnData
    control_key: str | None
    state_data: TensorLike
    perturbation_data: dict[str, TensorLike] | None
    target_perturbation_repr: dict[str, TensorLike] | None = None
    perturbations_with_rep: dict[str, Sequence[str]] | None = None
    has_controls: bool = True

    @property
    def seen_combinatorial_perturbations(
        self,
    ) -> list[list[str]] | None:
        """"""
        # no perturbation data is passed to the AnnotatedPerturbationData object or no perturbation with associated representation
        if (self.perturbation_data is None) or (self.perturbations_with_rep is None):
            return None
        return self.adata.obs[[pert for pert in self.perturbations_with_rep.keys()]].drop_duplicates().values.tolist()

    @property
    def num_seen_combinatorial_perturbations(
        self,
    ) -> int | None:
        """"""
        if self.seen_combinatorial_perturbations is None:
            return None
        return len(self.seen_combinatorial_perturbations)

    def get_controls(
        self,
        batch_size: int | None = None,
    ) -> dict[str, TensorLike]:
        """
        Retrieve control group data.

        :param batch_size: Number of samples to return. If None, all controls are returned.
        :type batch_size: int | None
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
        ctrl_obs_idx = np.argwhere(self.adata.obs[self.control_key] == True)[:, 0]
        ctrl_state_data = self.state_data[ctrl_obs_idx]

        # collect control annotations from perturbation data 
        if self.perturbation_data is not None:
            ctrl_perturbation_data = {key: val[ctrl_obs_idx] for key, val in self.perturbation_data.items()}
        if self.target_perturbation_repr is not None:
            ctrl_pert_repr = {key: val[ctrl_obs_idx] for key, val in self.target_perturbation_repr.items()}

        # collect batch subset of the observations 
        if batch_size is not None:
            batch_idxs = np.random.choice(ctrl_obs_idx.shape[0], size=batch_size)

            ctrl_state_data = ctrl_state_data[batch_idxs]

            if self.perturbation_data is not None:
                ctrl_perturbation_data = {key: val[batch_idxs] for key, val in ctrl_perturbation_data.items()}
            if self.target_perturbation_repr is not None:
                ctrl_pert_repr = {key: val[batch_idxs] for key, val in ctrl_pert_repr.items()}

        # Dictionary of controls 
        output_dict = {DataFields.STATE_DATA: ctrl_state_data,}        
        if self.perturbation_data is not None:
            output_dict[DataFields.PERTURBATION_DATA] = ctrl_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[DataFields.PERTURBATION_TARGET_REPR] = ctrl_pert_repr
        return output_dict

    def get_treatments(
        self,
        batch_size: int | None = None,
        treatments: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Retrieve treatment group data.

        :param batch_size: Number of samples to return. If None, all treatments are returned.
        :type batch_size: int | None

        :param treatment_ids: The identifier for the treatment to be sampled in the current batch.
            Defaults to `None`, in which case all individual treatments could be sampled.
        :type treatment_ids: int | None

        :return: Dictionary containing treatment state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # collect treatment ids and features
        if self.has_controls:
            trtm_obs_idx = np.argwhere(self.adata.obs[self.control_key] == False)[:, 0]
        else:
            trtm_obs_idx = np.arange(len(self.adata))
        # optionally selecting only the current treatment (used in case of OT couplings) 
        if treatments is not None:
            trtm_obs_idx = np.argwhere(self.adata.obs[[pert for pert in self.perturbations_with_rep.keys()]] == treatments)[:, 0]
        trtm_state_data = self.state_data[trtm_obs_idx]

        # collect treatment annotations from perturbation data 
        if self.perturbation_data is not None:
            trtm_perturbation_data = {key: val[trtm_obs_idx] for key, val in self.perturbation_data.items()}
        if self.target_perturbation_repr is not None:
            trtm_pert_repr = {key: val[trtm_obs_idx] for key, val in self.target_perturbation_repr.items()}
        
        # collect batch subset of the observations 
        if batch_size is not None:
            batch_idxs = np.random.choice(trtm_obs_idx.shape[0], size=batch_size)

            trtm_state_data = trtm_state_data[batch_idxs]
            
            if self.perturbation_data is not None:
                trtm_perturbation_data = {key: val[batch_idxs] for key, val in trtm_perturbation_data.items()}
            if self.target_perturbation_repr is not None:
                trtm_pert_repr = {key: val[batch_idxs] for key, val in trtm_pert_repr.items()}

        # dictionary of treatments 
        output_dict = {DataFields.STATE_DATA: trtm_state_data,}
        if self.perturbation_data is not None:
            output_dict[DataFields.PERTURBATION_DATA] = trtm_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[DataFields.PERTURBATION_TARGET_REPR] = trtm_pert_repr
        return output_dict

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, Any]:
        """"""
        # retrieving adata and states
        adata = self.adata[idx]
        state_data = self.state_data[idx]
        # retrieving optional data
        perturbation_data = None
        if self.perturbation_data is not None:
            perturbation_data = {perturbation: perturbation_data[idx] for perturbation, perturbation_data in self.perturbation_data.items()}
        target_perturbation_repr = None
        if self.target_perturbation_repr is not None:
            target_perturbation_repr = {target: target_data[idx] for target, target_data in self.target_perturbation_repr.items()}
        return AnnotatedPerturbationData(
            adata,
            self.control_key,
            state_data,
            perturbation_data=perturbation_data,
            target_perturbation_repr=target_perturbation_repr,
            perturbations_with_rep=self.perturbations_with_rep,
        )
    
    def __len__(
        self,
    ) -> int:
        """"""
        return self.adata.shape[0]
