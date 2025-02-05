import abc
from dataclasses import dataclass
from typing import Any

import anndata
import numpy as np

from sc_exp_design.constants import (
    PERTURBATION_DATA_KEY,
    PERTURBATION_TARGET_REPR_KEY,
    STATE_DATA_KEY,
)
from sc_exp_design.types import TensorLike

__all__ = [
    "BaseDataStruct",
    "TrainData",
    "PredictionData",
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
class TrainData(BaseDataStruct):
    """
    Data structure for training data containing control and perturbation information.
    """
    
    adata: anndata.AnnData
    control_key: str
    state_data: TensorLike
    perturbation_data: dict[str, TensorLike] | None
    target_perturbation_repr: dict[str, TensorLike] | None = None

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
        output_dict = {STATE_DATA_KEY: ctrl_state_data,}        
        if self.perturbation_data is not None:
            output_dict[PERTURBATION_DATA_KEY] = ctrl_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[PERTURBATION_TARGET_REPR_KEY] = ctrl_pert_repr
        return output_dict

    def get_treatments(
        self,
        batch_size: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Retrieve treatment group data.

        :param batch_size: Number of samples to return. If None, all treatments are returned.
        :type batch_size: int | None
        :return: Dictionary containing treatment state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # collect treatment ids and features
        trtm_obs_idx = np.argwhere(self.adata.obs[self.control_key] == False)[:, 0]
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
        output_dict = {STATE_DATA_KEY: trtm_state_data,}
        if self.perturbation_data is not None:
            output_dict[PERTURBATION_DATA_KEY] = trtm_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[PERTURBATION_TARGET_REPR_KEY] = trtm_pert_repr
        return output_dict


@dataclass
class PredictionData(BaseDataStruct):
    """
    Data structure for training data containing control and perturbation information.
    """

    adata: anndata.AnnData
    control_key: str
    state_data: TensorLike
    perturbation_data: dict[str, TensorLike] | None
    target_perturbation_repr: dict[str, TensorLike] | None = None

    def get_controls(
        self,
        batch_size: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Retrieve control group data.

        :param batch_size: Number of samples to return. If None, all controls are returned.
        :type batch_size: int | None
        :return: Dictionary containing control state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # collect control ids and features
        ctrl_obs_idx = np.argwhere(self.adata.obs[self.control_key] == True)[:, 0]
        ctrl_state_data = self.state_data[ctrl_obs_idx]

        # collect control annotations from perturbation data 
        if self.perturbation_data is not None:
            ctrl_perturbation_data = {key: val[ctrl_obs_idx] for key, val in self.perturbation_data.items()}
        if self.target_perturbation_repr is not None:
            ctrl_pert_repr = self.target_perturbation_repr[ctrl_obs_idx]

        # collect batch subset of the observations 
        if batch_size is not None:
            batch_idxs = np.random.choice(ctrl_obs_idx.shape[0], size=batch_size)

            ctrl_state_data = ctrl_state_data[batch_idxs]

            if self.perturbation_data is not None:
                ctrl_perturbation_data = {key: val[batch_idxs] for key, val in ctrl_perturbation_data.items()}
            if self.target_perturbation_repr is not None:
                ctrl_pert_repr = ctrl_pert_repr[batch_idxs]

        # dictionary of controls
        output_dict = {STATE_DATA_KEY: ctrl_state_data}

        if self.perturbation_data is not None:
            output_dict[PERTURBATION_DATA_KEY] =  ctrl_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[PERTURBATION_TARGET_REPR_KEY] = ctrl_pert_repr
        return output_dict

    def get_treatments(
        self,
        batch_size: int | None = None,
    ) -> tuple[TensorLike, TensorLike]:
        """
        Retrieve treatment group data.

        :param batch_size: Number of samples to return. If None, all treatments are returned.
        :type batch_size: int | None
        :return: Dictionary containing treatment state and perturbation data (if available).
        :rtype: Dict[str, TensorLike]
        """
        # collect treatment ids and features
        trtm_obs_idx = np.argwhere(self.adata.obs[self.control_key] == False)[:, 0]
        trtm_state_data = self.state_data[trtm_obs_idx]

        # collect treatment annotations from perturbation data 
        if self.perturbation_data is not None:
            trtm_perturbation_data = {key: val[trtm_obs_idx] for key, val in self.perturbation_data.items()}
        if self.target_perturbation_repr is not None:
            trtm_pert_repr = self.target_perturbation_repr[trtm_obs_idx]

        # collect batch subset of the observations
        if batch_size is not None:
            batch_idxs = np.random.choice(trtm_obs_idx.shape[0], size=batch_size)

            trtm_state_data = trtm_state_data[batch_idxs]
            if self.perturbation_data is not None:
                trtm_perturbation_data = {key: val[batch_idxs] for key, val in trtm_perturbation_data.items()}
            if self.target_perturbation_repr is not None:
                trtm_pert_repr = trtm_pert_repr[batch_idxs]

        # dictionary of controls
        output_dict = {STATE_DATA_KEY: trtm_state_data,}

        if self.perturbation_data is not None:
            output_dict[PERTURBATION_DATA_KEY] = trtm_perturbation_data
        if self.target_perturbation_repr is not None:
            output_dict[PERTURBATION_TARGET_REPR_KEY] = trtm_pert_repr
        return output_dict
