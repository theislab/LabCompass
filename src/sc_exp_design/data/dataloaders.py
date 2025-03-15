import abc
from collections.abc import Sequence
from typing import Literal

import random
import torch

from sc_exp_design.constants import DataFields
from sc_exp_design.couplings import Coupling, OTCoupling
from sc_exp_design.data.data import PredictionData, TrainData
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike

__all__ = [
    "BaseDataLoader",
    "TrainDataLoader",
    "ValidationDataLoader",
    "PredictionDataLoader",
]


class BaseDataLoader(abc.ABC):
    """"""

    @abc.abstractmethod
    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """"""
        raise NotImplementedError


class TrainDataLoader:
    """
    Data loader for training that samples matched control and perturbed cell states.
    """

    def __init__(
        self,
        data: TrainData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """
        Initializes the training data loader.

        :param data: Training dataset containing control and perturbed cell states.
        :type data: class:`TrainData`
        :param coupling: Coupling strategy used to match control and perturbed states.
        :type coupling: class:`Coupling`
        :param batch_size: Number of samples per batch.
        :type batch_size: class:`int`
        :param state_transforms: Optional transformations applied to the states, defaults to `None`.
        :type state_transforms: class:`Transform`, optional
        :param device_id: Device to use for tensor operations (`cuda` or `cpu`), defaults to `cuda`.
        :type device_id: class:`Literal[\"cuda\", \"cpu\"]`, optional
        """
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)

    def __sample_perturbation_id(
        self,
    ) -> Sequence[str]:
        """
        Samples the treatment for the current batch when using Optimal Transport couplings.
        This is needed as the OT problem should be solved individually for each perturbation.
        
        :return: Integer containing the index for the perturbation used in the current batch of data.
        :rtype: int
        """
        # no perturbation data is passed to the TrainData object
        if self.data.seen_combinatorial_perturbations is None:
            return None
        # need to sample one perturbation from the set of unique perturbations
        return random.choice(self.data.seen_combinatorial_perturbations)

    def sample(self) -> dict[str, TensorLike]:
        """
        Samples a batch of matched control and perturbed cell states.

        :return: Dictionary containing source (control) states, target (perturbed) states,
                 and optional perturbation representations.
        :rtype: dict[str, TensorLike]
        """
        # control states
        ctrl_data = self.data.get_controls(self.batch_size)
        ctrl_states = ctrl_data[DataFields.STATE_DATA]

        # sampling treatments for current batch needed for OT couplings when we sample only one condition per batch
        treatments = None
        if isinstance(self.coupling, OTCoupling):
            treatments = self.__sample_perturbation_id()

        # treatment states
        trtm_data = self.data.get_treatments(self.batch_size, treatments)
        trtm_states = trtm_data[DataFields.STATE_DATA]

        # matching the two groups
        source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

        # moving states to torch tensors
        source = torch.from_numpy(ctrl_states[source_idx]).to(self.device).float()
        target = torch.from_numpy(trtm_states[target_idx]).to(self.device).float()

        # handling transformations
        if self.state_transforms is not None:
            source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)

        # constructing output dictionary
        out_dict = {DataFields.SOURCE_STATE: source, DataFields.TARGET_STATE: target}

        # handling perturbation data
        if self.data.perturbation_data is not None:
            trtm_perts = trtm_data[DataFields.PERTURBATION_DATA]
            condition = {cond: torch.from_numpy(cond_data[target_idx]).to(self.device).float()
                         for cond, cond_data in trtm_perts.items()}
            out_dict[DataFields.PERTURBATION_DATA] = condition
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = trtm_data[DataFields.PERTURBATION_TARGET_REPR]
            trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]).to(self.device).float()
                                     for key, val in trtm_perts_target_rep.items()}
            out_dict[DataFields.PERTURBATION_TARGET_REPR] = trtm_perts_target_rep
        
        return out_dict

class ValidationDataLoader:
    """
    Data loader for validation that samples matched control and perturbed cell states.
    """

    def __init__(
        self,
        data: PredictionData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """
        Initializes the validation data loader.

        :param data: Validation dataset containing control and perturbed cell states.
        :type data: class:`PredictionData`
        :param coupling: Coupling strategy used to match control and perturbed states.
        :type coupling: class:`Coupling`
        :param batch_size: Number of samples per batch.
        :type batch_size: class:`int`
        :param state_transforms: Optional transformations applied to the states, defaults to `None`.
        :type state_transforms: class:`Transform`, optional
        :param device_id: Device to use for tensor operations (`cuda` or `cpu`), defaults to `cuda`.
        :type device_id: class:`Literal[\"cuda\", \"cpu\"]`, optional
        """
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)

    def sample(self) -> dict[str, TensorLike]:
        """
        Samples a batch of matched control and perturbed cell states for validation.

        :return: Dictionary containing source (control) states, target (perturbed) states,
                 and optional perturbation representations.
        :rtype: dict[str, TensorLike]
        """
        ctrl_data = self.data.get_controls(self.batch_size)
        ctrl_states = ctrl_data[DataFields.STATE_DATA]

        trtm_data = self.data.get_treatments(self.batch_size)
        trtm_states = trtm_data[DataFields.STATE_DATA]

        if self.data.perturbation_data is not None:
            trtm_perts = trtm_data[DataFields.PERTURBATION_DATA]
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = trtm_data[DataFields.PERTURBATION_TARGET_REPR]

        source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

        source = torch.from_numpy(ctrl_states[source_idx]).to(self.device).float()
        target = torch.from_numpy(trtm_states[target_idx]).to(self.device).float()

        if self.state_transforms is not None:
            source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)
            
        if self.data.perturbation_data is not None:
            condition = {key: torch.from_numpy(val[target_idx]).to(self.device).float()
                         for key, val in trtm_perts.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]).to(self.device).float()
                                     for key, val in trtm_perts_target_rep.items()}
        
        out_dict = {DataFields.SOURCE_STATE: source, DataFields.TARGET_STATE: target}
        out_dict[DataFields.PERTURBATION_DATA] = {}
        
        if self.data.perturbation_data is not None:
            out_dict[DataFields.PERTURBATION_DATA] = condition
        if self.data.target_perturbation_repr is not None:
            out_dict[DataFields.PERTURBATION_TARGET_REPR] = trtm_perts_target_rep
        
        return out_dict

class PredictionDataLoader:
    """"""
