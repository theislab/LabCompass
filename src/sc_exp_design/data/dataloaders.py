import abc
from collections.abc import Sequence
from typing import Literal

import numpy as np
import random
import numpy as np
import torch

from sc_exp_design.constants import DataFields
from sc_exp_design.couplings import Coupling, OTCoupling
from sc_exp_design.data.data import AnnotatedPerturbationData
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike

__all__ = [
    "SequentialDataLoader",
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


class SequentialDataLoader(BaseDataLoader):
    """"""
    def __init__(
        self,
        data: AnnotatedPerturbationData,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda"
    ) -> None:
        """"""
        self.data = data
        self.batch_size = batch_size
        self.state_transforms = state_transforms
        self.device_id = device_id
        self.device = torch.device(self.device_id)
    
    def sample(
        self,
    ) -> dict[str, TensorLike]:
        """"""
        # sampling batch indices
        batch_idxs = np.random.choice(self.data.state_data.shape[0], size=self.batch_size)
        # slicing the state data
        states = self.data.state_data[batch_idxs]

        # moving states to torch tensors
        states = torch.from_numpy(states).to(self.device).float()
        # handling transformations
        if self.state_transforms is not None:
            states = self.state_transforms.transform(states)

        # constructing output dictionary
        out = {
            DataFields.STATE_DATA: states,
        }

        # retrieving optional petrurbation data
        if self.data.perturbation_data is not None:
            perturbation_data = {}
            for covariate, covariate_data in self.data.perturbation_data.items():
                perturbation_data[covariate] = torch.from_numpy(covariate_data[batch_idxs]).to(self.device).float()
            out[DataFields.PERTURBATION_DATA] = perturbation_data
        
        # retrieving optional target covariates
        if self.data.target_perturbation_repr is not None:
            target_data = {}
            for covariate, covariate_data in self.data.target_perturbation_repr.items():
                target_data[covariate] = torch.from_numpy(covariate_data[batch_idxs]).to(self.device).float()
            out[DataFields.TARGET_CATEGORIES] = target_data
        return out


class TrainDataLoader(BaseDataLoader):
    """
    Data loader for training that samples matched control and perturbed cell states.
    """

    def __init__(
        self,
        data: AnnotatedPerturbationData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
        has_controls: bool = True
    ) -> None:
        """
        Initializes the training data loader.

        :param data: Training dataset containing control and perturbed cell states.
        :type data: class:`AnnotatedPerturbationData`
        :param coupling: Coupling strategy used to match control and perturbed states.
        :type coupling: class:`Coupling`
        :param batch_size: Number of samples per batch.
        :type batch_size: class:`int`
        :param state_transforms: Optional transformations applied to the states, defaults to `None`.
        :type state_transforms: class:`Transform`, optional
        :param device_id: Device to use for tensor operations (`cuda` or `cpu`), defaults to `cuda`.
        :type device_id: class:`Literal[\"cuda\", \"cpu\"]`, optional
        :param noise_source: Controls if the source samples are Gaussian (True) or control cells (False).
        """
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)
        self.has_controls = has_controls

    def __sample_perturbation_id(
        self,
    ) -> Sequence[str]:
        """
        Samples the treatment for the current batch when using Optimal Transport couplings.
        This is needed as the OT problem should be solved individually for each perturbation.
        
        :return: Integer containing the index for the perturbation used in the current batch of data.
        :rtype: int
        """
        # no perturbation data is passed to the AnnotatedPerturbationData object
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
        # sampling treatments for current batch needed for OT couplings when we sample only one condition per batch
        treatments = None
        if isinstance(self.coupling, OTCoupling):
            treatments = self.__sample_perturbation_id()

        # treatment states
        trtm_data = self.data.get_treatments(self.batch_size, treatments)
        trtm_states = trtm_data[DataFields.STATE_DATA]
        
        # control states
        target_idx = np.arange(self.batch_size)
        if self.has_controls:
            ctrl_data = self.data.get_controls(self.batch_size)
            ctrl_states = ctrl_data[DataFields.STATE_DATA]

            # matching the two groups
            source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

        # moving states to torch tensors
        if self.has_controls:
            source = torch.from_numpy(ctrl_states[source_idx]).to(self.device).float()
        target = torch.from_numpy(trtm_states[target_idx]).to(self.device).float()

        # handling transformations
        if self.state_transforms is not None:
            if self.has_controls:
                source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)

        # constructing output dictionary
        out_dict = {DataFields.TARGET_STATE: target}
        if self.has_controls:
            out_dict[DataFields.SOURCE_STATE] = source 

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

class ValidationDataLoader(BaseDataLoader):
    """"""

    def __init__(
        self,
        data: AnnotatedPerturbationData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
        has_controls: bool = True,
        num_treatments_to_load: int | None = None
    ) -> None:
        """"""
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)
        self.has_controls = has_controls
        self.num_treatments_to_load = num_treatments_to_load 
    
    def __sample_perturbation_id(
        self,
    ) -> Sequence[str]:
        """
        Samples the treatment for the current batch when using Optimal Transport couplings.
        This is needed as the OT problem should be solved individually for each perturbation.
        
        :return: Integer containing the index for the perturbation used in the current batch of data.
        :rtype: int
        """
        # no perturbation data is passed to the AnnotatedPerturbationData object
        if self.data.seen_combinatorial_perturbations is None:
            return None
        # retrieving the maximum number of treatements to load if specified
        if self.num_treatment_to_load is not None:
            return random.choices(self.data.seen_combinatorial_perturbations, k=self.num_treatments_to_load)
        # returning all the treaments otherwise
        return self.data.seen_combinatorial_perturbations

    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """"""
        # retrieving the perturbations to validate on for the current batch
        perturbations = self.__sample_perturbation_id()

        # retrieving target data
        target_data = {
            perturbation: self.data.get_treatments(self.batch_size, perturbation) for perturbation in perturbations
        }

        # matching the groups
        matched_indices = {
            perturbation: {"target_idx": np.arange(self.batch_size)} for perturbation in target_data.keys()
        }
        if self.has_controls:
            ctrl_data = self.data.get_controls(self.batch_size)
            ctrl_states = ctrl_data[DataFields.STATE_DATA]

            # constructing dictionary to store matched data
            matched_indices = {}

            # matching each group
            for perturbation, perturbation_data in target_data.items():
                # retrieving the treatment states
                trtm_states = perturbation_data[DataFields.STATE_DATA]

                # matching the two groups
                source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

                # storing matched indices
                matched_indices[perturbation] = {
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                }
        
        # retrieving matched data
        matched_data = {}
        for perturbation, indices_dict in matched_indices.items():
            # retrieving data associated to current perturbation
            perturbation_data = target_data[perturbation]

            # retrieving states
            trtm_states = perturbation_data[DataFields.STATE_DATA]

            # retrieving target indices
            target_idx = indices_dict["target_idx"]
            target = torch.from_numpy(trtm_states[target_idx]).to(self.device).float()

            # retrieving optional control indices
            if self.had_controls:
                source_idx = indices_dict["source_idx"]
                source = torch.from_numpy(ctrl_states[source_idx]).to(self.device).float()

            # handling transformations
            if self.state_transforms is not None:
                if self.has_controls:
                    source = self.state_transforms.transform(source)
                target = self.state_transforms.transform(target)
            
            # constructing output dictionary
            out_dict = {DataFields.TARGET_STATE: target}
            if self.has_controls:
                out_dict[DataFields.SOURCE_STATE] = source 

            # handling perturbation data
            if self.data.perturbation_data is not None:
                trtm_perts = perturbation_data[DataFields.PERTURBATION_DATA]
                condition = {cond: torch.from_numpy(cond_data[target_idx]).to(self.device).float()
                            for cond, cond_data in trtm_perts.items()}
                out_dict[DataFields.PERTURBATION_DATA] = condition

            # handling target data
            if self.data.target_perturbation_repr is not None:
                trtm_perts_target_rep = trtm_data[DataFields.PERTURBATION_TARGET_REPR]
                trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]).to(self.device).float()
                                        for key, val in trtm_perts_target_rep.items()}
                out_dict[DataFields.PERTURBATION_TARGET_REPR] = trtm_perts_target_rep
            
            # storing output dictionary for current perturbation
            matched_data[perturbation] = out_dict
        return matched_data

     
class PredictionDataLoader(BaseDataLoader):
    """"""
