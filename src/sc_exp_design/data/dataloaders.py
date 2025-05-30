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
    """
    Abstract class for data loading objects.

    Childer classes need to define the :method: `sample` method in order to be instantiated/.
    """

    @abc.abstractmethod
    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """"""
        raise NotImplementedError


class BaseCoupledDataLoader(BaseDataLoader):
    """
    Base class for coupled datasets. Derived from :class: `BaseDataLoader`.

    Defines the :method: `_get_matched_data` method, needed to construct the coupling
    between source and target distributions when control states are passed.
    """

    def _get_matched_data(
        self,
        treatments: Sequence[str] | None,
        control_states: TensorLike | None,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """
        Matches the control and to samples from the distributions perturbed with the treatments
        passed in :param: `treatements`.

        :param treatments: Sequence of string identifiers of the treatment to be loaded in the current batch.
            When `None`, all treatments will be retrieved. This argument is used to call the :method:`AnnotatedPerturbationData.get_treatments` method.
        :type treatments: class: `Sequence[str] | None`

        :param control_states: Tensor or array holding the state data for control observations. Calling this with :param: `control_states` as `None` will
            raise an :error: `AssertionError` when the :attr: `AnnotatedPerturbationData.has_controls` attribute is set to `True`.
        :type control_states: class: `TensorLike | None`

        :return: Returns the data of the batch in a dictionary.
        :rtype: class: `dict[str, TensorLike | dict[str, TensorLike]]`
        """
        # sanity check
        if self.has_controls:
            msg = f""
            assert control_states is not None, msg

        # treatment states
        trtm_data = self.data.get_treatments(self.batch_size, treatments)
        trtm_states = trtm_data[DataFields.STATE_DATA]

        # control states
        target_idx = np.arange(self.batch_size)
        if self.has_controls:

            # matching the two groups
            source_idx, target_idx = self.coupling.match_groups(control_states, trtm_states)

        # moving states to torch tensors
        if self.has_controls:
            source = torch.from_numpy(control_states[source_idx]).to(self.device).float()
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
            
        if self.data.target_reprs is not None:
            trtm_perts_target_rep = trtm_data[DataFields.TARGET_DATA]
            trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]).to(self.device).float()
                                     for key, val in trtm_perts_target_rep.items()}
            out_dict[DataFields.TARGET_DATA] = trtm_perts_target_rep
        
        return out_dict


class SequentialDataLoader(BaseDataLoader):
    """
    Class defining unpaired sequential data loading
    """
    def __init__(
        self,
        data: AnnotatedPerturbationData,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda"
    ) -> None:
        """
        Initialize the :class: `SequentialDataLoader` object.

        :param data: Annotated data to be loaded.
        :type data: class: `AnnotatedPerturbationData`

        :param batch_size: The number of observations to be loaded in each batch.
        :type batch_size: class: `int`

        :param state_transforms: Optional transformations to be applied to the states when loading the batch, defaults to `None`.
        :type state_transforms: class: `Transform | None`

        :param device_id: String identifier indicating the device to load the data on, defaults to `"cuda"`.
        :type device_id: class: `Literal["cpu", "cuda"]`
        """
        self.data = data
        self.batch_size = batch_size
        self.state_transforms = state_transforms
        self.device_id = device_id
        self.device = torch.device(self.device_id)
    
    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """
        Samples a batch of data.

        :return: The data of the batch in a dictionary
        :rtype: class: `dict[str, TensorLike | dict[str, TensorLike]]`
        """
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
        if self.data.target_reprs is not None:
            target_data = {}
            for covariate, covariate_data in self.data.target_reprs.items():
                target_data[covariate] = torch.from_numpy(covariate_data[batch_idxs]).to(self.device).float()
            out[DataFields.TARGET_CATEGORIES] = target_data
        return out


class TrainDataLoader(BaseCoupledDataLoader):
    """
    Data loader for training that samples matched control and perturbed cell states.

    Children class of :class: `BaseCoupledDataLoader`, from which it inherits the :method: `BaseClassDataLoader._match_groups` method.
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

        :param data: Annotated data to be loaded.
        :type data: class: `AnnotatedPerturbationData`

        :param coupling: Coupling strategy used to match control and perturbed states, should be a class providing the :method:`coupling.match_groups`.
            Should be an instance of a class derived from :class: `Coupling`.
        :type coupling: class:`Coupling`

        :param batch_size: The number of observations to be loaded in each batch.
        :type batch_size: class: `int`

        :param state_transforms: Optional transformations to be applied to the states when loading the batch, defaults to `None`.
        :type state_transforms: class: `Transform | None`

        :param device_id: String identifier indicating the device to load the data on, defaults to `"cuda"`.
        :type device_id: class: `Literal["cpu", "cuda"]`

        :param has_controls: Flag indicating whether source states are present in :param: `data`, defaults to `True`.
        :type has_controls: class: `bool`
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
    ) -> Sequence[str] | None:
        """
        Samples the treatment for the current batch when using Optimal Transport couplings.
        This is needed as the OT problem should be solved individually for each perturbation.
        
        :return: String identifier for the perturbation used in the current batch of data.
            When :attr: `self.data.seen_combinatorial_perturbations` is `None`, it returns `None`.
        :rtype: class: `Sequence[str] | None`
        """
        # no perturbation data is passed to the AnnotatedPerturbationData object
        if (self.data.seen_combinatorial_perturbations is None) or (len(self.data.perturbations_in_obsm) == 0):
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

        # control states
        control_states = None
        if self.has_controls:
            control_data = self.data.get_controls(self.batch_size)
            control_states = control_data[DataFields.STATE_DATA]

        # retrieving matched data
        return self._get_matched_data(treatments, control_states)


class ValidationDataLoader(BaseCoupledDataLoader):
    """
    Data loader for validation that samples matched control and perturbed cell states.

    Children class of :class: `BaseCoupledDataLoader`, from which it inherits the :method: `BaseClassDataLoader._match_groups` method.
    It groups batches of data by each individual condition that is being loaded, effectively returning as many batches as loaded condition.
    """

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
        """
        Initializes the training data loader.

        :param data: Annotated data to be loaded.
        :type data: class: `AnnotatedPerturbationData`

        :param coupling: Coupling strategy used to match control and perturbed states, should be a class providing the :method:`coupling.match_groups`.
            Should be an instance of a class derived from :class: `Coupling`.
        :type coupling: class:`Coupling`

        :param batch_size: The number of observations to be loaded in each batch.
        :type batch_size: class: `int`

        :param state_transforms: Optional transformations to be applied to the states when loading the batch, defaults to `None`.
        :type state_transforms: class: `Transform | None`

        :param device_id: String identifier indicating the device to load the data on, defaults to `"cuda"`.
        :type device_id: class: `Literal["cpu", "cuda"]`

        :param has_controls: Flag indicating whether source states are present in :param: `data`, defaults to `True`.
        :type has_controls: class: `bool`

        :param num_treatments_to_load: Specifies the maximum number of unique treatments to be loaded in a single batch.
            Defaults to `None`, in which case all unique treatments are loaded.
        :type num_treatments_to_load: class: `int | None`
        """
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
    ) -> Sequence[str | None]:
        """
        Samples the treatment for the current batch when using Optimal Transport couplings.
        This is needed as the OT problem should be solved individually for each perturbation.
        
        :return: String identifier for the perturbations to be loaded in the current batch of data.
            When :attr: `self.data.seen_combinatorial_perturbations` is `None`, it returns `(None, )`.
            This is also the case when there are perturbations present in :attr: `self.data.perturbations_in_obsm`.
        :rtype: class: `Sequence[str | None]`
        """
        # no perturbation data is passed to the AnnotatedPerturbationData object
        if (self.data.seen_combinatorial_perturbations is None) or (len(self.data.perturbations_in_obsm) == 0):
            return (None, )
        # retrieving the maximum number of treatements to load if specified
        if self.num_treatments_to_load is not None:
            return random.choices(self.data.seen_combinatorial_perturbations, k=self.num_treatments_to_load)
        # returning all the treaments otherwise
        return self.data.seen_combinatorial_perturbations

    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """
        Samples a batch of matched control and perturbed cell states for each perturbation to be loaded in the current iteration.

        :return: Dictionary mapping each perturbation to be loaded to a dictionary containing source (control) states, target (perturbed) states,
                and optional perturbation representations.
        :rtype: dict[str, TensorLike]
        """
        # retrieving the perturbations to validate on for the current batch
        treatments = self.__sample_perturbation_id()

        # control states
        control_states = None
        if self.has_controls:
            control_data = self.data.get_controls(self.batch_size)
            control_states = control_data[DataFields.STATE_DATA]

        # constructing output dictionary
        out_dict = {}
        # iterating over the perturbations
        for treatment in treatments:

            # retrieving matched treatment data
            treatement_data = self._get_matched_data(treatment, control_states)

            # constructing perturbation identifier to store the results
            if treatment is None:
                if self.data.perturbations_in_obsm==None:
                    treatment_id = "unconditional"
                else:
                    # concatenate perturbation names
                    treatment = [perturbation for perturbation in self.data.perturbations_with_rep]
                    treatment_id = "_".join(treatment)
            else:
                msg = f""
                assert isinstance(treatment, Sequence), msg
                treatment_id = "_".join(str(treatment))

            # storing output dictionary for current perturbation
            out_dict[treatment_id] = treatement_data
        return out_dict


class PredictionDataLoader(BaseDataLoader):
    """"""
