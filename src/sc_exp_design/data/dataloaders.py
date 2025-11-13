import abc
import gc
from collections.abc import Sequence
from typing import Literal
import random

import numpy as np
import torch

from sc_exp_design.constants import DataFields
from sc_exp_design.couplings import Coupling, OTCoupling
from sc_exp_design.data.container import DataMixin
from sc_exp_design.data.data import AnnotatedPerturbationData
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike

__all__ = [
    "SequentialDataLoader",
    "BaseDataLoader",
    "TrainDataLoader",
    "ValidationDataLoader",
]


class BaseDataLoader(abc.ABC):
    """
    Abstract class for data loading objects.

    Childer classes need to define the :method: `sample` method in order to be instantiated/.
    """

    def _move_to_tensor_and_slice(
        self,
        data: np.ndarray,
        idxs: np.ndarray | None,
    ) -> torch.tensor:
        """"""
        if idxs is None:
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.from_numpy(data[idxs])
        return tensor.type(torch.float32).to(self.device)

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

        # treatment states
        trtm_data = self.data.get_treatments(self.batch_size, treatments)
        trtm_states = trtm_data.state_data

        # control states and coupling
        source_idx, target_idx = None, None
        if self.has_controls:
            # sanity check
            msg = f""
            assert control_states is not None, msg
            # matching the two groups
            source_idx, target_idx = self.coupling.match_groups(control_states, trtm_states)
            source = self._move_to_tensor_and_slice(control_states, source_idx)

        # move target to tensor and permute it
        target = self._move_to_tensor_and_slice(trtm_states, target_idx)

        # handling transformations
        if self.state_transforms is not None:
            if self.has_controls:
                source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)

        # constructing output dictionary
        out_dict = {DataFields.TARGET_STATE: target}
        del target
        if self.has_controls:
            out_dict[DataFields.SOURCE_STATE] = source 
            del source

        # handling perturbation data
        if self.data.perturbation_data is not None:
            trtm_perts = DataMixin(trtm_data.perturbation_data)
            condition = trtm_perts.apply(lambda e: self._move_to_tensor_and_slice(e, target_idx))
            out_dict[DataFields.PERTURBATION_DATA] = condition
            del trtm_perts, condition
            
        if self.data.target_data is not None:
            trtm_perts_target_rep = DataMixin(trtm_data.target_data)
            trtm_perts_target_rep = trtm_perts_target_rep.apply(lambda e: self._move_to_tensor_and_slice(e, target_idx))
            out_dict[DataFields.TARGET_DATA] = trtm_perts_target_rep
            del trtm_perts_target_rep

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
                perturbation_data[covariate] = self._move_to_tensor_and_slice(covariate_data, batch_idxs)
            out[DataFields.PERTURBATION_DATA] = perturbation_data
        
        # retrieving optional target covariates
        if self.data.target_data is not None:
            target_data = {}
            for covariate, covariate_data in self.data.target_data.items():
                target_data[covariate] = self._move_to_tensor_and_slice(covariate_data, batch_idxs)
            out[DataFields.TARGET_CATEGORIES] = target_data
        return out


class SequentialValDataLoader(BaseDataLoader):

    def __init__(
        self,
        data: AnnotatedPerturbationData,
        batch_size: int | None = None,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda"
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.state_transforms = state_transforms
        self.device_id = device_id
        self.device = torch.device(self.device_id)

        self.samples = self._pre_sample()

    def _pre_sample(
        self,
    ):
        # handling batch size
        n_obs = len(self.data)
        if self.batch_size is not None:
            batch_idxs = np.arange(n_obs)
        else:
            batch_idxs = np.random.choice(n_obs, size=self.batch_size)

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
                perturbation_data[covariate] = self._move_to_tensor_and_slice(covariate_data, batch_idxs)
            out[DataFields.PERTURBATION_DATA] = perturbation_data
        
        # retrieving optional target covariates
        if self.data.target_data is not None:
            target_data = {}
            for covariate, covariate_data in self.data.target_data.items():
                target_data[covariate] = self._move_to_tensor_and_slice(covariate_data, batch_idxs)
            out[DataFields.TARGET_CATEGORIES] = target_data
        return out

    def sample(
        self,
    ):
        return self.samples


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
        if not self.data.allow_grouped_couplings:
            return None
        # need to sample one perturbation from the set of unique perturbations
        return random.choice(self.data.seen_combinations)

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
            control_states = control_data.state_data

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
        data: dict[str, AnnotatedPerturbationData],
        coupling: Coupling,
        batch_size: int | None = None,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
        has_controls: bool = True,
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

        self.samples = self._pre_sample()

    def _pre_sample_data(
        self,
        data: AnnotatedPerturbationData,
    ):
        
        # retrieving the perturbations to validate on for the current batch
        treatments = self.__sample_perturbation_id(data)

        # defining dictionary of results
        data_dict = {}

        # iterating over the perturbations
        for treatment in treatments:
            # defining dictionary for current treatment
            trtm_dict = {}

            # retrieving target data
            trtm_data = data.get_treatments(treatments=treatment)
            if self.batch_size is not None:
                print("retrieving indices")
                trtm_idxs = len(trtm_data)
                trtm_idxs = np.random.choice(trtm_idxs, size=self.batch_size)
            else:
                trtm_idxs = None
            trtm_states = trtm_data.state_data
            trtm_states = self._move_to_tensor_and_slice(trtm_states, trtm_idxs)
            trtm_dict[DataFields.TARGET_STATE] = trtm_states

            # retrieving perturbation data
            if data.perturbation_data is not None:
                trtm_perts = DataMixin(trtm_data.perturbation_data)
                # using same indices as before
                condition = trtm_perts.apply(lambda e: self._move_to_tensor_and_slice(e, trtm_idxs))
                trtm_dict[DataFields.PERTURBATION_DATA] = condition

            # sampling control cells
            if self.has_controls:
                control_data = data.get_treatments(treatments=treatment)
                if self.batch_size is not None:
                    ctrl_idxs = len(control_data)
                    ctrl_idxs = np.random.choice(ctrl_idxs, size=self.batch_size)
                else:
                    ctrl_idxs = None
                control_states = control_data.state_data
                control_states = self._move_to_tensor_and_slice(control_states, ctrl_idxs)
                trtm_dict[DataFields.SOURCE_STATE] = control_states
            data_dict[treatment] = trtm_dict
        return data_dict

    def _pre_sample(
        self,
    ):
        return {
            f"{name}_{self._parse_perturbation_id(data, pert)}": sample_dict 
                for name, data in self.data.items()
                    for pert, sample_dict in self._pre_sample_data(data).items()
        }
            

    def __sample_perturbation_id(
        self,
        data,
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
        if not data.allow_grouped_couplings:
            return (None, )
        # returning all the treaments otherwise
        return data.seen_combinations

    def _parse_perturbation_id(
        self,
        data,
        treatment: Sequence[str] | None,
    ) -> str:
        """"""
        if treatment is None:
            # unconditional generation
            if data.perturbations is None:
                return "unconditional"
            else:
                # concatenate perturbation names
                if not data.allow_grouped_couplings:            
                    # concatenate perturbation names
                    treatment = [perturbation for perturbation in data.perturbations]
                    return "_".join(treatment)

                else:
                    msg = f"When `self.data.seen_combinations` is provided `treatment` should not be None."
                    raise ValueError(msg)
        else:
            treatment = [str(e) for e in treatment]
            msg = f"{treatment}"
            assert isinstance(treatment, Sequence), msg
            return "_".join(treatment)

    def sample(
        self,
    ) -> dict[str, TensorLike | dict[str, TensorLike]]:
        """
        Samples a batch of matched control and perturbed cell states for each perturbation to be loaded in the current iteration.

        :return: Dictionary mapping each perturbation to be loaded to a dictionary containing source (control) states, target (perturbed) states,
                and optional perturbation representations.
        :rtype: dict[str, TensorLike]
        """
        return self.samples
