from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import logging
from typing import Any, ClassVar, Type

import numpy as np

from sc_exp_design.types import TensorLike

__all__ = ["DataContainer"]


class DataMixin(dict):
    """"""
    _required_type: ClassVar[Type[Any] | None] = None

    def __init__(
        self,
        mapping: Mapping[str, Any] | None = None,
        **kwargs
    ) -> None:
        """"""
        # creating empty dictionary
        if mapping is None:
            mapping = {}
        
        # adding keyword arguments
        if kwargs is not None:
            mapping.update(kwargs)
    
        # verifying that the keys are strings
        for key in mapping.keys():
            if not isinstance(key, str):
                msg = f""
                raise ValueError(msg)

        # calling parent constructor
        super().__init__(mapping)

        # verifying inputs
        self._verify_inputs()

    def _verify_inputs(
        self,
    ) -> None:
        """"""
        # checking that we have the required type when specified
        if self._required_type is not None:
            if not issubclass(self.data_type, self._required_type):
                msg = f"Data is of the wrong type. Got {self.data_type}, expected {self._required_type}."
                raise TypeError(msg)

        # iterating over each key to check that the type is the same
        for key, value in self.items():
            if not isinstance(value, self.data_type):
                msg = f"The values should share the same type. Got {type(value)} for {key}, expected {self.data_type}."
                raise TypeError(msg)

    def __getattr__(
        self,
        name: str,
    ) -> Callable:
        """"""
        # Handle array methods
        def wrapper(*args, **kwargs):
            return self._apply_function(lambda x: getattr(x, name)(*args, **kwargs))

        # returning wrapper function
        if all(hasattr(t, name) for t in self.values()):
            return wrapper
        raise AttributeError(f"'DataMixin' object has no attribute '{name}'")

    def __getitem__(
        self,
        idx: int | slice
    ):
        """"""
        return self.__class__({key: value[idx] for key, value in self.items()})

    def _apply_function(
        self,
        function: Callable[[Any], Any],
        fields: None | Sequence[str] = None,
        *args,
        **kwargs,
    ):
        """"""
        # handling optional fields
        if fields is None:
            fields = self.keys()

        # applying the function
        out_dict = {}
        for key, value in self.items():
            if key in fields:
                value = function(value, *args, **kwargs)
            out_dict[key] = value
        return self.__class__(**out_dict)
    
    def apply(
        self,
        function: Callable[[Any], Any],
        fields: None | Sequence[str] = None,
        *args,
        **kwargs,
    ):
        """"""
        return self._apply_function(
            function,
            fields,
            *args,
            **kwargs
        )

    @property
    def data_type(
        self,
    ) -> Type[Any]:
        """"""
        if len(self) == 0:
            return self._required_type
        return type(next(iter(self.values())))


class ArrayMixin(DataMixin):
    """"""
    _required_type: ClassVar[Type[Any]] = np.ndarray | np.generic


class BatchMixin(ArrayMixin):
    """"""
    _minimum_dims: ClassVar[int] = 1

    def _verify_inputs(
        self,
    ) -> None:
        """"""
        # calling method of parent class for usual checks
        super()._verify_inputs()

        # iterating over the elements
        for key, value in self.items():
            # verifying that the required dimensions match
            self.__verify_shape(value)

    def __verify_shape(
        self,
        data: TensorLike,
    ) -> None:
        """"""
        # we need at least self._minimum_dims + 1 dimensions
        if data.ndim < self._minimum_dims:
            msg = f""
            raise ValueError(msg)

        # retrieving reference dims
        reference_dims = next(iter(self.values())).shape[:self._minimum_dims]

        # iterating over the number of required dimensions
        for dim, reference_dim in enumerate(reference_dims):
            # raise error if does not match
            if data.shape[dim] != reference_dims[dim]:
                msg = f""
                raise ValueError(msg)

    @property
    def batch_size(
        self,
    ) -> int:
        """"""
        return next(iter(self.values())).shape[0]


@dataclass
class DataContainer:
    """Data structure for batch data.
    
    Handles jointly the data modalities to access them via slicing.

    :param state_data: The state data for the current batch.
    :type state_data: class: `np.ndarray`

    :param perturbation_data: Optional perturbation data for the current batch.
    :type perturbation_data: class `dict[str, np.ndarray] | None`

    :param target_data: Optional target data for the current batch.
    :type target_data: class `dict[str, np.ndarray] | None`
    """
    state_data: np.ndarray
    perturbation_data: BatchMixin | None
    target_data: BatchMixin | None

    def __post_init__(
        self,
    ) -> None:
        """Checks that all the data shares the same batch size"""

        # check type state data
        if not isinstance(self.state_data, np.ndarray):
            msg = f"State data of the wrong type. Expected `np.ndarray`, found {type(self.state_data)}."
            raise TypeError(msg)

        # check perturbation data
        if self.perturbation_data is not None:
            # check type
            if not isinstance(self.perturbation_data, BatchMixin):
                msg = f"Perturbation data is of the wrong type. Got {type(self.perturbation_data)}, expected `BatchMixin`"
                raise ValueError(msg)

            # raise error if shapes don't match
            if self.perturbation_data.batch_size != self.state_data.shape[0]:
                msg = f"Wrong batch dimension for perturbation covariate {perturbation_covariate}. Expected {self.num_observations}, found {covariate_data.shape[0]}. State data shape {self.state_data.shape}. Covariate data shape {covariate_data.shape}"
                raise ValueError(msg)

        # check target data
        if self.target_data is not None:
            # check type
            if not isinstance(self.target_data, BatchMixin):
                msg = f""
                raise ValueError(msg)

            # raise error if shapes don't match
            if self.target_data.batch_size != self.state_data.shape[0]:
                msg = f"Wrong batch dimension for perturbation covariate {perturbation_covariate}. Expected {self.num_observations}, found {covariate_data.shape[0]}. State data shape {self.state_data.shape}. Covariate data shape {covariate_data.shape}"
                raise ValueError(msg)

    def _apply(
        self,
        function: Callable,
    ) -> "DataContainer":
        """Applies a function to the data.
        
        :param function: The function to be called on each array.
        :type function: class: `Callable`
        """
        # applying function on state data
        state_data = function(self.state_data)

        # applying function on peturbation data
        perturbation_data = None
        if self.perturbation_data is not None:
            perturbation_data = self.perturbation_data.apply(function)

        # applying function on target data
        target_data = None
        if self.target_data is not None:
            target_data = self.target_data.apply(function)

        return self.__class__(
            state_data,
            perturbation_data,
            target_data,
        )

    def __getattr__(
        self,
        name: str,
    ) -> Callable:
        """"""

        # Handle array methods
        def wrapper(*args, **kwargs):
            return self._apply(lambda x: getattr(x, name)(*args, **kwargs))
        
        # retrieving array data
        data = [self.state_data]
        if self.perturbation_data is not None:
            data.extend(self.perturbation_data.values())
        if self.target_data is not None:
            data.extend(self.target_data.values())
        
        # returning wrapper function
        if all(hasattr(t, name) for t in data):
            return wrapper
        raise AttributeError(f"'DataContainer' object has no attribute '{name}'")

    def __len__(
        self,
    ) -> int:
        """Returns the number of observation currently stored."""
        return self.num_observations

    def __getitem__(
        self,
        idx: int | slice
    ) -> "DataContainer":
        """Retrieves the element at the `idx`-th position.

        :param idx: The index of the element to retrieve.
        :type idx: class: `int | slice`
        """
        # state data
        state_data = self.state_data[idx]

        # perturbation data  
        perturbation_data = None   
        if self.perturbation_data is not None:
            perturbation_data = self.perturbation_data[idx]

        # target data
        target_data = None
        if self.target_data is not None:
            target_data = self.target_data[idx]

        return self.__class__(
            state_data,
            perturbation_data,
            target_data,
        )

    @property
    def num_observations(
        self,
    ) -> int:
        """"""
        return self.state_data.shape[0]

    @property
    def perturbation_covariates(
        self,
    ) -> Sequence[str] | None:
        """Returns the sequence of modeled perturbation covariates"""
        if self.perturbation_data is None:
            return None
        return list(self.perturbation_data.keys())

    @property
    def target_covariates(
        self,
    ) -> Sequence[str] | None:
        """Returns the sequence of modeled target covariates"""
        if self.target_data is None:
            return None
        return list(self.target_data.keys())
