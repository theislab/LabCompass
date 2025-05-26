from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["DataContainer"]


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
    state_data: np.ndarray | int | float
    perturbation_data: dict[str, np.ndarray | int | float] | None
    target_data: dict[str, np.ndarray | int | float] | None

    def __post_init__(
        self,
    ) -> None:
        """Checks that all the data shares the same batch size"""

        # check type state data
        msg = f"State data of the wrong type. Expected `np.ndarray | int | float`, found {type(self.state_data)}."
        assert isinstance(self.state_data, np.ndarray | int | float | np.long), msg

        # retrieving reference number of observations                
        if isinstance(self.state_data, np.ndarray) and self.state_data.ndim != 1:
            self.num_observations = self.state_data.shape[0]
        else:
            self.num_observations = 1

        # check type perturbation data
        if self.perturbation_data is not None:
            msg = f"Perturbation data of the wrong type. Expected `dict`, found {type(self.perturbation_data)}."
            assert isinstance(self.perturbation_data, dict), msg

        # check type target data
        if self.target_data is not None:
            msg = f"Target data of the wrong type. Expected `dict`, found {type(self.target_data)}."
            assert isinstance(self.target_data, dict), msg

        # ensuring the data is stored in an array
        self.state_data = self._ensure_array(self.state_data)

        # perturbation data
        if self.perturbation_data is not None:
            for perturbation_covariate, covariate_data in self.perturbation_data.items():
                # check type
                msg = f"Data for perturbation covariate {perturbation_covariate} of the wrong type. Expected `np.ndarray | int | float`, found {type(covariate_data)}."
                assert isinstance(covariate_data, np.ndarray | int | float | np.long), msg

                # ensuring type
                covariate_data = self._ensure_array(covariate_data)

                # check shape
                msg = f"Wrong batch dimension for perturbation covariate {perturbation_covariate}. Expected {self.num_observations}, found {covariate_data.shape[0]}. State data shape {self.state_data.shape}. Covariate data shape {covariate_data.shape}"
                assert covariate_data.shape[0] == self.num_observations, msg

                # storing results
                self.perturbation_data[perturbation_covariate] = covariate_data

        # target data
        if self.target_data is not None:
            for target_covariate, covariate_data in self.target_data.items():
                # check type
                msg = f"Data for target covariate {target_covariate} of the wrong type. Expected `np.ndarray | int | float`, found {type(covariate_data)}."
                assert isinstance(covariate_data, np.ndarray | int | float | np.long), msg

                # ensuring type
                covariate_data = self._ensure_array(covariate_data)

                # check shape
                msg = f"Wrong batch dimension for target covariate {target_covariate}. Expected {self.num_observations}, found {covariate_data.shape[0]}. State data shape {self.state_data.shape}. Covariate data shape {covariate_data.shape}"
                assert covariate_data.shape[0] == self.num_observations, msg

                # storing results
                self.target_data[target_covariate] = covariate_data

    def _ensure_array(
        self,
        val: np.ndarray | int | float | np.generic,
    ) -> np.ndarray:
        """"""

        # handling case for numpy data types
        if isinstance(val, np.generic):
            val = val.item()

        # moving target data to numpy array in case its a scalar
        if isinstance(val, int | float):
            val = np.array([val])
        
        # when is an array we return it
        if isinstance(val, np.ndarray):
            # when we lose the batch dimension we need to unsqueeze
            if val.ndim == 1:
                if val.shape[0] != self.num_observations and self.num_observations == 1:
                    val = val.reshape(1, -1)
            return val

        # otherwise we raise type error
        msg = f"Unsupported data type {val}"
        raise TypeError(msg)

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
            perturbation_data = {k: function(v) for k, v in self.perturbation_data.items()}

        # applying function on target data
        target_data = None
        if self.target_data is not None:
            target_data = {k: function(v) for k, v in self.target_data.items()}

        return DataContainer(
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
            perturbation_data = {k: v[idx] for k, v in self.perturbation_data.items()}

        # target data
        target_data = None
        if self.target_data is not None:
            target_data = {k: v[idx] for k, v in self.target_data.items()}
        
        return DataContainer(
            state_data,
            perturbation_data,
            target_data,
        )

