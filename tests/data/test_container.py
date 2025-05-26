import numpy as np
import pytest

from sc_exp_design.data.data import DataContainer



class TestDataContainer:
    """
    """

    @pytest.mark.parametrize("load_perturbation_data", [False, True])
    @pytest.mark.parametrize("load_target_data", [False, True])
    def test_batch_data(
        self,
        states: np.ndarray,
        perturbations: dict[str, np.ndarray],
        targets: dict[str, np.ndarray],
        num_cells: int,
        batch_size: int,
        load_perturbation_data: bool,
        load_target_data: bool,
    ) -> None:
        """"""

        # setting perturbation data
        perturbation_data = None
        if load_perturbation_data:
            perturbation_data = perturbations

        # setting target data
        target_data = None
        if load_target_data:
            target_data = targets


        # initializing data
        data = DataContainer(
            states,
            perturbation_data,
            target_data,
        )

        # slicing with random indices
        batch_idxs = np.random.choice(num_cells, size=batch_size)
        batch_data = data[batch_idxs]

        # try methods (no axis)
        batch_data.min()
        batch_data.argmin()

        # try methods (axis 0)
        batch_data.min(axis=0)
        batch_data.argmin(axis=0)

        # try methods (axis 1)
        batch_data.min(axis=1)
        batch_data.argmin(axis=1)
