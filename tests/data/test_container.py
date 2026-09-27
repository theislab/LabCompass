import numpy as np
import pytest
import torch

from labcompass.data.container import BatchMixin, DataContainer


class TestDataContainer:
    """
    """

    def test_mapped_batch(
        self,
        num_genes: int,
        num_cells: int,
        num_control_cells: int,
        num_targets: int,
    ) -> None:
        """"""
        # creating data dict
        data_dict = {
            f"field{idx}": np.random.randn(num_cells, num_genes) for idx in range(num_targets)
        }

        # initializing mapped batch data with correct data
        BatchMixin(data_dict)

        # initializing with wrong type
        with pytest.raises(TypeError):
            data_dict.update({"wrong_element": torch.randn((num_cells, num_genes))})
            BatchMixin(data_dict)

        # initializing with wrong batch size
        with pytest.raises(ValueError):
            data_dict.update({"wrong_element": np.random.randn(num_cells - 1, num_genes)})
            BatchMixin(data_dict)

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
            perturbation_data = BatchMixin(perturbations)

        # setting target data
        target_data = None
        if load_target_data:
            target_data = BatchMixin(targets)

        # initializing data
        data = DataContainer(
            states,
            perturbation_data,
            target_data,
        )

        # slicing with random indices
        batch_idxs = np.random.choice(num_cells, size=batch_size)
        data[batch_idxs]
