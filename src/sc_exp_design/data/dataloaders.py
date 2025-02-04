from typing import Literal

import torch

from sc_exp_design.constants import (
    PERTURBATION_DATA_KEY,
    PERTURBATION_TARGET_REPR_KEY,
    SOURCE_STATE_KEY,
    STATE_DATA_KEY,
    TARGET_STATE_KEY,
)
from sc_exp_design.couplings import Coupling
from sc_exp_design.data.data import PredictionData, TrainData
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike

__all__ = [
    "TrainDataLoader",
    "ValidationDataLoader",
    "PredictionDataLoader",
]


class TrainDataLoader:
    """"""

    def __init__(
        self,
        data: TrainData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)

    def sample(
        self,
    ) -> dict[str, TensorLike]:
        """"""

        ctrl_data = self.data.get_controls(self.batch_size)
        ctrl_states = ctrl_data[STATE_DATA_KEY]

        trtm_data = self.data.get_treatments(self.batch_size)
        trtm_states = trtm_data[STATE_DATA_KEY]

        if self.data.perturbation_data is not None:
            trtm_perts = trtm_data[PERTURBATION_DATA_KEY]
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = trtm_data[PERTURBATION_TARGET_REPR_KEY]

        source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

        source = torch.from_numpy(ctrl_states[source_idx])
        target = torch.from_numpy(trtm_states[target_idx])

        if self.data.perturbation_data is not None:
            condition = {cond: torch.from_numpy(cond_data[target_idx]) for cond, cond_data in trtm_perts.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]) for key, val in trtm_perts_target_rep.items()}

        source = source.to(self.device)
        target = target.to(self.device)

        if self.data.perturbation_data is not None:        
            condition = {key: val.to(self.device) for key, val in condition.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: val.to(self.device) for key, val in trtm_perts_target_rep.items()}

        source = source.float()
        target = target.float()

        if self.data.perturbation_data is not None:
            condition = {key: val.float() for key, val in condition.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: val.float() for key, val in trtm_perts_target_rep.items()}

        if self.state_transforms is not None:
            source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)

        out_dict = {SOURCE_STATE_KEY: source, TARGET_STATE_KEY: target}
        
        out_dict[PERTURBATION_DATA_KEY] = None
        if self.data.perturbation_data is not None:
            out_dict[PERTURBATION_DATA_KEY] = condition
        if self.data.target_perturbation_repr is not None:
            out_dict[PERTURBATION_TARGET_REPR_KEY] = trtm_perts_target_rep
        return out_dict


class ValidationDataLoader:
    """"""

    def __init__(
        self,
        data: PredictionData,
        coupling: Coupling,
        batch_size: int,
        state_transforms: Transform | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        self.data = data
        self.coupling = coupling
        self.batch_size = batch_size
        self.device_id = device_id
        self.state_transforms = state_transforms
        self.device = torch.device(self.device_id)

    def sample(
        self,
    ) -> dict[str, TensorLike]:
        """"""

        ctrl_data = self.data.get_controls(self.batch_size)
        ctrl_states = ctrl_data[STATE_DATA_KEY]

        trtm_data = self.data.get_treatments(self.batch_size)
        trtm_states = trtm_data[STATE_DATA_KEY]

        if self.data.perturbation_data is not None:
            trtm_perts = trtm_data[PERTURBATION_DATA_KEY]
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = trtm_data[PERTURBATION_TARGET_REPR_KEY]

        source_idx, target_idx = self.coupling.match_groups(ctrl_states, trtm_states)

        source = torch.from_numpy(ctrl_states[source_idx])
        target = torch.from_numpy(trtm_states[target_idx])

        if self.data.perturbation_data is not None:
            condition = {key: torch.from_numpy(val[target_idx]) for key, val in trtm_perts.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: torch.from_numpy(val[target_idx]) for key, val in trtm_perts_target_rep.items()}

        source = source.to(self.device)
        target = target.to(self.device)

        if self.data.perturbation_data is not None:
            condition = {key: val.to(self.device) for key, val in condition.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: val.to(self.device) for key, val in trtm_perts_target_rep.items()}

        source = source.float()
        target = target.float()

        if self.data.perturbation_data is not None:
            condition = {key: val.float() for key, val in condition.items()}
        if self.data.target_perturbation_repr is not None:
            trtm_perts_target_rep = {key: val.float() for key, val in trtm_perts_target_rep.items()}

        if self.state_transforms is not None:
            source = self.state_transforms.transform(source)
            target = self.state_transforms.transform(target)

        out_dict = {SOURCE_STATE_KEY: source, TARGET_STATE_KEY: target,}

        out_dict[PERTURBATION_DATA_KEY] = {}
        if self.data.perturbation_data is not None:
            out_dict[PERTURBATION_DATA_KEY] =  condition
        if self.data.target_perturbation_repr is not None:
            out_dict[PERTURBATION_TARGET_REPR_KEY] = trtm_perts_target_rep
        return out_dict


class PredictionDataLoader:
    """"""
