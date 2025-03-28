import numpy as np
from sklearn.metrics import r2_score
from torch import Tensor

__all__ = [
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
]


def compute_r_squared(pred: Tensor, target: Tensor):
    """"""
    # moving to numpy in case inputs are tensors
    if isinstance(pred, Tensor):
        pred = pred.numpy()
    if isinstance(target, Tensor):
        target = target.numpy()
    # computing r2 score
    return r2_score(np.mean(pred, axis=0), np.mean(target, axis=0))


def compute_sinkhorn_div(pred: Tensor, target: Tensor, reg: float = 1e-2) -> float:
    """"""
    ...


def compute_e_distance(pred: Tensor, target: Tensor) -> float:
    """"""
    ...
