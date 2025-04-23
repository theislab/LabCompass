from collections.abc import Callable
from functools import partial
import math
from typing import Any, Literal

import numpy as np
import ot as pot
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
import torch

from sc_exp_design.types import TensorLike

__all__ = [
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
]


def compute_r_squared(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """"""
    # moving to numpy in case inputs are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    # computing r2 score
    return r2_score(np.mean(pred, axis=0), np.mean(target, axis=0))


def compute_e_distance(
    pred: TensorLike,
    target: TensorLike
) -> float:
    """Compute the energy distance as in Peidli et al."""
    # moving to numpy in case inputs are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    # computing energy distance
    sigma_pred = pairwise_distances(pred, pred, metric="sqeuclidean").mean()
    sigma_target = pairwise_distances(target, target, metric="sqeuclidean").mean()
    delta = pairwise_distances(pred, target, metric="sqeuclidean").mean()
    return 2 * delta - sigma_pred - sigma_target


def maximum_mean_discrepancy(
    pred: TensorLike,
    target: TensorLike,
    gamma: float = 1.0
) -> float:
    """Compute the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Args:
        x: a tensor of shape [num_samples, num_features]
        y: a tensor of shape [num_samples, num_features]
        exact: a bool

    Returns
    -------
        a scalar denoting the squared maximum mean discrepancy loss.
    """
    # moving to numpy in case inputs are tensors
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    # computing mmd
    xx = rbf_kernel(pred, pred, gamma)
    xy = rbf_kernel(pred, target, gamma)
    yy = rbf_kernel(target, target, gamma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_mmd(
    pred: TensorLike,
    target: TensorLike,
    gammas: float | None = None
) -> float:
    """Compute MMD across different length scales"""
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]
    mmds = [maximum_mean_discrepancy(pred, target, gamma=gamma) for gamma in gammas]  # type: ignore[union-attr]
    return np.nanmean(np.array(mmds))


def compute_wasserstein_distance(
    pred: TensorLike,
    target: TensorLike,
    method: Literal["exact", "sinkhorn"] = "exact",
    reg: float = 5e-2,
    power: int = 2,
    cost_fn: Callable[[TensorLike, TensorLike], TensorLike] | None = None,
    solver_kwargs: dict[str, Any] | None = None,
) -> float:
    """"""
    # handling optional solver kwargs
    if solver_kwargs is None:
        solver_kwargs = {
            "numItermax": int(1e5)
        }

    # retrieving target ot function
    if method == "exact":
        ot_fn = partial(pot.emd2, **solver_kwargs)
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg, **solver_kwargs)
    else:
        msg = f""
        raise ValueError(msg)
    # defaults to euclidean distance
    if cost_fn is None:
        cost_fn = lambda pred, target: torch.cdist(pred, target)**power
    
    # computing weights
    pred_weights = pot.unif(pred.shape[0])
    target_weights = pot.unif(target.shape[0])

    # moving to torch tensors in case inputs are arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

    # flattening tensors
    pred = torch.flatten(pred, start_dim=1)
    target = torch.flatten(target, start_dim=1)

    # computing cost matrix
    distance_matrix = cost_fn(pred, target)

    # solving the ot problem and returning cost
    ot_cost = ot_fn(
        pred_weights,
        target_weights,
        distance_matrix.detach().cpu().numpy()
    )

    # normalizing distance
    if cost_fn is None:
        ot_cost = math.pow(ot_cost, 1/power)
    return ot_cost


def compute_min_max_mse(
    pred: TensorLike,
    target: TensorLike
) -> float:
    """Compute min and max pointwise MSE between generated and observed cells"""
    mses = np.array([torch.nn.functional.mse_loss(torch.from_numpy(pred[i, :, :]), target, reduction="none").mean(dim=1) for i in range(pred.shape[0])])
    return np.nanmin(mses, axis=1), np.nanmax(mses, axis=1)


def compute_cell_props(
    pred: TensorLike,
    target: TensorLike,
    k: int = 20,
    n_iter: int = 50
) -> TensorLike:
    """Compute proportion of generated cells in knn neighbourhood of n observed cells"""
    graph = kneighbors_graph(torch.vstack([target, pred]).numpy(), n_neighbors=k, mode='connectivity')
    props = []
    for _ in range(n_iter):
        target_sampled_idx = np.random.choice(np.arange(0, target.shape[0]), size=1, replace=False)
        pred_in_idx_neigh = np.where(graph[target_sampled_idx, 1024:].toarray().flatten() != 0)[0]
        target_in_idx_neigh = np.where(graph[target_sampled_idx, :1024].toarray().flatten() != 0)[0]
        props.append(len(pred_in_idx_neigh) / (len(target_in_idx_neigh) + len(pred_in_idx_neigh)))
    return np.array(props)