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

from labcompass.types import TensorLike

__all__ = [
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
]


def compute_r_squared(
    pred: torch.Tensor,
    target: torch.Tensor
) -> float:
    """Compute the coefficient of determination (R²) between the mean profiles of `pred` and `target`.

    Inputs are averaged over the leading (cell/sample) axis before scoring, so the score reflects
    how well the mean predicted profile agrees with the mean target profile across features.

    :param pred: Predicted values, of shape `(num_cells, num_features)`. Converted to a
        :class:`numpy.ndarray` if passed as a :class:`torch.Tensor`.
    :type pred: class:`torch.Tensor`

    :param target: Target/observed values, of shape `(num_cells, num_features)`. Converted to a
        :class:`numpy.ndarray` if passed as a :class:`torch.Tensor`.
    :type target: class:`torch.Tensor`

    :return: The R² score, as computed by :func:`sklearn.metrics.r2_score`, between
        `pred.mean(axis=0)` and `target.mean(axis=0)`.
    :rtype: class:`float`
    """
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
    """Compute the energy distance between `pred` and `target`, as in Peidli et al.

    The energy distance is computed as `2 * delta - sigma_pred - sigma_target`, where `delta` is the
    mean pairwise squared Euclidean distance between `pred` and `target`, and `sigma_pred`/`sigma_target`
    are the mean pairwise squared Euclidean distances within `pred` and within `target` respectively
    (computed via :func:`sklearn.metrics.pairwise_distances` with `metric="sqeuclidean"`).

    :param pred: Predicted samples, of shape `(num_pred_samples, num_features)`. Converted to a
        :class:`numpy.ndarray` if passed as a :class:`torch.Tensor`.
    :type pred: class:`TensorLike`

    :param target: Target/observed samples, of shape `(num_target_samples, num_features)`. Converted to
        a :class:`numpy.ndarray` if passed as a :class:`torch.Tensor`.
    :type target: class:`TensorLike`

    :return: The energy distance between the two empirical distributions. Lower values indicate
        that `pred` and `target` are closer.
    :rtype: class:`float`
    """
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
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
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
    """Compute the (RBF-kernel) Maximum Mean Discrepancy between `pred` and `target`, averaged across multiple kernel length scales.

    For each value in `gammas`, the squared MMD is computed with an RBF kernel of that bandwidth via
    :func:`maximum_mean_discrepancy`, and the final score is the `numpy.nanmean` of the resulting values
    (skipping any `NaN` entries).

    :param pred: Predicted samples, of shape `(num_pred_samples, num_features)`.
    :type pred: class:`TensorLike`

    :param target: Target/observed samples, of shape `(num_target_samples, num_features)`.
    :type target: class:`TensorLike`

    :param gammas: Sequence of RBF kernel bandwidths (`gamma` values) to average the MMD over,
        defaults to `[2, 1, 0.5, 0.1, 0.01, 0.005]` when `None`.
    :type gammas: class:`list[float] | None`

    :return: The multi-scale MMD between the two empirical distributions. Lower values indicate
        that `pred` and `target` are closer.
    :rtype: class:`float`
    """
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
    """Compute an optimal-transport-based distance between `pred` and `target` using uniform marginal weights.

    Both inputs are flattened along all dimensions but the first, and a pairwise cost matrix is built with
    `cost_fn` (defaults to the squared/`power`-powered Euclidean distance, `torch.cdist(pred, target) ** power`).
    The optimal transport problem between the two (uniform-weighted) empirical distributions is then solved
    with the Python Optimal Transport (POT) library, either exactly (`method="exact"`, via `pot.emd2`) or with
    entropic regularization (`method="sinkhorn"`, via `pot.sinkhorn2` with regularization strength `reg`).
    When the default `cost_fn` is used, the resulting OT cost is further raised to the power `1 / power` to
    recover the actual `power`-Wasserstein distance; when a custom `cost_fn` is supplied, the raw OT cost is
    returned unchanged.

    :param pred: Predicted samples, flattened to shape `(num_pred_samples, -1)`.
    :type pred: class:`TensorLike`

    :param target: Target/observed samples, flattened to shape `(num_target_samples, -1)`.
    :type target: class:`TensorLike`

    :param method: Which OT solver to use, either `"exact"` for the exact transport cost or `"sinkhorn"`
        for the entropy-regularized transport cost, defaults to `"exact"`.
    :type method: class:`Literal["exact", "sinkhorn"]`

    :param reg: Entropic regularization strength, only used when `method="sinkhorn"`, defaults to `5e-2`.
    :type reg: class:`float`

    :param power: The power used in the default cost function and in the final root normalization of the
        OT cost, defaults to `2`.
    :type power: class:`int`

    :param cost_fn: Function computing the pairwise cost matrix between `pred` and `target`, defaults to
        `None`, in which case `torch.cdist(pred, target) ** power` is used.
    :type cost_fn: class:`Callable[[TensorLike, TensorLike], TensorLike] | None`

    :param solver_kwargs: Keyword arguments passed to the underlying POT solver (`pot.emd2` or
        `pot.sinkhorn2`), defaults to `None`, in which case `{"numItermax": int(1e5)}` is used.
    :type solver_kwargs: class:`dict[str, Any] | None`

    :return: The optimal transport cost (or `power`-Wasserstein distance, when using the default cost
        function) between the two empirical distributions.
    :rtype: class:`float`

    :raises ValueError: If `method` is neither `"exact"` nor `"sinkhorn"`.
    """
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
    """Compute, for each entry along the leading axis of `pred`, the minimum and maximum per-cell MSE against `target`.

    For every index `i` along the first axis of `pred`, the pointwise (element-wise) mean squared error
    between `pred[i]` and `target` is computed and averaged over the feature axis, giving one MSE value
    per cell. `numpy.nanmin`/`numpy.nanmax` are then taken across cells, yielding, for each `i`, the
    smallest and largest per-cell MSE.

    :param pred: Tensor or array of shape `(n, num_cells, num_features)`, containing `n` sets of
        generated/predicted cells to compare against `target`.
    :type pred: class:`TensorLike`

    :param target: Tensor or array of shape `(num_cells, num_features)` containing the observed cells.
    :type target: class:`TensorLike`

    :return: A `(min_mse, max_mse)` tuple, where both elements are arrays of length `n` holding,
        for each entry along the leading axis of `pred`, the minimum and maximum per-cell MSE
        against `target` respectively.
    :rtype: class:`tuple[np.ndarray, np.ndarray]`
    """
    # moving to torch tensors in case inputs are arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)
    mses = np.array([torch.nn.functional.mse_loss(torch.from_numpy(pred[i, :, :]), target, reduction="none").mean(dim=1) for i in range(pred.shape[0])])
    return np.nanmin(mses, axis=1), np.nanmax(mses, axis=1)


def compute_cell_props(
    pred: TensorLike,
    target: TensorLike,
    k: int = 20,
    n_iter: int = 50
) -> TensorLike:
    """Estimate, over repeated random draws, the proportion of predicted cells among the nearest neighbours of a target cell.

    A single k-nearest-neighbour connectivity graph (:func:`sklearn.neighbors.kneighbors_graph`) is built
    on the concatenation of `target` and `pred` (`target` rows first, followed by `pred` rows). For
    `n_iter` iterations, one `target` cell is sampled at random and its row in the graph is inspected to
    count how many of its `k` nearest neighbours fall in the `pred` block versus the `target` block; the
    proportion `num_pred_neighbours / (num_pred_neighbours + num_target_neighbours)` is recorded.

    Note that the boundary between the `target` and `pred` blocks within the graph is hard-coded to
    column index `1024`, so this function implicitly assumes `target` has exactly `1024` rows.

    :param pred: Predicted cells, of shape `(num_pred_cells, num_features)`.
    :type pred: class:`TensorLike`

    :param target: Observed/target cells, of shape `(num_target_cells, num_features)`.
    :type target: class:`TensorLike`

    :param k: Number of nearest neighbours used to build the connectivity graph, defaults to `20`.
    :type k: class:`int`

    :param n_iter: Number of random target cells sampled to estimate the proportion, defaults to `50`.
    :type n_iter: class:`int`

    :return: Array of length `n_iter` with the fraction of predicted-cell neighbours found among the
        `k` nearest neighbours of each sampled target cell.
    :rtype: class:`TensorLike`
    """
    graph = kneighbors_graph(torch.vstack([target, pred]).numpy(), n_neighbors=k, mode='connectivity')
    props = []
    for _ in range(n_iter):
        target_sampled_idx = np.random.choice(np.arange(0, target.shape[0]), size=1, replace=False)
        pred_in_idx_neigh = np.where(graph[target_sampled_idx, 1024:].toarray().flatten() != 0)[0]
        target_in_idx_neigh = np.where(graph[target_sampled_idx, :1024].toarray().flatten() != 0)[0]
        props.append(len(pred_in_idx_neigh) / (len(target_in_idx_neigh) + len(pred_in_idx_neigh)))
    return np.array(props)