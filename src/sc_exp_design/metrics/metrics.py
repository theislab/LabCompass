from collections.abc import Callable
from functools import partial
import math
from typing import Any, Literal

import numpy as np
import ot as pot
from sklearn.metrics import pairwise_distances, r2_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sklearn.utils import check_random_state

import igraph
from igraph import Graph
from collections.abc import Generator
import torch

from sc_exp_design.types import TensorLike

__all__ = [
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
    "compute_weighted_min_mse",
    "compute_weighted_max_mse",
    "compute_cell_props",
    "compute_marginal_e_distance",
    "compute_n_gen_knn_cell_props",
    "compute_coclustering"
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
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    e_dists = []
    for i in range(pred.shape[0]):
    # computing energy distance
        sigma_pred = pairwise_distances(pred[i, :, :], pred[i, :, :], metric="sqeuclidean").mean()
        sigma_target = pairwise_distances(target[i, :, :], target[i, :, :], metric="sqeuclidean").mean()
        delta = pairwise_distances(pred[i, :, :], target[i, :, :], metric="sqeuclidean").mean()
        e_dists.append(2 * delta - sigma_pred - sigma_target)
    return np.nanmean(e_dists)


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
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    mmds = []
    for i in range(pred.shape[0]):
        xx = rbf_kernel(pred[i, :, :], pred[i, :, :], gamma)
        xy = rbf_kernel(pred[i, :, :], target[i, :, :], gamma)
        yy = rbf_kernel(target[i, :, :], target[i, :, :], gamma)
        mmds.append(xx.mean() + yy.mean() - 2 * xy.mean())
    return np.nanmean(mmds)


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

    # moving to torch tensors in case inputs are arrays
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target)

        
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    ot_costs = []
    for i in range(pred.shape[0]):
        # computing weights
        pred_weights = pot.unif(pred.shape[1])
        target_weights = pot.unif(target.shape[1])
    
        # flattening tensors
        pred_temp = torch.flatten(pred[i, :, :], start_dim=1)
        target_temp = torch.flatten(target[i, :, :], start_dim=1)
    
        # computing cost matrix
        distance_matrix = cost_fn(pred_temp, target_temp)
    
        # solving the ot problem and returning cost
        ot_cost = ot_fn(
            pred_weights,
            target_weights,
            distance_matrix.detach().cpu().numpy()
        )
    
        # normalizing distance
        if cost_fn is None:
            ot_cost = math.pow(ot_cost, 1/power)
        ot_costs.append(ot_cost)
    return np.nanmean(ot_costs)


def compute_weighted_min_mse(
    pred: TensorLike,
    target: TensorLike,
    weights: TensorLike = None
) -> float:
    """Compute (weighted) min pointwise MSE between generated and observed cells"""
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    mses = np.array([np.average(torch.nn.functional.mse_loss(torch.from_numpy(pred[i, :, :]), torch.from_numpy(target[i, :, :]), reduction="none"), axis=1, weights=weights) for i in range(pred.shape[0])])
    return np.mean(np.nanmin(mses, axis=1))


def compute_weighted_max_mse(
    pred: TensorLike,
    target: TensorLike,
    weights: TensorLike = None
) -> float:
    """Compute (weighted) max pointwise MSE between generated and observed cells"""
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    mses = np.array([np.average(torch.nn.functional.mse_loss(torch.from_numpy(pred[i, :, :]), torch.from_numpy(target[i, :, :]), reduction="none"), axis=1, weights=weights) for i in range(pred.shape[0])])
    return np.mean(np.nanmax(mses, axis=1))


def compute_cell_props(
    pred: TensorLike,
    target: TensorLike,
    k: int = 20,
    n_iter: int = 50,
    graph: TensorLike = None
) -> TensorLike:
    """Compute proportion of generated cells in knn neighbourhood of n observed cells"""
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if len(pred.shape) == 3:
        rand_batch_sample = np.random.choice(np.arange(0, pred.shape[0]), size=1, replace=False)
        pred = pred[rand_batch_sample, :, :].squeeze()
    if len(target.shape) == 3:
        target = target[rand_batch_sample, :, :].squeeze()
    else:
	cell_ids = np.arange(0, pred.shape[0])
        
    if graph is None:
        graph = kneighbors_graph(np.vstack([target, pred]), n_neighbors=k, mode='connectivity')
    props = []
    for _ in range(n_iter):
        target_sampled_idx = np.random.choice(np.arange(0, target.shape[0]), size=1, replace=False)
        pred_in_idx_neigh = np.where(graph[target_sampled_idx, target.shape[0]:].toarray().flatten() != 0)[0]
        target_in_idx_neigh = np.where(graph[target_sampled_idx, :target.shape[0]].toarray().flatten() != 0)[0]
        props.append(len(pred_in_idx_neigh) / (len(target_in_idx_neigh) + len(pred_in_idx_neigh)))
    return np.mean(props)


def squared_energy_distance_1d(x, y):
    """Compute energy distance between 1D arrays using squared Euclidean distance."""
    x = np.asarray(x).reshape(-1, 1)
    y = np.asarray(y).reshape(-1, 1)

    # Pairwise squared distances
    xy_dist = np.sum((x - y.T)**2) / (len(x) * len(y))
    xx_dist = np.sum((x - x.T)**2) / (len(x)**2)
    yy_dist = np.sum((y - y.T)**2) / (len(y)**2)

    return 2 * xy_dist - xx_dist - yy_dist



def compute_marginal_e_distance(
    pred: TensorLike, 
    target: TensorLike,
    weights: TensorLike = None,
) -> float:
    """Compute average per-feature energy distance between two datasets pred and target."""
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
    if len(pred.shape) == 3:
        rand_batch_sample = np.random.choice(np.arange(0, pred.shape[0]), size=1, replace=False)
        pred = pred[rand_batch_sample, :, :].squeeze()
    if len(target.shape) == 3:
        target = target[rand_batch_sample, :, :].squeeze()
        
    assert pred.shape[1] == target.shape[1], "Feature dimensions must match"
    
    distances = [
        squared_energy_distance_1d(pred[:, i], target[:, i])
        for i in range(pred.shape[1])
    ]
    return np.average(distances, weights=weights) # return average per-feature distances


def compute_n_gen_knn_cell_props(
    pred: TensorLike,
    target: TensorLike,
    graph = None,
) -> float:
    """Compute proportion of n generated cells in knn neighbourhood of each observed cell"""
    if len(pred.shape) == 3:
        k = pred.shape[0]
    elif graph is not None:
        k = graph.shape[0]
    else:
        k=50
    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
        
    if len(pred.shape) == 3:
        cell_ids = np.tile(np.arange(0, pred.shape[1]), pred.shape[0])
        pred = np.vstack(pred)
        
    if len(target.shape) == 3:
        rand_batch_sample = np.random.choice(np.arange(0, target.shape[0]), size=1, replace=False)
        target = target[rand_batch_sample, :, :].squeeze()
    

    cell_ids = np.concatenate([cell_ids, np.arange(0, target.shape[0])])
    batch = np.concatenate([np.repeat("gen", pred.shape[0]), np.repeat("target", target.shape[0])])
    
    if graph is None:
        graph = kneighbors_graph(np.vstack([target, pred]), n_neighbors=k, mode='connectivity')
    props = []
    for cell_idx in set(cell_ids):
        target_idx = np.where((cell_ids == cell_idx) & (batch == "target"))[0]
        pred_idx = np.where((cell_ids == cell_idx) & (batch != "target"))[0]
        pred_in_neigh = (graph[target_idx, pred_idx].A != 0).sum() / k
        props.append(pred_in_neigh)
    return np.mean(props)


def get_igraph_from_adjacency(
    adjacency: TensorLike, 
    *, 
    directed: bool = False
) -> Graph:
    """Get igraph graph from adjacency matrix."""
    
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    g = Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])
    g.add_edges(list(zip(sources, targets, strict=True)))
    g.es["weight"] = weights
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f"The constructed graph has only {g.vcount()} nodes. "
            "Your adjacency matrix contained redundant nodes."
        )
    return g


def compute_clustering(
    adjacency: TensorLike, 
    k: int, 
    random_state: int = 0,
    **clustering_args
) -> TensorLike:
    """Computes Leiden clustering give adjacency matrix"""

    graph = kneighbors_graph(adjacency, n_neighbors=k, mode='connectivity', )
    g = get_igraph_from_adjacency(graph, directed=False)
    part = g.community_leiden(**clustering_args)
    groups = np.array(part.membership)
    return groups



def compute_coclustering(
    pred: TensorLike, 
    target: TensorLike, 
    n_iterations: float = -1, 
    resolution: float = 1., 
    k: int = 15, 
    **clustering_args
) -> float:
    """Computes proportion of generated and groudtruth cells that co-cluster"""
    
    clustering_args = dict(clustering_args)
    clustering_args["n_iterations"] = n_iterations
    if resolution is not None:
        clustering_args["resolution"] = resolution
    clustering_args.setdefault("objective_function", "modularity")

    if isinstance(pred, torch.Tensor):
        pred = pred.numpy()
    if isinstance(target, torch.Tensor):
        target = target.numpy()
        
    # handdling shapes
    if len(pred.shape) < 3:
        pred = pred.reshape((1, pred.shape[0], pred.shape[1]))
    if len(target.shape) < 3:
        target = target.reshape((1, target.shape[0], target.shape[1]))
    coclust_props = []
    
    for i in range(pred.shape[0]):
        combined = np.concatenate([target[i, :, :], pred[i, :, :]])
        clustering = compute_clustering(combined, k, **clustering_args)
        coclust_props.append(np.sum(clustering[:target.shape[1]] == clustering[target.shape[1]:]) / target.shape[1])
        
    return np.nanmean(coclust_props)
