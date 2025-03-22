import logging
import abc
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import numpy as np
import ot as pot
import torch

from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = ["Coupling", "FixedCoupling", "OTCoupling", "IndependentCoupling"]


class Coupling(abc.ABC):
    """Base class for objects implementing coupling strategies"""

    @abc.abstractmethod
    def match_groups(
        self,
        source: TensorLike,
        target: TensorLike,
        **kwargs,
    ) -> Any:
        """"""
        raise NotImplementedError


class FixedCoupling(Coupling):
    """Fixed coupling for already paired data.

    :param shuffle: Whether to shuffle the batch before returning the data, defaults to `False`
    :type shuffle: class:`bool`
    """

    def __init__(
        self,
        *args,
        shuffle: bool = False,
        **kwargs,
    ) -> None:
        self.shuffle = shuffle

    def match_groups(
        self,
        source: TensorLike,
        target: TensorLike,
    ) -> tuple[TensorLike, TensorLike]:
        """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

        :param source: A tensor or array of values containing the data coming from the source distribution.
        :type source: class:`TensorLike`

        :param target: A tensor or array of values containing the data coming from the target distribution.
        :type target: class:`TensorLike`
        """
        msg = f"The source and the target batches are expected to share the have the same batch size, found {source.shape[0]=} and {target.shape[0]=}"
        assert source.shape[0] == target.shape[0], msg
        if self.shuffle:
            random_perm_idx = torch.randperm(source.shape[0])
            return random_perm_idx, random_perm_idx
        else:
            idxs = torch.arange(source.shape[0])
            return idxs, idxs


class OTCoupling(Coupling):
    """Optimal transport (OT) coupling for unpaired data.

    :param method: The method used to solve the Optimal Transport problem.
    :type method: class:`Literal["exact", "sinkhorn", "unbalandced", "partial"]`

    :param cost_fn: Function used to compute the matrix for the displacement cost from :param:`source` to :param:`target`
    :type cost_fn: class:`Callable[[TensorLike, TensorLike], TensorLike]`

    :param reg: Regularization strength used in the `"sinkhorn"`, `"unbalanced"` and `"partial"` method.
    :type reg: class:`float | None`

    :param reg: Regularization strength used in the `"unbalanced"` method.
    :type reg: class:`float | None`
    """

    def __init__(
        self,
        method: Literal["exact", "sinkhorn", "unbalanced", "partial"] = "exact",
        solver_kwargs: dict[str, Any] | None = None,
        cost_fn: Callable[[TensorLike, TensorLike], TensorLike] | None = None,
        reg: float = 5e-1,
        reg_m: float = 1e-0,
        normalize_cost: bool = False,
        replace: bool = True,
    ) -> None:
        # empty dictionary if no solver kwargs provided
        if solver_kwargs is None:
            solver_kwargs = {}
        # sanity checks
        if method == "exact":
            ot_fn = pot.emd
        elif method == "sinkhorn":
            if reg is None:
                msg = f"{method=} requires `reg` to be a `float`, `None` found"
                raise ValueError(msg)
            ot_fn = partial(pot.sinkhorn, reg=reg, **solver_kwargs)
        elif method == "partial":
            if reg is None:
                msg = f"{method=} requires `reg` to be a `float`, `None` found"
                raise ValueError(msg)
            ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg, **solver_kwargs)
        elif method == "unbalanced":
            if reg is None:
                msg = f"{method=} requires `reg` to be a `float`, `None` found" 
                raise ValueError(msg)
            if reg_m is None:
                msg = f"{method=} requires `reg_m` to be a `float`, `None` found" 
                raise ValueError(msg)
            ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m, **solver_kwargs)
        # defaults to euclidean distance
        if cost_fn is None:
            cost_fn = lambda source, target: torch.cdist(source, target)**2
        self.method = method
        self.solver_kwargs = solver_kwargs
        self.ot_fn = ot_fn
        self.cost_fn = cost_fn
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.replace = replace

    def match_groups(
        self,
        source: TensorLike,
        target: TensorLike,
    ) -> tuple[TensorLike, TensorLike]:
        """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

        :param source: A tensor or array of values containing the data coming from the source distribution.
        :type source: class:`TensorLike`

        :param target: A tensor or array of values containing the data coming from the target distribution.
        :type target: class:`TensorLike`
        """
        # computing weights
        src_weights = pot.unif(source.shape[0])
        tgt_weights = pot.unif(target.shape[0])
        # moving arrays to torch tensors
        if isinstance(source, np.ndarray):
            source = torch.from_numpy(source)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        # flattening tensors
        source = torch.flatten(source, start_dim=1)
        target = torch.flatten(target, start_dim=1)
        # computing cost matrix
        distance_matrix = self.cost_fn(source, target)
        # optional normalization of cost
        if self.normalize_cost:
            distance_matrix = distance_matrix / distance_matrix.max()
        # computing coupling matrix
        coupling_matrix = self.ot_fn(
            src_weights,
            tgt_weights,
            distance_matrix.detach().cpu().numpy()
        )
        # checking for numerical errors in the coupling matrix
        if not np.all(np.isfinite(coupling_matrix)):
            msg = f"Non finite values found in `coupling_matrix` \n {coupling_matrix=} \n {source=} \n {target=} \n {distance_matrix.mean()=} \n {distance_matrix.max()=}"
            logger.warning(msg)
        if np.abs(coupling_matrix.sum()) < 1e-8:
            msg = f""
            logger.warning(msg)
            coupling_matrix = np.ones_like(coupling_matrix) / coupling_matrix.size
        # retrieving coupling probabilities
        coupling_probs = coupling_matrix.flatten()
        coupling_probs = coupling_probs / coupling_probs.sum()
        # sampling indices
        choices = np.random.choice(
            coupling_matrix.shape[0]*coupling_matrix.shape[1],
            p=coupling_probs,
            size=source.shape[0],
            replace=self.replace,
        )
        source_idxs, target_idxs = np.divmod(
            choices,
            coupling_matrix.shape[1]
        ) 
        return source_idxs, target_idxs


class IndependentCoupling(Coupling):
    """Samples independently from source and target distributions"""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""

    def match_groups(
        self,
        source: TensorLike,
        target: TensorLike,
    ) -> tuple[TensorLike, TensorLike]:
        """Matches the :param:`source` and :param:`target` groups and returns the respective indices.

        :param source: A tensor or array of values containing the data coming from the source distribution.
        :type source: class:`TensorLike`

        :param target: A tensor or array of values containing the data coming from the target distribution.
        :type target: class:`TensorLike`
        """
        # randomy permuting the tensors
        src_random_perm_idx = torch.randperm(source.shape[0])
        tgt_random_perm_idx = torch.randperm(target.shape[0])

        if source.shape[0] == target.shape[0]:
            return src_random_perm_idx, tgt_random_perm_idx
        elif source.shape[0] < target.shape[0]:
            return src_random_perm_idx, tgt_random_perm_idx[: source.shape[0]]
        else:
            return src_random_perm_idx[: target.shape[0]], tgt_random_perm_idx
