import abc
from collections.abc import Callable
from typing import Any, Literal

import ot as pot
import torch

from sc_exp_design.types import TensorLike

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
        method: Literal["exact", "sinkhorn", "unbalanced", "partial"],
        cost_fn: Callable[[TensorLike, TensorLike], TensorLike],
        reg: float | None,
        reg_m: float | None,
    ) -> None:
        # sanity checks
        if method in ["sinkhorn", "partial"] and reg is None:
            msg = f"{method=} requires `reg` to be a `float`, `None` found"
            raise ValueError(msg)
        if method == "unbalanced" and reg:
            msg = f"{method=} requires `reg` to be a `float`, `None` found"
            raise ValueError(msg)
        if method == "unbalanced" and reg_m is None:
            msg = f"{method=} requires `reg_m` to be a `float`, `None` found"
            raise ValueError(msg)
        self.method = method
        self.cost_fn = cost_fn
        self.reg = reg
        self.reg_m = reg_m

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
        src_weights = pot.unif(source.shape[0])
        tgt_weights = pot.unif(target.shape[0])
        distance_matrix = self.cost_fn(source, target)
        raise NotImplementedError


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
