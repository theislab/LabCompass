import abc
from collections.abc import Sequence

from sc_exp_design.metrics import METRICS
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike

__all__ = [
    "CallBack",
    "MetricsCallBack",
]


class CallBack(abc.ABC):
    """"""

    @abc.abstractmethod
    def run_on_grad_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def run_on_valid_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError


class MetricsCallBack(CallBack):
    """"""

    def __init__(
        self,
        metric_ids: Sequence[str],
        state_transforms: Transform | None = None,
    ) -> None:
        """"""
        self.metric_ids = metric_ids
        self.state_transforms = state_transforms

    def run_on_valid_step(self, preds: TensorLike, target: TensorLike) -> dict[str, float]:
        """"""
        metrics = {}
        if self.state_transforms is not None:
            preds = self.state_transforms(preds)
            target = self.state_transforms(target)
        for metric_id in self.metric_ids:
            metric = METRICS[metric_id]
            metrics[metric_id] = metric(preds, target)
        return metrics
