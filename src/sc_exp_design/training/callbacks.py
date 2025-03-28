from collections.abc import Sequence
from typing import Any

import wandb

from sc_exp_design.metrics import Metrics
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike


__all__ = [
    "BaseCallBack",
    "MetricsCallBack",
]


class BaseCallBack:
    """"""

    def run_on_train_begin(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError

    def run_on_valid_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError

    def run_on_train_end(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError


class MetricsCallBack(BaseCallBack):
    """"""

    def __init__(
        self,
        metric_ids: Sequence[str],
        state_transforms: Transform | None = None,
    ) -> None:
        """"""
        self.metric_ids = metric_ids
        self.state_transforms = state_transforms

    def run_on_valid_step(
            self,
            preds: TensorLike,
            target: TensorLike,
        ) -> dict[str, float]:
        """"""
        metrics = {}
        if self.state_transforms is not None:
            preds = self.state_transforms(preds)
            target = self.state_transforms(target)
        for metric_id in self.metric_ids:
            metric = dict(Metrics)[metric_id]
            metrics[metric_id] = metric(preds, target)
        return metrics


class WandBLogger(BaseCallBack):
    """"""

    def __init__(
        self,
        project_name: str,
        log_dir: str,
        config: dict[str, Any],
        **kwargs,
    ) -> None:
        """"""
        self.project_name = project_name
        self.log_dir = log_dir
        self.config = config

    def run_on_train_begin(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        raise NotImplementedError

    def run_on_valid_step(
        self,
        preds: TensorLike,
        target: TensorLike,
    ) -> dict[str, Any]:
        """"""


class TrainingCallBacks(BaseCallBack):
    """"""
    
    def __init__(
        self,
        callbacks: Sequence[BaseCallBack],
    ) -> None:
        """"""
        self.callbacks = callbacks
    
    def run_on_train_begin(
        self,
    ) -> None:
        """"""
        for callback in self.callbacks:
            callback.run_on_train_begin()

    def run_on_valid_step(
        self,
        preds: TensorLike,
        target: TensorLike,
    ) -> dict[str, Any]:
        """"""
        for callback in self.callbacks:
            callback.run_on_valid_step(preds, target)

    def run_on_train_end(
        self,
    ) -> None:
        """"""
        for callback in self.callbacks:
            callback.run_on_train_end()
