from collections.abc import Sequence
from typing import Any, Literal

import omegaconf
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
    callback_type: Literal["computational", "logging"]

    def run_on_train_begin(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        pass

    def run_on_valid_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        pass

    def run_on_train_end(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        pass


class ComputationalCallBack(BaseCallBack):
    """"""
    callback_type: Literal["computational", "logging"] = "computational"

    def run_on_valid_step(
        self,
        preds: TensorLike,
        target: TensorLike,
    ) -> dict[str, Any]:
        """"""
        raise NotImplementedError


class LoggingCallBack(BaseCallBack):
    """"""
    callback_type: Literal["computational", "logging"] = "logging"

    def run_on_valid_step(
        self,
        log_dict: dict[str, Any],
    ) -> None:
        """"""
        raise NotImplementedError


class MetricsCallBack(ComputationalCallBack):
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
            metric = vars(Metrics())[metric_id]
            metrics[metric_id] = metric(preds, target)
        return metrics


class WandBLogger(LoggingCallBack):
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
        self.kwargs = kwargs

    def run_on_train_begin(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        # moving configuration to omegaconf
        config = self.config
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)

        # initializing settings
        settings = wandb.Settings(**self.kwargs)

        # login to wandb and initialize run
        wandb.login()
        wandb.init(
            project=self.project_name,
            config=config,
            dir=self.log_dir,
            settings=settings,
        )

        # storing run name as an attribute
        self.run_name = wandb.run.name

    def run_on_valid_step(
        self,
        log_dict: dict[str, Any],
    ) -> None:
        """"""
        wandb.log(log_dict)

    def run_on_train_end(
        self,
    ) -> None:
        """"""
        wandb.finish()


class TrainingCallBacks(BaseCallBack):
    """"""
    
    def __init__(
        self,
        callbacks: Sequence[BaseCallBack],
    ) -> None:
        """"""
        self.callbacks = callbacks
    
    @property
    def computational_callbacks(
        self,
    ) -> Sequence[BaseCallBack]:
        """"""
        return [callback for callback in self.callbacks if callback.callback_type == "computational"]

    @property
    def logging_callbacks(
        self,
    ) -> Sequence[BaseCallBack]:
        """"""
        return [callback for callback in self.callbacks if callback.callback_type == "logging"]

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
        # run computational callbacks first
        callback_out = {}
        for callback in self.computational_callbacks:
            callback_metrics = callback.run_on_valid_step(preds, target)
            callback_out.update(callback_metrics)
        # then log the results
        for callback in self.logging_callbacks:
            callback_metrics = callback.run_on_valid_step(callback_out)
        return callback_out

    def run_on_train_end(
        self,
    ) -> None:
        """"""
        for callback in self.callbacks:
            callback.run_on_train_end()
