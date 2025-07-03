from collections.abc import Sequence
from typing import Any, Literal

import omegaconf
import wandb

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.metrics import Metrics
from sc_exp_design.transforms import Transform
from sc_exp_design.types import TensorLike
import numpy as np


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

    def run_on_train_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        pass

    def run_on_valid_step(
        self,
        prediction_dict: dict[str, dict[str, TensorLike]],
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


class LoggingCallBack(BaseCallBack):
    """"""
    callback_type: Literal["computational", "logging"] = "logging"


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

    def _run_on_valid_step(
            self,
            preds: TensorLike,
            target: TensorLike,
            condition: str = None
        ) -> dict[str, float]:
        """"""
        metrics = {}
        if self.state_transforms is not None:
            preds = self.state_transforms(preds, np.array([condition] * preds.shape[0]))
            target = self.state_transforms(target, np.array([condition] * preds.shape[0]))
        for metric_id in self.metric_ids:
            metric = vars(Metrics())[metric_id]
            metrics[metric_id] = metric(preds, target)
        return metrics
    
    def run_on_valid_step(
        self,
        predictions_dict: dict[str, dict[str, TensorLike]],
    ) -> dict[str, float]:
        """"""
        # defining output dictionary
        metrics = {}
        
        # iterating over the predictions for each condition
        for perturbation, perturbation_prediction_data in predictions_dict.items():
            # parsing prediction data dictionary
            predictions = perturbation_prediction_data[PredictionFields.PREDICTION_DATA]
            targets = perturbation_prediction_data[DataFields.TARGET_STATE]
            
            # computing the metrics for the current perturbation
            perturbation_metrics = self._run_on_valid_step(predictions, targets, perturbation)
            
            # updating the metrics 
            if self.state_transforms is not None:
                metrics.update(
                    {
                        f"{perturbation}_{metric_id}_{self.state_transforms.__class__.__name__}": metric_value for metric_id, metric_value in perturbation_metrics.items()
                    }
                )
            else:
                metrics.update(
                    {
                        f"{perturbation}_{metric_id}": metric_value for metric_id, metric_value in perturbation_metrics.items()
                    }
                )
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
            config=omegaconf.OmegaConf.to_container(config, resolve=True),
            dir=self.log_dir,
            settings=settings,
        )

        # storing run name as an attribute
        self.run_name = wandb.run.name

    def run_on_train_step(
        self,
        grad_step: int,
        logs: dict[str, Any],
    ) -> None:
        """"""
        # logging the training step
        wandb.log(
            {
                "train_step": grad_step,
                **logs,
            }
        )

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
    
    def run_on_train_step(
        self,
        grad_step: int,
        logs: dict[str, Any],
    ) -> None:
        """"""
        # then log the results
        for callback in self.logging_callbacks:
            callback.run_on_train_step(grad_step, logs)

    def run_on_valid_step(
        self,
        prediction_dict: dict[str, dict[str, TensorLike]],
    ) -> dict[str, Any]:
        """"""
        # run computational callbacks first
        callback_out = {}
        for callback in self.computational_callbacks:
            callback_metrics = callback.run_on_valid_step(prediction_dict)
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
