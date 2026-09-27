from collections.abc import Sequence
from typing import Any, Literal

import omegaconf
import wandb

from labcompass.constants import DataFields, PredictionFields
from labcompass.metrics import Metrics
from labcompass.transforms import Transform
from labcompass.types import TensorLike

__all__ = [
    "BaseCallBack",
    "MetricsCallBack",
]


class BaseCallBack:
    """Base class defining the callback interface invoked by :meth:`BaseTrainer.fit` at the different stages of training."""

    callback_type: Literal["computational", "logging"]

    def run_on_train_begin(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Hook invoked once before training begins. This base implementation does nothing; subclasses may override it to perform setup (e.g. initializing a logging run)."""
        pass

    def run_on_train_step(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Hook invoked periodically during training. This base implementation does nothing; subclasses may override it to log training metrics."""
        pass

    def run_on_valid_step(
        self,
        prediction_dict: dict[str, dict[str, TensorLike]],
    ) -> None:
        """Hook invoked after a validation step. This base implementation does nothing; subclasses may override it to compute or log validation metrics.

        :param prediction_dict: Dictionary mapping each perturbation identifier to a dictionary containing the model predictions and the target values for that perturbation.
        :type prediction_dict: class:`dict[str, dict[str, TensorLike]]`
        """
        pass

    def run_on_train_end(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Hook invoked once at the end of training, after the last gradient step. This base implementation does nothing; subclasses may override it to perform teardown (e.g. closing a logging run)."""
        pass


class ComputationalCallBack(BaseCallBack):
    """"""
    callback_type: Literal["computational", "logging"] = "computational"


class LoggingCallBack(BaseCallBack):
    """"""
    callback_type: Literal["computational", "logging"] = "logging"


class MetricsCallBack(ComputationalCallBack):
    """Computational callback that computes a set of metrics on model predictions during validation.

    :param metric_ids: Sequence of attribute names of :class:`labcompass.metrics.Metrics` identifying which metrics to compute at each validation step.
    :type metric_ids: class:`Sequence[str]`

    :param state_transforms: Optional transform applied to both predictions and targets before computing the metrics, e.g. to invert a preprocessing transform. Defaults to `None`.
    :type state_transforms: class:`Transform | None`
    """

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

    def run_on_valid_step(
        self,
        predictions_dict: dict[str, dict[str, TensorLike]],
    ) -> dict[str, float]:
        """Computes the configured metrics for every perturbation in `predictions_dict`, using its predictions and target states.

        :param predictions_dict: Dictionary mapping each perturbation identifier to a dictionary containing the predicted states (under :attr:`PredictionFields.PREDICTION_DATA`) and the target states (under :attr:`DataFields.TARGET_STATE`).
        :type predictions_dict: class:`dict[str, dict[str, TensorLike]]`

        :return: Dictionary mapping `"{perturbation}_{metric_id}"` to the corresponding metric value.
        :rtype: class:`dict[str, float]`
        """
        # defining output dictionary
        metrics = {}

        # iterating over the predictions for each condition
        for perturbation, perturbation_prediction_data in predictions_dict.items():
            # parsing prediction data dictionary
            predictions = perturbation_prediction_data[PredictionFields.PREDICTION_DATA]
            targets = perturbation_prediction_data[DataFields.TARGET_STATE]

            # computing the metrics for the current perturbation
            perturbation_metrics = self._run_on_valid_step(predictions, targets)

            # updating the metrics
            metrics.update(
                {
                    f"{perturbation}_{metric_id}": metric_value for metric_id, metric_value in perturbation_metrics.items()
                }
            )
        return metrics



class WandBLogger(LoggingCallBack):
    """Logging callback that streams training and validation metrics to Weights & Biases.

    :param project_name: Name of the W&B project under which the run is created.
    :type project_name: class:`str`

    :param log_dir: Local directory where W&B stores its run files.
    :type log_dir: class:`str`

    :param config: Configuration dictionary logged alongside the run. Converted to an :class:`omegaconf.DictConfig` if a plain `dict` is passed.
    :type config: class:`dict[str, Any]`

    :param kwargs: Additional keyword arguments forwarded to :class:`wandb.Settings` when initializing the run.
    :type kwargs: class:`Any`
    """

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
        """Logs into Weights & Biases and initializes a new run using :attr:`project_name`, :attr:`config`, :attr:`log_dir` and the settings built from :attr:`kwargs`. The resulting run name is stored as :attr:`run_name`."""
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
        """Logs the metrics collected at a training step to the current W&B run.

        :param grad_step: Index of the current gradient step, logged under the `"train_step"` key.
        :type grad_step: class:`int`

        :param logs: Dictionary of metric names and values collected at the current training step.
        :type logs: class:`dict[str, Any]`
        """
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
        """Logs a dictionary of validation metrics to the current W&B run.

        :param log_dict: Dictionary of metric names and values computed during the validation step.
        :type log_dict: class:`dict[str, Any]`
        """
        wandb.log(log_dict)

    def run_on_train_end(
        self,
    ) -> None:
        """Closes the current Weights & Biases run."""
        wandb.finish()


class TrainingCallBacks(BaseCallBack):
    """Aggregates multiple callbacks and dispatches each training hook to the appropriate subset of them, running computational callbacks before logging callbacks.

    :param callbacks: Sequence of callback instances to run during training.
    :type callbacks: class:`Sequence[BaseCallBack]`
    """

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
        """The subset of :attr:`callbacks` whose `callback_type` is `"computational"`.

        :return: The computational callbacks.
        :rtype: class:`Sequence[BaseCallBack]`
        """
        return [callback for callback in self.callbacks if callback.callback_type == "computational"]

    @property
    def logging_callbacks(
        self,
    ) -> Sequence[BaseCallBack]:
        """The subset of :attr:`callbacks` whose `callback_type` is `"logging"`.

        :return: The logging callbacks.
        :rtype: class:`Sequence[BaseCallBack]`
        """
        return [callback for callback in self.callbacks if callback.callback_type == "logging"]

    def run_on_train_begin(
        self,
    ) -> None:
        """Calls `run_on_train_begin` on every callback in :attr:`callbacks`."""
        for callback in self.callbacks:
            callback.run_on_train_begin()

    def run_on_train_step(
        self,
        grad_step: int,
        logs: dict[str, Any],
    ) -> None:
        """Calls `run_on_train_step` on every logging callback in :attr:`logging_callbacks`.

        :param grad_step: Index of the current gradient step, forwarded to each callback.
        :type grad_step: class:`int`

        :param logs: Dictionary of metric names and values collected at the current training step, forwarded to each callback.
        :type logs: class:`dict[str, Any]`
        """
        # then log the results
        for callback in self.logging_callbacks:
            callback.run_on_train_step(grad_step, logs)

    def run_on_valid_step(
        self,
        prediction_dict: dict[str, dict[str, TensorLike]],
    ) -> dict[str, Any]:
        """Runs all computational callbacks on the validation predictions to compute metrics, then forwards the resulting metrics to every logging callback.

        :param prediction_dict: Dictionary mapping each perturbation identifier to a dictionary containing the model predictions and the target values for that perturbation.
        :type prediction_dict: class:`dict[str, dict[str, TensorLike]]`

        :return: Dictionary of metric names and values computed by the computational callbacks.
        :rtype: class:`dict[str, Any]`
        """
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
        """Calls `run_on_train_end` on every callback in :attr:`callbacks`."""
        for callback in self.callbacks:
            callback.run_on_train_end()
