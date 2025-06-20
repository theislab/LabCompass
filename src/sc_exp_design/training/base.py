import abc
from collections.abc import Callable, Sequence
from typing import Any, Literal

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from torch import Tensor
from tqdm import tqdm

from sc_exp_design.constants import LossFields
from sc_exp_design.data.dataloaders import BaseDataLoader
from sc_exp_design.types import TensorLike


class BaseTrainer(abc.ABC):
    """"""

    @abc.abstractmethod
    def _train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """
        """
        raise NotImplementedError
        # return loss, log_dict

    @abc.abstractmethod
    def _validation_step(
        self,
        batch: dict[str, TensorLike],
    ) -> tuple[TensorLike]:
        """"""
        raise NotImplementedError
        # return predictions, target

    def __train_step(
        self,
        step_idx: int,
        batch: dict[str, TensorLike],
    ) -> TensorLike:
        """"""
        # optimizations step
        self.model.train()
        self.optimizer.zero_grad()
        loss, log_dict = self._train_step(step_idx, batch)
        loss.backward()
        self.optimizer.step()
        # learning rate scheduler step
        if self.lr_scheduler_step == "grad_step" and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log_dict

    def __validation_step(
        self,
        batch: dict[str, Tensor],
        val_id: str,
    ) -> dict[str, TensorLike]:
        """"""
        self.model.eval()
        with torch.no_grad():
            prediction_dict = self._validation_step(batch)
        # learning rate scheduler step
        if self.lr_scheduler_step == "valid_step" and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # running callbacks
        metrics = {}
        if self.callbacks is not None:
            metrics = self.callbacks.run_on_valid_step(prediction_dict, val_id)
        return metrics

    def __update_logs(
        self,
        metrics: dict[str, float],
    ) -> None:
        """"""
        for metric_id, metric_val in metrics.items():
            if metric_id not in self.training_logs.keys():
                self.training_logs[metric_id] = []
            self.training_logs[metric_id].append(metric_val)

    def fit(
        self,
        num_training_steps: int,
        train_dataloader: BaseDataLoader,
        validation_dataloaders: dict | None = None,
        valid_freq: int | None = None,
    ) -> None:
        """"""

        self.training_logs = {LossFields.LOSS: []}

        iterator = range(num_training_steps)
        prog_bar = tqdm(iterator)

        do_validation = validation_dataloaders is not None
        if do_validation and valid_freq is None:
            valid_freq = num_training_steps

        # retrieving the training step log interval
        grad_steps_log_interval = self.grad_steps_log_interval
        if self.grad_steps_log_interval is None:
            grad_steps_log_interval = 100

        # running callbacks
        if self.callbacks is not None:
            self.callbacks.run_on_train_begin()

        for grad_step in iterator:
            batch = train_dataloader.sample()
            log_dict = self.__train_step(grad_step, batch)
            self.__update_logs(log_dict)

            # updating progress bar and log
            if (grad_step + 1) % grad_steps_log_interval == 0 and grad_step > 0:
                prog_bar.set_description(f"Loss: {log_dict[LossFields.LOSS]:.4f}")
                if self.callbacks is not None:
                    self.callbacks.run_on_train_step(
                        grad_step=grad_step,
                        logs=log_dict,
                    )
            prog_bar.update()
        
            # validation step
            if do_validation:
                if (grad_step + 1) % valid_freq == 0 and grad_step > 0:
                    # skipping if no dataloader provided
                    if validation_dataloaders is None:
                        continue
                    metrics = {}
                    for val_id, validation_dataloader in validation_dataloaders.items():
                        batch = validation_dataloader.sample()
                        metrics = metrics | self.__validation_step(batch, val_id)
                    self.__update_logs(metrics)

                # running callbacks
        if self.callbacks is not None:
            self.callbacks.run_on_train_end()


    def plot_training_logs(
        self,
        figsize: Sequence[int] = (3, 3),
        keys_to_plot: str | Sequence[str] = LossFields.LOSS,
        show: bool = False,
    ) -> tuple[Figure, Axes]:
        """"""
        # handling keys to plot
        if isinstance(keys_to_plot, str):
            keys_to_plot = (keys_to_plot, )
        # sanity checks
        for key in keys_to_plot:
            msg = f""
            assert key in self.training_logs.keys(), msg
        # retrieving the logs we want to plot
        logs_to_plot = {log_id: log_data for log_id, log_data in self.training_logs.items() if log_id in keys_to_plot}

        fig, axes = plt.subplots(1, len(logs_to_plot), figsize=figsize)
        for idx, (loss_id, loss_history) in enumerate(logs_to_plot.items()):
            if len(logs_to_plot) == 1:
                current_axes = axes
            else:
                current_axes = axes[idx]
            current_axes.set_title(loss_id)
            current_axes.plot(loss_history)
        if show:
            fig.show()
        return fig, axes
