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

    def __use_callback_on_grad_step(
        self,
        step_idx: int,
    ) -> bool:
        """"""
        if self.grad_steps_log_interval is None:
            return False
        if (step_idx + 1)%self.grad_steps_log_interval == 0:
            return True
        return False

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
        # running callbacks
        if self.callbacks is not None:
            if self.__use_callback_on_grad_step(step_idx):
                self.callbacks.run_on_grad_step(log_dict)
        return log_dict

    def __validation_step(
        self,
        batch: dict[str, Tensor],
    ) -> dict[str, TensorLike]:
        """"""
        self.model.eval()
        with torch.no_grad():
            val_preds, val_gt = self._validation_step(batch)
            val_preds = val_preds.cpu().numpy()
            val_gt = val_gt.cpu().numpy()
        # learning rate scheduler step
        if self.lr_scheduler_step == "valid_step" and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        # running callbacks
        if self.callbacks is not None:
            metrics = self.callbacks.run_on_valid_step()
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
        validation_dataloader: BaseDataLoader | None = None,
        valid_freq: int | None = None,
    ) -> None:
        """"""

        self.training_logs = {LossFields.LOSS: []}

        iterator = range(num_training_steps)
        prog_bar = tqdm(iterator)

        do_validation = validation_dataloader is not None
        if do_validation and valid_freq is None:
            valid_freq = num_training_steps

        # running callbacks
        if self.callbacks is not None:
            self.callbacks.run_on_train_begin()

        for grad_step in iterator:
            batch = train_dataloader.sample()
            log_dict = self.__train_step(grad_step, batch)
            self.__update_logs(log_dict)

            # updaring progress bar
            if (grad_step + 1) % self.grad_step_interval_log and grad_step > 0:
                prog_bar.set_description(f"Loss: {log_dict[LossFields.LOSS]:.4f}")
                prog_bar.update()

            # validation step
            if do_validation:
                if (grad_step + 1) % valid_freq == 0 and grad_step > 0:
                    # skipping if no dataloader provided
                    if validation_dataloader is None:
                        continue

                    batch = validation_dataloader.sample()
                    metrics = self.__validation_step(batch)
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


