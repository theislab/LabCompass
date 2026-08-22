import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from anndata import AnnData
import torch

from labcompass.data.dataloaders import SequentialDataLoader, SequentialValDataLoader
from labcompass.data.datamanager import DataManager


from labcompass.networks.inference_networks import PerturbationApproximatePosterior
from labcompass.models.base import BaseModel
from labcompass.training import BaseCallBack, TargetPredictionTrainer
from labcompass.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["TargetPredictionModel"]


class TargetPredictionModel(BaseModel):
    """"""
    def __init__(
        self,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""

        self.device_id = device_id
        self.device = torch.device(self.device_id)
        self.target_prediction_model = None
        self.target_prediction_model_trained = False

    def prepare_train_data(
        self,
        train_adata: AnnData,
        sample_rep: str | None = None,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None = None,
        target_covariates_in_obsm: dict[str, bool] | None = None,
        target_covariates_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """"""

        data_manager = DataManager(
            train_adata,
            sample_rep=sample_rep,
            load_target_covariates=True,
            target_covariates=target_covariates,
            target_covariates_in_obsm=target_covariates_in_obsm,
            target_covariates_kwargs=target_covariates_kwargs,
        )
        train_data = data_manager.get_data(train_adata)

        self.data_manager = data_manager
        self.train_data = train_data

    def prepare_validation_data(
        self,
        validation_adata: AnnData,
    ) -> None:
        """Prepares the data for validation and initializs the :attr:`FlowMatching.validation_data` attribute of the model.

        :param validation_adata: An instance of :class:`AnnData` containing the validation data.
            It should satisfy the same requirements as the one used to construct the training data.
        :type validation_adata: class:`AnnData`
        """
        validation_data = self.data_manager.get_data(validation_adata)
        self.validation_data =  validation_data

    def prepare_model(
        self,
        target_covariates: str | Sequence[str],
        target_covariates_dims: int | dict[str, int],
        target_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
        target_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        target_covariates_use_shared_representation: bool = False,
        target_covariates_latent_dim: int = 1024,
        target_covariates_encoder_mlp_kwargs: dict[str, Any] | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
    ) -> None:
        """"""
        # preparing input with some sanity checks
        if isinstance(target_covariates, str):
            target_covariates = (target_covariates, )

        if isinstance(target_covariates_dims, int):
            msg = f"When `target_covariates_dims` is of type `int`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_dims = {target_covariates[0]: target_covariates_dims}
        
        if isinstance(target_covariates_noise_models, str):
            msg = f"When `target_covariates_noise_models` is of type `str`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_noise_models = {target_covariates[0]: target_covariates_noise_models}
        if target_covariates_noise_models is None:
            target_covariates_noise_models = {target_covariate: None for target_covariate in target_covariates}

        if target_covariates_predictor_kwargs is None:
            target_covariates_predictor_kwargs = {
                target_covariate: {} for target_covariate in target_covariates
            }

        # checking types
        msg = f""
        assert isinstance(target_covariates, Sequence), msg

        msg = f""
        assert isinstance(target_covariates_dims, dict), msg

        msg = f""
        assert isinstance(target_covariates_noise_models, dict), msg

        msg = f""
        assert isinstance(target_covariates_predictor_kwargs, dict), msg

        # storing the settings here as attributes
        self.target_covariates = target_covariates
        self.target_covariates_dims = target_covariates_dims
        self.target_covariates_noise_models = target_covariates_noise_models
        self.target_covariates_predictor_kwargs = target_covariates_predictor_kwargs
        self.target_covariates_use_shared_representation = target_covariates_use_shared_representation
        self.target_covariates_latent_dim = target_covariates_latent_dim
        self.target_covariates_encoder_mlp_kwargs = target_covariates_encoder_mlp_kwargs

        # initializing the predictor for each target covariate
        self.target_prediction_model = PerturbationApproximatePosterior(
            self.state_dim,
            freeze_grads=False,
            target_output_dims=self.target_covariates_dims,
            noise_models=self.target_covariates_noise_models,
            covariate_kwargs=self.target_covariates_predictor_kwargs,
            use_shared_representation=self.target_covariates_use_shared_representation,
            latent_dim=self.target_covariates_latent_dim,
            encoder_mlp_kwargs=self.target_covariates_encoder_mlp_kwargs,
        )
        self.target_prediction_model = self.target_prediction_model.float()
        self.target_prediction_model = self.target_prediction_model.to(self.device)

        # optimizer and scheduler 
        self.target_prediction_optimizer = optimizer_class(
            self.target_prediction_model.parameters(),
            **optimizer_kwargs,
        )

        self.target_prediction_lr_scheduler = None
        self.target_prediction_lr_scheduler_step = None
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler_class is not None:
            self.target_prediction_lr_scheduler = lr_scheduler_class(self.target_prediction_optimizer, **lr_scheduler_kwargs)
            self.target_prediction_lr_scheduler_step = lr_scheduler_step

    def train(
        self,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
        loss_fn_kwargs: dict[str, Any] | None = None
    ) -> None:
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        # initializing data loader
        self.target_predictor_train_dataloader = SequentialDataLoader(
            self.train_data,
            train_batch_size,
            state_transforms=state_transforms,
            device_id=self.device_id
        )

        # initialize trainer
        self.target_predictor_trainer = TargetPredictionTrainer(
            self.target_prediction_model,
            self.target_prediction_optimizer,
            lr_scheduler=self.target_prediction_lr_scheduler,
            lr_scheduler_step=self.target_prediction_lr_scheduler_step,
            callbacks=callbacks,
            grad_steps_log_interval=grad_steps_log_interval,
            loss_fn_kwargs=loss_fn_kwargs,
        )

        # optional validation data
        self.target_predictor_validation_dataloader = None
        if self.validation_data is not None:
            self.target_predictor_validation_dataloader = SequentialValDataLoader(
                self.validation_data,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )

        # fitting the trainer
        self.target_predictor_trainer.fit(
            num_training_steps,
            self.target_predictor_train_dataloader,
            self.target_predictor_validation_dataloader,
            valid_freq,
        )

        self.target_prediction_model_trained = True

    def predict(
        self,
        control_states: torch.Tensor,
        no_grad: bool = True,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """"""
        self.target_prediction_model.eval()
        if no_grad:
            with torch.no_grad():
                return self.target_prediction_model(control_states)
        else:
            return self.target_prediction_model(control_states)

    @property
    def state_dim(self):
        return self.train_data.state_data.shape[-1]
