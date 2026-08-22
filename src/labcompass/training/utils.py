from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn

from labcompass.constants import LossFields, ParamsFields

__all__ = [
    "gaussian_rec_loss",
    "neg_bin_rec_loss",
    "binary_classification_loss",
    "reconstruction_loss_noise_model",
]


def gaussian_rec_loss(
    params: dict[str, Tensor],
    target: Tensor,
    cov_estimation_mode: Literal["isotropic", "anisotropic"] = "isotropic",
) -> Tensor:
    """Computes the reconstruction loss for a Gaussian noise model

    :param params:
    :type params: class: `dict[str, torch.Tensor]`

    :param target:
    :type target: class: `torch.Tensor`

    :param cov_estimation_mode:
    :type cov_estimation_mode: class: `Literal["isotropic", "anisotropic"]`
    """
    if cov_estimation_mode == "isotropic":
        mean = params[ParamsFields.MEAN]
        cov = params[ParamsFields.COVARIANCE]
        dim = mean.shape[1]
        cov = cov**2
        loss = (
            (0.5 / cov) * torch.sum((target - mean) ** 2, axis=1)
            + 0.5 * torch.log(cov)
            + dim * 0.5 * torch.log(torch.tensor([2 * np.pi], device=mean.device))
        )
        loss = torch.mean(loss)
    elif cov_estimation_mode == "anisotropic":
        msg = f"{cov_estimation_mode=} not currently supported."
        raise NotImplementedError(msg)
    else:
        msg = f"{cov_estimation_mode=} not currently supported (possible values `['isotropic', 'anisotropic']`)"
        raise ValueError(msg)
    return loss


def neg_bin_rec_loss(
    params: dict[str, Tensor],
    target: Tensor,
) -> Tensor:
    """"""
    return torch.tensor([0.0])


def binary_classification_loss(
    params: Tensor,
    target: Tensor,
    loss_fn_kwargs: dict[str, Any] | None,
) -> Tensor:
    """"""
    loss_fn_kwargs = {} if loss_fn_kwargs is None else loss_fn_kwargs
    target = target.long()
    loss = nn.functional.cross_entropy(params, target, **loss_fn_kwargs)
    return loss


def reconstruction_loss_noise_model(
    params: dict[str, Tensor],
    samples: Tensor,
    noise_model: Literal["gaussian", "neg_bin"] | None,
    cov_estimation_mode: Literal["isotropic", "anisotropic"] | None = None,
    allow_noise_model_to_be_none: bool = True,
    loss_fn_kwargs: dict[str, Any] | None = None
) -> Tensor:
    """"""
    # loss on the source posterior
    if noise_model == "gaussian":
        msg = f"With {noise_model=} `cov_estimation_mode` needs to be in `['isotropic', 'anisotropic']`, found `None`."
        assert cov_estimation_mode is not None, msg
        loss_fn = partial(gaussian_rec_loss, cov_estimation_mode=cov_estimation_mode)
    elif noise_model == "neg_bin":
        loss_fn = neg_bin_rec_loss
    elif allow_noise_model_to_be_none and noise_model is None:
        loss_fn = binary_classification_loss
    else:
        msg = (
            f"{noise_model=} not supported (possible values `['gaussian', 'neg_bin', None]`)." if allow_noise_model_to_be_none else
            f"{noise_model=} not supported (possible values `['gaussian', 'neg_bin']`)."
        )
        raise ValueError(msg)
    loss = loss_fn(
        params,
        samples,
        loss_fn_kwargs,
    )
    return loss

def compute_pert_inference_loss(
    pert_posterior_params: dict[str, Tensor],
    pert_target_rep: dict[str, Tensor],
    pert_noise_models: dict[str, str],
    pert_cov_estimation_modes: dict[str, str] | None = None,
    add_loss: bool = True,
    allow_noise_model_to_be_none: bool = True,
    loss_fn_kwargs=None
) -> tuple[Tensor, dict[str, Tensor]]:
    """"""
    loss = torch.zeros((), requires_grad=True)
    loss_dict = {}
    for pert_target_cov_id, pert_target_cov_params in pert_posterior_params.items():
        # retrieving settings for current target covariate
        target_rep = pert_target_rep[pert_target_cov_id]
        noise_model = pert_noise_models[pert_target_cov_id]
        cov_estimation_mode = pert_cov_estimation_modes[pert_target_cov_id]
        # computing loss for noise model
        pert_posterior_loss = reconstruction_loss_noise_model(
            pert_target_cov_params,
            target_rep,
            noise_model,
            cov_estimation_mode=cov_estimation_mode,
            allow_noise_model_to_be_none=allow_noise_model_to_be_none,
            loss_fn_kwargs={} if loss_fn_kwargs is None else loss_fn_kwargs,
        )

        # adding loss
        loss = loss + pert_posterior_loss

        # updating log dict
        cov_loss_id = f"{pert_target_cov_id}_{LossFields.PERTURBATION_LOSS}"
        loss_dict[cov_loss_id] = pert_posterior_loss.detach().cpu().item()

    return loss, loss_dict
