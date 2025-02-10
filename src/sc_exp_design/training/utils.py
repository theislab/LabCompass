from functools import partial
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from sc_exp_design.constants import LossFields, ParamsFields

__all__ = [
    "gaussian_rec_loss",
    "neg_bin_rec_loss",
    "binary_classification_loss",
    "reconstruction_loss_noise_model",
    "compute_cond_vars_inference_loss",
    "compute_pert_inference_loss",
    "compute_latent_perturbation_inference_loss",
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
) -> Tensor:
    """"""
    target = target.long()
    loss = nn.functional.cross_entropy(params, target)
    return loss


def reconstruction_loss_noise_model(
    params: dict[str, Tensor],
    samples: Tensor,
    noise_model: Literal["gaussian", "neg_bin"],
    cov_estimation_mode: Literal["isotropic", "anisotropic"] | None = None,
) -> Tensor:
    """"""
    # loss on the source posterior
    if noise_model == "gaussian":
        msg = f"With {noise_model=} `cov_estimation_mode` needs to be in `['isotropic', 'anisotropic']`, found `None`."
        assert cov_estimation_mode is not None, msg
        loss_fn = partial(gaussian_rec_loss, cov_estimation_mode=cov_estimation_mode)
    elif noise_model == "neg_bin":
        loss_fn = neg_bin_rec_loss
    else:
        msg = f"{noise_model=} not supported (possible values `['gaussian', 'neg_bin']`)."
        raise ValueError(msg)
    loss = loss_fn(
        params,
        samples,
    )
    return loss


def compute_cond_vars_inference_loss(
    loss: Tensor,
    source: Tensor,
    target: Tensor,
    src_posterior_params: dict[str, Tensor],
    tgt_posterior_params: dict[str, Tensor],
    src_noise_model: str,
    tgt_noise_model: str,
    src_cov_estimation_mode: str | None,
    tgt_cov_estimation_mode: str | None,
    add_loss: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """"""
    # loss on the source posterior
    src_posterior_loss = reconstruction_loss_noise_model(
        src_posterior_params,
        source,
        src_noise_model,
        src_cov_estimation_mode,
    )
    # loss on the target posterior
    tgt_posterior_loss = reconstruction_loss_noise_model(
        tgt_posterior_params,
        target,
        tgt_noise_model,
        tgt_cov_estimation_mode,
    )
    # adding loss and updating creating log dictionary
    if add_loss:
        loss = loss + src_posterior_loss + tgt_posterior_loss
    loss_dict = {
        LossFields.SOURCE_LOSS: src_posterior_loss.detach().cpu().item(),
        LossFields.TARGET_LOSS: src_posterior_loss.detach().cpu().item(),
    }
    return loss, loss_dict


def compute_pert_inference_loss(
    loss: Tensor,
    pert_posterior_params: dict[str, Tensor],
    pert_target_rep: dict[str, Tensor],
    pert_noise_models: dict[str, str],
    pert_cov_estimation_modes: dict[str, str] | None = None,
    add_loss: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """"""
    loss_dict = {}
    for pert_target_cov_id, pert_target_cov_params in pert_posterior_params.items():
        target_rep = pert_target_rep[pert_target_cov_id]
        noise_model = pert_noise_models[pert_target_cov_id]
        if noise_model is None:
            pert_posterior_loss_fn = binary_classification_loss
        elif noise_model == "gaussian":
            msg = f"With {noise_model=} `pert_cov_estimation_mode` needs to be in `['isotropic', 'anisotropic']`, found `None`."
            assert pert_cov_estimation_modes is not None, msg
            cov_estimation_mode = pert_cov_estimation_modes[pert_target_cov_id]
            pert_posterior_loss_fn = partial(gaussian_rec_loss, cov_estimation_mode=cov_estimation_mode)
        elif noise_model == "neg_bin":
            pert_posterior_loss_fn = neg_bin_rec_loss
        else:
            msg = f"{noise_model=} not supported (possible values `['gaussian', 'neg_bin']`)."
            raise ValueError(msg)

        pert_posterior_loss = pert_posterior_loss_fn(
            pert_target_cov_params,
            target_rep,
        )
        # adding loss
        if add_loss:
            loss = loss + pert_posterior_loss

        # updating log dict
        cov_loss_id = f"{pert_target_cov_id}_{LossFields.PERTURBATION_LOSS}"
        loss_dict[cov_loss_id] = pert_posterior_loss.detach().cpu().item()

    return loss, loss_dict


def compute_latent_perturbation_inference_loss(
    loss: Tensor,
    params: dict[str, Tensor],
    latent_perturbation: Tensor,
    cov_estimation_mode: Literal["isotropic", "anisotropic"],
    add_loss: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """"""
    latent_pert_inference_loss = reconstruction_loss_noise_model(
        params, latent_perturbation, "gaussian", cov_estimation_mode=cov_estimation_mode
    )
    loss_dict = {LossFields.LATENT_PERTURBATION_INF_LOSS: latent_pert_inference_loss.detach().cpu().item()}
    if add_loss:
        loss = loss + latent_pert_inference_loss
    return loss, loss_dict
