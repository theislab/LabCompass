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
    """Computes the cross-entropy loss between predicted class logits and target class labels.

    :param params: Tensor of predicted class logits.
    :type params: class:`torch.Tensor`

    :param target: Tensor of target class labels, cast to `long` before computing the loss.
    :type target: class:`torch.Tensor`

    :param loss_fn_kwargs: Optional dictionary of keyword arguments forwarded to :func:`torch.nn.functional.cross_entropy`. `None` is treated as an empty dictionary.
    :type loss_fn_kwargs: class:`dict[str, Any] | None`

    :return: The cross-entropy loss.
    :rtype: class:`torch.Tensor`
    """
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
    """Computes the reconstruction loss for the given noise model, dispatching to the loss function matching `noise_model`.

    :param params: Dictionary of predicted noise-model parameters (e.g. mean/covariance for a Gaussian model) or, when `noise_model` is `None`, a tensor of predicted class logits.
    :type params: class:`dict[str, torch.Tensor]`

    :param samples: Tensor of target values (observed samples, or target class labels when `noise_model` is `None`) to compute the loss against.
    :type samples: class:`torch.Tensor`

    :param noise_model: Identifier of the noise model used to reconstruct `samples`. When `None` and `allow_noise_model_to_be_none` is `True`, :func:`binary_classification_loss` is used instead.
    :type noise_model: class:`Literal["gaussian", "neg_bin"] | None`

    :param cov_estimation_mode: Covariance estimation mode forwarded to :func:`gaussian_rec_loss` when `noise_model` is `"gaussian"`. Required in that case, defaults to `None`.
    :type cov_estimation_mode: class:`Literal["isotropic", "anisotropic"] | None`

    :param allow_noise_model_to_be_none: Whether `noise_model` is allowed to be `None`, in which case :func:`binary_classification_loss` is used, defaults to `True`.
    :type allow_noise_model_to_be_none: class:`bool`

    :param loss_fn_kwargs: Optional dictionary of keyword arguments forwarded to the selected loss function, defaults to `None`.
    :type loss_fn_kwargs: class:`dict[str, Any] | None`

    :return: The reconstruction loss.
    :rtype: class:`torch.Tensor`

    :raises ValueError: If `noise_model` is not one of `"gaussian"`, `"neg_bin"`, or (when `allow_noise_model_to_be_none` is `True`) `None`.
    """
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
    """Computes and aggregates the reconstruction loss for each perturbation target covariate.

    For every covariate in `pert_posterior_params`, looks up the corresponding target representation, noise model and covariance-estimation mode, computes its reconstruction loss via :func:`reconstruction_loss_noise_model`, and sums the per-covariate losses into a single scalar.

    :param pert_posterior_params: Dictionary mapping each target covariate name to the parameters predicted for its noise model.
    :type pert_posterior_params: class:`dict[str, torch.Tensor]`

    :param pert_target_rep: Dictionary mapping each target covariate name to its target representation.
    :type pert_target_rep: class:`dict[str, torch.Tensor]`

    :param pert_noise_models: Dictionary mapping each target covariate name to the identifier of its noise model, forwarded to :func:`reconstruction_loss_noise_model`.
    :type pert_noise_models: class:`dict[str, str]`

    :param pert_cov_estimation_modes: Dictionary mapping each target covariate name to its covariance-estimation mode, forwarded to :func:`reconstruction_loss_noise_model`, defaults to `None`.
    :type pert_cov_estimation_modes: class:`dict[str, str] | None`

    :param add_loss: Currently unused by this function.
    :type add_loss: class:`bool`

    :param allow_noise_model_to_be_none: Forwarded to :func:`reconstruction_loss_noise_model` for every covariate, defaults to `True`.
    :type allow_noise_model_to_be_none: class:`bool`

    :param loss_fn_kwargs: Optional dictionary of keyword arguments forwarded to :func:`reconstruction_loss_noise_model` for every covariate, defaults to `None` in which case an empty dictionary is used.
    :type loss_fn_kwargs: class:`dict[str, Any] | None`

    :return: A tuple `(loss, loss_dict)` where `loss` is the sum of the per-covariate reconstruction losses and `loss_dict` maps `"{covariate}_{LossFields.PERTURBATION_LOSS}"` to the corresponding detached loss value.
    :rtype: class:`tuple[torch.Tensor, dict[str, torch.Tensor]]`
    """
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
