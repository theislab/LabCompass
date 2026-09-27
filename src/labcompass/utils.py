import logging
import math
import random
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

__all__ = [
    "match_shapes",
    "set_reproducibility",
    "sinusoidal_time_features"
]


def match_shapes(
    input: float | Tensor,
    target: Tensor,
) -> Tensor:
    """
    :param `input`: The input which we have to broadcast to match `target` dimension.
        Could either be a `float` or a `torch.Tensor`.
        When `isinstance(input, Tensor)`, `input` will need to satisfy one of the following conditions:
            * Have only one elemen, `input.numel() == 1`.
            * Be a 1-dimensional tensor with shape `input.shape == (target.shape[0], )`.
            * Be a 2-dimensional tensor with a tailing singleton axis `input.shape == (target.shape[0], 1)`.
    :type `input`: `float | Tensor`

    :param `target`: The target tensor whose shape will be matched. It only supports 2-dimensional tensors.
    :type `target`: `Tensor`
    """
    # sanity check on inputs
    msg = f"`input` must be either a `float` or `Tensor`, found {type(input)}"
    assert isinstance(input, float | Tensor), msg

    msg = f"`target` must be either a `Tensor`, found {type(target)}"
    assert isinstance(target, Tensor), msg

    msg = f"`target` only supports 2 dimensional tensors, found {target.ndim}-dimensional tensor instead."
    assert (target.ndim == 2), msg

    # input is float
    if isinstance(input, float):
        return torch.ones((target.shape[0], 1), device=target.device)*input

    # input is tensor
    if isinstance(input, Tensor):
        # case 1 (1-element tensor): extract float value then recursive call
        # note: already handles the case when input.ndim == 1 but we only have one value
        if input.numel() == 1:
            input = input.item()
            return match_shapes(input, target)

        # case 2 (1-dimensional tensor with same shape): simply unsqueeze last dimension
        if input.ndim == 1:
            msg = f"When `input` is a `torch.tensor` with `input.ndim == 1`, `input` and `target` should share the same batch size, found {input.shape[0]=} and {target.shape[0]=}"
            assert input.shape[0] == target.shape[0], msg

            return torch.unsqueeze(input, dim=1)

        # case 3 (2-dimensional tensor with dummy trailing dimension): keep unchanges
        if input.ndim == 2:
            msg = f"When `input` is a `torch.tensor` with `input.ndim == 2`, `input` and `target` should share the same batch size, found {input.shape[0]=} and {target.shape[0]=}"
            assert input.shape[0] == target.shape[0], msg

            msg = f"When `input` is a `torch.tensor` with `input.ndim == 2`, the second dimension of `input` should be 1, found {input.shape[1]=}"
            assert input.shape[1] == 1, msg

            return input

        # raise value error if the cases are not matched
        raise ValueError


def set_reproducibility(random_seed: int) -> None:
    """Seeds `torch`, `random`, and `numpy` with `random_seed` and configures cuDNN backend flags for reproducibility.

    Sets `torch.backends.cudnn.deterministic` and `torch.backends.cudnn.benchmark` to `True`, and seeds
    :func:`torch.manual_seed`, :func:`random.seed`, and :func:`numpy.random.seed` with `random_seed`.

    :param random_seed: The seed value used to seed all random number generators.
    :type random_seed: class:`int`
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)


def get_conditions_to_pool(
    layers_before_pooling: dict[str, Any],
    covariates_not_pooled: Sequence[str] | None,
) -> Sequence[str]:
    """Determine which covariate keys of `layers_before_pooling` should be pooled together.

    Every key of `layers_before_pooling` is returned except those listed in `covariates_not_pooled`.

    :param layers_before_pooling: Mapping from covariate name to its pre-pooling representation/layer;
        only its keys are used to determine the covariates to pool.
    :type layers_before_pooling: class:`dict[str, Any]`

    :param covariates_not_pooled: Sequence of covariate names to exclude from pooling. When `None`,
        all keys of `layers_before_pooling` are returned.
    :type covariates_not_pooled: class:`Sequence[str] | None`

    :return: The covariate names to be pooled.
    :rtype: class:`Sequence[str]`
    """
    if covariates_not_pooled is not None:
        covariates_to_pool = [
            covariate
            for covariate in layers_before_pooling.keys()
            if covariate not in covariates_not_pooled
        ]
    else:
        covariates_to_pool = list(layers_before_pooling.keys())
    return covariates_to_pool


def sinusoidal_time_features(t: torch.Tensor,
                             num_freqs: int = 128,
                             max_period: int = 10000):
    """Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if len(t.shape)==1:
        t = t.unsqueeze(1)

    half = num_freqs // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0,
                       end=half,
                       dtype=torch.float32,
                       device=t.device)
        / half
    )
    args = t.float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if num_freqs % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def coerce_string_to_sequence(
    identifiers: Sequence[str] | str | None,
    allow_only_one_element: bool = False,
    allow_none: bool = False,
) -> Sequence[str]:
    """Configures the covariates metadata and performs some additional sanity checks

    :param identifiers: Sequence of covariate identifiers for the current perturbation.
    :type identifiers: class: `Sequence[str] | str`

    :param adata_field_key: Key indicating the field in the annotated data object
        where such identifiers are to be found.
    :type adata_field_key: class `Literal["uns", "obsm"]`

    :param allow_only_one_element: Whether to allow for only one identifier to be passed.
        This is needed for all the perturbations is :attr: `self.perturbations_in_obsm`.
    :type allow_only_one_element: class: `bool`
    """
    # when we allow none we return an empty sequence
    if identifiers is None:
        if allow_none:
            return ()
        msg = f"When {allow_none=}, an identifier should be passed. Found `None`."
        raise ValueError(msg)

    # when only one identifier is passed create a sequence with only one element
    if isinstance(identifiers, str):
        msg = f"Only one element provided in {identifiers=}. Setting it to a sequence."
        logger.info(msg)
        identifiers = (identifiers, )

    # optionally check that we only have one identifier
    if allow_only_one_element:
        if isinstance(identifiers, Sequence):
            if len(identifiers) != 1:
                msg = "When a perturbation is in .obsm, there should be only one representation."
                raise ValueError(identifiers)

    # checking that the representations are of the correct type
    if not isinstance(identifiers, Sequence):
        msg = f"{identifiers=} should be a string representation identifier, found {type(identifiers)}."
        raise TypeError(msg)
    return identifiers
