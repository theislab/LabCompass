import random

import numpy as np
import math
import torch
from torch import Tensor

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
    """"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

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
