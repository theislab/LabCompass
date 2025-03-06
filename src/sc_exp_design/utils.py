import random

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "match_shapes",
    "set_reproducibility",
]


def match_shapes(
    t: Tensor | float,
    source: Tensor,
    target: Tensor,
) -> Tensor:
    """"""
    assert source.shape == target.shape
    if isinstance(t, float):
        t = torch.ones_like(source) * t
    if isinstance(t, Tensor):
        if t.shape == source.shape:
            return t
        assert t.shape[0] == source.shape[0], ""
        assert t.ndim == 1, ""
        dims_to_repeat = (1, *source.shape[1:])[::-1]
        t = t.repeat(*dims_to_repeat).T
    return t


def set_reproducibility(random_seed: int) -> None:
    """"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
