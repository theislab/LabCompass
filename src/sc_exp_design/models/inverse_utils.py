from collections.abc import Callable, Sequence

import torch
from torch.optim import Optimizer

__all__ = ["LangevinOptimizer"]


class LangevinOptimizer(Optimizer):
    """"""

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        eta: float = 1e-1,
        noise_scale: float = 1e-1,
    ) -> None:
        """"""
        sqrt_eta = torch.sqrt(torch.tensor(eta))
        defaults = dict(
            eta=eta,
            sqrt_eta=sqrt_eta,
            noise_scale=noise_scale
        )

        super().__init__(
            params,
            defaults,
        )

    def __setstate__(
        self,
        state: dict, # what type is this??
    ) -> None:
        """"""
        super().__setstate__(state)


    def step(
        self,
        closure: Callable | None = None,
    ) -> torch.Tensor:
        """"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eta = group["eta"]
            sqrt_eta = group["sqrt_eta"]
            noise_scale = group["noise_scale"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                d_p = p.grad.data
                
                # Add Gaussian noise scaled by sqrt(eta)
                noise = -torch.randn_like(p.data) * sqrt_eta * noise_scale
                
                # Langevin update step
                p.data.add_(-d_p, alpha=eta/2).add_(noise)

        return loss
