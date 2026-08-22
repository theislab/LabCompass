from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.optim import Optimizer

__all__ = ["LangevinOptimizer"]


class LangevinOptimizer(Optimizer):
    """Initializes the :class:`LangevinOptimizer`, a :class:`torch.optim.Optimizer` implementing a Langevin
    dynamics update rule, used internally by :class:`labcompass.models.inverse.InverseModel` when its
    `inverse_method` is `"langevin"` to draw samples from the posterior distribution over the perturbation
    covariates.

    :param params: The parameters to optimize (or dictionaries defining parameter groups), following the standard
        :class:`torch.optim.Optimizer` interface.
    :type params: class:`Sequence[torch.nn.Parameter]`

    :param eta: The step size of the Langevin update, controlling both the gradient step and the scale of the
        injected noise (`sqrt(eta)`), defaults to `1e-1`.
    :type eta: class:`float`

    :param noise_scale: An additional multiplicative factor applied to the injected Gaussian noise, on top of
        `sqrt(eta)`, defaults to `1e-1`.
    :type noise_scale: class:`float`
    """

    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        eta: float = 1e-1,
        noise_scale: float = 1e-1,
    ) -> None:
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
        state: dict[str, Any],
    ) -> None:
        """"""
        super().__setstate__(state)

    def step(
        self,
        closure: Callable | None = None,
    ) -> torch.Tensor:
        """Performs a single Langevin dynamics optimization step.

        For every parameter with a non-`None` gradient, the update
        `p <- p - (eta/2 * grad + sqrt(eta) * noise_scale * z)` is applied in place, where `z` is standard
        Gaussian noise sampled independently for each parameter.

        :param closure: (Optional) a closure that reevaluates the model and returns the loss, called with
            gradient tracking enabled before the update is applied, defaults to `None`.
        :type closure: class:`Callable | None`

        :return: The loss returned by `closure`, or `None` if no closure was provided.
        :rtype: class:`torch.Tensor`
        """
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
                # get gradients
                d_p = p.grad.data
                # scale gradients
                grad = d_p*eta/2

                # Add Gaussian noise scaled by sqrt(eta)
                noise = torch.randn_like(p.data) * sqrt_eta * noise_scale
                
                # computing update term
                update = grad + noise

                # Langevin update step
                p.data.add_(-update)
        
        return loss
