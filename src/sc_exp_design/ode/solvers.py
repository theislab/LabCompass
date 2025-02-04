from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor, linspace, nn
from torchdyn.core import NeuralODE
from torchsde import sdeint

__all__ = ["ODESolver"]


class VF(nn.Module):
    """"""

    def __init__(
        self,
        drift_fn: Callable,
    ) -> None:
        """"""
        super().__init__()
        self.drift_fn = drift_fn

    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """"""
        t = t.repeat(xt.shape[0])
        return self.drift_fn(t, xt)


class SDE(nn.Module):
    """"""

    def __init__(
        self,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        diffusion_fn: Callable[
            [
                Tensor,
            ],
            Tensor,
        ],
        sde_type: Literal["ito", "stratonovich"] = "ito",
        noise_type: Literal["scalar", "additive", "diagonal", "general"] = "diagonal",
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        super().__init__()
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.device_id = device_id

        self.device = torch.device(self.device_id)

    def f(
        self,
        t: Tensor,
        xt: Tensor,
    ) -> Tensor:
        """"""
        return self.drift_fn(t, xt)

    def g(self, t: Tensor, xt: Tensor) -> Tensor:
        """"""
        return self.diffusion_fn(t, xt)


class ODESolver:
    """"""

    def __init__(
        self,
        drift_fn: Callable[[Tensor, Tensor], Tensor],
        num_time_steps: int = 500,
        solver_kwargs: dict[str, Any] | None = None,
        gamma_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        sde_type: Literal["ito", "stratonovich"] = "ito",
        noise_type: Literal["scalar", "additive", "diagonal", "general"] = "diagonal",
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        # default values for the solver arguments
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("solver", "dopri5")
        solver_kwargs.setdefault("sensitivity", "adjoint")
        solver_kwargs.setdefault("atol", 1e-5)
        solver_kwargs.setdefault("rtol", 1e-5)

        self.drift_fn = drift_fn
        self.num_time_steps = num_time_steps
        self.solver_kwargs = solver_kwargs
        self.gamma_fn = gamma_fn
        self.sde_type = sde_type
        self.noise_type = noise_type
        self.device_id = device_id

        self.device = torch.device(self.device_id)
        self.time = linspace(0.0, 1.0, self.num_time_steps)

    def integrate(
        self,
        source: Tensor,
        return_trajectory: bool = False,
    ) -> Tensor:
        """"""
        vf = VF(self.drift_fn)
        if self.gamma_fn is not None:
            sde = SDE(
                vf,
                self.gamma_fn,
                sde_type=self.sde_type,
                noise_type=self.noise_type,
                device_id=self.device_id,
            )
            trajectory = sdeint(sde, source, self.time)
        else:
            ode = NeuralODE(vf, **self.solver_kwargs)
            trajectory = ode.trajectory(source, t_span=self.time)
        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
