from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor, linspace, nn
from torchdiffeq import odeint

__all__ = ["ODESolver"]


class VF(nn.Module):
    """
    A class representing a Vector Field (VF) that computes the drift (rate of change) 
    of a process at a given time `t` and state `xt`. This class is often used in
    stochastic processes and differential equations, where the drift function is used 
    to model the evolution of a state.

    Attributes:
        drift_fn (Callable): A function that computes the drift (rate of change) 
                              at a given time `t` and state `xt`.

    Methods:
        __init__: Initializes the VF object with a drift function.
        forward: Computes the drift at a given time `t` and state `xt` using the drift function.
    """

    def __init__(
        self,
        drift_fn: Callable,
    ) -> None:
        """
        Initializes the Vector Field (VF) with a specified drift function.

        Args:
            drift_fn (Callable): A function that computes the drift (rate of change) 
                                  at a given time `t` and state `xt`. 
                                  The function should take two arguments: `t` and `xt`.
        """
        super().__init__()
        self.drift_fn = drift_fn

    def forward(
        self,
        t: Tensor,
        xt: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Computes the drift (rate of change) at a given time `t` and state `xt`.

        The time `t` is repeated to match the batch size of `xt` before passing both 
        `t` and `xt` to the drift function. The function then computes the drift (rate 
        of change) at each time point and state using the drift function provided during 
        initialization.

        Args:
            t (Tensor): A tensor representing time.
            xt (Tensor): A tensor representing the state of the system at time `t`.
            *args: Additional arguments passed to the drift function.
            **kwargs: Additional keyword arguments passed to the drift function.

        Returns:
            Tensor: The drift (rate of change) at each time step and state, as computed 
                    by the drift function.
        """
        t = t.repeat(*xt.shape[:-1])
        return self.drift_fn(t, xt)


class ODESolver:
    """
    A class for solving ordinary differential equations (ODEs) or stochastic differential equations (SDEs)
    using neural network-based solvers. This class supports both ODEs and SDEs with various configurations for
    drift and diffusion terms. 
    """

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
        """
        Initializes the ODESolver with the provided drift function, solver arguments, and SDE configuration.

        Args:
            drift_fn (Callable[[Tensor, Tensor], Tensor]): The drift function defining the rate of change.
            num_time_steps (int, optional): The number of time steps to use for integration. Defaults to 500.
            solver_kwargs (dict, optional): Additional solver arguments (e.g., solver type, tolerance). 
                                            Defaults to None (sets default values).
            gamma_fn (Callable[[Tensor, Tensor], Tensor], optional): A function for the diffusion term (SDE). 
                                                                    Defaults to None (for ODE).
            sde_type (str, optional): The type of SDE ('ito' or 'stratonovich'). Defaults to 'ito'.
            noise_type (str, optional): The type of noise ('scalar', 'additive', 'diagonal', 'general'). 
                                        Defaults to 'diagonal'.
            device_id (str, optional): The computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
        """
        # default values for the solver arguments
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
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
        self.time = linspace(0.0, 1.0, self.num_time_steps).to(self.device)

    def integrate(
        self,
        source: Tensor,
        return_trajectory: bool = False,
    ) -> Tensor:
        """
        Solves the ODE or SDE system starting from the given initial state `source`. Optionally returns the 
        trajectory over time or just the final state.

        Args:
            source (Tensor): The initial state of the system at time t=0.
            return_trajectory (bool, optional): Whether to return the entire trajectory over time or just 
                                                 the final state. Defaults to False (returns final state).

        Returns:
            Tensor: The final state at the last time step, or the full trajectory if `return_trajectory=True`.
        """
        vf = VF(self.drift_fn)
        trajectory = odeint(vf, source, self.time, **self.solver_kwargs)
        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]
