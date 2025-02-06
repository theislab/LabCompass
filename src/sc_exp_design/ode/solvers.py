from collections.abc import Callable
from typing import Any, Literal

import torch
from torch import Tensor, linspace, nn
from torchdyn.core import NeuralODE
from torchsde import sdeint

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
        t = t.repeat(xt.shape[0])
        return self.drift_fn(t, xt)


class SDE(nn.Module):
    """
    A class representing a Stochastic Differential Equation (SDE). This class computes the drift and diffusion 
    components of the SDE, which are used in simulations of stochastic processes. It can handle both Ito and 
    Stratonovich formulations of the SDE, as well as different noise types.
    """

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
        """
        Initializes the Stochastic Differential Equation (SDE) with the given drift and diffusion functions, 
        noise type, and device configuration.

        Args:
            drift_fn (Callable[[Tensor, Tensor], Tensor]): A function that computes the drift term of the SDE 
                                                           at a given time `t` and state `xt`.
            diffusion_fn (Callable[[Tensor], Tensor]): A function that computes the diffusion term (noise) at 
                                                       a given time `t` and state `xt`.
            sde_type (str, optional): Specifies the SDE formulation, either 'ito' or 'stratonovich'. Defaults to 'ito'.
            noise_type (str, optional): Specifies the type of noise, options include 'scalar', 'additive', 
                                         'diagonal', and 'general'. Defaults to 'diagonal'.
            device_id (str, optional): The device type, either 'cuda' or 'cpu'. Defaults to 'cuda'.
        """
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
        """
        Computes the drift term (rate of change) of the SDE at a given time `t` and state `xt`.

        Args:
            t (Tensor): The time variable.
            xt (Tensor): The state of the system at time `t`.

        Returns:
            Tensor: The drift term at the given time and state, computed by the drift function.
        """
        return self.drift_fn(t, xt)

    def g(self, t: Tensor, xt: Tensor) -> Tensor:
        """
        Computes the diffusion (noise) term of the SDE at a given time `t` and state `xt`.

        Args:
            t (Tensor): The time variable.
            xt (Tensor): The state of the system at time `t`.

        Returns:
            Tensor: The diffusion term (noise) at the given time and state, computed by the diffusion function.
        """
        return self.diffusion_fn(t, xt)


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
