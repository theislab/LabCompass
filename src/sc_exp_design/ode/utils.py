import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from sc_exp_design.networks.velocity_field import NeuralVelocityField 
from sc_exp_design.ode.solvers import ODESolver
from sc_exp_design.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = ["push_forward"]


def push_forward(
    velocity_field: NeuralVelocityField,
    source: TensorLike | None,
    condition: dict[str, TensorLike] | None,
    generate_from_noise: bool,
    noise_distribution: Callable[[Sequence[int]], TensorLike],
    num_time_steps: int,
    solver_kwargs: dict[str, Any],
    device_id: Literal["cuda", "cpu"],
    return_trajectory: bool = False,
    no_grad: bool = True,
    num_samples: int | None = None,
    batch_size: int | None = None,
) -> TensorLike:
    """"""
    # initializing device
    device = torch.device(device_id)

    # handling batch size
    if source is not None:
        batch_size = source.shape[: -1]
    if batch_size is None:
        batch_size = (1, )
    
    if isinstance(batch_size, int):
        batch_size = (batch_size,)
    
    # handling number of samples
    if num_samples is not None:
        if not generate_from_noise:
            msg = f""
            logger.warning(msg)
            num_samples = 1
    else:
        num_samples = 1
    msg = f""
    assert isinstance(num_samples, int), msg
    if condition is not None:
        condition = {
            condition_covariate: condition_data.repeat(num_samples, *(1 for _ in condition_data.shape)).squeeze()
            for condition_covariate, condition_data in condition.items()
        }
    if source is not None:
        source = source.repeat(num_samples, *(1 for _ in source.shape)).squeeze()

    # handling latent state
    initial_state = source
    if generate_from_noise:
        initial_state = noise_distribution((num_samples, *batch_size, velocity_field.config.flow_dim)).squeeze().to(device)
    msg = f""
    assert initial_state is not None, msg

    # defining velocity function
    vf = velocity_field.get_vf_fn(condition, source=source)
    # initializing the sampler clss
    ode_solver = ODESolver(
        vf,
        num_time_steps=num_time_steps,
        solver_kwargs=solver_kwargs,
        device_id=device_id,
    )
    if no_grad:
        with torch.no_grad():
            predictions = ode_solver.integrate(initial_state, return_trajectory=return_trajectory)
    else:
        predictions = ode_solver.integrate(initial_state, return_trajectory=return_trajectory)
    return predictions
