import logging
from collections.abc import Callable, Sequence
from typing import Any, Literal

import torch

from labcompass.networks.velocity_field import NeuralVelocityField 
from labcompass.ode.solvers import ODESolver
from labcompass.types import TensorLike

logger = logging.getLogger(__name__)

__all__ = ["get_initial_state_and_condition", "push_forward"]


def get_initial_state_and_condition(
    source,
    batch_size,
    num_samples,
    flow_dim,
    condition,
    noise_distribution,
    device_id,
    generate_from_noise,
):

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
            condition_covariate: condition_data.repeat(num_samples, *(1 for _ in condition_data.shape)).squeeze(dim=0)
            for condition_covariate, condition_data in condition.items()
        }
    if source is not None:
        source = source.repeat(num_samples, *(1 for _ in source.shape)).squeeze(dim=0)

    # handling latent state
    initial_state = source
    if initial_state is None:
        initial_state = noise_distribution((num_samples, *batch_size, flow_dim)).squeeze(dim=0).to(device)
    msg = f""
    assert initial_state is not None, msg
    return initial_state, condition

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
    cfg_guidance_strength: float = 1.0,
) -> TensorLike:
    """"""
    initial_state, condition = get_initial_state_and_condition(
        source,
        batch_size,
        num_samples,
        velocity_field.config.flow_dim,
        condition,
        noise_distribution,
        device_id,
        generate_from_noise,
    )
    # defining velocity function
    vf = velocity_field.get_vf_fn(condition, source=source, cfg_guidance_strength=cfg_guidance_strength)
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
