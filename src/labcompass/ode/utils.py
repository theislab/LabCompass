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
    """Builds the initial state and the (possibly repeated) conditioning tensors used to start the ODE integration.

    When `source` is provided, it is used as the initial state (repeated `num_samples` times along a new
    leading axis). Otherwise the initial state is drawn from `noise_distribution` with shape
    `(num_samples, *batch_size, flow_dim)`. In both cases, when `num_samples` is `1` the leading sample
    axis is squeezed away. Sampling more than one sample per observation (`num_samples > 1`) is only
    supported when `generate_from_noise` is `True`; otherwise `num_samples` is forced back to `1`.

    :param source: The initial/source state tensor, or `None` if it should be sampled from `noise_distribution`.
    :type source: class:`TensorLike | None`

    :param batch_size: The desired batch shape used to sample the initial state when `source` is `None`.
        Ignored (and overridden by `source.shape[:-1]`) when `source` is provided. When `None`, defaults to `(1,)`.
    :type batch_size: class:`int | Sequence[int] | None`

    :param num_samples: The number of samples to generate per observation. Forced to `1` when `generate_from_noise` is `False`,
        and defaults to `1` when `None`.
    :type num_samples: class:`int | None`

    :param flow_dim: The dimensionality of the flow (i.e.: the size of the last axis of the sampled state).
    :type flow_dim: class:`int`

    :param condition: Dictionary mapping each conditioning covariate to its tensor, or `None` if no conditioning is used.
        Each tensor is repeated `num_samples` times along a new leading axis (squeezed away when `num_samples` is `1`).
    :type condition: class:`dict[str, TensorLike] | None`

    :param noise_distribution: Function used to sample the initial state when `source` is `None`.
    :type noise_distribution: class:`Callable[[Sequence[int]], TensorLike]`

    :param device_id: The identifier for the device onto which the sampled initial state is moved.
    :type device_id: class:`Literal["cuda", "cpu"]`

    :param generate_from_noise: Whether the model generates from noise, which is required for `num_samples` to be greater than `1`.
    :type generate_from_noise: class:`bool`

    :return: A tuple with the initial state tensor and the (possibly repeated) condition dictionary.
    :rtype: class:`tuple[TensorLike, dict[str, TensorLike] | None]`
    """
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
            msg = ""
            logger.warning(msg)
            num_samples = 1
    else:
        num_samples = 1
    msg = ""
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
    msg = ""
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
    """Generates predictions by integrating the learnt velocity field's dynamics from an initial state.

    Builds the initial state and conditioning via :func:`get_initial_state_and_condition`, wraps `velocity_field`
    into a drift function via :method:`velocity_field.get_vf_fn`, and integrates it over time using an :class:`ODESolver`.

    :param velocity_field: The trained velocity field network defining the drift of the ODE.
    :type velocity_field: class:`NeuralVelocityField`

    :param source: The initial/source state tensor, or `None` if it should be sampled from `noise_distribution`.
    :type source: class:`TensorLike | None`

    :param condition: Dictionary mapping each conditioning covariate to its tensor, or `None` if no conditioning is used.
    :type condition: class:`dict[str, TensorLike] | None`

    :param generate_from_noise: Whether the model generates from noise, which is required for `num_samples` to be greater than `1`.
    :type generate_from_noise: class:`bool`

    :param noise_distribution: Function used to sample the initial state when `source` is `None`.
    :type noise_distribution: class:`Callable[[Sequence[int]], TensorLike]`

    :param num_time_steps: Number of time steps over which to integrate the dynamics.
    :type num_time_steps: class:`int`

    :param solver_kwargs: Dictionary containing the keyword arguments used to initialize the :class:`ODESolver`.
    :type solver_kwargs: class:`dict[str, Any]`

    :param device_id: The identifier for the device where to perform the computations.
    :type device_id: class:`Literal["cuda", "cpu"]`

    :param return_trajectory: Whether to return the full trajectory at each discretization point instead of just the final state, defaults to `False`.
    :type return_trajectory: class:`bool`

    :param no_grad: Whether to integrate the dynamics within a :func:`torch.no_grad` context, defaults to `True`.
    :type no_grad: class:`bool`

    :param num_samples: The number of samples to generate per observation.
        Only used when `generate_from_noise` is `True`, defaults to `None` in which case a single sample is generated.
    :type num_samples: class:`int | None`

    :param batch_size: The desired batch shape used to sample the initial state when neither `source` nor `condition` is provided, defaults to `None`.
    :type batch_size: class:`int | None`

    :param cfg_guidance_strength: Strength of the guidance term used when computing the velocity field, defaults to `1.0`.
    :type cfg_guidance_strength: class:`float`

    :return: The final state at the last time step, or the full trajectory if `return_trajectory` is `True`.
    :rtype: class:`TensorLike`
    """
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
