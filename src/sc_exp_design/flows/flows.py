import abc

import numpy as np
import torch
from torch import Tensor, nn

from sc_exp_design.utils import match_shapes

__all__ = [
    "BaseFlow",
    "ConstantNoiseFlow",
    "RectifiedFlow",
    "VariancePreservingFlow",
    "EncodingDecodingFlow",
]

class BaseFlow(abc.ABC, nn.Module):
    """
    Abstract base class for modeling flow-based transformations in a probabilistic setting.
    """

    def __init__(
        self,
        random_seed: int,
    ) -> None:
        """
        Initializes the BaseFlow class.

        :param random_seed: Random seed for reproducibility.
        :type random_seed: class: `int`
        """
        super().__init__()
        self.random_seed = random_seed

    @abc.abstractmethod
    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the mean function \( \mu_t \) for the flow transformation at time `t`.

        :param t: Time variable tensor.
        :type t: class: `torch.Tensor`

        :param source: Source distribution tensor.
        :type source: class: `torch.Tensor`

        :param target: Target distribution tensor.
        :type target: class: `torch.Tensor`

        :return: Computed mean function at time `t`.
        :rtype: class: `torch.Tensor`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """
        Computes the standard deviation function \( \sigma_t \) at time `t`.

        :param t: Time variable tensor.
        :type t: class: `torch.Tensor`

        :return: Computed standard deviation function at time `t`.
        :rtype: class: `torch.Tensor`
        """
        raise NotImplementedError

    def compute_x_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the latent representation \( x_t \) at time `t`.

        :param t: Time variable tensor.
        :type t: class: `torch.Tensor`

        :param source: Source distribution tensor.
        :type source: class: `torch.Tensor`

        :param target: Target distribution tensor.
        :type target: class: `torch.Tensor`

        :return: Computed latent representation at time `t`.
        :rtype: class: `torch.Tensor`
        """
        # handling shapes
        msg = f"`source` and `target` are supposed to have the same shape, found  {source.shape=} and {target.shape=}"
        assert source.shape == target.shape, msg
        t = match_shapes(t, source)
        # computing coefficients and ode noise
        mu_t = self.compute_mu_t(t, source, target)
        sigma_t = self.compute_sigma_t(t)
        epsilon = torch.randn_like(source)
        return mu_t + sigma_t * epsilon

    def compute_score_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor,
    ) -> Tensor:
        """
        Computes the score function \( \nabla log p(x_t) \) at time `t`.

        :param t: Time variable tensor.
        :type t: class: `torch.Tensor`

        :param source: Source distribution tensor.
        :type source: class: `torch.Tensor`

        :param target: Target distribution tensor.
        :type target: class: `torch.Tensor`

        :param xt: Latent representation tensor at time `t`.
        :type xt: class: `torch.Tensor`

        :return: Computed score function at time `t`.
        :rtype: class: `torch.Tensor`
        """
        # handling shapes
        msg = f"`source` and `target` are supposed to have the same shape, found  {source.shape=} and {target.shape=}"
        assert source.shape == target.shape, msg
        t = match_shapes(t, source)
        # computing coefficients and ode noise
        mu_t = self.compute_mu_t(t, source, target)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = sigma_t**2
        return -(1 / sigma_t) * (xt - mu_t)

    @abc.abstractmethod
    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor | None = None,
    ) -> Tensor:
        """
        Computes the drift function \( u_t \) at time `t`.

        :param t: Time variable tensor.
        :type t: class: `torch.Tensor`

        :param source: Source distribution tensor.
        :type source: class: `torch.Tensor`

        :param target: Target distribution tensor.
        :type target: class: `torch.Tensor`

        :param xt: Optional latent representation tensor at time `t`.
        :type xt: class: `torch.Tensor | None`

        :return: Computed drift function at time `t`.
        :rtype: class: `torch.Tensor`
        """
        raise NotImplementedError


class ConstantNoiseFlow(BaseFlow):
    """
    A flow model with constant noise level throughout the trajectory.
    """

    def __init__(
        self,
        sigma: float = 1.0,
        random_seed: int | None = 42,
    ) -> None:
        """
        :param sigma: Standard deviation of the noise, defaults to `1.0`.
        :type sigma: class: `float`

        :param random_seed: Random seed for reproducibility, defaults to `42`.
        :type random_seed: class: `int | None`
        """
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the mean trajectory between source and target.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :return: Computed mean trajectory.
        :rtype: class: `torch.Tensor`
        """
        return t * target + (1 - t) * source

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """
        Computes the constant noise level.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :return: Noise level tensor.
        :rtype: class: `torch.Tensor`
        """
        return torch.ones_like(t) * self.sigma

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor | None = None,
    ) -> Tensor:
        """
        Computes the velocity field.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :param xt: Optional input tensor at time t (not used, here only for compatibility).
        :type xt: class: `torch.Tensor | None`

        :return: Computed velocity field.
        :rtype: class: `torch.Tensor`
        """
        return target - source


class RectifiedFlow(ConstantNoiseFlow):
    """
    A rectified flow model with zero noise.

    Inherits from :class: `ConstantNoiseFlow` and sets automatically the :attr:`ConstantNoiseFlow.sigma` attribute to `0`
    for deterministic interpolation.
    """

    def __init__(
        self,
        random_seed: float | None = 42,
    ) -> None:
        """
        :param random_seed: Random seed for reproducibility, defaults to `42`.
        :type random_seed: class: `int | None`
        """
        super().__init__(sigma=0.0, random_seed=random_seed)


class VariancePreservingFlow(BaseFlow):
    """
    A flow model where variance is preserved over time.
    """

    def __init__(
        self,
        sigma: float = 0e-0,
        random_seed: float | None = 42,
    ) -> None:
        """
        :param sigma: Standard deviation of the noise, defaults to `0.0`.
        :type sigma: class: `float`

        :param random_seed: Random seed for reproducibility, defaults to `42`.
        :type random_seed: class: `int | None`
        """
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the mean trajectory using a sinusoidal function.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :return: Computed mean trajectory.
        :rtype: class: `torch.Tensor`
        """
        return torch.cos(0.5 * np.pi * t) * source + torch.sin(0.5 * np.pi * t) * target

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """
        Computes the time-dependent noise level.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :return: Noise level tensor.
        :rtype: class: `torch.Tensor`
        """
        return torch.sqrt(self.sigma * t * (1 - t))

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor | None = None,
    ) -> Tensor:
        """
        Computes the velocity field.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :param xt: Optional input tensor at time t, defaults to `None`.
        :type xt: class: `torch.Tensor | None`

        :return: Computed velocity field.
        :rtype: class: `torch.Tensor`
        """
        # handling shapes
        msg = f"`source` and `target` are supposed to have the same shape, found  {source.shape=} and {target.shape=}"
        assert source.shape == target.shape, msg
        t = match_shapes(t, source)
        # computing coefficients and ode noise
        mu = self.compute_mu_t(t, source, target)
        sigma = self.compute_sigma_t(t)
        sigma_dot = self.sigma * t / torch.sqrt(self.sigma * t * (1 - t))
        mu_dot = 0.5 * np.pi * target * torch.cos(0.5 * np.pi * t) - 0.5 * np.pi * source * torch.sin(0.5 * np.pi * t)
        if self.sigma == 0:
            return mu_dot
        return mu_dot + (sigma_dot / sigma) * (xt - mu)


class EncodingDecodingFlow(BaseFlow):
    """
    A flow model incorporating an encoding-decoding mechanism.
    """

    def __init__(
        self,
        sigma: float = 1e-0,
        random_seed: float | None = 42,
    ) -> None:
        """
        :param sigma: Standard deviation of the noise, defaults to `0.0`.
        :type sigma: class: `float`

        :param random_seed: Random seed for reproducibility, defaults to `42`.
        :type random_seed: class: `int | None`
        """
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """
        Computes the mean trajectory with an encoding-decoding scheme.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :return: Computed mean trajectory.
        :rtype: class: `torch.Tensor`
        """
        alpha = torch.cos(np.pi * t) ** 2
        beta = torch.sin(np.pi * t) ** 2
        mask = torch.where(t < 0.5, 1.0, 0.0)
        return alpha * source * mask + beta * target * (1 - mask)

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """
        Computes the noise level with a sinusoidal pattern.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :return: Noise level tensor.
        :rtype: class: `torch.Tensor`
        """
        return self.sigma * torch.sin(np.pi * t) ** 2

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor,
    ) -> Tensor:
        """
        Computes the velocity field for encoding-decoding dynamics.

        :param t: Time step tensor.
        :type t: class: `torch.Tensor`

        :param source: Source tensor.
        :type source: class: `torch.Tensor`

        :param target: Target tensor.
        :type target: class: `torch.Tensor`

        :param xt: Optional input tensor at time t, defaults to `None`.
        :type xt: class: `torch.Tensor | None`

        :return: Computed velocity field.
        :rtype: class: `torch.Tensor`
        """
        # handling shapes
        msg = f"`source` and `target` are supposed to have the same shape, found  {source.shape=} and {target.shape=}"
        assert source.shape == target.shape, msg
        t = match_shapes(t, source)
        # computing coefficients and ode noise
        sigma = self.compute_sigma_t(t)
        mu = self.compute_mu_t(t, source, target)
        alpha_dot = -2 * np.pi * torch.sin(np.pi * t) * torch.cos(np.pi * t)
        beta_dot = 2 * np.pi * torch.sin(np.pi * t) * torch.cos(np.pi * t)
        sigma_dot = 2 * np.pi * torch.sin(np.pi * t) * torch.cos(np.pi * t) * self.sigma
        mask = torch.where(t < 0.5, 1.0, 0.0)
        mu_dot = alpha_dot * mask * source + beta_dot * (1 - mask) * target
        if self.sigma == 0:
            return mu_dot
        return mu_dot + (sigma_dot / sigma)(xt - mu)
