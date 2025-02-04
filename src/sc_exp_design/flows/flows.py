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
    """"""

    def __init__(
        self,
        random_seed: int,
    ) -> None:
        """"""
        self.random_seed = random_seed

    @abc.abstractmethod
    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """"""
        raise NotImplementedError

    def compute_x_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """"""
        # matching shapes
        t = match_shapes(t, source, target)
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
        """"""
        # matching shapes
        t = match_shapes(t, source, target)
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
        """"""
        raise NotImplementedError


class ConstantNoiseFlow(BaseFlow):
    """"""

    def __init__(
        self,
        sigma: float = 1.0,
        random_seed: float | None = 42,
    ) -> None:
        """"""
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """"""
        return t * target + (1 - t) * source

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """"""
        return torch.ones_like(t) * self.sigma

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor | None = None,
    ) -> Tensor:
        """"""
        return target - source


class RectifiedFlow(ConstantNoiseFlow):
    """"""

    def __init__(
        self,
        random_seed: float | None = 42,
    ) -> None:
        """"""
        super().__init__(sigma=0.0, random_seed=random_seed)


class VariancePreservingFlow(BaseFlow):
    """"""

    def __init__(
        self,
        sigma: float = 0e-0,
        random_seed: float | None = 42,
    ) -> None:
        """"""
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """"""
        return torch.cos(0.5 * np.pi * t) * source + torch.sin(0.5 * np.pi * t) * target

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """"""
        return torch.sqrt(self.sigma * t * (1 - t))

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor | None = None,
    ) -> Tensor:
        """"""
        # matching shapes
        t = match_shapes(t, source, target)
        mu = self.compute_mu_t(t, source, target)
        sigma = self.compute_sigma_t(t)
        sigma_dot = self.sigma * t / torch.sqrt(self.sigma * t * (1 - t))
        mu_dot = 0.5 * np.pi * target * torch.cos(0.5 * np.pi * t) - 0.5 * np.pi * source * torch.sin(0.5 * np.pi * t)
        if self.sigma == 0:
            return mu_dot
        return mu_dot + (sigma_dot / sigma) * (xt - mu)


class EncodingDecodingFlow(BaseFlow):
    """"""

    def __init__(
        self,
        sigma: float = 1e-0,
        random_seed: float | None = 42,
    ) -> None:
        """"""
        super().__init__(random_seed=random_seed)
        self.sigma = sigma

    def compute_mu_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """"""
        alpha = torch.cos(np.pi * t) ** 2
        beta = torch.sin(np.pi * t) ** 2
        mask = torch.where(t < 0.5, 1.0, 0.0)
        return alpha * source * mask + beta * target * (1 - mask)

    def compute_sigma_t(
        self,
        t: Tensor,
    ) -> Tensor:
        """"""
        return self.sigma * torch.sin(np.pi * t) ** 2

    def compute_u_t(
        self,
        t: Tensor,
        source: Tensor,
        target: Tensor,
        xt: Tensor,
    ) -> Tensor:
        """"""
        t = match_shapes(t, source, target)
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
