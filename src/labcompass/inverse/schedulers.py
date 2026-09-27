import abc
import math

import torch


class LambdaScheduler(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        t,
        *args
    ):
        raise NotImplementedError


class ConstantLambdaScheduler(LambdaScheduler):
    def __init__(
        self,
        lmax: float = 1.0,
        **kwargs
    ):
        self.lmax = lmax

    def __call__(self, t, *args, **kwargs):
        return self.lmax * torch.ones((*t.shape[:-1], 1), device=t.device)


class ExponentialDecayScheduler(LambdaScheduler):
    def __init__(
        self,
        lmin: float = 0.0,
        lmax: float = 1.0,
        gamma: float = 1.0,
        **kwargs,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.gamma = gamma

    def __call__(self, t, *args, **kwargs):
        b = (self.lmin - self.lmax)/(math.exp(-self.gamma)-1)
        a = self.lmax - b
        return a + b*torch.exp(-self.gamma*t)


class LinearDecayScheduler(LambdaScheduler):
    def __init__(
        self,
        lmin: float = 0.0,
        lmax: float = 1.0,
        **kwargs,
    ):
        self.lmin = lmin
        self.lmax = lmax

    def __call__(self, t, *args, **kwargs):
        return self.lmax - (self.lmax - self.lmin)*t


class ReciprocalDecayScheduler(LambdaScheduler):
    def __init__(
        self,
        lmax: float = 1.0,
        **kwargs,
    ):
        self.lmax = lmax

    def __call__(self, t, *args, **kwargs):
        return torch.clamp(
            (1 - t)/t, min=0.0, max=self.lmax
        )

schedulers_dict = {
    "constant": ConstantLambdaScheduler,
    "exp-decay": ExponentialDecayScheduler,
    "reciprocal": ReciprocalDecayScheduler,
    "lin-decay": LinearDecayScheduler,
}
