import abc
from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn

__all__ = [
    "Transform",
    "InvertibleTransform",
    "ComposedTransform",
    "Standardizer",
]


class Transform(abc.ABC, nn.Module):
    """"""

    __is_invertible: bool = False

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__()

    @abc.abstractmethod
    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """"""
        raise NotImplementedError

    @property
    def is_invertible(
        self,
    ) -> bool:
        """"""
        return self.__is_invertible


class InvertibleTransform(Transform):
    """"""

    __is_invertible: bool = True

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """"""
        super().__init__()

    @abc.abstractmethod
    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """"""
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(
        self,
        input: Tensor,
    ) -> Tensor:
        """"""
        raise NotImplementedError


class ComposedTransform(Transform):
    """"""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """"""
        super().__init__()
        self.transforms = nn.Sequential(*transforms)

    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """"""

    def inverse_transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """"""


class Standardizer(InvertibleTransform):
    """"""

    def __init__(
        self, mean: Tensor, cov: Tensor, device_id: Literal["cuda", "cpu"] = "cuda", synch_devices: bool = True
    ) -> None:
        """"""
        super().__init__()
        self.device_id = device_id
        self.synch_devices = synch_devices
        self.device = torch.device(self.device_id)
        self.mean = mean.to(self.device)
        self.cov = cov.to(self.device)

    def transform(self, input_tensor: Tensor) -> Tensor:
        """"""
        # copying the parameters
        mean = self.mean.clone()
        cov = self.cov.clone()
        # optionally synching the devices
        if self.synch_devices:
            mean = mean.to(input_tensor.device)
            cov = cov.to(input_tensor.device)
        return (input_tensor - mean) / cov

    def inverse_transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """"""
        # copying the parameters
        mean = self.mean.clone()
        cov = self.cov.clone()
        # optionally synching the devices
        if self.synch_devices:
            mean = mean.to(input_tensor.device)
            cov = cov.to(input_tensor.device)
        return input_tensor * cov + mean
