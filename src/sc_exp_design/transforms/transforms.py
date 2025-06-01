import abc
from collections.abc import Sequence, Callable
from typing import Literal

import torch
from torch import Tensor, nn

import numpy as np

from sc_exp_design.types import TensorLike
__all__ = [
    "Transform",
    "InvertibleTransform",
    "ComposedTransform",
    "Standardizer",
    "VAETransform",
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





class VAETransform(abc.ABC, nn.Module):
    """"""

    __is_invertible: bool = False

    def __init__(
        self,
        vae: Callable[[TensorLike], TensorLike],
        **kwargs,
    ) -> None:
        """"""
        super().__init__()

        self.vae = vae

    #@abc.abstractmethod
    def transform(
        self,
        input_tensor: TensorLike,
        condition: TensorLike = None
    ) -> Tensor:
        """"""
        return self.vae.decode_latent_samples(input_tensor, cat_values=np.array(["R0"] * input_tensor.shape[0]), map_cat_values=True)

    def forward(
        self,
        input_tensor: TensorLike,
        condition: TensorLike = None
    ) -> Tensor:
        """"""
        return self.transform(input_tensor, condition)
    
    @property
    def is_invertible(
        self,
    ) -> bool:
        """"""
        return self.__is_invertible