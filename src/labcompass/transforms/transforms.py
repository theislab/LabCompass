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
    """Abstract base class for one-way transformations applied to model states, e.g. before feeding them into the model."""

    __is_invertible: bool = False

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the :class:`Transform` module. Accepts and ignores any positional or keyword arguments to support flexible subclass initialization."""
        super().__init__()

    @abc.abstractmethod
    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Applies the transformation to `input_tensor`.

        :param input_tensor: The tensor to transform.
        :type input_tensor: class:`torch.Tensor`

        :return: The transformed tensor.
        :rtype: class:`torch.Tensor`
        """
        raise NotImplementedError

    @property
    def is_invertible(
        self,
    ) -> bool:
        """Whether this transform also provides an :method:`inverse_transform` method.

        :rtype: class:`bool`
        """
        return self.__is_invertible


class InvertibleTransform(Transform):
    """Abstract base class for transformations that support both a forward :method:`transform` and an :method:`inverse_transform` mapping."""

    __is_invertible: bool = True

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """Initializes the :class:`InvertibleTransform` module. Accepts and ignores any positional or keyword arguments to support flexible subclass initialization."""
        super().__init__()

    @abc.abstractmethod
    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Applies the transformation to `input_tensor`.

        :param input_tensor: The tensor to transform.
        :type input_tensor: class:`torch.Tensor`

        :return: The transformed tensor.
        :rtype: class:`torch.Tensor`
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(
        self,
        input: Tensor,
    ) -> Tensor:
        """Applies the inverse of the transformation to `input`, mapping it back to the original space.

        :param input: The transformed tensor to invert.
        :type input: class:`torch.Tensor`

        :return: The tensor mapped back to the original space.
        :rtype: class:`torch.Tensor`
        """
        raise NotImplementedError


class ComposedTransform(Transform):
    """Composes a sequence of :class:`Transform` objects into a single transform, applied one after the other."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        """Initializes the :class:`ComposedTransform` by chaining a sequence of transforms.

        :param transforms: The ordered sequence of :class:`Transform` objects to compose.
        :type transforms: class:`Sequence[Transform]`
        """
        super().__init__()
        self.transforms = nn.Sequential(*transforms)

    def transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Applies the composed sequence of transforms to `input_tensor`, one after the other, in order.

        :param input_tensor: The tensor to transform.
        :type input_tensor: class:`torch.Tensor`

        :return: The tensor obtained by applying every transform in `self.transforms`, in order.
        :rtype: class:`torch.Tensor`
        """
        for t in self.transforms:
            input_tensor = t.transform(input_tensor)
        return input_tensor

    def inverse_transform(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Applies the inverse of the composed sequence of transforms to `input_tensor`, in reverse order.

        :param input_tensor: The transformed tensor to invert.
        :type input_tensor: class:`torch.Tensor`

        :return: The tensor obtained by applying the :method:`inverse_transform` of every transform in
            `self.transforms`, in reverse order.
        :rtype: class:`torch.Tensor`

        :raises AttributeError: If any transform in `self.transforms` does not itself define
            :method:`inverse_transform` (i.e. is not an :class:`InvertibleTransform`).
        """
        for t in reversed(list(self.transforms)):
            input_tensor = t.inverse_transform(input_tensor)
        return input_tensor


class Standardizer(InvertibleTransform):
    """Invertible transform that standardizes tensors by subtracting a fixed `mean` and dividing element-wise by a fixed scale tensor (`cov`)."""

    def __init__(
        self, mean: Tensor, cov: Tensor, device_id: Literal["cuda", "cpu"] = "cuda", synch_devices: bool = True
    ) -> None:
        """Initializes the :class:`Standardizer` transform.

        :param mean: Tensor of values to subtract from the input when standardizing (and to add back when inverting).
            Moved to `device_id`.
        :type mean: class:`torch.Tensor`

        :param cov: Tensor used to divide the input element-wise when standardizing (and to multiply by when inverting).
            Moved to `device_id`.
        :type cov: class:`torch.Tensor`

        :param device_id: The identifier for the device where `mean` and `cov` are stored, defaults to `"cuda"`.
        :type device_id: class:`Literal["cuda", "cpu"]`

        :param synch_devices: Whether to move `mean` and `cov` to the device of the input tensor before each call
            to :method:`transform`/:method:`inverse_transform`, defaults to `True`.
        :type synch_devices: class:`bool`
        """
        super().__init__()
        self.device_id = device_id
        self.synch_devices = synch_devices
        self.device = torch.device(self.device_id)
        self.mean = mean.to(self.device)
        self.cov = cov.to(self.device)

    def transform(self, input_tensor: Tensor) -> Tensor:
        """Standardizes `input_tensor` by subtracting `mean` and dividing element-wise by `cov`.

        :param input_tensor: The tensor to standardize.
        :type input_tensor: class:`torch.Tensor`

        :return: The standardized tensor.
        :rtype: class:`torch.Tensor`
        """
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
        """Reverses the standardization by multiplying `input_tensor` element-wise by `cov` and adding back `mean`.

        :param input_tensor: The standardized tensor to invert.
        :type input_tensor: class:`torch.Tensor`

        :return: The tensor mapped back to the original scale.
        :rtype: class:`torch.Tensor`
        """
        # copying the parameters
        mean = self.mean.clone()
        cov = self.cov.clone()
        # optionally synching the devices
        if self.synch_devices:
            mean = mean.to(input_tensor.device)
            cov = cov.to(input_tensor.device)
        return input_tensor * cov + mean
