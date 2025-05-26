from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, ClassVar, Literal

from numpy import ndarray
from torch import Tensor, nn

from sc_exp_design.exceptions import ConfigurationError

__all__ = ["TensorLike", "LinearModelConfig", "MLPConfigFields"]


TensorLike = Tensor | ndarray


@dataclass(frozen=True)
class MLPConfigFields:
    """"""
    types: ClassVar[dict[str, type]] = {
        "input_dim": int,
        "output_dim": int, 
        "hidden_dims": Sequence,
        "use_batchnorm": bool,
        "use_dropout": bool,
        "dropout_rate": float,
        "activation_class": type,
        "final_activation_class": type,
    }
    
    @classmethod
    def get_fields(
        cls,
    ) -> Sequence[str]:
        """"""
        return list(cls.types.keys())

    @classmethod
    def verify_keys(
        cls,
        layers_dict: dict[str, Any],
        require_input_dim_key: bool = True,
        require_output_dim_key: bool = True,
    ) -> None:
        """"""
        # retrieving the input dictionary keys
        keys = layers_dict.keys()
        # optional check on input dimension
        if require_input_dim_key:
            # verify that the key is present in the dictionary
            if "input_dim" not in keys:
                msg = f"The dictionary is expected to contain the `'input_dim'` key."
                raise KeyError(msg)
            # verify that the value is not None
            if layers_dict["input_dim"] is None:
                msg = f"`'input_dim'` value should be an integer, found `None`."
                raise ConfigurationError(msg)
        else:
            # otherwise we are already passing the argument so it should not be there
            if "input_dim" in keys:
                msg = f"When {require_input_dim=}, `layers_dict` should not contain the \"input_dim\" key as this will be automatically set."
                raise ConfigurationError(msg)
        # optional check on output dimension
        if require_output_dim_key:
            # verify that the key is present in the dictionary
            if not "output_dim" in keys:
                msg = f"The dictionary is expected to contain the `'output_dim'` key."
                raise KeyError(msg)
            # verify that the value is not None
            if layers_dict["output_dim"] is None:
                msg = f"`'output_dim'` value should be an integer, found `None`."
                raise ConfigurationError(msg)
        else:
            # otherwise we are already passing the argument so it should not be there
            if "output_dim" in keys:
                msg = f"When {require_output_dim=}, `layers_dict` should not contain the \"input_dim\" key as this will be automatically set."
                raise ConfigurationError(msg)
        # cheking the other keys
        for key, value in layers_dict.items(): 
            # valid configuration key
            if key not in cls.get_fields():
                msg = f"Key {key} not a valid MLP configuration field, possible options are {cls.get_fields()}"
                raise ConfigurationError(msg)
            # valid type
            if not isinstance(value, cls.types[key]):
                msg = f"Value of {key} expected to be of type {cls.types[key]}, found {type(value)}"
                raise TypeError(msg)
