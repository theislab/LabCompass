from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from numpy import ndarray
from torch import Tensor

__all__ = ["TensorLike", "MLPConfigFields"]


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
            msg = "The dictionary is expected to contain the `'input_dim'` key."
            assert "input_dim" in keys, msg
            # verify that the value is not None
            msg = "`'input_dim'` value should be an integer, found `None`."
            assert layers_dict["input_dim"] is not None, msg
        else:
            # otherwise we are already passing the argument so it should not be there
            msg = ""
            assert "input_dim" not in keys, msg
        # optional check on output dimension
        if require_output_dim_key:
            # verify that the key is present in the dictionary
            msg = "The dictionary is expected to contain the `'output_dim'` key."
            assert "output_dim" in keys, msg
            # verify that the value is not None
            msg = "`'output_dim'` value should be an integer, found `None`."
            assert layers_dict["output_dim"] is not None, msg
        else:
            # otherwise we are already passing the argument so it should not be there
            msg = ""
            assert "output_dim" not in keys, msg
        # cheking the other keys
        for key, value in layers_dict.items():
            # valid configuration key
            msg = f"Key {key} not a valid MLP configuration field, possible options are {cls.get_fields()}"
            assert key in cls.get_fields(), msg
            # valid type
            msg = f"Value of {key} expected to be of type {cls.types[key]}, found {type(value)}"
            assert isinstance(value, cls.types[key]), msg
