from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

from numpy import ndarray
from torch import Tensor, nn

__all__ = ["TensorLike", "LinearModelConfig", "LayersDict"]


TensorLike = Tensor | ndarray


@dataclass(frozen=True)
class MLPConfigFields:
    """"""
    types: dict[str, type] = dc_field(default_factory=lambda: {
            "input_dim": int,
            "output_dim": int, 
            "hidden_dims": Sequence,
            "use_batchnorm": bool,
            "use_dropout": bool,
            "dropout_rate": float,
            "activation_class": type,
            "final_activation_class": type,
        }
    )

    @property
    def fields(
        self,
    ) -> Sequence[str]:
        """"""
        return list(self.types.keys())


@dataclass
class LayersDict:
    """"""
    layer_type: Literal["mlp", "self_attention"] | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    hidden_dims: Sequence[int] = (64, 64, 64)
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.0
    activation_class: nn.Module = nn.SELU
    final_activation_class: nn.Module = nn.Identity

    @classmethod
    def verify_keys(
        cls,
        layers_dict: dict[str, Any],
        require_input_dim_key: bool = True,
        require_output_dim_key: bool = True,
    ) -> dict[str, Any]:
        """"""
        # retrieving the input dictionary keys
        keys = layers_dict.keys()
        # optional check on input dimension
        if require_input_dim_key:
            # verify that the key is present in the dictionary
            msg = f"The dictionary is expected to contain the `'input_dim'` key."
            assert "input_dim" in keys, msg
            # verify that the value is not None
            msg = f"`'input_dim'` value should be an integer, found `None`."
            assert layers_dict["input_dim"] is not None, msg
        # optional check on output dimension
        if require_output_dim_key:
            # verify that the key is present in the dictionary
            msg = f"The dictionary is expected to contain the `'output_dim'` key."
            assert "output_dim" in keys, msg
            # verify that the value is not None
            msg = f"`'output_dim'` value should be an integer, found `None`."
            assert layers_dict["output_dim"] is not None, msg
        # cheking the other keys
        expected_fields = MLPConfigFields().fields
        expected_types = MLPConfigFields().types
        for key, value in layers_dict.items(): 
            # valid configuration key
            msg = f"Key {key} not a valid MLP configuration field, possible options are {expected_fields}"
            assert key in expected_fields, msg
            # valid type
            msg = f"Value of {key} expected to be of type {expected_types[key]}, found {type(value)}"
            assert isinstance(value, expected_types[key]), msg
        return layers_dict
