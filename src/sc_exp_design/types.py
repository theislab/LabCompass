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
    fields: Sequence[str] = dc_field(default_factory=lambda: [
            "input_dim",
            "output_dim",
            "hidden_dims",
            "use_batchnorm",
            "use_dropout",
            "dropout_rate",
            "activation_class",
            "final_activation_class"
        ]
    )
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


@dataclass
class LayersDict:
    layer_type: Literal["mlp", "self_attention"] | None = None
    input_dim: int | None = None
    output_dim: int | None = None
    hidden_dims: Sequence[int] = (128, 64, 32)
    use_batchnorm: bool = False
    use_dropout: bool = False
    dropout_rate: float = 0.0
    activation_class: nn.Module = nn.ELU
    final_activation_class: nn.Module = nn.Identity
    embed_dim: int | Sequence[int] | None = None
    num_heads: int | Sequence[int] | None = None
    num_embeddings: int | None = None,
    embedding_dim: int = 1024,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = True,
    sparse: bool = False,

    @classmethod
    def verify_keys(
        cls,
        layers_dict: dict[str, Any],
        require_layer_type_key: bool = True,
        require_input_dim_key: bool = True,
        require_output_dim_key: bool = True,
        layer_type: Literal["mlp", "self_attention"] = "mlp",
    ) -> None:
        """"""
        # optional check on layer type key
        if require_layer_type_key:
            keys = layers_dict.keys()
            msg = f"{keys=}"
            assert "layer_type" in keys, msg
            layer_type = layers_dict["layer_type"]
        # multi layered perceptron
        if layer_type == "mlp":
            # optional check on input dimension
            if require_input_dim_key:
                # verify that the key is present in the dictionary
                msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'input_dim'` key."
                assert "input_dim" in keys, msg
                # verify that the value is not None
                msg = f"With {layers_dict['layer_type']=}, `'input_dim'` value should be an integer, found `None`."
                assert layers_dict["input_dim"] is not None, msg
            # optional check on output dimension
            if require_output_dim_key:
                # verify that the key is present in the dictionary
                msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'output_dim'` key."
                assert "output_dim" in keys, msg
                # verify that the value is not None
                msg = f"With {layers_dict['layer_type']=}, `'output_dim'` value should be an integer, found `None`."
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
        # self attention block
        elif layer_type == "self_attention":
            msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'num_embeddings'` key."
            assert "num_embeddings" in keys, msg
            msg = f"With {layers_dict['layer_type']=}, `'num_embeddings'` value should be an integer, found `None`."
            assert layers_dict["num_embeddings"] is not None, msg
