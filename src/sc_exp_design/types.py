from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from numpy import ndarray
from torch import Tensor, nn

TensorLike = Tensor | ndarray


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
        layers_dict: dict,
    ) -> None:
        """"""
        keys = layers_dict.keys()
        msg = f"{keys=}"
        assert "layer_type" in keys, msg
        if layers_dict["layer_type"] == "mlp":
            msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'input_dim'` key."
            assert "input_dim" in keys, msg
            msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'output_dim'` key."
            assert "output_dim" in keys, msg
            msg = f"With {layers_dict['layer_type']=}, `'input_dim'` value should be an integer, found `None`."
            assert layers_dict["input_dim"] is not None, msg
            msg = f"With {layers_dict['layer_type']=}, `'output_dim'` value should be an integer, found `None`."
            assert layers_dict["output_dim"] is not None, msg
        elif layers_dict["layer_type"] == "self_attention":
            msg = f"With {layers_dict['layer_type']=}, the dictionary is expected to contain the `'num_embeddings'` key."
            assert "num_embeddings" in keys, msg
            msg = f"With {layers_dict['layer_type']=}, `'num_embeddings'` value should be an integer, found `None`."
            assert layers_dict["num_embeddings"] is not None, msg
