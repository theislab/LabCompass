import abc
from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from labcompass.constants import DataFields
from labcompass.types import MLPConfigFields
from labcompass.utils import get_conditions_to_pool

__all__ = ["ConditionEncoder", "BaseModule", "MLPBlock", "SelfAttentionBlock", "AttentionPooling"]


class BaseModule(abc.ABC, nn.Module):
    """Base class for Neural Network Modules"""

    @abc.abstractmethod
    def forward(self, input_tensor: Tensor):
        """Performs the forward pass of the module. Must be overridden by every subclass.

        :param input_tensor: The input tensor to the module.
        :type input_tensor: class:`torch.Tensor`

        :raises NotImplementedError: Always, since this is an abstract method that subclasses must override.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _init_modules(
        self,
    ) -> None:
        """"""
        raise NotImplementedError


class BaseForwardModel(BaseModule):

    @abc.abstractmethod
    def forward(
        self,
        source_states: torch.Tensor,
        perturbations: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """"""
        raise NotImplementedError

    def predict(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        no_grad: bool = True,
    ) -> torch.Tensor:
        """"""
        # parsing batch dictionary
        source_states = batch[DataFields.SOURCE_STATE]
        perturbations = batch[DataFields.PERTURBATION_DATA]
        if no_grad:
            with torch.no_grad():
                out =  self.forward(source_states, perturbations)
        else:
            out =  self.forward(source_states, perturbations)
        return out


class MLPBlock(BaseModule):
    """Implements a Multi-Layered Perceptron with optional batch normalization and dropout

    :param input_dim: The dimensionality of the input data.
    :type input_dim: class:`int`

    :param output_dim: The dimensionality of the output data.
    :type output_dim: class:`int`

    :param hidden_dims: The dimensionalities of the hidden representations, defaults to `(128, 64, 10)`.
    :type hidden_dims: class:`Sequence[int]`

    :param use_batchnorm: Whether to use batch normalization, defaults to `False`.
    :type use_batchnorm: class:`bool`

    :param use_dropout: Whether to use random dropout, defaults to `False`.
    :type use_dropout: class:`bool`

    :param dropout_rate: The dropout rate used when :param:`use_dropout` is `True`, defaults to `0.0`.
    :type dropout_rate: class:`float`

    :param activation_class: A reference to a :class:`torch.nn.Module` used as activation for the hidden layers
        of the (decoder. Note that you should pass a class and not an instance, defaults to `torch.nn.ELU`.
    :type activation_class: class:`torch.nn.Module`

    :param final_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the output (final) layer
        of the decoder. Note that you should pass a class and not an instance, defaults to `torch.nn.Identity`.
    :type final_activation_class: class:`torch.nn.Module`
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.0,
        activation_class: nn.Module = nn.ELU,
        final_activation_class: nn.Module = nn.Identity,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.activation_class = activation_class
        self.final_activation_class = final_activation_class
        # initializing modules
        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules"""
        modules = []
        input_dim = self.input_dim
        # input and hidden layers
        for output_dim in self.hidden_dims:
            block = []
            # linear
            linear = nn.Linear(input_dim, output_dim)
            block.append(linear)
            # batchnorm
            if self.use_batchnorm:
                batchnorm = nn.BatchNorm1d(output_dim)
                block.append(batchnorm)
            # activation
            activation = self.activation_class()
            block.append(activation)
            # dropout
            if self.use_dropout:
                dropout = nn.Dropout(self.dropout_rate)
                block.append(dropout)
            # initialing block and updating current input dim
            block = nn.Sequential(*block)
            modules.append(block)
            input_dim = output_dim
        # output layer
        block = []
        linear = nn.Linear(input_dim, self.output_dim)
        block.append(linear)
        # optional final activation
        final_activation = self.final_activation_class()
        block.append(final_activation)
        block = nn.Sequential(*block)
        # initializing modules
        modules.append(block)
        self.net = nn.Sequential(*modules)

    def forward(
        self,
        input_tensor: Tensor,
    ) -> Tensor:
        """Performs a forward pass on the input tensor

        :param input_tensor: Input tensor of shape `(batch_size, self.input_dim)`
        :type input_tensor: class:`torch.Tensor`

        :return: Tensor of shape `(batch_size, self.output_dim)`
        :rtype: class:`torch.Tensor`
        """
        original_shape = input_tensor.shape[:-1]
        input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
        y = self.net(input_tensor)
        return y.reshape(*original_shape, -1)


class CategoricalEmbedder(BaseModule):
    """Implements a learnable embedding (i.e.: look-up table) to encode categorical variables

    :param num_embeddings: The number of unique categories/labels to embed in dense vectors, defaults to `None`.
    :type num_embeddings: class:`int | None`

    :param embedding_dim: The dimensionality of the dense embedding, defaults to `1024`.
    :type embedding_dim: class:`int`

    :param padding_idx: Indices for the entries that do not contribute to the gradients, check original :module:`pytorch`
        implementation for further reference, defaults to `None`.
    :type padding_idx: class:`int`

    :param max_norm: The value at which to clip the norm of each embedding, check original :module:`pytorch`
        implementation for further reference, defaults to `None` .
    :type max_norm: class:`bool | None`

    :param norm_type: The order of the norm used for computing the maximum norm when :param:`max_norm` is set to `True`, check original :module:`pytorch`
        implementation for further reference, defaults to `2.0`.
    :type norm_type: class:`float`

    :param scale_grad_by_freq: Whether to scale the gradients by the inverse of the frequency of each token at the mini-batch level, check original :module:`pytorch`
        implementation for further reference, defaults to `True`.
    :type scale_grad_by_freq: class:`bool`

    :param sparse: Whether to use sparse Tensors for the gradients with respect to the model's weights, check original :module:`pytorch`
        implementation for further reference, defaults to `False`.
    :type sparse: class: `bool`
    """

    def __init__(
        self,
        num_embeddings: int | None = None,
        embedding_dim: int = 1024,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = True,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules"""
        self.embedder = nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(
        self,
        condition: Tensor,
    ) -> Tensor:
        """Forward pass on the embedder

        :param condition: Tensor of any arbitrary shapes containing the labels of each category. Note that the maximum value
            of any entry in the array should be given by :attr:`CategoricalEmbedder.num_embeddings`, otherwise an error is thrown.
        :type condition: class:`Tensor`

        :return: Tensor of shape `(., self.embedding_dim)`
        :rtype: class:`torch.Tensor`
        """
        condition = condition.long()
        condition = self.embedder(condition)
        condition = condition.float()
        return condition


class SelfAttentionBlock(BaseModule):
    """Implements a Self-Attention block

    :param embed_dim: he number of dimensionality for the embeddings used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type embed_dim: class:`int | Sequence[int]`

    :param num_heads: The number of attention heads used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type num_heads: class:`int | Sequence[int]`

    :param dropout_rate: The dropout rate used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type dropout_rate: class:`float`

    :param embed_features: Whether to embed categorical features using :class:`CategoricalEmbedder`, defaults to `True`.
    :type embed_features: class:`bool`

    :param num_embeddings: The number of unique categories/labels to embed in dense vectors, defaults to `None`.
    :type num_embeddings: class:`int | None`

    :param embedding_dim: The dimensionality of the dense embedding, defaults to `1024`.
    :type embedding_dim: class:`int`

    :param padding_idx: Indices for the entries that do not contribute to the gradients, check original :module:`pytorch`
        implementation for further reference, defaults to `None`.
    :type padding_idx: class:`int`

    :param max_norm: The value at which to clip the norm of each embedding, check original :module:`pytorch`
        implementation for further reference, defaults to `None` .
    :type max_norm: class:`bool | None`

    :param norm_type: The order of the norm used for computing the maximum norm when :param:`max_norm` is set to `True`, check original :module:`pytorch`
        implementation for further reference, defaults to `2.0`.
    :type norm_type: class:`float`

    :param scale_grad_by_freq: Whether to scale the gradients by the inverse of the frequency of each token at the mini-batch level, check original :module:`pytorch`
        implementation for further reference, defaults to `True`.
    :type scale_grad_by_freq: class:`bool`

    :param sparse: Whether to use sparse Tensors for the gradients with respect to the model's weights, check original :module:`pytorch`
        implementation for further reference, defaults to `False`.
    """

    def __init__(
        self,
        embed_dim: int | Sequence[int],
        num_heads: int | Sequence[int],
        dropout_rate: float,
        embed_features: bool = True,
        num_embeddings: int | None = None,
        embedding_dim: int = 1024,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = True,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        # setting the attributes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.embed_features = embed_features
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.num_blocks = len(self.num_heads)
        # initializing modules
        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules"""
        # feature embedder
        modules = {}
        self.feature_embedder = None
        if self.embed_features:
            modules["feature_embedder"] = CategoricalEmbedder(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )

        for block_idx in range(self.num_blocks):
            embed_dim = self.embed_dim[block_idx]
            num_heads = self.num_heads[block_idx]

            modules[f"mha_{block_idx}"] = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=self.dropout_rate,
                batch_first=True,
            )

        self.net = nn.ModuleDict(modules)

    def forward(
        self,
        input_tensor: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass on the embedder

        :param condition: Tensor of shape `(batch_size, num_perts, num_pert_features)` containing the labels of each category. Note that the maximum value
            of any entry in the array should be given by :attr:`CategoricalEmbedder.num_embeddings`, otherwise an error is thrown.
        :type condition: class:`Tensor`

        :return: Tensor of shape `(batch_size, num_perts, self.embedding_dim)`
        :rtype: class:`torch.Tensor`
        """
        x = input_tensor
        if self.embed_features:
            x = self.net["feature_embedder"](x)
        for block_idx in range(self.num_blocks):
            x = self.net[f"mha_{block_idx}"](
                x,
                x,
                x,
                attn_mask=attention_mask,
                need_weights=False,
            )[0]
        return x


class AttentionPooling(BaseModule):
    """Implements Self Attention Block and pools by taking the first element of the sequence. This module expects already dense features.

    :param embed_dim: he number of dimensionality for the embeddings used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type embed_dim: class:`int | Sequence[int]`

    :param num_heads: The number of attention heads used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type num_heads: class:`int | Sequence[int]`

    :param dropout_rate: The dropout rate used in each of the :class:`torch.nn.MultiHeadAttention` comprising this module.
    :type dropout_rate: class:`float`

    :param embed_features: Whether to embed categorical features using :class:`CategoricalEmbedder`, defaults to `True`.
    :type embed_features: class:`bool`

    :param num_embeddings: The number of unique categories/labels to embed in dense vectors, defaults to `None`.
    :type num_embeddings: class:`int | None`

    :param embedding_dim: The dimensionality of the dense embedding, defaults to `1024`.
    :type embedding_dim: class:`int`

    :param padding_idx: Indices for the entries that do not contribute to the gradients, check original :module:`pytorch`
        implementation for further reference, defaults to `None`.
    :type padding_idx: class:`int`

    :param max_norm: The value at which to clip the norm of each embedding, check original :module:`pytorch`
        implementation for further reference, defaults to `None` .
    :type max_norm: class:`bool | None`

    :param norm_type: The order of the norm used for computing the maximum norm when :param:`max_norm` is set to `True`, check original :module:`pytorch`
        implementation for further reference, defaults to `2.0`.
    :type norm_type: class:`float`

    :param scale_grad_by_freq: Whether to scale the gradients by the inverse of the frequency of each token at the mini-batch level, check original :module:`pytorch`
        implementation for further reference, defaults to `True`.
    :type scale_grad_by_freq: class:`bool`

    :param sparse: Whether to use sparse Tensors for the gradients with respect to the model's weights, check original :module:`pytorch`
        implementation for further reference, defaults to `False`.
    """

    def __init__(
        self,
        embed_dim: int | Sequence[int],
        num_heads: int | Sequence[int],
        dropout_rate: float = 0.5,
        num_embeddings: int | None = None,
        embedding_dim: int = 1024,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = True,
        sparse: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # initializing modules
        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules"""
        # attention block
        self.attention_block = SelfAttentionBlock(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            embed_features=False,
            num_embeddings=self.num_embeddings,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

    def forward(
        self,
        input_tensor: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Forward pass on the embedder

        :param condition: Tensor of shape `(batch_size, num_perts, num_pert_features)` containing the labels of each category. Note that the maximum value
            of any entry in the array should be given by :attr:`CategoricalEmbedder.num_embeddings`, otherwise an error is thrown.
        :type condition: class:`Tensor`

        :return: Tensor of shape `(batch_size, num_perts, self.embedding_dim)`
        :rtype: class:`torch.Tensor`
        """
        out = self.attention_block(input_tensor, mask=attention_mask)
        return out[:, 0, :]


class ConditionEncoder(BaseModule):
    """Implements the encoder used to obtain the latent representation for the conditions.

    :param latent_dim: Latent dimensionality for the encoded conditions.
    :type latent_dim: class:`int`

    :param layers_before_pooling: Dictionary mapping each condition to be encoded to the configuration of its encoder.
        Each key of :attr:`.NeuralVelocityFieldConfig.condiion_layers_before_pooling` will be given by a :class:`str` with
        the identifier of the perturbation covariate to decode, while each value will be an instance of :class:`MLPConfigFields`.
        Defaults to `None`.
    :type layers_before_pooling: class:`dict[str, MLPConfigFields]`

    :param covariates_not_pooled: A sequence with the names of the perturbations covariates that are encoded (i.e.: they appear as keys
        in :attr:`ConditionEncoder.layers_before_pooling`) and whose latent representation will be directly concatenated to the
        :attr:`ConditionEncoder.after_pooling`, thus skipping the pooling step, defaults to `None`.
    :type covariates_not_pooled: class:`Sequence[str]`

    :param pooling: The pooling method used for the perturbation covariates, defaults tp `"mean"`.
    :type pooling: class:`Literal["mean", "self_attention"]`

    :param pooling_kwargs: The keyword arguments used to instantiate the pooling layer in the condition encoder.
        Should contain key-value pairs corresponding to the possible attributes used in the initialization of an :class:`AttentionPooling` object.
        Sets the attribute :attr:`ConditionEncoder.pooling_kwargs` which are then used to initialize :attr:`ConditionEncoder.pooling_layer`
        in case :attr:`ConditionEncoder.pooling` is set to `"self_attention"`, defaults to `None`.
    :type pooling_kwargs: class:`dict[str, Any]`

    :param layers_after_pooling: Configuration for the condition decoder, used to initialize the :attr:`ConditionEncoder.after_pooling` attribute of
        Should be either an instance of :class:`MLPConfigFields`, defaults to `None`.
    :type layers_after_pooling: class:`MLPConfigFields`
    """

    def __init__(
        self,
        latent_dim: int,
        layers_before_pooling: dict[str, MLPConfigFields] | None = None,
        covariates_not_pooled: Sequence[str] | None = None,
        pooling: Literal["mean", "sum", "self_attention"] = "mean",
        pooling_kwargs: dict[str, Any] | None = None,
        layers_after_pooling: MLPConfigFields | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.layers_before_pooling = layers_before_pooling
        self.covariates_not_pooled = covariates_not_pooled
        self.pooling = pooling
        self.pooling_kwargs = pooling_kwargs
        self.layers_after_pooling = layers_after_pooling
        # initializing modules
        self._init_modules()

    @property
    def covariates_to_pool(
        self,
    ) -> Sequence[str]:
        """Returns the name of the perturbation covariates that needs to be pooled."""
        return get_conditions_to_pool(
            self.layers_before_pooling,
            self.covariates_not_pooled
        )

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules."""
        # initializing the layers before pooling
        modules = {}
        for covariate, layers_dict in self.layers_before_pooling.items():
            covariate_layers = MLPBlock(**layers_dict)
            modules[covariate] = covariate_layers

        # pooling modules
        if self.pooling == "mean":
            self.pooling_layer = lambda x: torch.mean(x, dim=1)
        elif self.pooling == "sum":
            self.pooling_layer = lambda x: torch.sum(x, dim=1)
        elif self.pooling == "self_attention":
            msg = ""
            raise NotImplementedError(msg)
        else:
            msg = f"{self.pooling=} not available, possible options are `['mean', 'self_attention']`"
            raise ValueError(msg)

        # layers after pooling
        modules["after_pooling"] = MLPBlock(**self.layers_after_pooling)
        self.modules_dict = nn.ModuleDict(modules)

    def __get_mask(
        self,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        """Retrieves the mask for attention blocks (still to be correctly implemented)"""
        mask = torch.ones((batch_size, sequence_length, sequence_length), device=device)
        return mask

    def forward(
        self,
        conditions: dict[str, Tensor],
    ) -> Tensor:
        """Forward pass on the condition encoder

        :param conditions: Dictionary with keys corresponding to identifiers for the
            perturbation covarites to be used to guide the flow and values being their corresponding data.
        :type conditions: class:`dict[str, Tensor]`
        """
        # layers before pooling
        encoded_covariates = {}
        for covariate, covariate_data in conditions.items():
            if covariate not in self.modules_dict:
                continue
            before_pooling = self.modules_dict[covariate]
            encoded_covariate = before_pooling(covariate_data)
            encoded_covariates[covariate] = encoded_covariate

        # concatenating in two separete arrays the covariates
        # one for the covariates to pool the other for the remaining ones
        if self.covariates_not_pooled is not None:
            encoded_covariates_not_pooled = torch.concatenate(
                [encoded_covariates[covariate] for covariate in self.covariates_not_pooled], dim=-1
            )  # B x N_conditions x (D * N_cov_not_pooled)
            encoded_covariates_pooled = None
            if len(self.covariates_to_pool) > 0:
                encoded_covariates_pooled = torch.stack(
                    [
                        encoded_covariate
                        for covariate, encoded_covariate in encoded_covariates.items()
                        if covariate in self.covariates_to_pool
                    ],
                    dim=-1,
                )  # B x N_conditions x D
        else:
            encoded_covariates_pooled = torch.stack(
                [encoded_covariate for covariate, encoded_covariate in encoded_covariates.items()], dim=1
            )  # B x N_conditions x D

        if encoded_covariates_pooled is not None:
            # pooling the covariates
            if self.pooling == "attention":
                mask = self.__get_mask(encoded_covariates_pooled.shape[0], encoded_covariates_pooled.shape[1], encoded_covariates_pooled.device)
                z = self.pooling_layer(encoded_covariates_pooled, mask)
            else:
                z = self.pooling_layer(encoded_covariates_pooled)
            # concatenating with the covariates not pooled
            if self.covariates_not_pooled is not None:
                z = torch.concatenate((z, encoded_covariates_not_pooled), dim=-1)
        elif self.covariates_not_pooled is not None and encoded_covariates_pooled is None:
            z = encoded_covariates_not_pooled

        # layers after pooling
        z = self.modules_dict["after_pooling"](z)
        return z


class ResnetBlock(BaseModule):
    """
    A residual MLP block with optional normalization, dropout, and conditional embedding.

    Args:
        in_dim (int): Input feature dimension.
        out_dim (int, optional): Output feature dimension. Defaults to in_dim.
        dropout_prob (float, optional): Dropout probability. Defaults to 0.0.
        embedding_dim (int, optional): Dimensionality of the conditional embedding. Defaults to 128.
        normalization (str, optional): Type of normalization to use: 'layer', 'batch', or None. Defaults to None.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int | None = None,
        dropout_prob: float = 0.0,
        embedding_dim: int = 128,
        normalization: Literal["layer", "batch"] | None = None
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim
        self.dropout_prob = dropout_prob
        self.embedding_dim = embedding_dim
        self.normalization = normalization

        self._init_modules()

    def _init_modules(self):
        # First linear block
        self.net1 = self._build_block(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            apply_norm=self.normalization is not None,
            use_dropout=False
        )

        # Conditional projection block
        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embedding_dim, self.out_dim)
        )

        # Second linear block
        self.net2 = self._build_block(
            in_dim=self.out_dim,
            out_dim=self.out_dim,
            apply_norm=self.normalization is not None,
            use_dropout=self.dropout_prob > 0.0
        )

        # Optional skip projection
        self.skip_proj = (
            nn.Linear(self.in_dim, self.out_dim)
            if self.in_dim != self.out_dim
            else nn.Identity()
        )

    def _build_block(self, in_dim, out_dim, apply_norm=True, use_dropout=False):
        layers = []

        if apply_norm:
            norm_layer = self._get_normalization(in_dim)
            layers.append(norm_layer)

        layers.append(nn.SiLU())

        if use_dropout:
            layers.append(nn.Dropout(self.dropout_prob))

        layers.append(nn.Linear(in_dim, out_dim))

        return nn.Sequential(*layers)

    def _get_normalization(self, dim):
        if self.normalization == "layer":
            return nn.LayerNorm(dim)
        elif self.normalization == "batch":
            return nn.BatchNorm1d(num_features=dim)
        elif self.normalization is None:
            return None
        else:
            raise NotImplementedError(f"Unsupported normalization: {self.normalization}")

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_dim).
            cond (torch.Tensor): Conditional tensor of shape (B, embedding_dim).

        Returns
        -------
            torch.Tensor: Output tensor of shape (B, out_dim).
        """
        h = self.net1(x)
        h = h + self.cond_proj(cond)
        h = self.net2(h)

        x_proj = self.skip_proj(x)

        assert h.shape == x_proj.shape, f"Shape mismatch: {h.shape} vs {x_proj.shape}"
        return x_proj + h


class FiLMBlock(BaseModule):
    """
    A feature-wise Linear Modulation (FiLM) layer proposed by Perez et al. 2017 (https://arxiv.org/pdf/1709.07871).

    Args:
        in_dim (int): Input feature dimension.
        cond_dim (int): Condition feature dimention.
        out_dim (int, optional): Output feature dimension. Defaults to in_dim.
    """

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.cond_dim = cond_dim

        self._init_modules()

    def _init_modules(
        self,
    ) -> None:
        # Film generator
        self.film_generator = nn.Linear(self.cond_dim, self.in_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FiLM block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_dim).
            cond (torch.Tensor): Conditional tensor of shape (B, embedding_dim).

        Returns
        -------
            torch.Tensor: Output tensor of shape (B, out_dim).
        """
        gamma_beta = self.film_generator(cond)
        gamma, beta = torch.split(gamma_beta, self.in_dim, dim=-1)  # each shape: (batch, input_dim)
        assert gamma.shape == x.shape, f"Shape mismatch: {gamma.shape} vs {x.shape}"
        assert beta.shape == x.shape, f"Shape mismatch: {gamma.shape} vs {x.shape}"
        return nn.functional.silu(gamma * x + beta)
