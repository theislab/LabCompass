import abc
import itertools
from collections.abc import Iterator, Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sc_exp_design.types import LayersDict

__all__ = ["ConditionEncoder", "BaseModule", "MLPBlock", "SelfAttentionBlock", "AttentionPooling"]


class BaseModule(abc.ABC, nn.Module):
    """Base class for Neural Network Modules"""

    @abc.abstractmethod
    def forward(self, input_tensor: Tensor):
        """"""
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
        return self.net(input_tensor)


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
        self.feature_embedder = None
        if self.embed_features:
            self.feature_embedder = CategoricalEmbedder(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                padding_idx=self.padding_idx,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                scale_grad_by_freq=self.scale_grad_by_freq,
                sparse=self.sparse,
            )

        modules = []

        for block_idx in range(self.num_blocks):
            embed_dim = self.embed_dim[block_idx]
            num_heads = self.num_heads[block_idx]

            mha = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=self.dropout_rate,
                batch_first=True,
            )
            modules.append(mha)

        self.net = nn.ModuleList(modules)

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
            x = self.feature_embedder(x)
        for layer in self.net:
            x = layer(
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
        the identifier of the perturbation covariate to decode, while each value will be an instance of :class:`LayersDict`.
        Defaults to `None`.
    :type layers_before_pooling: class:`dict[str, LayersDict]`

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
        Should be either an instance of :class:`LayersDict`, defaults to `None`.
    :type layers_after_pooling: class:`LayersDict`
    """

    def __init__(
        self,
        latent_dim: int,
        layers_before_pooling: dict[str, LayersDict] | None = None,
        covariates_not_pooled: Sequence[str] | None = None,
        pooling: Literal["mean", "self_attention"] = "mean",
        pooling_kwargs: dict[str, Any] | None = None,
        layers_after_pooling: LayersDict | None = None,
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
        if self.covariates_not_pooled is not None:
            covariates_to_pool = [
                covariate
                for covariate in self.layers_before_pooling.keys()
                if covariate not in self.covariates_not_pooled
            ]
        else:
            covariates_to_pool = list(self.layers_before_pooling.keys())
        return covariates_to_pool

    def to(
        self,
        device: torch.device,
    ) -> nn.Module:
        """Moves the module to the target device

        :param device: The device which to perform the computations on.
        :type device: class:`torch.device`
        """
        self = super().to(device)
        before_pooling = {}
        for covariate_id, cov_before_pooling in self.before_pooling.items():
            before_pooling[covariate_id] = cov_before_pooling.to(device)
        self.before_pooling = before_pooling
        return self

    def parameters(
        self,
    ) -> Iterator[nn.Parameter]:
        """Returns an iterator with the parameters of each module"""
        parameters = [super().parameters()]
        for covariate_id, cov_before_pooling in self.before_pooling.items():
            parameters.append(cov_before_pooling.parameters())
        parameters = itertools.chain(*parameters)
        return parameters

    def train(self, train: bool = True) -> None:
        """Whether the forward pass should be computed in training (stochastic) or in evaluation (deterministic) mode

        :param train: Whether the computations are computed in the training/stochastic mode.
        :type train: class:`bool`
        """
        self = super().train(train)
        before_pooling = {}
        for covariate_id, cov_before_pooling in self.before_pooling.items():
            before_pooling[covariate_id] = cov_before_pooling.train(train)
        self.before_pooling = before_pooling
        return self

    def eval(
        self,
    ) -> None:
        """Sets the computation to the deterministic/evaluation mode."""
        self = super().eval()
        before_pooling = {}
        for covariate_id, cov_before_pooling in self.before_pooling.items():
            before_pooling[covariate_id] = cov_before_pooling.eval()
        self.before_pooling = before_pooling
        return self

    def _init_modules(
        self,
    ) -> None:
        """Initializes the modules."""
        # initializing the layers before pooling
        self.before_pooling = {}
        for covariate, layers_dict in self.layers_before_pooling.items():
            covariate_layers = self._get_layers(layers_dict)
            self.before_pooling[covariate] = covariate_layers

        # pooling modules
        if self.pooling == "mean":
            self.pooling_layer = lambda x, mask: torch.mean(x * mask, dim=-2)
        elif self.pooling == "self_attention":
            self.pooling_layer = AttentionPooling(**self.pooling_kwargs)
        else:
            msg = f"{self.pooling=} not available, possible options are `['mean', 'self_attention']`"
            raise ValueError(msg)

        # layers after pooling
        self.after_pooling = self._get_layers(self.layers_after_pooling)

    def __get_mask(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> Tensor:
        """Retrieves the mask for attention blocks (still to be correctly implemented)"""
        mask = torch.ones((sequence_length, sequence_length), device=device)
        return mask

    def _get_layers(
        self,
        layers_dict: LayersDict,
    ) -> nn.Module:
        """Initializes a given layer with the settings provided in :param:`layers_dict`.

        :param layers_dict: Instance of :class:`LayersDict` with the configurations used to initialize the layer
        :type layers_dict: class:`LayersDict`
        """
        if layers_dict.layer_type == "mlp":
            layer = MLPBlock(
                layers_dict.input_dim,
                layers_dict.output_dim,
                hidden_dims=layers_dict.hidden_dims,
                use_batchnorm=layers_dict.use_batchnorm,
                use_dropout=layers_dict.use_dropout,
                dropout_rate=layers_dict.dropout_rate,
                activation_class=layers_dict.activation_class,
                final_activation_class=layers_dict.final_activation_class,
            )
        elif layers_dict.layer_type == "self_attention":
            layer = SelfAttentionBlock(
                layers_dict.embed_dim,
                layers_dict.num_heads,
                layers_dict.dropout_rate,
                num_embeddings=layers_dict.num_embeddings,
                embedding_dim=layers_dict.embedding_dim,
                padding_idx=layers_dict.padding_idx,
                max_norm=layers_dict.max_norm,
                norm_type=layers_dict.norm_type,
                scale_grad_by_freq=layers_dict.scale_grad_by_freq,
                sparse=layers_dict.sparse,
            )
        else:
            msg = f"{layers_dict.layer_type=} not available, possible options are `['mlp', 'self_attention']`"
            raise ValueError(msg)
        return layer

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
            if covariate not in self.layers_before_pooling:
                continue
            before_pooling = self.before_pooling[covariate]
            if isinstance(before_pooling, SelfAttentionBlock):
                mask = self.__get_mask(covariate_data.shape[1], covariate_data.device)
                encoded_covariate = before_pooling(covariate_data, mask)
            else:
                encoded_covariate = before_pooling(covariate_data)
            encoded_covariates[covariate] = encoded_covariate

        # concatenating in two separete arrays the covariates
        # one for the covariates to pool the other for the remaining ones
        if self.covariates_not_pooled is not None:
            encoded_covariates_not_pooled = torch.concatenate(
                [encoded_covariates[covariate] for covariate in self.covariates_not_pooled], dim=-1
            )
            encoded_covariates_pooled = None
            if len(self.covariates_to_pool) > 0:
                encoded_covariates_pooled = torch.concatenate(
                    [
                        encoded_covariate
                        for covariate, encoded_covariate in encoded_covariates.items()
                        if covariate in self.covariates_to_pool
                    ],
                    dim=-1,
                )
        else:
            encoded_covariates_pooled = torch.concatenate(
                [encoded_covariate for covariate, encoded_covariate in encoded_covariates.items()], dim=-1
            )

        if encoded_covariates_pooled is not None:
            # pooling the covariates
            mask = self.__get_mask(encoded_covariates_pooled.shape[1], encoded_covariates_pooled.device)
            z = self.pooling_layer(encoded_covariates_pooled, mask)
            # concatenating with the covariates not pooled
            if self.covariates_not_pooled is not None:
                z = torch.concatenate((z, encoded_covariates_not_pooled), dim=-1)
        elif self.covariates_not_pooled is not None and encoded_covariates_pooled is None:
            z = encoded_covariates_not_pooled

        # layers after pooling
        if isinstance(self.after_pooling, SelfAttentionBlock):
            mask = self.__get_mask(z.shape[1], z.device)
            z = self.after_pooling(z, mask)
        else:
            z = self.after_pooling(z)
        return z
