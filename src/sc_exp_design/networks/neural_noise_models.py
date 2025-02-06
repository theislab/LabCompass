from collections.abc import Sequence
from typing import Literal

from torch import Tensor, nn

from sc_exp_design.constants import (
    COVARIANCE_KEY,
    MEAN_KEY,
)
from sc_exp_design.networks.blocks import BaseModule, MLPBlock

__all__ = [
    "MLPNegBinNoiseModel",
    "MLPGaussianNoiseModel",
]


class MLPNegBinNoiseModel(BaseModule):
    """"""

    def __init__(
        self,
    ) -> None:
        """"""
        super().__init__()

    def _init_modules(
        self,
    ) -> nn.Module:
        """"""

    def forward(
        self,
        input_tensor: Tensor,
        size_factor: Tensor,
    ) -> Tensor:
        """"""


class MLPGaussianNoiseModel(BaseModule):
    """
    Initializes the MLP Gaussian Noise Model.

    :param input_dim: Dimensionality of the input data.
    
    :param output_dim: Dimensionality of the output (target size).
    
    :param latent_dim: Dimensionality of the latent space representation, defaults to 1024.

    :param use_shared_representation: Whether to share the representation between encoder and decoder, defaults to `False`.
    
    :param encoder_hidden_dims: Hidden layer dimensions for the encoder network, defaults to `(1024, 1024, 1024)`.
    
    :param encoder_use_batchnorm: Whether to use batch normalization in the encoder, defaults to `False`.
    
    :param encoder_use_dropout: Whether to use dropout in the encoder, defaults to `False`.
    
    :param encoder_dropout_rate: Dropout rate for the encoder, defaults to `0.0`.
    
    :param encoder_activation_class: Activation function used in the encoder, defaults to `nn.ELU`.
    
    :param encoder_final_activation_class: Final activation function for the encoder, defaults to `nn.ELU`.
    
    :param mean_hidden_dims: Hidden layer dimensions for the mean network, defaults to `(1024, 1024, 1024)`.
    
    :param cov_hidden_dims: Hidden layer dimensions for the covariance network, defaults to `(1024, 1024, 1024)`.
    
    :param mean_use_batchnorm: Whether to use batch normalization in the mean network, defaults to `False`.
    
    :param cov_use_batchnorm: Whether to use batch normalization in the covariance network, defaults to `False`.
    
    :param mean_use_dropout: Whether to use dropout in the mean network, defaults to `False`.
    
    :param cov_use_dropout: Whether to use dropout in the covariance network, defaults to `False`.
    
    :param mean_dropout_rate: Dropout rate for the mean network, defaults to `0.0`.
    
    :param cov_dropout_rate: Dropout rate for the covariance network, defaults to `0.0`.
    
    :param mean_activation_class: Activation function for the mean network, defaults to `nn.ELU`.
    
    :param cov_activation_class: Activation function for the covariance network, defaults to `nn.ELU`.
    
    :param mean_final_activation_class: Final activation function for the mean network, defaults to `nn.Identity`.
    
    :param cov_final_activation_class: Final activation function for the covariance network, defaults to `nn.Softplus`.
    
    :param cov_estimation_mode: Covariance estimation mode, either 'isotropic' or 'anisotropic', defaults to 'isotropic'.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 1024,
        use_shared_representation: bool = False,
        encoder_hidden_dims: Sequence[int] = (1024, 1024, 1024),
        encoder_use_batchnorm: bool = False,
        encoder_use_dropout: bool = False,
        encoder_dropout_rate: float = 0.0,
        encoder_activation_class: nn.Module = nn.ELU,
        encoder_final_activation_class: nn.Module = nn.ELU,
        mean_hidden_dims: Sequence[int] = (1024, 1024, 1024),
        cov_hidden_dims: Sequence[int] = (1024, 1024, 1024),
        mean_use_batchnorm: bool = False,
        cov_use_batchnorm: bool = False,
        mean_use_dropout: bool = False,
        cov_use_dropout: bool = False,
        mean_dropout_rate: float = 0.0,
        cov_dropout_rate: float = 0.0,
        mean_activation_class: nn.Module = nn.ELU,
        cov_activation_class: nn.Module = nn.ELU,
        mean_final_activation_class: nn.Module = nn.Identity,
        cov_final_activation_class: nn.Module = nn.Softplus,
        cov_estimation_mode: Literal["isotropic", "anisotropic"] = "isotropic",
    ) -> None:
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.use_shared_representation = use_shared_representation
        self.encoder_hidden_dims = encoder_hidden_dims
        self.encoder_use_batchnorm = encoder_use_batchnorm
        self.encoder_use_dropout = encoder_use_dropout
        self.encoder_dropout_rate = encoder_dropout_rate
        self.encoder_activation_class = encoder_activation_class
        self.encoder_final_activation_class = encoder_final_activation_class
        self.mean_hidden_dims = mean_hidden_dims
        self.cov_hidden_dims = cov_hidden_dims
        self.mean_use_batchnorm = mean_use_batchnorm
        self.cov_use_batchnorm = cov_use_batchnorm
        self.mean_use_dropout = mean_use_dropout
        self.cov_use_dropout = cov_use_dropout
        self.mean_dropout_rate = mean_dropout_rate
        self.cov_dropout_rate = cov_dropout_rate
        self.mean_activation_class = mean_activation_class
        self.cov_activation_class = cov_activation_class
        self.mean_final_activation_class = mean_final_activation_class
        self.cov_final_activation_class = cov_final_activation_class
        self.cov_estimation_mode = cov_estimation_mode
        # initializing modules
        self._init_modules()

    @property
    def decoder_input_dim(
        self,
    ) -> None:
        """
        Returns the dimensionality of the decoder input based on whether shared representation is used.

        :return: Dimensionality of the decoder input.
        :rtype: int
        """
        if self.use_shared_representation:
            return self.latent_dim
        return self.input_dim

    @property
    def cov_output_dim(self) -> int:
        """
        Returns the dimensionality of the covariance output based on the estimation mode.

        :return: Dimensionality of the covariance output.
        :rtype: int
        """
        if self.cov_estimation_mode == "isotropic":
            return 1
        elif self.cov_estimation_mode == "anisotropic":
            return self.output_dim

    def _init_modules(
        self,
    ) -> nn.Module:
        """
        Initializes the components of the model (encoder, mean network, and covariance network).

        This method initializes the encoder, mean, and covariance networks using the 
        specified parameters for architecture, batch normalization, dropout, and activation functions.
        """
        if self.use_shared_representation:
            self.encoder = MLPBlock(
                self.input_dim,
                self.latent_dim,
                hidden_dims=self.encoder_hidden_dims,
                use_batchnorm=self.encoder_use_batchnorm,
                use_dropout=self.encoder_use_dropout,
                dropout_rate=self.encoder_dropout_rate,
            )
        self.mean_net = MLPBlock(
            self.decoder_input_dim,
            self.output_dim,
            hidden_dims=self.mean_hidden_dims,
            use_batchnorm=self.mean_use_batchnorm,
            use_dropout=self.mean_use_dropout,
            dropout_rate=self.mean_dropout_rate,
            activation_class=self.mean_activation_class,
            final_activation_class=self.mean_final_activation_class,
        )
        self.cov_net = MLPBlock(
            self.decoder_input_dim,
            self.cov_output_dim,
            hidden_dims=self.cov_hidden_dims,
            use_batchnorm=self.cov_use_batchnorm,
            use_dropout=self.cov_use_dropout,
            dropout_rate=self.cov_dropout_rate,
            activation_class=self.cov_activation_class,
            final_activation_class=self.cov_final_activation_class,
        )

    def forward(self, input_tensor: Tensor) -> dict[str, Tensor]:
        """
        Forward pass through the model.

        This method computes the mean and covariance predictions for the input data.

        :param input_tensor: Input tensor to the model.
        :type input_tensor: Tensor
        :return: Dictionary containing predicted mean and covariance.
        :rtype: dict[str, Tensor]
        """
        if self.use_shared_representation:
            input_tensor = self.encoder(input_tensor)
        mean_hat = self.mean_net(input_tensor)
        cov_hat = self.cov_net(input_tensor)
        return {MEAN_KEY: mean_hat, COVARIANCE_KEY: cov_hat}
