from collections.abc import Sequence
from typing import Literal, Any

from torch import Tensor, nn

from sc_exp_design.constants import ParamsFields
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
        use_shared_representation: bool = True,
        cov_estimation_mode: Literal["isotropic", "anisotropic"] = "isotropic",
        encoder_mlp_kwargs: dict[str, Any] | None = None,
        mean_mlp_kwargs: dict[str, Any] | None = None,
        cov_mlp_kwargs: dict[str, Any] | None = None,
    ) -> None:
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.use_shared_representation = use_shared_representation
        self.cov_estimation_mode = cov_estimation_mode
        
        if encoder_mlp_kwargs is None:
            encoder_mlp_kwargs = {}
        self.encoder_mlp_kwargs = encoder_mlp_kwargs
        
        if mean_mlp_kwargs is None:
            mean_mlp_kwargs = {}
        self.mean_mlp_kwargs = mean_mlp_kwargs

        if cov_mlp_kwargs is None:
            cov_mlp_kwargs = {}
        self.cov_mlp_kwargs = cov_mlp_kwargs
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
        modules = {}
        if self.use_shared_representation:
            modules["encoder"] = MLPBlock(
                self.input_dim,
                self.latent_dim,
                **self.encoder_mlp_kwargs,
            )
        modules["mean_net"] = MLPBlock(
            self.decoder_input_dim,
            self.output_dim,
            **self.mean_mlp_kwargs,
        )
        modules["cov_net"] = MLPBlock(
            self.decoder_input_dim,
            self.cov_output_dim,
            **self.cov_mlp_kwargs,
        )
        self.model_modules = nn.ModuleDict(modules)

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
            input_tensor = self.model_modules["encoder"](input_tensor)
        mean_hat = self.model_modules["mean_net"](input_tensor)
        cov_hat = self.model_modules["cov_net"](input_tensor)
        return {ParamsFields.MEAN: mean_hat, ParamsFields.COVARIANCE: cov_hat}
