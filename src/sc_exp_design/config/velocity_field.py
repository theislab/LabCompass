import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Literal

from torch import nn

from sc_exp_design.types import LayersDict

logger = logging.getLogger(__name__)

__all__ = ["NeuralVelocityFieldConfig"]


@dataclass(slots=True)
class NeuralVelocityFieldConfig:
    """Object for configuring :class:`NeuralVelocityField` objects.

    ## State Encoder Settings

    :param x_encoder_hidden_dims: The hidden dimensions for the state encoder.
        Sets the attribute :attr:`MLPBlock.hidden_dims` of :class:`NeuralVelocityField.x_encoder`, defaults to `(128, 64, 32)`.
    :type x_encoder_hidden_dims: class:`Sequence[int]`

    :param state_encoder_output_dim: The output dimensions for the state encoder.
        This represents the latent dimensionality of the space which the states will be embedded into.
        Sets the attribute :attr:`MLPBlock.output_dim` of :attr:`NeuralVelocityField.x_encoder`, defaults to `10`.
    :type state_encoder_output_dim: class:`int`

    :param x_encoder_use_batchnorm: Whether to use batch normalization when encoding the states.
        Sets the attribe :attr:`MLPBlock.use_batchnorm` of :attr:`NeuralVelocityField.x_encoder`, defaults to `False`.
    :type x_encoder_use_batchnorm: class:`bool`

    :param x_encoder_use_dropout: Whether to use dropout when encoding the states.
        Sets the attribe :attr:`MLPBlock.use_dropout` of :attr:`NeuralVelocityField.x_encoder`, defaults to `False`.
    :type x_encoder_use_dropout: class:`bool`

    :param x_encoder_dropout_rate: The dropout rate using when :attr:`.NeuralVelocityFieldConfig.x_encoder_use_dropout` is `True`.
        Sets the attribe :attr:`MLPBlock.dropout_rate` of :attr:`NeuralVelocityField.x_encoder`, defaults to `0.0`.
    :type x_encoder_dropout_rate: class:`float`

    :param x_encoder_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the hidden layers
        of the state encoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.activation_class` of :attr:`NeuralVelocityField.x_encoder`, defaults to `torch.nn.ELU`.
    :type x_encoder_activation_class: class:`torch.nn.Module`

    :param x_encoder_final_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the output (final) layer
        of the state encoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.final_activation_class` of :attr:`NeuralVelocityField.x_encoder`, defaults to `torch.nn.Identity`.
    :type x_encoder_final_activation_class: class:`torch.nn.Module`

    ## Time Encoder Settings

    :param encode_time: Whether to encode the current time step using a :class:`MLPBlock` or to simply pass it as it is (i.e.: Identity mapping),
        defaults to `False`.
    :type encode_time: class:`bool`

    :param time_encoder_input_dim: The input dimension for the (optional) time encoder.
        Sets the attribute :attr:`MLPBlock.input_dim` of `NeuralVelocityField.time_encoder`, defaults to `1`.
    :type time_encoder_hidden_dims: class:`int`

    :param time_encoder_hidden_dims: The hidden dimensions for the (optional) time encoder.
        Sets the attribute :attr:`MLPBlock.hidden_dims` of `NeuralVelocityField.time_encoder`, defaults to `(128, 64, 32)`.
    :type time_encoder_hidden_dims: class:`Sequence[int]`

    :param time_encoder_output_dim: The output dimensions for the (optional) time encoder.
        This represents the latent dimensionality of the space which the time step will be embedded into.
        Sets the attribute :attr:`MLPBlock.output_dim` of :attr:`NeuralVelocityField.time_encoder`, defaults to `10`.
    :type time_encoder_output_dim: class:`int`

    :param time_encoder_use_batchnorm: Whether to use batch normalization when encoding the time step.
        Sets the attribe :attr:`MLPBlock.use_batchnorm` of :attr:`NeuralVelocityField.time_encoder`, defaults to `False`.
    :type time_encoder_use_batchnorm: class:`bool`

    :param time_encoder_use_dropout: Whether to use dropout when encoding the encoding the time step.
        Sets the attribe :attr:`MLPBlock.use_dropout` of :attr:`NeuralVelocityField.time_encoder`, defaults to `False`.
    :type time_encoder_use_dropout: class:`bool`

    :param time_encoder_dropout_rate: The dropout rate using when :attr:`NeuralVelocityFieldConfig.time_encoder_use_dropout` is `True`.
        Sets the attribe :attr:`MLPBlock.dropout_rate` of :attr:`NeuralVelocityField.time_encoder`, defaults to `0.0`.
    :type time_encoder_dropout_rate: class:`float`

    :param time_encoder_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the hidden layers
        of the (optional) time step encoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.activation_class` of :attr:`NeuralVelocityField.time_encoder`, defaults to `torch.nn.ELU`.
    :type time_encoder_activation_class: class:`torch.nn.Module`

    :param time_encoder_final_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the output (final) layer
        of the (optional) time step encoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.final_activation_class` of :attr:`NeuralVelocityField.time_encoder`, defaults to `torch.nn.Identity`.
    :type time_encoder_final_activation_class: class:`torch.nn.Module`

    ## Condition Encoder Settings

    :param use_guidance: Whether to guide the velocity field by concatenating conditions to its inputs, defaults to `True`.
    :type use_guidance: class:`bool`

    :param perturbation_latent_dim: Latent dimensionality for the encoded conditions.
    :type perturbation_latent_dim: class:`int`

    :param perturbation_layers_before_pooling: Dictionary mapping each condition to be encoded to the configuration of its encoder.
        Each key of :attr:`.NeuralVelocityFieldConfig.condiion_layers_before_pooling` will be given by a :class:`str` with
        the identifier of the perturbation covariate to decode, while each value will be either an instance of :class:`LayersDict`,
        or a :class:`dict` that satisfies the following conditions:
        - It needs to contain a key named `"layer_type"`, with values either given by `"mlp"` or `"self_attention"`,
            which will respectively instantiate, for the given condition, an :class:`MLPBlock` or a :class:`SelfAttentionBlock`. The other key value pairs will be used
            to configure the encoder.
        - When `perturbation_layers_before_pooling[<PERTURBATION_COVARIATE_NAME>]["layer_type"] == "mlp"`, you need to specify at least two keys,
            required to instantiate a :class:`MLPBlock`, namely `"input_dim"` and `"output_dim"`. The other keys, in case not specified in the dictionary,
            will fall back to default values defined in :class:`MLPBlock`.
        - When `perturbation_layers_before_pooling[<PERTURBATION_COVARIATE_NAME>]["layer_type] == "self_attention"`, you need to specify at least one other key,
            required to instantiate a :class:`SelfAttentionBlock`, that is `"num_embeddings"`, that will specify the number of unique tokens in
            the lookup table for embedding categorical variables in a continuous representation. The other keys, in case not specified in the dictionary,
            will fall back to the default values defined in :class:`SelfAttentionBlock`.
        Defaults to `None`.
    :type perturbation_layers_before_pooling: class:`dict[str, LayersDict | dict[str, Any]] | None`

    :param perturbation_covariates_not_pooled: A sequence with the names of the perturbations covariates that are encoded (i.e.: they appear as keys
        in :attr:`NeuralVelocityFieldConfig.perturbation_layers_before_pooling`) and whose latent representation will be directly concatenated to the
        :attr:`ConditionEncoder.after_pooling`, thus skipping the pooling step, defaults to `None`.
    :type perturbation_covariates_not_pooled: class:`Sequence[str] | None`

    :param perturbation_pooling: The pooling method used for the perturbation covariates, defaults tp `"mean"`.
    :type perturbation_pooling: class:`Literal["mean", "self_attention"]`

    :param perturbation_pooling_kwargs: The keyword arguments used to instantiate the pooling layer in the condition encoder.
        Should contain key-value pairs corresponding to the possible attributes used in the initialization of an :class:`AttentionPooling` object.
        Sets the attribute :attr:`ConditionEncoder.pooling_kwargs` of the :attr:`NeuralVelocityField.condition_encoder`
        which are then used to initialize :attr:`ConditionEncoder.pooling_layer` in case :attr:`ConditionEncoder.pooling == "self_attention"`, defaults to `None`.
    :type perturbation_pooling_kwargs: class:`dict[str, Any] | None`

    :param perturbation_layers_after_pooling: Configuration for the condition decoder, used to initialize the :attr:`ConditionEncoder.after_pooling` attribute of
        :attr:`NeuralVelocityField.condition_encoder`. Should be either an instance of :class:`LayersDict`, or a :class:`dict` that satisfies the following conditions:
        - It needs to contain a key named `"layer_type"`, with values either given by `"mlp"` or `"self_attention"`,
            which will respectively instantiate, for the given condition, an :class:`MLPBlock` or a :class:`SelfAttentionBlock`. The other key value pairs will be used
            to configure the encoder.
        - When `perturbation_layers_before_pooling["layer_type"] == "mlp"`, you need to specify at least two keys,
            required to instantiate a :class:`MLPBlock`, namely `"input_dim"` and `"output_dim"`. The other keys, in case not specified in the dictionary,
            will fall back to default values defined in :class:`MLPBlock`.
        - When `perturbation_layers_before_pooling["layer_type] == "self_attention"`, you need to specify at least one other key,
            required to instantiate a :class:`SelfAttentionBlock`, that is `"num_embeddings"`, that will specify the number of unique tokens in
            the lookup table for embedding categorical variables in a continuous representation. The other keys, in case not specified in the dictionary,
            will fall back to the default values defined in :class:`SelfAttentionBlock`.
        Defaults to `None`.
    :type perturbation_layers_after_pooling: class:`LayersDict | None`

    ## Decoder Settings

    :param decoder_hidden_dims: The hidden dimensions for the decoder, mapping the latent representation to a velocity field on the target space.
        It will take as input the latent states, concatenated with the corresponding time step and possibly the (encoded) condition.
        Sets the attribute :attr:`MLPBlock.hidden_dims` of `NeuralVelocityField.decoder`, defaults to `(128, 64, 32)`.
    :type decoder_hidden_dims: class:`Sequence[int]`

    :param decoder_use_batchnorm: Whether to use batch normalization when decoding the latent states.
        Sets the attribe :attr:`MLPBlock.use_batchnorm` of :attr:`NeuralVelocityField.decoder`, defaults to `False`.
    :type decoder_use_batchnorm:

    :param decoder_use_dropout: Whether to use dropout when encoding the encoding the time step.
        Sets the attribe :attr:`MLPBlock.use_dropout` of :attr:`NeuralVelocityField.decoder`, defaults to `False`.
    :type decoder_use_dropout: class:`bool`

    :param decoder_dropout_rate: The dropout rate using when :attr:`NeuralVelocityFieldConfig.decoder` is `True`.
        Sets the attribe :attr:`MLPBlock.dropout_rate` of :attr:`NeuralVelocityField.decoder`, defaults to `0.0`.
    :type decoder_dropout_rate: class:`float`

    :param decoder_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the hidden layers
        of the (decoder. Note that you should pass a class and not an instance. Sets the attribe :attr:`MLPBlock.activation_class` of :attr:`NeuralVelocityField.decoder`,
        defaults to `torch.nn.ELU`.
    :type decoder_activation_class: class:`torch.nn.Module`

    :param decoder_final_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the output (final) layer
        of the decoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.final_activation_class` of :attr:`NeuralVelocityField.decoder`, defaults to `torch.nn.Identity`.
    :type decoder_final_activation_class: class:`torch.nn.Module`

    """

    flow_dim: int
    encode_state: bool = True
    state_encoder_output_dim: int = 10
    state_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    encode_time: bool = False
    use_sinusoidal_time_features: bool = False
    time_features_num_freqs: int = 128
    time_encoder_output_dim: int = 10
    time_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    use_guidance: bool = True
    encode_conditions: bool = False
    perturbation_latent_dim: int | None = None
    perturbation_layers_before_pooling: dict[str, LayersDict | dict[str, Any]] | None = None
    perturbation_covariates_not_pooled: Sequence[str] | None = None
    perturbation_pooling: Literal["mean", "self_attention"] = "mean"
    perturbation_pooling_kwargs: dict[str, Any] | None = None
    perturbation_layers_after_pooling: LayersDict | None = None
    decoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    use_source_as_condition: bool = False
    encode_source: bool = False
    source_latent_dim: int = 10
    souce_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    
    def __post_init__(self) -> None:
        """
        Compatibility checks and edits for a valid configuration 
        """
  
        # sanity check on condition encoder
        if self.use_guidance:
            msg = f"With {self.use_guidance=} you need to pass a dictionary in the proper format as the `self.perturbation_layers_before_pooling` attribute, found `None`"
            assert self.perturbation_layers_before_pooling is not None, msg
            if self.encode_conditions:
                msg = f"With {self.encode_conditions=} you need to pass an integer value as the `self.perturbation_latent_dim` attribute, found `None`"
                assert self.perturbation_latent_dim is not None, msg
                for condition, layers_dict in self.perturbation_layers_before_pooling.items():
                    if isinstance(layers_dict, dict):
                        LayersDict.verify_keys(layers_dict)
                        layers_dict = LayersDict(**layers_dict)
                    msg = f"`layers_dict` is expected to be an instance of `LayersDict`, found {type(layers_dict)}"
                    assert isinstance(layers_dict, LayersDict), msg
                    self.perturbation_layers_before_pooling[condition] = layers_dict
                msg = f"With {self.encode_conditions=} you need to pass a dictionary in the proper format as the `self.perturbation_layers_after_pooling` attribute, found `None`"
                assert self.perturbation_layers_after_pooling is not None, msg
                if isinstance(self.perturbation_layers_after_pooling, dict):
                    self.perturbation_layers_after_pooling["input_dim"] = self.perturbation_layers_after_pooling_input_dim
                    self.perturbation_layers_after_pooling["output_dim"] = self.perturbation_latent_dim
                    LayersDict.verify_keys(self.perturbation_layers_after_pooling)
                    self.perturbation_layers_after_pooling = LayersDict(**self.perturbation_layers_after_pooling)
                msg = f"`self.perturbation_layers_after_pooling` is expected to be an instance of `LayersDict`, found {type(self.perturbation_layers_after_pooling)}"
                assert isinstance(self.perturbation_layers_after_pooling, LayersDict), msg
        else:
            msg = f"With {self.use_guidance=} an unguided flow model will be initialized, thus the settings for the condition encoder will be ignored."
            logger.warning(msg)

    @property
    def time_encoder_input_dim(
        self,
    ) -> int:
        """"""
        if self.use_sinusoidal_time_features:
            return self.time_features_num_freqs*2
        return 1

    @property
    def condition_input_dim(
        self,
    ) -> int | None:
        """
        Collect the condition input dimensions for the encoding process from the configuration.
        """
        dim = 0
        for condition, layers_dict in self.perturbation_layers_before_pooling.items():
            if isinstance(layers_dict, LayersDict):
                layers_dict = vars(layers_dict)
            if self.encode_conditions:
                if layers_dict["layer_type"] == "mlp":
                    input_dim = layers_dict["input_dim"]
                elif layers_dict["layer_type"] == "self_attention":
                    input_dim = layers_dict["num_embeddings"]
            else:
                input_dim = layers_dict["input_dim"]
            dim = dim + input_dim
        return dim

    @property
    def perturbation_layers_after_pooling_input_dim(
        self,
    ) -> int | None:
        """
        Collect the condition input dimensions after pooling.
        """
        dim = 0
        for condition, layers_dict in self.perturbation_layers_before_pooling.items():
            if isinstance(layers_dict, LayersDict):
                layers_dict = vars(layers_dict)
            if layers_dict["layer_type"] == "mlp":
                output_dim = layers_dict["output_dim"]
            elif layers_dict["layer_type"] == "self_attention":
                output_dim = layers_dict["embed_dim"][-1]
            dim = dim + output_dim
        return dim

    @property
    def joint_latent_dim(
        self,
    ) -> int:
        """
        Collect the dimension of the joint latent space (concatenating time, latent feature dimensions and perturbation)
        """
        # perturbations
        perturbation_latent_dim = 0
        if self.use_guidance and self.encode_conditions:
            perturbation_latent_dim = self.perturbation_latent_dim
        elif self.use_guidance and (not self.encode_conditions):
            perturbation_latent_dim = self.condition_input_dim
        # time
        time_latent_dim = self.time_encoder_input_dim
        if self.encode_time:
            time_latent_dim = self.time_encoder_output_dim
        # states
        state_latent_dim = self.flow_dim
        if self.encode_state:
            state_latent_dim = self.state_encoder_output_dim 
        # source
        source_latent_dim = 0
        if self.use_source_as_condition:
            source_latent_dim = self.flow_dim
            if self.encode_source:
                source_latent_dim = self.source_latent_dim
        return state_latent_dim + time_latent_dim + perturbation_latent_dim + source_latent_dim

    @property
    def joint_original_dim(
        self,
    ) -> int:
        """
        Collect dimensionality in the original space
        """
        perturbation_dim = 0
        if self.use_guidance:
            perturbation_dim = self.condition_input_dim
        source_dim = 0
        if self.use_source_as_condition:
            source_dim = self.flow_dim
        return self.flow_dim + self.time_encoder_input_dim + perturbation_dim + source_dim

    @property
    def initialize_source_encoder(
        self,
    ) -> bool:
        """"""
        if self.use_source_as_condition:
            if self.encode_source:
                return True
        return False
