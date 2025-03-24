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

    ## Score Field Settings

    :param learn_score_field: Whether to lean an approximation to the score while learning the velocity field. The score
        can then be used to turn the ODE in an SDE by equating the Continuity Equation (ODE) to the corresponding
        Forward Fokker-Plank Equation, for the same initial conditions. While the trajectory will differ,
        the probability paths will remain unchanged, thus theoretically obtaining the same distributional
        properties for the generated data. In case initialized, it will share the same latent representation
        as the one used for the velocity field (i.e.: it will take as an input the latent states, concatenated with the current time step
        and possibly the (encoded) conditions), defaults to `"False"`.
    :type learn_score_field: class:`bool`

    :param score_field_freeze_grads: Whether to backpropagate the gradients to the upstream modules, thus optimizing also the latent reprsentation
        for the task of approximating the score field along with the velocity field, deafults to `True`.
    :type score_field_freeze_grads: class:`bool`

    :param score_hidden_dims: The hidden dimensions for the (optional) score encoder.
        Sets the attribute :attr:`MLPBlock.hidden_dims` of `NeuralVelocityField.score_decoder`, defaults to `(128, 64, 32)`.
    :type score_hidden_dims: class:`Sequence[int]`

    :param score_use_batchnorm: Whether to use batch normalization when decoding the joint latent representation to the approximate score field.
        Sets the attribe :attr:`MLPBlock.use_batchnorm` of :attr:`NeuralVelocityField.score_decoder`, defaults to `False`.
    :type score_use_batchnorm: class:`bool`

    :param score_use_dropout: Whether to use dropout when decoding the joint latent representation to the approximate score field.
        Sets the attribe :attr:`MLPBlock.use_dropout` of :attr:`NeuralVelocityField.score_decoder`, defaults to `False`.
    :type score_use_dropout: class:`bool`

    :param score_dropout_rate: The dropout rate using when :attr:`NeuralVelocityFieldConfig.score_use_dropout` is `True`.
        Sets the attribe :attr:`MLPBlock.dropout_rate` of :attr:`NeuralVelocityField.score_decoder`, defaults to `0.0`.
    :type score_dropout_rate: class:`float`

    :param score_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the hidden layers
        of the (optional) score field decoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.activation_class` of :attr:`NeuralVelocityField.score_decoder`, defaults to `torch.nn.ELU`.
    :type score_activation_class: class:`torch.nn.Module`

    :param score_final_activation_class: A reference to a :class:`torch.nn.Module` used as activation for the output (final) layer
        of the (optional) score field decoder. Note that you should pass a class and not an instance.
        Sets the attribe :attr:`MLPBlock.final_activation_class` of :attr:`NeuralVelocityField.score_decoder`, defaults to `torch.nn.Identity`.
    :type score_final_activation_class: class:`torch.nn.Module`

    ## Endpoints Inference Settings

    :param lean_posterior_on_cond_vars: Whether to learn an approximate posterior distribution over the conditioning variables (i.e.: endpoints)
        of the flow, defaults to `False`.
    :type lean_posterior_on_cond_vars: class:`bool`

    :param endpoints_approximate_posterior_freeze_grads: Whether to backpropagate the gradients to the upstream modules, thus optimizing also the latent reprsentation
        for the task of approximating a posterior on the conditioning variables (endpoints). It will only have and effect when using the latent representation,
        (i.e.: :attr:`NeuralVelocityFieldConfig.endpoints_approximate_posterior_use_latent_repr` is `True`), deafults to `True`.
    :type endpoints_approximate_posterior_freeze_grads: class:`bool`

    :param endpoints_approximate_posterior_use_latent_repr: Whether to use the latent representation or the orignal one when doing inference
        on the conditioning variables (endpoints), defaults to `True`.
    :type endpoints_approximate_posterior_use_latent_repr: class:`bool`

    :param src_noise_model: The noise model used to probabilistically decode the source state, defaults to `"gaussian"`.
    :type src_noise_model: class:`Literal["gaussian", "neg_bin"]`

    :param src_approximate_posterior_kwargs: Dictionary with key-value pairs given by the optional keyword arguments for the neural noise model used to decode the source state.
        Only supports Gaussian Noise Models for now. The :attr:`<NoiseModel>.latent_dim` will be inferred internally
        by the :class:`NeuralVelocityField` from either :attr:`NeuralVelocityField.joint_original_dim`, when :attr:`NeuralVelocityFieldConfig.src_approximate_posterior_kwargs`
        is `False`, or from :attr:`NeuralVelocityField.joint_latent_dim` otherwise, while :attr:`<NoiseModel>.output_dim` will be simply given by
        :attr:`NeuralVelocityField.flow_dim` (that is, we assume for now that source and target lie in the same space as the flow, which may not always be the case, for example
        when using GENOT the source may lie in a different space).
    :type src_approximate_posterior_kwargs: class:`dict[str, Any] | None`

    :param tgt_noise_model: The noise model used to probabilistically decode the source state, defaults to `"gaussian"`.
    :type tgt_noise_model: class:`Literal["gaussian", "neg_bin"]`

    :param tgt_approximate_posterior_kwargs: Dictionary with key-value pairs given by the optional keyword arguments for the neural noise model used to decode the source state.
        Only supports Gaussian Noise Models for now. The :attr:`<NoiseModel>.latent_dim` will be inferred internally
        by the :class:`NeuralVelocityField` from either :attr:`NeuralVelocityField.joint_original_dim`, when :attr:`NeuralVelocityFieldConfig.src_approximate_posterior_kwargs`
        is `False`, or from :attr:`NeuralVelocityField.joint_latent_dim` otherwise, while :attr:`<NoiseModel>.output_dim` will be simply given by
        :attr:`NeuralVelocityField.flow_dim`.
    :type tgt_approximate_posterior_kwargs: class:`dict[str, Any]`

    ## Perturbation Inference Settings

    :param lean_posterior_on_perts: Whether to lean an approximate posterior distribution on some target representation of the perturbations.é
    :type lean_posterior_on_perts: class:`bool`

    :param pert_approximate_posterior_freeze_grads: Whether to backpropagate the gradients to the upstream modules, thus optimizing also the latent reprsentation
        for the task of approximating a posterior on the target perturbation covariate, defaults to `True`.
    :type pert_approximate_posterior_freeze_grads: class:`bool`

    :param pert_approximate_posterior_input_type: The input used to predict the posterior distribution on the applied perturbations, defaults to `"endpoints"`.
    :type pert_approximate_posterior_input_type: class:`Literal["latent", "original", "endpoints", "one_step_prediction"]`

    :param pert_target_covariates_output_dims: Dictionary specifying the output dimensionality for each of the perturbation target covariates.

        Should have the same keys as :attr:`NeuralVelocityFieldConfig.pert_noise_model`, :attr:`NeuralVelocityFieldConfig.pert_approximate_posterior_kwargs`
        and :attr:`NeuralVelocityFieldConfig.pert_cov_estimation_modes`, throws an :class:`AssertionError` otherwise, defaults to `None`.
    :type pert_target_covariates_output_dims: class:`dict[str, int] | None`

    :param pert_noise_model: Dictionary specifying the noise model used to decode each target perturbation covariate.

        Should have the same keys as :attr:`NeuralVelocityFieldConfig.pert_target_covariates_output_dims`, :attr:`NeuralVelocityFieldConfig.pert_approximate_posterior_kwargs`
        and :attr:`NeuralVelocityFieldConfig.pert_cov_estimation_modes`, throws an :class:`AssertionError` otherwise, defaults to `None`.
    :type pert_noise_model: class:`dict[str, Literal["gaussian", "neg_bin"]] | None`

    :param pert_approximate_posterior_kwargs: Dictionary with keys given by a :class:`str` with the identifier of the
        target perturbation covariate to decode and values given by dictionary specifying the optional keyword arguments for the neural noise model used to decode
        target perturbation covariate. Only supports Gaussian Noise Models for now. The :attr:`<NoiseModel>.latent_dim` will be inferred internally
        by the :class:`NeuralVelocityField` from either :attr:`NeuralVelocityField.joint_original_dim`, when :attr:`NeuralVelocityFieldConfig.src_approximate_posterior_kwargs`
        is `False`, or from :attr:`NeuralVelocityField.joint_latent_dim` otherwise, while :attr:`<NoiseModel>.output_dim` will be simply given by
        the corresponding value in :attr:`NeuralVelocityFieldConfig.pert_target_covariates_output_dims`.

        Should have the same keys as :attr:`NeuralVelocityFieldConfig.pert_target_covariates_output_dims`, :attr:`NeuralVelocityFieldConfig.pert_noise_model`
        and :attr:`NeuralVelocityFieldConfig.pert_cov_estimation_modes`, throws an :class:`AssertionError` otherwise, defaults to `None`.
    :type pert_approximate_posterior_kwargs: class:`dict[str, dict[str, Any]] | None`

    :param pert_cov_estimation_modes: Specifies the covariance estimation mode when decoding the target perturbation covariates using a Gaussian noise model.

        Should have the same keys as :attr:`NeuralVelocityFieldConfig.pert_target_covariates_output_dims`, :attr:`NeuralVelocityFieldConfig.pert_noise_model`
        and :attr:`NeuralVelocityFieldConfig.pert_approximate_posterior_kwargs`, throws an :class:`AssertionError` otherwise, defaults to `None`.
    :type pert_cov_estimation_modes: class:`dict[str, Literal["isotropic", "anisotropic"] | None] | None`

    ## Latent Perturbation Inference Settings

    :param learn_posterior_on_latent_perts: Whether to lean an approximate posterior on the latent perturbations by using the endpoints (conditioning variables).
        Only effective when :attr:`NeuralVelocityFieldConfig.encode_conditions` is `True`.
        For now, it only supports the use of :class:`MLPGaussianNoiseModel`, defaults to `False`.
    :type learn_posterior_on_latent_perts: class:`bool`

    :param latent_perts_posterior_freeze_grads: Whether to backpropagate the gradients to the upstream modules, thus optimizing also the latent reprsentation
        for the task of approximating a posterior on the latent representation for the conditions, defaults to `True`.
    :type latent_perts_posterior_freeze_grads: class:`bool`

    :param latent_perts_approximate_posterior_kwargs: The :attr:`<NoiseModel>.latent_dim` will be inferred internally
        by the :class:`NeuralVelocityField` from :attr:`NeuralVelocityField.flow_dim` (namely `2*velocity_field.flow_dim`),
        while :attr:`<NoiseModel>.output_dim` will be simply given by :attr:`ConditionEncoder.latent_dim` of :attr:`NeuralVelocityField.condition_encoder`, defaults to `None`.
    :type latent_perts_approximate_posterior_kwargs: class`dict[str, Any]`
    """

    flow_dim: int
    encode_state: bool = True
    state_encoder_output_dim: int = 10
    state_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    encode_time: bool = False
    time_encoder_input_dim: int = 1
    time_encoder_output_dim: int = None 
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
    learn_score_field: bool = False
    score_field_freeze_grads: bool = True
    score_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    learn_posterior_on_cond_vars: bool = False
    endpoints_approximate_posterior_freeze_grads: bool = True
    endpoints_approximate_posterior_use_latent_repr: bool = True
    src_noise_model: Literal["gaussian", "neg_bin"] = "gaussian"
    src_approximate_posterior_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    tgt_noise_model: Literal["gaussian", "neg_bin"] = "gaussian"
    tgt_approximate_posterior_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    learn_posterior_on_perts: bool = False
    pert_approximate_posterior_freeze_grads: bool = True
    pert_approximate_posterior_input_type: Literal["latent", "original", "endpoints", "one_step_prediction"] = "endpoints"
    pert_target_covariates_output_dims: dict[str, int] | None = None
    pert_noise_model: dict[str, Literal["gaussian", "neg_bin"]] | None = None
    pert_approximate_posterior_kwargs: dict[str, dict[str, Any]] | None = None
    pert_cov_estimation_modes: dict[str, Literal["isotropic", "anisotropic"] | None] = None
    learn_posterior_on_latent_perts: bool = False
    latent_perts_posterior_freeze_grads: bool = True
    latent_perts_approximate_posterior_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})

    def __post_init__(self) -> None:
        """
        Compatibility checks and edits for a valid configuration 
        """
        # sanity checks posterior latent perturbations
        if self.learn_posterior_on_latent_perts and (not self.encode_conditions):
            msg = f"With {self.learn_posterior_on_latent_perts=}, `self.encode_condition` should be set to `True`, found {self.encode_conditions}. Setting `learn_posterior_on_latent_perts` to `False`."
            logger.warning(msg)
            self.learn_posterior_on_latent_perts = False
            
        # sanity check posterior on perturbations
        if self.learn_posterior_on_perts:
            msg = f"With {self.learn_posterior_on_perts=}, you need to pass the target output dims in `self.pert_target_covariates_output_dims`, found `None`."
            assert self.pert_target_covariates_output_dims is not None, msg
            if self.pert_noise_model is None:
                msg = "`self.pert_noise_model` is `None`. Setting it to a dictionary with the same keys as `self.pert_target_covariates_output_dims` and values `None` (i.e.: simply using `MLPBlock`)"
                logger.warning(msg)
                self.pert_noise_model = {key: None for key in self.pert_target_covariates_output_dims.keys()}
            if self.pert_approximate_posterior_kwargs is None:
                msg = "`self.pert_approximate_posterior_kwargs` is `None`. Setting it to a dictionary with the same keys as `self.pert_target_covariates_output_dims` and emty dictionaries as values (i.e.: ising default class configurations)"
                logger.warning(msg)
                self.pert_approximate_posterior_kwargs = {
                    key: {} for key in self.pert_target_covariates_output_dims.keys()
                }
            if self.pert_cov_estimation_modes is None:
                msg = "`self.pert_approximate_posterior_kwargs` is `None`. Setting it to a dictionary with the same keys as `self.pert_target_covariates_output_dims` and emty dictionaries as values (i.e.: ising default class configurations)"
                logger.warning(msg)
                self.pert_cov_estimation_modes = {key: None for key in self.pert_target_covariates_output_dims.keys()}
            msg = f"Dictionaries `self.pert_approximate_posterior_kwargs`, `self.pert_target_covariates_output_dims` and `self.pert_noise_model` are expected to have the same keys (found {self.pert_approximate_posterior_kwargs.keys()=}, {self.pert_target_covariates_output_dims.keys()=}, {self.pert_noise_model.keys()=})"
            assert set(self.pert_approximate_posterior_kwargs.keys()) == set(
                self.pert_target_covariates_output_dims.keys()
            ), msg
            assert set(self.pert_approximate_posterior_kwargs.keys()) == set(self.pert_noise_model.keys()), msg
            
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
        perturbation_latent_dim = 0
        if self.use_guidance and self.encode_conditions:
            perturbation_latent_dim = self.perturbation_latent_dim
        elif self.use_guidance and (not self.encode_conditions):
            perturbation_latent_dim = self.condition_input_dim
        time_latent_dim = self.time_encoder_input_dim
        if self.encode_time:
            time_latent_dim = self.time_encoder_output_dim
        if self.encode_state:
            return self.state_encoder_output_dim + time_latent_dim + perturbation_latent_dim
        return self.flow_dim + time_latent_dim + perturbation_latent_dim

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
        return self.flow_dim + self.time_encoder_input_dim + perturbation_dim

    @property
    def cond_vars_input_dim(
        self,
    ) -> int:
        """
        Computes the input dimension for the inference network on conditioning variables.
        
        Returns:
            int: Input dimension for conditioning variable inference.
        """
        # retrieving the input dimension for the inference network on the conditioning variables
        if self.endpoints_approximate_posterior_use_latent_repr:
            return self.joint_latent_dim
        return self.joint_original_dim
        
    @property
    def pert_input_dim(
        self,
    ) -> int:
        """
        Computes the input dimension for the inference network on perturbations.
        
        Returns:
            int: Input dimension for perturbation inference.
        
        Raises:
            ValueError: If an unsupported perturbation input type is provided.
        """
        # retrieving the input dimension for the inference network on the conditioning variables
        if self.pert_approximate_posterior_input_type == "latent":
            return self.joint_latent_dim
        elif self.pert_approximate_posterior_input_type in ["endpoints", "one_step_prediction"]:
            return self.flow_dim * 2
        elif self.pert_approximate_posterior_input_type == "original":
            return self.joint_original_dim
        else:
            msg = f"{self.pert_approximate_posterior_input_type=} is not supported, possible values are `['latent', 'endpoints', 'one_step_prediction', 'original']`"
            raise ValueError(msg)

