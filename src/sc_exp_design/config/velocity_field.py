import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dc_field
from functools import partial
from typing import Any, Literal

from sc_exp_design.types import MLPConfigFields
from sc_exp_design.utils import get_conditions_to_pool

logger = logging.getLogger(__name__)

__all__ = ["NeuralVelocityFieldConfig"]


@dataclass(slots=True)
class NeuralVelocityFieldConfig:
    """Object for configuring :class:`NeuralVelocityField` objects.

    :param flow_dim: The dimensionality of the flow. Needs to be set by the user.
    :type flow_dim: class: `int`

    :param encode_state: Whether to encode separately the state on which to compute the velocity field using an :class: `MLPBlock`,
        or simply concatenate it directly with time and conditions before being given as input to the VF decoder. 
    :type encode_state: class: `bool`

    :param state_encoder_output_dim: The output dimensions for the state encoder.
        This represents the dimensionality of the latent space which the states will be embedded into.
        Sets the attribute :attr:`MLPBlock.output_dim` of :attr:`NeuralVelocityField.x_encoder`.
        Only used if :param: `encode_state` is `True`, defaults to `10`.
    :type state_encoder_output_dim: class:`int`

    :param state_encoder_mlp_kwargs: Dictionary containing the configurations for the state encoder :class:`MLPBlock`.
        If provided, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
        It should NOT contain neither the `"input_dim"` nor the `"output_dim"` keys, as these will be respectively taken
        from :param: `flow_dim` and :param: `state_encoder_output_dim`. Only used if :param: `encode_state` is `True`, defaults to an empty dictionary.
    :type state_encoder_mlp_kwargs: class: `dict[str, Any]`

    :param encode_time: Whether to encode the current time step using a :class:`MLPBlock` or to simply pass it as it is (i.e.: Identity mapping), defaults to `False`.
    :type encode_time: class:`bool`

    :param use_sinusoidal_time_features: Whether to use the sinusoidal  features for the time representation, defaults to `False`.
    :type use_sinusoidal_time_features: class:`bool`

    :param time_features_num_freqs: The number of frequencies used to compute the sinusoidal time features, only used if :param: `use_sinusoidal_time_features` is `True`. Defaults to `128`.
    :type time_features_num_freqs: class: `int`

    :param time_features_max_periods: The maximum number of periods used to compute the sinusoidal time features, only used if :param: `use_sinusoidal_time_features` is `True`. Defaults to `10000`.
    :type time_features_max_periods: class: `int`

    :param time_encoder_output_dim: The output dimensions for the time encoder.
        This represents the dimensionality of the latent space the time step will be embedded into.
        Sets the attribute :attr:`MLPBlock.output_dim` of :attr:`NeuralVelocityField.time_encoder`.
        Only used if :param: `encode_time` is `True`, defaults to `10`.
    :type time_encoder_output_dim: class:`int`

    :param time_encoder_mlp_kwargs: Dictionary containing the configurations for the time encoder :class:`MLPBlock`.
        If provided, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
        It should NOT contain neither the `"input_dim"` nor the `"output_dim"` keys, as these will be respectively taken
        from :attr: `NeuralVelocityFieldConfig.time_encoder_input_dim` and :param: `time_encoder_output_dim`.
        Only used if :param: `encode_time` is `True`, defaults to an empty dictionary.
    :type time_encoder_mlp_kwargs: class: `dict[str, Any]`

    :param use_guidance: Whether to guide the velocity field by concatenating conditions to its inputs, defaults to `True`.
    :type use_guidance: class:`bool`

    :param encode_conditions: Whether to encode the guidance conditions using the :class: `ConditionEncoder` object from :module: `sc_exp_design.networks`, defaults to `False`.
    :type encode_conditions:

    :param perturbation_encoder_output_dim: Latent dimensionality for the encoded conditions. Only used if :param: `encode_conditions` is set to `True`, defaults to `10`.
    :type perturbation_encoder_output_dim: class:`int`

    :param perturbation_layers_before_pooling: Dictionary mapping each condition to be encoded to the configuration of its encoder.
        Each key of :attr:`.NeuralVelocityFieldConfig.condiion_layers_before_pooling` will be given by a :class:`str` with
        the identifier of the perturbation covariate to be modeled, while each value will be either:
        - When :param: `encode_conditions` is set to `True`, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
            In this case, you would need to pass both the "input_dim" and "output_dim" key to such dictionaries, as there is not other way to correctly pass this information.
        - When :param: `encode_conditions` is set to `True`, it needs to contain the `"input_dim"` key, specifying the dimensionality of each conditon to be concatenated as input
            to the velocity field decoder.
        This needs to include all the conditions that are desired to be modeled for the VF guidance. That is, each key of :param: `perturbation_layers_before_pooling` represents a
        perturbation whose effect on cells is modeled. Importantly, the output dimension of all the perturbations to be pooled and thate hence do not appear in :param: `perturbation_covariates_not_pulles`
        needs to be the same, as this is required to allow stacking the condition data over a new dimension before proceeding with the pooling.
        Defaults to `None`.
    :type perturbation_layers_before_pooling: class:`dict[str, dict[str, Any]] | None`

    :param perturbation_covariates_not_pooled: A sequence with the names of the perturbations covariates that are encoded (i.e.: they appear as keys
        in :attr:`NeuralVelocityFieldConfig.perturbation_layers_before_pooling`) and whose latent representation will be directly concatenated to the
        :attr:`ConditionEncoder.after_pooling`, thus skipping the pooling step, defaults to `None`.
    :type perturbation_covariates_not_pooled: class:`Sequence[str] | None`

    :param perturbation_pooling: The pooling method used for the perturbation covariates in :class: `ConditionEncoder`, defaults to `"mean"`.
    :type perturbation_pooling: class:`Literal["mean", "sum", "self_attention"]`

    :param perturbation_pooling_kwargs: The keyword arguments used to instantiate the pooling layer in the condition encoder.
        Should contain key-value pairs corresponding to the possible attributes used in the initialization of an :class:`AttentionPooling` object.
        Sets the attribute :attr:`ConditionEncoder.pooling_kwargs` of the :attr:`NeuralVelocityField.condition_encoder`
        which are then used to initialize :attr:`ConditionEncoder.pooling_layer` in case :attr:`ConditionEncoder.pooling == "self_attention"`, defaults to `None`.
    :type perturbation_pooling_kwargs: class:`dict[str, Any] | None`

    :param perturbation_layers_after_pooling: Configuration for the condition decoder, used to initialize the :attr:`ConditionEncoder.after_pooling` attribute of
        :attr:`NeuralVelocityField.condition_encoder`. Should be a :class:`dict`. If provided, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
        It should NOT contain neither the `"input_dim"` nor the `"output_dim"` keys, as these will be respectively taken
        from :attr: `NeuralVelocityFieldConfig.perturbation_layers_after_pooling_input_dim` and :param: `perturbation_encoder_output_dim`.
        Defaults to `None`.
    :type perturbation_layers_after_pooling: class:`MLPConfigFields | None`

    :param decoder_mlp_kwargs: Dictionary containing the configurations for the velocity decoder :class:`MLPBlock`.
        If provided, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
        It should NOT contain neither the `"input_dim"` nor the `"output_dim"` keys, as these will be respectively taken
        from :attr: `NeuralVelocityFieldConfig.decoder_input_dim` and :param: `flow_dim`.
        It is the only setting that is used in any case, as it is the one that ultimately computes the output value of :class: `NeuralVelocityField`, defaults to an empty dictionary.
    :type time_encoder_mlp_kwargs: class: `dict[str, Any]`

    :type use_source_as_condition: Whether to encode the information about the source states in the computation of the velocity field.
        This should be set to `True` when generating from noise samples and there exists a notion of control states. In such case it would be the only way to incorporate the information
        about the source states, as the interpolation will happen between noise samples and samples from the target distribution. Defaults to `False`.
    :type use_source_as_condition: class: `bool`

    :type encode_source: Whether to encode the information about the source using a separate :class: `MLPBlock` before concatenation with state, time and condition in the latent space.
        When `False` (and :param: `use_source_as_condition` is `True`), source states will be directly concatenated. 
        Only used if :param: `use_source_as_condition` is `True`, defaults to `False`.
    :type encode_source: class: `bool`

    :param source_encoder_output_dim: The output dimensions for the source encoder.
        This represents the latent dimensionality of the source states will be embedded into.
        Sets the attribute :attr:`MLPBlock.output_dim` of :attr:`NeuralVelocityField.time_encoder`.
        Only used if :param: `encode_source` is `True`, defaults to `10`.
    :type source_encoder_output_dim: class: `int`

    :param source_encoder_mlp_kwargs: Dictionary containing the configurations for the source encoder :class:`MLPBlock`.
        If provided, it needs to specify the requirements defined by the :class: `MLPConfigFields` of :module: `sc_exp_design.types`.
        It should NOT contain neither the `"input_dim"` nor the `"output_dim"` keys, as these will be respectively taken
        from :param: `flow_dim` and :param: `source_encoder_output_dim`.
        Only used if :param: `encode_source` is `True`, defaults to an empty dictionary.
    :type source_encoder_mlp_kwargs: class: `dict[str, Any]`

    :param conditioning_type: The conditioning strategy for the velocity field.
        When "resnet", it is required to have :param: `encode_state` set to `True`.
        When "film", it is also required to use use guidance by setting :param: `use_guidance` to `True`. Defaults to "concantenation"
    :type conditioning_type: class: `Literal["concatenation", "resnet", "film"]`

    :param n_resnet_blocks: The number of residual network blocks to be instanciated.
        Only used when :param: `use_resnet_blocks` is `True`, defaults to `3`.
    :type n_resnet_blocks: class: `int`

    :param resnet_dropout_prob: The dropout rate used by the residual network.
        Only used when :param: `use_resnet_blocks` is `True`, defaults to `0.0`.
    :type resnet_dropout_prob: class: `float`

    :param resnet_normalization: The normalization used by the residual network.
        Only used when :param: `use_resnet_blocks` is `True`, defaults to `None`.
    :type resnet_normalization: class: `Literal["layer", "batch"] | None`
    
    :param use_classifier_free_guidance: Whether to use classifier-free guidance, defaults to ´False´.
    :type use_classifier_free_guidance: class: `bool`

    :param cfg_null_condition_token: The value used to mask the condition token with when using classifier-free guidance, defaults to -1.0.
    :type cfg_null_condition_token: class: `float`
    """

    flow_dim: int
    encode_state: bool = True
    state_encoder_output_dim: int = 10
    state_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    encode_time: bool = False
    use_sinusoidal_time_features: bool = False
    time_features_num_freqs: int = 128
    time_features_max_periods: int = 10000
    time_encoder_output_dim: int = 10
    time_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    use_guidance: bool = True
    encode_conditions: bool = False
    perturbation_encoder_output_dim: int = 10
    perturbation_layers_before_pooling: dict[str, dict[str, Any]] | None = None
    perturbation_covariates_not_pooled: Sequence[str] | None = None
    perturbation_pooling: Literal["mean", "sum", "self_attention"] = "mean"
    perturbation_pooling_kwargs: dict[str, Any] | None = None
    perturbation_layers_after_pooling: dict[str, Any] | None = dc_field(default_factory=lambda: {})
    perturbation_output_dropout: float = 0.0
    decoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})
    use_source_as_condition: bool = False
    encode_source: bool = False
    source_encoder_output_dim: int = 10
    source_encoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {},)
    conditioning_type: Literal["concatenation", "resnet", "film"] = "concatenation"
    n_resnet_blocks: int = 3
    resnet_dropout_prob: float = 0.0
    resnet_normalization: Literal["layer", "batch"] | None = None
    use_classifier_free_guidance: bool = False
    cfg_null_condition_token: float = -1.0
    
    def __post_init__(self) -> None:
        """
        Compatibility checks and edits for a valid configuration. It performs the following checks:

        * Verifies the key word arguments for all :class: `MLPBlock` configurations using :class: `MLPConfigFields`.
        * Makes sure that when using the residual network the :attr:`NeuralVelocityFieldConfig.encode_state` is set to `True`.
        * When :attr: `NeuralVelocityFieldConfig.use_guidance` is set to `True` it verifies that the :attr: `NeuralVelocityFieldConfig.perturbation_layers_before_pooling` is not None.
        * When :attr: `NeuralVelocityFieldConfig.encode_conditions` is set to `False` it verifies that the :attr: `NeuralVelocityFieldConfig.perturbation_layers_before_pooling` is properly set.
        * When :attr: `NeuralVelocityFieldConfig.encode_conditions` is set to `True` it also verifies that the :attr: `NeuralVelocityFieldConfig.perturbation_layers_after_pooling` is properly set.
            It also sets the "input_dim" and "output_dim" keys using respectively :attr: `NeuralVelocityFieldConfig.perturbation_layers_after_pooling_input_dim` and :attr: `NeuralVelocityFieldConfig.perturbation_encoder_output_dim`.

        """
        # sanity check on mlp configurations
        MLPConfigFields.verify_keys(self.state_encoder_mlp_kwargs, require_input_dim_key=False, require_output_dim_key=False)
        MLPConfigFields.verify_keys(self.time_encoder_mlp_kwargs, require_input_dim_key=False, require_output_dim_key=False)
        MLPConfigFields.verify_keys(self.decoder_mlp_kwargs, require_input_dim_key=False, require_output_dim_key=False)
        MLPConfigFields.verify_keys(self.source_encoder_mlp_kwargs, require_input_dim_key=False, require_output_dim_key=False)

        # sanity check conditioning block
        if self.conditioning_type == "film":
            msg = f"You must encode the state and use guidance when using the {self.conditioning_type} conditioning."
            assert self.encode_state and self.use_guidance, msg

        elif self.conditioning_type == "resnet":
            msg = f"You must encode the state when using the {self.conditioning_type} conditioning."
            assert self.encode_state, msg

        elif self.conditioning_type != "concatenation":
            msg = f"Conditioning type {self.conditioning_type} is not supported. Possible values are [\"concatenation\", \"resnet\", \"film\"]"
            raise ValueError(msg)

        # sanity check on condition encoder
        if self.use_guidance:
            # layers before pooling
            msg = f"With {self.use_guidance=} you need to pass a dictionary in the proper format as the `self.perturbation_layers_before_pooling` attribute, found `None`"
            assert self.perturbation_layers_before_pooling is not None, msg 
            self._verify_perturbation_layers_before_pooling()

            # layers after pooling
            msg = f"With {self.encode_conditions=} you need to pass a dictionary in the proper format as the `self.perturbation_layers_after_pooling` attribute, found `None`"
            assert self.perturbation_layers_after_pooling is not None, msg
            self._verify_perturbation_layers_after_pooling()
        else:
            # sanity check on use classifier free guidance
            msg = f"With {self.use_classifier_free_guidance=} you need to instantiate a guided flow, but found {self.use_guidance=}."
            assert not self.use_classifier_free_guidance, msg

            msg = f"With {self.use_guidance=} an unguided flow model will be initialized, thus the settings for the condition encoder will be ignored."
            logger.warning(msg)

    def _verify_perturbation_layers_before_pooling(
        self,
    ) -> None:
        """"""

        for condition, layers_dict in self.perturbation_layers_before_pooling.items():
            if self.encode_conditions:
                msg = f"`layers_dict` for covariate {condition} is expected to be an instance of `dict`, found {type(layers_dict)}"
                assert isinstance(layers_dict, dict), msg
                MLPConfigFields.verify_keys(layers_dict)
            else:
                msg = f"`layers_dict` for covariate {condition} is expected to be an instance of `dict`, found {type(layers_dict)}"
                assert isinstance(layers_dict, dict), msg
                msg = f"`layers_dict` for covariate {condition} is expected to contain the \"input_dim\" key, which was not found."
                assert "input_dim" in layers_dict.keys(), msg
                msg = f"`layers_dict[\"input_dim\"] for covariate {condition} is expected to be an `int`, found {type(layers_dict['input_dim'])}"
                assert isinstance(layers_dict["input_dim"], int), msg 

    def _verify_perturbation_layers_after_pooling(
        self,
    ) -> None:
        """"""

        msg = f"`self.perturbation_layers_after_pooling` is expected to be an instance of `dict`, found {type(self.perturbation_layers_after_pooling)}"
        assert isinstance(self.perturbation_layers_after_pooling, dict), msg
        MLPConfigFields.verify_keys(self.perturbation_layers_after_pooling, require_input_dim_key=False, require_output_dim_key=False)
        self.perturbation_layers_after_pooling["input_dim"] = self.perturbation_layers_after_pooling_input_dim
        self.perturbation_layers_after_pooling["output_dim"] = self.perturbation_encoder_output_dim

    @property
    def time_encoder_input_dim(
        self,
    ) -> int:
        """
        Computes the input dimension for the time encoder.

        This will be given by :attr: `NeuralVelocityFieldConfig.time_features_num_freqs` when
        :attr: `NeuralVelocityFieldConfig.use_sinusoidal_time_features` is set to `True`,
        otherwise will return 1 (i.e.: scalar time).

        :rtype: class: `int`
        """
        if self.use_sinusoidal_time_features:
            return self.time_features_num_freqs
        return 1

    @property
    def condition_input_dim(
        self,
    ) -> int:
        """
        Collects the condition input dimensions for the encoding process from the configuration.

        This will be computed by summing up all the "input_dim" values of the configuration dictionaries
        mapped to each perturbation in :attr:`NeuralVelocityFieldConfig.perturbation_layers_before_pooling`,
        which contain the "input_dim" key as verified by the :method:`NeuralVelocityFieldConfig.__post_init__` method.

        :rtype: class: `int`
        """
        dim = 0
        for layers_dict in self.perturbation_layers_before_pooling.values():
            if isinstance(layers_dict, dict):
                input_dim = layers_dict["input_dim"]
            else:
                msg = f""
                raise TypeError(msg)
            dim = dim + input_dim
        return dim

    @property
    def perturbation_layers_after_pooling_input_dim(
        self,
    ) -> int:
        """
        Collect the condition input dimensions after pooling.

        This will be computed by summing up all the "output_dim" values of the configuration dictionaries
        mapped to each perturbation in :attr:`NeuralVelocityFieldConfig.perturbation_layers_before_pooling` and
        in :attr:`NeuralVelocityFieldConfig.perturbation_covariates_not_pooled`. On the other hand, the output
        dimensionality of the pooled covaraites will only be considered once.

        :rtype: class: `int`
        """
        # Initialize the dim as the output of the pooling layer 
        dim = 0

        # Perturbations to pull
        perturbation_covariate_pooled = get_conditions_to_pool(
            self.perturbation_layers_before_pooling,
            self.perturbation_covariates_not_pooled
        )    
        # Pooled layers 
        if len(perturbation_covariate_pooled) > 0:
            for perturbation_to_pool in perturbation_covariate_pooled:
                covariate_pool_dict = self.perturbation_layers_before_pooling[perturbation_to_pool]
                if dim == 0:
                    dim = dim + covariate_pool_dict["output_dim"]
                msg = f"The output layers of the pooled variables must all have the same dimensionality."
                assert covariate_pool_dict["output_dim"] == dim, msg
        
        # Not pooled layers 
        if self.perturbation_covariates_not_pooled is not None:
            for condition in self.perturbation_covariates_not_pooled:
                layers_dict = self.perturbation_layers_before_pooling[condition]
                output_dim = layers_dict["output_dim"]
                dim = dim + output_dim
        return dim

    @property
    def perturbation_latent_dim(
        self,
    ) -> int:
        """Returns the latent dimensionality of the perturbations. When no guidance is used returns 0."""
        if self.use_guidance and self.encode_conditions:
            return self.perturbation_encoder_output_dim
        elif self.use_guidance and (not self.encode_conditions):
            return self.condition_input_dim
        return 0

    @property
    def time_latent_dim(
        self,
    ) -> int:
        """Returns the latent dimensionality of the time index."""
        if self.encode_time:
            return self.time_encoder_output_dim
        return self.time_encoder_input_dim

    @property
    def source_latent_dim(
        self,
    ) -> int:
        """Returns the latent dimension of the source conditioning. Returns 0 when the source is not used to condition the VF."""
        if self.use_source_as_condition:
            if self.encode_source:
                return self.source_encoder_output_dim
            return self.flow_dim
        return 0

    @property
    def state_latent_dim(
        self,
    ) -> int:
        """Returns the latent dimension of the states."""
        if self.encode_state:
            return self.state_encoder_output_dim 
        return self.flow_dim

    @property
    def resnet_embedding_dim(
        self,
    ) -> int:
        """Returns the dimensionality of the residual network condition embedding."""
        return self.time_latent_dim + self.perturbation_latent_dim + self.source_latent_dim

    @property
    def decoder_input_dim(
        self,
    ) -> int:
        """
        Collects the dimension of the joint latent space, used to infer the input dimension for the VF decoder.
        It adds the correct dimensions for each different case.

        :rtype: class: `int`
        """    
        # concatenation state, conditions, source and time 
        if self.conditioning_type == "concatenation":
            return self.state_latent_dim + self.time_latent_dim + self.perturbation_latent_dim + self.source_latent_dim
        # resnet block
        elif self.conditioning_type == "resnet":
            return self.state_latent_dim
        # film block
        elif self.conditioning_type == "film":
            return self.state_latent_dim + self.time_latent_dim        

    @property
    def initialize_source_encoder(
        self,
    ) -> bool:
        """
        Flag indicating whether to initialize the condition encoder. This will return `True` only in the case
        when both :attr:`NeuralVelocityFieldConfig.use_source_as_condition` and :attr:`NeuralVelocityFieldConfig.encode_source`
        are set to `True`, otherwise it returns `False`.

        :rtype: class: `bool` 
        """
        if self.use_source_as_condition:
            if self.encode_source:
                return True
        return False
