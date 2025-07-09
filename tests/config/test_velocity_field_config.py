from typing import Any, Literal

import pytest

from sc_exp_design.config import NeuralVelocityFieldConfig


INVALID_STRING = "invalid_string"


class TestNeuralVelocityFieldConfig:
    @pytest.mark.parametrize("conditioning_type", ["concatenation", "resnet", "film", INVALID_STRING])
    def test_config_init_conditioning_type(
        self,
        num_genes: int,
        conditioning_type: Literal["concatenation", "resnet", "film"],
    ):
        # prepare perturbation layers before pooling dictionary
        perturbation_layers_before_pooling = {
            "condition": {
                "input_dim": 2,
                "output_dim": 2,
            }
        }

        if conditioning_type == INVALID_STRING:
            with pytest.raises(
                ValueError,
                # match=r'is not supported. Possible values are \[\"concatenation\", \"resnet\", \"film\"\]'
            ):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    conditioning_type=conditioning_type,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                conditioning_type=conditioning_type,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )

    @pytest.mark.parametrize("perturbation_layers_before_pooling", [
            None,
            {"condition": {}},
            {"condition": {"input_dim": 2, "output_dim":2}},
            {"condition": {"input_dim": 2, }},
            {"condition": {"output_dim":2, }}
        ]
    )        
    def test_config_init_perturbation_layers_before_pooling(
        self,
        num_genes: int,
        perturbation_layers_before_pooling: dict[str, dict[str, Any]]
    ):
        if perturbation_layers_before_pooling is None:
            with pytest.raises(
                AssertionError,
                # match=r"You need to pass a dictionary in the proper format as the `self.perturbation_layers_before_pooling` attribute, found `None`",
            ):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        elif "input_dim" not in perturbation_layers_before_pooling["condition"].keys() or "output_dim" not in perturbation_layers_before_pooling["condition"].keys():
            with pytest.raises(AssertionError):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )

    @pytest.mark.parametrize("perturbation_layers_after_pooling", [
            None,
            {},
            {"input_dim": 3},
            {"output_dim": 3},
        ]
    )
    def test_config_init_perturbation_layers_after_pooling(
        self,
        num_genes: int,
        perturbation_layers_after_pooling: dict[str, Any],
    ):
        # prepare perturbation layers before pooling dictionary
        perturbation_layers_before_pooling = {
            "condition": {
                "input_dim": 2,
                "output_dim": 2,
            }
        }

        if perturbation_layers_after_pooling is None:
            with pytest.raises(
                AssertionError,
                # match=r"You need to pass a dictionary in the proper format as the `self.perturbation_layers_after_pooling` attribute, found `None`"
            ):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    perturbation_layers_after_pooling=perturbation_layers_after_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        elif "input_dim" in perturbation_layers_after_pooling.keys() or "output_dim" in perturbation_layers_after_pooling.keys():
            with pytest.raises(
                AssertionError,
            ):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    perturbation_layers_after_pooling=perturbation_layers_after_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                perturbation_layers_after_pooling=perturbation_layers_after_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )
            assert "input_dim" in cvf_config.perturbation_layers_after_pooling.keys()
            assert "output_dim" in cvf_config.perturbation_layers_after_pooling.keys()

    @pytest.mark.parametrize("state_encoder_mlp_kwargs", [{}, {"input_dim": 3},{"output_dim": 3},])
    def test_config_init_state_encoder_mlp_kwargs(
        self,
        num_genes: int,
        state_encoder_mlp_kwargs: dict[str, Any],
    ):

        # prepare perturbation layers before pooling dictionary
        perturbation_layers_before_pooling = {
            "condition": {
                "input_dim": 2,
                "output_dim": 2,
            }
        }
        if "input_dim" in state_encoder_mlp_kwargs.keys() or "output_dim" in state_encoder_mlp_kwargs.keys():
            with pytest.raises(AssertionError):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                state_encoder_mlp_kwargs=state_encoder_mlp_kwargs,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,

            )
            assert "input_dim" not in cvf_config.state_encoder_mlp_kwargs.keys()
            assert "output_dim" not in cvf_config.state_encoder_mlp_kwargs.keys()

    @pytest.mark.parametrize("time_encoder_mlp_kwargs", [
            {},
            {"input_dim": 3},
            {"output_dim": 3},
        ]
    )
    def test_config_init_time_encoder_mlp_kwargs(
        self,
        num_genes: int,
        time_encoder_mlp_kwargs: dict[str, Any],
    ):

        # prepare perturbation layers before pooling dictionary
        perturbation_layers_before_pooling = {
            "condition": {
                "input_dim": 2,
                "output_dim": 2,
            }
        }
        if "input_dim" in time_encoder_mlp_kwargs.keys() or "output_dim" in time_encoder_mlp_kwargs.keys():
            with pytest.raises(AssertionError):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                time_encoder_mlp_kwargs=time_encoder_mlp_kwargs,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )
            assert "input_dim" not in cvf_config.time_encoder_mlp_kwargs.keys()
            assert "output_dim" not in cvf_config.time_encoder_mlp_kwargs.keys()

    @pytest.mark.parametrize("decoder_mlp_kwargs", [
            {},
            {"input_dim": 3},
            {"output_dim": 3},
        ]
    )
    def test_config_init_decoder_mlp_kwargs(
        self,
        num_genes: int,
        decoder_mlp_kwargs: dict[str, Any],
    ):

        # prepare perturbation layers before pooling dictionary
        perturbation_layers_before_pooling = {
            "condition": {
                "input_dim": 2,
                "output_dim": 2,
            }
        }
        if "input_dim" in decoder_mlp_kwargs.keys() or "output_dim" in decoder_mlp_kwargs.keys():
            with pytest.raises(AssertionError):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    decoder_mlp_kwargs=decoder_mlp_kwargs,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                decoder_mlp_kwargs=decoder_mlp_kwargs,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )
            assert "input_dim" not in cvf_config.decoder_mlp_kwargs.keys()
            assert "output_dim" not in cvf_config.decoder_mlp_kwargs.keys()

    @pytest.mark.parametrize("perturbation_layers_before_pooling", [
        {"condition0": {"input_dim": 2,"output_dim": 2,}, "condition1": {"input_dim": 2,"output_dim": 2,},},
        {"condition0": {"input_dim": 2,"output_dim": 2,}, "condition1": {"input_dim": 2,"output_dim": 4,},},
    ])
    def test_perturbation_layers_after_pooling_input_dim(
        self,
        num_genes: int,
        perturbation_layers_before_pooling: dict[str, dict[str, Any]],
    ):

        if perturbation_layers_before_pooling["condition0"]["output_dim"] != perturbation_layers_before_pooling["condition1"]["output_dim"]:
            with pytest.raises(
                AssertionError,
                # match=r"The output layers of the pooled variables must all have the same dimensionality\."
            ):
                cvf_config = NeuralVelocityFieldConfig(
                    flow_dim=num_genes,
                    perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                    use_guidance=True,
                    encode_conditions=True,
                    encode_source=True,
                    encode_state=True,
                    encode_time=True,
                    use_sinusoidal_time_features=True,
                )
                return None
        else:
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )
