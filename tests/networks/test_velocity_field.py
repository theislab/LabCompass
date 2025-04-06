from collections.abc import Sequence
from typing import Any, Literal

import pytest
import torch

import sc_exp_design
from sc_exp_design.constants import ParamsFields, VFStepFields

# dimensionalities
batch_size = 4
flow_dim = 2
treatment0_dim = 5
treatment1_dim = 7
perturbation_dim_before_pooling = 32
perturbation_latent_dim = 16
state_latent_dim = 16
time_features_num_freqs = 8
time_latent_dim = 16

# input data
x_test = torch.ones((batch_size, flow_dim)) * 10
source = torch.ones((batch_size, flow_dim)) * 10
target = torch.ones((batch_size, flow_dim)) * 10
t_test = torch.ones((batch_size,))
cond = {
    "treatment0_id": torch.ones(batch_size, treatment0_dim),
    "treatment0_dose": torch.zeros(batch_size, 1),
    "treatment1_id": torch.ones(batch_size, treatment1_dim),
    "treatment1_dose": torch.zeros(batch_size, 1),
}


# mlp configurations
linear_config = lambda: {
    "hidden_dims": (),
    "use_batchnorm": False,
    "use_dropout": False,
    "dropout_rate": 0.0,
    "activation_class": torch.nn.Identity,
    "final_activation_class": torch.nn.Identity,
}
mlp_config = lambda: {
    "hidden_dims": (32, 32,),
    "use_batchnorm": False,
    "use_dropout": False,
    "dropout_rate": 0.0,
    "activation_class": torch.nn.Identity,
    "final_activation_class": torch.nn.Identity,
}

# perturbation layers config
perturbation_layers_before_pooling_no_encoding = {
    "treatment0_id": {
        "input_dim": treatment0_dim
    },
    "treatment0_dose": {
        "input_dim": 1
    },
    "treatment1_id": {
        "input_dim": treatment1_dim
    },
    "treatment1_dose": {
        "input_dim": 1
    }
}
perturbation_layers_before_pooling_linear_encode = {
    "treatment0_id": {
        "input_dim": treatment0_dim,
        "output_dim": perturbation_dim_before_pooling,
        **linear_config(),
    },
    "treatment0_dose": {
        "input_dim": 1,
        "output_dim": perturbation_dim_before_pooling,
        **linear_config(),
    },
    "treatment1_id": {
        "input_dim": treatment1_dim,
        "output_dim": perturbation_dim_before_pooling,
        **linear_config(),
    },
    "treatment1_dose": {
        "input_dim": 1,
        "output_dim": perturbation_dim_before_pooling,
        **linear_config(),
    }
}


class TestNeuralVelocityField:
    @pytest.mark.parametrize("encode_state", [True, False])
    @pytest.mark.parametrize("state_encoder_mlp_kwargs", [linear_config, mlp_config])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("use_sinusoidal_time_features", [True, False])
    @pytest.mark.parametrize("time_encoder_mlp_kwargs", [linear_config, mlp_config])
    @pytest.mark.parametrize("use_guidance", [True, False])
    @pytest.mark.parametrize("encode_conditions", [True, False])
    @pytest.mark.parametrize("perturbation_pooling", ["mean", "sum"])
    @pytest.mark.parametrize("perturbation_layers_after_pooling", [linear_config, mlp_config])
    @pytest.mark.parametrize("decoder_mlp_kwargs", [linear_config, mlp_config])
    @pytest.mark.parametrize("use_source_as_condition", [True, False])
    @pytest.mark.parametrize("encode_source", [True, False])
    @pytest.mark.parametrize("source_encoder_mlp_kwargs", [linear_config, mlp_config])
    @pytest.mark.parametrize("use_resnet_blocks", [False, True])
    @pytest.mark.parametrize("resnet_normalization", ["layer", "batch", None])
    def test_conditional_velocity_field(
        self,
        encode_state: bool,
        state_encoder_mlp_kwargs: dict[str, Any],
        encode_time: bool,
        use_sinusoidal_time_features: bool,
        time_encoder_mlp_kwargs: dict[str, Any],
        use_guidance: bool,
        encode_conditions: bool,
        perturbation_pooling: Literal["mean", "sum"],
        perturbation_layers_after_pooling: dict[str, Any],
        decoder_mlp_kwargs: Sequence[int],
        use_source_as_condition: bool,
        encode_source: bool,
        source_encoder_mlp_kwargs: dict[str, Any],
        use_resnet_blocks: bool,
        resnet_normalization: Literal["layer", "batch"] | None,
    ):

        # retrieving current settings
        perturbation_layers_before_pooling = perturbation_layers_before_pooling_no_encoding
        if encode_conditions:
            perturbation_layers_before_pooling = perturbation_layers_before_pooling_linear_encode

        # we can only use resnet when encoding states
        if not encode_state:
            use_resnet_blocks = False

        # initializing configurations
        config = sc_exp_design.config.NeuralVelocityFieldConfig(
            flow_dim,
            encode_state=encode_state,
            state_encoder_output_dim=state_latent_dim,
            state_encoder_mlp_kwargs=state_encoder_mlp_kwargs(),
            encode_time=encode_time,
            use_sinusoidal_time_features=use_sinusoidal_time_features,
            time_features_num_freqs=time_features_num_freqs,
            time_encoder_output_dim=time_latent_dim,
            time_encoder_mlp_kwargs=time_encoder_mlp_kwargs(),
            use_guidance=use_guidance,
            encode_conditions=encode_conditions,
            perturbation_latent_dim=perturbation_latent_dim,
            perturbation_pooling=perturbation_pooling,
            perturbation_layers_before_pooling=perturbation_layers_before_pooling,
            perturbation_layers_after_pooling=perturbation_layers_after_pooling(),
            decoder_mlp_kwargs=decoder_mlp_kwargs(),
            use_source_as_condition=use_source_as_condition,
            encode_source=encode_source,
            source_latent_dim=state_latent_dim,
            source_encoder_mlp_kwargs=source_encoder_mlp_kwargs(),
        )

        # forward pass on velocity field
        cvf = sc_exp_design.networks.NeuralVelocityField(
            config,
        )
        vf_out = cvf.forward(t_test, x_test, cond, source=source)
  
        # retrieve target latent state dim
        expected_latent_state_dim = flow_dim
        if encode_state:
            expected_latent_state_dim = state_latent_dim
        # retrieve target latent time dim
        expected_latent_time_dim = 1
        if use_sinusoidal_time_features:
            expected_latent_time_dim = time_features_num_freqs
        if encode_time:
            expected_latent_time_dim = time_latent_dim
        # retrieve target latent condition dim
        expected_latent_condition_dim = 0
        if use_guidance:
            expected_latent_condition_dim = sum(
                [
                    mlp_conf["input_dim"] for mlp_conf in perturbation_layers_before_pooling.values()
                ]
            )
            if encode_conditions:
                expected_latent_condition_dim = perturbation_latent_dim
        # retrieve target latent source dim
        expected_latent_source_dim = 0
        if use_source_as_condition:
            expected_latent_source_dim = flow_dim
            if encode_source:
                expected_latent_source_dim = state_latent_dim
        expected_joint_latent_dim = expected_latent_state_dim + expected_latent_time_dim + expected_latent_condition_dim + expected_latent_source_dim

        # sanity check on velocity field output
        msg = f"The velocity field has the wrong shape. Got {vf_out[VFStepFields.VF].shape}, expected {(batch_size, flow_dim)}."
        assert vf_out[VFStepFields.VF].shape == (batch_size, flow_dim), msg

        # sanity check on latent states
        msg = f"Shape mismatch state latent repr. Got: {vf_out[VFStepFields.LATENT_STATE].shape}, expected: {(batch_size, expected_latent_state_dim)}"
        assert vf_out[VFStepFields.LATENT_STATE].shape == (batch_size, expected_latent_state_dim), msg

        msg = f"Shape mismatch joint latent repr. Got: {vf_out[VFStepFields.LATENT_REPR].shape}, expected: {(batch_size, expected_joint_latent_dim)}"
        assert vf_out[VFStepFields.LATENT_REPR].shape == (batch_size, expected_joint_latent_dim), msg
