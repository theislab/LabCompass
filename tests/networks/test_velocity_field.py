from collections.abc import Sequence
from typing import Any, Literal

import pytest
import torch

import sc_exp_design
from sc_exp_design.constants import (
    COVARIANCE_KEY,
    MEAN_KEY,
    PERTURBATION_PARAMS_KEYS,
    SOURCE_PARAMS_KEY,
    TARGET_PARAMS_KEY,
    VF_KEY,
)


batch_size = 10
flow_dim = 5
condition_dim = 1
perturbation_latent_dim = 7
pert_target_covariates_output_dims = {"cond": condition_dim}
perturbation_layers_before_pooling = {"cond": {"input_dim": 1}}
perturbation_layers_before_pooling_mlp = {"cond": {"input_dim": 1, "output_dim": 10, "layer_type": "mlp"}}
perturbation_layers_before_pooling_attention = {"cond": {"num_embeddings": 2, "layer_type": "self_attention"}}
perturbation_layers_after_pooling = {"layer_type": "mlp", "output_dim": 13, "input_dim":...}
condition_encoder_num_embeddings = 2

x_test = torch.ones((10, flow_dim)) * 10
source = torch.ones((10, flow_dim)) * 10
target = torch.ones((10, flow_dim)) * 10
t_test = torch.ones((10,))
cond = {"cond": torch.ones(10, condition_dim)}

src_noise_model = "gaussian"
tgt_noise_model = "gaussian"


class TestNeuralVelocityField:
    @pytest.mark.parametrize("x_encoder_hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("encode_time", [True, False])
    @pytest.mark.parametrize("time_encoder_hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("use_guidance", [True, False])
    @pytest.mark.parametrize("encode_conditions", [True, False])
    @pytest.mark.parametrize("perturbation_encoding", ["mlp", "self_attention"])
    @pytest.mark.parametrize("perturbation_pooling", ["mean", "self_attention"])
    @pytest.mark.parametrize("decoder_hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("learn_score_field", [True, False])
    @pytest.mark.parametrize("score_hidden_dims", [(32, 32), ()])
    @pytest.mark.parametrize("learn_posterior_on_cond_vars", [True, False])
    @pytest.mark.parametrize("src_approximate_posterior_kwargs", [{"cov_estimation_mode": "isotropic"}, {"cov_estimation_mode": "anisotropic"}])
    @pytest.mark.parametrize("tgt_approximate_posterior_kwargs", [{"cov_estimation_mode": "isotropic"}, {"cov_estimation_mode": "anisotropic"}])
    @pytest.mark.parametrize("learn_posterior_on_perts", [True, False])
    @pytest.mark.parametrize("pert_approximate_posterior_input_type", ["latent", "original", "endpoints", "one_step_prediction",])
    @pytest.mark.parametrize("learn_posterior_on_latent_perts", [True, False])
    def test_conditional_velocity_field(
        self,
        x_encoder_hidden_dims: Sequence[int],
        encode_time: bool,
        time_encoder_hidden_dims: Sequence[int],
        use_guidance: bool,
        encode_conditions: bool,
        perturbation_encoding: Literal["mlp", "self_attention"],
        perturbation_pooling: Literal["mean", "self_attention"],
        decoder_hidden_dims: Sequence[int],
        learn_score_field: bool,
        score_hidden_dims: Sequence[int],
        learn_posterior_on_cond_vars: bool,
        src_approximate_posterior_kwargs: dict[str, Any],
        tgt_approximate_posterior_kwargs: dict[str, Any],
        learn_posterior_on_perts: bool,
        pert_approximate_posterior_input_type: Literal["latent", "original", "endpoints", "one_step_prediction"],
        learn_posterior_on_latent_perts: bool,
    ):

        # retrieving current settings
        layers_before_pooling = perturbation_layers_before_pooling
        if use_guidance and encode_conditions and (perturbation_encoding == "mlp"):
            layers_before_pooling = perturbation_layers_before_pooling_mlp
        if use_guidance and encode_conditions and (perturbation_encoding == "self_attention"):
            layer_before_pooling = perturbation_layers_before_pooling_attention
        if not use_guidance and encode_conditions: 
            learn_posterior_on_latent_perts = False

        # initializing configurations
        config = sc_exp_design.networks.NeuralVelocityFieldConfig(
            pert_target_covariates_output_dims=pert_target_covariates_output_dims,
            perturbation_latent_dim=perturbation_latent_dim,
            perturbation_layers_before_pooling=layers_before_pooling,
            perturbation_layers_after_pooling=perturbation_layers_after_pooling,
            # sweep settings
            x_encoder_hidden_dims=x_encoder_hidden_dims,
            encode_time=encode_time,
            time_encoder_hidden_dims=time_encoder_hidden_dims,
            use_guidance=use_guidance,
            encode_conditions=encode_conditions,
            decoder_hidden_dims=decoder_hidden_dims,
            learn_score_field=learn_score_field,
            score_hidden_dims=score_hidden_dims,
            learn_posterior_on_cond_vars=learn_posterior_on_cond_vars,
            src_approximate_posterior_kwargs=src_approximate_posterior_kwargs,
            tgt_approximate_posterior_kwargs=tgt_approximate_posterior_kwargs,
            learn_posterior_on_perts=learn_posterior_on_perts,
            pert_approximate_posterior_input_type=pert_approximate_posterior_input_type,
            learn_posterior_on_latent_perts=learn_posterior_on_latent_perts,
        )

        # forward pass on velocity field
        cvf = sc_exp_design.networks.NeuralVelocityField(
            flow_dim,
            config,
        )
        vf_out = cvf.forward(t_test, x_test, cond)

        # sanity check on velocity field output
        msg = f"The velocity field has the wrong shape. Got {vf_out[VF_KEY].shape}, expected {(batch_size, flow_dim)}."
        assert vf_out[VF_KEY].shape == (batch_size, flow_dim), msg

        # sanity check on conditioning var posterior
        if learn_posterior_on_cond_vars:
            # source
            msg = f"The mean for the source state distribution has the wrong shape. Got {vf_out[SOURCE_PARAMS_KEY][MEAN_KEY].shape}, expected {(batch_size, flow_dim)}."
            assert vf_out[SOURCE_PARAMS_KEY][MEAN_KEY].shape == (batch_size, flow_dim), msg
            if src_approximate_posterior_kwargs["cov_estimation_mode"] == "isotropic":
                msg = f"The covariance for the source state distribution has the wrong shape. Got {vf_out[SOURCE_PARAMS_KEY][COVARIANCE_KEY].shape}, expected {(batch_size, 1)}."
                assert vf_out[SOURCE_PARAMS_KEY][COVARIANCE_KEY].shape == (batch_size, 1), msg
            elif src_approximate_posterior_kwargs["cov_estimation_mode"] == "anisotropic":
                msg = f"The covariance for the target source distribution has the wrong shape. Got {vf_out[SOURCE_PARAMS_KEY][MEAN_KEY].shape}, expected {(batch_size, flow_dim)}."
                assert vf_out[SOURCE_PARAMS_KEY][COVARIANCE_KEY].shape == (batch_size, flow_dim), msg
            # target
            msg = f"The mean for the target state distribution has the wrong shape. Got {vf_out[TARGET_PARAMS_KEY][MEAN_KEY].shape}, expected {(batch_size, flow_dim)}."
            assert vf_out[TARGET_PARAMS_KEY][MEAN_KEY].shape == (batch_size, flow_dim), msg
            if tgt_approximate_posterior_kwargs["cov_estimation_mode"] == "isotropic":
                msg = f"The covariance for the source state distribution has the wrong shape. Got {vf_out[TARGET_PARAMS_KEY][COVARIANCE_KEY].shape}, expected {(batch_size, 1)}."
                assert vf_out[TARGET_PARAMS_KEY][COVARIANCE_KEY].shape == (batch_size, 1), msg
            elif tgt_approximate_posterior_kwargs["cov_estimation_mode"] == "anisotropic":
                msg = f"The covariance for the source state distribution has the wrong shape. Got {vf_out[TARGET_PARAMS_KEY][COVARIANCE_KEY].shape}, expected {(batch_size, flow_dim)}."
                assert vf_out[TARGET_PARAMS_KEY][COVARIANCE_KEY].shape == (batch_size, flow_dim), msg

        # sanity check on perturbation posterior
        if learn_posterior_on_perts:
            if pert_approximate_posterior_input_type in ["endpoints", "one_step_prediction"]:
                ...
            else:
                for covariate_id, covariate_posterior_params in vf_out[PERTURBATION_PARAMS_KEYS].items():
                    msg = f"The posterior parameters for {covariate_id=} has the wrong shape. Got {covariate_posterior_params.shape}, expected {(batch_size, pert_target_covariates_output_dims[covariate_id])}"
                    assert covariate_posterior_params.shape == (
                        batch_size,
                        pert_target_covariates_output_dims[covariate_id],
                    ), msg

        # sanity check on latent perturbation posterior
        if learn_posterior_on_latent_perts and encode_conditions:
            latent_condition_inf_params = cvf.get_latent_condition_inf_params(source, target)
            msg = f"The mean of posterior parameters for the latent perturbations has the wrong shape. Got {latent_condition_inf_params[MEAN_KEY].shape}, expected {(batch_size, cvf.condition_encoder.latent_dim)}"
            assert latent_condition_inf_params[MEAN_KEY].shape == (batch_size, cvf.condition_encoder.latent_dim), msg
            if latent_perts_approximate_posterior_kwargs["cov_estimation_mode"] == "isotropic":
                msg = f"The covariance of posterior parameters for the latent perturbations has the wrong shape. Got {latent_condition_inf_params[COVARIANCE_KEY].shape}, expected {(batch_size, 1)}"
                assert latent_condition_inf_params[COVARIANCE_KEY].shape == (batch_size, 1)
            if latent_perts_approximate_posterior_kwargs["cov_estimation_mode"] == "anisotropic":
                msg = f"The covariance of posterior parameters for the latent perturbations has the wrong shape. Got {latent_condition_inf_params[COVARIANCE_KEY].shape}, expected {(batch_size, cvf.condition_encoder.latent_dim)}"
                assert latent_condition_inf_params[COVARIANCE_KEY].shape == (
                    batch_size,
                    cvf.condition_encoder.latent_dim,
                ), msg
