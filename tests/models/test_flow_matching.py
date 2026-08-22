from collections.abc import Sequence
from typing import Literal

import anndata
import pytest
import torch

from labcompass.config import NeuralVelocityFieldConfig
from labcompass.models import FlowMatching


class TestFlowMatching:
    @pytest.mark.parametrize("flow_type", ["constant_noise", "encoding_decoding", "rectified", "variance_preserving"])
    @pytest.mark.parametrize("coupling_type", ["independent", "ot"])
    @pytest.mark.parametrize("generate_from_noise", [True, False])
    @pytest.mark.parametrize("use_controls", [True, False])
    @pytest.mark.parametrize("use_perturbations_in_obsm", [True, False])
    @pytest.mark.parametrize("return_trajectory", [True, False])
    @pytest.mark.parametrize("no_grad", [True, False])

    def test_flow_matching_init(
        self,
        adata: anndata.AnnData,
        num_genes: int,
        num_unique_treatments: int,
        num_perturbation_feats: int,
        perturbations_in_obsm: Sequence[str] | None,
        flow_type: Literal["constant_noise", "encoding_decoding", "rectified", "variance_preserving"],
        coupling_type: Literal["independent", "ot"],
        generate_from_noise: bool,
        use_controls: bool,
        use_perturbations_in_obsm: bool,
        return_trajectory: bool,
        no_grad: bool,
    ) -> None:
        
        # setting arguments for data
        control_key = "is_control" if use_controls else None
        use_source_as_condition = use_controls and generate_from_noise
        sample_rep = "states"
        treatment0_label = "treatment0"
        perturbations = (treatment0_label, perturbations_in_obsm) if use_perturbations_in_obsm else (treatment0_label,)
        perturbation_reps = {
            treatment0_label: f"{treatment0_label}_label",
            perturbations_in_obsm: f"{perturbations_in_obsm}_features",
        } if use_perturbations_in_obsm else {
            treatment0_label: f"{treatment0_label}_label",
        }

        # initialize flow matching class
        flow_matching = FlowMatching(
            flow_type=flow_type,
            coupling_type=coupling_type,
            generate_from_noise=generate_from_noise,
        )

        # preparing data
        if use_perturbations_in_obsm and coupling_type == "ot":
            with pytest.raises(
                ValueError,
                match="With perturbations in obsm the coupling must be independent"
            ):
                flow_matching.prepare_train_data(
                    adata,
                    sample_rep=sample_rep,
                    control_key=control_key,
                    perturbations=perturbations,
                    perturbation_reps=perturbation_reps,
                    perturbations_in_obsm=perturbations_in_obsm if use_perturbations_in_obsm else None,
                )
                return None
        else:
            flow_matching.prepare_train_data(
                adata,
                sample_rep=sample_rep,
                control_key=control_key,
                perturbations=perturbations,
                perturbation_reps=perturbation_reps,
                perturbations_in_obsm=perturbations_in_obsm if use_perturbations_in_obsm else None,
            )
            
            assert flow_matching.data_manager is not None
            assert flow_matching.train_data is not None

            # setting arguments for model
            perturbation_covariates_latent_dim = 16
            perturbation_layers_before_pooling = {
                f"repr_{treatment0_label}_{treatment0_label}_label": {
                    "input_dim": num_unique_treatments,
                    "output_dim": perturbation_covariates_latent_dim
                },
            }
            if use_perturbations_in_obsm:
                perturbation_layers_before_pooling.update(
                    {
                        f"feats_{perturbations_in_obsm}_{perturbations_in_obsm}_features": {
                            "input_dim": num_perturbation_feats,
                            "output_dim":perturbation_covariates_latent_dim
                        }
                    }
                )

            # initializing velocity field configurations
            cvf_config = NeuralVelocityFieldConfig(
                flow_dim=num_genes,
                perturbation_layers_before_pooling=perturbation_layers_before_pooling,
                use_source_as_condition=use_source_as_condition,
                use_guidance=True,
                encode_conditions=True,
                encode_source=True,
                encode_state=True,
                encode_time=True,
                use_sinusoidal_time_features=True,
            )

            # preparing model
            if use_controls and generate_from_noise and (not use_source_as_condition):
                with pytest.raises(
                    ValueError,
                    # match="When no controls are available use_source_as_condition should be False."
                ):
                    flow_matching.prepare_model(cvf_config)
                    return None
            elif (not use_controls) and generate_from_noise and use_source_as_condition:
                with pytest.raises(
                    ValueError,
                    # match="When no controls are available use_source_as_condition should be False."
                ):
                    flow_matching.prepare_model(cvf_config)
                    return None
            elif (not use_controls) and (not generate_from_noise):    
                with pytest.raises(
                    ValueError,
                    # match="When no controls are available use_source_as_condition should be False."
                ):
                    flow_matching.prepare_model(cvf_config)
                    return None
            else:
                flow_matching.prepare_model(cvf_config)
                assert hasattr(flow_matching, "velocity_field")

                # training the model
                flow_matching.train(num_training_steps=3)
                assert hasattr(flow_matching, "trainer")
                assert hasattr(flow_matching, "train_dataloader")

                # preparing for prediction
                batch_size = 16
                num_samples = 50
                num_time_steps = 3
                source_states = None
                if use_controls:
                    source_states = torch.ones((batch_size, num_genes)).to(flow_matching.device)
                
                # predicting                
                preds = flow_matching.predict(
                    {
                        "condition": {cov: torch.ones((batch_size, cov_dict["input_dim"])).to(flow_matching.device) for cov, cov_dict in perturbation_layers_before_pooling.items()},
                        "source": source_states,
                    },
                    return_trajectory=return_trajectory,
                    no_grad=no_grad,
                    num_samples=num_samples,
                    num_time_steps=num_time_steps,
                )

                # check shape
                if generate_from_noise and (not return_trajectory):
                    assert preds.shape == (num_samples, batch_size, num_genes)
                elif generate_from_noise and return_trajectory:
                    assert preds.shape == (num_time_steps, num_samples, batch_size, num_genes)
                elif (not generate_from_noise) and (not return_trajectory):
                    assert preds.shape == (batch_size, num_genes)
                elif (not generate_from_noise) and return_trajectory:
                    assert preds.shape == (num_time_steps, batch_size, num_genes)

                # check grads
                if no_grad:
                    assert not preds.requires_grad
                else:
                    assert preds.requires_grad
