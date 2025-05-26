from collections.abc import Sequence
from typing import Literal

import anndata
import pytest

import sc_exp_design
from sc_exp_design.constants import DataFields


class TestDataLoaders:
    """"""

    @staticmethod
    def validate_batch(
        train_batch: dict,
        has_controls: bool,
        batch_size: int,
        num_genes: int,
        perturbations: str | list[str] | tuple[str, ...] | None,
        data: sc_exp_design.data.DataContainer,
    ) -> None:
        """
        Validates the contents and shapes of a batch of data.
        """
        # Control states
        if has_controls:
            msg = (
                f"When has_controls={has_controls} the batch dictionary should contain the key "
                f"{DataFields.SOURCE_STATE}. Found {train_batch.keys()}."
            )
            assert DataFields.SOURCE_STATE in train_batch.keys(), msg

            expected_shape = (batch_size, num_genes)
            control_states = train_batch[DataFields.SOURCE_STATE]
            msg = f"Shape error for control states. Got {control_states.shape}, expected {expected_shape}."
            assert hasattr(control_states, "shape"), "Control states must have a 'shape' attribute."
            assert control_states.shape == expected_shape, msg

        # Target states
        msg = f"The batch dictionary should contain the key {DataFields.TARGET_STATE}. Found {train_batch.keys()}."
        assert DataFields.TARGET_STATE in train_batch.keys(), msg

        expected_shape = (batch_size, num_genes)
        target_states = train_batch[DataFields.TARGET_STATE]
        msg = f"Shape error for target states. Got {target_states.shape}, expected {expected_shape}."
        assert hasattr(target_states, "shape"), "Target states must have a 'shape' attribute."
        assert target_states.shape == expected_shape, msg

        # Perturbation data
        if perturbations is not None:
            msg = (
                f"When perturbations are passed, the batch dictionary is expected to contain the "
                f"\"{DataFields.PERTURBATION_DATA}\" key. Found {train_batch.keys()}."
            )
            assert DataFields.PERTURBATION_DATA in train_batch.keys(), msg

            perturbation_data = train_batch[DataFields.PERTURBATION_DATA]
            assert isinstance(perturbation_data, dict), "Perturbation data must be a dictionary."

            for covariate in data.data.perturbation_covariates:
                msg = (
                    f"Perturbation covariate key {covariate} not found in "
                    f"perturbation data keys {perturbation_data.keys()}."
                )
                assert covariate in perturbation_data, msg

    @pytest.mark.parametrize("sample_rep", [None, "states"])
    @pytest.mark.parametrize("control_key", [None, "is_control"])
    @pytest.mark.parametrize("perturbations", [None, ("treatment0",), ("treatment0", "treatment1"), ("treatment0", "treatment1", "treatment2"), ])
    @pytest.mark.parametrize("perturbations_in_obsm", [None, "treatment2"])
    @pytest.mark.parametrize(
        "perturbation_covariates",
        [
            None, 
            {"treatment0":("treatment0_dose", )},
            {"treatment0":("treatment0_dose", "treatment0_time")},
            {"treatment0": ("treatment0_dose", ), "treatment1": ("treatment1_dose", )},
            {"treatment0": ("treatment0_dose", "treatment0_time"), "treatment1": ("treatment1_dose", "treatment1_time")},
            {"treatment0": ("treatment0_dose", ), "treatment1": ("treatment1_dose", "treatment1_time")},
            {"treatment0": ("treatment0_dose", "treatment0_time"), "treatment1": ("treatment1_dose", )},
        ]
    )
    @pytest.mark.parametrize(
        "perturbation_reps",
        [
            None, 
            {"treatment0":("treatment0_label", )},
            {"treatment0":("treatment0_label", "treatment0_group")},
            {"treatment0": ("treatment0_label", ), "treatment1": ("treatment1_label", )},
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_label", "treatment1_group")},
            {"treatment0": ("treatment0_label", ), "treatment1": ("treatment1_label", "treatment1_group")},
            {"treatment0": ("treatment0_label", "treatment0_group"), "treatment1": ("treatment1_group", )},
        ]
    )
    @pytest.mark.parametrize("load_target_covariates", [False, True])
    @pytest.mark.parametrize(
        "target_covariates",
        [
            None,
            {"target0": "one_hot"},
            {"target0": "label"},
            {"target0": "one_hot", "target1": "one_hot"},
            {"target0": "label", "target1": "one_hot"},
            {"target0": "one_hot", "target1": "label"},
            {"target0": "label", "target1": "label"},
            {"target2": "identity"},
            {"target0": "one_hot", "target2": "identity",},
            {"target0": "label", "target2": "identity",},
            {"target0": "one_hot", "target1": "one_hot", "target2": "identity",},
            {"target0": "label", "target1": "one_hot", "target2": "identity",},
            {"target0": "one_hot", "target1": "label", "target2": "identity",},
            {"target0": "label", "target1": "label", "target2": "identity",},
            {"target3": None},
        ]
    )
    @pytest.mark.parametrize("has_controls", [True, False])
    def test_train_dataloader(
        self,
        adata: anndata.AnnData,
        batch_size: int,
        num_genes: int,
        sample_rep: None | str,
        control_key: None | str,
        perturbations: None | str | Sequence[str],
        perturbations_in_obsm: Sequence[str] | None, 
        perturbation_covariates: dict[str, str | Sequence[str]] | None,
        perturbation_reps: dict[str, str | Sequence[str]] | None,
        load_target_covariates: bool,
        target_covariates: dict[str, Literal["one_hot", "label", "identity"] | None] | None,
        has_controls: bool,
    ) -> None:
        """"""
        # handling inputs
        if target_covariates is None:
            load_target_covariates = False

        # we need to be passing the representation
        if perturbation_reps is not None:
            if perturbations is not None:
                perturbations = tuple(perturbation for perturbation in perturbations if perturbation in perturbation_reps.keys())
        else:
            perturbations = None

        # when there are no controls
        if control_key is None:
            has_controls = False

        # when we are using perturbations in obsm, we need to ensure that
        # it appears in `perturbations`
        # also, we need to add the modeled features to the perturbation_reps dictionary
        if perturbations is None or ("treatment2" not in perturbations):
            perturbations_in_obsm = None
        if perturbations_in_obsm is not None:
            if perturbation_reps is None:
                perturbation_reps = {}
            perturbation_reps["treatment2"] = "cov_treatment2_features" # TODO: change "cov" to "feats" once finished the viral notebooks

        # initializing data manager
        data_manager = sc_exp_design.data.DataManager(
            adata,
            sample_rep=sample_rep,
            control_key=control_key,
            perturbations=perturbations,
            perturbations_in_obsm=perturbations_in_obsm,
            perturbation_covariates=perturbation_covariates,
            perturbation_reps=perturbation_reps,
            load_target_covariates=load_target_covariates,
            target_covariates=target_covariates,
            has_controls=has_controls,
        )

        # retrieving data
        data = data_manager.get_data()

        # initialize coupling
        coupling = sc_exp_design.couplings.IndependentCoupling()

        # initialize transforms
        state_transforms = None

        # initializing trainig data loader
        train_dataloader = sc_exp_design.data.TrainDataLoader(
            data,
            coupling,
            batch_size,
            state_transforms,
            has_controls=has_controls,
        )

        # sampling batch of train data and validating it
        train_batch = train_dataloader.sample()
        self.validate_batch(
            train_batch,
            has_controls,
            batch_size,
            num_genes,
            perturbations,
            data,
        ) 
