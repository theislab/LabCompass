from collections.abc import Sequence
from typing import Any, Literal

import pytest

from sc_exp_design.constants import DataFields
from sc_exp_design.data import DataContainer


@pytest.mark.parametrize("sample_rep", [None, "states"])
@pytest.mark.parametrize("control_key", [None, "is_control"])
@pytest.mark.parametrize("perturbations", [None, ("treatment0",), ("treatment0", "treatment1"), ("treatment0", "treatment1", "treatment2"), ])
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
        {"treatment2": ("treatment2_features",)}
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
        {"target3": "identity"},
    ]
)
class BaseDataTest:
    """"""