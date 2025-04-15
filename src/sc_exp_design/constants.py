from dataclasses import dataclass

__all__ = ["DataFields", "LossFields", "VFStepFields", "ParamsFields"]


@dataclass(frozen=True)
class DataFields:
    STATE_DATA: str = "state_data"
    PERTURBATION_DATA: str = "condition"
    TARGET_DATA: str = "target_data"
    SOURCE_STATE: str = "source"
    TARGET_STATE: str = "target"
    CONDITION_REP: str = "repr"
    CONDITION_COV: str = "cov"
    TARGET_CATEGORIES: str = "target_categories"


@dataclass(frozen=True)
class PredictionFields:
    PREDICTION_DATA: str = "predicted_states"
    TARGET_PREDICTION_DATA: str = "target_prediction_data"
    PREDICTED_PERTURBATION: str = "predicted_perturbation"


@dataclass(frozen=True)
class LossFields:
    LOSS: str = "loss"
    VF_LOSS: str = "vf_loss"
    PERTURBATION_LOSS: str = "pert_posterior_loss"


@dataclass(frozen=True)
class VFStepFields:
    VF: str = "vf"
    LATENT_REPR: str = "latent_repr"
    LATENT_STATE: str = "xt_latent"
    LATENT_TIME: str = "t_latent"
    LATENT_PERTURBATION: str = "cond_latent"


@dataclass(frozen=True)
class ParamsFields:
    MEAN: str = "mean"
    COVARIANCE: str = "covariance"
