from dataclasses import dataclass

__all__ = ["DataFields", "LossFields", "VFStepFields", "ParamsFields"]


@dataclass(frozen=True)
class DataFields:
    STATE_DATA: str = "state_data"
    PERTURBATION_DATA: str = "condition"
    PERTURBATION_TARGET_REPR: str = "perturbation_target_repr"
    SOURCE_STATE: str = "source"
    TARGET_STATE: str = "target"
    CONDITION_REP: str = "repr"
    CONDITION_COV: str = "cov"
    TARGET_CATEGORIES: str = "target_categories"


@dataclass(frozen=True)
class LossFields:
    LOSS: str = "loss"
    VF_LOSS: str = "vf_loss"
    SCORE_LOSS: str = "score_loss"
    SOURCE_LOSS: str = "src_posterior_loss"
    TARGET_LOSS: str = "tgt_posterior_loss"
    PERTURBATION_LOSS: str = "pert_posterior_loss"
    LATENT_PERTURBATION_REG_LOSS: str = "latent_pert_regulation"
    LATENT_PERTURBATION_INF_LOSS: str = "latent_pert_inference_loss"


@dataclass(frozen=True)
class VFStepFields:
    VF: str = "vf"
    SCORE: str = "score"
    LATENT_REPR: str = "latent_repr"
    LATENT_STATE: str = "xt_latent"
    LATENT_TIME: str = "t_latent"
    LATENT_PERTURBATION: str = "cond_latent"
    SOURCE_PARAMS: str = "src_posterior_params"
    TARGET_PARAMS: str = "tgt_posterior_params"
    PERTURBATION_PARAMS: str = "pert_posterior_params"


@dataclass(frozen=True)
class ParamsFields:
    MEAN: str = "mean"
    COVARIANCE: str = "covariance"
