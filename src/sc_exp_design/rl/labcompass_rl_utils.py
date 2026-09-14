"""Helpers to fine-tune the design *prior* with Flow-GRPO using a phenotype reward.

This packs the frozen surrogate `forward_model` + phenotype `loss_fns` + target
phenotype into a single `RewardModel(designs) -> reward[B]`, and adapts the
`FlowMatching` prior to the `prior_flow(x, t) -> velocity` contract expected by
`flow_grpo_finetune.train_flow_grpo`.

Conventions
-----------
sc_exp_design flow:  t=0 -> noise, t=1 -> data, integrate 0->1, v_sc = data-noise.
flow_grpo         :  t=1 -> noise, t=0 -> data, integrate 1->0, v = eps - x0.
=> policy maps   v_grpo(x, t) = -v_sc(x, 1 - t)   (derived + checked via eta=0 ODE).
"""

from __future__ import annotations
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.networks.blocks import ConditionEncoder


# --------------------------------------------------------------------------- #
# 0. fix for cloudpickled-by-value pooling lambdas (foreign-Python bytecode).
# --------------------------------------------------------------------------- #
def patch_pickled_lambdas(flow_matching) -> int:
    """Rebuild ConditionEncoder pooling lambdas (segfault under a different
    Python than the one that pickled the checkpoint). Call on any loaded model
    whose velocity field runs a forward pass."""
    n = 0
    for m in flow_matching.velocity_field.modules():
        if isinstance(m, ConditionEncoder):
            if getattr(m, "pooling", None) == "mean":
                m.pooling_layer = lambda t: torch.mean(t, dim=1); n += 1
            elif getattr(m, "pooling", None) == "sum":
                m.pooling_layer = lambda t: torch.sum(t, dim=1); n += 1
    return n


# --------------------------------------------------------------------------- #
# 1. target phenotype (== cond2target[target_comb]) straight from the adata
# --------------------------------------------------------------------------- #
def build_target_phenotype(validation_adata, ohe_region, target_comb: str) -> np.ndarray:
    """Empirical Region proportions of the REAL validation cells under a
    condition. Identical to the notebook's cond2target[target_comb], but does
    not require running the surrogate over every condition."""
    mask = (validation_adata.obs["condition"] == target_comb).values
    if mask.sum() == 0:
        raise ValueError(f"No real cells with condition == {target_comb!r}")
    regions = validation_adata.obs["Region"].values[mask].reshape(-1, 1)
    return ohe_region.transform(regions).mean(0, keepdims=True)  # [1, n_regions]


# --------------------------------------------------------------------------- #
# 2. Reward model  =  forward_model  +  loss_fns  +  target phenotype
# --------------------------------------------------------------------------- #
class RewardModel(nn.Module):
    """reward(designs[B, D]) = -phenotype_cross_entropy(target)   (higher=better).

    Mirrors LossGuidedFlow.compute_target_loss (fix_noise path) but returns the
    reward value only (no gradient needed for GRPO).
    """

    def __init__(
        self,
        forward_model,
        loss_fns: dict[str, Callable],
        target: dict[str, np.ndarray | torch.Tensor],
        num_forward_pass_per_sample: int = 64,
        n_time_steps_forward_model: int = 20,
        solver_kwargs: dict | None = None,
        non_linearity: nn.Module | Callable | None = None,
        uncertainty_weight: float = 0.0,
        patch: bool = True,
    ) -> None:
        super().__init__()
        if patch:
            patch_pickled_lambdas(forward_model.forward_model)
        # weight on the predictive-uncertainty penalty: reward = -(mean_loss + w * sigma)
        self.uncertainty_weight = float(uncertainty_weight)
        self._fm = forward_model                                   # plain ref (not a Module)
        # register the nets so .to()/.eval()/parameter-freezing reach them
        self.surrogate_vf = forward_model.forward_model.velocity_field
        self.target_predictor = forward_model.target_prediction_model
        self.loss_fns = loss_fns
        self.non_linearity = non_linearity if non_linearity is not None else nn.Identity()
        self.num_forward_pass_per_sample = num_forward_pass_per_sample
        self.n_time_steps_forward_model = n_time_steps_forward_model
        self.solver_kwargs = solver_kwargs or {"method": "euler"}
        self.surrogate_flow_dim = forward_model.forward_model.velocity_field.config.flow_dim
        self.pert_key = "repr_condition_conditions"
        # store the target(s) as buffers (moved by .to())
        for cov, val in target.items():
            t = torch.as_tensor(np.asarray(val), dtype=torch.float32)
            self.register_buffer(f"target_{cov}", t)
        self._target_covs = list(target.keys())

    @torch.no_grad()
    def forward(self, designs: torch.Tensor) -> torch.Tensor:
        dev = next(self.surrogate_vf.parameters()).device
        designs = designs.to(dev).float()
        B = designs.shape[0]
        S = self.num_forward_pass_per_sample

        x1 = self.non_linearity(designs)                          # [B, D]
        x1_fwd = x1.unsqueeze(1).repeat(1, S, 1)                  # [B, S, D]
        noise = self._fm.forward_model.noise_distribution(
            (B, S, self.surrogate_flow_dim)
        ).to(dev).float()                                         # [B, S, Dx]

        batch = {
            DataFields.SOURCE_STATE: noise,
            DataFields.PERTURBATION_DATA: {self.pert_key: x1_fwd},
        }
        out = self._fm.predict(
            batch, no_grad=True, fix_noise=True,
            num_time_steps=self.n_time_steps_forward_model,
            solver_kwargs=self.solver_kwargs,
        )
        pred = out[PredictionFields.TARGET_PREDICTION_DATA]       # {cov: [B, S, C]}

        total_mean = torch.zeros(B, device=dev)
        total_std = torch.zeros(B, device=dev)
        for cov in self._target_covs:
            tgt = getattr(self, f"target_{cov}")                 # [1, C]
            tgt = tgt.view(1, 1, -1).expand(B, S, -1)
            per = self.loss_fns[cov](pred[cov], tgt)             # [B, S] per-cell loss
            total_mean = total_mean + per.mean(dim=1)            # [B] expected loss
            total_std = total_std + per.std(dim=1)               # [B] predictive uncertainty (sigma)
        # reward = -(loss + w * sigma): minimize both expected loss AND its spread
        return -(total_mean + self.uncertainty_weight * total_std)


# --------------------------------------------------------------------------- #
# 3. Policy adapter:  FlowMatching prior  ->  prior_flow(x, t) -> velocity
# --------------------------------------------------------------------------- #
class FlowPolicy(nn.Module):
    """Wraps a FlowMatching prior's velocity_field so train_flow_grpo can call
    it as `policy(x, t) -> velocity` with the flow-grpo time convention.

    Shares the SAME velocity_field parameters as the prior, so GRPO updates the
    prior in place (prior_flow.predict reflects the fine-tuning afterwards).
    """

    def __init__(self, flow_matching) -> None:
        super().__init__()
        self.velocity_field = flow_matching.velocity_field        # registered submodule
        self.flow_dim = flow_matching.cvf_config.flow_dim

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # flow-grpo t (1=noise,0=data) -> sc time s (0=noise,1=data)
        s = 1.0 - t
        v_sc = self.velocity_field.forward(s, x, cond=None, source=None)
        return -v_sc
