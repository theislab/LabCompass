from sc_exp_design.metrics.metrics import (
    compute_e_distance,
    compute_r_squared,
    compute_sinkhorn_div,
)

METRICS = {
    "r_squared": compute_r_squared,
    "sinkhorn_div": compute_sinkhorn_div,
    "e_distance": compute_e_distance,
}

__all__ = [
    "compute_r_squared",
    "compute_sinkhorn_div",
    "compute_e_distance",
]
