from sc_exp_design.inverse.loss_guidance import LossGuidedFlow
from sc_exp_design.inverse.kkt import KKTConditions
from sc_exp_design.models import FlowMatching


class ImpliciDualGuidedFlow(LossGuidedFlow):
    def __init__(
        self,
        prior_flow: FlowMatching,
    ) -> None:
        super().__init__(prior_flow)

    def sample_posterior(
        self,
        N,
        loss_fn,
        ineq_constraints,
        reg_fn_lists=None,
        lambda_scheduler=None,
        c_scheduler=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
        num_time_steps=100,
        solver_kwargs=None,
        eps=1e-6,
        use_lstsq=True,
        g_tol=1e-6,
        use_multipliers=True,
    ):
        
        kkt = KKTConditions(
            loss_fn,
            ineq_constraints,
            eps=eps,
            use_lstsq=use_lstsq,
            g_tol=g_tol,
        )
        def lagrangian(x1):
            if use_multipliers:
                mul = kkt.compute_multipliers(x1)
            else:
                mul = None
            return kkt.compute_lagrangian(x1, mul)

        return super().sample_posterior(
            N,
            lagrangian,
            reg_fn_lists=reg_fn_lists,
            lambda_scheduler=lambda_scheduler,
            c_scheduler=c_scheduler,
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
            num_time_steps=num_time_steps,
            solver_kwargs=solver_kwargs,
        )
