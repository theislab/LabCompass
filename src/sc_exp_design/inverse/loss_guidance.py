from functools import partial

import torch
from torchdiffeq import odeint

from sc_exp_design.utils import match_shapes
from sc_exp_design.networks import NeuralVelocityField
from sc_exp_design.models import FlowMatching, FlowMap


__all__ = ["LossGuidedFlow"]


class LossGuidedFlow:
    def __init__(
        self,
        prior_flow: FlowMatching,
        prior_flow_map: FlowMap | None = None
    ) -> None:
        self.prior_flow = prior_flow
        self.prior_flow_map = prior_flow_map

    def compute_one_step_prediction(
        self,
        t,
        xt,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0
    ):
        # computing velocity field
        if self.prior_flow_map is not None:
            t_input = t[..., 0]
            return self.prior_flow_map.flow_map(
                t_input,
                torch.ones_like(t_input),
                xt
            )
        t = match_shapes(t, xt)
        vf_fn = self.prior_vf.get_vf_fn(
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
        )
        vt = vf_fn(t[:, 0], xt)
        return xt + (1 - t)*vt

    def compute_loss_from_terminal_state(
        self,
        x1,
        loss_fn,
        c,
        reg_fn_lists=None,
    ):
        reg_fn_lists = [] if reg_fn_lists is None else reg_fn_lists
        loss = loss_fn(x1)
        for reg_fn in reg_fn_lists:
            loss = loss + c*reg_fn(x1)
        return loss

    def compute_loss_from_interpolation(
        self,
        t,
        xt,
        loss_fn,
        reg_fn_lists=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
        c_scheduler=None,
    ):
        c = torch.ones_like(t) if c_scheduler is None else c_scheduler(t) 
        x1 = self.compute_one_step_prediction(
            t,
            xt,
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
        )
        return self.compute_loss_from_terminal_state(
            x1,
            loss_fn,
            c,
            reg_fn_lists=reg_fn_lists,
        )

    def compute_loss_gradients(
        self,
        t,
        xt,
        loss_fn,
        reg_fn_lists=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
        c_scheduler=None,
    ):
        def _compute_loss(xt):
            return self.compute_loss_from_interpolation(
                t,
                xt,
                loss_fn,
                reg_fn_lists=reg_fn_lists,
                cond=cond,
                source=source,
                cfg_guidance_strength=cfg_guidance_strength,
                c_scheduler=c_scheduler,
            )
        return torch.autograd.functional.vjp(_compute_loss, xt, torch.ones((xt.shape[0],), device=xt.device))

    def guided_vf_fn(
        self,
        t,
        xt,
        loss_fn=None,
        reg_fn_lists=None,
        lambda_scheduler=None,
        c_scheduler=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
    ):
        # computing unguided vf
        t = match_shapes(t, xt)
        vf_fn = self.prior_flow.velocity_field.get_vf_fn(
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
        )
        vt = vf_fn(t[:, 0], xt)

        # computing guidance term
        loss_val, gt = self.compute_loss_gradients(
            t,
            xt,
            loss_fn,
            reg_fn_lists,
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
            c_scheduler=c_scheduler,
        )

        # optional decay
        guidance_strength = lambda_scheduler(t) if lambda_scheduler is not None else torch.ones_like(t)

        # update stores
        self._loss_history.append(loss_val)
        self._lambda_history.append(guidance_strength)

        return vt - guidance_strength*gt

    def sample_posterior(
        self,
        N,
        loss_fn,
        reg_fn_lists=None,
        lambda_scheduler=None,
        c_scheduler=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
        num_time_steps=100,
        solver_kwargs=None,
    ):

        # create store for loss function and lambda schedueler values
        self._loss_history = []
        self._lambda_history = [] 

        # default values for the solver arguments
        if solver_kwargs is None:
            solver_kwargs = {}
        solver_kwargs.setdefault("method", "euler")
        solver_kwargs.setdefault("atol", 1e-5)
        solver_kwargs.setdefault("rtol", 1e-5)

        x0 = self.prior_flow.noise_distribution(
            (N, self.prior_flow.cvf_config.flow_dim)
        ).float().to(self.prior_flow.device)

        vf_fn = partial(
            self.guided_vf_fn,
            loss_fn=loss_fn,
            lambda_scheduler=lambda_scheduler,
            c_scheduler=c_scheduler,
            reg_fn_lists=reg_fn_lists,
            cond=cond,
            source=source,
            cfg_guidance_strength=cfg_guidance_strength,
        )
        time = torch.linspace(0.0, 1.0, num_time_steps)
        traj =  odeint(
            vf_fn,
            x0,
            time,
            **solver_kwargs
        ).detach().cpu().numpy()
        loss_history = torch.stack(self._loss_history, dim=0)
        lambda_history = torch.stack(self._lambda_history, dim=0)
        return traj, loss_history, lambda_history

    @property
    def prior_vf(self) -> NeuralVelocityField:
        return self.prior_flow.velocity_field
