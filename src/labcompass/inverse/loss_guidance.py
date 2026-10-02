from functools import partial

import torch
from torchdiffeq import odeint
from torchsde import sdeint

from labcompass.inverse.sde import SDE
from labcompass.models import FlowMap, FlowMatching
from labcompass.networks import NeuralVelocityField
from labcompass.utils import match_shapes

__all__ = ["LossGuidedFlow"]


class LossGuidedFlow:
    """Initializes the :class:`LossGuidedFlow`, a training-free guidance wrapper around a pretrained flow-matching model.

    At sampling time, the base velocity field of `prior_flow` is steered towards minimizing a task-specific
    loss evaluated on a one-step estimate of the terminal state, by subtracting a (scheduled) gradient of
    that loss from the unguided velocity.

    :param prior_flow: The pretrained :class:`FlowMatching` model whose velocity field is guided during sampling.
    :type prior_flow: class:`FlowMatching`

    :param prior_flow_map: Optional flow map used to directly predict the terminal state from an
        intermediate state, instead of taking a single Euler step with the velocity field, defaults to `None`.
    :type prior_flow_map: class:`FlowMap | None`
    """

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
        """Predict the terminal state `x1` from the intermediate state `xt` at time `t`.

        If :attr:`prior_flow_map` is set, the terminal state is obtained directly via its `t -> 1`
        transport map. Otherwise it is estimated with a single Euler step along the (optionally
        classifier-free-guided) velocity field, `x1 = xt + (1 - t) * v_t`, assuming a linear (rectified)
        probability path.

        :param t: Time(s) at which `xt` is evaluated.
        :type t: class:`float | Tensor`

        :param xt: The intermediate state, of shape `(batch_size, flow_dim)`.
        :type xt: class:`Tensor`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :return: The predicted terminal state `x1`.
        :rtype: class:`Tensor`
        """
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
        """Compute the guidance loss at the terminal state `x1`, adding any regularization terms weighted by `c`.

        The returned loss is `loss_fn(x1) + c * sum(reg_fn(x1) for reg_fn in reg_fn_lists)`.

        :param x1: The (predicted) terminal state, of shape `(batch_size, flow_dim)`.
        :type x1: class:`Tensor`

        :param loss_fn: Task-specific loss function evaluated at `x1`.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param c: Weight(s) applied to each regularization term in `reg_fn_lists`.
        :type c: class:`Tensor`

        :param reg_fn_lists: Optional list of additional regularization functions evaluated at `x1`,
            defaults to `None`, in which case no regularization term is added.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :return: The (regularized) loss value.
        :rtype: class:`Tensor`
        """
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
        """Compute the guidance loss at time `t` from the intermediate state `xt`.

        The terminal state is first estimated with :meth:`compute_one_step_prediction`, and the loss
        (plus any regularization terms, weighted by `c_scheduler(t)` if provided, otherwise `1`) is then
        evaluated at that terminal state via :meth:`compute_loss_from_terminal_state`.

        :param t: Time(s) at which `xt` is evaluated.
        :type t: class:`float | Tensor`

        :param xt: The intermediate state, of shape `(batch_size, flow_dim)`.
        :type xt: class:`Tensor`

        :param loss_fn: Task-specific loss function evaluated at the predicted terminal state.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param reg_fn_lists: Optional list of additional regularization functions, defaults to `None`.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :param c_scheduler: Optional function of `t` returning the weight of the regularization terms,
            defaults to `None`, in which case a weight of `1` is used.
        :type c_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :return: The (regularized) loss value at the predicted terminal state.
        :rtype: class:`Tensor`
        """
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
        """Compute the guidance loss and its gradient with respect to `xt`.

        Internally calls :meth:`compute_loss_from_interpolation` and differentiates it with respect to
        `xt` using :func:`torch.autograd.functional.vjp` with an all-ones cotangent vector, which is
        equivalent to backpropagating the per-sample loss with unit gradient weights.

        :param t: Time(s) at which `xt` is evaluated.
        :type t: class:`float | Tensor`

        :param xt: The intermediate state, of shape `(batch_size, flow_dim)`.
        :type xt: class:`Tensor`

        :param loss_fn: Task-specific loss function evaluated at the predicted terminal state.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param reg_fn_lists: Optional list of additional regularization functions, defaults to `None`.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :param c_scheduler: Optional function of `t` returning the weight of the regularization terms,
            defaults to `None`.
        :type c_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :return: A `(loss, gradient)` tuple, where `gradient` is the gradient of the loss with respect
            to `xt`.
        :rtype: class:`tuple[Tensor, Tensor]`
        """
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
        """Compute the loss-guided velocity at `(t, xt)`, used as the vector field for guided sampling.

        The unguided (optionally classifier-free-guided) velocity `vt` is computed from
        :attr:`prior_flow`'s velocity field, and a guidance term `guidance_strength * gt` is subtracted
        from it, where `gt` is the gradient of the guidance loss (from :meth:`compute_loss_gradients`)
        and `guidance_strength` is `lambda_scheduler(t)` if provided, otherwise `1`. The loss value and
        guidance strength for this call are appended to :attr:`_loss_history` and :attr:`_lambda_history`.

        :param t: Time(s) at which `xt` is evaluated.
        :type t: class:`float | Tensor`

        :param xt: The current state, of shape `(batch_size, flow_dim)`.
        :type xt: class:`Tensor`

        :param loss_fn: Task-specific loss function evaluated at the predicted terminal state, defaults to `None`.
        :type loss_fn: class:`Callable[[Tensor], Tensor] | None`

        :param reg_fn_lists: Optional list of additional regularization functions, defaults to `None`.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :param lambda_scheduler: Optional function of `t` returning the guidance strength, defaults to
            `None`, in which case a guidance strength of `1` is used.
        :type lambda_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param c_scheduler: Optional function of `t` returning the weight of the regularization terms,
            defaults to `None`.
        :type c_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :return: The guided velocity `vt - guidance_strength * gt`.
        :rtype: class:`Tensor`
        """
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

    def recompute_losses_and_lambda_scheduler(
        self,
        time,
        traj,
        loss_fn,
        reg_fn_lists=None,
        lambda_scheduler=None,
        c_scheduler=None,
        cond=None,
        source=None,
        cfg_guidance_strength=1.0,
    ):
        """Recompute the guidance loss and guidance strength along a given trajectory.

        SDE solvers may evaluate the guided velocity field at times that do not coincide with the
        requested `time` grid, and adaptive steppers may take a variable number of steps. This method
        re-evaluates the guidance loss and the scheduled guidance strength at each time step of the
        supplied discretization, so that the reported histories are always aligned with `time`.

        Unlike :meth:`guided_vf_fn`, this method only needs the scalar loss value and therefore calls
        :meth:`compute_loss_from_interpolation` directly (rather than :meth:`compute_loss_gradients`),
        avoiding the cost of a vector-Jacobian product / backward pass. The evaluation is additionally
        wrapped in :func:`torch.no_grad` so that no autograd graph is retained.

        :param time: The discretization times, of shape `(num_time_steps,)`.
        :type time: class:`Tensor`

        :param traj: The trajectory of states, of shape `(num_time_steps, batch_size, flow_dim)`.
        :type traj: class:`Tensor`

        :param loss_fn: Task-specific loss function evaluated at the predicted terminal state.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param reg_fn_lists: Optional list of additional regularization functions, defaults to `None`.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :param lambda_scheduler: Optional function of `t` returning the guidance strength, defaults to
            `None`, in which case a guidance strength of `1` is used.
        :type lambda_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param c_scheduler: Optional function of `t` returning the weight of the regularization terms,
            defaults to `None`.
        :type c_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :return: A `(losses, lambdas)` tuple, where `losses` has shape
            `(num_time_steps, batch_size)` and `lambdas` has shape `(num_time_steps, batch_size, 1)`
            (or `(num_time_steps,)` if the scheduler returns scalars).
        :rtype: class:`tuple[Tensor, Tensor]`
        """
        losses = []
        lambdas = []
        with torch.no_grad():
            for idx, t in enumerate(time):
                xt = traj[idx, :, :]
                loss = self.compute_loss_from_interpolation(
                    t,
                    xt,
                    loss_fn,
                    reg_fn_lists=reg_fn_lists,
                    cond=cond,
                    source=source,
                    cfg_guidance_strength=cfg_guidance_strength,
                    c_scheduler=c_scheduler,
                )
                losses.append(loss)
                t_matched = match_shapes(t, xt)
                lam = lambda_scheduler(t_matched) if lambda_scheduler is not None else torch.ones_like(t_matched)
                lambdas.append(lam)
        losses = torch.stack(losses, dim=0)
        lambdas = torch.stack(lambdas, dim=0)
        return losses, lambdas

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
        sde_sampling=False,
    ):
        """Draw `N` guided samples by integrating the loss-guided velocity field from noise to data.

        An initial state is sampled from :attr:`prior_flow`'s noise distribution and integrated from
        `t=0` to `t=1` over `num_time_steps` steps, using :meth:`guided_vf_fn` as the vector field.
        If `sde_sampling` is `False`, the (deterministic) ODE is integrated with
        :func:`torchdiffeq.odeint`. If `sde_sampling` is `True`, the velocity field is wrapped in an
        :class:`SDE` and integrated with :func:`torchsde.sdeint`; since the SDE solver's internal
        evaluations do not align with `time`, the loss / lambda histories are recomputed on the
        resulting trajectory via :meth:`recompute_losses_and_lambda_scheduler`. `solver_kwargs`
        defaults to Euler integration (`{"method": "euler", "atol": 1e-5, "rtol": 1e-5}`) for any key
        not explicitly provided.

        :param N: Number of samples to generate.
        :type N: class:`int`

        :param loss_fn: Task-specific loss function evaluated at the predicted terminal state at each step.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param reg_fn_lists: Optional list of additional regularization functions, defaults to `None`.
        :type reg_fn_lists: class:`list[Callable[[Tensor], Tensor]] | None`

        :param lambda_scheduler: Optional function of `t` returning the guidance strength, defaults to `None`.
        :type lambda_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param c_scheduler: Optional function of `t` returning the weight of the regularization terms,
            defaults to `None`.
        :type c_scheduler: class:`Callable[[Tensor], Tensor] | None`

        :param cond: Optional conditioning information passed to the velocity field, defaults to `None`.
        :type cond: class:`dict[str, Tensor] | None`

        :param source: Optional source state passed to the velocity field, defaults to `None`.
        :type source: class:`Tensor | None`

        :param cfg_guidance_strength: Strength of the classifier-free-guidance term, defaults to `1.0`.
        :type cfg_guidance_strength: class:`float`

        :param num_time_steps: Number of discretization steps used to integrate the ODE/SDE, defaults to `100`.
        :type num_time_steps: class:`int`

        :param solver_kwargs: Keyword arguments passed to :func:`torchdiffeq.odeint` (or
            :func:`torchsde.sdeint`), defaults to `None`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param sde_sampling: Whether to integrate the guided dynamics as an SDE (via
            :func:`torchsde.sdeint`) rather than an ODE (via :func:`torchdiffeq.odeint`),
            defaults to `False`.
        :type sde_sampling: class:`bool`

        :return: A `(trajectory, loss_history, lambda_history)` tuple. `trajectory` is a
            :class:`numpy.ndarray` of shape `(num_time_steps, N, flow_dim)` with the integrated states at
            each time step; `loss_history` and `lambda_history` stack, respectively, the guidance loss
            and guidance strength recorded at every call to :meth:`guided_vf_fn` during integration
            (or recomputed along `time` when `sde_sampling=True`).
        :rtype: class:`tuple[np.ndarray, Tensor, Tensor]`
        """
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

        if sde_sampling:
            sde = SDE(vf_fn)
            traj = sdeint(sde, x0, time, **solver_kwargs)
            loss_history, lambda_history = self.recompute_losses_and_lambda_scheduler(
                time,
                traj,
                loss_fn,
                reg_fn_lists=reg_fn_lists,
                lambda_scheduler=lambda_scheduler,
                c_scheduler=c_scheduler,
                cond=cond,
                source=source,
                cfg_guidance_strength=cfg_guidance_strength,
            )
            traj = traj.detach().cpu().numpy()
            loss_history = loss_history.detach().cpu().numpy()
            lambda_history = lambda_history.detach().cpu().numpy()
        else:
            traj = odeint(
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
        """The velocity field of :attr:`prior_flow`.

        :return: The underlying :class:`NeuralVelocityField` used for unguided predictions.
        :rtype: class:`NeuralVelocityField`
        """
        return self.prior_flow.velocity_field
        