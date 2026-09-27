from labcompass.inverse.loss_guidance import LossGuidedFlow
from labcompass.inverse.kkt import KKTConditions
from labcompass.models import FlowMatching


class ImpliciDualGuidedFlow(LossGuidedFlow):
    """Initializes :class:`ImpliciDualGuidedFlow`, a :class:`LossGuidedFlow` variant that guides sampling towards a KKT-constrained Lagrangian instead of a plain loss.

    :param prior_flow: The pretrained :class:`FlowMatching` model whose velocity field is guided during sampling.
    :type prior_flow: class:`FlowMatching`
    """

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
        """Draw `N` guided samples using the Lagrangian of `loss_fn` under `ineq_constraints` as the guidance loss.

        At each guidance step, a :class:`KKTConditions` instance built from `loss_fn` and
        `ineq_constraints` is used to (optionally) solve for the Lagrange multipliers of the active
        constraints at the predicted terminal state, and the resulting Lagrangian is passed as the
        guidance loss to :meth:`LossGuidedFlow.sample_posterior`.

        :param N: Number of samples to generate.
        :type N: class:`int`

        :param loss_fn: Task-specific primal objective evaluated at the predicted terminal state.
        :type loss_fn: class:`Callable[[Tensor], Tensor]`

        :param ineq_constraints: List of inequality constraint functions `g_i(x1) <= 0` used to build
            the :class:`KKTConditions` Lagrangian.
        :type ineq_constraints: class:`list[Callable[[Tensor], Tensor]]`

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

        :param num_time_steps: Number of discretization steps used to integrate the ODE, defaults to `100`.
        :type num_time_steps: class:`int`

        :param solver_kwargs: Keyword arguments passed to :func:`torchdiffeq.odeint`, defaults to `None`.
        :type solver_kwargs: class:`dict[str, Any] | None`

        :param eps: Small value added to the diagonal of the multiplier linear system, for numerical
            stability, defaults to `1e-6`.
        :type eps: class:`float`

        :param use_lstsq: Whether to solve the multiplier system with :func:`torch.linalg.lstsq` instead
            of explicit matrix inversion, defaults to `True`.
        :type use_lstsq: class:`bool`

        :param g_tol: Tolerance below which a constraint is considered active, defaults to `1e-6`.
        :type g_tol: class:`float`

        :param use_multipliers: Whether to solve for the Lagrange multipliers of the active constraints
            via :meth:`KKTConditions.compute_multipliers`; when `False`, the Lagrangian reduces to the
            unconstrained `loss_fn`, defaults to `True`.
        :type use_multipliers: class:`bool`

        :return: A `(trajectory, loss_history, lambda_history)` tuple, see
            :meth:`LossGuidedFlow.sample_posterior`.
        :rtype: class:`tuple[np.ndarray, Tensor, Tensor]`
        """
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
