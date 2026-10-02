"""Tests for :class:`LossGuidedFlow`.

These tests assume :class:`LossGuidedFlow` lives at
``labcompass.inverse.loss_guided_flow`` — adjust the import path if yours differs.

The heavy dependencies (prior flow, velocity field, ``odeint``, ``sdeint``, ``SDE``)
are replaced with lightweight test doubles so the suite runs in a few seconds.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from labcompass.inverse import loss_guided_flow as lgf_module
from labcompass.inverse.loss_guided_flow import LossGuidedFlow


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class DummyVelocityField:
    """Returns a constant velocity, and records each call."""

    def __init__(self, dim: int = 4, vf_value: float = 0.5):
        self.config = MagicMock()
        self.config.flow_dim = dim
        self.dim = dim
        self.vf_value = vf_value
        self.calls = []

    def get_vf_fn(self, cond=None, source=None, cfg_guidance_strength=1.0):
        def _vf(t, x):
            self.calls.append((t.detach().clone(), x.detach().clone()))
            return torch.full_like(x, self.vf_value)
        return _vf

    def eval(self):
        return self


class DummyFlowMatching:
    def __init__(self, dim: int = 4, device: str = "cpu"):
        self.cvf_config = MagicMock()
        self.cvf_config.flow_dim = dim
        self.velocity_field = DummyVelocityField(dim)
        self.device = torch.device(device)
        self.device_id = device
        self.noise_calls = []

    def noise_distribution(self, shape):
        self.noise_calls.append(shape)
        return torch.randn(shape)


class DummyFlowMap:
    def __init__(self, factor: float = 2.0):
        self.factor = factor
        self.calls = []

    def flow_map(self, t0, t1, x):
        self.calls.append((t0.detach().clone(), t1.detach().clone(), x.detach().clone()))
        return x * self.factor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def prior_flow():
    return DummyFlowMatching(dim=4)


@pytest.fixture
def prior_flow_map():
    return DummyFlowMap(factor=2.0)


@pytest.fixture
def lgf(prior_flow):
    return LossGuidedFlow(prior_flow=prior_flow)


@pytest.fixture
def lgf_with_flowmap(prior_flow, prior_flow_map):
    return LossGuidedFlow(prior_flow=prior_flow, prior_flow_map=prior_flow_map)


@pytest.fixture(autouse=True)
def reset_history(lgf, lgf_with_flowmap):
    """Ensure the history stores are always freshly initialized."""
    for obj in (lgf, lgf_with_flowmap):
        obj._loss_history = []
        obj._lambda_history = []
    yield


# ---------------------------------------------------------------------------
# compute_one_step_prediction
# ---------------------------------------------------------------------------

class TestComputeOneStepPrediction:

    def test_euler_step_uses_velocity_field(self, lgf):
        t = torch.tensor([0.25])
        xt = torch.zeros(1, 4)
        out = lgf.compute_one_step_prediction(t, xt)
        # vt = 0.5 everywhere; x1 = xt + (1 - t) * vt = 0.375
        torch.testing.assert_close(out, torch.full((1, 4), 0.375))

    def test_flow_map_shortcut_used_when_available(self, lgf_with_flowmap):
        t = torch.tensor([[0.3]])
        xt = torch.ones(2, 4)
        out = lgf_with_flowmap.compute_one_step_prediction(t, xt)
        torch.testing.assert_close(out, xt * 2.0)
        # Flow map called with t[:, 0] as source and ones as target
        t0, t1, x = lgf_with_flowmap.prior_flow_map.calls[0]
        torch.testing.assert_close(t0, torch.tensor([0.3]))
        torch.testing.assert_close(t1, torch.ones(1))

    def test_flow_map_shortcut_skips_velocity_field(self, lgf_with_flowmap):
        t = torch.tensor([[0.5]])
        xt = torch.zeros(1, 4)
        lgf_with_flowmap.compute_one_step_prediction(t, xt)
        assert lgf_with_flowmap.prior_flow.velocity_field.calls == []


# ---------------------------------------------------------------------------
# compute_loss_from_terminal_state
# ---------------------------------------------------------------------------

class TestComputeLossFromTerminalState:

    def test_loss_only(self, lgf):
        x1 = torch.zeros(3, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        out = lgf.compute_loss_from_terminal_state(x1, loss_fn, c=torch.ones(3))
        torch.testing.assert_close(out, torch.zeros(3))

    def test_loss_with_single_regularizer(self, lgf):
        x1 = torch.ones(3, 4)
        loss_fn = lambda x: torch.zeros(x.shape[0])
        reg = lambda x: x.sum(dim=-1)
        c = torch.full((3,), 2.0)
        out = lgf.compute_loss_from_terminal_state(x1, loss_fn, c, reg_fn_lists=[reg])
        # 0 + 2 * 4 = 8
        torch.testing.assert_close(out, torch.full((3,), 8.0))

    def test_loss_with_multiple_regularizers(self, lgf):
        x1 = torch.ones(2, 3)
        loss_fn = lambda x: torch.zeros(x.shape[0])
        reg1 = lambda x: x.sum(dim=-1)          # 3
        reg2 = lambda x: (x ** 2).sum(dim=-1)   # 3
        c = torch.ones(2)
        out = lgf.compute_loss_from_terminal_state(
            x1, loss_fn, c, reg_fn_lists=[reg1, reg2]
        )
        torch.testing.assert_close(out, torch.full((2,), 6.0))


# ---------------------------------------------------------------------------
# compute_loss_from_interpolation
# ---------------------------------------------------------------------------

class TestComputeLossFromInterpolation:

    def test_default_c_is_one(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        # x1 = xt + 0.5 * 0.5 = 0.25 everywhere -> sum = 1.0
        out = lgf.compute_loss_from_interpolation(t, xt, loss_fn)
        torch.testing.assert_close(out, torch.tensor([1.0]))

    def test_c_scheduler_scales_regularizer(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        reg = lambda x: x.sum(dim=-1)
        c_scheduler = lambda t: torch.full_like(t, 3.0)
        # loss = 1.0, reg = 1.0, c = 3.0 -> 1.0 + 3 * 1.0 = 4.0
        out = lgf.compute_loss_from_interpolation(
            t, xt, loss_fn, reg_fn_lists=[reg], c_scheduler=c_scheduler
        )
        torch.testing.assert_close(out, torch.tensor([4.0]))

    def test_c_scheduler_is_called_with_time(self, lgf):
        t = torch.tensor([0.7])
        xt = torch.zeros(1, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        received = {}

        def c_scheduler(t_in):
            received["t"] = t_in
            return torch.ones_like(t_in)

        lgf.compute_loss_from_interpolation(t, xt, loss_fn, c_scheduler=c_scheduler)
        torch.testing.assert_close(received["t"], t)


# ---------------------------------------------------------------------------
# compute_loss_gradients
# ---------------------------------------------------------------------------

class TestComputeLossGradients:

    def test_returns_loss_and_gradient_with_correct_shapes(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(2, 4, requires_grad=True)
        loss_fn = lambda x: (x ** 2).sum(dim=-1)
        loss, grad = lgf.compute_loss_gradients(t, xt, loss_fn)
        assert loss.shape == (2,)
        assert grad.shape == xt.shape

    def test_gradient_matches_analytic_solution(self, lgf):
        # vf = 0.5, t = 0.5 -> x1 = xt + 0.25
        # loss = sum(x1) -> d loss / d xt = 1 for every element
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        loss, grad = lgf.compute_loss_gradients(t, xt, loss_fn)
        torch.testing.assert_close(grad, torch.ones(1, 4))

    def test_gradient_flows_through_regularizer(self, lgf):
        # loss = sum(x1) + 1 * sum(x1) = 2 * sum(x1) -> grad = 2
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        reg = lambda x: x.sum(dim=-1)
        _, grad = lgf.compute_loss_gradients(t, xt, loss_fn, reg_fn_lists=[reg])
        torch.testing.assert_close(grad, torch.full((1, 4), 2.0))


# ---------------------------------------------------------------------------
# guided_vf_fn
# ---------------------------------------------------------------------------

class TestGuidedVfFn:

    def test_returns_vt_minus_scaled_gradient(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        # vt = 0.5, grad = 1, default strength = 1 -> 0.5 - 1 = -0.5
        out = lgf.guided_vf_fn(t, xt, loss_fn=loss_fn)
        torch.testing.assert_close(out, torch.full((1, 4), -0.5))

    def test_lambda_scheduler_scales_guidance(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        lam = lambda t: torch.full_like(t, 0.25)
        # vt = 0.5, grad = 1, strength = 0.25 -> 0.25
        out = lgf.guided_vf_fn(t, xt, loss_fn=loss_fn, lambda_scheduler=lam)
        torch.testing.assert_close(out, torch.full((1, 4), 0.25))

    def test_populates_history_stores(self, lgf):
        t = torch.tensor([0.5])
        xt = torch.zeros(1, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        lambda_scheduler = lambda t: torch.full_like(t, 0.1)
        lgf.guided_vf_fn(t, xt, loss_fn=loss_fn, lambda_scheduler=lambda_scheduler)
        assert len(lgf._loss_history) == 1
        assert len(lgf._lambda_history) == 1


# ---------------------------------------------------------------------------
# recompute_losses_and_lambda_scheduler
# ---------------------------------------------------------------------------

class TestRecomputeLossesAndLambdaScheduler:

    def test_losses_first_dim_matches_num_time_steps(self, lgf):
        time = torch.linspace(0.0, 1.0, 5)
        traj = torch.zeros(5, 3, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        losses, lambdas = lgf.recompute_losses_and_lambda_scheduler(
            time, traj, loss_fn
        )
        assert losses.shape[0] == 5
        assert lambdas.shape[0] == 5

    def test_does_not_call_vjp(self, lgf):
        """No backward pass should be triggered by this diagnostics method."""
        time = torch.linspace(0.0, 1.0, 4)
        traj = torch.zeros(4, 2, 4)
        loss_fn = lambda x: x.sum(dim=-1)
        with patch(
            "torch.autograd.functional.vjp",
            side_effect=AssertionError("vjp should not be called"),
        ):
            lgf.recompute_losses_and_lambda_scheduler(time, traj, loss_fn)

    def test_runs_under_no_grad(self, lgf):
        """Returned tensors must not carry a grad_fn."""
        time = torch.linspace(0.0, 1.0, 3)
        traj = torch.zeros(3, 2, 4, requires_grad=True)
        loss_fn = lambda x: x.sum(dim=-1)
        losses, lambdas = lgf.recompute_losses_and_lambda_scheduler(
            time, traj, loss_fn
        )
        assert not losses.requires_grad
        assert not lambdas.requires_grad

    def test_lambda_scheduler_applied_per_step(self, lgf):
        time = torch.tensor([0.0, 0.5, 1.0])
        traj = torch.zeros(3, 1, 4)
        loss_fn = lambda x: torch.zeros(x.shape[0])
        lam_sched = lambda t: torch.full_like(t, 2.0)
        _, lambdas = lgf.recompute_losses_and_lambda_scheduler(
            time, traj, loss_fn, lambda_scheduler=lam_sched
        )
        assert torch.all(lambdas == 2.0)

    def test_default_lambda_is_one(self, lgf):
        time = torch.tensor([0.0, 0.5])
        traj = torch.zeros(2, 1, 4)
        loss_fn = lambda x: torch.zeros(x.shape[0])
        _, lambdas = lgf.recompute_losses_and_lambda_scheduler(time, traj, loss_fn)
        assert torch.all(lambdas == 1.0)


# ---------------------------------------------------------------------------
# sample_posterior — ODE path
# ---------------------------------------------------------------------------

class TestSamplePosteriorODE:

    def test_output_shapes(self, lgf):
        N = 3
        T = 5
        loss_fn = lambda x: x.sum(dim=-1)

        def fake_odeint(fn, x0, time, **kwargs):
            # Emulate a solver that evaluates fn at every time step
            for t in time:
                fn(t.unsqueeze(0), x0)
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "odeint", side_effect=fake_odeint):
            traj, loss_hist, lam_hist = lgf.sample_posterior(
                N=N, loss_fn=loss_fn, num_time_steps=T
            )

        assert traj.shape == (T, N, 4)
        assert loss_hist.shape[0] == T
        assert lam_hist.shape[0] == T

    def test_solver_kwargs_defaults(self, lgf):
        captured = {}

        def fake_odeint(fn, x0, time, **kwargs):
            captured.update(kwargs)
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "odeint", side_effect=fake_odeint):
            lgf.sample_posterior(
                N=1, loss_fn=lambda x: x.sum(dim=-1), num_time_steps=3
            )

        assert captured["method"] == "euler"
        assert captured["atol"] == 1e-5
        assert captured["rtol"] == 1e-5

    def test_solver_kwargs_override(self, lgf):
        captured = {}

        def fake_odeint(fn, x0, time, **kwargs):
            captured.update(kwargs)
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "odeint", side_effect=fake_odeint):
            lgf.sample_posterior(
                N=1,
                loss_fn=lambda x: x.sum(dim=-1),
                num_time_steps=3,
                solver_kwargs={"method": "rk4", "atol": 1e-3, "rtol": 1e-3},
            )

        assert captured["method"] == "rk4"
        assert captured["atol"] == 1e-3

    def test_noise_sampled_with_correct_shape(self, lgf):
        def fake_odeint(fn, x0, time, **kwargs):
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "odeint", side_effect=fake_odeint):
            lgf.sample_posterior(
                N=7, loss_fn=lambda x: x.sum(dim=-1), num_time_steps=3
            )

        assert lgf.prior_flow.noise_calls == [(7, 4)]


# ---------------------------------------------------------------------------
# sample_posterior — SDE path
# ---------------------------------------------------------------------------

class TestSamplePosteriorSDE:

    def test_output_shapes(self, lgf):
        N = 3
        T = 6
        loss_fn = lambda x: x.sum(dim=-1)

        def fake_sdeint(sde, x0, time, **kwargs):
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "sdeint", side_effect=fake_sdeint), \
             patch.object(lgf_module, "SDE", MagicMock()):
            traj, loss_hist, lam_hist = lgf.sample_posterior(
                N=N, loss_fn=loss_fn, num_time_steps=T, sde_sampling=True
            )

        assert traj.shape == (T, N, 4)
        assert loss_hist.shape[0] == T
        assert lam_hist.shape[0] == T

    def test_sde_is_constructed_with_vf_fn(self, lgf):
        def fake_sdeint(sde, x0, time, **kwargs):
            return torch.zeros(len(time), *x0.shape)

        sde_cls = MagicMock()
        sde_instance = MagicMock()
        sde_cls.return_value = sde_instance

        with patch.object(lgf_module, "sdeint", side_effect=fake_sdeint), \
             patch.object(lgf_module, "SDE", sde_cls):
            lgf.sample_posterior(
                N=2, loss_fn=lambda x: x.sum(dim=-1),
                num_time_steps=4, sde_sampling=True,
            )

        assert sde_cls.call_count == 1
        # First positional argument is the guided velocity field callable
        assert callable(sde_cls.call_args.args[0])

    def test_recompute_is_invoked(self, lgf):
        def fake_sdeint(sde, x0, time, **kwargs):
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "sdeint", side_effect=fake_sdeint), \
             patch.object(lgf_module, "SDE", MagicMock()), \
             patch.object(
                 lgf, "recompute_losses_and_lambda_scheduler",
                 wraps=lgf.recompute_losses_and_lambda_scheduler,
             ) as mock_recompute:
            lgf.sample_posterior(
                N=2, loss_fn=lambda x: x.sum(dim=-1),
                num_time_steps=4, sde_sampling=True,
            )

        assert mock_recompute.call_count == 1

    def test_recompute_is_not_invoked_for_ode(self, lgf):
        def fake_odeint(fn, x0, time, **kwargs):
            for t in time:
                fn(t.unsqueeze(0), x0)
            return torch.zeros(len(time), *x0.shape)

        with patch.object(lgf_module, "odeint", side_effect=fake_odeint), \
             patch.object(
                 lgf, "recompute_losses_and_lambda_scheduler",
                 wraps=lgf.recompute_losses_and_lambda_scheduler,
             ) as mock_recompute:
            lgf.sample_posterior(
                N=2, loss_fn=lambda x: x.sum(dim=-1),
                num_time_steps=4, sde_sampling=False,
            )

        assert mock_recompute.call_count == 0

    def test_does_not_call_vjp_after_sampling(self, lgf):
        """The SDE diagnostics pass should be free of backpropagation."""
        def fake_sdeint(sde, x0, time, **kwargs):
            return torch.zeros(len(time), *x0.shape)

        # guided_vf_fn still backprops (it needs the guidance gradient), so
        # we only assert that the diagnostics pass after sdeint is not
        # triggering any extra vjp calls.
        with patch.object(lgf_module, "sdeint", side_effect=fake_sdeint), \
             patch.object(lgf_module, "SDE", MagicMock()), \
             patch(
                 "torch.autograd.functional.vjp",
                 side_effect=AssertionError("vjp should not be called"),
             ):
            # NOTE: guided_vf_fn is never invoked because our fake sdeint
            # doesn't call it.
            lgf.sample_posterior(
                N=2, loss_fn=lambda x: x.sum(dim=-1),
                num_time_steps=4, sde_sampling=True,
            )
