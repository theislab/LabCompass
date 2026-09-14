"""
Flow-GRPO fine-tuning of a flow-matching prior from a reward model.

Setup this script assumes (edit the two adapter functions to match your objects):
  * prior_flow(x, t) -> velocity, same shape as x.
        x : [B, *data_shape]   the current latent
        t : [B]                time in [0, 1], 1 = pure noise, 0 = data
  * reward_model(condition) -> reward [B]   (higher is better)
        condition == the sample produced by prior_flow (final x at t = 0)

Convention (rectified / linear flow matching):
  x_t = (1 - t) * x0 + t * eps,   v = eps - x0 = dx_t/dt,
  sampling runs t: 1 -> 0, so dt < 0.
  With noise_level (eta) = 0 the SDE step reduces to the deterministic Euler ODE,
  which is the check that your velocity/dt sign convention is correct.

The SDE step + log-prob below is the scheduler-free equivalent of
flow_grpo/diffusers_patch/sd3_sde_with_logprob.py (sde_type='sde'). It is
mathematically identical; it just takes sigma/dt directly instead of reading
them from a diffusers FlowMatchEulerDiscreteScheduler.
"""

import math
import copy
import torch
import torch.nn as nn


# ======================================================================
# 1. Flow-GRPO SDE step with Gaussian log-prob  (the only flow-specific part)
# ======================================================================
def sde_step_with_logprob(v, sigma, dt, sample, sigma_max,
                          noise_level=0.7, prev_sample=None, generator=None):
    """One SDE denoising step and the log-prob of the transition.

    v          : velocity from the flow model,         [B, *shape]
    sigma      : current time t, broadcastable,        [1 or B, 1, 1, ...]
    dt         : t_next - t (negative),                [1, 1, ...]
    sample     : current latent x_t,                   [B, *shape]
    sigma_max  : substitute value for the sigma==1 step to avoid /0
    prev_sample: if given (update phase), score THIS transition instead of
                 drawing a fresh one.
    returns    : prev_sample, log_prob[B], prev_sample_mean, std
    """
    v = v.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    # at t == 1 (pure noise) 1 - sigma == 0 -> clamp, exactly as the official code
    denom = 1.0 - torch.where(sigma == 1, torch.full_like(sigma, sigma_max), sigma)
    std_dev_t = torch.sqrt(sigma / denom) * noise_level

    prev_sample_mean = (
        sample * (1 + std_dev_t ** 2 / (2 * sigma) * dt)
        + v * (1 + std_dev_t ** 2 * (1 - sigma) / (2 * sigma)) * dt
    )
    std = std_dev_t * torch.sqrt(-dt)

    if prev_sample is None:
        noise = torch.randn(v.shape, generator=generator, device=v.device, dtype=v.dtype)
        prev_sample = prev_sample_mean + std * noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std ** 2)
        - torch.log(std)
        - 0.5 * math.log(2 * math.pi)
    )
    # mean over all non-batch dims (official uses mean, not sum)
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, prev_sample_mean, std


# ======================================================================
# 2. Config
# ======================================================================
class FlowGRPOConfig:
    T             = 10      # denoising steps during training (denoising reduction)
    batch_size    = 256     # samples per rollout = one group (unconditional case)
    noise_level   = 0.7     # eta; MUST be > 0 or every step is deterministic (ratio == 1)
    clip_eps      = 0.2     # PPO clip range
    inner_epochs  = 4       # off-policy reuse of each rollout
    lr            = 3e-4
    kl_coef       = 0.0     # >0 keeps the policy near the frozen prior (anti reward-hacking)
    adv_eps       = 1e-4
    max_grad_norm = 1.0
    total_iters   = 400


# ======================================================================
# 3. Rollout: sample a batch of trajectories and their per-step log-probs
# ======================================================================
@torch.no_grad()
def sample_trajectory(prior_flow, batch_size, data_shape, cfg, device):
    ts = torch.linspace(1.0, 0.0, cfg.T + 1, device=device)   # t: 1 -> 0
    sigma_max = float(ts[1])                                   # substitute for the t==1 step
    bshape = (1, *([1] * len(data_shape)))                     # broadcast shape for sigma/dt

    x = torch.randn(batch_size, *data_shape, device=device)   # pure noise at t = 1
    latents, log_probs = [x], []
    for i in range(cfg.T):
        t = ts[i].expand(batch_size)
        sigma = ts[i].view(bshape)
        dt = (ts[i + 1] - ts[i]).view(bshape)
        v = prior_flow(x, t)
        x, lp, _, _ = sde_step_with_logprob(
            v, sigma, dt, x, sigma_max, noise_level=cfg.noise_level)
        latents.append(x)
        log_probs.append(lp)

    condition = x                                    # final sample at t = 0
    latents = torch.stack(latents, dim=1)            # [B, T+1, *shape]
    log_probs = torch.stack(log_probs, dim=1)        # [B, T]
    return condition, latents, log_probs, ts, sigma_max


# ======================================================================
# 4. Recompute per-step log-prob (with grad) + optional KL to the frozen prior
# ======================================================================
def step_log_prob(prior_flow, ref_flow, latents, ts, sigma_max, i, cfg):
    B = latents.shape[0]
    data_shape = latents.shape[2:]
    bshape = (1, *([1] * len(data_shape)))
    t = ts[i].expand(B)
    sigma = ts[i].view(bshape)
    dt = (ts[i + 1] - ts[i]).view(bshape)

    x_t = latents[:, i]
    x_next = latents[:, i + 1]

    v = prior_flow(x_t, t)                                   # WITH grad
    _, log_prob, mean_theta, std = sde_step_with_logprob(
        v, sigma, dt, x_t, sigma_max,
        noise_level=cfg.noise_level, prev_sample=x_next)

    kl = None
    if ref_flow is not None:
        with torch.no_grad():
            v_ref = ref_flow(x_t, t)
            _, _, mean_ref, _ = sde_step_with_logprob(
                v_ref, sigma, dt, x_t, sigma_max,
                noise_level=cfg.noise_level, prev_sample=x_next)
        # KL of two Gaussians with equal std reduces to a scaled squared-mean gap
        kl = ((mean_theta - mean_ref) ** 2 / (2 * std ** 2))
        kl = kl.mean(dim=tuple(range(1, kl.ndim)))
    return log_prob, kl


# ======================================================================
# 5. Main Flow-GRPO training loop
# ======================================================================
def train_flow_grpo(prior_flow, reward_model, data_shape, cfg=None,
                    device="cpu", log_every=20, on_log=None):
    cfg = cfg or FlowGRPOConfig()
    prior_flow.to(device)
    reward_model.to(device).eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    ref_flow = None
    if cfg.kl_coef > 0:
        ref_flow = copy.deepcopy(prior_flow).to(device).eval()
        for p in ref_flow.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(prior_flow.parameters(), lr=cfg.lr)
    history = []

    for it in range(cfg.total_iters):
        # --- 1. rollout (no grad) ---
        prior_flow.eval()
        condition, latents, logp_old, ts, sigma_max = sample_trajectory(
            prior_flow, cfg.batch_size, data_shape, cfg, device)

        # --- 2. reward -> group-relative advantage ---
        with torch.no_grad():
            r = reward_model(condition).float().view(-1)          # [B]
        # unconditional => the whole batch is ONE group; standardize over it.
        # WITH conditions/prompts: standardize within each prompt's group instead
        # (this is what PerPromptStatTracker does in the official repo).
        adv = ((r - r.mean()) / (r.std() + cfg.adv_eps)).detach()  # [B]

        # --- 3. PPO-clip update, reuse the rollout inner_epochs times ---
        prior_flow.train()
        last_ratio = 1.0
        for _ in range(cfg.inner_epochs):
            opt.zero_grad()
            for i in range(cfg.T):
                log_prob, kl = step_log_prob(
                    prior_flow, ref_flow, latents, ts, sigma_max, i, cfg)
                ratio = torch.exp(log_prob - logp_old[:, i])
                pg = -torch.min(ratio * adv,
                                torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv)
                loss = pg.mean() / cfg.T                          # accumulate over timesteps
                if kl is not None:
                    loss = loss + cfg.kl_coef * kl.mean() / cfg.T
                loss.backward()
                last_ratio = ratio.mean().item()
            torch.nn.utils.clip_grad_norm_(prior_flow.parameters(), cfg.max_grad_norm)
            opt.step()

        rec = {"iter": it, "reward": r.mean().item(),
               "reward_max": r.max().item(), "ratio": last_ratio}
        history.append(rec)
        if it % log_every == 0 or it == cfg.total_iters - 1:
            print(f"iter {it:4d} | reward {rec['reward']:+.4f} "
                  f"| best {rec['reward_max']:+.4f} | ratio {rec['ratio']:.3f}")
            if on_log is not None:
                on_log(it, prior_flow)
    return history


# ======================================================================
# 6. Self-contained 2D toy to verify the wiring (delete for real use)
# ======================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    DEV = "cpu"
    DATA_SHAPE = (2,)

    # ---- a tiny velocity network: prior_flow(x, t) -> velocity ----
    class TinyFlow(nn.Module):
        def __init__(self, dim=2, h=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim + 1, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, h), nn.SiLU(),
                nn.Linear(h, dim),
            )
        def forward(self, x, t):
            if t.ndim == 1:
                t = t.view(-1, 1)
            return self.net(torch.cat([x, t], dim=-1))

    prior_flow = TinyFlow().to(DEV)

    # ---- pretrain the prior with plain flow matching on a base blob ----
    # base data ~ N(mean=(0,0), std=0.5). This is the "prior" we start from.
    BASE_MEAN = torch.tensor([0.0, 0.0])
    BASE_STD = 0.5
    def sample_base(n):
        return BASE_MEAN + BASE_STD * torch.randn(n, 2)

    opt_pre = torch.optim.Adam(prior_flow.parameters(), lr=2e-3)
    print("== pretraining prior (flow matching) ==")
    for step in range(3000):
        x0 = sample_base(512).to(DEV)               # data  (t=0)
        eps = torch.randn_like(x0)                   # noise (t=1)
        t = torch.rand(x0.shape[0], 1, device=DEV)
        x_t = (1 - t) * x0 + t * eps
        v_target = eps - x0
        v_pred = prior_flow(x_t, t.squeeze(-1))
        loss = ((v_pred - v_target) ** 2).mean()
        opt_pre.zero_grad(); loss.backward(); opt_pre.step()
        if step % 1000 == 0:
            print(f"  pretrain step {step:4d} | fm loss {loss.item():.4f}")

    # ---- reward model: prefer samples near TARGET=(2,2) ----
    TARGET = torch.tensor([2.0, 2.0], device=DEV)
    class RewardModel(nn.Module):
        def forward(self, condition):
            return -((condition - TARGET) ** 2).sum(dim=-1)   # higher = closer
    reward_model = RewardModel().to(DEV)

    # ---- snapshot the prior's samples BEFORE fine-tuning ----
    @torch.no_grad()
    def draw(model, n=2000):
        cfg = FlowGRPOConfig(); cfg.batch_size = n
        cond, *_ = sample_trajectory(model, n, DATA_SHAPE, cfg, DEV)
        return cond.cpu()
    before = draw(prior_flow)
    print(f"\nprior mean before: {before.mean(0).tolist()}  "
          f"reward before: {reward_model(before.to(DEV)).mean().item():+.4f}")

    # ---- Flow-GRPO fine-tuning ----
    print("\n== flow-GRPO fine-tuning from reward ==")
    cfg = FlowGRPOConfig()
    cfg.total_iters = 300
    cfg.batch_size = 256
    cfg.lr = 3e-4
    cfg.kl_coef = 0.0            # try 0.05 to stay closer to the prior
    hist = train_flow_grpo(prior_flow, reward_model, DATA_SHAPE, cfg, DEV, log_every=50)

    after = draw(prior_flow)
    print(f"\nprior mean after:  {after.mean(0).tolist()}  "
          f"reward after:  {reward_model(after.to(DEV)).mean().item():+.4f}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
        ax[0].scatter(before[:, 0], before[:, 1], s=5, alpha=.3, label="before (prior)")
        ax[0].scatter(after[:, 0], after[:, 1], s=5, alpha=.3, label="after (GRPO)")
        ax[0].scatter([TARGET[0].item()], [TARGET[1].item()], c="red", marker="*",
                      s=250, edgecolor="k", label="reward target", zorder=5)
        ax[0].set_title("generated 'condition' distribution"); ax[0].legend(); ax[0].axis("equal")
        ax[0].set_xlim(-3, 4); ax[0].set_ylim(-3, 4)
        it = [h["iter"] for h in hist]
        ax[1].plot(it, [h["reward"] for h in hist], label="mean reward")
        ax[1].plot(it, [h["reward_max"] for h in hist], alpha=.5, label="best in batch")
        ax[1].set_xlabel("GRPO iter"); ax[1].set_ylabel("reward"); ax[1].set_title("reward curve")
        ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig("/home/claude/flow_grpo_toy.png", dpi=110)
        print("\nsaved figure -> flow_grpo_toy.png")
    except Exception as e:
        print("plot skipped:", e)
