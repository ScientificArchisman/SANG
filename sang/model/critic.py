"""The person critic: a learned reward for SANG's motion (plan, design and gates: docs/person_critic_plan.md).

One trunk, two logits, nothing hand-picked:
  real  this person's real motion (1) vs SANG's motion for the same audio, photo, prefix and footage (0). At the
        optimum its logit is log p_real / p_SANG: the RL reward, the guidance signal (Discriminator Guidance, Kim et
        al., ICML 2023, arXiv 2211.17091) and what the test-time fit adapts to one person.
  sync  real motion with its own audio (1) vs the same motion with the audio shifted by 3-15 frames (0): lip sync
        learned in motion space. SyncNet stays an evaluation metric only, so it cannot be gamed.
The trunk is a MotionFlowTransformer initialised from the generator, so it reads audio, photo, prefix and footage
the way SANG does. It also takes the flow time t: t > 0 scores noisy states (guidance), t = 0 clean windows (reward).
"""
import contextlib
import copy
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sang.model.flow import MotionFlowTransformer, build, flow_times, sample

TPF = 2  # WavLM ticks per video frame


class Critic(nn.Module):
    def __init__(self, trunk: MotionFlowTransformer):
        super().__init__()
        self.trunk = trunk
        self.head = nn.Linear(trunk.in_proj.out_features, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode_demo(self, batch: dict):
        return self.trunk.encode_demo(batch)

    def forward(self, x_t, t, audio, ref, prefix=None, prefix_keep=None, demo=None, demo_keep=None) -> torch.Tensor:
        """Same inputs as the generator -> [B, 2] logits (real, sync), mean-pooled over the target frames."""
        h = self.trunk(x_t, t, audio, ref, prefix=prefix, prefix_keep=prefix_keep, demo=demo, demo_keep=demo_keep,
                       features=True)
        return self.head(h.float().mean(1))


def critic_from_generator(ck: dict) -> Critic:
    """A fresh critic: the generator checkpoint's EMA weights as trunk, a zero head (every logit 0 at the start)."""
    trunk = build(ck["cfg"])
    trunk.load_state_dict(ck["ema"])
    return Critic(trunk)


def load_critic(path: str, device: str = "cuda") -> Critic:
    """A scripts/train_critic.py checkpoint, frozen, in eval mode."""
    ck = torch.load(path, map_location=device, weights_only=False)
    c = Critic(build(ck["gen_cfg"]))
    c.load_state_dict(ck["critic"])
    return c.to(device).eval().requires_grad_(False)


def noised(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) x + t eps, the generator's own interpolation."""
    tb = t.view(-1, 1, 1)
    return (1 - tb) * x + tb * torch.randn_like(x)


def auc(pos: torch.Tensor, neg: torch.Tensor) -> float:
    """P(score of a positive > score of a negative) over all pairs; ties count half."""
    d = pos.float().reshape(-1, 1) - neg.float().reshape(1, -1)
    return float((d > 0).float().mean() + 0.5 * (d == 0).float().mean())


@torch.no_grad()
def gen_fakes(gen: MotionFlowTransformer, batch: dict, steps: int = 6, g: float = 2.0,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """SANG's samples for a batch's own audio, photo, prefix and footage, with the deployed one-pass sampler."""
    demo = gen.encode_demo(batch)
    keep = batch["demo_keep"].bool() if demo is not None else None
    return sample(gen, batch["audio"], batch["ref"], batch["target"].shape[1], prefix=batch["prefix"], steps=steps,
                  cfg_audio=g, demo=demo, demo_keep=keep, generator=generator)


def critic_loss(critic: Critic, batch: dict, fake: torch.Tensor, p_clean: float = 0.25, p_demo: float = 0.2,
                shift: tuple[int, int] = (3, 15), sync: bool = True, gp: float = 0.0) -> tuple[torch.Tensor, dict]:
    """BCE of both logits on one batch of real windows (codes) and SANG's windows `fake` for the same inputs.

    Every window gets one flow time: t = 0 with p_clean (the reward), else t ~ U(0, 1) (guidance); real and fake
    share it. The sync pass reuses the real pass's noisy x with the audio shifted, so only the audio differs.
    The critic's footage is dropped with p_demo, so it also scores without footage.
    gp > 0 adds R1, gp/2 E_real ||grad_x l||^2 for both logits (Mescheder et al. 2018; AMP's penalty, arXiv
    2104.02180): a smooth critic that a small change of the motion cannot flip, so the policy cannot hack it."""
    x0 = batch["target"]
    B, dev = x0.shape[0], x0.device
    t = torch.rand(B, device=dev)
    t = torch.where(torch.rand(B, device=dev) < p_clean, torch.zeros_like(t), t)
    kw = {"prefix": batch["prefix"]}
    demo = critic.encode_demo(batch)
    if demo is not None:
        kw.update(demo=demo, demo_keep=batch["demo_keep"].bool() & (torch.rand(B, device=dev) >= p_demo))
    xr = noised(x0, t).requires_grad_(gp > 0)
    with sdpa_kernel(SDPBackend.MATH) if gp > 0 else contextlib.nullcontext():  # flash attention has no 2nd backward
        lr = critic(xr, t, batch["audio"], batch["ref"], **kw).float()
    lf = critic(noised(fake, t), t, batch["audio"], batch["ref"], **kw).float()
    one, zero = torch.ones(B, device=dev), torch.zeros(B, device=dev)
    bce = F.binary_cross_entropy_with_logits
    out = {"real": bce(lr[:, 0], one) + bce(lf[:, 0], zero)}
    if sync:
        k = random.choice((-1, 1)) * random.randint(*shift)
        ln = critic(xr, t, batch["audio"].roll(TPF * k, 1), batch["ref"], **kw).float()
        out["sync"] = bce(lr[:, 1], one) + bce(ln[:, 1], zero)
    if gp > 0:
        sign = torch.randint(0, 2, lr.shape, device=dev) * 2 - 1  # random signs: one backward, unbiased for both heads
        grad, = torch.autograd.grad((lr * sign).sum(), xr, create_graph=True)
        out["gp"] = 0.5 * gp * grad.float().pow(2).sum((1, 2)).mean()
    return sum(out.values()), {k: v.detach() for k, v in out.items()}


def make_guide(critic: Critic, demo: torch.Tensor | None = None, w_real: float = 1.0, w_sync: float = 0.0,
               cap: float = 5.0):
    """Critic guidance for sang.model.flow.sample: v' = v - min(t / (1 - t), cap) grad_x (w_real l_real + w_sync l_sync).

    Discriminator Guidance adds grad log(p_real / p_SANG), the logit's gradient, to the score. For
    x_t = (1 - t) x0 + t eps the score is -eps_hat / t and v = (eps_hat - x_t) / (1 - t), so a score correction s
    is the velocity correction -t / (1 - t) s. The factor diverges at t = 1, where the critic sees pure noise, so
    it is capped (with 6 steps the cap only touches the first step)."""
    w = torch.tensor([w_real, w_sync])

    def guide(x, t, audio, ref, prefix=None, prefix_keep=None):
        with torch.enable_grad():
            x = x.detach().float().requires_grad_(True)
            r = critic(x, t, audio, ref, prefix=prefix, prefix_keep=prefix_keep, demo=demo).float() @ w.to(x.device)
            (grad,) = torch.autograd.grad(r.sum(), x)
        f = (t.float() / (1 - t.float()).clamp_min(1e-6)).clamp(max=cap).view(-1, 1, 1)
        return -f * grad

    return guide


class LoRA(nn.Module):
    """base(x) + b(a(x)) with b = 0 at the start, so a fresh adapter changes nothing."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=dev) / base.in_features ** 0.5)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b)


def add_lora(critic: Critic, rank: int = 4, blocks: int = 1) -> list[nn.Parameter]:
    """Freeze the critic; add LoRA to its head and to the SwiGLU and adaLN layers of its last `blocks` blocks.
    Returns the adapter parameters, the only ones the test-time fit trains."""
    critic.requires_grad_(False)
    sites = [(critic, "head")]
    for blk in critic.trunk.blocks[len(critic.trunk.blocks) - blocks:]:
        sites += [(blk.ff, "w1"), (blk.ff, "w2"), (blk.ff, "w3"), (blk.ada, "1")]
    params = []
    for mod, name in sites:
        lora = LoRA(getattr(mod, name), rank)
        setattr(mod, name, lora)
        params += [lora.a, lora.b]
    return params


def fit_person(critic: Critic, gen: MotionFlowTransformer, u: torch.Tensor, audio: torch.Tensor, ref: torch.Tensor,
               steps: int = 100, rank: int = 4, blocks: int = 1, lr: float = 1e-3, batch: int = 32, pool: int = 128,
               P: int = 10, L: int = 64, gen_steps: int = 6, g: float = 2.0, p_clean: float = 0.25,
               seed: int = 0) -> Critic:
    """Test-time fit of the realism logit to one person (LoRA only; stage 3 of docs/person_critic_plan.md).

    Real = `pool` random windows of the person's footage; fake = SANG's windows for the same audio, photo and
    prefix. Both the critic and SANG see the footage with that window masked out, so neither can win by recognising
    frames it was shown (a critic that could would make guidance replay the footage).
    u [N, k] footage codes relative to the photo, audio [1, >= 2N, 1024] its WavLM, ref [1, REF_DIM].
    Returns a fitted copy; the critic itself when steps = 0 (the no-fit ablation) or the footage is shorter than
    one window."""
    N, W, dev = u.shape[0], P + L, u.device
    if steps <= 0 or N < W:
        return critic
    rng = torch.Generator(device=dev).manual_seed(seed)
    A = audio[0, : TPF * N].float()
    A = F.pad(A, (0, 0, 0, TPF * N - A.shape[0]))
    s = torch.randint(0, N - W + 1, (pool,), generator=rng, device=dev)
    idx = s[:, None] + torch.arange(W, device=dev)
    win = u[idx]
    mask = torch.ones(pool, N, dtype=torch.bool, device=dev)
    mask[torch.arange(pool, device=dev)[:, None], idx] = False
    b = {"target": win[:, P:], "prefix": win[:, :P], "audio": A[(TPF * s)[:, None] + torch.arange(TPF * W, device=dev)],
         "ref": ref.expand(pool, -1), "demo_motion": u[None].expand(pool, -1, -1),
         "demo_audio": A[None].expand(pool, -1, -1), "demo_mask": mask,
         "demo_keep": torch.ones(pool, dtype=torch.bool, device=dev)}
    sub = lambda j: {k: v[j] for k, v in b.items()}
    fake = torch.cat([gen_fakes(gen, sub(slice(i, i + batch)), gen_steps, g, rng) for i in range(0, pool, batch)])
    fitted = copy.deepcopy(critic)
    opt = torch.optim.Adam(add_lora(fitted, rank, blocks), lr=lr)
    fitted.train()
    with torch.enable_grad():
        for _ in range(steps):
            j = torch.randint(0, pool, (batch,), generator=rng, device=dev)
            loss, _ = critic_loss(fitted, sub(j), fake[j], p_clean=p_clean, p_demo=0.0, sync=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return fitted.eval().requires_grad_(False)


def rl_loss(policy: MotionFlowTransformer, ref_model: MotionFlowTransformer, critic: Critic, batch: dict,
            steps: int = 6, g: float = 2.0, kl: float = 1.0, w_real: float = 1.0, w_sync: float = 1.0,
            shift: tuple[int, int] = (3, 15), generator: torch.Generator | None = None) -> tuple[torch.Tensor, dict]:
    """KL-regularised reward maximisation, backpropagated through the whole Euler sampler (DRaFT-style).

    reward = w_real l_real + w_sync (l_sync - l_sync with the audio shifted) of the clean sample (critic frozen). The
    sync term is a contrast on the same sample, one shift each way by shift frames, as SyncNet's confidence is: motion the
    sync head likes under any audio earns nothing (RL v1 raised l_sync above real while LSE-C fell, research log §3.30). path KL = 1/2 sum_i dt_i
    ||v_theta - v_ref||^2 along the policy's own path, summed over frames and codes: Girsanov's KL for unit
    diffusion, the usual stand-in for a deterministic sampler. With the critic refit to ref_model each round and
    kl = 1, the optimum of E[log p_real / p_ref] - KL(pi || p_ref) is p_real itself."""
    x0 = batch["target"]
    B, L, dev = x0.shape[0], x0.shape[1], x0.device
    kw = {"prefix": batch["prefix"], "prefix_keep": torch.ones(B, dtype=torch.bool, device=dev),
          "beta": torch.full((B,), 1.0 / g, device=dev), "demo_keep": batch.get("demo_keep")}
    d_pol = policy.encode_demo(batch)
    with torch.no_grad():
        d_ref, d_cri = ref_model.encode_demo(batch), critic.encode_demo(batch)
    x = torch.randn(B, L, policy.t_dim, device=dev, generator=generator)
    ts = flow_times(steps)
    path_kl = torch.zeros(B, device=dev)
    for i in range(steps):
        t = torch.full((B,), ts[i], device=dev)
        v = policy(x, t, batch["audio"], batch["ref"], demo=d_pol, **kw)
        with torch.no_grad():
            v0 = ref_model(x, t, batch["audio"], batch["ref"], demo=d_ref, **kw)
        dt = ts[i] - ts[i + 1]
        path_kl = path_kl + 0.5 * dt * (v.float() - v0.float()).pow(2).sum((1, 2))
        x = x - dt * v
    score = lambda audio: critic(x, torch.zeros(B, device=dev), audio, batch["ref"], prefix=batch["prefix"],
                                 demo=d_cri, demo_keep=batch.get("demo_keep")).float()
    r = score(batch["audio"])
    ks = (-random.randint(*shift), random.randint(*shift))
    r_shift = torch.stack([score(batch["audio"].roll(TPF * k, 1))[:, 1] for k in ks]).mean(0)
    reward = w_real * r[:, 0] + w_sync * (r[:, 1] - r_shift)
    loss = (kl * path_kl - reward).mean()
    return loss, {"reward": reward.mean().detach(), "l_real": r[:, 0].mean().detach(),
                  "l_sync": r[:, 1].mean().detach(), "l_sync_shift": r_shift.mean().detach(),
                  "path_kl": path_kl.mean().detach()}


@torch.no_grad()
def score_clip(critic: Critic, z: torch.Tensor, audio: torch.Tensor, ref: torch.Tensor, start: torch.Tensor,
               demo: torch.Tensor | None = None, window: int = 64) -> dict:
    """Mean (real, sync) logits at t = 0 over the full windows of a clip z [B, n, k], windowed as
    sang.model.flow.generate makes them: the first continues from `start` [B, P, k] (audio carries P frames of lead),
    later ones from the clip's own last P frames. {} if the clip is shorter than one window."""
    B, n, _ = z.shape
    P = start.shape[1]
    out = []
    for pos in range(0, n - window + 1, window):
        prefix = start if pos == 0 else z[:, pos - P:pos]
        a = audio[:, pos * TPF:(P + pos + window) * TPF]
        out.append(critic(z[:, pos:pos + window], torch.zeros(B, device=z.device), a, ref, prefix=prefix,
                          demo=demo).float())
    if not out:
        return {}
    m = torch.stack(out).mean((0, 1))
    return {"critic_real": float(m[0]), "critic_sync": float(m[1])}
