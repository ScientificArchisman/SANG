"""Proper scores that fine-tune SANG's deployed sampler (reports/SANG RL for video realism.md idea 1; reports/SANG style
aware scoring rules.md rank 1). No critic and no reward model: each loss is minimised only by the real motion law.

The person critic, its RL, its guidance and its test-time fit, the adversarial set critic, the person-set distance
(idea 2) and the test-time LoRA fit of the generator all failed their tests (research log §3.29-3.33, 3.35, 3.37);
their code was deleted after commit 407a1bf.
"""
import torch

from sang.model.flow import MotionFlowTransformer, flow_times, sample


@torch.no_grad()
def gen_fakes(gen: MotionFlowTransformer, batch: dict, steps: int = 6, g: float = 2.0,
              generator: torch.Generator | None = None) -> torch.Tensor:
    """SANG's samples for a batch's own audio, photo, prefix and footage, with the deployed one-pass sampler."""
    demo = gen.encode_demo(batch)
    keep = batch["demo_keep"].bool() if demo is not None else None
    return sample(gen, batch["audio"], batch["ref"], batch["target"].shape[1], prefix=batch["prefix"], steps=steps,
                  cfg_audio=g, demo=demo, demo_keep=keep, generator=generator)


def rollout(policy: MotionFlowTransformer, ref_model: MotionFlowTransformer, batch: dict, steps: int = 6,
            g: float = 2.0, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The deployed one-pass Euler sampler with gradients (DRaFT-style) -> clean windows [B, L, k] and the path KL
    1/2 sum_i dt_i ||v_theta - v_ref||^2 [B] along the policy's own path, summed over frames and codes."""
    x0 = batch["target"]
    B, L, dev = x0.shape[0], x0.shape[1], x0.device
    kw = {"prefix": batch["prefix"], "prefix_keep": torch.ones(B, dtype=torch.bool, device=dev),
          "beta": torch.full((B,), 1.0 / g, device=dev), "demo_keep": batch.get("demo_keep")}
    d_pol = policy.encode_demo(batch)
    with torch.no_grad():
        d_ref = ref_model.encode_demo(batch)
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
    return x, path_kl


def energy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy distance per group between samples x [G, n, D] (n >= 2) and real y [G, k, D]:
    2 E||x - y|| - E||x - x'|| - E||y - y'|| (the last term only for k > 1; it is constant for the generator).
    Unbiased, so 0 in expectation when both sides come from one law, and its expectation over the data is minimised
    only by the real law (strictly proper; Szekely and Rizzo). k = 1 is twice the energy score of x for y."""
    n, k = x.shape[1], y.shape[1]
    assert n >= 2, "the energy distance needs two or more samples per group"
    i, j = torch.triu_indices(n, n, 1, device=x.device)  # pairs i < j: the norm's gradient is undefined at 0
    out = 2 * (x[:, :, None] - y[:, None]).norm(dim=-1).flatten(1).mean(1) - (x[:, i] - x[:, j]).norm(dim=-1).mean(1)
    if k > 1:
        a, b = torch.triu_indices(k, k, 1, device=y.device)
        out = out - (y[:, a] - y[:, b]).norm(dim=-1).mean(1)
    return out


def haar(x: torch.Tensor) -> list[torch.Tensor]:
    """Orthonormal Haar levels of x [..., T, C] over time (T a power of 2): the details at 1, 2, 4, ... frames, then
    the window mean (T = 64: 7 levels of 32, 16, 8, 4, 2, 1 and 1 coefficients)."""
    assert x.shape[-2] & (x.shape[-2] - 1) == 0, "the Haar levels need a power-of-2 window"
    out = []
    while x.shape[-2] > 1:
        a, b = x[..., 0::2, :], x[..., 1::2, :]
        out.append((a - b) / 2 ** 0.5)
        x = (a + b) / 2 ** 0.5
    return out + [x]


def block_scale(y: torch.Tensor, start: torch.Tensor, decode) -> tuple[torch.Tensor, torch.Tensor]:
    """The constants mbes divides by, from pairs of different real windows y [N, T, K] (codes, N >= 2) measured from
    their start [N, 1, K]: E||Y - Y'|| of the whole window, and of every (Haar level, decoded channel) block [levels, C]."""
    with torch.autocast(y.device.type, enabled=False):
        ys, w = (y - start).float(), decode(y.float()) - decode(start.float())
        c0 = (ys - ys.roll(1, 0)).flatten(1).norm(dim=-1).mean()
        return c0, torch.stack([(h - h.roll(1, 0)).norm(dim=-2).mean(0) for h in haar(w)])


def mbes(x: torch.Tensor, y: torch.Tensor, start: torch.Tensor, decode, c0: torch.Tensor,
         cb: torch.Tensor) -> torch.Tensor:
    """Multiscale-block energy score per input (reports/SANG style aware scoring rules.md, rank 1): samples x
    [B, m, T, K] (m >= 2) and the real window y [B, T, K], codes measured from start [B, 1, K], the last prefix frame.
      ES(whole window) + c0 x mean over blocks b of ES(block b) / c_b
    with a block one decoded motion channel at one Haar level (69 x 7 at T = 64) and c_b its real spread
    (block_scale), so every block counts equally. Each term is proper and the first strictly, so the sum is strictly
    proper; the blocks see errors confined to a few channels or one time scale (head range, nod rhythm, drift), which
    the whole-window score dilutes. c0 keeps it in the units of the whole-window score, so kl keeps its meaning."""
    m = x.shape[1]
    i, j = torch.triu_indices(m, m, 1, device=x.device)
    es = lambda u, v, dim: (u - v[:, None]).norm(dim=dim).mean(1) - 0.5 * (u[:, i] - u[:, j]).norm(dim=dim).mean(1)
    with torch.autocast(x.device.type, enabled=False):
        x, y, start = x.float(), y.float(), start.float()
        full = es((x - start[:, None]).flatten(2), (y - start).flatten(1), -1)
        p = decode(start)
        hx, hy = haar(decode(x) - p[:, None]), haar(decode(y) - p)
        blocks = torch.stack([es(a, b, -2) for a, b in zip(hx, hy)], 1) / cb  # [B, levels, C]
        return full + c0 * blocks.flatten(1).mean(1)


def set_loss(policy: MotionFlowTransformer, ref_model: MotionFlowTransformer, batch: dict, m: int = 8,
             steps: int = 6, g: float = 2.0, kl: float = 1.0, w_set: float = 1.0,
             generator: torch.Generator | None = None, blocks: tuple | None = None) -> tuple[torch.Tensor, dict]:
    """Proper multi-sample fine-tuning of the sampler (idea 1): every window of the batch is rolled out m times.
      ed     energy distance between the m samples and the real window, on the standardised codes minus the window's
             last prefix frame
      mbes   with blocks = (decode, c0, cb): the multiscale-block energy score in place of ed
    loss = w_set score + kl path KL. The score has no reward model to exploit: its minimiser is the data."""
    B = batch["target"].shape[0]
    rep = {k: v.repeat_interleave(m, 0) for k, v in batch.items()}
    x, path_kl = rollout(policy, ref_model, rep, steps, g, generator)
    loss, parts = kl * path_kl.mean(), {"path_kl": path_kl.mean().detach()}
    x = x.view(B, m, *x.shape[1:])
    start = batch["prefix"][:, -1:]
    if blocks is not None:
        e = mbes(x, batch["target"], start, *blocks).mean()
        parts["mbes"] = e.detach()
    else:  # each window from where it starts: its photo and prefix offset cancel
        s = start.float()
        e = energy((x.float() - s[:, None]).flatten(2), (batch["target"].float() - s).flatten(1)[:, None]).mean()
        parts["ed"] = e.detach()
    return loss + w_set * e, parts
