"""Rectified-flow head over continuous VAE latents.

Convention: t=0 clean data, t=1 noise; z_t = (1-t) z0 + t eps; the head predicts v = eps - z0.
Sampling integrates z <- z - dt v from t=1 down to 0.

The head is the MAR-style per-token diffusion MLP (Li et al. 2024, "Autoregressive Image
Generation without Vector Quantization"; NOVA uses the same): residual MLP blocks whose LayerNorm
is modulated (adaLN) by the timestep embedding plus the transformer's per-position hidden state.
The previous head injected t once at the input and had no normalisation.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


def add_noise(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
    """z0 [B,...], t [B] -> (z_t, velocity target). t broadcasts over trailing dims."""
    noise = torch.randn_like(z0) if noise is None else noise
    tb = t.reshape(-1, *([1] * (z0.ndim - 1)))
    return (1 - tb) * z0 + tb * noise, noise - z0


def x0_from_v(z_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """One-step clean-latent estimate: z0 = z_t - t v. Exact for a perfect v; used for
    pixel-space losses on the prediction (LatentSync applies SyncNet to exactly this)."""
    return z_t - t.reshape(-1, *([1] * (z_t.ndim - 1))) * v


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.reshape(-1, 1).float() * freqs.reshape(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class AdaLNBlock(nn.Module):
    """LayerNorm -> adaLN(shift, scale) -> MLP -> gated residual. Conditioning y is per position."""

    def __init__(self, hidden: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 3 * hidden))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)  # blocks start as identity

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.ada(y).chunk(3, dim=-1)
        return x + gate * self.mlp(_modulate(self.norm(x), shift, scale))


class DiffusionHead(nn.Module):
    """v-prediction MLP: z_t [B,L,z_ch], t [B], h [B,L,dim] -> v [B,L,z_ch]."""

    def __init__(self, z_ch: int, dim: int, hidden: int = 1024, depth: int = 6):
        super().__init__()
        self.z_ch = z_ch
        self.time_emb = SinusoidalTimeEmb(hidden)
        self.time_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.cond_proj = nn.Linear(dim, hidden)
        self.in_proj = nn.Linear(z_ch, hidden)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.out_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.out = nn.Linear(hidden, z_ch)
        nn.init.zeros_(self.out_ada[-1].weight)
        nn.init.zeros_(self.out_ada[-1].bias)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # v starts at 0

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y = self.time_mlp(self.time_emb(t)).unsqueeze(1) + self.cond_proj(h)  # [B,L,hidden]
        x = self.in_proj(z_t)
        for blk in self.blocks:
            x = blk(x, y)
        shift, scale = self.out_ada(y).chunk(2, dim=-1)
        return self.out(_modulate(self.out_norm(x), shift, scale))


def flow_loss(head: DiffusionHead, h: torch.Tensor, z0: torch.Tensor):
    """h [B,L,dim], z0 [B,L,z_ch] -> (loss, parts, z0_hat). z0_hat keeps the graph so
    pixel-space losses can be applied to the prediction."""
    B = z0.shape[0]
    t = torch.rand(B, device=z0.device)
    z_t, v_target = add_noise(z0, t)
    v_pred = head(z_t, t, h)
    loss = F.mse_loss(v_pred.float(), v_target.float())
    with torch.no_grad():
        err = (v_pred.float() - v_target.float()).pow(2).mean(dim=(1, 2))
        zero = torch.zeros((), device=z0.device)
        parts = {"diff": loss.detach(),
                 "diff_lo": err[t < 0.5].mean() if (t < 0.5).any() else zero,
                 "diff_hi": err[t >= 0.5].mean() if (t >= 0.5).any() else zero}
    return loss, parts, x0_from_v(z_t, t, v_pred)


@torch.no_grad()
def flow_sample(head: DiffusionHead, h: torch.Tensor, shape, steps: int = 12,
                anchor: torch.Tensor | None = None, t_start: float = 1.0,
                h_uncond: torch.Tensor | None = None, cfg: float = 1.0) -> torch.Tensor:
    """Euler-integrate the flow from t_start to 0.

    Starts from pure noise (t_start=1). With an `anchor` latent and t_start<1 it starts from
    (1-t_start) anchor + t_start noise -- the training-time z_t of the anchor, i.e. SDEdit-style
    identity anchoring that is consistent with the objective. (The old sampler started from the
    raw anchor at t=1, which the head never sees in training.)
    cfg>1 with h_uncond applies classifier-free guidance on the velocity."""
    B = h.shape[0]
    noise = torch.randn(B, *shape, device=h.device, dtype=h.dtype)
    z = noise if anchor is None or t_start >= 1.0 else (1 - t_start) * anchor + t_start * noise
    dt = t_start / steps
    for i in range(steps):
        t = torch.full((B,), t_start - i * dt, device=h.device)
        v = head(z, t, h)
        if h_uncond is not None and cfg != 1.0:
            v_u = head(z, t, h_uncond)
            v = v_u + cfg * (v - v_u)
        z = z - dt * v
    return z
