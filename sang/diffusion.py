"""Continuous-latent diffusion/flow head for SANG v4.

The transformer produces a per-position conditioning hidden state `h`; a small MLP
`eps_theta(z_t, t, h)` predicts the velocity of a noised continuous Wan2.1 VAE latent.
Training = flow-matching MSE (LeapTalk/EARTalking); generation = few-step ODE from the
bridge prior (ref latent) to the clean latent.

Flow matching (rectified flow): z_t = (1-t)*z0 + t*noise, target velocity v = noise - z0.
t=0 is clean data, t=1 is noise.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def cosine_t(B: int, device) -> torch.Tensor:
    """Sample t ~ U(0,1) per example; cosine weighting optional later. Returns [B]."""
    return torch.rand(B, device=device)


def add_noise(z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
    """z0 [B,...], t [B] in [0,1] -> (z_t, target_velocity). t broadcast over trailing dims."""
    noise = torch.randn_like(z0) if noise is None else noise
    tb = t.reshape(-1, *([1] * (z0.ndim - 1)))
    z_t = (1 - tb) * z0 + tb * noise
    return z_t, noise - z0  # velocity target


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.reshape(-1, 1) * freqs.reshape(1, -1)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class DiffusionHead(nn.Module):
    """eps/v-prediction MLP conditioned on transformer hidden state h and timestep t.

    Predicts velocity v for a noised latent patch z_t. Operates per spatial position:
    z_t [B, L, z_ch], h [B, L, dim] -> v [B, L, z_ch]."""

    def __init__(self, z_ch: int, dim: int, hidden: int | None = None, depth: int = 3):
        super().__init__()
        hidden = hidden or dim
        self.z_ch = z_ch
        self.time_emb = SinusoidalTimeEmb(dim)
        self.in_proj = nn.Linear(z_ch, hidden)
        self.cond_proj = nn.Linear(dim, hidden)
        self.time_proj = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)
        ])
        self.out = nn.Linear(hidden, z_ch)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """z_t [B,L,z_ch], t [B], h [B,L,dim] -> predicted velocity [B,L,z_ch]."""
        temb = self.time_proj(self.time_emb(t)).unsqueeze(1)      # [B,1,hidden]
        x = self.in_proj(z_t) + self.cond_proj(h) + temb          # [B,L,hidden]
        for blk in self.blocks:
            x = x + blk(x)
        return self.out(x)


def flow_loss(head: DiffusionHead, h: torch.Tensor, z0: torch.Tensor,
              t: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """h [B,L,dim] conditioning, z0 [B,L,z_ch] clean latent -> (loss, parts)."""
    B = z0.shape[0]
    t = cosine_t(B, z0.device) if t is None else t
    z_t, v_target = add_noise(z0, t)
    v_pred = head(z_t, t, h)
    loss = F.mse_loss(v_pred, v_target)
    # per-t-bucket diagnostic: low t (near data) vs high t (near noise)
    with torch.no_grad():
        err = (v_pred - v_target).pow(2).mean(dim=tuple(range(1, v_pred.ndim)))
        lo = err[t < 0.5].mean() if (t < 0.5).any() else torch.tensor(0.0)
        hi = err[t >= 0.5].mean() if (t >= 0.5).any() else torch.tensor(0.0)
    return loss, {"diff": loss.detach(), "diff_lo": lo, "diff_hi": hi}


@torch.no_grad()
def flow_sample(head: DiffusionHead, h: torch.Tensor, z_init: torch.Tensor,
                steps: int = 8, t_start: float = 1.0) -> torch.Tensor:
    """Integrate the learned flow backward from t_start to 0 (denoise).

    Convention: t=0 clean data, t=1 noise, velocity v = noise - z0 points data->noise, so to denoise
    we step z <- z - dt*v while walking t from t_start down to 0. With the bridge, z_init is the ref
    latent and t_start<1 starts partway (less to denoise); pure generation uses z_init=noise, t_start=1.

    h [B,L,dim], z_init [B,L,z_ch] -> clean latent estimate [B,L,z_ch]."""
    B = z_init.shape[0]
    z = z_init
    dt = t_start / steps
    for i in range(steps):
        t_cur = torch.full((B,), t_start - i * dt, device=z_init.device)
        z = z - dt * head(z, t_cur, h)
    return z
