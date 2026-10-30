"""The motion latent the model generates in, and its map to and from 70-d LivePortrait motion.

MotionLatent is a PCA over everything LivePortrait renders (sang.model.target.wide, 69 numbers per frame), taken
RELATIVE to a reference frame (the photo), so identity stays out of the code. Its dimensions come out ordered by
how much variance they keep, so every prefix of k codes decodes. The generator uses 42: the smallest k whose
render ceiling (eval_hdtf.py --gt-latent) matches the full motion's (docs/research_log.md 3.26).

Codes are standardised per dimension so the flow model sees unit-scale targets; a dropped dimension is zero in
the raw code space, as in training.
"""
import torch
from torch import nn

from sang.model.target import WIDE_DIM, from_wide, wide


class MotionLatent(nn.Module):
    def __init__(self, K: int = 64, dim: int = WIDE_DIM):
        super().__init__()
        self.K, self.dim = K, dim
        for name, n, v in (("mu", dim, 0.0), ("sd", dim, 1.0), ("c_mu", K, 0.0), ("c_sd", K, 1.0)):
            self.register_buffer(name, torch.full((n,), v))
        self.enc, self.dec = nn.Linear(dim, K, bias=False), nn.Linear(K, dim, bias=False)

    def code(self, d: torch.Tensor) -> torch.Tensor:
        """Raw codes [..., K] of motion deltas [..., dim]."""
        return self.enc((d.float() - self.mu) / self.sd)

    def recon(self, c: torch.Tensor) -> torch.Tensor:
        """Raw codes [..., k <= K] -> deltas [..., dim]; the missing dimensions are zero."""
        if c.shape[-1] < self.K:
            c = torch.cat([c, c.new_zeros(*c.shape[:-1], self.K - c.shape[-1])], -1)
        return self.dec(c) * self.sd + self.mu

    def encode(self, d: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Deltas -> the first k standardised codes."""
        return ((self.code(d) - self.c_mu) / self.c_sd)[..., :k]

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        """The first k standardised codes -> deltas."""
        k = u.shape[-1]
        return self.recon(u.float() * self.c_sd[:k] + self.c_mu[:k])

    @torch.no_grad()
    def set_code_stats(self, d: torch.Tensor, chunk: int = 65536) -> "MotionLatent":
        c = torch.cat([self.code(d[i:i + chunk].to(self.mu.device)) for i in range(0, len(d), chunk)])
        self.c_mu.copy_(c.mean(0))
        self.c_sd.copy_(c.std(0).clamp_min(1e-6))
        return self

    def pack(self) -> dict:
        return {"K": self.K, "kind": "pca", "dim": self.dim, "state": self.state_dict()}

    @classmethod
    def unpack(cls, d: dict) -> "MotionLatent":
        lat = cls(d["K"], d["dim"])
        lat.load_state_dict(d["state"])
        return lat.eval().requires_grad_(False)

    @classmethod
    def of(cls, ck: dict) -> "MotionLatent":
        """The latent a training checkpoint generates in."""
        return cls.unpack(ck["latent"])

    def from_motion(self, m: torch.Tensor, m_ref: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Motion [n,70] relative to a reference frame [1,70] -> the first k standardised codes [n, k]."""
        return self.encode(wide(m) - wide(m_ref), k)

    def to_motion(self, u: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """The first k standardised codes [n, k] + the reference frame's motion [1,70] -> motion [n,70]."""
        return from_wide(wide(base) + self.decode(u), base)


def fit_pca(d: torch.Tensor, K: int | None = None) -> MotionLatent:
    """Principal components of the standardised deltas [N, dim], strongest first (at most min(N, dim))."""
    mu, sd = d.mean(0), d.std(0).clamp_min(1e-6)
    _, _, v = torch.linalg.svd(((d - mu) / sd).double(), full_matrices=False)
    lat = MotionLatent(min(K or d.shape[1], len(v)), dim=d.shape[1])
    lat.mu.copy_(mu)
    lat.sd.copy_(sd)
    w = v[: lat.K].float()
    lat.enc.weight.data.copy_(w)
    lat.dec.weight.data.copy_(w.T)
    return lat.set_code_stats(d).eval().requires_grad_(False)

