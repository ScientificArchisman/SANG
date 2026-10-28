"""Learned motion spaces, and the map between a model's output space and 70-d LivePortrait motion.

MotionLatent is an autoencoder over everything LivePortrait renders (sang.motion.wide, 69 numbers per frame),
taken RELATIVE to a reference frame (the photo), so identity stays out of the code. It is trained with nested
dropout (Rippel et al. 2014; OAT, arXiv 2602.04215): every step keeps a random prefix of the K code dimensions
and zeroes the rest, so every prefix decodes and the dimensions come out ordered by how much they matter.
kind='pca' is the linear special case, fitted in closed form. How many codes the generator uses is read off a
measured curve (scripts/fit_latent.py in motion space, the eval_hdtf.py ceiling row with --gt-latent in pixels),
not chosen by hand: nothing is held from the photo and no keypoints are hand-picked.

Codes are standardised per dimension so the flow model sees unit-scale targets; a dropped dimension is zero in
the raw code space, as in training.
"""
import torch
from torch import nn

from sang.motion import WIDE_DIM, WIDE_REGIONS, from_target, smooth_placement, to_target, wide, from_wide


class MotionLatent(nn.Module):
    def __init__(self, K: int = 64, kind: str = "ae", hidden: int = 512, dim: int = WIDE_DIM):
        super().__init__()
        if kind not in ("ae", "pca"):
            raise ValueError(f"kind must be 'ae' or 'pca', got {kind!r}")
        self.K, self.kind, self.hidden, self.dim = K, kind, hidden, dim
        for name, n, v in (("mu", dim, 0.0), ("sd", dim, 1.0), ("c_mu", K, 0.0), ("c_sd", K, 1.0)):
            self.register_buffer(name, torch.full((n,), v))
        if kind == "pca":
            self.enc, self.dec = nn.Linear(dim, K, bias=False), nn.Linear(K, dim, bias=False)
        else:
            mlp = lambda a, b: nn.Sequential(nn.Linear(a, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                             nn.Linear(hidden, b))
            self.enc, self.dec = mlp(dim, K), mlp(K, dim)

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
        return {"K": self.K, "kind": self.kind, "hidden": self.hidden, "dim": self.dim, "state": self.state_dict()}

    @classmethod
    def unpack(cls, d: dict) -> "MotionLatent":
        lat = cls(d["K"], d["kind"], d["hidden"], d["dim"])
        lat.load_state_dict(d["state"])
        return lat.eval().requires_grad_(False)


def _norm_stats(d: torch.Tensor):
    return d.mean(0), d.std(0).clamp_min(1e-6)


def fit_pca(d: torch.Tensor, K: int | None = None) -> MotionLatent:
    """Principal components of the standardised deltas [N, dim], strongest first (at most min(N, dim))."""
    mu, sd = _norm_stats(d)
    _, _, v = torch.linalg.svd(((d - mu) / sd).double(), full_matrices=False)
    lat = MotionLatent(min(K or d.shape[1], len(v)), "pca", dim=d.shape[1])
    lat.mu.copy_(mu)
    lat.sd.copy_(sd)
    w = v[: lat.K].float()
    lat.enc.weight.data.copy_(w)
    lat.dec.weight.data.copy_(w.T)
    return lat.set_code_stats(d).eval().requires_grad_(False)


def fit_ae(d: torch.Tensor, K: int = 64, hidden: int = 512, steps: int = 20000, batch: int = 4096,
           lr: float = 1e-3, device: str = "cuda", log_every: int = 2000) -> MotionLatent:
    """Ordered autoencoder on deltas [N, dim]: each sample keeps a uniform random prefix of 1..K codes."""
    lat = MotionLatent(K, "ae", hidden, d.shape[1]).to(device)
    mu, sd = _norm_stats(d)
    lat.mu.copy_(mu)
    lat.sd.copy_(sd)
    x_all = ((d - mu) / sd).to(device)
    opt = torch.optim.AdamW(lat.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ar = torch.arange(K, device=device)
    for step in range(1, steps + 1):
        x = x_all[torch.randint(len(x_all), (batch,), device=device)]
        k = torch.randint(1, K + 1, (batch, 1), device=device)
        loss = ((lat.dec(lat.enc(x) * (ar < k)) - x) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and step % log_every == 0:
            print(f"  ae step {step}: loss {float(loss):.4f}", flush=True)
    return lat.set_code_stats(d).eval().requires_grad_(False)


@torch.no_grad()
def curve(lat: MotionLatent, d: torch.Tensor, ks: list[int]) -> list[dict]:
    """Fraction of variance kept (R^2) of deltas [N, dim] decoded from the first k codes, overall and per region,
    in the latent's standardised units (raw degrees of rotation would otherwise outweigh every keypoint)."""
    d = d.to(lat.mu.device)
    var = (d / lat.sd).var(0)
    out = []
    for k in ks:
        err = ((lat.recon(lat.code(d)[:, :k]) - d) / lat.sd).pow(2).mean(0)
        r2 = lambda idx: float(1 - err[idx].sum() / var[idx].sum().clamp_min(1e-12))
        out.append({"k": k, "r2": r2(list(range(d.shape[1]))), **{f"r2_{n}": r2(i) for n, i in WIDE_REGIONS.items()}})
    return out


class Space:
    """A model's output space <-> 70-d motion. Without a latent: the normalised 42/45-d target. With one: the
    first k standardised codes of the motion relative to the reference frame."""

    def __init__(self, norm, latent: MotionLatent | None = None):
        self.norm, self.latent = norm, latent

    @classmethod
    def of(cls, ck: dict, norm, device: str = "cpu") -> "Space":
        """The space of a training checkpoint (its 'latent' entry, if any)."""
        return cls(norm, MotionLatent.unpack(ck["latent"]).to(device) if ck.get("latent") else None)

    def encode(self, m: torch.Tensor, m_ref: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """Motion [n,70] relative to a reference frame [1,70] -> model space [n, t_dim]."""
        if self.latent is None:
            return self.norm.target(to_target(m, ref=m_ref if self.norm.place else None))
        return self.latent.encode(wide(m) - wide(m_ref), k)

    def decode(self, z: torch.Tensor, base: torch.Tensor, held: str = "camera", ramp: float = 1.0,
               pamp: float = 1.0, psmooth: int = 1) -> torch.Tensor:
        """Model space [n, t_dim] + the reference frame's motion [1,70] -> motion [n,70]. ramp / pamp / psmooth
        are render-time controls of a 42/45-d target: the head rotation scaled about the reference's pose, the
        head placement scaled and smoothed (sang.motion.smooth_placement)."""
        if self.latent is not None:
            if (ramp, pamp, psmooth) != (1.0, 1.0, 1):
                raise ValueError("ramp / pamp / psmooth act on a 42/45-d target, not on a latent")
            return from_wide(wide(base) + self.latent.decode(z), base)
        y = self.norm.untarget(z)
        if ramp != 1.0:
            r0 = to_target(base)[0, :3]
            y = torch.cat([r0 + ramp * (y[:, :3] - r0), y[:, 3:]], 1)
        return from_target(smooth_placement(y, pamp, int(psmooth)), base, held=held)

