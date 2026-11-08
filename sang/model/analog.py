"""Analog calibration of the sampler (reports/SANG retrieved speaker calibration.md, rank 2). The style memory's
nearest TalkVid speakers show how the deployed velocity errs for people like a new person, and a closed-form
correction removes that person's share of the error. No DiT weight changes; a fit is a few hundred forward passes and
one small solve.

At each of the sampler's times t_k, real windows x0 are noised to x_t = (1 - t) x0 + t eps, and the deployed velocity
v (one pass at beta = 1/g, with footage tokens that never include the window: cross_fit) is compared with the flow
target, per frame:
    phi = [U^T v, 1],   y = U^T (eps - x0 - v)
U [k, r] is the top principal directions of the residual over TalkVid. A least-squares fit of y on phi is additive in
its sufficient statistics (phi phi^T, phi y^T, y y^T, n). So every training speaker's are summed once
(scripts/build_analog.py), and a person's fit only adds them up:
    M = alpha * own / n_own + (1 - alpha) * neighbours / n_nb,   W = M_pp^-1 M_py
The population's fit W_pop is what the sampler does to everyone, guidance included. The correction D is only this
person's departure from it, under the path KL (1/2) sum_k dt ||dv_k||^2 with weight lam:
    min_D E_M ||(y - phi W_pop) - phi D||^2 + lam E_M ||phi D||^2   ->   D = (W - W_pop) / (1 + lam)   (exact)
The sampler then steps with v + U (phi(v) D_k).

Modes (who supplies M): own = the person's footage (alpha 1), matched = the N_SPK speakers nearest to the footage's
style (alpha 0), random = N_SPK random speakers (the control), own_matched / own_random = both, by alpha.
"""
import random

import torch

from sang.model.flow import flow_times
from sang.model.memory import N_SPK, TPF, Memory, embed, windows

MODES = ("none", "own", "matched", "random", "own_matched", "own_random")
KEYS = ("pp", "py", "yy", "n")


@torch.no_grad()
def residuals(gen, batch: dict, demo: torch.Tensor | None, steps: int, g: float, draws: int = 1,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Real windows (prefix [B,P,k], target [B,L,k], audio, ref) and their footage tokens demo [B,M,dim] -> the deployed
    velocity v and the residual eps - x0 - v at the sampler's times, [steps, B*draws, L, k] (draws fresh noises per
    window, window-major)."""
    rep = lambda x: None if x is None else x.repeat_interleave(draws, 0)
    x0 = rep(batch["target"]).float()
    pre, audio, ref, demo = rep(batch["prefix"]), rep(batch["audio"]), rep(batch["ref"]), rep(demo)
    B, dev = len(x0), x0.device
    keep = torch.ones(B, dtype=torch.bool, device=dev)
    beta = torch.full((B,), 1.0 / g, device=dev)
    vs, rs = [], []
    for t in flow_times(steps)[:-1]:
        eps = torch.randn(x0.shape, device=dev, generator=generator)
        v = gen((1 - t) * x0 + t * eps, torch.full((B,), t, device=dev), audio, ref, prefix=pre, prefix_keep=keep,
                beta=beta, demo=demo).float()
        vs.append(v)
        rs.append(eps - x0 - v)
    return torch.stack(vs), torch.stack(rs)


def moments(v: torch.Tensor, r: torch.Tensor) -> dict:
    """v, r [K, B, L, k] -> each entry's sufficient statistics: pp [B, K, k+1, k+1], py [B, K, k+1, k],
    yy [B, K, k, k], n [B] (frames)."""
    phi = torch.cat([v, torch.ones_like(v[..., :1])], -1).double()
    y = r.double()
    ein = lambda a, b: torch.einsum("kbli,kblj->bkij", a, b)
    return {"pp": ein(phi, phi), "py": ein(phi, y), "yy": ein(y, y),
            "n": torch.full((v.shape[1],), float(v.shape[2]), dtype=torch.float64, device=v.device)}


def total(S: dict, idx=None) -> dict:
    """The sum of statistics over their first dimension (or over the rows idx)."""
    return {k: (S[k] if idx is None else S[k][idx]).sum(0) for k in KEYS}


def project(S: dict, U: torch.Tensor) -> dict:
    """Full-space statistics -> those of phi = [U^T v, 1] and y = U^T r (U [k, r])."""
    k, r = U.shape
    U = U.to(S["pp"])
    T = torch.zeros(r + 1, k + 1, dtype=U.dtype, device=U.device)
    T[:r, :k], T[r, k] = U.T, 1.0
    return {"pp": T @ S["pp"] @ T.T, "py": T @ S["py"] @ U, "yy": U.T @ S["yy"] @ U, "n": S["n"]}


def mix(own: dict | None, nb: dict | None, alpha: float) -> dict:
    """alpha * own + (1 - alpha) * nb as normalised moments; a missing side takes no weight."""
    parts = [(w, s) for w, s in ((alpha, own), (1 - alpha, nb)) if s is not None and w > 0]
    if not parts:
        raise ValueError("no statistics to fit on")
    tot = sum(w for w, _ in parts)
    out = {k: sum(w / tot * s[k] / s["n"] for w, s in parts) for k in KEYS[:3]}
    return {**out, "n": torch.ones((), dtype=out["pp"].dtype, device=out["pp"].device)}


def fit(S: dict, ridge: float = 1e-6) -> torch.Tensor:
    """The least-squares W [K, r+1, r] of y on phi."""
    pp = S["pp"] / S["n"]
    eye = torch.eye(pp.shape[-1], dtype=pp.dtype, device=pp.device)
    pp = pp + ridge * pp.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye  # a speaker may lack a direction
    return torch.linalg.solve(pp, S["py"] / S["n"])


def correction(S: dict, W_pop: torch.Tensor, lam: float) -> torch.Tensor:
    """D [K, r+1, r]: the person's departure from the population fit, shrunk by the path KL."""
    return (fit(S) - W_pop) / (1 + lam)


def loss(S: dict, W_pop: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Per step [K]: E_S ||(y - phi W_pop) - phi D||^2, how far the corrected residual is from the population's."""
    E = W_pop + D
    tr = lambda m: m.diagonal(dim1=-2, dim2=-1).sum(-1)
    pp, py, yy = (S[k] / S["n"] for k in KEYS[:3])
    Et = E.transpose(-1, -2)
    return tr(yy) - 2 * tr(Et @ py) + tr(Et @ pp @ E)


def cross_fit(gen, codes: torch.Tensor, ticks: torch.Tensor, ref: torch.Tensor, P: int, L: int, stride: int = 32):
    """The footage in two halves, each half's windows with the other half's footage tokens: no window reads its own
    frames, which training never shows. -> (batch, tokens [n, M, dim]) for residuals, or None if a half is shorter
    than a window."""
    T = len(codes) // 2
    halves = [(codes[:T], ticks[:TPF * T]), (codes[T:2 * T], ticks[TPF * T:TPF * 2 * T])]
    out = []
    for (c, a), (fc, fa) in zip(halves, halves[::-1]):
        b = windows(c, a, ref, P, L, stride)
        if b is None:
            return None
        tok = gen.demo(fc[None], fa.float()[None], torch.ones(1, len(fc), dtype=torch.bool, device=fc.device))
        out.append((b, tok.expand(len(b["target"]), -1, -1)))
    return {k: torch.cat([b[k] for b, _ in out]) for k in out[0][0]}, torch.cat([t for _, t in out])


def corrector(U: torch.Tensor, D: torch.Tensor):
    """sang.model.flow.sample's correct: at step i, v + U (phi(v) D_i)."""
    U, D = U.float(), D.float()

    def correct(i, v):
        return v + ((v @ U) @ D[i, :-1] + D[i, -1]) @ U.T
    return correct


class Analog:
    """scripts/build_analog.py's analog.pt: a new person's correction from their footage, by mode."""

    def __init__(self, path: str, device):
        a = torch.load(path, map_location="cpu", weights_only=False)
        self.a, self.dev = a, device
        self.memory = Memory(a["memory"])
        if self.memory.spk_names != a["spk_names"]:
            raise ValueError(f"{path} was built from another memory than {a['memory']}: rebuild it")
        self.key, self.key_gen = self.memory.encoders(device)
        self.U = a["U"].double().to(device)
        self.W_pop = a["W_pop"].double().to(device)
        self.spk = {k: a["spk"][k].double().to(device) for k in KEYS}

    @torch.no_grad()
    def __call__(self, gen, codes, ticks, ref, mode: str, P: int, L: int, steps: int, cfg_audio: float,
                 seed: int = 0):
        """The footage codes [T, k] relative to the photo ref and its ticks [2T, D] -> (sample's correct, info:
        nb_sim = the neighbours' mean cosine to the footage's style, d_norm = |D|); (None, {}) for mode none or
        footage too short for the mode (own needs two windows' worth, the neighbours one)."""
        if mode not in MODES:
            raise ValueError(f"unknown analog mode {mode}; known: {', '.join(MODES)}")
        if (steps, float(cfg_audio)) != (len(self.W_pop), float(self.a["g"])):
            raise ValueError(f"the analog statistics are for g={self.a['g']:g}, steps={len(self.W_pop)}; the "
                             f"sampler is g={cfg_audio:g}, steps={steps}")
        if mode == "none":
            return None, {}
        info, own, nb = {}, None, None
        if mode.startswith("own"):
            cf = cross_fit(gen, codes, ticks, ref, P, L, int(self.a["stride"]))
            if cf is not None:
                g = torch.Generator(device=codes.device).manual_seed(seed)
                v, r = residuals(gen, *cf, steps, cfg_audio, int(self.a["draws_own"]), g)
                own = project(total(moments(v, r)), self.U)
        if mode != "own":
            z = embed(self.key, self.key_gen, codes, ticks, ref, self.memory.m["key"], P, L)
            if z is not None:
                rng = random.Random(f"{seed}:{float(z.sum()):.6f}") if mode.endswith("random") else None
                spk = self.memory.neighbours(z, N_SPK, rng)
                info["nb_sim"] = float((self.memory.spk_z[spk] @ z.float().cpu()).mean())
                nb = total(self.spk, spk.to(self.dev))
        if own is None and nb is None:
            return None, {}
        arm = self.a["arms"][mode]
        D = correction(mix(own, nb, arm["alpha"]), self.W_pop, arm["lam"])
        info["d_norm"] = float(D.norm())
        return corrector(self.U, D), info
