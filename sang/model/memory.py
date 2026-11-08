"""Style memory (research log §7.9, reports/SANG style memory.md): who among the TalkVid speakers moves like a new
person. Its nearest speakers calibrate the generator for that person (sang.model.analog).

  StyleKey  a window -> a unit style vector, trained with a supervised InfoNCE over speakers
            (scripts/train_style.py). It never hears the voice, so it cannot tell speakers apart by it. key=residual
            reads per frame the frozen generator's mean motion for the window's audio, photo and prefix (one pass at
            t = 1, no footage) and the real codes minus it: what the speech predicts and how this person departs from
            it (head range and drift, mouth gain and timing, blinks). key=motion reads the real codes alone (the
            ablation).
  Memory    every training speaker's mean style (spk_z), every clip's (query_z) and which speaker each training clip
            is (clip_spk).
"""
import random

import torch
import torch.nn.functional as F
from torch import nn

from sang.model.flow import DemoEncoder, build

TPF, N_SPK = 2, 32


class StyleKey(nn.Module):
    """[B, T, t_in] frames (key_input) -> unit vectors [B, out]: the footage encoder's layout with one query and no
    audio input."""

    def __init__(self, t_in: int, dim: int = 256, out: int = 128, heads: int = 4):
        super().__init__()
        self.cfg = {"t_in": t_in, "dim": dim, "out": out, "heads": heads}
        self.enc = DemoEncoder(dim, t_in, 0, TPF, 1, heads)
        self.head = nn.Linear(dim, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        return F.normalize(self.head(self.enc(x, None, mask)[:, 0]).float(), dim=-1)


@torch.no_grad()
def predicted(gen, batch: dict, generator: torch.Generator | None = None) -> torch.Tensor:
    """gen's mean motion for the window's audio, photo and prefix: x1 - v(x1, t = 1) is E[x0 | condition] for a
    rectified flow (beta = 1, no guidance; no footage)."""
    x0 = batch["target"]
    B, dev = x0.shape[0], x0.device
    x1 = torch.randn(x0.shape, device=dev, generator=generator)
    v = gen(x1, torch.ones(B, device=dev), batch["audio"], batch["ref"], prefix=batch["prefix"],
            prefix_keep=torch.ones(B, dtype=torch.bool, device=dev))
    return x1 - v.float()


def key_input(gen, batch: dict, key: str, generator: torch.Generator | None = None) -> torch.Tensor:
    """What the style key reads of a batch's target frames: residual = [real codes minus gen's mean, the mean]
    [B, L, 2k]; motion = the real codes [B, L, k]. The audio reaches it only through gen's mean."""
    x = batch["target"].float()
    if key == "motion":
        return x
    mu = predicted(gen, batch, generator)
    return torch.cat([x - mu, mu], -1)


def supcon(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """Supervised InfoNCE (Khosla et al. 2020) of unit vectors z [n, e]: every other window of the batch is a
    candidate, the same label's are the positives."""
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    logp = (z @ z.T / tau).masked_fill(eye, float("-inf")).log_softmax(1)
    pos = (labels[:, None] == labels[None]) & ~eye
    return -(logp.masked_fill(~pos, 0).sum(1) / pos.sum(1).clamp_min(1)).mean()


def windows(codes: torch.Tensor, ticks: torch.Tensor, ref: torch.Tensor, P: int, L: int,
            stride: int = 32) -> dict | None:
    """A stretch of codes [T, k] relative to the photo ref [REF_DIM], with its audio ticks [TPF*T, D] -> its
    (P + L)-frame windows every `stride` frames as a batch (prefix, target, audio, ref); None if shorter than one."""
    starts = range(0, len(codes) - P - L + 1, stride)
    if not starts:
        return None
    return {"prefix": torch.stack([codes[s:s + P] for s in starts]),
            "target": torch.stack([codes[s + P:s + P + L] for s in starts]),
            "audio": torch.stack([ticks[TPF * s:TPF * (s + P + L)] for s in starts]),
            "ref": ref.reshape(1, -1).expand(len(starts), -1)}


@torch.no_grad()
def embed(key: StyleKey, gen, codes: torch.Tensor, ticks: torch.Tensor, ref: torch.Tensor, kind: str, P: int,
          L: int, stride: int = 32) -> torch.Tensor | None:
    """The style of one stretch (windows' arguments) -> the unit mean of its windows' keys [e]; None if the stretch
    is shorter than one window."""
    b = windows(codes, ticks, ref, P, L, stride)
    return None if b is None else F.normalize(key(key_input(gen, b, kind)).mean(0), dim=0)


class Memory:
    """scripts/train_style.py's memory.pt."""

    def __init__(self, path_or_dict):
        m = torch.load(path_or_dict, map_location="cpu", weights_only=False) if isinstance(path_or_dict, str) \
            else path_or_dict
        self.m, self.paths, self.clip_spk, self.spk_names = m, m["paths"], m["clip_spk"], m["spk_names"]
        self.spk_z = m["spk_z"].float()
        self.query = {p: i for i, p in enumerate(m["query_paths"])}
        self.query_z = m["query_z"].float()

    def encoders(self, device) -> tuple[StyleKey, torch.nn.Module]:
        """The style key and its frozen residual generator, for a new person's footage (embed)."""
        key = StyleKey(**self.m["key_cfg"]).to(device).eval()
        # memories saved before the neighbour tag was deleted carry its unused (zero) weight
        key.load_state_dict({k: v for k, v in self.m["key_state"].items() if k != "enc.nb"})
        ck = torch.load(self.m["gen"], map_location=device, weights_only=False)
        gen = build(ck["cfg"]).to(device).eval()
        gen.load_state_dict(ck["ema"])
        return key, gen

    def neighbours(self, z: torch.Tensor, n: int = N_SPK, rng: random.Random | None = None) -> torch.Tensor:
        """The n speakers nearest to style z (cosine); rng = n random speakers instead (the control)."""
        S = len(self.spk_z)
        if rng is not None:
            return torch.tensor(rng.sample(range(S), min(n, S)))
        return (self.spk_z @ z.float().cpu()).topk(min(n, S)).indices
