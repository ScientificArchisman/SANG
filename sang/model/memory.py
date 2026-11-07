"""Style memory (research log §7.9, reports/SANG style memory.md): other TalkVid speakers' windows, retrieved as extra
footage of a new person.

  StyleKey  a window -> a unit style vector, trained with a supervised InfoNCE over speakers
            (scripts/train_style.py). It never hears the voice, so it cannot tell speakers apart by it. key=residual
            reads per frame the frozen generator's mean motion for the window's audio, photo and prefix (one pass at
            t = 1, no footage) and the real codes minus it: what the speech predicts and how this person departs from
            it (head range and drift, mouth gain and timing, blinks). key=motion reads the real codes alone (the
            ablation).
  Memory    every training window's audio key, and every speaker's mean style. Two keys pick a window's
            demonstrations: the footage's style -> the N_SPK nearest speakers; the window's own audio -> the N_WIN
            windows of those speakers whose audio is nearest.
The demonstrations enter the footage encoder as extra frames tagged as another speaker's (DemoEncoder.nb), with their
motion re-centred on this person's mean pose, so only the dynamics carry over.
"""
import functools
import random

import torch
import torch.nn.functional as F
from torch import nn

from sang.data.sync import shift_ticks
from sang.model.flow import DemoEncoder, build
from sang.model.target import wide

TPF, N_SPK, N_WIN = 2, 32, 4


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


@torch.no_grad()
def embed(key: StyleKey, gen, codes: torch.Tensor, ticks: torch.Tensor, ref: torch.Tensor, kind: str, P: int,
          L: int, stride: int = 32) -> torch.Tensor | None:
    """The style of one stretch: codes [T, k] relative to the photo ref [REF_DIM], its audio ticks [TPF*T, D] ->
    the unit mean of its (P + L)-frame windows' keys [e]; None if the stretch is shorter than one window."""
    starts = range(0, len(codes) - P - L + 1, stride)
    if not starts:
        return None
    b = {"prefix": torch.stack([codes[s:s + P] for s in starts]), "target": torch.stack([codes[s + P:s + P + L] for s in starts]),
         "audio": torch.stack([ticks[TPF * s:TPF * (s + P + L)] for s in starts]), "ref": ref.expand(len(starts), -1)}
    return F.normalize(key(key_input(gen, b, kind)).mean(0), dim=0)


class Memory:
    """scripts/train_style.py's memory.pt. Neighbour windows are read from the TalkVid cache files (load), or from a
    training dataset's RAM (bind)."""

    def __init__(self, path_or_dict):
        m = torch.load(path_or_dict, map_location="cpu", weights_only=False) if isinstance(path_or_dict, str) \
            else path_or_dict
        self.m, self.L = m, int(m["L"])
        self.paths, self.shift, self.clip_spk = m["paths"], m["shift"], m["clip_spk"]
        self.spk_z, self.win_key = m["spk_z"].float(), m["win_key"].float()
        self.win_clip, self.win_start = m["win_clip"], m["win_start"]
        self.spk_index = {s: i for i, s in enumerate(m["spk_names"])}
        self.query = {p: i for i, p in enumerate(m["query_paths"])}
        self.query_z = m["query_z"].float()
        spk = self.clip_spk[self.win_clip]
        order = torch.argsort(spk, stable=True)
        self.spk_windows = list(torch.split(order, torch.bincount(spk, minlength=len(self.spk_z)).tolist()))
        self.load = functools.lru_cache(64)(self._read)

    def _read(self, c: int):
        d = torch.load(self.paths[c], map_location="cpu", weights_only=True)
        return wide(d["m"].float()), shift_ticks(d["audio"], int(self.shift[c]))

    def bind(self, ds) -> None:
        """Read the neighbour windows from ds (a MotionWindows over the memory's clips), already in RAM."""
        row = {r["path"]: j for j, r in enumerate(ds.rows)}
        missing = [p for p in self.paths if p not in row]
        if missing:
            raise ValueError(f"{len(missing)} memory clips are not in this dataset, e.g. {missing[0]}: rebuild the "
                             "memory from the same cache and split")

        def load(c):
            y, _, a = ds.clip(row[self.paths[c]])
            return y, a
        self.load = load

    def encoders(self, device) -> tuple[StyleKey, torch.nn.Module]:
        """The style key and its frozen residual generator, for a new person's footage (embed)."""
        key = StyleKey(**self.m["key_cfg"]).to(device).eval()
        key.load_state_dict(self.m["key_state"])
        ck = torch.load(self.m["gen"], map_location=device, weights_only=False)
        gen = build(ck["cfg"]).to(device).eval()
        gen.load_state_dict(ck["ema"])
        return key, gen

    def akey(self, ticks: torch.Tensor) -> torch.Tensor:
        """A window's audio key: its mean WavLM tick projected on the memory's principal directions, unit length."""
        return F.normalize((ticks.float().mean(-2).cpu() - self.m["a_mu"]) @ self.m["a_proj"], dim=-1)

    def neighbours(self, z: torch.Tensor, n: int = N_SPK, exclude: int | None = None,
                   rng: random.Random | None = None) -> torch.Tensor:
        """The n speakers nearest to style z (cosine), never `exclude`; rng = n random speakers instead (the
        control)."""
        S = len(self.spk_z)
        if rng is not None:
            return torch.tensor(rng.sample([s for s in range(S) if s != exclude], min(n, S - (exclude is not None))))
        sim = self.spk_z @ z.float().cpu()
        if exclude is not None:
            sim[exclude] = float("-inf")
        return sim.topk(min(n, S - (exclude is not None))).indices

    def windows(self, spk: torch.Tensor, akey: torch.Tensor, n: int = N_WIN,
                rng: random.Random | None = None) -> torch.Tensor:
        """The n windows of speakers spk whose audio key is nearest to akey; rng = n random ones of theirs."""
        cand = torch.cat([self.spk_windows[s] for s in spk.tolist()])
        if rng is not None:
            return cand[torch.tensor(rng.sample(range(len(cand)), min(n, len(cand))))]
        return cand[(self.win_key[cand] @ akey).topk(min(n, len(cand))).indices]

    def fetch(self, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Window w: its 69-d motion minus its own mean [L, 69] (another speaker's pose offset does not carry over)
        and its audio ticks [TPF*L, D]."""
        y, a = self.load(int(self.win_clip[w]))
        s = int(self.win_start[w])
        y = y[s:s + self.L]
        return y - y.mean(0), a[TPF * s:TPF * (s + self.L)]
