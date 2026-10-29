"""The motion cache as training and evaluation read it: the speaker split, target normalisation and windows."""
import hashlib
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

from sang.data.sync import clip_key, shift_ticks
from sang.eval.naturalness import typical_frames
from sang.model.flow import Norm
from sang.model.target import to_target, wide

TPF = 2


def load_cfg(config: str, overrides: list[str]) -> dict:
    """A yaml config with key=value overrides parsed as YAML ('1e-4' as a float)."""
    cfg = yaml.safe_load(Path(config).read_text())
    for o in overrides:
        k, v = o.split("=", 1)
        val = yaml.safe_load(v)
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        cfg[k] = val
    return cfg


def split_by_speaker(rows: list[dict], val_frac: float, seed: int):
    """Speaker-disjoint split, decided per speaker by a hash of its name.

    The speaker is the clip's parent directory, so val measures unseen identities. Hashing (rather
    than shuffling the speakers present) keeps every speaker on the same side as the cache grows,
    so every run validates on the same people. `seed` salts the hash."""
    def is_val(spk: str) -> bool:
        return int(hashlib.md5(f"{seed}:{spk}".encode()).hexdigest(), 16) % 10_000 < val_frac * 10_000

    val = {Path(r["path"]).parent.name for r in rows if is_val(Path(r["path"]).parent.name)}
    return ([r for r in rows if Path(r["path"]).parent.name not in val],
            [r for r in rows if Path(r["path"]).parent.name in val])


def val_clips(rows: list[dict], min_frames: int) -> list[str]:
    """One clip per val speaker (the first by name that starts at frame 0 and is long enough), for demos."""
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["clip"]):
        spk = Path(r["path"]).parent.name
        if spk not in seen and r["n"] >= min_frames and r.get("start", 0) == 0:
            seen.add(spk)
            out.append(r["clip"])
    return out


def fit_norm(rows: list[dict], n_clips: int = 500, seed: int = 0) -> Norm:
    pick = random.Random(seed).sample(rows, min(n_clips, len(rows)))
    ys, kps = [], []
    for r in pick:
        d = torch.load(r["path"], map_location="cpu", weights_only=True)
        ys.append(to_target(d["m"].float()))
        kps.append(d["kp"].float())
    return Norm.fit(torch.cat(ys), torch.cat(kps))


def wide_deltas(rows: list[dict], n_clips: int, seed: int = 0) -> torch.Tensor:
    """sang.model.target.wide of every frame minus that of a random frame of the same clip: [N, 69], the deltas a
    motion latent (sang.model.latent) is fitted on, drawn as training draws reference frames."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for r in random.Random(seed).sample(rows, min(n_clips, len(rows))):
        w = wide(torch.load(r["path"], map_location="cpu", weights_only=True)["m"].float())
        out.append(w - w[torch.randint(len(w), (len(w),), generator=g)])
    return torch.cat(out)


class MotionWindows(Dataset):
    """Random windows of P prefix + L target frames. The reference is a random frame of the same
    clip (Ditto's choice), which at inference becomes the source image. ref_frame='median': a random
    one of the clip's TYPICAL frames instead (lip opening in the middle `ref_band` of the clip by rank,
    sang.eval.naturalness.typical_frames).

    Targets: the normalised 42-d target; with wide=True the raw sang.model.target.wide vector minus the
    reference frame's (69-d deltas for a motion latent, which the training loop encodes on the GPU).

    preload=True reads every clip once at start-up and keeps it in RAM (~2 MB per 20 s clip): reading
    one .pt per SAMPLE from BeeGFS capped the first run at ~1.4 it/s."""

    def __init__(self, rows, L: int, P: int, norm: Norm, per_clip: int, fixed: bool = False,
                 preload: bool = True, threads: int = 16, shifts: dict | None = None,
                 ref_frame: str = "random", lip=None, ref_band: float = 0.2, wide: bool = False):
        if ref_frame not in ("random", "median"):
            raise ValueError(f"ref_frame must be 'random' or 'median', got {ref_frame!r}")
        if ref_frame == "median" and lip is None:
            raise ValueError("ref_frame='median' needs the lip-opening readout (openness.json)")
        self.ref_frame, self.lip, self.ref_band, self.wide = ref_frame, lip, ref_band, wide
        self.rows = [r for r in rows if r["n"] >= L + P]
        self.L, self.P, self.norm, self.per_clip = L, P, norm, per_clip
        self.shifts = shifts or {}
        self.fixed = fixed
        if not self.rows:
            raise ValueError(f"no clip has {L + P} frames")
        self.mem = None
        if preload:
            from concurrent.futures import ThreadPoolExecutor
            t0 = time.time()
            with ThreadPoolExecutor(threads) as ex:
                self.mem = list(ex.map(self._load, self.rows))
            gb = sum(m[2].numel() * m[2].element_size() for m in self.mem) / 1e9
            print(f"  preloaded {len(self.mem)} clips in {time.time() - t0:.0f} s, audio {gb:.1f} GB", flush=True)

    def _load(self, r):
        d = torch.load(r["path"], map_location="cpu", weights_only=True)
        m = d["m"].float()
        y_raw = to_target(m)
        y = wide(m) if self.wide else self.norm.target(y_raw)
        ref = self.norm.ref(d["kp"].float(), y_raw)
        audio = shift_ticks(d["audio"], self.shifts.get(clip_key(r), 0))
        cands = (torch.from_numpy(typical_frames(self.lip(y_raw).numpy(), self.ref_band))
                 if self.ref_frame == "median" else None)
        return y, ref, audio.half().contiguous(), cands

    def __len__(self):
        return len(self.rows) * self.per_clip

    def __getitem__(self, i):
        j = i % len(self.rows)
        y_all, ref_all, audio, cands = self.mem[j] if self.mem is not None else self._load(self.rows[j])
        n, span = y_all.shape[0], self.L + self.P
        rng = random.Random(i) if self.fixed else random
        s = rng.randint(0, n - span)
        k = int(cands[rng.randrange(len(cands))]) if cands is not None else rng.randint(0, n - 1)
        y = y_all[s:s + span] - y_all[k] if self.wide else y_all[s:s + span]
        return {"prefix": y[: self.P], "target": y[self.P:], "ref": ref_all[k],
                "audio": audio[TPF * s: TPF * (s + span)]}

