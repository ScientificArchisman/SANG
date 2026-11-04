"""The motion cache as training and evaluation read it: the speaker split, the photo normalisation and windows."""
import hashlib
import random
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset

from sang.data.sync import clip_key, shift_ticks
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


def fit_norm(rows: list[dict], n_clips: int = 500, seed: int = 0) -> Norm:
    """Statistics of the photo input (sang.model.flow.Norm) over random frames of n_clips clips."""
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
    clip (Ditto's choice), which at inference becomes the source image.

    Targets: the raw sang.model.target.wide vector minus the reference frame's, 69-d deltas that the training loop
    encodes into latent codes on the GPU. The reference is described to the model as Norm.ref of its x_c and
    motion.

    preload=True reads every clip once at start-up and keeps it in RAM (~2 MB per 20 s clip): reading
    one .pt per SAMPLE from BeeGFS capped the first run at ~1.4 it/s.

    demo=N > 0 adds extra footage of the same person: a random stretch of demo_min..N frames of ANOTHER clip of the
    same video directory (so never the target's own frames), as 69-d deltas to the same photo frame, with its
    audio, padded to N frames (demo_mask marks the real ones). demo_keep is False where the video has no other
    clip."""

    def __init__(self, rows, L: int, P: int, norm: Norm, per_clip: int, fixed: bool = False,
                 preload: bool = True, threads: int = 16, shifts: dict | None = None, demo: int = 0,
                 demo_min: int = 25):
        self.rows = [r for r in rows if r["n"] >= L + P]
        self.L, self.P, self.norm, self.per_clip = L, P, norm, per_clip
        self.demo, self.demo_min = demo, min(demo_min, demo)
        by_video = {}
        for i, r in enumerate(self.rows):
            by_video.setdefault(Path(r["path"]).parent.name, []).append(i)
        self.peers = [[j for j in by_video[Path(r["path"]).parent.name] if j != i] for i, r in enumerate(self.rows)]
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
        y = wide(m)
        ref = self.norm.ref(d["kp"].float(), m)
        audio = shift_ticks(d["audio"], self.shifts.get(clip_key(r), 0))
        return y, ref, audio.half().contiguous()

    def __len__(self):
        return len(self.rows) * self.per_clip

    def __getitem__(self, i):
        j = i % len(self.rows)
        y_all, ref_all, audio = self.mem[j] if self.mem is not None else self._load(self.rows[j])
        n, span = y_all.shape[0], self.L + self.P
        rng = random.Random(i) if self.fixed else random
        s = rng.randint(0, n - span)
        k = rng.randint(0, n - 1)
        y = y_all[s:s + span] - y_all[k]
        out = {"prefix": y[: self.P], "target": y[self.P:], "ref": ref_all[k],
               "audio": audio[TPF * s: TPF * (s + span)]}
        if self.demo:
            N = self.demo
            dm, mask = torch.zeros(N, y_all.shape[1]), torch.zeros(N, dtype=torch.bool)
            da = torch.zeros(TPF * N, audio.shape[1], dtype=audio.dtype)
            if self.peers[j]:
                p = rng.choice(self.peers[j])
                y2, _, a2 = self.mem[p] if self.mem is not None else self._load(self.rows[p])
                n = min(len(y2), rng.randint(self.demo_min, N))
                s2 = rng.randint(0, len(y2) - n)
                dm[:n] = y2[s2:s2 + n] - y_all[k]
                a = a2[TPF * s2: TPF * (s2 + n)]
                da[:len(a)] = a
                mask[:n] = True
            out.update(demo_motion=dm, demo_audio=da, demo_mask=mask, demo_keep=torch.tensor(bool(self.peers[j])))
        return out

