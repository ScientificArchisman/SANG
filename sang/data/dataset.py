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

    preload=True reads every clip once at start-up and keeps it in RAM (~2 MB per 20 s clip), because reading
    one .pt per sample from a network file system starves the GPU.

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

    def clip(self, j):
        return self.mem[j] if self.mem is not None else self._load(self.rows[j])

    def __getitem__(self, i):
        return self.window(i % len(self.rows), random.Random(i) if self.fixed else random)

    def window(self, j, rng, peer=None, stretch=None):
        """A random window of clip j; peer (a clip index) and stretch (start, frames) fix the footage (PersonSets)."""
        y_all, ref_all, audio = self.clip(j)
        n, span = y_all.shape[0], self.L + self.P
        s = rng.randint(0, n - span)
        k = rng.randint(0, n - 1)
        y = y_all[s:s + span] - y_all[k]
        out = {"prefix": y[: self.P], "target": y[self.P:], "ref": ref_all[k],
               "audio": audio[TPF * s: TPF * (s + span)]}
        if self.demo:
            N = self.demo
            dm, mask = torch.zeros(N, y_all.shape[1]), torch.zeros(N, dtype=torch.bool)
            da = torch.zeros(TPF * N, audio.shape[1], dtype=audio.dtype)
            if peer is None and self.peers[j]:
                peer = rng.choice(self.peers[j])
            if peer is not None:
                y2, _, a2 = self.clip(peer)
                if stretch is None:
                    n = min(len(y2), rng.randint(self.demo_min, N))
                    stretch = (rng.randint(0, len(y2) - n), n)
                s2, n = stretch
                dm[:n] = y2[s2:s2 + n] - y_all[k]
                a = a2[TPF * s2: TPF * (s2 + n)]
                da[:len(a)] = a
                mask[:n] = True
            out.update(demo_motion=dm, demo_audio=da, demo_mask=mask, demo_keep=torch.tensor(peer is not None))
        return out


class PersonSets(Dataset):
    """K windows of one person (a video directory) that all read the same footage: one stretch of a clip that none of
    the K windows comes from, so the real windows are held out from what the generator sees (scripts/train/train_set.py
    trains with K = 1, its EVAL reads K = 8). Only persons with two or more clips; K = 1 is one window with footage, as
    MotionWindows gives."""

    def __init__(self, ds: MotionWindows, K: int, fixed: bool = False):
        if not ds.demo:
            raise ValueError("PersonSets needs footage (demo > 0)")
        by_video = {}
        for j, r in enumerate(ds.rows):
            by_video.setdefault(Path(r["path"]).parent.name, []).append(j)
        self.ds, self.K, self.fixed = ds, K, fixed
        self.groups = [v for v in by_video.values() if len(v) > 1]

    def __len__(self):
        return len(self.groups) * self.ds.per_clip

    def __getitem__(self, i):
        rng = random.Random(i) if self.fixed else random
        clips = self.groups[i % len(self.groups)]
        p = rng.choice(clips)
        y2 = self.ds.clip(p)[0]
        n = min(len(y2), rng.randint(self.ds.demo_min, self.ds.demo))
        stretch = (rng.randint(0, len(y2) - n), n)
        rest = [j for j in clips if j != p]
        items = [self.ds.window(rng.choice(rest), rng, p, stretch) for _ in range(self.K)]
        return {k: torch.stack([w[k] for w in items]) for k in items[0]}

