"""Naturalness metrics in motion space (no rendering): lip sync against the real trajectory, closure at
/p b m/, blinks, and head strokes against loudness onsets.

Openness is read out of the 42-d target with LINEAR maps fitted to LivePortrait's own landmark ratios
(scripts/calibrate_openness.py). Phonemes come from a language-agnostic IPA recogniser (wav2vec2 XLSR-53
espeak, CTC at 50 Hz), because TalkVid spans 15 languages and has no transcripts to force-align.
"""
import json
from pathlib import Path

import numpy as np
import torch

from sang.model.target import T_DIM

FPS, SR, TPF = 25, 16000, 2
HOP = SR // FPS
PHONEME_MODEL = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
BILABIAL = set("pbmɸβ")


def _dist_ratio(lmk: np.ndarray, i1: int, i2: int, i3: int, i4: int) -> np.ndarray:
    lmk = np.asarray(lmk, np.float64)
    return (np.linalg.norm(lmk[..., i1, :] - lmk[..., i2, :], axis=-1)
            / (np.linalg.norm(lmk[..., i3, :] - lmk[..., i4, :], axis=-1) + 1e-6))


def lip_ratio(lmk: np.ndarray) -> np.ndarray:
    """Lip gap / mouth width. [..., 203, 2] -> [...]."""
    return _dist_ratio(lmk, 90, 102, 48, 66)


def eye_ratio(lmk: np.ndarray) -> np.ndarray:
    """Mean of the two eyes' lid gap / eye width. [..., 203, 2] -> [...]."""
    return 0.5 * (_dist_ratio(lmk, 6, 18, 0, 12) + _dist_ratio(lmk, 30, 42, 24, 36))


class Readout:
    """r = w . y + b on the RAW 42-d target (w is zero outside its region)."""

    def __init__(self, w, b: float, r2: float = float("nan")):
        self.w, self.b, self.r2 = torch.as_tensor(w, dtype=torch.float32), float(b), float(r2)

    def __call__(self, y_raw: torch.Tensor) -> torch.Tensor:
        return y_raw.float() @ self.w.to(y_raw.device) + self.b

    def to_dict(self) -> dict:
        return {"w": self.w.tolist(), "b": self.b, "r2": self.r2}


def fit_readout(Y: torch.Tensor, r: torch.Tensor, cols: list[int], groups: np.ndarray,
                lam: float = 1e-2) -> Readout:
    """Ridge from Y[:, cols] to r. R^2 is measured on held-out CLIPS (every 5th group), then the
    map is refitted on everything."""
    X, r = Y[:, cols].double(), r.double()

    def solve(Xs, rs):
        mu, sd = Xs.mean(0), Xs.std(0).clamp_min(1e-8)
        Z = (Xs - mu) / sd
        beta = torch.linalg.solve(Z.T @ Z + lam * len(Z) * torch.eye(Z.shape[1], dtype=Z.dtype), Z.T @ (rs - rs.mean()))
        w = beta / sd
        return w, float(rs.mean() - mu @ w)

    g = np.unique(groups)
    held = torch.from_numpy(np.isin(groups, g[::5]))
    w, b = solve(X[~held], r[~held])
    pred = X[held] @ w + b
    r2 = float(1 - ((pred - r[held]) ** 2).sum() / ((r[held] - r[held].mean()) ** 2).sum())
    w, b = solve(X, r)
    full = torch.zeros(T_DIM, dtype=torch.float64)
    full[cols] = w
    return Readout(full.float(), b, r2)


def load_readouts(path: Path) -> dict[str, Readout]:
    d = json.loads(Path(path).read_text())
    return {k: Readout(v["w"], v["b"], v.get("r2", float("nan"))) for k, v in d.items() if k in ("lip", "eye")}


def energy_db(wav: torch.Tensor, n: int) -> np.ndarray:
    """RMS level per video frame, dB. [N] -> [n]."""
    w = torch.nn.functional.pad(wav.float().flatten(), (0, max(0, n * HOP - wav.numel())))[: n * HOP]
    return (10 * torch.log10(w.view(n, HOP).pow(2).mean(1) + 1e-10)).numpy()


def pauses(db: np.ndarray, below: float = 30.0, min_len: int = 3) -> np.ndarray:
    """Frames more than `below` dB under the clip's loud level (95th pct), in runs >= min_len."""
    quiet = db < np.percentile(db, 95) - below
    out = np.zeros_like(quiet)
    for s, e in _runs(quiet):
        if e - s >= min_len:
            out[s:e] = True
    return out


def is_bilabial(tok: str) -> bool:
    return bool(tok) and not tok.startswith(("<", "[")) and tok[0] in BILABIAL


class PhonemeRecognizer:
    """wav2vec2 XLSR-53 fine-tuned to espeak IPA with CTC: 50 Hz posteriors, any language.
    Download once on the login node: huggingface-cli download facebook/wav2vec2-xlsr-53-espeak-cv-ft"""

    def __init__(self, device: str = "cuda"):
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        self.m = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL, local_files_only=True).to(device).eval()
        vocab = json.loads(Path(hf_hub_download(PHONEME_MODEL, "vocab.json", local_files_only=True)).read_text())
        self.bilabial = torch.tensor(sorted(i for t, i in vocab.items() if is_bilabial(t)))
        self.device = device

    @torch.no_grad()
    def posteriors(self, wav: torch.Tensor) -> torch.Tensor:
        w = wav.float().flatten().to(self.device)
        w = (w - w.mean()) / (w.var() + 1e-7).sqrt()
        return self.m(w[None]).logits[0].float().softmax(-1).cpu()

    def bilabials(self, wav: torch.Tensor, n: int, min_prob: float = 0.5) -> list[int]:
        return bilabial_events(self.posteriors(wav), self.bilabial, n, min_prob)


def bilabial_events(post: torch.Tensor, ids: torch.Tensor, n: int, min_prob: float = 0.5) -> list[int]:
    """CTC posteriors [ticks, V] -> the video frame at the centre of each run of ticks whose
    bilabial probability exceeds min_prob. Tick k covers ~[20k, 20k + 25] ms."""
    pb = post[:, ids].sum(1).numpy() if len(ids) else np.zeros(len(post))
    out = []
    for s, e in _runs(pb > min_prob):
        f = int(((s + e - 1) / 2 * 0.02 + 0.0125) * FPS)
        if f < n:
            out.append(f)
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[s, e) runs of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[::2], d[1::2]))


def blink_events(eye: np.ndarray, drop: float = 0.35, win: int = 51, max_len: int = 15) -> list[tuple[int, int, int]]:
    """Eye-openness trace [n] -> [(start, deepest, end)]. A blink is a dip of more than `drop`
    (relative) below the running median over `win` frames (2 s), shorter than max_len frames
    (0.6 s: longer closures are looking down or eyes shut, not blinks). Gaps <= 2 frames merge."""
    eye = np.asarray(eye, np.float64)
    n = len(eye)
    if n == 0:
        return []
    h = win // 2
    padded = np.pad(eye, h, mode="edge")
    base = np.array([np.median(padded[i:i + win]) for i in range(n)])
    rel = (base - eye) / np.maximum(np.abs(base), 1e-6)
    runs = _runs(rel > drop)
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return [(s, s + int(np.argmin(eye[s:e])), e) for s, e in merged if e - s <= max_len]


def closure_minima(lip: np.ndarray, events: list[int], offset: int = 0, half: int = 2) -> np.ndarray:
    """Lowest lip openness within +/-half frames of each bilabial (shifted by offset)."""
    lip = np.asarray(lip)
    out = [lip[max(0, f + offset - half):f + offset + half + 1].min()
           for f in events if 0 <= f + offset < len(lip)]
    return np.array(out)


def closure_offset(lip: np.ndarray, events: list[int], reach: int = 4) -> list[int]:
    """Where, relative to each CTC bilabial frame, the REAL lips are most closed."""
    out = []
    for f in events:
        lo, hi = max(0, f - reach), min(len(lip), f + reach + 1)
        if hi - lo == 2 * reach + 1:
            out.append(int(np.argmin(lip[lo:hi])) + lo - f)
    return out


def peaks(x: np.ndarray, thresh: float, min_dist: int = 3) -> np.ndarray:
    """Local maxima above thresh, greedily at least min_dist apart (largest first)."""
    x = np.asarray(x, np.float64)
    if len(x) < 3:
        return np.array([], int)
    c = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > thresh)) + 1
    keep = []
    for i in c[np.argsort(-x[c])]:
        if all(abs(i - j) >= min_dist for j in keep):
            keep.append(i)
    return np.array(sorted(keep), int)


def head_beats(rot: np.ndarray) -> np.ndarray:
    """Peaks of head angular speed (deg/frame) above its 60th percentile: the strokes of nods/turns."""
    s = np.linalg.norm(np.diff(np.asarray(rot, np.float64), axis=0), axis=1)
    return peaks(s, np.percentile(s, 60) if len(s) else 0.0) + 1 if len(s) else np.array([], int)


def audio_onsets(db: np.ndarray, rise: float = 3.0) -> np.ndarray:
    """Loudness onsets: peaks of the frame-to-frame dB rise above `rise` dB."""
    d = np.diff(np.asarray(db, np.float64), prepend=db[0])
    return peaks(d, rise)


def beat_alignment(hb: np.ndarray, ab: np.ndarray, n: int, sigma: float = 2.0,
                   shifts: int = 8, seed: int = 0) -> tuple[float, float]:
    """Bailando-style score: mean over head beats of exp(-d^2 / 2 sigma^2), d = frames to the
    nearest audio onset. Also the same score against circularly shifted onsets (>= 1 s), i.e.
    chance for this clip's beat densities. (nan, nan) if either side has no beats."""
    if len(hb) == 0 or len(ab) == 0:
        return float("nan"), float("nan")

    def score(a):
        d = np.abs(hb[:, None] - a[None, :]).min(1)
        return float(np.exp(-d ** 2 / (2 * sigma ** 2)).mean())

    rng = np.random.default_rng(seed)
    lo, hi = FPS, max(FPS + 1, n - FPS)
    chance = [score(np.sort((ab + rng.integers(lo, hi)) % n)) for _ in range(shifts)]
    return score(ab), float(np.mean(chance))


def pause_near(pause: np.ndarray, reach: int = 5) -> np.ndarray:
    """Frames within `reach` of a pause frame."""
    return np.convolve(pause.astype(float), np.ones(2 * reach + 1), mode="same") > 0


def ccc(x, y) -> float:
    """Concordance of two trajectories: timing AND level AND amplitude (1 = identical)."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    cov = float(np.mean((x - x.mean()) * (y - y.mean())))
    return 2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2 + 1e-12)

