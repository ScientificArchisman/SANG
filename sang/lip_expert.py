"""C2: a motion-space speech expert (THUNDER-style, arXiv:2504.13386) and its go/no-go probe
(scripts/lip_expert.py).

The expert reads speech back from mouth motion: a small temporal conv net maps the 18 mouth columns of
SANG's normalised 42-d target to the top-K principal components of the frame-level WavLM features SANG
is conditioned on (mean of the two 50 Hz ticks of each frame). Inputs and targets have their mean over
the window (training, 64 frames as in SANG) or the clip (evaluation) removed, so the expert reads
movement, not a speaker's resting mouth or a recording's average voice.

Probe (C2.0): trained on real training motion only, the expert is scored on the val clips' real motion
and on SANG's generated motion for the same audio. R^2 = 1 - SSE / SST over all frames and the K
components (variance-weighted: components keep their PCA scale).
    headroom = (R^2_real - R^2_gen) / R^2_real
GO if the expert reads real mouths (R^2_real >= min_r2) and SANG's mouths clearly worse (headroom >=
min_headroom): a loss that makes SANG's mouth 'say' the audio has something to push. NO-GO if SANG's
mouths read as well as real ones. WEAK EXPERT if even real mouths carry too little decodable speech.
"""
import numpy as np
import torch
import torch.nn as nn


def frame_audio(ticks: torch.Tensor, n: int) -> torch.Tensor:
    """WavLM ticks [2n', D] (50 Hz) -> frame features [min(n, n'), D]: the mean of each frame's two ticks."""
    m = min(n, ticks.shape[0] // 2)
    return ticks[: 2 * m].float().view(m, 2, -1).mean(1)


def fit_pca(frames: np.ndarray, k: int) -> dict:
    """Frames [N, D] -> {"mean": [D], "basis": [D, k]} (top-k principal directions)."""
    x = np.asarray(frames, np.float64)
    mu = x.mean(0)
    cov = np.cov((x - mu).T)
    w, v = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1][:k]
    return {"mean": mu.astype(np.float32), "basis": v[:, order].astype(np.float32),
            "explained": float(w[order].sum() / w.sum())}


def project(frames: np.ndarray, pca: dict) -> np.ndarray:
    return ((np.asarray(frames, np.float32) - pca["mean"]) @ pca["basis"]).astype(np.float32)


def centre(x: np.ndarray) -> np.ndarray:
    """Remove the mean over time (axis 0)."""
    x = np.asarray(x, np.float32)
    return x - x.mean(0, keepdims=True)


class LipExpert(nn.Module):
    """Mouth motion [B, T, d_in] -> speech features [B, T, k]. Residual dilated 1-D convs, ~1.6 M
    parameters at width 256, receptive field +-19 frames (0.76 s)."""

    def __init__(self, d_in: int = 18, k: int = 64, width: int = 256, dilations=(1, 2, 4, 8, 1, 1)):
        super().__init__()
        self.inp = nn.Conv1d(d_in, width, 5, padding=2)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), nn.Conv1d(width, width, 3, padding=d, dilation=d),
                                                  nn.GELU(), nn.Conv1d(width, width, 1)) for d in dilations)
        self.out = nn.Conv1d(width, k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return self.out(h).transpose(1, 2)


def r2(pred: list[np.ndarray], target: list[np.ndarray]) -> float:
    """Pooled R^2 over clips, frames and components; targets are already centred per clip."""
    sse = sum(float(((p - t) ** 2).sum()) for p, t in zip(pred, target))
    sst = sum(float((t ** 2).sum()) for t in target)
    return 1 - sse / max(sst, 1e-12)


def clip_r2(p: np.ndarray, t: np.ndarray, rescale: bool = False) -> float:
    """R^2 of one clip. rescale: first scale the prediction by its least-squares factor, so the score
    measures what the mouth tells about the speech, not whether its movement is the right size."""
    if rescale:
        p = p * float((p * t).sum()) / max(float((p * p).sum()), 1e-12)
    return 1 - float(((p - t) ** 2).sum()) / max(float((t ** 2).sum()), 1e-12)


def verdict(r2_real: float, r2_gen: float, min_r2: float = 0.1, min_headroom: float = 0.15) -> str:
    if r2_real < min_r2:
        return "WEAK EXPERT"
    return "GO" if (r2_real - r2_gen) / r2_real >= min_headroom else "NO-GO"
