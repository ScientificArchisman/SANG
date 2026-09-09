import sys
from pathlib import Path

import torch
import torchaudio
from pesq import pesq
from pystoi import stoi

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party"))
from vidtok.modules.util import compute_psnr, compute_ssim


def si_sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-invariant SDR (dB) between 1-D signals."""
    ref, est = ref - ref.mean(), est - est.mean()
    proj = (est @ ref) / (ref @ ref + eps) * ref
    noise = est - proj
    return float(10 * torch.log10((proj @ proj + eps) / (noise @ noise + eps)))


def audio_metrics(ref: torch.Tensor, est: torch.Tensor, sr: int) -> dict:
    """PESQ-wb, STOI and SI-SDR between mono ref/est sampled at sr."""
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n].float(), est[..., :n].float()
    r16 = torchaudio.functional.resample(ref, sr, 16000).squeeze()
    e16 = torchaudio.functional.resample(est, sr, 16000).squeeze()
    return {
        "pesq": float(pesq(16000, r16.numpy(), e16.numpy(), "wb")),
        "stoi": float(stoi(r16.numpy(), e16.numpy(), 16000, extended=False)),
        "si_sdr": si_sdr(ref.squeeze(), est.squeeze()),
    }


def video_metrics(ref: torch.Tensor, rec: torch.Tensor) -> tuple[float, float]:
    """PSNR/SSIM between [B, 3, T, H, W] clips in [-1, 1] (VidTok frame metrics, data range 1)."""
    ref = ((ref + 1) / 2).clamp(0, 1).float()
    rec = ((rec + 1) / 2).clamp(0, 1).float()
    return float(compute_psnr(ref, rec)), float(compute_ssim(ref, rec))

