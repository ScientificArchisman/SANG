"""Reconstruction metrics."""
from vidtok.modules.util import compute_psnr, compute_ssim
import torch


def video_metrics(ref: torch.Tensor, rec: torch.Tensor) -> tuple[float, float]:
    """PSNR/SSIM between [B, 3, T, H, W] clips in [-1, 1] (VidTok frame metrics, data range 1)."""
    ref = ((ref + 1) / 2).clamp(0, 1).float()
    rec = ((rec + 1) / 2).clamp(0, 1).float()
    return float(compute_psnr(ref, rec)), float(compute_ssim(ref, rec))

