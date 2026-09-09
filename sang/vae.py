"""Frozen Wan2.1 3D KL-VAE (LeapTalk / EARTalking tokenizer). Continuous latents, no FSQ.

diffusers is vendored at third_party/pydeps (conda envs are root-owned). Latents are stored
mean/std-normalized so the diffusion head sees ~unit-scale channels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
_PYDEPS = REPO / "third_party" / "pydeps"
if str(_PYDEPS) not in sys.path:
    sys.path.append(str(_PYDEPS))  # after conda site-packages so env Pillow/numpy win

WAN_DIR = REPO / "checkpoints" / "wan_vae"


class WanVAE(nn.Module):
    """encode_video / decode_video: pixels [-1,1] <-> normalized latents [B, 16, Tv, H/8, W/8]."""

    z_ch = 16
    spatial = 8
    temporal = 4

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.vae = inner
        mean = torch.tensor(inner.config.latents_mean, dtype=torch.float32).view(1, -1, 1, 1, 1)
        std = torch.tensor(inner.config.latents_std, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)

    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        z = self.vae.encode(video).latent_dist.mode()
        return (z - self.latents_mean) / self.latents_std

    @torch.no_grad()
    def decode_video(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.latents_std + self.latents_mean
        return self.vae.decode(z).sample


def load_wan_vae(weights: str | None = None, device: str = "cpu") -> WanVAE:
    from diffusers import AutoencoderKLWan

    path = Path(weights) if weights else WAN_DIR
    if not (path / "config.json").exists():
        raise FileNotFoundError(
            f"Wan VAE not at {path}. Download with: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('Wan-AI/Wan2.1-T2V-1.3B-Diffusers', allow_patterns=['vae/*'], "
            f"local_dir='{WAN_DIR}')\""
        )
    inner = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=torch.float32)
    return WanVAE(inner).to(device).eval()
