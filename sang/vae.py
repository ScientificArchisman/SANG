"""Frozen Wan2.1 3D KL-VAE. Continuous latents, mean/std-normalised to ~unit scale.

diffusers is vendored at third_party/pydeps (the conda envs are root-owned).
"""
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
    """pixels [B,3,T,H,W] in [-1,1] <-> normalised latents [B,16,Tv,H/8,W/8]."""

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
        return self._decode(z)

    def decode_video_grad(self, z: torch.Tensor, checkpoint: bool = True) -> torch.Tensor:
        """Differentiable decode for pixel-space losses on a predicted latent. The VAE is frozen,
        so gradient only flows to z. Activation checkpointing recomputes the decoder in backward
        instead of storing it -- the decoder at 256 px is the memory peak of the step."""
        if checkpoint:
            from torch.utils.checkpoint import checkpoint as ckpt
            return ckpt(self._decode, z, use_reentrant=False)
        return self._decode(z)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z * self.latents_std + self.latents_mean).sample


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
