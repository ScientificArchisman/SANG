"""Pixel-space perceptual losses: SyncNet lip-sync + a face/lip loss weight."""
from pathlib import Path

import torch
from torch import nn


def face_lip_weight(shape: tuple[int, int, int, int, int], device, lam_face: float = 1.0,
                    lam_lip: float = 2.0) -> torch.Tensor:
    """W = 1 + M_face + 2*M_lip as a fixed anatomical prior. shape [B,3,T,H,W] -> [1,1,1,H,W].

    Geometry assumes face_crop: video.face_box centres the face and scales it to 1/margin of the
    frame, so a fixed ellipse is a good prior and no landmark lookup is needed in the hot loop.
    Under the old blind centre crop the premise was false -- the mouth landed inside the lip
    ellipse in 2 of 12 sampled clips."""
    H, W = shape[-2], shape[-1]
    ys = torch.arange(H, device=device).float() / H
    xs = torch.arange(W, device=device).float() / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # With face_box(margin=1.6) the face spans ~1/1.6 of the frame, centred.
    face = (((xx - 0.5) / 0.34) ** 2 + ((yy - 0.50) / 0.40) ** 2 <= 1.0).float()
    # Mouth sits ~3/4 down the face box -> y ~= 0.66 in the cropped frame.
    lip = (((xx - 0.5) / 0.14) ** 2 + ((yy - 0.66) / 0.08) ** 2 <= 1.0).float()
    w = 1.0 + lam_face * face + lam_lip * lip
    return w.reshape(1, 1, 1, H, W)

REPO = Path(__file__).resolve().parents[1]
SYNCNET = REPO / "third_party" / "syncnet" / "stable_syncnet.pt"


class SyncNetLoss(nn.Module):
    """LatentSync-style lip-sync loss on decoded mouth crops (uses stable_syncnet.pt)."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if not SYNCNET.exists():
            raise FileNotFoundError(f"missing {SYNCNET}; run scripts/fetch_loss_weights.py")
        from sang.syncnet import StableSyncNet  # local wrapper, lazily imported
        self.syncnet = StableSyncNet.from_checkpoint(SYNCNET, device=device).eval()
        for p in self.syncnet.parameters():
            p.requires_grad = False

    def forward(self, frames: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """frames [B,3,T,H,W] in [-1,1], audio mels [B,T,n_mel] -> scalar."""
        return self.syncnet.loss(frames, audio)


def build_losses(cfg: dict, device: str) -> dict[str, nn.Module]:
    """Instantiate optional perceptual losses from config flags."""
    out = {}
    if cfg.get("syncnet_loss", False):
        out["syncnet"] = SyncNetLoss(device)
    return out
