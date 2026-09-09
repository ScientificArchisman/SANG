"""Pixel-space perceptual losses for SANG v3 (SyncNet lip-sync + TREPA temporal)."""
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def face_lip_weight(shape: tuple[int, int, int, int, int], device, lam_face: float = 1.0,
                    lam_lip: float = 2.0) -> torch.Tensor:
    """LeapTalk W = 1 + M_face + M_lip as a fixed anatomical prior (frames are centre-cropped
    talking heads, so the face is centred and the mouth sits lower-centre). No MediaPipe in the
    hot loop. shape [B,3,T,H,W] -> weight [1,1,1,H,W] broadcast over B,C,T."""
    H, W = shape[-2], shape[-1]
    ys = torch.arange(H, device=device).float() / H
    xs = torch.arange(W, device=device).float() / W
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    # face: centred ellipse covering most of the frame
    face = (((xx - 0.5) / 0.42) ** 2 + ((yy - 0.52) / 0.46) ** 2 <= 1.0).float()
    # lip: smaller ellipse, lower-centre
    lip = (((xx - 0.5) / 0.20) ** 2 + ((yy - 0.68) / 0.13) ** 2 <= 1.0).float()
    w = 1.0 + lam_face * face + lam_lip * lip
    return w.reshape(1, 1, 1, H, W)

REPO = Path(__file__).resolve().parents[1]
SYNCNET = REPO / "third_party" / "syncnet" / "stable_syncnet.pt"
VIDEOMAE = REPO / "third_party" / "videomaev2"


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


class TREPALoss(nn.Module):
    """Temporal Representation Alignment via VideoMAE-v2 (LatentSync TREPA)."""

    IMG_SIZE = 224
    NFRAMES = 16 

    def __init__(self, device: str = "cuda"):
        super().__init__()
        if not (VIDEOMAE / "model.safetensors").exists():
            raise FileNotFoundError(f"missing {VIDEOMAE}/model.safetensors; run scripts/fetch_loss_weights.py")
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(str(VIDEOMAE), trust_remote_code=True).to(device).eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """[B,3,T,H,W] in [-1,1] -> [B,3,16,224,224] in [0,1] for VideoMAE."""
        B, C, T, H, W = x.shape
        if (T, H, W) != (self.NFRAMES, self.IMG_SIZE, self.IMG_SIZE):
            x = F.interpolate(x, size=(self.NFRAMES, self.IMG_SIZE, self.IMG_SIZE),
                              mode="trilinear", align_corners=False)
        return (x / 2 + 0.5).clamp(0, 1)

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.is_cuda):
            if hasattr(self.encoder, "extract_features"):
                return self.encoder.extract_features(x)
            return self.encoder.model.forward_features(x)

    def _features(self, x: torch.Tensor, *, checkpoint: bool) -> torch.Tensor:
        x = self._prep(x)
        if checkpoint:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._run_encoder, x, use_reentrant=False)
        with torch.no_grad():
            return self._run_encoder(x)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """pred/gt [B,3,T,H,W] in [-1,1] -> scalar."""
        f_gt = self._features(gt, checkpoint=False)
        f_pred = self._features(pred, checkpoint=True)
        return F.mse_loss(F.normalize(f_pred.float(), dim=-1), F.normalize(f_gt.float(), dim=-1))


def build_losses(cfg: dict, device: str) -> dict[str, nn.Module]:
    """Instantiate optional perceptual losses from config flags."""
    out = {}
    if cfg.get("syncnet_loss", False):
        out["syncnet"] = SyncNetLoss(device)
    if cfg.get("trepa_loss", False):
        out["trepa"] = TREPALoss(device)
    return out
