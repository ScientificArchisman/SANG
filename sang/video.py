import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from omegaconf import OmegaConf
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party"))
from vidtok.modules.util import instantiate_from_config

VIDTOK_CFG = REPO / "configs" / "vidtok.yaml"
LEVELS = {4096: [8] * 4, 32768: [8] * 5, 262144: [8] * 6}


def load_vidtok(codebook: int = 262144, weights: str | None = None, device: str = "cpu"):
    """Load a frozen VidTok FSQ tokenizer in eval mode; weights default to checkpoints/."""
    levels = LEVELS[codebook]
    cfg = OmegaConf.load(VIDTOK_CFG)
    cfg.model.params.encoder_config.params.z_channels = len(levels)
    cfg.model.params.regularizer_config.params.levels = levels
    ckpt = Path(weights) if weights else REPO / "checkpoints" / f"vidtok_fsq_causal_488_{codebook}.ckpt"
    if ckpt.exists():
        cfg.model.params.ckpt_path = str(ckpt)
    return instantiate_from_config(cfg.model).to(device).eval()


def decode_frames(path: str, frames: int, start: int | None = None, fps: float | None = None):
    """Decode `frames` frames (optionally resampled to `fps`) -> ([T, H, W, 3] uint8, native_fps, start)."""
    vr = VideoReader(path)
    native = vr.get_avg_fps()
    stride = 1.0 if fps is None else native / fps
    span = int(round((frames - 1) * stride)) + 1
    if len(vr) < span:
        raise ValueError(f"{path}: {len(vr)} frames < {span}")
    start = (len(vr) - span) // 2 if start is None else start
    idx = [start + int(round(i * stride)) for i in range(frames)]
    return vr.get_batch(idx).asnumpy(), native, start


def crop_box(h: int, w: int) -> tuple[int, int, int]:
    """Centre square crop matching crop_resize: (top, left, side)."""
    side = min(h, w)
    return (h - side) // 2, (w - side) // 2, side


def upscale_to_original(frames: np.ndarray, orig_h: int, orig_w: int,
                        box: tuple[int, int, int] | None = None) -> np.ndarray:
    """Paste output [T,res,res,3] into [T,orig_h,orig_w,3]. Pass the same `box` crop_resize used,
    or the generation lands in the centre instead of where the face was."""
    top, left, side = box if box is not None else crop_box(orig_h, orig_w)
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()
    patch = F.interpolate(x, size=(side, side), mode="bilinear", align_corners=False).round().byte()
    out = np.zeros((frames.shape[0], orig_h, orig_w, 3), dtype=np.uint8)
    out[:, top : top + side, left : left + side, :] = patch.permute(0, 2, 3, 1).numpy()
    return out


def face_box(frames: np.ndarray, margin: float = 1.6, probe: int = 1) -> tuple[int, int, int] | None:
    """Square face crop (top, left, side) for a whole window, or None if no face is found.

    One box per window (union over `probe` frames) so the crop does not jitter. `margin` scales
    the face bbox and absorbs intra-window motion, so a single mid-window probe is enough for a
    0.64 s window; detection costs ~93 ms, which dominates cache build. The box is shifted, not
    shrunk, at frame edges so face scale stays constant across clips."""
    from sang import face as _face  # local: mediapipe needs libGLESv2, not present everywhere

    T, H, W = frames.shape[:3]
    idx = [T // 2] if probe <= 1 else sorted({0, T // 2, T - 1})
    xs0, ys0, xs1, ys1 = [], [], [], []
    for i in idx[:probe]:
        pts = _face.landmarks_px(np.ascontiguousarray(frames[i]))
        if pts is None:
            continue
        xs0.append(pts[:, 0].min()); xs1.append(pts[:, 0].max())
        ys0.append(pts[:, 1].min()); ys1.append(pts[:, 1].max())
    if not xs0:
        return None
    x0, x1, y0, y1 = min(xs0), max(xs1), min(ys0), max(ys1)
    side = int(round(max(x1 - x0, y1 - y0) * margin))
    side = max(16, min(side, H, W))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left = int(round(min(max(0.0, cx - side / 2.0), W - side)))
    top = int(round(min(max(0.0, cy - side / 2.0), H - side)))
    return top, left, side


def crop_resize(frames, res: int, box: tuple[int, int, int] | None = None) -> torch.Tensor:
    """Square crop then resize [T, H, W, 3] uint8 -> [1, 3, T, res, res] in [-1, 1].

    `box` is (top, left, side) — e.g. from `face_box`. When None, falls back to the centre crop."""
    x = torch.from_numpy(frames).permute(3, 0, 1, 2).float()
    if box is None:
        c = min(x.shape[-2], x.shape[-1])
        top, left = (x.shape[-2] - c) // 2, (x.shape[-1] - c) // 2
    else:
        top, left, c = box
    x = x[..., top : top + c, left : left + c]
    x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return (x / 127.5 - 1.0).unsqueeze(0)


def to_uint8_frames(x: torch.Tensor) -> np.ndarray:
    """Clip tensor [1, 3, T, H, W] in [-1, 1] -> [T, H, W, 3] uint8 RGB (what VidTok actually saw)."""
    return ((x[0].permute(1, 2, 3, 0).clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


def frames_to_input(frames: np.ndarray, res: int | None = None) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [1, 3, T, res, res] in [-1, 1] (resize only if res given and differs)."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float()
    if res is not None and (x.shape[-1] != res or x.shape[-2] != res):
        x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
    return (x / 127.5 - 1.0).unsqueeze(0)


@torch.no_grad()
def encode(model, video: torch.Tensor) -> torch.Tensor:
    """Video [1, 3, T, H, W] in [-1, 1] -> FSQ index grid [1, T_lat, H_lat, W_lat]."""
    return model.encode(video, return_reg_log=True)[1]["indices"]


@torch.no_grad()
def encode_latent(model, video: torch.Tensor) -> torch.Tensor:
    """Video [1, 3, T, H, W] in [-1, 1] -> continuous latent [1, z_ch, T_lat, H_lat, W_lat]."""
    return model.encode_video(video)


@torch.no_grad()
def reconstruct(model, video: torch.Tensor, return_indices: bool = False):
    """Encode then decode a [B, 3, T, H, W] clip in [-1, 1]; causal T should be 4k+1."""
    idx = encode(model, video)
    rec = model.decode(idx, decode_from_indices=True)
    return (rec, idx) if return_indices else rec
