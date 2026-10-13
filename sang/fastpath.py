"""Phase C speed track: GPU paste-back, a background video writer and stage timers (report 'SANG latency
reduction' 1.4 and 2.1). The renderer's own fast path is MotionCodec.render_iter (sang/motion.py).

GpuPaster replaces sang.motion.paste_back (CPU cv2.warpAffine over the full frame, then a float blend of
the full frame) with grid_sample inside the bounding box of the crop's footprint; outside it the output
IS the background, as before. Sampling convention: cv2.warpAffine maps output pixel (x, y) to crop
pixel M^-1 (x, y) with pixel centres at integers, which is grid_sample's align_corners=True after
scaling by 2 / (S - 1). cv2 interpolates in fixed point (1/32 px) and rounds the warped crop to uint8;
the warped crop is rounded here too, so outputs differ from the CPU path by at most a grey level or so
(tests/test_fastpath.py).
"""
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sang.motion import LP, as3x3


def crop_mask(S: int) -> np.ndarray:
    """LivePortrait's feathered crop mask at S x S (as sang.motion.paste_back), else a 16-px linear feather."""
    import cv2
    tpl = LP / "src" / "utils" / "resources" / "mask_template.png"
    if tpl.exists():
        return cv2.resize(cv2.imread(str(tpl), cv2.IMREAD_GRAYSCALE), (S, S)).astype(np.float32) / 255.0
    r = np.minimum(np.arange(S), np.arange(S)[::-1]).astype(np.float32)
    return np.clip(np.minimum(r[:, None], r[None, :]) / 16.0, 0, 1)


class GpuPaster:
    """Paste rendered crops [b, S, S, 3] (uint8, GPU) into `background` with M_c2o (crop -> original)."""

    def __init__(self, M_c2o: np.ndarray, background: np.ndarray, S: int, device: str = "cuda",
                 mask: np.ndarray | None = None):
        H, W = background.shape[:2]
        M = as3x3(M_c2o)
        corners = M @ np.array([[0, 0, 1], [S - 1, 0, 1], [0, S - 1, 1], [S - 1, S - 1, 1]], np.float64).T
        x0, y0 = np.floor(corners[:2].min(1)).astype(int) - 1
        x1, y1 = np.ceil(corners[:2].max(1)).astype(int) + 2
        self.x0, self.y0 = max(0, x0), max(0, y0)
        self.x1, self.y1 = min(W, x1), min(H, y1)
        self.bg_full = background
        self.empty = self.x1 <= self.x0 or self.y1 <= self.y0
        if self.empty:
            return
        ys, xs = np.mgrid[self.y0:self.y1, self.x0:self.x1].astype(np.float64)
        Minv = np.linalg.inv(M)
        sx = Minv[0, 0] * xs + Minv[0, 1] * ys + Minv[0, 2]
        sy = Minv[1, 0] * xs + Minv[1, 1] * ys + Minv[1, 2]
        g = np.stack([sx / (S - 1) * 2 - 1, sy / (S - 1) * 2 - 1], -1).astype(np.float32)
        self.grid = torch.from_numpy(g)[None].to(device)                          # [1, h, w, 2]
        m = torch.from_numpy(crop_mask(S) if mask is None else mask.astype(np.float32))[None, None].to(device)
        self.mask = F.grid_sample(m, self.grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        self.bg = torch.from_numpy(np.ascontiguousarray(background[self.y0:self.y1, self.x0:self.x1])).to(device)
        self.bg = self.bg.permute(2, 0, 1)[None].float()                          # [1, 3, h, w]
        self.device = device

    @torch.no_grad()
    def __call__(self, crops: torch.Tensor) -> np.ndarray:
        """[b, S, S, 3] uint8 (any device) -> [b, H, W, 3] uint8 numpy."""
        b = len(crops)
        out = np.repeat(self.bg_full[None], b, 0)
        if self.empty:
            return out
        f = crops.to(self.device).permute(0, 3, 1, 2).float()
        w = F.grid_sample(f, self.grid.expand(b, -1, -1, -1), mode="bilinear", padding_mode="zeros",
                          align_corners=True).round()                             # cv2 rounds the warped crop
        roi = (self.mask * w + (1.0 - self.mask) * self.bg).clamp(0, 255).to(torch.uint8)
        out[:, self.y0:self.y1, self.x0:self.x1] = roi.permute(0, 2, 3, 1).cpu().numpy()
        return out


class AsyncWriter:
    """libx264 writer on a background thread (same settings as sang.bench.write_mp4), so encoding overlaps
    with rendering. write() takes [b, H, W, 3] uint8 batches; close() joins and muxes the audio."""

    def __init__(self, path: Path, fps: float = 25.0, audio: Path | None = None, depth: int = 8):
        import imageio.v2 as imageio
        self.path, self.audio = Path(path), audio
        self.silent = self.path.with_suffix(".silent.mp4") if audio else self.path
        self.w = imageio.get_writer(str(self.silent), fps=fps, codec="libx264",
                                    output_params=["-pix_fmt", "yuv420p", "-crf", "17"])
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.err = None
        self.busy = 0.0                                              # seconds spent encoding, for the timers
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        try:
            while (b := self.q.get()) is not None:
                t = time.perf_counter()
                for f in b:
                    self.w.append_data(np.ascontiguousarray(f))
                self.busy += time.perf_counter() - t
        except Exception as e:                                       # surfaced by close()
            self.err = e
            while self.q.get() is not None:
                pass

    def write(self, frames: np.ndarray) -> None:
        if self.err:
            raise self.err
        self.q.put(frames)

    def close(self) -> Path:
        self.q.put(None)
        self.t.join()
        self.w.close()
        if self.err:
            raise self.err
        if self.audio:
            import imageio_ffmpeg
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(self.silent),
                            "-i", str(self.audio), "-c:v", "copy", "-c:a", "aac", "-shortest", str(self.path)], check=True)
            self.silent.unlink(missing_ok=True)
        return self.path


class Timer:
    """Wall-clock per stage; synchronises CUDA at stage boundaries only (profile mode)."""

    def __init__(self, cuda: bool = True):
        self.cuda = cuda and torch.cuda.is_available()
        self.t: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str):
        if self.cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        self.t[name] = self.t.get(name, 0.0) + time.perf_counter() - t
