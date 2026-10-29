"""Video decoding shared by the motion codec (sang.render.liveportrait.decode_clip) and the clip filter."""
from decord import VideoReader

MAX_DECODE_HEIGHT = 1080


def open_video(path, max_height: int = MAX_DECODE_HEIGHT) -> VideoReader:
    """VideoReader that decodes at most `max_height` tall. 1440p/4K sources otherwise decode at
    native size (a 4K 17-frame batch is 420 MB and decord buffers whole GOPs: 4.6 GB transient
    peaks measured); the output crop is 256 px, so a face still has >=256 px at 1080p."""
    vr = VideoReader(str(path))
    h, w = vr[0].shape[:2]
    if h <= max_height:
        return vr
    scale = max_height / h
    del vr
    return VideoReader(str(path), width=int(round(w * scale / 2)) * 2, height=max_height)


def decode_frames(path: str, frames: int, start: int | None = None, fps: float | None = None):
    """Decode `frames` frames (optionally resampled to `fps`) -> ([T, H, W, 3] uint8, native_fps, start)."""
    vr = open_video(path)
    native = vr.get_avg_fps()
    stride = 1.0 if fps is None else native / fps
    span = int(round((frames - 1) * stride)) + 1
    if len(vr) < span:
        raise ValueError(f"{path}: {len(vr)} frames < {span}")
    start = (len(vr) - span) // 2 if start is None else start
    idx = [start + int(round(i * stride)) for i in range(frames)]
    return vr.get_batch(idx).asnumpy(), native, start

