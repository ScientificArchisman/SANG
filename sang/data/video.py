"""Video decoding shared by the motion codec (sang.render.liveportrait.decode_clip) and the clip filter, and mp4
writing."""
import subprocess
from pathlib import Path

import numpy as np

MAX_DECODE_HEIGHT = 1080


def open_video(path, max_height: int = MAX_DECODE_HEIGHT):
    """VideoReader that decodes at most `max_height` tall. 1440p/4K sources otherwise decode at
    native size (a 4K 17-frame batch is 420 MB and decord buffers whole GOPs: 4.6 GB transient
    peaks measured); the output crop is 256 px, so a face still has >=256 px at 1080p."""
    from decord import VideoReader
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



def write_mp4(frames_rgb: np.ndarray, path: Path, fps: float = 25.0, audio: Path | None = None) -> Path:
    """[T,H,W,3] uint8 -> an H.264 mp4 (yuv420p), with the audio file muxed in when given."""
    import imageio.v2 as imageio

    path = Path(path)
    silent = path.with_suffix(".silent.mp4") if audio else path
    with imageio.get_writer(str(silent), fps=fps, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p", "-crf", "17"]) as w:
        for f in frames_rgb:
            w.append_data(np.ascontiguousarray(f))
    if audio:
        import imageio_ffmpeg
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", str(silent), "-i", str(audio),
                        "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)], check=True)
        silent.unlink(missing_ok=True)
    return path
