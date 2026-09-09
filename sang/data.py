from pathlib import Path

import torch
from decord import AudioReader, VideoReader

from sang.video import (crop_resize, decode_frames, encode, encode_latent, face_box, frames_to_input,
                        to_uint8_frames)


def audio_samples(frames: int, fps: float, sr: int) -> int:
    """Samples spanned by `frames` frames at `fps`. N frames span N-1 intervals, so 17 @ 25 fps
    is 0.64 s; using frames/fps stretches the audio 6% against the video."""
    return round((frames - 1) / fps * sr)


@torch.no_grad()
def clip_tokens(path, vidtok, audio_enc, frames: int = 17, res: int = 128, start=None,
                fps: float = 25, face_cond: bool = False, face_crop: bool = False) -> dict:
    """One TalkVid clip -> aligned dict(video idx [1,Tv,h,w], audio codes [1,K,Ta], ref [1,3,res,res], native fps).

    Frames are resampled to `fps` and the audio window is a fixed `frames/fps` seconds, so Ta is
    constant across clips (clean batching). With `face_cond`, also returns a VidTok-encoded face-mesh
    structure grid `struct [1,Tv,h,w]` on the exact same crop."""
    dev = next(vidtok.parameters()).device
    fr, native, start = decode_frames(str(path), frames, start, fps=fps)
    video = crop_resize(fr, res, box=face_box(fr) if face_crop else None).to(dev)

    sr = audio_enc.sample_rate
    n = audio_samples(frames, fps, sr)
    a0 = int(start / native * sr)
    wav = torch.from_numpy(AudioReader(str(Path(path).with_suffix(".m4a")), sample_rate=sr, mono=True)[:].asnumpy())
    win = wav[:, a0:a0 + n]
    if win.shape[-1] < n:
        win = torch.nn.functional.pad(win, (0, n - win.shape[-1]))
    audio = audio_enc.encode(win[None].to(dev))

    out = {"video": encode(vidtok, video), "audio": audio, "ref": video[:, :, 0], "fps": native, "pixels": video}
    if face_cond:
        out["struct"] = struct_grid(vidtok, video)
    return out


@torch.no_grad()
def struct_grid(vidtok, video: torch.Tensor) -> torch.Tensor:
    """Clip [1,3,T,res,res] in [-1,1] -> VidTok-encoded face-mesh structure grid [1,Tv,h,w] on the same crop."""
    from sang import face  

    if not hasattr(vidtok, "regularization"):
        raise RuntimeError("struct conditioning encodes the mesh with VidTok; not available on the Wan VAE track")
    dev = next(vidtok.parameters()).device
    struct_img = face.render_structure(to_uint8_frames(video))
    return encode(vidtok, frames_to_input(struct_img).to(dev))


@torch.no_grad()
def video_struct_windows(path, vidtok, frames: int = 17, res: int = 128, fps: float = 25,
                         max_windows: int = 64) -> list[torch.Tensor]:
    """Driving video -> per-window face-mesh structure grids [1,Tv,h,w] (identity-free; for reenactment)."""
    dev = next(vidtok.parameters()).device
    vr = VideoReader(str(path))
    native = vr.get_avg_fps()
    stride = native / fps
    span = int(round((frames - 1) * stride)) + 1
    nwin = min(max_windows, len(vr) // span)
    if nwin < 1:
        raise ValueError(f"{path}: {len(vr)} frames < {span}")
    starts = [w * span for w in range(nwin)]
    arr = vr.get_batch([s + int(round(i * stride)) for s in starts for i in range(frames)]).asnumpy()
    fr = arr.reshape(nwin, frames, *arr.shape[1:])
    return [struct_grid(vidtok, crop_resize(fr[w], res).to(dev)) for w in range(nwin)]


@torch.no_grad()
def clip_windows(path, vidtok, audio_enc, frames: int = 17, res: int = 128, fps: float = 25,
                 max_windows: int = 8, face_cond: bool = False,
                 continuous: bool = False, face_crop: bool = False) -> list[dict]:
    """One clip -> up to `max_windows` non-overlapping windows.

    Each window is time-aligned like clip_tokens: dict(video, audio codes [1,K,Ta]); with `face_cond`
    each window also carries a face-mesh structure grid `struct [1,Tv,h,w]`.
    `video` is FSQ index grid [1,Tv,h,w] by default, or continuous latent [1,z_ch,Tv,h,w] when
    `continuous`. Frames are decoded per-window so native-res numpy does not accumulate."""
    dev = next(vidtok.parameters()).device
    vr = VideoReader(str(path))
    native = vr.get_avg_fps()
    stride = native / fps
    span = int(round((frames - 1) * stride)) + 1
    nwin = min(max_windows, len(vr) // span)
    if nwin < 1:
        raise ValueError(f"{path}: {len(vr)} frames < {span}")
    # Spread windows over the whole clip. Packing them from t=0 used only the first
    # max_windows*span frames, so a 130 s clip contributed as much as a 6 s one.
    last = len(vr) - span
    starts = [round(w * last / (nwin - 1)) for w in range(nwin)] if nwin > 1 else [0]

    sr = audio_enc.sample_rate
    n = audio_samples(frames, fps, sr)
    wav = torch.from_numpy(AudioReader(str(Path(path).with_suffix(".m4a")), sample_rate=sr, mono=True)[:].asnumpy())
    enc = encode_latent if continuous else encode
    out = []
    for w in range(nwin):
        idxs = [starts[w] + int(round(i * stride)) for i in range(frames)]
        fr = vr.get_batch(idxs).asnumpy()
        # Falls back to the centre crop when no face is found and flags it for filtering;
        # dropping here would leave holes in the window indices build_cache relies on.
        box = face_box(fr) if face_crop else None
        found = box is not None
        video = crop_resize(fr, res, box=box).to(dev)
        del fr
        a0 = int(starts[w] / native * sr)
        win = wav[:, a0:a0 + n]
        if win.shape[-1] < n:
            win = torch.nn.functional.pad(win, (0, n - win.shape[-1]))
        lat = enc(vidtok, video)
        aud = audio_enc.encode(win[None].to(dev))
        d = {"video": lat.cpu(), "audio": aud.cpu()}
        if face_crop:
            d["face_found"] = found
        if face_cond:
            d["struct"] = struct_grid(vidtok, video).cpu()
        del video, lat, aud
        out.append(d)
    del vr, wav
    return out
