"""Clip -> aligned (video latent, audio features[, mel]) windows.

Caching is split into a CPU side (decode, face crop, audio slice, mel) that is safe to run in a
recycled worker process, and a GPU side that batches a clip's windows into one VAE and one WavLM
call. decord's VideoReader leaks ~14 MB per clip and mediapipe ~1.4 MB per detection, which grew
job 162800 to 118 GB RSS during cache build; workers with a bounded task count make that moot.
"""
from pathlib import Path

import numpy as np
import torch
from decord import AudioReader, VideoReader

from sang.video import (crop_resize, decode_frames, encode, encode_latent, face_box, frames_to_input,
                        open_video, to_uint8_frames)

SR = 16000
MEL_VERSION = 3  # 1: mel cut from packed-from-t=0 starts (misaligned once windows were spread)
                 # 2: log-power torchaudio mel on the WavLM window
                 # 3: Wav2Lip mel, cut SYNC_MEL_OFFSET after the WavLM window (see both below)
# stable_syncnet.pt calls a window "in sync" when its mel starts 60 ms AFTER the video's container
# time: on 28 ground-truth windows the loss is 0.66 at offset 0 and 0.45 at +60 ms, a smooth V on
# either side (probe, 2026-09-10). decord agrees with ffmpeg to 1 ms on both streams, so the offset
# is the checkpoint's convention, not ours; only the SyncNet mel moves, WavLM stays on container
# time. Left at 0 the loss sat on its ground-truth floor (0.60) and pulled the lips 60 ms early.
SYNC_MEL_OFFSET = int(0.060 * SR)  # samples


def audio_samples(frames: int, fps: float, sr: int) -> int:
    """Samples spanned by `frames` frames at `fps`. N frames span N-1 intervals, so 17 @ 25 fps
    is 0.64 s; using frames/fps stretches the audio 6% against the video."""
    return round((frames - 1) / fps * sr)


def window_starts(n_frames: int, span: int, nwin: int) -> list[int]:
    """Window start frames spread over the whole clip. Packing them from t=0 used only the first
    nwin*span frames, so a 130 s clip contributed as much as a 6 s one."""
    last = n_frames - span
    return [round(w * last / (nwin - 1)) for w in range(nwin)] if nwin > 1 else [0]


_MEL_FB: dict[tuple, torch.Tensor] = {}


def mel_spectrogram(wav: torch.Tensor, sr: int = SR, n_mels: int = 80) -> torch.Tensor:
    """[1, N] waveform -> [1, n_mels, T] mel in the Wav2Lip/LatentSync format stable_syncnet.pt was
    trained on: preemphasis 0.97, n_fft 800, hop 200 (12.5 ms), 55-7600 Hz slaney filterbank,
    20*log10 magnitude minus a 20 dB reference, normalised to [-4, 4] (Wav2Lip audio.py with its
    hparams). 0.64 s -> 52 steps, the width the audio encoder reduces to 1. The previous natural-log
    power mel spanned [-10, 9] and cost ~0.06 of in/out-of-sync separation on ground truth."""
    import torchaudio
    n_fft = 800
    key = (sr, n_mels, n_fft)
    if key not in _MEL_FB:
        _MEL_FB[key] = torchaudio.functional.melscale_fbanks(
            n_fft // 2 + 1, 55.0, 7600.0, n_mels, sr, norm="slaney", mel_scale="slaney").T
    wav = wav.float()
    y = torch.cat([wav[:, :1], wav[:, 1:] - 0.97 * wav[:, :-1]], dim=1)
    spec = torch.stft(y, n_fft, hop_length=200, win_length=n_fft, window=torch.hann_window(n_fft),
                      center=True, pad_mode="constant", return_complex=True).abs()   # [1, F, T]
    db = 20.0 * torch.log10(torch.matmul(_MEL_FB[key], spec).clamp_min(1e-5)) - 20.0
    return ((db + 100.0) / 100.0 * 8.0 - 4.0).clamp(-4.0, 4.0)


def _clip_geometry(vr, frames: int, fps: float, max_windows: int):
    native = vr.get_avg_fps()
    stride = native / fps
    span = int(round((frames - 1) * stride)) + 1
    nwin = min(max_windows, len(vr) // span)
    if nwin < 1:
        raise ValueError(f"{len(vr)} frames < {span}")
    return native, stride, span, window_starts(len(vr), span, nwin)


def _audio_window(wav: torch.Tensor, start_frame: int, native: float, n: int, offset: int = 0) -> torch.Tensor:
    """`n` samples from the container time of `start_frame`, shifted `offset` samples (zero-padded)."""
    a0 = int(start_frame / native * SR) + offset
    win = wav[:, a0:a0 + n]
    if win.shape[-1] < n:
        win = torch.nn.functional.pad(win, (0, n - win.shape[-1]))
    return win


def clip_cpu_windows(path, frames: int = 17, res: int = 128, fps: float = 25, max_windows: int = 8,
                     face_crop: bool = False, with_mel: bool = False) -> list[dict]:
    """CPU side of caching for one clip. Each window: pixels [T,res,res,3] uint8, wav [1,n] float32,
    start frame, face_found (if face_crop), mel [80,T] fp16 (if with_mel). The mel is cut from the
    same waveform as WavLM, from the same start plus SYNC_MEL_OFFSET (SyncNet's sync convention)."""
    vr = open_video(path)
    native, stride, span, starts = _clip_geometry(vr, frames, fps, max_windows)
    n = audio_samples(frames, fps, SR)
    wav = torch.from_numpy(AudioReader(str(Path(path).with_suffix(".m4a")), sample_rate=SR, mono=True)[:].asnumpy())
    out = []
    for s in starts:
        fr = vr.get_batch([s + int(round(i * stride)) for i in range(frames)]).asnumpy()
        box = face_box(fr) if face_crop else None
        win = _audio_window(wav, s, native, n)
        d = {"pixels": to_uint8_frames(crop_resize(fr, res, box=box)), "wav": win.numpy(), "start": s}
        if face_crop:
            d["face_found"] = box is not None
        if with_mel:
            d["mel"] = mel_spectrogram(_audio_window(wav, s, native, n, SYNC_MEL_OFFSET))[0].half()
        out.append(d)
    return out


def clip_mels(path, frames: int = 17, fps: float = 25, max_windows: int = 8) -> list[torch.Tensor]:
    """Mels only, for repairing entries cached before MEL_VERSION (same starts as clip_cpu_windows)."""
    vr = open_video(path)
    native, _, _, starts = _clip_geometry(vr, frames, fps, max_windows)
    n = audio_samples(frames, fps, SR)
    wav = torch.from_numpy(AudioReader(str(Path(path).with_suffix(".m4a")), sample_rate=SR, mono=True)[:].asnumpy())
    return [mel_spectrogram(_audio_window(wav, s, native, n, SYNC_MEL_OFFSET))[0].half() for s in starts]


def cpu_task(job) -> tuple:
    """Pool entry point: ("windows", path, kw) | ("mels", path, kw) -> (path, result | None, err)."""
    kind, path, kw = job
    try:
        fn = clip_cpu_windows if kind == "windows" else clip_mels
        return path, fn(path, **kw), None
    except Exception as e:  # a bad clip must not take the pool down
        return path, None, f"{type(e).__name__}: {e}"


def pooled(tasks, fn, workers: int, tasks_per_worker: int = 100, timeout: float = 900.0,
           chunk: int | None = None):
    """Map `fn` over `tasks` in spawn workers, yielding (index, value, err) in submission order.

    Replaces multiprocessing.Pool(maxtasksperchild=...), which cannot survive a worker
    casualty: its _handle_results thread returns silently on a truncated result frame
    (`except (OSError, EOFError): return`, no traceback), after which the parent blocks
    forever in IMapIterator.next() while _handle_workers spins at 100% CPU. Job 162848 sat
    like that for 7h45m after a single recycle, with the GPU idle and nothing in the log.
    ProcessPoolExecutor reports a dead worker as BrokenProcessPool instead, and every wait
    here is bounded by `timeout`, so a casualty costs seconds rather than the allocation.

    `err` is a message (and `value` None) for a task that raised, crashed its worker, or
    overran `timeout` -- the caller skips that clip and keeps going.

    Worker lifetime stays bounded -- decord and mediapipe leak ~26 MB per clip between them --
    by retiring the whole executor every workers*tasks_per_worker tasks; ProcessPoolExecutor
    only grew max_tasks_per_child in 3.11 and this runs on 3.10. At most `chunk` futures are
    outstanding, so workers cannot run ahead of the GPU consumer and park ~27 MB each.
    """
    import collections
    import concurrent.futures as cf
    import multiprocessing as mp
    from concurrent.futures.process import BrokenProcessPool

    workers = max(1, int(workers))
    chunk = chunk or 4 * workers
    generation = max(1, int(tasks_per_worker)) * workers
    ctx = mp.get_context("spawn")
    i, n = 0, len(tasks)
    while i < n:
        stop = min(n, i + generation)
        ex = cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
        broke = False
        try:
            pending, nxt = collections.deque(), i
            while i < stop and not broke:
                while nxt < stop and len(pending) < chunk:
                    pending.append((nxt, ex.submit(fn, tasks[nxt])))
                    nxt += 1
                idx, fut = pending.popleft()
                try:
                    value, err = fut.result(timeout=timeout), None
                except (BrokenProcessPool, cf.TimeoutError) as e:
                    # Either a worker died or this task is wedged; the executor is no longer
                    # trustworthy. Retire it and resubmit the rest of the generation below.
                    value, err, broke = None, f"{type(e).__name__}: {e}", True
                except Exception as e:  # the task itself raised -- the pool is still good
                    value, err = None, f"{type(e).__name__}: {e}"
                yield idx, value, err
                i = idx + 1
        finally:
            if broke:  # a wedged worker would outlive the executor still holding its RSS
                for proc in list((getattr(ex, "_processes", None) or {}).values()):
                    proc.kill()
            # At a clean boundary the workers are idle, so waiting is immediate and keeps the
            # retiring generation from overlapping the next one in memory. Never wait on a
            # broken pool: that is the hang this function exists to avoid.
            ex.shutdown(wait=not broke, cancel_futures=True)


@torch.no_grad()
def encode_windows(vidtok, audio_enc, wins: list[dict], continuous: bool = False,
                   face_cond: bool = False) -> list[dict]:
    """GPU side: one VAE call and one WavLM call for all of a clip's windows."""
    dev = next(vidtok.parameters()).device
    vid = torch.cat([frames_to_input(w["pixels"]) for w in wins], 0).to(dev)     # [W,3,T,res,res]
    lat = (encode_latent if continuous else encode)(vidtok, vid)
    aud = audio_enc.encode(torch.from_numpy(np.stack([w["wav"] for w in wins])).to(dev))  # [W,Ta,D]
    out = []
    for i, w in enumerate(wins):
        d = {"video": lat[i:i + 1].cpu(), "audio": aud[i:i + 1].cpu(), "start": w["start"]}
        for k in ("face_found", "mel"):
            if k in w:
                d[k] = w[k]
        if face_cond:
            d["struct"] = struct_grid(vidtok, vid[i:i + 1]).cpu()
        out.append(d)
    return out


@torch.no_grad()
def clip_windows(path, vidtok, audio_enc, frames: int = 17, res: int = 128, fps: float = 25,
                 max_windows: int = 8, face_cond: bool = False, continuous: bool = False,
                 face_crop: bool = False, with_mel: bool = False) -> list[dict]:
    """Single-process convenience: CPU side then GPU side for one clip."""
    wins = clip_cpu_windows(path, frames, res, fps, max_windows, face_crop, with_mel)
    return encode_windows(vidtok, audio_enc, wins, continuous, face_cond)


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
    vr = open_video(path)
    _, stride, _, starts = _clip_geometry(vr, frames, fps, max_windows)
    arr = vr.get_batch([s + int(round(i * stride)) for s in starts for i in range(frames)]).asnumpy()
    fr = arr.reshape(nwin, frames, *arr.shape[1:])
    return [struct_grid(vidtok, crop_resize(fr[w], res).to(dev)) for w in range(nwin)]


